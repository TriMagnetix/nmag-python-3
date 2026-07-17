from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

import nmag
import nmesh
from nmag import checkpoint as checkpoint_module


def _write_mesh(path: Path, *, shifted: bool = False) -> None:
    offset = 0.25 if shifted else 0.0
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [offset, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset, 1.0, 0.0],
            [offset, 0.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[1],
    )
    mesh.save(str(path))


def _simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    damping: float = 0.1,
    shifted_mesh: bool = False,
    current_density: bool = True,
    config: nmag.NmagConfig | None = None,
) -> nmag.Simulation:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / f"{name}.nmesh.h5"
    _write_mesh(mesh_path, shifted=shifted_mesh)
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(8.0e5, "A/m"),
        exchange_coupling=nmag.SI(0.0, "J/m"),
        llg_damping=damping,
        llg_polarisation=0.7,
        llg_xi=0.02,
        do_precession=False,
    )
    simulation = nmag.Simulation(name=name, do_demag=False, config=config)
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m(lambda point: [1.0, 0.1 * point[0] / 1.0e-9, 0.05])
    simulation.set_pinning([1.0, 0.0, 0.5, 1.0])
    simulation.set_H_ext([1.0e4, 2.0e4, 0.0], nmag.SI("A/m"))
    if current_density:
        simulation.set_current_density([1.0e12, 0.0, 0.0], nmag.SI("A/m^2"))
    simulation.set_params(
        stopping_dm_dt=nmag.SI(2.0e8, "1/s"),
        ts_rel_tol=1.0e-8,
        ts_abs_tol=1.0e-8,
        ts_max_step=nmag.SI(1.0e-15, "s"),
    )
    return simulation


def _state_snapshot(simulation: nmag.Simulation) -> dict[str, object]:
    return {
        "m": np.array(simulation._fields["m"], copy=True),
        "pin": np.array(simulation._fields["pin"], copy=True),
        "H_ext": np.array(simulation._fields["H_ext"], copy=True),
        "current_density": (
            None
            if "current_density" not in simulation._fields
            else np.array(simulation._fields["current_density"], copy=True)
        ),
        "clock": simulation.clock.checkpoint_state(),
        "config": simulation.integrator_config,
        "stopping_dm_dt": simulation.stopping_dm_dt,
        "max_time": simulation.max_time_reached,
        "max_dm_dt": simulation.max_dm_dt,
        "convergence": simulation.convergence.checkpoint_state(),
    }


def test_full_restart_round_trip_restores_dynamic_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _simulation(tmp_path, monkeypatch, name="restart-source")
    source.advance_time(nmag.SI(2.0e-15, "s"))
    source.convergence.check(source.step, source.stopping_dm_dt * 2.0, source.stopping_dm_dt)
    source.max_dm_dt = source.stopping_dm_dt * 2.0
    expected = _state_snapshot(source)
    checkpoint = source.save_restart_file(tmp_path / "state.h5")

    with h5py.File(str(checkpoint), "r") as h5:
        dataset = h5["state/m"]
        assert np.asarray(cast(Any, dataset)).dtype == np.dtype(np.float64)
        assert cast(Any, dataset).chunks is None
        assert cast(Any, dataset).compression is None

    target = _simulation(tmp_path, monkeypatch, name="restart-target")
    target.reinitialise()
    assert target._integrator is not None
    target.load_restart_file(checkpoint)

    np.testing.assert_array_equal(target._fields["m"], expected["m"])
    np.testing.assert_array_equal(target._fields["pin"], expected["pin"])
    np.testing.assert_array_equal(target._fields["H_ext"], expected["H_ext"])
    np.testing.assert_array_equal(target._fields["current_density"], expected["current_density"])
    assert target.clock.checkpoint_state() == expected["clock"]
    assert target.integrator_config == expected["config"]
    assert target.stopping_dm_dt == expected["stopping_dm_dt"]
    assert target.max_time_reached == expected["max_time"]
    assert target.max_dm_dt == expected["max_dm_dt"]
    assert target.convergence.checkpoint_state() == expected["convergence"]
    assert target._integrator is None
    assert target._integrator_is_stale is True
    assert target.last_integrator_stats.status == "restarted"


def test_magnetisation_only_transfer_allows_different_material_dynamics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _simulation(tmp_path, monkeypatch, name="transfer-source")
    source.advance_time(nmag.SI(1.0e-15, "s"))
    checkpoint = source.save_restart_file(tmp_path / "transfer.h5")
    expected_m = np.array(source._fields["m"], copy=True)

    target = _simulation(
        tmp_path,
        monkeypatch,
        name="transfer-target",
        damping=0.25,
        current_density=False,
    )
    with pytest.raises(ValueError, match="materials"):
        target.load_restart_file(checkpoint)

    target.load_m_from_h5file(checkpoint)

    np.testing.assert_array_equal(target._fields["m"], expected_m)
    assert "current_density" not in target._fields


def test_full_restart_clears_current_density_absent_from_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _simulation(tmp_path, monkeypatch, name="no-current", current_density=False)
    checkpoint = source.save_restart_file(tmp_path / "no-current.h5")
    target = _simulation(tmp_path, monkeypatch, name="with-current")

    target.load_restart_file(checkpoint)

    assert "current_density" not in target._fields


@pytest.mark.parametrize("kind", ["version", "shape", "nonfinite"])
def test_invalid_checkpoint_does_not_mutate_target_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    source = _simulation(tmp_path, monkeypatch, name=f"invalid-source-{kind}")
    checkpoint = source.save_restart_file(tmp_path / f"invalid-{kind}.h5")
    with h5py.File(str(checkpoint), "r+") as h5:
        if kind == "version":
            h5.attrs["version"] = 99
        elif kind == "shape":
            state = cast(Any, h5["state"])
            del state["pin"]
            state.create_dataset("pin", data=np.ones(2))
        else:
            cast(Any, h5["state/m"])[0, 0] = np.nan

    target = _simulation(tmp_path, monkeypatch, name=f"invalid-target-{kind}")
    before = _state_snapshot(target)
    with pytest.raises(ValueError):
        target.load_restart_file(checkpoint)

    np.testing.assert_array_equal(target._fields["m"], before["m"])
    np.testing.assert_array_equal(target._fields["pin"], before["pin"])
    assert target.clock.checkpoint_state() == before["clock"]


def test_mesh_mismatch_is_rejected_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _simulation(tmp_path, monkeypatch, name="mesh-source")
    checkpoint = source.save_restart_file(tmp_path / "mesh.h5")
    target = _simulation(tmp_path, monkeypatch, name="mesh-target", shifted_mesh=True)
    before = _state_snapshot(target)

    with pytest.raises(ValueError, match="mesh"):
        target.load_m_from_h5file(checkpoint)

    np.testing.assert_array_equal(target._fields["m"], before["m"])


def test_atomic_save_preserves_existing_checkpoint_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="atomic")
    checkpoint = simulation.save_restart_file(tmp_path / "atomic.h5")
    original_m = np.array(simulation._fields["m"], copy=True)
    simulation.set_m([0.0, 0.0, 1.0])

    def fail_replace(_source: str, _target: str) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        simulation.save_restart_file(checkpoint)

    restored = _simulation(tmp_path, monkeypatch, name="atomic-restored")
    restored.load_m_from_h5file(checkpoint)
    np.testing.assert_array_equal(restored._fields["m"], original_m)
    assert not checkpoint.with_name(f".{checkpoint.name}.tmp").exists()


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_checkpoint_resume_matches_uninterrupted_stt_dynamics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    if backend == "rust":
        pytest.importorskip("nmag_accel")
    config = nmag.NmagConfig(accelerator="rust" if backend == "rust" else "off")
    uninterrupted = _simulation(
        tmp_path, monkeypatch, name=f"uninterrupted-{backend}", config=config
    )
    uninterrupted.advance_time(nmag.SI(2.0e-15, "s"))

    staged = _simulation(tmp_path, monkeypatch, name=f"staged-{backend}", config=config)
    staged.advance_time(nmag.SI(1.0e-15, "s"))
    checkpoint = staged.save_restart_file(tmp_path / f"resume-{backend}.h5")
    resumed = _simulation(tmp_path, monkeypatch, name=f"resumed-{backend}", config=config)
    resumed.load_restart_file(checkpoint)
    resumed.advance_time(nmag.SI(2.0e-15, "s"))

    np.testing.assert_allclose(resumed._fields["m"], uninterrupted._fields["m"], rtol=1.0e-8, atol=1.0e-10)
    np.testing.assert_array_equal(resumed._fields["m"][1], staged._fields["m"][1])


def test_save_restart_action_writes_default_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="scheduled-restart")

    simulation.action_abbreviations["save_restart"](simulation)

    assert simulation.get_restart_file_name().exists()
