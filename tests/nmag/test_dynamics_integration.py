from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nmag
import nmag.simulation.dynamics as dynamics_module
import nmesh


def _simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    damping: float = 0.5,
    precession: bool = True,
) -> nmag.Simulation:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / f"{name}.nmesh.h5"
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[1],
    )
    mesh.save(str(mesh_path))
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(0.0, "J/m"),
        llg_damping=damping,
        do_precession=precession,
    )
    simulation = nmag.Simulation(name=name, do_demag=False)
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    return simulation


def test_zero_torque_state_remains_constant(tmp_path, monkeypatch):
    simulation = _simulation(tmp_path, monkeypatch, name="zero-torque")
    simulation.set_m([1.0, 0.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))

    reached = simulation.advance_time(nmag.SI(2.0e-12, "s"))

    assert reached.in_units_of(nmag.SI(1.0, "s")) == pytest.approx(2.0e-12)
    np.testing.assert_allclose(simulation.get_subfield("m"), [[1.0, 0.0, 0.0]] * 4)
    assert simulation.last_integrator_stats.accepted_steps > 0
    assert simulation.last_integrator_stats.failed is False
    assert simulation.effective_integrator_max_step.in_units_of(nmag.SI("s")) == pytest.approx(
        1.0e-12
    )


def test_damping_moves_magnetisation_towards_external_field(tmp_path, monkeypatch):
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="damping",
        precession=False,
    )
    simulation.set_m([0.0, 1.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))

    simulation.advance_time(nmag.SI(2.0e-12, "s"))
    average = np.asarray(simulation.get_subfield_average("m"))

    assert average[0] > 0.0
    assert average[1] < 1.0


def test_precession_preserves_magnetisation_norm(tmp_path, monkeypatch):
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="precession",
        damping=0.0,
        precession=True,
    )
    simulation.set_m([0.0, 1.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))

    simulation.advance_time(nmag.SI(5.0e-12, "s"))
    norms = np.linalg.norm(np.asarray(simulation.get_subfield("m")), axis=1)

    np.testing.assert_allclose(norms, 1.0, rtol=1.0e-8, atol=1.0e-10)


def test_advance_time_honours_maximum_accepted_steps(tmp_path, monkeypatch):
    simulation = _simulation(tmp_path, monkeypatch, name="max-it")
    simulation.set_m([0.0, 1.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))

    reached = simulation.advance_time(nmag.SI(1.0e-9, "s"), max_it=2)

    assert simulation.step == 2
    assert simulation.last_integrator_stats.accepted_steps == 2
    assert reached < nmag.SI(1.0e-9, "s")


def test_exact_stop_interpolates_and_reinitialises(tmp_path, monkeypatch):
    simulation = _simulation(tmp_path, monkeypatch, name="exact-stop")
    simulation.set_m([0.0, 1.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))
    simulation.set_params(ts_max_step=nmag.SI(1.0e-12, "s"))

    reached = simulation.advance_time(nmag.SI(1.5e-12, "s"), exact_tstop=True)

    assert reached.in_units_of(nmag.SI(1.0, "s")) == pytest.approx(1.5e-12)
    assert simulation.time.in_units_of(nmag.SI(1.0, "s")) == pytest.approx(1.5e-12)
    assert simulation.stage_time.in_units_of(nmag.SI(1.0, "s")) == pytest.approx(1.5e-12)
    assert simulation._integrator_is_stale is False


def test_setters_invalidate_an_initialised_integrator(tmp_path, monkeypatch):
    simulation = _simulation(tmp_path, monkeypatch, name="invalidate")
    simulation.set_m([0.0, 1.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))
    simulation.reinitialise()
    assert simulation._integrator_is_stale is False

    simulation.set_H_ext([2.0e5, 0.0, 0.0], nmag.SI("A/m"))

    assert simulation._integrator_is_stale is True


def test_relax_reaches_convergence_and_preserves_legacy_save_cadence(tmp_path, monkeypatch):
    simulation = _simulation(tmp_path, monkeypatch, name="relax")
    simulation.set_m([1.0, 0.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))

    simulation.relax()

    assert simulation.clock.convergence is True
    assert simulation.step == 10
    lines = (tmp_path / "relax_dat.ndt").read_text(encoding="utf-8").splitlines()
    header = lines[1].split("\t")
    rows = [dict(zip(header, line.split("\t"), strict=True)) for line in lines[2:]]
    assert [row["id"] for row in rows] == ["0", "1"]
    assert len({row["step"] for row in rows}) == 1


def test_failed_integrator_is_reported(tmp_path, monkeypatch):
    simulation = _simulation(tmp_path, monkeypatch, name="failure")
    simulation.set_m([1.0, 0.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))

    class FailedIntegrator:
        def __init__(self, *_args, **_kwargs):
            self.t = 0.0
            self.y = np.asarray(simulation._fields["m"]).reshape(-1)
            self.status = "running"

        def step(self):
            self.status = "failed"
            return "forced failure"

        def dense_output(self):
            raise AssertionError("dense output should not be requested")

    monkeypatch.setattr(dynamics_module, "_dop853_class", lambda: FailedIntegrator)

    with pytest.raises(RuntimeError, match="forced failure"):
        simulation.advance_time(nmag.SI(1.0e-12, "s"))

    assert simulation.last_integrator_stats.failed is True
    assert simulation.last_integrator_stats.status == "failed"


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("stopping_dm_dt", 0.0),
        ("ts_rel_tol", 0.0),
        ("ts_abs_tol", -1.0),
        ("ts_max_step", float("nan")),
    ],
)
def test_set_params_rejects_invalid_values(tmp_path, monkeypatch, keyword, value):
    simulation = _simulation(tmp_path, monkeypatch, name=f"invalid-{keyword}")

    with pytest.raises(ValueError):
        simulation.set_params(**{keyword: value})
