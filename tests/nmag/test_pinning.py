from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nmag
import nmesh


def _simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
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
        llg_damping=0.5,
        do_precession=False,
    )
    simulation = nmag.Simulation(name=name, do_demag=False)
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m([0.0, 1.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))
    return simulation


def test_default_pinning_leaves_every_node_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="pin-default")

    np.testing.assert_array_equal(simulation.get_subfield("pin"), np.ones(4))
    assert simulation.get_subfield_average("pin") == 1.0


def test_set_pinning_accepts_scalar_array_and_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="pin-inputs")

    simulation.set_pinning(0.5)
    np.testing.assert_array_equal(simulation.get_subfield("pin"), np.full(4, 0.5))
    assert simulation.get_subfield_average("pin") == 0.5

    simulation.set_pinning([0.0, 1.0, 0.25, 0.75])
    np.testing.assert_array_equal(
        simulation.get_subfield("pin"),
        [0.0, 1.0, 0.25, 0.75],
    )

    simulation.set_pinning(lambda point: 0.0 if point[0] == 0.0 else nmag.SI(1))
    np.testing.assert_array_equal(simulation.get_subfield("pin"), [0.0, 1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "values",
    ([0.0, 1.0], [0.0, 1.0, 1.0, float("nan")]),
)
def test_set_pinning_rejects_invalid_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    values: list[float],
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="pin-invalid")

    with pytest.raises(ValueError, match="pinning"):
        simulation.set_pinning(values)


def test_set_pinning_requires_a_mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    simulation = nmag.Simulation(name="pin-no-mesh", do_demag=False)

    with pytest.raises(RuntimeError, match="mesh must be loaded"):
        simulation.set_pinning(0.0)


def test_pinned_nodes_remain_fixed_while_free_nodes_evolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="pin-dynamics")
    simulation.set_pinning(lambda point: 1.0 if point[0] > 0.0 else 0.0)
    initial = np.asarray(simulation.get_subfield("m"), dtype=float)

    simulation.advance_time(nmag.SI(2.0e-12, "s"))

    final = np.asarray(simulation.get_subfield("m"), dtype=float)
    np.testing.assert_array_equal(final[[0, 2, 3]], initial[[0, 2, 3]])
    assert final[1, 0] > initial[1, 0]
    assert final[1, 1] < initial[1, 1]


def test_set_pinning_invalidates_an_initialised_integrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="pin-invalidate")
    simulation.reinitialise()
    assert simulation._integrator_is_stale is False

    simulation.set_pinning(0.0)

    assert simulation._integrator_is_stale is True
