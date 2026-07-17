from __future__ import annotations

from pathlib import Path

import numpy as np

import nmag
import nmesh

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE_MESH = FIXTURES / "nmag_doc_example1" / "sphere1.nmesh.h5"


def test_two_tetra_nonuniform_state_advances_with_demag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "two-tetra.nmesh.h5"
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3], [1, 2, 3, 4]],
        simplices_regions=[1, 1],
    )
    mesh.save(str(mesh_path))
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
        do_precession=False,
    )
    simulation = nmag.Simulation(name="two-tetra")
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m(lambda point: [1.0, point[0] * 0.25, point[1] * 0.25])
    initial = np.asarray(simulation.get_subfield("m"))

    simulation.advance_time(nmag.SI(1.0e-12, "s"), max_it=2)
    final = np.asarray(simulation.get_subfield("m"))

    assert simulation.step == 2
    assert np.all(np.isfinite(final))
    assert not np.allclose(final, initial)
    np.testing.assert_allclose(np.linalg.norm(final, axis=1), 1.0, rtol=2.0e-6)


def test_sphere_dynamic_demag_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
        do_precession=False,
    )
    simulation = nmag.Simulation(name="sphere-dynamic")
    simulation.load_mesh(
        str(SPHERE_MESH),
        [("sphere", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m([1.0, 0.0, 0.0])
    simulation.set_H_ext([0.0, 1.0e4, 0.0], nmag.SI("A/m"))

    simulation.advance_time(nmag.SI(1.0e-9, "s"), max_it=1)
    final = np.asarray(simulation.get_subfield("m"))

    assert simulation.step == 1
    assert np.all(np.isfinite(final))
    np.testing.assert_allclose(np.linalg.norm(final, axis=1), 1.0, rtol=2.0e-6)


def test_exchange_spectral_cap_matches_a_smaller_step_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "stiff-two-tetra.nmesh.h5"
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3], [1, 2, 3, 4]],
        simplices_regions=[1, 1],
    )
    mesh.save(str(mesh_path))

    def make_simulation(name: str) -> nmag.Simulation:
        material = nmag.MagMaterial(
            name="Py",
            Ms=nmag.SI(1.0e6, "A/m"),
            exchange_coupling=nmag.SI(13.0e-12, "J/m"),
            do_precession=False,
        )
        simulation = nmag.Simulation(name=name, do_demag=False)
        simulation.load_mesh(
            str(mesh_path),
            [("magnetic", material)],
            unit_length=nmag.SI(1.0e-9, "m"),
        )
        simulation.set_m(lambda point: [1.0, point[0] * 0.25, point[1] * 0.25])
        return simulation

    automatic = make_simulation("automatic-cap")
    automatic.reinitialise()
    automatic_cap = automatic.effective_integrator_max_step.in_units_of(nmag.SI("s"))
    assert 0.0 < automatic_cap < 1.0e-12
    assert automatic._exchange_spectral_bound_cache is not None

    uniform = make_simulation("uniform-initial-state")
    uniform.set_m([0.0, 1.0, 0.0])
    uniform.reinitialise()
    assert uniform.effective_integrator_max_step.in_units_of(nmag.SI("s")) == automatic_cap

    reference = make_simulation("reference-cap")
    reference.set_params(ts_max_step=automatic_cap / 4.0)
    target = nmag.SI(2.0e-12, "s")
    automatic.advance_time(target)
    reference.advance_time(target)

    np.testing.assert_allclose(
        automatic.get_subfield("m"),
        reference.get_subfield("m"),
        rtol=2.0e-5,
        atol=2.0e-7,
    )

    automatic._invalidate_demag(clear_geometry=True)
    assert automatic._exchange_spectral_bound_cache is None
