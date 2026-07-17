from __future__ import annotations

import numpy as np

import nmag
import nmesh
from nmag.backends import DEMAG_LINEAR_SOLVER_BACKEND_ENV


def _simulation(tmp_path, monkeypatch, backend: str) -> nmag.Simulation:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, backend)
    mesh_path = tmp_path / "mesh.nmesh.h5"
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
    material = nmag.MagMaterial(name="Py", Ms=nmag.SI(1.0e6, "A/m"))
    simulation = nmag.Simulation(name=f"factor-{backend}")
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    return simulation


def test_scipy_demag_factorization_is_reused_across_magnetisation_states(tmp_path, monkeypatch):
    simulation = _simulation(tmp_path, monkeypatch, "scipy")
    simulation.set_m([1.0, 0.0, 0.0])
    first = np.asarray(simulation.get_subfield("H_demag"))
    factorization = simulation._demag_gauge_factorization_cache

    simulation.set_m([0.0, 1.0, 0.0])
    second = np.asarray(simulation.get_subfield("H_demag"))

    assert factorization is not None
    assert simulation._demag_gauge_factorization_cache is factorization
    assert not np.allclose(first, second)


def test_cached_scipy_demag_matches_uncached_numpy_reference(tmp_path, monkeypatch):
    simulation = _simulation(tmp_path, monkeypatch, "numpy")
    simulation.set_m([1.0, 2.0, 3.0])
    reference = np.asarray(simulation.get_subfield("H_demag"))

    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "scipy")
    simulation._invalidate_demag(clear_geometry=True)
    actual = np.asarray(simulation.get_subfield("H_demag"))

    np.testing.assert_allclose(actual, reference, rtol=1.0e-11, atol=1.0e-6)
    assert simulation._demag_gauge_factorization_cache is not None


def test_geometry_invalidation_clears_demag_factorizations(tmp_path, monkeypatch):
    simulation = _simulation(tmp_path, monkeypatch, "scipy")
    simulation.set_m([1.0, 0.0, 0.0])
    simulation.get_subfield("H_demag")

    assert simulation._demag_gauge_factorization_cache is not None
    simulation._invalidate_demag(clear_geometry=True)

    assert simulation._demag_gauge_factorization_cache is None
    assert simulation._demag_dirichlet_factorization_cache is None
