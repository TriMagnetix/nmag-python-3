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


def test_scipy_dirichlet_factorization_caches_boundary_stiffness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "scipy")
    from pathlib import Path

    sphere_fixture = (
        Path(__file__).resolve().parents[1] / "fixtures" / "nmag_doc_example1" / "sphere1.nmesh.h5"
    )
    material = nmag.MagMaterial(name="Py", Ms=nmag.SI(1.0e6, "A/m"))
    simulation = nmag.Simulation(name="factor-dirichlet")
    simulation.load_mesh(
        str(sphere_fixture),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m([1.0, 0.0, 0.0])
    first = np.asarray(simulation.get_subfield("H_demag"))
    dirichlet_cache = simulation._demag_dirichlet_factorization_cache
    assert dirichlet_cache is not None
    assert len(dirichlet_cache) == 4

    simulation.set_m([0.0, 1.0, 0.0])
    second = np.asarray(simulation.get_subfield("H_demag"))
    assert simulation._demag_dirichlet_factorization_cache is dirichlet_cache
    assert not np.allclose(first, second)


def test_gauge_fixed_sparse_factorization_matches_dense_augmented_solve(monkeypatch):
    from nmag.backends import DEMAG_FEM_MATRIX_BACKEND_ENV
    from nmag.demag.linear import _factorize_dense_scipy, _factorize_gauge_fixed

    n = 70
    main_diag = np.full(n, 2.0, dtype=float)
    main_diag[0] = 1.0
    main_diag[-1] = 1.0
    off_diag = np.full(n - 1, -1.0, dtype=float)
    laplacian = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    sparse_factorization = _factorize_gauge_fixed(laplacian)
    assert sparse_factorization.splu is not None
    assert sparse_factorization.gauge_fixed is True

    constraint = np.ones(n, dtype=float)
    augmented = np.block(
        [
            [laplacian, constraint[:, np.newaxis]],
            [constraint[np.newaxis, :], np.zeros((1, 1), dtype=float)],
        ]
    )
    dense_factorization = _factorize_dense_scipy(augmented)

    rng = np.random.default_rng(42)
    div_1d = rng.standard_normal(n)
    np.testing.assert_allclose(
        sparse_factorization.solve_gauge(div_1d),
        dense_factorization.solve_gauge(div_1d),
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    rhs_1d = rng.standard_normal(n + 1)
    np.testing.assert_allclose(
        sparse_factorization.solve(rhs_1d),
        dense_factorization.solve(rhs_1d),
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    rhs_2d = rng.standard_normal((n + 1, 3))
    np.testing.assert_allclose(
        sparse_factorization.solve(rhs_2d),
        dense_factorization.solve(rhs_2d),
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "dense")
    forced_dense = _factorize_gauge_fixed(laplacian)
    assert forced_dense.splu is None


def test_factorize_gauge_fixed_handles_disconnected_components():
    import pytest

    from nmag.demag.linear import _factorize_gauge_fixed, _factorize_scipy

    # Two disconnected 35-node components (n=70 >= 64 sparse threshold) -> rank-deficient submatrix
    comp_size = 35
    main_diag = np.full(comp_size, 2.0, dtype=float)
    main_diag[0] = 1.0
    main_diag[-1] = 1.0
    off_diag = np.full(comp_size - 1, -1.0, dtype=float)
    comp = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    zeros = np.zeros((comp_size, comp_size), dtype=float)
    disconnected = np.block([[comp, zeros], [zeros, comp]])

    factorization = _factorize_gauge_fixed(disconnected)
    assert factorization.splu is None
    divergence = np.zeros(2 * comp_size, dtype=float)
    divergence[0] = 1.0
    divergence[comp_size - 1] = -1.0
    divergence[comp_size] = 0.5
    divergence[-1] = -0.5
    phi = factorization.solve_gauge(divergence)
    assert np.all(np.isfinite(phi))
    np.testing.assert_allclose(disconnected @ phi, divergence, atol=1.0e-10)

    # Scale-relative row-sum check: non-zero row sums at 1e-9 scale must not use node-0 pinning
    non_laplacian_nm = disconnected.copy() * 1.0e-9
    non_laplacian_nm[0, 0] += 1.0e-10
    factorization_nm = _factorize_gauge_fixed(non_laplacian_nm)
    assert factorization_nm.splu is None

    # Singular 70x70 system in _factorize_scipy falls back to least-squares instead of corrupt splu
    singular_70 = disconnected.copy()
    scipy_factor = _factorize_scipy(singular_70)
    assert scipy_factor.splu is None
    sol = scipy_factor.solve(divergence)
    assert np.all(np.isfinite(sol))
    np.testing.assert_allclose(singular_70 @ sol, divergence, atol=1.0e-10)

    # 0x0 gauge-fixed system rejects inconsistent non-zero constraint RHS
    empty_gauge = _factorize_gauge_fixed(np.zeros((0, 0), dtype=float))
    np.testing.assert_allclose(empty_gauge.solve(np.array([0.0])), [0.0])
    with pytest.raises(np.linalg.LinAlgError):
        empty_gauge.solve(np.array([1.0]))

    # Disconnected nanometre-scale (1e-9) stiffness matrix solves accurately via scaled border
    from nmag.demag.linear import _solve_gauge_fixed

    disconnected_nm = disconnected.copy() * 1.0e-9
    divergence_nm = divergence * 1.0e-9
    factorization_disc_nm = _factorize_gauge_fixed(disconnected_nm)
    phi_disc_nm = factorization_disc_nm.solve_gauge(divergence_nm)
    assert np.all(np.isfinite(phi_disc_nm))
    np.testing.assert_allclose(disconnected_nm @ phi_disc_nm, divergence_nm, atol=1.0e-19)

    phi_direct_nm = _solve_gauge_fixed(disconnected_nm, divergence_nm)
    assert np.all(np.isfinite(phi_direct_nm))
    np.testing.assert_allclose(disconnected_nm @ phi_direct_nm, divergence_nm, atol=1.0e-19)

    # Non-zero gauge target g in solve(rhs) with constraint_scale != 1.0
    connected_small_nm = comp[:10, :10].copy() * 1.0e-9
    connected_small_nm[0, 0] = 1.0e-9
    connected_small_nm[-1, -1] = 1.0e-9
    small_factor = _factorize_gauge_fixed(connected_small_nm)
    rhs_aug = np.zeros(11, dtype=float)
    rhs_aug[0] = 1.0e-9
    rhs_aug[9] = -1.0e-9
    rhs_aug[10] = 3.5
    sol_aug = small_factor.solve(rhs_aug)
    np.testing.assert_allclose(np.sum(sol_aug[:10]), 3.5, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        connected_small_nm @ sol_aug[:10] + sol_aug[10],
        rhs_aug[:10],
        atol=1.0e-19,
    )


def test_disconnected_nanometre_mesh_demag_consistency_across_backends(tmp_path, monkeypatch):
    import nmesh
    from nmag.backends import (
        DEMAG_LINEAR_SOLVER_BACKEND_ENV,
        DEMAG_SOLVE_CONDITION_DIAGNOSTICS_ENV,
    )

    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [10.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
        [10.0, 1.0, 0.0],
        [10.0, 0.0, 1.0],
    ]
    simplices = [[0, 1, 2, 3], [4, 5, 6, 7]]
    mesh = nmesh.mesh_from_points_and_simplices(points, simplices, [1, 1])
    mesh_path = tmp_path / "disconnected_two_tet.nmesh"
    mesh.save(str(mesh_path))

    monkeypatch.setenv(DEMAG_SOLVE_CONDITION_DIAGNOSTICS_ENV, "1")
    results = {}
    for backend in ("auto", "numpy", "scipy"):
        monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, backend)
        sim = nmag.Simulation(do_demag=True)
        mat = nmag.MagMaterial(
            name="Py",
            Ms=nmag.SI(8e5, "A/m"),
            exchange_coupling=nmag.SI(1.3e-11, "J/m"),
        )
        sim.load_mesh(str(mesh_path), [("region1", mat)], unit_length=nmag.SI(1e-9, "m"))
        sim.set_m([1.0, 0.0, 0.0])
        results[backend] = np.asarray(sim.get_subfield("H_demag"), dtype=float)
        diag = sim.last_demag_solve_diagnostics
        assert diag["phi1_centered_residual_relative_l2"] < 1.0e-12

    np.testing.assert_allclose(results["auto"], results["scipy"], rtol=1.0e-10, atol=1.0e-6)
    np.testing.assert_allclose(results["numpy"], results["scipy"], rtol=1.0e-10, atol=1.0e-6)

    # Single-tetrahedron condition number diagnostic reflects scaled gauge constraint border
    single_mesh = nmesh.mesh_from_points_and_simplices(points[:4], [[0, 1, 2, 3]], [1])
    single_path = tmp_path / "single_tet.nmesh"
    single_mesh.save(str(single_path))
    sim_single = nmag.Simulation(do_demag=True)
    mat_single = nmag.MagMaterial(name="Py", Ms=nmag.SI(8e5, "A/m"))
    sim_single.load_mesh(str(single_path), [("region1", mat_single)], unit_length=nmag.SI(1e-9, "m"))
    sim_single.set_m([1.0, 0.0, 0.0])
    _ = sim_single.get_subfield("H_demag")
    assert sim_single.last_demag_solve_diagnostics["gauge_augmented_condition_number"] < 100.0




