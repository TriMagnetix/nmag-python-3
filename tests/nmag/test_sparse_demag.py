from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import nmag
import nmag.backends as backends
import nmesh
from nmag.backends import (
    DEMAG_DENSE_MAX_POINTS_ENV,
    DEMAG_FEM_AUTO_SPARSE_MIN_POINTS_ENV,
    DEMAG_FEM_MATRIX_BACKEND_ENV,
    MEMORY_MODE_ENV,
    _selected_demag_fem_matrix_backend,
)
from nmag.demag.linear import _validate_dense_demag_size

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE_MESH = FIXTURES / "nmag_doc_example1" / "sphere1.nmesh.h5"


def _two_tetra_mesh(path: Path) -> None:
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
    mesh.save(str(path))


def _simulation(mesh_path: Path, name: str) -> nmag.Simulation:
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
        llg_damping=0.5,
        do_precession=False,
    )
    simulation = nmag.Simulation(name=name)
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m(lambda point: [1.0, 0.2 * point[0], 0.1 * point[1]])
    return simulation


def test_auto_matrix_backend_switches_to_sparse_without_rejecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DEMAG_FEM_MATRIX_BACKEND_ENV, raising=False)
    monkeypatch.setenv(DEMAG_FEM_AUTO_SPARSE_MIN_POINTS_ENV, "5")
    monkeypatch.setattr(backends, "available_memory_bytes", lambda: None)

    assert _selected_demag_fem_matrix_backend(4) == "dense"
    assert _selected_demag_fem_matrix_backend(5) == "sparse"


def test_low_memory_mode_forces_sparse_fem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEMAG_FEM_MATRIX_BACKEND_ENV, raising=False)
    monkeypatch.setenv(MEMORY_MODE_ENV, "low")

    assert _selected_demag_fem_matrix_backend(1) == "sparse"


def test_auto_matrix_backend_respects_available_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DEMAG_FEM_MATRIX_BACKEND_ENV, raising=False)
    monkeypatch.setenv(DEMAG_FEM_AUTO_SPARSE_MIN_POINTS_ENV, "10000")
    monkeypatch.setattr(backends, "available_memory_bytes", lambda: 1_000)

    assert _selected_demag_fem_matrix_backend(10) == "sparse"


def test_auto_matrix_backend_keeps_dense_above_fallback_when_memory_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DEMAG_FEM_MATRIX_BACKEND_ENV, raising=False)
    monkeypatch.setenv(DEMAG_FEM_AUTO_SPARSE_MIN_POINTS_ENV, "5")
    monkeypatch.setattr(backends, "available_memory_bytes", lambda: 1_000_000)

    assert _selected_demag_fem_matrix_backend(10) == "dense"


def test_dense_demag_has_no_default_point_count_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DEMAG_DENSE_MAX_POINTS_ENV, raising=False)

    _validate_dense_demag_size(1_000_000)


def test_sparse_fem_stiffness_matches_dense(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh_path = tmp_path / "two-tetra.nmesh.h5"
    _two_tetra_mesh(mesh_path)
    simulation = _simulation(mesh_path, "sparse-stiffness")
    points = np.asarray(simulation.mesh.points, dtype=float)
    simplices = np.asarray(simulation.mesh.simplices, dtype=int)

    dense, dense_gradients, dense_volumes = simulation._demag_fem_geometry_for_mesh_python(
        points,
        simplices,
    )
    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "sparse")
    simulation._invalidate_demag(clear_geometry=True)
    actual, gradients, volumes = simulation._demag_fem_geometry_for_mesh(points, simplices)

    assert sparse.isspmatrix_csr(actual)
    assert actual.nnz <= 16 * len(simplices)
    np.testing.assert_allclose(actual.toarray(), dense, rtol=1.0e-14, atol=1.0e-30)
    np.testing.assert_allclose(gradients, dense_gradients, rtol=1.0e-14, atol=1.0e-30)
    np.testing.assert_allclose(volumes, dense_volumes, rtol=1.0e-14, atol=1.0e-30)


def test_sparse_demag_and_exchange_match_dense(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "two-tetra.nmesh.h5"
    _two_tetra_mesh(mesh_path)

    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "dense")
    dense = _simulation(mesh_path, "dense-fields")
    dense_demag = np.asarray(dense.get_subfield("H_demag"))
    dense_exchange = np.asarray(dense._subfield_array("H_exch"))

    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "sparse")
    sparse_simulation = _simulation(mesh_path, "sparse-fields")
    sparse_demag = np.asarray(sparse_simulation.get_subfield("H_demag"))
    sparse_exchange = np.asarray(sparse_simulation._subfield_array("H_exch"))

    np.testing.assert_allclose(sparse_demag, dense_demag, rtol=2.0e-11, atol=1.0e-6)
    np.testing.assert_allclose(sparse_exchange, dense_exchange, rtol=2.0e-13, atol=1.0e-6)


def test_sparse_sphere_demag_matches_dense(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "dense")
    dense = _simulation(SPHERE_MESH, "dense-sphere")
    dense.set_m([1.0, 0.0, 0.0])
    dense_field = np.asarray(dense.get_subfield("H_demag"))

    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "sparse")
    sparse_simulation = _simulation(SPHERE_MESH, "sparse-sphere")
    sparse_simulation.set_m([1.0, 0.0, 0.0])
    sparse_field = np.asarray(sparse_simulation.get_subfield("H_demag"))

    np.testing.assert_allclose(sparse_field, dense_field, rtol=2.0e-10, atol=2.0e-5)


def test_sparse_condition_diagnostics_do_not_materialize_dense_matrices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "two-tetra.nmesh.h5"
    _two_tetra_mesh(mesh_path)
    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "sparse")
    monkeypatch.setenv("NMAG_DEMAG_SOLVE_CONDITION_DIAGNOSTICS", "1")

    simulation = _simulation(mesh_path, "sparse-diagnostics")
    simulation.get_subfield("H_demag")

    diagnostics = simulation.last_demag_solve_diagnostics
    assert diagnostics["condition_numbers_skipped_for_sparse_storage"] == 1
    assert "stiffness_condition_number" not in diagnostics


def test_sparse_fixed_time_dynamics_match_dense(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "two-tetra.nmesh.h5"
    _two_tetra_mesh(mesh_path)
    target = nmag.SI(2.0e-13, "s")

    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "dense")
    dense = _simulation(mesh_path, "dense-dynamics")
    dense.advance_time(target)

    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "sparse")
    sparse_simulation = _simulation(mesh_path, "sparse-dynamics")
    sparse_simulation.advance_time(target)

    np.testing.assert_allclose(
        sparse_simulation.get_subfield("m"),
        dense.get_subfield("m"),
        rtol=2.0e-8,
        atol=2.0e-9,
    )
