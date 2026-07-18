from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nmag
import nmag.backends as backends
import nmesh
from nmag.backends import (
    DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES_ENV,
    DEMAG_BEM_STORAGE_BACKEND_ENV,
    DEMAG_FEM_MATRIX_BACKEND_ENV,
    MEMORY_MODE_ENV,
    _selected_demag_bem_storage_backend,
)
from nmag.demag.bem_operator import MatrixFreeLindholmBemOperator

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


def _simulation(
    mesh_path: Path,
    name: str,
    *,
    accelerator: str = "auto",
    storage: str = "auto",
) -> nmag.Simulation:
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
        llg_damping=0.5,
        do_precession=False,
    )
    simulation = nmag.Simulation(
        name=name,
        config=nmag.NmagConfig(accelerator=accelerator, demag_bem_storage=storage),
    )
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m(lambda point: [1.0, 0.2 * point[0], 0.1 * point[1]])
    return simulation


def test_auto_bem_storage_switches_without_rejecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DEMAG_BEM_STORAGE_BACKEND_ENV, raising=False)
    monkeypatch.setenv(DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES_ENV, "5")
    monkeypatch.setattr(backends, "available_memory_bytes", lambda: None)
    monkeypatch.setattr(backends, "_rust_accelerator_available", lambda: False)

    assert _selected_demag_bem_storage_backend(4) == "dense"
    assert _selected_demag_bem_storage_backend(5) == "matrix-free"


def test_low_memory_mode_forces_matrix_free_bem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEMAG_BEM_STORAGE_BACKEND_ENV, raising=False)
    monkeypatch.setenv(MEMORY_MODE_ENV, "low")

    assert _selected_demag_bem_storage_backend(1) == "matrix-free"


def test_auto_bem_storage_respects_available_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DEMAG_BEM_STORAGE_BACKEND_ENV, raising=False)
    monkeypatch.setenv(DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES_ENV, "10000")
    monkeypatch.setattr(backends, "available_memory_bytes", lambda: 1_000)
    monkeypatch.setattr(backends, "_rust_accelerator_available", lambda: False)

    assert _selected_demag_bem_storage_backend(10) == "matrix-free"


def test_auto_bem_storage_keeps_dense_above_fallback_when_memory_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DEMAG_BEM_STORAGE_BACKEND_ENV, raising=False)
    monkeypatch.setenv(DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES_ENV, "5")
    monkeypatch.setattr(backends, "available_memory_bytes", lambda: 1_000_000)

    assert _selected_demag_bem_storage_backend(10) == "dense"


def test_auto_bem_storage_prefers_hierarchy_when_rust_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES_ENV, "5")
    monkeypatch.setattr(backends, "available_memory_bytes", lambda: None)
    monkeypatch.setattr(backends, "_rust_accelerator_available", lambda: True)

    assert _selected_demag_bem_storage_backend(5) == "hierarchical"


def test_matrix_free_bem_action_matches_dense(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "two-tetra.nmesh.h5"
    _two_tetra_mesh(mesh_path)
    simulation = _simulation(mesh_path, "bem-action")
    points = np.asarray(simulation.mesh.points, dtype=float)
    simplices = np.asarray(simulation.mesh.simplices, dtype=int)
    faces = simulation._boundary_faces_for_demag_mesh(points, simplices)

    boundary_nodes, dense = simulation._build_lindholm_bem_matrix_numba(
        points,
        simplices,
        faces,
    )
    operator_nodes, operator = simulation._build_lindholm_bem_operator(
        points,
        simplices,
        faces,
    )
    values = np.linspace(-0.75, 1.25, len(boundary_nodes))

    assert isinstance(operator, MatrixFreeLindholmBemOperator)
    np.testing.assert_array_equal(operator_nodes, boundary_nodes)
    np.testing.assert_allclose(operator @ values, dense @ values, rtol=2.0e-13, atol=2.0e-15)


def test_rust_matrix_free_bem_matches_numba_for_random_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("nmag_accel")
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "two-tetra.nmesh.h5"
    _two_tetra_mesh(mesh_path)
    numba_simulation = _simulation(mesh_path, "bem-random-numba", accelerator="off")
    rust_simulation = _simulation(mesh_path, "bem-random-rust", accelerator="rust")
    points = np.asarray(numba_simulation.mesh.points, dtype=float)
    simplices = np.asarray(numba_simulation.mesh.simplices, dtype=int)
    faces = numba_simulation._boundary_faces_for_demag_mesh(points, simplices)

    _nodes, numba_operator = numba_simulation._build_lindholm_bem_operator(
        points,
        simplices,
        faces,
    )
    _nodes, rust_operator = rust_simulation._build_lindholm_bem_operator(
        points,
        simplices,
        faces,
    )

    generator = np.random.default_rng(20260715)
    for _ in range(8):
        values = generator.standard_normal(numba_operator.shape[1])
        np.testing.assert_allclose(
            rust_operator @ values,
            numba_operator @ values,
            rtol=3.0e-13,
            atol=3.0e-15,
        )


def test_rust_matrix_free_bem_rejects_mismatched_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust_accel = pytest.importorskip("nmag_accel")
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "two-tetra.nmesh.h5"
    _two_tetra_mesh(mesh_path)
    simulation = _simulation(mesh_path, "bem-invalid")
    points = np.asarray(simulation.mesh.points, dtype=float)
    simplices = np.asarray(simulation.mesh.simplices, dtype=np.int64)
    faces = simulation._boundary_faces_for_demag_mesh(points, simplices)
    boundary_nodes, local_index_by_point, face_nodes = simulation._lindholm_bem_boundary_index(
        points,
        faces,
    )

    with pytest.raises(ValueError, match="one entry per boundary node"):
        rust_accel.apply_lindholm_bem_matrix_free(
            points,
            simplices,
            face_nodes,
            boundary_nodes,
            local_index_by_point,
            np.zeros(len(boundary_nodes) - 1),
        )


def test_matrix_free_bem_uses_less_storage_for_sphere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    simulation = _simulation(SPHERE_MESH, "bem-storage")
    points = np.asarray(simulation.mesh.points, dtype=float)
    simplices = np.asarray(simulation.mesh.simplices, dtype=int)
    faces = simulation._boundary_faces_for_demag_mesh(points, simplices)

    boundary_nodes, operator = simulation._build_lindholm_bem_operator(
        points,
        simplices,
        faces,
    )
    dense_bytes = len(boundary_nodes) ** 2 * np.dtype(np.float64).itemsize

    assert operator.storage_bytes < dense_bytes
    assert not hasattr(operator, "toarray")


def test_matrix_free_demag_matches_dense(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "two-tetra.nmesh.h5"
    _two_tetra_mesh(mesh_path)

    monkeypatch.setenv(DEMAG_BEM_STORAGE_BACKEND_ENV, "dense")
    dense = _simulation(mesh_path, "dense-bem", storage="dense")
    dense_field = np.asarray(dense.get_subfield("H_demag"))

    monkeypatch.setenv(DEMAG_BEM_STORAGE_BACKEND_ENV, "matrix-free")
    matrix_free = _simulation(mesh_path, "matrix-free-bem", storage="matrix-free")
    matrix_free_field = np.asarray(matrix_free.get_subfield("H_demag"))

    np.testing.assert_allclose(matrix_free_field, dense_field, rtol=2.0e-12, atol=1.0e-7)


def test_low_memory_fixed_time_dynamics_match_dense(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "two-tetra.nmesh.h5"
    _two_tetra_mesh(mesh_path)
    target = nmag.SI(2.0e-13, "s")

    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "dense")
    monkeypatch.setenv(DEMAG_BEM_STORAGE_BACKEND_ENV, "dense")
    dense = _simulation(mesh_path, "dense-dynamics", storage="dense")
    dense.advance_time(target)

    monkeypatch.setenv(DEMAG_FEM_MATRIX_BACKEND_ENV, "sparse")
    monkeypatch.setenv(DEMAG_BEM_STORAGE_BACKEND_ENV, "matrix-free")
    low_memory = _simulation(mesh_path, "low-memory-dynamics", storage="matrix-free")
    low_memory.advance_time(target)

    assert low_memory._llg_affine_operator_cache is None
    assert low_memory.clock.time_reached_si == dense.clock.time_reached_si
    np.testing.assert_allclose(
        low_memory.get_subfield("m"),
        dense.get_subfield("m"),
        rtol=2.0e-8,
        atol=2.0e-9,
    )
