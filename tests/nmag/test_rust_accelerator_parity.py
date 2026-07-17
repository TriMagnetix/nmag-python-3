from __future__ import annotations

import numpy as np
import pytest

import nmesh
from nmag.demag import _oriented_boundary_faces
from nmag.simulation import Simulation

nmag_accel = pytest.importorskip("nmag_accel")


def _simulation(points: np.ndarray, simplices: np.ndarray) -> Simulation:
    simulation = Simulation(name="rust-contract")
    simulation.mesh = nmesh.mesh_from_points_and_simplices(
        points=points.tolist(),
        simplices_indices=simplices.tolist(),
        simplices_regions=[1] * len(simplices),
    )
    return simulation


def _boundary_index(
    points: np.ndarray,
    simplices: np.ndarray,
) -> tuple[list[tuple[int, tuple[int, int, int]]], np.ndarray, np.ndarray, np.ndarray]:
    boundary_faces = _oriented_boundary_faces(points, simplices)
    boundary_nodes = np.asarray(
        sorted({node for _, face in boundary_faces for node in face}),
        dtype=np.int64,
    )
    local_index_by_point = np.full(len(points), -1, dtype=np.int64)
    local_index_by_point[boundary_nodes] = np.arange(len(boundary_nodes), dtype=np.int64)
    face_nodes = np.asarray([face for _, face in boundary_faces], dtype=np.int64)
    return boundary_faces, boundary_nodes, local_index_by_point, face_nodes


def _two_tetra_mesh() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
    )


def _random_two_tetra_mesh(seed: int) -> tuple[np.ndarray, np.ndarray]:
    points, simplices = _two_tetra_mesh()
    rng = np.random.default_rng(seed)
    points += rng.normal(scale=0.03, size=points.shape)
    return points, simplices


@pytest.mark.parametrize("seed", range(3))
def test_randomized_mesh_kernels_match_python_references(seed: int) -> None:
    points, simplices = _random_two_tetra_mesh(seed)
    simulation = _simulation(points, simplices)

    python_geometry = simulation._demag_fem_geometry_for_mesh_python(points, simplices)
    rust_geometry = nmag_accel.build_demag_fem_geometry(points, simplices)
    for rust_part, python_part in zip(rust_geometry, python_geometry, strict=True):
        np.testing.assert_allclose(rust_part, python_part, rtol=1.0e-12, atol=1.0e-14)

    expected_faces = _oriented_boundary_faces(points, simplices)
    owners, faces = nmag_accel.build_oriented_boundary_faces(points, simplices)
    actual_faces = [
        (int(owner), tuple(int(node) for node in face))
        for owner, face in zip(owners, faces, strict=True)
    ]
    assert actual_faces == expected_faces

    rust_probe_geometry = nmag_accel.build_probe_tetrahedral_geometry(points, simplices)
    python_probe_geometry = simulation._probe_tetrahedral_geometry_python()
    for rust_part, python_part in zip(rust_probe_geometry, python_probe_geometry, strict=True):
        np.testing.assert_allclose(rust_part, python_part, rtol=1.0e-13, atol=1.0e-14)

    boundary_faces, boundary_nodes, local_indices, face_nodes = _boundary_index(points, simplices)
    python_nodes, python_bem = simulation._build_lindholm_bem_matrix_python(
        points,
        simplices,
        boundary_faces,
    )
    rust_bem = nmag_accel.build_lindholm_bem_matrix(
        points,
        simplices,
        face_nodes,
        boundary_nodes,
        local_indices,
    )
    np.testing.assert_array_equal(boundary_nodes, python_nodes)
    np.testing.assert_allclose(rust_bem, python_bem, rtol=1.0e-12, atol=1.0e-14)


@pytest.mark.parametrize("seed", range(3))
def test_randomized_field_kernels_match_python_references(seed: int) -> None:
    points, simplices = _two_tetra_mesh()
    simulation = _simulation(points, simplices)
    _, gradients, volumes = simulation._demag_fem_geometry_for_mesh_python(points, simplices)

    rng = np.random.default_rng(seed + 100)
    m = rng.normal(size=(len(points), 3))
    h_total = rng.normal(size=(len(points), 3))
    ms_by_cell = rng.uniform(0.5, 2.0, size=len(simplices))
    expected_divergence = simulation._assemble_demag_fem_divergence_python(
        simplices,
        gradients,
        volumes,
        m,
        ms_by_cell,
        len(points),
    )
    rust_divergence = nmag_accel.build_demag_fem_divergence(
        simplices,
        gradients,
        volumes,
        m,
        ms_by_cell,
        len(points),
    )
    np.testing.assert_allclose(rust_divergence, expected_divergence, rtol=1.0e-13)

    cell_h = rng.normal(size=(len(simplices), 3))
    recovery_volumes = np.asarray([volumes[0], -volumes[1]])
    weights = rng.uniform(0.25, 2.0, size=len(points))
    expected_nodal = simulation._recover_demag_nodal_field_python(
        simplices,
        recovery_volumes,
        cell_h,
        weights,
        len(points),
    )
    rust_nodal = nmag_accel.recover_demag_nodal_field(
        simplices,
        recovery_volumes,
        cell_h,
        weights,
        len(points),
    )
    np.testing.assert_allclose(rust_nodal, expected_nodal, rtol=1.0e-13)

    average_volumes = rng.uniform(-0.5, 2.0, size=len(simplices))
    expected_average = simulation._demag_cell_field_average_python(cell_h, average_volumes)
    rust_average = nmag_accel.demag_cell_field_average(cell_h, average_volumes)
    np.testing.assert_allclose(rust_average, expected_average, rtol=1.0e-13)

    pin = rng.uniform(0.0, 1.0, size=len(points))
    ms_by_point = rng.uniform(0.5, 2.0, size=len(points))
    coefficients = (0.75, -0.25, 0.125)
    expected_rhs = simulation._llg_rhs_python(
        m,
        h_total,
        pin,
        ms_by_point,
        *coefficients,
    )
    rust_rhs = nmag_accel.llg_rhs(
        m,
        h_total,
        pin,
        ms_by_point,
        *coefficients,
    )
    np.testing.assert_allclose(rust_rhs, expected_rhs, rtol=1.0e-13, atol=1.0e-14)

    normalised_m = m / np.linalg.norm(m, axis=1, keepdims=True)
    simulation._fields["m"] = normalised_m
    expected_maxangle = simulation._maxangle_average_reference("m")
    rust_maxangle = nmag_accel.maxangle_between_edges(normalised_m, simplices)
    assert rust_maxangle == pytest.approx(expected_maxangle, rel=1.0e-13)
