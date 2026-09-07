from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull, Delaunay

nmag_accel: Any = pytest.importorskip("nmag_accel")


@dataclass(frozen=True, slots=True)
class OperatorMesh:
    points: np.ndarray
    simplices: np.ndarray


def _sphere_mesh(boundary_nodes: int = 80) -> OperatorMesh:
    indices = np.arange(boundary_nodes, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / boundary_nodes
    radius = np.sqrt(1.0 - z * z)
    angle = np.pi * (3.0 - np.sqrt(5.0)) * indices
    surface = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    points = np.vstack((surface, np.zeros((1, 3))))
    faces = np.asarray(ConvexHull(surface).simplices, dtype=np.int64)
    centre = np.full((len(faces), 1), boundary_nodes, dtype=np.int64)
    return OperatorMesh(points, np.hstack((faces, centre)))


def _thin_film_mesh() -> OperatorMesh:
    nx = 6
    ny = 6
    points = np.asarray(
        [(x, y, 0.15 * z) for z in range(2) for y in range(ny) for x in range(nx)],
        dtype=np.float64,
    )

    def index(x: int, y: int, z: int) -> int:
        return (z * ny + y) * nx + x

    tetrahedra = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    simplices: list[list[int]] = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            corners = (
                index(x, y, 0),
                index(x + 1, y, 0),
                index(x, y + 1, 0),
                index(x + 1, y + 1, 0),
                index(x, y, 1),
                index(x + 1, y, 1),
                index(x, y + 1, 1),
                index(x + 1, y + 1, 1),
            )
            simplices.extend([[corners[i] for i in tetra] for tetra in tetrahedra])
    return OperatorMesh(points, np.asarray(simplices, dtype=np.int64))


def _disconnected_tetrahedra() -> OperatorMesh:
    first = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return OperatorMesh(
        points=np.vstack((first, first + np.asarray([4.0, 0.5, -0.25]))),
        simplices=np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
    )


def _random_mesh() -> OperatorMesh:
    points = np.random.default_rng(20260717).uniform(-1.0, 1.0, size=(28, 3))
    return OperatorMesh(points=points, simplices=np.asarray(Delaunay(points).simplices, dtype=np.int64))


@pytest.mark.parametrize(
    "mesh_factory",
    [
        _disconnected_tetrahedra,
        _random_mesh,
        _sphere_mesh,
        _thin_film_mesh,
    ],
    ids=("disconnected", "random", "sphere", "thin-film"),
)
def test_hierarchical_matvec_matches_dense_on_varied_geometries(
    mesh_factory: Callable[[], OperatorMesh],
) -> None:
    mesh = mesh_factory()
    _owners, raw_faces = nmag_accel.build_oriented_boundary_faces(
        mesh.points,
        mesh.simplices,
    )
    faces = np.asarray(raw_faces, dtype=np.int64)
    boundary_nodes = np.unique(faces)
    local_index = np.full(len(mesh.points), -1, dtype=np.int64)
    local_index[boundary_nodes] = np.arange(len(boundary_nodes), dtype=np.int64)
    hierarchical = nmag_accel.build_lindholm_hmatrix(
        mesh.points,
        mesh.simplices,
        faces,
        boundary_nodes,
        local_index,
        memory_budget_bytes=256 * 1024 * 1024,
    )
    dense = np.asarray(
        nmag_accel.build_lindholm_bem_matrix(
            mesh.points,
            mesh.simplices,
            faces,
            boundary_nodes,
            local_index,
        )
    )

    generator = np.random.default_rng(417)
    for _ in range(3):
        values = generator.standard_normal(len(boundary_nodes))
        first = np.asarray(hierarchical.matvec(values))
        second = np.asarray(hierarchical.matvec(values))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, dense @ values, rtol=1.0e-6, atol=1.0e-10)
