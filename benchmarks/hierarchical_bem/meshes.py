from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True, slots=True)
class OperatorMesh:
    points: np.ndarray
    simplices: np.ndarray


def fibonacci_sphere_mesh(boundary_nodes: int) -> OperatorMesh:
    """Create a convex star tetrahedralization with an exact boundary size."""

    if boundary_nodes < 4:
        raise ValueError("boundary_nodes must be at least four")
    indices = np.arange(boundary_nodes, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / boundary_nodes
    radius = np.sqrt(1.0 - z * z)
    angle = np.pi * (3.0 - np.sqrt(5.0)) * indices
    surface = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    points = np.vstack((surface, np.zeros((1, 3), dtype=np.float64)))
    faces = np.asarray(ConvexHull(surface).simplices, dtype=np.int64)
    centre = np.full((len(faces), 1), boundary_nodes, dtype=np.int64)
    return OperatorMesh(points=points, simplices=np.hstack((faces, centre)))


def structured_thin_film_mesh(target_boundary_nodes: int) -> OperatorMesh:
    """Create a two-layer structured film near the requested boundary size."""

    if target_boundary_nodes < 8:
        raise ValueError("target_boundary_nodes must be at least eight")
    nx = max(2, int(np.sqrt(target_boundary_nodes / 2.0)))
    ny = max(2, int(np.ceil(target_boundary_nodes / (2.0 * nx))))
    points = np.asarray(
        [(x, y, 0.15 * z) for z in range(2) for y in range(ny) for x in range(nx)],
        dtype=np.float64,
    )

    def point_index(x: int, y: int, z: int) -> int:
        return (z * ny + y) * nx + x

    local_tetrahedra = (
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
                point_index(x, y, 0),
                point_index(x + 1, y, 0),
                point_index(x, y + 1, 0),
                point_index(x + 1, y + 1, 0),
                point_index(x, y, 1),
                point_index(x + 1, y, 1),
                point_index(x, y + 1, 1),
                point_index(x + 1, y + 1, 1),
            )
            simplices.extend([[corners[index] for index in tetra] for tetra in local_tetrahedra])
    return OperatorMesh(points=points, simplices=np.asarray(simplices, dtype=np.int64))
