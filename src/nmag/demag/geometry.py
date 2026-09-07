from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

TETRA_FACE_VERTICES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def _validate_tetrahedral_cells(points: FloatArray, simplices: IntArray) -> None:
    if len(simplices) == 0:
        return
    cell_points = points[simplices]
    edges = cell_points[:, 1:] - cell_points[:, [0]]
    length_scales = np.max(np.linalg.norm(edges, axis=2), axis=1)
    determinants = np.abs(np.linalg.det(edges))
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_determinants = determinants / (length_scales**3)
    invalid = (
        ~np.isfinite(normalized_determinants)
        | (length_scales <= 0.0)
        | (normalized_determinants <= 1.0e-12)
    )
    if np.any(invalid):
        indices = np.flatnonzero(invalid).tolist()
        raise ValueError(
            f"Demag FEM requires non-degenerate tetrahedra; invalid cell indices: {indices}."
        )


def _simplex_volumes(points: FloatArray, simplices: IntArray) -> FloatArray:
    """Return 3D simplex volumes for tetrahedral mesh cells."""
    if len(simplices) == 0:
        return np.empty(0, dtype=float)
    matrices = points[simplices[:, 1:]] - points[simplices[:, [0]]]
    return np.abs(np.linalg.det(matrices)) / 6.0


def _mean_edge_length(points: FloatArray, simplices: IntArray) -> float:
    lengths: list[float] = []
    for simplex in simplices:
        for left_index, left in enumerate(simplex):
            for right in simplex[left_index + 1 :]:
                lengths.append(float(np.linalg.norm(points[int(left)] - points[int(right)])))
    if not lengths:
        return 1.0
    return max(float(np.mean(lengths)), np.finfo(float).eps)


def _tetrahedral_barycentric_coordinates(
    tetrahedron: FloatArray,
    point: FloatArray,
    *,
    tolerance: float = 1.0e-12,
) -> FloatArray | None:
    matrix = np.column_stack(
        [
            tetrahedron[1] - tetrahedron[0],
            tetrahedron[2] - tetrahedron[0],
            tetrahedron[3] - tetrahedron[0],
        ]
    )
    try:
        local = np.asarray(np.linalg.solve(matrix, point - tetrahedron[0]), dtype=float)
    except np.linalg.LinAlgError:
        return None
    barycentric = np.asarray(
        [1.0 - float(np.sum(local)), local[0], local[1], local[2]],
        dtype=float,
    )
    if np.all(barycentric >= -tolerance) and np.all(barycentric <= 1.0 + tolerance):
        return np.minimum(np.maximum(barycentric, 0.0), 1.0)
    return None


def _oriented_boundary_faces(
    points: FloatArray, simplices: IntArray
) -> list[tuple[int, tuple[int, int, int]]]:
    if len(simplices) == 0:
        return []

    cell_indices = np.repeat(
        np.arange(len(simplices), dtype=np.int64),
        len(TETRA_FACE_VERTICES),
    )
    face_nodes: IntArray = np.asarray(
        [
            [int(simplex[first]), int(simplex[second]), int(simplex[third])]
            for simplex in simplices
            for first, second, third in TETRA_FACE_VERTICES
        ],
        dtype=np.int64,
    )
    face_keys = np.sort(face_nodes, axis=1)
    unique_face_keys, inverse, counts = np.unique(
        face_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    non_manifold_faces = unique_face_keys[counts > 2]
    if len(non_manifold_faces):
        raise ValueError(
            "Non-manifold mesh: at least one face is shared by more than two tetrahedra."
        )
    boundary_positions = np.flatnonzero(counts[inverse] == 1)
    if len(boundary_positions) == 0:
        return []

    boundary_cells = cell_indices[boundary_positions]
    boundary_face_nodes: IntArray = np.asarray(
        [face_nodes[int(position)] for position in boundary_positions],
        dtype=np.int64,
    )
    face_points = points[boundary_face_nodes]
    cell_centers = np.mean(points[simplices[boundary_cells]], axis=1)
    face_centers = np.mean(face_points, axis=1)
    normals = np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0])
    points_inward = np.einsum("ij,ij->i", normals, cell_centers - face_centers) > 0.0
    for position in np.flatnonzero(points_inward):
        position_index = int(position)
        second = int(boundary_face_nodes[position_index, 1])
        boundary_face_nodes[position_index, 1] = boundary_face_nodes[position_index, 2]
        boundary_face_nodes[position_index, 2] = second

    return [
        (
            int(cell_index),
            (int(face[0]), int(face[1]), int(face[2])),
        )
        for cell_index, face in zip(boundary_cells, boundary_face_nodes, strict=True)
    ]
