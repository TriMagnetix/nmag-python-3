from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ....backends import (
    _load_rust_accelerator,
    _selected_demag_boundary_face_backend,
    _selected_demag_fem_geometry_backend,
    _selected_demag_fem_matrix_backend,
)
from ....demag import (
    _oriented_boundary_faces,
    _validate_dense_demag_size,
    _validate_tetrahedral_cells,
)
from ...support import _simulation_compatibility_binding


class SimulationDemagFemGeometryMixin:
    if TYPE_CHECKING:
        _demag_fem_geometry_cache: Any | None
        _demag_boundary_faces_cache: Any | None

        def __getattr__(self, name: str) -> Any: ...

    def _demag_fem_geometry_for_mesh(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        token = self._mesh_geometry_token()
        self._ensure_demag_geometry_cache_token(token)
        matrix_backend = _selected_demag_fem_matrix_backend(len(points))
        if self._demag_fem_geometry_cache is not None:
            cached_backend, stiffness, gradients, volumes = self._demag_fem_geometry_cache
            if cached_backend == matrix_backend:
                return stiffness, gradients, volumes

        if matrix_backend == "sparse":
            stiffness, gradients, volumes = self._demag_fem_geometry_for_mesh_sparse(
                points,
                simplices,
            )
            self._demag_fem_geometry_cache = (
                matrix_backend,
                stiffness,
                gradients,
                volumes,
            )
            return stiffness, gradients, volumes

        backend = _selected_demag_fem_geometry_backend(
            len(simplices), getattr(self, "config", None)
        )
        with self._record_active_subfield_array_timing_block(
            f"demag_auxiliary:assemble_fem:geometry:{backend}",
        ):
            if backend == "rust":
                geometry = self._demag_fem_geometry_for_mesh_rust(
                    points,
                    simplices,
                )
            else:
                geometry = self._demag_fem_geometry_for_mesh_python(
                    points,
                    simplices,
                )
        stiffness, gradients, volumes = geometry
        self._demag_fem_geometry_cache = (matrix_backend, stiffness, gradients, volumes)
        return stiffness, gradients, volumes

    def _demag_fem_geometry_for_mesh_sparse(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        from scipy import sparse

        _validate_tetrahedral_cells(points, simplices)
        point_count = len(points)
        if len(simplices) == 0:
            return (
                sparse.csr_matrix((point_count, point_count), dtype=float),
                np.empty((0, 4, 3), dtype=float),
                np.empty(0, dtype=float),
            )

        cell_points = points[simplices]
        matrices = np.concatenate(
            [np.ones((len(simplices), 4, 1), dtype=float), cell_points],
            axis=2,
        )
        inverses = np.linalg.inv(matrices)
        gradients = np.transpose(inverses[:, 1:, :], (0, 2, 1))
        volumes = np.abs(np.linalg.det(cell_points[:, 1:] - cell_points[:, [0]])) / 6.0
        local_stiffness = volumes[:, np.newaxis, np.newaxis] * np.einsum(
            "cik,cjk->cij",
            gradients,
            gradients,
        )
        rows = np.broadcast_to(simplices[:, :, np.newaxis], local_stiffness.shape).ravel()
        columns = np.broadcast_to(simplices[:, np.newaxis, :], local_stiffness.shape).ravel()
        stiffness = sparse.coo_matrix(
            (local_stiffness.ravel(), (rows, columns)),
            shape=(point_count, point_count),
        ).tocsr()
        stiffness.sum_duplicates()
        return stiffness, gradients, volumes

    def _demag_fem_geometry_for_mesh_python(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_count = len(points)
        _validate_dense_demag_size(point_count)
        _validate_tetrahedral_cells(points, simplices)
        stiffness = np.zeros((point_count, point_count), dtype=float)
        gradients_by_cell = np.zeros((len(simplices), 4, 3), dtype=float)
        volumes = np.zeros(len(simplices), dtype=float)

        if len(simplices) == 0:
            return stiffness, gradients_by_cell, volumes

        cell_points = points[simplices]
        matrices = np.concatenate(
            [
                np.ones((len(simplices), 4, 1), dtype=float),
                cell_points,
            ],
            axis=2,
        )
        try:
            inverses = np.linalg.inv(matrices)
        except np.linalg.LinAlgError:
            return self._demag_fem_geometry_for_mesh_loop(
                points,
                simplices,
                stiffness,
                gradients_by_cell,
                volumes,
            )

        gradients_by_cell[:] = np.transpose(inverses[:, 1:, :], (0, 2, 1))
        volumes[:] = np.abs(np.linalg.det(cell_points[:, 1:] - cell_points[:, [0]])) / 6.0
        positive = volumes > 0.0
        if np.any(positive):
            local_stiffness = volumes[positive, np.newaxis, np.newaxis] * np.einsum(
                "cik,cjk->cij",
                gradients_by_cell[positive],
                gradients_by_cell[positive],
            )
            positive_simplices = simplices[positive]
            np.add.at(
                stiffness,
                (
                    positive_simplices[:, :, np.newaxis],
                    positive_simplices[:, np.newaxis, :],
                ),
                local_stiffness,
            )

        return stiffness, gradients_by_cell, volumes

    def _demag_fem_geometry_for_mesh_rust(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _validate_dense_demag_size(len(points))
        _validate_tetrahedral_cells(points, simplices)
        rust_accel = _simulation_compatibility_binding(
            "_load_rust_accelerator",
            _load_rust_accelerator,
        )("NmagConfig.accelerator['fem_geometry']")
        stiffness, gradients_by_cell, volumes = rust_accel.build_demag_fem_geometry(
            np.asarray(points, dtype=np.float64),
            np.asarray(simplices, dtype=np.int64),
        )
        return (
            np.asarray(stiffness, dtype=float),
            np.asarray(gradients_by_cell, dtype=float),
            np.asarray(volumes, dtype=float),
        )

    def _demag_fem_geometry_for_mesh_loop(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        stiffness: np.ndarray,
        gradients_by_cell: np.ndarray,
        volumes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for cell_index, simplex in enumerate(simplices):
            cell_points = points[simplex]
            matrix = np.column_stack([np.ones(4), cell_points])
            try:
                inverse = np.asarray(np.linalg.inv(matrix), dtype=float)
            except np.linalg.LinAlgError as error:
                raise ValueError(f"Demag FEM cell {cell_index} is singular.") from error
            gradients = np.transpose(inverse[1:, :])
            volume = abs(float(np.linalg.det(cell_points[1:] - cell_points[[0]]))) / 6.0
            if volume <= 0.0:
                raise ValueError(f"Demag FEM cell {cell_index} has zero volume.")
            volumes[cell_index] = volume
            gradients_by_cell[cell_index] = gradients
            stiffness[np.ix_(simplex, simplex)] += volume * (gradients @ np.transpose(gradients))

        return stiffness, gradients_by_cell, volumes

    def _boundary_faces_for_demag_mesh(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
    ) -> list[tuple[int, tuple[int, int, int]]]:
        token = self._mesh_geometry_token()
        self._ensure_demag_geometry_cache_token(token)
        if self._demag_boundary_faces_cache is not None:
            return self._demag_boundary_faces_cache

        backend = _selected_demag_boundary_face_backend(getattr(self, "config", None))
        with self._record_active_subfield_array_timing_block(
            f"demag_auxiliary:boundary_faces:{backend}",
        ):
            if backend == "rust":
                boundary_faces = self._boundary_faces_for_demag_mesh_rust(
                    points,
                    simplices,
                )
            else:
                boundary_faces = _simulation_compatibility_binding(
                    "_oriented_boundary_faces",
                    _oriented_boundary_faces,
                )(points, simplices)
        self._demag_boundary_faces_cache = boundary_faces
        self._demag_bem_cache = None
        return boundary_faces

    def _boundary_faces_for_demag_mesh_rust(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
    ) -> list[tuple[int, tuple[int, int, int]]]:
        rust_accel = _simulation_compatibility_binding(
            "_load_rust_accelerator",
            _load_rust_accelerator,
        )("NmagConfig.accelerator['boundary_faces']")
        owners, faces = rust_accel.build_oriented_boundary_faces(
            np.asarray(points, dtype=np.float64),
            np.asarray(simplices, dtype=np.int64),
        )
        owner_array = np.asarray(owners, dtype=int).flatten()
        face_values = np.asarray(faces, dtype=int).flatten()
        face_array = np.column_stack((face_values[0::3], face_values[1::3], face_values[2::3]))
        return [
            (int(owner), (int(face[0]), int(face[1]), int(face[2])))
            for owner, face in zip(owner_array, face_array, strict=True)
        ]
