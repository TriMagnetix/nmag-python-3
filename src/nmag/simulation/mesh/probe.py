from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ...backends import (
    _load_rust_accelerator,
    _selected_probe_geometry_backend,
)
from ...demag import _tetrahedral_barycentric_coordinates
from ..support import _simulation_compatibility_binding


class SimulationMeshProbeMixin:
    if TYPE_CHECKING:
        _probe_geometry_cache_token: Any | None
        _probe_tetrahedral_cache: Any | None

        def __getattr__(self, name: str) -> Any: ...

    def _probe_is_within_mesh_bounds(self, pos: Sequence[float]) -> bool:
        probe = np.asarray(pos, dtype=float)
        if probe.shape != (3,):
            raise ValueError(f"Probe position must be a 3-vector, got shape {probe.shape}.")

        lower, upper, span = self._mesh_bounds()
        tolerance = 1.0e-12 * span
        return bool(np.all(probe >= lower - tolerance) and np.all(probe <= upper + tolerance))

    def _probe_is_legacy_null_vertex(self, probe: np.ndarray, points: np.ndarray) -> bool:
        """Legacy nmag can return None at exact boundary mesh vertex 0."""
        if len(points) == 0 or not np.array_equal(probe, points[0]):
            return False
        simplices = np.asarray(self._require_mesh().simplices, dtype=int)
        boundary_nodes = {
            point_index
            for _owner, face in self._boundary_faces_for_demag_mesh(points, simplices)
            for point_index in face
        }
        return 0 in boundary_nodes

    def _probe_tetrahedral_field(
        self,
        probe: np.ndarray,
        nodal_values: np.ndarray,
    ) -> list[float] | None:
        simplices, origins, inverse_matrices, lower, upper = self._probe_tetrahedral_geometry()
        if len(simplices) == 0:
            return None
        tolerance = 1.0e-12
        started = time.perf_counter()
        candidate_indices = np.flatnonzero(
            np.all(probe >= lower - tolerance, axis=1) & np.all(probe <= upper + tolerance, axis=1)
        )
        self._record_probe_timing(
            "tetrahedral_candidate_search",
            time.perf_counter() - started,
        )
        if len(candidate_indices) == 0:
            return None

        started = time.perf_counter()
        offsets = probe - origins[candidate_indices]
        local = np.einsum(
            "nij,nj->ni",
            inverse_matrices[candidate_indices],
            offsets,
        )
        barycentric = np.column_stack(
            [
                1.0 - np.sum(local, axis=1),
                local,
            ]
        )
        inside = np.all(barycentric >= -tolerance, axis=1) & np.all(
            barycentric <= 1.0 + tolerance,
            axis=1,
        )
        self._record_probe_timing(
            "tetrahedral_barycentric",
            time.perf_counter() - started,
        )
        if not np.any(inside):
            return None

        started = time.perf_counter()
        candidate_position = int(np.flatnonzero(inside)[0])
        cache_index = int(candidate_indices[candidate_position])
        simplex = simplices[cache_index]
        clipped = np.minimum(np.maximum(barycentric[candidate_position], 0.0), 1.0)
        result = (clipped @ nodal_values[simplex]).tolist()
        self._record_probe_timing(
            "tetrahedral_value_interpolation",
            time.perf_counter() - started,
        )
        return result

    def _probe_tetrahedral_geometry(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        total_started = time.perf_counter()
        token = self._mesh_geometry_token()
        if self._probe_geometry_cache_token == token and self._probe_tetrahedral_cache is not None:
            self._record_probe_timing("tetrahedral_geometry_cache_hit", 0.0)
            self._record_probe_timing(
                "tetrahedral_geometry",
                time.perf_counter() - total_started,
            )
            return self._probe_tetrahedral_cache

        build_started = time.perf_counter()
        backend = _selected_probe_geometry_backend(getattr(self, "config", None))
        self._probe_tetrahedral_cache = (
            self._probe_tetrahedral_geometry_rust()
            if backend == "rust"
            else self._probe_tetrahedral_geometry_python()
        )
        self._probe_geometry_cache_token = token
        self._record_probe_timing(
            f"tetrahedral_geometry_build:{backend}",
            time.perf_counter() - build_started,
        )
        self._record_probe_timing(
            "tetrahedral_geometry_build",
            time.perf_counter() - build_started,
        )
        self._record_probe_timing(
            "tetrahedral_geometry",
            time.perf_counter() - total_started,
        )
        return self._probe_tetrahedral_cache

    def _probe_tetrahedral_geometry_python(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = self._mesh_points()
        simplices = np.asarray(self._require_mesh().simplices, dtype=int)
        if simplices.size == 0:
            empty_simplices = np.zeros((0, 4), dtype=int)
            empty_vectors = np.zeros((0, 3), dtype=float)
            empty_matrices = np.zeros((0, 3, 3), dtype=float)
            return (
                empty_simplices,
                empty_vectors,
                empty_matrices,
                empty_vectors,
                empty_vectors,
            )

        valid_simplices: list[np.ndarray] = []
        origins: list[np.ndarray] = []
        inverse_matrices: list[np.ndarray] = []
        lower: list[np.ndarray] = []
        upper: list[np.ndarray] = []
        for simplex in simplices:
            tetrahedron = points[simplex]
            matrix = np.column_stack(
                [
                    tetrahedron[1] - tetrahedron[0],
                    tetrahedron[2] - tetrahedron[0],
                    tetrahedron[3] - tetrahedron[0],
                ]
            )
            try:
                inverse = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                continue
            valid_simplices.append(simplex)
            origins.append(tetrahedron[0])
            inverse_matrices.append(inverse)
            lower.append(np.min(tetrahedron, axis=0))
            upper.append(np.max(tetrahedron, axis=0))

        if not valid_simplices:
            empty_vectors = np.empty((0, 3), dtype=float)
            return (
                np.empty((0, 4), dtype=int),
                empty_vectors,
                np.empty((0, 3, 3), dtype=float),
                empty_vectors,
                empty_vectors,
            )
        return (
            np.asarray(valid_simplices, dtype=int),
            np.asarray(origins, dtype=float),
            np.asarray(inverse_matrices, dtype=float),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        )

    def _probe_tetrahedral_geometry_rust(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rust_accel = _simulation_compatibility_binding(
            "_load_rust_accelerator",
            _load_rust_accelerator,
        )("NmagConfig.accelerator['probe_geometry']")
        geometry = rust_accel.build_probe_tetrahedral_geometry(
            np.asarray(self._mesh_points(), dtype=np.float64),
            np.asarray(self._require_mesh().simplices, dtype=np.int64),
        )
        valid_simplices, origins, inverse_matrices, lower, upper = geometry
        return (
            np.asarray(valid_simplices, dtype=int),
            np.asarray(origins, dtype=float),
            np.asarray(inverse_matrices, dtype=float),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        )

    def _probe_tetrahedral_field_reference(
        self,
        probe: np.ndarray,
        nodal_values: np.ndarray,
    ) -> list[float] | None:
        """Scalar reference path for checking cached probe interpolation."""
        if self.mesh is None:
            raise RuntimeError("A mesh must be loaded before probing fields.")
        simplices = np.asarray(self._require_mesh().simplices, dtype=int)
        if simplices.size == 0:
            return None
        points = self._mesh_points()
        for simplex in simplices:
            barycentric = _tetrahedral_barycentric_coordinates(
                points[simplex],
                probe,
            )
            if barycentric is None:
                continue
            return (barycentric @ nodal_values[simplex]).tolist()
        return None
