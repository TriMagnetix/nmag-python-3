from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ....backends import (
    _load_rust_accelerator,
    _selected_demag_bem_storage_backend,
    _selected_lindholm_bem_backend,
)
from ....demag import (
    _boundary_node_solid_angles,
    _build_lindholm_bem_matrix_fast,
    _lindholm_triangle_contributions,
    _precompute_lindholm_face_geometry,
)
from ....demag.bem_operator import MatrixFreeLindholmBemOperator
from ...support import _simulation_compatibility_binding
from .diagnostics import bem_operator_stats, log_bem_operator_selection
from .hierarchical import build_hierarchical_lindholm_operator


class SimulationDemagBemOperatorMixin:
    if TYPE_CHECKING:
        _demag_bem_cache: Any | None

        def __getattr__(self, name: str) -> Any: ...

    def _lindholm_bem_for_demag_mesh(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        boundary_faces: list[tuple[int, tuple[int, int, int]]],
    ) -> tuple[np.ndarray, Any]:
        cache_started = time.perf_counter()
        token = self._mesh_geometry_token()
        self._ensure_demag_geometry_cache_token(token)
        boundary_node_count = len({node for _, face in boundary_faces for node in face})
        config = getattr(self, "config", None)
        storage_backend = _selected_demag_bem_storage_backend(boundary_node_count, config)
        if self._demag_bem_cache is not None:
            cached_backend, boundary_nodes, operator, stats = self._demag_bem_cache
            if cached_backend == storage_backend:
                cache_kind = "matrix" if storage_backend == "dense" else "operator"
                self._last_bem_operator_stats = stats
                self._record_active_subfield_array_timing(
                    f"demag_auxiliary:bem_{cache_kind}_cache_hit",
                    time.perf_counter() - cache_started,
                )
                return boundary_nodes, operator

        fallback_reason: str | None = None
        effective_backend = storage_backend
        if storage_backend == "hierarchical":
            boundary_nodes, bem, fallback_reason = build_hierarchical_lindholm_operator(
                self,
                points,
                simplices,
                boundary_faces,
            )
            if fallback_reason is not None:
                effective_backend = "matrix-free"
        elif storage_backend == "matrix-free":
            boundary_nodes, bem = self._build_lindholm_bem_operator(
                points,
                simplices,
                boundary_faces,
            )
        else:
            boundary_nodes, bem = self._build_lindholm_bem_matrix(
                points,
                simplices,
                boundary_faces,
            )
        elapsed = time.perf_counter() - cache_started
        requested_backend = getattr(config, "demag_bem_storage", storage_backend)
        stats = bem_operator_stats(
            requested_backend=requested_backend,
            effective_backend=effective_backend,
            fallback_reason=fallback_reason,
            operator=bem,
            boundary_node_count=len(boundary_nodes),
            boundary_faces=len(boundary_faces),
            setup_seconds=elapsed,
        )
        log_bem_operator_selection(stats, storage_backend)
        self._last_bem_operator_stats = stats
        self._demag_boundary_faces_cache = boundary_faces
        self._demag_bem_cache = (storage_backend, boundary_nodes, bem, stats)
        return boundary_nodes, cast(Any, bem)

    def _build_lindholm_bem_operator(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        boundary_faces: list[tuple[int, tuple[int, int, int]]],
    ) -> tuple[np.ndarray, MatrixFreeLindholmBemOperator]:
        boundary_nodes, local_index_by_point, face_nodes = self._lindholm_bem_boundary_index(
            points,
            boundary_faces,
        )
        (
            face_points,
            zeta_vectors,
            eta_vectors,
            edge_lengths,
            corner_cosines,
            denominator_factors,
        ) = _precompute_lindholm_face_geometry(points, face_nodes)
        rust_accelerator = None
        if _selected_lindholm_bem_backend(getattr(self, "config", None)) == "rust":
            rust_accelerator = _simulation_compatibility_binding(
                "_load_rust_accelerator",
                _load_rust_accelerator,
            )()
            if not hasattr(rust_accelerator, "apply_lindholm_bem_matrix_free"):
                raise RuntimeError(
                    "The installed nmag_accel does not provide matrix-free Lindholm BEM; "
                    "rebuild it with ./scripts/build-accelerator.sh --release."
                )
        return boundary_nodes, MatrixFreeLindholmBemOperator(
            points=np.ascontiguousarray(points, dtype=np.float64),
            simplices=np.ascontiguousarray(simplices, dtype=np.int64),
            boundary_faces=np.ascontiguousarray(face_nodes, dtype=np.int64),
            face_points=np.ascontiguousarray(face_points, dtype=np.float64),
            zeta_vectors=np.ascontiguousarray(zeta_vectors, dtype=np.float64),
            eta_vectors=np.ascontiguousarray(eta_vectors, dtype=np.float64),
            edge_lengths=np.ascontiguousarray(edge_lengths, dtype=np.float64),
            corner_cosines=np.ascontiguousarray(corner_cosines, dtype=np.float64),
            denominator_factors=np.ascontiguousarray(denominator_factors, dtype=np.float64),
            boundary_nodes=np.ascontiguousarray(boundary_nodes, dtype=np.int64),
            local_index_by_point=np.ascontiguousarray(local_index_by_point, dtype=np.int64),
            solid_angles=np.ascontiguousarray(
                _boundary_node_solid_angles(points, simplices),
                dtype=np.float64,
            ),
            rust_accelerator=rust_accelerator,
        )

    def _build_lindholm_bem_matrix(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        boundary_faces: list[tuple[int, tuple[int, int, int]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        backend = _selected_lindholm_bem_backend(getattr(self, "config", None))
        if backend == "python":
            return self._build_lindholm_bem_matrix_python(
                points,
                simplices,
                boundary_faces,
            )
        if backend == "rust":
            return self._build_lindholm_bem_matrix_rust(
                points,
                simplices,
                boundary_faces,
            )
        return self._build_lindholm_bem_matrix_numba(
            points,
            simplices,
            boundary_faces,
        )

    def _lindholm_bem_boundary_index(
        self,
        points: np.ndarray,
        boundary_faces: list[tuple[int, tuple[int, int, int]]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boundary_nodes = np.asarray(
            sorted({point_index for _, face in boundary_faces for point_index in face}),
            dtype=np.int64,
        )
        local_index_by_point = np.full(len(points), -1, dtype=np.int64)
        local_index_by_point[boundary_nodes] = np.arange(len(boundary_nodes), dtype=np.int64)
        face_nodes = np.asarray([face for _, face in boundary_faces], dtype=np.int64)
        return boundary_nodes, local_index_by_point, face_nodes

    def _build_lindholm_bem_matrix_python(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        boundary_faces: list[tuple[int, tuple[int, int, int]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:lindholm_bem:index",
        ):
            boundary_nodes, local_index_by_point, _face_nodes = self._lindholm_bem_boundary_index(
                points,
                boundary_faces,
            )

        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:lindholm_bem:python",
        ):
            bem = np.zeros((len(boundary_nodes), len(boundary_nodes)), dtype=float)
            solid_angles = _boundary_node_solid_angles(points, simplices)

            for row, observer_index in enumerate(boundary_nodes):
                observer = points[int(observer_index)]
                for _, face in boundary_faces:
                    p0, p1, p2 = points[list(face)]
                    normal = np.cross(p1 - p0, p2 - p0)
                    normal_length = float(np.linalg.norm(normal))
                    if normal_length == 0.0:
                        continue
                    contributions = _lindholm_triangle_contributions(
                        observer,
                        p0,
                        p1,
                        p2,
                        normal / normal_length,
                    )
                    for point_index, contribution in zip(face, contributions):  # noqa: B905
                        bem[row, int(local_index_by_point[int(point_index)])] += contribution
                bem[row, row] += solid_angles[int(observer_index)] / (4.0 * np.pi) - 1.0
        return boundary_nodes, bem

    def _build_lindholm_bem_matrix_numba(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        boundary_faces: list[tuple[int, tuple[int, int, int]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:lindholm_bem:index",
        ):
            boundary_nodes, local_index_by_point, face_nodes = self._lindholm_bem_boundary_index(
                points,
                boundary_faces,
            )

        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:lindholm_bem:numba",
        ):
            (
                face_points,
                zeta_vectors,
                eta_vectors,
                edge_lengths,
                corner_cosines,
                denominator_factors,
            ) = _precompute_lindholm_face_geometry(np.asarray(points, dtype=np.float64), face_nodes)
            bem = _build_lindholm_bem_matrix_fast(
                np.asarray(points, dtype=np.float64),
                np.asarray(simplices, dtype=np.int64),
                face_nodes,
                face_points,
                zeta_vectors,
                eta_vectors,
                edge_lengths,
                corner_cosines,
                denominator_factors,
                np.asarray(boundary_nodes, dtype=np.int64),
                local_index_by_point,
            )
        return boundary_nodes, bem

    def _build_lindholm_bem_matrix_rust(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        boundary_faces: list[tuple[int, tuple[int, int, int]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:lindholm_bem:index",
        ):
            boundary_nodes, local_index_by_point, face_nodes = self._lindholm_bem_boundary_index(
                points,
                boundary_faces,
            )

        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:lindholm_bem:rust",
        ):
            rust_accel = _simulation_compatibility_binding(
                "_load_rust_accelerator",
                _load_rust_accelerator,
            )()
            bem = np.asarray(
                rust_accel.build_lindholm_bem_matrix(
                    np.asarray(points, dtype=np.float64),
                    np.asarray(simplices, dtype=np.int64),
                    face_nodes,
                    np.asarray(boundary_nodes, dtype=np.int64),
                    local_index_by_point,
                ),
                dtype=float,
            )
        return boundary_nodes, bem
