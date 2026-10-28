from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..support import _MESH_EDGE_SET_THRESHOLD


class SimulationMeshGeometryMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def _invalidate_demag(self, *, clear_geometry: bool = False) -> None:
        self._demag_cache_token = None
        self._demag_dipoles = None
        self._demag_nodal_cache = None
        self._demag_cell_cache_token = None
        self._demag_cell_field_cache = None
        self._demag_aux_cache_token = None
        self._demag_phi_cache = None
        self._demag_rho_cache = None
        self._demag_volumes_cache = None
        self.last_demag_solve_diagnostics = {}
        self.last_demag_internal_vectors = {}
        self._exchange_cache_token = None
        self._exchange_nodal_cache = None
        self._anisotropy_fields_cache = None
        if clear_geometry:
            self._demag_geometry_cache_token = None
            self._demag_boundary_faces_cache = None
            self._demag_bem_cache = None
            self._demag_bem_env_cache = None
            self._last_bem_operator_stats = None
            self._demag_fem_geometry_cache = None
            self._demag_fem_env_cache = None
            self._demag_gauge_factorization_cache = None
            self._demag_dirichlet_factorization_cache = None
            self._demag_nodal_recovery_operator_cache = None
            self._demag_fem_divergence_operator_cache = None
            self._demag_cell_gradient_inputs_cache = None
            self._demag_ms_values_cache = None
            self._demag_volume_charge_scales_cache = None
            self._nodal_ms_values_cache = None
            self._mesh_points_cache = None
            self._mesh_simplices_cache = None
            self._mesh_bounds_cache = None
            self._mesh_edge_cache = None
            self._simplex_volume_cache = None
            self._volume_average_node_weights_cache = None
            self._incident_cell_volume_sums_cache = None
            self._exchange_operator_cache = None
            self._exchange_spectral_bound_cache = None
            self._nodal_material_coefficients_cache = None
            self._nodal_anisotropy_groups_cache = None
            self._anisotropy_fields_cache = None
            self._llg_affine_operator_cache = None
            self._probe_geometry_cache_token = None
            self._probe_tetrahedral_cache = None

    def _mesh_points(self) -> np.ndarray:
        mesh = self._require_mesh()
        raw_points = mesh.points
        token = self._mesh_geometry_token()
        points_len = len(raw_points)
        if self._mesh_points_cache is not None:
            cached_token, cached_raw, cached_len, cached_points = self._mesh_points_cache
            if (
                cached_token == token
                and cached_raw is raw_points
                and cached_len == points_len
            ):
                return cached_points

        points = np.array(raw_points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 3:
            raise NotImplementedError("The demag MVP currently supports only 3D meshes.")
        points.setflags(write=False)
        self._mesh_points_cache = (token, raw_points, points_len, points)
        return points

    def _mesh_simplices(self) -> np.ndarray:
        mesh = self._require_mesh()
        raw_simplices = mesh.simplices
        token = self._mesh_geometry_token()
        simplices_len = len(raw_simplices)
        if self._mesh_simplices_cache is not None:
            cached_token, cached_raw, cached_len, cached_simplices = self._mesh_simplices_cache
            if (
                cached_token == token
                and cached_raw is raw_simplices
                and cached_len == simplices_len
            ):
                return cached_simplices

        simplices = np.array(raw_simplices, dtype=int, copy=True)
        if simplices.size == 0 and simplices.ndim == 1:
            simplices = simplices.reshape((0, 4))
        simplices.setflags(write=False)
        self._mesh_simplices_cache = (token, raw_simplices, simplices_len, simplices)
        return simplices

    def _mesh_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        token = self._mesh_geometry_token()
        if self._mesh_bounds_cache is not None:
            cached_token, lower, upper, span = self._mesh_bounds_cache
            if cached_token == token:
                return lower, upper, span

        points = self._mesh_points()
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        span = np.maximum(upper - lower, np.ones(3))
        lower.setflags(write=False)
        upper.setflags(write=False)
        span.setflags(write=False)
        self._mesh_bounds_cache = (token, lower, upper, span)
        return lower, upper, span

    def _mesh_edges(self) -> np.ndarray:
        token = self._mesh_geometry_token()
        if self._mesh_edge_cache is not None:
            cached_token, cached_edges = self._mesh_edge_cache
            if cached_token == token:
                return cached_edges

        mesh = self._require_mesh()
        simplices = self._mesh_simplices()
        if simplices.ndim != 2 or simplices.shape[1] < 2 or len(simplices) == 0:
            edges = np.zeros((0, 2), dtype=int)
        else:
            local_pairs = np.asarray(
                [
                    (left, right)
                    for left in range(simplices.shape[1])
                    for right in range(left + 1, simplices.shape[1])
                ],
                dtype=int,
            )
            candidate_edge_count = len(simplices) * len(local_pairs)
            if candidate_edge_count <= _MESH_EDGE_SET_THRESHOLD:
                seen_edges: set[tuple[int, int]] = set()
                for simplex in simplices:
                    for left_position, left in enumerate(simplex):
                        left_index = int(left)
                        for right in simplex[left_position + 1 :]:
                            right_index = int(right)
                            if left_index <= right_index:
                                seen_edges.add((left_index, right_index))
                            else:
                                seen_edges.add((right_index, left_index))
                edges = np.asarray(sorted(seen_edges), dtype=int)
            else:
                expanded_edges = np.asarray(simplices[:, local_pairs], dtype=np.int64)
                flat_edges = expanded_edges.flatten()
                edges = np.column_stack((flat_edges[0::2], flat_edges[1::2]))
                edges.sort(axis=1)
                node_count = max(int(np.max(simplices)) + 1, len(mesh.points))
                edge_codes = edges[:, 0].astype(np.int64, copy=False) * node_count + edges[
                    :, 1
                ].astype(np.int64, copy=False)
                unique_codes = np.unique(edge_codes)
                edges = np.empty((len(unique_codes), 2), dtype=int)
                edges[:, 0] = unique_codes // node_count
                edges[:, 1] = unique_codes % node_count

        edges.setflags(write=False)
        self._mesh_edge_cache = (token, edges)
        return edges

    def _mesh_geometry_token(self) -> int:
        mesh = self._require_mesh()
        points = mesh.points
        simplices = mesh.simplices
        sample = (tuple(points[0]), tuple(points[-1])) if len(points) > 0 else ()
        return hash(
            (
                id(mesh.raw_mesh),
                id(points),
                id(simplices),
                len(points),
                len(simplices),
                sample,
            )
        )

    def _require_mesh(self) -> Any:
        if self.mesh is None:
            raise RuntimeError("A mesh must be loaded before using simulation fields.")
        return self.mesh

    def _ensure_demag_geometry_cache_token(self, token: int) -> None:
        if self._demag_geometry_cache_token == token:
            return
        self._demag_geometry_cache_token = token
        self._demag_boundary_faces_cache = None
        self._demag_bem_cache = None
        self._demag_bem_env_cache = None
        self._last_bem_operator_stats = None
        self._demag_fem_geometry_cache = None
        self._demag_fem_env_cache = None
        self._demag_gauge_factorization_cache = None
        self._demag_dirichlet_factorization_cache = None
        self._demag_nodal_recovery_operator_cache = None
        self._demag_fem_divergence_operator_cache = None
        self._demag_cell_gradient_inputs_cache = None
        self._demag_ms_values_cache = None
        self._demag_volume_charge_scales_cache = None
        self._nodal_ms_values_cache = None
        self._exchange_operator_cache = None
        self._nodal_material_coefficients_cache = None
        self._nodal_anisotropy_groups_cache = None
        self._anisotropy_fields_cache = None
