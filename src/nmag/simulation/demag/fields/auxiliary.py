from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ....backends import (
    _demag_internal_trace_enabled,
    _selected_demag_linear_solver_backend,
)
from ....demag import _demag_solve_diagnostics, _solve_gauge_fixed
from ....demag.linear import _factorize_gauge_fixed, _is_sparse_matrix, _solve_sparse_gauge_fixed
from ...support import LEGACY_RHO_BOXED_VOLUME_M3


class SimulationDemagAuxiliaryMixin:
    if TYPE_CHECKING:
        _demag_phi_cache: Any | None
        _demag_rho_cache: Any | None
        _demag_volumes_cache: Any | None
        _demag_gauge_factorization_cache: Any | None

        def __getattr__(self, name: str) -> Any: ...

    def _get_demag_auxiliary_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_started = time.perf_counter()
        token = self._demag_token()
        if (
            self._demag_aux_cache_token == token
            and self._demag_phi_cache is not None
            and self._demag_rho_cache is not None
            and self._demag_volumes_cache is not None
        ):
            self._record_active_subfield_array_timing(
                "demag_auxiliary:cache_hit",
                time.perf_counter() - cache_started,
            )
            return self._demag_phi_cache, self._demag_rho_cache, self._demag_volumes_cache

        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:mesh_inputs",
        ):
            points = self._mesh_points()
            simplices = np.asarray(self._require_mesh().simplices, dtype=int)
        if simplices.size == 0:
            phi = np.zeros(len(points), dtype=float)
            rho = np.zeros(len(points), dtype=float)
            volumes = np.zeros(0, dtype=float)
        else:
            if simplices.ndim != 2 or simplices.shape[1] != 4:
                raise NotImplementedError("The demag MVP currently supports tetrahedral 3D meshes.")
            with self._record_active_subfield_array_timing_block(
                "demag_auxiliary:boundary_faces",
            ):
                boundary_faces = self._boundary_faces_for_demag_mesh(points, simplices)
            if not boundary_faces:
                phi = np.zeros(len(points), dtype=float)
                rho = np.zeros(len(points), dtype=float)
                volumes = np.zeros(len(simplices), dtype=float)
            else:
                with self._record_active_subfield_array_timing_block(
                    "demag_auxiliary:material_inputs",
                ):
                    m = np.asarray(self._fields["m"], dtype=float)
                    regions = list(self._require_mesh().regions or [1] * len(simplices))
                    ms_values = self._simplex_material_ms_values(regions)
                    volume_charge_scales = self._simplex_volume_charge_scales(regions)
                with self._record_active_subfield_array_timing_block(
                    "demag_auxiliary:fem_bem_total",
                ):
                    phi, rho, volumes = self._fem_bem_auxiliary_at_nodes(
                        points,
                        simplices,
                        boundary_faces,
                        m,
                        ms_values,
                        volume_charge_scales,
                    )

        self._demag_aux_cache_token = token
        self._demag_phi_cache = phi
        self._demag_rho_cache = rho
        self._demag_volumes_cache = volumes
        return phi, rho, volumes

    def _fem_bem_auxiliary_at_nodes(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        boundary_faces: list[tuple[int, tuple[int, int, int]]],
        m: np.ndarray,
        ms_values: np.ndarray,
        volume_charge_scales: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:assemble_fem",
        ):
            stiffness, divergence, volumes = self._assemble_demag_fem_system(
                points,
                simplices,
                m,
                ms_values,
                volume_charge_scales,
            )
        with self._record_active_subfield_array_timing_block("demag_auxiliary:rho_box"):
            rho = self._scalar_cofield_to_legacy_boxed_field(
                divergence,
                len(points),
            )
        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:gauge_solve",
        ):
            if _is_sparse_matrix(stiffness):
                phi1 = _solve_sparse_gauge_fixed(
                    stiffness,
                    divergence,
                    record_timing=self._record_active_subfield_array_timing,
                    timing_prefix="demag_auxiliary:gauge_solve",
                )
            elif _selected_demag_linear_solver_backend(stiffness.shape[0]) == "scipy":
                if self._demag_gauge_factorization_cache is None:
                    self._demag_gauge_factorization_cache = _factorize_gauge_fixed(stiffness)
                augmented_rhs = np.concatenate([divergence, [0.0]])
                phi1 = self._demag_gauge_factorization_cache.solve(augmented_rhs)[:-1]
            else:
                phi1 = _solve_gauge_fixed(
                    stiffness,
                    divergence,
                    record_timing=self._record_active_subfield_array_timing,
                    timing_prefix="demag_auxiliary:gauge_solve",
                )
        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:lindholm_bem",
        ):
            boundary_nodes, bem = self._lindholm_bem_for_demag_mesh(
                points,
                simplices,
                boundary_faces,
            )
        phi2 = np.zeros(len(points), dtype=float)
        phi2_boundary = np.zeros(len(boundary_nodes), dtype=float)
        if len(boundary_nodes) > 0:
            with self._record_active_subfield_array_timing_block(
                "demag_auxiliary:bem_multiply",
            ):
                phi2_boundary = bem @ phi1[boundary_nodes]
            with self._record_active_subfield_array_timing_block(
                "demag_auxiliary:dirichlet_extension",
            ):
                phi2 = self._dirichlet_extension_from_boundary(
                    stiffness,
                    boundary_nodes,
                    phi2_boundary,
                )
        self.last_demag_solve_diagnostics = _demag_solve_diagnostics(
            stiffness,
            divergence,
            boundary_nodes,
            phi1,
            phi2_boundary,
            phi2,
            phi1 + phi2,
        )
        if _demag_internal_trace_enabled():
            self.last_demag_internal_vectors = {
                "rho": np.array(rho, copy=True),
                "phi1": np.array(phi1, copy=True),
                "phi2": np.array(phi2, copy=True),
                "phi": np.array(phi1 + phi2, copy=True),
            }
        return phi1 + phi2, rho, volumes

    def _scalar_cofield_to_legacy_boxed_field(
        self,
        cofield: np.ndarray,
        point_count: int,
    ) -> np.ndarray:
        if cofield.size == 0:
            return np.zeros(point_count, dtype=float)

        nodal = np.zeros(point_count, dtype=float)
        # Legacy Nmag's simulation unit system always boxes rho with the
        # fixed 1 nm base volume, rather than the loaded mesh's unit_length.
        nodal[: len(cofield)] = cofield / LEGACY_RHO_BOXED_VOLUME_M3
        return nodal
