from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ....backends import _selected_demag_linear_solver_backend
from ....demag import _solve_linear_system
from ....demag.linear import (
    _SPARSE_FACTORIZATION_MAX_DENSITY,
    _SPARSE_FACTORIZATION_MIN_POINTS,
    ScipyLUFactorization,
    _allow_sparse_factorization,
    _factorize_scipy,
    _is_sparse_matrix,
    _solve_sparse_linear_system,
)


class SimulationDemagBemDirichletMixin:
    if TYPE_CHECKING:
        _demag_dirichlet_factorization_cache: (
            tuple[np.ndarray, np.ndarray, Any, ScipyLUFactorization] | None
        )

        def __getattr__(self, name: str) -> Any: ...

    def _dirichlet_extension_from_boundary(
        self,
        stiffness: np.ndarray,
        boundary_nodes: np.ndarray,
        boundary_values: np.ndarray,
    ) -> np.ndarray:
        point_count = stiffness.shape[0]
        is_sparse = _is_sparse_matrix(stiffness)
        use_scipy_cache = (
            not is_sparse and _selected_demag_linear_solver_backend(point_count) == "scipy"
        )
        cached = self._demag_dirichlet_factorization_cache if use_scipy_cache else None

        if cached is not None:
            interior_nodes, cached_boundary, boundary_stiffness, factorization = cached
            if cached_boundary is not boundary_nodes and not np.array_equal(
                cached_boundary, boundary_nodes
            ):
                raise RuntimeError("Cached demag boundary partition does not match the mesh.")
            with self._record_active_subfield_array_timing_block(
                "demag_auxiliary:dirichlet_extension:index",
            ):
                phi = np.zeros(point_count, dtype=float)
                phi[boundary_nodes] = boundary_values
            self._record_active_subfield_array_timing(
                "demag_auxiliary:dirichlet_extension:slice",
                0.0,
            )
            with self._record_active_subfield_array_timing_block(
                "demag_auxiliary:dirichlet_extension:rhs",
            ):
                interior_rhs = -boundary_stiffness @ boundary_values
            with self._record_active_subfield_array_timing_block(
                "demag_auxiliary:dirichlet_extension:solve",
            ):
                phi[interior_nodes] = factorization.solve(interior_rhs)
            return phi

        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:dirichlet_extension:index",
        ):
            phi = np.zeros(point_count, dtype=float)
            phi[boundary_nodes] = boundary_values
            interior_mask = np.ones(point_count, dtype=bool)
            interior_mask[boundary_nodes] = False
            interior_nodes = np.flatnonzero(interior_mask)
        if len(interior_nodes) == 0:
            self._record_active_subfield_array_timing(
                "demag_auxiliary:dirichlet_extension:no_interior",
                0.0,
            )
            return phi

        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:dirichlet_extension:slice",
        ):
            if is_sparse:
                interior_stiffness = stiffness[interior_nodes][:, interior_nodes]
                boundary_stiffness = stiffness[interior_nodes][:, boundary_nodes]
            else:
                interior_stiffness = stiffness[np.ix_(interior_nodes, interior_nodes)]
                boundary_stiffness = stiffness[np.ix_(interior_nodes, boundary_nodes)]
        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:dirichlet_extension:rhs",
        ):
            interior_rhs = -boundary_stiffness @ boundary_values
        if _is_sparse_matrix(interior_stiffness):
            phi[interior_nodes] = _solve_sparse_linear_system(
                interior_stiffness,
                interior_rhs,
                record_timing=self._record_active_subfield_array_timing,
                timing_prefix="demag_auxiliary:dirichlet_extension",
            )
        elif use_scipy_cache:
            factorization = _factorize_scipy(interior_stiffness)
            cached_boundary_stiffness: Any = boundary_stiffness
            if _allow_sparse_factorization() and point_count >= _SPARSE_FACTORIZATION_MIN_POINTS:
                from scipy import sparse

                csr_boundary = sparse.csr_matrix(boundary_stiffness)
                if csr_boundary.nnz <= _SPARSE_FACTORIZATION_MAX_DENSITY * boundary_stiffness.size:
                    cached_boundary_stiffness = csr_boundary
            self._demag_dirichlet_factorization_cache = (
                interior_nodes,
                boundary_nodes,
                cached_boundary_stiffness,
                factorization,
            )
            with self._record_active_subfield_array_timing_block(
                "demag_auxiliary:dirichlet_extension:solve",
            ):
                phi[interior_nodes] = factorization.solve(interior_rhs)
        else:
            phi[interior_nodes] = _solve_linear_system(
                interior_stiffness,
                interior_rhs,
                record_timing=self._record_active_subfield_array_timing,
                timing_prefix="demag_auxiliary:dirichlet_extension",
            )
        return phi
