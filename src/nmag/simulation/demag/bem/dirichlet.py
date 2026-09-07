from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ....backends import _selected_demag_linear_solver_backend
from ....demag import _solve_linear_system
from ....demag.linear import _factorize_scipy, _is_sparse_matrix, _solve_sparse_linear_system


class SimulationDemagBemDirichletMixin:
    if TYPE_CHECKING:
        _demag_dirichlet_factorization_cache: Any | None

        def __getattr__(self, name: str) -> Any: ...

    def _dirichlet_extension_from_boundary(
        self,
        stiffness: np.ndarray,
        boundary_nodes: np.ndarray,
        boundary_values: np.ndarray,
    ) -> np.ndarray:
        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:dirichlet_extension:index",
        ):
            phi = np.zeros(stiffness.shape[0], dtype=float)
            phi[boundary_nodes] = boundary_values
            boundary_set = set(int(index) for index in boundary_nodes)
            interior_nodes = np.asarray(
                [index for index in range(stiffness.shape[0]) if index not in boundary_set],
                dtype=int,
            )
        if len(interior_nodes) == 0:
            self._record_active_subfield_array_timing(
                "demag_auxiliary:dirichlet_extension:no_interior",
                0.0,
            )
            return phi

        with self._record_active_subfield_array_timing_block(
            "demag_auxiliary:dirichlet_extension:slice",
        ):
            if _is_sparse_matrix(stiffness):
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
        elif _selected_demag_linear_solver_backend(stiffness.shape[0]) == "scipy":
            cached = self._demag_dirichlet_factorization_cache
            if cached is None:
                factorization = _factorize_scipy(interior_stiffness)
                self._demag_dirichlet_factorization_cache = (
                    interior_nodes,
                    boundary_nodes,
                    factorization,
                )
            else:
                cached_interior, cached_boundary, factorization = cached
                if not (
                    np.array_equal(cached_interior, interior_nodes)
                    and np.array_equal(cached_boundary, boundary_nodes)
                ):
                    raise RuntimeError("Cached demag boundary partition does not match the mesh.")
            phi[interior_nodes] = factorization.solve(interior_rhs)
        else:
            phi[interior_nodes] = _solve_linear_system(
                interior_stiffness,
                interior_rhs,
                record_timing=self._record_active_subfield_array_timing,
                timing_prefix="demag_auxiliary:dirichlet_extension",
            )
        return phi
