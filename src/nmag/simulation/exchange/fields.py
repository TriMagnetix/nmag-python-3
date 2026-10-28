from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...demag.linear import _is_sparse_matrix


class SimulationExchangeFieldMixin:
    if TYPE_CHECKING:
        _exchange_operator_cache: (
            tuple[int, Any, np.ndarray, Any, Any, np.ndarray, bool, np.ndarray] | None
        )

        def __getattr__(self, name: str) -> Any: ...

    def _get_dm_dcurrent(self) -> np.ndarray:
        if "current_density" not in self._fields:
            return np.zeros((len(self._mesh_points()), 3), dtype=float)
        points = self._mesh_points()
        simplices = self._mesh_simplices()
        if simplices.ndim != 2 or simplices.shape[1] != 4:
            raise NotImplementedError("Spin-transfer torque currently supports tetrahedral meshes.")
        stiffness, gradients_by_cell, volumes = self._demag_fem_geometry_for_mesh(
            points,
            simplices,
        )
        del stiffness
        m = np.asarray(self._fields["m"], dtype=float)
        cell_gradients = np.einsum(
            "cla,clj->caj",
            m[simplices],
            gradients_by_cell,
            optimize=True,
        )
        nodal_gradients = np.zeros((len(points), 3, 3), dtype=float)
        weighted_gradients = cell_gradients * volumes[:, np.newaxis, np.newaxis]
        for local_index in range(4):
            np.add.at(
                nodal_gradients,
                simplices[:, local_index],
                weighted_gradients,
            )
        weights = self._incident_cell_volume_sums(points, simplices, volumes)
        present = weights > 0.0
        nodal_gradients[present] /= weights[present, np.newaxis, np.newaxis]
        current_density = np.asarray(self._fields["current_density"], dtype=float)
        if current_density.shape != (len(points), 3):
            raise ValueError(
                "current_density must contain one 3-vector per mesh point; "
                f"got {current_density.shape}."
            )
        return np.einsum(
            "nij,nj->ni",
            nodal_gradients,
            current_density,
            optimize=True,
        )

    def _get_exchange_nodal_field(self) -> np.ndarray:
        token = self._demag_token()
        if self._exchange_cache_token == token:
            return self._exchange_nodal_cache

        points = self._mesh_points()
        simplices = self._mesh_simplices()
        exchange = np.zeros((len(points), 3), dtype=float)
        if simplices.size == 0:
            exchange.setflags(write=False)
            self._exchange_cache_token = token
            self._exchange_nodal_cache = exchange
            return exchange
        if simplices.ndim != 2 or simplices.shape[1] != 4:
            raise NotImplementedError("The exchange MVP currently supports tetrahedral 3D meshes.")
        if self._exchange_is_zero_by_construction():
            exchange.setflags(write=False)
            self._exchange_cache_token = token
            self._exchange_nodal_cache = exchange
            return exchange

        stiffness, _gradients_by_cell, volumes = self._demag_fem_geometry_for_mesh(
            points, simplices
        )
        geom_token = self._mesh_geometry_token()
        coefficients = self._nodal_material_coefficients()
        op_cache = self._exchange_operator_cache
        if (
            op_cache is not None
            and op_cache[0] == geom_token
            and op_cache[1] is stiffness
            and op_cache[2] is volumes
            and op_cache[3] is coefficients
        ):
            _, _, _, _, sparse_stiffness, present, has_present, scale_present = op_cache
        else:
            from scipy import sparse

            sparse_stiffness = (
                stiffness if _is_sparse_matrix(stiffness) else sparse.csr_matrix(stiffness)
            )
            lumped_volumes = (
                self._incident_cell_volume_sums(
                    points,
                    simplices,
                    volumes,
                )
                / 4.0
            )
            present = lumped_volumes > 0.0
            has_present = bool(np.any(present))
            if has_present:
                prefactors = coefficients.exchange_prefactor
                scale_present = -(
                    prefactors[present, np.newaxis] / lumped_volumes[present, np.newaxis]
                )
            else:
                scale_present = np.zeros((0, 1), dtype=float)
            self._exchange_operator_cache = (
                geom_token,
                stiffness,
                volumes,
                coefficients,
                sparse_stiffness,
                present,
                has_present,
                scale_present,
            )

        if has_present:
            cofield = sparse_stiffness @ np.asarray(self._fields["m"], dtype=float)
            exchange[present] = scale_present * cofield[present]

        exchange.setflags(write=False)
        self._exchange_cache_token = token
        self._exchange_nodal_cache = exchange
        return exchange

    def _demag_token(self) -> tuple[int, int, int]:
        if self.mesh is None:
            raise RuntimeError("A mesh must be loaded before using demag fields.")
        if "m" not in self._fields:
            raise RuntimeError("Magnetisation must be set before using demag fields.")
        return (
            self._mesh_geometry_token(),
            self._material_state_token(),
            id(self._fields["m"]),
        )
