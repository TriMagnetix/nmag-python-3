from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np


class SimulationExchangeFieldMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def _get_dm_dcurrent(self) -> np.ndarray:
        if "current_density" not in self._fields:
            return np.zeros((len(self._mesh_points()), 3), dtype=float)
        points = self._mesh_points()
        simplices = np.asarray(self._require_mesh().simplices, dtype=int)
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
        simplices = np.asarray(self._require_mesh().simplices, dtype=int)
        exchange = np.zeros((len(points), 3), dtype=float)
        if simplices.size == 0:
            self._exchange_cache_token = token
            self._exchange_nodal_cache = exchange
            return exchange
        if simplices.ndim != 2 or simplices.shape[1] != 4:
            raise NotImplementedError("The exchange MVP currently supports tetrahedral 3D meshes.")
        if self._exchange_is_zero_by_construction():
            self._exchange_cache_token = token
            self._exchange_nodal_cache = exchange
            return exchange

        stiffness, _gradients_by_cell, volumes = self._demag_fem_geometry_for_mesh(
            points, simplices
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
        if np.any(present):
            cofield = stiffness @ np.asarray(self._fields["m"], dtype=float)
            prefactors = self._nodal_material_coefficients().exchange_prefactor
            exchange[present] = -(
                prefactors[present, np.newaxis]
                * cofield[present]
                / lumped_volumes[present, np.newaxis]
            )

        self._exchange_cache_token = token
        self._exchange_nodal_cache = exchange
        return exchange

    def _demag_token(self) -> tuple[int, int]:
        if self.mesh is None:
            raise RuntimeError("A mesh must be loaded before using demag fields.")
        if "m" not in self._fields:
            raise RuntimeError("Magnetisation must be set before using demag fields.")
        return (id(self._require_mesh().raw_mesh), id(self._fields["m"]))
