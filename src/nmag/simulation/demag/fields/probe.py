from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ....demag import _mean_edge_length, _simplex_volumes
from ...support import MU0_OVER_4PI, _simulation_compatibility_binding


class SimulationDemagProbeMixin:
    if TYPE_CHECKING:
        _demag_dipoles: tuple[np.ndarray, np.ndarray, float] | None

        def __getattr__(self, name: str) -> Any: ...

    def _cell_demag_from_potential(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        _, gradients_by_cell, _ = self._demag_fem_geometry_for_mesh(points, simplices)
        return -np.einsum(
            "ci,cij->cj",
            phi[simplices],
            gradients_by_cell,
        )

    def _build_demag_dipoles(self) -> tuple[np.ndarray, np.ndarray, float]:
        if "m" not in self._fields:
            raise RuntimeError("Magnetisation must be set before using demag fields.")

        points = self._mesh_points()
        mesh = self._require_mesh()
        simplices = np.asarray(mesh.simplices, dtype=int)
        if simplices.size == 0:
            return np.empty((0, 3)), np.empty((0, 3)), 1.0
        if simplices.ndim != 2 or simplices.shape[1] != 4:
            raise NotImplementedError("The demag MVP currently supports tetrahedral 3D meshes.")

        token = (id(mesh.raw_mesh), id(self._fields["m"]))
        if self._demag_cache_token == token and self._demag_dipoles is not None:
            return self._demag_dipoles

        m = np.asarray(self._fields["m"], dtype=float)
        centroids = np.mean(points[simplices], axis=1)
        volumes = _simulation_compatibility_binding(
            "_simplex_volumes",
            _simplex_volumes,
        )(points, simplices)
        regions = list(mesh.regions or [1] * len(simplices))
        ms_values = np.asarray(
            [self._simplex_material_ms(region) for region in regions], dtype=float
        )
        cell_m = np.mean(m[simplices], axis=1)
        dipole_moments = cell_m * (ms_values * volumes)[:, np.newaxis]
        softening = 0.25 * _mean_edge_length(points, simplices)
        self._demag_cache_token = token
        self._demag_dipoles = (centroids, dipole_moments, softening)
        return self._demag_dipoles

    def _probe_demag_at(self, pos: Sequence[float]) -> np.ndarray:
        centroids, dipole_moments, softening = self._build_demag_dipoles()
        if len(centroids) == 0:
            return np.zeros(3, dtype=float)

        r = np.asarray(pos, dtype=float) - centroids
        r2 = np.sum(r * r, axis=1) + softening * softening
        r3 = r2**1.5
        r5 = r2**2.5
        mdotr = np.sum(dipole_moments * r, axis=1)
        terms = (3.0 * r * mdotr[:, np.newaxis] / r5[:, np.newaxis]) - (
            dipole_moments / r3[:, np.newaxis]
        )
        return MU0_OVER_4PI * np.sum(terms, axis=0)
