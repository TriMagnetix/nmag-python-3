from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ....demag import _mean_edge_length, _simplex_volumes
from ...support import MU0_OVER_4PI, _simulation_compatibility_binding


class SimulationDemagProbeMixin:
    if TYPE_CHECKING:
        _demag_dipoles: tuple[np.ndarray, np.ndarray, float] | None
        _demag_cell_gradient_inputs_cache: (
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
            | None
        )

        def __getattr__(self, name: str) -> Any: ...

    def _cell_demag_from_potential(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        if len(simplices) == 0:
            return np.zeros((0, 3), dtype=float)
        _, gradients_by_cell, _ = self._demag_fem_geometry_for_mesh(points, simplices)
        inputs_immutable = not simplices.flags.writeable and not gradients_by_cell.flags.writeable
        cached = self._demag_cell_gradient_inputs_cache
        if (
            inputs_immutable
            and cached is not None
            and cached[0] is simplices
            and cached[1] is gradients_by_cell
        ):
            _, _, s0, s1, s2, s3, neg_g0, neg_g1, neg_g2, neg_g3 = cached
        else:
            s0 = np.ascontiguousarray(simplices[:, 0])
            s1 = np.ascontiguousarray(simplices[:, 1])
            s2 = np.ascontiguousarray(simplices[:, 2])
            s3 = np.ascontiguousarray(simplices[:, 3])
            neg_g0 = np.ascontiguousarray(-gradients_by_cell[:, 0])
            neg_g1 = np.ascontiguousarray(-gradients_by_cell[:, 1])
            neg_g2 = np.ascontiguousarray(-gradients_by_cell[:, 2])
            neg_g3 = np.ascontiguousarray(-gradients_by_cell[:, 3])
            if inputs_immutable:
                self._demag_cell_gradient_inputs_cache = (
                    simplices,
                    gradients_by_cell,
                    s0,
                    s1,
                    s2,
                    s3,
                    neg_g0,
                    neg_g1,
                    neg_g2,
                    neg_g3,
                )
        out = phi[s0, np.newaxis] * neg_g0
        out += phi[s1, np.newaxis] * neg_g1
        out += phi[s2, np.newaxis] * neg_g2
        out += phi[s3, np.newaxis] * neg_g3
        return out

    def _build_demag_dipoles(self) -> tuple[np.ndarray, np.ndarray, float]:
        if "m" not in self._fields:
            raise RuntimeError("Magnetisation must be set before using demag fields.")

        points = self._mesh_points()
        mesh = self._require_mesh()
        simplices = self._mesh_simplices()
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
