from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ....backends import (
    _load_rust_accelerator,
    _selected_demag_cell_average_backend,
    _selected_demag_nodal_recovery_backend,
)
from ...support import _simulation_compatibility_binding


class SimulationDemagRecoveryMixin:
    if TYPE_CHECKING:
        _demag_nodal_cache: np.ndarray | None
        _demag_cell_field_cache: tuple[np.ndarray, np.ndarray] | None

        def __getattr__(self, name: str) -> Any: ...

    def _get_demag_nodal_field(self) -> np.ndarray:
        token = self._demag_token()
        if self._demag_cache_token == token and self._demag_nodal_cache is not None:
            return self._demag_nodal_cache

        points = self._mesh_points()
        simplices = np.asarray(self._require_mesh().simplices, dtype=int)
        if simplices.size == 0:
            self._demag_cache_token = token
            self._demag_nodal_cache = np.zeros((len(points), 3), dtype=float)
            return self._demag_nodal_cache
        if simplices.ndim != 2 or simplices.shape[1] != 4:
            raise NotImplementedError("The demag MVP currently supports tetrahedral 3D meshes.")

        cell_h, volumes = self._get_demag_cell_field()
        with self._record_active_subfield_array_timing_block(
            "H_demag_detail:nodal_recovery",
        ):
            weights = self._incident_cell_volume_sums(points, simplices, volumes)
            backend = _selected_demag_nodal_recovery_backend(
                len(simplices), getattr(self, "config", None)
            )
            with self._record_active_subfield_array_timing_block(
                f"H_demag_detail:nodal_recovery:{backend}",
            ):
                if backend == "rust":
                    nodal_h = self._recover_demag_nodal_field_rust(
                        simplices,
                        volumes,
                        cell_h,
                        weights,
                        len(points),
                    )
                else:
                    nodal_h = self._recover_demag_nodal_field_python(
                        simplices,
                        volumes,
                        cell_h,
                        weights,
                        len(points),
                    )
        self._demag_cache_token = token
        self._demag_nodal_cache = nodal_h
        return nodal_h

    def _get_demag_cell_field(self) -> tuple[np.ndarray, np.ndarray]:
        cache_started = time.perf_counter()
        token = self._demag_token()
        if self._demag_cell_cache_token == token and self._demag_cell_field_cache is not None:
            self._record_active_subfield_array_timing(
                "H_demag_detail:cell_field_cache_hit",
                time.perf_counter() - cache_started,
            )
            return self._demag_cell_field_cache

        points = self._mesh_points()
        simplices = np.asarray(self._require_mesh().simplices, dtype=int)
        if simplices.size == 0:
            cell_h = np.zeros((0, 3), dtype=float)
            volumes = np.zeros(0, dtype=float)
            self._demag_cell_cache_token = token
            self._demag_cell_field_cache = (cell_h, volumes)
            return self._demag_cell_field_cache
        if simplices.ndim != 2 or simplices.shape[1] != 4:
            raise NotImplementedError("The demag MVP currently supports tetrahedral 3D meshes.")

        with self._record_active_subfield_array_timing_block(
            "H_demag_detail:auxiliary_fields",
        ):
            phi, _rho, volumes = self._get_demag_auxiliary_fields()
        with self._record_active_subfield_array_timing_block(
            "H_demag_detail:cell_gradient",
        ):
            cell_h = self._cell_demag_from_potential(points, simplices, phi)
        self._demag_cell_cache_token = token
        self._demag_cell_field_cache = (cell_h, volumes)
        return self._demag_cell_field_cache

    def _demag_cell_field_average(self) -> np.ndarray:
        cell_h, volumes = self._get_demag_cell_field()
        backend = _selected_demag_cell_average_backend(
            len(volumes), getattr(self, "config", None)
        )
        with self._record_active_subfield_array_timing_block(
            "H_demag_detail:cell_average",
        ):
            with self._record_active_subfield_array_timing_block(
                f"H_demag_detail:cell_average:{backend}",
            ):
                if backend == "rust":
                    return self._demag_cell_field_average_rust(cell_h, volumes)
                return self._demag_cell_field_average_python(cell_h, volumes)

    def _demag_cell_field_average_python(
        self,
        cell_h: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        if cell_h.size == 0 or volumes.size == 0:
            return np.zeros(3, dtype=float)
        positive = volumes > 0.0
        if not np.any(positive):
            return np.zeros(3, dtype=float)

        if np.all(positive):
            total_volume = float(np.sum(volumes))
            if total_volume <= 0.0:
                return np.zeros(3, dtype=float)
            return np.dot(volumes, cell_h) / total_volume

        total_volume = float(np.sum(volumes[positive]))
        if total_volume <= 0.0:
            return np.zeros(3, dtype=float)
        return np.sum(cell_h[positive] * volumes[positive, np.newaxis], axis=0) / total_volume

    def _demag_cell_field_average_rust(
        self,
        cell_h: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        rust_accel = _simulation_compatibility_binding(
            "_load_rust_accelerator",
            _load_rust_accelerator,
        )("NmagConfig.accelerator['cell_average']")
        try:
            demag_cell_field_average = rust_accel.demag_cell_field_average
        except AttributeError as exc:
            raise RuntimeError(
                "NmagConfig.accelerator['cell_average']='rust' requires a nmag_accel build "
                "with demag_cell_field_average support."
            ) from exc
        return np.asarray(
            demag_cell_field_average(
                np.asarray(cell_h, dtype=np.float64),
                np.asarray(volumes, dtype=np.float64),
            ),
            dtype=float,
        )

    def _recover_demag_nodal_field_python(
        self,
        simplices: np.ndarray,
        volumes: np.ndarray,
        cell_h: np.ndarray,
        weights: np.ndarray,
        point_count: int,
    ) -> np.ndarray:
        nodal_h = np.zeros((point_count, 3), dtype=float)
        positive = volumes > 0.0
        if np.any(positive):
            positive_simplices = simplices[positive]
            weighted_cell_h = cell_h[positive] * volumes[positive, np.newaxis]
            for local_index in range(4):
                np.add.at(
                    nodal_h,
                    positive_simplices[:, local_index],
                    weighted_cell_h,
                )

        present = weights > 0.0
        nodal_h[present] /= weights[present, np.newaxis]
        return nodal_h

    def _recover_demag_nodal_field_rust(
        self,
        simplices: np.ndarray,
        volumes: np.ndarray,
        cell_h: np.ndarray,
        weights: np.ndarray,
        point_count: int,
    ) -> np.ndarray:
        rust_accel = _simulation_compatibility_binding(
            "_load_rust_accelerator",
            _load_rust_accelerator,
        )("NmagConfig.accelerator['nodal_recovery']")
        return np.asarray(
            rust_accel.recover_demag_nodal_field(
                np.asarray(simplices, dtype=np.int64),
                np.asarray(volumes, dtype=np.float64),
                np.asarray(cell_h, dtype=np.float64),
                np.asarray(weights, dtype=np.float64),
                int(point_count),
            ),
            dtype=float,
        )
