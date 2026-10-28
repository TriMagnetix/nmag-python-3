from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ....backends import (
    _load_rust_accelerator,
    _selected_demag_fem_assembly_backend,
    _selected_memory_mode,
)
from ...support import _simulation_compatibility_binding
from .charges import volume_charge_scaling_correction

_DIVERGENCE_OPERATOR_MIN_SIMPLICES = 32
_DIVERGENCE_OPERATOR_MAX_SIMPLICES = 250_000


class SimulationDemagFemAssemblyMixin:
    if TYPE_CHECKING:
        _demag_fem_divergence_operator_cache: (
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Any] | None
        )

        def __getattr__(self, name: str) -> Any: ...

    def _assemble_demag_fem_system(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        m: np.ndarray,
        ms_values: np.ndarray,
        volume_charge_scales: np.ndarray | None = None,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        point_count = len(points)
        stiffness, gradients_by_cell, volumes = self._demag_fem_geometry_for_mesh(
            points,
            simplices,
        )
        assembly_backend = _selected_demag_fem_assembly_backend(
            len(simplices), getattr(self, "config", None)
        )
        with self._record_active_subfield_array_timing_block(
            f"demag_auxiliary:assemble_fem:divergence:{assembly_backend}",
        ):
            if assembly_backend == "rust":
                divergence = self._assemble_demag_fem_divergence_rust(
                    simplices,
                    gradients_by_cell,
                    volumes,
                    m,
                    ms_values,
                    point_count,
                )
            else:
                divergence = self._assemble_demag_fem_divergence_python(
                    simplices,
                    gradients_by_cell,
                    volumes,
                    m,
                    ms_values,
                    point_count,
                )
            if (
                len(simplices) > 0
                and volume_charge_scales is not None
                and not np.all(volume_charge_scales == 1.0)
            ):
                divergence += volume_charge_scaling_correction(
                    simplices,
                    gradients_by_cell,
                    volumes,
                    m,
                    ms_values,
                    volume_charge_scales,
                    point_count,
                )
        return stiffness, divergence, volumes

    def _assemble_demag_fem_divergence_python(
        self,
        simplices: np.ndarray,
        gradients_by_cell: np.ndarray,
        volumes: np.ndarray,
        m: np.ndarray,
        ms_values: np.ndarray,
        point_count: int,
    ) -> np.ndarray:
        if len(simplices) == 0:
            return np.zeros(point_count, dtype=float)

        inputs_immutable = (
            not simplices.flags.writeable
            and not gradients_by_cell.flags.writeable
            and not volumes.flags.writeable
            and not ms_values.flags.writeable
        )
        if (
            inputs_immutable
            and _DIVERGENCE_OPERATOR_MIN_SIMPLICES
            <= len(simplices)
            <= _DIVERGENCE_OPERATOR_MAX_SIMPLICES
            and _selected_memory_mode() != "low"
        ):
            cached = self._demag_fem_divergence_operator_cache
            if (
                cached is not None
                and cached[0] is simplices
                and cached[1] is gradients_by_cell
                and cached[2] is volumes
                and cached[3] is ms_values
                and cached[4] == point_count
            ):
                div_operator = cached[5]
            else:
                from scipy import sparse

                cell_weights = (0.25 * ms_values * volumes)[:, np.newaxis, np.newaxis]
                weighted_grads = gradients_by_cell * cell_weights  # (C, 4, 3)
                # Each cell c contributes weighted_grads[c, i, alpha] * m[simplices[c, j], alpha]
                # to row simplices[c, i].
                rows = np.repeat(simplices, 12)
                base_cols = 3 * simplices[:, :, np.newaxis] + np.arange(3)[np.newaxis, np.newaxis, :]
                cols = np.tile(base_cols.reshape(len(simplices), 1, 12), (1, 4, 1)).ravel()
                data = np.tile(weighted_grads[:, :, np.newaxis, :], (1, 1, 4, 1)).ravel()
                div_operator = sparse.csr_matrix(
                    (data, (rows, cols)),
                    shape=(point_count, 3 * point_count),
                )
                self._demag_fem_divergence_operator_cache = (
                    simplices,
                    gradients_by_cell,
                    volumes,
                    ms_values,
                    point_count,
                    div_operator,
                )
            return np.asarray(
                div_operator @ np.ascontiguousarray(m, dtype=float).ravel(),
                dtype=float,
            )

        scaled_m = (
            m[simplices[:, 0]]
            + m[simplices[:, 1]]
            + m[simplices[:, 2]]
            + m[simplices[:, 3]]
        ) * ((0.25 * ms_values * volumes)[:, np.newaxis])
        cell_divergence = np.einsum(
            "cij,cj->ci",
            gradients_by_cell,
            scaled_m,
        )
        return np.bincount(
            simplices.ravel(),
            weights=cell_divergence.ravel(),
            minlength=point_count,
        ).astype(float, copy=False)

    def _assemble_demag_fem_divergence_rust(
        self,
        simplices: np.ndarray,
        gradients_by_cell: np.ndarray,
        volumes: np.ndarray,
        m: np.ndarray,
        ms_values: np.ndarray,
        point_count: int,
    ) -> np.ndarray:
        rust_accel = _simulation_compatibility_binding(
            "_load_rust_accelerator",
            _load_rust_accelerator,
        )("NmagConfig.accelerator['fem_assembly']")
        return np.asarray(
            rust_accel.build_demag_fem_divergence(
                np.asarray(simplices, dtype=np.int64),
                np.asarray(gradients_by_cell, dtype=np.float64),
                np.asarray(volumes, dtype=np.float64),
                np.asarray(m, dtype=np.float64),
                np.asarray(ms_values, dtype=np.float64),
                int(point_count),
            ),
            dtype=float,
        )
