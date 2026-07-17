from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ....backends import (
    _load_rust_accelerator,
    _selected_demag_fem_assembly_backend,
)
from ...support import _simulation_compatibility_binding


class SimulationDemagFemAssemblyMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def _assemble_demag_fem_system(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        m: np.ndarray,
        ms_values: np.ndarray,
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
        divergence = np.zeros(point_count, dtype=float)
        if len(simplices) == 0:
            return divergence

        cell_m = m[simplices] * ms_values[:, np.newaxis, np.newaxis]
        average_m = np.mean(cell_m, axis=1)
        cell_divergence = volumes[:, np.newaxis] * np.einsum(
            "cij,cj->ci",
            gradients_by_cell,
            average_m,
        )
        np.add.at(divergence, simplices, cell_divergence)
        return divergence

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
