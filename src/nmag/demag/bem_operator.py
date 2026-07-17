from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lindholm_fast import _apply_lindholm_bem_matrix_free_fast


@dataclass(frozen=True, slots=True)
class MatrixFreeLindholmBemOperator:
    """Exact Lindholm BEM action without storing the dense boundary matrix."""

    points: np.ndarray
    simplices: np.ndarray
    boundary_faces: np.ndarray
    face_points: np.ndarray
    zeta_vectors: np.ndarray
    eta_vectors: np.ndarray
    edge_lengths: np.ndarray
    corner_cosines: np.ndarray
    denominator_factors: np.ndarray
    boundary_nodes: np.ndarray
    local_index_by_point: np.ndarray
    solid_angles: np.ndarray
    rust_accelerator: Any | None = None

    @property
    def shape(self) -> tuple[int, int]:
        size = len(self.boundary_nodes)
        return size, size

    def __matmul__(self, values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        expected = (len(self.boundary_nodes),)
        if vector.shape != expected:
            raise ValueError(f"BEM input must have shape {expected}, got {vector.shape}.")
        if self.rust_accelerator is not None:
            return np.asarray(
                self.rust_accelerator.apply_lindholm_bem_matrix_free(
                    self.points,
                    self.simplices,
                    self.boundary_faces,
                    self.boundary_nodes,
                    self.local_index_by_point,
                    vector,
                ),
                dtype=float,
            )
        return np.asarray(
            _apply_lindholm_bem_matrix_free_fast(
                self.points,
                self.boundary_faces,
                self.face_points,
                self.zeta_vectors,
                self.eta_vectors,
                self.edge_lengths,
                self.corner_cosines,
                self.denominator_factors,
                self.boundary_nodes,
                self.local_index_by_point,
                self.solid_angles,
                vector,
            ),
            dtype=float,
        )

    @property
    def storage_bytes(self) -> int:
        arrays = (
            self.points,
            self.simplices,
            self.boundary_faces,
            self.face_points,
            self.zeta_vectors,
            self.eta_vectors,
            self.edge_lengths,
            self.corner_cosines,
            self.denominator_factors,
            self.boundary_nodes,
            self.local_index_by_point,
            self.solid_angles,
        )
        return sum(int(array.nbytes) for array in arrays)
