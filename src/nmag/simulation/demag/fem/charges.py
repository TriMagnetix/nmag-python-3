"""Finite-element separation of demagnetization volume charges."""

from __future__ import annotations

import numpy as np


def volume_charge_scaling_correction(
    simplices: np.ndarray,
    gradients_by_cell: np.ndarray,
    volumes: np.ndarray,
    m: np.ndarray,
    ms_values: np.ndarray,
    volume_charge_scales: np.ndarray,
    point_count: int,
) -> np.ndarray:
    """Return the weak-source correction for material volume-charge scales.

    The standard FEM assembly is the combined surface and volume source
    ``integral(grad(test) . M)``. Integration by parts identifies the volume
    part on a linear tetrahedron as ``-volume * div(M) / 4`` at every local
    node. Scaling that term separately preserves the unscaled surface charge.
    """
    correction = np.zeros(point_count, dtype=float)
    if len(simplices) == 0 or np.all(volume_charge_scales == 1.0):
        return correction

    cell_m = m[simplices] * ms_values[:, np.newaxis, np.newaxis]
    cell_divergence = np.einsum("cij,cij->c", gradients_by_cell, cell_m)
    local_correction = (
        -(volume_charge_scales - 1.0) * volumes * cell_divergence / simplices.shape[1]
    )
    np.add.at(
        correction,
        simplices,
        np.broadcast_to(local_correction[:, np.newaxis], simplices.shape),
    )
    return correction
