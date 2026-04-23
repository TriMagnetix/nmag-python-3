"""Numba-accelerated kernels for the relaxation meshing pipeline."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ...utils.constants import MIN_DIVISION_MAGNITUDE
from ._constants import DENSITY_EPSILON, STATE_BOUNDARY, STATE_FIXED, STATE_MOBILE, STATE_SIMPLE


@njit(cache=True)
def _default_relaxation_force(reduced_distance: float) -> float:
    """Return the legacy default mobile-mobile force law."""

    if reduced_distance > 1.0:
        return 0.0
    return 1.0 - reduced_distance


@njit(cache=True)
def _default_boundary_force(reduced_distance: float) -> float:
    """Return the legacy default boundary interaction force law."""

    if reduced_distance > 1.0:
        return 0.0
    if reduced_distance < MIN_DIVISION_MAGNITUDE:
        return 1.0e12
    return 1.0 / reduced_distance - 1.0


@njit(cache=True)
def accumulate_neighbor_forces_default(
    points: np.ndarray,
    states: np.ndarray,
    point_densities: np.ndarray,
    neighbor_pairs: np.ndarray,
    a0: float,
    neigh_force_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate default neighbor forces and controller statistics."""

    point_count, dim = points.shape
    total_forces = np.zeros((point_count, dim), dtype=np.float64)
    neighbor_force_sums = np.zeros(point_count, dtype=np.float64)
    neighbor_force_counts = np.zeros(point_count, dtype=np.int64)
    inv_dim = 1.0 / float(max(dim, 1))
    a0_safe = max(a0, DENSITY_EPSILON)

    for pair_index in range(len(neighbor_pairs)):
        left = int(neighbor_pairs[pair_index, 0])
        right = int(neighbor_pairs[pair_index, 1])
        state_left = int(states[left])
        state_right = int(states[right])
        left_dynamic = state_left == STATE_MOBILE or state_left == STATE_BOUNDARY
        right_dynamic = state_right == STATE_MOBILE or state_right == STATE_BOUNDARY
        if not left_dynamic and not right_dynamic:
            continue

        true_distance_sq = 0.0
        for axis in range(dim):
            delta = points[right, axis] - points[left, axis]
            true_distance_sq += delta * delta
        true_distance = math.sqrt(true_distance_sq)
        if true_distance <= DENSITY_EPSILON:
            continue

        avg_density = 0.5 * (point_densities[left] + point_densities[right])
        inv_length_scale = math.pow(avg_density, inv_dim) / a0_safe
        reduced_distance = true_distance * inv_length_scale
        boundary_interaction = (
            state_left == STATE_FIXED
            or state_left == STATE_BOUNDARY
            or state_left == STATE_SIMPLE
            or state_right == STATE_FIXED
            or state_right == STATE_BOUNDARY
            or state_right == STATE_SIMPLE
        )
        scalar_force = (
            _default_boundary_force(reduced_distance)
            if boundary_interaction
            else _default_relaxation_force(reduced_distance)
        )

        scaled_force = neigh_force_scale * abs(scalar_force)
        neighbor_force_sums[left] += scaled_force
        neighbor_force_sums[right] += scaled_force
        neighbor_force_counts[left] += 1
        neighbor_force_counts[right] += 1

        if scalar_force == 0.0:
            continue

        force_factor = neigh_force_scale * (-scalar_force)
        for axis in range(dim):
            contribution = force_factor * (points[right, axis] - points[left, axis])
            total_forces[left, axis] += contribution
            total_forces[right, axis] -= contribution

    return total_forces, neighbor_force_sums, neighbor_force_counts
