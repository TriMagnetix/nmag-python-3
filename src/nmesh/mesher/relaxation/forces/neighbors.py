"""Neighbor-pair force calculations for the relaxation mesher."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from ...meshing_parameters import default_boundary_node_force_fun, default_relaxation_force_fun
from .._constants import DENSITY_EPSILON, STATE_BOUNDARY, STATE_FIXED, STATE_MOBILE, STATE_SIMPLE
from .._types import FloatArray
from .jit import accumulate_neighbor_forces_default
from .types import ForceParameters


def _build_neighbor_pairs(simplices: np.ndarray) -> np.ndarray:
    """Return sorted unique point-pair edges induced by the simplices."""
    if len(simplices) == 0:
        return np.empty((0, 2), dtype=np.int64)

    pairs: set[tuple[int, int]] = set()
    for simplex in simplices:
        for left, right in combinations(simplex.tolist(), 2):
            i = int(left)
            j = int(right)
            pairs.add((i, j) if i <= j else (j, i))
    return np.asarray(sorted(pairs), dtype=np.int64)


def _is_dynamic_state(state: int) -> bool:
    """Return whether the state participates in relaxation movement."""
    return state in (STATE_MOBILE, STATE_BOUNDARY)


def _is_boundary_interaction(state_a: int, state_b: int) -> bool:
    """Return whether a neighbor interaction should use the boundary law."""
    return state_a in (STATE_FIXED, STATE_BOUNDARY, STATE_SIMPLE) or state_b in (
        STATE_FIXED,
        STATE_BOUNDARY,
        STATE_SIMPLE,
    )


def _neighbor_forces_python(
    points: FloatArray,
    states: np.ndarray,
    point_densities: FloatArray,
    neighbor_map: list[list[int]],
    dim: int,
    a0: float,
    neigh_force_scale: float,
    force_fun: Any,
    boundary_force_fun: Any,
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Fallback neighbor-force path for custom Python force callbacks."""
    total_forces = np.zeros((len(points), dim), dtype=float)
    neighbor_force_sums = np.zeros(len(points), dtype=float)
    neighbor_force_counts = np.zeros(len(points), dtype=int)

    for left, neighbors in enumerate(neighbor_map):
        point_left = points[left]
        state_left = int(states[left])
        for right in neighbors:
            if right <= left:
                continue
            contribution, scalar_force = _apply_pairwise_force(
                point_left=point_left,
                point_right=points[right],
                state_left=state_left,
                state_right=int(states[right]),
                density_left=float(point_densities[left]),
                density_right=float(point_densities[right]),
                dim=dim,
                a0=a0,
                neigh_force_scale=neigh_force_scale,
                force_fun=force_fun,
                boundary_force_fun=boundary_force_fun,
            )
            if scalar_force is None:
                continue
            neighbor_force_sums[left] += abs(scalar_force)
            neighbor_force_sums[right] += abs(scalar_force)
            neighbor_force_counts[left] += 1
            neighbor_force_counts[right] += 1
            if contribution is None:
                continue
            total_forces[left] += contribution
            total_forces[right] -= contribution

    return total_forces, neighbor_force_sums, neighbor_force_counts


def _apply_pairwise_force(
    *,
    point_left: FloatArray,
    point_right: FloatArray,
    state_left: int,
    state_right: int,
    density_left: float,
    density_right: float,
    dim: int,
    a0: float,
    neigh_force_scale: float,
    force_fun: Any,
    boundary_force_fun: Any,
) -> tuple[FloatArray | None, float | None]:
    """Compute the pairwise neighbor contribution for one point pair."""
    if not _is_dynamic_state(state_left) and not _is_dynamic_state(state_right):
        return None, None

    delta = point_right - point_left
    true_distance = float(np.linalg.norm(delta))
    if true_distance <= DENSITY_EPSILON:
        return None, None

    avg_density = 0.5 * (density_left + density_right)
    inv_length_scale = (avg_density ** (1.0 / max(dim, 1))) / max(a0, DENSITY_EPSILON)
    reduced_distance = true_distance * inv_length_scale
    scalar_force = float(
        (boundary_force_fun if _is_boundary_interaction(state_left, state_right) else force_fun)(
            reduced_distance
        )
    )
    scaled_force = neigh_force_scale * abs(scalar_force)
    if scalar_force == 0.0:
        return None, scaled_force
    return neigh_force_scale * (-scalar_force) * delta, scaled_force


def _build_neighbor_map(point_count: int, simplices: np.ndarray) -> list[list[int]]:
    """Build undirected point adjacency from simplices."""
    neighbors: list[set[int]] = [set() for _ in range(point_count)]
    for simplex in simplices:
        for left, right in combinations(simplex.tolist(), 2):
            i = int(left)
            j = int(right)
            neighbors[i].add(j)
            neighbors[j].add(i)
    return [sorted(group) for group in neighbors]


def _compute_neighbor_forces(
    points: FloatArray,
    states: np.ndarray,
    simplices: np.ndarray,
    neighbor_map: list[list[int]],
    point_densities: FloatArray,
    dim: int,
    a0: float,
    config: ForceParameters,
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Compute neighbor-force contributions through the JIT or Python path."""
    point_count = len(points)
    if len(simplices) == 0:
        return (
            np.zeros((point_count, dim), dtype=float),
            np.zeros(point_count, dtype=float),
            np.zeros(point_count, dtype=int),
        )

    if (
        config.force_fun is default_relaxation_force_fun
        and config.boundary_force_fun is default_boundary_node_force_fun
    ):
        neighbor_pairs = _build_neighbor_pairs(simplices)
        return accumulate_neighbor_forces_default(
            np.asarray(points, dtype=np.float64),
            np.asarray(states, dtype=np.int64),
            point_densities.astype(np.float64, copy=False),
            neighbor_pairs,
            float(a0),
            config.neigh_force_scale,
        )

    return _neighbor_forces_python(
        points,
        states,
        point_densities,
        neighbor_map,
        dim,
        a0,
        config.neigh_force_scale,
        config.force_fun,
        config.boundary_force_fun,
    )
