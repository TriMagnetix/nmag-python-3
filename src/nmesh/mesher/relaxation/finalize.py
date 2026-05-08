"""Final mesh cleanup helpers for the relaxation meshing pipeline."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ._constants import BOUNDARY_FUZZ, STATE_BOUNDARY, STATE_FIXED, STATE_MOBILE, STATE_SIMPLE
from ._types import FloatArray
from .geometry import FemGeometry
from .topology import _triangulate_points


def snap_final_boundary_points(
    points: FloatArray,
    states: np.ndarray,
    geometry: FemGeometry,
    a0: float,
    params: dict[str, Any],
) -> tuple[FloatArray, np.ndarray]:
    """Snap final points near fixed/boundary neighbors onto implicit boundaries."""

    if len(points) < geometry.dim + 1:
        return points, states

    simplices = _triangulate_points(points, geometry.dim, states)
    if len(simplices) == 0:
        return points, states

    neighbor_map = _build_neighbor_map(len(points), simplices)
    cleaned_points = np.array(points, copy=True)
    cleaned_states = np.array(states, copy=True)
    acceptable_fuzz = float(params.get("boundary_condition_acceptable_fuzz", BOUNDARY_FUZZ))
    max_steps = int(params.get("boundary_condition_max_nr_correction_steps", 200))

    for point_index, neighbors in enumerate(neighbor_map):
        if not neighbors or not _has_boundary_like_neighbor(cleaned_states, neighbors):
            continue

        original = cleaned_points[point_index]
        snapped = geometry.project_point_to_boundary_from_inside(
            original,
            acceptable_fuzz=acceptable_fuzz,
            max_steps=max_steps,
        )
        if geometry.boundary_distance(snapped) > max(acceptable_fuzz, BOUNDARY_FUZZ):
            continue
        if geometry.classify_points(snapped.reshape(1, -1))[0] < 0:
            continue
        local_rod = a0 / (geometry.density_at(original) ** (1.0 / max(geometry.dim, 1)))
        if float(np.linalg.norm(snapped - original)) < 0.2 * local_rod:
            cleaned_points[point_index] = snapped
            if cleaned_states[point_index] in (STATE_MOBILE, STATE_BOUNDARY):
                cleaned_states[point_index] = STATE_BOUNDARY

    return cleaned_points, cleaned_states


def _build_neighbor_map(point_count: int, simplices: np.ndarray) -> list[list[int]]:
    """Build undirected point adjacency from simplices."""

    neighbors: list[set[int]] = [set() for _ in range(point_count)]
    for simplex in simplices:
        for left, right in itertools.combinations(simplex.tolist(), 2):
            left_index = int(left)
            right_index = int(right)
            neighbors[left_index].add(right_index)
            neighbors[right_index].add(left_index)
    return [sorted(group) for group in neighbors]


def _has_boundary_like_neighbor(states: np.ndarray, neighbors: list[int]) -> bool:
    """Return whether any neighbor should trigger final boundary snapping."""

    boundary_like_states = {STATE_FIXED, STATE_BOUNDARY, STATE_SIMPLE}
    return any(int(states[neighbor]) in boundary_like_states for neighbor in neighbors)
