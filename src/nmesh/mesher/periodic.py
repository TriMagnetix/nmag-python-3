from __future__ import annotations

"""Helpers for building periodic-equivalence groups from mesh coordinates."""

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def _point_key(values: np.ndarray, decimals: int = 8) -> tuple[float, ...]:
    """Return a rounded tuple key suitable for coordinate-based lookups."""

    return tuple(np.round(values.astype(float, copy=False), decimals=decimals).tolist())


def merge_periodic_groups(groups: Iterable[Iterable[int]]) -> list[list[int]]:
    """Merge overlapping periodic point groups into disjoint sorted groups."""

    merged: list[set[int]] = []
    for group in groups:
        candidate = {int(index) for index in group}
        if len(candidate) < 2:
            continue

        overlaps: list[set[int]] = []
        survivors: list[set[int]] = []
        for existing in merged:
            if existing & candidate:
                overlaps.append(existing)
            else:
                survivors.append(existing)

        for overlap in overlaps:
            candidate.update(overlap)

        survivors.append(candidate)
        merged = survivors

    return [sorted(group) for group in merged]


def build_periodic_groups(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    periodic_axes: Iterable[bool | float],
    *,
    tolerance: float,
) -> list[list[int]]:
    """Build periodic point groups by matching opposite boundary points."""

    if points.size == 0:
        return []

    periodic_flags = [bool(value) for value in periodic_axes]
    raw_groups: list[list[int]] = []

    for axis, enabled in enumerate(periodic_flags):
        if not enabled:
            continue

        min_mask = np.isclose(points[:, axis], bbox_min[axis], atol=tolerance, rtol=0.0)
        max_mask = np.isclose(points[:, axis], bbox_max[axis], atol=tolerance, rtol=0.0)
        if not np.any(min_mask) or not np.any(max_mask):
            continue

        other_axes = [other for other in range(points.shape[1]) if other != axis]
        min_points = np.flatnonzero(min_mask)
        max_points = np.flatnonzero(max_mask)

        max_lookup: defaultdict[tuple[float, ...], list[int]] = defaultdict(list)
        for index in max_points:
            max_lookup[_point_key(points[index, other_axes])].append(int(index))

        for index in min_points:
            key = _point_key(points[index, other_axes])
            for partner in max_lookup.get(key, []):
                raw_groups.append([int(index), int(partner)])

    return merge_periodic_groups(raw_groups)
