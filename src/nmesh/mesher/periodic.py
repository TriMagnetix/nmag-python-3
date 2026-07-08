"""Helpers for building periodic-equivalence groups from mesh coordinates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


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
    if not any(periodic_flags):
        return []

    grouped: defaultdict[tuple[tuple[str, float | int], ...], list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        key = _periodic_equivalence_key(
            point,
            bbox_min,
            bbox_max,
            periodic_flags,
            tolerance,
        )
        if key is None:
            continue
        grouped[key].append(int(index))

    return [sorted(group) for group in grouped.values() if len(group) >= 2]


def _periodic_equivalence_key(
    point: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    periodic_flags: list[bool],
    tolerance: float,
) -> tuple[tuple[str, float | int], ...] | None:
    """Return a canonical key for all periodic copies of one boundary point."""

    key: list[tuple[str, float | int]] = []
    touches_periodic_boundary = False
    for axis, value in enumerate(point):
        coord = float(value)
        if not periodic_flags[axis]:
            key.append(("coord", coord))
            continue

        on_min = np.isclose(coord, bbox_min[axis], atol=tolerance, rtol=0.0)
        on_max = np.isclose(coord, bbox_max[axis], atol=tolerance, rtol=0.0)
        if on_min or on_max:
            key.append(("periodic", axis))
            touches_periodic_boundary = True
        else:
            key.append(("coord", coord))

    if not touches_periodic_boundary:
        return None
    return tuple(key)
