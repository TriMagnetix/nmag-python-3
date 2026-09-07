"""Periodic-boundary seed-point and state helpers."""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from .._constants import BOUNDARY_FUZZ, DENSITY_EPSILON, STATE_FIXED
from .._types import FloatArray
from ..geometry import FemGeometry
from .points import _dedupe_points, _filter_relevant_points


def _periodic_outer_box_points(
    geometry: FemGeometry, a0: float, periodic: list[float] | list[bool]
) -> FloatArray:
    """Create paired fixed seed points on periodic outer-box faces."""

    periodic_flags = [bool(value) for value in periodic]
    if not any(periodic_flags):
        return np.empty((0, geometry.dim), dtype=float)
    points: list[FloatArray] = []
    spacing = max(a0, DENSITY_EPSILON)
    for periodic_axis, enabled in enumerate(periodic_flags):
        if not enabled:
            continue
        points.extend(_periodic_face_points(geometry, periodic_axis, spacing))
    if not points:
        return np.empty((0, geometry.dim), dtype=float)
    return _filter_relevant_points(geometry, _dedupe_points(np.asarray(points, dtype=float)))


def _periodic_face_points(
    geometry: FemGeometry,
    periodic_axis: int,
    spacing: float,
) -> list[FloatArray]:
    """Return matching points on the two periodic faces of one axis."""

    other_axes = [axis for axis in range(geometry.dim) if axis != periodic_axis]
    face_axes: list[FloatArray] = []
    for axis in other_axes:
        extent = float(geometry.bbox_max[axis] - geometry.bbox_min[axis])
        count = max(2, int(math.floor(extent / spacing)) + 1)
        face_axes.append(
            np.linspace(
                float(geometry.bbox_min[axis]),
                float(geometry.bbox_max[axis]),
                count,
                dtype=float,
            )
        )

    points: list[FloatArray] = []
    for coordinates in product(*(axis.tolist() for axis in face_axes)):
        for side in (geometry.bbox_min[periodic_axis], geometry.bbox_max[periodic_axis]):
            point = np.zeros(geometry.dim, dtype=float)
            point[periodic_axis] = side
            for axis, value in zip(other_axes, coordinates, strict=True):
                point[axis] = value
            points.append(point)
    return points


def _apply_periodic_fixed_states(
    states: np.ndarray,
    all_points: FloatArray,
    geometry: FemGeometry,
    periodic: list[float] | list[bool],
    a0: float,
) -> None:
    """Mark points on periodic boundaries as fixed."""

    _ = a0
    if len(all_points) == 0:
        return
    periodic_mask = np.zeros(len(all_points), dtype=bool)
    for axis, enabled in enumerate(map(bool, periodic)):
        if enabled:
            periodic_mask |= np.isclose(
                all_points[:, axis], geometry.bbox_min[axis], atol=BOUNDARY_FUZZ, rtol=0.0
            )
            periodic_mask |= np.isclose(
                all_points[:, axis], geometry.bbox_max[axis], atol=BOUNDARY_FUZZ, rtol=0.0
            )
    states[periodic_mask] = STATE_FIXED
