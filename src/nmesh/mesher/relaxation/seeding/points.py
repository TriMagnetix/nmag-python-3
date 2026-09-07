"""Point normalization and state classification for relaxation seeding."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._constants import BOUNDARY_FUZZ, STATE_BOUNDARY, STATE_MOBILE
from .._types import FloatArray
from ..geometry import FemGeometry


def _as_float_array(points: Any, dim: int | None = None) -> FloatArray:
    """Convert point-like input into a 2D float array with optional dimension validation."""

    if points is None:
        return np.empty((0, 0 if dim is None else dim), dtype=float)
    coords = np.asarray(points, dtype=float)
    if coords.size == 0:
        return np.empty((0, 0 if dim is None else dim), dtype=float)
    if coords.ndim == 1:
        dim = int(coords.shape[0]) if dim is None else dim
        coords = coords[np.newaxis, :]
    if dim is not None and coords.shape[1] != dim:
        raise ValueError(f"Expected points with dimension {dim}, got {coords.shape[1]}")
    return coords.astype(float, copy=False)


def _point_key(point: FloatArray, decimals: int = 10) -> tuple[float, ...]:
    """Return a rounded tuple key for deduplication of point coordinates."""

    return tuple(np.round(np.asarray(point, dtype=float), decimals=decimals).tolist())


def _dedupe_points(points: FloatArray) -> FloatArray:
    """Drop duplicate points while preserving first-seen order."""

    if len(points) == 0:
        return points
    keep_indices: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for index, point in enumerate(points):
        key = _point_key(point)
        if key not in seen:
            seen.add(key)
            keep_indices.append(index)
    return points[np.asarray(keep_indices, dtype=int)]


def _dedupe_fixed_mobile(
    fixed_points: FloatArray,
    mobile_points: FloatArray,
    simply_points: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Deduplicate seed points across fixed, mobile, and simply categories."""

    seen: set[tuple[float, ...]] = set()

    def filter_points(points: FloatArray) -> FloatArray:
        keep_indices: list[int] = []
        for index, point in enumerate(points):
            key = _point_key(point)
            if key not in seen:
                seen.add(key)
                keep_indices.append(index)
        if not keep_indices:
            return np.empty((0, points.shape[1]), dtype=float)
        return points[np.asarray(keep_indices, dtype=int)]

    return tuple(map(filter_points, (fixed_points, mobile_points, simply_points)))  # type: ignore[return-value]


def _classify_dynamic_states(geometry: FemGeometry, points: FloatArray, a0: float) -> np.ndarray:
    """Return mobile or boundary states for the supplied movable points."""

    _ = a0
    if len(points) == 0:
        return np.empty(0, dtype=int)
    states = np.full(len(points), STATE_MOBILE, dtype=int)
    states[geometry.boundary_mask(points, tolerance=BOUNDARY_FUZZ)] = STATE_BOUNDARY
    return states


def _filter_relevant_points(geometry: FemGeometry, points: FloatArray) -> FloatArray:
    """Keep only points that belong to one of the meshed regions."""

    return points if len(points) == 0 else points[geometry.classify_points(points) >= 0]
