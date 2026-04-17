"""Seed-point preparation helpers for the relaxation meshing pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._constants import BOUNDARY_FUZZ, DEFAULT_RNG_SEED, DENSITY_EPSILON, STATE_FIXED, STATE_MOBILE, STATE_SIMPLE
from ._types import FloatArray
from .geometry import FemGeometry


def _as_float_array(points: Any, dim: int | None = None) -> FloatArray:
    """Convert point-like input into a 2D float array with optional dimension validation."""

    if points is None:
        target_dim = 0 if dim is None else dim
        return np.empty((0, target_dim), dtype=float)

    coords = np.asarray(points, dtype=float)
    if coords.size == 0:
        target_dim = 0 if dim is None else dim
        return np.empty((0, target_dim), dtype=float)

    if coords.ndim == 1:
        if dim is None:
            dim = int(coords.shape[0])
        coords = coords.reshape(1, dim)

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
        if key in seen:
            continue
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
        """Return only points not seen in an earlier category."""

        keep_indices: list[int] = []
        for index, point in enumerate(points):
            key = _point_key(point)
            if key in seen:
                continue
            seen.add(key)
            keep_indices.append(index)
        if not keep_indices:
            return np.empty((0, points.shape[1]), dtype=float)
        return points[np.asarray(keep_indices, dtype=int)]

    return (
        filter_points(fixed_points),
        filter_points(mobile_points),
        filter_points(simply_points),
    )


def _grid_candidate_points(geometry: FemGeometry, a0: float) -> FloatArray:
    """Generate deterministic candidate seed points over the bounding box."""

    extents = np.maximum(geometry.bbox_max - geometry.bbox_min, a0)
    sample_points = [geometry.bbox_min, geometry.bbox_max, 0.5 * (geometry.bbox_min + geometry.bbox_max)]
    max_density = max(geometry.density_at(point) for point in sample_points)
    density_scale = min(max_density ** (1.0 / max(geometry.dim, 1)), 2.0)
    step = max(a0 / density_scale, a0 * 0.5)

    counts = np.maximum(np.floor(extents / step).astype(int) + 1, 2)
    candidate_count = int(np.prod(counts))
    if candidate_count > 15_000:
        approximate = min(8_000, max(500, int(np.prod(extents / max(a0, DENSITY_EPSILON)) * 4)))
        rng = np.random.default_rng(DEFAULT_RNG_SEED)
        return rng.uniform(geometry.bbox_min, geometry.bbox_max, size=(approximate, geometry.dim))

    axes = [
        np.linspace(geometry.bbox_min[axis], geometry.bbox_max[axis], int(counts[axis]))
        for axis in range(geometry.dim)
    ]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, geometry.dim)


def _keep_point_for_density(point: FloatArray, density_here: float, rng: np.random.Generator) -> bool:
    """Return whether a candidate point survives density-based thinning."""

    _ = point
    if density_here >= 1.0:
        return True
    return bool(rng.random() <= density_here)


def _prepare_initial_points(
    geometry: FemGeometry,
    a0: float,
    fixed_points: FloatArray,
    mobile_points: FloatArray,
    simply_points: FloatArray,
    periodic: list[float] | list[bool],
    rng: np.random.Generator,
) -> tuple[FloatArray, np.ndarray]:
    """Prepare the initial point cloud and point-state array for relaxation."""

    fixed_points, mobile_points, simply_points = _dedupe_fixed_mobile(
        fixed_points, mobile_points, simply_points
    )

    def filter_relevant(points: FloatArray) -> FloatArray:
        """Keep only points that belong to a meshed region."""

        if len(points) == 0:
            return points
        mask = geometry.classify_points(points) >= 0
        return points[mask]

    fixed_points = filter_relevant(fixed_points)
    mobile_points = filter_relevant(mobile_points)
    simply_points = filter_relevant(simply_points)

    candidates = _grid_candidate_points(geometry, a0)
    region_ids = geometry.classify_points(candidates)
    candidates = candidates[region_ids >= 0]

    if len(candidates) > 0:
        candidate_keys = {_point_key(point) for point in fixed_points}
        candidate_keys.update(_point_key(point) for point in mobile_points)
        candidate_keys.update(_point_key(point) for point in simply_points)

        selected_candidates: list[FloatArray] = []
        for point in candidates:
            key = _point_key(point)
            if key in candidate_keys:
                continue
            density_here = geometry.density_at(point)
            if not _keep_point_for_density(point, density_here, rng):
                continue
            selected_candidates.append(point)
            candidate_keys.add(key)
        generated_points = (
            np.asarray(selected_candidates, dtype=float)
            if selected_candidates
            else np.empty((0, geometry.dim), dtype=float)
        )
    else:
        generated_points = np.empty((0, geometry.dim), dtype=float)

    hint_points = [
        hint_points
        for hint_points in geometry.piece_hints
        if len(hint_points) > 0
    ]
    if hint_points:
        hint_block = _dedupe_points(np.vstack(hint_points))
        hint_block = filter_relevant(hint_block)
    else:
        hint_block = np.empty((0, geometry.dim), dtype=float)

    all_points = np.vstack(
        (
            fixed_points,
            simply_points,
            mobile_points,
            hint_block,
            generated_points,
        )
    )

    states = np.concatenate(
        (
            np.full(len(fixed_points), STATE_FIXED, dtype=int),
            np.full(len(simply_points), STATE_SIMPLE, dtype=int),
            np.full(len(mobile_points), STATE_MOBILE, dtype=int),
            np.full(len(hint_block), STATE_FIXED, dtype=int),
            np.full(len(generated_points), STATE_MOBILE, dtype=int),
        )
    )

    if len(all_points) == 0:
        return all_points, states

    periodic_flags = [bool(value) for value in periodic]
    periodic_mask = np.zeros(len(all_points), dtype=bool)
    tolerance = max(0.05 * a0, BOUNDARY_FUZZ)
    for axis, enabled in enumerate(periodic_flags):
        if not enabled:
            continue
        periodic_mask |= np.isclose(
            all_points[:, axis], geometry.bbox_min[axis], atol=tolerance, rtol=0.0
        )
        periodic_mask |= np.isclose(
            all_points[:, axis], geometry.bbox_max[axis], atol=tolerance, rtol=0.0
        )
    states[periodic_mask] = STATE_FIXED

    return all_points, states
