"""Seed-point preparation helpers for the relaxation meshing pipeline."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._constants import (
    BOUNDARY_FUZZ,
    DENSITY_EPSILON,
    STATE_BOUNDARY,
    STATE_FIXED,
    STATE_MOBILE,
    STATE_SIMPLE,
)
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


def _box_volume(geometry: FemGeometry) -> float:
    """Return the volume of the meshing bounding box."""

    return float(np.prod(np.maximum(geometry.bbox_max - geometry.bbox_min, 0.0)))


def _random_point_in_box(geometry: FemGeometry, rng: np.random.Generator) -> FloatArray:
    """Sample one uniformly random point from the bounding box."""

    return geometry.bbox_min + rng.random(geometry.dim) * (geometry.bbox_max - geometry.bbox_min)


def _sampling_density(geometry: FemGeometry, point: FloatArray) -> float:
    """Evaluate density only for points belonging to the meshed domain."""

    if geometry.classify_points(np.asarray(point, dtype=float).reshape(1, -1))[0] < 0:
        return 0.0
    return geometry.density_at(point)


def _estimate_density_max_and_average(
    geometry: FemGeometry,
    nr_probes: int,
    rng: np.random.Generator,
    *,
    conservative_factor: float = 1.12,
) -> tuple[float, float]:
    """Estimate the legacy random-sampling max and average density values."""

    if nr_probes <= 0:
        return 0.0, 0.0

    first_value = _sampling_density(geometry, _random_point_in_box(geometry, rng))
    max_seen = first_value
    # The legacy estimator intentionally seeded the max with the first probe
    # while leaving it out of the average accumulator.
    density_sum = 0.0
    for _ in range(1, nr_probes):
        value = _sampling_density(geometry, _random_point_in_box(geometry, rng))
        max_seen = max(max_seen, value)
        density_sum += value
    return max_seen * conservative_factor, density_sum / float(nr_probes)


def _sphere_volume(dim: int) -> float:
    """Return the d-dimensional unit sphere volume."""

    return (math.pi ** (0.5 * dim)) / math.gamma(1.0 + 0.5 * dim)


def _sphere_packing_ratio_lattice_type_d(dim: int) -> float:
    """Return the D-lattice sphere-packing ratio used by the legacy seeder."""

    if dim <= 1:
        return 1.0

    lattice_vectors = np.zeros((dim, dim), dtype=float)
    for row in range(dim):
        if row == dim - 1:
            lattice_vectors[row, max(dim - 2, 0) :] = 1.0
        else:
            lattice_vectors[row, row] = 1.0
            lattice_vectors[row, row + 1] = -1.0

    lattice_cell_volume = abs(float(np.linalg.det(lattice_vectors)))
    if lattice_cell_volume <= DENSITY_EPSILON:
        return 1.0
    sphere_radius = 0.5 * math.sqrt(2.0)
    sphere_volume = (sphere_radius**dim) * _sphere_volume(dim)
    return sphere_volume / lattice_cell_volume


def _estimate_initial_point_count(
    geometry: FemGeometry,
    a0: float,
    fixed_points: FloatArray,
    average_density: float,
) -> int:
    """Estimate the number of random initial nodes using the OCaml formula."""

    _ = fixed_points
    node_volume = (
        _sphere_volume(geometry.dim)
        * ((a0 * 0.5 * 0.7) ** geometry.dim)
        / max(_sphere_packing_ratio_lattice_type_d(geometry.dim), DENSITY_EPSILON)
    )
    estimated_integrated_density = average_density * _box_volume(geometry)
    estimated_nodes = estimated_integrated_density / max(node_volume, DENSITY_EPSILON)
    return min(10_000, max(geometry.dim + 1 + 5, int(estimated_nodes)))


def _distribute_points_randomly(
    geometry: FemGeometry,
    nr_points: int,
    max_density: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw initial points with rejection sampling against the density field."""

    if nr_points <= 0 or max_density <= DENSITY_EPSILON:
        return np.empty((0, geometry.dim), dtype=float)

    # OCaml's Array.make evaluates the initial random point once before the
    # rejection loop; keep the same RNG consumption for deterministic parity.
    _ = _random_point_in_box(geometry, rng)
    result = np.empty((nr_points, geometry.dim), dtype=float)
    accepted = 0
    while accepted < nr_points:
        point = _random_point_in_box(geometry, rng)
        if rng.random() * max_density > _sampling_density(geometry, point):
            continue
        result[accepted] = point
        accepted += 1
    return result


def _classify_dynamic_states(geometry: FemGeometry, points: FloatArray, a0: float) -> np.ndarray:
    """Return mobile or boundary states for the supplied movable points."""

    _ = a0
    if len(points) == 0:
        return np.empty(0, dtype=int)
    mask = geometry.boundary_mask(points, tolerance=BOUNDARY_FUZZ)
    states = np.full(len(points), STATE_MOBILE, dtype=int)
    states[mask] = STATE_BOUNDARY
    return states


def _filter_relevant_points(geometry: FemGeometry, points: FloatArray) -> FloatArray:
    """Keep only points that belong to one of the meshed regions."""

    if len(points) == 0:
        return points
    mask = geometry.classify_points(points) >= 0
    return points[mask]


def _select_generated_points(
    geometry: FemGeometry,
    a0: float,
    fixed_points: FloatArray,
    mobile_points: FloatArray,
    simply_points: FloatArray,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> FloatArray:
    """Generate density-weighted random seed points like the legacy mesher."""

    nr_probes = int(params.get("nr_probes_for_determining_volume", 100_000))
    max_density, average_density = _estimate_density_max_and_average(geometry, nr_probes, rng)
    nr_points = _estimate_initial_point_count(geometry, a0, fixed_points, average_density)
    candidates = _distribute_points_randomly(geometry, nr_points, max_density, rng)
    if len(candidates) == 0:
        return np.empty((0, geometry.dim), dtype=float)

    candidate_keys = {_point_key(point) for point in fixed_points}
    candidate_keys.update(_point_key(point) for point in mobile_points)
    candidate_keys.update(_point_key(point) for point in simply_points)

    selected_candidates: list[FloatArray] = []
    for point in candidates:
        key = _point_key(point)
        if key in candidate_keys:
            continue
        if geometry.classify_points(np.asarray(point, dtype=float).reshape(1, -1))[0] < 0:
            continue
        selected_candidates.append(point)
        candidate_keys.add(key)

    if not selected_candidates:
        return np.empty((0, geometry.dim), dtype=float)
    return np.asarray(selected_candidates, dtype=float)


def _collect_hint_points(geometry: FemGeometry) -> FloatArray:
    """Merge, deduplicate, and filter hint points from all geometry pieces."""

    hint_points = [hint_points for hint_points in geometry.piece_hints if len(hint_points) > 0]
    if not hint_points:
        return np.empty((0, geometry.dim), dtype=float)
    hint_block = _dedupe_points(np.vstack(hint_points))
    return _filter_relevant_points(geometry, hint_block)


def _periodic_outer_box_points(
    geometry: FemGeometry,
    a0: float,
    periodic: list[float] | list[bool],
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

        other_axes = [axis for axis in range(geometry.dim) if axis != periodic_axis]
        face_axes: list[FloatArray] = []
        for axis in other_axes:
            extent = float(geometry.bbox_max[axis] - geometry.bbox_min[axis])
            count = max(2, int(math.floor(extent / spacing)) + 1)
            face_axes.append(np.linspace(geometry.bbox_min[axis], geometry.bbox_max[axis], count))

        coordinate_rows = (
            np.zeros((1, 0), dtype=float)
            if not other_axes
            else np.array(np.meshgrid(*face_axes, indexing="ij")).T.reshape(-1, len(other_axes))
        )
        for coordinates in coordinate_rows:
            for side in (geometry.bbox_min[periodic_axis], geometry.bbox_max[periodic_axis]):
                point = np.zeros(geometry.dim, dtype=float)
                point[periodic_axis] = side
                for axis, value in zip(other_axes, coordinates):
                    point[axis] = value
                points.append(point)

    if not points:
        return np.empty((0, geometry.dim), dtype=float)
    return _filter_relevant_points(geometry, _dedupe_points(np.asarray(points, dtype=float)))


def _apply_periodic_fixed_states(
    states: np.ndarray,
    all_points: FloatArray,
    geometry: FemGeometry,
    periodic: list[float] | list[bool],
    a0: float,
) -> None:
    """Mark points on periodic boundaries as fixed."""

    if len(all_points) == 0:
        return

    periodic_flags = [bool(value) for value in periodic]
    periodic_mask = np.zeros(len(all_points), dtype=bool)
    _ = a0
    tolerance = BOUNDARY_FUZZ
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


def _prepare_initial_points(
    geometry: FemGeometry,
    a0: float,
    fixed_points: FloatArray,
    mobile_points: FloatArray,
    simply_points: FloatArray,
    periodic: list[float] | list[bool],
    rng: np.random.Generator,
    params: dict[str, Any] | None = None,
) -> tuple[FloatArray, np.ndarray]:
    """Prepare the initial point cloud and point-state array for relaxation."""

    params = {} if params is None else params
    fixed_points, mobile_points, simply_points = _dedupe_fixed_mobile(
        fixed_points, mobile_points, simply_points
    )
    fixed_points = _filter_relevant_points(geometry, fixed_points)
    mobile_points = _filter_relevant_points(geometry, mobile_points)
    simply_points = _filter_relevant_points(geometry, simply_points)

    if len(simply_points) > 0:
        states = np.full(len(simply_points), STATE_SIMPLE, dtype=int)
        _apply_periodic_fixed_states(states, simply_points, geometry, periodic, a0)
        return simply_points, states

    generated_points = _select_generated_points(
        geometry,
        a0,
        fixed_points,
        mobile_points,
        simply_points,
        rng,
        params,
    ) if len(mobile_points) == 0 else np.empty((0, geometry.dim), dtype=float)
    hint_block = _collect_hint_points(geometry)
    periodic_block = _periodic_outer_box_points(geometry, a0, periodic)

    all_points = np.vstack(
        (
            fixed_points,
            simply_points,
            mobile_points,
            hint_block,
            periodic_block,
            generated_points,
        )
    )

    states = np.concatenate(
        (
            np.full(len(fixed_points), STATE_FIXED, dtype=int),
            np.full(len(simply_points), STATE_SIMPLE, dtype=int),
            _classify_dynamic_states(geometry, mobile_points, a0),
            np.full(len(hint_block), STATE_FIXED, dtype=int),
            np.full(len(periodic_block), STATE_FIXED, dtype=int),
            _classify_dynamic_states(geometry, generated_points, a0),
        )
    )

    _apply_periodic_fixed_states(states, all_points, geometry, periodic, a0)
    return all_points, states
