"""Density-weighted random sampling helpers for relaxation seed points."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .._constants import DENSITY_EPSILON
from .._types import FloatArray
from ..geometry import FemGeometry
from .points import _dedupe_points, _filter_relevant_points, _point_key


def _box_volume(geometry: FemGeometry) -> float:
    return float(np.prod(np.maximum(geometry.bbox_max - geometry.bbox_min, 0.0)))


def _random_point_in_box(geometry: FemGeometry, rng: np.random.Generator) -> FloatArray:
    return geometry.bbox_min + rng.random(geometry.dim) * (geometry.bbox_max - geometry.bbox_min)


def _sampling_density(geometry: FemGeometry, point: FloatArray) -> float:
    if geometry.classify_points(np.asarray(point, dtype=float)[np.newaxis, :])[0] < 0:
        return 0.0
    return geometry.density_at(point)


def _estimate_density_max_and_average(
    geometry: FemGeometry,
    nr_probes: int,
    rng: np.random.Generator,
    *,
    conservative_factor: float = 1.12,
) -> tuple[float, float]:
    """Estimate the legacy random-sampling maximum and average density."""

    if nr_probes <= 0:
        return 0.0, 0.0
    max_seen = _sampling_density(geometry, _random_point_in_box(geometry, rng))
    density_sum = 0.0
    for _ in range(1, nr_probes):
        value = _sampling_density(geometry, _random_point_in_box(geometry, rng))
        max_seen = max(max_seen, value)
        density_sum += value
    return max_seen * conservative_factor, density_sum / float(nr_probes)


def _sphere_volume(dim: int) -> float:
    return (math.pi ** (0.5 * dim)) / math.gamma(1.0 + 0.5 * dim)


def _sphere_packing_ratio_lattice_type_d(dim: int) -> float:
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
    return ((0.5 * math.sqrt(2.0)) ** dim) * _sphere_volume(dim) / lattice_cell_volume


def _estimate_initial_point_count(
    geometry: FemGeometry, a0: float, fixed_points: FloatArray, average_density: float
) -> int:
    _ = fixed_points
    node_volume = (
        _sphere_volume(geometry.dim)
        * ((a0 * 0.5 * 0.7) ** geometry.dim)
        / max(_sphere_packing_ratio_lattice_type_d(geometry.dim), DENSITY_EPSILON)
    )
    estimated_nodes = average_density * _box_volume(geometry) / max(node_volume, DENSITY_EPSILON)
    return min(10_000, max(geometry.dim + 1 + 5, int(estimated_nodes)))


def _distribute_points_randomly(
    geometry: FemGeometry, nr_points: int, max_density: float, rng: np.random.Generator
) -> FloatArray:
    if nr_points <= 0 or max_density <= DENSITY_EPSILON:
        return np.empty((0, geometry.dim), dtype=float)
    _ = _random_point_in_box(geometry, rng)
    result = np.empty((nr_points, geometry.dim), dtype=float)
    accepted = 0
    while accepted < nr_points:
        point = _random_point_in_box(geometry, rng)
        if rng.random() * max_density <= _sampling_density(geometry, point):
            result[accepted] = point
            accepted += 1
    return result


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

    max_density, average_density = _estimate_density_max_and_average(
        geometry, int(params.get("nr_probes_for_determining_volume", 100_000)), rng
    )
    candidates = _distribute_points_randomly(
        geometry,
        _estimate_initial_point_count(geometry, a0, fixed_points, average_density),
        max_density,
        rng,
    )
    candidate_keys = {_point_key(point) for point in fixed_points}
    candidate_keys.update(_point_key(point) for point in mobile_points)
    candidate_keys.update(_point_key(point) for point in simply_points)
    selected: list[FloatArray] = []
    for point in candidates:
        key = _point_key(point)
        if (
            key not in candidate_keys
            and geometry.classify_points(np.asarray(point, dtype=float)[np.newaxis, :])[0] >= 0
        ):
            selected.append(point)
            candidate_keys.add(key)
    return (
        np.asarray(selected, dtype=float) if selected else np.empty((0, geometry.dim), dtype=float)
    )


def _collect_hint_points(geometry: FemGeometry) -> FloatArray:
    """Merge, deduplicate, and filter hint points from all geometry pieces."""

    hint_points = [points for points in geometry.piece_hints if len(points) > 0]
    if not hint_points:
        return np.empty((0, geometry.dim), dtype=float)
    return _filter_relevant_points(geometry, _dedupe_points(np.vstack(hint_points)))
