"""Surface-recovery helpers for the relaxation meshing pipeline."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from ._constants import BOUNDARY_FUZZ, DENSITY_EPSILON, STATE_BOUNDARY, STATE_FIXED, STATE_MOBILE, STATE_SIMPLE
from ._types import FloatArray
from .geometry import FemGeometry
from .topology import _simplex_measures


def mirror_surface_recovery_points(
    points: FloatArray,
    states: np.ndarray,
    geometry: FemGeometry,
    a0: float,
    simplices: np.ndarray,
) -> FloatArray:
    """Return legacy mirror/prevention points for poor boundary simplices."""

    dim = geometry.dim
    if dim not in (2, 3) or len(simplices) == 0:
        return np.empty((0, dim), dtype=float)

    simplex_measures = _simplex_measures(points, simplices, dim)
    seen_surfaces: set[tuple[tuple[float, ...], ...]] = set()
    additions: list[FloatArray] = []

    for simplex, simplex_volume in zip(simplices, simplex_measures):
        simplex_points = np.asarray(points[simplex], dtype=float)
        boundary_points, interior_points = _split_boundary_and_interior_points(
            simplex,
            simplex_points,
            states,
        )
        if dim == 2:
            additions.extend(
                _mirror_points_2d(
                    boundary_points,
                    interior_points,
                    simplex_points,
                    float(simplex_volume),
                    geometry,
                    a0,
                    seen_surfaces,
                )
            )
        else:
            additions.extend(
                _prevention_points_3d(
                    boundary_points,
                    simplex_points,
                    float(simplex_volume),
                    geometry,
                    a0,
                    seen_surfaces,
                )
            )

    if not additions:
        return np.empty((0, dim), dtype=float)
    return _dedupe_recovery_points(np.asarray(additions, dtype=float))


def _split_boundary_and_interior_points(
    simplex: np.ndarray,
    simplex_points: FloatArray,
    states: np.ndarray,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Split simplex vertices into boundary-like and mobile interior vertices."""

    boundary_points: list[FloatArray] = []
    interior_points: list[FloatArray] = []
    for local_index, point_index in enumerate(simplex.tolist()):
        state = int(states[int(point_index)])
        point = simplex_points[local_index]
        if state in (STATE_BOUNDARY, STATE_FIXED, STATE_SIMPLE):
            boundary_points.append(point)
        elif state == STATE_MOBILE:
            interior_points.append(point)
    return boundary_points, interior_points


def _mirror_points_2d(
    boundary_points: list[FloatArray],
    interior_points: list[FloatArray],
    simplex_points: FloatArray,
    simplex_volume: float,
    geometry: FemGeometry,
    a0: float,
    seen_surfaces: set[tuple[tuple[float, ...], ...]],
) -> list[FloatArray]:
    """Return mirrored points for the legacy 2D boundary-simplex cases."""

    additions: list[FloatArray] = []
    if len(boundary_points) == 3:
        shifted = list(boundary_points)
        for _ in range(3):
            vertex_point = shifted[0]
            surface_points = np.asarray(shifted[1:], dtype=float)
            if _register_surface(surface_points, seen_surfaces):
                shifted = shifted[1:] + shifted[:1]
                continue
            additions.extend(
                _node_action_2d(surface_points, vertex_point, simplex_points, simplex_volume, geometry, a0)
            )
            shifted = shifted[1:] + shifted[:1]
        return additions

    if len(boundary_points) == 2 and len(interior_points) == 1:
        surface_points = np.asarray(boundary_points, dtype=float)
        if not _register_surface(surface_points, seen_surfaces):
            additions.extend(
                _node_action_2d(
                    surface_points,
                    interior_points[0],
                    simplex_points,
                    simplex_volume,
                    geometry,
                    a0,
                )
            )
    return additions


def _node_action_2d(
    surface_points: FloatArray,
    vertex_point: FloatArray,
    simplex_points: FloatArray,
    simplex_volume: float,
    geometry: FemGeometry,
    a0: float,
) -> list[FloatArray]:
    """Mirror one 2D point when the surface angle and volume tests request it."""

    angle = _solid_angle_2d(surface_points, vertex_point)
    if angle > 0.55 * math.pi and _volume_ratio(simplex_points, simplex_volume, geometry, a0) > 3.0e-2:
        return [_mirror_point(surface_points, vertex_point, 2)]
    return []


def _prevention_points_3d(
    boundary_points: list[FloatArray],
    simplex_points: FloatArray,
    simplex_volume: float,
    geometry: FemGeometry,
    a0: float,
    seen_surfaces: set[tuple[tuple[float, ...], ...]],
) -> list[FloatArray]:
    """Return midpoint prevention points for legacy 3D boundary edges."""

    if len(boundary_points) < 2:
        return []

    additions: list[FloatArray] = []
    for left, right in combinations(boundary_points, 2):
        surface_points = np.asarray([left, right], dtype=float)
        already_seen = _register_surface(surface_points, seen_surfaces)
        if already_seen:
            continue
        middle_point = 0.5 * (surface_points[0] + surface_points[1])
        middle_density = geometry.density_at(middle_point)
        middle_rod_length = a0 / (middle_density ** (1.0 / max(geometry.dim, 1)))
        distance = float(np.linalg.norm(surface_points[0] - surface_points[1]))
        if (
            distance > 1.2 * middle_rod_length
            and _volume_ratio(simplex_points, simplex_volume, geometry, a0) > 3.0e-2
            and _is_outer_region_constrained(geometry, middle_point)
        ):
            additions.append(middle_point)
    return additions


def _surface_key(surface_points: FloatArray) -> tuple[tuple[float, ...], ...]:
    """Return a stable set-like key for a boundary surface."""

    return tuple(sorted(tuple(np.round(point, decimals=10).tolist()) for point in surface_points))


def _register_surface(
    surface_points: FloatArray,
    seen_surfaces: set[tuple[tuple[float, ...], ...]],
) -> bool:
    """Register a surface and return whether it had already been processed."""

    key = _surface_key(surface_points)
    if key in seen_surfaces:
        return True
    seen_surfaces.add(key)
    return False


def _solid_angle_2d(surface_points: FloatArray, vertex_point: FloatArray) -> float:
    """Return the angle subtended by two surface points from a vertex."""

    first = surface_points[0] - vertex_point
    second = surface_points[1] - vertex_point
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom <= DENSITY_EPSILON:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
    return float(math.acos(cosine))


def _mirror_point(surface_points: FloatArray, point: FloatArray, dim: int) -> FloatArray:
    """Mirror a point across a line/plane defined by boundary points."""

    if len(surface_points) == dim:
        origin = surface_points[0]
        directions = surface_points[1:] - origin
        _, _, vh = np.linalg.svd(directions, full_matrices=True)
        normal = np.asarray(vh[-1], dtype=float)
        normal_norm_sq = float(np.dot(normal, normal))
        if normal_norm_sq <= DENSITY_EPSILON:
            return np.asarray(point, dtype=float)
        factor = float(np.dot(point - origin, normal)) / normal_norm_sq
        projection = point - factor * normal
        return np.asarray(2.0 * projection - point, dtype=float)

    if len(surface_points) == dim - 1 and dim == 3:
        start = surface_points[0]
        segment = surface_points[1] - start
        denom = float(np.dot(segment, segment))
        if denom <= DENSITY_EPSILON:
            return np.asarray(point, dtype=float)
        fraction = float(np.dot(point - start, segment)) / denom
        projection = start + fraction * segment
        return np.asarray(2.0 * projection - point, dtype=float)

    return np.asarray(point, dtype=float)


def _volume_ratio(
    simplex_points: FloatArray,
    simplex_volume: float,
    geometry: FemGeometry,
    a0: float,
) -> float:
    """Return the legacy real-to-ideal simplex volume ratio."""

    if simplex_volume <= DENSITY_EPSILON:
        return 0.0
    center = np.mean(simplex_points, axis=0)
    density_here = geometry.density_at(center)
    rod_scaled = a0 / (density_here ** (1.0 / max(geometry.dim, 1)))
    ideal_volume = _regular_simplex_volume(rod_scaled, geometry.dim)
    return abs(simplex_volume / max(ideal_volume, DENSITY_EPSILON))


def _regular_simplex_volume(edge_length: float, dim: int) -> float:
    """Return the volume of a regular simplex with the supplied edge length."""

    numerator = edge_length**dim * math.sqrt(dim + 1.0)
    denominator = math.factorial(dim) * math.sqrt(2.0**dim)
    return numerator / denominator


def _is_outer_region_constrained(geometry: FemGeometry, point: FloatArray) -> bool:
    """Return whether a point satisfies the legacy outer-region boundary check."""

    if not geometry.points_in_bbox(np.asarray(point, dtype=float).reshape(1, -1))[0]:
        return False
    for body in geometry.region_bodies:
        if body is None:
            continue
        if float(body.evaluate(point)) >= BOUNDARY_FUZZ:
            return False
    return True


def _dedupe_recovery_points(points: FloatArray) -> FloatArray:
    """Deduplicate generated recovery points while preserving order."""

    keep_indices: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for index, point in enumerate(points):
        key = tuple(np.round(point, decimals=10).tolist())
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(index)
    return points[np.asarray(keep_indices, dtype=int)]
