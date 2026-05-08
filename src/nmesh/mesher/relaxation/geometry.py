"""Geometry modeling helpers for the relaxation meshing pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...backend import RawMesh
from ...geometry.primitives import Body
from ._constants import BOUNDARY_FUZZ, DENSITY_EPSILON
from ._types import DensityFunction, FloatArray, RegionFunction
from .density import _compile_density_function


@dataclass(frozen=True, slots=True)
class FemGeometry:
    """Geometry bundle used by the Python meshing engine."""

    dim: int
    bbox_min: FloatArray
    bbox_max: FloatArray
    density_fun: DensityFunction
    region_ids: tuple[int, ...]
    region_functions: tuple[RegionFunction, ...]
    region_bodies: tuple[Body | None, ...]
    piece_hints: tuple[FloatArray, ...]
    mesh_exterior: bool

    def points_in_bbox(self, points: FloatArray) -> np.ndarray:
        """Return a mask showing which points lie inside the bounding box."""

        return np.all(
            (points >= self.bbox_min - BOUNDARY_FUZZ)
            & (points <= self.bbox_max + BOUNDARY_FUZZ),
            axis=1,
        )

    def classify_points(self, points: FloatArray) -> np.ndarray:
        """Return the first matching region id for each point, or -1 outside."""

        if points.size == 0:
            return np.empty(0, dtype=int)

        coords = np.asarray(points, dtype=float)
        assigned = np.full(len(coords), -1, dtype=int)
        bbox_mask = self.points_in_bbox(coords)

        for region_id, region_fun in zip(self.region_ids, self.region_functions):
            mask = bbox_mask & region_fun(coords) & (assigned < 0)
            assigned[mask] = region_id

        return assigned

    def density_at(self, point: FloatArray) -> float:
        """Evaluate the density function with a small positive lower bound."""

        return max(float(self.density_fun(point)), DENSITY_EPSILON)

    def boundary_mask(self, points: FloatArray, tolerance: float) -> np.ndarray:
        """Return a mask for points that lie close to a region or box boundary."""

        if points.size == 0:
            return np.empty(0, dtype=bool)

        coords = np.asarray(points, dtype=float)
        mask = np.zeros(len(coords), dtype=bool)
        for body in self.region_bodies:
            if body is None:
                continue
            mask |= np.abs(np.asarray(body.evaluate(coords), dtype=float)) <= tolerance

        if self.mesh_exterior:
            mask |= np.any(
                np.isclose(coords, self.bbox_min, atol=tolerance, rtol=0.0)
                | np.isclose(coords, self.bbox_max, atol=tolerance, rtol=0.0),
                axis=1,
            )

        return mask & self.points_in_bbox(coords)

    def boundary_distance(self, point: FloatArray) -> float:
        """Return the smallest available distance-like scalar to a boundary."""

        coord = np.asarray(point, dtype=float)
        distances: list[float] = []
        for body in self.region_bodies:
            if body is None:
                continue
            distances.append(abs(float(body.evaluate(coord))))

        if self.mesh_exterior:
            distances.extend(
                min(abs(float(coord[axis] - self.bbox_min[axis])), abs(float(coord[axis] - self.bbox_max[axis])))
                for axis in range(self.dim)
            )

        if not distances:
            return math.inf
        return min(distances)

    def boundary_normal(self, point: FloatArray) -> FloatArray:
        """Estimate a unit normal for the closest relevant boundary."""

        coord = np.asarray(point, dtype=float)
        best_distance = math.inf
        best_normal = np.zeros(self.dim, dtype=float)
        extent = max(float(np.max(self.bbox_max - self.bbox_min)), 1.0)
        epsilon = max(BOUNDARY_FUZZ * 10.0, extent * 1.0e-6)

        for body in self.region_bodies:
            if body is None:
                continue
            distance = abs(float(body.evaluate(coord)))
            if distance > best_distance:
                continue
            gradient = np.zeros(self.dim, dtype=float)
            for axis in range(self.dim):
                offset = np.zeros(self.dim, dtype=float)
                offset[axis] = epsilon
                forward = float(body.evaluate(coord + offset))
                backward = float(body.evaluate(coord - offset))
                gradient[axis] = (forward - backward) / (2.0 * epsilon)
            norm = float(np.linalg.norm(gradient))
            if norm <= DENSITY_EPSILON:
                continue
            best_distance = distance
            best_normal = gradient / norm

        if self.mesh_exterior:
            for axis in range(self.dim):
                dist_min = abs(float(coord[axis] - self.bbox_min[axis]))
                if dist_min < best_distance:
                    best_distance = dist_min
                    best_normal = np.zeros(self.dim, dtype=float)
                    best_normal[axis] = 1.0
                dist_max = abs(float(coord[axis] - self.bbox_max[axis]))
                if dist_max < best_distance:
                    best_distance = dist_max
                    best_normal = np.zeros(self.dim, dtype=float)
                    best_normal[axis] = -1.0

        return best_normal

    def boundary_gradient(self, point: FloatArray, body: Body) -> FloatArray:
        """Estimate the gradient of one implicit body at a point."""

        coord = np.asarray(point, dtype=float)
        gradient = np.zeros(self.dim, dtype=float)
        extent = max(float(np.max(self.bbox_max - self.bbox_min)), 1.0)
        epsilon = max(BOUNDARY_FUZZ * 10.0, extent * 1.0e-6)
        for axis in range(self.dim):
            offset = np.zeros(self.dim, dtype=float)
            offset[axis] = epsilon
            forward = float(body.evaluate(coord + offset))
            backward = float(body.evaluate(coord - offset))
            gradient[axis] = (forward - backward) / (2.0 * epsilon)
        return gradient

    def project_point_to_boundary_from_inside(
        self,
        point: FloatArray,
        *,
        acceptable_fuzz: float,
        max_steps: int,
    ) -> FloatArray:
        """Project an interior point onto implicit boundaries using the legacy correction."""

        coords = np.asarray(point, dtype=float).copy()
        bodies = [body for body in self.region_bodies if body is not None]
        if not bodies:
            return self._project_point_to_box_boundary(coords)

        for _ in range(max(max_steps, 0)):
            violated_body: Body | None = None
            violated_value = 0.0
            for body in bodies:
                value = float(body.evaluate(coords))
                if value > acceptable_fuzz:
                    violated_body = body
                    violated_value = value
                    break

            if violated_body is None:
                return coords

            gradient = self.boundary_gradient(coords, violated_body)
            gradient_norm_sq = float(np.dot(gradient, gradient))
            scale = 1.0e-6 if gradient_norm_sq <= DENSITY_EPSILON else -violated_value / gradient_norm_sq
            coords = coords + scale * gradient

        return coords

    def _project_point_to_box_boundary(self, point: FloatArray) -> FloatArray:
        """Project a point to the nearest bounding-box face."""

        coords = np.asarray(point, dtype=float).copy()
        distances_min = np.abs(coords - self.bbox_min)
        distances_max = np.abs(coords - self.bbox_max)
        min_axis = int(np.argmin(distances_min))
        max_axis = int(np.argmin(distances_max))
        if distances_min[min_axis] <= distances_max[max_axis]:
            coords[min_axis] = self.bbox_min[min_axis]
        else:
            coords[max_axis] = self.bbox_max[max_axis]
        return coords

    def project_segment_to_domain(
        self,
        start: FloatArray,
        end: FloatArray,
        *,
        iterations: int = 18,
    ) -> FloatArray:
        """Project an outside point back into the domain along a segment."""

        lower = np.asarray(start, dtype=float).copy()
        upper = np.asarray(end, dtype=float).copy()
        if self.classify_points(lower.reshape(1, -1))[0] < 0:
            return lower
        if self.classify_points(upper.reshape(1, -1))[0] >= 0:
            return upper

        for _ in range(iterations):
            middle = 0.5 * (lower + upper)
            if self.classify_points(middle.reshape(1, -1))[0] >= 0:
                lower = middle
            else:
                upper = middle

        return lower


def _raw_mesh_points(raw_mesh: RawMesh, dim: int) -> FloatArray:
    """Return mesh points as a validated floating-point array."""

    coords = np.asarray(raw_mesh.points, dtype=float)
    if coords.size == 0:
        return np.empty((0, dim), dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, dim)
    if coords.shape[1] != dim:
        raise ValueError(f"Expected hint mesh points with dimension {dim}, got {coords.shape[1]}")
    return coords.astype(float, copy=False)


def _dedupe_hint_points(points: FloatArray) -> FloatArray:
    """Drop duplicate hint points while preserving first-seen order."""

    if len(points) == 0:
        return points

    keep_indices: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for index, point in enumerate(points):
        key = tuple(np.round(np.asarray(point, dtype=float), decimals=10).tolist())
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(index)
    return points[np.asarray(keep_indices, dtype=int)]


def _process_bodies_and_hints(
    bodies: list[Body],
    hints: list[list[Any]],
    dim: int,
) -> tuple[list[RegionFunction], list[FloatArray], list[int], list[Body | None]]:
    """Convert bodies and hint meshes into region predicates and hint blocks."""

    body_functions: list[RegionFunction] = []
    piece_hints: list[FloatArray] = []
    region_ids: list[int] = []
    region_bodies: list[Body | None] = []
    next_region_id = 1

    for body in bodies:
        body_functions.append(
            lambda points, member=body: np.asarray(member.evaluate(points), dtype=float)
            >= -BOUNDARY_FUZZ
        )
        piece_hints.append(np.empty((0, dim), dtype=float))
        region_ids.append(next_region_id)
        region_bodies.append(body)
        next_region_id += 1

    for hint_mesh, hint_body in hints:
        hint_points = _raw_mesh_points(hint_mesh, dim)
        if len(hint_points) > 0:
            mask = np.asarray(hint_body.evaluate(hint_points), dtype=float) >= -BOUNDARY_FUZZ
            hint_points = _dedupe_hint_points(hint_points[mask])
        body_functions.append(
            lambda points, member=hint_body: np.asarray(member.evaluate(points), dtype=float)
            >= -BOUNDARY_FUZZ
        )
        piece_hints.append(hint_points)
        region_ids.append(next_region_id)
        region_bodies.append(hint_body)
        next_region_id += 1

    return body_functions, piece_hints, region_ids, region_bodies


def _build_region_lists(
    bbox_min: FloatArray,
    bbox_max: FloatArray,
    body_functions: list[RegionFunction],
    region_ids: list[int],
    region_bodies: list[Body | None],
    *,
    mesh_exterior: bool,
) -> tuple[list[RegionFunction], list[int], list[Body | None]]:
    """Build the ordered region lists, including the exterior region when requested."""

    region_functions: list[RegionFunction] = []
    ordered_region_ids: list[int] = []
    ordered_region_bodies: list[Body | None] = []

    if mesh_exterior:

        def exterior(points: FloatArray) -> np.ndarray:
            """Classify points that lie inside the box but outside all bodies."""

            bbox_mask = np.all(
                (points >= bbox_min - BOUNDARY_FUZZ)
                & (points <= bbox_max + BOUNDARY_FUZZ),
                axis=1,
            )
            inside_any = np.zeros(len(points), dtype=bool)
            for body_fun in body_functions:
                inside_any |= body_fun(points)
            return bbox_mask & ~inside_any

        region_functions.append(exterior)
        ordered_region_ids.append(0)
        ordered_region_bodies.append(None)

    region_functions.extend(body_functions)
    ordered_region_ids.extend(region_ids)
    ordered_region_bodies.extend(region_bodies)
    return region_functions, ordered_region_ids, ordered_region_bodies


def fem_geometry_from_bodies(
    bounding_box: tuple[FloatArray, FloatArray],
    bodies: list[Body],
    hints: list[list[Any]],
    *,
    density: str | DensityFunction | None = None,
    mesh_exterior: bool = False,
) -> FemGeometry:
    """Construct the geometry bundle consumed by the Python meshing engine."""

    bbox_min = np.asarray(bounding_box[0], dtype=float)
    bbox_max = np.asarray(bounding_box[1], dtype=float)
    dim = int(len(bbox_min))
    density_fun = _compile_density_function(density)
    body_functions, piece_hints, region_ids, region_bodies = _process_bodies_and_hints(
        bodies,
        hints,
        dim,
    )
    region_functions, ordered_region_ids, ordered_region_bodies = _build_region_lists(
        bbox_min,
        bbox_max,
        body_functions,
        region_ids,
        region_bodies,
        mesh_exterior=mesh_exterior,
    )

    return FemGeometry(
        dim=dim,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        density_fun=density_fun,
        region_ids=tuple(ordered_region_ids),
        region_functions=tuple(region_functions),
        region_bodies=tuple(ordered_region_bodies),
        piece_hints=tuple(piece_hints),
        mesh_exterior=bool(mesh_exterior),
    )
