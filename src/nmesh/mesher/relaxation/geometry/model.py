"""Runtime geometry queries for the relaxation meshing pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ....geometry.primitives import Body
from .._constants import BOUNDARY_FUZZ, DENSITY_EPSILON
from .._types import DensityFunction, FloatArray, RegionFunction


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
            (points >= self.bbox_min - BOUNDARY_FUZZ) & (points <= self.bbox_max + BOUNDARY_FUZZ),
            axis=1,
        )

    def classify_points(self, points: FloatArray) -> np.ndarray:
        """Return the first matching region id for each point, or -1 outside."""

        if points.size == 0:
            return np.empty(0, dtype=int)
        coords = np.asarray(points, dtype=float)
        assigned = np.full(len(coords), -1, dtype=int)
        bbox_mask = self.points_in_bbox(coords)
        for region_id, region_fun in zip(self.region_ids, self.region_functions, strict=True):
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
            if body is not None:
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
        distances = [
            abs(float(body.evaluate(coord))) for body in self.region_bodies if body is not None
        ]
        if self.mesh_exterior:
            distances.extend(
                min(
                    abs(float(coord[axis] - self.bbox_min[axis])),
                    abs(float(coord[axis] - self.bbox_max[axis])),
                )
                for axis in range(self.dim)
            )
        return min(distances) if distances else math.inf

    def boundary_normal(self, point: FloatArray) -> FloatArray:
        """Estimate a unit normal for the closest relevant boundary."""

        coord = np.asarray(point, dtype=float)
        best_distance = math.inf
        best_normal = np.zeros(self.dim, dtype=float)
        for body in self.region_bodies:
            if body is None:
                continue
            distance = abs(float(body.evaluate(coord)))
            if distance > best_distance:
                continue
            gradient = self.boundary_gradient(coord, body)
            norm = float(np.linalg.norm(gradient))
            if norm > DENSITY_EPSILON:
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
            gradient[axis] = (
                float(body.evaluate(coord + offset)) - float(body.evaluate(coord - offset))
            ) / (2.0 * epsilon)
        return gradient

    def project_point_to_boundary_from_inside(
        self, point: FloatArray, *, acceptable_fuzz: float, max_steps: int
    ) -> FloatArray:
        """Project an interior point onto implicit boundaries using the legacy correction."""

        coords: FloatArray = np.array(point, dtype=np.float64, copy=True)
        bodies = [body for body in self.region_bodies if body is not None]
        if not bodies:
            return self._project_point_to_box_boundary(coords)
        for _ in range(max(max_steps, 0)):
            violated_body: Body | None = None
            violated_value = 0.0
            for body in bodies:
                value = float(body.evaluate(coords))
                if value > acceptable_fuzz:
                    violated_body, violated_value = body, value
                    break
            if violated_body is None:
                return coords
            gradient = self.boundary_gradient(coords, violated_body)
            gradient_norm_sq = float(np.dot(gradient, gradient))
            scale = (
                1.0e-6
                if gradient_norm_sq <= DENSITY_EPSILON
                else -violated_value / gradient_norm_sq
            )
            coords = coords + scale * gradient
        return coords

    def _project_point_to_box_boundary(self, point: FloatArray) -> FloatArray:
        coords: FloatArray = np.array(point, dtype=np.float64, copy=True)
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
        self, start: FloatArray, end: FloatArray, *, iterations: int = 18
    ) -> FloatArray:
        """Project an outside point back into the domain along a segment."""

        lower: FloatArray = np.array(start, dtype=np.float64, copy=True)
        upper: FloatArray = np.array(end, dtype=np.float64, copy=True)
        if self.classify_points(lower[np.newaxis, :])[0] < 0:
            return lower
        if self.classify_points(upper[np.newaxis, :])[0] >= 0:
            return upper
        for _ in range(iterations):
            middle = 0.5 * (lower + upper)
            if self.classify_points(middle[np.newaxis, :])[0] >= 0:
                lower = middle
            else:
                upper = middle
        return lower
