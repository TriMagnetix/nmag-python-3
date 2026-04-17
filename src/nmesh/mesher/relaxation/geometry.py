"""Geometry modeling helpers for the relaxation meshing pipeline."""

from __future__ import annotations

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

    body_functions: list[RegionFunction] = []
    piece_hints: list[FloatArray] = []
    region_ids: list[int] = []
    next_region_id = 1

    for body in bodies:
        body_functions.append(
            lambda points, member=body: np.asarray(member.evaluate(points), dtype=float)
            >= -BOUNDARY_FUZZ
        )
        piece_hints.append(np.empty((0, dim), dtype=float))
        region_ids.append(next_region_id)
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
        next_region_id += 1

    region_functions: list[RegionFunction] = []
    ordered_region_ids: list[int] = []

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

    region_functions.extend(body_functions)
    ordered_region_ids.extend(region_ids)

    return FemGeometry(
        dim=dim,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        density_fun=density_fun,
        region_ids=tuple(ordered_region_ids),
        region_functions=tuple(region_functions),
        piece_hints=tuple(piece_hints),
        mesh_exterior=bool(mesh_exterior),
    )
