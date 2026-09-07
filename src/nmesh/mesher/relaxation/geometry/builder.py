"""Construction helpers for relaxation geometry bundles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from ....backend import RawMesh
from ....geometry.primitives import Body
from .._constants import BOUNDARY_FUZZ
from .._types import DensityFunction, FloatArray, RegionFunction
from ..density import _compile_density_function
from .model import FemGeometry


def _raw_mesh_points(raw_mesh: RawMesh, dim: int) -> FloatArray:
    """Return mesh points as a validated floating-point array."""

    coords: FloatArray = np.asarray(raw_mesh.points, dtype=np.float64)
    if coords.size == 0:
        return np.empty((0, dim), dtype=float)
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]
    if coords.shape[1] != dim:
        raise ValueError(f"Expected hint mesh points with dimension {dim}, got {coords.shape[1]}")
    return coords


def _dedupe_hint_points(points: FloatArray) -> FloatArray:
    if len(points) == 0:
        return points
    keep_indices: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for index, point in enumerate(points):
        key = tuple(cast(list[float], np.round(np.asarray(point), decimals=10).tolist()))
        if key not in seen:
            seen.add(key)
            keep_indices.append(index)
    return points[np.asarray(keep_indices, dtype=int)]


def _process_bodies_and_hints(
    bodies: list[Body], hints: Sequence[Sequence[object]], dim: int
) -> tuple[list[RegionFunction], list[FloatArray], list[int], list[Body | None]]:
    """Convert bodies and hint meshes into region predicates and hint blocks."""

    body_functions: list[RegionFunction] = []
    piece_hints: list[FloatArray] = []
    region_ids: list[int] = []
    region_bodies: list[Body | None] = []
    for region_id, body in enumerate(bodies, start=1):
        body_functions.append(
            lambda points, member=body: (
                np.asarray(member.evaluate(points), dtype=float) >= -BOUNDARY_FUZZ
            )
        )
        piece_hints.append(np.empty((0, dim), dtype=float))
        region_ids.append(region_id)
        region_bodies.append(body)
    next_region_id = len(region_ids) + 1
    for hint in hints:
        if len(hint) != 2:
            raise ValueError("Each mesh hint must contain a mesh and a body.")
        hint_mesh, hint_body = hint
        if not isinstance(hint_mesh, RawMesh) or not isinstance(hint_body, Body):
            raise TypeError("Mesh hints must contain a RawMesh and Body.")
        hint_points = _raw_mesh_points(hint_mesh, dim)
        if len(hint_points) > 0:
            mask = np.asarray(hint_body.evaluate(hint_points), dtype=float) >= -BOUNDARY_FUZZ
            hint_points = _dedupe_hint_points(hint_points[mask])
        body_functions.append(
            lambda points, member=hint_body: (
                np.asarray(member.evaluate(points), dtype=float) >= -BOUNDARY_FUZZ
            )
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
    """Build ordered region lists, including the exterior region when requested."""

    region_functions: list[RegionFunction] = []
    ordered_region_ids: list[int] = []
    ordered_region_bodies: list[Body | None] = []
    if mesh_exterior:

        def exterior(points: FloatArray) -> np.ndarray:
            bbox_mask = np.all(
                (points >= bbox_min - BOUNDARY_FUZZ) & (points <= bbox_max + BOUNDARY_FUZZ),
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
    hints: Sequence[Sequence[object]],
    *,
    density: str | DensityFunction | None = None,
    mesh_exterior: bool = False,
) -> FemGeometry:
    """Construct the geometry bundle consumed by the Python meshing engine."""

    bbox_min: FloatArray = np.asarray(bounding_box[0], dtype=np.float64)
    bbox_max: FloatArray = np.asarray(bounding_box[1], dtype=np.float64)
    dim = int(len(bbox_min))
    body_functions, piece_hints, region_ids, region_bodies = _process_bodies_and_hints(
        bodies, hints, dim
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
        density_fun=_compile_density_function(density),
        region_ids=tuple(ordered_region_ids),
        region_functions=tuple(region_functions),
        region_bodies=tuple(ordered_region_bodies),
        piece_hints=tuple(piece_hints),
        mesh_exterior=bool(mesh_exterior),
    )
