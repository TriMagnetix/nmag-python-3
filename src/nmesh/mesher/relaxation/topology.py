"""Topology extraction and RawMesh assembly helpers."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
from scipy.spatial import Delaunay, QhullError

from ...backend import RawMesh
from ..periodic import build_periodic_groups
from ._constants import BOUNDARY_FUZZ, STATE_MOBILE
from ._types import FloatArray
from .geometry import FemGeometry


def _simplex_measures(points: FloatArray, simplices: np.ndarray, dim: int) -> FloatArray:
    """Compute 1D lengths or higher-dimensional simplex volumes."""

    if len(simplices) == 0:
        return np.empty(0, dtype=float)

    if dim == 1:
        return np.abs(points[simplices[:, 1], 0] - points[simplices[:, 0], 0])

    matrices = points[simplices[:, 1:]] - points[simplices[:, [0]]]
    determinants = np.linalg.det(matrices)
    return np.abs(determinants) / math.factorial(dim)


def _surface_faces(simplices: np.ndarray, regions: np.ndarray, dim: int) -> list[list[int]]:
    """Extract boundary faces by counting face ownership across simplices."""

    if len(simplices) == 0:
        return []

    face_map: dict[tuple[int, ...], set[int]] = {}
    face_size = dim
    for simplex, region in zip(simplices, regions):
        for face in combinations(simplex.tolist(), face_size):
            key = tuple(sorted(face))
            face_map.setdefault(key, set()).add(int(region))

    counts: dict[tuple[int, ...], int] = {}
    for simplex in simplices:
        for face in combinations(simplex.tolist(), face_size):
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1

    surfaces: list[list[int]] = []
    for face, count in counts.items():
        if count == 1 or len(face_map.get(face, set())) > 1:
            surfaces.append(list(face))
    return surfaces


def _unique_links(simplices: np.ndarray) -> list[tuple[int, int]]:
    """Return the sorted unique undirected edges induced by the simplices."""

    links: set[tuple[int, int]] = set()
    for simplex in simplices:
        for edge_start, edge_end in combinations(simplex.tolist(), 2):
            start = int(edge_start)
            end = int(edge_end)
            if start <= end:
                links.add((start, end))
            else:
                links.add((end, start))
    return sorted(links)


def _point_regions(point_count: int, simplices: np.ndarray, regions: np.ndarray) -> list[list[int]]:
    """Build the region-membership list for each mesh point."""

    memberships: list[set[int]] = [set() for _ in range(point_count)]
    for simplex, region in zip(simplices, regions):
        for point_index in simplex:
            memberships[int(point_index)].add(int(region))
    return [sorted(group) for group in memberships]


def _region_volumes(region_ids: np.ndarray, measures: FloatArray) -> list[float]:
    """Aggregate simplex measures into per-region total volumes."""

    if len(region_ids) == 0:
        return []

    order = sorted({int(region) for region in region_ids})
    totals = {region: 0.0 for region in order}
    for region, measure in zip(region_ids, measures):
        totals[int(region)] += float(measure)
    return [totals[region] for region in order]


def _triangulate_points(points: FloatArray, dim: int, states: np.ndarray | None = None) -> np.ndarray:
    """Triangulate the point cloud, retrying with light jitter for degenerate inputs."""

    if len(points) < dim + 1:
        return np.empty((0, dim + 1), dtype=int)

    if dim == 1:
        order = np.argsort(points[:, 0], kind="mergesort")
        return np.column_stack((order[:-1], order[1:])).astype(int)

    try:
        return Delaunay(points).simplices.astype(int, copy=False)
    except QhullError:
        jittered = np.array(points, copy=True)
        movable = np.ones(len(points), dtype=bool) if states is None else states == STATE_MOBILE
        if np.any(movable):
            amplitudes = np.linspace(1.0e-9, 1.0e-8, np.count_nonzero(movable))
            offsets = np.column_stack(
                [
                    amplitudes * np.sin(np.arange(1, len(amplitudes) + 1) * (axis + 1))
                    for axis in range(dim)
                ]
            )
            jittered[movable] += offsets
        try:
            return Delaunay(jittered).simplices.astype(int, copy=False)
        except QhullError:
            return np.empty((0, dim + 1), dtype=int)


def assemble_raw_mesh(
    points: FloatArray,
    geometry: FemGeometry,
    periodic: list[float] | list[bool],
) -> RawMesh:
    """Assemble a ``RawMesh`` from relaxed points and geometry classification."""

    coords = np.asarray(points, dtype=float)
    dim = geometry.dim
    simplices = _triangulate_points(coords, dim)

    if len(simplices) > 0:
        centroids = np.mean(coords[simplices], axis=1)
        region_ids = geometry.classify_points(centroids)
        measures = _simplex_measures(coords, simplices, dim)
        keep = (region_ids >= 0) & (measures > BOUNDARY_FUZZ)
        simplices = simplices[keep]
        region_ids = region_ids[keep]
        measures = measures[keep]
    else:
        region_ids = np.empty(0, dtype=int)
        measures = np.empty(0, dtype=float)

    surfaces = _surface_faces(simplices, region_ids, dim)
    links = _unique_links(simplices)
    point_regions = _point_regions(len(coords), simplices, region_ids)
    region_volumes = _region_volumes(region_ids, measures)
    periodic_groups = build_periodic_groups(
        coords,
        geometry.bbox_min,
        geometry.bbox_max,
        periodic,
        tolerance=BOUNDARY_FUZZ,
    )

    return RawMesh(
        points=coords.tolist(),
        simplices=simplices.tolist(),
        regions=region_ids.astype(int).tolist(),
        point_regions=point_regions,
        surfaces=surfaces,
        links=links,
        region_volumes=region_volumes,
        periodic_point_indices=periodic_groups,
        permutation=list(range(len(coords))),
        dim=dim,
    )


def _callback_mesh_info(raw_mesh: RawMesh) -> list[list[Any]]:
    """Build the legacy-style callback payload expected by old consumers."""

    simplices = [
        [simplex, (([], 0.0), ([], 0.0), region)]
        for simplex, region in zip(raw_mesh.simplices, raw_mesh.regions)
    ]
    surfaces = [
        [surface, (([], 0.0), ([], 0.0), 1)]
        for surface in raw_mesh.surfaces
    ]
    return [
        ["COORDS", "Node coordinates", raw_mesh.points],
        ["LINKS", "Mesh links", raw_mesh.links],
        ["POINT-BODIES", "Point region memberships", raw_mesh.point_regions],
        ["SIMPLICES", "Simplex connectivity", simplices],
        ["SURFACES", "Surface connectivity", surfaces],
    ]
