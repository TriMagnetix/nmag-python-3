"""Topology extraction and RawMesh assembly helpers."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
from scipy.spatial import Delaunay, QhullError

from ...backend import RawMesh
from ..periodic import build_periodic_groups
from ._constants import BOUNDARY_FUZZ, STATE_BOUNDARY, STATE_MOBILE
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


def _regular_boundary_ratio(dim: int) -> float:
    """Return the normalized volume-order ratio of an ideal regular simplex."""

    if dim <= 0:
        return 1.0
    return ((dim + 1) ** ((dim + 1) / 2.0)) / (math.factorial(dim) * (dim ** (dim / 2.0)))


def _simplex_volume_order_ratio(points: FloatArray, simplices: np.ndarray, dim: int) -> FloatArray:
    """Return simplex volume divided by the local length scale raised to ``dim``."""

    if len(simplices) == 0:
        return np.empty(0, dtype=float)

    centroids = np.mean(points[simplices], axis=1)
    offsets = points[simplices] - centroids[:, None, :]
    max_radius = np.max(np.linalg.norm(offsets, axis=2), axis=1)
    measures = _simplex_measures(points, simplices, dim)
    order_scale = np.maximum(max_radius, BOUNDARY_FUZZ) ** dim
    return measures / order_scale


def _classify_simplices_with_probes(
    points: FloatArray,
    simplices: np.ndarray,
    geometry: FemGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify simplices using centroid and near-vertex probe points."""

    if len(simplices) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=bool)

    centroids = np.mean(points[simplices], axis=1)
    region_ids = geometry.classify_points(centroids)
    offsets = points[simplices] - centroids[:, None, :]
    probes = (centroids[:, None, :] + 0.9 * offsets).reshape(-1, geometry.dim)
    probe_regions = geometry.classify_points(probes).reshape(len(simplices), geometry.dim + 1)
    single_region = (not geometry.mesh_exterior) and (len(set(geometry.region_ids)) <= 1)
    if single_region:
        consistent = (region_ids >= 0) & np.all(probe_regions == region_ids[:, None], axis=1)
    else:
        consistent = (region_ids >= 0) & np.all(probe_regions >= 0, axis=1)
    return region_ids, consistent


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
        movable = (
            np.ones(len(points), dtype=bool)
            if states is None
            else np.isin(states, [STATE_MOBILE, STATE_BOUNDARY])
        )
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
    *,
    states: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
) -> RawMesh:
    """Assemble a ``RawMesh`` from relaxed points and geometry classification."""

    coords = np.asarray(points, dtype=float)
    dim = geometry.dim
    simplices = _triangulate_points(coords, dim, states)

    if len(simplices) > 0:
        region_ids, probe_consistent = _classify_simplices_with_probes(coords, simplices, geometry)
        measures = _simplex_measures(coords, simplices, dim)
        boundary_mask = geometry.boundary_mask(
            coords,
            tolerance=max(BOUNDARY_FUZZ * 10.0, 1.0e-5),
        )
        all_boundary = np.all(boundary_mask[simplices], axis=1)
        boundary_ratio = _simplex_volume_order_ratio(coords, simplices, dim)
        normalized_ratio = boundary_ratio / max(_regular_boundary_ratio(dim), BOUNDARY_FUZZ)
        smallest_allowed_ratio = float(
            (params or {}).get("controller_smallest_allowed_volume_ratio", 1.0)
        )
        flat_boundary = all_boundary & (normalized_ratio < smallest_allowed_ratio)
        keep = probe_consistent & (measures > BOUNDARY_FUZZ) & ~flat_boundary
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
