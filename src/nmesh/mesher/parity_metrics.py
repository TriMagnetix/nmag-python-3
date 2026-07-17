"""Geometry metrics for modernization mesh comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import RawMesh


@dataclass(frozen=True, slots=True)
class MeshMetricSummary:
    """Geometry-oriented mesh summary for modernization parity checks."""

    dim: int
    point_count: int
    simplex_count: int
    surface_count: int
    link_count: int
    periodic_group_count: int
    region_counts: tuple[tuple[int, int], ...]
    region_volumes: tuple[tuple[int, float], ...]
    bbox_min: tuple[float, ...]
    bbox_max: tuple[float, ...]
    simplex_measure_min: float
    simplex_measure_mean: float
    simplex_measure_max: float
    edge_length_min: float
    edge_length_mean: float
    edge_length_max: float


def mesh_metric_summary(raw_mesh: RawMesh) -> MeshMetricSummary:
    """Return numeric mesh metrics useful for legacy-vs-modern comparisons."""

    points = np.asarray(raw_mesh.points, dtype=float)
    dim = int(raw_mesh.dim)
    if points.size == 0:
        points = np.empty((0, dim), dtype=float)
    elif points.ndim == 1:
        points = points[np.newaxis, :]

    simplices = np.asarray(raw_mesh.simplices, dtype=int)
    simplex_measures = _simplex_measures(points, simplices, dim)
    edge_lengths = _edge_lengths(points, raw_mesh.links)
    region_counts = tuple(
        sorted(
            (int(region), int(sum(1 for value in raw_mesh.regions if int(value) == int(region))))
            for region in set(raw_mesh.regions)
        )
    )
    region_volumes = tuple(
        sorted(
            zip(
                sorted(set(map(int, raw_mesh.regions))),
                map(float, raw_mesh.region_volumes),
                strict=True,
            )
        )
    )
    return MeshMetricSummary(
        dim=dim,
        point_count=len(raw_mesh.points),
        simplex_count=len(raw_mesh.simplices),
        surface_count=len(raw_mesh.surfaces),
        link_count=len(raw_mesh.links),
        periodic_group_count=len(raw_mesh.periodic_point_indices),
        region_counts=region_counts,
        region_volumes=region_volumes,
        bbox_min=tuple(np.min(points, axis=0).tolist()) if len(points) else (),
        bbox_max=tuple(np.max(points, axis=0).tolist()) if len(points) else (),
        simplex_measure_min=_safe_min(simplex_measures),
        simplex_measure_mean=_safe_mean(simplex_measures),
        simplex_measure_max=_safe_max(simplex_measures),
        edge_length_min=_safe_min(edge_lengths),
        edge_length_mean=_safe_mean(edge_lengths),
        edge_length_max=_safe_max(edge_lengths),
    )
def _simplex_measures(points: np.ndarray, simplices: np.ndarray, dim: int) -> np.ndarray:
    if len(simplices) == 0:
        return np.empty(0, dtype=float)
    if dim == 1:
        return np.abs(points[simplices[:, 1], 0] - points[simplices[:, 0], 0])
    matrices = points[simplices[:, 1:]] - points[simplices[:, [0]]]
    return np.abs(np.linalg.det(matrices)) / float(_factorial(dim))


def _edge_lengths(points: np.ndarray, links: list[tuple[int, int]]) -> np.ndarray:
    if not links:
        return np.empty(0, dtype=float)
    return np.asarray(
        [np.linalg.norm(points[int(left)] - points[int(right)]) for left, right in links],
        dtype=float,
    )


def _factorial(value: int) -> int:
    result = 1
    for item in range(2, value + 1):
        result *= item
    return result


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if len(values) else 0.0


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) else 0.0
