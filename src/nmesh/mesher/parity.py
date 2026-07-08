"""Canonical mesh signatures for exact parity comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from ..backend import RawMesh

CoordinateKey: TypeAlias = tuple[str | int, ...]

__all__ = [
    "CanonicalMeshSignature",
    "MeshMetricComparison",
    "MeshMetricSummary",
    "assert_canonical_mesh_equal",
    "canonical_mesh_signature",
    "compare_mesh_metrics",
    "mesh_metric_summary",
    "read_ascii_nmesh",
]


@dataclass(frozen=True, slots=True)
class CanonicalMeshSignature:
    """Order-independent mesh signature for legacy parity checks.

    Topology and metadata are compared exactly after canonical point reindexing.
    Coordinate tolerance is opt-in and only affects coordinate keys; leaving it
    as ``None`` uses exact binary floating-point identity.
    """

    dim: int
    point_keys: tuple[CoordinateKey, ...]
    point_regions: tuple[tuple[int, ...], ...]
    simplices: tuple[tuple[int, tuple[int, ...]], ...]
    surfaces: tuple[tuple[int, ...], ...]
    links: tuple[tuple[int, int], ...]
    periodic_groups: tuple[tuple[int, ...], ...]
    region_volumes: tuple[float, ...]


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


@dataclass(frozen=True, slots=True)
class MeshMetricComparison:
    """Result of a metric-level comparison between two meshes."""

    actual: MeshMetricSummary
    expected: MeshMetricSummary
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether all configured metric checks passed."""

        return not self.failures


def canonical_mesh_signature(
    raw_mesh: RawMesh,
    *,
    coordinate_tolerance: float | None = None,
) -> CanonicalMeshSignature:
    """Return a canonical signature suitable for exact parity assertions."""

    point_keys = [_coordinate_key(point, coordinate_tolerance) for point in raw_mesh.points]
    canonical_order = sorted(range(len(point_keys)), key=point_keys.__getitem__)
    canonical_index = {old_index: new_index for new_index, old_index in enumerate(canonical_order)}
    _require_unique_points(point_keys)

    ordered_point_keys = tuple(point_keys[index] for index in canonical_order)
    ordered_point_regions = tuple(
        tuple(sorted(map(int, _point_regions_at(raw_mesh, index))))
        for index in canonical_order
    )
    canonical_simplices = tuple(
        sorted(
            (
                int(region),
                tuple(canonical_index[int(point_index)] for point_index in simplex),
            )
            for region, simplex in zip(raw_mesh.regions, raw_mesh.simplices)
        )
    )
    canonical_surfaces = tuple(
        sorted(
            tuple(sorted(canonical_index[int(point_index)] for point_index in surface))
            for surface in raw_mesh.surfaces
        )
    )
    canonical_links = tuple(
        sorted(
            tuple(sorted((canonical_index[int(left)], canonical_index[int(right)])))
            for left, right in raw_mesh.links
        )
    )
    canonical_periodic = tuple(
        sorted(
            tuple(sorted(canonical_index[int(point_index)] for point_index in group))
            for group in raw_mesh.periodic_point_indices
        )
    )
    return CanonicalMeshSignature(
        dim=int(raw_mesh.dim),
        point_keys=ordered_point_keys,
        point_regions=ordered_point_regions,
        simplices=canonical_simplices,
        surfaces=canonical_surfaces,
        links=canonical_links,
        periodic_groups=canonical_periodic,
        region_volumes=tuple(float(volume) for volume in raw_mesh.region_volumes),
    )


def assert_canonical_mesh_equal(
    actual: RawMesh,
    expected: RawMesh,
    *,
    coordinate_tolerance: float | None = None,
) -> None:
    """Assert exact canonical parity between two meshes."""

    actual_signature = canonical_mesh_signature(
        actual,
        coordinate_tolerance=coordinate_tolerance,
    )
    expected_signature = canonical_mesh_signature(
        expected,
        coordinate_tolerance=coordinate_tolerance,
    )
    if actual_signature != expected_signature:
        raise AssertionError(
            "Canonical mesh signatures differ:\n"
            f"actual={actual_signature!r}\n"
            f"expected={expected_signature!r}"
        )


def _coordinate_key(
    point: list[float],
    coordinate_tolerance: float | None,
) -> CoordinateKey:
    """Return an exact or explicitly quantized coordinate key."""

    if coordinate_tolerance is None:
        return tuple(float(value).hex() for value in point)
    if coordinate_tolerance <= 0.0:
        raise ValueError("coordinate_tolerance must be positive when provided")
    return tuple(int(round(float(value) / coordinate_tolerance)) for value in point)


def _require_unique_points(point_keys: list[CoordinateKey]) -> None:
    """Reject signatures where canonical point identity is ambiguous."""

    if len(set(point_keys)) != len(point_keys):
        raise ValueError(
            "Canonical mesh signature requires unique point coordinates at the "
            "selected coordinate tolerance"
        )


def _point_regions_at(raw_mesh: RawMesh, index: int) -> list[int]:
    """Return point-region membership for a point, tolerating absent metadata."""

    if index < len(raw_mesh.point_regions):
        return raw_mesh.point_regions[index]
    return []


def read_ascii_nmesh(path: str | Path) -> RawMesh:
    """Read the legacy ASCII ``.nmesh``/PYFEM mesh format into ``RawMesh``."""

    source = Path(path)
    lines = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    cursor = 0
    point_count = int(lines[cursor])
    cursor += 1
    points = [_parse_float_row(lines[cursor + index]) for index in range(point_count)]
    cursor += point_count
    dim = len(points[0]) if points else 0

    simplex_count = int(lines[cursor])
    cursor += 1
    simplices: list[list[int]] = []
    regions: list[int] = []
    for _ in range(simplex_count):
        values = _parse_int_row(lines[cursor])
        cursor += 1
        regions.append(values[0])
        simplices.append(values[1:])

    surface_count = int(lines[cursor])
    cursor += 1
    surfaces: list[list[int]] = []
    for _ in range(surface_count):
        values = _parse_int_row(lines[cursor])
        cursor += 1
        surfaces.append(values[-dim:] if dim > 0 else [])

    periodic_groups: list[list[int]] = []
    if cursor < len(lines):
        periodic_count = int(lines[cursor])
        cursor += 1
        for _ in range(periodic_count):
            values = _parse_int_row(lines[cursor])
            cursor += 1
            periodic_groups.append(values[1:])

    return RawMesh(
        points=points,
        simplices=simplices,
        regions=regions,
        point_regions=_build_point_regions(len(points), simplices, regions),
        surfaces=surfaces,
        links=_build_links(simplices),
        region_volumes=_region_volumes(points, simplices, regions, dim),
        periodic_point_indices=periodic_groups,
        permutation=list(range(len(points))),
        dim=dim,
    )


def mesh_metric_summary(raw_mesh: RawMesh) -> MeshMetricSummary:
    """Return numeric mesh metrics useful for legacy-vs-modern comparisons."""

    points = np.asarray(raw_mesh.points, dtype=float)
    dim = int(raw_mesh.dim)
    if points.size == 0:
        points = np.empty((0, dim), dtype=float)
    elif points.ndim == 1:
        points = points.reshape(1, dim)

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
        sorted(zip(sorted(set(map(int, raw_mesh.regions))), map(float, raw_mesh.region_volumes)))
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


def compare_mesh_metrics(
    actual: RawMesh,
    expected: RawMesh,
    *,
    count_relative_tolerance: float = 0.25,
    volume_relative_tolerance: float = 0.05,
    length_relative_tolerance: float = 0.10,
    absolute_tolerance: float = 1.0e-9,
) -> MeshMetricComparison:
    """Compare two meshes with documented metric-level tolerances."""

    actual_summary = mesh_metric_summary(actual)
    expected_summary = mesh_metric_summary(expected)
    failures: list[str] = []
    _compare_exact("dim", actual_summary.dim, expected_summary.dim, failures)
    _compare_relative_count(
        "point_count",
        actual_summary.point_count,
        expected_summary.point_count,
        count_relative_tolerance,
        failures,
    )
    _compare_relative_count(
        "simplex_count",
        actual_summary.simplex_count,
        expected_summary.simplex_count,
        count_relative_tolerance,
        failures,
    )
    _compare_relative_count(
        "surface_count",
        actual_summary.surface_count,
        expected_summary.surface_count,
        count_relative_tolerance,
        failures,
    )
    _compare_exact("region ids", _region_ids(actual_summary), _region_ids(expected_summary), failures)
    _compare_exact(
        "periodic_group_count",
        actual_summary.periodic_group_count,
        expected_summary.periodic_group_count,
        failures,
    )
    _compare_vector(
        "bbox_min",
        actual_summary.bbox_min,
        expected_summary.bbox_min,
        length_relative_tolerance,
        absolute_tolerance,
        failures,
    )
    _compare_vector(
        "bbox_max",
        actual_summary.bbox_max,
        expected_summary.bbox_max,
        length_relative_tolerance,
        absolute_tolerance,
        failures,
    )
    _compare_region_volumes(
        actual_summary.region_volumes,
        expected_summary.region_volumes,
        volume_relative_tolerance,
        absolute_tolerance,
        failures,
    )
    for field_name in (
        "simplex_measure_min",
        "simplex_measure_mean",
        "simplex_measure_max",
        "edge_length_min",
        "edge_length_mean",
        "edge_length_max",
    ):
        _compare_float(
            field_name,
            float(getattr(actual_summary, field_name)),
            float(getattr(expected_summary, field_name)),
            length_relative_tolerance,
            absolute_tolerance,
            failures,
        )
    return MeshMetricComparison(
        actual=actual_summary,
        expected=expected_summary,
        failures=tuple(failures),
    )


def _parse_float_row(line: str) -> list[float]:
    return [float(value) for value in line.split()]


def _parse_int_row(line: str) -> list[int]:
    return [int(value) for value in line.split()]


def _build_point_regions(
    point_count: int,
    simplices: list[list[int]],
    regions: list[int],
) -> list[list[int]]:
    memberships: list[set[int]] = [set() for _ in range(point_count)]
    for simplex, region in zip(simplices, regions):
        for point_index in simplex:
            memberships[int(point_index)].add(int(region))
    return [sorted(group) for group in memberships]


def _build_links(simplices: list[list[int]]) -> list[tuple[int, int]]:
    links: set[tuple[int, int]] = set()
    for simplex in simplices:
        for left_index, left in enumerate(simplex):
            for right in simplex[left_index + 1 :]:
                a = int(left)
                b = int(right)
                links.add((a, b) if a <= b else (b, a))
    return sorted(links)


def _region_volumes(
    points: list[list[float]],
    simplices: list[list[int]],
    regions: list[int],
    dim: int,
) -> list[float]:
    if not simplices:
        return []
    coords = np.asarray(points, dtype=float)
    simplex_array = np.asarray(simplices, dtype=int)
    measures = _simplex_measures(coords, simplex_array, dim)
    totals = {int(region): 0.0 for region in regions}
    for region, measure in zip(regions, measures):
        totals[int(region)] += float(measure)
    return [totals[region] for region in sorted(totals)]


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


def _compare_exact(name: str, actual: object, expected: object, failures: list[str]) -> None:
    if actual != expected:
        failures.append(f"{name}: actual={actual!r}, expected={expected!r}")


def _compare_relative_count(
    name: str,
    actual: int,
    expected: int,
    relative_tolerance: float,
    failures: list[str],
) -> None:
    allowed = max(1.0, abs(float(expected)) * relative_tolerance)
    if abs(float(actual - expected)) > allowed:
        failures.append(
            f"{name}: actual={actual}, expected={expected}, allowed_delta={allowed:g}"
        )


def _compare_vector(
    name: str,
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    relative_tolerance: float,
    absolute_tolerance: float,
    failures: list[str],
) -> None:
    if len(actual) != len(expected):
        failures.append(f"{name}: actual={actual!r}, expected={expected!r}")
        return
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        _compare_float(
            f"{name}[{index}]",
            actual_value,
            expected_value,
            relative_tolerance,
            absolute_tolerance,
            failures,
        )


def _compare_region_volumes(
    actual: tuple[tuple[int, float], ...],
    expected: tuple[tuple[int, float], ...],
    relative_tolerance: float,
    absolute_tolerance: float,
    failures: list[str],
) -> None:
    if tuple(region for region, _volume in actual) != tuple(region for region, _volume in expected):
        failures.append(f"region volumes ids: actual={actual!r}, expected={expected!r}")
        return
    for (region, actual_volume), (_expected_region, expected_volume) in zip(actual, expected):
        _compare_float(
            f"region_volumes[{region}]",
            actual_volume,
            expected_volume,
            relative_tolerance,
            absolute_tolerance,
            failures,
        )


def _compare_float(
    name: str,
    actual: float,
    expected: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    failures: list[str],
) -> None:
    allowed = max(absolute_tolerance, abs(expected) * relative_tolerance)
    if abs(actual - expected) > allowed:
        failures.append(
            f"{name}: actual={actual:g}, expected={expected:g}, allowed_delta={allowed:g}"
        )


def _region_ids(summary: MeshMetricSummary) -> tuple[int, ...]:
    return tuple(region for region, _count in summary.region_counts)
