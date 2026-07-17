"""Tolerance-aware comparisons for mesh metric summaries."""

from __future__ import annotations

from dataclasses import dataclass

from ..backend import RawMesh
from .parity_metrics import MeshMetricSummary, mesh_metric_summary


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
    _compare_exact(
        "region ids", _region_ids(actual_summary), _region_ids(expected_summary), failures
    )
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
        failures.append(f"{name}: actual={actual}, expected={expected}, allowed_delta={allowed:g}")


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
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
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
    for (region, actual_volume), (_expected_region, expected_volume) in zip(
        actual, expected, strict=True
    ):
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
