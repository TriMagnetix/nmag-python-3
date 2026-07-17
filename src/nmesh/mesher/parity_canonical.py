"""Canonical mesh signatures for exact parity comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from ..backend import RawMesh

CoordinateKey: TypeAlias = tuple[str | int, ...]

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
        tuple(sorted(map(int, _point_regions_at(raw_mesh, index)))) for index in canonical_order
    )
    canonical_simplices = tuple(
        sorted(
            (
                int(region),
                tuple(canonical_index[int(point_index)] for point_index in simplex),
            )
            for region, simplex in zip(raw_mesh.regions, raw_mesh.simplices, strict=True)
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
            (
                min(canonical_index[int(left)], canonical_index[int(right)]),
                max(canonical_index[int(left)], canonical_index[int(right)]),
            )
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

