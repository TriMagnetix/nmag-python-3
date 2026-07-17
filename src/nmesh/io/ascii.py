"""Reader for the legacy ASCII ``.nmesh``/PYFEM mesh format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..backend import RawMesh


def read_ascii_nmesh(path: str | Path) -> RawMesh:
    """Read a legacy ASCII ``.nmesh``/PYFEM mesh into :class:`RawMesh`."""
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
    for simplex, region in zip(simplices, regions, strict=True):
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
    for region, measure in zip(regions, measures, strict=True):
        totals[int(region)] += float(measure)
    return [totals[region] for region in sorted(totals)]


def _simplex_measures(points: np.ndarray, simplices: np.ndarray, dim: int) -> np.ndarray:
    if len(simplices) == 0:
        return np.empty(0, dtype=float)
    if dim == 1:
        edges = points[simplices[:, 1]] - points[simplices[:, 0]]
        return np.linalg.norm(edges, axis=1)

    edge_matrices = points[simplices[:, 1:]] - points[simplices[:, [0]]]
    gram_matrices = edge_matrices @ np.swapaxes(edge_matrices, 1, 2)
    return np.sqrt(np.abs(np.linalg.det(gram_matrices))) / float(_factorial(dim))


def _factorial(value: int) -> int:
    result = 1
    for item in range(2, value + 1):
        result *= item
    return result
