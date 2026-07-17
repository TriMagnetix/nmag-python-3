"""Small utility constructors and reporting helpers for mesh objects."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from . import utils
from .mesh_io import mesh_from_points_and_simplices
from .mesh_model import MeshBase, Point, Simplex

log = logging.getLogger(__name__)


def memory_report(tag: str) -> None:
    """Reports memory usage."""
    t, vmem, rss = utils.time_vmem_rss()
    log.log(15, f"Memory report: T= {t:f} VMEM= {int(vmem)} KB RSS= {int(rss)} KB {tag}")


# --- Configuration ---


def outer_corners(mesh: MeshBase) -> tuple[Point | None, Point | None]:
    """Determines the bounding box of the mesh nodes."""
    coords = mesh.points
    if not coords:
        return None, None
    transpose = list(zip(*coords, strict=True))
    return [min(t) for t in transpose], [max(t) for t in transpose]


def generate_1d_mesh_components(
    regions: Sequence[tuple[float, float]],
    discretization: float,
) -> tuple[list[Point], list[Simplex], list[int]]:
    """Generates 1D mesh components (points, simplices, regions)."""
    points: list[Point] = []
    simplices: list[Simplex] = []
    regions_ids: list[int] = []
    point_map: dict[float, int] = {}

    def get_idx(value: float) -> int:
        vk = round(value, 8)
        if vk not in point_map:
            point_map[vk] = len(points)
            points.append([float(value)])
        return point_map[vk]

    for rid, (start, end) in enumerate(regions, 1):
        if start > end:
            start, end = end, start
        steps = max(1, int(abs((end - start) / discretization)))
        step = (end - start) / steps
        last = get_idx(start)
        for i in range(1, steps + 1):
            curr = get_idx(start + i * step)
            simplices.append([last, curr])
            regions_ids.append(rid)
            last = curr

    return points, simplices, regions_ids


def generate_1d_mesh(
    regions: Sequence[tuple[float, float]],
    discretization: float,
) -> MeshBase:
    """Generates a 1D mesh with specified regions and step size."""
    pts, simps, regs = generate_1d_mesh_components(regions, discretization)
    return mesh_from_points_and_simplices(pts, simps, regs)


def to_lists(mesh: MeshBase) -> list[object]:
    """Returns mesh data as Python lists."""
    return mesh.to_lists()


tolists = to_lists
