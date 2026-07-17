"""Legacy and modern mesh file loading and serialization helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .backend import RawMesh, backend
from .mesh_generation import _as_float_points, _as_int_simplices, _as_region_ids
from .mesh_model import MeshBase

log = logging.getLogger(__name__)


def _is_nmesh_ascii_file(filename: str | Path) -> bool:
    try:
        with Path(filename).open(encoding="utf-8") as stream:
            return stream.readline().startswith("# PYFEM")
    except (OSError, UnicodeDecodeError):
        return False


def _is_nmesh_hdf5_file(filename: str | Path) -> bool:
    from .io.legacy_nmesh_hdf5 import is_legacy_nmesh_hdf5

    return is_legacy_nmesh_hdf5(filename)


def hdf5_mesh_get_permutation(filename: str | Path) -> list[int] | None:
    """Stub for retrieving permutation from HDF5."""
    log.warning("hdf5_mesh_get_permutation: HDF5 support is stubbed.")
    return None


# --- Mesh Classes ---


class MeshFromFile(MeshBase):
    """Loads a mesh from a file."""

    def __init__(
        self,
        filename: str | Path,
        reorder: bool = False,
        distribute: bool = True,
    ) -> None:
        if reorder:
            raise NotImplementedError("Mesh reordering is not implemented.")
        if not distribute:
            raise NotImplementedError("Manual mesh distribution is not implemented.")
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File {filename} not found")

        if _is_nmesh_ascii_file(path):
            from .io.ascii import read_ascii_nmesh

            raw = read_ascii_nmesh(path)
        elif _is_nmesh_hdf5_file(path):
            from .io.legacy_nmesh_hdf5 import load_raw_mesh_from_legacy_nmesh_hdf5

            raw = load_raw_mesh_from_legacy_nmesh_hdf5(path)
        else:
            from .io import load_raw_mesh_with_meshio

            raw = load_raw_mesh_with_meshio(path)

        super().__init__(raw)


class mesh_from_points_and_simplices(MeshBase):
    """Wrapper for backward compatibility."""

    def __init__(
        self,
        points: Sequence[Sequence[float]] | None = None,
        simplices_indices: Sequence[Sequence[int]] | None = None,
        simplices_regions: Sequence[int] | None = None,
        periodic_point_indices: Sequence[Sequence[int]] | None = None,
        initial: int = 0,
        do_reorder: bool = False,
        do_distribute: bool = True,
    ) -> None:
        if do_reorder:
            raise NotImplementedError("Mesh reordering is not implemented.")
        if not do_distribute:
            raise NotImplementedError("Manual mesh distribution is not implemented.")
        points_list = _as_float_points(points)
        simplices_list = _as_int_simplices(simplices_indices)
        if initial == 1:
            simplices_list = [[index - 1 for index in simplex] for simplex in simplices_list]

        raw = backend.mesh_from_points_and_simplices(
            len(points_list[0]) if points_list else 3,
            points_list,
            simplices_list,
            _as_region_ids(simplices_regions),
            [list(map(int, group)) for group in (periodic_point_indices or [])],
            do_reorder,
            do_distribute,
        )
        super().__init__(raw)


def load(
    filename: str | Path,
    reorder: bool = False,
    distribute: bool = True,
) -> MeshFromFile:
    """Utility function to load a mesh."""
    return MeshFromFile(filename, reorder, distribute)


def save(mesh: MeshBase, filename: str | Path) -> None:
    """Alias for mesh.save for backward compatibility."""
    mesh.save(filename)


# --- Utilities ---


def write_mesh(
    mesh_data: RawMesh
    | tuple[
        Sequence[Sequence[float]],
        Sequence[tuple[int, Sequence[int]]],
        Sequence[tuple[int, Sequence[int]]],
    ],
    out: str | Path | TextIO | None = None,
    check: bool = True,
    float_fmt: str = " %.17g",
) -> None:
    """
    Writes mesh data to a file in nmesh format.

    `mesh_data` may be a `RawMesh` or a legacy `(points, simplices, surfaces)` tuple.
    """
    if isinstance(mesh_data, RawMesh):
        points = mesh_data.points
        simplices = list(
            zip(
                mesh_data.regions or [1] * len(mesh_data.simplices),
                mesh_data.simplices,
                strict=True,
            )
        )
        surfaces = list(
            zip(
                [1] * len(mesh_data.surfaces),
                mesh_data.surfaces,
                strict=True,
            )
        )
        periodic_groups = mesh_data.periodic_point_indices
    else:
        points, simplices, surfaces = mesh_data
        periodic_groups = []

    lines = ["# PYFEM mesh file version 1.0"]
    dim = len(points[0]) if points else 0
    lines.append(
        f"# dim = {dim} \t nodes = {len(points)} \t simplices = {len(simplices)} "
        f"\t surfaces = {len(surfaces)} \t periodic = {len(periodic_groups)}"
    )

    lines.append(str(len(points)))
    for p in points:
        lines.append("".join(float_fmt % x for x in p))

    lines.append(str(len(simplices)))
    for body, nodes in simplices:
        lines.append(f" {body} " + " ".join(str(n) for n in nodes))

    lines.append(str(len(surfaces)))
    for body, nodes in surfaces:
        lines.append(f" {body} " + " ".join(str(n) for n in nodes))

    lines.append(str(len(periodic_groups)))
    for group_index, point_indices in enumerate(periodic_groups):
        lines.append(f" {group_index} " + " ".join(str(index) for index in point_indices))

    content = "\n".join(lines) + "\n"

    if out is None:
        print(content)
    elif isinstance(out, (str, Path)):
        Path(out).write_text(content, encoding="utf-8")
    else:
        out.write(content)
