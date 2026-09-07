"""Mesh file IO built around ``meshio`` and legacy Nmesh HDF5 support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import meshio
import numpy as np
from numpy.typing import NDArray

from ..backend import RawMesh
from ..utils.types import ArrayLike
from .legacy_nmesh_hdf5 import (
    is_legacy_nmesh_hdf5,
    load_raw_mesh_from_legacy_nmesh_hdf5,
)

_CELL_TYPE_BY_DIM = {
    1: "line",
    2: "triangle",
    3: "tetra",
}

_DIM_BY_CELL_TYPE = {value: key for key, value in _CELL_TYPE_BY_DIM.items()}

_REGION_DATA_KEYS = ("region", "gmsh:physical", "cell_tags", "gmsh:geometrical")


class _CellBlock(Protocol):
    type: str
    data: NDArray[np.integer]


class _MeshData(Protocol):
    points: NDArray[np.floating]
    cells: Sequence[_CellBlock]
    cell_data: Mapping[str, Sequence[ArrayLike]]


def _read_meshio(path: Path) -> _MeshData:
    read = cast("object", vars(meshio)["read"])
    if not callable(read):
        raise TypeError("meshio.read is not callable")
    return cast(_MeshData, read(path))


def _write_meshio(path: Path, mesh: object) -> None:
    write = cast("object", vars(meshio)["write"])
    if not callable(write):
        raise TypeError("meshio.write is not callable")
    write(path, mesh)


def _cell_type_for(raw_mesh: RawMesh) -> str:
    """Return the meshio cell type that matches the raw mesh topology."""
    if raw_mesh.simplices and len(raw_mesh.simplices) > 0:
        simplex_size = len(raw_mesh.simplices[0])
        if simplex_size == 2:
            return "line"
        if simplex_size == 3:
            return "triangle"
        if simplex_size == 4:
            return "tetra"

    return _CELL_TYPE_BY_DIM.get(raw_mesh.dim, "tetra")


def _regions_from_meshio(
    mesh: _MeshData,
    cell_blocks: Sequence[tuple[int, _CellBlock]],
) -> list[int]:
    """Extract region ids aligned with the selected meshio cell blocks."""
    cell_data = getattr(mesh, "cell_data", {})
    total_count = sum(len(cell_block.data) for _, cell_block in cell_blocks)
    for key in _REGION_DATA_KEYS:
        if key not in cell_data:
            continue

        values_by_block = cell_data[key]
        regions: list[int] = []
        for block_index, cell_block in cell_blocks:
            if block_index >= len(values_by_block):
                raise ValueError(
                    f"Region data {key!r} has no entry for cell block {block_index} "
                    f"({cell_block.type})"
                )

            values = np.asarray(values_by_block[block_index])
            expected_count = len(cell_block.data)
            if values.size != expected_count:
                raise ValueError(
                    f"Region data {key!r} for cell block {block_index} "
                    f"({cell_block.type}) has {values.size} values; "
                    f"expected {expected_count}"
                )
            integer_values = np.asarray(values, dtype=np.int_)
            regions.extend(cast(list[int], integer_values.tolist()))
        return regions

    return [1] * total_count


def _load_raw_mesh_from_meshio(path: Path) -> RawMesh:
    """Load a supported simplex mesh directly through ``meshio``."""
    mesh = _read_meshio(path)
    supported_blocks = [
        (index, cell_block)
        for index, cell_block in enumerate(mesh.cells)
        if cell_block.type in _DIM_BY_CELL_TYPE
    ]
    if not supported_blocks:
        raise ValueError(f"No supported simplex cells found in {path}")

    dim = max(_DIM_BY_CELL_TYPE[cell_block.type] for _, cell_block in supported_blocks)
    selected_blocks = [
        (index, cell_block)
        for index, cell_block in supported_blocks
        if _DIM_BY_CELL_TYPE[cell_block.type] == dim
    ]
    simplices = [
        simplex
        for _, cell_block in selected_blocks
        for simplex in np.asarray(cell_block.data, dtype=int).tolist()
    ]
    return RawMesh(
        points=mesh.points.astype(float).tolist(),
        simplices=simplices,
        regions=_regions_from_meshio(mesh, selected_blocks),
        dim=dim,
    )


def save_raw_mesh_with_meshio(path: str | Path, raw_mesh: RawMesh) -> None:
    """Write a raw mesh to any meshio-supported format."""
    cell_type = _cell_type_for(raw_mesh)
    cells: list[tuple[str, ArrayLike] | meshio.CellBlock] = [
        (cell_type, np.asarray(raw_mesh.simplices, dtype=int))
    ]
    cell_data: dict[str, list[ArrayLike]] | None = None
    if raw_mesh.regions:
        cell_data = {"region": [np.asarray(raw_mesh.regions, dtype=int)]}

    mesh = meshio.Mesh(
        points=np.asarray(raw_mesh.points, dtype=float),
        cells=cells,
        cell_data=cell_data,
    )
    _write_meshio(Path(path), mesh)


def load_raw_mesh_with_meshio(path: str | Path) -> RawMesh:
    """Load a raw mesh through ``meshio`` or the recognized legacy HDF5 loader.

    Args:
        path: Path to the mesh file.

    Returns:
        The loaded mesh as a RawMesh object.

    Raises:
        ValueError: If the file format is not supported or the file is malformed.
        IOError/OSError: If the file cannot be read.
    """
    path = Path(path)
    if is_legacy_nmesh_hdf5(path):
        return load_raw_mesh_from_legacy_nmesh_hdf5(path)

    return _load_raw_mesh_from_meshio(path)
