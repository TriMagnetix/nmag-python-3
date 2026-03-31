from pathlib import Path
from typing import Any

import meshio
import numpy as np
from numpy.typing import NDArray

from .backend import RawMesh


IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


_CELL_TYPE_BY_DIM: dict[int, str] = {
    1: "line",
    2: "triangle",
    3: "tetra",
}

_DIM_BY_CELL_TYPE: dict[str, int] = {value: key for key, value in _CELL_TYPE_BY_DIM.items()}


def _cell_type_for(raw_mesh: RawMesh) -> str:
    if raw_mesh.simplices:
        simplex_size = len(raw_mesh.simplices[0])
        if simplex_size == 2:
            return "line"
        if simplex_size == 3:
            return "triangle"
        if simplex_size == 4:
            return "tetra"

    return _CELL_TYPE_BY_DIM.get(raw_mesh.dim, "tetra")


def _regions_from_meshio(mesh: meshio.Mesh, cell_type: str, count: int) -> list[int]:
    cell_data_dict: dict[str, dict[str, NDArray[Any]]] = getattr(mesh, "cell_data_dict", {})
    for key in ("region", "gmsh:physical", "cell_tags", "gmsh:geometrical"):
        values_by_type = cell_data_dict.get(key, {})
        if cell_type in values_by_type:
            return values_by_type[cell_type].astype(int).tolist()
    return [1] * count


def save_raw_mesh_with_meshio(path: str | Path, raw_mesh: RawMesh) -> None:
    """Write a raw mesh to any meshio-supported format."""

    cell_type = _cell_type_for(raw_mesh)
    cells: list[tuple[str, IntArray]] = [
        (cell_type, np.asarray(raw_mesh.simplices, dtype=int))
    ]
    cell_data: dict[str, list[IntArray]] | None = None
    if raw_mesh.regions:
        cell_data = {"region": [np.asarray(raw_mesh.regions, dtype=int)]}

    mesh = meshio.Mesh(
        points=np.asarray(raw_mesh.points, dtype=float),
        cells=cells,
        cell_data=cell_data,
    )
    meshio.write(Path(path), mesh)


def load_raw_mesh_with_meshio(path: str | Path) -> RawMesh:
    """Load a raw mesh from any meshio-supported simplex format."""

    mesh = meshio.read(Path(path))
    supported: tuple[str, IntArray] | None = next(
        ((cell_block.type, cell_block.data) for cell_block in mesh.cells if cell_block.type in _DIM_BY_CELL_TYPE),
        None,
    )
    if supported is None:
        raise ValueError(f"No supported simplex cells found in {path}")

    cell_type, simplices = supported
    return RawMesh(
        points=mesh.points.astype(float).tolist(),
        simplices=simplices.astype(int).tolist(),
        regions=_regions_from_meshio(mesh, cell_type, len(simplices)),
        dim=_DIM_BY_CELL_TYPE[cell_type],
    )
