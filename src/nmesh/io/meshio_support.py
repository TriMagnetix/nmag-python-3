"""Mesh file IO built around ``meshio`` with legacy HDF5 fallback support."""

from pathlib import Path

import meshio
import numpy as np

from ..backend import RawMesh
from .legacy_nmesh_hdf5 import load_raw_mesh_from_legacy_nmesh_hdf5


_CELL_TYPE_BY_DIM = {
    1: "line",
    2: "triangle",
    3: "tetra",
}

_DIM_BY_CELL_TYPE = {value: key for key, value in _CELL_TYPE_BY_DIM.items()}


def _cell_type_for(raw_mesh: RawMesh) -> str:
    """Return the meshio cell type that matches the raw mesh topology."""
    if raw_mesh.simplices:
        simplex_size = len(raw_mesh.simplices[0])
        if simplex_size == 2:
            return "line"
        if simplex_size == 3:
            return "triangle"
        if simplex_size == 4:
            return "tetra"

    return _CELL_TYPE_BY_DIM.get(raw_mesh.dim, "tetra")


def _regions_from_meshio(mesh, cell_type: str, count: int) -> list[int]:
    """Extract region ids from common meshio cell-data keys."""
    cell_data_dict = getattr(mesh, "cell_data_dict", {})
    for key in ("region", "gmsh:physical", "cell_tags", "gmsh:geometrical"):
        values_by_type = cell_data_dict.get(key, {})
        if cell_type in values_by_type:
            return values_by_type[cell_type].astype(int).tolist()
    return [1] * count


def _load_raw_mesh_from_meshio(path: Path) -> RawMesh:
    """Load a supported simplex mesh directly through ``meshio``."""
    mesh = meshio.read(path)
    supported = next(
        (
            (cell_block.type, cell_block.data)
            for cell_block in mesh.cells
            if cell_block.type in _DIM_BY_CELL_TYPE
        ),
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


def save_raw_mesh_with_meshio(path: str | Path, raw_mesh: RawMesh) -> None:
    """Write a raw mesh to any meshio-supported format."""
    cell_type = _cell_type_for(raw_mesh)
    cells = [(cell_type, np.asarray(raw_mesh.simplices, dtype=int))]
    cell_data = None
    if raw_mesh.regions:
        cell_data = {"region": [np.asarray(raw_mesh.regions, dtype=int)]}

    mesh = meshio.Mesh(
        points=np.asarray(raw_mesh.points, dtype=float),
        cells=cells,
        cell_data=cell_data,
    )
    meshio.write(Path(path), mesh)


def load_raw_mesh_with_meshio(path: str | Path) -> RawMesh:
    """Load a raw mesh via ``meshio`` or the legacy ``.nmesh.h5`` fallback."""
    path = Path(path)
    meshio_error = None

    try:
        return _load_raw_mesh_from_meshio(path)
    except Exception as exc:
        meshio_error = exc

    if path.suffix.lower() == ".h5":
        try:
            return load_raw_mesh_from_legacy_nmesh_hdf5(path)
        except Exception as legacy_exc:
            raise ValueError(
                f"Unable to read {path} with meshio or the legacy nmesh HDF5 "
                f"loader: meshio error was {meshio_error!r}; legacy loader "
                f"error was {legacy_exc!r}"
            ) from legacy_exc

    raise meshio_error
