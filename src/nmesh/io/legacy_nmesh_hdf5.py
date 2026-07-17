"""Support for the legacy ``.nmesh.h5`` mesh format."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
from numpy.typing import NDArray

from ..backend import RawMesh
from .ascii import _build_links, _build_point_regions, _region_volumes

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

_REQUIRED_MESH_DATASETS = frozenset({"points", "simplices", "simplicesregions"})


def _require_group(
    parent: h5py.File | h5py.Group,
    name: str,
    *,
    path: Path,
) -> h5py.Group:
    """Return a required child group after validating its HDF5 node type."""
    node = parent.get(name)
    if node is None:
        raise ValueError(f"{path} is missing the /{name} group")
    if not isinstance(node, h5py.Group):
        raise ValueError(f"{path} has /{name}, but it is not an HDF5 group")
    return node


def _optional_dataset(
    parent: h5py.File | h5py.Group,
    name: str,
    *,
    path: Path,
    hdf5_path: str,
) -> h5py.Dataset | None:
    """Return an optional dataset after validating its HDF5 node type."""
    node = parent.get(name)
    if node is None:
        return None
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"{path} has {hdf5_path}, but it is not an HDF5 dataset")
    return node


def _require_dataset(
    parent: h5py.File | h5py.Group,
    name: str,
    *,
    path: Path,
    hdf5_path: str,
) -> h5py.Dataset:
    """Return a required dataset after validating its HDF5 node type."""
    dataset = _optional_dataset(
        parent,
        name,
        path=path,
        hdf5_path=hdf5_path,
    )
    if dataset is None:
        raise ValueError(
            f"{path} is missing one of /mesh/points, /mesh/simplices, or /mesh/simplicesregions"
        )
    return dataset


def _decode_hdf5_string(value: Any) -> str | None:
    """Convert HDF5 scalar or array string values into plain Python strings."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.shape == () or array.size == 1:
            return _decode_hdf5_string(array.item())
    return str(cast(object, value))


def is_legacy_nmesh_hdf5(path: str | Path) -> bool:
    """Return whether *path* has the recognizable legacy Nmesh HDF5 layout.

    Older Nmesh files identify themselves with ``/etc/filetype == "nmesh"``.
    Some valid historical files omit that optional marker, so the complete set
    of required datasets in ``/mesh`` is also accepted as a signature.
    """
    path = Path(path)
    is_hdf5 = cast(Callable[[str], bool], vars(h5py)["is_hdf5"])
    if not is_hdf5(str(path)):
        return False

    try:
        with h5py.File(str(path), "r") as handle:
            filetype_node = handle.get("etc/filetype")
            if isinstance(filetype_node, h5py.Dataset):
                return _decode_hdf5_string(filetype_node[()]) == "nmesh"

            mesh_group = handle.get("mesh")
            return isinstance(mesh_group, h5py.Group) and _REQUIRED_MESH_DATASETS.issubset(
                mesh_group.keys()
            )
    except OSError:
        return False


def _infer_dim(points: FloatArray, simplices: IntArray) -> int:
    """Infer the mesh dimension from simplex arity or point coordinates."""
    if simplices.ndim == 2 and simplices.shape[1] in (2, 3, 4):
        return simplices.shape[1] - 1
    if points.ndim == 2 and points.shape[1] > 0:
        return int(points.shape[1])
    return 3


def _periodic_points_from_hdf5(
    periodic_raw: IntArray | None,
) -> list[list[int]]:
    """Decode periodic-point rows, dropping the legacy ``-1`` padding markers."""
    if periodic_raw is None:
        return []

    periodic = np.asarray(periodic_raw, dtype=int)
    if periodic.size == 0:
        return []
    if periodic.ndim == 1:
        periodic = periodic[np.newaxis, :]

    return [[idx for idx in row.tolist() if idx != -1] for row in periodic]


def load_raw_mesh_from_legacy_nmesh_hdf5(path: str | Path) -> RawMesh:
    """Load a :class:`RawMesh` from the legacy ``.nmesh.h5`` file layout."""
    path = Path(path)

    with h5py.File(str(path), "r") as handle:
        mesh_group = _require_group(handle, "mesh", path=path)

        filetype_node = _optional_dataset(
            handle,
            "etc/filetype",
            path=path,
            hdf5_path="/etc/filetype",
        )
        filetype = _decode_hdf5_string(filetype_node[()] if filetype_node is not None else None)
        if filetype not in (None, "nmesh"):
            raise ValueError(f"{path} has filetype '{filetype}', expected 'nmesh'")

        points_dataset = _require_dataset(
            mesh_group,
            "points",
            path=path,
            hdf5_path="/mesh/points",
        )
        simplices_dataset = _require_dataset(
            mesh_group,
            "simplices",
            path=path,
            hdf5_path="/mesh/simplices",
        )
        regions_dataset = _require_dataset(
            mesh_group,
            "simplicesregions",
            path=path,
            hdf5_path="/mesh/simplicesregions",
        )
        points: FloatArray = np.asarray(points_dataset[...], dtype=np.float64)
        simplices: IntArray = np.asarray(simplices_dataset[...], dtype=np.int_)
        regions: IntArray = np.asarray(regions_dataset[...], dtype=np.int_).flatten()

        periodic_dataset = _optional_dataset(
            mesh_group,
            "periodicpointindices",
            path=path,
            hdf5_path="/mesh/periodicpointindices",
        )
        permutation_dataset = _optional_dataset(
            mesh_group,
            "permutation",
            path=path,
            hdf5_path="/mesh/permutation",
        )

        periodic_point_indices = _periodic_points_from_hdf5(
            None if periodic_dataset is None else np.asarray(periodic_dataset[...], dtype=np.int_)
        )
        permutation: list[int] = (
            []
            if permutation_dataset is None
            else cast(
                list[int],
                np.asarray(permutation_dataset[...], dtype=np.int_).flatten().tolist(),
            )
        )

    # Validate data consistency
    if len(points) == 0:
        raise ValueError(f"{path} contains no points")
    if len(simplices) == 0:
        raise ValueError(f"{path} contains no simplices")
    if len(regions) != len(simplices):
        raise ValueError(
            f"{path} has mismatched regions ({len(regions)}) and simplices ({len(simplices)})"
        )

    # Validate simplex indices are within bounds
    simplices_list = simplices.tolist()
    max_index = max(max(simplex) for simplex in simplices_list)
    if max_index >= len(points):
        raise ValueError(
            f"{path} has simplex with out-of-bounds point index {max_index} "
            f"(only {len(points)} points)"
        )

    points_list = points.tolist()
    simplices_list = simplices.tolist()
    regions_list = regions.tolist()
    dim = _infer_dim(points, simplices)

    return RawMesh(
        points=points_list,
        simplices=simplices_list,
        regions=regions_list,
        point_regions=_build_point_regions(len(points_list), simplices_list, regions_list),
        links=_build_links(simplices_list),
        region_volumes=_region_volumes(points_list, simplices_list, regions_list, dim),
        periodic_point_indices=periodic_point_indices,
        permutation=permutation,
        dim=dim,
    )


def save_raw_mesh_as_legacy_nmesh_hdf5(path: str | Path, raw_mesh: RawMesh) -> None:
    """Write a :class:`RawMesh` using the legacy ``.nmesh.h5`` file layout."""

    path = Path(path)
    points = np.asarray(raw_mesh.points, dtype=float)
    simplices = np.asarray(raw_mesh.simplices, dtype=np.int32)
    regions = np.asarray(raw_mesh.regions or [1] * len(raw_mesh.simplices), dtype=np.int32)
    if len(points) == 0:
        raise ValueError("Cannot save an empty mesh without points.")
    if len(simplices) == 0:
        raise ValueError("Cannot save an empty mesh without simplices.")
    if len(regions) != len(simplices):
        raise ValueError(
            f"Cannot save mesh with {len(regions)} regions for {len(simplices)} simplices."
        )

    with h5py.File(str(path), "w") as handle:
        _set_pytables_group_attrs(handle, title="")
        etc_group = handle.create_group("etc")
        mesh_group = handle.create_group("mesh")
        _set_pytables_group_attrs(etc_group, title="Configuration and version data")
        _set_pytables_group_attrs(mesh_group, title="Mesh data")

        filetype = etc_group.create_dataset("filetype", data=np.asarray([b"nmesh"]))
        _set_pytables_array_attrs(filetype, title="data file type", version="2.3")
        fileversion = etc_group.create_dataset("fileversion", data=np.asarray([b"1.0"]))
        _set_pytables_array_attrs(fileversion, title="data file type version", version="2.3")
        points_dataset = mesh_group.create_dataset("points", data=points)
        _set_pytables_array_attrs(
            points_dataset,
            title="Positions of mesh nodes (=points)",
        )
        simplices_dataset = mesh_group.create_dataset("simplices", data=simplices)
        _set_pytables_array_attrs(
            simplices_dataset,
            title="Indices of nodes (starting from zero). Each row is one simplex.",
        )
        regions_dataset = mesh_group.create_dataset("simplicesregions", data=regions)
        _set_pytables_array_attrs(
            regions_dataset,
            title="Region ids (one for each simplex).",
        )

        if raw_mesh.periodic_point_indices:
            periodic = _padded_int_rows(raw_mesh.periodic_point_indices, fill=-1)
            periodic_dataset = mesh_group.create_dataset("periodicpointindices", data=periodic)
            _set_pytables_array_attrs(periodic_dataset, title="Periodic point indices")
        if raw_mesh.permutation:
            permutation_dataset = mesh_group.create_dataset(
                "permutation",
                data=np.asarray(raw_mesh.permutation, dtype=np.int32),
            )
            _set_pytables_array_attrs(permutation_dataset, title="Node permutation")


def _padded_int_rows(rows: list[list[int]], *, fill: int) -> np.ndarray:
    width = max(len(row) for row in rows)
    result = np.full((len(rows), width), fill, dtype=np.int32)
    for row_index, row in enumerate(rows):
        result[row_index, : len(row)] = np.asarray(row, dtype=np.int32)
    return result


def _set_pytables_group_attrs(group: h5py.Group | h5py.File, *, title: str) -> None:
    group.attrs["CLASS"] = np.bytes_(b"GROUP")
    group.attrs["TITLE"] = np.bytes_(title.encode("utf-8"))
    group.attrs["VERSION"] = np.bytes_(b"1.0")
    if isinstance(group, h5py.File):
        group.attrs["PYTABLES_FORMAT_VERSION"] = np.bytes_(b"2.0")


def _set_pytables_array_attrs(
    dataset: h5py.Dataset,
    *,
    title: str,
    version: str = "1.0",
) -> None:
    dataset.attrs["CLASS"] = np.bytes_(b"CARRAY")
    dataset.attrs["TITLE"] = np.bytes_(title.encode("utf-8"))
    dataset.attrs["VERSION"] = np.bytes_(version.encode("utf-8"))
