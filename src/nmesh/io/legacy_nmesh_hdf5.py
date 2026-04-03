"""Support for the legacy ``.nmesh.h5`` mesh format."""

from pathlib import Path

import h5py
import numpy as np

from ..backend import RawMesh


def _decode_hdf5_string(value) -> str | None:
    """Convert HDF5 scalar or array string values into plain Python strings."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_hdf5_string(value[()])
        if value.size == 1:
            return _decode_hdf5_string(value.reshape(-1)[0])
    return str(value)


def _infer_dim(points: np.ndarray, simplices: np.ndarray) -> int:
    """Infer the mesh dimension from simplex arity or point coordinates."""
    if simplices.ndim == 2 and simplices.shape[1] in (2, 3, 4):
        return simplices.shape[1] - 1
    if points.ndim == 2 and points.shape[1] > 0:
        return int(points.shape[1])
    return 3


def _periodic_points_from_hdf5(periodic_raw: np.ndarray | None) -> list[list[int]]:
    """Decode periodic-point rows, dropping the legacy ``-1`` padding markers."""
    if periodic_raw is None:
        return []

    periodic = np.asarray(periodic_raw, dtype=int)
    if periodic.size == 0:
        return []
    if periodic.ndim == 1:
        periodic = periodic.reshape(1, -1)

    return [
        [int(index) for index in row.tolist() if int(index) != -1]
        for row in periodic
    ]


def load_raw_mesh_from_legacy_nmesh_hdf5(path: str | Path) -> RawMesh:
    """Load a :class:`RawMesh` from the legacy ``.nmesh.h5`` file layout."""
    path = Path(path)

    with h5py.File(path, "r") as handle:
        mesh_group = handle.get("mesh")
        if mesh_group is None:
            raise ValueError(f"{path} is missing the /mesh group")

        filetype_node = handle.get("etc/filetype")
        filetype = _decode_hdf5_string(
            filetype_node[()] if filetype_node is not None else None
        )
        if filetype not in (None, "nmesh"):
            raise ValueError(f"{path} is not a legacy nmesh HDF5 file")

        try:
            points = np.asarray(mesh_group["points"][...], dtype=float)
            simplices = np.asarray(mesh_group["simplices"][...], dtype=int)
            regions = np.asarray(
                mesh_group["simplicesregions"][...], dtype=int
            ).reshape(-1)
        except KeyError as exc:
            raise ValueError(
                f"{path} is missing one of /mesh/points, /mesh/simplices, or "
                "/mesh/simplicesregions"
            ) from exc

        periodic_raw = mesh_group.get("periodicpointindices")
        permutation_raw = mesh_group.get("permutation")

        periodic_point_indices = _periodic_points_from_hdf5(
            None if periodic_raw is None else periodic_raw[...]
        )
        permutation = (
            []
            if permutation_raw is None
            else np.asarray(permutation_raw[...], dtype=int).reshape(-1).tolist()
        )

    return RawMesh(
        points=points.tolist(),
        simplices=simplices.tolist(),
        regions=regions.tolist(),
        periodic_point_indices=periodic_point_indices,
        permutation=permutation,
        dim=_infer_dim(points, simplices),
    )
