"""Export modern Nmag spatial snapshots to VTK-compatible mesh files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import h5py
import meshio
import numpy as np

import nmesh


def _dataset_names(handle: h5py.File) -> list[str]:
    fields = handle.get("fields")
    if not isinstance(fields, h5py.Group):
        raise ValueError("Snapshot is missing the /fields group.")
    return sorted(str(name) for name in fields.keys())


def _read_points(handle: h5py.File) -> np.ndarray:
    try:
        points = np.asarray(handle["mesh/points"], dtype=float)
    except KeyError as exc:
        raise ValueError("Snapshot is missing /mesh/points.") from exc
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Snapshot points must have shape (N, 3), got {points.shape}.")
    return points


def _read_topology(
    handle: h5py.File,
    mesh_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if "mesh/simplices" in handle:
        simplices = np.asarray(handle["mesh/simplices"], dtype=np.int64)
        regions = (
            np.asarray(handle["mesh/simplicesregions"], dtype=np.int64)
            if "mesh/simplicesregions" in handle
            else None
        )
    elif mesh_path is not None:
        source_mesh = nmesh.load(mesh_path)
        simplices = np.asarray(source_mesh.simplices, dtype=np.int64)
        regions = np.asarray(source_mesh.regions, dtype=np.int64)
    else:
        raise ValueError(
            "Snapshot contains point coordinates but no cell topology; provide --mesh."
        )

    if simplices.ndim != 2 or simplices.shape[1] != 4:
        raise ValueError(f"VTK export requires tetrahedral cells, got {simplices.shape}.")
    if regions is not None and len(regions) != len(simplices):
        raise ValueError("Mesh region data does not match the number of cells.")
    return simplices, regions, _read_points(handle)


def export_vtk(
    snapshot_path: str | Path,
    output_path: str | Path,
    *,
    mesh_path: str | Path | None = None,
    fields: Iterable[str] = ("m",),
) -> Path:
    """Export selected point fields from a modern Nmag HDF5 snapshot."""

    snapshot = Path(snapshot_path)
    output = Path(output_path)
    selected_fields = tuple(dict.fromkeys(fields))
    if not selected_fields:
        raise ValueError("At least one point field must be selected.")

    with h5py.File(str(snapshot), "r") as handle:
        points = _read_points(handle)
        simplices, regions, _ = _read_topology(handle, mesh_path)
        if np.any(simplices < 0) or np.any(simplices >= len(points)):
            raise ValueError("Mesh cell indices are outside the snapshot point range.")

        point_data: dict[str, np.ndarray] = {}
        field_group = handle.get("fields")
        if not isinstance(field_group, h5py.Group):
            raise ValueError("Snapshot is missing the /fields group.")
        for field_name in selected_fields:
            if field_name not in field_group:
                available = ", ".join(_dataset_names(handle))
                raise KeyError(f"Field {field_name!r} is not present; available fields: {available}")
            values = np.asarray(field_group[field_name])
            if values.ndim not in (1, 2) or values.shape[0] != len(points):
                raise ValueError(
                    f"Field {field_name!r} must have one value per point, got {values.shape}."
                )
            if values.ndim == 2 and values.shape[1] not in (1, 3):
                raise ValueError(
                    f"Field {field_name!r} must be scalar or three-component vector, "
                    f"got {values.shape}."
                )
            point_data[field_name] = np.asarray(values, dtype=float)

    cell_data: dict[str, list[np.ndarray]] = {}
    if regions is not None:
        cell_data["region"] = [regions]
    mesh = meshio.Mesh(
        points=points,
        cells=[("tetra", simplices)],
        point_data=point_data,
        cell_data=cell_data or None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(output, mesh)
    return output


def resolve_snapshot(path: str | Path) -> Path:
    """Resolve a snapshot path or a simulation base name."""

    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.suffix == "":
        suffixed = candidate.with_name(f"{candidate.name}_dat.h5")
        if suffixed.is_file():
            return suffixed
    raise FileNotFoundError(f"Nmag snapshot does not exist: {candidate}")
