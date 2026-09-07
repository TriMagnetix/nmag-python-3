"""Mesh file input and output helpers.

This package provides:
- `load_raw_mesh_with_meshio`: Load meshes via meshio or recognized legacy HDF5
- `save_raw_mesh_with_meshio`: Save meshes to any meshio-supported format
- `load_raw_mesh_from_legacy_nmesh_hdf5`: Direct legacy .nmesh.h5 loader
- `save_raw_mesh_as_legacy_nmesh_hdf5`: Direct legacy .nmesh.h5 writer

For most use cases, use `load_raw_mesh_with_meshio` which handles
both modern formats (via meshio) and recognized legacy .nmesh.h5 files.
"""

from typing import Any

_LEGACY_HDF5_EXPORTS = {
    "is_legacy_nmesh_hdf5",
    "load_raw_mesh_from_legacy_nmesh_hdf5",
    "save_raw_mesh_as_legacy_nmesh_hdf5",
}

_MESHIO_EXPORTS = {
    "load_raw_mesh_with_meshio",
    "save_raw_mesh_with_meshio",
}

__all__ = [
    "is_legacy_nmesh_hdf5",
    "load_raw_mesh_from_legacy_nmesh_hdf5",
    "load_raw_mesh_with_meshio",
    "save_raw_mesh_as_legacy_nmesh_hdf5",
    "save_raw_mesh_with_meshio",
]


def __getattr__(name: str) -> Any:
    if name in _LEGACY_HDF5_EXPORTS:
        from . import legacy_nmesh_hdf5

        value = getattr(legacy_nmesh_hdf5, name)
        globals()[name] = value
        return value

    if name in _MESHIO_EXPORTS:
        from . import meshio_support

        value = getattr(meshio_support, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
