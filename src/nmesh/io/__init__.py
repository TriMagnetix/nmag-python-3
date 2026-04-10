"""Mesh file input and output helpers.

This package provides:
- `load_raw_mesh_with_meshio`: Load meshes via meshio with .h5 fallback
- `save_raw_mesh_with_meshio`: Save meshes to any meshio-supported format
- `load_raw_mesh_from_legacy_nmesh_hdf5`: Direct legacy .nmesh.h5 loader

For most use cases, use `load_raw_mesh_with_meshio` which handles
both modern formats (via meshio) and legacy .nmesh.h5 files automatically.
"""

from .legacy_nmesh_hdf5 import load_raw_mesh_from_legacy_nmesh_hdf5
from .meshio_support import load_raw_mesh_with_meshio, save_raw_mesh_with_meshio
