from nmesh.backend import nmesh_backend as backend
from nmesh.base import (
    MeshBase, Mesh, MeshFromFile, mesh_from_points_and_simplices
)
from nmesh.geometry import (
    MeshObject, Box, Ellipsoid, Conic, Helix, union, difference, intersect
)
from nmesh.features import MeshingParameters, get_default_meshing_parameters
from nmesh.utils import (
    outer_corners, generate_1d_mesh_components, generate_1d_mesh, write_mesh, memory_report
)

def load(filename, reorder=False, do_distribute=True):
    """Load nmesh file with name filename."""
    import os
    if not os.path.exists(filename):
        raise ValueError(f"file '{filename}' does not exist")
    
    # Simple extension based check
    if filename.lower().endswith('.h5'):
        # For now, we don't have HDF5 support implemented
        raise NotImplementedError("HDF5 mesh loading is not yet implemented in Python 3 version.")
    
    return MeshFromFile(filename, reorder=reorder, distribute=do_distribute)

def save(mesh, filename):
    """Alias for mesh.save for backward compatibility."""
    mesh.save(filename)

# --- Exception Aliases ---
NmeshUserError = ValueError
NmeshIOError = IOError
NmeshStandardError = RuntimeError
NMeshTypeError = TypeError

# --- Compatibility Aliases ---
tolists = lambda mesh: mesh.to_lists()
mesh_1d = generate_1d_mesh_components
