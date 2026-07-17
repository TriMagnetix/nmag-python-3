"""Public nmesh compatibility façade with lazy geometry and driver exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .mesh_generation import Mesh, get_default_meshing_parameters
from .mesh_io import (
    MeshFromFile,
    hdf5_mesh_get_permutation,
    load,
    mesh_from_points_and_simplices,
    save,
    write_mesh,
)
from .mesh_model import MeshBase, Point, Simplex
from .mesh_utilities import (
    generate_1d_mesh,
    generate_1d_mesh_components,
    memory_report,
    outer_corners,
    to_lists,
    tolists,
)

if TYPE_CHECKING:
    pass

_GEOMETRY_EXPORTS = {
    "AffineTransform",
    "Body",
    "Box",
    "Conic",
    "Ellipsoid",
    "Helix",
    "MeshObject",
    "bc_box",
    "bc_ellipsoid",
    "bc_frustum",
    "bc_helix",
    "difference",
    "intersect",
    "inverse_axis_rotation",
    "inverse_plane_rotation",
    "inverse_scale",
    "inverse_shift",
    "union",
}
_MESHER_EXPORTS = {
    "Callback",
    "EngineFunc",
    "MeshEngineCommand",
    "MeshEngineStatus",
    "MeshingParameters",
    "do_every_n_steps_driver",
    "make_mg_gendriver",
}
__all__ = [
    "Mesh",
    "MeshBase",
    "MeshFromFile",
    "Point",
    "Simplex",
    "generate_1d_mesh",
    "generate_1d_mesh_components",
    "get_default_meshing_parameters",
    "hdf5_mesh_get_permutation",
    "load",
    "memory_report",
    "mesh_from_points_and_simplices",
    "outer_corners",
    "save",
    "to_lists",
    "tolists",
    "write_mesh",
    *_GEOMETRY_EXPORTS,
    *_MESHER_EXPORTS,
]


def __getattr__(name: str) -> Any:
    if name in _GEOMETRY_EXPORTS:
        from . import geometry

        value = getattr(geometry, name)
        globals()[name] = value
        return value
    if name == "MeshingParameters":
        from .mesher.meshing_parameters import MeshingParameters

        globals()[name] = MeshingParameters
        return MeshingParameters
    if name in _MESHER_EXPORTS:
        from .mesher import driver

        value = getattr(driver, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
