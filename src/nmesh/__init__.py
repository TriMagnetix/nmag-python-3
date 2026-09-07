from __future__ import annotations

from typing import Any

from .nmesh import (
    Mesh,
    MeshBase,
    MeshFromFile,
    Point,
    Simplex,
    generate_1d_mesh,
    generate_1d_mesh_components,
    get_default_meshing_parameters,
    hdf5_mesh_get_permutation,
    load,
    memory_report,
    mesh_from_points_and_simplices,
    outer_corners,
    save,
    to_lists,
    tolists,
    write_mesh,
)

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
    "AffineTransform",
    "Body",
    "Box",
    "Callback",
    "Conic",
    "Ellipsoid",
    "EngineFunc",
    "Helix",
    "MeshEngineCommand",
    "MeshEngineStatus",
    "MeshObject",
    "MeshingParameters",
    "bc_box",
    "bc_ellipsoid",
    "bc_frustum",
    "bc_helix",
    "difference",
    "do_every_n_steps_driver",
    "intersect",
    "inverse_axis_rotation",
    "inverse_plane_rotation",
    "inverse_scale",
    "inverse_shift",
    "make_mg_gendriver",
    "union",
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

    if name in {
        "Callback",
        "EngineFunc",
        "MeshEngineCommand",
        "MeshEngineStatus",
        "do_every_n_steps_driver",
        "make_mg_gendriver",
    }:
        from .mesher import driver

        value = getattr(driver, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
