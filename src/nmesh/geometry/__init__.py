"""
Geometry and CSG operations for the nmesh meshing library.

This package provides pure-Python geometric primitives and boolean operations.

Modules:
    primitives: Basic geometric shapes (Box, Ellipsoid, Conic, Helix)
    boolean_operations: CSG operations (union, difference, intersect)
    transform: Affine transformations (shift, scale, rotate)
"""

from .boolean_operations import difference, intersect, union
from .primitives import (
    Body,
    Box,
    Conic,
    Ellipsoid,
    Helix,
    MeshObject,
    bc_box,
    bc_ellipsoid,
    bc_frustum,
    bc_helix,
)
from .transform import (
    AffineTransform,
    inverse_axis_rotation,
    inverse_plane_rotation,
    inverse_scale,
    inverse_shift,
)
