"""
Geometry and CSG operations for the nmesh meshing library.

This package provides pure-Python geometric primitives and boolean operations,
replacing the OCaml-based geometry layer from mesh.ml and lib1.py.

Modules:
    primitives: Basic geometric shapes (Box, Ellipsoid, Conic, Helix)
    boolean_operations: CSG operations (union, difference, intersect)
    transform: Affine transformations (shift, scale, rotate)

Part of Section 3 of the OCaml-to-Python migration.
"""

from .boolean_operations import *
from .primitives import *
from .transform import *
