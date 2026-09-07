"""Public compatibility exports for relaxation geometry."""

from .builder import fem_geometry_from_bodies
from .model import FemGeometry

__all__ = ["FemGeometry", "fem_geometry_from_bodies"]
