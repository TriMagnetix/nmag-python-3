"""Mesh-operation compatibility façade composed from focused mixins."""

from .geometry import SimulationMeshGeometryMixin
from .materials import SimulationMeshMaterialMixin
from .probe import SimulationMeshProbeMixin


class SimulationMeshMixin(
    SimulationMeshGeometryMixin,
    SimulationMeshProbeMixin,
    SimulationMeshMaterialMixin,
):
    """Provide mesh caches, probing, and material-aware averaging."""
