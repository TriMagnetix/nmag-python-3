"""FEM demagnetization compatibility façade composed from focused mixins."""

from .assembly import SimulationDemagFemAssemblyMixin
from .geometry import SimulationDemagFemGeometryMixin


class SimulationDemagFemMixin(
    SimulationDemagFemAssemblyMixin,
    SimulationDemagFemGeometryMixin,
):
    """Provide FEM assembly, geometry, and boundary-face operations."""
