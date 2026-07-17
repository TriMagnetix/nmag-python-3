"""Demag field compatibility façade composed from recovery and auxiliary mixins."""

from .auxiliary import SimulationDemagAuxiliaryMixin
from .recovery import SimulationDemagRecoveryMixin


class SimulationDemagFieldsMixin(
    SimulationDemagRecoveryMixin,
    SimulationDemagAuxiliaryMixin,
):
    """Provide recovered demag fields and FEM/BEM scalar-potential solves."""
