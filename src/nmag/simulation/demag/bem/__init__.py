"""BEM demagnetization compatibility façade composed from focused mixins."""

from .dirichlet import SimulationDemagBemDirichletMixin
from .operator import SimulationDemagBemOperatorMixin


class SimulationDemagBemMixin(
    SimulationDemagBemOperatorMixin,
    SimulationDemagBemDirichletMixin,
):
    """Provide Lindholm BEM construction and Dirichlet extension."""
