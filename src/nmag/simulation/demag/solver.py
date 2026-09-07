from __future__ import annotations

from ..exchange import SimulationExchangeMixin
from .bem import SimulationDemagBemMixin
from .fem import SimulationDemagFemMixin
from .fields import SimulationDemagFieldsMixin
from .fields.probe import SimulationDemagProbeMixin


class SimulationDemagMixin(
    SimulationExchangeMixin,
    SimulationDemagFieldsMixin,
    SimulationDemagFemMixin,
    SimulationDemagBemMixin,
    SimulationDemagProbeMixin,
):
    """Compose the exchange, demag field, FEM, BEM, and probe implementations."""
