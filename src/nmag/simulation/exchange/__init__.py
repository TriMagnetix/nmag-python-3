"""Exchange and LLG compatibility façade composed from focused mixins."""

from .coefficients import SimulationExchangeCoefficientMixin
from .fields import SimulationExchangeFieldMixin
from .llg_rhs import SimulationLlgRhsMixin


class SimulationExchangeMixin(
    SimulationExchangeCoefficientMixin,
    SimulationLlgRhsMixin,
    SimulationExchangeFieldMixin,
):
    """Provide material coefficients, LLG kernels, and exchange/STT fields."""
