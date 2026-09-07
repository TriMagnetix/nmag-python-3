"""Focused field-operation mixins exposed through one compatibility façade."""

from .arrays import SimulationFieldArrayMixin
from .availability import SimulationFieldAvailabilityMixin
from .averages import SimulationFieldAverageMixin
from .derived import SimulationFieldDerivedMixin
from .maxangle import SimulationFieldMaxangleMixin
from .probes import SimulationFieldProbeMixin


class SimulationFieldMixin(
    SimulationFieldAvailabilityMixin,
    SimulationFieldAverageMixin,
    SimulationFieldMaxangleMixin,
    SimulationFieldProbeMixin,
    SimulationFieldArrayMixin,
    SimulationFieldDerivedMixin,
):
    """Compose the public field API from focused implementation mixins."""
