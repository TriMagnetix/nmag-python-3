"""Public relaxation-engine façade composed from focused implementation phases."""

from .state import RelaxationEngineStateMixin
from .steps import (
    RelaxationEngineStepMixin,
)
from .steps import (
    mesh_bodies_raw as mesh_bodies_raw,  # noqa: F401
)
from .topology import RelaxationEngineTopologyMixin


class RelaxationEngine(
    RelaxationEngineStateMixin,
    RelaxationEngineTopologyMixin,
    RelaxationEngineStepMixin,
):
    """Iterative meshing engine that exposes the legacy driver protocol."""
