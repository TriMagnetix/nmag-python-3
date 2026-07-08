"""Internal type aliases for the relaxation meshing package."""

from typing import Any, Callable, TypeAlias

import numpy as np

from ...utils.types import FloatArray
from ..driver import MeshEngineStatus

DensityFunction: TypeAlias = Callable[[FloatArray], float]
RegionFunction: TypeAlias = Callable[[FloatArray], np.ndarray]
EngineResult: TypeAlias = tuple[MeshEngineStatus, Any]

__all__ = ["DensityFunction", "EngineResult", "FloatArray", "RegionFunction"]
