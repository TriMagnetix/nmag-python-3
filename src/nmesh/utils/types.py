"""Common type aliases used throughout the nmesh package."""

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["ArrayLike", "FloatArray", "BoolArray", "IntArray"]

ArrayLike: TypeAlias = Any
FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int_]
