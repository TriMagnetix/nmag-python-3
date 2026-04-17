"""Internal constants for the relaxation meshing package."""

from ...utils.constants import BOUNDARY_FUZZ

DENSITY_EPSILON = 1.0e-12
DEFAULT_RNG_SEED = 97
STATE_FIXED = 0
STATE_MOBILE = 1
STATE_SIMPLE = 2

__all__ = [
    "BOUNDARY_FUZZ",
    "DEFAULT_RNG_SEED",
    "DENSITY_EPSILON",
    "STATE_FIXED",
    "STATE_MOBILE",
    "STATE_SIMPLE",
]
