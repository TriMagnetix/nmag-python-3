"""
Common numerical constants used throughout the nmesh package.

These constants provide consistent numerical tolerance values across different
modules for operations like scale validation, division safety checks, and
floating-point comparisons.
"""

__all__ = [
    "BOUNDARY_FUZZ",
    "MIN_ABS_SCALE_FACTOR",
    "MIN_DIVISION_MAGNITUDE",
]

# Default tolerance used when testing whether points lie on or near boundaries.
BOUNDARY_FUZZ = 1e-6

# Smallest allowed absolute scale factor for affine transformations.
MIN_ABS_SCALE_FACTOR = 1e-12

# Smallest divisor magnitude accepted before force-law clamping kicks in.
MIN_DIVISION_MAGNITUDE = 1e-15
