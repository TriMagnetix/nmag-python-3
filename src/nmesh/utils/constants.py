"""
Common numerical constants used throughout the nmesh package.

These constants provide consistent numerical tolerance values across different
modules for operations like scale validation, division safety checks, and
floating-point comparisons.
"""

__all__ = [
    "EPSILON_SCALE",
    "EPSILON_DIVISION",
]

# Numerical tolerance for scale factor validation (geometry transformations)
# Prevents numerical instability from very small scale factors in affine transforms
EPSILON_SCALE = 1e-12

# Safety epsilon for division operations (mesher computations)
# Prevents division by zero in mesh relaxation and point density calculations
EPSILON_DIVISION = 1e-15
