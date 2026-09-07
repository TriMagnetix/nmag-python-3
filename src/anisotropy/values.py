"""Validation and scalar conversion helpers for anisotropy models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from si.physical import SI

FloatArray = NDArray[np.float64]
# NumPy's ArrayLike alias is intentionally broad, but recent NumPy stubs expose
# unresolved type variables through it under strict Pyright checking. These
# inputs are normalized immediately with np.asarray, so Any is intentional at
# this runtime-validation boundary.
ArrayLike: TypeAlias = Any
RawEnergyFunction = Callable[[ArrayLike], object]

_ENERGY_DENSITY_UNIT = SI(1.0, "J/m^3")
_FINITE_DIFFERENCE_STEP = float(np.cbrt(np.finfo(np.float64).eps))


def _normalize(values: ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"Anisotropy axes must be 3-vectors, got shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Anisotropy axes must contain only finite values.")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def energy_density_value(value: object) -> float:
    result = (
        value.in_units_of(_ENERGY_DENSITY_UNIT)
        if isinstance(value, SI)
        else float(cast(Any, value))
    )
    if not np.isfinite(result):
        raise ValueError("Anisotropy energy density must be finite.")
    return result


def _magnetisation_array(values: ArrayLike) -> tuple[FloatArray, bool]:
    array = np.asarray(values, dtype=np.float64)
    scalar = array.ndim == 1
    if scalar:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Magnetisation must have shape (3,) or (n, 3), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Magnetisation must contain only finite values.")
    return np.ascontiguousarray(array), scalar


def _callable_energy(function: RawEnergyFunction, magnetisation: FloatArray) -> FloatArray:
    return np.asarray(
        [energy_density_value(function(vector)) for vector in magnetisation],
        dtype=np.float64,
    )


def _finite_difference_gradient(
    function: RawEnergyFunction,
    magnetisation: FloatArray,
) -> FloatArray:
    gradient = np.empty_like(magnetisation)
    for row, vector in enumerate(magnetisation):
        for component in range(3):
            step = _FINITE_DIFFERENCE_STEP * max(1.0, abs(float(vector[component])))
            plus = np.array(vector, copy=True)
            minus = np.array(vector, copy=True)
            plus[component] += step
            minus[component] -= step
            gradient[row, component] = (
                energy_density_value(function(plus)) - energy_density_value(function(minus))
            ) / (2.0 * step)
    if not np.all(np.isfinite(gradient)):
        raise ValueError("Anisotropy energy gradient must be finite.")
    return gradient
