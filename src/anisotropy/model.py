"""Typed anisotropy models and numerical evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias, cast

import numpy as np

from si.physical import SI

from .values import (
    ArrayLike,
    FloatArray,
    _callable_energy,
    _finite_difference_gradient,
    _magnetisation_array,
)

EnergyDensity: TypeAlias = float | SI
EnergyFunction: TypeAlias = Callable[[ArrayLike], EnergyDensity]
GradientFunction: TypeAlias = Callable[[FloatArray], FloatArray]
VectorizedEnergyFunction: TypeAlias = Callable[[FloatArray], FloatArray]
EnergyGradientFunction: TypeAlias = Callable[[FloatArray], tuple[FloatArray, FloatArray]]
AnisotropyStringifier: TypeAlias = Callable[["PredefinedAnisotropy"], str]
AnisotropyModel: TypeAlias = "PredefinedAnisotropy | EnergyFunction | None"


@dataclass(frozen=True, slots=True)
class PredefinedAnisotropy:
    """An anisotropy energy model with optional analytic vectorized evaluation.

    Instances returned by :func:`uniaxial_anisotropy` and
    :func:`cubic_anisotropy` can be added, subtracted, negated, and multiplied by
    scalar coefficients before being assigned to :class:`nmag.MagMaterial`.
    """

    function: EnergyFunction | None = None
    order: int | None = None
    anis_type: str = "functional"
    axis1: ArrayLike | None = None
    axis2: ArrayLike | None = None
    axis3: ArrayLike | None = None
    K1: EnergyDensity | None = None
    K2: EnergyDensity | None = None
    K3: EnergyDensity | None = None
    stringifier: AnisotropyStringifier | None = field(default=None, repr=False)
    _vectorized_energy: VectorizedEnergyFunction | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _vectorized_gradient: GradientFunction | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _vectorized_energy_gradient: EnergyGradientFunction | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.function is None and self.order is None:
            raise ValueError("PredefinedAnisotropy requires either a 'function' or an 'order'.")
        if self.function is not None and not callable(self.function):
            raise TypeError("Anisotropy function must be callable.")
        order = cast(object, self.order)
        if order is not None and (type(order) is bool or not isinstance(order, int) or order <= 0):
            raise ValueError("Anisotropy order must be a positive integer.")
        for name in ("axis1", "axis2", "axis3"):
            axis = getattr(self, name)
            if axis is not None:
                frozen_axis = np.array(axis, dtype=np.float64, copy=True)
                frozen_axis.setflags(write=False)
                object.__setattr__(self, name, frozen_axis)

    def has_function(self) -> bool:
        return self.function is not None

    def energy_density(self, magnetisation: ArrayLike) -> FloatArray | float:
        values, scalar = _magnetisation_array(magnetisation)
        if self._vectorized_energy is not None:
            result = np.asarray(self._vectorized_energy(values), dtype=np.float64)
        elif self.function is not None:
            result = _callable_energy(self.function, values)
        else:
            raise ValueError("This anisotropy model has no energy function.")
        if result.shape != (len(values),) or not np.all(np.isfinite(result)):
            raise ValueError("Anisotropy energy evaluation returned invalid values.")
        return float(result[0]) if scalar else result

    def energy_gradient(self, magnetisation: ArrayLike) -> FloatArray:
        values, scalar = _magnetisation_array(magnetisation)
        if self._vectorized_gradient is not None:
            result = np.asarray(self._vectorized_gradient(values), dtype=np.float64)
        elif self.function is not None:
            result = _finite_difference_gradient(self.function, values)
        else:
            raise ValueError("This anisotropy model has no energy function.")
        if result.shape != values.shape or not np.all(np.isfinite(result)):
            raise ValueError("Anisotropy gradient evaluation returned invalid values.")
        return result[0] if scalar else result

    def energy_and_gradient(self, magnetisation: ArrayLike) -> tuple[FloatArray, FloatArray]:
        values, _scalar = _magnetisation_array(magnetisation)
        if self._vectorized_energy_gradient is not None:
            energy, gradient = self._vectorized_energy_gradient(values)
            energy = np.asarray(energy, dtype=np.float64)
            gradient = np.asarray(gradient, dtype=np.float64)
        else:
            energy = np.asarray(self.energy_density(values), dtype=np.float64)
            gradient = np.asarray(self.energy_gradient(values), dtype=np.float64)
        if energy.shape != (len(values),) or gradient.shape != values.shape:
            raise ValueError("Anisotropy energy/gradient evaluation returned invalid shapes.")
        if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(gradient)):
            raise ValueError("Anisotropy energy/gradient evaluation returned non-finite values.")
        return energy, gradient

    def __str__(self) -> str:
        suffix = f", {self.stringifier(self)}" if self.stringifier else ""
        return f"<PredefinedAnisotropy:{self.anis_type}{suffix}>"

    def __repr__(self) -> str:
        details = self.stringifier(self) if self.stringifier else "?"
        return f'PredefinedAnisotropy(anis_type="{self.anis_type}", {details})'

    def __neg__(self) -> PredefinedAnisotropy:
        want_anisotropy(self)
        return _combine_anisotropy(self, None, -1.0)

    def __pos__(self) -> PredefinedAnisotropy:
        return self

    def __add__(self, other: object) -> PredefinedAnisotropy:
        want_anisotropy(other)
        assert isinstance(other, PredefinedAnisotropy)
        return _combine_anisotropy(self, other, 1.0)

    def __sub__(self, other: object) -> PredefinedAnisotropy:
        want_anisotropy(other)
        assert isinstance(other, PredefinedAnisotropy)
        return _combine_anisotropy(self, other, -1.0)


def _combine_anisotropy(
    left: PredefinedAnisotropy,
    right: PredefinedAnisotropy | None,
    right_scale: float,
) -> PredefinedAnisotropy:
    def vectorized_energy_gradient(values: FloatArray) -> tuple[FloatArray, FloatArray]:
        left_energy, left_gradient = left.energy_and_gradient(values)
        if right is None:
            return right_scale * left_energy, right_scale * left_gradient
        right_energy, right_gradient = right.energy_and_gradient(values)
        return (
            left_energy + right_scale * right_energy,
            left_gradient + right_scale * right_gradient,
        )

    def vectorized_energy(values: FloatArray) -> FloatArray:
        left_energy = np.asarray(left.energy_density(values), dtype=np.float64)
        if right is None:
            return right_scale * left_energy
        return left_energy + right_scale * np.asarray(
            right.energy_density(values), dtype=np.float64
        )

    def vectorized_gradient(values: FloatArray) -> FloatArray:
        left_gradient = np.asarray(left.energy_gradient(values), dtype=np.float64)
        if right is None:
            return right_scale * left_gradient
        return left_gradient + right_scale * np.asarray(
            right.energy_gradient(values),
            dtype=np.float64,
        )

    def scalar_energy(values: ArrayLike) -> float:
        array, _scalar = _magnetisation_array(values)
        return float(vectorized_energy(array)[0])

    order = left.order if right is None else max(left.order or 0, right.order or 0)
    return PredefinedAnisotropy(
        function=scalar_energy,
        order=order or None,
        _vectorized_energy=vectorized_energy,
        _vectorized_gradient=vectorized_gradient,
        _vectorized_energy_gradient=vectorized_energy_gradient,
    )


def want_anisotropy(value: object, want_function: bool = True) -> None:
    if not isinstance(value, PredefinedAnisotropy):
        raise TypeError(
            f"Operand must be a PredefinedAnisotropy object, not {type(value).__name__}"
        )
    if want_function and not value.has_function():
        raise ValueError("Cannot operate on an anisotropy object that lacks an energy function.")
