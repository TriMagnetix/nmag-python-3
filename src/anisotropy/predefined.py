"""Vectorized predefined magnetic anisotropy models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .model import EnergyDensity, PredefinedAnisotropy
from .values import FloatArray, _normalize, energy_density_value


def _constant(value: EnergyDensity, name: str) -> float:
    try:
        return energy_density_value(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an energy density compatible with J/m^3.") from exc


def uniaxial_anisotropy(
    axis: ArrayLike,
    K1: EnergyDensity,
    K2: EnergyDensity = 0.0,
) -> PredefinedAnisotropy:
    """Create a uniaxial anisotropy energy model.

    The energy density is ``-K1 (axis·m)^2 - K2 (axis·m)^4``.

    Args:
        axis: Three-component easy-axis direction; normalized internally.
        K1: Second-order coefficient as J/m³ or a compatible SI quantity.
        K2: Fourth-order coefficient as J/m³ or a compatible SI quantity.

    Returns:
        Vectorized predefined anisotropy with analytic gradient evaluation.

    Raises:
        ValueError: If the axis is zero or a coefficient is non-finite.
    """

    normalized_axis = _normalize(axis)
    k1 = _constant(K1, "K1")
    k2 = _constant(K2, "K2")

    def energy(values: FloatArray) -> FloatArray:
        projection = values @ normalized_axis
        return -k1 * projection**2 - k2 * projection**4

    def gradient(values: FloatArray) -> FloatArray:
        projection = values @ normalized_axis
        factors = -2.0 * k1 * projection - 4.0 * k2 * projection**3
        return factors[:, np.newaxis] * normalized_axis

    def energy_gradient(values: FloatArray) -> tuple[FloatArray, FloatArray]:
        projection = values @ normalized_axis
        energy_values = -k1 * projection**2 - k2 * projection**4
        factors = -2.0 * k1 * projection - 4.0 * k2 * projection**3
        return energy_values, factors[:, np.newaxis] * normalized_axis

    def scalar_energy(values: ArrayLike) -> float:
        vector = np.asarray(values, dtype=np.float64)
        return float(energy(vector[np.newaxis, :])[0])

    def stringifier(model: PredefinedAnisotropy) -> str:
        assert model.axis1 is not None
        axis_values = np.asarray(model.axis1, dtype=np.float64).tolist()
        result = f"axis={axis_values}, K1={model.K1}"
        if k2 != 0.0:
            result += f", K2={model.K2}"
        return result

    return PredefinedAnisotropy(
        anis_type="uniaxial",
        function=scalar_energy,
        order=4 if k2 != 0.0 else 2,
        axis1=normalized_axis,
        K1=K1,
        K2=K2,
        stringifier=stringifier,
        _vectorized_energy=energy,
        _vectorized_gradient=gradient,
        _vectorized_energy_gradient=energy_gradient,
    )


def cubic_anisotropy(
    axis1: ArrayLike,
    axis2: ArrayLike,
    K1: EnergyDensity,
    K2: EnergyDensity = 0.0,
    K3: EnergyDensity = 0.0,
) -> PredefinedAnisotropy:
    """Create conventional fourth-, sixth-, and eighth-order cubic anisotropy.

    Args:
        axis1: First crystalline axis; normalized internally.
        axis2: Second crystalline axis, required to be orthogonal to ``axis1``.
        K1: Fourth-order coefficient as J/m³ or a compatible SI quantity.
        K2: Sixth-order coefficient as J/m³ or a compatible SI quantity.
        K3: Eighth-order coefficient as J/m³ or a compatible SI quantity.

    Returns:
        Vectorized predefined anisotropy with analytic gradient evaluation.

    Raises:
        ValueError: If axes are invalid or coefficients are non-finite.
    """

    first = _normalize(axis1)
    raw_second = np.asarray(axis2, dtype=np.float64)
    third = _normalize(np.cross(first, raw_second))
    second = _normalize(np.cross(third, first))
    axes: FloatArray = np.asarray((first, second, third), dtype=np.float64)
    axes_transpose: FloatArray = np.ascontiguousarray(np.transpose(axes))
    k1, k2, k3 = (_constant(value, name) for value, name in ((K1, "K1"), (K2, "K2"), (K3, "K3")))

    def energy(values: FloatArray) -> FloatArray:
        projections: FloatArray = np.asarray(values @ axes_transpose, dtype=np.float64)
        squared: FloatArray = projections**2
        fourth: FloatArray = squared**2
        pair2 = (
            squared[:, 0] * squared[:, 1]
            + squared[:, 0] * squared[:, 2]
            + squared[:, 1] * squared[:, 2]
        )
        triple2 = squared[:, 0] * squared[:, 1] * squared[:, 2]
        pair4 = (
            fourth[:, 0] * fourth[:, 1] + fourth[:, 0] * fourth[:, 2] + fourth[:, 1] * fourth[:, 2]
        )
        return k1 * pair2 + k2 * triple2 + k3 * pair4

    def gradient(values: FloatArray) -> FloatArray:
        projections: FloatArray = np.asarray(values @ axes_transpose, dtype=np.float64)
        squared: FloatArray = projections**2
        fourth: FloatArray = squared**2
        derivatives: FloatArray = np.empty_like(projections)
        for axis in range(3):
            other = [index for index in range(3) if index != axis]
            derivatives[:, axis] = (
                2.0 * k1 * projections[:, axis] * (squared[:, other[0]] + squared[:, other[1]])
                + 2.0 * k2 * projections[:, axis] * squared[:, other[0]] * squared[:, other[1]]
                + 4.0 * k3 * projections[:, axis] ** 3 * (fourth[:, other[0]] + fourth[:, other[1]])
            )
        return derivatives @ axes

    def energy_gradient(values: FloatArray) -> tuple[FloatArray, FloatArray]:
        projections: FloatArray = np.asarray(values @ axes_transpose, dtype=np.float64)
        squared: FloatArray = projections**2
        fourth: FloatArray = squared**2
        pair2 = (
            squared[:, 0] * squared[:, 1]
            + squared[:, 0] * squared[:, 2]
            + squared[:, 1] * squared[:, 2]
        )
        energy_values = (
            k1 * pair2
            + k2 * squared[:, 0] * squared[:, 1] * squared[:, 2]
            + k3
            * (
                fourth[:, 0] * fourth[:, 1]
                + fourth[:, 0] * fourth[:, 2]
                + fourth[:, 1] * fourth[:, 2]
            )
        )
        derivatives: FloatArray = np.empty_like(projections)
        for axis in range(3):
            other = [index for index in range(3) if index != axis]
            derivatives[:, axis] = (
                2.0 * k1 * projections[:, axis] * (squared[:, other[0]] + squared[:, other[1]])
                + 2.0 * k2 * projections[:, axis] * squared[:, other[0]] * squared[:, other[1]]
                + 4.0 * k3 * projections[:, axis] ** 3 * (fourth[:, other[0]] + fourth[:, other[1]])
            )
        return energy_values, derivatives @ axes

    def scalar_energy(values: ArrayLike) -> float:
        vector = np.asarray(values, dtype=np.float64)
        return float(energy(vector[np.newaxis, :])[0])

    def stringifier(model: PredefinedAnisotropy) -> str:
        assert model.axis1 is not None and model.axis2 is not None
        first_values = np.asarray(model.axis1, dtype=np.float64).tolist()
        second_values = np.asarray(model.axis2, dtype=np.float64).tolist()
        result = f"axis1={first_values}, axis2={second_values}, K1={model.K1}"
        if k2 != 0.0:
            result += f", K2={model.K2}"
        if k3 != 0.0:
            result += f", K3={model.K3}"
        return result

    return PredefinedAnisotropy(
        anis_type="cubic",
        function=scalar_energy,
        order=8 if k3 != 0.0 else 6 if k2 != 0.0 else 4,
        axis1=first,
        axis2=second,
        axis3=third,
        K1=K1,
        K2=K2,
        K3=K3,
        stringifier=stringifier,
        _vectorized_energy=energy,
        _vectorized_gradient=gradient,
        _vectorized_energy_gradient=energy_gradient,
    )
