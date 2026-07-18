"""Evaluate arbitrary anisotropy models and build compatibility samples."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .model import AnisotropyModel, FloatArray, PredefinedAnisotropy
from .values import _callable_energy, _finite_difference_gradient, _magnetisation_array, _normalize


def evaluate_energy_density(model: AnisotropyModel, magnetisation: ArrayLike) -> FloatArray:
    values, _scalar = _magnetisation_array(magnetisation)
    if model is None:
        return np.zeros(len(values), dtype=np.float64)
    if isinstance(model, PredefinedAnisotropy):
        return np.asarray(model.energy_density(values), dtype=np.float64)
    return _callable_energy(model, values)


def evaluate_energy_gradient(model: AnisotropyModel, magnetisation: ArrayLike) -> FloatArray:
    values, _scalar = _magnetisation_array(magnetisation)
    if model is None:
        return np.zeros_like(values)
    if isinstance(model, PredefinedAnisotropy):
        return np.asarray(model.energy_gradient(values), dtype=np.float64)
    return _finite_difference_gradient(model, values)


def evaluate_energy_and_gradient(
    model: AnisotropyModel,
    magnetisation: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    values, _scalar = _magnetisation_array(magnetisation)
    if model is None:
        return np.zeros(len(values), dtype=np.float64), np.zeros_like(values)
    if isinstance(model, PredefinedAnisotropy):
        return model.energy_and_gradient(values)
    return _callable_energy(model, values), _finite_difference_gradient(model, values)


def anisotropy_signature_values(
    model: AnisotropyModel,
    order: int | None,
) -> FloatArray:
    if model is None:
        return np.asarray([0.0], dtype=np.float64)
    probes = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            _normalize([1.0, 1.0, 1.0]),
            _normalize([1.0, 2.0, 3.0]),
        ],
        dtype=np.float64,
    )
    energy, gradient = evaluate_energy_and_gradient(model, probes)
    return np.concatenate(
        (
            np.asarray([float(order or 0)], dtype=np.float64),
            energy,
            np.ravel(gradient),
        )
    )
