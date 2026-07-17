"""Shared types, configuration, and scalar helpers for meshing forces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...meshing_parameters import (
    default_boundary_node_force_fun,
    default_initial_relaxation_weight,
    default_relaxation_force_fun,
)
from .._types import FloatArray


@dataclass(frozen=True, slots=True)
class ForceSummary:
    """Container for the per-step force calculation results."""

    total_forces: FloatArray
    neighbor_map: list[list[int]]
    simplices: np.ndarray
    point_density: FloatArray
    point_average_force: FloatArray
    point_effective_force: FloatArray
    max_effective_force: float


@dataclass(frozen=True, slots=True)
class ForceParameters:
    """Resolved controller parameters for one relaxation-force evaluation."""

    shape_force_scale: float
    volume_force_scale: float
    neigh_force_scale: float
    irrel_force_scale: float
    sliver_correction: float
    smallest_allowed_volume_ratio: float
    relaxation_weight: float
    force_fun: Any
    boundary_force_fun: Any


def _corner_force_threshold(dim: int) -> float:
    """Return the solid-angle threshold below which corner suppression applies."""
    if dim == 2:
        return 0.75 * math.pi
    if dim == 3:
        return 0.85 * math.pi
    return -math.inf


def _regular_simplex_volume(edge_length: float, dim: int) -> float:
    """Return the volume of a regular simplex with the supplied edge length."""
    if dim <= 0:
        return edge_length
    numerator = edge_length**dim * math.sqrt(dim + 1.0)
    denominator = math.factorial(dim) * math.sqrt(2.0**dim)
    return numerator / denominator


def _sphere_volume(radius: float, dim: int) -> float:
    """Return the d-dimensional volume of a sphere."""
    return (math.pi ** (0.5 * dim) / math.gamma(0.5 * dim + 1.0)) * (radius**dim)


def _extract_force_parameters(params: dict[str, Any], step: int) -> ForceParameters:
    """Resolve the mesher controller parameters used by ``compute_forces``."""
    settling_steps = int(params.get("controller_initial_settling_steps", 100))
    relaxation_weight_fun = params.get(
        "initial_relaxation_weight_fun",
        default_initial_relaxation_weight,
    )
    return ForceParameters(
        shape_force_scale=float(params.get("controller_shape_force_scale", 0.1)),
        volume_force_scale=float(params.get("controller_volume_force_scale", 0.0)),
        neigh_force_scale=float(params.get("controller_neigh_force_scale", 1.0)),
        irrel_force_scale=float(params.get("controller_irrel_elem_force_scale", 1.0)),
        sliver_correction=float(params.get("controller_sliver_correction", 1.0)),
        smallest_allowed_volume_ratio=float(
            params.get("controller_smallest_allowed_volume_ratio", 1.0)
        ),
        relaxation_weight=float(relaxation_weight_fun(step, settling_steps, 0.0, 1.0)),
        force_fun=params.get("relaxation_force_fun", default_relaxation_force_fun),
        boundary_force_fun=params.get(
            "boundary_node_force_fun",
            default_boundary_node_force_fun,
        ),
    )
