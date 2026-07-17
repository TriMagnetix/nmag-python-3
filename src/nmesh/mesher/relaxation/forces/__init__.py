"""Compatibility facade and orchestration for relaxation meshing forces."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._types import FloatArray
from ..geometry import FemGeometry
from ..topology import _triangulate_points
from .neighbors import (
    _build_neighbor_map,
    _compute_neighbor_forces,
)
from .simplex import (
    _classify_relevant_simplices,
    _compute_simplex_forces,
    _simplex_incidence_data,
)
from .summary import _finalize_force_summary
from .types import (
    ForceParameters,
    ForceSummary,
    _extract_force_parameters,
)

__all__ = [
    "ForceParameters",
    "ForceSummary",
    "_classify_relevant_simplices",
    "_extract_force_parameters",
    "compute_forces",
]


def compute_forces(
    points: FloatArray,
    states: np.ndarray,
    geometry: FemGeometry,
    a0: float,
    params: dict[str, Any],
    step: int,
    simplices: np.ndarray | None = None,
) -> ForceSummary:
    """Compute neighbor, shape, volume, and irrelevant-element forces."""
    point_count = len(points)
    dim = geometry.dim
    if simplices is None:
        simplices = _triangulate_points(points, dim, states)
    neighbor_map = _build_neighbor_map(point_count, simplices)
    config = _extract_force_parameters(params, step)
    point_densities = np.asarray([geometry.density_at(point) for point in points], dtype=float)

    total_forces, neighbor_force_sums, neighbor_force_counts = _compute_neighbor_forces(
        points,
        states,
        simplices,
        neighbor_map,
        point_densities,
        dim,
        a0,
        config,
    )
    simplex_measures, incident_simplices, angle_sums = _simplex_incidence_data(
        points, simplices, dim
    )
    _compute_simplex_forces(
        total_forces,
        points,
        states,
        geometry,
        simplices,
        point_densities,
        simplex_measures,
        angle_sums,
        dim,
        a0,
        config,
    )
    return _finalize_force_summary(
        total_forces=total_forces,
        neighbor_map=neighbor_map,
        simplices=simplices,
        point_densities=point_densities,
        neighbor_force_sums=neighbor_force_sums,
        neighbor_force_counts=neighbor_force_counts,
        simplex_measures=simplex_measures,
        incident_simplices=incident_simplices,
        angle_sums=angle_sums,
        points=points,
        states=states,
        geometry=geometry,
        a0=a0,
        dim=dim,
    )
