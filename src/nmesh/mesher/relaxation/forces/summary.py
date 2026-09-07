"""Per-point force metrics for the relaxation mesher."""

from __future__ import annotations

import math

import numpy as np

from .._constants import BOUNDARY_FUZZ, DENSITY_EPSILON, STATE_BOUNDARY, STATE_FIXED, STATE_SIMPLE
from .._types import FloatArray
from ..geometry import FemGeometry
from .simplex import _project_force_to_tangent
from .types import ForceSummary, _sphere_volume


def _finalize_force_summary(
    total_forces: FloatArray,
    neighbor_map: list[list[int]],
    simplices: np.ndarray,
    point_densities: FloatArray,
    neighbor_force_sums: FloatArray,
    neighbor_force_counts: np.ndarray,
    simplex_measures: FloatArray,
    incident_simplices: list[list[int]],
    angle_sums: FloatArray,
    points: FloatArray,
    states: np.ndarray,
    geometry: FemGeometry,
    a0: float,
    dim: int,
) -> ForceSummary:
    """Assemble the per-point density and effective-force metrics."""
    point_count = len(points)
    point_density, point_average_force, point_effective_force = _empty_force_arrays(point_count)

    if point_count == 0:
        return _make_force_summary(
            total_forces,
            neighbor_map,
            simplices,
            point_density,
            point_average_force,
            point_effective_force,
        )

    boundary_mask = geometry.boundary_mask(points, tolerance=max(0.05 * a0, BOUNDARY_FUZZ))

    for point_index, point in enumerate(points):
        point_state = int(states[point_index])
        density_here = float(point_densities[point_index])
        effective_rod_length = a0 / (density_here ** (1.0 / max(dim, 1)))
        ideal_local_volume = _sphere_volume(0.5 * effective_rod_length, dim)
        angle = float(angle_sums[point_index])
        point_average_force[point_index] = _average_neighbor_force(
            neighbor_force_sums[point_index],
            int(neighbor_force_counts[point_index]),
            angle,
        )
        corrected_volume = _corrected_point_volume(
            simplex_measures=simplex_measures,
            incident=incident_simplices[point_index],
            angle=angle,
            dim=dim,
            state=point_state,
            has_multiple_immobile_neighbors=_has_multiple_immobile_neighbors(
                point_index,
                states,
                neighbor_map,
            ),
            ideal_local_volume=ideal_local_volume,
        )
        point_density[point_index] = ideal_local_volume / max(corrected_volume, DENSITY_EPSILON)
        point_effective_force[point_index] = _point_effective_force(
            total_force=total_forces[point_index],
            state=point_state,
            geometry=geometry,
            point=point,
            is_boundary=bool(boundary_mask[point_index]),
            density_here=density_here,
            a0=a0,
            dim=dim,
        )

    return _make_force_summary(
        total_forces,
        neighbor_map,
        simplices,
        point_density,
        point_average_force,
        point_effective_force,
    )


def _empty_force_arrays(point_count: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Allocate the per-point summary arrays used by ``_finalize_force_summary``."""
    return (
        np.ones(point_count, dtype=float),
        np.zeros(point_count, dtype=float),
        np.zeros(point_count, dtype=float),
    )


def _make_force_summary(
    total_forces: FloatArray,
    neighbor_map: list[list[int]],
    simplices: np.ndarray,
    point_density: FloatArray,
    point_average_force: FloatArray,
    point_effective_force: FloatArray,
) -> ForceSummary:
    """Construct a ``ForceSummary`` from the completed per-point arrays."""
    return ForceSummary(
        total_forces=total_forces,
        neighbor_map=neighbor_map,
        simplices=simplices,
        point_density=point_density,
        point_average_force=point_average_force,
        point_effective_force=point_effective_force,
        max_effective_force=float(np.max(point_effective_force, initial=0.0)),
    )


def _average_neighbor_force(force_sum: float, force_count: int, angle: float) -> float:
    """Return the mean absolute neighbor-force magnitude for a point."""
    if angle <= DENSITY_EPSILON:
        return 1.0e4
    if force_count <= 0:
        return 0.0
    return force_sum / float(force_count)


def _has_multiple_immobile_neighbors(
    point_index: int,
    states: np.ndarray,
    neighbor_map: list[list[int]],
) -> bool:
    """Return whether a boundary point is trapped between multiple immobile nodes."""
    if int(states[point_index]) != STATE_BOUNDARY:
        return False
    immobile_neighbors = sum(
        1
        for neighbor_index in neighbor_map[point_index]
        if int(states[neighbor_index]) in (STATE_FIXED, STATE_SIMPLE)
    )
    return immobile_neighbors > 1


def _corrected_point_volume(
    *,
    simplex_measures: FloatArray,
    incident: list[int],
    angle: float,
    dim: int,
    state: int,
    has_multiple_immobile_neighbors: bool,
    ideal_local_volume: float,
) -> float:
    """Return the local corrected control volume for one point."""
    if has_multiple_immobile_neighbors or angle <= DENSITY_EPSILON:
        return 1.0e-4 * ideal_local_volume
    if not incident:
        return 1.0e-4 * ideal_local_volume

    incident_measures = simplex_measures[np.asarray(incident, dtype=int)]
    if len(incident) == 1:
        local_volume = float(incident_measures[0]) / float(max(dim, 1))
    else:
        local_volume = float(np.sum(incident_measures)) / float(dim + 1)

    correction = _voronoi_angle_correction(dim, angle, len(incident))
    corrected_volume = local_volume * correction
    if state == STATE_BOUNDARY:
        corrected_volume *= 1.2
    return corrected_volume


def _voronoi_angle_correction(dim: int, angle: float, incident_count: int) -> float:
    """Return the legacy angular correction for Voronoi control volumes."""
    if dim == 1:
        return 1.5
    if dim == 2:
        return 2.0 * math.pi / angle
    if dim == 3:
        return 4.0 * math.pi / angle
    return math.factorial(dim + 1) / float(max(incident_count, 1))


def _point_effective_force(
    *,
    total_force: FloatArray,
    state: int,
    geometry: FemGeometry,
    point: FloatArray,
    is_boundary: bool,
    density_here: float,
    a0: float,
    dim: int,
) -> float:
    """Return the normalized effective force magnitude for one point."""
    effective_force = np.asarray(total_force, dtype=float)
    if state == STATE_BOUNDARY or is_boundary:
        effective_force = _project_force_to_tangent(geometry, point, effective_force)
    return (
        float(np.linalg.norm(effective_force))
        * (density_here ** (1.0 / max(dim, 1)))
        / max(a0, DENSITY_EPSILON)
    )
