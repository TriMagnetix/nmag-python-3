"""Simplex shape, volume, and relevance forces for the relaxation mesher."""

from __future__ import annotations

import math

import numpy as np

from .._constants import BOUNDARY_FUZZ, DENSITY_EPSILON, STATE_MOBILE
from .._types import FloatArray
from ..geometry import FemGeometry
from ..topology import (
    _boundary_state_mask,
    _classify_simplices_with_probes,
    _simplex_measures,
    _simplex_volume_order_ratio,
)
from .types import ForceParameters, _corner_force_threshold, _regular_simplex_volume


def _vertex_angle(points: FloatArray, simplex: np.ndarray, local_index: int, dim: int) -> float:
    """Return the angle or solid angle covered by a simplex at one vertex."""
    vertex = points[int(simplex[local_index])]
    others: list[FloatArray] = [
        np.asarray(points[int(simplex[index])] - vertex, dtype=float)
        for index in range(len(simplex))
        if index != local_index
    ]

    if dim == 1:
        return math.pi

    if dim == 2:
        first = others[0]
        second = others[1]
        denom = math.sqrt(float(np.sum(first * first))) * math.sqrt(float(np.sum(second * second)))
        if denom <= DENSITY_EPSILON:
            return 0.0
        cosine = min(1.0, max(-1.0, float(np.sum(first * second)) / denom))
        return float(math.acos(cosine))

    if dim == 3:
        a, b, c = others
        numer = abs(float(np.sum(a * np.cross(b, c))))
        norm_a = math.sqrt(float(np.sum(a * a)))
        norm_b = math.sqrt(float(np.sum(b * b)))
        norm_c = math.sqrt(float(np.sum(c * c)))
        denom = (
            norm_a * norm_b * norm_c
            + float(np.sum(a * b)) * norm_c
            + float(np.sum(a * c)) * norm_b
            + float(np.sum(b * c)) * norm_a
        )
        if numer <= DENSITY_EPSILON and denom <= DENSITY_EPSILON:
            return 0.0
        return float(2.0 * math.atan2(numer, max(denom, DENSITY_EPSILON)))

    return 1.0


def _shape_force_matrix(vertices: FloatArray, dim: int) -> FloatArray:
    """Build the covariance-derived shape force matrix for one simplex."""
    covariance = np.asarray(np.transpose(vertices) @ vertices, dtype=float)
    covariance /= max(float(len(vertices) - 1), 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    safe_eigenvalues = np.maximum(
        np.asarray(eigenvalues, dtype=float),
        DENSITY_EPSILON,
    )
    product = float(np.prod(safe_eigenvalues))
    if product <= DENSITY_EPSILON:
        return np.zeros((dim, dim), dtype=float)

    scaled = safe_eigenvalues * ((1.0 / product) ** (1.0 / max(dim, 1)))
    diagonal = -np.log(np.maximum(scaled, DENSITY_EPSILON))
    transformed = np.zeros((dim, dim), dtype=float)
    transformed[np.arange(dim), np.arange(dim)] = diagonal
    return np.asarray(eigenvectors @ transformed @ np.transpose(eigenvectors), dtype=float)


def _project_force_to_tangent(
    geometry: FemGeometry,
    point: FloatArray,
    force: FloatArray,
) -> FloatArray:
    """Remove the normal component of a force at a boundary point."""
    normal = geometry.boundary_normal(point)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= BOUNDARY_FUZZ:
        return np.asarray(force, dtype=float)
    unit_normal = normal / normal_norm
    return np.asarray(force, dtype=float) - float(np.dot(force, unit_normal)) * unit_normal


def _simplex_force_contribution(
    simplex_points: FloatArray,
    dim: int,
    density_here: float,
    a0: float,
    relaxation_weight: float,
    shape_force_scale: float,
    volume_force_scale: float,
    sliver_correction: float,
    volume: float,
) -> FloatArray:
    """Compute the shape and volume force contribution for one simplex."""
    center = np.mean(simplex_points, axis=0)
    vertices = simplex_points - center
    ideal_edge_length = a0 / (density_here ** (1.0 / max(dim, 1)))
    ideal_volume = _regular_simplex_volume(ideal_edge_length, dim)
    return _compute_volume_forces(
        vertices,
        relaxation_weight,
        volume_force_scale,
        volume,
        ideal_volume,
    ) + _compute_shape_forces(
        vertices=vertices,
        dim=dim,
        relaxation_weight=relaxation_weight,
        shape_force_scale=shape_force_scale,
        sliver_correction=sliver_correction,
        volume=volume,
        ideal_volume=ideal_volume,
    )


def _compute_volume_forces(
    vertices: FloatArray,
    relaxation_weight: float,
    volume_force_scale: float,
    volume: float,
    ideal_volume: float,
) -> FloatArray:
    """Return the isotropic volume-restoring forces for one simplex."""
    forces = np.zeros_like(vertices)
    if volume_force_scale <= 0.0 or volume <= DENSITY_EPSILON or ideal_volume <= DENSITY_EPSILON:
        return forces
    volume_factor = volume_force_scale * math.log(ideal_volume / volume)
    return forces + relaxation_weight * volume_factor * vertices


def _compute_shape_forces(
    *,
    vertices: FloatArray,
    dim: int,
    relaxation_weight: float,
    shape_force_scale: float,
    sliver_correction: float,
    volume: float,
    ideal_volume: float,
) -> FloatArray:
    """Return the covariance-derived shape-correction forces for one simplex."""
    forces = np.zeros_like(vertices)
    if shape_force_scale <= 0.0:
        return forces

    shape_matrix = _shape_force_matrix(vertices, dim)
    if not np.any(shape_matrix):
        return forces

    vol_correction = _shape_volume_correction(volume, ideal_volume, dim)
    for index, vertex in enumerate(vertices):
        norm = float(np.linalg.norm(vertex))
        if norm <= DENSITY_EPSILON:
            continue
        raw_force = shape_matrix @ vertex
        normal = vertex / norm
        projection = float(np.dot(raw_force, normal))
        angular = raw_force - projection * normal
        forces[index] += relaxation_weight * vol_correction * shape_force_scale * angular
        if projection > 0.0 and sliver_correction > 0.0:
            longitudinal = raw_force - angular
            forces[index] += (
                shape_force_scale
                * sliver_correction
                * relaxation_weight
                * max(vol_correction - 1.0, 0.0)
                * longitudinal
            )
    return forces


def _shape_volume_correction(volume: float, ideal_volume: float, dim: int) -> float:
    """Return the volume-dependent multiplier applied to shape forces."""
    if volume <= DENSITY_EPSILON or ideal_volume <= DENSITY_EPSILON:
        return 0.0
    offset = 0.0 if dim <= 2 else 1.0
    return max(1.0, offset + math.log(ideal_volume / volume))


def _simplex_incidence_data(
    points: FloatArray,
    simplices: np.ndarray,
    dim: int,
) -> tuple[FloatArray, list[list[int]], FloatArray]:
    """Collect simplex measures plus per-point incident simplex and angle data."""
    simplex_measures = _simplex_measures(points, simplices, dim)
    incident_simplices: list[list[int]] = [[] for _ in range(len(points))]
    angle_sums = np.zeros(len(points), dtype=float)
    for simplex_index, simplex in enumerate(simplices):
        for local_index, point_index in enumerate(simplex.tolist()):
            point_id = int(point_index)
            incident_simplices[point_id].append(simplex_index)
            angle_sums[point_id] += _vertex_angle(points, simplex, local_index, dim)
    return simplex_measures, incident_simplices, angle_sums


def _compute_simplex_forces(
    total_forces: FloatArray,
    points: FloatArray,
    states: np.ndarray,
    geometry: FemGeometry,
    simplices: np.ndarray,
    point_densities: FloatArray,
    simplex_measures: FloatArray,
    angle_sums: FloatArray,
    dim: int,
    a0: float,
    config: ForceParameters,
) -> None:
    """Accumulate simplex shape, volume, and irrelevant-element forces."""
    del point_densities
    if len(simplices) == 0:
        return
    if (
        config.shape_force_scale <= 0.0
        and config.volume_force_scale <= 0.0
        and config.irrel_force_scale <= 0.0
    ):
        return

    relevant_simplices = _classify_relevant_simplices(
        points,
        states,
        geometry,
        simplices,
        simplex_measures,
        dim,
        config,
    )
    suppress_corner_forces = angle_sums < _corner_force_threshold(dim)
    for simplex_index, simplex in enumerate(simplices):
        simplex_points = points[simplex]
        simplex_force = np.zeros_like(simplex_points)
        volume = float(simplex_measures[simplex_index])
        if relevant_simplices[simplex_index]:
            density_here = geometry.density_at(np.mean(simplex_points, axis=0))
            simplex_force = _simplex_force_contribution(
                simplex_points=simplex_points,
                dim=dim,
                density_here=density_here,
                a0=a0,
                relaxation_weight=config.relaxation_weight,
                shape_force_scale=config.shape_force_scale,
                volume_force_scale=config.volume_force_scale,
                sliver_correction=config.sliver_correction,
                volume=volume,
            )
        elif config.irrel_force_scale > 0.0:
            center = np.mean(simplex_points, axis=0)
            for local_index, point_index in enumerate(simplex.tolist()):
                if int(states[point_index]) != STATE_MOBILE:
                    continue
                simplex_force[local_index] = config.irrel_force_scale * (
                    center - simplex_points[local_index]
                )

        for local_index, point_index in enumerate(simplex.tolist()):
            point_id = int(point_index)
            if suppress_corner_forces[point_id]:
                continue
            total_forces[point_id] += simplex_force[local_index]


def _classify_relevant_simplices(
    points: FloatArray,
    states: np.ndarray,
    geometry: FemGeometry,
    simplices: np.ndarray,
    simplex_measures: FloatArray,
    dim: int,
    config: ForceParameters,
) -> np.ndarray:
    """Return the legacy relevant-simplex mask for force calculation."""
    if len(simplices) == 0:
        return np.empty(0, dtype=bool)

    region_ids, probe_consistent = _classify_simplices_with_probes(
        points,
        simplices,
        geometry,
    )
    boundary_mask = _boundary_state_mask(points, states, geometry)
    all_boundary = np.all(boundary_mask[simplices], axis=1)
    boundary_ratio = _simplex_volume_order_ratio(points, simplices, dim)
    flat_boundary = all_boundary & (boundary_ratio < config.smallest_allowed_volume_ratio)
    return (
        (region_ids >= 0) & probe_consistent & (simplex_measures > BOUNDARY_FUZZ) & ~flat_boundary
    )
