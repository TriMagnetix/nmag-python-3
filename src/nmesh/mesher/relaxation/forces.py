"""Force calculations for the relaxation meshing pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from ..meshing_parameters import (
    default_boundary_node_force_fun,
    default_initial_relaxation_weight,
    default_relaxation_force_fun,
)
from ._constants import (
    BOUNDARY_FUZZ,
    DENSITY_EPSILON,
    STATE_BOUNDARY,
    STATE_FIXED,
    STATE_MOBILE,
    STATE_SIMPLE,
)
from ._types import FloatArray
from .geometry import FemGeometry
from .jit import accumulate_neighbor_forces_default
from .topology import _simplex_measures, _triangulate_points


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


def _build_neighbor_pairs(simplices: np.ndarray) -> np.ndarray:
    """Return sorted unique point-pair edges induced by the simplices."""

    if len(simplices) == 0:
        return np.empty((0, 2), dtype=np.int64)

    pairs: set[tuple[int, int]] = set()
    for simplex in simplices:
        for left, right in combinations(simplex.tolist(), 2):
            i = int(left)
            j = int(right)
            pairs.add((i, j) if i <= j else (j, i))
    return np.asarray(sorted(pairs), dtype=np.int64)


def _is_dynamic_state(state: int) -> bool:
    """Return whether the state participates in relaxation movement."""

    return state in (STATE_MOBILE, STATE_BOUNDARY)


def _is_boundary_interaction(state_a: int, state_b: int) -> bool:
    """Return whether a neighbor interaction should use the boundary law."""

    return (
        state_a in (STATE_FIXED, STATE_BOUNDARY, STATE_SIMPLE)
        or state_b in (STATE_FIXED, STATE_BOUNDARY, STATE_SIMPLE)
    )


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


def _neighbor_forces_python(
    points: FloatArray,
    states: np.ndarray,
    point_densities: FloatArray,
    neighbor_map: list[list[int]],
    dim: int,
    a0: float,
    neigh_force_scale: float,
    force_fun: Any,
    boundary_force_fun: Any,
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Fallback neighbor-force path for custom Python force callbacks."""

    total_forces = np.zeros((len(points), dim), dtype=float)
    neighbor_force_sums = np.zeros(len(points), dtype=float)
    neighbor_force_counts = np.zeros(len(points), dtype=int)

    for left, neighbors in enumerate(neighbor_map):
        point_left = points[left]
        state_left = int(states[left])
        for right in neighbors:
            if right <= left:
                continue
            contribution, scalar_force = _apply_pairwise_force(
                point_left=point_left,
                point_right=points[right],
                state_left=state_left,
                state_right=int(states[right]),
                density_left=float(point_densities[left]),
                density_right=float(point_densities[right]),
                dim=dim,
                a0=a0,
                neigh_force_scale=neigh_force_scale,
                force_fun=force_fun,
                boundary_force_fun=boundary_force_fun,
            )
            if scalar_force is None:
                continue
            neighbor_force_sums[left] += abs(scalar_force)
            neighbor_force_sums[right] += abs(scalar_force)
            neighbor_force_counts[left] += 1
            neighbor_force_counts[right] += 1
            if contribution is None:
                continue
            total_forces[left] += contribution
            total_forces[right] -= contribution

    return total_forces, neighbor_force_sums, neighbor_force_counts


def _apply_pairwise_force(
    *,
    point_left: FloatArray,
    point_right: FloatArray,
    state_left: int,
    state_right: int,
    density_left: float,
    density_right: float,
    dim: int,
    a0: float,
    neigh_force_scale: float,
    force_fun: Any,
    boundary_force_fun: Any,
) -> tuple[FloatArray | None, float | None]:
    """Compute the pairwise neighbor contribution for one point pair."""

    if not _is_dynamic_state(state_left) and not _is_dynamic_state(state_right):
        return None, None

    delta = point_right - point_left
    true_distance = float(np.linalg.norm(delta))
    if true_distance <= DENSITY_EPSILON:
        return None, None

    avg_density = 0.5 * (density_left + density_right)
    inv_length_scale = (avg_density ** (1.0 / max(dim, 1))) / max(a0, DENSITY_EPSILON)
    reduced_distance = true_distance * inv_length_scale
    scalar_force = float(
        (boundary_force_fun if _is_boundary_interaction(state_left, state_right) else force_fun)(
            reduced_distance
        )
    )
    scaled_force = neigh_force_scale * abs(scalar_force)
    if scalar_force == 0.0:
        return None, scaled_force
    return neigh_force_scale * (-scalar_force) * delta, scaled_force


def _extract_force_parameters(params: dict[str, Any], step: int) -> ForceParameters:
    """Resolve the mesher controller parameters used by ``compute_forces``."""

    settling_steps = int(params.get("controller_initial_settling_steps", 100))
    relaxation_weight_fun = params.get("initial_relaxation_weight_fun", default_initial_relaxation_weight)
    return ForceParameters(
        shape_force_scale=float(params.get("controller_shape_force_scale", 0.1)),
        volume_force_scale=float(params.get("controller_volume_force_scale", 0.0)),
        neigh_force_scale=float(params.get("controller_neigh_force_scale", 1.0)),
        irrel_force_scale=float(params.get("controller_irrel_elem_force_scale", 1.0)),
        sliver_correction=float(params.get("controller_sliver_correction", 1.0)),
        relaxation_weight=float(relaxation_weight_fun(step, settling_steps, 0.0, 1.0)),
        force_fun=params.get("relaxation_force_fun", default_relaxation_force_fun),
        boundary_force_fun=params.get("boundary_node_force_fun", default_boundary_node_force_fun),
    )


def _build_neighbor_map(point_count: int, simplices: np.ndarray) -> list[list[int]]:
    """Build undirected point adjacency from simplices."""

    neighbors: list[set[int]] = [set() for _ in range(point_count)]
    for simplex in simplices:
        for left, right in combinations(simplex.tolist(), 2):
            i = int(left)
            j = int(right)
            neighbors[i].add(j)
            neighbors[j].add(i)
    return [sorted(group) for group in neighbors]


def _compute_neighbor_forces(
    points: FloatArray,
    states: np.ndarray,
    simplices: np.ndarray,
    neighbor_map: list[list[int]],
    point_densities: FloatArray,
    dim: int,
    a0: float,
    config: ForceParameters,
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Compute neighbor-force contributions through the JIT or Python path."""

    point_count = len(points)
    if len(simplices) == 0:
        return (
            np.zeros((point_count, dim), dtype=float),
            np.zeros(point_count, dtype=float),
            np.zeros(point_count, dtype=int),
        )

    if (
        config.force_fun is default_relaxation_force_fun
        and config.boundary_force_fun is default_boundary_node_force_fun
    ):
        neighbor_pairs = _build_neighbor_pairs(simplices)
        return accumulate_neighbor_forces_default(
            np.asarray(points, dtype=np.float64),
            np.asarray(states, dtype=np.int64),
            point_densities.astype(np.float64, copy=False),
            neighbor_pairs,
            float(a0),
            config.neigh_force_scale,
        )

    return _neighbor_forces_python(
        points,
        states,
        point_densities,
        neighbor_map,
        dim,
        a0,
        config.neigh_force_scale,
        config.force_fun,
        config.boundary_force_fun,
    )


def _vertex_angle(points: FloatArray, simplex: np.ndarray, local_index: int, dim: int) -> float:
    """Return the angle or solid angle covered by a simplex at one vertex."""

    vertex = points[int(simplex[local_index])]
    others = [
        points[int(simplex[index])] - vertex
        for index in range(len(simplex))
        if index != local_index
    ]

    if dim == 1:
        return math.pi

    if dim == 2:
        first = others[0]
        second = others[1]
        denom = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denom <= DENSITY_EPSILON:
            return 0.0
        cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
        return float(math.acos(cosine))

    if dim == 3:
        a, b, c = others
        numer = abs(float(np.dot(a, np.cross(b, c))))
        denom = (
            float(np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c))
            + float(np.dot(a, b) * np.linalg.norm(c))
            + float(np.dot(a, c) * np.linalg.norm(b))
            + float(np.dot(b, c) * np.linalg.norm(a))
        )
        if numer <= DENSITY_EPSILON and denom <= DENSITY_EPSILON:
            return 0.0
        return float(2.0 * math.atan2(numer, max(denom, DENSITY_EPSILON)))

    return 1.0


def _shape_force_matrix(vertices: FloatArray, dim: int) -> FloatArray:
    """Build the covariance-derived shape force matrix for one simplex."""

    covariance = np.asarray(vertices.T @ vertices, dtype=float)
    covariance /= max(float(len(vertices) - 1), 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    safe_eigenvalues = np.clip(eigenvalues.astype(float, copy=False), DENSITY_EPSILON, None)
    product = float(np.prod(safe_eigenvalues))
    if product <= DENSITY_EPSILON:
        return np.zeros((dim, dim), dtype=float)

    scaled = safe_eigenvalues * ((1.0 / product) ** (1.0 / max(dim, 1)))
    transformed = np.asarray(np.diag(-np.log(np.clip(scaled, DENSITY_EPSILON, None))), dtype=float)
    return np.asarray(eigenvectors @ transformed @ eigenvectors.T, dtype=float)


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

    if len(simplices) == 0:
        return
    if (
        config.shape_force_scale <= 0.0
        and config.volume_force_scale <= 0.0
        and config.irrel_force_scale <= 0.0
    ):
        return

    centroid_regions = geometry.classify_points(np.mean(points[simplices], axis=1))
    suppress_corner_forces = angle_sums < _corner_force_threshold(dim)
    for simplex_index, simplex in enumerate(simplices):
        simplex_points = points[simplex]
        simplex_force = np.zeros_like(simplex_points)
        volume = float(simplex_measures[simplex_index])
        if centroid_regions[simplex_index] >= 0 and volume > BOUNDARY_FUZZ:
            density_here = float(np.mean(point_densities[np.asarray(simplex, dtype=int)]))
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
                if not _is_dynamic_state(int(states[point_index])):
                    continue
                simplex_force[local_index] = config.irrel_force_scale * (center - simplex_points[local_index])

        for local_index, point_index in enumerate(simplex.tolist()):
            point_id = int(point_index)
            if suppress_corner_forces[point_id]:
                continue
            total_forces[point_id] += simplex_force[local_index]


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
    full_angle = _full_angle(dim)

    for point_index, point in enumerate(points):
        point_average_force[point_index] = _average_neighbor_force(
            neighbor_force_sums[point_index],
            int(neighbor_force_counts[point_index]),
        )
        corrected_volume = _corrected_point_volume(
            simplex_measures=simplex_measures,
            incident=incident_simplices[point_index],
            angle=float(angle_sums[point_index]),
            dim=dim,
            a0=a0,
            is_boundary=bool(boundary_mask[point_index]),
            full_angle=full_angle,
        )
        density_here = float(point_densities[point_index])
        point_density[point_index] = _point_density_ratio(
            density_here=density_here,
            corrected_volume=corrected_volume,
            a0=a0,
            dim=dim,
        )
        point_effective_force[point_index] = _point_effective_force(
            total_force=total_forces[point_index],
            state=int(states[point_index]),
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


def _full_angle(dim: int) -> float:
    """Return the complete angular measure used for Voronoi-style correction."""

    if dim == 1:
        return 1.5
    if dim == 2:
        return 2.0 * math.pi
    if dim == 3:
        return 4.0 * math.pi
    return 1.0


def _average_neighbor_force(force_sum: float, force_count: int) -> float:
    """Return the mean absolute neighbor-force magnitude for a point."""

    if force_count <= 0:
        return 0.0
    return force_sum / float(force_count)


def _corrected_point_volume(
    *,
    simplex_measures: FloatArray,
    incident: list[int],
    angle: float,
    dim: int,
    a0: float,
    is_boundary: bool,
    full_angle: float,
) -> float:
    """Return the local corrected control volume for one point."""

    if incident:
        local_volume = float(np.sum(simplex_measures[np.asarray(incident, dtype=int)])) / float(dim + 1)
    else:
        local_volume = _sphere_volume(0.5 * a0, dim)

    if angle <= DENSITY_EPSILON:
        return 1.0e-4 * _sphere_volume(0.5 * a0, dim)

    correction = (
        full_angle / angle
        if dim in (1, 2, 3)
        else max(1.0, float(dim + 1) / float(len(incident) or 1))
    )
    corrected_volume = local_volume * correction
    if is_boundary:
        corrected_volume *= 1.2
    return corrected_volume


def _point_density_ratio(
    *,
    density_here: float,
    corrected_volume: float,
    a0: float,
    dim: int,
) -> float:
    """Return the desired-to-actual local density ratio for one point."""

    effective_rod_length = a0 / (density_here ** (1.0 / max(dim, 1)))
    ideal_local_volume = _sphere_volume(0.5 * effective_rod_length, dim)
    return ideal_local_volume / max(corrected_volume, DENSITY_EPSILON)


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


def compute_forces(
    points: FloatArray,
    states: np.ndarray,
    geometry: FemGeometry,
    a0: float,
    params: dict[str, Any],
    step: int,
) -> ForceSummary:
    """Compute neighbor, shape, volume, and irrelevant-element forces."""

    point_count = len(points)
    dim = geometry.dim
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
    simplex_measures, incident_simplices, angle_sums = _simplex_incidence_data(points, simplices, dim)
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
