from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, TypeAlias

from ..utils.constants import MIN_DIVISION_MAGNITUDE

log = logging.getLogger(__name__)

# Point density constants: Control probabilistic insertion/deletion of mesh points during relaxation.
DENSITY_ADD_PROBABILITY = 0.1  # 10% chance to add point when density too low
FORCE_LOW_ADD_PROBABILITY = 0.2  # 20% chance to add when force < threshold
FORCE_LOW_THRESHOLD = 0.07  # Force threshold below which points may be added

DENSITY_DELETE_BASE_PROBABILITY = 0.3  # Base 30% chance to delete when density too high
DENSITY_DELETE_SLOPE = 0.1  # Additional 10% per unit above threshold
FORCE_HIGH_DELETE_BASE_PROBABILITY = 0.4  # Base 40% chance to delete when force too high
FORCE_HIGH_DELETE_SLOPE = 0.1  # Additional 10% per unit above 0.5
FORCE_HIGH_THRESHOLD = 0.5  # Force threshold above which points may be deleted


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Specification for a meshing parameter with public and internal names.

    The public_name is the user-friendly API (concise, intuitive).
    The internal_name is the implementation detail (verbose, namespaced).
    This separation is good design even without backward compatibility concerns.
    """

    public_name: str
    internal_name: str
    default: int | float
    cast: type[int] | type[float]


PUBLIC_PARAMETER_SPECS = (
    ParameterSpec("shape_force_scale", "controller_shape_force_scale", 0.1, float),
    ParameterSpec("volume_force_scale", "controller_volume_force_scale", 0.0, float),
    ParameterSpec("neigh_force_scale", "controller_neigh_force_scale", 1.0, float),
    ParameterSpec(
        "irrel_elem_force_scale",
        "controller_irrel_elem_force_scale",
        1.0,
        float,
    ),
    ParameterSpec("time_step_scale", "controller_time_step_scale", 0.1, float),
    ParameterSpec("thresh_add", "controller_thresh_add", 1.0, float),
    ParameterSpec("thresh_del", "controller_thresh_del", 2.0, float),
    ParameterSpec("topology_threshold", "controller_topology_threshold", 0.2, float),
    ParameterSpec(
        "tolerated_rel_move",
        "controller_tolerated_rel_movement",
        0.002,
        float,
    ),
    ParameterSpec("max_steps", "controller_step_limit_max", 1000, int),
    ParameterSpec(
        "initial_settling_steps",
        "controller_initial_settling_steps",
        100,
        int,
    ),
    ParameterSpec("sliver_correction", "controller_sliver_correction", 1.0, float),
    ParameterSpec(
        "smallest_volume_ratio",
        "controller_smallest_allowed_volume_ratio",
        1.0,
        float,
    ),
    ParameterSpec("max_relaxation", "controller_movement_max_freedom", 3.0, float),
    ParameterSpec(
        "initial_points_volume_ratio",
        "controller_initial_points_volume_ratio",
        0.9,
        float,
    ),
    ParameterSpec(
        "splitting_connection_ratio",
        "controller_splitting_connection_ratio",
        1.6,
        float,
    ),
    ParameterSpec(
        "exp_neigh_force_scale",
        "controller_exp_neigh_force_scale",
        0.9,
        float,
    ),
)

PUBLIC_PARAMETER_SPECS_BY_NAME = {spec.public_name: spec for spec in PUBLIC_PARAMETER_SPECS}

PUBLIC_TO_INTERNAL = {spec.public_name: spec.internal_name for spec in PUBLIC_PARAMETER_SPECS}
INTERNAL_TO_PUBLIC = {spec.internal_name: spec.public_name for spec in PUBLIC_PARAMETER_SPECS}


class PointFate(IntEnum):
    DO_NOTHING = 0
    ADD_ANOTHER = 1
    DELETE = 2


class SimplexRegion(IntEnum):
    OUTSIDE = 0
    INSIDE = 1


class RandomSource(Protocol):
    def random(self) -> float: ...


RelaxationWeightFunction: TypeAlias = Callable[[int, int, float, float], float]
ForceFunction: TypeAlias = Callable[[float], float]
PointDensityHandler: TypeAlias = Callable[
    [RandomSource, tuple[float, float], float, float], PointFate
]
MesherParameter: TypeAlias = (
    int | float | RelaxationWeightFunction | ForceFunction | PointDensityHandler
)


def default_initial_relaxation_weight(
    iteration_step: int, max_step: int, init_val: float, final_val: float
) -> float:
    """Linear function from init_val to final_val, saturating at max_step.

    Args:
        iteration_step: Current iteration step number
        max_step: Maximum number of steps for interpolation
        init_val: Initial weight value at step 0
        final_val: Final weight value at max_step and beyond

    Returns:
        Interpolated weight value between init_val and final_val
    """
    if max_step <= 0:
        return final_val
    return init_val + (final_val - init_val) * min(1.0, float(iteration_step) / float(max_step))


def default_relaxation_force_fun(reduced_distance: float) -> float:
    """Repulsing force between two mobile nodes.

    Args:
        reduced_distance: Distance normalized by ideal neighbor distance

    Returns:
        Repulsive force magnitude (0.0 if distance > 1.0, else 1.0 - distance)
    """
    if reduced_distance > 1.0:
        return 0.0
    return 1.0 - reduced_distance


def default_boundary_node_force_fun(reduced_distance: float) -> float:
    """Strongly repelling potential for boundary points.

    This implements a 1/r - 1 potential that enforces strong repulsion
    near boundary nodes to prevent mesh points from violating boundary conditions.

    Args:
        reduced_distance: Distance normalized by ideal neighbor distance

    Returns:
        Repulsive force magnitude (very large for small distances, 0.0 if distance > 1.0)
    """
    if reduced_distance > 1.0:
        return 0.0
    if reduced_distance < MIN_DIVISION_MAGNITUDE:
        return 1e12
    return 1.0 / reduced_distance - 1.0


def default_handle_point_density_fun(
    rng: RandomSource,
    avg_stats: tuple[float, float],
    thresh_add: float,
    thresh_del: float,
) -> PointFate:
    """Default function to insert or delete points based on density and force.

    Implements probabilistic point insertion/deletion based on local Voronoi density
    and neighbor force magnitudes. Matches OCaml mdefault_controller_handle_point_density_fun.

    Args:
        rng: Random number generator with .random() method
        avg_stats: Tuple of (avg_density, avg_force) for the point
        thresh_add: Density threshold below which points may be added
        thresh_del: Density threshold above which points may be deleted

    Returns:
        PointFate enum: ADD_ANOTHER, DELETE, or DO_NOTHING
    """
    avg_density, avg_force = avg_stats
    if avg_density < thresh_add:
        if rng.random() < DENSITY_ADD_PROBABILITY:
            log.debug("Dtl (dens_avg=%s) - adding point.", avg_density)
            return PointFate.ADD_ANOTHER
        return PointFate.DO_NOTHING

    if avg_force < FORCE_LOW_THRESHOLD:
        if rng.random() < FORCE_LOW_ADD_PROBABILITY:
            log.debug("Ftl (avg_force=%s) - adding point.", avg_force)
            return PointFate.ADD_ANOTHER
        return PointFate.DO_NOTHING

    if avg_density > thresh_del:
        prob = DENSITY_DELETE_BASE_PROBABILITY + (avg_density - thresh_del) * DENSITY_DELETE_SLOPE
        if rng.random() < prob:
            log.debug("Dth (dens_avg=%s) - axing point.", avg_density)
            return PointFate.DELETE
        return PointFate.DO_NOTHING

    if avg_force > FORCE_HIGH_THRESHOLD:
        prob = (
            FORCE_HIGH_DELETE_BASE_PROBABILITY
            + (avg_force - FORCE_HIGH_THRESHOLD) * FORCE_HIGH_DELETE_SLOPE
        )
        if rng.random() < prob:
            log.debug("Fth (avg_force=%s) - axing point.", avg_force)
            return PointFate.DELETE
        return PointFate.DO_NOTHING

    return PointFate.DO_NOTHING


def _candidate_keys(name: str) -> list[str]:
    """Returns all possible names (public and internal) for a parameter.

    This allows users to refer to parameters by either their public API name
    or the internal implementation name, whichever is more convenient.
    """
    keys = [name]
    internal = PUBLIC_TO_INTERNAL.get(name)
    public = INTERNAL_TO_PUBLIC.get(name)

    if internal is not None and internal not in keys:
        keys.append(internal)
    if public is not None and public not in keys:
        keys.append(public)

    return keys


def _cast_numeric_parameter(spec: ParameterSpec, value: object) -> int | float:
    if not isinstance(value, (str, int, float)):
        raise TypeError(f"Meshing parameter {spec.public_name!r} must be numeric.")
    return spec.cast(value)
