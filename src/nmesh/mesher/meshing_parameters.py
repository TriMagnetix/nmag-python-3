import copy
import logging
from dataclasses import dataclass
from enum import IntEnum
from functools import partialmethod
from typing import Any

from mock_features import MockFeatures

log = logging.getLogger(__name__)

# Point density constants, these control probabilistic insertion/deletion of mesh points during relaxation
DENSITY_ADD_PROBABILITY = 0.1  # 10% chance to add point when density too low
FORCE_LOW_ADD_PROBABILITY = 0.2  # 20% chance to add when force < threshold
FORCE_LOW_THRESHOLD = 0.07  # Force threshold below which points may be added

DENSITY_DELETE_BASE_PROBABILITY = 0.3  # Base 30% chance to delete when density too high
DENSITY_DELETE_SLOPE = 0.1  # Additional 10% per unit above threshold
FORCE_HIGH_DELETE_BASE_PROBABILITY = 0.4  # Base 40% chance to delete when force too high
FORCE_HIGH_DELETE_SLOPE = 0.1  # Additional 10% per unit above 0.5
FORCE_HIGH_THRESHOLD = 0.5  # Force threshold above which points may be deleted

# Safety epsilon for division operations
EPSILON_DIVISION_SAFETY = 1e-15


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    legacy_name: str
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

PUBLIC_PARAMETER_SPECS_BY_LEGACY = {
    spec.legacy_name: spec for spec in PUBLIC_PARAMETER_SPECS
}

LEGACY_TO_INTERNAL = {
    spec.legacy_name: spec.internal_name for spec in PUBLIC_PARAMETER_SPECS
}
INTERNAL_TO_LEGACY = {
    spec.internal_name: spec.legacy_name for spec in PUBLIC_PARAMETER_SPECS
}
LEGACY_SETTER_NAMES = (
    "shape_force_scale",
    "volume_force_scale",
    "neigh_force_scale",
    "irrel_elem_force_scale",
    "time_step_scale",
    "thresh_add",
    "thresh_del",
    "topology_threshold",
    "tolerated_rel_move",
    "max_steps",
    "initial_settling_steps",
    "sliver_correction",
    "smallest_volume_ratio",
    "max_relaxation",
)


class PointFate(IntEnum):
    DO_NOTHING = 0
    ADD_ANOTHER = 1
    DELETE = 2


class SimplexRegion(IntEnum):
    OUTSIDE = 0
    INSIDE = 1


def default_initial_relaxation_weight(iteration_step, max_step, init_val, final_val):
    """Linear function from init_val to final_val, saturating at max_step."""
    if max_step <= 0:
        return final_val
    return init_val + (final_val - init_val) * min(
        1.0, float(iteration_step) / float(max_step)
    )


def default_relaxation_force_fun(reduced_distance):
    """Repulsing force between two mobile nodes."""
    if reduced_distance > 1.0:
        return 0.0
    return 1.0 - reduced_distance


def default_boundary_node_force_fun(reduced_distance):
    """Strongly repelling potential for boundary points."""
    if reduced_distance > 1.0:
        return 0.0
    if reduced_distance < EPSILON_DIVISION_SAFETY:
        return 1e12
    return 1.0 / reduced_distance - 1.0


def default_handle_point_density_fun(rng, avg_stats, thresh_add, thresh_del):
    """Default function to insert or delete points based on density and force. 0"""
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
        prob = FORCE_HIGH_DELETE_BASE_PROBABILITY + (avg_force - FORCE_HIGH_THRESHOLD) * FORCE_HIGH_DELETE_SLOPE
        if rng.random() < prob:
            log.debug("Fth (avg_force=%s) - axing point.", avg_force)
            return PointFate.DELETE
        return PointFate.DO_NOTHING

    return PointFate.DO_NOTHING


def _candidate_keys(name: str) -> list[str]:
    keys = [name]
    internal = LEGACY_TO_INTERNAL.get(name)
    legacy = INTERNAL_TO_LEGACY.get(name)

    if internal is not None and internal not in keys:
        keys.append(internal)
    if legacy is not None and legacy not in keys:
        keys.append(legacy)

    return keys


class MeshingParameters(MockFeatures):
    def __init__(self, string=None, file=None):
        super().__init__()
        self.dim = None
        self._setup_defaults()
        if file:
            self.from_file(file)
        if string:
            self.from_string(string)
        self.add_section("user-modifications")

    def _setup_defaults(self):
        self._params = {
            "nr_probes_for_determining_volume": 100000,
            "boundary_condition_acceptable_fuzz": 1e-6,
            "boundary_condition_max_nr_correction_steps": 200,
            "boundary_condition_debuglevel": 0,
            "relaxation_debuglevel": 0,
            "controller_step_limit_min": 500,
            "controller_max_time_step": 10.0,
            "initial_relaxation_weight_fun": default_initial_relaxation_weight,
            "relaxation_force_fun": default_relaxation_force_fun,
            "boundary_node_force_fun": default_boundary_node_force_fun,
            "handle_point_density_fun": default_handle_point_density_fun,
        }
        self._params.update(
            {spec.internal_name: spec.default for spec in PUBLIC_PARAMETER_SPECS}
        )

    def _get_section_name(self):
        if self.dim is None:
            raise RuntimeError("Dimension not set in MeshingParameters")
        return f"nmesh-{self.dim}D" if self.dim in [2, 3] else "nmesh-ND"

    def _lookup(self, section: str, name: str) -> Any | None:
        for key in _candidate_keys(name):
            value = self.get(section, key)
            if value is not None:
                return value
        return None

    def _canonical_key(self, name: str) -> str:
        internal = LEGACY_TO_INTERNAL.get(name, name)
        if internal in self._params or name in LEGACY_TO_INTERNAL:
            return internal
        return name

    def __getitem__(self, name: str) -> Any | None:
        user_value = self._lookup("user-modifications", name)
        if user_value is not None:
            return user_value

        if self.dim is not None:
            section_value = self._lookup(self._get_section_name(), name)
            if section_value is not None:
                return section_value

        canonical = self._canonical_key(name)
        if canonical in self._params:
            return self._params[canonical]

        return None

    def __setitem__(self, key: str, value: Any) -> None:
        canonical = self._canonical_key(key)
        self._params[canonical] = value
        self.set("user-modifications", canonical, value)

    def _sync_dimension_section(self, dim: int) -> str:
        self.dim = dim
        section = self._get_section_name()

        for key, value in self.items("user-modifications"):
            if not isinstance(key, str):
                continue
            section_key = INTERNAL_TO_LEGACY.get(key, key)
            self.set(section, section_key, value)

        return section

    def to_mesher_config(self, dim):
        self._sync_dimension_section(dim)

        resolved = {}
        for spec in PUBLIC_PARAMETER_SPECS:
            value = self[spec.legacy_name]
            if value is None:
                continue
            resolved[spec.internal_name] = spec.cast(value)

        return resolved

    def apply_to_mesher(self, mesher, dim):
        self._sync_dimension_section(dim)
        mesher.setdefault("parameters", {})

        for spec in PUBLIC_PARAMETER_SPECS:
            value = self[spec.legacy_name]
            if value is None:
                continue
            mesher["parameters"][spec.internal_name] = spec.cast(value)

        return mesher

    def _set_parameter(self, name, value):
        self[name] = PUBLIC_PARAMETER_SPECS_BY_LEGACY[name].cast(value)

    def copy(self):
        return copy.deepcopy(self)


for _name in LEGACY_SETTER_NAMES:
    setattr(
        MeshingParameters,
        f"set_{_name}",
        partialmethod(MeshingParameters._set_parameter, _name),
    )
