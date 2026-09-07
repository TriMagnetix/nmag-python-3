"""State initialization and configuration access for relaxation meshing."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .._constants import BOUNDARY_FUZZ, DEFAULT_RNG_SEED, STATE_BOUNDARY, STATE_MOBILE
from .._types import FloatArray
from ..geometry import FemGeometry
from ..seeding import _prepare_initial_points
from ..topology import assemble_raw_mesh
from ..topology.finalize import snap_final_boundary_points

log = logging.getLogger(__name__)


class RelaxationEngineStateMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def __init__(
        self,
        geometry: FemGeometry,
        mesher: dict[str, Any],
        a0: float,
        fixed_points: FloatArray,
        mobile_points: FloatArray,
        simply_points: FloatArray,
        periodic: list[float] | list[bool],
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize the engine state from geometry, seeds, and mesher parameters."""

        self.geometry = geometry
        self.params = dict(mesher.get("parameters", {}))
        self.a0 = float(a0)
        self.periodic = list(periodic)
        self.rng = rng or np.random.default_rng(DEFAULT_RNG_SEED)
        self.points, self.states = _prepare_initial_points(
            geometry,
            self.a0,
            fixed_points,
            mobile_points,
            simply_points,
            self.periodic,
            self.rng,
            self.params,
        )
        self.step = 0
        self.finished = False
        self.last_addition_deletion_step = 0
        self.last_max_displacement = math.inf
        self.last_max_effective_force = math.inf
        self.last_point_density = np.zeros(len(self.points), dtype=float)
        self.max_rel_movement_since_last_triangulation = 0.0
        self.current_simplices: np.ndarray | None = None
        self.points_at_last_triangulation = np.array(self.points, copy=True)
        self._refresh_boundary_states()
        self.cached_raw_mesh = assemble_raw_mesh(
            self.points,
            self.geometry,
            self.periodic,
            states=self.states,
            params=self.params,
        )

    @property
    def max_steps(self) -> int:
        """Return the configured maximum relaxation-step budget."""

        return int(self.params.get("controller_step_limit_max", 1000))

    @property
    def tolerated_rel_move(self) -> float:
        """Return the relative movement threshold used for convergence."""

        return float(self.params.get("controller_tolerated_rel_movement", 0.002))

    @property
    def time_step_scale(self) -> float:
        """Return the scale factor applied to each relaxation displacement."""

        return float(self.params.get("controller_time_step_scale", 0.1))

    @property
    def max_time_step(self) -> float:
        """Return the largest time-step magnitude allowed by the controller."""

        return float(self.params.get("controller_max_time_step", 10.0))

    @property
    def boundary_drift_tolerance(self) -> float:
        """Return the tolerated increase in boundary distance for boundary points.

        Legacy boundary points are only allowed to move when they do not drift
        farther from the boundary. The Python port keeps only the global
        floating-point boundary fuzz here; it does not scale this tolerance with
        ``a0`` because that would turn a numerical guard into different meshing
        behavior.
        """

        return BOUNDARY_FUZZ

    @property
    def min_equilibrium_steps(self) -> int:
        """Return the minimum number of steps before equilibrium can stop the loop."""

        return int(self.params.get("controller_step_limit_min", 500))

    @property
    def post_change_settling_steps(self) -> int:
        """Return the legacy settling window after point add/delete checks."""

        return 50

    def _rebuild_mesh(self, *, final: bool = False) -> None:
        """Refresh the cached ``RawMesh`` snapshot from the current point cloud."""

        points, states = (
            self._final_output_points_and_states() if final else (self.points, self.states)
        )
        self.cached_raw_mesh = assemble_raw_mesh(
            points,
            self.geometry,
            self.periodic,
            states=states,
            params=self.params,
        )

    def _final_output_points_and_states(self) -> tuple[FloatArray, np.ndarray]:
        """Return the final point cloud after the legacy high-density cleanup."""

        if len(self.last_point_density) != len(self.points):
            return self.points, self.states
        dynamic_mask = np.isin(self.states, [STATE_MOBILE, STATE_BOUNDARY])
        keep = (~dynamic_mask) | (self.last_point_density < 100.0)
        if np.all(keep):
            points = self.points
            states = self.states
        else:
            points = self.points[keep]
            states = self.states[keep]
        dynamic_mask = np.isin(states, [STATE_MOBILE, STATE_BOUNDARY])
        domain_mask = self.geometry.classify_points(points) >= 0
        keep_domain = (~dynamic_mask) | domain_mask
        if not np.all(keep_domain):
            points = points[keep_domain]
            states = states[keep_domain]
        return snap_final_boundary_points(points, states, self.geometry, self.a0, self.params)
