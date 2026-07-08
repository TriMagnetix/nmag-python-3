"""Iterative meshing engine for the pure-Python relaxation pipeline."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from ...backend import RawMesh
from ...geometry.primitives import Body
from ..driver import MeshEngineCommand, MeshEngineStatus
from ..meshing_parameters import PointFate, default_handle_point_density_fun
from ._constants import BOUNDARY_FUZZ, DEFAULT_RNG_SEED, STATE_BOUNDARY, STATE_FIXED, STATE_MOBILE, STATE_SIMPLE
from ._types import DensityFunction, EngineResult, FloatArray
from .finalize import snap_final_boundary_points
from .forces import ForceSummary, compute_forces
from .geometry import FemGeometry, fem_geometry_from_bodies
from .recovery import mirror_surface_recovery_points
from .seeding import _as_float_array, _classify_dynamic_states, _dedupe_points, _prepare_initial_points
from .topology import _callback_mesh_info, assemble_raw_mesh

log = logging.getLogger(__name__)


class RelaxationEngine:
    """Small iterative meshing engine that exposes the legacy driver protocol."""

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

        points, states = self._final_output_points_and_states() if final else (self.points, self.states)
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

    def _mark_topology_stale(self) -> None:
        """Request a fresh Delaunay triangulation before the next force step."""

        self.current_simplices = None
        self.max_rel_movement_since_last_triangulation = 0.0
        self.points_at_last_triangulation = np.array(self.points, copy=True)

    def _refresh_boundary_states(self) -> None:
        """Refresh dynamic points between mobile and boundary states."""

        dynamic_mask = np.isin(self.states, [STATE_MOBILE, STATE_BOUNDARY])
        if not np.any(dynamic_mask):
            return
        refreshed = _classify_dynamic_states(self.geometry, self.points[dynamic_mask], self.a0)
        self.states[dynamic_mask] = refreshed

    def _effective_time_step(self, force_summary: ForceSummary) -> float:
        """Compute the controller-like time step for the current force field."""

        freedom = float(self.params.get("controller_movement_max_freedom", 3.0))
        movement_weight = self._initial_relaxation_weight(freedom, 1.0)
        capped_max_time_step = movement_weight * self.max_time_step
        if force_summary.max_effective_force <= BOUNDARY_FUZZ:
            return capped_max_time_step
        return min(
            capped_max_time_step,
            (movement_weight * self.time_step_scale) / force_summary.max_effective_force,
        )

    def _initial_relaxation_weight(self, init_val: float, final_val: float) -> float:
        """Evaluate the legacy initial-relaxation ramp for this step."""

        settling_steps = int(self.params.get("controller_initial_settling_steps", 100))
        weight_fun = self.params.get("initial_relaxation_weight_fun")
        if callable(weight_fun):
            return float(weight_fun(self.step, settling_steps, init_val, final_val))
        fraction = min(1.0, float(self.step) / max(float(settling_steps), 1.0))
        return init_val + (final_val - init_val) * fraction

    def _density_thresholds(self) -> tuple[float, float]:
        """Return add/delete thresholds with the legacy initial relaxation ramp."""

        freedom = float(self.params.get("controller_movement_max_freedom", 3.0))
        thresh_add = (
            self._initial_relaxation_weight(-0.1 * freedom, 0.0)
            + float(self.params.get("controller_thresh_add", 1.0))
        )
        thresh_del = (
            self._initial_relaxation_weight(0.1 * freedom, 0.0)
            + float(self.params.get("controller_thresh_del", 2.0))
        )
        return thresh_add, thresh_del

    @staticmethod
    def _is_positive_square_number(value: int) -> bool:
        """Return whether ``value`` is a positive perfect square."""

        if value <= 0:
            return False
        root = int(math.sqrt(value) + 0.5)
        return root * root == value

    def _should_attempt_point_change(self) -> bool:
        """Return whether the legacy controller schedules point fate changes."""

        return (
            self.step < self.max_steps
            and self._is_positive_square_number(self.step - 10)
        )

    def _step_limit_reached(self) -> bool:
        """Return whether the legacy max-step and post-change settling rules are met."""

        return (
            self.step >= self.max_steps
            and self.step >= self.last_addition_deletion_step + self.post_change_settling_steps
        )

    def _topology_threshold(self) -> float:
        """Return the relaxed legacy threshold for topology refresh."""

        freedom = float(self.params.get("controller_movement_max_freedom", 3.0))
        threshold = float(self.params.get("controller_topology_threshold", 0.2))
        return self._initial_relaxation_weight(freedom, 1.0) * threshold

    def _record_triangulation(self, force_summary: ForceSummary) -> None:
        """Cache the current topology after a force calculation builds it."""

        if self.current_simplices is None:
            self.current_simplices = np.asarray(force_summary.simplices, dtype=int)
            self.points_at_last_triangulation = np.array(self.points, copy=True)
            self.max_rel_movement_since_last_triangulation = 0.0

    def _update_topology_movement(self) -> None:
        """Track movement since the last triangulation in OCaml controller units."""

        if len(self.points) != len(self.points_at_last_triangulation):
            self._mark_topology_stale()
            return

        if len(self.points) == 0:
            self.max_rel_movement_since_last_triangulation = 0.0
            return

        displacements = np.linalg.norm(self.points - self.points_at_last_triangulation, axis=1)
        density_scale = np.asarray(
            [
                self.geometry.density_at(point) ** (1.0 / max(self.geometry.dim, 1))
                for point in self.points
            ],
            dtype=float,
        )
        scaled = displacements * density_scale / max(self.a0, BOUNDARY_FUZZ)
        self.max_rel_movement_since_last_triangulation = max(
            self.max_rel_movement_since_last_triangulation,
            float(np.max(scaled, initial=0.0)),
        )

    def _refresh_topology_if_needed(self) -> None:
        """Invalidate cached topology when movement exceeds the legacy threshold."""

        if self.max_rel_movement_since_last_triangulation > self._topology_threshold():
            self._mark_topology_stale()

    def _attempt_add_delete_points(self, force_summary: ForceSummary) -> None:
        """Apply the mesher density heuristic to add or remove mobile points."""

        if len(self.points) == 0:
            return

        additions, removals = self._evaluate_point_densities(force_summary)
        recovery_points = mirror_surface_recovery_points(
            self.points,
            self.states,
            self.geometry,
            self.a0,
            force_summary.simplices,
        )
        if len(recovery_points) > 0:
            additions.extend(recovery_points)
        self._apply_point_changes(additions, removals)
        self._refresh_boundary_states()

    def _evaluate_point_densities(
        self,
        force_summary: ForceSummary,
    ) -> tuple[list[FloatArray], list[int]]:
        """Decide which points to add or remove from the current cloud."""

        handler = self.params.get("handle_point_density_fun", default_handle_point_density_fun)
        thresh_add, thresh_del = self._density_thresholds()
        additions: list[FloatArray] = []
        removals: list[int] = []

        for index, state in enumerate(self.states):
            if state not in (STATE_MOBILE, STATE_BOUNDARY) or index in removals:
                continue

            neigh = force_summary.neighbor_map[index]
            if not neigh:
                continue

            point = self.points[index]
            avg_density = float(force_summary.point_density[index])
            avg_force = float(force_summary.point_average_force[index])

            fate = handler(self.rng, (avg_density, avg_force), thresh_add, thresh_del)
            if fate == PointFate.ADD_ANOTHER:
                additions.append(self._random_point_close_to(point))
            elif fate == PointFate.DELETE and len(self.points) - len(removals) > self.geometry.dim + 1:
                removals.append(index)

        return additions, removals

    def _random_point_close_to(self, point: FloatArray) -> FloatArray:
        """Return a Gaussian insertion candidate using the legacy rod-length scale."""

        density_here = self.geometry.density_at(point)
        effective_rod_length = self.a0 * (density_here ** (-1.0 / max(self.geometry.dim, 1)))
        candidate = np.asarray(
            self.rng.normal(loc=np.asarray(point, dtype=float), scale=effective_rod_length),
            dtype=float,
        )
        if self.geometry.classify_points(candidate.reshape(1, -1))[0] >= 0:
            return candidate
        return self.geometry.project_segment_to_domain(np.asarray(point, dtype=float), candidate)

    def _apply_point_changes(self, additions: list[FloatArray], removals: list[int]) -> None:
        """Mutate the point cloud according to the evaluated add/remove plan."""

        if removals:
            keep = np.ones(len(self.points), dtype=bool)
            keep[np.asarray(removals, dtype=int)] = False
            self.points = self.points[keep]
            self.states = self.states[keep]
            self._mark_topology_stale()

        if additions:
            additions_arr = _dedupe_points(np.asarray(additions, dtype=float))
            if len(additions_arr) > 0:
                self.points = np.vstack((self.points, additions_arr))
                addition_states = _classify_dynamic_states(self.geometry, additions_arr, self.a0)
                self.states = np.concatenate(
                    (
                        self.states,
                        addition_states,
                    )
                )
                self._mark_topology_stale()

    def _compute_constrained_displacement(
        self,
        point: FloatArray,
        state: int,
        displacement: FloatArray,
        max_norm: float,
    ) -> FloatArray:
        """Return a clipped, boundary-aware candidate position for one point."""

        bounded_displacement = np.array(displacement, copy=True)
        if state == STATE_BOUNDARY:
            normal = self.geometry.boundary_normal(point)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > BOUNDARY_FUZZ:
                bounded_displacement -= float(np.dot(bounded_displacement, normal)) * (normal / normal_norm)

        disp_norm = float(np.linalg.norm(bounded_displacement))
        if disp_norm > max_norm > 0.0:
            bounded_displacement *= max_norm / disp_norm

        candidate = point + bounded_displacement
        if state == STATE_BOUNDARY:
            old_boundary_distance = self.geometry.boundary_distance(point)
            new_boundary_distance = self.geometry.boundary_distance(candidate)
            if new_boundary_distance > old_boundary_distance + self.boundary_drift_tolerance:
                return np.asarray(point, dtype=float)

        if self.geometry.classify_points(candidate.reshape(1, -1))[0] < 0:
            candidate = self.geometry.project_segment_to_domain(point, candidate)

        if self.geometry.classify_points(candidate.reshape(1, -1))[0] >= 0:
            return np.asarray(candidate, dtype=float)
        return np.asarray(point, dtype=float)

    def _apply_relaxation_displacements(
        self,
        force_summary: ForceSummary,
        time_step: float,
    ) -> float:
        """Move dynamic points and return the largest density-scaled relative step."""

        new_points = np.array(self.points, copy=True)
        max_relative_displacement = 0.0
        max_norm = math.inf
        for index, state in enumerate(self.states):
            if state not in (STATE_MOBILE, STATE_BOUNDARY):
                continue

            point = self.points[index]
            displacement = np.array(force_summary.total_forces[index], copy=True) * time_step
            candidate = self._compute_constrained_displacement(point, int(state), displacement, max_norm)
            new_points[index] = candidate
            density_scale = self.geometry.density_at(candidate) ** (1.0 / max(self.geometry.dim, 1))
            relative_displacement = (
                float(np.linalg.norm(candidate - point))
                * density_scale
                / max(self.a0, BOUNDARY_FUZZ)
            )
            max_relative_displacement = max(max_relative_displacement, relative_displacement)
        self.points = new_points
        return max_relative_displacement

    def _check_convergence(self) -> None:
        """Update the engine state when a convergence condition is satisfied."""

        if (
            self.step > self.min_equilibrium_steps
            and self.last_max_displacement <= self.tolerated_rel_move
        ):
            log.debug(
                "Relaxation finished by movement convergence at step %d (rel_move=%g, effective_force=%g)",
                self.step,
                self.last_max_displacement,
                self.last_max_effective_force,
            )
            self.finished = True
            return

        if (
            self.step >= 2 and self.last_max_effective_force <= BOUNDARY_FUZZ
        ):
            log.debug(
                "Relaxation finished by effective-force equilibrium at step %d (effective_force=%g, rel_move=%g)",
                self.step,
                self.last_max_effective_force,
                self.last_max_displacement,
            )
            self.finished = True

    def _step_once(self) -> None:
        """Execute one relaxation step over all currently mobile or boundary points."""

        if len(self.points) < self.geometry.dim + 1:
            self.finished = True
            return

        force_summary = compute_forces(
            self.points,
            self.states,
            self.geometry,
            self.a0,
            self.params,
            self.step,
            simplices=self.current_simplices,
        )
        self._record_triangulation(force_summary)
        self.last_point_density = np.asarray(force_summary.point_density, dtype=float)
        time_step = self._effective_time_step(force_summary)
        self.last_max_displacement = self._apply_relaxation_displacements(force_summary, time_step)
        self.last_max_effective_force = force_summary.max_effective_force
        self._refresh_boundary_states()
        self._update_topology_movement()
        self._refresh_topology_if_needed()

        if self._should_attempt_point_change():
            self.last_addition_deletion_step = self.step
            self._attempt_add_delete_points(force_summary)

        self._check_convergence()

    def callback_payload(self) -> list[list[Any]]:
        """Return the callback payload for the engine's current mesh snapshot."""

        return _callback_mesh_info(self.cached_raw_mesh)

    def run(self, command: MeshEngineCommand) -> EngineResult:
        """Handle one driver command and return the next engine status tuple."""

        if command == MeshEngineCommand.DO_EXTRACT:
            self._rebuild_mesh()
            return MeshEngineStatus.PRODUCED_INTERMEDIATE_MESH, (
                self.callback_payload(),
                self.run,
            )

        if command != MeshEngineCommand.DO_STEP:
            raise ValueError(f"Unsupported mesh engine command: {command!r}")

        if self.finished or self._step_limit_reached():
            self._rebuild_mesh(final=True)
            return MeshEngineStatus.FINISHED_STEP_LIMIT_REACHED, None

        self.step += 1
        self._step_once()

        if self.finished:
            self._rebuild_mesh(final=True)
            return MeshEngineStatus.FINISHED_FORCE_EQUILIBRIUM_REACHED, None

        if self._step_limit_reached():
            self._rebuild_mesh(final=True)
            return MeshEngineStatus.FINISHED_STEP_LIMIT_REACHED, None

        return MeshEngineStatus.CAN_CONTINUE, self.run


def mesh_bodies_raw(
    gendriver: Any,
    mesher: dict[str, Any],
    bb_min: list[float],
    bb_max: list[float],
    mesh_ext: int,
    objects: list[Body],
    a0: float,
    density: str | DensityFunction | None,
    fixed: list[list[float]],
    mobile: list[list[float]],
    simply: list[list[float]],
    periodic: list[float] | list[bool],
    hints: list[list[Any]],
) -> RawMesh:
    """Public backend entry point for meshing bodies with the Python engine."""

    geometry = fem_geometry_from_bodies(
        (np.asarray(bb_min, dtype=float), np.asarray(bb_max, dtype=float)),
        list(objects),
        list(hints),
        density=density,
        mesh_exterior=bool(mesh_ext),
    )

    engine = RelaxationEngine(
        geometry,
        mesher,
        float(a0),
        _as_float_array(fixed, dim=geometry.dim),
        _as_float_array(mobile, dim=geometry.dim),
        _as_float_array(simply, dim=geometry.dim),
        list(periodic),
        rng=np.random.default_rng(DEFAULT_RNG_SEED),
    )

    if callable(gendriver):
        gendriver(engine.run)
    else:
        engine.run(MeshEngineCommand.DO_STEP)

    engine._rebuild_mesh(final=True)
    return engine.cached_raw_mesh
