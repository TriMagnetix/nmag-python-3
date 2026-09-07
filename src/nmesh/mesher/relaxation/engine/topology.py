"""Topology refresh and point insertion/deletion for relaxation meshing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ...meshing_parameters import PointFate, default_handle_point_density_fun
from .._constants import BOUNDARY_FUZZ, STATE_BOUNDARY, STATE_MOBILE
from .._types import FloatArray
from ..forces import ForceSummary
from ..seeding import _classify_dynamic_states, _dedupe_points
from ..topology.recovery import mirror_surface_recovery_points


class RelaxationEngineTopologyMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

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
            result: Any = weight_fun(self.step, settling_steps, init_val, final_val)
            return float(result)
        fraction = min(1.0, float(self.step) / max(float(settling_steps), 1.0))
        return init_val + (final_val - init_val) * fraction

    def _density_thresholds(self) -> tuple[float, float]:
        """Return add/delete thresholds with the legacy initial relaxation ramp."""

        freedom = float(self.params.get("controller_movement_max_freedom", 3.0))
        thresh_add = self._initial_relaxation_weight(-0.1 * freedom, 0.0) + float(
            self.params.get("controller_thresh_add", 1.0)
        )
        thresh_del = self._initial_relaxation_weight(0.1 * freedom, 0.0) + float(
            self.params.get("controller_thresh_del", 2.0)
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

        return self.step < self.max_steps and self._is_positive_square_number(self.step - 10)

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
            elif (
                fate == PointFate.DELETE
                and len(self.points) - len(removals) > self.geometry.dim + 1
            ):
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
        if self.geometry.classify_points(candidate[np.newaxis, :])[0] >= 0:
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
