"""Relaxation displacement, convergence, and command execution phases."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ....backend import RawMesh
from ....geometry.primitives import Body
from ...driver import MeshEngineCommand, MeshEngineStatus
from .._constants import BOUNDARY_FUZZ, DEFAULT_RNG_SEED, STATE_BOUNDARY, STATE_MOBILE
from .._types import DensityFunction, EngineResult, FloatArray
from ..forces import ForceSummary, compute_forces
from ..geometry import FemGeometry, fem_geometry_from_bodies
from ..seeding import _as_float_array
from ..topology import _callback_mesh_info

log = logging.getLogger(__name__)


class RelaxationEngineStepMixin:
    if TYPE_CHECKING:
        geometry: FemGeometry
        params: dict[str, Any]
        a0: float
        periodic: list[float] | list[bool]
        points: FloatArray
        states: np.ndarray[Any, Any]
        step: int
        finished: bool
        last_addition_deletion_step: int
        last_max_displacement: float
        last_max_effective_force: float
        last_point_density: FloatArray
        max_rel_movement_since_last_triangulation: float
        current_simplices: np.ndarray[Any, Any] | None
        points_at_last_triangulation: FloatArray
        cached_raw_mesh: RawMesh

        def __getattr__(self, name: str) -> Any: ...

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
                bounded_displacement -= float(np.dot(bounded_displacement, normal)) * (
                    normal / normal_norm
                )

        disp_norm = float(np.linalg.norm(bounded_displacement))
        if disp_norm > max_norm > 0.0:
            bounded_displacement *= max_norm / disp_norm

        candidate = point + bounded_displacement
        if state == STATE_BOUNDARY:
            old_boundary_distance = self.geometry.boundary_distance(point)
            new_boundary_distance = self.geometry.boundary_distance(candidate)
            if new_boundary_distance > old_boundary_distance + self.boundary_drift_tolerance:
                return np.asarray(point, dtype=float)

        if self.geometry.classify_points(candidate[np.newaxis, :])[0] < 0:
            candidate = self.geometry.project_segment_to_domain(point, candidate)

        if self.geometry.classify_points(candidate[np.newaxis, :])[0] >= 0:
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
            candidate = self._compute_constrained_displacement(
                point, int(state), displacement, max_norm
            )
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

        if self.step >= 2 and self.last_max_effective_force <= BOUNDARY_FUZZ:
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

    # Import lazily so the public façade can compose the engine mixins first.
    from . import RelaxationEngine

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
