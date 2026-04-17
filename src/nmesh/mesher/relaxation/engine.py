"""Iterative meshing engine for the pure-Python relaxation pipeline."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...backend import RawMesh
from ...geometry.primitives import Body
from ..driver import MeshEngineCommand, MeshEngineStatus
from ..meshing_parameters import PointFate, default_handle_point_density_fun
from ._constants import BOUNDARY_FUZZ, DEFAULT_RNG_SEED, STATE_MOBILE
from ._types import DensityFunction, EngineResult, FloatArray
from .geometry import FemGeometry, fem_geometry_from_bodies
from .seeding import _as_float_array, _dedupe_points, _prepare_initial_points
from .topology import _callback_mesh_info, assemble_raw_mesh


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
        )
        self.step = 0
        self.finished = False
        self.last_max_displacement = math.inf
        self.cached_raw_mesh = assemble_raw_mesh(self.points, self.geometry, self.periodic)

    @property
    def max_steps(self) -> int:
        """Return the capped step budget for this Python port."""

        return min(int(self.params.get("controller_step_limit_max", 1000)), 60)

    @property
    def tolerated_rel_move(self) -> float:
        """Return the relative movement threshold used for convergence."""

        return float(self.params.get("controller_tolerated_rel_movement", 0.002))

    @property
    def time_step_scale(self) -> float:
        """Return the scale factor applied to each relaxation displacement."""

        return float(self.params.get("controller_time_step_scale", 0.1))

    def _rebuild_mesh(self) -> None:
        """Refresh the cached ``RawMesh`` snapshot from the current point cloud."""

        self.cached_raw_mesh = assemble_raw_mesh(self.points, self.geometry, self.periodic)

    def _neighbor_map(self) -> list[list[int]]:
        """Derive point adjacency from the cached simplices."""

        neighbors: list[set[int]] = [set() for _ in range(len(self.points))]
        for simplex in self.cached_raw_mesh.simplices:
            for index in simplex:
                neighbors[index].update(int(item) for item in simplex if item != index)
        return [sorted(group) for group in neighbors]

    def _attempt_add_delete_points(self, displacements: FloatArray) -> None:
        """Apply the mesher density heuristic to add or remove mobile points."""

        if len(self.points) == 0:
            return

        handler = self.params.get("handle_point_density_fun", default_handle_point_density_fun)
        thresh_add = float(self.params.get("controller_thresh_add", 1.0))
        thresh_del = float(self.params.get("controller_thresh_del", 2.0))
        neighbors = self._neighbor_map()

        additions: list[FloatArray] = []
        removals: list[int] = []

        for index, state in enumerate(self.states):
            if state != STATE_MOBILE or index in removals:
                continue

            neigh = neighbors[index]
            if not neigh:
                continue

            point = self.points[index]
            neigh_coords = self.points[np.asarray(neigh, dtype=int)]
            distances = np.linalg.norm(neigh_coords - point, axis=1)
            mean_distance = float(np.mean(distances))
            density_here = self.geometry.density_at(point)
            ideal_distance = self.a0 / (density_here ** (1.0 / max(self.geometry.dim, 1)))
            avg_density = ideal_distance / max(mean_distance, BOUNDARY_FUZZ)
            avg_force = float(np.linalg.norm(displacements[index]))

            fate = handler(self.rng, (avg_density, avg_force), thresh_add, thresh_del)
            if fate == PointFate.ADD_ANOTHER:
                farthest = neigh_coords[int(np.argmax(distances))]
                candidate = 0.5 * (point + farthest)
                if self.geometry.classify_points(candidate.reshape(1, -1))[0] >= 0:
                    additions.append(candidate)
            elif fate == PointFate.DELETE and len(self.points) - len(removals) > self.geometry.dim + 1:
                removals.append(index)

        if removals:
            keep = np.ones(len(self.points), dtype=bool)
            keep[np.asarray(removals, dtype=int)] = False
            self.points = self.points[keep]
            self.states = self.states[keep]

        if additions:
            additions_arr = _dedupe_points(np.asarray(additions, dtype=float))
            if len(additions_arr) > 0:
                self.points = np.vstack((self.points, additions_arr))
                self.states = np.concatenate(
                    (
                        self.states,
                        np.full(len(additions_arr), STATE_MOBILE, dtype=int),
                    )
                )

    def _step_once(self) -> None:
        """Execute one relaxation step over all currently mobile points."""

        if len(self.points) < self.geometry.dim + 1:
            self.finished = True
            return

        self._rebuild_mesh()
        neighbors = self._neighbor_map()
        new_points = np.array(self.points, copy=True)
        displacements = np.zeros_like(new_points)
        max_displacement = 0.0

        for index, state in enumerate(self.states):
            if state != STATE_MOBILE:
                continue

            neigh = neighbors[index]
            if not neigh:
                continue

            point = self.points[index]
            neigh_coords = self.points[np.asarray(neigh, dtype=int)]
            deltas = neigh_coords - point
            distances = np.linalg.norm(deltas, axis=1)
            nonzero = distances > BOUNDARY_FUZZ
            if not np.any(nonzero):
                continue

            deltas = deltas[nonzero]
            distances = distances[nonzero]
            density_here = self.geometry.density_at(point)
            ideal_distance = self.a0 / (density_here ** (1.0 / max(self.geometry.dim, 1)))
            scale = (distances - ideal_distance) / distances
            displacement = np.mean(deltas * scale[:, None], axis=0)
            displacement *= self.time_step_scale
            max_norm = self.a0 * float(self.params.get("controller_movement_max_freedom", 3.0)) * 0.05
            disp_norm = float(np.linalg.norm(displacement))
            if disp_norm > max_norm > 0.0:
                displacement *= max_norm / disp_norm

            candidate = point + displacement
            if self.geometry.classify_points(candidate.reshape(1, -1))[0] >= 0:
                new_points[index] = candidate
                displacements[index] = displacement
                max_displacement = max(max_displacement, float(np.linalg.norm(displacement)))

        self.points = new_points
        self.last_max_displacement = max_displacement / max(self.a0, BOUNDARY_FUZZ)

        if self.step > 0 and self.step % 5 == 0:
            self._attempt_add_delete_points(displacements)

        self._rebuild_mesh()
        if self.last_max_displacement <= self.tolerated_rel_move:
            self.finished = True

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

        if self.finished or self.step >= self.max_steps:
            self._rebuild_mesh()
            return MeshEngineStatus.FINISHED_STEP_LIMIT_REACHED, None

        self.step += 1
        self._step_once()

        if self.finished:
            self._rebuild_mesh()
            return MeshEngineStatus.FINISHED_FORCE_EQUILIBRIUM_REACHED, None

        if self.step >= self.max_steps:
            self._rebuild_mesh()
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

    engine._rebuild_mesh()
    return engine.cached_raw_mesh
