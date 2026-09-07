from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from si.physical import SI

from ..checkpoint import read_checkpoint, runtime_state, save_checkpoint, validate_mesh
from ..dynamics import IntegratorStats


class SimulationRestartMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def get_restart_file_name(self) -> Path:
        """Return the default native checkpoint path beside simulation output."""
        return self.writer.h5_filename.with_name(f"{self.name}_restart.h5")

    def save_restart_file(self, filename: str | Path | None = None) -> Path:
        """Atomically save a complete native checkpoint.

        Args:
            filename: Destination path, or ``None`` for the simulation's default
                restart filename.

        Returns:
            The checkpoint path.

        Raises:
            RuntimeError: If mesh or magnetization state is incomplete.
        """
        destination = self.get_restart_file_name() if filename is None else Path(filename)
        return save_checkpoint(self, destination)

    def load_m_from_h5file(self, filename: str | Path) -> None:
        """Load only checkpoint magnetization into the configured simulation.

        Args:
            filename: Native checkpoint created for the same mesh.

        Raises:
            ValueError: If the checkpoint or mesh is incompatible.
        """
        point_count = len(self._mesh_points())
        contents = read_checkpoint(Path(filename), point_count)
        validate_mesh(self, contents)
        self._fields["m"] = np.array(contents.magnetisation, dtype=float, copy=True)
        self._invalidate_after_checkpoint_restore()

    def load_restart_file(self, filename: str | Path | None = None) -> None:
        """Restore complete native state into a compatible loaded simulation.

        The target simulation must already have the same mesh and compatible
        materials. Magnetization, pinning, current density, applied field,
        clock, integrator controls, and convergence state are restored.

        Args:
            filename: Source path, or ``None`` for the default restart filename.

        Raises:
            ValueError: If checkpoint schema, mesh, materials, or dimensions are
                incompatible.
        """
        source = self.get_restart_file_name() if filename is None else Path(filename)
        point_count = len(self._mesh_points())
        contents = read_checkpoint(source, point_count)
        runtime = runtime_state(self, contents)

        self._fields["m"] = np.array(contents.magnetisation, dtype=float, copy=True)
        self._fields["pin"] = np.array(contents.pinning, dtype=float, copy=True)
        self._fields["H_ext"] = np.array(contents.external_field, dtype=float, copy=True)
        if contents.current_density is None:
            self._fields.pop("current_density", None)
        else:
            self._fields["current_density"] = np.array(
                contents.current_density,
                dtype=float,
                copy=True,
            )
        self.clock = runtime.clock
        self._integrator_config = runtime.integrator_config
        self.stopping_dm_dt = runtime.stopping_dm_dt
        self.max_time_reached = SI(runtime.maximum_time_seconds, "s")
        self.max_dm_dt = runtime.maximum_dm_dt
        self.convergence = runtime.convergence
        self._restarting = False
        self._invalidate_after_checkpoint_restore()

    def _invalidate_after_checkpoint_restore(self) -> None:
        self._invalidate_demag()
        self._subfield_array_cache = None
        self._subfield_average_cache = None
        self._integrator = None
        self._integrator_is_stale = True
        self._integrator_rhs_evaluations = 0
        self._integrator_effective_max_step_seconds = self._integrator_config.maximum_step_seconds
        self._last_integrator_stats = IntegratorStats(status="restarted")
        self._stage_wall_started = time.perf_counter()
