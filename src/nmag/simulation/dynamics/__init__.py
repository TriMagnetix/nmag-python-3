from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from simulation import hysteresis as hysteresis_module

from ...backends import _selected_integrator_backend
from ..implicit_dynamics import relax_with_diffsol
from ..support import _simulation_compatibility_binding
from .integrator import (
    SimulationIntegratorMixin,
)
from .integrator import (
    _dop853_class as _dop853_class,  # noqa: F401
)


class SimulationDynamicsMixin(SimulationIntegratorMixin):
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def do_next_stage(self, stage: int | None = None) -> None:
        self.max_dm_dt = None
        self.convergence.reset()
        self.clock.inc_stage(stage=stage)
        self._stage_wall_started = time.perf_counter()
        self._invalidate_integrator()

    def is_converged(self) -> bool:
        if self.max_dm_dt is None:
            self.clock.convergence = False
            return False
        converged = self.convergence.check(self.step, self.max_dm_dt, self.stopping_dm_dt)
        self.clock.convergence = converged
        return converged

    @staticmethod
    def hysteresis_next_stage(sim: Any) -> None:
        sim.clock.stage_end = True

    @staticmethod
    def hysteresis_exit(sim: Any) -> None:
        sim.clock.exit_hysteresis = True
        sim.clock.stage_end = True

    simulation_relax = hysteresis_module.simulation_relax

    def relax(
        self,
        H_applied: Any = None,
        save: list[tuple[object, ...]] | None = None,
        do: list[tuple[object, ...]] | None = None,
        convergence_check: Any = None,
    ) -> None:
        """Relax magnetization until the convergence schedule completes.

        Args:
            H_applied: Optional applied-field value for compatibility with the
                staged relaxation runner.
            save: Scheduled save tuples such as
                ``[("averages", every("step", 10))]``. Omit for averages and
                fields at stage end.
            do: Scheduled action tuples. Omit for the default stage lifecycle.
            convergence_check: Custom :class:`when.When` condition. Omit for
                the accepted-step convergence cadence.

        Raises:
            NotImplementedError: If a custom schedule is requested with the
                experimental Diffsol backend.
        """
        backend = _simulation_compatibility_binding(
            "_selected_integrator_backend", _selected_integrator_backend
        )(getattr(self, "config", None))
        if backend == "scipy":
            hysteresis_module.simulation_relax(
                cast(Any, self),
                H_applied=H_applied,
                save=save,
                do=do,
                convergence_check=convergence_check,
            )
            return
        if save is not None or do not in (None, []) or convergence_check is not None:
            raise NotImplementedError(
                "The experimental Diffsol backend currently supports only the default "
                "relaxation save and convergence schedule."
            )
        relax_with_diffsol(self, H_applied=H_applied)

    simulation_hysteresis = hysteresis_module.simulation_hysteresis
