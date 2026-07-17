from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from si.physical import SI

from ...dynamics import IntegratorStats
from ..support import _si_unit

FloatArray = NDArray[np.float64]


class SimulationTimeAdvanceMixin:
    if TYPE_CHECKING:
        _integrator: Any
        _integrator_is_stale: bool
        _integrator_rhs_evaluations: int
        _last_integrator_stats: IntegratorStats

        def __getattr__(self, name: str) -> Any: ...

    def advance_time(
        self,
        target_time: SI,
        max_it: int = -1,
        exact_tstop: bool | None = None,
    ) -> SI:
        target_seconds, current_seconds = self._advance_request(target_time, max_it)
        if target_seconds == current_seconds or max_it == 0:
            return SI(current_seconds, "s")
        integrator = self._active_integrator(current_seconds)
        state_before: FloatArray = np.array(integrator.y, dtype=np.float64, copy=True)
        time_before = float(integrator.t)
        rhs_before = self._integrator_rhs_evaluations
        accepted_steps, last_step_seconds = 0, 0.0
        failed = False
        status = "running"
        started = time.perf_counter()
        try:
            integrator, accepted_steps, last_step_seconds = self._advance_integrator(
                integrator,
                target_seconds,
                max_it,
                self._integrator_config.exact_tstop if exact_tstop is None else exact_tstop,
                time_before,
            )
            reached_seconds = self._accept_integrated_state(
                integrator,
                state_before,
                time_before,
                accepted_steps,
                last_step_seconds,
                started,
            )
            status = str(integrator.status)
            return SI(reached_seconds, "s")
        except Exception:
            failed = True
            status = "failed"
            raise
        finally:
            self._last_integrator_stats = IntegratorStats(
                accepted_steps=accepted_steps,
                rhs_evaluations=self._integrator_rhs_evaluations - rhs_before,
                rejected_steps=None,
                failed=failed,
                status=status,
                last_step_seconds=last_step_seconds,
                simulated_seconds=max(
                    0.0, float(getattr(integrator, "t", time_before)) - time_before
                ),
                wall_seconds=time.perf_counter() - started,
            )

    def _advance_request(self, target_time: SI, max_it: int) -> tuple[float, float]:
        target_seconds = target_time.in_units_of(_si_unit("s"))
        current_seconds = self.clock.time_reached_si.in_units_of(_si_unit("s"))
        if target_seconds < current_seconds:
            raise ValueError("target_time cannot be earlier than the current stage time.")
        if max_it < -1:
            raise ValueError("max_it must be -1 or a non-negative integer.")
        return target_seconds, current_seconds

    def _active_integrator(self, current_seconds: float) -> Any:
        if self._integrator is None or self._integrator_is_stale:
            self.reinitialise(initial_time=current_seconds)
        if self._integrator is None:
            raise RuntimeError("The time integrator was not initialised.")
        return self._integrator

    def _advance_integrator(
        self,
        integrator: Any,
        target_seconds: float,
        max_it: int,
        exact: bool,
        time_before: float,
    ) -> tuple[Any, int, float]:
        accepted_steps, last_step_seconds = 0, 0.0
        while self._can_continue_advancing(integrator, target_seconds, max_it, accepted_steps):
            previous_time = float(integrator.t)
            message = integrator.step()
            if integrator.status == "failed":
                raise RuntimeError(message or "DOP853 failed to advance the LLG state.")
            accepted_steps += 1
            last_step_seconds = float(integrator.t) - previous_time
            if exact and float(integrator.t) > target_seconds:
                integrator = self._restart_at_exact_target(
                    integrator,
                    target_seconds,
                    time_before,
                )
                last_step_seconds = target_seconds - previous_time
                break
        return integrator, accepted_steps, last_step_seconds

    @staticmethod
    def _can_continue_advancing(
        integrator: Any,
        target_seconds: float,
        max_it: int,
        accepted_steps: int,
    ) -> bool:
        return float(integrator.t) < target_seconds and (max_it < 0 or accepted_steps < max_it)

    def _restart_at_exact_target(
        self,
        integrator: Any,
        target_seconds: float,
        time_before: float,
    ) -> Any:
        accepted_state = np.asarray(integrator.dense_output()(target_seconds), dtype=np.float64)
        self._fields["m"] = np.array(np.reshape(accepted_state, (-1, 3)), copy=True)
        self._invalidate_demag()
        self._integrator_is_stale = True
        self.reinitialise(initial_time=target_seconds)
        if self._integrator is None:
            raise RuntimeError("The time integrator could not be reinitialised.")
        # Reinitialisation positions the solver at the target. Keep the clock at
        # the start of this advance until record_advance accounts for the delta.
        self.clock.stage_time = SI(time_before, "s")
        return self._integrator

    def _accept_integrated_state(
        self,
        integrator: Any,
        state_before: FloatArray,
        time_before: float,
        accepted_steps: int,
        last_step_seconds: float,
        started: float,
    ) -> float:
        accepted_state = np.asarray(integrator.y, dtype=np.float64)
        if not np.all(np.isfinite(accepted_state)):
            raise FloatingPointError("The integrator accepted a non-finite magnetisation state.")
        reached_seconds = float(integrator.t)
        self._fields["m"] = np.array(np.reshape(accepted_state, (-1, 3)), copy=True)
        self._invalidate_demag()
        elapsed_seconds = reached_seconds - time_before
        self.clock.record_advance(
            stage_time_seconds=reached_seconds,
            accepted_steps=accepted_steps,
            last_step_seconds=last_step_seconds,
            wall_seconds=time.perf_counter() - started,
        )
        self._record_max_dm_dt(accepted_state, state_before, elapsed_seconds)
        return reached_seconds

    def _record_max_dm_dt(
        self,
        accepted_state: FloatArray,
        state_before: FloatArray,
        elapsed_seconds: float,
    ) -> None:
        if elapsed_seconds <= 0.0:
            return
        displacement: FloatArray = np.reshape(accepted_state - state_before, (-1, 3))
        self.max_dm_dt = float(np.max(np.linalg.norm(displacement, axis=1)) / elapsed_seconds)
