from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from si.physical import SI

from ...backends import _selected_llg_backend
from ...dynamics import (
    DEFAULT_STOPPING_DM_DT,
    ConvergenceTracker,
    IntegratorConfig,
    IntegratorStats,
)
from ..support import _si_unit, _simulation_compatibility_binding
from .advance import SimulationTimeAdvanceMixin

FloatArray = NDArray[np.float64]
Action = Callable[[Any], Any]


class _DenseOutput(Protocol):
    def __call__(self, time_seconds: float) -> FloatArray: ...


class _OdeIntegrator(Protocol):
    t: float
    y: FloatArray
    status: str

    def step(self) -> str | None: ...

    def dense_output(self) -> _DenseOutput: ...


def _dop853_class() -> Callable[..., _OdeIntegrator]:
    from scipy.integrate import DOP853

    return cast(Callable[..., _OdeIntegrator], DOP853)


def _dop853_constructor() -> Callable[..., _OdeIntegrator]:
    """Honor the historical module-level test and extension override."""
    dynamics_module = sys.modules.get("nmag.simulation.dynamics")
    return getattr(dynamics_module, "_dop853_class", _dop853_class)()


class SimulationIntegratorSetupMixin:
    if TYPE_CHECKING:
        _integrator: _OdeIntegrator | None
        _integrator_is_stale: bool
        _integrator_rhs_evaluations: int
        _integrator_config: IntegratorConfig
        _integrator_effective_max_step_seconds: float
        _last_integrator_stats: IntegratorStats
        _stage_wall_started: float

        def __getattr__(self, name: str) -> Any: ...

    def _initialise_dynamics(self) -> None:
        self._integrator = None
        self._integrator_is_stale = True
        self._integrator_rhs_evaluations = 0
        self._integrator_config = IntegratorConfig()
        self._integrator_effective_max_step_seconds = self._integrator_config.maximum_step_seconds
        self._last_integrator_stats = IntegratorStats()
        self._stage_wall_started = time.perf_counter()
        self.stopping_dm_dt = DEFAULT_STOPPING_DM_DT
        self.max_dm_dt: float | None = None
        self.convergence = ConvergenceTracker()
        self.max_time_reached = SI(1.0, "s")
        self._restarting = False
        self.action_abbreviations: dict[str, Action] = {
            # Legacy relaxation emits separate average and field rows at stage end.
            "save_averages": lambda sim: sim.save_data(),
            "save_fields": lambda sim: sim.save_data(fields="all"),
            "save_restart": lambda sim: sim.save_restart_file(),
            "do_next_stage": self.hysteresis_next_stage,
            "do_exit": self.hysteresis_exit,
        }

    @property
    def last_integrator_stats(self) -> IntegratorStats:
        return self._last_integrator_stats

    @property
    def integrator_config(self) -> IntegratorConfig:
        config = self._integrator_config
        return IntegratorConfig(
            relative_tolerance=config.relative_tolerance,
            absolute_tolerance=config.absolute_tolerance,
            maximum_step_seconds=config.maximum_step_seconds,
            exact_tstop=config.exact_tstop,
        )

    @property
    def effective_integrator_max_step(self) -> SI:
        return SI(self._integrator_effective_max_step_seconds, "s")

    def _invalidate_integrator(self) -> None:
        self._integrator_is_stale = True

    def set_params(
        self,
        stopping_dm_dt: SI | float | None = None,
        ts_rel_tol: float | None = None,
        ts_abs_tol: float | None = None,
        exact_tstop: bool | None = None,
        ts_max_step: SI | float | None = None,
    ) -> None:
        """Update convergence and DOP853 integration controls.

        Args:
            stopping_dm_dt: Positive convergence rate in 1/s.
            ts_rel_tol: Positive relative error tolerance.
            ts_abs_tol: Positive absolute error tolerance.
            exact_tstop: Whether :meth:`advance_time` normally reconstructs
                state exactly at its requested target.
            ts_max_step: Positive configured maximum step in seconds. Exchange
                stability can impose a smaller effective maximum.

        Raises:
            ValueError: If a supplied rate, tolerance, or step is not finite
                and positive.
        """
        if stopping_dm_dt is not None:
            value = (
                stopping_dm_dt.in_units_of(_si_unit("1/s"))
                if isinstance(stopping_dm_dt, SI)
                else float(stopping_dm_dt)
            )
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("stopping_dm_dt must be a positive finite rate.")
            self.stopping_dm_dt = value
        if ts_rel_tol is not None:
            self._integrator_config.relative_tolerance = float(ts_rel_tol)
        if ts_abs_tol is not None:
            self._integrator_config.absolute_tolerance = float(ts_abs_tol)
        if exact_tstop is not None:
            self._integrator_config.exact_tstop = bool(exact_tstop)
        if ts_max_step is not None:
            self._integrator_config.maximum_step_seconds = (
                ts_max_step.in_units_of(_si_unit("s"))
                if isinstance(ts_max_step, SI)
                else float(ts_max_step)
            )
        self._integrator_config.validate()
        self._invalidate_integrator()

    def _llg_time_derivative(self, _time_seconds: float, state: FloatArray) -> FloatArray:
        self._integrator_rhs_evaluations += 1
        m: FloatArray = np.reshape(np.asarray(state, dtype=np.float64), (-1, 3))
        if not np.all(np.isfinite(m)):
            raise FloatingPointError("The integrator supplied a non-finite magnetisation state.")

        self._fields["m"] = m
        self._invalidate_demag()
        h_total = self._subfield_array("H_total")
        pin = self._subfield_array("pin")
        unit_scale = np.ones(len(m), dtype=float)
        coefficients = self._nodal_material_coefficients()
        dm_dcurrent = self._get_dm_dcurrent() if "current_density" in self._fields else None
        backend = _simulation_compatibility_binding("_selected_llg_backend", _selected_llg_backend)(
            len(m), getattr(self, "config", None)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            if backend == "rust":
                derivative = self._llg_rhs_rust(
                    m,
                    h_total,
                    pin,
                    unit_scale,
                    coefficients.precession,
                    coefficients.damping,
                    coefficients.normalisation,
                    dm_dcurrent,
                    coefficients.stt_adiabatic,
                    coefficients.stt_nonadiabatic,
                )
            else:
                derivative = self._llg_rhs_python(
                    m,
                    h_total,
                    pin,
                    unit_scale,
                    coefficients.precession,
                    coefficients.damping,
                    coefficients.normalisation,
                    dm_dcurrent,
                    coefficients.stt_adiabatic,
                    coefficients.stt_nonadiabatic,
                )
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError("The LLG right-hand side produced non-finite values.")
        return np.ravel(np.asarray(derivative, dtype=np.float64))

    def reinitialise(
        self,
        rel_tolerance: float | None = None,
        abs_tolerance: float | None = None,
        initial_time: SI | float | None = None,
    ) -> None:
        """Rebuild the adaptive integrator from the current physical state.

        Args:
            rel_tolerance: Optional new relative tolerance.
            abs_tolerance: Optional new absolute tolerance.
            initial_time: Non-negative physical start time. Omit it to retain
                the simulation clock.

        Raises:
            RuntimeError: If mesh or magnetization has not been set.
            ValueError: If the time or tolerances are invalid.
        """
        if rel_tolerance is not None or abs_tolerance is not None:
            self.set_params(ts_rel_tol=rel_tolerance, ts_abs_tol=abs_tolerance)
        if self.mesh is None or "m" not in self._fields:
            raise RuntimeError("A mesh and magnetisation are required before time integration.")

        if initial_time is None:
            initial_seconds = self.clock.time_reached_si.in_units_of(_si_unit("s"))
        elif isinstance(initial_time, SI):
            initial_seconds = initial_time.in_units_of(_si_unit("s"))
        else:
            initial_seconds = float(initial_time)
        if not np.isfinite(initial_seconds) or initial_seconds < 0.0:
            raise ValueError("initial_time must be a finite non-negative time.")

        self._integrator_config.validate()
        state: FloatArray = np.array(
            np.ravel(np.asarray(self._fields["m"], dtype=np.float64)),
            dtype=np.float64,
            copy=True,
        )
        upper_bound = self.max_time_reached.in_units_of(_si_unit("s"))
        if upper_bound <= initial_seconds:
            upper_bound = float(np.nextafter(initial_seconds, np.inf))
        self._integrator_effective_max_step_seconds = min(
            self._integrator_config.maximum_step_seconds,
            self._exchange_explicit_step_limit_seconds(),
        )
        self._integrator = _dop853_constructor()(
            self._llg_time_derivative,
            initial_seconds,
            state,
            upper_bound,
            rtol=self._integrator_config.relative_tolerance,
            atol=self._integrator_config.absolute_tolerance,
            max_step=self._integrator_effective_max_step_seconds,
        )
        self.clock.time_reached_su = initial_seconds
        self.clock.time_reached_si = SI(initial_seconds, "s")
        self.clock.stage_time = SI(initial_seconds, "s")
        self._integrator_is_stale = False


class SimulationIntegratorMixin(SimulationIntegratorSetupMixin, SimulationTimeAdvanceMixin):
    """Compose adaptive-integrator setup with its stepping lifecycle."""
