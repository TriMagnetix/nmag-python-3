from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DEFAULT_STOPPING_DM_DT = float(np.deg2rad(1.0) * 1.0e9)
DOP853_EXCHANGE_RESOLUTION_RADIUS = 3.0


@dataclass(frozen=True, slots=True)
class NodalMaterialCoefficients:
    exchange_prefactor: FloatArray
    precession: FloatArray
    damping: FloatArray
    normalisation: FloatArray
    stt_adiabatic: FloatArray
    stt_nonadiabatic: FloatArray


@dataclass(slots=True)
class IntegratorConfig:
    relative_tolerance: float = 1.0e-6
    absolute_tolerance: float = 1.0e-6
    maximum_step_seconds: float = 1.0e-12
    exact_tstop: bool = True

    def validate(self) -> None:
        for name, value in (
            ("relative_tolerance", self.relative_tolerance),
            ("absolute_tolerance", self.absolute_tolerance),
            ("maximum_step_seconds", self.maximum_step_seconds),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number, got {value!r}.")


@dataclass(frozen=True, slots=True)
class IntegratorStats:
    accepted_steps: int = 0
    rhs_evaluations: int = 0
    rejected_steps: int | None = None
    failed: bool = False
    status: str = "not_started"
    last_step_seconds: float = 0.0
    simulated_seconds: float = 0.0
    wall_seconds: float = 0.0
    jacobian_vector_evaluations: int | None = None
    nonlinear_iterations: int | None = None
    nonlinear_failures: int | None = None
    linear_solver_setups: int | None = None


@dataclass(slots=True)
class ConvergenceTracker:
    required_checks: int = 2
    statistics_window: int = 5
    _remaining_checks: int = field(init=False)
    _history: deque[tuple[int, float, float, float | None]] = field(
        default_factory=lambda: deque(maxlen=50),
        init=False,
    )
    _previous_max_dm_dt: float | None = field(default=None, init=False)
    _window_changes: list[float] = field(default_factory=lambda: list[float](), init=False)

    def __post_init__(self) -> None:
        if self.required_checks < 1:
            raise ValueError("required_checks must be at least one.")
        if self.statistics_window < 1:
            raise ValueError("statistics_window must be at least one.")
        self._remaining_checks = self.required_checks

    def reset(self) -> None:
        self._remaining_checks = self.required_checks
        self._previous_max_dm_dt = None
        self._window_changes.clear()

    def check(self, step: int, max_dm_dt: float, stopping_dm_dt: float) -> bool:
        if max_dm_dt < stopping_dm_dt:
            self._remaining_checks -= 1
            if self._remaining_checks == 0:
                self.reset()
                return True
            return False

        self._remaining_checks = self.required_checks
        quality = self._quality(max_dm_dt)
        self._history.append((step, max_dm_dt, stopping_dm_dt, quality))
        return False

    def _quality(self, max_dm_dt: float) -> float | None:
        if self._previous_max_dm_dt is None:
            self._previous_max_dm_dt = max_dm_dt
            return None

        self._window_changes.append(max_dm_dt - self._previous_max_dm_dt)
        self._previous_max_dm_dt = max_dm_dt
        if len(self._window_changes) < self.statistics_window:
            return None

        signed = abs(sum(self._window_changes))
        absolute = sum(abs(change) for change in self._window_changes)
        self._window_changes.clear()
        return signed / absolute if absolute > 0.0 else None

    def get_log(self) -> str:
        lines = ["# Convergence log", "# step, max dm/dt, stopping dm/dt, quality"]
        lines.extend(
            f"{step} {max_dm_dt} {stopping_dm_dt} {quality}"
            for step, max_dm_dt, stopping_dm_dt, quality in self._history
        )
        return "\n".join(lines) + "\n"

    def checkpoint_state(self) -> dict[str, object]:
        """Return the convergence cadence state without exposing mutable internals."""
        return {
            "required_checks": self.required_checks,
            "statistics_window": self.statistics_window,
            "remaining_checks": self._remaining_checks,
            "history": [list(entry) for entry in self._history],
            "previous_max_dm_dt": self._previous_max_dm_dt,
            "window_changes": list(self._window_changes),
        }

    @classmethod
    def from_checkpoint_state(cls, state: dict[str, object]) -> ConvergenceTracker:
        cls._validate_checkpoint_keys(state)
        required_checks, statistics_window, remaining_checks = cls._checkpoint_counters(state)
        tracker = cls(
            required_checks=required_checks,
            statistics_window=statistics_window,
        )
        if not 1 <= remaining_checks <= tracker.required_checks:
            raise ValueError("Checkpoint remaining convergence checks are invalid.")
        tracker._remaining_checks = remaining_checks
        tracker._history = cls._checkpoint_history(state["history"])
        tracker._previous_max_dm_dt = cls._optional_finite_checkpoint_value(
            state["previous_max_dm_dt"],
            "Checkpoint previous convergence rate must be finite or null.",
        )
        tracker._window_changes = cls._finite_checkpoint_values(
            state["window_changes"],
            "Checkpoint convergence window changes must be finite numbers.",
        )
        return tracker

    @staticmethod
    def _validate_checkpoint_keys(state: dict[str, object]) -> None:
        required = {
            "required_checks",
            "statistics_window",
            "remaining_checks",
            "history",
            "previous_max_dm_dt",
            "window_changes",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"Checkpoint convergence state is missing {', '.join(missing)}.")

    @staticmethod
    def _checkpoint_counters(state: dict[str, object]) -> tuple[int, int, int]:
        values = tuple(
            state[name] for name in ("required_checks", "statistics_window", "remaining_checks")
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("Checkpoint convergence counters must be integers.")
        return cast(tuple[int, int, int], values)

    @classmethod
    def _checkpoint_history(cls, value: object) -> deque[tuple[int, float, float, float | None]]:
        if not isinstance(value, list):
            raise ValueError("Checkpoint convergence history must be a list.")
        history: deque[tuple[int, float, float, float | None]] = deque(maxlen=50)
        for raw_entry in cast(list[object], value):
            history.append(cls._checkpoint_history_entry(raw_entry))
        return history

    @classmethod
    def _checkpoint_history_entry(cls, value: object) -> tuple[int, float, float, float | None]:
        if not isinstance(value, list):
            raise ValueError("Checkpoint convergence history entries must have four values.")
        entry = cast(list[object], value)
        if len(entry) != 4:
            raise ValueError("Checkpoint convergence history entries must have four values.")
        step, maximum, stopping, quality = entry
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("Checkpoint convergence history steps must be integers.")
        rates = cls._finite_checkpoint_values(
            [maximum, stopping],
            "Checkpoint convergence history rates must be finite.",
        )
        return (
            step,
            rates[0],
            rates[1],
            cls._optional_finite_checkpoint_value(
                quality,
                "Checkpoint convergence history quality must be finite or null.",
            ),
        )

    @staticmethod
    def _optional_finite_checkpoint_value(value: object, message: str) -> float | None:
        if value is None:
            return None
        values = ConvergenceTracker._finite_checkpoint_values([value], message)
        return values[0]

    @staticmethod
    def _finite_checkpoint_values(value: object, message: str) -> list[float]:
        if not isinstance(value, list):
            raise ValueError(message)
        parsed: list[float] = []
        for raw_value in cast(list[object], value):
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not isfinite(float(raw_value))
            ):
                raise ValueError(message)
            parsed.append(float(raw_value))
        return parsed
