from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any

from tabulate import tabulate

from si.physical import SI

_zero_seconds_cache: SI | None = None


def _zero_seconds() -> SI:
    global _zero_seconds_cache
    if _zero_seconds_cache is None:
        _zero_seconds_cache = SI(0.0, "s")
    return SI(_zero_seconds_cache._quantity)


def fmt_time(t: SI, fmt_ps: str = ".2f", fmt_ns: str = ".2f") -> str:
    """Formats an SI time object into picoseconds or nanoseconds."""
    t_ps = float(t / SI(1e-12, "s"))

    ps_str = f"{t_ps:{fmt_ps}}"
    ns_str = f"{(t_ps / 1000.0):{fmt_ns}}"

    return f"{ps_str} ps" if t_ps < 100.0 else f"{ns_str} ns"


@dataclass
class SimulationClock:
    """
    This object specifies all the parameters which define the current time
    in the simulation, such as the simulation time, step number, ...

    Attributes:
      id: Unique identifier for data saved. Incremented on save.
      stage: Stage number. Increments when the external field changes.
      step: Total number of steps performed (always increases).
      stage_step: Step number from the beginning of the current stage.
      zero_stage_step: The value of 'step' at the beginning of the stage.
      time: Total simulation time (always increases).
      stage_time: The simulation time from the beginning of the stage.
      zero_stage_time: The value of 'time' at the beginning of the stage.
      real_time: The real world time used for advancing time.
      last_step_dt_si: Last time step's length in SI units.
      convergence: Flag indicating if convergence is reached.
      stage_end: Flag indicating the end of a stage.
      exit_hysteresis: Flag to signal exit from hysteresis loop.
    """

    id: int = -1
    stage: int = 1
    step: int = 0
    time: SI = field(default_factory=_zero_seconds)
    stage_step: int = 0
    stage_time: SI = field(default_factory=_zero_seconds)
    real_time: SI = field(default_factory=_zero_seconds)
    stage_end: bool = False
    convergence: bool = False
    exit_hysteresis: bool = False
    zero_stage_time: SI = field(default_factory=_zero_seconds)
    zero_stage_step: int = 0
    time_reached_su: float = 0.0
    time_reached_si: SI = field(default_factory=_zero_seconds)
    last_step_dt_su: float = 0.0
    last_step_dt_si: SI = field(default_factory=_zero_seconds)

    def __getitem__(self, key: str, /) -> Any:
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None, /) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def copy(self) -> dict[str, Any]:
        return asdict(self)

    def checkpoint_state(self) -> dict[str, int | float | bool]:
        """Return a JSON-safe representation of the simulation clock."""
        return {
            "id": self.id,
            "stage": self.stage,
            "step": self.step,
            "stage_step": self.stage_step,
            "zero_stage_step": self.zero_stage_step,
            "time_seconds": self.time.in_units_of(SI(1.0, "s")),
            "stage_time_seconds": self.stage_time.in_units_of(SI(1.0, "s")),
            "real_time_seconds": self.real_time.in_units_of(SI(1.0, "s")),
            "zero_stage_time_seconds": self.zero_stage_time.in_units_of(SI(1.0, "s")),
            "time_reached_seconds": self.time_reached_si.in_units_of(SI(1.0, "s")),
            "last_step_seconds": self.last_step_dt_si.in_units_of(SI(1.0, "s")),
            "stage_end": self.stage_end,
            "convergence": self.convergence,
            "exit_hysteresis": self.exit_hysteresis,
        }

    @classmethod
    def from_checkpoint_state(cls, state: dict[str, object]) -> SimulationClock:
        """Build a validated clock from :meth:`checkpoint_state` data."""
        integer_names = ("id", "stage", "step", "stage_step", "zero_stage_step")
        seconds_names = (
            "time_seconds",
            "stage_time_seconds",
            "real_time_seconds",
            "zero_stage_time_seconds",
            "time_reached_seconds",
            "last_step_seconds",
        )
        boolean_names = ("stage_end", "convergence", "exit_hysteresis")
        required = {*integer_names, *seconds_names, *boolean_names}
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"Checkpoint clock state is missing {', '.join(missing)}.")

        integers: dict[str, int] = {}
        for name in integer_names:
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Checkpoint clock {name} must be an integer.")
            integers[name] = value
        if integers["stage"] < 1 or any(
            integers[name] < 0 for name in ("step", "stage_step", "zero_stage_step")
        ):
            raise ValueError("Checkpoint clock counters must be non-negative and stage at least one.")

        seconds: dict[str, float] = {}
        for name in seconds_names:
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Checkpoint clock {name} must be a finite number.")
            converted = float(value)
            if not isfinite(converted) or converted < 0.0:
                raise ValueError(f"Checkpoint clock {name} must be a non-negative finite number.")
            seconds[name] = converted

        booleans: dict[str, bool] = {}
        for name in boolean_names:
            value = state[name]
            if not isinstance(value, bool):
                raise ValueError(f"Checkpoint clock {name} must be a boolean.")
            booleans[name] = value

        return cls(
            id=integers["id"],
            stage=integers["stage"],
            step=integers["step"],
            stage_step=integers["stage_step"],
            zero_stage_step=integers["zero_stage_step"],
            time=SI(seconds["time_seconds"], "s"),
            stage_time=SI(seconds["stage_time_seconds"], "s"),
            real_time=SI(seconds["real_time_seconds"], "s"),
            zero_stage_time=SI(seconds["zero_stage_time_seconds"], "s"),
            time_reached_su=seconds["time_reached_seconds"],
            time_reached_si=SI(seconds["time_reached_seconds"], "s"),
            last_step_dt_su=seconds["last_step_seconds"],
            last_step_dt_si=SI(seconds["last_step_seconds"], "s"),
            stage_end=booleans["stage_end"],
            convergence=booleans["convergence"],
            exit_hysteresis=booleans["exit_hysteresis"],
        )

    # __init__ and __repr__ are GONE (auto-generated)

    def inc_stage(self, stage: int | None = None) -> None:
        """Advance the clock to the next stage."""
        if stage is None:
            self.stage += 1
        else:
            self.stage = stage
        self.stage_step = 0
        self.stage_time = SI(0.0, "s")
        self.convergence = False
        self.zero_stage_step = self.step
        self.zero_stage_time = self.time

    def record_advance(
        self,
        *,
        stage_time_seconds: float,
        accepted_steps: int,
        last_step_seconds: float,
        wall_seconds: float,
    ) -> None:
        """Atomically record an accepted group of integration steps."""
        previous_stage_seconds = self.stage_time.in_units_of(SI(1.0, "s"))
        delta_seconds = stage_time_seconds - previous_stage_seconds
        if delta_seconds < 0.0:
            raise ValueError("stage time cannot move backwards.")
        if accepted_steps < 0:
            raise ValueError("accepted_steps cannot be negative.")

        self.step += accepted_steps
        self.stage_step += accepted_steps
        self.time += SI(delta_seconds, "s")
        self.stage_time = SI(stage_time_seconds, "s")
        self.time_reached_su = stage_time_seconds
        self.time_reached_si = SI(stage_time_seconds, "s")
        self.last_step_dt_su = last_step_seconds
        self.last_step_dt_si = SI(last_step_seconds, "s")
        self.real_time += SI(max(0.0, wall_seconds), "s")

    # This method was updated to use tabulate, the format of the data printed out might
    # look slightly different, but there is a lot less manual formatting code here now.
    def __str__(self) -> str:
        ft = fmt_time

        rows = [
            [
                f"ID={self.id}",
                f"Step={self.step}",
                f"Time={ft(self.time)}",
                f"Last step size={ft(self.last_step_dt_si)}",
            ],
            [
                "",
                f"Stage={self.stage}",
                f"Stage-step={self.stage_step}",
                f"Stage-time={ft(self.stage_time)}",
            ],
            [
                "",
                f"Convergence={self.convergence}",
                f"Stage-end={self.stage_end}",
                f"Exit hysteresis={self.exit_hysteresis}",
            ],
        ]

        table = tabulate(rows, tablefmt="pipe")

        sep_line = "=" * (len(table.splitlines()[0]))
        return f"{sep_line}\n{table}\n{sep_line}"
