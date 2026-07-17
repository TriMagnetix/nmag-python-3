"""
This file contains the implementation of the hysteresis method
of the Simulation class.
"""

from __future__ import annotations

from typing import Any, Protocol

from si.physical import SI
from when import When

from .clock import SimulationClock
from .hysteresis_runner import (
    _update_progress_file as _update_progress_file,  # noqa: F401
)
from .hysteresis_runner import (
    simulation_hysteresis as simulation_hysteresis,
)
from .hysteresis_runner import (
    simulation_relax as simulation_relax,
)
from .hysteresis_schedule import (
    Action,
    ScheduleInput,
)
from .hysteresis_schedule import (
    _append_x_list as _append_x_list,  # noqa: F401
)
from .hysteresis_schedule import (
    _join_save_and_do_lists as _join_save_and_do_lists,  # noqa: F401
)
from .hysteresis_schedule import (
    _string_normalise as _string_normalise,  # noqa: F401
)


class _ConvergenceSource(Protocol):
    def get_log(self) -> str: ...


class HysteresisSimulation(Protocol):
    name: str
    clock: SimulationClock
    convergence: _ConvergenceSource
    action_abbreviations: dict[str, Action]
    _restarting: bool
    max_time_reached: SI

    def simulation_hysteresis(
        self,
        H_ext_list: list[Any],
        save: ScheduleInput | None = None,
        do: ScheduleInput | None = None,
        convergence_check: When | None = None,
        progress_message_minimum_delay: float = 60.0,
    ) -> None: ...

    def load_restart_file(self, filename: str | None = None) -> None: ...

    def do_next_stage(self, stage: int | None = None) -> None: ...

    def set_H_ext(self, values: Any, unit: SI | None = None) -> None: ...

    def reinitialise(self, initial_time: float | None = None) -> None: ...

    def is_converged(self) -> bool: ...

    def advance_time(
        self,
        target_time: SI,
        max_it: int = -1,
        exact_tstop: bool | None = None,
    ) -> SI: ...
