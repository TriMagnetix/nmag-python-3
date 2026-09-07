"""Execution loops for relaxation and multi-stage hysteresis."""

from __future__ import annotations

import logging
import time
from typing import Any

from si.physical import SI
from throttler import Throttler
from when import When, at, every

from .clock import SimulationClock
from .hysteresis_schedule import (
    CounterTriple,
    ScheduleInput,
    _join_save_and_do_lists,
    _next_deltas,
    _next_time,
)

log = logging.getLogger("nmag")
_progress_throttler = Throttler()


def _update_progress_file(
    sim: Any,
    H_ext: Any,
    progress_file_name: str,
    progress_message_minimum_delay: float,
) -> None:
    """
    Writes the current simulation progress to a file, throttled
    to a minimum delay.
    """
    if _progress_throttler.is_allowed("hysteresis_reporting", progress_message_minimum_delay):
        try:
            with open(progress_file_name, "w", encoding="utf-8") as f:
                f.write(f"{time.asctime()}\n")
                f.write(f"{sim.clock}\n")
                f.write(f"{sim.convergence.get_log()}\n")

            log.info(
                f"it {sim.clock.step}, time {sim.clock.time_reached_si.dens_str()}; "
                f"stage {sim.clock.stage}; H_ext={str(H_ext)}"
            )
        except Exception as e:
            log.warning(f"Could not write progress file '{progress_file_name}'. Error: {e}")


def simulation_relax(
    self: Any,
    H_applied: Any = None,
    save: ScheduleInput | None = None,
    do: ScheduleInput | None = None,
    convergence_check: When | None = None,
) -> None:
    """
    This method carries out the time integration of the LLG until
    the system reaches a (metastable) equilibrium.
    Internally, this uses the hysteresis() loop command.

    (Docstring parameters omitted for brevity, they are unchanged)
    """
    if save is None:
        save = [("averages", "fields", at("stage_end"))]
    if do is None:
        do = []
    if convergence_check is None:
        convergence_check = every("step", 5)

    log.debug("Entering 'relax'")
    fields = [H_applied]
    log.debug(f"Calling hysteresis({fields})")
    self.simulation_hysteresis(
        fields,
        save=save,
        do=do,
        convergence_check=convergence_check,
    )


def simulation_hysteresis(
    self: Any,
    H_ext_list: list[Any],
    save: ScheduleInput | None = None,
    do: ScheduleInput | None = None,
    convergence_check: When | None = None,
    progress_message_minimum_delay: float = 60.0,
) -> None:
    """
    This method executes a simulation where the applied field
    is set in sequence to the values specified in ``H_ext_list``.

    (Full docstring omitted for brevity, it is unchanged)
    """
    save, do, convergence_check = _hysteresis_defaults(save, do, convergence_check)
    log.debug(
        f"simulation_hysteresis(): Entering with H_ext_list={H_ext_list}, "
        f"save={save}, do={do}, convergence_check={convergence_check}"
    )

    thing_when_tuples = _scheduled_actions(self, save, do)
    log.debug(f"simulation_hysteresis(): thing_when_tuples={thing_when_tuples}")
    next_save_time = _initial_next_save_times(thing_when_tuples)
    match_tolerances = _hysteresis_match_tolerances()
    progress_file_name = f"{self.name}_progress.txt"
    _prepare_hysteresis_run(self, progress_file_name)
    stage: int | None = int(self.clock.stage)
    self.clock.exit_hysteresis = False
    for H_ext in H_ext_list[stage - 1 :]:
        _prepare_hysteresis_stage(
            self, H_ext, stage, thing_when_tuples, next_save_time, match_tolerances
        )
        stage = None
        if _run_hysteresis_stage(
            self,
            H_ext,
            thing_when_tuples,
            next_save_time,
            convergence_check,
            match_tolerances,
            progress_file_name,
            progress_message_minimum_delay,
        ):
            return


def _hysteresis_defaults(
    save: ScheduleInput | None,
    do: ScheduleInput | None,
    convergence_check: When | None,
) -> tuple[ScheduleInput, ScheduleInput, When]:
    return (
        [("averages", "fields", at("stage_end"))] if save is None else save,
        [] if do is None else do,
        every("step", 5) if convergence_check is None else convergence_check,
    )


def _scheduled_actions(self: Any, save: ScheduleInput, do: ScheduleInput) -> list[tuple[Any, When]]:
    return _join_save_and_do_lists(save, do, predefined_actions=self.action_abbreviations)


def _initial_next_save_times(
    actions: list[tuple[Any, When]],
) -> dict[str, CounterTriple | None]:
    next_save_time: dict[str, CounterTriple | None] = {}
    for what, _ in actions:
        key = str(what)
        if key in next_save_time:
            raise ValueError(
                "Error in optional argument 'save' or 'do' of method 'hysteresis': "
                "the list of (thing_to_save, when) contains two or more specifications "
                f"for thing_to_save = {key}. You should remove the duplicate entry and "
                "eventually use the operator | (such as in: (thing_to_save, when1 | when2))."
            )
        next_save_time[key] = None
    return next_save_time


def _hysteresis_match_tolerances() -> dict[str, Any]:
    negligible_time = SI(1e-20, "s")
    return {"time": negligible_time, "stage_time": negligible_time}


def _prepare_hysteresis_run(self: Any, progress_file_name: str) -> None:
    if self._restarting:
        log.info("Hysteresis loop: restarting from a previously saved configuration...")
        self.load_restart_file()
        self._restarting = False
        return
    log.info("Hysteresis loop: starting a new simulation.")
    log.info(f"Hysteresis loop: check file '{progress_file_name}' for progress data")


def _next_event_time(
    event: When, clock: SimulationClock, tolerances: dict[str, Any]
) -> CounterTriple:
    return _next_time(event, clock, tols=tolerances)


def _next_event_deltas(
    event: When,
    clock: SimulationClock,
    tolerances: dict[str, Any],
    suggest: CounterTriple | None = None,
) -> CounterTriple:
    return _next_deltas(event, clock, suggest=suggest, tols=tolerances)


def _prepare_hysteresis_stage(
    self: Any,
    applied_field: Any,
    stage: int | None,
    actions: list[tuple[Any, When]],
    next_save_time: dict[str, CounterTriple | None],
    tolerances: dict[str, Any],
) -> None:
    log.info(f"hysteresis: starting new stage: field = {str(applied_field)}")
    self.do_next_stage(stage=stage)
    self.clock.stage_end = False
    if applied_field is not None:
        self.set_H_ext(applied_field)
    self.reinitialise(initial_time=0)
    for what, when in actions:
        key = str(what)
        next_save_time[key] = _next_event_time(when, self.clock, tolerances)
        log.debug(f"hysteresis: will save {what} at {next_save_time[key]}")


def _run_hysteresis_stage(
    self: Any,
    applied_field: Any,
    actions: list[tuple[Any, When]],
    next_save_time: dict[str, CounterTriple | None],
    convergence_check: When,
    tolerances: dict[str, Any],
    progress_file_name: str,
    progress_delay: float,
) -> bool:
    while True:
        self.clock.stage_end = converged = self.is_converged()
        deltas = _next_event_deltas(convergence_check, self.clock, tolerances)
        deltas = _dispatch_scheduled_actions(
            self,
            actions,
            next_save_time,
            tolerances,
            deltas,
        )
        if self.clock.exit_hysteresis:
            log.debug(
                "Exit from the hysteresis loop has been forced using the tag 'exit': exiting now!"
            )
            return True
        if self.clock.stage_end:
            log.debug(
                f"Reached end of stage in hysteresis command, converged={converged}, exiting now!"
            )
            return False
        _advance_hysteresis_stage(self, deltas, applied_field, progress_file_name, progress_delay)


def _dispatch_scheduled_actions(
    self: Any,
    actions: list[tuple[Any, When]],
    next_save_time: dict[str, CounterTriple | None],
    tolerances: dict[str, Any],
    deltas: CounterTriple,
) -> CounterTriple:
    for what, when in actions:
        key = str(what)
        next_time = _next_event_time(when, self.clock, tolerances)
        if when.match_time(self.clock) or next_time != next_save_time[key]:
            log.info(
                f"hysteresis: saving {what} at id={self.clock.id},step={self.clock.step}.\n{self.clock}"
            )
            what(self)
        next_save_time[key] = next_time
        deltas = _next_event_deltas(when, self.clock, tolerances, suggest=deltas)
    return deltas


def _advance_hysteresis_stage(
    self: Any,
    deltas: CounterTriple,
    applied_field: Any,
    progress_file_name: str,
    progress_delay: float,
) -> None:
    delta_step, delta_time, delta_real_time = deltas
    target_time = (
        self.max_time_reached if delta_time is None else self.clock.stage_time + delta_time
    )
    log.debug(
        f"predicted advance: delta_step={delta_step}, delta_time={delta_time}, "
        f"delta_real_time={delta_real_time}"
    )
    time_reached = self.advance_time(target_time, max_it=-1 if delta_step is None else delta_step)
    if time_reached > 0.99 * self.max_time_reached:
        raise RuntimeError(
            f"Simulation time reached {self.max_time_reached}: "
            "are you starting from a zero torque configuration?"
        )
    _update_progress_file(self, applied_field, progress_file_name, progress_delay)
