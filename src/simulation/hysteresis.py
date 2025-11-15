
"""
This file contains the implementation of the hysteresis method
of the Simulation class.
"""

from __future__ import annotations
import re
import time
import logging
from typing import (
    Callable, Any, List, Tuple, Dict, Optional, Union
)
from si.physical import SI
from throttler import Throttler
from when import at, every, never, TimeDict

log = logging.getLogger('nmag')

_subsequent_spaces = re.compile(r'[ \t_]+')

_progress_throttler = Throttler()

def _update_progress_file(
    sim: Any,
    H_ext: Any,
    progress_file_name: str,
    progress_message_minimum_delay: float
):
    """
    Writes the current simulation progress to a file, throttled
    to a minimum delay.
    """
    if _progress_throttler.is_allowed('hysteresis_reporting',
                                      progress_message_minimum_delay):
        
        try:
            with open(progress_file_name, 'w') as f:
                f.write(f"{time.asctime()}\n")
                f.write(f"{sim.clock}\n")
                f.write(f"{sim.convergence.get_log()}\n")

            log.info(
                f"it {sim.clock.step}, time {sim.clock.time_reached_si.dens_str()}; "
                f"stage {sim.clock.stage}; H_ext={str(H_ext)}"
            )
        except Exception as e:
            log.warning(
                f"Could not write progress file '{progress_file_name}'. Error: {e}"
            )

def _string_normalise(
    s: str, 
    lower: bool = True, 
    spaces: Optional[str] = '_'
) -> str:
    """
    Used by `_append_x_list` to obtain a standard string format.
    Examples:
        'save  fields' --> 'save_fields'
        'Save_Fields'  --> 'save_fields'
    """
    ns = s
    if lower:
        ns = s.lower()
    if spaces is not None:
        ns = re.sub(_subsequent_spaces, spaces, ns)
    return ns

def _append_x_list(
    target_list: List[Tuple[Callable, Any]],
    input_list: List[Tuple],
    prefix: str = "",
    predefined_actions: Dict[str, Callable] = {}
):
    """
    Internally used by `_join_save_and_do_lists` to parse the
    multi-argument 'save' and 'do' tuples.
    """
    try:
        for tuple_item in input_list:
            list_item = list(tuple_item)
            if not list_item:
                continue
                
            things = list_item[0:-1]
            when = list_item[-1]
            
            for thing in things:
                if isinstance(thing, Callable):
                    target_list.append((thing, when))
                    continue

                if isinstance(thing, str):
                    normalised_thing = _string_normalise(f"{prefix} {thing}")
                    if normalised_thing in predefined_actions:
                        action = predefined_actions[normalised_thing]
                        target_list.append((action, when))
                        continue
                    else:
                        # Try without prefix
                        normalised_thing = _string_normalise(thing)
                        if normalised_thing in predefined_actions:
                            action = predefined_actions[normalised_thing]
                            target_list.append((action, when))
                            continue

                # If we are here, the 'thing' is not a callable and not
                # a recognized string.
                msg = (
                    f"Error in optional argument '{prefix}' of method "
                    f"'hysteresis': you want to {prefix} '{thing}' "
                    f"but I don't know how to do it. Hint: when specifying "
                    f"what to {prefix} you can use a string or a function. "
                    f"Available strings are: {', '.join(predefined_actions.keys())}."
                )
                raise ValueError(msg)

    except TypeError:
        msg = (
            f"Bad syntax for argument '{prefix}' of the method 'hysteresis': "
            f"remember that you should provide a list of tuples "
            f"(things_to_{prefix}, when). You can also provide tuples "
            f"with many things to save, such as (thing1, thing2, "
            f"..., when)."
        )
        raise ValueError(msg)

def _join_save_and_do_lists(
    save_list: List[Tuple],
    do_list: List[Tuple],
    predefined_actions: Dict[str, Callable] = {}
) -> List[Tuple[Callable, Any]]:
    """
    Takes 'save' and 'do' parameters from the 'hysteresis' method
    and joins them into a unique list.
    Tags (e.g., "save averages") are replaced with the appropriate functions.
    Also checks that the provided tags actually exist.
    """
    joint_list: List[Tuple[Callable, Any]] = []

    # Note: It is important to process 'do' before 'save'.
    # A 'do' command might be 'next_stage', and 'save' commands
    # need to know if this is the last step of the stage.
    _append_x_list(joint_list, do_list, prefix="do",
                   predefined_actions=predefined_actions)
    _append_x_list(joint_list, save_list, prefix="save",
                   predefined_actions=predefined_actions)

    # XXX Matteo, can we at this point order the save_ entries such that
    # the save_restart is the last? See my explanation in ticket:169.
    # Hans, 11/ll/2008

    return joint_list

def _next_deltas(
    event: Any, 
    clock: Any, 
    suggest: Optional[Tuple] = None, 
    tols: Optional[Dict] = None
) -> Tuple:
    """
    Return the next occurrence of the given event as a triple of deltas:
    - (delta_step, delta_time, delta_real_time)
    """
    def delta(name: str) -> Optional[float]:
        n = event.next_time(name, clock, tols=tols)
        return n - clock[name] if not isinstance(n, bool) else None

    def minimum(ls: List[Optional[float]]) -> Optional[float]:
        return min((item for item in ls if item is not None), default=None)

    delta_step, delta_time, delta_real_time = suggest or (None, None, None)

    delta_step = minimum([delta('step'), delta('stage_step'), delta_step])
    delta_time = minimum([delta('time'), delta('stage_time'), delta_time])
    delta_real_time = minimum([delta('real_time'), delta_real_time])

    return (delta_step, delta_time, delta_real_time)

def _next_time(
    event: Any, 
    clock: Any, 
    tols: Optional[Dict] = None
) -> Tuple:
    """
    Returns a triple of absolute counter values for the next event:
    - (next_step, next_time, next_real_time)
    """
    def delta(name: str, starting_time: Optional[str] = None) -> Optional[float]:
        n = event.next_time(name, clock, tols=tols)
        if isinstance(n, bool):
            return None
        return n - clock[starting_time] if starting_time else n

    def minimum(ls: List[Optional[float]]) -> Optional[float]:
        return min((item for item in ls if item is not None), default=None)

    next_time = minimum([delta('time', 'zero_stage_time'),
                         delta('stage_time')])
    
    next_step = minimum([delta('step', 'zero_stage_step'),
                         delta('stage_step')])

    next_real_time = delta('real_time')
    
    return (next_step, next_time, next_real_time)

def simulation_relax(
    self,
    H_applied: Any = None,
    save: List[Tuple] = [('averages', 'fields', at('stage_end'))],
        do: List[Tuple] = [],
        convergence_check: Any = every('step', 5)
    ):
        """
        This method carries out the time integration of the LLG until
        the system reaches a (metastable) equilibrium.
        Internally, this uses the hysteresis() loop command.
        
        (Docstring parameters omitted for brevity, they are unchanged)
        """
        log.debug("Entering 'relax'")
        fields = [H_applied]
        log.debug(f"Calling hysteresis({fields})")
        return self.simulation_hysteresis(
            fields,
            save=save,
            do=do,
            convergence_check=convergence_check
        )

def simulation_hysteresis(
    self,  # 'self' is kept as per the original file structure
    H_ext_list: List[Any],
    save: List[Tuple] = [('averages', 'fields', at('stage_end'))],
        do: List[Tuple] = [],
        convergence_check: Any = every('step', 5),
        progress_message_minimum_delay: float = 60.0
    ):
        """
        This method executes a simulation where the applied field
        is set in sequence to the values specified in ``H_ext_list``.
        
        (Full docstring omitted for brevity, it is unchanged)
        """
        log.debug(
            f"simulation_hysteresis(): Entering with H_ext_list={H_ext_list}, "
            f"save={save}, do={do}, convergence_check={convergence_check}"
        )

        thing_when_tuples = _join_save_and_do_lists(
            save, 
            do,
            predefined_actions=self.action_abbreviations
        )

        log.debug(f"simulation_hysteresis(): thing_when_tuples={thing_when_tuples}")

        next_save_time: Dict[str, Any] = {}
        for what, _ in thing_when_tuples:
            key = str(what)
            if key in next_save_time:
                msg = (
                    f"Error in optional argument 'save' or 'do' of method "
                    f"'hysteresis': the list of (thing_to_save, when) "
                    f"contains two or more specifications for "
                    f"thing_to_save = {key}. You should remove the duplicate "
                    f"entry and eventually use the operator | (such as in: "
                    f"(thing_to_save, when1 | when2))."
                )
                raise ValueError(msg)
            next_save_time[key] = None

        # Tolerances for floating point comparisons
        negligible_time = SI(1e-20, "s")
        match_tolerances = {
            'time': negligible_time,
            'stage_time': negligible_time
        }

        def my_next_time(event, clock):
            return _next_time(event, clock, tols=match_tolerances)

        def my_next_deltas(event, clock, suggest=None):
            return _next_deltas(event, clock, suggest=suggest,
                                tols=match_tolerances)

        progress_file_name = f"{self.name}_progress.txt"

        if self._restarting:
            log.info("Hysteresis loop: restarting from a previously saved "
                     "configuration...")
            self.load_restart_file()
            self._restarting = False # Reset flag
        else:
            log.info("Hysteresis loop: starting a new simulation.")
            log.info(f"Hysteresis loop: check file '{progress_file_name}' "
                     "for progress data")

        stage = self.clock.stage
        self.clock.exit_hysteresis = False
        
        for H_ext in H_ext_list[stage-1:]:
            log.info(f"hysteresis: starting new stage: field = {str(H_ext)}")
            self.do_next_stage(stage=stage)
            stage = None  # Next time, just increase the stage counter
            self.clock.stage_end = False
            if H_ext:
                self.set_H_ext(H_ext)
            self.reinitialise(initial_time=0)

            # Update schedule for when to save what
            for what, when in thing_when_tuples:
                key = str(what)
                next_save_time[key] = my_next_time(when, self.clock)
                log.debug(f"hysteresis: will save {what} at {next_save_time[key]}")

            # --- Main Stage Loop ---
            while True:
                self.clock.stage_end = converged = self.is_converged()
                log.debug(f"hysteresis loop, stage {self.clock.stage}, "
                          f"converged = {converged}")

                # Find out the next time we need to check for convergence
                deltas = my_next_deltas(convergence_check, self.clock)
                log.debug(f"Time to next event: deltas = {deltas}")

                # Check what needs to be saved/done
                for what, when in thing_when_tuples:
                    key = str(what)
                    time_matches = when.match_time(self.clock)
                    nst = my_next_time(when, self.clock)

                    # Check if the event is scheduled to run now
                    if time_matches or nst != next_save_time[key]:
                        log.debug(
                            f"hysteresis: analysing {what}: time planned "
                            f"for saving was {next_save_time[key]}, "
                            f"now is {nst}. Matching? {time_matches}"
                        )
                        log.info(
                            f"hysteresis: saving {what} at id={self.clock.id},"
                            f"step={self.clock.step}.\n{self.clock}"
                        )
                        what(self) # Run the action/save function

                    next_save_time[key] = nst
                    deltas = my_next_deltas(when, self.clock, suggest=deltas)

                (delta_step, delta_time, delta_real_time) = deltas
                log.debug(f"hysteresis: current time is {self.clock.time}")
                log.debug(
                    f"predicted advance: delta_step={delta_step}, "
                    f"delta_time={delta_time}, "
                    f"delta_real_time={delta_real_time}"
                )

                if delta_time is None:
                    target_time = self.max_time_reached
                else:
                    target_time = self.clock.stage_time + delta_time

                delta_step = delta_step if delta_step is not None else -1

                if self.clock.exit_hysteresis:
                    log.debug("Exit from the hysteresis loop has been forced "
                              "using the tag 'exit': exiting now!")
                    return

                if self.clock.stage_end:
                    log.debug(f"Reached end of stage in hysteresis command, "
                              f"converged={converged}, exiting now!")
                    break

                log.debug(f"About to call advance time with target_time={target_time} "
                          f"and max_it={delta_step}")
                
                time_reached = self.advance_time(target_time, max_it=delta_step)
                
                if time_reached > 0.99 * self.max_time_reached:
                    msg = (f"Simulation time reached {self.max_time_reached}: "
                           "are you starting from a zero torque configuration?")
                    raise RuntimeError(msg)

                # Write progress data to file
                _update_progress_file(self, H_ext, progress_file_name,
                                      progress_message_minimum_delay)
            
            # --- End of Main Stage Loop ---
        # --- End of H_ext_list Loop ---
