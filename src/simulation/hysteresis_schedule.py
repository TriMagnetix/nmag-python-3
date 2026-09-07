"""Schedule parsing and counter arithmetic for hysteresis runs."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast

from when import When

from .clock import SimulationClock

Action = Callable[[Any], Any]
ScheduleInput = list[tuple[object, ...]]
ScheduledAction = tuple[Action, When]
CounterTriple = tuple[Any | None, Any | None, Any | None]

_subsequent_spaces = re.compile(r"[ \t_]+")


def _string_normalise(s: str, lower: bool = True, spaces: str | None = "_") -> str:
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
    target_list: list[ScheduledAction],
    input_list: ScheduleInput,
    prefix: str = "",
    predefined_actions: dict[str, Action] | None = None,
) -> None:
    """
    Internally used by `_join_save_and_do_lists` to parse the
    multi-argument 'save' and 'do' tuples.
    """
    if predefined_actions is None:
        predefined_actions = {}

    try:
        for tuple_item in input_list:
            list_item = list(tuple_item)
            if not list_item:
                continue

            things = list_item[0:-1]
            when = list_item[-1]

            for thing in things:
                if callable(thing):
                    target_list.append((cast(Action, thing), cast(When, when)))
                    continue

                if isinstance(thing, str):
                    normalised_thing = _string_normalise(f"{prefix} {thing}")
                    if normalised_thing in predefined_actions:
                        action = predefined_actions[normalised_thing]
                        target_list.append((action, cast(When, when)))
                        continue
                    else:
                        # Try without prefix
                        normalised_thing = _string_normalise(thing)
                        if normalised_thing in predefined_actions:
                            action = predefined_actions[normalised_thing]
                            target_list.append((action, cast(When, when)))
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

    except TypeError as e:
        msg = (
            f"Bad syntax for argument '{prefix}' of the method 'hysteresis': "
            f"remember that you should provide a list of tuples "
            f"(things_to_{prefix}, when). You can also provide tuples "
            f"with many things to save, such as (thing1, thing2, "
            f"..., when)."
        )
        raise ValueError(msg) from e


def _join_save_and_do_lists(
    save_list: ScheduleInput,
    do_list: ScheduleInput,
    predefined_actions: dict[str, Action] | None = None,
) -> list[ScheduledAction]:
    """
    Takes 'save' and 'do' parameters from the 'hysteresis' method
    and joins them into a unique list.
    Tags (e.g., "save averages") are replaced with the appropriate functions.
    Also checks that the provided tags actually exist.
    """
    if predefined_actions is None:
        predefined_actions = {}

    joint_list: list[ScheduledAction] = []

    # Note: It is important to process 'do' before 'save'.
    # A 'do' command might be 'next_stage', and 'save' commands
    # need to know if this is the last step of the stage.
    _append_x_list(joint_list, do_list, prefix="do", predefined_actions=predefined_actions)
    _append_x_list(joint_list, save_list, prefix="save", predefined_actions=predefined_actions)

    # XXX Matteo, can we at this point order the save_ entries such that
    # the save_restart is the last? See my explanation in ticket:169.
    # Hans, 11/ll/2008

    return joint_list


def _next_deltas(
    event: When,
    clock: SimulationClock,
    suggest: CounterTriple | None = None,
    tols: dict[str, Any] | None = None,
) -> CounterTriple:
    """
    Return the next occurrence of the given event as a triple of deltas:
    - (delta_step, delta_time, delta_real_time)
    """

    def delta(name: str) -> Any | None:
        n = event.next_time(name, clock, tols=tols)
        return n - clock[name] if not isinstance(n, bool) else None

    def minimum(ls: list[Any | None]) -> Any | None:
        return min((item for item in ls if item is not None), default=None)

    delta_step, delta_time, delta_real_time = suggest or (None, None, None)

    delta_step = minimum([delta("step"), delta("stage_step"), delta_step])
    delta_time = minimum([delta("time"), delta("stage_time"), delta_time])
    delta_real_time = minimum([delta("real_time"), delta_real_time])

    return (delta_step, delta_time, delta_real_time)


def _next_time(
    event: When,
    clock: SimulationClock,
    tols: dict[str, Any] | None = None,
) -> CounterTriple:
    """
    Returns a triple of absolute counter values for the next event:
    - (next_step, next_time, next_real_time)
    """

    def delta(name: str, starting_time: str | None = None) -> Any | None:
        n = event.next_time(name, clock, tols=tols)
        if isinstance(n, bool):
            return None
        return n - clock[starting_time] if starting_time else n

    def minimum(ls: list[Any | None]) -> Any | None:
        return min((item for item in ls if item is not None), default=None)

    next_time = minimum([delta("time", "zero_stage_time"), delta("stage_time")])

    next_step = minimum([delta("step", "zero_stage_step"), delta("stage_step")])

    next_real_time = delta("real_time")

    return (next_step, next_time, next_real_time)
