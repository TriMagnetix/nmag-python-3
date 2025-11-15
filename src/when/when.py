"""
Time specification utilities for specifying when to do things
(such as saving fields or averages).
"""

import abc
import math
from typing import Any, Dict, Union, Optional

TimeDict = Dict[str, Any]
NextTime = Union[bool, int, float]


def _float_is_integer(f: float) -> bool:
    return math.isclose(f, round(f))


# --- Abstract Base Class for Specification Logic ---

class _WhenSpec(abc.ABC):
    """
    Abstract base class for a time specification.
    This is an internal implementation detail. The user interacts
    with the 'When' class, which wraps this.
    """

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return the string representation."""
        raise NotImplementedError

    @abc.abstractmethod
    def match_time(self, this_time: TimeDict) -> bool:
        """Check if the current time matches this specification."""
        raise NotImplementedError

    @abc.abstractmethod
    def next_time(self, identifier: str, this_time: TimeDict) -> NextTime:
        """Calculate the next matching time for the given identifier."""
        raise NotImplementedError


# --- Concrete Specification Implementations ---

class _AtSpec(_WhenSpec):
    """Specification for a single point in time."""
    
    def __init__(self, identifier: str, value: Any):
        self.identifier = identifier
        self.value = value

    def __repr__(self) -> str:
        return f"at({self.identifier!r}, {self.value!r})"

    def match_time(self, this_time: TimeDict) -> bool:
        return this_time.get(self.identifier) == self.value

    def next_time(self, identifier: str, this_time: TimeDict) -> NextTime:
        if self.identifier != identifier:
            return True  # Irrelevant to this identifier, so don't block
        
        t = this_time.get(identifier)
        if isinstance(t, bool):
            if self.value is True:
                return True
            else:
                return t == self.value
        else:
            if t < self.value:
                return self.value
            else:
                return False


class _EverySpec(_WhenSpec):
    """Specification for a periodic event."""

    def __init__(self, identifier: str, delta: Optional[float], 
                 first: float, last: Optional[float]):
        self.identifier = identifier
        self.delta = delta
        self.first = first
        self.last = last

    def __repr__(self) -> str:
        opts = ""
        if self.first != 0:
            opts += f", first={self.first}"
        if self.last is not None:
            opts += f", last={self.last}"
        
        delta_str = str(self.delta) if self.delta is not None else "None"
        return f"every({delta_str}, {self.identifier!r}{opts})"

    def match_time(self, this_time: TimeDict) -> bool:
        t = this_time.get(self.identifier)
        if t is None:
            return False
            
        if self.first is not None and t < self.first:
            return False
        if self.last is not None and t > self.last:
            return False
        if self.delta is None:
            return True
        
        assert self.first is not None, "delta requires a 'first' value"
        
        if self.delta <= 0:
            return False
            
        pos = float((t - self.first) / self.delta)
        return _float_is_integer(pos)

    def next_time(self, identifier: str, this_time: TimeDict) -> NextTime:
        if self.identifier != identifier:
            return self.match_time(this_time)

        t = this_time.get(identifier)
        if t is None:
            return False

        if self.last is not None and t >= self.last:
            return False
            
        if self.first is not None:
            if t < self.first and self.delta is not None:
                return self.first
                
        if self.delta is None:
            return True

        assert self.first is not None, "delta requires a 'first' value"
        
        if self.delta <= 0:
            return False

        pos = float((t - self.first) / self.delta)
        
        if _float_is_integer(pos):
            next_pos = int(round(pos)) + 1
        else:
            next_pos = int(pos) + 1
            
        next_t = self.delta * next_pos + self.first

        if self.last is not None and next_t > self.last:
            return False
            
        return next_t


class _OrSpec(_WhenSpec):
    """Specification for a logical OR of two specifications."""

    def __init__(self, spec1: _WhenSpec, spec2: _WhenSpec):
        self.spec1 = spec1
        self.spec2 = spec2

    def __repr__(self) -> str:
        return f"({self.spec1!r} | {self.spec2!r})"

    def match_time(self, this_time: TimeDict) -> bool:
        return self.spec1.match_time(this_time) or \
               self.spec2.match_time(this_time)

    def next_time(self, identifier: str, this_time: TimeDict) -> NextTime:
        nt1 = self.spec1.next_time(identifier, this_time)
        nt2 = self.spec2.next_time(identifier, this_time)

        nt1_is_bool = isinstance(nt1, bool)
        nt2_is_bool = isinstance(nt2, bool)

        if nt1_is_bool and nt2_is_bool:
            return nt1 or nt2
        
        if nt1 is False:
            return nt2
        if nt2 is False:
            return nt1

        if nt1 is True:
            return True
        if nt2 is True:
            return True
            
        return min(nt1, nt2)


class _AndSpec(_WhenSpec):
    """Specification for a logical AND of two specifications."""

    def __init__(self, spec1: _WhenSpec, spec2: _WhenSpec):
        self.spec1 = spec1
        self.spec2 = spec2

    def __repr__(self) -> str:
        return f"({self.spec1!r} & {self.spec2!r})"

    def match_time(self, this_time: TimeDict) -> bool:
        return self.spec1.match_time(this_time) and \
               self.spec2.match_time(this_time)

    def next_time(self, identifier: str, this_time: TimeDict) -> NextTime:
        # Use a copy to avoid side effects on the original time dict
        temp_time = this_time.copy()
        save_t = temp_time.get(identifier)
        if save_t is None:
            return False # Can't calculate if identifier is missing

        ntmax: Optional[Union[int, float]] = None
        
        both_match = False
        while not both_match:
            nt1 = self.spec1.next_time(identifier, temp_time)
            nt2 = self.spec2.next_time(identifier, temp_time)

            if nt1 is False or nt2 is False:
                return False

            if nt1 is True:
                return nt2
            if nt2 is True:
                return nt1

            if nt1 > nt2:
                ntmax, argmin = nt1, self.spec2
            else:
                ntmax, argmin = nt2, self.spec1

            temp_time[identifier] = ntmax
            both_match = argmin.match_time(temp_time)


        assert ntmax is not None, "Logic error: 'and' loop exited without setting ntmax"

        return ntmax


class _NeverSpec(_WhenSpec):
    """Specification that never matches."""
    
    def __repr__(self) -> str:
        return "never"

    def match_time(self, this_time: TimeDict) -> bool:
        return False

    def next_time(self, identifier: str, this_time: TimeDict) -> NextTime:
        return False


# --- Public-Facing 'When' Class ---

class When:
    """
    Class used to express when a certain thing should be done.
    
    This class is a wrapper around a 'spec' object that implements
    the actual logic.
    """
    def __init__(self, spec: _WhenSpec):
        self.spec = spec

    def __repr__(self) -> str:
        return f"<When: {self.spec!r}>"

    def __str__(self) -> str:
        return repr(self.spec)

    def match_time(self, this_time: TimeDict) -> bool:
        """Checks if the specification matches the given time."""
        return self.spec.match_time(this_time)

    def next_time(self, identifier: str, this_time: TimeDict, 
                  tols: Optional[Dict[str, float]] = None) -> NextTime:
        """
        Calculates the next matching time for the given identifier.
        
        The 'tols' argument is used to apply a tolerance to prevent
        re-triggering at the exact same time due to floating point
        inaccuracies.
        """
        nt = self.spec.next_time(identifier, this_time)

        # Apply tolerance logic from the original class
        if (tols is not None 
                and identifier in tols 
                and isinstance(nt, (int, float))):
            
            tol = tols[identifier]
            tt = this_time[identifier]
            
            if tol > 0.0 and abs(nt - tt) < tol:
                # We are too close to the current time.
                # Advance time slightly and recalculate.
                temp_time = this_time.copy()
                temp_time[identifier] = nt + tol
                nt = self.spec.next_time(identifier, temp_time)

        return nt

    def __or__(self, other: "When") -> "When":
        """Combines two 'When' objects with a logical OR."""
        if not isinstance(other, When):
            return NotImplemented
        return When(_OrSpec(self.spec, other.spec))

    def __and__(self, other: "When") -> "When":
        """
        Combines two 'When' objects with a logical AND.
        
        WARNING: As in the original, this can lead to infinite loops
        if the two conditions are mutually exclusive (e.g.,
        every('step', 2) & every('step', 2, first=1)).
        Use with care.
        """
        if not isinstance(other, When):
            return NotImplemented
        return When(_AndSpec(self.spec, other.spec))


# --- Factory Functions (Public API) ---

def at(identifier: str, value: Any = True) -> When:
    """
    Specifies an action at an exact point in time.
    Examples: at('convergence'), at('step', 10)
    """
    return When(_AtSpec(identifier, value))


def every(arg1: Union[str, float, int],
          arg2: Optional[Union[str, float, int]] = None,
          first: float = 0.0,
          last: Optional[float] = None) -> When:
    """
    Specifies an action that should be performed periodically.
    Examples:
      every('step', 10)
      every('step', 5, first=10, last=100)
      every(10, 'step')  # Legacy support, should not be used
    """
    
    identifier: Optional[str] = None
    delta: Optional[Union[float, int]] = None

    # Handle swapped arguments with explicit type-checking
    if isinstance(arg1, str):
        identifier = arg1
        if isinstance(arg2, (int, float)):
            delta = arg2
        elif arg2 is None:
            delta = None  # e.g., every('step', first=10)
        # If arg2 is a str, we let it fail validation below
            
    elif isinstance(arg1, (int, float)):
        delta = arg1
        if isinstance(arg2, str):
            identifier = arg2
        # If arg2 is numeric or None, we let it fail validation below

    # --- Validation ---

    # 1. Check if identifier was found and is a string
    if not isinstance(identifier, str):
        raise ValueError(
            "Bad usage of 'every': you must specify an identifier (string). "
            "Example: every('step', 10)"
        )

    # 2. Check 'last' vs 'first'
    if last is not None and last <= first:
        raise ValueError(
            "Bad usage of 'every': 'last' must be greater than 'first'. "
            f"Got first={first}, last={last}"
        )

    # 3. Check delta
    if delta is not None:
        if delta <= 0:
            raise ValueError(
                "Bad usage of 'every': delta must be positive. "
                f"Got delta={delta}"
            )

    return When(_EverySpec(identifier, delta, first, last))


# A singleton instance for the 'never' specification
never = When(_NeverSpec())
