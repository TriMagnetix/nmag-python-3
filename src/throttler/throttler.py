import time
from typing import Hashable

"""
This replaces the reporttools.py from the original nmag repo.
https://github.com/nmag-project/nmag-src/blob/master/interface/nsim/reporttools.py.
It is converted into a class to avoid global variables and uses monotonic time instead.
"""
class Throttler:
    """
    Manages state to allow actions at most every N seconds per key.

    Attributes:
        last_called (dict[Hashable, float]): A dictionary mapping
            a unique key to the timestamp (from time.monotonic())
            of its last allowed call.
    """
    def __init__(self):
        """Initializes the throttle state."""
        # Stores the time of the last successful call for each key
        self.last_called: dict[Hashable, float] = {}

    def is_allowed(self, key: Hashable, report_delay: float) -> bool:
        """
        Checks if an action for a given key should be allowed.

        Returns True if 'report_delay' seconds have passed since the
        last time this method returned True for the same 'key',
        or if this is the first call for this 'key'.

        Args:
            key: A unique, hashable identifier for the action
                 being rate-limited (e.g., a string, a tuple).
            report_delay: The minimum number of seconds that must
                          pass between True returns.

        Returns:
            True if the action is allowed, False otherwise.
        """
        now = time.monotonic()

        # Get the last time this key was allowed.
        # Default to 0.0, ensuring the first call always passes
        last_time = self.last_called.get(key, 0.0)

        if now - last_time >= report_delay:
            # Time has elapsed. Allow the action and update the timestamp.
            self.last_called[key] = now
            return True

        # Not enough time has passed. Deny the action.
        return False
