from __future__ import annotations

from typing import Any

from . import geometry as _geometry
from . import lindholm as _lindholm
from . import lindholm_fast as _fast
from . import linear as _linear

_HELPER_MODULES = (_geometry, _lindholm, _fast, _linear)

# Preserve the established patch points used by downstream tests and tooling.
np = _linear.np
_scipy_linalg = _linear._scipy_linalg


def _solve_gauge_fixed(*args: Any, **kwargs: Any) -> Any:
    """Forward to the solver while preserving facade-level monkeypatches."""

    _linear._scipy_linalg = _scipy_linalg
    return _linear._solve_gauge_fixed(*args, **kwargs)


def _solve_linear_system(*args: Any, **kwargs: Any) -> Any:
    """Forward to the solver while preserving facade-level monkeypatches."""

    _linear._scipy_linalg = _scipy_linalg
    return _linear._solve_linear_system(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Resolve compatibility exports from the focused demag modules."""

    for module in _HELPER_MODULES:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
