"""Compatibility exports and seed-point assembly for relaxation meshing."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._constants import STATE_FIXED, STATE_SIMPLE
from .._types import FloatArray
from ..geometry import FemGeometry
from .periodic import _apply_periodic_fixed_states, _periodic_outer_box_points
from .points import _as_float_array as _as_float_array  # noqa: F401
from .points import _classify_dynamic_states as _classify_dynamic_states  # noqa: F401
from .points import _dedupe_fixed_mobile, _filter_relevant_points  # noqa: F401
from .points import _dedupe_points as _dedupe_points
from .sampling import _collect_hint_points, _select_generated_points


def _prepare_initial_points(
    geometry: FemGeometry,
    a0: float,
    fixed_points: FloatArray,
    mobile_points: FloatArray,
    simply_points: FloatArray,
    periodic: list[float] | list[bool],
    rng: np.random.Generator,
    params: dict[str, Any] | None = None,
) -> tuple[FloatArray, np.ndarray]:
    """Prepare the initial point cloud and point-state array for relaxation."""

    params = {} if params is None else params
    fixed_points, mobile_points, simply_points = _dedupe_fixed_mobile(
        fixed_points, mobile_points, simply_points
    )
    fixed_points = _filter_relevant_points(geometry, fixed_points)
    mobile_points = _filter_relevant_points(geometry, mobile_points)
    simply_points = _filter_relevant_points(geometry, simply_points)

    if len(simply_points) > 0:
        states = np.full(len(simply_points), STATE_SIMPLE, dtype=int)
        _apply_periodic_fixed_states(states, simply_points, geometry, periodic, a0)
        return simply_points, states

    generated_points = (
        _select_generated_points(
            geometry,
            a0,
            fixed_points,
            mobile_points,
            simply_points,
            rng,
            params,
        )
        if len(mobile_points) == 0
        else np.empty((0, geometry.dim), dtype=float)
    )
    hint_block = _collect_hint_points(geometry)
    periodic_block = _periodic_outer_box_points(geometry, a0, periodic)
    all_points = np.vstack(
        (fixed_points, simply_points, mobile_points, hint_block, periodic_block, generated_points)
    )
    states = np.concatenate(
        (
            np.full(len(fixed_points), STATE_FIXED, dtype=int),
            np.full(len(simply_points), STATE_SIMPLE, dtype=int),
            _classify_dynamic_states(geometry, mobile_points, a0),
            np.full(len(hint_block), STATE_FIXED, dtype=int),
            np.full(len(periodic_block), STATE_FIXED, dtype=int),
            _classify_dynamic_states(geometry, generated_points, a0),
        )
    )
    _apply_periodic_fixed_states(states, all_points, geometry, periodic, a0)
    return all_points, states
