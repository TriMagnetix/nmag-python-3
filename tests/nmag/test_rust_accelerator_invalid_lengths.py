from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from rust_accelerator_validation_support import (
    BOUNDARY_NODES,
    FACE_NODES,
    LOCAL_INDEX_BY_POINT,
    POINTS,
    SIMPLICES,
    InvalidCall,
    nmag_accel,
)

MISMATCHED_LENGTH_CALLS: tuple[InvalidCall, ...] = (
    (
        "lindholm point map",
        lambda: nmag_accel.build_lindholm_bem_matrix(
            POINTS,
            SIMPLICES,
            FACE_NODES,
            BOUNDARY_NODES,
            LOCAL_INDEX_BY_POINT[:-1],
        ),
        ValueError,
        "one entry per point",
    ),
    (
        "fem divergence gradients",
        lambda: nmag_accel.build_demag_fem_divergence(
            SIMPLICES,
            np.zeros((2, 4, 3)),
            np.ones(1),
            POINTS,
            np.ones(1),
            4,
        ),
        ValueError,
        r"gradients_by_cell must have shape \(n, 4, 3\)",
    ),
    (
        "fem divergence volumes",
        lambda: nmag_accel.build_demag_fem_divergence(
            SIMPLICES,
            np.zeros((1, 4, 3)),
            np.ones(2),
            POINTS,
            np.ones(1),
            4,
        ),
        ValueError,
        "one value per simplex",
    ),
    (
        "fem divergence magnetisation",
        lambda: nmag_accel.build_demag_fem_divergence(
            SIMPLICES,
            np.zeros((1, 4, 3)),
            np.ones(1),
            POINTS[:-1],
            np.ones(1),
            4,
        ),
        ValueError,
        r"m must have shape \(point_count, 3\)",
    ),
    (
        "fem divergence saturation",
        lambda: nmag_accel.build_demag_fem_divergence(
            SIMPLICES,
            np.zeros((1, 4, 3)),
            np.ones(1),
            POINTS,
            np.ones(2),
            4,
        ),
        ValueError,
        "one value per simplex",
    ),
    (
        "nodal recovery volumes",
        lambda: nmag_accel.recover_demag_nodal_field(
            SIMPLICES,
            np.ones(2),
            np.ones((1, 3)),
            np.ones(4),
            4,
        ),
        ValueError,
        "one value per simplex",
    ),
    (
        "nodal recovery cells",
        lambda: nmag_accel.recover_demag_nodal_field(
            SIMPLICES,
            np.ones(1),
            np.ones((2, 3)),
            np.ones(4),
            4,
        ),
        ValueError,
        r"cell_h must have shape \(n, 3\)",
    ),
    (
        "nodal recovery weights",
        lambda: nmag_accel.recover_demag_nodal_field(
            SIMPLICES,
            np.ones(1),
            np.ones((1, 3)),
            np.ones(3),
            4,
        ),
        ValueError,
        "one value per point",
    ),
    (
        "cell average volumes",
        lambda: nmag_accel.demag_cell_field_average(np.ones((2, 3)), np.ones(1)),
        ValueError,
        "one value per cell_h row",
    ),
    (
        "llg total field",
        lambda: nmag_accel.llg_rhs(
            POINTS,
            POINTS[:-1],
            np.ones(4),
            np.ones(4),
            1.0,
            1.0,
            1.0,
        ),
        ValueError,
        "same shape as m",
    ),
    (
        "llg pin",
        lambda: nmag_accel.llg_rhs(
            POINTS,
            POINTS,
            np.ones(3),
            np.ones(4),
            1.0,
            1.0,
            1.0,
        ),
        ValueError,
        "one value per mesh point",
    ),
    (
        "llg saturation",
        lambda: nmag_accel.llg_rhs(
            POINTS,
            POINTS,
            np.ones(4),
            np.ones(3),
            1.0,
            1.0,
            1.0,
        ),
        ValueError,
        "one value per mesh point",
    ),
)


@pytest.mark.parametrize(
    ("_label", "call", "error", "match"),
    MISMATCHED_LENGTH_CALLS,
    ids=[case[0] for case in MISMATCHED_LENGTH_CALLS],
)
def test_functions_reject_mismatched_lengths(
    _label: str,
    call: Callable[[], object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        call()
