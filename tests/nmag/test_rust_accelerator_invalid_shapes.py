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

INVALID_RANK_AND_SHAPE_CALLS: tuple[InvalidCall, ...] = (
    (
        "lindholm rank",
        lambda: nmag_accel.build_lindholm_bem_matrix(
            POINTS.ravel(),
            SIMPLICES,
            FACE_NODES,
            BOUNDARY_NODES,
            LOCAL_INDEX_BY_POINT,
        ),
        TypeError,
        "points",
    ),
    (
        "lindholm shape",
        lambda: nmag_accel.build_lindholm_bem_matrix(
            POINTS,
            SIMPLICES,
            FACE_NODES[:, :2],
            BOUNDARY_NODES,
            LOCAL_INDEX_BY_POINT,
        ),
        ValueError,
        r"face_nodes must have shape \(n, 3\)",
    ),
    (
        "hierarchical lindholm point shape",
        lambda: nmag_accel.build_lindholm_hmatrix(
            POINTS[:, :2],
            SIMPLICES,
            FACE_NODES,
            BOUNDARY_NODES,
            LOCAL_INDEX_BY_POINT,
        ),
        ValueError,
        r"points must have shape \(n, 3\)",
    ),
    (
        "hierarchical lindholm face shape",
        lambda: nmag_accel.build_lindholm_hmatrix(
            POINTS,
            SIMPLICES,
            FACE_NODES[:, :2],
            BOUNDARY_NODES,
            LOCAL_INDEX_BY_POINT,
        ),
        ValueError,
        r"face_nodes must have shape \(n, 3\)",
    ),
    (
        "fem geometry rank",
        lambda: nmag_accel.build_demag_fem_geometry(POINTS.ravel(), SIMPLICES),
        TypeError,
        "points",
    ),
    (
        "fem geometry shape",
        lambda: nmag_accel.build_demag_fem_geometry(POINTS[:, :2], SIMPLICES),
        ValueError,
        r"points must have shape \(n, 3\)",
    ),
    (
        "fem divergence rank",
        lambda: nmag_accel.build_demag_fem_divergence(
            SIMPLICES,
            np.zeros((1, 12)),
            np.ones(1),
            POINTS,
            np.ones(1),
            4,
        ),
        TypeError,
        "gradients_by_cell",
    ),
    (
        "fem divergence shape",
        lambda: nmag_accel.build_demag_fem_divergence(
            SIMPLICES,
            np.zeros((1, 4, 2)),
            np.ones(1),
            POINTS,
            np.ones(1),
            4,
        ),
        ValueError,
        r"gradients_by_cell must have shape \(n, 4, 3\)",
    ),
    (
        "nodal recovery rank",
        lambda: nmag_accel.recover_demag_nodal_field(
            SIMPLICES,
            np.ones(1),
            np.ones(3),
            np.ones(4),
            4,
        ),
        TypeError,
        "cell_h",
    ),
    (
        "nodal recovery shape",
        lambda: nmag_accel.recover_demag_nodal_field(
            SIMPLICES,
            np.ones(1),
            np.ones((1, 2)),
            np.ones(4),
            4,
        ),
        ValueError,
        r"cell_h must have shape \(n, 3\)",
    ),
    (
        "cell average rank",
        lambda: nmag_accel.demag_cell_field_average(np.ones(3), np.ones(1)),
        TypeError,
        "cell_h",
    ),
    (
        "cell average shape",
        lambda: nmag_accel.demag_cell_field_average(np.ones((1, 2)), np.ones(1)),
        ValueError,
        r"cell_h must have shape \(n, 3\)",
    ),
    (
        "boundary faces rank",
        lambda: nmag_accel.build_oriented_boundary_faces(POINTS.ravel(), SIMPLICES),
        TypeError,
        "points",
    ),
    (
        "boundary faces shape",
        lambda: nmag_accel.build_oriented_boundary_faces(POINTS, SIMPLICES[:, :3]),
        ValueError,
        r"simplices must have shape \(n, 4\)",
    ),
    (
        "probe geometry rank",
        lambda: nmag_accel.build_probe_tetrahedral_geometry(POINTS.ravel(), SIMPLICES),
        TypeError,
        "points",
    ),
    (
        "probe geometry shape",
        lambda: nmag_accel.build_probe_tetrahedral_geometry(POINTS[:, :2], SIMPLICES),
        ValueError,
        r"points must have shape \(n, 3\)",
    ),
    (
        "llg rank",
        lambda: nmag_accel.llg_rhs(
            POINTS.ravel(),
            POINTS,
            np.ones(4),
            np.ones(4),
            1.0,
            1.0,
            1.0,
        ),
        TypeError,
        "m",
    ),
    (
        "llg shape",
        lambda: nmag_accel.llg_rhs(
            POINTS[:, :2],
            POINTS[:, :2],
            np.ones(4),
            np.ones(4),
            1.0,
            1.0,
            1.0,
        ),
        ValueError,
        r"m must have shape \(n, 3\)",
    ),
    (
        "maxangle rank",
        lambda: nmag_accel.maxangle_between_edges(POINTS.ravel(), SIMPLICES),
        TypeError,
        "m",
    ),
    (
        "maxangle shape",
        lambda: nmag_accel.maxangle_between_edges(POINTS, SIMPLICES[:, :1]),
        ValueError,
        "simplex_size>=2",
    ),
)


@pytest.mark.parametrize(
    ("_label", "call", "error", "match"),
    INVALID_RANK_AND_SHAPE_CALLS,
    ids=[case[0] for case in INVALID_RANK_AND_SHAPE_CALLS],
)
def test_functions_reject_invalid_rank_and_shape(
    _label: str,
    call: Callable[[], object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        call()
