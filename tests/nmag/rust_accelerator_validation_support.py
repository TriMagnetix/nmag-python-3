from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

nmag_accel = pytest.importorskip("nmag_accel")

POINTS = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
SIMPLICES = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
FACE_NODES = np.asarray([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64)
BOUNDARY_NODES = np.arange(4, dtype=np.int64)
LOCAL_INDEX_BY_POINT = np.arange(4, dtype=np.int64)

InvalidCall = tuple[str, Callable[[], object], type[Exception], str]


def _replace_simplex_index(function_name: str, value: int) -> object:
    simplices = np.asarray([[value, 1, 2, 3]], dtype=np.int64)
    if function_name == "build_demag_fem_geometry":
        return nmag_accel.build_demag_fem_geometry(POINTS, simplices)
    if function_name == "build_demag_fem_divergence":
        return nmag_accel.build_demag_fem_divergence(
            simplices,
            np.zeros((1, 4, 3)),
            np.ones(1),
            POINTS,
            np.ones(1),
            4,
        )
    if function_name == "recover_demag_nodal_field":
        return nmag_accel.recover_demag_nodal_field(
            simplices,
            np.ones(1),
            np.ones((1, 3)),
            np.ones(4),
            4,
        )
    if function_name == "build_oriented_boundary_faces":
        return nmag_accel.build_oriented_boundary_faces(POINTS, simplices)
    if function_name == "build_probe_tetrahedral_geometry":
        return nmag_accel.build_probe_tetrahedral_geometry(POINTS, simplices)
    if function_name == "maxangle_between_edges":
        return nmag_accel.maxangle_between_edges(POINTS, simplices)
    if function_name == "build_lindholm_bem_matrix":
        return nmag_accel.build_lindholm_bem_matrix(
            POINTS,
            simplices,
            FACE_NODES,
            BOUNDARY_NODES,
            LOCAL_INDEX_BY_POINT,
        )
    raise AssertionError(f"unhandled function {function_name}")
