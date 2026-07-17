from __future__ import annotations

import pytest
from rust_accelerator_validation_support import (
    BOUNDARY_NODES,
    FACE_NODES,
    LOCAL_INDEX_BY_POINT,
    POINTS,
    SIMPLICES,
    _replace_simplex_index,
    nmag_accel,
)


@pytest.mark.parametrize(
    "function_name",
    [
        "build_demag_fem_geometry",
        "build_demag_fem_divergence",
        "recover_demag_nodal_field",
        "build_oriented_boundary_faces",
        "build_probe_tetrahedral_geometry",
        "maxangle_between_edges",
        "build_lindholm_bem_matrix",
    ],
)
@pytest.mark.parametrize("index", [-1, 4], ids=["negative", "out-of-range"])
def test_indexed_functions_reject_invalid_simplex_indices(
    function_name: str,
    index: int,
) -> None:
    with pytest.raises(ValueError, match="simplex node"):
        _replace_simplex_index(function_name, index)


@pytest.mark.parametrize("index", [-1, 4], ids=["negative", "out-of-range"])
def test_lindholm_rejects_invalid_boundary_and_face_indices(index: int) -> None:
    boundary_nodes = BOUNDARY_NODES.copy()
    boundary_nodes[0] = index
    with pytest.raises(ValueError, match="boundary node"):
        nmag_accel.build_lindholm_bem_matrix(
            POINTS,
            SIMPLICES,
            FACE_NODES,
            boundary_nodes,
            LOCAL_INDEX_BY_POINT,
        )

    face_nodes = FACE_NODES.copy()
    face_nodes[0, 0] = index
    with pytest.raises(ValueError, match="face node"):
        nmag_accel.build_lindholm_bem_matrix(
            POINTS,
            SIMPLICES,
            face_nodes,
            BOUNDARY_NODES,
            LOCAL_INDEX_BY_POINT,
        )


@pytest.mark.parametrize("local_index", [-1, 4], ids=["negative", "out-of-range"])
def test_lindholm_rejects_invalid_local_boundary_indices(local_index: int) -> None:
    local_indices = LOCAL_INDEX_BY_POINT.copy()
    local_indices[1] = local_index
    with pytest.raises(ValueError, match="face node"):
        nmag_accel.build_lindholm_bem_matrix(
            POINTS,
            SIMPLICES,
            FACE_NODES,
            BOUNDARY_NODES,
            local_indices,
        )
