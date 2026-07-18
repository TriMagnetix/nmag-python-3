from __future__ import annotations

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


def test_public_api_surface() -> None:
    expected_functions = {
        "build_demag_fem_divergence",
        "build_demag_fem_geometry",
        "build_lindholm_bem_matrix",
        "apply_lindholm_bem_matrix_free",
        "build_oriented_boundary_faces",
        "build_probe_tetrahedral_geometry",
        "demag_cell_field_average",
        "llg_rhs",
        "llg_rhs_heterogeneous",
        "llg_rhs_stt_heterogeneous",
        "parallel_runtime_info",
        "integrate_llg_bdf",
        "maxangle_between_edges",
        "recover_demag_nodal_field",
    }

    assert nmag_accel.API_VERSION == 2
    assert all(callable(getattr(nmag_accel, name)) for name in expected_functions)


def test_mesh_geometry_functions_accept_a_unit_tetrahedron() -> None:
    stiffness, gradients, volumes = nmag_accel.build_demag_fem_geometry(POINTS, SIMPLICES)
    np.testing.assert_allclose(volumes, [1.0 / 6.0])
    np.testing.assert_allclose(np.sum(gradients[0], axis=0), np.zeros(3), atol=1.0e-15)
    np.testing.assert_allclose(stiffness, stiffness.T)

    owners, faces = nmag_accel.build_oriented_boundary_faces(POINTS, SIMPLICES)
    np.testing.assert_array_equal(owners, np.zeros(4, dtype=np.int64))
    np.testing.assert_array_equal(faces, FACE_NODES)

    probe_geometry = nmag_accel.build_probe_tetrahedral_geometry(POINTS, SIMPLICES)
    valid_simplices, origins, inverse_matrices, lower, upper = probe_geometry
    np.testing.assert_array_equal(valid_simplices, SIMPLICES)
    np.testing.assert_allclose(origins, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(inverse_matrices, [np.eye(3)])
    np.testing.assert_allclose(lower, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(upper, [[1.0, 1.0, 1.0]])


def test_fem_field_functions_accept_simple_inputs() -> None:
    _, gradients, volumes = nmag_accel.build_demag_fem_geometry(POINTS, SIMPLICES)
    m = np.tile([1.0, 0.0, 0.0], (4, 1))
    divergence = nmag_accel.build_demag_fem_divergence(
        SIMPLICES,
        gradients,
        volumes,
        m,
        np.asarray([2.0]),
        4,
    )
    np.testing.assert_allclose(divergence, [-1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0])

    cell_h = np.asarray([[2.0, -3.0, 5.0]])
    nodal_h = nmag_accel.recover_demag_nodal_field(
        SIMPLICES,
        volumes,
        cell_h,
        np.full(4, volumes[0]),
        4,
    )
    np.testing.assert_allclose(nodal_h, np.tile(cell_h, (4, 1)))

    average = nmag_accel.demag_cell_field_average(
        np.asarray([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]]),
        np.asarray([1.0, 3.0]),
    )
    np.testing.assert_allclose(average, [4.0, 5.0, 6.0])


def test_llg_maxangle_and_lindholm_accept_simple_inputs() -> None:
    m = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    h_total = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rhs = nmag_accel.llg_rhs(
        m,
        h_total,
        np.ones(2),
        np.ones(2),
        1.0,
        0.0,
        0.0,
    )
    np.testing.assert_allclose(rhs, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    assert nmag_accel.maxangle_between_edges(m, np.asarray([[0, 1]])) == pytest.approx(90.0)

    bem = nmag_accel.build_lindholm_bem_matrix(
        POINTS,
        SIMPLICES,
        FACE_NODES,
        BOUNDARY_NODES,
        LOCAL_INDEX_BY_POINT,
    )
    assert bem.shape == (4, 4)
    assert np.all(np.isfinite(bem))


def test_diffsol_bdf_keeps_a_zero_torque_state_fixed() -> None:
    result = nmag_accel.integrate_llg_bdf(
        np.asarray([1.0, 0.0, 0.0]),
        np.zeros((3, 3)),
        np.asarray([1.0e5, 0.0, 0.0]),
        np.ones(1),
        -1.0,
        -0.5,
        0.0,
        0.0,
        1.0,
        1.0e-6,
        1.0e-6,
        1.0e-15,
        1.0e-3,
        1.0,
        5,
        2,
        1000,
    )

    np.testing.assert_array_equal(result.state, [1.0, 0.0, 0.0])
    assert result.converged is True
    assert result.accepted_steps == 10
    assert result.max_dm_dt == 0.0
