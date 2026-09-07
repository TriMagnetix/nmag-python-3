from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np

import nmag.demag as demag
from nmag.backends import DEMAG_LINEAR_SOLVER_BACKEND_ENV


def _record_timings() -> tuple[dict[str, float], Callable[[str, float], None]]:
    timings: dict[str, float] = {}

    def record(name: str, seconds: float) -> None:
        timings[name] = timings.get(name, 0.0) + seconds

    return timings, record


def test_gauge_solver_recovers_from_numpy_solve_failure(monkeypatch):
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "numpy")
    timings, record_timing = _record_timings()

    def fail_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(demag.np.linalg, "solve", fail_solve)

    solution = demag._solve_gauge_fixed(
        np.asarray([[2.0, 0.0], [0.0, 2.0]]),
        np.asarray([2.0, -2.0]),
        record_timing=record_timing,
        timing_prefix="gauge",
    )

    np.testing.assert_allclose(solution, [1.0, -1.0])
    assert timings["gauge:setup"] >= 0.0
    assert timings["gauge:numpy_solve_failed"] >= 0.0
    assert timings["gauge:numpy_lstsq"] >= 0.0


def test_scipy_solver_recovers_from_singular_system(monkeypatch):
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "scipy")
    timings, record_timing = _record_timings()

    def fail_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("singular")

    def least_squares(matrix, rhs, **_kwargs):
        return np.linalg.lstsq(matrix, rhs, rcond=None)

    fake_linalg = SimpleNamespace(solve=fail_solve, lstsq=least_squares)
    monkeypatch.setattr(demag, "_scipy_linalg", lambda: fake_linalg)

    solution = demag._solve_linear_system(
        np.asarray([[1.0, 1.0], [2.0, 2.0]]),
        np.asarray([2.0, 4.0]),
        record_timing=record_timing,
        timing_prefix="linear",
    )

    np.testing.assert_allclose(solution, [1.0, 1.0])
    assert timings["linear:scipy_import"] >= 0.0
    assert timings["linear:scipy_solve_failed"] >= 0.0
    assert timings["linear:scipy_lstsq"] >= 0.0


def test_scipy_gauge_solver_recovers_from_solve_failure(monkeypatch):
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "scipy")
    timings, record_timing = _record_timings()

    def fail_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced failure")

    def least_squares(matrix, rhs, **_kwargs):
        return np.linalg.lstsq(matrix, rhs, rcond=None)

    fake_linalg = SimpleNamespace(solve=fail_solve, lstsq=least_squares)
    monkeypatch.setattr(demag, "_scipy_linalg", lambda: fake_linalg)

    solution = demag._solve_gauge_fixed(
        np.asarray([[2.0, 0.0], [0.0, 2.0]]),
        np.asarray([2.0, -2.0]),
        record_timing=record_timing,
        timing_prefix="gauge",
    )

    np.testing.assert_allclose(solution, [1.0, -1.0])
    assert timings["gauge:setup"] >= 0.0
    assert timings["gauge:scipy_import"] >= 0.0
    assert timings["gauge:scipy_solve_failed"] >= 0.0
    assert timings["gauge:scipy_lstsq"] >= 0.0


def test_numpy_solver_records_least_squares_recovery(monkeypatch):
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "numpy")
    timings, record_timing = _record_timings()

    solution = demag._solve_linear_system(
        np.asarray([[1.0, 1.0], [2.0, 2.0]]),
        np.asarray([2.0, 4.0]),
        record_timing=record_timing,
        timing_prefix="linear",
    )

    np.testing.assert_allclose(solution, [1.0, 1.0])
    assert timings["linear:numpy_solve_failed"] >= 0.0
    assert timings["linear:numpy_lstsq"] >= 0.0


def test_lindholm_python_contributions_handle_degenerate_and_reversed_faces():
    observer = np.asarray([0.2, 0.3, 1.0])
    p0 = np.asarray([0.0, 0.0, 0.0])
    p1 = np.asarray([1.0, 0.0, 0.0])
    p2 = np.asarray([0.0, 1.0, 0.0])

    contributions = demag._lindholm_triangle_contributions(
        observer,
        p0,
        p1,
        p2,
        np.asarray([0.0, 0.0, 1.0]),
    )
    reversed_contributions = demag._lindholm_triangle_contributions(
        observer,
        p0,
        p1,
        p2,
        np.asarray([0.0, 0.0, -1.0]),
    )
    degenerate_contributions = demag._lindholm_triangle_contributions(
        observer,
        p0,
        p1,
        np.asarray([2.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    )

    assert np.all(np.isfinite(contributions))
    assert not np.allclose(contributions, 0.0)
    np.testing.assert_allclose(reversed_contributions, -contributions)
    np.testing.assert_array_equal(degenerate_contributions, np.zeros(3))


def test_boundary_faces_are_outward_for_reversed_tetrahedron_orientation():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    simplices = np.asarray([[0, 2, 1, 3]], dtype=int)

    boundary_faces = demag._oriented_boundary_faces(points, simplices)

    assert len(boundary_faces) == 4
    cell_center = np.mean(points[simplices[0]], axis=0)
    for cell_index, face in boundary_faces:
        assert cell_index == 0
        face_points = points[np.asarray(face)]
        normal = np.cross(face_points[1] - face_points[0], face_points[2] - face_points[0])
        face_center = np.mean(face_points, axis=0)
        assert float(np.dot(normal, cell_center - face_center)) < 0.0


def test_barycentric_coordinates_reject_singular_tetrahedron():
    tetrahedron = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    assert demag._tetrahedral_barycentric_coordinates(
        tetrahedron,
        np.asarray([0.25, 0.25, 0.0]),
    ) is None
