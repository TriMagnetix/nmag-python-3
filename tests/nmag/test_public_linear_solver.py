from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import public_api_support_module as helpers
import pytest

import nmag
from nmag.simulation import (
    DEMAG_LINEAR_SOLVER_AUTO_SCIPY_MIN_SIZE_ENV,
    DEMAG_LINEAR_SOLVER_BACKEND_ENV,
    _selected_demag_linear_solver_backend,
    _solve_gauge_fixed,
    _solve_linear_system,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"

def test_demag_linear_solver_auto_uses_numpy_without_loading_scipy():
    env = helpers.subprocess_env_with_repo_src()
    env.pop(DEMAG_LINEAR_SOLVER_BACKEND_ENV, None)
    env.pop(DEMAG_LINEAR_SOLVER_AUTO_SCIPY_MIN_SIZE_ENV, None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "import sys; "
                "import numpy as np; "
                "import nmag.simulation as sim; "
                "matrix = np.asarray([[2.0, 0.0], [0.0, 4.0]]); "
                "rhs = np.asarray([2.0, 8.0]); "
                "gauge_matrix = np.asarray([[2.0, 0.0], [0.0, 2.0]]); "
                "gauge_rhs = np.asarray([2.0, -2.0]); "
                "print(json.dumps({"
                "'backend': sim._selected_demag_linear_solver_backend(), "
                "'linear': sim._solve_linear_system(matrix, rhs).tolist(), "
                "'gauge': sim._solve_gauge_fixed(gauge_matrix, gauge_rhs).tolist(), "
                "'scipy_cached': sim._SCIPY_LINALG is not None, "
                "'scipy_loaded': 'scipy.linalg' in sys.modules"
                "}))"
            ),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "backend": "numpy",
        "linear": [1.0, 2.0],
        "gauge": [1.0, -1.0],
        "scipy_cached": False,
        "scipy_loaded": False,
    }


def test_demag_linear_solver_backend_env_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "definitely-not-a-backend")

    with pytest.raises(ValueError, match=DEMAG_LINEAR_SOLVER_BACKEND_ENV):
        _selected_demag_linear_solver_backend()


def test_demag_linear_solver_scipy_backend_matches_numpy(monkeypatch):
    matrix = np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=float)
    rhs = np.asarray([9.0, 8.0], dtype=float)
    gauge_matrix = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=float)
    gauge_rhs = np.asarray([2.0, -2.0], dtype=float)

    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "numpy")
    numpy_solution = _solve_linear_system(matrix, rhs)
    numpy_gauge = _solve_gauge_fixed(gauge_matrix, gauge_rhs)

    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "scipy")
    scipy_solution = _solve_linear_system(matrix, rhs)
    scipy_gauge = _solve_gauge_fixed(gauge_matrix, gauge_rhs)

    assert _selected_demag_linear_solver_backend() == "scipy"
    np.testing.assert_allclose(scipy_solution, numpy_solution, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(scipy_gauge, numpy_gauge, rtol=1.0e-12, atol=1.0e-12)


def test_demag_linear_solver_auto_threshold_reaches_solver_calls(monkeypatch):
    monkeypatch.delenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, raising=False)
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_AUTO_SCIPY_MIN_SIZE_ENV, "2")
    timings = {}

    def record_timing(name, seconds):
        timings[name] = timings.get(name, 0.0) + seconds

    solution = _solve_linear_system(
        np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=float),
        np.asarray([9.0, 8.0], dtype=float),
        record_timing=record_timing,
        timing_prefix="linear",
    )

    np.testing.assert_allclose(solution, [2.0, 3.0])
    assert "linear:scipy_import" in timings
    assert "linear:scipy_solve" in timings
    assert "linear:numpy_solve" not in timings


def test_demag_linear_solver_records_numpy_timing_subphases(monkeypatch):
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "numpy")
    timings = {}

    def record_timing(name, seconds):
        timings[name] = timings.get(name, 0.0) + seconds

    solution = _solve_linear_system(
        np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=float),
        np.asarray([9.0, 8.0], dtype=float),
        record_timing=record_timing,
        timing_prefix="linear",
    )
    gauge_solution = _solve_gauge_fixed(
        np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=float),
        np.asarray([2.0, -2.0], dtype=float),
        record_timing=record_timing,
        timing_prefix="gauge",
    )

    np.testing.assert_allclose(solution, [2.0, 3.0])
    np.testing.assert_allclose(gauge_solution, [1.0, -1.0])
    assert timings["linear:numpy_solve"] >= 0.0
    assert timings["gauge:setup"] >= 0.0
    assert timings["gauge:numpy_solve"] >= 0.0


def test_demag_linear_solver_records_scipy_import_timing(monkeypatch):
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "scipy")
    timings = {}

    def record_timing(name, seconds):
        timings[name] = timings.get(name, 0.0) + seconds

    solution = _solve_linear_system(
        np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=float),
        np.asarray([9.0, 8.0], dtype=float),
        record_timing=record_timing,
        timing_prefix="linear",
    )
    gauge_solution = _solve_gauge_fixed(
        np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=float),
        np.asarray([2.0, -2.0], dtype=float),
        record_timing=record_timing,
        timing_prefix="gauge",
    )

    np.testing.assert_allclose(solution, [2.0, 3.0])
    np.testing.assert_allclose(gauge_solution, [1.0, -1.0])
    assert timings["linear:scipy_import"] >= 0.0
    assert timings["linear:scipy_solve"] >= 0.0
    assert timings["gauge:setup"] >= 0.0
    assert timings["gauge:scipy_import"] >= 0.0
    assert timings["gauge:scipy_solve"] >= 0.0


def test_demag_linear_solver_rejects_inaccurate_numpy_fallback(monkeypatch):
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "numpy")
    timings = {}

    def record_timing(name, seconds):
        timings[name] = timings.get(name, 0.0) + seconds

    with pytest.raises(np.linalg.LinAlgError, match="relative residual"):
        _solve_linear_system(
            np.zeros((2, 2), dtype=float),
            np.ones(2, dtype=float),
            record_timing=record_timing,
            timing_prefix="linear",
        )

    assert timings["linear:numpy_solve_failed"] >= 0.0


def test_demag_linear_solver_accepts_consistent_singular_system(monkeypatch, caplog):
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "numpy")
    solution = _solve_linear_system(
        np.asarray([[1.0, 1.0], [2.0, 2.0]]),
        np.asarray([2.0, 4.0]),
    )

    np.testing.assert_allclose(solution, [1.0, 1.0])
    assert "least-squares solution" in caplog.text


def test_dirichlet_extension_records_solver_subphases(monkeypatch):
    monkeypatch.setenv(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "numpy")
    sim = nmag.Simulation(name="mvp")
    timings = {}
    previous_timings = sim._active_subfield_array_timings
    sim._active_subfield_array_timings = timings
    try:
        phi = sim._dirichlet_extension_from_boundary(
            np.asarray(
                [
                    [2.0, -1.0, 0.0],
                    [-1.0, 2.0, -1.0],
                    [0.0, -1.0, 2.0],
                ],
                dtype=float,
            ),
            np.asarray([0, 2], dtype=int),
            np.asarray([1.0, 3.0], dtype=float),
        )
    finally:
        sim._active_subfield_array_timings = previous_timings

    np.testing.assert_allclose(phi, [1.0, 2.0, 3.0])
    assert timings["demag_auxiliary:dirichlet_extension:index"] >= 0.0
    assert timings["demag_auxiliary:dirichlet_extension:slice"] >= 0.0
    assert timings["demag_auxiliary:dirichlet_extension:rhs"] >= 0.0
    assert timings["demag_auxiliary:dirichlet_extension:numpy_solve"] >= 0.0


