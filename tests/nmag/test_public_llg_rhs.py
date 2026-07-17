from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nmag
import nmag.simulation as nmag_sim

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"

def test_llg_rhs_python_matches_nested_cross_reference():
    sim = nmag.Simulation(name="mvp")
    m = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.8, 0.6],
            [0.2, -0.3, 0.9327379053088815],
        ],
        dtype=float,
    )
    h_total = np.asarray(
        [
            [0.0, 1.0e5, -2.0e5],
            [3.0e5, -4.0e4, 2.0e4],
            [-1.0e5, 7.0e4, 8.0e4],
        ],
        dtype=float,
    )
    pin = np.asarray([1.0, 0.25, 0.0], dtype=float)
    ms_values = np.asarray([8.0e5, 6.5e5, 7.1e5], dtype=float)
    c1 = -2.211e5
    c2 = -1.1055e4
    c3 = 1.0e11

    optimized = sim._llg_rhs_python(m, h_total, pin, ms_values, c1, c2, c3)
    reference = ms_values[:, np.newaxis] * (
        c1 * np.cross(m, h_total)
        + c2 * np.cross(m, np.cross(m, h_total))
        + c3 * (1.0 - np.einsum("ij,ij->i", m, m))[:, np.newaxis] * m
    ) * pin[:, np.newaxis]

    np.testing.assert_allclose(optimized, reference, rtol=1.0e-15, atol=1.0e-6)


def test_llg_rhs_rust_wrapper_delegates_to_accelerator(monkeypatch):
    sim = nmag.Simulation(name="mvp")
    m = np.asarray([[1.0, 0.0, 0.0]], dtype=float)
    h_total = np.asarray([[0.0, 2.0, 0.0]], dtype=float)
    pin = np.asarray([1.0], dtype=float)
    ms_values = np.asarray([8.0e5], dtype=float)

    class FakeRustAccelerator:
        def __init__(self):
            self.calls = []

        def llg_rhs(self, *args):
            self.calls.append(args)
            return np.asarray([[1.0, 2.0, 3.0]], dtype=float)

    fake = FakeRustAccelerator()
    monkeypatch.setattr(nmag_sim, "_load_rust_accelerator", lambda required_by: fake)

    result = sim._llg_rhs_rust(m, h_total, pin, ms_values, -1.0, -0.5, 0.25)

    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]])
    assert len(fake.calls) == 1
    assert fake.calls[0][4:] == (-1.0, -0.5, 0.25)


def test_llg_rhs_rust_kernel_matches_python_reference_when_available():
    rust_accel = pytest.importorskip("nmag_accel")
    if not hasattr(rust_accel, "llg_rhs"):
        pytest.skip("installed nmag_accel extension does not expose llg_rhs")

    sim = nmag.Simulation(name="mvp")
    m = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.8, 0.6],
            [0.2, -0.3, 0.9327379053088815],
        ],
        dtype=float,
    )
    h_total = np.asarray(
        [
            [0.0, 1.0e5, -2.0e5],
            [3.0e5, -4.0e4, 2.0e4],
            [-1.0e5, 7.0e4, 8.0e4],
        ],
        dtype=float,
    )
    pin = np.asarray([1.0, 0.25, 0.0], dtype=float)
    ms_values = np.asarray([8.0e5, 6.5e5, 7.1e5], dtype=float)
    c1 = -2.211e5
    c2 = -1.1055e4
    c3 = 1.0e11

    reference = sim._llg_rhs_python(m, h_total, pin, ms_values, c1, c2, c3)
    accelerated = np.asarray(
        rust_accel.llg_rhs(m, h_total, pin, ms_values, c1, c2, c3),
        dtype=float,
    )

    np.testing.assert_allclose(accelerated, reference, rtol=1.0e-15, atol=1.0e-6)


