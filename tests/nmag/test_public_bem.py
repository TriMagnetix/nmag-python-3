from __future__ import annotations

import numpy as np
import public_api_support_module as helpers
import pytest

import nmag
import nmag.backends as nmag_backends
from nmag.backends import _selected_lindholm_bem_backend


def _simulation(tmp_path, *, accelerator: str = "off") -> nmag.Simulation:
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_tetra_mesh(mesh_path)
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    simulation = nmag.Simulation(
        name=f"bem-{accelerator}",
        config=nmag.NmagConfig(accelerator=accelerator),
    )
    simulation.load_mesh(
        str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m")
    )
    return simulation


def test_lindholm_bem_matrix_matches_legacy_scalar_formula(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = _simulation(tmp_path)
    points = np.asarray(sim.mesh.points, dtype=float)
    simplices = np.asarray(sim.mesh.simplices, dtype=int)
    boundary_faces = sim._boundary_faces_for_demag_mesh(points, simplices)
    fast_nodes, fast_bem = sim._build_lindholm_bem_matrix_numba(points, simplices, boundary_faces)
    reference_nodes, reference_bem = sim._build_lindholm_bem_matrix_python(
        points, simplices, boundary_faces
    )

    np.testing.assert_array_equal(fast_nodes, reference_nodes)
    np.testing.assert_allclose(fast_bem, reference_bem, rtol=1e-12, atol=1e-14)


def test_lindholm_bem_policy_auto_and_off(monkeypatch):
    monkeypatch.setattr(nmag_backends, "_rust_accelerator_available", lambda: True)
    assert _selected_lindholm_bem_backend(nmag.NmagConfig()) == "rust"
    assert _selected_lindholm_bem_backend(nmag.NmagConfig(accelerator="off")) == "numba"


def test_lindholm_bem_rust_matches_python(tmp_path, monkeypatch):
    pytest.importorskip("nmag_accel")
    monkeypatch.chdir(tmp_path)
    sim = _simulation(tmp_path, accelerator="rust")
    points = np.asarray(sim.mesh.points, dtype=float)
    simplices = np.asarray(sim.mesh.simplices, dtype=int)
    boundary_faces = sim._boundary_faces_for_demag_mesh(points, simplices)
    rust_nodes, rust_bem = sim._build_lindholm_bem_matrix(points, simplices, boundary_faces)
    python_nodes, python_bem = sim._build_lindholm_bem_matrix_python(
        points, simplices, boundary_faces
    )

    np.testing.assert_array_equal(rust_nodes, python_nodes)
    np.testing.assert_allclose(rust_bem, python_bem, rtol=1e-12, atol=1e-14)
