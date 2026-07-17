from __future__ import annotations

from pathlib import Path

import numpy as np
import public_api_support_module as helpers
import pytest

import nmag
from nmag.simulation import (
    _oriented_boundary_faces,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"


def test_probe_geometry_rust_backend_matches_python(tmp_path, monkeypatch):
    pytest.importorskip("nmag_accel")
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_tetra_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp")
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    sim.set_m([1.0, 0.0, 0.0])
    nodal_values = np.asarray(sim.get_subfield("H_demag"))

    python_geometry = sim._probe_tetrahedral_geometry_python()
    rust_geometry = sim._probe_tetrahedral_geometry_rust()

    np.testing.assert_array_equal(rust_geometry[0], python_geometry[0])
    for rust_array, python_array in zip(rust_geometry[1:], python_geometry[1:], strict=True):
        np.testing.assert_allclose(rust_array, python_array, rtol=1e-12, atol=1e-14)

    sim._probe_geometry_cache_token = None
    sim._probe_tetrahedral_cache = None
    probe = np.asarray([0.5e-9, 0.0, 0.0])
    optimized = sim._probe_tetrahedral_field(probe, nodal_values)
    reference = sim._probe_tetrahedral_field_reference(probe, nodal_values)
    np.testing.assert_allclose(optimized, reference, rtol=0.0, atol=1.0e-9)
    assert sim.last_probe_timings_seconds["tetrahedral_geometry_build:rust"] >= 0.0


def test_demag_boundary_face_rust_backend_matches_python(tmp_path, monkeypatch):
    rust_accel = pytest.importorskip("nmag_accel")
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_tetra_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp", config=nmag.NmagConfig(accelerator="rust"))
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))

    points = np.asarray(sim.mesh.points, dtype=float)
    simplices = np.asarray(sim.mesh.simplices, dtype=int)
    python_faces = _oriented_boundary_faces(points, simplices)
    rust_faces = sim._boundary_faces_for_demag_mesh_rust(points, simplices)

    assert rust_faces == python_faces

    raw_owners, raw_faces = rust_accel.build_oriented_boundary_faces(
        np.asarray(points, dtype=np.float64),
        np.asarray(simplices, dtype=np.int64),
    )
    assert raw_owners.dtype == np.int64
    assert raw_faces.dtype == np.int64
    assert np.asarray(raw_faces).shape == (len(python_faces), 3)
    assert [
        (int(owner), tuple(int(point_index) for point_index in face))
        for owner, face in zip(np.asarray(raw_owners), np.asarray(raw_faces), strict=True)
    ] == python_faces

    selected_faces = sim._boundary_faces_for_demag_mesh(points, simplices)

    assert selected_faces == python_faces
    assert sim._demag_boundary_faces_cache == python_faces


def test_demag_probe_returns_none_outside_tetrahedral_domain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_single_region_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp", config=nmag.NmagConfig(accelerator="rust"))
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    sim.set_m([1.0, 0.0, 0.0])

    assert sim.probe_subfield_siv("H_demag", [0.5e-9, 0.5e-9, 0.5e-9]) is None


def test_demag_nodal_average_preserves_legacy_volume_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_tetra_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp")
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    sim.set_m([1.0, 0.0, 0.0])

    h_demag = np.asarray(sim.get_subfield("H_demag"))
    expected_legacy = np.asarray([
        [-116434.9609375, -22579.560546875, -22579.560546875],
        [-93772.078125, 83.321533203125, 83.321533203125],
        [-93772.078125, 83.321533203125, 83.321533203125],
        [-93772.078125, 83.321533203125, 83.321533203125],
        [-82440.640625, 11414.7626953125, 11414.7626953125],
    ])

    assert h_demag.shape == (5, 3)
    np.testing.assert_allclose(h_demag, expected_legacy, rtol=1e-8, atol=5e-3)
    assert sim._incident_cell_volume_sums_cache is not None
    first_cache = sim._incident_cell_volume_sums_cache

    cached_weights = sim._incident_cell_volume_sums(
        sim._mesh_points(),
        np.asarray(sim.mesh.simplices, dtype=int),
        sim._demag_volumes_cache,
    )

    assert sim._incident_cell_volume_sums_cache is first_cache
    assert cached_weights is first_cache[3]

    sim._invalidate_demag(clear_geometry=True)
    assert sim._incident_cell_volume_sums_cache is None


def test_demag_nodal_recovery_rust_backend_matches_python(tmp_path, monkeypatch):
    rust_accel = pytest.importorskip("nmag_accel")
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_tetra_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp", config=nmag.NmagConfig(accelerator="rust"))
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    sim.set_m([1.0, 0.0, 0.0])

    points = sim._mesh_points()
    simplices = np.asarray(sim.mesh.simplices, dtype=int)
    phi, _rho, volumes = sim._get_demag_auxiliary_fields()
    cell_h = sim._cell_demag_from_potential(points, simplices, phi)
    weights = sim._incident_cell_volume_sums(points, simplices, volumes)
    reference = sim._recover_demag_nodal_field_python(
        simplices,
        volumes,
        cell_h,
        weights,
        len(points),
    )

    raw_rust = rust_accel.recover_demag_nodal_field(
        np.asarray(simplices, dtype=np.int64),
        np.asarray(volumes, dtype=np.float64),
        np.asarray(cell_h, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        len(points),
    )
    assert isinstance(raw_rust, np.ndarray)
    assert raw_rust.dtype == np.float64
    np.testing.assert_allclose(raw_rust, reference, rtol=1e-12, atol=1e-9)

    def fail_python_recovery(*_args, **_kwargs):
        raise AssertionError("forced rust nodal recovery should not use Python helper")

    sim._invalidate_demag(clear_geometry=False)
    monkeypatch.setattr(sim, "_recover_demag_nodal_field_python", fail_python_recovery)
    rust_selected = np.asarray(sim.get_subfield("H_demag"))

    np.testing.assert_allclose(rust_selected, reference, rtol=1e-12, atol=1e-9)


def test_demag_cell_average_rust_backend_matches_python(tmp_path, monkeypatch):
    pytest.importorskip("nmag_accel")
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_tetra_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp", config=nmag.NmagConfig(accelerator="rust"))
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    sim.set_m(lambda point: [1.0, 0.0, 0.0] if point[0] <= 5.0e-10 else [0.0, 1.0, 0.0])

    cell_h, volumes = sim._get_demag_cell_field()
    reference = sim._demag_cell_field_average_python(cell_h, volumes)
    raw_rust = sim._demag_cell_field_average_rust(cell_h, volumes)

    assert isinstance(raw_rust, np.ndarray)
    assert raw_rust.dtype == np.float64
    np.testing.assert_allclose(raw_rust, reference, rtol=1e-12, atol=1e-9)

    def fail_python_average(*_args, **_kwargs):
        raise AssertionError("forced rust cell average should not use Python helper")

    monkeypatch.setattr(sim, "_demag_cell_field_average_python", fail_python_average)
    selected_average = sim._demag_cell_field_average()

    np.testing.assert_allclose(selected_average, reference, rtol=1e-12, atol=1e-9)
