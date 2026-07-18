from __future__ import annotations

from pathlib import Path

import numpy as np
import public_api_support_module as helpers

import nmag
import nmag.simulation as nmag_sim
import nmesh
from nmag.simulation import (
    _oriented_boundary_faces,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"



def test_oriented_boundary_faces_matches_scalar_reference():
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        simplices_indices=[
            [0, 1, 3, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [0, 4, 5, 7],
            [0, 5, 1, 7],
        ],
        simplices_regions=[1] * 6,
    )
    points = np.asarray(mesh.points, dtype=float)
    simplices = np.asarray(mesh.simplices, dtype=int)

    assert _oriented_boundary_faces(points, simplices) == helpers.reference_oriented_boundary_faces(
        points,
        simplices,
    )


def test_demag_probe_uses_point_dipole_far_field(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_single_region_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(6.0, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp")
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1, "m"))
    sim.set_m([1.0, 0.0, 0.0])

    assert sim.probe_subfield_siv("H_demag", [1000.25, 0.25, 0.25]) is None

    probe = np.asarray(sim._probe_demag_at([1000.25, 0.25, 0.25]))
    expected = np.asarray([1.0 / (2.0 * np.pi * 1000.0**3), 0.0, 0.0])

    np.testing.assert_allclose(probe, expected, rtol=1e-5, atol=1e-12)


def test_demag_spatial_field_moves_toward_legacy_single_tetra_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_single_region_mesh(mesh_path)

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
        -68710.02865918343,
        -33347.84566311808,
        -33347.84566311808,
    ])

    assert h_demag.shape == (4, 3)
    np.testing.assert_allclose(h_demag, np.tile(h_demag[0], (4, 1)), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(h_demag[0], expected_legacy, rtol=1e-8, atol=1e-5)


def test_demag_saved_rho_uses_legacy_simulation_volume_normalization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_single_region_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp")
    sim.load_mesh(
        str(mesh_path),
        [("generated", material)],
        unit_length=nmag.SI(10e-9, "m"),
    )
    sim.set_m([1.0, 0.0, 0.0])

    _phi, rho, _volumes = sim._get_demag_auxiliary_fields()
    points = sim._mesh_points()
    simplices = np.asarray(sim.mesh.simplices, dtype=int)
    ms_values = sim._simplex_material_ms_values(sim.mesh.regions)
    _stiffness, weak_divergence, _volumes = sim._assemble_demag_fem_system(
        points,
        simplices,
        np.asarray(sim._fields["m"], dtype=float),
        ms_values,
        np.ones(len(simplices), dtype=float),
    )

    np.testing.assert_allclose(rho, weak_divergence / 1.0e-27)


def test_demag_probe_preserves_legacy_null_first_vertex_behavior(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_single_region_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp")
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    sim.set_m([1.0, 0.0, 0.0])

    assert sim.probe_subfield_siv("H_demag", [0.0, 0.0, 0.0]) is None
    assert sim._demag_boundary_faces_cache is not None

    def fail_boundary_face_rebuild(_points, _simplices):
        raise AssertionError("exact vertex probe should reuse cached boundary faces")

    monkeypatch.setattr(nmag_sim, "_oriented_boundary_faces", fail_boundary_face_rebuild)

    assert sim.probe_subfield_siv("H_demag", [0.0, 0.0, 0.0]) is None
    assert sim.probe_subfield_siv("H_demag", [1.0e-12, 1.0e-12, 1.0e-12]) is not None
    assert sim.probe_subfield_siv("H_demag", [1.0e-9, 0.0, 0.0]) is not None


def test_doc_sphere1_demag_at_origin_matches_analytic_sphere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="sphere1")
    sim.load_mesh(
        str(SPHERE1_MESH),
        [("sphere", material)],
        unit_length=nmag.SI(1e-9, "m"),
    )
    sim.set_m([1.0, 0.0, 0.0])
    sim.set_H_ext([0.0, 0.0, 0.0], nmag.SI("A/m"))
    sim.save_data(fields="all")

    h_demag = sim.probe_subfield_siv("H_demag", [0.0, 0.0, 0.0])
    assert h_demag is not None
    assert abs(h_demag[0] / (-material.Ms.value / 3.0) - 1.0) < 0.01
    np.testing.assert_allclose(
        sim.get_subfield_average("m"),
        [1.0, 0.0, 0.0],
        rtol=0.0,
        atol=1.0e-14,
    )


def test_demag_probe_interpolates_inside_tetrahedron(tmp_path, monkeypatch):
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

    expected_legacy = np.asarray([
        -105103.52119991671,
        -11248.119506505756,
        -11248.119506505815,
    ])

    probe = np.asarray(sim.probe_subfield_siv("H_demag", [0.5e-9, 0.0, 0.0]))
    np.testing.assert_allclose(probe, expected_legacy, rtol=1e-8, atol=5e-3)


def test_demag_probe_reuses_tetrahedral_interpolation_geometry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_tetra_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp", config=nmag.NmagConfig(accelerator="off"))
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    sim.set_m([1.0, 0.0, 0.0])
    sim.get_subfield("H_demag")
    sim.reset_probe_timings()

    inversions = []
    original_inv = nmag_sim.np.linalg.inv

    def counting_inv(matrix):
        inversions.append(matrix.copy())
        return original_inv(matrix)

    monkeypatch.setattr(nmag_sim.np.linalg, "inv", counting_inv)

    first_probe = sim.probe_subfield_siv("H_demag", [0.5e-9, 0.0, 0.0])
    second_probe = sim.probe_subfield_siv("H_demag", [0.25e-9, 0.0, 0.0])

    assert first_probe is not None
    assert second_probe is not None
    assert len(inversions) == 2
    assert sim._probe_tetrahedral_cache is not None
    assert sim.last_probe_timings_seconds["total"] >= 0.0
    assert sim.last_probe_timings_seconds["bounds_check"] >= 0.0
    assert sim.last_probe_timings_seconds["demag_nodal_field"] >= 0.0
    assert sim.last_probe_timings_seconds["tetrahedral_interpolation"] >= 0.0
    assert sim.last_probe_timings_seconds["tetrahedral_geometry"] >= 0.0
    assert sim.last_probe_timings_seconds["tetrahedral_geometry_build"] >= 0.0
    assert sim.last_probe_timings_seconds["tetrahedral_geometry_cache_hit"] >= 0.0
    assert sim.last_probe_timings_seconds["tetrahedral_candidate_search"] >= 0.0
    assert sim.last_probe_timings_seconds["tetrahedral_barycentric"] >= 0.0
    assert sim.last_probe_timings_seconds["tetrahedral_value_interpolation"] >= 0.0
    sim.reset_probe_timings()
    assert sim.last_probe_timings_seconds == {}


def test_demag_probe_cached_path_matches_scalar_reference(tmp_path, monkeypatch):
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

    probes = [
        [0.25e-9, 0.25e-9, 0.25e-9],
        [0.5e-9, 0.0, 0.0],
        [0.75e-9, 0.1e-9, 0.1e-9],
        [1.0e-9, 0.0, 0.0],
        [1.1e-9, 0.0, 0.0],
    ]

    for probe in probes:
        probe_array = np.asarray(probe, dtype=float)
        optimized = sim._probe_tetrahedral_field(probe_array, nodal_values)
        reference = sim._probe_tetrahedral_field_reference(probe_array, nodal_values)
        if reference is None:
            assert optimized is None
        else:
            np.testing.assert_allclose(optimized, reference, rtol=0.0, atol=1.0e-9)
