from __future__ import annotations

from pathlib import Path

import numpy as np
import public_api_support_module as helpers

import nmag
import nmag.simulation as nmag_sim
import nmesh

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"


def test_demag_cell_average_python_uses_volume_weighted_positive_cells():
    sim = nmag.Simulation(name="mvp")
    cell_h = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [5.0, 7.0, 11.0],
            [13.0, 17.0, 19.0],
        ],
        dtype=float,
    )
    volumes = np.asarray([2.0, 3.0, 5.0], dtype=float)

    expected = np.dot(volumes, cell_h) / float(np.sum(volumes))
    np.testing.assert_allclose(
        sim._demag_cell_field_average_python(cell_h, volumes),
        expected,
        rtol=1e-15,
        atol=0.0,
    )

    mixed_volumes = np.asarray([2.0, 0.0, 5.0], dtype=float)
    expected_mixed = (
        np.sum(cell_h[[0, 2]] * mixed_volumes[[0, 2], np.newaxis], axis=0)
        / float(np.sum(mixed_volumes[[0, 2]]))
    )
    np.testing.assert_allclose(
        sim._demag_cell_field_average_python(cell_h, mixed_volumes),
        expected_mixed,
        rtol=1e-15,
        atol=0.0,
    )


def test_demag_matches_legacy_two_tetra_nonuniform_magnetisation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_tetra_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp")
    sim.load_mesh(
        str(mesh_path),
        [("generated", material)],
        unit_length=nmag.SI(1e-9, "m"),
    )
    sim.set_m(
        lambda point: [1.0, 0.0, 0.0]
        if point[0] <= 5.0e-10
        else [0.0, 1.0, 0.0]
    )

    h_demag = np.asarray(sim.get_subfield("H_demag"))
    expected_legacy = np.asarray([
        [-77328.546875, -61685.9765625, -22579.560546875],
        [-54665.6640625, -39023.09765625, 83.321533203125],
        [-54665.6640625, -39023.09765625, 83.321533203125],
        [-54665.6640625, -39023.09765625, 83.321533203125],
        [-43334.22265625, -27691.654296875, 11414.7626953125],
    ])

    assert h_demag.shape == (5, 3)
    np.testing.assert_allclose(h_demag, expected_legacy, rtol=1e-8, atol=5e-3)


def test_demag_reuses_mesh_only_bem_cache_when_magnetisation_changes(
    tmp_path,
    monkeypatch,
):
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

    calls = []
    original_build_bem = sim._build_lindholm_bem_matrix

    def counting_build_bem(points, simplices, boundary_faces):
        calls.append(len(boundary_faces))
        return original_build_bem(points, simplices, boundary_faces)

    monkeypatch.setattr(sim, "_build_lindholm_bem_matrix", counting_build_bem)

    fem_builds = []
    original_fem_geometry = sim._demag_fem_geometry_for_mesh

    def counting_fem_geometry(points, simplices):
        if sim._demag_fem_geometry_cache is None:
            fem_builds.append(len(simplices))
        return original_fem_geometry(points, simplices)

    monkeypatch.setattr(sim, "_demag_fem_geometry_for_mesh", counting_fem_geometry)

    ms_builds = []
    original_simplex_material_ms = sim._simplex_material_ms

    def counting_simplex_material_ms(region_id):
        ms_builds.append(region_id)
        return original_simplex_material_ms(region_id)

    monkeypatch.setattr(sim, "_simplex_material_ms", counting_simplex_material_ms)

    first_h_demag = np.asarray(sim.get_subfield("H_demag"))
    sim.set_m([0.0, 1.0, 0.0])
    second_h_demag = np.asarray(sim.get_subfield("H_demag"))

    assert calls == [6]
    assert fem_builds == [2]
    assert ms_builds == [1]
    assert not np.allclose(first_h_demag, second_h_demag)


def test_simplex_material_ms_values_fast_path_for_uniform_regions(
    tmp_path,
    monkeypatch,
):
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

    def fail_unique(*args, **kwargs):
        raise AssertionError("uniform regions should not call np.unique")

    monkeypatch.setattr(nmag_sim.np, "unique", fail_unique)

    np.testing.assert_allclose(
        sim._simplex_material_ms_values([1, 1]),
        [1.0e6, 1.0e6],
    )


def test_simplex_material_ms_values_maps_mixed_regions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        simplices_indices=[
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        simplices_regions=[1, 2],
    )
    mesh_path = tmp_path / "mesh.nmesh"
    mesh.save(mesh_path)

    material_a = nmag.MagMaterial(
        name="A",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    material_b = nmag.MagMaterial(
        name="B",
        Ms=nmag.SI(2.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    sim = nmag.Simulation(name="mvp")
    sim.load_mesh(
        str(mesh_path),
        [("region-a", material_a), ("region-b", material_b)],
        unit_length=nmag.SI(1e-9, "m"),
    )

    np.testing.assert_allclose(
        sim._simplex_material_ms_values([1, 2]),
        [1.0e6, 2.0e6],
    )


