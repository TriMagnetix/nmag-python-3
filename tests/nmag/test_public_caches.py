from __future__ import annotations

from pathlib import Path

import numpy as np
import public_api_support_module as helpers

import nmag

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"



def test_save_data_reuses_h_total_component_average_cache(tmp_path, monkeypatch):
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
    sim.set_m(lambda point: [1.0, 0.0, 0.0] if point[0] <= 5.0e-10 else [0.0, 1.0, 0.0])
    sim.set_H_ext([1.0, 2.0, 3.0], nmag.SI("A/m"))

    sim.save_data()

    timings = sim.last_save_timings_seconds
    assert (
        timings["average_detail:H_total_Py:component_average:H_demag:direct_cell_average"]
        >= 0.0
    )
    assert timings["average_detail:H_demag:cache_hit"] >= 0.0
    assert "average_detail:H_demag:direct_cell_average" not in timings

    np.testing.assert_allclose(
        sim.get_subfield_average("H_demag"),
        sim._demag_cell_field_average(),
    )
    assert sim.last_subfield_average_timings_seconds["direct_cell_average"] >= 0.0
    assert (
        sim.last_subfield_average_timings_seconds[
            "subfield_array:H_demag_detail:cell_average"
        ]
        >= 0.0
    )
    assert (
        sim.last_subfield_average_timings_seconds[
            "subfield_array:H_demag_detail:cell_average:python"
        ]
        >= 0.0
    )
    assert "cache_hit" not in sim.last_subfield_average_timings_seconds


def test_save_data_records_demag_cell_cache_hit_on_unchanged_state(
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
    sim.set_H_ext([1.0, 2.0, 3.0], nmag.SI("A/m"))

    sim.save_data()
    first_save_timings = dict(sim.last_save_timings_seconds)
    assert (
        "average_detail:H_total_Py:subfield_array:"
        "demag_auxiliary:fem_bem_total"
    ) in first_save_timings
    assert (
        "average_detail:H_total_Py:subfield_array:"
        "H_demag_detail:cell_field_cache_hit"
    ) not in first_save_timings

    sim.save_data()
    second_save_timings = sim.last_save_timings_seconds

    assert (
        "average_detail:H_total_Py:subfield_array:"
        "H_demag_detail:cell_field_cache_hit"
    ) in second_save_timings
    assert (
        "average_detail:H_total_Py:subfield_array:"
        "demag_auxiliary:fem_bem_total"
    ) not in second_save_timings
    assert (
        "average_detail:H_total_Py:component_average:H_demag:"
        "direct_cell_average"
    ) in second_save_timings

    assert sim.get_subfield_average("rho") is not None
    assert (
        "subfield_array:demag_auxiliary:cache_hit"
        in sim.last_subfield_average_timings_seconds
    )


def test_save_data_records_bem_matrix_cache_hit_after_magnetisation_change(
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
    sim.set_H_ext([1.0, 2.0, 3.0], nmag.SI("A/m"))

    sim.save_data()
    first_save_timings = dict(sim.last_save_timings_seconds)
    assert (
        "average_detail:H_total_Py:subfield_array:"
        "demag_auxiliary:bem_matrix_cache_hit"
    ) not in first_save_timings
    assert any(
        key.startswith(
            "average_detail:H_total_Py:subfield_array:"
            "demag_auxiliary:lindholm_bem:"
        )
        for key in first_save_timings
    )

    sim.set_m(
        lambda point: [0.0, 1.0, 0.0]
        if point[1] <= 5.0e-10
        else [1.0, 0.0, 0.0]
    )
    sim.save_data()
    second_save_timings = sim.last_save_timings_seconds

    assert (
        "average_detail:H_total_Py:subfield_array:"
        "demag_auxiliary:fem_bem_total"
    ) in second_save_timings
    assert (
        "average_detail:H_total_Py:subfield_array:"
        "demag_auxiliary:bem_matrix_cache_hit"
    ) in second_save_timings
    assert not any(
        key.startswith(
            "average_detail:H_total_Py:subfield_array:"
            "demag_auxiliary:lindholm_bem:"
        )
        for key in second_save_timings
    )


def test_mesh_points_and_bounds_reuse_geometry_cache(tmp_path, monkeypatch):
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

    first_points = sim._mesh_points()
    second_points = sim._mesh_points()
    assert second_points is first_points

    first_bounds = sim._mesh_bounds()
    second_bounds = sim._mesh_bounds()
    assert second_bounds[0] is first_bounds[0]
    assert second_bounds[1] is first_bounds[1]
    assert second_bounds[2] is first_bounds[2]

    sim._invalidate_demag(clear_geometry=True)
    assert sim._mesh_points_cache is None
    assert sim._mesh_bounds_cache is None

    refreshed_points = sim._mesh_points()
    assert sim._mesh_points_cache is not None
    assert refreshed_points is sim._mesh_points()
