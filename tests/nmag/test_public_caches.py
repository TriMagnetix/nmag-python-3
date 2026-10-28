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
    assert not first_points.flags.writeable

    first_simplices = sim._mesh_simplices()
    second_simplices = sim._mesh_simplices()
    assert second_simplices is first_simplices
    assert not first_simplices.flags.writeable

    first_bounds = sim._mesh_bounds()
    second_bounds = sim._mesh_bounds()
    assert second_bounds[0] is first_bounds[0]
    assert second_bounds[1] is first_bounds[1]
    assert second_bounds[2] is first_bounds[2]

    sim._invalidate_demag(clear_geometry=True)
    assert sim._mesh_points_cache is None
    assert sim._mesh_simplices_cache is None
    assert sim._mesh_bounds_cache is None

    refreshed_points = sim._mesh_points()
    assert sim._mesh_points_cache is not None
    assert refreshed_points is sim._mesh_points()

    refreshed_simplices = sim._mesh_simplices()
    assert sim._mesh_simplices_cache is not None
    assert refreshed_simplices is sim._mesh_simplices()


def test_rhs_hot_path_caches_and_lazy_diagnostics(tmp_path, monkeypatch):
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

    _ = sim._subfield_array("dmdt")
    assert sim._last_demag_solve_diagnostics is None
    assert sim._last_demag_solve_diagnostics_inputs is not None
    assert sim._demag_nodal_recovery_operator_cache is not None
    assert sim._exchange_operator_cache is not None

    first_recovery_cache = sim._demag_nodal_recovery_operator_cache
    first_exchange_cache = sim._exchange_operator_cache

    sim.set_m(lambda point: [0.0, 1.0, 0.0] if point[0] <= 5.0e-10 else [0.0, 0.0, 1.0])
    _ = sim._subfield_array("dmdt")
    assert sim._demag_nodal_recovery_operator_cache is first_recovery_cache
    assert sim._exchange_operator_cache is first_exchange_cache
    assert sim._last_demag_solve_diagnostics is None

    diagnostics = sim.last_demag_solve_diagnostics
    assert diagnostics["phi1_centered_residual_relative_l2"] >= 0.0
    assert sim._last_demag_solve_diagnostics_inputs is None
    diagnostics["node_count"] = -999
    assert sim.last_demag_solve_diagnostics["node_count"] > 0

    # Invalidating nodal material coefficients happens automatically when material parameters change
    material.exchange_coupling = nmag.SI(26.0e-12, "J/m")
    exchange_doubled = sim._get_exchange_nodal_field()
    assert sim._exchange_operator_cache is not first_exchange_cache
    assert np.max(np.abs(exchange_doubled)) > 0.0

    sim._invalidate_demag(clear_geometry=True)
    assert sim._demag_nodal_recovery_operator_cache is None
    assert sim._exchange_operator_cache is None


def test_material_and_geometry_mutations_automatically_invalidate_caches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_tetra_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
        anisotropy=nmag.uniaxial_anisotropy(axis=[0.0, 0.0, 1.0], K1=nmag.SI(1.0e4, "J/m^3")),
    )
    sim = nmag.Simulation(name="mvp-invalidation")
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    sim.set_m(lambda point: [1.0, 0.0, 0.0] if point[0] <= 5.0e-10 else [0.0, 1.0, 0.0])

    h_demag_1 = np.array(sim._get_demag_nodal_field(), copy=True)
    nodal_ms_1 = np.array(sim._nodal_ms_values(), copy=True)
    anis_field_1, _ = sim._get_anisotropy_fields()
    anis_field_1 = np.array(anis_field_1, copy=True)
    thermal_1 = material.thermal_factor

    # Mutate Ms and anisotropy on the material without calling set_m() or _invalidate_demag()
    material.Ms = nmag.SI(2.0e6, "A/m")
    material.anisotropy = nmag.uniaxial_anisotropy(
        axis=[0.0, 0.0, 1.0], K1=nmag.SI(4.0e4, "J/m^3")
    )
    assert material.thermal_factor != thermal_1
    nodal_ms_2 = sim._nodal_ms_values()
    np.testing.assert_allclose(nodal_ms_2, 2.0 * nodal_ms_1)
    h_demag_2 = sim._get_demag_nodal_field()
    np.testing.assert_allclose(h_demag_2, 2.0 * h_demag_1, rtol=1.0e-10, atol=1.0e-6)
    anis_field_2, _ = sim._get_anisotropy_fields()
    np.testing.assert_allclose(anis_field_2, 2.0 * anis_field_1, rtol=1.0e-10, atol=1.0e-6)

    # Scale mesh node positions directly without manual cache invalidation
    points_before = np.array(sim._mesh_points(), copy=True)
    sim.mesh.scale_node_positions(2.0)
    points_after = sim._mesh_points()
    np.testing.assert_allclose(points_after, 2.0 * points_before)

    # Empty simplex guard in _cell_demag_from_potential and default volume_charge_scales
    empty_cell_h = sim._cell_demag_from_potential(
        points_after,
        np.empty((0, 4), dtype=int),
        np.zeros(len(points_after), dtype=float),
    )
    assert empty_cell_h.shape == (0, 3)

    simplices = sim._mesh_simplices()
    ms_vals = sim._simplex_material_ms_values(sim.mesh.regions)
    stiff, div, vols = sim._assemble_demag_fem_system(
        points_after,
        simplices,
        sim._fields["m"],
        ms_vals,
    )
    assert stiff is not None and len(div) == len(points_after) and len(vols) == len(simplices)


def test_low_memory_mode_and_out_of_bounds_recovery_validation(tmp_path, monkeypatch):
    import pytest

    from nmag.backends import MEMORY_MODE_ENV

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NMAG_ACCELERATOR", "off")
    sim = nmag.Simulation(name="mvp-low-mem")
    sim.load_mesh(
        str(SPHERE1_MESH),
        [("sphere", nmag.MagMaterial(name="Py", Ms=nmag.SI(1.0e6, "A/m")))],
        unit_length=nmag.SI(1e-9, "m"),
    )
    sim.set_m([1.0, 0.0, 0.0])
    default_h = np.asarray(sim.get_subfield("H_demag"))
    assert sim._demag_fem_divergence_operator_cache is not None
    assert sim._demag_nodal_recovery_operator_cache is not None

    monkeypatch.setenv(MEMORY_MODE_ENV, "low")
    sim._invalidate_demag(clear_geometry=True)
    low_mem_h = np.asarray(sim.get_subfield("H_demag"))
    assert sim._demag_fem_divergence_operator_cache is None
    assert sim._demag_nodal_recovery_operator_cache is None
    np.testing.assert_allclose(low_mem_h, default_h, rtol=1.0e-10, atol=1.0e-4)

    with pytest.raises(IndexError):
        sim._recover_demag_nodal_field_python(
            simplices=np.array([[0, 1, 2, 999]], dtype=int),
            volumes=np.array([0.0], dtype=float),
            cell_h=np.zeros((1, 3), dtype=float),
            weights=np.ones(4, dtype=float),
            point_count=4,
        )

    # Direct mesh_from_points_and_simplices accepts NumPy arrays
    import nmesh

    pts_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    simps_np = np.array([[0, 1, 2, 3]], dtype=int)
    regs_np = np.array([1], dtype=int)
    np_mesh = nmesh.mesh_from_points_and_simplices(
        points=pts_np,
        simplices_indices=simps_np,
        simplices_regions=regs_np,
    )
    assert len(np_mesh.points) == 4


