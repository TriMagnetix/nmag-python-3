from __future__ import annotations

from pathlib import Path

import h5py
import public_api_support_module as helpers

import nmag

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"



def test_get_all_field_names_uses_cheap_availability_predicate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_single_region_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )

    sim = nmag.Simulation(name="mvp")
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    sim.set_m([1.0, 0.0, 0.0])
    sim.set_H_ext([1.0, 2.0, 3.0], nmag.SI("A/m"))

    def fail_compute(subfieldname):
        raise AssertionError(f"availability should not compute {subfieldname}")

    state_check_counts = {"tetrahedral": 0, "anisotropy": 0}
    original_tetrahedral_check = sim._has_tetrahedral_mesh_or_empty
    original_anisotropy_check = sim._anisotropy_is_zero_by_construction

    def counting_tetrahedral_check():
        state_check_counts["tetrahedral"] += 1
        return original_tetrahedral_check()

    def counting_anisotropy_check():
        state_check_counts["anisotropy"] += 1
        return original_anisotropy_check()

    monkeypatch.setattr(sim, "_compute_subfield_array", fail_compute)
    monkeypatch.setattr(sim, "_has_tetrahedral_mesh_or_empty", counting_tetrahedral_check)
    monkeypatch.setattr(
        sim,
        "_anisotropy_is_zero_by_construction",
        counting_anisotropy_check,
    )

    field_names = sim.get_all_field_names()

    assert state_check_counts == {"tetrahedral": 1, "anisotropy": 1}
    assert "H_total" in field_names
    assert "H_demag" in field_names
    assert "m" in field_names
    assert "current_density" not in field_names
    assert "dm_dcurrent" not in field_names


def test_get_all_field_names_keeps_exchange_available_with_unsupported_anisotropy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_single_region_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
        anisotropy=nmag.uniaxial_anisotropy(axis=[0.0, 0.0, 1.0], K1=1.0e5),
    )

    sim = nmag.Simulation(name="mvp")
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    sim.set_m([1.0, 0.0, 0.0])

    field_names = sim.get_all_field_names()

    assert "H_exch" in field_names
    assert "E_exch" in field_names
    assert "H_anis" not in field_names
    assert "E_anis" not in field_names
    assert "H_total" not in field_names
    assert "E_total" not in field_names


def test_h_demag_save_and_average_use_internal_demag_paths(tmp_path, monkeypatch):
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

    def fail_public_h_demag_path(subfieldname, units=None):
        if subfieldname == "H_demag":
            raise AssertionError("H_demag storage should not round-trip through get_subfield().")
        return original_get_subfield(subfieldname, units=units)

    original_get_subfield = sim.get_subfield
    monkeypatch.setattr(sim, "get_subfield", fail_public_h_demag_path)

    average = sim.get_subfield_average("H_demag")
    sim.save_spatial_fields(str(tmp_path / "demag_only.h5"), ["H_demag"])

    assert len(average) == 3
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
    with h5py.File(tmp_path / "demag_only.h5") as h5:
        assert h5["fields/H_demag"].shape == (4, 3)


def test_save_data_reuses_subfield_arrays_across_ndt_and_spatial_fields(tmp_path, monkeypatch):
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
    sim.set_H_ext([1.0, 2.0, 3.0], nmag.SI("A/m"))

    counts = {}
    original_compute = sim._compute_subfield_array

    def counting_compute(subfieldname):
        counts[subfieldname] = counts.get(subfieldname, 0) + 1
        return original_compute(subfieldname)

    monkeypatch.setattr(sim, "_compute_subfield_array", counting_compute)

    sim.save_data(fields="all")

    assert sim._subfield_array_cache is None
    assert sim.last_save_timings_seconds["total"] >= 0.0
    assert sim.last_save_timings_seconds["save_spatial_fields"] >= 0.0
    assert sim.last_spatial_save_timings_seconds["total"] >= 0.0
    assert sim.last_spatial_save_timings_seconds["field:H_demag:total"] >= 0.0
    assert sim.last_spatial_save_timings_seconds["field:H_demag:compute"] >= 0.0
    assert sim.last_spatial_save_timings_seconds["field:H_demag:write"] >= 0.0
    assert sim.last_spatial_save_timings_seconds["mesh_points"] >= 0.0
    assert sim.last_save_timings_seconds["spatial_detail:total"] >= 0.0
    assert sim.last_save_timings_seconds["spatial_detail:field:H_demag:total"] >= 0.0
    assert sim.last_save_timings_seconds["spatial_detail:field:H_demag:compute"] >= 0.0
    assert sim.last_save_timings_seconds["spatial_detail:field:H_demag:write"] >= 0.0
    assert sim.last_save_timings_seconds["spatial_detail:field_selection"] >= 0.0
    assert sim.last_save_timings_seconds["spatial_detail:write_dispatch"] >= 0.0
    assert sim.last_save_timings_seconds["spatial_detail:mesh_points"] >= 0.0
    assert sim.last_save_timings_seconds["average:H_demag"] >= 0.0
    assert sim.last_save_timings_seconds[
        "average_detail:H_total_Py:composed_average"
    ] >= 0.0
    assert sim.last_save_timings_seconds["average_detail:H_total_Py:total"] >= 0.0
    assert sim.last_save_timings_seconds[
        "average_detail:H_total_Py:component_average:H_ext:constant"
    ] >= 0.0
    assert sim.last_save_timings_seconds[
        "average_detail:H_total_Py:component_average:H_demag:direct_cell_average"
    ] >= 0.0
    assert sim.last_save_timings_seconds[
        "average_detail:H_total_Py:component_average:H_anis:constant"
    ] >= 0.0
    assert sim.last_save_timings_seconds[
        "average_detail:H_total_Py:component_average:H_exch:constant"
    ] >= 0.0
    assert "average_detail:H_total_Py:subfield_array:H_demag:compute" not in (
        sim.last_save_timings_seconds
    )
    assert (
        "average_detail:H_total_Py:subfield_array:H_total:compute"
        not in sim.last_save_timings_seconds
    )
    assert sim.last_save_timings_seconds[
        "average_detail:H_total_Py:subfield_array:H_demag_detail:auxiliary_fields"
    ] >= 0.0
    assert sim.last_save_timings_seconds[
        "average_detail:H_total_Py:subfield_array:demag_auxiliary:gauge_solve"
    ] >= 0.0
    assert sim.last_save_timings_seconds[
        "average_detail:H_total_Py:subfield_array:demag_auxiliary:gauge_solve:setup"
    ] >= 0.0
    assert any(
        sim.last_save_timings_seconds.get(
            f"average_detail:H_total_Py:subfield_array:demag_auxiliary:gauge_solve:{phase}",
        )
        is not None
        for phase in ("numpy_solve", "numpy_lstsq", "scipy_solve", "scipy_lstsq")
    )
    assert sim.last_save_timings_seconds[
        "average_detail:H_total_Py:subfield_array:demag_auxiliary:lindholm_bem:index"
    ] >= 0.0
    assert any(
        sim.last_save_timings_seconds.get(
            f"average_detail:H_total_Py:subfield_array:demag_auxiliary:lindholm_bem:{backend}",
        )
        is not None
        for backend in ("python", "numba", "rust")
    )
    assert "average_detail:H_total_Py:subfield_array:H_demag_detail:nodal_recovery" not in (
        sim.last_save_timings_seconds
    )
    assert sim.last_save_timings_seconds[
        "average_detail:maxangle_m_Py:uniform_check"
    ] >= 0.0
    assert "average_detail:maxangle_m_Py:edge_angles" not in sim.last_save_timings_seconds
    assert sim.last_demag_solve_diagnostics["phi1_centered_residual_max_abs"] >= 0.0
    assert sim.last_demag_solve_diagnostics["dirichlet_residual_max_abs"] >= 0.0
    assert (
        sim.last_demag_solve_diagnostics["phi1_centered_residual_relative_l2"]
        >= 0.0
    )
    assert sim.last_demag_solve_diagnostics[
        "dirichlet_residual_relative_l2"
    ] >= 0.0
    assert (
        sim.last_demag_solve_diagnostics["phi1_centered_residual_relative_l2"]
        < 1.0e-9
    )
    assert (
        sim.last_demag_solve_diagnostics["dirichlet_residual_relative_l2"]
        < 1.0e-9
    )
    assert sim.last_demag_solve_diagnostics["phi1_gauge_sum_abs"] >= 0.0
    assert sim.last_demag_solve_diagnostics["node_count"] == len(sim.mesh.points)
    assert sim.last_demag_solve_diagnostics["boundary_node_count"] <= len(sim.mesh.points)
    assert sim.last_demag_solve_diagnostics["gauge_augmented_size"] == len(sim.mesh.points) + 1
    assert sim.last_demag_solve_diagnostics["dirichlet_interior_node_count"] >= 0
    assert sim.last_subfield_average_timings_seconds["total"] >= 0.0
    for fieldname in [
        "H_total",
        "H_ext",
        "H_demag",
        "H_anis",
        "H_exch",
        "M",
        "m",
        "dmdt",
        "E_total",
        "E_ext",
        "E_demag",
        "pin",
        "phi",
        "rho",
    ]:
        assert counts[fieldname] == 1
    assert counts.get("current_density", 0) == 0
    assert counts.get("dm_dcurrent", 0) == 0


