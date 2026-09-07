from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import public_api_support_module as helpers
import pytest

import nmag
import nmag.demag as nmag_demag
import nmag.simulation as nmag_sim

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"



def test_uniform_maxangle_average_uses_zero_fast_path(tmp_path, monkeypatch):
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

    def fail_arccos(_value):
        raise AssertionError("uniform maxangle should not enter the edge loop")

    monkeypatch.setattr(nmag_sim.np, "arccos", fail_arccos)

    assert sim.get_maxangle_average("m") == 0.0


def test_nonuniform_maxangle_average_uses_cached_vectorized_edges(tmp_path, monkeypatch):
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
    sim.set_m(lambda point: [1.0, point[0] * 1e9, point[1] * 1e9])

    expected = sim._maxangle_average_reference("m")
    norms = np.linalg.norm(np.asarray(sim.get_subfield("m"), dtype=float), axis=1)
    np.testing.assert_allclose(norms, 1.0)

    def fail_normalised_rows(_values):
        raise AssertionError("set_m stores normalised rows for maxangle")

    monkeypatch.setattr(nmag_demag, "_normalised_rows", fail_normalised_rows)
    actual = sim.get_maxangle_average("m")

    assert actual == pytest.approx(expected)
    assert sim.last_maxangle_timings_seconds["input_arrays"] >= 0.0
    assert sim.last_maxangle_timings_seconds["uniform_check"] >= 0.0
    assert sim.last_maxangle_timings_seconds["mesh_edges"] >= 0.0
    assert sim.last_maxangle_timings_seconds["edge_angles"] >= 0.0
    assert sim.last_maxangle_timings_seconds["total"] >= 0.0
    assert sim._mesh_edge_cache is not None
    cached_edges = sim._mesh_edge_cache[1]
    assert cached_edges.shape == (9, 2)
    np.testing.assert_array_equal(
        cached_edges,
        np.asarray(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
                [3, 4],
            ],
            dtype=int,
        ),
    )

    def fail_unique(*_args, **_kwargs):
        raise AssertionError("cached maxangle should not rebuild mesh edges")

    monkeypatch.setattr(nmag_sim.np, "unique", fail_unique)

    assert sim.get_maxangle_average("m") == pytest.approx(expected)
    assert sim._mesh_edge_cache is not None
    assert sim._mesh_edge_cache[1] is cached_edges


def test_nonuniform_maxangle_auto_uses_rust_when_available(tmp_path, monkeypatch):
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
    sim.load_mesh(
        str(mesh_path),
        [("generated", material)],
        unit_length=nmag.SI(1e-9, "m"),
    )
    sim.set_m(lambda point: [1.0, point[0] * 1e9, point[1] * 1e9])

    assert sim.get_maxangle_average("m") == pytest.approx(
        sim._maxangle_average_reference("m"),
    )
    assert "mesh_edges_and_angles:rust" in sim.last_maxangle_timings_seconds
    assert "mesh_edges" not in sim.last_maxangle_timings_seconds
    assert "edge_angles" not in sim.last_maxangle_timings_seconds


def test_maxangle_rust_backend_matches_python_reference(tmp_path, monkeypatch):
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
    sim.set_m(lambda point: [1.0, point[0] * 1e9, point[1] * 1e9])

    actual = sim.get_maxangle_average("m")
    expected = sim._maxangle_average_reference("m")

    assert actual == pytest.approx(expected)
    assert sim.last_maxangle_timings_seconds["input_arrays"] >= 0.0
    assert sim.last_maxangle_timings_seconds["uniform_check"] >= 0.0
    assert sim.last_maxangle_timings_seconds["mesh_edges_and_angles:rust"] >= 0.0
    assert sim.last_maxangle_timings_seconds["total"] >= 0.0


def test_uniform_exchange_field_uses_zero_fast_path(tmp_path, monkeypatch):
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

    def fail_fem_geometry(_points, _simplices):
        raise AssertionError("uniform exchange should not assemble FEM geometry")

    monkeypatch.setattr(sim, "_demag_fem_geometry_for_mesh", fail_fem_geometry)

    np.testing.assert_allclose(sim.get_subfield_average("H_exch"), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(sim._subfield_array("H_exch"), np.zeros((5, 3)))


def test_nonuniform_magnetisation_publishes_computed_exchange(tmp_path, monkeypatch):
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
    sim.set_m(lambda point: [1.0, point[0] * 1e9, 0.0])
    sim.set_H_ext([0.0, 0.0, 0.0], nmag.SI("A/m"))

    h_exch_direct = np.asarray(sim._subfield_array("H_exch"))
    assert sim._incident_cell_volume_sums_cache is not None
    first_cache = sim._incident_cell_volume_sums_cache
    points = sim._mesh_points()
    simplices = np.asarray(sim.mesh.simplices, dtype=int)
    volumes, _positive = sim._simplex_volume_weights(points, simplices)
    cached_weights = sim._incident_cell_volume_sums(points, simplices, volumes)

    assert sim._incident_cell_volume_sums_cache is first_cache
    assert cached_weights is first_cache[3]

    sim.save_data(fields="all")

    with h5py.File(tmp_path / "mvp_dat.h5") as h5:
        field_names = set(h5["fields"].keys())
        e_exch = h5["fields/E_exch"][:]
        h_exch = h5["fields/H_exch"][:]
        dmdt = h5["fields/dmdt"][:]
        phi = h5["fields/phi"][:]
        rho = h5["fields/rho"][:]
    assert "E_demag" in field_names
    assert "E_ext" in field_names
    assert "E_exch" in field_names
    assert "E_total" in field_names
    assert "H_exch" in field_names
    assert "H_total" in field_names
    assert e_exch.shape == (4,)
    assert h_exch.shape == (4, 3)
    assert dmdt.shape == (4, 3)
    assert phi.shape == (4,)
    assert rho.shape == (4,)
    np.testing.assert_allclose(h_exch, h_exch_direct)
    assert float(np.max(np.abs(e_exch))) > 0.0
    assert float(np.max(np.abs(h_exch))) > 0.0
    assert float(np.max(np.abs(dmdt))) > 0.0
    assert float(np.max(np.abs(phi))) > 0.0
    assert float(np.max(np.abs(rho))) > 0.0

    ndt_lines = (tmp_path / "mvp_dat.ndt").read_text(encoding="utf-8").splitlines()
    header = ndt_lines[1].split("\t")
    row = dict(zip(header, ndt_lines[2].split("\t"), strict=True))
    assert "E_demag_Py" in header
    assert "E_exch_Py" in header
    assert "E_total_Py" in header
    assert "H_exch_Py_0" in header
    assert "H_total_Py_0" in header
    assert "dmdt_Py_0" in header
    assert "maxangle_m_Py" in header
    assert abs(float(row["dmdt_Py_0"])) > 0.0
    assert float(row["maxangle_m_Py"]) > 0.0
