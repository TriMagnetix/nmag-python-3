from __future__ import annotations

from pathlib import Path

import numpy as np
import public_api_support_module as helpers
import pytest

import nmag
import nmag.simulation as nmag_sim
from nmag.simulation import (
    DEMAG_DENSE_MAX_POINTS_ENV,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"



def test_dense_demag_refuses_mesh_above_configured_limit(monkeypatch):
    monkeypatch.setenv(DEMAG_DENSE_MAX_POINTS_ENV, "4")
    sim = nmag.Simulation(name="mvp")
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    simplices = np.asarray([[0, 1, 2, 3]], dtype=int)

    with pytest.raises(MemoryError, match="safety limit is 4"):
        sim._demag_fem_geometry_for_mesh_python(points, simplices)


@pytest.mark.parametrize("value", ["0", "bad"])
def test_dense_demag_rejects_invalid_configured_limit(monkeypatch, value):
    monkeypatch.setenv(DEMAG_DENSE_MAX_POINTS_ENV, value)
    sim = nmag.Simulation(name="mvp")

    with pytest.raises(ValueError, match=DEMAG_DENSE_MAX_POINTS_ENV):
        sim._demag_fem_geometry_for_mesh_python(
            np.empty((0, 3), dtype=float),
            np.empty((0, 4), dtype=int),
        )


def test_subfield_average_uses_cell_volume_weighting(tmp_path, monkeypatch):
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
        lambda point: (
            [1.0, 0.0, 0.0] if point[0] <= 5.0e-10 else [0.0, 1.0, 0.0]
        ),
    )

    points = np.asarray(sim.mesh.points, dtype=float)
    simplices = np.asarray(sim.mesh.simplices, dtype=int)
    volumes = np.abs(
        np.linalg.det(points[simplices[:, 1:]] - points[simplices[:, [0]]])
    ) / 6.0
    m = np.asarray(sim.get_subfield("m"), dtype=float)
    expected_average = np.sum(
        np.mean(m[simplices], axis=1) * volumes[:, np.newaxis],
        axis=0,
    ) / np.sum(volumes)

    np.testing.assert_allclose(sim.get_subfield_average("m"), expected_average)
    assert not np.allclose(expected_average, np.mean(m, axis=0))


def test_h_demag_direct_average_matches_nodal_average_reference(tmp_path, monkeypatch):
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

    direct_average = sim._demag_cell_field_average()
    nodal_reference = sim._field_average(sim._get_demag_nodal_field())

    np.testing.assert_allclose(
        direct_average,
        nodal_reference,
        rtol=1.0e-12,
        atol=1.0e-9,
    )

    sim._invalidate_demag()
    np.testing.assert_allclose(
        sim.get_subfield_average("H_demag"),
        nodal_reference,
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    assert sim._demag_cell_field_cache is not None
    assert sim._demag_nodal_cache is None
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


def test_subfield_averages_reuse_simplex_volume_weights(tmp_path, monkeypatch):
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

    volume_calls = []
    original_simplex_volumes = nmag_sim._simplex_volumes

    def counting_simplex_volumes(points, simplices):
        volume_calls.append(len(simplices))
        return original_simplex_volumes(points, simplices)

    monkeypatch.setattr(nmag_sim, "_simplex_volumes", counting_simplex_volumes)

    sim.get_subfield_average("m")
    assert sim._volume_average_node_weights_cache is not None
    first_weight_cache = sim._volume_average_node_weights_cache
    sim._scalar_field_average(np.arange(len(sim.mesh.points), dtype=float))

    assert volume_calls == [2]
    assert sim._volume_average_node_weights_cache is first_weight_cache

    sim._invalidate_demag(clear_geometry=True)
    assert sim._volume_average_node_weights_cache is None
    sim.get_subfield_average("m")

    assert volume_calls == [2, 2]


def test_nodal_ms_values_reuse_mesh_geometry_cache(tmp_path, monkeypatch):
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

    first_values = sim._nodal_ms_values()
    assert sim._nodal_ms_values_cache is not None

    def fail_simplex_material_values(_regions):
        raise AssertionError("cached nodal Ms values should not recompute simplex Ms")

    monkeypatch.setattr(sim, "_simplex_material_ms_values", fail_simplex_material_values)

    assert sim._nodal_ms_values() is first_values

    sim._invalidate_demag(clear_geometry=True)
    assert sim._nodal_ms_values_cache is None


def test_uniform_subfield_averages_skip_volume_weights(tmp_path, monkeypatch):
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

    def fail_volume_weights(_points, _simplices):
        raise AssertionError("uniform averages should not need volume weights")

    monkeypatch.setattr(sim, "_simplex_volume_weights", fail_volume_weights)

    np.testing.assert_allclose(sim.get_subfield_average("m"), [1.0, 0.0, 0.0])
    assert sim.get_subfield_average("pin") == 1.0


def test_constant_subfield_averages_skip_array_materialization(tmp_path, monkeypatch):
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
    sim.set_H_ext([1.0, 2.0, 3.0], nmag.SI("A/m"))

    def fail_array_materialization(subfieldname):
        raise AssertionError(f"{subfieldname} should use a direct constant average")

    monkeypatch.setattr(sim, "_compute_subfield_array", fail_array_materialization)

    assert sim.get_subfield_average("H_ext") == [1.0, 2.0, 3.0]
    assert sim.last_subfield_average_timings_seconds["constant_average"] >= 0.0
    assert sim.get_subfield_average("pin") == 1.0
    assert sim.get_subfield_average("H_anis") == [0.0, 0.0, 0.0]
    assert sim.get_subfield_average("E_anis") == 0.0
    assert sim.get_subfield_average("H_exch") == [0.0, 0.0, 0.0]
    assert sim.get_subfield_average("E_exch") == 0.0


def test_h_total_average_uses_component_averages(tmp_path, monkeypatch):
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

    expected = sim._field_average(sim._derived_subfield_array("H_total"))

    original_compute = sim._compute_subfield_array

    def fail_h_total_materialization(subfieldname):
        if subfieldname == "H_total":
            raise AssertionError("H_total average should compose component averages")
        return original_compute(subfieldname)

    monkeypatch.setattr(sim, "_compute_subfield_array", fail_h_total_materialization)

    np.testing.assert_allclose(sim.get_subfield_average("H_total"), expected)
    timings = sim.last_subfield_average_timings_seconds
    assert timings["composed_average"] >= 0.0
    assert timings["component_average:H_ext:constant"] >= 0.0
    assert timings["component_average:H_demag:direct_cell_average"] >= 0.0
    assert timings["component_average:H_anis:constant"] >= 0.0
    assert timings["component_average:H_exch:array"] >= 0.0
    assert timings["component_average:H_exch:field_average"] >= 0.0
    assert "subfield_array:H_total:compute" not in timings


