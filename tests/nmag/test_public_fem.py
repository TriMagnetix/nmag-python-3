from __future__ import annotations

from pathlib import Path

import numpy as np
import public_api_support_module as helpers
import pytest

import nmag
import nmesh

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"



def test_demag_fem_geometry_vectorized_path_matches_loop(tmp_path, monkeypatch):
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
    vectorized = sim._demag_fem_geometry_for_mesh(points, simplices)
    loop = sim._demag_fem_geometry_for_mesh_loop(
        points,
        simplices,
        np.zeros_like(vectorized[0]),
        np.zeros_like(vectorized[1]),
        np.zeros_like(vectorized[2]),
    )

    for vectorized_part, loop_part in zip(vectorized, loop, strict=True):
        np.testing.assert_allclose(vectorized_part, loop_part, rtol=1e-12, atol=1e-24)


def test_demag_fem_geometry_rust_backend_matches_python(tmp_path, monkeypatch):
    rust_accel = pytest.importorskip("nmag_accel")
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

    points = np.asarray(sim.mesh.points, dtype=float)
    simplices = np.asarray(sim.mesh.simplices, dtype=int)
    python_geometry = sim._demag_fem_geometry_for_mesh_python(points, simplices)
    rust_geometry = sim._demag_fem_geometry_for_mesh_rust(points, simplices)

    for rust_part, python_part in zip(rust_geometry, python_geometry, strict=True):
        np.testing.assert_allclose(rust_part, python_part, rtol=1e-12, atol=1e-24)

    raw_rust_geometry = rust_accel.build_demag_fem_geometry(
        np.asarray(points, dtype=np.float64),
        np.asarray(simplices, dtype=np.int64),
    )
    assert isinstance(raw_rust_geometry[0], np.ndarray)
    assert raw_rust_geometry[0].dtype == np.float64
    for rust_part, python_part in zip(raw_rust_geometry, python_geometry, strict=True):
        np.testing.assert_allclose(rust_part, python_part, rtol=1e-12, atol=1e-24)

    sim._demag_fem_geometry_cache = None
    selected_geometry = sim._demag_fem_geometry_for_mesh(points, simplices)
    for selected_part, python_part in zip(selected_geometry, python_geometry, strict=True):
        np.testing.assert_allclose(selected_part, python_part, rtol=1e-12, atol=1e-24)


def test_demag_fem_assembly_rust_backend_matches_python(tmp_path, monkeypatch):
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
    sim.set_m(lambda point: [1.0, point[0] * 1e9, point[1] * 1e9])

    points = sim._mesh_points()
    simplices = np.asarray(sim.mesh.simplices, dtype=int)
    regions = list(sim.mesh.regions or [1] * len(simplices))
    ms_values = sim._simplex_material_ms_values(regions)
    stiffness, gradients, volumes = sim._demag_fem_geometry_for_mesh_python(points, simplices)
    m = np.asarray(sim._fields["m"], dtype=float)

    python_divergence = sim._assemble_demag_fem_divergence_python(
        simplices,
        gradients,
        volumes,
        m,
        ms_values,
        len(points),
    )
    rust_divergence = sim._assemble_demag_fem_divergence_rust(
        simplices,
        gradients,
        volumes,
        m,
        ms_values,
        len(points),
    )

    np.testing.assert_allclose(rust_divergence, python_divergence, rtol=1e-12, atol=1e-20)

    selected_stiffness, selected_divergence, selected_volumes = sim._assemble_demag_fem_system(
        points,
        simplices,
        m,
        ms_values,
        np.ones(len(simplices), dtype=float),
    )

    np.testing.assert_allclose(selected_stiffness, stiffness, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(selected_divergence, python_divergence, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(selected_volumes, volumes, rtol=1e-12, atol=1e-20)


def test_demag_fem_geometry_rejects_degenerate_cells():
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        simplices_indices=[
            [0, 1, 2, 3],
            [0, 1, 2, 4],
        ],
        simplices_regions=[1, 1],
    )
    sim = nmag.Simulation(name="mvp")
    sim.mesh = mesh

    with pytest.raises(ValueError, match=r"invalid cell indices: \[1\]"):
        sim._demag_fem_geometry_for_mesh(
            np.asarray(mesh.points, dtype=float),
            np.asarray(mesh.simplices, dtype=int),
        )


def test_demag_fem_geometry_rust_backend_rejects_degenerate_cells(monkeypatch):
    pytest.importorskip("nmag_accel")
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        simplices_indices=[
            [0, 1, 2, 3],
            [0, 1, 2, 4],
        ],
        simplices_regions=[1, 1],
    )
    sim = nmag.Simulation(name="mvp")
    sim.mesh = mesh

    points = np.asarray(mesh.points, dtype=float)
    simplices = np.asarray(mesh.simplices, dtype=int)
    with pytest.raises(ValueError, match=r"invalid cell indices: \[1\]"):
        sim._demag_fem_geometry_for_mesh_rust(points, simplices)
