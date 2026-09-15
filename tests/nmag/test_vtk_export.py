from __future__ import annotations

import h5py
import meshio
import numpy as np

import nmesh
from nmag.vtk_export import export_vtk, resolve_snapshot


def _write_snapshot(path, mesh_path):
    points = np.asarray(nmesh.load(mesh_path).points, dtype=float)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mesh/points", data=points)
        handle.create_dataset("fields/m", data=np.tile([1.0, 0.0, 0.0], (len(points), 1)))
        handle.create_dataset("fields/pin", data=np.ones(len(points)))


def _write_mesh(path):
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[4],
    )
    mesh.save(path)


def test_export_vtk_writes_selected_point_fields_and_regions(tmp_path):
    mesh_path = tmp_path / "mesh.nmesh.h5"
    snapshot_path = tmp_path / "simulation_dat.h5"
    output_path = tmp_path / "simulation.vtu"
    _write_mesh(mesh_path)
    _write_snapshot(snapshot_path, mesh_path)

    export_vtk(snapshot_path, output_path, mesh_path=mesh_path, fields=("m",))

    exported = meshio.read(output_path)
    assert exported.points.shape == (4, 3)
    assert exported.cells[0].type == "tetra"
    assert np.array_equal(exported.cell_data["region"][0], [4])
    assert exported.point_data["m"].shape == (4, 3)
    assert "pin" not in exported.point_data


def test_export_vtk_requires_external_topology_for_modern_snapshot(tmp_path):
    snapshot_path = tmp_path / "simulation_dat.h5"
    with h5py.File(snapshot_path, "w") as handle:
        handle.create_dataset("mesh/points", data=np.zeros((4, 3)))
        handle.create_dataset("fields/m", data=np.zeros((4, 3)))

    try:
        export_vtk(snapshot_path, tmp_path / "out.vtk")
    except ValueError as exc:
        assert "provide --mesh" in str(exc)
    else:
        raise AssertionError("export_vtk accepted a snapshot without cell topology")


def test_resolve_snapshot_accepts_simulation_base_name(tmp_path):
    snapshot = tmp_path / "simulation_dat.h5"
    snapshot.touch()

    assert resolve_snapshot(tmp_path / "simulation") == snapshot
