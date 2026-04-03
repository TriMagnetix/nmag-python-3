"""Tests for meshio-backed mesh file IO."""

from nmesh.backend import RawMesh
from nmesh.io import load_raw_mesh_with_meshio, save_raw_mesh_with_meshio


def test_save_and_load_raw_mesh_with_meshio_round_trip(tmp_path):
    """Meshes saved through meshio should load back with the same topology."""
    path = tmp_path / "mesh.vtu"
    raw_mesh = RawMesh(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices=[[0, 1, 2, 3]],
        regions=[7],
        dim=3,
    )

    save_raw_mesh_with_meshio(path, raw_mesh)
    loaded_mesh = load_raw_mesh_with_meshio(path)

    assert loaded_mesh.points == raw_mesh.points
    assert loaded_mesh.simplices == raw_mesh.simplices
    assert loaded_mesh.regions == raw_mesh.regions
    assert loaded_mesh.dim == raw_mesh.dim
