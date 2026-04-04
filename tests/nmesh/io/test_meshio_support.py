"""Tests for meshio-backed mesh file IO."""

import pytest

from nmesh.backend import RawMesh
from nmesh.io import load_raw_mesh_with_meshio, save_raw_mesh_with_meshio


h5py = pytest.importorskip("h5py")


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


def test_load_1d_mesh(tmp_path):
    """Should correctly load 1D line meshes."""
    path = tmp_path / "line.vtu"
    raw_mesh = RawMesh(
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        simplices=[[0, 1], [1, 2]],
        regions=[1, 2],
        dim=1,
    )

    save_raw_mesh_with_meshio(path, raw_mesh)
    loaded_mesh = load_raw_mesh_with_meshio(path)

    assert loaded_mesh.dim == 1
    assert len(loaded_mesh.simplices[0]) == 2


def test_load_2d_mesh(tmp_path):
    """Should correctly load 2D triangle meshes."""
    path = tmp_path / "triangle.vtu"
    raw_mesh = RawMesh(
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        simplices=[[0, 1, 2]],
        regions=[1],
        dim=2,
    )

    save_raw_mesh_with_meshio(path, raw_mesh)
    loaded_mesh = load_raw_mesh_with_meshio(path)

    assert loaded_mesh.dim == 2
    assert len(loaded_mesh.simplices[0]) == 3


def test_fallback_to_legacy_loader_for_h5(tmp_path):
    """Should fall back to legacy loader for .h5 files that meshio can't read."""
    path = tmp_path / "legacy.h5"

    # Create a legacy nmesh HDF5 file (meshio doesn't recognize .h5 extension)
    with h5py.File(path, "w") as handle:
        etc_group = handle.create_group("etc")
        mesh_group = handle.create_group("mesh")

        etc_group.create_dataset("filetype", data=b"nmesh")
        mesh_group.create_dataset(
            "points", data=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        mesh_group.create_dataset("simplices", data=[[0, 1, 2]])
        mesh_group.create_dataset("simplicesregions", data=[1])

    # This should succeed via the legacy fallback (meshio doesn't support .h5)
    raw_mesh = load_raw_mesh_with_meshio(path)

    assert len(raw_mesh.points) == 3
    assert len(raw_mesh.simplices) == 1
    assert raw_mesh.regions == [1]


def test_h5_file_fails_both_loaders(tmp_path):
    """Should raise ValueError with both errors when .h5 fails both loaders."""
    path = tmp_path / "invalid.h5"

    # Create an invalid HDF5 file (no mesh group)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("dummy", data=[1, 2, 3])

    # Both meshio and legacy loader will fail, should see combined error
    with pytest.raises(ValueError, match="Unable to read.*meshio.*legacy"):
        load_raw_mesh_with_meshio(path)


def test_non_h5_file_only_tries_meshio(tmp_path):
    """Should only try meshio for non-.h5 files."""
    path = tmp_path / "nonexistent.vtu"

    # This should raise an error from meshio only (not try the legacy loader)
    # meshio._exceptions.ReadError is raised for missing files
    with pytest.raises(Exception):  # Accept any exception from meshio
        load_raw_mesh_with_meshio(path)


def test_save_mesh_without_regions(tmp_path):
    """Should handle meshes without region data."""
    path = tmp_path / "no_regions.vtu"
    raw_mesh = RawMesh(
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        simplices=[[0, 1, 2]],
        regions=[],
        dim=2,
    )

    save_raw_mesh_with_meshio(path, raw_mesh)
    loaded_mesh = load_raw_mesh_with_meshio(path)

    assert len(loaded_mesh.points) == 3
    assert len(loaded_mesh.simplices) == 1
