"""Tests for legacy ``.nmesh.h5`` loading support."""

import pytest

from nmesh.io.legacy_nmesh_hdf5 import (
    load_raw_mesh_from_legacy_nmesh_hdf5,
)

h5py = pytest.importorskip("h5py")


def _write_legacy_nmesh_hdf5(path):
    """Create a tiny legacy nmesh HDF5 file for regression testing."""
    with h5py.File(path, "w") as handle:
        etc_group = handle.create_group("etc")
        mesh_group = handle.create_group("mesh")

        etc_group.create_dataset("filetype", data=b"nmesh")
        etc_group.create_dataset("fileversion", data=b"1.0")

        mesh_group.create_dataset(
            "points",
            data=[
                [0.0, 259.0, 3.0],
                [0.0, 259.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        mesh_group.create_dataset(
            "simplices",
            data=[
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ],
            dtype="i4",
        )
        mesh_group.create_dataset("simplicesregions", data=[1, 7], dtype="i4")
        mesh_group.create_dataset(
            "periodicpointindices",
            data=[
                [0, 3, -1],
                [1, 2, 4],
            ],
            dtype="i4",
        )
        mesh_group.create_dataset("permutation", data=[4, 3, 2, 1, 0], dtype="i4")


def test_legacy_loader_rejects_missing_mesh_group(tmp_path):
    """Should raise ValueError if /mesh group is missing."""
    path = tmp_path / "invalid.nmesh.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("etc")

    with pytest.raises(ValueError, match="missing the /mesh group"):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_rejects_mesh_dataset_instead_of_group(tmp_path):
    path = tmp_path / "mesh_dataset.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mesh", data=[1])

    with pytest.raises(ValueError, match="/mesh, but it is not an HDF5 group"):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_rejects_wrong_filetype(tmp_path):
    """Should raise ValueError if filetype is not 'nmesh'."""
    path = tmp_path / "wrong_type.h5"
    with h5py.File(path, "w") as handle:
        etc_group = handle.create_group("etc")
        etc_group.create_dataset("filetype", data=b"other")
        handle.create_group("mesh")

    with pytest.raises(ValueError, match="has filetype 'other', expected 'nmesh'"):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_rejects_missing_required_datasets(tmp_path):
    """Should raise ValueError if required datasets are missing."""
    path = tmp_path / "incomplete.nmesh.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("etc")
        mesh_group = handle.create_group("mesh")
        # Only add points, missing simplices and regions
        mesh_group.create_dataset("points", data=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="missing one of"):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_rejects_group_instead_of_required_dataset(tmp_path):
    path = tmp_path / "points_group.h5"
    with h5py.File(path, "w") as handle:
        mesh_group = handle.create_group("mesh")
        mesh_group.create_group("points")

    with pytest.raises(ValueError, match="/mesh/points, but it is not an HDF5 dataset"):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_rejects_group_instead_of_optional_dataset(tmp_path):
    path = tmp_path / "periodic_group.h5"
    _write_legacy_nmesh_hdf5(path)
    with h5py.File(path, "a") as handle:
        del handle["mesh/periodicpointindices"]
        handle["mesh"].create_group("periodicpointindices")

    with pytest.raises(
        ValueError,
        match="/mesh/periodicpointindices, but it is not an HDF5 dataset",
    ):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_rejects_group_instead_of_filetype_dataset(tmp_path):
    path = tmp_path / "filetype_group.h5"
    _write_legacy_nmesh_hdf5(path)
    with h5py.File(path, "a") as handle:
        del handle["etc/filetype"]
        handle["etc"].create_group("filetype")

    with pytest.raises(ValueError, match="/etc/filetype, but it is not an HDF5 dataset"):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_rejects_empty_points(tmp_path):
    """Should raise ValueError if mesh has no points."""
    path = tmp_path / "empty_points.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("etc")
        mesh_group = handle.create_group("mesh")
        mesh_group.create_dataset("points", data=[])
        mesh_group.create_dataset("simplices", data=[[0, 1, 2]])
        mesh_group.create_dataset("simplicesregions", data=[1])

    with pytest.raises(ValueError, match="contains no points"):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_rejects_empty_simplices(tmp_path):
    """Should raise ValueError if mesh has no simplices."""
    path = tmp_path / "empty_simplices.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("etc")
        mesh_group = handle.create_group("mesh")
        mesh_group.create_dataset("points", data=[[0.0, 0.0, 0.0]])
        mesh_group.create_dataset("simplices", data=[])
        mesh_group.create_dataset("simplicesregions", data=[])

    with pytest.raises(ValueError, match="contains no simplices"):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_rejects_mismatched_regions(tmp_path):
    """Should raise ValueError if regions count doesn't match simplices."""
    path = tmp_path / "mismatched.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("etc")
        mesh_group = handle.create_group("mesh")
        mesh_group.create_dataset("points", data=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mesh_group.create_dataset("simplices", data=[[0, 1]])
        mesh_group.create_dataset("simplicesregions", data=[1, 2])  # Too many regions

    with pytest.raises(ValueError, match="mismatched regions"):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_rejects_out_of_bounds_indices(tmp_path):
    """Should raise ValueError if simplex references non-existent point."""
    path = tmp_path / "bad_indices.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("etc")
        mesh_group = handle.create_group("mesh")
        mesh_group.create_dataset("points", data=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mesh_group.create_dataset("simplices", data=[[0, 1, 5]])  # Index 5 doesn't exist
        mesh_group.create_dataset("simplicesregions", data=[1])

    with pytest.raises(ValueError, match="out-of-bounds point index"):
        load_raw_mesh_from_legacy_nmesh_hdf5(path)


def test_legacy_loader_handles_minimal_valid_mesh(tmp_path):
    """Should load a mesh with only required fields."""
    path = tmp_path / "minimal.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("etc")
        mesh_group = handle.create_group("mesh")
        mesh_group.create_dataset("points", data=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        mesh_group.create_dataset("simplices", data=[[0, 1, 2]])
        mesh_group.create_dataset("simplicesregions", data=[1])

    raw_mesh = load_raw_mesh_from_legacy_nmesh_hdf5(path)

    assert len(raw_mesh.points) == 3
    assert len(raw_mesh.simplices) == 1
    assert raw_mesh.periodic_point_indices == []
    assert raw_mesh.permutation == []


def test_legacy_loader_handles_1d_mesh(tmp_path):
    """Should correctly infer dimension for 1D line meshes."""
    path = tmp_path / "line.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("etc")
        mesh_group = handle.create_group("mesh")
        mesh_group.create_dataset("points", data=[[0.0], [1.0], [2.0]])
        mesh_group.create_dataset("simplices", data=[[0, 1], [1, 2]])
        mesh_group.create_dataset("simplicesregions", data=[1, 1])

    raw_mesh = load_raw_mesh_from_legacy_nmesh_hdf5(path)

    assert raw_mesh.dim == 1
    assert len(raw_mesh.simplices[0]) == 2


def test_legacy_loader_handles_2d_mesh(tmp_path):
    """Should correctly infer dimension for 2D triangle meshes."""
    path = tmp_path / "triangle.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("etc")
        mesh_group = handle.create_group("mesh")
        mesh_group.create_dataset("points", data=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        mesh_group.create_dataset("simplices", data=[[0, 1, 2], [1, 2, 3]])
        mesh_group.create_dataset("simplicesregions", data=[1, 1])

    raw_mesh = load_raw_mesh_from_legacy_nmesh_hdf5(path)

    assert raw_mesh.dim == 2
    assert len(raw_mesh.simplices[0]) == 3
