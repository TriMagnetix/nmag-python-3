"""Tests for legacy ``.nmesh.h5`` loading support."""

import pytest

from nmesh.io.legacy_nmesh_hdf5 import load_raw_mesh_from_legacy_nmesh_hdf5


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


def test_load_raw_mesh_from_legacy_nmesh_hdf5(tmp_path):
    """The legacy loader should decode the classic PyTables-style layout."""
    sample_path = tmp_path / "model.nmesh.h5"
    _write_legacy_nmesh_hdf5(sample_path)

    raw_mesh = load_raw_mesh_from_legacy_nmesh_hdf5(sample_path)

    assert raw_mesh.dim == 3
    assert len(raw_mesh.points) == 5
    assert len(raw_mesh.simplices) == 2
    assert len(raw_mesh.regions) == 2
    assert raw_mesh.points[0] == [0.0, 259.0, 3.0]
    assert raw_mesh.simplices[0] == [0, 1, 2, 3]
    assert raw_mesh.regions == [1, 7]
    assert raw_mesh.periodic_point_indices == [[0, 3], [1, 2, 4]]
    assert raw_mesh.permutation == [4, 3, 2, 1, 0]
