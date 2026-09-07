"""Tests for legacy ``.nmesh.h5`` loading support."""

import pytest

import nmesh
from nmesh.backend import RawMesh
from nmesh.io.legacy_nmesh_hdf5 import (
    is_legacy_nmesh_hdf5,
    load_raw_mesh_from_legacy_nmesh_hdf5,
    save_raw_mesh_as_legacy_nmesh_hdf5,
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
    assert raw_mesh.point_regions == [[1], [1, 7], [1, 7], [1, 7], [7]]
    assert raw_mesh.links
    assert raw_mesh.region_volumes


def test_save_raw_mesh_as_legacy_nmesh_hdf5_round_trip(tmp_path):
    path = tmp_path / "roundtrip.nmesh.h5"
    raw_mesh = RawMesh(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices=[[0, 1, 2, 3]],
        regions=[7],
        periodic_point_indices=[[0, 2], [1, 3]],
        permutation=[3, 2, 1, 0],
        dim=3,
    )

    save_raw_mesh_as_legacy_nmesh_hdf5(path, raw_mesh)
    loaded = load_raw_mesh_from_legacy_nmesh_hdf5(path)

    assert path.exists()
    with h5py.File(path, "r") as handle:
        assert handle.attrs["CLASS"] == b"GROUP"
        assert handle.attrs["PYTABLES_FORMAT_VERSION"] == b"2.0"
        assert handle["mesh"].attrs["CLASS"] == b"GROUP"
        assert handle["mesh/points"].attrs["CLASS"] == b"CARRAY"
        assert handle["etc/filetype"].shape == (1,)
    assert loaded.points == raw_mesh.points
    assert loaded.simplices == raw_mesh.simplices
    assert loaded.regions == raw_mesh.regions
    assert loaded.periodic_point_indices == raw_mesh.periodic_point_indices
    assert loaded.permutation == raw_mesh.permutation
    assert loaded.dim == raw_mesh.dim


def test_nmesh_load_accepts_binary_legacy_hdf5(tmp_path):
    path = tmp_path / "roundtrip.nmesh.h5"
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[7],
    )

    mesh.save(path)
    loaded = nmesh.load(path)

    assert loaded.points == mesh.points
    assert loaded.simplices == mesh.simplices
    assert loaded.regions == mesh.regions
    assert loaded.dim == 3


def test_nmesh_load_reads_legacy_hdf5_without_meshio_probe(tmp_path, monkeypatch):
    path = tmp_path / "direct.nmesh.h5"
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[7],
    )
    mesh.save(path)

    from nmesh import io as nmesh_io

    def fail_if_used(_path):
        raise AssertionError("nmesh.load() should not try meshio for legacy HDF5")

    monkeypatch.setattr(nmesh_io, "load_raw_mesh_with_meshio", fail_if_used)

    loaded = nmesh.load(path)

    assert loaded.points == mesh.points
    assert loaded.simplices == mesh.simplices
    assert loaded.regions == mesh.regions


def test_nmesh_load_routes_unrecognized_hdf5_to_meshio(tmp_path, monkeypatch):
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("unrelated", data=[1])

    expected = RawMesh(
        points=[[0.0, 0.0], [1.0, 0.0]],
        simplices=[[0, 1]],
        regions=[1],
        dim=1,
    )
    from nmesh import io as nmesh_io

    monkeypatch.setattr(nmesh_io, "load_raw_mesh_with_meshio", lambda loaded_path: expected)

    assert nmesh.load(path).raw_mesh is expected


def test_nmesh_load_keeps_malformed_recognized_legacy_error(tmp_path):
    path = tmp_path / "malformed.h5"
    with h5py.File(path, "w") as handle:
        etc_group = handle.create_group("etc")
        etc_group.create_dataset("filetype", data=b"nmesh")

    with pytest.raises(ValueError, match="missing the /mesh group"):
        nmesh.load(path)


def test_legacy_hdf5_recognition_uses_file_contents(tmp_path):
    legacy_path = tmp_path / "legacy.data"
    _write_legacy_nmesh_hdf5(legacy_path)
    other_hdf5_path = tmp_path / "other.h5"
    with h5py.File(other_hdf5_path, "w") as handle:
        handle.create_dataset("unrelated", data=[1])

    assert is_legacy_nmesh_hdf5(legacy_path)
    assert not is_legacy_nmesh_hdf5(other_hdf5_path)
