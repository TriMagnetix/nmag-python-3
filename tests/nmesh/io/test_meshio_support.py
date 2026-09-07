"""Tests for meshio-backed mesh file IO."""

import meshio
import numpy as np
import pytest
from meshio._exceptions import ReadError as MeshioReadError

import nmesh
from nmesh.backend import RawMesh
from nmesh.io import load_raw_mesh_with_meshio, meshio_support, save_raw_mesh_with_meshio

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


@pytest.mark.parametrize("suffix", [".vtu", ".msh", ".xdmf"])
def test_nmesh_load_reads_meshio_written_modern_meshes(tmp_path, suffix):
    path = tmp_path / f"triangle{suffix}"
    meshio.write(
        path,
        meshio.Mesh(
            points=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            cells=[("triangle", np.asarray([[0, 1, 2]]))],
        ),
    )

    loaded = nmesh.load(path)

    assert loaded.points == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert loaded.simplices == [[0, 1, 2]]
    assert loaded.raw_mesh.dim == 2


def test_load_combines_same_dimension_blocks_with_aligned_regions(tmp_path, monkeypatch):
    mesh = meshio.Mesh(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        cells=[
            ("triangle", np.asarray([[0, 1, 2]])),
            ("line", np.asarray([[0, 1], [1, 4]])),
            ("triangle", np.asarray([[1, 3, 2], [1, 4, 3]])),
        ],
        cell_data={
            "region": [np.asarray([7]), np.asarray([90, 91]), np.asarray([8, 9])]
        },
    )
    monkeypatch.setattr(meshio_support.meshio, "read", lambda _path: mesh)

    loaded = load_raw_mesh_with_meshio(tmp_path / "multiple.vtu")

    assert loaded.simplices == [[0, 1, 2], [1, 3, 2], [1, 4, 3]]
    assert loaded.regions == [7, 8, 9]
    assert loaded.dim == 2


def test_load_multi_block_mesh_defaults_all_regions(tmp_path, monkeypatch):
    mesh = meshio.Mesh(
        points=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        ),
        cells=[
            ("line", np.asarray([[0, 1]])),
            ("line", np.asarray([[1, 2], [2, 3]])),
        ],
    )
    monkeypatch.setattr(meshio_support.meshio, "read", lambda _path: mesh)

    loaded = load_raw_mesh_with_meshio(tmp_path / "multiple.vtu")

    assert loaded.simplices == [[0, 1], [1, 2], [2, 3]]
    assert loaded.regions == [1, 1, 1]


def test_load_rejects_inconsistent_region_block_length(tmp_path, monkeypatch):
    mesh = meshio.Mesh(
        points=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
        cells=[
            ("triangle", np.asarray([[0, 1, 2]])),
            ("triangle", np.asarray([[1, 3, 2]])),
        ],
        cell_data={"region": [np.asarray([3]), np.asarray([4])]},
    )
    mesh.cell_data["region"][1] = np.asarray([4, 5])
    monkeypatch.setattr(meshio_support.meshio, "read", lambda _path: mesh)

    with pytest.raises(
        ValueError,
        match=r"Region data 'region' for cell block 1 .* has 2 values; expected 1",
    ):
        load_raw_mesh_with_meshio(tmp_path / "invalid.vtu")


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
    assert raw_mesh.dim == 2
    assert raw_mesh.region_volumes == pytest.approx([0.5])


def test_unrecognized_h5_file_retains_meshio_format_error(tmp_path):
    """Unrecognized HDF5 files should retain meshio's format error."""
    path = tmp_path / "invalid.h5"

    # Create an invalid HDF5 file (no mesh group)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("dummy", data=[1, 2, 3])

    with pytest.raises(MeshioReadError, match="Could not deduce file format"):
        load_raw_mesh_with_meshio(path)


def test_non_h5_file_only_tries_meshio(tmp_path):
    """Should only try meshio for non-.h5 files."""
    path = tmp_path / "nonexistent.vtu"

    # This should raise an error from meshio only (not try the legacy loader)
    # meshio._exceptions.ReadError is raised for missing files
    with pytest.raises(MeshioReadError):
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
