"""Mesh model wrapper and backend-backed cached accessors."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from .backend import RawMesh, backend

PYFEM_SUFFIXES = {"", ".nmesh", ".pyfem"}
Point = list[float]
Simplex = list[int]
_T = TypeVar("_T")


class MeshBase:
    """Base class for all mesh objects, providing access to mesh data."""

    def __init__(self, raw_mesh: RawMesh) -> None:
        self.raw_mesh = raw_mesh
        self._cache: dict[str, Any] = {}

    def _cached_backend_value(
        self,
        cache_key: str,
        getter: Callable[[RawMesh], _T],
    ) -> _T:
        if cache_key not in self._cache:
            self._cache[cache_key] = getter(self.raw_mesh)
        return cast(_T, self._cache[cache_key])

    def scale_node_positions(self, scale: float) -> None:
        """Scales all node positions in the mesh."""
        backend.mesh_scale_node_positions(self.raw_mesh, float(scale))
        for key in (
            "points",
            "simplices",
            "regions",
            "point_regions",
            "links",
            "region_volumes",
            "periodic_indices",
        ):
            self._cache.pop(key, None)

    def save(self, filename: str | Path) -> None:
        """Saves the mesh to a file (ASCII or HDF5)."""
        path = Path(filename)
        suffix = path.suffix.lower()
        if suffix == ".h5":
            from .io import save_raw_mesh_as_legacy_nmesh_hdf5

            save_raw_mesh_as_legacy_nmesh_hdf5(path, self.raw_mesh)
            return

        if suffix not in PYFEM_SUFFIXES:
            from .io import save_raw_mesh_with_meshio

            save_raw_mesh_with_meshio(path, self.raw_mesh)
            return

        from .mesh_io import write_mesh

        write_mesh(self.raw_mesh, out=path)

    def __str__(self) -> str:
        pts = backend.mesh_nr_points(self.raw_mesh)
        simps = backend.mesh_nr_simplices(self.raw_mesh)
        return f"Mesh with {pts} points and {simps} simplices"

    def to_lists(self) -> list[object]:
        """Returns mesh data as Python lists."""
        return backend.mesh_plotinfo(self.raw_mesh)

    @property
    def points(self) -> list[Point]:
        return self._cached_backend_value("points", backend.mesh_plotinfo_points)

    @property
    def simplices(self) -> list[Simplex]:
        return self._cached_backend_value("simplices", backend.mesh_plotinfo_simplices)

    @property
    def regions(self) -> list[int]:
        return self._cached_backend_value("regions", backend.mesh_plotinfo_simplicesregions)

    @property
    def dim(self) -> int:
        return backend.mesh_dim(self.raw_mesh)

    @property
    def surfaces(self) -> list[list[int]]:
        return backend.mesh_plotinfo_surfaces_and_surfacesregions(self.raw_mesh)[0]

    @property
    def point_regions(self) -> list[list[int]]:
        """Returns regions for each point."""
        return self._cached_backend_value("point_regions", backend.mesh_plotinfo_pointsregions)

    @property
    def links(self) -> list[tuple[int, int]]:
        """Returns all links (pairs of point indices)."""
        return self._cached_backend_value("links", backend.mesh_plotinfo_links)

    @property
    def region_volumes(self) -> list[float]:
        """Returns volume of each region."""
        return self._cached_backend_value("region_volumes", backend.mesh_plotinfo_regionvolumes)

    @property
    def num_regions(self) -> int:
        """Returns the number of regions."""
        return len(self.region_volumes)

    @property
    def periodic_point_indices(self) -> list[list[int]]:
        """Returns indices of periodic nodes."""
        return self._cached_backend_value(
            "periodic_indices",
            backend.mesh_plotinfo_periodic_points_indices,
        )

    @property
    def permutation(self) -> list[int]:
        """Returns the node permutation mapping."""
        return backend.mesh_get_permutation(self.raw_mesh)

    def set_vertex_distribution(self, dist: object) -> None:
        """Sets vertex distribution."""
        backend.mesh_set_vertex_distribution(self.raw_mesh, dist)
