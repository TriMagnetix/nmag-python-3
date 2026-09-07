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
    """Expose mesh geometry and topology through cached Python values.

    Attributes:
        points: Node coordinates.
        simplices: Point indices for every simplex cell.
        regions: Region ID for every simplex.
        dim: Coordinate dimension.
        surfaces: Boundary simplex indices.
        point_regions: Incident region IDs for every point.
        links: Unique mesh edges as point-index pairs.
        region_volumes: Computed volume for each region.
    """

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
        """Multiply every node coordinate by ``scale`` and clear caches."""
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
        """Save the mesh using the format selected by the filename suffix."""
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
        """Return the backend's legacy mesh-data list representation."""
        return backend.mesh_plotinfo(self.raw_mesh)

    @property
    def points(self) -> list[Point]:
        """Return node coordinates."""
        return self._cached_backend_value("points", backend.mesh_plotinfo_points)

    @property
    def simplices(self) -> list[Simplex]:
        """Return point indices for every simplex cell."""
        return self._cached_backend_value("simplices", backend.mesh_plotinfo_simplices)

    @property
    def regions(self) -> list[int]:
        """Return the region ID for every simplex cell."""
        return self._cached_backend_value("regions", backend.mesh_plotinfo_simplicesregions)

    @property
    def dim(self) -> int:
        """Return the mesh coordinate dimension."""
        return backend.mesh_dim(self.raw_mesh)

    @property
    def surfaces(self) -> list[list[int]]:
        """Return point indices for detected boundary simplices."""
        return backend.mesh_plotinfo_surfaces_and_surfacesregions(self.raw_mesh)[0]

    @property
    def point_regions(self) -> list[list[int]]:
        """Return incident region IDs for each point."""
        return self._cached_backend_value("point_regions", backend.mesh_plotinfo_pointsregions)

    @property
    def links(self) -> list[tuple[int, int]]:
        """Return unique mesh edges as point-index pairs."""
        return self._cached_backend_value("links", backend.mesh_plotinfo_links)

    @property
    def region_volumes(self) -> list[float]:
        """Return geometric volume for each region."""
        return self._cached_backend_value("region_volumes", backend.mesh_plotinfo_regionvolumes)

    @property
    def num_regions(self) -> int:
        """Return the number of mesh regions."""
        return len(self.region_volumes)

    @property
    def periodic_point_indices(self) -> list[list[int]]:
        """Return groups of equivalent periodic point indices."""
        return self._cached_backend_value(
            "periodic_indices",
            backend.mesh_plotinfo_periodic_points_indices,
        )

    @property
    def permutation(self) -> list[int]:
        """Return the node permutation mapping."""
        return backend.mesh_get_permutation(self.raw_mesh)

    def set_vertex_distribution(self, dist: object) -> None:
        """Sets vertex distribution."""
        backend.mesh_set_vertex_distribution(self.raw_mesh, dist)
