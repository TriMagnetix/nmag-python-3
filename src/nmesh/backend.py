from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class RawMesh:
    points: list[list[float]] = field(default_factory=list)
    simplices: list[list[int]] = field(default_factory=list)
    regions: list[int] = field(default_factory=list)
    point_regions: list[list[int]] = field(default_factory=list)
    surfaces: list[Any] = field(default_factory=list)
    links: list[tuple[int, int]] = field(default_factory=list)
    region_volumes: list[float] = field(default_factory=list)
    periodic_point_indices: list[list[int]] = field(default_factory=list)
    permutation: list[int] = field(default_factory=list)
    dim: int = 3


@runtime_checkable
class MeshBackendProtocol(Protocol):
    def mesh_scale_node_positions(self, raw_mesh: RawMesh, scale: float) -> None: ...
    def mesh_writefile(self, path: str, raw_mesh: RawMesh) -> None: ...
    def mesh_nr_simplices(self, raw_mesh: RawMesh) -> int: ...
    def mesh_nr_points(self, raw_mesh: RawMesh) -> int: ...
    def mesh_plotinfo(self, raw_mesh: RawMesh) -> list[Any]: ...
    def mesh_plotinfo_points(self, raw_mesh: RawMesh) -> list[list[float]]: ...
    def mesh_plotinfo_pointsregions(self, raw_mesh: RawMesh) -> list[list[int]]: ...
    def mesh_plotinfo_simplices(self, raw_mesh: RawMesh) -> list[list[int]]: ...
    def mesh_plotinfo_simplicesregions(self, raw_mesh: RawMesh) -> list[int]: ...
    def mesh_plotinfo_surfaces_and_surfacesregions(self, raw_mesh: RawMesh) -> list[Any]: ...
    def mesh_plotinfo_links(self, raw_mesh: RawMesh) -> list[tuple[int, int]]: ...
    def mesh_dim(self, raw_mesh: RawMesh) -> int: ...
    def mesh_plotinfo_regionvolumes(self, raw_mesh: RawMesh) -> list[float]: ...
    def mesh_plotinfo_periodic_points_indices(self, raw_mesh: RawMesh) -> list[list[int]]: ...
    def mesh_set_vertex_distribution(self, raw_mesh: RawMesh, dist: Any) -> None: ...
    def mesh_get_permutation(self, raw_mesh: RawMesh) -> list[int]: ...
    def mesh_readfile(self, filename: str, do_reorder: bool, do_distribute: bool) -> RawMesh: ...
    def copy_mesher_defaults(self, defaults: dict[str, Any]) -> dict[str, Any]: ...
    def mesh_bodies_raw(
        self,
        driver: Any,
        mesher: dict[str, Any],
        bb_min: Any,
        bb_max: Any,
        mesh_ext: int,
        objects: Any,
        a0: float,
        density: str,
        fixed: Any,
        mobile: Any,
        simply: Any,
        periodic: Any,
        cache: Any,
        hints: Any,
    ) -> RawMesh: ...
    def mesh_from_points_and_simplices(
        self,
        dim: int,
        points: list[list[float]],
        simplices: list[list[int]],
        regions: list[int],
        periodic: list[list[int]],
        reorder: bool,
        distribute: bool,
    ) -> RawMesh: ...
    def body_union(self, objs: Any) -> Any: ...
    def body_difference(self, obj1: Any, objs: Any) -> Any: ...
    def body_intersection(self, objs: Any) -> Any: ...
    def body_shifted_sc(self, obj: Any, shift: Any) -> Any: ...
    def body_shifted_bc(self, obj: Any, shift: Any) -> Any: ...
    def body_scaled(self, obj: Any, scale: Any) -> Any: ...
    def body_rotated_sc(self, obj: Any, a1: Any, a2: Any, ang: Any) -> Any: ...
    def body_rotated_bc(self, obj: Any, a1: Any, a2: Any, ang: Any) -> Any: ...
    def body_rotated_axis_sc(self, obj: Any, axis: Any, ang: Any) -> Any: ...
    def body_rotated_axis_bc(self, obj: Any, axis: Any, ang: Any) -> Any: ...
    def body_box(self, p1: Any, p2: Any) -> Any: ...
    def body_ellipsoid(self, length: Any) -> Any: ...
    def body_frustum(self, c1: Any, r1: Any, c2: Any, r2: Any) -> Any: ...
    def body_helix(self, c1: Any, r1: Any, c2: Any, r2: Any) -> Any: ...

    @property
    def mesher_defaults(self) -> dict[str, Any]: ...


class StubMeshBackend:
    """Lightweight in-memory backend used until the Python mesher is complete."""

    def mesh_scale_node_positions(self, raw_mesh: RawMesh, scale: float):
        for point in raw_mesh.points:
            for index, value in enumerate(point):
                point[index] = value * scale

    def mesh_writefile(self, path: str, raw_mesh: RawMesh):
        return None

    def mesh_nr_simplices(self, raw_mesh: RawMesh) -> int:
        return len(raw_mesh.simplices)

    def mesh_nr_points(self, raw_mesh: RawMesh) -> int:
        return len(raw_mesh.points)

    def mesh_plotinfo(self, raw_mesh: RawMesh):
        return [
            raw_mesh.points,
            raw_mesh.links,
            [raw_mesh.simplices, raw_mesh.point_regions, raw_mesh.regions],
        ]

    def mesh_plotinfo_points(self, raw_mesh: RawMesh):
        return raw_mesh.points

    def mesh_plotinfo_pointsregions(self, raw_mesh: RawMesh):
        return raw_mesh.point_regions

    def mesh_plotinfo_simplices(self, raw_mesh: RawMesh):
        return raw_mesh.simplices

    def mesh_plotinfo_simplicesregions(self, raw_mesh: RawMesh):
        return raw_mesh.regions

    def mesh_plotinfo_surfaces_and_surfacesregions(self, raw_mesh: RawMesh):
        return [raw_mesh.surfaces, []]

    def mesh_plotinfo_links(self, raw_mesh: RawMesh):
        return raw_mesh.links

    def mesh_dim(self, raw_mesh: RawMesh) -> int:
        if raw_mesh.points:
            return len(raw_mesh.points[0])
        return raw_mesh.dim

    def mesh_plotinfo_regionvolumes(self, raw_mesh: RawMesh):
        return raw_mesh.region_volumes

    def mesh_plotinfo_periodic_points_indices(self, raw_mesh: RawMesh):
        return raw_mesh.periodic_point_indices

    def mesh_set_vertex_distribution(self, raw_mesh: RawMesh, dist):
        return None

    def mesh_get_permutation(self, raw_mesh: RawMesh):
        return raw_mesh.permutation

    def mesh_readfile(self, filename: str, do_reorder: bool, do_distribute: bool):
        return RawMesh()

    def copy_mesher_defaults(self, defaults: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(defaults)

    def mesh_bodies_raw(
        self,
        driver,
        mesher: dict[str, Any],
        bb_min,
        bb_max,
        mesh_ext: int,
        objects,
        a0: float,
        density: str,
        fixed,
        mobile,
        simply,
        periodic,
        cache,
        hints,
    ):
        return RawMesh(dim=len(bb_min))

    def mesh_from_points_and_simplices(
        self,
        dim: int,
        points,
        simplices,
        regions,
        periodic,
        reorder: bool,
        distribute: bool,
    ) -> RawMesh:
        return RawMesh(
            points=points,
            simplices=simplices,
            regions=regions,
            dim=dim,
            periodic_point_indices=periodic,
        )

    def body_union(self, objs):
        return ("union", objs)

    def body_difference(self, obj1, objs):
        return ("difference", obj1, objs)

    def body_intersection(self, objs):
        return ("intersection", objs)

    def body_shifted_sc(self, obj, shift):
        return ("shifted_sc", obj, shift)

    def body_shifted_bc(self, obj, shift):
        return ("shifted_bc", obj, shift)

    def body_scaled(self, obj, scale):
        return ("scaled", obj, scale)

    def body_rotated_sc(self, obj, a1, a2, ang):
        return ("rotated_sc", obj, a1, a2, ang)

    def body_rotated_bc(self, obj, a1, a2, ang):
        return ("rotated_bc", obj, a1, a2, ang)

    def body_rotated_axis_sc(self, obj, axis, ang):
        return ("rotated_axis_sc", obj, axis, ang)

    def body_rotated_axis_bc(self, obj, axis, ang):
        return ("rotated_axis_bc", obj, axis, ang)

    def body_box(self, p1, p2):
        return ("box", p1, p2)

    def body_ellipsoid(self, length):
        return ("ellipsoid", length)

    def body_frustum(self, c1, r1, c2, r2):
        return ("frustum", c1, r1, c2, r2)

    def body_helix(self, c1, r1, c2, r2):
        return ("helix", c1, r1, c2, r2)

    @property
    def mesher_defaults(self) -> dict[str, Any]:
        return {"parameters": {}}


backend: MeshBackendProtocol = StubMeshBackend()
