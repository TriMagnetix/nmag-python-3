from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from .geometry import Body

Point: TypeAlias = list[float]
Simplex: TypeAlias = list[int]
Surface: TypeAlias = list[int]
BodyHandle: TypeAlias = object
MesherDriver: TypeAlias = Callable[..., object]
MesherConfig: TypeAlias = dict[str, Any]


@dataclass(slots=True)
class RawMesh:
    points: list[Point] = field(default_factory=list[Point])
    simplices: list[Simplex] = field(default_factory=list[Simplex])
    regions: list[int] = field(default_factory=list[int])
    point_regions: list[list[int]] = field(default_factory=list[list[int]])
    surfaces: list[Surface] = field(default_factory=list[Surface])
    links: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])
    region_volumes: list[float] = field(default_factory=list[float])
    periodic_point_indices: list[list[int]] = field(default_factory=list[list[int]])
    permutation: list[int] = field(default_factory=list[int])
    dim: int = 3


@runtime_checkable
class MeshBackendProtocol(Protocol):
    def mesh_scale_node_positions(self, raw_mesh: RawMesh, scale: float) -> None: ...
    def mesh_writefile(self, path: str, raw_mesh: RawMesh) -> None: ...
    def mesh_nr_simplices(self, raw_mesh: RawMesh) -> int: ...
    def mesh_nr_points(self, raw_mesh: RawMesh) -> int: ...
    def mesh_plotinfo(self, raw_mesh: RawMesh) -> list[object]: ...
    def mesh_plotinfo_points(self, raw_mesh: RawMesh) -> list[list[float]]: ...
    def mesh_plotinfo_pointsregions(self, raw_mesh: RawMesh) -> list[list[int]]: ...
    def mesh_plotinfo_simplices(self, raw_mesh: RawMesh) -> list[list[int]]: ...
    def mesh_plotinfo_simplicesregions(self, raw_mesh: RawMesh) -> list[int]: ...
    def mesh_plotinfo_surfaces_and_surfacesregions(
        self, raw_mesh: RawMesh
    ) -> tuple[list[Surface], list[int]]: ...
    def mesh_plotinfo_links(self, raw_mesh: RawMesh) -> list[tuple[int, int]]: ...
    def mesh_dim(self, raw_mesh: RawMesh) -> int: ...
    def mesh_plotinfo_regionvolumes(self, raw_mesh: RawMesh) -> list[float]: ...
    def mesh_plotinfo_periodic_points_indices(self, raw_mesh: RawMesh) -> list[list[int]]: ...
    def mesh_set_vertex_distribution(self, raw_mesh: RawMesh, dist: object) -> None: ...
    def mesh_get_permutation(self, raw_mesh: RawMesh) -> list[int]: ...
    def mesh_readfile(self, filename: str, do_reorder: bool, do_distribute: bool) -> RawMesh: ...
    def copy_mesher_defaults(self, defaults: MesherConfig) -> MesherConfig: ...
    def mesh_bodies_raw(
        self,
        driver: MesherDriver,
        mesher: MesherConfig,
        bb_min: Point,
        bb_max: Point,
        mesh_ext: int,
        objects: list[Body],
        a0: float,
        density: str,
        fixed: list[Point],
        mobile: list[Point],
        simply: list[Point],
        periodic: list[float],
        cache: str,
        hints: Sequence[Sequence[object]],
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
    def body_union(self, objs: Sequence[BodyHandle]) -> BodyHandle: ...
    def body_difference(self, obj1: BodyHandle, objs: Sequence[BodyHandle]) -> BodyHandle: ...
    def body_intersection(self, objs: Sequence[BodyHandle]) -> BodyHandle: ...
    def body_shifted_sc(self, obj: BodyHandle, shift: Sequence[float]) -> BodyHandle: ...
    def body_shifted_bc(self, obj: BodyHandle, shift: Sequence[float]) -> BodyHandle: ...
    def body_scaled(self, obj: BodyHandle, scale: Sequence[float]) -> BodyHandle: ...
    def body_rotated_sc(self, obj: BodyHandle, a1: int, a2: int, ang: float) -> BodyHandle: ...
    def body_rotated_bc(self, obj: BodyHandle, a1: int, a2: int, ang: float) -> BodyHandle: ...
    def body_rotated_axis_sc(
        self, obj: BodyHandle, axis: Sequence[float], ang: float
    ) -> BodyHandle: ...
    def body_rotated_axis_bc(
        self, obj: BodyHandle, axis: Sequence[float], ang: float
    ) -> BodyHandle: ...
    def body_box(self, p1: Sequence[float], p2: Sequence[float]) -> BodyHandle: ...
    def body_ellipsoid(self, length: Sequence[float]) -> BodyHandle: ...
    def body_frustum(
        self, c1: Sequence[float], r1: float, c2: Sequence[float], r2: float
    ) -> BodyHandle: ...
    def body_helix(
        self, c1: Sequence[float], r1: float, c2: Sequence[float], r2: float
    ) -> BodyHandle: ...

    @property
    def mesher_defaults(self) -> MesherConfig: ...


class StubMeshBackend:
    """Lightweight in-memory backend used until the Python mesher is complete."""

    def mesh_scale_node_positions(self, raw_mesh: RawMesh, scale: float) -> None:
        for point in raw_mesh.points:
            for index, value in enumerate(point):
                point[index] = value * scale
        volume_scale = abs(scale) ** raw_mesh.dim
        raw_mesh.region_volumes = [volume * volume_scale for volume in raw_mesh.region_volumes]

    def mesh_writefile(self, path: str, raw_mesh: RawMesh) -> None:
        return None

    def mesh_nr_simplices(self, raw_mesh: RawMesh) -> int:
        return len(raw_mesh.simplices)

    def mesh_nr_points(self, raw_mesh: RawMesh) -> int:
        return len(raw_mesh.points)

    def mesh_plotinfo(self, raw_mesh: RawMesh) -> list[object]:
        return [
            raw_mesh.points,
            raw_mesh.links,
            [raw_mesh.simplices, raw_mesh.point_regions, raw_mesh.regions],
        ]

    def mesh_plotinfo_points(self, raw_mesh: RawMesh) -> list[Point]:
        return raw_mesh.points

    def mesh_plotinfo_pointsregions(self, raw_mesh: RawMesh) -> list[list[int]]:
        return raw_mesh.point_regions

    def mesh_plotinfo_simplices(self, raw_mesh: RawMesh) -> list[Simplex]:
        return raw_mesh.simplices

    def mesh_plotinfo_simplicesregions(self, raw_mesh: RawMesh) -> list[int]:
        return raw_mesh.regions

    def mesh_plotinfo_surfaces_and_surfacesregions(
        self, raw_mesh: RawMesh
    ) -> tuple[list[Surface], list[int]]:
        return raw_mesh.surfaces, []

    def mesh_plotinfo_links(self, raw_mesh: RawMesh) -> list[tuple[int, int]]:
        return raw_mesh.links

    def mesh_dim(self, raw_mesh: RawMesh) -> int:
        if raw_mesh.points:
            return len(raw_mesh.points[0])
        return raw_mesh.dim

    def mesh_plotinfo_regionvolumes(self, raw_mesh: RawMesh) -> list[float]:
        return raw_mesh.region_volumes

    def mesh_plotinfo_periodic_points_indices(self, raw_mesh: RawMesh) -> list[list[int]]:
        return raw_mesh.periodic_point_indices

    def mesh_set_vertex_distribution(self, raw_mesh: RawMesh, dist: object) -> None:
        raise NotImplementedError("Manual vertex distribution is not implemented.")

    def mesh_get_permutation(self, raw_mesh: RawMesh) -> list[int]:
        return raw_mesh.permutation

    def mesh_readfile(self, filename: str, do_reorder: bool, do_distribute: bool) -> RawMesh:
        return RawMesh()

    def copy_mesher_defaults(self, defaults: MesherConfig) -> MesherConfig:
        return copy.deepcopy(defaults)

    def mesh_bodies_raw(
        self,
        driver: MesherDriver,
        mesher: MesherConfig,
        bb_min: Point,
        bb_max: Point,
        mesh_ext: int,
        objects: list[Body],
        a0: float,
        density: str,
        fixed: list[Point],
        mobile: list[Point],
        simply: list[Point],
        periodic: list[float],
        cache: str,
        hints: Sequence[Sequence[object]],
    ) -> RawMesh:
        from .mesher.relaxation import mesh_bodies_raw as python_mesh_bodies_raw

        return python_mesh_bodies_raw(
            driver,
            mesher,
            bb_min,
            bb_max,
            mesh_ext,
            objects,
            a0,
            density,
            fixed,
            mobile,
            simply,
            periodic,
            [list(hint) for hint in hints],
        )

    def mesh_from_points_and_simplices(
        self,
        dim: int,
        points: list[Point],
        simplices: list[Simplex],
        regions: list[int],
        periodic: list[list[int]],
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

    def body_union(self, objs: Sequence[BodyHandle]) -> BodyHandle:
        return ("union", objs)

    def body_difference(self, obj1: BodyHandle, objs: Sequence[BodyHandle]) -> BodyHandle:
        return ("difference", obj1, objs)

    def body_intersection(self, objs: Sequence[BodyHandle]) -> BodyHandle:
        return ("intersection", objs)

    def body_shifted_sc(self, obj: BodyHandle, shift: Sequence[float]) -> BodyHandle:
        return ("shifted_sc", obj, shift)

    def body_shifted_bc(self, obj: BodyHandle, shift: Sequence[float]) -> BodyHandle:
        return ("shifted_bc", obj, shift)

    def body_scaled(self, obj: BodyHandle, scale: Sequence[float]) -> BodyHandle:
        return ("scaled", obj, scale)

    def body_rotated_sc(self, obj: BodyHandle, a1: int, a2: int, ang: float) -> BodyHandle:
        return ("rotated_sc", obj, a1, a2, ang)

    def body_rotated_bc(self, obj: BodyHandle, a1: int, a2: int, ang: float) -> BodyHandle:
        return ("rotated_bc", obj, a1, a2, ang)

    def body_rotated_axis_sc(
        self, obj: BodyHandle, axis: Sequence[float], ang: float
    ) -> BodyHandle:
        return ("rotated_axis_sc", obj, axis, ang)

    def body_rotated_axis_bc(
        self, obj: BodyHandle, axis: Sequence[float], ang: float
    ) -> BodyHandle:
        return ("rotated_axis_bc", obj, axis, ang)

    def body_box(self, p1: Sequence[float], p2: Sequence[float]) -> BodyHandle:
        return ("box", p1, p2)

    def body_ellipsoid(self, length: Sequence[float]) -> BodyHandle:
        return ("ellipsoid", length)

    def body_frustum(
        self, c1: Sequence[float], r1: float, c2: Sequence[float], r2: float
    ) -> BodyHandle:
        return ("frustum", c1, r1, c2, r2)

    def body_helix(
        self, c1: Sequence[float], r1: float, c2: Sequence[float], r2: float
    ) -> BodyHandle:
        return ("helix", c1, r1, c2, r2)

    @property
    def mesher_defaults(self) -> MesherConfig:
        return {"parameters": {}}


backend: MeshBackendProtocol = StubMeshBackend()
