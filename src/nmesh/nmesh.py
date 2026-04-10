from __future__ import annotations

import math
import logging
from collections.abc import Iterable, Sequence
from typing import Any, TextIO
from pathlib import Path
import itertools
from . import utils

from .backend import RawMesh, backend
from .mesher import MeshingParameters, make_mg_gendriver
from .io import load_raw_mesh_with_meshio, save_raw_mesh_with_meshio

# Setup logging
log = logging.getLogger(__name__)

PYFEM_SUFFIXES = {"", ".nmesh", ".pyfem"}
Point = list[float]
Simplex = list[int]


def _as_float_points(points: Sequence[Sequence[float]] | None) -> list[Point]:
    return [list(map(float, point)) for point in (points or [])]


def _as_int_simplices(simplices: Sequence[Sequence[int]] | None) -> list[Simplex]:
    return [list(map(int, simplex)) for simplex in (simplices or [])]


def _as_region_ids(regions: Sequence[int] | None) -> list[int]:
    return [int(region) for region in (regions or [])]


def _normalise_periodic(periodic: Sequence[bool] | Sequence[float] | None, dim: int) -> list[float]:
    if not periodic:
        return [0.0] * dim
    return [1.0 if bool(value) else 0.0 for value in periodic]

def memory_report(tag: str):
    """Reports memory usage."""
    t, vmem, rss = utils.time_vmem_rss()
    log.log(15, f"Memory report: T= {t:f} VMEM= {int(vmem)} KB RSS= {int(rss)} KB {tag}")

# --- Configuration ---

def get_default_meshing_parameters():
    """Returns default meshing parameters."""
    return MeshingParameters()

# --- Loading Utilities ---

def _is_nmesh_ascii_file(filename: str | Path) -> bool:
    try:
        with Path(filename).open(encoding="utf-8") as stream:
            return stream.readline().startswith("# PYFEM")
    except OSError:
        return False

def _is_nmesh_hdf5_file(filename: str | Path) -> bool:
    # This would normally use tables.isPyTablesFile
    return str(filename).lower().endswith('.h5')

def hdf5_mesh_get_permutation(filename: str | Path):
    """Stub for retrieving permutation from HDF5."""
    log.warning("hdf5_mesh_get_permutation: HDF5 support is stubbed.")
    return None

# --- Mesh Classes ---

class MeshBase:
    """Base class for all mesh objects, providing access to mesh data."""
    def __init__(self, raw_mesh: RawMesh):
        self.raw_mesh = raw_mesh
        self._cache: dict[str, Any] = {}

    def _cached_backend_value(self, cache_key: str, getter):
        if cache_key not in self._cache:
            self._cache[cache_key] = getter(self.raw_mesh)
        return self._cache[cache_key]

    def scale_node_positions(self, scale: float):
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

    def save(self, filename: str | Path):
        """Saves the mesh to a file (ASCII or HDF5)."""
        path = Path(filename)
        suffix = path.suffix.lower()
        if suffix == ".h5":
            log.info("Saving to HDF5 (stub): %s", path)
            return

        if suffix not in PYFEM_SUFFIXES:
            save_raw_mesh_with_meshio(path, self.raw_mesh)
            return

        if isinstance(self.raw_mesh, RawMesh):
            write_mesh(self.raw_mesh, out=path)
            return

        backend.mesh_writefile(str(path), self.raw_mesh)

    def __str__(self):
        pts = backend.mesh_nr_points(self.raw_mesh)
        simps = backend.mesh_nr_simplices(self.raw_mesh)
        return f"Mesh with {pts} points and {simps} simplices"

    def to_lists(self):
        """Returns mesh data as Python lists."""
        return backend.mesh_plotinfo(self.raw_mesh)

    @property
    def points(self):
        return self._cached_backend_value("points", backend.mesh_plotinfo_points)

    @property
    def simplices(self):
        return self._cached_backend_value("simplices", backend.mesh_plotinfo_simplices)

    @property
    def regions(self):
        return self._cached_backend_value("regions", backend.mesh_plotinfo_simplicesregions)

    @property
    def dim(self):
        return backend.mesh_dim(self.raw_mesh)

    @property
    def surfaces(self):
        return backend.mesh_plotinfo_surfaces_and_surfacesregions(self.raw_mesh)[0]

    @property
    def point_regions(self):
        """Returns regions for each point."""
        return self._cached_backend_value(
            "point_regions", backend.mesh_plotinfo_pointsregions
        )

    @property
    def links(self):
        """Returns all links (pairs of point indices)."""
        return self._cached_backend_value("links", backend.mesh_plotinfo_links)

    @property
    def region_volumes(self):
        """Returns volume of each region."""
        return self._cached_backend_value(
            "region_volumes", backend.mesh_plotinfo_regionvolumes
        )

    @property
    def num_regions(self):
        """Returns the number of regions."""
        return len(self.region_volumes)

    @property
    def periodic_point_indices(self):
        """Returns indices of periodic nodes."""
        return self._cached_backend_value(
            "periodic_indices",
            backend.mesh_plotinfo_periodic_points_indices,
        )

    @property
    def permutation(self):
        """Returns the node permutation mapping."""
        return backend.mesh_get_permutation(self.raw_mesh)

    def set_vertex_distribution(self, dist):
        """Sets vertex distribution."""
        backend.mesh_set_vertex_distribution(self.raw_mesh, dist)

class Mesh(MeshBase):
    """Class for generating a mesh from geometric objects."""
    def __init__(
        self,
        bounding_box,
        objects=None,
        a0=1.0,
        density="",
        periodic=None,
        fixed_points=None,
        mobile_points=None,
        simply_points=None,
        callback=None,
        mesh_bounding_box=False,
        meshing_parameters=None,
        cache_name="",
        hints=None,
        **kwargs,
    ):
        if bounding_box is None:
            raise ValueError("Bounding box must be provided.")

        object_list = list(objects or [])
        hint_list = list(hints or [])
        bb = _as_float_points(bounding_box)
        dim = len(bb[0])
        mesh_ext = 1 if mesh_bounding_box else 0

        self.bounding_box = bb
        self.mesh_exterior = mesh_ext

        if not object_list and not mesh_bounding_box:
            raise ValueError("No objects to mesh and bounding box meshing disabled.")

        if periodic and not mesh_bounding_box and any(periodic):
            raise ValueError(
                "Can only produce periodic meshes when meshing the bounding box."
            )

        params = meshing_parameters or get_default_meshing_parameters()
        self.meshing_parameters = params
        for k, v in kwargs.items():
            params[k] = v

        obj_bodies = []
        self.obj = obj_bodies
        self.cache_name = cache_name
        self.density = density
        self._fixed_points = _as_float_points(fixed_points)
        self._mobile_points = _as_float_points(mobile_points)
        self._simply_points = _as_float_points(simply_points)

        for obj in object_list:
            obj_bodies.append(obj.obj)
            self._fixed_points.extend(obj.fixed_points)
            self._mobile_points.extend(obj.mobile_points)

        resolved_hints = [
            [hint_mesh.raw_mesh, hint_object.obj]
            for hint_mesh, hint_object in hint_list
        ]

        periodic_floats = _normalise_periodic(periodic, dim)
        self.periodic = periodic_floats

        cb_func, cb_interval = (
            callback if callback else (lambda a, b, c: None, 1_000_000)
        )
        self.fun_driver = cb_func
        self.driver = make_mg_gendriver(cb_interval, cb_func)
        self.mesher_config = backend.copy_mesher_defaults(backend.mesher_defaults)
        params.apply_to_mesher(self.mesher_config, dim)

        raw = backend.mesh_bodies_raw(
            self.driver,
            self.mesher_config,
            bb[0],
            bb[1],
            mesh_ext,
            obj_bodies,
            float(a0),
            density,
            self._fixed_points,
            self._mobile_points,
            self._simply_points,
            periodic_floats,
            cache_name,
            resolved_hints,
        )

        if raw is None:
            raise RuntimeError("Mesh generation failed.")
        super().__init__(raw)

    def default_fun(self, nr_piece, n, mesh):
        """Default callback function."""
        pass

    def extended_fun_driver(self, nr_piece, iteration_nr, mesh):
        """Extended driver callback."""
        if hasattr(self, 'fun_driver'):
            self.fun_driver(nr_piece, iteration_nr, mesh)

    def fixed_points(self, points: Sequence[Sequence[float]] | None):
        """Adds fixed points to the mesh configuration."""
        if points:
            self._fixed_points.extend(_as_float_points(points))

    def mobile_points(self, points: Sequence[Sequence[float]] | None):
        """Adds mobile points to the mesh configuration."""
        if points:
            self._mobile_points.extend(_as_float_points(points))

    def simply_points(self, points: Sequence[Sequence[float]] | None):
        """Adds simply points to the mesh configuration."""
        if points:
            self._simply_points.extend(_as_float_points(points))

class MeshFromFile(MeshBase):
    """Loads a mesh from a file."""
    def __init__(self, filename, reorder=False, distribute=True):
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File {filename} not found")

        if _is_nmesh_ascii_file(path):
            raw = backend.mesh_readfile(str(path), reorder, distribute)
        elif _is_nmesh_hdf5_file(path):
            raw = backend.mesh_readfile(str(path), reorder, distribute)
        else:
            raw = load_raw_mesh_with_meshio(path)

        super().__init__(raw)

class mesh_from_points_and_simplices(MeshBase):
    """Wrapper for backward compatibility."""
    def __init__(
        self,
        points=None,
        simplices_indices=None,
        simplices_regions=None,
        periodic_point_indices=None,
        initial=0,
        do_reorder=False,
        do_distribute=True,
    ):
        points_list = _as_float_points(points)
        simplices_list = _as_int_simplices(simplices_indices)
        if initial == 1:
            simplices_list = [[index - 1 for index in simplex] for simplex in simplices_list]

        raw = backend.mesh_from_points_and_simplices(
            len(points_list[0]) if points_list else 3,
            points_list,
            simplices_list,
            _as_region_ids(simplices_regions),
            list(periodic_point_indices or []),
            do_reorder,
            do_distribute,
        )
        super().__init__(raw)

def load(filename, reorder=False, distribute=True):
    """Utility function to load a mesh."""
    return MeshFromFile(filename, reorder, distribute)

def save(mesh: MeshBase, filename: str | Path):
    """Alias for mesh.save for backward compatibility."""
    mesh.save(filename)

# --- Geometry ---

class MeshObject:
    """Base class for geometric primitives and CSG operations."""
    def __init__(
        self,
        dim: int,
        fixed: Sequence[Sequence[float]] | None = None,
        mobile: Sequence[Sequence[float]] | None = None,
    ):
        self.dim = dim
        self.fixed_points = _as_float_points(fixed)
        self.mobile_points = _as_float_points(mobile)
        self.obj: Any = None

    def shift(self, vector, system_coords=True):
        self.obj = (backend.body_shifted_sc if system_coords else backend.body_shifted_bc)(self.obj, vector)
    
    def scale(self, factors):
        self.obj = backend.body_scaled(self.obj, factors)
    
    def rotate(self, a1, a2, angle, system_coords=True):
        rad = math.radians(angle)
        self.obj = (backend.body_rotated_sc if system_coords else backend.body_rotated_bc)(self.obj, a1, a2, rad)

    def rotate_3d(self, axis, angle, system_coords=True):
        rad = math.radians(angle)
        self.obj = (backend.body_rotated_axis_sc if system_coords else backend.body_rotated_axis_bc)(self.obj, axis, rad)

    def transform(self, transformations: Iterable[tuple] | None, system_coords=True):
        """Applies a list of transformation tuples."""
        for t in transformations or []:
            name, *args = t
            match name:
                case "shift":
                    self.shift(args[0], system_coords)
                case "scale":
                    self.scale(args[0])
                case "rotate":
                    self.rotate(args[0][0], args[0][1], args[1], system_coords)
                case "rotate2d":
                    self.rotate(0, 1, args[0], system_coords)
                case "rotate3d":
                    self.rotate_3d(args[0], args[1], system_coords)
                case _:
                    raise ValueError(f"Unknown transformation {name!r}")

class Box(MeshObject):
    def __init__(
        self,
        p1,
        p2,
        transform=None,
        fixed=None,
        mobile=None,
        system_coords=True,
        use_fixed_corners=False,
    ):
        dim = len(p1)
        fixed_points = _as_float_points(fixed)
        if use_fixed_corners:
            fixed_points.extend([list(c) for c in itertools.product(*zip(p1, p2))])
        super().__init__(dim, fixed_points, mobile)
        self.obj = backend.body_box([float(x) for x in p1], [float(x) for x in p2])
        self.transform(transform, system_coords)

class Ellipsoid(MeshObject):
    def __init__(
        self,
        lengths,
        transform=None,
        fixed=None,
        mobile=None,
        system_coords=True,
    ):
        super().__init__(len(lengths), fixed, mobile)
        self.obj = backend.body_ellipsoid([float(x) for x in lengths])
        self.transform(transform, system_coords)

class Conic(MeshObject):
    def __init__(
        self,
        c1,
        r1,
        c2,
        r2,
        transform=None,
        fixed=None,
        mobile=None,
        system_coords=True,
    ):
        super().__init__(len(c1), fixed, mobile)
        self.obj = backend.body_frustum(c1, r1, c2, r2)
        self.transform(transform, system_coords)

class Helix(MeshObject):
    def __init__(
        self,
        c1,
        r1,
        c2,
        r2,
        transform=None,
        fixed=None,
        mobile=None,
        system_coords=True,
    ):
        super().__init__(len(c1), fixed, mobile)
        self.obj = backend.body_helix(c1, r1, c2, r2)
        self.transform(transform, system_coords)

# --- CSG ---

def union(objects: Sequence[MeshObject]) -> MeshObject:
    if len(objects) < 2:
        raise ValueError("Union requires at least two objects")
    res = MeshObject(objects[0].dim)
    for o in objects:
        res.fixed_points.extend(o.fixed_points)
        res.mobile_points.extend(o.mobile_points)
    res.obj = backend.body_union([o.obj for o in objects])
    return res

def difference(mother: MeshObject, subtract: Sequence[MeshObject]) -> MeshObject:
    res = MeshObject(mother.dim, mother.fixed_points[:], mother.mobile_points[:])
    for o in subtract:
        res.fixed_points.extend(o.fixed_points)
        res.mobile_points.extend(o.mobile_points)
    res.obj = backend.body_difference(mother.obj, [o.obj for o in subtract])
    return res

def intersect(objects: Sequence[MeshObject]) -> MeshObject:
    if len(objects) < 2:
        raise ValueError("Intersection requires at least two objects")
    res = MeshObject(objects[0].dim)
    for o in objects:
        res.fixed_points.extend(o.fixed_points)
        res.mobile_points.extend(o.mobile_points)
    res.obj = backend.body_intersection([o.obj for o in objects])
    return res

# --- Utilities ---

def outer_corners(mesh: MeshBase):
    """Determines the bounding box of the mesh nodes."""
    coords = mesh.points
    if not coords:
        return None, None
    transpose = list(zip(*coords))
    return [min(t) for t in transpose], [max(t) for t in transpose]

def generate_1d_mesh_components(
    regions: Sequence[tuple[float, float]],
    discretization: float,
) -> tuple[list[Point], list[Simplex], list[int]]:
    """Generates 1D mesh components (points, simplices, regions)."""
    points: list[Point] = []
    simplices: list[Simplex] = []
    regions_ids: list[int] = []
    point_map: dict[float, int] = {}

    def get_idx(value: float) -> int:
        vk = round(value, 8)
        if vk not in point_map:
            point_map[vk] = len(points)
            points.append([float(value)])
        return point_map[vk]

    for rid, (start, end) in enumerate(regions, 1):
        if start > end:
            start, end = end, start
        steps = max(1, int(abs((end - start) / discretization)))
        step = (end - start) / steps
        last = get_idx(start)
        for i in range(1, steps + 1):
            curr = get_idx(start + i * step)
            simplices.append([last, curr])
            regions_ids.append(rid)
            last = curr

    return points, simplices, regions_ids

def generate_1d_mesh(
    regions: Sequence[tuple[float, float]],
    discretization: float,
) -> MeshBase:
    """Generates a 1D mesh with specified regions and step size."""
    pts, simps, regs = generate_1d_mesh_components(regions, discretization)
    return mesh_from_points_and_simplices(pts, simps, regs)

def to_lists(mesh: MeshBase):
    """Returns mesh data as Python lists."""
    return mesh.to_lists()

tolists = to_lists

def write_mesh(
    mesh_data: RawMesh | tuple[
        Sequence[Sequence[float]],
        Sequence[tuple[int, Sequence[int]]],
        Sequence[tuple[int, Sequence[int]]],
    ],
    out: str | Path | TextIO | None = None,
    check=True,
    float_fmt=" %f",
):
    """
    Writes mesh data to a file in nmesh format.

    `mesh_data` may be a `RawMesh` or a legacy `(points, simplices, surfaces)` tuple.
    """
    if isinstance(mesh_data, RawMesh):
        points = mesh_data.points
        simplices = list(
            zip(
                mesh_data.regions or [1] * len(mesh_data.simplices),
                mesh_data.simplices,
            )
        )
        surfaces = list(
            zip(
                [1] * len(mesh_data.surfaces),
                mesh_data.surfaces,
            )
        )
    else:
        points, simplices, surfaces = mesh_data

    lines = ["# PYFEM mesh file version 1.0"]
    dim = len(points[0]) if points else 0
    lines.append(
        f"# dim = {dim} \t nodes = {len(points)} \t simplices = {len(simplices)} \t surfaces = {len(surfaces)} \t periodic = 0"
    )

    lines.append(str(len(points)))
    for p in points:
        lines.append("".join(float_fmt % x for x in p))

    lines.append(str(len(simplices)))
    for body, nodes in simplices:
        lines.append(f" {body} " + " ".join(str(n) for n in nodes))

    lines.append(str(len(surfaces)))
    for body, nodes in surfaces:
        lines.append(f" {body} " + " ".join(str(n) for n in nodes))

    lines.append("0")

    content = "\n".join(lines) + "\n"

    if out is None:
        print(content)
    elif isinstance(out, (str, Path)):
        Path(out).write_text(content, encoding="utf-8")
    else:
        out.write(content)
