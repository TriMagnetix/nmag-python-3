import math
import logging
from typing import List, Tuple, Optional, Any, Union
from pathlib import Path
import itertools

# Setup logging
log = logging.getLogger(__name__)

# --- Stubs for External Dependencies ---

class OCamlStub:
    """Stub for the OCaml backend interface."""
    def time_vmem_rss(self):
        return 0.0, 0, 0

    # Mesher defaults setters
    def mesher_defaults_set_shape_force_scale(self, mesher, scale): pass
    def mesher_defaults_set_volume_force_scale(self, mesher, scale): pass
    def mesher_defaults_set_neigh_force_scale(self, mesher, scale): pass
    def mesher_defaults_set_irrel_elem_force_scale(self, mesher, scale): pass
    def mesher_defaults_set_time_step_scale(self, mesher, scale): pass
    def mesher_defaults_set_thresh_add(self, mesher, thresh): pass
    def mesher_defaults_set_thresh_del(self, mesher, thresh): pass
    def mesher_defaults_set_topology_threshold(self, mesher, thresh): pass
    def mesher_defaults_set_tolerated_rel_movement(self, mesher, scale): pass
    def mesher_defaults_set_max_relaxation_steps(self, mesher, steps): pass
    def mesher_defaults_set_initial_settling_steps(self, mesher, steps): pass
    def mesher_defaults_set_sliver_correction(self, mesher, scale): pass
    def mesher_defaults_set_smallest_allowed_volume_ratio(self, mesher, scale): pass
    def mesher_defaults_set_movement_max_freedom(self, mesher, scale): pass

    # Mesh operations
    def mesh_scale_node_positions(self, raw_mesh, scale): pass
    def mesh_writefile(self, path, raw_mesh): pass
    def mesh_nr_simplices(self, raw_mesh): return 0
    def mesh_nr_points(self, raw_mesh): return 0
    def mesh_plotinfo(self, raw_mesh): return [[], [], [[], [], []]]
    def mesh_plotinfo_points(self, raw_mesh): return []
    def mesh_plotinfo_pointsregions(self, raw_mesh): return []
    def mesh_plotinfo_simplices(self, raw_mesh): return []
    def mesh_plotinfo_simplicesregions(self, raw_mesh): return []
    def mesh_plotinfo_surfaces_and_surfacesregions(self, raw_mesh): return [[], []]
    def mesh_plotinfo_links(self, raw_mesh): return []
    def mesh_dim(self, raw_mesh): return 3
    def mesh_plotinfo_regionvolumes(self, raw_mesh): return []
    def mesh_plotinfo_periodic_points_indices(self, raw_mesh): return []
    def mesh_set_vertex_distribution(self, raw_mesh, dist): pass
    def mesh_get_permutation(self, raw_mesh): return []
    def mesh_readfile(self, filename, do_reorder, do_distribute): return "STUB_MESH"
    
    # Driver and Mesh creation
    def make_mg_gendriver(self, interval, callback): return "STUB_DRIVER"
    def copy_mesher_defaults(self, defaults): return "STUB_MESHER"
    def mesh_bodies_raw(self, driver, mesher, bb_min, bb_max, mesh_ext, objects, a0, density, fixed, mobile, simply, periodic, cache, hints): return "STUB_MESH"
    def mesh_from_points_and_simplices(self, dim, points, simplices, regions, periodic, reorder, distribute): return "STUB_MESH"

    # Body operations
    def body_union(self, objs): return "STUB_OBJ_UNION"
    def body_difference(self, obj1, objs): return "STUB_OBJ_DIFF"
    def body_intersection(self, objs): return "STUB_OBJ_INTERSECT"
    def body_shifted_sc(self, obj, shift): return obj
    def body_shifted_bc(self, obj, shift): return obj
    def body_scaled(self, obj, scale): return obj
    def body_rotated_sc(self, obj, a1, a2, ang): return obj
    def body_rotated_bc(self, obj, a1, a2, ang): return obj
    def body_rotated_axis_sc(self, obj, axis, ang): return obj
    def body_rotated_axis_bc(self, obj, axis, ang): return obj
    
    # Primitives
    def body_box(self, p1, p2): return "STUB_BOX"
    def body_ellipsoid(self, length): return "STUB_ELLIPSOID"
    def body_frustum(self, c1, r1, c2, r2): return "STUB_FRUSTUM"
    def body_helix(self, c1, r1, c2, r2): return "STUB_HELIX"
    
    @property
    def mesher_defaults(self): return "STUB_DEFAULTS"

ocaml = OCamlStub()

def memory_report(tag: str):
    """Reports memory usage via OCaml backend."""
    t, vmem, rss = ocaml.time_vmem_rss()
    log.log(15, f"Memory report: T= {t:f} VMEM= {int(vmem)} KB RSS= {int(rss)} KB {tag}")

# --- Configuration ---

class FeaturesStub:
    """Stub for nsim.features.Features."""
    def __init__(self, local=True):
        self._data = {}

    def from_file(self, file_path): pass
    def from_string(self, string): pass
    def add_section(self, section):
        if section not in self._data:
            self._data[section] = {}

    def get(self, section, name, raw=False):
        return self._data.get(section, {}).get(name)

    def set(self, section, name, value):
        if section not in self._data:
            self._data[section] = {}
        self._data[section][name] = value

    def items(self, section):
        return self._data.get(section, {}).items()

    def to_string(self):
        return str(self._data)

class MeshingParameters(FeaturesStub):
    """Parameters for the meshing algorithm, supporting multiple dimensions."""
    def __init__(self, string=None, file=None):
        super().__init__(local=True)
        self.dim = None
        if file: self.from_file(file)
        if string: self.from_string(string)
        self.add_section('user-modifications')

    def _get_section_name(self):
        if self.dim is None:
            raise RuntimeError("Dimension not set in MeshingParameters")
        return f'nmesh-{self.dim}D' if self.dim in [2, 3] else 'nmesh-ND'

    def __getitem__(self, name):
        val = self.get('user-modifications', name)
        if val is not None:
            return val
        section = self._get_section_name()
        return self.get(section, name)

    def __setitem__(self, key, value):
        self.set('user-modifications', key, value)

    def set_shape_force_scale(self, v): self["shape_force_scale"] = float(v)
    def set_volume_force_scale(self, v): self["volume_force_scale"] = float(v)
    def set_neigh_force_scale(self, v): self["neigh_force_scale"] = float(v)
    def set_irrel_elem_force_scale(self, v): self["irrel_elem_force_scale"] = float(v)
    def set_time_step_scale(self, v): self["time_step_scale"] = float(v)
    def set_thresh_add(self, v): self["thresh_add"] = float(v)
    def set_thresh_del(self, v): self["thresh_del"] = float(v)
    def set_topology_threshold(self, v): self["topology_threshold"] = float(v)
    def set_tolerated_rel_move(self, v): self["tolerated_rel_move"] = float(v)
    def set_max_steps(self, v): self["max_steps"] = int(v)
    def set_initial_settling_steps(self, v): self["initial_settling_steps"] = int(v)
    def set_sliver_correction(self, v): self["sliver_correction"] = float(v)
    def set_smallest_volume_ratio(self, v): self["smallest_volume_ratio"] = float(v)
    def set_max_relaxation(self, v): self["max_relaxation"] = float(v)

    def pass_parameters_to_ocaml(self, mesher, dim):
        self.dim = dim
        for key, value in self.items('user-modifications'):
            section = self._get_section_name()
            self.set(section, key, str(value))

        params = [
            ("shape_force_scale", ocaml.mesher_defaults_set_shape_force_scale),
            ("volume_force_scale", ocaml.mesher_defaults_set_volume_force_scale),
            ("neigh_force_scale", ocaml.mesher_defaults_set_neigh_force_scale),
            ("irrel_elem_force_scale", ocaml.mesher_defaults_set_irrel_elem_force_scale),
            ("time_step_scale", ocaml.mesher_defaults_set_time_step_scale),
            ("thresh_add", ocaml.mesher_defaults_set_thresh_add),
            ("thresh_del", ocaml.mesher_defaults_set_thresh_del),
            ("topology_threshold", ocaml.mesher_defaults_set_topology_threshold),
            ("tolerated_rel_move", ocaml.mesher_defaults_set_tolerated_rel_movement),
            ("max_steps", ocaml.mesher_defaults_set_max_relaxation_steps),
            ("initial_settling_steps", ocaml.mesher_defaults_set_initial_settling_steps),
            ("sliver_correction", ocaml.mesher_defaults_set_sliver_correction),
            ("smallest_volume_ratio", ocaml.mesher_defaults_set_smallest_allowed_volume_ratio),
            ("max_relaxation", ocaml.mesher_defaults_set_movement_max_freedom),
        ]

        for key, setter in params:
            val = self[key]
            if val is not None:
                setter(mesher, float(val) if "steps" not in key else int(val))

def get_default_meshing_parameters():
    """Returns default meshing parameters."""
    return MeshingParameters()

# --- Loading Utilities ---

def _is_nmesh_ascii_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.readline().startswith("# PYFEM")
    except: return False

def _is_nmesh_hdf5_file(filename):
    # This would normally use tables.isPyTablesFile
    return str(filename).lower().endswith('.h5')

def hdf5_mesh_get_permutation(filename):
    """Stub for retrieving permutation from HDF5."""
    log.warning("hdf5_mesh_get_permutation: HDF5 support is stubbed.")
    return None

# --- Mesh Classes ---

class MeshBase:
    """Base class for all mesh objects, providing access to mesh data."""
    def __init__(self, raw_mesh):
        self.raw_mesh = raw_mesh
        self._cache = {}

    def scale_node_positions(self, scale: float):
        """Scales all node positions in the mesh."""
        ocaml.mesh_scale_node_positions(self.raw_mesh, float(scale))
        self._cache.pop('points', None)
        self._cache.pop('region_volumes', None)

    def save(self, filename: Union[str, Path]):
        """Saves the mesh to a file (ASCII or HDF5)."""
        path = str(filename)
        if path.lower().endswith('.h5'):
            log.info(f"Saving to HDF5 (stub): {path}")
        else:
            ocaml.mesh_writefile(path, self.raw_mesh)

    def __str__(self):
        pts = ocaml.mesh_nr_points(self.raw_mesh)
        simps = ocaml.mesh_nr_simplices(self.raw_mesh)
        return f"Mesh with {pts} points and {simps} simplices"

    def to_lists(self):
        """Returns mesh data as Python lists."""
        return ocaml.mesh_plotinfo(self.raw_mesh)

    @property
    def points(self):
        if 'points' not in self._cache:
            self._cache['points'] = ocaml.mesh_plotinfo_points(self.raw_mesh)
        return self._cache['points']

    @property
    def simplices(self):
        if 'simplices' not in self._cache:
            self._cache['simplices'] = ocaml.mesh_plotinfo_simplices(self.raw_mesh)
        return self._cache['simplices']

    @property
    def regions(self):
        if 'regions' not in self._cache:
            self._cache['regions'] = ocaml.mesh_plotinfo_simplicesregions(self.raw_mesh)
        return self._cache['regions']

    @property
    def dim(self):
        return ocaml.mesh_dim(self.raw_mesh)

    @property
    def surfaces(self):
        return ocaml.mesh_plotinfo_surfaces_and_surfacesregions(self.raw_mesh)[0]

    @property
    def point_regions(self):
        """Returns regions for each point."""
        if 'point_regions' not in self._cache:
            self._cache['point_regions'] = ocaml.mesh_plotinfo_pointsregions(self.raw_mesh)
        return self._cache['point_regions']

    @property
    def links(self):
        """Returns all links (pairs of point indices)."""
        if 'links' not in self._cache:
            self._cache['links'] = ocaml.mesh_plotinfo_links(self.raw_mesh)
        return self._cache['links']

    @property
    def region_volumes(self):
        """Returns volume of each region."""
        if 'region_volumes' not in self._cache:
            self._cache['region_volumes'] = ocaml.mesh_plotinfo_regionvolumes(self.raw_mesh)
        return self._cache['region_volumes']

    @property
    def num_regions(self):
        """Returns the number of regions."""
        return len(self.region_volumes)

    @property
    def periodic_point_indices(self):
        """Returns indices of periodic nodes."""
        if 'periodic_indices' not in self._cache:
            self._cache['periodic_indices'] = ocaml.mesh_plotinfo_periodic_points_indices(self.raw_mesh)
        return self._cache['periodic_indices']

    @property
    def permutation(self):
        """Returns the node permutation mapping."""
        return ocaml.mesh_get_permutation(self.raw_mesh)

    def set_vertex_distribution(self, dist):
        """Sets vertex distribution."""
        ocaml.mesh_set_vertex_distribution(self.raw_mesh, dist)

class Mesh(MeshBase):
    """Class for generating a mesh from geometric objects."""
    def __init__(self, bounding_box, objects=[], a0=1.0, density="", 
                 periodic=[], fixed_points=[], mobile_points=[], simply_points=[],
                 callback=None, mesh_bounding_box=False, meshing_parameters=None,
                 cache_name="", hints=[], **kwargs):
        
        if bounding_box is None:
            raise ValueError("Bounding box must be provided.")
        
        bb = [[float(x) for x in p] for p in bounding_box]
        dim = len(bb[0])
        mesh_ext = 1 if mesh_bounding_box else 0
        
        if not objects and not mesh_bounding_box:
            raise ValueError("No objects to mesh and bounding box meshing disabled.")

        params = meshing_parameters or get_default_meshing_parameters()
        for k, v in kwargs.items():
            params[k] = v

        obj_bodies = []
        # Store points lists to allow appending later (mimicking original API)
        self._fixed_points = [list(map(float, p)) for p in fixed_points]
        self._mobile_points = [list(map(float, p)) for p in mobile_points]
        self._simply_points = [list(map(float, p)) for p in simply_points]
        
        for obj in objects:
            obj_bodies.append(obj.obj)
            self._fixed_points.extend(obj.fixed_points)
            self._mobile_points.extend(obj.mobile_points)

        periodic_floats = [1.0 if p else 0.0 for p in periodic] if periodic else [0.0] * dim
        
        cb_func, cb_interval = callback if callback else (lambda a,b,c: None, 1000000)
        self.fun_driver = cb_func
        driver = ocaml.make_mg_gendriver(cb_interval, cb_func)
        mesher = ocaml.copy_mesher_defaults(ocaml.mesher_defaults)
        params.pass_parameters_to_ocaml(mesher, dim)

        # Note: In the original code, mesh generation happens in __init__. 
        # Adding points via methods afterwards wouldn't affect the already generated mesh 
        # unless we regenerate or if those methods were meant for pre-generation setup.
        # However, checking lib1.py, __init__ calls mesh_bodies_raw immediately.
        # The methods fixed_points/mobile_points in lib1.py just append to self.fixed_points
        # which seems useless after __init__ unless the user manually triggers something else.
        # But we will preserve them for API compatibility.

        raw = ocaml.mesh_bodies_raw(
            driver, mesher, bb[0], bb[1], mesh_ext, obj_bodies, float(a0), 
            density, self._fixed_points, self._mobile_points, self._simply_points, periodic_floats, 
            cache_name, hints
        )
        
        if raw is None: raise RuntimeError("Mesh generation failed.")
        super().__init__(raw)

    def default_fun(self, nr_piece, n, mesh):
        """Default callback function."""
        pass

    def extended_fun_driver(self, nr_piece, iteration_nr, mesh):
        """Extended driver callback."""
        if hasattr(self, 'fun_driver'):
            self.fun_driver(nr_piece, iteration_nr, mesh)

    def fixed_points(self, points: List[List[float]]):
        """Adds fixed points to the mesh configuration."""
        if points:
            self._fixed_points.extend(points)

    def mobile_points(self, points: List[List[float]]):
        """Adds mobile points to the mesh configuration."""
        if points:
            self._mobile_points.extend(points)

    def simply_points(self, points: List[List[float]]):
        """Adds simply points to the mesh configuration."""
        if points:
            self._simply_points.extend(points)

class MeshFromFile(MeshBase):
    """Loads a mesh from a file."""
    def __init__(self, filename, reorder=False, distribute=True):
        path = Path(filename)
        if not path.exists(): raise FileNotFoundError(f"File {filename} not found")
        
        # Determine format
        if _is_nmesh_ascii_file(filename):
            raw = ocaml.mesh_readfile(str(path), reorder, distribute)
        elif _is_nmesh_hdf5_file(filename):
            # load_hdf5 logic would go here
            raw = ocaml.mesh_readfile(str(path), reorder, distribute) 
        else:
            raise ValueError(f"Unknown mesh file format: {filename}")
            
        super().__init__(raw)

class mesh_from_points_and_simplices(MeshBase):
    """Wrapper for backward compatibility."""
    def __init__(self, points=[], simplices_indices=[], simplices_regions=[],
                 periodic_point_indices=[], initial=0, do_reorder=False,
                 do_distribute=True):
        
        # Adjust for 1-based indexing if initial=1
        if initial == 1:
            simplices_indices = [[idx - 1 for idx in s] for s in simplices_indices]
            
        raw = ocaml.mesh_from_points_and_simplices(
            len(points[0]) if points else 3,
            [[float(x) for x in p] for p in points],
            [[int(x) for x in s] for s in simplices_indices],
            [int(r) for r in simplices_regions],
            periodic_point_indices, do_reorder, do_distribute
        )
        super().__init__(raw)

def load(filename, reorder=False, distribute=True):
    """Utility function to load a mesh."""
    return MeshFromFile(filename, reorder, distribute)

def save(mesh: MeshBase, filename: Union[str, Path]):
    """Alias for mesh.save for backward compatibility."""
    mesh.save(filename)

# --- Exception Aliases ---
NmeshUserError = ValueError
NmeshIOError = IOError
NmeshStandardError = RuntimeError

# --- Geometry ---

class MeshObject:
    """Base class for geometric primitives and CSG operations."""
    def __init__(self, dim, fixed=[], mobile=[]):
        self.dim = dim
        self.fixed_points = fixed
        self.mobile_points = mobile
        self.obj: Any = None

    def shift(self, vector, system_coords=True):
        self.obj = (ocaml.body_shifted_sc if system_coords else ocaml.body_shifted_bc)(self.obj, vector)
    
    def scale(self, factors):
        self.obj = ocaml.body_scaled(self.obj, factors)
    
    def rotate(self, a1, a2, angle, system_coords=True):
        rad = math.radians(angle)
        self.obj = (ocaml.body_rotated_sc if system_coords else ocaml.body_rotated_bc)(self.obj, a1, a2, rad)

    def rotate_3d(self, axis, angle, system_coords=True):
        rad = math.radians(angle)
        self.obj = (ocaml.body_rotated_axis_sc if system_coords else ocaml.body_rotated_axis_bc)(self.obj, axis, rad)

    def transform(self, transformations, system_coords=True):
        """Applies a list of transformation tuples."""
        for t in transformations:
            name, *args = t
            if name == "shift": self.shift(args[0], system_coords)
            elif name == "scale": self.scale(args[0])
            elif name == "rotate": self.rotate(args[0][0], args[0][1], args[1], system_coords)
            elif name == "rotate2d": self.rotate(0, 1, args[0], system_coords)
            elif name == "rotate3d": self.rotate_3d(args[0], args[1], system_coords)

class Box(MeshObject):
    def __init__(self, p1, p2, transform=[], fixed=[], mobile=[], system_coords=True, use_fixed_corners=False):
        dim = len(p1)
        if use_fixed_corners:
            fixed.extend([list(c) for c in itertools.product(*zip(p1, p2))])
        super().__init__(dim, fixed, mobile)
        self.obj = ocaml.body_box([float(x) for x in p1], [float(x) for x in p2])
        self.transform(transform, system_coords)

class Ellipsoid(MeshObject):
    def __init__(self, lengths, transform=[], fixed=[], mobile=[], system_coords=True):
        super().__init__(len(lengths), fixed, mobile)
        self.obj = ocaml.body_ellipsoid([float(x) for x in lengths])
        self.transform(transform, system_coords)

class Conic(MeshObject):
    def __init__(self, c1, r1, c2, r2, transform=[], fixed=[], mobile=[], system_coords=True):
        super().__init__(len(c1), fixed, mobile)
        self.obj = ocaml.body_frustum(c1, r1, c2, r2)
        self.transform(transform, system_coords)

class Helix(MeshObject):
    def __init__(self, c1, r1, c2, r2, transform=[], fixed=[], mobile=[], system_coords=True):
        super().__init__(len(c1), fixed, mobile)
        self.obj = ocaml.body_helix(c1, r1, c2, r2)
        self.transform(transform, system_coords)

# --- CSG ---

def union(objects: List[MeshObject]) -> MeshObject:
    if len(objects) < 2: raise ValueError("Union requires at least two objects")
    res = MeshObject(objects[0].dim)
    for o in objects:
        res.fixed_points.extend(o.fixed_points)
        res.mobile_points.extend(o.mobile_points)
    res.obj = ocaml.body_union([o.obj for o in objects])
    return res

def difference(mother: MeshObject, subtract: List[MeshObject]) -> MeshObject:
    res = MeshObject(mother.dim, mother.fixed_points[:], mother.mobile_points[:])
    for o in subtract:
        res.fixed_points.extend(o.fixed_points)
        res.mobile_points.extend(o.mobile_points)
    res.obj = ocaml.body_difference(mother.obj, [o.obj for o in subtract])
    return res

def intersect(objects: List[MeshObject]) -> MeshObject:
    if len(objects) < 2: raise ValueError("Intersection requires at least two objects")
    res = MeshObject(objects[0].dim)
    for o in objects:
        res.fixed_points.extend(o.fixed_points)
        res.mobile_points.extend(o.mobile_points)
    res.obj = ocaml.body_intersection([o.obj for o in objects])
    return res

# --- Utilities ---

def outer_corners(mesh: MeshBase):
    """Determines the bounding box of the mesh nodes."""
    coords = mesh.points
    if not coords: return None, None
    transpose = list(zip(*coords))
    return [min(t) for t in transpose], [max(t) for t in transpose]

def generate_1d_mesh_components(regions: List[Tuple[float, float]], discretization: float) -> Tuple:
    """Generates 1D mesh components (points, simplices, regions)."""
    points, simplices, regions_ids = [], [], []
    point_map = {}
    
    def get_idx(v):
        vk = round(v, 8)
        if vk not in point_map:
            point_map[vk] = len(points)
            points.append([float(v)])
        return point_map[vk]

    for rid, (start, end) in enumerate(regions, 1):
        if start > end: start, end = end, start
        steps = max(1, int(abs((end - start) / discretization)))
        step = (end - start) / steps
        last = get_idx(start)
        for i in range(1, steps + 1):
            curr = get_idx(start + i * step)
            simplices.append([last, curr])
            regions_ids.append(rid)
            last = curr
            
    # Note: original unidmesher also returned surfaces, but simplified here
    # Standard format for mesh_from_points_and_simplices: 
    # simplices are list of point indices, regions are separate list
    return points, simplices, regions_ids

def generate_1d_mesh(regions: List[Tuple[float, float]], discretization: float) -> MeshBase:
    """Generates a 1D mesh with specified regions and step size."""
    pts, simps, regs = generate_1d_mesh_components(regions, discretization)
    return mesh_from_points_and_simplices(pts, simps, regs)

def to_lists(mesh: MeshBase):
    """Returns mesh data as Python lists."""
    return mesh.to_lists()

tolists = to_lists

def write_mesh(mesh_data, out=None, check=True, float_fmt=" %f"):
    """
    Writes mesh data (points, simplices, surfaces) to a file in nmesh format.
    mesh_data: (points, simplices, surfaces)
    """
    points, simplices, surfaces = mesh_data
    
    lines = ["# PYFEM mesh file version 1.0"]
    dim = len(points[0]) if points else 0
    lines.append(f"# dim = {dim} \t nodes = {len(points)} \t simplices = {len(simplices)} \t surfaces = {len(surfaces)} \t periodic = 0")
    
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
        Path(out).write_text(content)
    else:
        out.write(content)
