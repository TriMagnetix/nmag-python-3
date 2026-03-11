import logging
import os
from typing import List, Union, Optional
from pathlib import Path
from nmesh.backend import nmesh_backend as backend, RawMesh

log = logging.getLogger(__name__)

class MeshBase:
    """Base class for all mesh objects, providing access to mesh data."""
    def __init__(self, raw_mesh):
        self.raw_mesh = raw_mesh
        self._cache = {}

    def scale_node_positions(self, scale: float):
        """Scales all node positions in the mesh."""
        backend.mesh_scale_node_positions(self.raw_mesh, float(scale))
        self._cache = {} # Clear cache

    def save_hdf5(self, filename):
        log.warning("save_hdf5: HDF5 support not yet implemented.")
        pass

    def save(self, file_name, directory=None, format=None):
        """Saves the mesh to a file."""
        from simulation.features import features
        # In original, output_file_location comes from nsim.snippets
        # Here we simplify it or use Path
        path = str(file_name)
        if directory:
            path = os.path.join(directory, path)
            
        if format == 'hdf5' or path.lower().endswith('.h5'):
            self.save_hdf5(path)
        else:
            backend.mesh_writefile(path, self.raw_mesh)

    def __str__(self):
        pts = backend.mesh_nr_points(self.raw_mesh)
        simps = backend.mesh_nr_simplices(self.raw_mesh)
        return f"Mesh with {pts} points and {simps} simplices"

    def tolists(self):
        """Alias for to_lists for backward compatibility."""
        return self.to_lists()

    def to_lists(self):
        """Returns mesh data as Python lists."""
        return backend.mesh_plotinfo(self.raw_mesh)

    @property
    def points(self):
        if 'points' not in self._cache:
            self._cache['points'] = backend.mesh_plotinfo_points(self.raw_mesh)
        return self._cache['points']

    @property
    def pointsregions(self):
        if 'pointsregions' not in self._cache:
            self._cache['pointsregions'] = backend.mesh_plotinfo_pointsregions(self.raw_mesh)
        return self._cache['pointsregions']
    
    point_regions = pointsregions

    @property
    def simplices(self):
        if 'simplices' not in self._cache:
            self._cache['simplices'] = backend.mesh_plotinfo_simplices(self.raw_mesh)
        return self._cache['simplices']

    @property
    def simplicesregions(self):
        if 'simplicesregions' not in self._cache:
            self._cache['simplicesregions'] = backend.mesh_plotinfo_simplicesregions(self.raw_mesh)
        return self._cache['simplicesregions']
    
    regions = simplicesregions

    @property
    def dim(self):
        return backend.mesh_dim(self.raw_mesh)

    @property
    def surfaces_and_surfacesregions(self):
        if 'surfaces_all' not in self._cache:
            self._cache['surfaces_all'] = backend.mesh_plotinfo_surfaces_and_surfacesregions(self.raw_mesh)
        return self._cache['surfaces_all']

    @property
    def surfaces(self):
        return self.surfaces_and_surfacesregions[0]

    @property
    def surfacesregions(self):
        return self.surfaces_and_surfacesregions[1]

    @property
    def links(self):
        """Returns all links (pairs of point indices)."""
        if 'links' not in self._cache:
            self._cache['links'] = backend.mesh_plotinfo_links(self.raw_mesh)
        return self._cache['links']

    @property
    def regionvolumes(self):
        """Returns volume of each region."""
        if 'regionvolumes' not in self._cache:
            self._cache['regionvolumes'] = backend.mesh_plotinfo_regionvolumes(self.raw_mesh)
        return self._cache['regionvolumes']
    
    region_volumes = regionvolumes

    @property
    def numregions(self):
        """Returns the number of regions."""
        return len(self.region_volumes)
    
    num_regions = numregions

    @property
    def periodicpointindices(self):
        """Returns indices of periodic nodes."""
        if 'periodicpointindices' not in self._cache:
            self._cache['periodicpointindices'] = backend.mesh_plotinfo_periodic_points_indices(self.raw_mesh)
        return self._cache['periodicpointindices']
    
    periodic_point_indices = periodicpointindices

    @property
    def permutation(self):
        """Returns the node permutation mapping."""
        return backend.mesh_get_permutation(self.raw_mesh)

    def set_vertex_distribution(self, dist):
        """Sets vertex distribution."""
        backend.mesh_set_vertex_distribution(self.raw_mesh, dist)

class Mesh(MeshBase):
    """Class for generating a mesh from geometric objects."""
    def __init__(self, bounding_box=None, objects=[], a0=1.0, density="", 
                 periodic=[], fixed_points=[], mobile_points=[], simply_points=[],
                 callback=None, mesh_bounding_box=False, meshing_parameters=None,
                 cache_name="", hints=[], **kwargs):
        
        if bounding_box is None:
            raise ValueError("Bounding box must be provided.")
        
        bb = [[float(x) for x in p] for p in bounding_box]
        dim = len(bb[0])
        mesh_ext = 1 if mesh_bounding_box else 0
        
        from nmesh.features import get_default_meshing_parameters
        params = meshing_parameters or get_default_meshing_parameters()
        for k, v in kwargs.items():
            params[k] = v

        obj_bodies = []
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
        driver = backend.make_mg_gendriver(cb_interval, cb_func)
        mesher = backend.copy_mesher_defaults(backend.mesher_defaults)
        params.pass_parameters_to_ocaml(mesher, dim)

        raw = backend.mesh_bodies_raw(
            driver, mesher, bb[0], bb[1], mesh_ext, obj_bodies, float(a0), 
            density, self._fixed_points, self._mobile_points, self._simply_points, periodic_floats, 
            cache_name, hints
        )
        
        if raw is None: raise RuntimeError("Mesh generation failed.")
        super().__init__(raw)

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
        
        raw = backend.mesh_readfile(str(path), reorder, distribute)
        super().__init__(raw)

class mesh_from_points_and_simplices(MeshBase):
    """Wrapper for backward compatibility."""
    def __init__(self, points=[], simplices_indices=[], simplices_regions=[],
                 periodic_point_indices=[], initial=0, do_reorder=False,
                 do_distribute=True):
        
        if initial == 1:
            simplices_indices = [[idx - 1 for idx in s] for s in simplices_indices]
            
        raw = backend.mesh_from_points_and_simplices(
            len(points[0]) if points else 3,
            [[float(x) for x in p] for p in points],
            [[int(x) for x in s] for s in simplices_indices],
            [int(r) for r in simplices_regions],
            periodic_point_indices, do_reorder, do_distribute
        )
        super().__init__(raw)
