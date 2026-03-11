import math
import logging
import time
import itertools
import numpy as np
import re
import copy
from pathlib import Path

log = logging.getLogger(__name__)

class MesherDefaults:
    def __init__(self):
        # Default values from nmag-src/src/mesh.ml
        self.mdefault_controller_initial_points_volume_ratio = 0.9
        self.mdefault_controller_splitting_connection_ratio = 1.6
        self.mdefault_controller_exp_neigh_force_scale = 0.9
        self.mdefault_nr_probes_for_determining_volume = 100000
        self.mdefault_boundary_condition_acceptable_fuzz = 1.0e-6
        self.mdefault_boundary_condition_max_nr_correction_steps = 200
        self.mdefault_boundary_condition_debuglevel = 0
        self.mdefault_relaxation_debuglevel = 0
        self.mdefault_controller_movement_max_freedom = 3.0
        self.mdefault_controller_topology_threshold = 0.2
        self.mdefault_controller_step_limit_min = 500
        self.mdefault_controller_step_limit_max = 1000
        self.mdefault_controller_max_time_step = 10.0
        self.mdefault_controller_time_step_scale = 0.1
        self.mdefault_controller_tolerated_rel_movement = 0.002
        self.mdefault_controller_shape_force_scale = 0.1
        self.mdefault_controller_volume_force_scale = 0.0
        self.mdefault_controller_neigh_force_scale = 1.0
        self.mdefault_controller_irrel_elem_force_scale = 1.0
        self.mdefault_controller_initial_settling_steps = 100
        self.mdefault_controller_thresh_add = 1.0
        self.mdefault_controller_thresh_del = 2.0
        self.mdefault_controller_sliver_correction = 1.0
        self.mdefault_controller_smallest_allowed_volume_ratio = 1.0

    def mesher_defaults_set_shape_force_scale(self, v): self.mdefault_controller_shape_force_scale = v
    def mesher_defaults_set_volume_force_scale(self, v): self.mdefault_controller_volume_force_scale = v
    def mesher_defaults_set_neigh_force_scale(self, v): self.mdefault_controller_neigh_force_scale = v
    def mesher_defaults_set_irrel_elem_force_scale(self, v): self.mdefault_controller_irrel_elem_force_scale = v
    def mesher_defaults_set_time_step_scale(self, v): self.mdefault_controller_time_step_scale = v
    def mesher_defaults_set_thresh_add(self, v): self.mdefault_controller_thresh_add = v
    def mesher_defaults_set_thresh_del(self, v): self.mdefault_controller_thresh_del = v
    def mesher_defaults_set_topology_threshold(self, v): self.mdefault_controller_topology_threshold = v
    def mesher_defaults_set_tolerated_rel_movement(self, v): self.mdefault_controller_tolerated_rel_movement = v
    def mesher_defaults_set_max_relaxation_steps(self, v): self.mdefault_controller_step_limit_max = v
    def mesher_defaults_set_initial_settling_steps(self, v): self.mdefault_controller_initial_settling_steps = v
    def mesher_defaults_set_sliver_correction(self, v): self.mdefault_controller_sliver_correction = v
    def mesher_defaults_set_smallest_allowed_volume_ratio(self, v): self.mdefault_controller_smallest_allowed_volume_ratio = v
    def mesher_defaults_set_movement_max_freedom(self, v): self.mdefault_controller_movement_max_freedom = v

class RawMesh:
    """Internal representation of a mesh."""
    def __init__(self, dim, points, simplices, regions, periodic_points=None, permutation=None):
        self.dim = dim
        self.points = points  # List[List[float]]
        self.simplices = simplices  # List[List[int]]
        self.regions = regions  # List[int]
        self.periodic_points = periodic_points or []
        self.permutation = permutation
        self._links = None
        self._surfaces = None
        self._surfaces_regions = None
        self._region_volumes = None
        self._point_regions = None

    def get_links(self):
        if self._links is None:
            links_set = set()
            for sx in self.simplices:
                for p1, p2 in itertools.combinations(sx, 2):
                    links_set.add(tuple(sorted((p1, p2))))
            self._links = [list(link) for link in links_set]
        return self._links

    def get_surfaces(self):
        if self._surfaces is None:
            faces = {}
            for i, sx in enumerate(self.simplices):
                region = self.regions[i]
                for j in range(len(sx)):
                    face = tuple(sorted(sx[:j] + sx[j+1:]))
                    if face not in faces:
                        faces[face] = []
                    faces[face].append(region)
            
            self._surfaces = []
            self._surfaces_regions = []
            for face, regs in faces.items():
                if len(regs) == 1:
                    self._surfaces.append(list(face))
                    self._surfaces_regions.append(regs[0])
                elif len(regs) == 2 and regs[0] != regs[1]:
                    self._surfaces.append(list(face))
                    self._surfaces_regions.append(regs[0])
                    self._surfaces.append(list(face))
                    self._surfaces_regions.append(regs[1])
        return self._surfaces, self._surfaces_regions

    def get_region_volumes(self):
        if self._region_volumes is None:
            if not self.simplices:
                self._region_volumes = []
                return self._region_volumes
            
            max_reg = max(self.regions) if self.regions else 0
            volumes = [0.0] * (max_reg + 1)
            
            fact_dim = math.factorial(self.dim)
            for i, sx in enumerate(self.simplices):
                reg = self.regions[i]
                pts = [self.points[idx] for idx in sx]
                if len(pts) > 1:
                    mat = np.array([np.array(pts[j]) - np.array(pts[0]) for j in range(1, len(pts))])
                    vol = abs(np.linalg.det(mat)) / fact_dim
                    if reg >= 0:
                        volumes[reg] += vol
            self._region_volumes = volumes
        return self._region_volumes

    def get_point_regions(self):
        if self._point_regions is None:
            pt_regs = [set() for _ in range(len(self.points))]
            for i, sx in enumerate(self.simplices):
                reg = self.regions[i]
                for pt_idx in sx:
                    pt_regs[pt_idx].add(reg)
            self._point_regions = [list(regs) for regs in pt_regs]
        return self._point_regions

class AffineTrafo:
    def __init__(self, dim, matrix=None, displacement=None):
        self.dim = dim
        self.matrix = matrix if matrix is not None else np.eye(dim)
        self.displacement = displacement if displacement is not None else np.zeros(dim)

    def combine(self, other: 'AffineTrafo'):
        new_matrix = np.dot(self.matrix, other.matrix)
        new_displacement = self.displacement + np.dot(self.matrix, other.displacement)
        return AffineTrafo(self.dim, new_matrix, new_displacement)

    def apply_to_pos(self, pos):
        return np.dot(self.matrix, np.array(pos)) + self.displacement

class Body:
    """Representation of a geometric body with a boundary condition."""
    def __init__(self, trafo: AffineTrafo, bc_func):
        self.trafo = trafo
        self.bc_func = bc_func

    def __call__(self, pos):
        return self.bc_func(self.trafo.apply_to_pos(pos))

class NMeshBackend:
    """Implementation of the NMesh backend in Python."""
    def __init__(self):
        self._mesher_defaults = MesherDefaults()

    def time_vmem_rss(self):
        try:
            import psutil
            process = psutil.Process()
            mem = process.memory_info()
            return time.time(), mem.vms / 1024.0, mem.rss / 1024.0
        except ImportError:
            return time.time(), 0.0, 0.0

    # Mesh operations
    def mesh_scale_node_positions(self, raw_mesh, scale):
        for p in raw_mesh.points:
            for i in range(len(p)):
                p[i] *= float(scale)
        raw_mesh._region_volumes = None

    def mesh_writefile(self, path, raw_mesh):
        from .utils import write_mesh
        write_mesh((raw_mesh.points, list(zip(raw_mesh.regions, raw_mesh.simplices)), []), out=path)

    def mesh_nr_simplices(self, raw_mesh):
        return len(raw_mesh.simplices)

    def mesh_nr_points(self, raw_mesh):
        return len(raw_mesh.points)

    def mesh_plotinfo(self, raw_mesh):
        surfaces, _ = raw_mesh.get_surfaces()
        simps_info = []
        for i, sx in enumerate(raw_mesh.simplices):
            simps_info.append([sx, [[[], 0.0], [[], 0.0], raw_mesh.regions[i]]])
        return [raw_mesh.points, raw_mesh.get_links(), simps_info, raw_mesh.get_point_regions()]

    def mesh_plotinfo_points(self, raw_mesh):
        return raw_mesh.points

    def mesh_plotinfo_pointsregions(self, raw_mesh):
        return raw_mesh.get_point_regions()

    def mesh_plotinfo_simplices(self, raw_mesh):
        return raw_mesh.simplices

    def mesh_plotinfo_simplicesregions(self, raw_mesh):
        return raw_mesh.regions

    def mesh_plotinfo_surfaces_and_surfacesregions(self, raw_mesh):
        return raw_mesh.get_surfaces()

    def mesh_plotinfo_links(self, raw_mesh):
        return raw_mesh.get_links()

    def mesh_dim(self, raw_mesh):
        return raw_mesh.dim

    def mesh_plotinfo_regionvolumes(self, raw_mesh):
        return raw_mesh.get_region_volumes()

    def mesh_plotinfo_periodic_points_indices(self, raw_mesh):
        return raw_mesh.periodic_points

    def mesh_set_vertex_distribution(self, raw_mesh, dist):
        pass

    def mesh_get_permutation(self, raw_mesh):
        return raw_mesh.permutation or list(range(len(raw_mesh.points)))

    def mesh_readfile(self, filename, do_reorder, do_distribute):
        path = Path(filename)
        if not path.exists(): raise FileNotFoundError(f"File {filename} not found")
        with open(path, 'r') as f:
            lines = f.readlines()
        if not lines or not lines[0].startswith("# PYFEM"):
            raise ValueError(f"Invalid mesh file: {filename}")
        m = re.search(r"dim\s*=\s*(\d+)\s*nodes\s*=\s*(\d+)\s*simplices\s*=\s*(\d+)", lines[1])
        if not m: raise ValueError(f"Invalid header in mesh file: {filename}")
        dim = int(m.group(1))
        ptr = 2
        while ptr < len(lines) and (lines[ptr].strip().startswith("#") or not lines[ptr].strip()):
            ptr += 1
        n_pts = int(lines[ptr].strip())
        ptr += 1
        points = []
        for _ in range(n_pts):
            points.append([float(x) for x in lines[ptr].split()])
            ptr += 1
        n_simps = int(lines[ptr].strip())
        ptr += 1
        simplices = []
        regions = []
        for _ in range(n_simps):
            parts = [float(x) for x in lines[ptr].split()]
            regions.append(int(parts[0]))
            simplices.append([int(x) for x in parts[1:]])
            ptr += 1
        return RawMesh(dim, points, simplices, regions)
    
    # Driver and Mesh creation
    def make_mg_gendriver(self, interval, callback):
        return (interval, callback)
    
    def symm_grad(self, f, x, epsilon=1e-7):
        dim = len(x)
        grad = np.zeros(dim)
        for i in range(dim):
            x_plus = np.array(x, dtype=float)
            x_minus = np.array(x, dtype=float)
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * epsilon)
        return grad

    def _enforce_boundary_conditions(self, mesher_defaults, bcs, coords):
        acceptable_fuzz = mesher_defaults.mdefault_boundary_condition_acceptable_fuzz
        max_steps = mesher_defaults.mdefault_boundary_condition_max_nr_correction_steps
        for _ in range(max_steps):
            violated_idx = -1
            for i, bc in enumerate(bcs):
                if bc(coords) < -acceptable_fuzz:
                    violated_idx = i
                    break
            if violated_idx == -1:
                return True
            bc = bcs[violated_idx]
            val = bc(coords)
            grad = self.symm_grad(bc, coords)
            grad_sq = np.sum(grad**2)
            if grad_sq < 1e-12:
                break
            scale = -val / grad_sq
            coords += scale * grad
        return False

    def _enforce_boundary_conditions_reversed(self, mesher_defaults, bcs, coords):
        acceptable_fuzz = mesher_defaults.mdefault_boundary_condition_acceptable_fuzz
        max_steps = mesher_defaults.mdefault_boundary_condition_max_nr_correction_steps
        for _ in range(max_steps):
            violated_idx = -1
            for i, bc in enumerate(bcs):
                if bc(coords) > acceptable_fuzz:
                    violated_idx = i
                    break
            if violated_idx == -1:
                return True
            bc = bcs[violated_idx]
            val = bc(coords)
            grad = self.symm_grad(bc, coords)
            grad_sq = np.sum(grad**2)
            if grad_sq < 1e-12:
                break
            scale = -val / grad_sq
            coords += scale * grad
        return False

    def mesher_defaults_set_shape_force_scale(self, m, v): m.mesher_defaults_set_shape_force_scale(v)
    def mesher_defaults_set_volume_force_scale(self, m, v): m.mesher_defaults_set_volume_force_scale(v)
    def mesher_defaults_set_neigh_force_scale(self, m, v): m.mesher_defaults_set_neigh_force_scale(v)
    def mesher_defaults_set_irrel_elem_force_scale(self, m, v): m.mesher_defaults_set_irrel_elem_force_scale(v)
    def mesher_defaults_set_time_step_scale(self, m, v): m.mesher_defaults_set_time_step_scale(v)
    def mesher_defaults_set_thresh_add(self, m, v): m.mesher_defaults_set_thresh_add(v)
    def mesher_defaults_set_thresh_del(self, m, v): m.mesher_defaults_set_thresh_del(v)
    def mesher_defaults_set_topology_threshold(self, m, v): m.mesher_defaults_set_topology_threshold(v)
    def mesher_defaults_set_tolerated_rel_movement(self, m, v): m.mesher_defaults_set_tolerated_rel_movement(v)
    def mesher_defaults_set_max_relaxation_steps(self, m, v): m.mesher_defaults_set_max_relaxation_steps(v)
    def mesher_defaults_set_initial_settling_steps(self, m, v): m.mesher_defaults_set_initial_settling_steps(v)
    def mesher_defaults_set_sliver_correction(self, m, v): m.mesher_defaults_set_sliver_correction(v)
    def mesher_defaults_set_smallest_allowed_volume_ratio(self, m, v): m.mesher_defaults_set_smallest_allowed_volume_ratio(v)
    def mesher_defaults_set_movement_max_freedom(self, m, v): m.mesher_defaults_set_movement_max_freedom(v)
        
    def _all_combinations(self, n):
        comb = []
        for i in range(1 << n):
            c = [(i & (1 << j)) != 0 for j in range(n)]
            comb.append(c)
        comb.sort(key=lambda x: sum(x))
        return [np.array(c) for c in comb]

    def _periodic_directions(self, filter_mask):
        dim = len(filter_mask)
        components = []
        for i in range(dim):
            if filter_mask[i]:
                c = [False] * dim
                c[i] = True
                components.append(np.array(c))
        def get_sub_masks(comp):
            inv = ~comp
            sub = [inv]
            for i in range(dim):
                if inv[i]:
                    c = [False] * dim
                    c[i] = True
                    sub.append(np.array(c))
            return sub
        all_masks = []
        for c in components:
            all_masks.extend(get_sub_masks(c))
        unique = {}
        for m in all_masks:
            unique[tuple(m)] = m
        res = list(unique.values())
        res.sort(key=lambda x: sum(x))
        return res

    def _mask_coords(self, mask, nw, se, pt):
        res = []
        for i in range(len(pt)):
            if mask[i]:
                res.append(pt[i])
            else:
                if abs(pt[i] - nw[i]) < 1e-10 or abs(pt[i] - se[i]) < 1e-10:
                    pass
                else:
                    return None
        return np.array(res)

    def _unmask_coords(self, mask, point, nw, se):
        dim = len(mask)
        unmasked_count = dim - sum(mask)
        combs = self._all_combinations(unmasked_count)
        res = []
        for comb in combs:
            new_pt = np.zeros(dim)
            p_idx = 0
            c_idx = 0
            for i in range(dim):
                if mask[i]:
                    new_pt[i] = point[p_idx]
                    p_idx += 1
                else:
                    new_pt[i] = nw[i] if comb[c_idx] else se[i]
                    c_idx += 1
            res.append(new_pt)
        return res

    def mesh_periodic_outer_box(self, fixed_points, fem_geometry, mdefaults, length_scale, filter_mask):
        return np.array([]), []

    def _relaxation_force(self, reduced_dist):
        if reduced_dist > 1.0: return 0.0
        return 1.0 - reduced_dist

    def _boundary_node_force(self, reduced_dist):
        if reduced_dist > 1.0: return 0.0
        if reduced_dist < 1e-10: return 100.0
        return (1.0 / reduced_dist) - 1.0

    def _sample_points(self, dim, nw, se, density_fun, target_count, rng=None):
        if rng is None: rng = np.random.default_rng()
        points = []
        max_attempts = target_count * 100
        attempts = 0
        while len(points) < target_count and attempts < max_attempts:
            p = rng.uniform(nw, se)
            d = density_fun(p)
            if rng.random() < d:
                points.append(p.tolist())
            attempts += 1
        return points

    def mesh_bodies_raw(self, driver, mesher, bb_min, bb_max, mesh_ext, objects, a0, density, fixed, mobile, simply, periodic, cache, hints):
        import scipy.spatial
        dim = len(bb_min)
        nw, se = np.array(bb_min), np.array(bb_max)
        if not fixed and not mobile and not simply:
            node_vol = (a0**dim) * 0.7
            def global_density(p):
                if not objects: return 1.0
                for obj in objects:
                    if obj(p) >= -1e-6:
                        return 1.0
                return 0.0
            box_vol = np.prod(se - nw)
            target_count = int(box_vol / node_vol)
            target_count = max(min(target_count, 10000), dim + 1 + 5)
            mobile = self._sample_points(dim, nw, se, global_density, target_count)
        all_points = list(fixed) + list(mobile) + list(simply)
        if not all_points and not objects:
             return RawMesh(dim, [], [], [])
        points_np = np.array(all_points)
        if points_np.shape[0] <= dim:
             return RawMesh(dim, all_points, [], [])
        max_relaxation_steps = min(mesher.mdefault_controller_step_limit_max, 50) 
        for step in range(max_relaxation_steps):
            try:
                tri = scipy.spatial.Delaunay(points_np)
            except:
                break
            forces = np.zeros_like(points_np)
            indptr, indices = tri.vertex_neighbor_vertices
            for i in range(len(points_np)):
                if i < len(fixed): continue
                pt = points_np[i]
                target_a = a0
                for neighbor_idx in indices[indptr[i]:indptr[i+1]]:
                    neighbor_pt = points_np[neighbor_idx]
                    vec = pt - neighbor_pt
                    dist = np.linalg.norm(vec)
                    if dist < 1e-12: continue
                    reduced_dist = dist / target_a
                    f_mag = self._relaxation_force(reduced_dist)
                    forces[i] += (f_mag / dist) * vec
            dt = mesher.mdefault_controller_time_step_scale * a0
            points_np += dt * forces
            if objects:
                bcs = [obj.bc_func for obj in objects]
                for i in range(len(fixed), len(points_np)):
                    self._enforce_boundary_conditions(mesher, bcs, points_np[i])
        try:
            tri = scipy.spatial.Delaunay(points_np)
        except:
            return RawMesh(dim, points_np.tolist(), [], [])
        simplices = tri.simplices.tolist()
        final_simplices = []
        final_regions = []
        if not objects:
            final_simplices = simplices
            final_regions = [1] * len(simplices)
        else:
            for sx in simplices:
                sx_pts = points_np[sx]
                cog = np.mean(sx_pts, axis=0)
                best_region = 0
                for i, obj in enumerate(objects, 1):
                    if obj(cog) >= -1e-6:
                        best_region = i
                        break
                if best_region > 0:
                    final_simplices.append(sx)
                    final_regions.append(best_region)
        return RawMesh(dim, points_np.tolist(), final_simplices, final_regions)

    # Body operations
    def body_union(self, objs):
        def bc(pos):
            return max(o(pos) for o in objs)
        return Body(AffineTrafo(objs[0].trafo.dim), bc)

    def body_difference(self, mother, subs):
        def bc(pos):
            res = mother(pos)
            for s in subs:
                res = min(res, -s(pos))
            return res
        return Body(AffineTrafo(mother.trafo.dim), bc)

    def body_intersection(self, objs):
        def bc(pos):
            return min(o(pos) for o in objs)
        return Body(AffineTrafo(objs[0].trafo.dim), bc)

    def _body_transform(self, body, matrix, displacement):
        trafo = AffineTrafo(body.trafo.dim, matrix, displacement)
        new_trafo = trafo.combine(body.trafo)
        return Body(new_trafo, body.bc_func)

    def body_shifted_sc(self, body, shift):
        dim = body.trafo.dim
        return self._body_transform(body, np.eye(dim), -np.array(shift))

    def body_shifted_bc(self, body, shift):
        return self.body_shifted_sc(body, shift)

    def body_scaled(self, body, scale):
        dim = body.trafo.dim
        s = np.array(scale)
        if s.ndim == 0: s = np.full(dim, s)
        return self._body_transform(body, np.diag(1.0/s), np.zeros(dim))

    def body_rotated_sc(self, body, a1, a2, rad):
        dim = body.trafo.dim
        mat = np.eye(dim)
        c, s = math.cos(rad), math.sin(rad)
        mat[a1, a1] = c
        mat[a1, a2] = s
        mat[a2, a1] = -s
        mat[a2, a2] = c
        return self._body_transform(body, mat, np.zeros(dim))

    def body_rotated_bc(self, body, a1, a2, rad):
        return self.body_rotated_sc(body, a1, a2, rad)

    def body_rotated_axis_sc(self, body, axis, rad):
        dim = body.trafo.dim
        if dim != 3: return body
        axis = np.array(axis) / np.linalg.norm(axis)
        c, s = math.cos(rad), math.sin(rad)
        t = 1 - c
        x, y, z = axis
        mat = np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
        ])
        return self._body_transform(body, mat.T, np.zeros(dim))

    def body_rotated_axis_bc(self, body, axis, rad):
        return self.body_rotated_axis_sc(body, axis, rad)
    
    # Primitives
    def body_box(self, p1, p2):
        nw, se = np.array(p1), np.array(p2)
        mid = (nw + se) * 0.5
        inv_half_len = 2.0 / np.abs(nw - se)
        def bc(pos):
            rel_dist = np.abs((pos - mid) * inv_half_len)
            return 1.0 - np.max(rel_dist)
        return Body(AffineTrafo(len(p1)), bc)

    def body_ellipsoid(self, radii):
        r = np.array(radii)
        inv_r = 1.0 / r
        def bc(pos):
            return 1.0 - np.sum((pos * inv_r)**2)
        return Body(AffineTrafo(len(radii)), bc)

    def body_frustum(self, c1, r1, c2, r2):
        p1, p2 = np.array(c1), np.array(c2)
        axis = p2 - p1
        axis_len_sq = np.sum(axis**2)
        def bc(pos):
            vec = pos - p1
            projection = np.dot(vec, axis) / axis_len_sq
            if projection < 0 or projection > 1: return -1.0
            r_at_p = r1 + projection * (r2 - r1)
            dist_sq = np.sum((vec - projection * axis)**2)
            return r_at_p**2 - dist_sq
        return Body(AffineTrafo(len(c1)), bc)

    def body_helix(self, c1, r1, c2, r2):
        return self.body_frustum(c1, r1, c2, r2)

    def mesh_from_points_and_simplices(self, dim, points, simplices, regions, periodic, reorder, distribute):
        return RawMesh(dim, points, simplices, regions, periodic)

    def copy_mesher_defaults(self, defaults):
        return copy.deepcopy(defaults)

    @property
    def mesher_defaults(self):
        return self._mesher_defaults

nmesh_backend = NMeshBackend()
