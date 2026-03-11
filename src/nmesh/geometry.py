import math
from typing import List, Any
from nmesh.backend import nmesh_backend as backend

class MeshObject:
    """Base class for geometric primitives and CSG operations."""
    def __init__(self, dim, fixed=None, mobile=None):
        self.dim = dim
        self.fixed_points = fixed if fixed is not None else []
        self.mobile_points = mobile if mobile is not None else []
        self.obj: Any = None

    def shift(self, vector, system_coords=True):
        self.obj = (backend.body_shifted_sc if system_coords else backend.body_shifted_bc)(self.obj, vector)
        return self
    
    def scale(self, factors):
        self.obj = backend.body_scaled(self.obj, factors)
        return self
    
    def rotate(self, a1, a2, angle, system_coords=True):
        rad = math.radians(angle)
        self.obj = (backend.body_rotated_sc if system_coords else backend.body_rotated_bc)(self.obj, a1, a2, rad)
        return self

    def rotate_3d(self, axis, angle, system_coords=True):
        rad = math.radians(angle)
        self.obj = (backend.body_rotated_axis_sc if system_coords else backend.body_rotated_axis_bc)(self.obj, axis, rad)
        return self

    def transform(self, transformations, system_coords=True):
        """Applies a list of transformation tuples."""
        for t in transformations:
            name, *args = t
            if name == "shift": self.shift(args[0], system_coords)
            elif name == "scale": self.scale(args[0])
            elif name == "rotate": self.rotate(args[0][0], args[0][1], args[1], system_coords)
            elif name == "rotate2d": self.rotate(0, 1, args[0], system_coords)
            elif name == "rotate3d": self.rotate_3d(args[0], args[1], system_coords)
        return self

class Box(MeshObject):
    def __init__(self, p1, p2, transform=None, fixed=None, mobile=None, system_coords=True, use_fixed_corners=False):
        import itertools
        dim = len(p1)
        fixed_pts = fixed if fixed is not None else []
        if use_fixed_corners:
            fixed_pts.extend([list(c) for c in itertools.product(*zip(p1, p2))])
        super().__init__(dim, fixed_pts, mobile)
        self.obj = backend.body_box([float(x) for x in p1], [float(x) for x in p2])
        if transform:
            self.transform(transform, system_coords)

class Ellipsoid(MeshObject):
    def __init__(self, lengths, transform=None, fixed=None, mobile=None, system_coords=True):
        super().__init__(len(lengths), fixed, mobile)
        self.obj = backend.body_ellipsoid([float(x) for x in lengths])
        if transform:
            self.transform(transform, system_coords)

class Conic(MeshObject):
    def __init__(self, c1, r1, c2, r2, transform=None, fixed=None, mobile=None, system_coords=True):
        super().__init__(len(c1), fixed, mobile)
        self.obj = backend.body_frustum(c1, r1, c2, r2)
        if transform:
            self.transform(transform, system_coords)

class Helix(MeshObject):
    def __init__(self, c1, r1, c2, r2, transform=None, fixed=None, mobile=None, system_coords=True):
        super().__init__(len(c1), fixed, mobile)
        self.obj = backend.body_helix(c1, r1, c2, r2)
        if transform:
            self.transform(transform, system_coords)

# --- CSG ---

def union(objects: List[MeshObject]) -> MeshObject:
    if len(objects) < 2: raise ValueError("Union requires at least two objects")
    res = MeshObject(objects[0].dim)
    for o in objects:
        res.fixed_points.extend(o.fixed_points)
        res.mobile_points.extend(o.mobile_points)
    res.obj = backend.body_union([o.obj for o in objects])
    return res

def difference(mother: MeshObject, subtract: List[MeshObject]) -> MeshObject:
    res = MeshObject(mother.dim, mother.fixed_points[:], mother.mobile_points[:])
    for o in subtract:
        res.fixed_points.extend(o.fixed_points)
        res.mobile_points.extend(o.mobile_points)
    res.obj = backend.body_difference(mother.obj, [o.obj for o in subtract])
    return res

def intersect(objects: List[MeshObject]) -> MeshObject:
    if len(objects) < 2: raise ValueError("Intersection requires at least two objects")
    res = MeshObject(objects[0].dim)
    for o in objects:
        res.fixed_points.extend(o.fixed_points)
        res.mobile_points.extend(o.mobile_points)
    res.obj = backend.body_intersection([o.obj for o in objects])
    return res
