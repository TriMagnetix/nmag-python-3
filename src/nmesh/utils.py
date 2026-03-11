import logging
from typing import List, Tuple
from pathlib import Path
from nmesh.backend import nmesh_backend as backend, RawMesh

log = logging.getLogger(__name__)

def _is_nmesh_ascii_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.readline().startswith("# PYFEM")
    except: return False

def outer_corners(mesh):
    """Determines the bounding box of the mesh nodes."""
    coords = mesh.points
    if not coords: return None, None
    transpose = list(zip(*coords))
    return [min(t) for t in transpose], [max(t) for t in transpose]

def generate_1d_mesh_components(regions: List[Tuple[float, float]], discretization: float,
                                tolerance=lambda x: x) -> Tuple:
    """Generates 1D mesh components (points, simplices, surfaces)."""
    points = []
    simplices = []
    surfaces = []
    pnt_hash = {}
    srf_hash = {}

    def add_point(y):
        i = len(points)
        x = tolerance(y)
        if x in pnt_hash:
            return pnt_hash[x]
        pnt_hash[x] = i
        points.append([float(x)])
        return i

    def add_surface(y, idx, body):
        if y in srf_hash:
            i, _, _ = srf_hash[y]
            srf_hash[y] = (i + 1, body, idx)
        else:
            srf_hash[y] = (0, body, idx)

    nbody = 0
    for (left_x, right_x) in regions:
        nbody += 1
        if left_x > right_x: left_x, right_x = right_x, left_x
        width = right_x - left_x
        num_pts_per_reg = max(1, abs(int(width / discretization)))
        step = width / num_pts_per_reg

        last_idx = add_point(left_x)
        add_surface(left_x, last_idx, nbody)
        for i in range(1, num_pts_per_reg + 1):
            idx = add_point(left_x + i * step)
            simplices.append((nbody, [last_idx, idx]))
            last_idx = idx

        add_surface(right_x, last_idx, nbody)

    for s in srf_hash:
        count, body, idx = srf_hash[s]
        if count == 0:
            surfaces.append((body, [idx]))

    return (points, simplices, surfaces)

def generate_1d_mesh(regions: List[Tuple[float, float]], discretization: float):
    """Generates a 1D mesh."""
    from nmesh.base import mesh_from_points_and_simplices
    points, simplices, surfaces = generate_1d_mesh_components(regions, discretization)
    
    simplices_indices = [indices for _, indices in simplices]
    simplices_regions = [region for region, _ in simplices]
    
    return mesh_from_points_and_simplices(
        points=points,
        simplices_indices=simplices_indices,
        simplices_regions=simplices_regions,
        do_distribute=False
    )

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
        import sys
        sys.stdout.write(content)
    elif isinstance(out, (str, Path)):
        Path(out).write_text(content)
    else:
        out.write(content)

def memory_report(tag: str):
    """Reports memory usage."""
    t, vmem, rss = backend.time_vmem_rss()
    log.log(15, f"Memory report: T= {t:f} VMEM= {int(vmem)} KB RSS= {int(rss)} KB {tag}")
