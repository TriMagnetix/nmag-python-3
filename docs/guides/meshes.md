# Meshes

Nmag represents the magnetic body with first-order tetrahedral finite elements.
The mesh coordinates define geometry, tetrahedra define the volume cells, and
positive integer region IDs determine material assignment.

## Load an existing mesh

```python
import nmesh

mesh = nmesh.load("geometry.msh")
print(len(mesh.points), len(mesh.simplices), sorted(set(mesh.regions)))
```

`nmesh.load` recognizes legacy Nmesh HDF5 by its internal layout. Other formats
are read through Meshio, so formats such as Gmsh and VTU can be used without
renaming them to resemble Nmesh files.

Only line, triangle, and tetrahedron cell blocks are imported. When a file mixes
supported dimensions, the highest-dimensional cells are selected. Region IDs
are read, in order of preference, from `region`, `gmsh:physical`, `cell_tags`,
or `gmsh:geometrical`; if none exists, region `1` is assigned.

!!! warning

    Meshio can represent many cell types, but the micromagnetic simulation path
    supports only 3D tetrahedra. Loading a format successfully does not make its
    non-tetrahedral cells valid for simulation.

## Construct a simplex mesh

Small meshes and programmatically generated meshes can be built directly:

```python
import nmesh

mesh = nmesh.mesh_from_points_and_simplices(
    points=[
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    simplices_indices=[[0, 1, 2, 3]],
    simplices_regions=[1],
)
mesh.save("tetrahedron.nmesh.h5")
```

Coordinates remain in the units chosen by the mesh author. They are converted
to metres only when `Simulation.load_mesh` applies `unit_length`.

## Generate geometry with Nmesh

The compatibility geometry API provides bodies such as `Box`, `Ellipsoid`, and
`Conic`, boolean operations, and the `nmesh.Mesh` generator. This is useful for
simple procedural geometries, but imported meshes from dedicated tools are often
easier to inspect and reproduce for complex models.

Whatever tool creates the mesh, check:

- tetrahedron orientation and nonzero volume;
- region tags and their intended material ordering;
- boundary resolution and element quality; and
- convergence as the characteristic element length is reduced.

The discretization length should be comfortably smaller than the physical
length scales that control the magnetization texture. Monitor
`maxangle_m_<material>` in NDT output: large angles between neighboring nodal
magnetizations are a warning that the mesh does not resolve the state.

## Save and inspect

`Mesh.save(path)` and `nmesh.save(mesh, path)` write an Nmesh HDF5 file.
Mesh objects also expose points, simplex indices, region IDs, surfaces, and
region volumes for validation before simulation.
