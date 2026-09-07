# Mesh API

The reference uses implementation paths where necessary for static
documentation, but the documented primitives and operations are exported from
the top-level `nmesh` package. For example, use `nmesh.Box`, not an internal
module import.

## Loading and construction

::: nmesh.mesh_io.load

::: nmesh.mesh_io.mesh_from_points_and_simplices

::: nmesh.mesh_io.save

## Mesh objects

::: nmesh.mesh_model.MeshBase
    options:
      members:
        - points
        - simplices
        - regions
        - dim
        - surfaces
        - point_regions
        - links
        - region_volumes
        - num_regions
        - periodic_point_indices
        - permutation
        - scale_node_positions
        - save
        - to_lists

::: nmesh.mesh_generation.Mesh
    options:
      members: false

## Geometry primitives

::: nmesh.geometry.primitives.Box
    options:
      members: false

::: nmesh.geometry.primitives.Ellipsoid
    options:
      members: false

::: nmesh.geometry.primitives.Conic
    options:
      members: false

::: nmesh.geometry.boolean_operations.union

::: nmesh.geometry.boolean_operations.difference

::: nmesh.geometry.boolean_operations.intersect
