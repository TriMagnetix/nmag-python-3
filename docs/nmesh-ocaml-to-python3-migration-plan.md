### NMesh OCaml to Python3 Migration Plan (Incremental, Section-by-Section)

**Summary**
- Migrate [nmesh.py](/mnt/g/Code/nmag/nmag-python-3/src/nmesh/nmesh.py) from OCaml-backed calls to pure Python by porting logic from [mesh.ml](/mnt/g/Code/nmag/nmag-src/src/mesh.ml) and [snippets.ml](/mnt/g/Code/nmag/nmag-src/src/snippets.ml).
- Keep modern API only.
- Deliver full parity (not a reduced subset), but in controlled sections with explicit exit criteria.
- Use `scipy`, `h5py`, and `pymetis` as required dependencies.

**Public Interfaces / Type Changes**
- Keep current Python entrypoints in `nmesh.py` (`MeshBase`, `Mesh`, `MeshFromFile`, `MeshObject`, primitives, CSG, `load/save`, `mesh_from_points_and_simplices`).
- Replace OCaml-pill-style `raw_mesh` internals with a Python `RawMesh` data model (points, simplices, regions, neighbors, periodic groups, permutation, distribution).
- Remove `OCamlStub` after parity is verified; do not add extra legacy-name compatibility surface.

**Implementation Sections**
1. Section 1: Backend seam + core data model
Scope: Introduce internal pure-Python backend interfaces in `nmesh.py`.
Implement: `RawMesh`, `Body`, `MesherDefaults`, `Driver` abstractions; route all current `ocaml.*` call sites through backend methods with matching signatures.
Exit criteria: `nmesh.py` control flow no longer depends on an imported OCaml module.

2. Section 2: Port snippets foundations used by meshing
Scope: Port utility primitives needed by geometry and mesher internals.
Implement: array/index/filter helpers, `mx_mult`, `mx_x_vec`, determinant/inverse wrappers, list intersections, and `time_vmem_rss` equivalent behavior.
Exit criteria: utility tests pass and numeric behavior matches OCaml reference tolerance.

3. Section 3: Port body geometry + CSG
Scope: Replace body primitives/transforms currently delegated to OCaml.
Implement: `bc_box`, `bc_ellipsoid`, `bc_frustum`, `bc_helix`, affine transform composition, union/difference/intersection, shifted/scaled/rotated variants.
Exit criteria: geometry/CSG tests validate inside/outside logic, transforms, and dimension consistency.

4. Section 4: Port mesher defaults + driver semantics
Scope: Move mesher parameter and callback behavior fully into Python.
Implement: Python defaults mirroring `opt_mesher_defaults`; all setter mappings used by `MeshingParameters`; callback cadence/payload logic from `make_mg_gendriver`.
Exit criteria: setter and callback tests reproduce expected legacy behavior.

5. Section 5: Port core meshing pipeline
Scope: Implement pure-Python equivalent of `mesh_bodies_raw` + `mesh_it` workflow.
Implement: fem-geometry from bodies/hints, fixed/mobile/simply point handling, periodic bookkeeping, iterative relax/retriangulate loop, stop conditions, connectivity growth.
Exit criteria: deterministic mesh generation succeeds for 1D/2D/3D reference scenarios with fixed seeds.

6. Section 6: Port mesh query/extraction APIs
Scope: Replace all `mesh_plotinfo*`, counts/dim queries, regions, links, surfaces, volumes, permutation/distribution accessors.
Implement: `mesh_nr_points`, `mesh_nr_simplices`, `mesh_dim`, `mesh_plotinfo*`, `mesh_get_permutation`, `mesh_set_vertex_distribution`, and cache invalidation in `MeshBase`.
Exit criteria: accessor tests pass, including cache behavior and periodic data exposure.

7. Section 7: Port I/O + constructors
Scope: Complete constructor and file parity without OCaml.
Implement: ASCII `# PYFEM` read/write parity, `mesh_from_points_and_simplices`, `MeshFromFile`, `load/save`; HDF5 via `h5py` using `/mesh/{points,simplices,simplicesregions,permutation,periodicpointindices}` compatibility schema.
Exit criteria: ASCII/HDF5 roundtrip tests pass and preserve permutation/periodic info.

8. Section 8: Reorder/distribute parity + cleanup
Scope: Final parity for reorder/distribute behavior and remove migration scaffolding.
Implement: reordering strategy via `scipy.sparse.csgraph`; partitioning via `pymetis`; finalize `do_reorder/do_distribute` semantics and delete stub-only branches/comments.
Exit criteria: distribution/permutation invariants pass; `OCamlStub` removed.

**Test Plan**
1. Build OCaml parity fixtures for canonical geometries (single body, CSG, periodic, mesh-from-points).
2. Add section-gated tests for utilities, geometry/CSG, defaults/driver, meshing, query APIs, ASCII I/O, HDF5 I/O, reorder/distribute.
3. Use deterministic seeds and tolerance-based numeric assertions for coordinates/volumes.
4. Run full CI with required native deps and fail on parity regressions.

**Assumptions / Defaults Locked**
- Pure Python only target.
- Modern API only.
- Full parity required (not staged down to serial-only).
- New dependencies are allowed (`scipy`, `h5py`, `pymetis`).
