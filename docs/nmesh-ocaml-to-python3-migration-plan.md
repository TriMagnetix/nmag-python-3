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

## Section 1: Backend seam + core data model

### 1.1 Goal
Introduce internal pure-Python backend interfaces in `nmesh.py` so all `ocaml.*` calls are routed through a Python backend object.

### 1.2 Work packages
1. Define backend protocol for mesh ops, body ops, defaults, and driver creation.
2. Implement `RawMesh`, `Body`, `MesherDefaults`, and `Driver` Python types.
3. Replace direct `ocaml.*` usage in classes/functions with backend calls.
4. Keep method signatures stable at the `nmesh.py` public boundary.

### 1.3 Acceptance
1. `nmesh.py` imports and runs without an OCaml module.
2. All previous `ocaml.*` call sites are backend-routed.
3. Public constructor signatures remain unchanged.

## Section 2: Port snippets foundations used by meshing

### 2.1 Goal
Port utility primitives from `snippets.ml` required by geometry and mesher internals.

### 2.2 Work packages
1. Port array/list helpers (`filter`, `position`, `one_shorter`, intersections, sorting checks).
2. Port numeric helpers (`mx_mult`, `mx_x_vec`, determinant/inverse wrappers).
3. Port timing/memory reporting equivalent for `time_vmem_rss`.
4. Add unit tests for utility parity and numerical tolerance behavior.

### 2.3 Acceptance
1. Utility tests pass with deterministic inputs.
2. Numerical helpers match OCaml reference outputs within tolerance.
3. No utility dependency on OCaml runtime remains.

## Section 3: Port body geometry + CSG

### 3.1 Goal
Replace OCaml body primitive and transformation logic with Python equivalents.

### 3.2 Work packages
1. Port primitive boundary-condition builders: `bc_box`, `bc_ellipsoid`, `bc_frustum`, `bc_helix`.
2. Port affine transform operations and composition order semantics.
3. Port CSG operations: `union`, `difference`, `intersection`.
4. Keep shifted/scaled/rotated variants behaviorally compatible with current `nmesh.py` API.

### 3.3 Acceptance
1. Geometry tests validate in/out classification for representative points.
2. Transform order tests validate equivalence to OCaml semantics.
3. CSG tests confirm region and object composition behavior.

## Section 4: Port mesher defaults + driver semantics

### 4.1 Goal
Move mesher parameter behavior and callback driver semantics fully to Python.

### 4.2 Work packages
1. Mirror `opt_mesher_defaults` values and field structure in Python.
2. Implement full setter mapping used by `MeshingParameters.pass_parameters_to_ocaml`.
3. Port callback cadence and payload flow used by `make_mg_gendriver`.
4. Preserve current modern API while internalizing behavior.

### 4.3 Acceptance
1. Setter-based tests reproduce expected defaults/overrides.
2. Callback interval tests verify invocation cadence and payload shape.
3. No dependency on OCaml mesher-default objects remains.

## Section 5: Port core meshing pipeline

### 5.1 Goal
Implement pure-Python equivalent of `mesh_bodies_raw` and `mesh_it` flow.

### 5.2 Work packages
1. Port `fem_geometry_from_bodies` behavior for bodies, hints, and density handling.
2. Port fixed/mobile/simply point filtering and initial point preparation.
3. Port periodic point bookkeeping and periodic index mapping behavior.
4. Port iterative relax/retriangulate loop and stop conditions.
5. Port connectivity/bookkeeping growth path needed by downstream plotinfo accessors.

### 5.3 Acceptance
1. Deterministic 1D/2D/3D meshing succeeds with fixed RNG seeds.
2. Periodic and hint-driven scenarios complete with valid topology.
3. Mesh-generation failure modes raise consistent Python exceptions.

## Section 6: Port mesh query/extraction APIs

### 6.1 Goal
Replace `mesh_plotinfo*` and related mesh query accessors with pure Python implementations.

### 6.2 Work packages
1. Implement `mesh_nr_points`, `mesh_nr_simplices`, `mesh_dim`.
2. Implement `mesh_plotinfo*` family (points, simplices, regions, links, surfaces, periodic indices, full plotinfo bundle).
3. Implement `mesh_get_permutation` and `mesh_set_vertex_distribution`.
4. Preserve and validate `MeshBase` cache invalidation behavior.

### 6.3 Acceptance
1. Accessor tests pass for populated meshes and edge cases.
2. Cache tests validate stale-data invalidation when nodes are rescaled/updated.
3. Returned structures match expected shape/type contracts.

## Section 7: Port I/O + constructors

### 7.1 Goal
Complete pure-Python constructor and file I/O parity for ASCII and HDF5 paths.

### 7.2 Work packages
1. Port ASCII `# PYFEM` reader/writer rules, including validation and orientation-sensitive output behavior.
2. Implement `mesh_from_points_and_simplices` constructor parity and index handling.
3. Implement `MeshFromFile`, `load`, `save` without OCaml backend.
4. Implement HDF5 compatibility via `h5py` for `/mesh/{points,simplices,simplicesregions,permutation,periodicpointindices}`.

### 7.3 Acceptance
1. ASCII roundtrip tests preserve topology, region ids, and periodic groups.
2. HDF5 roundtrip tests preserve permutation and periodic data.
3. Constructor tests validate initial index modes and error handling.

## Section 8: Reorder/distribute parity + cleanup

### 8.1 Goal
Finish reorder/distribution behavior and remove migration scaffolding.

### 8.2 Work packages
1. Implement reorder strategy using `scipy.sparse.csgraph`-based flow.
2. Implement partitioning/distribution flow using `pymetis`.
3. Validate `do_reorder` and `do_distribute` semantics across load/generation paths.
4. Remove `OCamlStub` and any obsolete shim code/comments.

### 8.3 Acceptance
1. Distribution/permutation invariants are validated in tests.
2. Reorder/distribute behavior is consistent between constructors and file-load paths.
3. `nmesh.py` contains no OCaml stub backend path.

**Test Plan**

### T1 Fixtures
1. Build OCaml parity fixtures for canonical geometries (single body, CSG, periodic, mesh-from-points).
2. Store deterministic fixture metadata (seed, dimensions, expected invariants).

### T2 Section-gated tests
1. Utilities/numerics tests for snippet ports.
2. Geometry/CSG tests for primitives, transforms, and CSG.
3. Defaults/driver tests for setter and callback behavior.
4. Core meshing tests for deterministic output and convergence behavior.
5. Query/extraction tests for `mesh_plotinfo*` and cache semantics.
6. ASCII/HDF5 I/O roundtrip tests.
7. Reorder/distribute tests with permutation integrity checks.

### T3 CI policy
1. Run full CI with required native dependencies installed.
2. Fail build on parity regression or deterministic fixture drift.

**Assumptions / Defaults Locked**
- Pure Python only target.
- Modern API only.
- Full parity required (not staged down to serial-only).
- New dependencies are allowed (`scipy`, `h5py`, `pymetis`).
