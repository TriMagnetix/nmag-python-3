# Refined NMesh OCaml to Python 3 Migration Plan (V3 - Comprehensive Engineering & Management)

This plan details the migration of the `nmesh` library from a hybrid Python 2 / OCaml implementation to a pure Python 3 implementation, combining technical depth with clear work packages and acceptance criteria.

## 1. Objectives
- **Pure Python 3:** Eliminate the OCaml dependency and the `ocaml` Python module.
- **NumPy-First Architecture:** Mandate NumPy as the foundation for all internal data storage and linear algebra.
- **Performance Parity:** Use vectorization and optimized libraries (`scipy`, `numba`) to ensure performance parity with OCaml.
- **API Parity:** Maintain backward compatibility with the existing `nmesh` Python API.
- **Modular Design:** Split the monolithic `nmesh.py` into maintainable sub-modules.
- **Incremental Verification:** Mandate unit tests for every module to ensure parity with the legacy implementation.

## 2. Proposed Module Structure (`nmesh/` directory)
The `nmesh` package will be reorganized as follows:

```text
nmesh/
├── __init__.py          # Exposed public API (Mesh, MeshFromFile, load, save, etc.)
├── core.py              # RawMesh data model and core state management
├── geometry/
│   ├── __init__.py      # Geometry primitives (Box, Ellipsoid, etc.)
│   ├── csg.py           # CSG operations (union, difference, intersect)
│   └── transform.py     # Affine transformations and matrix logic
├── mesher/
│   ├── __init__.py      # High-level mesh_it_work coordination
│   ├── defaults.py      # Mesher parameter management (MeshingParameters)
│   ├── driver.py        # Callback driver semantics (make_mg_gendriver)
│   ├── forces.py        # Physics/Force calculations (Shape, Volume, etc.)
│   ├── relaxation.py    # Iterative relaxation loop and JIT-optimized logic
│   └── periodic.py      # Periodic boundary condition bookkeeping
├── io/
│   ├── __init__.py      # Unified load/save interface
│   ├── ascii.py         # # PYFEM 1.0 reader/writer
│   └── h5.py            # h5py-based HDF5 operations
└── utils.py             # Math/NumPy snippets and general helpers
```

## 3. Core Data Model (`RawMesh`)
| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `points` | `(N, dim)` | `float64` | Cartesian coordinates of all nodes. |
| `simplices` | `(M, dim+1)` | `int32` | Indices into `points` forming the mesh elements. |
| `regions` | `(M,)` | `int32` | Region ID for each simplex. |
| `point_regions` | `(N, K)` | `int32` | Sparse mapping or ragged list of regions each point belongs to. |
| `periodic_indices`| `(P, 2)` | `int32` | Pairs of indices representing periodic node equivalences. |
| `permutation` | `(N,)` | `int32` | Map from original input indices to current reordered indices. |

---

## 4. Migration Sections

### Section 1: Foundation & Utilities (Porting `snippets.ml`)
**Goal:** Port utility primitives and math required by geometry and mesher internals.
- **Work Packages:**
    1. Port array/list helpers (`filter`, `position`, `one_shorter`) using NumPy vectorization.
    2. Port numeric helpers (determinant, inverse, cross product) using `numpy.linalg`.
    3. Port timing/memory reporting equivalent for `time_vmem_rss`.
- **Acceptance:**
    1. Utility tests pass with deterministic inputs in `tests/nmesh/test_utils.py`.
    2. Numerical helpers match OCaml reference outputs within `1e-9` tolerance.

### Section 2: Mesher Defaults & Driver (Porting `mesh.ml` / `lib1.py`)
**Goal:** Move mesher parameter behavior and callback driver semantics fully to Python.
- **Work Packages:**
    1. Mirror `opt_mesher_defaults` values and field structure in `nmesh.mesher.defaults`.
    2. Implement full `MeshingParameters` setter mapping in pure Python.
    3. Port callback cadence and payload flow used by `make_mg_gendriver`.
- **Acceptance:**
    1. Setter-based tests reproduce expected overrides in `tests/nmesh/test_defaults.py`.
    2. Callback interval tests verify invocation cadence and payload shape.

### Section 3: Geometry & CSG (Porting `mesh.ml` / `lib1.py`)
**Goal:** Replace OCaml body primitive and transformation logic with Python equivalents.
- **Work Packages:**
    1. Port primitive boundary-condition builders: `bc_box`, `bc_ellipsoid`, `bc_frustum`, `bc_helix`.
    2. Transition to NumPy-compatible Signed Distance Functions (SDFs).
    3. Implement affine transform logic (`shift`, `scale`, `rotate`) using matrix multiplication.
    4. Implement CSG operations: `union`, `difference`, `intersection`.
- **Acceptance:**
    1. Geometry tests validate in/out classification in `tests/nmesh/test_geometry.py`.
    2. Transform order tests validate equivalence to OCaml semantics.

### Section 4: Core Meshing Engine (Porting `mesh.ml`)
**Goal:** Implement pure-Python equivalent of `mesh_bodies_raw` and `mesh_it_work`.
- **Work Packages:**
    1. Port `fem_geometry_from_bodies` for bodies, hints, and density handling.
    2. Implement point preparation: fixed/mobile/simply filtering and initial distribution.
    3. Port periodic point bookkeeping and index mapping.
    4. Port **Iterative Relaxation Loop**:
        - Shape, Volume, Neighbor, and Irrelevant Element force calculations.
        - JIT-optimized smoothing loop using `numba`.
        - Insertion/Deletion of points (Voronoi/Delaunay criteria).
- **Acceptance:**
    1. Deterministic meshing succeeds with fixed RNG seeds in `tests/nmesh/test_integration.py`.
    2. Periodic and hint-driven scenarios complete with valid topology.

### Section 5: Query & Extraction APIs (Porting `mesh.ml`)
**Goal:** Replace `mesh_plotinfo*` and related mesh query accessors.
- **Work Packages:**
    1. Implement `points`, `simplices`, `regions`, `links`, and `surfaces` accessors.
    2. Use `scipy.spatial.Delaunay` for adjacency and surface extraction.
    3. Implement cache invalidation logic for `MeshBase`.
- **Acceptance:**
    1. Accessor tests pass for populated meshes and edge cases.
    2. Cache tests validate stale-data invalidation when nodes are scaled.

### Section 6: I/O & Constructors (Porting `lib1.py`)
**Goal:** Complete pure-Python constructor and file I/O parity.
- **Work Packages:**
    1. Port ASCII `# PYFEM` reader/writer rules.
    2. Implement `MeshFromFile`, `load`, `save` without OCaml backend.
    3. Implement HDF5 compatibility via `h5py` for all core datasets.
- **Acceptance:**
    1. ASCII and HDF5 round-trip tests preserve topology and metadata in `tests/nmesh/test_io.py`.

### Section 7: Optimization & Distribution (Porting `mesh.ml`)
**Goal:** Finish reorder/distribution behavior using Metis.
- **Work Packages:**
    1. Implement reorder strategy using `scipy.sparse.csgraph.reverse_cuthill_mckee`.
    2. Implement partitioning/distribution flow using `pymetis`.
- **Acceptance:**
    1. Distribution invariants and connectivity bandwidth reductions are validated.

---

## 5. Performance Strategy
1. **Vectorization:** Avoid Python loops for force calculations; use NumPy broadcasting.
2. **JIT Compilation:** Apply `@numba.njit` to the relaxation loop and hot math functions.
3. **Sparse Structures:** Use `scipy.sparse` for large connectivity matrices.

## 6. Technical Mapping Table
| OCaml Symbol | Python / NumPy / SciPy Equivalent |
|--------------|-----------------------------------|
| `Qhull.delaunay` | `scipy.spatial.Delaunay` |
| `determinant` | `numpy.linalg.det` |
| `Mt19937` | `numpy.random.Generator(PCG64)` |
| `Metis` / `Parmetis` | `pymetis.part_mesh` |
| `PyTables` | `h5py` |
