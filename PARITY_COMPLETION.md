# NMesh Section 4 Parity Completion

**Status:** exact standalone behavioral parity required  
**Branch:** `nmesh-section4`  
**Reviewed:** 2026-05-04  
**Reference sources:** `nmag-src/src/mesh.ml`, `nmag-src/src/snippets.ml`, `nmag-src/interface/nmeshlib/lib1.py`

## Position

The Python 3 `nmesh` port must implement the full behavior of the original OCaml/Python2 mesher without requiring OCaml at runtime. Missing behavior is implementation work, not an accepted end state.

The target is exact behavioral parity, not approximate parity. Topology decisions, region assignment, point-state transitions, add/delete scheduling, boundary recovery, periodic grouping, and file-visible mesh metadata must match the legacy implementation for the same inputs. Numeric tolerances are only allowed for floating-point coordinate comparisons, and each tolerance must be tied to a specific floating-point or triangulator tie-break reason.

The port may use Python, NumPy, SciPy, Numba, Rust, C++, or C as needed. The public Python API should remain stable, and compiled helpers should be isolated behind that API.

## Completed In This Section 4 Pass

- Removed the Python-side 60-step cap; configured `controller_step_limit_max` is honored.
- Implemented legacy controller cadence: square-number add/delete checks, relaxed add/delete thresholds, topology-threshold retriangulation, minimum-step convergence, force-equilibrium stopping, and 50-step post-change settling.
- Aligned force metrics with legacy behavior: density-scaled movement, centroid density for simplex forces, tangent-projected boundary effective force, corner suppression, and OCaml-style Voronoi density correction.
- Classified force-time simplices with the same centroid, near-vertex probe, `Boundary` state, and raw volume-order ratio rules used by the legacy relevant/irrelevant topology split.
- Restricted irrelevant-element forces to mobile nodes, matching the active legacy path.
- Replaced deterministic grid seeding with the legacy random density estimator, D-lattice sphere-packing node estimate, rejection sampler, and RNG consumption order.
- Replaced midpoint point insertion with Gaussian insertion around the source point using local effective rod length.
- Ported active boundary recovery behavior from `mirror_simplices`: 2D mirrored points and 3D boundary-edge midpoint prevention points.
- Seeded paired fixed points on periodic outer-box faces, fixed only exact periodic boundary nodes, and canonicalized multi-axis periodic equivalence groups.
- Added final high-density moving-point cleanup, outside dynamic point cleanup, and final boundary snapping before final assembly.
- Normalized final simplex orientation to the legacy positive-volume convention.
- Fixed flat-boundary filtering so it follows the legacy `Boundary` state and raw volume-order ratio rules instead of deleting all geometrically boundary-adjacent multi-region simplices.
- Changed unsupported density snippets to fail loudly rather than silently falling back to density `1.0`.
- Added parity coverage for coarse adjacent pieces, concave difference domains, multi-axis periodic groups, final cleanup, and force-time relevant/irrelevant classification.
- Added canonical mesh signature helpers so parity fixtures compare topology and metadata exactly, with coordinate tolerance explicitly opt-in.
- Added a parity comparison runner (`tools/nmesh_parity_compare.py`) that generates modern scenario meshes and compares them with legacy `.nmesh` artifacts or a configured legacy runner command.
- Tightened runtime matching to avoid mesh-size-scaled boundary drift tolerance and rounded non-periodic periodic-group coordinates.

## Validation Gate

These are verification tasks, not accepted limitations:

- Build reference fixtures from legacy examples and compare canonicalized topology, regions, surfaces, links, periodic groups, controller decisions, and mesh metadata exactly using `nmesh.mesher.parity`.
- Compare coordinates with the narrowest practical floating-point tolerance, documenting each tolerance and why exact binary equality is not the right assertion.
- Run the modernization scenario matrix through `tools/nmesh_parity_compare.py` and record metric-level acceptance thresholds for physically equivalent meshes when exact topology is not expected.
- Complete reference validation for periodic outer-box workflows on multiple periodic directions and complex periodic entities.
- Expand multi-piece/interface parity fixtures to include more interface-prevention cases.
- Profile large 3D meshes with legacy defaults and move any proven hot path to a compiled Rust, C++, or C helper behind the Python API.

## Sign-Off Criteria

- Legacy reference examples produce exact canonical mesh topology and metadata parity.
- Periodic, hint-driven, concave, and multi-piece examples match legacy behavior exactly, apart from explicitly documented floating-point coordinate tolerance.
- Add/delete, retriangulation, boundary recovery, and final cleanup decisions are covered by parity tests.
- Unsupported density syntax never changes mesh density silently.
- The full project test suite and parity fixtures pass in the project venv.
