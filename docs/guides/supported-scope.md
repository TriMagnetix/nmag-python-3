# Supported scope and limitations

## Supported simulation model

Nmag for Python 3 supports 3D, first-order tetrahedral micromagnetic workflows
with:

- legacy Nmesh HDF5 and modern Meshio-supported mesh input;
- single- and multi-region material assignment where material degrees of
  freedom are compatible at shared nodes;
- demagnetization, exchange, uniform applied field, uniaxial and cubic
  anisotropy, and custom polynomial anisotropy energy;
- pinning and Zhang-Li current-density spin-transfer torque;
- adaptive time advancement and relaxation with SciPy DOP853;
- NDT/HDF5 data, spatial field probing, and native restart checkpoints; and
- optional Rust kernels and resource-aware dense, hierarchical, or matrix-free
  demagnetization storage.

## Important limitations

The following are not currently supported:

- thermal or stochastic dynamics;
- Slonczewski spin-transfer torque;
- periodic micromagnetic boundary conditions;
- custom `phi_BEM`/legacy HLib configuration;
- material-specific magnetization degrees of freedom where shared geometric
  nodes require conflicting material coefficients;
- local inter-material exchange coupling;
- spatially or time-varying applied fields;
- complete multi-stage legacy hysteresis compatibility;
- legacy restart-file import; and
- a distributed-memory MPI solver.

The experimental Diffsol backend has a narrower scope: dense isotropic
relaxation with the default save and convergence schedule. It does not support
anisotropy, low-memory mode, custom schedules, or public `advance_time`.

## Numerical responsibility

Passing a solver test does not establish that a scientific model is resolved.
For each new workflow:

1. verify material values and units;
2. check geometry, region tags, tetrahedron quality, and boundary resolution;
3. refine the mesh and compare relevant observables;
4. tighten tolerances and maximum time step independently;
5. inspect maximum neighboring spin angle and spatial fields; and
6. compare against an analytic result, trusted package, experiment, or a
   previously validated Nmag case.

Adaptive solvers need not take the same internal steps or stop at the same time
to represent equivalent physical states. Compare field and magnetization arrays,
energies, and problem-specific observables rather than requiring identical step
histories.

## Resource limits

There is no universal mesh-size limit. Memory and time depend strongly on the
number of volume and boundary nodes, geometry, selected demagnetization storage,
and requested dynamics. Dense boundary storage grows quadratically. Auto mode
uses available-memory estimates, while low-memory mode avoids dense BEM storage
at the cost of repeated computation.

Review `last_bem_operator_stats`, host memory, and elapsed setup/action time
before scaling a study. Never disable a resource guard without estimating the
resulting allocation.
