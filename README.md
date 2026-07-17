# Nmag for Python 3

`nmag-python-3` is a modern, standalone Python implementation of the Nmag
micromagnetic simulation interface. It can read legacy Nmesh files and modern
simplex meshes, assign magnetic materials, calculate effective fields, relax
isotropic LLG systems, save NDT/HDF5 results, and probe fields inside
tetrahedral meshes.

The current release includes a minimum viable dynamic solver for the supported
isotropic workflows. Dynamic results should still be validated against legacy
Nmag when introducing a new geometry or material configuration.

## Requirements

- Linux
- Python 3.10 or newer
- A C compiler for Python packages that do not have a wheel for your platform
- Rust only when building the optional native accelerators

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip build-essential
```

## Installation

From a clone of this repository:

```bash
./scripts/setup.sh
```

The setup script creates or reuses `.venv`, asks whether to build the optional
Rust accelerator, and runs the appropriate checks. It always invokes the
virtual environment explicitly, so activation is never required.

Run checks later with:

```bash
./scripts/verify.sh
./scripts/verify.sh --rust
```

`--rust` adds Rust formatting, lint, and native tests. Use it after choosing
the accelerator during setup.

## First Static Simulation

Save this as `example.py` and run it with `python example.py`:

```python
import nmag
import nmesh


mesh = nmesh.mesh_from_points_and_simplices(
    points=[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    simplices_indices=[[0, 1, 2, 3]],
    simplices_regions=[1],
)
mesh.save("sample.nmesh.h5")

permalloy = nmag.MagMaterial(
    name="Py",
    Ms=nmag.SI(1e6, "A/m"),
    exchange_coupling=nmag.SI(13e-12, "J/m"),
)

simulation = nmag.Simulation(name="sample")
simulation.load_mesh(
    "sample.nmesh.h5",
    [("magnetic", permalloy)],
    unit_length=nmag.SI(1e-9, "m"),
)
simulation.set_m([1.0, 0.0, 0.0])
simulation.set_H_ext([0.0, 0.0, 0.0], nmag.SI("A/m"))
simulation.save_data(fields="all")

print(simulation.get_subfield_average("H_demag"))
```

This writes:

- `sample_dat.ndt`: one tabular row of averaged quantities.
- `sample_dat.h5`: spatial field arrays and metadata.

Mesh coordinates are multiplied by `unit_length` when loaded. In the example,
one mesh coordinate unit is one nanometre.

## Relaxation

After loading a mesh and setting the initial magnetization, relax the system
with:

```python
simulation.relax()

print(simulation.time)
print(simulation.step)
print(simulation.last_integrator_stats)
print(simulation.effective_integrator_max_step)
```

The default convergence threshold is one degree per nanosecond. Convergence is
checked every five accepted integration steps and must be satisfied twice in a
row. The default integrator uses SciPy DOP853 with relative and absolute
tolerances of `1e-6`. The configured maximum step defaults to 1 ps. For meshes
with exchange coupling, Nmag derives a smaller effective ceiling when needed
to resolve the fastest lumped-FEM exchange mode without explicit-step
instability.

Because convergence scheduling is based on accepted steps, DOP853 and legacy
CVODE can confirm the same threshold at different simulated times. Validate a
new dynamic workflow's spatial stop state as well as its averages.

Change these values before calling `relax()` when a simulation requires it:

```python
simulation.set_params(
    stopping_dm_dt=nmag.SI(0.5e9 * 3.141592653589793 / 180.0, "1/s"),
    ts_rel_tol=1e-7,
    ts_abs_tol=1e-7,
    ts_max_step=nmag.SI(0.5e-12, "s"),
)
```

`advance_time(target_time, max_it=-1, exact_tstop=None)` exposes the underlying
accepted-step integration for workflows that need explicit time control.

Pin selected nodes by setting the local multiplier for the complete
magnetization derivative. Zero fixes a node and one leaves it free:

```python
simulation.set_pinning(lambda position: 0.0 if position[2] < 5e-9 else 1.0)
```

Current-density (Zhang-Li) spin-transfer torque is enabled by assigning a
uniform, nodal, or position-dependent current-density field:

```python
simulation.set_current_density([0.0, 0.0, 1e12], nmag.SI("A/m^2"))
simulation.advance_time(nmag.SI(1e-12, "s"))
```

The torque uses each material's `llg_polarisation`, `llg_xi`, `llg_damping`,
and `Ms`. `dm_dcurrent` exposes the recovered FEM directional derivative.

Native restart checkpoints preserve the loaded simulation's magnetisation,
pinning, current density, external field, clock, and supported dynamics state:

```python
checkpoint = simulation.save_restart_file("relaxed_state.h5")

# In a compatible simulation with the same mesh and materials already loaded:
simulation.load_restart_file(checkpoint)
```

`load_m_from_h5file(checkpoint)` transfers only magnetisation and requires only
the same mesh. Checkpoints are native `nmag-python-3` HDF5 files; legacy Nmag
restart files are intentionally unsupported. DOP853 is reinitialized from the
saved physical state, so subsequent adaptive step sizes can differ while
fixed-time physical results remain equivalent.

Call `simulation.get_restart_file_name()` to inspect the default checkpoint
path. With `from when import at`, a relaxation schedule such as
`save=[("save_restart", at("stage_end"))]` writes that default checkpoint at
the configured save point.

An experimental implicit Rust backend is available for stiff relaxation after
building the accelerator:

```python
config = nmag.NmagConfig(integrator_backend="diffsol", accelerator="rust")
simulation = nmag.Simulation(config=config)
```

It uses Diffsol BDF with an analytic LLG Jacobian-vector product and reports
Newton and linear-solver statistics through `last_integrator_stats`. The
default remains SciPy. This first backend supports `relax()` with its default
save/convergence schedule and dense isotropic systems; it does not yet replace
`advance_time()` or custom relaxation schedules. There is no fixed mesh or
state-count limit. Before allocating dense solver matrices, Nmag estimates peak
memory and uses 80% of currently available memory as its default budget. Set
`NMAG_DIFFSOL_DENSE_MEMORY_LIMIT_GIB` to a reviewed GiB value or `unlimited` to
override that resource guard. A 540-node sphere with 1,620 magnetization state
values is included in the validated range; dense time and memory growth still
make this backend best suited to meshes that fit the available machine.

For larger meshes or memory-constrained machines, select the low-memory path:

```bash
NMAG_MEMORY_MODE=low python simulation.py
```

This uses sparse CSR FEM matrices, iterative scalar-potential solves, and an
exact matrix-free Lindholm BEM action. It preserves the same discretized model
while trading additional computation for lower memory use. The optional Rust
accelerator substantially reduces matrix-free BEM time. SciPy DOP853 remains
the dynamics integrator because the current Diffsol backend constructs a dense
affine operator and is incompatible with low-memory mode.

## Supported Scope

The current public workflow supports:

- loading ASCII `.nmesh` and legacy `.nmesh.h5` tetrahedral meshes;
- loading modern simplex mesh files through `meshio`, including VTU, Gmsh, and
  XDMF formats;
- one or more material regions;
- uniform or position-dependent normalized magnetization through `set_m`;
- uniform external fields through `set_H_ext`;
- scalar, nodal, or position-dependent pinning through `set_pinning`;
- material-specific exchange and LLG coefficients for regions whose nodes do
  not require conflicting material-specific magnetization degrees of freedom;
- uniform, nodal, or position-dependent current-density (Zhang-Li)
  spin-transfer torque through `set_current_density`;
- static demagnetization, exchange, external, total fields, and derived energy
  quantities used by the MVP;
- adaptive LLG time integration and convergence-based relaxation for supported
  isotropic material configurations;
- native HDF5 restart checkpoints for compatible loaded simulations;
- NDT/HDF5 output, field averages, and point probes;
- optional Python, NumPy/Numba, SciPy, and shared-memory Rust calculation
  backends.

Important current limitations:

- HLib and custom `phi_BEM` support are not implemented;
- material-specific `set_m` subfields are not implemented;
- thermal dynamics, dynamic anisotropy, Slonczewski spin-transfer torque, and
  the complete multi-stage hysteresis API are not implemented;
- a geometric node shared by regions with different exchange, LLG, or STT
  coefficients requires material-specific magnetization degrees of freedom
  and is rejected explicitly;
- the low-memory BEM path has linear storage but still performs every boundary
  node/face interaction, so very large boundaries can take substantial time;
- the experimental Diffsol backend remains dense and is not available in
  low-memory mode.

Unsupported requests raise `NotImplementedError` or a specific configuration
error rather than silently falling back to legacy behavior.

## Mesh Formats

`nmesh.load()` detects legacy Nmesh HDF5 from its internal layout and routes it
to the dedicated compatibility reader. Other formats are loaded through
`meshio`; their filename does not need to resemble an Nmesh file. Only line,
triangle, and tetrahedron cells are imported, and mixed-cell files use the
highest-dimensional supported cells. Region IDs are read from `region`,
`gmsh:physical`, `cell_tags`, or `gmsh:geometrical` metadata when present;
otherwise the importer assigns region `1`.

For example:

```python
mesh = nmesh.load("geometry.vtu")
```

## Optional Rust Accelerators

`./scripts/setup.sh` offers to build the extension. To add it after a standard
setup, install Rust with [rustup](https://rustup.rs/), install the Rust extra,
then run `./scripts/build-accelerator.sh`.

After editing `rust/nmag_accel`, rebuild the extension with:

```bash
./scripts/build-accelerator.sh
```

This makes a fast debug build in `.venv`; pass `--release` for an optimized
build.

## Manual Setup

The scripts are the recommended path. These commands are useful for CI or when
you prefer to manage the environment yourself:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Optional accelerator
python -m pip install -e '.[rust]'
./scripts/build-accelerator.sh --release
```

Normal use does not require Rust. Each simulation receives an immutable
`NmagConfig`; `auto` uses available Rust kernels only where the implementation
expects them to help, `off` always keeps the Python/Numba paths, and `rust`
requires the extension and fails clearly when it is unavailable.

```python
from pathlib import Path

import nmag

config = nmag.NmagConfig(
    default_name="relaxation",
    output_directory=Path("results"),
    output_policy="error",
    accelerator="auto",
    accelerator_overrides={nmag.RustKernel.LLG: "rust"},
)
simulation = nmag.Simulation(config=config)
```

`Simulation(name="...")` overrides `default_name`. `output_policy="error"`
refuses existing NDT/HDF5 outputs, `"replace"` removes them before writing,
and `"append"` validates the prior NDT schema before adding output rows. Append
does not restore physical state; use `load_restart_file()` explicitly for that.

For a simulation created without an explicit config only,
`NMAG_ACCELERATOR=auto|off|rust` selects the default accelerator mode. An
explicit `NmagConfig` always wins. The former per-kernel `NMAG_*_BACKEND`
variables are no longer supported. Expert callers can use
`accelerator_overrides` with `RustKernel` values as shown above. Diffsol is an
explicit `integrator_backend="diffsol"` configuration and is never selected by
the global accelerator mode.

Memory and storage selectors are:

- `NMAG_MEMORY_MODE=auto|low`
- `NMAG_DEMAG_FEM_MATRIX_BACKEND=auto|dense|sparse`
- `NMAG_DEMAG_BEM_STORAGE_BACKEND=auto|dense|matrix-free`
- `NMAG_DEMAG_LINEAR_SOLVER_BACKEND=auto|numpy|scipy`

There is no default point-count rejection. When current host/container memory
can be measured, auto mode keeps dense FEM and BEM storage while each
conservative estimate remains within 20% of available memory, then switches to
sparse FEM or matrix-free BEM. If memory cannot be measured, 2,048 volume or
boundary points is the conservative fallback crossover. Override those fallback counts with
`NMAG_DEMAG_FEM_AUTO_SPARSE_MIN_POINTS` and
`NMAG_DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES`. Explicit dense selection
is retained as the reference backend; `NMAG_DEMAG_DENSE_MAX_POINTS` is an
optional user-defined safety cap rather than a built-in limitation.

Sparse FEM systems use normalized iterative SciPy solves. Dense auto mode uses
reusable SciPy LU factorizations for systems with at least 128 points and keeps
NumPy as the small-system reference path. Override that threshold with
`NMAG_DEMAG_LINEAR_SOLVER_AUTO_SCIPY_MIN_SIZE`.

Rust kernels use Rayon shared-memory parallelism above an operation-size
crossover. Set `RAYON_NUM_THREADS` before starting Python to cap workers for
reproducible profiling or shared-machine resource limits. Changing the global
pool after numerical work begins is intentionally unsupported. Inspect the
effective process configuration with:

```python
print(nmag.parallel_runtime_info())
```

Independent-point kernels are bitwise identical between one and multiple
workers; reduction-based kernels are tested with numerical parity tolerances.

## Development Checks

Use the verification script when possible:

```bash
./scripts/verify.sh
./scripts/verify.sh --rust
```

The equivalent manual commands are:

```bash
.venv/bin/ruff check src tests
.venv/bin/pyright
.venv/bin/python -m pytest
cargo fmt --manifest-path rust/nmag_accel/Cargo.toml --all -- --check
cargo clippy --manifest-path rust/nmag_accel/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/nmag_accel/Cargo.toml
```

## Project Status

Static numerical behavior has been compared against legacy Nmag across
canonical sphere, cube, elongated-prism, frontend-derived, and nonuniform
magnetization fixtures. The Python 3 solver converges the scalar-potential
system more tightly than legacy's default iterative tolerance and retains full
LLG coefficient precision. The relaxation implementation preserves legacy
field units and convergence rules while using a maintained SciPy integrator;
accepted adaptive step counts and convergence times are not expected to match
CVODE. Magnon-cascade average parity passes, while its strict spatial stop-state
gate remains open because DOP853 confirms two five-step checks earlier.

The original project is available at
[nmag-project/nmag-src](https://github.com/nmag-project/nmag-src).
