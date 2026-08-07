# Configuration and performance

`NmagConfig` keeps process choices explicit and local to one simulation:

```python
from pathlib import Path

import nmag

config = nmag.NmagConfig(
    default_name="relaxation",
    output_directory=Path("results"),
    output_policy="error",
    accelerator="auto",
    demag_bem_storage="auto",
)
```

An explicit configuration takes precedence over environment selectors.

## Optional Rust acceleration

`accelerator` accepts:

- `"auto"`: use installed Rust kernels where Nmag expects a benefit;
- `"off"`: keep Python/Numba implementations; or
- `"rust"`: require the extension and fail if it is unavailable.

Advanced callers can override individual calculation families with
`accelerator_overrides={nmag.RustKernel.LLG: "rust"}`. Cap Rayon worker threads
before starting Python when sharing a machine:

```bash
RAYON_NUM_THREADS=4 .venv/bin/python simulation.py
```

Inspect the effective setup with `nmag.parallel_runtime_info()`.

## Demagnetization storage

The boundary-element operator is often the dominant memory cost.
`demag_bem_storage` accepts `"auto"`, `"dense"`, `"hierarchical"`, or
`"matrix-free"`.

Auto mode keeps dense storage while it fits the resource budget, then selects a
certified hierarchical operator when Rust is available, or exact matrix-free
action otherwise. Hierarchical construction trades setup time for fast repeated
application; matrix-free action trades computation for linear storage.

Select the complete low-memory path for large meshes:

```bash
NMAG_MEMORY_MODE=low .venv/bin/python simulation.py
```

This selects sparse FEM systems, iterative scalar-potential solves, and exact
matrix-free Lindholm BEM. SciPy DOP853 remains the integrator.

## Environment selectors

For simulations created without an explicit config:

```text
NMAG_ACCELERATOR=auto|off|rust
NMAG_MEMORY_MODE=auto|low
NMAG_DEMAG_FEM_MATRIX_BACKEND=auto|dense|sparse
NMAG_DEMAG_BEM_STORAGE_BACKEND=auto|dense|hierarchical|matrix-free
NMAG_DEMAG_LINEAR_SOLVER_BACKEND=auto|numpy|scipy
```

Prefer `NmagConfig` in reusable programs. Environment settings are convenient
for benchmark sweeps and operational overrides.

## Diagnostics

- `last_integrator_stats` reports accepted steps, right-hand-side evaluations,
  simulated time, wall time, status, and failure state.
- `last_bem_operator_stats` reports the selected boundary operator, setup and
  storage measurements, compression information, and fallback reason.
- `last_*_timings_seconds` mappings expose operation timings for profiling.

Treat diagnostics as measurements of one run and host, not portable performance
guarantees.
