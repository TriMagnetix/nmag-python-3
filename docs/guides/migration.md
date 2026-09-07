# Migrating from legacy Nmag

The Python 3 interface deliberately preserves the recognizable Nmag workflow,
but it is a standalone implementation rather than a drop-in replacement for the
old `nsim` runtime.

## Common translations

| Legacy usage | Python 3 usage |
| --- | --- |
| Run with `nsim script.py` | Run with `.venv/bin/python script.py` |
| `from nmag import SI` | `import nmag`, then `nmag.SI(...)` |
| `nsim.when.at/every` | `from when import at, every` |
| `--clean` | `NmagConfig(output_policy="replace")` |
| Process-wide solver flags | Per-simulation `NmagConfig` and `set_params` |
| `nmagpp`/`ncol` analysis | H5py, NumPy, pandas, Meshio, or other current tools |
| Legacy restart files | Native `save_restart_file`/`load_restart_file` |

The core construction pattern remains familiar:

```python
import nmag

material = nmag.MagMaterial(
    name="Py",
    Ms=nmag.SI(1e6, "A/m"),
    exchange_coupling=nmag.SI(13e-12, "J/m"),
)

simulation = nmag.Simulation(name="ported")
simulation.load_mesh(
    "legacy.nmesh.h5",
    [("magnetic", material)],
    unit_length=nmag.SI(1e-9, "m"),
)
simulation.set_m([1, 0, 0])
simulation.set_H_ext([0, 0, 0], nmag.SI("A/m"))
simulation.relax()
```

## Do not translate blindly

Remove old command-line options and imports before debugging physics. In
particular, do not carry over:

- Python 2 syntax;
- imports from `ocaml`, `nsim`, or old `nmag` implementation modules;
- CVODE/PVODE, PETSc, HLib, or MPI configuration arguments;
- assumptions that legacy output helpers or restart formats exist; or
- a physics option listed as unsupported in the
  [current scope](supported-scope.md).

The rewrite uses SciPy DOP853 by default, so accepted step counts and relaxation
times are not expected to match legacy CVODE exactly. Start comparison at fixed
physical times, then compare the spatial magnetization and effective fields.

## Porting checklist

1. Make the script valid Python 3 and replace legacy imports.
2. Run it with Python, not `nsim`.
3. Add an explicit `NmagConfig`, especially for output lifecycle.
4. Confirm mesh regions and `unit_length` before assigning materials.
5. Replace unsupported physics with a reviewed model decision; do not silently
   omit it.
6. Establish one small fixed-time or static reference before running a long
   relaxation.
7. Inspect NDT/HDF5 data directly with maintained Python tools.

The historical [Nmag 0.2 manual](https://nmag.readthedocs.io/en/latest/) is a
valuable description of the original interface and micromagnetic background.
Use this manual for the Python 3 contract whenever the two differ.
