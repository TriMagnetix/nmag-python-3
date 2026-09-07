# Simulation workflow

## 1. Choose output behavior

`NmagConfig` holds immutable choices for one simulation. Give each study a
dedicated output directory and choose deliberately how existing files are
handled:

```python
from pathlib import Path

import nmag

results = Path("results")
results.mkdir(exist_ok=True)
config = nmag.NmagConfig(
    output_directory=results,
    output_policy="error",
    accelerator="auto",
)
simulation = nmag.Simulation(name="sample", config=config)
```

The default `"error"` policy protects existing NDT and HDF5 files. `"replace"`
starts those files again, while `"append"` checks the existing NDT schema and
adds rows. Append does not restore physical state; use a restart file for that.

## 2. Prepare a mesh and materials

The simulation solver expects a 3D tetrahedral mesh whose cells carry integer
region IDs. A material definition supplies saturation magnetization and the
physics enabled for that region:

```python
permalloy = nmag.MagMaterial(
    name="Py",
    Ms=nmag.SI(1e6, "A/m"),
    exchange_coupling=nmag.SI(13e-12, "J/m"),
    llg_damping=0.02,
)

simulation.load_mesh(
    "sample.msh",
    [("magnetic", permalloy)],
    unit_length=nmag.SI(1e-9, "m"),
)
```

The material list is ordered by mesh region: the first entry maps to region 1,
the second to region 2, and so on. Nmag checks that configured and actual
regions match. Region names then appear in field and output labels.

`unit_length` converts mesh coordinate units into metres. If a Gmsh file already
stores metres, use `nmag.SI(1, "m")`.

## 3. Set the state and applied fields

Magnetization is a unit direction vector. Assign it uniformly, per node, or
with a callable evaluated at physical positions in metres:

```python
simulation.set_m([1.0, 0.0, 0.0])

# A position-dependent alternative:
# simulation.set_m(lambda position: [1, 0, 0] if position[0] < 0 else [0, 1, 0])
```

The applied field can be supplied as a vector plus a unit:

```python
simulation.set_H_ext([0.0, 0.0, 8e3], nmag.SI("A/m"))
```

Pinning and Zhang-Li current density are optional:

```python
simulation.set_pinning(lambda position: 0.0 if position[2] < 2e-9 else 1.0)
simulation.set_current_density([0.0, 0.0, 1e12], nmag.SI("A/m^2"))
```

A pin value of zero fixes the complete local magnetization derivative; one
leaves it free.

## 4. Calculate, advance, or relax

Field access triggers the required calculation lazily:

```python
average_demag = simulation.get_subfield_average("H_demag")
```

For dynamics, either advance to a requested physical time or relax until the
configured convergence test passes:

```python
simulation.advance_time(nmag.SI(1e-12, "s"))
simulation.relax()
```

## 5. Save and inspect results

```python
simulation.save_data(fields="all")
print(simulation.time)
print(simulation.last_integrator_stats)
```

See [fields, output, and restarts](output.md) for selective output and safe
continuation, and [dynamics and relaxation](dynamics.md) before tuning an
integrator.
