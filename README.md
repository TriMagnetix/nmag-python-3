# Nmag for Python 3

`nmag-python-3` is a standalone Python implementation of the Nmag finite-element
micromagnetic simulation interface. It supports tetrahedral meshes, magnetic
materials, static effective fields, adaptive LLG dynamics and relaxation,
NDT/HDF5 output, field probing, and native restart checkpoints.

The human documentation is available at
**<https://trimagnetix.github.io/nmag-python-3/>**. Start with the
[installation guide](https://trimagnetix.github.io/nmag-python-3/getting-started/installation/)
and [quickstart](https://trimagnetix.github.io/nmag-python-3/getting-started/quickstart/).

## Requirements

- Linux
- Python 3.10 or newer
- A C compiler for Python dependencies without a platform wheel
- Rust and the matching Python development library only for optional native
  acceleration

## Installation

From a clone of this repository:

```bash
git switch mvp
./scripts/setup.sh
```

The setup script creates or reuses `.venv`, offers to build the optional Rust
accelerator, and runs the standard checks. Run a simulation with:

```bash
.venv/bin/python simulation.py
```

## Minimal simulation

```python
import nmag
import nmesh

mesh = nmesh.mesh_from_points_and_simplices(
    points=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    simplices_indices=[[0, 1, 2, 3]],
    simplices_regions=[1],
)
mesh.save("sample.nmesh.h5")

material = nmag.MagMaterial(
    name="Py",
    Ms=nmag.SI(1e6, "A/m"),
    exchange_coupling=nmag.SI(13e-12, "J/m"),
)

simulation = nmag.Simulation(name="sample")
simulation.load_mesh(
    "sample.nmesh.h5",
    [("magnetic", material)],
    unit_length=nmag.SI(1e-9, "m"),
)
simulation.set_m([1, 0, 0])
simulation.set_H_ext([0, 0, 0], nmag.SI("A/m"))
simulation.save_data(fields="all")
```

For a tested, rerunnable version and a canonical sphere example, see the
[getting-started guide](https://trimagnetix.github.io/nmag-python-3/getting-started/quickstart/).

## Status

The supported solver covers 3D tetrahedral demagnetization, exchange, uniform
applied fields, uniaxial and cubic anisotropy, custom polynomial anisotropy,
pinning, Zhang-Li current torque, adaptive dynamics and relaxation, checkpoints,
and resource-aware demagnetization storage.

Thermal dynamics, Slonczewski torque, periodic micromagnetic boundaries,
material-specific magnetization at incompatible shared nodes, local
inter-material coupling, and full legacy hysteresis compatibility are not yet
supported. Validate new geometries and material models before relying on
production results. The complete and current list is maintained in the
[supported-scope guide](https://trimagnetix.github.io/nmag-python-3/guides/supported-scope/).

## Configuration and acceleration

Each simulation accepts an immutable `NmagConfig`. `accelerator="auto"` uses
installed Rust kernels where helpful, `"off"` keeps Python/Numba paths, and
`"rust"` requires the extension. Advanced callers can use
`accelerator_overrides={nmag.RustKernel.LLG: "rust"}`.

For a simulation created without an explicit configuration,
`NMAG_ACCELERATOR=auto|off|rust` selects the process default. Prefer
`NmagConfig` in reusable programs. See the
[configuration guide](https://trimagnetix.github.io/nmag-python-3/guides/configuration/)
for storage, memory, and integrator choices.

## Development checks

```bash
./scripts/verify.sh
./scripts/verify.sh --rust
```

Build the documentation locally with:

```bash
.venv/bin/python -m pip install -e '.[docs]'
.venv/bin/mkdocs build --strict
```

## Project lineage

This project modernizes the original
[nmag-project/nmag-src](https://github.com/nmag-project/nmag-src). The
[historical Nmag 0.2 manual](https://nmag.readthedocs.io/en/latest/) remains a
useful background reference, but its Python 2 runtime and some of its features
do not describe this rewrite.

Nmag for Python 3 is distributed under the GNU General Public License version 2
or, at your option, any later version. See [LICENSE](LICENSE).
