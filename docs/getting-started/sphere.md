# Uniformly magnetized sphere

A uniformly magnetized sphere has an analytic demagnetization field at its
centre: the component parallel to the magnetization is approximately
`-Ms / 3`. This classic Nmag example therefore gives a compact physical check of
mesh loading, material assignment, the FEM/BEM demagnetization calculation,
spatial output, and field probing.

Download both files into the same directory:

[Download the script](../examples/sphere_demag.py){ .md-button .md-button--primary }
[Download the mesh](../assets/sphere1.nmesh.h5){ .md-button }

```python title="sphere_demag.py"
--8<-- "docs/examples/sphere_demag.py"
```

Run the example:

```bash
.venv/bin/python sphere_demag.py
```

You can also pass an explicit mesh path:

```bash
.venv/bin/python sphere_demag.py /path/to/sphere1.nmesh.h5
```

The script prints the interpolated `H_demag` at the origin and the analytic
x component. The supplied mesh agrees at the percent level. It also writes
`sphere1_dat.ndt` and `sphere1_dat.h5` under `results/`.

## What the example establishes

- The mesh contains region `1`, named `"sphere"` when it is mapped to the
  material in `load_mesh`.
- Mesh coordinates are in nanometres because `unit_length` is `SI(1e-9, "m")`.
- `set_m([1, 0, 0])` assigns a normalized, uniform magnetization direction.
- `save_data(fields="all")` calculates and stores every available field.
- `probe_subfield_siv` accepts a position in metres and returns ordinary SI
  values, which is convenient for plotting and numerical analysis.

The example is tested against the same pinned mesh used by the production
solver tests. For a new scientific model, repeat this pattern with an analytic,
legacy, or convergence reference appropriate to that model.
