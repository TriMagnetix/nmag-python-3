# Quickstart

This example constructs one tetrahedron, assigns a magnetic material, computes
the static fields, and writes both averaged and spatial output. It is deliberately
small: its purpose is to verify the complete software path, not to represent a
well-resolved physical body.

[Download `quickstart.py`](../examples/quickstart.py){ .md-button }

```python title="quickstart.py"
--8<-- "docs/examples/quickstart.py"
```

Run it from an installed checkout:

```bash
.venv/bin/python quickstart.py
```

The script creates a `results/` directory containing:

- `quickstart.nmesh.h5`, the generated mesh;
- `quickstart_dat.ndt`, a tab-separated row of averaged quantities; and
- `quickstart_dat.h5`, spatial field arrays and mesh metadata.

The mesh coordinates are dimensionless until `load_mesh` applies
`unit_length`. Here one coordinate unit is one nanometre. Material parameters
and fields use explicit [SI quantities](../guides/materials.md#si-quantities).

`output_policy="replace"` makes this learning example safe to rerun. For normal
work, the default `"error"` policy protects existing result files; use
checkpoints when a run must continue from earlier state.

Next, use a resolved geometry in the [sphere tutorial](sphere.md).
