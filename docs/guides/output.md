# Fields, output, and restarts

## Discover and read fields

After loading a mesh and setting magnetization:

```python
print(simulation.get_all_field_names())
print(simulation.is_subfield_available("H_demag"))

nodal_field = simulation.get_subfield("H_demag")
average_field = simulation.get_subfield_average("H_demag")
```

Common fields include magnetization direction `m`, magnetization `M`, applied
field `H_ext`, exchange field `H_exch`, demagnetization field `H_demag`,
anisotropy field `H_anis`, total field `H_total`, energy densities, pinning, and
the scalar-potential quantities `phi` and `rho`. Availability depends on the
loaded materials and whether demagnetization is enabled.

Material-specific averages can be requested by material name:

```python
py_average = simulation.get_subfield_average("m", "Py")
```

## Probe a position

`probe_subfield_siv` takes positions in metres and returns ordinary SI values:

```python
field = simulation.probe_subfield_siv("H_demag", [0.0, 0.0, 0.0])
```

It returns `None` outside the mesh. `probe_subfield` provides the same behavior
and can accept an explicit unit argument.

## NDT and HDF5 output

```python
simulation.save_data()                  # averaged quantities only
simulation.save_data(fields=["m", "H_total"])
simulation.save_data(fields="all")      # every available spatial field
```

For a simulation named `sample`, Nmag writes:

- `sample_dat.ndt`: tab-separated, SI-labelled averaged quantities, one row per
  save event;
- `sample_dat.h5`: mesh metadata and requested nodal or cell fields.

Use HDF5 readers such as H5py for analysis. The current rewrite does not ship
the legacy `ncol` or `nmagpp` executables.

To write selected spatial fields to another compact HDF5 file:

```python
simulation.save_spatial_fields("snapshot.h5", ["m", "H_demag"])
```

## Output lifecycle

`NmagConfig.output_policy` controls files created by a new simulation:

- `"error"` refuses to overwrite existing NDT/HDF5 output;
- `"replace"` starts those output files again; and
- `"append"` validates the old NDT schema before adding rows.

Append is an output choice, not a physical restart.

## Restart checkpoints

```python
checkpoint = simulation.save_restart_file("relaxed_state.h5")

# Create a compatible simulation and load the same mesh/materials first.
continued.load_restart_file(checkpoint)
```

Native checkpoints preserve magnetization, pinning, current density, external
field, clock, and supported dynamics state. Mesh and material fingerprints must
match. DOP853 is reconstructed from the physical state, so later adaptive step
sizes need not reproduce an earlier run exactly.

`load_m_from_h5file(path)` transfers only magnetization and requires the same
mesh, making it useful when other simulation settings should remain new.

These are native Python 3 checkpoint files. Legacy Nmag restart files are not
supported.
