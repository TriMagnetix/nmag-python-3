# Dynamics and relaxation

Nmag integrates the Landau-Lifshitz-Gilbert dynamics of the nodal magnetization.
The default backend is SciPy DOP853 with relative and absolute tolerances of
`1e-6` and a configured maximum step of 1 ps.

For meshes with exchange coupling, Nmag may derive a smaller effective maximum
step to resolve the fastest lumped-FEM exchange mode. Inspect both the configured
and effective values:

```python
print(simulation.integrator_config)
print(simulation.effective_integrator_max_step)
```

## Advance to a physical time

```python
reached = simulation.advance_time(nmag.SI(10e-12, "s"))
print(reached, simulation.last_integrator_stats)
```

The target cannot precede the current stage time. `max_it` can cap accepted
steps for interactive work, and `exact_tstop` controls whether the final state
is reconstructed at the requested time.

## Relax a state

```python
simulation.relax()
print(simulation.time)
print(simulation.step)
```

The default convergence threshold is one degree per nanosecond. It is checked
every five accepted integration steps and must be satisfied twice consecutively.
Because that cadence counts accepted steps, a different integrator can confirm
the same physical threshold at a different simulated time. Compare spatial stop
states as well as averages when validating a workflow.

Tune a simulation before starting integration:

```python
simulation.set_params(
    stopping_dm_dt=nmag.SI(0.5e9 * 3.141592653589793 / 180.0, "1/s"),
    ts_rel_tol=1e-7,
    ts_abs_tol=1e-7,
    ts_max_step=nmag.SI(0.5e-12, "s"),
)
```

Tighter tolerances do not compensate for an under-resolved mesh. Perform mesh,
step, and tolerance convergence checks separately.

## Scheduled saves

The default relaxation saves averages and fields at the end of the stage. A
custom schedule uses `when` specifications:

```python
from when import at, every

simulation.relax(
    save=[
        ("averages", every("step", 10)),
        ("fields", at("stage_end")),
        ("restart", at("stage_end")),
    ]
)
```

The public abbreviations normalize to `save_averages`, `save_fields`, and
`save_restart`. Use simulated-time schedules when comparisons must be independent
of adaptive step counts.

## Experimental Diffsol backend

After building the Rust accelerator:

```python
config = nmag.NmagConfig(integrator_backend="diffsol", accelerator="rust")
simulation = nmag.Simulation(config=config)
```

Diffsol is an opt-in dense implicit relaxation backend. It supports the default
relaxation schedule for dense isotropic systems, but not `advance_time`, custom
save/convergence schedules, anisotropy, or low-memory mode. SciPy remains the
general production backend.
