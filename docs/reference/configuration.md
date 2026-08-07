# Configuration and diagnostics API

## Simulation configuration

::: nmag.config.NmagConfig
    options:
      members:
        - from_environment
        - accelerator_mode_for

::: nmag.config.HierarchicalBemConfig
    options:
      members: false

::: nmag.config.RustKernel
    options:
      members: true

## Runtime diagnostics

::: nmag.dynamics.IntegratorConfig
    options:
      members: false

::: nmag.dynamics.IntegratorStats
    options:
      members: false

::: nmag.demag.bem_operator.BemOperatorStats
    options:
      members: false

::: nmag.parallel.ParallelRuntimeInfo
    options:
      members: false

::: nmag.parallel.parallel_runtime_info
