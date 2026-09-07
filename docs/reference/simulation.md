# Simulation API

This page contains the supported, user-facing simulation surface. Compatibility
helpers and internal solver methods are intentionally omitted.

::: nmag.simulation.Simulation
    options:
      members:
        - id
        - stage
        - step
        - stage_step
        - time
        - stage_time
        - real_time
        - last_step_dt
        - last_bem_operator_stats
        - last_integrator_stats
        - integrator_config
        - effective_integrator_max_step
        - load_mesh
        - set_m
        - set_H_ext
        - set_pinning
        - set_current_density
        - get_all_field_names
        - is_subfield_available
        - get_subfield
        - get_subfield_average
        - get_maxangle_average
        - probe_subfield
        - probe_subfield_siv
        - save_data
        - save_spatial_fields
        - save_mesh
        - advance_time
        - relax
        - set_params
        - reinitialise
        - get_restart_file_name
        - save_restart_file
        - load_restart_file
        - load_m_from_h5file
