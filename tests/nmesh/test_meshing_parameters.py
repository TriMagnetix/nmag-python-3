from nmesh.mesher.meshing_parameters import (
    MeshingParameters,
    PointFate,
)


def test_meshing_parameters_defaults():
    params = MeshingParameters()
    assert params["controller_shape_force_scale"] == 0.1
    assert params["controller_volume_force_scale"] == 0.0
    assert params["controller_neigh_force_scale"] == 1.0
    assert params["controller_step_limit_max"] == 1000


def test_meshing_parameters_setters():
    params = MeshingParameters()
    params.dim = 3
    params.set_shape_force_scale(0.5)
    assert params["shape_force_scale"] == 0.5
    assert params["controller_shape_force_scale"] == 0.5

    params.set_max_steps(2000)
    assert params["max_steps"] == 2000
    assert params["controller_step_limit_max"] == 2000


def test_meshing_parameters_preserve_public_setter_api():
    """Verify all public parameter setters are dynamically generated."""
    params = MeshingParameters()
    for setter_name in (
        "set_shape_force_scale",
        "set_volume_force_scale",
        "set_neigh_force_scale",
        "set_irrel_elem_force_scale",
        "set_time_step_scale",
        "set_thresh_add",
        "set_thresh_del",
        "set_topology_threshold",
        "set_tolerated_rel_move",
        "set_max_steps",
        "set_initial_settling_steps",
        "set_sliver_correction",
        "set_smallest_volume_ratio",
        "set_max_relaxation",
    ):
        assert callable(getattr(params, setter_name))


def test_meshing_parameters_getitem_setitem():
    params = MeshingParameters()
    params.dim = 3
    params["custom_param"] = 123
    assert params["custom_param"] == 123

    # Test overriding defaults
    params["controller_shape_force_scale"] = 0.9
    assert params["shape_force_scale"] == 0.9
    assert params["controller_shape_force_scale"] == 0.9

    params["max_steps"] = 1500
    assert params["controller_step_limit_max"] == 1500
    assert params["max_steps"] == 1500


def test_meshing_parameters_copy():
    params = MeshingParameters()
    params["val"] = 1
    params2 = params.copy()
    params2["val"] = 2
    assert params["val"] == 1
    assert params2["val"] == 2


def test_meshing_parameters_apply_to_mesher():
    params = MeshingParameters()
    params["shape_force_scale"] = 0.5
    params["max_steps"] = 2000

    mesher = {"parameters": {"existing": 1}}
    params.apply_to_mesher(mesher, 3)

    assert mesher["parameters"]["existing"] == 1
    assert mesher["parameters"]["controller_shape_force_scale"] == 0.5
    assert mesher["parameters"]["controller_step_limit_max"] == 2000


def test_meshing_parameters_can_load_config_string():
    """Test loading config with public API names (user-friendly INI format)."""
    params = MeshingParameters(
        string="""
[nmesh-3D]
shape_force_scale : 0.25
max_steps : 1500
"""
    )
    params.dim = 3

    # Can access by public name
    assert params["shape_force_scale"] == 0.25
    # Can also access by internal name
    assert params["controller_shape_force_scale"] == 0.25
    assert params["max_steps"] == 1500
    assert params["controller_step_limit_max"] == 1500


def test_to_mesher_config_returns_resolved_dict():
    """Test that to_mesher_config returns dict with internal parameter names.

    This verifies the public API → internal name translation. Users provide
    concise names, but the mesher receives verbose, namespaced names.
    """
    params = MeshingParameters()
    params["shape_force_scale"] = 0.7  # Public API name
    params["max_steps"] = 1500

    config = params.to_mesher_config(dim=3)

    # Should have internal names (what mesher expects)
    assert config["controller_shape_force_scale"] == 0.7
    assert config["controller_step_limit_max"] == 1500

    # Should NOT have public API names (mesher uses internal names)
    assert "shape_force_scale" not in config
    assert "max_steps" not in config

    # Should have all public parameters with correct types
    assert isinstance(config["controller_shape_force_scale"], float)
    assert isinstance(config["controller_step_limit_max"], int)


def test_internal_parameters_have_defaults():
    """Test that non-public internal parameters have correct defaults."""
    params = MeshingParameters()

    # Volume determination
    assert params["nr_probes_for_determining_volume"] == 100000

    # Boundary condition parameters
    assert params["boundary_condition_acceptable_fuzz"] == 1e-6
    assert params["boundary_condition_max_nr_correction_steps"] == 200
    assert params["boundary_condition_debuglevel"] == 0

    # Relaxation parameters
    assert params["relaxation_debuglevel"] == 0
    assert params["controller_step_limit_min"] == 500
    assert params["controller_max_time_step"] == 10.0

    # Function parameters should be callable
    assert callable(params["initial_relaxation_weight_fun"])
    assert callable(params["relaxation_force_fun"])
    assert callable(params["boundary_node_force_fun"])
    assert callable(params["handle_point_density_fun"])


def test_dimension_specific_config_overrides():
    """Test that dimension-specific sections override defaults correctly."""
    params = MeshingParameters(
        string="""
[nmesh-2D]
max_steps : 500
shape_force_scale : 0.2

[nmesh-3D]
max_steps : 1000
shape_force_scale : 0.3
"""
    )

    # Test 2D dimension
    params.dim = 2
    assert params["max_steps"] == 500
    assert params["shape_force_scale"] == 0.2

    # Test 3D dimension
    params.dim = 3
    assert params["max_steps"] == 1000
    assert params["shape_force_scale"] == 0.3

    # Verify internal names also work
    assert params["controller_step_limit_max"] == 1000
    assert params["controller_shape_force_scale"] == 0.3


def test_user_modifications_override_dimension_sections():
    """Test that user-modifications take precedence over dimension sections."""
    params = MeshingParameters(
        string="""
[nmesh-3D]
max_steps : 1000
"""
    )
    params.dim = 3

    # User modification should override dimension section
    params["max_steps"] = 2000
    assert params["max_steps"] == 2000
    assert params["controller_step_limit_max"] == 2000


def test_default_functions_are_assigned_correctly():
    """Verify default force functions are assigned and work correctly."""
    params = MeshingParameters()

    # Test initial relaxation weight function
    weight_fun = params["initial_relaxation_weight_fun"]
    assert weight_fun(0, 100, 1.0, 5.0) == 1.0  # At start
    assert weight_fun(50, 100, 1.0, 5.0) == 3.0  # At middle
    assert weight_fun(100, 100, 1.0, 5.0) == 5.0  # At end

    # Test relaxation force function
    force_fun = params["relaxation_force_fun"]
    assert force_fun(0.5) == 0.5
    assert force_fun(1.5) == 0.0

    # Test boundary node force function
    boundary_fun = params["boundary_node_force_fun"]
    assert boundary_fun(0.5) == 1.0  # 1/0.5 - 1 = 1.0
    assert boundary_fun(1.5) == 0.0

    # Test handle point density function
    density_fun = params["handle_point_density_fun"]

    class MockRng:
        def random(self):
            return 0.99  # High value = do nothing

    result = density_fun(MockRng(), (1.5, 0.3), 1.0, 2.0)
    assert result == PointFate.DO_NOTHING
