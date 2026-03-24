import pytest
import numpy as np
from nmesh.mesher import (
    MeshingParameters,
    default_initial_relaxation_weight,
    default_relaxation_force_fun,
    default_boundary_node_force_fun,
    default_handle_point_density_fun,
    PointFate,
    EPSILON_DIVISION_SAFETY,
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

def test_meshing_parameters_preserve_legacy_setter_api():
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

def test_meshing_parameters_can_load_legacy_config_string():
    params = MeshingParameters(
        string="""
[nmesh-3D]
shape_force_scale : 0.25
max_steps : 1500
"""
    )
    params.dim = 3

    assert params["shape_force_scale"] == 0.25
    assert params["controller_shape_force_scale"] == 0.25
    assert params["max_steps"] == 1500
    assert params["controller_step_limit_max"] == 1500


# Test default force functions
def test_default_initial_relaxation_weight_normal_case():
    # At step 0, should be init_val
    assert default_initial_relaxation_weight(0, 100, 1.0, 5.0) == 1.0

    # At step 50 (halfway), should be midpoint
    assert default_initial_relaxation_weight(50, 100, 1.0, 5.0) == 3.0

    # At step 100 (max), should be final_val
    assert default_initial_relaxation_weight(100, 100, 1.0, 5.0) == 5.0

    # Beyond max_step, should saturate at final_val
    assert default_initial_relaxation_weight(150, 100, 1.0, 5.0) == 5.0


def test_default_initial_relaxation_weight_edge_cases():
    # When max_step <= 0, should return final_val immediately
    assert default_initial_relaxation_weight(0, 0, 1.0, 5.0) == 5.0
    assert default_initial_relaxation_weight(50, 0, 1.0, 5.0) == 5.0
    assert default_initial_relaxation_weight(0, -10, 1.0, 5.0) == 5.0


def test_default_relaxation_force_fun():
    # Distance > 1.0 should return 0
    assert default_relaxation_force_fun(1.5) == 0.0
    assert default_relaxation_force_fun(2.0) == 0.0

    # At exactly 1.0
    assert default_relaxation_force_fun(1.0) == 0.0

    # At 0.5, force should be 0.5
    assert default_relaxation_force_fun(0.5) == 0.5

    # At 0.0, force should be 1.0 (maximum repulsion)
    assert default_relaxation_force_fun(0.0) == 1.0


def test_default_boundary_node_force_fun():
    # Distance > 1.0 should return 0
    assert default_boundary_node_force_fun(1.5) == 0.0
    assert default_boundary_node_force_fun(2.0) == 0.0

    # At exactly 1.0
    assert default_boundary_node_force_fun(1.0) == 0.0

    # Very small distance should return large repulsion
    assert default_boundary_node_force_fun(EPSILON_DIVISION_SAFETY / 2) == 1e12
    assert default_boundary_node_force_fun(0.0) == 1e12

    # Normal distance: 1/d - 1
    result = default_boundary_node_force_fun(0.5)
    expected = 1.0 / 0.5 - 1.0
    assert abs(result - expected) < 1e-10


def test_default_handle_point_density_fun_low_density():
    # Mock RNG that always returns low value to trigger add
    class MockRng:
        def random(self):
            return 0.05  # < 0.1, will trigger add

    rng = MockRng()
    result = default_handle_point_density_fun(rng, (0.5, 0.1), 1.0, 2.0)
    assert result == PointFate.ADD_ANOTHER


def test_default_handle_point_density_fun_low_force():
    # Mock RNG for force-based add
    class MockRng:
        def random(self):
            return 0.15  # < 0.2, will trigger add

    rng = MockRng()
    # Density okay but force very low
    result = default_handle_point_density_fun(rng, (1.5, 0.05), 1.0, 2.0)
    assert result == PointFate.ADD_ANOTHER


def test_default_handle_point_density_fun_high_density():
    # Mock RNG for density-based delete
    class MockRng:
        def random(self):
            return 0.25  # Will trigger delete with right probability

    rng = MockRng()
    # Density too high
    result = default_handle_point_density_fun(rng, (2.5, 0.2), 1.0, 2.0)
    assert result == PointFate.DELETE


def test_default_handle_point_density_fun_high_force():
    # Mock RNG for force-based delete
    class MockRng:
        def random(self):
            return 0.35  # Will trigger delete with right probability

    rng = MockRng()
    # Force too high
    result = default_handle_point_density_fun(rng, (1.5, 0.6), 1.0, 2.0)
    assert result == PointFate.DELETE


def test_default_handle_point_density_fun_do_nothing():
    # Mock RNG that returns high values (won't trigger anything)
    class MockRng:
        def random(self):
            return 0.99

    rng = MockRng()
    # Normal conditions, high RNG value = do nothing
    result = default_handle_point_density_fun(rng, (1.5, 0.3), 1.0, 2.0)
    assert result == PointFate.DO_NOTHING
