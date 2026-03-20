import pytest
from nmesh.mesher import MeshingParameters

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
