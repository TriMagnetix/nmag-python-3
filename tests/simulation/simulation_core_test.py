"""Tests for the shared simulation core contract."""

from inspect import signature

from simulation.clock import SimulationClock
from simulation.simulation_core import SimulationCore


def test_simulation_core_uses_package_sibling_imports():
    assert SimulationCore.__module__ == "simulation.simulation_core"
    assert SimulationClock.__module__ == "simulation.clock"


def test_simulation_core_sequence_defaults_are_immutable():
    spatial_default = signature(SimulationCore.save_spatial_fields).parameters[
        "fieldnames"
    ].default
    restart_default = signature(SimulationCore.save_restart_file).parameters[
        "fieldnames"
    ].default

    assert spatial_default is None
    assert restart_default is None
