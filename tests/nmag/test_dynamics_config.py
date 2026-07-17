from __future__ import annotations

import pytest

from nmag.dynamics import ConvergenceTracker, IntegratorConfig
from si.physical import SI
from simulation.clock import SimulationClock


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relative_tolerance", 0.0),
        ("absolute_tolerance", -1.0),
        ("maximum_step_seconds", float("inf")),
    ],
)
def test_integrator_config_rejects_invalid_positive_values(field, value):
    config = IntegratorConfig()
    setattr(config, field, value)

    with pytest.raises(ValueError, match=field):
        config.validate()


def test_convergence_requires_consecutive_checks_below_threshold():
    tracker = ConvergenceTracker(required_checks=2)

    assert not tracker.check(step=5, max_dm_dt=0.5, stopping_dm_dt=1.0)
    assert not tracker.check(step=10, max_dm_dt=2.0, stopping_dm_dt=1.0)
    assert not tracker.check(step=15, max_dm_dt=0.5, stopping_dm_dt=1.0)
    assert tracker.check(step=20, max_dm_dt=0.25, stopping_dm_dt=1.0)

    assert not tracker.check(step=25, max_dm_dt=0.25, stopping_dm_dt=1.0)


def test_convergence_log_records_nonconverged_checks():
    tracker = ConvergenceTracker()
    tracker.check(step=5, max_dm_dt=2.0, stopping_dm_dt=1.0)

    log = tracker.get_log()

    assert "# Convergence log" in log
    assert "5 2.0 1.0" in log


def test_clock_records_an_advance_atomically():
    clock = SimulationClock()

    clock.record_advance(
        stage_time_seconds=2.0e-12,
        accepted_steps=3,
        last_step_seconds=0.5e-12,
        wall_seconds=0.25,
    )

    assert clock.step == 3
    assert clock.stage_step == 3
    seconds = SI(1.0, "s")
    assert clock.time.in_units_of(seconds) == pytest.approx(2.0e-12)
    assert clock.stage_time.in_units_of(seconds) == pytest.approx(2.0e-12)
    assert clock.last_step_dt_si.in_units_of(seconds) == pytest.approx(0.5e-12)
    assert clock.real_time.in_units_of(seconds) == pytest.approx(0.25)


def test_clock_rejects_backwards_stage_time():
    clock = SimulationClock()
    clock.record_advance(
        stage_time_seconds=2.0,
        accepted_steps=1,
        last_step_seconds=2.0,
        wall_seconds=0.0,
    )

    with pytest.raises(ValueError, match="backwards"):
        clock.record_advance(
            stage_time_seconds=1.0,
            accepted_steps=1,
            last_step_seconds=1.0,
            wall_seconds=0.0,
        )
