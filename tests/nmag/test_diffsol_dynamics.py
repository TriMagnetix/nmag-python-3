from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import nmag
import nmag.backends as backend_module
import nmag.simulation.implicit_dynamics as implicit_dynamics
import nmesh
from nmag.simulation.implicit_dynamics import build_affine_field_operator


def _simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    do_demag: bool = False,
    config: nmag.NmagConfig | None = None,
) -> nmag.Simulation:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / f"{name}.nmesh.h5"
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[1],
    )
    mesh.save(str(mesh_path))
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
        llg_damping=0.5,
        do_precession=False,
    )
    simulation = nmag.Simulation(name=name, do_demag=do_demag, config=config)
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    return simulation


def test_integrator_backend_defaults_to_scipy() -> None:
    assert backend_module._selected_integrator_backend(nmag.NmagConfig()) == "scipy"


def test_integrator_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="integrator_backend"):
        nmag.NmagConfig(integrator_backend="unknown")  # type: ignore[arg-type]


def test_low_memory_mode_rejects_dense_diffsol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(backend_module.MEMORY_MODE_ENV, "low")
    with pytest.raises(ValueError, match="dense affine operator"):
        backend_module._selected_integrator_backend(
            nmag.NmagConfig(integrator_backend="diffsol")
        )


def test_diffsol_dense_memory_guard_has_no_fixed_state_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(implicit_dynamics.DIFFSOL_DENSE_MEMORY_LIMIT_ENV, raising=False)
    monkeypatch.setattr(implicit_dynamics, "_available_memory_bytes", lambda: 1024**3)

    implicit_dynamics._validate_diffsol_dense_memory(1620)


def test_diffsol_dense_memory_guard_reports_required_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(implicit_dynamics.DIFFSOL_DENSE_MEMORY_LIMIT_ENV, "0.01")

    with pytest.raises(MemoryError, match="There is no fixed state-count limit"):
        implicit_dynamics._validate_diffsol_dense_memory(1620)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "not-memory"])
def test_diffsol_dense_memory_guard_rejects_invalid_budgets(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(implicit_dynamics.DIFFSOL_DENSE_MEMORY_LIMIT_ENV, value)

    with pytest.raises(ValueError, match=implicit_dynamics.DIFFSOL_DENSE_MEMORY_LIMIT_ENV):
        implicit_dynamics._diffsol_dense_memory_budget()


def test_diffsol_dense_memory_guard_allows_explicit_unlimited_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(implicit_dynamics.DIFFSOL_DENSE_MEMORY_LIMIT_ENV, "unlimited")
    monkeypatch.setattr(implicit_dynamics, "_available_memory_bytes", lambda: 1)

    implicit_dynamics._validate_diffsol_dense_memory(100_000)


def test_affine_operator_reproduces_dynamic_field_and_is_cached(tmp_path, monkeypatch) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="affine", do_demag=True)
    simulation.set_m([0.25, 0.5, 0.75])
    state = np.ravel(np.asarray(simulation.get_subfield("m")))

    operator = build_affine_field_operator(simulation)
    dynamic_field = np.ravel(
        simulation._get_exchange_nodal_field() + simulation._get_demag_nodal_field()
    )

    np.testing.assert_allclose(operator @ state, dynamic_field, rtol=2.0e-11, atol=1.0e-7)
    assert build_affine_field_operator(simulation) is operator
    simulation._invalidate_demag(clear_geometry=True)
    assert simulation._llg_affine_operator_cache is None


def test_diffsol_relaxation_aligns_with_external_field(tmp_path, monkeypatch) -> None:
    rust_accel = pytest.importorskip("nmag_accel")
    if not hasattr(rust_accel, "integrate_llg_bdf"):
        pytest.skip("installed nmag_accel does not include the Diffsol backend")
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="diffsol-relax",
        config=nmag.NmagConfig(integrator_backend="diffsol", accelerator="rust"),
    )
    simulation.set_m([0.0, 1.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))

    simulation.relax()

    average = np.asarray(simulation.get_subfield_average("m"))
    assert average[0] > 0.999
    assert abs(average[1]) < 3.0e-3
    assert simulation.clock.convergence is True
    assert simulation.last_integrator_stats.status == "converged"
    assert simulation.last_integrator_stats.rejected_steps is not None
    assert simulation.last_integrator_stats.nonlinear_iterations is not None


def test_diffsol_rejects_custom_schedules(tmp_path, monkeypatch) -> None:
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="diffsol-schedule",
        config=nmag.NmagConfig(integrator_backend="diffsol", accelerator="rust"),
    )
    simulation.set_m([1.0, 0.0, 0.0])

    with pytest.raises(NotImplementedError, match="default relaxation"):
        simulation.relax(save=[])


def test_diffsol_failure_updates_integrator_statistics(tmp_path, monkeypatch) -> None:
    import nmag.simulation as simulation_module

    monkeypatch.setattr(
        simulation_module,
        "_load_rust_accelerator",
        lambda _required_by: SimpleNamespace(),
        raising=False,
    )
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="diffsol-failure",
        config=nmag.NmagConfig(
            integrator_backend="diffsol",
            accelerator_overrides={kernel: "off" for kernel in nmag.RustKernel},
        ),
    )
    simulation.set_m([1.0, 0.0, 0.0])

    with pytest.raises(RuntimeError, match="Diffsol backend"):
        simulation.relax()

    assert simulation.last_integrator_stats.failed is True
    assert simulation.last_integrator_stats.status == "failed"


def test_diffsol_kernel_matches_scipy_at_a_fixed_time() -> None:
    rust_accel = pytest.importorskip("nmag_accel")
    if not hasattr(rust_accel, "integrate_llg_bdf"):
        pytest.skip("installed nmag_accel does not include the Diffsol backend")
    initial = np.asarray([0.0, 1.0, 0.0])
    constant_field = np.asarray([1.0e5, 0.0, 0.0])
    coefficients = (0.0, -8.0e4, 0.0)
    final_time = 20.0e-12

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        mdoth = float(np.dot(state, constant_field))
        mdotm = float(np.dot(state, state))
        damping = state * mdoth - constant_field * mdotm
        return coefficients[1] * damping

    scipy_result = solve_ivp(
        rhs,
        (0.0, final_time),
        initial,
        method="DOP853",
        rtol=1.0e-9,
        atol=1.0e-11,
        max_step=1.0e-12,
    )
    diffsol_result = rust_accel.integrate_llg_bdf(
        initial,
        np.zeros((3, 3)),
        constant_field,
        np.ones(1),
        *coefficients,
        0.0,
        final_time,
        1.0e-9,
        1.0e-11,
        1.0e-15,
        1.0e-12,
        1.0e-300,
        5,
        2,
        1000,
    )

    assert diffsol_result.converged is False
    np.testing.assert_allclose(diffsol_result.state, scipy_result.y[:, -1], rtol=2.0e-8)


def test_diffsol_kernel_rejects_a_mismatched_operator() -> None:
    rust_accel = pytest.importorskip("nmag_accel")
    if not hasattr(rust_accel, "integrate_llg_bdf"):
        pytest.skip("installed nmag_accel does not include the Diffsol backend")
    with pytest.raises(ValueError, match="field_operator must have shape"):
        rust_accel.integrate_llg_bdf(
            np.asarray([1.0, 0.0, 0.0]),
            np.zeros((2, 2)),
            np.zeros(3),
            np.ones(1),
            -1.0,
            -0.5,
            0.0,
            0.0,
            1.0,
            1.0e-6,
            1.0e-6,
            1.0e-15,
            1.0e-12,
            1.0,
            5,
            2,
            100,
        )
