from __future__ import annotations

import math
import os
import time
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..backends import _load_rust_accelerator
from ..dynamics import IntegratorStats
from ..resources import available_memory_bytes as _available_memory_bytes
from .support import _si_unit, _simulation_compatibility_binding

FloatArray = NDArray[np.float64]
DIFFSOL_DENSE_MEMORY_LIMIT_ENV = "NMAG_DIFFSOL_DENSE_MEMORY_LIMIT_GIB"
DIFFSOL_DENSE_MATRIX_COPIES_ESTIMATE = 6
DIFFSOL_AVAILABLE_MEMORY_FRACTION = 0.8
DEFAULT_BDF_INITIAL_STEP_SECONDS = 1.0e-15
DEFAULT_BDF_MAXIMUM_STEPS = 1_000_000


class _BdfResult(Protocol):
    state: list[float]
    reached_time: float
    accepted_steps: int
    rejected_steps: int
    rhs_evaluations: int
    jacobian_vector_evaluations: int
    nonlinear_iterations: int
    nonlinear_failures: int
    linear_solver_setups: int
    last_step: float
    max_dm_dt: float
    converged: bool


def _estimated_dense_bdf_peak_bytes(state_size: int) -> int:
    if state_size < 1:
        raise ValueError("state_size must be positive.")
    matrix_bytes = state_size * state_size * np.dtype(np.float64).itemsize
    return DIFFSOL_DENSE_MATRIX_COPIES_ESTIMATE * matrix_bytes


def _diffsol_dense_memory_budget() -> tuple[int | None, str]:
    configured = os.environ.get(DIFFSOL_DENSE_MEMORY_LIMIT_ENV)
    if configured is not None:
        value = configured.strip().lower()
        if value == "unlimited":
            return None, "explicit unlimited override"
        try:
            gibibytes = float(value)
        except ValueError as error:
            raise ValueError(
                f"{DIFFSOL_DENSE_MEMORY_LIMIT_ENV} must be a positive number of GiB "
                f"or 'unlimited', got {configured!r}."
            ) from error
        if not math.isfinite(gibibytes) or gibibytes <= 0.0:
            raise ValueError(
                f"{DIFFSOL_DENSE_MEMORY_LIMIT_ENV} must be a positive number of GiB "
                f"or 'unlimited', got {configured!r}."
            )
        return int(gibibytes * 1024**3), "explicit environment budget"

    available = _available_memory_bytes()
    if available is None:
        return None, "available memory could not be determined"
    return int(available * DIFFSOL_AVAILABLE_MEMORY_FRACTION), "80% of available memory"


def _validate_diffsol_dense_memory(state_size: int) -> None:
    estimated_bytes = _estimated_dense_bdf_peak_bytes(state_size)
    budget_bytes, budget_source = _diffsol_dense_memory_budget()
    if budget_bytes is None or estimated_bytes <= budget_bytes:
        return

    estimated_gib = estimated_bytes / 1024**3
    budget_gib = budget_bytes / 1024**3
    raise MemoryError(
        f"The dense Diffsol solve for {state_size} state values is estimated to require "
        f"approximately {estimated_gib:.2f} GiB, exceeding the {budget_gib:.2f} GiB "
        f"budget ({budget_source}). Set {DIFFSOL_DENSE_MEMORY_LIMIT_ENV} to a reviewed "
        "GiB budget or 'unlimited' to proceed. There is no fixed state-count limit."
    )


def build_affine_field_operator(simulation: Any) -> FloatArray:
    """Build the dense linear map from nodal m to demag plus exchange field."""
    mesh_token = simulation._mesh_geometry_token()
    cached = simulation._llg_affine_operator_cache
    if cached is not None and cached[0] == mesh_token:
        return cached[1]

    original_m = np.asarray(simulation._fields["m"], dtype=np.float64)
    state_size = int(original_m.size)
    _validate_diffsol_dense_memory(state_size)

    def linear_field(state: FloatArray) -> FloatArray:
        simulation._fields["m"] = np.reshape(state, (-1, 3))
        simulation._invalidate_demag()
        field = np.asarray(simulation._get_exchange_nodal_field(), dtype=np.float64)
        if simulation.do_demag:
            field = field + np.asarray(simulation._get_demag_nodal_field(), dtype=np.float64)
        return np.ravel(field)

    try:
        zero = np.zeros(state_size, dtype=np.float64)
        baseline = linear_field(zero)
        operator = np.empty((state_size, state_size), dtype=np.float64)
        basis = np.zeros(state_size, dtype=np.float64)
        for column in range(state_size):
            basis[column] = 1.0
            operator[:, column] = linear_field(basis) - baseline
            basis[column] = 0.0
    finally:
        simulation._fields["m"] = np.array(original_m, copy=True)
        simulation._invalidate_demag()

    operator = np.ascontiguousarray(operator)
    simulation._llg_affine_operator_cache = (mesh_token, operator)
    return operator


def relax_with_diffsol(simulation: Any, H_applied: Any = None) -> None:
    if simulation.mesh is None or "m" not in simulation._fields:
        raise RuntimeError("A mesh and magnetisation are required before relaxation.")
    if not simulation._anisotropy_is_zero_by_construction():
        raise NotImplementedError(
            "The experimental Diffsol backend does not support dynamic anisotropy."
        )

    simulation.do_next_stage(stage=simulation.clock.stage)
    simulation.clock.stage_end = False
    if H_applied is not None:
        simulation.set_H_ext(H_applied)

    started = time.perf_counter()
    try:
        operator = build_affine_field_operator(simulation)
        point_count = len(simulation._fields["m"])
        h_ext = np.asarray(simulation._fields.get("H_ext", np.zeros(3)), dtype=np.float64)
        constant_field = np.ascontiguousarray(np.tile(h_ext, point_count))
        pin = np.ascontiguousarray(simulation._subfield_array("pin"), dtype=np.float64)
        c1, c2, c3 = simulation._llg_coefficients()
        config = simulation.integrator_config
        rust_accel = _simulation_compatibility_binding(
            "_load_rust_accelerator", _load_rust_accelerator
        )("NmagConfig(integrator_backend='diffsol')")
        try:
            integrate_llg_bdf = rust_accel.integrate_llg_bdf
        except AttributeError as exc:
            raise RuntimeError(
                "NmagConfig(integrator_backend='diffsol') requires an nmag_accel build with the "
                "Diffsol backend. Rebuild it with `./scripts/build-accelerator.sh --release`."
            ) from exc

        result: _BdfResult = integrate_llg_bdf(
            np.ascontiguousarray(np.ravel(simulation._fields["m"]), dtype=np.float64),
            operator,
            constant_field,
            pin,
            c1,
            c2,
            c3,
            0.0,
            simulation.max_time_reached.in_units_of(_si_unit("s")),
            config.relative_tolerance,
            config.absolute_tolerance,
            min(DEFAULT_BDF_INITIAL_STEP_SECONDS, config.maximum_step_seconds),
            config.maximum_step_seconds,
            simulation.stopping_dm_dt,
            5,
            simulation.convergence.required_checks,
            DEFAULT_BDF_MAXIMUM_STEPS,
        )
    except Exception:
        simulation._last_integrator_stats = IntegratorStats(
            failed=True,
            status="failed",
            wall_seconds=time.perf_counter() - started,
        )
        raise
    wall_seconds = time.perf_counter() - started
    accepted_state = np.asarray(result.state, dtype=np.float64)
    if not np.all(np.isfinite(accepted_state)):
        raise FloatingPointError("Diffsol returned a non-finite magnetisation state.")

    simulation._fields["m"] = np.reshape(accepted_state, (-1, 3))
    simulation._invalidate_demag()
    simulation._invalidate_integrator()
    simulation.max_dm_dt = float(result.max_dm_dt)
    simulation.clock.record_advance(
        stage_time_seconds=float(result.reached_time),
        accepted_steps=int(result.accepted_steps),
        last_step_seconds=float(result.last_step),
        wall_seconds=wall_seconds,
    )
    simulation.clock.convergence = bool(result.converged)
    simulation.clock.stage_end = bool(result.converged)
    simulation._last_integrator_stats = IntegratorStats(
        accepted_steps=int(result.accepted_steps),
        rhs_evaluations=int(result.rhs_evaluations),
        rejected_steps=int(result.rejected_steps),
        failed=False,
        status="converged" if result.converged else "maximum_time_reached",
        last_step_seconds=float(result.last_step),
        simulated_seconds=float(result.reached_time),
        wall_seconds=wall_seconds,
        jacobian_vector_evaluations=int(result.jacobian_vector_evaluations),
        nonlinear_iterations=int(result.nonlinear_iterations),
        nonlinear_failures=int(result.nonlinear_failures),
        linear_solver_setups=int(result.linear_solver_setups),
    )
    if not result.converged:
        raise RuntimeError(
            f"Diffsol reached {simulation.max_time_reached} without satisfying convergence."
        )

    simulation.save_data()
    simulation.save_data(fields="all")
