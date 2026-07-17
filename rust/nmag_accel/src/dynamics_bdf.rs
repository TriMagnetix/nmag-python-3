use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use diffsol::{NalgebraLU, NalgebraMat, OdeBuilder, OdeSolverMethod, Vector, VectorHost};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(get_all)]
pub(crate) struct BdfIntegrationResult {
    state: Vec<f64>,
    reached_time: f64,
    accepted_steps: usize,
    rejected_steps: usize,
    rhs_evaluations: usize,
    jacobian_vector_evaluations: usize,
    nonlinear_iterations: usize,
    nonlinear_failures: usize,
    linear_solver_setups: usize,
    last_step: f64,
    max_dm_dt: f64,
    converged: bool,
}

#[derive(Clone, Copy)]
struct LlgCoefficients {
    precession: f64,
    damping: f64,
    normalisation: f64,
}

struct AffineLlg {
    operator: Vec<f64>,
    constant_field: Vec<f64>,
    pin: Vec<f64>,
    coefficients: LlgCoefficients,
    state_size: usize,
    time_scale_seconds: f64,
}

impl AffineLlg {
    fn field(&self, state: &[f64], output: &mut [f64]) {
        for (row, value) in output.iter_mut().enumerate() {
            let offset = row * self.state_size;
            *value = self.constant_field[row]
                + self.operator[offset..offset + self.state_size]
                    .iter()
                    .zip(state)
                    .map(|(coefficient, state_value)| coefficient * state_value)
                    .sum::<f64>();
        }
    }

    fn rhs(&self, state: &[f64], output: &mut [f64]) {
        let mut field = vec![0.0; self.state_size];
        self.field(state, &mut field);
        for point in 0..self.pin.len() {
            let offset = point * 3;
            let m = &state[offset..offset + 3];
            let h = &field[offset..offset + 3];
            let mxh = [
                m[1] * h[2] - m[2] * h[1],
                m[2] * h[0] - m[0] * h[2],
                m[0] * h[1] - m[1] * h[0],
            ];
            let mdoth = m
                .iter()
                .zip(h)
                .map(|(left, right)| left * right)
                .sum::<f64>();
            let mdotm = m.iter().map(|value| value * value).sum::<f64>();
            for component in 0..3 {
                let damping = m[component] * mdoth - h[component] * mdotm;
                let normalisation = (1.0 - mdotm) * m[component];
                output[offset + component] = self.pin[point]
                    * self.time_scale_seconds
                    * (self.coefficients.precession * mxh[component]
                        + self.coefficients.damping * damping
                        + self.coefficients.normalisation * normalisation);
            }
        }
    }

    fn jacobian_vector(&self, state: &[f64], vector: &[f64], output: &mut [f64]) {
        let mut field = vec![0.0; self.state_size];
        let mut field_direction = vec![0.0; self.state_size];
        self.field(state, &mut field);
        self.field(vector, &mut field_direction);
        for (value, constant) in field_direction.iter_mut().zip(&self.constant_field) {
            *value -= constant;
        }

        for point in 0..self.pin.len() {
            let offset = point * 3;
            let m = &state[offset..offset + 3];
            let v = &vector[offset..offset + 3];
            let h = &field[offset..offset + 3];
            let dh = &field_direction[offset..offset + 3];
            let cross_direction = [
                v[1] * h[2] - v[2] * h[1] + m[1] * dh[2] - m[2] * dh[1],
                v[2] * h[0] - v[0] * h[2] + m[2] * dh[0] - m[0] * dh[2],
                v[0] * h[1] - v[1] * h[0] + m[0] * dh[1] - m[1] * dh[0],
            ];
            let mdoth = dot(m, h);
            let v_doth = dot(v, h);
            let m_dot_dh = dot(m, dh);
            let mdotm = dot(m, m);
            let mdotv = dot(m, v);
            for component in 0..3 {
                let damping_direction = v[component] * mdoth + m[component] * (v_doth + m_dot_dh)
                    - dh[component] * mdotm
                    - h[component] * (2.0 * mdotv);
                let normalisation_direction =
                    -2.0 * mdotv * m[component] + (1.0 - mdotm) * v[component];
                output[offset + component] = self.pin[point]
                    * self.time_scale_seconds
                    * (self.coefficients.precession * cross_direction[component]
                        + self.coefficients.damping * damping_direction
                        + self.coefficients.normalisation * normalisation_direction);
            }
        }
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn validate_finite(values: &[f64], name: &str) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate_llg_bdf(
    initial_state: PyReadonlyArray1<'_, f64>,
    field_operator: PyReadonlyArray2<'_, f64>,
    constant_field: PyReadonlyArray1<'_, f64>,
    pin: PyReadonlyArray1<'_, f64>,
    precession_coeff: f64,
    damping_coeff: f64,
    normalisation_coeff: f64,
    initial_time: f64,
    maximum_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    initial_step: f64,
    maximum_step: f64,
    convergence_threshold: f64,
    convergence_check_steps: usize,
    required_successes: usize,
    maximum_steps: usize,
) -> PyResult<BdfIntegrationResult> {
    let initial_state = initial_state.as_slice()?.to_vec();
    let operator_shape = field_operator.shape();
    let operator = field_operator.as_slice()?.to_vec();
    let constant_field = constant_field.as_slice()?.to_vec();
    let pin = pin.as_slice()?.to_vec();
    let state_size = initial_state.len();

    if state_size == 0 || !state_size.is_multiple_of(3) {
        return Err(PyValueError::new_err(
            "initial_state length must be a positive multiple of three",
        ));
    }
    if operator_shape != [state_size, state_size] {
        return Err(PyValueError::new_err(format!(
            "field_operator must have shape ({state_size}, {state_size})"
        )));
    }
    if constant_field.len() != state_size {
        return Err(PyValueError::new_err(format!(
            "constant_field must contain {state_size} values"
        )));
    }
    if pin.len() * 3 != state_size {
        return Err(PyValueError::new_err(
            "pin must contain one value per mesh point",
        ));
    }
    for (value, name) in [
        (relative_tolerance, "relative_tolerance"),
        (absolute_tolerance, "absolute_tolerance"),
        (initial_step, "initial_step"),
        (maximum_step, "maximum_step"),
        (convergence_threshold, "convergence_threshold"),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "{name} must be positive and finite"
            )));
        }
    }
    if !initial_time.is_finite() || !maximum_time.is_finite() || maximum_time <= initial_time {
        return Err(PyValueError::new_err(
            "maximum_time must be finite and greater than initial_time",
        ));
    }
    if convergence_check_steps == 0 || required_successes == 0 || maximum_steps == 0 {
        return Err(PyValueError::new_err(
            "step and convergence counts must be positive",
        ));
    }
    validate_finite(&initial_state, "initial_state")?;
    validate_finite(&operator, "field_operator")?;
    validate_finite(&constant_field, "constant_field")?;
    validate_finite(&pin, "pin")?;
    validate_finite(
        &[precession_coeff, damping_coeff, normalisation_coeff],
        "LLG coefficients",
    )?;

    let system = Arc::new(AffineLlg {
        operator,
        constant_field,
        pin,
        coefficients: LlgCoefficients {
            precession: precession_coeff,
            damping: damping_coeff,
            normalisation: normalisation_coeff,
        },
        state_size,
        time_scale_seconds: 1.0e-12,
    });
    let rhs_system = Arc::clone(&system);
    let jacobian_system = Arc::clone(&system);
    let rhs_evaluations = Arc::new(AtomicUsize::new(0));
    let jacobian_evaluations = Arc::new(AtomicUsize::new(0));
    let rhs_counter = Arc::clone(&rhs_evaluations);
    let jacobian_counter = Arc::clone(&jacobian_evaluations);
    let initial_state_for_builder = initial_state.clone();

    let time_scale_seconds = system.time_scale_seconds;
    let scaled_maximum_time = (maximum_time - initial_time) / time_scale_seconds;
    let problem = OdeBuilder::<NalgebraMat<f64>>::new()
        .t0(0.0)
        .h0(initial_step / time_scale_seconds)
        .rtol(relative_tolerance)
        .atol([absolute_tolerance])
        .rhs_implicit(
            move |state, _parameters, _time, output| {
                rhs_counter.fetch_add(1, Ordering::Relaxed);
                rhs_system.rhs(state.as_slice(), output.as_mut_slice());
            },
            move |state, _parameters, _time, vector, output| {
                jacobian_counter.fetch_add(1, Ordering::Relaxed);
                jacobian_system.jacobian_vector(
                    state.as_slice(),
                    vector.as_slice(),
                    output.as_mut_slice(),
                );
            },
        )
        .init(
            move |_parameters, _time, output| {
                output
                    .as_mut_slice()
                    .copy_from_slice(&initial_state_for_builder);
            },
            state_size,
        )
        .build()
        .map_err(|error| {
            PyRuntimeError::new_err(format!("could not build BDF problem: {error}"))
        })?;
    let mut solver = problem
        .bdf::<NalgebraLU<f64>>()
        .map_err(|error| PyRuntimeError::new_err(format!("could not initialise BDF: {error}")))?;
    let mut check_state = initial_state;
    let mut check_time = 0.0;
    let mut consecutive_successes = 0;
    let mut max_dm_dt = f64::INFINITY;
    let mut last_step = 0.0;
    let mut converged = false;
    let scaled_maximum_step = maximum_step / time_scale_seconds;

    while solver.get_statistics().number_of_steps < maximum_steps {
        let previous_time = solver.state().t;
        if previous_time >= scaled_maximum_time {
            break;
        }
        solver
            .set_stop_time((previous_time + scaled_maximum_step).min(scaled_maximum_time))
            .map_err(|error| {
                PyRuntimeError::new_err(format!("could not set BDF step ceiling: {error}"))
            })?;
        solver
            .step()
            .map_err(|error| PyRuntimeError::new_err(format!("BDF integration failed: {error}")))?;
        let state = solver.state();
        last_step = state.t - previous_time;
        if state.y.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(PyRuntimeError::new_err("BDF accepted a non-finite state"));
        }

        let accepted_steps = solver.get_statistics().number_of_steps;
        if accepted_steps.is_multiple_of(convergence_check_steps) {
            let elapsed = state.t - check_time;
            max_dm_dt = state
                .y
                .as_slice()
                .chunks_exact(3)
                .zip(check_state.chunks_exact(3))
                .map(|(current, previous)| {
                    current
                        .iter()
                        .zip(previous)
                        .map(|(left, right)| (left - right).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / (elapsed * time_scale_seconds)
                })
                .fold(0.0, f64::max);
            if max_dm_dt < convergence_threshold {
                consecutive_successes += 1;
                if consecutive_successes >= required_successes {
                    converged = true;
                    break;
                }
            } else {
                consecutive_successes = 0;
            }
            check_state = state.y.clone_as_vec();
            check_time = state.t;
        }

        if state.t >= scaled_maximum_time {
            break;
        }
    }

    let state = solver.state();
    let statistics = solver.get_statistics();
    if statistics.number_of_steps >= maximum_steps && !converged && state.t < scaled_maximum_time {
        return Err(PyRuntimeError::new_err(format!(
            "BDF reached the maximum of {maximum_steps} accepted steps"
        )));
    }
    Ok(BdfIntegrationResult {
        state: state.y.clone_as_vec(),
        reached_time: initial_time + state.t * time_scale_seconds,
        accepted_steps: statistics.number_of_steps,
        rejected_steps: statistics.number_of_error_test_failures,
        rhs_evaluations: rhs_evaluations.load(Ordering::Relaxed),
        jacobian_vector_evaluations: jacobian_evaluations.load(Ordering::Relaxed),
        nonlinear_iterations: statistics.number_of_nonlinear_solver_iterations,
        nonlinear_failures: statistics.number_of_nonlinear_solver_fails,
        linear_solver_setups: statistics.number_of_linear_solver_setups,
        last_step: last_step * time_scale_seconds,
        max_dm_dt,
        converged,
    })
}

#[cfg(test)]
#[path = "../tests/unit/dynamics_bdf.rs"]
mod tests;
