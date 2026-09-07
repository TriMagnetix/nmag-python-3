use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::f64::consts::PI;

use crate::common::{checked_index, cross3, dot3, should_parallelise};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn llg_rhs<'py>(
    py: Python<'py>,
    m: PyReadonlyArray2<'py, f64>,
    h_total: PyReadonlyArray2<'py, f64>,
    pin: PyReadonlyArray1<'py, f64>,
    ms_values: PyReadonlyArray1<'py, f64>,
    precession_coeff: f64,
    damping_coeff: f64,
    normalisation_coeff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = m.as_array();
    let h_total = h_total.as_array();
    let pin = pin.as_array();
    let ms_values = ms_values.as_array();

    let point_count = validate_llg_inputs(m, h_total, pin, ms_values)?;
    let values = compute_llg_values(m, h_total, pin, ms_values, None, point_count, |_| {
        [
            precession_coeff,
            damping_coeff,
            normalisation_coeff,
            0.0,
            0.0,
        ]
    });
    llg_values_into_python(py, point_count, values)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn llg_rhs_heterogeneous<'py>(
    py: Python<'py>,
    m: PyReadonlyArray2<'py, f64>,
    h_total: PyReadonlyArray2<'py, f64>,
    pin: PyReadonlyArray1<'py, f64>,
    ms_values: PyReadonlyArray1<'py, f64>,
    precession_coeff: PyReadonlyArray1<'py, f64>,
    damping_coeff: PyReadonlyArray1<'py, f64>,
    normalisation_coeff: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = m.as_array();
    let h_total = h_total.as_array();
    let pin = pin.as_array();
    let ms_values = ms_values.as_array();
    let precession_coeff = precession_coeff.as_array();
    let damping_coeff = damping_coeff.as_array();
    let normalisation_coeff = normalisation_coeff.as_array();
    let point_count = validate_llg_inputs(m, h_total, pin, ms_values)?;
    for (name, values) in [
        ("precession_coeff", precession_coeff),
        ("damping_coeff", damping_coeff),
        ("normalisation_coeff", normalisation_coeff),
    ] {
        if values.len() != point_count {
            return Err(PyValueError::new_err(format!(
                "{name} must contain one value per mesh point"
            )));
        }
    }
    let values = compute_llg_values(
        m,
        h_total,
        pin,
        ms_values,
        None,
        point_count,
        |point_index| {
            [
                precession_coeff[point_index],
                damping_coeff[point_index],
                normalisation_coeff[point_index],
                0.0,
                0.0,
            ]
        },
    );
    llg_values_into_python(py, point_count, values)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn llg_rhs_stt_heterogeneous<'py>(
    py: Python<'py>,
    m: PyReadonlyArray2<'py, f64>,
    h_total: PyReadonlyArray2<'py, f64>,
    pin: PyReadonlyArray1<'py, f64>,
    ms_values: PyReadonlyArray1<'py, f64>,
    precession_coeff: PyReadonlyArray1<'py, f64>,
    damping_coeff: PyReadonlyArray1<'py, f64>,
    normalisation_coeff: PyReadonlyArray1<'py, f64>,
    dm_dcurrent: PyReadonlyArray2<'py, f64>,
    stt_adiabatic_coeff: PyReadonlyArray1<'py, f64>,
    stt_nonadiabatic_coeff: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = m.as_array();
    let h_total = h_total.as_array();
    let pin = pin.as_array();
    let ms_values = ms_values.as_array();
    let precession_coeff = precession_coeff.as_array();
    let damping_coeff = damping_coeff.as_array();
    let normalisation_coeff = normalisation_coeff.as_array();
    let dm_dcurrent = dm_dcurrent.as_array();
    let stt_adiabatic_coeff = stt_adiabatic_coeff.as_array();
    let stt_nonadiabatic_coeff = stt_nonadiabatic_coeff.as_array();
    let point_count = validate_llg_inputs(m, h_total, pin, ms_values)?;
    if dm_dcurrent.shape() != m.shape() {
        return Err(PyValueError::new_err(
            "dm_dcurrent must have the same shape as m",
        ));
    }
    for (name, values) in [
        ("precession_coeff", precession_coeff),
        ("damping_coeff", damping_coeff),
        ("normalisation_coeff", normalisation_coeff),
        ("stt_adiabatic_coeff", stt_adiabatic_coeff),
        ("stt_nonadiabatic_coeff", stt_nonadiabatic_coeff),
    ] {
        if values.len() != point_count {
            return Err(PyValueError::new_err(format!(
                "{name} must contain one value per mesh point"
            )));
        }
    }
    let values = compute_llg_values(
        m,
        h_total,
        pin,
        ms_values,
        Some(dm_dcurrent),
        point_count,
        |point_index| {
            [
                precession_coeff[point_index],
                damping_coeff[point_index],
                normalisation_coeff[point_index],
                stt_adiabatic_coeff[point_index],
                stt_nonadiabatic_coeff[point_index],
            ]
        },
    );
    llg_values_into_python(py, point_count, values)
}

fn validate_llg_inputs(
    m: ArrayView2<'_, f64>,
    h_total: ArrayView2<'_, f64>,
    pin: ArrayView1<'_, f64>,
    ms_values: ArrayView1<'_, f64>,
) -> PyResult<usize> {
    if m.ndim() != 2 || m.shape()[1] != 3 {
        return Err(PyValueError::new_err("m must have shape (n, 3)"));
    }
    if h_total.ndim() != 2 || h_total.shape() != m.shape() {
        return Err(PyValueError::new_err(
            "h_total must have the same shape as m",
        ));
    }
    let point_count = m.shape()[0];
    if pin.len() != point_count {
        return Err(PyValueError::new_err(
            "pin must contain one value per mesh point",
        ));
    }
    if ms_values.len() != point_count {
        return Err(PyValueError::new_err(
            "ms_values must contain one value per mesh point",
        ));
    }

    Ok(point_count)
}

fn compute_llg_values<C>(
    m: ArrayView2<'_, f64>,
    h_total: ArrayView2<'_, f64>,
    pin: ArrayView1<'_, f64>,
    ms_values: ArrayView1<'_, f64>,
    dm_dcurrent: Option<ArrayView2<'_, f64>>,
    point_count: usize,
    coefficients: C,
) -> Vec<f64>
where
    C: Fn(usize) -> [f64; 5] + Sync,
{
    let mut values = vec![0.0; point_count * 3];
    let compute_point = |point_index: usize, output: &mut [f64]| {
        let m_value = [
            m[[point_index, 0]],
            m[[point_index, 1]],
            m[[point_index, 2]],
        ];
        let h_value = [
            h_total[[point_index, 0]],
            h_total[[point_index, 1]],
            h_total[[point_index, 2]],
        ];
        let mxh = cross3(m_value, h_value);
        let mdoth = dot3(m_value, h_value);
        let mdotm = dot3(m_value, m_value);
        let norm_error_scale = 1.0 - mdotm;
        let scale = ms_values[point_index] * pin[point_index];
        let directional = dm_dcurrent.map_or([0.0; 3], |values| {
            [
                values[[point_index, 0]],
                values[[point_index, 1]],
                values[[point_index, 2]],
            ]
        });
        let mx_directional = cross3(m_value, directional);
        let mdot_directional = dot3(m_value, directional);
        let [precession_coeff, damping_coeff, normalisation_coeff, stt_adiabatic_coeff, stt_nonadiabatic_coeff] =
            coefficients(point_index);
        for component in 0..3 {
            let damping = m_value[component] * mdoth - h_value[component] * mdotm;
            let norm_error = norm_error_scale * m_value[component];
            let mxmx_directional =
                m_value[component] * mdot_directional - directional[component] * mdotm;
            output[component] = scale
                * (precession_coeff * mxh[component]
                    + damping_coeff * damping
                    + normalisation_coeff * norm_error
                    + stt_adiabatic_coeff * mxmx_directional
                    + stt_nonadiabatic_coeff * mx_directional[component]);
        }
    };
    if should_parallelise(point_count) {
        values
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(point_index, output)| compute_point(point_index, output));
    } else {
        values
            .chunks_mut(3)
            .enumerate()
            .for_each(|(point_index, output)| compute_point(point_index, output));
    }

    values
}

fn llg_values_into_python<'py>(
    py: Python<'py>,
    point_count: usize,
    values: Vec<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let result = Array2::from_shape_vec((point_count, 3), values)
        .map_err(|err| PyValueError::new_err(format!("could not shape LLG RHS: {err}")))?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn maxangle_between_edges(
    m: PyReadonlyArray2<'_, f64>,
    simplices: PyReadonlyArray2<'_, i64>,
) -> PyResult<f64> {
    let m = m.as_array();
    let simplices = simplices.as_array();

    if m.ndim() != 2 || m.shape()[1] != 3 {
        return Err(PyValueError::new_err("m must have shape (n, 3)"));
    }
    if simplices.ndim() != 2 || simplices.shape()[1] < 2 {
        return Err(PyValueError::new_err(
            "simplices must have shape (n, simplex_size>=2)",
        ));
    }

    if simplices.shape()[0] == 0 {
        return Ok(0.0);
    }

    let point_count = m.shape()[0];
    let simplex_size = simplices.shape()[1];
    let mut edges: HashSet<(usize, usize)> =
        HashSet::with_capacity(simplices.shape()[0] * simplex_size * (simplex_size - 1) / 2);
    for simplex_index in 0..simplices.shape()[0] {
        for left_position in 0..simplex_size {
            let left = checked_index(
                simplices[[simplex_index, left_position]],
                point_count,
                "simplex node",
            )?;
            for right_position in (left_position + 1)..simplex_size {
                let right = checked_index(
                    simplices[[simplex_index, right_position]],
                    point_count,
                    "simplex node",
                )?;
                if left <= right {
                    edges.insert((left, right));
                } else {
                    edges.insert((right, left));
                }
            }
        }
    }

    if edges.is_empty() {
        return Ok(0.0);
    }

    let max_angle = edges
        .par_iter()
        .map(|(left, right)| {
            let dot = (m[[*left, 0]] * m[[*right, 0]]
                + m[[*left, 1]] * m[[*right, 1]]
                + m[[*left, 2]] * m[[*right, 2]])
            .clamp(-1.0, 1.0);
            dot.acos()
        })
        .reduce(|| 0.0, f64::max);

    Ok(max_angle * 180.0 / PI)
}

#[cfg(test)]
#[path = "../tests/unit/llg_maxangle.rs"]
mod tests;
