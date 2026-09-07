use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::common::{
    checked_index, checked_indices, cross3, dot3, norm3, normalised3, point3, safe_log_ratio,
    scale3, should_parallelise, subtract3, Vec3,
};

pub(crate) struct FaceGeometry {
    pub(crate) points: [Vec3; 3],
    pub(crate) local_indices: [usize; 3],
    pub(crate) zeta_vector: Vec3,
    pub(crate) eta_vectors: [Vec3; 3],
    pub(crate) edge_lengths: [f64; 3],
    pub(crate) corner_cosines: [f64; 3],
    pub(crate) denominator_factor: f64,
}

#[pyfunction]
pub(crate) fn build_lindholm_bem_matrix<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    simplices: PyReadonlyArray2<'py, i64>,
    face_nodes: PyReadonlyArray2<'py, i64>,
    boundary_nodes: PyReadonlyArray1<'py, i64>,
    local_index_by_point: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let points = points.as_array();
    let simplices = simplices.as_array();
    let face_nodes = face_nodes.as_array();
    let boundary_nodes = boundary_nodes.as_array();
    let local_index_by_point = local_index_by_point.as_array();

    if points.ndim() != 2 || points.shape()[1] != 3 {
        return Err(PyValueError::new_err("points must have shape (n, 3)"));
    }
    if simplices.ndim() != 2 || simplices.shape()[1] != 4 {
        return Err(PyValueError::new_err("simplices must have shape (n, 4)"));
    }
    if face_nodes.ndim() != 2 || face_nodes.shape()[1] != 3 {
        return Err(PyValueError::new_err("face_nodes must have shape (n, 3)"));
    }
    if local_index_by_point.len() != points.shape()[0] {
        return Err(PyValueError::new_err(
            "local_index_by_point must contain one entry per point",
        ));
    }

    let boundary_count = boundary_nodes.len();
    let observer_indices = checked_indices(boundary_nodes, points.shape()[0], "boundary node")?;
    let face_geometries =
        lindholm_face_geometries(points, face_nodes, local_index_by_point, boundary_count)?;
    let solid_angles = boundary_node_solid_angles(points, simplices)?;

    // Rayon rejects zero-sized chunks; after all input validation, an empty boundary
    // naturally has an empty dense matrix.
    if boundary_count == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let mut bem_values = vec![0.0; boundary_count * boundary_count];
    bem_values
        .par_chunks_mut(boundary_count)
        .enumerate()
        .for_each(|(row, row_values)| {
            let observer_index = observer_indices[row];
            let observer = point3(points, observer_index);
            for face in &face_geometries {
                if face.denominator_factor == 0.0 {
                    continue;
                }
                let contributions = lindholm_triangle_contributions_precomputed(observer, face);
                for index in 0..3 {
                    row_values[face.local_indices[index]] += contributions[index];
                }
            }
            row_values[row] += solid_angles[observer_index] / (4.0 * PI) - 1.0;
        });

    let bem = Array2::from_shape_vec((boundary_count, boundary_count), bem_values)
        .map_err(|err| PyValueError::new_err(format!("could not shape BEM matrix: {err}")))?;
    Ok(bem.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn apply_lindholm_bem_matrix_free<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    simplices: PyReadonlyArray2<'py, i64>,
    face_nodes: PyReadonlyArray2<'py, i64>,
    boundary_nodes: PyReadonlyArray1<'py, i64>,
    local_index_by_point: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let points = points.as_array();
    let simplices = simplices.as_array();
    let face_nodes = face_nodes.as_array();
    let boundary_nodes = boundary_nodes.as_array();
    let local_index_by_point = local_index_by_point.as_array();
    let values = values.as_array();

    if points.ndim() != 2 || points.shape()[1] != 3 {
        return Err(PyValueError::new_err("points must have shape (n, 3)"));
    }
    if simplices.ndim() != 2 || simplices.shape()[1] != 4 {
        return Err(PyValueError::new_err("simplices must have shape (n, 4)"));
    }
    if face_nodes.ndim() != 2 || face_nodes.shape()[1] != 3 {
        return Err(PyValueError::new_err("face_nodes must have shape (n, 3)"));
    }
    if local_index_by_point.len() != points.shape()[0] {
        return Err(PyValueError::new_err(
            "local_index_by_point must contain one entry per point",
        ));
    }

    let boundary_count = boundary_nodes.len();
    if values.len() != boundary_count {
        return Err(PyValueError::new_err(format!(
            "values must contain one entry per boundary node ({boundary_count})"
        )));
    }
    let observer_indices = checked_indices(boundary_nodes, points.shape()[0], "boundary node")?;
    let face_geometries =
        lindholm_face_geometries(points, face_nodes, local_index_by_point, boundary_count)?;
    let solid_angles = boundary_node_solid_angles(points, simplices)?;
    let mut result = vec![0.0; boundary_count];

    let compute_row = |row: usize, output: &mut f64| {
        let observer_index = observer_indices[row];
        let observer = point3(points, observer_index);
        let mut value = 0.0;
        for face in &face_geometries {
            if face.denominator_factor == 0.0 {
                continue;
            }
            let contributions = lindholm_triangle_contributions_precomputed(observer, face);
            for index in 0..3 {
                value += contributions[index] * values[face.local_indices[index]];
            }
        }
        value += (solid_angles[observer_index] / (4.0 * PI) - 1.0) * values[row];
        *output = value;
    };
    let interaction_count = boundary_count.saturating_mul(face_geometries.len());
    if should_parallelise(interaction_count) {
        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, output)| compute_row(row, output));
    } else {
        result
            .iter_mut()
            .enumerate()
            .for_each(|(row, output)| compute_row(row, output));
    }

    Ok(Array1::from_vec(result).into_pyarray(py))
}

pub(crate) fn lindholm_face_geometries(
    points: ArrayView2<'_, f64>,
    face_nodes: ArrayView2<'_, i64>,
    local_index_by_point: ArrayView1<'_, i64>,
    boundary_count: usize,
) -> PyResult<Vec<FaceGeometry>> {
    let mut geometries = Vec::with_capacity(face_nodes.shape()[0]);
    for face_index in 0..face_nodes.shape()[0] {
        let point_indices = [
            checked_index(face_nodes[[face_index, 0]], points.shape()[0], "face node")?,
            checked_index(face_nodes[[face_index, 1]], points.shape()[0], "face node")?,
            checked_index(face_nodes[[face_index, 2]], points.shape()[0], "face node")?,
        ];
        let mut local_indices = [0usize; 3];
        for index in 0..3 {
            local_indices[index] = checked_boundary_local_index(
                local_index_by_point[point_indices[index]],
                boundary_count,
            )?;
        }

        let face_points = [
            point3(points, point_indices[0]),
            point3(points, point_indices[1]),
            point3(points, point_indices[2]),
        ];
        let edge_vectors = [
            subtract3(face_points[2], face_points[1]),
            subtract3(face_points[0], face_points[2]),
            subtract3(face_points[1], face_points[0]),
        ];
        let xis = [
            normalised3(edge_vectors[0]),
            normalised3(edge_vectors[1]),
            normalised3(edge_vectors[2]),
        ];
        let area_vector = cross3(
            subtract3(face_points[1], face_points[0]),
            subtract3(face_points[2], face_points[0]),
        );
        let double_area = norm3(area_vector);
        let zeta_vector = if double_area == 0.0 {
            [0.0, 0.0, 0.0]
        } else {
            scale3(area_vector, 1.0 / double_area)
        };
        geometries.push(FaceGeometry {
            points: face_points,
            local_indices,
            zeta_vector,
            eta_vectors: [
                cross3(zeta_vector, xis[0]),
                cross3(zeta_vector, xis[1]),
                cross3(zeta_vector, xis[2]),
            ],
            edge_lengths: [
                norm3(edge_vectors[0]),
                norm3(edge_vectors[1]),
                norm3(edge_vectors[2]),
            ],
            corner_cosines: [
                dot3(xis[0], xis[1]),
                dot3(xis[1], xis[2]),
                dot3(xis[2], xis[0]),
            ],
            denominator_factor: if double_area == 0.0 {
                0.0
            } else {
                1.0 / (4.0 * PI * double_area)
            },
        });
    }
    Ok(geometries)
}

fn checked_boundary_local_index(value: i64, boundary_count: usize) -> PyResult<usize> {
    if value < 0 {
        return Err(PyValueError::new_err("face node is not a boundary node"));
    }
    let local_index = usize::try_from(value)
        .map_err(|_| PyValueError::new_err("face node has an invalid boundary-node index"))?;
    if local_index >= boundary_count {
        return Err(PyValueError::new_err(format!(
            "face node boundary index {local_index} is out of bounds for {boundary_count} nodes"
        )));
    }
    Ok(local_index)
}

pub(crate) fn boundary_node_solid_angles(
    points: ArrayView2<'_, f64>,
    simplices: ArrayView2<'_, i64>,
) -> PyResult<Vec<f64>> {
    let mut angles = vec![0.0; points.shape()[0]];
    for simplex_index in 0..simplices.shape()[0] {
        let simplex = [
            checked_index(
                simplices[[simplex_index, 0]],
                points.shape()[0],
                "simplex node",
            )?,
            checked_index(
                simplices[[simplex_index, 1]],
                points.shape()[0],
                "simplex node",
            )?,
            checked_index(
                simplices[[simplex_index, 2]],
                points.shape()[0],
                "simplex node",
            )?,
            checked_index(
                simplices[[simplex_index, 3]],
                points.shape()[0],
                "simplex node",
            )?,
        ];
        for local_index in 0..4 {
            let observer_index = simplex[local_index];
            let mut other = [0usize; 3];
            let mut cursor = 0;
            for candidate in simplex {
                if candidate != observer_index {
                    other[cursor] = candidate;
                    cursor += 1;
                }
            }
            if cursor != 3 {
                return Err(PyValueError::new_err(
                    "simplex contains duplicate node indices",
                ));
            }
            angles[observer_index] += triangle_space_angle(
                point3(points, observer_index),
                point3(points, other[0]),
                point3(points, other[1]),
                point3(points, other[2]),
            )
            .abs();
        }
    }
    Ok(angles)
}

pub(crate) fn lindholm_triangle_contributions_precomputed(
    observer: Vec3,
    face: &FaceGeometry,
) -> Vec3 {
    let p0 = face.points[0];
    let p1 = face.points[1];
    let p2 = face.points[2];
    let r0 = subtract3(p0, observer);
    let r1 = subtract3(p1, observer);
    let r2 = subtract3(p2, observer);
    if face.denominator_factor == 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let r0_len = norm3(r0);
    let r1_len = norm3(r1);
    let r2_len = norm3(r2);
    let dot01 = dot3(r0, r1);
    let dot12 = dot3(r1, r2);
    let dot20 = dot3(r2, r0);
    let angle = triangle_space_angle_from_offsets(r0_len, r1_len, r2_len, dot01, dot12, dot20);
    let s0_len = face.edge_lengths[0];
    let s1_len = face.edge_lengths[1];
    let s2_len = face.edge_lengths[2];
    let c01 = face.corner_cosines[0];
    let c12 = face.corner_cosines[1];
    let c20 = face.corner_cosines[2];
    let zeta = dot3(face.zeta_vector, r0);
    let log0 = safe_log_ratio(r1_len + r2_len + s0_len, r1_len + r2_len - s0_len);
    let log1 = safe_log_ratio(r2_len + r0_len + s1_len, r2_len + r0_len - s1_len);
    let log2 = safe_log_ratio(r0_len + r1_len + s2_len, r0_len + r1_len - s2_len);
    let sign_zeta = if zeta < 0.0 {
        -1.0
    } else if zeta > 0.0 {
        1.0
    } else {
        0.0
    };
    let angle_pm = angle * sign_zeta;
    let gamma0_log = log0 + c01 * log1 + c20 * log2;
    let gamma1_log = c01 * log0 + log1 + c12 * log2;
    let gamma2_log = c20 * log0 + c12 * log1 + log2;
    let mut values = [
        s0_len
            * face.denominator_factor
            * (angle_pm * dot3(face.eta_vectors[0], r1) - zeta * gamma0_log),
        s1_len
            * face.denominator_factor
            * (angle_pm * dot3(face.eta_vectors[1], r2) - zeta * gamma1_log),
        s2_len
            * face.denominator_factor
            * (angle_pm * dot3(face.eta_vectors[2], r0) - zeta * gamma2_log),
    ];
    for value in &mut values {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    values
}

fn triangle_space_angle(observer: Vec3, p0: Vec3, p1: Vec3, p2: Vec3) -> f64 {
    let r0 = subtract3(p0, observer);
    let r1 = subtract3(p1, observer);
    let r2 = subtract3(p2, observer);
    let r0_len = norm3(r0);
    let r1_len = norm3(r1);
    let r2_len = norm3(r2);
    let dot01 = dot3(r0, r1);
    let dot12 = dot3(r1, r2);
    let dot20 = dot3(r2, r0);
    triangle_space_angle_from_offsets(r0_len, r1_len, r2_len, dot01, dot12, dot20)
}

fn triangle_space_angle_from_offsets(
    r0_len: f64,
    r1_len: f64,
    r2_len: f64,
    dot01: f64,
    dot12: f64,
    dot20: f64,
) -> f64 {
    let numerator = r0_len * r1_len * r2_len + r0_len * dot12 + r1_len * dot20 + r2_len * dot01;
    let denominator_squared =
        2.0 * (r1_len * r2_len + dot12) * (r2_len * r0_len + dot20) * (r0_len * r1_len + dot01);
    if denominator_squared <= 0.0 {
        return 0.0;
    }
    let quotient = (numerator / denominator_squared.sqrt()).clamp(-1.0, 1.0);
    2.0 * quotient.acos()
}

#[cfg(test)]
#[path = "../tests/unit/lindholm.rs"]
mod tests;
