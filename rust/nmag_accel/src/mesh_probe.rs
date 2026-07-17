use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::common::{
    average3, average4, bounds3, checked_index, columns3, cross3, dot3, inverse3, point3,
    subtract3, Vec3,
};

pub(crate) type BoundaryFacesResult<'py> =
    PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<i64>>)>;
pub(crate) type ProbeTetrahedralGeometryResult<'py> = PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)>;

const TETRA_FACE_VERTICES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

#[pyfunction]
pub(crate) fn build_oriented_boundary_faces<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    simplices: PyReadonlyArray2<'py, i64>,
) -> BoundaryFacesResult<'py> {
    let points = points.as_array();
    let simplices = simplices.as_array();

    if points.ndim() != 2 || points.shape()[1] != 3 {
        return Err(PyValueError::new_err("points must have shape (n, 3)"));
    }
    if simplices.ndim() != 2 || simplices.shape()[1] != 4 {
        return Err(PyValueError::new_err("simplices must have shape (n, 4)"));
    }

    let point_count = points.shape()[0];
    let cell_count = simplices.shape()[0];
    let mut records: Vec<(usize, [usize; 3], [usize; 3])> = Vec::with_capacity(cell_count * 4);
    let mut counts: HashMap<[usize; 3], usize> = HashMap::with_capacity(cell_count * 4);

    for cell_index in 0..cell_count {
        let simplex = [
            checked_index(simplices[[cell_index, 0]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 1]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 2]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 3]], point_count, "simplex node")?,
        ];
        for local_face in TETRA_FACE_VERTICES {
            let face = [
                simplex[local_face[0]],
                simplex[local_face[1]],
                simplex[local_face[2]],
            ];
            let mut key = face;
            key.sort_unstable();
            *counts.entry(key).or_insert(0) += 1;
            records.push((cell_index, face, key));
        }
    }
    if counts.values().any(|&count| count > 2) {
        return Err(PyValueError::new_err(
            "non-manifold mesh: a face is shared by more than two tetrahedra",
        ));
    }

    let mut owner_values: Vec<i64> = Vec::new();
    let mut face_values: Vec<i64> = Vec::new();
    for (cell_index, mut face, key) in records {
        if counts.get(&key).copied().unwrap_or(0) != 1 {
            continue;
        }
        let simplex = [
            checked_index(simplices[[cell_index, 0]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 1]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 2]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 3]], point_count, "simplex node")?,
        ];
        let face_points = [
            point3(points, face[0]),
            point3(points, face[1]),
            point3(points, face[2]),
        ];
        let cell_center = average4([
            point3(points, simplex[0]),
            point3(points, simplex[1]),
            point3(points, simplex[2]),
            point3(points, simplex[3]),
        ]);
        let face_center = average3(face_points);
        let normal = cross3(
            subtract3(face_points[1], face_points[0]),
            subtract3(face_points[2], face_points[0]),
        );
        if dot3(normal, subtract3(cell_center, face_center)) > 0.0 {
            face.swap(1, 2);
        }
        owner_values.push(cell_index as i64);
        face_values.extend(face.map(|index| index as i64));
    }

    let owner_count = owner_values.len();
    let owners = ndarray::Array1::from_vec(owner_values);
    let faces = Array2::from_shape_vec((owner_count, 3), face_values)
        .map_err(|err| PyValueError::new_err(format!("could not shape boundary faces: {err}")))?;
    Ok((owners.into_pyarray(py), faces.into_pyarray(py)))
}

#[pyfunction]
pub(crate) fn build_probe_tetrahedral_geometry<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    simplices: PyReadonlyArray2<'py, i64>,
) -> ProbeTetrahedralGeometryResult<'py> {
    let points = points.as_array();
    let simplices = simplices.as_array();

    if points.ndim() != 2 || points.shape()[1] != 3 {
        return Err(PyValueError::new_err("points must have shape (n, 3)"));
    }
    if simplices.ndim() != 2 || simplices.shape()[1] != 4 {
        return Err(PyValueError::new_err("simplices must have shape (n, 4)"));
    }

    let mut valid_simplices: Vec<[i64; 4]> = Vec::with_capacity(simplices.shape()[0]);
    let mut origins: Vec<Vec3> = Vec::with_capacity(simplices.shape()[0]);
    let mut inverse_matrices: Vec<[[f64; 3]; 3]> = Vec::with_capacity(simplices.shape()[0]);
    let mut lower_bounds: Vec<Vec3> = Vec::with_capacity(simplices.shape()[0]);
    let mut upper_bounds: Vec<Vec3> = Vec::with_capacity(simplices.shape()[0]);

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
        let tetrahedron = [
            point3(points, simplex[0]),
            point3(points, simplex[1]),
            point3(points, simplex[2]),
            point3(points, simplex[3]),
        ];
        let matrix = columns3(
            subtract3(tetrahedron[1], tetrahedron[0]),
            subtract3(tetrahedron[2], tetrahedron[0]),
            subtract3(tetrahedron[3], tetrahedron[0]),
        );
        let Some(inverse) = inverse3(matrix) else {
            continue;
        };
        valid_simplices.push([
            simplex[0] as i64,
            simplex[1] as i64,
            simplex[2] as i64,
            simplex[3] as i64,
        ]);
        origins.push(tetrahedron[0]);
        inverse_matrices.push(inverse);
        lower_bounds.push(bounds3(tetrahedron, f64::min));
        upper_bounds.push(bounds3(tetrahedron, f64::max));
    }

    let count = valid_simplices.len();
    let mut simplex_array = Array2::<i64>::zeros((count, 4));
    let mut origin_array = Array2::<f64>::zeros((count, 3));
    let mut inverse_array = Array3::<f64>::zeros((count, 3, 3));
    let mut lower_array = Array2::<f64>::zeros((count, 3));
    let mut upper_array = Array2::<f64>::zeros((count, 3));

    for row in 0..count {
        for column in 0..4 {
            simplex_array[[row, column]] = valid_simplices[row][column];
        }
        for column in 0..3 {
            origin_array[[row, column]] = origins[row][column];
            lower_array[[row, column]] = lower_bounds[row][column];
            upper_array[[row, column]] = upper_bounds[row][column];
            for inner_column in 0..3 {
                inverse_array[[row, column, inner_column]] =
                    inverse_matrices[row][column][inner_column];
            }
        }
    }

    Ok((
        simplex_array.into_pyarray(py),
        origin_array.into_pyarray(py),
        inverse_array.into_pyarray(py),
        lower_array.into_pyarray(py),
        upper_array.into_pyarray(py),
    ))
}

#[cfg(test)]
#[path = "../tests/unit/mesh_probe.rs"]
mod tests;
