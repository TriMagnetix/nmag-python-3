use ndarray::{Array1, Array2, Array3, ArrayView2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::common::{
    checked_index, columns3, determinant3, dot3, inverse3, point3, should_parallelise, subtract3,
    Vec3,
};

pub(crate) type FemGeometryResult<'py> = PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray1<f64>>,
)>;

pub(crate) struct FemCellGeometry {
    simplex: [usize; 4],
    gradients: [Vec3; 4],
    volume: f64,
}

#[pyfunction]
pub(crate) fn build_demag_fem_geometry<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    simplices: PyReadonlyArray2<'py, i64>,
) -> FemGeometryResult<'py> {
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
    let mut stiffness = Array2::<f64>::zeros((point_count, point_count));
    let mut gradients_by_cell = Array3::<f64>::zeros((cell_count, 4, 3));
    let mut volumes = vec![0.0; cell_count];

    let mut simplex_indices: Vec<[usize; 4]> = Vec::with_capacity(cell_count);
    for cell_index in 0..cell_count {
        simplex_indices.push([
            checked_index(simplices[[cell_index, 0]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 1]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 2]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 3]], point_count, "simplex node")?,
        ]);
    }

    let cell_geometries: Vec<Option<FemCellGeometry>> = simplex_indices
        .par_iter()
        .map(|simplex| fem_cell_geometry(points, *simplex))
        .collect();

    for (cell_index, cell_geometry) in cell_geometries.into_iter().enumerate() {
        let Some(cell_geometry) = cell_geometry else {
            continue;
        };
        volumes[cell_index] = cell_geometry.volume;
        for local_index in 0..4 {
            for component in 0..3 {
                gradients_by_cell[[cell_index, local_index, component]] =
                    cell_geometry.gradients[local_index][component];
            }
        }
        for local_row in 0..4 {
            let global_row = cell_geometry.simplex[local_row];
            for local_column in 0..4 {
                let global_column = cell_geometry.simplex[local_column];
                stiffness[[global_row, global_column]] += cell_geometry.volume
                    * dot3(
                        cell_geometry.gradients[local_row],
                        cell_geometry.gradients[local_column],
                    );
            }
        }
    }

    Ok((
        stiffness.into_pyarray(py),
        gradients_by_cell.into_pyarray(py),
        Array1::from_vec(volumes).into_pyarray(py),
    ))
}

#[pyfunction]
pub(crate) fn build_demag_fem_divergence<'py>(
    py: Python<'py>,
    simplices: PyReadonlyArray2<'py, i64>,
    gradients_by_cell: PyReadonlyArray3<'py, f64>,
    volumes: PyReadonlyArray1<'py, f64>,
    m: PyReadonlyArray2<'py, f64>,
    ms_values: PyReadonlyArray1<'py, f64>,
    point_count: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let simplices = simplices.as_array();
    let gradients_by_cell = gradients_by_cell.as_array();
    let volumes = volumes.as_array();
    let m = m.as_array();
    let ms_values = ms_values.as_array();

    if simplices.ndim() != 2 || simplices.shape()[1] != 4 {
        return Err(PyValueError::new_err("simplices must have shape (n, 4)"));
    }
    if gradients_by_cell.ndim() != 3
        || gradients_by_cell.shape()[0] != simplices.shape()[0]
        || gradients_by_cell.shape()[1] != 4
        || gradients_by_cell.shape()[2] != 3
    {
        return Err(PyValueError::new_err(
            "gradients_by_cell must have shape (n, 4, 3)",
        ));
    }
    if volumes.len() != simplices.shape()[0] {
        return Err(PyValueError::new_err(
            "volumes must contain one value per simplex",
        ));
    }
    if ms_values.len() != simplices.shape()[0] {
        return Err(PyValueError::new_err(
            "ms_values must contain one value per simplex",
        ));
    }
    if m.ndim() != 2 || m.shape()[0] != point_count || m.shape()[1] != 3 {
        return Err(PyValueError::new_err("m must have shape (point_count, 3)"));
    }

    let cell_count = simplices.shape()[0];
    let mut simplex_indices: Vec<[usize; 4]> = Vec::with_capacity(cell_count);
    for cell_index in 0..cell_count {
        simplex_indices.push([
            checked_index(simplices[[cell_index, 0]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 1]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 2]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 3]], point_count, "simplex node")?,
        ]);
    }

    let cell_contribution = |cell_index: usize| {
        let simplex = simplex_indices[cell_index];
        let ms = ms_values[cell_index];
        let mut average_m = [0.0; 3];
        for point_index in simplex {
            for component in 0..3 {
                average_m[component] += m[[point_index, component]] * ms;
            }
        }
        for component in &mut average_m {
            *component *= 0.25;
        }

        let mut contributions = [0.0; 4];
        for local_index in 0..4 {
            let gradient = [
                gradients_by_cell[[cell_index, local_index, 0]],
                gradients_by_cell[[cell_index, local_index, 1]],
                gradients_by_cell[[cell_index, local_index, 2]],
            ];
            contributions[local_index] = volumes[cell_index] * dot3(gradient, average_m);
        }
        (simplex, contributions)
    };
    let cell_contributions: Vec<([usize; 4], [f64; 4])> = if should_parallelise(cell_count) {
        (0..cell_count)
            .into_par_iter()
            .map(cell_contribution)
            .collect()
    } else {
        (0..cell_count).map(cell_contribution).collect()
    };

    let mut divergence = vec![0.0; point_count];
    for (simplex, contributions) in cell_contributions {
        for local_index in 0..4 {
            divergence[simplex[local_index]] += contributions[local_index];
        }
    }

    Ok(Array1::from_vec(divergence).into_pyarray(py))
}

#[pyfunction]
pub(crate) fn recover_demag_nodal_field<'py>(
    py: Python<'py>,
    simplices: PyReadonlyArray2<'py, i64>,
    volumes: PyReadonlyArray1<'py, f64>,
    cell_h: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    point_count: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let simplices = simplices.as_array();
    let volumes = volumes.as_array();
    let cell_h = cell_h.as_array();
    let weights = weights.as_array();

    if simplices.ndim() != 2 || simplices.shape()[1] != 4 {
        return Err(PyValueError::new_err("simplices must have shape (n, 4)"));
    }
    if volumes.len() != simplices.shape()[0] {
        return Err(PyValueError::new_err(
            "volumes must contain one value per simplex",
        ));
    }
    if cell_h.ndim() != 2 || cell_h.shape()[0] != simplices.shape()[0] || cell_h.shape()[1] != 3 {
        return Err(PyValueError::new_err("cell_h must have shape (n, 3)"));
    }
    if weights.len() != point_count {
        return Err(PyValueError::new_err(
            "weights must contain one value per point",
        ));
    }

    let cell_count = simplices.shape()[0];
    let mut simplex_indices: Vec<[usize; 4]> = Vec::with_capacity(cell_count);
    for cell_index in 0..cell_count {
        simplex_indices.push([
            checked_index(simplices[[cell_index, 0]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 1]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 2]], point_count, "simplex node")?,
            checked_index(simplices[[cell_index, 3]], point_count, "simplex node")?,
        ]);
    }

    let cell_contribution = |cell_index: usize| {
        let volume = volumes[cell_index];
        if volume <= 0.0 {
            return None;
        }
        Some((
            simplex_indices[cell_index],
            [
                cell_h[[cell_index, 0]] * volume,
                cell_h[[cell_index, 1]] * volume,
                cell_h[[cell_index, 2]] * volume,
            ],
        ))
    };
    let cell_contributions: Vec<Option<([usize; 4], Vec3)>> = if should_parallelise(cell_count) {
        (0..cell_count)
            .into_par_iter()
            .map(cell_contribution)
            .collect()
    } else {
        (0..cell_count).map(cell_contribution).collect()
    };

    let mut nodal_h = vec![0.0; point_count * 3];
    for local_index in 0..4 {
        for contribution in &cell_contributions {
            let Some((simplex, weighted_h)) = contribution else {
                continue;
            };
            let point_index = simplex[local_index];
            let base = point_index * 3;
            for component in 0..3 {
                nodal_h[base + component] += weighted_h[component];
            }
        }
    }

    for point_index in 0..point_count {
        let weight = weights[point_index];
        if weight > 0.0 {
            let base = point_index * 3;
            for component in 0..3 {
                nodal_h[base + component] /= weight;
            }
        }
    }

    let nodal_array = Array2::from_shape_vec((point_count, 3), nodal_h)
        .map_err(|err| PyValueError::new_err(format!("could not shape nodal field: {err}")))?;
    Ok(nodal_array.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn demag_cell_field_average<'py>(
    py: Python<'py>,
    cell_h: PyReadonlyArray2<'py, f64>,
    volumes: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cell_h = cell_h.as_array();
    let volumes = volumes.as_array();

    if cell_h.ndim() != 2 || cell_h.shape()[1] != 3 {
        return Err(PyValueError::new_err("cell_h must have shape (n, 3)"));
    }
    if volumes.len() != cell_h.shape()[0] {
        return Err(PyValueError::new_err(
            "volumes must contain one value per cell_h row",
        ));
    }

    let (weighted_sum, total_volume) = (0..cell_h.shape()[0])
        .into_par_iter()
        .filter_map(|cell_index| {
            let volume = volumes[cell_index];
            if volume <= 0.0 {
                return None;
            }
            Some((
                [
                    cell_h[[cell_index, 0]] * volume,
                    cell_h[[cell_index, 1]] * volume,
                    cell_h[[cell_index, 2]] * volume,
                ],
                volume,
            ))
        })
        .reduce(
            || ([0.0, 0.0, 0.0], 0.0),
            |left, right| {
                (
                    [
                        left.0[0] + right.0[0],
                        left.0[1] + right.0[1],
                        left.0[2] + right.0[2],
                    ],
                    left.1 + right.1,
                )
            },
        );

    let average = if total_volume > 0.0 {
        [
            weighted_sum[0] / total_volume,
            weighted_sum[1] / total_volume,
            weighted_sum[2] / total_volume,
        ]
    } else {
        [0.0, 0.0, 0.0]
    };
    Ok(Array1::from_vec(average.to_vec()).into_pyarray(py))
}

pub(crate) fn fem_cell_geometry(
    points: ArrayView2<'_, f64>,
    simplex: [usize; 4],
) -> Option<FemCellGeometry> {
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
    let determinant = determinant3(matrix);
    if determinant == 0.0 || !determinant.is_finite() {
        return None;
    }
    let inverse = inverse3(matrix)?;
    let gradients = [
        [
            -(inverse[0][0] + inverse[1][0] + inverse[2][0]),
            -(inverse[0][1] + inverse[1][1] + inverse[2][1]),
            -(inverse[0][2] + inverse[1][2] + inverse[2][2]),
        ],
        inverse[0],
        inverse[1],
        inverse[2],
    ];
    Some(FemCellGeometry {
        simplex,
        gradients,
        volume: determinant.abs() / 6.0,
    })
}

#[cfg(test)]
#[path = "../tests/unit/fem.rs"]
mod tests;
