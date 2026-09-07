use ndarray::{ArrayView1, ArrayView2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyResult;

pub(crate) type Vec3 = [f64; 3];

pub(crate) const PARALLEL_MIN_ITEMS: usize = 2_048;

pub(crate) fn should_parallelise(item_count: usize) -> bool {
    item_count >= PARALLEL_MIN_ITEMS
}

pub(crate) fn checked_index(value: i64, length: usize, label: &str) -> PyResult<usize> {
    if value < 0 || value as usize >= length {
        return Err(PyValueError::new_err(format!(
            "{label} index {value} is out of bounds for {length} points"
        )));
    }
    Ok(value as usize)
}

pub(crate) fn checked_indices(
    values: ArrayView1<'_, i64>,
    length: usize,
    label: &str,
) -> PyResult<Vec<usize>> {
    values
        .iter()
        .map(|value| checked_index(*value, length, label))
        .collect()
}

pub(crate) fn point3(points: ArrayView2<'_, f64>, index: usize) -> Vec3 {
    [points[[index, 0]], points[[index, 1]], points[[index, 2]]]
}

pub(crate) fn subtract3(left: Vec3, right: Vec3) -> Vec3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub(crate) fn scale3(vector: Vec3, scale: f64) -> Vec3 {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

pub(crate) fn average3(vectors: [Vec3; 3]) -> Vec3 {
    scale3(
        [
            vectors[0][0] + vectors[1][0] + vectors[2][0],
            vectors[0][1] + vectors[1][1] + vectors[2][1],
            vectors[0][2] + vectors[1][2] + vectors[2][2],
        ],
        1.0 / 3.0,
    )
}

pub(crate) fn average4(vectors: [Vec3; 4]) -> Vec3 {
    scale3(
        [
            vectors[0][0] + vectors[1][0] + vectors[2][0] + vectors[3][0],
            vectors[0][1] + vectors[1][1] + vectors[2][1] + vectors[3][1],
            vectors[0][2] + vectors[1][2] + vectors[2][2] + vectors[3][2],
        ],
        0.25,
    )
}

pub(crate) fn dot3(left: Vec3, right: Vec3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(crate) fn cross3(left: Vec3, right: Vec3) -> Vec3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub(crate) fn columns3(column0: Vec3, column1: Vec3, column2: Vec3) -> [[f64; 3]; 3] {
    [
        [column0[0], column1[0], column2[0]],
        [column0[1], column1[1], column2[1]],
        [column0[2], column1[2], column2[2]],
    ]
}

pub(crate) fn inverse3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[0][2];
    let d = matrix[1][0];
    let e = matrix[1][1];
    let f = matrix[1][2];
    let g = matrix[2][0];
    let h = matrix[2][1];
    let i = matrix[2][2];

    let cofactor00 = e * i - f * h;
    let cofactor01 = -(d * i - f * g);
    let cofactor02 = d * h - e * g;
    let cofactor10 = -(b * i - c * h);
    let cofactor11 = a * i - c * g;
    let cofactor12 = -(a * h - b * g);
    let cofactor20 = b * f - c * e;
    let cofactor21 = -(a * f - c * d);
    let cofactor22 = a * e - b * d;
    let determinant = a * cofactor00 + b * cofactor01 + c * cofactor02;
    if determinant == 0.0 || !determinant.is_finite() {
        return None;
    }
    let inv_det = 1.0 / determinant;
    Some([
        [
            cofactor00 * inv_det,
            cofactor10 * inv_det,
            cofactor20 * inv_det,
        ],
        [
            cofactor01 * inv_det,
            cofactor11 * inv_det,
            cofactor21 * inv_det,
        ],
        [
            cofactor02 * inv_det,
            cofactor12 * inv_det,
            cofactor22 * inv_det,
        ],
    ])
}

pub(crate) fn determinant3(matrix: [[f64; 3]; 3]) -> f64 {
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[0][2];
    let d = matrix[1][0];
    let e = matrix[1][1];
    let f = matrix[1][2];
    let g = matrix[2][0];
    let h = matrix[2][1];
    let i = matrix[2][2];
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

pub(crate) fn bounds3(tetrahedron: [Vec3; 4], op: fn(f64, f64) -> f64) -> Vec3 {
    [
        op(
            op(tetrahedron[0][0], tetrahedron[1][0]),
            op(tetrahedron[2][0], tetrahedron[3][0]),
        ),
        op(
            op(tetrahedron[0][1], tetrahedron[1][1]),
            op(tetrahedron[2][1], tetrahedron[3][1]),
        ),
        op(
            op(tetrahedron[0][2], tetrahedron[1][2]),
            op(tetrahedron[2][2], tetrahedron[3][2]),
        ),
    ]
}

pub(crate) fn norm3(vector: Vec3) -> f64 {
    dot3(vector, vector).sqrt()
}

pub(crate) fn normalised3(vector: Vec3) -> Vec3 {
    let length = norm3(vector);
    if length == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        scale3(vector, 1.0 / length)
    }
}

pub(crate) fn safe_log_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 0.0 {
        f64::NAN
    } else {
        (numerator / denominator).ln()
    }
}

#[cfg(test)]
#[path = "../tests/unit/common.rs"]
mod tests;
