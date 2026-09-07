mod aca;
mod cluster;
mod entry;
mod operator;

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use entry::ExactLindholmEvaluator;
use operator::{BuildOptions, HMatrixOperator};

#[pyclass(frozen, module = "nmag_accel")]
pub(crate) struct LindholmHMatrix {
    operator: HMatrixOperator,
}

#[pymethods]
impl LindholmHMatrix {
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = values.as_slice()?.to_vec();
        if values.len() != self.operator.len() {
            return Err(PyValueError::new_err(format!(
                "values must contain one entry per boundary node ({})",
                self.operator.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "values must contain only finite numbers",
            ));
        }
        let result = py.detach(|| self.operator.matvec(&values));
        Ok(Array1::from_vec(result).into_pyarray(py))
    }

    #[getter]
    fn size(&self) -> usize {
        self.operator.len()
    }

    #[getter]
    fn storage_bytes(&self) -> usize {
        self.operator.diagnostics().storage_bytes
    }

    #[getter]
    fn dense_blocks(&self) -> usize {
        self.operator.diagnostics().dense_blocks
    }

    #[getter]
    fn low_rank_blocks(&self) -> usize {
        self.operator.diagnostics().low_rank_blocks
    }

    #[getter]
    fn maximum_rank(&self) -> usize {
        self.operator.diagnostics().maximum_rank
    }

    #[getter]
    fn mean_rank(&self) -> f64 {
        self.operator.diagnostics().mean_rank
    }

    #[getter]
    fn sampled_relative_error(&self) -> f64 {
        self.operator.diagnostics().sampled_relative_error
    }
}

#[pyfunction]
#[pyo3(signature = (
    points,
    simplices,
    face_nodes,
    boundary_nodes,
    local_index_by_point,
    relative_tolerance=1.0e-6,
    admissibility_eta=2.0,
    leaf_size=32,
    max_rank=128,
    validation_vectors=4,
    validation_rows=64,
    memory_budget_bytes=usize::MAX,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_lindholm_hmatrix(
    points: PyReadonlyArray2<'_, f64>,
    simplices: PyReadonlyArray2<'_, i64>,
    face_nodes: PyReadonlyArray2<'_, i64>,
    boundary_nodes: PyReadonlyArray1<'_, i64>,
    local_index_by_point: PyReadonlyArray1<'_, i64>,
    relative_tolerance: f64,
    admissibility_eta: f64,
    leaf_size: usize,
    max_rank: usize,
    validation_vectors: usize,
    validation_rows: usize,
    memory_budget_bytes: usize,
) -> PyResult<LindholmHMatrix> {
    if points.shape().len() != 2 || points.shape()[1] != 3 {
        return Err(PyValueError::new_err("points must have shape (n, 3)"));
    }
    if simplices.shape().len() != 2 || simplices.shape()[1] != 4 {
        return Err(PyValueError::new_err("simplices must have shape (n, 4)"));
    }
    if face_nodes.shape().len() != 2 || face_nodes.shape()[1] != 3 {
        return Err(PyValueError::new_err("face_nodes must have shape (n, 3)"));
    }
    if local_index_by_point.len() != points.shape()[0] {
        return Err(PyValueError::new_err(
            "local_index_by_point must contain one entry per point",
        ));
    }
    if points.as_array().iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "points must contain only finite numbers",
        ));
    }
    if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
        return Err(PyValueError::new_err(
            "relative_tolerance must be finite and positive",
        ));
    }
    if !admissibility_eta.is_finite() || admissibility_eta <= 0.0 {
        return Err(PyValueError::new_err(
            "admissibility_eta must be finite and positive",
        ));
    }
    if leaf_size == 0 || max_rank == 0 || validation_vectors == 0 || validation_rows == 0 {
        return Err(PyValueError::new_err(
            "leaf_size, max_rank, validation_vectors, and validation_rows must be positive",
        ));
    }
    let evaluator = ExactLindholmEvaluator::new(
        points.as_array(),
        simplices.as_array(),
        face_nodes.as_array(),
        boundary_nodes.as_array(),
        local_index_by_point.as_array(),
    )?;
    let operator = HMatrixOperator::build(
        &evaluator,
        BuildOptions {
            tolerance: relative_tolerance,
            eta: admissibility_eta,
            leaf_size,
            max_rank,
            validation_vectors,
            validation_rows,
            memory_budget: memory_budget_bytes,
        },
    )
    .map_err(PyValueError::new_err)?;
    Ok(LindholmHMatrix { operator })
}

#[cfg(test)]
#[path = "../../tests/unit/lindholm_hmatrix.rs"]
mod tests;
