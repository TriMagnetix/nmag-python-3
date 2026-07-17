use pyo3::prelude::*;

use crate::common::PARALLEL_MIN_ITEMS;

#[pyfunction]
pub(crate) fn parallel_runtime_info() -> (usize, usize) {
    (rayon::current_num_threads(), PARALLEL_MIN_ITEMS)
}
