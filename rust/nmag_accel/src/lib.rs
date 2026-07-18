mod common;
mod dynamics_bdf;
mod fem;
mod lindholm;
mod lindholm_hmatrix;
mod llg_maxangle;
mod mesh_probe;
mod parallel;

use pyo3::prelude::*;

const API_VERSION: u32 = 2;

#[pymodule]
fn nmag_accel(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("API_VERSION", API_VERSION)?;
    module.add_function(wrap_pyfunction!(
        lindholm::build_lindholm_bem_matrix,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        lindholm::apply_lindholm_bem_matrix_free,
        module
    )?)?;
    module.add_class::<lindholm_hmatrix::LindholmHMatrix>()?;
    module.add_function(wrap_pyfunction!(
        lindholm_hmatrix::build_lindholm_hmatrix,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(fem::build_demag_fem_geometry, module)?)?;
    module.add_function(wrap_pyfunction!(fem::build_demag_fem_divergence, module)?)?;
    module.add_function(wrap_pyfunction!(fem::recover_demag_nodal_field, module)?)?;
    module.add_function(wrap_pyfunction!(fem::demag_cell_field_average, module)?)?;
    module.add_function(wrap_pyfunction!(
        mesh_probe::build_oriented_boundary_faces,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        mesh_probe::build_probe_tetrahedral_geometry,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(llg_maxangle::llg_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(
        llg_maxangle::llg_rhs_heterogeneous,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        llg_maxangle::llg_rhs_stt_heterogeneous,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(dynamics_bdf::integrate_llg_bdf, module)?)?;
    module.add_function(wrap_pyfunction!(
        llg_maxangle::maxangle_between_edges,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(parallel::parallel_runtime_info, module)?)?;
    Ok(())
}
