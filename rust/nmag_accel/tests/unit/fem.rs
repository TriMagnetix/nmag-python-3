use super::*;
use ndarray::array;

#[test]
fn degenerate_tetrahedron_has_no_geometry() {
    let points = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ];
    assert!(fem_cell_geometry(points.view(), [0, 1, 2, 3]).is_none());
}

#[test]
fn tetrahedron_volume_is_positive_for_reversed_orientation() {
    let points = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    let geometry = fem_cell_geometry(points.view(), [0, 2, 1, 3]).unwrap();
    assert!((geometry.volume - 1.0 / 6.0).abs() < 1e-14);
}

#[test]
fn reference_tetrahedron_gradients_preserve_constant_fields() {
    let points = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    let geometry = fem_cell_geometry(points.view(), [0, 1, 2, 3]).unwrap();
    let gradient_sum = geometry.gradients.iter().fold([0.0; 3], |sum, gradient| {
        [
            sum[0] + gradient[0],
            sum[1] + gradient[1],
            sum[2] + gradient[2],
        ]
    });
    assert!(gradient_sum.iter().all(|value| value.abs() < 1e-14));

    let local_stiffness: [[f64; 4]; 4] = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            geometry.volume * dot3(geometry.gradients[row], geometry.gradients[column])
        })
    });
    for (row_index, row) in local_stiffness.iter().enumerate() {
        assert!((row.iter().sum::<f64>()).abs() < 1e-14);
        for (column_index, value) in row.iter().enumerate() {
            assert!((value - local_stiffness[column_index][row_index]).abs() < 1e-14);
        }
    }
}
