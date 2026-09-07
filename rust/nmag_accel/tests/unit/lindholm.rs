use super::*;

#[test]
fn boundary_local_indices_reject_interior_and_oversized_nodes() {
    assert!(checked_boundary_local_index(-1, 2).is_err());
    assert!(checked_boundary_local_index(2, 2).is_err());
    assert_eq!(checked_boundary_local_index(1, 2).unwrap(), 1);
}

#[test]
fn invalid_log_ratio_and_zero_angle_are_handled() {
    assert!(safe_log_ratio(1.0, 0.0).is_nan());
    assert_eq!(
        triangle_space_angle_from_offsets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        0.0
    );
}
