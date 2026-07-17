use super::*;

#[test]
fn parallel_threshold_has_an_explicit_serial_boundary() {
    assert!(!should_parallelise(0));
    assert!(!should_parallelise(PARALLEL_MIN_ITEMS - 1));
    assert!(should_parallelise(PARALLEL_MIN_ITEMS));
}

#[test]
fn checked_index_rejects_negative_and_oversized_values() {
    assert!(checked_index(-1, 2, "node").is_err());
    assert!(checked_index(2, 2, "node").is_err());
    assert_eq!(checked_index(1, 2, "node").unwrap(), 1);
}

#[test]
fn vector_helpers_and_inverse_are_consistent() {
    assert_eq!(cross3([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
    assert_eq!(average4([[0.0, 0.0, 0.0]; 4]), [0.0, 0.0, 0.0]);
    let matrix = columns3([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]);
    let inverse = inverse3(matrix).unwrap();
    assert_eq!(inverse[0][0], 0.5);
    assert_eq!(inverse[1][1], 1.0 / 3.0);
    assert_eq!(inverse[2][2], 0.25);
    assert!(inverse3([[0.0; 3]; 3]).is_none());
}
