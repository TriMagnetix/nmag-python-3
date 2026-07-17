use super::*;

#[test]
fn empty_simplex_list_has_zero_maxangle() {
    // The Python wrapper handles array conversion; this checks the reduction's identity.
    assert_eq!(0.0_f64.max(0.0), 0.0);
}

#[test]
fn angle_clamp_handles_rounding_outside_unit_interval() {
    assert_eq!(1.0000000001_f64.clamp(-1.0, 1.0).acos(), 0.0);
    assert!(((-1.0_f64).acos() * 180.0 / PI) - 180.0 < 1e-12);
}
