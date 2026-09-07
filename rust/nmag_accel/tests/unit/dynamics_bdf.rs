use super::*;

#[test]
fn affine_llg_jacobian_vector_matches_central_difference() {
    let system = AffineLlg {
        operator: vec![0.2, -0.1, 0.0, 0.3, 0.1, 0.0, -0.2, 0.0, 0.4],
        constant_field: vec![0.5, -0.25, 0.75],
        pin: vec![0.8],
        coefficients: LlgCoefficients {
            precession: -1.7,
            damping: -0.3,
            normalisation: 0.2,
        },
        state_size: 3,
        time_scale_seconds: 1.0,
    };
    let state = vec![0.4, -0.7, 0.2];
    let direction = vec![0.3, 0.1, -0.5];
    let mut analytic = vec![0.0; 3];
    system.jacobian_vector(&state, &direction, &mut analytic);

    let epsilon = 1.0e-7;
    let plus: Vec<_> = state
        .iter()
        .zip(&direction)
        .map(|(value, delta)| value + epsilon * delta)
        .collect();
    let minus: Vec<_> = state
        .iter()
        .zip(&direction)
        .map(|(value, delta)| value - epsilon * delta)
        .collect();
    let mut rhs_plus = vec![0.0; 3];
    let mut rhs_minus = vec![0.0; 3];
    system.rhs(&plus, &mut rhs_plus);
    system.rhs(&minus, &mut rhs_minus);
    for component in 0..3 {
        let numerical = (rhs_plus[component] - rhs_minus[component]) / (2.0 * epsilon);
        assert!((analytic[component] - numerical).abs() < 1.0e-8);
    }
}

#[test]
fn constant_field_does_not_contribute_to_field_direction() {
    let system = AffineLlg {
        operator: vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
        constant_field: vec![10.0, 20.0, 30.0],
        pin: vec![1.0],
        coefficients: LlgCoefficients {
            precession: 0.0,
            damping: 0.0,
            normalisation: 0.0,
        },
        state_size: 3,
        time_scale_seconds: 1.0,
    };
    let mut field = vec![0.0; 3];
    system.field(&[1.0, 2.0, 3.0], &mut field);
    assert_eq!(field, vec![12.0, 26.0, 42.0]);
}
