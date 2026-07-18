use ndarray::{arr1, arr2};

use super::cluster::ClusterTree;
use super::entry::ExactLindholmEvaluator;
use super::operator::{BuildOptions, HMatrixOperator};

fn tetrahedron_evaluator() -> ExactLindholmEvaluator {
    let points = arr2(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let simplices = arr2(&[[0_i64, 1, 2, 3]]);
    let faces = arr2(&[[1_i64, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]);
    let boundary_nodes = arr1(&[0_i64, 1, 2, 3]);
    let local_indices = arr1(&[0_i64, 1, 2, 3]);
    ExactLindholmEvaluator::new(
        points.view(),
        simplices.view(),
        faces.view(),
        boundary_nodes.view(),
        local_indices.view(),
    )
    .unwrap()
}

fn options(memory_budget: usize) -> BuildOptions {
    BuildOptions {
        tolerance: 1.0e-12,
        eta: 2.0,
        leaf_size: 2,
        max_rank: 4,
        validation_vectors: 2,
        validation_rows: 4,
        memory_budget,
    }
}

#[test]
fn cluster_tree_is_deterministic_and_covers_every_point() {
    let points = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ];
    let first = ClusterTree::build(&points, 1);
    let second = ClusterTree::build(&points, 1);

    assert_eq!(first.permutation, vec![0, 2, 1, 3]);
    assert_eq!(first.permutation, second.permutation);
    assert_eq!(first.nodes[first.root].start, 0);
    assert_eq!(first.nodes[first.root].end, points.len());
}

#[test]
fn hierarchical_action_matches_exact_entries_deterministically() {
    let evaluator = tetrahedron_evaluator();
    let operator = HMatrixOperator::build(&evaluator, options(1_000_000)).unwrap();
    let input = [0.25, -0.5, 1.25, 0.75];
    let expected = (0..evaluator.len())
        .map(|row| evaluator.row_dot(row, &input))
        .collect::<Vec<_>>();

    let first = operator.matvec(&input);
    let second = operator.matvec(&input);
    assert_eq!(first, second);
    for (actual, expected) in first.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-13);
    }
    assert!(operator.diagnostics().sampled_relative_error <= 1.0e-12);
}

#[test]
fn construction_stops_at_the_memory_budget() {
    let evaluator = tetrahedron_evaluator();
    let error = HMatrixOperator::build(&evaluator, options(1))
        .err()
        .unwrap();

    assert!(error.contains("memory budget exceeded"));
}
