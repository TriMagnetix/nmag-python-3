use super::*;
use ndarray::array;

#[test]
fn boundary_face_vertex_order_is_stable() {
    let simplex = [4usize, 2, 9, 1];
    let face = [
        simplex[TETRA_FACE_VERTICES[0][0]],
        simplex[TETRA_FACE_VERTICES[0][1]],
        simplex[TETRA_FACE_VERTICES[0][2]],
    ];
    assert_eq!(face, [2, 9, 1]);
}

#[test]
fn probe_geometry_skips_singular_tetrahedra() {
    let points = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ];
    let matrix = columns3(
        subtract3(point3(points.view(), 1), point3(points.view(), 0)),
        subtract3(point3(points.view(), 2), point3(points.view(), 0)),
        subtract3(point3(points.view(), 3), point3(points.view(), 0)),
    );
    assert!(inverse3(matrix).is_none());
}
