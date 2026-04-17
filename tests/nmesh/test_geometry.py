import pytest
import numpy as np

from nmesh.geometry import (
    Box,
    Conic,
    Ellipsoid,
    Helix,
    bc_box,
    bc_ellipsoid,
    difference,
    intersect,
    union,
)


def test_boundary_builders_classify_points():
    box = bc_box([0.0, 0.0], [2.0, 4.0])
    ellipsoid = bc_ellipsoid([2.0, 3.0])

    assert box(np.array([[1.0, 2.0]])).item() > 0.0
    assert np.isclose(box(np.array([[0.0, 2.0]])).item(), 0.0)
    assert box(np.array([[3.0, 2.0]])).item() < 0.0

    assert ellipsoid(np.array([[0.0, 0.0]])).item() > 0.0
    assert np.isclose(ellipsoid(np.array([[2.0, 0.0]])).item(), 0.0)
    assert ellipsoid(np.array([[2.5, 0.0]])).item() < 0.0


def test_boundary_values_are_close_to_zero_on_surfaces():
    box = Box([0.0, 0.0], [2.0, 2.0])
    ellipsoid = Ellipsoid([2.0, 3.0])

    assert abs(box.signed_distance([0.0, 1.0])) < 1e-10
    assert abs(ellipsoid.signed_distance([2.0, 0.0])) < 1e-10


def test_invalid_shapes_raise_clear_errors():
    with pytest.raises(ValueError, match="radii must be positive"):
        Ellipsoid([0.0, 1.0])

    with pytest.raises(ValueError, match="edge lengths must be non-zero"):
        Box([0.0, 0.0], [0.0, 2.0])

    with pytest.raises(ValueError, match="Helix radii must be positive"):
        Helix([0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 2.0], 0.4)


def test_geometry_objects_classify_points():
    box = Box([0.0, 0.0], [2.0, 1.0], use_fixed_corners=True)
    ellipsoid = Ellipsoid([2.0, 1.0], transform=[("shift", [1.0, 0.0])])
    cone = Conic([0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 2.0], 0.5)
    helix = Helix([0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 2.0], 0.4)

    assert box.contains([1.0, 0.5])
    assert not box.contains([3.0, 0.5])
    assert len(box.fixed_points) == 4

    assert ellipsoid.contains([1.0, 0.0])
    assert not ellipsoid.contains([3.5, 0.0])

    assert cone.contains([0.0, 0.0, 0.5])
    assert not cone.contains([2.0, 0.0, 0.5])

    assert helix.contains([0.5, 0.5, 0.05])
    assert not helix.contains([2.0, 2.0, 0.0])


def test_csg_operations_preserve_sign_semantics():
    left = Box([0.0, 0.0], [2.0, 2.0])
    right = Box([1.0, 0.0], [3.0, 2.0])

    merged = union([left, right])
    overlap = intersect([left, right])
    cut = difference(left, [right])

    assert merged.contains([0.5, 1.0])
    assert merged.contains([2.5, 1.0])
    assert overlap.contains([1.5, 1.0])
    assert not overlap.contains([0.5, 1.0])
    assert cut.contains([0.5, 1.0])
    assert not cut.contains([1.5, 1.0])


def test_csg_operations_support_more_than_two_objects():
    boxes = [Box([float(index), 0.0], [float(index) + 1.5, 2.0]) for index in range(5)]
    merged = union(boxes)
    overlap = intersect([Box([0.0, 0.0], [4.0, 2.0]), Box([1.0, 0.0], [5.0, 2.0]), Box([2.0, 0.0], [6.0, 2.0])])

    assert merged.contains([0.5, 1.0])
    assert merged.contains([4.5, 1.0])
    assert overlap.contains([2.5, 1.0])
    assert not overlap.contains([0.5, 1.0])


def test_transform_order_matches_system_coordinate_semantics():
    transformed = Box(
        [0.0, 0.0],
        [2.0, 1.0],
        transform=[("shift", [1.0, 0.0]), ("rotate2d", 90.0)],
    )

    inside_point = [-0.5, 1.25]
    outside_point = [0.5, 1.25]

    assert transformed.contains(inside_point)
    assert not transformed.contains(outside_point)


def test_body_coordinate_transform_order_matches_ocaml_composition():
    transformed = Box([0.0, 0.0], [2.0, 1.0])
    transformed.shift([1.0, 0.0], system_coords=False)
    transformed.rotate(0, 1, 90.0, system_coords=False)

    inside_point = [0.25, 1.0]
    outside_point = [1.5, 1.0]

    assert transformed.contains(inside_point)
    assert not transformed.contains(outside_point)


def test_higher_dimensional_box_and_ellipsoid_classification():
    box = Box([0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0])
    ellipsoid = Ellipsoid([2.0, 2.0, 2.0, 2.0])

    assert box.contains([1.0, 1.0, 1.0, 1.0])
    assert not box.contains([3.0, 1.0, 1.0, 1.0])
    assert ellipsoid.contains([1.0, 1.0, 0.0, 0.0])
    assert not ellipsoid.contains([2.5, 0.0, 0.0, 0.0])
