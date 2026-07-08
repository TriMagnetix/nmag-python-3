import numpy as np

import nmesh


def test_deterministic_meshing_uses_fixed_rng_seed():
    kwargs = dict(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[nmesh.Box([0.1, 0.1], [0.9, 0.9])],
        a0=0.35,
        max_steps=20,
        nr_probes_for_determining_volume=500,
    )

    mesh_a = nmesh.Mesh(**kwargs)
    mesh_b = nmesh.Mesh(**kwargs)

    assert mesh_a.points == mesh_b.points
    assert mesh_a.simplices == mesh_b.simplices
    assert mesh_a.regions == mesh_b.regions


def test_periodic_scenario_completes_with_valid_topology():
    mesh = nmesh.Mesh(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[nmesh.Box([0.25, 0.25], [0.75, 0.75])],
        mesh_bounding_box=True,
        periodic=[True, False],
        a0=0.4,
        max_steps=20,
        nr_probes_for_determining_volume=500,
    )

    assert len(mesh.points) > 0
    assert len(mesh.simplices) > 0
    assert len(mesh.periodic_point_indices) > 0
    assert all(len(group) >= 2 for group in mesh.periodic_point_indices)
    assert all(region >= 0 for region in mesh.regions)


def test_multi_axis_periodic_mesh_groups_corners_and_edges():
    mesh = nmesh.Mesh(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[nmesh.Box([0.0, 0.0], [1.0, 1.0])],
        mesh_bounding_box=True,
        periodic=[True, True],
        a0=0.4,
        max_steps=20,
        nr_probes_for_determining_volume=600,
    )

    group_sizes = sorted(len(group) for group in mesh.periodic_point_indices)
    assert len(mesh.simplices) > 0
    assert 4 in group_sizes
    assert group_sizes.count(2) >= 2


def test_hint_driven_scenario_completes_with_valid_topology():
    seed_mesh = nmesh.mesh_from_points_and_simplices(
        points=[[0.2, 0.2], [0.8, 0.2], [0.5, 0.75]],
        simplices_indices=[[0, 1, 2]],
        simplices_regions=[5],
    )

    mesh = nmesh.Mesh(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[],
        mesh_bounding_box=True,
        hints=[(seed_mesh, nmesh.Box([0.1, 0.1], [0.9, 0.9]))],
        a0=0.4,
        max_steps=20,
        nr_probes_for_determining_volume=500,
    )

    points = np.asarray(mesh.points, dtype=float)
    assert len(mesh.points) > 0
    assert len(mesh.simplices) > 0
    assert all(mesh.points.count(point) == 1 for point in mesh.points)
    assert np.any(np.all(np.isclose(points, [0.2, 0.2]), axis=1))
    assert np.any(np.all(np.isclose(points, [0.8, 0.2]), axis=1))
    assert np.any(np.all(np.isclose(points, [0.5, 0.75]), axis=1))


def test_adjacent_regions_keep_valid_piece_topology():
    mesh = nmesh.Mesh(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[
            nmesh.Box([0.0, 0.0], [0.5, 1.0]),
            nmesh.Box([0.5, 0.0], [1.0, 1.0]),
        ],
        a0=0.5,
        max_steps=20,
        nr_probes_for_determining_volume=500,
    )

    assert len(mesh.points) > 0
    assert len(mesh.simplices) > 0
    assert sorted(set(mesh.regions)) == [1, 2]
    assert len(mesh.surfaces) > 0


def test_concave_difference_domain_keeps_hole_empty():
    hole = nmesh.Box([0.35, 0.35], [0.65, 0.65])
    shell = nmesh.difference(
        nmesh.Box([0.0, 0.0], [1.0, 1.0]),
        [hole],
    )

    mesh = nmesh.Mesh(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[shell],
        a0=0.25,
        max_steps=20,
        nr_probes_for_determining_volume=1000,
    )

    points = np.asarray(mesh.points, dtype=float)
    centroids = np.mean(points[np.asarray(mesh.simplices, dtype=int)], axis=1)
    inside_hole = np.asarray(hole.obj.evaluate(centroids), dtype=float) > 0.0
    assert len(mesh.simplices) > 0
    assert not np.any(inside_hole)
    assert len(mesh.surfaces) >= 8
