import numpy as np

import nmesh


def test_deterministic_meshing_uses_fixed_rng_seed():
    kwargs = dict(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[nmesh.Box([0.1, 0.1], [0.9, 0.9])],
        a0=0.35,
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
    )

    assert len(mesh.points) > 0
    assert len(mesh.simplices) > 0
    assert len(mesh.periodic_point_indices) > 0
    assert all(len(group) >= 2 for group in mesh.periodic_point_indices)
    assert all(region >= 0 for region in mesh.regions)


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
    )

    points = np.asarray(mesh.points, dtype=float)
    assert len(mesh.points) > 0
    assert len(mesh.simplices) > 0
    assert all(mesh.points.count(point) == 1 for point in mesh.points)
    assert np.any(np.all(np.isclose(points, [0.2, 0.2]), axis=1))
    assert np.any(np.all(np.isclose(points, [0.8, 0.2]), axis=1))
    assert np.any(np.all(np.isclose(points, [0.5, 0.75]), axis=1))
