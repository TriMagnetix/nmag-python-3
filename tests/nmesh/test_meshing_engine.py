import unittest

import numpy as np

from nmesh import Box, Mesh
from nmesh.mesher.relaxation import (
    _compile_density_function,
    assemble_raw_mesh,
    fem_geometry_from_bodies,
)
from nmesh.mesher.relaxation._constants import STATE_BOUNDARY, STATE_MOBILE
from nmesh.mesher.relaxation.forces import compute_forces
from nmesh.mesher.relaxation.seeding import _classify_dynamic_states


class TestMeshingEngine(unittest.TestCase):
    def test_boundary_state_detection_marks_surface_points(self):
        geometry = fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])),
            [Box([0.0, 0.0], [1.0, 1.0]).obj],
            [],
        )

        points = np.asarray(
            [
                [0.0, 0.5],
                [0.5, 0.5],
            ],
            dtype=float,
        )

        states = _classify_dynamic_states(geometry, points, a0=0.5)

        self.assertEqual(states.tolist(), [STATE_BOUNDARY, STATE_MOBILE])

    def test_force_summary_suppresses_shape_force_at_boundary_corners(self):
        geometry = fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])),
            [Box([0.0, 0.0], [1.0, 1.0]).obj],
            [],
        )

        points = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.6, 0.45],
            ],
            dtype=float,
        )
        states = np.full(len(points), STATE_MOBILE, dtype=int)
        summary = compute_forces(
            points,
            states,
            geometry,
            a0=0.5,
            params={
                "controller_shape_force_scale": 0.3,
                "controller_volume_force_scale": 0.0,
                "controller_neigh_force_scale": 0.0,
                "controller_irrel_elem_force_scale": 0.0,
                "controller_sliver_correction": 1.0,
                "controller_initial_settling_steps": 100,
            },
            step=10,
        )

        self.assertGreaterEqual(summary.simplices.shape[0], 2)
        self.assertTrue(np.allclose(summary.total_forces[:4], 0.0))
        self.assertGreater(np.linalg.norm(summary.total_forces[4]), 0.0)
        self.assertGreater(summary.max_effective_force, 0.0)

    def test_boundary_effective_force_uses_tangential_projection(self):
        geometry = fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])),
            [Box([0.0, 0.0], [1.0, 1.0]).obj],
            [],
        )

        points = np.asarray(
            [
                [0.0, 0.5],
                [0.2, 0.4],
                [0.2, 0.6],
            ],
            dtype=float,
        )
        states = np.asarray([STATE_BOUNDARY, STATE_MOBILE, STATE_MOBILE], dtype=int)
        summary = compute_forces(
            points,
            states,
            geometry,
            a0=0.4,
            params={
                "controller_shape_force_scale": 0.0,
                "controller_volume_force_scale": 0.0,
                "controller_neigh_force_scale": 1.0,
                "controller_irrel_elem_force_scale": 0.0,
                "controller_sliver_correction": 1.0,
                "controller_initial_settling_steps": 100,
            },
            step=10,
        )

        self.assertGreater(abs(summary.total_forces[0][0]), 0.0)
        self.assertAlmostEqual(summary.total_forces[0][1], 0.0, places=7)
        self.assertAlmostEqual(summary.point_effective_force[0], 0.0, places=7)

    def test_geometry_projects_outside_point_back_to_boundary(self):
        geometry = fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])),
            [Box([0.0, 0.0], [1.0, 1.0]).obj],
            [],
        )

        projected = geometry.project_segment_to_domain(
            np.asarray([0.2, 0.5], dtype=float),
            np.asarray([-0.2, 0.5], dtype=float),
        )

        self.assertGreaterEqual(geometry.classify_points(projected.reshape(1, -1))[0], 0)
        self.assertAlmostEqual(projected[0], 0.0, places=4)

    def test_assemble_raw_mesh_filters_flat_boundary_sliver(self):
        geometry = fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, 0.01])),
            [Box([0.0, 0.0], [1.0, 0.01]).obj],
            [],
        )

        raw_mesh = assemble_raw_mesh(
            np.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, 0.01],
                ],
                dtype=float,
            ),
            geometry,
            [False, False],
            params={"controller_smallest_allowed_volume_ratio": 1.0},
        )

        self.assertEqual(raw_mesh.simplices, [])

    def test_assemble_raw_mesh_keeps_regular_boundary_simplex(self):
        height = np.sqrt(3.0) / 2.0
        geometry = fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, height])),
            [Box([0.0, 0.0], [1.0, height]).obj],
            [],
        )

        raw_mesh = assemble_raw_mesh(
            np.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, height],
                ],
                dtype=float,
            ),
            geometry,
            [False, False],
            params={"controller_smallest_allowed_volume_ratio": 1.0},
        )

        self.assertEqual(len(raw_mesh.simplices), 1)

    def test_density_string_translation_supports_c_style_blocks(self):
        density = _compile_density_function(
            """
            double upper=7.1;
            double lower=5.9;
            if ((x[0] < 2.1) && (x[0] > -2.1) && (x[1] > lower) && (x[1] < upper))
              {density = 4.0;}
            else {
              double sigma=1.0;
              double xdev = 0.0-x[0];
              double ydev = 0.0-x[1];
              double rdev2 = xdev*xdev+ydev*ydev;
              density=1.0+4.0*exp(-rdev2/(sigma*sigma));
            }
            """
        )

        self.assertAlmostEqual(density(np.asarray([0.0, 6.0])), 4.0)
        self.assertAlmostEqual(density(np.asarray([0.0, 0.0])), 5.0)

    def test_meshing_is_deterministic_for_identical_inputs(self):
        kwargs = dict(
            bounding_box=[[0.0, 0.0], [1.0, 1.0]],
            objects=[Box([0.1, 0.1], [0.9, 0.9])],
            a0=0.5,
        )

        mesh_a = Mesh(**kwargs)
        mesh_b = Mesh(**kwargs)

        self.assertEqual(mesh_a.points, mesh_b.points)
        self.assertEqual(mesh_a.simplices, mesh_b.simplices)
        self.assertEqual(mesh_a.regions, mesh_b.regions)


if __name__ == "__main__":
    unittest.main()
