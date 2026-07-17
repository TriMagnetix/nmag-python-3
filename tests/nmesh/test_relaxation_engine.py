import unittest

import numpy as np

from nmesh import Box, Mesh
from nmesh.mesher.periodic import build_periodic_groups
from nmesh.mesher.relaxation import (
    RelaxationEngine,
    _compile_density_function,
    assemble_raw_mesh,
    fem_geometry_from_bodies,
)
from nmesh.mesher.relaxation._constants import BOUNDARY_FUZZ, STATE_SIMPLE
from nmesh.mesher.relaxation.seeding import _prepare_initial_points
from nmesh.mesher.relaxation.topology import _orient_simplices_positive


class TestRelaxationEngine(unittest.TestCase):
    def _box_geometry(self):
        return fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])),
            [Box([0.0, 0.0], [1.0, 1.0]).obj],
            [],
        )

    def test_assemble_raw_mesh_filters_flat_boundary_sliver(self):
        geometry = fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, 0.01])),
            [Box([0.0, 0.0], [1.0, 0.01]).obj],
            [],
        )
        raw_mesh = assemble_raw_mesh(
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 0.01]], dtype=float),
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
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, height]], dtype=float),
            geometry,
            [False, False],
            params={"controller_smallest_allowed_volume_ratio": 1.0},
        )

        self.assertEqual(len(raw_mesh.simplices), 1)

    def test_simplex_orientation_is_normalized_positive(self):
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        simplices = np.asarray([[0, 2, 1]], dtype=int)

        oriented = _orient_simplices_positive(points, simplices, dim=2)
        determinant = np.linalg.det(points[oriented[0, 1:]] - points[oriented[0, [0]]])

        self.assertGreater(determinant, 0.0)

    def test_periodic_groups_merge_multi_axis_corners(self):
        points = np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.0], [0.5, 1.0]],
            dtype=float,
        )

        groups = build_periodic_groups(
            points,
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
            [True, True],
            tolerance=1.0e-6,
        )

        self.assertIn([0, 1, 2, 3], groups)
        self.assertIn([4, 5], groups)

    def test_periodic_groups_do_not_round_non_periodic_coordinates(self):
        points = np.asarray([[0.0, 0.5], [1.0, 0.5 + 1.0e-9]], dtype=float)

        groups = build_periodic_groups(
            points,
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
            [True, False],
            tolerance=1.0e-6,
        )

        self.assertEqual(groups, [])

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

    def test_invalid_density_string_raises_instead_of_falling_back(self):
        with self.assertRaises(ValueError):
            _compile_density_function("density = ;")

    def test_engine_honors_configured_max_steps_without_python_cap(self):
        engine = self._new_engine(
            {"controller_step_limit_max": 250, "nr_probes_for_determining_volume": 200}
        )

        self.assertEqual(engine.max_steps, 250)

    def test_boundary_drift_tolerance_does_not_scale_with_a0(self):
        engine = RelaxationEngine(
            self._box_geometry(),
            {"parameters": {"nr_probes_for_determining_volume": 200}},
            10.0,
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            [False, False],
        )

        self.assertEqual(engine.boundary_drift_tolerance, BOUNDARY_FUZZ)

    def test_point_change_schedule_uses_legacy_square_steps(self):
        engine = self._new_engine(
            {"controller_step_limit_max": 100, "nr_probes_for_determining_volume": 200}
        )

        scheduled_steps = []
        for step in range(1, 30):
            engine.step = step
            if engine._should_attempt_point_change():
                scheduled_steps.append(step)

        self.assertEqual(scheduled_steps, [11, 14, 19, 26])

    def test_simply_points_disable_generated_seed_points(self):
        simply_points = np.asarray([[0.1, 0.1], [0.9, 0.1], [0.5, 0.8]], dtype=float)

        points, states = _prepare_initial_points(
            self._box_geometry(),
            0.25,
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            simply_points,
            [False, False],
            np.random.default_rng(97),
        )

        self.assertEqual(points.tolist(), simply_points.tolist())
        self.assertEqual(states.tolist(), [STATE_SIMPLE, STATE_SIMPLE, STATE_SIMPLE])

    def test_user_mobile_points_disable_generated_seed_points(self):
        mobile_points = np.asarray([[0.2, 0.2], [0.8, 0.2], [0.5, 0.7]], dtype=float)

        points, _states = _prepare_initial_points(
            self._box_geometry(),
            0.25,
            np.empty((0, 2), dtype=float),
            mobile_points,
            np.empty((0, 2), dtype=float),
            [False, False],
            np.random.default_rng(97),
        )

        self.assertEqual(points.tolist(), mobile_points.tolist())

    def test_step_limit_waits_for_post_change_settling(self):
        engine = self._new_engine(
            {"controller_step_limit_max": 20, "nr_probes_for_determining_volume": 200}
        )

        engine.last_addition_deletion_step = 11
        engine.step = 20
        self.assertFalse(engine._step_limit_reached())
        engine.step = 61
        self.assertTrue(engine._step_limit_reached())

    def test_meshing_is_deterministic_for_identical_inputs(self):
        kwargs = dict(
            bounding_box=[[0.0, 0.0], [1.0, 1.0]],
            objects=[Box([0.1, 0.1], [0.9, 0.9])],
            a0=0.5,
            max_steps=20,
            nr_probes_for_determining_volume=500,
        )

        mesh_a = Mesh(**kwargs)
        mesh_b = Mesh(**kwargs)

        self.assertEqual(mesh_a.points, mesh_b.points)
        self.assertEqual(mesh_a.simplices, mesh_b.simplices)
        self.assertEqual(mesh_a.regions, mesh_b.regions)

    def _new_engine(self, parameters):
        return RelaxationEngine(
            self._box_geometry(),
            {"parameters": parameters},
            0.5,
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            [False, False],
        )
