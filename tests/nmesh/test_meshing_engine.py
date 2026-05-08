import unittest

import numpy as np

from nmesh import Box, Mesh
from nmesh.mesher.relaxation import (
    _compile_density_function,
    assemble_raw_mesh,
    fem_geometry_from_bodies,
    RelaxationEngine,
)
from nmesh.mesher.relaxation._constants import BOUNDARY_FUZZ, STATE_BOUNDARY, STATE_MOBILE, STATE_SIMPLE
from nmesh.mesher.relaxation._constants import STATE_FIXED
from nmesh.mesher.relaxation.finalize import snap_final_boundary_points
from nmesh.mesher.relaxation.forces import (
    _classify_relevant_simplices,
    _extract_force_parameters,
    compute_forces,
)
from nmesh.mesher.relaxation.recovery import mirror_surface_recovery_points
from nmesh.mesher.relaxation.seeding import _classify_dynamic_states, _prepare_initial_points
from nmesh.mesher.relaxation.topology import _orient_simplices_positive
from nmesh.mesher.periodic import build_periodic_groups


class TestMeshingEngine(unittest.TestCase):
    def _box_geometry(self):
        return fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])),
            [Box([0.0, 0.0], [1.0, 1.0]).obj],
            [],
        )

    def test_boundary_state_detection_marks_surface_points(self):
        geometry = self._box_geometry()

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
        geometry = self._box_geometry()

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
        geometry = self._box_geometry()

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

    def test_force_relevant_simplex_classifier_rejects_boundary_crossing_probe(self):
        geometry = self._box_geometry()
        points = np.asarray(
            [
                [0.5, 0.5],
                [1.2, 0.5],
                [0.5, 0.8],
            ],
            dtype=float,
        )
        simplices = np.asarray([[0, 1, 2]], dtype=int)
        states = np.full(len(points), STATE_MOBILE, dtype=int)
        measures = np.asarray([0.105], dtype=float)

        mask = _classify_relevant_simplices(
            points,
            states,
            geometry,
            simplices,
            measures,
            2,
            _extract_force_parameters({}, step=10),
        )

        self.assertFalse(bool(mask[0]))

    def test_surface_recovery_mirrors_poor_2d_boundary_simplex(self):
        geometry = self._box_geometry()
        points = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.2],
            ],
            dtype=float,
        )
        states = np.asarray([STATE_BOUNDARY, STATE_BOUNDARY, STATE_MOBILE], dtype=int)

        recovery_points = mirror_surface_recovery_points(
            points,
            states,
            geometry,
            a0=0.5,
            simplices=np.asarray([[0, 1, 2]], dtype=int),
        )

        self.assertEqual(len(recovery_points), 1)
        self.assertTrue(np.allclose(recovery_points[0], [0.5, -0.2]))

    def test_geometry_projects_outside_point_back_to_boundary(self):
        geometry = self._box_geometry()

        projected = geometry.project_segment_to_domain(
            np.asarray([0.2, 0.5], dtype=float),
            np.asarray([-0.2, 0.5], dtype=float),
        )

        self.assertGreaterEqual(geometry.classify_points(projected.reshape(1, -1))[0], 0)
        self.assertAlmostEqual(projected[0], 0.0, places=4)

    def test_final_cleanup_snaps_near_fixed_neighbors_to_boundary(self):
        geometry = self._box_geometry()
        points = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [0.05, 0.5],
            ],
            dtype=float,
        )
        states = np.asarray([STATE_FIXED, STATE_FIXED, STATE_MOBILE], dtype=int)

        snapped_points, snapped_states = snap_final_boundary_points(
            points,
            states,
            geometry,
            a0=0.5,
            params={
                "boundary_condition_acceptable_fuzz": 1.0e-6,
                "boundary_condition_max_nr_correction_steps": 200,
            },
        )

        self.assertAlmostEqual(snapped_points[2][0], 0.0, places=5)
        self.assertEqual(int(snapped_states[2]), STATE_BOUNDARY)

    def test_final_cleanup_removes_outside_dynamic_points(self):
        geometry = self._box_geometry()
        engine = RelaxationEngine(
            geometry,
            {"parameters": {}},
            0.5,
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            [False, False],
        )
        engine.points = np.asarray(
            [
                [0.2, 0.2],
                [0.8, 0.2],
                [0.2, 0.8],
                [1.2, 0.5],
            ],
            dtype=float,
        )
        engine.states = np.asarray(
            [STATE_MOBILE, STATE_MOBILE, STATE_MOBILE, STATE_MOBILE],
            dtype=int,
        )
        engine.last_point_density = np.ones(len(engine.points), dtype=float)

        points, states = engine._final_output_points_and_states()

        self.assertEqual(len(points), 3)
        self.assertEqual(states.tolist(), [STATE_MOBILE, STATE_MOBILE, STATE_MOBILE])

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

    def test_simplex_orientation_is_normalized_positive(self):
        points = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        simplices = np.asarray([[0, 2, 1]], dtype=int)

        oriented = _orient_simplices_positive(points, simplices, dim=2)
        determinant = np.linalg.det(points[oriented[0, 1:]] - points[oriented[0, [0]]])

        self.assertGreater(determinant, 0.0)

    def test_periodic_groups_merge_multi_axis_corners(self):
        points = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.5, 0.0],
                [0.5, 1.0],
            ],
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
        points = np.asarray(
            [
                [0.0, 0.5],
                [1.0, 0.5 + 1.0e-9],
            ],
            dtype=float,
        )

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
        engine = RelaxationEngine(
            self._box_geometry(),
            {"parameters": {"controller_step_limit_max": 250, "nr_probes_for_determining_volume": 200}},
            0.5,
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            [False, False],
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
        engine = RelaxationEngine(
            self._box_geometry(),
            {"parameters": {"controller_step_limit_max": 100, "nr_probes_for_determining_volume": 200}},
            0.5,
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            [False, False],
        )

        scheduled_steps = []
        for step in range(1, 30):
            engine.step = step
            if engine._should_attempt_point_change():
                scheduled_steps.append(step)

        self.assertEqual(scheduled_steps, [11, 14, 19, 26])

    def test_simply_points_disable_generated_seed_points(self):
        simply_points = np.asarray(
            [
                [0.1, 0.1],
                [0.9, 0.1],
                [0.5, 0.8],
            ],
            dtype=float,
        )

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
        mobile_points = np.asarray(
            [
                [0.2, 0.2],
                [0.8, 0.2],
                [0.5, 0.7],
            ],
            dtype=float,
        )

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
        engine = RelaxationEngine(
            self._box_geometry(),
            {"parameters": {"controller_step_limit_max": 20, "nr_probes_for_determining_volume": 200}},
            0.5,
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            [False, False],
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


if __name__ == "__main__":
    unittest.main()
