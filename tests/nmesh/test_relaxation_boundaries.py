import unittest

import numpy as np

from nmesh import Box
from nmesh.mesher.relaxation import RelaxationEngine, fem_geometry_from_bodies
from nmesh.mesher.relaxation._constants import STATE_BOUNDARY, STATE_FIXED, STATE_MOBILE
from nmesh.mesher.relaxation.seeding import _classify_dynamic_states
from nmesh.mesher.relaxation.topology.finalize import snap_final_boundary_points
from nmesh.mesher.relaxation.topology.recovery import mirror_surface_recovery_points


class TestRelaxationBoundaries(unittest.TestCase):
    def _box_geometry(self):
        return fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])),
            [Box([0.0, 0.0], [1.0, 1.0]).obj],
            [],
        )

    def test_boundary_state_detection_marks_surface_points(self):
        geometry = self._box_geometry()
        points = np.asarray([[0.0, 0.5], [0.5, 0.5]], dtype=float)

        states = _classify_dynamic_states(geometry, points, a0=0.5)

        self.assertEqual(states.tolist(), [STATE_BOUNDARY, STATE_MOBILE])

    def test_surface_recovery_mirrors_poor_2d_boundary_simplex(self):
        geometry = self._box_geometry()
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 0.2]], dtype=float)
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
        points = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.05, 0.5]], dtype=float)
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
        engine = RelaxationEngine(
            self._box_geometry(),
            {"parameters": {}},
            0.5,
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            [False, False],
        )
        engine.points = np.asarray([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [1.2, 0.5]], dtype=float)
        engine.states = np.asarray(
            [STATE_MOBILE, STATE_MOBILE, STATE_MOBILE, STATE_MOBILE], dtype=int
        )
        engine.last_point_density = np.ones(len(engine.points), dtype=float)

        points, states = engine._final_output_points_and_states()

        self.assertEqual(len(points), 3)
        self.assertEqual(states.tolist(), [STATE_MOBILE, STATE_MOBILE, STATE_MOBILE])
