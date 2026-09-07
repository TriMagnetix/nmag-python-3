import unittest

import numpy as np

from nmesh import Box
from nmesh.mesher.relaxation import fem_geometry_from_bodies
from nmesh.mesher.relaxation._constants import STATE_BOUNDARY, STATE_MOBILE
from nmesh.mesher.relaxation.forces import (
    _classify_relevant_simplices,
    _extract_force_parameters,
    compute_forces,
)


class TestRelaxationForces(unittest.TestCase):
    def _box_geometry(self):
        return fem_geometry_from_bodies(
            (np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])),
            [Box([0.0, 0.0], [1.0, 1.0]).obj],
            [],
        )

    def test_force_summary_suppresses_shape_force_at_boundary_corners(self):
        geometry = self._box_geometry()
        points = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.6, 0.45]],
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
        points = np.asarray([[0.0, 0.5], [0.2, 0.4], [0.2, 0.6]], dtype=float)
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

    def test_custom_neighbor_force_callbacks_use_python_path(self):
        geometry = self._box_geometry()
        points = np.asarray([[0.0, 0.5], [0.2, 0.4], [0.2, 0.6]], dtype=float)
        states = np.asarray([STATE_BOUNDARY, STATE_MOBILE, STATE_MOBILE], dtype=int)
        force_calls = []
        boundary_force_calls = []

        def force_fun(reduced_distance):
            force_calls.append(reduced_distance)
            return 1.0 - reduced_distance

        def boundary_force_fun(reduced_distance):
            boundary_force_calls.append(reduced_distance)
            return 1.0 - reduced_distance

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
                "relaxation_force_fun": force_fun,
                "boundary_node_force_fun": boundary_force_fun,
            },
            step=10,
            simplices=np.asarray([[0, 1, 2]], dtype=int),
        )

        self.assertEqual(len(force_calls), 1)
        self.assertAlmostEqual(force_calls[0], 0.5)
        self.assertEqual(len(boundary_force_calls), 2)
        self.assertGreater(np.linalg.norm(summary.total_forces), 0.0)

    def test_force_relevant_simplex_classifier_rejects_boundary_crossing_probe(self):
        geometry = self._box_geometry()
        points = np.asarray([[0.5, 0.5], [1.2, 0.5], [0.5, 0.8]], dtype=float)
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
