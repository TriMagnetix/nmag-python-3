import unittest

import numpy as np

from nmesh import Box, Mesh
from nmesh.mesher.relaxation import _compile_density_function


class TestMeshingEngine(unittest.TestCase):
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
