import unittest
import os
import numpy as np
import nmesh
from nmesh.backend import nmesh_backend as backend

class TestNMesh(unittest.TestCase):
    def test_meshing_parameters(self):
        """Test MeshingParameters setters and getters."""
        params = nmesh.get_default_meshing_parameters()
        params.dim = 3
        
        # Test individual setters
        params.set_shape_force_scale(0.5)
        self.assertEqual(params["shape_force_scale"], 0.5)
        
        params.set_max_steps(5000)
        self.assertEqual(params["max_steps"], 5000)
        
        # Test item setting
        params["volume_force_scale"] = 0.1
        self.assertEqual(params["volume_force_scale"], 0.1)

    def test_box_primitive(self):
        """Test Box primitive creation and transformations."""
        p1 = [0.0, 0.0, 0.0]
        p2 = [1.0, 1.0, 1.0]
        b = nmesh.Box(p1, p2, use_fixed_corners=True)
        
        self.assertEqual(b.dim, 3)
        # 8 corners for a 3D box
        self.assertEqual(len(b.fixed_points), 8)
        
        # Test transformation
        b.shift([1.0, 0.0, 0.0])
        b.scale([2.0, 2.0, 2.0])
        b.rotate(0, 1, 90)

    def test_csg_operations(self):
        """Test CSG operations like union and difference."""
        b1 = nmesh.Box([0,0,0], [1,1,1])
        b2 = nmesh.Box([0.5,0.5,0.5], [1.5,1.5,1.5])
        
        u = nmesh.union([b1, b2])
        self.assertEqual(u.dim, 3)
        
        d = nmesh.difference(b1, [b2])
        self.assertEqual(d.dim, 3)

    def test_mesh_generation(self):
        """Test functional Mesh generation."""
        bb = [[0,0,0], [1,1,1]]
        obj = nmesh.Box([0.2,0.2,0.2], [0.8,0.8,0.8])
        
        # We use a large a0 to keep it fast
        m = nmesh.Mesh(bounding_box=bb, objects=[obj], a0=0.5)
        
        # It should have some points and simplices now
        self.assertGreater(len(m.points), 0)
        self.assertGreater(len(m.simplices), 0)
        self.assertGreater(len(m.regions), 0)
        
        # Check if points are within bounding box
        for p in m.points:
            for i in range(3):
                self.assertGreaterEqual(p[i], 0.0 - 1e-9)
                self.assertLessEqual(p[i], 1.0 + 1e-9)

    def test_1d_mesh_generation(self):
        """Test 1D mesh generation logic."""
        regions = [(0.0, 1.0), (1.0, 2.0)]
        discretization = 0.5
        
        m = nmesh.generate_1d_mesh(regions, discretization)
        self.assertIsInstance(m, nmesh.MeshBase)
        
        pts, simps, regs = nmesh.generate_1d_mesh_components(regions, discretization)
        self.assertEqual(len(pts), 5) # 0.0, 0.5, 1.0, 1.5, 2.0
        self.assertEqual(len(simps), 4)
        self.assertEqual(len(regs), 4)

    def test_outer_corners(self):
        """Test outer_corners utility."""
        from nmesh.base import MeshBase
        class MockMesh(MeshBase):
            @property
            def points(self):
                return [[0,0], [1,2], [-1,1]]
        
        m = MockMesh(None)
        min_c, max_corner = nmesh.outer_corners(m)
        self.assertEqual(min_c, [-1, 0])
        self.assertEqual(max_corner, [1, 2])

    def test_write_read_mesh(self):
        """Test writing and reading mesh back."""
        points = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        simplices = [(1, [0, 1, 2]), (1, [1, 2, 3])]
        surfaces = []
        data = (points, simplices, surfaces)
        
        test_file = "test_temp.nmesh"
        nmesh.write_mesh(data, out=test_file)
        
        try:
            m = nmesh.load(test_file)
            self.assertEqual(len(m.points), 4)
            self.assertEqual(len(m.simplices), 2)
            self.assertEqual(m.dim, 2)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_periodicity_helpers(self):
        """Test internal periodicity helpers."""
        
        # Test _all_combinations
        combs = backend._all_combinations(2)
        self.assertEqual(len(combs), 4)
        
        # Test _periodic_directions
        masks = backend._periodic_directions([True, False, True])
        # Should return masks for all sub-entities (edges, faces)
        self.assertGreater(len(masks), 0)

    def test_gradient(self):
        """Test numeric gradient calculation."""
        def f(x): return x[0]**2 + x[1]**2
        
        grad = backend.symm_grad(f, [1.0, 2.0])
        # Gradient of x^2 + y^2 is [2x, 2y] -> [2, 4]
        self.assertAlmostEqual(grad[0], 2.0, places=5)
        self.assertAlmostEqual(grad[1], 4.0, places=5)

if __name__ == '__main__':
    unittest.main()
