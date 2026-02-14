import unittest
import math
from pathlib import Path
import nmesh

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

    def test_mesh_generation_stub(self):
        """Test Mesh class initialization with stubs."""
        bb = [[0,0,0], [1,1,1]]
        obj = nmesh.Box([0.2,0.2,0.2], [0.8,0.8,0.8])
        
        m = nmesh.Mesh(bounding_box=bb, objects=[obj], a0=0.1)
        self.assertEqual(str(m), "Mesh with 0 points and 0 simplices") # From stubs
        
        # Test properties (should return empty lists from stubs)
        self.assertEqual(m.points, [])
        self.assertEqual(m.simplices, [])
        self.assertEqual(m.regions, [])

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
        class MockMesh(nmesh.MeshBase):
            @property
            def points(self):
                return [[0,0], [1,2], [-1,1]]
        
        m = MockMesh("raw")
        min_c, max_corner = nmesh.outer_corners(m)
        self.assertEqual(min_c, [-1, 0])
        self.assertEqual(max_corner, [1, 2])

    def test_write_mesh(self):
        """Test write_mesh utility."""
        points = [[0.0, 0.0], [1.0, 1.0]]
        simplices = [(1, [0, 1])]
        surfaces = [(1, [0])]
        data = (points, simplices, surfaces)
        
        import io
        out = io.StringIO()
        nmesh.write_mesh(data, out=out)
        content = out.getvalue()
        self.assertIn("# PYFEM mesh file version 1.0", content)
        self.assertIn("nodes = 2", content)

if __name__ == '__main__':
    unittest.main()
