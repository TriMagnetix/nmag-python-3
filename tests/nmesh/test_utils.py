import unittest
import numpy as np
import nmesh.utils as utils
import os

class TestUtils(unittest.TestCase):
    def test_array_filter(self):
        arr = [1, 2, 3, 4, 5]
        p = lambda x: x % 2 == 0
        expected = [2, 4]
        np.testing.assert_array_equal(utils.array_filter(p, arr), expected)

    def test_array_position(self):
        arr = [10, 20, 30, 40, 30]
        self.assertEqual(utils.array_position(30, arr), 2)
        self.assertEqual(utils.array_position(30, arr, start=3), 4)
        self.assertEqual(utils.array_position(50, arr), -1)

    def test_array_position_if(self):
        arr = [1, 3, 5, 8, 10]
        p = lambda x: x % 2 == 0
        self.assertEqual(utils.array_position_if(p, arr), 3)
        self.assertEqual(utils.array_position_if(p, arr, start=4), 4)

    def test_array_one_shorter(self):
        arr = [1, 2, 3, 4]
        np.testing.assert_array_equal(utils.array_one_shorter(arr, 1), [1, 3, 4])

    def test_determinant(self):
        mx = [[1, 2], [3, 4]]
        # det = 1*4 - 2*3 = -2
        self.assertAlmostEqual(utils.determinant(mx), -2.0, places=9)

    def test_inverse(self):
        mx = [[1, 2], [3, 4]]
        inv = utils.inverse(mx)
        expected = [[-2.0, 1.0], [1.5, -0.5]]
        np.testing.assert_array_almost_equal(inv, expected, decimal=9)

    def test_cross_product_3d(self):
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        expected = [0, 0, 1]
        np.testing.assert_array_equal(utils.cross_product_3d(v1, v2), expected)

    def test_time_vmem_rss(self):
        t, vmem, rss = utils.time_vmem_rss()
        self.assertGreaterEqual(t, 0.0)
        # On non-linux this might be 0.0, but if /proc/self/status exists it should be > 0
        if os.path.exists("/proc/self/status"):
            self.assertGreater(vmem, 0.0)
            self.assertGreater(rss, 0.0)

if __name__ == '__main__':
    unittest.main()
