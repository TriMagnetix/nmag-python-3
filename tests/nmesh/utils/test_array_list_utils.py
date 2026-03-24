import unittest

import numpy as np
import pytest
import nmesh.utils.array_list_utils as array_utils


class TestArrayListUtils(unittest.TestCase):
    def test_array_filter(self):
        arr = [1, 2, 3, 4, 5]
        p = lambda x: x % 2 == 0
        expected = [2, 4]
        np.testing.assert_array_equal(array_utils.array_filter(p, arr), expected)

    def test_array_position(self):
        arr = [10, 20, 30, 40, 30]
        self.assertEqual(array_utils.array_position(30, arr), 2)
        self.assertEqual(array_utils.array_position(30, arr, start=3), 4)
        self.assertEqual(array_utils.array_position(50, arr), -1)

    def test_array_position_if(self):
        arr = [1, 3, 5, 8, 10]
        p = lambda x: x % 2 == 0
        self.assertEqual(array_utils.array_position_if(p, arr), 3)
        self.assertEqual(array_utils.array_position_if(p, arr, start=4), 4)

    def test_array_one_shorter(self):
        arr = [1, 2, 3, 4]
        np.testing.assert_array_equal(array_utils.array_one_shorter(arr, 1), [1, 3, 4])

    def test_determinant(self):
        mx = [[1, 2], [3, 4]]
        self.assertAlmostEqual(array_utils.determinant(mx), -2.0, places=9)

    def test_inverse(self):
        mx = [[1, 2], [3, 4]]
        inv = array_utils.inverse(mx)
        expected = [[-2.0, 1.0], [1.5, -0.5]]
        np.testing.assert_array_almost_equal(inv, expected, decimal=9)

    def test_det_and_inv(self):
        mx = [[1, 2], [3, 4]]
        det, inv = array_utils.det_and_inv(mx)
        self.assertAlmostEqual(det, -2.0, places=9)
        expected = [[-2.0, 1.0], [1.5, -0.5]]
        np.testing.assert_array_almost_equal(inv, expected, decimal=9)

    def test_cross_product_3d(self):
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        expected = [0, 0, 1]
        np.testing.assert_array_equal(array_utils.cross_product_3d(v1, v2), expected)

    def test_array_filter_empty_result(self):
        arr = [1, 3, 5, 7]
        p = lambda x: x % 2 == 0  # No even numbers
        result = array_utils.array_filter(p, arr)
        self.assertEqual(len(result), 0)

    def test_array_filter_all_match(self):
        arr = [2, 4, 6, 8]
        p = lambda x: x % 2 == 0  # All even
        result = array_utils.array_filter(p, arr)
        np.testing.assert_array_equal(result, arr)

    def test_array_position_not_found(self):
        arr = [1, 2, 3, 4]
        self.assertEqual(array_utils.array_position(10, arr), -1)

    def test_array_position_at_start_index(self):
        arr = [10, 20, 30, 40]
        self.assertEqual(array_utils.array_position(10, arr, start=0), 0)

    def test_array_position_if_not_found(self):
        arr = [1, 3, 5, 7]
        p = lambda x: x % 2 == 0  # No even numbers
        self.assertEqual(array_utils.array_position_if(p, arr), -1)

    def test_array_position_if_at_boundary(self):
        arr = [1, 2, 3, 4, 5]
        p = lambda x: x == 5
        self.assertEqual(array_utils.array_position_if(p, arr), 4)

    def test_array_one_shorter_first_element(self):
        arr = [1, 2, 3, 4]
        np.testing.assert_array_equal(array_utils.array_one_shorter(arr, 0), [2, 3, 4])

    def test_array_one_shorter_last_element(self):
        arr = [1, 2, 3, 4]
        np.testing.assert_array_equal(array_utils.array_one_shorter(arr, 3), [1, 2, 3])

    def test_determinant_singular_matrix(self):
        # Singular matrix (det = 0)
        mx = [[1, 2], [2, 4]]
        det = array_utils.determinant(mx)
        self.assertAlmostEqual(det, 0.0, places=9)

    def test_determinant_3x3(self):
        mx = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
        det = array_utils.determinant(mx)
        # Manual calculation: 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
        self.assertAlmostEqual(det, 1.0, places=9)

    def test_inverse_2x2(self):
        mx = [[4, 7], [2, 6]]
        inv = array_utils.inverse(mx)
        # Verify A * A^-1 = I
        identity = np.dot(mx, inv)
        expected = np.eye(2)
        np.testing.assert_array_almost_equal(identity, expected, decimal=9)

    def test_det_and_inv_consistency(self):
        mx = [[3, 1], [1, 2]]
        det, inv = array_utils.det_and_inv(mx)
        # Check determinant matches
        expected_det = array_utils.determinant(mx)
        self.assertAlmostEqual(det, expected_det, places=9)
        # Check inverse is correct
        identity = np.dot(mx, inv)
        np.testing.assert_array_almost_equal(identity, np.eye(2), decimal=9)

    def test_cross_product_3d_anticommutative(self):
        # v1 × v2 = -(v2 × v1)
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        cross_12 = array_utils.cross_product_3d(v1, v2)
        cross_21 = array_utils.cross_product_3d(v2, v1)
        np.testing.assert_array_almost_equal(cross_12, -cross_21, decimal=9)


if __name__ == "__main__":
    unittest.main()
