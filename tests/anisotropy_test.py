import unittest
import numpy as np

# Import the classes and functions to be tested
from anisotropy import (
    PredefinedAnisotropy,
    uniaxial_anisotropy,
    cubic_anisotropy,
    want_anisotropy,
)
# Import the hidden helper function for normalization directly since it isn't exposed in the init file
from anisotropy.anisotropy import _normalize

class TestAnisotropy(unittest.TestCase):
    """Unit tests for the anisotropy definitions and helper functions."""

    ## Test Helper Functions
    # --------------------------------------------------------------------------
    
    def testnormalize(self):
        """Tests the vector normalization helper function."""
        v = [1, 2, 3]
        norm_v = _normalize(v)
        # Check that the norm is 1
        self.assertAlmostEqual(np.linalg.norm(norm_v), 1.0)
        # Check that the direction is correct
        np.testing.assert_array_almost_equal(norm_v, np.array(v) / np.linalg.norm(v))
        # Check that a zero vector raises a ValueError
        with self.assertRaises(ValueError):
            _normalize([0, 0, 0])

    def test_want_anisotropy(self):
        """Tests the anisotropy object validation function."""
        a_with_func = PredefinedAnisotropy(function=lambda m: 0, order=2)
        a_without_func = PredefinedAnisotropy(order=2)

        # These should pass without error
        want_anisotropy(a_with_func, want_function=True)
        want_anisotropy(a_without_func, want_function=False)

        # Test for TypeError with incorrect object type
        with self.assertRaises(TypeError):
            want_anisotropy("not an anisotropy object")

        # Test for ValueError when a function is required but missing
        with self.assertRaises(ValueError):
            want_anisotropy(a_without_func, want_function=True)

    ## Test PredefinedAnisotropy Class
    # --------------------------------------------------------------------------
    
    def test_predefined_anisotropy_init(self):
        """Tests the constructor of the PredefinedAnisotropy class."""
        # Test successful creation with a function and order
        a = PredefinedAnisotropy(function=lambda m: 1, order=2)
        self.assertTrue(a.has_function())
        self.assertEqual(a.order, 2)

        # Test successful creation with only an order
        b = PredefinedAnisotropy(order=4)
        self.assertFalse(b.has_function())
        self.assertEqual(b.order, 4)

        # Test that initialization fails if neither 'function' nor 'order' is given
        with self.assertRaises(ValueError):
            PredefinedAnisotropy()

        # Test that axes are correctly converted to NumPy arrays
        c = PredefinedAnisotropy(order=2, axis1=[1, 0, 0])
        self.assertIsInstance(c.axis1, np.ndarray)

    def test_predefined_anisotropy_str_repr(self):
        """Tests the string representations (__str__ and __repr__) of the class."""
        # Test without a custom stringifier
        a = PredefinedAnisotropy(anis_type="test", order=2)
        self.assertEqual(str(a), "<PredefinedAnisotropy:test>")
        self.assertEqual(repr(a), 'PredefinedAnisotropy(anis_type="test", ?)')

        # Test with a custom stringifier
        stringifier = lambda x: f"order={x.order}"
        b = PredefinedAnisotropy(anis_type="test", order=2, stringifier=stringifier)
        self.assertEqual(str(b), "<PredefinedAnisotropy:test, order=2>")
        self.assertEqual(repr(b), 'PredefinedAnisotropy(anis_type="test", order=2)')

    def test_predefined_anisotropy_operators(self):
        """Tests the overloaded operators (+, -, neg, pos)."""
        a1 = PredefinedAnisotropy(function=lambda m: np.dot(m, m), order=2)
        a2 = PredefinedAnisotropy(function=lambda m: 2 * np.dot(m, m), order=4)
        m_vec = np.array([0.5, 0.5, 0.5])

        # Test unary positive operator (+)
        pos_a1 = +a1
        self.assertIs(pos_a1, a1) # Should return the same object

        # Test unary negative operator (-)
        neg_a1 = -a1
        self.assertIsNot(neg_a1, a1) # Should return a new object
        self.assertEqual(neg_a1.function(m_vec), -a1.function(m_vec))
        self.assertEqual(neg_a1.order, a1.order)

        # Test addition (+)
        add_res = a1 + a2
        self.assertEqual(add_res.order, 4)  # Order should be max(2, 4)
        self.assertAlmostEqual(add_res.function(m_vec), 3 * np.dot(m_vec, m_vec))

        # Test subtraction (-)
        sub_res = a2 - a1
        self.assertEqual(sub_res.order, 4)  # Order should be max(4, 2)
        self.assertAlmostEqual(sub_res.function(m_vec), np.dot(m_vec, m_vec))
        
        # Test that operations with invalid types raise TypeError
        with self.assertRaises(TypeError):
            _ = a1 + 5
        with self.assertRaises(TypeError):
            _ = a1 - "some_string"

    ## Test Factory Functions
    # --------------------------------------------------------------------------

    def test_uniaxial_anisotropy(self):
        """Tests the uniaxial_anisotropy factory function."""
        axis = [0, 0, 1]
        K1 = 100.0
        K2 = 10.0
        anis = uniaxial_anisotropy(axis, K1, K2)

        self.assertEqual(anis.anis_type, "uniaxial")
        self.assertEqual(anis.order, 4)
        self.assertEqual(anis.K1, K1)
        self.assertEqual(anis.K2, K2)
        np.testing.assert_array_almost_equal(anis.axis1, [0, 0, 1])

        # Test energy when magnetization is parallel to the axis
        m_parallel = [0, 0, 1]
        expected_energy_parallel = -K1 * (1)**2 - K2 * (1)**4
        self.assertAlmostEqual(anis.function(m_parallel), expected_energy_parallel)

        # Test energy when magnetization is perpendicular to the axis
        m_perp = [1, 0, 0]
        self.assertAlmostEqual(anis.function(m_perp), 0.0)

        # Test case with only K1
        anis_k1 = uniaxial_anisotropy(axis, K1)
        self.assertEqual(anis_k1.order, 2)
        self.assertEqual(anis_k1.K2, 0)
        self.assertAlmostEqual(anis_k1.function(m_parallel), -K1)

        # Test that the input axis is correctly normalized
        anis_unnormalized = uniaxial_anisotropy([0, 5, 0], K1)
        np.testing.assert_array_almost_equal(anis_unnormalized.axis1, [0, 1, 0])

    def test_cubic_anisotropy(self):
        """Tests the cubic_anisotropy factory function."""
        ax1, ax2 = [1, 0, 0], [0, 1, 0]
        K1, K2, K3 = 100.0, 10.0, 1.0
        anis = cubic_anisotropy(ax1, ax2, K1, K2, K3)

        self.assertEqual(anis.anis_type, "cubic")
        self.assertEqual(anis.order, 8)
        np.testing.assert_array_almost_equal(anis.axis1, [1, 0, 0])
        np.testing.assert_array_almost_equal(anis.axis2, [0, 1, 0])
        np.testing.assert_array_almost_equal(anis.axis3, [0, 0, 1])

        # Test energy when magnetization is along a primary axis (should be 0)
        m_axis = [0, 1, 0]
        self.assertAlmostEqual(anis.function(m_axis), 0.0)

        # Test energy when magnetization is along a face diagonal [110]
        m_face_diag = _normalize([1, 1, 0])  # [1/sqrt(2), 1/sqrt(2), 0]
        # Expected energy: K1 * (a1^2*a2^2) + K3 * (a1^2*a2^2)^2
        # E = K1 * (0.5*0.5) + K3 * (0.5*0.5)^2 = K1/4 + K3/16
        self.assertAlmostEqual(anis.function(m_face_diag), K1/4 + K3/16)
        
        # Test energy when magnetization is along a space diagonal [111]
        m_space_diag = _normalize([1, 1, 1])
        # a1=a2=a3 = 1/sqrt(3), so a_i^2 = 1/3
        # E = K1*(3*(1/3*1/3)) + K2*(1/3*1/3*1/3) + K3*(3*(1/3*1/3)^2)
        # E = K1/3 + K2/27 + K3/27
        self.assertAlmostEqual(anis.function(m_space_diag), K1/3 + K2/27 + K3/27)

        # Test automatic order detection
        self.assertEqual(cubic_anisotropy(ax1, ax2, K1, K2, 0).order, 6)
        self.assertEqual(cubic_anisotropy(ax1, ax2, K1, 0, 0).order, 4)

        # Test orthonormalization of input axes
        anis_ortho = cubic_anisotropy([1, 1, 0], [0, 1, 1], K1)
        # Check that the resulting axes form an orthonormal basis
        self.assertAlmostEqual(np.dot(anis_ortho.axis1, anis_ortho.axis2), 0)
        self.assertAlmostEqual(np.dot(anis_ortho.axis1, anis_ortho.axis3), 0)
        self.assertAlmostEqual(np.dot(anis_ortho.axis2, anis_ortho.axis3), 0)

if __name__ == '__main__':
    unittest.main()
