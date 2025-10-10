import unittest

# Import from the actual library files
from si import constants
from si.physical import SI
from mag_material.mag_material import MagMaterial
from anisotropy.anisotropy import PredefinedAnisotropy, uniaxial_anisotropy

class TestMagMaterial(unittest.TestCase):
    """Unit tests for the MagMaterial class using a real SI library."""

    def test_successful_initialization_defaults(self):
        """Test creating a material with a name and default parameters."""
        mat = MagMaterial(name='Permalloy')
        self.assertEqual(mat.name, 'Permalloy')
        self.assertAlmostEqual(mat.Ms.value, 0.86e6)
        self.assertEqual(mat.llg_damping, 0.5)
        self.assertIsNone(mat.anisotropy)
        self.assertTrue(mat.do_precession)

    def test_successful_initialization_custom(self):
        """Test creating a material with custom-defined parameters."""
        ms_val = SI(1.0e6, constants.Ampere / constants.meter)
        a_val = SI(1.0e-11, constants.Joule / constants.meter)
        anis = uniaxial_anisotropy(axis=[0, 0, 1], K1=1e5)

        mat = MagMaterial(name='Iron',
                          Ms=ms_val,
                          exchange_coupling=a_val,
                          llg_damping=0.01,
                          anisotropy=anis,
                          do_precession=False)

        self.assertEqual(mat.name, 'Iron')
        self.assertEqual(mat.Ms, ms_val)
        self.assertEqual(mat.exchange_coupling, a_val)
        self.assertEqual(mat.llg_damping, 0.01)
        self.assertIs(mat.anisotropy, anis)
        self.assertEqual(mat.anisotropy_order, 2)
        self.assertFalse(mat.do_precession)

    def test_anisotropy_validation(self):
        """Test the validation logic for anisotropy arguments."""
        # Case 1: Predefined anisotropy with a conflicting custom order should fail.
        anis = PredefinedAnisotropy(order=4)
        with self.assertRaisesRegex(ValueError, "Cannot specify custom 'anisotropy_order'"):
            MagMaterial(name='ErrMat1', anisotropy=anis, anisotropy_order=2)

        # Case 2: Custom anisotropy function without specifying an order should fail.
        custom_func = lambda m: m[2]**2
        with self.assertRaisesRegex(ValueError, "must specify 'anisotropy_order'"):
            MagMaterial(name='ErrMat2', anisotropy=custom_func)

        # Case 3: Successful creation with a custom function and specified order.
        mat = MagMaterial(name='GoodMat', anisotropy=custom_func, anisotropy_order=2)
        self.assertEqual(mat.anisotropy, custom_func)
        self.assertEqual(mat.anisotropy_order, 2)

    def test_parameter_validation(self):
        """Test validation of physical parameter units and values."""
        # Case 1: Providing a parameter with incorrect units should raise a TypeError.
        with self.assertRaisesRegex(TypeError, "requires units compatible with"):
            MagMaterial(name='UnitError', Ms=SI(constants.Joule / constants.meter))

        # Case 2: Providing a negative exchange coupling should raise a ValueError.
        # with self.assertRaisesRegex(ValueError, "exchange coupling constant must be positive"):
        #     MagMaterial(name='ValueError', exchange_coupling=SI(-1e-12, constants.Joule / constants.meter))

    def test_calculated_coefficients(self):
        """Verify the correctness of internally calculated 'su_' coefficients."""
        # Use simple values to make manual calculation easy
        mat = MagMaterial(name='TestCalc',
                          Ms=SI(constants.Ampere / constants.meter),
                          exchange_coupling=SI(constants.Joule / constants.meter),
                          llg_gamma_G=SI(constants.meter / (constants.Ampere * constants.second)),
                          llg_damping=0.5,
                          do_precession=True)
        print(mat.su_llg_coeff1, mat.su_llg_coeff2, mat.su_exch_prefactor)
        # Manually calculate expected values
        gilbert_to_ll = 1.0 / (1.0 + 0.5**2)
        expected_coeff1 = -1.0 * gilbert_to_ll
        expected_coeff2 = expected_coeff1 * 0.5
        expected_exch = -2.0 * 1.0 / (constants.mu0 * 1.0)

        self.assertAlmostEqual(mat.su_llg_coeff1.value, expected_coeff1)
        self.assertAlmostEqual(mat.su_llg_coeff2.value, expected_coeff2)
        self.assertAlmostEqual(mat.su_exch_prefactor, expected_exch)


    def test_no_precession_flag(self):
        """Test that `do_precession=False` correctly nullifies the precession term."""
        mat = MagMaterial(name='NoPrecession', do_precession=False)
        self.assertEqual(mat.su_llg_coeff1, 0.0)
        self.assertNotEqual(mat.su_llg_coeff2, 0.0)

    def test_str_representation(self):
        """Test the __str__ method for standard and extended printing."""
        mat = MagMaterial(name='StringTest')
        s_std = str(mat)
        self.assertIn("Material 'StringTest'", s_std)
        self.assertIn("exchange_coupling", s_std)
        self.assertNotIn("su_llg_coeff1", s_std)

        mat.extended_print = True
        s_ext = str(mat)
        self.assertIn("su_llg_coeff1", s_ext)
        self.assertGreater(len(s_ext.splitlines()), len(s_std.splitlines()))

if __name__ == '__main__':
    unittest.main()
