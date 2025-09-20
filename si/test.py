"""
Unit tests for the Physical (SI) class using the pint library.
"""

import unittest
import pint
from physical import SI

class TestHardcodedUnits(unittest.TestCase):
  """
  Tests that the new Physical (SI) class, backed by pint, correctly
  interprets all the unit strings that were hard-coded in the original
  legacy class.

  Each test compares a quantity created with a derived unit string (e.g., "J")
  to an equivalent quantity constructed from base SI units. This verifies
  that the underlying pint parser recognizes the units and their correct
  physical definitions.
  """

  def test_base_units(self):
    """Test that the seven base SI units are parsed correctly."""
    self.assertEqual(SI(1, "m"), SI(1, "meter"))
    self.assertEqual(SI(1, "kg"), SI(1, "kilogram"))
    self.assertEqual(SI(1, "s"), SI(1, "second"))
    self.assertEqual(SI(1, "A"), SI(1, "ampere"))
    self.assertEqual(SI(1, "K"), SI(1, "kelvin"))
    self.assertEqual(SI(1, "mol"), SI(1, "mole"))
    self.assertEqual(SI(1, "cd"), SI(1, "candela"))

  def test_joule(self):
    """Test Joule (J) against its definition: kg * m^2 / s^2"""
    joule_from_string = SI(1, 'J')
    joule_from_base = SI(1, 'kg * m**2 / s**2')
    self.assertEqual(joule_from_string, joule_from_base)

  def test_newton(self):
    """Test Newton (N) against its definition: kg * m / s^2"""
    newton_from_string = SI(1, 'N')
    newton_from_base = SI(1, 'kg * m / s**2')
    self.assertEqual(newton_from_string, newton_from_base)

  def test_watt(self):
    """Test Watt (W) against its definition: J/s or kg * m^2 / s^3"""
    watt_from_string = SI(1, 'W')
    watt_from_base = SI(1, 'kg * m**2 / s**3')
    self.assertEqual(watt_from_string, watt_from_base)

  def test_tesla(self):
    """Test Tesla (T) against its definition: kg / (s^2 * A)"""
    tesla_from_string = SI(1, 'T')
    tesla_from_base = SI(1, 'kg / (s**2 * A)')
    self.assertEqual(tesla_from_string, tesla_from_base)

  def test_volt(self):
    """Test Volt (V) against its definition: W/A or kg * m^2 / (s^3 * A)"""
    volt_from_string = SI(1, 'V')
    volt_from_base = SI(1, 'kg * m**2 / (s**3 * A)')
    self.assertEqual(volt_from_string, volt_from_base)

  def test_coulomb(self):
    """Test Coulomb (C) against its definition: s * A"""
    coulomb_from_string = SI(1, 'C')
    coulomb_from_base = SI(1, 's * A')
    self.assertEqual(coulomb_from_string, coulomb_from_base)

  def test_ohm(self):
    """Test Ohm against its definition: V/A or kg * m^2 / (s^3 * A^2)"""
    # pint expects lowercase 'ohm'
    ohm_from_string = SI(1, 'ohm')
    ohm_from_base = SI(1, 'kg * m**2 / (s**3 * A**2)')
    self.assertEqual(ohm_from_string, ohm_from_base)

  def test_henry(self):
    """Test Henry (H) against its definition: V*s/A or kg * m^2 / (s^2 * A^2)"""
    henry_from_string = SI(1, 'H')
    henry_from_base = SI(1, 'kg * m**2 / (s**2 * A**2)')
    self.assertEqual(henry_from_string, henry_from_base)

class TestClassFunctionality(unittest.TestCase):
  """
  Tests the core functionality of the Physical (SI) class, including
  initialization methods, string representations, and unit conversions.
  """

  def test_initialization_methods(self):
    """Verify that the SI class can be initialized in various ways."""
    # Standard initialization with a string
    s1 = SI(10, "m/s")
    self.assertEqual(s1.magnitude, 10)
    self.assertEqual(str(s1._quantity.units), "meter / second")

    # Legacy list-based initialization
    s2 = SI(10, ['m', 1, 's', -1])
    self.assertEqual(s1, s2)

    # Shorthand initialization (value is 1.0)
    s3 = SI("m/s")
    self.assertEqual(s3.magnitude, 1.0)
    self.assertEqual(s3, SI(1.0, "m/s"))

    # Dimensionless initialization
    s4 = SI(5.0)
    s5 = SI(5.0, [])
    self.assertTrue(s4._quantity.dimensionless)
    self.assertEqual(s4, s5)
    
    # --- Edge Cases for constructor ---
    # Test that an invalid list (uneven items) raises a ValueError
    with self.assertRaises(ValueError):
      SI(10, ['m', 1, 's'])

    # Test that a non-numeric power in the list raises an error
    with self.assertRaises((ValueError, TypeError)):
      SI(1, ['m', 'one'])

  def test_dens_str(self):
    """Test the dense string representation method."""
    # Test a standard case
    q1 = SI(15.5, "A/m**2")
    self.assertEqual(q1.dens_str(), "<15.5A/m^2>")

    # Test scientific notation
    q2 = SI(1.23e-7, "N*m")
    self.assertEqual(q2.dens_str(), "<1.23e-07m*N>")

    # Test dimensionless quantity of 1
    q3 = SI(1.0)
    self.assertEqual(q3.dens_str(), "<1>")

    # Test quantity of 1 with units
    q4 = SI(1.0, "m")
    self.assertEqual(q4.dens_str(), "<m>")

  def test_in_units_of(self):
    """Test the unit conversion method and its edge cases."""
    # Standard conversion
    velocity_ms = SI(20, "m/s")
    km_per_hour_unit = SI(1, "km/h")
    velocity_kmh = velocity_ms.in_units_of(km_per_hour_unit)
    self.assertAlmostEqual(velocity_kmh, 72.0)

    distance_km = SI(2.5, "km")
    meter_unit = SI(1, "m")
    distance_m = distance_km.in_units_of(meter_unit)
    self.assertAlmostEqual(distance_m, 2500.0)

    # --- Edge Cases for in_units_of ---
    # Test that incompatible units raise a DimensionalityError
    with self.assertRaises(pint.errors.DimensionalityError):
      velocity_ms.in_units_of(meter_unit)

    # Test that a non-Physical object raises a TypeError
    with self.assertRaises(TypeError):
      velocity_ms.in_units_of("km/h")
      
    # Test division by a zero-magnitude unit
    quantity = SI(10, 'm')
    zero_unit = SI(0, 'm')
    with self.assertRaises(ZeroDivisionError):
      quantity.in_units_of(zero_unit)

    # Test conversion between dimensionless quantities
    dimless_q1 = SI(100)
    dimless_q2 = SI(2)
    self.assertAlmostEqual(dimless_q1.in_units_of(dimless_q2), 50.0)

  def test_float_conversion(self):
    """Test conversion to float for dimensionless quantities."""
    # A dimensionless quantity should be convertible to float
    dimless_quantity = SI(10, 'm') / SI(2, 'm')
    self.assertAlmostEqual(float(dimless_quantity), 5.0)

    # A quantity with dimensions should raise a DimensionalityError
    dim_quantity = SI(10, 'm')
    with self.assertRaises(pint.errors.DimensionalityError):
      float(dim_quantity)

  def test_zero_operations(self):
    """Test comparisons and arithmetic with zero."""
    # Test comparison with a raw number zero
    self.assertEqual(SI(0, 'm'), 0)
    self.assertEqual(SI(0, 'kg*m/s'), 0)
    self.assertNotEqual(SI(5, 'm'), 0)

    # Test addition with a raw number zero
    q = SI(5, 'm')
    self.assertEqual(q + 0, q)
    self.assertEqual(0 + q, q)

# This allows the test to be run from the command line
if __name__ == '__main__':
  unittest.main()

