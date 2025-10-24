import unittest
from dataclasses import FrozenInstanceError
from si.physical import SI

from simulation.quantity import Quantity

class TestQuantity(unittest.TestCase):

    def setUp(self):
        """Set up some common SI units for testing."""
        self.unit_A_m = SI("A/m")
        self.unit_J = SI("J")

    def test_quantity_init(self):
        """Test basic initialization with all arguments."""
        q = Quantity(
            name="H_demag",
            type="field",
            units=self.unit_A_m,
            signature="H_demag_*",
            context="demag"
        )
        self.assertEqual(q.name, "H_demag")
        self.assertEqual(q.type, "field")
        self.assertEqual(q.units, self.unit_A_m)
        self.assertEqual(q.signature, "H_demag_*")
        self.assertEqual(q.context, "demag")
        self.assertIsNone(q.parent)

    def test_post_init_signature_defaults_to_name(self):
        """
        Test __post_init__: signature should default to 'name' if None.
        """
        # Initialize with signature=None (the default)
        q = Quantity(name="E_total", type="field", units=self.unit_J)
        
        # Check that __post_init__ set the signature
        self.assertEqual(q.name, "E_total")
        self.assertEqual(q.signature, "E_total")

    def test_post_init_signature_is_retained(self):
        """
        Test __post_init__: signature should be retained if provided.
        """
        q = Quantity(
            name="m",
            type="pfield",
            units=SI(1),
            signature="_?_*"
        )
        self.assertEqual(q.name, "m")
        self.assertEqual(q.signature, "_?_*") # Should not be 'm'

    def test_sub_quantity(self):
        """Test the sub_quantity method."""
        parent_q = Quantity(
            name="m",
            type="pfield",
            units=SI(1),
            signature="_?_*",
            context="main"
        )
        
        child_q = parent_q.sub_quantity(name="m_Py_0")
        
        # Test child's new name
        self.assertEqual(child_q.name, "m_Py_0")
        
        # Test that child is linked to parent
        self.assertEqual(child_q.parent, parent_q)
        
        # Test that other properties were inherited
        self.assertEqual(child_q.type, parent_q.type)
        self.assertEqual(child_q.units, parent_q.units)
        self.assertEqual(child_q.context, parent_q.context)
        
        # Test that the *processed* signature was inherited
        self.assertEqual(child_q.signature, parent_q.signature)

if __name__ == '__main__':
    unittest.main()
