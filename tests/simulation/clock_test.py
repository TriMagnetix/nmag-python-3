import unittest
from si.physical import SI

from simulation.clock import fmt_time, SimulationClock

class TestFmtTime(unittest.TestCase):

    def test_fmt_time_picoseconds(self):
        """Tests that times < 100 ps are formatted as picoseconds."""
        test_cases = [
            (0, "0.00 ps"),
            (50, "50.00 ps"),
            (99.9, "99.90 ps"),
        ]
        for time_val_ps, expected_str in test_cases:
            with self.subTest(time_val_ps=time_val_ps):
                t = SI(time_val_ps * 1e-12, "s")
                self.assertEqual(fmt_time(t), expected_str)

    def test_fmt_time_nanoseconds(self):
        """Tests that times >= 100 ps are formatted as nanoseconds."""
        test_cases = [
            (100, "0.10 ns"),
            (150, "0.15 ns"),
            (1234, "1.23 ns"),
        ]
        for time_val_ps, expected_str in test_cases:
            with self.subTest(time_val_ps=time_val_ps):
                t = SI(time_val_ps * 1e-12, "s")
                self.assertEqual(fmt_time(t), expected_str)

    def test_fmt_time_custom_format(self):
        """Tests the custom format specifier arguments."""
        t_ps = SI(50.123e-12, "s")
        t_ns = SI(150.123e-12, "s")
        
        # Test custom picosecond format specifier
        self.assertEqual(fmt_time(t_ps, fmt_ps=".1f"), "50.1 ps")
        
        # Test custom nanosecond format specifier
        self.assertEqual(fmt_time(t_ns, fmt_ns=".1f"), "0.2 ns")

class TestSimulationClock(unittest.TestCase):

    def setUp(self):
        """Provides a default, fresh SimulationClock instance for each test."""
        self.clock = SimulationClock()

    def test_clock_init_defaults(self):
        """Test that the dataclass initializes with all default values."""
        self.assertEqual(self.clock.id, -1)
        self.assertEqual(self.clock.stage, 1)
        self.assertEqual(self.clock.step, 0)
        self.assertEqual(self.clock.time, SI(0.0, "s"))
        self.assertFalse(self.clock.convergence)

    def test_clock_init_custom(self):
        """Test that the dataclass __init__ correctly overrides defaults."""
        custom_time = SI(1e-9, "s")
        c = SimulationClock(
            step=100, 
            stage=5, 
            time=custom_time, 
            convergence=True
        )
        
        self.assertEqual(c.step, 100)
        self.assertEqual(c.stage, 5)
        self.assertEqual(c.time, custom_time)
        self.assertTrue(c.convergence)
        
        # Check that others are still default
        self.assertEqual(c.id, -1)
        self.assertEqual(c.stage_step, 0)

    def test_clock_init_invalid_arg(self):
        """Test that the dataclass __init__ raises a TypeError for an unknown argument."""
        with self.assertRaises(TypeError) as cm:
            SimulationClock(invalid_key="foo")
        
        # The error message from dataclass is slightly different
        self.assertIn("unexpected keyword argument", str(cm.exception))
        self.assertIn("'invalid_key'", str(cm.exception))


    def test_inc_stage_default(self):
        """Test 'inc_stage()' with no arguments (default increment)."""
        # Set a non-default state
        self.clock.step = 50
        self.clock.time = SI(10e-9, "s")
        self.clock.stage = 3
        self.clock.stage_step = 20
        self.clock.convergence = True
        
        self.clock.inc_stage()
        
        self.assertEqual(self.clock.stage, 4)
        self.assertEqual(self.clock.stage_step, 0)
        self.assertEqual(self.clock.stage_time, SI(0.0, "s"))
        self.assertFalse(self.clock.convergence)
        self.assertEqual(self.clock.zero_stage_step, 50)
        self.assertEqual(self.clock.zero_stage_time, SI(10e-9, "s"))

    def test_inc_stage_specific(self):
        """Test 'inc_stage()' when setting a specific stage number."""
        self.clock.step = 50
        self.clock.time = SI(10e-9, "s")
        self.clock.stage = 3
        
        self.clock.inc_stage(stage=10)
        
        self.assertEqual(self.clock.stage, 10)
        self.assertEqual(self.clock.stage_step, 0)
        self.assertFalse(self.clock.convergence)
        self.assertEqual(self.clock.zero_stage_step, 50)
        self.assertEqual(self.clock.zero_stage_time, SI(10e-9, "s"))

    def test_repr(self):
        """Test the dataclass-generated __repr__ magic method."""
        r = repr(self.clock)
        self.assertTrue(r.startswith("SimulationClock("))
        self.assertTrue(r.endswith(")"))
        self.assertIn("step=0", r)
        self.assertIn("stage=1", r)
        self.assertIn("time=Physical(0.0, 'second')", r)

    def test_str(self):
        """Test the __str__ magic method (using tabulate)."""
        s = str(self.clock)
        
        # Check for tabulate "pipe" format
        self.assertIn("|", s)
        
        # Check for header/footer
        self.assertTrue(s.startswith("="))
        self.assertTrue(s.endswith("="))
        
        # Check for content (note: no spaces around '=')
        self.assertIn("ID=-1", s)
        self.assertIn("Step=0", s)
        self.assertIn("Stage=1", s)
        
        # Check that it still calls fmt_time
        self.assertIn("Time=0.00 ps", s)
        
        # Check that old formatting is gone (no right-justified keys)
        self.assertNotIn("           ID=", s)

    def test_str_with_nanoseconds(self):
        """Test that __str__ (tabulate) correctly formats nanoseconds."""
        self.clock.time = SI(120e-12, "s")
        s = str(self.clock)
        
        # Check for tabulate format
        self.assertIn("|", s)
        
        # Check for the formatted nanosecond string
        self.assertIn("0.12 ns", s)
        self.assertIn("Time=0.12 ns", s)


if __name__ == '__main__':
    unittest.main()
