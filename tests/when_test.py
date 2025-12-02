import unittest
from when import at, every, never, TimeDict
from typing import Any

# --- Unit Tests (unittest.TestCase format) ---

class TestTimeSpec(unittest.TestCase):

    def setUp(self):
        """Provides a base time dictionary for each test."""
        self.time: TimeDict = {
            'stage': 0,
            'step': 0,
            'time': 0.0,
            'stage_step': 0,
            'stage_time': 0.0,
            'real_time': 0.0,
            'convergence': False
        }

    def test_at(self):
        w_step = at('step', 10)
        w_conv = at('convergence')

        self.time['step'] = 5
        self.assertFalse(w_step.match_time(self.time))
        self.assertEqual(w_step.next_time('step', self.time), 10)

        self.time['step'] = 10
        self.assertTrue(w_step.match_time(self.time))
        self.assertIs(w_step.next_time('step', self.time), False)

        self.time['step'] = 11
        self.assertFalse(w_step.match_time(self.time))
        self.assertIs(w_step.next_time('step', self.time), False)

        self.time['convergence'] = False
        self.assertFalse(w_conv.match_time(self.time))
        self.assertIs(w_conv.next_time('convergence', self.time), True)

        self.time['convergence'] = True
        self.assertTrue(w_conv.match_time(self.time))
        self.assertIs(w_conv.next_time('convergence', self.time), True)

    def test_every(self):
        w = every('step', 10)
        
        self.time['step'] = 0
        self.assertTrue(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 10)

        self.time['step'] = 5
        self.assertFalse(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 10)

        self.time['step'] = 10
        self.assertTrue(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 20)
        
        self.time['step'] = 11
        self.assertFalse(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 20)

    def test_every_first_last(self):
        w = every('step', 5, first=10, last=20)

        # Before 'first'
        self.time['step'] = 5
        self.assertFalse(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 10)

        # At 'first'
        self.time['step'] = 10
        self.assertTrue(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 15)

        # Between
        self.time['step'] = 12
        self.assertFalse(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 15)

        # At 'last'
        self.time['step'] = 20
        self.assertTrue(w.match_time(self.time))
        self.assertIs(w.next_time('step', self.time), False) # 'last' is inclusive for match, not for next

        # After 'last'
        self.time['step'] = 21
        self.assertFalse(w.match_time(self.time))
        self.assertIs(w.next_time('step', self.time), False)

    def test_every_no_delta(self):
        w = every('step', first=5, last=10)
        
        self.time['step'] = 4
        self.assertFalse(w.match_time(self.time))
        self.assertIs(w.next_time('step', self.time), True)
        
        self.time['step'] = 5
        self.assertTrue(w.match_time(self.time))
        self.assertIs(w.next_time('step', self.time), True)

        self.time['step'] = 10
        self.assertTrue(w.match_time(self.time))
        self.assertIs(w.next_time('step', self.time), False)

        self.time['step'] = 11
        self.assertFalse(w.match_time(self.time))
        self.assertIs(w.next_time('step', self.time), False)

    def test_every_validation(self):
        with self.assertRaisesRegex(ValueError, "delta must be positive"):
            every('step', 0)
        
        with self.assertRaisesRegex(ValueError, "delta must be positive"):
            every('step', -1)

        with self.assertRaisesRegex(ValueError, "'last' must be greater than 'first'"):
            every('step', 10, first=10, last=10)
            
        with self.assertRaisesRegex(ValueError, "must specify an identifier"):
            every(10, 20)

    def test_never(self):
        w = never
        self.time['step'] = 0
        self.assertFalse(w.match_time(self.time))
        self.assertIs(w.next_time('step', self.time), False)
        
        self.time['step'] = 100
        self.assertFalse(w.match_time(self.time))
        self.assertIs(w.next_time('step', self.time), False)

    def test_or(self):
        w = at('step', 5) | at('step', 10)
        
        self.time['step'] = 0
        self.assertFalse(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 5)

        self.time['step'] = 5
        self.assertTrue(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 10)

        self.time['step'] = 6
        self.assertFalse(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 10)

        self.time['step'] = 10
        self.assertTrue(w.match_time(self.time))
        self.assertIs(w.next_time('step', self.time), False)

    def test_and(self):
        # Matches on steps 10, 20, 30... AND 15, 30, 45...
        # Should only match on 30, 60, 90...
        w = every('step', 10) & every('step', 15, first=0)

        self.time['step'] = 0
        self.assertTrue(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 30)

        self.time['step'] = 10
        self.assertFalse(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 30)

        self.time['step'] = 15
        self.assertFalse(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 30)

        self.time['step'] = 30
        self.assertTrue(w.match_time(self.time))
        self.assertEqual(w.next_time('step', self.time), 60)

    def test_repr(self):
        w1 = at('convergence')
        self.assertEqual(str(w1), "at('convergence', True)")
        
        w2 = every('step', 10, first=1, last=100)
        self.assertEqual(str(w2), "every(10, 'step', first=1, last=100)")
        
        w3 = w1 | w2
        self.assertEqual(str(w3), "(at('convergence', True) | every(10, 'step', first=1, last=100))")
        
        w4 = w1 & w2
        self.assertEqual(str(w4), "(at('convergence', True) & every(10, 'step', first=1, last=100))")

        w5 = never
        self.assertEqual(str(w5), "never")

    def test_tols(self):
        w = every('step', 10)
        tols = {'step': 0.01}
        
        # Standard case, no tol
        self.time['step'] = 0
        self.assertEqual(w.next_time('step', self.time), 10)
        
        # Standard case, with tol
        self.assertEqual(w.next_time('step', self.time, tols), 10)
        
        # Current time is very close to next event.
        # Without tol, it should return 10.
        self.time['step'] = 9.999
        self.assertEqual(w.next_time('step', self.time), 10)
        
        # With tol, it should detect it's too close and skip to
        # the *next* event after that.
        # abs(10.0 - 9.999) = 0.001 < 0.01
        self.assertEqual(w.next_time('step', self.time, tols), 20.0)
        
        # Current time is slightly *after* an event
        self.time['step'] = 10.001
        self.assertEqual(w.next_time('step', self.time), 20.0)
        
        # With tol, result is the same
        self.assertEqual(w.next_time('step', self.time, tols), 20.0)

    # --- Integration Tests (from original __main__) ---

    def test_main_integration_loop(self):
        """
        Replicates the 'step' test from the original __main__ block.
        """
        w = every('step', 2, last=21) & every('step', 4, first=10) | at('step', 15)
        
        results = []
        this: Any = 0
        
        for _ in range(25): # Safety break
            self.time['step'] = this
            next_t = w.next_time('step', self.time)
            results.append((this, next_t))
            
            if next_t is False:
                break
            this = next_t
            
        expected_sequence = [
            (0, 10),
            (10, 14),
            (14, 15),
            (15, 18),
            (18, False)
        ]
        
        self.assertEqual(results, expected_sequence)

    def test_time_or_loop(self):
        """
        Replicates the 'time' test from the original __main__ block.
        Uses floats to check float logic.
        """
        w = every('time', 100.0) | every('time', 30.0)
        
        results = []
        this: Any = 0.0
        
        for _ in range(10):
            self.time['time'] = this
            next_t = w.next_time('time', self.time, tols={'time': 1e-9})
            results.append((this, next_t))
            
            if next_t is False:
                break
            this = next_t

        expected_sequence = [
            (0.0, 30.0),
            (30.0, 60.0),
            (60.0, 90.0),
            (90.0, 100.0),
            (100.0, 120.0),
            (120.0, 150.0),
            (150.0, 180.0),
            (180.0, 200.0),
            (200.0, 210.0),
            (210.0, 240.0)
        ]
        
        self.assertEqual(results, expected_sequence)

if __name__ == '__main__':
    unittest.main()
