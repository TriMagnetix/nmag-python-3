import unittest
from unittest.mock import patch
from throttler import Throttler

class TestThrottler(unittest.TestCase):
    """
    Tests the Throttler class.
    
    We use @patch to mock 'time.monotonic' to give us full
    control over the "current time" during tests.
    """

    def setUp(self):
        """
        This method is run before each test.
        It ensures each test gets a fresh, isolated Throttler instance.
        """
        self.throttler = Throttler()
        self.key = "test_key_1"
        self.delay = 5.0  # 5-second delay for testing

    @patch('time.monotonic')
    def test_allows_first_call(self, mock_monotonic):
        """Tests that the very first call for a key is always allowed."""
        mock_monotonic.return_value = 1000.0
        
        is_allowed = self.throttler.is_allowed(self.key, self.delay)
        
        self.assertTrue(is_allowed, "First call should always be allowed")
        self.assertEqual(self.throttler.last_called[self.key], 1000.0)

    @patch('time.monotonic')
    def test_throttles_immediate_second_call(self, mock_monotonic):
        """Tests that a second call, before the delay expires, is throttled."""
        
        # First call
        mock_monotonic.return_value = 1000.0
        self.throttler.is_allowed(self.key, self.delay) 
        
        # Second call, 2 seconds later (delay is 5.0)
        mock_monotonic.return_value = 1002.0
        is_allowed = self.throttler.is_allowed(self.key, self.delay)
        
        self.assertFalse(is_allowed, "Second call before delay should be throttled")
        self.assertEqual(self.throttler.last_called[self.key], 1000.0)

    @patch('time.monotonic')
    def test_allows_call_after_delay_expires(self, mock_monotonic):
        """Tests that a call is allowed again after the delay has passed."""
        
        mock_monotonic.return_value = 1000.0
        self.throttler.is_allowed(self.key, self.delay) # Allowed

        # Call just *before* the delay expires (t=1004.99)
        mock_monotonic.return_value = 1004.99
        self.assertFalse(self.throttler.is_allowed(self.key, self.delay), "Call just before expiry should fail")
        
        # Call *exactly* when the delay expires (t=1005.0)
        mock_monotonic.return_value = 1005.0
        is_allowed = self.throttler.is_allowed(self.key, self.delay)
        
        self.assertTrue(is_allowed, "Call exactly at expiry time should be allowed")
        self.assertEqual(self.throttler.last_called[self.key], 1005.0)

    @patch('time.monotonic')
    def test_keys_are_isolated(self, mock_monotonic):
        """Tests that two different keys have independent timers."""
        key_a = "key_A"
        key_b = "key_B"
        delay = 10.0

        mock_monotonic.return_value = 1000.0
        self.assertTrue(self.throttler.is_allowed(key_a, delay), "First call for Key A should be allowed")
        
        mock_monotonic.return_value = 1001.0
        self.assertTrue(self.throttler.is_allowed(key_b, delay), "First call for Key B should be allowed")
        
        mock_monotonic.return_value = 1002.0
        self.assertFalse(self.throttler.is_allowed(key_a, delay), "Second call for Key A should be throttled")

        self.assertEqual(self.throttler.last_called[key_a], 1000.0)
        self.assertEqual(self.throttler.last_called[key_b], 1001.0)

# This allows running the tests directly from the command line
if __name__ == '__main__':
    unittest.main()
