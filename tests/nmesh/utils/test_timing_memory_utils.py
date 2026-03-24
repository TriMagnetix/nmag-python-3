import os
import tempfile
import unittest

import nmesh.utils.timing_memory_utils as timing_utils


class TestTimingMemoryUtils(unittest.TestCase):
    def setUp(self):
        timing_utils._time_zero = None

    def test_time_vmem_rss(self):
        t, vmem, rss = timing_utils.time_vmem_rss()
        self.assertGreaterEqual(t, 0.0)
        # On non-Linux this might be 0.0, but if /proc/self/status exists it should be > 0.
        if os.path.exists("/proc/self/status"):
            self.assertGreater(vmem, 0.0)
            self.assertGreater(rss, 0.0)

    def test_memstats_from_file(self):
        content = "Name:\tpython\nVmSize:\t   1234 kB\nVmRSS:\t   456 kB\n"
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            self.assertEqual(timing_utils.memstats(tmp_path), [1234.0, 456.0])
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
