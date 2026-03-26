import logging
import os
import re
import time

log = logging.getLogger(__name__)

_time_zero = None

def time_passed():
    """Returns elapsed time since the first call to this function."""
    global _time_zero
    if _time_zero is None:
        _time_zero = time.time()
        return 0.0
    return time.time() - _time_zero

def memstats(self_status_file="/proc/self/status"):
    """Reads VmSize and VmRSS from /proc/self/status (Linux)."""
    vmsize_vmrss = [0.0, 0.0]
    if not os.path.exists(self_status_file):
        return vmsize_vmrss

    # Matches "VmSize: 1234 kB" or "VmRSS: 5678 KB".
    re_pattern = re.compile(r"^(VmSize|VmRSS):\s+(\d+)\s+[kK][bB]", re.MULTILINE)
    try:
        with open(self_status_file, "r") as fd:
            content = fd.read()
            found = 0
            for match in re_pattern.finditer(content):
                key = match.group(1)
                value = float(match.group(2))
                if key == "VmSize":
                    vmsize_vmrss[0] = value
                else:
                    vmsize_vmrss[1] = value
                found += 1
                if found >= 2:
                    break
    except Exception as e:
        log.debug(f"Failed to read memstats: {e}")
    return vmsize_vmrss

def time_vmem_rss():
    """Returns (elapsed_time, vmem, rss) where memory is in KB."""
    t = time_passed()
    mem = memstats()
    return t, mem[0], mem[1]
