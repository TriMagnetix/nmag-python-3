import numpy as np
import time
import os
import re
import logging

log = logging.getLogger(__name__)

# --- Timing and Memory Utilities ---

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
    
    # Matches "VmSize: 1234 kB" or "VmRSS: 5678 KB"
    re_pattern = re.compile(r"^(VmSize|VmRSS):\s+(\d+)\s+[kK][bB]", re.MULTILINE)
    try:
        with open(self_status_file, 'r') as fd:
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
        pass
    return vmsize_vmrss

def time_vmem_rss():
    """Returns (elapsed_time, vmem, rss) where memory is in KB."""
    t = time_passed()
    mem = memstats()
    return t, mem[0], mem[1]

# --- Array and List Helpers (NumPy Vectorized) ---

def array_filter(p, arr):
    """Port of Snippets.array_filter using NumPy boolean indexing."""
    arr = np.asanyarray(arr)
    # Apply predicate to create a boolean mask
    mask = np.vectorize(p)(arr)
    return arr[mask]

def array_position(x, arr, start=0):
    """Port of Snippets.array_position."""
    arr = np.asanyarray(arr)
    sub_arr = arr[start:]
    indices = np.where(sub_arr == x)[0]
    if len(indices) > 0:
        return int(indices[0] + start)
    return -1

def array_position_if(p, arr, start=0):
    """Port of Snippets.array_position_if."""
    arr = np.asanyarray(arr)
    sub_arr = arr[start:]
    mask = np.vectorize(p)(sub_arr)
    indices = np.where(mask)[0]
    if len(indices) > 0:
        return int(indices[0] + start)
    return -1

def array_one_shorter(arr, pos):
    """Port of Snippets.array_one_shorter."""
    return np.delete(np.asanyarray(arr), pos)

# --- Numerical Helpers ---

def determinant(mx):
    """Port of Snippets.determinant using numpy.linalg.det."""
    return float(np.linalg.det(np.asanyarray(mx)))

def inverse(mx):
    """Port of Snippets.compute_inv_on_scratchpads using numpy.linalg.inv."""
    return np.linalg.inv(np.asanyarray(mx))

def det_and_inv(mx):
    """Port of Snippets.det_and_inv."""
    arr = np.asanyarray(mx)
    det = np.linalg.det(arr)
    inv = np.linalg.inv(arr)
    return float(det), inv

def cross_product_3d(v1, v2):
    """Port of Snippets.cross_product_3d."""
    return np.cross(np.asanyarray(v1), np.asanyarray(v2))
