from __future__ import annotations

import json
import subprocess
import sys


def _run_module_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_importing_nmag_does_not_import_numba():
    result = _run_module_probe(
        """
import json
import sys
import nmag

print(json.dumps({
    "has_simulation": hasattr(nmag, "Simulation"),
    "numba_loaded": "numba" in sys.modules,
}))
"""
    )

    assert result == {
        "has_simulation": True,
        "numba_loaded": False,
    }


def test_first_jitted_helper_call_compiles_numba_functions_lazily():
    result = _run_module_probe(
        """
import json
import sys

import numpy as np
import nmag.simulation as sim

before = {
    "numba_loaded": "numba" in sys.modules,
    "compiled": sim._JIT_COMPILED,
    "dot_type": type(sim._dot3).__name__,
}
value = sim._dot3(
    np.asarray([1.0, 2.0, 3.0], dtype=float),
    np.asarray([4.0, 5.0, 6.0], dtype=float),
)
after = {
    "numba_loaded": "numba" in sys.modules,
    "compiled": sim._JIT_COMPILED,
    "dot_type": type(sim._dot3).__name__,
}
print(json.dumps({
    "before": before,
    "after": after,
    "value": float(value),
}))
"""
    )

    assert result["before"] == {
        "numba_loaded": False,
        "compiled": False,
        "dot_type": "function",
    }
    assert result["after"] == {
        "numba_loaded": True,
        "compiled": True,
        "dot_type": "CPUDispatcher",
    }
    assert result["value"] == 32.0
