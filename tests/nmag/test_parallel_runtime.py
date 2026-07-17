from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import nmag


def test_parallel_runtime_reports_unaccelerated_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nmag.parallel._rust_accelerator_available", lambda: False)

    info = nmag.parallel_runtime_info()

    assert info.logical_cpus >= 1
    assert info.rust_available is False
    assert info.rust_worker_threads is None
    assert info.rust_parallel_min_items is None


@pytest.mark.parametrize("thread_count", [1, 2])
def test_rayon_thread_count_is_selected_at_process_start(thread_count: int) -> None:
    pytest.importorskip("nmag_accel")
    script = """
import json
import nmag
info = nmag.parallel_runtime_info()
print(json.dumps({"threads": info.rust_worker_threads, "minimum": info.rust_parallel_min_items}))
"""
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(thread_count)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout)

    assert result == {"threads": thread_count, "minimum": 2048}


def test_serial_and_parallel_llg_results_are_bitwise_identical() -> None:
    pytest.importorskip("nmag_accel")
    script = """
import hashlib
import numpy as np
import nmag_accel
n = 4096
index = np.arange(n, dtype=float)
m = np.column_stack((np.ones(n), np.sin(index * 0.01), np.cos(index * 0.01)))
m /= np.linalg.norm(m, axis=1)[:, None]
h = np.column_stack((index, 2.0 * index, -0.5 * index))
result = nmag_accel.llg_rhs(m, h, np.ones(n), np.ones(n), -2.21e5, -1.1e5, 1.0e11)
print(hashlib.sha256(np.asarray(result).tobytes()).hexdigest())
"""

    digests = []
    for thread_count in (1, 4):
        environment = os.environ.copy()
        environment["RAYON_NUM_THREADS"] = str(thread_count)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        digests.append(completed.stdout.strip())

    assert digests[0] == digests[1]
