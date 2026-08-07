from __future__ import annotations

import os
from dataclasses import dataclass

from .backends import _load_rust_accelerator, _rust_accelerator_available


@dataclass(frozen=True, slots=True)
class ParallelRuntimeInfo:
    """Shared-memory execution settings visible to the current process.

    Attributes:
        logical_cpus: Logical processors reported by the operating system.
        rust_available: Whether a compatible native extension is installed.
        rust_worker_threads: Rayon worker count when Rust is available.
        rust_parallel_min_items: Operation-size crossover for parallel kernels.
    """

    logical_cpus: int
    rust_available: bool
    rust_worker_threads: int | None
    rust_parallel_min_items: int | None


def parallel_runtime_info() -> ParallelRuntimeInfo:
    """Return effective Rust worker and crossover settings without changing them."""
    logical_cpus = os.cpu_count() or 1
    if not _rust_accelerator_available():
        return ParallelRuntimeInfo(logical_cpus, False, None, None)

    accelerator = _load_rust_accelerator("NmagConfig.accelerator")
    try:
        worker_threads, minimum_items = accelerator.parallel_runtime_info()
    except AttributeError as exc:
        raise RuntimeError(
            "The installed nmag_accel does not expose parallel runtime metadata. "
            "Rebuild it with ./scripts/build-accelerator.sh."
        ) from exc
    return ParallelRuntimeInfo(
        logical_cpus=logical_cpus,
        rust_available=True,
        rust_worker_threads=int(worker_threads),
        rust_parallel_min_items=int(minimum_items),
    )
