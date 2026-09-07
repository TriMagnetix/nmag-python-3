from __future__ import annotations

import os
from pathlib import Path


def _read_integer(path: Path) -> int | None:
    try:
        raw_value = path.read_text(encoding="ascii").strip()
        return None if raw_value == "max" else int(raw_value)
    except (OSError, ValueError):
        return None


def available_memory_bytes() -> int | None:
    """Return the tightest host or cgroup estimate of currently available memory."""
    candidates: list[int] = []
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and available_pages > 0:
            candidates.append(page_size * available_pages)
    except (KeyError, OSError, ValueError):
        pass

    for limit_path, usage_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ):
        limit = _read_integer(limit_path)
        usage = _read_integer(usage_path)
        if limit is not None and usage is not None and limit > usage:
            candidates.append(limit - usage)

    return min(candidates) if candidates else None
