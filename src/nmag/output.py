"""Output-file lifecycle helpers shared by simulation implementations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .config import OutputPolicy


def prepare_output_files(paths: Iterable[Path], policy: OutputPolicy) -> None:
    """Create parents and apply an explicit output lifecycle policy."""

    destinations = tuple(paths)
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)

    existing = tuple(destination for destination in destinations if destination.exists())
    if policy == "error" and existing:
        rendered = ", ".join(str(destination) for destination in existing)
        raise FileExistsError(
            f"Output file(s) already exist: {rendered}. "
            "Use NmagConfig(output_policy='replace' or 'append') explicitly."
        )
    if policy == "replace":
        for destination in existing:
            destination.unlink()
