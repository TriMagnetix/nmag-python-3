#!/usr/bin/env python3
"""Increment the patch component of the project version in pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
VERSION_PATTERN = re.compile(r'(?m)^(version\s*=\s*")([^"\n]+)("\s*)$')


def main() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    matches = list(VERSION_PATTERN.finditer(text))
    if len(matches) != 1:
        raise SystemExit("Expected exactly one project version in pyproject.toml")

    match = matches[0]
    version = match.group(2)
    components = version.split(".")
    if len(components) != 3 or not all(component.isdigit() for component in components):
        raise SystemExit(f"Patch bump requires a numeric X.Y.Z version, got {version!r}")

    major, minor, patch = (int(component) for component in components)
    new_version = f"{major}.{minor}.{patch + 1}"
    updated = text[: match.start(2)] + new_version + text[match.end(2) :]
    PYPROJECT.write_text(updated, encoding="utf-8")
    print(new_version)


if __name__ == "__main__":
    main()
