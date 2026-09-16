"""Readers for Netgen Neutral tetrahedral meshes."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import TextIO

from ..backend import RawMesh


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


class _LineReader:
    def __init__(self, stream: TextIO, path: Path) -> None:
        self.stream = stream
        self.path = path
        self.line_number = 0

    def next(self, description: str) -> tuple[int, str]:
        for line in self.stream:
            self.line_number += 1
            stripped = line.strip()
            if stripped:
                return self.line_number, stripped
        raise ValueError(f"{self.path} ended while reading {description}.")


def _parse_count(line: str, path: Path, line_number: int, description: str) -> int:
    try:
        count = int(line)
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number} has an invalid {description} count: {line!r}."
        ) from exc
    if count < 0:
        raise ValueError(f"{path}:{line_number} has a negative {description} count.")
    return count


def read_netgen_neutral(path: str | Path) -> RawMesh:
    """Read a Netgen Neutral tetrahedral mesh into a :class:`RawMesh`.

    Netgen Neutral element indices are one-based.  They are converted to the
    zero-based indexing used by ``nmesh`` while region IDs are preserved.
    Gzip-compressed files are accepted when the filename ends in ``.gz``.
    """

    mesh_path = Path(path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Netgen mesh does not exist: {mesh_path}")

    try:
        with _open_text(mesh_path) as stream:
            reader = _LineReader(stream, mesh_path)
            point_count_line, point_count_text = reader.next("point")
            point_count = _parse_count(
                point_count_text, mesh_path, point_count_line, "point"
            )
            if point_count == 0:
                raise ValueError(f"{mesh_path} contains no points.")

            points: list[list[float]] = []
            for index in range(point_count):
                line_number, line = reader.next(f"point {index + 1}")
                values = line.split()
                if len(values) != 3:
                    raise ValueError(
                        f"{mesh_path}:{line_number} point {index + 1} must have three coordinates."
                    )
                try:
                    points.append([float(value) for value in values])
                except ValueError as exc:
                    raise ValueError(
                        f"{mesh_path}:{line_number} contains a non-numeric point coordinate."
                    ) from exc

            simplex_count_line, simplex_count_text = reader.next("tetrahedron")
            simplex_count = _parse_count(
                simplex_count_text,
                mesh_path,
                simplex_count_line,
                "tetrahedron",
            )
            if simplex_count == 0:
                raise ValueError(f"{mesh_path} contains no tetrahedra.")

            simplices: list[list[int]] = []
            regions: list[int] = []
            for index in range(simplex_count):
                line_number, line = reader.next(f"tetrahedron {index + 1}")
                values = line.split()
                if len(values) != 5:
                    raise ValueError(
                        f"{mesh_path}:{line_number} tetrahedron {index + 1} must contain "
                        "one region ID and four point indices."
                    )
                try:
                    region = int(values[0])
                    indices = [int(value) - 1 for value in values[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"{mesh_path}:{line_number} contains a non-integer tetrahedron value."
                    ) from exc
                if region <= 0:
                    raise ValueError(
                        f"{mesh_path}:{line_number} has invalid region ID {region}; expected > 0."
                    )
                if any(point < 0 or point >= point_count for point in indices):
                    raise ValueError(
                        f"{mesh_path}:{line_number} contains a point index outside 1..{point_count}."
                    )
                simplices.append(indices)
                regions.append(region)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Unable to read Netgen mesh {mesh_path}: {exc}") from exc

    return RawMesh(points=points, simplices=simplices, regions=regions, dim=3)
