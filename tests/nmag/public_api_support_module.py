"""Shared mesh fixtures and reference helpers for public API regression tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import nmesh
from nmag.simulation import TETRA_FACE_VERTICES


def subprocess_env_with_repo_src() -> dict[str, str]:
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    pythonpath = os.environ.get("PYTHONPATH")
    return {
        **os.environ,
        "PYTHONPATH": src_path if not pythonpath else f"{src_path}{os.pathsep}{pythonpath}",
    }


def write_single_region_mesh(path: Path) -> None:
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[1],
    )
    mesh.save(path)


def write_two_tetra_mesh(path: Path) -> None:
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        simplices_indices=[
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        simplices_regions=[1, 1],
    )
    mesh.save(path)


def write_two_region_mesh(path: Path) -> None:
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3], [1, 2, 3, 4]],
        simplices_regions=[1, 2],
    )
    mesh.save(path)


def reference_oriented_boundary_faces(
    points: np.ndarray,
    simplices: np.ndarray,
) -> list[tuple[int, tuple[int, int, int]]]:
    face_owners: dict[tuple[int, int, int], tuple[int, tuple[int, int, int]] | None] = {}
    for cell_index, simplex in enumerate(simplices):
        cell_center = np.mean(points[simplex], axis=0)
        for local_face in TETRA_FACE_VERTICES:
            face = tuple(int(simplex[index]) for index in local_face)
            key = tuple(sorted(face))
            if key in face_owners:
                face_owners[key] = None
                continue

            a, b, c = points[list(face)]
            normal = np.cross(b - a, c - a)
            face_center = (a + b + c) / 3.0
            if np.dot(normal, cell_center - face_center) > 0.0:
                face = (face[0], face[2], face[1])
            face_owners[key] = (cell_index, face)

    return [owner for owner in face_owners.values() if owner is not None]
