from __future__ import annotations

import numpy as np
import pytest

from nmag.demag import _oriented_boundary_faces


def _non_manifold_face_mesh() -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    simplices = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 4],
            [0, 1, 2, 5],
        ],
        dtype=np.int64,
    )
    return points, simplices


def test_python_boundary_face_builder_rejects_non_manifold_face():
    points, simplices = _non_manifold_face_mesh()

    with pytest.raises(ValueError, match="Non-manifold mesh"):
        _oriented_boundary_faces(points, simplices)


def test_rust_boundary_face_builder_rejects_non_manifold_face():
    rust_accel = pytest.importorskip("nmag_accel")
    points, simplices = _non_manifold_face_mesh()

    with pytest.raises(ValueError, match="non-manifold mesh"):
        rust_accel.build_oriented_boundary_faces(points, simplices)
