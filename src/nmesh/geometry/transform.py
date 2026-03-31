from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "AffineTransform",
    "inverse_shift",
    "inverse_scale",
    "inverse_plane_rotation",
    "inverse_axis_rotation",
]


def _as_vector(values: ArrayLike, dim: int | None = None) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("Expected a one-dimensional vector")
    if dim is not None and len(vector) != dim:
        raise ValueError(f"Expected a vector of length {dim}, got {len(vector)}")
    return vector.astype(np.float64, copy=False)


@dataclass(frozen=True, slots=True)
class AffineTransform:
    """Affine map used to evaluate transformed bodies in their local coordinates."""

    matrix: FloatArray
    displacement: FloatArray

    @classmethod
    def identity(cls, dim: int) -> "AffineTransform":
        """Return the identity transform for a space of the given dimension."""

        return cls(np.eye(dim, dtype=float), np.zeros(dim, dtype=float))

    @property
    def dim(self) -> int:
        return int(self.displacement.shape[0])

    def apply(self, points: ArrayLike) -> FloatArray:
        """Apply the affine transform to an array of points."""

        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != self.dim:
            raise ValueError(
                f"Expected points with shape (N, {self.dim}), got {pts.shape}"
            )
        return ((self.matrix @ pts.T).T + self.displacement).astype(np.float64, copy=False)

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        """Return the transform equivalent to applying `other` then `self`."""

        if self.dim != other.dim:
            raise ValueError("Cannot compose affine transforms with different dimensions")
        return AffineTransform(
            self.matrix @ other.matrix,
            self.matrix @ other.displacement + self.displacement,
        )


def inverse_shift(vector: ArrayLike) -> AffineTransform:
    """Return the inverse transform for a translation by `vector`."""

    shift = _as_vector(vector)
    return AffineTransform(np.eye(len(shift), dtype=float), -shift)


def inverse_scale(factors: ArrayLike) -> AffineTransform:
    """Return the inverse transform for a per-axis scale."""

    scale = _as_vector(factors)
    if np.any(scale == 0.0):
        raise ValueError("Scale factors must be non-zero")
    return AffineTransform(np.diag(1.0 / scale), np.zeros(len(scale), dtype=float))


def inverse_plane_rotation(dim: int, axis1: int, axis2: int, radians: float) -> AffineTransform:
    """Return the inverse transform for a planar rotation in `dim` dimensions."""

    if axis1 == axis2:
        raise ValueError("Rotation axes must be different")
    if not (0 <= axis1 < dim and 0 <= axis2 < dim):
        raise ValueError("Rotation axis index out of bounds")

    matrix = np.eye(dim, dtype=float)
    co = np.cos(-radians)
    si = np.sin(-radians)
    matrix[axis1, axis1] = co
    matrix[axis2, axis2] = co
    matrix[axis2, axis1] = si
    matrix[axis1, axis2] = -si
    return AffineTransform(matrix, np.zeros(dim, dtype=float))


def inverse_axis_rotation(axis: ArrayLike, radians: float) -> AffineTransform:
    """Return the inverse transform for a three-dimensional axis rotation."""

    axis_vector = _as_vector(axis, dim=3)
    norm = np.linalg.norm(axis_vector)
    if norm == 0.0:
        raise ValueError("Rotation axis must be non-zero")

    ax, ay, az = axis_vector / norm
    co = np.cos(-radians)
    si = np.sin(-radians)
    t = 1.0 - co

    matrix = np.array(
        [
            [co + t * ax * ax, t * ax * ay - az * si, t * ax * az + ay * si],
            [t * ax * ay + az * si, co + t * ay * ay, t * ay * az - ax * si],
            [t * ax * az - ay * si, t * ay * az + ax * si, co + t * az * az],
        ],
        dtype=float,
    )
    return AffineTransform(matrix, np.zeros(3, dtype=float))
