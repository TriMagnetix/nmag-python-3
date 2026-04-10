"""
Geometric primitives and implicit surface definitions.

This module provides pure-Python implementations of geometric primitives
(Box, Ellipsoid, Conic, Helix) using NumPy-based signed distance functions,
replacing the OCaml-based body primitives from mesh.ml.

Each primitive is represented as an implicit body with a scalar field
where positive values indicate interior points and negative values indicate
exterior points. Transformations are applied via affine matrix composition.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from ..utils.types import BoolArray, FloatArray
from .transform import (
    AffineTransform,
    inverse_axis_rotation,
    inverse_plane_rotation,
    inverse_scale,
    inverse_shift,
    _as_vector,
)

SignedField: TypeAlias = Callable[[FloatArray], FloatArray]
ShiftTransform: TypeAlias = tuple[Literal["shift"], ArrayLike]
ScaleTransform: TypeAlias = tuple[Literal["scale"], ArrayLike]
RotateTransform: TypeAlias = tuple[Literal["rotate"], tuple[int, int], float]
Rotate2DTransform: TypeAlias = tuple[Literal["rotate2d"], float]
Rotate3DTransform: TypeAlias = tuple[Literal["rotate3d"], ArrayLike, float]
TransformationStep: TypeAlias = (
    ShiftTransform
    | ScaleTransform
    | RotateTransform
    | Rotate2DTransform
    | Rotate3DTransform
)

# Number of complete spiral rotations for the helix primitive (4 turns = 8π)
_HELIX_SPIRAL_TURNS = 4.0

def _as_float_points(points: Sequence[Sequence[float]] | None) -> list[list[float]]:
    return [list(map(float, point)) for point in (points or [])]


def _coerce_query_points(points: ArrayLike, dim: int) -> tuple[FloatArray, bool]:
    coords = np.asarray(points, dtype=float)
    if coords.ndim == 1:
        if len(coords) != dim:
            raise ValueError(f"Expected a point of length {dim}, got {len(coords)}")
        return coords.reshape(1, dim).astype(np.float64, copy=False), True
    if coords.ndim == 2 and coords.shape[1] == dim:
        return coords.astype(np.float64, copy=False), False
    raise ValueError(f"Expected points with shape ({dim},) or (N, {dim}), got {coords.shape}")


@dataclass(frozen=True, slots=True)
class Body:
    """Implicit body evaluated through a sign-preserving scalar field."""

    dim: int
    evaluator: SignedField
    transform: AffineTransform

    def evaluate(self, points: ArrayLike) -> float | FloatArray:
        """Evaluate the body's scalar field at one point or an array of points."""

        coords, scalar = _coerce_query_points(points, self.dim)
        local_coords = self.transform.apply(coords)
        values = np.asarray(self.evaluator(local_coords), dtype=float).reshape(-1)
        if scalar:
            return float(values[0])
        return values.astype(np.float64, copy=False)

    def transformed(
        self,
        inverse_transform: AffineTransform,
        *,
        system_coords: bool,
    ) -> "Body":
        """Return a copy of the body with one more affine transform applied.

        Args:
            inverse_transform: Inverse affine map for the requested transformation.
            system_coords: When `True`, apply the new transform in system/space
                coordinates. When `False`, apply it in the body's local coordinates.

        Returns:
            A new body carrying the composed affine transform.

        Note:
            Composition order matters:
            - `system_coords=True` applies the new transform outside the existing one
            - `system_coords=False` applies the new transform inside the existing one
        """

        combined = (
            self.transform.compose(inverse_transform)
            if system_coords
            else inverse_transform.compose(self.transform)
        )
        return Body(self.dim, self.evaluator, combined)


def _make_body(dim: int, evaluator: SignedField) -> Body:
    """Create a body whose transform starts as the identity."""

    return Body(dim, evaluator, AffineTransform.identity(dim))


def bc_ellipsoid(radii: ArrayLike) -> SignedField:
    """Build the implicit field for an axis-aligned ellipsoid."""

    radii_vector = _as_vector(radii)
    if np.any(radii_vector <= 0.0):
        raise ValueError("Ellipsoid radii must be positive")

    def evaluate(points: FloatArray) -> FloatArray:
        scaled = points / radii_vector
        return 1.0 - np.sum(scaled * scaled, axis=1)

    return evaluate


def bc_box(corner_nw: ArrayLike, corner_se: ArrayLike) -> SignedField:
    """Build the implicit field for an axis-aligned box."""

    point1 = _as_vector(corner_nw)
    point2 = _as_vector(corner_se, dim=len(point1))
    half_lengths = np.abs(point2 - point1) * 0.5
    if np.any(half_lengths == 0.0):
        raise ValueError("Box edge lengths must be non-zero")
    midpoint = (point1 + point2) * 0.5

    def evaluate(points: FloatArray) -> FloatArray:
        relative = np.abs((points - midpoint) / half_lengths)
        return 1.0 - np.max(relative, axis=1)

    return evaluate


def _project_to_axis(points: FloatArray, axis: FloatArray) -> FloatArray:
    return points @ axis


def bc_frustum(
    center1: ArrayLike,
    radius1: float,
    center2: ArrayLike,
    radius2: float,
) -> SignedField:
    """Build the implicit field for a conical frustum."""

    point1 = _as_vector(center1)
    point2 = _as_vector(center2, dim=len(point1))
    if min(radius1, radius2) < 0.0 or max(radius1, radius2) == 0.0:
        raise ValueError("Frustum radii must be non-negative and not both zero")

    axis = point2 - point1
    axis_projection_center1 = float(np.dot(axis, point1))
    axis_projection_center2 = float(np.dot(axis, point2))
    axis_projection_delta = axis_projection_center2 - axis_projection_center1
    if axis_projection_delta == 0.0:
        raise ValueError("Frustum endpoints must not coincide")

    min_ap = min(axis_projection_center1, axis_projection_center2)
    max_ap = max(axis_projection_center1, axis_projection_center2)
    centre12 = 0.5 * (axis_projection_center1 + axis_projection_center2)
    radius_delta = radius2 - radius1

    def evaluate(points: FloatArray) -> FloatArray:
        ap = _project_to_axis(points, axis)
        out_top_bottom = np.where(
            ap > centre12,
            (ap - max_ap) / (centre12 - max_ap),
            np.where(
                ap < centre12,
                (min_ap - ap) / (min_ap - centre12),
                1.0,
            ),
        )
        axis_factor = (ap - axis_projection_center1) / axis_projection_delta
        radius_here = radius1 + radius_delta * axis_factor
        axis_projection = point1 + np.outer(axis_factor, axis)
        delta = points - axis_projection
        axis_distance = np.sqrt(np.sum(delta * delta, axis=1))
        return np.minimum(out_top_bottom, radius_here - axis_distance)

    return evaluate


def bc_helix(
    center1: ArrayLike,
    radius1: float,
    center2: ArrayLike,
    radius2: float,
) -> SignedField:
    """Build the implicit field for the tapered helix primitive."""

    point1 = _as_vector(center1, dim=3)
    point2 = _as_vector(center2, dim=3)
    if min(radius1, radius2) <= 0.0:
        raise ValueError("Helix radii must be positive")

    axis = point2 - point1
    axis_projection_center1 = float(np.dot(axis, point1))
    axis_projection_center2 = float(np.dot(axis, point2))
    axis_projection_delta = axis_projection_center2 - axis_projection_center1
    if axis_projection_delta == 0.0:
        raise ValueError("Helix endpoints must not coincide")

    min_ap = min(axis_projection_center1, axis_projection_center2)
    max_ap = max(axis_projection_center1, axis_projection_center2)
    centre12 = 0.5 * (axis_projection_center1 + axis_projection_center2)

    def evaluate(points: FloatArray) -> FloatArray:
        ap = _project_to_axis(points, axis)
        axis_factor = (ap - axis_projection_center1) / axis_projection_delta
        out_top_bottom = np.where(
            ap > centre12,
            (ap - max_ap) / (centre12 - max_ap),
            np.where(
                ap < centre12,
                (min_ap - ap) / (min_ap - centre12),
                1.0,
            ),
        )
        alpha = 2.0 * math.pi * _HELIX_SPIRAL_TURNS * axis_factor
        spiral_direction = np.column_stack(
            (np.cos(alpha), np.sin(alpha), np.zeros_like(alpha))
        )
        helix_circle_radius = radius2 * (1.0 - axis_factor)
        helix_spiral_radius = radius1 * (1.0 - axis_factor)
        axis_projection = point1 + np.outer(axis_factor, axis)
        spiral_centres = axis_projection + helix_spiral_radius[:, None] * spiral_direction
        helix_value = helix_circle_radius**2 - np.sum(
            (points - spiral_centres) ** 2,
            axis=1,
        )
        return np.minimum(out_top_bottom, helix_value)

    return evaluate


class MeshObject:
    """Base class for geometric primitives and CSG expressions."""

    def __init__(
        self,
        dim: int,
        fixed: Sequence[Sequence[float]] | None = None,
        mobile: Sequence[Sequence[float]] | None = None,
        *,
        body: Body | None = None,
    ):
        self.dim = int(dim)
        self.fixed_points = _as_float_points(fixed)
        self.mobile_points = _as_float_points(mobile)
        self.obj = body

    def _require_body(self) -> Body:
        if self.obj is None:
            raise ValueError("MeshObject does not wrap a body")
        return self.obj

    def signed_distance(self, points: ArrayLike) -> float | FloatArray:
        """Evaluate the object's scalar field at one point or many points."""

        return self._require_body().evaluate(points)

    def contains(self, points: ArrayLike) -> bool | BoolArray:
        """Return whether the supplied point or points lie inside the object."""

        values = self.signed_distance(points)
        if isinstance(values, float):
            return values > 0.0
        return values > 0.0

    def shift(self, vector: ArrayLike, system_coords: bool = True):
        """Translate the object by the given vector."""

        body = self._require_body()
        inverse_transform = inverse_shift(_as_vector(vector, dim=self.dim))
        self.obj = body.transformed(inverse_transform, system_coords=system_coords)

    def scale(self, factors: ArrayLike):
        """Scale the object in body coordinates by the supplied per-axis factors."""

        body = self._require_body()
        inverse_transform = inverse_scale(_as_vector(factors, dim=self.dim))
        self.obj = body.transformed(inverse_transform, system_coords=False)

    def rotate(self, a1: int, a2: int, angle: float, system_coords: bool = True):
        """Rotate the object in the plane spanned by the two axis indices."""

        body = self._require_body()
        radians = math.radians(float(angle))
        inverse_transform = inverse_plane_rotation(self.dim, int(a1), int(a2), radians)
        self.obj = body.transformed(inverse_transform, system_coords=system_coords)

    def rotate_3d(self, axis: ArrayLike, angle: float, system_coords: bool = True):
        """Rotate a three-dimensional object about the supplied axis vector."""

        body = self._require_body()
        if self.dim != 3:
            raise ValueError("3D axis rotation is only available for three-dimensional bodies")
        radians = math.radians(float(angle))
        inverse_transform = inverse_axis_rotation(axis, radians)
        self.obj = body.transformed(inverse_transform, system_coords=system_coords)

    def transform(
        self,
        transformations: Iterable[TransformationStep] | None,
        system_coords: bool = True,
    ) -> None:
        """Apply a sequence of named transform tuples in the given order."""

        for transformation in transformations or []:
            match transformation:
                case ("shift", vector):
                    self.shift(vector, system_coords)
                case ("scale", factors):
                    self.scale(factors)
                case ("rotate", (axis1, axis2), angle):
                    self.rotate(axis1, axis2, angle, system_coords)
                case ("rotate2d", angle):
                    self.rotate(0, 1, angle, system_coords)
                case ("rotate3d", axis, angle):
                    self.rotate_3d(axis, angle, system_coords)
                case _:
                    raise ValueError(f"Unknown transformation {transformation!r}")


class Box(MeshObject):
    """Axis-aligned box defined by two opposite corners."""

    def __init__(
        self,
        p1: ArrayLike,
        p2: ArrayLike,
        transform: Iterable[TransformationStep] | None = None,
        fixed: Sequence[Sequence[float]] | None = None,
        mobile: Sequence[Sequence[float]] | None = None,
        system_coords: bool = True,
        use_fixed_corners: bool = False,
    ):
        point1 = _as_vector(p1)
        point2 = _as_vector(p2, dim=len(point1))
        fixed_points = _as_float_points(fixed)
        if use_fixed_corners:
            fixed_points.extend(
                [list(corner) for corner in itertools.product(*zip(point1, point2))]
            )
        body = _make_body(len(point1), bc_box(point1, point2))
        super().__init__(len(point1), fixed_points, mobile, body=body)
        self.transform(transform, system_coords)


class Ellipsoid(MeshObject):
    """Ellipsoid with principal radii aligned to the coordinate axes."""

    def __init__(
        self,
        lengths: ArrayLike,
        transform: Iterable[TransformationStep] | None = None,
        fixed: Sequence[Sequence[float]] | None = None,
        mobile: Sequence[Sequence[float]] | None = None,
        system_coords: bool = True,
    ):
        radii = _as_vector(lengths)
        body = _make_body(len(radii), bc_ellipsoid(radii))
        super().__init__(len(radii), fixed, mobile, body=body)
        self.transform(transform, system_coords)


class Conic(MeshObject):
    """Conical frustum defined by two centres and their radii."""

    def __init__(
        self,
        c1: ArrayLike,
        r1: float,
        c2: ArrayLike,
        r2: float,
        transform: Iterable[TransformationStep] | None = None,
        fixed: Sequence[Sequence[float]] | None = None,
        mobile: Sequence[Sequence[float]] | None = None,
        system_coords: bool = True,
    ):
        point1 = _as_vector(c1)
        point2 = _as_vector(c2, dim=len(point1))
        body = _make_body(
            len(point1),
            bc_frustum(point1, float(r1), point2, float(r2)),
        )
        super().__init__(len(point1), fixed, mobile, body=body)
        self.transform(transform, system_coords)


class Helix(MeshObject):
    """Tapered helix primitive used by the original mesher geometry layer."""

    def __init__(
        self,
        c1: ArrayLike,
        r1: float,
        c2: ArrayLike,
        r2: float,
        transform: Iterable[TransformationStep] | None = None,
        fixed: Sequence[Sequence[float]] | None = None,
        mobile: Sequence[Sequence[float]] | None = None,
        system_coords: bool = True,
    ):
        point1 = _as_vector(c1, dim=3)
        point2 = _as_vector(c2, dim=3)
        body = _make_body(
            3,
            bc_helix(point1, float(r1), point2, float(r2)),
        )
        super().__init__(3, fixed, mobile, body=body)
        self.transform(transform, system_coords)
