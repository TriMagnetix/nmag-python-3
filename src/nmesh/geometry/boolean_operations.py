"""
Constructive Solid Geometry (CSG) operations for mesh objects.

This module provides boolean operations (union, difference, intersection)
for combining geometric primitives into complex shapes. Operations are
implemented using signed distance field composition with NumPy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..utils.types import FloatArray
from .primitives import Body, MeshObject, _make_body


def _validate_objects(objects: Sequence[MeshObject], operation: str) -> tuple[MeshObject, ...]:
    if not isinstance(objects, Sequence):
        raise TypeError(f"{operation} expects a sequence of MeshObject instances")
    object_tuple = tuple(objects)
    if len(object_tuple) < 2:
        raise ValueError(f"{operation} requires at least two objects")
    dims = {obj.dim for obj in object_tuple}
    if len(dims) != 1:
        raise ValueError(f"{operation} requires objects with the same dimension")
    return object_tuple


def _collect_points(objects: Sequence[MeshObject]) -> tuple[list[list[float]], list[list[float]]]:
    fixed_points: list[list[float]] = []
    mobile_points: list[list[float]] = []
    for obj in objects:
        fixed_points.extend(obj.fixed_points)
        mobile_points.extend(obj.mobile_points)
    return fixed_points, mobile_points


def _body_for(obj: MeshObject) -> Body:
    if obj.obj is None:
        raise ValueError("Boolean operations require MeshObject instances with bodies")
    return obj.obj


def _stack_evaluations(bodies: Sequence[Body], points: FloatArray) -> FloatArray:
    """
    Evaluate multiple bodies at the same points and stack results.

    Args:
        bodies: Sequence of Body instances to evaluate
        points: Array of shape (N, dim) containing query points

    Returns:
        Array of shape (len(bodies), N) containing signed distance values

    Note:
        Expects each body.evaluate() to return a 1D array of length N.
        No shape validation is performed for performance reasons.
    """
    return np.stack(
        [np.asarray(body.evaluate(points), dtype=float) for body in bodies],
        axis=0,
    ).astype(np.float64, copy=False)


def union(objects: Sequence[MeshObject]) -> MeshObject:
    """Return a body that is inside any of the supplied objects."""

    members = _validate_objects(objects, "Union")
    fixed_points, mobile_points = _collect_points(members)
    dim = members[0].dim
    bodies = tuple(_body_for(member) for member in members)
    body = _make_body(
        dim,
        lambda points: np.max(_stack_evaluations(bodies, points), axis=0),
    )
    return MeshObject(dim, fixed_points, mobile_points, body=body)


def difference(mother: MeshObject, subtract: Sequence[MeshObject]) -> MeshObject:
    """Return the mother object with each subtractor carved out of it."""

    mother_body = _body_for(mother)
    subtractors = tuple(subtract)
    if not subtractors:
        return MeshObject(
            mother.dim,
            mother.fixed_points[:],
            mother.mobile_points[:],
            body=mother_body,
        )
    dims = {mother.dim, *(obj.dim for obj in subtractors)}
    if len(dims) != 1:
        raise ValueError("Difference requires objects with the same dimension")

    fixed_points = mother.fixed_points[:]
    mobile_points = mother.mobile_points[:]
    for obj in subtractors:
        fixed_points.extend(obj.fixed_points)
        mobile_points.extend(obj.mobile_points)

    dim = mother.dim
    subtractor_bodies = tuple(_body_for(obj) for obj in subtractors)
    body = _make_body(
        dim,
        lambda points: np.min(
            np.vstack(
                (
                    np.asarray(mother_body.evaluate(points), dtype=float)[None, :],
                    -_stack_evaluations(subtractor_bodies, points),
                )
            ),
            axis=0,
        ),
    )
    return MeshObject(dim, fixed_points, mobile_points, body=body)


def intersect(objects: Sequence[MeshObject]) -> MeshObject:
    """Return a body that is inside all of the supplied objects."""

    members = _validate_objects(objects, "Intersection")
    fixed_points, mobile_points = _collect_points(members)
    dim = members[0].dim
    bodies = tuple(_body_for(member) for member in members)
    body = _make_body(
        dim,
        lambda points: np.min(_stack_evaluations(bodies, points), axis=0),
    )
    return MeshObject(dim, fixed_points, mobile_points, body=body)
