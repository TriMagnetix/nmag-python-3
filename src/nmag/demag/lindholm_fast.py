from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np

_JIT_FUNCTIONS: list[Callable[..., Any]] = []
_jit_compiled = False
P = ParamSpec("P")
R = TypeVar("R")


def __getattr__(name: str) -> Any:
    if name == "_JIT_COMPILED":
        return _jit_compiled
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _compile_jit_functions() -> None:
    global _jit_compiled
    if _jit_compiled:
        return

    try:
        from numba import njit
    except ImportError:  # pragma: no cover - numba is a project dependency.
        compiled_functions = {function.__name__: function for function in _JIT_FUNCTIONS}
    else:
        compiled_functions = {
            function.__name__: njit(cache=True)(function) for function in _JIT_FUNCTIONS
        }

    globals().update(compiled_functions)
    _jit_compiled = True


def _jit(function: Callable[P, R]) -> Callable[P, R]:
    _JIT_FUNCTIONS.append(function)

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        _compile_jit_functions()
        compiled = cast(Callable[P, R], globals()[function.__name__])
        return compiled(*args, **kwargs)

    cast(Any, wrapper).py_func = function
    return wrapper


@_jit
def _dot3(left: np.ndarray, right: np.ndarray) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


@_jit
def _cross3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(3, dtype=np.float64)
    result[0] = left[1] * right[2] - left[2] * right[1]
    result[1] = left[2] * right[0] - left[0] * right[2]
    result[2] = left[0] * right[1] - left[1] * right[0]
    return result


@_jit
def _subtract3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(3, dtype=np.float64)
    result[0] = left[0] - right[0]
    result[1] = left[1] - right[1]
    result[2] = left[2] - right[2]
    return result


@_jit
def _norm3(vector: np.ndarray) -> float:
    return np.sqrt(_dot3(vector, vector))


@_jit
def _normalised3(vector: np.ndarray) -> np.ndarray:
    result = np.zeros(3, dtype=np.float64)
    length = _norm3(vector)
    if length == 0.0:
        return result
    result[0] = vector[0] / length
    result[1] = vector[1] / length
    result[2] = vector[2] / length
    return result


@_jit
def _triangle_space_angle_fast(
    observer: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    r0 = _subtract3(p0, observer)
    r1 = _subtract3(p1, observer)
    r2 = _subtract3(p2, observer)
    r0_len = _norm3(r0)
    r1_len = _norm3(r1)
    r2_len = _norm3(r2)
    dot01 = _dot3(r0, r1)
    dot12 = _dot3(r1, r2)
    dot20 = _dot3(r2, r0)
    numerator = r0_len * r1_len * r2_len + r0_len * dot12 + r1_len * dot20 + r2_len * dot01
    denominator_squared = (
        2.0 * (r1_len * r2_len + dot12) * (r2_len * r0_len + dot20) * (r0_len * r1_len + dot01)
    )
    if denominator_squared <= 0.0:
        return 0.0
    quotient = numerator / np.sqrt(denominator_squared)
    if quotient < -1.0:
        quotient = -1.0
    elif quotient > 1.0:
        quotient = 1.0
    return 2.0 * np.arccos(quotient)


@_jit
def _safe_log_ratio_fast(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return np.nan
    return np.log(numerator / denominator)


@_jit
def _lindholm_triangle_contributions_fast(
    observer: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    outward_surface_normal: np.ndarray,
) -> np.ndarray:
    angle = _triangle_space_angle_fast(observer, p0, p1, p2)
    r0 = _subtract3(p0, observer)
    r1 = _subtract3(p1, observer)
    r2 = _subtract3(p2, observer)
    s0 = _subtract3(r2, r1)
    s1 = _subtract3(r0, r2)
    s2 = _subtract3(r1, r0)
    xi0 = _normalised3(s0)
    xi1 = _normalised3(s1)
    xi2 = _normalised3(s2)
    area_vector = _cross3(_subtract3(p1, p0), _subtract3(p2, p0))
    area = 0.5 * _norm3(area_vector)
    contributions = np.zeros(3, dtype=np.float64)
    if area == 0.0:
        return contributions

    zeta_vector = _normalised3(area_vector)
    r0_len = _norm3(r0)
    r1_len = _norm3(r1)
    r2_len = _norm3(r2)
    s0_len = _norm3(s0)
    s1_len = _norm3(s1)
    s2_len = _norm3(s2)
    c01 = _dot3(xi0, xi1)
    c12 = _dot3(xi1, xi2)
    c20 = _dot3(xi2, xi0)
    zeta = _dot3(zeta_vector, r0)
    eta0 = _cross3(zeta_vector, xi0)
    eta1 = _cross3(zeta_vector, xi1)
    eta2 = _cross3(zeta_vector, xi2)
    log0 = _safe_log_ratio_fast(r1_len + r2_len + s0_len, r1_len + r2_len - s0_len)
    log1 = _safe_log_ratio_fast(r2_len + r0_len + s1_len, r2_len + r0_len - s1_len)
    log2 = _safe_log_ratio_fast(r0_len + r1_len + s2_len, r0_len + r1_len - s2_len)
    sign_zeta = 0.0
    if zeta < 0.0:
        sign_zeta = -1.0
    elif zeta > 0.0:
        sign_zeta = 1.0
    angle_pm = angle * sign_zeta
    denominator_factor = 1.0 / (8.0 * np.pi * area)
    gamma0_log = log0 + c01 * log1 + c20 * log2
    gamma1_log = c01 * log0 + log1 + c12 * log2
    gamma2_log = c20 * log0 + c12 * log1 + log2
    values = np.empty(3, dtype=np.float64)
    values[0] = s0_len * denominator_factor * (angle_pm * _dot3(eta0, r1) - zeta * gamma0_log)
    values[1] = s1_len * denominator_factor * (angle_pm * _dot3(eta1, r2) - zeta * gamma1_log)
    values[2] = s2_len * denominator_factor * (angle_pm * _dot3(eta2, r0) - zeta * gamma2_log)
    if _dot3(zeta_vector, outward_surface_normal) < 0.0:
        values[0] = -values[0]
        values[1] = -values[1]
        values[2] = -values[2]
    for index in range(3):
        if np.isfinite(values[index]):
            contributions[index] = values[index]
    return contributions


def _normalised_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    result = np.zeros_like(vectors, dtype=float)
    positive = lengths > 0.0
    result[positive] = vectors[positive] / lengths[positive, np.newaxis]
    return result


def _precompute_lindholm_face_geometry(
    points: np.ndarray,
    face_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    face_points = points[face_nodes]
    if len(face_nodes) == 0:
        return (
            face_points,
            np.empty((0, 3), dtype=float),
            np.empty((0, 3, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=float),
        )

    edge0 = face_points[:, 2] - face_points[:, 1]
    edge1 = face_points[:, 0] - face_points[:, 2]
    edge2 = face_points[:, 1] - face_points[:, 0]
    edge_vectors = np.stack([edge0, edge1, edge2], axis=1)
    edge_lengths = np.linalg.norm(edge_vectors, axis=2)
    xis = np.zeros_like(edge_vectors, dtype=float)
    for index in range(3):
        xis[:, index] = _normalised_rows(edge_vectors[:, index])

    area_vectors = np.cross(
        face_points[:, 1] - face_points[:, 0],
        face_points[:, 2] - face_points[:, 0],
    )
    double_areas = np.linalg.norm(area_vectors, axis=1)
    zeta_vectors = np.zeros_like(area_vectors, dtype=float)
    positive_area = double_areas > 0.0
    zeta_vectors[positive_area] = (
        area_vectors[positive_area] / double_areas[positive_area, np.newaxis]
    )

    eta_vectors = np.stack(
        [
            np.cross(zeta_vectors, xis[:, 0]),
            np.cross(zeta_vectors, xis[:, 1]),
            np.cross(zeta_vectors, xis[:, 2]),
        ],
        axis=1,
    )
    corner_cosines = np.stack(
        [
            np.einsum("ij,ij->i", xis[:, 0], xis[:, 1]),
            np.einsum("ij,ij->i", xis[:, 1], xis[:, 2]),
            np.einsum("ij,ij->i", xis[:, 2], xis[:, 0]),
        ],
        axis=1,
    )
    denominator_factors = np.zeros(len(face_nodes), dtype=float)
    denominator_factors[positive_area] = 1.0 / (4.0 * np.pi * double_areas[positive_area])
    return face_points, zeta_vectors, eta_vectors, edge_lengths, corner_cosines, denominator_factors


@_jit
def _lindholm_triangle_contributions_precomputed_fast(
    observer: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    zeta_vector: np.ndarray,
    eta0: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    edge_lengths: np.ndarray,
    corner_cosines: np.ndarray,
    denominator_factor: float,
) -> np.ndarray:
    contributions = np.zeros(3, dtype=np.float64)
    if denominator_factor == 0.0:
        return contributions

    r0 = _subtract3(p0, observer)
    r1 = _subtract3(p1, observer)
    r2 = _subtract3(p2, observer)
    r0_len = _norm3(r0)
    r1_len = _norm3(r1)
    r2_len = _norm3(r2)
    dot01 = _dot3(r0, r1)
    dot12 = _dot3(r1, r2)
    dot20 = _dot3(r2, r0)
    numerator = r0_len * r1_len * r2_len + r0_len * dot12 + r1_len * dot20 + r2_len * dot01
    denominator_squared = (
        2.0 * (r1_len * r2_len + dot12) * (r2_len * r0_len + dot20) * (r0_len * r1_len + dot01)
    )
    angle = 0.0
    if denominator_squared > 0.0:
        quotient = numerator / np.sqrt(denominator_squared)
        if quotient < -1.0:
            quotient = -1.0
        elif quotient > 1.0:
            quotient = 1.0
        angle = 2.0 * np.arccos(quotient)

    s0_len = edge_lengths[0]
    s1_len = edge_lengths[1]
    s2_len = edge_lengths[2]
    c01 = corner_cosines[0]
    c12 = corner_cosines[1]
    c20 = corner_cosines[2]
    zeta = _dot3(zeta_vector, r0)
    log0 = _safe_log_ratio_fast(r1_len + r2_len + s0_len, r1_len + r2_len - s0_len)
    log1 = _safe_log_ratio_fast(r2_len + r0_len + s1_len, r2_len + r0_len - s1_len)
    log2 = _safe_log_ratio_fast(r0_len + r1_len + s2_len, r0_len + r1_len - s2_len)
    sign_zeta = 0.0
    if zeta < 0.0:
        sign_zeta = -1.0
    elif zeta > 0.0:
        sign_zeta = 1.0
    angle_pm = angle * sign_zeta
    gamma0_log = log0 + c01 * log1 + c20 * log2
    gamma1_log = c01 * log0 + log1 + c12 * log2
    gamma2_log = c20 * log0 + c12 * log1 + log2

    value0 = s0_len * denominator_factor * (angle_pm * _dot3(eta0, r1) - zeta * gamma0_log)
    value1 = s1_len * denominator_factor * (angle_pm * _dot3(eta1, r2) - zeta * gamma1_log)
    value2 = s2_len * denominator_factor * (angle_pm * _dot3(eta2, r0) - zeta * gamma2_log)
    if np.isfinite(value0):
        contributions[0] = value0
    if np.isfinite(value1):
        contributions[1] = value1
    if np.isfinite(value2):
        contributions[2] = value2
    return contributions


@_jit
def _boundary_node_solid_angles_fast(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    angles = np.zeros(len(points), dtype=np.float64)
    for simplex_index in range(len(simplices)):
        simplex = simplices[simplex_index]
        point0 = int(simplex[0])
        point1 = int(simplex[1])
        point2 = int(simplex[2])
        point3 = int(simplex[3])
        angles[point0] += abs(
            _triangle_space_angle_fast(
                points[point0],
                points[point1],
                points[point2],
                points[point3],
            )
        )
        angles[point1] += abs(
            _triangle_space_angle_fast(
                points[point1],
                points[point0],
                points[point2],
                points[point3],
            )
        )
        angles[point2] += abs(
            _triangle_space_angle_fast(
                points[point2],
                points[point0],
                points[point1],
                points[point3],
            )
        )
        angles[point3] += abs(
            _triangle_space_angle_fast(
                points[point3],
                points[point0],
                points[point1],
                points[point2],
            )
        )
    return angles


@_jit
def _build_lindholm_bem_matrix_fast(
    points: np.ndarray,
    simplices: np.ndarray,
    boundary_faces: np.ndarray,
    face_points: np.ndarray,
    zeta_vectors: np.ndarray,
    eta_vectors: np.ndarray,
    edge_lengths: np.ndarray,
    corner_cosines: np.ndarray,
    denominator_factors: np.ndarray,
    boundary_nodes: np.ndarray,
    local_index_by_point: np.ndarray,
) -> np.ndarray:
    bem = np.zeros((len(boundary_nodes), len(boundary_nodes)), dtype=np.float64)
    solid_angles = _boundary_node_solid_angles_fast(points, simplices)

    for row in range(len(boundary_nodes)):
        observer_index = int(boundary_nodes[row])
        observer = points[observer_index]
        for face_index in range(len(boundary_faces)):
            face = boundary_faces[face_index]
            point0 = int(face[0])
            point1 = int(face[1])
            point2 = int(face[2])
            if denominator_factors[face_index] == 0.0:
                continue
            contributions = _lindholm_triangle_contributions_precomputed_fast(
                observer,
                face_points[face_index, 0],
                face_points[face_index, 1],
                face_points[face_index, 2],
                zeta_vectors[face_index],
                eta_vectors[face_index, 0],
                eta_vectors[face_index, 1],
                eta_vectors[face_index, 2],
                edge_lengths[face_index],
                corner_cosines[face_index],
                denominator_factors[face_index],
            )
            bem[row, int(local_index_by_point[point0])] += contributions[0]
            bem[row, int(local_index_by_point[point1])] += contributions[1]
            bem[row, int(local_index_by_point[point2])] += contributions[2]
        bem[row, row] += solid_angles[observer_index] / (4.0 * np.pi) - 1.0
    return bem


@_jit
def _apply_lindholm_bem_matrix_free_fast(
    points: np.ndarray,
    boundary_faces: np.ndarray,
    face_points: np.ndarray,
    zeta_vectors: np.ndarray,
    eta_vectors: np.ndarray,
    edge_lengths: np.ndarray,
    corner_cosines: np.ndarray,
    denominator_factors: np.ndarray,
    boundary_nodes: np.ndarray,
    local_index_by_point: np.ndarray,
    solid_angles: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(boundary_nodes), dtype=np.float64)
    for row in range(len(boundary_nodes)):
        observer_index = int(boundary_nodes[row])
        observer = points[observer_index]
        value = 0.0
        for face_index in range(len(boundary_faces)):
            if denominator_factors[face_index] == 0.0:
                continue
            face = boundary_faces[face_index]
            contributions = _lindholm_triangle_contributions_precomputed_fast(
                observer,
                face_points[face_index, 0],
                face_points[face_index, 1],
                face_points[face_index, 2],
                zeta_vectors[face_index],
                eta_vectors[face_index, 0],
                eta_vectors[face_index, 1],
                eta_vectors[face_index, 2],
                edge_lengths[face_index],
                corner_cosines[face_index],
                denominator_factors[face_index],
            )
            value += contributions[0] * values[int(local_index_by_point[int(face[0])])]
            value += contributions[1] * values[int(local_index_by_point[int(face[1])])]
            value += contributions[2] * values[int(local_index_by_point[int(face[2])])]
        value += (solid_angles[observer_index] / (4.0 * np.pi) - 1.0) * values[row]
        result[row] = value
    return result
