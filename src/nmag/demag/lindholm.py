from __future__ import annotations

import numpy as np


def _triangle_space_angle(
    observer: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """Legacy unsigned space angle of a triangle as seen from ``observer``."""

    r0 = p0 - observer
    r1 = p1 - observer
    r2 = p2 - observer
    r0_len = float(np.linalg.norm(r0))
    r1_len = float(np.linalg.norm(r1))
    r2_len = float(np.linalg.norm(r2))
    dot01 = float(np.dot(r0, r1))
    dot12 = float(np.dot(r1, r2))
    dot20 = float(np.dot(r2, r0))
    numerator = r0_len * r1_len * r2_len + r0_len * dot12 + r1_len * dot20 + r2_len * dot01
    denominator_squared = (
        2.0 * (r1_len * r2_len + dot12) * (r2_len * r0_len + dot20) * (r0_len * r1_len + dot01)
    )
    if denominator_squared <= 0.0:
        return 0.0
    quotient = numerator / float(np.sqrt(denominator_squared))
    return float(2.0 * np.arccos(min(1.0, max(-1.0, quotient))))


def _normalised(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        return np.zeros_like(vector, dtype=float)
    return vector / length


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.log(numerator / denominator))


def _lindholm_triangle_contributions(
    observer: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    outward_surface_normal: np.ndarray,
) -> np.ndarray:
    """Legacy Lindholm BEM row contribution for one first-order triangle."""

    angle = _triangle_space_angle(observer, p0, p1, p2)
    r0 = p0 - observer
    r1 = p1 - observer
    r2 = p2 - observer
    s0 = r2 - r1
    s1 = r0 - r2
    s2 = r1 - r0
    xi0 = _normalised(s0)
    xi1 = _normalised(s1)
    xi2 = _normalised(s2)
    area_vector = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * float(np.linalg.norm(area_vector))
    if area == 0.0:
        return np.zeros(3, dtype=float)

    zeta_vector = _normalised(area_vector)
    r0_len = float(np.linalg.norm(r0))
    r1_len = float(np.linalg.norm(r1))
    r2_len = float(np.linalg.norm(r2))
    s0_len = float(np.linalg.norm(s0))
    s1_len = float(np.linalg.norm(s1))
    s2_len = float(np.linalg.norm(s2))
    c01 = float(np.dot(xi0, xi1))
    c12 = float(np.dot(xi1, xi2))
    c20 = float(np.dot(xi2, xi0))
    zeta = float(np.dot(zeta_vector, r0))
    eta0 = np.cross(zeta_vector, xi0)
    eta1 = np.cross(zeta_vector, xi1)
    eta2 = np.cross(zeta_vector, xi2)
    log_terms = np.asarray(
        [
            _safe_log_ratio(r1_len + r2_len + s0_len, r1_len + r2_len - s0_len),
            _safe_log_ratio(r2_len + r0_len + s1_len, r2_len + r0_len - s1_len),
            _safe_log_ratio(r0_len + r1_len + s2_len, r0_len + r1_len - s2_len),
        ],
        dtype=float,
    )
    gamma0 = np.asarray([1.0, c01, c20], dtype=float)
    gamma1 = np.asarray([c01, 1.0, c12], dtype=float)
    gamma2 = np.asarray([c20, c12, 1.0], dtype=float)
    sign_zeta = -1.0 if zeta < 0.0 else (1.0 if zeta > 0.0 else 0.0)
    angle_pm = angle * sign_zeta
    denominator_factor = 1.0 / (8.0 * np.pi * area)
    with np.errstate(invalid="ignore"):
        contributions = np.asarray(
            [
                s0_len
                * denominator_factor
                * (angle_pm * float(np.dot(eta0, r1)) - zeta * float(np.dot(gamma0, log_terms))),
                s1_len
                * denominator_factor
                * (angle_pm * float(np.dot(eta1, r2)) - zeta * float(np.dot(gamma1, log_terms))),
                s2_len
                * denominator_factor
                * (angle_pm * float(np.dot(eta2, r0)) - zeta * float(np.dot(gamma2, log_terms))),
            ],
            dtype=float,
        )
    contributions = np.where(np.isfinite(contributions), contributions, 0.0)
    if float(np.dot(zeta_vector, outward_surface_normal)) < 0.0:
        contributions = -contributions
    return contributions


def _boundary_node_solid_angles(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    angles = np.zeros(len(points), dtype=float)
    for simplex in simplices:
        for local_index, point_index in enumerate(simplex):
            other_indices = [index for index in range(4) if index != local_index]
            angles[int(point_index)] += abs(
                _triangle_space_angle(
                    points[int(point_index)],
                    points[int(simplex[other_indices[0]])],
                    points[int(simplex[other_indices[1]])],
                    points[int(simplex[other_indices[2]])],
                )
            )
    return angles
