from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backends import (
    DEMAG_DENSE_MAX_POINTS_ENV,
    LEAST_SQUARES_RELATIVE_RESIDUAL_TOLERANCE,
    _demag_solve_condition_diagnostics_enabled,
    _selected_demag_linear_solver_backend,
)

LOGGER = logging.getLogger(__name__)
_scipy_linalg_cache: Any | None = None


@dataclass(frozen=True, slots=True)
class ScipyLUFactorization:
    """Reusable LU factorization for a geometry-dependent dense system."""

    lu: np.ndarray
    pivots: np.ndarray

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(
            _scipy_linalg().lu_solve(
                (self.lu, self.pivots),
                np.asarray(rhs, dtype=float),
                check_finite=False,
            ),
            dtype=float,
        )


def _factorize_scipy(matrix: np.ndarray) -> ScipyLUFactorization:
    lu, pivots = _scipy_linalg().lu_factor(
        np.asarray(matrix, dtype=float),
        check_finite=False,
    )
    return ScipyLUFactorization(np.asarray(lu, dtype=float), np.asarray(pivots, dtype=int))


def _factorize_gauge_fixed(matrix: np.ndarray) -> ScipyLUFactorization:
    constraint = np.ones(matrix.shape[0], dtype=float)
    augmented = np.block(
        [
            [matrix, constraint[:, np.newaxis]],
            [constraint[np.newaxis, :], np.zeros((1, 1), dtype=float)],
        ]
    )
    return _factorize_scipy(augmented)


def __getattr__(name: str) -> Any:
    if name == "_SCIPY_LINALG":
        return _scipy_linalg_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _scipy_linalg() -> Any:
    global _scipy_linalg_cache
    if _scipy_linalg_cache is None:
        from scipy import linalg

        _scipy_linalg_cache = linalg
    return _scipy_linalg_cache


def _demag_dense_max_points() -> int | None:
    raw_value = os.environ.get(DEMAG_DENSE_MAX_POINTS_ENV)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{DEMAG_DENSE_MAX_POINTS_ENV} must be a positive integer, got {raw_value!r}."
        ) from error
    if value < 1:
        raise ValueError(
            f"{DEMAG_DENSE_MAX_POINTS_ENV} must be a positive integer, got {raw_value!r}."
        )
    return value


def _validate_dense_demag_size(point_count: int) -> None:
    maximum = _demag_dense_max_points()
    if maximum is None or point_count <= maximum:
        return
    estimated_gib = point_count * point_count * 8 / (1024**3)
    raise MemoryError(
        "Dense demag FEM assembly refused a mesh with "
        f"{point_count} points (one dense matrix is approximately {estimated_gib:.2f} GiB). "
        f"The current safety limit is {maximum}; change {DEMAG_DENSE_MAX_POINTS_ENV} only "
        "when the required memory is known to be available."
    )


def _scipy_sparse_modules() -> tuple[Any, Any]:
    from scipy import sparse
    from scipy.sparse import linalg as sparse_linalg

    return sparse, sparse_linalg


def _is_sparse_matrix(matrix: Any) -> bool:
    sparse, _sparse_linalg = _scipy_sparse_modules()
    return bool(sparse.issparse(matrix))


def _solve_sparse_gauge_fixed(
    matrix: Any,
    rhs: np.ndarray,
    *,
    record_timing: Callable[[str, float], None] | None = None,
    timing_prefix: str = "linear_solver:gauge",
) -> np.ndarray:
    _sparse, sparse_linalg = _scipy_sparse_modules()
    started = time.perf_counter()
    point_count = int(matrix.shape[0])
    diagonal = np.asarray(matrix.diagonal(), dtype=float)
    matrix_scale = float(np.max(np.abs(diagonal))) if diagonal.size else 1.0
    if matrix_scale <= np.finfo(float).tiny:
        return np.zeros_like(rhs, dtype=float)

    scaled_matrix = matrix / matrix_scale
    scaled_rhs = np.asarray(rhs, dtype=float) / matrix_scale

    def apply_constrained_matrix(vector: np.ndarray) -> np.ndarray:
        return np.asarray(scaled_matrix @ vector, dtype=float) + np.mean(vector)

    constrained = sparse_linalg.LinearOperator(
        matrix.shape,
        matvec=apply_constrained_matrix,
        dtype=np.dtype(float),
    )
    inverse_diagonal = 1.0 / (diagonal / matrix_scale + 1.0 / max(point_count, 1))

    def apply_preconditioner(vector: np.ndarray) -> np.ndarray:
        return inverse_diagonal * vector

    preconditioner = sparse_linalg.LinearOperator(
        matrix.shape,
        matvec=apply_preconditioner,
        dtype=np.dtype(float),
    )
    solution, status = sparse_linalg.cg(
        constrained,
        scaled_rhs,
        rtol=1.0e-12,
        atol=1.0e-14,
        maxiter=max(1000, 10 * point_count),
        M=preconditioner,
    )
    if int(status) != 0:
        result = sparse_linalg.lsmr(
            constrained,
            scaled_rhs,
            atol=1.0e-12,
            btol=1.0e-12,
            conlim=0.0,
            maxiter=max(1000, 10 * point_count),
        )
        solution = result[0]
        if int(result[1]) not in {0, 1, 2}:
            raise np.linalg.LinAlgError(
                "Sparse gauge solve did not converge "
                f"(CG status {int(status)}, LSMR status {int(result[1])})."
            )
    solution = np.asarray(solution, dtype=float)
    solution -= np.mean(solution)
    if record_timing is not None:
        record_timing(f"{timing_prefix}:sparse_iterative", time.perf_counter() - started)
    return solution


def _solve_sparse_linear_system(
    matrix: Any,
    rhs: np.ndarray,
    *,
    record_timing: Callable[[str, float], None] | None = None,
    timing_prefix: str = "linear_solver",
) -> np.ndarray:
    _sparse, sparse_linalg = _scipy_sparse_modules()
    started = time.perf_counter()
    diagonal = np.asarray(matrix.diagonal(), dtype=float)
    matrix_scale = float(np.max(np.abs(diagonal))) if diagonal.size else 1.0
    if matrix_scale <= np.finfo(float).tiny:
        return np.zeros_like(rhs, dtype=float)
    scaled_matrix = matrix / matrix_scale
    scaled_rhs = np.asarray(rhs, dtype=float) / matrix_scale
    scaled_diagonal = diagonal / matrix_scale
    inverse_diagonal = np.ones_like(scaled_diagonal)
    nonzero = np.abs(scaled_diagonal) > np.finfo(float).tiny
    inverse_diagonal[nonzero] = 1.0 / scaled_diagonal[nonzero]

    def apply_preconditioner(vector: np.ndarray) -> np.ndarray:
        return inverse_diagonal * vector

    preconditioner = sparse_linalg.LinearOperator(
        matrix.shape,
        matvec=apply_preconditioner,
        dtype=np.dtype(float),
    )
    solution, status = sparse_linalg.cg(
        scaled_matrix,
        scaled_rhs,
        rtol=1.0e-12,
        atol=0.0,
        maxiter=max(1000, 10 * int(matrix.shape[0])),
        M=preconditioner,
    )
    if int(status) != 0:
        fallback = sparse_linalg.lsmr(
            scaled_matrix,
            scaled_rhs,
            atol=1.0e-12,
            btol=1.0e-12,
            maxiter=max(1000, 10 * int(matrix.shape[0])),
        )
        solution = fallback[0]
        if int(fallback[1]) not in {0, 1, 2}:
            raise np.linalg.LinAlgError(
                "Sparse linear solve did not converge "
                f"(CG status {int(status)}, LSMR status {int(fallback[1])})."
            )
    if record_timing is not None:
        record_timing(f"{timing_prefix}:sparse_iterative", time.perf_counter() - started)
    return np.asarray(solution, dtype=float)


def _validated_least_squares_solution(
    matrix: np.ndarray,
    rhs: np.ndarray,
    solution: np.ndarray,
) -> np.ndarray:
    residual = matrix @ solution - rhs
    rhs_norm = float(np.linalg.norm(rhs))
    residual_norm = float(np.linalg.norm(residual))
    relative_residual = residual_norm / rhs_norm if rhs_norm else residual_norm
    if not np.isfinite(relative_residual) or (
        relative_residual > LEAST_SQUARES_RELATIVE_RESIDUAL_TOLERANCE
    ):
        raise np.linalg.LinAlgError(
            "Singular demag system has no accurate least-squares solution: "
            f"relative residual={relative_residual:.3e}."
        )
    LOGGER.warning(
        "Using a least-squares solution for a singular demag system (relative residual %.3e).",
        relative_residual,
    )
    return solution


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _l2_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values.ravel())) if values.size else 0.0


def _safe_relative_residual(numerator: np.ndarray, scale: np.ndarray) -> float:
    numerator_l2 = _l2_norm(numerator)
    scale_l2 = _l2_norm(scale)
    if scale_l2 == 0.0:
        return 0.0 if numerator_l2 == 0.0 else float("inf")
    return float(numerator_l2 / scale_l2)


def _dense_condition_number(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.linalg.cond(np.asarray(matrix, dtype=float)))


def _solve_gauge_fixed(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    record_timing: Callable[[str, float], None] | None = None,
    timing_prefix: str = "linear_solver:gauge",
) -> np.ndarray:
    augmented, augmented_rhs = _gauge_augmented_system(matrix, rhs, record_timing, timing_prefix)
    return _solve_dense_system(
        augmented,
        augmented_rhs,
        record_timing=record_timing,
        timing_prefix=timing_prefix,
    )[:-1]


def _gauge_augmented_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
    record_timing: Callable[[str, float], None] | None,
    timing_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    started = time.perf_counter()
    try:
        constraint = np.ones(len(rhs), dtype=float)
        augmented = np.block(
            [
                [matrix, constraint[:, np.newaxis]],
                [constraint[np.newaxis, :], np.zeros((1, 1), dtype=float)],
            ]
        )
        augmented_rhs = np.concatenate([rhs, [0.0]])
    finally:
        if record_timing is not None:
            record_timing(f"{timing_prefix}:setup", time.perf_counter() - started)
    return augmented, augmented_rhs


def _solve_dense_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    record_timing: Callable[[str, float], None] | None,
    timing_prefix: str,
) -> np.ndarray:
    backend = _selected_demag_linear_solver_backend(int(matrix.shape[0]))
    if backend == "scipy":
        return _solve_dense_scipy(matrix, rhs, record_timing, timing_prefix)
    return _solve_dense_numpy(matrix, rhs, record_timing, timing_prefix)


def _solve_dense_scipy(
    matrix: np.ndarray,
    rhs: np.ndarray,
    record_timing: Callable[[str, float], None] | None,
    timing_prefix: str,
) -> np.ndarray:
    started = time.perf_counter()
    linalg = _scipy_linalg()
    if record_timing is not None:
        record_timing(f"{timing_prefix}:scipy_import", time.perf_counter() - started)
    started = time.perf_counter()
    try:
        result = linalg.solve(matrix, rhs, assume_a="sym", check_finite=False)
        if record_timing is not None:
            record_timing(f"{timing_prefix}:scipy_solve", time.perf_counter() - started)
        return result
    except np.linalg.LinAlgError:
        if record_timing is not None:
            record_timing(
                f"{timing_prefix}:scipy_solve_failed",
                time.perf_counter() - started,
            )
        started = time.perf_counter()
        result = _validated_least_squares_solution(
            matrix,
            rhs,
            linalg.lstsq(matrix, rhs, check_finite=False)[0],
        )
        if record_timing is not None:
            record_timing(f"{timing_prefix}:scipy_lstsq", time.perf_counter() - started)
        return result


def _solve_linear_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    record_timing: Callable[[str, float], None] | None = None,
    timing_prefix: str = "linear_solver",
) -> np.ndarray:
    return _solve_dense_system(
        matrix,
        rhs,
        record_timing=record_timing,
        timing_prefix=timing_prefix,
    )


def _solve_dense_numpy(
    matrix: np.ndarray,
    rhs: np.ndarray,
    record_timing: Callable[[str, float], None] | None,
    timing_prefix: str,
) -> np.ndarray:
    started = time.perf_counter()
    try:
        result = np.linalg.solve(matrix, rhs)
        if record_timing is not None:
            record_timing(f"{timing_prefix}:numpy_solve", time.perf_counter() - started)
        return result
    except np.linalg.LinAlgError:
        if record_timing is not None:
            record_timing(
                f"{timing_prefix}:numpy_solve_failed",
                time.perf_counter() - started,
            )
        started = time.perf_counter()
        result = _validated_least_squares_solution(
            matrix,
            rhs,
            np.linalg.lstsq(matrix, rhs, rcond=None)[0],
        )
        if record_timing is not None:
            record_timing(f"{timing_prefix}:numpy_lstsq", time.perf_counter() - started)
        return result


def _demag_solve_diagnostics(
    stiffness: Any,
    divergence: np.ndarray,
    boundary_nodes: np.ndarray,
    phi1: np.ndarray,
    phi2_boundary: np.ndarray,
    phi2: np.ndarray,
    phi: np.ndarray,
) -> dict[str, float | int]:
    phi1_residual = np.asarray(stiffness @ phi1, dtype=float) - divergence
    phi1_centered_residual = (
        phi1_residual - np.mean(phi1_residual) if phi1_residual.size else phi1_residual
    )
    divergence_centered = divergence - np.mean(divergence) if divergence.size else divergence

    boundary_mask = np.zeros(len(phi), dtype=bool)
    boundary_mask[boundary_nodes] = True
    interior_nodes = np.flatnonzero(~boundary_mask)
    interior_rows = (
        stiffness[interior_nodes]
        if _is_sparse_matrix(stiffness)
        else stiffness[np.ix_(interior_nodes, np.arange(len(phi)))]
    )
    dirichlet_residual = (
        np.asarray(interior_rows @ phi2, dtype=float)
        if interior_nodes.size
        else np.empty(0, dtype=float)
    )
    dirichlet_scale = (
        np.asarray(np.abs(interior_rows) @ np.abs(phi2), dtype=float)
        if interior_nodes.size
        else np.empty(0, dtype=float)
    )

    diagnostics: dict[str, float | int] = {
        "node_count": int(len(phi)),
        "boundary_node_count": int(len(boundary_nodes)),
        "gauge_augmented_size": int(len(divergence) + 1),
        "divergence_sum_abs": abs(float(np.sum(divergence))),
        "phi1_gauge_sum_abs": abs(float(np.sum(phi1))),
        "phi1_centered_residual_max_abs": _max_abs(phi1_centered_residual),
        "phi1_centered_residual_l2": _l2_norm(phi1_centered_residual),
        "phi1_centered_residual_relative_l2": _safe_relative_residual(
            phi1_centered_residual,
            divergence_centered,
        ),
        "dirichlet_interior_node_count": int(interior_nodes.size),
        "dirichlet_residual_max_abs": _max_abs(dirichlet_residual),
        "dirichlet_residual_l2": _l2_norm(dirichlet_residual),
        "dirichlet_residual_relative_l2": _safe_relative_residual(
            dirichlet_residual,
            dirichlet_scale,
        ),
        "phi1_max_abs": _max_abs(phi1),
        "phi2_boundary_max_abs": _max_abs(phi2_boundary),
        "phi2_max_abs": _max_abs(phi2),
        "phi_max_abs": _max_abs(phi),
    }
    if _demag_solve_condition_diagnostics_enabled():
        if _is_sparse_matrix(stiffness):
            diagnostics["condition_numbers_skipped_for_sparse_storage"] = 1
        else:
            constraint = np.ones(len(divergence), dtype=float)
            augmented_stiffness = np.block(
                [
                    [stiffness, constraint[:, np.newaxis]],
                    [constraint[np.newaxis, :], np.zeros((1, 1), dtype=float)],
                ]
            )
            diagnostics["gauge_augmented_condition_number"] = _dense_condition_number(
                augmented_stiffness,
            )
            diagnostics["stiffness_condition_number"] = _dense_condition_number(stiffness)
            diagnostics["dirichlet_interior_condition_number"] = (
                _dense_condition_number(stiffness[np.ix_(interior_nodes, interior_nodes)])
                if interior_nodes.size
                else 0.0
            )
    return diagnostics
