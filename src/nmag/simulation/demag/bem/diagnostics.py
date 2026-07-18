from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ....demag.bem_operator import (
    BemOperatorStats,
    HierarchicalLindholmBemOperator,
)

logger = logging.getLogger(__name__)


def bem_operator_stats(
    *,
    requested_backend: str,
    effective_backend: str,
    fallback_reason: str | None,
    operator: Any,
    boundary_node_count: int,
    boundary_faces: int,
    setup_seconds: float,
) -> BemOperatorStats:
    dense_equivalent = boundary_node_count**2 * np.dtype(np.float64).itemsize
    if isinstance(operator, HierarchicalLindholmBemOperator):
        rust_operator = operator.rust_operator
        dense_blocks = int(rust_operator.dense_blocks)
        low_rank_blocks = int(rust_operator.low_rank_blocks)
        maximum_rank = int(rust_operator.maximum_rank)
        mean_rank = float(rust_operator.mean_rank)
        sampled_error = float(rust_operator.sampled_relative_error)
        storage_bytes = operator.storage_bytes
    else:
        dense_blocks = 1 if isinstance(operator, np.ndarray) and operator.size else 0
        low_rank_blocks = 0
        maximum_rank = 0
        mean_rank = 0.0
        sampled_error = 0.0
        storage_bytes = (
            int(operator.nbytes) if isinstance(operator, np.ndarray) else operator.storage_bytes
        )
    return BemOperatorStats(
        requested_backend=requested_backend,
        effective_backend=effective_backend,
        fallback_reason=fallback_reason,
        boundary_nodes=boundary_node_count,
        boundary_faces=boundary_faces,
        setup_seconds=setup_seconds,
        storage_bytes=storage_bytes,
        dense_equivalent_bytes=dense_equivalent,
        dense_blocks=dense_blocks,
        low_rank_blocks=low_rank_blocks,
        maximum_rank=maximum_rank,
        mean_rank=mean_rank,
        sampled_relative_error=sampled_error,
    )


def log_bem_operator_selection(stats: BemOperatorStats, selected_backend: str) -> None:
    fallback = (
        f" after {selected_backend} fallback: {stats.fallback_reason}"
        if stats.fallback_reason
        else ""
    )
    logger.info(
        "Lindholm BEM selected %s storage%s (%.3f MiB, %.3f dense ratio)",
        stats.effective_backend,
        fallback,
        stats.storage_bytes / (1024 * 1024),
        stats.compression_ratio,
    )
