from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ....backends import _load_rust_accelerator, _selected_lindholm_bem_backend
from ....demag.bem_operator import HierarchicalLindholmBemOperator
from ....resources import available_memory_bytes
from ...support import _simulation_compatibility_binding

logger = logging.getLogger(__name__)


def build_hierarchical_lindholm_operator(
    simulation: Any,
    points: np.ndarray,
    simplices: np.ndarray,
    boundary_faces: list[tuple[int, tuple[int, int, int]]],
) -> tuple[np.ndarray, Any, str | None]:
    config = simulation.config
    if _selected_lindholm_bem_backend(config) != "rust":
        boundary_nodes, operator = simulation._build_lindholm_bem_operator(
            points,
            simplices,
            boundary_faces,
        )
        return boundary_nodes, operator, "Rust accelerator is unavailable or disabled"

    rust_accelerator = _simulation_compatibility_binding(
        "_load_rust_accelerator",
        _load_rust_accelerator,
    )()
    if not hasattr(rust_accelerator, "build_lindholm_hmatrix"):
        raise RuntimeError(
            "The installed nmag_accel does not provide hierarchical Lindholm BEM; "
            "rebuild it with ./scripts/build-accelerator.sh --release."
        )
    boundary_nodes, local_index_by_point, face_nodes = simulation._lindholm_bem_boundary_index(
        points,
        boundary_faces,
    )
    hierarchical = config.hierarchical_bem
    available = available_memory_bytes()
    memory_budget = (
        int(available * hierarchical.memory_fraction)
        if available is not None
        else np.iinfo(np.uintp).max
    )
    try:
        rust_operator = rust_accelerator.build_lindholm_hmatrix(
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(simplices, dtype=np.int64),
            np.ascontiguousarray(face_nodes, dtype=np.int64),
            np.ascontiguousarray(boundary_nodes, dtype=np.int64),
            np.ascontiguousarray(local_index_by_point, dtype=np.int64),
            relative_tolerance=hierarchical.relative_tolerance,
            admissibility_eta=hierarchical.admissibility_eta,
            leaf_size=hierarchical.leaf_size,
            max_rank=hierarchical.max_rank,
            validation_vectors=hierarchical.validation_vectors,
            validation_rows=hierarchical.validation_rows,
            memory_budget_bytes=memory_budget,
        )
    except (MemoryError, ValueError) as exc:
        logger.info("Hierarchical Lindholm BEM rejected; using exact matrix-free BEM: %s", exc)
        boundary_nodes, operator = simulation._build_lindholm_bem_operator(
            points,
            simplices,
            boundary_faces,
        )
        return boundary_nodes, operator, str(exc)
    return boundary_nodes, HierarchicalLindholmBemOperator(rust_operator), None
