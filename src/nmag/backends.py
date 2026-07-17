"""Backend selection for Nmag's optional acceleration and memory policies."""

from __future__ import annotations

import importlib
import os
from types import ModuleType

from .config import NmagConfig, RustKernel
from .resources import available_memory_bytes

RUST_ACCELERATOR_API_VERSION = 1
MEMORY_MODE_ENV = "NMAG_MEMORY_MODE"
MEMORY_MODES = frozenset({"auto", "low"})
AUTO_DENSE_MEMORY_FRACTION = 0.2

DEMAG_BEM_STORAGE_BACKEND_ENV = "NMAG_DEMAG_BEM_STORAGE_BACKEND"
DEMAG_BEM_STORAGE_BACKENDS = frozenset({"auto", "dense", "matrix-free"})
DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES_ENV = (
    "NMAG_DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES"
)
DEFAULT_DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES = 2048
DEMAG_FEM_MATRIX_BACKEND_ENV = "NMAG_DEMAG_FEM_MATRIX_BACKEND"
DEMAG_FEM_MATRIX_BACKENDS = frozenset({"auto", "dense", "sparse"})
DEMAG_FEM_AUTO_SPARSE_MIN_POINTS_ENV = "NMAG_DEMAG_FEM_AUTO_SPARSE_MIN_POINTS"
DEFAULT_DEMAG_FEM_AUTO_SPARSE_MIN_POINTS = 2048
DEMAG_LINEAR_SOLVER_BACKEND_ENV = "NMAG_DEMAG_LINEAR_SOLVER_BACKEND"
DEMAG_LINEAR_SOLVER_BACKENDS = frozenset({"auto", "numpy", "scipy"})
DEMAG_LINEAR_SOLVER_AUTO_SCIPY_MIN_SIZE_ENV = "NMAG_DEMAG_LINEAR_SOLVER_AUTO_SCIPY_MIN_SIZE"
DEFAULT_DEMAG_LINEAR_SOLVER_AUTO_SCIPY_MIN_SIZE = 128
DEMAG_SOLVE_CONDITION_DIAGNOSTICS_ENV = "NMAG_DEMAG_SOLVE_CONDITION_DIAGNOSTICS"
DEMAG_INTERNAL_TRACE_ENV = "NMAG_DEMAG_INTERNAL_TRACE"
DEMAG_DENSE_MAX_POINTS_ENV = "NMAG_DEMAG_DENSE_MAX_POINTS"
LEAST_SQUARES_RELATIVE_RESIDUAL_TOLERANCE = 1.0e-10

DEMAG_FEM_GEOMETRY_RUST_MIN_CELLS = 2048
DEMAG_FEM_ASSEMBLY_RUST_MIN_CELLS = 2048
DEMAG_NODAL_RECOVERY_RUST_MIN_CELLS = 2048
DEMAG_CELL_AVERAGE_RUST_MIN_CELLS = 2048
LLG_RUST_MIN_POINTS = 2048


def _default_config() -> NmagConfig:
    """Return the environment-derived default for standalone helper calls."""

    return NmagConfig.from_environment()


def _policy(config: NmagConfig | None) -> NmagConfig:
    return _default_config() if config is None else config


def _rust_accelerator_available() -> bool:
    try:
        accelerator = importlib.import_module("nmag_accel")
    except ImportError:
        return False
    return getattr(accelerator, "API_VERSION", None) == RUST_ACCELERATOR_API_VERSION


def _load_rust_accelerator(required_by: str = "NmagConfig.accelerator") -> ModuleType:
    try:
        accelerator = importlib.import_module("nmag_accel")
    except ImportError as exc:
        raise RuntimeError(
            f"{required_by}=rust requires the optional nmag_accel extension. Build it with "
            "`maturin develop --release --manifest-path rust/nmag_accel/Cargo.toml`."
        ) from exc

    actual_version = getattr(accelerator, "API_VERSION", None)
    if actual_version != RUST_ACCELERATOR_API_VERSION:
        raise RuntimeError(
            f"{required_by}=rust found an incompatible nmag_accel extension "
            f"(API_VERSION={actual_version!r}, expected {RUST_ACCELERATOR_API_VERSION}). "
            "Rebuild it with `maturin develop --release --manifest-path "
            "rust/nmag_accel/Cargo.toml`."
        )
    return accelerator


def _select_rust_backend(
    *,
    config: NmagConfig | None,
    kernel: RustKernel,
    fallback: str,
    item_count: int | None = None,
    rust_min_items: int | None = None,
) -> str:
    """Resolve one optional Rust kernel without process-global selectors."""

    mode = _policy(config).accelerator_mode_for(kernel)
    if mode == "off":
        return fallback
    if mode == "rust":
        _load_rust_accelerator(f"NmagConfig.accelerator[{kernel.value!r}]")
        return "rust"
    if not _rust_accelerator_available():
        return fallback
    if (
        item_count is not None
        and rust_min_items is not None
        and item_count < rust_min_items
    ):
        return fallback
    return "rust"


def _rust_selection_metadata(
    *,
    config: NmagConfig | None,
    kernel: RustKernel,
    selected_backend: str,
    fallback: str,
    item_count: int | None = None,
    rust_min_items: int | None = None,
) -> dict[str, object]:
    policy = _policy(config)
    requested = policy.accelerator_mode_for(kernel)
    available = _rust_accelerator_available()
    if requested == "off":
        reason = "accelerator_off"
    elif requested == "rust":
        reason = "explicit_rust_request"
    elif not available:
        reason = "rust_accelerator_unavailable"
    elif rust_min_items is not None and item_count is not None and item_count < rust_min_items:
        reason = "item_count_below_rust_min"
    elif selected_backend == fallback:
        reason = "auto_fallback"
    else:
        reason = "rust_accelerator_available"
    metadata: dict[str, object] = {
        "requested": requested,
        "selected": selected_backend,
        "rust_available": available,
        "selection_reason": reason,
    }
    if item_count is not None:
        metadata["item_count"] = int(item_count)
    if rust_min_items is not None:
        metadata["rust_min_items"] = int(rust_min_items)
    return metadata


def _selected_integrator_backend(config: NmagConfig | None = None) -> str:
    policy = _policy(config)
    backend = policy.integrator_backend
    if backend == "diffsol" and _selected_memory_mode() == "low":
        raise ValueError(
            "NmagConfig(integrator_backend='diffsol') constructs a dense affine operator "
            f"and is incompatible with {MEMORY_MODE_ENV}=low; use scipy instead."
        )
    if backend == "diffsol" and policy.accelerator == "off":
        raise ValueError(
            "NmagConfig(integrator_backend='diffsol') requires the Rust accelerator; "
            "accelerator='off' disables it."
        )
    return backend


def _selected_memory_mode() -> str:
    mode = os.environ.get(MEMORY_MODE_ENV, "auto").strip().lower()
    if mode not in MEMORY_MODES:
        choices = ", ".join(sorted(MEMORY_MODES))
        raise ValueError(f"Unsupported {MEMORY_MODE_ENV}={mode!r}; choose one of {choices}.")
    return mode


def _auto_prefers_low_memory_storage(
    *, estimated_dense_bytes: int, item_count: int | None, fallback_min_items: int
) -> bool:
    available = available_memory_bytes()
    if available is not None:
        return estimated_dense_bytes > available * AUTO_DENSE_MEMORY_FRACTION
    return item_count is not None and item_count >= fallback_min_items


def _selected_lindholm_bem_backend(config: NmagConfig | None = None) -> str:
    return _select_rust_backend(
        config=config, kernel=RustKernel.LINDHOLM_BEM, fallback="numba"
    )


def _lindholm_bem_backend_selection_metadata(config: NmagConfig | None = None) -> dict[str, object]:
    return _rust_selection_metadata(
        config=config,
        kernel=RustKernel.LINDHOLM_BEM,
        selected_backend=_selected_lindholm_bem_backend(config),
        fallback="numba",
    )


def _selected_probe_geometry_backend(config: NmagConfig | None = None) -> str:
    return _select_rust_backend(
        config=config, kernel=RustKernel.PROBE_GEOMETRY, fallback="python"
    )


def _probe_geometry_backend_selection_metadata(config: NmagConfig | None = None) -> dict[str, object]:
    return _rust_selection_metadata(
        config=config,
        kernel=RustKernel.PROBE_GEOMETRY,
        selected_backend=_selected_probe_geometry_backend(config),
        fallback="python",
    )


def _selected_demag_fem_geometry_backend(
    cell_count: int | None = None, config: NmagConfig | None = None
) -> str:
    return _select_rust_backend(
        config=config,
        kernel=RustKernel.FEM_GEOMETRY,
        fallback="python",
        item_count=cell_count,
        rust_min_items=DEMAG_FEM_GEOMETRY_RUST_MIN_CELLS,
    )


def _demag_fem_geometry_backend_selection_metadata(
    cell_count: int | None = None, config: NmagConfig | None = None
) -> dict[str, object]:
    return _rust_selection_metadata(
        config=config,
        kernel=RustKernel.FEM_GEOMETRY,
        selected_backend=_selected_demag_fem_geometry_backend(cell_count, config),
        fallback="python",
        item_count=cell_count,
        rust_min_items=DEMAG_FEM_GEOMETRY_RUST_MIN_CELLS,
    )


def _selected_demag_boundary_face_backend(config: NmagConfig | None = None) -> str:
    return _select_rust_backend(
        config=config, kernel=RustKernel.BOUNDARY_FACES, fallback="python"
    )


def _demag_boundary_face_backend_selection_metadata(
    config: NmagConfig | None = None,
) -> dict[str, object]:
    return _rust_selection_metadata(
        config=config,
        kernel=RustKernel.BOUNDARY_FACES,
        selected_backend=_selected_demag_boundary_face_backend(config),
        fallback="python",
    )


def _selected_demag_fem_assembly_backend(
    cell_count: int | None = None, config: NmagConfig | None = None
) -> str:
    return _select_rust_backend(
        config=config,
        kernel=RustKernel.FEM_ASSEMBLY,
        fallback="python",
        item_count=cell_count,
        rust_min_items=DEMAG_FEM_ASSEMBLY_RUST_MIN_CELLS,
    )


def _demag_fem_assembly_backend_selection_metadata(
    cell_count: int | None = None, config: NmagConfig | None = None
) -> dict[str, object]:
    return _rust_selection_metadata(
        config=config,
        kernel=RustKernel.FEM_ASSEMBLY,
        selected_backend=_selected_demag_fem_assembly_backend(cell_count, config),
        fallback="python",
        item_count=cell_count,
        rust_min_items=DEMAG_FEM_ASSEMBLY_RUST_MIN_CELLS,
    )


def _selected_demag_nodal_recovery_backend(
    cell_count: int | None = None, config: NmagConfig | None = None
) -> str:
    return _select_rust_backend(
        config=config,
        kernel=RustKernel.NODAL_RECOVERY,
        fallback="python",
        item_count=cell_count,
        rust_min_items=DEMAG_NODAL_RECOVERY_RUST_MIN_CELLS,
    )


def _demag_nodal_recovery_backend_selection_metadata(
    cell_count: int | None = None, config: NmagConfig | None = None
) -> dict[str, object]:
    return _rust_selection_metadata(
        config=config,
        kernel=RustKernel.NODAL_RECOVERY,
        selected_backend=_selected_demag_nodal_recovery_backend(cell_count, config),
        fallback="python",
        item_count=cell_count,
        rust_min_items=DEMAG_NODAL_RECOVERY_RUST_MIN_CELLS,
    )


def _selected_demag_cell_average_backend(
    cell_count: int | None = None, config: NmagConfig | None = None
) -> str:
    return _select_rust_backend(
        config=config,
        kernel=RustKernel.CELL_AVERAGE,
        fallback="python",
        item_count=cell_count,
        rust_min_items=DEMAG_CELL_AVERAGE_RUST_MIN_CELLS,
    )


def _demag_cell_average_backend_selection_metadata(
    cell_count: int | None = None, config: NmagConfig | None = None
) -> dict[str, object]:
    return _rust_selection_metadata(
        config=config,
        kernel=RustKernel.CELL_AVERAGE,
        selected_backend=_selected_demag_cell_average_backend(cell_count, config),
        fallback="python",
        item_count=cell_count,
        rust_min_items=DEMAG_CELL_AVERAGE_RUST_MIN_CELLS,
    )


def _selected_llg_backend(
    point_count: int | None = None, config: NmagConfig | None = None
) -> str:
    return _select_rust_backend(
        config=config,
        kernel=RustKernel.LLG,
        fallback="python",
        item_count=point_count,
        rust_min_items=LLG_RUST_MIN_POINTS,
    )


def _llg_backend_selection_metadata(
    point_count: int | None = None, config: NmagConfig | None = None
) -> dict[str, object]:
    return _rust_selection_metadata(
        config=config,
        kernel=RustKernel.LLG,
        selected_backend=_selected_llg_backend(point_count, config),
        fallback="python",
        item_count=point_count,
        rust_min_items=LLG_RUST_MIN_POINTS,
    )


def _selected_maxangle_backend(config: NmagConfig | None = None) -> str:
    return _select_rust_backend(
        config=config, kernel=RustKernel.MAXANGLE, fallback="python"
    )


def _maxangle_backend_selection_metadata(config: NmagConfig | None = None) -> dict[str, object]:
    return _rust_selection_metadata(
        config=config,
        kernel=RustKernel.MAXANGLE,
        selected_backend=_selected_maxangle_backend(config),
        fallback="python",
    )


def _demag_bem_auto_matrix_free_min_boundary_nodes() -> int:
    return _positive_environment_int(
        DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES_ENV,
        DEFAULT_DEMAG_BEM_AUTO_MATRIX_FREE_MIN_BOUNDARY_NODES,
        "boundary-node count",
    )


def _demag_fem_auto_sparse_min_points() -> int:
    return _positive_environment_int(
        DEMAG_FEM_AUTO_SPARSE_MIN_POINTS_ENV,
        DEFAULT_DEMAG_FEM_AUTO_SPARSE_MIN_POINTS,
        "point count",
    )


def _demag_linear_solver_auto_scipy_min_size() -> int:
    return _positive_environment_int(
        DEMAG_LINEAR_SOLVER_AUTO_SCIPY_MIN_SIZE_ENV,
        DEFAULT_DEMAG_LINEAR_SOLVER_AUTO_SCIPY_MIN_SIZE,
        "dense dimension",
    )


def _positive_environment_int(name: str, default: int, description: str) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        minimum = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported {name}={raw_value!r}; expected a positive integer {description}."
        ) from exc
    if minimum <= 0:
        raise ValueError(
            f"Unsupported {name}={raw_value!r}; expected a positive integer {description}."
        )
    return minimum


def _selected_demag_bem_storage_backend(boundary_node_count: int | None = None) -> str:
    backend = os.environ.get(DEMAG_BEM_STORAGE_BACKEND_ENV, "auto").strip().lower()
    if backend not in DEMAG_BEM_STORAGE_BACKENDS:
        choices = ", ".join(sorted(DEMAG_BEM_STORAGE_BACKENDS))
        raise ValueError(f"Unsupported {DEMAG_BEM_STORAGE_BACKEND_ENV}={backend!r}; choose one of {choices}.")
    if backend != "auto":
        return backend
    if _selected_memory_mode() == "low":
        return "matrix-free"
    count = boundary_node_count or 0
    if _auto_prefers_low_memory_storage(
        estimated_dense_bytes=count * count * 8,
        item_count=boundary_node_count,
        fallback_min_items=_demag_bem_auto_matrix_free_min_boundary_nodes(),
    ):
        return "matrix-free"
    return "dense"


def _selected_demag_fem_matrix_backend(point_count: int | None = None) -> str:
    backend = os.environ.get(DEMAG_FEM_MATRIX_BACKEND_ENV, "auto").strip().lower()
    if backend not in DEMAG_FEM_MATRIX_BACKENDS:
        choices = ", ".join(sorted(DEMAG_FEM_MATRIX_BACKENDS))
        raise ValueError(f"Unsupported {DEMAG_FEM_MATRIX_BACKEND_ENV}={backend!r}; choose one of {choices}.")
    if backend != "auto":
        return backend
    if _selected_memory_mode() == "low":
        return "sparse"
    count = point_count or 0
    if _auto_prefers_low_memory_storage(
        estimated_dense_bytes=count * count * 8 * 4,
        item_count=point_count,
        fallback_min_items=_demag_fem_auto_sparse_min_points(),
    ):
        return "sparse"
    return "dense"


def _selected_demag_linear_solver_backend(dense_dimension: int | None = None) -> str:
    backend = os.environ.get(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "auto").strip().lower()
    if backend not in DEMAG_LINEAR_SOLVER_BACKENDS:
        choices = ", ".join(sorted(DEMAG_LINEAR_SOLVER_BACKENDS))
        raise ValueError(f"Unsupported {DEMAG_LINEAR_SOLVER_BACKEND_ENV}={backend!r}; choose one of {choices}.")
    if backend != "auto":
        return backend
    if dense_dimension is not None and dense_dimension >= _demag_linear_solver_auto_scipy_min_size():
        return "scipy"
    return "numpy"


def _demag_linear_solver_backend_selection_metadata(
    dense_dimension: int | None = None,
) -> dict[str, object]:
    selected = _selected_demag_linear_solver_backend(dense_dimension)
    requested = os.environ.get(DEMAG_LINEAR_SOLVER_BACKEND_ENV, "auto").strip().lower()
    metadata: dict[str, object] = {"requested": requested, "selected": selected}
    if dense_dimension is not None:
        metadata["dense_dimension"] = int(dense_dimension)
    metadata["auto_scipy_min_size"] = _demag_linear_solver_auto_scipy_min_size()
    return metadata


def _demag_solve_condition_diagnostics_enabled() -> bool:
    return _boolean_environment(DEMAG_SOLVE_CONDITION_DIAGNOSTICS_ENV)


def _demag_internal_trace_enabled() -> bool:
    return _boolean_environment(DEMAG_INTERNAL_TRACE_ENV)


def _boolean_environment(name: str) -> bool:
    raw_value = os.environ.get(name, "0")
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Unsupported {name}={raw_value!r}; choose a boolean value.")
