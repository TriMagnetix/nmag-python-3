"""Public Python 3 entrypoint for the nmag rewrite."""

from typing import Any

from anisotropy import (
    PredefinedAnisotropy,
    cubic_anisotropy,
    uniaxial_anisotropy,
    want_anisotropy,
)
from mag_material import MagMaterial
from si.physical import SI, Physical

from .config import ACCELERATOR_ENV, HierarchicalBemConfig, NmagConfig, RustKernel
from .demag.bem_operator import BemOperatorStats
from .dynamics import IntegratorConfig, IntegratorStats
from .parallel import ParallelRuntimeInfo, parallel_runtime_info
from .simulation import Simulation

_OUTPUT_EXPORTS = {"export_vtk", "resolve_snapshot"}

__all__ = [
    "MagMaterial",
    "NmagConfig",
    "HierarchicalBemConfig",
    "BemOperatorStats",
    "IntegratorConfig",
    "IntegratorStats",
    "ParallelRuntimeInfo",
    "Physical",
    "PredefinedAnisotropy",
    "SI",
    "Simulation",
    "RustKernel",
    "ACCELERATOR_ENV",
    "cubic_anisotropy",
    "parallel_runtime_info",
    "uniaxial_anisotropy",
    "want_anisotropy",
    "export_vtk",
    "resolve_snapshot",
]


def __getattr__(name: str) -> Any:
    if name in _OUTPUT_EXPORTS:
        from . import vtk_export

        value = getattr(vtk_export, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
