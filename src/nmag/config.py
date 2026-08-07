"""Typed runtime configuration for supported Nmag Python workflows."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

AcceleratorMode = Literal["auto", "off", "rust"]
OutputPolicy = Literal["error", "replace", "append"]
IntegratorBackend = Literal["scipy", "diffsol"]
DemagBemStorage = Literal["auto", "dense", "hierarchical", "matrix-free"]

ACCELERATOR_ENV = "NMAG_ACCELERATOR"
DEMAG_BEM_STORAGE_ENV = "NMAG_DEMAG_BEM_STORAGE_BACKEND"


@dataclass(frozen=True, slots=True)
class HierarchicalBemConfig:
    """Control accuracy and resources for compressed Lindholm BEM.

    Attributes:
        relative_tolerance: Requested relative compression/certification error.
        admissibility_eta: Geometric separation threshold for low-rank blocks.
        leaf_size: Maximum cluster leaf size.
        max_rank: Maximum accepted low-rank block rank.
        validation_vectors: Random vectors used to certify completed action.
        validation_rows: Exact rows sampled during certification.
        memory_fraction: Fraction of currently available memory allowed for the
            completed hierarchy.
    """

    relative_tolerance: float = 1.0e-6
    admissibility_eta: float = 2.0
    leaf_size: int = 32
    max_rank: int = 128
    validation_vectors: int = 4
    validation_rows: int = 64
    memory_fraction: float = 0.20

    def __post_init__(self) -> None:
        for name in ("relative_tolerance", "admissibility_eta", "memory_fraction"):
            raw_value = getattr(self, name)
            if type(raw_value) is bool or not isinstance(raw_value, Real):
                raise TypeError(f"{name} must be a real number.")
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.memory_fraction > 1.0:
            raise ValueError("memory_fraction must not exceed 1.0.")
        for name in ("leaf_size", "max_rank", "validation_vectors", "validation_rows"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


class RustKernel(str, Enum):
    """Rust-accelerated calculation families accepted as override keys."""

    LINDHOLM_BEM = "lindholm_bem"
    PROBE_GEOMETRY = "probe_geometry"
    FEM_GEOMETRY = "fem_geometry"
    BOUNDARY_FACES = "boundary_faces"
    FEM_ASSEMBLY = "fem_assembly"
    NODAL_RECOVERY = "nodal_recovery"
    CELL_AVERAGE = "cell_average"
    LLG = "llg"
    MAXANGLE = "maxangle"


def _validate_mode(value: object, *, field_name: str) -> AcceleratorMode:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string mode.")
    if value not in {"auto", "off", "rust"}:
        raise ValueError(f"{field_name} must be one of 'auto', 'off', or 'rust', got {value!r}.")
    return cast(AcceleratorMode, value)


def _empty_overrides() -> Mapping[RustKernel, AcceleratorMode]:
    return {}


@dataclass(frozen=True, slots=True)
class NmagConfig:
    """Define immutable output, acceleration, storage, and integrator policy.

    Attributes:
        default_name: Simulation name used when ``Simulation(name=...)`` is
            omitted.
        output_directory: Directory for NDT, HDF5, and default checkpoint files.
            The directory must exist before output is written.
        output_policy: ``"error"` protects old output, ``"replace"`` starts it
            again, and ``"append"`` validates and extends the NDT schema.
        accelerator: Global ``"auto"``, ``"off"``, or strict ``"rust"`` mode.
        accelerator_overrides: Per-:class:`RustKernel` mode overrides.
        integrator_backend: ``"scipy"`` or experimental ``"diffsol"``.
        demag_bem_storage: ``"auto"``, ``"dense"``, ``"hierarchical"``, or
            ``"matrix-free"`` boundary-operator storage.
        hierarchical_bem: Accuracy and resource settings for hierarchy builds.
    """

    default_name: str = "nmag_simulation"
    output_directory: Path = Path(".")
    output_policy: OutputPolicy = "error"
    accelerator: AcceleratorMode = "auto"
    accelerator_overrides: Mapping[RustKernel, AcceleratorMode] = field(
        default_factory=_empty_overrides
    )
    integrator_backend: IntegratorBackend = "scipy"
    demag_bem_storage: DemagBemStorage = "auto"
    hierarchical_bem: HierarchicalBemConfig = field(default_factory=HierarchicalBemConfig)

    def __post_init__(self) -> None:
        if not self.default_name:
            raise ValueError("default_name must not be empty.")
        if self.output_policy not in {"error", "replace", "append"}:
            raise ValueError("output_policy must be 'error', 'replace', or 'append'.")
        if self.integrator_backend not in {"scipy", "diffsol"}:
            raise ValueError("integrator_backend must be 'scipy' or 'diffsol'.")
        if self.demag_bem_storage not in {"auto", "dense", "hierarchical", "matrix-free"}:
            raise ValueError(
                "demag_bem_storage must be 'auto', 'dense', 'hierarchical', or 'matrix-free'."
            )
        if not isinstance(cast(object, self.hierarchical_bem), HierarchicalBemConfig):
            raise TypeError("hierarchical_bem must be a HierarchicalBemConfig instance.")
        object.__setattr__(self, "output_directory", Path(self.output_directory).expanduser())
        object.__setattr__(
            self,
            "accelerator",
            _validate_mode(self.accelerator, field_name="accelerator"),
        )

        validated_overrides: dict[RustKernel, AcceleratorMode] = {}
        raw_overrides = cast(Mapping[object, object], self.accelerator_overrides)
        for raw_kernel, raw_mode in raw_overrides.items():
            if not isinstance(raw_kernel, RustKernel):
                raise TypeError("accelerator_overrides keys must be RustKernel values.")
            validated_overrides[raw_kernel] = _validate_mode(
                raw_mode,
                field_name=f"accelerator_overrides[{raw_kernel.value!r}]",
            )
        object.__setattr__(
            self,
            "accelerator_overrides",
            MappingProxyType(validated_overrides),
        )

    @classmethod
    def from_environment(cls) -> NmagConfig:
        """Build default configuration from supported process-level selectors.

        Returns:
            Configuration using ``NMAG_ACCELERATOR`` and
            ``NMAG_DEMAG_BEM_STORAGE_BACKEND`` when set.
        """

        accelerator = os.environ.get(ACCELERATOR_ENV, "auto").strip().lower()
        bem_storage = os.environ.get(DEMAG_BEM_STORAGE_ENV, "auto").strip().lower()
        return cls(
            accelerator=_validate_mode(accelerator, field_name=ACCELERATOR_ENV),
            demag_bem_storage=cast(DemagBemStorage, bem_storage),
        )

    def accelerator_mode_for(self, kernel: RustKernel) -> AcceleratorMode:
        """Return a kernel override, falling back to the global accelerator mode."""

        return self.accelerator_overrides.get(kernel, self.accelerator)
