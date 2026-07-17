"""Typed runtime configuration for supported Nmag Python workflows."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

AcceleratorMode = Literal["auto", "off", "rust"]
OutputPolicy = Literal["error", "replace", "append"]
IntegratorBackend = Literal["scipy", "diffsol"]

ACCELERATOR_ENV = "NMAG_ACCELERATOR"


class RustKernel(str, Enum):
    """Rust-accelerated calculation families with optional overrides."""

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
    """Immutable configuration supplied to one :class:`nmag.Simulation`."""

    default_name: str = "nmag_simulation"
    output_directory: Path = Path(".")
    output_policy: OutputPolicy = "error"
    accelerator: AcceleratorMode = "auto"
    accelerator_overrides: Mapping[RustKernel, AcceleratorMode] = field(
        default_factory=_empty_overrides
    )
    integrator_backend: IntegratorBackend = "scipy"

    def __post_init__(self) -> None:
        if not self.default_name:
            raise ValueError("default_name must not be empty.")
        if self.output_policy not in {"error", "replace", "append"}:
            raise ValueError("output_policy must be 'error', 'replace', or 'append'.")
        if self.integrator_backend not in {"scipy", "diffsol"}:
            raise ValueError("integrator_backend must be 'scipy' or 'diffsol'.")
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
        """Build default configuration with the single supported environment override."""

        accelerator = os.environ.get(ACCELERATOR_ENV, "auto").strip().lower()
        return cls(accelerator=_validate_mode(accelerator, field_name=ACCELERATOR_ENV))

    def accelerator_mode_for(self, kernel: RustKernel) -> AcceleratorMode:
        """Return the configured mode for one accelerated calculation family."""

        return self.accelerator_overrides.get(kernel, self.accelerator)
