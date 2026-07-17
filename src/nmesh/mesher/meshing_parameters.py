from __future__ import annotations

import copy
from functools import partialmethod
from os import PathLike
from typing import Any, cast

from ..utils.constants import BOUNDARY_FUZZ, MIN_DIVISION_MAGNITUDE
from .meshing_defaults import (
    INTERNAL_TO_PUBLIC,
    PUBLIC_PARAMETER_SPECS,
    PUBLIC_PARAMETER_SPECS_BY_NAME,
    PUBLIC_TO_INTERNAL,
    MesherParameter,
    ParameterSpec,
    PointFate,
    SimplexRegion,
    _candidate_keys,
    _cast_numeric_parameter,
    default_boundary_node_force_fun,
    default_handle_point_density_fun,
    default_initial_relaxation_weight,
    default_relaxation_force_fun,
)
from .sectioned_config import SectionedConfig

__all__ = [
    "MeshingParameters",
    "MesherParameter",
    "ParameterSpec",
    "PointFate",
    "SimplexRegion",
    "default_initial_relaxation_weight",
    "default_relaxation_force_fun",
    "default_boundary_node_force_fun",
    "default_handle_point_density_fun",
    "MIN_DIVISION_MAGNITUDE",
]


class MeshingParameters(SectionedConfig):
    def __init__(
        self,
        string: str | None = None,
        file: str | PathLike[str] | None = None,
    ) -> None:
        super().__init__()
        self.dim: int | None = None
        self._setup_defaults()
        if file:
            self.from_file(file)
        if string:
            self.from_string(string)
        self.add_section("user-modifications")

    def _setup_defaults(self) -> None:
        self._params: dict[str, MesherParameter] = {
            # Volume determination
            "nr_probes_for_determining_volume": 100000,
            # Boundary condition parameters
            "boundary_condition_acceptable_fuzz": BOUNDARY_FUZZ,
            "boundary_condition_max_nr_correction_steps": 200,
            "boundary_condition_debuglevel": 0,
            # Relaxation parameters
            "relaxation_debuglevel": 0,
            "controller_step_limit_min": 500,
            "controller_max_time_step": 10.0,
            # Function-based parameters (callbacks for physics and point management)
            "initial_relaxation_weight_fun": default_initial_relaxation_weight,
            "relaxation_force_fun": default_relaxation_force_fun,
            "boundary_node_force_fun": default_boundary_node_force_fun,
            "handle_point_density_fun": default_handle_point_density_fun,
        }
        self._params.update({spec.internal_name: spec.default for spec in PUBLIC_PARAMETER_SPECS})

    def _get_section_name(self) -> str:
        if self.dim is None:
            raise RuntimeError("Dimension not set in MeshingParameters")
        return f"nmesh-{self.dim}D" if self.dim in [2, 3] else "nmesh-ND"

    def _lookup(self, section: str, name: str) -> MesherParameter | None:
        for key in _candidate_keys(name):
            value = self.get(section, key)
            if value is not None:
                return cast(MesherParameter, value)
        return None

    def _canonical_key(self, name: str) -> str:
        """Converts public API names to internal parameter names.

        Always stores parameters internally using verbose, namespaced names
        for clarity, even when users provide concise public names.
        """
        internal = PUBLIC_TO_INTERNAL.get(name, name)
        if internal in self._params or name in PUBLIC_TO_INTERNAL:
            return internal
        return name

    def __getitem__(self, name: str) -> MesherParameter | None:
        user_value = self._lookup("user-modifications", name)
        if user_value is not None:
            return user_value

        if self.dim is not None:
            section_value = self._lookup(self._get_section_name(), name)
            if section_value is not None:
                return section_value

        canonical = self._canonical_key(name)
        if canonical in self._params:
            return self._params[canonical]

        return None

    def __setitem__(self, key: str, value: MesherParameter) -> None:
        canonical = self._canonical_key(key)
        self._params[canonical] = value
        self.set("user-modifications", canonical, value)

    def _sync_dimension_section(self, dim: int) -> str:
        """Syncs user modifications to dimension-specific config section.

        Converts internal parameter names back to public names when writing
        to config sections for user-friendly INI file format.
        """
        self.dim = dim
        section = self._get_section_name()

        for key, value in self.items("user-modifications"):
            section_key = INTERNAL_TO_PUBLIC.get(key, key)
            self.set(section, section_key, value)

        return section

    def to_mesher_config(self, dim: int) -> dict[str, MesherParameter]:
        """Resolves all parameters to internal names for mesher consumption.

        Returns a dict with internal parameter names, suitable for passing
        to the meshing engine. This keeps the engine code clean and consistent.
        """
        self._sync_dimension_section(dim)

        resolved: dict[str, MesherParameter] = {}
        for spec in PUBLIC_PARAMETER_SPECS:
            value = self[spec.public_name]
            if value is None:
                continue
            resolved[spec.internal_name] = _cast_numeric_parameter(spec, value)

        for key, value in self._params.items():
            resolved.setdefault(key, value)

        return resolved

    def apply_to_mesher(self, mesher: dict[str, Any], dim: int) -> dict[str, Any]:
        """Applies resolved parameters to mesher config using internal names."""
        self._sync_dimension_section(dim)
        mesher.setdefault("parameters", {})

        for spec in PUBLIC_PARAMETER_SPECS:
            value = self[spec.public_name]
            if value is None:
                continue
            mesher["parameters"][spec.internal_name] = _cast_numeric_parameter(spec, value)

        for key, value in self._params.items():
            mesher["parameters"].setdefault(key, value)

        return mesher

    def _set_parameter(self, name: str, value: object) -> None:
        """Internal helper for generated setter methods."""
        spec = PUBLIC_PARAMETER_SPECS_BY_NAME[name]
        self[name] = _cast_numeric_parameter(spec, value)

    def copy(self) -> MeshingParameters:
        return copy.deepcopy(self)


for _spec in PUBLIC_PARAMETER_SPECS:
    setattr(
        MeshingParameters,
        f"set_{_spec.public_name}",
        partialmethod(MeshingParameters._set_parameter, _spec.public_name),
    )
