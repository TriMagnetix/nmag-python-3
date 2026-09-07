from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from si.physical import SI

if TYPE_CHECKING:
    from simulation.quantity import Quantity

from ..support import (
    DERIVED_FIELD_NAMES,
    _si_unit,
    _simulation_compatibility_binding,
)

_known_field_quantities_cache: list[Quantity] | None = None
_known_quantities_by_name_cache: dict[str, Quantity] | None = None


def _known_field_quantities() -> list[Quantity]:
    global _known_field_quantities_cache
    if _known_field_quantities_cache is None:
        from simulation.quantity import known_field_quantities

        _known_field_quantities_cache = known_field_quantities
    return _known_field_quantities_cache


def _known_quantities_by_name() -> dict[str, Quantity]:
    global _known_quantities_by_name_cache
    if _known_quantities_by_name_cache is None:
        from simulation.quantity import known_quantities_by_name

        _known_quantities_by_name_cache = known_quantities_by_name
    return _known_quantities_by_name_cache


class SimulationFieldAvailabilityMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def get_all_field_names(self) -> list[str]:
        """Return the field names currently available for saving or access."""
        has_mesh = self.mesh is not None
        has_m = "m" in self._fields
        has_mesh_and_m = has_mesh and has_m
        has_tetrahedral_mesh = self._has_tetrahedral_mesh_or_empty()
        return [
            quantity.name
            for quantity in _simulation_compatibility_binding(
                "_known_field_quantities", _known_field_quantities
            )()
            if self._is_subfield_available_from_state(
                quantity.name,
                has_mesh=has_mesh,
                has_m=has_m,
                has_mesh_and_m=has_mesh_and_m,
                has_tetrahedral_mesh=has_tetrahedral_mesh,
            )
        ]

    def is_subfield_available(self, subfieldname: str) -> bool:
        """Return whether the current model can provide a named field.

        The check does not trigger expensive FEM/BEM work. Actual field access
        can still raise a numerical error during calculation.

        Args:
            subfieldname: Candidate field name.

        Returns:
            True when the field is set or derivable from current state.
        """
        if subfieldname == "H_ext":
            return True
        if subfieldname not in DERIVED_FIELD_NAMES and subfieldname != "H_demag":
            return subfieldname in self._fields

        has_mesh = self.mesh is not None
        has_m = "m" in self._fields
        has_tetrahedral_mesh = self._has_tetrahedral_mesh_or_empty()
        return self._is_subfield_available_from_state(
            subfieldname,
            has_mesh=has_mesh,
            has_m=has_m,
            has_mesh_and_m=has_mesh and has_m,
            has_tetrahedral_mesh=has_tetrahedral_mesh,
        )

    def _is_subfield_available_from_state(
        self,
        subfieldname: str,
        *,
        has_mesh: bool,
        has_m: bool,
        has_mesh_and_m: bool,
        has_tetrahedral_mesh: bool,
    ) -> bool:
        has_m_on_tetrahedral_mesh = self._has_m_on_tetrahedral_mesh(
            has_mesh_and_m,
            has_tetrahedral_mesh,
        )
        state_availability = {
            "pin": has_mesh,
            "H_anis": has_mesh_and_m,
            "E_anis": has_mesh_and_m,
            "H_exch": has_m_on_tetrahedral_mesh,
            "E_exch": has_m_on_tetrahedral_mesh,
            "dm_dcurrent": self._has_current_density(has_m_on_tetrahedral_mesh),
            "dmdt": has_m_on_tetrahedral_mesh,
            "H_total": has_m_on_tetrahedral_mesh,
            "E_total": has_m_on_tetrahedral_mesh,
            "E_demag": self._has_demag_magnetisation(has_mesh_and_m),
            "phi": self._has_demag_magnetisation(has_mesh_and_m),
            "rho": self._has_demag_magnetisation(has_mesh_and_m),
            "M": has_mesh_and_m,
            "E_ext": has_mesh_and_m,
            "H_demag": self._has_demag_nodal_field(has_m, has_tetrahedral_mesh),
            "H_ext": True,
        }
        if subfieldname in state_availability:
            return state_availability[subfieldname]
        return subfieldname not in DERIVED_FIELD_NAMES and subfieldname in self._fields

    @staticmethod
    def _has_m_on_tetrahedral_mesh(has_mesh_and_m: bool, has_tetrahedral_mesh: bool) -> bool:
        return has_mesh_and_m and has_tetrahedral_mesh

    def _has_current_density(self, has_m_on_tetrahedral_mesh: bool) -> bool:
        return has_m_on_tetrahedral_mesh and "current_density" in self._fields

    def _has_demag_magnetisation(self, has_mesh_and_m: bool) -> bool:
        return self.do_demag and has_mesh_and_m

    def _has_demag_nodal_field(self, has_m: bool, has_tetrahedral_mesh: bool) -> bool:
        return self.do_demag and has_m and has_tetrahedral_mesh

    def get_materials_of_field(self, field_name: str) -> list[Any]:
        quantity = _simulation_compatibility_binding(
            "_known_quantities_by_name", _known_quantities_by_name
        )()[field_name]
        if "?" in (quantity.signature or ""):
            return self.materials
        return []

    def get_subfield(self, subfieldname: str, units: SI | None = None) -> Any:
        """Return all nodal or cell values for one available field.

        Args:
            subfieldname: Name returned by :meth:`get_all_field_names`.
            units: Optional compatible SI unit for returned numeric values.

        Returns:
            Field values as Python scalars or nested lists.

        Raises:
            KeyError: If the field is unknown, unset, or disabled.
        """
        data, field_units = self._subfield_data_and_units(subfieldname)
        if units is None or field_units is None:
            return data.tolist() if isinstance(data, np.ndarray) else data
        return (np.asarray(data) * field_units.in_units_of(units)).tolist()

    def _subfield_data_and_units(self, subfieldname: str) -> tuple[Any, SI | None]:
        if subfieldname == "H_demag":
            if not self.is_subfield_available(subfieldname):
                raise KeyError("Subfield 'H_demag' is unavailable when demag is disabled.")
            data = np.asarray(self._get_demag_nodal_field(), dtype=float)
            return data, _si_unit("A/m")
        if subfieldname in self._fields:
            data = self._fields[subfieldname]
        elif self.is_subfield_available(subfieldname):
            data = self._subfield_array(subfieldname)
        else:
            raise KeyError(f"Unknown or unset subfield '{subfieldname}'.")
        quantity = _simulation_compatibility_binding(
            "_known_quantities_by_name", _known_quantities_by_name
        )().get(subfieldname)
        return data, None if quantity is None else quantity.units

    def _has_tetrahedral_mesh_or_empty(self) -> bool:
        if self.mesh is None:
            return False
        simplices = np.asarray(self.mesh.simplices)
        return simplices.size == 0 or (simplices.ndim == 2 and simplices.shape[1] == 4)
