from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...backends import _selected_llg_backend
from ..support import (
    LEGACY_DMDT_MAGNETISATION_SCALE,
    MU0,
    _simulation_compatibility_binding,
)


class SimulationFieldDerivedMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def _derived_subfield_array(self, subfieldname: str) -> np.ndarray:
        if self.mesh is None:
            raise KeyError(f"Derived field '{subfieldname}' requires a mesh.")
        point_count = len(self.mesh.points)
        if subfieldname == "pin":
            return self._derived_pin(point_count)
        if "m" not in self._fields:
            raise KeyError(f"Derived field '{subfieldname}' requires magnetisation.")

        resolvers = {
            "phi": self._derived_phi,
            "rho": self._derived_rho,
            "H_anis": self._derived_h_anis,
            "H_exch": self._derived_h_exch,
            "dm_dcurrent": self._derived_dm_dcurrent,
            "dmdt": self._derived_dmdt,
            "M": self._derived_magnetisation,
            "H_total": self._derived_h_total,
            "E_anis": self._derived_e_anis,
            "E_exch": self._derived_e_exch,
            "E_ext": self._derived_e_ext,
            "E_demag": self._derived_e_demag,
            "E_total": self._derived_e_total,
        }
        resolver = resolvers.get(subfieldname)
        if resolver is None:
            raise KeyError(f"Unknown or unset subfield '{subfieldname}'.") from None
        return resolver(point_count)

    def _derived_pin(self, point_count: int) -> np.ndarray:
        return np.asarray(
            self._fields.get("pin", np.ones(point_count, dtype=float)),
            dtype=float,
        )

    def _derived_phi(self, point_count: int) -> np.ndarray:
        if not self.do_demag:
            return np.zeros(point_count, dtype=float)
        phi, _rho, _volumes = self._get_demag_auxiliary_fields()
        return phi

    def _derived_rho(self, point_count: int) -> np.ndarray:
        if not self.do_demag:
            return np.zeros(point_count, dtype=float)
        _phi, rho, _volumes = self._get_demag_auxiliary_fields()
        return rho

    def _derived_h_anis(self, point_count: int) -> np.ndarray:
        if not self._anisotropy_is_zero_by_construction():
            raise KeyError("H_anis is not implemented for anisotropic materials.")
        return np.zeros((point_count, 3), dtype=float)

    def _derived_h_exch(self, _point_count: int) -> np.ndarray:
        return self._get_exchange_nodal_field()

    def _derived_dm_dcurrent(self, _point_count: int) -> np.ndarray:
        return self._get_dm_dcurrent()

    def _derived_dmdt(self, _point_count: int) -> np.ndarray:
        coefficients = self._nodal_material_coefficients()
        m = np.asarray(self._fields["m"], dtype=float)
        dmdt_scale_values = np.full(len(m), LEGACY_DMDT_MAGNETISATION_SCALE, dtype=float)
        dm_dcurrent = (
            self._subfield_array("dm_dcurrent") if "current_density" in self._fields else None
        )
        backend = self._llg_backend(len(m))
        rhs = self._llg_rhs_rust if backend == "rust" else self._llg_rhs_python
        with self._record_active_subfield_array_timing_block(f"dmdt_detail:llg_rhs:{backend}"):
            return rhs(
                m,
                self._subfield_array("H_total"),
                self._subfield_array("pin"),
                dmdt_scale_values,
                coefficients.precession,
                coefficients.damping,
                coefficients.normalisation,
                dm_dcurrent,
                coefficients.stt_adiabatic,
                coefficients.stt_nonadiabatic,
            )

    def _llg_backend(self, point_count: int) -> str:
        return _simulation_compatibility_binding("_selected_llg_backend", _selected_llg_backend)(
            point_count, getattr(self, "config", None)
        )

    def _derived_magnetisation(self, _point_count: int) -> np.ndarray:
        return np.asarray(self._fields["m"], dtype=float) * self._nodal_ms_values()[:, np.newaxis]

    def _derived_h_total(self, _point_count: int) -> np.ndarray:
        h_total = np.asarray(self._subfield_array("H_ext"), dtype=float)
        if self.do_demag:
            h_total = h_total + np.asarray(self._subfield_array("H_demag"), dtype=float)
        return h_total + self._subfield_array("H_anis") + self._subfield_array("H_exch")

    def _derived_e_anis(self, point_count: int) -> np.ndarray:
        if not self._anisotropy_is_zero_by_construction():
            raise KeyError("E_anis is not implemented for anisotropic materials.")
        return np.zeros(point_count, dtype=float)

    def _derived_e_exch(self, _point_count: int) -> np.ndarray:
        return (
            -0.5
            * MU0
            * self._nodal_ms_values()
            * np.einsum(
                "ij,ij->i",
                self._subfield_array("H_exch"),
                np.asarray(self._fields["m"], dtype=float),
            )
        )

    def _derived_e_ext(self, _point_count: int) -> np.ndarray:
        return -MU0 * np.einsum(
            "ij,ij->i",
            self._subfield_array("M"),
            self._subfield_array("H_ext"),
        )

    def _derived_e_demag(self, point_count: int) -> np.ndarray:
        if not self.do_demag:
            return np.zeros(point_count, dtype=float)
        return (
            -0.5
            * MU0
            * np.einsum(
                "ij,ij->i",
                self._subfield_array("M"),
                self._subfield_array("H_demag"),
            )
        )

    def _derived_e_total(self, _point_count: int) -> np.ndarray:
        return (
            self._subfield_array("E_ext")
            + self._subfield_array("E_anis")
            + self._subfield_array("E_exch")
            + self._subfield_array("E_demag")
        )

    def _anisotropy_is_zero_by_construction(self) -> bool:
        return all(getattr(material, "anisotropy", None) is None for material in self.materials)

    def _exchange_is_zero_by_construction(self) -> bool:
        if "m" not in self._fields:
            return False
        m = np.asarray(self._fields["m"], dtype=float)
        if m.ndim != 2 or len(m) == 0:
            return False
        return bool(np.allclose(m, m[0], rtol=0.0, atol=1.0e-12))
