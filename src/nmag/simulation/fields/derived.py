from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...backends import _selected_llg_backend
from ..support import (
    LEGACY_DMDT_MAGNETISATION_SCALE,
    MU0,
    _simulation_compatibility_binding,
)

_is_uniform_within_atol_kernel: Callable[[np.ndarray, float], bool] | None = None


def _is_uniform_within_atol_raw(m: np.ndarray, atol: float) -> bool:
    n = m.shape[0]
    if n == 0:
        return False
    m00 = m[0, 0]
    m01 = m[0, 1]
    m02 = m[0, 2]
    if not (np.isfinite(m00) and np.isfinite(m01) and np.isfinite(m02)):
        return False
    for i in range(1, n):
        if not (
            abs(m[i, 0] - m00) <= atol
            and abs(m[i, 1] - m01) <= atol
            and abs(m[i, 2] - m02) <= atol
        ):
            return False
    return True


def _get_is_uniform_within_atol_kernel() -> Callable[[np.ndarray, float], bool] | None:
    global _is_uniform_within_atol_kernel
    if _is_uniform_within_atol_kernel is not None:
        return _is_uniform_within_atol_kernel
    try:
        from numba import njit
    except ImportError:  # pragma: no cover - numba is a core dependency.
        return None
    _is_uniform_within_atol_kernel = cast(
        Callable[[np.ndarray, float], bool],
        njit(cache=True)(_is_uniform_within_atol_raw),
    )
    return _is_uniform_within_atol_kernel


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
        pin = self._fields.get("pin")
        if pin is not None:
            return np.asarray(pin, dtype=float)
        return np.ones(point_count, dtype=float)

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
        field, _energy = self._get_anisotropy_fields()
        return field

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
        if (
            getattr(self, "_rhs_cached_tokens", None) is not None
            and getattr(self, "_subfield_array_cache", None) is None
            and getattr(self, "_active_subfield_array_timings", None) is None
        ):
            h_ext_raw = self._fields.get("H_ext")
            h_accum: np.ndarray | None = None
            if h_ext_raw is not None:
                h_ext_vec = np.asarray(h_ext_raw, dtype=float)
                if h_ext_vec.shape != (3,):
                    raise ValueError(
                        f"H_ext must be a homogeneous 3-vector, got shape {h_ext_vec.shape}."
                    )
                if h_ext_vec[0] != 0.0 or h_ext_vec[1] != 0.0 or h_ext_vec[2] != 0.0:
                    h_accum = np.broadcast_to(h_ext_vec, (_point_count, 3)).copy()
            if self.do_demag:
                h_demag = np.asarray(self._get_demag_nodal_field(), dtype=float)
                if h_accum is None:
                    h_accum = h_demag.copy()
                else:
                    h_accum += h_demag
            if not self._anisotropy_is_zero_by_construction():
                h_anis, _ = self._get_anisotropy_fields()
                if h_accum is None:
                    h_accum = h_anis.copy()
                else:
                    h_accum += h_anis
            h_exch = np.asarray(self._get_exchange_nodal_field(), dtype=float)
            if h_accum is None:
                return h_exch
            h_accum += h_exch
            return h_accum

        h_ext = np.asarray(self._subfield_array("H_ext"), dtype=float)
        cache = getattr(self, "_subfield_array_cache", None)
        h_total = (
            np.array(h_ext, dtype=float, copy=True)
            if (not h_ext.flags.writeable or (cache is not None and cache.get("H_ext") is h_ext))
            else h_ext
        )
        if self.do_demag:
            h_total += np.asarray(self._subfield_array("H_demag"), dtype=float)
        if not self._anisotropy_is_zero_by_construction():
            h_total += self._subfield_array("H_anis")
        h_total += self._subfield_array("H_exch")
        return h_total

    def _derived_e_anis(self, point_count: int) -> np.ndarray:
        _field, energy = self._get_anisotropy_fields()
        return energy

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

    def _exchange_is_zero_by_construction(self) -> bool:
        if "m" not in self._fields:
            return False
        m = np.asarray(self._fields["m"], dtype=float)
        if m.ndim != 2 or len(m) == 0:
            return False
        if m.shape[1] == 3 and m.dtype == np.float64:
            kernel = _get_is_uniform_within_atol_kernel()
            if kernel is not None:
                return bool(kernel(m, 1.0e-12))
        if len(m) > 1 and np.max(np.abs(m[1] - m[0])) > 1.0e-12:
            return False
        return bool(np.allclose(m, m[0], rtol=0.0, atol=1.0e-12))
