from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...backends import _load_rust_accelerator
from ..support import _simulation_compatibility_binding

_llg_rhs_no_stt_numba_kernel: Callable[..., np.ndarray] | None = None


def _llg_rhs_no_stt_raw(
    m: np.ndarray,
    h_total: np.ndarray,
    pin: np.ndarray,
    ms_values: np.ndarray,
    precession_values: np.ndarray,
    damping_values: np.ndarray,
    normalisation_values: np.ndarray,
) -> np.ndarray:
    n = m.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        m0 = m[i, 0]
        m1 = m[i, 1]
        m2 = m[i, 2]
        h0 = h_total[i, 0]
        h1 = h_total[i, 1]
        h2 = h_total[i, 2]
        mdoth = m0 * h0 + m1 * h1 + m2 * h2
        mdotm = m0 * m0 + m1 * m1 + m2 * m2
        d0 = m0 * mdoth - h0 * mdotm
        d1 = m1 * mdoth - h1 * mdotm
        d2 = m2 * mdoth - h2 * mdotm
        ne = 1.0 - mdotm
        damp = damping_values[i]
        norm_c = normalisation_values[i]
        prec = precession_values[i]
        r0 = damp * d0 + norm_c * (ne * m0)
        r1 = damp * d1 + norm_c * (ne * m1)
        r2 = damp * d2 + norm_c * (ne * m2)
        if prec != 0.0:
            r0 += prec * (m1 * h2 - m2 * h1)
            r1 += prec * (m2 * h0 - m0 * h2)
            r2 += prec * (m0 * h1 - m1 * h0)
        ms_i = ms_values[i]
        pin_i = pin[i]
        out[i, 0] = ms_i * r0 * pin_i
        out[i, 1] = ms_i * r1 * pin_i
        out[i, 2] = ms_i * r2 * pin_i
    return out


def _get_llg_rhs_no_stt_numba_kernel() -> Callable[..., np.ndarray] | None:
    global _llg_rhs_no_stt_numba_kernel
    if _llg_rhs_no_stt_numba_kernel is not None:
        return _llg_rhs_no_stt_numba_kernel
    try:
        from numba import njit
    except ImportError:  # pragma: no cover - numba is a core dependency.
        return None
    _llg_rhs_no_stt_numba_kernel = cast(
        Callable[..., np.ndarray],
        njit(cache=True)(_llg_rhs_no_stt_raw),
    )
    return _llg_rhs_no_stt_numba_kernel


class SimulationLlgRhsMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def _llg_rhs_python(
        self,
        m: np.ndarray,
        h_total: np.ndarray,
        pin: np.ndarray,
        ms_values: np.ndarray,
        precession_coeff: float | np.ndarray,
        damping_coeff: float | np.ndarray,
        normalisation_coeff: float | np.ndarray,
        dm_dcurrent: np.ndarray | None = None,
        stt_adiabatic_coeff: float | np.ndarray = 0.0,
        stt_nonadiabatic_coeff: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        m, h_total, pin, ms_values = self._validated_llg_inputs(m, h_total, pin, ms_values)
        precession_values, damping_values, normalisation_values = self._nodal_coefficients(
            len(m),
            (
                ("precession", precession_coeff),
                ("damping", damping_coeff),
                ("normalisation", normalisation_coeff),
            ),
        )
        if dm_dcurrent is None:
            kernel = _get_llg_rhs_no_stt_numba_kernel()
            if kernel is not None:
                return kernel(
                    m,
                    h_total,
                    pin,
                    ms_values,
                    precession_values,
                    damping_values,
                    normalisation_values,
                )
        rhs, mdotm = self._llg_field_rhs(
            m,
            h_total,
            precession_values,
            damping_values,
            normalisation_values,
        )
        if dm_dcurrent is not None:
            rhs += self._llg_stt_rhs(
                m,
                mdotm,
                dm_dcurrent,
                stt_adiabatic_coeff,
                stt_nonadiabatic_coeff,
            )
        return ms_values[:, np.newaxis] * rhs * pin[:, np.newaxis]

    @staticmethod
    def _validated_llg_inputs(
        m: np.ndarray,
        h_total: np.ndarray,
        pin: np.ndarray,
        ms_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = np.asarray(m, dtype=float)
        h_total = np.asarray(h_total, dtype=float)
        pin = np.asarray(pin, dtype=float)
        ms_values = np.asarray(ms_values, dtype=float)
        if m.ndim != 2 or m.shape[1] != 3:
            raise ValueError(f"m must have shape (n, 3), got {m.shape}.")
        if h_total.shape != m.shape:
            raise ValueError(f"h_total must have shape {m.shape}, got {h_total.shape}.")
        if pin.shape != (len(m),):
            raise ValueError(f"pin must have shape ({len(m)},), got {pin.shape}.")
        if ms_values.shape != (len(m),):
            raise ValueError(f"ms_values must have shape ({len(m)},), got {ms_values.shape}.")
        return m, h_total, pin, ms_values

    @staticmethod
    def _nodal_coefficients(
        point_count: int,
        coefficients: tuple[tuple[str, float | np.ndarray], ...],
    ) -> tuple[np.ndarray, ...]:
        arrays: list[np.ndarray] = []
        for name, coefficient in coefficients:
            array = np.asarray(coefficient, dtype=float)
            if array.ndim == 0:
                array = np.full(point_count, float(array), dtype=float)
            if array.shape != (point_count,):
                raise ValueError(
                    f"{name} coefficient must have shape ({point_count},), got {array.shape}."
                )
            arrays.append(array)
        return tuple(arrays)

    @staticmethod
    def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
        out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
        out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        return out

    @staticmethod
    def _dot3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]

    @classmethod
    def _llg_field_rhs(
        cls,
        m: np.ndarray,
        h_total: np.ndarray,
        precession_values: np.ndarray,
        damping_values: np.ndarray,
        normalisation_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mdoth = cls._dot3(m, h_total)
        mdotm = cls._dot3(m, m)
        damping = m * mdoth[:, np.newaxis] - h_total * mdotm[:, np.newaxis]
        norm_error = (1.0 - mdotm)[:, np.newaxis] * m
        rhs = (
            damping_values[:, np.newaxis] * damping
            + normalisation_values[:, np.newaxis] * norm_error
        )
        if np.any(precession_values):
            rhs += precession_values[:, np.newaxis] * cls._cross3(m, h_total)
        return rhs, mdotm

    def _llg_stt_rhs(
        self,
        m: np.ndarray,
        mdotm: np.ndarray,
        dm_dcurrent: np.ndarray,
        stt_adiabatic_coeff: float | np.ndarray,
        stt_nonadiabatic_coeff: float | np.ndarray,
    ) -> np.ndarray:
        directional_derivative = np.asarray(dm_dcurrent, dtype=float)
        if directional_derivative.shape != m.shape:
            raise ValueError(
                f"dm_dcurrent must have shape {m.shape}, got {directional_derivative.shape}."
            )
        adiabatic_values, nonadiabatic_values = self._nodal_coefficients(
            len(m),
            (
                ("stt_adiabatic", stt_adiabatic_coeff),
                ("stt_nonadiabatic", stt_nonadiabatic_coeff),
            ),
        )
        mx_directional = self._cross3(m, directional_derivative)
        mdot_directional = self._dot3(m, directional_derivative)
        mxmx_directional = (
            m * mdot_directional[:, np.newaxis] - directional_derivative * mdotm[:, np.newaxis]
        )
        return (
            adiabatic_values[:, np.newaxis] * mxmx_directional
            + nonadiabatic_values[:, np.newaxis] * mx_directional
        )

    def _llg_rhs_rust(
        self,
        m: np.ndarray,
        h_total: np.ndarray,
        pin: np.ndarray,
        ms_values: np.ndarray,
        precession_coeff: float | np.ndarray,
        damping_coeff: float | np.ndarray,
        normalisation_coeff: float | np.ndarray,
        dm_dcurrent: np.ndarray | None = None,
        stt_adiabatic_coeff: float | np.ndarray = 0.0,
        stt_nonadiabatic_coeff: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        rust_accel = _simulation_compatibility_binding(
            "_load_rust_accelerator",
            _load_rust_accelerator,
        )("NmagConfig.accelerator['llg']")
        try:
            llg_rhs = rust_accel.llg_rhs
        except AttributeError as exc:
            raise RuntimeError(
                "NmagConfig.accelerator['llg']='rust' requires an nmag_accel build with llg_rhs. "
                "Rebuild it with "
                "`maturin develop --release --manifest-path rust/nmag_accel/Cargo.toml`."
            ) from exc
        coefficient_arrays = [
            np.asarray(value, dtype=float)
            for value in (
                precession_coeff,
                damping_coeff,
                normalisation_coeff,
            )
        ]
        if dm_dcurrent is None and all(array.ndim == 0 for array in coefficient_arrays):
            result = llg_rhs(
                np.asarray(m, dtype=float),
                np.asarray(h_total, dtype=float),
                np.asarray(pin, dtype=float),
                np.asarray(ms_values, dtype=float),
                *(float(array) for array in coefficient_arrays),
            )
        else:
            try:
                heterogeneous_rhs = (
                    rust_accel.llg_rhs_heterogeneous
                    if dm_dcurrent is None
                    else rust_accel.llg_rhs_stt_heterogeneous
                )
            except AttributeError as exc:
                raise RuntimeError(
                    "NmagConfig.accelerator['llg']='rust' requires an nmag_accel build with "
                    "the heterogeneous LLG kernels. Rebuild the accelerator."
                ) from exc
            expanded = [
                np.full(len(m), float(array), dtype=float) if array.ndim == 0 else array
                for array in coefficient_arrays
            ]
            arguments: list[np.ndarray] = [
                np.asarray(m, dtype=float),
                np.asarray(h_total, dtype=float),
                np.asarray(pin, dtype=float),
                np.asarray(ms_values, dtype=float),
                *expanded,
            ]
            if dm_dcurrent is not None:
                stt_arrays = [
                    np.asarray(value, dtype=float)
                    for value in (stt_adiabatic_coeff, stt_nonadiabatic_coeff)
                ]
                expanded_stt = [
                    np.full(len(m), float(array), dtype=float) if array.ndim == 0 else array
                    for array in stt_arrays
                ]
                arguments.extend([np.asarray(dm_dcurrent, dtype=float), *expanded_stt])
            result = heterogeneous_rhs(*arguments)
        return np.asarray(result, dtype=float)
