from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...backends import _load_rust_accelerator
from ..support import _simulation_compatibility_binding


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
    def _llg_field_rhs(
        m: np.ndarray,
        h_total: np.ndarray,
        precession_values: np.ndarray,
        damping_values: np.ndarray,
        normalisation_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mxh = np.cross(m, h_total)
        mdoth = np.einsum("ij,ij->i", m, h_total)
        mdotm = np.einsum("ij,ij->i", m, m)
        damping = m * mdoth[:, np.newaxis] - h_total * mdotm[:, np.newaxis]
        norm_error = (1.0 - mdotm)[:, np.newaxis] * m
        rhs = (
            precession_values[:, np.newaxis] * mxh
            + damping_values[:, np.newaxis] * damping
            + normalisation_values[:, np.newaxis] * norm_error
        )
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
        mx_directional = np.cross(m, directional_derivative)
        mdot_directional = np.einsum("ij,ij->i", m, directional_derivative)
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
