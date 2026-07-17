from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..support import (
    _NO_COMPOSED_AVERAGE,
    _NO_CONSTANT_AVERAGE,
    _copy_average_value,
)

_NO_ARRAY_AVERAGE = object()


class SimulationFieldAverageMixin:
    if TYPE_CHECKING:
        _active_subfield_array_timings: dict[str, float] | None

        def __getattr__(self, name: str) -> Any: ...

    def get_subfield_average(self, subfieldname: str, mat_name: str | None = None) -> Any:
        timings: dict[str, float] = {}
        self.last_subfield_average_timings_seconds = {}
        total_started = time.perf_counter()
        cache_key = (subfieldname, mat_name)
        cached_average = self._cached_subfield_average(cache_key)
        if cached_average is not _NO_CONSTANT_AVERAGE:
            timings["cache_hit"] = time.perf_counter() - total_started
            self._finish_subfield_average_timings(timings, total_started)
            return _copy_average_value(cached_average)

        started = time.perf_counter()
        constant_average = self._constant_subfield_average(subfieldname)
        if constant_average is not _NO_CONSTANT_AVERAGE:
            timings["constant_average"] = time.perf_counter() - started
            self._finish_subfield_average_timings(timings, total_started)
            self._cache_subfield_average(cache_key, constant_average)
            return constant_average

        if subfieldname == "H_demag":
            result = self._direct_demag_average(timings)
            self._finish_subfield_average_timings(timings, total_started)
            self._cache_subfield_average(cache_key, result)
            return result

        composed_average = self._try_composed_subfield_average(subfieldname, mat_name, timings)
        if composed_average is not _NO_COMPOSED_AVERAGE:
            result = np.asarray(composed_average, dtype=float).tolist()
            self._finish_subfield_average_timings(timings, total_started)
            self._cache_subfield_average(cache_key, result)
            return result

        average = self._array_subfield_average(subfieldname, mat_name, timings)
        self._finish_subfield_average_timings(timings, total_started)
        if average is _NO_ARRAY_AVERAGE:
            return None
        self._cache_subfield_average(cache_key, average)
        return average

    def _finish_subfield_average_timings(
        self,
        timings: dict[str, float],
        total_started: float,
    ) -> None:
        timings["total"] = time.perf_counter() - total_started
        self.last_subfield_average_timings_seconds = dict(sorted(timings.items()))

    def _direct_demag_average(self, timings: dict[str, float]) -> list[float]:
        subfield_timings: dict[str, float] = {}
        previous_subfield_timings = self._active_subfield_array_timings
        self._active_subfield_array_timings = subfield_timings
        try:
            started = time.perf_counter()
            average = self._demag_cell_field_average()
            timings["direct_cell_average"] = time.perf_counter() - started
        finally:
            if self._active_subfield_array_timings is subfield_timings:
                self._active_subfield_array_timings = previous_subfield_timings
        self._append_subfield_array_timings(timings, subfield_timings)
        return np.asarray(average, dtype=float).tolist()

    def _try_composed_subfield_average(
        self,
        subfieldname: str,
        mat_name: str | None,
        timings: dict[str, float],
    ) -> object:
        composed_timings: dict[str, float] = {}
        subfield_timings: dict[str, float] = {}
        previous_subfield_timings = self._active_subfield_array_timings
        self._active_subfield_array_timings = subfield_timings
        try:
            started = time.perf_counter()
            try:
                average = self._composed_subfield_average(
                    subfieldname,
                    composed_timings,
                    mat_name=mat_name,
                )
            except KeyError:
                average = _NO_COMPOSED_AVERAGE
            timings["composed_average"] = time.perf_counter() - started
        finally:
            if self._active_subfield_array_timings is subfield_timings:
                self._active_subfield_array_timings = previous_subfield_timings
        if average is _NO_COMPOSED_AVERAGE:
            timings.pop("composed_average")
            return average
        timings.update(composed_timings)
        self._append_subfield_array_timings(timings, subfield_timings)
        return average

    def _array_subfield_average(
        self,
        subfieldname: str,
        mat_name: str | None,
        timings: dict[str, float],
    ) -> Any | object:
        subfield_timings: dict[str, float] = {}
        previous_subfield_timings = self._active_subfield_array_timings
        self._active_subfield_array_timings = subfield_timings
        data: np.ndarray | None
        try:
            try:
                started = time.perf_counter()
                data = self._subfield_array(subfieldname)
                timings["array"] = time.perf_counter() - started
            except KeyError:
                data = None
        finally:
            if self._active_subfield_array_timings is subfield_timings:
                self._active_subfield_array_timings = previous_subfield_timings
        self._append_subfield_array_timings(timings, subfield_timings)
        if data is None or data.size == 0:
            return _NO_ARRAY_AVERAGE
        started = time.perf_counter()
        average_value = self._field_average(data, mat_name=mat_name)
        average = (
            average_value.tolist()
            if isinstance(average_value, np.ndarray)
            else float(average_value)
        )
        timings["field_average"] = time.perf_counter() - started
        return average

    def _append_subfield_array_timings(
        self,
        timings: dict[str, float],
        subfield_timings: dict[str, float],
    ) -> None:
        timings.update(
            (f"subfield_array:{name}", seconds)
            for name, seconds in sorted(subfield_timings.items())
        )

    def _constant_subfield_average(self, subfieldname: str) -> object:
        if subfieldname == "H_ext":
            return self._constant_h_ext_average()
        if self.mesh is None:
            return _NO_CONSTANT_AVERAGE
        handlers = {
            "pin": self._constant_pin_average,
            "H_anis": self._constant_h_anis_average,
            "E_anis": self._constant_e_anis_average,
            "H_exch": self._constant_h_exch_average,
            "E_exch": self._constant_e_exch_average,
        }
        handler = handlers.get(subfieldname)
        return _NO_CONSTANT_AVERAGE if handler is None else handler()

    def _constant_h_ext_average(self) -> object:
        h_ext = np.asarray(self._fields.get("H_ext", np.zeros(3)), dtype=float)
        return h_ext.tolist() if h_ext.shape == (3,) else _NO_CONSTANT_AVERAGE

    def _constant_pin_average(self) -> object:
        pin = self._fields.get("pin")
        if pin is None:
            return 1.0
        pin_values = np.asarray(pin, dtype=float)
        if pin_values.size > 0 and np.all(pin_values == pin_values[0]):
            return float(pin_values[0])
        return _NO_CONSTANT_AVERAGE

    def _constant_h_anis_average(self) -> object:
        return self._constant_anisotropy_average([0.0, 0.0, 0.0])

    def _constant_e_anis_average(self) -> object:
        return self._constant_anisotropy_average(0.0)

    def _constant_anisotropy_average(self, zero_value: object) -> object:
        if "m" not in self._fields or not self._anisotropy_is_zero_by_construction():
            return _NO_CONSTANT_AVERAGE
        return zero_value

    def _constant_h_exch_average(self) -> object:
        return [0.0, 0.0, 0.0] if self._exchange_is_zero_by_construction() else _NO_CONSTANT_AVERAGE

    def _constant_e_exch_average(self) -> object:
        return 0.0 if self._exchange_is_zero_by_construction() else _NO_CONSTANT_AVERAGE

    def _composed_subfield_average(
        self,
        subfieldname: str,
        timings: dict[str, float],
        *,
        mat_name: str | None = None,
    ) -> object:
        if subfieldname != "H_total":
            return _NO_COMPOSED_AVERAGE
        if self.mesh is None or "m" not in self._fields:
            return _NO_COMPOSED_AVERAGE

        total = np.asarray(
            self._average_component_for_composition("H_ext", timings),
            dtype=float,
        )
        if self.do_demag:
            total = total + np.asarray(
                self._average_component_for_composition("H_demag", timings),
                dtype=float,
            )
        total = total + np.asarray(
            self._average_component_for_composition(
                "H_anis",
                timings,
                mat_name=mat_name,
            ),
            dtype=float,
        )
        total = total + np.asarray(
            self._average_component_for_composition(
                "H_exch",
                timings,
                mat_name=mat_name,
            ),
            dtype=float,
        )
        return total

    def _average_component_for_composition(
        self,
        subfieldname: str,
        timings: dict[str, float],
        *,
        mat_name: str | None = None,
    ) -> np.ndarray:
        cache_key = (subfieldname, mat_name)
        cached_average = self._cached_subfield_average(cache_key)
        if cached_average is not _NO_CONSTANT_AVERAGE:
            timings[f"component_average:{subfieldname}:cache_hit"] = 0.0
            return np.asarray(cached_average, dtype=float)

        started = time.perf_counter()
        constant_average = self._constant_subfield_average(subfieldname)
        if constant_average is not _NO_CONSTANT_AVERAGE:
            timings[f"component_average:{subfieldname}:constant"] = time.perf_counter() - started
            self._cache_subfield_average(cache_key, constant_average)
            return np.asarray(constant_average, dtype=float)

        if subfieldname == "H_demag":
            started = time.perf_counter()
            average = self._demag_cell_field_average()
            timings[f"component_average:{subfieldname}:direct_cell_average"] = (
                time.perf_counter() - started
            )
            self._cache_subfield_average(cache_key, average)
            return np.asarray(average, dtype=float)

        started = time.perf_counter()
        data = self._subfield_array(subfieldname)
        timings[f"component_average:{subfieldname}:array"] = time.perf_counter() - started
        if data.size == 0:
            raise KeyError(f"Component field '{subfieldname}' is empty.")

        started = time.perf_counter()
        average = self._field_average(data, mat_name=mat_name)
        timings[f"component_average:{subfieldname}:field_average"] = time.perf_counter() - started
        self._cache_subfield_average(cache_key, average)
        return np.asarray(average, dtype=float)
