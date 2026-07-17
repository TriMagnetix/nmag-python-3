from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from ..support import (
    _NO_CONSTANT_AVERAGE,
    DERIVED_FIELD_NAMES,
    _copy_average_value,
)


class SimulationFieldArrayMixin:
    if TYPE_CHECKING:
        _subfield_array_cache: dict[str, np.ndarray] | None
        _subfield_average_cache: dict[tuple[str, str | None], object] | None
        _active_subfield_array_timings: dict[str, float] | None

        def __getattr__(self, name: str) -> Any: ...

    def _subfield_array(self, subfieldname: str) -> np.ndarray:
        started = time.perf_counter()
        if self._subfield_array_cache is not None and subfieldname in self._subfield_array_cache:
            self._record_active_subfield_array_timing(
                f"{subfieldname}:cache_hit",
                time.perf_counter() - started,
            )
            return self._subfield_array_cache[subfieldname]

        data = self._compute_subfield_array(subfieldname)
        self._record_active_subfield_array_timing(
            f"{subfieldname}:compute",
            time.perf_counter() - started,
        )
        if self._subfield_array_cache is not None:
            self._subfield_array_cache[subfieldname] = data
        return data

    def _record_active_subfield_array_timing(self, name: str, seconds: float) -> None:
        timings = self._active_subfield_array_timings
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + seconds

    @contextmanager
    def _record_active_subfield_array_timing_block(self, name: str) -> Generator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self._record_active_subfield_array_timing(
                name,
                time.perf_counter() - started,
            )

    @contextmanager
    def _subfield_array_cache_scope(self) -> Generator[None]:
        previous_cache = self._subfield_array_cache
        self._subfield_array_cache = previous_cache if previous_cache is not None else {}
        try:
            yield
        finally:
            self._subfield_array_cache = previous_cache

    @contextmanager
    def _subfield_average_cache_scope(self) -> Generator[None]:
        previous_cache = self._subfield_average_cache
        self._subfield_average_cache = previous_cache if previous_cache is not None else {}
        try:
            yield
        finally:
            self._subfield_average_cache = previous_cache

    def _cached_subfield_average(
        self,
        key: tuple[str, str | None],
    ) -> object:
        cache = self._subfield_average_cache
        if cache is None or key not in cache:
            return _NO_CONSTANT_AVERAGE
        return cache[key]

    def _cache_subfield_average(
        self,
        key: tuple[str, str | None],
        value: object,
    ) -> None:
        cache = self._subfield_average_cache
        if cache is not None:
            cache[key] = _copy_average_value(value)

    def _compute_subfield_array(self, subfieldname: str) -> np.ndarray:
        if subfieldname in DERIVED_FIELD_NAMES:
            return self._derived_subfield_array(subfieldname)
        if subfieldname == "H_demag":
            return np.asarray(self._get_demag_nodal_field(), dtype=float)
        if subfieldname == "H_ext":
            h_ext = np.asarray(self._fields.get("H_ext", np.zeros(3)), dtype=float)
            if h_ext.shape != (3,):
                raise ValueError(f"H_ext must be a homogeneous 3-vector, got shape {h_ext.shape}.")
            point_count = len(self.mesh.points) if self.mesh is not None else 1
            return np.tile(h_ext, (point_count, 1))
        if subfieldname not in self._fields:
            raise KeyError(f"Unknown or unset subfield '{subfieldname}'.")
        return np.asarray(self._fields[subfieldname], dtype=float)

    def _is_subfield_available(self, subfieldname: str) -> bool:
        return self.is_subfield_available(subfieldname)
