from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ...backends import (
    _load_rust_accelerator,
    _selected_maxangle_backend,
)
from ...demag import _normalised
from ..support import _simulation_compatibility_binding


class SimulationFieldMaxangleMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def get_maxangle_average(self, field_name: str) -> float | None:
        timings: dict[str, float] = {}
        total_started = time.perf_counter()
        self.last_maxangle_timings_seconds = {}
        inputs = self._maxangle_inputs(field_name, timings)
        if inputs is None:
            return self._finish_maxangle(timings, total_started, None)
        m, simplices = inputs
        if self._is_uniform_magnetisation(m, timings):
            return self._finish_maxangle(timings, total_started, 0.0)
        backend = _simulation_compatibility_binding(
            "_selected_maxangle_backend", _selected_maxangle_backend
        )(getattr(self, "config", None))
        if backend == "rust":
            return self._finish_maxangle(
                timings,
                total_started,
                self._timed_rust_maxangle(m, simplices, timings),
            )
        return self._finish_maxangle(
            timings,
            total_started,
            self._timed_python_maxangle(m, timings),
        )

    def _finish_maxangle(
        self,
        timings: dict[str, float],
        total_started: float,
        value: float | None,
    ) -> float | None:
        timings["total"] = time.perf_counter() - total_started
        self.last_maxangle_timings_seconds = dict(sorted(timings.items()))
        return value

    def _maxangle_inputs(
        self,
        field_name: str,
        timings: dict[str, float] | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if field_name != "m" or "m" not in self._fields or self.mesh is None:
            return None
        started = time.perf_counter()
        m = np.asarray(self._fields["m"], dtype=float)
        simplices = np.asarray(self._require_mesh().simplices, dtype=int)
        if timings is not None:
            timings["input_arrays"] = time.perf_counter() - started
        if m.ndim != 2 or m.shape[1] != 3 or simplices.ndim != 2 or simplices.shape[1] < 2:
            return None
        return (m, simplices) if len(m) > 0 else None

    def _is_uniform_magnetisation(self, m: np.ndarray, timings: dict[str, float]) -> bool:
        started = time.perf_counter()
        is_uniform = bool(np.all(m == m[0]))
        timings["uniform_check"] = time.perf_counter() - started
        return is_uniform

    def _timed_rust_maxangle(
        self,
        m: np.ndarray,
        simplices: np.ndarray,
        timings: dict[str, float],
    ) -> float:
        started = time.perf_counter()
        maxangle = self._maxangle_average_rust(m, simplices)
        timings["mesh_edges_and_angles:rust"] = time.perf_counter() - started
        return maxangle

    def _timed_python_maxangle(self, m: np.ndarray, timings: dict[str, float]) -> float:
        started = time.perf_counter()
        edges = self._mesh_edges()
        timings["mesh_edges"] = time.perf_counter() - started
        if len(edges) == 0:
            return 0.0

        started = time.perf_counter()
        # set_m normalises stored magnetisation vectors, so maxangle can use
        # the cached nodal rows directly instead of renormalising every edge.
        edge_values = m[edges]
        dots = np.einsum("ij,ij->i", edge_values[:, 0, :], edge_values[:, 1, :])
        if dots.size == 0:
            timings["edge_angles"] = time.perf_counter() - started
            return 0.0
        bounded_dots = np.minimum(np.maximum(dots, -1.0), 1.0)
        maxangle = float(np.max(np.arccos(bounded_dots))) * 180.0 / np.pi
        timings["edge_angles"] = time.perf_counter() - started
        return maxangle

    def _maxangle_average_rust(self, m: np.ndarray, simplices: np.ndarray) -> float:
        rust_accel = _simulation_compatibility_binding(
            "_load_rust_accelerator", _load_rust_accelerator
        )("NmagConfig.accelerator['maxangle']")
        try:
            return float(rust_accel.maxangle_between_edges(m, simplices))
        except AttributeError as exc:
            raise RuntimeError(
                "NmagConfig.accelerator['maxangle']='rust' requires an nmag_accel build with "
                "maxangle_between_edges. Rebuild it with "
                "`maturin develop --release --manifest-path rust/nmag_accel/Cargo.toml`."
            ) from exc

    def _maxangle_average_reference(self, field_name: str) -> float | None:
        inputs = self._maxangle_inputs(field_name, None)
        if inputs is None:
            return None
        m, simplices = inputs
        if np.all(m == m[0]):
            return 0.0
        return self._reference_maxangle_from_simplices(m, simplices)

    def _reference_maxangle_from_simplices(
        self,
        m: np.ndarray,
        simplices: np.ndarray,
    ) -> float:
        max_angle = 0.0
        seen_edges: set[tuple[int, int]] = set()
        for simplex in simplices:
            for left_position, left in enumerate(simplex):
                for right in simplex[left_position + 1 :]:
                    edge = (min(int(left), int(right)), max(int(left), int(right)))
                    if edge in seen_edges:
                        continue
                    seen_edges.add(edge)
                    left_m = _simulation_compatibility_binding("_normalised", _normalised)(
                        m[edge[0]]
                    )
                    right_m = _simulation_compatibility_binding("_normalised", _normalised)(
                        m[edge[1]]
                    )
                    raw_dot = float(np.sum(left_m * right_m))
                    dot = min(1.0, max(-1.0, raw_dot))
                    max_angle = max(max_angle, float(np.arccos(dot)))
        return max_angle * 180.0 / np.pi
