from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from si.physical import SI


class SimulationFieldProbeMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def probe_subfield_siv(
        self,
        subfieldname: str,
        pos: Sequence[float],
        unit: SI | None = None,
    ) -> Any:
        """Probe a stored field at an SI position."""
        if subfieldname == "H_demag":
            total_started = time.perf_counter()
            try:
                started = time.perf_counter()
                within_bounds = self._probe_is_within_mesh_bounds(pos)
                self._record_probe_timing(
                    "bounds_check",
                    time.perf_counter() - started,
                )
                if not within_bounds:
                    return None

                started = time.perf_counter()
                points = self._mesh_points()
                probe = np.asarray(pos, dtype=float)
                self._record_probe_timing(
                    "prepare_probe",
                    time.perf_counter() - started,
                )

                started = time.perf_counter()
                legacy_null_vertex = self._probe_is_legacy_null_vertex(probe, points)
                self._record_probe_timing(
                    "legacy_null_vertex_check",
                    time.perf_counter() - started,
                )
                if legacy_null_vertex:
                    return None

                started = time.perf_counter()
                nodal_field = self._get_demag_nodal_field()
                self._record_probe_timing(
                    "demag_nodal_field",
                    time.perf_counter() - started,
                )

                started = time.perf_counter()
                result = self._probe_tetrahedral_field(probe, nodal_field)
                self._record_probe_timing(
                    "tetrahedral_interpolation",
                    time.perf_counter() - started,
                )
                return result
            finally:
                self._record_probe_timing(
                    "total",
                    time.perf_counter() - total_started,
                )
        if subfieldname == "H_ext":
            return self._fields.get("H_ext", np.zeros(3)).tolist()
        if subfieldname != "m":
            raise KeyError(f"Unknown or unset subfield '{subfieldname}'.")
        if self.mesh is None or "m" not in self._fields:
            return None

        points = np.asarray(self.mesh.points, dtype=float)
        if points.size == 0:
            return None
        probe = np.asarray(pos, dtype=float)
        nearest_index = int(np.argmin(np.linalg.norm(points - probe, axis=1)))
        return np.asarray(self._fields["m"][nearest_index], dtype=float).tolist()

    def _record_probe_timing(self, name: str, value: float) -> None:
        self.last_probe_timings_seconds[name] = (
            self.last_probe_timings_seconds.get(name, 0.0) + value
        )
