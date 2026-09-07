from __future__ import annotations

import csv
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from si.physical import SI
from simulation.data_writer_collection import (
    DataWriterCollectionMixin,
    SimulationSource,
    _SubfieldRequest,
)
from simulation.quantity import Quantity, known_quantities

log = logging.getLogger("nmag")


class DataWriter(DataWriterCollectionMixin):
    def __init__(self, ndt_filename: Path, h5_filename: Path, *, append: bool = False) -> None:
        """
        :param ndt_filename: Path to the tabular output (.ndt).
        :param h5_filename: Path to the spatial output (.h5).
        """
        self.ndt_filename = ndt_filename
        self.h5_filename = h5_filename
        self.quantities = known_quantities

        self.quantities_by_name: dict[str, Quantity] = {q.name: q for q in known_quantities}

        self._header_written: bool = False
        self._column_names: list[str] | None = None
        self._column_index_by_name: dict[str, int] | None = None
        self._column_units: dict[str, str] | None = None
        self._subfield_requests: list[_SubfieldRequest] = []
        self._writes_maxangle: bool = False
        self._last_saved_step: int = -1
        self.last_save_timings_seconds: dict[str, float] = {}
        self._active_save_timings: dict[str, float] | None = None
        self._append_schema_pending = False
        if append and self.ndt_filename.exists():
            self._recover_ndt_schema()
            self._append_schema_pending = True

    def _recover_ndt_schema(self) -> None:
        """Recover the schema and last step from an NDT file opened for append."""

        with open(self.ndt_filename, newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.reader(stream, delimiter="\t") if row]
        if len(rows) < 2 or not rows[0][0].startswith("# Simulation:"):
            raise ValueError(f"Cannot append to malformed NDT output: {self.ndt_filename}.")
        column_names = rows[1]
        if len(column_names) != len(set(column_names)):
            raise ValueError(f"Cannot append to NDT output with duplicate columns: {self.ndt_filename}.")
        self._column_names = column_names
        self._column_index_by_name = {name: index for index, name in enumerate(column_names)}
        self._column_units = {}
        self._header_written = True
        step_index = self._column_index_by_name.get("step")
        if step_index is not None:
            for row in reversed(rows[2:]):
                if len(row) != len(column_names):
                    continue
                try:
                    self._last_saved_step = int(float(row[step_index]))
                except ValueError:
                    continue
                break

    def save(
        self,
        source: SimulationSource,
        fields: str | list[str] | None = None,
        avoid_same_step: bool = False,
    ) -> None:

        current_step = source.step
        if avoid_same_step and current_step == self._last_saved_step:
            return

        timings: dict[str, float] = {}
        self._active_save_timings = timings
        total_started = time.perf_counter()
        try:
            with self._record_timing("write_ndt_row"):
                self._write_ndt_row(source)

            if fields is not None:
                with self._record_timing("save_spatial_fields"):
                    self._trigger_field_save(source, fields)
        finally:
            timings["total"] = time.perf_counter() - total_started
            self.last_save_timings_seconds = dict(sorted(timings.items()))
            self._active_save_timings = None

        self._last_saved_step = current_step

    @contextmanager
    def _record_timing(self, name: str) -> Generator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            timings = self._active_save_timings
            if timings is not None:
                timings[name] = timings.get(name, 0.0) + time.perf_counter() - started

    def _trigger_field_save(self, source: SimulationSource, fields: str | list[str]) -> None:
        with self._record_timing("spatial_detail:field_selection"):
            field_names_to_save: list[str] = []

            if fields == "all":
                field_names_to_save = source.get_all_field_names()
            elif isinstance(fields, list):
                field_names_to_save = fields
            else:
                raise ValueError(f"Invalid fields argument: {fields}")

        if field_names_to_save:
            with self._record_timing("spatial_detail:write_dispatch"):
                source.save_spatial_fields(
                    filename=str(self.h5_filename), fieldnames=field_names_to_save
                )
            self._record_source_spatial_save_timings(source)

    def _write_ndt_row(self, source: SimulationSource) -> None:
        quantities: list[Quantity] = []
        with self._record_timing("gather_ndt_columns"):
            if self._append_schema_pending:
                with self._record_timing("gather_ndt_append_schema_columns"):
                    columns, _quantities = self._gather_data(source)
            elif self._header_written:
                with self._record_timing("gather_ndt_existing_schema_columns"):
                    columns = self._gather_existing_schema_data(source)
                quantities: list[Quantity] = []
            else:
                with self._record_timing("gather_ndt_initial_schema_columns"):
                    columns, quantities = self._gather_data(source)

        if not self._header_written:
            with self._record_timing("build_ndt_header_schema"):
                col_names: list[str] = []
                col_units: dict[str, str] = {}

                for (name, _), qty in zip(columns, quantities, strict=True):
                    col_names.append(name)
                    if qty.units:
                        col_units[name] = qty.units.dens_str()
                    else:
                        col_units[name] = "-"

                column_index_by_name = {name: index for index, name in enumerate(col_names)}
            with self._record_timing("format_ndt_values"):
                row_values: list[Any] = [self._ndt_cell_value(value) for _, value in columns]

            with open(self.ndt_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                with self._record_timing("write_ndt_header"):
                    f.write(f"# Simulation: {source.name}\n")
                    writer.writerow(col_names)

                with self._record_timing("write_ndt_values"):
                    writer.writerow(row_values)

            self._column_names = col_names
            self._column_index_by_name = column_index_by_name
            self._column_units = col_units
            self._header_written = True
            return

        if self._column_names is None:
            log.error("Column names not initialized.")
            return

        with self._record_timing("validate_ndt_schema"):
            actual_names = [name for name, _ in columns]
            if actual_names != self._column_names:
                raise ValueError(
                    "Cannot append output with a different NDT schema. "
                    f"Expected {self._column_names!r}, got {actual_names!r}."
                )
            self._append_schema_pending = False

        with self._record_timing("format_ndt_values"):
            row_values = self._existing_schema_row_values(columns)

        with self._record_timing("write_ndt_values"):
            with open(self.ndt_filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(row_values)

    def _existing_schema_row_values(
        self,
        columns: list[tuple[str, Any]],
    ) -> list[Any]:
        if self._column_names is None:
            return []
        if self._column_index_by_name is None:
            self._column_index_by_name = {
                name: index for index, name in enumerate(self._column_names)
            }

        row_values: list[Any] = [None] * len(self._column_names)
        for name, value in columns:
            index = self._column_index_by_name.get(name)
            if index is not None:
                row_values[index] = self._ndt_cell_value(value)
        return row_values

    @staticmethod
    def _ndt_cell_value(value: Any) -> Any:
        if isinstance(value, SI):
            return value.magnitude
        return value

    def _record_source_average_timings(
        self,
        source: SimulationSource,
        prefix: str,
    ) -> None:
        timings = self._active_save_timings
        if timings is None:
            return
        source_timings = getattr(
            source,
            "last_subfield_average_timings_seconds",
            None,
        )
        if not isinstance(source_timings, dict):
            return
        for name, seconds in cast(dict[object, object], source_timings).items():
            if isinstance(seconds, (int, float)):
                timing_name = f"average_detail:{prefix}:{name!s}"
                timings[timing_name] = timings.get(timing_name, 0.0) + float(seconds)

    def _record_source_spatial_save_timings(self, source: SimulationSource) -> None:
        timings = self._active_save_timings
        if timings is None:
            return
        source_timings = getattr(
            source,
            "last_spatial_save_timings_seconds",
            None,
        )
        if not isinstance(source_timings, dict):
            return
        for name, seconds in cast(dict[object, object], source_timings).items():
            if isinstance(seconds, (int, float)):
                timing_name = f"spatial_detail:{name!s}"
                timings[timing_name] = timings.get(timing_name, 0.0) + float(seconds)

    def _record_source_maxangle_timings(
        self,
        source: SimulationSource,
        prefix: str,
    ) -> None:
        timings = self._active_save_timings
        if timings is None:
            return
        source_timings = getattr(
            source,
            "last_maxangle_timings_seconds",
            None,
        )
        if not isinstance(source_timings, dict):
            return
        for name, seconds in cast(dict[object, object], source_timings).items():
            if isinstance(seconds, (int, float)):
                timing_name = f"average_detail:{prefix}:{name!s}"
                timings[timing_name] = timings.get(timing_name, 0.0) + float(seconds)
