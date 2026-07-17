from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from si.physical import SI
from simulation.quantity import Quantity

log = logging.getLogger("nmag")


@dataclass(frozen=True)
class _SubfieldRequest:
    field_name: str
    prefix: str
    mat_name: str | None = None


class MaterialSource(Protocol):
    name: str


@runtime_checkable
class SimulationSource(Protocol):
    name: str
    last_spatial_save_timings_seconds: dict[str, float]
    last_maxangle_timings_seconds: dict[str, float]

    @property
    def id(self) -> int: ...

    @property
    def step(self) -> int: ...

    @property
    def stage(self) -> int: ...

    @property
    def stage_step(self) -> int: ...

    @property
    def time(self) -> SI: ...

    @property
    def stage_time(self) -> SI: ...

    @property
    def real_time(self) -> SI: ...

    @property
    def last_step_dt(self) -> SI: ...

    def get_subfield_average(self, subfieldname: str, mat_name: str | None = None) -> Any: ...

    def get_maxangle_average(self, field_name: str) -> float | None: ...

    def get_materials_of_field(self, field_name: str) -> list[MaterialSource]: ...

    def get_all_field_names(self) -> list[str]: ...

    def save_spatial_fields(self, filename: str, fieldnames: list[str]) -> None: ...


class DataWriterCollectionMixin:
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def _base_columns(self, source: SimulationSource) -> list[tuple[str, Any]]:
        lt = time.localtime()
        lt_str = f"{lt[0]:04d}/{lt[1]:02d}/{lt[2]:02d}-{lt[3]:02d}:{lt[4]:02d}:{lt[5]:02d}"

        return [
            ("id", source.id),
            ("step", source.step),
            ("stage_step", source.stage_step),
            ("stage", source.stage),
            ("last_step_dt", source.last_step_dt),
            ("time", source.time),
            ("stage_time", source.stage_time),
            ("real_time", source.real_time),
            ("unixtime", SI(time.time(), "s")),
            ("localtime", lt_str),
        ]

    def _append_average_columns(
        self,
        columns: list[tuple[str, Any]],
        current_quantities: list[Quantity] | None,
        quantity: Quantity | None,
        prefix: str,
        average: Any,
    ) -> None:
        if isinstance(average, list):
            for i, comp_value in enumerate(cast(list[Any], average)):
                comp_name = f"{prefix}_{i}"
                columns.append((comp_name, comp_value))
                if current_quantities is not None and quantity is not None:
                    current_quantities.append(quantity.sub_quantity(comp_name))
        else:
            columns.append((prefix, average))
            if current_quantities is not None and quantity is not None:
                current_quantities.append(quantity.sub_quantity(prefix))

    @staticmethod
    def _source_has_subfield(source: SimulationSource, field_name: str) -> bool:
        predicate = getattr(source, "is_subfield_available", None)
        if predicate is None:
            return True
        try:
            return bool(predicate(field_name))
        except KeyError:
            log.debug("Skipping unavailable subfield %s", field_name)
            return False

    def _gather_data(
        self, source: SimulationSource
    ) -> tuple[list[tuple[str, Any]], list[Quantity]]:
        columns = self._base_columns(source)
        current_quantities: list[Quantity] = [self.quantities_by_name[name] for name, _ in columns]
        subfield_requests: list[_SubfieldRequest] = []
        for quantity in self.quantities:
            self._collect_quantity_subfields(
                source,
                quantity,
                columns,
                current_quantities,
                subfield_requests,
            )

        writes_maxangle = self._collect_initial_maxangle(
            source,
            columns,
            current_quantities,
        )

        self._subfield_requests = subfield_requests
        self._writes_maxangle = writes_maxangle

        return columns, current_quantities

    def _collect_quantity_subfields(
        self,
        source: SimulationSource,
        quantity: Quantity,
        columns: list[tuple[str, Any]],
        current_quantities: list[Quantity],
        subfield_requests: list[_SubfieldRequest],
    ) -> None:
        field_name = quantity.name
        if quantity.type not in ["field", "pfield"] or not self._source_has_subfield(
            source,
            field_name,
        ):
            return
        if "?" not in (quantity.signature or ""):
            self._collect_subfield_average(
                source,
                field_name,
                field_name,
                quantity,
                columns,
                current_quantities,
                subfield_requests,
            )
            return
        for material in source.get_materials_of_field(field_name):
            self._collect_subfield_average(
                source,
                field_name,
                f"{field_name}_{material.name}",
                quantity,
                columns,
                current_quantities,
                subfield_requests,
                mat_name=material.name,
            )

    def _collect_subfield_average(
        self,
        source: SimulationSource,
        field_name: str,
        prefix: str,
        quantity: Quantity,
        columns: list[tuple[str, Any]],
        current_quantities: list[Quantity],
        subfield_requests: list[_SubfieldRequest],
        *,
        mat_name: str | None = None,
    ) -> None:
        average = self._subfield_average(source, field_name, prefix, mat_name)
        if average is None:
            return
        self._append_average_columns(columns, current_quantities, quantity, prefix, average)
        subfield_requests.append(_SubfieldRequest(field_name, prefix, mat_name))

    def _subfield_average(
        self,
        source: SimulationSource,
        field_name: str,
        prefix: str,
        mat_name: str | None = None,
    ) -> Any | None:
        try:
            with self._record_timing(f"average:{prefix}"):
                average = source.get_subfield_average(field_name, mat_name)
        except KeyError:
            log.debug("Skipping unavailable subfield average %s", prefix)
            average = None
        self._record_source_average_timings(source, prefix)
        return average

    def _collect_initial_maxangle(
        self,
        source: SimulationSource,
        columns: list[tuple[str, Any]],
        current_quantities: list[Quantity],
    ) -> bool:
        maxangle_quantity = self.quantities_by_name.get("maxangle")
        maxangle = self._maxangle_average(source)
        if maxangle_quantity is None or maxangle is None:
            return False
        columns.append(("maxangle_m_Py", maxangle))
        current_quantities.append(maxangle_quantity.sub_quantity("maxangle_m_Py"))
        return True

    def _maxangle_average(self, source: SimulationSource) -> float | None:
        try:
            with self._record_timing("average:maxangle_m_Py"):
                maxangle = source.get_maxangle_average("m")
        except KeyError:
            log.debug("Skipping unavailable max-angle average for m")
            maxangle = None
        self._record_source_maxangle_timings(source, "maxangle_m_Py")
        return maxangle

    def _gather_existing_schema_data(self, source: SimulationSource) -> list[tuple[str, Any]]:
        columns = self._base_columns(source)

        for request in self._subfield_requests:
            try:
                with self._record_timing(f"average:{request.prefix}"):
                    avg = source.get_subfield_average(request.field_name, request.mat_name)
                self._record_source_average_timings(source, request.prefix)
                if avg is None:
                    continue
            except KeyError:
                self._record_source_average_timings(source, request.prefix)
                log.debug("Skipping unavailable subfield average %s", request.prefix)
                continue

            self._append_average_columns(
                columns,
                None,
                None,
                request.prefix,
                avg,
            )

        if self._writes_maxangle:
            maxangle = self._maxangle_average(source)
            if maxangle is not None:
                columns.append(("maxangle_m_Py", maxangle))

        return columns
