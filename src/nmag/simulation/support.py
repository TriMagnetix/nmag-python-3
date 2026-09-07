from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Sequence
from functools import cache, lru_cache
from typing import Any, cast

import numpy as np

from si.physical import SI


@cache
def _si_unit(unit: str) -> SI:
    return SI(1, unit)


@lru_cache(maxsize=1)
def _si_dimensionless() -> SI:
    return SI(1)


Vector = Sequence[float]
VectorField = Callable[[Sequence[float]], Sequence[float]]
VectorFieldInput = Sequence[float] | np.ndarray | VectorField
ScalarField = Callable[[Sequence[float]], float | SI]
ScalarFieldInput = float | SI | Sequence[float] | np.ndarray | ScalarField
MU0 = 4.0 * np.pi * 1.0e-7
MU0_OVER_4PI = 1.0 / (4.0 * np.pi)
# Legacy Nmag's default simulation units make one magnetisation unit 1e6 A/m.
# Its exported dmdt field uses that global reference rather than a material's Ms.
LEGACY_DMDT_MAGNETISATION_SCALE = 1.0e6
# Legacy Nmag's fixed simulation base length is one nanometre.  The demag
# solver's weak divergence is boxed with this base volume when exporting rho,
# independently of the unit_length supplied for a particular mesh.
LEGACY_RHO_BOXED_VOLUME_M3 = 1.0e-27
DERIVED_FIELD_NAMES = {
    "M",
    "H_total",
    "H_anis",
    "H_exch",
    "dmdt",
    "E_total",
    "E_ext",
    "E_anis",
    "E_exch",
    "E_demag",
    "phi",
    "rho",
    "pin",
    "dm_dcurrent",
}
_MESH_EDGE_SET_THRESHOLD = 4096
_NO_CONSTANT_AVERAGE = object()
_NO_COMPOSED_AVERAGE = object()


def _copy_average_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return cast(list[Any], value.tolist())
    if isinstance(value, list):
        return list(cast(list[Any], value))
    return value


def _as_vector3(values: Sequence[Any], *, unit: SI | None = None) -> list[float]:
    if len(values) != 3:
        raise ValueError(f"Expected a 3-vector, got {len(values)} components.")

    if unit is not None:
        scale = unit.in_units_of(_si_unit("A/m"))
        return [float(value) * scale for value in values]

    result: list[float] = []
    for value in values:
        if isinstance(value, SI):
            result.append(value.in_units_of(_si_unit("A/m")))
        else:
            result.append(float(value))
    return result


def _normalise_m(values: Sequence[Any]) -> list[float]:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"Magnetisation must be a 3-vector, got shape {vector.shape}.")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Magnetisation vectors must be non-zero.")

    return cast(list[float], (vector / norm).tolist())


def _dimensionless_value(value: Any, *, fieldname: str) -> float:
    result = value.in_units_of(_si_dimensionless()) if isinstance(value, SI) else float(value)
    if not np.isfinite(result):
        raise ValueError(f"{fieldname} values must be finite.")
    return result


def _scalar_nodal_field(
    values: ScalarFieldInput,
    points: Sequence[Sequence[float]],
    *,
    fieldname: str,
) -> np.ndarray:
    point_count = len(points)
    if callable(values):
        return np.asarray(
            [_dimensionless_value(values(point), fieldname=fieldname) for point in points],
            dtype=float,
        )
    if isinstance(values, SI) or np.isscalar(values):
        return np.full(
            point_count,
            _dimensionless_value(values, fieldname=fieldname),
            dtype=float,
        )

    array = np.asarray(values, dtype=float)
    if array.shape != (point_count,):
        raise ValueError(
            f"{fieldname} must be a scalar, callable, or one value per mesh point; "
            f"expected shape ({point_count},), got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{fieldname} values must be finite.")
    return np.array(array, dtype=float, copy=True)


def _vector_in_units(
    values: Sequence[Any],
    *,
    unit: SI | None,
    expected_unit: SI,
    fieldname: str,
) -> np.ndarray:
    if len(values) != 3:
        raise ValueError(f"{fieldname} values must be 3-vectors, got {len(values)} components.")
    if unit is not None:
        vector = np.asarray(values, dtype=float) * unit.in_units_of(expected_unit)
    else:
        vector = np.asarray(
            [
                value.in_units_of(expected_unit) if isinstance(value, SI) else float(value)
                for value in values
            ],
            dtype=float,
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{fieldname} values must be finite.")
    return vector


def _vector_nodal_field(
    values: VectorFieldInput,
    points: Sequence[Sequence[float]],
    *,
    unit: SI | None,
    expected_unit: SI,
    fieldname: str,
) -> np.ndarray:
    if callable(values):
        return np.asarray(
            [
                _vector_in_units(
                    values(point),
                    unit=unit,
                    expected_unit=expected_unit,
                    fieldname=fieldname,
                )
                for point in points
            ],
            dtype=float,
        )

    array = np.asarray(values, dtype=object if unit is None else float)
    if array.shape == (3,):
        vector = _vector_in_units(
            cast(Sequence[Any], values),
            unit=unit,
            expected_unit=expected_unit,
            fieldname=fieldname,
        )
        return np.tile(vector, (len(points), 1))
    if array.shape != (len(points), 3):
        raise ValueError(
            f"{fieldname} must be a 3-vector, callable, or one vector per mesh point; "
            f"expected shape ({len(points)}, 3), got {array.shape}."
        )
    return np.asarray(
        [
            _vector_in_units(
                cast(Sequence[Any], row),
                unit=unit,
                expected_unit=expected_unit,
                fieldname=fieldname,
            )
            for row in array
        ],
        dtype=float,
    )


def _flatten_materials(region_materials: Iterable[tuple[str, Any]]) -> list[Any]:
    materials: list[Any] = []
    seen_names: set[str] = set()
    for _, entry in region_materials:
        region_mats: list[Any] = cast(list[Any], entry) if isinstance(entry, list) else [entry]
        for material in region_mats:
            if material.name not in seen_names:
                seen_names.add(material.name)
                materials.append(material)
    return materials


def _simulation_compatibility_binding(
    name: str, fallback: Callable[..., Any]
) -> Callable[..., Any]:
    """Use a legacy simulation-module override when tests or extensions provide one."""

    simulation_module = sys.modules.get("nmag.simulation")
    return getattr(simulation_module, name, fallback) if simulation_module is not None else fallback
