"""
Original file: https://github.com/nmag-project/nmag-src/blob/master/interface/nsim/si_units/lib.py
Support for Physical quantities carrying SI units.

This module provides a backward-compatible wrapper around the 'pint' library,
replicating the interface of a legacy Physical/SI class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any, TypeAlias, cast

# Pint's runtime classes are constructed by the active application registry.
# Keep the dynamic boundary explicit while retaining concrete types internally.
_pint_module_cache: Any | None = None
_pint_registry_cache: Any | None = None
_pint_quantity_type_cache: Any | None = None
_pint_unit_type_cache: Any | None = None
_pint_quantity_factory_cache: Any | None = None

Dims: TypeAlias = tuple[int, int, int, int, int, int, int]


class _LazyUnitRegistry:
    """Proxy Pint's application registry without constructing it at import time."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_pint_registry(), name)


ureg = _LazyUnitRegistry()


@dataclass(frozen=True)
class _FastUnit:
    dims: Dims
    scale: float
    compact: str
    long: str

    def __str__(self) -> str:
        return self.long

    def __format__(self, spec: str) -> str:
        return self.compact if spec == "~" else self.long


class _FastQuantity:
    """Small exact SI quantity used for common MVP units before Pint is needed."""

    def __init__(self, magnitude: int | float, unit: _FastUnit) -> None:
        self.magnitude = float(magnitude)
        self.units = unit

    @property
    def dimensionless(self) -> bool:
        return all(power == 0 for power in self.units.dims)

    def is_compatible_with(self, other: Any) -> bool:
        if isinstance(other, _FastQuantity):
            return self.units.dims == other.units.dims
        if isinstance(other, (int, float)):
            return self.dimensionless or float(other) == 0.0
        return self._as_pint().is_compatible_with(other)

    def to_base_units(self) -> _FastQuantity:
        return _FastQuantity(
            self.magnitude * self.units.scale,
            _unit_from_dims(self.units.dims),
        )

    def _as_pint(self) -> Any:
        # Generated fast units carry their factor separately from their compact
        # spelling. Convert through base SI so the factor is never lost.
        if self.dimensionless:
            return _pint_registry().Quantity(self.magnitude * self.units.scale)
        return _pint_registry().Quantity(
            self.magnitude * self.units.scale,
            _pint_registry().Unit(_compact_from_dims(self.units.dims)),
        )

    def __abs__(self) -> _FastQuantity:
        return _FastQuantity(abs(self.magnitude), self.units)

    def __neg__(self) -> _FastQuantity:
        return _FastQuantity(-self.magnitude, self.units)

    def __pos__(self) -> _FastQuantity:
        return _FastQuantity(self.magnitude, self.units)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FastQuantity):
            return self.units.dims == other.units.dims and math.isclose(
                self.magnitude * self.units.scale,
                other.magnitude * other.units.scale,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        if isinstance(other, (int, float)):
            return self.magnitude == 0.0 and float(other) == 0.0
        return self._as_pint() == other

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __lt__(self, other: Any) -> bool:
        return self.to_base_units().magnitude < _fast_base_magnitude(other, self.units.dims)

    def __le__(self, other: Any) -> bool:
        return self.to_base_units().magnitude <= _fast_base_magnitude(other, self.units.dims)

    def __gt__(self, other: Any) -> bool:
        return self.to_base_units().magnitude > _fast_base_magnitude(other, self.units.dims)

    def __ge__(self, other: Any) -> bool:
        return self.to_base_units().magnitude >= _fast_base_magnitude(other, self.units.dims)

    def __add__(self, other: Any) -> Any:
        if isinstance(other, (int, float)) and float(other) == 0.0:
            return _FastQuantity(self.magnitude, self.units)
        other_base = _fast_base_magnitude(other, self.units.dims)
        return _FastQuantity(
            (self.to_base_units().magnitude + other_base) / self.units.scale,
            self.units,
        )

    def __radd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, (int, float)) and float(other) == 0.0:
            return _FastQuantity(self.magnitude, self.units)
        other_base = _fast_base_magnitude(other, self.units.dims)
        return _FastQuantity(
            (self.to_base_units().magnitude - other_base) / self.units.scale,
            self.units,
        )

    def __rsub__(self, other: Any) -> Any:
        if isinstance(other, (int, float)) and float(other) == 0.0:
            return _FastQuantity(-self.magnitude, self.units)
        return _quantity_from_pint(other) - self

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (int, float)):
            return _FastQuantity(self.magnitude * float(other), self.units)
        if isinstance(other, _FastQuantity):
            dims = _add_dims(self.units.dims, other.units.dims)
            unit = _unit_from_dims(dims, self.units.scale * other.units.scale)
            return _FastQuantity(self.magnitude * other.magnitude, unit)
        return self._as_pint() * other

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, (int, float)):
            return _FastQuantity(self.magnitude / float(other), self.units)
        if isinstance(other, _FastQuantity):
            dims = _subtract_dims(self.units.dims, other.units.dims)
            unit = _unit_from_dims(dims, self.units.scale / other.units.scale)
            return _FastQuantity(self.magnitude / other.magnitude, unit)
        return self._as_pint() / other

    def __rtruediv__(self, other: Any) -> Any:
        if isinstance(other, (int, float)):
            dims = _negate_dims(self.units.dims)
            unit = _unit_from_dims(dims, 1.0 / self.units.scale)
            return _FastQuantity(float(other) / self.magnitude, unit)
        return other / self._as_pint()

    def __pow__(self, exponent: int | float) -> Any:
        if not isinstance(exponent, int):
            return self._as_pint() ** exponent
        dims = _multiply_dims(self.units.dims, exponent)
        unit = _unit_from_dims(dims, self.units.scale**exponent)
        return _FastQuantity(self.magnitude**exponent, unit)

    def __str__(self) -> str:
        if self.dimensionless:
            return str(self.magnitude)
        return f"{self.magnitude} {self.units.long}"

    def __format__(self, spec: str) -> str:
        if spec == "~":
            if self.dimensionless:
                return str(self.magnitude)
            return f"{self.magnitude} {self.units.compact}"
        return str(self)


_DIMS_DIMLESS: Dims = (0, 0, 0, 0, 0, 0, 0)
_DIMS_KG: Dims = (1, 0, 0, 0, 0, 0, 0)
_DIMS_M: Dims = (0, 1, 0, 0, 0, 0, 0)
_DIMS_S: Dims = (0, 0, 1, 0, 0, 0, 0)
_DIMS_A: Dims = (0, 0, 0, 1, 0, 0, 0)
_DIMS_K: Dims = (0, 0, 0, 0, 1, 0, 0)
_DIMS_MOL: Dims = (0, 0, 0, 0, 0, 1, 0)
_DIMS_CD: Dims = (0, 0, 0, 0, 0, 0, 1)


def _dims(
    *,
    kg: int = 0,
    m: int = 0,
    s: int = 0,
    A: int = 0,
    K: int = 0,
    mol: int = 0,
    cd: int = 0,
) -> Dims:
    return (kg, m, s, A, K, mol, cd)


def _add_dims(left: Dims, right: Dims) -> Dims:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
        left[4] + right[4],
        left[5] + right[5],
        left[6] + right[6],
    )


def _subtract_dims(left: Dims, right: Dims) -> Dims:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
        left[3] - right[3],
        left[4] - right[4],
        left[5] - right[5],
        left[6] - right[6],
    )


def _negate_dims(dims: Dims) -> Dims:
    return (-dims[0], -dims[1], -dims[2], -dims[3], -dims[4], -dims[5], -dims[6])


def _multiply_dims(dims: Dims, factor: int) -> Dims:
    return (
        dims[0] * factor,
        dims[1] * factor,
        dims[2] * factor,
        dims[3] * factor,
        dims[4] * factor,
        dims[5] * factor,
        dims[6] * factor,
    )


_FAST_UNITS: dict[str, _FastUnit] = {
    "": _FastUnit(_DIMS_DIMLESS, 1.0, "", "dimensionless"),
    "1": _FastUnit(_DIMS_DIMLESS, 1.0, "", "dimensionless"),
    "m": _FastUnit(_DIMS_M, 1.0, "m", "meter"),
    "meter": _FastUnit(_DIMS_M, 1.0, "m", "meter"),
    "km": _FastUnit(_DIMS_M, 1000.0, "km", "kilometer"),
    "kilometer": _FastUnit(_DIMS_M, 1000.0, "km", "kilometer"),
    "kg": _FastUnit(_DIMS_KG, 1.0, "kg", "kilogram"),
    "kilogram": _FastUnit(_DIMS_KG, 1.0, "kg", "kilogram"),
    "s": _FastUnit(_DIMS_S, 1.0, "s", "second"),
    "second": _FastUnit(_DIMS_S, 1.0, "s", "second"),
    "A": _FastUnit(_DIMS_A, 1.0, "A", "ampere"),
    "ampere": _FastUnit(_DIMS_A, 1.0, "A", "ampere"),
    "K": _FastUnit(_DIMS_K, 1.0, "K", "kelvin"),
    "kelvin": _FastUnit(_DIMS_K, 1.0, "K", "kelvin"),
    "mol": _FastUnit(_DIMS_MOL, 1.0, "mol", "mole"),
    "mole": _FastUnit(_DIMS_MOL, 1.0, "mol", "mole"),
    "cd": _FastUnit(_DIMS_CD, 1.0, "cd", "candela"),
    "candela": _FastUnit(_DIMS_CD, 1.0, "cd", "candela"),
    "A/m": _FastUnit(_dims(A=1, m=-1), 1.0, "A/m", "ampere / meter"),
    "A/m**2": _FastUnit(_dims(A=1, m=-2), 1.0, "A/m**2", "ampere / meter ** 2"),
    "A/m/s": _FastUnit(_dims(A=1, m=-1, s=-1), 1.0, "A/m/s", "ampere / meter / second"),
    "m/s": _FastUnit(_dims(m=1, s=-1), 1.0, "m/s", "meter / second"),
    "km/h": _FastUnit(_dims(m=1, s=-1), 1000.0 / 3600.0, "km/h", "kilometer / hour"),
    "1/s": _FastUnit(_dims(s=-1), 1.0, "1/s", "1 / second"),
    "m/A/s": _FastUnit(_dims(m=1, A=-1, s=-1), 1.0, "m/A/s", "meter / ampere / second"),
    "J": _FastUnit(_dims(kg=1, m=2, s=-2), 1.0, "J", "joule"),
    "J/m": _FastUnit(_dims(kg=1, m=1, s=-2), 1.0, "J/m", "joule / meter"),
    "J/m**3": _FastUnit(_dims(kg=1, m=-1, s=-2), 1.0, "J/m**3", "joule / meter ** 3"),
    "N": _FastUnit(_dims(kg=1, m=1, s=-2), 1.0, "N", "newton"),
    "N*m": _FastUnit(_dims(kg=1, m=2, s=-2), 1.0, "m*N", "meter * newton"),
}


def _normalized_unit_string(unit: str) -> str:
    return unit.strip().replace(" ", "").replace("^", "**")


def _unit_from_dims(
    dims: Dims,
    scale: float = 1.0,
) -> _FastUnit:
    for unit in _FAST_UNITS.values():
        if unit.dims == dims and unit.scale == scale:
            return unit
    compact = _compact_from_dims(dims)
    return _FastUnit(dims, scale, compact, compact or "dimensionless")


def _compact_from_dims(dims: Dims) -> str:
    names = ("kg", "m", "s", "A", "K", "mol", "cd")
    parts: list[str] = []
    for name, power in zip(names, dims, strict=True):
        if power == 0:
            continue
        parts.append(name if power == 1 else f"{name}**{power}")
    return "*".join(parts)


def _fast_unit(unit: str) -> _FastUnit | None:
    return _FAST_UNITS.get(_normalized_unit_string(unit))


def _fast_dimensions_list(dimensions: list[Any]) -> _FastUnit | None:
    if not dimensions:
        return _FAST_UNITS[""]
    if len(dimensions) % 2 != 0:
        raise ValueError(
            "Physical quantity: Bad dimensions list given! "
            "Must contain pairs of unit names and powers. "
            f"Received: {dimensions}"
        )
    dims = _DIMS_DIMLESS
    for i in range(0, len(dimensions), 2):
        unit = _fast_unit(str(dimensions[i]))
        if unit is None or unit.scale != 1.0:
            return None
        power = dimensions[i + 1]
        if not isinstance(power, int):
            return None
        dims = _add_dims(dims, _multiply_dims(unit.dims, power))
    return _unit_from_dims(dims)


def _fast_base_magnitude(other: Any, expected_dims: Dims) -> float:
    if isinstance(other, _FastQuantity):
        if other.units.dims != expected_dims:
            raise _pint_module().errors.DimensionalityError(
                other.units,
                _unit_from_dims(expected_dims),
            )
        return other.to_base_units().magnitude
    if isinstance(other, (int, float)):
        if any(expected_dims) and float(other) != 0.0:
            raise _pint_module().errors.DimensionalityError(
                "dimensionless",
                _unit_from_dims(expected_dims),
            )
        return float(other)
    return _quantity_from_pint(other).to_base_units().magnitude


def _quantity_from_pint(value: Any) -> Any:
    if isinstance(value, _FastQuantity):
        return value
    quantity_type, unit_type, quantity_factory = _pint_handles()
    if isinstance(value, unit_type):
        return quantity_factory(1.0, value)
    if isinstance(value, quantity_type):
        return value
    return quantity_factory(value)


def _as_pint_quantity(value: Any) -> Any:
    if isinstance(value, _FastQuantity):
        return value._as_pint()
    return value


def _quantities_compatible(left: Any, right: Any) -> bool:
    if isinstance(left, _FastQuantity) and isinstance(right, _FastQuantity):
        return left.units.dims == right.units.dims
    return _as_pint_quantity(left).is_compatible_with(_as_pint_quantity(right))


def _base_magnitude(value: Any) -> float:
    if isinstance(value, _FastQuantity):
        return value.to_base_units().magnitude
    return value.to_base_units().magnitude


def _pint_module() -> Any:
    global _pint_module_cache
    if _pint_module_cache is None:
        import pint

        _pint_module_cache = pint
    return _pint_module_cache


def _pint_registry() -> Any:
    global _pint_registry_cache
    if _pint_registry_cache is None:
        _pint_registry_cache = _pint_module().get_application_registry()
    return _pint_registry_cache


def _pint_handles() -> tuple[Any, Any, Any]:
    global _pint_quantity_factory_cache, _pint_quantity_type_cache, _pint_unit_type_cache
    if (
        _pint_quantity_type_cache is None
        or _pint_unit_type_cache is None
        or _pint_quantity_factory_cache is None
    ):
        registry = _pint_registry()
        quantity = registry.Quantity
        _pint_quantity_factory_cache = quantity
        _pint_quantity_type_cache = quantity
        _pint_unit_type_cache = registry.Unit
    assert _pint_quantity_type_cache is not None
    assert _pint_unit_type_cache is not None
    assert _pint_quantity_factory_cache is not None
    return _pint_quantity_type_cache, _pint_unit_type_cache, _pint_quantity_factory_cache


@cache
def _pint_unit(unit: str) -> Any:
    return _pint_registry().Unit(unit)


class Physical:
    """Represent a numeric value with checked physical dimensions.

    ``nmag.SI`` is an alias for this class and is the preferred spelling in
    simulation scripts. Common units use a lightweight internal representation;
    other units are delegated to Pint without changing the public behavior.

    Args:
        value: Numeric magnitude, another compatible quantity, a unit string
            such as ``"A/m"``, or a complete quantity string.
        dimensions: Unit string, legacy ``[unit, power, ...]`` list, or ``None``.

    Raises:
        TypeError: If the value or dimensions form is unsupported.
        ValueError: If a legacy dimensions list is malformed.
    """

    _quantity: Any

    def __init__(self, value: Any, dimensions: Any | None = None) -> None:
        """Create a dimensionally checked quantity."""
        fast_quantity = self._fast_quantity(value, dimensions)
        if fast_quantity is not None:
            self._quantity = fast_quantity
            return
        self._quantity = self._pint_quantity(value, dimensions)

    @staticmethod
    def _fast_quantity(value: Any, dimensions: Any | None) -> _FastQuantity | None:
        if isinstance(value, _FastQuantity):
            return value
        if dimensions is None:
            return Physical._fast_quantity_without_dimensions(value)
        return Physical._fast_quantity_with_dimensions(value, dimensions)

    @staticmethod
    def _fast_quantity_without_dimensions(value: Any) -> _FastQuantity | None:
        if isinstance(value, str):
            unit = _fast_unit(value)
            return None if unit is None else _FastQuantity(1.0, unit)
        if isinstance(value, (int, float)):
            return _FastQuantity(float(value), _FAST_UNITS[""])
        return None

    @staticmethod
    def _fast_quantity_with_dimensions(value: Any, dimensions: Any) -> _FastQuantity | None:
        if isinstance(dimensions, str):
            unit = _fast_unit(dimensions)
            return None if unit is None else _FastQuantity(value, unit)
        if isinstance(dimensions, list):
            unit = _fast_dimensions_list(cast(list[Any], dimensions))
            return None if unit is None else _FastQuantity(value, unit)
        return None

    @staticmethod
    def _pint_quantity(value: Any, dimensions: Any | None) -> Any:
        quantity_type, unit_type, quantity_factory = _pint_handles()
        if isinstance(value, quantity_type):
            return value
        if isinstance(value, unit_type):
            return quantity_factory(1.0, value)
        if dimensions is None and isinstance(value, str):
            return Physical._pint_string_quantity(value, quantity_factory)
        return Physical._pint_value_with_dimensions(value, dimensions, unit_type, quantity_factory)

    @staticmethod
    def _pint_string_quantity(value: str, quantity_factory: Any) -> Any:
        try:
            return quantity_factory(1.0, _pint_unit(value))
        except Exception:
            return quantity_factory(value)

    @staticmethod
    def _pint_value_with_dimensions(
        value: Any,
        dimensions: Any | None,
        unit_type: Any,
        quantity_factory: Any,
    ) -> Any:
        magnitude = float(value)
        if dimensions is None:
            return quantity_factory(magnitude)
        if isinstance(dimensions, str):
            return quantity_factory(magnitude, _pint_unit(dimensions))
        if isinstance(dimensions, unit_type):
            return quantity_factory(magnitude, dimensions)
        if isinstance(dimensions, list):
            return Physical._pint_list_dimensions(
                magnitude, cast(list[Any], dimensions), quantity_factory
            )
        raise TypeError(
            "Unsupported type for 'dimensions'. "
            f"Must be str, list, or None, but got {type(dimensions)}."
        )

    @staticmethod
    def _pint_list_dimensions(
        magnitude: float,
        dimensions: list[Any],
        quantity_factory: Any,
    ) -> Any:
        if not dimensions:
            return quantity_factory(magnitude)
        if len(dimensions) % 2 != 0:
            raise ValueError(
                "Physical quantity: Bad dimensions list given! "
                "Must contain pairs of unit names and powers. "
                f"Received: {dimensions}"
            )
        unit_string = " * ".join(
            f"{dimensions[index]}**{dimensions[index + 1]}"
            for index in range(0, len(dimensions), 2)
        )
        return quantity_factory(magnitude, unit_string)

    @property
    def magnitude(self) -> Any:
        """Return the numeric magnitude in the quantity's stored unit."""
        return self._quantity.magnitude

    value = magnitude  # Alias for backward compatibility

    def dens_str(self) -> str:
        """Return a compact legacy-compatible representation such as ``<A/m>``."""
        # Handle the special case for a dimensionless quantity of 1.0
        if self._quantity.dimensionless and self._quantity.magnitude == 1.0:
            return "<1>"

        # For non-dimensionless quantities with a magnitude of 1.0, omit the number.
        # pint.Unit will not have a magnitude, even though it can be thought of as it having a magnitude of 1.0.
        if hasattr(self._quantity, "magnitude") and self._quantity.magnitude == 1.0:
            # Format just the units
            compact_str = f"{self._quantity.units:~}"
        else:
            # Format the whole quantity
            compact_str = f"{self._quantity:~}"

        # Clean up the formatting for backward compatibility
        cleaned_str = compact_str.replace(" ", "").replace("**", "^")
        return f"<{cleaned_str}>"

    def in_units_of(self, unit_quantity: object) -> float:
        """Return the magnitude expressed in multiples of another quantity.

        Args:
            unit_quantity: A :class:`Physical` representing one requested unit,
                for example ``nmag.SI(1, "A/m")``.

        Returns:
            Numeric magnitude in the requested unit.

        Raises:
            TypeError: If ``unit_quantity`` is not a :class:`Physical`.
            DimensionalityError: If the dimensions are incompatible.
        """
        if not isinstance(unit_quantity, Physical):
            raise TypeError("Argument must be an instance of Physical.")

        # The original class implicitly worked by dividing the magnitudes of
        # quantities that were already converted to base SI units. This
        # implementation faithfully reproduces that logic.
        if not _quantities_compatible(self._quantity, unit_quantity._quantity):
            raise _pint_module().errors.DimensionalityError(
                self._quantity.units, unit_quantity._quantity.units
            )

        self_base_mag = _base_magnitude(self._quantity)
        unit_base_mag = _base_magnitude(unit_quantity._quantity)

        if unit_base_mag == 0:
            raise ZeroDivisionError("Cannot express in units of a zero quantity.")

        return self_base_mag / unit_base_mag

    # Helper method to safely get the pint.Quantity from another object
    def _unwrap(self, other: Any) -> Any:
        if isinstance(other, Physical):
            other_quantity = other._quantity
            if not isinstance(self._quantity, _FastQuantity) and isinstance(
                other_quantity, _FastQuantity
            ):
                return other_quantity._as_pint()
            return other_quantity
        return other

    # --- Magic Operators ---

    def __str__(self) -> str:
        """'Nice' presentation of the pint object."""
        # We can customize this, but pint's default is quite good.
        # Example: "10.0 meter / second"
        return str(self._quantity)

    def __repr__(self) -> str:
        """
        Returns a representation of the object that can be used to
        re-instantiate it using eval.
        We probably don't have to do this for dens_str since it is just
        used for printing into large files. If we do have to do that
        eventually, we might need to change the logic for that function.
        """
        # This creates a string like 'Physical(10.0, "meter / second")'
        return f"Physical({self._quantity.magnitude}, '{self._quantity.units}')"

    def __float__(self) -> float:
        """
        Return the value of the Physical Object if it is dimensionless.
        """
        if not self._quantity.dimensionless:
            raise _pint_module().errors.DimensionalityError(
                self._quantity.units,
                "dimensionless",
                extra_msg="\nObjects can be converted to float only when they are dimensionless.",
            )
        return _base_magnitude(self._quantity)

    # --- Unary Operators ---

    def __abs__(self) -> Physical:
        return Physical(abs(self._quantity))

    def __neg__(self) -> Physical:
        return Physical(-1 * self._quantity)

    def __pos__(self) -> Physical:
        return Physical(self._quantity)

    # --- Comparison Operators ---

    def __lt__(self, other: Any) -> bool:
        return self._quantity < self._unwrap(other)

    def __le__(self, other: Any) -> bool:
        return self._quantity <= self._unwrap(other)

    def __gt__(self, other: Any) -> bool:
        return self._quantity > self._unwrap(other)

    def __ge__(self, other: Any) -> bool:
        return self._quantity >= self._unwrap(other)

    def __eq__(self, other: object) -> bool:
        try:
            other_quantity = self._unwrap(other)
            if isinstance(self._quantity, _FastQuantity) or isinstance(
                other_quantity,
                _FastQuantity,
            ):
                if isinstance(other_quantity, (int, float)):
                    return self._quantity == other_quantity
                return _quantities_compatible(self._quantity, other_quantity) and math.isclose(
                    _base_magnitude(self._quantity),
                    _base_magnitude(other_quantity),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            return self._quantity == other_quantity
        except _pint_module().errors.DimensionalityError:
            # If units are not compatible, they cannot be equal.
            return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    # --- Binary Arithmetic Operators ---

    def __add__(self, other: Any) -> Physical:
        return Physical(self._quantity + self._unwrap(other))

    def __radd__(self, other: Any) -> Physical:
        return Physical(self._unwrap(other) + self._quantity)

    def __sub__(self, other: Any) -> Physical:
        return Physical(self._quantity - self._unwrap(other))

    def __rsub__(self, other: Any) -> Physical:
        return Physical(self._unwrap(other) - self._quantity)

    def __mul__(self, other: Any) -> Physical:
        return Physical(self._quantity * self._unwrap(other))

    def __rmul__(self, other: Any) -> Physical:
        return Physical(self._unwrap(other) * self._quantity)

    def __truediv__(self, other: Any) -> Physical:
        return Physical(self._quantity / self._unwrap(other))

    def __rtruediv__(self, other: Any) -> Physical:
        return Physical(self._unwrap(other) / self._quantity)

    def __pow__(self, exponent: int | float) -> Physical:
        return Physical(self._quantity**exponent)


SI = Physical
