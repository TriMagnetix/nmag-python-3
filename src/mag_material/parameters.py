"""Normalization and validation for :class:`mag_material.MagMaterial` inputs."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from numpy.typing import ArrayLike

from anisotropy import PredefinedAnisotropy
from si.physical import SI

MaterialScalar = SI | float
AnisotropyFunction = Callable[[ArrayLike], float]


@dataclass(slots=True)
class MaterialParameters:
    """Fully resolved constructor values for a magnetic material."""

    ms: SI
    damping: MaterialScalar
    gamma: SI
    normalisation: SI
    xi: MaterialScalar
    polarisation: MaterialScalar
    do_precession: bool
    exchange: SI
    anisotropy: PredefinedAnisotropy | AnisotropyFunction | None
    anisotropy_order: int | None
    properties: list[str]
    scale_volume_charges: float


@lru_cache(maxsize=1)
def _expected_parameter_units() -> tuple[tuple[str, SI], ...]:
    one = SI(1)
    return (
        ("Ms", SI(1, "A/m")),
        ("llg_gamma_G", SI(1, "m/A/s")),
        ("llg_damping", one),
        ("llg_normalisationfactor", SI(1, "1/s")),
        ("llg_xi", one),
        ("llg_polarisation", one),
        ("exchange_coupling", SI(1, "J/m")),
    )


def _value_by_public_name(parameters: MaterialParameters, name: str) -> MaterialScalar:
    values: dict[str, MaterialScalar] = {
        "Ms": parameters.ms,
        "llg_gamma_G": parameters.gamma,
        "llg_damping": parameters.damping,
        "llg_normalisationfactor": parameters.normalisation,
        "llg_xi": parameters.xi,
        "llg_polarisation": parameters.polarisation,
        "exchange_coupling": parameters.exchange,
    }
    return values[name]


def _validate_units(name: str, parameters: MaterialParameters) -> None:
    for attribute, expected_unit in _expected_parameter_units():
        value = _value_by_public_name(parameters, attribute)
        value_to_check = value._quantity if isinstance(value, SI) else value
        if not expected_unit._quantity.is_compatible_with(value_to_check):
            raise TypeError(
                f"The argument '{attribute}' for material '{name}' requires units compatible "
                f"with {expected_unit.dens_str()}, but received a value of '{value}'."
            )


def _resolve_anisotropy(
    anisotropy: PredefinedAnisotropy | AnisotropyFunction | None,
    anisotropy_order: int | None,
) -> tuple[PredefinedAnisotropy | AnisotropyFunction | None, int | None]:
    if isinstance(anisotropy, PredefinedAnisotropy):
        if anisotropy_order:
            raise ValueError(
                "Cannot specify custom 'anisotropy_order' when using a predefined anisotropy."
            )
        return anisotropy, anisotropy.order
    if anisotropy and not anisotropy_order:
        raise ValueError("You must specify 'anisotropy_order' when using a custom anisotropy function.")
    return anisotropy, anisotropy_order


def resolve_material_parameters(
    name: str,
    *,
    ms: SI | None,
    damping: MaterialScalar,
    gamma: SI | None,
    normalisation: SI | None,
    xi: MaterialScalar,
    polarisation: MaterialScalar,
    do_precession: bool,
    exchange: SI | None,
    anisotropy: PredefinedAnisotropy | AnisotropyFunction | None,
    anisotropy_order: int | None,
    properties: list[str] | None,
    scale_volume_charges: float,
) -> MaterialParameters:
    """Apply legacy defaults and reject invalid material constructor values."""
    try:
        resolved_volume_charge_scale = float(scale_volume_charges)
    except (TypeError, ValueError) as error:
        raise TypeError("scale_volume_charges must be a finite real number.") from error
    if not math.isfinite(resolved_volume_charge_scale):
        raise ValueError("scale_volume_charges must be finite.")

    resolved_anisotropy, resolved_order = _resolve_anisotropy(anisotropy, anisotropy_order)
    parameters = MaterialParameters(
        ms=SI(0.86e6, "A/m") if ms is None else ms,
        damping=damping,
        gamma=SI(2.210173e5, "m/A/s") if gamma is None else gamma,
        normalisation=SI(0.1e12, "1/s") if normalisation is None else normalisation,
        xi=xi,
        polarisation=polarisation,
        do_precession=do_precession,
        exchange=SI(1.3e-11, "J/m") if exchange is None else exchange,
        anisotropy=resolved_anisotropy,
        anisotropy_order=resolved_order,
        properties=["magnetic", "material"] if properties is None else properties,
        scale_volume_charges=resolved_volume_charge_scale,
    )
    _validate_units(name, parameters)
    if parameters.exchange < 0.0:
        raise ValueError(
            "The exchange coupling constant must be positive. For "
            f"material '{name}', you specified: {parameters.exchange}."
        )
    return parameters
