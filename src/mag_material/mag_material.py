"""
Definition of the MagMaterial class, which allows to define a magnetic
material and all its parameters (saturation magnetisation, exchange coupling,
etc.)
"""

import logging
from typing import Any, Protocol, cast

from anisotropy import PredefinedAnisotropy
from si.physical import SI

from .parameters import (
    AnisotropyFunction,
    MaterialScalar,
    resolve_material_parameters,
)

log = logging.getLogger("nmag")


class _SIConstants(Protocol):
    boltzmann_constant: SI
    gamma0: SI
    mu0: SI
    bohr_magneton: SI
    positron_charge: SI


_constants_cache: _SIConstants | None = None


def _si_constants() -> _SIConstants:
    global _constants_cache
    if _constants_cache is None:
        from si import constants

        _constants_cache = cast(_SIConstants, constants)
    return _constants_cache


class MagMaterial:
    """Define the physical parameters assigned to a magnetic mesh region.

    Args:
        name: Unique material name used in field and output labels.
        Ms: Saturation magnetization, compatible with A/m.
        llg_damping: Dimensionless Gilbert damping coefficient.
        llg_gamma_G: Gyromagnetic ratio used by the LLG equation.
        llg_normalisationfactor: Coefficient for numerical norm correction.
        llg_xi: Dimensionless nonadiabatic Zhang-Li coefficient.
        llg_polarisation: Dimensionless spin-current polarization.
        do_precession: Include the precessional LLG term when true.
        exchange_coupling: Exchange constant, compatible with J/m.
        anisotropy: Predefined anisotropy or custom energy callable.
        anisotropy_order: Polynomial order required for a custom callable.
        properties: Compatibility property labels. Most users should omit this.
        scale_volume_charges: Multiplier for interior demagnetization volume
            charge. Boundary surface charges remain physical and unscaled.

    Raises:
        TypeError: If a parameter has an unsupported type or dimensions.
        ValueError: If a value is non-finite, physically invalid, or a custom
            anisotropy omits its order.
    """

    def __init__(
        self,
        name: str,
        Ms: SI | None = None,
        llg_damping: MaterialScalar = 0.5,
        llg_gamma_G: SI | None = None,
        llg_normalisationfactor: SI | None = None,
        llg_xi: MaterialScalar = 0.0,
        llg_polarisation: MaterialScalar = 0.0,
        do_precession: bool = True,
        exchange_coupling: SI | None = None,
        anisotropy: PredefinedAnisotropy | AnisotropyFunction | None = None,
        anisotropy_order: int | None = None,
        properties: list[str] | None = None,
        scale_volume_charges: float = 1.0,
    ) -> None:
        """Initialize a material after validating units and physical values."""
        parameters = resolve_material_parameters(
            name,
            ms=Ms,
            damping=llg_damping,
            gamma=llg_gamma_G,
            normalisation=llg_normalisationfactor,
            xi=llg_xi,
            polarisation=llg_polarisation,
            do_precession=do_precession,
            exchange=exchange_coupling,
            anisotropy=anisotropy,
            anisotropy_order=anisotropy_order,
            properties=properties,
            scale_volume_charges=scale_volume_charges,
        )

        self.name = name
        self.Ms = parameters.ms
        self.llg_gamma_G = parameters.gamma
        self.llg_damping = parameters.damping
        self.llg_normalisationfactor = parameters.normalisation
        self.llg_xi = parameters.xi
        self.llg_polarisation = parameters.polarisation
        self.do_precession = parameters.do_precession
        self.properties = parameters.properties
        self.exchange_coupling = parameters.exchange
        self.scale_volume_charges = parameters.scale_volume_charges
        self.anisotropy = parameters.anisotropy
        self.anisotropy_order = parameters.anisotropy_order

        # SU units units for backwards compatibility, they used to be stripped
        # of units, but now we can do calculations with the units attached
        self.su_Ms = self.Ms
        self.su_llg_gamma_G = self.llg_gamma_G
        self.su_llg_damping = self.llg_damping
        self.su_llg_normalisationfactor = self.llg_normalisationfactor
        self.su_exchange_coupling = self.exchange_coupling

        gilbert_to_ll = 1.0 / (1.0 + self.su_llg_damping**2)
        self.su_llg_coeff1 = -self.su_llg_gamma_G * gilbert_to_ll
        self.su_llg_coeff2 = self.su_llg_coeff1 * self.su_llg_damping

        if not self.do_precession:
            log.info(
                "Setting su_llg_coeff1 to zero; thus no precession for material '%s'", self.name
            )
            self.su_llg_coeff1 = 0.0

        self.su_anisotropy = self.anisotropy
        self._derived_su_constants: dict[str, MaterialScalar] | None = None
        self._derived_su_token: tuple[int, ...] | None = None

        self.extended_print = False
        self._initialized = True
        if log.isEnabledFor(logging.INFO):
            log.info("Created new Material:\n %s", self)

    def _sync_llg_coeffs(self) -> None:
        gilbert_to_ll = 1.0 / (1.0 + self.su_llg_damping**2)
        base_coeff1 = -self.su_llg_gamma_G * gilbert_to_ll
        object.__setattr__(self, "su_llg_coeff2", base_coeff1 * self.su_llg_damping)
        object.__setattr__(
            self,
            "su_llg_coeff1",
            0.0 if not self.do_precession else base_coeff1,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if not getattr(self, "_initialized", False):
            return
        if name == "Ms":
            object.__setattr__(self, "su_Ms", value)
        elif name == "llg_gamma_G":
            object.__setattr__(self, "su_llg_gamma_G", value)
            self._sync_llg_coeffs()
        elif name == "llg_damping":
            object.__setattr__(self, "su_llg_damping", value)
            self._sync_llg_coeffs()
        elif name in {"su_llg_gamma_G", "su_llg_damping", "do_precession"}:
            self._sync_llg_coeffs()
        elif name == "llg_normalisationfactor":
            object.__setattr__(self, "su_llg_normalisationfactor", value)
        elif name == "exchange_coupling":
            object.__setattr__(self, "su_exchange_coupling", value)
        elif name == "anisotropy":
            object.__setattr__(self, "su_anisotropy", value)
        if not name.startswith("_"):
            object.__setattr__(self, "_derived_su_constants", None)
            object.__setattr__(self, "_derived_su_token", None)

    def _derived_su_values(self) -> dict[str, MaterialScalar]:
        token = (
            id(self.Ms),
            id(self.llg_damping),
            id(self.su_llg_damping),
            id(self.llg_gamma_G),
            id(self.su_llg_gamma_G),
            id(self.llg_normalisationfactor),
            id(self.su_llg_normalisationfactor),
            id(self.llg_polarisation),
            id(self.llg_xi),
            id(self.exchange_coupling),
            id(self.su_exchange_coupling),
            id(self.su_Ms),
            bool(self.do_precession),
            id(self.anisotropy),
            id(self.su_anisotropy),
        )
        if self._derived_su_constants is None or self._derived_su_token != token:
            constants = _si_constants()
            ms = self.Ms.in_units_of(SI("A/m"))
            su_ms = self.su_Ms.in_units_of(SI("A/m"))
            thermal_factor = (
                (
                    (2.0 * constants.boltzmann_constant * self.llg_damping)
                    / (-constants.gamma0 * constants.mu0 * SI(1.0, "A/m"))
                )
                * 0.0
                if ms == 0.0
                else (2.0 * constants.boltzmann_constant * self.llg_damping)
                / (-constants.gamma0 * constants.mu0 * self.Ms)
            )
            damping = (
                self.su_llg_damping.in_units_of(SI(1))
                if isinstance(self.su_llg_damping, SI)
                else float(self.su_llg_damping)
            )
            gilbert_to_ll = 1.0 / (1.0 + damping * damping)
            polarisation = (
                self.llg_polarisation.in_units_of(SI(1))
                if isinstance(self.llg_polarisation, SI)
                else float(self.llg_polarisation)
            )
            xi = self.llg_xi.in_units_of(SI(1)) if isinstance(self.llg_xi, SI) else float(self.llg_xi)
            bohr_magneton = float(
                cast(Any, 1.0 * constants.bohr_magneton).to("J/T").magnitude
            )
            charge = float(cast(Any, 1.0 * constants.positron_charge).to("C").magnitude)
            su_f = (
                SI(0.0, "m^3/A/s")
                if ms == 0.0
                else SI(
                    -gilbert_to_ll
                    * polarisation
                    * bohr_magneton
                    / (charge * ms * (1.0 + xi * xi)),
                    "m^3/A/s",
                )
            )
            su_exch_prefactor = (
                (2.0 * self.su_exchange_coupling / (constants.mu0 * SI(1.0, "A/m"))) * 0.0
                if su_ms == 0.0
                else 2.0 * self.su_exchange_coupling / (constants.mu0 * self.su_Ms)
            )
            self._derived_su_constants = {
                "thermal_factor": thermal_factor,
                "su_thermal_factor": thermal_factor,
                "su_llg_stt_prefactor": 0.0 if su_f == 0.0 else 1.0,
                "su_llg_stt_nadiab": su_f * (xi - damping),
                "su_llg_stt_adiab": su_f * (1.0 + damping * xi),
                "su_exch_prefactor": su_exch_prefactor,
            }
            self._derived_su_token = token
        return self._derived_su_constants

    @property
    def thermal_factor(self) -> MaterialScalar:
        return self._derived_su_values()["thermal_factor"]

    @property
    def su_thermal_factor(self) -> MaterialScalar:
        return self._derived_su_values()["su_thermal_factor"]

    @property
    def su_llg_stt_prefactor(self) -> MaterialScalar:
        return self._derived_su_values()["su_llg_stt_prefactor"]

    @property
    def su_llg_stt_nadiab(self) -> MaterialScalar:
        return self._derived_su_values()["su_llg_stt_nadiab"]

    @property
    def su_llg_stt_adiab(self) -> MaterialScalar:
        return self._derived_su_values()["su_llg_stt_adiab"]

    @property
    def su_exch_prefactor(self) -> MaterialScalar:
        return self._derived_su_values()["su_exch_prefactor"]

    def __str__(self) -> str:
        repr_str = f"Material '{self.name}'\n"

        attrs = list(filter(lambda a: a[0] != "_", dir(self)))

        if not self.extended_print:
            attrs = [
                "name",
                "Ms",
                "exchange_coupling",
                "anisotropy",
                "anisotropy_order",
                "llg_gamma_G",
                "llg_damping",
                "llg_normalisationfactor",
                "do_precession",
                "llg_polarisation",
                "llg_xi",
                "thermal_factor",
                "extended_print",
            ]

        for attr in attrs:
            if hasattr(self, attr):
                value_str = str(getattr(self, attr))
                repr_str += f" {attr:>25} = {value_str}\n"

        return repr_str
