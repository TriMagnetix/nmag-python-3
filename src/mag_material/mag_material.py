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
    """
    Represents a magnetic material, defining its physical properties such as
    saturation magnetisation, exchange coupling, and LLG parameters.

    All physical quantities should be provided as SI objects to ensure
    dimensional correctness.
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
        """
        Initializes a magnetic material with its physical properties.

        :Parameters:
          `name` : string
            The name of the material (e.g., 'Py', 'Fe').

          `Ms` : SI Object
            Saturation magnetisation in Amperes per meter.

          `llg_damping` : float or SI Object
            The dimensionless Gilbert damping parameter (alpha).

          `llg_gamma_G` : SI Object
            The gyromagnetic ratio for the LLG equation.

          `exchange_coupling` : SI Object
            The exchange coupling constant 'A' in Joules per meter.

          `anisotropy` : PredefinedAnisotropy or callable
            The anisotropy model.

          `anisotropy_order` : int
            The order of a custom polynomial anisotropy function.

          `do_precession` : bool
            If False, switches off the precessional term in the LLG equation.

          `llg_normalisationfactor` : SI Object
            A coefficient for an extra term added to the LLG right-hand side to correct
            numerical errors in the magnetization norm.

          `llg_xi` : float or SI Object
            For spin-transfer-torque, this is the ratio between the exchange and the
            spin-flip relaxation times (xi = tau_ex / tau_sf).

          `llg_polarisation` : float or SI Object
            For spin-transfer-torque, this is the polarisation of the spin-current.

          `properties` : list of strings
            A list of properties associated with the material, used internally by the
            simulation to set up operators.

          `scale_volume_charges` : float
            A debugging parameter for developers.
        """
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

        self.extended_print = False
        if log.isEnabledFor(logging.INFO):
            log.info("Created new Material:\n %s", self)

    def _derived_su_values(self) -> dict[str, MaterialScalar]:
        if self._derived_su_constants is None:
            constants = _si_constants()
            thermal_factor = (2.0 * constants.boltzmann_constant * self.llg_damping) / (
                -constants.gamma0 * constants.mu0 * self.Ms
            )
            gilbert_to_ll = 1.0 / (1.0 + self.su_llg_damping**2)
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
            ms = self.Ms.in_units_of(SI("A/m"))
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
            self._derived_su_constants = {
                "thermal_factor": thermal_factor,
                "su_thermal_factor": thermal_factor,
                "su_llg_stt_prefactor": 0.0 if su_f == 0.0 else 1.0,
                "su_llg_stt_nadiab": su_f * (xi - self.su_llg_damping),
                "su_llg_stt_adiab": su_f * (1.0 + self.su_llg_damping * xi),
                "su_exch_prefactor": (
                    2.0 * self.su_exchange_coupling / (constants.mu0 * self.su_Ms)
                ),
            }
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
