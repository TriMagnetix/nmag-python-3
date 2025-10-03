"""
Definition of the MagMaterial class, which allows to define a magnetic
material and all its parameters (saturation magnetisation, exchange coupling,
etc.)
"""

import logging

from si.physical import SI
from si import constants
from anisotropy.anisotropy import PredefinedAnisotropy

log = logging.getLogger('nmag')

class MagMaterial(object):
    """
    Represents a magnetic material, defining its physical properties such as
    saturation magnetisation, exchange coupling, and LLG parameters.

    All physical quantities should be provided as SI objects to ensure
    dimensional correctness.
    """

    def __init__(self,
                 name,
                 Ms=SI(0.86e6 * constants.Ampere / constants.meter),
                 llg_damping=0.5,
                 llg_gamma_G=SI(2.210173e5 * constants.meter / (constants.Ampere * constants.second)),
                 llg_normalisationfactor=SI(0.1e12 * 1 / constants.second),
                 llg_xi=0.0,
                 llg_polarisation=0.0,
                 do_precession=True,
                 exchange_coupling=SI(1.3e-11 * constants.Joule / constants.meter),
                 anisotropy=None,
                 anisotropy_order=None,
                 properties=["magnetic", "material"],
                 scale_volume_charges=1.0
                 ):
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
        self.name = name
        self.Ms = Ms
        self.llg_gamma_G = llg_gamma_G
        self.llg_damping = llg_damping
        self.llg_normalisationfactor = llg_normalisationfactor
        self.llg_xi = llg_xi
        self.llg_polarisation = llg_polarisation
        self.do_precession = do_precession
        self.properties = properties
        self.exchange_coupling = exchange_coupling
        self.scale_volume_charges = scale_volume_charges

        # --- Unit validation ---
        one = SI(1)
        expected_units = (
            ("Ms", SI(constants.Ampere / constants.meter)),
            ("llg_gamma_G", SI(constants.meter / (constants.Ampere * constants.second))),
            ("llg_damping", one),
            ("llg_normalisationfactor", SI(1 / constants.second)),
            ("llg_xi", one),
            ("llg_polarisation", one),
            ("exchange_coupling", SI(constants.Joule / constants.meter))
        )

        for attr_name, expected_unit in expected_units:
            value = getattr(self, attr_name)
            
            # The value could be an SI object or a raw number (e.g., float)
            # Check compatibility between its quantities and raw numbers
            if isinstance(value, SI):
                value_to_check = value._quantity
            else:
                value_to_check = value # It's a raw number/

            if not expected_unit._quantity.is_compatible_with(value_to_check):
                raise TypeError(
                    f"The argument '{attr_name}' for material '{self.name}' "
                    f"requires units compatible with {expected_unit.dens_str()}, "
                    f"but received a value of '{value}'."
                )
            
        # Check for physically valid exchange coupling
        if self.exchange_coupling < 0.0:
            raise ValueError(
                f"The exchange coupling constant must be positive. For "
                f"material '{self.name}', you specified: {self.exchange_coupling}."
            )

        # TODO: Revisit anisotropy handling, do we really need both of these statements?
        # --- Anisotropy Handling ---
        if isinstance(anisotropy, PredefinedAnisotropy):
            if anisotropy_order:
                raise ValueError(
                    "Cannot specify custom 'anisotropy_order' when using "
                    "a predefined anisotropy."
                )
            # In a real implementation, you might extract properties here.
            # For now, we just store the object.
            self.anisotropy = anisotropy
            self.anisotropy_order = anisotropy.order
        else:
            if anisotropy and not anisotropy_order:
                raise ValueError(
                    "You must specify 'anisotropy_order' when using a "
                    "custom anisotropy function."
                )
            self.anisotropy = anisotropy
            self.anisotropy_order = anisotropy_order

        # SU units units for backwards compatibility, they used to be stripped 
        # of units, but now we can do calculations with the units attached
        self.su_Ms = self.Ms
        self.su_llg_gamma_G = self.llg_gamma_G
        self.su_llg_damping = self.llg_damping
        self.su_llg_normalisationfactor = self.llg_normalisationfactor
        self.su_exchange_coupling = self.exchange_coupling 

        self.thermal_factor = (2.0 * constants.boltzmann_constant * self.llg_damping) / \
                              (-constants.gamma0 * constants.mu0 * self.Ms)  
        self.su_thermal_factor = self.thermal_factor

        gilbert_to_ll = 1.0 / (1.0 + self.su_llg_damping ** 2)
        self.su_llg_coeff1 = -self.su_llg_gamma_G * gilbert_to_ll
        self.su_llg_coeff2 = self.su_llg_coeff1 * self.su_llg_damping
        su_f = -gilbert_to_ll * (llg_polarisation * constants.bohr_magneton / (constants.positron_charge * self.Ms * (1 + llg_xi ** 2)))
        if su_f == 0.0:
            self.su_llg_stt_prefactor = 0.0
        else:
            self.su_llg_stt_prefactor = 1.0

        self.su_llg_stt_nadiab = su_f * (self.llg_xi - self.su_llg_damping)
        self.su_llg_stt_adiab = su_f * (1.0 + self.llg_damping * self.llg_xi)

        if self.do_precession == False:
            log.info ("Setting su_llg_coeff1 to zero; thus no precession for material '%s'" % self.name)
            self.su_llg_coeff1 = 0.0

        self.su_exch_prefactor = -2.0 * self.su_exchange_coupling / (constants.mu0 * self.su_Ms)

        self.su_anisotropy = self.anisotropy
            
        self.extended_print = False
        log.info(f"Created new Material:\n {self}")

    def __str__(self):
        repr_str = f"Material '{self.name}'\n"
        
        attrs = list(filter(lambda a: a[0] != '_', dir(self)))
        
        if not self.extended_print:
            attrs = ['name', 'Ms', 'exchange_coupling', 'anisotropy',
                     'anisotropy_order', 'llg_gamma_G', 'llg_damping',
                     'llg_normalisationfactor', 'do_precession',
                     'llg_polarisation', 'llg_xi', 'thermal_factor',
                     'extended_print']

        for attr in attrs:
            if hasattr(self, attr):
                value_str = str(getattr(self, attr))
                repr_str += f" {attr:>25} = {value_str}\n"

        return repr_str