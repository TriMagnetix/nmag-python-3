"""
Original file: https://github.com/nmag-project/nmag-src/blob/master/interface/nsim/si_units/si.py
Some useful SI constants, defined using the pint library's authoritative values.

This module provides common physical constants as pint Quantity objects
for direct use in calculations. It replaces hard-coded values with references
to the pint library's built-in constants, which are based on up-to-date
CODATA values.
"""
from physical import ureg, SI

# --- Base and Common Units ---
kilogram = ureg.kilogram
meter = ureg.meter
metre = meter  # alternative spelling
Ampere = ureg.ampere
Kelvin = ureg.kelvin
second = ureg.second
candela = ureg.candela
mol = ureg.mole
Joule = ureg.joule
Newton = ureg.newton
Tesla = ureg.tesla
Gauss = ureg.gauss

# --- Physical Constants (from pint's library) ---

mu0 = ureg.mu_0
Oersted = ureg.oersted
Oe = Oersted
bohr_magneton = ureg.bohr_magneton
positron_charge = ureg.e
electron_charge = -ureg.e
boltzmann_constant = ureg.k
plank_constant = ureg.h
reduced_plank_constant = ureg.hbar

# --- Custom or Specific Values ---

degrees_per_ns = SI(1 * ureg.degree / ureg.nanosecond)

# This constant, often used in micromagnetics, is the product of the
# electron gyromagnetic ratio (gamma_e) and the vacuum permeability (mu_0).
# The negative sign is a convention used in the LLG equation.
gamma0 = -SI(ureg.electron_gyromagnetic_ratio * ureg.mu_0)
