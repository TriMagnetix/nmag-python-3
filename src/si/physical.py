"""
Original file: https://github.com/nmag-project/nmag-src/blob/master/interface/nsim/si_units/lib.py
Support for Physical quantities carrying SI units.

This module provides a backward-compatible wrapper around the 'pint' library,
replicating the interface of a legacy Physical/SI class.
"""

import pint

ureg = pint.get_application_registry()

class Physical:
  """
  A wrapper class around the pint library to provide backward compatibility
  with a legacy Physical quantity class.

  It supports initialization with a value and a unit string, or a value
  and a list-based dimension format (e.g., ['m', 1, 's', -1]).
  All arithmetic and comparison operations are handled by the underlying
  pint.Quantity object.
  """
    
  def __init__(self, value, dimensions=None):
    """
    Creates a Physical object.

    :param value: The magnitude of the quantity. Can also be a pint.Quantity
                  object, or a string like "10 m/s".
    :param dimensions: The units of the quantity. Can be a string
                       (e.g., "m/s", "J/mol"), a list of units and
                       their powers (e.g., ['m', 1, 's', -1]), or None.
                       If None and value is a string, value will be parsed
                       as the quantity.
    """
    # Case 1: The value is already a pint Quantity/Unit object
    # A pint unit is a quntity with a unit like meter, second, etc. attached. 
    # This allows us to pass a pint unit or quantity directly.
    if isinstance(value, ureg.Quantity) or isinstance(value, ureg.Unit):
      self._quantity = value
      return

    # Case 2: Shorthand initialization like Physical("m/s")
    # In the old class, this was equivalent to Physical(1.0, "m/s")
    if dimensions is None and isinstance(value, str):
      self._quantity = ureg.Quantity(value)
      return
      
    # Ensure value is a numeric type for pint
    val = float(value)

    # Case 3: Standard initialization with string dimensions or pint unit
    # e.g., Physical(10, "m/s") or Physical(10, ureg.meter)
    if dimensions is None or (isinstance(dimensions, str) or isinstance(dimensions, ureg.Unit)):
      unit_str = dimensions if dimensions is not None else ""
      self._quantity = ureg.Quantity(val, unit_str)

    # Case 4: Handle the legacy list format, e.g., ['m', 1, 'kg', 1, 's', -2]
    elif isinstance(dimensions, list):
      if not dimensions:  # Empty list means dimensionless
        self._quantity = ureg.Quantity(val)
      else:
        if len(dimensions) % 2 != 0:
          raise ValueError(
            "Physical quantity: Bad dimensions list given! "
            "Must contain pairs of unit names and powers. "
            f"Received: {dimensions}"
          )
        # Convert the list to a pint-compatible string "m**1 * kg**1 * s**-2"
        unit_str_parts = []
        for i in range(0, len(dimensions), 2):
          unit = dimensions[i]
          power = dimensions[i+1]
          unit_str_parts.append(f"{unit}**{power}")
        
        unit_string = " * ".join(unit_str_parts)
        self._quantity = ureg.Quantity(val, unit_string)
    else:
      raise TypeError(
        "Unsupported type for 'dimensions'. "
        f"Must be str, list, or None, but got {type(dimensions)}."
      )

  @property
  def magnitude(self):
    """Returns the numerical magnitude of the quantity."""
    return self._quantity.magnitude
  
  value = magnitude  # Alias for backward compatibility

  def dens_str(self):
    """
    Provide a dense string describing the object, similar to the original class.
    Example: <1e+06A/m>
    """
    # Handle the special case for a dimensionless quantity of 1.0
    if self._quantity.dimensionless and self._quantity.magnitude == 1.0:
      return "<1>"
    
    # For non-dimensionless quantities with a magnitude of 1.0, omit the number.
    # pint.Unit will not have a magnitude, even though it can be thought of as it having a magnitude of 1.0.
    if hasattr(self._quantity, 'magnitude') and self._quantity.magnitude == 1.0:
      # Format just the units
      compact_str = f"{self._quantity.units:~}"
    else:
      # Format the whole quantity
      compact_str = f"{self._quantity:~}"

    # Clean up the formatting for backward compatibility
    cleaned_str = compact_str.replace(" ", "").replace("**", "^")
    return f"<{cleaned_str}>"

  def in_units_of(self, unit_quantity):
    """
    Expresses the object in multiples of 'unit_quantity' and returns the
    resulting magnitude as a float.

    This is equivalent to performing a division and taking the magnitude
    of the dimensionless result.

    :param unit_quantity: Another Physical object whose units to convert to.
    :return: A float representing the magnitude in the new units.
    """
    if not isinstance(unit_quantity, Physical):
      raise TypeError(
        "Argument must be an instance of Physical."
      )

    # The original class implicitly worked by dividing the magnitudes of
    # quantities that were already converted to base SI units. This
    # implementation faithfully reproduces that logic.
    if not self._quantity.is_compatible_with(unit_quantity._quantity):
      raise pint.errors.DimensionalityError(
          self._quantity.units, unit_quantity._quantity.units
      )

    self_base_mag = self._quantity.to_base_units().magnitude
    unit_base_mag = unit_quantity._quantity.to_base_units().magnitude

    if unit_base_mag == 0:
        raise ZeroDivisionError("Cannot express in units of a zero quantity.")

    return self_base_mag / unit_base_mag

  # Helper method to safely get the pint.Quantity from another object
  def _unwrap(self, other):
    if isinstance(other, Physical):
      return other._quantity
    return other

  # --- Magic Operators ---

  def __str__(self):
    """'Nice' presentation of the pint object."""
    # We can customize this, but pint's default is quite good.
    # Example: "10.0 meter / second"
    return str(self._quantity)

  def __repr__(self):
    """
    Returns a representation of the object that can be used to
    re-instantiate it using eval.
    We probably don't have to do this for dens_str since it is just 
    used for printing into large files. If we do have to do that 
    eventually, we might need to change the logic for that function.
    """
    # This creates a string like 'Physical(10.0, "meter / second")'
    return f"Physical({self._quantity.magnitude}, '{self._quantity.units}')"

  def __float__(self):
    """
    Return the value of the Physical Object if it is dimensionless.
    """
    if not self._quantity.dimensionless:
      raise pint.errors.DimensionalityError(
        self._quantity.units, 'dimensionless',
        extra_msg="\nObjects can be converted to float only when they are dimensionless."
      )
    return self._quantity.magnitude

  # --- Unary Operators ---
  
  def __abs__(self):
    return Physical(abs(self._quantity))

  def __neg__(self):
    return Physical(-1 * self._quantity)

  def __pos__(self):
    return Physical(self._quantity)

  # --- Comparison Operators ---

  def __lt__(self, other):
    return self._quantity < self._unwrap(other)

  def __le__(self, other):
    return self._quantity <= self._unwrap(other)

  def __gt__(self, other):
    return self._quantity > self._unwrap(other)

  def __ge__(self, other):
    return self._quantity >= self._unwrap(other)

  def __eq__(self, other):
    try:
      return self._quantity == self._unwrap(other)
    except pint.errors.DimensionalityError:
      # If units are not compatible, they cannot be equal.
      return False

  def __ne__(self, other):
    try:
      return self._quantity != self._unwrap(other)
    except pint.errors.DimensionalityError:
      # If units are not compatible, they are not equal.
      return True

  # --- Binary Arithmetic Operators ---

  def __add__(self, other):
    return Physical(self._quantity + self._unwrap(other))

  def __radd__(self, other):
    return Physical(self._unwrap(other) + self._quantity)

  def __sub__(self, other):
    return Physical(self._quantity - self._unwrap(other))

  def __rsub__(self, other):
    return Physical(self._unwrap(other) - self._quantity)

  def __mul__(self, other):
    return Physical(self._quantity * self._unwrap(other))

  def __rmul__(self, other):
    return Physical(self._unwrap(other) * self._quantity)

  def __truediv__(self, other):
    return Physical(self._quantity / self._unwrap(other))

  def __rtruediv__(self, other):
    return Physical(self._unwrap(other) / self._quantity)

  def __pow__(self, exponent):
    return Physical(self._quantity ** exponent)

SI = Physical
