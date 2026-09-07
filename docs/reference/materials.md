# Materials and units API

## SI quantities

`nmag.SI` and `nmag.Physical` are aliases for the same dimensional quantity
class. Prefer `nmag.SI` in simulation scripts.

::: si.physical.Physical
    options:
      members:
        - magnitude
        - dens_str
        - in_units_of

## Magnetic materials

::: mag_material.mag_material.MagMaterial
    options:
      members: false

## Anisotropy

::: anisotropy.predefined.uniaxial_anisotropy

::: anisotropy.predefined.cubic_anisotropy

::: anisotropy.model.PredefinedAnisotropy
    options:
      members: false
