# Units, materials, and anisotropy

## SI quantities

`nmag.SI` attaches physical dimensions to a value:

```python
import nmag

length = nmag.SI(5e-9, "m")
field = nmag.SI(8e3, "A/m")
exchange = nmag.SI(13e-12, "J/m")
```

Use `quantity.in_units_of(nmag.SI(1, "unit"))` when an external library needs
an ordinary number. Nmag checks compatible dimensions at API boundaries; do not
strip units from material constants merely to silence an error.

## Magnetic materials

```python
permalloy = nmag.MagMaterial(
    name="Py",
    Ms=nmag.SI(1e6, "A/m"),
    exchange_coupling=nmag.SI(13e-12, "J/m"),
    llg_damping=0.02,
)
```

The most commonly used parameters are:

| Parameter | Meaning | Typical units |
| --- | --- | --- |
| `Ms` | Saturation magnetization | A/m |
| `exchange_coupling` | Exchange constant | J/m |
| `llg_damping` | Gilbert damping | dimensionless |
| `llg_gamma_G` | Gyromagnetic ratio | m/(A s) |
| `llg_polarisation` | Current polarization for Zhang-Li torque | dimensionless |
| `llg_xi` | Nonadiabatic Zhang-Li coefficient | dimensionless |

`scale_volume_charges` is an expert control that scales only the interior
volume-charge source of the demagnetization calculation. The physical default
is `1.0`; surface charges are never scaled by this setting.

## Uniaxial anisotropy

```python
easy_axis = nmag.uniaxial_anisotropy(
    axis=[0.0, 0.0, 1.0],
    K1=nmag.SI(1e5, "J/m^3"),
    K2=nmag.SI(2e4, "J/m^3"),
)

material = nmag.MagMaterial(
    name="uniaxial",
    Ms=nmag.SI(8e5, "A/m"),
    exchange_coupling=nmag.SI(10e-12, "J/m"),
    anisotropy=easy_axis,
)
```

The axis is normalized by the constructor. `K1` and `K2` may be SI energy
densities or plain values interpreted as J/m³.

## Cubic and custom anisotropy

Use `nmag.cubic_anisotropy(axis1, axis2, K1, K2, K3)` for crystalline cubic
terms. The two supplied axes must define an orthogonal orientation.

A custom polynomial energy can be supplied as a callable with an explicit
order:

```python
def energy(m):
    return nmag.SI(1e5 * m[2] ** 2, "J/m^3")

material = nmag.MagMaterial(
    name="custom",
    anisotropy=energy,
    anisotropy_order=2,
)
```

Predefined models use vectorized analytic derivatives. Custom callables use a
validated finite-difference derivative, so they are intended for smaller or
exploratory models. The experimental Diffsol backend does not currently support
anisotropy.
