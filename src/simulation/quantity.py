from dataclasses import dataclass
from typing import Optional

from si.physical import SI


@dataclass(frozen=True)
class Quantity:
    """A dataclass describing a physical quantity for data saving.

    A quantity is a description of any thing that may be saved inside
    a ndt file. It may just represent an integer-floating point number
    or a vector field. The Quantity class is a descriptor for such a thing.
    It contains details about the corresponding type, SI unit, the id name,
    the signature (indices and per-material).
    """

    name: str
    type: str  # 'int', 'float', 'field', 'pfield', 'date'
    units: SI | None
    signature: str | None = None
    context: str | None = None
    parent: Optional["Quantity"] = None

    def __post_init__(self) -> None:
        if self.signature is None:
            object.__setattr__(self, "signature", self.name)

    def sub_quantity(self, name: str) -> "Quantity":
        """Creates a new Quantity representing a sub-component."""
        return self.__class__(
            name=name,
            type=self.type,
            units=self.units,
            signature=self.signature,
            context=self.context,
            parent=self,
        )


UNIT_DIMENSIONLESS = SI(1)
UNIT_SECONDS = SI("s")
UNIT_A_PER_M = SI("A/m")
UNIT_A_PER_M2 = SI("A/m^2")
UNIT_A_PER_M_PER_S = SI("A/m/s")
UNIT_INVERSE_SECONDS = SI("1/s")
UNIT_J_PER_M3 = SI("J/m^3")
UNIT_AMPERE = SI("A")


known_quantities: list[Quantity] = [
    #         name                 type       unit           signature context
    Quantity("id", "int", UNIT_DIMENSIONLESS, None),
    Quantity("step", "int", UNIT_DIMENSIONLESS, None),
    Quantity("stage_step", "int", UNIT_DIMENSIONLESS, None),
    Quantity("stage", "int", UNIT_DIMENSIONLESS, None),
    Quantity("last_step_dt", "float", UNIT_SECONDS, None),
    Quantity("time", "float", UNIT_SECONDS, None),
    Quantity("stage_time", "float", UNIT_SECONDS, None),
    Quantity("real_time", "float", UNIT_SECONDS, None),
    Quantity("unixtime", "float", UNIT_SECONDS, None),
    Quantity("maxangle", "float", UNIT_DIMENSIONLESS, None),
    Quantity("localtime", "date", None, None),
    Quantity("H_total", "field", UNIT_A_PER_M, "_?_*"),
    Quantity("M", "field", UNIT_A_PER_M, "_?_*"),
    Quantity("m", "pfield", UNIT_DIMENSIONLESS, "_?_*"),
    Quantity("pin", "pfield", UNIT_DIMENSIONLESS, None),
    Quantity("current_density", "pfield", UNIT_A_PER_M2, "_*", "stt"),
    Quantity("dmdt", "field", UNIT_A_PER_M_PER_S, "_?_*"),
    Quantity("dm_dcurrent", "field", UNIT_INVERSE_SECONDS, "_?_*", "stt"),
    Quantity("H_ext", "field", UNIT_A_PER_M, "_*"),
    Quantity("H_anis", "field", UNIT_A_PER_M, "_?_*", "anis"),
    Quantity("H_exch", "field", UNIT_A_PER_M, "_?_*", "exch"),
    Quantity("H_demag", "field", UNIT_A_PER_M, "_*", "demag"),
    Quantity("E_total", "field", UNIT_J_PER_M3, "_?"),
    Quantity("E_ext", "field", UNIT_J_PER_M3, "_?"),
    Quantity("E_anis", "field", UNIT_J_PER_M3, "_?", "anis"),
    Quantity("E_exch", "field", UNIT_J_PER_M3, "_?", "exch"),
    Quantity("E_demag", "field", UNIT_J_PER_M3, "_?", "demag"),
    Quantity("phi", "field", UNIT_AMPERE, None, "demag"),
    Quantity("rho", "field", UNIT_A_PER_M2, None, "demag"),
]

known_quantities_by_name: dict[str, Quantity] = {q.name: q for q in known_quantities}

known_field_quantities: list[Quantity] = [
    q for q in known_quantities if q.type in ("field", "pfield")
]
