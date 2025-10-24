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
    the signature (indices and per-material)."""
    name: str
    type: str  # 'int', 'float', 'field', 'pfield', 'date'
    units: Optional[SI]
    signature: Optional[str] = None
    context: Optional[str] = None
    parent: Optional['Quantity'] = None

    def __post_init__(self):
        if self.signature is None:
            object.__setattr__(self, 'signature', self.name)

    def sub_quantity(self, name: str) -> 'Quantity':
        """Creates a new Quantity representing a sub-component."""
        return self.__class__(
            name=name,
            type=self.type,
            units=self.units,
            signature=self.signature,
            context=self.context,
            parent=self
        )
