from dataclasses import dataclass
from typing import Optional, List, Dict

# Assuming this is your import path based on your snippet
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
    units: Optional[SI]
    signature: Optional[str] = None
    context: Optional[str] = None
    parent: Optional['Quantity'] = None

    def __post_init__(self):
        # Because frozen=True, we must use object.__setattr__ to modify self
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

known_quantities: List[Quantity] = [
    #         name                 type       unit           signature context
    Quantity('id',                 'int',     SI(1),         None),
    Quantity('step',               'int',     SI(1),         None),
    Quantity('stage_step',         'int',     SI(1),         None),
    Quantity('stage',              'int',     SI(1),         None),
    Quantity('last_time_dt',       'float',   SI('s'),       None),
    Quantity('time',               'float',   SI('s'),       None),
    Quantity('stage_time',         'float',   SI('s'),       None),
    Quantity('real_time',          'float',   SI('s'),       None),
    Quantity('unixtime',           'float',   SI('s'),       None),
    Quantity('maxangle',           'float',   SI(1),         None),
    Quantity('localtime',          'date',    None,          None),
    Quantity('H_total',            'field',   SI('A/m'),     '_?_*'),
    Quantity('M',                  'field',   SI('A/m'),     '_?_*'),
    Quantity('m',                  'pfield',  SI(1),         '_?_*'),
    Quantity('pin',                'pfield',  SI(1),         None),
    Quantity('current_density',    'pfield',  SI('A/m^2'),   '_*',      'stt'),
    Quantity('dmdt',               'field',   SI('1/s'),     '_?_*'),
    Quantity('dm_dcurrent',        'field',   SI('1/s'),     '_?_*',    'stt'),
    Quantity('H_ext',              'field',   SI('A/m'),     '_*'),
    Quantity('H_anis',             'field',   SI('A/m'),     '_?_*',    'anis'),
    Quantity('H_exch',             'field',   SI('A/m'),     '_?_*',    'exch'),
    Quantity('H_demag',            'field',   SI('A/m'),     '_*',      'demag'),
    Quantity('E_total',            'field',   SI('J/m^3'),   '_?'),
    Quantity('E_ext',              'field',   SI('J/m^3'),   '_?'),
    Quantity('E_anis',             'field',   SI('J/m^3'),   '_?',      'anis'),
    Quantity('E_exch',             'field',   SI('J/m^3'),   '_?',      'exch'),
    Quantity('E_demag',            'field',   SI('J/m^3'),   '_?',      'demag'),
    Quantity('phi',                'field',   SI('A'),       None,      'demag'),
    Quantity('rho',                'field',   SI('A/m^2'),   None,      'demag')
]

known_quantities_by_name: Dict[str, Quantity] = {
    q.name: q for q in known_quantities
}

known_field_quantities: List[Quantity] = [
    q for q in known_quantities if q.type in ('field', 'pfield')
]