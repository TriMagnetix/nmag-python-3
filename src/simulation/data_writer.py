"""
Module defining the DataWriter class and the interface it expects from data sources.
"""
from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import (
    List, Optional, Tuple, Dict, Any, Union, Protocol, runtime_checkable
)

import pandas as pd
from nsim.si_units import SI
from quantity import (Quantity, known_quantities)

log = logging.getLogger('nmag')

# Since the SimulationCore class was so coupled to the data writing, 
# we create a sort of interface for it (Protocol) so that the DataWriter
# isn't dependent on the concrete SimulationCore class. This makes
# testing easier and help decouple the writer from the core logic.
# https://typing.python.org/en/latest/spec/protocol.html
@runtime_checkable
class SimulationSource(Protocol):
    # Metadata
    name: str
    id: int
    
    # Clock State
    step: int
    stage: int
    stage_step: int
    time: float
    stage_time: float
    real_time: float

    # Data Access Methods
    def get_subfield_average(self, subfieldname: str, mat_name: Optional[str] = None) -> Any: ...
    
    def get_materials_of_field(self, field_name: str) -> List[Any]: ...
    
    def get_all_field_names(self) -> List[str]: ...

    # Field Saving Capability
    def save_spatial_fields(self, filename: str, fieldnames: List[str]) -> None: ...

class DataWriter:
    def __init__(self, ndt_filename: Path, h5_filename: Path):
        """
        :param ndt_filename: Path to the tabular output (.ndt).
        :param h5_filename: Path to the spatial output (.h5).
        :param quantities: The schema definition.
        """
        self.ndt_filename = ndt_filename
        self.h5_filename = h5_filename
        self.quantities = known_quantities
        
        self.quantities_by_name: Dict[str, Quantity] = {
            q.name: q for q in known_quantities
        }
        
        self._header_written: bool = False
        self._column_names: Optional[List[str]] = None
        self._column_units: Optional[Dict[str, str]] = None
        self._last_saved_step: int = -1

    def save(self, 
             source: SimulationSource, 
             fields: Optional[Union[str, List[str]]] = None, 
             avoid_same_step: bool = False):
        
        # 1. Check Step Counter
        current_step = source.step
        if avoid_same_step and current_step == self._last_saved_step:
            return
        
        self._last_saved_step = current_step

        # 2. Write Tabular Data
        self._write_ndt_row(source)

        # 3. Trigger Spatial Field Saving
        if fields is not None:
            self._trigger_field_save(source, fields)

    def _trigger_field_save(self, source: SimulationSource, fields: Union[str, List[str]]):
        field_names_to_save: List[str] = []
        
        if fields == 'all':
            field_names_to_save = source.get_all_field_names()
        elif isinstance(fields, list):
            field_names_to_save = fields
        else:
            raise ValueError(f"Invalid fields argument: {fields}")

        if field_names_to_save:
            # The writer manages the filename, but delegates the writing logic to the source
            source.save_spatial_fields(
                filename=str(self.h5_filename),
                fieldnames=field_names_to_save
            )

    def _gather_data(self, source: SimulationSource) -> Tuple[List[Tuple[str, Any]], List[Quantity]]:
        lt = time.localtime()
        lt_str = (f"{lt[0]:04d}/{lt[1]:02d}/{lt[2]:02d}-"
                  f"{lt[3]:02d}:{lt[4]:02d}:{lt[5]:02d}")

        columns: List[Tuple[str, Any]] = [
            ('id', source.id),
            ('step', source.step),
            ('stage_step', source.stage_step),
            ('stage', source.stage),
            ('time', source.time),
            ('stage_time', source.stage_time),
            ('real_time', source.real_time),
            ('unixtime', SI(time.time(), 's')),
            ('localtime', lt_str)
        ]
        
        current_quantities: List[Quantity] = [
            self.quantities_by_name[name] for name, _ in columns
        ]

        def process_subfield(field_name: str, prefix: str, quantity: Quantity, mat_name: Optional[str] = None):
            try:
                avg = source.get_subfield_average(field_name, mat_name)
                if avg is None: return
            except Exception:
                return

            if isinstance(avg, list):
                for i, comp_value in enumerate(avg):
                    comp_name = f"{prefix}_{i}"
                    columns.append((comp_name, comp_value))
                    current_quantities.append(quantity.sub_quantity(comp_name))
            else:
                columns.append((prefix, avg))
                current_quantities.append(quantity.sub_quantity(prefix))

        for quantity in self.quantities:
            field_name = quantity.name
            if quantity.type in ['field', 'pfield']:
                if '?' in (quantity.signature or ""):
                    # Note: We assume the objects returned here have a .name attribute
                    mats = source.get_materials_of_field(field_name)
                    for material in mats:
                        prefix = f"{field_name}_{material.name}"
                        process_subfield(field_name, prefix, quantity, mat_name=material.name)
                else:
                    process_subfield(field_name, field_name, quantity)

        return columns, current_quantities

    def _write_ndt_row(self, source: SimulationSource):
        columns, quantities = self._gather_data(source)

        if not self._header_written:
            # ... (Same header writing logic as before) ...
            # Make sure to change 'sim.name' to 'source.name'
            col_names = []
            col_units = {}
            # ... (building logic) ...
            
            self._column_names = col_names
            self._column_units = col_units
            
            # Write Header
            with open(self.ndt_filename, 'w') as f:
                 f.write(f"# Simulation: {source.name}\n")
                 # ... rest of header logic ...
            
            # Pandas header write
            pd.DataFrame(columns=self._column_names).to_csv(
                self.ndt_filename, sep='\t', index=False, mode='a'
            )
            self._header_written = True

        # Write Data Row
        row_data_dict = dict(columns)
        for name, value in row_data_dict.items():
            if isinstance(value, SI):
                row_data_dict[name] = float(value)
        
        pd.DataFrame([row_data_dict], columns=self._column_names).to_csv(
            self.ndt_filename, sep='\t', index=False, mode='a', header=False
        )