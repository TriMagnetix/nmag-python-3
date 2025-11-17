"""
Module which defines the SimulationCore class, an abstract class from which
the real simulation objects are derived (by inheritance).

... (rest of docstring) ...
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Sequence, Union
)

# --- NEW: Import pandas for data writing ---
import pandas as pd

import nsim
import nsim.setup
import nsim.snippets
from clock import SimulationClock
from nsim.si_units import SI
from nmag_exceptions import NmagInternalError, NmagUserError

# --- NEW: Import Quantity from its own file ---
from quantity import Quantity

# Assuming hysteresis_m is a relative import in a package
# If not, this might need to be 'import hysteresis as hysteresis_m'
from . import hysteresis as hysteresis_m

# Setup
features, _ = nsim.setup.get_features()
log = logging.getLogger('nmag')


# -----------------------------------------------------------------------------
# Quantities Definition
# -----------------------------------------------------------------------------

# --- MOVED: The Quantity class is now in quantity.py ---

# The master list of all quantities the simulation can save.
known_quantities: List[Quantity] = [
    #         name                 type       unit           signature context
    Quantity('id',                 'int',     SI(1),         None),
    Quantity('step',               'int',     SI(1),         None),
    # ... (all other Quantity definitions) ...
    Quantity('rho',                'field',   SI('A/m^2'),   None,      'demag')
]

# Helper lookups generated from the master list
known_quantities_by_name: Dict[str, Quantity] = {
    q.name: q for q in known_quantities
}
known_field_quantities: List[Quantity] = [
    q for q in known_quantities if q.type in ('field', 'pfield')
]


def _not_implemented(fn_name: str, cl_name: str):
    """Helper for abstract methods."""
    raise NotImplementedError(
        f"{fn_name} is not implemented by {cl_name}"
    )


# -----------------------------------------------------------------------------
# Simulation Core
# -----------------------------------------------------------------------------

class SimulationCore:
    """
    Abstract base class for simulations.
    ... (rest of docstring) ...
    """

    def __init__(self,
                 name: Optional[str] = None,
                 do_demag: bool = True,
                 sim_id: str = "Generic Simulation class"):

        self.class_id: str = sim_id
        self.units: Optional[Any] = None  # To be set by subclass
        self.do_demag: bool = do_demag
        self.materials: Optional[List[Any]] = None  # To be set by subclass

        # Abbreviations for hysteresis 'save' and 'do' actions
        self.action_abbreviations: Dict[str, Callable] = {}

        # Known quantities for data IO
        self.known_quantities = known_quantities
        self.known_quantities_by_name = known_quantities_by_name
        self.known_field_quantities = known_field_quantities

        # --- Set simulation name ---
        if name is None:
            self.name: str = features.get('etc', 'runid')
        else:
            self.name: str = name
        log.info(f"Simulation(name={self.name}) object created")

        # --- Check for existing files ---
        self._restarting: bool = False
        data_filenames: List[Path] = [
            self._ndtfilename(), self._h5filename(), self._tolfilename()
        ]

        if features.get('nmag', 'clean', raw=True):
            # Delete existing data files
            str_files = [str(f) for f in data_filenames]
            nsim.snippets.rename_old_files(str_files)

        elif features.get('nmag', 'restart', raw=True):
            log.info("Starting simulation in restart mode...")
            self._restarting = True

        else:
            # Check that no data files exist
            for filename in data_filenames:
                if filename.exists():
                    msg = (
                        f"Error: Found old file {filename} -- cannot proceed. "
                        # ... (rest of error message) ...
                    )
                    raise NmagUserError(msg)

        # Simulation clock
        self.clock: SimulationClock = SimulationClock()

        # Max time for advance_time (approximates infinity)
        self.max_time_reached: SI = SI(1, "s")

        # --- Add default save/do abbreviations ---
        self.add_save_abbrev(
            'save_averages',
            lambda sim: sim.save_data(avoid_same_step=True)
        )
        # ... (rest of abbreviations) ...
        self.add_do_abbrev('do_exit', SimulationCore.hysteresis_exit)

        # --- NEW: NDT file writer setup (using pandas) ---
        # We no longer need the _ndt_writer object.
        # We just need to track the column order and if the header is written.
        # We will now output a Tab-Separated Value (TSV) file, which is
        # a more standard format than the old custom-padded one.
        # The file extension is still ".ndt" for compatibility.
        self._ndt_column_names: Optional[List[str]] = None
        self._ndt_column_units: Optional[Dict[str, str]] = None
        self._ndt_header_written: bool = False
        
        # We can also store the metadata (units) in a separate file
        # or as comments in the header if desired.
        # For this refactor, we'll write a simple header + comments.

        # List of enabled physics components (e.g., "exch", "demag")
        self._components: Optional[List[str]] = None

    # --- Clock Properties ---
    # ... (id, stage, step, time, etc. properties) ...
    @property
    def real_time(self) -> float:
        """Time passed in the 'real' world."""
        return self.clock.real_time

    # --- Components and Fields ---
    @property
    def components(self) -> List[str]:
        # ... (implementation) ...
        return components

    def get_all_field_names(self) -> List[str]:
        # ... (implementation) ...
        pass

    # --- Hysteresis Control ---
    # ... (hysteresis_next_stage, hysteresis_exit, relax, hysteresis) ...
    hysteresis.__argdoclong__ = \
        hysteresis_m.simulation_hysteresis.__argdoclong__

    # --- Action Abbreviations ---
    # ... (add_action_abbrev, add_save_abbrev, add_do_abbrev) ...
    def add_do_abbrev(self, abbreviation: str, function: Callable):
        """Add an abbreviation for the 'do' argument of hysteresis."""
        self.add_action_abbrev(abbreviation, function, prefix='do')

    # --- Simulation Flow ---
    # ... (do_next_stage, is_converged) ...

    # --- File Naming ---
    def _get_filename(self, ext: str) -> Path:
        # ... (implementation) ...
        return Path(nsim.snippets.output_file_location(basename))

    def _ndtfilename(self) -> Path:
        # Note: We keep the .ndt extension, but will write TSV format.
        return self._get_filename("_dat.ndt")

    def _h5filename(self) -> Path:
        return self._get_filename("_dat.h5")

    # ... (_statfilename, _tolfilename, get_restart_file_name) ...
    def get_restart_file_name(self) -> str:
        """Return the default name for the restart file."""
        return f"{self.name}_restart.h5"

    # --- Data Gathering ---
    def get_materials_of_field(self, field_name: str) -> Optional[List[Any]]:
        # ... (implementation) ...
        pass

    def get_ndt_columns(self) -> Tuple[List[Tuple[str, Any]], List[Quantity]]:
        # ... (implementation) ...
        # This function remains IDENTICAL to the previous version.
        # It just gathers the data; saving is handled separately.
        # ...
        lt = time.localtime()
        lt_str = (f"{lt[0]:04d}/{lt[1]:02d}/{lt[2]:02d}-"
                  f"{lt[3]:02d}:{lt[4]:02d}:{lt[5]:02d}")
        columns: List[Tuple[str, Any]] = [
            ('id', self.id),
            ('step', self.step),
            # ... (all other columns) ...
            ('localtime', lt_str)
        ]
        quantities: List[Quantity] = [
            self.known_quantities_by_name[name] for name, _ in columns
        ]
        
        # ... (process_subfield helper function) ...
        def process_subfield(field_name: str, prefix: str,
                             quantity: Quantity, mat_name: Optional[str] = None):
            try:
                avg = self.get_subfield_average(field_name, mat_name)
                if avg is None:
                    return
            except Exception:
                return

            if isinstance(avg, list):
                for i, comp_value in enumerate(avg):
                    comp_name = f"{prefix}_{i}"
                    columns.append((comp_name, comp_value))
                    quantities.append(quantity.sub_quantity(comp_name))
            else:
                columns.append((prefix, avg))
                quantities.append(quantity.sub_quantity(prefix))

        # ... (Loop over known_quantities) ...
        for quantity in self.known_quantities:
            # ... (logic to call process_subfield) ...
            pass # (logic is unchanged)

        return columns, quantities

    # --- Data Saving ---
    def _save_fields(self,
                     filename: Optional[str] = None,
                     fieldnames: List[str] = []):
        """Abstract method to save spatially-resolved fields."""
        self._raise_not_implemented("load_m_from_file")

    def _save_averages(self,
                       fields: Optional[Union[str, List[str]]] = None,
                       avoid_same_step: bool = False):
        """
        Save the averages of all available fields into the NDT file
        using pandas.
        """
        # Get the data
        columns, quantities = self.get_ndt_columns()

        # --- 1. Define columns and header (on first call only) ---
        if not self._ndt_header_written:
            # Check that all columns correspond to known quantities
            for q in quantities:
                if not (q in self.known_quantities
                        or q.parent in self.known_quantities):
                    raise NmagInternalError(
                        f"The quantity '{q.name}' is not listed "
                        "as a known quantity and cannot be saved!"
                    )

            # Build the column name list and units dict
            col_names = []
            col_units = {}
            for kq in self.known_quantities:
                selected_qs = [q for q in quantities
                               if q == kq or q.parent == kq]
                for q in selected_qs:
                    col_names.append(q.name)
                    units_str = "1"
                    if q.units is not None:
                        units_str = q.units.dens_str()
                    col_units[q.name] = units_str
            
            self._ndt_column_names = col_names
            self._ndt_column_units = col_units

            # --- Write header and metadata comments ---
            try:
                with open(self._ndtfilename(), 'w') as f:
                    f.write("# Nmag Data Table (TSV format)\n")
                    f.write(f"# Simulation: {self.name}\n")
                    f.write(f"# Date: {time.asctime()}\n")
                    f.write("# --- Units ---\n")
                    for name in self._ndt_column_names:
                        unit = self._ndt_column_units.get(name, 'unknown')
                        f.write(f"# {name}: {unit}\n")
                    f.write("# ---\n")

                # Write the actual header row using pandas
                # We create an empty dataframe with the right columns
                # just to write the header.
                df_header = pd.DataFrame(columns=self._ndt_column_names)
                df_header.to_csv(
                    self._ndtfilename(), 
                    sep='\t',        # Use tabs
                    index=False,
                    mode='a'         # Append to the metadata we just wrote
                )
                self._ndt_header_written = True

            except IOError as e:
                raise NmagUserError(f"Could not write to NDT file: {e}")

        # --- 2. Write the data row ---
        
        # Convert list of tuples to a dictionary for this row
        row_data_dict = dict(columns)

        # Handle SI objects: convert them to simple floats for saving
        for name, value in row_data_dict.items():
            if isinstance(value, SI):
                row_data_dict[name] = float(value)
        
        # Create a single-row DataFrame
        # Using self._ndt_column_names ensures all columns are present
        # and in the correct order, even if some are missing (-> NaN)
        df_row = pd.DataFrame(
            [row_data_dict], 
            columns=self._ndt_column_names
        )

        # Append the row to the CSV file *without* the header
        try:
            df_row.to_csv(
                self._ndtfilename(),
                sep='\t',
                index=False,
                mode='a',          # Append mode
                header=False       # Do not write header again
            )
        except IOError as e:
            raise NmagUserError(f"Could not append to NDT file: {e}")

    def save_data(self,
                  fields: Optional[Union[str, List[str]]] = None,
                  avoid_same_step: bool = False):
        """
        Save simulation data.
        ... (docstring) ...
        """
        # This part is now much simpler, as _save_averages
        # handles the logic of first-write vs. append.
        self._save_averages(fields=fields, avoid_same_step=avoid_same_step)

        # --- Handle field saving (logic unchanged) ---
        if fields is not None:
            field_names_to_save: List[str] = []
            if fields == 'all':
                field_names_to_save = self.get_all_field_names()
            elif isinstance(fields, list):
                field_names_to_save = fields
            else:
                raise NmagUserError(
                    f"save_data 'fields' argument must be None, "
                    f"'all', or a list of strings, not {type(fields)}"
                )

            if field_names_to_save:
                self._save_fields(filename=str(self._h5filename()),
                                  fieldnames=field_names_to_save)

    # -------------------------------------------------------------------------
    # Abstract Methods (to be implemented by FD/FE subclasses)
    # -------------------------------------------------------------------------

    def _raise_not_implemented(self, fn_name: str):
        # ... (implementation) ...
        print(f"{fn_name} is not implemented by {self.class_id}")

    # ... (all other abstract methods: save_mesh, load_mesh, set_params, etc.) ...
    
    def get_subfield_average(self,
                             subfieldname: str,
                             mat_name: Optional[str] = None):
        self._raise_not_implemented("get_subfield_average")