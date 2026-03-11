"""
Module which defines the SimulationCore class, an abstract class from which
the real simulation objects are derived (by inheritance).
Such a design has the goal of separating the parts of the Simulation object
which depend on the specific discretisation (FD, FE) from the parts which
do not depend on it (such as the hysteresis logic).
Here is the structure we have in mind:

                  /----> FDSimulation ----\
SimulationCore ---|                       |---> Simulation
                  \----> FESimulation ----/
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Sequence, Union
)

from clock import SimulationClock
from si.physical import SI
from quantity import (
    known_quantities, 
    known_quantities_by_name, 
    known_field_quantities
)
import hysteresis as hysteresis_m
from mock_features import MockFeatures
from data_writer import DataWriter

# This is a temporary stub to replace nsim.setup.get_features()
# until the full setup module is ported.
features = MockFeatures()

log = logging.getLogger('nmag')

class SimulationCore(ABC):
    """
    Abstract base class for simulations.

    Handles simulation state, clock, data saving, and hysteresis logic
    that is independent of the specific discretisation (FD/FE).
    """

    def __init__(self,
                 name: Optional[str] = None,
                 do_demag: bool = True,
                 sim_id: str = "Generic Simulation class"):

        self.class_id: str = sim_id # String identifying the kind of Simulation class
        self.units: Optional[Any] = None # Simulation units used by this class
        self.do_demag: bool = do_demag # Whether we should include the demag field
        # List of all the materials used by the Simulation object
        self.materials: Optional[List[Any]] = None

        # Dictionary used by the hysteresis method to find abbreviations for
        # frequently used things to save or do.
        # Example: for ``sim.hysteresis(..., save=[('averages', at(...))])``
        # the string 'averages' needs to be a key in this dictionary.
        # The corresponding value is the function to call.
        self.action_abbreviations: Dict[str, Callable] = {}

        # Every quantity the user may want to save needs to be listed here used for data IO
        self.known_quantities = known_quantities
        self.known_quantities_by_name = known_quantities_by_name
        self.known_field_quantities = known_field_quantities

        if name is None:
            self.name: str = features.get('etc', 'runid')
        else:
            self.name: str = name
        log.info(f"Simulation(name={self.name}) object created")

        self._restarting: bool = False
        data_filenames: List[Path] = [
            self._ndtfilename(), self._h5filename(), self._tolfilename()
        ]
        self._manage_output_files(data_filenames)

        self.clock: SimulationClock = SimulationClock()

        # The advance_time method does not allow to carry on the simulation
        # up to t = infinite. Sometimes we want to simulate for n steps,
        # without any time limits. However we must give a time limit.
        # This is then how we approximate t = infinite.
        # For now, we do not provide any function to set or change it.
        # The user should just use:
        #   sim = Simulation()
        #   sim.max_time_reached = SI(1000, "s")
        self.max_time_reached: SI = SI(1, "s")

        # Add abbreviations so that things can be saved just by giving
        # corresponding ID strings.
        # Example: hysteresis(..., save=[('averages', ...)])
        self.add_save_abbrev(
            'save_averages',
            lambda sim: sim.save_data(avoid_same_step=True)
        )
        self.add_save_abbrev(
            'save_fields',
            lambda sim: sim.save_data(fields='all', avoid_same_step=True)
        )
        self.add_save_abbrev(
            'save_field_m',
            lambda sim: sim.save_data(fields=['m'], avoid_same_step=True)
        )
        self.add_save_abbrev(
            'save_restart',
            lambda sim: sim.save_restart_file()
        )
        self.add_do_abbrev('do_next_stage',
                           SimulationCore.hysteresis_next_stage)
        self.add_do_abbrev('do_exit', SimulationCore.hysteresis_exit)

        self.writer = DataWriter(
            ndt_filename=self._ndtfilename(),
            h5_filename=self._h5filename(),
        )

        # The following list contains a description of the physics components
        # which are included in the physical model For example,
        # ["exch", "demag"] indicates that exchange and demag are included.
        # In this case, spin transfer torque is not. This information
        # is used to understand which fields are relevant and which are not
        # (so that we do not save empty fields). Following the previous
        # example, dm_dcurrent, current_density won't be saved.
        self._components: Optional[List[str]] = None

    def _manage_output_files(self, data_filenames: List[Path]):
        """
        Manages existing output files based on configuration (clean/restart).
        """
        if features.get('nmag', 'clean', raw=True):
            for file_path in data_filenames:
                if file_path.exists():
                    new_path = file_path.parent / (file_path.name + ".old")
                    log.info(f"Found old file {file_path}, renaming it to {new_path}")
                    file_path.rename(new_path)
                    
        elif features.get('nmag', 'restart', raw=True):
            log.info("Starting simulation in restart mode...")
            self._restarting = True
            
        else:
            for filename in data_filenames:
                if filename.exists():
                    msg = (
                        f"Error: Found old file {filename} -- cannot proceed. "
                        "To start a simulation script with old data "
                        "files present you either need to use '--clean' "
                        "(and then the old files will be deleted), "
                        "or use '--restart' in which case the run "
                        "will be continued."
                    )
                    raise FileExistsError(msg)

    @property
    def id(self) -> int:
        """ID."""
        return self.clock.id

    @property
    def stage(self) -> int:
        """Stage number."""
        return self.clock.stage

    @property
    def step(self) -> int:
        """Global step number (always increases)."""
        return self.clock.step

    @property
    def time(self) -> float:
        """Global time reached (always increases)."""
        return self.clock.time

    @property
    def stage_step(self) -> int:
        """Step number counted from the beginning of the current stage."""
        return self.clock.stage_step

    @property
    def stage_time(self) -> float:
        """Time reached counted from the beginning of the current stage."""
        return self.clock.stage_time

    @property
    def real_time(self) -> float:
        """Time passed in the 'real' world."""
        return self.clock.real_time

    @property
    def components(self) -> List[str]:
        """Get the physical components included in the model."""
        if self._components is not None:
            return self._components

        else:
            components = ["exch"]
            if self.do_demag:
                components.append("demag")
            self._components = components
            return components

    def get_all_field_names(self) -> List[str]:
        """Get all field names relevant to the enabled components."""
        return [
            q.name for q in self.known_field_quantities
            if q.context is None or q.context in self.components
        ]

    @staticmethod
    def hysteresis_next_stage(sim: SimulationCore):
        """
        Terminate the current stage of the hysteresis computation
        and start the next one.
        """
        sim.clock.stage_end = True

    @staticmethod
    def hysteresis_exit(sim: SimulationCore):
        """
        Exit from the running hysteresis computation.
        """
        sim.clock.exit_hysteresis = True
        sim.clock.stage_end = True

    simulation_relax = hysteresis_m.simulation_relax
    relax = simulation_relax
    
    simulation_hysteresis = hysteresis_m.simulation_hysteresis
    hysteresis = simulation_hysteresis

    def add_action_abbrev(self,
                          abbreviation: str,
                          function: Callable,
                          prefix: Optional[str] = None):
        """Add an abbreviation for a 'save' or 'do' action."""
        if prefix is None:
            self.action_abbreviations[abbreviation] = function
            return

        else:
            valid_prefixes = ["save", "do"]
            if prefix not in valid_prefixes:
                raise ValueError(
                    f"Valid prefixes for action abbreviations "
                    f"are {valid_prefixes}, you gave '{prefix}'!"
                )

            if abbreviation.startswith(prefix):
                self.action_abbreviations[abbreviation] = function
            else:
                full_abbreviation = f"{prefix}_{abbreviation}"
                self.action_abbreviations[full_abbreviation] = function

    def add_save_abbrev(self, abbreviation: str, function: Callable):
        """Add an abbreviation to be used in the 'save' argument of the
        hysteresis method. For example, if you use the following:

            def funky_function(sim): print "Hello, I'm Funky!"
            sim.add_save_abbrev('funky', funky_function)

        Then you can call:

            sim.hysteresis(Hs, save=[('funky', at('convergence'))])

        and this will be equivalent to:

            sim.hysteresis(Hs, save=[(funky_function, at('convergence')])
        """
        self.add_action_abbrev(abbreviation, function, prefix='save')

    def add_do_abbrev(self, abbreviation: str, function: Callable):
        """Add an abbreviation for the 'do' argument of hysteresis."""
        self.add_action_abbrev(abbreviation, function, prefix='do')

    def do_next_stage(self, stage: Optional[int] = None):
        """Increment the simulation stage."""
        self.clock.inc_stage(stage=stage)

    def is_converged(self) -> bool:
        """Returns True when convergence has been reached."""
        return self.clock.convergence

    def _get_filename(self, ext: str) -> Path:
        """Get the full, absolute path for an output file."""
        basename = self.name + ext
        return Path(features.get('etc', 'savedir')) / basename

    def _ndtfilename(self) -> Path:
        return self._get_filename("_dat.ndt")

    def _h5filename(self) -> Path:
        return self._get_filename("_dat.h5")

    def _statfilename(self) -> Path:
        return self._get_filename("_cvode.log")

    def _tolfilename(self) -> Path:
        return self._get_filename("_tol.log")

    def get_restart_file_name(self) -> str:
        """Return the default name for the restart file."""
        return f"{self.name}_restart.h5"

    def get_materials_of_field(self, field_name: str) -> List[Any]:
        """
        Returns all materials for a per-material field.
        Returns an empty list if the field is not per-material or 
        if materials have not been defined yet.
        """
        quantity = self.known_quantities_by_name[field_name]
        
        if '?' in (quantity.signature or ""):
            return self.materials if self.materials is not None else []
        
        return []


    @abstractmethod
    def save_spatial_fields(self,
                     filename: Optional[str] = None,
                     fieldnames: List[str] = []):
        """Abstract method to save spatially-resolved fields."""
        pass


    def save_data(self,
                  fields: Optional[Union[str, List[str]]] = None,
                  avoid_same_step: bool = False):
        """
        Save simulation data.

        - Averages are saved to the *.ndt file (TSV format).
        - Spatially resolved fields are saved to the *.h5 file.

        :Parameters:
          `fields` : None, 'all' or list of fieldnames
            If None, only saves averages.
            If 'all', saves all available fields.
            If a list (e.g., ['m', 'H_demag']), saves only those fields.

          `avoid_same_step` : bool
            If True, only save if clock.step has changed since last save.
            This prevents duplicate data points during hysteresis loops.
        """
        self.writer.save(self, fields, avoid_same_step)

    @abstractmethod
    def save_mesh(self, filename: str):
        """Save the mesh to a file."""
        pass

    @abstractmethod
    def load_mesh(self,
                  filename: str,
                  region_names_and_mag_mats: Any,
                  unit_length: float,
                  do_reorder: bool = False,
                  manual_distribution: Any = None):
        """Load a mesh from a file."""
        pass

    @abstractmethod
    def create_mesh(self,
                    cell_nums: Sequence[int],
                    cell_sizes: Sequence[float],
                    materials: Any,
                    regions: Optional[Callable] = None,
                    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """Create a new mesh."""
        pass

    @abstractmethod
    def set_params(self,
                   stopping_dm_dt: Optional[SI] = None,
                   ts_rel_tol: Optional[float] = None,
                   ts_abs_tol: Optional[float] = None):
        """Set simulation parameters."""
        pass

    @abstractmethod
    def reinitialise(self, initial_time: Optional[float] = None):
        """Re-initialise the simulation."""
        pass

    @abstractmethod
    def set_local_magnetic_coupling(self, mat1: Any, mat2: Any, coupling: Any):
        """Set local magnetic coupling between materials."""
        pass

    @abstractmethod
    def set_H_ext(self, values: Any, unit: Optional[SI] = None):
        """Set the external magnetic field."""
        pass

    @abstractmethod
    def set_m(self, values: Any, subfieldname: Optional[str] = None):
        """Set the magnetization."""
        pass

    @abstractmethod
    def set_pinning(self, values: Any):
        """Set pinning sites."""
        pass

    @abstractmethod
    def set_current_density(self, values: Any, unit: Optional[SI] = None):
        """Set current density."""
        pass

    @abstractmethod
    def advance_time(self,
                     target_time: SI,
                     max_it: int = -1,
                     exact_tstop: Optional[bool] = None):
        """Advance the simulation time."""
        pass

    @abstractmethod
    def save_restart_file(self,
                          filename: Optional[str] = None,
                          fieldnames: List[str] = ['m'],
                          all: bool = False):
        """Save a restart file."""
        pass

    @abstractmethod
    def load_restart_file(self, filename: Optional[str] = None):
        """Load a restart file."""
        pass

    @abstractmethod
    def save_m_to_file(self,
                       filename: str,
                       format: Optional[str] = None):
        """Save magnetization to a file."""
        pass

    @abstractmethod
    def load_m_from_file(self,
                         filename: str,
                         format: Optional[str] = None):
        """Load magnetization from a file."""
        pass

    @abstractmethod
    def probe_subfield(self,
                       subfieldname: str,
                       pos: SI,
                       unit: Optional[SI] = None):
        """Probe a subfield at a specific position."""
        pass

    @abstractmethod
    def probe_subfield_siv(self,
                           subfieldname: str,
                           pos: List[float],
                           unit: Optional[SI] = None):
        """Probe a subfield returning SI values."""
        pass

    @abstractmethod
    def get_subfield(self,
                     subfieldname: str,
                     units: Optional[SI] = None):
        """Get the entire array for a subfield."""
        pass

    @abstractmethod
    def get_subfield_average(self,
                             subfieldname: str,
                             mat_name: Optional[str] = None):
        """Get the average value of a subfield."""
        pass
