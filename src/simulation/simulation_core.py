r"""
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

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from nmag.config import NmagConfig
from nmag.output import prepare_output_files
from si.physical import SI

from . import hysteresis as hysteresis_m
from .clock import SimulationClock
from .data_writer import DataWriter
from .quantity import known_field_quantities, known_quantities, known_quantities_by_name

Action = Callable[[Any], Any]

log = logging.getLogger("nmag")


class SimulationCore(ABC):
    """
    Abstract base class for simulations.

    Handles simulation state, clock, data saving, and hysteresis logic
    that is independent of the specific discretisation (FD/FE).
    """

    def __init__(
        self,
        name: str | None = None,
        do_demag: bool = True,
        sim_id: str = "Generic Simulation class",
        config: NmagConfig | None = None,
    ) -> None:

        self.class_id: str = sim_id  # String identifying the kind of Simulation class
        self.units: Any | None = None  # Simulation units used by this class
        self.do_demag: bool = do_demag  # Whether we should include the demag field
        # List of all the materials used by the Simulation object
        self.materials: list[Any] | None = None

        # Dictionary used by the hysteresis method to find abbreviations for
        # frequently used things to save or do.
        # Example: for ``sim.hysteresis(..., save=[('averages', at(...))])``
        # the string 'averages' needs to be a key in this dictionary.
        # The corresponding value is the function to call.
        self.action_abbreviations: dict[str, Action] = {}

        # Every quantity the user may want to save needs to be listed here used for data IO
        self.known_quantities = known_quantities
        self.known_quantities_by_name = known_quantities_by_name
        self.known_field_quantities = known_field_quantities

        self.config = NmagConfig.from_environment() if config is None else config
        self.name = name or self.config.default_name
        log.info(f"Simulation(name={self.name}) object created")

        self._restarting: bool = False
        data_filenames: list[Path] = [self._ndtfilename(), self._h5filename()]
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
        self.add_save_abbrev("save_averages", lambda sim: sim.save_data(avoid_same_step=True))
        self.add_save_abbrev(
            "save_fields", lambda sim: sim.save_data(fields="all", avoid_same_step=True)
        )
        self.add_save_abbrev(
            "save_field_m", lambda sim: sim.save_data(fields=["m"], avoid_same_step=True)
        )
        self.add_save_abbrev("save_restart", lambda sim: sim.save_restart_file())
        self.add_do_abbrev("do_next_stage", SimulationCore.hysteresis_next_stage)
        self.add_do_abbrev("do_exit", SimulationCore.hysteresis_exit)

        self.writer = DataWriter(
            ndt_filename=self._ndtfilename(),
            h5_filename=self._h5filename(),
            append=self.config.output_policy == "append",
        )
        self.last_spatial_save_timings_seconds: dict[str, float] = {}
        self.last_maxangle_timings_seconds: dict[str, float] = {}

        # The following list contains a description of the physics components
        # which are included in the physical model For example,
        # ["exch", "demag"] indicates that exchange and demag are included.
        # In this case, spin transfer torque is not. This information
        # is used to understand which fields are relevant and which are not
        # (so that we do not save empty fields). Following the previous
        # example, dm_dcurrent, current_density won't be saved.
        self._components: list[str] | None = None

    def _manage_output_files(self, data_filenames: list[Path]) -> None:
        """Apply this simulation's explicit NDT/HDF5 output policy."""

        prepare_output_files(data_filenames, self.config.output_policy)

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
    def time(self) -> SI:
        """Global time reached (always increases)."""
        return self.clock.time

    @property
    def stage_step(self) -> int:
        """Step number counted from the beginning of the current stage."""
        return self.clock.stage_step

    @property
    def stage_time(self) -> SI:
        """Time reached counted from the beginning of the current stage."""
        return self.clock.stage_time

    @property
    def real_time(self) -> SI:
        """Time passed in the 'real' world."""
        return self.clock.real_time

    @property
    def last_step_dt(self) -> SI:
        """Length of the last simulation step."""
        return self.clock.last_step_dt_si

    @property
    def components(self) -> list[str]:
        """Get the physical components included in the model."""
        if self._components is not None:
            return self._components

        else:
            components = ["exch"]
            if self.do_demag:
                components.append("demag")
            self._components = components
            return components

    def get_all_field_names(self) -> list[str]:
        """Get all field names relevant to the enabled components."""
        return [
            q.name
            for q in self.known_field_quantities
            if q.context is None or q.context in self.components
        ]

    @staticmethod
    def hysteresis_next_stage(sim: SimulationCore) -> None:
        """
        Terminate the current stage of the hysteresis computation
        and start the next one.
        """
        sim.clock.stage_end = True

    @staticmethod
    def hysteresis_exit(sim: SimulationCore) -> None:
        """
        Exit from the running hysteresis computation.
        """
        sim.clock.exit_hysteresis = True
        sim.clock.stage_end = True

    simulation_relax = hysteresis_m.simulation_relax
    relax = simulation_relax

    simulation_hysteresis = hysteresis_m.simulation_hysteresis
    hysteresis = simulation_hysteresis

    def add_action_abbrev(
        self,
        abbreviation: str,
        function: Action,
        prefix: str | None = None,
    ) -> None:
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

    def add_save_abbrev(self, abbreviation: str, function: Action) -> None:
        """Add an abbreviation to be used in the 'save' argument of the
        hysteresis method. For example, if you use the following:

            def funky_function(sim): print "Hello, I'm Funky!"
            sim.add_save_abbrev('funky', funky_function)

        Then you can call:

            sim.hysteresis(Hs, save=[('funky', at('convergence'))])

        and this will be equivalent to:

            sim.hysteresis(Hs, save=[(funky_function, at('convergence')])
        """
        self.add_action_abbrev(abbreviation, function, prefix="save")

    def add_do_abbrev(self, abbreviation: str, function: Action) -> None:
        """Add an abbreviation for the 'do' argument of hysteresis."""
        self.add_action_abbrev(abbreviation, function, prefix="do")

    def do_next_stage(self, stage: int | None = None) -> None:
        """Increment the simulation stage."""
        self.clock.inc_stage(stage=stage)

    def is_converged(self) -> bool:
        """Returns True when convergence has been reached."""
        return self.clock.convergence

    def _get_filename(self, ext: str) -> Path:
        """Get the full, absolute path for an output file."""
        basename = self.name + ext
        return self.config.output_directory / basename

    def _ndtfilename(self) -> Path:
        return self._get_filename("_dat.ndt")

    def _h5filename(self) -> Path:
        return self._get_filename("_dat.h5")

    def _statfilename(self) -> Path:
        return self._get_filename("_cvode.log")

    def _tolfilename(self) -> Path:
        return self._get_filename("_tol.log")

    def get_restart_file_name(self) -> Path:
        """Return the default name for the restart file."""
        return self.config.output_directory / f"{self.name}_restart.h5"

    def get_materials_of_field(self, field_name: str) -> list[Any]:
        """
        Returns all materials for a per-material field.
        Returns an empty list if the field is not per-material or
        if materials have not been defined yet.
        """
        quantity = self.known_quantities_by_name[field_name]

        if "?" in (quantity.signature or ""):
            return self.materials if self.materials is not None else []

        return []

    @abstractmethod
    def save_spatial_fields(
        self,
        filename: str | None = None,
        fieldnames: list[str] | None = None,
    ) -> None:
        """Abstract method to save spatially-resolved fields."""
        pass

    def save_data(
        self,
        fields: str | list[str] | None = None,
        avoid_same_step: bool = False,
    ) -> None:
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
    def save_mesh(self, filename: str) -> None:
        """Save the mesh to a file."""
        pass

    @abstractmethod
    def load_mesh(
        self,
        filename: str,
        region_names_and_mag_mats: Any,
        unit_length: float,
        do_reorder: bool = False,
        manual_distribution: Any = None,
    ) -> None:
        """Load a mesh from a file."""
        pass

    @abstractmethod
    def create_mesh(
        self,
        cell_nums: Sequence[int],
        cell_sizes: Sequence[float],
        materials: Any,
        regions: Callable[..., Any] | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Create a new mesh."""
        pass

    @abstractmethod
    def set_params(
        self,
        stopping_dm_dt: SI | None = None,
        ts_rel_tol: float | None = None,
        ts_abs_tol: float | None = None,
    ) -> None:
        """Set simulation parameters."""
        pass

    @abstractmethod
    def reinitialise(self, initial_time: float | None = None) -> None:
        """Re-initialise the simulation."""
        pass

    @abstractmethod
    def set_local_magnetic_coupling(self, mat1: Any, mat2: Any, coupling: Any) -> None:
        """Set local magnetic coupling between materials."""
        pass

    @abstractmethod
    def set_H_ext(self, values: Any, unit: SI | None = None) -> None:
        """Set the external magnetic field."""
        pass

    @abstractmethod
    def set_m(self, values: Any, subfieldname: str | None = None) -> None:
        """Set the magnetization."""
        pass

    @abstractmethod
    def set_pinning(self, values: Any) -> None:
        """Set pinning sites."""
        pass

    @abstractmethod
    def set_current_density(self, values: Any, unit: SI | None = None) -> None:
        """Set current density."""
        pass

    @abstractmethod
    def advance_time(
        self,
        target_time: SI,
        max_it: int = -1,
        exact_tstop: bool | None = None,
    ) -> SI:
        """Advance the simulation time."""
        pass

    @abstractmethod
    def save_restart_file(
        self, filename: str | None = None, fieldnames: list[str] | None = None, all: bool = False
    ) -> None:
        """Save a restart file."""
        pass

    @abstractmethod
    def load_restart_file(self, filename: str | None = None) -> None:
        """Load a restart file."""
        pass

    @abstractmethod
    def save_m_to_file(self, filename: str, format: str | None = None) -> None:
        """Save magnetization to a file."""
        pass

    @abstractmethod
    def load_m_from_file(self, filename: str, format: str | None = None) -> None:
        """Load magnetization from a file."""
        pass

    @abstractmethod
    def probe_subfield(self, subfieldname: str, pos: SI, unit: SI | None = None) -> Any:
        """Probe a subfield at a specific position."""
        pass

    @abstractmethod
    def probe_subfield_siv(
        self,
        subfieldname: str,
        pos: list[float],
        unit: SI | None = None,
    ) -> Any:
        """Probe a subfield returning SI values."""
        pass

    @abstractmethod
    def get_subfield(self, subfieldname: str, units: SI | None = None) -> Any:
        """Get the entire array for a subfield."""
        pass

    @abstractmethod
    def get_subfield_average(self, subfieldname: str, mat_name: str | None = None) -> Any:
        """Get the average value of a subfield."""
        pass

    @abstractmethod
    def get_maxangle_average(self, field_name: str) -> float | None:
        """Get the average maximum angle for a field."""
        pass
