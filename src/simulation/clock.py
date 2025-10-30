from si.physical import SI
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from tabulate import tabulate

def fmt_time(t: SI, fmt_ps: str = ".2f", fmt_ns: str = ".2f") -> str:
    """Formats an SI time object into picoseconds or nanoseconds."""
    t_ps = float(t / SI(1e-12, "s"))
    
    ps_str = f"{t_ps:{fmt_ps}}"
    ns_str = f"{(t_ps / 1000.0):{fmt_ns}}"
    
    return f"{ps_str} ps" if t_ps < 100.0 else f"{ns_str} ns"
    
@dataclass
class SimulationClock:
    """
    This object specifies all the parameters which define the current time
    in the simulation, such as the simulation time, step number, ...
    
    Attributes:
      id: Unique identifier for data saved. Incremented on save.
      stage: Stage number. Increments when the external field changes.
      step: Total number of steps performed (always increases).
      stage_step: Step number from the beginning of the current stage.
      zero_stage_step: The value of 'step' at the beginning of the stage.
      time: Total simulation time (always increases).
      stage_time: The simulation time from the beginning of the stage.
      zero_stage_time: The value of 'time' at the beginning of the stage.
      real_time: The real world time used for advancing time.
      last_step_dt_si: Last time step's length in SI units.
      convergence: Flag indicating if convergence is reached.
      stage_end: Flag indicating the end of a stage.
      exit_hysteresis: Flag to signal exit from hysteresis loop.
    """
    id: int = -1
    stage: int = 1
    step: int = 0
    time: SI = field(default_factory=lambda: SI(0.0, "s"))
    stage_step: int = 0
    stage_time: SI = field(default_factory=lambda: SI(0.0, "s"))
    real_time: SI = field(default_factory=lambda: SI(0.0, "s"))
    stage_end: bool = False
    convergence: bool = False
    exit_hysteresis: bool = False
    zero_stage_time: SI = field(default_factory=lambda: SI(0.0, "s"))
    zero_stage_step: int = 0
    time_reached_su: float = 0.0
    time_reached_si: SI = field(default_factory=lambda: SI(0.0, "s"))
    last_step_dt_su: float = 0.0
    last_step_dt_si: SI = field(default_factory=lambda: SI(0.0, "s"))

    # __init__ and __repr__ are GONE (auto-generated)

    def inc_stage(self, stage: Optional[int] = None):
        """Advance the clock to the next stage."""
        if stage is None:
            self.stage += 1
        else:
            self.stage = stage
        self.stage_step = 0
        self.stage_time = SI(0.0, "s")
        self.convergence = False
        self.zero_stage_step = self.step
        self.zero_stage_time = self.time

    # This method was updated to use tabulate, the format of the data printed out might
    # look slightly different, but there is a lot less manual formatting code here now.
    def __str__(self) -> str:
        ft = fmt_time

        rows = [
            [f"ID={self.id}", f"Step={self.step}", 
            f"Time={ft(self.time)}", f"Last step size={ft(self.last_step_dt_si)}"],
            
            ["", f"Stage={self.stage}", 
            f"Stage-step={self.stage_step}", f"Stage-time={ft(self.stage_time)}"],
            
            ["", f"Convergence={self.convergence}", 
            f"Stage-end={self.stage_end}", f"Exit hysteresis={self.exit_hysteresis}"]
        ]

        table = tabulate(rows, tablefmt="pipe")

        sep_line = "=" * (len(table.splitlines()[0]))
        return f"{sep_line}\n{table}\n{sep_line}"
