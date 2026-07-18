"""Native HDF5 checkpoints for the supported Python 3 simulation state."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

from si.physical import SI
from simulation.clock import SimulationClock

from .dynamics import ConvergenceTracker, IntegratorConfig

CHECKPOINT_FORMAT = "nmag-python-3-checkpoint"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class CheckpointContents:
    magnetisation: np.ndarray
    pinning: np.ndarray
    external_field: np.ndarray
    current_density: np.ndarray | None
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class RestartRuntimeState:
    clock: SimulationClock
    integrator_config: IntegratorConfig
    stopping_dm_dt: float
    maximum_time_seconds: float
    maximum_dm_dt: float | None
    convergence: ConvergenceTracker


def _array_fingerprint(digest: Any, label: str, values: object, dtype: Any) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest.update(label.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())


def mesh_fingerprint(simulation: Any) -> str:
    """Return a stable hash of the loaded physical mesh and its topology."""
    mesh = simulation._require_mesh()
    simplices = np.asarray(mesh.simplices, dtype=np.int64)
    regions_raw = mesh.regions
    regions = np.asarray(
        [1] * len(simplices) if regions_raw is None else regions_raw,
        dtype=np.int64,
    )
    digest = hashlib.sha256()
    _array_fingerprint(digest, "points", simulation._mesh_points(), np.dtype(np.float64))
    _array_fingerprint(digest, "simplices", simplices, np.dtype(np.int64))
    _array_fingerprint(digest, "regions", regions, np.dtype(np.int64))
    return digest.hexdigest()


def material_fingerprint(simulation: Any) -> str:
    """Hash mesh-region material values that influence supported dynamics.

    Saturation magnetisation is represented per simplex here rather than by
    volume-averaged nodal recovery. The latter can differ by harmless last-bit
    geometry-kernel roundoff after a run, which must not make an otherwise
    compatible checkpoint unloadable.
    """
    coefficients = simulation._nodal_material_coefficients()
    mesh = simulation._require_mesh()
    regions = np.asarray(mesh.regions or [1] * len(mesh.simplices), dtype=np.int64)
    digest = hashlib.sha256()
    for name, values in (
        ("simplex_ms", simulation._simplex_material_ms_values(regions)),
        ("volume_charge_scale", simulation._simplex_volume_charge_scales(regions)),
        ("exchange_prefactor", coefficients.exchange_prefactor),
        ("precession", coefficients.precession),
        ("damping", coefficients.damping),
        ("normalisation", coefficients.normalisation),
        ("stt_adiabatic", coefficients.stt_adiabatic),
        ("stt_nonadiabatic", coefficients.stt_nonadiabatic),
    ):
        _array_fingerprint(digest, name, values, np.dtype(np.float64))
    return digest.hexdigest()


def _finite_array(name: str, values: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.shape != shape:
        raise ValueError(f"Checkpoint {name} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Checkpoint {name} values must be finite.")
    return array


def _finite_number(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Checkpoint {name} must be a finite number.")
    number = float(value)
    if not np.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"Checkpoint {name} must be a {qualifier} number.")
    return number


def _snapshot(simulation: Any) -> CheckpointContents:
    point_count = len(simulation._mesh_points())
    if "m" not in simulation._fields:
        raise RuntimeError("Magnetisation must be set before saving a restart checkpoint.")

    magnetisation = _finite_array("magnetisation", simulation._fields["m"], (point_count, 3))
    pinning = _finite_array(
        "pinning",
        simulation._fields.get("pin", np.ones(point_count, dtype=float)),
        (point_count,),
    )
    external_field = _finite_array(
        "external field",
        simulation._fields.get("H_ext", np.zeros(3, dtype=float)),
        (3,),
    )
    current_density = None
    if "current_density" in simulation._fields:
        current_density = _finite_array(
            "current density",
            simulation._fields["current_density"],
            (point_count, 3),
        )

    config = simulation.integrator_config
    config.validate()
    max_dm_dt = simulation.max_dm_dt
    if max_dm_dt is not None:
        max_dm_dt = _finite_number("maximum dm/dt", max_dm_dt)
    dynamics_state: dict[str, object] = {
        "relative_tolerance": config.relative_tolerance,
        "absolute_tolerance": config.absolute_tolerance,
        "maximum_step_seconds": config.maximum_step_seconds,
        "exact_tstop": config.exact_tstop,
        "stopping_dm_dt": _finite_number(
            "stopping dm/dt", simulation.stopping_dm_dt, positive=True
        ),
        "maximum_time_seconds": simulation.max_time_reached.in_units_of(SI(1.0, "s")),
        "maximum_dm_dt": max_dm_dt,
        "convergence": simulation.convergence.checkpoint_state(),
    }
    _finite_number("maximum time", dynamics_state["maximum_time_seconds"], positive=True)
    metadata: dict[str, object] = {
        "mesh_fingerprint": mesh_fingerprint(simulation),
        "material_fingerprint": material_fingerprint(simulation),
        "do_demag": bool(simulation.do_demag),
        "clock": simulation.clock.checkpoint_state(),
        "dynamics": dynamics_state,
    }
    return CheckpointContents(magnetisation, pinning, external_field, current_density, metadata)


def save_checkpoint(simulation: Any, destination: Path) -> Path:
    """Atomically persist a complete native checkpoint and return its path."""
    contents = _snapshot(simulation)
    destination = destination.expanduser()
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with h5py.File(str(temporary), "w") as h5:
            h5.attrs["format"] = CHECKPOINT_FORMAT
            h5.attrs["version"] = CHECKPOINT_VERSION
            h5.attrs["metadata"] = json.dumps(contents.metadata, allow_nan=False, sort_keys=True)
            state = h5.create_group("state")
            state.create_dataset("m", data=contents.magnetisation, dtype=np.float64)
            state.create_dataset("pin", data=contents.pinning, dtype=np.float64)
            state.create_dataset("H_ext", data=contents.external_field, dtype=np.float64)
            if contents.current_density is not None:
                state.create_dataset(
                    "current_density",
                    data=contents.current_density,
                    dtype=np.float64,
                )
            h5.flush()
        with temporary.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _attribute_text(value: object, name: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise ValueError(f"Checkpoint {name} attribute must be text.")


def read_checkpoint(filename: Path, point_count: int) -> CheckpointContents:
    """Read and fully validate native checkpoint data without mutating a simulation."""
    try:
        with h5py.File(str(filename), "r") as h5:
            if _attribute_text(h5.attrs.get("format"), "format") != CHECKPOINT_FORMAT:
                raise ValueError("File is not an nmag-python-3 native checkpoint.")
            version = cast(object, h5.attrs.get("version"))
            if isinstance(version, bool) or not isinstance(version, (int, np.integer)):
                raise ValueError("Checkpoint version must be an integer.")
            if str(cast(object, version)) != str(CHECKPOINT_VERSION):
                raise ValueError(
                    f"Unsupported checkpoint version {version}; expected {CHECKPOINT_VERSION}."
                )
            try:
                decoded_metadata = json.loads(_attribute_text(h5.attrs.get("metadata"), "metadata"))
            except json.JSONDecodeError as exc:
                raise ValueError("Checkpoint metadata is not valid JSON.") from exc
            if not isinstance(decoded_metadata, dict):
                raise ValueError("Checkpoint metadata must be an object.")
            metadata = cast(dict[str, object], decoded_metadata)
            try:
                state = h5["state"]
                if not isinstance(state, h5py.Group):
                    raise ValueError("Checkpoint state must be an HDF5 group.")
                magnetisation = _finite_array("magnetisation", state["m"], (point_count, 3))
                pinning = _finite_array("pinning", state["pin"], (point_count,))
                external_field = _finite_array("external field", state["H_ext"], (3,))
                current_density = (
                    _finite_array("current density", state["current_density"], (point_count, 3))
                    if "current_density" in state
                    else None
                )
            except KeyError as exc:
                raise ValueError(
                    f"Checkpoint is missing required state dataset {exc.args[0]!r}."
                ) from exc
    except OSError as exc:
        raise ValueError(f"Unable to read checkpoint {filename}: {exc}") from exc
    return CheckpointContents(magnetisation, pinning, external_field, current_density, metadata)


def validate_mesh(simulation: Any, contents: CheckpointContents) -> None:
    saved = contents.metadata.get("mesh_fingerprint")
    if not isinstance(saved, str):
        raise ValueError("Checkpoint metadata is missing a mesh fingerprint.")
    if saved != mesh_fingerprint(simulation):
        raise ValueError("Checkpoint mesh does not match the loaded simulation mesh.")


def runtime_state(simulation: Any, contents: CheckpointContents) -> RestartRuntimeState:
    """Validate full-restart compatibility and parse all non-array runtime state."""
    validate_mesh(simulation, contents)
    saved_material = contents.metadata.get("material_fingerprint")
    if not isinstance(saved_material, str):
        raise ValueError("Checkpoint metadata is missing a material fingerprint.")
    if saved_material != material_fingerprint(simulation):
        raise ValueError("Checkpoint materials do not match the loaded simulation.")
    saved_demag = contents.metadata.get("do_demag")
    if not isinstance(saved_demag, bool) or saved_demag != simulation.do_demag:
        raise ValueError("Checkpoint demagnetization mode does not match the loaded simulation.")

    clock_state = contents.metadata.get("clock")
    dynamics = contents.metadata.get("dynamics")
    if not isinstance(clock_state, dict) or not isinstance(dynamics, dict):
        raise ValueError("Checkpoint metadata is missing clock or dynamics state.")
    parsed_clock_state = cast(dict[str, object], clock_state)
    parsed_dynamics = cast(dict[str, object], dynamics)
    clock = SimulationClock.from_checkpoint_state(parsed_clock_state)
    try:
        exact_tstop = parsed_dynamics["exact_tstop"]
        if not isinstance(exact_tstop, bool):
            raise ValueError("Checkpoint exact_tstop must be a boolean.")
        config = IntegratorConfig(
            relative_tolerance=_finite_number(
                "relative tolerance", parsed_dynamics["relative_tolerance"], positive=True
            ),
            absolute_tolerance=_finite_number(
                "absolute tolerance", parsed_dynamics["absolute_tolerance"], positive=True
            ),
            maximum_step_seconds=_finite_number(
                "maximum step", parsed_dynamics["maximum_step_seconds"], positive=True
            ),
            exact_tstop=exact_tstop,
        )
        config.validate()
        stopping_dm_dt = _finite_number(
            "stopping dm/dt", parsed_dynamics["stopping_dm_dt"], positive=True
        )
        maximum_time = _finite_number(
            "maximum time", parsed_dynamics["maximum_time_seconds"], positive=True
        )
        maximum_dm_dt_raw = parsed_dynamics["maximum_dm_dt"]
        maximum_dm_dt = (
            None
            if maximum_dm_dt_raw is None
            else _finite_number("maximum dm/dt", maximum_dm_dt_raw)
        )
        convergence_state = parsed_dynamics["convergence"]
    except KeyError as exc:
        raise ValueError(f"Checkpoint dynamics state is missing {exc.args[0]!r}.") from exc
    if not isinstance(convergence_state, dict):
        raise ValueError("Checkpoint convergence state must be an object.")
    convergence = ConvergenceTracker.from_checkpoint_state(
        cast(dict[str, object], convergence_state)
    )
    return RestartRuntimeState(
        clock,
        config,
        stopping_dm_dt,
        maximum_time,
        maximum_dm_dt,
        convergence,
    )
