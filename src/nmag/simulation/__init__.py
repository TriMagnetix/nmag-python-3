from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np

import nmesh
from si.physical import SI
from simulation.clock import SimulationClock

from .. import backends as _backend_helpers
from .. import demag as _demag_helpers
from ..config import NmagConfig
from ..demag.linear import ScipyLUFactorization
from ..dynamics import NodalMaterialCoefficients
from ..output import prepare_output_files
from . import fields as _field_helpers
from . import support as _support_helpers
from .demag.solver import SimulationDemagMixin
from .dynamics import SimulationDynamicsMixin
from .fields import SimulationFieldMixin
from .mesh import SimulationMeshMixin
from .restart import SimulationRestartMixin
from .support import (
    ScalarFieldInput,
    Vector,
    VectorField,
    VectorFieldInput,
    _as_vector3,
    _flatten_materials,
    _normalise_m,
    _scalar_nodal_field,
    _si_unit,
    _vector_nodal_field,
)

if TYPE_CHECKING:
    from simulation.data_writer import DataWriter

_data_writer_class_cache: type[DataWriter] | None = None


def __getattr__(name: str) -> Any:
    """Resolve compatibility exports moved to focused helper modules."""
    try:
        return getattr(_support_helpers, name)
    except AttributeError:
        pass
    try:
        return getattr(_demag_helpers, name)
    except AttributeError:
        pass
    try:
        return getattr(_backend_helpers, name)
    except AttributeError:
        pass
    try:
        return getattr(_field_helpers, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def _data_writer_class() -> type[DataWriter]:
    global _data_writer_class_cache
    if _data_writer_class_cache is None:
        from simulation.data_writer import DataWriter

        _data_writer_class_cache = DataWriter
    return _data_writer_class_cache


class Simulation(
    SimulationMeshMixin,
    SimulationDemagMixin,
    SimulationFieldMixin,
    SimulationDynamicsMixin,
    SimulationRestartMixin,
):
    """Public finite-element simulation API for the Python 3 rewrite.

    The class owns mesh/material bookkeeping, fields, dense first-order FEM/BEM
    demagnetization, and adaptive LLG relaxation for the supported isotropic
    workflows.
    """

    def __init__(
        self,
        name: str | None = None,
        phi_BEM: Any | None = None,
        periodic_bc: Any | None = None,
        do_demag: bool = True,
        do_sl_stt: bool = False,
        config: NmagConfig | None = None,
    ) -> None:
        if phi_BEM is not None:
            raise NotImplementedError("phi_BEM/HLib support is not ported yet.")
        if periodic_bc is not None:
            raise NotImplementedError("Periodic boundary conditions are not ported yet.")
        if do_sl_stt:
            raise NotImplementedError(
                "Spin-transfer torque using the Slonczewski model is not ported yet; "
                "current-density (Zhang-Li) torque is supported."
            )
        self.config = NmagConfig.from_environment() if config is None else config
        self.name = name or self.config.default_name
        self.do_demag = do_demag
        self.do_sl_stt = do_sl_stt
        self.periodic_bc = periodic_bc
        self.clock = SimulationClock()
        data_writer_class = _data_writer_class()
        ndt_filename = self.config.output_directory / f"{self.name}_dat.ndt"
        h5_filename = self.config.output_directory / f"{self.name}_dat.h5"
        prepare_output_files((ndt_filename, h5_filename), self.config.output_policy)
        self.writer = data_writer_class(
            ndt_filename=ndt_filename,
            h5_filename=h5_filename,
            append=self.config.output_policy == "append",
        )
        self.last_save_timings_seconds: dict[str, float] = {}
        self.last_spatial_save_timings_seconds: dict[str, float] = {}
        self.last_probe_timings_seconds: dict[str, float] = {}
        self.last_subfield_average_timings_seconds: dict[str, float] = {}
        self.last_maxangle_timings_seconds: dict[str, float] = {}
        self.last_demag_solve_diagnostics: dict[str, float | int] = {}
        self.last_demag_internal_vectors: dict[str, np.ndarray] = {}
        self._active_subfield_array_timings: dict[str, float] | None = None

        self.mesh: Any | None = None
        self.mesh_unit_length: SI | None = None
        self.region_name_list: list[str] = []
        self.region_name_of_id: dict[int, str] = {}
        self.region_id_of_name: dict[str, int] = {}
        self.mats_of_region_name: dict[str, list[Any]] = {}
        self.mat_of_mat_name: dict[str, Any] = {}
        self.materials: list[Any] = []
        self._fields: dict[str, Any] = {}
        self._demag_cache_token: tuple[int, int] | None = None
        self._demag_dipoles: tuple[np.ndarray, np.ndarray, float] | None = None
        self._demag_nodal_cache: np.ndarray | None = None
        self._demag_cell_cache_token: tuple[int, int] | None = None
        self._demag_cell_field_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._demag_geometry_cache_token: int | None = None
        self._demag_boundary_faces_cache: list[tuple[int, tuple[int, int, int]]] | None = None
        self._demag_bem_cache: tuple[str, np.ndarray, Any] | None = None
        self._demag_fem_geometry_cache: tuple[str, Any, np.ndarray, np.ndarray] | None = None
        self._demag_gauge_factorization_cache: ScipyLUFactorization | None = None
        self._demag_dirichlet_factorization_cache: (
            tuple[np.ndarray, np.ndarray, ScipyLUFactorization] | None
        ) = None
        self._demag_ms_values_cache: tuple[int, np.ndarray] | None = None
        self._demag_volume_charge_scales_cache: tuple[int, np.ndarray] | None = None
        self._nodal_ms_values_cache: tuple[int, np.ndarray] | None = None
        self._demag_aux_cache_token: tuple[int, int] | None = None
        self._demag_phi_cache: np.ndarray | None = None
        self._demag_rho_cache: np.ndarray | None = None
        self._demag_volumes_cache: np.ndarray | None = None
        self._mesh_points_cache: tuple[int, np.ndarray] | None = None
        self._mesh_bounds_cache: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._mesh_edge_cache: tuple[int, np.ndarray] | None = None
        self._simplex_volume_cache: tuple[int, np.ndarray, np.ndarray] | None = None
        self._volume_average_node_weights_cache: tuple[int, int, np.ndarray] | None = None
        self._incident_cell_volume_sums_cache: tuple[int, int, int, np.ndarray] | None = None
        self._exchange_cache_token: tuple[int, int] | None = None
        self._exchange_nodal_cache: np.ndarray | None = None
        self._exchange_spectral_bound_cache: tuple[int, float] | None = None
        self._nodal_material_coefficients_cache: tuple[int, NodalMaterialCoefficients] | None = None
        self._llg_affine_operator_cache: tuple[int, np.ndarray] | None = None
        self._probe_geometry_cache_token: int | None = None
        self._probe_tetrahedral_cache: (
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
            | None
        ) = None
        self._subfield_array_cache: dict[str, np.ndarray] | None = None
        self._subfield_average_cache: dict[tuple[str, str | None], object] | None = None
        self._initialise_dynamics()

    @property
    def id(self) -> int:
        return self.clock.id

    @property
    def stage(self) -> int:
        return self.clock.stage

    @property
    def step(self) -> int:
        return self.clock.step

    @property
    def stage_step(self) -> int:
        return self.clock.stage_step

    @property
    def time(self) -> SI:
        return self.clock.time

    @property
    def stage_time(self) -> SI:
        return self.clock.stage_time

    @property
    def real_time(self) -> SI:
        return self.clock.real_time

    @property
    def last_step_dt(self) -> SI:
        return self.clock.last_step_dt_si

    def load_mesh(
        self,
        filename: str,
        region_names_and_mag_mats: Sequence[tuple[str, Any]],
        unit_length: SI,
        do_reorder: bool = False,
        manual_distribution: Any = None,
    ) -> nmesh.Mesh:
        """Load a mesh and register magnetic materials by region name."""
        if self.mesh is not None:
            raise RuntimeError("Mesh is already present.")

        self.mesh = nmesh.load(
            filename,
            reorder=do_reorder,
            distribute=manual_distribution is None,
        )
        self.mesh_unit_length = unit_length

        scale = unit_length.in_units_of(_si_unit("m"))
        if scale != 1.0:
            self.mesh.scale_node_positions(scale)
        self._fields["pin"] = np.ones(len(self.mesh.points), dtype=float)

        if manual_distribution is not None:
            self.mesh.set_vertex_distribution(manual_distribution)

        self.region_name_list = [name for name, _ in region_names_and_mag_mats]
        self.region_name_of_id = {
            index: name for index, name in enumerate(self.region_name_list, start=1)
        }
        self.region_id_of_name = {name: index for index, name in self.region_name_of_id.items()}

        mesh_region_ids = {int(region_id) for region_id in self.mesh.regions}
        configured_region_ids = set(self.region_name_of_id)
        if mesh_region_ids != configured_region_ids:
            raise ValueError(
                "Mesh material regions do not match the configured region list: "
                f"mesh={sorted(mesh_region_ids)}, configured={sorted(configured_region_ids)}. "
                "Materials must be supplied in ascending mesh-region order."
            )

        self.mats_of_region_name = {}
        self.mat_of_mat_name = {}
        for name, materials in region_names_and_mag_mats:
            mats = cast(list[Any], materials) if isinstance(materials, list) else [materials]
            self.mats_of_region_name[name] = mats
            for material in mats:
                self.mat_of_mat_name[material.name] = material

        self.materials = _flatten_materials(region_names_and_mag_mats)
        self._invalidate_demag(clear_geometry=True)
        self._invalidate_integrator()
        return self.mesh

    def set_m(self, values: Vector | VectorField, subfieldname: str | None = None) -> None:
        """Set the normalised magnetisation field."""
        if subfieldname is not None:
            raise NotImplementedError("Material-specific m subfields are not ported yet.")

        if callable(values):
            if self.mesh is None:
                raise RuntimeError("A mesh must be loaded before setting m from a function.")
            self._fields["m"] = np.asarray(
                [_normalise_m(values(point)) for point in self.mesh.points],
                dtype=float,
            )
            self._invalidate_demag()
            self._invalidate_integrator()
            return

        vector = _normalise_m(values)
        point_count = len(self.mesh.points) if self.mesh is not None else 1
        self._fields["m"] = np.tile(np.asarray(vector, dtype=float), (point_count, 1))
        self._invalidate_demag()
        self._invalidate_integrator()

    def set_H_ext(self, values: Vector, unit: SI | None = None) -> None:
        """Set a homogeneous external magnetic field in SI A/m."""
        self._fields["H_ext"] = np.asarray(_as_vector3(values, unit=unit), dtype=float)
        self._invalidate_demag()
        self._invalidate_integrator()

    def set_pinning(self, values: ScalarFieldInput) -> None:
        """Set the nodal multiplier for the complete magnetisation derivative.

        A value of one leaves a node free and zero fixes its magnetisation.
        Other finite values retain legacy Nmag's local-rate scaling semantics.
        """
        if self.mesh is None:
            raise RuntimeError("A mesh must be loaded before setting pinning.")
        self._fields["pin"] = _scalar_nodal_field(
            values,
            self.mesh.points,
            fieldname="pinning",
        )
        self._invalidate_integrator()

    def set_current_density(
        self,
        values: VectorFieldInput,
        unit: SI | None = None,
    ) -> None:
        """Set the electric-current-density field used by spin-transfer torque."""
        if self.mesh is None:
            raise RuntimeError("A mesh must be loaded before setting current density.")
        self._fields["current_density"] = _vector_nodal_field(
            values,
            self.mesh.points,
            unit=unit,
            expected_unit=_si_unit("A/m^2"),
            fieldname="current_density",
        )
        self._invalidate_integrator()

    def save_data(
        self, fields: str | list[str] | None = None, avoid_same_step: bool = False
    ) -> None:
        """Save averages and optionally stored spatial fields."""
        if avoid_same_step and self.step == self.writer._last_saved_step:
            return
        self.clock.id += 1
        with self._subfield_array_cache_scope(), self._subfield_average_cache_scope():
            self.writer.save(self, fields=fields, avoid_same_step=avoid_same_step)
        self.last_save_timings_seconds = dict(self.writer.last_save_timings_seconds)

    def save_spatial_fields(self, filename: str, fieldnames: list[str]) -> None:
        """Write available spatial fields to a compact HDF5 file."""
        timings: dict[str, float] = {}
        total_started = time.perf_counter()
        try:
            with h5py.File(filename, "a") as h5:
                if self.mesh is not None:
                    started = time.perf_counter()
                    mesh_group = h5.require_group("mesh")
                    if "points" in mesh_group:
                        del mesh_group["points"]
                    mesh_group.create_dataset("points", data=np.asarray(self.mesh.points))
                    timings["mesh_points"] = time.perf_counter() - started

                fields_group = h5.require_group("fields")
                for fieldname in fieldnames:
                    field_started = time.perf_counter()
                    started = time.perf_counter()
                    data = self._subfield_array(fieldname)
                    timings[f"field:{fieldname}:compute"] = time.perf_counter() - started

                    started = time.perf_counter()
                    if fieldname in fields_group:
                        del fields_group[fieldname]
                    fields_group.create_dataset(fieldname, data=data)
                    timings[f"field:{fieldname}:write"] = time.perf_counter() - started
                    timings[f"field:{fieldname}:total"] = time.perf_counter() - field_started
        finally:
            timings["total"] = time.perf_counter() - total_started
            self.last_spatial_save_timings_seconds = dict(sorted(timings.items()))

    def reset_probe_timings(self) -> None:
        """Reset passive probe timing counters used by parity/profile tools."""
        self.last_probe_timings_seconds = {}

    def probe_subfield(
        self,
        subfieldname: str,
        pos: Sequence[float],
        unit: SI | None = None,
    ) -> Any:
        return self.probe_subfield_siv(subfieldname, pos, unit=unit)

    def save_mesh(self, filename: str) -> None:
        if self.mesh is None:
            raise RuntimeError("No mesh has been loaded.")
        nmesh.save(self.mesh, filename)
