"""Evaluate anisotropy energy density and effective field."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from anisotropy import evaluate_energy_and_gradient

from ..support import MU0
from .materials import SimulationAnisotropyMaterialMixin


class SimulationAnisotropyMixin(
    SimulationAnisotropyMaterialMixin,
):
    if TYPE_CHECKING:
        _anisotropy_fields_cache: tuple[tuple[int, int], np.ndarray, np.ndarray] | None

        def __getattr__(self, name: str) -> Any: ...

    def _get_anisotropy_fields(self) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(self._fields["m"], dtype=np.float64)
        token = (id(self._require_mesh().raw_mesh), id(self._fields["m"]))
        cached = self._anisotropy_fields_cache
        if cached is not None and cached[0] == token:
            return cached[1], cached[2]

        field = np.zeros_like(m)
        energy = np.zeros(len(m), dtype=np.float64)
        for group in self._nodal_anisotropy_groups():
            if group.model is None:
                continue
            nodes = group.nodes
            group_m = m[nodes]
            group_energy, gradient = evaluate_energy_and_gradient(group.model, group_m)
            energy[nodes] = group_energy
            if group.saturation_magnetisation != 0.0:
                raw_field = -gradient / (MU0 * group.saturation_magnetisation)
                norm_squared = np.einsum("ij,ij->i", group_m, group_m)
                radial_scale = np.divide(
                    np.einsum("ij,ij->i", raw_field, group_m),
                    norm_squared,
                    out=np.zeros(len(group_m), dtype=np.float64),
                    where=norm_squared > 0.0,
                )
                field[nodes] = raw_field - radial_scale[:, np.newaxis] * group_m

        if not np.all(np.isfinite(field)) or not np.all(np.isfinite(energy)):
            raise FloatingPointError("Anisotropy evaluation produced non-finite fields.")
        self._anisotropy_fields_cache = (token, field, energy)
        return field, energy

    def _anisotropy_is_zero_by_construction(self) -> bool:
        return all(getattr(material, "anisotropy", None) is None for material in self.materials)
