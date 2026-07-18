"""Map material anisotropy models onto geometric mesh nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from anisotropy import AnisotropyModel, PredefinedAnisotropy, anisotropy_signature_values

from ..support import _si_unit


@dataclass(frozen=True, slots=True)
class NodalAnisotropyGroup:
    model: AnisotropyModel
    order: int | None
    nodes: np.ndarray
    saturation_magnetisation: float


class SimulationAnisotropyMaterialMixin:
    if TYPE_CHECKING:
        _nodal_anisotropy_groups_cache: tuple[int, tuple[NodalAnisotropyGroup, ...]] | None

        def __getattr__(self, name: str) -> Any: ...

    def _nodal_anisotropy_groups(self) -> tuple[NodalAnisotropyGroup, ...]:
        token = self._mesh_geometry_token()
        cached = self._nodal_anisotropy_groups_cache
        if cached is not None and cached[0] == token:
            return cached[1]

        point_count = len(self._mesh_points())
        simplices = np.asarray(self._require_mesh().simplices, dtype=np.int64)
        regions = np.asarray(
            self._require_mesh().regions or [1] * len(simplices),
            dtype=np.int64,
        )
        assignments: list[tuple[np.ndarray, Any, float] | None] = [None] * point_count

        if len(simplices) == 0:
            if self.materials:
                material = self.materials[0]
                signature = anisotropy_signature_values(
                    material.anisotropy,
                    material.anisotropy_order,
                )
                ms = float(material.Ms.in_units_of(_si_unit("A/m")))
                assignments = [(signature, material, ms)] * point_count
        else:
            if len(regions) != len(simplices):
                raise ValueError("Mesh regions must contain one entry per simplex.")
            for region in np.unique(regions):
                material = self._simplex_material(int(region))
                signature = anisotropy_signature_values(
                    material.anisotropy,
                    material.anisotropy_order,
                )
                ms = float(material.Ms.in_units_of(_si_unit("A/m")))
                nodes = np.unique(simplices[regions == region].ravel())
                for node in nodes:
                    previous = assignments[int(node)]
                    if previous is not None:
                        previous_signature, _previous_material, previous_ms = previous
                        if not self._anisotropy_materials_compatible(
                            _previous_material,
                            material,
                            previous_signature,
                            signature,
                        ) or not np.isclose(previous_ms, ms, rtol=1.0e-12, atol=0.0):
                            raise NotImplementedError(
                                "A mesh node shared by materials with different anisotropy or Ms "
                                "requires material-specific magnetisation DOFs; node "
                                f"{int(node)} includes region {int(region)}."
                            )
                    else:
                        assignments[int(node)] = (signature, material, ms)

        if self.materials:
            fallback = self.materials[0]
            fallback_signature = anisotropy_signature_values(
                fallback.anisotropy,
                fallback.anisotropy_order,
            )
            fallback_ms = float(fallback.Ms.in_units_of(_si_unit("A/m")))
            assignments = [
                assignment or (fallback_signature, fallback, fallback_ms)
                for assignment in assignments
            ]

        grouped_nodes: dict[bytes, list[int]] = {}
        grouped_materials: dict[bytes, tuple[Any, float]] = {}
        for node, assignment in enumerate(assignments):
            if assignment is None:
                continue
            signature, material, ms = assignment
            key = signature.tobytes() + np.float64(ms).tobytes()
            grouped_nodes.setdefault(key, []).append(node)
            grouped_materials[key] = (material, ms)

        groups = tuple(
            NodalAnisotropyGroup(
                model=grouped_materials[key][0].anisotropy,
                order=grouped_materials[key][0].anisotropy_order,
                nodes=np.asarray(nodes, dtype=np.int64),
                saturation_magnetisation=grouped_materials[key][1],
            )
            for key, nodes in grouped_nodes.items()
        )
        self._nodal_anisotropy_groups_cache = (token, groups)
        return groups

    @staticmethod
    def _anisotropy_materials_compatible(
        first: Any,
        second: Any,
        first_signature: np.ndarray,
        second_signature: np.ndarray,
    ) -> bool:
        first_model = first.anisotropy
        second_model = second.anisotropy
        if first_model is None or second_model is None:
            return first_model is second_model
        if isinstance(first_model, PredefinedAnisotropy) and isinstance(
            second_model,
            PredefinedAnisotropy,
        ):
            return bool(
                np.allclose(
                    first_signature,
                    second_signature,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )
        return first_model is second_model and first.anisotropy_order == second.anisotropy_order
