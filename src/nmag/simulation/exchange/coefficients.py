from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from si.physical import SI

from ...dynamics import DOP853_EXCHANGE_RESOLUTION_RADIUS, NodalMaterialCoefficients
from ..support import MU0, _si_dimensionless, _si_unit


class SimulationExchangeCoefficientMixin:
    if TYPE_CHECKING:
        _exchange_spectral_bound_cache: tuple[int, float] | None
        _nodal_material_coefficients_cache: tuple[int, NodalMaterialCoefficients] | None

        def __getattr__(self, name: str) -> Any: ...

    @staticmethod
    def _material_dynamic_coefficients(
        material: Any,
    ) -> tuple[float, float, float, float, float, float]:
        exchange = material.exchange_coupling.in_units_of(_si_unit("J/m"))
        ms = material.Ms.in_units_of(_si_unit("A/m"))
        exchange_prefactor = 0.0 if ms == 0.0 else 2.0 * exchange / (MU0 * ms)
        gamma = material.llg_gamma_G.in_units_of(_si_unit("m/A/s"))
        damping = (
            material.llg_damping.in_units_of(_si_dimensionless())
            if isinstance(material.llg_damping, SI)
            else float(material.llg_damping)
        )
        gilbert_to_ll = 1.0 / (1.0 + damping * damping)
        precession = -gamma * gilbert_to_ll
        damping_coeff = precession * damping
        if material.do_precession is False:
            precession = 0.0
        normalisation = material.llg_normalisationfactor.in_units_of(_si_unit("1/s"))
        stt_adiabatic = material.su_llg_stt_adiab.in_units_of(_si_unit("m^3/A/s"))
        stt_nonadiabatic = material.su_llg_stt_nadiab.in_units_of(_si_unit("m^3/A/s"))
        return (
            exchange_prefactor,
            precession,
            damping_coeff,
            normalisation,
            stt_adiabatic,
            stt_nonadiabatic,
        )

    def _nodal_material_coefficients(self) -> NodalMaterialCoefficients:
        token = self._mesh_geometry_token()
        cached = self._nodal_material_coefficients_cache
        if cached is not None and cached[0] == token:
            return cached[1]

        point_count = len(self._mesh_points())
        simplices = np.asarray(self._require_mesh().simplices, dtype=int)
        regions = np.asarray(self._require_mesh().regions or [1] * len(simplices), dtype=int)
        if len(regions) != len(simplices):
            raise ValueError("Mesh regions must contain one entry per simplex.")

        region_coefficients = {
            int(region): self._material_dynamic_coefficients(self._simplex_material(int(region)))
            for region in np.unique(regions)
        }
        if not region_coefficients:
            defaults = (
                self._material_dynamic_coefficients(self.materials[0])
                if self.materials
                else (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )
            arrays = tuple(np.full(point_count, value, dtype=float) for value in defaults)
        else:
            nodal_regions = np.full(point_count, -1, dtype=int)
            for region, coefficients in region_coefficients.items():
                region_nodes = np.unique(simplices[regions == region].ravel())
                assigned = nodal_regions[region_nodes]
                for previous_region in np.unique(assigned[assigned >= 0]):
                    previous = region_coefficients[int(previous_region)]
                    if not np.allclose(previous, coefficients, rtol=1.0e-12, atol=0.0):
                        conflict_nodes = region_nodes[assigned == previous_region]
                        raise NotImplementedError(
                            "A mesh node shared by regions with different exchange or LLG "
                            f"coefficients requires material-specific magnetisation DOFs; node "
                            f"{int(conflict_nodes[0])} belongs to regions {int(previous_region)} "
                            f"and {region}."
                        )
                nodal_regions[region_nodes[assigned < 0]] = region

            first_region = next(iter(region_coefficients))
            nodal_regions[nodal_regions < 0] = first_region
            arrays = tuple(
                np.asarray(
                    [region_coefficients[int(region)][component] for region in nodal_regions],
                    dtype=float,
                )
                for component in range(6)
            )

        coefficients = NodalMaterialCoefficients(*arrays)
        self._nodal_material_coefficients_cache = (token, coefficients)
        return coefficients

    def _exchange_prefactor(self) -> float:
        if not self.materials:
            return 0.0

        prefactors = [
            self._material_dynamic_coefficients(material)[0] for material in self.materials
        ]

        first = prefactors[0]
        if any(
            abs(prefactor - first) > 1.0e-12 * max(1.0, abs(first)) for prefactor in prefactors[1:]
        ):
            raise NotImplementedError("The exchange MVP currently supports one exchange prefactor.")
        return float(first)

    def _llg_coefficients(self) -> tuple[float, float, float]:
        if not self.materials:
            return 0.0, 0.0, 0.0

        coefficients = [
            self._material_dynamic_coefficients(material)[1:] for material in self.materials
        ]

        first = coefficients[0]
        for coeffs in coefficients[1:]:
            if any(
                abs(value - reference) > 1.0e-12 * max(1.0, abs(reference))
                for value, reference in zip(coeffs, first, strict=True)
            ):
                raise NotImplementedError(
                    "The dmdt MVP currently supports one LLG coefficient set."
                )
        return (float(first[0]), float(first[1]), float(first[2]))

    def _exchange_laplacian_spectral_bound(self) -> float:
        """Return a cached upper bound for the lumped FEM Laplacian spectrum."""
        token = self._mesh_geometry_token()
        if self._exchange_spectral_bound_cache is not None:
            cached_token, cached_bound = self._exchange_spectral_bound_cache
            if cached_token == token:
                return cached_bound

        points = self._mesh_points()
        simplices = np.asarray(self._require_mesh().simplices, dtype=int)
        if simplices.size == 0:
            bound = 0.0
        else:
            stiffness, _gradients_by_cell, volumes = self._demag_fem_geometry_for_mesh(
                points, simplices
            )
            lumped_volumes = self._incident_cell_volume_sums(points, simplices, volumes) / 4.0
            present = lumped_volumes > 0.0
            if not np.any(present):
                bound = 0.0
            else:
                absolute_row_sums = np.asarray(
                    np.abs(stiffness[present]).sum(axis=1),
                ).ravel()
                bound = float(np.max(absolute_row_sums / lumped_volumes[present]))

        self._exchange_spectral_bound_cache = (token, bound)
        return bound

    def _exchange_explicit_step_limit_seconds(self) -> float:
        """Resolve stiff exchange modes accurately with the explicit integrator."""
        spectral_bound = self._exchange_laplacian_spectral_bound()
        coefficients = self._nodal_material_coefficients()
        dynamic_coefficients = np.hypot(coefficients.precession, coefficients.damping)
        maximum_rate = (
            float(np.max(np.abs(coefficients.exchange_prefactor) * dynamic_coefficients))
            * spectral_bound
        )
        if maximum_rate <= 0.0:
            return float("inf")
        return DOP853_EXCHANGE_RESOLUTION_RADIUS / maximum_rate
