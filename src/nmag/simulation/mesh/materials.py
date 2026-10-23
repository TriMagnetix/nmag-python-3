from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...demag import _simplex_volumes
from ..support import _si_unit, _simulation_compatibility_binding


def _scalar_token(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    magnitude = getattr(value, "value", None)
    if isinstance(magnitude, (int, float)):
        return (id(value), float(magnitude))
    return id(value)


def _anisotropy_spec_token(spec: Any) -> Any:
    if spec is None:
        return None
    terms = getattr(spec, "terms", None)
    if isinstance(terms, Sequence):
        term_tokens: list[Any] = []
        for term in terms:
            axis = getattr(term, "axis", None)
            axes = getattr(term, "axes", None)
            axis_token: Any = None
            if axis is not None:
                axis_token = tuple(np.asarray(axis, dtype=float).reshape(-1).tolist())
            elif axes is not None:
                axis_token = tuple(
                    tuple(np.asarray(vec, dtype=float).reshape(-1).tolist())
                    for vec in axes
                )
            term_tokens.append(
                (
                    getattr(term, "kind", None),
                    axis_token,
                    _scalar_token(getattr(term, "k1", None)),
                    _scalar_token(getattr(term, "k2", None)),
                    _scalar_token(getattr(term, "k3", None)),
                )
            )
        return (id(spec), tuple(term_tokens))
    return id(spec)


def _regions_token(regions: Any) -> Any:
    if regions is None:
        return None
    if isinstance(regions, tuple):
        return (id(regions), regions)
    if isinstance(regions, list):
        return (id(regions), tuple(regions))
    if isinstance(regions, np.ndarray):
        return (id(regions), regions.shape, regions.tobytes())
    if isinstance(regions, Sequence):
        return (id(regions), tuple(regions))
    return id(regions)


class SimulationMeshMaterialMixin:
    if TYPE_CHECKING:
        _simplex_volume_cache: Any | None
        _volume_average_node_weights_cache: Any | None
        _incident_cell_volume_sums_cache: Any | None
        _demag_ms_values_cache: Any | None
        _demag_volume_charge_scales_cache: Any | None
        _nodal_ms_values_cache: Any | None

        def __getattr__(self, name: str) -> Any: ...

    def _material_state_token(self) -> int:
        mesh = getattr(self, "mesh", None)
        regions = getattr(mesh, "regions", None) if mesh is not None else None
        region_map = tuple(sorted(getattr(self, "region_name_of_id", {}).items()))
        mats_by_region = tuple(
            (
                region_name,
                tuple(id(mat) for mat in mats),
            )
            for region_name, mats in sorted(getattr(self, "mats_of_region_name", {}).items())
        )
        material_tokens = tuple(
            (
                id(mat),
                getattr(mat, "name", None),
                _scalar_token(getattr(mat, "Ms", None)),
                _scalar_token(getattr(mat, "exchange_coupling", None)),
                _scalar_token(getattr(mat, "llg_gamma_G", None)),
                _scalar_token(getattr(mat, "llg_damping", None)),
                _scalar_token(getattr(mat, "llg_normalisationfactor", None)),
                _scalar_token(getattr(mat, "llg_xi", None)),
                _scalar_token(getattr(mat, "llg_polarisation", None)),
                bool(getattr(mat, "do_precession", True)),
                float(getattr(mat, "scale_volume_charges", 1.0)),
                _anisotropy_spec_token(getattr(mat, "anisotropy", None)),
                getattr(mat, "anisotropy_order", None),
            )
            for mat in getattr(self, "materials", ())
        )
        return hash(
            (
                _regions_token(regions),
                region_map,
                mats_by_region,
                material_tokens,
            )
        )

    def _material_region_ids(self, material_name: str) -> np.ndarray:
        region_ids = [
            self.region_id_of_name[region_name]
            for region_name, materials in self.mats_of_region_name.items()
            if any(material.name == material_name for material in materials)
        ]
        if not region_ids:
            raise KeyError(f"Unknown material '{material_name}'.")
        return np.asarray(region_ids, dtype=int)

    def _field_average(
        self,
        data: np.ndarray,
        *,
        mat_name: str | None = None,
    ) -> np.ndarray | float:
        if self.mesh is None:
            return np.mean(data, axis=0)
        if data.ndim == 1:
            return self._scalar_field_average(data, mat_name=mat_name)
        if data.ndim != 2:
            return np.mean(data, axis=0)
        if len(data) > 0 and np.all(data == data[0]):
            return np.asarray(data[0], dtype=float)

        simplices = self._mesh_simplices()
        points = self._mesh_points()
        if (
            len(data) != len(points)
            or simplices.ndim != 2
            or simplices.shape[1] != 4
            or len(simplices) == 0
        ):
            return np.mean(data, axis=0)

        volumes, positive = self._simplex_volume_weights(points, simplices)
        if mat_name is not None:
            regions = np.asarray(self._require_mesh().regions, dtype=int)
            positive = positive & np.isin(regions, self._material_region_ids(mat_name))
        if not np.any(positive):
            return np.mean(data, axis=0)

        weights = (
            self._volume_average_node_weights(points, simplices, len(data))
            if mat_name is None
            else None
        )
        if weights is None:
            cell_values = np.mean(data[simplices[positive]], axis=1)
            return np.sum(cell_values * volumes[positive, np.newaxis], axis=0) / float(
                np.sum(volumes[positive])
            )
        return weights @ data

    def _scalar_field_average(
        self,
        data: np.ndarray,
        *,
        mat_name: str | None = None,
    ) -> float:
        if len(data) > 0 and np.all(data == data[0]):
            return float(data[0])

        simplices = self._mesh_simplices()
        points = self._mesh_points()
        if len(data) != len(points) or simplices.ndim != 2 or simplices.shape[1] != 4:
            return float(np.mean(data))

        volumes, positive = self._simplex_volume_weights(points, simplices)
        if mat_name is not None:
            regions = np.asarray(self._require_mesh().regions, dtype=int)
            positive = positive & np.isin(regions, self._material_region_ids(mat_name))
        if not np.any(positive):
            return float(np.mean(data))

        weights = (
            self._volume_average_node_weights(points, simplices, len(data))
            if mat_name is None
            else None
        )
        if weights is None:
            cell_values = np.mean(data[simplices[positive]], axis=1)
            return float(np.sum(cell_values * volumes[positive]) / float(np.sum(volumes[positive])))
        return float(weights @ data)

    def _simplex_volume_weights(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        token = self._mesh_geometry_token()
        if self._simplex_volume_cache is not None:
            cached_token, volumes, positive = self._simplex_volume_cache
            if cached_token == token and len(volumes) == len(simplices):
                return volumes, positive

        volumes = _simulation_compatibility_binding(
            "_simplex_volumes",
            _simplex_volumes,
        )(points, simplices)
        positive = volumes > 0.0
        volumes.setflags(write=False)
        positive.setflags(write=False)
        self._simplex_volume_cache = (token, volumes, positive)
        return volumes, positive

    def _volume_average_node_weights(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        point_count: int,
    ) -> np.ndarray | None:
        token = self._mesh_geometry_token()
        if self._volume_average_node_weights_cache is not None:
            cached_token, cached_point_count, weights = self._volume_average_node_weights_cache
            if cached_token == token and cached_point_count == point_count:
                return weights

        volumes, positive = self._simplex_volume_weights(points, simplices)
        if not np.any(positive):
            return None

        weights = np.zeros(point_count, dtype=float)
        positive_simplices = simplices[positive]
        positive_volumes = volumes[positive]
        np.add.at(weights, positive_simplices.flatten(), np.repeat(positive_volumes / 4.0, 4))
        total_volume = float(np.sum(positive_volumes))
        if total_volume <= 0.0:
            return None

        weights /= total_volume
        weights.setflags(write=False)
        self._volume_average_node_weights_cache = (token, point_count, weights)
        return weights

    def _incident_cell_volume_sums(
        self,
        points: np.ndarray,
        simplices: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        token = self._mesh_geometry_token()
        point_count = len(points)
        cell_count = len(simplices)
        if self._incident_cell_volume_sums_cache is not None:
            cached_token, cached_point_count, cached_cell_count, weights, cached_volumes = (
                self._incident_cell_volume_sums_cache
            )
            if (
                cached_token == token
                and cached_point_count == point_count
                and cached_cell_count == cell_count
                and (
                    (not volumes.flags.writeable and cached_volumes is volumes)
                    or np.array_equal(cached_volumes, volumes)
                )
            ):
                return weights

        weights = np.zeros(point_count, dtype=float)
        positive = volumes > 0.0
        if np.any(positive):
            positive_simplices = simplices[positive]
            positive_volumes = volumes[positive]
            for local_index in range(4):
                np.add.at(
                    weights,
                    positive_simplices[:, local_index],
                    positive_volumes,
                )

        weights.setflags(write=False)
        cached_vols = volumes if not volumes.flags.writeable else volumes.copy()
        self._incident_cell_volume_sums_cache = (
            token,
            point_count,
            cell_count,
            weights,
            cached_vols,
        )
        return weights

    def _simplex_material(self, region_id: int) -> Any:
        region_name = self.region_name_of_id.get(int(region_id))
        raw_materials: Any = self.mats_of_region_name.get(region_name, []) if region_name else []
        materials = cast(list[Any], raw_materials)
        if not materials:
            raise ValueError(f"Mesh region {region_id} has no configured magnetic material.")
        if len(materials) != 1:
            raise NotImplementedError(
                "Multiple magnetic material subfields in one mesh region are not supported."
            )
        return materials[0]

    def _simplex_material_ms(self, region_id: int) -> float:
        return self._simplex_material(region_id).Ms.in_units_of(_si_unit("A/m"))

    def _simplex_material_ms_values(self, regions: Sequence[int]) -> np.ndarray:
        token = self._mesh_geometry_token()
        material_token = self._material_state_token()
        self._ensure_demag_geometry_cache_token(token)
        mesh = getattr(self, "mesh", None)
        mesh_regions = getattr(mesh, "regions", None) if mesh is not None else None
        if self._demag_ms_values_cache is not None:
            cached_token, cached_material_token, cached_regions, cached_region_ids, cached_values = (
                self._demag_ms_values_cache
            )
            if (
                cached_token == token
                and cached_material_token == material_token
                and len(cached_values) == len(regions)
                and (
                    (cached_regions is regions and regions is mesh_regions)
                    or (isinstance(regions, tuple) and cached_regions is regions)
                    or (
                        isinstance(regions, np.ndarray)
                        and not regions.flags.writeable
                        and cached_regions is regions
                    )
                    or np.array_equal(cached_region_ids, np.asarray(regions, dtype=int))
                )
            ):
                return cached_values

        region_ids = np.asarray(regions, dtype=int)
        if region_ids.size == 0:
            ms_values = np.zeros(0, dtype=float)
            ms_values.setflags(write=False)
            self._demag_ms_values_cache = (
                token,
                material_token,
                regions,
                region_ids,
                ms_values,
            )
            return ms_values

        first_region = int(region_ids[0])
        if np.all(region_ids == first_region):
            ms_values = np.full(
                len(region_ids),
                self._simplex_material_ms(first_region),
                dtype=float,
            )
            ms_values.setflags(write=False)
            self._demag_ms_values_cache = (
                token,
                material_token,
                regions,
                region_ids,
                ms_values,
            )
            return ms_values

        unique_region_ids, inverse = np.unique(region_ids, return_inverse=True)
        ms_lookup = np.asarray(
            [self._simplex_material_ms(int(region)) for region in unique_region_ids],
            dtype=float,
        )
        ms_values = ms_lookup[inverse]
        ms_values.setflags(write=False)
        self._demag_ms_values_cache = (
            token,
            material_token,
            regions,
            region_ids,
            ms_values,
        )
        return ms_values

    def _simplex_volume_charge_scales(self, regions: Sequence[int]) -> np.ndarray:
        token = self._mesh_geometry_token()
        material_token = self._material_state_token()
        self._ensure_demag_geometry_cache_token(token)
        mesh = getattr(self, "mesh", None)
        mesh_regions = getattr(mesh, "regions", None) if mesh is not None else None
        if self._demag_volume_charge_scales_cache is not None:
            cached_token, cached_material_token, cached_regions, cached_region_ids, cached_values = (
                self._demag_volume_charge_scales_cache
            )
            if (
                cached_token == token
                and cached_material_token == material_token
                and len(cached_values) == len(regions)
                and (
                    (cached_regions is regions and regions is mesh_regions)
                    or (isinstance(regions, tuple) and cached_regions is regions)
                    or (
                        isinstance(regions, np.ndarray)
                        and not regions.flags.writeable
                        and cached_regions is regions
                    )
                    or np.array_equal(cached_region_ids, np.asarray(regions, dtype=int))
                )
            ):
                return cached_values

        region_ids = np.asarray(regions, dtype=int)
        if region_ids.size == 0:
            scales = np.zeros(0, dtype=float)
        else:
            unique_region_ids, inverse = np.unique(region_ids, return_inverse=True)
            scale_lookup = np.asarray(
                [
                    float(self._simplex_material(int(region)).scale_volume_charges)
                    for region in unique_region_ids
                ],
                dtype=float,
            )
            scales = scale_lookup[inverse]

        scales.setflags(write=False)
        self._demag_volume_charge_scales_cache = (
            token,
            material_token,
            regions,
            region_ids,
            scales,
        )
        return scales

    def _nodal_ms_values(self) -> np.ndarray:
        if self.mesh is None:
            raise RuntimeError("A mesh must be loaded before using material fields.")

        token = (self._mesh_geometry_token(), self._material_state_token())
        if self._nodal_ms_values_cache is not None:
            cached_token, cached_values = self._nodal_ms_values_cache
            if cached_token == token:
                return cached_values

        points = self._mesh_points()
        simplices = self._mesh_simplices()
        if simplices.size == 0 or simplices.ndim != 2 or simplices.shape[1] != 4:
            default_ms = (
                self.materials[0].Ms.in_units_of(_si_unit("A/m")) if self.materials else 0.0
            )
            nodal_ms = np.full(len(points), default_ms, dtype=float)
            nodal_ms.setflags(write=False)
            self._nodal_ms_values_cache = (token, nodal_ms)
            return nodal_ms

        regions = self.mesh.regions if self.mesh.regions else [1] * len(simplices)
        simplex_ms = self._simplex_material_ms_values(regions)
        volumes, positive = self._simplex_volume_weights(points, simplices)
        nodal_ms = np.zeros(len(points), dtype=float)
        weights = self._incident_cell_volume_sums(points, simplices, volumes)
        if np.any(positive):
            positive_simplices = simplices[positive]
            weighted_ms = simplex_ms[positive] * volumes[positive]
            for local_index in range(4):
                np.add.at(nodal_ms, positive_simplices[:, local_index], weighted_ms)

        present = weights > 0.0
        nodal_ms[present] /= weights[present]
        if np.any(~present):
            nodal_ms[~present] = simplex_ms[positive][0] if np.any(positive) else 0.0
        nodal_ms.setflags(write=False)
        self._nodal_ms_values_cache = (token, nodal_ms)
        return nodal_ms
