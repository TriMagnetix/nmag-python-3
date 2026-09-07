"""Construction of meshes from geometry and point/simplicial inputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .backend import backend
from .mesh_model import MeshBase, Point, Simplex

if TYPE_CHECKING:
    from .geometry import Body, MeshObject
    from .mesher.meshing_parameters import MeshingParameters

MeshCallback = Callable[[int, int, object], None]


def _as_float_points(points: Sequence[Sequence[float]] | None) -> list[Point]:
    return [list(map(float, point)) for point in (points or [])]


def _as_int_simplices(simplices: Sequence[Sequence[int]] | None) -> list[Simplex]:
    return [list(map(int, simplex)) for simplex in (simplices or [])]


def _as_region_ids(regions: Sequence[int] | None) -> list[int]:
    return [int(region) for region in (regions or [])]


def _normalise_periodic(periodic: Sequence[bool] | Sequence[float] | None, dim: int) -> list[float]:
    if not periodic:
        return [0.0] * dim
    return [1.0 if bool(value) else 0.0 for value in periodic]


def get_default_meshing_parameters() -> MeshingParameters:
    """Returns default meshing parameters."""
    from .mesher.meshing_parameters import MeshingParameters

    return MeshingParameters()


# --- Loading Utilities ---


class Mesh(MeshBase):
    """Generate a simplex mesh from implicit geometric objects.

    Args:
        bounding_box: Lower and upper coordinate corners enclosing the model.
        objects: Geometry primitives or boolean combinations to mesh.
        a0: Target initial point spacing in mesh-coordinate units.
        density: Optional legacy density expression.
        periodic: Periodic coordinate selectors for bounding-box meshing.
        fixed_points: Additional points held fixed during relaxation.
        mobile_points: Additional mobile seed points.
        simply_points: Additional simple seed points.
        callback: Mesher progress callback and interval.
        mesh_bounding_box: Mesh the complete bounding box when true.
        meshing_parameters: Explicit mesher controls.
        cache_name: Optional compatibility cache label.
        hints: Mesh/object hint pairs.
        **kwargs: Named meshing-parameter overrides.

    Raises:
        ValueError: If the bounding box or object configuration is invalid.
    """

    def __init__(
        self,
        bounding_box: Sequence[Sequence[float]] | None,
        objects: Sequence[MeshObject] | None = None,
        a0: float = 1.0,
        density: str = "",
        periodic: Sequence[bool] | Sequence[float] | None = None,
        fixed_points: Sequence[Sequence[float]] | None = None,
        mobile_points: Sequence[Sequence[float]] | None = None,
        simply_points: Sequence[Sequence[float]] | None = None,
        callback: tuple[MeshCallback, int] | None = None,
        mesh_bounding_box: bool = False,
        meshing_parameters: MeshingParameters | None = None,
        cache_name: str = "",
        hints: Sequence[tuple[MeshBase, MeshObject]] | None = None,
        **kwargs: Any,
    ) -> None:
        if bounding_box is None:
            raise ValueError("Bounding box must be provided.")

        object_list = list(objects or [])
        hint_list = list(hints or [])
        bb = _as_float_points(bounding_box)
        dim = len(bb[0])
        mesh_ext = 1 if mesh_bounding_box else 0

        self.bounding_box = bb
        self.mesh_exterior = mesh_ext

        if not object_list and not mesh_bounding_box:
            raise ValueError("No objects to mesh and bounding box meshing disabled.")

        if periodic and not mesh_bounding_box and any(periodic):
            raise ValueError("Can only produce periodic meshes when meshing the bounding box.")

        params = meshing_parameters or get_default_meshing_parameters()
        self.meshing_parameters = params
        for k, v in kwargs.items():
            params[k] = v

        obj_bodies: list[Body] = []
        self.obj = obj_bodies
        self.cache_name = cache_name
        self.density = density
        self._fixed_points = _as_float_points(fixed_points)
        self._mobile_points = _as_float_points(mobile_points)
        self._simply_points = _as_float_points(simply_points)

        for obj in object_list:
            obj_bodies.append(obj._require_body())
            self._fixed_points.extend(obj.fixed_points)
            self._mobile_points.extend(obj.mobile_points)

        resolved_hints = [
            [hint_mesh.raw_mesh, hint_object._require_body()]
            for hint_mesh, hint_object in hint_list
        ]

        from .mesher.driver import make_mg_gendriver

        periodic_floats = _normalise_periodic(periodic, dim)
        self.periodic = periodic_floats

        def no_callback(_piece: int, _step: int, _mesh: object) -> None:
            return None

        cb_func, cb_interval = callback if callback else (no_callback, 1_000_000)
        self.fun_driver = cb_func
        self.driver = make_mg_gendriver(cb_interval, cb_func)
        self.mesher_config = backend.copy_mesher_defaults(backend.mesher_defaults)
        params.apply_to_mesher(self.mesher_config, dim)

        raw = backend.mesh_bodies_raw(
            self.driver,
            self.mesher_config,
            bb[0],
            bb[1],
            mesh_ext,
            obj_bodies,
            float(a0),
            density,
            self._fixed_points,
            self._mobile_points,
            self._simply_points,
            periodic_floats,
            cache_name,
            resolved_hints,
        )

        super().__init__(raw)

    def default_fun(self, nr_piece: int, n: int, mesh: object) -> None:
        """Default callback function."""
        pass

    def extended_fun_driver(self, nr_piece: int, iteration_nr: int, mesh: object) -> None:
        """Extended driver callback."""
        if hasattr(self, "fun_driver"):
            self.fun_driver(nr_piece, iteration_nr, mesh)

    def fixed_points(self, points: Sequence[Sequence[float]] | None) -> None:
        """Adds fixed points to the mesh configuration."""
        if points:
            self._fixed_points.extend(_as_float_points(points))

    def mobile_points(self, points: Sequence[Sequence[float]] | None) -> None:
        """Adds mobile points to the mesh configuration."""
        if points:
            self._mobile_points.extend(_as_float_points(points))

    def simply_points(self, points: Sequence[Sequence[float]] | None) -> None:
        """Adds simply points to the mesh configuration."""
        if points:
            self._simply_points.extend(_as_float_points(points))
