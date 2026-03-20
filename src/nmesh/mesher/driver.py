import inspect
import logging
from enum import Enum

log = logging.getLogger(__name__)

LEGACY_CALLBACK_DOCS = {
    "COORDS": "Coordinates of points",
    "LINKS": "Links in the mesh (pairs of point indices)",
    "SIMPLICES": (
        "Simplex info (points-indices,((circumcirc center,cc radius),"
        "(ic center,ic radius),region))"
    ),
    "POINT-BODIES": (
        "Which bodies does the corresponding point belong to (body index list)"
    ),
    "SURFACES": (
        "Surface elements info (points-indices,((circumcirc center,cc radius),"
        "(ic center,ic radius),region))"
    ),
    "REGION-VOLUMES": "Volume for every region",
}


class MeshEngineCommand(Enum):
    DO_STEP = 1
    DO_EXTRACT = 2


class MeshEngineStatus(Enum):
    FINISHED_STEP_LIMIT_REACHED = 1
    FINISHED_FORCE_EQUILIBRIUM_REACHED = 2
    CAN_CONTINUE = 3
    PRODUCED_INTERMEDIATE_MESH = 4


def _looks_like_legacy_payload(mesh):
    return (
        isinstance(mesh, list)
        and all(
            isinstance(entry, tuple)
            and len(entry) == 3
            and isinstance(entry[0], str)
            for entry in mesh
        )
    )


def _mesh_payload(mesh):
    if _looks_like_legacy_payload(mesh):
        return mesh

    return [
        ("COORDS", LEGACY_CALLBACK_DOCS["COORDS"], getattr(mesh, "points", [])),
        ("LINKS", LEGACY_CALLBACK_DOCS["LINKS"], getattr(mesh, "links", [])),
        (
            "SIMPLICES",
            LEGACY_CALLBACK_DOCS["SIMPLICES"],
            getattr(mesh, "simplices", []),
        ),
        (
            "POINT-BODIES",
            LEGACY_CALLBACK_DOCS["POINT-BODIES"],
            getattr(mesh, "point_regions", []),
        ),
        (
            "SURFACES",
            LEGACY_CALLBACK_DOCS["SURFACES"],
            getattr(mesh, "surfaces", []),
        ),
        (
            "REGION-VOLUMES",
            LEGACY_CALLBACK_DOCS["REGION-VOLUMES"],
            getattr(mesh, "region_volumes", []),
        ),
    ]


def _callback_accepts_piece_number(callback):
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return True

    try:
        signature.bind_partial(0, 0, [])
        return True
    except TypeError:
        return False


def _invoke_callback(callback, accepts_piece_number, nr_piece, nr_step, mesh):
    payload = _mesh_payload(mesh)
    if accepts_piece_number:
        callback(nr_piece, nr_step, payload)
    else:
        callback(nr_step, payload)


def do_every_n_steps_driver(nr_steps_per_bunch, callback, engine_func):
    """
    Python port of Mesh.do_every_n_steps_driver using an iterative loop.
    """
    if nr_steps_per_bunch <= 0:
        raise ValueError("nr_steps_per_bunch must be positive")

    nr_step = 0
    status_out = engine_func(MeshEngineCommand.DO_STEP)

    while True:
        log.info("do_every_n_steps_driver [%d]", nr_step)
        status, data = status_out

        if status in (
            MeshEngineStatus.FINISHED_STEP_LIMIT_REACHED,
            MeshEngineStatus.FINISHED_FORCE_EQUILIBRIUM_REACHED,
        ):
            return status_out

        if status == MeshEngineStatus.CAN_CONTINUE:
            cont = data
            if (nr_step % nr_steps_per_bunch != 0) or nr_step == 0:
                nr_step += 1
                status_out = cont(MeshEngineCommand.DO_STEP)
                continue

            log.debug("Scheduling Mesh Extraction!")
            status_out = cont(MeshEngineCommand.DO_EXTRACT)
            continue

        if status == MeshEngineStatus.PRODUCED_INTERMEDIATE_MESH:
            mesh, cont = data
            log.debug("Extracted Mesh!")
            if nr_step != 0:
                callback(nr_step, mesh)
            nr_step += 1
            status_out = cont(MeshEngineCommand.DO_STEP)
            continue

        raise ValueError(f"Unknown mesh engine status: {status}")


def make_mg_gendriver(interval, callback):
    """
    Returns a gendriver compatible with both the legacy piece-aware API and the
    simplified direct-driver test usage.
    """
    accepts_piece_number = _callback_accepts_piece_number(callback)

    def gendriver(piece_or_engine):
        if callable(piece_or_engine):
            return do_every_n_steps_driver(
                interval,
                lambda nr_step, mesh: _invoke_callback(
                    callback,
                    accepts_piece_number,
                    0,
                    nr_step,
                    mesh,
                ),
                piece_or_engine,
            )

        nr_piece = int(piece_or_engine)

        def driver(engine_func):
            return do_every_n_steps_driver(
                interval,
                lambda nr_step, mesh: _invoke_callback(
                    callback,
                    accepts_piece_number,
                    nr_piece,
                    nr_step,
                    mesh,
                ),
                engine_func,
            )

        return driver

    return gendriver
