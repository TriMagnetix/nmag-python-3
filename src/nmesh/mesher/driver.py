import logging
from enum import Enum

log = logging.getLogger(__name__)


class MeshEngineCommand(Enum):
    DO_STEP = 1
    DO_EXTRACT = 2


class MeshEngineStatus(Enum):
    FINISHED_STEP_LIMIT_REACHED = 1
    FINISHED_FORCE_EQUILIBRIUM_REACHED = 2
    CAN_CONTINUE = 3
    PRODUCED_INTERMEDIATE_MESH = 4


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
    Returns a gendriver using the callback signature:
    callback(piece_number, iteration_step, mesh)
    """
    def gendriver(piece_or_engine):
        if callable(piece_or_engine):
            return do_every_n_steps_driver(
                interval,
                lambda nr_step, mesh: callback(0, nr_step, mesh),
                piece_or_engine,
            )

        nr_piece = int(piece_or_engine)

        def driver(engine_func):
            return do_every_n_steps_driver(
                interval,
                lambda nr_step, mesh: callback(nr_piece, nr_step, mesh),
                engine_func,
            )

        return driver

    return gendriver
