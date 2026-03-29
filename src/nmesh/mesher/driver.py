import logging
from enum import Enum
from typing import Any, Callable, Tuple, Union, overload

log = logging.getLogger(__name__)


class MeshEngineCommand(Enum):
    """Commands that can be sent to the mesh engine."""
    DO_STEP = 1  # Execute one relaxation step
    DO_EXTRACT = 2  # Extract intermediate mesh for callback


class MeshEngineStatus(Enum):
    """Status returned by the mesh engine."""
    FINISHED_STEP_LIMIT_REACHED = 1  # Maximum iteration steps reached
    FINISHED_FORCE_EQUILIBRIUM_REACHED = 2  # Forces converged to equilibrium
    CAN_CONTINUE = 3  # Engine can continue, provides continuation function
    PRODUCED_INTERMEDIATE_MESH = 4  # Intermediate mesh extracted for callback


# Type aliases for improved readability
EngineFunc = Callable[[MeshEngineCommand], Tuple[MeshEngineStatus, Any]]
Callback = Callable[[int, Any], None]


def do_every_n_steps_driver(
    nr_steps_per_bunch: int, callback: Callback, engine_func: EngineFunc
) -> Tuple[MeshEngineStatus, Any]:
    """
    Python port of Mesh.do_every_n_steps_driver using an iterative loop.

    Drives the meshing engine, invoking a callback at regular step intervals.
    Note: The callback is ONLY invoked at multiples of nr_steps_per_bunch during
    the meshing process. When the engine finishes (step limit or equilibrium reached),
    NO final callback is made - the final mesh state is stored in the engine's
    internal state and returned via the status tuple.

    Args:
        nr_steps_per_bunch: Number of steps between callback invocations
        callback: Function called as callback(step_number, mesh)
        engine_func: Mesh engine function that accepts commands and returns status

    Returns:
        Final (status, data) tuple when meshing completes

    Raises:
        ValueError: If nr_steps_per_bunch <= 0 or unknown engine status encountered
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


def make_mg_gendriver(
    interval: int, callback: Callable[[int, int, Any], None]
) -> Callable[[Union[int, EngineFunc]], Union[Tuple[MeshEngineStatus, Any], Callable[[EngineFunc], Tuple[MeshEngineStatus, Any]]]]:
    """
    Returns a gendriver using the callback signature for multi-geometry meshing.

    Creates a driver factory that can handle both single-body and multi-body
    meshing scenarios. When meshing multiple geometric pieces, each piece can
    have its callbacks distinguished by piece number.

    Args:
        interval: Number of steps between callback invocations
        callback: Function called as callback(piece_number, iteration_step, mesh)

    Returns:
        A gendriver function that can be called with either:
        - An engine function (for single-body, piece_number defaults to 0)
        - A piece number (returns a driver for that specific piece)

    Example:
        driver = make_mg_gendriver(100, my_callback)
        # Single body:
        driver(engine_func)
        # Multi-body:
        driver(0)(engine_func_piece_0)
        driver(1)(engine_func_piece_1)
    """
    def gendriver(
        piece_or_engine: Union[int, EngineFunc]
    ) -> Union[Tuple[MeshEngineStatus, Any], Callable[[EngineFunc], Tuple[MeshEngineStatus, Any]]]:
        if callable(piece_or_engine):
            # Type checker needs help here - we know it's an EngineFunc
            engine_func: EngineFunc = piece_or_engine
            return do_every_n_steps_driver(
                interval,
                lambda nr_step, mesh: callback(0, nr_step, mesh),
                engine_func,
            )

        nr_piece = int(piece_or_engine)

        def driver(engine_func: EngineFunc) -> Tuple[MeshEngineStatus, Any]:
            return do_every_n_steps_driver(
                interval,
                lambda nr_step, mesh: callback(nr_piece, nr_step, mesh),
                engine_func,
            )

        return driver

    return gendriver
