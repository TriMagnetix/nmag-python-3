import pytest
from nmesh.mesher.driver import (
    make_mg_gendriver,
    MeshEngineStatus,
    MeshEngineCommand,
    do_every_n_steps_driver,
    _callback_accepts_piece_number,
)


def make_engine(limit, mesh_factory):
    class Engine:
        def __init__(self):
            self.step = 0

        def run(self, cmd):
            if cmd == MeshEngineCommand.DO_STEP:
                self.step += 1
                if self.step > limit:
                    return MeshEngineStatus.FINISHED_STEP_LIMIT_REACHED, None
                return MeshEngineStatus.CAN_CONTINUE, self.run

            if cmd == MeshEngineCommand.DO_EXTRACT:
                return MeshEngineStatus.PRODUCED_INTERMEDIATE_MESH, (
                    mesh_factory(),
                    self.run,
                )

            raise AssertionError(f"Unexpected command: {cmd}")

    return Engine().run


def test_driver_cadence():
    steps = []
    payloads = []

    def callback(step, mesh):
        steps.append(step)
        payloads.append(mesh)

    def mesh_factory():
        return [
            ("COORDS", "Coordinates", []),
            ("LINKS", "Links", []),
            ("SIMPLICES", "Simplex info", []),
        ]

    driver = make_mg_gendriver(3, callback)
    driver(make_engine(10, mesh_factory))

    assert steps == [3, 6, 9]
    for payload in payloads:
        assert isinstance(payload, list)
        assert payload[0][0] == "COORDS"


def test_driver_supports_legacy_callback_signature_and_piece_numbers():
    calls = []

    class MockMesh:
        points = [[0.0, 0.0, 0.0]]
        links = [(0, 1)]
        simplices = [([0, 1, 2, 3], (([], 0.0), ([], 0.0), 1))]
        point_regions = [[1]]
        surfaces = [([0, 1, 2], (([], 0.0), ([], 0.0), 1))]
        region_volumes = [1.0]

    def callback(piece, step, mesh):
        calls.append((piece, step, mesh))

    driver = make_mg_gendriver(2, callback)
    driver(7)(make_engine(5, MockMesh))

    assert [call[:2] for call in calls] == [(7, 2), (7, 4)]
    payload = calls[0][2]
    assert [tag for tag, _, _ in payload] == [
        "COORDS",
        "LINKS",
        "SIMPLICES",
        "POINT-BODIES",
        "SURFACES",
        "REGION-VOLUMES",
    ]
    assert payload[0][2] == [[0.0, 0.0, 0.0]]


def test_driver_handles_large_step_counts_without_recursion():
    steps = []

    def callback(step, mesh):
        steps.append(step)

    driver = make_mg_gendriver(100, callback)
    status, _ = driver(make_engine(1100, lambda: []))

    assert status == MeshEngineStatus.FINISHED_STEP_LIMIT_REACHED
    assert steps[0] == 100
    assert steps[-1] == 1000
    assert len(steps) == 10


def test_do_every_n_steps_driver_invalid_interval():
    """Test that invalid interval raises ValueError."""
    def callback(step, mesh):
        pass

    engine = make_engine(10, lambda: [])

    with pytest.raises(ValueError, match="nr_steps_per_bunch must be positive"):
        do_every_n_steps_driver(0, callback, engine)

    with pytest.raises(ValueError, match="nr_steps_per_bunch must be positive"):
        do_every_n_steps_driver(-5, callback, engine)


def test_callback_accepts_piece_number_detection():
    """Test signature detection for callbacks."""

    # Two-argument callback (no piece number)
    def callback_2_args(step, mesh):
        pass

    assert not _callback_accepts_piece_number(callback_2_args)

    # Three-argument callback (with piece number)
    def callback_3_args(piece, step, mesh):
        pass

    assert _callback_accepts_piece_number(callback_3_args)

    # Lambda with 2 args
    assert not _callback_accepts_piece_number(lambda step, mesh: None)

    # Lambda with 3 args
    assert _callback_accepts_piece_number(lambda piece, step, mesh: None)


def test_callback_accepts_piece_number_with_non_inspectable():
    """Test that non-inspectable callables default to True."""

    # Built-in functions can't be inspected
    class NonInspectable:
        def __call__(self, *args):
            pass

    obj = NonInspectable()
    # Should default to True when inspection fails
    result = _callback_accepts_piece_number(obj)
    assert isinstance(result, bool)


def test_driver_stops_on_force_equilibrium():
    """Test that driver properly handles force equilibrium status."""
    steps = []

    def callback(step, mesh):
        steps.append(step)

    def engine_that_reaches_equilibrium(cmd):
        if cmd == MeshEngineCommand.DO_STEP:
            return MeshEngineStatus.FINISHED_FORCE_EQUILIBRIUM_REACHED, None
        return MeshEngineStatus.CAN_CONTINUE, engine_that_reaches_equilibrium

    driver = make_mg_gendriver(5, callback)
    status, _ = driver(engine_that_reaches_equilibrium)

    assert status == MeshEngineStatus.FINISHED_FORCE_EQUILIBRIUM_REACHED
    assert len(steps) == 0  # No callbacks should have been invoked


def test_driver_handles_immediate_extract():
    """Test driver when engine immediately produces mesh."""
    meshes = []

    def callback(step, mesh):
        meshes.append(mesh)

    class ImmediateEngine:
        def __init__(self):
            self.extracted = False

        def run(self, cmd):
            if cmd == MeshEngineCommand.DO_STEP and not self.extracted:
                return MeshEngineStatus.CAN_CONTINUE, self.run
            if cmd == MeshEngineCommand.DO_EXTRACT:
                self.extracted = True
                return MeshEngineStatus.PRODUCED_INTERMEDIATE_MESH, (
                    [("COORDS", "desc", [])],
                    self.run,
                )
            return MeshEngineStatus.FINISHED_STEP_LIMIT_REACHED, None

    engine = ImmediateEngine()
    driver = make_mg_gendriver(1, callback)
    driver(engine.run)

    assert len(meshes) >= 1
