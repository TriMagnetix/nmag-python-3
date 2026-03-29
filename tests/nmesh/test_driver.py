import pytest
from nmesh.mesher.driver import (
    make_mg_gendriver,
    MeshEngineStatus,
    MeshEngineCommand,
    do_every_n_steps_driver,
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
    pieces = []
    payloads = []

    def callback(piece, step, mesh):
        pieces.append(piece)
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

    assert pieces == [0, 0, 0]
    assert steps == [3, 6, 9]
    for payload in payloads:
        assert isinstance(payload, list)
        assert payload[0][0] == "COORDS"


def test_driver_passes_piece_numbers_and_raw_mesh():
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
    assert isinstance(calls[0][2], MockMesh)
    assert calls[0][2].points == [[0.0, 0.0, 0.0]]


def test_driver_handles_large_step_counts_without_recursion():
    steps = []
    pieces = []

    def callback(piece, step, mesh):
        pieces.append(piece)
        steps.append(step)

    driver = make_mg_gendriver(100, callback)
    status, _ = driver(make_engine(1100, lambda: []))

    assert status == MeshEngineStatus.FINISHED_STEP_LIMIT_REACHED
    assert all(piece == 0 for piece in pieces)
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


def test_driver_stops_on_force_equilibrium():
    """Test that driver properly handles force equilibrium status."""
    steps = []

    def callback(piece, step, mesh):
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

    def callback(piece, step, mesh):
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


def test_gendriver_with_multiple_pieces():
    """Test driver with multiple piece numbers for multi-body meshes."""
    callbacks = []

    def callback(piece, step, mesh):
        callbacks.append((piece, step))

    def mesh_factory():
        return [("COORDS", "Coordinates", [])]

    driver = make_mg_gendriver(5, callback)

    # Test piece 0
    driver(make_engine(15, mesh_factory))
    piece_0_calls = [cb for cb in callbacks if cb[0] == 0]
    assert piece_0_calls == [(0, 5), (0, 10)]

    # Reset and test piece 1
    callbacks.clear()
    driver(1)(make_engine(15, mesh_factory))
    piece_1_calls = [cb for cb in callbacks if cb[0] == 1]
    assert piece_1_calls == [(1, 5), (1, 10)]

    # Reset and test piece 2
    callbacks.clear()
    driver(2)(make_engine(15, mesh_factory))
    piece_2_calls = [cb for cb in callbacks if cb[0] == 2]
    assert piece_2_calls == [(2, 5), (2, 10)]


def test_driver_callback_receives_correct_step_numbers():
    """Verify callback gets step number where extraction was scheduled, not after."""
    steps = []

    def callback(piece, step, mesh):
        steps.append(step)

    driver = make_mg_gendriver(7, callback)
    driver(make_engine(30, lambda: []))

    # Should be called at steps 7, 14, 21, 28
    assert steps == [7, 14, 21, 28]


def test_driver_with_single_step_interval():
    """Test driver with interval=1 calls callback at every step during meshing.

    Note: The callback is NOT called when the engine finishes - only during
    intermediate mesh extractions. With limit=5, the engine takes 6 DO_STEP
    commands before finishing (including initial step from nr_step=0).
    """
    steps = []

    def callback(piece, step, mesh):
        steps.append(step)

    driver = make_mg_gendriver(1, callback)
    driver(make_engine(5, lambda: []))

    # Callbacks at steps 1, 2, 3, 4 (NOT 5 - that's when it finishes)
    # Step 0: DO_STEP (initial, no extraction)
    # Step 1: DO_EXTRACT → callback(1) → DO_STEP
    # Step 2: DO_EXTRACT → callback(2) → DO_STEP
    # Step 3: DO_EXTRACT → callback(3) → DO_STEP
    # Step 4: DO_EXTRACT → callback(4) → DO_STEP (this makes engine.step > limit)
    # Engine returns FINISHED, no callback
    assert steps == [1, 2, 3, 4]


def test_driver_never_calls_callback_at_step_zero():
    """Verify callback is never invoked at step 0, even with interval=1."""
    steps = []

    def callback(piece, step, mesh):
        steps.append(step)

    driver = make_mg_gendriver(1, callback)
    driver(make_engine(10, lambda: []))

    # Step 0 should never appear
    assert 0 not in steps
    assert steps[0] == 1  # First callback should be at step 1


def test_driver_step_counting_across_extractions():
    """Verify driver's nr_step counter (callback cadence) vs engine's internal steps.

    Important distinction:
    - driver's nr_step: tracks callback cadence (when to call callback)
    - engine's step_count: internal engine state (how many DO_STEPs received)

    These are decoupled because extractions (DO_EXTRACT) don't increment engine steps.
    """
    driver_steps = []
    engine_steps_at_callback = []

    class TrackingEngine:
        def __init__(self):
            self.step_count = 0

        def run(self, cmd):
            if cmd == MeshEngineCommand.DO_STEP:
                self.step_count += 1
                if self.step_count >= 13:
                    return MeshEngineStatus.FINISHED_STEP_LIMIT_REACHED, None
                return MeshEngineStatus.CAN_CONTINUE, self.run

            if cmd == MeshEngineCommand.DO_EXTRACT:
                # Extraction doesn't increment engine's step_count
                # It just returns current state
                return MeshEngineStatus.PRODUCED_INTERMEDIATE_MESH, (
                    {"engine_steps": self.step_count},
                    self.run,
                )

            raise AssertionError(f"Unexpected command: {cmd}")

    def callback(piece, nr_step, mesh):
        driver_steps.append(nr_step)
        engine_steps_at_callback.append(mesh.get("engine_steps"))

    engine = TrackingEngine()
    driver = make_mg_gendriver(3, callback)
    driver(engine.run)

    # Driver's nr_step values when callbacks occur
    assert driver_steps == [3, 6, 9]

    # Engine's internal step_count at each extraction
    # After initial DO_STEP at driver nr_step=0: engine.step_count=1
    # After 2 more DO_STEPs to reach driver nr_step=3: engine.step_count=4
    # The DO_EXTRACT doesn't increment engine.step_count
    assert engine_steps_at_callback == [4, 7, 10]


def test_driver_with_piece_number_zero_explicitly():
    """Test that driver(0) behaves identically to driver(callable)."""
    callbacks_direct = []
    callbacks_piece_0 = []

    def callback_direct(piece, step, mesh):
        callbacks_direct.append((piece, step))

    def callback_piece_0(piece, step, mesh):
        callbacks_piece_0.append((piece, step))

    driver1 = make_mg_gendriver(5, callback_direct)
    driver2 = make_mg_gendriver(5, callback_piece_0)

    # Direct call (should default to piece 0)
    driver1(make_engine(15, lambda: []))

    # Explicit piece 0 call
    driver2(0)(make_engine(15, lambda: []))

    # Both should have piece number 0
    assert all(cb[0] == 0 for cb in callbacks_direct)
    assert all(cb[0] == 0 for cb in callbacks_piece_0)

    # Step numbers should match
    assert [cb[1] for cb in callbacks_direct] == [cb[1] for cb in callbacks_piece_0]
