from nmesh.mesher import make_mg_gendriver, MeshEngineStatus, MeshEngineCommand


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
