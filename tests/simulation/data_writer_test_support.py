import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from si.physical import SI
from simulation.data_writer import DataWriter


class MockMaterial:
    def __init__(self, name: str):
        self.name = name


class MockSimulation:
    """
    A mock implementation of the SimulationSource protocol.
    """

    def __init__(self):
        self.name = "Test_Sim"
        self.id = 1
        self.step = 0
        self.stage = 1
        self.stage_step = 0
        self.last_step_dt = SI(0.0, "s")
        self.time = 0.0
        self.stage_time = 0.0
        self.real_time = 123.45

        # Track calls to save_spatial_fields for assertion
        self.save_spatial_calls: list[tuple] = []
        self.last_subfield_average_timings_seconds = {}
        self.last_spatial_save_timings_seconds = {}

    def get_subfield_average(self, subfieldname: str, mat_name: str | None = None) -> Any:
        self.last_subfield_average_timings_seconds = {
            "array": 0.01,
            "field_average": 0.02,
            "subfield_array:m:compute": 0.005,
            "total": 0.03,
        }
        # Return specific test values based on field names
        if subfieldname == "m":
            # Test returning a list (vector)
            return [1.0, 0.0, 0.0]
        elif subfieldname == "H_ext":
            # Test returning an SI object
            return SI(500.0, "A/m")
        elif subfieldname == "E_total":
            return 1.5e-21
        return 0.0

    def get_materials_of_field(self, field_name: str) -> list[Any]:
        # Return a list of mock objects with a .name attribute
        if field_name == "m":
            return [MockMaterial("Permalloy")]
        return []

    def get_all_field_names(self) -> list[str]:
        return ["m", "H_ext", "E_total"]

    def get_maxangle_average(self, field_name: str) -> float | None:
        self.last_maxangle_timings_seconds = {}
        return None

    def save_spatial_fields(self, filename: str, fieldnames: list[str]) -> None:
        self.save_spatial_calls.append((filename, fieldnames))
        self.last_spatial_save_timings_seconds = {
            "field:m:compute": 0.004,
            "field:m:write": 0.006,
            "field:m:total": 0.01,
            "mesh_points": 0.002,
            "total": 0.012,
        }


class SchemaChangingSimulation(MockSimulation):
    def __init__(self):
        super().__init__()
        self.requested_subfields: list[str] = []
        self.fail_on_late_demag = False

    def get_subfield_average(self, subfieldname: str, mat_name: str | None = None) -> Any:
        self.requested_subfields.append(subfieldname)
        if subfieldname == "H_demag" and self.fail_on_late_demag:
            raise AssertionError("late H_demag should not be queried")
        if subfieldname == "m":
            return [1.0, 0.0, 0.0]
        if subfieldname == "H_ext":
            return [2.0, 3.0, 4.0]
        return None


class AvailabilityAwareSimulation(MockSimulation):
    def __init__(self, available_subfields: set[str]):
        super().__init__()
        self.available_subfields = available_subfields
        self.availability_queries: list[str] = []
        self.requested_subfields: list[str] = []
        self.material_queries: list[str] = []

    def is_subfield_available(self, subfieldname: str) -> bool:
        self.availability_queries.append(subfieldname)
        return subfieldname in self.available_subfields

    def get_materials_of_field(self, field_name: str) -> list[Any]:
        self.material_queries.append(field_name)
        if field_name not in self.available_subfields:
            raise AssertionError(f"unavailable field should not request materials: {field_name}")
        return super().get_materials_of_field(field_name)

    def get_subfield_average(self, subfieldname: str, mat_name: str | None = None) -> Any:
        self.requested_subfields.append(subfieldname)
        if subfieldname not in self.available_subfields:
            raise AssertionError(f"unavailable field should not be averaged: {subfieldname}")
        return super().get_subfield_average(subfieldname, mat_name=mat_name)


class FailingSimulation(MockSimulation):
    def __init__(self):
        super().__init__()
        self.fail_average = True

    def get_subfield_average(self, subfieldname: str, mat_name: str | None = None) -> Any:
        if self.fail_average and subfieldname == "H_ext":
            raise RuntimeError("numerical average failed")
        return super().get_subfield_average(subfieldname, mat_name=mat_name)


class DataWriterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.ndt_path = Path(self.test_dir) / "output.ndt"
        self.h5_path = Path(self.test_dir) / "output.h5"
        self.writer = DataWriter(self.ndt_path, self.h5_path)
        self.source = MockSimulation()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)
