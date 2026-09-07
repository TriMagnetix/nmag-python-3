from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import patch

from data_writer_test_support import (
    AvailabilityAwareSimulation,
    DataWriterTestCase,
    FailingSimulation,
    SchemaChangingSimulation,
)

from si.physical import SI
from simulation.quantity import Quantity


class TestDataWriterSchema(DataWriterTestCase):
    def test_file_creation_and_header(self):
        """Test that the .ndt file is created and the header is written correctly."""
        self.writer.save(self.source)

        self.assertTrue(self.ndt_path.exists(), "NDT file was not created")

        with open(self.ndt_path, newline="") as f:
            lines = f.readlines()

        # Check metadata comment
        self.assertTrue(lines[0].startswith("# Simulation: Test_Sim"))

        # Check header columns (tab-separated)
        header = lines[1].strip().split("\t")
        self.assertIn("step", header)
        self.assertIn("time", header)
        self.assertIn("m_Permalloy_0", header)
        self.assertIn("H_ext", header)

    def test_si_unit_extraction_fix(self):
        """
        Ensure SI objects are converted to floats.
        """
        # The mock returns SI(500.0, 'A/m') for 'H_ext'
        self.writer.save(self.source)

        with open(self.ndt_path) as f:
            lines = [line.strip() for line in f.readlines() if not line.startswith("#")]

        header = lines[0].split("\t")
        data = lines[1].split("\t")

        # Find index of H_ext
        h_ext_idx = header.index("H_ext")
        h_ext_val = data[h_ext_idx]

        self.assertEqual(float(h_ext_val), 500.0)

    def test_avoid_same_step(self):
        """Test that data is not written if the step hasn't changed and the flag is set."""
        self.source.step = 10
        self.writer.save(self.source)  # First save

        self.writer.save(self.source, avoid_same_step=True)

        with open(self.ndt_path) as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)

    def test_repeated_save_reuses_existing_quantity_header(self):
        """Quantity descriptors are only needed while writing the first header."""
        self.writer.save(self.source)

        self.source.step = 1
        with patch.object(
            Quantity,
            "sub_quantity",
            side_effect=AssertionError("sub_quantity should not run after header"),
        ):
            self.writer.save(self.source)

        with open(self.ndt_path) as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 4)

    def test_existing_schema_row_values_use_cached_column_index(self):
        self.writer.save(self.source)

        self.assertIsNotNone(self.writer._column_index_by_name)
        row_values = self.writer._existing_schema_row_values(
            [
                ("H_ext", SI(750.0, "A/m")),
                ("step", 3),
                ("missing_from_schema", 99),
            ],
        )

        self.assertEqual(row_values[self.writer._column_index_by_name["step"]], 3)
        self.assertEqual(row_values[self.writer._column_index_by_name["H_ext"]], 750.0)
        self.assertNotIn(99, row_values)

    def test_ndt_timings_split_initial_and_existing_schema_paths(self):
        self.writer.save(self.source)

        first_timings = self.writer.last_save_timings_seconds
        self.assertGreaterEqual(first_timings["gather_ndt_columns"], 0.0)
        self.assertGreaterEqual(
            first_timings["gather_ndt_initial_schema_columns"],
            0.0,
        )
        self.assertGreaterEqual(first_timings["build_ndt_header_schema"], 0.0)
        self.assertGreaterEqual(first_timings["format_ndt_values"], 0.0)
        self.assertGreaterEqual(first_timings["write_ndt_header"], 0.0)
        self.assertNotIn("gather_ndt_existing_schema_columns", first_timings)

        self.source.step = 1
        self.writer.save(self.source)

        second_timings = self.writer.last_save_timings_seconds
        self.assertGreaterEqual(second_timings["gather_ndt_columns"], 0.0)
        self.assertGreaterEqual(
            second_timings["gather_ndt_existing_schema_columns"],
            0.0,
        )
        self.assertGreaterEqual(second_timings["format_ndt_values"], 0.0)
        self.assertNotIn("gather_ndt_initial_schema_columns", second_timings)
        self.assertNotIn("build_ndt_header_schema", second_timings)
        self.assertNotIn("write_ndt_header", second_timings)

    def test_first_save_uses_single_ndt_open_for_header_and_row(self):
        """The first save should not reopen the NDT file after writing the header."""
        real_open = builtins.open
        ndt_open_modes: list[str] = []

        def recording_open(file, *args, **kwargs):
            if isinstance(file, (str, Path)) and Path(file) == self.ndt_path:
                mode = args[0] if args else kwargs.get("mode", "r")
                ndt_open_modes.append(mode)
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=recording_open):
            self.writer.save(self.source)

        self.assertEqual(ndt_open_modes, ["w"])

        self.source.step = 1
        ndt_open_modes.clear()
        with patch("builtins.open", side_effect=recording_open):
            self.writer.save(self.source)

        self.assertEqual(ndt_open_modes, ["a"])

        with open(self.ndt_path) as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 4)

    def test_repeated_save_only_queries_header_schema(self):
        """Late fields cannot be written without changing the existing NDT header."""
        source = SchemaChangingSimulation()
        self.writer.save(source)
        self.assertIn("H_demag", source.requested_subfields)

        source.step = 1
        source.requested_subfields.clear()
        source.fail_on_late_demag = True
        self.writer.save(source)

        self.assertNotIn("H_demag", source.requested_subfields)

    def test_first_save_uses_optional_subfield_availability_predicate(self):
        source = AvailabilityAwareSimulation({"m", "H_ext"})
        self.writer.save(source)

        self.assertIn("current_density", source.availability_queries)
        self.assertIn("H_demag", source.availability_queries)
        self.assertEqual(source.requested_subfields, ["m", "H_ext"])
        self.assertEqual(source.material_queries, ["m"])

        with open(self.ndt_path, newline="") as f:
            lines = [line.strip() for line in f if not line.startswith("#")]
        header = lines[0].split("\t")
        self.assertIn("m_Permalloy_0", header)
        self.assertIn("H_ext", header)
        self.assertNotIn("H_demag", header)

    def test_numerical_failure_is_not_silently_skipped(self):
        source = FailingSimulation()

        with self.assertRaisesRegex(RuntimeError, "numerical average failed"):
            self.writer.save(source, avoid_same_step=True)

        self.assertEqual(self.writer._last_saved_step, -1)
        self.assertFalse(self.writer._header_written)

    def test_failed_first_save_can_retry_and_build_complete_schema(self):
        source = FailingSimulation()

        with self.assertRaises(RuntimeError):
            self.writer.save(source, avoid_same_step=True)

        source.fail_average = False
        self.writer.save(source, avoid_same_step=True)

        self.assertEqual(self.writer._last_saved_step, source.step)
        with open(self.ndt_path, newline="") as f:
            lines = [line.strip() for line in f if not line.startswith("#")]
        header = lines[0].split("\t")
        self.assertIn("H_ext", header)
        self.assertEqual(len(lines), 2)
