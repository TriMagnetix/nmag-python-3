from __future__ import annotations

from unittest.mock import patch

from data_writer_test_support import DataWriterTestCase, MockSimulation


class TestDataWriterSpatial(DataWriterTestCase):
    def test_failed_spatial_save_does_not_block_retry_at_same_step(self):
        source = MockSimulation()

        with patch.object(
            source,
            "save_spatial_fields",
            side_effect=[OSError("spatial write failed"), None],
        ) as save_spatial_fields:
            with self.assertRaisesRegex(OSError, "spatial write failed"):
                self.writer.save(source, fields=["m"], avoid_same_step=True)

            self.assertEqual(self.writer._last_saved_step, -1)
            self.writer.save(source, fields=["m"], avoid_same_step=True)

        self.assertEqual(save_spatial_fields.call_count, 2)
        self.assertEqual(self.writer._last_saved_step, source.step)

    def test_spatial_save_trigger(self):
        """Test that the writer correctly delegates spatial saving to the source."""
        self.writer.save(self.source, fields=["m", "H_ext"])

        self.assertEqual(len(self.source.save_spatial_calls), 1)
        filename, fields = self.source.save_spatial_calls[0]

        self.assertEqual(filename, str(self.h5_path))
        self.assertEqual(fields, ["m", "H_ext"])

    def test_spatial_save_all(self):
        """Test the 'all' keyword for fields."""
        self.writer.save(self.source, fields="all")

        self.assertEqual(len(self.source.save_spatial_calls), 1)
        _, fields = self.source.save_spatial_calls[0]

        # Mock returns ['m', 'H_ext', 'E_total'] for get_all_field_names
        self.assertEqual(fields, ["m", "H_ext", "E_total"])

    def test_save_records_operation_timings(self):
        self.writer.save(self.source, fields=["m"])

        timings = self.writer.last_save_timings_seconds
        self.assertGreaterEqual(timings["total"], 0.0)
        self.assertGreaterEqual(timings["write_ndt_row"], 0.0)
        self.assertGreaterEqual(timings["gather_ndt_columns"], 0.0)
        self.assertGreaterEqual(timings["write_ndt_values"], 0.0)
        self.assertGreaterEqual(timings["save_spatial_fields"], 0.0)
        self.assertGreaterEqual(timings["spatial_detail:field_selection"], 0.0)
        self.assertGreaterEqual(timings["spatial_detail:write_dispatch"], 0.0)
        self.assertGreaterEqual(timings["average:m_Permalloy"], 0.0)
        self.assertEqual(timings["average_detail:m_Permalloy:array"], 0.01)
        self.assertEqual(timings["average_detail:m_Permalloy:field_average"], 0.02)
        self.assertEqual(
            timings["average_detail:m_Permalloy:subfield_array:m:compute"],
            0.005,
        )
        self.assertEqual(timings["average_detail:m_Permalloy:total"], 0.03)
        self.assertGreaterEqual(timings["average:H_ext"], 0.0)
        self.assertEqual(timings["spatial_detail:field:m:compute"], 0.004)
        self.assertEqual(timings["spatial_detail:field:m:write"], 0.006)
        self.assertEqual(timings["spatial_detail:field:m:total"], 0.01)
        self.assertEqual(timings["spatial_detail:mesh_points"], 0.002)
        self.assertEqual(timings["spatial_detail:total"], 0.012)
