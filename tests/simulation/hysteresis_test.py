import unittest
from unittest.mock import MagicMock

from simulation.hysteresis import (
    _append_x_list,
    _join_save_and_do_lists,
    _string_normalise,
    simulation_relax,
    # _update_progress_file will be tested later with the sim object
    # _next_deltas and _next_time will be tested later with the sim object
    # simulation_relax and simulation_hysteresis will be tested later
)
from when import When, at, every


class TestHysteresisHelpers(unittest.TestCase):
    """
    Tests the helper functions within the hysteresis.py file
    that do not depend on a live Simulation object.

    Tests for sim-dependent functions (like _update_progress_file)
    and integration tests (for simulation_hysteresis) will be
    added after the Simulation class is defined.
    """

    def test_string_normalise(self):
        """Tests the string normalization function."""
        self.assertEqual(
            _string_normalise("save  fields"), "save_fields"
        )
        self.assertEqual(
            _string_normalise("Save_Fields"), "save_fields"
        )
        self.assertEqual(
            _string_normalise("Save_Fields", lower=False), "Save_Fields"
        )
        self.assertEqual(
            _string_normalise("save fields", spaces=None), "save fields"
        )
        self.assertEqual(
            _string_normalise(" save__fields ", spaces='-'), "-save-fields-"
        )

    def test_append_x_list(self):
        """Tests the logic for parsing 'save' and 'do' lists."""

        # Create mock action functions
        mock_action_avg = MagicMock(name="averages_func")
        mock_action_fld = MagicMock(name="fields_func")

        predefined = {
            "save_averages": mock_action_avg,
            "do_fields": mock_action_fld
        }

        target_list = []

        # Test 1: A simple string lookup
        input_list_1 = [('averages', at('stage_end'))]
        _append_x_list(
            target_list,
            input_list_1,
            prefix="save",
            predefined_actions=predefined
        )
        self.assertEqual(len(target_list), 1)
        self.assertEqual(target_list[0][0], mock_action_avg)
        self.assertIsInstance(target_list[0][1], When)

        # Test 2: A direct callable
        def my_func(sim):
            return "test"
        input_list_2 = [(my_func, every('step', 10))]
        _append_x_list(
            target_list,
            input_list_2,
            prefix="save",
            predefined_actions=predefined
        )
        self.assertEqual(len(target_list), 2)
        self.assertEqual(target_list[1][0], my_func)

        # Test 3: A multi-item tuple
        target_list = []
        input_list_3 = [('averages', my_func, at('convergence'))]
        _append_x_list(
            target_list,
            input_list_3,
            prefix="save",
            predefined_actions=predefined
        )
        self.assertEqual(len(target_list), 2)
        self.assertEqual(target_list[0][0], mock_action_avg)
        self.assertEqual(target_list[1][0], my_func)

        # Test 4: An invalid string
        input_list_4 = [('invalid_string', at('stage_end'))]
        with self.assertRaises(ValueError) as cm:
            _append_x_list(
                target_list,
                input_list_4,
                prefix="save",
                predefined_actions=predefined
            )
            self.assertIn("I don't know how to do it", str(cm.exception))

        # Test 5: Bad syntax (not a list of tuples)
        input_list_5 = None # type: ignore
        with self.assertRaises(ValueError) as cm:
            _append_x_list(
                target_list,
                input_list_5, # type: ignore
                prefix="save",
                predefined_actions=predefined
            )
        self.assertIn("Bad syntax", str(cm.exception))


    def test_join_save_and_do_lists(self):
        """Tests that 'do' actions are ordered before 'save' actions."""
        mock_do = MagicMock(name="do_func")
        mock_save = MagicMock(name="save_func")

        predefined = {
            "do_next_stage": mock_do,
            "save_averages": mock_save
        }

        save_list = [('averages', at('stage_end'))]
        do_list = [('next_stage', at('convergence'))]

        joint_list = _join_save_and_do_lists(save_list, do_list, predefined)

        self.assertEqual(len(joint_list), 2)
        # Check that the 'do' item is first
        self.assertEqual(joint_list[0][0], mock_do)
        # Check that the 'save' item is second
        self.assertEqual(joint_list[1][0], mock_save)

    def test_relax_defaults_are_isolated(self):
        """Each relax call gets fresh mutable defaults and event objects."""
        sim = MagicMock()

        simulation_relax(sim, H_applied="field")
        first_call = sim.simulation_hysteresis.call_args

        simulation_relax(sim, H_applied="field")
        second_call = sim.simulation_hysteresis.call_args

        self.assertIsNot(first_call.kwargs["save"], second_call.kwargs["save"])
        self.assertIsNot(first_call.kwargs["do"], second_call.kwargs["do"])
        self.assertIsNot(
            first_call.kwargs["convergence_check"],
            second_call.kwargs["convergence_check"],
        )

    # Tests for _update_progress_file, _next_deltas, _next_time,
    # and simulation_hysteresis will be added as integration tests
    # once the Simulation class is available, as they all
    # depend heavily on the 'sim' object.

if __name__ == '__main__':
    unittest.main()
