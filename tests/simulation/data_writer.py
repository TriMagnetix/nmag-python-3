import unittest
import tempfile
import shutil
import csv
import os
from pathlib import Path
from typing import List, Any, Optional

from si.physical import SI
from simulation.quantity import known_quantities
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
        self.time = 0.0
        self.stage_time = 0.0
        self.real_time = 123.45
        
        # Track calls to save_spatial_fields for assertion
        self.save_spatial_calls: List[tuple] = []

    def get_subfield_average(self, subfieldname: str, mat_name: Optional[str] = None) -> Any:
        # Return specific test values based on field names
        if subfieldname == 'm':
            # Test returning a list (vector)
            return [1.0, 0.0, 0.0]
        elif subfieldname == 'H_ext':
             # Test returning an SI object
            return SI(500.0, 'A/m') 
        elif subfieldname == 'E_total':
            return 1.5e-21
        return 0.0

    def get_materials_of_field(self, field_name: str) -> List[Any]:
        # Return a list of mock objects with a .name attribute
        if field_name == 'm':
            return [MockMaterial("Permalloy")]
        return []

    def get_all_field_names(self) -> List[str]:
        return ['m', 'H_ext', 'E_total']

    def save_spatial_fields(self, filename: str, fieldnames: List[str]) -> None:
        self.save_spatial_calls.append((filename, fieldnames))


class TestDataWriter(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for output files
        self.test_dir = tempfile.mkdtemp()
        self.ndt_path = Path(self.test_dir) / "output.ndt"
        self.h5_path = Path(self.test_dir) / "output.h5"
        
        self.writer = DataWriter(self.ndt_path, self.h5_path)
        self.source = MockSimulation()

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_file_creation_and_header(self):
        """Test that the .ndt file is created and the header is written correctly."""
        self.writer.save(self.source)

        self.assertTrue(self.ndt_path.exists(), "NDT file was not created")

        with open(self.ndt_path, 'r', newline='') as f:
            lines = f.readlines()
            
        # Check metadata comment
        self.assertTrue(lines[0].startswith("# Simulation: Test_Sim"))
        
        # Check header columns (tab-separated)
        header = lines[1].strip().split('\t')
        self.assertIn('step', header)
        self.assertIn('time', header)
        self.assertIn('m_Permalloy_0', header)
        self.assertIn('H_ext', header)

    def test_si_unit_extraction_fix(self):
        """
        Ensure SI objects are converted to floats.
        """
        # The mock returns SI(500.0, 'A/m') for 'H_ext'
        self.writer.save(self.source)

        with open(self.ndt_path, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            pass
        
        with open(self.ndt_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if not l.startswith("#")]
        
        header = lines[0].split('\t')
        data = lines[1].split('\t')
        
        # Find index of H_ext
        h_ext_idx = header.index('H_ext')
        h_ext_val = data[h_ext_idx]
        
        self.assertEqual(float(h_ext_val), 500.0)

    def test_avoid_same_step(self):
        """Test that data is not written if the step hasn't changed and the flag is set."""
        self.source.step = 10
        self.writer.save(self.source) # First save
        
        self.writer.save(self.source, avoid_same_step=True)
        
        with open(self.ndt_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)

    def test_spatial_save_trigger(self):
        """Test that the writer correctly delegates spatial saving to the source."""
        self.writer.save(self.source, fields=['m', 'H_ext'])
        
        self.assertEqual(len(self.source.save_spatial_calls), 1)
        filename, fields = self.source.save_spatial_calls[0]
        
        self.assertEqual(filename, str(self.h5_path))
        self.assertEqual(fields, ['m', 'H_ext'])

    def test_spatial_save_all(self):
        """Test the 'all' keyword for fields."""
        self.writer.save(self.source, fields='all')
        
        self.assertEqual(len(self.source.save_spatial_calls), 1)
        _, fields = self.source.save_spatial_calls[0]
        
        # Mock returns ['m', 'H_ext', 'E_total'] for get_all_field_names
        self.assertEqual(fields, ['m', 'H_ext', 'E_total'])

if __name__ == '__main__':
    unittest.main()
