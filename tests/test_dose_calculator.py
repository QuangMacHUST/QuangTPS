"""
Unit tests for dose calculation functionality.

This module tests the dose calculation engine, including pencil beam algorithm implementation.
"""

import unittest
import numpy as np
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to allow importing QuangTPS
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from quangtps.dose.dose_calculator import DoseCalculator, PencilBeamKernel, PhotonEnergyParams
    from quangtps.beams.beam import Beam, BeamSet
    from quangtps.structures.structure import Structure
    from quangtps.structures.structure_set import StructureSet
except ImportError:
    # Create mock classes for testing if imports fail
    class DoseCalculator:
        def __init__(self):
            self.image = None
            self.structure_set = None
            self.beam_set = None
            self.result_dose = None
            
        def calculate_dose(self):
            return np.zeros((10, 10, 10))
    
    class PencilBeamKernel:
        def __init__(self, energy="6MV"):
            self.energy = energy
            self.kernel = np.zeros((3, 3, 3))
            
    class PhotonEnergyParams:
        def __init__(self, energy="6MV"):
            self.energy = energy
            self.mu = 0.05
            
    class Beam:
        def __init__(self, name=""):
            self.name = name
            self.energy = "6MV"
            self.gantry_angle = 0
            self.field_size = (10, 10)
            self.weight = 1.0
            
    class BeamSet:
        def __init__(self, name=""):
            self.name = name
            self.beams = []
            self.prescription_dose = 0
            
    class Structure:
        def __init__(self, name=""):
            self.name = name
            self.mask = None
            
    class StructureSet:
        def __init__(self, name=""):
            self.name = name
            self.structures = []

class TestPhotonEnergyParams(unittest.TestCase):
    """Test the PhotonEnergyParams class."""
    
    def test_init(self):
        """Test initialization of PhotonEnergyParams."""
        params = PhotonEnergyParams("6MV")
        self.assertEqual(params.energy, "6MV")
        self.assertTrue(isinstance(params.spectrum, dict))
        self.assertTrue(isinstance(params.mu, float))
        self.assertTrue(isinstance(params.dmax, float))
        
    def test_different_energies(self):
        """Test different energies have different parameters."""
        params_6mv = PhotonEnergyParams("6MV")
        params_10mv = PhotonEnergyParams("10MV")
        
        self.assertNotEqual(params_6mv.mu, params_10mv.mu)
        self.assertNotEqual(params_6mv.dmax, params_10mv.dmax)
        
class TestPencilBeamKernel(unittest.TestCase):
    """Test the PencilBeamKernel class."""
    
    def test_init(self):
        """Test initialization of PencilBeamKernel."""
        kernel = PencilBeamKernel("6MV")
        self.assertEqual(kernel.energy, "6MV")
        self.assertTrue(isinstance(kernel.kernel, np.ndarray))
        
        # Kernel should be 3D with odd dimensions
        self.assertEqual(len(kernel.kernel.shape), 3)
        self.assertEqual(kernel.kernel.shape[0] % 2, 1)
        self.assertEqual(kernel.kernel.shape[1] % 2, 1)
        
    def test_get_kernel_at_depth(self):
        """Test getting kernel slice at a specific depth."""
        kernel = PencilBeamKernel("6MV")
        
        # Get kernel at different depths
        slice_0 = kernel.get_kernel_at_depth(0)
        slice_5 = kernel.get_kernel_at_depth(5)
        
        # Should be 2D arrays
        self.assertEqual(len(slice_0.shape), 2)
        self.assertEqual(len(slice_5.shape), 2)
        
        # Should be the same dimensions
        self.assertEqual(slice_0.shape, slice_5.shape)
        
        # Slices should be different since dose changes with depth
        self.assertFalse(np.array_equal(slice_0, slice_5))
        
class TestDoseCalculator(unittest.TestCase):
    """Test the DoseCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = DoseCalculator()
        
        # Create a simple CT image (just a numpy array)
        self.image = np.ones((50, 50, 20), dtype=np.float32) * 1000  # HU water = 0
        
        # Create a simple structure set
        self.structure_set = StructureSet("Test Set")
        
        # Create a simple target structure
        target = Structure("Target")
        target.mask = np.zeros((50, 50, 20), dtype=bool)
        target.mask[20:30, 20:30, 5:15] = True  # 10x10x10 cube
        self.structure_set.structures.append(target)
        
        # Create a simple beam set
        self.beam_set = BeamSet("Test Plan")
        self.beam_set.prescription_dose = 2.0  # Gy
        
        # Add a beam
        beam = Beam("AP")
        beam.energy = "6MV"
        beam.gantry_angle = 0.0
        beam.field_size = (10.0, 10.0)  # cm
        beam.weight = 1.0
        self.beam_set.beams.append(beam)
        
        # Set up calculator
        self.calc.set_image(self.image)
        self.calc.set_structure_set(self.structure_set)
        self.calc.set_beam_set(self.beam_set)
        
    def test_initialization(self):
        """Test initialization of DoseCalculator."""
        self.assertIsNotNone(self.calc)
        self.assertEqual(self.calc.image.shape, (50, 50, 20))
        self.assertEqual(len(self.calc.structure_set.structures), 1)
        self.assertEqual(len(self.calc.beam_set.beams), 1)
        
    def test_get_kernel(self):
        """Test getting a kernel for a specific energy."""
        kernel_6mv = self.calc.get_kernel("6MV")
        self.assertIsNotNone(kernel_6mv)
        self.assertEqual(kernel_6mv.energy, "6MV")
        
        # Should cache kernels
        kernel_6mv_2 = self.calc.get_kernel("6MV")
        self.assertIs(kernel_6mv, kernel_6mv_2)
        
        # Different energies should give different kernels
        kernel_10mv = self.calc.get_kernel("10MV")
        self.assertIsNotNone(kernel_10mv)
        self.assertEqual(kernel_10mv.energy, "10MV")
        self.assertIsNot(kernel_6mv, kernel_10mv)
        
    def test_dose_calculation(self):
        """Test basic dose calculation functionality."""
        # Calculate dose
        result = self.calc.calculate_dose()
        
        # Should return a numpy array
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, np.ndarray))
        
        # Result should be a 3D array
        self.assertEqual(len(result.shape), 3)
        
        # Maximum dose should be positive
        self.assertGreater(np.max(result), 0)
        
        # Should store the result
        self.assertIs(self.calc.result_dose, result)
        
    def test_multiple_beams(self):
        """Test calculation with multiple beams."""
        # Add a second beam
        beam2 = Beam("RLAT")
        beam2.energy = "6MV"
        beam2.gantry_angle = 90.0
        beam2.field_size = (10.0, 10.0)  # cm
        beam2.weight = 1.0
        self.beam_set.beams.append(beam2)
        
        # Calculate dose
        result = self.calc.calculate_dose()
        
        # Should return a numpy array
        self.assertIsNotNone(result)
        
        # Should be able to get dose at a point
        dose = self.calc.get_dose_at_point(25, 25, 10)
        self.assertGreaterEqual(dose, 0)
        
    def test_dose_grid_info(self):
        """Test getting dose grid information."""
        # Calculate dose
        self.calc.calculate_dose()
        
        # Get grid info
        info = self.calc.get_dose_grid_info()
        
        # Should have expected keys
        self.assertTrue('resolution' in info)
        self.assertTrue('shape' in info)
        self.assertTrue('max_dose' in info)
        self.assertTrue('min_dose' in info)
        self.assertTrue('mean_dose' in info)
        
        # Values should be reasonable
        self.assertTrue(isinstance(info['max_dose'], float))
        self.assertGreaterEqual(info['max_dose'], 0)
        self.assertGreaterEqual(info['mean_dose'], 0)
        
    def test_clear(self):
        """Test clearing the calculator."""
        # Calculate dose
        self.calc.calculate_dose()
        
        # Clear
        self.calc.clear()
        
        # Should reset all data
        self.assertIsNone(self.calc.image)
        self.assertIsNone(self.calc.structure_set)
        self.assertIsNone(self.calc.beam_set)
        self.assertIsNone(self.calc.result_dose)

if __name__ == '__main__':
    unittest.main() 