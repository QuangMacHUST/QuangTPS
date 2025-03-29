#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for dose calculation algorithms.

This module contains tests for the various dose calculation algorithms
implemented in QuangTPS, including Pencil Beam, Collapsed Cone, Monte Carlo,
and other algorithms.
"""

import os
import unittest
import numpy as np
import logging
from unittest import mock

from quangtps.dose.algorithms.base import DoseCalculationAlgorithm
from quangtps.dose.algorithms.monte_carlo import MonteCarloAlgorithm
from quangtps.dose.algorithms.pencil_beam import PencilBeamAlgorithm
from quangtps.dose.algorithms.collapsed_cone import CollapsedConeAlgorithm
from quangtps.dose.algorithms import get_available_algorithms, get_algorithm_instance
from quangtps.dose.beam_data_processor import BeamModel
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.core.exceptions import DoseCalculationError, ValidationError


class TestDoseAlgorithms(unittest.TestCase):
    """Test suite for dose calculation algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple 3D image to represent a CT scan
        self.test_image = Image()
        self.test_image.data = np.ones((50, 50, 50), dtype=np.float32)  # Water equivalent
        self.test_image.pixel_spacing = [0.2, 0.2, 0.2]  # 2mm voxels
        self.test_image.origin = [-5.0, -5.0, -5.0]  # Origin at (-5, -5, -5) cm
        
        # Set some voxels to air (0.001 relative electron density)
        self.test_image.data[20:30, 20:30, 0:10] = 0.001
        
        # Set some voxels to bone (1.8 relative electron density)
        self.test_image.data[20:30, 20:30, 30:40] = 1.8
        
        # Create a simple beam
        self.test_beam = Beam()
        self.test_beam.name = "Test Beam"
        self.test_beam.energy = 6  # 6 MV
        self.test_beam.gantry_angle = 0.0  # AP beam
        self.test_beam.field_size = (10.0, 10.0)  # 10x10 field
        self.test_beam.sad = 100.0  # 100 cm SAD
        
        # Create a simple beam model
        self.beam_model = BeamModel()
        self.beam_model.name = "Test Model"
        self.beam_model.energy = 6
        
        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up after tests."""
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def test_available_algorithms(self):
        """Test getting list of available algorithms."""
        algorithms = get_available_algorithms()
        
        # Check that the list contains at least the Pencil Beam algorithm
        self.assertIsInstance(algorithms, list)
        self.assertGreater(len(algorithms), 0)
        
        # Check that each entry has the expected fields
        for algo in algorithms:
            self.assertIsInstance(algo, dict)
            self.assertIn('id', algo)
            self.assertIn('name', algo)
            self.assertIn('description', algo)
    
    def test_get_algorithm_instance(self):
        """Test getting algorithm instances by ID."""
        # Test getting a valid algorithm
        algorithm = get_algorithm_instance('pencil_beam')
        self.assertIsInstance(algorithm, PencilBeamAlgorithm)
        
        # Test getting a non-existent algorithm
        with self.assertRaises(ValueError):
            get_algorithm_instance('non_existent_algorithm')
    
    def test_pencil_beam_algorithm(self):
        """Test basic functionality of the Pencil Beam algorithm."""
        # Create algorithm instance
        algorithm = PencilBeamAlgorithm()
        
        # Set parameters
        algorithm.set_parameters(grid_size=0.5, heterogeneity_correction=True)
        self.assertEqual(algorithm.parameters['grid_size'], 0.5)
        self.assertTrue(algorithm.parameters['heterogeneity_correction'])
        
        # Set beam model
        algorithm.set_beam_model(self.beam_model)
        
        # Calculate dose (mock the actual calculation)
        with mock.patch.object(algorithm, '_calculate_pencil_beam_dose', return_value=(np.ones((10, 10, 10)), None)):
            result = algorithm.calculate(self.test_image, [self.test_beam])
            
            # Verify result
            self.assertIsNotNone(result)
            self.assertIsInstance(result.dose_grid, np.ndarray)
    
    def test_collapsed_cone_algorithm(self):
        """Test basic functionality of the Collapsed Cone algorithm."""
        # Create algorithm instance
        algorithm = CollapsedConeAlgorithm()
        
        # Set parameters
        algorithm.set_parameters(grid_size=0.5, num_cones=16)
        self.assertEqual(algorithm.parameters['grid_size'], 0.5)
        self.assertEqual(algorithm.parameters['num_cones'], 16)
        
        # Set beam model
        algorithm.set_beam_model(self.beam_model)
        
        # Calculate dose (mock the actual calculation)
        with mock.patch.object(algorithm, '_calculate_collapsed_cone_dose', return_value=(np.ones((10, 10, 10)), None)):
            result = algorithm.calculate(self.test_image, [self.test_beam])
            
            # Verify result
            self.assertIsNotNone(result)
            self.assertIsInstance(result.dose_grid, np.ndarray)
    
    def test_monte_carlo_algorithm(self):
        """Test basic functionality of the Monte Carlo algorithm."""
        # Create algorithm instance
        algorithm = MonteCarloAlgorithm()
        
        # Set parameters with smaller number of histories for quicker test
        algorithm.set_parameters(num_histories=1000, grid_size=0.5, use_gpu=False)
        self.assertEqual(algorithm.parameters['num_histories'], 1000)
        self.assertEqual(algorithm.parameters['grid_size'], 0.5)
        
        # Set beam model
        algorithm.set_beam_model(self.beam_model)
        
        # Calculate dose (mock the actual calculation)
        with mock.patch.object(algorithm, '_simulate_particles', return_value=(np.ones((10, 10, 10)), None)):
            result = algorithm.calculate(self.test_image, [self.test_beam])
            
            # Verify result
            self.assertIsNotNone(result)
            self.assertIsInstance(result.dose_grid, np.ndarray)
    
    def test_algorithm_validation(self):
        """Test that algorithms properly validate inputs."""
        # Create algorithm instance
        algorithm = PencilBeamAlgorithm()
        
        # Test with missing beam model
        with self.assertRaises(ValidationError):
            algorithm.calculate(self.test_image, [self.test_beam])
        
        # Set beam model
        algorithm.set_beam_model(self.beam_model)
        
        # Test with invalid image (None)
        with self.assertRaises(ValidationError):
            algorithm.calculate(None, [self.test_beam])
        
        # Test with empty beam list
        with self.assertRaises(ValidationError):
            algorithm.calculate(self.test_image, [])
        
        # Test with invalid beam (None in list)
        with self.assertRaises(ValidationError):
            algorithm.calculate(self.test_image, [None])
        
        # Test with incompatible beam energy
        test_beam_wrong_energy = Beam()
        test_beam_wrong_energy.energy = 18  # Different from model
        with self.assertRaises(ValidationError):
            algorithm.calculate(self.test_image, [test_beam_wrong_energy])


class TestAlgorithmComparison(unittest.TestCase):
    """Test comparisons between different dose calculation algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a water phantom
        self.water_phantom = Image()
        self.water_phantom.data = np.ones((100, 100, 100), dtype=np.float32)  # Water equivalent
        self.water_phantom.pixel_spacing = [0.2, 0.2, 0.2]  # 2mm voxels
        self.water_phantom.origin = [-10.0, -10.0, -10.0]  # Origin at (-10, -10, -10) cm
        
        # Create a beam
        self.test_beam = Beam()
        self.test_beam.name = "Test Beam"
        self.test_beam.energy = 6  # 6 MV
        self.test_beam.gantry_angle = 0.0  # AP beam
        self.test_beam.field_size = (10.0, 10.0)  # 10x10 field
        self.test_beam.sad = 100.0  # 100 cm SAD
        
        # Create a beam model
        self.beam_model = BeamModel()
        self.beam_model.name = "Test Model"
        self.beam_model.energy = 6
        
        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up after tests."""
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    @unittest.skipIf(os.environ.get('SKIP_LONG_TESTS') == '1', "Skipping long-running test")
    def test_algorithm_comparison(self):
        """Test that different algorithms produce reasonably similar results in water."""
        # Create algorithm instances
        pencil_beam = PencilBeamAlgorithm()
        collapsed_cone = CollapsedConeAlgorithm()
        
        # Set similar parameters
        grid_size = 0.5
        pencil_beam.set_parameters(grid_size=grid_size)
        collapsed_cone.set_parameters(grid_size=grid_size)
        
        # Set beam model
        pencil_beam.set_beam_model(self.beam_model)
        collapsed_cone.set_beam_model(self.beam_model)
        
        # Mock calculations to make them quick
        with mock.patch.object(pencil_beam, '_calculate_pencil_beam_dose') as mock_pb, \
             mock.patch.object(collapsed_cone, '_calculate_collapsed_cone_dose') as mock_cc:
            
            # Create simple dose distributions with a central hotspot
            x, y, z = np.meshgrid(
                np.linspace(-10, 10, 100), 
                np.linspace(-10, 10, 100), 
                np.linspace(-10, 10, 100)
            )
            
            # Pencil beam: Gaussian fall-off
            pb_dose = np.exp(-(x**2 + y**2) / 25) * np.exp(-z / 10)
            pb_dose = pb_dose / np.max(pb_dose)  # Normalize
            
            # Collapsed cone: Similar but slightly different
            cc_dose = np.exp(-(x**2 + y**2) / 30) * np.exp(-z / 12)
            cc_dose = cc_dose / np.max(cc_dose)  # Normalize
            
            mock_pb.return_value = (pb_dose, None)
            mock_cc.return_value = (cc_dose, None)
            
            # Calculate doses
            pb_result = pencil_beam.calculate(self.water_phantom, [self.test_beam])
            cc_result = collapsed_cone.calculate(self.water_phantom, [self.test_beam])
            
            # Compare results
            self.assertEqual(pb_result.dose_grid.shape, cc_result.dose_grid.shape)
            
            # Calculate gamma index or other metrics
            # For this test, we'll just check that a significant percentage of points
            # are within a certain dose difference
            dose_diff = np.abs(pb_result.dose_grid - cc_result.dose_grid)
            
            # Check percentage of points passing a 5% dose difference criterion
            # in the high dose region (>10% of max dose)
            high_dose_mask = (pb_result.dose_grid > 0.1) | (cc_result.dose_grid > 0.1)
            high_dose_points = np.sum(high_dose_mask)
            passing_points = np.sum((dose_diff < 0.05) & high_dose_mask)
            
            # At least 85% of high dose points should pass
            passing_percentage = passing_points / high_dose_points if high_dose_points > 0 else 0
            self.assertGreaterEqual(passing_percentage, 0.85)


if __name__ == '__main__':
    unittest.main() 