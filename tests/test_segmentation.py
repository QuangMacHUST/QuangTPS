#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the segmentation module.

This script tests the functionality of the segmentation module,
including model downloading, segmentation inference, and sliding window.
"""

import os
import sys
import unittest
import logging
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create minimal mock classes for testing
class MockPatient:
    """Mock Patient class for testing."""
    def __init__(self, images=None, structure_set=None):
        self.images = images or []
        self.structure_set = structure_set

try:
    from quangtps.segmentation.model_downloader import (
        get_available_remote_models,
        download_model,
        ensure_default_models,
        MODELS_DIR
    )
    from quangtps.segmentation.deep_learning_segmentation import (
        SegmentationModel,
        available_models
    )
    # Import UNet only if PyTorch is available
    try:
        import torch
        from quangtps.segmentation.deep_learning_segmentation import UNet
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        print("PyTorch not available. Some tests will be skipped.")
        
    # Import other necessary modules
    from quangtps.core.image import Image
    from quangtps.core.structures import Structure, StructureSet
    
    # Monkey patch the Patient import
    import quangtps.segmentation.deep_learning_segmentation
    quangtps.segmentation.deep_learning_segmentation.Patient = MockPatient
except ImportError as e:
    print(f"Error importing QuangTPS modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TestSegmentationModule(unittest.TestCase):
    """Test cases for the segmentation module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test models
        cls.temp_dir = tempfile.mkdtemp(prefix="quangtps_test_")
        cls.original_models_dir = MODELS_DIR
        
        # Create a test image
        cls.test_image = cls._create_test_image()
        
        # Ensure test directory exists but is empty
        os.makedirs(cls.temp_dir, exist_ok=True)
        
        # Redirect models directory to temp directory
        sys.modules["quangtps.segmentation.deep_learning_segmentation"].MODELS_DIR = cls.temp_dir
        sys.modules["quangtps.segmentation.model_downloader"].MODELS_DIR = cls.temp_dir
        
        logger.info(f"Using temporary models directory: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Restore original models directory
        sys.modules["quangtps.segmentation.deep_learning_segmentation"].MODELS_DIR = cls.original_models_dir
        sys.modules["quangtps.segmentation.model_downloader"].MODELS_DIR = cls.original_models_dir
        
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
        logger.info(f"Removed temporary directory: {cls.temp_dir}")
    
    @staticmethod
    def _create_test_image():
        """Create a simple test image."""
        # Create a synthetic CT image (128x128x128 volume)
        data = np.zeros((128, 128, 128), dtype=np.float32)
        
        # Add a simple phantom: a sphere in the center
        center = np.array([64, 64, 64])
        radius = 30
        
        # Create coordinate grids
        x, y, z = np.ogrid[:128, :128, :128]
        
        # Create a sphere mask
        mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
        
        # Set sphere value to simulate bone density (1000 HU)
        data[mask] = 1000
        
        # Add a smaller sphere inside for soft tissue (-100 HU)
        mask_inner = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= (radius/2)**2
        data[mask_inner] = -100
        
        # Create Image object
        image = Image(
            id="test_image",
            data=data,
            modality="CT",
            pixel_spacing=[1.0, 1.0],
            slice_thickness=1.0,
            origin=[0, 0, 0]
        )
        
        return image
    
    def test_model_listing(self):
        """Test fetching the list of available models."""
        models = get_available_remote_models()
        
        # Check that we got a non-empty list
        self.assertIsInstance(models, list)
        self.assertTrue(len(models) > 0, "No models found in repositories")
        
        # Check model structure
        sample_model = models[0]
        self.assertIn('name', sample_model)
        self.assertIn('version', sample_model)
        
        logger.info(f"Found {len(models)} models in the repositories")
    
    def test_model_download(self):
        """Test downloading a model."""
        # Get available models
        models = get_available_remote_models()
        
        # Skip test if no models available
        if not models:
            self.skipTest("No models available for testing")
        
        # Pick the smallest model to download
        target_model = min(models, key=lambda m: m.get('size', float('inf')))
        model_name = target_model['name']
        
        logger.info(f"Selected model for testing: {model_name}")
        
        # Download the model
        success = download_model(model_name)
        
        # Check download success
        self.assertTrue(success, f"Failed to download model {model_name}")
        
        # Check model exists in models directory
        expected_file = os.path.join(self.temp_dir, target_model.get('filename', f"{model_name}.pt"))
        self.assertTrue(os.path.exists(expected_file), f"Model file not found at {expected_file}")
        
        logger.info(f"Successfully downloaded model: {model_name}")
    
    def test_model_inference(self):
        """Test model inference if a model is available."""
        if not HAS_TORCH:
            self.skipTest("PyTorch not available for inference test")
            
        # Get locally available models
        models = available_models()
        
        # Skip test if no local models
        if not models:
            try:
                # Try to download a small default model for testing
                ensure_default_models()
                models = available_models()
                if not models:
                    self.skipTest("Could not download any models for testing")
            except Exception as e:
                self.skipTest(f"Could not download default models: {e}")
        
        # Choose a model
        model_info = models[0]
        model_path = model_info['path']
        
        logger.info(f"Testing inference with model: {model_info['name']}")
        
        # Create segmentation model
        try:
            model = SegmentationModel(model_path)
            
            # Check model loading
            self.assertIsNotNone(model.model, "Model not loaded correctly")
            
            # Perform segmentation
            structure_set = model.segment(self.test_image)
            
            # Check segmentation result
            self.assertIsInstance(structure_set, StructureSet)
            self.assertTrue(len(structure_set.structures) > 0, "No structures in segmentation result")
            
            logger.info(f"Segmentation successful, found {len(structure_set.structures)} structures")
            
        except Exception as e:
            self.fail(f"Segmentation inference failed: {e}")
    
    def test_sliding_window(self):
        """Test sliding window implementation if a model is available."""
        if not HAS_TORCH:
            self.skipTest("PyTorch not available for sliding window test")
            
        # Get locally available models
        models = available_models()
        
        # Skip test if no local models
        if not models:
            self.skipTest("No models available for sliding window test")
        
        # Choose a model
        model_info = models[0]
        model_path = model_info['path']
        
        # Create segmentation model
        try:
            model = SegmentationModel(model_path)
            
            # Force sliding window by modifying the device and overriding normal segmentation
            original_device = model.device
            model.device = torch.device('cpu')  # Force CPU for testing
            
            # Check if _sliding_window_inference method exists
            if not hasattr(model, '_sliding_window_inference'):
                self.skipTest("Sliding window inference not implemented")
            
            # Create a minimal test case
            # This is just to verify the method doesn't crash
            dummy_tensor = torch.zeros((1, 1, 32, 32, 32))
            
            # Create a mock model with expected behavior
            mock_model = MagicMock()
            mock_model.return_value = torch.zeros((1, 2, 32, 32, 32))
            
            # Replace the real model with our mock
            original_model = model.model
            model.model = mock_model
            
            try:
                # Call sliding window inference
                result = model._sliding_window_inference(dummy_tensor, (16, 16, 16), 0.5)
                
                # Check result shape (first dimension is batch size, second is channels)
                self.assertEqual(result.shape[0], dummy_tensor.shape[0])
                self.assertEqual(result.shape[2:], dummy_tensor.shape[2:])
            finally:
                # Restore original model and device
                model.model = original_model
                model.device = original_device
            
        except Exception as e:
            self.fail(f"Sliding window test failed: {e}")


if __name__ == "__main__":
    unittest.main() 