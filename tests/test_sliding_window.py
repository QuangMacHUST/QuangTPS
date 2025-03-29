#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the sliding window inference in SegmentationModel.

This test focuses solely on the sliding window implementation.
"""

import os
import sys
import unittest
import logging
import numpy as np
import tempfile
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try importing PyTorch - handle gracefully if not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. Skipping sliding window test.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TestSlidingWindow(unittest.TestCase):
    """Test case for sliding window inference."""
    
    def test_sliding_window(self):
        """Test the sliding window inference method."""
        if not HAS_TORCH:
            self.skipTest("PyTorch not available")
        
        # Import here to avoid import errors if PyTorch is not available
        from quangtps.segmentation.deep_learning_segmentation import SegmentationModel
        
        # Create a minimal model for testing
        class MinimalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Very simple model just for testing
                self.conv = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1)
            
            def forward(self, x):
                return self.conv(x)
        
        # Create a temporary file for the model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
            model_path = temp_file.name
            
            # Create and save a dummy model
            model = MinimalModel()
            
            # Save model with required metadata
            metadata = {
                'name': 'test_model',
                'out_channels': 2,
                'version': '1.0',
                'structure_names': ['test_structure'],
                'activation': 'softmax'
            }
            
            # Save model with metadata
            torch.save({
                'model_state_dict': model.state_dict(),
                'info': metadata
            }, model_path)
            
        try:
            # Load model
            segmentation_model = SegmentationModel(model_path)
            
            # Set device to CPU for testing
            segmentation_model.device = torch.device('cpu')
            
            # Check if _sliding_window_inference method exists
            self.assertTrue(hasattr(segmentation_model, '_sliding_window_inference'), 
                          "Sliding window inference method not implemented")
            
            # Create a minimal test case
            dummy_tensor = torch.zeros((1, 1, 32, 32, 32))
            
            # Create a mock model with expected behavior
            mock_model = MagicMock()
            mock_model.return_value = torch.zeros((1, 2, 32, 32, 32))
            
            # Replace the real model with our mock
            original_model = segmentation_model.model
            segmentation_model.model = mock_model
            
            try:
                # Call sliding window inference
                result = segmentation_model._sliding_window_inference(dummy_tensor, (16, 16, 16), 0.5)
                
                # Check result shape (first dimension is batch size, second is channels)
                self.assertEqual(result.shape[0], dummy_tensor.shape[0])
                if segmentation_model.model_info.get('out_channels', 0) > 1:
                    # For multi-class output, result is of shape (B, D, H, W)
                    self.assertEqual(result.shape[1:], dummy_tensor.shape[2:])
                else:
                    # For binary output, result is of shape (B, 1, D, H, W)
                    self.assertEqual(result.shape[1], 1)
                    self.assertEqual(result.shape[2:], dummy_tensor.shape[2:])
                
                logger.info("Sliding window test passed")
            finally:
                # Restore original model
                segmentation_model.model = original_model
        finally:
            # Clean up temporary file
            os.unlink(model_path)


if __name__ == "__main__":
    unittest.main() 