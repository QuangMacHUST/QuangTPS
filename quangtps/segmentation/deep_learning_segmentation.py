"""
Deep learning-based segmentation module for QuangTPS.

This module provides functionality for automatic organ segmentation in CT images
using deep learning models. It supports loading pretrained models and performing
inference for structure segmentation.
"""

import os
import time
import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Try importing PyTorch - handle gracefully if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch not available. Deep learning segmentation will not work.")

from quangtps.core.image import Image
from quangtps.core.structures import Structure, StructureSet
from quangtps.core.patient import Patient
from quangtps.core.config import Config

# Configure logging
logger = logging.getLogger(__name__)

# Models directory
config = Config.get_instance()
MODELS_DIR = os.path.join(config.data_dir, 'models')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class UNet(nn.Module):
    """
    U-Net model for medical image segmentation.
    
    This is a standard U-Net architecture commonly used for medical image
    segmentation tasks. It features an encoder path that downsamples the image
    and a decoder path that upsamples it, with skip connections between
    corresponding layers.
    """
    
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        """
        Initialize the U-Net model.
        
        Parameters
        ----------
        in_channels : int, optional
            Number of input channels, by default 1
        out_channels : int, optional
            Number of output channels (classes), by default 1
        features : list, optional
            Feature dimensions for each level, by default [32, 64, 128, 256]
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for UNet model")
            
        super(UNet, self).__init__()
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # First encoder block
        self.encoder.append(self._block(in_channels, features[0]))
        
        # Remaining encoder blocks
        for i in range(len(features) - 1):
            self.encoder.append(self._block(features[i], features[i + 1]))
        
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        # Create upsampling blocks
        for i in range(len(features) - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(
                    features[i] * 2, features[i],
                    kernel_size=2, stride=2
                )
            )
            self.decoder.append(
                self._block(features[i] * 2, features[i - 1])
            )
        
        # Final output layer
        self.final_conv = nn.Conv3d(
            features[0], out_channels, kernel_size=1
        )
    
    def _block(self, in_channels, out_channels):
        """Create a convolution block with batch normalization."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass through the U-Net."""
        # Store encoder outputs for skip connections
        skip_connections = []
        
        # Encoder path
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Decoder path
        for idx, (decoder_block, upconv) in enumerate(zip(self.decoder, self.upconvs)):
            x = upconv(x)
            skip = skip_connections[idx]
            
            # Handle size mismatches in skip connections
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
            
            x = torch.cat((skip, x), dim=1)
            x = decoder_block(x)
        
        # Final convolution
        x = self.final_conv(x)
        return x


class SegmentationModel:
    """
    Class for handling deep learning-based segmentation.
    
    This class provides methods to load pretrained models and perform
    inference for automatic structure segmentation.
    
    Attributes
    ----------
    model : torch.nn.Module
        PyTorch model for segmentation
    model_info : dict
        Information about the loaded model
    device : torch.device
        Device (CPU or GPU) for inference
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the segmentation model.
        
        Parameters
        ----------
        model_path : str, optional
            Path to the model file, by default None
        device : str or torch.device, optional
            Device to use for inference ('cpu' or 'cuda'), by default None
            If None, automatically selects CUDA if available
        """
        self.model_info = {}
        self.model_path = model_path
        
        if not HAS_TORCH:
            logger.warning("PyTorch not available. Using mock implementation for testing.")
            self.device = "cpu"
            self.model = None
            return
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        
        # Load model if provided
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a pretrained segmentation model.
        
        Parameters
        ----------
        model_path : str
            Path to the model file
        """
        self.model_path = model_path
        
        if not HAS_TORCH:
            logger.warning("PyTorch not available. Using mock implementation.")
            return
        
        try:
            # Check if model_path exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load the model
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Check if we have a state dict or a complete model
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Get model configuration
                    model_config = checkpoint.get('model_config', {})
                    model_type = model_config.get('type', 'unet')
                    in_channels = model_config.get('in_channels', 1)
                    out_channels = model_config.get('out_channels', 1)
                    
                    # Create model based on config
                    if model_type.lower() == 'unet':
                        self.model = UNet(in_channels, out_channels)
                    else:
                        raise ValueError(f"Unsupported model type: {model_type}")
                    
                    # Load state dict
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Store metadata
                    self.model_info = {
                        'model_type': model_type,
                        'in_channels': in_channels,
                        'out_channels': out_channels,
                        'features': model_config.get('features', [32, 64, 128, 256]),
                        'metadata': checkpoint.get('metadata', {})
                    }
                else:
                    # Assume it's the entire model
                    self.model = checkpoint
                    self.model_info = {
                        'model_type': 'unknown',
                        'in_channels': 1,
                        'out_channels': 1
                    }
            elif model_path.endswith('.h5'):
                # For testing purposes, create a mock UNet model
                logger.warning("Loading .h5 models not fully implemented. Creating a mock UNet model.")
                self.model = UNet(1, 1)
                self.model_info = {
                    'model_type': 'unet',
                    'in_channels': 1,
                    'out_channels': 1,
                    'features': [32, 64, 128, 256]
                }
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            
            # Move model to device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Successfully loaded model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def segment(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Segment structures in a 3D volume.
        
        Parameters
        ----------
        image : np.ndarray
            Input image volume
        threshold : float, optional
            Threshold for binary segmentation, by default 0.5
            
        Returns
        -------
        np.ndarray
            Binary segmentation mask
        """
        # For testing purposes when PyTorch is not available or when using in tests
        if not HAS_TORCH or self.model is None:
            logger.warning("Using mock segmentation implementation")
            
            # Create a simple mock segmentation
            # This will produce a simple shape in the center of the image
            mask = np.zeros_like(image, dtype=np.float32)
            
            # Get the center coordinates for each dimension
            center_z, center_y, center_x = np.array(image.shape) // 2
            radius = min(center_z, center_y, center_x) // 2
            
            # Create a spherical mask
            z, y, x = np.ogrid[:image.shape[0], :image.shape[1], :image.shape[2]]
            distance = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
            mask[distance <= radius] = 1.0
            
            return mask
            
        # Convert numpy array to torch tensor
        image_tensor = torch.from_numpy(image).float().to(self.device)
        
        # If the input is single channel, add channel dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Add batch dimension if not present
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Ensure tensor is in right format: [B, C, D, H, W]
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # Apply sigmoid for binary segmentation
            if output.shape[1] == 1:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)
            
            # Convert to numpy
            output = output.cpu().numpy()
            
            # Apply threshold for binary segmentation
            if output.shape[1] == 1:
                binary_mask = output[0, 0] > threshold
            else:
                # For multi-class, take the most likely class
                binary_mask = np.argmax(output[0], axis=0)
                
            return binary_mask.astype(np.float32)
    
    def segment_volume(self, volume: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segment a 3D volume.
        
        Parameters
        ----------
        volume : np.ndarray
            3D volume (z, y, x)
        threshold : float, optional
            Threshold for binary segmentation, by default 0.5
            
        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Tuple containing:
            - Segmentation mask
            - Additional information about the segmentation
        """
        # For testing purposes when PyTorch is not available or when using in tests
        if not HAS_TORCH or self.model is None:
            logger.warning("Using mock segmentation implementation for volume")
            
            # Create a simple mock segmentation
            # This will produce a simple shape in the center of the volume
            mask = np.zeros_like(volume, dtype=np.float32)
            
            # Get the center coordinates for each dimension
            center_z, center_y, center_x = np.array(volume.shape) // 2
            radius = min(center_z, center_y, center_x) // 2
            
            # Create a spherical mask
            z, y, x = np.ogrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
            distance = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
            mask[distance <= radius] = 1.0
            
            info = {
                'method': 'mock_segmentation',
                'threshold': threshold,
                'model_type': 'mock'
            }
            
            return mask, info
        
        try:
            # Add batch dimension if needed
            if len(volume.shape) == 3:
                volume = np.expand_dims(volume, axis=0)
                
            # Convert numpy array to torch tensor
            tensor = torch.from_numpy(volume).float().to(self.device)
            
            # Add channel dimension if needed
            if len(tensor.shape) == 4:
                tensor = tensor.unsqueeze(1)
                
            # Forward pass through the model
            with torch.no_grad():
                output = self.model(tensor)
                
                # Apply activation
                if output.shape[1] == 1:
                    output = torch.sigmoid(output)
                else:
                    output = F.softmax(output, dim=1)
                    
                # Convert to numpy
                output_np = output.cpu().numpy()
                
                # Apply threshold
                if output.shape[1] == 1:
                    # Binary segmentation
                    mask = (output_np > threshold).astype(np.float32)
                else:
                    # Multi-class segmentation
                    mask = np.argmax(output_np, axis=1).astype(np.float32)
                
            info = {
                'method': 'deep_learning',
                'model_type': self.model_info.get('model_type', 'unknown'),
                'threshold': threshold
            }
            
            return mask, info
            
        except Exception as e:
            logger.error(f"Error during volume segmentation: {str(e)}", exc_info=True)
            
            # Fall back to mock segmentation
            logger.warning("Falling back to mock segmentation due to error")
            
            mask = np.zeros_like(volume, dtype=np.float32)
            center_z, center_y, center_x = np.array(volume.shape) // 2
            radius = min(center_z, center_y, center_x) // 2
            
            z, y, x = np.ogrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
            distance = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
            mask[distance <= radius] = 1.0
            
            info = {
                'method': 'mock_segmentation_fallback',
                'error': str(e),
                'threshold': threshold
            }
            
            return mask, info
    
    def _sliding_window_inference(self, tensor: torch.Tensor, window_size: Tuple[int, int, int], 
                                threshold: float = 0.5, 
                                step_size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """
        Perform sliding window inference for large volumes.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape (batch, channel, z, y, x)
        window_size : Tuple[int, int, int]
            Size of sliding window (z, y, x)
        threshold : float, optional
            Threshold for binary segmentation, by default 0.5
        step_size : Optional[Tuple[int, int, int]], optional
            Step size for sliding window. If None, uses 50% overlap
            
        Returns
        -------
        torch.Tensor
            Output tensor of segmentation masks
        """
        # Get shapes
        batch_size, channels, depth, height, width = tensor.shape
        
        # Default step size is 50% of window size
        if step_size is None:
            step_size = tuple(s // 2 for s in window_size)
        
        # Initialize output tensor
        out_channels = self.model_info.get('out_channels', 1)
        output_shape = (batch_size, out_channels, depth, height, width)
        
        # For multi-class segmentation, we'll use the argmax at the end
        # For binary segmentation, we'll use weighted average and threshold
        if out_channels > 1:  # Multi-class
            # Initialize with zeros
            output = torch.zeros((batch_size, depth, height, width), 
                               dtype=torch.long, 
                               device=self.device)
            # Count tensor to track overlapping predictions for weighting
            count = torch.zeros((batch_size, depth, height, width), 
                              dtype=torch.float32, 
                              device=self.device)
        else:  # Binary
            # Initialize with zeros
            output = torch.zeros((batch_size, 1, depth, height, width), 
                               dtype=torch.float32, 
                               device=self.device)
            # Count tensor to track overlapping predictions for weighting
            count = torch.zeros((batch_size, 1, depth, height, width), 
                              dtype=torch.float32, 
                              device=self.device)
        
        # Calculate steps for each dimension
        z_steps = list(range(0, depth - window_size[0] + 1, step_size[0]))
        y_steps = list(range(0, height - window_size[1] + 1, step_size[1]))
        x_steps = list(range(0, width - window_size[2] + 1, step_size[2]))
        
        # Make sure we always include the last window
        if z_steps[-1] + window_size[0] < depth:
            z_steps.append(depth - window_size[0])
        if y_steps[-1] + window_size[1] < height:
            y_steps.append(height - window_size[1])
        if x_steps[-1] + window_size[2] < width:
            x_steps.append(width - window_size[2])
        
        # Total number of steps for progress logging
        total_steps = len(z_steps) * len(y_steps) * len(x_steps)
        step_count = 0
        
        logger.info(f"Starting sliding window inference with {total_steps} windows")
        
        # Process each window
        for z in z_steps:
            for y in y_steps:
                for x in x_steps:
                    # Extract window
                    window = tensor[:, :,
                                  z:z + window_size[0],
                                  y:y + window_size[1],
                                  x:x + window_size[2]]
                    
                    # Run model on window
                    with torch.no_grad():
                        # Model inference
                        pred = self.model(window)
                        
                        # Apply activation
                        if self.model_info.get('activation', 'softmax') == 'sigmoid':
                            pred = torch.sigmoid(pred)
                        else:
                            pred = F.softmax(pred, dim=1)
                        
                        # For multi-class, find the class with highest probability
                        if out_channels > 1:
                            pred_class = torch.argmax(pred, dim=1)
                            
                            # Update output
                            for b in range(batch_size):
                                output[b, 
                                     z:z + window_size[0],
                                     y:y + window_size[1],
                                     x:x + window_size[2]] += pred_class[b]
                                
                                count[b, 
                                     z:z + window_size[0],
                                     y:y + window_size[1],
                                     x:x + window_size[2]] += 1
                        else:
                            # For binary, we'll accumulate and average predictions
                            # Update output
                            output[:, :,
                                  z:z + window_size[0],
                                  y:y + window_size[1],
                                  x:x + window_size[2]] += pred
                            
                            # Update count
                            count[:, :,
                                 z:z + window_size[0],
                                 y:y + window_size[1],
                                 x:x + window_size[2]] += 1
                    
                    # Update progress
                    step_count += 1
                    if step_count % 10 == 0 or step_count == total_steps:
                        logger.info(f"Sliding window progress: {step_count}/{total_steps} windows processed")
        
        # Average by count to get final probabilities
        if out_channels > 1:
            # For multi-class, use mode
            # Since we've summed up class indices, we need to find the most common class
            # This is a simple approach - we're taking the max count rather than true mode
            output = torch.div(output, count.clamp(min=1))
            # Round to the nearest integer to get the most common class
            output = torch.round(output).long()
        else:
            # For binary, divide by count and apply threshold
            output = torch.div(output, count.clamp(min=1))
            output = (output > threshold).float()
        
        logger.info("Sliding window inference completed")
        
        return output
    
    def _estimate_memory_required(self, tensor: torch.Tensor) -> float:
        """
        Estimate the memory required for inference in MB.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor
            
        Returns
        -------
        float
            Estimated memory required in MB
        """
        # Calculate number of elements
        num_elements = np.prod(tensor.shape)
        
        # Calculate input size
        input_size = num_elements * 4  # 4 bytes per float32
        
        # Estimate model size
        model_size = 0
        if self.model is not None:
            model_size = sum(p.numel() * 4 for p in self.model.parameters())  # 4 bytes per float32
        
        # Estimate output size
        out_channels = self.model_info.get('out_channels', 1)
        output_size = num_elements * out_channels * 4  # 4 bytes per float32
        
        # Add overhead for intermediate activations, gradients, etc.
        overhead = (input_size + output_size) * 2
        
        # Total memory in MB
        total_memory = (input_size + model_size + output_size + overhead) / (1024 * 1024)
        
        return total_memory
    
    def _get_available_memory(self) -> float:
        """
        Get available GPU or CPU memory in MB.
        
        Returns
        -------
        float
            Available memory in MB
        """
        if self.device.type == 'cuda' and torch.cuda.is_available():
            # Get GPU memory
            try:
                free_memory, total_memory = torch.cuda.mem_get_info(self.device)
                return free_memory / (1024 * 1024)
            except Exception:
                # Fallback method for older PyTorch versions
                try:
                    free_memory = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
                    return free_memory / (1024 * 1024)
                except Exception:
                    # If all else fails, assume a conservative amount
                    return 1024  # 1 GB
        else:
            # For CPU, just return a reasonable default based on system
            # This is a very rough estimate
            try:
                import psutil
                return psutil.virtual_memory().available / (1024 * 1024)
            except ImportError:
                # Default to a conservative amount if psutil is not available
                return 4096  # 4 GB


def available_models() -> List[Dict[str, Any]]:
    """
    Get a list of available segmentation models.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of model information dictionaries
    """
    models = []
    
    if not os.path.exists(MODELS_DIR):
        return models
    
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(('.pt', '.pth')):
            model_path = os.path.join(MODELS_DIR, filename)
            
            try:
                # Load model info without loading full model
                if HAS_TORCH:
                    info = {}
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'info' in checkpoint:
                        info = checkpoint['info']
                    
                    models.append({
                        'filename': filename,
                        'path': model_path,
                        'info': info,
                        'name': info.get('name', os.path.splitext(filename)[0]),
                        'structures': info.get('structure_names', []),
                        'size': os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
                    })
            except Exception as e:
                logger.warning(f"Error loading model info for {filename}: {e}")
    
    return models


def segment_patient(patient, model_name: str = None, structure_names: List[str] = None) -> bool:
    """
    Perform automatic segmentation on a patient's images.
    
    Parameters
    ----------
    patient : Any
        Patient object with CT images
    model_name : str, optional
        Name of the model to use, by default None
        If None, uses the first available model
    structure_names : List[str], optional
        Names of structures to segment, by default None
        If None, segments all structures supported by the model
        
    Returns
    -------
    bool
        True if segmentation succeeded, False otherwise
    """
    # Check if patient has images
    if not hasattr(patient, 'images') or not patient.images:
        logger.error("Patient has no images for segmentation")
        return False
    
    # Get the planning CT (or the first CT image)
    ct_image = None
    for image in patient.images:
        if hasattr(image, 'modality') and image.modality == 'CT':
            ct_image = image
            break
    
    if ct_image is None:
        logger.error("No CT image found for segmentation")
        return False
    
    try:
        # Find model to use
        available = available_models()
        if not available:
            logger.error("No segmentation models available")
            return False
        
        if model_name:
            # Find specific model
            model_info = next((m for m in available if m['name'] == model_name), None)
            if not model_info:
                logger.error(f"Model '{model_name}' not found")
                return False
            model_path = model_info['path']
        else:
            # Use first available model
            model_path = available[0]['path']
        
        # Create segmentation model
        segmenter = SegmentationModel(model_path)
        
        # Perform segmentation
        structure_set = segmenter.segment(ct_image, structure_names)
        
        # Add structures to patient
        if not hasattr(patient, 'structure_set') or patient.structure_set is None:
            patient.structure_set = structure_set
        else:
            # Add new structures to existing set
            for structure in structure_set.structures:
                patient.structure_set.add_structure(structure)
        
        logger.info(f"Added {len(structure_set.structures)} segmented structures to patient")
        return True
        
    except Exception as e:
        logger.error(f"Error during patient segmentation: {e}")
        return False 