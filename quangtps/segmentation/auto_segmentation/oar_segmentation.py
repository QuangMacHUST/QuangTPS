#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OAR Segmentation Module for QuangTPS.

This module provides functionality for automatic organs-at-risk (OAR) segmentation
using deep learning models, specifically focused on critical structures in radiotherapy.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
from scipy import ndimage
import SimpleITK as sitk

from quangtps.core.exceptions import ValidationError
from quangtps.segmentation.auto_segmentation.model_loader import ModelLoader
from quangtps.segmentation.auto_segmentation.unet import UNetModel
from quangtps.segmentation.structures.structure_set import StructureSet
from quangtps.segmentation.structures.structure import Structure

logger = logging.getLogger(__name__)


class OARSegmentor:
    """
    Class for automatic organs-at-risk (OAR) segmentation.
    
    This class provides methods to automatically segment critical structures
    using various deep learning models for different anatomical sites.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize OAR segmentor.
        
        Parameters
        ----------
        model_path : str, optional
            Path to pretrained model. If None, default models will be used.
        """
        self.model_loader = ModelLoader()
        self.model_path = model_path
        self.models = {}
        self._load_default_models()
        
        # Define standard OAR names by anatomical site
        self.site_oars = {
            'brain': ['BrainStem', 'Chiasm', 'OpticNerve_L', 'OpticNerve_R', 'Eye_L', 'Eye_R', 
                      'Lens_L', 'Lens_R', 'Cochlea_L', 'Cochlea_R', 'Pituitary', 'Hippocampus_L', 'Hippocampus_R'],
            
            'head_neck': ['Parotid_L', 'Parotid_R', 'Submandibular_L', 'Submandibular_R', 'Mandible', 
                          'SpinalCord', 'BrainStem', 'Larynx', 'Esophagus', 'OralCavity', 'Lips'],
            
            'thorax': ['Lung_L', 'Lung_R', 'Heart', 'SpinalCord', 'Esophagus', 'Trachea', 'Bronchus'],
            
            'abdomen': ['Liver', 'Kidney_L', 'Kidney_R', 'Stomach', 'Duodenum', 'SmallBowel', 
                        'Colon', 'SpinalCord', 'Pancreas', 'Spleen'],
            
            'pelvis': ['Bladder', 'Rectum', 'Sigmoid', 'Femur_L', 'Femur_R', 'Bowel', 'Prostate', 
                      'Penile_Bulb', 'Urethra', 'Vagina', 'Uterus']
        }
    
    def _load_default_models(self):
        """Load default OAR segmentation models."""
        try:
            # Determine model directory
            if self.model_path is not None and os.path.exists(self.model_path):
                model_dir = self.model_path
            else:
                # Use default model path
                model_dir = os.path.join(os.path.dirname(__file__), 'models')
            
            # Define model paths for different anatomical sites
            model_paths = {
                'brain_oar': os.path.join(model_dir, 'brain_oar_model.pth'),
                'head_neck_oar': os.path.join(model_dir, 'head_neck_oar_model.pth'),
                'thorax_oar': os.path.join(model_dir, 'thorax_oar_model.pth'),
                'abdomen_oar': os.path.join(model_dir, 'abdomen_oar_model.pth'),
                'pelvis_oar': os.path.join(model_dir, 'pelvis_oar_model.pth')
            }
            
            # Load available models
            for site, path in model_paths.items():
                if os.path.exists(path):
                    logger.info(f"Loading {site} OAR segmentation model from {path}")
                    self.models[site] = self.model_loader.load_model(path)
                else:
                    logger.warning(f"Model for {site} not found at {path}")
            
        except Exception as e:
            logger.error(f"Error loading default OAR segmentation models: {str(e)}")
    
    def _determine_anatomical_site(self, image_data: np.ndarray) -> str:
        """
        Determine anatomical site from image data.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
            
        Returns
        -------
        str
            Anatomical site ('brain', 'head_neck', 'thorax', 'abdomen', 'pelvis')
        """
        # Simple heuristic based on image center and histogram characteristics
        # This can be replaced with a more sophisticated classification model
        
        # Get image center slice
        center_slice = image_data[image_data.shape[0]//2, :, :]
        
        # Calculate basic statistics
        mean_hu = np.mean(center_slice)
        std_hu = np.std(center_slice)
        
        # Simple heuristic detection based on histogram characteristics
        # These thresholds are approximate and should be adjusted based on actual data
        if mean_hu > -200 and mean_hu < 100 and std_hu > 200:
            return 'head_neck'
        elif mean_hu > -600 and mean_hu < -200:
            return 'thorax'
        elif mean_hu > -100 and mean_hu < 100 and std_hu < 200:
            return 'abdomen'
        elif mean_hu > -50 and mean_hu < 100 and np.percentile(center_slice, 75) > 100:
            return 'pelvis'
        elif mean_hu > 0 and std_hu > 250:
            return 'brain'
        else:
            # Default to thorax
            return 'thorax'
    
    def _preprocess_image(self, image_data: np.ndarray, spacing: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        spacing : Tuple[float, float, float], optional
            Image spacing in mm
            
        Returns
        -------
        np.ndarray
            Preprocessed image data
        """
        # Clone image to avoid modifying the original
        processed = image_data.copy()
        
        # Convert to float32
        processed = processed.astype(np.float32)
        
        # Clip and normalize HU values for CT
        # For CT images, typically in Hounsfield units
        min_hu = -1000
        max_hu = 1000
        processed = np.clip(processed, min_hu, max_hu)
        processed = (processed - min_hu) / (max_hu - min_hu)  # Normalize to [0, 1]
        
        # Resample to 1mm isotropic if spacing is provided and not already 1mm
        if spacing is not None and (abs(spacing[0] - 1.0) > 0.01 or abs(spacing[1] - 1.0) > 0.01 or abs(spacing[2] - 1.0) > 0.01):
            # Convert to SimpleITK for resampling
            sitk_image = sitk.GetImageFromArray(processed)
            sitk_image.SetSpacing(spacing)
            
            # Calculate new size
            new_spacing = [1.0, 1.0, 1.0]
            new_size = [int(round(original_size * original_spacing / new_spacing)) 
                       for original_size, original_spacing in zip(sitk_image.GetSize(), spacing)]
            
            # Resample
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_size)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled_image = resampler.Execute(sitk_image)
            
            # Convert back to numpy
            processed = sitk.GetArrayFromImage(resampled_image)
        
        return processed
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Post-process segmentation mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary segmentation mask
            
        Returns
        -------
        np.ndarray
            Post-processed mask
        """
        # Fill holes
        mask = ndimage.binary_fill_holes(mask)
        
        # Remove small objects
        labeled_mask, num_features = ndimage.label(mask)
        if num_features > 1:
            component_sizes = np.bincount(labeled_mask.ravel())[1:]
            too_small = component_sizes < np.max(component_sizes) * 0.05  # 5% of the largest component
            too_small_mask = too_small[labeled_mask - 1]
            labeled_mask[too_small_mask] = 0
            mask = labeled_mask > 0
        
        return mask.astype(np.uint8)
    
    def _resize_to_original(self, mask: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Resize mask to original image dimensions.
        
        Parameters
        ----------
        mask : np.ndarray
            Segmentation mask
        original_shape : Tuple[int, ...]
            Original image shape
            
        Returns
        -------
        np.ndarray
            Resized mask
        """
        if mask.shape == original_shape:
            return mask
        
        # Convert to SimpleITK for resampling
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        
        # Calculate resize factors
        factor_z = original_shape[0] / mask.shape[0]
        factor_y = original_shape[1] / mask.shape[1]
        factor_x = original_shape[2] / mask.shape[2]
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(original_shape[::-1])  # SimpleITK uses x,y,z ordering
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # For masks, use nearest neighbor
        resampled_mask = resampler.Execute(sitk_mask)
        
        # Convert back to numpy
        resized_mask = sitk.GetArrayFromImage(resampled_mask)
        
        return resized_mask
    
    def segment_oars(self, image_data: np.ndarray, site: str = 'auto', 
                   spacing: Optional[Tuple[float, float, float]] = None,
                   threshold: float = 0.5,
                   structures: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Segment multiple organs-at-risk from image data.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data (CT, MRI, etc.)
        site : str, optional
            Anatomical site ('brain', 'head_neck', 'thorax', 'abdomen', 'pelvis', or 'auto')
        spacing : Tuple[float, float, float], optional
            Image spacing in mm
        threshold : float, optional
            Probability threshold for segmentation
        structures : List[str], optional
            List of specific structures to segment. If None, all available structures for the site will be segmented.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping structure names to binary masks
        """
        try:
            # Determine anatomical site if auto
            if site == 'auto':
                site = self._determine_anatomical_site(image_data)
            
            # Check if model for the specified site is available
            site_key = f"{site}_oar"
            if site_key not in self.models:
                available_sites = [s.replace('_oar', '') for s in self.models.keys()]
                if len(available_sites) == 0:
                    raise ValidationError(f"No OAR segmentation models available")
                logger.warning(f"Model for {site} not available. Using {available_sites[0]} model instead.")
                site = available_sites[0]
                site_key = f"{site}_oar"
            
            # Prepare image data for model input
            processed_image = self._preprocess_image(image_data, spacing)
            
            # Get the list of structures to segment
            if structures is None:
                if site in self.site_oars:
                    structures = self.site_oars[site]
                else:
                    structures = []
                    logger.warning(f"No predefined OARs for site {site}")
            
            # Filter structures based on what the model can segment
            if len(structures) == 0:
                raise ValidationError(f"No structures to segment for site {site}")
            
            # Run model inference
            model = self.models[site_key]
            result_masks = {}
            
            with torch.no_grad():
                if isinstance(model, UNetModel) or isinstance(model, nn.Module):
                    # For PyTorch models
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    model.eval()
                    
                    # Convert to tensor and add batch dimension
                    if len(processed_image.shape) == 3:
                        # Add channel dimension for 3D data
                        processed_image = processed_image.reshape(1, 1, *processed_image.shape)
                    elif len(processed_image.shape) == 4:
                        # Already has channel dimension
                        processed_image = processed_image.reshape(1, *processed_image.shape)
                    
                    input_tensor = torch.from_numpy(processed_image).float().to(device)
                    outputs = model(input_tensor)
                    
                    # Process model outputs for each structure
                    if isinstance(outputs, dict):
                        # Multi-structure output format
                        for struct_name, output_tensor in outputs.items():
                            if struct_name in structures:
                                prob_map = output_tensor.cpu().numpy().squeeze()
                                mask = (prob_map >= threshold).astype(np.uint8)
                                mask = self._postprocess_mask(mask)
                                
                                # Resize back to original dimensions if necessary
                                if mask.shape != image_data.shape:
                                    mask = self._resize_to_original(mask, image_data.shape)
                                
                                result_masks[struct_name] = mask
                    elif isinstance(outputs, torch.Tensor):
                        # Multi-channel output format (one channel per structure)
                        outputs_np = outputs.cpu().numpy().squeeze()
                        
                        # For multi-class segmentation, each channel represents one structure
                        for i, struct_name in enumerate(structures):
                            if i < outputs_np.shape[0]:  # Make sure we don't exceed number of output channels
                                prob_map = outputs_np[i]
                                mask = (prob_map >= threshold).astype(np.uint8)
                                mask = self._postprocess_mask(mask)
                                
                                # Resize back to original dimensions if necessary
                                if mask.shape != image_data.shape:
                                    mask = self._resize_to_original(mask, image_data.shape)
                                
                                result_masks[struct_name] = mask
                else:
                    # For other model types (e.g., scikit-learn, TensorFlow)
                    # The implementation depends on the specific model interface
                    logger.error("Unsupported model type")
                    raise ValidationError("Unsupported model type")
            
            return result_masks
            
        except Exception as e:
            logger.error(f"Error segmenting OARs: {str(e)}")
            raise ValidationError(f"Error segmenting OARs: {str(e)}")
    
    def segment_single_oar(self, image_data: np.ndarray, structure_name: str,
                         site: str = 'auto', spacing: Optional[Tuple[float, float, float]] = None,
                         threshold: float = 0.5) -> np.ndarray:
        """
        Segment a single organ-at-risk from image data.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data (CT, MRI, etc.)
        structure_name : str
            Name of the structure to segment
        site : str, optional
            Anatomical site ('brain', 'head_neck', 'thorax', 'abdomen', 'pelvis', or 'auto')
        spacing : Tuple[float, float, float], optional
            Image spacing in mm
        threshold : float, optional
            Probability threshold for segmentation
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented structure
        """
        # Use the more general method but for a single structure
        result = self.segment_oars(
            image_data=image_data,
            site=site,
            spacing=spacing,
            threshold=threshold,
            structures=[structure_name]
        )
        
        if structure_name in result:
            return result[structure_name]
        else:
            logger.warning(f"Structure {structure_name} could not be segmented")
            return np.zeros_like(image_data, dtype=np.uint8)
    
    def refine_segmentation(self, image_data: np.ndarray, mask: np.ndarray,
                          structure_name: str, refinement_method: str = 'boundary') -> np.ndarray:
        """
        Refine an existing segmentation.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data (CT, MRI, etc.)
        mask : np.ndarray
            Binary mask of initial segmentation
        structure_name : str
            Name of the structure being segmented
        refinement_method : str, optional
            Method for refinement ('boundary', 'active_contour', 'graph_cut')
            
        Returns
        -------
        np.ndarray
            Refined binary mask
        """
        try:
            refined_mask = mask.copy()
            
            if refinement_method == 'boundary':
                # Simple boundary refinement based on gradient
                gradient = ndimage.gaussian_gradient_magnitude(image_data, sigma=1)
                
                # Apply gradient-based refinement at the boundary
                dilated = ndimage.binary_dilation(mask)
                eroded = ndimage.binary_erosion(mask)
                boundary = np.logical_and(dilated, np.logical_not(eroded))
                
                # Adjust boundary based on gradient
                boundary_indices = np.where(boundary)
                for i in range(len(boundary_indices[0])):
                    z, y, x = boundary_indices[0][i], boundary_indices[1][i], boundary_indices[2][i]
                    
                    # Check gradient value
                    if gradient[z, y, x] > np.mean(gradient) + np.std(gradient):
                        refined_mask[z, y, x] = 0
                    else:
                        refined_mask[z, y, x] = 1
                
            elif refinement_method == 'active_contour':
                # Active contour refinement (simplified)
                from skimage import segmentation
                
                # Process each slice
                for z in range(image_data.shape[0]):
                    slice_data = image_data[z]
                    slice_mask = mask[z]
                    
                    if np.any(slice_mask):
                        # Normalize slice for active contour
                        slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
                        
                        # Apply active contour
                        refined_slice = segmentation.active_contour(
                            slice_norm,
                            initial_contour=slice_mask,
                            alpha=0.01,
                            beta=0.1,
                            gamma=0.001
                        )
                        
                        refined_mask[z] = refined_slice
            
            elif refinement_method == 'graph_cut':
                # Graph-cut refinement (simplified)
                try:
                    from skimage import segmentation, future
                    
                    # Process the volume with graph cuts
                    # This is a simple implementation - in practice, this would be more sophisticated
                    sigma = 1.0
                    
                    # Create graph weights
                    edges = future.graph.rag_boundary(image_data, mask)
                    refined_mask = future.graph.cut_normalized(mask, edges)
                    
                except ImportError:
                    logger.warning("Graph-cut refinement requires scikit-image future module. Using original mask.")
            
            else:
                logger.warning(f"Unknown refinement method: {refinement_method}. Using original mask.")
            
            # Ensure mask is binary
            refined_mask = refined_mask.astype(np.bool).astype(np.uint8)
            
            return refined_mask
            
        except Exception as e:
            logger.error(f"Error refining segmentation: {str(e)}")
            return mask  # Return original mask on error


def create_oar_segmentor(model_path: Optional[str] = None) -> OARSegmentor:
    """
    Create and return an OARSegmentor instance.
    
    Parameters
    ----------
    model_path : str, optional
        Path to pretrained models
        
    Returns
    -------
    OARSegmentor
        Initialized OAR segmentor
    """
    return OARSegmentor(model_path=model_path)
