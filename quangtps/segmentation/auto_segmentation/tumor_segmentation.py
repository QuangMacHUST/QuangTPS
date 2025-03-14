#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tumor Segmentation Module for QuangTPS.

This module provides functionality for automatic tumor segmentation
using deep learning models, specifically focused on GTV (Gross Tumor Volume)
and CTV (Clinical Target Volume) delineation.
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
from quangtps.segmentation.auto_segmentation.unet import UNet
from quangtps.segmentation.structures.structure_set import StructureSet
from quangtps.segmentation.structures.structure_library import Structure

logger = logging.getLogger(__name__)


class TumorSegmentor:
    """
    Class for automatic tumor segmentation.
    
    This class provides methods to automatically segment tumors (GTV, CTV)
    using various deep learning models and techniques.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize tumor segmentor.
        
        Parameters
        ----------
        model_path : str, optional
            Path to pretrained model. If None, default models will be used.
        """
        self.model_loader = ModelLoader()
        self.model_path = model_path
        self.models = {}
        self._load_default_models()
    
    def _load_default_models(self):
        """Load default tumor segmentation models."""
        try:
            # Load GTV segmentation model
            if self.model_path is not None and os.path.exists(self.model_path):
                model_dir = self.model_path
            else:
                # Use default model path
                model_dir = os.path.join(os.path.dirname(__file__), 'models')
            
            # Define model paths for different tumor sites
            model_paths = {
                'lung': os.path.join(model_dir, 'lung_gtv_model.pth'),
                'brain': os.path.join(model_dir, 'brain_gtv_model.pth'),
                'h_n': os.path.join(model_dir, 'head_neck_gtv_model.pth'),
                'prostate': os.path.join(model_dir, 'prostate_gtv_model.pth'),
                'liver': os.path.join(model_dir, 'liver_gtv_model.pth')
            }
            
            # Load available models
            for site, path in model_paths.items():
                if os.path.exists(path):
                    logger.info(f"Loading {site} tumor segmentation model from {path}")
                    self.models[site] = self.model_loader.load_model(path)
                else:
                    logger.warning(f"Model for {site} not found at {path}")
            
        except Exception as e:
            logger.error(f"Error loading default tumor segmentation models: {str(e)}")
    
    def segment_gtv(self, image_data: np.ndarray, site: str = 'auto', 
                  spacing: Tuple[float, float, float] = None, 
                  threshold: float = 0.5) -> np.ndarray:
        """
        Segment Gross Tumor Volume (GTV) from image data.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data (CT, MRI, PET, etc.)
        site : str, optional
            Anatomical site ('lung', 'brain', 'h_n', 'prostate', 'liver', or 'auto')
        spacing : Tuple[float, float, float], optional
            Image spacing in mm
        threshold : float, optional
            Probability threshold for segmentation
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented GTV
        """
        try:
            # Determine anatomical site if auto
            if site == 'auto':
                site = self._determine_anatomical_site(image_data)
            
            # Check if model for the specified site is available
            if site not in self.models:
                available_sites = list(self.models.keys())
                if len(available_sites) == 0:
                    raise ValidationError(f"No tumor segmentation models available")
                logger.warning(f"Model for {site} not available. Using {available_sites[0]} model instead.")
                site = available_sites[0]
            
            # Prepare image data for model input
            processed_image = self._preprocess_image(image_data, spacing)
            
            # Run model inference
            model = self.models[site]
            with torch.no_grad():
                if isinstance(model, UNet) or isinstance(model, nn.Module):
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
                    output = model(input_tensor)
                    
                    # Convert output to numpy and apply threshold
                    if isinstance(output, torch.Tensor):
                        prob_map = output.cpu().numpy().squeeze()
                    else:
                        prob_map = output.squeeze()
                else:
                    # For other model types (e.g., scikit-learn, TensorFlow)
                    prob_map = model.predict(processed_image)
            
            # Apply threshold to get binary mask
            mask = (prob_map >= threshold).astype(np.uint8)
            
            # Post-process mask (remove small objects, fill holes)
            mask = self._postprocess_mask(mask)
            
            # Resize back to original dimensions if necessary
            if mask.shape != image_data.shape:
                mask = self._resize_to_original(mask, image_data.shape)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error segmenting GTV: {str(e)}")
            raise ValidationError(f"Error segmenting GTV: {str(e)}")
    
    def segment_ctv(self, image_data: np.ndarray, gtv_mask: np.ndarray, 
                   site: str = 'auto', margin_mm: float = 5.0,
                   spacing: Tuple[float, float, float] = None) -> np.ndarray:
        """
        Segment Clinical Target Volume (CTV) from GTV and image data.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data (CT, MRI, PET, etc.)
        gtv_mask : np.ndarray
            Binary mask of GTV
        site : str, optional
            Anatomical site ('lung', 'brain', 'h_n', 'prostate', 'liver', or 'auto')
        margin_mm : float, optional
            Default margin in mm to expand from GTV to CTV if no model is available
        spacing : Tuple[float, float, float], optional
            Image spacing in mm
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented CTV
        """
        try:
            # Determine anatomical site if auto
            if site == 'auto':
                site = self._determine_anatomical_site(image_data)
            
            # Check if we have a specific CTV model for this site
            ctv_model_key = f"{site}_ctv"
            if ctv_model_key in self.models:
                # Use dedicated CTV segmentation model
                processed_image = self._preprocess_image(image_data, spacing)
                
                # Combine original image with GTV mask for better context
                if len(processed_image.shape) == 3:
                    # Add GTV mask as an additional channel
                    combined_input = np.stack([processed_image, gtv_mask], axis=0)
                else:
                    # Already has channel dimension
                    combined_input = np.concatenate([processed_image, gtv_mask[np.newaxis, ...]], axis=0)
                
                # Run model inference
                model = self.models[ctv_model_key]
                with torch.no_grad():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    model.eval()
                    
                    # Add batch dimension
                    combined_input = combined_input.reshape(1, *combined_input.shape)
                    input_tensor = torch.from_numpy(combined_input).float().to(device)
                    output = model(input_tensor)
                    
                    # Convert output to numpy and apply threshold
                    prob_map = output.cpu().numpy().squeeze()
                
                # Apply threshold to get binary mask
                ctv_mask = (prob_map >= 0.5).astype(np.uint8)
                
            else:
                # Fallback to rule-based CTV generation
                logger.info(f"No specific CTV model for {site}. Using rule-based expansion.")
                ctv_mask = self._expand_gtv_to_ctv(gtv_mask, site, margin_mm, spacing)
            
            # Post-process mask (remove small objects, fill holes)
            ctv_mask = self._postprocess_mask(ctv_mask)
            
            # Resize back to original dimensions if necessary
            if ctv_mask.shape != image_data.shape:
                ctv_mask = self._resize_to_original(ctv_mask, image_data.shape)
            
            return ctv_mask
            
        except Exception as e:
            logger.error(f"Error segmenting CTV: {str(e)}")
            raise ValidationError(f"Error segmenting CTV: {str(e)}")
    
    def segment_ptv(self, ctv_mask: np.ndarray, 
                   margin_mm: float = 5.0,
                   spacing: Tuple[float, float, float] = None) -> np.ndarray:
        """
        Generate Planning Target Volume (PTV) from CTV.
        
        Parameters
        ----------
        ctv_mask : np.ndarray
            Binary mask of CTV
        margin_mm : float, optional
            Margin in mm to expand from CTV to PTV
        spacing : Tuple[float, float, float], optional
            Image spacing in mm
            
        Returns
        -------
        np.ndarray
            Binary mask of generated PTV
        """
        try:
            # PTV is usually a geometric expansion of CTV
            ptv_mask = self._expand_volume(ctv_mask, margin_mm, spacing)
            
            return ptv_mask
            
        except Exception as e:
            logger.error(f"Error generating PTV: {str(e)}")
            raise ValidationError(f"Error generating PTV: {str(e)}")
    
    def create_tumor_structure_set(self, image_data: np.ndarray, site: str = 'auto',
                                 spacing: Tuple[float, float, float] = None,
                                 origin: Tuple[float, float, float] = None) -> StructureSet:
        """
        Create a complete tumor structure set with GTV, CTV, and PTV.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data (CT, MRI, PET, etc.)
        site : str, optional
            Anatomical site
        spacing : Tuple[float, float, float], optional
            Image spacing in mm
        origin : Tuple[float, float, float], optional
            Image origin coordinates
            
        Returns
        -------
        StructureSet
            Structure set containing GTV, CTV, and PTV
        """
        try:
            # Create empty structure set
            structure_set = StructureSet()
            
            # Segment GTV
            gtv_mask = self.segment_gtv(image_data, site, spacing)
            
            # Create GTV structure
            gtv = Structure(
                id=f"GTV_{site}" if site != 'auto' else "GTV",
                name=f"GTV_{site}" if site != 'auto' else "GTV",
                mask=gtv_mask,
                color=[255, 0, 0],  # Red
                type="GTV",
                spacing=spacing,
                origin=origin
            )
            
            # Add GTV to structure set
            structure_set.add_structure(gtv)
            
            # Segment CTV
            ctv_mask = self.segment_ctv(image_data, gtv_mask, site, spacing=spacing)
            
            # Create CTV structure
            ctv = Structure(
                id=f"CTV_{site}" if site != 'auto' else "CTV",
                name=f"CTV_{site}" if site != 'auto' else "CTV",
                mask=ctv_mask,
                color=[255, 165, 0],  # Orange
                type="CTV",
                spacing=spacing,
                origin=origin
            )
            
            # Add CTV to structure set
            structure_set.add_structure(ctv)
            
            # Generate PTV
            ptv_mask = self.segment_ptv(ctv_mask, spacing=spacing)
            
            # Create PTV structure
            ptv = Structure(
                id=f"PTV_{site}" if site != 'auto' else "PTV",
                name=f"PTV_{site}" if site != 'auto' else "PTV",
                mask=ptv_mask,
                color=[255, 255, 0],  # Yellow
                type="PTV",
                spacing=spacing,
                origin=origin
            )
            
            # Add PTV to structure set
            structure_set.add_structure(ptv)
            
            return structure_set
            
        except Exception as e:
            logger.error(f"Error creating tumor structure set: {str(e)}")
            raise ValidationError(f"Error creating tumor structure set: {str(e)}")
    
    def _determine_anatomical_site(self, image_data: np.ndarray) -> str:
        """
        Automatically determine anatomical site from image data.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
            
        Returns
        -------
        str
            Predicted anatomical site
        """
        # This is a simplified approach. In a real system, this would be
        # a more sophisticated classifier.
        
        # Example implementation based on image dimensions and intensity statistics
        shape = image_data.shape
        aspect_ratio = shape[0] / shape[1]
        
        # Get image statistics
        mean_intensity = np.mean(image_data)
        std_intensity = np.std(image_data)
        
        # Simple heuristic rules (these should be replaced with a proper classifier)
        if shape[0] < 200 and aspect_ratio > 0.9 and aspect_ratio < 1.1:
            return 'brain'
        elif shape[0] > 300 and shape[2] > 100:
            if mean_intensity < -200:  # Mostly air, likely chest
                return 'lung'
            else:
                return 'h_n'
        else:
            # Default to the most common site or the one with the most robust model
            return 'lung'
    
    def _preprocess_image(self, image_data: np.ndarray, 
                         spacing: Tuple[float, float, float] = None) -> np.ndarray:
        """
        Preprocess image data for model input.
        
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
        # Convert to float32
        processed = image_data.astype(np.float32)
        
        # Normalize intensity (for CT images, commonly use HU window)
        # Example: window for soft tissue: center=40, width=400
        # This maps HU values in [center-width/2, center+width/2] to [0, 1]
        center, width = 40, 400
        processed = np.clip(processed, center - width/2, center + width/2)
        processed = (processed - (center - width/2)) / width
        
        # Resample to isotropic spacing if needed and if spacing is provided
        if spacing is not None and (spacing[0] != spacing[1] or spacing[0] != spacing[2]):
            # Create SimpleITK image
            sitk_image = sitk.GetImageFromArray(processed)
            sitk_image.SetSpacing(spacing)
            
            # Define target isotropic spacing
            target_spacing = (1.0, 1.0, 1.0)  # 1mm isotropic
            
            # Calculate new size
            new_size = [int(round(processed.shape[i] * spacing[i] / target_spacing[i])) for i in range(3)]
            
            # Resample
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing)
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
        # Remove small objects
        labeled, num_objects = ndimage.label(mask)
        if num_objects > 0:
            # Calculate object sizes
            sizes = np.bincount(labeled.ravel())[1:]
            
            # Keep only the largest object if multiple objects exist
            if len(sizes) > 1:
                max_label = np.argmax(sizes) + 1
                mask = (labeled == max_label).astype(np.uint8)
        
        # Fill holes
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        
        return mask
    
    def _resize_to_original(self, mask: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Resize mask back to original image dimensions.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask to resize
        original_shape : Tuple[int, ...]
            Original image shape
            
        Returns
        -------
        np.ndarray
            Resized mask
        """
        if mask.shape == original_shape:
            return mask
        
        # Create SimpleITK images
        sitk_mask = sitk.GetImageFromArray(mask)
        
        # Calculate spacing to match original shape
        spacing = tuple([mask.shape[i] / original_shape[i] for i in range(3)])
        sitk_mask.SetSpacing(spacing)
        
        # Define target spacing and size
        target_spacing = (1.0, 1.0, 1.0)
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(original_shape[::-1])  # SimpleITK uses (x,y,z) order
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for masks
        resampled_mask = resampler.Execute(sitk_mask)
        
        # Convert back to numpy
        resized_mask = sitk.GetArrayFromImage(resampled_mask)
        
        # Ensure binary
        resized_mask = (resized_mask > 0.5).astype(np.uint8)
        
        return resized_mask
    
    def _expand_gtv_to_ctv(self, gtv_mask: np.ndarray, site: str, 
                          margin_mm: float, spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Expand GTV to CTV using site-specific rules.
        
        Parameters
        ----------
        gtv_mask : np.ndarray
            Binary mask of GTV
        site : str
            Anatomical site
        margin_mm : float
            Default margin in mm
        spacing : Tuple[float, float, float]
            Image spacing in mm
            
        Returns
        -------
        np.ndarray
            CTV mask
        """
        # Site-specific CTV margins based on clinical guidelines
        site_margins = {
            'lung': 8.0,  # 8mm for lung
            'brain': 2.0,  # 2mm for brain
            'h_n': 5.0,   # 5mm for head and neck
            'prostate': 7.0,  # 7mm for prostate
            'liver': 5.0   # 5mm for liver
        }
        
        # Use site-specific margin if available, otherwise use default
        margin = site_margins.get(site, margin_mm)
        
        # Expand GTV to create CTV
        ctv_mask = self._expand_volume(gtv_mask, margin, spacing)
        
        return ctv_mask
    
    def _expand_volume(self, mask: np.ndarray, margin_mm: float, 
                      spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Expand a binary mask by a specified margin.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask to expand
        margin_mm : float
            Margin in mm
        spacing : Tuple[float, float, float]
            Image spacing in mm
            
        Returns
        -------
        np.ndarray
            Expanded mask
        """
        if spacing is None:
            # Default to 1mm isotropic if spacing not provided
            spacing = (1.0, 1.0, 1.0)
        
        # Calculate margin in voxels for each dimension
        margin_voxels = [int(round(margin_mm / spacing[i])) for i in range(3)]
        
        # Use binary dilation for expansion
        struct_elem = ndimage.generate_binary_structure(3, 1)  # Basic 3D connectivity
        expanded_mask = ndimage.binary_dilation(
            mask, 
            structure=struct_elem, 
            iterations=max(margin_voxels)
        ).astype(np.uint8)
        
        return expanded_mask


# Function to create and return an instance of TumorSegmentor
def create_tumor_segmentor(model_path: Optional[str] = None) -> TumorSegmentor:
    """
    Create and return a TumorSegmentor instance.
    
    Parameters
    ----------
    model_path : str, optional
        Path to pretrained models
        
    Returns
    -------
    TumorSegmentor
        Initialized tumor segmentor
    """
    return TumorSegmentor(model_path)
