#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Segmentation Refinement for QuangTPS.

This module provides tools for refining and improving segmentation results,
including post-processing, smoothing, and gap-filling methods.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import SimpleITK as sitk
from scipy import ndimage
import cv2

from quangtps.core.exceptions import ValidationError
from quangtps.segmentation.structures.structure_library import Structure

logger = logging.getLogger(__name__)


class SegmentationRefinement:
    """
    Class for refining segmentation results.
    
    This class provides methods for post-processing and refining
    segmentation masks to improve quality and accuracy.
    """
    
    def __init__(self):
        """Initialize segmentation refinement tools."""
        pass
    
    def smooth_contours(self, mask: np.ndarray, 
                      smooth_sigma: float = 0.5, 
                      slice_by_slice: bool = True) -> np.ndarray:
        """
        Smooth contours of a segmentation mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Input segmentation mask
        smooth_sigma : float, optional
            Sigma parameter for Gaussian smoothing (higher = smoother)
        slice_by_slice : bool, optional
            Whether to process each slice separately
            
        Returns
        -------
        np.ndarray
            Smoothed segmentation mask
        """
        if mask.ndim < 2:
            raise ValidationError("Mask must have at least 2 dimensions")
        
        # Ensure we're working with binary masks
        binary_mask = (mask > 0).astype(np.float32)
        
        if slice_by_slice and mask.ndim > 2:
            # Process each slice separately
            smoothed_mask = np.zeros_like(binary_mask)
            
            for i in range(binary_mask.shape[0]):
                slice_mask = binary_mask[i]
                
                # Skip empty slices
                if np.sum(slice_mask) == 0:
                    continue
                
                # Apply Gaussian smoothing
                smoothed_slice = ndimage.gaussian_filter(slice_mask, sigma=smooth_sigma)
                
                # Threshold to re-binarize (typically 0.5 for binary smoothing)
                smoothed_slice = (smoothed_slice > 0.5).astype(np.float32)
                
                # Store smoothed slice
                smoothed_mask[i] = smoothed_slice
        else:
            # Process entire volume at once
            smoothed_mask = ndimage.gaussian_filter(binary_mask, sigma=smooth_sigma)
            smoothed_mask = (smoothed_mask > 0.5).astype(np.float32)
        
        # Return binary mask with original data type
        return smoothed_mask.astype(mask.dtype)
    
    def remove_small_objects(self, mask: np.ndarray, 
                           min_size: int = 10, 
                           connectivity: int = 1) -> np.ndarray:
        """
        Remove small isolated objects from a segmentation mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Input segmentation mask
        min_size : int, optional
            Minimum size (in voxels) of objects to keep
        connectivity : int, optional
            Connectivity for determining connected components
            
        Returns
        -------
        np.ndarray
            Refined segmentation mask
        """
        # Ensure we're working with binary masks
        binary_mask = (mask > 0).astype(np.bool_)
        
        # Label connected components
        labeled_mask, num_features = ndimage.label(binary_mask, 
                                                 structure=ndimage.generate_binary_structure(mask.ndim, connectivity))
        
        # Count size of each component
        component_sizes = np.bincount(labeled_mask.ravel())
        
        # Skip background (index 0)
        component_sizes[0] = 0
        
        # Create mask of components to remove
        remove_mask = component_sizes < min_size
        remove_indices = remove_mask[labeled_mask]
        
        # Remove small components
        filtered_mask = np.logical_not(remove_indices).astype(mask.dtype)
        
        # Preserve original values where mask is still True
        result = np.zeros_like(mask)
        result[filtered_mask > 0] = mask[filtered_mask > 0]
        
        return result
    
    def fill_holes(self, mask: np.ndarray, 
                 slice_by_slice: bool = True) -> np.ndarray:
        """
        Fill holes in a segmentation mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Input segmentation mask
        slice_by_slice : bool, optional
            Whether to process each slice separately
            
        Returns
        -------
        np.ndarray
            Refined segmentation mask
        """
        # Ensure we're working with binary masks
        binary_mask = (mask > 0).astype(np.bool_)
        
        if slice_by_slice and mask.ndim > 2:
            # Process each slice separately
            filled_mask = np.zeros_like(binary_mask)
            
            for i in range(binary_mask.shape[0]):
                slice_mask = binary_mask[i]
                
                # Skip empty slices
                if np.sum(slice_mask) == 0:
                    continue
                
                # Fill holes
                filled_slice = ndimage.binary_fill_holes(slice_mask)
                
                # Store filled slice
                filled_mask[i] = filled_slice
        else:
            # Process entire volume at once
            filled_mask = ndimage.binary_fill_holes(binary_mask)
        
        # Preserve original values where mask is still True
        result = np.zeros_like(mask)
        result[filled_mask > 0] = mask[filled_mask > 0]
        
        # If there are new voxels from filling, use the mode value from the mask
        new_voxels = np.logical_and(filled_mask, np.logical_not(binary_mask))
        if np.any(new_voxels):
            # Use mode of non-zero values if available
            mask_values = mask[mask > 0]
            if len(mask_values) > 0:
                mode_value = np.bincount(mask_values.flatten()).argmax()
                result[new_voxels] = mode_value
            else:
                result[new_voxels] = 1  # Default value if no mode available
        
        return result
    
    def morphological_operations(self, mask: np.ndarray, 
                              operation: str = 'dilate', 
                              size: int = 1,
                              slice_by_slice: bool = True) -> np.ndarray:
        """
        Apply morphological operations to a segmentation mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Input segmentation mask
        operation : str, optional
            Type of operation: 'dilate', 'erode', 'open', 'close'
        size : int, optional
            Size of structuring element
        slice_by_slice : bool, optional
            Whether to process each slice separately
            
        Returns
        -------
        np.ndarray
            Processed segmentation mask
        """
        # Ensure we're working with binary masks
        binary_mask = (mask > 0).astype(np.bool_)
        
        # Create structuring element
        if mask.ndim == 3 and not slice_by_slice:
            # 3D structuring element for volumetric processing
            struct_element = ndimage.generate_binary_structure(3, 1)
            if size > 1:
                struct_element = ndimage.iterate_structure(struct_element, size)
        else:
            # 2D structuring element for slice-by-slice processing
            struct_element = ndimage.generate_binary_structure(2, 1)
            if size > 1:
                struct_element = ndimage.iterate_structure(struct_element, size)
        
        # Apply the selected operation
        if slice_by_slice and mask.ndim > 2:
            # Process each slice separately
            result_mask = np.zeros_like(binary_mask)
            
            for i in range(binary_mask.shape[0]):
                slice_mask = binary_mask[i]
                
                # Skip empty slices for some operations
                if operation in ['erode', 'open'] and np.sum(slice_mask) == 0:
                    continue
                
                # Apply operation
                if operation == 'dilate':
                    processed_slice = ndimage.binary_dilation(slice_mask, struct_element)
                elif operation == 'erode':
                    processed_slice = ndimage.binary_erosion(slice_mask, struct_element)
                elif operation == 'open':
                    processed_slice = ndimage.binary_opening(slice_mask, struct_element)
                elif operation == 'close':
                    processed_slice = ndimage.binary_closing(slice_mask, struct_element)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                # Store processed slice
                result_mask[i] = processed_slice
        else:
            # Process entire volume at once
            if operation == 'dilate':
                result_mask = ndimage.binary_dilation(binary_mask, struct_element)
            elif operation == 'erode':
                result_mask = ndimage.binary_erosion(binary_mask, struct_element)
            elif operation == 'open':
                result_mask = ndimage.binary_opening(binary_mask, struct_element)
            elif operation == 'close':
                result_mask = ndimage.binary_closing(binary_mask, struct_element)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        # Preserve original values where result mask is True and original was True
        preserved_mask = np.zeros_like(mask)
        preserved_mask[np.logical_and(result_mask, binary_mask)] = mask[np.logical_and(result_mask, binary_mask)]
        
        # For new voxels (True in result but False in original), use mode value or default
        new_voxels = np.logical_and(result_mask, np.logical_not(binary_mask))
        if np.any(new_voxels):
            # Use mode of non-zero values if available
            mask_values = mask[mask > 0]
            if len(mask_values) > 0:
                mode_value = np.bincount(mask_values.flatten()).argmax()
                preserved_mask[new_voxels] = mode_value
            else:
                preserved_mask[new_voxels] = 1  # Default value if no mode available
        
        return preserved_mask
    
    def auto_refinement(self, mask: np.ndarray, 
                      spacing: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Apply automatic refinement to a segmentation mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Input segmentation mask
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm, used to adjust parameters
            
        Returns
        -------
        np.ndarray
            Refined segmentation mask
        """
        # Set default spacing if not provided
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        
        # Calculate median spacing for parameter adjustment
        median_spacing = np.median(spacing)
        
        # Adjust parameters based on spacing
        min_object_size = int(10 * (1.0 / median_spacing)**3)  # Scale with voxel volume
        smooth_sigma = 0.5 * median_spacing  # Scale with spacing
        morph_size = max(1, int(round(1.0 / median_spacing)))  # Scale with spacing
        
        # Apply a sequence of refinements
        refined_mask = mask.copy()
        
        # Step 1: Remove small isolated objects
        refined_mask = self.remove_small_objects(refined_mask, min_size=min_object_size)
        
        # Step 2: Fill holes in the mask
        refined_mask = self.fill_holes(refined_mask)
        
        # Step 3: Apply closing to connect nearby components
        refined_mask = self.morphological_operations(refined_mask, operation='close', size=morph_size)
        
        # Step 4: Smooth contours
        refined_mask = self.smooth_contours(refined_mask, smooth_sigma=smooth_sigma)
        
        return refined_mask
    
    def conditional_refinement(self, mask: np.ndarray, 
                             reference: Optional[np.ndarray] = None,
                             spacing: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Apply conditional refinement based on reference comparison.
        
        Parameters
        ----------
        mask : np.ndarray
            Input segmentation mask
        reference : np.ndarray, optional
            Reference (ground truth) mask
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm
            
        Returns
        -------
        np.ndarray
            Refined segmentation mask
        """
        # If no reference is provided, fall back to auto refinement
        if reference is None:
            return self.auto_refinement(mask, spacing)
        
        # Ensure masks have the same shape
        if mask.shape != reference.shape:
            logger.warning("Mask and reference shapes do not match, falling back to auto refinement")
            return self.auto_refinement(mask, spacing)
        
        # Set default spacing if not provided
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        
        # Compute metrics to guide the refinement process
        from quangtps.segmentation.validation.metrics import SegmentationMetrics
        metrics = SegmentationMetrics()
        dice = metrics.dice_coefficient(mask, reference)
        
        # Create binary versions for comparison
        mask_bin = mask > 0
        ref_bin = reference > 0
        
        # Initialize refined mask
        refined_mask = mask.copy()
        
        # Apply different refinement strategies based on metrics
        if dice < 0.3:
            # Poor overlap, needs significant correction
            # Combine the two masks and then refine
            logger.info("Low Dice coefficient (<0.3), applying major refinement")
            combined = np.logical_or(mask_bin, ref_bin).astype(mask.dtype)
            # Apply auto refinement to the combined mask
            refined_mask = self.auto_refinement(combined, spacing)
            # Preserve original intensity values from the mask where possible
            intensity_values = np.unique(mask[mask > 0])
            if len(intensity_values) > 0:
                # Use the most common non-zero value
                common_value = np.bincount(mask[mask > 0].flatten()).argmax()
                refined_mask[refined_mask > 0] = common_value
        
        elif dice < 0.7:
            # Moderate overlap, use structural guidance
            logger.info("Moderate Dice coefficient (0.3-0.7), applying guided refinement")
            
            # Find false negatives (in reference but not in mask)
            false_negatives = np.logical_and(ref_bin, np.logical_not(mask_bin))
            
            # Find false positives (in mask but not in reference)
            false_positives = np.logical_and(mask_bin, np.logical_not(ref_bin))
            
            # Dilate the mask to capture nearby false negatives
            dilated = self.morphological_operations(mask, operation='dilate', size=2)
            dilated_bin = dilated > 0
            
            # Recover false negatives that are within the dilated region
            recoverable_fn = np.logical_and(false_negatives, dilated_bin)
            
            # Remove isolated false positives
            fp_removed = self.remove_small_objects(false_positives.astype(np.int32), min_size=20)
            
            # Combine the results
            refined_bin = np.logical_or(mask_bin, recoverable_fn)
            refined_bin = np.logical_and(refined_bin, np.logical_not(fp_removed))
            
            # Fill holes and smooth
            refined_bin = ndimage.binary_fill_holes(refined_bin)
            refined_bin = self.smooth_contours(refined_bin.astype(np.int32))
            
            # Convert back to mask with original intensity values
            refined_mask = np.zeros_like(mask)
            if np.any(refined_bin):
                intensity_values = np.unique(mask[mask > 0])
                if len(intensity_values) > 0:
                    # Use the most common non-zero value
                    common_value = np.bincount(mask[mask > 0].flatten()).argmax()
                    refined_mask[refined_bin > 0] = common_value
                else:
                    refined_mask[refined_bin > 0] = 1
        
        else:
            # Good overlap, just minor refinements
            logger.info("Good Dice coefficient (>0.7), applying minor refinement")
            # Fill holes 
            refined_mask = self.fill_holes(mask)
            # Smooth contours slightly
            refined_mask = self.smooth_contours(refined_mask, smooth_sigma=0.3)
        
        return refined_mask
    
    def refine_structure(self, structure: Structure, 
                       reference: Optional[Structure] = None,
                       method: str = 'auto') -> Structure:
        """
        Refine a structure and return a new refined structure.
        
        Parameters
        ----------
        structure : Structure
            Structure to refine
        reference : Structure, optional
            Reference structure for guided refinement
        method : str, optional
            Refinement method: 'auto', 'smooth', 'fill', 'close', 'conditional'
            
        Returns
        -------
        Structure
            Refined structure
        """
        # Get masks and spacing
        mask = structure.mask
        spacing = structure.spacing
        
        ref_mask = None
        if reference is not None:
            ref_mask = reference.mask
        
        # Apply selected refinement method
        if method == 'auto':
            refined_mask = self.auto_refinement(mask, spacing)
        elif method == 'smooth':
            refined_mask = self.smooth_contours(mask)
        elif method == 'fill':
            refined_mask = self.fill_holes(mask)
        elif method == 'close':
            refined_mask = self.morphological_operations(mask, operation='close')
        elif method == 'conditional' and ref_mask is not None:
            refined_mask = self.conditional_refinement(mask, ref_mask, spacing)
        else:
            refined_mask = self.auto_refinement(mask, spacing)
        
        # Create a new structure with refined mask
        refined_structure = structure.copy()
        refined_structure.mask = refined_mask
        refined_structure.name = f"{structure.name}_refined"
        
        return refined_structure
    
    def batch_refine_structures(self, structure_set,
                              reference_set = None,
                              method: str = 'auto') -> Any:
        """
        Refine multiple structures in a structure set.
        
        Parameters
        ----------
        structure_set : StructureSet or List[Structure]
            Structure set or list of structures to refine
        reference_set : StructureSet or List[Structure], optional
            Reference structure set or list for guided refinement
        method : str, optional
            Refinement method: 'auto', 'smooth', 'fill', 'close', 'conditional'
            
        Returns
        -------
        StructureSet or List[Structure]
            Refined structure set or list
        """
        # Handle different input types
        if hasattr(structure_set, 'get_all_structures'):
            structures = structure_set.get_all_structures()
            is_structure_set = True
        else:
            structures = structure_set
            is_structure_set = False
        
        # Create reference mapping if provided
        ref_map = {}
        if reference_set is not None:
            if hasattr(reference_set, 'get_all_structures'):
                ref_structures = reference_set.get_all_structures()
            else:
                ref_structures = reference_set
            
            # Map by ID
            ref_map = {struct.id: struct for struct in ref_structures}
        
        # Refine each structure
        refined_structures = []
        for structure in structures:
            # Find matching reference structure if available
            ref = ref_map.get(structure.id) if ref_map else None
            
            # Refine structure
            refined = self.refine_structure(structure, ref, method)
            refined_structures.append(refined)
        
        # Return in the same format as input
        if is_structure_set:
            from quangtps.segmentation.structures.structure_set import StructureSet
            return StructureSet(structures=refined_structures)
        else:
            return refined_structures
