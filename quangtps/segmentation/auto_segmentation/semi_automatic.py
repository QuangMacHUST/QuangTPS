#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semi-Automatic Segmentation Module for QuangTPS.

This module provides functionality for semi-automatic segmentation using
various algorithms such as thresholding, region growing, and watershed.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import ndimage
from skimage import measure, filters, segmentation, morphology
import SimpleITK as sitk

from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ThresholdSegmenter:
    """
    Class for threshold-based segmentation.
    
    This class implements various thresholding techniques for
    semi-automatic image segmentation.
    """
    
    def __init__(self):
        """Initialize threshold segmenter."""
        pass
    
    def simple_threshold(self, image_data: np.ndarray, lower_threshold: float, 
                       upper_threshold: float = None) -> np.ndarray:
        """
        Apply simple threshold segmentation.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        lower_threshold : float
            Lower intensity threshold
        upper_threshold : float, optional
            Upper intensity threshold. If None, segment all values >= lower_threshold
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        if upper_threshold is None:
            mask = image_data >= lower_threshold
        else:
            mask = (image_data >= lower_threshold) & (image_data <= upper_threshold)
        
        return mask.astype(np.uint8)
    
    def otsu_threshold(self, image_data: np.ndarray, 
                     slice_by_slice: bool = True) -> np.ndarray:
        """
        Apply Otsu's thresholding method.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        slice_by_slice : bool, optional
            Whether to apply Otsu's method slice by slice
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        mask = np.zeros_like(image_data, dtype=np.uint8)
        
        if slice_by_slice:
            # Apply Otsu's method to each slice
            for z in range(image_data.shape[0]):
                slice_data = image_data[z]
                # Skip empty slices
                if np.max(slice_data) == np.min(slice_data):
                    continue
                
                # Apply Otsu's method
                threshold = filters.threshold_otsu(slice_data)
                mask[z] = (slice_data >= threshold).astype(np.uint8)
        else:
            # Apply Otsu's method to the whole volume
            threshold = filters.threshold_otsu(image_data)
            mask = (image_data >= threshold).astype(np.uint8)
        
        return mask
    
    def adaptive_threshold(self, image_data: np.ndarray, block_size: int = 35, 
                         offset: float = 0, slice_by_slice: bool = True) -> np.ndarray:
        """
        Apply adaptive thresholding method.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        block_size : int, optional
            Size of local neighborhood for thresholding
        offset : float, optional
            Constant subtracted from weighted mean
        slice_by_slice : bool, optional
            Whether to apply adaptive threshold slice by slice
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        mask = np.zeros_like(image_data, dtype=np.uint8)
        
        if slice_by_slice:
            # Apply adaptive threshold to each slice
            for z in range(image_data.shape[0]):
                slice_data = image_data[z]
                # Skip empty slices
                if np.max(slice_data) == np.min(slice_data):
                    continue
                
                # Apply adaptive threshold
                binary_slice = filters.threshold_local(slice_data, block_size, offset=offset)
                mask[z] = (slice_data > binary_slice).astype(np.uint8)
        else:
            # Adaptive threshold for 3D is more complex, we approximate with Gaussian-weighted mean
            # This is a simplified implementation
            filtered_image = ndimage.gaussian_filter(image_data, sigma=block_size/3)
            mask = (image_data > filtered_image + offset).astype(np.uint8)
        
        return mask
    
    def multi_threshold(self, image_data: np.ndarray, 
                       num_thresholds: int = 3) -> np.ndarray:
        """
        Apply multi-Otsu thresholding method.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        num_thresholds : int, optional
            Number of thresholds to compute
            
        Returns
        -------
        np.ndarray
            Labeled image with segmentation results
        """
        # Multi-Otsu thresholding returns labeled regions, not binary masks
        result = np.zeros_like(image_data, dtype=np.uint8)
        
        # Apply Multi-Otsu slice by slice
        for z in range(image_data.shape[0]):
            slice_data = image_data[z]
            # Skip empty slices
            if np.max(slice_data) == np.min(slice_data):
                continue
            
            # Apply Multi-Otsu thresholding
            thresholds = filters.threshold_multiotsu(slice_data, classes=num_thresholds+1)
            
            # Categorize pixels based on thresholds
            regions = np.digitize(slice_data, bins=thresholds)
            result[z] = regions
        
        return result
    
    def hu_threshold(self, ct_data: np.ndarray, tissue_type: str) -> np.ndarray:
        """
        Apply threshold based on standard HU values for different tissues.
        
        Parameters
        ----------
        ct_data : np.ndarray
            CT image data in Hounsfield Units
        tissue_type : str
            Tissue type to segment ('bone', 'soft_tissue', 'fat', 'air', 'lung')
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        # Define HU ranges for different tissue types
        hu_ranges = {
            'bone': (300, 3000),
            'soft_tissue': (20, 100),
            'fat': (-100, -20),
            'air': (-1000, -950),
            'lung': (-950, -300)
        }
        
        if tissue_type not in hu_ranges:
            raise ValidationError(f"Tissue type {tissue_type} not supported")
        
        # Get thresholds for the specified tissue type
        lower, upper = hu_ranges[tissue_type]
        
        # Apply thresholding
        mask = (ct_data >= lower) & (ct_data <= upper)
        
        return mask.astype(np.uint8)
    
    def post_process(self, mask: np.ndarray, min_size: int = 100, 
                    closing_radius: int = 2) -> np.ndarray:
        """
        Post-process a binary mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask to post-process
        min_size : int, optional
            Minimum size (in voxels) for connected components
        closing_radius : int, optional
            Radius for morphological closing
            
        Returns
        -------
        np.ndarray
            Post-processed binary mask
        """
        # Morphological closing
        if closing_radius > 0:
            struct_elem = ndimage.generate_binary_structure(3, 1)
            mask = ndimage.binary_closing(mask, structure=struct_elem, iterations=closing_radius)
        
        # Remove small objects
        if min_size > 0:
            mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
        
        # Fill holes
        mask = ndimage.binary_fill_holes(mask)
        
        return mask.astype(np.uint8)


class RegionGrowingSegmenter:
    """
    Class for region growing segmentation.
    
    This class implements region growing algorithms for
    semi-automatic image segmentation.
    """
    
    def __init__(self):
        """Initialize region growing segmenter."""
        pass
    
    def region_grow(self, image_data: np.ndarray, seed_point: Tuple[int, int, int],
                  tolerance: float = 10, min_size: int = 100) -> np.ndarray:
        """
        Apply 3D region growing segmentation.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        seed_point : Tuple[int, int, int]
            Seed point (z, y, x) for region growing
        tolerance : float, optional
            Intensity tolerance for inclusion in the region
        min_size : int, optional
            Minimum size (in voxels) for the result
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        # Ensure the seed point is in bounds
        if not (0 <= seed_point[0] < image_data.shape[0] and
                0 <= seed_point[1] < image_data.shape[1] and
                0 <= seed_point[2] < image_data.shape[2]):
            raise ValidationError(f"Seed point {seed_point} is out of bounds")
        
        # Create output mask
        mask = np.zeros_like(image_data, dtype=np.uint8)
        z, y, x = seed_point
        
        # Get seed intensity
        seed_intensity = image_data[z, y, x]
        
        # Define intensity range
        lower_threshold = seed_intensity - tolerance
        upper_threshold = seed_intensity + tolerance
        
        # Create a mask of candidate voxels
        candidates = (image_data >= lower_threshold) & (image_data <= upper_threshold)
        
        # Label connected components in the candidate mask
        labeled_array, num_features = ndimage.label(candidates)
        
        # Get the label of the seed point
        seed_label = labeled_array[z, y, x]
        
        # If seed point is not in any region, return empty mask
        if seed_label == 0:
            return mask
        
        # Extract the region containing the seed point
        mask = (labeled_array == seed_label).astype(np.uint8)
        
        # Post-process the mask
        if min_size > 0:
            mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
        
        return mask.astype(np.uint8)
    
    def adaptive_region_grow(self, image_data: np.ndarray, seed_point: Tuple[int, int, int],
                           initial_tolerance: float = 10, max_tolerance: float = 50,
                           tolerance_step: float = 5, target_size: int = 1000) -> np.ndarray:
        """
        Apply adaptive region growing with increasing tolerance.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        seed_point : Tuple[int, int, int]
            Seed point (z, y, x) for region growing
        initial_tolerance : float, optional
            Initial intensity tolerance
        max_tolerance : float, optional
            Maximum intensity tolerance
        tolerance_step : float, optional
            Increment for tolerance in each iteration
        target_size : int, optional
            Target size (in voxels) for the segmented region
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        best_mask = None
        best_size_diff = float('inf')
        
        # Try different tolerance values
        for tolerance in np.arange(initial_tolerance, max_tolerance + tolerance_step, tolerance_step):
            # Perform region growing with current tolerance
            current_mask = self.region_grow(image_data, seed_point, tolerance=tolerance, min_size=0)
            
            # Calculate size of the segmented region
            current_size = np.sum(current_mask)
            
            # Check if this is closer to the target size
            size_diff = abs(current_size - target_size)
            
            if size_diff < best_size_diff:
                best_size_diff = size_diff
                best_mask = current_mask
            
            # If we exceed the target size significantly, stop
            if current_size > target_size * 2:
                break
        
        return best_mask if best_mask is not None else np.zeros_like(image_data, dtype=np.uint8)
    
    def multi_seed_region_grow(self, image_data: np.ndarray, seed_points: List[Tuple[int, int, int]],
                             tolerance: float = 10, min_size: int = 100) -> np.ndarray:
        """
        Apply region growing from multiple seed points.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        seed_points : List[Tuple[int, int, int]]
            List of seed points (z, y, x) for region growing
        tolerance : float, optional
            Intensity tolerance for inclusion in the region
        min_size : int, optional
            Minimum size (in voxels) for the result
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        # Initialize result mask
        result_mask = np.zeros_like(image_data, dtype=np.uint8)
        
        # Apply region growing from each seed point
        for seed_point in seed_points:
            current_mask = self.region_grow(image_data, seed_point, tolerance=tolerance, min_size=0)
            
            # Add to result mask
            result_mask = np.logical_or(result_mask, current_mask).astype(np.uint8)
        
        # Post-process the combined mask
        if min_size > 0:
            result_mask = morphology.remove_small_objects(result_mask.astype(bool), min_size=min_size)
        
        return result_mask.astype(np.uint8)
    
    def confidence_connected(self, image_data: np.ndarray, seed_point: Tuple[int, int, int],
                          multiplier: float = 2.5, iterations: int = 3) -> np.ndarray:
        """
        Apply confidence connected region growing using SimpleITK.
        
        This method estimates the statistics of a region using the seed point
        and includes neighboring pixels with similar intensities.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        seed_point : Tuple[int, int, int]
            Seed point (z, y, x) for region growing
        multiplier : float, optional
            Factor to multiply standard deviation for inclusion
        iterations : int, optional
            Number of iterations for the algorithm
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        try:
            # Convert to SimpleITK image
            sitk_image = sitk.GetImageFromArray(image_data)
            
            # Create confidence connected filter
            cc_filter = sitk.ConfidenceConnectedImageFilter()
            cc_filter.SetNumberOfIterations(iterations)
            cc_filter.SetMultiplier(multiplier)
            cc_filter.SetInitialNeighborhoodRadius(1)
            
            # Convert seed point to SimpleITK format (x, y, z)
            sitk_seed = (seed_point[2], seed_point[1], seed_point[0])
            cc_filter.AddSeed(sitk_seed)
            
            # Execute the filter
            sitk_result = cc_filter.Execute(sitk_image)
            
            # Convert back to numpy
            result_mask = sitk.GetArrayFromImage(sitk_result)
            
            return result_mask
            
        except Exception as e:
            logger.error(f"Error in confidence connected segmentation: {str(e)}")
            return np.zeros_like(image_data, dtype=np.uint8)


class WatershedSegmenter:
    """
    Class for watershed-based segmentation.
    
    This class implements watershed algorithms for
    semi-automatic image segmentation.
    """
    
    def __init__(self):
        """Initialize watershed segmenter."""
        pass
    
    def gradient_watershed(self, image_data: np.ndarray, 
                         markers: Optional[np.ndarray] = None,
                         sigma: float = 1.0) -> np.ndarray:
        """
        Apply watershed segmentation based on gradient.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        markers : np.ndarray, optional
            Initial markers for watershed. If None, automatic markers are generated.
        sigma : float, optional
            Sigma for the Gaussian filter used to compute gradient
            
        Returns
        -------
        np.ndarray
            Labeled image with segmentation results
        """
        # Initialize result array
        result = np.zeros_like(image_data, dtype=np.int32)
        
        # Process each slice
        for z in range(image_data.shape[0]):
            slice_data = image_data[z]
            
            # Skip empty slices
            if np.max(slice_data) == np.min(slice_data):
                continue
            
            # Compute gradient magnitude
            gradient = filters.sobel(slice_data)
            
            # Create markers if not provided
            if markers is None:
                # Automatic marker generation using local minima
                slice_markers = np.zeros_like(slice_data, dtype=np.int32)
                
                # Identify background
                background = slice_data < filters.threshold_otsu(slice_data)
                
                # Find foreground markers
                distance = ndimage.distance_transform_edt(~background)
                local_max = morphology.local_maxima(distance)
                markers = ndimage.label(local_max)[0]
                
                # Add background marker
                slice_markers = markers + 1
                slice_markers[background] = 1
            else:
                # Use provided markers for this slice
                slice_markers = markers[z]
            
            # Apply watershed
            labels = segmentation.watershed(gradient, markers=slice_markers)
            result[z] = labels
        
        return result
    
    def marker_watershed(self, image_data: np.ndarray, foreground_markers: np.ndarray,
                       background_markers: np.ndarray) -> np.ndarray:
        """
        Apply watershed segmentation with user-defined markers.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        foreground_markers : np.ndarray
            Binary mask of foreground markers
        background_markers : np.ndarray
            Binary mask of background markers
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        # Initialize result array
        result = np.zeros_like(image_data, dtype=np.uint8)
        
        # Process each slice
        for z in range(image_data.shape[0]):
            slice_data = image_data[z]
            fg_markers = foreground_markers[z]
            bg_markers = background_markers[z]
            
            # Skip slices without markers
            if np.sum(fg_markers) == 0 and np.sum(bg_markers) == 0:
                continue
            
            # Compute gradient magnitude
            gradient = filters.sobel(slice_data)
            
            # Create marker image
            markers = np.zeros_like(slice_data, dtype=np.int32)
            markers[fg_markers > 0] = 2  # Foreground
            markers[bg_markers > 0] = 1  # Background
            
            # Apply watershed
            labels = segmentation.watershed(gradient, markers=markers)
            
            # Extract segmented region
            result[z] = (labels == 2).astype(np.uint8)
        
        return result
    
    def interactive_watershed(self, image_data: np.ndarray, seed_points: List[Tuple[int, int, int]],
                            bg_points: List[Tuple[int, int, int]], sigma: float = 1.0) -> np.ndarray:
        """
        Apply interactive watershed segmentation.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        seed_points : List[Tuple[int, int, int]]
            List of foreground seed points (z, y, x)
        bg_points : List[Tuple[int, int, int]]
            List of background seed points (z, y, x)
        sigma : float, optional
            Sigma for the Gaussian filter used to compute gradient
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        # Create marker image
        markers = np.zeros_like(image_data, dtype=np.int32)
        
        # Set foreground seed points
        for z, y, x in seed_points:
            if (0 <= z < image_data.shape[0] and 
                0 <= y < image_data.shape[1] and 
                0 <= x < image_data.shape[2]):
                markers[z, y, x] = 2
        
        # Set background seed points
        for z, y, x in bg_points:
            if (0 <= z < image_data.shape[0] and 
                0 <= y < image_data.shape[1] and 
                0 <= x < image_data.shape[2]):
                markers[z, y, x] = 1
        
        # Dilate markers to create regions
        fg_markers = ndimage.binary_dilation(markers == 2)
        bg_markers = ndimage.binary_dilation(markers == 1)
        
        # Update markers
        markers = np.zeros_like(image_data, dtype=np.int32)
        markers[fg_markers] = 2
        markers[bg_markers] = 1
        
        # Apply watershed to the volume
        # Compute gradient magnitude
        gradient = ndimage.gaussian_gradient_magnitude(image_data, sigma)
        
        # Apply watershed
        result = segmentation.watershed(gradient, markers=markers)
        
        # Extract segmented region
        return (result == 2).astype(np.uint8)


# Create convenience functions to instantiate the segmenters

def create_threshold_segmenter() -> ThresholdSegmenter:
    """
    Create and return a ThresholdSegmenter instance.
    
    Returns
    -------
    ThresholdSegmenter
        Initialized threshold segmenter
    """
    return ThresholdSegmenter()


def create_region_growing_segmenter() -> RegionGrowingSegmenter:
    """
    Create and return a RegionGrowingSegmenter instance.
    
    Returns
    -------
    RegionGrowingSegmenter
        Initialized region growing segmenter
    """
    return RegionGrowingSegmenter()


def create_watershed_segmenter() -> WatershedSegmenter:
    """
    Create and return a WatershedSegmenter instance.
    
    Returns
    -------
    WatershedSegmenter
        Initialized watershed segmenter
    """
    return WatershedSegmenter()
