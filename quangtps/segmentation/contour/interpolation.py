#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for contour interpolation.

This module provides functionality for interpolating contours between slices,
which is essential for creating consistent 3D structures from manually drawn
contours on a subset of slices.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
import SimpleITK as sitk
from skimage import measure, morphology, draw, segmentation, filters
from scipy import ndimage, interpolate

logger = logging.getLogger(__name__)


class InterpolationMethod(str, Enum):
    """Enum for different interpolation methods."""
    LINEAR = "LINEAR"  # Linear interpolation between contours
    SHAPE_BASED = "SHAPE_BASED"  # Shape-based interpolation (distance map)
    MORPHOLOGICAL = "MORPHOLOGICAL"  # Morphological interpolation
    ELASTIC = "ELASTIC"  # Elastic interpolation


class ContourInterpolator:
    """
    Class for interpolating contours between slices.
    
    This class provides methods to interpolate contours between slices,
    which is useful when contours are drawn only on a subset of slices.
    """
    
    def __init__(self, method: InterpolationMethod = InterpolationMethod.SHAPE_BASED,
                 smoothing: float = 0.5):
        """
        Initialize contour interpolator.
        
        Parameters
        ----------
        method : InterpolationMethod, optional
            Interpolation method to use
        smoothing : float, optional
            Smoothing factor for the interpolation (0-1)
        """
        self.method = method
        self.smoothing = smoothing
    
    def interpolate_contours(self, 
                          slice_contours: Dict[int, np.ndarray],
                          slice_spacing: float,
                          total_slices: int) -> Dict[int, np.ndarray]:
        """
        Interpolate contours between slices.
        
        Parameters
        ----------
        slice_contours : Dict[int, np.ndarray]
            Dictionary mapping slice indices to contour points
        slice_spacing : float
            Spacing between slices in mm
        total_slices : int
            Total number of slices
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary with interpolated contours for all slices
        """
        # Validate input
        if not slice_contours:
            logger.warning("No contours provided for interpolation")
            return {}
        
        # Extract slice indices with contours
        slices_with_contours = sorted(slice_contours.keys())
        
        if len(slices_with_contours) <= 1:
            logger.warning("At least two contoured slices are required for interpolation")
            return slice_contours
        
        # Determine image bounds for all contours
        all_contour_points = np.vstack([slice_contours[s] for s in slices_with_contours])
        min_coords = np.min(all_contour_points, axis=0) - 10  # Add margin
        max_coords = np.max(all_contour_points, axis=0) + 10  # Add margin
        
        # Create a mask volume with existing contours
        volume_shape = (total_slices, 
                       int(max_coords[1] - min_coords[1]), 
                       int(max_coords[0] - min_coords[0]))
        
        volume = np.zeros(volume_shape, dtype=np.uint8)
        
        # Fill in the known contours
        for slice_idx, contour in slice_contours.items():
            if 0 <= slice_idx < total_slices:
                normalized_contour = contour.copy()
                normalized_contour[:, 0] -= min_coords[0]
                normalized_contour[:, 1] -= min_coords[1]
                
                slice_mask = np.zeros((volume_shape[1], volume_shape[2]), dtype=np.uint8)
                rr, cc = draw.polygon(normalized_contour[:, 1], normalized_contour[:, 0])
                
                # Filter out points outside the image
                valid_indices = (rr >= 0) & (rr < volume_shape[1]) & (cc >= 0) & (cc < volume_shape[2])
                if np.any(valid_indices):
                    slice_mask[rr[valid_indices], cc[valid_indices]] = 1
                    volume[slice_idx] = slice_mask
        
        # Perform interpolation based on selected method
        interpolated_volume = self._interpolate_volume(volume, slices_with_contours, self.method)
        
        # Extract contours from the interpolated volume
        result_contours = {}
        
        for slice_idx in range(total_slices):
            if slice_idx in slice_contours:
                # Use original contour
                result_contours[slice_idx] = slice_contours[slice_idx]
            else:
                # Extract contour from interpolated volume
                if np.any(interpolated_volume[slice_idx]):
                    contours = measure.find_contours(interpolated_volume[slice_idx], 0.5)
                    if contours:
                        # Use the largest contour
                        largest_contour = max(contours, key=len)
                        
                        # Convert back to original coordinate system
                        extracted_contour = np.fliplr(largest_contour)
                        extracted_contour[:, 0] += min_coords[0]
                        extracted_contour[:, 1] += min_coords[1]
                        
                        result_contours[slice_idx] = extracted_contour
        
        return result_contours
    
    def _interpolate_volume(self, 
                          volume: np.ndarray,
                          slices_with_contours: List[int],
                          method: InterpolationMethod) -> np.ndarray:
        """
        Interpolate a binary volume.
        
        Parameters
        ----------
        volume : np.ndarray
            Binary volume with known contours
        slices_with_contours : List[int]
            List of slice indices with known contours
        method : InterpolationMethod
            Interpolation method to use
            
        Returns
        -------
        np.ndarray
            Interpolated binary volume
        """
        result_volume = volume.copy()
        
        if method == InterpolationMethod.LINEAR:
            return self._linear_interpolation(volume, slices_with_contours)
        elif method == InterpolationMethod.SHAPE_BASED:
            return self._shape_based_interpolation(volume, slices_with_contours)
        elif method == InterpolationMethod.MORPHOLOGICAL:
            return self._morphological_interpolation(volume, slices_with_contours)
        elif method == InterpolationMethod.ELASTIC:
            return self._elastic_interpolation(volume, slices_with_contours)
        else:
            logger.warning(f"Unknown interpolation method: {method}, using shape-based")
            return self._shape_based_interpolation(volume, slices_with_contours)
    
    def _linear_interpolation(self, 
                            volume: np.ndarray,
                            slices_with_contours: List[int]) -> np.ndarray:
        """
        Perform linear interpolation between slices.
        
        Parameters
        ----------
        volume : np.ndarray
            Binary volume with known contours
        slices_with_contours : List[int]
            List of slice indices with known contours
            
        Returns
        -------
        np.ndarray
            Interpolated binary volume
        """
        result_volume = volume.copy()
        total_slices = volume.shape[0]
        
        # Interpolate between each pair of known slices
        for i in range(len(slices_with_contours) - 1):
            start_idx = slices_with_contours[i]
            end_idx = slices_with_contours[i + 1]
            
            if end_idx - start_idx <= 1:
                continue  # No slices to interpolate
            
            start_mask = volume[start_idx].astype(float)
            end_mask = volume[end_idx].astype(float)
            
            # Smooth masks if requested
            if self.smoothing > 0:
                sigma = self.smoothing * 2
                start_mask = filters.gaussian(start_mask, sigma=sigma)
                end_mask = filters.gaussian(end_mask, sigma=sigma)
                start_mask = (start_mask > 0.5).astype(float)
                end_mask = (end_mask > 0.5).astype(float)
            
            # Linear interpolation between slices
            for slice_idx in range(start_idx + 1, end_idx):
                weight = (slice_idx - start_idx) / (end_idx - start_idx)
                interpolated_mask = (1 - weight) * start_mask + weight * end_mask
                result_volume[slice_idx] = (interpolated_mask > 0.5).astype(np.uint8)
        
        return result_volume
    
    def _shape_based_interpolation(self, 
                                 volume: np.ndarray,
                                 slices_with_contours: List[int]) -> np.ndarray:
        """
        Perform shape-based interpolation between slices.
        
        Parameters
        ----------
        volume : np.ndarray
            Binary volume with known contours
        slices_with_contours : List[int]
            List of slice indices with known contours
            
        Returns
        -------
        np.ndarray
            Interpolated binary volume
        """
        result_volume = volume.copy()
        total_slices = volume.shape[0]
        
        # Convert to SimpleITK image for shape-based interpolation
        sitk_volume = sitk.GetImageFromArray(volume.astype(np.uint8))
        
        # Calculate signed distance maps for slices with contours
        distance_maps = {}
        for slice_idx in slices_with_contours:
            mask = volume[slice_idx]
            
            # Compute signed distance transform
            dist_in = ndimage.distance_transform_edt(mask)
            dist_out = ndimage.distance_transform_edt(1 - mask)
            signed_dist = dist_in - dist_out
            
            distance_maps[slice_idx] = signed_dist
        
        # Interpolate distance maps between known slices
        all_signed_dists = np.zeros_like(volume, dtype=float)
        
        for slice_idx in slices_with_contours:
            all_signed_dists[slice_idx] = distance_maps[slice_idx]
        
        # Interpolate between each pair of known slices
        for i in range(len(slices_with_contours) - 1):
            start_idx = slices_with_contours[i]
            end_idx = slices_with_contours[i + 1]
            
            if end_idx - start_idx <= 1:
                continue  # No slices to interpolate
            
            start_dist = distance_maps[start_idx]
            end_dist = distance_maps[end_idx]
            
            # Apply smoothing if requested
            if self.smoothing > 0:
                sigma = self.smoothing * 2
                start_dist = filters.gaussian(start_dist, sigma=sigma)
                end_dist = filters.gaussian(end_dist, sigma=sigma)
            
            # Linear interpolation of distance maps
            for slice_idx in range(start_idx + 1, end_idx):
                weight = (slice_idx - start_idx) / (end_idx - start_idx)
                interpolated_dist = (1 - weight) * start_dist + weight * end_dist
                all_signed_dists[slice_idx] = interpolated_dist
        
        # Convert interpolated distance maps back to binary masks
        for slice_idx in range(total_slices):
            if slice_idx not in slices_with_contours:
                result_volume[slice_idx] = (all_signed_dists[slice_idx] > 0).astype(np.uint8)
        
        return result_volume
    
    def _morphological_interpolation(self, 
                                   volume: np.ndarray,
                                   slices_with_contours: List[int]) -> np.ndarray:
        """
        Perform morphological interpolation between slices.
        
        Parameters
        ----------
        volume : np.ndarray
            Binary volume with known contours
        slices_with_contours : List[int]
            List of slice indices with known contours
            
        Returns
        -------
        np.ndarray
            Interpolated binary volume
        """
        result_volume = volume.copy()
        total_slices = volume.shape[0]
        
        # Interpolate between each pair of known slices
        for i in range(len(slices_with_contours) - 1):
            start_idx = slices_with_contours[i]
            end_idx = slices_with_contours[i + 1]
            
            if end_idx - start_idx <= 1:
                continue  # No slices to interpolate
            
            # Get binary masks for start and end slices
            start_mask = volume[start_idx]
            end_mask = volume[end_idx]
            
            # Compute the number of steps needed
            steps = end_idx - start_idx - 1
            
            # For morphological interpolation, we do successive dilations and erosions
            # First dilate the start mask enough to overlap with end mask
            dilated_start = start_mask.copy()
            dilated_end = end_mask.copy()
            
            # Determine the dilation size based on the number of steps
            dilation_size = max(1, steps // 2)
            
            # Dilate both masks to create overlap
            for _ in range(dilation_size):
                dilated_start = morphology.binary_dilation(dilated_start, morphology.disk(1))
                dilated_end = morphology.binary_dilation(dilated_end, morphology.disk(1))
            
            # Create intermediate masks by alpha blending
            for slice_idx in range(start_idx + 1, end_idx):
                alpha = (slice_idx - start_idx) / (end_idx - start_idx)
                
                # Blend masks with weights
                intermediate = np.zeros_like(start_mask)
                intermediate[dilated_start > 0] += (1 - alpha)
                intermediate[dilated_end > 0] += alpha
                
                # Threshold the result
                result_volume[slice_idx] = (intermediate > 0.5).astype(np.uint8)
                
                # Apply smoothing if requested
                if self.smoothing > 0:
                    sigma = self.smoothing * 2
                    result_volume[slice_idx] = filters.gaussian(result_volume[slice_idx], sigma=sigma) > 0.5
        
        return result_volume.astype(np.uint8)
    
    def _elastic_interpolation(self, 
                             volume: np.ndarray,
                             slices_with_contours: List[int]) -> np.ndarray:
        """
        Perform elastic interpolation between slices.
        
        Parameters
        ----------
        volume : np.ndarray
            Binary volume with known contours
        slices_with_contours : List[int]
            List of slice indices with known contours
            
        Returns
        -------
        np.ndarray
            Interpolated binary volume
        """
        result_volume = volume.copy()
        total_slices = volume.shape[0]
        
        try:
            # Convert to SimpleITK image
            sitk_volume = sitk.GetImageFromArray(volume.astype(np.uint8))
            
            # Create a label map indicating which slices have contours
            label_map = np.zeros((total_slices,), dtype=np.uint8)
            for idx in slices_with_contours:
                label_map[idx] = 1
            
            # For elastic interpolation, we use SimpleITK's BSpline interpolation
            for i in range(len(slices_with_contours) - 1):
                start_idx = slices_with_contours[i]
                end_idx = slices_with_contours[i + 1]
                
                if end_idx - start_idx <= 1:
                    continue  # No slices to interpolate
                
                # Extract contours from start and end slices
                start_contours = measure.find_contours(volume[start_idx], 0.5)
                end_contours = measure.find_contours(volume[end_idx], 0.5)
                
                if not start_contours or not end_contours:
                    continue
                
                # Use the largest contours
                start_contour = max(start_contours, key=len)
                end_contour = max(end_contours, key=len)
                
                # Resample contours to the same number of points
                n_points = min(len(start_contour), len(end_contour))
                n_points = max(n_points, 50)  # Ensure at least 50 points
                
                start_contour_resampled = self._resample_contour(start_contour, n_points)
                end_contour_resampled = self._resample_contour(end_contour, n_points)
                
                # Interpolate between contour points
                for slice_idx in range(start_idx + 1, end_idx):
                    weight = (slice_idx - start_idx) / (end_idx - start_idx)
                    
                    # Linear interpolation between corresponding points
                    interpolated_contour = (1 - weight) * start_contour_resampled + weight * end_contour_resampled
                    
                    # Create mask from interpolated contour
                    mask = np.zeros_like(volume[slice_idx])
                    rr, cc = draw.polygon(interpolated_contour[:, 0], interpolated_contour[:, 1])
                    
                    # Filter out points outside the image
                    valid_indices = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
                    if np.any(valid_indices):
                        mask[rr[valid_indices], cc[valid_indices]] = 1
                        
                        # Apply smoothing if requested
                        if self.smoothing > 0:
                            sigma = self.smoothing * 2
                            mask = filters.gaussian(mask, sigma=sigma) > 0.5
                        
                        result_volume[slice_idx] = mask.astype(np.uint8)
        
        except Exception as e:
            logger.error(f"Error in elastic interpolation: {str(e)}")
            # Fall back to shape-based interpolation
            return self._shape_based_interpolation(volume, slices_with_contours)
        
        return result_volume
    
    def _resample_contour(self, contour: np.ndarray, n_points: int) -> np.ndarray:
        """
        Resample a contour to have a specific number of points.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points
        n_points : int
            Desired number of points
            
        Returns
        -------
        np.ndarray
            Resampled contour
        """
        # Calculate cumulative distance along the contour
        distances = np.zeros(len(contour))
        for i in range(1, len(contour)):
            distances[i] = distances[i-1] + np.linalg.norm(contour[i] - contour[i-1])
        
        # Create new distance samples
        new_distances = np.linspace(0, distances[-1], n_points)
        
        # Interpolate x and y coordinates
        x_interp = np.interp(new_distances, distances, contour[:, 0])
        y_interp = np.interp(new_distances, distances, contour[:, 1])
        
        return np.column_stack((x_interp, y_interp))
    
    @staticmethod
    def interpolate_mask_volume(mask_volume: np.ndarray,
                              slices_with_masks: List[int],
                              method: InterpolationMethod = InterpolationMethod.SHAPE_BASED,
                              smoothing: float = 0.5) -> np.ndarray:
        """
        Interpolate a binary mask volume with gaps.
        
        Parameters
        ----------
        mask_volume : np.ndarray
            3D binary mask volume with known masks on certain slices
        slices_with_masks : List[int]
            List of slice indices with known masks
        method : InterpolationMethod, optional
            Interpolation method to use
        smoothing : float, optional
            Smoothing factor (0-1)
            
        Returns
        -------
        np.ndarray
            Interpolated 3D mask volume
        """
        interpolator = ContourInterpolator(method=method, smoothing=smoothing)
        
        # Create a copy of the input volume
        result_volume = mask_volume.copy()
        
        # Perform interpolation
        interpolated_volume = interpolator._interpolate_volume(
            mask_volume, slices_with_masks, method
        )
        
        # Copy interpolated slices to result volume
        for z in range(mask_volume.shape[0]):
            if z not in slices_with_masks:
                result_volume[z] = interpolated_volume[z]
        
        return result_volume
