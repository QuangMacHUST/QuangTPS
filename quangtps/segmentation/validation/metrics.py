#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Segmentation Metrics for QuangTPS.

This module provides various metrics for evaluating segmentation quality,
including volume-based, surface-based, and overlap-based metrics.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import SimpleITK as sitk
from scipy import ndimage
import math
from skimage import measure
from scipy.spatial import distance

from quangtps.core.exceptions import ValidationError
from quangtps.segmentation.structures.structure import Structure
from quangtps.segmentation.structures.structure_set import StructureSet
logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """
    Base class for segmentation metrics calculation.
    
    This class provides common functionality for calculating 
    various segmentation quality metrics.
    """
    
    def __init__(self):
        """Initialize segmentation metrics calculator."""
        pass
    
    def dice_coefficient(self, segmentation: np.ndarray, 
                        reference: np.ndarray) -> float:
        """
        Calculate Dice similarity coefficient.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
            
        Returns
        -------
        float
            Dice coefficient (0-1, higher is better)
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Ensure binary masks
        seg_bin = (segmentation > 0).astype(np.int32)
        ref_bin = (reference > 0).astype(np.int32)
        
        # Calculate intersection and sums
        intersection = np.sum(seg_bin * ref_bin)
        seg_sum = np.sum(seg_bin)
        ref_sum = np.sum(ref_bin)
        
        # Avoid division by zero
        if seg_sum + ref_sum == 0:
            return 1.0  # Both masks are empty, perfect agreement
        
        # Calculate Dice
        dice = 2.0 * intersection / (seg_sum + ref_sum)
        
        return float(dice)
    
    def jaccard_index(self, segmentation: np.ndarray, 
                     reference: np.ndarray) -> float:
        """
        Calculate Jaccard similarity index.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
            
        Returns
        -------
        float
            Jaccard index (0-1, higher is better)
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Ensure binary masks
        seg_bin = (segmentation > 0).astype(np.int32)
        ref_bin = (reference > 0).astype(np.int32)
        
        # Calculate intersection and union
        intersection = np.sum(seg_bin * ref_bin)
        union = np.sum(seg_bin) + np.sum(ref_bin) - intersection
        
        # Avoid division by zero
        if union == 0:
            return 1.0  # Both masks are empty, perfect agreement
        
        # Calculate Jaccard
        jaccard = intersection / union
        
        return float(jaccard)
    
    def sensitivity(self, segmentation: np.ndarray, 
                  reference: np.ndarray) -> float:
        """
        Calculate sensitivity (true positive rate).
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
            
        Returns
        -------
        float
            Sensitivity (0-1, higher is better)
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Ensure binary masks
        seg_bin = (segmentation > 0).astype(np.int32)
        ref_bin = (reference > 0).astype(np.int32)
        
        # Calculate true positives and false negatives
        true_positives = np.sum(seg_bin * ref_bin)
        reference_sum = np.sum(ref_bin)
        
        # Avoid division by zero
        if reference_sum == 0:
            return 1.0  # Reference mask is empty, no false negatives
        
        # Calculate sensitivity
        sensitivity = true_positives / reference_sum
        
        return float(sensitivity)
    
    def specificity(self, segmentation: np.ndarray, 
                  reference: np.ndarray) -> float:
        """
        Calculate specificity (true negative rate).
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
            
        Returns
        -------
        float
            Specificity (0-1, higher is better)
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Ensure binary masks
        seg_bin = (segmentation > 0).astype(np.int32)
        ref_bin = (reference > 0).astype(np.int32)
        
        # Calculate true negatives and false positives
        true_negatives = np.sum((1 - seg_bin) * (1 - ref_bin))
        total_negatives = np.sum(1 - ref_bin)
        
        # Avoid division by zero
        if total_negatives == 0:
            return 1.0  # Reference mask is full, no true negatives
        
        # Calculate specificity
        specificity = true_negatives / total_negatives
        
        return float(specificity)
    
    def precision(self, segmentation: np.ndarray, 
                reference: np.ndarray) -> float:
        """
        Calculate precision (positive predictive value).
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
            
        Returns
        -------
        float
            Precision (0-1, higher is better)
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Ensure binary masks
        seg_bin = (segmentation > 0).astype(np.int32)
        ref_bin = (reference > 0).astype(np.int32)
        
        # Calculate true positives and false positives
        true_positives = np.sum(seg_bin * ref_bin)
        segmentation_sum = np.sum(seg_bin)
        
        # Avoid division by zero
        if segmentation_sum == 0:
            return 1.0  # Segmentation mask is empty, no false positives
        
        # Calculate precision
        precision = true_positives / segmentation_sum
        
        return float(precision)
    
    def calculate_all_overlap_metrics(self, segmentation: np.ndarray, 
                                    reference: np.ndarray) -> Dict[str, float]:
        """
        Calculate all overlap-based metrics.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing all metrics
        """
        return {
            'dice': self.dice_coefficient(segmentation, reference),
            'jaccard': self.jaccard_index(segmentation, reference),
            'sensitivity': self.sensitivity(segmentation, reference),
            'specificity': self.specificity(segmentation, reference),
            'precision': self.precision(segmentation, reference)
        }
    
    @staticmethod
    def calculate_confusion_matrix(segmentation: np.ndarray, 
                                reference: np.ndarray) -> Dict[str, int]:
        """
        Calculate confusion matrix elements.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
            
        Returns
        -------
        Dict[str, int]
            Dictionary containing TP, TN, FP, FN counts
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Ensure binary masks
        seg_bin = (segmentation > 0).astype(np.int32)
        ref_bin = (reference > 0).astype(np.int32)
        
        # Calculate confusion matrix elements
        true_positives = np.sum(seg_bin * ref_bin)
        true_negatives = np.sum((1 - seg_bin) * (1 - ref_bin))
        false_positives = np.sum(seg_bin * (1 - ref_bin))
        false_negatives = np.sum((1 - seg_bin) * ref_bin)
        
        return {
            'TP': int(true_positives),
            'TN': int(true_negatives),
            'FP': int(false_positives),
            'FN': int(false_negatives)
        }


class VolumeMetrics(SegmentationMetrics):
    """
    Class for calculating volume-based segmentation metrics.
    
    This class extends SegmentationMetrics with volume-specific
    evaluation metrics.
    """
    
    def __init__(self):
        """Initialize volume metrics calculator."""
        super().__init__()
    
    def volume_difference(self, segmentation: np.ndarray, reference: np.ndarray,
                        spacing: Tuple[float, float, float] = None) -> float:
        """
        Calculate absolute volume difference.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm
            
        Returns
        -------
        float
            Absolute volume difference in ml (0 is perfect)
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Set default spacing if not provided
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        
        # Calculate voxel volume in ml (assuming spacing in mm)
        voxel_volume = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        
        # Count voxels in each mask
        seg_volume = np.sum(segmentation > 0) * voxel_volume
        ref_volume = np.sum(reference > 0) * voxel_volume
        
        # Calculate volume difference
        volume_diff = abs(seg_volume - ref_volume)
        
        return float(volume_diff)
    
    def relative_volume_difference(self, segmentation: np.ndarray, 
                                 reference: np.ndarray) -> float:
        """
        Calculate relative volume difference.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
            
        Returns
        -------
        float
            Relative volume difference (-1 to inf, 0 is perfect)
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Count voxels in each mask
        seg_volume = np.sum(segmentation > 0)
        ref_volume = np.sum(reference > 0)
        
        # Avoid division by zero
        if ref_volume == 0:
            if seg_volume == 0:
                return 0.0  # Both volumes are zero, perfect match
            else:
                return float('inf')  # Reference is empty but segmentation is not
        
        # Calculate relative difference
        rel_diff = (seg_volume - ref_volume) / ref_volume
        
        return float(rel_diff)
    
    def volume_overlap_error(self, segmentation: np.ndarray, 
                           reference: np.ndarray) -> float:
        """
        Calculate volume overlap error.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
            
        Returns
        -------
        float
            Volume overlap error (0-1, lower is better)
        """
        # Volume overlap error is 1 - Dice coefficient
        dice = self.dice_coefficient(segmentation, reference)
        return 1.0 - dice
    
    def calculate_all_volume_metrics(self, segmentation: np.ndarray, 
                                   reference: np.ndarray,
                                   spacing: Tuple[float, float, float] = None) -> Dict[str, float]:
        """
        Calculate all volume-based metrics.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing all metrics
        """
        return {
            'volume_difference': self.volume_difference(segmentation, reference, spacing),
            'relative_volume_difference': self.relative_volume_difference(segmentation, reference),
            'volume_overlap_error': self.volume_overlap_error(segmentation, reference)
        }


class SurfaceMetrics(SegmentationMetrics):
    """
    Class for calculating surface-based segmentation metrics.
    
    This class extends SegmentationMetrics with surface-specific
    evaluation metrics.
    """
    
    def __init__(self):
        """Initialize surface metrics calculator."""
        super().__init__()
    
    def _extract_surface(self, binary_segmentation: np.ndarray) -> np.ndarray:
        """
        Extract surface voxels from a binary segmentation.
        
        Parameters
        ----------
        binary_segmentation : np.ndarray
            Binary segmentation mask
            
        Returns
        -------
        np.ndarray
            Binary mask of surface voxels
        """
        # Apply binary erosion
        eroded = ndimage.binary_erosion(binary_segmentation)
        
        # Surface is the difference between original and eroded mask
        surface = np.logical_xor(binary_segmentation, eroded)
        
        return surface
    
    def _compute_surface_distances(self, surface_1: np.ndarray, 
                                 surface_2: np.ndarray,
                                 spacing: Tuple[float, float, float] = None) -> np.ndarray:
        """
        Compute directed surface distances.
        
        Parameters
        ----------
        surface_1 : np.ndarray
            First surface
        surface_2 : np.ndarray
            Second surface
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm
            
        Returns
        -------
        np.ndarray
            Array of directed distances from surface_1 to surface_2
        """
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        
        # Get indices of surface voxels
        surface_1_indices = np.where(surface_1)
        surface_2_indices = np.where(surface_2)
        
        # Convert to point coordinates
        surface_1_points = np.array([
            surface_1_indices[0] * spacing[0],
            surface_1_indices[1] * spacing[1],
            surface_1_indices[2] * spacing[2]
        ]).T
        
        surface_2_points = np.array([
            surface_2_indices[0] * spacing[0],
            surface_2_indices[1] * spacing[1],
            surface_2_indices[2] * spacing[2]
        ]).T
        
        # If either surface is empty, return appropriate distances
        if len(surface_1_points) == 0:
            return np.array([])
        
        if len(surface_2_points) == 0:
            return np.array([np.inf] * len(surface_1_points))
        
        # Use scipy's cdist for fast computation of distances
        distances = distance.cdist(surface_1_points, surface_2_points, 'euclidean')
        
        # Get minimum distance for each point on surface_1
        min_distances = np.min(distances, axis=1)
        
        return min_distances
    
    def hausdorff_distance(self, segmentation: np.ndarray, 
                          reference: np.ndarray,
                          spacing: Tuple[float, float, float] = None) -> float:
        """
        Calculate Hausdorff distance.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm
            
        Returns
        -------
        float
            Hausdorff distance in mm
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Set default spacing if not provided
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        
        # Ensure binary masks
        seg_bin = (segmentation > 0)
        ref_bin = (reference > 0)
        
        # Extract surfaces
        seg_surface = self._extract_surface(seg_bin)
        ref_surface = self._extract_surface(ref_bin)
        
        # Compute directed surface distances
        seg_to_ref_distances = self._compute_surface_distances(seg_surface, ref_surface, spacing)
        ref_to_seg_distances = self._compute_surface_distances(ref_surface, seg_surface, spacing)
        
        # Handle empty surfaces
        if len(seg_to_ref_distances) == 0 and len(ref_to_seg_distances) == 0:
            return 0.0  # Both surfaces are empty
        
        if len(seg_to_ref_distances) == 0:
            return np.max(ref_to_seg_distances)
        
        if len(ref_to_seg_distances) == 0:
            return np.max(seg_to_ref_distances)
        
        # Compute Hausdorff distance
        hausdorff = max(np.max(seg_to_ref_distances), np.max(ref_to_seg_distances))
        
        return float(hausdorff)
    
    def average_surface_distance(self, segmentation: np.ndarray, 
                               reference: np.ndarray,
                               spacing: Tuple[float, float, float] = None) -> float:
        """
        Calculate average surface distance.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm
            
        Returns
        -------
        float
            Average surface distance in mm
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Set default spacing if not provided
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        
        # Ensure binary masks
        seg_bin = (segmentation > 0)
        ref_bin = (reference > 0)
        
        # Extract surfaces
        seg_surface = self._extract_surface(seg_bin)
        ref_surface = self._extract_surface(ref_bin)
        
        # Compute directed surface distances
        seg_to_ref_distances = self._compute_surface_distances(seg_surface, ref_surface, spacing)
        ref_to_seg_distances = self._compute_surface_distances(ref_surface, seg_surface, spacing)
        
        # Handle empty surfaces
        if len(seg_to_ref_distances) == 0 and len(ref_to_seg_distances) == 0:
            return 0.0  # Both surfaces are empty
        
        if len(seg_to_ref_distances) == 0:
            return np.mean(ref_to_seg_distances)
        
        if len(ref_to_seg_distances) == 0:
            return np.mean(seg_to_ref_distances)
        
        # Compute average surface distance
        avg_distance = (np.sum(seg_to_ref_distances) + np.sum(ref_to_seg_distances)) / (
            len(seg_to_ref_distances) + len(ref_to_seg_distances)
        )
        
        return float(avg_distance)
    
    def symmetric_surface_distance(self, segmentation: np.ndarray, 
                                reference: np.ndarray,
                                spacing: Tuple[float, float, float] = None) -> float:
        """
        Calculate symmetric average surface distance.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm
            
        Returns
        -------
        float
            Symmetric average surface distance in mm
        """
        # This is the same as average surface distance in our implementation
        return self.average_surface_distance(segmentation, reference, spacing)
    
    def percentile_hausdorff_distance(self, segmentation: np.ndarray, 
                                   reference: np.ndarray,
                                   spacing: Tuple[float, float, float] = None,
                                   percentile: float = 95.0) -> float:
        """
        Calculate percentile Hausdorff distance.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm
        percentile : float, optional
            Percentile to use (default: 95.0)
            
        Returns
        -------
        float
            Percentile Hausdorff distance in mm
        """
        if segmentation.shape != reference.shape:
            raise ValidationError("Segmentation and reference must have the same shape")
        
        # Set default spacing if not provided
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        
        # Ensure binary masks
        seg_bin = (segmentation > 0)
        ref_bin = (reference > 0)
        
        # Extract surfaces
        seg_surface = self._extract_surface(seg_bin)
        ref_surface = self._extract_surface(ref_bin)
        
        # Compute directed surface distances
        seg_to_ref_distances = self._compute_surface_distances(seg_surface, ref_surface, spacing)
        ref_to_seg_distances = self._compute_surface_distances(ref_surface, seg_surface, spacing)
        
        # Handle empty surfaces
        if len(seg_to_ref_distances) == 0 and len(ref_to_seg_distances) == 0:
            return 0.0  # Both surfaces are empty
        
        if len(seg_to_ref_distances) == 0:
            return np.percentile(ref_to_seg_distances, percentile)
        
        if len(ref_to_seg_distances) == 0:
            return np.percentile(seg_to_ref_distances, percentile)
        
        # Combine distances
        all_distances = np.concatenate([seg_to_ref_distances, ref_to_seg_distances])
        
        # Compute percentile Hausdorff distance
        percentile_hausdorff = np.percentile(all_distances, percentile)
        
        return float(percentile_hausdorff)
    
    def calculate_all_surface_metrics(self, segmentation: np.ndarray, 
                                    reference: np.ndarray,
                                    spacing: Tuple[float, float, float] = None) -> Dict[str, float]:
        """
        Calculate all surface-based metrics.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing all metrics
        """
        return {
            'hausdorff_distance': self.hausdorff_distance(segmentation, reference, spacing),
            'average_surface_distance': self.average_surface_distance(segmentation, reference, spacing),
            'symmetric_surface_distance': self.symmetric_surface_distance(segmentation, reference, spacing),
            'percentile_hausdorff_distance_95': self.percentile_hausdorff_distance(
                segmentation, reference, spacing, 95.0
            )
        }


def calculate_comprehensive_metrics(segmentation: np.ndarray, reference: np.ndarray,
                               spacing: Tuple[float, float, float] = None) -> Dict[str, float]:
    """
    Calculate a comprehensive set of segmentation evaluation metrics.
    
    Parameters
    ----------
    segmentation : np.ndarray
        Segmentation mask to evaluate
    reference : np.ndarray
        Reference (ground truth) mask
    spacing : Tuple[float, float, float], optional
        Voxel spacing in mm
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing all metrics
    """
    # Initialize metric calculators
    seg_metrics = SegmentationMetrics()
    vol_metrics = VolumeMetrics()
    surf_metrics = SurfaceMetrics()
    
    # Get all metrics
    overlap_metrics = seg_metrics.calculate_all_overlap_metrics(segmentation, reference)
    volume_metrics = vol_metrics.calculate_all_volume_metrics(segmentation, reference, spacing)
    surface_metrics = surf_metrics.calculate_all_surface_metrics(segmentation, reference, spacing)
    
    # Combine all metrics
    all_metrics = {}
    all_metrics.update(overlap_metrics)
    all_metrics.update(volume_metrics)
    all_metrics.update(surface_metrics)
    
    return all_metrics
