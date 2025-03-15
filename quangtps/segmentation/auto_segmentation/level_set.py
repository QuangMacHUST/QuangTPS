#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Level Set Segmentation Module for QuangTPS.

This module provides functionality for level set based segmentation,
which represents contours implicitly as the zero level set of
a higher dimensional function.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import ndimage
from skimage import filters, segmentation, morphology, measure
import SimpleITK as sitk

from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class LevelSetSegmenter:
    """
    Base class for level set segmentation.
    
    This class implements various level set algorithms for
    semi-automatic image segmentation.
    """
    
    def __init__(self, smoothing: float = 1.0, 
                max_iterations: int = 100,
                convergence_threshold: float = 0.001):
        """
        Initialize level set segmenter.
        
        Parameters
        ----------
        smoothing : float, optional
            Smoothing parameter for the level set evolution
        max_iterations : int, optional
            Maximum number of iterations
        convergence_threshold : float, optional
            Threshold for convergence check
        """
        self.smoothing = smoothing
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def segment_slice(self, slice_data: np.ndarray, 
                     initial_mask: np.ndarray) -> np.ndarray:
        """
        Apply level set segmentation to a 2D slice.
        
        Parameters
        ----------
        slice_data : np.ndarray
            2D image slice
        initial_mask : np.ndarray
            Initial binary mask
            
        Returns
        -------
        np.ndarray
            Segmented binary mask
        """
        # This is a base method that will be implemented in subclasses
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def segment_volume(self, volume_data: np.ndarray, 
                      initial_masks: Union[np.ndarray, List[np.ndarray]],
                      slices_with_masks: Optional[List[int]] = None) -> np.ndarray:
        """
        Apply level set segmentation to a 3D volume.
        
        Parameters
        ----------
        volume_data : np.ndarray
            3D image volume
        initial_masks : Union[np.ndarray, List[np.ndarray]]
            Either a 3D binary mask volume or a list of 2D masks for specific slices
        slices_with_masks : Optional[List[int]], optional
            List of slice indices corresponding to initial_masks (if it's a list)
            
        Returns
        -------
        np.ndarray
            Segmented 3D binary mask
        """
        # Check input types
        if isinstance(initial_masks, list):
            if slices_with_masks is None or len(initial_masks) != len(slices_with_masks):
                raise ValidationError("If initial_masks is a list, slices_with_masks must be provided with matching length")
            
            # Create a 3D mask from the 2D slices
            mask_3d = np.zeros_like(volume_data, dtype=np.uint8)
            for mask, idx in zip(initial_masks, slices_with_masks):
                if 0 <= idx < volume_data.shape[0]:
                    mask_3d[idx] = mask
            
            initial_masks = mask_3d
        
        # Check dimensions
        if initial_masks.shape != volume_data.shape:
            raise ValidationError(f"Shape mismatch: volume_data {volume_data.shape}, initial_masks {initial_masks.shape}")
        
        # Process each slice with non-empty initial mask
        result_3d = np.zeros_like(volume_data, dtype=np.uint8)
        
        # First segment slices with initial masks
        for z in range(volume_data.shape[0]):
            if np.any(initial_masks[z]):
                result_3d[z] = self.segment_slice(volume_data[z], initial_masks[z])
        
        # Perform 3D smoothing if needed
        if np.any(result_3d):
            result_3d = morphology.binary_closing(result_3d, footprint=np.ones((3, 3, 3)))
            result_3d = morphology.remove_small_holes(result_3d, area_threshold=64)
        
        return result_3d.astype(np.uint8)
    
    def _init_level_set_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Initialize level set function from a binary mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask
            
        Returns
        -------
        np.ndarray
            Initial level set function (signed distance transform)
        """
        # Convert mask to signed distance function
        distance = ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask) + 0.5
        return distance


class MorphologicalLevelSet(LevelSetSegmenter):
    """
    Class for morphological level set segmentation.
    
    This class implements a fast level set algorithm based on
    morphological operations rather than differential equations.
    """
    
    def __init__(self, smoothing: float = 1.0, 
                max_iterations: int = 100,
                convergence_threshold: float = 0.001,
                threshold: float = 0.5,
                balloon: float = 0.0):
        """
        Initialize morphological level set segmenter.
        
        Parameters
        ----------
        smoothing : float, optional
            Smoothing parameter for the level set evolution
        max_iterations : int, optional
            Maximum number of iterations
        convergence_threshold : float, optional
            Threshold for convergence check
        threshold : float, optional
            Threshold for the speed function
        balloon : float, optional
            Balloon force (positive for expansion, negative for contraction)
        """
        super().__init__(smoothing, max_iterations, convergence_threshold)
        self.threshold = threshold
        self.balloon = balloon
    
    def segment_slice(self, slice_data: np.ndarray, 
                     initial_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological level set segmentation to a 2D slice.
        
        Parameters
        ----------
        slice_data : np.ndarray
            2D image slice
        initial_mask : np.ndarray
            Initial binary mask
            
        Returns
        -------
        np.ndarray
            Segmented binary mask
        """
        # Normalize image data to [0, 1]
        image = slice_data.astype(float)
        if np.max(image) > np.min(image):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Create speed image (edge indicator function)
        edge_image = filters.sobel(filters.gaussian(image, self.smoothing))
        speed_image = 1.0 / (1.0 + edge_image)
        
        # Add balloon force
        if self.balloon != 0:
            speed_image = speed_image + self.balloon
            speed_image = np.clip(speed_image, 0, 1)
        
        # Apply morphological level set
        try:
            ls = segmentation.morphological_chan_vese(
                image,
                iterations=self.max_iterations,
                init_level_set=initial_mask,
                smoothing=int(self.smoothing * 2),
                threshold=self.threshold
            )
            return ls.astype(np.uint8)
        
        except Exception as e:
            logger.error(f"Error in morphological level set segmentation: {str(e)}")
            return initial_mask.astype(np.uint8)


class GeodesicLevelSet(LevelSetSegmenter):
    """
    Class for geodesic level set segmentation.
    
    This class implements geodesic active contours, which evolve
    a contour to minimize a geodesic energy.
    """
    
    def __init__(self, smoothing: float = 1.0, 
                max_iterations: int = 100,
                convergence_threshold: float = 0.001,
                alpha: float = 0.1,
                balloon: float = 0.0):
        """
        Initialize geodesic level set segmenter.
        
        Parameters
        ----------
        smoothing : float, optional
            Smoothing parameter for the level set evolution
        max_iterations : int, optional
            Maximum number of iterations
        convergence_threshold : float, optional
            Threshold for convergence check
        alpha : float, optional
            Weight of the length term
        balloon : float, optional
            Balloon force (positive for expansion, negative for contraction)
        """
        super().__init__(smoothing, max_iterations, convergence_threshold)
        self.alpha = alpha
        self.balloon = balloon
    
    def segment_slice(self, slice_data: np.ndarray, 
                     initial_mask: np.ndarray) -> np.ndarray:
        """
        Apply geodesic level set segmentation to a 2D slice.
        
        Parameters
        ----------
        slice_data : np.ndarray
            2D image slice
        initial_mask : np.ndarray
            Initial binary mask
            
        Returns
        -------
        np.ndarray
            Segmented binary mask
        """
        # Normalize image data to [0, 1]
        image = slice_data.astype(float)
        if np.max(image) > np.min(image):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Create gradient magnitude image
        gimg = filters.gradient_magnitude(filters.gaussian(image, self.smoothing))
        
        # Create inverse edge indicator function
        edge_indicator = 1.0 / (1.0 + gimg)
        
        # Initialize level set
        init_ls = self._init_level_set_from_mask(initial_mask)
        
        # Apply geodesic active contours
        try:
            ls = segmentation.morphological_geodesic_active_contour(
                edge_indicator,
                init_level_set=init_ls,
                iterations=self.max_iterations,
                smoothing=1,
                threshold=self.convergence_threshold,
                balloon=self.balloon
            )
            return ls.astype(np.uint8)
        
        except Exception as e:
            logger.error(f"Error in geodesic level set segmentation: {str(e)}")
            return initial_mask.astype(np.uint8)


class ChanVeseLevelSet(LevelSetSegmenter):
    """
    Class for Chan-Vese level set segmentation.
    
    This class implements the Chan-Vese model, which is a region-based
    active contour model that works well for images without clear edges.
    """
    
    def __init__(self, smoothing: float = 1.0, 
                max_iterations: int = 100,
                convergence_threshold: float = 0.001,
                lambda1: float = 1.0,
                lambda2: float = 1.0,
                mu: float = 0.25):
        """
        Initialize Chan-Vese level set segmenter.
        
        Parameters
        ----------
        smoothing : float, optional
            Smoothing parameter for the level set evolution
        max_iterations : int, optional
            Maximum number of iterations
        convergence_threshold : float, optional
            Threshold for convergence check
        lambda1 : float, optional
            Weight of the inside region term
        lambda2 : float, optional
            Weight of the outside region term
        mu : float, optional
            Weight of the length term
        """
        super().__init__(smoothing, max_iterations, convergence_threshold)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mu = mu
    
    def segment_slice(self, slice_data: np.ndarray, 
                     initial_mask: np.ndarray) -> np.ndarray:
        """
        Apply Chan-Vese level set segmentation to a 2D slice.
        
        Parameters
        ----------
        slice_data : np.ndarray
            2D image slice
        initial_mask : np.ndarray
            Initial binary mask
            
        Returns
        -------
        np.ndarray
            Segmented binary mask
        """
        # Normalize image data to [0, 1]
        image = slice_data.astype(float)
        if np.max(image) > np.min(image):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Apply Chan-Vese level set
        try:
            # Smooth the image
            image = filters.gaussian(image, self.smoothing)
            
            # Initialize level set
            init_ls = self._init_level_set_from_mask(initial_mask)
            
            # Apply Chan-Vese algorithm
            ls = segmentation.chan_vese(
                image,
                mu=self.mu,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
                tol=self.convergence_threshold,
                max_iter=self.max_iterations,
                dt=0.5,
                init_level_set=init_ls
            )
            return ls.astype(np.uint8)
        
        except Exception as e:
            logger.error(f"Error in Chan-Vese level set segmentation: {str(e)}")
            return initial_mask.astype(np.uint8)


class SimpleLevelSet(LevelSetSegmenter):
    """
    Class for simple level set segmentation using SimpleITK.
    
    This class uses SimpleITK's implementation of various level set methods
    for segmentation.
    """
    
    def __init__(self, smoothing: float = 1.0, 
                max_iterations: int = 100,
                convergence_threshold: float = 0.001,
                propagation_scaling: float = 1.0,
                curvature_scaling: float = 1.0,
                advection_scaling: float = 1.0):
        """
        Initialize SimpleITK level set segmenter.
        
        Parameters
        ----------
        smoothing : float, optional
            Smoothing parameter for the level set evolution
        max_iterations : int, optional
            Maximum number of iterations
        convergence_threshold : float, optional
            Threshold for convergence check
        propagation_scaling : float, optional
            Scaling of the propagation term
        curvature_scaling : float, optional
            Scaling of the curvature term
        advection_scaling : float, optional
            Scaling of the advection term
        """
        super().__init__(smoothing, max_iterations, convergence_threshold)
        self.propagation_scaling = propagation_scaling
        self.curvature_scaling = curvature_scaling
        self.advection_scaling = advection_scaling
    
    def segment_slice(self, slice_data: np.ndarray, 
                     initial_mask: np.ndarray) -> np.ndarray:
        """
        Apply SimpleITK level set segmentation to a 2D slice.
        
        Parameters
        ----------
        slice_data : np.ndarray
            2D image slice
        initial_mask : np.ndarray
            Initial binary mask
            
        Returns
        -------
        np.ndarray
            Segmented binary mask
        """
        try:
            # Convert to SimpleITK images
            sitk_image = sitk.GetImageFromArray(slice_data)
            sitk_mask = sitk.GetImageFromArray(initial_mask)
            
            # Create feature image (edge potential)
            feature_image = sitk.GradientMagnitudeRecursiveGaussian(sitk_image, self.smoothing)
            feature_image = sitk.BoundedReciprocal(sitk.Add(feature_image, 1.0))
            
            # Initialize level set
            init_ls = sitk.SignedMaurerDistanceMap(sitk_mask, insideIsPositive=True, useImageSpacing=False)
            
            # Create geodesic active contour filter
            level_set_filter = sitk.GeodesicActiveContourLevelSetImageFilter()
            level_set_filter.SetPropagationScaling(self.propagation_scaling)
            level_set_filter.SetCurvatureScaling(self.curvature_scaling)
            level_set_filter.SetAdvectionScaling(self.advection_scaling)
            level_set_filter.SetMaximumRMSError(self.convergence_threshold)
            level_set_filter.SetNumberOfIterations(self.max_iterations)
            
            # Execute the filter
            level_set = level_set_filter.Execute(init_ls, feature_image)
            
            # Convert to binary mask
            threshold_filter = sitk.BinaryThresholdImageFilter()
            threshold_filter.SetLowerThreshold(-1000)
            threshold_filter.SetUpperThreshold(0.0)
            threshold_filter.SetInsideValue(0)
            threshold_filter.SetOutsideValue(1)
            result = threshold_filter.Execute(level_set)
            
            # Convert back to numpy
            return sitk.GetArrayFromImage(result).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error in SimpleITK level set segmentation: {str(e)}")
            return initial_mask.astype(np.uint8)


# Create convenience functions to instantiate the segmenters

def create_morphological_level_set(**kwargs) -> MorphologicalLevelSet:
    """
    Create and return a MorphologicalLevelSet instance.
    
    Parameters
    ----------
    **kwargs
        Parameters for MorphologicalLevelSet
        
    Returns
    -------
    MorphologicalLevelSet
        Initialized morphological level set segmenter
    """
    return MorphologicalLevelSet(**kwargs)


def create_geodesic_level_set(**kwargs) -> GeodesicLevelSet:
    """
    Create and return a GeodesicLevelSet instance.
    
    Parameters
    ----------
    **kwargs
        Parameters for GeodesicLevelSet
        
    Returns
    -------
    GeodesicLevelSet
        Initialized geodesic level set segmenter
    """
    return GeodesicLevelSet(**kwargs)


def create_chan_vese_level_set(**kwargs) -> ChanVeseLevelSet:
    """
    Create and return a ChanVeseLevelSet instance.
    
    Parameters
    ----------
    **kwargs
        Parameters for ChanVeseLevelSet
        
    Returns
    -------
    ChanVeseLevelSet
        Initialized Chan-Vese level set segmenter
    """
    return ChanVeseLevelSet(**kwargs)


def create_simple_level_set(**kwargs) -> SimpleLevelSet:
    """
    Create and return a SimpleLevelSet instance.
    
    Parameters
    ----------
    **kwargs
        Parameters for SimpleLevelSet
        
    Returns
    -------
    SimpleLevelSet
        Initialized SimpleITK level set segmenter
    """
    return SimpleLevelSet(**kwargs)
