#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Active Contour (Snake) Segmentation Module for QuangTPS.

This module provides functionality for active contour based segmentation,
which is a deformable model that moves under the influence of internal
forces (within the curve) and external forces (from the image).
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import interpolate, ndimage
from skimage import filters, segmentation, feature, morphology, measure
import SimpleITK as sitk

from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ActiveContourSegmenter:
    """
    Class for active contour (snake) segmentation.
    
    This class implements various active contour algorithms for
    semi-automatic image segmentation, focusing on slice-by-slice operations
    with possible 3D extensions.
    """
    
    def __init__(self, alpha: float = 0.01, beta: float = 0.1, 
                gamma: float = 0.01, max_iterations: int = 100,
                convergence_threshold: float = 0.1):
        """
        Initialize active contour segmenter.
        
        Parameters
        ----------
        alpha : float, optional
            Weight of the snake length energy term (elasticity)
        beta : float, optional
            Weight of the snake smoothness energy term (rigidity)
        gamma : float, optional
            Weight of the external force (edge attraction)
        max_iterations : int, optional
            Maximum number of iterations
        convergence_threshold : float, optional
            Threshold for convergence check
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def segment_slice(self, slice_data: np.ndarray, 
                      initial_contour: np.ndarray) -> np.ndarray:
        """
        Apply active contour segmentation to a 2D slice.
        
        Parameters
        ----------
        slice_data : np.ndarray
            2D image slice
        initial_contour : np.ndarray
            Initial contour points as array of shape (n, 2)
            
        Returns
        -------
        np.ndarray
            Final contour points as array of shape (n, 2)
        """
        # Normalize image data to [0, 1]
        image = slice_data.astype(float)
        if np.max(image) > np.min(image):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Create edge map for external energy
        edge_map = filters.sobel(image)
        
        # Convert initial contour to parametric snake format
        snake = initial_contour.copy()
        
        # Run active contour algorithm
        try:
            snake = segmentation.active_contour(
                edge_map,
                snake,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                max_iterations=self.max_iterations,
                convergence=self.convergence_threshold
            )
        except Exception as e:
            logger.error(f"Error in active contour segmentation: {str(e)}")
            return initial_contour
        
        return snake
    
    def segment_volume(self, volume_data: np.ndarray, 
                      initial_contours: List[np.ndarray],
                      slices_with_contours: List[int]) -> List[np.ndarray]:
        """
        Apply active contour segmentation to a 3D volume slice by slice.
        
        Parameters
        ----------
        volume_data : np.ndarray
            3D image volume
        initial_contours : List[np.ndarray]
            List of initial contour points for specific slices
        slices_with_contours : List[int]
            List of slice indices corresponding to initial_contours
            
        Returns
        -------
        List[np.ndarray]
            List of final contour points for all slices
        """
        if len(initial_contours) != len(slices_with_contours):
            raise ValidationError("The number of initial contours must match the number of slice indices")
        
        # Process slices with provided initial contours
        result_contours = {}
        for i, slice_idx in enumerate(slices_with_contours):
            if 0 <= slice_idx < volume_data.shape[0]:
                slice_data = volume_data[slice_idx]
                result_contours[slice_idx] = self.segment_slice(slice_data, initial_contours[i])
        
        # Propagate to other slices by interpolation and segmentation
        all_contours = [None] * volume_data.shape[0]
        for slice_idx in sorted(result_contours.keys()):
            all_contours[slice_idx] = result_contours[slice_idx]
        
        # Propagate contours to empty slices
        self._propagate_contours(volume_data, all_contours)
        
        return all_contours
    
    def _propagate_contours(self, volume_data: np.ndarray, 
                           all_contours: List[Optional[np.ndarray]]) -> None:
        """
        Propagate contours to slices without initial contours.
        
        Parameters
        ----------
        volume_data : np.ndarray
            3D image volume
        all_contours : List[Optional[np.ndarray]]
            List of contour points (None for empty slices)
        """
        # Find slices with contours
        valid_slices = [i for i, contour in enumerate(all_contours) if contour is not None]
        
        if len(valid_slices) < 2:
            return  # Need at least 2 slices for interpolation
        
        # Interpolate between valid slices
        for start_idx, end_idx in zip(valid_slices[:-1], valid_slices[1:]):
            # Skip adjacent slices
            if end_idx - start_idx <= 1:
                continue
            
            start_contour = all_contours[start_idx]
            end_contour = all_contours[end_idx]
            
            # Ensure contours have the same number of points
            n_points = min(len(start_contour), len(end_contour))
            start_contour = self._resample_contour(start_contour, n_points)
            end_contour = self._resample_contour(end_contour, n_points)
            
            # Interpolate for each slice between
            for slice_idx in range(start_idx + 1, end_idx):
                weight = (slice_idx - start_idx) / (end_idx - start_idx)
                interpolated_contour = (1 - weight) * start_contour + weight * end_contour
                
                # Use interpolated contour as initial for active contour
                slice_data = volume_data[slice_idx]
                all_contours[slice_idx] = self.segment_slice(slice_data, interpolated_contour)
    
    def _resample_contour(self, contour: np.ndarray, n_points: int) -> np.ndarray:
        """
        Resample a contour to have a specific number of points.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as array of shape (n, 2)
        n_points : int
            Desired number of points
            
        Returns
        -------
        np.ndarray
            Resampled contour with n_points
        """
        # Create a closed curve
        closed_contour = np.vstack([contour, contour[0]])
        
        # Calculate cumulative distance along the contour
        dist = np.zeros(len(closed_contour))
        for i in range(1, len(closed_contour)):
            dist[i] = dist[i-1] + np.linalg.norm(closed_contour[i] - closed_contour[i-1])
        
        # Create new contour with equidistant points
        new_dist = np.linspace(0, dist[-1], n_points)
        
        # Interpolate x and y coordinates
        fx = interpolate.interp1d(dist, closed_contour[:, 0])
        fy = interpolate.interp1d(dist, closed_contour[:, 1])
        
        # Create new contour
        new_contour = np.zeros((n_points, 2))
        new_contour[:, 0] = fx(new_dist)
        new_contour[:, 1] = fy(new_dist)
        
        return new_contour
    
    def contour_to_mask(self, contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert a contour to a binary mask.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as array of shape (n, 2)
        shape : Tuple[int, int]
            Shape of the output mask (height, width)
            
        Returns
        -------
        np.ndarray
            Binary mask of the contour
        """
        # Create empty mask
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Convert contour to polygon vertices
        r = contour[:, 0].round().astype(int)
        c = contour[:, 1].round().astype(int)
        
        # Clip to image boundaries
        r = np.clip(r, 0, shape[0] - 1)
        c = np.clip(c, 0, shape[1] - 1)
        
        # Create polygon
        rr, cc = polygon(r, c, shape)
        mask[rr, cc] = 1
        
        return mask
    
    def mask_to_contour(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract contour from a binary mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask
            
        Returns
        -------
        np.ndarray
            Contour points as array of shape (n, 2)
        """
        # Find contours
        contours = measure.find_contours(mask, 0.5)
        
        # If multiple contours, select the longest one
        if not contours:
            return np.array([])
        
        if len(contours) > 1:
            lengths = [len(c) for c in contours]
            contour = contours[np.argmax(lengths)]
        else:
            contour = contours[0]
        
        return contour


def polygon(r, c, shape=None):
    """
    Generate coordinates of pixels within a polygon.
    
    Parameters
    ----------
    r : np.ndarray
        Row coordinates of the polygon vertices
    c : np.ndarray
        Column coordinates of the polygon vertices
    shape : Tuple[int, int], optional
        Shape of the output image
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Row and column coordinates of pixels within the polygon
    """
    from skimage.draw import polygon
    return polygon(r, c, shape)


class GVFSnake(ActiveContourSegmenter):
    """
    Class for Gradient Vector Flow (GVF) Snake implementation.
    
    This class extends ActiveContourSegmenter with GVF Snake algorithm,
    which improves the capture range and ability to move into boundary concavities.
    """
    
    def __init__(self, alpha: float = 0.01, beta: float = 0.1, 
                gamma: float = 0.01, mu: float = 0.2, max_iterations: int = 100,
                gvf_iterations: int = 100, convergence_threshold: float = 0.1):
        """
        Initialize GVF snake segmenter.
        
        Parameters
        ----------
        alpha : float, optional
            Weight of the snake length energy term (elasticity)
        beta : float, optional
            Weight of the snake smoothness energy term (rigidity)
        gamma : float, optional
            Weight of the external force (edge attraction)
        mu : float, optional
            Regularization parameter for GVF
        max_iterations : int, optional
            Maximum number of iterations for snake evolution
        gvf_iterations : int, optional
            Maximum number of iterations for GVF computation
        convergence_threshold : float, optional
            Threshold for convergence check
        """
        super().__init__(alpha, beta, gamma, max_iterations, convergence_threshold)
        self.mu = mu
        self.gvf_iterations = gvf_iterations
    
    def _compute_gvf(self, edge_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gradient Vector Flow field.
        
        Parameters
        ----------
        edge_map : np.ndarray
            Edge map of the image
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            GVF field as u and v components
        """
        # Initialize GVF with gradient of edge map
        gy, gx = np.gradient(edge_map)
        f = np.square(gx) + np.square(gy)
        g = f.copy()
        u = gx.copy()
        v = gy.copy()
        
        # Compute GVF iteratively
        for _ in range(self.gvf_iterations):
            # Laplacian of u and v
            u_lap = ndimage.laplace(u)
            v_lap = ndimage.laplace(v)
            
            # Update u and v
            u = u + self.mu * u_lap - g * (u - gx)
            v = v + self.mu * v_lap - g * (v - gy)
        
        return u, v
    
    def segment_slice(self, slice_data: np.ndarray, 
                      initial_contour: np.ndarray) -> np.ndarray:
        """
        Apply GVF snake segmentation to a 2D slice.
        
        Parameters
        ----------
        slice_data : np.ndarray
            2D image slice
        initial_contour : np.ndarray
            Initial contour points as array of shape (n, 2)
            
        Returns
        -------
        np.ndarray
            Final contour points as array of shape (n, 2)
        """
        # Normalize image data to [0, 1]
        image = slice_data.astype(float)
        if np.max(image) > np.min(image):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Create edge map
        edge_map = filters.gaussian(image, 1.0)
        edge_map = filters.sobel(edge_map)
        
        # Compute GVF field
        u, v = self._compute_gvf(edge_map)
        
        # Snake evolution
        snake = initial_contour.copy()
        prev_snake = snake.copy()
        
        for _ in range(self.max_iterations):
            # Compute derivatives along the snake
            n = len(snake)
            indices = np.arange(n)
            
            # Calculate second and fourth derivatives (for internal energy)
            # Using finite difference with cyclic boundary conditions
            prev_indices = (indices - 1) % n
            next_indices = (indices + 1) % n
            prev2_indices = (indices - 2) % n
            next2_indices = (indices + 2) % n
            
            # First order derivatives
            dx = snake[next_indices] - snake[prev_indices]
            
            # Second order derivatives
            d2x = snake[next_indices] + snake[prev_indices] - 2 * snake
            
            # Fourth order derivatives (approximated)
            d4x = (snake[next2_indices] + snake[prev2_indices] - 
                   4 * (snake[next_indices] + snake[prev_indices]) + 
                   6 * snake)
            
            # Linear system components for internal forces
            A = -self.alpha * dx + self.beta * d4x
            
            # External force (interpolated from GVF)
            fx = np.zeros_like(snake)
            for i, (y, x) in enumerate(snake.astype(int)):
                # Ensure coordinates are within bounds
                y = np.clip(y, 0, u.shape[0] - 1)
                x = np.clip(x, 0, u.shape[1] - 1)
                fx[i, 0] = v[y, x]  # GVF y component (rows)
                fx[i, 1] = u[y, x]  # GVF x component (columns)
            
            # Update snake
            snake = snake + A + self.gamma * fx
            
            # Check for convergence
            if np.max(np.abs(snake - prev_snake)) < self.convergence_threshold:
                break
            
            prev_snake = snake.copy()
        
        return snake


# Create convenience function to instantiate the segmenter
def create_active_contour_segmenter(gvf: bool = False, **kwargs) -> Union[ActiveContourSegmenter, GVFSnake]:
    """
    Create and return an ActiveContourSegmenter instance.
    
    Parameters
    ----------
    gvf : bool, optional
        Whether to use GVF snake (True) or regular snake (False)
    **kwargs
        Additional parameters for the segmenter
        
    Returns
    -------
    Union[ActiveContourSegmenter, GVFSnake]
        Initialized active contour segmenter
    """
    if gvf:
        return GVFSnake(**kwargs)
    else:
        return ActiveContourSegmenter(**kwargs)
