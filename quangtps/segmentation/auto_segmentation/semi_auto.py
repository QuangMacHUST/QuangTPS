
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semi-Automatic Segmentation Module for QuangTPS.

This module provides semi-automatic segmentation methods that combine user input
with automated segmentation techniques for radiotherapy treatment planning.
"""

import os
import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from skimage import measure, morphology, filters, segmentation

from quangtps.core.config import Config
from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

class SemiAutoSegmentor:
    """
    Class for semi-automatic segmentation methods.
    
    This class implements various interactive segmentation methods that combine
    user input (e.g., seed points, initial contours) with automated algorithms
    to generate segmentation results.
    """
    
    def __init__(self):
        """Initialize semi-automatic segmentor."""
        self.config = Config.get_instance()
        
    def livewire(self, image: np.ndarray, seed_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Implement live-wire (intelligent scissors) segmentation.
        
        This method uses Dijkstra's algorithm to find the minimum cost path
        between seed points based on image gradients.
        
        Parameters
        ----------
        image : np.ndarray
            Image to segment
        seed_points : List[Tuple[int, int]]
            List of user-defined seed points
            
        Returns
        -------
        List[Tuple[int, int]]
            List of points forming the contour
        """
        try:
            if len(seed_points) < 2:
                raise ValidationError("At least two seed points are required for livewire segmentation")
            
            # Convert to grayscale if needed
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Calculate gradient magnitude as cost function
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Invert gradient to use as cost (lower cost for strong edges)
            max_gradient = np.max(gradient_magnitude)
            if max_gradient > 0:
                cost = 1.0 - gradient_magnitude / max_gradient
            else:
                cost = np.ones_like(gradient_magnitude)
            
            # Create full contour by connecting seed points
            contour_points = []
            
            for i in range(len(seed_points)):
                start_point = seed_points[i]
                end_point = seed_points[(i + 1) % len(seed_points)]  # Wrap around
                
                # Find path between current seed point and next seed point
                path = self._find_min_cost_path(cost, start_point, end_point)
                
                # Add path points to contour (excluding last point to avoid duplication)
                if i < len(seed_points) - 1:
                    contour_points.extend(path[:-1])
                else:
                    contour_points.extend(path)
            
            return contour_points
            
        except Exception as e:
            logger.error(f"Error in livewire segmentation: {str(e)}")
            raise ValidationError(f"Error in livewire segmentation: {str(e)}")
    
    def _find_min_cost_path(self, cost_map: np.ndarray, start: Tuple[int, int], 
                           end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find minimum cost path using Dijkstra's algorithm.
        
        Parameters
        ----------
        cost_map : np.ndarray
            Cost map where higher values indicate higher costs
        start : Tuple[int, int]
            Starting point (y, x)
        end : Tuple[int, int]
            Ending point (y, x)
            
        Returns
        -------
        List[Tuple[int, int]]
            Points forming the minimum cost path
        """
        # Initialize data structures
        height, width = cost_map.shape
        distances = np.full((height, width), np.inf)
        visited = np.zeros((height, width), dtype=bool)
        previous = np.zeros((height, width, 2), dtype=int)
        
        # Start point
        y_start, x_start = start
        distances[y_start, x_start] = 0
        
        # Define 8-connected neighbors
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Dijkstra's algorithm
        while True:
            # Find unvisited node with minimum distance
            min_dist = np.inf
            current = None
            
            unvisited_y, unvisited_x = np.where(~visited & (distances < np.inf))
            
            if len(unvisited_y) == 0:
                break
                
            for i in range(len(unvisited_y)):
                y, x = unvisited_y[i], unvisited_x[i]
                if distances[y, x] < min_dist:
                    min_dist = distances[y, x]
                    current = (y, x)
            
            if current is None or current == end:
                break
                
            y_current, x_current = current
            visited[y_current, x_current] = True
            
            # Update distances to neighbors
            for dy, dx in neighbors:
                y_neighbor = y_current + dy
                x_neighbor = x_current + dx
                
                # Check bounds
                if (0 <= y_neighbor < height and 0 <= x_neighbor < width and 
                    not visited[y_neighbor, x_neighbor]):
                    
                    # For diagonal neighbors, use Euclidean distance
                    if dx != 0 and dy != 0:
                        step_cost = 1.414 * cost_map[y_neighbor, x_neighbor]
                    else:
                        step_cost = cost_map[y_neighbor, x_neighbor]
                    
                    new_distance = distances[y_current, x_current] + step_cost
                    
                    if new_distance < distances[y_neighbor, x_neighbor]:
                        distances[y_neighbor, x_neighbor] = new_distance
                        previous[y_neighbor, x_neighbor] = [y_current, x_current]
        
        # Reconstruct path
        path = []
        current = end
        
        # Check if endpoint was reached
        y_end, x_end = end
        if distances[y_end, x_end] == np.inf:
            # If no path found, create straight line path
            path = self._create_straight_line_path(start, end)
        else:
            # Reconstruct the path from end to start
            while current != start:
                path.append(current)
                y, x = current
                prev_y, prev_x = previous[y, x]
                current = (prev_y, prev_x)
                
            path.append(start)
            path.reverse()
        
        return path
    
    def _create_straight_line_path(self, start: Tuple[int, int], 
                                  end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Create a straight line path between two points using Bresenham's algorithm.
        
        Parameters
        ----------
        start : Tuple[int, int]
            Starting point (y, x)
        end : Tuple[int, int]
            Ending point (y, x)
            
        Returns
        -------
        List[Tuple[int, int]]
            Points forming a straight line
        """
        y0, x0 = start
        y1, x1 = end
        
        # Swap coordinates for Bresenham's algorithm
        # (Bresenham works on (x,y) but our image coordinates are (y,x))
        line_points = list(zip(*line(x0, y0, x1, y1)))
        
        # Convert back to (y,x) format
        path = [(y, x) for x, y in line_points]
        
        return path
    
    def active_contour(self, image: np.ndarray, initial_contour: np.ndarray, 
                      alpha: float = 0.01, beta: float = 0.1, 
                      gamma: float = 0.01, iterations: int = 100) -> np.ndarray:
        """
        Active contour (snake) segmentation.
        
        Parameters
        ----------
        image : np.ndarray
            Image to segment
        initial_contour : np.ndarray
            Initial contour points as array of shape (n, 2)
        alpha : float, optional
            Snake elasticity parameter
        beta : float, optional
            Snake stiffness parameter
        gamma : float, optional
            External force weight
        iterations : int, optional
            Maximum number of iterations
            
        Returns
        -------
        np.ndarray
            Final contour points
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Calculate gradient magnitude (external force)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Create external force field
            external_force = np.stack((-sobelx, -sobely), axis=2)
            
            # Normalize external force
            max_magnitude = np.max(np.sqrt(np.sum(external_force**2, axis=2)))
            if max_magnitude > 0:
                external_force = external_force / max_magnitude
            
            # Run active contour algorithm
            snake = segmentation.active_contour(
                gray,
                initial_contour,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                max_iterations=iterations
            )
            
            return snake
            
        except Exception as e:
            logger.error(f"Error in active contour segmentation: {str(e)}")
            raise ValidationError(f"Error in active contour segmentation: {str(e)}")
    
    def region_growing(self, image: np.ndarray, seed_point: Tuple[int, int], 
                      tolerance: float = 0.1, connectivity: int = 8) -> np.ndarray:
        """
        Interactive region growing segmentation.
        
        Parameters
        ----------
        image : np.ndarray
            Image to segment
        seed_point : Tuple[int, int]
            Seed point (y, x)
        tolerance : float, optional
            Intensity tolerance (fraction of image intensity range)
        connectivity : int, optional
            Pixel connectivity (4 or 8)
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Create seed array
            seed_array = np.zeros_like(gray, dtype=bool)
            y, x = seed_point
            seed_array[y, x] = True
            
            # Calculate tolerance value
            img_min, img_max = np.min(gray), np.max(gray)
            tol_value = tolerance * (img_max - img_min)
            
            # Get seed value
            seed_value = gray[y, x]
            
            # Create masks for lower and upper bounds
            lower_bound = gray >= (seed_value - tol_value)
            upper_bound = gray <= (seed_value + tol_value)
            in_range = lower_bound & upper_bound
            
            # Apply region growing
            if connectivity == 4:
                struct_elem = morphology.diamond(1)
            else:  # connectivity == 8
                struct_elem = morphology.square(3)
                
            mask = morphology.flood(gray, seed_point, connectivity=connectivity)
            mask = mask & in_range
            
            return mask
            
        except Exception as e:
            logger.error(f"Error in region growing segmentation: {str(e)}")
            raise ValidationError(f"Error in region growing segmentation: {str(e)}")
    
    def watershed(self, image: np.ndarray, markers: np.ndarray) -> np.ndarray:
        """
        Watershed segmentation.
        
        Parameters
        ----------
        image : np.ndarray
            Image to segment
        markers : np.ndarray
            Initial markers where different positive integers represent different regions
            
        Returns
        -------
        np.ndarray
            Segmentation result
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Calculate gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(sobelx**2 + sobely**2)
            
            # Normalize gradient
            gradient = cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)
            
            # Apply watershed algorithm
            segments = segmentation.watershed(gradient, markers, watershed_line=True)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error in watershed segmentation: {str(e)}")
            raise ValidationError(f"Error in watershed segmentation: {str(e)}")
    
    def graph_cut(self, image: np.ndarray, foreground_seeds: List[Tuple[int, int]], 
                 background_seeds: List[Tuple[int, int]]) -> np.ndarray:
        """
        Graph cut segmentation.
        
        Parameters
        ----------
        image : np.ndarray
            Image to segment
        foreground_seeds : List[Tuple[int, int]]
            List of foreground seed points (y, x)
        background_seeds : List[Tuple[int, int]]
            List of background seed points (y, x)
            
        Returns
        -------
        np.ndarray
            Binary mask of segmented region
        """
        try:
            # For graph cut, we need OpenCV's grabcut
            # Initialize mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Set foreground and background seeds
            for y, x in foreground_seeds:
                mask[y, x] = cv2.GC_FGD  # Definite foreground
                
            for y, x in background_seeds:
                mask[y, x] = cv2.GC_BGD  # Definite background
            
            # Set remaining pixels as probable background
            mask[mask == 0] = cv2.GC_PR_BGD
            
            # Run grabcut algorithm
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Ensure image is in correct format
            if len(image.shape) == 2:
                # Convert grayscale to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image.copy()
            
            # Run GrabCut
            cv2.grabCut(image_rgb, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            
            # Extract result
            result_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
            
            return result_mask
            
        except Exception as e:
            logger.error(f"Error in graph cut segmentation: {str(e)}")
            raise ValidationError(f"Error in graph cut segmentation: {str(e)}")
    
    def level_set(self, image: np.ndarray, init_ls: np.ndarray, 
                 num_iter: int = 100, lambda1: float = 1.0, 
                 lambda2: float = 1.0) -> np.ndarray:
        """
        Level set segmentation.
        
        Parameters
        ----------
        image : np.ndarray
            Image to segment
        init_ls : np.ndarray
            Initial level set
        num_iter : int, optional
            Number of iterations
        lambda1 : float, optional
            Weight parameter for inside region
        lambda2 : float, optional
            Weight parameter for outside region
            
        Returns
        -------
        np.ndarray
            Final level set
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Normalize image to [0, 1]
            gray = gray.astype(np.float64)
            gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))
            
            # Chan-Vese segmentation
            final_ls = segmentation.chan_vese(
                gray,
                init_level_set=init_ls,
                max_iter=num_iter,
                lambda1=lambda1,
                lambda2=lambda2
            )
            
            return final_ls
            
        except Exception as e:
            logger.error(f"Error in level set segmentation: {str(e)}")
            raise ValidationError(f"Error in level set segmentation: {str(e)}")
    
    def get_contour_from_mask(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Extract contours from a binary mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask
            
        Returns
        -------
        List[np.ndarray]
            List of contours, each as a numpy array of shape (n, 1, 2)
        """
        try:
            # Ensure mask is binary and of type uint8
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            return contours
            
        except Exception as e:
            logger.error(f"Error extracting contours from mask: {str(e)}")
            raise ValidationError(f"Error extracting contours from mask: {str(e)}")
    
    def get_mask_from_contour(self, contours: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a binary mask from contours.
        
        Parameters
        ----------
        contours : List[np.ndarray]
            List of contours
        shape : Tuple[int, int]
            Shape of the output mask (height, width)
            
        Returns
        -------
        np.ndarray
            Binary mask
        """
        try:
            # Create empty mask
            mask = np.zeros(shape, dtype=np.uint8)
            
            # Fill contours
            cv2.drawContours(mask, contours, -1, 1, -1)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error creating mask from contours: {str(e)}")
            raise ValidationError(f"Error creating mask from contours: {str(e)}")


# Helper function for Bresenham's line algorithm
def line(x0, y0, x1, y1):
    """
    Implementation of Bresenham's line algorithm.
    
    Parameters
    ----------
    x0, y0 : int
        Starting point coordinates
    x1, y1 : int
        Ending point coordinates
        
    Returns
    -------
    Generator
        Yields (x, y) coordinates along the line
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        yield (x0, y0)
        
        if x0 == x1 and y0 == y1:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy