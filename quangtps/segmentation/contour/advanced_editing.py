#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for advanced contour editing operations.

This module provides functionality for advanced contour manipulations such as
push/pull operations, smoothing, contour simplification, and automatic contour
correction - essential features for efficient contour editing in radiotherapy
treatment planning.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage import measure, morphology, draw, filters
from scipy import ndimage, interpolate, spatial

logger = logging.getLogger(__name__)


class EditOperation(str, Enum):
    """Enum for different contour editing operations."""
    PUSH = "PUSH"  # Push contour outward at a point
    PULL = "PULL"  # Pull contour inward at a point
    SMOOTH = "SMOOTH"  # Smooth a region of the contour
    SHARPEN = "SHARPEN"  # Sharpen a region of the contour
    SIMPLIFY = "SIMPLIFY"  # Reduce number of points while preserving shape
    RESAMPLE = "RESAMPLE"  # Resample contour to have evenly spaced points


class ContourEditor:
    """
    Class providing advanced contour editing capabilities.
    
    This class implements various algorithms for manipulating contours,
    such as local deformations, smoothing, and geometric operations
    to facilitate precise contour editing.
    """
    
    def __init__(self):
        """Initialize contour editor with default parameters."""
        # Default parameters
        self.brush_radius = 10.0  # Default brush radius in pixels
        self.push_pull_strength = 0.5  # Strength factor for push/pull operations (0-1)
        self.smoothing_strength = 0.5  # Strength factor for smoothing (0-1)
        self.simplification_tolerance = 1.0  # Tolerance for contour simplification in pixels
    
    def push_pull_contour(self, 
                        contour: np.ndarray, 
                        point: Tuple[float, float], 
                        operation: EditOperation,
                        radius: Optional[float] = None,
                        strength: Optional[float] = None) -> np.ndarray:
        """
        Push or pull a contour locally around a point.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
        point : Tuple[float, float]
            Center point of the operation
        operation : EditOperation
            PUSH or PULL operation
        radius : float, optional
            Radius of influence in pixels
        strength : float, optional
            Strength of the operation (0-1)
            
        Returns
        -------
        np.ndarray
            Modified contour
        """
        if contour.size == 0 or len(contour) < 3:
            return contour
        
        # Use default parameters if not specified
        if radius is None:
            radius = self.brush_radius
        
        if strength is None:
            strength = self.push_pull_strength
        
        # Ensure strength is between 0 and 1
        strength = max(0.0, min(1.0, strength))
        
        # Convert to numpy array if it's a list
        if isinstance(contour, list):
            contour = np.array(contour)
        
        # Create a copy of the contour
        result = contour.copy()
        
        # Calculate distances from each contour point to the center point
        distances = np.sqrt(np.sum((result - point)**2, axis=1))
        
        # Find points within the radius
        mask = distances <= radius
        if not np.any(mask):
            return result  # No points within radius
        
        # Calculate influence weights based on distance (bell curve falloff)
        weights = np.exp(-0.5 * (distances[mask] / (radius / 2))**2)
        
        # Normalize weights
        weights = weights / np.max(weights)
        
        # Calculate vectors from center to contour points
        vectors = result[mask] - point
        
        # Normalize vectors
        lengths = np.sqrt(np.sum(vectors**2, axis=1))
        # Avoid division by zero
        lengths[lengths == 0] = 1
        normalized_vectors = vectors / lengths[:, np.newaxis]
        
        # Apply push/pull operation
        if operation == EditOperation.PUSH:
            # Move points outward
            result[mask] += normalized_vectors * strength * weights[:, np.newaxis] * radius
        elif operation == EditOperation.PULL:
            # Move points inward
            result[mask] -= normalized_vectors * strength * weights[:, np.newaxis] * radius
        
        return result
    
    def smooth_contour(self, 
                     contour: np.ndarray,
                     point: Optional[Tuple[float, float]] = None,
                     radius: Optional[float] = None,
                     strength: Optional[float] = None,
                     iterations: int = 1) -> np.ndarray:
        """
        Smooth a contour, either entirely or locally around a point.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
        point : Tuple[float, float], optional
            Center point for local smoothing, if None, entire contour is smoothed
        radius : float, optional
            Radius of influence for local smoothing
        strength : float, optional
            Strength of smoothing effect (0-1)
        iterations : int, optional
            Number of smoothing iterations
            
        Returns
        -------
        np.ndarray
            Smoothed contour
        """
        if contour.size == 0 or len(contour) < 3:
            return contour
        
        # Use default parameters if not specified
        if radius is None:
            radius = self.brush_radius
        
        if strength is None:
            strength = self.smoothing_strength
        
        # Ensure strength is between 0 and 1
        strength = max(0.0, min(1.0, strength))
        
        # Convert to numpy array if it's a list
        if isinstance(contour, list):
            contour = np.array(contour)
        
        # Create a copy of the contour
        result = contour.copy()
        
        # Determine if this is a closed contour
        is_closed = np.all(result[0] == result[-1])
        
        # Handle local vs. global smoothing
        if point is not None:
            # Local smoothing - calculate distances to determine affected points
            distances = np.sqrt(np.sum((result - point)**2, axis=1))
            mask = distances <= radius
            
            if not np.any(mask):
                return result  # No points within radius
            
            # Calculate influence weights
            weights = np.exp(-0.5 * (distances / (radius / 2))**2)
            weights = weights / np.max(weights)
        else:
            # Global smoothing - all points affected
            mask = np.ones(len(result), dtype=bool)
            weights = np.ones(len(result))
        
        # Apply Laplacian smoothing for the specified number of iterations
        for _ in range(iterations):
            # Create a copy for calculations
            temp = result.copy()
            
            # Apply smoothing to each affected point
            for i in range(len(result)):
                if not mask[i]:
                    continue
                
                # Get indices of neighboring points
                if is_closed:
                    # For closed contours, wrap around
                    prev_idx = (i - 1) % len(result)
                    next_idx = (i + 1) % len(result)
                else:
                    # For open contours, handle endpoints specially
                    if i == 0:
                        # First point - only has one neighbor
                        prev_idx = i
                        next_idx = i + 1
                    elif i == len(result) - 1:
                        # Last point - only has one neighbor
                        prev_idx = i - 1
                        next_idx = i
                    else:
                        # Interior point - has two neighbors
                        prev_idx = i - 1
                        next_idx = i + 1
                
                # Calculate new position using Laplacian smoothing
                new_pos = (result[prev_idx] + result[next_idx]) / 2
                
                # Apply weighted adjustment based on strength
                temp[i] = result[i] + (new_pos - result[i]) * strength * weights[i]
            
            # Update result with smoothed positions
            result = temp.copy()
        
        return result
    
    def simplify_contour(self, 
                       contour: np.ndarray,
                       tolerance: Optional[float] = None) -> np.ndarray:
        """
        Simplify a contour by reducing the number of points while preserving shape.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
        tolerance : float, optional
            Tolerance level for simplification in pixels
            
        Returns
        -------
        np.ndarray
            Simplified contour
        """
        if contour.size == 0 or len(contour) < 3:
            return contour
        
        # Use default tolerance if not specified
        if tolerance is None:
            tolerance = self.simplification_tolerance
        
        # Convert to numpy array if it's a list
        if isinstance(contour, list):
            contour = np.array(contour)
        
        # Check if this is a closed contour
        is_closed = np.all(contour[0] == contour[-1])
        
        # Remove the last point for processing if it's a closed contour
        if is_closed:
            contour_to_process = contour[:-1]
        else:
            contour_to_process = contour
        
        # Apply Douglas-Peucker algorithm for curve simplification
        mask = np.ones(len(contour_to_process), dtype=bool)
        
        # Start with all points and recursively simplify
        self._douglas_peucker(contour_to_process, 0, len(contour_to_process) - 1, mask, tolerance)
        
        # Apply the mask to get the simplified contour
        simplified = contour_to_process[mask]
        
        # If it was a closed contour, add the first point at the end again
        if is_closed:
            simplified = np.vstack([simplified, simplified[0]])
        
        return simplified
    
    def _douglas_peucker(self, 
                       points: np.ndarray, 
                       start_idx: int, 
                       end_idx: int, 
                       mask: np.ndarray, 
                       tolerance: float):
        """
        Recursive implementation of Douglas-Peucker algorithm.
        
        Parameters
        ----------
        points : np.ndarray
            Contour points
        start_idx : int
            Start index of the current segment
        end_idx : int
            End index of the current segment
        mask : np.ndarray
            Boolean mask of points to keep
        tolerance : float
            Tolerance level for simplification
        """
        if end_idx <= start_idx + 1:
            return
        
        # Find the point with the maximum distance
        max_dist = 0
        max_idx = start_idx
        
        start_point = points[start_idx]
        end_point = points[end_idx]
        
        # Calculate the line vector
        line_vec = end_point - start_point
        line_length = np.linalg.norm(line_vec)
        
        if line_length == 0:
            # Start and end points are identical, so just keep start and end
            mask[start_idx+1:end_idx] = False
            return
        
        # Normalize the line vector
        line_vec = line_vec / line_length
        
        # Calculate perpendicular distances from each point to the line
        for i in range(start_idx + 1, end_idx):
            # Vector from start point to current point
            vec = points[i] - start_point
            
            # Project onto line
            proj = np.dot(vec, line_vec)
            
            # Calculate perpendicular distance
            perp_vec = vec - proj * line_vec
            dist = np.linalg.norm(perp_vec)
            
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # If the maximum distance is greater than the tolerance, keep this point
        # and recursively process the two resulting segments
        if max_dist > tolerance:
            # Process the segment before the max_idx point
            self._douglas_peucker(points, start_idx, max_idx, mask, tolerance)
            
            # Process the segment after the max_idx point
            self._douglas_peucker(points, max_idx, end_idx, mask, tolerance)
        else:
            # If all points in this segment are within tolerance, remove them
            mask[start_idx+1:end_idx] = False
    
    def resample_contour(self, 
                       contour: np.ndarray, 
                       num_points: Optional[int] = None,
                       spacing: Optional[float] = None) -> np.ndarray:
        """
        Resample a contour to have evenly spaced points.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
        num_points : int, optional
            Desired number of points after resampling
        spacing : float, optional
            Desired spacing between points in pixels
            
        Returns
        -------
        np.ndarray
            Resampled contour
        """
        if contour.size == 0 or len(contour) < 3:
            return contour
        
        # Convert to numpy array if it's a list
        if isinstance(contour, list):
            contour = np.array(contour)
        
        # Check if this is a closed contour
        is_closed = np.all(contour[0] == contour[-1])
        
        # Calculate the cumulative distance along the contour
        distances = np.zeros(len(contour))
        for i in range(1, len(contour)):
            distances[i] = distances[i-1] + np.linalg.norm(contour[i] - contour[i-1])
        
        total_length = distances[-1]
        
        # Determine how many points to create
        if num_points is not None:
            new_count = num_points
        elif spacing is not None:
            new_count = int(total_length / spacing) + 1
        else:
            # Default: keep the same number of points
            new_count = len(contour)
        
        # Make sure we have at least 3 points for a proper contour
        new_count = max(3, new_count)
        
        # Create new evenly spaced distance values
        if is_closed:
            # For closed contours, distribute points evenly
            new_distances = np.linspace(0, total_length, new_count, endpoint=False)
        else:
            # For open contours, include both endpoints
            new_distances = np.linspace(0, total_length, new_count)
        
        # Interpolate x and y coordinates at the new distances
        result_x = np.interp(new_distances, distances, contour[:, 0])
        result_y = np.interp(new_distances, distances, contour[:, 1])
        
        # Combine into a new contour
        result = np.column_stack((result_x, result_y))
        
        # If it was a closed contour, append the first point at the end
        if is_closed:
            result = np.vstack([result, result[0]])
        
        return result
    
    def sharpen_contour(self, 
                      contour: np.ndarray,
                      point: Optional[Tuple[float, float]] = None,
                      radius: Optional[float] = None,
                      strength: Optional[float] = None) -> np.ndarray:
        """
        Sharpen a contour by accentuating changes in direction.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
        point : Tuple[float, float], optional
            Center point for local sharpening, if None, entire contour is sharpened
        radius : float, optional
            Radius of influence for local sharpening
        strength : float, optional
            Strength of sharpening effect (0-1)
            
        Returns
        -------
        np.ndarray
            Sharpened contour
        """
        if contour.size == 0 or len(contour) < 3:
            return contour
        
        # Use default parameters if not specified
        if radius is None:
            radius = self.brush_radius
        
        if strength is None:
            strength = self.smoothing_strength
        
        # Ensure strength is between 0 and 1
        strength = max(0.0, min(1.0, strength))
        
        # Convert to numpy array if it's a list
        if isinstance(contour, list):
            contour = np.array(contour)
        
        # Create a copy of the contour
        result = contour.copy()
        
        # First smooth the contour to get a reference
        smoothed = self.smooth_contour(contour, point, radius, 0.5, 1)
        
        # Calculate the difference between original and smoothed
        diff = contour - smoothed
        
        # Apply the difference in the opposite direction to sharpen
        if point is not None:
            # Local sharpening
            distances = np.sqrt(np.sum((result - point)**2, axis=1))
            weights = np.exp(-0.5 * (distances / (radius / 2))**2)
            weights = weights / np.max(weights)
            mask = distances <= radius
            
            if np.any(mask):
                # Apply weighted adjustment
                result[mask] = contour[mask] + diff[mask] * strength * weights[mask, np.newaxis]
        else:
            # Global sharpening
            result = contour + diff * strength
        
        return result
    
    def apply_operation(self,
                      contour: np.ndarray,
                      point: Tuple[float, float],
                      operation: EditOperation,
                      radius: Optional[float] = None,
                      strength: Optional[float] = None,
                      iterations: int = 1) -> np.ndarray:
        """
        Apply a specified editing operation to a contour.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
        point : Tuple[float, float]
            Center point of the operation
        operation : EditOperation
            Type of operation to apply
        radius : float, optional
            Radius of influence
        strength : float, optional
            Strength of the operation
        iterations : int, optional
            Number of iterations for operations like smoothing
            
        Returns
        -------
        np.ndarray
            Modified contour
        """
        if contour.size == 0:
            return contour
        
        # Apply the requested operation
        if operation == EditOperation.PUSH or operation == EditOperation.PULL:
            return self.push_pull_contour(contour, point, operation, radius, strength)
        
        elif operation == EditOperation.SMOOTH:
            return self.smooth_contour(contour, point, radius, strength, iterations)
        
        elif operation == EditOperation.SHARPEN:
            return self.sharpen_contour(contour, point, radius, strength)
        
        elif operation == EditOperation.SIMPLIFY:
            return self.simplify_contour(contour, strength)
        
        elif operation == EditOperation.RESAMPLE:
            # For resampling, use strength as a factor to determine number of points
            current_points = len(contour)
            if strength is None:
                strength = 1.0
                
            # Strength 0.5 = same number of points, 0 = fewer points, 1 = more points
            factor = 0.5 + strength
            num_points = int(current_points * factor)
            
            return self.resample_contour(contour, num_points)
        
        # Unknown operation
        logger.warning(f"Unknown operation: {operation}")
        return contour
    
    def auto_smooth(self, contour: np.ndarray, level: float = 0.5) -> np.ndarray:
        """
        Automatically smooth a contour with adaptive strength.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
        level : float, optional
            Overall smoothing level (0-1)
            
        Returns
        -------
        np.ndarray
            Smoothed contour
        """
        if contour.size == 0 or len(contour) < 3:
            return contour
        
        # Apply smoothing proportional to contour complexity
        return self.smooth_contour(contour, None, None, level, 3)
    
    def auto_correct(self, contour: np.ndarray) -> np.ndarray:
        """
        Automatically correct common contour issues like self-intersections.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
            
        Returns
        -------
        np.ndarray
            Corrected contour
        """
        if contour.size == 0 or len(contour) < 3:
            return contour
        
        # Convert to numpy array if it's a list
        if isinstance(contour, list):
            contour = np.array(contour)
        
        # First, check if this is a closed contour
        is_closed = np.all(contour[0] == contour[-1])
        
        # Create simplified version to reduce complexity
        simplified = self.simplify_contour(contour, 0.5)
        
        # If contour is too simple already, just return original
        if len(simplified) <= 4:
            return contour
        
        # Check for self-intersections
        has_intersections = self._check_self_intersections(simplified)
        
        if has_intersections:
            # Try progressively stronger simplification to remove intersections
            for tolerance in [1.0, 2.0, 3.0]:
                corrected = self.simplify_contour(contour, tolerance)
                if not self._check_self_intersections(corrected):
                    # Found a solution
                    return corrected
            
            # If simplification doesn't work, try smoothing
            corrected = self.smooth_contour(contour, None, None, 0.7, 3)
            if not self._check_self_intersections(corrected):
                return corrected
        
        # If no intersections or couldn't fix, return original
        return contour
    
    def _check_self_intersections(self, contour: np.ndarray) -> bool:
        """
        Check if a contour has self-intersections.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points
            
        Returns
        -------
        bool
            True if contour has self-intersections
        """
        n = len(contour)
        
        # Need at least 4 points to have intersections
        if n < 4:
            return False
        
        # Check each segment against all non-adjacent segments
        for i in range(n - 1):
            for j in range(i + 2, n - 1):
                # Skip adjacent segments
                if i == 0 and j == n - 2:
                    continue
                
                # Check if segments intersect
                p1, p2 = contour[i], contour[i + 1]
                p3, p4 = contour[j], contour[j + 1]
                
                if self._segments_intersect(p1, p2, p3, p4):
                    return True
        
        return False
    
    def _segments_intersect(self, p1, p2, p3, p4):
        """
        Check if two line segments intersect.
        
        Parameters
        ----------
        p1, p2 : array-like
            Endpoints of first segment
        p3, p4 : array-like
            Endpoints of second segment
            
        Returns
        -------
        bool
            True if segments intersect
        """
        # Calculate the orientation of three points
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise
        
        # Check if point q is on segment pr
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        # Calculate the four orientations required
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases
        if o1 == 0 and on_segment(p1, p3, p2): return True
        if o2 == 0 and on_segment(p1, p4, p2): return True
        if o3 == 0 and on_segment(p3, p1, p4): return True
        if o4 == 0 and on_segment(p3, p2, p4): return True
        
        return False
