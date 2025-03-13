#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Boolean operations on contours.

This module provides functionality for performing boolean operations between contours,
such as union, intersection, subtraction, and exclusive OR. These operations are
essential for creating complex structures in radiotherapy treatment planning.
"""

import logging
import numpy as np
from typing import List
from enum import Enum

from skimage import measure, draw
import cv2

logger = logging.getLogger(__name__)


class BooleanOperation(str, Enum):
    """Enum for different boolean operations."""
    UNION = "UNION"  # A ∪ B: Points in either A or B
    INTERSECTION = "INTERSECTION"  # A ∩ B: Points in both A and B
    SUBTRACTION = "SUBTRACTION"  # A - B: Points in A but not in B
    EXCLUSIVE_OR = "EXCLUSIVE_OR"  # A ⊕ B: Points in either A or B but not both
    

class BooleanOperations:
    """
    Class for performing boolean operations on contours and structure masks.
    
    This class provides methods to combine contours using standard boolean operations:
    union, intersection, subtraction and exclusive OR (XOR). Operations can be performed
    on both 2D contours and 3D structure masks.
    """
    
    @staticmethod
    def union_2d_contours(contour1: np.ndarray, contour2: np.ndarray) -> np.ndarray:
        """
        Perform union operation between two 2D contours.
        
        Parameters
        ----------
        contour1 : np.ndarray
            First contour points as nx2 array
        contour2 : np.ndarray
            Second contour points as nx2 array
            
        Returns
        -------
        np.ndarray
            Resulting contour points
        """
        # Convert contours to masks
        merged_mask = BooleanOperations._contours_to_boolean_mask(
            [contour1, contour2], BooleanOperation.UNION
        )
        
        # Convert mask back to contour
        return BooleanOperations._mask_to_contour(merged_mask)
    
    @staticmethod
    def intersection_2d_contours(contour1: np.ndarray, contour2: np.ndarray) -> np.ndarray:
        """
        Perform intersection operation between two 2D contours.
        
        Parameters
        ----------
        contour1 : np.ndarray
            First contour points as nx2 array
        contour2 : np.ndarray
            Second contour points as nx2 array
            
        Returns
        -------
        np.ndarray
            Resulting contour points
        """
        # Convert contours to masks
        merged_mask = BooleanOperations._contours_to_boolean_mask(
            [contour1, contour2], BooleanOperation.INTERSECTION
        )
        
        # Convert mask back to contour
        return BooleanOperations._mask_to_contour(merged_mask)
    
    @staticmethod
    def subtract_2d_contours(contour1: np.ndarray, contour2: np.ndarray) -> np.ndarray:
        """
        Subtract contour2 from contour1.
        
        Parameters
        ----------
        contour1 : np.ndarray
            Base contour points as nx2 array
        contour2 : np.ndarray
            Contour to subtract as nx2 array
            
        Returns
        -------
        np.ndarray
            Resulting contour points
        """
        # Convert contours to masks
        merged_mask = BooleanOperations._contours_to_boolean_mask(
            [contour1, contour2], BooleanOperation.SUBTRACTION
        )
        
        # Convert mask back to contour
        return BooleanOperations._mask_to_contour(merged_mask)
    
    @staticmethod
    def exclusive_or_2d_contours(contour1: np.ndarray, contour2: np.ndarray) -> np.ndarray:
        """
        Perform exclusive OR operation between two 2D contours.
        
        Parameters
        ----------
        contour1 : np.ndarray
            First contour points as nx2 array
        contour2 : np.ndarray
            Second contour points as nx2 array
            
        Returns
        -------
        np.ndarray
            Resulting contour points
        """
        # Convert contours to masks
        merged_mask = BooleanOperations._contours_to_boolean_mask(
            [contour1, contour2], BooleanOperation.EXCLUSIVE_OR
        )
        
        # Convert mask back to contour
        return BooleanOperations._mask_to_contour(merged_mask)
    
    @staticmethod
    def perform_boolean_operation_2d(contours: List[np.ndarray], 
                                    operation: BooleanOperation) -> np.ndarray:
        """
        Perform specified boolean operation on a list of 2D contours.
        
        Parameters
        ----------
        contours : List[np.ndarray]
            List of contour points arrays
        operation : BooleanOperation
            Boolean operation to perform
            
        Returns
        -------
        np.ndarray
            Resulting contour points
        """
        if not contours:
            return np.array([])
        
        if len(contours) == 1:
            return contours[0]
        
        # Convert contours to mask
        merged_mask = BooleanOperations._contours_to_boolean_mask(contours, operation)
        
        # Convert mask back to contour
        return BooleanOperations._mask_to_contour(merged_mask)
    
    @staticmethod
    def perform_boolean_operation_3d(masks: List[np.ndarray], 
                                    operation: BooleanOperation) -> np.ndarray:
        """
        Perform specified boolean operation on a list of 3D structure masks.
        
        Parameters
        ----------
        masks : List[np.ndarray]
            List of binary masks (3D arrays)
        operation : BooleanOperation
            Boolean operation to perform
            
        Returns
        -------
        np.ndarray
            Resulting 3D mask
        """
        if not masks:
            return np.array([])
        
        if len(masks) == 1:
            return masks[0]
        
        # Ensure all masks have the same dimensions
        shapes = [mask.shape for mask in masks]
        if len(set(shapes)) > 1:
            logger.error(f"Masks have different shapes: {shapes}")
            raise ValueError("All masks must have the same dimensions")
        
        # Initialize with first mask
        result_mask = masks[0].copy()
        
        # Apply operation sequentially
        for i in range(1, len(masks)):
            if operation == BooleanOperation.UNION:
                result_mask = np.logical_or(result_mask, masks[i])
            elif operation == BooleanOperation.INTERSECTION:
                result_mask = np.logical_and(result_mask, masks[i])
            elif operation == BooleanOperation.SUBTRACTION:
                result_mask = np.logical_and(result_mask, np.logical_not(masks[i]))
            elif operation == BooleanOperation.EXCLUSIVE_OR:
                result_mask = np.logical_xor(result_mask, masks[i])
        
        return result_mask.astype(np.uint8)
    
    @staticmethod
    def _contours_to_boolean_mask(contours: List[np.ndarray], 
                                operation: BooleanOperation) -> np.ndarray:
        """
        Convert contours to masks and perform a boolean operation.
        
        Parameters
        ----------
        contours : List[np.ndarray]
            List of contour points arrays
        operation : BooleanOperation
            Boolean operation to perform
            
        Returns
        -------
        np.ndarray
            Resulting binary mask
        """
        if not contours:
            return np.array([])
        
        # Determine the bounds for the mask
        all_points = np.vstack(contours)
        min_x, min_y = np.floor(np.min(all_points, axis=0)).astype(int)
        max_x, max_y = np.ceil(np.max(all_points, axis=0)).astype(int)
        
        # Add padding to avoid edge cases
        padding = 5
        width = max_x - min_x + 2 * padding
        height = max_y - min_y + 2 * padding
        
        # Initialize the mask
        if operation == BooleanOperation.INTERSECTION:
            result_mask = np.ones((height, width), dtype=np.uint8)
        else:
            result_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Convert each contour to a mask and apply boolean operation
        for i, contour in enumerate(contours):
            # Normalize coordinates to mask space
            normalized_contour = contour.copy()
            normalized_contour[:, 0] -= min_x - padding
            normalized_contour[:, 1] -= min_y - padding
            
            # Create mask for this contour
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Convert contour points to integer coordinates
            rr, cc = draw.polygon(normalized_contour[:, 1], normalized_contour[:, 0])
            
            # Filter out points outside the image
            valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            rr, cc = rr[valid_indices], cc[valid_indices]
            
            # Fill the contour
            if len(rr) > 0 and len(cc) > 0:
                contour_mask[rr, cc] = 1
            
            # Apply boolean operation
            if i == 0 and operation != BooleanOperation.INTERSECTION:
                result_mask = contour_mask
            else:
                if operation == BooleanOperation.UNION:
                    result_mask = np.logical_or(result_mask, contour_mask)
                elif operation == BooleanOperation.INTERSECTION:
                    result_mask = np.logical_and(result_mask, contour_mask)
                elif operation == BooleanOperation.SUBTRACTION:
                    result_mask = np.logical_and(result_mask, np.logical_not(contour_mask))
                elif operation == BooleanOperation.EXCLUSIVE_OR:
                    result_mask = np.logical_xor(result_mask, contour_mask)
        
        return result_mask.astype(np.uint8)
    
    @staticmethod
    def _mask_to_contour(mask: np.ndarray) -> np.ndarray:
        """
        Convert a binary mask to contour points.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask
            
        Returns
        -------
        np.ndarray
            Contour points as nx2 array
        """
        if mask.size == 0 or np.all(mask == 0):
            return np.array([])
        
        # Find contours in the mask
        contours = measure.find_contours(mask, 0.5)
        
        # Return the longest contour (main boundary)
        if not contours:
            return np.array([])
        
        # Sort contours by length and take the longest one
        longest_contour = max(contours, key=len)
        
        # Swap x, y coordinates to match expected format
        return np.fliplr(longest_contour)
    
    @staticmethod
    def simplify_contour(contour: np.ndarray, tolerance: float = 0.5) -> np.ndarray:
        """
        Simplify a contour by reducing the number of points while preserving its shape.
        
        Parameters
        ----------
        contour : np.ndarray
            Input contour points as nx2 array
        tolerance : float
            Maximum distance between original and simplified curves
            
        Returns
        -------
        np.ndarray
            Simplified contour points
        """
        if len(contour) <= 3:
            return contour
        
        # Convert to format used by OpenCV
        contour_cv = contour.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply Douglas-Peucker algorithm
        epsilon = tolerance * cv2.arcLength(contour_cv, True)
        simplified = cv2.approxPolyDP(contour_cv, epsilon, True)
        
        # Convert back to nx2 array
        return simplified.reshape(-1, 2)
    
    @staticmethod
    def smooth_contour(contour: np.ndarray, factor: float = 0.25) -> np.ndarray:
        """
        Smooth a contour using Chaikin's algorithm.
        
        Parameters
        ----------
        contour : np.ndarray
            Input contour points as nx2 array
        factor : float
            Smoothing factor (0 to 0.5)
            
        Returns
        -------
        np.ndarray
            Smoothed contour points
        """
        if len(contour) <= 3:
            return contour
        
        # Ensure the contour is closed
        closed = np.allclose(contour[0], contour[-1])
        
        if closed:
            # Use the points without the duplicated end point
            points = contour[:-1]
        else:
            points = contour.copy()
        
        # Apply Chaikin's algorithm
        result = []
        for i in range(len(points)):
            p0 = points[i]
            p1 = points[(i + 1) % len(points)]
            
            # Create two new points
            q0 = p0 * (1 - factor) + p1 * factor
            q1 = p0 * factor + p1 * (1 - factor)
            
            result.append(q0)
            result.append(q1)
        
        # Convert to numpy array
        smoothed = np.array(result)
        
        # Make sure the contour is still closed if the original was
        if closed:
            smoothed = np.vstack([smoothed, smoothed[0]])
        
        return smoothed