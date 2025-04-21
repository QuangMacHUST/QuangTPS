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
from typing import List, Tuple, Dict, Any, Optional
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
    def intersection_2d_contours(
        contour1: np.ndarray, contour2: np.ndarray
    ) -> np.ndarray:
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
    def exclusive_or_2d_contours(
        contour1: np.ndarray, contour2: np.ndarray
    ) -> np.ndarray:
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
    def perform_boolean_operation_2d(
        contours: List[np.ndarray], operation: BooleanOperation
    ) -> np.ndarray:
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
    def perform_boolean_operation_3d(
        masks: List[np.ndarray], operation: BooleanOperation
    ) -> np.ndarray:
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
    def _contours_to_boolean_mask(
        contours: List[np.ndarray], operation: BooleanOperation
    ) -> np.ndarray:
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

        # Create mask for each contour
        masks = []
        for contour in contours:
            mask = np.zeros((height, width), dtype=np.uint8)

            # Normalize coordinates to the mask space
            normalized_contour = contour.copy()
            normalized_contour[:, 0] -= min_x - padding
            normalized_contour[:, 1] -= min_y - padding

            # Convert contour points to integer coordinates
            points = normalized_contour.astype(np.int32)

            # Draw the contour
            cv2.fillPoly(mask, [points], (1,))
            masks.append(mask)

        # Apply the operation
        for i, mask in enumerate(masks):
            if i == 0 and operation != BooleanOperation.INTERSECTION:
                result_mask = mask.copy()
            else:
                if operation == BooleanOperation.UNION:
                    result_mask = np.logical_or(result_mask, mask).astype(np.uint8)
                elif operation == BooleanOperation.INTERSECTION:
                    result_mask = np.logical_and(result_mask, mask).astype(np.uint8)
                elif operation == BooleanOperation.SUBTRACTION:
                    if i > 0:  # Only subtract after the first mask
                        result_mask = np.logical_and(
                            result_mask, np.logical_not(mask)
                        ).astype(np.uint8)
                elif operation == BooleanOperation.EXCLUSIVE_OR:
                    result_mask = np.logical_xor(result_mask, mask).astype(np.uint8)

        return result_mask

    @staticmethod
    def _mask_to_contour(mask: np.ndarray) -> np.ndarray:
        """
        Convert a binary mask to a contour.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask

        Returns
        -------
        np.ndarray
            Contour points
        """
        if mask.size == 0 or np.max(mask) == 0:
            return np.array([])

        # Find contours in the mask
        contours = measure.find_contours(mask, 0.5)

        if not contours:
            return np.array([])

        # Select the longest contour
        longest_contour = max(contours, key=len)

        # Simplify the contour
        simplified_contour = BooleanOperations.simplify_contour(longest_contour)

        # Swap x and y coordinates (find_contours returns [y, x])
        result_contour = np.fliplr(simplified_contour)

        return result_contour

    @staticmethod
    def simplify_contour(contour: np.ndarray, tolerance: float = 0.5) -> np.ndarray:
        """
        Simplify a contour by removing redundant points.

        Parameters
        ----------
        contour : np.ndarray
            Input contour points
        tolerance : float
            Tolerance parameter for simplification

        Returns
        -------
        np.ndarray
            Simplified contour points
        """
        if len(contour) <= 3:
            return contour

        # Convert to OpenCV format
        cv_contour = np.expand_dims(contour.astype(np.float32), 1)

        # Apply Douglas-Peucker algorithm
        epsilon = tolerance * cv2.arcLength(cv_contour, True)
        simplified = cv2.approxPolyDP(cv_contour, epsilon, True)

        # Convert back to original format
        return simplified.reshape(-1, 2)

    @staticmethod
    def smooth_contour(contour: np.ndarray, factor: float = 0.25) -> np.ndarray:
        """
        Smooth a contour to reduce jaggedness.

        Parameters
        ----------
        contour : np.ndarray
            Input contour points
        factor : float
            Smoothing factor (0 = no smoothing, 1 = maximum smoothing)

        Returns
        -------
        np.ndarray
            Smoothed contour points
        """
        if len(contour) <= 3:
            return contour

        # Limit factor to range [0, 1]
        factor = max(0, min(1, factor))

        # Create a closed contour by adding the first point at the end
        closed_contour = np.vstack([contour, contour[0:1]])

        # Apply simple moving average
        kernel_size = max(3, int(len(closed_contour) * factor / 10)) | 1  # Ensure odd
        kernel = np.ones(kernel_size) / kernel_size

        # Apply convolution for smoothing
        x_convolved = np.convolve(closed_contour[:, 0], kernel, mode="same")
        y_convolved = np.convolve(closed_contour[:, 1], kernel, mode="same")

        # Combine smoothed coordinates
        smoothed = np.column_stack([x_convolved, y_convolved])

        # Remove the last point (which is a duplicate of the first)
        return smoothed[:-1]


class BooleanOperator:
    """
    User-friendly interface for applying boolean operations to contours.

    This class provides a simplified interface for applying various boolean operations
    to contours in radiotherapy treatment planning. It acts as a wrapper around the
    BooleanOperations class, providing more intuitive methods for common boolean operations.

    Typical use cases:
    - Creating composite structures from multiple structures
    - Creating subtraction structures (e.g., PTV minus OAR)
    - Creating ring structures for optimization
    """

    def __init__(self):
        """Initialize the BooleanOperator."""
        pass

    def union(
        self, contours_a: List[np.ndarray], contours_b: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Perform union operation (A OR B) on two sets of contours.

        Parameters
        ----------
        contours_a : List[np.ndarray]
            First set of contours
        contours_b : List[np.ndarray]
            Second set of contours

        Returns
        -------
        List[np.ndarray]
            Resulting contours after union operation
        """
        result = []
        # Process each slice (assuming contours_a and contours_b have the same number of slices)
        for i in range(max(len(contours_a), len(contours_b))):
            a_contour = contours_a[i] if i < len(contours_a) else np.array([])
            b_contour = contours_b[i] if i < len(contours_b) else np.array([])

            # Skip if both contours are empty
            if a_contour.size == 0 and b_contour.size == 0:
                result.append(np.array([]))
                continue

            # If one contour is empty, use the other
            if a_contour.size == 0:
                result.append(b_contour)
                continue
            if b_contour.size == 0:
                result.append(a_contour)
                continue

            # Perform union operation
            union_contour = BooleanOperations.union_2d_contours(a_contour, b_contour)
            result.append(union_contour)

        return result

    def intersection(
        self, contours_a: List[np.ndarray], contours_b: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Perform intersection operation (A AND B) on two sets of contours.

        Parameters
        ----------
        contours_a : List[np.ndarray]
            First set of contours
        contours_b : List[np.ndarray]
            Second set of contours

        Returns
        -------
        List[np.ndarray]
            Resulting contours after intersection operation
        """
        result = []
        # Process each slice
        for i in range(min(len(contours_a), len(contours_b))):
            a_contour = contours_a[i]
            b_contour = contours_b[i]

            # Skip if either contour is empty
            if a_contour.size == 0 or b_contour.size == 0:
                result.append(np.array([]))
                continue

            # Perform intersection operation
            intersection_contour = BooleanOperations.intersection_2d_contours(
                a_contour, b_contour
            )
            result.append(intersection_contour)

        # Pad with empty contours if needed
        while len(result) < max(len(contours_a), len(contours_b)):
            result.append(np.array([]))

        return result

    def subtraction(
        self, contours_a: List[np.ndarray], contours_b: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Perform subtraction operation (A - B) on two sets of contours.

        Parameters
        ----------
        contours_a : List[np.ndarray]
            Base contours
        contours_b : List[np.ndarray]
            Contours to subtract

        Returns
        -------
        List[np.ndarray]
            Resulting contours after subtraction operation
        """
        result = []
        # Process each slice
        for i in range(len(contours_a)):
            a_contour = contours_a[i]

            # If base contour is empty, result is empty
            if a_contour.size == 0:
                result.append(np.array([]))
                continue

            # If subtraction contour is not available for this slice, keep original
            if i >= len(contours_b) or contours_b[i].size == 0:
                result.append(a_contour)
                continue

            # Perform subtraction operation
            subtraction_contour = BooleanOperations.subtract_2d_contours(
                a_contour, contours_b[i]
            )
            result.append(subtraction_contour)

        return result

    def exclusive_or(
        self, contours_a: List[np.ndarray], contours_b: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Perform exclusive OR operation (A XOR B) on two sets of contours.

        Parameters
        ----------
        contours_a : List[np.ndarray]
            First set of contours
        contours_b : List[np.ndarray]
            Second set of contours

        Returns
        -------
        List[np.ndarray]
            Resulting contours after XOR operation
        """
        result = []
        # Process each slice
        for i in range(max(len(contours_a), len(contours_b))):
            a_contour = contours_a[i] if i < len(contours_a) else np.array([])
            b_contour = contours_b[i] if i < len(contours_b) else np.array([])

            # If both contours are empty, result is empty
            if a_contour.size == 0 and b_contour.size == 0:
                result.append(np.array([]))
                continue

            # If one contour is empty, use the other
            if a_contour.size == 0:
                result.append(b_contour)
                continue
            if b_contour.size == 0:
                result.append(a_contour)
                continue

            # Perform XOR operation
            xor_contour = BooleanOperations.exclusive_or_2d_contours(
                a_contour, b_contour
            )
            result.append(xor_contour)

        return result

    def perform_operation(
        self,
        contours_a: List[np.ndarray],
        contours_b: List[np.ndarray],
        operation: BooleanOperation,
    ) -> List[np.ndarray]:
        """
        Perform specified boolean operation on two sets of contours.

        Parameters
        ----------
        contours_a : List[np.ndarray]
            First set of contours
        contours_b : List[np.ndarray]
            Second set of contours
        operation : BooleanOperation
            Boolean operation to perform

        Returns
        -------
        List[np.ndarray]
            Resulting contours after the operation
        """
        if operation == BooleanOperation.UNION:
            return self.union(contours_a, contours_b)
        elif operation == BooleanOperation.INTERSECTION:
            return self.intersection(contours_a, contours_b)
        elif operation == BooleanOperation.SUBTRACTION:
            return self.subtraction(contours_a, contours_b)
        elif operation == BooleanOperation.EXCLUSIVE_OR:
            return self.exclusive_or(contours_a, contours_b)
        else:
            logger.error(f"Unknown boolean operation: {operation}")
            return []

    def multi_union(self, contours_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Perform union operation on multiple sets of contours.

        Parameters
        ----------
        contours_list : List[List[np.ndarray]]
            List of contour sets to union

        Returns
        -------
        List[np.ndarray]
            Resulting contours after union operation
        """
        if not contours_list:
            return []

        if len(contours_list) == 1:
            return contours_list[0]

        result = contours_list[0]
        for contours in contours_list[1:]:
            result = self.union(result, contours)

        return result
