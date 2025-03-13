#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for margin operations on contours.

This module provides functionality for creating various types of margins around contours,
such as expansions, contractions, and rings. These operations are essential for
creating PTV, PRV, and other derivative structures in radiotherapy treatment planning.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict
from enum import Enum

from skimage import measure, morphology, draw

logger = logging.getLogger(__name__)


class MarginType(str, Enum):
    """Enum for different margin types."""
    UNIFORM = "UNIFORM"  # Uniform expansion/contraction in all directions
    ANISOTROPIC = "ANISOTROPIC"  # Different margins in different dimensions
    RING = "RING"  # Create a ring around a structure (shell)
    SURFACE = "SURFACE"  # Create a thin surface layer around a structure
    LIMIT_TO_BODY = "LIMIT_TO_BODY"  # Expansion limited by body contour
    AVOID_STRUCTURE = "AVOID_STRUCTURE"  # Expansion avoiding specified structures


class MarginGenerator:
    """
    Class for generating margins around contours.
    
    This class provides methods to create various types of margins for radiotherapy 
    treatment planning, including uniform expansions for PTV creation, rings for
    dose evaluation structures, and margin operations that respect anatomical boundaries.
    """
    
    @staticmethod
    def apply_uniform_margin_2d(contour: np.ndarray, margin_mm: float, 
                              pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """
        Apply a uniform margin to a 2D contour.
        
        Parameters
        ----------
        contour : np.ndarray
            Input contour points as nx2 array
        margin_mm : float
            Margin in millimeters (positive for expansion, negative for contraction)
        pixel_spacing : Tuple[float, float]
            Pixel spacing in mm/pixel for x and y directions
            
        Returns
        -------
        np.ndarray
            Contour with margin applied
        """
        if contour.size == 0:
            return np.array([])
        
        # Convert margin from mm to pixels
        margin_x_px = margin_mm / pixel_spacing[0]
        margin_y_px = margin_mm / pixel_spacing[1]
        
        # Convert to mask
        all_points = contour.copy()
        min_x, min_y = np.floor(np.min(all_points, axis=0)).astype(int) - int(abs(margin_x_px) * 2)
        max_x, max_y = np.ceil(np.max(all_points, axis=0)).astype(int) + int(abs(margin_x_px) * 2)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Normalize coordinates to mask space
        normalized_contour = contour.copy()
        normalized_contour[:, 0] -= min_x
        normalized_contour[:, 1] -= min_y
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Convert contour points to integer coordinates
        rr, cc = draw.polygon(normalized_contour[:, 1], normalized_contour[:, 0])
        
        # Filter out points outside the image
        valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid_indices], cc[valid_indices]
        
        # Fill the contour
        if len(rr) > 0 and len(cc) > 0:
            mask[rr, cc] = 1
        
        # Apply morphological operation based on margin sign
        if margin_mm > 0:
            # Expansion: Use dilation
            # Convert margin from mm to pixels (take minimum of x,y to be conservative)
            radius = int(min(margin_x_px, margin_y_px))
            if radius <= 0:
                radius = 1
                
            # Create a disk structuring element
            selem = morphology.disk(radius)
            dilated_mask = morphology.binary_dilation(mask, selem)
            result_mask = dilated_mask.astype(np.uint8)
        else:
            # Contraction: Use erosion
            # Convert margin from mm to pixels (take minimum to be conservative)
            radius = int(min(abs(margin_x_px), abs(margin_y_px)))
            if radius <= 0:
                radius = 1
                
            # Create a disk structuring element
            selem = morphology.disk(radius)
            eroded_mask = morphology.binary_erosion(mask, selem)
            result_mask = eroded_mask.astype(np.uint8)
        
        # Convert mask back to contour
        contours = measure.find_contours(result_mask, 0.5)
        
        # Return the longest contour
        if not contours:
            return np.array([])
        
        # Sort contours by length and take the longest one
        longest_contour = max(contours, key=len)
        
        # Swap x, y coordinates and transform back to original coordinate space
        result_contour = np.fliplr(longest_contour)
        result_contour[:, 0] += min_x
        result_contour[:, 1] += min_y
        
        return result_contour
    
    @staticmethod
    def apply_anisotropic_margin_2d(contour: np.ndarray, 
                                   margins_mm: Dict[str, float],
                                   pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """
        Apply different margins in different directions.
        
        Parameters
        ----------
        contour : np.ndarray
            Input contour points as nx2 array
        margins_mm : Dict[str, float]
            Margins in mm for each direction: 'LEFT', 'RIGHT', 'ANTERIOR', 'POSTERIOR'
        pixel_spacing : Tuple[float, float]
            Pixel spacing in mm/pixel for x and y directions
            
        Returns
        -------
        np.ndarray
            Contour with anisotropic margins applied
        """
        if contour.size == 0:
            return np.array([])
        
        # Validate and normalize input margins
        required_keys = ['LEFT', 'RIGHT', 'ANTERIOR', 'POSTERIOR']
        margins = {k: margins_mm.get(k, 0.0) for k in required_keys}
        
        # Convert to mask
        all_points = contour.copy()
        
        # Add extra space for margins
        max_margin_x = max(abs(margins['LEFT']), abs(margins['RIGHT']))
        max_margin_y = max(abs(margins['ANTERIOR']), abs(margins['POSTERIOR']))
        
        # Convert margins from mm to pixels
        margin_x_px = max_margin_x / pixel_spacing[0]
        margin_y_px = max_margin_y / pixel_spacing[1]
        
        min_x, min_y = np.floor(np.min(all_points, axis=0)).astype(int) - int(margin_x_px * 2)
        max_x, max_y = np.ceil(np.max(all_points, axis=0)).astype(int) + int(margin_y_px * 2)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Normalize coordinates to mask space
        normalized_contour = contour.copy()
        normalized_contour[:, 0] -= min_x
        normalized_contour[:, 1] -= min_y
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill the contour
        rr, cc = draw.polygon(normalized_contour[:, 1], normalized_contour[:, 0])
        valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid_indices], cc[valid_indices]
        
        if len(rr) > 0 and len(cc) > 0:
            mask[rr, cc] = 1
        
        # Apply directional margins
        result_mask = mask.copy()
        
        # Convert margins from mm to pixels
        margin_left_px = int(margins['LEFT'] / pixel_spacing[0])
        margin_right_px = int(margins['RIGHT'] / pixel_spacing[0])
        margin_anterior_px = int(margins['ANTERIOR'] / pixel_spacing[1])
        margin_posterior_px = int(margins['POSTERIOR'] / pixel_spacing[1])
        
        # For X direction (left-right)
        if margin_left_px != 0 or margin_right_px != 0:
            for y in range(height):
                # Find leftmost and rightmost filled pixels in this row
                filled_x = np.where(mask[y, :] > 0)[0]
                if len(filled_x) > 0:
                    left_x, right_x = filled_x[0], filled_x[-1]
                    
                    # Apply left margin
                    if margin_left_px > 0:  # Expansion
                        new_left_x = max(0, left_x - margin_left_px)
                        result_mask[y, new_left_x:left_x] = 1
                    elif margin_left_px < 0:  # Contraction
                        new_left_x = min(width - 1, left_x + abs(margin_left_px))
                        result_mask[y, left_x:new_left_x] = 0
                    
                    # Apply right margin
                    if margin_right_px > 0:  # Expansion
                        new_right_x = min(width - 1, right_x + margin_right_px)
                        result_mask[y, right_x:new_right_x + 1] = 1
                    elif margin_right_px < 0:  # Contraction
                        new_right_x = max(0, right_x - abs(margin_right_px))
                        result_mask[y, new_right_x + 1:right_x + 1] = 0
        
        # For Y direction (anterior-posterior)
        if margin_anterior_px != 0 or margin_posterior_px != 0:
            for x in range(width):
                # Find topmost and bottommost filled pixels in this column
                filled_y = np.where(mask[:, x] > 0)[0]
                if len(filled_y) > 0:
                    top_y, bottom_y = filled_y[0], filled_y[-1]
                    
                    # Apply anterior (top) margin
                    if margin_anterior_px > 0:  # Expansion
                        new_top_y = max(0, top_y - margin_anterior_px)
                        result_mask[new_top_y:top_y, x] = 1
                    elif margin_anterior_px < 0:  # Contraction
                        new_top_y = min(height - 1, top_y + abs(margin_anterior_px))
                        result_mask[top_y:new_top_y, x] = 0
                    
                    # Apply posterior (bottom) margin
                    if margin_posterior_px > 0:  # Expansion
                        new_bottom_y = min(height - 1, bottom_y + margin_posterior_px)
                        result_mask[bottom_y:new_bottom_y + 1, x] = 1
                    elif margin_posterior_px < 0:  # Contraction
                        new_bottom_y = max(0, bottom_y - abs(margin_posterior_px))
                        result_mask[new_bottom_y + 1:bottom_y + 1, x] = 0
        
        # Convert mask back to contour
        contours = measure.find_contours(result_mask, 0.5)
        
        # Return the longest contour
        if not contours:
            return np.array([])
        
        # Sort contours by length and take the longest one
        longest_contour = max(contours, key=len)
        
        # Swap x, y coordinates and transform back to original coordinate space
        result_contour = np.fliplr(longest_contour)
        result_contour[:, 0] += min_x
        result_contour[:, 1] += min_y
        
        return result_contour
    
    @staticmethod
    def create_ring_2d(contour: np.ndarray, inner_margin_mm: float, outer_margin_mm: float,
                     pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """
        Create a ring (shell) structure around a contour.
        
        Parameters
        ----------
        contour : np.ndarray
            Input contour points as nx2 array
        inner_margin_mm : float
            Inner margin in mm (can be negative for contraction)
        outer_margin_mm : float
            Outer margin in mm (must be greater than inner_margin_mm)
        pixel_spacing : Tuple[float, float]
            Pixel spacing in mm/pixel for x and y directions
            
        Returns
        -------
        List[np.ndarray]
            List of contours forming the ring (may be multiple disconnected contours)
        """
        if contour.size == 0:
            return []
        
        if outer_margin_mm <= inner_margin_mm:
            logger.error(f"Outer margin ({outer_margin_mm}mm) must be greater than inner margin ({inner_margin_mm}mm)")
            raise ValueError("Outer margin must be greater than inner margin")
        
        # Create inner and outer contours
        inner_contour = MarginGenerator.apply_uniform_margin_2d(
            contour, inner_margin_mm, pixel_spacing
        )
        
        outer_contour = MarginGenerator.apply_uniform_margin_2d(
            contour, outer_margin_mm, pixel_spacing
        )
        
        # Convert to masks
        all_points = np.vstack([inner_contour, outer_contour])
        min_x, min_y = np.floor(np.min(all_points, axis=0)).astype(int)
        max_x, max_y = np.ceil(np.max(all_points, axis=0)).astype(int)
        
        width = max_x - min_x + 10  # Add padding
        height = max_y - min_y + 10  # Add padding
        
        # Normalize coordinates
        inner_normalized = inner_contour.copy()
        inner_normalized[:, 0] -= min_x - 5
        inner_normalized[:, 1] -= min_y - 5
        
        outer_normalized = outer_contour.copy()
        outer_normalized[:, 0] -= min_x - 5
        outer_normalized[:, 1] -= min_y - 5
        
        # Create masks
        inner_mask = np.zeros((height, width), dtype=np.uint8)
        outer_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill inner contour
        rr, cc = draw.polygon(inner_normalized[:, 1], inner_normalized[:, 0])
        valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid_indices], cc[valid_indices]
        
        if len(rr) > 0 and len(cc) > 0:
            inner_mask[rr, cc] = 1
        
        # Fill outer contour
        rr, cc = draw.polygon(outer_normalized[:, 1], outer_normalized[:, 0])
        valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid_indices], cc[valid_indices]
        
        if len(rr) > 0 and len(cc) > 0:
            outer_mask[rr, cc] = 1
        
        # Create ring mask
        ring_mask = np.logical_and(outer_mask, np.logical_not(inner_mask)).astype(np.uint8)
        
        # Find all contours in the ring mask
        contours = measure.find_contours(ring_mask, 0.5)
        
        # Convert back to original coordinate space
        result_contours = []
        for c in contours:
            # Swap x, y and transform back
            result = np.fliplr(c)
            result[:, 0] += min_x - 5
            result[:, 1] += min_y - 5
            
            # Only add significant contours
            if len(result) > 5:  # Minimum number of points
                result_contours.append(result)
        
        # Sort by contour length (largest first)
        result_contours.sort(key=len, reverse=True)
        
        return result_contours
    
    @staticmethod
    def create_surface_layer_2d(contour: np.ndarray, thickness_mm: float,
                              pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """
        Create a thin surface layer around a contour.
        
        Parameters
        ----------
        contour : np.ndarray
            Input contour points as nx2 array
        thickness_mm : float
            Thickness of the surface layer in mm (must be positive)
        pixel_spacing : Tuple[float, float]
            Pixel spacing in mm/pixel for x and y directions
            
        Returns
        -------
        List[np.ndarray]
            List of contours forming the surface layer
        """
        if contour.size == 0:
            return []
        
        if thickness_mm <= 0:
            logger.error(f"Surface thickness ({thickness_mm}mm) must be positive")
            raise ValueError("Surface thickness must be positive")
        
        # This is essentially a ring with inner_margin=0 and outer_margin=thickness
        return MarginGenerator.create_ring_2d(contour, 0, thickness_mm, pixel_spacing)
    
    @staticmethod
    def limit_margin_to_body_2d(contour: np.ndarray, margin_mm: float, 
                               body_contour: np.ndarray,
                               pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """
        Apply margin to a contour but limit expansion to stay within body contour.
        
        Parameters
        ----------
        contour : np.ndarray
            Input contour points as nx2 array
        margin_mm : float
            Margin in mm (positive for expansion, negative for contraction)
        body_contour : np.ndarray
            Body contour that limits the expansion
        pixel_spacing : Tuple[float, float]
            Pixel spacing in mm/pixel for x and y directions
            
        Returns
        -------
        np.ndarray
            Contour with margin applied, limited by body contour
        """
        if contour.size == 0 or body_contour.size == 0:
            return np.array([])
        
        # If margin is negative (contraction), just apply it without limiting
        if margin_mm <= 0:
            return MarginGenerator.apply_uniform_margin_2d(contour, margin_mm, pixel_spacing)
        
        # Apply margin to get expanded contour
        expanded_contour = MarginGenerator.apply_uniform_margin_2d(contour, margin_mm, pixel_spacing)
        
        # Convert contours to masks
        all_points = np.vstack([expanded_contour, body_contour])
        min_x, min_y = np.floor(np.min(all_points, axis=0)).astype(int)
        max_x, max_y = np.ceil(np.max(all_points, axis=0)).astype(int)
        
        width = max_x - min_x + 10  # Add padding
        height = max_y - min_y + 10  # Add padding
        
        # Normalize coordinates
        expanded_normalized = expanded_contour.copy()
        expanded_normalized[:, 0] -= min_x - 5
        expanded_normalized[:, 1] -= min_y - 5
        
        body_normalized = body_contour.copy()
        body_normalized[:, 0] -= min_x - 5
        body_normalized[:, 1] -= min_y - 5
        
        # Create masks
        expanded_mask = np.zeros((height, width), dtype=np.uint8)
        body_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill expanded contour
        rr, cc = draw.polygon(expanded_normalized[:, 1], expanded_normalized[:, 0])
        valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid_indices], cc[valid_indices]
        
        if len(rr) > 0 and len(cc) > 0:
            expanded_mask[rr, cc] = 1
        
        # Fill body contour
        rr, cc = draw.polygon(body_normalized[:, 1], body_normalized[:, 0])
        valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid_indices], cc[valid_indices]
        
        if len(rr) > 0 and len(cc) > 0:
            body_mask[rr, cc] = 1
        
        # Limit expanded contour to body
        limited_mask = np.logical_and(expanded_mask, body_mask).astype(np.uint8)
        
        # Find contours in the limited mask
        contours = measure.find_contours(limited_mask, 0.5)
        
        # Return the longest contour
        if not contours:
            return np.array([])
        
        # Sort contours by length and take the longest one
        longest_contour = max(contours, key=len)
        
        # Swap x, y coordinates and transform back to original coordinate space
        result_contour = np.fliplr(longest_contour)
        result_contour[:, 0] += min_x - 5
        result_contour[:, 1] += min_y - 5
        
        return result_contour
    
    @staticmethod
    def avoid_structures_2d(contour: np.ndarray, margin_mm: float, 
                           avoid_contours: List[np.ndarray],
                           pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """
        Apply margin to a contour but avoid specified structures.
        
        Parameters
        ----------
        contour : np.ndarray
            Input contour points as nx2 array
        margin_mm : float
            Margin in mm (positive for expansion, negative for contraction)
        avoid_contours : List[np.ndarray]
            List of contours to avoid
        pixel_spacing : Tuple[float, float]
            Pixel spacing in mm/pixel for x and y directions
            
        Returns
        -------
        np.ndarray
            Contour with margin applied, avoiding specified structures
        """
        if contour.size == 0 or not avoid_contours:
            return MarginGenerator.apply_uniform_margin_2d(contour, margin_mm, pixel_spacing)
        
        # If margin is negative (contraction), just apply it without avoiding
        if margin_mm <= 0:
            return MarginGenerator.apply_uniform_margin_2d(contour, margin_mm, pixel_spacing)
        
        # Apply margin to get expanded contour
        expanded_contour = MarginGenerator.apply_uniform_margin_2d(contour, margin_mm, pixel_spacing)
        
        # Find bounding box for all contours
        all_contours = [expanded_contour] + avoid_contours
        all_points = np.vstack(all_contours)
        min_x, min_y = np.floor(np.min(all_points, axis=0)).astype(int)
        max_x, max_y = np.ceil(np.max(all_points, axis=0)).astype(int)
        
        width = max_x - min_x + 10  # Add padding
        height = max_y - min_y + 10  # Add padding
        
        # Normalize coordinates
        expanded_normalized = expanded_contour.copy()
        expanded_normalized[:, 0] -= min_x - 5
        expanded_normalized[:, 1] -= min_y - 5
        
        # Create masks
        expanded_mask = np.zeros((height, width), dtype=np.uint8)
        avoid_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill expanded contour
        rr, cc = draw.polygon(expanded_normalized[:, 1], expanded_normalized[:, 0])
        valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr, cc = rr[valid_indices], cc[valid_indices]
        
        if len(rr) > 0 and len(cc) > 0:
            expanded_mask[rr, cc] = 1
        
        # Fill avoid contours
        for avoid_contour in avoid_contours:
            avoid_normalized = avoid_contour.copy()
            avoid_normalized[:, 0] -= min_x - 5
            avoid_normalized[:, 1] -= min_y - 5
            
            rr, cc = draw.polygon(avoid_normalized[:, 1], avoid_normalized[:, 0])
            valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            rr, cc = rr[valid_indices], cc[valid_indices]
            
            if len(rr) > 0 and len(cc) > 0:
                avoid_mask[rr, cc] = 1
        
        # Subtract avoid regions from expanded contour
        result_mask = np.logical_and(expanded_mask, np.logical_not(avoid_mask)).astype(np.uint8)
        
        # Find contours in the result mask
        contours = measure.find_contours(result_mask, 0.5)
        
        # Return the longest contour
        if not contours:
            return np.array([])
        
        # Sort contours by length and take the longest one
        longest_contour = max(contours, key=len)
        
        # Swap x, y coordinates and transform back to original coordinate space
        result_contour = np.fliplr(longest_contour)
        result_contour[:, 0] += min_x - 5
        result_contour[:, 1] += min_y - 5
        
        return result_contour