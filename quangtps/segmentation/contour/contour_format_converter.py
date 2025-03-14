#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contour Format Converter Module for QuangTPS.

This module provides functionality to convert contours between different formats
commonly used in radiotherapy treatment planning, such as DICOM RT Structure Set,
DICOM Segmentation, Polygon, and Point-based representations.
"""

import os
import numpy as np
import cv2
import logging
import pydicom
from typing import Dict, List, Optional, Tuple, Union, Any
import SimpleITK as sitk

from quangtps.core.exceptions import ValidationError
from quangtps.dicom.dicom_writer import DicomWriter

logger = logging.getLogger(__name__)

class ContourFormatConverter:
    """
    Class for converting contours between different formats.
    
    This class supports conversion between various contour formats:
    - Point-based representation (list of points)
    - Polygons (numpy arrays)
    - Binary masks (numpy arrays)
    - DICOM RTSTRUCT contours
    - DICOM SEG objects
    """
    
    def __init__(self):
        """Initialize contour format converter."""
        pass
    
    def points_to_polygon(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Convert a list of points to a polygon.
        
        Parameters
        ----------
        points : List[Tuple[float, float]]
            List of (x, y) points
            
        Returns
        -------
        np.ndarray
            Polygon as a numpy array of shape (n, 1, 2)
        """
        try:
            if not points:
                raise ValidationError("Empty points list")
            
            # Ensure the contour is closed
            if points[0] != points[-1]:
                points = points + [points[0]]
            
            # Convert to numpy array in the format expected by OpenCV
            polygon = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
            
            return polygon
            
        except Exception as e:
            logger.error(f"Error converting points to polygon: {str(e)}")
            raise ValidationError(f"Error converting points to polygon: {str(e)}")
    
    def polygon_to_points(self, polygon: np.ndarray) -> List[Tuple[float, float]]:
        """
        Convert a polygon to a list of points.
        
        Parameters
        ----------
        polygon : np.ndarray
            Polygon as a numpy array
            
        Returns
        -------
        List[Tuple[float, float]]
            List of (x, y) points
        """
        try:
            # Polygon could be in different formats from OpenCV
            if len(polygon.shape) == 3:
                # Format (n, 1, 2)
                points = [(float(x), float(y)) for [x, y] in polygon[:, 0, :]]
            else:
                # Format (n, 2)
                points = [(float(x), float(y)) for [x, y] in polygon]
            
            return points
            
        except Exception as e:
            logger.error(f"Error converting polygon to points: {str(e)}")
            raise ValidationError(f"Error converting polygon to points: {str(e)}")
    
    def points_to_mask(self, points: List[Tuple[float, float]], shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert a list of points to a binary mask.
        
        Parameters
        ----------
        points : List[Tuple[float, float]]
            List of (x, y) points
        shape : Tuple[int, int]
            Shape of the output mask (height, width)
            
        Returns
        -------
        np.ndarray
            Binary mask
        """
        try:
            if not points:
                return np.zeros(shape, dtype=np.uint8)
            
            # Convert to a polygon
            polygon = self.points_to_polygon(points)
            
            # Create mask and fill the polygon
            mask = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)[:, 0, :]], 1)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error converting points to mask: {str(e)}")
            raise ValidationError(f"Error converting points to mask: {str(e)}")
    
    def mask_to_contours(self, mask: np.ndarray) -> List[np.ndarray]:
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
    
    def mask_to_points(self, mask: np.ndarray) -> List[List[Tuple[float, float]]]:
        """
        Convert a binary mask to lists of points (multiple contours).
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask
            
        Returns
        -------
        List[List[Tuple[float, float]]]
            List of contours, each as a list of (x, y) points
        """
        try:
            # Extract contours
            contours = self.mask_to_contours(mask)
            
            # Convert each contour to a list of points
            points_list = [self.polygon_to_points(contour) for contour in contours]
            
            return points_list
            
        except Exception as e:
            logger.error(f"Error converting mask to points: {str(e)}")
            raise ValidationError(f"Error converting mask to points: {str(e)}")
    
    def dicom_contour_to_points(self, dicom_contour: List[float]) -> List[Tuple[float, float, float]]:
        """
        Convert a DICOM RT contour to a list of 3D points.
        
        Parameters
        ----------
        dicom_contour : List[float]
            DICOM RT contour data as a flat list of coordinates [x1,y1,z1,x2,y2,z2,...]
            
        Returns
        -------
        List[Tuple[float, float, float]]
            List of 3D points (x, y, z)
        """
        try:
            if len(dicom_contour) % 3 != 0:
                raise ValidationError("DICOM contour data length is not a multiple of 3")
            
            # Group the flat list into triplets (x,y,z)
            points_3d = [(dicom_contour[i], dicom_contour[i+1], dicom_contour[i+2]) 
                        for i in range(0, len(dicom_contour), 3)]
            
            return points_3d
            
        except Exception as e:
            logger.error(f"Error converting DICOM contour to points: {str(e)}")
            raise ValidationError(f"Error converting DICOM contour to points: {str(e)}")
    
    def points_to_dicom_contour(self, points_3d: List[Tuple[float, float, float]]) -> List[float]:
        """
        Convert a list of 3D points to a DICOM RT contour data format.
        
        Parameters
        ----------
        points_3d : List[Tuple[float, float, float]]
            List of 3D points (x, y, z)
            
        Returns
        -------
        List[float]
            DICOM RT contour data as a flat list of coordinates [x1,y1,z1,x2,y2,z2,...]
        """
        try:
            # Flatten the list of points into a single list
            dicom_contour = []
            for x, y, z in points_3d:
                dicom_contour.extend([float(x), float(y), float(z)])
            
            return dicom_contour
            
        except Exception as e:
            logger.error(f"Error converting points to DICOM contour: {str(e)}")
            raise ValidationError(f"Error converting points to DICOM contour: {str(e)}")
    
    def simplify_contour(self, points: List[Tuple[float, float]], epsilon: float = 1.0) -> List[Tuple[float, float]]:
        """
        Simplify a contour using the Douglas-Peucker algorithm.
        
        Parameters
        ----------
        points : List[Tuple[float, float]]
            List of (x, y) points
        epsilon : float, optional
            Approximation accuracy parameter
            
        Returns
        -------
        List[Tuple[float, float]]
            Simplified contour
        """
        try:
            if len(points) < 3:
                return points
            
            # Convert to polygon
            polygon = self.points_to_polygon(points)
            
            # Apply Douglas-Peucker algorithm
            simplified = cv2.approxPolyDP(polygon, epsilon, closed=True)
            
            # Convert back to list of points
            simplified_points = self.polygon_to_points(simplified)
            
            return simplified_points
            
        except Exception as e:
            logger.error(f"Error simplifying contour: {str(e)}")
            raise ValidationError(f"Error simplifying contour: {str(e)}")
    
    def smooth_contour(self, points: List[Tuple[float, float]], method: str = 'spline', 
                        params: Dict[str, Any] = None) -> List[Tuple[float, float]]:
        """
        Smooth a contour using various methods.
        
        Parameters
        ----------
        points : List[Tuple[float, float]]
            List of (x, y) points
        method : str, optional
            Smoothing method ('spline', 'gaussian', 'moving_average')
        params : Dict[str, Any], optional
            Method-specific parameters
            
        Returns
        -------
        List[Tuple[float, float]]
            Smoothed contour
        """
        try:
            if not points or len(points) < 3:
                return points
            
            # Default parameters
            if params is None:
                params = {}
            
            # Close the contour if not already closed
            if points[0] != points[-1]:
                closed_points = points + [points[0]]
            else:
                closed_points = points
            
            # Extract x and y coordinates
            x = np.array([p[0] for p in closed_points])
            y = np.array([p[1] for p in closed_points])
            
            if method == 'spline':
                from scipy.interpolate import splprep, splev
                
                # Get parameters or use defaults
                s = params.get('smoothing', 0.0)
                k = params.get('k', 3)  # Spline degree
                n = params.get('n', len(closed_points) * 2)  # Number of points in output
                
                # Fit a spline to the points
                tck, u = splprep([x, y], s=s, k=k, per=True)
                
                # Evaluate the spline at n points
                u_new = np.linspace(0, 1, n)
                x_new, y_new = splev(u_new, tck)
                
                smoothed_points = [(float(x_new[i]), float(y_new[i])) for i in range(n)]
                
            elif method == 'gaussian':
                from scipy.ndimage import gaussian_filter1d
                
                # Get parameters or use defaults
                sigma = params.get('sigma', 1.0)
                
                # Apply Gaussian filter separately to x and y coordinates
                x_smooth = gaussian_filter1d(x, sigma, mode='wrap')
                y_smooth = gaussian_filter1d(y, sigma, mode='wrap')
                
                smoothed_points = [(float(x_smooth[i]), float(y_smooth[i])) for i in range(len(closed_points))]
                
            elif method == 'moving_average':
                window_size = params.get('window_size', 3)
                
                if window_size < 1 or window_size >= len(closed_points):
                    return points
                
                # Ensuring the contour is closed for the moving average
                extended_x = np.concatenate((x[-window_size+1:], x, x[:window_size]))
                extended_y = np.concatenate((y[-window_size+1:], y, y[:window_size]))
                
                x_smooth = np.zeros_like(x)
                y_smooth = np.zeros_like(y)
                
                for i in range(len(closed_points)):
                    start_idx = i
                    end_idx = i + window_size * 2 - 1
                    x_smooth[i] = np.mean(extended_x[start_idx:end_idx+1])
                    y_smooth[i] = np.mean(extended_y[start_idx:end_idx+1])
                
                smoothed_points = [(float(x_smooth[i]), float(y_smooth[i])) for i in range(len(closed_points))]
                
            else:
                raise ValidationError(f"Unknown smoothing method: {method}")
            
            # Ensure the contour is closed
            if smoothed_points[0] != smoothed_points[-1]:
                smoothed_points.append(smoothed_points[0])
            
            return smoothed_points
            
        except Exception as e:
            logger.error(f"Error smoothing contour: {str(e)}")
            raise ValidationError(f"Error smoothing contour: {str(e)}")
    
    def interpolate_z_contours(self, contours_3d: List[List[Tuple[float, float, float]]], 
                             z_values: List[float], target_z: float) -> List[Tuple[float, float, float]]:
        """
        Interpolate a contour at a specific z position from contours at different z positions.
        
        Parameters
        ----------
        contours_3d : List[List[Tuple[float, float, float]]]
            List of 3D contours at different z positions
        z_values : List[float]
            Z values for each contour
        target_z : float
            Z position to interpolate at
            
        Returns
        -------
        List[Tuple[float, float, float]]
            Interpolated 3D contour at target_z
        """
        try:
            if len(contours_3d) != len(z_values):
                raise ValidationError("Number of contours does not match number of z values")
            
            if len(contours_3d) < 2:
                raise ValidationError("At least two contours needed for interpolation")
            
            # Find the nearest slices above and below the target
            sorted_indices = np.argsort(z_values)
            sorted_z = [z_values[i] for i in sorted_indices]
            
            if target_z <= sorted_z[0]:
                # Target is below or at the lowest slice, return the lowest contour
                return [(x, y, target_z) for x, y, _ in contours_3d[sorted_indices[0]]]
                
            if target_z >= sorted_z[-1]:
                # Target is above or at the highest slice, return the highest contour
                return [(x, y, target_z) for x, y, _ in contours_3d[sorted_indices[-1]]]
            
            # Find indices of the slices that bracket the target
            for i in range(len(sorted_z) - 1):
                if sorted_z[i] <= target_z <= sorted_z[i+1]:
                    lower_idx = sorted_indices[i]
                    upper_idx = sorted_indices[i+1]
                    break
            else:
                raise ValidationError("Failed to find bracketing slices")
            
            # Calculate the weights for interpolation
            z_lower = z_values[lower_idx]
            z_upper = z_values[upper_idx]
            weight_upper = (target_z - z_lower) / (z_upper - z_lower)
            weight_lower = 1.0 - weight_upper
            
            # Get the contours to interpolate
            contour_lower = contours_3d[lower_idx]
            contour_upper = contours_3d[upper_idx]
            
            # Convert both contours to masks
            shape = (1000, 1000)  # Arbitrary shape, should be sufficient for most contours
            mask_lower = self.points_to_mask([(x, y) for x, y, _ in contour_lower], shape)
            mask_upper = self.points_to_mask([(x, y) for x, y, _ in contour_upper], shape)
            
            # Combine the masks with weights
            combined_mask = (mask_lower * weight_lower + mask_upper * weight_upper > 0.5).astype(np.uint8)
            
            # Extract the contour from the combined mask
            contours = self.mask_to_contours(combined_mask)
            
            if not contours:
                # If no contour found, use the lower contour
                return [(x, y, target_z) for x, y, _ in contour_lower]
            
            # Sort contours by area and take the largest one
            areas = [cv2.contourArea(c) for c in contours]
            largest_contour = contours[np.argmax(areas)]
            
            # Convert to 3D points with the target z
            interpolated_contour = [(x, y, target_z) for x, y in self.polygon_to_points(largest_contour)]
            
            return interpolated_contour
            
        except Exception as e:
            logger.error(f"Error interpolating contours: {str(e)}")
            raise ValidationError(f"Error interpolating contours: {str(e)}")
    
    def merge_contours(self, contours: List[List[Tuple[float, float]]], shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        """
        Merge multiple contours into a single contour.
        
        Parameters
        ----------
        contours : List[List[Tuple[float, float]]]
            List of contours, each as a list of (x, y) points
        shape : Tuple[int, int]
            Shape of the mask (height, width)
            
        Returns
        -------
        List[Tuple[float, float]]
            Merged contour
        """
        try:
            if not contours:
                return []
            
            if len(contours) == 1:
                return contours[0]
            
            # Convert all contours to masks and combine
            combined_mask = np.zeros(shape, dtype=np.uint8)
            
            for contour in contours:
                mask = self.points_to_mask(contour, shape)
                combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
            
            # Extract the outer contour from the combined mask
            extracted_contours = self.mask_to_contours(combined_mask)
            
            if not extracted_contours:
                return []
            
            # Sort contours by area and return the largest one
            areas = [cv2.contourArea(c) for c in extracted_contours]
            largest_contour = extracted_contours[np.argmax(areas)]
            
            return self.polygon_to_points(largest_contour)
            
        except Exception as e:
            logger.error(f"Error merging contours: {str(e)}")
            raise ValidationError(f"Error merging contours: {str(e)}")
