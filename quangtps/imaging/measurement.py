#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Measurement tools for radiotherapy imaging.

This module provides tools for measuring distances, areas, volumes,
and other quantities in radiotherapy imaging data.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import SimpleITK as sitk
from dataclasses import dataclass

from quangtps.imaging.image import Image
from quangtps.imaging.contour import Contour

logger = logging.getLogger(__name__)

@dataclass
class Point3D:
    """3D point with x, y, z coordinates in physical space (mm)."""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point3D') -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x)**2 + 
                      (self.y - other.y)**2 + 
                      (self.z - other.z)**2)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point3D':
        """Create a Point3D from a numpy array."""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


@dataclass
class Line3D:
    """3D line defined by two points."""
    start: Point3D
    end: Point3D
    
    def length(self) -> float:
        """Calculate the length of the line."""
        return self.start.distance_to(self.end)
    
    def direction(self) -> np.ndarray:
        """Calculate the direction vector of the line."""
        vec = np.array([self.end.x - self.start.x,
                       self.end.y - self.start.y,
                       self.end.z - self.start.z])
        return vec / np.linalg.norm(vec)


class Ruler:
    """
    Ruler tool for measuring distances in images.
    
    This class provides methods for measuring distances between points
    in 2D and 3D medical images, including physical and voxel distances.
    """
    
    def __init__(self, image: Optional[Image] = None):
        """
        Initialize the ruler tool.
        
        Parameters
        ----------
        image : Optional[Image]
            Reference image with spacing information
        """
        self.image = image
        self.points: List[Point3D] = []
    
    def set_image(self, image: Image) -> None:
        """
        Set the reference image.
        
        Parameters
        ----------
        image : Image
            Reference image with spacing information
        """
        self.image = image
    
    def add_point(self, point: Point3D) -> None:
        """
        Add a measurement point.
        
        Parameters
        ----------
        point : Point3D
            3D point to add
        """
        self.points.append(point)
    
    def clear_points(self) -> None:
        """Clear all measurement points."""
        self.points.clear()
    
    def measure_distance(self, p1: Point3D, p2: Point3D) -> float:
        """
        Measure the physical distance between two points.
        
        Parameters
        ----------
        p1 : Point3D
            First point
        p2 : Point3D
            Second point
            
        Returns
        -------
        float
            Distance in millimeters
        """
        return p1.distance_to(p2)
    
    def measure_path_length(self) -> float:
        """
        Measure the total length of the path defined by all points.
        
        Returns
        -------
        float
            Total path length in millimeters
        """
        if len(self.points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(self.points)):
            total_length += self.points[i-1].distance_to(self.points[i])
        
        return total_length
    
    def physical_to_voxel(self, point: Point3D) -> Tuple[int, int, int]:
        """
        Convert a physical point to voxel coordinates.
        
        Parameters
        ----------
        point : Point3D
            Physical point (in mm)
            
        Returns
        -------
        Tuple[int, int, int]
            Voxel coordinates (i, j, k)
            
        Raises
        ------
        ValueError
            If no reference image has been set
        """
        if self.image is None:
            raise ValueError("No reference image has been set")
        
        # Convert physical point to voxel coordinates
        point_arr = np.array([point.x, point.y, point.z])
        origin = np.array(self.image.origin)
        spacing = np.array(self.image.spacing)
        
        # Calculate voxel coordinates
        voxel_coords = np.round((point_arr - origin) / spacing).astype(int)
        
        return tuple(voxel_coords)
    
    def voxel_to_physical(self, i: int, j: int, k: int) -> Point3D:
        """
        Convert voxel coordinates to physical point.
        
        Parameters
        ----------
        i, j, k : int
            Voxel coordinates
            
        Returns
        -------
        Point3D
            Physical point (in mm)
            
        Raises
        ------
        ValueError
            If no reference image has been set
        """
        if self.image is None:
            raise ValueError("No reference image has been set")
        
        # Convert voxel coordinates to physical point
        voxel_coords = np.array([i, j, k])
        origin = np.array(self.image.origin)
        spacing = np.array(self.image.spacing)
        
        # Calculate physical coordinates
        physical_coords = origin + voxel_coords * spacing
        
        return Point3D(x=physical_coords[0], y=physical_coords[1], z=physical_coords[2])


class AreaMeasurement:
    """
    Tools for measuring areas in radiotherapy imaging.
    
    This class provides methods for measuring areas of contours
    and regions in 2D slices.
    """
    
    @staticmethod
    def measure_contour_area(contour: Contour) -> float:
        """
        Measure the area of a contour.
        
        Parameters
        ----------
        contour : Contour
            The contour to measure
            
        Returns
        -------
        float
            Area in square millimeters
        """
        return contour.get_area()
    
    @staticmethod
    def measure_roi_area(mask: np.ndarray, spacing: Tuple[float, float]) -> float:
        """
        Measure the area of a region of interest in a 2D mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask (2D)
        spacing : Tuple[float, float]
            Pixel spacing (dx, dy) in mm
            
        Returns
        -------
        float
            Area in square millimeters
        """
        if mask.ndim != 2:
            raise ValueError("Mask must be 2D")
        
        # Count non-zero pixels and multiply by pixel area
        pixel_count = np.count_nonzero(mask)
        pixel_area = spacing[0] * spacing[1]
        
        return pixel_count * pixel_area


class VolumeMeasurement:
    """
    Tools for measuring volumes in radiotherapy imaging.
    
    This class provides methods for measuring volumes of structures
    and regions in 3D images.
    """
    
    @staticmethod
    def measure_structure_volume(contours: List[Contour], slice_spacing: float) -> float:
        """
        Measure the volume of a structure defined by contours on multiple slices.
        
        Parameters
        ----------
        contours : List[Contour]
            List of contours defining the structure
        slice_spacing : float
            Spacing between slices in mm
            
        Returns
        -------
        float
            Volume in cubic millimeters
        """
        if not contours:
            return 0.0
        
        # Calculate volume using contour areas and slice thickness
        volume = 0.0
        for contour in contours:
            area = contour.get_area()
            volume += area * slice_spacing
        
        return volume
    
    @staticmethod
    def measure_roi_volume(mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
        """
        Measure the volume of a region of interest in a 3D mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask (3D)
        spacing : Tuple[float, float, float]
            Voxel spacing (dx, dy, dz) in mm
            
        Returns
        -------
        float
            Volume in cubic millimeters
        """
        if mask.ndim != 3:
            raise ValueError("Mask must be 3D")
        
        # Count non-zero voxels and multiply by voxel volume
        voxel_count = np.count_nonzero(mask)
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        
        return voxel_count * voxel_volume


class DensityMeasurement:
    """
    Tools for measuring densities and Hounsfield Units in CT images.
    
    This class provides methods for measuring CT densities, such as
    average HU in a region, and converting HU to physical density.
    """
    
    @staticmethod
    def measure_average_hu(ct_image: np.ndarray, mask: np.ndarray) -> float:
        """
        Measure the average Hounsfield Units in a region of interest.
        
        Parameters
        ----------
        ct_image : np.ndarray
            CT image data in HU
        mask : np.ndarray
            Binary mask defining the region of interest
            
        Returns
        -------
        float
            Average HU value
        """
        if ct_image.shape != mask.shape:
            raise ValueError("CT image and mask must have the same shape")
        
        # Extract values within the mask
        values = ct_image[mask > 0]
        
        if len(values) == 0:
            return 0.0
        
        return float(np.mean(values))
    
    @staticmethod
    def measure_hu_statistics(ct_image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistical measures of Hounsfield Units in a region.
        
        Parameters
        ----------
        ct_image : np.ndarray
            CT image data in HU
        mask : np.ndarray
            Binary mask defining the region of interest
            
        Returns
        -------
        Dict[str, float]
            Dictionary with statistical measures (min, max, mean, median, std)
        """
        if ct_image.shape != mask.shape:
            raise ValueError("CT image and mask must have the same shape")
        
        # Extract values within the mask
        values = ct_image[mask > 0]
        
        if len(values) == 0:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0
            }
        
        return {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values))
        }
    
    @staticmethod
    def hu_to_density(hu: float) -> float:
        """
        Convert Hounsfield Units to physical density (g/cm³).
        
        Parameters
        ----------
        hu : float
            Hounsfield Units value
            
        Returns
        -------
        float
            Physical density in g/cm³
        """
        # Standard conversion formula
        # Different for HU < 0 (air, lung) and HU >= 0 (water, soft tissue, bone)
        if hu < 0:
            return 1.0 + hu / 1000.0
        else:
            return 1.0 + hu / 1950.0


class AngleMeasurement:
    """
    Tools for measuring angles in radiotherapy imaging.
    
    This class provides methods for measuring angles between lines,
    planes, and structures in medical images.
    """
    
    @staticmethod
    def measure_angle_between_lines(line1: Line3D, line2: Line3D) -> float:
        """
        Measure the angle between two 3D lines.
        
        Parameters
        ----------
        line1 : Line3D
            First line
        line2 : Line3D
            Second line
            
        Returns
        -------
        float
            Angle in degrees
        """
        # Get direction vectors
        dir1 = line1.direction()
        dir2 = line2.direction()
        
        # Calculate cosine of the angle
        cos_angle = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
        
        # Calculate angle in radians and convert to degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    @staticmethod
    def measure_angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Measure the angle between two vectors.
        
        Parameters
        ----------
        vec1 : np.ndarray
            First vector
        vec2 : np.ndarray
            Second vector
            
        Returns
        -------
        float
            Angle in degrees
        """
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1_norm = vec1 / norm1
        vec2_norm = vec2 / norm2
        
        # Calculate cosine of the angle
        cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        
        # Calculate angle in radians and convert to degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
