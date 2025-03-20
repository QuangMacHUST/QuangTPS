#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for geometry classes used in structure definitions.

This module provides classes for representing 3D points, contours, and other
geometric entities used in radiotherapy structure definition.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Any, Dict


class Point:
    """
    Class representing a 3D point in space.
    
    A Point contains x, y, z coordinates, typically in the patient coordinate system
    measured in millimeters.
    """
    
    def __init__(self, x: float, y: float, z: float):
        """
        Initialize a 3D point.
        
        Parameters
        ----------
        x : float
            X-coordinate in mm
        y : float
            Y-coordinate in mm
        z : float
            Z-coordinate in mm
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    @classmethod
    def from_array(cls, arr: Union[np.ndarray, List, Tuple]):
        """
        Create a Point from an array-like object.
        
        Parameters
        ----------
        arr : array-like
            Array containing [x, y, z] coordinates
            
        Returns
        -------
        Point
            New Point object
        """
        if len(arr) < 3:
            raise ValueError("Array must contain at least 3 elements for [x, y, z]")
        return cls(arr[0], arr[1], arr[2])
    
    def to_array(self) -> np.ndarray:
        """
        Convert the point to a numpy array.
        
        Returns
        -------
        np.ndarray
            Array containing [x, y, z] coordinates
        """
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'Point') -> float:
        """
        Calculate the Euclidean distance to another point.
        
        Parameters
        ----------
        other : Point
            The other point
            
        Returns
        -------
        float
            The distance in mm
        """
        return np.sqrt((self.x - other.x)**2 + 
                       (self.y - other.y)**2 + 
                       (self.z - other.z)**2)
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert the point to a dictionary.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with x, y, z keys
        """
        return {'x': self.x, 'y': self.y, 'z': self.z}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Point':
        """
        Create a Point from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with x, y, z keys
            
        Returns
        -------
        Point
            New Point object
        """
        return cls(data['x'], data['y'], data['z'])
    
    def __eq__(self, other: 'Point') -> bool:
        """Check if two points are equal (same coordinates)."""
        if not isinstance(other, Point):
            return False
        return (self.x == other.x and 
                self.y == other.y and 
                self.z == other.z)
    
    def __str__(self) -> str:
        """String representation of the point."""
        return f"Point({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Contour:
    """
    Class representing a contour in a radiotherapy structure.
    
    A contour is a closed polygon on a specific z-plane (slice), represented
    by a series of connected 3D points.
    """
    
    def __init__(self, points: Union[List[Point], np.ndarray], z: float = None):
        """
        Initialize a contour.
        
        Parameters
        ----------
        points : List[Point] or np.ndarray
            List of Point objects or numpy array of shape (N, 3) where N is the number of points
        z : float, optional
            Z-coordinate of the contour (slice position). If None, uses the z-value from the first point
        """
        self.points: List[Point] = []
        
        # Convert numpy array to list of Point objects if needed
        if isinstance(points, np.ndarray):
            for i in range(points.shape[0]):
                self.points.append(Point(points[i, 0], points[i, 1], 
                                         points[i, 2] if points.shape[1] > 2 else (z or 0.0)))
        else:
            self.points = list(points)
        
        # Use specified z or get from first point
        self.z = z if z is not None else (self.points[0].z if self.points else 0.0)
    
    @property
    def num_points(self) -> int:
        """Get the number of points in the contour."""
        return len(self.points)
    
    def is_closed(self) -> bool:
        """
        Check if the contour is closed (first and last points are the same).
        
        Returns
        -------
        bool
            True if the contour is closed
        """
        if len(self.points) < 3:
            return False
        first = self.points[0]
        last = self.points[-1]
        return first == last or first.distance_to(last) < 0.001
    
    def close(self) -> None:
        """Ensure the contour is closed by adding a copy of the first point at the end if needed."""
        if not self.is_closed() and len(self.points) >= 3:
            self.points.append(Point(self.points[0].x, self.points[0].y, self.points[0].z))
    
    def to_array(self) -> np.ndarray:
        """
        Convert the contour to a numpy array.
        
        Returns
        -------
        np.ndarray
            Array of shape (N, 3) containing [x, y, z] for each point
        """
        return np.array([[p.x, p.y, p.z] for p in self.points])
    
    def calculate_area(self) -> float:
        """
        Calculate the area of the contour using the Shoelace formula.
        
        Returns
        -------
        float
            Area in mm²
        """
        if len(self.points) < 3:
            return 0.0
        
        # Make sure the contour is closed
        self.close()
        
        # Extract x, y coordinates
        x = np.array([p.x for p in self.points])
        y = np.array([p.y for p in self.points])
        
        # Shoelace formula
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the contour to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the contour
        """
        return {
            'points': [p.to_dict() for p in self.points],
            'z': self.z
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contour':
        """
        Create a Contour from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation of the contour
            
        Returns
        -------
        Contour
            New Contour object
        """
        points = [Point.from_dict(p) for p in data['points']]
        return cls(points, data.get('z'))
    
    def __str__(self) -> str:
        """String representation of the contour."""
        return f"Contour(z={self.z:.2f}, points={len(self.points)})"
