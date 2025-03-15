#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for polygon contour drawing tools.

This module provides interactive tools for creating and editing polygon-based
contours, which are essential in radiotherapy treatment planning.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from skimage import draw

from quangtps.segmentation.contour.contour_tools import ContourTool

logger = logging.getLogger(__name__)


class PolygonTool(ContourTool):
    """
    Tool for creating and editing polygon-based contours.
    
    This tool extends the basic ContourTool with specific functionality
    for creating polygon contours with interactive editing capabilities.
    """
    
    def __init__(self):
        """Initialize polygon tool."""
        super().__init__()
        
        # Polygon-specific attributes
        self.is_polygon_closed = False
        self.temp_point = None  # For showing temporary point during mouse movement
        self.selected_point_index = None  # Index of selected vertex
        self.hover_distance_threshold = 10  # Pixel distance for hovering detection
        self.snap_to_grid = False  # Whether to snap vertices to grid
        self.grid_size = 1.0  # Grid size in pixels
        
    def start_new_polygon(self):
        """Start a new polygon contour."""
        self.active_contour = []
        self.is_polygon_closed = False
        self.temp_point = None
        self.selected_point_index = None
        
    def add_point(self, point: Tuple[float, float]):
        """
        Add a point to the current polygon.
        
        Parameters
        ----------
        point : Tuple[float, float]
            (x, y) coordinates of the point to add
        """
        if self.is_polygon_closed:
            # If polygon is closed, add a new polygon
            self.close_contour()
            self.active_contour = []
            self.is_polygon_closed = False
        
        # Apply snapping if enabled
        if self.snap_to_grid:
            point = (
                round(point[0] / self.grid_size) * self.grid_size,
                round(point[1] / self.grid_size) * self.grid_size
            )
        
        # Check if we're closing the polygon
        if len(self.active_contour) >= 3 and self._is_near_first_point(point):
            self.close_polygon()
        else:
            super().add_point(point)
    
    def close_polygon(self):
        """Close the current polygon by connecting the last point to the first."""
        if self.active_contour and len(self.active_contour) >= 3:
            # Add the first point again to close the polygon
            self.active_contour.append(self.active_contour[0])
            self.is_polygon_closed = True
            
            # Convert to numpy array if it's a list
            if isinstance(self.active_contour, list):
                self.active_contour = np.array(self.active_contour)
    
    def _is_near_first_point(self, point: Tuple[float, float], threshold: float = 10.0) -> bool:
        """
        Check if a point is near the first point of the polygon.
        
        Parameters
        ----------
        point : Tuple[float, float]
            Point to check
        threshold : float, optional
            Distance threshold in pixels
            
        Returns
        -------
        bool
            True if the point is near the first point
        """
        if not self.active_contour:
            return False
        
        first_point = self.active_contour[0]
        distance = np.sqrt((point[0] - first_point[0])**2 + (point[1] - first_point[1])**2)
        return distance < threshold
    
    def set_temp_point(self, point: Tuple[float, float]):
        """
        Set a temporary point for interactive feedback.
        
        Parameters
        ----------
        point : Tuple[float, float]
            Temporary point coordinates
        """
        self.temp_point = point
    
    def clear_temp_point(self):
        """Clear the temporary point."""
        self.temp_point = None
    
    def select_point(self, point: Tuple[float, float]) -> bool:
        """
        Select a vertex point near the given coordinates.
        
        Parameters
        ----------
        point : Tuple[float, float]
            Coordinates to check
            
        Returns
        -------
        bool
            True if a point was selected
        """
        if not self.active_contour or len(self.active_contour) == 0:
            return False
        
        min_distance = float('inf')
        selected_index = None
        
        for i, vertex in enumerate(self.active_contour):
            distance = np.sqrt((point[0] - vertex[0])**2 + (point[1] - vertex[1])**2)
            
            if distance < min_distance and distance < self.hover_distance_threshold:
                min_distance = distance
                selected_index = i
        
        self.selected_point_index = selected_index
        return selected_index is not None
    
    def move_selected_point(self, point: Tuple[float, float]):
        """
        Move the selected vertex to a new position.
        
        Parameters
        ----------
        point : Tuple[float, float]
            New point coordinates
        """
        if self.selected_point_index is not None and self.active_contour:
            # Apply snapping if enabled
            if self.snap_to_grid:
                point = (
                    round(point[0] / self.grid_size) * self.grid_size,
                    round(point[1] / self.grid_size) * self.grid_size
                )
            
            # Move the selected point
            self.active_contour[self.selected_point_index] = point
            
            # If this is the first or last point and the polygon is closed,
            # update the other end point as well
            if self.is_polygon_closed and len(self.active_contour) > 1:
                if self.selected_point_index == 0:
                    self.active_contour[-1] = point
                elif self.selected_point_index == len(self.active_contour) - 1:
                    self.active_contour[0] = point
    
    def deselect_point(self):
        """Deselect the currently selected vertex."""
        self.selected_point_index = None
    
    def insert_point(self, point: Tuple[float, float], segment_index: int):
        """
        Insert a point in a specific segment of the polygon.
        
        Parameters
        ----------
        point : Tuple[float, float]
            Point to insert
        segment_index : int
            Index of the segment to insert the point into
        """
        if not self.active_contour or segment_index >= len(self.active_contour) - 1:
            return
        
        # Apply snapping if enabled
        if self.snap_to_grid:
            point = (
                round(point[0] / self.grid_size) * self.grid_size,
                round(point[1] / self.grid_size) * self.grid_size
            )
        
        # Convert to list if it's a numpy array
        if isinstance(self.active_contour, np.ndarray):
            self.active_contour = self.active_contour.tolist()
        
        # Insert the point after the segment start
        self.active_contour.insert(segment_index + 1, point)
        
        # Convert back to numpy array if needed
        if isinstance(self.active_contour, list):
            self.active_contour = np.array(self.active_contour)
    
    def delete_selected_point(self):
        """Delete the currently selected vertex."""
        if (self.selected_point_index is not None and 
            self.active_contour and 
            len(self.active_contour) > 3):  # Minimum 3 points for a polygon
            
            # Convert to list if it's a numpy array
            if isinstance(self.active_contour, np.ndarray):
                self.active_contour = self.active_contour.tolist()
            
            # Special case for first and last points in a closed polygon
            if self.is_polygon_closed and len(self.active_contour) > 1:
                if self.selected_point_index == 0:
                    # Delete first and last point (they're the same)
                    self.active_contour.pop(0)
                    self.active_contour.pop(-1)
                    # Add the first point at the end to keep it closed
                    self.active_contour.append(self.active_contour[0])
                elif self.selected_point_index == len(self.active_contour) - 1:
                    # Delete last point and update first point
                    self.active_contour.pop(-1)
                    # Add the first point at the end to keep it closed
                    self.active_contour.append(self.active_contour[0])
                else:
                    # Normal case
                    self.active_contour.pop(self.selected_point_index)
            else:
                # Normal case for open polygon
                self.active_contour.pop(self.selected_point_index)
            
            # Convert back to numpy array if needed
            if isinstance(self.active_contour, list):
                self.active_contour = np.array(self.active_contour)
            
            self.selected_point_index = None
    
    def find_segment(self, point: Tuple[float, float]) -> Optional[int]:
        """
        Find the closest polygon segment to a point.
        
        Parameters
        ----------
        point : Tuple[float, float]
            Point to check
            
        Returns
        -------
        Optional[int]
            Index of the starting vertex of the closest segment, or None
        """
        if not self.active_contour or len(self.active_contour) < 2:
            return None
        
        min_distance = float('inf')
        closest_segment = None
        
        # Check distance to each segment
        num_points = len(self.active_contour)
        for i in range(num_points - 1):
            p1 = self.active_contour[i]
            p2 = self.active_contour[i + 1]
            
            # Calculate distance from point to line segment
            distance = self._point_to_segment_distance(point, p1, p2)
            
            if distance < min_distance and distance < self.hover_distance_threshold:
                min_distance = distance
                closest_segment = i
        
        return closest_segment
    
    def _point_to_segment_distance(self, point: Tuple[float, float], 
                                  p1: Tuple[float, float], 
                                  p2: Tuple[float, float]) -> float:
        """
        Calculate the minimum distance from a point to a line segment.
        
        Parameters
        ----------
        point : Tuple[float, float]
            Point to check
        p1 : Tuple[float, float]
            First endpoint of the segment
        p2 : Tuple[float, float]
            Second endpoint of the segment
            
        Returns
        -------
        float
            Minimum distance from point to segment
        """
        x, y = point
        x1, y1 = p1
        x2, y2 = p2
        
        # Calculate squared length of segment
        l2 = (x2 - x1)**2 + (y2 - y1)**2
        
        # If segment is a point, return distance to that point
        if l2 == 0:
            return np.sqrt((x - x1)**2 + (y - y1)**2)
        
        # Calculate projection of point onto the segment line
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2))
        
        # Calculate closest point on segment
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        
        # Return distance to closest point
        return np.sqrt((x - px)**2 + (y - py)**2)
    
    def draw_contours(self, axes, active_color='r', completed_color='g', 
                    vertex_color='b', selected_color='y'):
        """
        Draw all contours on the given axes.
        
        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Axes to draw on
        active_color : str, optional
            Color for the active contour
        completed_color : str, optional
            Color for completed contours
        vertex_color : str, optional
            Color for vertices
        selected_color : str, optional
            Color for selected vertex
        """
        # Draw stored contours
        for contour in self.contours:
            if len(contour) > 1:
                axes.plot(contour[:, 0], contour[:, 1], f'{completed_color}-', linewidth=2)
                
                # Draw vertices
                axes.plot(contour[:, 0], contour[:, 1], f'{vertex_color}o', markersize=4)
        
        # Draw active contour
        if self.active_contour is not None and len(self.active_contour) > 0:
            active_points = np.array(self.active_contour)
            
            # Draw lines between points
            if len(active_points) > 1:
                axes.plot(active_points[:, 0], active_points[:, 1], f'{active_color}-', linewidth=2)
            
            # Draw vertices
            axes.plot(active_points[:, 0], active_points[:, 1], f'{vertex_color}o', markersize=4)
            
            # Draw temporary point if exists
            if self.temp_point is not None and not self.is_polygon_closed:
                axes.plot([active_points[-1, 0], self.temp_point[0]], 
                         [active_points[-1, 1], self.temp_point[1]], 
                         f'{active_color}--', linewidth=1)
            
            # Draw selected point with different color
            if self.selected_point_index is not None:
                selected_point = active_points[self.selected_point_index]
                axes.plot([selected_point[0]], [selected_point[1]], 
                         f'{selected_color}o', markersize=6)
    
    def create_mask(self, shape):
        """
        Create a binary mask from the current contour.
        
        Parameters
        ----------
        shape : tuple
            Shape of the output mask (height, width)
            
        Returns
        -------
        np.ndarray
            Binary mask with 1 inside the contour and 0 outside
        """
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Create mask from active contour if it exists
        if self.active_contour is not None and len(self.active_contour) > 2:
            # Convert contour to integer coordinates
            contour = np.array(self.active_contour).astype(int)
            
            # Make sure it's closed for proper mask creation
            if not self.is_polygon_closed and np.any(contour[0] != contour[-1]):
                contour = np.vstack([contour, contour[0]])
            
            # Create polygon mask
            rr, cc = draw.polygon(contour[:, 1], contour[:, 0], shape)
            mask[rr, cc] = 1
        
        return mask
    
    def set_snap_to_grid(self, enable: bool, grid_size: float = 1.0):
        """
        Set whether vertices should snap to a grid.
        
        Parameters
        ----------
        enable : bool
            Whether to enable snapping
        grid_size : float, optional
            Grid size in pixels
        """
        self.snap_to_grid = enable
        self.grid_size = grid_size
        
        # If snapping is enabled, snap existing points
        if enable and self.active_contour is not None and len(self.active_contour) > 0:
            # Convert to numpy array if it's a list
            if isinstance(self.active_contour, list):
                self.active_contour = np.array(self.active_contour)
            
            # Snap all points
            self.active_contour = np.round(self.active_contour / grid_size) * grid_size


class SplineTool(PolygonTool):
    """
    Tool for creating and editing spline-based contours.
    
    This tool extends the PolygonTool to create smooth spline curves
    that pass through the control points.
    """
    
    def __init__(self):
        """Initialize spline tool."""
        super().__init__()
        
        # Spline-specific attributes
        self.spline_resolution = 100  # Number of points in the spline
        self.tension = 0.5  # Controls the "tightness" of the spline (0-1)
        self.spline_points = None  # Computed spline points
    
    def compute_spline(self):
        """Compute spline curve from control points."""
        if not self.active_contour or len(self.active_contour) < 2:
            self.spline_points = None
            return
        
        # Get control points
        control_points = np.array(self.active_contour)
        
        # If the polygon is closed, add the first points at the end
        if self.is_polygon_closed:
            # Add three points from the beginning to ensure smooth closure
            if len(control_points) > 3:
                control_points = np.vstack([control_points, control_points[1:4]])
        
        # Calculate the spline using cardinal spline interpolation
        self.spline_points = self._cardinal_spline(control_points, self.tension)
    
    def _cardinal_spline(self, points: np.ndarray, tension: float = 0.5) -> np.ndarray:
        """
        Compute cardinal spline through given points.
        
        Parameters
        ----------
        points : np.ndarray
            Control points as nx2 array
        tension : float, optional
            Tension parameter (0-1)
            
        Returns
        -------
        np.ndarray
            Spline points as mx2 array
        """
        n_points = len(points)
        
        if n_points < 2:
            return points
        
        if n_points == 2:
            # Linear interpolation for two points
            t = np.linspace(0, 1, self.spline_resolution)
            return np.array([(1-t_i)*points[0] + t_i*points[1] for t_i in t])
        
        # If not closed, duplicate first and last points to get nice end conditions
        if not self.is_polygon_closed:
            # Extend by repeating first and last points
            extended_points = np.vstack([
                [points[0]], points, [points[-1]]
            ])
        else:
            # For closed curve, use the actual connectivity
            extended_points = np.vstack([
                [points[-2]], points, [points[1]]
            ])
        
        # Compute spline segments
        n_extended = len(extended_points)
        spline_segments = []
        
        for i in range(1, n_extended - 2):
            p0 = extended_points[i-1]
            p1 = extended_points[i]
            p2 = extended_points[i+1]
            p3 = extended_points[i+2]
            
            # Number of points in this segment (proportional to segment length)
            segment_length = np.linalg.norm(p2 - p1)
            n_segment = max(2, int(self.spline_resolution * segment_length / 
                                 np.sum([np.linalg.norm(extended_points[j+1] - extended_points[j]) 
                                        for j in range(1, n_extended-2)])))
            
            # Parameter values
            t = np.linspace(0, 1, n_segment)
            
            # Cardinal spline basis
            s = (1 - tension) / 2
            
            # Compute spline points
            segment_points = []
            for t_i in t:
                t2 = t_i * t_i
                t3 = t2 * t_i
                
                # Cardinal spline matrix
                h1 = 2*t3 - 3*t2 + 1
                h2 = -2*t3 + 3*t2
                h3 = t3 - 2*t2 + t_i
                h4 = t3 - t2
                
                # Compute point
                point = (h1*p1 + h2*p2 + s*(h3*(p2-p0) + h4*(p3-p1)))
                segment_points.append(point)
            
            spline_segments.append(segment_points)
        
        # Combine all segments
        spline_points = np.vstack(spline_segments)
        
        # If closed, connect back to the first point
        if self.is_polygon_closed:
            spline_points = np.vstack([spline_points, [spline_points[0]]])
        
        return spline_points
    
    def add_point(self, point: Tuple[float, float]):
        """
        Add a point to the current spline.
        
        Parameters
        ----------
        point : Tuple[float, float]
            (x, y) coordinates of the point to add
        """
        super().add_point(point)
        self.compute_spline()
    
    def move_selected_point(self, point: Tuple[float, float]):
        """
        Move the selected vertex to a new position.
        
        Parameters
        ----------
        point : Tuple[float, float]
            New point coordinates
        """
        super().move_selected_point(point)
        self.compute_spline()
    
    def delete_selected_point(self):
        """Delete the currently selected vertex."""
        super().delete_selected_point()
        self.compute_spline()
    
    def insert_point(self, point: Tuple[float, float], segment_index: int):
        """
        Insert a point in a specific segment of the spline.
        
        Parameters
        ----------
        point : Tuple[float, float]
            Point to insert
        segment_index : int
            Index of the segment to insert the point into
        """
        super().insert_point(point, segment_index)
        self.compute_spline()
    
    def close_polygon(self):
        """Close the current spline by connecting the last point to the first."""
        super().close_polygon()
        self.compute_spline()
    
    def draw_contours(self, axes, active_color='r', completed_color='g', 
                    vertex_color='b', selected_color='y'):
        """
        Draw all contours and splines on the given axes.
        
        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Axes to draw on
        active_color : str, optional
            Color for the active contour
        completed_color : str, optional
            Color for completed contours
        vertex_color : str, optional
            Color for vertices
        selected_color : str, optional
            Color for selected vertex
        """
        # Draw stored contours
        for contour in self.contours:
            if len(contour) > 1:
                axes.plot(contour[:, 0], contour[:, 1], f'{completed_color}-', linewidth=2)
                
                # Draw vertices
                axes.plot(contour[:, 0], contour[:, 1], f'{vertex_color}o', markersize=4)
        
        # Draw active contour control points
        if self.active_contour is not None and len(self.active_contour) > 0:
            active_points = np.array(self.active_contour)
            
            # Draw control points
            axes.plot(active_points[:, 0], active_points[:, 1], f'{vertex_color}o', markersize=4)
            
            # Draw lines between control points (dashed)
            if len(active_points) > 1:
                axes.plot(active_points[:, 0], active_points[:, 1], f'{active_color}--', linewidth=1)
            
            # Draw temporary point if exists
            if self.temp_point is not None and not self.is_polygon_closed:
                axes.plot([active_points[-1, 0], self.temp_point[0]], 
                         [active_points[-1, 1], self.temp_point[1]], 
                         f'{active_color}--', linewidth=1)
            
            # Draw selected point with different color
            if self.selected_point_index is not None:
                selected_point = active_points[self.selected_point_index]
                axes.plot([selected_point[0]], [selected_point[1]], 
                         f'{selected_color}o', markersize=6)
        
        # Draw spline curve
        if self.spline_points is not None and len(self.spline_points) > 1:
            axes.plot(self.spline_points[:, 0], self.spline_points[:, 1], 
                     f'{active_color}-', linewidth=2)
    
    def create_mask(self, shape):
        """
        Create a binary mask from the current spline.
        
        Parameters
        ----------
        shape : tuple
            Shape of the output mask (height, width)
            
        Returns
        -------
        np.ndarray
            Binary mask with 1 inside the spline and 0 outside
        """
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Create mask from spline points if they exist
        if self.spline_points is not None and len(self.spline_points) > 2:
            # Convert spline points to integer coordinates
            contour = np.array(self.spline_points).astype(int)
            
            # Make sure it's closed for proper mask creation
            if not self.is_polygon_closed and np.any(contour[0] != contour[-1]):
                contour = np.vstack([contour, contour[0]])
            
            # Create polygon mask
            rr, cc = draw.polygon(contour[:, 1], contour[:, 0], shape)
            mask[rr, cc] = 1
        
        return mask
