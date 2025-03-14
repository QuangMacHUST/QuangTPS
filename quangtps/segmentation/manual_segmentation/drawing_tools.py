#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drawing Tools for Manual Segmentation in QuangTPS.

This module implements various drawing tools for manual contouring of anatomical
structures and tumors in radiotherapy treatment planning.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from abc import ABC, abstractmethod
from scipy import ndimage
import cv2

from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class DrawingTool(ABC):
    """Abstract base class for all drawing tools."""
    
    def __init__(self, name: str, cursor_type: str = "crosshair"):
        """
        Initialize drawing tool.
        
        Parameters
        ----------
        name : str
            Name of the tool
        cursor_type : str, optional
            Type of cursor to display when tool is active
        """
        self.name = name
        self.cursor_type = cursor_type
        self.active = False
        self.points = []
        self.callback = None
    
    def activate(self):
        """Activate the tool."""
        self.active = True
        self.points = []
        logger.debug(f"Activated {self.name} tool")
    
    def deactivate(self):
        """Deactivate the tool."""
        self.active = False
        self.points = []
        logger.debug(f"Deactivated {self.name} tool")
    
    def set_callback(self, callback: Callable):
        """
        Set callback function to be called when drawing is completed.
        
        Parameters
        ----------
        callback : Callable
            Callback function
        """
        self.callback = callback
    
    def reset(self):
        """Reset the tool state."""
        self.points = []
    
    @abstractmethod
    def on_mouse_down(self, x: int, y: int, slice_idx: int):
        """
        Handle mouse down event.
        
        Parameters
        ----------
        x : int
            X coordinate
        y : int
            Y coordinate
        slice_idx : int
            Slice index
        """
        pass
    
    @abstractmethod
    def on_mouse_move(self, x: int, y: int, slice_idx: int):
        """
        Handle mouse move event.
        
        Parameters
        ----------
        x : int
            X coordinate
        y : int
            Y coordinate
        slice_idx : int
            Slice index
        """
        pass
    
    @abstractmethod
    def on_mouse_up(self, x: int, y: int, slice_idx: int):
        """
        Handle mouse up event.
        
        Parameters
        ----------
        x : int
            X coordinate
        y : int
            Y coordinate
        slice_idx : int
            Slice index
        """
        pass
    
    @abstractmethod
    def apply_to_mask(self, mask: np.ndarray, slice_idx: int) -> np.ndarray:
        """
        Apply the tool's current state to modify a mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Mask to modify
        slice_idx : int
            Slice index
            
        Returns
        -------
        np.ndarray
            Modified mask
        """
        pass


class PenTool(DrawingTool):
    """
    Pen tool for precise point-by-point contouring.
    
    This tool allows for precise placement of points to create a contour.
    """
    
    def __init__(self, line_width: int = 1):
        """
        Initialize pen tool.
        
        Parameters
        ----------
        line_width : int, optional
            Width of the line in pixels
        """
        super().__init__(name="Pen", cursor_type="crosshair")
        self.line_width = line_width
        self.is_drawing = False
    
    def on_mouse_down(self, x: int, y: int, slice_idx: int):
        """Start drawing at the specified point."""
        self.is_drawing = True
        self.points.append((x, y))
    
    def on_mouse_move(self, x: int, y: int, slice_idx: int):
        """Continue drawing if mouse button is held down."""
        if self.is_drawing:
            self.points.append((x, y))
    
    def on_mouse_up(self, x: int, y: int, slice_idx: int):
        """Complete the drawing operation."""
        self.is_drawing = False
        self.points.append((x, y))
        if self.callback:
            self.callback(self.points, slice_idx)
    
    def apply_to_mask(self, mask: np.ndarray, slice_idx: int) -> np.ndarray:
        """Draw the contour on the specified slice of the mask."""
        if len(self.points) < 2:
            return mask
        
        # Create a copy of the mask
        result = mask.copy()
        
        # Draw the contour on the specific slice
        if 0 <= slice_idx < result.shape[0]:
            slice_mask = result[slice_idx]
            
            # Convert points to the format expected by OpenCV
            contour_points = np.array(self.points, dtype=np.int32)
            
            # Draw the contour
            cv2.polylines(
                slice_mask, 
                [contour_points], 
                isClosed=True, 
                color=1, 
                thickness=self.line_width
            )
            
            # Update the slice in the mask
            result[slice_idx] = slice_mask
        
        return result


class BrushTool(DrawingTool):
    """
    Brush tool for freeform painting on the mask.
    
    This tool allows for painting with a circular brush.
    """
    
    def __init__(self, brush_size: int = 5, brush_value: int = 1):
        """
        Initialize brush tool.
        
        Parameters
        ----------
        brush_size : int, optional
            Radius of the brush in pixels
        brush_value : int, optional
            Value to paint with (1 for adding, 0 for erasing)
        """
        super().__init__(name="Brush", cursor_type="circle")
        self.brush_size = brush_size
        self.brush_value = brush_value
        self.is_painting = False
        self.last_point = None
    
    def set_brush_size(self, size: int):
        """
        Set brush size.
        
        Parameters
        ----------
        size : int
            New brush size in pixels
        """
        self.brush_size = max(1, size)
    
    def set_brush_value(self, value: int):
        """
        Set brush value.
        
        Parameters
        ----------
        value : int
            New brush value (1 for adding, 0 for erasing)
        """
        self.brush_value = value
    
    def on_mouse_down(self, x: int, y: int, slice_idx: int):
        """Start painting at the specified point."""
        self.is_painting = True
        self.points = [(x, y)]
        self.last_point = (x, y)
    
    def on_mouse_move(self, x: int, y: int, slice_idx: int):
        """Continue painting if mouse button is held down."""
        if self.is_painting:
            # Interpolate between last point and current point to avoid gaps
            if self.last_point:
                last_x, last_y = self.last_point
                
                # Calculate distance
                dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                
                # If distance is greater than brush_size/2, interpolate
                if dist > self.brush_size / 2:
                    steps = int(np.ceil(dist / (self.brush_size / 2)))
                    for i in range(1, steps):
                        interp_x = int(last_x + (x - last_x) * (i / steps))
                        interp_y = int(last_y + (y - last_y) * (i / steps))
                        self.points.append((interp_x, interp_y))
            
            self.points.append((x, y))
            self.last_point = (x, y)
    
    def on_mouse_up(self, x: int, y: int, slice_idx: int):
        """Complete the painting operation."""
        self.is_painting = False
        self.points.append((x, y))
        if self.callback:
            self.callback(self.points, slice_idx)
        self.last_point = None
    
    def apply_to_mask(self, mask: np.ndarray, slice_idx: int) -> np.ndarray:
        """Paint on the specified slice of the mask."""
        if not self.points:
            return mask
        
        # Create a copy of the mask
        result = mask.copy()
        
        # Paint on the specific slice
        if 0 <= slice_idx < result.shape[0]:
            slice_mask = result[slice_idx]
            
            # Create a circular brush
            y, x = np.ogrid[-self.brush_size:self.brush_size+1, -self.brush_size:self.brush_size+1]
            brush = (x*x + y*y <= self.brush_size*self.brush_size).astype(np.uint8)
            
            # Apply brush at each point
            for px, py in self.points:
                # Calculate brush boundaries
                x_min = max(0, px - self.brush_size)
                x_max = min(slice_mask.shape[1], px + self.brush_size + 1)
                y_min = max(0, py - self.brush_size)
                y_max = min(slice_mask.shape[0], py + self.brush_size + 1)
                
                # Calculate brush offset
                brush_x_min = max(0, self.brush_size - px)
                brush_y_min = max(0, self.brush_size - py)
                brush_x_max = brush_x_min + (x_max - x_min)
                brush_y_max = brush_y_min + (y_max - y_min)
                
                # Apply brush
                if self.brush_value == 1:
                    # Add to mask
                    slice_mask[y_min:y_max, x_min:x_max] = np.maximum(
                        slice_mask[y_min:y_max, x_min:x_max],
                        brush[brush_y_min:brush_y_max, brush_x_min:brush_x_max]
                    )
                else:
                    # Erase from mask
                    slice_mask[y_min:y_max, x_min:x_max] = np.minimum(
                        slice_mask[y_min:y_max, x_min:x_max],
                        1 - brush[brush_y_min:brush_y_max, brush_x_min:brush_x_max]
                    )
            
            # Update the slice in the mask
            result[slice_idx] = slice_mask
        
        return result


class PolygonTool(DrawingTool):
    """
    Polygon tool for creating polygonal contours.
    
    This tool allows for creating closed polygons by placing vertices.
    """
    
    def __init__(self, line_width: int = 1):
        """
        Initialize polygon tool.
        
        Parameters
        ----------
        line_width : int, optional
            Width of the polygon outline in pixels
        """
        super().__init__(name="Polygon", cursor_type="crosshair")
        self.line_width = line_width
        self.is_closed = False
        self.close_threshold = 10  # Pixels distance to close polygon
    
    def on_mouse_down(self, x: int, y: int, slice_idx: int):
        """Add a vertex at the specified point."""
        # If polygon is already closed, reset it
        if self.is_closed:
            self.reset()
            self.is_closed = False
        
        # Check if click is near the first point (to close polygon)
        if len(self.points) > 2:
            first_x, first_y = self.points[0]
            dist = np.sqrt((x - first_x)**2 + (y - first_y)**2)
            
            if dist < self.close_threshold:
                # Close the polygon
                self.is_closed = True
                if self.callback:
                    self.callback(self.points, slice_idx)
                return
        
        # Add the point
        self.points.append((x, y))
    
    def on_mouse_move(self, x: int, y: int, slice_idx: int):
        """Update visual feedback for polygon creation."""
        # This method would typically update a visual preview
        # but doesn't modify the points list
        pass
    
    def on_mouse_up(self, x: int, y: int, slice_idx: int):
        """Complete the vertex placement."""
        # No additional action needed, as vertex was added on mouse_down
        pass
    
    def apply_to_mask(self, mask: np.ndarray, slice_idx: int) -> np.ndarray:
        """Draw the polygon on the specified slice of the mask."""
        if len(self.points) < 3:
            return mask
        
        # Create a copy of the mask
        result = mask.copy()
        
        # Draw the polygon on the specific slice
        if 0 <= slice_idx < result.shape[0]:
            slice_mask = result[slice_idx]
            
            # Convert points to the format expected by OpenCV
            contour_points = np.array(self.points, dtype=np.int32)
            
            # Create an empty mask for this slice
            temp_mask = np.zeros_like(slice_mask)
            
            # Draw the filled polygon
            cv2.fillPoly(temp_mask, [contour_points], color=1)
            
            # Draw the outline
            cv2.polylines(
                temp_mask, 
                [contour_points], 
                isClosed=True, 
                color=1, 
                thickness=self.line_width
            )
            
            # Update the slice in the mask
            result[slice_idx] = temp_mask
        
        return result


class FreehandTool(DrawingTool):
    """
    Freehand tool for drawing irregular shapes.
    
    This tool allows for drawing freehand shapes by tracking mouse movement.
    """
    
    def __init__(self, line_width: int = 1):
        """
        Initialize freehand tool.
        
        Parameters
        ----------
        line_width : int, optional
            Width of the line in pixels
        """
        super().__init__(name="Freehand", cursor_type="crosshair")
        self.line_width = line_width
        self.is_drawing = False
    
    def on_mouse_down(self, x: int, y: int, slice_idx: int):
        """Start drawing at the specified point."""
        self.is_drawing = True
        self.points = [(x, y)]
    
    def on_mouse_move(self, x: int, y: int, slice_idx: int):
        """Continue drawing if mouse button is held down."""
        if self.is_drawing:
            self.points.append((x, y))
    
    def on_mouse_up(self, x: int, y: int, slice_idx: int):
        """Complete the drawing operation."""
        self.is_drawing = False
        # Close the shape by adding the first point again
        if len(self.points) > 2:
            self.points.append(self.points[0])
        
        if self.callback:
            self.callback(self.points, slice_idx)
    
    def apply_to_mask(self, mask: np.ndarray, slice_idx: int) -> np.ndarray:
        """Draw the freehand shape on the specified slice of the mask."""
        if len(self.points) < 3:
            return mask
        
        # Create a copy of the mask
        result = mask.copy()
        
        # Draw the shape on the specific slice
        if 0 <= slice_idx < result.shape[0]:
            slice_mask = result[slice_idx]
            
            # Convert points to the format expected by OpenCV
            contour_points = np.array(self.points, dtype=np.int32)
            
            # Create an empty mask for this slice
            temp_mask = np.zeros_like(slice_mask)
            
            # Draw the filled shape
            cv2.fillPoly(temp_mask, [contour_points], color=1)
            
            # Update the slice in the mask
            result[slice_idx] = temp_mask
        
        return result


class EraserTool(BrushTool):
    """
    Eraser tool for removing parts of a contour.
    
    This tool is a specialized brush tool that erases instead of adds.
    """
    
    def __init__(self, eraser_size: int = 5):
        """
        Initialize eraser tool.
        
        Parameters
        ----------
        eraser_size : int, optional
            Radius of the eraser in pixels
        """
        super().__init__(brush_size=eraser_size, brush_value=0)
        self.name = "Eraser"


class DrawingToolManager:
    """
    Manager class for handling multiple drawing tools.
    
    This class provides a centralized way to manage different drawing tools
    and switch between them.
    """
    
    def __init__(self):
        """Initialize the drawing tool manager."""
        self.tools = {}
        self.active_tool = None
        
        # Initialize default tools
        self._init_default_tools()
    
    def _init_default_tools(self):
        """Initialize the default set of drawing tools."""
        self.add_tool(PenTool())
        self.add_tool(BrushTool())
        self.add_tool(PolygonTool())
        self.add_tool(FreehandTool())
        self.add_tool(EraserTool())
    
    def add_tool(self, tool: DrawingTool):
        """
        Add a drawing tool to the manager.
        
        Parameters
        ----------
        tool : DrawingTool
            Drawing tool to add
        """
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[DrawingTool]:
        """
        Get a tool by name.
        
        Parameters
        ----------
        name : str
            Name of the tool
            
        Returns
        -------
        DrawingTool or None
            The requested tool, or None if not found
        """
        return self.tools.get(name)
    
    def activate_tool(self, name: str) -> bool:
        """
        Activate a tool by name.
        
        Parameters
        ----------
        name : str
            Name of the tool to activate
            
        Returns
        -------
        bool
            True if tool was activated, False otherwise
        """
        # Deactivate current tool if any
        if self.active_tool:
            self.active_tool.deactivate()
        
        # Get and activate the new tool
        tool = self.get_tool(name)
        if tool:
            tool.activate()
            self.active_tool = tool
            logger.info(f"Activated tool: {name}")
            return True
        else:
            logger.warning(f"Tool not found: {name}")
            self.active_tool = None
            return False
    
    def get_active_tool(self) -> Optional[DrawingTool]:
        """
        Get the currently active tool.
        
        Returns
        -------
        DrawingTool or None
            The active tool, or None if no tool is active
        """
        return self.active_tool
    
    def reset_active_tool(self):
        """Reset the state of the active tool."""
        if self.active_tool:
            self.active_tool.reset()
    
    def set_tool_callback(self, name: str, callback: Callable) -> bool:
        """
        Set callback function for a specific tool.
        
        Parameters
        ----------
        name : str
            Name of the tool
        callback : Callable
            Callback function
            
        Returns
        -------
        bool
            True if callback was set, False otherwise
        """
        tool = self.get_tool(name)
        if tool:
            tool.set_callback(callback)
            return True
        return False
    
    def set_callback_for_all_tools(self, callback: Callable):
        """
        Set the same callback function for all tools.
        
        Parameters
        ----------
        callback : Callable
            Callback function
        """
        for tool in self.tools.values():
            tool.set_callback(callback)
    
    def on_mouse_down(self, x: int, y: int, slice_idx: int):
        """
        Handle mouse down event for the active tool.
        
        Parameters
        ----------
        x : int
            X coordinate
        y : int
            Y coordinate
        slice_idx : int
            Slice index
        """
        if self.active_tool:
            self.active_tool.on_mouse_down(x, y, slice_idx)
    
    def on_mouse_move(self, x: int, y: int, slice_idx: int):
        """
        Handle mouse move event for the active tool.
        
        Parameters
        ----------
        x : int
            X coordinate
        y : int
            Y coordinate
        slice_idx : int
            Slice index
        """
        if self.active_tool:
            self.active_tool.on_mouse_move(x, y, slice_idx)
    
    def on_mouse_up(self, x: int, y: int, slice_idx: int):
        """
        Handle mouse up event for the active tool.
        
        Parameters
        ----------
        x : int
            X coordinate
        y : int
            Y coordinate
        slice_idx : int
            Slice index
        """
        if self.active_tool:
            self.active_tool.on_mouse_up(x, y, slice_idx)
    
    def apply_active_tool_to_mask(self, mask: np.ndarray, slice_idx: int) -> np.ndarray:
        """
        Apply the active tool to modify a mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Mask to modify
        slice_idx : int
            Slice index
            
        Returns
        -------
        np.ndarray
            Modified mask
        """
        if self.active_tool:
            return self.active_tool.apply_to_mask(mask, slice_idx)
        return mask
