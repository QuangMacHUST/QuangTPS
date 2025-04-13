#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polygon Contouring Tool

This module implements the polygon drawing tool for manual structure
contouring in the QuangTPS treatment planning system.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QSlider, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QCursor, QPixmap, QPolygonF

from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class PolygonTool:
    """
    Polygon contouring tool for manual structure delineation.
    
    This tool allows creating precise polygonal contours by placing vertices
    at specific points. The user can add, move, and delete vertices to
    create an accurate contour.
    """
    
    def __init__(self):
        """Initialize the polygon tool."""
        # Tool properties
        self.line_width = 1
        self.vertex_size = 4
        self.close_threshold = 15  # Distance in pixels to auto-close polygon
        
        # Drawing state
        self.points = []  # Points in the polygon
        self.is_drawing = False
        self.current_contour = None
        self.hover_point = None  # Point being hovered for move/delete
        self.selected_point_index = -1  # Index of selected point
        self.mouse_pos = (0, 0)  # Current mouse position for preview
        
        # Create custom cursor
        self._cursor = self._create_cursor()
    
    def on_mouse_press(self, event, image_data=None):
        """
        Handle mouse press event to add or modify points.
        
        Args:
            event: Mouse event object
            image_data: Optional image data for reference
        """
        pos = event.pos()
        point = (pos.x(), pos.y())
        
        # Check if we're near an existing point
        near_point_idx = self._find_near_point(point)
        
        # Right button: cancel last point or exit drawing
        if event.button() == Qt.RightButton:
            if self.points:
                if self.selected_point_index >= 0:
                    # Deselect point
                    self.selected_point_index = -1
                else:
                    # Remove last point
                    self.points.pop()
                    logger.debug("Removed last point")
            else:
                # Exit drawing mode if no points
                self.is_drawing = False
            return
        
        # Middle button: delete point
        if event.button() == Qt.MiddleButton and near_point_idx >= 0:
            self.points.pop(near_point_idx)
            self.selected_point_index = -1
            logger.debug(f"Deleted point at index {near_point_idx}")
            return
        
        # Left button
        if event.button() == Qt.LeftButton:
            # If we clicked near the first point and have enough points, close the polygon
            if near_point_idx == 0 and len(self.points) > 2:
                # Finalize polygon
                self._finalize_polygon()
                return
                
            # If we're hovering over a point, select it for moving
            if near_point_idx >= 0:
                self.selected_point_index = near_point_idx
                logger.debug(f"Selected point at index {near_point_idx}")
                return
                
            # Otherwise, add a new point
            self.is_drawing = True
            self.points.append(point)
            logger.debug(f"Added point at {point}")
    
    def on_mouse_move(self, event, image_data=None):
        """
        Handle mouse move event for interactive feedback.
        
        Args:
            event: Mouse event object
            image_data: Optional image data for reference
        """
        pos = event.pos()
        point = (pos.x(), pos.y())
        self.mouse_pos = point
        
        # If we're dragging a selected point, move it
        if self.selected_point_index >= 0 and event.buttons() & Qt.LeftButton:
            self.points[self.selected_point_index] = point
            logger.debug(f"Moved point {self.selected_point_index} to {point}")
            return
            
        # Check if hovering near a point
        self.hover_point = self._find_near_point(point)
    
    def on_mouse_release(self, event, image_data=None):
        """
        Handle mouse release event to finish actions.
        
        Args:
            event: Mouse event object
            image_data: Optional image data for reference
        """
        # If we were moving a point, deselect it
        if self.selected_point_index >= 0 and event.button() == Qt.LeftButton:
            self.selected_point_index = -1
    
    def draw_preview(self, painter, image_data=None):
        """
        Draw the current polygon preview.
        
        Args:
            painter: QPainter object
            image_data: Optional image data for reference
        """
        if not self.points:
            return
            
        # Set up painter
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw lines between points
        pen = QPen(Qt.green)
        pen.setWidth(self.line_width)
        painter.setPen(pen)
        
        # Draw lines connecting points
        for i in range(1, len(self.points)):
            x1, y1 = self.points[i-1]
            x2, y2 = self.points[i]
            painter.drawLine(x1, y1, x2, y2)
        
        # Draw line from last point to mouse position if in drawing mode
        if self.is_drawing and self.points:
            x1, y1 = self.points[-1]
            x2, y2 = self.mouse_pos
            painter.setPen(QPen(QColor(0, 255, 0, 150), self.line_width, Qt.DashLine))
            painter.drawLine(x1, y1, x2, y2)
            
            # If we have more than 2 points, also draw line from mouse to first point
            if len(self.points) > 2:
                x1, y1 = self.points[0]
                painter.drawLine(x2, y2, x1, y1)
                
                # Check if we're close enough to the first point to close
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < self.close_threshold:
                    # Draw a highlight circle on the first point
                    painter.setPen(QPen(Qt.yellow, 2))
                    painter.setBrush(QBrush(QColor(255, 255, 0, 100)))
                    painter.drawEllipse(x1 - 5, y1 - 5, 10, 10)
        
        # Draw points
        for i, (x, y) in enumerate(self.points):
            if i == self.selected_point_index:
                # Selected point
                painter.setPen(QPen(Qt.yellow, 1))
                painter.setBrush(QBrush(Qt.yellow))
                painter.drawEllipse(x - self.vertex_size, y - self.vertex_size, 
                                  self.vertex_size * 2, self.vertex_size * 2)
            elif i == self.hover_point:
                # Hovered point
                painter.setPen(QPen(Qt.white, 1))
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
                painter.drawEllipse(x - self.vertex_size, y - self.vertex_size, 
                                  self.vertex_size * 2, self.vertex_size * 2)
            elif i == 0:
                # First point - slightly larger to show it's special
                painter.setPen(QPen(Qt.green, 1))
                painter.setBrush(QBrush(QColor(0, 255, 0, 200)))
                painter.drawEllipse(x - self.vertex_size - 1, y - self.vertex_size - 1, 
                                  (self.vertex_size + 1) * 2, (self.vertex_size + 1) * 2)
            else:
                # Regular points
                painter.setPen(QPen(Qt.green, 1))
                painter.setBrush(QBrush(QColor(0, 255, 0, 200)))
                painter.drawEllipse(x - self.vertex_size, y - self.vertex_size, 
                                  self.vertex_size * 2, self.vertex_size * 2)
    
    def get_cursor(self):
        """
        Get the cursor for this tool.
        
        Returns:
            QCursor: The cursor for the polygon tool
        """
        return self._cursor
    
    def set_line_width(self, width):
        """
        Set the line width for drawing.
        
        Args:
            width: Line width in pixels
        """
        self.line_width = max(1, int(width))
    
    def set_vertex_size(self, size):
        """
        Set the size of vertices.
        
        Args:
            size: Vertex size in pixels
        """
        self.vertex_size = max(2, int(size))
    
    def reset(self):
        """Reset the tool state."""
        self.points = []
        self.is_drawing = False
        self.current_contour = None
        self.hover_point = None
        self.selected_point_index = -1
    
    def get_contour(self):
        """
        Get the finalized contour.
        
        Returns:
            The contour as a numpy array of shape (N, 2)
        """
        return self.current_contour
    
    def _find_near_point(self, point):
        """
        Find the index of a point near the given coordinates.
        
        Args:
            point: Tuple (x, y) coordinates
            
        Returns:
            Index of the nearest point if within threshold, -1 otherwise
        """
        if not self.points:
            return -1
            
        x, y = point
        threshold = self.vertex_size * 2
        
        for i, (px, py) in enumerate(self.points):
            dx = px - x
            dy = py - y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < threshold:
                return i
                
        return -1
    
    def _finalize_polygon(self):
        """Finalize the polygon contour."""
        if len(self.points) < 3:
            logger.warning("Not enough points to create a valid polygon")
            return
            
        # Convert to numpy array
        points_array = np.array(self.points)
        
        # Make sure the polygon is closed
        if not np.array_equal(points_array[0], points_array[-1]):
            points_array = np.vstack([points_array, points_array[0]])
        
        # Store the contour
        self.current_contour = points_array
        
        # Reset drawing state
        self.is_drawing = False
        
        logger.debug(f"Finalized polygon with {len(points_array)} points")
    
    def _create_cursor(self):
        """
        Create a custom cursor for the polygon tool.
        
        Returns:
            QCursor: Custom cursor for the polygon tool
        """
        # Create a pixmap for the cursor
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        
        # Draw cursor shape
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw crosshair
        painter.setPen(QPen(Qt.black, 1))
        painter.drawLine(12, 0, 12, 24)  # Vertical line
        painter.drawLine(0, 12, 24, 12)  # Horizontal line
        
        # Draw small target circle
        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(QBrush(QColor(0, 255, 0, 100)))
        painter.drawEllipse(8, 8, 8, 8)
        
        painter.end()
        
        # Create cursor with hotspot at center
        return QCursor(pixmap, 12, 12)


class PolygonToolWidget(QWidget):
    """
    Widget for controlling the polygon contouring tool.
    
    This widget provides UI controls for configuring the polygon drawing
    tool parameters like line width and vertex size.
    """
    
    # Signals
    contour_created = pyqtSignal(object)  # Emits contour when created
    
    def __init__(self, parent=None):
        """Initialize the polygon tool widget."""
        super().__init__(parent)
        
        # Create the tool
        self.tool = PolygonTool()
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Instructions label
        instructions = QLabel(
            "Left-click: Add/move vertex\n"
            "Right-click: Remove last vertex\n"
            "Middle-click: Delete vertex\n"
            "Close by clicking near first vertex"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Line width control
        width_group = QGroupBox("Line Width")
        width_layout = QHBoxLayout(width_group)
        
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(1, 5)
        self.width_slider.setValue(self.tool.line_width)
        self.width_slider.setTickInterval(1)
        self.width_slider.setTickPosition(QSlider.TicksBelow)
        
        self.width_label = QLabel(f"{self.tool.line_width} px")
        
        width_layout.addWidget(self.width_slider)
        width_layout.addWidget(self.width_label)
        
        # Vertex size control
        vertex_group = QGroupBox("Vertex Size")
        vertex_layout = QHBoxLayout(vertex_group)
        
        self.vertex_slider = QSlider(Qt.Horizontal)
        self.vertex_slider.setRange(2, 8)
        self.vertex_slider.setValue(self.tool.vertex_size)
        self.vertex_slider.setTickInterval(1)
        self.vertex_slider.setTickPosition(QSlider.TicksBelow)
        
        self.vertex_label = QLabel(f"{self.tool.vertex_size} px")
        
        vertex_layout.addWidget(self.vertex_slider)
        vertex_layout.addWidget(self.vertex_label)
        
        # Reset button
        self.reset_button = QPushButton("Reset Polygon")
        
        # Add all controls to main layout
        layout.addWidget(width_group)
        layout.addWidget(vertex_group)
        layout.addWidget(self.reset_button)
        layout.addStretch(1)  # Push everything to the top
        
        # Connect signals
        self.width_slider.valueChanged.connect(self._on_width_changed)
        self.vertex_slider.valueChanged.connect(self._on_vertex_size_changed)
        self.reset_button.clicked.connect(self._on_reset)
    
    def on_mouse_press(self, event, image_data=None):
        """
        Handle mouse press event.
        
        Args:
            event: Mouse event
            image_data: Image data
        """
        self.tool.on_mouse_press(event, image_data)
        
        # If the contour was finalized, emit signal
        contour = self.tool.get_contour()
        if contour is not None and not self.tool.is_drawing:
            self.contour_created.emit(contour)
            self.tool.reset()
    
    def on_mouse_move(self, event, image_data=None):
        """
        Handle mouse move event.
        
        Args:
            event: Mouse event
            image_data: Image data
        """
        self.tool.on_mouse_move(event, image_data)
    
    def on_mouse_release(self, event, image_data=None):
        """
        Handle mouse release event.
        
        Args:
            event: Mouse event
            image_data: Image data
        """
        self.tool.on_mouse_release(event, image_data)
    
    def _on_width_changed(self, value):
        """
        Handle line width slider change.
        
        Args:
            value: New line width value
        """
        self.width_label.setText(f"{value} px")
        self.tool.set_line_width(value)
    
    def _on_vertex_size_changed(self, value):
        """
        Handle vertex size slider change.
        
        Args:
            value: New vertex size value
        """
        self.vertex_label.setText(f"{value} px")
        self.tool.set_vertex_size(value)
    
    def _on_reset(self):
        """Handle reset button click."""
        self.tool.reset()
    
    def get_cursor(self):
        """
        Get the cursor for this tool.
        
        Returns:
            QCursor: The tool's cursor
        """
        return self.tool.get_cursor()


# For testing
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    widget = PolygonToolWidget()
    widget.setWindowTitle("Polygon Tool Test")
    widget.resize(300, 300)
    widget.show()
    
    sys.exit(app.exec_()) 