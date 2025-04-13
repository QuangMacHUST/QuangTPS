#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Freehand Contouring Tool

This module implements the freehand drawing tool for manual structure 
contouring in the QuangTPS treatment planning system.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QSlider, QCheckBox, QComboBox, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QCursor, QPixmap

from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class FreehandTool:
    """
    Freehand contouring tool for manual structure delineation.
    
    This tool allows drawing freehand contours by tracking mouse movements
    while the mouse button is pressed.
    """
    
    def __init__(self):
        """Initialize the freehand tool."""
        # Tool properties
        self.line_width = 1
        self.smoothing_enabled = False
        self.smoothing_level = 3  # Default smoothing level
        
        # Drawing state
        self.is_drawing = False
        self.points = []  # Points collected during drawing
        self.current_contour = None  # Current contour being drawn
        
        # Create custom cursor
        self._cursor = self._create_cursor()
    
    def on_mouse_press(self, event, image_data=None):
        """
        Handle mouse press event to start drawing.
        
        Args:
            event: Mouse event object
            image_data: Optional image data for reference
        """
        # Reset points and start drawing
        self.points = []
        self.is_drawing = True
        
        # Add first point
        pos = event.pos()
        self.points.append((pos.x(), pos.y()))
        
        logger.debug("Started freehand drawing")
    
    def on_mouse_move(self, event, image_data=None):
        """
        Handle mouse move event to continue drawing.
        
        Args:
            event: Mouse event object
            image_data: Optional image data for reference
        """
        if not self.is_drawing:
            return
        
        # Add point to the contour
        pos = event.pos()
        self.points.append((pos.x(), pos.y()))
        
        # We don't finalize the contour yet, just collect points
    
    def on_mouse_release(self, event, image_data=None):
        """
        Handle mouse release event to finish drawing.
        
        Args:
            event: Mouse event object
            image_data: Optional image data for reference
        """
        if not self.is_drawing:
            return
            
        # Add final point
        pos = event.pos()
        self.points.append((pos.x(), pos.y()))
        
        # Stop drawing
        self.is_drawing = False
        
        # Need at least 3 points to form a valid contour
        if len(self.points) < 3:
            logger.warning("Not enough points to create a contour")
            self.points = []
            return
        
        # Process the contour
        self._process_contour()
        
        logger.debug(f"Finished freehand drawing with {len(self.points)} points")
    
    def draw_preview(self, painter, image_data=None):
        """
        Draw the current contour preview.
        
        Args:
            painter: QPainter object
            image_data: Optional image data for reference
        """
        if not self.is_drawing or not self.points:
            return
        
        # Set up painter
        pen = QPen(Qt.green)
        pen.setWidth(self.line_width)
        painter.setPen(pen)
        
        # Draw line segments between adjacent points
        for i in range(1, len(self.points)):
            x1, y1 = self.points[i-1]
            x2, y2 = self.points[i]
            painter.drawLine(x1, y1, x2, y2)
    
    def get_cursor(self):
        """
        Get the cursor for this tool.
        
        Returns:
            QCursor: The cursor for the freehand tool
        """
        return self._cursor
    
    def set_line_width(self, width):
        """
        Set the line width for drawing.
        
        Args:
            width: Line width in pixels
        """
        self.line_width = max(1, int(width))
    
    def set_smoothing(self, enabled):
        """
        Enable or disable contour smoothing.
        
        Args:
            enabled: Whether smoothing is enabled
        """
        self.smoothing_enabled = enabled
    
    def set_smoothing_level(self, level):
        """
        Set the smoothing level.
        
        Args:
            level: Smoothing level (1-10)
        """
        self.smoothing_level = max(1, min(10, level))
    
    def get_contour(self):
        """
        Get the finalized contour.
        
        Returns:
            The contour as a numpy array of shape (N, 2)
        """
        return self.current_contour
    
    def _process_contour(self):
        """Process the collected points to create a finalized contour."""
        # Convert to numpy array for easier processing
        if not self.points:
            return
        
        points_array = np.array(self.points)
        
        # Close the contour if it's not already closed
        first_point = points_array[0]
        last_point = points_array[-1]
        
        # If the first and last points are not the same, add the first point at the end
        if not np.array_equal(first_point, last_point):
            points_array = np.vstack([points_array, first_point])
        
        # Apply smoothing if enabled
        if self.smoothing_enabled and len(points_array) > 3:
            points_array = self._smooth_contour(points_array)
        
        # Store processed contour
        self.current_contour = points_array
    
    def _smooth_contour(self, points_array):
        """
        Apply smoothing to the contour.
        
        Args:
            points_array: Numpy array of shape (N, 2) with contour points
            
        Returns:
            Smoothed contour as numpy array
        """
        # Simple moving average smoothing with window size based on smoothing level
        window_size = max(3, self.smoothing_level * 2 + 1)
        
        # If we don't have enough points for the window, reduce it
        if len(points_array) < window_size:
            window_size = max(3, len(points_array) - 1)
            if window_size % 2 == 0:
                window_size -= 1
        
        # Apply moving average smoothing
        half_window = window_size // 2
        smoothed_points = np.zeros_like(points_array)
        
        # Handle the first and last points (to keep the contour closed)
        for i in range(len(points_array)):
            # Create indices array with wrapping around the contour
            indices = [(i + j) % len(points_array) for j in range(-half_window, half_window + 1)]
            # Average the points
            smoothed_points[i] = np.mean(points_array[indices], axis=0)
        
        return smoothed_points
    
    def _create_cursor(self):
        """
        Create a custom cursor for the freehand tool.
        
        Returns:
            QCursor: Custom cursor for the freehand tool
        """
        # Create a pixmap for the cursor
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        
        # Draw cursor shape
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw circle
        pen = QPen(Qt.black)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(0, 255, 0, 100)))
        painter.drawEllipse(2, 2, 20, 20)
        
        # Draw crosshair
        painter.setPen(QPen(Qt.black, 1))
        painter.drawLine(12, 0, 12, 24)  # Vertical line
        painter.drawLine(0, 12, 24, 12)  # Horizontal line
        
        painter.end()
        
        # Create cursor with hotspot at center
        return QCursor(pixmap, 12, 12)


class FreehandToolWidget(QWidget):
    """
    Widget for controlling the freehand contouring tool.
    
    This widget provides UI controls for configuring the freehand drawing
    tool parameters like line width and smoothing.
    """
    
    # Signals
    contour_created = pyqtSignal(object)  # Emits contour when created
    
    def __init__(self, parent=None):
        """Initialize the freehand tool widget."""
        super().__init__(parent)
        
        # Create the tool
        self.tool = FreehandTool()
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Line width control
        width_group = QGroupBox("Line Width")
        width_layout = QHBoxLayout(width_group)
        
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(1, 10)
        self.width_slider.setValue(self.tool.line_width)
        self.width_slider.setTickInterval(1)
        self.width_slider.setTickPosition(QSlider.TicksBelow)
        
        self.width_label = QLabel(f"{self.tool.line_width} px")
        
        width_layout.addWidget(self.width_slider)
        width_layout.addWidget(self.width_label)
        
        # Smoothing controls
        smoothing_group = QGroupBox("Smoothing")
        smoothing_layout = QVBoxLayout(smoothing_group)
        
        self.smoothing_checkbox = QCheckBox("Enable Smoothing")
        self.smoothing_checkbox.setChecked(self.tool.smoothing_enabled)
        
        smoothing_level_layout = QHBoxLayout()
        smoothing_level_layout.addWidget(QLabel("Level:"))
        
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(1, 10)
        self.smoothing_slider.setValue(self.tool.smoothing_level)
        self.smoothing_slider.setTickInterval(1)
        self.smoothing_slider.setTickPosition(QSlider.TicksBelow)
        self.smoothing_slider.setEnabled(self.tool.smoothing_enabled)
        
        self.smoothing_label = QLabel(f"{self.tool.smoothing_level}")
        
        smoothing_level_layout.addWidget(self.smoothing_slider)
        smoothing_level_layout.addWidget(self.smoothing_label)
        
        smoothing_layout.addWidget(self.smoothing_checkbox)
        smoothing_layout.addLayout(smoothing_level_layout)
        
        # Add all controls to main layout
        layout.addWidget(width_group)
        layout.addWidget(smoothing_group)
        layout.addStretch(1)  # Push everything to the top
        
        # Connect signals
        self.width_slider.valueChanged.connect(self._on_width_changed)
        self.smoothing_checkbox.toggled.connect(self._on_smoothing_toggled)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_level_changed)
    
    def on_mouse_press(self, event, image_data=None):
        """
        Handle mouse press event.
        
        Args:
            event: Mouse event
            image_data: Image data
        """
        self.tool.on_mouse_press(event, image_data)
    
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
        
        # Get the contour and emit signal
        contour = self.tool.get_contour()
        if contour is not None:
            self.contour_created.emit(contour)
    
    def _on_width_changed(self, value):
        """
        Handle line width slider change.
        
        Args:
            value: New line width value
        """
        self.width_label.setText(f"{value} px")
        self.tool.set_line_width(value)
    
    def _on_smoothing_toggled(self, checked):
        """
        Handle smoothing checkbox toggle.
        
        Args:
            checked: Whether smoothing is enabled
        """
        self.smoothing_slider.setEnabled(checked)
        self.tool.set_smoothing(checked)
    
    def _on_smoothing_level_changed(self, value):
        """
        Handle smoothing level slider change.
        
        Args:
            value: New smoothing level value
        """
        self.smoothing_label.setText(f"{value}")
        self.tool.set_smoothing_level(value)
    
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
    
    widget = FreehandToolWidget()
    widget.setWindowTitle("Freehand Tool Test")
    widget.resize(300, 200)
    widget.show()
    
    sys.exit(app.exec_()) 