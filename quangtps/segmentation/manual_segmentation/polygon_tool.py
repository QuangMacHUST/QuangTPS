#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polygon Contour Tool Module
==========================

This module provides a polygon-based contouring tool for creating and
editing contours in radiotherapy treatment planning.
"""

import os
import logging
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QCheckBox, QSpinBox,
    QDoubleSpinBox, QGroupBox, QFormLayout, QFrame, QSizePolicy
)
from PyQt5.QtGui import (
    QColor, QIcon, QPixmap, QPainter, QPen, QBrush, QCursor,
    QPainterPath, QPolygon
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QPointF

logger = logging.getLogger(__name__)

class PolygonMode(Enum):
    """Enum for the different modes of the polygon tool."""
    CREATE = 1  # Create a new polygon
    EDIT = 2    # Edit an existing polygon
    DELETE = 3  # Delete points from a polygon

class PolygonContourTool:
    """
    Tool for creating and editing contours using polygons.
    
    This class provides functionality for creating and editing structure
    contours using polygon-based drawing.
    """
    
    def __init__(self, mode=PolygonMode.CREATE):
        """Initialize polygon contour tool with specified mode."""
        # Tool settings
        self.mode = mode
        self.close_threshold = 10  # pixels
        self.snap_to_points = True
        self.snap_threshold = 10   # pixels
        
        # State variables
        self.points = []
        self.temp_point = None
        self.is_drawing = False
        self.is_closed = False
        self.structure = None
        self.slice_index = None
        self.orientation = None
        self.image_shape = (512, 512)  # Default shape
        
        # Create cursors
        self._update_cursor()
    
    def start_polygon(self, point, slice_index, orientation):
        """Start creating a new polygon at the specified point."""
        self.points = [point]
        self.temp_point = None
        self.is_drawing = True
        self.is_closed = False
        self.slice_index = slice_index
        self.orientation = orientation
        
        return True
    
    def add_point(self, point):
        """Add a point to the polygon."""
        if not self.is_drawing:
            return False
        
        # Check if we're closing the polygon
        if len(self.points) > 2 and self._is_near_first_point(point):
            self.is_closed = True
            self.is_drawing = False
            
            # Create contour from points if structure is set
            if self.structure is not None:
                # Convert points to numpy array
                points_array = np.array(self.points)
                
                # Add contour to structure
                self.structure.add_contour(
                    points_array, self.slice_index, self.orientation
                )
            
            # Clear points
            self.points = []
            self.temp_point = None
            
            return True
        
        # Add point to polygon
        self.points.append(point)
        
        return True
    
    def move_temp_point(self, point):
        """Update the temporary point for preview."""
        if not self.is_drawing:
            return False
        
        self.temp_point = point
        
        return True
    
    def cancel_polygon(self):
        """Cancel the current polygon drawing."""
        self.points = []
        self.temp_point = None
        self.is_drawing = False
        self.is_closed = False
        
        return True
    
    def set_mode(self, mode):
        """Set the polygon tool mode."""
        self.mode = mode
        self._update_cursor()
    
    def set_snap_to_points(self, enabled):
        """Enable or disable snapping to existing points."""
        self.snap_to_points = enabled
    
    def set_image_shape(self, shape):
        """Set the image shape for drawing."""
        self.image_shape = shape
    
    def get_cursor(self):
        """Get the cursor for the current mode."""
        return self.cursor
    
    def get_preview_points(self):
        """Get the polygon points for preview display."""
        preview_points = self.points.copy()
        
        if self.is_drawing and self.temp_point is not None:
            preview_points.append(self.temp_point)
        
        return preview_points
    
    def _is_near_first_point(self, point):
        """Check if a point is near the first point of the polygon."""
        if not self.points:
            return False
        
        first_point = self.points[0]
        
        # Calculate distance
        dx = point[0] - first_point[0]
        dy = point[1] - first_point[1]
        distance = np.sqrt(dx * dx + dy * dy)
        
        return distance <= self.close_threshold
    
    def _update_cursor(self):
        """Update the cursor based on the current mode."""
        if self.mode == PolygonMode.CREATE:
            # Create a crosshair cursor
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            
            # Draw crosshair
            painter.drawLine(16, 0, 16, 32)
            painter.drawLine(0, 16, 32, 16)
            
            # Draw small circle at center
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(14, 14, 4, 4)
            
            painter.end()
            
            self.cursor = QCursor(pixmap, 16, 16)
        
        elif self.mode == PolygonMode.EDIT:
            # Create a pencil cursor
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            
            # Draw pencil icon
            pen = QPen(QColor(0, 0, 0), 1)
            painter.setPen(pen)
            
            # Draw pencil body
            points = [
                QPoint(10, 22),
                QPoint(18, 14),
                QPoint(22, 18),
                QPoint(14, 26)
            ]
            painter.setBrush(QBrush(QColor(200, 200, 200)))
            painter.drawPolygon(QPolygon(points))
            
            # Draw pencil tip
            points = [
                QPoint(8, 24),
                QPoint(10, 22),
                QPoint(14, 26),
                QPoint(12, 28)
            ]
            painter.setBrush(QBrush(QColor(50, 50, 50)))
            painter.drawPolygon(QPolygon(points))
            
            painter.end()
            
            self.cursor = QCursor(pixmap, 8, 24)
        
        elif self.mode == PolygonMode.DELETE:
            # Create an eraser cursor
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            
            # Draw eraser icon
            pen = QPen(QColor(0, 0, 0), 1)
            painter.setPen(pen)
            
            # Draw eraser body
            painter.setBrush(QBrush(QColor(220, 220, 220)))
            painter.drawRect(8, 12, 16, 12)
            
            # Draw eraser top
            painter.setBrush(QBrush(QColor(240, 128, 128)))
            painter.drawRect(12, 8, 8, 4)
            
            painter.end()
            
            self.cursor = QCursor(pixmap, 16, 16)
        
        else:
            # Default cursor
            self.cursor = QCursor(Qt.ArrowCursor)

class PolygonContourToolWidget(QWidget):
    """
    Widget for configuring polygon contouring tool.
    
    This class provides a UI for configuring the polygon contouring tool,
    allowing the user to select different modes and options.
    """
    
    # Signals
    toolChanged = pyqtSignal(dict)  # Emitted when tool settings change
    
    def __init__(self, parent=None):
        """Initialize polygon tool widget."""
        super().__init__(parent)
        
        # Initialize tool options
        self.mode = PolygonMode.CREATE
        self.snap_to_points = True
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create mode selection group
        mode_group = QGroupBox("Polygon Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        # Create mode radio buttons
        self.create_radio = QRadioButton("Create Polygon")
        self.create_radio.setChecked(True)
        self.create_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.create_radio)
        
        self.edit_radio = QRadioButton("Edit Points")
        self.edit_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.edit_radio)
        
        self.delete_radio = QRadioButton("Delete Points")
        self.delete_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.delete_radio)
        
        # Group radio buttons
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.create_radio, PolygonMode.CREATE.value)
        self.mode_group.addButton(self.edit_radio, PolygonMode.EDIT.value)
        self.mode_group.addButton(self.delete_radio, PolygonMode.DELETE.value)
        
        # Add mode group to layout
        layout.addWidget(mode_group)
        
        # Create options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        # Create options
        self.snap_checkbox = QCheckBox("Snap to Points")
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.toggled.connect(self._on_snap_changed)
        options_layout.addWidget(self.snap_checkbox)
        
        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        options_layout.addWidget(separator)
        
        # Add instructions label
        instructions = QLabel(
            "Click to create points.\n"
            "Click near first point to close polygon.\n"
            "Press ESC to cancel."
        )
        instructions.setStyleSheet("color: #606060; font-size: 9pt;")
        options_layout.addWidget(instructions)
        
        # Add options group to layout
        layout.addWidget(options_group)
        
        # Add help button
        help_button = QPushButton("Polygon Tool Help")
        help_button.clicked.connect(self._show_help)
        layout.addWidget(help_button)
        
        # Add a stretch to push everything to the top
        layout.addStretch(1)
        
        # Apply styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 8px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 3px;
                background-color: #f0f0f0;
            }
            
            QRadioButton, QCheckBox {
                padding: 2px;
            }
            
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px;
            }
            
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            
            QFrame[frameShape="4"] {
                color: #cccccc;
                margin: 5px 0;
            }
        """)
    
    def set_mode(self, mode):
        """Set the polygon tool mode."""
        if mode == PolygonMode.CREATE:
            self.create_radio.setChecked(True)
        elif mode == PolygonMode.EDIT:
            self.edit_radio.setChecked(True)
        elif mode == PolygonMode.DELETE:
            self.delete_radio.setChecked(True)
    
    def get_options(self):
        """Get the current polygon tool options."""
        return {
            'mode': self.mode,
            'snap_to_points': self.snap_to_points
        }
    
    def _on_mode_changed(self):
        """Handle mode radio button changes."""
        if self.create_radio.isChecked():
            self.mode = PolygonMode.CREATE
        elif self.edit_radio.isChecked():
            self.mode = PolygonMode.EDIT
        elif self.delete_radio.isChecked():
            self.mode = PolygonMode.DELETE
        
        # Emit signal
        self.toolChanged.emit(self.get_options())
    
    def _on_snap_changed(self, checked):
        """Handle snap to points checkbox changes."""
        self.snap_to_points = checked
        
        # Emit signal
        self.toolChanged.emit(self.get_options())
    
    def _show_help(self):
        """Show help information for the polygon tool."""
        from PyQt5.QtWidgets import QMessageBox
        
        QMessageBox.information(
            self,
            "Polygon Tool Help",
            """
            <b>Polygon Contouring Tool</b>
            <p>This tool allows you to create structure contours using polygons.</p>
            
            <b>Create Mode:</b>
            <ul>
                <li>Click to add points to the polygon</li>
                <li>Click near the first point to close the polygon</li>
                <li>Press ESC to cancel the current polygon</li>
            </ul>
            
            <b>Edit Mode:</b>
            <ul>
                <li>Click and drag to move existing points</li>
                <li>Double-click to add a new point between two existing points</li>
            </ul>
            
            <b>Delete Mode:</b>
            <ul>
                <li>Click on a point to remove it from the polygon</li>
            </ul>
            
            <p>Use the "Snap to Points" option to snap points to existing contours.</p>
            """
        )

def test_polygon_tool():
    """Test function for the polygon contour tool widget."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    widget = PolygonContourToolWidget()
    widget.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_polygon_tool() 