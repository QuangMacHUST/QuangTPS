#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Freehand Contouring Tool Module
===============================

This module provides Eclipse-like freehand contouring tools for QuangTPS.
"""

import numpy as np
import logging
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Union

from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF, QSize
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QComboBox, QGroupBox, QRadioButton,
    QButtonGroup, QSpinBox, QFrame, QGridLayout
)
from PyQt5.QtGui import QIcon, QColor, QCursor, QPainter, QPixmap, QPen, QBrush

logger = logging.getLogger(__name__)

class FreehandMode(Enum):
    """Enum for the different modes of the freehand tool."""
    PENCIL = 1  # Draw with a single-pixel pencil
    BRUSH = 2   # Draw with a round brush of variable size
    ERASER = 3  # Erase with a round brush of variable size

class FreehandContourTool:
    """
    Tool for freehand contouring.
    
    This class implements freehand drawing tools for creating and editing
    contours, similar to Eclipse's brush and pencil tools.
    """
    
    def __init__(self, mode=FreehandMode.BRUSH, brush_size=5):
        """Initialize the freehand contouring tool."""
        self.mode = mode
        self.brush_size = brush_size
        self.points = []
        self.is_drawing = False
        self.slice_index = None
        self.orientation = None
        self.structure = None
        self.image_shape = None
        self.temp_mask = None
        self.brush_preview_opacity = 0.5
        self.brush_color = QColor(255, 0, 0, 200)  # Red with alpha
        self.eraser_color = QColor(0, 255, 255, 200)  # Cyan with alpha
        
        # Create a cursor image for the brush
        self._update_cursor()
    
    def start_drawing(self, point, slice_index, orientation):
        """Start drawing at the specified point."""
        self.is_drawing = True
        self.points = [(point[0], point[1])]
        self.slice_index = slice_index
        self.orientation = orientation
        
        # Create a temporary mask for drawing preview
        if self.image_shape is not None:
            self.temp_mask = np.zeros(self.image_shape, dtype=np.uint8)
            
            # Draw the first point
            self._draw_point(point)
            
        return self.points
    
    def continue_drawing(self, point):
        """Continue drawing to the specified point."""
        if not self.is_drawing:
            return None
            
        # Draw line from last point to current point
        last_point = self.points[-1]
        self._draw_line(last_point, point)
        
        # Add the new point
        self.points.append((point[0], point[1]))
        
        return self.points
    
    def stop_drawing(self):
        """Stop drawing and return the final contour points."""
        self.is_drawing = False
        
        # If we have a structure and enough points, apply the changes
        if self.structure is not None and len(self.points) > 0:
            if self.mode == FreehandMode.PENCIL:
                # For pencil mode, add the points directly to the structure
                if self.temp_mask is not None:
                    if self.mode == FreehandMode.ERASER:
                        self.structure.remove_points_from_mask(self.temp_mask, 
                                                            self.slice_index, 
                                                            self.orientation)
                    else:
                        self.structure.add_points_to_mask(self.temp_mask, 
                                                       self.slice_index, 
                                                       self.orientation)
            elif self.mode == FreehandMode.BRUSH or self.mode == FreehandMode.ERASER:
                # For brush mode, update the mask in the structure
                if self.temp_mask is not None:
                    if self.mode == FreehandMode.ERASER:
                        self.structure.remove_points_from_mask(self.temp_mask, 
                                                            self.slice_index, 
                                                            self.orientation)
                    else:
                        self.structure.add_points_to_mask(self.temp_mask, 
                                                       self.slice_index, 
                                                       self.orientation)
        
        final_points = self.points
        self.points = []
        self.temp_mask = None
        
        return final_points, self.slice_index, self.orientation
    
    def set_mode(self, mode):
        """Set the drawing mode."""
        if self.mode != mode:
            self.mode = mode
            self._update_cursor()
    
    def set_brush_size(self, size):
        """Set the brush size."""
        if self.brush_size != size:
            self.brush_size = size
            self._update_cursor()
    
    def set_image_shape(self, shape):
        """Set the image shape for the mask."""
        if shape is not None and len(shape) == 2:
            self.image_shape = shape
            if self.is_drawing:
                # Resize the temp mask if we're currently drawing
                self.temp_mask = np.zeros(shape, dtype=np.uint8)
                
                # Redraw all points
                for i in range(len(self.points) - 1):
                    self._draw_line(self.points[i], self.points[i+1])
    
    def get_cursor(self):
        """Get the cursor for the current tool."""
        if hasattr(self, 'cursor'):
            return self.cursor
        else:
            # Default cursor
            return QCursor(Qt.CrossCursor)
    
    def get_temp_mask(self):
        """Get the temporary mask for preview during drawing."""
        return self.temp_mask
    
    def _update_cursor(self):
        """Update the cursor based on the current mode and brush size."""
        if self.mode == FreehandMode.PENCIL:
            # Use a simple cross cursor for pencil
            self.cursor = QCursor(Qt.CrossCursor)
        elif self.mode == FreehandMode.BRUSH or self.mode == FreehandMode.ERASER:
            # Create a custom cursor with a circle to represent brush size
            size = max(16, self.brush_size * 2 + 4)
            cursor_pixmap = QPixmap(size, size)
            cursor_pixmap.fill(Qt.transparent)
            
            painter = QPainter(cursor_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw circle representing brush size
            if self.mode == FreehandMode.BRUSH:
                painter.setPen(QPen(QColor(255, 0, 0), 1))
                painter.setBrush(QBrush(QColor(255, 0, 0, 64)))
            else:  # ERASER
                painter.setPen(QPen(QColor(0, 255, 255), 1))
                painter.setBrush(QBrush(QColor(0, 255, 255, 64)))
                
            painter.drawEllipse(size // 2 - self.brush_size, 
                              size // 2 - self.brush_size,
                              self.brush_size * 2, 
                              self.brush_size * 2)
            
            # Draw crosshair at center
            painter.setPen(QPen(Qt.black, 1))
            painter.drawLine(size // 2, 0, size // 2, size)
            painter.drawLine(0, size // 2, size, size // 2)
            
            painter.end()
            
            self.cursor = QCursor(cursor_pixmap, size // 2, size // 2)
    
    def _draw_point(self, point):
        """Draw a single point on the temporary mask."""
        if self.temp_mask is None or self.image_shape is None:
            return
            
        x, y = int(point[0]), int(point[1])
        
        # Ensure point is within bounds
        if x < 0 or x >= self.image_shape[1] or y < 0 or y >= self.image_shape[0]:
            return
        
        if self.mode == FreehandMode.PENCIL:
            # Draw a single pixel
            self.temp_mask[y, x] = 1
        elif self.mode == FreehandMode.BRUSH or self.mode == FreehandMode.ERASER:
            # Draw a circle with radius equal to brush size
            for dy in range(-self.brush_size, self.brush_size + 1):
                for dx in range(-self.brush_size, self.brush_size + 1):
                    # Check if within circle
                    if dx*dx + dy*dy <= self.brush_size*self.brush_size:
                        px, py = x + dx, y + dy
                        # Check if within bounds
                        if 0 <= px < self.image_shape[1] and 0 <= py < self.image_shape[0]:
                            self.temp_mask[py, px] = 1
    
    def _draw_line(self, point1, point2):
        """Draw a line between two points on the temporary mask."""
        if self.temp_mask is None or self.image_shape is None:
            return
            
        x1, y1 = int(point1[0]), int(point1[1])
        x2, y2 = int(point2[0]), int(point2[1])
        
        # Use Bresenham's algorithm to draw a line
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            # Draw at current point
            self._draw_point((x1, y1))
            
            if x1 == x2 and y1 == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

class FreehandToolWidget(QWidget):
    """
    Widget for configuring freehand contouring tool.
    
    This class provides a UI for configuring the freehand contouring tool,
    allowing the user to select between pencil, brush, and eraser modes
    and adjust the brush size.
    """
    
    # Signals
    toolChanged = pyqtSignal(dict)  # Emitted when tool settings change
    
    def __init__(self, parent=None):
        """Initialize the freehand tool widget."""
        super().__init__(parent)
        
        # Setup UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Tool mode selection
        mode_group = QGroupBox("Tool Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_buttons = QButtonGroup(self)
        
        self.pencil_btn = QRadioButton("Pencil")
        self.mode_buttons.addButton(self.pencil_btn, FreehandMode.PENCIL.value)
        mode_layout.addWidget(self.pencil_btn)
        
        self.brush_btn = QRadioButton("Brush")
        self.brush_btn.setChecked(True)
        self.mode_buttons.addButton(self.brush_btn, FreehandMode.BRUSH.value)
        mode_layout.addWidget(self.brush_btn)
        
        self.eraser_btn = QRadioButton("Eraser")
        self.mode_buttons.addButton(self.eraser_btn, FreehandMode.ERASER.value)
        mode_layout.addWidget(self.eraser_btn)
        
        self.mode_buttons.buttonClicked.connect(self._on_mode_changed)
        
        main_layout.addWidget(mode_group)
        
        # Brush size
        brush_group = QGroupBox("Brush Size")
        brush_layout = QVBoxLayout(brush_group)
        
        # Brush preview
        self.brush_preview = QFrame()
        self.brush_preview.setFixedSize(50, 50)
        self.brush_preview.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.brush_preview.paintEvent = self._paint_brush_preview
        brush_layout.addWidget(self.brush_preview, 0, Qt.AlignCenter)
        
        # Size slider
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 25)
        self.size_slider.setValue(5)
        self.size_slider.valueChanged.connect(self.on_brush_size_changed)
        size_layout.addWidget(self.size_slider)
        
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 25)
        self.size_spin.setValue(5)
        self.size_spin.valueChanged.connect(self.size_slider.setValue)
        size_layout.addWidget(self.size_spin)
        
        brush_layout.addLayout(size_layout)
        
        main_layout.addWidget(brush_group)
        
        # Add stretch to push everything to the top
        main_layout.addStretch(1)
        
        # Apply Eclipse-like styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QRadioButton {
                spacing: 5px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #cccccc;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #2070c0;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        
    def set_mode(self, mode):
        """Set the drawing mode."""
        if mode == FreehandMode.PENCIL:
            self.pencil_btn.setChecked(True)
        elif mode == FreehandMode.BRUSH:
            self.brush_btn.setChecked(True)
        elif mode == FreehandMode.ERASER:
            self.eraser_btn.setChecked(True)
            
        self._update_brush_preview()
        self._on_tool_changed()
    
    def on_brush_size_changed(self, size):
        """Handle brush size changes."""
        if self.size_spin.value() != size:
            self.size_spin.setValue(size)
            
        self._update_brush_preview()
        self._on_tool_changed()
    
    def get_options(self):
        """Get the current tool options."""
        # Get the selected mode
        mode_id = self.mode_buttons.checkedId()
        mode = FreehandMode(mode_id) if mode_id > 0 else FreehandMode.BRUSH
        
        # Compile options
        options = {
            'mode': mode,
            'brush_size': self.size_slider.value(),
        }
        
        return options
    
    def _paint_brush_preview(self, event):
        """Paint the brush preview area."""
        painter = QPainter(self.brush_preview)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw black background
        painter.fillRect(event.rect(), Qt.black)
        
        # Get the selected mode
        mode_id = self.mode_buttons.checkedId()
        mode = FreehandMode(mode_id) if mode_id > 0 else FreehandMode.BRUSH
        
        # Draw brush circle
        brush_size = self.size_slider.value()
        center_x = self.brush_preview.width() / 2
        center_y = self.brush_preview.height() / 2
        
        if mode == FreehandMode.PENCIL:
            # Draw a small dot for pencil
            painter.setPen(QPen(Qt.red, 1))
            painter.setBrush(QBrush(Qt.red))
            painter.drawEllipse(int(center_x - 1), int(center_y - 1), 2, 2)
        elif mode == FreehandMode.BRUSH:
            # Draw a red circle for brush
            painter.setPen(QPen(Qt.red, 1))
            painter.setBrush(QBrush(QColor(255, 0, 0, 128)))
            painter.drawEllipse(int(center_x - brush_size), int(center_y - brush_size), 
                              brush_size * 2, brush_size * 2)
        elif mode == FreehandMode.ERASER:
            # Draw a cyan circle for eraser
            painter.setPen(QPen(QColor(0, 255, 255), 1))
            painter.setBrush(QBrush(QColor(0, 255, 255, 128)))
            painter.drawEllipse(int(center_x - brush_size), int(center_y - brush_size), 
                              brush_size * 2, brush_size * 2)
                              
        painter.end()
    
    def _update_brush_preview(self):
        """Update the brush preview."""
        if hasattr(self, 'brush_preview'):
            self.brush_preview.update()
    
    def _on_mode_changed(self):
        """Handle mode selection change."""
        # Update the brush preview
        self._update_brush_preview()
        
        # Update controls based on mode
        mode_id = self.mode_buttons.checkedId()
        if mode_id > 0:
            mode = FreehandMode(mode_id)
            
            # Enable/disable brush size controls
            if mode == FreehandMode.PENCIL:
                self.size_slider.setEnabled(False)
                self.size_spin.setEnabled(False)
            else:
                self.size_slider.setEnabled(True)
                self.size_spin.setEnabled(True)
        
        self._on_tool_changed()
    
    def _on_tool_changed(self):
        """Handle tool changes and emit signal."""
        options = self.get_options()
        self.toolChanged.emit(options)

def test_freehand_tool():
    """Test function for the freehand tool."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    class TestStructure:
        def __init__(self):
            self.masks = {}
        
        def add_points_to_mask(self, points, slice_index, orientation):
            key = (slice_index, orientation)
            if key not in self.masks:
                self.masks[key] = np.zeros((100, 100), dtype=np.uint8)
            self.masks[key] |= points
            
        def remove_points_from_mask(self, points, slice_index, orientation):
            key = (slice_index, orientation)
            if key in self.masks:
                self.masks[key] &= ~points
            
        def add_contour(self, points, slice_index, orientation):
            print(f"Added contour with {len(points)} points at slice {slice_index} ({orientation})")
    
    class TestView(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFixedSize(500, 500)
            self.freehand_tool = FreehandContourTool()
            self.freehand_tool.set_image_shape((500, 500))
            self.freehand_tool.structure = TestStructure()
            
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.black)
            
            # Draw the temporary mask
            mask = self.freehand_tool.get_temp_mask()
            if mask is not None:
                for y in range(mask.shape[0]):
                    for x in range(mask.shape[1]):
                        if mask[y, x]:
                            painter.setPen(QPen(Qt.red, 1))
                            painter.drawPoint(x, y)
            
        def mousePressEvent(self, event):
            self.freehand_tool.start_drawing((event.x(), event.y()), 0, "axial")
            self.update()
            
        def mouseMoveEvent(self, event):
            self.freehand_tool.continue_drawing((event.x(), event.y()))
            self.update()
            
        def mouseReleaseEvent(self, event):
            self.freehand_tool.stop_drawing()
            self.update()
    
    app = QApplication(sys.argv)
    
    # Create the test view
    view = TestView()
    view.show()
    
    # Create the tool widget
    widget = FreehandToolWidget()
    widget.toolChanged.connect(lambda options: set_tool_options(view.freehand_tool, options))
    widget.show()
    
    def set_tool_options(tool, options):
        tool.set_mode(options['mode'])
        tool.set_brush_size(options['brush_size'])
        view.setCursor(tool.get_cursor())
    
    return app.exec_()

if __name__ == "__main__":
    test_freehand_tool() 