"""
Eclipse-style contouring tools for QuangTPS.

This module provides advanced contouring tools for structure delineation,
similar to those found in Eclipse treatment planning system.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSplitter, QSlider, QFrame, QGridLayout, QSpinBox, QSizePolicy,
    QToolBar, QAction, QComboBox, QCheckBox, QToolButton, QMenu,
    QGroupBox, QRadioButton, QButtonGroup, QDoubleSpinBox
)
from PyQt5.QtGui import (
    QColor, QImage, QPixmap, QPainter, QPen, QIcon, QCursor,
    QPolygon, QBrush, QPainterPath, QMouseEvent
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QPointF, QRectF

logger = logging.getLogger(__name__)

class ContourTool:
    """Base class for all contouring tools."""
    
    def __init__(self, name, icon=None):
        self.name = name
        self.icon = icon
        self.parameters = {}
        self.active = False
    
    def get_cursor(self):
        """Return the cursor to use when this tool is active."""
        return Qt.CrossCursor
    
    def get_parameters_widget(self):
        """Return a widget for tool parameters."""
        return None
    
    def set_parameter(self, key, value):
        """Set a tool parameter."""
        self.parameters[key] = value
    
    def begin_interaction(self, position, image_data=None):
        """Start interaction at the given position."""
        pass
    
    def update_interaction(self, position, image_data=None):
        """Update interaction at the given position."""
        pass
    
    def end_interaction(self, position, image_data=None):
        """End interaction at the given position."""
        return None
    
    def draw_preview(self, painter, image_data=None):
        """Draw a preview of the current interaction."""
        pass

class BrushTool(ContourTool):
    """Công cụ cọ vẽ tự do."""
    
    def __init__(self):
        super().__init__("Brush", QIcon("quangtps/ui/icons/new_icons/brush.png"))
        self.parameters = {
            "radius": 5,
            "hardness": 1.0,  # 0.0-1.0, where 1.0 is fully hard
            "erase_mode": False
        }
        self.path = QPainterPath()
        self.last_point = None
        self.current_mask = None
        self.temp_mask = None
    
    def get_cursor(self):
        """Return a custom cursor for the brush tool."""
        # In a real implementation, create a circular cursor
        # with size based on brush radius
        return Qt.CrossCursor
    
    def get_parameters_widget(self):
        """Return widget for brush parameters."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Brush size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        size_spin = QSpinBox()
        size_spin.setRange(1, 100)
        size_spin.setValue(self.parameters["radius"])
        size_spin.valueChanged.connect(lambda value: self.set_parameter("radius", value))
        size_layout.addWidget(size_spin)
        layout.addLayout(size_layout)
        
        # Hardness slider
        hardness_layout = QHBoxLayout()
        hardness_layout.addWidget(QLabel("Hardness:"))
        hardness_slider = QSlider(Qt.Horizontal)
        hardness_slider.setRange(0, 100)
        hardness_slider.setValue(int(self.parameters["hardness"] * 100))
        hardness_slider.valueChanged.connect(lambda value: self.set_parameter("hardness", value / 100.0))
        hardness_layout.addWidget(hardness_slider)
        layout.addLayout(hardness_layout)
        
        # Erase mode
        erase_checkbox = QCheckBox("Erase Mode")
        erase_checkbox.setChecked(self.parameters["erase_mode"])
        erase_checkbox.stateChanged.connect(lambda state: self.set_parameter("erase_mode", bool(state)))
        layout.addWidget(erase_checkbox)
        
        return widget
    
    def begin_interaction(self, position, image_data=None):
        """Start brush interaction."""
        self.path = QPainterPath()
        self.path.moveTo(position)
        self.last_point = position
        
        # Create empty mask if needed
        if image_data is not None:
            h, w = image_data.shape
            if self.current_mask is None or self.current_mask.shape != (h, w):
                self.current_mask = np.zeros((h, w), dtype=bool)
            self.temp_mask = self.current_mask.copy()
            
            # Apply first brush stroke
            self._apply_brush_at(position, image_data)
    
    def update_interaction(self, position, image_data=None):
        """Update brush interaction."""
        if self.last_point is not None:
            # Add to path
            self.path.lineTo(position)
            self.last_point = position
            
            # Apply brush stroke
            if image_data is not None and self.temp_mask is not None:
                self._apply_brush_at(position, image_data)
    
    def end_interaction(self, position, image_data=None):
        """End brush interaction."""
        if image_data is not None and self.temp_mask is not None:
            # Final result is the temporary mask
            result = self.temp_mask.copy()
            
            # Reset temporary data
            self.path = QPainterPath()
            self.last_point = None
            self.temp_mask = None
            
            return result
        return None
    
    def _apply_brush_at(self, position, image_data):
        """Apply brush at the given position to the temporary mask."""
        if self.temp_mask is None:
            return
            
        # Get parameters
        radius = self.parameters["radius"]
        hardness = self.parameters["hardness"]
        erase_mode = self.parameters["erase_mode"]
        
        # Create brush mask
        h, w = image_data.shape
        y, x = int(position.y()), int(position.x())
        
        # Ensure coordinates are within bounds
        if not (0 <= x < w and 0 <= y < h):
            return
            
        # Create circular brush
        brush_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(brush_mask, (x, y), radius, 255, -1)
        
        # Apply hardness (use distance transform if hardness < 1.0)
        if hardness < 1.0:
            # Create distance field
            dist = cv2.distanceTransform((brush_mask > 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)
            max_dist = np.max(dist)
            if max_dist > 0:
                # Normalize and apply hardness
                norm_dist = dist / max_dist
                # Adjust transition based on hardness
                transition_start = 1.0 - (1.0 - hardness) * 0.9  # Adjust this for different falloff
                mask = ((norm_dist > transition_start) |
                       ((norm_dist > 0) & (np.random.random(norm_dist.shape) < 
                                           ((norm_dist - transition_start) / (1.0 - transition_start)))))
                brush_mask = mask.astype(np.uint8) * 255
        
        # Apply to temp mask
        if erase_mode:
            self.temp_mask[brush_mask > 0] = False
        else:
            self.temp_mask[brush_mask > 0] = True
    
    def draw_preview(self, painter, image_data=None):
        """Draw brush preview."""
        if self.path.isEmpty():
            return
            
        # Draw the brush path
        painter.setPen(QPen(Qt.green, 1, Qt.SolidLine))
        painter.drawPath(self.path)
        
        # Draw current brush position
        if self.last_point:
            painter.setPen(QPen(Qt.green, 1, Qt.DotLine))
            radius = self.parameters["radius"]
            painter.drawEllipse(self.last_point, radius, radius)

class PencilTool(ContourTool):
    """Công cụ bút chì để vẽ viền chính xác."""
    
    def __init__(self):
        super().__init__("Pencil", QIcon("quangtps/ui/icons/new_icons/pencil.png"))
        self.parameters = {
            "width": 1,
            "snap_to_gradient": False,
            "erase_mode": False
        }
        self.points = []
        self.temp_mask = None
    
    def get_parameters_widget(self):
        """Return widget for pencil parameters."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Line width
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        width_spin = QSpinBox()
        width_spin.setRange(1, 10)
        width_spin.setValue(self.parameters["width"])
        width_spin.valueChanged.connect(lambda value: self.set_parameter("width", value))
        width_layout.addWidget(width_spin)
        layout.addLayout(width_layout)
        
        # Snap to gradient
        snap_checkbox = QCheckBox("Snap to Edges")
        snap_checkbox.setChecked(self.parameters["snap_to_gradient"])
        snap_checkbox.stateChanged.connect(lambda state: self.set_parameter("snap_to_gradient", bool(state)))
        layout.addWidget(snap_checkbox)
        
        # Erase mode
        erase_checkbox = QCheckBox("Erase Mode")
        erase_checkbox.setChecked(self.parameters["erase_mode"])
        erase_checkbox.stateChanged.connect(lambda state: self.set_parameter("erase_mode", bool(state)))
        layout.addWidget(erase_checkbox)
        
        return widget
    
    def begin_interaction(self, position, image_data=None):
        """Start pencil interaction."""
        self.points = [position]
        
        # Create empty mask if needed
        if image_data is not None:
            h, w = image_data.shape
            if self.temp_mask is None or self.temp_mask.shape != (h, w):
                self.temp_mask = np.zeros((h, w), dtype=bool)
            
            # Apply first point
            self._update_mask(image_data)
    
    def update_interaction(self, position, image_data=None):
        """Update pencil interaction."""
        # Add point to list
        self.points.append(position)
        
        # Update mask
        if image_data is not None:
            self._update_mask(image_data)
    
    def end_interaction(self, position, image_data=None):
        """End pencil interaction."""
        if image_data is not None and self.temp_mask is not None:
            # Add final point
            self.points.append(position)
            self._update_mask(image_data)
            
            # Return the result
            result = self.temp_mask.copy()
            
            # Reset temporary data
            self.points = []
            self.temp_mask = None
            
            return result
        return None
    
    def _update_mask(self, image_data):
        """Update the temporary mask with the current points."""
        if len(self.points) < 2 or self.temp_mask is None:
            return
        
        # Convert points to pixel coordinates
        h, w = image_data.shape
        points_array = np.array([(int(p.x()), int(p.y())) for p in self.points])
        
        # Create empty mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw the line
        width = self.parameters["width"]
        for i in range(1, len(points_array)):
            cv2.line(mask, 
                    tuple(points_array[i-1]), 
                    tuple(points_array[i]), 
                    255, width)
        
        # Apply to temp mask
        if self.parameters["erase_mode"]:
            self.temp_mask[mask > 0] = False
        else:
            self.temp_mask[mask > 0] = True
    
    def draw_preview(self, painter, image_data=None):
        """Draw pencil preview."""
        if len(self.points) < 2:
            return
        
        # Set up pen
        if self.parameters["erase_mode"]:
            painter.setPen(QPen(Qt.red, self.parameters["width"], Qt.SolidLine))
        else:
            painter.setPen(QPen(Qt.green, self.parameters["width"], Qt.SolidLine))
        
        # Draw the line segments
        for i in range(1, len(self.points)):
            painter.drawLine(self.points[i-1], self.points[i])

class PolygonTool(ContourTool):
    """Công cụ vẽ đa giác."""
    
    def __init__(self):
        super().__init__("Polygon", QIcon("quangtps/ui/icons/new_icons/polygon.png"))
        self.parameters = {
            "line_width": 1,
            "close_threshold": 10  # Pixels distance to close polygon
        }
        self.points = []
        self.is_closed = False
        self.temp_mask = None
    
    def get_parameters_widget(self):
        """Return widget for polygon parameters."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Line width
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Line Width:"))
        width_spin = QSpinBox()
        width_spin.setRange(1, 5)
        width_spin.setValue(self.parameters["line_width"])
        width_spin.valueChanged.connect(lambda value: self.set_parameter("line_width", value))
        width_layout.addWidget(width_spin)
        layout.addLayout(width_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Close button
        close_button = QPushButton("Close Polygon")
        close_button.clicked.connect(self._close_polygon)
        button_layout.addWidget(close_button)
        
        # Clear button
        clear_button = QPushButton("Clear Points")
        clear_button.clicked.connect(self._clear_points)
        button_layout.addWidget(clear_button)
        
        layout.addLayout(button_layout)
        return widget
    
    def _close_polygon(self):
        """Close the polygon manually."""
        if len(self.points) > 2:
            self.is_closed = True
    
    def _clear_points(self):
        """Clear all points."""
        self.points = []
        self.is_closed = False
    
    def begin_interaction(self, position, image_data=None):
        """Start polygon interaction by adding a point."""
        # Check if near first point (to close polygon)
        if len(self.points) > 2:
            first_point = self.points[0]
            dist = np.sqrt((position.x() - first_point.x())**2 + (position.y() - first_point.y())**2)
            if dist < self.parameters["close_threshold"]:
                self.is_closed = True
                return
        
        # Add point
        self.points.append(position)
        
        # Create empty mask if needed
        if image_data is not None and self.temp_mask is None:
            h, w = image_data.shape
            self.temp_mask = np.zeros((h, w), dtype=bool)
    
    def update_interaction(self, position, image_data=None):
        """Update polygon interaction (preview only)."""
        # This is handled in draw_preview
        pass
    
    def end_interaction(self, position, image_data=None):
        """End polygon interaction."""
        if self.is_closed and image_data is not None and self.temp_mask is not None:
            # Convert points to pixel coordinates
            h, w = image_data.shape
            points_array = np.array([(int(p.x()), int(p.y())) for p in self.points])
            
            # Create a mask from the polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            # Draw filled polygon
            cv2.fillPoly(mask, [points_array], 255)
            
            # Apply to temp mask
            self.temp_mask[mask > 0] = True
            
            # Return result and reset
            result = self.temp_mask.copy()
            self.points = []
            self.is_closed = False
            self.temp_mask = None
            
            return result
        return None
    
    def draw_preview(self, painter, image_data=None):
        """Draw polygon preview."""
        if len(self.points) < 1:
            return
        
        # Set up pen and brush
        painter.setPen(QPen(Qt.green, self.parameters["line_width"], Qt.SolidLine))
        
        # Draw the polygon points and lines
        for i, point in enumerate(self.points):
            # Draw point
            painter.drawEllipse(point, 3, 3)
            
            # Draw line to previous point
            if i > 0:
                painter.drawLine(self.points[i-1], point)
        
        # Draw closing line if needed
        if len(self.points) > 2:
            if self.is_closed:
                painter.drawLine(self.points[-1], self.points[0])
            else:
                # Draw dashed line to show potential closure
                painter.setPen(QPen(Qt.green, self.parameters["line_width"], Qt.DashLine))
                painter.drawLine(self.points[-1], self.points[0])

class ThresholdTool(ContourTool):
    """Công cụ tạo contour bằng phương pháp ngưỡng."""
    
    def __init__(self):
        super().__init__("Threshold", QIcon("quangtps/ui/icons/new_icons/threshold.png"))
        self.parameters = {
            "lower_threshold": -100,
            "upper_threshold": 100,
            "connectivity": 4,  # 4 or 8 connectivity
            "apply_on_release": True
        }
        self.seed_point = None
        self.preview_mask = None
        self.temp_mask = None
    
    def get_parameters_widget(self):
        """Return widget for threshold parameters."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Lower threshold
        lower_layout = QHBoxLayout()
        lower_layout.addWidget(QLabel("Lower:"))
        lower_spin = QSpinBox()
        lower_spin.setRange(-1000, 3000)
        lower_spin.setValue(self.parameters["lower_threshold"])
        lower_spin.valueChanged.connect(lambda value: self.set_parameter("lower_threshold", value))
        lower_layout.addWidget(lower_spin)
        layout.addLayout(lower_layout)
        
        # Upper threshold
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(QLabel("Upper:"))
        upper_spin = QSpinBox()
        upper_spin.setRange(-1000, 3000)
        upper_spin.setValue(self.parameters["upper_threshold"])
        upper_spin.valueChanged.connect(lambda value: self.set_parameter("upper_threshold", value))
        upper_layout.addWidget(upper_spin)
        layout.addLayout(upper_layout)
        
        # Connectivity
        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("Connectivity:"))
        conn_group = QButtonGroup(widget)
        conn4_radio = QRadioButton("4-connected")
        conn8_radio = QRadioButton("8-connected")
        conn_group.addButton(conn4_radio)
        conn_group.addButton(conn8_radio)
        conn4_radio.setChecked(self.parameters["connectivity"] == 4)
        conn8_radio.setChecked(self.parameters["connectivity"] == 8)
        conn4_radio.toggled.connect(lambda checked: self.set_parameter("connectivity", 4 if checked else 8))
        conn_layout.addWidget(conn4_radio)
        conn_layout.addWidget(conn8_radio)
        layout.addLayout(conn_layout)
        
        # Apply on release/click
        apply_check = QCheckBox("Apply on Release")
        apply_check.setChecked(self.parameters["apply_on_release"])
        apply_check.stateChanged.connect(lambda state: self.set_parameter("apply_on_release", bool(state)))
        layout.addWidget(apply_check)
        
        # Apply button
        apply_button = QPushButton("Apply Threshold")
        apply_button.clicked.connect(lambda: self._update_preview())
        layout.addWidget(apply_button)
        
        return widget
    
    def _update_preview(self):
        """Update preview based on current thresholds."""
        if self.seed_point is None or self.temp_mask is None:
            return
            
        self.preview_mask = self._apply_threshold(self.seed_point)
    
    def begin_interaction(self, position, image_data=None):
        """Start threshold interaction by setting seed point."""
        self.seed_point = position
        
        # Create empty mask if needed
        if image_data is not None:
            h, w = image_data.shape
            self.temp_mask = np.zeros((h, w), dtype=bool)
            
            # Apply threshold if not on release
            if not self.parameters["apply_on_release"]:
                self.preview_mask = self._apply_threshold(position)
    
    def update_interaction(self, position, image_data=None):
        """Update threshold interaction (move seed point)."""
        self.seed_point = position
        
        # Apply threshold if not on release
        if not self.parameters["apply_on_release"] and image_data is not None:
            self.preview_mask = self._apply_threshold(position)
    
    def end_interaction(self, position, image_data=None):
        """End threshold interaction."""
        if image_data is not None and self.temp_mask is not None:
            # Apply threshold
            self.preview_mask = self._apply_threshold(position)
            
            # Return result if we have a preview
            if self.preview_mask is not None:
                result = self.preview_mask.copy()
                self.seed_point = None
                self.preview_mask = None
                self.temp_mask = None
                return result
        return None
    
    def _apply_threshold(self, position):
        """Apply threshold operation starting from seed point."""
        if position is None:
            return None
            
        # Get seed point coordinates
        x, y = int(position.x()), int(position.y())
        
        # Get image data (limited to our focus area)
        h, w = self.temp_mask.shape
        
        # Ensure seed point is within bounds
        if not (0 <= x < w and 0 <= y < h):
            return None
        
        # TODO: This is a placeholder - in a real implementation
        # you would use SimpleITK or OpenCV's floodFill to apply
        # region growing with thresholds
        
        # For now, just create a circular mask as an example
        mask = np.zeros((h, w), dtype=bool)
        cv2.circle(mask, (x, y), 50, True, -1)
        
        return mask
    
    def draw_preview(self, painter, image_data=None):
        """Draw threshold preview."""
        # Draw seed point
        if self.seed_point:
            painter.setPen(QPen(Qt.blue, 1, Qt.SolidLine))
            painter.drawEllipse(self.seed_point, 5, 5)
            painter.drawLine(self.seed_point.x() - 10, self.seed_point.y(),
                           self.seed_point.x() + 10, self.seed_point.y())
            painter.drawLine(self.seed_point.x(), self.seed_point.y() - 10,
                           self.seed_point.x(), self.seed_point.y() + 10)
            
            # Draw threshold values
            painter.drawText(
                self.seed_point.x() + 15, 
                self.seed_point.y() - 5,
                f"T: [{self.parameters['lower_threshold']}:{self.parameters['upper_threshold']}]"
            )

class ContouringWidget(QWidget):
    """Widget that provides Eclipse-style contouring functionality."""
    
    contour_created = pyqtSignal(object)  # Emits the contour mask when created
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup layout
        self.main_layout = QVBoxLayout(self)
        
        # Image data
        self.image_data = None
        self.structure_mask = None
        self.displayed_image = None
        self.scale_factor = 1.0
        
        # Window/level
        self.window_width = 400
        self.window_level = 40
        
        # Interaction state
        self.active_tool = None
        self.is_interacting = False
        self.overlay_mask = None
        
        # Available tools
        self.tools = {
            'brush': BrushTool(),
            'pencil': PencilTool(),
            'polygon': PolygonTool(),
            'threshold': ThresholdTool()
        }
        
        # Creating the tools panel
        self._create_tools_panel()
        
        # Set initial tool
        self.set_active_tool('brush')
        
        # Set mouse tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
    
    def _create_tools_panel(self):
        """Create the tools panel with buttons and options."""
        # Tools group
        tools_group = QGroupBox("Contouring Tools")
        tools_layout = QVBoxLayout(tools_group)
        
        # Tools buttons
        tools_buttons_layout = QHBoxLayout()
        
        for tool_name, tool in self.tools.items():
            button = QToolButton()
            button.setIcon(tool.icon)
            button.setIconSize(QSize(24, 24))
            button.setToolTip(tool.name)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, name=tool_name: self.set_active_tool(name))
            tools_buttons_layout.addWidget(button)
            tool.button = button
        
        tools_layout.addLayout(tools_buttons_layout)
        
        # Tool parameters container
        self.tool_params_container = QWidget()
        self.tool_params_layout = QVBoxLayout(self.tool_params_container)
        self.tool_params_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.addWidget(self.tool_params_container)
        
        # Add to main layout
        self.main_layout.addWidget(tools_group)
        
        # Add image display area
        self.display_area = QWidget()
        self.display_area.setMinimumSize(400, 400)
        self.display_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.display_area)
    
    def set_active_tool(self, tool_name):
        """Set the active contouring tool."""
        # Deactivate current tool
        if self.active_tool:
            self.active_tool.active = False
            self.active_tool.button.setChecked(False)
        
        # Set new active tool
        self.active_tool = self.tools.get(tool_name)
        if self.active_tool:
            self.active_tool.active = True
            self.active_tool.button.setChecked(True)
            
            # Update cursor
            self.setCursor(self.active_tool.get_cursor())
            
            # Update parameters widget
            self._update_tool_parameters()
    
    def _update_tool_parameters(self):
        """Update the tool parameters panel."""
        # Clear existing widgets
        for i in reversed(range(self.tool_params_layout.count())): 
            widget = self.tool_params_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # Add new parameters widget
        if self.active_tool:
            params_widget = self.active_tool.get_parameters_widget()
            if params_widget:
                self.tool_params_layout.addWidget(params_widget)
    
    def set_image_data(self, image_data):
        """Set the image data for contouring."""
        self.image_data = image_data
        
        # Reset structure mask to match image dimensions
        if image_data is not None:
            h, w = image_data.shape
            self.structure_mask = np.zeros((h, w), dtype=bool)
            self.overlay_mask = None
        
        self._update_display()
        self.update()
    
    def set_structure_mask(self, mask):
        """Set the current structure mask."""
        self.structure_mask = mask
        self.overlay_mask = None
        self._update_display()
        self.update()
    
    def set_window_level(self, window_width, window_level):
        """Set the window/level for image display."""
        self.window_width = max(1, window_width)
        self.window_level = window_level
        self._update_display()
        self.update()
    
    def _update_display(self):
        """Update the displayed image with current window/level and overlay."""
        if self.image_data is None:
            self.displayed_image = None
            return
            
        # Window/level processing
        low = self.window_level - self.window_width / 2
        high = self.window_level + self.window_width / 2
        
        # Clip and rescale image data to 0-255
        image = np.clip(self.image_data, low, high)
        image = 255 * (image - low) / (high - low)
        image = image.astype(np.uint8)
        
        # Create RGB image for overlay
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply structure mask overlay (green)
        if self.structure_mask is not None:
            rgb_image[self.structure_mask] = (0, 255, 0)  # Green for structure
        
        # Apply temporary overlay mask (blue)
        if self.overlay_mask is not None:
            rgb_image[self.overlay_mask] = (0, 0, 255)  # Blue for preview
        
        # Convert to QImage
        height, width = rgb_image.shape[:2]
        bytes_per_line = 3 * width
        self.displayed_image = QImage(rgb_image.data, width, height, 
                                     bytes_per_line, QImage.Format_RGB888)
    
    def paintEvent(self, event):
        """Paint the image and contouring overlays."""
        painter = QPainter(self)
        
        # Fill background
        painter.fillRect(self.rect(), Qt.black)
        
        # Draw image if available
        if self.displayed_image:
            # Calculate display rect
            image_width = self.displayed_image.width() * self.scale_factor
            image_height = self.displayed_image.height() * self.scale_factor
            
            # Center image in widget
            x = (self.width() - image_width) / 2
            y = (self.height() - image_height) / 2 
            
            # Adjust for tools panel height
            tools_height = self.display_area.y()
            y += tools_height / 2
            
            # Draw image
            draw_rect = QRectF(x, y, image_width, image_height)
            painter.drawImage(draw_rect, self.displayed_image)
            
            # Set transform for tool drawing
            painter.save()
            painter.translate(x, y)
            painter.scale(self.scale_factor, self.scale_factor)
            
            # Draw active tool preview
            if self.active_tool and self.is_interacting:
                self.active_tool.draw_preview(painter, self.image_data)
            
            painter.restore()
    
    def mousePressEvent(self, event):
        """Handle mouse press events for contouring."""
        if event.button() == Qt.LeftButton and self.active_tool and self.image_data is not None:
            # Convert to image coordinates
            pos = self._event_to_image_pos(event)
            if pos:
                self.is_interacting = True
                self.active_tool.begin_interaction(pos, self.image_data)
                self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for contouring."""
        if self.is_interacting and self.active_tool and self.image_data is not None:
            # Convert to image coordinates
            pos = self._event_to_image_pos(event)
            if pos:
                self.active_tool.update_interaction(pos, self.image_data)
                self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for contouring."""
        if event.button() == Qt.LeftButton and self.is_interacting and self.active_tool:
            # Convert to image coordinates
            pos = self._event_to_image_pos(event)
            if pos:
                # End interaction and get result
                result = self.active_tool.end_interaction(pos, self.image_data)
                
                # Apply result if available
                if result is not None:
                    self.structure_mask = result
                    self.contour_created.emit(result)
                
                self.is_interacting = False
                self.update()
    
    def _event_to_image_pos(self, event):
        """Convert mouse event position to image coordinates."""
        if self.displayed_image is None:
            return None
            
        # Calculate image display rect
        image_width = self.displayed_image.width() * self.scale_factor
        image_height = self.displayed_image.height() * self.scale_factor
        
        # Center image in widget
        x = (self.width() - image_width) / 2
        y = (self.height() - image_height) / 2
        
        # Adjust for tools panel height
        tools_height = self.display_area.y()
        y += tools_height / 2
        
        # Check if click is within image bounds
        if (x <= event.x() <= x + image_width and y <= event.y() <= y + image_height):
            # Convert to image coordinates
            image_x = (event.x() - x) / self.scale_factor
            image_y = (event.y() - y) / self.scale_factor
            
            # Create point
            return QPointF(image_x, image_y)
        
        return None
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale_factor *= 1.1  # Zoom in
        else:
            self.scale_factor /= 1.1  # Zoom out
        
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))  # Limit zoom
        self.update()

# Test function
def test():
    """Test the contouring tools with a sample image."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create sample image data
    image_size = (300, 300)
    image_data = np.zeros(image_size, dtype=np.float32)
    
    # Add a circle
    center = np.array(image_size) / 2
    radius = min(image_size) / 4
    
    for y in range(image_size[0]):
        for x in range(image_size[1]):
            dist = np.sqrt(((np.array([y, x]) - center)**2).sum())
            if dist < radius:
                image_data[y, x] = 100  # Bright circle
            else:
                image_data[y, x] = -100  # Dark background
    
    # Add some noise
    image_data += np.random.normal(0, 10, image_size)
    
    # Create and show widget
    widget = ContouringWidget()
    widget.set_image_data(image_data)
    widget.set_window_level(400, 0)
    widget.show()
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(test()) 