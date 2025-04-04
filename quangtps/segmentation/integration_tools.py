#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Segmentation Tools Integration Module
=====================================

This module integrates various segmentation tools into a cohesive Eclipse-like
interface for structure contouring in QuangTPS.
"""

import logging
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QStackedWidget, QComboBox, QPushButton, QGroupBox, 
    QFrame, QSplitter, QToolBar, QAction, QSizePolicy
)
from PyQt5.QtGui import QIcon, QPixmap, QCursor

from quangtps.segmentation.manual_segmentation.freehand_tool import (
    FreehandContourTool, FreehandMode, FreehandToolWidget
)
from quangtps.segmentation.manual_segmentation.polygon_tool import (
    PolygonContourTool, PolygonMode, PolygonContourToolWidget
)
from quangtps.segmentation.manual_segmentation.threshold_tool import (
    ThresholdContourTool, ThresholdMode, ThresholdOperation, ThresholdToolWidget
)

logger = logging.getLogger(__name__)

class SegmentationInterface(QWidget):
    """
    Integrated interface for segmentation tools.
    
    This class provides an Eclipse-like interface for structure contouring,
    integrating various tools like brush, polygon, and threshold-based segmentation.
    """
    
    # Signals
    toolChanged = pyqtSignal(str, dict)  # Tool name and options
    structureModified = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the segmentation interface."""
        super().__init__(parent)
        
        # Initialize tool instances
        self.freehand_tool = FreehandContourTool()
        self.polygon_tool = PolygonContourTool()
        self.threshold_tool = ThresholdContourTool()
        
        # Current tool
        self.current_tool = "freehand"
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the toolbar for tool selection
        self.create_toolbar()
        
        # Main content widget
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tool options panel
        options_panel = QFrame()
        options_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        options_panel.setMaximumWidth(300)
        options_panel.setFrameShape(QFrame.StyledPanel)
        options_layout = QVBoxLayout(options_panel)
        
        # Tool options title
        options_title = QLabel("Tool Options")
        options_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2070c0;")
        options_layout.addWidget(options_title)
        
        # Tool options stacked widget
        self.tool_options_container = QStackedWidget()
        
        # Create tool option widgets
        self.freehand_options = FreehandToolWidget()
        self.freehand_options.toolChanged.connect(self.on_freehand_options_changed)
        self.tool_options_container.addWidget(self.freehand_options)
        
        self.polygon_options = PolygonContourToolWidget()
        self.polygon_options.toolChanged.connect(self.on_polygon_options_changed)
        self.tool_options_container.addWidget(self.polygon_options)
        
        self.threshold_options = ThresholdToolWidget()
        self.threshold_options.toolChanged.connect(self.on_threshold_options_changed)
        self.threshold_options.applyThreshold.connect(self.on_apply_threshold)
        self.tool_options_container.addWidget(self.threshold_options)
        
        options_layout.addWidget(self.tool_options_container)
        
        # Add a stretch to push tool options to the top
        options_layout.addStretch(1)
        
        # Add options panel to content
        content_layout.addWidget(options_panel)
        
        # Add content widget to main layout
        main_layout.addWidget(content_widget)
        
        # Set default tool to freehand
        self.set_current_tool("freehand")
        
        # Apply Eclipse-like styling
        self.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QToolBar {
                background-color: #e0e0e0;
                border: none;
                spacing: 3px;
            }
            QToolBar QToolButton {
                background-color: transparent;
                border-radius: 3px;
                padding: 3px;
            }
            QToolBar QToolButton:hover {
                background-color: #d0d0d0;
            }
            QToolBar QToolButton:checked {
                background-color: #2070c0;
            }
            QToolBar QToolButton:checked:hover {
                background-color: #3080d0;
            }
        """)
    
    def create_toolbar(self):
        """Create the toolbar for tool selection."""
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.toolbar.setIconSize(QSize(24, 24))
        
        # Get icon directory
        icons_dir = os.path.join(os.path.dirname(__file__), "..", "ui", "icons")
        if not os.path.exists(icons_dir):
            icons_dir = os.path.join(os.path.dirname(__file__), "..", "resources", "icons")
        
        # Function to create a tool button
        def create_tool_action(name, icon_file, tooltip):
            icon_path = os.path.join(icons_dir, icon_file) if os.path.exists(os.path.join(icons_dir, icon_file)) else None
            action = QAction(name, self)
            if icon_path:
                action.setIcon(QIcon(icon_path))
            action.setToolTip(tooltip)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, n=name.lower(): self.set_current_tool(n))
            return action
        
        # Create tool actions
        self.select_action = create_tool_action("Select", "select.png", "Select structures")
        self.freehand_action = create_tool_action("Brush", "brush.png", "Freehand drawing with brush")
        self.polygon_action = create_tool_action("Polygon", "polygon.png", "Draw polygon contours")
        self.threshold_action = create_tool_action("Threshold", "threshold.png", "Create contours based on thresholds")
        
        # Add actions to toolbar
        self.toolbar.addAction(self.select_action)
        self.toolbar.addAction(self.freehand_action)
        self.toolbar.addAction(self.polygon_action)
        self.toolbar.addAction(self.threshold_action)
        
        # Add toolbar to layout
        self.layout().addWidget(self.toolbar)
    
    def set_current_tool(self, tool_name):
        """Set the current tool."""
        # Update current tool
        self.current_tool = tool_name
        
        # Update action states
        self.select_action.setChecked(tool_name == "select")
        self.freehand_action.setChecked(tool_name == "freehand" or tool_name == "brush")
        self.polygon_action.setChecked(tool_name == "polygon")
        self.threshold_action.setChecked(tool_name == "threshold")
        
        # Show corresponding tool options
        if tool_name == "freehand" or tool_name == "brush":
            self.tool_options_container.setCurrentWidget(self.freehand_options)
        elif tool_name == "polygon":
            self.tool_options_container.setCurrentWidget(self.polygon_options)
        elif tool_name == "threshold":
            self.tool_options_container.setCurrentWidget(self.threshold_options)
        
        # Emit tool changed signal with current options
        if tool_name == "freehand" or tool_name == "brush":
            options = self.freehand_options.get_options()
        elif tool_name == "polygon":
            options = self.polygon_options.get_options()
        elif tool_name == "threshold":
            options = self.threshold_options.get_options()
        else:
            options = {}
        
        self.toolChanged.emit(tool_name, options)
    
    def set_image_data(self, image_data):
        """Set the image data for the tools."""
        # Set image data for threshold tool
        self.threshold_tool.set_image_data(image_data)
        
        # Set image range for threshold widget
        if image_data is not None and hasattr(image_data, 'data') and image_data.data is not None:
            min_val = np.min(image_data.data)
            max_val = np.max(image_data.data)
            self.threshold_options.set_image_range(int(min_val), int(max_val))
    
    def set_slice_shape(self, shape):
        """Set the slice shape for the tools."""
        self.freehand_tool.set_image_shape(shape)
        self.polygon_tool.set_image_shape(shape)
    
    def set_structure(self, structure):
        """Set the structure for editing."""
        self.freehand_tool.structure = structure
        self.polygon_tool.structure = structure
        self.threshold_tool.structure = structure
    
    def handle_mouse_press(self, point, slice_index, orientation):
        """Handle mouse press events in the viewer."""
        if self.current_tool == "freehand" or self.current_tool == "brush":
            self.freehand_tool.start_drawing(point, slice_index, orientation)
            return True
        elif self.current_tool == "polygon":
            self.polygon_tool.start_polygon(point, slice_index, orientation)
            return True
        elif self.current_tool == "threshold":
            self.threshold_tool.set_seed_point(point, slice_index, orientation)
            return True
        return False
    
    def handle_mouse_move(self, point):
        """Handle mouse move events in the viewer."""
        if self.current_tool == "freehand" or self.current_tool == "brush":
            if self.freehand_tool.is_drawing:
                self.freehand_tool.continue_drawing(point)
                return True
        elif self.current_tool == "polygon":
            self.polygon_tool.add_point(point)
            return True
        return False
    
    def handle_mouse_release(self, point):
        """Handle mouse release events in the viewer."""
        if self.current_tool == "freehand" or self.current_tool == "brush":
            if self.freehand_tool.is_drawing:
                self.freehand_tool.stop_drawing()
                self.structureModified.emit()
                return True
        return False
    
    def get_cursor_for_viewer(self):
        """Get the cursor for the current tool."""
        if self.current_tool == "freehand" or self.current_tool == "brush":
            return self.freehand_tool.get_cursor()
        elif self.current_tool == "polygon":
            return self.polygon_tool.get_cursor()
        elif self.current_tool == "threshold":
            return self.threshold_tool.get_cursor()
        return QCursor(Qt.ArrowCursor)
    
    def get_overlay_for_viewer(self, orientation, slice_index):
        """Get the tool overlay for display in the viewer."""
        if self.current_tool == "freehand" or self.current_tool == "brush":
            # Return the temporary mask if drawing
            if self.freehand_tool.is_drawing:
                return self.freehand_tool.get_temp_mask()
        elif self.current_tool == "polygon":
            # Return polygon preview
            return None  # Would need to be implemented in the viewer
        elif self.current_tool == "threshold":
            # Return the threshold mask
            if (self.threshold_tool.orientation == orientation and 
                self.threshold_tool.slice_index == slice_index):
                return self.threshold_tool.get_preview_mask()
        return None
    
    def on_freehand_options_changed(self, options):
        """Handle changes to freehand tool options."""
        self.freehand_tool.set_mode(options['mode'])
        self.freehand_tool.set_brush_size(options['brush_size'])
        self.toolChanged.emit("freehand", options)
    
    def on_polygon_options_changed(self, options):
        """Handle changes to polygon tool options."""
        self.polygon_tool.set_mode(options['mode'])
        self.toolChanged.emit("polygon", options)
    
    def on_threshold_options_changed(self, options):
        """Handle changes to threshold tool options."""
        self.threshold_tool.set_operation(options['operation'])
        self.threshold_tool.set_mode(options['mode'])
        self.threshold_tool.set_thresholds(options['lower_threshold'], options['upper_threshold'])
        self.threshold_tool.region_growing_enabled = options['region_growing']
        self.threshold_tool.smooth_contours = options['smooth_contours']
        self.toolChanged.emit("threshold", options)
    
    def on_apply_threshold(self):
        """Handle apply threshold button click."""
        result = self.threshold_tool.apply_threshold()
        if result:
            contour_points, slice_index, orientation = result
            if contour_points and len(contour_points) > 0:
                # If a structure is set, add the contours to it
                if self.threshold_tool.structure:
                    for points in contour_points:
                        self.threshold_tool.structure.add_contour(points, slice_index, orientation)
                    self.structureModified.emit()
                    
                    # Clear preview after applying
                    self.threshold_tool.preview_mask = None 