"""
Segmentation Interface

This module provides the interface for integrating various segmentation tools
in the QuangTPS treatment planning system.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QToolBar, 
    QPushButton, QLabel, QComboBox, QToolButton, QAction, QMenu
)
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtCore import Qt, pyqtSignal

from quangtps.core.logging import get_logger
from quangtps.segmentation.manual_segmentation.freehand_tool import FreehandTool, FreehandToolWidget
from quangtps.segmentation.manual_segmentation.polygon_tool import PolygonTool, PolygonToolWidget
from quangtps.segmentation.manual_segmentation.threshold_tool import ThresholdTool, ThresholdToolWidget

logger = get_logger(__name__)

class SegmentationInterface(QWidget):
    """
    Interface for segmentation tools in QuangTPS.
    
    This class provides an Eclipse-style interface for different segmentation
    tools, including manual drawing tools (freehand, polygon, brush) and
    automatic segmentation methods.
    """
    
    # Signals
    contour_created = pyqtSignal(object, object)  # Emits (contour, structure_info)
    contour_modified = pyqtSignal(object, object)  # Emits (contour, structure_info)
    
    def __init__(self, parent=None):
        """Initialize the segmentation interface."""
        super().__init__(parent)
        
        # Current state
        self.current_tool = None
        self.current_structure = None
        self.current_slice = None
        self.current_orientation = "axial"
        self.image_data = None
        
        # Available tools
        self.tools = {}
        
        # Initialize UI
        self._init_ui()
        
        # Create and register tools
        self._register_tools()
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar for tool selection
        self.tool_toolbar = QToolBar("Segmentation Tools")
        self.tool_toolbar.setIconSize(Qt.QSize(24, 24))
        self.tool_toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        main_layout.addWidget(self.tool_toolbar)
        
        # Tab widget for tool options
        self.tool_tabs = QTabWidget()
        self.tool_tabs.setTabPosition(QTabWidget.South)
        
        main_layout.addWidget(self.tool_tabs)
    
    def _register_tools(self):
        """Register available segmentation tools."""
        # Manual tools
        self._add_tool("freehand", "Freehand", FreehandToolWidget())
        self._add_tool("polygon", "Polygon", PolygonToolWidget())
        self._add_tool("threshold", "Threshold", ThresholdToolWidget())
        
        # Select default tool
        if self.tools:
            default_tool = "freehand"
            if default_tool in self.tools:
                self.set_active_tool(default_tool)
    
    def _add_tool(self, tool_id: str, tool_name: str, tool_widget: QWidget):
        """
        Add a tool to the interface.
        
        Args:
            tool_id: Unique ID for the tool
            tool_name: Display name for the tool
            tool_widget: Widget for tool options
        """
        # Add to tools dictionary
        self.tools[tool_id] = {
            "id": tool_id,
            "name": tool_name,
            "widget": tool_widget
        }
        
        # Create action for toolbar
        icon = QIcon(f"quangtps/ui/icons/new_icons/{tool_id}.png")
        if not icon.isNull():
            action = QAction(icon, tool_name, self)
        else:
            action = QAction(tool_name, self)
            
        action.setCheckable(True)
        action.setData(tool_id)
        action.triggered.connect(lambda checked, tid=tool_id: self.set_active_tool(tid))
        
        self.tool_toolbar.addAction(action)
        
        # Add widget to tabs
        self.tool_tabs.addTab(tool_widget, tool_name)
        
        # Connect signals
        if hasattr(tool_widget, "contour_created"):
            tool_widget.contour_created.connect(self._on_contour_created)
    
    def set_active_tool(self, tool_id: str):
        """
        Set the active segmentation tool.
        
        Args:
            tool_id: ID of the tool to activate
        """
        if tool_id not in self.tools:
            logger.warning(f"Unknown tool ID: {tool_id}")
            return
        
        self.current_tool = tool_id
        
        # Update toolbar actions
        for action in self.tool_toolbar.actions():
            action_tool_id = action.data()
            action.setChecked(action_tool_id == tool_id)
        
        # Update tab widget
        tool_index = list(self.tools.keys()).index(tool_id)
        self.tool_tabs.setCurrentIndex(tool_index)
        
        logger.debug(f"Activated tool: {tool_id}")
    
    def set_current_structure(self, structure):
        """
        Set the current structure for segmentation.
        
        Args:
            structure: Structure object to segment
        """
        self.current_structure = structure
    
    def set_current_slice(self, slice_index: int, orientation: str = "axial"):
        """
        Set the current slice for segmentation.
        
        Args:
            slice_index: Index of the current slice
            orientation: Orientation plane ("axial", "sagittal", or "coronal")
        """
        self.current_slice = slice_index
        self.current_orientation = orientation
    
    def set_image_data(self, image_data):
        """
        Set the current image data.
        
        Args:
            image_data: Image data array
        """
        self.image_data = image_data
    
    def handle_mouse_event(self, event_type: str, event, 
                         slice_index: Optional[int] = None, 
                         orientation: Optional[str] = None):
        """
        Handle mouse events from the image viewer.
        
        Args:
            event_type: Type of event ("press", "move", "release")
            event: Mouse event object
            slice_index: Index of the current slice
            orientation: Orientation plane
        """
        if self.current_tool is None:
            return
        
        # Update current slice and orientation if provided
        if slice_index is not None:
            self.current_slice = slice_index
        if orientation is not None:
            self.current_orientation = orientation
        
        # Get the current tool widget
        tool_widget = self.tools[self.current_tool]["widget"]
        
        # Dispatch event to the tool
        if event_type == "press":
            if hasattr(tool_widget, "on_mouse_press"):
                tool_widget.on_mouse_press(event, self.image_data)
                
        elif event_type == "move":
            if hasattr(tool_widget, "on_mouse_move"):
                tool_widget.on_mouse_move(event, self.image_data)
                
        elif event_type == "release":
            if hasattr(tool_widget, "on_mouse_release"):
                tool_widget.on_mouse_release(event, self.image_data)
    
    def get_cursor_for_tool(self):
        """
        Get the cursor for the current tool.
        
        Returns:
            QCursor object or None
        """
        if self.current_tool is None:
            return None
        
        tool_widget = self.tools[self.current_tool]["widget"]
        
        if hasattr(tool_widget, "tool") and hasattr(tool_widget.tool, "get_cursor"):
            return tool_widget.tool.get_cursor()
        
        return None
    
    def draw_overlay(self, painter, slice_index=None, orientation=None):
        """
        Draw tool-specific overlays on the image display.
        
        Args:
            painter: QPainter object
            slice_index: Index of the current slice
            orientation: Orientation plane
        """
        if self.current_tool is None:
            return
        
        # Update current slice and orientation if provided
        if slice_index is not None:
            self.current_slice = slice_index
        if orientation is not None:
            self.current_orientation = orientation
        
        # Get the current tool widget
        tool_widget = self.tools[self.current_tool]["widget"]
        
        # Draw tool-specific preview
        if hasattr(tool_widget, "tool") and hasattr(tool_widget.tool, "draw_preview"):
            tool_widget.tool.draw_preview(painter, self.image_data)
    
    def _on_contour_created(self, contour):
        """
        Handle creation of a new contour.
        
        Args:
            contour: Contour data
        """
        if self.current_structure is None:
            logger.warning("No current structure selected")
            return
        
        # Emit signal with contour and structure info
        structure_info = {
            "structure": self.current_structure,
            "slice_index": self.current_slice,
            "orientation": self.current_orientation
        }
        
        self.contour_created.emit(contour, structure_info)
        
        logger.debug(f"Created contour for {self.current_structure.name} " +
                    f"on slice {self.current_slice} ({self.current_orientation})")


# For testing
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    widget = SegmentationInterface()
    widget.setWindowTitle("Segmentation Interface Test")
    widget.resize(400, 600)
    widget.show()
    
    sys.exit(app.exec_()) 