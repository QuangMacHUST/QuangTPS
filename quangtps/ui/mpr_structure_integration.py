#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MPR Structure Integration

This module provides integration between the MPR viewer and structure contouring tools,
providing an Eclipse-like interface for structure delineation and editing.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton, 
    QLabel, QToolBar, QAction, QMenu, QToolButton, QDialog,
    QListWidget, QListWidgetItem, QComboBox, QMessageBox,
    QGroupBox, QFormLayout, QLineEdit, QColorDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import QIcon, QColor

from quangtps.core.logging import get_logger
from quangtps.ui.mpr_viewer import MPRViewer
from quangtps.segmentation.segmentation_interface import SegmentationInterface

logger = get_logger(__name__)

class MPRStructureIntegration(QWidget):
    """
    Integration between MPR viewer and structure contouring tools.
    
    This class provides an Eclipse-style interface for structure delineation,
    combining the MPR viewer and segmentation tools in a cohesive UI.
    """
    
    # Signals
    structure_modified = pyqtSignal(object)  # Emits structure object when modified
    structure_added = pyqtSignal(object)     # Emits when a new structure is added
    structure_deleted = pyqtSignal(object)   # Emits when a structure is deleted
    
    def __init__(self, parent=None):
        """Initialize the MPR structure integration widget."""
        super().__init__(parent)
        
        # Current state
        self.image = None
        self.structure_set = None
        self.active_structure = None
        self.current_slice = {
            "axial": 0,
            "sagittal": 0,
            "coronal": 0
        }
        
        # Initialize UI
        self._init_ui()
        
        # Connect signals
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left pane (structure list and controls)
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        
        # Structure list
        structure_group = QGroupBox("Structures")
        structure_layout = QVBoxLayout(structure_group)
        
        self.structure_list = QListWidget()
        self.structure_list.setSelectionMode(QListWidget.SingleSelection)
        structure_layout.addWidget(self.structure_list)
        
        # Structure controls
        structure_buttons = QHBoxLayout()
        
        self.add_structure_button = QPushButton("Add")
        self.add_structure_button.setIcon(QIcon("quangtps/ui/icons/new_icons/add.png"))
        self.add_structure_button.clicked.connect(self._on_add_structure)
        structure_buttons.addWidget(self.add_structure_button)
        
        self.delete_structure_button = QPushButton("Delete")
        self.delete_structure_button.setIcon(QIcon("quangtps/ui/icons/new_icons/delete.png"))
        self.delete_structure_button.clicked.connect(self._on_delete_structure)
        structure_buttons.addWidget(self.delete_structure_button)
        
        structure_layout.addLayout(structure_buttons)
        
        left_layout.addWidget(structure_group)
        
        # Contouring tools
        tools_group = QGroupBox("Contouring Tools")
        tools_layout = QVBoxLayout(tools_group)
        
        self.segmentation_interface = SegmentationInterface()
        tools_layout.addWidget(self.segmentation_interface)
        
        left_layout.addWidget(tools_group)
        
        # Propagation controls
        propagation_group = QGroupBox("Propagation")
        propagation_layout = QVBoxLayout(propagation_group)
        
        self.copy_to_next_button = QPushButton("Copy to Next Slice")
        self.copy_to_next_button.setIcon(QIcon("quangtps/ui/icons/new_icons/copy.png"))
        self.copy_to_next_button.clicked.connect(self._on_copy_to_next)
        propagation_layout.addWidget(self.copy_to_next_button)
        
        self.copy_to_all_button = QPushButton("Copy to All Slices")
        self.copy_to_all_button.setIcon(QIcon("quangtps/ui/icons/new_icons/copy_all.png"))
        self.copy_to_all_button.clicked.connect(self._on_copy_to_all)
        propagation_layout.addWidget(self.copy_to_all_button)
        
        self.auto_segment_button = QPushButton("Auto-Segment")
        self.auto_segment_button.setIcon(QIcon("quangtps/ui/icons/new_icons/auto_segment.png"))
        self.auto_segment_button.clicked.connect(self._on_auto_segment)
        propagation_layout.addWidget(self.auto_segment_button)
        
        left_layout.addWidget(propagation_group)
        
        # Add left pane to splitter
        main_splitter.addWidget(left_pane)
        
        # Right pane (MPR viewer)
        self.mpr_viewer = MPRViewer()
        main_splitter.addWidget(self.mpr_viewer)
        
        # Set splitter sizes
        main_splitter.setSizes([300, 700])
    
    def _connect_signals(self):
        """Connect signals between components."""
        # MPR viewer signals
        self.mpr_viewer.mouse_pressed.connect(self._on_viewer_mouse_pressed)
        self.mpr_viewer.mouse_moved.connect(self._on_viewer_mouse_moved)
        self.mpr_viewer.mouse_released.connect(self._on_viewer_mouse_released)
        self.mpr_viewer.slice_changed.connect(self._on_slice_changed)
        
        # Structure list signals
        self.structure_list.currentItemChanged.connect(
            lambda current, previous: self._on_structure_selected(current)
        )
        
        # Segmentation interface signals
        self.segmentation_interface.contour_created.connect(self._on_contour_created)
        self.segmentation_interface.contour_modified.connect(self._on_contour_modified)
    
    def set_image(self, image):
        """
        Set the current image for visualization and contouring.
        
        Args:
            image: Image object with 3D data
        """
        self.image = image
        
        # Set image in MPR viewer
        self.mpr_viewer.set_image(image)
        
        # Set image data in segmentation interface
        if hasattr(image, "data") and image.data is not None:
            self.segmentation_interface.set_image_data(image.data)
            
        # Enable MPR viewer
        self.mpr_viewer.setEnabled(image is not None)
        
        # Enable/disable UI elements
        self._update_ui_state()
    
    def set_structure_set(self, structure_set):
        """
        Set the current structure set.
        
        Args:
            structure_set: Structure set object
        """
        self.structure_set = structure_set
        
        # Update structure list
        self._update_structure_list()
        
        # Update MPR viewer
        self.mpr_viewer.set_structure_set(structure_set)
        
        # Enable/disable UI elements
        self._update_ui_state()
    
    def set_active_structure(self, structure):
        """
        Set the active structure for contouring.
        
        Args:
            structure: Structure object
        """
        self.active_structure = structure
        
        # Update structure list selection
        self._select_structure(structure)
        
        # Set structure in segmentation interface
        self.segmentation_interface.set_current_structure(structure)
        
        # Update MPR viewer
        self.mpr_viewer.set_active_structure(structure)
        
        # Enable/disable UI elements
        self._update_ui_state()
    
    def _on_viewer_mouse_pressed(self, orientation: str, slice_index: int, position: QPoint, button: int):
        """
        Handle mouse pressed event from MPR viewer.
        
        Args:
            orientation: Orientation plane ("axial", "sagittal", or "coronal")
            slice_index: Slice index in the specified orientation
            position: Mouse position in image coordinates
            button: Qt mouse button code
        """
        # Update current slice
        self.current_slice[orientation] = slice_index
        
        # Update slice in segmentation interface
        self.segmentation_interface.set_current_slice(slice_index, orientation)
        
        # Forward event to segmentation interface
        if self.active_structure is not None:
            # Create a compatible mouse event
            event = self._create_mouse_event(position, button)
            
            # Pass event to segmentation interface
            self.segmentation_interface.handle_mouse_event(
                "press", event, slice_index, orientation
            )
    
    def _on_viewer_mouse_moved(self, orientation: str, slice_index: int, position: QPoint, buttons: int):
        """
        Handle mouse moved event from MPR viewer.
        
        Args:
            orientation: Orientation plane
            slice_index: Slice index in the specified orientation
            position: Mouse position in image coordinates
            buttons: Qt mouse buttons state
        """
        # Forward event to segmentation interface
        if self.active_structure is not None:
            # Create a compatible mouse event
            event = self._create_mouse_event(position, buttons)
            
            # Pass event to segmentation interface
            self.segmentation_interface.handle_mouse_event(
                "move", event, slice_index, orientation
            )
    
    def _on_viewer_mouse_released(self, orientation: str, slice_index: int, position: QPoint, button: int):
        """
        Handle mouse released event from MPR viewer.
        
        Args:
            orientation: Orientation plane
            slice_index: Slice index in the specified orientation
            position: Mouse position in image coordinates
            button: Qt mouse button code
        """
        # Forward event to segmentation interface
        if self.active_structure is not None:
            # Create a compatible mouse event
            event = self._create_mouse_event(position, button)
            
            # Pass event to segmentation interface
            self.segmentation_interface.handle_mouse_event(
                "release", event, slice_index, orientation
            )
    
    def _on_slice_changed(self, orientation: str, slice_index: int):
        """
        Handle slice changed event from MPR viewer.
        
        Args:
            orientation: Orientation plane
            slice_index: New slice index
        """
        # Update current slice
        self.current_slice[orientation] = slice_index
        
        # Update slice in segmentation interface
        self.segmentation_interface.set_current_slice(slice_index, orientation)
        
        # Set overlay painter for segmentation interface preview
        self.mpr_viewer.set_overlay_painter(
            orientation,
            self.get_overlay_painter(orientation, slice_index)
        )
    
    def _on_contour_created(self, contour, structure_info):
        """
        Handle contour creation.
        
        Args:
            contour: Contour data
            structure_info: Dictionary with structure information
        """
        if self.active_structure is None:
            logger.warning("No active structure selected")
            return
        
        structure = structure_info["structure"]
        slice_index = structure_info["slice_index"]
        orientation = structure_info["orientation"]
        
        # Add contour to structure
        if hasattr(structure, "add_contour"):
            structure.add_contour(contour, slice_index, orientation)
        
        # Update MPR viewer
        self.mpr_viewer.update_views()
        
        # Emit signal
        self.structure_modified.emit(structure)
        
        logger.debug(f"Added contour to {structure.name} " +
                    f"on slice {slice_index} ({orientation})")
    
    def _on_contour_modified(self, contour, structure_info):
        """
        Handle contour modification.
        
        Args:
            contour: Contour data
            structure_info: Dictionary with structure information
        """
        if self.active_structure is None:
            logger.warning("No active structure selected")
            return
        
        structure = structure_info["structure"]
        slice_index = structure_info["slice_index"]
        orientation = structure_info["orientation"]
        
        # Update contour in structure
        if hasattr(structure, "update_contour"):
            structure.update_contour(contour, slice_index, orientation)
        elif hasattr(structure, "add_contour"):
            # Fallback to add_contour if update_contour is not available
            structure.add_contour(contour, slice_index, orientation)
        
        # Update MPR viewer
        self.mpr_viewer.update_views()
        
        # Emit signal
        self.structure_modified.emit(structure)
        
        logger.debug(f"Modified contour in {structure.name} " +
                    f"on slice {slice_index} ({orientation})")
    
    def _on_add_structure(self):
        """Handle add structure button click."""
        if self.structure_set is None:
            logger.warning("No structure set available")
            return
        
        # Create dialog for structure properties
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Structure")
        
        # Dialog layout
        layout = QVBoxLayout(dialog)
        
        # Form layout for structure properties
        form_layout = QFormLayout()
        
        # Structure name
        name_edit = QLineEdit()
        form_layout.addRow("Name:", name_edit)
        
        # Structure type
        type_combo = QComboBox()
        type_combo.addItems(["PTV", "CTV", "GTV", "OAR", "BODY", "EXTERNAL", "Other"])
        form_layout.addRow("Type:", type_combo)
        
        # Structure color
        self.color_button = QPushButton()
        self.current_color = QColor(255, 0, 0)  # Default to red
        self.color_button.setStyleSheet(
            f"background-color: {self.current_color.name()}; min-height: 20px;"
        )
        self.color_button.clicked.connect(self._on_color_select)
        form_layout.addRow("Color:", self.color_button)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        add_button = QPushButton("Add")
        add_button.clicked.connect(dialog.accept)
        button_layout.addWidget(add_button)
        
        layout.addLayout(button_layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Error", "Structure name cannot be empty")
                return
            
            # Create new structure
            try:
                structure_type = type_combo.currentText()
                
                # Add structure to set
                new_structure = self.structure_set.add_structure(
                    name, structure_type, self.current_color
                )
                
                # Update structure list
                self._update_structure_list()
                
                # Set as active structure
                self.set_active_structure(new_structure)
                
                # Emit signal
                self.structure_added.emit(new_structure)
                
                logger.debug(f"Added structure: {name} ({structure_type})")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add structure: {str(e)}")
                logger.error(f"Error adding structure: {str(e)}")
    
    def _on_color_select(self):
        """Handle color selection button click."""
        color = QColorDialog.getColor(self.current_color, self, "Select Structure Color")
        if color.isValid():
            self.current_color = color
            self.color_button.setStyleSheet(
                f"background-color: {color.name()}; min-height: 20px;"
            )
    
    def _on_delete_structure(self):
        """Handle delete structure button click."""
        if not self.active_structure:
            QMessageBox.warning(self, "Warning", "No structure selected")
            return
        
        # Confirm deletion
        response = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete the structure '{self.active_structure.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if response == QMessageBox.Yes:
            try:
                # Remove structure from set
                structure = self.active_structure
                self.structure_set.remove_structure(structure)
                
                # Clear active structure
                self.active_structure = None
                
                # Update structure list
                self._update_structure_list()
                
                # Update MPR viewer
                self.mpr_viewer.set_active_structure(None)
                self.mpr_viewer.update_views()
                
                # Emit signal
                self.structure_deleted.emit(structure)
                
                logger.debug(f"Deleted structure: {structure.name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete structure: {str(e)}")
                logger.error(f"Error deleting structure: {str(e)}")
    
    def _on_structure_selected(self, item):
        """
        Handle structure selection in list.
        
        Args:
            item: Selected QListWidgetItem
        """
        if item is None:
            self.active_structure = None
        else:
            # Get structure from item data
            structure = item.data(Qt.UserRole)
            self.active_structure = structure
        
        # Update segmentation interface
        self.segmentation_interface.set_current_structure(self.active_structure)
        
        # Update MPR viewer
        self.mpr_viewer.set_active_structure(self.active_structure)
        
        # Enable/disable UI elements
        self._update_ui_state()
    
    def _on_copy_to_next(self):
        """Handle copy to next slice button click."""
        if not self.active_structure:
            QMessageBox.warning(self, "Warning", "No structure selected")
            return
        
        # Get current orientation and slice
        orientation = self.mpr_viewer.get_active_orientation()
        current_slice = self.current_slice[orientation]
        
        # Check if we're at the last slice
        max_slice = self.mpr_viewer.get_max_slice(orientation)
        if current_slice >= max_slice - 1:
            QMessageBox.information(self, "Information", "Already at the last slice")
            return
        
        try:
            # Copy contour to next slice
            next_slice = current_slice + 1
            
            # Call the structure's copy contour method
            if hasattr(self.active_structure, "copy_contour_to_slice"):
                self.active_structure.copy_contour_to_slice(
                    current_slice, next_slice, orientation
                )
            
            # Move to next slice
            self.mpr_viewer.set_slice(orientation, next_slice)
            
            # Update views
            self.mpr_viewer.update_views()
            
            # Emit signal
            self.structure_modified.emit(self.active_structure)
            
            logger.debug(f"Copied contour from slice {current_slice} to {next_slice}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to copy contour: {str(e)}")
            logger.error(f"Error copying contour: {str(e)}")
    
    def _on_copy_to_all(self):
        """Handle copy to all slices button click."""
        if not self.active_structure:
            QMessageBox.warning(self, "Warning", "No structure selected")
            return
        
        # Get current orientation and slice
        orientation = self.mpr_viewer.get_active_orientation()
        current_slice = self.current_slice[orientation]
        
        # Get slice range
        max_slice = self.mpr_viewer.get_max_slice(orientation)
        
        # Confirm operation
        response = QMessageBox.question(
            self, "Confirm Operation",
            f"Are you sure you want to copy the current contour to all {max_slice} slices?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if response == QMessageBox.Yes:
            try:
                # Copy contour to all slices
                for slice_idx in range(max_slice):
                    # Skip current slice
                    if slice_idx == current_slice:
                        continue
                    
                    # Call the structure's copy contour method
                    if hasattr(self.active_structure, "copy_contour_to_slice"):
                        self.active_structure.copy_contour_to_slice(
                            current_slice, slice_idx, orientation
                        )
                
                # Update views
                self.mpr_viewer.update_views()
                
                # Emit signal
                self.structure_modified.emit(self.active_structure)
                
                logger.debug(f"Copied contour from slice {current_slice} to all slices")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to copy contours: {str(e)}")
                logger.error(f"Error copying contours: {str(e)}")
    
    def _on_auto_segment(self):
        """Handle auto-segment button click."""
        if not self.active_structure:
            QMessageBox.warning(self, "Warning", "No structure selected")
            return
        
        # This is a placeholder for a future auto-segmentation feature
        QMessageBox.information(
            self, "Auto-Segmentation",
            "Auto-segmentation feature will be available in a future update."
        )
        
        logger.debug("Auto-segmentation requested (not implemented)")
    
    def _create_mouse_event(self, position, button):
        """
        Create a mouse event compatible with segmentation tools.
        
        Args:
            position: QPoint with mouse coordinates
            button: Qt mouse button code
            
        Returns:
            Mouse event object
        """
        class MouseEvent:
            def __init__(self, pos, btn):
                self._pos = pos
                self._button = btn
            
            def pos(self):
                return self._pos
            
            def buttons(self):
                return self._button
            
            def button(self):
                return self._button
        
        return MouseEvent(position, button)
    
    def get_overlay_painter(self, orientation, slice_index):
        """
        Get an overlay painter function for the given orientation and slice.
        
        Args:
            orientation: Orientation plane
            slice_index: Slice index
            
        Returns:
            Function to paint overlay
        """
        def paint_overlay(painter):
            """
            Paint segmentation tool overlay.
            
            Args:
                painter: QPainter object
            """
            # Set current slice and orientation in segmentation interface
            self.segmentation_interface.set_current_slice(slice_index, orientation)
            
            # Draw tool-specific overlay
            self.segmentation_interface.draw_overlay(painter, slice_index, orientation)
        
        return paint_overlay
    
    def _update_structure_list(self):
        """Update the structure list widget."""
        self.structure_list.clear()
        
        if self.structure_set is None:
            return
        
        # Add structures to list
        for structure in self.structure_set.structures:
            item = QListWidgetItem(structure.name)
            
            # Set icon or colored square
            pixmap_size = QSize(16, 16)
            if hasattr(structure, "color"):
                color = structure.color
                if isinstance(color, (list, tuple)) and len(color) >= 3:
                    # Convert RGB to QColor
                    if len(color) == 3:
                        qcolor = QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    else:  # RGBA
                        qcolor = QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*255))
                elif isinstance(color, QColor):
                    qcolor = color
                else:
                    # Default color
                    qcolor = QColor(255, 0, 0)
                
                # Set background color for item
                item.setBackground(qcolor)
            
            # Store structure in item data
            item.setData(Qt.UserRole, structure)
            
            self.structure_list.addItem(item)
    
    def _select_structure(self, structure):
        """
        Select a structure in the list.
        
        Args:
            structure: Structure object to select
        """
        if structure is None:
            self.structure_list.clearSelection()
            return
        
        # Find item with matching structure
        for i in range(self.structure_list.count()):
            item = self.structure_list.item(i)
            if item.data(Qt.UserRole) == structure:
                self.structure_list.setCurrentItem(item)
                break
    
    def _update_ui_state(self):
        """Update UI elements based on current state."""
        has_image = self.image is not None
        has_structure_set = self.structure_set is not None
        has_active_structure = self.active_structure is not None
        
        # Structure list
        self.structure_list.setEnabled(has_structure_set)
        
        # Structure controls
        self.add_structure_button.setEnabled(has_structure_set)
        self.delete_structure_button.setEnabled(has_active_structure)
        
        # Segmentation interface
        self.segmentation_interface.setEnabled(has_active_structure and has_image)
        
        # Propagation controls
        self.copy_to_next_button.setEnabled(has_active_structure)
        self.copy_to_all_button.setEnabled(has_active_structure)
        self.auto_segment_button.setEnabled(has_active_structure)


# For testing
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    # Test structure class
    class TestStructure:
        def __init__(self, name, color):
            self.name = name
            self.color = color
            self.contours = {}
        
        def add_contour(self, contour, slice_index, orientation):
            key = f"{orientation}_{slice_index}"
            self.contours[key] = contour
        
        def copy_contour_to_slice(self, from_slice, to_slice, orientation):
            from_key = f"{orientation}_{from_slice}"
            to_key = f"{orientation}_{to_slice}"
            if from_key in self.contours:
                self.contours[to_key] = self.contours[from_key]
    
    # Test structure set class
    class TestStructureSet:
        def __init__(self):
            self.structures = []
        
        def add_structure(self, name, type_name, color):
            structure = TestStructure(name, color)
            self.structures.append(structure)
            return structure
        
        def remove_structure(self, structure):
            if structure in self.structures:
                self.structures.remove(structure)
    
    app = QApplication(sys.argv)
    
    # Create test structure set with some structures
    structure_set = TestStructureSet()
    
    for name, color in [
        ("Body", QColor(0, 0, 255)),
        ("PTV", QColor(255, 0, 0)),
        ("CTV", QColor(0, 255, 0)),
        ("Bladder", QColor(255, 255, 0))
    ]:
        structure_set.add_structure(name, "Type", color)
    
    # Create MPR structure integration widget
    widget = MPRStructureIntegration()
    widget.setWindowTitle("MPR Structure Integration")
    widget.resize(1000, 800)
    
    # Set test structure set
    widget.set_structure_set(structure_set)
    
    # Create test image (simple noise)
    class TestImage:
        def __init__(self):
            self.data = np.random.rand(100, 100, 50) * 100
    
    test_image = TestImage()
    widget.set_image(test_image)
    
    widget.show()
    
    sys.exit(app.exec_()) 