#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MPR Structure Integration Module
===============================

This module integrates the MPR viewer with structure segmentation tools,
providing functionality for displaying and editing structures in the viewer.
"""

import logging
import numpy as np
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QPen, QBrush, QCursor

from quangtps.core.structures import Structure, StructureSet
from quangtps.ui.mpr_viewer import MPRViewer
from quangtps.ui.structure_tab import StructureTab

logger = logging.getLogger(__name__)

class MPRStructureIntegration(QObject):
    """
    Integration class for connecting MPR viewer with structure tools.
    
    This class handles the interaction between the MPR viewer and the structure
    editing tools, managing structure display, mouse events, and other
    integration aspects.
    """
    
    # Signals
    structureEdited = pyqtSignal(Structure)
    
    def __init__(self, mpr_viewer, structure_tab):
        """Initialize the integration with viewer and structure tab."""
        super().__init__()
        
        self.mpr_viewer = mpr_viewer
        self.structure_tab = structure_tab
        
        # Initialize state
        self.structure_set = None
        self.selected_structure = None
        self.editing_enabled = False
        
        # Set up connections
        self.setup_connections()
    
    def setup_connections(self):
        """Set up signal-slot connections."""
        # Connect structure tab signals
        self.structure_tab.structureSetChanged.connect(self.set_structure_set)
        self.structure_tab.structureSelectionChanged.connect(self.set_selected_structure)
        self.structure_tab.structureModified.connect(self.refresh_structure_overlay)
        self.structure_tab.structureAdded.connect(self.on_structure_added)
        self.structure_tab.structureRemoved.connect(self.on_structure_removed)
        self.structure_tab.structureVisibilityChanged.connect(self.on_structure_visibility_changed)
        
        # Connect MPR viewer signals for mouse interaction
        self.mpr_viewer.mousePressed.connect(self.on_mouse_pressed)
        self.mpr_viewer.mouseMoved.connect(self.on_mouse_moved)
        self.mpr_viewer.mouseReleased.connect(self.on_mouse_released)
        
        # Connect MPR viewer signals for overlay
        self.mpr_viewer.sliceChanged.connect(self.update_structure_overlay)
        self.mpr_viewer.orientationChanged.connect(self.update_structure_overlay)
        self.mpr_viewer.zoomChanged.connect(self.update_structure_overlay)
    
    def set_structure_set(self, structure_set):
        """Set the current structure set."""
        self.structure_set = structure_set
        
        # Update the viewer with all structures
        self.refresh_all_structure_overlays()
    
    def set_selected_structure(self, structure):
        """Set the currently selected structure."""
        self.selected_structure = structure
        
        # Enable structure editing in the viewer
        self.editing_enabled = (structure is not None)
        
        # Update cursor in the viewer
        if self.editing_enabled:
            self.mpr_viewer.setCursor(self.structure_tab.get_cursor_for_viewer())
        else:
            self.mpr_viewer.setCursor(Qt.ArrowCursor)
    
    def enable_editing(self, enabled=True):
        """Enable or disable structure editing."""
        self.editing_enabled = enabled and (self.selected_structure is not None)
        
        # Update cursor in the viewer
        if self.editing_enabled:
            self.mpr_viewer.setCursor(self.structure_tab.get_cursor_for_viewer())
        else:
            self.mpr_viewer.setCursor(Qt.ArrowCursor)
    
    def refresh_all_structure_overlays(self):
        """Refresh all structure overlays in the viewer."""
        self.mpr_viewer.clear_all_overlays()
        
        if not self.structure_set:
            return
        
        for structure in self.structure_set.structures:
            if structure.visible:
                self.add_structure_overlay(structure)
        
        # Request redraw
        self.mpr_viewer.update_all_views()
    
    def refresh_structure_overlay(self, structure):
        """Refresh the overlay for a specific structure."""
        if not structure:
            return
        
        # Clear current overlay for this structure and add again
        self.mpr_viewer.remove_structure_overlay(structure.id)
        
        if structure.visible:
            self.add_structure_overlay(structure)
        
        # Request redraw
        self.mpr_viewer.update_all_views()
    
    def add_structure_overlay(self, structure):
        """Add a structure overlay to the viewer."""
        if not structure:
            return
        
        # Convert structure to overlay format expected by the viewer
        structure_color = QColor(*structure.color)
        
        # Add overlay to viewer
        self.mpr_viewer.add_structure_overlay(
            structure.id,
            structure,
            structure_color,
            structure == self.selected_structure
        )
    
    def update_structure_overlay(self):
        """Update structure overlays when slice or orientation changes."""
        # This is called when slice index or orientation changes
        # Check if we need to update the tool overlay
        if self.editing_enabled and self.selected_structure:
            orientation = self.mpr_viewer.current_orientation
            slice_index = self.mpr_viewer.get_current_slice_index()
            
            # Get tool overlay from structure tab
            overlay = self.structure_tab.get_overlay_for_viewer(orientation, slice_index)
            
            if overlay is not None:
                # Add as temporary overlay
                self.mpr_viewer.add_temp_overlay(overlay)
    
    def on_structure_added(self, structure):
        """Handle structure added event."""
        if structure.visible:
            self.add_structure_overlay(structure)
            self.mpr_viewer.update_all_views()
    
    def on_structure_removed(self, structure):
        """Handle structure removed event."""
        self.mpr_viewer.remove_structure_overlay(structure.id)
        self.mpr_viewer.update_all_views()
    
    def on_structure_visibility_changed(self, structure, visible):
        """Handle structure visibility change event."""
        if visible:
            self.add_structure_overlay(structure)
        else:
            self.mpr_viewer.remove_structure_overlay(structure.id)
        
        self.mpr_viewer.update_all_views()
    
    def on_mouse_pressed(self, view_id, event_pos, world_pos):
        """Handle mouse press events from the viewer."""
        if not self.editing_enabled or not self.selected_structure:
            return
        
        # Get current slice and orientation
        orientation = self.mpr_viewer.views[view_id].orientation
        slice_index = self.mpr_viewer.views[view_id].slice_index
        
        # Convert position to image coordinates
        image_pos = self.mpr_viewer.views[view_id].view_to_image(event_pos)
        
        # Forward to structure tab
        handled = self.structure_tab.handle_mouse_event(
            "press", image_pos, slice_index, orientation
        )
        
        if handled:
            # Update overlay immediately
            self.update_structure_overlay()
            self.mpr_viewer.update_view(view_id)
    
    def on_mouse_moved(self, view_id, event_pos, world_pos):
        """Handle mouse move events from the viewer."""
        if not self.editing_enabled or not self.selected_structure:
            return
        
        # Convert position to image coordinates
        image_pos = self.mpr_viewer.views[view_id].view_to_image(event_pos)
        
        # Forward to structure tab
        handled = self.structure_tab.handle_mouse_event("move", image_pos)
        
        if handled:
            # Update overlay immediately
            self.update_structure_overlay()
            self.mpr_viewer.update_view(view_id)
    
    def on_mouse_released(self, view_id, event_pos, world_pos):
        """Handle mouse release events from the viewer."""
        if not self.editing_enabled or not self.selected_structure:
            return
        
        # Convert position to image coordinates
        image_pos = self.mpr_viewer.views[view_id].view_to_image(event_pos)
        
        # Forward to structure tab
        handled = self.structure_tab.handle_mouse_event("release", image_pos)
        
        if handled:
            # Refresh the structure overlay
            self.refresh_structure_overlay(self.selected_structure)
            
            # Emit signal
            self.structureEdited.emit(self.selected_structure) 