#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure Browser Module
=======================

This module provides a widget for browsing and managing structures in
a structure set, with functionality similar to Eclipse's Structure panel.
"""

import os
import logging
from typing import List, Dict, Optional, Any, Tuple, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QMenu, QAction,
    QColorDialog, QCheckBox, QLineEdit, QComboBox, QInputDialog,
    QAbstractItemView, QSizePolicy, QFrame, QToolButton, QMessageBox
)
from PyQt5.QtGui import QColor, QIcon, QBrush, QFont, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QSize

# Import local modules if they exist
try:
    from quangtps.structures.structure import Structure
    from quangtps.structures.structure_set import StructureSet
except ImportError:
    logging.warning("Failed to import QuangTPS structure modules in structure browser")

logger = logging.getLogger(__name__)


class StructureTreeItem(QTreeWidgetItem):
    """Tree widget item representing a structure."""
    
    def __init__(self, structure: 'Structure', parent=None):
        """Initialize the structure tree item."""
        super().__init__(parent)
        self.structure = structure
        self.update_item()
    
    def update_item(self):
        """Update the item display from the structure properties."""
        # Set basic properties
        self.setText(0, self.structure.name)
        
        # Set color indicator
        color = self.structure.color
        if color is None:
            color = QColor(255, 0, 0)  # Default to red
        
        self.setForeground(0, QBrush(color))
        
        # Set icon based on visibility
        # We store actual visibility in column 1, as a checkbox will be used
        # to represent it in the tree widget
        visible = self.structure.visible if hasattr(self.structure, 'visible') else True
        self.setCheckState(1, Qt.Checked if visible else Qt.Unchecked)
        
        # Set type
        self.setText(2, self.structure.type if hasattr(self.structure, 'type') else "")
        
        # Set volume
        volume = self.structure.get_volume() if hasattr(self.structure, 'get_volume') else None
        self.setText(3, f"{volume:.2f} cc" if volume is not None else "")


class StructureBrowser(QWidget):
    """
    Widget for browsing and managing structures.
    
    This widget provides a tree view of structures in a structure set,
    with functionality for viewing, editing, and managing structures.
    """
    
    # Signals
    structureSelected = pyqtSignal(object)
    structureVisibilityChanged = pyqtSignal(object, bool)
    structureColorChanged = pyqtSignal(object, QColor)
    structureRenamed = pyqtSignal(object, str)
    structureDeleted = pyqtSignal(object)
    structureCreated = pyqtSignal(object)
    
    def __init__(self, parent=None):
        """Initialize the structure browser widget."""
        super().__init__(parent)
        
        # Initialize variables
        self.structure_set = None
        self.structure_map = {}  # Maps structure IDs to tree items
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Structures")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        
        filter_label = QLabel("Filter:")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter structures...")
        self.filter_edit.textChanged.connect(self.on_filter_changed)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(filter_label)
        header_layout.addWidget(self.filter_edit)
        
        main_layout.addLayout(header_layout)
        
        # Tree widget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Name", "Visible", "Type", "Volume"])
        self.tree_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree_widget.setColumnWidth(0, 150)  # Name column
        self.tree_widget.setColumnWidth(1, 50)   # Visibility column
        self.tree_widget.setColumnWidth(2, 100)  # Type column
        self.tree_widget.setColumnWidth(3, 80)   # Volume column
        
        # Set stretch for name column
        header = self.tree_widget.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        
        # Connect signals
        self.tree_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.tree_widget.itemChanged.connect(self.on_item_changed)
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self.show_context_menu)
        
        main_layout.addWidget(self.tree_widget)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        self.new_structure_button = QPushButton("New")
        self.new_structure_button.setToolTip("Create a new structure")
        self.new_structure_button.clicked.connect(self.on_new_structure)
        
        self.delete_structure_button = QPushButton("Delete")
        self.delete_structure_button.setToolTip("Delete the selected structure")
        self.delete_structure_button.clicked.connect(self.on_delete_structure)
        
        self.show_all_button = QPushButton("Show All")
        self.show_all_button.setToolTip("Make all structures visible")
        self.show_all_button.clicked.connect(self.on_show_all)
        
        self.hide_all_button = QPushButton("Hide All")
        self.hide_all_button.setToolTip("Hide all structures")
        self.hide_all_button.clicked.connect(self.on_hide_all)
        
        toolbar_layout.addWidget(self.new_structure_button)
        toolbar_layout.addWidget(self.delete_structure_button)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.show_all_button)
        toolbar_layout.addWidget(self.hide_all_button)
        
        main_layout.addLayout(toolbar_layout)
        
        # Apply styling
        self.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 4px 8px;
            }
            
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
            
            QLineEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 2px 4px;
            }
            
            QTreeWidget {
                border: 1px solid #cccccc;
                background-color: white;
            }
            
            QTreeWidget::item {
                padding: 3px 0px;
            }
            
            QTreeWidget::item:selected {
                background-color: #d8e9f5;
                color: black;
            }
        """)
        
        # Initialize empty state
        self.update_ui_state()
    
    def set_structure_set(self, structure_set: 'StructureSet'):
        """
        Set the structure set to display.
        
        Args:
            structure_set: The structure set to display
        """
        self.structure_set = structure_set
        self.update_structure_list()
    
    def update_structure_list(self):
        """Update the structure list from the current structure set."""
        self.tree_widget.clear()
        self.structure_map = {}
        
        if not self.structure_set or not self.structure_set.structures:
            return
        
        # Add each structure to the tree
        for structure in self.structure_set.structures:
            self.add_structure_to_tree(structure)
        
        # Update UI state
        self.update_ui_state()
    
    def add_structure_to_tree(self, structure: 'Structure'):
        """
        Add a structure to the tree.
        
        Args:
            structure: The structure to add
        """
        if not structure:
            return
        
        # Create a tree item for the structure
        item = StructureTreeItem(structure)
        self.tree_widget.addTopLevelItem(item)
        
        # Store in the structure map
        structure_id = structure.id if hasattr(structure, 'id') else id(structure)
        self.structure_map[structure_id] = item
    
    def update_structure_item(self, structure: 'Structure'):
        """
        Update the tree item for a structure.
        
        Args:
            structure: The structure to update
        """
        if not structure:
            return
        
        # Find the item for the structure
        structure_id = structure.id if hasattr(structure, 'id') else id(structure)
        item = self.structure_map.get(structure_id)
        
        if item:
            item.update_item()
    
    def get_selected_structure(self) -> Optional['Structure']:
        """
        Get the currently selected structure.
        
        Returns:
            The selected structure, or None if none selected
        """
        selected_items = self.tree_widget.selectedItems()
        if selected_items:
            return selected_items[0].structure
        return None
    
    def set_selected_structure(self, structure: 'Structure'):
        """
        Select a structure in the tree.
        
        Args:
            structure: The structure to select
        """
        if not structure:
            return
        
        # Find the item for the structure
        structure_id = structure.id if hasattr(structure, 'id') else id(structure)
        item = self.structure_map.get(structure_id)
        
        if item:
            self.tree_widget.setCurrentItem(item)
    
    def on_selection_changed(self):
        """Handle selection changes in the tree widget."""
        structure = self.get_selected_structure()
        if structure:
            self.structureSelected.emit(structure)
        
        # Update UI state
        self.update_ui_state()
    
    def on_item_changed(self, item, column):
        """
        Handle changes to items in the tree widget.
        
        Args:
            item: The item that changed
            column: The column that changed
        """
        if not hasattr(item, 'structure'):
            return
        
        # Handle visibility changes (column 1)
        if column == 1:
            visible = item.checkState(1) == Qt.Checked
            item.structure.visible = visible
            self.structureVisibilityChanged.emit(item.structure, visible)
    
    def on_filter_changed(self, text):
        """
        Handle filter text changes.
        
        Args:
            text: The new filter text
        """
        # Hide/show items based on filter
        for i in range(self.tree_widget.topLevelItemCount()):
            item = self.tree_widget.topLevelItem(i)
            item.setHidden(text and text.lower() not in item.text(0).lower())
    
    def show_context_menu(self, position):
        """
        Show the context menu for the tree widget.
        
        Args:
            position: The position to show the menu at
        """
        # Get the item at the position
        item = self.tree_widget.itemAt(position)
        if not item or not hasattr(item, 'structure'):
            return
        
        # Create the context menu
        menu = QMenu(self)
        
        # Add actions
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self.on_rename_structure(item.structure))
        menu.addAction(rename_action)
        
        change_color_action = QAction("Change Color", self)
        change_color_action.triggered.connect(lambda: self.on_change_color(item.structure))
        menu.addAction(change_color_action)
        
        menu.addSeparator()
        
        if hasattr(item.structure, 'visible') and item.structure.visible:
            hide_action = QAction("Hide", self)
            hide_action.triggered.connect(lambda: self.on_toggle_visibility(item.structure, False))
            menu.addAction(hide_action)
        else:
            show_action = QAction("Show", self)
            show_action.triggered.connect(lambda: self.on_toggle_visibility(item.structure, True))
            menu.addAction(show_action)
        
        menu.addSeparator()
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self.on_delete_structure(item.structure))
        menu.addAction(delete_action)
        
        # Show the menu
        menu.exec_(self.tree_widget.viewport().mapToGlobal(position))
    
    def on_new_structure(self):
        """Handle new structure button click."""
        if not self.structure_set:
            return
        
        # Get a name for the new structure
        name, ok = QInputDialog.getText(
            self, "New Structure", "Enter structure name:"
        )
        
        if ok and name:
            # Create a new structure
            structure = Structure()
            structure.name = name
            structure.color = QColor(255, 0, 0)  # Default red
            structure.visible = True
            
            # Add to structure set
            self.structure_set.add_structure(structure)
            
            # Add to tree
            self.add_structure_to_tree(structure)
            
            # Select the new structure
            self.set_selected_structure(structure)
            
            # Emit signal
            self.structureCreated.emit(structure)
    
    def on_delete_structure(self, structure=None):
        """
        Handle structure deletion.
        
        Args:
            structure: The structure to delete, or None to use selected
        """
        if structure is None:
            structure = self.get_selected_structure()
        
        if not structure or not self.structure_set:
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Structure",
            f"Are you sure you want to delete '{structure.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove from structure set
            if hasattr(self.structure_set, 'remove_structure'):
                self.structure_set.remove_structure(structure)
            elif hasattr(self.structure_set, 'structures'):
                if structure in self.structure_set.structures:
                    self.structure_set.structures.remove(structure)
            
            # Remove from tree
            structure_id = structure.id if hasattr(structure, 'id') else id(structure)
            item = self.structure_map.get(structure_id)
            
            if item:
                index = self.tree_widget.indexOfTopLevelItem(item)
                if index >= 0:
                    self.tree_widget.takeTopLevelItem(index)
                
                del self.structure_map[structure_id]
            
            # Emit signal
            self.structureDeleted.emit(structure)
    
    def on_rename_structure(self, structure):
        """
        Handle structure renaming.
        
        Args:
            structure: The structure to rename
        """
        if not structure:
            return
        
        # Get a new name
        name, ok = QInputDialog.getText(
            self, "Rename Structure", "Enter new name:",
            text=structure.name
        )
        
        if ok and name:
            # Update the structure
            old_name = structure.name
            structure.name = name
            
            # Update the tree item
            self.update_structure_item(structure)
            
            # Emit signal
            self.structureRenamed.emit(structure, name)
    
    def on_change_color(self, structure):
        """
        Handle structure color change.
        
        Args:
            structure: The structure to change color for
        """
        if not structure:
            return
        
        # Get the current color
        current_color = structure.color
        if current_color is None:
            current_color = QColor(255, 0, 0)  # Default red
        
        # Get a new color
        color = QColorDialog.getColor(
            current_color, self, "Select Structure Color"
        )
        
        if color.isValid():
            # Update the structure
            structure.color = color
            
            # Update the tree item
            self.update_structure_item(structure)
            
            # Emit signal
            self.structureColorChanged.emit(structure, color)
    
    def on_toggle_visibility(self, structure, visible):
        """
        Toggle structure visibility.
        
        Args:
            structure: The structure to toggle
            visible: Whether to make the structure visible or hidden
        """
        if not structure:
            return
        
        # Update the structure
        structure.visible = visible
        
        # Update the tree item
        self.update_structure_item(structure)
        
        # Emit signal
        self.structureVisibilityChanged.emit(structure, visible)
    
    def on_show_all(self):
        """Show all structures."""
        if not self.structure_set or not self.structure_set.structures:
            return
        
        # Update all structures
        for structure in self.structure_set.structures:
            structure.visible = True
            self.update_structure_item(structure)
            self.structureVisibilityChanged.emit(structure, True)
    
    def on_hide_all(self):
        """Hide all structures."""
        if not self.structure_set or not self.structure_set.structures:
            return
        
        # Update all structures
        for structure in self.structure_set.structures:
            structure.visible = False
            self.update_structure_item(structure)
            self.structureVisibilityChanged.emit(structure, False)
    
    def update_ui_state(self):
        """Update UI state based on current conditions."""
        has_structure_set = self.structure_set is not None
        has_selection = self.get_selected_structure() is not None
        
        self.new_structure_button.setEnabled(has_structure_set)
        self.delete_structure_button.setEnabled(has_selection)
        self.show_all_button.setEnabled(has_structure_set)
        self.hide_all_button.setEnabled(has_structure_set)


def test_structure_browser():
    """Test function for the structure browser."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    # Create a dummy structure set for testing
    class DummyStructure:
        def __init__(self, name, color=None, visible=True):
            self.name = name
            self.color = color
            self.visible = visible
            self.type = "ORGAN"
            self._volume = 100.0
        
        def get_volume(self):
            return self._volume
    
    class DummyStructureSet:
        def __init__(self):
            self.structures = []
        
        def add_structure(self, structure):
            self.structures.append(structure)
        
        def remove_structure(self, structure):
            if structure in self.structures:
                self.structures.remove(structure)
    
    app = QApplication(sys.argv)
    
    widget = StructureBrowser()
    
    # Create a sample structure set with structures
    structure_set = DummyStructureSet()
    structure_set.add_structure(DummyStructure("PTV", QColor(255, 0, 0)))
    structure_set.add_structure(DummyStructure("Lung Left", QColor(0, 255, 0)))
    structure_set.add_structure(DummyStructure("Lung Right", QColor(0, 0, 255)))
    structure_set.add_structure(DummyStructure("Heart", QColor(255, 0, 255)))
    structure_set.add_structure(DummyStructure("Spinal Cord", QColor(255, 255, 0)))
    
    widget.set_structure_set(structure_set)
    widget.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    test_structure_browser() 