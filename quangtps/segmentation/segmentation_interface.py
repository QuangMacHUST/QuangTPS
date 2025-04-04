"""
Segmentation Interface Module for QuangTPS.

This module provides an Eclipse-like interface for segmentation,
integrating the MPR viewer with various contouring tools.
"""

import logging
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    from PyQt5.QtCore import Qt, QObject, pyqtSignal, QPoint, QRect, QTimer
    from PyQt5.QtGui import QPainter, QPen, QColor, QCursor, QIcon, QPixmap
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
        QScrollArea, QFrame, QSplitter, QTabWidget, QGroupBox, QStackedWidget,
        QToolBar, QAction, QToolButton, QMenu, QDockWidget, QMessageBox,
        QSlider, QSpinBox, QCheckBox, QRadioButton, QButtonGroup
    )
except ImportError as e:
    logging.error(f"Unable to import PyQt5: {e}")

from quangtps.ui.mpr_viewer import MPRViewer
from quangtps.segmentation.manual_segmentation import (
    FreehandContourTool, FreehandMode, FreehandToolWidget,
    PolygonContourTool, PolygonMode, PolygonContourToolWidget,
    ThresholdContourTool, ThresholdMode, ThresholdOperation, ThresholdToolWidget
)
from quangtps.segmentation.auto.engine import AutoSegmentationEngine
from quangtps.segmentation.structures.structure import Structure
from quangtps.segmentation.structures.structure_set import StructureSet

logger = logging.getLogger(__name__)

class StructureItem(QWidget):
    """
    Widget representing a structure in the structure list.
    
    This class displays a structure with its name, color, and visibility controls,
    similar to Eclipse's structure list items.
    """
    
    # Signals for interaction
    structureSelected = pyqtSignal(Structure)
    structureVisibilityChanged = pyqtSignal(Structure, bool)
    structureColorChanged = pyqtSignal(Structure, str)
    structureTypeChanged = pyqtSignal(Structure, str)
    structureDeleted = pyqtSignal(Structure)
    
    def __init__(self, structure: Structure, parent=None):
        """Initialize the structure item widget."""
        super().__init__(parent)
        self.structure = structure
        self.selected = False
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Visibility checkbox
        self.visible_checkbox = QCheckBox()
        self.visible_checkbox.setChecked(True)
        self.visible_checkbox.toggled.connect(self.on_visibility_changed)
        layout.addWidget(self.visible_checkbox)
        
        # Color indicator
        self.color_indicator = QFrame()
        self.color_indicator.setFixedSize(16, 16)
        self.color_indicator.setStyleSheet(f"background-color: {self.structure.color}; border: 1px solid #888;")
        self.color_indicator.mousePressEvent = self.on_color_click
        layout.addWidget(self.color_indicator)
        
        # Structure name
        self.name_label = QLabel(self.structure.name)
        layout.addWidget(self.name_label, 1)  # 1 = stretch factor
        
        # Structure type indicator (PTV, OAR, etc.)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["PTV", "CTV", "GTV", "OAR", "BODY", "OTHER"])
        try:
            current_type = self.structure.type.upper() if hasattr(self.structure, 'type') else "OTHER"
            index = self.type_combo.findText(current_type)
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
        except Exception as e:
            logger.error(f"Error setting structure type: {e}")
        
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        layout.addWidget(self.type_combo)
        
        # Delete button
        self.delete_button = QPushButton("✕")
        self.delete_button.setFixedSize(20, 20)
        self.delete_button.clicked.connect(self.on_delete_clicked)
        layout.addWidget(self.delete_button)
        
        self.setLayout(layout)
        
    def on_visibility_changed(self, visible):
        """Handle visibility checkbox toggle."""
        self.structureVisibilityChanged.emit(self.structure, visible)
        
    def on_color_click(self, event):
        """Handle color indicator click for color selection."""
        from PyQt5.QtWidgets import QColorDialog
        
        current_color = QColor(self.structure.color)
        new_color = QColorDialog.getColor(current_color, self, "Select Color")
        
        if new_color.isValid():
            color_str = new_color.name()
            self.structure.color = color_str
            self.color_indicator.setStyleSheet(f"background-color: {color_str}; border: 1px solid #888;")
            self.structureColorChanged.emit(self.structure, color_str)
            
    def on_type_changed(self, type_str):
        """Handle structure type changes."""
        if hasattr(self.structure, 'type'):
            self.structure.type = type_str
            self.structureTypeChanged.emit(self.structure, type_str)
            
    def on_delete_clicked(self):
        """Handle delete button click."""
        self.structureDeleted.emit(self.structure)
        
    def set_selected(self, selected):
        """Set the selected state of this structure item."""
        self.selected = selected
        if selected:
            self.setStyleSheet("background-color: #0078D7; color: white;")
        else:
            self.setStyleSheet("")
            
    def mousePressEvent(self, event):
        """Handle mouse press to select the structure."""
        self.structureSelected.emit(self.structure)
        super().mousePressEvent(event)


class StructureListWidget(QScrollArea):
    """
    Widget for displaying and managing the list of structures.
    
    This class provides an Eclipse-like interface for structure management,
    including structure visibility, color, and selection.
    """
    
    # Signals for interaction
    structureSelected = pyqtSignal(Structure)
    
    def __init__(self, parent=None):
        """Initialize the structure list widget."""
        super().__init__(parent)
        self.structures = []
        self.structure_widgets = {}
        self.current_structure = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWidgetResizable(True)
        
        # Container widget
        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(1)
        self.layout.addStretch(1)  # Push all items to the top
        
        self.setWidget(container)
        
    def set_structure_set(self, structure_set: StructureSet):
        """Set the structure set to display."""
        # Clear existing structures
        self.clear_structures()
        
        # Add new structures
        if structure_set and hasattr(structure_set, 'structures'):
            for structure in structure_set.structures:
                self.add_structure(structure)
                
    def add_structure(self, structure: Structure):
        """Add a structure to the list."""
        if structure in self.structures:
            return
            
        self.structures.append(structure)
        
        # Create widget for this structure
        structure_widget = StructureItem(structure)
        structure_widget.structureSelected.connect(self.on_structure_selected)
        structure_widget.structureVisibilityChanged.connect(self.on_structure_visibility_changed)
        structure_widget.structureColorChanged.connect(self.on_structure_color_changed)
        structure_widget.structureTypeChanged.connect(self.on_structure_type_changed)
        structure_widget.structureDeleted.connect(self.on_structure_deleted)
        
        # Insert widget before the stretch item
        self.layout.insertWidget(self.layout.count() - 1, structure_widget)
        
        # Store reference
        self.structure_widgets[structure] = structure_widget
        
    def remove_structure(self, structure: Structure):
        """Remove a structure from the list."""
        if structure not in self.structures:
            return
            
        self.structures.remove(structure)
        
        # Remove widget
        if structure in self.structure_widgets:
            widget = self.structure_widgets[structure]
            self.layout.removeWidget(widget)
            widget.deleteLater()
            del self.structure_widgets[structure]
            
        # Update selection if needed
        if self.current_structure == structure:
            self.current_structure = None
            if self.structures:
                self.select_structure(self.structures[0])
                
    def clear_structures(self):
        """Clear all structures from the list."""
        for structure in list(self.structures):
            self.remove_structure(structure)
            
    def select_structure(self, structure: Structure):
        """Select a structure in the list."""
        if structure not in self.structures:
            return
            
        # Update selection
        if self.current_structure and self.current_structure in self.structure_widgets:
            self.structure_widgets[self.current_structure].set_selected(False)
            
        self.current_structure = structure
        
        if structure in self.structure_widgets:
            self.structure_widgets[structure].set_selected(True)
            
        # Emit signal
        self.structureSelected.emit(structure)
        
    def on_structure_selected(self, structure: Structure):
        """Handle structure selection from structure items."""
        self.select_structure(structure)
        
    def on_structure_visibility_changed(self, structure: Structure, visible: bool):
        """Handle structure visibility changes."""
        # This will be connected to the parent widget to update views
        pass
        
    def on_structure_color_changed(self, structure: Structure, color: str):
        """Handle structure color changes."""
        # This will be connected to the parent widget to update views
        pass
        
    def on_structure_type_changed(self, structure: Structure, type_str: str):
        """Handle structure type changes."""
        # This will be connected to the parent widget to update views
        pass
        
    def on_structure_deleted(self, structure: Structure):
        """Handle structure deletion."""
        reply = QMessageBox.question(
            self, 
            "Delete Structure",
            f"Are you sure you want to delete the structure '{structure.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.remove_structure(structure)


class ToolSelectorWidget(QTabWidget):
    """
    Widget for selecting and configuring contouring tools.
    
    This class provides a tabbed interface for selecting between
    different contouring tools similar to Eclipse's interface.
    """
    
    # Signal for tool selection
    toolSelected = pyqtSignal(str)  # Tool type: 'freehand', 'polygon', 'threshold', 'auto'
    
    def __init__(self, parent=None):
        """Initialize the tool selector widget."""
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Freehand tab
        self.freehand_widget = FreehandToolWidget()
        self.addTab(self.freehand_widget, "Brush")
        
        # Polygon tab
        self.polygon_widget = PolygonContourToolWidget()
        self.addTab(self.polygon_widget, "Polygon")
        
        # Threshold tab
        self.threshold_widget = ThresholdToolWidget()
        self.addTab(self.threshold_widget, "Threshold")
        
        # Auto-segmentation tab
        self.auto_widget = QWidget()
        auto_layout = QVBoxLayout(self.auto_widget)
        
        # AI model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Lung", "Prostate", "Head and Neck", "Abdomen"])
        model_layout.addWidget(self.model_combo)
        auto_layout.addLayout(model_layout)
        
        # Segmentation button
        self.segment_btn = QPushButton("Auto-Segment")
        auto_layout.addWidget(self.segment_btn)
        
        # Configuration options
        auto_layout.addWidget(QLabel("Options:"))
        self.use_existing_check = QCheckBox("Use existing structures as reference")
        auto_layout.addWidget(self.use_existing_check)
        self.refine_check = QCheckBox("Refine results automatically")
        auto_layout.addWidget(self.refine_check)
        
        auto_layout.addStretch(1)
        self.addTab(self.auto_widget, "Auto")
        
        # Connect signals
        self.currentChanged.connect(self.on_tab_changed)
        
    def on_tab_changed(self, index):
        """Handle tab selection changes."""
        tab_names = ["freehand", "polygon", "threshold", "auto"]
        if 0 <= index < len(tab_names):
            self.toolSelected.emit(tab_names[index])
            
    def get_freehand_tool_widget(self) -> FreehandToolWidget:
        """Get the freehand tool widget."""
        return self.freehand_widget
        
    def get_polygon_tool_widget(self) -> PolygonContourToolWidget:
        """Get the polygon tool widget."""
        return self.polygon_widget
        
    def get_threshold_tool_widget(self) -> ThresholdToolWidget:
        """Get the threshold tool widget."""
        return self.threshold_widget


class SegmentationInterface(QWidget):
    """
    Main segmentation interface widget integrating all contouring tools with MPR viewer.
    
    This class provides an Eclipse-like interface for segmentation,
    combining structure management, MPR viewing, and contouring tools.
    """
    
    def __init__(self, parent=None):
        """Initialize the segmentation interface."""
        super().__init__(parent)
        self.init_ui()
        self.current_tool = None
        self.current_structure = None
        self.structure_set = None
        self.image_data = None
        self.contour_tools = {}
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout(self)
        
        # Left panel - Structure list and tools
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Structure list
        structures_group = QGroupBox("Structures")
        structures_layout = QVBoxLayout(structures_group)
        
        # Add/Create structure buttons
        structure_buttons_layout = QHBoxLayout()
        self.add_structure_btn = QPushButton("Add")
        self.add_structure_btn.clicked.connect(self.on_add_structure)
        structure_buttons_layout.addWidget(self.add_structure_btn)
        
        self.create_from_roi_btn = QPushButton("Create from ROI")
        self.create_from_roi_btn.clicked.connect(self.on_create_from_roi)
        structure_buttons_layout.addWidget(self.create_from_roi_btn)
        
        structures_layout.addLayout(structure_buttons_layout)
        
        # Structure list widget
        self.structure_list = StructureListWidget()
        self.structure_list.structureSelected.connect(self.on_structure_selected)
        structures_layout.addWidget(self.structure_list)
        
        left_layout.addWidget(structures_group)
        
        # Tools section
        tools_group = QGroupBox("Contouring Tools")
        tools_layout = QVBoxLayout(tools_group)
        
        self.tool_selector = ToolSelectorWidget()
        self.tool_selector.toolSelected.connect(self.on_tool_selected)
        tools_layout.addWidget(self.tool_selector)
        
        left_layout.addWidget(tools_group)
        
        # Set a reasonable width for the left panel
        left_panel.setFixedWidth(300)
        main_layout.addWidget(left_panel)
        
        # Right panel - MPR viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.mpr_viewer = MPRViewer()
        right_layout.addWidget(self.mpr_viewer)
        
        main_layout.addWidget(right_panel, 1)  # 1 = stretch factor
        
        self.setLayout(main_layout)
        
    def set_image_data(self, image_data):
        """Set the image data to display and use for contouring."""
        self.image_data = image_data
        self.mpr_viewer.set_image_data(image_data)
        
        # Update threshold tool with image data
        if 'threshold' in self.contour_tools:
            self.contour_tools['threshold'].set_image_data(image_data)
            
    def set_structure_set(self, structure_set: StructureSet):
        """Set the structure set to use for contouring."""
        self.structure_set = structure_set
        self.structure_list.set_structure_set(structure_set)
        self.mpr_viewer.set_structure_set(structure_set)
        
        # Set initial structure if available
        if structure_set and hasattr(structure_set, 'structures') and structure_set.structures:
            self.on_structure_selected(structure_set.structures[0])
            
    def on_structure_selected(self, structure: Structure):
        """Handle structure selection."""
        self.current_structure = structure
        
        # Update all tools with the new structure
        for tool in self.contour_tools.values():
            tool.set_structure(structure)
            
    def on_tool_selected(self, tool_type: str):
        """Handle tool selection."""
        # Deactivate current tool
        if self.current_tool:
            self.current_tool.deactivate()
            
        # Activate selected tool
        if tool_type in self.contour_tools:
            self.current_tool = self.contour_tools[tool_type]
            self.current_tool.activate()
        else:
            self.current_tool = None
            
    def initialize_contour_tools(self):
        """Initialize all contouring tools."""
        # Create tools
        self.contour_tools = {
            'freehand': FreehandContourTool(self.mpr_viewer),
            'polygon': PolygonContourTool(self.mpr_viewer),
            'threshold': ThresholdContourTool(self.mpr_viewer, image_data=self.image_data)
        }
        
        # Set current structure for all tools
        if self.current_structure:
            for tool in self.contour_tools.values():
                tool.set_structure(self.current_structure)
                
        # Connect tool widgets to tools
        self.tool_selector.get_freehand_tool_widget().set_freehand_tool(self.contour_tools['freehand'])
        self.tool_selector.get_polygon_tool_widget().set_polygon_tool(self.contour_tools['polygon'])
        self.tool_selector.get_threshold_tool_widget().set_threshold_tool(self.contour_tools['threshold'])
        
        # Set freehand tool as default
        self.on_tool_selected('freehand')
        
    def on_add_structure(self):
        """Handle add structure button click."""
        from PyQt5.QtWidgets import QInputDialog
        
        # Get structure name
        name, ok = QInputDialog.getText(self, "Add Structure", "Structure name:")
        if ok and name:
            # Check if structure with this name already exists
            if self.structure_set:
                for structure in self.structure_set.structures:
                    if structure.name == name:
                        QMessageBox.warning(self, "Warning", f"Structure '{name}' already exists.")
                        return
                
                # Create new structure
                try:
                    structure = Structure(name=name)
                    self.structure_set.add_structure(structure)
                    self.structure_list.add_structure(structure)
                    self.on_structure_selected(structure)
                except Exception as e:
                    logger.error(f"Error adding structure: {e}")
                    QMessageBox.warning(self, "Error", f"Failed to add structure: {e}")
            else:
                QMessageBox.warning(self, "Warning", "No structure set available.")
                
    def on_create_from_roi(self):
        """Handle create from ROI button click."""
        # This would use a more complex dialog to create structure from user-drawn ROI
        # For now, just show a placeholder message
        QMessageBox.information(self, "Create from ROI", "This feature will allow creating structures from ROIs.")

def test_segmentation_interface():
    """Test function for the segmentation interface."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create test image data
    class TestImage:
        def __init__(self):
            self.data = np.zeros((100, 512, 512))
            # Add some test patterns
            for z in range(100):
                # Circular pattern that varies with slice
                center_x, center_y = 256, 256
                radius = 100 + z
                for x in range(512):
                    for y in range(512):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist < radius:
                            self.data[z, y, x] = 100 + z
            
            self.shape = self.data.shape
            self.spacing = (1.0, 1.0, 1.0)  # 1mm spacing
        
        def __getitem__(self, indices):
            return self.data[indices]
    
    # Create test structures
    class TestStructure(Structure):
        def __init__(self, name, color, type_str="OTHER"):
            self.name = name
            self.color = color
            self.type = type_str
            self.visible = True
            # Placeholder for contour data
            self.contours = {}
            
        def add_contour(self, points, slice_index, orientation):
            if orientation not in self.contours:
                self.contours[orientation] = {}
            if slice_index not in self.contours[orientation]:
                self.contours[orientation][slice_index] = []
            self.contours[orientation][slice_index].append(points)
    
    # Create test structure set
    class TestStructureSet(StructureSet):
        def __init__(self):
            self.structures = []
            
        def add_structure(self, structure):
            self.structures.append(structure)
    
    # Create test data
    image_data = TestImage()
    structure_set = TestStructureSet()
    
    # Add some test structures
    structure_set.add_structure(TestStructure("PTV", "#FF0000", "PTV"))
    structure_set.add_structure(TestStructure("CTV", "#00FF00", "CTV"))
    structure_set.add_structure(TestStructure("Lung_L", "#0000FF", "OAR"))
    structure_set.add_structure(TestStructure("Lung_R", "#FF00FF", "OAR"))
    structure_set.add_structure(TestStructure("Heart", "#FFFF00", "OAR"))
    
    # Create interface
    interface = SegmentationInterface()
    interface.set_image_data(image_data)
    interface.set_structure_set(structure_set)
    interface.initialize_contour_tools()
    
    interface.setWindowTitle("QuangTPS - Segmentation Interface")
    interface.resize(1200, 800)
    interface.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_segmentation_interface() 