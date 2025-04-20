#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab Structure/Contour cho QuangTPS.

Module này triển khai giao diện Structure tab tương tự Eclipse của Varian,
cho phép người dùng vẽ, quản lý và chỉnh sửa structure và contour.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QDialog,
    QColorDialog,
    QComboBox,
    QLineEdit,
    QFormLayout,
    QMessageBox,
    QFileDialog,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QProgressDialog,
    QMenu,
    QAction,
    QToolBar,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QToolButton,
    QFrame,
    QScrollArea,
    QGridLayout,
    QInputDialog,
    QStackedWidget,
    QAbstractItemView,
    QSizePolicy,
)
from PyQt5.QtGui import QColor, QIcon, QBrush, QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QTimer

try:
    from quangtps.segmentation.structures.structure import (
        Structure,
        StructureType,
        StructurePriority,
    )
    from quangtps.segmentation.structures.structure_set import StructureSet
    from quangtps.segmentation.contour.contour_manager import ContourManager
    from quangtps.segmentation.contour.polygon_tool import PolygonTool
    from quangtps.segmentation.contour.margin import MarginTool
    from quangtps.segmentation.contour.boolean_operations import BooleanOperator
    from quangtps.segmentation.contour.interpolation import ContourInterpolator
    from quangtps.segmentation.contour.advanced_editing import AdvancedContourEditor
    from quangtps.segmentation.auto_segmentation.semi_automatic import (
        SemiAutomaticSegmentation,
    )
    from quangtps.segmentation.auto.engine import AutoSegmentationEngine
    from quangtps.ui.image_display import ImageDisplay
    from quangtps.imaging.image import Image
    from quangtps.core.patient import Patient
    from quangtps.core.services import ServiceRegistry
    from quangtps.ui.mpr_viewer import MPRViewer
    from quangtps.segmentation.segmentation_interface import SegmentationInterface
    from quangtps.segmentation.manual_segmentation.brush_tool import (
        BrushTool,
        BrushToolWidget,
    )
    from quangtps.segmentation.manual_segmentation.threshold_tool import (
        ThresholdContourTool,
        ThresholdToolWidget,
        ThresholdMode,
        ThresholdOperation,
    )
except ImportError as e:
    logging.error(f"Error importing structure modules: {e}")

logger = logging.getLogger(__name__)


class ObjectExplorerPanel(QWidget):
    """
    Panel displaying available objects (patients, images, structures, etc.)

    This class provides an Eclipse-like object explorer panel with hierarchical
    organization of patient data, images, structure sets, and plans.
    """

    # Signals
    patientSelected = pyqtSignal(Patient)
    imageSelected = pyqtSignal(object)  # Image object
    structureSetSelected = pyqtSignal(StructureSet)
    planSelected = pyqtSignal(object)  # Plan object

    def __init__(self, parent=None):
        """Initialize the object explorer panel."""
        super().__init__(parent)
        self.init_ui()
        self.current_patient = None

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Object Explorer")
        header.setStyleSheet(
            "font-weight: bold; background-color: #0078D7; color: white; padding: 5px;"
        )
        layout.addWidget(header)

        # Tree-like structure using expandable sections

        # Patients section
        self.patients_group = QGroupBox("Patients")
        patients_layout = QVBoxLayout(self.patients_group)
        self.patients_container = QWidget()
        self.patients_layout = QVBoxLayout(self.patients_container)
        self.patients_layout.setContentsMargins(0, 0, 0, 0)
        self.patients_layout.setSpacing(1)
        patients_scroll = QScrollArea()
        patients_scroll.setWidgetResizable(True)
        patients_scroll.setWidget(self.patients_container)
        patients_layout.addWidget(patients_scroll)
        layout.addWidget(self.patients_group)

        # Images section
        self.images_group = QGroupBox("Images")
        images_layout = QVBoxLayout(self.images_group)
        self.images_container = QWidget()
        self.images_layout = QVBoxLayout(self.images_container)
        self.images_layout.setContentsMargins(0, 0, 0, 0)
        self.images_layout.setSpacing(1)
        images_scroll = QScrollArea()
        images_scroll.setWidgetResizable(True)
        images_scroll.setWidget(self.images_container)
        images_layout.addWidget(images_scroll)
        layout.addWidget(self.images_group)

        # Structure Sets section
        self.structure_sets_group = QGroupBox("Structure Sets")
        structure_sets_layout = QVBoxLayout(self.structure_sets_group)
        self.structure_sets_container = QWidget()
        self.structure_sets_layout = QVBoxLayout(self.structure_sets_container)
        self.structure_sets_layout.setContentsMargins(0, 0, 0, 0)
        self.structure_sets_layout.setSpacing(1)
        structure_sets_scroll = QScrollArea()
        structure_sets_scroll.setWidgetResizable(True)
        structure_sets_scroll.setWidget(self.structure_sets_container)
        structure_sets_layout.addWidget(structure_sets_scroll)
        layout.addWidget(self.structure_sets_group)

        # Plans section
        self.plans_group = QGroupBox("Plans")
        plans_layout = QVBoxLayout(self.plans_group)
        self.plans_container = QWidget()
        self.plans_layout = QVBoxLayout(self.plans_container)
        self.plans_layout.setContentsMargins(0, 0, 0, 0)
        self.plans_layout.setSpacing(1)
        plans_scroll = QScrollArea()
        plans_scroll.setWidgetResizable(True)
        plans_scroll.setWidget(self.plans_container)
        plans_layout.addWidget(plans_scroll)
        layout.addWidget(self.plans_group)

        self.setLayout(layout)

    def set_patient(self, patient: Patient):
        """Set the current patient and update displayed objects."""
        self.current_patient = patient
        self.update_displayed_objects()

    def update_displayed_objects(self):
        """Update all displayed objects based on the current patient."""
        self._clear_layout(self.patients_layout)
        self._clear_layout(self.images_layout)
        self._clear_layout(self.structure_sets_layout)
        self._clear_layout(self.plans_layout)

        if not self.current_patient:
            return

        # Add patient item
        patient_item = self._create_item(
            self.current_patient.name,
            f"ID: {self.current_patient.id}",
            is_selected=True,
            item_type="patient",
            item_data=self.current_patient,
        )
        self.patients_layout.addWidget(patient_item)

        # Add image items if available
        try:
            patient_db = ServiceRegistry.get_service("PatientDB")
            if patient_db:
                # Get images for this patient
                images = patient_db.get_images_for_patient(self.current_patient.id)
                if images:
                    for image in images:
                        image_item = self._create_item(
                            image.description
                            if hasattr(image, "description")
                            else "Image",
                            f"Series: {image.series_id if hasattr(image, 'series_id') else 'N/A'}",
                            is_selected=False,
                            item_type="image",
                            item_data=image,
                        )
                        self.images_layout.addWidget(image_item)

                # Get structure sets for this patient
                structure_sets = patient_db.get_structure_sets_for_patient(
                    self.current_patient.id
                )
                if structure_sets:
                    for ss in structure_sets:
                        ss_item = self._create_item(
                            ss.name if hasattr(ss, "name") else "Structure Set",
                            f"ID: {ss.id if hasattr(ss, 'id') else 'N/A'}",
                            is_selected=False,
                            item_type="structure_set",
                            item_data=ss,
                        )
                        self.structure_sets_layout.addWidget(ss_item)

                # Get plans for this patient
                plans = patient_db.get_plans_for_patient(self.current_patient.id)
                if plans:
                    for plan in plans:
                        plan_item = self._create_item(
                            plan.name if hasattr(plan, "name") else "Plan",
                            f"ID: {plan.id if hasattr(plan, 'id') else 'N/A'}",
                            is_selected=False,
                            item_type="plan",
                            item_data=plan,
                        )
                        self.plans_layout.addWidget(plan_item)
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")

    def _create_item(
        self,
        title: str,
        subtitle: str,
        is_selected: bool = False,
        item_type: str = "",
        item_data: Any = None,
    ) -> QWidget:
        """Create an item widget for the object explorer."""
        item = QWidget()
        layout = QVBoxLayout(item)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)

        # Subtitle
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet("font-size: 8pt; color: #666;")
            layout.addWidget(subtitle_label)

        item.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 3px;"
        )
        if is_selected:
            item.setStyleSheet(
                "background-color: #0078D7; color: white; border: 1px solid #0078D7; border-radius: 3px;"
            )

        # Store data
        item.setProperty("item_type", item_type)
        item.setProperty("item_data", item_data)

        # Connect mouse events
        item.mousePressEvent = lambda e, i=item: self._on_item_clicked(i)

        return item

    def _on_item_clicked(self, item: QWidget):
        """Handle item click in the object explorer."""
        item_type = item.property("item_type")
        item_data = item.property("item_data")

        if item_type == "patient":
            self.patientSelected.emit(item_data)
        elif item_type == "image":
            self.imageSelected.emit(item_data)
        elif item_type == "structure_set":
            self.structureSetSelected.emit(item_data)
        elif item_type == "plan":
            self.planSelected.emit(item_data)

        # Update selection visuals
        self._update_selection_for_type(item_type, item)

    def _update_selection_for_type(self, item_type: str, selected_item: QWidget):
        """Update visual selection state for items of a specific type."""
        container_layout = None
        if item_type == "patient":
            container_layout = self.patients_layout
        elif item_type == "image":
            container_layout = self.images_layout
        elif item_type == "structure_set":
            container_layout = self.structure_sets_layout
        elif item_type == "plan":
            container_layout = self.plans_layout

        if container_layout:
            # Update all items of this type
            for i in range(container_layout.count()):
                item = container_layout.itemAt(i).widget()
                if item:
                    if item == selected_item:
                        item.setStyleSheet(
                            "background-color: #0078D7; color: white; border: 1px solid #0078D7; border-radius: 3px;"
                        )
                    else:
                        item.setStyleSheet(
                            "background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 3px;"
                        )

    def _clear_layout(self, layout):
        """Clear all widgets from a layout."""
        if layout is None:
            return

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()


class StructureTab(QWidget):
    """
    Structure tab for the QuangTPS application.

    This tab provides an interface for structure management, including
    creation, editing, and visualization of structures for treatment planning.
    """

    # Signals
    structureSetChanged = pyqtSignal(StructureSet)
    structureSelectionChanged = pyqtSignal(Structure)
    structureAdded = pyqtSignal(Structure)
    structureRemoved = pyqtSignal(Structure)
    structureModified = pyqtSignal(Structure)
    structureVisibilityChanged = pyqtSignal(Structure, bool)
    windowClosed = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the structure tab."""
        super().__init__(parent)

        # Internal state
        self.image = None
        self.structure_set = None
        self.selected_structure = None

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create a splitter for the sidebar and content
        splitter = QSplitter(Qt.Horizontal)

        # Left sidebar for structure management
        self.sidebar = self.create_structure_sidebar()
        splitter.addWidget(self.sidebar)

        # Right content area for tools and viewer
        self.content = self.create_content_area()
        splitter.addWidget(self.content)

        # Set initial sizes (30% sidebar, 70% content)
        splitter.setSizes([300, 700])

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # Status bar at the bottom
        status_bar = QWidget()
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(5, 2, 5, 2)

        self.status_label = QLabel("No image loaded")
        status_layout.addWidget(self.status_label)

        main_layout.addWidget(status_bar)

        # Set up connections between widgets
        self.setup_connections()

        # Apply Eclipse-like styling
        self.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }

            QListWidget {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }

            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #e0e0e0;
            }

            QListWidget::item:selected {
                background-color: #2070c0;
                color: white;
            }

            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px 10px;
            }

            QPushButton:hover {
                background-color: #e0e0e0;
            }

            QPushButton:pressed {
                background-color: #d0d0d0;
            }

            QToolBar {
                background-color: #e0e0e0;
                border: none;
                spacing: 3px;
            }

            QLabel {
                color: #404040;
            }
        """)

    def create_structure_sidebar(self):
        """Create the structure management sidebar."""
        sidebar = QFrame()
        sidebar.setFrameShape(QFrame.StyledPanel)
        sidebar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sidebar.setMinimumWidth(250)
        sidebar.setMaximumWidth(400)

        layout = QVBoxLayout(sidebar)

        # Structure set header
        struct_set_header = QWidget()
        struct_set_layout = QHBoxLayout(struct_set_header)
        struct_set_layout.setContentsMargins(0, 0, 0, 0)

        struct_set_label = QLabel("Structure Set:")
        struct_set_layout.addWidget(struct_set_label)

        self.struct_set_name = QLabel("None")
        self.struct_set_name.setStyleSheet("font-weight: bold;")
        struct_set_layout.addWidget(self.struct_set_name)

        struct_set_layout.addStretch(1)

        layout.addWidget(struct_set_header)

        # Structure list widget
        self.structure_list = QListWidget()
        self.structure_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.structure_list.customContextMenuRequested.connect(
            self.show_structure_context_menu
        )
        layout.addWidget(self.structure_list)

        # Structure action buttons
        action_buttons = QWidget()
        action_layout = QHBoxLayout(action_buttons)
        action_layout.setContentsMargins(0, 0, 0, 0)

        self.add_structure_btn = QPushButton("Add")
        self.add_structure_btn.setIcon(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "add.png"))
        )
        self.add_structure_btn.clicked.connect(self.add_new_structure)
        action_layout.addWidget(self.add_structure_btn)

        self.delete_structure_btn = QPushButton("Delete")
        self.delete_structure_btn.setIcon(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "delete.png"))
        )
        self.delete_structure_btn.clicked.connect(self.delete_selected_structure)
        self.delete_structure_btn.setEnabled(False)
        action_layout.addWidget(self.delete_structure_btn)

        self.edit_structure_btn = QPushButton("Properties")
        self.edit_structure_btn.setIcon(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "edit.png"))
        )
        self.edit_structure_btn.clicked.connect(self.edit_structure_properties)
        self.edit_structure_btn.setEnabled(False)
        action_layout.addWidget(self.edit_structure_btn)

        layout.addWidget(action_buttons)

        # Structure properties
        prop_group = QGroupBox("Structure Properties")
        prop_layout = QVBoxLayout(prop_group)

        self.prop_name = QLabel("Name: -")
        self.prop_type = QLabel("Type: -")
        self.prop_color = QLabel("Color: -")
        self.prop_volume = QLabel("Volume: - cc")

        prop_layout.addWidget(self.prop_name)
        prop_layout.addWidget(self.prop_type)
        prop_layout.addWidget(self.prop_color)
        prop_layout.addWidget(self.prop_volume)

        layout.addWidget(prop_group)

        # Add stretch to push content to the top
        layout.addStretch(1)

        return sidebar

    def create_content_area(self):
        """Create the content area with tools and viewer."""
        content = QFrame()
        content.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title for the content area
        title = QLabel("Structure Segmentation Tools")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Integrated segmentation interface
        self.segmentation_interface = SegmentationInterface()
        self.segmentation_interface.structureModified.connect(
            self.on_structure_modified
        )
        layout.addWidget(self.segmentation_interface)

        # Add quick actions toolbar
        quick_actions = QToolBar()

        # Add "Auto Segment" action
        auto_segment_action = QAction("Auto Segment", self)
        auto_segment_action.setIcon(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "auto_segment.png"))
        )
        auto_segment_action.triggered.connect(self.auto_segment)
        quick_actions.addAction(auto_segment_action)

        # Add "Smart Brush" action
        smart_brush_action = QAction("Smart Brush", self)
        smart_brush_action.setIcon(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "smart_brush.png"))
        )
        smart_brush_action.triggered.connect(
            lambda: self.segmentation_interface.set_current_tool("smart_brush")
        )
        quick_actions.addAction(smart_brush_action)

        # Add "Copy to all slices" action
        copy_slices_action = QAction("Copy to All Slices", self)
        copy_slices_action.setIcon(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "copy_slices.png"))
        )
        copy_slices_action.triggered.connect(self.copy_to_all_slices)
        quick_actions.addAction(copy_slices_action)

        # Add toolbar to layout
        layout.addWidget(quick_actions)

        # Add instructions
        instructions = QLabel(
            "Select a structure from the list and use the tools above to contour. "
            "Right-click for additional options."
        )
        instructions.setStyleSheet("font-style: italic; color: #606060; padding: 5px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        return content

    def setup_connections(self):
        """Set up connections between widgets."""
        self.structure_list.currentItemChanged.connect(
            self.on_structure_selection_changed
        )

    def set_image(self, image):
        """Set the current image for structure editing."""
        self.image = image

        if image:
            # Update status bar
            dimensions = f"{image.shape[0]}x{image.shape[1]}x{image.shape[2]}"
            self.status_label.setText(f"Image loaded - Size: {dimensions}")

            # Reset or create a new structure set if needed
            if not self.structure_set:
                self.structure_set = StructureSet()
                self.structure_set.name = "RTStruct"
                self.structure_set.image_ref = image
                self.structureSetChanged.emit(self.structure_set)

            # Update structure set name
            self.struct_set_name.setText(self.structure_set.name)

            # Set image in segmentation interface
            self.segmentation_interface.set_image_data(image)

            # Enable the add structure button
            self.add_structure_btn.setEnabled(True)
        else:
            # Clear UI if no image
            self.status_label.setText("No image loaded")
            self.add_structure_btn.setEnabled(False)
            self.delete_structure_btn.setEnabled(False)
            self.edit_structure_btn.setEnabled(False)

    def set_structure_set(self, structure_set):
        """Set the structure set for editing."""
        self.structure_set = structure_set

        if structure_set:
            # Update structure set name
            self.struct_set_name.setText(self.structure_set.name)

            # Clear structure list
            self.structure_list.clear()

            # Add structures to list
            for structure in structure_set.structures:
                self.add_structure_to_list(structure)

            # Enable the add structure button
            self.add_structure_btn.setEnabled(True)

            # Emit signal
            self.structureSetChanged.emit(structure_set)
        else:
            # Clear UI if no structure set
            self.structure_list.clear()
            self.struct_set_name.setText("None")
            self.add_structure_btn.setEnabled(False)
            self.delete_structure_btn.setEnabled(False)
            self.edit_structure_btn.setEnabled(False)

        # Clear selected structure
        self.selected_structure = None
        self.update_property_display()

    def add_structure_to_list(self, structure):
        """Add a structure to the list widget."""
        if not structure:
            return

        item = QListWidgetItem(structure.name)

        # Set color indicator (20x20 pixel square)
        pixmap = QPixmap(20, 20)
        pixmap.fill(QColor(*structure.color))
        item.setIcon(QIcon(pixmap))

        # Store the structure reference
        item.setData(Qt.UserRole, structure)

        # Add to list widget
        self.structure_list.addItem(item)

    def add_new_structure(self):
        """Add a new structure to the structure set."""
        if not self.structure_set:
            return

        # Get structure name
        name, ok = QInputDialog.getText(
            self,
            "New Structure",
            "Enter structure name:",
            text=f"Structure {len(self.structure_set.structures) + 1}",
        )

        if not ok or not name:
            return

        # Select color
        color_dialog = QColorDialog(self)
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)

        # Set initial color (cycle through some presets)
        preset_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 0),  # Dark Green
            (0, 128, 255),  # Light Blue
        ]

        index = len(self.structure_set.structures) % len(preset_colors)
        color_dialog.setCurrentColor(QColor(*preset_colors[index]))

        if color_dialog.exec_():
            qcolor = color_dialog.currentColor()
            color = (qcolor.red(), qcolor.green(), qcolor.blue())
        else:
            color = preset_colors[index]

        # Create new structure
        structure = Structure()
        structure.name = name
        structure.color = color
        structure.image_ref = self.image

        # Add to structure set
        self.structure_set.add_structure(structure)

        # Add to list widget
        self.add_structure_to_list(structure)

        # Select the new structure
        for i in range(self.structure_list.count()):
            item = self.structure_list.item(i)
            if item.data(Qt.UserRole) == structure:
                self.structure_list.setCurrentItem(item)
                break

        # Emit signal
        self.structureAdded.emit(structure)

    def on_structure_selection_changed(self, current, previous):
        """Handle structure selection change in list."""
        if current:
            structure = current.data(Qt.UserRole)
            self.selected_structure = structure

            # Update UI
            self.delete_structure_btn.setEnabled(True)
            self.edit_structure_btn.setEnabled(True)

            # Update property display
            self.update_property_display()

            # Set the current structure in segmentation interface
            self.segmentation_interface.set_structure(structure)

            # Emit signal
            self.structureSelectionChanged.emit(structure)
        else:
            self.selected_structure = None

            # Update UI
            self.delete_structure_btn.setEnabled(False)
            self.edit_structure_btn.setEnabled(False)

            # Update property display
            self.update_property_display()

            # Clear structure in segmentation interface
            self.segmentation_interface.set_structure(None)

    def update_property_display(self):
        """Update the property display for the selected structure."""
        if self.selected_structure:
            self.prop_name.setText(f"Name: {self.selected_structure.name}")
            self.prop_type.setText(f"Type: {self.selected_structure.type}")

            color_text = f"RGB({self.selected_structure.color[0]}, {self.selected_structure.color[1]}, {self.selected_structure.color[2]})"
            self.prop_color.setText(f"Color: {color_text}")

            # Calculate volume if possible
            if self.image:
                voxel_size = (
                    self.image.voxel_size
                    if hasattr(self.image, "voxel_size")
                    else (1.0, 1.0, 1.0)
                )
                voxel_volume = (
                    voxel_size[0] * voxel_size[1] * voxel_size[2] / 1000
                )  # Convert to cc
                structure_volume = self.selected_structure.get_volume(voxel_volume)
                self.prop_volume.setText(f"Volume: {structure_volume:.2f} cc")
            else:
                self.prop_volume.setText("Volume: - cc")
        else:
            self.prop_name.setText("Name: -")
            self.prop_type.setText("Type: -")
            self.prop_color.setText("Color: -")
            self.prop_volume.setText("Volume: - cc")

    def delete_selected_structure(self):
        """Delete the currently selected structure."""
        if not self.selected_structure:
            return

        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the structure '{self.selected_structure.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if confirm != QMessageBox.Yes:
            return

        # Remove from structure set
        self.structure_set.remove_structure(self.selected_structure)

        # Get current selection
        current_row = self.structure_list.currentRow()

        # Remove from list widget
        self.structure_list.takeItem(current_row)

        # Emit signal
        self.structureRemoved.emit(self.selected_structure)

        # Clear selected structure
        self.selected_structure = None

        # Update property display
        self.update_property_display()

    def edit_structure_properties(self):
        """Edit properties of the selected structure."""
        if not self.selected_structure:
            return

        # Get new name
        name, ok = QInputDialog.getText(
            self, "Edit Structure", "Structure name:", text=self.selected_structure.name
        )

        if not ok:
            return

        if name and name != self.selected_structure.name:
            self.selected_structure.name = name

            # Update list widget
            current_item = self.structure_list.currentItem()
            current_item.setText(name)

        # Select new color
        color_dialog = QColorDialog(self)
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)
        color_dialog.setCurrentColor(QColor(*self.selected_structure.color))

        if color_dialog.exec_():
            qcolor = color_dialog.currentColor()
            self.selected_structure.color = (
                qcolor.red(),
                qcolor.green(),
                qcolor.blue(),
            )

            # Update color indicator in list
            current_item = self.structure_list.currentItem()
            pixmap = QPixmap(20, 20)
            pixmap.fill(QColor(*self.selected_structure.color))
            current_item.setIcon(QIcon(pixmap))

        # Update property display
        self.update_property_display()

        # Emit signal
        self.structureModified.emit(self.selected_structure)

    def show_structure_context_menu(self, position):
        """Show context menu for structures."""
        if not self.structure_list.count():
            return

        selected_item = self.structure_list.itemAt(position)
        if not selected_item:
            return

        # Create context menu
        context_menu = QMenu(self)

        # Add actions
        edit_action = QAction("Edit Properties", self)
        edit_action.triggered.connect(self.edit_structure_properties)
        context_menu.addAction(edit_action)

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self.delete_selected_structure)
        context_menu.addAction(delete_action)

        context_menu.addSeparator()

        # Add visibility toggle
        structure = selected_item.data(Qt.UserRole)

        visibility_action = QAction("Show/Hide", self)
        visibility_action.setCheckable(True)
        visibility_action.setChecked(structure.visible)
        visibility_action.triggered.connect(
            lambda checked, s=structure: self.toggle_structure_visibility(s, checked)
        )
        context_menu.addAction(visibility_action)

        # Add copy/paste options
        context_menu.addSeparator()

        copy_action = QAction("Copy to Next Slice", self)
        copy_action.triggered.connect(lambda: self.copy_to_next_slice(structure))
        context_menu.addAction(copy_action)

        copy_all_action = QAction("Copy to All Slices", self)
        copy_all_action.triggered.connect(self.copy_to_all_slices)
        context_menu.addAction(copy_all_action)

        # Show the menu
        context_menu.exec_(self.structure_list.mapToGlobal(position))

    def toggle_structure_visibility(self, structure, visible):
        """Toggle the visibility of a structure."""
        structure.visible = visible
        self.structureVisibilityChanged.emit(structure, visible)

    def copy_to_next_slice(self, structure=None):
        """Copy the current structure contour to the next slice."""
        if not structure:
            structure = self.selected_structure

        if not structure:
            return

        # This would be implemented to copy the current slice's contour
        # to the next slice for the selected structure
        QMessageBox.information(
            self,
            "Feature Not Implemented",
            "Copy to next slice feature is not yet implemented.",
        )

    def copy_to_all_slices(self):
        """Copy the current structure contour to all slices."""
        if not self.selected_structure:
            return

        # This would be implemented to copy the current slice's contour
        # to all slices for the selected structure
        QMessageBox.information(
            self,
            "Feature Not Implemented",
            "Copy to all slices feature is not yet implemented.",
        )

    def auto_segment(self):
        """Perform auto-segmentation for the selected structure."""
        if not self.selected_structure or not self.image:
            return

        # This would be implemented to perform automatic segmentation
        # for the selected structure using image data
        QMessageBox.information(
            self,
            "Feature Not Implemented",
            "Auto-segmentation feature is not yet implemented.",
        )

    def on_structure_modified(self):
        """Handle structure modification from the segmentation interface."""
        if self.selected_structure:
            # Update property display
            self.update_property_display()

            # Emit signal
            self.structureModified.emit(self.selected_structure)

    def handle_mouse_event(self, event_type, point, slice_index=None, orientation=None):
        """Handle mouse events from the image viewer."""
        if not self.selected_structure:
            return False

        if event_type == "press":
            return self.segmentation_interface.handle_mouse_press(
                point, slice_index, orientation
            )
        elif event_type == "move":
            return self.segmentation_interface.handle_mouse_move(point)
        elif event_type == "release":
            return self.segmentation_interface.handle_mouse_release(point)

        return False

    def get_cursor_for_viewer(self):
        """Get the cursor for the current tool."""
        return self.segmentation_interface.get_cursor_for_viewer()

    def get_overlay_for_viewer(self, orientation, slice_index):
        """Get overlay for display in the viewer."""
        return self.segmentation_interface.get_overlay_for_viewer(
            orientation, slice_index
        )

    def closeEvent(self, event):
        """Handle the close event."""
        self.windowClosed.emit()
        super().closeEvent(event)


def test_structure_tab():
    """Test function for the structure tab."""
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)

    # Create main window
    main_window = QMainWindow()

    # Create test patient
    class TestPatient(Patient):
        def __init__(self, id, name):
            self.id = id
            self.name = name

    # Create test image
    class TestImage:
        def __init__(self):
            self.id = "test_image_1"
            self.description = "Test CT Image"
            self.series_id = "series_1"
            self.data = np.zeros((100, 512, 512))
            # Add some test patterns
            for z in range(100):
                # Circular pattern that varies with slice
                center_x, center_y = 256, 256
                radius = 100 + z
                for x in range(512):
                    for y in range(512):
                        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        if dist < radius:
                            self.data[z, y, x] = 100 + z

            self.shape = self.data.shape
            self.spacing = (1.0, 1.0, 1.0)  # 1mm spacing

        def __getitem__(self, indices):
            return self.data[indices]

    # Create mock PatientDB for testing
    class TestPatientDB:
        def __init__(self):
            self.patients = {}
            self.images = {}
            self.structure_sets = {}
            self.plans = {}

        def get_images_for_patient(self, patient_id):
            return [img for img in self.images.values() if img.patient_id == patient_id]

        def get_structure_sets_for_patient(self, patient_id):
            return [
                ss for ss in self.structure_sets.values() if ss.patient_id == patient_id
            ]

        def get_plans_for_patient(self, patient_id):
            return [
                plan for plan in self.plans.values() if plan.patient_id == patient_id
            ]

        def add_structure_set(self, structure_set):
            ss_id = f"ss_{len(self.structure_sets) + 1}"
            structure_set.id = ss_id
            self.structure_sets[ss_id] = structure_set
            return ss_id

        def get_image_data(self, image_id):
            if image_id in self.images:
                return self.images[image_id].data
            return None

    # Create test data
    test_patient = TestPatient("patient_1", "John Doe")
    test_image = TestImage()
    test_image.patient_id = test_patient.id

    # Register mock services
    ServiceRegistry._services = {}  # Reset services
    ServiceRegistry.register_service("PatientDB", TestPatientDB())

    # Add test data to mock DB
    patient_db = ServiceRegistry.get_service("PatientDB")
    patient_db.patients[test_patient.id] = test_patient
    patient_db.images[test_image.id] = test_image

    # Create structure tab
    structure_tab = StructureTab()
    structure_tab.set_patient(test_patient)

    # Set as central widget
    main_window.setCentralWidget(structure_tab)
    main_window.setWindowTitle("QuangTPS - Structure Tab")
    main_window.resize(1200, 800)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    test_structure_tab()
