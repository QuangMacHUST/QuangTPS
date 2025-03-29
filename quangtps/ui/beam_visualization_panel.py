#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beam Visualization Panel

This module provides a panel for visualizing and managing radiotherapy beams
in the QuangTPS treatment planning system. It integrates the 3D visualization
module and provides controls for editing beam parameters.
"""

import logging
import numpy as np
from typing import List, Dict, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QGridLayout, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QSpinBox, QCheckBox, QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.structures.structure_set import StructureSet
from quangtps.dose.dose_grid import DoseGrid
from quangtps.core.services import ServiceManager

# Import the 3D visualization module
try:
    from quangtps.ui.beam_3d_visualization import Beam3DVisualization, HAS_PYVISTA
except ImportError:
    HAS_PYVISTA = False
    logging.warning("3D visualization module not available")

logger = logging.getLogger(__name__)

class BeamVisualizationPanel(QWidget):
    """
    A panel for visualizing and editing radiotherapy beams.
    
    This panel provides an interface for:
    - Visualizing beams in 3D
    - Editing beam parameters
    - Adding, removing, and copying beams
    - Calculating beam dose
    - Displaying beam statistics
    """
    
    # Signals
    beam_added = pyqtSignal(Beam)
    beam_modified = pyqtSignal(Beam)
    beam_removed = pyqtSignal(Beam)
    beam_selected = pyqtSignal(Beam)
    calculate_dose_requested = pyqtSignal(Beam)
    
    def __init__(self, parent=None):
        """Initialize the beam visualization panel."""
        super().__init__(parent)
        
        # Initialize data
        self.patient_image = None
        self.structure_set = None
        self.current_plan = None
        self.dose_grid = None
        self.selected_beam = None
        
        # Set up services
        self.service_manager = ServiceManager()
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Splitter for 3D view and beam controls
        splitter = QSplitter(Qt.Horizontal)
        
        # Tab widget for visualization and beam table
        vis_tabs = QTabWidget()
        
        # 3D Visualization
        if HAS_PYVISTA:
            self.vis_3d = Beam3DVisualization()
            vis_tabs.addTab(self.vis_3d, "3D View")
            
            # Connect signals
            self.vis_3d.beam_selected.connect(self._on_beam_selected_3d)
        else:
            # Fallback if 3D visualization is not available
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout()
            fallback_label = QLabel("3D visualization requires PyVista.\nPlease install with: pip install pyvista pyvistaqt")
            fallback_label.setAlignment(Qt.AlignCenter)
            fallback_layout.addWidget(fallback_label)
            fallback_widget.setLayout(fallback_layout)
            vis_tabs.addTab(fallback_widget, "3D View")
        
        # Beam table view
        self.beam_table = self._create_beam_table()
        vis_tabs.addTab(self.beam_table, "Beam List")
        
        # Add visualization tabs to splitter
        splitter.addWidget(vis_tabs)
        
        # Beam editing panel
        beam_edit_panel = self._create_beam_edit_panel()
        splitter.addWidget(beam_edit_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Add buttons for common actions
        button_layout = QHBoxLayout()
        
        self.btn_add_beam = QPushButton("Add Beam")
        self.btn_add_beam.clicked.connect(self._add_new_beam)
        
        self.btn_remove_beam = QPushButton("Remove Beam")
        self.btn_remove_beam.clicked.connect(self._remove_selected_beam)
        self.btn_remove_beam.setEnabled(False)
        
        self.btn_copy_beam = QPushButton("Copy Beam")
        self.btn_copy_beam.clicked.connect(self._copy_selected_beam)
        self.btn_copy_beam.setEnabled(False)
        
        self.btn_calculate_dose = QPushButton("Calculate Dose")
        self.btn_calculate_dose.clicked.connect(self._calculate_beam_dose)
        self.btn_calculate_dose.setEnabled(False)
        
        button_layout.addWidget(self.btn_add_beam)
        button_layout.addWidget(self.btn_copy_beam)
        button_layout.addWidget(self.btn_remove_beam)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_calculate_dose)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def _create_beam_table(self):
        """Create a table widget for displaying beam information."""
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "Name", "Gantry", "Collimator", "Couch", "Field Size", "Weight"
        ])
        
        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Name column stretches
        for i in range(1, 6):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        # Connect signals
        table.itemSelectionChanged.connect(self._on_beam_selected_table)
        
        return table
    
    def _create_beam_edit_panel(self):
        """Create a panel for editing beam parameters."""
        group_box = QGroupBox("Beam Parameters")
        layout = QGridLayout()
        
        # Beam name
        layout.addWidget(QLabel("Name:"), 0, 0)
        self.beam_name_edit = QComboBox()
        self.beam_name_edit.setEditable(True)
        self.beam_name_edit.addItems(["AP", "PA", "LAO", "RAO", "LPO", "RPO", "RLAT", "LLAT"])
        self.beam_name_edit.currentTextChanged.connect(self._on_beam_param_changed)
        layout.addWidget(self.beam_name_edit, 0, 1, 1, 2)
        
        # Beam angles
        layout.addWidget(QLabel("Gantry Angle:"), 1, 0)
        self.gantry_angle_edit = QDoubleSpinBox()
        self.gantry_angle_edit.setRange(0, 360)
        self.gantry_angle_edit.setValue(0)
        self.gantry_angle_edit.valueChanged.connect(self._on_beam_param_changed)
        layout.addWidget(self.gantry_angle_edit, 1, 1)
        layout.addWidget(QLabel("°"), 1, 2)
        
        layout.addWidget(QLabel("Collimator Angle:"), 2, 0)
        self.collimator_angle_edit = QDoubleSpinBox()
        self.collimator_angle_edit.setRange(0, 360)
        self.collimator_angle_edit.setValue(0)
        self.collimator_angle_edit.valueChanged.connect(self._on_beam_param_changed)
        layout.addWidget(self.collimator_angle_edit, 2, 1)
        layout.addWidget(QLabel("°"), 2, 2)
        
        layout.addWidget(QLabel("Couch Angle:"), 3, 0)
        self.couch_angle_edit = QDoubleSpinBox()
        self.couch_angle_edit.setRange(0, 360)
        self.couch_angle_edit.setValue(0)
        self.couch_angle_edit.valueChanged.connect(self._on_beam_param_changed)
        layout.addWidget(self.couch_angle_edit, 3, 1)
        layout.addWidget(QLabel("°"), 3, 2)
        
        # Field size
        layout.addWidget(QLabel("Field Size X:"), 4, 0)
        self.field_size_x_edit = QDoubleSpinBox()
        self.field_size_x_edit.setRange(1, 40)
        self.field_size_x_edit.setValue(10)
        self.field_size_x_edit.setSingleStep(0.5)
        self.field_size_x_edit.valueChanged.connect(self._on_beam_param_changed)
        layout.addWidget(self.field_size_x_edit, 4, 1)
        layout.addWidget(QLabel("cm"), 4, 2)
        
        layout.addWidget(QLabel("Field Size Y:"), 5, 0)
        self.field_size_y_edit = QDoubleSpinBox()
        self.field_size_y_edit.setRange(1, 40)
        self.field_size_y_edit.setValue(10)
        self.field_size_y_edit.setSingleStep(0.5)
        self.field_size_y_edit.valueChanged.connect(self._on_beam_param_changed)
        layout.addWidget(self.field_size_y_edit, 5, 1)
        layout.addWidget(QLabel("cm"), 5, 2)
        
        # Energy
        layout.addWidget(QLabel("Energy:"), 6, 0)
        self.energy_edit = QComboBox()
        self.energy_edit.addItems(["6 MV", "10 MV", "15 MV", "6 FFF", "10 FFF"])
        self.energy_edit.currentTextChanged.connect(self._on_beam_param_changed)
        layout.addWidget(self.energy_edit, 6, 1, 1, 2)
        
        # Weight
        layout.addWidget(QLabel("Weight:"), 7, 0)
        self.weight_edit = QDoubleSpinBox()
        self.weight_edit.setRange(0, 100)
        self.weight_edit.setValue(1)
        self.weight_edit.setSingleStep(0.1)
        self.weight_edit.valueChanged.connect(self._on_beam_param_changed)
        layout.addWidget(self.weight_edit, 7, 1)
        layout.addWidget(QLabel("%"), 7, 2)
        
        # Isocenter setting
        layout.addWidget(QLabel("Isocenter:"), 8, 0)
        isocenter_layout = QHBoxLayout()
        
        self.iso_x_edit = QDoubleSpinBox()
        self.iso_x_edit.setRange(-500, 500)
        self.iso_x_edit.setSingleStep(1)
        self.iso_x_edit.valueChanged.connect(self._on_beam_param_changed)
        isocenter_layout.addWidget(self.iso_x_edit)
        
        self.iso_y_edit = QDoubleSpinBox()
        self.iso_y_edit.setRange(-500, 500)
        self.iso_y_edit.setSingleStep(1)
        self.iso_y_edit.valueChanged.connect(self._on_beam_param_changed)
        isocenter_layout.addWidget(self.iso_y_edit)
        
        self.iso_z_edit = QDoubleSpinBox()
        self.iso_z_edit.setRange(-500, 500)
        self.iso_z_edit.setSingleStep(1)
        self.iso_z_edit.valueChanged.connect(self._on_beam_param_changed)
        isocenter_layout.addWidget(self.iso_z_edit)
        
        layout.addLayout(isocenter_layout, 8, 1, 1, 2)
        
        # MLC use checkbox
        layout.addWidget(QLabel("Use MLC:"), 9, 0)
        self.use_mlc_checkbox = QCheckBox()
        self.use_mlc_checkbox.toggled.connect(self._on_mlc_toggled)
        layout.addWidget(self.use_mlc_checkbox, 9, 1)
        
        # MLC Editor button (enabled only when Use MLC is checked)
        self.edit_mlc_button = QPushButton("Edit MLC")
        self.edit_mlc_button.clicked.connect(self._edit_mlc)
        self.edit_mlc_button.setEnabled(False)
        layout.addWidget(self.edit_mlc_button, 9, 2)
        
        # Dose calculation algorithm
        layout.addWidget(QLabel("Dose Algorithm:"), 10, 0)
        self.dose_algo_combo = QComboBox()
        self.dose_algo_combo.addItems([
            "Pencil Beam",
            "Collapsed Cone",
            "Monte Carlo",
            "Convolution",
            "AAA"
        ])
        layout.addWidget(self.dose_algo_combo, 10, 1, 1, 2)
        
        # Apply button
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self._apply_beam_changes)
        self.apply_button.setEnabled(False)
        layout.addWidget(self.apply_button, 11, 0, 1, 3)
        
        # Disable all controls initially
        self._set_controls_enabled(False)
        
        group_box.setLayout(layout)
        return group_box
    
    def _set_controls_enabled(self, enabled):
        """Enable or disable beam editing controls."""
        self.beam_name_edit.setEnabled(enabled)
        self.gantry_angle_edit.setEnabled(enabled)
        self.collimator_angle_edit.setEnabled(enabled)
        self.couch_angle_edit.setEnabled(enabled)
        self.field_size_x_edit.setEnabled(enabled)
        self.field_size_y_edit.setEnabled(enabled)
        self.energy_edit.setEnabled(enabled)
        self.weight_edit.setEnabled(enabled)
        self.iso_x_edit.setEnabled(enabled)
        self.iso_y_edit.setEnabled(enabled)
        self.iso_z_edit.setEnabled(enabled)
        self.use_mlc_checkbox.setEnabled(enabled)
        self.edit_mlc_button.setEnabled(enabled and self.use_mlc_checkbox.isChecked())
        self.apply_button.setEnabled(enabled)
    
    def _update_beam_controls(self):
        """Update beam control values based on selected beam."""
        if not self.selected_beam:
            self._set_controls_enabled(False)
            return
            
        # Block signals during update
        self._block_beam_signals(True)
        
        # Update values
        self.beam_name_edit.setCurrentText(self.selected_beam.name)
        self.gantry_angle_edit.setValue(self.selected_beam.gantry_angle)
        self.collimator_angle_edit.setValue(self.selected_beam.collimator_angle)
        self.couch_angle_edit.setValue(self.selected_beam.couch_angle)
        
        if hasattr(self.selected_beam, 'field_size') and self.selected_beam.field_size:
            self.field_size_x_edit.setValue(self.selected_beam.field_size[0])
            self.field_size_y_edit.setValue(self.selected_beam.field_size[1])
        
        if hasattr(self.selected_beam, 'energy') and self.selected_beam.energy:
            self.energy_edit.setCurrentText(self.selected_beam.energy)
        
        if hasattr(self.selected_beam, 'weight') and self.selected_beam.weight is not None:
            self.weight_edit.setValue(self.selected_beam.weight)
        
        if hasattr(self.selected_beam, 'isocenter') and self.selected_beam.isocenter:
            self.iso_x_edit.setValue(self.selected_beam.isocenter[0])
            self.iso_y_edit.setValue(self.selected_beam.isocenter[1])
            self.iso_z_edit.setValue(self.selected_beam.isocenter[2])
        
        has_mlc = hasattr(self.selected_beam, 'mlc') and self.selected_beam.mlc is not None
        self.use_mlc_checkbox.setChecked(has_mlc)
        self.edit_mlc_button.setEnabled(has_mlc)
        
        # Enable controls
        self._set_controls_enabled(True)
        
        # Unblock signals
        self._block_beam_signals(False)
    
    def _block_beam_signals(self, block):
        """Block or unblock signals from beam parameter controls."""
        self.beam_name_edit.blockSignals(block)
        self.gantry_angle_edit.blockSignals(block)
        self.collimator_angle_edit.blockSignals(block)
        self.couch_angle_edit.blockSignals(block)
        self.field_size_x_edit.blockSignals(block)
        self.field_size_y_edit.blockSignals(block)
        self.energy_edit.blockSignals(block)
        self.weight_edit.blockSignals(block)
        self.iso_x_edit.blockSignals(block)
        self.iso_y_edit.blockSignals(block)
        self.iso_z_edit.blockSignals(block)
        self.use_mlc_checkbox.blockSignals(block)
    
    def set_patient_data(self, image: Image, structures: StructureSet = None):
        """
        Set the patient data for visualization.
        
        Parameters
        ----------
        image : Image
            The patient CT or MRI image
        structures : StructureSet, optional
            The patient structure set
        """
        self.patient_image = image
        self.structure_set = structures
        
        # Update 3D visualization if available
        if HAS_PYVISTA and hasattr(self, 'vis_3d'):
            self.vis_3d.set_patient_data(image)
            if structures:
                self.vis_3d.set_structure_set(structures)
    
    def set_plan(self, plan: Plan):
        """
        Set the treatment plan for visualization and editing.
        
        Parameters
        ----------
        plan : Plan
            The treatment plan containing beams
        """
        self.current_plan = plan
        
        # Update 3D visualization if available
        if HAS_PYVISTA and hasattr(self, 'vis_3d'):
            self.vis_3d.set_plan(plan)
        
        # Update beam table
        self._update_beam_table()
        
        # Clear selected beam
        self.selected_beam = None
        self._update_beam_controls()
        
        # Update button states
        self.btn_add_beam.setEnabled(plan is not None)
        self.btn_remove_beam.setEnabled(False)
        self.btn_copy_beam.setEnabled(False)
        self.btn_calculate_dose.setEnabled(False)
    
    def set_dose_grid(self, dose_grid: DoseGrid):
        """
        Set the dose grid for visualization.
        
        Parameters
        ----------
        dose_grid : DoseGrid
            The dose grid containing dose distribution
        """
        self.dose_grid = dose_grid
        
        # Update 3D visualization if available
        if HAS_PYVISTA and hasattr(self, 'vis_3d'):
            self.vis_3d.set_dose_grid(dose_grid)
    
    def _update_beam_table(self):
        """Update the beam table with current plan information."""
        table = self.beam_table
        table.setRowCount(0)
        
        if not self.current_plan or not self.current_plan.beams:
            return
            
        # Add each beam to the table
        for i, beam in enumerate(self.current_plan.beams):
            table.insertRow(i)
            
            # Name
            table.setItem(i, 0, QTableWidgetItem(beam.name))
            
            # Angles
            table.setItem(i, 1, QTableWidgetItem(f"{beam.gantry_angle:.1f}°"))
            table.setItem(i, 2, QTableWidgetItem(f"{beam.collimator_angle:.1f}°"))
            table.setItem(i, 3, QTableWidgetItem(f"{beam.couch_angle:.1f}°"))
            
            # Field size
            if hasattr(beam, 'field_size') and beam.field_size:
                field_size_text = f"{beam.field_size[0]}×{beam.field_size[1]} cm"
            else:
                field_size_text = "MLC"
            table.setItem(i, 4, QTableWidgetItem(field_size_text))
            
            # Weight
            if hasattr(beam, 'weight') and beam.weight is not None:
                weight_text = f"{beam.weight:.1f}%"
            else:
                weight_text = "1.0%"
            table.setItem(i, 5, QTableWidgetItem(weight_text))
    
    def _on_beam_selected_table(self):
        """Handle beam selection from the table."""
        selected_items = self.beam_table.selectedItems()
        if not selected_items:
            return
            
        row = selected_items[0].row()
        if self.current_plan and row < len(self.current_plan.beams):
            self.selected_beam = self.current_plan.beams[row]
            self._update_beam_controls()
            
            # Update 3D visualization
            if HAS_PYVISTA and hasattr(self, 'vis_3d'):
                self.vis_3d.selected_beam = self.selected_beam
                self.vis_3d._update_visualization()
            
            # Update button states
            self.btn_remove_beam.setEnabled(True)
            self.btn_copy_beam.setEnabled(True)
            self.btn_calculate_dose.setEnabled(True)
            
            # Emit signal
            self.beam_selected.emit(self.selected_beam)
    
    def _on_beam_selected_3d(self, beam):
        """Handle beam selection from 3D view."""
        if beam in self.current_plan.beams:
            index = self.current_plan.beams.index(beam)
            self.beam_table.selectRow(index)
    
    def _on_beam_param_changed(self):
        """Handle changes to beam parameters in the UI."""
        if not self.selected_beam:
            return
            
        # Enable apply button to allow saving changes
        self.apply_button.setEnabled(True)
    
    def _on_mlc_toggled(self, use_mlc):
        """Handle toggling of MLC checkbox."""
        self.edit_mlc_button.setEnabled(use_mlc)
        self.apply_button.setEnabled(True)
    
    def _apply_beam_changes(self):
        """Apply changes to the selected beam."""
        if not self.selected_beam:
            return
            
        # Update beam with values from UI
        self.selected_beam.name = self.beam_name_edit.currentText()
        self.selected_beam.gantry_angle = self.gantry_angle_edit.value()
        self.selected_beam.collimator_angle = self.collimator_angle_edit.value()
        self.selected_beam.couch_angle = self.couch_angle_edit.value()
        
        # Field size
        self.selected_beam.field_size = (
            self.field_size_x_edit.value(), 
            self.field_size_y_edit.value()
        )
        
        # Energy
        self.selected_beam.energy = self.energy_edit.currentText()
        
        # Weight
        self.selected_beam.weight = self.weight_edit.value()
        
        # Isocenter
        self.selected_beam.isocenter = [
            self.iso_x_edit.value(),
            self.iso_y_edit.value(),
            self.iso_z_edit.value()
        ]
        
        # MLC
        if self.use_mlc_checkbox.isChecked():
            # Create MLC if it doesn't exist
            if not hasattr(self.selected_beam, 'mlc') or self.selected_beam.mlc is None:
                # Create a default MLC with basic configuration
                mlc_service = self.service_manager.get_service('MLCService')
                if mlc_service:
                    self.selected_beam.mlc = mlc_service.create_default_mlc()
        else:
            # Remove MLC if it exists
            if hasattr(self.selected_beam, 'mlc'):
                self.selected_beam.mlc = None
        
        # Update beam table
        self._update_beam_table()
        
        # Update 3D visualization
        if HAS_PYVISTA and hasattr(self, 'vis_3d'):
            self.vis_3d._update_visualization()
        
        # Emit signal that beam was modified
        self.beam_modified.emit(self.selected_beam)
        
        # Disable apply button since changes are now saved
        self.apply_button.setEnabled(False)
        
        logger.info(f"Applied changes to beam: {self.selected_beam.name}")
    
    def _add_new_beam(self):
        """Add a new beam to the current plan."""
        if not self.current_plan:
            return
            
        # Create a default beam
        beam_service = self.service_manager.get_service('BeamService')
        if not beam_service:
            logger.error("Beam service not available")
            return
            
        new_beam = beam_service.create_beam()
        
        # Set default parameters
        new_beam.name = f"Beam {len(self.current_plan.beams) + 1}"
        new_beam.gantry_angle = 0
        new_beam.collimator_angle = 0
        new_beam.couch_angle = 0
        new_beam.field_size = (10, 10)
        new_beam.energy = "6 MV"
        new_beam.weight = 1.0
        
        # Use plan isocenter if available
        if hasattr(self.current_plan, 'isocenter') and self.current_plan.isocenter:
            new_beam.isocenter = self.current_plan.isocenter
        elif self.patient_image is not None:
            # Use center of image as isocenter
            center = [
                self.patient_image.origin[0] + self.patient_image.spacing[0] * self.patient_image.data.shape[2] / 2,
                self.patient_image.origin[1] + self.patient_image.spacing[1] * self.patient_image.data.shape[1] / 2,
                self.patient_image.origin[2] + self.patient_image.spacing[2] * self.patient_image.data.shape[0] / 2
            ]
            new_beam.isocenter = center
        else:
            new_beam.isocenter = [0, 0, 0]
        
        # Add beam to plan
        self.current_plan.add_beam(new_beam)
        
        # Update the beam table
        self._update_beam_table()
        
        # Select the new beam
        self.beam_table.selectRow(len(self.current_plan.beams) - 1)
        
        # Update 3D visualization
        if HAS_PYVISTA and hasattr(self, 'vis_3d'):
            self.vis_3d.set_plan(self.current_plan)
        
        # Emit signal
        self.beam_added.emit(new_beam)
        
        logger.info(f"Added new beam: {new_beam.name}")
    
    def _remove_selected_beam(self):
        """Remove the selected beam from the current plan."""
        if not self.selected_beam or not self.current_plan:
            return
            
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Remove Beam",
            f"Are you sure you want to remove beam '{self.selected_beam.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm != QMessageBox.Yes:
            return
            
        # Remove the beam
        beam_to_remove = self.selected_beam
        self.current_plan.remove_beam(beam_to_remove)
        
        # Update the beam table
        self._update_beam_table()
        
        # Clear selection
        self.selected_beam = None
        self._update_beam_controls()
        
        # Update button states
        self.btn_remove_beam.setEnabled(False)
        self.btn_copy_beam.setEnabled(False)
        self.btn_calculate_dose.setEnabled(False)
        
        # Update 3D visualization
        if HAS_PYVISTA and hasattr(self, 'vis_3d'):
            self.vis_3d.set_plan(self.current_plan)
            self.vis_3d.selected_beam = None
        
        # Emit signal
        self.beam_removed.emit(beam_to_remove)
        
        logger.info(f"Removed beam: {beam_to_remove.name}")
    
    def _copy_selected_beam(self):
        """Create a copy of the selected beam."""
        if not self.selected_beam or not self.current_plan:
            return
            
        # Create a deep copy of the beam
        beam_service = self.service_manager.get_service('BeamService')
        if not beam_service:
            logger.error("Beam service not available")
            return
            
        new_beam = beam_service.copy_beam(self.selected_beam)
        
        # Modify name to indicate it's a copy
        new_beam.name = f"{self.selected_beam.name} (Copy)"
        
        # Optional: Adjust angle slightly to make it distinguishable
        new_beam.gantry_angle = (self.selected_beam.gantry_angle + 10) % 360
        
        # Add beam to plan
        self.current_plan.add_beam(new_beam)
        
        # Update the beam table
        self._update_beam_table()
        
        # Select the new beam
        self.beam_table.selectRow(len(self.current_plan.beams) - 1)
        
        # Update 3D visualization
        if HAS_PYVISTA and hasattr(self, 'vis_3d'):
            self.vis_3d.set_plan(self.current_plan)
        
        # Emit signal
        self.beam_added.emit(new_beam)
        
        logger.info(f"Copied beam: {self.selected_beam.name} to {new_beam.name}")
    
    def _calculate_beam_dose(self):
        """Calculate dose for the selected beam."""
        if not self.selected_beam:
            return
            
        # Check that we have patient data
        if not self.patient_image:
            QMessageBox.warning(
                self,
                "Cannot Calculate Dose",
                "Patient image data is required for dose calculation."
            )
            return
            
        # Get selected algorithm
        algorithm_name = self.dose_algo_combo.currentText()
        
        # Prepare progress dialog
        # This would typically be implemented with a proper progress dialog,
        # but for simplicity we're just logging the message
        logger.info(f"Calculating dose for beam {self.selected_beam.name} using {algorithm_name}...")
        
        # Emit signal to request dose calculation
        self.calculate_dose_requested.emit(self.selected_beam)
    
    def _edit_mlc(self):
        """Open the MLC editor for the selected beam."""
        if not self.selected_beam or not hasattr(self.selected_beam, 'mlc') or self.selected_beam.mlc is None:
            return
            
        # This would open an MLC editor dialog
        # For now, we just log that it was requested
        logger.info(f"Opening MLC editor for beam {self.selected_beam.name}")
        
        # Get the MLC editor service
        mlc_editor_service = self.service_manager.get_service('MLCEditorService')
        if mlc_editor_service:
            mlc_editor_service.open_editor(self.selected_beam.mlc, self)
        else:
            logger.error("MLC editor service not available")

    def _update_3d_view(self):
        """Update the 3D visualization."""
        # This method creates a simplified 3D visualization directly in this widget
        # For advanced 3D visualization, use beam_3d_visualization.py instead
        if not hasattr(self, '3d_view') or self._3d_view is None:
            return
            
        # Clear the existing 3D view
        self._3d_view.clear()
        
        # Exit if no patient image or no beams
        if not self.patient_image or not self.beams:
            return
            
        # Draw patient outline
        if self.patient_image:
            self._draw_patient_outline()

        # Draw all beams
        for i, beam in enumerate(self.beams):
            # Get beam parameters
            gantry_angle = beam.gantry_angle if hasattr(beam, 'gantry_angle') else 0
            couch_angle = beam.couch_angle if hasattr(beam, 'couch_angle') else 0
            field_size = beam.field_size if hasattr(beam, 'field_size') else (100, 100)
            isocenter = beam.isocenter if hasattr(beam, 'isocenter') else (0, 0, 0)
            
            # Draw the linac for this beam
            self._draw_linac(gantry_angle, couch_angle, isocenter)
            
            # Draw the beam
            self._draw_beam_3d(gantry_angle, couch_angle, field_size, isocenter, i)
            
        # Update the 3D view
        self._3d_view.update()
        
    def _draw_patient_outline(self):
        """Draw the patient outline in the 3D view."""
        if not hasattr(self, '_3d_view') or self._3d_view is None or not self.patient_image:
            return
            
        try:
            # Use a simple cube to represent the patient for now
            # In a real implementation, you would extract the actual patient surface
            # from the CT data or structure set
            
            # Get image dimensions
            dims = self.patient_image.data.shape
            
            # Create a box representing the image volume
            box = np.array([
                [0, 0, 0],
                [dims[0], 0, 0],
                [dims[0], dims[1], 0],
                [0, dims[1], 0],
                [0, 0, dims[2]],
                [dims[0], 0, dims[2]],
                [dims[0], dims[1], dims[2]],
                [0, dims[1], dims[2]],
            ])
            
            # Draw the box as a wireframe
            for i in range(4):
                self._3d_view.add_line(box[i], box[(i+1)%4], color='gray')
                self._3d_view.add_line(box[i+4], box[(i+1)%4+4], color='gray')
                self._3d_view.add_line(box[i], box[i+4], color='gray')
            
        except Exception as e:
            logger.error(f"Error drawing patient outline: {str(e)}")
            
    def _draw_linac(self, gantry_angle, couch_angle, isocenter):
        """
        Draw a simplified linac representation.
        
        Parameters
        ----------
        gantry_angle : float
            Gantry angle in degrees
        couch_angle : float
            Couch angle in degrees
        isocenter : tuple
            Isocenter coordinates (x, y, z)
        """
        if not hasattr(self, '_3d_view') or self._3d_view is None:
            return
            
        try:
            # Constants
            sad = 1000.0  # Source-to-axis distance (mm)
            
            # Convert angles to radians
            gantry_rad = np.radians(gantry_angle)
            couch_rad = np.radians(couch_angle)
            
            # Calculate source position
            source_pos = np.array([
                isocenter[0] + sad * np.sin(gantry_rad) * np.cos(couch_rad),
                isocenter[1] + sad * np.sin(couch_rad),
                isocenter[2] - sad * np.cos(gantry_rad) * np.cos(couch_rad)
            ])
            
            # Draw line from source to isocenter
            self._3d_view.add_line(source_pos, isocenter, color='yellow')
            
            # Draw a point at the source
            self._3d_view.add_point(source_pos, color='red', size=5)
            
            # Draw a point at isocenter
            self._3d_view.add_point(isocenter, color='green', size=5)
            
        except Exception as e:
            logger.error(f"Error drawing linac: {str(e)}")
            
    def _draw_beam_3d(self, gantry_angle, couch_angle, field_size, isocenter, beam_index):
        """
        Draw a simplified 3D representation of a beam.
        
        Parameters
        ----------
        gantry_angle : float
            Gantry angle in degrees
        couch_angle : float
            Couch angle in degrees
        field_size : tuple
            Field size (width, height) in mm
        isocenter : tuple
            Isocenter coordinates (x, y, z)
        beam_index : int
            Index of the beam (for color selection)
        """
        if not hasattr(self, '_3d_view') or self._3d_view is None:
            return
            
        try:
            # Constants
            sad = 1000.0  # Source-to-axis distance (mm)
            ext_distance = 500.0  # Distance to extend the beam beyond isocenter
            
            # Define beam colors based on index
            colors = [
                'red', 'green', 'blue', 'yellow', 'magenta', 'cyan',
                'orange', 'purple', 'white', 'pink'
            ]
            color = colors[beam_index % len(colors)]
            
            # Convert angles to radians
            gantry_rad = np.radians(gantry_angle)
            couch_rad = np.radians(couch_angle)
            
            # Calculate source position
            source_pos = np.array([
                isocenter[0] + sad * np.sin(gantry_rad) * np.cos(couch_rad),
                isocenter[1] + sad * np.sin(couch_rad),
                isocenter[2] - sad * np.cos(gantry_rad) * np.cos(couch_rad)
            ])
            
            # Calculate beam direction vector
            beam_dir = np.array(isocenter) - source_pos
            beam_dir = beam_dir / np.linalg.norm(beam_dir)
            
            # Calculate the extension point beyond isocenter
            ext_point = np.array(isocenter) + ext_distance * beam_dir
            
            # Calculate field corners at isocenter
            # We need to find perpendicular vectors to the beam direction
            width, height = field_size
            
            # Find a perpendicular vector (cross product with up vector)
            # If beam direction is too close to up vector, use a different vector
            up_vector = np.array([0, 1, 0])
            if abs(np.dot(beam_dir, up_vector)) > 0.95:
                up_vector = np.array([1, 0, 0])
                
            # Get perpendicular vectors
            perp1 = np.cross(beam_dir, up_vector)
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(beam_dir, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # Scale perpendicular vectors by field size
            perp1 = perp1 * (width / 2)
            perp2 = perp2 * (height / 2)
            
            # Calculate field corners at isocenter
            corners_iso = [
                np.array(isocenter) + perp1 + perp2,
                np.array(isocenter) - perp1 + perp2,
                np.array(isocenter) - perp1 - perp2,
                np.array(isocenter) + perp1 - perp2
            ]
            
            # Calculate field corners at extension point
            ext_factor = ext_distance / sad  # Scale factor for field size at extension
            corners_ext = [
                ext_point + perp1 * (1 + ext_factor) + perp2 * (1 + ext_factor),
                ext_point - perp1 * (1 + ext_factor) + perp2 * (1 + ext_factor),
                ext_point - perp1 * (1 + ext_factor) - perp2 * (1 + ext_factor),
                ext_point + perp1 * (1 + ext_factor) - perp2 * (1 + ext_factor)
            ]
            
            # Draw the field at isocenter
            for i in range(4):
                self._3d_view.add_line(corners_iso[i], corners_iso[(i+1)%4], color=color)
            
            # Draw the field at extension
            for i in range(4):
                self._3d_view.add_line(corners_ext[i], corners_ext[(i+1)%4], color=color)
            
            # Connect the two fields
            for i in range(4):
                self._3d_view.add_line(corners_iso[i], corners_ext[i], color=color)
            
            # Draw lines from source to field corners
            for corner in corners_iso:
                self._3d_view.add_line(source_pos, corner, color=color)
            
        except Exception as e:
            logger.error(f"Error drawing beam: {str(e)}") 