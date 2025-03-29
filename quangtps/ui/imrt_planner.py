#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMRT Planner module for QuangTPS.

This module provides a user interface for creating and managing IMRT treatment plans,
including beam setup, optimization parameters, and fluence map editing.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox, 
    QTabWidget, QSplitter, QFrame, QMessageBox, QListWidget,
    QListWidgetItem, QCheckBox, QSpinBox, QLineEdit, QDialog,
    QDialogButtonBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon

from quangtps.treatment.techniques.imrt import IMRT, IMRTOptimizationType, IMRTDeliveryType
from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.ui.beam_visualization import BeamVisualization
from quangtps.imaging.structures import Structure
from quangtps.ui.mlc_editor import MLCEditor
from quangtps.treatment.mlc.mlc_model import MLCModel
from quangtps.ui.dialogs.beam_dialog import BeamDialog
from quangtps.common.paths import get_icon_path

logger = logging.getLogger(__name__)


class IMRTPlanner(QWidget):
    """
    Interface for IMRT treatment planning.
    
    This widget provides a user interface for creating and managing
    IMRT treatment plans, including optimization parameters, beam setup,
    and fluence map editing.
    """
    
    plan_created = pyqtSignal(Plan)
    
    def __init__(self, parent=None):
        """
        Initialize the IMRT planning widget.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        self.imrt = None
        self.plan = None
        self.structures = {}
        self.current_beam = None
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout(self)
        
        # Left panel: Plan settings and beam list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Plan group
        plan_group = QGroupBox("IMRT Plan")
        plan_layout = QVBoxLayout(plan_group)
        
        # Form layout for plan info
        plan_form = QFormLayout()
        
        self.plan_name_edit = QLineEdit("IMRT Plan")
        plan_form.addRow("Plan name:", self.plan_name_edit)
        
        self.site_combo = QComboBox()
        self.site_combo.addItems(["Head & Neck", "Prostate", "Lung", "Breast", "Brain", "Other"])
        plan_form.addRow("Anatomical site:", self.site_combo)
        
        self.prescription_spin = QDoubleSpinBox()
        self.prescription_spin.setRange(0, 100)
        self.prescription_spin.setDecimals(1)
        self.prescription_spin.setValue(70.0)
        self.prescription_spin.setSuffix(" Gy")
        plan_form.addRow("Prescription dose:", self.prescription_spin)
        
        self.fractions_spin = QSpinBox()
        self.fractions_spin.setRange(1, 40)
        self.fractions_spin.setValue(35)
        plan_form.addRow("Number of fractions:", self.fractions_spin)
        
        # IMRT specific settings
        self.optimization_type_combo = QComboBox()
        for opt_type in IMRTOptimizationType:
            self.optimization_type_combo.addItem(opt_type.value, opt_type)
        plan_form.addRow("Optimization type:", self.optimization_type_combo)
        
        self.delivery_type_combo = QComboBox()
        for del_type in IMRTDeliveryType:
            self.delivery_type_combo.addItem(del_type.value, del_type)
        plan_form.addRow("Delivery type:", self.delivery_type_combo)
        
        plan_layout.addLayout(plan_form)
        
        # Create plan button
        self.create_plan_button = QPushButton("Create Plan")
        self.create_plan_button.clicked.connect(self._on_create_plan)
        plan_layout.addWidget(self.create_plan_button)
        
        left_layout.addWidget(plan_group)
        
        # Beam group
        beams_group = QGroupBox("Beams")
        beams_layout = QVBoxLayout(beams_group)
        
        # Template selection
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Template:"))
        
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "Select template...",
            "Head & Neck 7-fields",
            "Prostate 5-fields",
            "Lung 5-fields",
            "Breast 4-fields",
            "Brain 7-fields"
        ])
        template_layout.addWidget(self.template_combo)
        
        self.add_template_button = QPushButton("Add")
        self.add_template_button.clicked.connect(self._on_add_template)
        template_layout.addWidget(self.add_template_button)
        
        beams_layout.addLayout(template_layout)
        
        # Beam list
        self.beams_list = QListWidget()
        self.beams_list.currentItemChanged.connect(self._on_beam_selected)
        beams_layout.addWidget(self.beams_list)
        
        # Beam buttons
        beam_buttons_layout = QHBoxLayout()
        
        self.add_beam_button = QPushButton("Add")
        self.add_beam_button.clicked.connect(self._on_add_beam)
        beam_buttons_layout.addWidget(self.add_beam_button)
        
        self.edit_beam_button = QPushButton("Edit")
        self.edit_beam_button.clicked.connect(self._on_edit_beam)
        beam_buttons_layout.addWidget(self.edit_beam_button)
        
        self.remove_beam_button = QPushButton("Remove")
        self.remove_beam_button.clicked.connect(self._on_remove_beam)
        beam_buttons_layout.addWidget(self.remove_beam_button)
        
        beams_layout.addLayout(beam_buttons_layout)
        
        left_layout.addWidget(beams_group)
        
        # Optimization group
        optimization_group = QGroupBox("Optimization")
        optimization_layout = QVBoxLayout(optimization_group)
        
        # Optimization parameters
        optimization_params_form = QFormLayout()
        
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(10, 1000)
        self.iterations_spin.setValue(100)
        optimization_params_form.addRow("Iterations:", self.iterations_spin)
        
        self.convergence_spin = QDoubleSpinBox()
        self.convergence_spin.setRange(0.0001, 0.1)
        self.convergence_spin.setDecimals(4)
        self.convergence_spin.setValue(0.001)
        self.convergence_spin.setSingleStep(0.0001)
        optimization_params_form.addRow("Convergence threshold:", self.convergence_spin)
        
        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setRange(0, 1)
        self.smoothing_spin.setDecimals(2)
        self.smoothing_spin.setValue(0.3)
        self.smoothing_spin.setSingleStep(0.05)
        optimization_params_form.addRow("Smoothing factor:", self.smoothing_spin)
        
        optimization_layout.addLayout(optimization_params_form)
        
        # Objectives and constraints
        objectives_tabs = QTabWidget()
        
        # Objectives tab
        objectives_tab = QWidget()
        objectives_layout = QVBoxLayout(objectives_tab)
        
        self.objectives_table = QTableWidget()
        self.objectives_table.setColumnCount(5)
        self.objectives_table.setHorizontalHeaderLabels(["Structure", "Type", "Dose (Gy)", "Volume (%)", "Weight"])
        self.objectives_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        objectives_layout.addWidget(self.objectives_table)
        
        objectives_buttons = QHBoxLayout()
        
        self.add_objective_button = QPushButton("Add Objective")
        self.add_objective_button.clicked.connect(self._on_add_objective)
        objectives_buttons.addWidget(self.add_objective_button)
        
        self.remove_objective_button = QPushButton("Remove Objective")
        self.remove_objective_button.clicked.connect(self._on_remove_objective)
        objectives_buttons.addWidget(self.remove_objective_button)
        
        objectives_layout.addLayout(objectives_buttons)
        
        objectives_tabs.addTab(objectives_tab, "Objectives")
        
        # Constraints tab
        constraints_tab = QWidget()
        constraints_layout = QVBoxLayout(constraints_tab)
        
        self.constraints_table = QTableWidget()
        self.constraints_table.setColumnCount(4)
        self.constraints_table.setHorizontalHeaderLabels(["Structure", "Type", "Dose (Gy)", "Volume (%)"])
        self.constraints_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        constraints_layout.addWidget(self.constraints_table)
        
        constraints_buttons = QHBoxLayout()
        
        self.add_constraint_button = QPushButton("Add Constraint")
        self.add_constraint_button.clicked.connect(self._on_add_constraint)
        constraints_buttons.addWidget(self.add_constraint_button)
        
        self.remove_constraint_button = QPushButton("Remove Constraint")
        self.remove_constraint_button.clicked.connect(self._on_remove_constraint)
        constraints_buttons.addWidget(self.remove_constraint_button)
        
        constraints_layout.addLayout(constraints_buttons)
        
        objectives_tabs.addTab(constraints_tab, "Constraints")
        
        optimization_layout.addWidget(objectives_tabs)
        
        # Optimization buttons
        self.run_optimization_button = QPushButton("Run Optimization")
        self.run_optimization_button.clicked.connect(self._on_run_optimization)
        optimization_layout.addWidget(self.run_optimization_button)
        
        # Add MCO button
        self.run_mco_button = QPushButton("Multi-Criteria Optimization")
        self.run_mco_button.clicked.connect(self._on_run_mco)
        optimization_layout.addWidget(self.run_mco_button)
        
        left_layout.addWidget(optimization_group)
        
        # Right panel: Beam visualization and fluence editor
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        right_tabs = QTabWidget()
        
        # Beam visualization tab
        self.beam_visualization = BeamVisualization()
        right_tabs.addTab(self.beam_visualization, "Beam Visualization")
        
        # Fluence editor tab
        self.fluence_editor = MLCEditor()
        right_tabs.addTab(self.fluence_editor, "Fluence Map Editor")
        
        right_layout.addWidget(right_tabs)
        
        # Calculate and apply buttons
        buttons_layout = QHBoxLayout()
        
        self.calculate_button = QPushButton("Calculate Dose")
        self.calculate_button.clicked.connect(self._on_calculate_dose)
        buttons_layout.addWidget(self.calculate_button)
        
        self.apply_button = QPushButton("Apply Plan")
        self.apply_button.clicked.connect(self._on_apply_plan)
        buttons_layout.addWidget(self.apply_button)
        
        right_layout.addLayout(buttons_layout)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)
        
        # Update UI state
        self._update_ui_state()
    
    def set_structures(self, structures: Dict[str, Structure]):
        """
        Set the structures for planning.
        
        Parameters
        ----------
        structures : Dict[str, Structure]
            Dictionary of structures with names as keys
        """
        self.structures = structures
        self.beam_visualization.set_structures(structures)
        self._update_structure_combo_boxes()
    
    def _update_structure_combo_boxes(self):
        """Update structure dropdown menus."""
        # This would be called when adding objectives/constraints
        pass
    
    def _update_ui_state(self):
        """Update UI state based on current plan."""
        has_plan = self.plan is not None
        has_beams = has_plan and hasattr(self.plan, 'beams') and len(self.plan.beams) > 0
        has_selected_beam = self.current_beam is not None
        
        # Update button states
        self.edit_beam_button.setEnabled(has_selected_beam)
        self.remove_beam_button.setEnabled(has_selected_beam)
        self.calculate_button.setEnabled(has_beams)
        self.run_optimization_button.setEnabled(has_beams and len(self.structures) > 0)
        self.run_mco_button.setEnabled(has_beams and len(self.structures) > 0)
        self.apply_button.setEnabled(has_beams)
        
        # Update other UI elements based on plan state
    
    def _on_create_plan(self):
        """Handle create plan button click."""
        plan_name = self.plan_name_edit.text()
        
        if not plan_name:
            QMessageBox.warning(self, "Error", "Please enter a plan name")
            return
        
        # Create IMRT plan
        self.imrt = IMRT(
            name=plan_name,
            optimization_type=self.optimization_type_combo.currentData(),
            delivery_type=self.delivery_type_combo.currentData()
        )
        
        # Create base plan object
        self.plan = Plan()
        self.plan.name = plan_name
        self.plan.technique = "IMRT"
        
        # Set optimization parameters
        self.imrt.set_optimization_parameters(
            iterations=self.iterations_spin.value(),
            convergence_threshold=self.convergence_spin.value(),
            smoothing_factor=self.smoothing_spin.value()
        )
        
        # Set prescription
        self.plan.prescription_dose = self.prescription_spin.value()
        self.plan.num_fractions = self.fractions_spin.value()
        
        # Clear beam list
        self.beams_list.clear()
        self.current_beam = None
        
        # Update UI state
        self._update_ui_state()
        
        # Show message
        QMessageBox.information(self, "Information", f"Created IMRT plan: {plan_name}")
    
    def _on_add_template(self):
        """Handle adding template beams."""
        if not self.plan:
            QMessageBox.warning(self, "Warning", "Please create a plan first")
            return
        
        template_name = self.template_combo.currentText()
        if template_name == "Select template...":
            return
        
        # Add template beams based on selection
        try:
            if template_name == "Head & Neck 7-fields":
                self._add_head_neck_template()
            elif template_name == "Prostate 5-fields":
                self._add_prostate_template()
            elif template_name == "Lung 5-fields":
                self._add_lung_template()
            elif template_name == "Breast 4-fields":
                self._add_breast_template()
            elif template_name == "Brain 7-fields":
                self._add_brain_template()
            
            # Update beam list
            self._update_beam_list()
            
            # Update UI state
            self._update_ui_state()
            
            QMessageBox.information(self, "Information", f"Added template: {template_name}")
        except Exception as e:
            logger.error(f"Error adding template: {e}")
            QMessageBox.critical(self, "Error", f"Failed to add template: {str(e)}")
    
    def _add_head_neck_template(self):
        """Add Head & Neck 7-field template."""
        # Example implementation - would create 7 beams for H&N
        angles = [0, 51, 102, 153, 204, 255, 306]
        for i, angle in enumerate(angles):
            beam = Beam()
            beam.name = f"HN-{i+1}"
            beam.gantry_angle = angle
            beam.collimator_angle = 0
            beam.couch_angle = 0
            beam.energy = "6MV"
            beam.field_size = (10, 10)  # Placeholder, would be tailored to target
            
            # Add beam to plan
            self.plan.beams.append(beam)
            if hasattr(self.imrt, 'add_beam'):
                self.imrt.add_beam(beam)
    
    def _add_prostate_template(self):
        """Add Prostate 5-field template."""
        angles = [0, 72, 144, 216, 288]
        for i, angle in enumerate(angles):
            beam = Beam()
            beam.name = f"Prostate-{i+1}"
            beam.gantry_angle = angle
            beam.collimator_angle = 0
            beam.couch_angle = 0
            beam.energy = "6MV"
            beam.field_size = (8, 8)  # Placeholder
            
            # Add beam to plan
            self.plan.beams.append(beam)
            if hasattr(self.imrt, 'add_beam'):
                self.imrt.add_beam(beam)
    
    # Similar methods for other templates would be implemented
    
    def _update_beam_list(self):
        """Update the beam list widget."""
        self.beams_list.clear()
        
        if not self.plan or not hasattr(self.plan, 'beams') or not self.plan.beams:
            return
        
        for i, beam in enumerate(self.plan.beams):
            item = QListWidgetItem(f"{beam.name} - {beam.gantry_angle}°")
            item.setData(Qt.UserRole, beam)
            self.beams_list.addItem(item)
    
    def _on_beam_selected(self, current, previous):
        """Handle beam selection from the list."""
        if current:
            self.current_beam = current.data(Qt.UserRole)
            self.beam_visualization.set_beam(self.current_beam)
            
            # Update fluence editor if available
            if hasattr(self.fluence_editor, 'set_beam'):
                self.fluence_editor.set_beam(self.current_beam)
        else:
            self.current_beam = None
            self.beam_visualization.set_beam(None)
        
        self._update_ui_state()
    
    def _on_add_beam(self):
        """Handle add beam button click."""
        if not self.plan:
            QMessageBox.warning(self, "Warning", "Please create a plan first")
            return
        
        # Open beam dialog
        dialog = BeamDialog(self)
        
        if dialog.exec_() == QDialog.Accepted:
            # Get beam from dialog
            beam_setup = dialog.beam_setup
            
            # Convert to Beam object
            beam = Beam()
            beam.name = beam_setup.name
            beam.gantry_angle = beam_setup.gantry_angle
            beam.collimator_angle = beam_setup.collimator_angle
            beam.couch_angle = beam_setup.couch_angle
            beam.energy = beam_setup.energy
            beam.field_size = (beam_setup.field_width, beam_setup.field_height)
            beam.isocenter = (beam_setup.isocenter_x, beam_setup.isocenter_y, beam_setup.isocenter_z)
            
            # Add beam to plan
            if not hasattr(self.plan, 'beams'):
                self.plan.beams = []
            
            self.plan.beams.append(beam)
            
            # Add to IMRT technique
            if hasattr(self.imrt, 'add_beam'):
                self.imrt.add_beam(beam)
            
            # Update beam list
            self._update_beam_list()
            
            # Update UI state
            self._update_ui_state()
    
    def _on_edit_beam(self):
        """Handle edit beam button click."""
        if not self.current_beam:
            return
        
        # Create beam setup from current beam
        beam_setup = self._create_beam_setup_from_beam(self.current_beam)
        
        # Open beam dialog
        dialog = BeamDialog(self, beam_setup)
        
        if dialog.exec_() == QDialog.Accepted:
            # Update beam from dialog
            updated_setup = dialog.beam_setup
            
            # Update beam properties
            self.current_beam.name = updated_setup.name
            self.current_beam.gantry_angle = updated_setup.gantry_angle
            self.current_beam.collimator_angle = updated_setup.collimator_angle
            self.current_beam.couch_angle = updated_setup.couch_angle
            self.current_beam.energy = updated_setup.energy
            self.current_beam.field_size = (updated_setup.field_width, updated_setup.field_height)
            self.current_beam.isocenter = (updated_setup.isocenter_x, updated_setup.isocenter_y, updated_setup.isocenter_z)
            
            # Update beam visualization
            self.beam_visualization.set_beam(self.current_beam)
            
            # Update beam list
            self._update_beam_list()
    
    def _create_beam_setup_from_beam(self, beam):
        """
        Create a BeamSetup object from a Beam object.
        
        Parameters
        ----------
        beam : Beam
            The beam to convert
            
        Returns
        -------
        BeamSetup
            BeamSetup object with properties from the beam
        """
        from quangtps.planning.beam import BeamSetup
        
        setup = BeamSetup()
        setup.name = beam.name
        setup.gantry_angle = beam.gantry_angle
        setup.collimator_angle = beam.collimator_angle
        setup.couch_angle = beam.couch_angle
        setup.energy = beam.energy
        
        if hasattr(beam, 'field_size') and beam.field_size:
            setup.field_width = beam.field_size[0]
            setup.field_height = beam.field_size[1]
        
        if hasattr(beam, 'isocenter') and beam.isocenter:
            setup.isocenter_x = beam.isocenter[0]
            setup.isocenter_y = beam.isocenter[1]
            setup.isocenter_z = beam.isocenter[2]
        
        return setup
    
    def _on_remove_beam(self):
        """Handle remove beam button click."""
        if not self.current_beam:
            return
        
        # Confirm deletion
        result = QMessageBox.question(
            self, 
            "Confirm Deletion",
            f"Are you sure you want to remove beam '{self.current_beam.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Remove from plan
            if hasattr(self.plan, 'beams'):
                self.plan.beams.remove(self.current_beam)
            
            # Update beam list
            self._update_beam_list()
            
            # Clear current beam
            self.current_beam = None
            self.beam_visualization.set_beam(None)
            
            # Update UI state
            self._update_ui_state()
    
    def _on_add_objective(self):
        """Handle add objective button click."""
        if not self.imrt:
            QMessageBox.warning(self, "Warning", "Please create a plan first")
            return
        
        # Logic to create objective dialog and add to plan
        # For now, just add a sample objective to the table
        row_position = self.objectives_table.rowCount()
        self.objectives_table.insertRow(row_position)
        
        # Add sample data
        self.objectives_table.setItem(row_position, 0, QTableWidgetItem("PTV"))
        self.objectives_table.setItem(row_position, 1, QTableWidgetItem("Min Dose"))
        self.objectives_table.setItem(row_position, 2, QTableWidgetItem("70.0"))
        self.objectives_table.setItem(row_position, 3, QTableWidgetItem("95"))
        self.objectives_table.setItem(row_position, 4, QTableWidgetItem("1.0"))
    
    def _on_remove_objective(self):
        """Handle remove objective button click."""
        selected_rows = self.objectives_table.selectionModel().selectedRows()
        
        if not selected_rows:
            return
        
        # Confirm deletion
        result = QMessageBox.question(
            self, 
            "Confirm Deletion",
            "Are you sure you want to remove the selected objective?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Remove selected row
            for row in sorted(selected_rows, reverse=True):
                self.objectives_table.removeRow(row.row())
    
    def _on_add_constraint(self):
        """Handle add constraint button click."""
        if not self.imrt:
            QMessageBox.warning(self, "Warning", "Please create a plan first")
            return
        
        # Logic to create constraint dialog and add to plan
        # For now, just add a sample constraint to the table
        row_position = self.constraints_table.rowCount()
        self.constraints_table.insertRow(row_position)
        
        # Add sample data
        self.constraints_table.setItem(row_position, 0, QTableWidgetItem("Brainstem"))
        self.constraints_table.setItem(row_position, 1, QTableWidgetItem("Max Dose"))
        self.constraints_table.setItem(row_position, 2, QTableWidgetItem("54.0"))
        self.constraints_table.setItem(row_position, 3, QTableWidgetItem("0"))
    
    def _on_remove_constraint(self):
        """Handle remove constraint button click."""
        selected_rows = self.constraints_table.selectionModel().selectedRows()
        
        if not selected_rows:
            return
        
        # Confirm deletion
        result = QMessageBox.question(
            self, 
            "Confirm Deletion",
            "Are you sure you want to remove the selected constraint?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Remove selected row
            for row in sorted(selected_rows, reverse=True):
                self.constraints_table.removeRow(row.row())
    
    def _on_run_optimization(self):
        """Handle run optimization button click."""
        if not self.imrt or not self.plan or not hasattr(self.plan, 'beams') or not self.plan.beams:
            QMessageBox.warning(self, "Warning", "Please create a complete plan with beams first")
            return
        
        # Set optimization parameters
        self.imrt.set_optimization_parameters(
            iterations=self.iterations_spin.value(),
            convergence_threshold=self.convergence_spin.value(),
            smoothing_factor=self.smoothing_spin.value()
        )
        
        # Add objectives and constraints from tables
        self._add_objectives_to_imrt()
        self._add_constraints_to_imrt()
        
        # Run optimization
        # In a real implementation, this would be a long-running process
        # that should show progress and run in a separate thread
        QMessageBox.information(
            self, 
            "Optimization",
            "Optimization would run here. This is a placeholder."
        )
    
    def _add_objectives_to_imrt(self):
        """Add objectives from table to IMRT plan."""
        # Clear existing objectives
        if hasattr(self.imrt, 'dose_objectives'):
            self.imrt.dose_objectives = []
        
        # Add objectives from table
        for row in range(self.objectives_table.rowCount()):
            structure = self.objectives_table.item(row, 0).text()
            obj_type = self.objectives_table.item(row, 1).text()
            dose = float(self.objectives_table.item(row, 2).text())
            volume = float(self.objectives_table.item(row, 3).text()) if self.objectives_table.item(row, 3).text() else None
            weight = float(self.objectives_table.item(row, 4).text())
            
            # Add to IMRT plan
            if hasattr(self.imrt, 'add_objective'):
                self.imrt.add_objective(structure, obj_type, dose, volume, weight)
    
    def _add_constraints_to_imrt(self):
        """Add constraints from table to IMRT plan."""
        # Clear existing constraints
        if hasattr(self.imrt, 'constraints'):
            self.imrt.constraints = []
        
        # Add constraints from table
        for row in range(self.constraints_table.rowCount()):
            structure = self.constraints_table.item(row, 0).text()
            con_type = self.constraints_table.item(row, 1).text()
            dose = float(self.constraints_table.item(row, 2).text())
            volume = float(self.constraints_table.item(row, 3).text()) if self.constraints_table.item(row, 3).text() else None
            
            # Add to IMRT plan
            if hasattr(self.imrt, 'add_constraint'):
                self.imrt.add_constraint(structure, con_type, dose, volume)
    
    def _on_calculate_dose(self):
        """Handle calculate dose button click."""
        if not self.plan or not hasattr(self.plan, 'beams') or not self.plan.beams:
            QMessageBox.warning(self, "Warning", "Please create a complete plan with beams first")
            return
        
        # Logic to calculate dose
        QMessageBox.information(
            self, 
            "Dose Calculation",
            "Dose calculation would run here. This is a placeholder."
        )
    
    def _on_apply_plan(self):
        """Handle apply plan button click."""
        if not self.plan:
            return
        
        # Apply the IMRT technique to the plan
        self.plan.technique_data = self.imrt.to_dict() if hasattr(self.imrt, 'to_dict') else {}
        
        # Emit signal with the created plan
        self.plan_created.emit(self.plan)
        
        QMessageBox.information(self, "Success", "IMRT plan has been created and applied.")
    
    def _on_run_mco(self):
        """Handle multi-criteria optimization button click."""
        if not self.imrt or not self.plan or not hasattr(self.plan, 'beams') or not self.plan.beams:
            QMessageBox.warning(self, "Warning", "Please create a complete plan with beams first")
            return
        
        # Add objectives and constraints to the IMRT plan
        self._add_objectives_to_imrt()
        self._add_constraints_to_imrt()
        
        try:
            # Import MCO components
            from quangtps.optimization.mco.mco_engine import create_mco_engine
            from quangtps.optimization.objectives import get_objectives_from_plan
            from quangtps.optimization.constraints import get_constraints_from_plan
            
            # Check if we have objectives
            objectives = get_objectives_from_plan(self.plan)
            if not objectives:
                # Ask user if they want to import an MCO template
                reply = QMessageBox.question(
                    self, 
                    "No Objectives Found",
                    "No optimization objectives found. Would you like to import a template?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # Import template
                    from quangtps.ui.mco_template_dialog import select_mco_template
                    template = select_mco_template(self)
                    
                    if template:
                        # Create objectives and constraints from template
                        objectives = template.create_objectives()
                        constraints = template.create_constraints()
                        
                        # Add them to the plan
                        for name, obj in objectives.items():
                            if hasattr(self.imrt, 'add_objective'):
                                self.imrt.add_objective(
                                    obj.structure, 
                                    obj.objective_type.name, 
                                    obj.dose_parameter, 
                                    obj.volume_parameter if hasattr(obj, 'volume_parameter') else None,
                                    obj.weight
                                )
                        
                        for constraint in constraints:
                            if hasattr(self.imrt, 'add_constraint'):
                                self.imrt.add_constraint(
                                    constraint.structure,
                                    constraint.constraint_type.name,
                                    constraint.dose_parameter,
                                    constraint.volume_parameter if hasattr(constraint, 'volume_parameter') else None
                                )
                    else:
                        return
                else:
                    return
            
            # Get constraints
            constraints = get_constraints_from_plan(self.plan)
            
            # Launch the MCO navigator
            from quangtps.ui.mco_navigator_dialog import show_mco_navigator
            updated_plan = show_mco_navigator(self.plan, self)
            
            if updated_plan:
                self.plan = updated_plan
                # Update UI to show optimization completed
                QMessageBox.information(
                    self,
                    "MCO Completed",
                    "Multi-criteria optimization completed successfully.\nThe plan has been updated."
                )
                
                # Emit plan created signal
                self.plan_created.emit(self.plan)
            
        except ImportError:
            QMessageBox.warning(
                self,
                "Feature Not Available",
                "Multi-criteria optimization is not available in this version."
            )
        except Exception as e:
            logger.error(f"Error running MCO: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred during multi-criteria optimization:\n{str(e)}"
            ) 