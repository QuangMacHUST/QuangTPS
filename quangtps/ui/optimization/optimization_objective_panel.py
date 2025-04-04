#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimization Objective Panel Module
================================

This module provides an Eclipse-like interface for setting up and managing
optimization objectives for IMRT and VMAT treatment planning.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, 
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QGroupBox, QFrame, QSplitter, QAbstractItemView,
    QMenu, QAction, QToolButton, QSizePolicy
)
from PyQt5.QtGui import QColor, QBrush, QIcon, QPixmap, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint

# Try to import QuangTPS modules
try:
    from quangtps.planning.prescription import Prescription, DoseLevel, PrescriptionTarget
    from quangtps.structures.structure import Structure
    from quangtps.structures.structure_set import StructureSet
    from quangtps.optimization.objectives import (
        ObjectiveFunction, DoseObjective, DVHObjective, 
        LowerDoseObjective, UpperDoseObjective,
        MeanDoseObjective, MaxDoseObjective, MinDoseObjective,
        LowerDVHObjective, UpperDVHObjective, 
        ConformityObjective, HomogeneityObjective
    )
except ImportError:
    logging.warning("Failed to import QuangTPS optimization modules")
    # Create placeholder enums for types if real ones aren't available
    class ObjectiveType(Enum):
        LOWER_DOSE = 1
        UPPER_DOSE = 2
        MEAN_DOSE = 3
        MAX_DOSE = 4
        MIN_DOSE = 5
        LOWER_DVH = 6
        UPPER_DVH = 7
        CONFORMITY = 8
        HOMOGENEITY = 9

logger = logging.getLogger(__name__)

class OptimizationRow:
    """
    Helper class to store row data for optimization objectives table.
    """
    
    def __init__(self, 
                 structure=None, 
                 objective_type=None,
                 dose=0.0, 
                 volume=0.0, 
                 weight=1.0,
                 priority=100,
                 enabled=True):
        """Initialize the optimization row with provided data."""
        self.structure = structure
        self.structure_id = structure.id if hasattr(structure, 'id') else None
        self.structure_name = structure.name if hasattr(structure, 'name') else None
        self.objective_type = objective_type
        self.dose = dose
        self.volume = volume
        self.weight = weight
        self.priority = priority
        self.enabled = enabled

class OptimizationObjectivePanel(QWidget):
    """
    Panel for setting up and managing optimization objectives.
    
    This widget provides an Eclipse-like interface for configuring objectives
    for IMRT and VMAT optimization, with a table of objectives and controls
    for adding, editing, and removing objectives.
    """
    
    # Signals
    objectivesChanged = pyqtSignal()
    startOptimizationRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the optimization objective panel."""
        super().__init__(parent)
        
        # Initialize variables
        self.structure_set = None
        self.prescription = None
        self.objectives = []
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create objectives table
        self.objectives_table = QTableWidget()
        self.objectives_table.setColumnCount(7)
        self.objectives_table.setHorizontalHeaderLabels([
            "Structure", "Type", "Dose (Gy)", "Volume (%)", "Weight", "Priority", "Enabled"
        ])
        self.objectives_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.objectives_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.objectives_table.setAlternatingRowColors(True)
        self.objectives_table.verticalHeader().setVisible(False)
        self.objectives_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.objectives_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.objectives_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.objectives_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        
        self.objectives_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.objectives_table.customContextMenuRequested.connect(self._show_context_menu)
        
        # Create controls layout
        controls_layout = QHBoxLayout()
        
        # Create add objective controls
        add_group = QGroupBox("Add Objective")
        add_layout = QFormLayout(add_group)
        
        self.structure_combo = QComboBox()
        self.structure_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        add_layout.addRow("Structure:", self.structure_combo)
        
        self.objective_type_combo = QComboBox()
        self.objective_type_combo.addItems([
            "Upper Dose", "Lower Dose", "Mean Dose", "Max Dose", "Min Dose",
            "Upper DVH", "Lower DVH", "Conformity", "Homogeneity"
        ])
        add_layout.addRow("Type:", self.objective_type_combo)
        
        self.dose_spin = QDoubleSpinBox()
        self.dose_spin.setRange(0, 100)
        self.dose_spin.setDecimals(1)
        self.dose_spin.setSuffix(" Gy")
        add_layout.addRow("Dose:", self.dose_spin)
        
        self.volume_spin = QDoubleSpinBox()
        self.volume_spin.setRange(0, 100)
        self.volume_spin.setDecimals(1)
        self.volume_spin.setSuffix(" %")
        add_layout.addRow("Volume:", self.volume_spin)
        
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 100)
        self.weight_spin.setDecimals(1)
        self.weight_spin.setValue(1.0)
        add_layout.addRow("Weight:", self.weight_spin)
        
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(0, 999)
        self.priority_spin.setValue(100)
        add_layout.addRow("Priority:", self.priority_spin)
        
        add_buttons_layout = QHBoxLayout()
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self._add_objective)
        add_buttons_layout.addWidget(self.add_button)
        
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_objective)
        add_buttons_layout.addWidget(self.remove_button)
        
        add_layout.addRow("", add_buttons_layout)
        
        # Create template controls
        template_group = QGroupBox("Templates")
        template_layout = QVBoxLayout(template_group)
        
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "IMRT H&N", "IMRT Prostate", "IMRT Lung", "VMAT Breast", 
            "VMAT SBRT Lung", "VMAT Brain", "Custom..."
        ])
        template_layout.addWidget(self.template_combo)
        
        template_buttons_layout = QHBoxLayout()
        self.load_template_button = QPushButton("Load")
        self.load_template_button.clicked.connect(self._load_template)
        template_buttons_layout.addWidget(self.load_template_button)
        
        self.save_template_button = QPushButton("Save")
        self.save_template_button.clicked.connect(self._save_template)
        template_buttons_layout.addWidget(self.save_template_button)
        
        template_layout.addLayout(template_buttons_layout)
        
        # Create optimization controls
        optimize_group = QGroupBox("Optimization")
        optimize_layout = QVBoxLayout(optimize_group)
        
        self.normalize_checkbox = QCheckBox("Normalize to Prescription")
        self.normalize_checkbox.setChecked(True)
        optimize_layout.addWidget(self.normalize_checkbox)
        
        self.start_button = QPushButton("Start Optimization")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_button.clicked.connect(self._start_optimization)
        optimize_layout.addWidget(self.start_button)
        
        # Add controls to layout
        controls_layout.addWidget(add_group, 3)
        controls_layout.addWidget(template_group, 2)
        controls_layout.addWidget(optimize_group, 2)
        
        # Add widgets to main layout
        main_layout.addWidget(self.objectives_table, 3)
        main_layout.addLayout(controls_layout, 1)
        
        # Apply styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 8px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 3px;
                background-color: #f5f5f5;
            }
            
            QTableWidget {
                border: 1px solid #cccccc;
                gridline-color: #dddddd;
            }
            
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #cccccc;
                font-weight: bold;
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
            
            QComboBox, QDoubleSpinBox, QSpinBox {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 3px;
            }
        """)
        
        # Set up connections
        self.objective_type_combo.currentIndexChanged.connect(self._on_objective_type_changed)
        
        # Initial UI update
        self._on_objective_type_changed(0)
        self._update_ui_state()
    
    def set_structure_set(self, structure_set):
        """Set the structure set and update available structures."""
        self.structure_set = structure_set
        
        # Update structure combo
        self._update_structure_combo()
    
    def set_prescription(self, prescription):
        """Set the prescription to use for optimization."""
        self.prescription = prescription
        
        # Update dose values with prescription
        if self.prescription:
            # Find first target with dose level
            for target in self.prescription.targets:
                if hasattr(target, 'dose_level') and target.dose_level:
                    self.dose_spin.setValue(target.dose_level.dose)
                    break
    
    def get_objectives(self):
        """Get the list of optimization objectives."""
        return self.objectives
    
    def set_objectives(self, objectives):
        """Set the list of optimization objectives."""
        self.objectives = objectives
        
        # Update table
        self._update_objectives_table()
    
    def clear_objectives(self):
        """Clear all optimization objectives."""
        self.objectives = []
        
        # Update table
        self._update_objectives_table()
    
    def _update_structure_combo(self):
        """Update the structure combo with available structures."""
        self.structure_combo.clear()
        
        if not self.structure_set:
            return
        
        # Add all structures
        for structure in self.structure_set.structures:
            self.structure_combo.addItem(structure.name, structure)
    
    def _update_objectives_table(self):
        """Update the objectives table with current objectives."""
        # Clear table
        self.objectives_table.setRowCount(0)
        
        # Add rows for each objective
        for i, objective in enumerate(self.objectives):
            self.objectives_table.insertRow(i)
            
            # Structure
            structure_item = QTableWidgetItem(objective.structure_name)
            self.objectives_table.setItem(i, 0, structure_item)
            
            # Type
            type_item = QTableWidgetItem(self._get_objective_type_name(objective.objective_type))
            self.objectives_table.setItem(i, 1, type_item)
            
            # Dose
            dose_item = QTableWidgetItem(f"{objective.dose:.1f} Gy")
            self.objectives_table.setItem(i, 2, dose_item)
            
            # Volume
            volume_item = QTableWidgetItem(f"{objective.volume:.1f} %" if objective.volume > 0 else "-")
            self.objectives_table.setItem(i, 3, volume_item)
            
            # Weight
            weight_item = QTableWidgetItem(f"{objective.weight:.1f}")
            self.objectives_table.setItem(i, 4, weight_item)
            
            # Priority
            priority_item = QTableWidgetItem(f"{objective.priority}")
            self.objectives_table.setItem(i, 5, priority_item)
            
            # Enabled
            enabled_item = QTableWidgetItem("Yes" if objective.enabled else "No")
            enabled_item.setTextAlignment(Qt.AlignCenter)
            self.objectives_table.setItem(i, 6, enabled_item)
            
            # Set background color for enabled/disabled objectives
            if not objective.enabled:
                brush = QBrush(QColor(240, 240, 240))
                for col in range(7):
                    self.objectives_table.item(i, col).setBackground(brush)
        
        # Emit signal
        self.objectivesChanged.emit()
    
    def _get_objective_type_name(self, objective_type):
        """Get the display name for an objective type."""
        # Handle both enum and string types
        if isinstance(objective_type, str):
            return objective_type
        
        # Handle enum types (from real implementation or placeholder)
        type_names = {
            1: "Lower Dose",  # LOWER_DOSE
            2: "Upper Dose",  # UPPER_DOSE
            3: "Mean Dose",   # MEAN_DOSE
            4: "Max Dose",    # MAX_DOSE
            5: "Min Dose",    # MIN_DOSE
            6: "Lower DVH",   # LOWER_DVH
            7: "Upper DVH",   # UPPER_DVH
            8: "Conformity",  # CONFORMITY
            9: "Homogeneity"  # HOMOGENEITY
        }
        
        # Try to get enum value or use the value itself
        value = getattr(objective_type, "value", objective_type)
        
        return type_names.get(value, "Unknown")
    
    def _get_objective_type_from_name(self, name):
        """Get the objective type enum from a display name."""
        # Map from display name to enum value
        type_map = {
            "Lower Dose": 1,  # LOWER_DOSE
            "Upper Dose": 2,  # UPPER_DOSE
            "Mean Dose": 3,   # MEAN_DOSE
            "Max Dose": 4,    # MAX_DOSE
            "Min Dose": 5,    # MIN_DOSE
            "Lower DVH": 6,   # LOWER_DVH
            "Upper DVH": 7,   # UPPER_DVH
            "Conformity": 8,  # CONFORMITY
            "Homogeneity": 9  # HOMOGENEITY
        }
        
        # Try to convert to real enum if available
        value = type_map.get(name, 0)
        
        try:
            # Try to get the enum by value
            return ObjectiveType(value)
        except (NameError, ValueError):
            # Return the raw value if enum not available
            return value
    
    def _add_objective(self):
        """Add a new objective from the UI controls."""
        # Get selected structure
        structure = self.structure_combo.currentData()
        if not structure:
            return
        
        # Get objective type
        type_name = self.objective_type_combo.currentText()
        objective_type = self._get_objective_type_from_name(type_name)
        
        # Get values
        dose = self.dose_spin.value()
        volume = self.volume_spin.value()
        weight = self.weight_spin.value()
        priority = self.priority_spin.value()
        
        # Create objective
        objective = OptimizationRow(
            structure=structure,
            objective_type=objective_type,
            dose=dose,
            volume=volume,
            weight=weight,
            priority=priority,
            enabled=True
        )
        
        # Add to list
        self.objectives.append(objective)
        
        # Update table
        self._update_objectives_table()
    
    def _remove_objective(self):
        """Remove the selected objective."""
        # Get selected row
        selected_rows = self.objectives_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        # Get row index
        row = selected_rows[0].row()
        
        # Remove from list
        if row < len(self.objectives):
            del self.objectives[row]
        
        # Update table
        self._update_objectives_table()
    
    def _show_context_menu(self, position):
        """Show context menu for objectives table."""
        # Get selected row
        selected_rows = self.objectives_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        # Get row index
        row = selected_rows[0].row()
        
        # Create menu
        menu = QMenu(self)
        
        # Add actions
        edit_action = QAction("Edit Objective", self)
        edit_action.triggered.connect(lambda: self._edit_objective(row))
        menu.addAction(edit_action)
        
        # Toggle enabled action
        is_enabled = self.objectives[row].enabled
        toggle_action = QAction("Disable" if is_enabled else "Enable", self)
        toggle_action.triggered.connect(lambda: self._toggle_objective(row))
        menu.addAction(toggle_action)
        
        menu.addSeparator()
        
        remove_action = QAction("Remove", self)
        remove_action.triggered.connect(self._remove_objective)
        menu.addAction(remove_action)
        
        # Show menu
        menu.exec_(self.objectives_table.mapToGlobal(position))
    
    def _edit_objective(self, row):
        """Edit the objective at the specified row."""
        if row >= len(self.objectives):
            return
        
        objective = self.objectives[row]
        
        # Set UI controls to match objective
        # Structure
        index = self.structure_combo.findText(objective.structure_name)
        if index >= 0:
            self.structure_combo.setCurrentIndex(index)
        
        # Type
        type_name = self._get_objective_type_name(objective.objective_type)
        index = self.objective_type_combo.findText(type_name)
        if index >= 0:
            self.objective_type_combo.setCurrentIndex(index)
        
        # Values
        self.dose_spin.setValue(objective.dose)
        self.volume_spin.setValue(objective.volume)
        self.weight_spin.setValue(objective.weight)
        self.priority_spin.setValue(objective.priority)
        
        # Remove the old objective
        del self.objectives[row]
        
        # Update table
        self._update_objectives_table()
    
    def _toggle_objective(self, row):
        """Toggle the enabled state of the objective at the specified row."""
        if row >= len(self.objectives):
            return
        
        # Toggle enabled state
        self.objectives[row].enabled = not self.objectives[row].enabled
        
        # Update table
        self._update_objectives_table()
    
    def _on_objective_type_changed(self, index):
        """Handle objective type changed."""
        # Get current type
        type_name = self.objective_type_combo.currentText()
        
        # Enable/disable volume control based on type
        is_dvh_type = "DVH" in type_name
        self.volume_spin.setEnabled(is_dvh_type)
        
        # Set appropriate defaults based on type
        if type_name == "Upper Dose" or type_name == "Upper DVH":
            # Upper constraints typically use high doses
            if self.prescription:
                for target in self.prescription.targets:
                    if hasattr(target, 'dose_level') and target.dose_level:
                        self.dose_spin.setValue(target.dose_level.dose * 1.05)
                        break
            
            if is_dvh_type:
                self.volume_spin.setValue(0.0)  # Default to D0 (maximum)
        
        elif type_name == "Lower Dose" or type_name == "Lower DVH":
            # Lower constraints typically use prescription dose
            if self.prescription:
                for target in self.prescription.targets:
                    if hasattr(target, 'dose_level') and target.dose_level:
                        self.dose_spin.setValue(target.dose_level.dose * 0.95)
                        break
            
            if is_dvh_type:
                self.volume_spin.setValue(95.0)  # Default to D95
        
        elif type_name == "Mean Dose":
            # Mean dose typically set to prescription dose
            if self.prescription:
                for target in self.prescription.targets:
                    if hasattr(target, 'dose_level') and target.dose_level:
                        self.dose_spin.setValue(target.dose_level.dose)
                        break
        
        elif type_name == "Max Dose":
            # Max dose typically set to 105-110% of prescription
            if self.prescription:
                for target in self.prescription.targets:
                    if hasattr(target, 'dose_level') and target.dose_level:
                        self.dose_spin.setValue(target.dose_level.dose * 1.07)
                        break
        
        elif type_name == "Min Dose":
            # Min dose typically set to 95-100% of prescription
            if self.prescription:
                for target in self.prescription.targets:
                    if hasattr(target, 'dose_level') and target.dose_level:
                        self.dose_spin.setValue(target.dose_level.dose * 0.95)
                        break
    
    def _load_template(self):
        """Load a template of optimization objectives."""
        # Get selected template
        template_name = self.template_combo.currentText()
        
        if template_name == "Custom...":
            # TODO: Show dialog to select custom template
            return
        
        # Clear existing objectives
        self.objectives = []
        
        # Create template objectives based on selected template
        if template_name == "IMRT H&N":
            # Example H&N template with common objectives
            if self.structure_set:
                # Look for common structures
                for structure in self.structure_set.structures:
                    if "PTV" in structure.name.upper():
                        # PTV objectives
                        ptv_objective = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Lower DVH"),
                            dose=70.0,
                            volume=95.0,
                            weight=80.0,
                            priority=100,
                            enabled=True
                        )
                        self.objectives.append(ptv_objective)
                        
                        # Also add max dose constraint
                        max_objective = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Max Dose"),
                            dose=75.0,
                            volume=0.0,
                            weight=50.0,
                            priority=100,
                            enabled=True
                        )
                        self.objectives.append(max_objective)
                    
                    elif "PAROTID" in structure.name.upper():
                        # Parotid objectives
                        parotid_objective = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Mean Dose"),
                            dose=26.0,
                            volume=0.0,
                            weight=10.0,
                            priority=80,
                            enabled=True
                        )
                        self.objectives.append(parotid_objective)
                    
                    elif "CORD" in structure.name.upper() or "SPINAL" in structure.name.upper():
                        # Spinal cord objectives
                        cord_objective = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Max Dose"),
                            dose=45.0,
                            volume=0.0,
                            weight=70.0,
                            priority=150,
                            enabled=True
                        )
                        self.objectives.append(cord_objective)
                    
                    elif "BRAIN" in structure.name.upper() or "STEM" in structure.name.upper():
                        # Brainstem objectives
                        brain_objective = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Max Dose"),
                            dose=54.0,
                            volume=0.0,
                            weight=70.0,
                            priority=150,
                            enabled=True
                        )
                        self.objectives.append(brain_objective)
                    
                    elif "MANDIBLE" in structure.name.upper():
                        # Mandible objectives
                        mandible_objective = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Max Dose"),
                            dose=70.0,
                            volume=0.0,
                            weight=20.0,
                            priority=80,
                            enabled=True
                        )
                        self.objectives.append(mandible_objective)
        
        elif template_name == "IMRT Prostate":
            # Example prostate template with common objectives
            if self.structure_set:
                # Look for common structures
                for structure in self.structure_set.structures:
                    if "PTV" in structure.name.upper():
                        # PTV objectives
                        ptv_objective = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Lower DVH"),
                            dose=78.0,
                            volume=95.0,
                            weight=100.0,
                            priority=100,
                            enabled=True
                        )
                        self.objectives.append(ptv_objective)
                        
                        # Also add max dose constraint
                        max_objective = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Max Dose"),
                            dose=82.0,
                            volume=0.0,
                            weight=80.0,
                            priority=100,
                            enabled=True
                        )
                        self.objectives.append(max_objective)
                    
                    elif "RECTUM" in structure.name.upper() or "RECT" in structure.name.upper():
                        # Rectum objectives
                        rectum_objective1 = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Upper DVH"),
                            dose=75.0,
                            volume=15.0,
                            weight=60.0,
                            priority=90,
                            enabled=True
                        )
                        self.objectives.append(rectum_objective1)
                        
                        rectum_objective2 = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Upper DVH"),
                            dose=70.0,
                            volume=25.0,
                            weight=40.0,
                            priority=90,
                            enabled=True
                        )
                        self.objectives.append(rectum_objective2)
                    
                    elif "BLADDER" in structure.name.upper():
                        # Bladder objectives
                        bladder_objective1 = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Upper DVH"),
                            dose=80.0,
                            volume=15.0,
                            weight=60.0,
                            priority=90,
                            enabled=True
                        )
                        self.objectives.append(bladder_objective1)
                        
                        bladder_objective2 = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Upper DVH"),
                            dose=75.0,
                            volume=25.0,
                            weight=40.0,
                            priority=90,
                            enabled=True
                        )
                        self.objectives.append(bladder_objective2)
                    
                    elif "FEMUR" in structure.name.upper() or "FEMORAL" in structure.name.upper():
                        # Femoral head objectives
                        femur_objective = OptimizationRow(
                            structure=structure,
                            objective_type=self._get_objective_type_from_name("Max Dose"),
                            dose=50.0,
                            volume=0.0,
                            weight=20.0,
                            priority=80,
                            enabled=True
                        )
                        self.objectives.append(femur_objective)
        
        # Add more templates as needed for other sites
        
        # Update table
        self._update_objectives_table()
    
    def _save_template(self):
        """Save current objectives as a template."""
        # TODO: Implement template saving
        pass
    
    def _start_optimization(self):
        """Start the optimization process."""
        # Emit signal to request optimization start
        self.startOptimizationRequested.emit()
    
    def _update_ui_state(self):
        """Update UI enabled states based on current data."""
        # Enable/disable add button based on whether we have structures
        has_structures = self.structure_combo.count() > 0
        self.add_button.setEnabled(has_structures)
        
        # Enable/disable remove button based on selection
        has_selection = len(self.objectives_table.selectionModel().selectedRows()) > 0
        self.remove_button.setEnabled(has_selection)
        
        # Enable/disable start button based on whether we have objectives
        has_objectives = len(self.objectives) > 0
        self.start_button.setEnabled(has_objectives)

def test_optimization_objective_panel():
    """Test function for the optimization objective panel."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create test structures and structure set
    class TestStructure:
        def __init__(self, name, id=None):
            self.name = name
            self.id = id or name
    
    class TestStructureSet:
        def __init__(self):
            self.structures = [
                TestStructure("PTV"),
                TestStructure("OAR1_Parotid_L"),
                TestStructure("OAR2_Parotid_R"),
                TestStructure("OAR3_SpinalCord"),
                TestStructure("OAR4_Brainstem"),
                TestStructure("BODY")
            ]
    
    # Create test prescription
    class TestPrescription:
        def __init__(self):
            self.targets = []
    
    class TestTarget:
        def __init__(self, name, dose):
            self.name = name
            self.dose_level = TestDoseLevel(dose)
    
    class TestDoseLevel:
        def __init__(self, dose):
            self.dose = dose
    
    # Create test data
    structure_set = TestStructureSet()
    
    prescription = TestPrescription()
    prescription.targets.append(TestTarget("PTV", 70.0))
    
    # Create widget
    widget = OptimizationObjectivePanel()
    widget.set_structure_set(structure_set)
    widget.set_prescription(prescription)
    widget.show()
    
    return app.exec_()

if __name__ == "__main__":
    test_optimization_objective_panel() 