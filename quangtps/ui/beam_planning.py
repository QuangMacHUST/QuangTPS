"""
External beam planning module for QuangTPS.

This module provides components for creating and managing external beam
radiation therapy plans, including beam setup and modification.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTableWidget, QTableWidgetItem, QSplitter, QComboBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QTabWidget, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from quangtps.core.types import BeamParameters, BeamEnergyType, TechniqueType

logger = logging.getLogger(__name__)

class BeamSetupWidget(QWidget):
    """Widget for setting up and modifying radiation therapy beams."""
    
    beam_added = pyqtSignal(object)  # Emits new beam
    beam_modified = pyqtSignal(object, int)  # Emits modified beam and index
    beam_removed = pyqtSignal(int)  # Emits index of removed beam
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.beams = []
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        main_layout = QVBoxLayout(self)
        
        # Beam list
        beam_list_group = QGroupBox("Beam List")
        beam_list_layout = QVBoxLayout(beam_list_group)
        
        self.beam_table = QTableWidget()
        self.beam_table.setColumnCount(6)
        self.beam_table.setHorizontalHeaderLabels([
            "Name", "Energy", "Technique", "Gantry", "Collimator", "Couch"
        ])
        self.beam_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.beam_table.setSelectionMode(QTableWidget.SingleSelection)
        self.beam_table.itemSelectionChanged.connect(self.on_beam_selection_changed)
        beam_list_layout.addWidget(self.beam_table)
        
        # Beam list buttons
        beam_buttons_layout = QHBoxLayout()
        
        self.add_beam_button = QPushButton("Add Beam")
        self.add_beam_button.clicked.connect(self.on_add_beam)
        beam_buttons_layout.addWidget(self.add_beam_button)
        
        self.remove_beam_button = QPushButton("Remove Beam")
        self.remove_beam_button.clicked.connect(self.on_remove_beam)
        self.remove_beam_button.setEnabled(False)
        beam_buttons_layout.addWidget(self.remove_beam_button)
        
        self.clone_beam_button = QPushButton("Clone Beam")
        self.clone_beam_button.clicked.connect(self.on_clone_beam)
        self.clone_beam_button.setEnabled(False)
        beam_buttons_layout.addWidget(self.clone_beam_button)
        
        beam_list_layout.addLayout(beam_buttons_layout)
        main_layout.addWidget(beam_list_group)
        
        # Beam parameters
        beam_params_group = QGroupBox("Beam Parameters")
        beam_params_layout = QFormLayout(beam_params_group)
        
        # Beam name
        self.beam_name_input = QComboBox()
        self.beam_name_input.setEditable(True)
        self.beam_name_input.addItems(["AP", "PA", "LAO", "RAO", "LPO", "RPO"])
        beam_params_layout.addRow("Beam Name:", self.beam_name_input)
        
        # Energy
        self.energy_combo = QComboBox()
        self.energy_combo.addItems([e.name for e in BeamEnergyType])
        beam_params_layout.addRow("Energy:", self.energy_combo)
        
        # Technique
        self.technique_combo = QComboBox()
        self.technique_combo.addItems([t.name for t in TechniqueType])
        beam_params_layout.addRow("Technique:", self.technique_combo)
        
        # Angles
        self.gantry_angle = QDoubleSpinBox()
        self.gantry_angle.setRange(0, 360)
        self.gantry_angle.setDecimals(1)
        self.gantry_angle.setSingleStep(10)
        beam_params_layout.addRow("Gantry Angle (°):", self.gantry_angle)
        
        self.collimator_angle = QDoubleSpinBox()
        self.collimator_angle.setRange(0, 360)
        self.collimator_angle.setDecimals(1)
        self.collimator_angle.setSingleStep(10)
        beam_params_layout.addRow("Collimator Angle (°):", self.collimator_angle)
        
        self.couch_angle = QDoubleSpinBox()
        self.couch_angle.setRange(0, 360)
        self.couch_angle.setDecimals(1)
        self.couch_angle.setSingleStep(10)
        beam_params_layout.addRow("Couch Angle (°):", self.couch_angle)
        
        # Field size
        self.field_x = QDoubleSpinBox()
        self.field_x.setRange(0, 40)
        self.field_x.setDecimals(1)
        self.field_x.setValue(10.0)
        self.field_x.setSingleStep(1)
        beam_params_layout.addRow("Field X (cm):", self.field_x)
        
        self.field_y = QDoubleSpinBox()
        self.field_y.setRange(0, 40)
        self.field_y.setDecimals(1)
        self.field_y.setValue(10.0)
        self.field_y.setSingleStep(1)
        beam_params_layout.addRow("Field Y (cm):", self.field_y)
        
        # Update button
        self.update_beam_button = QPushButton("Update Beam")
        self.update_beam_button.clicked.connect(self.on_update_beam)
        self.update_beam_button.setEnabled(False)
        beam_params_layout.addRow("", self.update_beam_button)
        
        main_layout.addWidget(beam_params_group)
        
        # Weight normalization
        weight_group = QGroupBox("Beam Weighting")
        weight_layout = QVBoxLayout(weight_group)
        
        # Equal weights button
        self.equal_weights_button = QPushButton("Set Equal Weights")
        self.equal_weights_button.clicked.connect(self.on_set_equal_weights)
        weight_layout.addWidget(self.equal_weights_button)
        
        main_layout.addWidget(weight_group)
        
        # Enable/disable beam parameter controls
        self.set_beam_params_enabled(False)
    
    def set_beam_params_enabled(self, enabled):
        """Enable or disable beam parameter controls."""
        for control in [
            self.beam_name_input, self.energy_combo, self.technique_combo,
            self.gantry_angle, self.collimator_angle, self.couch_angle,
            self.field_x, self.field_y, self.update_beam_button
        ]:
            control.setEnabled(enabled)
    
    def on_beam_selection_changed(self):
        """Handle beam selection changed event."""
        selected_rows = self.beam_table.selectedIndexes()
        if not selected_rows:
            self.remove_beam_button.setEnabled(False)
            self.clone_beam_button.setEnabled(False)
            self.set_beam_params_enabled(False)
            return
        
        # Enable buttons
        self.remove_beam_button.setEnabled(True)
        self.clone_beam_button.setEnabled(True)
        self.set_beam_params_enabled(True)
        
        # Get selected beam
        row = selected_rows[0].row()
        if 0 <= row < len(self.beams):
            beam = self.beams[row]
            self.load_beam_parameters(beam)
    
    def load_beam_parameters(self, beam):
        """Load beam parameters into the UI."""
        self.beam_name_input.setCurrentText(beam.name)
        
        # Find energy index
        energy_index = 0
        for i, e in enumerate(BeamEnergyType):
            if e == beam.energy:
                energy_index = i
                break
        self.energy_combo.setCurrentIndex(energy_index)
        
        # Find technique index
        technique_index = 0
        for i, t in enumerate(TechniqueType):
            if t == beam.technique:
                technique_index = i
                break
        self.technique_combo.setCurrentIndex(technique_index)
        
        # Set angles
        self.gantry_angle.setValue(beam.gantry_angle)
        self.collimator_angle.setValue(beam.collimator_angle)
        self.couch_angle.setValue(beam.couch_angle)
        
        # Set field size
        self.field_x.setValue(beam.field_x)
        self.field_y.setValue(beam.field_y)
    
    def get_current_beam_parameters(self):
        """Get beam parameters from the UI."""
        beam = BeamParameters()
        beam.name = self.beam_name_input.currentText()
        beam.energy = BeamEnergyType[self.energy_combo.currentText()]
        beam.technique = TechniqueType[self.technique_combo.currentText()]
        beam.gantry_angle = self.gantry_angle.value()
        beam.collimator_angle = self.collimator_angle.value()
        beam.couch_angle = self.couch_angle.value()
        beam.field_x = self.field_x.value()
        beam.field_y = self.field_y.value()
        return beam
    
    def update_beam_table(self):
        """Update the beam table display."""
        self.beam_table.setRowCount(len(self.beams))
        
        for i, beam in enumerate(self.beams):
            # Name
            self.beam_table.setItem(i, 0, QTableWidgetItem(beam.name))
            
            # Energy
            self.beam_table.setItem(i, 1, QTableWidgetItem(beam.energy.name))
            
            # Technique
            self.beam_table.setItem(i, 2, QTableWidgetItem(beam.technique.name))
            
            # Angles
            self.beam_table.setItem(i, 3, QTableWidgetItem(f"{beam.gantry_angle:.1f}°"))
            self.beam_table.setItem(i, 4, QTableWidgetItem(f"{beam.collimator_angle:.1f}°"))
            self.beam_table.setItem(i, 5, QTableWidgetItem(f"{beam.couch_angle:.1f}°"))
    
    def on_add_beam(self):
        """Add a new beam."""
        # Create new beam with default parameters
        beam = BeamParameters()
        beam.name = f"Beam {len(self.beams) + 1}"
        beam.energy = BeamEnergyType.X6MV
        beam.technique = TechniqueType.STATIC
        beam.gantry_angle = 0.0
        beam.collimator_angle = 0.0
        beam.couch_angle = 0.0
        beam.field_x = 10.0
        beam.field_y = 10.0
        
        # Add to list and update display
        self.beams.append(beam)
        self.update_beam_table()
        
        # Emit signal
        self.beam_added.emit(beam)
        
        # Select the new beam
        self.beam_table.selectRow(len(self.beams) - 1)
    
    def on_remove_beam(self):
        """Remove the selected beam."""
        selected_rows = self.beam_table.selectedIndexes()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        if 0 <= row < len(self.beams):
            # Remove beam and update display
            self.beams.pop(row)
            self.update_beam_table()
            
            # Emit signal
            self.beam_removed.emit(row)
            
            # Update selection or disable controls
            if self.beams:
                self.beam_table.selectRow(min(row, len(self.beams) - 1))
            else:
                self.set_beam_params_enabled(False)
                self.remove_beam_button.setEnabled(False)
                self.clone_beam_button.setEnabled(False)
    
    def on_clone_beam(self):
        """Clone the selected beam."""
        selected_rows = self.beam_table.selectedIndexes()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        if 0 <= row < len(self.beams):
            # Clone beam
            original_beam = self.beams[row]
            new_beam = BeamParameters.from_dict(original_beam.to_dict())
            new_beam.name = f"{original_beam.name} (copy)"
            
            # Add to list and update display
            self.beams.append(new_beam)
            self.update_beam_table()
            
            # Emit signal
            self.beam_added.emit(new_beam)
            
            # Select the new beam
            self.beam_table.selectRow(len(self.beams) - 1)
    
    def on_update_beam(self):
        """Update the selected beam with current parameters."""
        selected_rows = self.beam_table.selectedIndexes()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        if 0 <= row < len(self.beams):
            # Get updated parameters
            updated_beam = self.get_current_beam_parameters()
            
            # Update beam and display
            self.beams[row] = updated_beam
            self.update_beam_table()
            
            # Emit signal
            self.beam_modified.emit(updated_beam, row)
            
            # Maintain selection
            self.beam_table.selectRow(row)
    
    def on_set_equal_weights(self):
        """Set equal weights for all beams."""
        if not self.beams:
            return
        
        weight = 1.0 / len(self.beams)
        for beam in self.beams:
            beam.weight = weight
        
        # Update display
        self.update_beam_table()

class MLCEditorWidget(QWidget):
    """Widget for editing multi-leaf collimator (MLC) shapes."""
    
    mlc_changed = pyqtSignal(object)  # Emits MLC data when changed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.mlc_data = None
        self.selected_leaf = -1
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        main_layout = QVBoxLayout(self)
        
        # Add placeholder - in a real implementation, this would be
        # a graphical MLC editor with leaf positions
        placeholder = QLabel("MLC Editor - Not implemented")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("background-color: #f0f0f0; padding: 20px;")
        main_layout.addWidget(placeholder)
        
        # Basic controls
        controls_layout = QHBoxLayout()
        
        # MLC presets
        preset_combo = QComboBox()
        preset_combo.addItems(["Square", "Rectangle", "Circular", "Custom"])
        controls_layout.addWidget(QLabel("Shape:"))
        controls_layout.addWidget(preset_combo)
        
        # Apply button
        apply_button = QPushButton("Apply Shape")
        controls_layout.addWidget(apply_button)
        
        main_layout.addLayout(controls_layout)

class BeamPlanningTab(QWidget):
    """Tab for external beam planning."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        main_layout = QVBoxLayout(self)
        
        # Create splitter for beam setup and visualization
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - beam setup
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.beam_setup = BeamSetupWidget()
        left_layout.addWidget(self.beam_setup)
        
        splitter.addWidget(left_widget)
        
        # Right side - visualization and MLC editor
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        tab_widget = QTabWidget()
        
        # Beam's eye view tab
        bev_tab = QWidget()
        bev_layout = QVBoxLayout(bev_tab)
        bev_placeholder = QLabel("Beam's Eye View - Not implemented")
        bev_placeholder.setAlignment(Qt.AlignCenter)
        bev_placeholder.setStyleSheet("background-color: #f0f0f0; padding: 20px;")
        bev_layout.addWidget(bev_placeholder)
        tab_widget.addTab(bev_tab, "Beam's Eye View")
        
        # MLC editor tab
        mlc_tab = QWidget()
        mlc_layout = QVBoxLayout(mlc_tab)
        self.mlc_editor = MLCEditorWidget()
        mlc_layout.addWidget(self.mlc_editor)
        tab_widget.addTab(mlc_tab, "MLC Editor")
        
        right_layout.addWidget(tab_widget)
        
        splitter.addWidget(right_widget)
        
        # Set initial sizes
        splitter.setSizes([300, 500])
        
        main_layout.addWidget(splitter)
    
    def set_plan(self, plan):
        """Set the current plan for editing."""
        # Clear existing beams
        self.beam_setup.beams = []
        
        # Add beams from plan
        if plan and plan.beam_set:
            for beam in plan.beam_set:
                self.beam_setup.beams.append(beam)
        
        # Update display
        self.beam_setup.update_beam_table()

# Test function
def test():
    """Test the beam planning component with a sample plan."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create widget
    widget = BeamPlanningTab()
    
    # Create sample plan with beams
    from quangtps.planning.plan import Plan
    
    plan = Plan()
    
    # Add some sample beams
    beam1 = BeamParameters()
    beam1.name = "AP"
    beam1.energy = BeamEnergyType.X6MV
    beam1.technique = TechniqueType.STATIC
    beam1.gantry_angle = 0.0
    beam1.collimator_angle = 0.0
    beam1.couch_angle = 0.0
    beam1.field_x = 10.0
    beam1.field_y = 10.0
    
    beam2 = BeamParameters()
    beam2.name = "LPO"
    beam2.energy = BeamEnergyType.X10MV
    beam2.technique = TechniqueType.STATIC
    beam2.gantry_angle = 90.0
    beam2.collimator_angle = 0.0
    beam2.couch_angle = 0.0
    beam2.field_x = 8.0
    beam2.field_y = 12.0
    
    # Add beams to plan
    plan.beam_set = [beam1, beam2]
    
    # Set plan to widget
    widget.set_plan(plan)
    
    # Show widget
    widget.show()
    
    return app.exec_()

if __name__ == "__main__":
    import sys
    sys.exit(test()) 