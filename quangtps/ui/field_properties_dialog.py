import logging
from typing import Dict, Optional, List, Any

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, 
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QPushButton, QTabWidget, QWidget, QGroupBox, QCheckBox,
    QDialogButtonBox, QFrame, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPixmap

from quangtps.planning.beam import Beam
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.planning.mlc import MLCController
from quangtps.ui.widgets.mlc_editor import MLCEditorWidget
from quangtps.treatment.beams.beam_geometry import BeamGeometry
from quangtps.treatment.techniques.conformal import ConformalBeam
from quangtps.treatment.techniques.imrt import IMRT
from quangtps.ui.beam_eye_view import BeamEyeView
from quangtps.dose.dose_calculator import DoseCalculator
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class FieldPropertiesDialog(QDialog):
    """
    Dialog for editing the properties of a radiation treatment field.
    Provides comprehensive editing capabilities for all beam parameters.
    Mimics the Eclipse field properties dialog.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.beam = None
        self.original_beam = None  # To track changes
        self.machine_registry = ServiceRegistry.get_service("MachineRegistry")
        
        self._init_ui()
        self._connect_signals()
        
        # Set window properties
        self.setWindowTitle("Field Properties")
        self.setMinimumSize(700, 500)
        self.resize(800, 600)
    
    def _init_ui(self):
        """Initialize the UI"""
        main_layout = QVBoxLayout(self)
        
        # Field Name Section
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Field ID:"))
        self.field_id_edit = QLineEdit()
        self.field_id_edit.setMaximumWidth(200)
        name_layout.addWidget(self.field_id_edit)
        
        name_layout.addSpacing(20)
        
        name_layout.addWidget(QLabel("Description:"))
        self.description_edit = QLineEdit()
        name_layout.addWidget(self.description_edit)
        
        main_layout.addLayout(name_layout)
        
        # Tab widget for different property categories
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.setup_tab = self._create_setup_tab()
        self.mlc_tab = self._create_mlc_tab()
        self.calculation_tab = self._create_calculation_tab()
        self.dose_tab = self._create_dose_tab()
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.setup_tab, "Setup")
        self.tab_widget.addTab(self.mlc_tab, "MLC")
        self.tab_widget.addTab(self.calculation_tab, "Calculation")
        self.tab_widget.addTab(self.dose_tab, "Dose")
        
        main_layout.addWidget(self.tab_widget, 1)  # 1 = stretch factor
        
        # Button box
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        main_layout.addWidget(self.button_box)
    
    def _connect_signals(self):
        """Connect UI signals to slots"""
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
    
    def _create_setup_tab(self):
        """Create the Setup tab with beam geometry parameters"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Machine and Energy section
        machine_group = QGroupBox("Machine and Energy")
        machine_layout = QGridLayout(machine_group)
        
        # Machine
        machine_layout.addWidget(QLabel("Machine:"), 0, 0)
        self.machine_combo = QComboBox()
        self._populate_machines()
        machine_layout.addWidget(self.machine_combo, 0, 1)
        
        # Energy
        machine_layout.addWidget(QLabel("Energy:"), 1, 0)
        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["6", "10", "15", "18"])
        machine_layout.addWidget(self.energy_combo, 1, 1)
        
        # Technique
        machine_layout.addWidget(QLabel("Technique:"), 2, 0)
        self.technique_combo = QComboBox()
        self.technique_combo.addItems(["Static", "Arc", "IMRT", "VMAT"])
        machine_layout.addWidget(self.technique_combo, 2, 1)
        
        # Add machine group to layout
        layout.addWidget(machine_group)
        
        # Geometry section
        geometry_group = QGroupBox("Geometry")
        geometry_layout = QGridLayout(geometry_group)
        
        # Gantry
        geometry_layout.addWidget(QLabel("Gantry Angle:"), 0, 0)
        self.gantry_spin = QDoubleSpinBox()
        self.gantry_spin.setRange(0, 359.9)
        self.gantry_spin.setDecimals(1)
        self.gantry_spin.setSingleStep(10)
        geometry_layout.addWidget(self.gantry_spin, 0, 1)
        
        # For arc techniques
        self.arc_frame = QFrame()
        arc_layout = QHBoxLayout(self.arc_frame)
        arc_layout.setContentsMargins(0, 0, 0, 0)
        
        arc_layout.addWidget(QLabel("Stop Angle:"))
        self.stop_angle_spin = QDoubleSpinBox()
        self.stop_angle_spin.setRange(0, 359.9)
        self.stop_angle_spin.setDecimals(1)
        self.stop_angle_spin.setSingleStep(10)
        arc_layout.addWidget(self.stop_angle_spin)
        
        arc_layout.addWidget(QLabel("Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["CW", "CCW"])
        arc_layout.addWidget(self.direction_combo)
        
        geometry_layout.addWidget(self.arc_frame, 0, 2, 1, 2)
        self.arc_frame.setVisible(False)  # Hide initially
        
        # Collimator
        geometry_layout.addWidget(QLabel("Collimator Angle:"), 1, 0)
        self.collimator_spin = QDoubleSpinBox()
        self.collimator_spin.setRange(0, 359.9)
        self.collimator_spin.setDecimals(1)
        self.collimator_spin.setSingleStep(10)
        geometry_layout.addWidget(self.collimator_spin, 1, 1)
        
        # Couch
        geometry_layout.addWidget(QLabel("Couch Angle:"), 2, 0)
        self.couch_spin = QDoubleSpinBox()
        self.couch_spin.setRange(0, 359.9)
        self.couch_spin.setDecimals(1)
        self.couch_spin.setSingleStep(10)
        geometry_layout.addWidget(self.couch_spin, 2, 1)
        
        # SSD
        geometry_layout.addWidget(QLabel("SSD (cm):"), 3, 0)
        self.ssd_spin = QDoubleSpinBox()
        self.ssd_spin.setRange(50, 200)
        self.ssd_spin.setDecimals(1)
        self.ssd_spin.setSingleStep(1)
        geometry_layout.addWidget(self.ssd_spin, 3, 1)
        
        # Add geometry group to layout
        layout.addWidget(geometry_group)
        
        # Field size section
        field_group = QGroupBox("Field Size")
        field_layout = QGridLayout(field_group)
        
        # X1
        field_layout.addWidget(QLabel("X1 (cm):"), 0, 0)
        self.x1_spin = QDoubleSpinBox()
        self.x1_spin.setRange(-20, 0)
        self.x1_spin.setDecimals(1)
        self.x1_spin.setSingleStep(0.5)
        field_layout.addWidget(self.x1_spin, 0, 1)
        
        # X2
        field_layout.addWidget(QLabel("X2 (cm):"), 0, 2)
        self.x2_spin = QDoubleSpinBox()
        self.x2_spin.setRange(0, 20)
        self.x2_spin.setDecimals(1)
        self.x2_spin.setSingleStep(0.5)
        field_layout.addWidget(self.x2_spin, 0, 3)
        
        # Y1
        field_layout.addWidget(QLabel("Y1 (cm):"), 1, 0)
        self.y1_spin = QDoubleSpinBox()
        self.y1_spin.setRange(-20, 0)
        self.y1_spin.setDecimals(1)
        self.y1_spin.setSingleStep(0.5)
        field_layout.addWidget(self.y1_spin, 1, 1)
        
        # Y2
        field_layout.addWidget(QLabel("Y2 (cm):"), 1, 2)
        self.y2_spin = QDoubleSpinBox()
        self.y2_spin.setRange(0, 20)
        self.y2_spin.setDecimals(1)
        self.y2_spin.setSingleStep(0.5)
        field_layout.addWidget(self.y2_spin, 1, 3)
        
        # Field size buttons
        button_layout = QHBoxLayout()
        
        # Common field sizes
        self.field_size_combo = QComboBox()
        self.field_size_combo.addItems(["Custom", "5x5", "10x10", "15x15", "20x20"])
        button_layout.addWidget(QLabel("Presets:"))
        button_layout.addWidget(self.field_size_combo)
        
        # Symmetric/Asymmetric
        self.symmetric_check = QCheckBox("Symmetric")
        button_layout.addWidget(self.symmetric_check)
        
        button_layout.addStretch()
        
        field_layout.addLayout(button_layout, 2, 0, 1, 4)
        
        # Add field group to layout
        layout.addWidget(field_group)
        
        # Add stretch at the end
        layout.addStretch()
        
        return tab
    
    def _create_mlc_tab(self):
        """Create the MLC tab for multi-leaf collimator settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # MLC Model section
        mlc_model_group = QGroupBox("MLC Model")
        mlc_model_layout = QFormLayout(mlc_model_group)
        
        self.mlc_model_combo = QComboBox()
        self.mlc_model_combo.addItems(["Millennium 120", "HD120", "Agility"])
        mlc_model_layout.addRow("MLC Type:", self.mlc_model_combo)
        
        self.mlc_enabled_check = QCheckBox("Enable MLC")
        mlc_model_layout.addRow("", self.mlc_enabled_check)
        
        layout.addWidget(mlc_model_group)
        
        # MLC Editor placeholder
        mlc_editor_group = QGroupBox("MLC Configuration")
        mlc_editor_layout = QVBoxLayout(mlc_editor_group)
        
        mlc_editor_layout.addWidget(QLabel("MLC editor not implemented in this dialog."))
        mlc_editor_layout.addWidget(QLabel("Use the dedicated MLC editor for detailed control."))
        
        self.mlc_editor_button = QPushButton("Open MLC Editor...")
        mlc_editor_layout.addWidget(self.mlc_editor_button)
        
        layout.addWidget(mlc_editor_group)
        
        # For IMRT/VMAT: Control points section
        control_points_group = QGroupBox("Control Points")
        control_points_layout = QFormLayout(control_points_group)
        
        self.num_control_points_spin = QSpinBox()
        self.num_control_points_spin.setRange(2, 200)
        self.num_control_points_spin.setValue(10)
        control_points_layout.addRow("Number of Control Points:", self.num_control_points_spin)
        
        layout.addWidget(control_points_group)
        
        # Add stretch at the end
        layout.addStretch()
        
        return tab
    
    def _create_calculation_tab(self):
        """Create the Calculation tab for dose calculation settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Calculation Algorithm section
        calc_group = QGroupBox("Calculation Algorithm")
        calc_layout = QFormLayout(calc_group)
        
        self.calc_algorithm_combo = QComboBox()
        self.calc_algorithm_combo.addItems(["AAA", "Acuros XB", "Monte Carlo", "Collapsed Cone", "Pencil Beam"])
        calc_layout.addRow("Algorithm:", self.calc_algorithm_combo)
        
        self.calc_grid_size_combo = QComboBox()
        self.calc_grid_size_combo.addItems(["1.0 mm", "1.5 mm", "2.0 mm", "2.5 mm", "3.0 mm"])
        calc_layout.addRow("Grid Size:", self.calc_grid_size_combo)
        
        self.heterogeneity_check = QCheckBox("Heterogeneity Correction")
        self.heterogeneity_check.setChecked(True)
        calc_layout.addRow("", self.heterogeneity_check)
        
        layout.addWidget(calc_group)
        
        # Calculation Options section
        options_group = QGroupBox("Calculation Options")
        options_layout = QFormLayout(options_group)
        
        self.surface_dose_check = QCheckBox("Calculate Surface Dose")
        options_layout.addRow("", self.surface_dose_check)
        
        self.use_reference_point_check = QCheckBox("Use Reference Point")
        options_layout.addRow("", self.use_reference_point_check)
        
        layout.addWidget(options_group)
        
        # Add stretch at the end
        layout.addStretch()
        
        return tab
    
    def _create_dose_tab(self):
        """Create the Dose tab for monitor units and weighting"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Dose section
        dose_group = QGroupBox("Dose")
        dose_layout = QFormLayout(dose_group)
        
        self.mu_spin = QDoubleSpinBox()
        self.mu_spin.setRange(0, 9999)
        self.mu_spin.setDecimals(1)
        self.mu_spin.setSingleStep(10)
        dose_layout.addRow("Monitor Units (MU):", self.mu_spin)
        
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 1)
        self.weight_spin.setDecimals(3)
        self.weight_spin.setSingleStep(0.1)
        dose_layout.addRow("Relative Weight:", self.weight_spin)
        
        layout.addWidget(dose_group)
        
        # Dosimetric Properties section
        dosimetric_group = QGroupBox("Dosimetric Properties")
        dosimetric_layout = QFormLayout(dosimetric_group)
        
        self.dose_rate_combo = QComboBox()
        self.dose_rate_combo.addItems(["600 MU/min", "1000 MU/min", "1400 MU/min", "2400 MU/min"])
        dosimetric_layout.addRow("Dose Rate:", self.dose_rate_combo)
        
        layout.addWidget(dosimetric_group)
        
        # Bolus section
        bolus_group = QGroupBox("Bolus")
        bolus_layout = QFormLayout(bolus_group)
        
        self.bolus_check = QCheckBox("Apply Bolus")
        bolus_layout.addRow("", self.bolus_check)
        
        self.bolus_thickness_spin = QDoubleSpinBox()
        self.bolus_thickness_spin.setRange(0, 10)
        self.bolus_thickness_spin.setDecimals(1)
        self.bolus_thickness_spin.setSingleStep(0.5)
        self.bolus_thickness_spin.setSuffix(" cm")
        self.bolus_thickness_spin.setEnabled(False)  # Disabled initially
        bolus_layout.addRow("Thickness:", self.bolus_thickness_spin)
        
        layout.addWidget(bolus_group)
        
        # Add stretch at the end
        layout.addStretch()
        
        return tab
    
    def _populate_machines(self):
        """Populate the machine combobox with available machines"""
        self.machine_combo.clear()
        
        if self.machine_registry:
            machines = self.machine_registry.get_all_machines()
            machine_names = [machine.name for machine in machines]
            self.machine_combo.addItems(machine_names)
        else:
            # Fallback if machine registry isn't available
            default_machines = ["TrueBeam", "VitalBeam", "Halcyon", "Clinac 21EX", "Unique"]
            self.machine_combo.addItems(default_machines)
    
    def _update_ui_for_technique(self):
        """Update UI based on selected technique"""
        technique = self.technique_combo.currentText()
        
        # Show/hide arc settings
        is_arc = technique in ["Arc", "VMAT"]
        self.arc_frame.setVisible(is_arc)
        
        # Enable/disable MLC settings based on technique
        is_mlc_technique = technique in ["IMRT", "VMAT"]
        self.tab_widget.setTabEnabled(1, is_mlc_technique)  # MLC tab
        
        # Update control points settings
        self.num_control_points_spin.setEnabled(is_mlc_technique)
    
    def _connect_technique_signals(self):
        """Connect signals related to technique selection"""
        self.technique_combo.currentTextChanged.connect(self._update_ui_for_technique)
        self.bolus_check.toggled.connect(self.bolus_thickness_spin.setEnabled)
        self.field_size_combo.currentTextChanged.connect(self._on_field_size_preset)
        self.symmetric_check.toggled.connect(self._on_symmetric_toggled)
    
    def _on_field_size_preset(self, preset):
        """Handle field size preset selection"""
        if preset == "Custom":
            return
        
        # Parse field size from preset (e.g., "10x10" -> 10)
        try:
            size = float(preset.split("x")[0])
            half_size = size / 2.0
            
            # Set field size
            self.x1_spin.setValue(-half_size)
            self.x2_spin.setValue(half_size)
            self.y1_spin.setValue(-half_size)
            self.y2_spin.setValue(half_size)
        except (ValueError, IndexError):
            pass
    
    def _on_symmetric_toggled(self, checked):
        """Handle symmetric field checkbox toggle"""
        if checked:
            # Make field symmetric based on current values
            max_x = max(abs(self.x1_spin.value()), abs(self.x2_spin.value()))
            max_y = max(abs(self.y1_spin.value()), abs(self.y2_spin.value()))
            
            self.x1_spin.setValue(-max_x)
            self.x2_spin.setValue(max_x)
            self.y1_spin.setValue(-max_y)
            self.y2_spin.setValue(max_y)
            
            # Connect value change signals for keeping symmetry
            self.x1_spin.valueChanged.connect(self._keep_x_symmetry)
            self.x2_spin.valueChanged.connect(self._keep_x_symmetry)
            self.y1_spin.valueChanged.connect(self._keep_y_symmetry)
            self.y2_spin.valueChanged.connect(self._keep_y_symmetry)
        else:
            # Disconnect symmetry signals
            self.x1_spin.valueChanged.disconnect(self._keep_x_symmetry)
            self.x2_spin.valueChanged.disconnect(self._keep_x_symmetry)
            self.y1_spin.valueChanged.disconnect(self._keep_y_symmetry)
            self.y2_spin.valueChanged.disconnect(self._keep_y_symmetry)
    
    def _keep_x_symmetry(self, value):
        """Keep X field size symmetric"""
        sender = self.sender()
        if sender == self.x1_spin:
            self.x2_spin.blockSignals(True)
            self.x2_spin.setValue(-value)
            self.x2_spin.blockSignals(False)
        elif sender == self.x2_spin:
            self.x1_spin.blockSignals(True)
            self.x1_spin.setValue(-value)
            self.x1_spin.blockSignals(False)
    
    def _keep_y_symmetry(self, value):
        """Keep Y field size symmetric"""
        sender = self.sender()
        if sender == self.y1_spin:
            self.y2_spin.blockSignals(True)
            self.y2_spin.setValue(-value)
            self.y2_spin.blockSignals(False)
        elif sender == self.y2_spin:
            self.y1_spin.blockSignals(True)
            self.y1_spin.setValue(-value)
            self.y1_spin.blockSignals(False)
    
    def set_beam(self, beam: Beam):
        """Set the beam to be edited"""
        self.beam = beam
        self.original_beam = beam  # Store the original
        
        # Update UI with beam properties
        self._populate_ui_from_beam()
        
        # Connect signals after populating to avoid triggering callbacks
        self._connect_technique_signals()
    
    def _populate_ui_from_beam(self):
        """Populate UI fields with beam properties"""
        if not self.beam:
            return
        
        # Basic properties
        self.field_id_edit.setText(self.beam.id)
        self.description_edit.setText(getattr(self.beam, 'description', ''))
        
        # Machine and energy
        if hasattr(self.beam, 'machine') and self.beam.machine:
            machine_name = self.beam.machine.name
            index = self.machine_combo.findText(machine_name)
            if index >= 0:
                self.machine_combo.setCurrentIndex(index)
        
        if hasattr(self.beam, 'energy') and self.beam.energy:
            index = self.energy_combo.findText(str(self.beam.energy))
            if index >= 0:
                self.energy_combo.setCurrentIndex(index)
        
        # Technique
        technique_map = {
            'Static': 'Static',
            'ConformalBeam': 'Static',
            'IMRT': 'IMRT',
            'VMAT': 'VMAT',
            'Arc': 'Arc'
        }
        
        technique_name = getattr(self.beam, 'technique', None)
        if technique_name:
            if hasattr(technique_name, 'name'):
                technique_name = technique_name.name
            
            technique = technique_map.get(technique_name, 'Static')
            index = self.technique_combo.findText(technique)
            if index >= 0:
                self.technique_combo.setCurrentIndex(index)
        
        # Geometry
        if hasattr(self.beam, 'gantry_angle'):
            self.gantry_spin.setValue(self.beam.gantry_angle)
        
        if hasattr(self.beam, 'stop_angle'):
            self.stop_angle_spin.setValue(self.beam.stop_angle)
        
        if hasattr(self.beam, 'collimator_angle'):
            self.collimator_spin.setValue(self.beam.collimator_angle)
        
        if hasattr(self.beam, 'couch_angle'):
            self.couch_spin.setValue(self.beam.couch_angle)
        
        if hasattr(self.beam, 'ssd') and self.beam.ssd is not None:
            self.ssd_spin.setValue(self.beam.ssd)
        
        # Field size
        if hasattr(self.beam, 'x1') and self.beam.x1 is not None:
            self.x1_spin.setValue(self.beam.x1)
        
        if hasattr(self.beam, 'x2') and self.beam.x2 is not None:
            self.x2_spin.setValue(self.beam.x2)
        
        if hasattr(self.beam, 'y1') and self.beam.y1 is not None:
            self.y1_spin.setValue(self.beam.y1)
        
        if hasattr(self.beam, 'y2') and self.beam.y2 is not None:
            self.y2_spin.setValue(self.beam.y2)
        
        # Dose parameters
        if hasattr(self.beam, 'mu') and self.beam.mu is not None:
            self.mu_spin.setValue(self.beam.mu)
        
        if hasattr(self.beam, 'weight') and self.beam.weight is not None:
            self.weight_spin.setValue(self.beam.weight)
        
        # MLC settings
        has_mlc = hasattr(self.beam, 'mlc') and self.beam.mlc is not None
        self.mlc_enabled_check.setChecked(has_mlc)
        
        if has_mlc:
            # MLC model
            mlc_model = getattr(self.beam.mlc, 'model', 'Millennium 120')
            index = self.mlc_model_combo.findText(mlc_model)
            if index >= 0:
                self.mlc_model_combo.setCurrentIndex(index)
            
            # Number of control points
            if hasattr(self.beam.mlc, 'control_points'):
                num_control_points = len(self.beam.mlc.control_points)
                self.num_control_points_spin.setValue(num_control_points)
        
        # Update UI for technique
        self._update_ui_for_technique()
    
    def accept(self):
        """Override accept to update beam properties"""
        if not self.beam:
            super().accept()
            return
        
        # Update beam properties from UI
        self._update_beam_from_ui()
        
        super().accept()
    
    def _update_beam_from_ui(self):
        """Update beam properties from UI fields"""
        # Basic properties
        self.beam.id = self.field_id_edit.text()
        self.beam.description = self.description_edit.text()
        
        # Machine and energy
        machine_name = self.machine_combo.currentText()
        if self.machine_registry:
            self.beam.machine = self.machine_registry.get_machine(machine_name)
        
        self.beam.energy = self.energy_combo.currentText()
        
        # Technique - would require more complex handling in a real implementation
        # For now, we just store the technique name
        technique_name = self.technique_combo.currentText()
        if hasattr(self.beam, 'technique') and self.beam.technique:
            if hasattr(self.beam.technique, 'name'):
                self.beam.technique.name = technique_name
        
        # Geometry
        self.beam.gantry_angle = self.gantry_spin.value()
        self.beam.collimator_angle = self.collimator_spin.value()
        self.beam.couch_angle = self.couch_spin.value()
        self.beam.ssd = self.ssd_spin.value()
        
        # For arc techniques
        if self.technique_combo.currentText() in ["Arc", "VMAT"]:
            self.beam.stop_angle = self.stop_angle_spin.value()
            self.beam.direction = self.direction_combo.currentText()
        
        # Field size
        self.beam.x1 = self.x1_spin.value()
        self.beam.x2 = self.x2_spin.value()
        self.beam.y1 = self.y1_spin.value()
        self.beam.y2 = self.y2_spin.value()
        
        # Dose parameters
        self.beam.mu = self.mu_spin.value()
        self.beam.weight = self.weight_spin.value()
        
        # MLC settings
        if self.mlc_enabled_check.isChecked():
            if not hasattr(self.beam, 'mlc') or self.beam.mlc is None:
                # Create MLC if it doesn't exist
                self.beam.mlc = MLCController()
            
            # Set MLC model
            self.beam.mlc.model = self.mlc_model_combo.currentText()
            
            # Control points would be handled by dedicated MLC editor
        else:
            # Remove MLC if disabled
            self.beam.mlc = None 