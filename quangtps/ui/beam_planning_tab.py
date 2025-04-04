"""
Beam Planning Tab Module for QuangTPS.

This module provides an Eclipse-like interface for beam planning and management,
integrating with the beam visualization system.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
    from PyQt5.QtGui import QIcon, QFont, QColor
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
        QComboBox, QGroupBox, QTableWidget, QTableWidgetItem, 
        QSplitter, QTabWidget, QFrame, QHeaderView, QCheckBox,
        QDoubleSpinBox, QSpinBox, QLineEdit, QFormLayout,
        QMessageBox, QMenu, QAction, QToolBar, QFileDialog, QStatusBar
    )
except ImportError as e:
    logging.error(f"Unable to import PyQt5: {e}")

try:
    from quangtps.planning.beam import Beam
    from quangtps.planning.plan import Plan
    from quangtps.planning.mlc import MLC, MLCType
    from quangtps.ui.beam_visualization import BeamVisualizationPanel
except ImportError as e:
    logging.error(f"Error importing planning modules: {e}")

logger = logging.getLogger(__name__)

class BeamTableWidget(QTableWidget):
    """
    Table widget for displaying and managing beams.
    
    This widget displays a list of beams with their properties
    and allows for selection, editing, and management of beams.
    """
    
    beamSelected = pyqtSignal(Beam)
    beamChanged = pyqtSignal(Beam)
    
    def __init__(self, parent=None):
        """Initialize the beam table widget."""
        super().__init__(parent)
        self.beams = []
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Set up table
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels([
            "Name", "Energy", "MU", "Gantry", "Collimator", "Field Size", "Use"
        ])
        
        # Set selection behavior
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)
        
        # Set column widths
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Connect signals
        self.itemSelectionChanged.connect(self.on_selection_changed)
        self.itemChanged.connect(self.on_item_changed)
        
    def set_beams(self, beams: List[Beam]):
        """Set the beams to display in the table."""
        self.beams = beams
        self.update_table()
        
    def add_beam(self, beam: Beam):
        """Add a beam to the table."""
        self.beams.append(beam)
        self.update_table()
        return len(self.beams) - 1  # Return index of added beam
        
    def remove_beam(self, beam: Beam):
        """Remove a beam from the table."""
        if beam in self.beams:
            index = self.beams.index(beam)
            self.beams.remove(beam)
            self.update_table()
            return index
        return -1
        
    def update_table(self):
        """Update the table display with current beam data."""
        # Disconnect signals during update to prevent unwanted events
        try:
            self.itemSelectionChanged.disconnect()
            self.itemChanged.disconnect()
        except TypeError:
            # Signals were not connected
            pass
            
        # Set up table rows
        self.setRowCount(len(self.beams))
        
        # Fill table with beam data
        for row, beam in enumerate(self.beams):
            # Name
            name_item = QTableWidgetItem(beam.name if hasattr(beam, 'name') else f"Beam {row+1}")
            name_item.setData(Qt.UserRole, beam)
            self.setItem(row, 0, name_item)
            
            # Energy
            energy_item = QTableWidgetItem(beam.energy if hasattr(beam, 'energy') else "6MV")
            self.setItem(row, 1, energy_item)
            
            # MU
            mu_item = QTableWidgetItem(str(beam.mu) if hasattr(beam, 'mu') else "100")
            self.setItem(row, 2, mu_item)
            
            # Gantry
            gantry_item = QTableWidgetItem(f"{beam.gantry_angle:.1f}°" if hasattr(beam, 'gantry_angle') else "0.0°")
            self.setItem(row, 3, gantry_item)
            
            # Collimator
            colli_item = QTableWidgetItem(f"{beam.collimator_angle:.1f}°" if hasattr(beam, 'collimator_angle') else "0.0°")
            self.setItem(row, 4, colli_item)
            
            # Field Size
            if hasattr(beam, 'x_jaw_pos') and hasattr(beam, 'y_jaw_pos'):
                field_size = f"{beam.x_jaw_pos + beam.x_jaw_neg:.1f} × {beam.y_jaw_pos + beam.y_jaw_neg:.1f}"
            else:
                field_size = "10 × 10"
            field_item = QTableWidgetItem(field_size)
            self.setItem(row, 5, field_item)
            
            # Use (Checkbox)
            use_item = QTableWidgetItem()
            use_item.setCheckState(Qt.Checked if getattr(beam, 'use', True) else Qt.Unchecked)
            self.setItem(row, 6, use_item)
            
        # Reconnect signals
        self.itemSelectionChanged.connect(self.on_selection_changed)
        self.itemChanged.connect(self.on_item_changed)
        
    def get_selected_beam(self) -> Optional[Beam]:
        """Get the currently selected beam."""
        selected_rows = self.selectionModel().selectedRows()
        if not selected_rows:
            return None
            
        row = selected_rows[0].row()
        beam_item = self.item(row, 0)
        if beam_item:
            return beam_item.data(Qt.UserRole)
        return None
        
    def on_selection_changed(self):
        """Handle selection changes in the table."""
        beam = self.get_selected_beam()
        if beam:
            self.beamSelected.emit(beam)
            
    def on_item_changed(self, item):
        """Handle item changes in the table."""
        row = item.row()
        col = item.column()
        
        if row < 0 or row >= len(self.beams):
            return
            
        beam = self.beams[row]
        
        # Update beam based on changed column
        try:
            if col == 0:  # Name
                beam.name = item.text()
            elif col == 1:  # Energy
                beam.energy = item.text()
            elif col == 2:  # MU
                beam.mu = float(item.text())
            elif col == 3:  # Gantry
                value = float(item.text().rstrip('°'))
                beam.gantry_angle = value
            elif col == 4:  # Collimator
                value = float(item.text().rstrip('°'))
                beam.collimator_angle = value
            elif col == 5:  # Field Size
                # Parse "X × Y" format
                parts = item.text().split('×')
                if len(parts) == 2:
                    x_size = float(parts[0].strip())
                    y_size = float(parts[1].strip())
                    # Assuming symmetric fields for simplicity
                    beam.x_jaw_pos = x_size / 2
                    beam.x_jaw_neg = x_size / 2
                    beam.y_jaw_pos = y_size / 2
                    beam.y_jaw_neg = y_size / 2
            elif col == 6:  # Use
                beam.use = (item.checkState() == Qt.Checked)
                
            # Update table display with new values
            self.update_table()
            
            # Notify about beam change
            self.beamChanged.emit(beam)
        except Exception as e:
            logger.error(f"Error updating beam: {e}")
            # Revert to original values
            self.update_table()


class BeamPropertiesWidget(QWidget):
    """
    Widget for editing beam properties.
    
    This widget provides form fields for editing beam properties
    such as name, energy, gantry angle, collimator angle, etc.
    """
    
    beamChanged = pyqtSignal(Beam)
    
    def __init__(self, parent=None):
        """Initialize the beam properties widget."""
        super().__init__(parent)
        self.beam = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        
        # Beam Name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter beam name")
        form_layout.addRow("Name:", self.name_edit)
        
        # Energy Selection
        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["6MV", "10MV", "15MV", "6FFF", "10FFF"])
        form_layout.addRow("Energy:", self.energy_combo)
        
        # Gantry Angle
        self.gantry_spin = QDoubleSpinBox()
        self.gantry_spin.setRange(0, 359.9)
        self.gantry_spin.setDecimals(1)
        self.gantry_spin.setSingleStep(10)
        self.gantry_spin.setSuffix("°")
        form_layout.addRow("Gantry:", self.gantry_spin)
        
        # Collimator Angle
        self.colli_spin = QDoubleSpinBox()
        self.colli_spin.setRange(0, 359.9)
        self.colli_spin.setDecimals(1)
        self.colli_spin.setSingleStep(45)
        self.colli_spin.setSuffix("°")
        form_layout.addRow("Collimator:", self.colli_spin)
        
        # Field Size X
        self.field_x_spin = QDoubleSpinBox()
        self.field_x_spin.setRange(0, 40)
        self.field_x_spin.setDecimals(1)
        self.field_x_spin.setSingleStep(1)
        self.field_x_spin.setSuffix(" cm")
        form_layout.addRow("Field Width:", self.field_x_spin)
        
        # Field Size Y
        self.field_y_spin = QDoubleSpinBox()
        self.field_y_spin.setRange(0, 40)
        self.field_y_spin.setDecimals(1)
        self.field_y_spin.setSingleStep(1)
        self.field_y_spin.setSuffix(" cm")
        form_layout.addRow("Field Height:", self.field_y_spin)
        
        # MU
        self.mu_spin = QDoubleSpinBox()
        self.mu_spin.setRange(0, 9999)
        self.mu_spin.setDecimals(0)
        self.mu_spin.setSingleStep(10)
        form_layout.addRow("MU:", self.mu_spin)
        
        # MLC Type
        self.mlc_combo = QComboBox()
        self.mlc_combo.addItems(["None", "5mm", "10mm", "HD120"])
        form_layout.addRow("MLC:", self.mlc_combo)
        
        # Add to main layout
        main_layout.addLayout(form_layout)
        
        # Add MLC button
        self.mlc_button = QPushButton("Edit MLC...")
        self.mlc_button.setEnabled(False)
        main_layout.addWidget(self.mlc_button)
        
        # Add spacer
        main_layout.addStretch()
        
        # Connect signals
        self.name_edit.textChanged.connect(self.on_field_changed)
        self.energy_combo.currentTextChanged.connect(self.on_field_changed)
        self.gantry_spin.valueChanged.connect(self.on_field_changed)
        self.colli_spin.valueChanged.connect(self.on_field_changed)
        self.field_x_spin.valueChanged.connect(self.on_field_changed)
        self.field_y_spin.valueChanged.connect(self.on_field_changed)
        self.mu_spin.valueChanged.connect(self.on_field_changed)
        self.mlc_combo.currentTextChanged.connect(self.on_mlc_changed)
        self.mlc_button.clicked.connect(self.on_edit_mlc)
        
        # Set initial state
        self.set_enabled(False)
        
    def set_enabled(self, enabled: bool):
        """Enable or disable form fields."""
        self.name_edit.setEnabled(enabled)
        self.energy_combo.setEnabled(enabled)
        self.gantry_spin.setEnabled(enabled)
        self.colli_spin.setEnabled(enabled)
        self.field_x_spin.setEnabled(enabled)
        self.field_y_spin.setEnabled(enabled)
        self.mu_spin.setEnabled(enabled)
        self.mlc_combo.setEnabled(enabled)
        
    def set_beam(self, beam: Optional[Beam]):
        """Set the beam to edit."""
        self.beam = beam
        
        if not beam:
            self.set_enabled(False)
            return
            
        # Disconnect signals during update
        try:
            self.name_edit.textChanged.disconnect()
            self.energy_combo.currentTextChanged.disconnect()
            self.gantry_spin.valueChanged.disconnect()
            self.colli_spin.valueChanged.disconnect()
            self.field_x_spin.valueChanged.disconnect()
            self.field_y_spin.valueChanged.disconnect()
            self.mu_spin.valueChanged.disconnect()
            self.mlc_combo.currentTextChanged.disconnect()
        except TypeError:
            # Signals were not connected
            pass
            
        # Update fields with beam properties
        self.name_edit.setText(beam.name if hasattr(beam, 'name') else "")
        self.energy_combo.setCurrentText(beam.energy if hasattr(beam, 'energy') else "6MV")
        self.gantry_spin.setValue(beam.gantry_angle if hasattr(beam, 'gantry_angle') else 0)
        self.colli_spin.setValue(beam.collimator_angle if hasattr(beam, 'collimator_angle') else 0)
        
        # Field sizes
        if hasattr(beam, 'x_jaw_pos') and hasattr(beam, 'x_jaw_neg'):
            self.field_x_spin.setValue(beam.x_jaw_pos + beam.x_jaw_neg)
        else:
            self.field_x_spin.setValue(10.0)
            
        if hasattr(beam, 'y_jaw_pos') and hasattr(beam, 'y_jaw_neg'):
            self.field_y_spin.setValue(beam.y_jaw_pos + beam.y_jaw_neg)
        else:
            self.field_y_spin.setValue(10.0)
            
        # MU
        self.mu_spin.setValue(beam.mu if hasattr(beam, 'mu') else 100)
        
        # MLC
        if hasattr(beam, 'mlc') and beam.mlc:
            if isinstance(beam.mlc, MLC):
                mlc_type = beam.mlc.mlc_type
                if mlc_type == MLCType.HD120:
                    self.mlc_combo.setCurrentText("HD120")
                    self.mlc_button.setEnabled(True)
                elif mlc_type == MLCType.MILLENNIUM_120:
                    self.mlc_combo.setCurrentText("10mm")
                    self.mlc_button.setEnabled(True)
                elif mlc_type == MLCType.MILLENNIUM_120_HD:
                    self.mlc_combo.setCurrentText("5mm")
                    self.mlc_button.setEnabled(True)
                else:
                    self.mlc_combo.setCurrentText("None")
                    self.mlc_button.setEnabled(False)
            else:
                self.mlc_combo.setCurrentText("None")
                self.mlc_button.setEnabled(False)
        else:
            self.mlc_combo.setCurrentText("None")
            self.mlc_button.setEnabled(False)
            
        # Reconnect signals
        self.name_edit.textChanged.connect(self.on_field_changed)
        self.energy_combo.currentTextChanged.connect(self.on_field_changed)
        self.gantry_spin.valueChanged.connect(self.on_field_changed)
        self.colli_spin.valueChanged.connect(self.on_field_changed)
        self.field_x_spin.valueChanged.connect(self.on_field_changed)
        self.field_y_spin.valueChanged.connect(self.on_field_changed)
        self.mu_spin.valueChanged.connect(self.on_field_changed)
        self.mlc_combo.currentTextChanged.connect(self.on_mlc_changed)
        
        # Enable form
        self.set_enabled(True)
        
    def on_field_changed(self):
        """Handle form field changes."""
        if not self.beam:
            return
            
        try:
            # Update beam with form values
            self.beam.name = self.name_edit.text()
            self.beam.energy = self.energy_combo.currentText()
            self.beam.gantry_angle = self.gantry_spin.value()
            self.beam.collimator_angle = self.colli_spin.value()
            
            # Field sizes (assuming symmetric for simplicity)
            x_size = self.field_x_spin.value()
            y_size = self.field_y_spin.value()
            self.beam.x_jaw_pos = x_size / 2
            self.beam.x_jaw_neg = x_size / 2
            self.beam.y_jaw_pos = y_size / 2
            self.beam.y_jaw_neg = y_size / 2
            
            # MU
            self.beam.mu = self.mu_spin.value()
            
            # Notify about beam change
            self.beamChanged.emit(self.beam)
        except Exception as e:
            logger.error(f"Error updating beam properties: {e}")
            
    def on_mlc_changed(self, mlc_type: str):
        """Handle MLC type changes."""
        if not self.beam:
            return
            
        try:
            # Update MLC based on selected type
            if mlc_type == "None":
                self.beam.mlc = None
                self.mlc_button.setEnabled(False)
            else:
                if not hasattr(self.beam, 'mlc') or not self.beam.mlc:
                    # Create new MLC
                    if mlc_type == "HD120":
                        self.beam.mlc = MLC(MLCType.HD120)
                    elif mlc_type == "10mm":
                        self.beam.mlc = MLC(MLCType.MILLENNIUM_120)
                    elif mlc_type == "5mm":
                        self.beam.mlc = MLC(MLCType.MILLENNIUM_120_HD)
                else:
                    # Update existing MLC type
                    if mlc_type == "HD120":
                        self.beam.mlc.mlc_type = MLCType.HD120
                    elif mlc_type == "10mm":
                        self.beam.mlc.mlc_type = MLCType.MILLENNIUM_120
                    elif mlc_type == "5mm":
                        self.beam.mlc.mlc_type = MLCType.MILLENNIUM_120_HD
                        
                # Enable MLC button
                self.mlc_button.setEnabled(True)
                
            # Notify about beam change
            self.beamChanged.emit(self.beam)
        except Exception as e:
            logger.error(f"Error updating MLC type: {e}")
            
    def on_edit_mlc(self):
        """Open MLC editor for the current beam."""
        if not self.beam or not hasattr(self.beam, 'mlc') or not self.beam.mlc:
            return
            
        # In a real implementation, this would open the MLC editor dialog
        QMessageBox.information(self, "MLC Editor", 
                              "MLC Editor would open here for configuring leaf positions.")


class BeamPlanningTab(QWidget):
    """
    Tab for planning radiation beams.
    
    This tab provides an Eclipse-like interface for beam planning,
    integrating a beam table, beam properties editor, and beam visualization.
    """
    
    def __init__(self, parent=None):
        """Initialize the beam planning tab."""
        super().__init__(parent)
        self.plan = None
        self.image_data = None
        self.structure_set = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for main content
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel (beam list and properties)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create beam list section
        beam_list_group = QGroupBox("Beams")
        beam_list_layout = QVBoxLayout(beam_list_group)
        
        # Create toolbar for beam actions
        beam_toolbar = QToolBar()
        beam_toolbar.setIconSize(QSize(16, 16))
        
        # Add beam action
        add_beam_action = QAction(QIcon(get_icon_path("add")), "Add Beam", self)
        add_beam_action.triggered.connect(self.on_add_beam)
        beam_toolbar.addAction(add_beam_action)
        
        # Copy beam action
        copy_beam_action = QAction(QIcon(get_icon_path("copy")), "Copy Beam", self)
        copy_beam_action.triggered.connect(self.on_copy_beam)
        beam_toolbar.addAction(copy_beam_action)
        
        # Delete beam action
        delete_beam_action = QAction(QIcon(get_icon_path("delete")), "Delete Beam", self)
        delete_beam_action.triggered.connect(self.on_delete_beam)
        beam_toolbar.addAction(delete_beam_action)
        
        beam_toolbar.addSeparator()
        
        # Technique selection
        technique_label = QLabel("Technique:")
        beam_toolbar.addWidget(technique_label)
        
        self.technique_combo = QComboBox()
        self.technique_combo.addItem("Conformal")
        self.technique_combo.addItem("Arc")
        self.technique_combo.addItem("IMRT")
        beam_toolbar.addWidget(self.technique_combo)
        
        # Add setup buttons for different techniques
        conformal_setup_action = QAction(QIcon(get_icon_path("setup")), "Conformal Setup", self)
        conformal_setup_action.triggered.connect(self.on_conformal_setup)
        beam_toolbar.addAction(conformal_setup_action)
        
        arc_setup_action = QAction(QIcon(get_icon_path("arc")), "Arc Setup", self)
        arc_setup_action.triggered.connect(self.on_arc_setup)
        beam_toolbar.addAction(arc_setup_action)
        
        imrt_setup_action = QAction(QIcon(get_icon_path("imrt")), "IMRT Setup", self)
        imrt_setup_action.triggered.connect(self.on_imrt_setup)
        beam_toolbar.addAction(imrt_setup_action)
        
        beam_list_layout.addWidget(beam_toolbar)
        
        # Create beam table
        self.beam_table = BeamTableWidget()
        self.beam_table.beamSelected.connect(self.on_beam_selected)
        self.beam_table.beamChanged.connect(self.on_beam_changed)
        beam_list_layout.addWidget(self.beam_table)
        
        left_layout.addWidget(beam_list_group, 1)  # 1 = stretch factor
        
        # Create beam properties section
        beam_props_group = QGroupBox("Beam Properties")
        beam_props_layout = QVBoxLayout(beam_props_group)
        
        # Create beam properties widget
        self.beam_properties = BeamPropertiesWidget()
        self.beam_properties.beamChanged.connect(self.on_beam_changed)
        beam_props_layout.addWidget(self.beam_properties)
        
        left_layout.addWidget(beam_props_group, 1)  # 1 = stretch factor
        
        # Add Eclipse-like tab section for beam configuration
        beam_config_tabs = QTabWidget()
        
        # Field tab
        field_tab = QWidget()
        field_layout = QFormLayout(field_tab)
        
        # Machine/energy selection
        machine_label = QLabel("Machine:")
        self.machine_combo = QComboBox()
        self.machine_combo.addItems(["TrueBeam", "VitalBeam", "Halcyon"])
        field_layout.addRow(machine_label, self.machine_combo)
        
        energy_label = QLabel("Energy:")
        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["6X", "10X", "15X", "6FFF", "10FFF"])
        field_layout.addRow(energy_label, self.energy_combo)
        
        # Dose rate selection
        dose_rate_label = QLabel("Dose Rate:")
        self.dose_rate_combo = QComboBox()
        self.dose_rate_combo.addItems(["Low", "Medium", "High", "Max"])
        field_layout.addRow(dose_rate_label, self.dose_rate_combo)
        
        # SSD setting
        ssd_label = QLabel("SSD (cm):")
        self.ssd_spin = QDoubleSpinBox()
        self.ssd_spin.setRange(70.0, 150.0)
        self.ssd_spin.setValue(100.0)
        self.ssd_spin.setDecimals(1)
        field_layout.addRow(ssd_label, self.ssd_spin)
        
        beam_config_tabs.addTab(field_tab, "Field")
        
        # Jaw/MLC tab
        jaw_mlc_tab = QWidget()
        jaw_mlc_layout = QFormLayout(jaw_mlc_tab)
        
        # X1 jaw
        x1_label = QLabel("X1 (cm):")
        self.x1_spin = QDoubleSpinBox()
        self.x1_spin.setRange(-20.0, 0.0)
        self.x1_spin.setValue(-5.0)
        self.x1_spin.setDecimals(1)
        jaw_mlc_layout.addRow(x1_label, self.x1_spin)
        
        # X2 jaw
        x2_label = QLabel("X2 (cm):")
        self.x2_spin = QDoubleSpinBox()
        self.x2_spin.setRange(0.0, 20.0)
        self.x2_spin.setValue(5.0)
        self.x2_spin.setDecimals(1)
        jaw_mlc_layout.addRow(x2_label, self.x2_spin)
        
        # Y1 jaw
        y1_label = QLabel("Y1 (cm):")
        self.y1_spin = QDoubleSpinBox()
        self.y1_spin.setRange(-20.0, 0.0)
        self.y1_spin.setValue(-5.0)
        self.y1_spin.setDecimals(1)
        jaw_mlc_layout.addRow(y1_label, self.y1_spin)
        
        # Y2 jaw
        y2_label = QLabel("Y2 (cm):")
        self.y2_spin = QDoubleSpinBox()
        self.y2_spin.setRange(0.0, 20.0)
        self.y2_spin.setValue(5.0)
        self.y2_spin.setDecimals(1)
        jaw_mlc_layout.addRow(y2_label, self.y2_spin)
        
        # MLC type selection
        mlc_label = QLabel("MLC Type:")
        self.mlc_combo = QComboBox()
        self.mlc_combo.addItems(["Millennium 120", "HD-MLC", "None"])
        jaw_mlc_layout.addRow(mlc_label, self.mlc_combo)
        
        # MLC editor button
        self.edit_mlc_button = QPushButton("Edit MLC...")
        self.edit_mlc_button.clicked.connect(self.on_edit_mlc)
        jaw_mlc_layout.addRow("", self.edit_mlc_button)
        
        beam_config_tabs.addTab(jaw_mlc_tab, "Jaw/MLC")
        
        # Wedge tab
        wedge_tab = QWidget()
        wedge_layout = QFormLayout(wedge_tab)
        
        # Wedge type selection
        wedge_type_label = QLabel("Wedge Type:")
        self.wedge_type_combo = QComboBox()
        self.wedge_type_combo.addItems(["None", "Enhanced Dynamic Wedge", "Physical Wedge"])
        wedge_layout.addRow(wedge_type_label, self.wedge_type_combo)
        
        # Wedge angle
        wedge_angle_label = QLabel("Wedge Angle:")
        self.wedge_angle_combo = QComboBox()
        self.wedge_angle_combo.addItems(["15°", "30°", "45°", "60°"])
        self.wedge_angle_combo.setEnabled(False)
        wedge_layout.addRow(wedge_angle_label, self.wedge_angle_combo)
        
        # Wedge orientation
        wedge_orient_label = QLabel("Orientation:")
        self.wedge_orient_combo = QComboBox()
        self.wedge_orient_combo.addItems(["IN", "OUT", "LEFT", "RIGHT"])
        self.wedge_orient_combo.setEnabled(False)
        wedge_layout.addRow(wedge_orient_label, self.wedge_orient_combo)
        
        # Connect wedge type changes
        self.wedge_type_combo.currentTextChanged.connect(self._on_wedge_type_changed)
        
        beam_config_tabs.addTab(wedge_tab, "Wedge")
        
        # Applicator tab (for electron beams)
        applicator_tab = QWidget()
        applicator_layout = QFormLayout(applicator_tab)
        
        # Applicator type
        applicator_label = QLabel("Applicator:")
        self.applicator_combo = QComboBox()
        self.applicator_combo.addItems(["None", "6×6", "10×10", "15×15", "20×20", "25×25"])
        applicator_layout.addRow(applicator_label, self.applicator_combo)
        
        # Insert
        insert_label = QLabel("Insert:")
        self.insert_combo = QComboBox()
        self.insert_combo.addItems(["None", "Circle 1cm", "Circle 2cm", "Circle 3cm", "Custom"])
        applicator_layout.addRow(insert_label, self.insert_combo)
        
        beam_config_tabs.addTab(applicator_tab, "Applicator")
        
        # Add tabs to properties section
        beam_props_layout.addWidget(beam_config_tabs)
        
        # Add left panel to main splitter
        main_splitter.addWidget(left_panel)
        
        # Create right panel (visualization)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create visualization widget
        self.beam_visualization = BeamVisualizationPanel()
        right_layout.addWidget(self.beam_visualization)
        
        # Add right panel to main splitter
        main_splitter.addWidget(right_panel)
        
        # Set stretch factors
        main_splitter.setStretchFactor(0, 1)  # Left panel
        main_splitter.setStretchFactor(1, 2)  # Right panel
        
        # Add splitter to main layout
        main_layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        main_layout.addWidget(self.status_bar)
        
        # Disable beam properties until a beam is selected
        self.beam_properties.set_enabled(False)
        
        # Connect wedge type changes
        self.wedge_type_combo.currentTextChanged.connect(self._on_wedge_type_changed)
        
        # Connect technique changes
        self.technique_combo.currentTextChanged.connect(self._on_technique_changed)
        
    def set_plan(self, plan: Optional[Plan]):
        """Set the plan for beam planning."""
        self.plan = plan
        
        # Update UI elements with plan data
        if plan and hasattr(plan, 'beam_set') and hasattr(plan.beam_set, 'beams'):
            self.beam_table.set_beams(plan.beam_set.beams)
            self.beam_visualization.set_beams(plan.beam_set.beams)
            
            # Enable buttons if plan is available
            self.add_beam_btn.setEnabled(True)
            self.conformal_btn.setEnabled(True)
            self.arc_btn.setEnabled(True)
            self.imrt_btn.setEnabled(True)
        else:
            # Clear UI elements
            self.beam_table.set_beams([])
            self.beam_visualization.set_beams([])
            self.beam_properties.set_beam(None)
            
            # Disable buttons
            self.add_beam_btn.setEnabled(False)
            self.copy_beam_btn.setEnabled(False)
            self.delete_beam_btn.setEnabled(False)
            self.conformal_btn.setEnabled(False)
            self.arc_btn.setEnabled(False)
            self.imrt_btn.setEnabled(False)
            
    def set_image_data(self, image_data):
        """Set the image data for beam planning."""
        self.image_data = image_data
        self.beam_visualization.set_image_data(image_data)
        
    def set_structure_set(self, structure_set):
        """Set the structure set for beam planning."""
        self.structure_set = structure_set
        self.beam_visualization.set_structure_set(structure_set)
        
    def on_add_beam(self):
        """Add a new beam to the plan."""
        if not self.plan or not hasattr(self.plan, 'beam_set'):
            return
            
        try:
            # Create a new beam
            beam_name = f"Beam {len(self.plan.beam_set.beams) + 1}"
            new_beam = Beam(name=beam_name)
            
            # Set default properties
            new_beam.energy = "6MV"
            new_beam.gantry_angle = 0
            new_beam.collimator_angle = 0
            new_beam.mu = 100
            new_beam.x_jaw_pos = 5.0
            new_beam.x_jaw_neg = 5.0
            new_beam.y_jaw_pos = 5.0
            new_beam.y_jaw_neg = 5.0
            
            # Add beam to the plan
            self.plan.beam_set.add_beam(new_beam)
            
            # Update UI
            self.beam_table.set_beams(self.plan.beam_set.beams)
            self.beam_visualization.set_beams(self.plan.beam_set.beams)
            
            # Select the new beam
            self.beam_table.selectRow(len(self.plan.beam_set.beams) - 1)
        except Exception as e:
            logger.error(f"Error adding beam: {e}")
            QMessageBox.warning(self, "Error", f"Failed to add beam: {str(e)}")
            
    def on_copy_beam(self):
        """Copy the selected beam."""
        if not self.plan or not hasattr(self.plan, 'beam_set'):
            return
            
        try:
            # Get selected beam
            beam = self.beam_table.get_selected_beam()
            if not beam:
                return
                
            # Create a copy
            import copy
            new_beam = copy.deepcopy(beam)
            
            # Update name
            new_beam.name = f"{beam.name} Copy"
            
            # Add beam to the plan
            self.plan.beam_set.add_beam(new_beam)
            
            # Update UI
            self.beam_table.set_beams(self.plan.beam_set.beams)
            self.beam_visualization.set_beams(self.plan.beam_set.beams)
            
            # Select the new beam
            self.beam_table.selectRow(len(self.plan.beam_set.beams) - 1)
        except Exception as e:
            logger.error(f"Error copying beam: {e}")
            QMessageBox.warning(self, "Error", f"Failed to copy beam: {str(e)}")
            
    def on_delete_beam(self):
        """Delete the selected beam."""
        if not self.plan or not hasattr(self.plan, 'beam_set'):
            return
            
        try:
            # Get selected beam
            beam = self.beam_table.get_selected_beam()
            if not beam:
                return
                
            # Confirm deletion
            result = QMessageBox.question(
                self, "Confirm Deletion", 
                f"Are you sure you want to delete beam '{beam.name}'?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if result != QMessageBox.Yes:
                return
                
            # Remove beam from the plan
            self.plan.beam_set.remove_beam(beam)
            
            # Update UI
            self.beam_table.set_beams(self.plan.beam_set.beams)
            self.beam_visualization.set_beams(self.plan.beam_set.beams)
            
            # Clear properties if no beams left
            if not self.plan.beam_set.beams:
                self.beam_properties.set_beam(None)
                self.copy_beam_btn.setEnabled(False)
                self.delete_beam_btn.setEnabled(False)
        except Exception as e:
            logger.error(f"Error deleting beam: {e}")
            QMessageBox.warning(self, "Error", f"Failed to delete beam: {str(e)}")
            
    def on_beam_selected(self, beam: Beam):
        """Handle beam selection in the table."""
        self.beam_properties.set_beam(beam)
        self.beam_visualization.set_current_beam(beam)
        
        # Enable/disable buttons
        self.copy_beam_btn.setEnabled(True)
        self.delete_beam_btn.setEnabled(True)
        
    def on_beam_changed(self, beam: Beam):
        """Handle beam property changes."""
        # Update table
        self.beam_table.update_table()
        
        # Update visualization
        self.beam_visualization.set_current_beam(beam)
        
    def on_conformal_setup(self):
        """Set up conformal beams for the current plan."""
        if not self.plan or not self.structure_set:
            QMessageBox.warning(self, "Error", "Plan or structure set not available")
            return
            
        # Ask user for target structure
        structures = self.structure_set.structures if hasattr(self.structure_set, 'structures') else []
        if not structures:
            QMessageBox.warning(self, "Error", "No structures available for conformal setup")
            return
            
        # In a real implementation, this would open a dialog to select the target
        # For this example, just show a message
        QMessageBox.information(
            self, "Conformal Setup", 
            "This would create conformal beams around the selected target structure."
        )
        
    def on_arc_setup(self):
        """Set up arc beams for the current plan."""
        if not self.plan or not self.structure_set:
            QMessageBox.warning(self, "Error", "Plan or structure set not available")
            return
            
        # In a real implementation, this would open a dialog to set up arc parameters
        # For this example, just show a message
        QMessageBox.information(
            self, "Arc Setup", 
            "This would create arc beams with specified parameters."
        )
        
    def on_imrt_setup(self):
        """Set up IMRT beams for the current plan."""
        if not self.plan or not self.structure_set:
            QMessageBox.warning(self, "Error", "Plan or structure set not available")
            return
            
        # In a real implementation, this would open a dialog to set up IMRT parameters
        # For this example, just show a message
        QMessageBox.information(
            self, "IMRT Setup", 
            "This would create IMRT beams with specified parameters and optimization objectives."
        )

    def _on_wedge_type_changed(self, wedge_type):
        """Handle wedge type changes."""
        # Enable/disable wedge controls based on type
        has_wedge = wedge_type != "None"
        self.wedge_angle_combo.setEnabled(has_wedge)
        self.wedge_orient_combo.setEnabled(has_wedge)
        
        # Update current beam if one is selected
        current_beam = self.beam_table.get_selected_beam()
        if current_beam and hasattr(current_beam, 'modifiers'):
            if hasattr(current_beam.modifiers, 'wedge'):
                if has_wedge:
                    # Set wedge properties
                    current_beam.modifiers.wedge.type = wedge_type
                    current_beam.modifiers.wedge.angle = float(self.wedge_angle_combo.currentText().replace('°', ''))
                    current_beam.modifiers.wedge.orientation = self.wedge_orient_combo.currentText()
                else:
                    # Remove wedge
                    current_beam.modifiers.wedge = None
                
                # Update beam display
                self.on_beam_changed(current_beam)

    def _on_technique_changed(self, technique):
        """Handle technique changes."""
        # Update UI based on selected technique
        if technique == "Conformal":
            # Enable/disable relevant controls
            self.mlc_combo.setEnabled(True)
            self.wedge_type_combo.setEnabled(True)
            self._on_wedge_type_changed(self.wedge_type_combo.currentText())
        elif technique == "Arc":
            # Enable/disable relevant controls
            self.mlc_combo.setEnabled(True)
            self.wedge_type_combo.setEnabled(False)
            self.wedge_angle_combo.setEnabled(False)
            self.wedge_orient_combo.setEnabled(False)
        elif technique == "IMRT":
            # Enable/disable relevant controls
            self.mlc_combo.setEnabled(True)
            self.wedge_type_combo.setEnabled(False)
            self.wedge_angle_combo.setEnabled(False)
            self.wedge_orient_combo.setEnabled(False)
            
        # Update current beam if one is selected
        current_beam = self.beam_table.get_selected_beam()
        if current_beam:
            # Set technique
            current_beam.technique = technique
            
            # Update beam display
            self.on_beam_changed(current_beam)


def test_beam_planning_tab():
    """Test function for the beam planning tab."""
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    
    # Create test data
    class TestBeam(Beam):
        def __init__(self, name, gantry_angle, collimator_angle):
            self.name = name
            self.gantry_angle = gantry_angle
            self.collimator_angle = collimator_angle
            self.energy = "6MV"
            self.x_jaw_pos = 5.0
            self.x_jaw_neg = 5.0
            self.y_jaw_pos = 5.0
            self.y_jaw_neg = 5.0
            self.mu = 100
            self.mlc = None
            self.use = True
            
    class TestBeamSet:
        def __init__(self):
            self.beams = [
                TestBeam("Anterior", 0, 0),
                TestBeam("Left Lateral", 90, 0),
                TestBeam("Right Lateral", 270, 0),
                TestBeam("Posterior", 180, 0),
            ]
            
        def add_beam(self, beam):
            self.beams.append(beam)
            
        def remove_beam(self, beam):
            if beam in self.beams:
                self.beams.remove(beam)
    
    class TestPlan:
        def __init__(self):
            self.beam_set = TestBeamSet()
    
    # Create test image data
    class TestImage:
        def __init__(self):
            self.shape = (100, 512, 512)
            
    # Create test structure
    class TestStructure:
        def __init__(self, name, color):
            self.name = name
            self.color = color
            
    # Create test structure set
    class TestStructureSet:
        def __init__(self):
            self.structures = [
                TestStructure("PTV", "#FF0000"),
                TestStructure("CTV", "#00FF00"),
                TestStructure("OAR", "#0000FF")
            ]
    
    # Create main window
    main_window = QMainWindow()
    
    # Create test data
    test_plan = TestPlan()
    test_image = TestImage()
    test_structure_set = TestStructureSet()
    
    # Create beam planning tab
    planning_tab = BeamPlanningTab()
    planning_tab.set_image_data(test_image)
    planning_tab.set_structure_set(test_structure_set)
    planning_tab.set_plan(test_plan)
    
    # Set as central widget
    main_window.setCentralWidget(planning_tab)
    main_window.setWindowTitle("QuangTPS - Beam Planning")
    main_window.resize(1200, 800)
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_beam_planning_tab()
