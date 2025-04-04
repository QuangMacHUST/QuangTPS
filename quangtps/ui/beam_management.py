import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QGroupBox, QDialog, QDialogButtonBox, QFormLayout, QHeaderView,
    QMessageBox, QTabWidget, QSplitter, QFrame, QMenu, QAction,
    QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QColor, QCursor

from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.planning.mlc import MLCController
from quangtps.planning.prescription import Prescription
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.beams.beam_geometry import BeamGeometry
from quangtps.treatment.techniques.treatment_technique import TreatmentTechnique
from quangtps.treatment.techniques.conformal import ConformalBeam
from quangtps.treatment.techniques.imrt import IMRT
from quangtps.treatment.techniques.vmat import VMAT
from quangtps.ui.field_properties_dialog import FieldPropertiesDialog
from quangtps.core.logging import get_logger
from quangtps.core.services import ServiceRegistry

logger = get_logger(__name__)

class BeamManagementWidget(QWidget):
    """
    Widget for managing beams in a treatment plan.
    Provides functionality for adding, editing, deleting, and configuring beams.
    Mimics the Eclipse beam management interface.
    """
    
    beam_added = pyqtSignal(Beam)
    beam_modified = pyqtSignal(Beam)
    beam_removed = pyqtSignal(str)  # beam_id
    beam_selected = pyqtSignal(Beam)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.plan: Optional[Plan] = None
        self.prescription: Optional[Prescription] = None
        self.machines: Dict[str, TreatmentMachine] = {}
        self.selected_beam: Optional[Beam] = None
        self.service_registry = ServiceRegistry()
        
        # Initialize UI
        self._init_ui()
        self._connect_signals()
        self._populate_machines()
    
    def _init_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Top section - Prescription summary and machine selection
        top_frame = QFrame()
        top_frame.setFrameShape(QFrame.StyledPanel)
        top_layout = QHBoxLayout(top_frame)
        
        # Prescription summary
        prescription_group = QGroupBox("Prescription")
        prescription_layout = QVBoxLayout(prescription_group)
        self.prescription_label = QLabel("No prescription set")
        prescription_layout.addWidget(self.prescription_label)
        top_layout.addWidget(prescription_group, 2)
        
        # Machine selection
        machine_group = QGroupBox("Treatment Machine")
        machine_layout = QFormLayout(machine_group)
        self.machine_combo = QComboBox()
        self.technique_combo = QComboBox()
        
        # Add treatment techniques
        self.technique_combo.addItem("Conformal")
        self.technique_combo.addItem("IMRT")
        self.technique_combo.addItem("VMAT")
        self.technique_combo.addItem("SBRT")
        self.technique_combo.addItem("SRS")
        
        machine_layout.addRow("Machine:", self.machine_combo)
        machine_layout.addRow("Technique:", self.technique_combo)
        top_layout.addWidget(machine_group, 1)
        
        main_layout.addWidget(top_frame)
        
        # Middle section - Beam table
        self.beam_table = QTableWidget()
        self.beam_table.setColumnCount(9)
        self.beam_table.setHorizontalHeaderLabels([
            "ID", "Technique", "Energy", "Gantry", "Collimator", "Couch", "MU", "X1-X2", "Y1-Y2"
        ])
        self.beam_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.beam_table.setSelectionMode(QTableWidget.SingleSelection)
        self.beam_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.beam_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.beam_table.verticalHeader().setVisible(False)
        
        main_layout.addWidget(self.beam_table, 1)
        
        # Bottom section - Action buttons
        button_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add")
        self.edit_button = QPushButton("Edit")
        self.delete_button = QPushButton("Delete")
        self.copy_button = QPushButton("Copy")
        
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.copy_button)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
    
    def _connect_signals(self):
        # Button actions
        self.add_button.clicked.connect(self._on_add_beam)
        self.edit_button.clicked.connect(self._on_edit_beam)
        self.delete_button.clicked.connect(self._on_delete_beam)
        self.copy_button.clicked.connect(self._on_copy_beam)
        
        # Table selection
        self.beam_table.itemSelectionChanged.connect(self._on_beam_selection_changed)
        self.beam_table.customContextMenuRequested.connect(self._show_context_menu)
        
        # Machine and technique changes
        self.machine_combo.currentTextChanged.connect(self._on_machine_changed)
        self.technique_combo.currentTextChanged.connect(self._on_technique_changed)
    
    def _populate_machines(self):
        # Get machines from service registry
        machine_service = self.service_registry.get_service("MachineService")
        if machine_service:
            self.machines = machine_service.get_all_machines()
            
            # Populate combo box
            self.machine_combo.clear()
            for machine_id, machine in self.machines.items():
                self.machine_combo.addItem(machine.name, machine_id)
    
    def set_plan(self, plan: Plan):
        """Set the current plan and update the UI."""
        self.plan = plan
        self._update_ui()
    
    def set_prescription(self, prescription: Prescription):
        """Set the prescription and update the summary."""
        self.prescription = prescription
        self._update_prescription_summary()
    
    def _update_ui(self):
        # Update buttons state
        has_plan = self.plan is not None
        
        self.add_button.setEnabled(has_plan)
        self.edit_button.setEnabled(has_plan and self.selected_beam is not None)
        self.delete_button.setEnabled(has_plan and self.selected_beam is not None)
        self.copy_button.setEnabled(has_plan and self.selected_beam is not None)
        
        # Update beam table
        if has_plan:
            self._populate_beam_table()
            self._update_prescription_summary()
    
    def _update_prescription_summary(self):
        if self.prescription:
            text = f"Target: {self.prescription.target_volume}\n"
            text += f"Dose: {self.prescription.dose:.1f} Gy in {self.prescription.num_fractions} fractions"
            self.prescription_label.setText(text)
        else:
            self.prescription_label.setText("No prescription set")
    
    def _populate_beam_table(self):
        if not self.plan:
            return
        
        # Clear table
        self.beam_table.setRowCount(0)
        
        # Add beams
        row = 0
        for beam_id, beam in self.plan.beams.items():
            self.beam_table.insertRow(row)
            
            # ID
            id_item = QTableWidgetItem(beam_id)
            id_item.setData(Qt.UserRole, beam_id)
            self.beam_table.setItem(row, 0, id_item)
            
            # Technique
            technique = getattr(beam, 'technique', 'Conformal')
            technique_item = QTableWidgetItem(technique)
            self.beam_table.setItem(row, 1, technique_item)
            
            # Energy
            energy = getattr(beam, 'energy', '6X')
            energy_item = QTableWidgetItem(energy)
            self.beam_table.setItem(row, 2, energy_item)
            
            # Gantry angle
            gantry = getattr(beam, 'gantry_angle', 0)
            gantry_item = QTableWidgetItem(f"{gantry:.1f}°")
            self.beam_table.setItem(row, 3, gantry_item)
            
            # Collimator angle
            collimator = getattr(beam, 'collimator_angle', 0)
            collimator_item = QTableWidgetItem(f"{collimator:.1f}°")
            self.beam_table.setItem(row, 4, collimator_item)
            
            # Couch angle
            couch = getattr(beam, 'couch_angle', 0)
            couch_item = QTableWidgetItem(f"{couch:.1f}°")
            self.beam_table.setItem(row, 5, couch_item)
            
            # MU
            mu = getattr(beam, 'mu', 0)
            mu_item = QTableWidgetItem(f"{mu:.1f}")
            self.beam_table.setItem(row, 6, mu_item)
            
            # Field size X
            x1 = getattr(beam, 'x1', -5)
            x2 = getattr(beam, 'x2', 5)
            x_item = QTableWidgetItem(f"{x1:.1f} - {x2:.1f}")
            self.beam_table.setItem(row, 7, x_item)
            
            # Field size Y
            y1 = getattr(beam, 'y1', -5)
            y2 = getattr(beam, 'y2', 5)
            y_item = QTableWidgetItem(f"{y1:.1f} - {y2:.1f}")
            self.beam_table.setItem(row, 8, y_item)
            
            row += 1
    
    def _on_beam_selection_changed(self):
        # Get selected beam
        selected_rows = self.beam_table.selectedItems()
        if not selected_rows:
            self.selected_beam = None
            self.edit_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self.copy_button.setEnabled(False)
            return
        
        # Get beam ID from first column
        selected_row = self.beam_table.selectedItems()[0].row()
        beam_id = self.beam_table.item(selected_row, 0).data(Qt.UserRole)
        
        # Get beam
        if self.plan and beam_id in self.plan.beams:
            self.selected_beam = self.plan.beams[beam_id]
            self.beam_selected.emit(self.selected_beam)
            
            # Enable buttons
            self.edit_button.setEnabled(True)
            self.delete_button.setEnabled(True)
            self.copy_button.setEnabled(True)
    
    def _on_add_beam(self):
        if not self.plan:
            return
        
        # Create a new beam
        beam_id = self._generate_beam_id()
        
        # Get selected machine
        machine_id = self.machine_combo.currentData()
        machine = self._get_machine(machine_id)
        
        # Create beam with selected technique
        technique_name = self.technique_combo.currentText()
        technique = self._create_technique(technique_name)
        
        # Create a new beam
        new_beam = Beam(
            beam_id=beam_id,
            machine=machine,
            technique=technique,
            gantry_angle=0,
            collimator_angle=0,
            couch_angle=0,
            isocenter=(0, 0, 0),
            mu=200
        )
        
        # Default field size
        new_beam.x1 = -5.0
        new_beam.x2 = 5.0
        new_beam.y1 = -5.0
        new_beam.y2 = 5.0
        
        # Add to plan
        self.plan.add_beam(new_beam)
        
        # Refresh table
        self._populate_beam_table()
        
        # Select the newly added beam
        for row in range(self.beam_table.rowCount()):
            if self.beam_table.item(row, 0).data(Qt.UserRole) == new_beam.beam_id:
                self.beam_table.selectRow(row)
                break
        
        # Open edit dialog
        self._edit_beam(new_beam, is_new=True)
        
        # Emit signal
        self.beam_added.emit(new_beam)
    
    def _on_edit_beam(self):
        if not self.selected_beam:
            return
        
        self._edit_beam(self.selected_beam)
    
    def _on_delete_beam(self):
        if not self.plan or not self.selected_beam:
            return
        
        # Confirm deletion
        result = QMessageBox.question(
            self,
            "Delete Beam",
            f"Are you sure you want to delete beam {self.selected_beam.beam_id}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if result != QMessageBox.Yes:
            return
        
        beam_id = self.selected_beam.beam_id
        
        # Remove from plan
        self.plan.remove_beam(beam_id)
        
        # Update UI
        self._populate_beam_table()
        
        # Reset selection
        self.selected_beam = None
        
        # Emit signal
        self.beam_removed.emit(beam_id)
    
    def _on_copy_beam(self):
        if not self.plan or not self.selected_beam:
            return
        
        # Generate new beam ID
        new_beam_id = self._generate_beam_id()
        
        # Clone the beam
        new_beam = self.selected_beam.clone()
        new_beam.beam_id = new_beam_id
        
        # Add to plan
        self.plan.add_beam(new_beam)
        
        # Update UI
        self._populate_beam_table()
        
        # Select the newly added beam
        for row in range(self.beam_table.rowCount()):
            if self.beam_table.item(row, 0).data(Qt.UserRole) == new_beam.beam_id:
                self.beam_table.selectRow(row)
                break
        
        # Emit signal
        self.beam_added.emit(new_beam)
    
    def _edit_beam(self, beam, is_new=False):
        # Open beam properties dialog
        dialog = FieldPropertiesDialog(self)
        dialog.set_beam(beam)
        
        if dialog.exec_() == QDialog.Accepted:
            # Refresh table
            self._populate_beam_table()
            
            # Emit signal
            self.beam_modified.emit(beam)
    
    def _generate_beam_id(self):
        """Generate a unique beam ID."""
        if not self.plan:
            return "Field 1"
        
        # Get existing beam IDs
        existing_ids = set(self.plan.beams.keys())
        
        index = 1
        while True:
            beam_id = f"Field {index}"
            if beam_id not in existing_ids:
                return beam_id
            index += 1
    
    def _get_machine(self, machine_id):
        """Get machine by ID."""
        if not machine_id:
            # Return default machine
            machine_service = self.service_registry.get_service("MachineService")
            if machine_service:
                return machine_service.get_default_machine()
            
            # Fallback to a basic linac
            return Linac(
                name="Default Linac",
                machine_id="default_linac"
            )
        
        # Get from cache
        if machine_id in self.machines:
            return self.machines[machine_id]
        
        # Get from service
        machine_service = self.service_registry.get_service("MachineService")
        if machine_service:
            return machine_service.get_machine(machine_id)
        
        # Fallback
        return Linac(
            name="Default Linac",
            machine_id=machine_id
        )
    
    def _create_technique(self, technique_name):
        """Create a treatment technique based on the selected name."""
        if technique_name == "Conformal":
            return ConformalBeam()
        elif technique_name == "IMRT":
            return IMRT()
        elif technique_name == "VMAT":
            return VMAT()
        else:
            # Default to conformal if technique not recognized
            return ConformalBeam()
    
    def _on_technique_changed(self, technique_name):
        """Handle technique change in the combo box."""
        # This method can be expanded to update the UI based on the selected technique
        logger.debug(f"Selected technique: {technique_name}")
    
    def _on_machine_changed(self, machine_name):
        """Handle machine change in the combo box."""
        logger.debug(f"Selected machine: {machine_name}")
    
    def _show_context_menu(self, position):
        """Show context menu for beam table."""
        if not self.selected_beam:
            return
        
        menu = QMenu(self)
        
        edit_action = menu.addAction("Edit")
        edit_action.triggered.connect(self._on_edit_beam)
        
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(self._on_copy_beam)
        
        menu.addSeparator()
        
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(self._on_delete_beam)
        
        menu.exec_(self.beam_table.viewport().mapToGlobal(position))
