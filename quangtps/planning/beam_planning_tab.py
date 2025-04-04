"""
Beam planning tab for QuangTPS.

This module provides a tab for beam setup and planning in the QuangTPS application.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, cast
import logging
import numpy as np
import math

from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QTableWidget, QTableWidgetItem, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget,
    QCheckBox, QMenu, QAction, QHeaderView, QMessageBox,
    QToolBar, QToolButton, QFrame, QSizePolicy
)
from PyQt5.QtGui import QIcon, QPainter, QPen, QColor, QBrush, QFont

# Try to import required classes, use placeholders if not available
try:
    from quangtps.beams.beam import Beam, BeamSet, BeamType
    from quangtps.beams.beam_geometry import BeamGeometry
    from quangtps.beams.beam_modifiers import Wedge, Block, Bolus, Compensator
    from quangtps.beams.mlc import MLC, MLCType
except ImportError:
    logging.warning("Failed to import beam classes, using placeholders")
    class Beam:
        def __init__(self, name=""):
            self.id = ""
            self.name = name
            self.energy = "6MV"
            self.gantry_angle = 0.0
            self.collimator_angle = 0.0
            self.couch_angle = 0.0
            self.field_size = (10.0, 10.0)
            self.isocenter = (0.0, 0.0, 0.0)
            self.weight = 1.0
            self.mlc = None
            self.modifiers = []
    
    class BeamSet:
        def __init__(self, name=""):
            self.id = ""
            self.name = name
            self.beams = []
            self.prescription_dose = 0.0
    
    class BeamType:
        STATIC = "static"
        ARC = "arc"
        DYNAMIC = "dynamic"
        
    class BeamGeometry:
        def __init__(self):
            pass
    
    class Wedge:
        def __init__(self, name=""):
            self.id = ""
            self.name = name
            self.angle = 15.0
            
    class Block:
        def __init__(self, name=""):
            self.id = ""
            self.name = name
            
    class Bolus:
        def __init__(self, name=""):
            self.id = ""
            self.name = name
            
    class Compensator:
        def __init__(self, name=""):
            self.id = ""
            self.name = name
            
    class MLC:
        def __init__(self, name=""):
            self.id = ""
            self.name = name
            
    class MLCType:
        STATIC = "static"
        DYNAMIC = "dynamic"
        STEP_AND_SHOOT = "step_and_shoot"

try:
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure import Structure
except ImportError:
    logging.warning("Failed to import structure classes, using placeholders")
    class Structure:
        def __init__(self, name=""):
            self.id = ""
            self.name = name
            self.type = ""
            self.color = (255, 0, 0)
            
    class StructureSet:
        def __init__(self, name=""):
            self.id = ""
            self.name = name
            self.structures = []

logger = logging.getLogger(__name__)

class BeamAngleVisualizer(QWidget):
    """Widget for visualizing beam angles in a BEV diagram"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.beams = []
        self.current_beam_index = -1
        self.isocenter = (0, 0, 0)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def set_beams(self, beams: List[Beam]):
        """Set the beams to visualize"""
        self.beams = beams
        self.update()
        
    def set_current_beam(self, index: int):
        """Set the currently selected beam"""
        if index >= -1 and (index < len(self.beams) or index == -1):
            self.current_beam_index = index
            self.update()
            
    def set_isocenter(self, isocenter: Tuple[float, float, float]):
        """Set the isocenter position"""
        self.isocenter = isocenter
        self.update()
        
    def paintEvent(self, event):
        """Paint the beam angle diagram"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        radius = min(width, height) / 2 - 20
        
        # Draw outer circle (patient outline)
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(QColor(240, 240, 240)))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw coordinate axes
        painter.setPen(QPen(Qt.gray, 1, Qt.DashLine))
        painter.drawLine(center_x - radius, center_y, center_x + radius, center_y)  # X axis
        painter.drawLine(center_x, center_y - radius, center_x, center_y + radius)  # Y axis
        
        # Draw cardinal directions
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(Qt.black, 1))
        
        # Anterior, Posterior, Left, Right labels
        painter.drawText(center_x - 5, center_y - radius - 5, "A")
        painter.drawText(center_x - 5, center_y + radius + 15, "P")
        painter.drawText(center_x - radius - 15, center_y + 5, "L")
        painter.drawText(center_x + radius + 5, center_y + 5, "R")
        
        # Draw beam angles
        for i, beam in enumerate(self.beams):
            # Calculate beam position based on gantry angle
            angle_rad = math.radians(beam.gantry_angle)
            beam_x = center_x + radius * math.sin(angle_rad)
            beam_y = center_y - radius * math.cos(angle_rad)
            
            # Draw beam line
            if i == self.current_beam_index:
                # Highlight current beam
                painter.setPen(QPen(Qt.red, 3))
            else:
                painter.setPen(QPen(Qt.blue, 2))
                
            painter.drawLine(center_x, center_y, beam_x, beam_y)
            
            # Draw beam point/label
            if i == self.current_beam_index:
                painter.setBrush(QBrush(Qt.red))
                painter.drawEllipse(beam_x - 6, beam_y - 6, 12, 12)
                
                # Draw beam name for selected beam
                text_angle = beam.gantry_angle
                if text_angle > 90 and text_angle < 270:
                    text_angle += 180  # Flip text to be readable
                    
                painter.save()
                painter.translate(beam_x, beam_y)
                painter.rotate(-text_angle)
                painter.drawText(15, 5, beam.name)
                painter.restore()
            else:
                painter.setBrush(QBrush(Qt.blue))
                painter.drawEllipse(beam_x - 4, beam_y - 4, 8, 8)
        
        # Draw isocenter marker
        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(QBrush(Qt.yellow))
        painter.drawEllipse(center_x - 5, center_y - 5, 10, 10)
        painter.drawLine(center_x - 7, center_y, center_x + 7, center_y)
        painter.drawLine(center_x, center_y - 7, center_x, center_y + 7)

class BeamTableWidget(QTableWidget):
    """Table widget for displaying and editing beams"""
    
    beam_selected = pyqtSignal(int)  # Signal emitted when a beam is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.beam_set = None
        self.beams = []
        
        # Set up table
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(["Name", "Energy", "Gantry", "Coll", "Couch", "Weight"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)
        self.setEditTriggers(QTableWidget.NoEditTriggers)  # Make table read-only
        
        # Connect signals
        self.cellClicked.connect(self._on_cell_clicked)
        
    def set_beam_set(self, beam_set: BeamSet):
        """Set the beam set to display"""
        self.beam_set = beam_set
        self.beams = beam_set.beams if beam_set else []
        self._update_table()
        
    def _update_table(self):
        """Update the table contents"""
        self.setRowCount(0)  # Clear table
        
        if not self.beams:
            return
            
        for beam in self.beams:
            row = self.rowCount()
            self.insertRow(row)
            
            # Add beam data to the table
            self.setItem(row, 0, QTableWidgetItem(beam.name))
            self.setItem(row, 1, QTableWidgetItem(str(beam.energy)))
            self.setItem(row, 2, QTableWidgetItem(f"{beam.gantry_angle:.1f}°"))
            self.setItem(row, 3, QTableWidgetItem(f"{beam.collimator_angle:.1f}°"))
            self.setItem(row, 4, QTableWidgetItem(f"{beam.couch_angle:.1f}°"))
            self.setItem(row, 5, QTableWidgetItem(f"{beam.weight:.2f}"))
            
        # Auto-adjust row heights
        self.resizeRowsToContents()
        
    def _on_cell_clicked(self, row, column):
        """Handle cell click event"""
        self.beam_selected.emit(row)
        
    def get_selected_beam_index(self) -> int:
        """Get the index of the selected beam"""
        selected_rows = self.selectionModel().selectedRows()
        if selected_rows:
            return selected_rows[0].row()
        return -1

class BeamPlanningTab(QWidget):
    """
    Tab for beam planning and setup.
    
    This tab allows users to create and manage beams for treatment planning.
    """
    
    # Signals
    beam_set_changed = pyqtSignal(BeamSet)  # Emitted when beam set is changed
    dose_calculation_requested = pyqtSignal()  # Emitted when dose calculation is requested
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.beam_set = None
        self.structure_set = None
        self.image = None
        self.isocenter = (0.0, 0.0, 0.0)
        self.current_beam_index = -1
        
        # Set up UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface"""
        main_layout = QVBoxLayout(self)
        
        # Add toolbar
        toolbar = QToolBar()
        toolbar.setIconSize(QPoint(24, 24))
        
        # Add beam button
        self.add_beam_btn = QToolButton()
        self.add_beam_btn.setText("Add Beam")
        self.add_beam_btn.setToolTip("Add a new beam")
        self.add_beam_btn.clicked.connect(self._on_add_beam)
        toolbar.addWidget(self.add_beam_btn)
        
        # Remove beam button
        self.remove_beam_btn = QToolButton()
        self.remove_beam_btn.setText("Remove")
        self.remove_beam_btn.setToolTip("Remove selected beam")
        self.remove_beam_btn.clicked.connect(self._on_remove_beam)
        toolbar.addWidget(self.remove_beam_btn)
        
        toolbar.addSeparator()
        
        # Isocenter button
        self.isocenter_btn = QToolButton()
        self.isocenter_btn.setText("Isocenter")
        self.isocenter_btn.setToolTip("Set isocenter position")
        self.isocenter_btn.clicked.connect(self._on_set_isocenter)
        toolbar.addWidget(self.isocenter_btn)
        
        toolbar.addSeparator()
        
        # Calculate dose button
        self.calculate_btn = QToolButton()
        self.calculate_btn.setText("Calculate")
        self.calculate_btn.setToolTip("Calculate dose for current beam set")
        self.calculate_btn.clicked.connect(self._on_calculate_dose)
        toolbar.addWidget(self.calculate_btn)
        
        main_layout.addWidget(toolbar)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Beam list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Beam table
        beam_group = QGroupBox("Beams")
        beam_layout = QVBoxLayout(beam_group)
        
        self.beam_table = BeamTableWidget()
        self.beam_table.beam_selected.connect(self._on_beam_selected)
        beam_layout.addWidget(self.beam_table)
        
        left_layout.addWidget(beam_group)
        
        # Beam properties
        properties_group = QGroupBox("Beam Properties")
        properties_layout = QVBoxLayout(properties_group)
        
        # Create form layout for properties
        form_layout = QHBoxLayout()
        
        # Left column
        left_form = QVBoxLayout()
        
        # Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_edit = QComboBox()
        self.name_edit.setEditable(True)
        self.name_edit.addItems(["AP", "PA", "LAO", "RAO", "LPO", "RPO", "RL", "LL"])
        self.name_edit.currentTextChanged.connect(self._on_property_changed)
        name_layout.addWidget(self.name_edit)
        left_form.addLayout(name_layout)
        
        # Energy
        energy_layout = QHBoxLayout()
        energy_layout.addWidget(QLabel("Energy:"))
        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["6MV", "10MV", "15MV", "6FFF", "10FFF", "6MeV", "9MeV", "12MeV", "15MeV"])
        self.energy_combo.currentTextChanged.connect(self._on_property_changed)
        energy_layout.addWidget(self.energy_combo)
        left_form.addLayout(energy_layout)
        
        # Gantry angle
        gantry_layout = QHBoxLayout()
        gantry_layout.addWidget(QLabel("Gantry:"))
        self.gantry_spin = QDoubleSpinBox()
        self.gantry_spin.setRange(0, 359.9)
        self.gantry_spin.setDecimals(1)
        self.gantry_spin.setSingleStep(10)
        self.gantry_spin.valueChanged.connect(self._on_property_changed)
        gantry_layout.addWidget(self.gantry_spin)
        left_form.addLayout(gantry_layout)
        
        form_layout.addLayout(left_form)
        
        # Right column
        right_form = QVBoxLayout()
        
        # Collimator angle
        coll_layout = QHBoxLayout()
        coll_layout.addWidget(QLabel("Collimator:"))
        self.coll_spin = QDoubleSpinBox()
        self.coll_spin.setRange(0, 359.9)
        self.coll_spin.setDecimals(1)
        self.coll_spin.setSingleStep(10)
        self.coll_spin.valueChanged.connect(self._on_property_changed)
        coll_layout.addWidget(self.coll_spin)
        right_form.addLayout(coll_layout)
        
        # Couch angle
        couch_layout = QHBoxLayout()
        couch_layout.addWidget(QLabel("Couch:"))
        self.couch_spin = QDoubleSpinBox()
        self.couch_spin.setRange(0, 359.9)
        self.couch_spin.setDecimals(1)
        self.couch_spin.setSingleStep(10)
        self.couch_spin.valueChanged.connect(self._on_property_changed)
        couch_layout.addWidget(self.couch_spin)
        right_form.addLayout(couch_layout)
        
        # Weight
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel("Weight:"))
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 10)
        self.weight_spin.setDecimals(2)
        self.weight_spin.setSingleStep(0.1)
        self.weight_spin.setValue(1.0)
        self.weight_spin.valueChanged.connect(self._on_property_changed)
        weight_layout.addWidget(self.weight_spin)
        right_form.addLayout(weight_layout)
        
        form_layout.addLayout(right_form)
        properties_layout.addLayout(form_layout)
        
        # Field size
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Field Size (cm):"))
        field_layout.addWidget(QLabel("X:"))
        self.field_x_spin = QDoubleSpinBox()
        self.field_x_spin.setRange(0.5, 40)
        self.field_x_spin.setDecimals(1)
        self.field_x_spin.setSingleStep(1)
        self.field_x_spin.setValue(10.0)
        self.field_x_spin.valueChanged.connect(self._on_property_changed)
        field_layout.addWidget(self.field_x_spin)
        
        field_layout.addWidget(QLabel("Y:"))
        self.field_y_spin = QDoubleSpinBox()
        self.field_y_spin.setRange(0.5, 40)
        self.field_y_spin.setDecimals(1)
        self.field_y_spin.setSingleStep(1)
        self.field_y_spin.setValue(10.0)
        self.field_y_spin.valueChanged.connect(self._on_property_changed)
        field_layout.addWidget(self.field_y_spin)
        properties_layout.addLayout(field_layout)
        
        # Modifiers tab widget
        modifiers_tabs = QTabWidget()
        
        # MLC tab
        mlc_tab = QWidget()
        mlc_layout = QVBoxLayout(mlc_tab)
        
        mlc_check = QCheckBox("Use MLC")
        mlc_check.toggled.connect(self._on_mlc_toggled)
        mlc_layout.addWidget(mlc_check)
        
        mlc_layout.addWidget(QLabel("MLC visualization will be shown here"))
        
        modifiers_tabs.addTab(mlc_tab, "MLC")
        
        # Wedge tab
        wedge_tab = QWidget()
        wedge_layout = QVBoxLayout(wedge_tab)
        
        wedge_check = QCheckBox("Use Wedge")
        wedge_check.toggled.connect(self._on_wedge_toggled)
        wedge_layout.addWidget(wedge_check)
        
        wedge_angle_layout = QHBoxLayout()
        wedge_angle_layout.addWidget(QLabel("Angle:"))
        wedge_angle_spin = QComboBox()
        wedge_angle_spin.addItems(["15°", "30°", "45°", "60°"])
        wedge_angle_layout.addWidget(wedge_angle_spin)
        wedge_layout.addLayout(wedge_angle_layout)
        
        wedge_orient_layout = QHBoxLayout()
        wedge_orient_layout.addWidget(QLabel("Orientation:"))
        wedge_orient_combo = QComboBox()
        wedge_orient_combo.addItems(["IN", "OUT", "LEFT", "RIGHT"])
        wedge_orient_layout.addWidget(wedge_orient_combo)
        wedge_layout.addLayout(wedge_orient_layout)
        
        modifiers_tabs.addTab(wedge_tab, "Wedge")
        
        # Block tab
        block_tab = QWidget()
        block_layout = QVBoxLayout(block_tab)
        
        block_check = QCheckBox("Use Block")
        block_layout.addWidget(block_check)
        
        block_layout.addWidget(QLabel("Block editor will be shown here"))
        
        modifiers_tabs.addTab(block_tab, "Block")
        
        # Add the modifiers tab widget
        properties_layout.addWidget(modifiers_tabs)
        
        left_layout.addWidget(properties_group)
        
        # Add prescription group
        prescription_group = QGroupBox("Prescription")
        prescription_layout = QHBoxLayout(prescription_group)
        
        prescription_layout.addWidget(QLabel("Dose:"))
        self.dose_spin = QDoubleSpinBox()
        self.dose_spin.setRange(0, 100)
        self.dose_spin.setDecimals(1)
        self.dose_spin.setSingleStep(1)
        self.dose_spin.setSuffix(" Gy")
        self.dose_spin.valueChanged.connect(self._on_prescription_changed)
        prescription_layout.addWidget(self.dose_spin)
        
        prescription_layout.addWidget(QLabel("Fractions:"))
        self.fractions_spin = QSpinBox()
        self.fractions_spin.setRange(1, 50)
        self.fractions_spin.setSingleStep(1)
        self.fractions_spin.setValue(1)
        prescription_layout.addWidget(self.fractions_spin)
        
        left_layout.addWidget(prescription_group)
        
        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Beam angle visualizer
        angle_group = QGroupBox("Beam Arrangement")
        angle_layout = QVBoxLayout(angle_group)
        
        self.angle_visualizer = BeamAngleVisualizer()
        angle_layout.addWidget(self.angle_visualizer)
        
        right_layout.addWidget(angle_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set splitter sizes
        splitter.setSizes([400, 400])
        
        main_layout.addWidget(splitter)
        
        # Disable property controls initially
        self._enable_property_controls(False)
        
    def set_beam_set(self, beam_set: BeamSet):
        """Set the beam set for the tab"""
        self.beam_set = beam_set
        self.beam_table.set_beam_set(beam_set)
        
        if beam_set and hasattr(beam_set, 'prescription_dose'):
            self.dose_spin.setValue(beam_set.prescription_dose)
            
        # Update beam visualization
        self._update_beam_visualization()
        
        # Clear current beam selection
        self.current_beam_index = -1
        self._show_beam_properties()
        
    def set_structure_set(self, structure_set: StructureSet):
        """Set the structure set for the tab"""
        self.structure_set = structure_set
        
    def set_image(self, image):
        """Set the planning image for the tab"""
        self.image = image
        
    def _enable_property_controls(self, enable: bool):
        """Enable or disable property controls"""
        self.name_edit.setEnabled(enable)
        self.energy_combo.setEnabled(enable)
        self.gantry_spin.setEnabled(enable)
        self.coll_spin.setEnabled(enable)
        self.couch_spin.setEnabled(enable)
        self.weight_spin.setEnabled(enable)
        self.field_x_spin.setEnabled(enable)
        self.field_y_spin.setEnabled(enable)
        self.remove_beam_btn.setEnabled(enable and self.beam_set and len(self.beam_set.beams) > 0)
        
    def _update_beam_visualization(self):
        """Update the beam visualization"""
        if self.beam_set:
            self.angle_visualizer.set_beams(self.beam_set.beams)
            self.angle_visualizer.set_current_beam(self.current_beam_index)
            self.angle_visualizer.set_isocenter(self.isocenter)
            
    def _show_beam_properties(self):
        """Update the UI to show properties of the selected beam"""
        self._enable_property_controls(self.current_beam_index >= 0)
        
        if self.current_beam_index >= 0 and self.beam_set and len(self.beam_set.beams) > self.current_beam_index:
            beam = self.beam_set.beams[self.current_beam_index]
            
            # Block signals to avoid triggering change events
            self.name_edit.blockSignals(True)
            self.energy_combo.blockSignals(True)
            self.gantry_spin.blockSignals(True)
            self.coll_spin.blockSignals(True)
            self.couch_spin.blockSignals(True)
            self.weight_spin.blockSignals(True)
            self.field_x_spin.blockSignals(True)
            self.field_y_spin.blockSignals(True)
            
            # Set values
            self.name_edit.setCurrentText(beam.name)
            
            # Find energy in combo or add it if not present
            energy_index = self.energy_combo.findText(beam.energy)
            if energy_index >= 0:
                self.energy_combo.setCurrentIndex(energy_index)
            else:
                self.energy_combo.addItem(beam.energy)
                self.energy_combo.setCurrentText(beam.energy)
                
            self.gantry_spin.setValue(beam.gantry_angle)
            self.coll_spin.setValue(beam.collimator_angle)
            self.couch_spin.setValue(beam.couch_angle)
            self.weight_spin.setValue(beam.weight)
            
            if hasattr(beam, 'field_size') and isinstance(beam.field_size, (list, tuple)) and len(beam.field_size) >= 2:
                self.field_x_spin.setValue(beam.field_size[0])
                self.field_y_spin.setValue(beam.field_size[1])
            
            # Unblock signals
            self.name_edit.blockSignals(False)
            self.energy_combo.blockSignals(False)
            self.gantry_spin.blockSignals(False)
            self.coll_spin.blockSignals(False)
            self.couch_spin.blockSignals(False)
            self.weight_spin.blockSignals(False)
            self.field_x_spin.blockSignals(False)
            self.field_y_spin.blockSignals(False)
            
    def _on_beam_selected(self, index: int):
        """Handle beam selection"""
        if index >= 0 and self.beam_set and len(self.beam_set.beams) > index:
            self.current_beam_index = index
            self._show_beam_properties()
            self.angle_visualizer.set_current_beam(index)
        
    def _on_add_beam(self):
        """Add a new beam"""
        if not self.beam_set:
            QMessageBox.warning(self, "Error", "No beam set available")
            return
            
        # Create a new beam with default parameters
        new_beam = Beam(name=f"Beam {len(self.beam_set.beams) + 1}")
        new_beam.energy = "6MV"
        new_beam.gantry_angle = 0.0
        new_beam.collimator_angle = 0.0
        new_beam.couch_angle = 0.0
        new_beam.field_size = (10.0, 10.0)
        new_beam.isocenter = self.isocenter
        new_beam.weight = 1.0
        
        # Add beam to the beam set
        self.beam_set.beams.append(new_beam)
        
        # Update the beam table
        self.beam_table.set_beam_set(self.beam_set)
        
        # Select the new beam
        self.current_beam_index = len(self.beam_set.beams) - 1
        self.beam_table.selectRow(self.current_beam_index)
        self._show_beam_properties()
        
        # Update the visualization
        self._update_beam_visualization()
        
        # Emit signal that beam set has changed
        self.beam_set_changed.emit(self.beam_set)
        
    def _on_remove_beam(self):
        """Remove the selected beam"""
        if not self.beam_set or self.current_beam_index < 0:
            return
            
        # Remove the beam
        self.beam_set.beams.pop(self.current_beam_index)
        
        # Update the beam table
        self.beam_table.set_beam_set(self.beam_set)
        
        # Update selection
        if len(self.beam_set.beams) > 0:
            new_index = min(self.current_beam_index, len(self.beam_set.beams) - 1)
            self.current_beam_index = new_index
            self.beam_table.selectRow(new_index)
        else:
            self.current_beam_index = -1
            
        self._show_beam_properties()
        
        # Update the visualization
        self._update_beam_visualization()
        
        # Emit signal that beam set has changed
        self.beam_set_changed.emit(self.beam_set)
        
    def _on_set_isocenter(self):
        """Set the isocenter position (this would typically come from the MPR view)"""
        # In a real implementation, this would use the current position from MPR views
        # For now, just show a message
        QMessageBox.information(self, "Set Isocenter", 
                               "In a full implementation, this would capture the current isocenter from MPR views.")
        
    def _on_property_changed(self):
        """Handle property changes"""
        if self.current_beam_index < 0 or not self.beam_set or len(self.beam_set.beams) <= self.current_beam_index:
            return
            
        beam = self.beam_set.beams[self.current_beam_index]
        
        # Update beam properties
        beam.name = self.name_edit.currentText()
        beam.energy = self.energy_combo.currentText()
        beam.gantry_angle = self.gantry_spin.value()
        beam.collimator_angle = self.coll_spin.value()
        beam.couch_angle = self.couch_spin.value()
        beam.weight = self.weight_spin.value()
        beam.field_size = (self.field_x_spin.value(), self.field_y_spin.value())
        
        # Update the beam table and visualization
        self.beam_table.set_beam_set(self.beam_set)
        self._update_beam_visualization()
        
        # Emit signal that beam set has changed
        self.beam_set_changed.emit(self.beam_set)
        
    def _on_prescription_changed(self):
        """Handle prescription changes"""
        if self.beam_set and hasattr(self.beam_set, 'prescription_dose'):
            self.beam_set.prescription_dose = self.dose_spin.value()
            
            # Emit signal that beam set has changed
            self.beam_set_changed.emit(self.beam_set)
        
    def _on_calculate_dose(self):
        """Handle dose calculation request"""
        if not self.beam_set or len(self.beam_set.beams) == 0:
            QMessageBox.warning(self, "Error", "No beams available for dose calculation")
            return
            
        # Emit signal to request dose calculation
        self.dose_calculation_requested.emit()
        
    def _on_mlc_toggled(self, checked: bool):
        """Handle MLC toggle"""
        if self.current_beam_index < 0 or not self.beam_set or len(self.beam_set.beams) <= self.current_beam_index:
            return
            
        beam = self.beam_set.beams[self.current_beam_index]
        
        if checked:
            # Create MLC if it doesn't exist
            if not beam.mlc:
                beam.mlc = MLC(name=f"MLC_{beam.name}")
                # Initialize with a rectangular field matching the collimator
                if hasattr(beam.mlc, 'create_rectangular_field'):
                    try:
                        beam.mlc.create_rectangular_field(
                            width=beam.field_size[0], 
                            height=beam.field_size[1]
                        )
                    except Exception as e:
                        logger.error(f"Error creating MLC field: {str(e)}")
        else:
            # Remove MLC
            beam.mlc = None
            
        # Emit signal that beam set has changed
        self.beam_set_changed.emit(self.beam_set)
        
    def _on_wedge_toggled(self, checked: bool):
        """Handle wedge toggle"""
        if self.current_beam_index < 0 or not self.beam_set or len(self.beam_set.beams) <= self.current_beam_index:
            return
            
        beam = self.beam_set.beams[self.current_beam_index]
        
        # Find existing wedge if any
        existing_wedge = None
        for modifier in beam.modifiers:
            if isinstance(modifier, Wedge):
                existing_wedge = modifier
                break
                
        if checked:
            # Add wedge if it doesn't exist
            if not existing_wedge:
                wedge = Wedge(name=f"Wedge_{beam.name}")
                beam.modifiers.append(wedge)
        else:
            # Remove wedge if it exists
            if existing_wedge:
                beam.modifiers.remove(existing_wedge)
                
        # Emit signal that beam set has changed
        self.beam_set_changed.emit(self.beam_set) 