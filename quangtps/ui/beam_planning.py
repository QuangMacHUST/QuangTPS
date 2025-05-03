#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp giao diện người dùng cho lập kế hoạch chùm tia.

Module này định nghĩa các widget và giao diện người dùng để thiết lập,
chỉnh sửa và hiển thị các tham số và cấu hình chùm tia.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QRadioButton,
    QFormLayout,
    QDoubleSpinBox,
    QSlider,
    QTabWidget,
    QToolBar,
    QAction,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSplitter,
    QFrame,
    QSizePolicy,
    QMenu,
    QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QColor, QBrush

from quangtps.planning.beam import (
    BeamParameters,
    BeamEnergyType,
    TechniqueType,
    MLCType,
    SourceType,
)
from quangtps.planning.plan import Plan
from quangtps.treatment.beams.beam import Beam
from quangtps.ui.mlc_editor import MLCEditor
from quangtps.ui.beam_eye_view import BeamEyeView
from quangtps.treatment.mlc.mlc_editor_controller import MLCEditorController
from quangtps.optimization.mlc_optimization import optimize_mlc_shape
from quangtps.planning.mlc import MLC, MLCLeaf, MLCSequence, MLC_CONFIGURATIONS
from quangtps.common.paths import get_icon_path

logger = logging.getLogger(__name__)


class BeamParameters:
    """Lớp đại diện cho tham số chùm tia."""

    def __init__(self):
        """Khởi tạo tham số chùm tia với giá trị mặc định."""
        self.name = "Beam 1"
        self.technique = TechniqueType.STATIC
        self.energy = BeamEnergyType.X6MV
        self.source = SourceType.PHOTON
        self.gantry_angle = 0.0
        self.collimator_angle = 0.0
        self.couch_angle = 0.0
        self.sad = 100.0
        self.field_x = 10.0
        self.field_y = 10.0
        self.weight = 1.0
        self.mlc_type = MLCType.MILLENNIUM120
        self.mlc = None
        self.isocenter = (0.0, 0.0, 0.0)


class BeamSetupWidget(QWidget):
    """Widget for beam setup and configuration."""

    beam_selected = pyqtSignal(int)  # Emits index of selected beam
    beam_added = pyqtSignal()
    beam_removed = pyqtSignal(int)  # Emits index of removed beam
    beam_changed = pyqtSignal(int)  # Emits index of modified beam

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data
        self.beams = []
        self.selected_beam_index = -1

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        main_layout = QVBoxLayout(self)

        # Beam table
        self.beam_table = QTableWidget()
        self.beam_table.setColumnCount(5)
        self.beam_table.setHorizontalHeaderLabels(
            ["Name", "Energy", "Gantry", "Coll", "Couch"]
        )
        self.beam_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.beam_table.setSelectionMode(QTableWidget.SingleSelection)
        self.beam_table.selectionModel().selectionChanged.connect(
            self._on_beam_selected
        )
        header = self.beam_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        main_layout.addWidget(self.beam_table)

        # Beam control buttons
        button_layout = QHBoxLayout()
        self.add_beam_btn = QPushButton("Add Beam")
        self.add_beam_btn.clicked.connect(self._on_add_beam)
        self.remove_beam_btn = QPushButton("Remove Beam")
        self.remove_beam_btn.clicked.connect(self._on_remove_beam)
        button_layout.addWidget(self.add_beam_btn)
        button_layout.addWidget(self.remove_beam_btn)
        main_layout.addLayout(button_layout)

        # Beam properties
        properties_group = QGroupBox("Beam Properties")
        properties_layout = QFormLayout(properties_group)

        self.name_edit = QComboBox()
        self.name_edit.setEditable(True)
        self.name_edit.addItems(
            [
                "AP",
                "PA",
                "LAT",
                "RPO",
                "LPO",
                "RAO",
                "LAO",
                "RLAT",
                "LLAT",
                "Vertex",
            ]
        )
        self.name_edit.currentTextChanged.connect(self._on_property_changed)

        self.energy_combo = QComboBox()
        self.energy_combo.addItems(
            [e.name for e in BeamEnergyType if e != BeamEnergyType.CUSTOM]
        )
        self.energy_combo.currentIndexChanged.connect(self._on_property_changed)

        self.technique_combo = QComboBox()
        self.technique_combo.addItems([t.name for t in TechniqueType])
        self.technique_combo.currentIndexChanged.connect(self._on_property_changed)

        self.gantry_spin = QDoubleSpinBox()
        self.gantry_spin.setRange(0, 360)
        self.gantry_spin.setSingleStep(10)
        self.gantry_spin.setSuffix("°")
        self.gantry_spin.valueChanged.connect(self._on_property_changed)

        self.coll_spin = QDoubleSpinBox()
        self.coll_spin.setRange(0, 360)
        self.coll_spin.setSingleStep(10)
        self.coll_spin.setSuffix("°")
        self.coll_spin.valueChanged.connect(self._on_property_changed)

        self.couch_spin = QDoubleSpinBox()
        self.couch_spin.setRange(0, 360)
        self.couch_spin.setSingleStep(10)
        self.couch_spin.setSuffix("°")
        self.couch_spin.valueChanged.connect(self._on_property_changed)

        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 100)
        self.weight_spin.setSingleStep(1)
        self.weight_spin.setSuffix("%")
        self.weight_spin.setValue(100)
        self.weight_spin.valueChanged.connect(self._on_property_changed)

        self.field_x_spin = QDoubleSpinBox()
        self.field_x_spin.setRange(0, 40)
        self.field_x_spin.setSingleStep(1)
        self.field_x_spin.setSuffix(" cm")
        self.field_x_spin.setValue(10)
        self.field_x_spin.valueChanged.connect(self._on_property_changed)

        self.field_y_spin = QDoubleSpinBox()
        self.field_y_spin.setRange(0, 40)
        self.field_y_spin.setSingleStep(1)
        self.field_y_spin.setSuffix(" cm")
        self.field_y_spin.setValue(10)
        self.field_y_spin.valueChanged.connect(self._on_property_changed)

        properties_layout.addRow("Name:", self.name_edit)
        properties_layout.addRow("Energy:", self.energy_combo)
        properties_layout.addRow("Technique:", self.technique_combo)
        properties_layout.addRow("Gantry angle:", self.gantry_spin)
        properties_layout.addRow("Collimator angle:", self.coll_spin)
        properties_layout.addRow("Couch angle:", self.couch_spin)
        properties_layout.addRow("Weight:", self.weight_spin)
        properties_layout.addRow("Field X:", self.field_x_spin)
        properties_layout.addRow("Field Y:", self.field_y_spin)

        main_layout.addWidget(properties_group)

        # Add some preset templates
        templates_group = QGroupBox("Beam Templates")
        templates_layout = QHBoxLayout(templates_group)

        self.ap_pa_btn = QPushButton("AP/PA")
        self.ap_pa_btn.clicked.connect(self._create_ap_pa)
        templates_layout.addWidget(self.ap_pa_btn)

        self.four_field_btn = QPushButton("4-Field Box")
        self.four_field_btn.clicked.connect(self._create_four_field)
        templates_layout.addWidget(self.four_field_btn)

        self.arc_btn = QPushButton("Arc")
        self.arc_btn.clicked.connect(self._create_arc)
        templates_layout.addWidget(self.arc_btn)

        main_layout.addWidget(templates_group)

        # Disable property controls initially
        self._enable_property_controls(False)

    def update_beam_table(self):
        """Update the beam table with current beams."""
        self.beam_table.setRowCount(0)

        for i, beam in enumerate(self.beams):
            row = self.beam_table.rowCount()
            self.beam_table.insertRow(row)

            name_item = QTableWidgetItem(beam.name)
            energy_item = QTableWidgetItem(str(beam.energy))
            gantry_item = QTableWidgetItem(f"{beam.gantry_angle}°")
            coll_item = QTableWidgetItem(f"{beam.collimator_angle}°")
            couch_item = QTableWidgetItem(f"{beam.couch_angle}°")

            self.beam_table.setItem(row, 0, name_item)
            self.beam_table.setItem(row, 1, energy_item)
            self.beam_table.setItem(row, 2, gantry_item)
            self.beam_table.setItem(row, 3, coll_item)
            self.beam_table.setItem(row, 4, couch_item)

        # Select previously selected beam if still exists
        if self.selected_beam_index >= 0 and self.selected_beam_index < len(self.beams):
            self.beam_table.selectRow(self.selected_beam_index)
        elif len(self.beams) > 0:
            self.beam_table.selectRow(0)
            self.selected_beam_index = 0
        else:
            self.selected_beam_index = -1
            self._enable_property_controls(False)

    def _on_beam_selected(self, selected, deselected):
        """Handle beam selection in the table."""
        indexes = selected.indexes()
        if indexes:
            self.selected_beam_index = indexes[0].row()
            self._update_property_controls()
            self._enable_property_controls(True)
            self.beam_selected.emit(self.selected_beam_index)
        else:
            self.selected_beam_index = -1
            self._enable_property_controls(False)

    def _update_property_controls(self):
        """Update property controls with selected beam data."""
        if self.selected_beam_index < 0 or self.selected_beam_index >= len(self.beams):
            return

        beam = self.beams[self.selected_beam_index]

        # Block signals to prevent triggering _on_property_changed
        self.name_edit.blockSignals(True)
        self.energy_combo.blockSignals(True)
        self.technique_combo.blockSignals(True)
        self.gantry_spin.blockSignals(True)
        self.coll_spin.blockSignals(True)
        self.couch_spin.blockSignals(True)
        self.weight_spin.blockSignals(True)
        self.field_x_spin.blockSignals(True)
        self.field_y_spin.blockSignals(True)

        # Update controls
        current_index = self.name_edit.findText(beam.name)
        if current_index >= 0:
            self.name_edit.setCurrentIndex(current_index)
        else:
            self.name_edit.setCurrentText(beam.name)

        energy_index = self.energy_combo.findText(beam.energy.name)
        if energy_index >= 0:
            self.energy_combo.setCurrentIndex(energy_index)

        technique_index = self.technique_combo.findText(beam.technique.name)
        if technique_index >= 0:
            self.technique_combo.setCurrentIndex(technique_index)

        self.gantry_spin.setValue(beam.gantry_angle)
        self.coll_spin.setValue(beam.collimator_angle)
        self.couch_spin.setValue(beam.couch_angle)
        self.weight_spin.setValue(beam.weight)
        self.field_x_spin.setValue(beam.field_x)
        self.field_y_spin.setValue(beam.field_y)

        # Re-enable signals
        self.name_edit.blockSignals(False)
        self.energy_combo.blockSignals(False)
        self.technique_combo.blockSignals(False)
        self.gantry_spin.blockSignals(False)
        self.coll_spin.blockSignals(False)
        self.couch_spin.blockSignals(False)
        self.weight_spin.blockSignals(False)
        self.field_x_spin.blockSignals(False)
        self.field_y_spin.blockSignals(False)

    def _enable_property_controls(self, enable):
        """Enable or disable property controls."""
        self.name_edit.setEnabled(enable)
        self.energy_combo.setEnabled(enable)
        self.technique_combo.setEnabled(enable)
        self.gantry_spin.setEnabled(enable)
        self.coll_spin.setEnabled(enable)
        self.couch_spin.setEnabled(enable)
        self.weight_spin.setEnabled(enable)
        self.field_x_spin.setEnabled(enable)
        self.field_y_spin.setEnabled(enable)
        self.remove_beam_btn.setEnabled(enable and len(self.beams) > 0)

    def _on_add_beam(self):
        """Add a new beam."""
        # Create new beam
        new_beam = BeamParameters()
        new_beam.name = f"Beam {len(self.beams) + 1}"

        # Add it to the list
        self.beams.append(new_beam)

        # Update table
        self.update_beam_table()

        # Select new beam
        self.beam_table.selectRow(len(self.beams) - 1)
        self.selected_beam_index = len(self.beams) - 1

        # Enable controls
        self._enable_property_controls(True)

        # Emit signal
        self.beam_added.emit()

    def _on_remove_beam(self):
        """Remove the selected beam."""
        if self.selected_beam_index < 0 or self.selected_beam_index >= len(self.beams):
            return

        # Remove beam
        del self.beams[self.selected_beam_index]

        # Update table
        self.update_beam_table()

        # Emit signal
        self.beam_removed.emit(self.selected_beam_index)

    def _on_property_changed(self):
        """Handle property changes."""
        if self.selected_beam_index < 0 or self.selected_beam_index >= len(self.beams):
            return

        beam = self.beams[self.selected_beam_index]

        # Update beam properties
        beam.name = self.name_edit.currentText()
        beam.energy = BeamEnergyType[self.energy_combo.currentText()]
        beam.technique = TechniqueType[self.technique_combo.currentText()]
        beam.gantry_angle = self.gantry_spin.value()
        beam.collimator_angle = self.coll_spin.value()
        beam.couch_angle = self.couch_spin.value()
        beam.weight = self.weight_spin.value()
        beam.field_x = self.field_x_spin.value()
        beam.field_y = self.field_y_spin.value()

        # Update table
        self.update_beam_table()

        # Emit signal
        self.beam_changed.emit(self.selected_beam_index)

    def _create_ap_pa(self):
        """Create an AP/PA beam arrangement."""
        # Clear existing beams
        self.beams.clear()

        # AP beam
        ap_beam = BeamParameters()
        ap_beam.name = "AP"
        ap_beam.gantry_angle = 0
        ap_beam.weight = 50
        self.beams.append(ap_beam)

        # PA beam
        pa_beam = BeamParameters()
        pa_beam.name = "PA"
        pa_beam.gantry_angle = 180
        pa_beam.weight = 50
        self.beams.append(pa_beam)

        # Update table
        self.update_beam_table()

        # Select first beam
        self.beam_table.selectRow(0)
        self.selected_beam_index = 0
        self._enable_property_controls(True)

    def _create_four_field(self):
        """Create a 4-field box arrangement."""
        # Clear existing beams
        self.beams.clear()

        # AP beam
        ap_beam = BeamParameters()
        ap_beam.name = "AP"
        ap_beam.gantry_angle = 0
        ap_beam.weight = 25
        self.beams.append(ap_beam)

        # PA beam
        pa_beam = BeamParameters()
        pa_beam.name = "PA"
        pa_beam.gantry_angle = 180
        pa_beam.weight = 25
        self.beams.append(pa_beam)

        # Right lateral
        rlat_beam = BeamParameters()
        rlat_beam.name = "RLAT"
        rlat_beam.gantry_angle = 270
        rlat_beam.weight = 25
        self.beams.append(rlat_beam)

        # Left lateral
        llat_beam = BeamParameters()
        llat_beam.name = "LLAT"
        llat_beam.gantry_angle = 90
        llat_beam.weight = 25
        self.beams.append(llat_beam)

        # Update table
        self.update_beam_table()

        # Select first beam
        self.beam_table.selectRow(0)
        self.selected_beam_index = 0
        self._enable_property_controls(True)

    def _create_arc(self):
        """Create an arc beam arrangement."""
        # Clear existing beams
        self.beams.clear()

        # Arc beam
        arc_beam = BeamParameters()
        arc_beam.name = "ARC"
        arc_beam.gantry_angle = 0  # Start angle
        arc_beam.technique = TechniqueType.ARC
        arc_beam.weight = 100
        self.beams.append(arc_beam)

        # Update table
        self.update_beam_table()

        # Select beam
        self.beam_table.selectRow(0)
        self.selected_beam_index = 0
        self._enable_property_controls(True)


class MLCEditorWidget(QWidget):
    """Widget for editing multi-leaf collimator (MLC) shapes."""

    mlc_changed = pyqtSignal(object)  # Emits MLC data when changed

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data
        self.mlc = None
        self.beam = None
        self.target_structure = None
        self.oar_structures = []
        self.controller = MLCEditorController()

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        main_layout = QVBoxLayout(self)

        # Create toolbar
        toolbar = QToolBar("MLC Editor Tools")
        toolbar.setIconSize(QSize(24, 24))

        # Add toolbar actions
        fit_action = QAction(
            QIcon(get_icon_path("fit_mlc.png")), "Fit MLC to Structure", self
        )
        fit_action.triggered.connect(self.fit_mlc_to_structure)
        toolbar.addAction(fit_action)

        optimize_action = QAction(
            QIcon(get_icon_path("optimize.png")), "Optimize MLC", self
        )
        optimize_action.triggered.connect(self.optimize_mlc)
        toolbar.addAction(optimize_action)

        clear_action = QAction(QIcon(get_icon_path("clear.png")), "Clear MLC", self)
        clear_action.triggered.connect(self.clear_mlc)
        toolbar.addAction(clear_action)

        main_layout.addWidget(toolbar)

        # Create tabs for different views
        tabs = QTabWidget()

        # MLC Editor tab
        self.mlc_editor = MLCEditor(self)
        self.mlc_editor.mlc_changed.connect(self._on_mlc_changed)
        tabs.addTab(self.mlc_editor, "MLC Editor")

        # BEV tab
        self.bev_widget = BeamEyeView(self)
        self.bev_widget.mlcChanged.connect(self._on_bev_mlc_changed)
        tabs.addTab(self.bev_widget, "Beam's Eye View")

        main_layout.addWidget(tabs)

        # Setup optimization options at the bottom
        opt_group = QGroupBox("Optimization Settings")
        opt_layout = QFormLayout(opt_group)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(
            ["Gradient Descent", "Simulated Annealing", "Genetic Algorithm"]
        )
        opt_layout.addRow("Algorithm:", self.algorithm_combo)

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(10, 1000)
        self.iterations_spin.setValue(100)
        self.iterations_spin.setSingleStep(10)
        opt_layout.addRow("Iterations:", self.iterations_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0001, 0.1)
        self.threshold_spin.setValue(0.001)
        self.threshold_spin.setSingleStep(0.001)
        self.threshold_spin.setDecimals(4)
        opt_layout.addRow("Convergence threshold:", self.threshold_spin)

        main_layout.addWidget(opt_group)

    def set_beam(self, beam):
        """Set the beam for MLC editing."""
        self.beam = beam

        # Set beam to BEV
        self.bev_widget.set_beam(beam)

        # Create MLC if needed
        if not beam.mlc:
            beam.mlc = MLC(MLCType.MILLENNIUM120)

        # Set MLC to editor and controller
        self.mlc = beam.mlc
        self.mlc_editor.set_mlc(self.mlc)
        self.controller.set_mlc(self.mlc)

    def set_structures(self, target=None, oars=None):
        """Set target and OAR structures for MLC optimization."""
        self.target_structure = target
        self.oar_structures = oars or []

        # Set structures in BEV
        structures = []
        if target:
            structures.append(target)
        if oars:
            structures.extend(oars)

        self.bev_widget.set_structures(structures)

    def fit_mlc_to_structure(self):
        """Fit MLC to the target structure."""
        if not self.target_structure or not self.mlc:
            QMessageBox.warning(self, "Error", "Target structure or MLC not available")
            return

        try:
            # Create field with target structure
            from quangtps.planning.mlc import create_shape_based_mlc

            field_size = max(self.beam.field_x, self.beam.field_y)
            new_mlc = create_shape_based_mlc(
                self.target_structure,
                self.mlc.mlc_type,
                field_size,
                margin=0.5,  # Add 5mm margin
                beam=self.beam,
            )

            # Set new MLC
            self.mlc = new_mlc
            self.mlc_editor.set_mlc(self.mlc)
            self.controller.set_mlc(self.mlc)

            # Update beam
            if self.beam:
                self.beam.mlc = self.mlc

            # Emit signal
            self.mlc_changed.emit(self.mlc)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not fit MLC to structure: {str(e)}"
            )

    def optimize_mlc(self):
        """Optimize MLC shape based on target and OAR structures."""
        if not self.target_structure or not self.mlc:
            QMessageBox.warning(self, "Error", "Target structure or MLC not available")
            return

        try:
            # Get algorithm and parameters
            algorithm_map = {0: "gradient", 1: "simulated_annealing", 2: "genetic"}
            algorithm = algorithm_map[self.algorithm_combo.currentIndex()]
            iterations = self.iterations_spin.value()
            threshold = self.threshold_spin.value()

            # Optimize MLC
            optimized_mlc = optimize_mlc_shape(
                original_mlc=self.mlc,
                target=self.target_structure,
                oars=self.oar_structures,
                field_size=max(self.beam.field_x, self.beam.field_y),
                beam=self.beam,
                algorithm=algorithm,
                iterations=iterations,
                convergence_threshold=threshold,
            )

            # Set optimized MLC
            self.mlc = optimized_mlc
            self.mlc_editor.set_mlc(self.mlc)
            self.controller.set_mlc(self.mlc)

            # Update beam
            if self.beam:
                self.beam.mlc = self.mlc

            # Emit signal
            self.mlc_changed.emit(self.mlc)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"MLC optimization failed: {str(e)}")

    def clear_mlc(self):
        """Clear MLC shape (open all leaves)."""
        if not self.mlc:
            return

        try:
            for leaf in self.mlc.leaves:
                if leaf.bank == "A":
                    self.mlc.set_leaf_position(leaf.index, -20.0)
                else:
                    self.mlc.set_leaf_position(leaf.index, 20.0)

            # Update editor
            self.mlc_editor.set_mlc(self.mlc)

            # Update beam
            if self.beam:
                self.beam.mlc = self.mlc

            # Emit signal
            self.mlc_changed.emit(self.mlc)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not clear MLC: {str(e)}")

    def _on_mlc_changed(self, mlc):
        """Handle MLC changes from the editor."""
        self.mlc = mlc

        # Update beam
        if self.beam:
            self.beam.mlc = mlc

        # Update BEV
        self.bev_widget.update_mlc(mlc)

        # Emit signal
        self.mlc_changed.emit(mlc)

    def _on_bev_mlc_changed(self, mlc):
        """Handle MLC changes from BEV view."""
        self.mlc = mlc

        # Update editor
        self.mlc_editor.set_mlc(mlc)

        # Update beam
        if self.beam:
            self.beam.mlc = mlc

        # Emit signal
        self.mlc_changed.emit(mlc)


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
        self.bev_viewer = BeamEyeView()
        bev_layout.addWidget(self.bev_viewer)
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

        # Connect signals
        self.beam_setup.beam_selected.connect(self._on_beam_selected)
        self.beam_setup.beam_changed.connect(self._on_beam_changed)
        self.mlc_editor.mlc_changed.connect(self._on_mlc_changed)

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

    def set_structures(self, structures):
        """Set structures for BEV and MLC optimization."""
        # Set structures to BEV viewer
        self.bev_viewer.set_structures(structures)

        # Find target and OAR structures
        target = None
        oars = []

        for structure in structures:
            # Simple heuristic: structures with "PTV" or "CTV" are targets
            if "PTV" in structure.name.upper() or "CTV" in structure.name.upper():
                target = structure
            # Others are OARs
            else:
                oars.append(structure)

        # Set target and OARs to MLC editor
        self.mlc_editor.set_structures(target, oars)

    def _on_beam_selected(self, beam_index):
        """Handle beam selection."""
        if beam_index < 0 or beam_index >= len(self.beam_setup.beams):
            return

        beam = self.beam_setup.beams[beam_index]

        # Update BEV
        self.bev_viewer.set_beam(beam)

        # Update MLC editor
        self.mlc_editor.set_beam(beam)

    def _on_beam_changed(self, beam_index):
        """Handle beam parameter changes."""
        if beam_index < 0 or beam_index >= len(self.beam_setup.beams):
            return

        beam = self.beam_setup.beams[beam_index]

        # Update BEV with new beam parameters
        self.bev_viewer.update_beam_parameters(beam)

    def _on_mlc_changed(self, mlc):
        """Handle MLC changes."""
        # If we have a selected beam, update its MLC
        beam_index = self.beam_setup.selected_beam_index
        if beam_index >= 0 and beam_index < len(self.beam_setup.beams):
            beam = self.beam_setup.beams[beam_index]
            beam.mlc = mlc


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
    sys.exit(test())
