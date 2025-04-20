#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab External Beam Planning cho QuangTPS.

Module này triển khai giao diện Eclipse-like External Beam Planning,
tích hợp các tính năng lập kế hoạch và tính liều vào một tab duy nhất.
Đây là sự kết hợp các tính năng của planning_tab.py và dose_tab.py,
với cải tiến giao diện mô phỏng theo phần mềm Eclipse TPS của Varian.
"""

import os
import sys
import logging
import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Union

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
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QDateEdit,
    QInputDialog,
    QSizePolicy,
)
from PyQt5.QtGui import QColor, QIcon, QBrush, QPixmap, QImage, QPainter, QPen, QCursor
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QDate

# Import matplotlib for visualization if available
try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib not available for DVH visualization")
    MATPLOTLIB_AVAILABLE = False

# Import QuangTPS modules
try:
    # Import core modules
    from quangtps.core.services import ServiceRegistry
    from quangtps.core.patient import Patient
    from quangtps.database.patient_db import PatientDB
    from quangtps.database.plan_db import PlanDB
    from quangtps.database.structure_db import StructureDB
    from quangtps.database.beam_db import BeamDB
    from quangtps.database.dose_db import DoseDB
    from quangtps.planning.plan import Plan, PlanStatus, PlanType
    from quangtps.planning.beam import Beam
    from quangtps.planning.prescription import Prescription

    # Import planning modules
    from quangtps.treatment.techniques.crt_manager import CRTManager
    from quangtps.treatment.techniques.imrt import IMRT
    from quangtps.treatment.techniques.vmat import VMAT
    from quangtps.treatment.techniques.treatment_technique import TreatmentTechnique

    # Import dose calculation modules
    from quangtps.dose.dose_calculator import DoseCalculator
    from quangtps.dose.dose_grid import DoseGrid

    # Import optimization modules
    from quangtps.optimization.optimization_engine import (
        OptimizationEngine,
        OptimizationParameters,
    )
    from quangtps.optimization.objectives import ObjectiveCollection
    from quangtps.optimization.constraints import ConstraintCollection

    # Import evaluation modules
    from quangtps.evaluation.plan_evaluation import PlanEvaluation
    from quangtps.evaluation.dvh.dvh_calculation import (
        calculate_dvh,
        calculate_dvh_metrics,
    )
    from quangtps.evaluation.dvh.dvh_visualization import plot_dvh

    # Import UI modules
    from quangtps.ui.dialogs.beam_dialog import BeamDialog
    from quangtps.ui.beam_visualization_panel import BeamVisualizationPanel
    from quangtps.ui.dose_visualization_3d import DoseVisualization3D

    # Import MCO-related modules
    from quangtps.optimization.mco.mco_engine import MCOEngine

    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logging.error(f"Error importing QuangTPS modules: {e}")

logger = logging.getLogger(__name__)


class ExternalBeamPlanningTab(QWidget):
    """
    Tab External Beam Planning cho QuangTPS với giao diện kiểu Eclipse.

    Tab này tích hợp các tính năng lập kế hoạch và tính liều trong một giao diện
    thống nhất, tương tự như Eclipse TPS của Varian. Bao gồm các tính năng:
    - Quản lý kế hoạch và chùm tia
    - Thiết lập kỹ thuật điều trị (3D CRT, IMRT, VMAT)
    - Tối ưu hóa kế hoạch (Multi-Criteria Optimization)
    - Tính toán liều
    - Phân tích và đánh giá kế hoạch (DVH, metrics, dose visualization)
    """

    # Tín hiệu
    plan_created = pyqtSignal(object)
    plan_updated = pyqtSignal(object)
    plan_deleted = pyqtSignal(str)
    patient_loaded = pyqtSignal(object)
    calculation_started = pyqtSignal()
    calculation_finished = pyqtSignal()

    def __init__(self, parent=None):
        """
        Khởi tạo tab External Beam Planning.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)

        # Khởi tạo trạng thái
        self.current_patient = None
        self.current_plan = None
        self.current_beam = None
        self.current_image = None
        self.current_structure_set = None
        self.current_dose_grid = None

        # Initialize services
        self.service_registry = ServiceRegistry()
        self.plan_db = self.service_registry.get_service("PlanDB")
        self.patient_db = self.service_registry.get_service("PatientDB")
        self.structure_db = self.service_registry.get_service("StructureDB")
        self.beam_db = self.service_registry.get_service("BeamDB")
        self.dose_db = self.service_registry.get_service("DoseDB")

        self.dose_calculator = self.service_registry.get_service("DoseCalculator")
        self.optimization_engine = self.service_registry.get_service(
            "OptimizationEngine"
        )

        # Khởi tạo các managers
        self.crt_manager = CRTManager() if MODULES_AVAILABLE else None

        # Thiết lập giao diện
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện tab External Beam Planning."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Toolbar chính
        self.main_toolbar = QToolBar("Planning Tools")

        # Patient & Plan selection controls
        patient_label = QLabel("Patient:")
        self.main_toolbar.addWidget(patient_label)

        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(200)
        self.main_toolbar.addWidget(self.patient_combo)
        self.main_toolbar.addSeparator()

        plan_label = QLabel("Plan:")
        self.main_toolbar.addWidget(plan_label)

        self.plan_combo = QComboBox()
        self.plan_combo.setMinimumWidth(150)
        self.main_toolbar.addWidget(self.plan_combo)

        # Add plan management buttons
        new_plan_btn = QPushButton("New Plan")
        new_plan_btn.setIcon(QIcon.fromTheme("document-new"))
        self.main_toolbar.addWidget(new_plan_btn)

        save_plan_btn = QPushButton("Save Plan")
        save_plan_btn.setIcon(QIcon.fromTheme("document-save"))
        self.main_toolbar.addWidget(save_plan_btn)

        self.main_toolbar.addSeparator()

        # Calculation buttons
        calc_btn = QPushButton("Calculate Dose")
        calc_btn.setIcon(QIcon.fromTheme("system-run"))
        self.main_toolbar.addWidget(calc_btn)

        optimize_btn = QPushButton("Optimize")
        optimize_btn.setIcon(QIcon.fromTheme("preferences-system"))
        self.main_toolbar.addWidget(optimize_btn)

        main_layout.addWidget(self.main_toolbar)

        # Main splitter (Eclipse-like layout)
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Left panel: Plan Explorer
        self.plan_explorer = QWidget()
        plan_explorer_layout = QVBoxLayout(self.plan_explorer)
        plan_explorer_layout.setContentsMargins(0, 0, 0, 0)

        # Create Object Explorer (Eclipse-like)
        self.object_explorer = QTreeWidget()
        self.object_explorer.setHeaderLabels(["Objects"])
        self.object_explorer.setMinimumWidth(250)
        self.object_explorer.setContextMenuPolicy(Qt.CustomContextMenu)
        self.object_explorer.customContextMenuRequested.connect(
            self._show_object_explorer_menu
        )
        plan_explorer_layout.addWidget(self.object_explorer)

        # Create beam management panel
        self.beams_table = QTableWidget()
        self.beams_table.setColumnCount(5)
        self.beams_table.setHorizontalHeaderLabels(
            ["ID", "Name", "Technique", "Energy", "MU"]
        )
        self.beams_table.setMinimumHeight(200)
        plan_explorer_layout.addWidget(QLabel("Beams:"))
        plan_explorer_layout.addWidget(self.beams_table)

        # Beam control buttons
        beam_buttons_layout = QHBoxLayout()
        self.add_beam_btn = QPushButton("Add")
        self.add_beam_btn.clicked.connect(self._add_beam)

        self.edit_beam_btn = QPushButton("Edit")
        self.edit_beam_btn.clicked.connect(self._edit_beam)

        self.delete_beam_btn = QPushButton("Delete")
        self.delete_beam_btn.clicked.connect(self._delete_beam)

        beam_buttons_layout.addWidget(self.add_beam_btn)
        beam_buttons_layout.addWidget(self.edit_beam_btn)
        beam_buttons_layout.addWidget(self.delete_beam_btn)
        plan_explorer_layout.addLayout(beam_buttons_layout)

        # Add to main splitter
        self.main_splitter.addWidget(self.plan_explorer)

        # Middle: Treatment visualization area with tabs (Eclipse-like)
        self.treatment_view = QTabWidget()

        # MPR View
        from quangtps.ui.mpr_viewer import MPRViewer

        self.mpr_viewer = MPRViewer()
        self.mpr_viewer.setMinimumWidth(600)
        self.treatment_view.addTab(self.mpr_viewer, "MPR")

        # Connect MPR viewer signals
        self.mpr_viewer.sliceChanged.connect(self._on_slice_changed)
        self.mpr_viewer.mousePressed.connect(self._on_mpr_mouse_pressed)
        self.mpr_viewer.mouseMoved.connect(self._on_mpr_mouse_moved)
        self.mpr_viewer.mouseReleased.connect(self._on_mpr_mouse_released)

        # 3D View with dose visualization
        self.dose_3d_view = DoseVisualization3D()
        self.treatment_view.addTab(self.dose_3d_view, "3D")

        # Beam's Eye View
        self.bev_view = QWidget()
        self.treatment_view.addTab(self.bev_view, "BEV")

        # Add treatment view to splitter
        self.main_splitter.addWidget(self.treatment_view)

        # Right: Planning Controls
        self.planning_controls = QTabWidget()
        self.planning_controls.setMinimumWidth(250)

        # Prescription panel
        self.prescription_panel = QWidget()
        prescription_layout = QVBoxLayout(self.prescription_panel)

        # Prescription form
        prescription_form = QFormLayout()

        self.target_combo = QComboBox()
        prescription_form.addRow("Target:", self.target_combo)

        self.technique_combo = QComboBox()
        self.technique_combo.addItems(["3D CRT", "IMRT", "VMAT"])
        prescription_form.addRow("Technique:", self.technique_combo)

        self.dose_input = QDoubleSpinBox()
        self.dose_input.setRange(0.1, 100.0)
        self.dose_input.setValue(2.0)
        self.dose_input.setSuffix(" Gy")
        prescription_form.addRow("Dose:", self.dose_input)

        self.fractions_input = QSpinBox()
        self.fractions_input.setRange(1, 50)
        self.fractions_input.setValue(1)
        prescription_form.addRow("Fractions:", self.fractions_input)

        prescription_layout.addLayout(prescription_form)

        # Apply prescription button
        self.apply_prescription_btn = QPushButton("Apply Prescription")
        prescription_layout.addWidget(self.apply_prescription_btn)

        prescription_layout.addStretch()

        # Add to planning controls
        self.planning_controls.addTab(self.prescription_panel, "Prescription")

        # Optimization panel
        self.optimization_panel = QWidget()
        optimization_layout = QVBoxLayout(self.optimization_panel)

        # Add optimization controls here
        self.normal_optimization_btn = QPushButton("Standard Optimization")
        self.normal_optimization_btn.clicked.connect(self._optimize_plan)

        self.mco_btn = QPushButton("Multi-Criteria Optimization")
        self.mco_btn.clicked.connect(self._open_mco_navigator)

        optimization_layout.addWidget(self.normal_optimization_btn)
        optimization_layout.addWidget(self.mco_btn)
        optimization_layout.addStretch()

        # Add to planning controls
        self.planning_controls.addTab(self.optimization_panel, "Optimization")

        # Dose calculation panel
        self.dose_panel = QWidget()
        dose_layout = QVBoxLayout(self.dose_panel)

        # Algorithm selection
        algorithm_form = QFormLayout()
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(
            ["Pencil Beam", "AAA", "Acuros XB", "Monte Carlo"]
        )
        algorithm_form.addRow("Algorithm:", self.algorithm_combo)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["2.5 mm", "2.0 mm", "1.5 mm", "1.0 mm"])
        algorithm_form.addRow("Resolution:", self.resolution_combo)

        dose_layout.addLayout(algorithm_form)

        # Calculation button
        self.calculate_dose_btn = QPushButton("Calculate Dose")
        self.calculate_dose_btn.clicked.connect(self._calculate_dose)
        dose_layout.addWidget(self.calculate_dose_btn)

        # Dose display options
        dose_layout.addWidget(QLabel("Dose Display:"))

        # Colorwash slider
        colorwash_layout = QHBoxLayout()
        colorwash_layout.addWidget(QLabel("Colorwash:"))
        self.dose_slider = QSlider(Qt.Horizontal)
        self.dose_slider.setRange(0, 100)
        self.dose_slider.setValue(50)
        self.dose_slider.valueChanged.connect(self._update_dose_display)
        colorwash_layout.addWidget(self.dose_slider)
        self.dose_value_label = QLabel("50%")
        colorwash_layout.addWidget(self.dose_value_label)
        dose_layout.addLayout(colorwash_layout)

        dose_layout.addStretch()

        # Add to planning controls
        self.planning_controls.addTab(self.dose_panel, "Dose")

        # Evaluation panel
        self.evaluation_panel = QWidget()
        evaluation_layout = QVBoxLayout(self.evaluation_panel)

        # DVH button
        self.show_dvh_btn = QPushButton("Show DVH")
        self.show_dvh_btn.clicked.connect(self._show_dvh)
        evaluation_layout.addWidget(self.show_dvh_btn)

        # Plan evaluation button
        self.evaluate_plan_btn = QPushButton("Evaluate Plan")
        self.evaluate_plan_btn.clicked.connect(self._evaluate_plan)
        evaluation_layout.addWidget(self.evaluate_plan_btn)

        # Initialize matplotlib for DVH visualization
        if MATPLOTLIB_AVAILABLE:
            self.dvh_figure = Figure(figsize=(4, 4), dpi=100)
            self.dvh_ax = self.dvh_figure.add_subplot(111)
            self.dvh_canvas = FigureCanvas(self.dvh_figure)
            self.dvh_canvas.setMinimumHeight(200)
            evaluation_layout.addWidget(self.dvh_canvas)

            # DVH metrics table
            self.metrics_table = QTableWidget(0, 6)
            self.metrics_table.setHorizontalHeaderLabels(
                ["Structure", "Min", "Max", "Mean", "D95", "V95"]
            )
            self.metrics_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch
            )
            self.metrics_table.setMinimumHeight(100)
            evaluation_layout.addWidget(self.metrics_table)

        evaluation_layout.addStretch()

        # Add to planning controls
        self.planning_controls.addTab(self.evaluation_panel, "Evaluation")

        # Add planning controls to splitter
        self.main_splitter.addWidget(self.planning_controls)

        # Set initial splitter sizes
        self.main_splitter.setSizes([250, 600, 250])

        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        main_layout.addWidget(self.status_bar)

        # Connect signals
        self.patient_combo.currentIndexChanged.connect(self._on_patient_changed)
        self.plan_combo.currentIndexChanged.connect(self._on_plan_changed)
        new_plan_btn.clicked.connect(self._create_new_plan)
        save_plan_btn.clicked.connect(self._save_current_plan)
        calc_btn.clicked.connect(self._calculate_dose)
        optimize_btn.clicked.connect(self._optimize_plan)
        self.treatment_view.currentChanged.connect(self._on_view_tab_changed)
        self.apply_prescription_btn.clicked.connect(self._apply_prescription)
        self.dose_slider.valueChanged.connect(self._update_dose_display)

        # Set up plan name/date edit fields
        self.plan_name_edit = QLineEdit()
        self.plan_date_edit = QDateEdit()
        self.plan_date_edit.setCalendarPopup(True)
        self.plan_status_combo = QComboBox()
        self.plan_status_combo.addItems(["PLANNING", "APPROVED", "DELIVERED"])

    # Add MPR viewer event handlers
    def _on_slice_changed(self, slice_idx, orientation):
        """Handle slice change in MPR viewer."""
        # Update the corresponding view
        if hasattr(self, "current_dose_grid") and self.current_dose_grid:
            self._update_dose_overlay()

        # Update status bar with slice position
        from quangtps.ui.mpr_viewer import ViewOrientation

        orientation_str = "Axial"
        if orientation == ViewOrientation.SAGITTAL:
            orientation_str = "Sagittal"
        elif orientation == ViewOrientation.CORONAL:
            orientation_str = "Coronal"

        self.status_bar.showMessage(f"{orientation_str} Slice: {slice_idx}")

    def _on_mpr_mouse_pressed(self, view_id, view_pos, image_pos):
        """Handle mouse press event in MPR viewer."""
        # Can be used for various interactions (contouring, beam positioning, etc.)
        pass

    def _on_mpr_mouse_moved(self, view_id, view_pos, image_pos):
        """Handle mouse move event in MPR viewer."""
        # Can be used for various interactions and to show current position/dose value
        if hasattr(self, "current_dose_grid") and self.current_dose_grid:
            try:
                # Get the dose at the current position
                x, y, z = (
                    image_pos.x(),
                    image_pos.y(),
                    self.mpr_viewer.get_current_slice_index(),
                )
                dose_value = self.current_dose_grid.get_dose_at_point(x, y, z)

                # Display in status bar
                self.status_bar.showMessage(
                    f"Position: ({x}, {y}, {z}), Dose: {dose_value:.2f} Gy"
                )
            except:
                # In case of errors (out of bounds, etc.)
                self.status_bar.showMessage(
                    f"Position: ({image_pos.x()}, {image_pos.y()})"
                )
        else:
            # Just show position if no dose grid
            self.status_bar.showMessage(f"Position: ({image_pos.x()}, {image_pos.y()})")

    def _on_mpr_mouse_released(self, view_id, view_pos, image_pos):
        """Handle mouse release event in MPR viewer."""
        # Can be used for various interactions (contouring, beam positioning, etc.)
        pass

    def _on_view_tab_changed(self, index):
        """Handle tab change in treatment view."""
        view_type = self.treatment_view.tabText(index)
        self.status_bar.showMessage(f"Switched to {view_type} view")

    def _apply_prescription(self):
        """
        Apply the prescription to the current plan.

        This method retrieves the values from the prescription input fields
        and applies them to the current treatment plan.
        """
        try:
            # Get prescription values from input fields
            target_name = self.target_combo.currentText()
            # Find target ID if available
            target_id = None
            if self.current_plan and hasattr(self.current_plan, "structure_set"):
                for structure in self.current_plan.structure_set.structures:
                    if structure.name == target_name:
                        target_id = structure.id
                        break

            dose = self.dose_input.value()
            fractions = self.fractions_input.value()
            technique = self.technique_combo.currentText()

            # Create prescription
            if (
                hasattr(self.current_plan, "prescription")
                and self.current_plan.prescription is not None
            ):
                if hasattr(self.current_plan.prescription, "targets"):
                    # Update existing prescription
                    self.current_plan.prescription.dose = dose
                    self.current_plan.prescription.fractions = fractions
                    self.current_plan.prescription.technique = technique
                    # Update target if we have the ID
                    if target_id and target_name:
                        self.current_plan.prescription.add_target(
                            name=target_name, dose=dose
                        )
            else:
                # Create new prescription
                from quangtps.planning.prescription import Prescription

                prescription = Prescription(
                    dose=dose, fractions=fractions, technique=technique
                )
                # Add target if we have the ID and name
                if target_id and target_name:
                    prescription.add_target(name=target_name, dose=dose)
                self.current_plan.prescription = prescription

            # Update UI
            self.status_bar.showMessage(
                f"Prescription: {dose} Gy in {fractions} fractions to {target_name}"
            )

            # Update dose visualization
            if hasattr(self.dose_3d_view, "prescription_spinbox"):
                self.dose_3d_view.prescription_spinbox.setValue(dose)

            # Save the plan
            self._save_current_plan()

        except Exception as e:
            logger.error(f"Error applying prescription: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to apply prescription: {str(e)}"
            )

    def _update_dose_overlay(self):
        """Update dose overlay on MPR images."""
        if not hasattr(self, "current_dose_grid") or not self.current_dose_grid:
            return

        try:
            # Convert dose grid to overlay format expected by MPR viewer
            # This will depend on how your MPR viewer handles overlays
            logger.debug("Updating dose overlay on MPR viewer")

            # Example implementation (actual implementation depends on your MPR viewer API)
            if hasattr(self.mpr_viewer, "add_dose_overlay"):
                # Get colormap and opacity from dose slider
                opacity = self.dose_slider.value() / 100.0

                # Get prescription dose for normalization
                prescription_dose = 2.0  # Default
                if hasattr(self.current_plan, "prescription") and hasattr(
                    self.current_plan.prescription, "dose"
                ):
                    prescription_dose = self.current_plan.prescription.dose

                # Add dose overlay to MPR viewer
                self.mpr_viewer.add_dose_overlay(
                    self.current_dose_grid,
                    prescription_dose=prescription_dose,
                    opacity=opacity,
                )

                # Force refresh
                self.mpr_viewer.update_all_views()

        except Exception as e:
            logger.error(f"Error updating dose overlay: {e}")

    def _show_dvh(self):
        """Show DVH for the current plan."""
        if not hasattr(self, "current_dose_grid") or not self.current_dose_grid:
            QMessageBox.warning(self, "Warning", "Please calculate dose first")
            return

        # Switch to the evaluation tab
        self.planning_controls.setCurrentIndex(3)  # Index of Evaluation tab

        try:
            # Calculate DVH
            self._update_evaluation()

            # Can also show a standalone DVH dialog
            from quangtps.ui.dvh_view import DVHView

            dvh_dialog = QDialog(self)
            dvh_dialog.setWindowTitle("Dose Volume Histogram")
            dvh_dialog.setMinimumSize(800, 600)

            dvh_layout = QVBoxLayout(dvh_dialog)
            dvh_view = DVHView()
            dvh_view.set_treatment_plan(self.current_plan)
            dvh_layout.addWidget(dvh_view)

            dvh_dialog.exec_()

        except Exception as e:
            logger.error(f"Error showing DVH: {e}")
            QMessageBox.critical(self, "Error", f"Failed to show DVH: {str(e)}")

    def _evaluate_plan(self):
        """Open plan evaluation dialog."""
        if not hasattr(self, "current_plan") or not self.current_plan:
            QMessageBox.warning(self, "Warning", "Please select a plan first")
            return

        try:
            # Placeholder - in a full implementation, this would open the plan evaluation dialog
            QMessageBox.information(
                self,
                "Plan Evaluation",
                "This is a placeholder for plan evaluation.\n"
                "In a full implementation, this would open the plan evaluation dialog.",
            )
        except Exception as e:
            logger.error(f"Error evaluating plan: {e}")
            QMessageBox.critical(self, "Error", f"Failed to evaluate plan: {str(e)}")

    def _show_object_explorer_menu(self, position):
        """Show context menu for the object explorer."""
        # Get selected item
        selected_items = self.object_explorer.selectedItems()
        if not selected_items:
            return

        # Create context menu
        context_menu = QMenu(self)

        # Add actions based on the type of selected item
        selected_item = selected_items[0]
        item_type = selected_item.text(1)

        if item_type == "Patient":
            context_menu.addAction("View Patient Details")
        elif item_type == "Structure":
            context_menu.addAction("Hide Structure")
            context_menu.addAction("Show Structure")
            context_menu.addSeparator()
            context_menu.addAction("Change Color")
        elif item_type == "Plan":
            context_menu.addAction("Delete Plan")

        # Show the menu
        context_menu.exec_(self.object_explorer.mapToGlobal(position))

    def set_patient(self, patient):
        """Set the current patient."""
        if not patient:
            return

        self.current_patient = patient

        # Update patient combo
        current_text = self.patient_combo.currentText()
        if current_text != patient.name:
            index = self.patient_combo.findText(patient.name)
            if index >= 0:
                self.patient_combo.setCurrentIndex(index)
            else:
                self.patient_combo.addItem(patient.name)
                self.patient_combo.setCurrentIndex(self.patient_combo.count() - 1)

        # Load patient data
        self.current_image = None
        if hasattr(patient, "images") and patient.images:
            self.current_image = patient.images[0]  # Use first image for now

            # Set image in 3D view
            if hasattr(self.dose_3d_view, "set_image_data"):
                # Get image data, spacing, and origin
                image_data = getattr(self.current_image, "data", None)
                spacing = getattr(self.current_image, "spacing", None)
                origin = getattr(self.current_image, "origin", None)

                if image_data is not None:
                    self.dose_3d_view.set_image_data(image_data, spacing, origin)

            # Set image in MPR viewer
            if hasattr(self.mpr_viewer, "set_image"):
                self.mpr_viewer.set_image(self.current_image)

        self.current_structure_set = None
        if hasattr(patient, "structure_set") and patient.structure_set:
            self.current_structure_set = patient.structure_set

            # Add structures to 3D view
            if (
                hasattr(self.dose_3d_view, "add_structure")
                and self.current_structure_set
            ):
                for structure in self.current_structure_set.structures:
                    # Get structure data
                    structure_id = getattr(structure, "id", f"struct_{id(structure)}")
                    mask = getattr(structure, "mask", None)
                    color = getattr(structure, "color", (1.0, 0.0, 0.0))
                    name = getattr(structure, "name", structure_id)

                    if mask is not None:
                        self.dose_3d_view.add_structure(
                            structure_id, mask, color, 0.5, name
                        )

            # Add structures to MPR viewer
            if (
                hasattr(self.mpr_viewer, "add_structure_overlay")
                and self.current_structure_set
            ):
                for structure in self.current_structure_set.structures:
                    # Get structure data
                    structure_id = getattr(structure, "id", f"struct_{id(structure)}")
                    mask = getattr(structure, "mask", None)
                    color = getattr(structure, "color", (1.0, 0.0, 0.0))

                    if mask is not None:
                        self.mpr_viewer.add_structure_overlay(
                            structure_id, structure, color
                        )

        # Load patient plans
        self._load_patient_plans()

        # Update target structures dropdown
        self.target_combo.clear()
        if self.current_structure_set:
            for structure in self.current_structure_set.structures:
                if hasattr(structure, "type") and structure.type == "PTV":
                    self.target_combo.addItem(structure.name, structure.id)

        # Update object explorer
        self._update_object_explorer()

        # Emit signal
        self.patient_loaded.emit(patient)

        # Update status
        self.status_bar.showMessage(f"Patient {patient.name} loaded.")

    def _load_patient_plans(self):
        """
        Load all plans for the current patient
        """
        self.plan_combo.clear()

        if not self.current_patient:
            return

        try:
            plans = self.plan_db.get_plans_by_patient_id(self.current_patient.id)

            for plan in plans:
                self.plan_combo.addItem(plan.name, plan.id)

        except Exception as e:
            logger.error(f"Error loading patient plans: {e}")
            QMessageBox.critical(self, "Error", f"Could not load plans: {str(e)}")

    def _update_object_explorer(self):
        """Update the object explorer with patient data."""
        self.object_explorer.clear()

        if not self.current_patient:
            return

        # Add patient root item
        patient_item = QTreeWidgetItem(self.object_explorer)
        patient_item.setText(0, self.current_patient.name)
        patient_item.setText(1, "Patient")

        # Add studies, images, etc.
        # This would be expanded in a full implementation

    def _update_evaluation(self):
        """
        Update the evaluation tab with DVH data from the current treatment plan.
        Uses the DVHCalculator from evaluation module to calculate and display DVH data.
        """
        import logging
        import numpy as np
        import matplotlib.cm as cm
        from PyQt5.QtWidgets import QTableWidgetItem
        import SimpleITK as sitk
        from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator

        logging.info("Updating evaluation tab")

        # Clear previous data
        self.dvh_ax.clear()

        # Set up the plot
        self.dvh_ax.set_xlabel("Dose (Gy)")
        self.dvh_ax.set_ylabel("Volume (%)")
        self.dvh_ax.set_title("Dose Volume Histogram")
        self.dvh_ax.grid(True)

        # Check if we have a treatment plan loaded
        if not self.current_plan:
            logging.warning("No treatment plan loaded, showing placeholder DVH")
            self._show_placeholder_dvh()
            return

        # Check if plan has structure set and dose
        if (
            not hasattr(self.current_plan, "structure_set")
            or self.current_plan.structure_set is None
        ):
            logging.warning("No structure set available in the current plan")
            self._show_placeholder_dvh()
            return

        if not hasattr(self.current_plan, "dose") or self.current_plan.dose is None:
            logging.warning("No dose grid available in the current plan")
            self._show_placeholder_dvh()
            return

        # Get dose grid data
        dose_array = self.current_plan.dose.dose_grid

        # Convert dose array to SimpleITK image for DVHCalculator
        if hasattr(self.current_plan.dose, "spacing") and hasattr(
            self.current_plan.dose, "origin"
        ):
            spacing = self.current_plan.dose.spacing
            origin = self.current_plan.dose.origin
        else:
            # Default values if not available
            spacing = (1.0, 1.0, 1.0)
            origin = (0.0, 0.0, 0.0)

        # Create SimpleITK dose image
        dose_sitk = sitk.GetImageFromArray(dose_array)
        dose_sitk.SetSpacing(spacing)
        dose_sitk.SetOrigin(origin)

        # Prepare metrics table
        self.metrics_table.setRowCount(0)

        # Define colors for different structure types
        structure_colors = {
            "PTV": "red",
            "CTV": "orange",
            "GTV": "yellow",
            "OAR": "blue",
            "ORGAN": "green",
            "OTHER": "gray",
        }

        # Create DVH Calculator with 1000 bins for smooth curves
        dvh_calculator = DVHCalculator(num_bins=1000)

        # Track if we have plotted any data
        has_data = False

        try:
            # Process each structure
            for i, structure in enumerate(self.current_plan.structure_set.structures):
                # Skip empty structures or those flagged as not for calculation
                if (
                    not hasattr(structure, "mask")
                    or not structure.mask.any()
                    or getattr(structure, "skip_calc", False)
                ):
                    continue

                # Determine structure type and color
                structure_type = "OTHER"
                for type_key in structure_colors.keys():
                    if type_key in structure.name.upper():
                        structure_type = type_key
                        break

                color = structure_colors.get(structure_type, structure_colors["OTHER"])

                # Convert structure mask to SimpleITK image
                mask_sitk = sitk.GetImageFromArray(structure.mask.astype(np.uint8))
                mask_sitk.SetSpacing(spacing)
                mask_sitk.SetOrigin(origin)

                try:
                    # Calculate DVH data using DVHCalculator
                    dvh_data = dvh_calculator.calculate_dvh_data(
                        dose_sitk, mask_sitk, structure.name, cumulative=True
                    )

                    # Plot DVH
                    self.dvh_ax.plot(
                        dvh_data.dose_bins,
                        dvh_data.volume_bins,
                        color=color,
                        label=structure.name,
                        linewidth=2,
                    )

                    # Add row to metrics table
                    row = self.metrics_table.rowCount()
                    self.metrics_table.insertRow(row)

                    # Structure name
                    self.metrics_table.setItem(row, 0, QTableWidgetItem(structure.name))

                    # Metrics: Min, Max, Mean, D95, V95%
                    self.metrics_table.setItem(
                        row, 1, QTableWidgetItem(f"{dvh_data.min_dose:.1f} Gy")
                    )
                    self.metrics_table.setItem(
                        row, 2, QTableWidgetItem(f"{dvh_data.max_dose:.1f} Gy")
                    )
                    self.metrics_table.setItem(
                        row, 3, QTableWidgetItem(f"{dvh_data.mean_dose:.1f} Gy")
                    )

                    # D95 (dose to 95% of the volume)
                    d95 = dvh_data.get_dx(95.0)
                    self.metrics_table.setItem(
                        row, 4, QTableWidgetItem(f"{d95:.1f} Gy")
                    )

                    # V95% (volume receiving 95% of prescription dose)
                    if (
                        hasattr(self.current_plan, "prescription")
                        and self.current_plan.prescription
                        and hasattr(self.current_plan.prescription, "dose")
                    ):
                        prescription_dose = self.current_plan.prescription.dose
                        v95 = dvh_data.get_vx(0.95 * prescription_dose, percent=True)
                        self.metrics_table.setItem(
                            row, 5, QTableWidgetItem(f"{v95:.1f}%")
                        )
                    else:
                        # If no prescription, use 95% of max dose for this structure
                        v95 = dvh_data.get_vx(0.95 * dvh_data.max_dose, percent=True)
                        self.metrics_table.setItem(
                            row, 5, QTableWidgetItem(f"{v95:.1f}%")
                        )

                    has_data = True

                except Exception as e:
                    logging.error(
                        f"Error calculating DVH for structure {structure.name}: {str(e)}"
                    )
                    continue

        except Exception as e:
            logging.error(f"Error updating evaluation: {str(e)}")
            self._show_placeholder_dvh()
            return

        # Add legend if we have data
        if has_data:
            self.dvh_ax.legend(loc="upper right")
        else:
            self._show_placeholder_dvh()
            return

        # Draw the canvas
        self.dvh_canvas.draw()

    def _calculate_dose(self):
        """Calculate dose for the current plan."""
        if not self.current_plan:
            QMessageBox.warning(
                self, "No Plan", "Please create or select a plan first."
            )
            return

        if not self.current_plan.beams:
            QMessageBox.warning(
                self, "No Beams", "Please add at least one beam to the plan."
            )
            return

        # Use the selected algorithm and resolution
        algorithm = self.algorithm_combo.currentText()
        resolution = self.resolution_combo.currentText()

        # Show progress dialog
        progress = QProgressDialog("Calculating dose...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Dose Calculation")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            # Update status
            self.status_bar.showMessage(f"Calculating dose using {algorithm}...")
            self.calculation_started.emit()

            # Calculate dose
            if self.dose_calculator:
                # Convert resolution string to value
                resolution_value = float(resolution.split()[0])

                # Set algorithm and resolution
                self.dose_calculator.set_algorithm(algorithm)
                self.dose_calculator.set_resolution(resolution_value)

                # Calculate dose
                progress.setValue(10)
                self.current_dose_grid = self.dose_calculator.calculate_dose(
                    self.current_plan
                )
                progress.setValue(90)

                # Update the dose display
                if self.current_dose_grid:
                    # Set dose grid in 3D view
                    self.dose_3d_view.set_dose_grid(self.current_dose_grid)

                    # Set prescription dose
                    prescription = float(self.dose_input.value())
                    if hasattr(self.dose_3d_view, "prescription_spinbox"):
                        self.dose_3d_view.prescription_spinbox.setValue(prescription)

                    # Update MPR view with dose overlay
                    self._update_dose_overlay()

                    # Update dose display
                    self._update_dose_display(self.dose_slider.value())

                    # Update evaluation
                    self._update_evaluation()

                    self.status_bar.showMessage(
                        "Dose calculation completed successfully."
                    )
                else:
                    self.status_bar.showMessage("Dose calculation failed.")
            else:
                self.status_bar.showMessage("Dose calculator not available.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Dose calculation failed: {str(e)}")
            self.status_bar.showMessage("Dose calculation failed with error.")
            logging.error(f"Dose calculation error: {str(e)}")
        finally:
            progress.setValue(100)
            self.calculation_finished.emit()

            # Ensure we're showing the correct view for dose visualization
            if self.current_dose_grid:
                # Switch to the view that best shows the dose
                current_tab = self.treatment_view.currentIndex()
                if current_tab == 0:  # If we're on MPR view, update it
                    self._update_dose_overlay()
                elif current_tab != 1:  # If not on 3D view, switch to it
                    self.treatment_view.setCurrentIndex(1)  # Switch to 3D view

    def _show_placeholder_dvh(self):
        """
        Show placeholder DVH data for demonstration when no real data is available.
        Creates sample DVH curves and metrics for educational purposes.
        """
        import numpy as np
        from PyQt5.QtWidgets import QTableWidgetItem

        # Clear and set up plot
        self.dvh_ax.clear()
        self.dvh_ax.set_xlabel("Dose (Gy)")
        self.dvh_ax.set_ylabel("Volume (%)")
        self.dvh_ax.set_title("Dose Volume Histogram (Demo)")
        self.dvh_ax.grid(True)

        # Sample data for demonstration
        # Create sample dose points (0 to 80 Gy)
        dose_points = np.linspace(0, 80, 100)

        # PTV curve (ideal sharp falloff at prescription dose)
        prescription = 60  # Gy
        ptv_volumes = 100 * np.ones_like(dose_points)
        ptv_volumes[dose_points > prescription * 0.95] = 100 * np.exp(
            -(dose_points[dose_points > prescription * 0.95] - prescription * 0.95) / 2
        )

        # OAR curve (gradual falloff)
        oar_volumes = 100 * np.exp(-dose_points / 20)

        # Plot curves
        self.dvh_ax.plot(dose_points, ptv_volumes, "r-", label="PTV (Demo)")
        self.dvh_ax.plot(dose_points, oar_volumes, "b-", label="OAR (Demo)")

        # Set axis limits
        self.dvh_ax.set_xlim(0, 80)
        self.dvh_ax.set_ylim(0, 100)

        # Add legend
        self.dvh_ax.legend(loc="upper right")

        # Draw the canvas
        self.dvh_canvas.draw()

        # Add placeholder metrics to table
        self.metrics_table.setRowCount(0)

        # Add PTV row
        self.metrics_table.insertRow(0)
        self.metrics_table.setItem(0, 0, QTableWidgetItem("PTV (Demo)"))
        self.metrics_table.setItem(
            0, 1, QTableWidgetItem(f"{prescription * 0.9:.1f} Gy")
        )
        self.metrics_table.setItem(
            0, 2, QTableWidgetItem(f"{prescription * 1.1:.1f} Gy")
        )
        self.metrics_table.setItem(0, 3, QTableWidgetItem(f"{prescription:.1f} Gy"))
        self.metrics_table.setItem(
            0, 4, QTableWidgetItem(f"{prescription * 0.98:.1f} Gy")
        )
        self.metrics_table.setItem(0, 5, QTableWidgetItem("95.0%"))

        # Add OAR row
        self.metrics_table.insertRow(1)
        self.metrics_table.setItem(1, 0, QTableWidgetItem("OAR (Demo)"))
        self.metrics_table.setItem(1, 1, QTableWidgetItem("0.0 Gy"))
        self.metrics_table.setItem(
            1, 2, QTableWidgetItem(f"{prescription * 0.8:.1f} Gy")
        )
        self.metrics_table.setItem(
            1, 3, QTableWidgetItem(f"{prescription * 0.3:.1f} Gy")
        )
        self.metrics_table.setItem(1, 4, QTableWidgetItem("0.0 Gy"))
        self.metrics_table.setItem(1, 5, QTableWidgetItem("15.0%"))
