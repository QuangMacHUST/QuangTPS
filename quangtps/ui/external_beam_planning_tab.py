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
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QListWidgetItem, QSplitter, QDialog, QColorDialog, QComboBox, 
    QLineEdit, QFormLayout, QMessageBox, QFileDialog, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QProgressDialog, QMenu, QAction,
    QToolBar, QGroupBox, QRadioButton, QButtonGroup, QCheckBox, QSlider,
    QSpinBox, QDoubleSpinBox, QToolButton, QFrame, QScrollArea, QStatusBar,
    QTableWidget, QTableWidgetItem, QDateEdit, QInputDialog
)
from PyQt5.QtGui import QColor, QIcon, QBrush, QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QDate

# Import matplotlib for visualization if available
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
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
    from quangtps.optimization.optimization_engine import OptimizationEngine, OptimizationParameters
    from quangtps.optimization.objectives import ObjectiveCollection
    from quangtps.optimization.constraints import ConstraintCollection
    
    # Import evaluation modules
    from quangtps.evaluation.plan_evaluation import PlanEvaluation
    from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh, calculate_dvh_metrics
    from quangtps.evaluation.dvh.dvh_visualization import plot_dvh
    
    # Import UI modules
    from quangtps.ui.dialogs.beam_dialog import BeamDialog
    from quangtps.ui.beam_visualization_panel import BeamVisualizationPanel
    
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
        self.optimization_engine = self.service_registry.get_service("OptimizationEngine")
        
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
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Object Explorer (shows structures, plans, etc.)
        self.object_explorer = QTreeWidget()
        self.object_explorer.setHeaderLabels(["Name", "Type"])
        self.object_explorer.setColumnCount(2)
        self.object_explorer.setMinimumWidth(250)
        left_layout.addWidget(self.object_explorer, 1)
        
        # Patient info panel
        patient_info = QGroupBox("Patient Information")
        patient_info_layout = QFormLayout()
        self.patient_name_label = QLabel("No patient selected")
        self.patient_id_label = QLabel("")
        patient_info_layout.addRow("Name:", self.patient_name_label)
        patient_info_layout.addRow("ID:", self.patient_id_label)
        patient_info.setLayout(patient_info_layout)
        left_layout.addWidget(patient_info)
        
        # Center panel
        self.center_panel = QTabWidget()
        
        # Plan tab
        self.plan_tab = QWidget()
        plan_layout = QVBoxLayout(self.plan_tab)
        
        # Plan properties
        plan_properties = QGroupBox("Plan Properties")
        plan_properties_layout = QFormLayout()
        
        self.plan_name_edit = QLineEdit()
        plan_properties_layout.addRow("Name:", self.plan_name_edit)
        
        self.plan_date_edit = QDateEdit()
        self.plan_date_edit.setCalendarPopup(True)
        self.plan_date_edit.setDate(QDate.currentDate())
        plan_properties_layout.addRow("Date:", self.plan_date_edit)
        
        self.plan_type_combo = QComboBox()
        self.plan_type_combo.addItems(["Treatment", "QA", "Research"])
        plan_properties_layout.addRow("Type:", self.plan_type_combo)
        
        self.plan_status_combo = QComboBox()
        self.plan_status_combo.addItems(["Planning", "Approved", "Delivered"])
        plan_properties_layout.addRow("Status:", self.plan_status_combo)
        
        plan_properties.setLayout(plan_properties_layout)
        plan_layout.addWidget(plan_properties)
        
        # Prescription group
        prescription_group = QGroupBox("Prescription")
        prescription_layout = QFormLayout()
        
        self.dose_edit = QDoubleSpinBox()
        self.dose_edit.setRange(0, 100)
        self.dose_edit.setSuffix(" Gy")
        prescription_layout.addRow("Dose:", self.dose_edit)
        
        self.fractions_edit = QSpinBox()
        self.fractions_edit.setRange(1, 50)
        prescription_layout.addRow("Fractions:", self.fractions_edit)
        
        prescription_group.setLayout(prescription_layout)
        plan_layout.addWidget(prescription_group)
        
        # Beams table
        beams_group = QGroupBox("Beams")
        beams_layout = QVBoxLayout()
        
        self.beams_table = QTableWidget()
        self.beams_table.setColumnCount(6)
        self.beams_table.setHorizontalHeaderLabels(["ID", "Name", "Technique", "Gantry", "Collimator", "Energy"])
        beams_layout.addWidget(self.beams_table)
        
        # Beam buttons
        beam_buttons = QHBoxLayout()
        add_beam_btn = QPushButton("Add Beam")
        edit_beam_btn = QPushButton("Edit Beam")
        delete_beam_btn = QPushButton("Delete Beam")
        beam_buttons.addWidget(add_beam_btn)
        beam_buttons.addWidget(edit_beam_btn)
        beam_buttons.addWidget(delete_beam_btn)
        beams_layout.addLayout(beam_buttons)
        
        beams_group.setLayout(beams_layout)
        plan_layout.addWidget(beams_group)
        
        # Add stretch to push everything to the top
        plan_layout.addStretch()
        
        # Evaluation tab
        self.evaluation_tab = QWidget()
        evaluation_layout = QVBoxLayout(self.evaluation_tab)
        
        if MATPLOTLIB_AVAILABLE:
            self.dvh_figure = Figure(figsize=(8, 6))
            self.dvh_canvas = FigureCanvas(self.dvh_figure)
            self.dvh_ax = self.dvh_figure.add_subplot(111)
            self.dvh_ax.set_xlabel('Dose (Gy)')
            self.dvh_ax.set_ylabel('Volume (%)')
            self.dvh_ax.set_title('Dose Volume Histogram')
            self.dvh_ax.grid(True)
            evaluation_layout.addWidget(self.dvh_canvas)
        else:
            evaluation_layout.addWidget(QLabel("Matplotlib not available for DVH visualization"))
        
        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels(["Structure", "Min Dose", "Max Dose", "Mean Dose", "D95", "V20"])
        evaluation_layout.addWidget(self.metrics_table)
        
        # Add tabs to center panel
        self.center_panel.addTab(self.plan_tab, "Planning")
        self.center_panel.addTab(self.evaluation_tab, "Plan Evaluation")
        
        # Right panel: Dose visualization
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Dose visualization controls
        dose_controls = QGroupBox("Dose Display")
        dose_controls_layout = QHBoxLayout()
        
        self.dose_slider = QSlider(Qt.Horizontal)
        self.dose_slider.setRange(0, 100)
        self.dose_slider.setValue(70)
        dose_controls_layout.addWidget(QLabel("Normalization:"))
        dose_controls_layout.addWidget(self.dose_slider)
        self.dose_value_label = QLabel("70%")
        dose_controls_layout.addWidget(self.dose_value_label)
        
        dose_controls.setLayout(dose_controls_layout)
        right_layout.addWidget(dose_controls)
        
        # Placeholder for dose visualization (would integrate with image_display.py)
        self.dose_display = QLabel("Dose visualization will be shown here")
        self.dose_display.setStyleSheet("background-color: #444; color: white; padding: 20px;")
        self.dose_display.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.dose_display, 1)
        
        # Add all panels to main splitter
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.center_panel)
        self.main_splitter.addWidget(self.right_panel)
        
        # Set initial split sizes (left:center:right)
        self.main_splitter.setSizes([200, 500, 300])
        
        main_layout.addWidget(self.main_splitter, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        main_layout.addWidget(self.status_bar)
        
        # Connect signals
        if MODULES_AVAILABLE:
            self.patient_combo.currentIndexChanged.connect(self._on_patient_changed)
            self.plan_combo.currentIndexChanged.connect(self._on_plan_changed)
            new_plan_btn.clicked.connect(self._create_new_plan)
            save_plan_btn.clicked.connect(self._save_current_plan)
            calc_btn.clicked.connect(self._calculate_dose)
            optimize_btn.clicked.connect(self._optimize_plan)
            add_beam_btn.clicked.connect(self._add_beam)
            edit_beam_btn.clicked.connect(self._edit_beam)
            delete_beam_btn.clicked.connect(self._delete_beam)
            self.dose_slider.valueChanged.connect(self._update_dose_display)
    
    def _on_patient_changed(self, index):
        """Handle patient selection change."""
        if index < 0 or not MODULES_AVAILABLE:
            return
            
        patient_id = self.patient_combo.itemData(index)
        if patient_id:
            try:
                patient = self.patient_db.get_patient(patient_id)
                self.set_patient(patient)
            except Exception as e:
                logger.error(f"Error loading patient: {e}")
                QMessageBox.warning(self, "Error", f"Failed to load patient: {str(e)}")
    
    def _on_plan_changed(self, index):
        """Handle plan selection change."""
        if index < 0 or not MODULES_AVAILABLE:
            return
            
        plan_id = self.plan_combo.itemData(index)
        if plan_id and self.current_patient:
            try:
                plan = self.plan_db.get_plan(plan_id)
                self.set_plan(plan)
            except Exception as e:
                logger.error(f"Error loading plan: {e}")
                QMessageBox.warning(self, "Error", f"Failed to load plan: {str(e)}")
    
    def _create_new_plan(self):
        """
        Create a new treatment plan for the current patient
        """
        if not self.current_patient:
            QMessageBox.warning(self, "Warning", "Please select a patient first")
            return
        
        # Create plan dialog
        plan_name, ok = QInputDialog.getText(self, "New Plan", "Enter plan name:")
        
        if not ok or not plan_name:
            return
        
        try:
            plan = Plan(plan_name=plan_name, patient_id=self.current_patient.id)
            plan.name = plan_name
            plan.patient_id = self.current_patient.id
            plan.type = PlanType.EXTERNAL_BEAM
            plan.status = PlanStatus.PLANNING
            plan.created_date = datetime.datetime.now()
            
            # Save to database
            plan_id = self.plan_db.create_plan(plan)
            plan.id = plan_id
            
            # Update UI
            self._load_patient_plans()
            self.plan_combo.setCurrentText(plan.name)
            self.set_plan(plan)
            
            # Emit signal
            self.plan_created.emit(plan)
            
            QMessageBox.information(self, "Success", f"Plan '{plan_name}' created successfully.")
        
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            QMessageBox.critical(self, "Error", f"Could not create plan: {str(e)}")
    
    def _save_current_plan(self):
        """Save the current plan to the database."""
        if not self.current_plan or not MODULES_AVAILABLE:
            return
            
        try:
            # Update plan from UI
            self.current_plan.name = self.plan_name_edit.text()
            self.current_plan.status = PlanStatus[self.plan_status_combo.currentText().upper()]
            
            # Save to database
            self.plan_db.update_plan(self.current_plan)
            
            self.plan_updated.emit(self.current_plan)
            self.status_bar.showMessage(f"Plan saved: {self.current_plan.name}")
            
        except Exception as e:
            logger.error(f"Error saving plan: {e}")
            QMessageBox.warning(self, "Error", f"Failed to save plan: {str(e)}")
    
    def _calculate_dose(self):
        """Calculate dose for the current plan."""
        if not self.current_plan or not MODULES_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Please select a plan first")
            return
        
        self.status_bar.showMessage("Calculating dose... Please wait")
        self.calculation_started.emit()
        
        # This would be a long-running operation in a real implementation
        # For now, we'll just show a placeholder message
        QMessageBox.information(self, "Dose Calculation", 
                           "This is a placeholder for dose calculation.\n"
                           "In a full implementation, this would calculate dose for all beams.")
        
        self.status_bar.showMessage("Dose calculation complete")
        self.calculation_finished.emit()
    
    def _optimize_plan(self):
        """Open the optimization dialog."""
        if not self.current_plan:
            QMessageBox.warning(self, "Warning", "Please select a plan first")
            return
        
        try:
            # Check if we should use MCO
            use_mco = QMessageBox.question(
                self,
                "Optimization Method",
                "Would you like to use Multi-Criteria Optimization (MCO)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            ) == QMessageBox.Yes
            
            if use_mco:
                # Use MCO Navigator
                self._open_mco_navigator()
            else:
                # Use traditional optimization
                QMessageBox.information(
                    self,
                    "Standard Optimization",
                    "This is a placeholder for standard optimization.\n"
                    "In a full implementation, this would open the standard optimization dialog."
                )
        except Exception as e:
            logger.error(f"Error opening optimization dialog: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open optimization dialog: {str(e)}")
    
    def _open_mco_navigator(self):
        """Open the MCO Navigator dialog."""
        if not self.current_plan:
            return
        
        try:
            # Import the MCO Navigator dialog
            from quangtps.ui.mco_navigator_dialog import MCONavigatorDialog
            
            # Create and show the dialog
            dialog = MCONavigatorDialog(self.current_plan, self)
            dialog.solutionAccepted.connect(self._on_mco_solution_accepted)
            
            # Show as modal dialog
            dialog.exec_()
        except ImportError:
            logger.error("MCO Navigator dialog not available")
            QMessageBox.warning(
                self,
                "MCO Not Available",
                "The Multi-Criteria Optimization module is not available.\n"
                "Please make sure all dependencies are installed."
            )
        except Exception as e:
            logger.error(f"Error opening MCO Navigator: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open MCO Navigator: {str(e)}")
    
    def _on_mco_solution_accepted(self, solution):
        """
        Handle an accepted MCO solution.
        
        Parameters
        ----------
        solution : MCOSolution
            The selected solution from the MCO Navigator
        """
        try:
            # Replace the current plan with the solution's plan
            optimized_plan = solution.plan
            
            # Update plan in database
            optimized_plan.id = self.current_plan.id  # Keep same ID
            self.plan_db.update_plan(optimized_plan)
            
            # Update current plan
            self.current_plan = optimized_plan
            
            # Update UI
            self.set_plan(optimized_plan)
            
            # Emit plan updated signal
            self.plan_updated.emit(optimized_plan)
            
            # Display confirmation
            QMessageBox.information(
                self,
                "MCO Solution Applied",
                "The selected MCO solution has been applied to the current plan."
            )
            
            # Log metrics
            logger.info(f"Applied MCO solution with objectives: {solution.objectives}")
            
        except Exception as e:
            logger.error(f"Error applying MCO solution: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply MCO solution: {str(e)}")
    
    def _add_beam(self):
        """Add a new beam to the current plan."""
        if not self.current_plan or not MODULES_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Please select a plan first")
            return
            
        QMessageBox.information(self, "Add Beam", 
                           "This is a placeholder for adding a beam.\n"
                           "In a full implementation, this would open the beam dialog.")
    
    def _edit_beam(self):
        """Edit the selected beam."""
        if not self.current_plan or not MODULES_AVAILABLE:
            return
            
        selected_items = self.beams_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a beam to edit")
            return
            
        QMessageBox.information(self, "Edit Beam", 
                           "This is a placeholder for editing a beam.\n"
                           "In a full implementation, this would open the beam dialog with the selected beam.")
    
    def _delete_beam(self):
        """Delete the selected beam."""
        if not self.current_plan or not MODULES_AVAILABLE:
            return
            
        selected_items = self.beams_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a beam to delete")
            return
            
        QMessageBox.information(self, "Delete Beam", 
                           "This is a placeholder for deleting a beam.\n"
                           "In a full implementation, this would delete the selected beam.")
    
    def _update_dose_display(self, value):
        """Update the dose display based on slider value."""
        self.dose_value_label.setText(f"{value}%")
        # In a real implementation, this would update the dose visualization
    
    def set_patient(self, patient):
        """Set the current patient and update the UI."""
        self.current_patient = patient
        
        if patient:
            self.patient_name_label.setText(patient.name)
            self.patient_id_label.setText(patient.id)
            
            # Clear current plan
            self.current_plan = None
            
            # Load patient's plans
            self._load_patient_plans()
            
            # Update object explorer
            self._update_object_explorer()
            
            self.patient_loaded.emit(patient)
        else:
            self.patient_name_label.setText("No patient selected")
            self.patient_id_label.setText("")
            self.plan_combo.clear()
    
    def set_plan(self, plan):
        """Set the current plan and update the UI."""
        self.current_plan = plan
        
        if plan:
            # Update plan properties
            self.plan_name_edit.setText(plan.name)
            
            try:
                plan_date = plan.created_date
                if isinstance(plan_date, datetime.datetime):
                    self.plan_date_edit.setDate(QDate(plan_date.year, plan_date.month, plan_date.day))
            except (AttributeError, ValueError):
                # Use current date as fallback
                self.plan_date_edit.setDate(QDate.currentDate())
            
            # Update beams table
            self._update_beams_table()
            
            # Update evaluation metrics if dose grid exists
            if hasattr(plan, 'dose_grid') and plan.dose_grid is not None:
                self._update_evaluation()
        
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
    
    def _update_beams_table(self):
        """Update the beams table with the current plan's beams."""
        if not self.current_plan:
            self.beams_table.setRowCount(0)
            return
            
        self.beams_table.setRowCount(0)
        
        if not hasattr(self.current_plan, 'beams') or not self.current_plan.beams:
            return
            
        for i, beam in enumerate(self.current_plan.beams):
            self.beams_table.insertRow(i)
            
            # Fill in beam data
            self.beams_table.setItem(i, 0, QTableWidgetItem(str(beam.id)))
            self.beams_table.setItem(i, 1, QTableWidgetItem(beam.name))
            self.beams_table.setItem(i, 2, QTableWidgetItem(beam.technique if hasattr(beam, 'technique') else ""))
            
            if hasattr(beam, 'geometry'):
                self.beams_table.setItem(i, 3, QTableWidgetItem(f"{beam.geometry.gantry_angle:.1f}°" if hasattr(beam.geometry, 'gantry_angle') else ""))
                self.beams_table.setItem(i, 4, QTableWidgetItem(f"{beam.geometry.collimator_angle:.1f}°" if hasattr(beam.geometry, 'collimator_angle') else ""))
            
            self.beams_table.setItem(i, 5, QTableWidgetItem(beam.energy if hasattr(beam, 'energy') else ""))
    
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
        """Update the evaluation tab with current plan data."""
        if not self.current_plan or not hasattr(self.current_plan, 'dose_grid') or not self.current_plan.dose_grid:
            return
            
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # Placeholder - in a real implementation, this would calculate and display DVH data
        self.dvh_ax.clear()
        self.dvh_ax.set_xlabel('Dose (Gy)')
        self.dvh_ax.set_ylabel('Volume (%)')
        self.dvh_ax.set_title('Dose Volume Histogram')
        self.dvh_ax.grid(True)
        
        # Example plot
        x = np.linspace(0, 80, 100)
        y1 = 100 * np.exp(-x/20)
        y2 = 100 * np.exp(-x/40)
        
        self.dvh_ax.plot(x, y1, 'r-', label='PTV')
        self.dvh_ax.plot(x, y2, 'b-', label='OAR')
        self.dvh_ax.legend()
        
        self.dvh_canvas.draw()
        
        # Update metrics table (placeholder)
        self.metrics_table.setRowCount(2)
        
        row = 0
        self.metrics_table.setItem(row, 0, QTableWidgetItem("PTV"))
        self.metrics_table.setItem(row, 1, QTableWidgetItem("35.2 Gy"))
        self.metrics_table.setItem(row, 2, QTableWidgetItem("78.1 Gy"))
        self.metrics_table.setItem(row, 3, QTableWidgetItem("68.4 Gy"))
        self.metrics_table.setItem(row, 4, QTableWidgetItem("65.3 Gy"))
        self.metrics_table.setItem(row, 5, QTableWidgetItem("100%"))
        
        row = 1
        self.metrics_table.setItem(row, 0, QTableWidgetItem("OAR"))
        self.metrics_table.setItem(row, 1, QTableWidgetItem("0.0 Gy"))
        self.metrics_table.setItem(row, 2, QTableWidgetItem("54.7 Gy"))
        self.metrics_table.setItem(row, 3, QTableWidgetItem("24.3 Gy"))
        self.metrics_table.setItem(row, 4, QTableWidgetItem("8.2 Gy"))
        self.metrics_table.setItem(row, 5, QTableWidgetItem("35%")) 