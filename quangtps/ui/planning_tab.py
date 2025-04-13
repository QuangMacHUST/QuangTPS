#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module giao diện tab lập kế hoạch.

Module này cung cấp giao diện người dùng cho phần lập kế hoạch điều trị.
"""

from typing import List, Dict, Any, Union
from datetime import datetime, date
import json
import logging
import os
import sys
from pathlib import Path
import uuid

from PyQt5.QtCore import Qt, pyqtSignal, QSize, QDate
from PyQt5.QtGui import QIcon, QColor, QPixmap, QPalette
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QTabWidget, QLineEdit, QScrollArea, QSplitter,
    QMessageBox, QGroupBox, QHeaderView, QCheckBox, QComboBox, QFileDialog,
    QDoubleSpinBox, QSpinBox, QRadioButton, QFrame, QApplication,
    QFormLayout, QTextEdit, QDateEdit, QDialog, QButtonGroup, QProgressDialog,
    QToolBar, QAction
)

from quangtps.planning.plan import Plan, PlanStatus, PlanType
from quangtps.planning.beam import BeamArrangement
from quangtps.database.plan_db import PlanDB
from quangtps.ui.dialogs.beam_dialog import BeamDialog
from quangtps.planning.prescription import Prescription
from quangtps.planning.optimization import OptimizationSettings
from quangtps.treatment.beams.beam import Beam
from quangtps.optimization.methods import MCOEngine, MCONavigator, MCOTrade, MCOMethod
from quangtps.ui.mco_navigation_dialog import MCONavigationDialog
from quangtps.optimization.objectives import ObjectiveCollection
from quangtps.optimization import ConstraintCollection
from quangtps.optimization.optimization_engine import OptimizationEngine, OptimizationParameters
from quangtps.dose.dose_grid import DoseGrid
# Import 3D CRT Planner
from quangtps.ui.crt_planner import CRTPlanner
from quangtps.treatment.techniques.crt_manager import CRTManager
from quangtps.ui.beam_visualization_panel import BeamVisualizationPanel
# Check for 3D visualization dependency
from quangtps.ui.dependency_installer import check_and_install_feature_dependencies
import numpy as np
import importlib
# Import robust optimization dialog (commented out, using local implementation instead)
#from quangtps.ui.robust_optimization_dialog import show_robust_optimization_dialog
from quangtps.core.logging import get_logger
from quangtps.treatment.techniques.crt.crt_planner import CRTPlanner
from quangtps.treatment.techniques.imrt.imrt_planner import IMRTPlanner

logger = get_logger(__name__)


def show_robust_optimization_dialog(plan, structures, dose_grid=None, parent=None):
    """
    Show the robust optimization dialog.
    
    Args:
        plan: Treatment plan to optimize
        structures: Dictionary of structures
        dose_grid: Optional dose grid for analysis
        parent: Parent widget
    
    Returns:
        int: Dialog result (QDialog.Accepted or QDialog.Rejected)
    """
    from quangtps.ui.robust_optimization_dialog import RobustOptimizationDialog
    
    dialog = RobustOptimizationDialog(plan, structures, dose_grid, parent)
    
    # Connect signals
    dialog.planOptimized.connect(lambda optimized_plan: setattr(plan, 'beams', optimized_plan.beams))
    
    # Show dialog
    result = dialog.exec_()
    
    return result


class PlanningTab(QWidget):
    """
    Tab lập kế hoạch xạ trị.
    
    Tab này bao gồm các công cụ để tạo và chỉnh sửa kế hoạch xạ trị,
    thiết lập kỹ thuật điều trị, thông số chùm tia, và tối ưu hóa kế hoạch.
    """
    
    # Tín hiệu
    plan_created = pyqtSignal(object)
    plan_updated = pyqtSignal(object)
    plan_deleted = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Khởi tạo tab lập kế hoạch.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Trạng thái
        self.current_plan = None
        self.current_technique = None
        self.current_patient_id = None
        
        # Kết nối cơ sở dữ liệu
        self.plan_db = PlanDB()
        
        # Khởi tạo CRT Manager
        self.crt_manager = CRTManager()
        
        # Thiết lập giao diện
        self._init_ui()
        
        logger.info("Khởi tạo tab lập kế hoạch hoàn tất")
        
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add toolbar for common actions
        self.toolbar = QToolBar("Planning Tools")
        
        # Add patient/plan selection
        patient_label = QLabel("Patient:")
        self.toolbar.addWidget(patient_label)
        
        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(200)
        self.patient_combo.currentIndexChanged.connect(self._on_patient_changed)
        self.toolbar.addWidget(self.patient_combo)
        self.toolbar.addSeparator()
        
        # Add plan selection
        plan_label = QLabel("Plan:")
        self.toolbar.addWidget(plan_label)
        
        self.plan_combo = QComboBox()
        self.plan_combo.setMinimumWidth(150)
        self.plan_combo.currentIndexChanged.connect(self._on_plan_changed)
        self.toolbar.addWidget(self.plan_combo)
        
        # Add plan management buttons
        new_plan_btn = QPushButton("New Plan")
        new_plan_btn.setIcon(QIcon.fromTheme("document-new"))
        new_plan_btn.clicked.connect(self._create_plan_dialog)
        self.toolbar.addWidget(new_plan_btn)
        
        save_plan_btn = QPushButton("Save Plan")
        save_plan_btn.setIcon(QIcon.fromTheme("document-save"))
        save_plan_btn.clicked.connect(self._save_plan)
        self.toolbar.addWidget(save_plan_btn)
        
        main_layout.addWidget(self.toolbar)
        
        # Main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel for plan properties and general settings
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        
        # Plan properties group
        plan_group = QGroupBox("Plan Properties")
        plan_layout = QFormLayout()
        
        self.plan_name_edit = QLineEdit()
        plan_layout.addRow("Name:", self.plan_name_edit)
        
        self.plan_desc_edit = QLineEdit()
        plan_layout.addRow("Description:", self.plan_desc_edit)
        
        self.plan_date_edit = QDateEdit()
        self.plan_date_edit.setCalendarPopup(True)
        self.plan_date_edit.setDate(QDate.currentDate())
        plan_layout.addRow("Date:", self.plan_date_edit)
        
        self.plan_type_combo = QComboBox()
        self.plan_type_combo.addItems(["Treatment", "QA", "Research", "Other"])
        plan_layout.addRow("Type:", self.plan_type_combo)
        
        self.plan_status_combo = QComboBox()
        self.plan_status_combo.addItems(["Planning", "Approved", "Delivered", "Archived"])
        plan_layout.addRow("Status:", self.plan_status_combo)
        
        plan_group.setLayout(plan_layout)
        left_layout.addWidget(plan_group)
        
        # Prescription group
        prescription_group = QGroupBox("Prescription")
        prescription_layout = QFormLayout()
        
        self.rx_dose_edit = QDoubleSpinBox()
        self.rx_dose_edit.setRange(0, 1000)
        self.rx_dose_edit.setSuffix(" Gy")
        prescription_layout.addRow("Dose:", self.rx_dose_edit)
        
        self.rx_fractions_edit = QSpinBox()
        self.rx_fractions_edit.setRange(1, 100)
        prescription_layout.addRow("Fractions:", self.rx_fractions_edit)
        
        self.technique_combo = QComboBox()
        self.technique_combo.addItems(["3D-CRT", "IMRT", "VMAT", "SRS", "SBRT"])
        self.technique_combo.currentIndexChanged.connect(self._on_technique_changed)
        prescription_layout.addRow("Technique:", self.technique_combo)
        
        prescription_group.setLayout(prescription_layout)
        left_layout.addWidget(prescription_group)
        
        # Advanced Planning Options group
        advanced_group = QGroupBox("Advanced Planning")
        advanced_layout = QVBoxLayout()
        
        # MCO button
        self.mco_button = QPushButton("Multi-Criteria Optimization")
        self.mco_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.mco_button.clicked.connect(self._open_mco_dialog)
        advanced_layout.addWidget(self.mco_button)
        
        # Robust optimization button
        self.robust_button = QPushButton("Robust Optimization")
        self.robust_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.robust_button.clicked.connect(self._open_robust_optimization)
        advanced_layout.addWidget(self.robust_button)
        
        advanced_group.setLayout(advanced_layout)
        left_layout.addWidget(advanced_group)
        
        # Add some spacing
        left_layout.addStretch()
        
        # Right splitter for beams and visualization
        self.right_splitter = QSplitter(Qt.Vertical)
        
        # Beams tab widget
        self.beams_tab = QTabWidget()
        
        # 3D-CRT planner
        self.crt_planner = CRTPlanner()
        self.crt_planner.plan_created.connect(self._on_crt_plan_created)
        self.beams_tab.addTab(self.crt_planner, "3D-CRT")
        
        # IMRT planner
        self.imrt_planner = IMRTPlanner()
        self.beams_tab.addTab(self.imrt_planner, "IMRT")
        
        # VMAT planner
        self.vmat_planner = QWidget()  # Placeholder for now
        self.beams_tab.addTab(self.vmat_planner, "VMAT")
        
        # SRS planner
        self.srs_planner = QWidget()  # Placeholder for now
        self.beams_tab.addTab(self.srs_planner, "SRS")
        
        # Disable tabs until they're implemented
        self.beams_tab.setTabEnabled(2, False)  # VMAT
        self.beams_tab.setTabEnabled(3, False)  # SRS
        
        # Add tabs to right splitter
        self.right_splitter.addWidget(self.beams_tab)
        
        # Visualization panel
        self.viz_panel = QTabWidget()
        
        # 2D beam visualization
        self.beam_viz_panel = BeamVisualizationPanel()
        self.beam_viz_panel.beam_added.connect(self._on_beam_added)
        self.beam_viz_panel.beam_modified.connect(self._on_beam_modified)
        self.beam_viz_panel.beam_removed.connect(self._on_beam_removed)
        self.beam_viz_panel.beam_selected.connect(self._on_beam_selected)
        self.beam_viz_panel.calculate_dose_requested.connect(self._on_calculate_beam_dose)
        self.viz_panel.addTab(self.beam_viz_panel, "2D View")
        
        # 3D beam visualization (added if dependencies available)
        self.beam_3d_view = None
        
        # Add 3D visualization tab if dependencies are available
        try:
            import pyvista
            from quangtps.ui.beam_3d_visualization import Beam3DVisualization
            self.beam_3d_view = Beam3DVisualization()
            self.beam_3d_view.beam_selected.connect(self._on_beam_selected)
            self.viz_panel.addTab(self.beam_3d_view, "3D View")
        except ImportError:
            # Add a placeholder tab with a button to install dependencies
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            placeholder_layout.addStretch()
            
            message = QLabel("3D visualization requires additional dependencies.")
            message.setAlignment(Qt.AlignCenter)
            placeholder_layout.addWidget(message)
            
            install_button = QPushButton("Install Dependencies")
            install_button.clicked.connect(self._check_3d_visualization_dependencies)
            placeholder_layout.addWidget(install_button, 0, Qt.AlignCenter)
            
            placeholder_layout.addStretch()
            self.viz_panel.addTab(placeholder, "3D View")
        
        self.right_splitter.addWidget(self.viz_panel)
        
        # Set initial splitter sizes
        self.right_splitter.setSizes([500, 500])
        
        # Add panels to main splitter
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_splitter)
        
        # Set main splitter sizes (left panel gets less space)
        self.main_splitter.setSizes([300, 700])
        
        # Add main splitter to layout
        main_layout.addWidget(self.main_splitter, 1)
        
        # Status bar
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label, 1)
        main_layout.addLayout(status_layout)
        
        # Initialize current plan
        self.current_patient_id = None
        self.current_plan = None
        self.current_structures = None
        
        # Populate patient list
        self._populate_patient_list()
        
        # Add plan comparison action
        self.compare_plans_action = QAction(QIcon("quangtps/ui/icons/new_icons/compare.png"), "Compare Plans", self)
        self.compare_plans_action.setStatusTip("Compare multiple treatment plans")
        self.compare_plans_action.triggered.connect(self._on_compare_plans)
        self.toolbar.addAction(self.compare_plans_action)
    
    def _on_technique_changed(self, index):
        """
        Xử lý khi kỹ thuật xạ trị được thay đổi.
        
        Parameters
        ----------
        index : int
            Chỉ số của kỹ thuật được chọn
        """
        technique = self.technique_combo.currentText()
        
        # Chuyển đến tab tương ứng
        if technique == "3D-CRT":
            # Chuyển đến tab 3D CRT Planner
            for i in range(self.beams_tab.count()):
                if self.beams_tab.tabText(i) == "3D-CRT":
                    self.beams_tab.setCurrentIndex(i)
                    break
        elif technique == "IMRT" or technique == "VMAT":
            # Chuyển đến tab tối ưu hóa
            for i in range(self.beams_tab.count()):
                if self.beams_tab.tabText(i) == "Tối ưu hóa":
                    self.beams_tab.setCurrentIndex(i)
                    break
        else:
            # Mặc định chuyển đến tab chùm tia
            self.beams_tab.setCurrentIndex(0)
    
    def _on_crt_plan_created(self, plan):
        """
        Xử lý khi kế hoạch 3D CRT được tạo.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị được tạo từ CRTPlanner
        """
        try:
            # Cập nhật kế hoạch hiện tại
            self.current_plan = plan
            
            # Cập nhật thông tin kế hoạch
            self.plan_name_edit.setText(plan.name)
            self.plan_desc_edit.setText(f"Kế hoạch 3D CRT với {len(plan.beams)} chùm tia")
            
            # Cập nhật bảng chùm tia
            self._populate_beams_table()
            
            # Hiển thị thông báo
            QMessageBox.information(
                self, 
                "Thông báo", 
                f"Đã tạo kế hoạch 3D CRT '{plan.name}' với {len(plan.beams)} chùm tia"
            )
        except Exception as e:
            logger.error(f"Lỗi khi xử lý kế hoạch 3D CRT: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể áp dụng kế hoạch 3D CRT: {e}")
    
    def set_plan(self, plan):
        """
        Thiết lập kế hoạch hiện tại.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị
        """
        self.current_plan = plan
        
        # Update UI with plan data
        self._populate_plan_data()
        
        # Set the plan in the beam visualization panel
        if hasattr(self, 'beam_viz_panel'):
            self.beam_viz_panel.set_plan(plan)
            
            # If the plan has dose data, update the visualization
            if hasattr(plan, 'dose_grid') and plan.dose_grid is not None:
                self.beam_viz_panel.set_dose_grid(plan.dose_grid)
    
    def set_patient(self, patient_id):
        """
        Thiết lập bệnh nhân hiện tại.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        """
        self.current_patient_id = patient_id
        
        # Load patient data for the beam visualization panel
        if hasattr(self, 'beam_viz_panel'):
            from quangtps.core.services import ServiceManager
            service_manager = ServiceManager()
            patient_service = service_manager.get_service('PatientService')
            
            if patient_service:
                patient_image = patient_service.get_patient_image(patient_id)
                structures = patient_service.get_patient_structures(patient_id)
                
                if patient_image:
                    self.beam_viz_panel.set_patient_data(patient_image, structures)
    
    def set_structures(self, structures):
        """
        Thiết lập danh sách cấu trúc.
        
        Parameters
        ----------
        structures : List
            Danh sách các cấu trúc
        """
        # Cập nhật danh sách cấu trúc cho CRT Planner
        self.crt_planner.set_structures(structures)
    
    def _populate_plan_data(self):
        """Điền thông tin kế hoạch vào giao diện."""
        if not self.current_plan:
            return
            
        # Thông tin cơ bản
        self.plan_name_edit.setText(self.current_plan.name)
        self.plan_desc_edit.setText(self.current_plan.description)
        
        # Thiết lập ngày tạo
        if self.current_plan.created_date:
            qdate = QDate(
                self.current_plan.created_date.year,
                self.current_plan.created_date.month,
                self.current_plan.created_date.day
            )
            self.plan_date_edit.setDate(qdate)
        
        # Thiết lập mục đích và trạng thái
        self.plan_type_combo.setCurrentText(str(self.current_plan.type))
        self.plan_status_combo.setCurrentText(str(self.current_plan.status))
        
        # Thông tin liều lượng
        if self.current_plan.prescription:
            self.rx_dose_edit.setValue(self.current_plan.prescription.total_dose)
            self.rx_fractions_edit.setValue(self.current_plan.prescription.fractions)
        
        # Cập nhật bảng chùm tia
        self._populate_beams_table()
        
        # Cập nhật thuật toán tối ưu hóa
        if self.current_plan.optimization_settings:
            self.opt_algorithm_field.setCurrentText(self.current_plan.optimization_settings.algorithm)
            self.opt_iterations_field.setText(str(self.current_plan.optimization_settings.max_iterations))
            self.opt_convergence_field.setText(str(self.current_plan.optimization_settings.convergence_threshold))
    
    def _populate_beams_table(self):
        """Điền thông tin chùm tia vào bảng."""
        self.beams_table.setRowCount(0)
        
        if not self.current_plan or not self.current_plan.beam_arrangement:
            return
            
        beams = self.current_plan.beam_arrangement.beams
        self.beams_table.setRowCount(len(beams))
        
        for row, beam in enumerate(beams):
            # ID
            id_item = QTableWidgetItem(str(beam.beam_id))
            self.beams_table.setItem(row, 0, id_item)
            
            # Tên
            name_item = QTableWidgetItem(beam.name)
            self.beams_table.setItem(row, 1, name_item)
            
            # Góc gantry
            gantry_item = QTableWidgetItem(f"{beam.gantry_angle:.1f}°")
            self.beams_table.setItem(row, 2, gantry_item)
            
            # Góc collimator
            collimator_item = QTableWidgetItem(f"{beam.collimator_angle:.1f}°")
            self.beams_table.setItem(row, 3, collimator_item)
            
            # Góc bàn
            couch_item = QTableWidgetItem(f"{beam.couch_angle:.1f}°")
            self.beams_table.setItem(row, 4, couch_item)
            
            # MU
            mu_item = QTableWidgetItem(f"{beam.monitor_units:.1f}")
            self.beams_table.setItem(row, 5, mu_item)
            
            # Trạng thái
            status_item = QTableWidgetItem(beam.status)
            self.beams_table.setItem(row, 6, status_item)
    
    def _save_plan(self):
        """Lưu thông tin kế hoạch."""
        # Kiểm tra ID bệnh nhân
        if not self.current_patient_id:
            QMessageBox.warning(
                self, 
                "Chưa chọn bệnh nhân", 
                "Vui lòng chọn một bệnh nhân trước khi lưu kế hoạch."
            )
            return
        
        # Lấy thông tin từ giao diện
        plan_name = self.plan_name_edit.text().strip()
        if not plan_name:
            QMessageBox.warning(
                self, 
                "Thiếu tên kế hoạch", 
                "Vui lòng nhập tên kế hoạch."
            )
            return
        
        # Kiểm tra và lấy các giá trị từ giao diện
        description = self.plan_desc_edit.text().strip()
        plan_date = self.plan_date_edit.date().toPyDate()
        plan_type = self.plan_type_combo.currentText()
        plan_status = self.plan_status_combo.currentText()
        
        # Thông tin liều lượng
        prescribed_dose = self.rx_dose_edit.value()
        fractions = self.rx_fractions_edit.value()
        
        # Tạo hoặc cập nhật kế hoạch
        if self.current_plan:
            # Cập nhật kế hoạch hiện tại
            self.current_plan.name = plan_name
            self.current_plan.description = description
            self.current_plan.created_date = plan_date
            self.current_plan.type = PlanType(plan_type)
            self.current_plan.status = PlanStatus(plan_status)
            
            # Cập nhật đơn thuốc
            if not self.current_plan.prescription:
                self.current_plan.prescription = Prescription()
                
            self.current_plan.prescription.total_dose = prescribed_dose
            self.current_plan.prescription.fractions = fractions
            
            # Lưu kế hoạch vào cơ sở dữ liệu
            try:
                self.plan_db.update_plan(self.current_plan)
                QMessageBox.information(
                    self, 
                    "Lưu kế hoạch", 
                    f"Đã cập nhật kế hoạch {plan_name} thành công."
                )
                self.plan_updated.emit(self.current_plan)
                logger.info(f"Đã cập nhật kế hoạch ID={self.current_plan.plan_id}")
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Lỗi", 
                    f"Không thể cập nhật kế hoạch: {str(e)}"
                )
                logger.error(f"Lỗi cập nhật kế hoạch: {str(e)}")
        else:
            # Tạo kế hoạch mới
            self._create_new_plan(
                plan_name, 
                description, 
                plan_date, 
                plan_type, 
                plan_status, 
                prescribed_dose, 
                fractions
            )
    
    def _create_new_plan(self, name, description, date, plan_type, status, dose, fractions):
        """
        Tạo một kế hoạch mới.
        
        Parameters
        ----------
        name : str
            Tên kế hoạch
        description : str
            Mô tả kế hoạch
        date : datetime.date
            Ngày tạo kế hoạch
        plan_type : str
            Loại kế hoạch (CURATIVE, PALLIATIVE, ...)
        status : str
            Trạng thái kế hoạch (DRAFT, APPROVED, ...)
        dose : float
            Liều chỉ định (Gy)
        fractions : int
            Số phân liều
        """
        try:
            # Tạo đối tượng kế hoạch mới
            new_plan = Plan()
            new_plan.patient_id = self.current_patient_id
            new_plan.name = name
            new_plan.description = description
            new_plan.created_date = date
            new_plan.type = PlanType(plan_type)
            new_plan.status = PlanStatus(status)
            
            # Thiết lập đơn thuốc
            new_prescription = Prescription()
            new_prescription.total_dose = dose
            new_prescription.fractions = fractions
            new_plan.prescription = new_prescription
            
            # Thiết lập thông số tối ưu mặc định
            optimization = OptimizationSettings()
            optimization.algorithm = self.opt_algorithm_field.currentText()
            optimization.max_iterations = int(self.opt_iterations_field.text())
            optimization.convergence_threshold = float(self.opt_convergence_field.text())
            new_plan.optimization_settings = optimization
            
            # Thiết lập kỹ thuật điều trị
            self.current_technique = "3D-CRT"
            new_plan.technique = "3D-CRT"
            
            # Tạo bố trí chùm tia mặc định
            new_plan.beam_arrangement = BeamArrangement()
            
            # Lưu kế hoạch mới vào cơ sở dữ liệu
            plan_id = self.plan_db.create_plan(new_plan)
            
            # Cập nhật ID và thiết lập kế hoạch hiện tại
            new_plan.plan_id = plan_id
            self.current_plan = new_plan
            
            # Thông báo
            QMessageBox.information(
                self, 
                "Tạo kế hoạch", 
                f"Đã tạo kế hoạch {name} thành công."
            )
            
            # Phát tín hiệu cập nhật
            self.plan_created.emit(new_plan)
            logger.info(f"Đã tạo kế hoạch mới ID={plan_id}")
            
            # Cập nhật giao diện
            self._populate_plan_data()
            self.beams_tab.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Lỗi", 
                f"Không thể tạo kế hoạch mới: {str(e)}"
            )
            logger.error(f"Lỗi tạo kế hoạch mới: {str(e)}")
    
    def _clear_plan_data(self):
        """Xóa thông tin kế hoạch khỏi giao diện."""
        # Xóa thông tin kế hoạch
        self.plan_name_edit.clear()
        self.plan_desc_edit.clear()
        self.plan_date_edit.setDate(QDate.currentDate())
        self.plan_type_combo.setCurrentIndex(0)
        self.plan_status_combo.setCurrentIndex(0)
        
        # Xóa thông tin liều lượng
        self.rx_dose_edit.setValue(0)
        self.rx_fractions_edit.setValue(0)
        
        # Xóa bảng ràng buộc
        self.constraints_table.setRowCount(0)
        
        # Xóa bảng chùm tia
        self.beams_table.setRowCount(0)
    
    def _add_constraint(self):
        """Thêm ràng buộc mới."""
        logger.info("Thêm ràng buộc")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _run_optimization(self):
        """Chạy quy trình tối ưu hóa kế hoạch điều trị."""
        # Kiểm tra có kế hoạch hiện tại không
        plan = self._get_current_plan()
        if not plan:
            QMessageBox.warning(
                self,
                "Không thể chạy tối ưu hóa",
                "Vui lòng tạo kế hoạch điều trị trước."
            )
            return

        # Kiểm tra có đủ chùm tia không
        if not plan.beam_arrangement or len(plan.beam_arrangement.beam_setups) == 0:
            QMessageBox.warning(
                self,
                "Không thể chạy tối ưu hóa",
                "Vui lòng thêm ít nhất một chùm tia vào kế hoạch."
            )
            return

        # Kiểm tra có cấu trúc đích và cấu trúc nguy cấp không
        if not plan.targets or not plan.oars:
            QMessageBox.warning(
                self,
                "Không thể chạy tối ưu hóa",
                "Vui lòng thiết lập các cấu trúc đích và cấu trúc nguy cấp trước."
            )
            return

        # Hiển thị thông báo tiến trình
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Tối ưu hóa")
        msg.setText("Đang chạy tối ưu hóa kế hoạch...")
        msg.setStandardButtons(QMessageBox.NoButton)
        
        # Tạo và hiển thị hộp thoại không chặn
        msg.show()
        
        try:
            # Tạo cơ sở Pareto
            num_basis_points = 7  # Số điểm cơ sở
            
            # Cập nhật tiến trình
            progress_dialog = QProgressDialog("Đang tạo cơ sở mặt Pareto...", "Hủy", 0, 100, self)
            progress_dialog.setWindowTitle("Tối ưu hóa")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setValue(0)
            progress_dialog.show()
            
            # Cập nhật tiến trình
            for i in range(1, 6):
                progress_dialog.setValue(i * 20)
                QApplication.processEvents()
                
                if progress_dialog.wasCanceled():
                    raise InterruptedError("Người dùng đã hủy quá trình tạo cơ sở Pareto")
            
            # Tạo cơ sở mặt Pareto
            pareto_basis = self.run_mco_optimization()
            
            progress_dialog.setValue(100)
            
            # Hiển thị kết quả
            QMessageBox.information(
                self,
                "Tối ưu hóa hoàn tất",
                f"Tối ưu hóa kế hoạch đã hoàn tất.\n\n"
                f"Số lặp: {pareto_basis.num_iterations}\n"
                f"Thời gian: {pareto_basis.elapsed_time:.2f} giây\n"
                f"Cải thiện: {pareto_basis.get_improvement_percentage():.2f}%"
            )
            
            # Cập nhật kế hoạch
            self._update_plan()
            
            # Log
            logger.info(f"Tối ưu hóa kế hoạch {plan.name} hoàn tất sau {pareto_basis.num_iterations} lần lặp")
            
        except Exception as e:
            # Xử lý lỗi
            QMessageBox.critical(
                self,
                "Lỗi tối ưu hóa",
                f"Đã xảy ra lỗi trong quá trình tối ưu hóa: {str(e)}"
            )
            logger.error(f"Lỗi tối ưu hóa: {str(e)}", exc_info=True)
        finally:
            # Đóng hộp thoại tiến trình
            msg.close()
            
    def _update_plan_display(self):
        """Update the display with current plan data."""
        if self.current_plan is None:
            # Clear everything
            self._clear_plan_data()
            return
        
        # Update basic plan information
        self.plan_name_edit.setText(self.current_plan.name)
        self.plan_desc_edit.setText(self.current_plan.description)
        
        if hasattr(self.current_plan, 'date') and self.current_plan.date:
            try:
                plan_date = QDate.fromString(self.current_plan.date, "yyyy-MM-dd")
                self.plan_date_edit.setDate(plan_date)
            except:
                self.plan_date_edit.setDate(QDate.currentDate())
        
        # Set plan type and status if available
        if hasattr(self.current_plan, 'type'):
            index = self.plan_type_combo.findText(self.current_plan.type)
            if index >= 0:
                self.plan_type_combo.setCurrentIndex(index)
        
        if hasattr(self.current_plan, 'status'):
            index = self.plan_status_combo.findText(self.current_plan.status)
            if index >= 0:
                self.plan_status_combo.setCurrentIndex(index)
        
        # Set prescription if available
        if hasattr(self.current_plan, 'prescription'):
            rx = self.current_plan.prescription
            if hasattr(rx, 'dose'):
                self.rx_dose_edit.setValue(rx.dose)
            if hasattr(rx, 'fractions'):
                self.rx_fractions_edit.setValue(rx.fractions)
        
        # Set technique if available
        if hasattr(self.current_plan, 'technique'):
            index = self.technique_combo.findText(self.current_plan.technique)
            if index >= 0:
                self.technique_combo.setCurrentIndex(index)
        
        # Update beam visualization
        if hasattr(self.current_plan, 'beams'):
            self.beam_viz_panel.set_plan(self.current_plan)
            
            # Update 3D visualization if available
            if self.beam_3d_view is not None:
                if self.current_structures:
                    self.beam_3d_view.set_patient_data(self.current_plan.image, self.current_structures)
                else:
                    self.beam_3d_view.set_patient_data(self.current_plan.image)
                self.beam_3d_view.set_plan(self.current_plan)
                if hasattr(self.current_plan, 'dose_grid') and self.current_plan.dose_grid is not None:
                    self.beam_3d_view.set_dose_grid(self.current_plan.dose_grid)
        
        # Update the appropriate technique planner
        technique = self.current_plan.technique if hasattr(self.current_plan, 'technique') else "3D-CRT"
        
        if technique == "3D-CRT":
            self.beams_tab.setCurrentIndex(0)
            self.crt_planner.set_plan(self.current_plan)
        elif technique == "IMRT":
            self.beams_tab.setCurrentIndex(1)
            self.imrt_planner.set_plan(self.current_plan)
        elif technique == "VMAT":
            # Future implementation
            pass
        elif technique == "SRS" or technique == "SBRT":
            # Future implementation
            pass
    
    def run_mco_optimization(self):
        """Run multi-criteria optimization (MCO)."""
        try:
            if not self.current_plan:
                QMessageBox.warning(self, "Warning", "Please create or open a plan before optimization.")
                return
                
            # Show progress dialog
            progress_dialog = QProgressDialog("Preparing multi-criteria optimization...", "Cancel", 0, 100, self)
            progress_dialog.setWindowTitle("MCO Optimization")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()
            QApplication.processEvents()
            
            # Prepare objectives
            progress_dialog.setValue(10)
            QApplication.processEvents()
            
            # Check if objectives exist
            if not hasattr(self.current_plan, 'objectives') or not self.current_plan.objectives:
                progress_dialog.close()
                QMessageBox.warning(self, "Warning", "Plan must have objectives defined for MCO.")
                return
                
            # Check number of objectives
            if len(self.current_plan.objectives) < 2:
                progress_dialog.close()
                QMessageBox.warning(self, "Warning", "At least 2 objective functions are required for MCO.")
                return
            
            # Set up MCO engine
            progress_dialog.setValue(30)
            progress_dialog.setLabelText("Setting up MCO engine...")
            QApplication.processEvents()
            
            mco_engine = MCOEngine(method=MCOMethod.WEIGHTED_SUM)
            
            # Add objectives to MCO engine
            for obj in self.current_plan.objectives:
                # Determine if this is a target or OAR objective
                obj_type = "Other"
                if hasattr(obj, 'structure') and obj.structure:
                    if hasattr(obj.structure, 'type'):
                        obj_type = obj.structure.type
                
                # Set appropriate range and default weight based on objective type
                if "Target" in obj_type:
                    weight_range = (0.1, 1.0)
                    default_weight = 0.7
                elif "OAR" in obj_type:
                    weight_range = (0.0, 0.9)
                    default_weight = 0.3
                else:
                    weight_range = (0.0, 1.0)
                    default_weight = 0.5
                
                # Add to MCO engine
                mco_engine.add_objective(
                    objective=obj,
                    name=obj.name if hasattr(obj, 'name') and obj.name else f"Objective {id(obj)}",
                    weight_range=weight_range,
                    current_weight=default_weight,
                    show_in_navigation=True
                )
            
            # Add constraints
            progress_dialog.setValue(50)
            progress_dialog.setLabelText("Adding constraints to MCO...")
            QApplication.processEvents()
            
            if hasattr(self.current_plan, 'constraints') and self.current_plan.constraints:
                for constraint in self.current_plan.constraints:
                    mco_engine.add_constraint(constraint)
            
            # Initialize Pareto space
            progress_dialog.setValue(70)
            progress_dialog.setLabelText("Initializing Pareto space...")
            QApplication.processEvents()
            
            # Check for cancellation
            if progress_dialog.wasCanceled():
                return
            
            # Get dose grid and structures
            dose_grid = self.current_plan.dose_grid if hasattr(self.current_plan, 'dose_grid') else None
            structures = self.current_plan.structures if hasattr(self.current_plan, 'structures') else None
            
            # Set initial state if dose grid and structures are available
            if dose_grid and structures:
                mco_engine.set_initial_state(dose_grid, structures)
            
            progress_dialog.setValue(90)
            QApplication.processEvents()
            
            # Store MCO engine in the plan
            self.current_plan.mco_engine = mco_engine
            
            # Close progress dialog
            progress_dialog.close()
            
            # Show MCO navigation dialog
            self._open_mco_dialog()
            
        except Exception as e:
            logger.error(f"Error during MCO optimization: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error during multi-criteria optimization: {str(e)}")
    
    def accept_mco_trade(self, trade):
        """Accept an MCO trade and apply it to the current plan."""
        if not self.current_plan:
            logger.error("Cannot accept MCO trade: No current plan")
            return
            
        try:
            logger.info(f"Applying MCO trade with {len(trade.objective_values)} objective values")
            
            # Update the plan's dose grid with the one from the trade
            if trade.dose_grid is not None:
                self.current_plan.dose_grid = trade.dose_grid
                logger.info("Updated plan dose grid from MCO trade")
            
            # Update objective weights in the plan to match the trade
            if hasattr(self.current_plan, 'objectives') and self.current_plan.objectives:
                for obj in self.current_plan.objectives:
                    obj_name = obj.name if hasattr(obj, 'name') and obj.name else f"Objective {id(obj)}"
                    if obj_name in trade.weights:
                        obj.weight = trade.weights[obj_name]
                        logger.info(f"Updated weight for objective {obj_name}: {obj.weight}")
            
            # Store the DVH data from the trade
            if trade.dvh_data:
                self.current_plan.dvh_data = trade.dvh_data
                logger.info("Updated plan DVH data from MCO trade")
            
            # Store the trade itself for reference
            self.current_plan.selected_mco_trade = trade
            
            # Update evaluation metrics
            self.current_plan.evaluation_metrics = {}
            for obj_name, obj_value in trade.objective_values.items():
                self.current_plan.evaluation_metrics[f"MCO_{obj_name}"] = obj_value
            
            # Flag plan as modified
            self.current_plan.modified = True
            
            logger.info("Successfully applied MCO trade to current plan")
            
        except Exception as e:
            logger.error(f"Error applying MCO trade: {str(e)}")
            QMessageBox.critical(self, "Error", f"Could not apply MCO trade: {str(e)}")

    def run_kbp_optimization(self):
        """Thực hiện tối ưu hóa dựa trên kiến thức."""
        # Kiểm tra xem có kế hoạch hiện tại không
        current_plan = self._get_current_plan()
        if not current_plan:
            QMessageBox.warning(self, "Cảnh báo", "Bạn cần tạo kế hoạch trước khi tối ưu.")
            return
        
        # Lấy liều kê đơn
        try:
            dose = float(self.rx_dose_edit.value())
            if dose <= 0:
                QMessageBox.warning(self, "Cảnh báo", "Liều kê đơn không hợp lệ.")
                return
        except ValueError:
            QMessageBox.warning(self, "Cảnh báo", "Liều kê đơn không hợp lệ.")
            return
        
        try:
            # Mở dialog KBP
            from quangtps.ui.kbp_dialog import KBPDialog
            
            kbp_dialog = KBPDialog(
                patient_id=current_plan["patient_id"],
                structure_set_id=current_plan["structure_set_id"],
                prescription_dose=dose,
                parent=self
            )
            
            # Kết nối tín hiệu
            kbp_dialog.recommendationApplied.connect(self.apply_kbp_recommendation)
            
            # Hiển thị dialog
            kbp_dialog.exec_()
            
        except ImportError as e:
            QMessageBox.critical(self, "Lỗi", f"Module tối ưu hóa KBP không khả dụng: {str(e)}")
            return
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể mở dialog KBP: {str(e)}")
            return

    def apply_kbp_recommendation(self, recommendation, objectives, constraints):
        """
        Áp dụng đề xuất KBP vào kế hoạch.
        
        Args:
            recommendation: Đề xuất KBP
            objectives: Tập hợp mục tiêu tối ưu
            constraints: Tập hợp ràng buộc
        """
        # Lấy kế hoạch hiện tại
        current_plan = self._get_current_plan()
        if not current_plan:
            QMessageBox.warning(self, "Cảnh báo", "Không tìm thấy kế hoạch hiện tại.")
            return
        
        try:
            # Cập nhật kế hoạch với các mục tiêu và ràng buộc mới
            self.plan_db.update_plan_optimization(
                current_plan["id"],
                objectives=objectives,
                constraints=constraints
            )
            
            # Cập nhật hiển thị
            self._populate_plan_data()
            
            # Thông báo thành công
            QMessageBox.information(
                self, "Thành công", 
                "Đã áp dụng các tham số tối ưu từ mô hình KBP vào kế hoạch."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể áp dụng đề xuất KBP: {str(e)}")
            logger.exception("Error applying KBP recommendation")

    def _open_mco_dialog(self):
        """Open the MCO navigation dialog."""
        if not self.current_plan:
            QMessageBox.warning(self, "No Plan Selected", "Please select a plan first.")
            return
        
        # Check for optimization objectives
        if not hasattr(self.current_plan, 'objectives') or not self.current_plan.objectives:
            QMessageBox.warning(self, "Missing Objectives", 
                              "The plan must have optimization objectives defined before using MCO.")
            return
        
        # Check if we have calculated dose or structures available
        has_dose = hasattr(self.current_plan, 'dose_grid') and self.current_plan.dose_grid is not None
        has_structures = hasattr(self.current_plan, 'structures') and self.current_plan.structures is not None
        
        if not has_dose or not has_structures:
            response = QMessageBox.question(
                self,
                "Missing Data",
                "MCO works best with a calculated dose grid and defined structures. Do you want to proceed anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if response == QMessageBox.No:
                return
        
        try:
            # Create MCO engine if it doesn't exist or reset it
            if not hasattr(self.current_plan, 'mco_engine') or self.current_plan.mco_engine is None:
                # Set up MCO engine with the current plan's objectives
                mco_engine = MCOEngine(method=MCOMethod.WEIGHTED_SUM)
                
                # Add all objectives from the plan
                for obj in self.current_plan.objectives:
                    # Determine if this is a target or OAR objective
                    obj_type = "Other"
                    if hasattr(obj, 'structure') and obj.structure:
                        if hasattr(obj.structure, 'type'):
                            obj_type = obj.structure.type
                    
                    # Set appropriate range and default weight based on objective type
                    if "Target" in obj_type:
                        weight_range = (0.1, 1.0)
                        default_weight = 0.7
                    elif "OAR" in obj_type:
                        weight_range = (0.0, 0.9)
                        default_weight = 0.3
                    else:
                        weight_range = (0.0, 1.0)
                        default_weight = 0.5
                    
                    # Add to MCO engine
                    mco_engine.add_objective(
                        objective=obj,
                        name=obj.name if hasattr(obj, 'name') and obj.name else f"Objective {id(obj)}",
                        weight_range=weight_range,
                        current_weight=default_weight,
                        show_in_navigation=True
                    )
                
                # Add constraints if available
                if hasattr(self.current_plan, 'constraints'):
                    for constraint in self.current_plan.constraints:
                        mco_engine.add_constraint(constraint)
                
                # Set initial state if dose grid and structures are available
                if has_dose and has_structures:
                    mco_engine.set_initial_state(
                        dose_grid=self.current_plan.dose_grid,
                        structures=self.current_plan.structures
                    )
                
                # Store the engine in the plan for future use
                self.current_plan.mco_engine = mco_engine
            
            # Create and show the MCO navigation dialog
            dialog = MCONavigationDialog(self.current_plan.mco_engine, self)
            
            # Connect the signal to handle when a trade is accepted
            dialog.tradeAccepted.connect(self.accept_mco_trade)
            
            # Show the dialog
            result = dialog.exec_()
            
            # Handle the result
            if result == QDialog.Accepted:
                # The trade has already been applied via the signal connection
                # Update UI to reflect changes
                self._update_plan_display()
                QMessageBox.information(
                    self,
                    "Plan Updated",
                    "The plan has been updated with the selected MCO trade-off."
                )
                
                # Emit signal that plan was updated
                self.plan_updated.emit(self.current_plan)
                
        except Exception as e:
            logger.error(f"Error opening MCO dialog: {str(e)}")
            QMessageBox.critical(
                self,
                "MCO Error",
                f"An error occurred while setting up MCO: {str(e)}"
            )

    def _check_3d_visualization_dependencies(self):
        """Check if 3D visualization dependencies are available and offer to install them."""
        from quangtps.ui.dependency_installer import check_and_install_feature_dependencies
        
        # Check dependencies only if 3D visualization is requested
        if self.beam_3d_view is not None:
            return True
            
        # Check if dependencies are available
        dependencies_available = importlib.util.find_spec("pyvista") is not None and \
                                importlib.util.find_spec("pyvistaqt") is not None and \
                                importlib.util.find_spec("vtk") is not None
        
        if not dependencies_available:
            # Ask user if they want to install dependencies
            response = QMessageBox.question(
                self,
                "3D Visualization Dependencies",
                "The 3D beam visualization feature requires additional packages. "
                "Would you like to install them now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if response == QMessageBox.Yes:
                # Install dependencies
                success = check_and_install_feature_dependencies("3d_visualization", self)
                if success:
                    QMessageBox.information(
                        self,
                        "Installation Complete",
                        "Dependencies installed successfully. Please restart the application "
                        "to use the 3D visualization feature."
                    )
                return success
            
            return False
        
        return True

    def _on_beam_added(self, beam):
        """Handle the addition of a new beam."""
        if self.current_plan:
            # Update the plan with the new beam
            if not hasattr(self.current_plan, 'beams'):
                self.current_plan.beams = []
            
            # Make sure the beam is not already in the plan
            if beam not in self.current_plan.beams:
                # Add beam to the plan
                self.current_plan.add_beam(beam)
            
            # Update the UI
            self._update_plan_display()
            
            logger.info(f"Beam added to plan: {beam.name}")

    def _on_beam_modified(self, beam):
        """Handle modifications to a beam."""
        if self.current_plan:
            # Update the plan with the modified beam
            self._update_plan_display()
            
            logger.info(f"Beam modified: {beam.name}")

    def _on_beam_removed(self, beam):
        """Handle removal of a beam."""
        if self.current_plan and hasattr(self.current_plan, 'beams'):
            # Update the plan
            self._update_plan_display()
            
            logger.info(f"Beam removed: {beam.name}")

    def _on_beam_selected(self, beam):
        """Handle selection of a beam."""
        # This method can be used to update other UI elements based on beam selection
        logger.debug(f"Beam selected: {beam.name}")

    def _on_calculate_beam_dose(self, beam):
        """Handle request to calculate dose for a beam."""
        if not self.current_plan:
            QMessageBox.warning(
                self, 
                "No Plan", 
                "Please create or load a plan before calculating dose."
            )
            return
        
        # Get the dose calculation service
        from quangtps.core.services import ServiceManager
        service_manager = ServiceManager()
        dose_service = service_manager.get_service('DoseCalculationService')
        
        if not dose_service:
            QMessageBox.warning(
                self, 
                "Service Unavailable", 
                "Dose calculation service is not available."
            )
            return
        
        # Show progress dialog
        progress = QProgressDialog("Calculating dose...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            # Calculate dose
            progress.setValue(10)
            QApplication.processEvents()
            
            # Get patient data
            patient_service = service_manager.get_service('PatientService')
            if not patient_service:
                QMessageBox.warning(self, "Service Unavailable", "Patient service is not available.")
                return
                
            ct_image = patient_service.get_patient_image(self.current_patient_id)
            if not ct_image:
                QMessageBox.warning(self, "No Image Data", "Patient CT image is not available.")
                return
            
            progress.setValue(25)
            QApplication.processEvents()
            
            # Calculate dose for the beam
            dose_grid = dose_service.calculate_beam_dose(beam, ct_image)
            
            progress.setValue(90)
            QApplication.processEvents()
            
            # Update the current plan with the dose
            if dose_grid:
                self.current_plan.set_dose_grid(dose_grid)
                
                # Update visualization
                self.beam_viz_panel.set_dose_grid(dose_grid)
                
                QMessageBox.information(self, "Dose Calculation", "Beam dose calculation completed successfully.")
            else:
                QMessageBox.warning(self, "Calculation Failed", "Failed to calculate dose for the beam.")
            
            progress.setValue(100)
            
        except Exception as e:
            logger.error(f"Error calculating dose: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred during dose calculation: {str(e)}")
        finally:
            progress.close()

    def _open_robust_optimization(self):
        """Open the robust optimization dialog."""
        if not self.current_plan:
            QMessageBox.warning(self, "No Plan Selected", "Please select a plan first.")
            return
        
        # Check if dose has been calculated
        has_dose = False
        if hasattr(self.current_plan, 'dose_grid') and self.current_plan.dose_grid is not None:
            has_dose = True
        
        if not has_dose:
            response = QMessageBox.question(
                self,
                "No Dose Grid",
                "Robust optimization works best with a calculated dose grid. Do you want to proceed anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if response == QMessageBox.No:
                return
        
        # Get structures from patient
        structures = {}
        if hasattr(self.current_plan, 'patient') and self.current_plan.patient is not None:
            if hasattr(self.current_plan.patient, 'structures'):
                structures = self.current_plan.patient.structures
        
        # Show the robust optimization dialog
        dose_grid = self.current_plan.dose_grid if has_dose else None
        result = show_robust_optimization_dialog(self.current_plan, structures, dose_grid, self)
        
        # Handle the result
        if result == QDialog.Accepted:
            QMessageBox.information(
                self,
                "Plan Updated",
                "The plan has been robustly optimized. Please calculate the dose again to see the results."
            )
            
            # Update the plan
            self.plan_updated.emit(self.current_plan)

    def _on_compare_plans(self):
        """Open the plan comparison dialog."""
        # Check if there's an active plan
        if not self.current_plan:
            QMessageBox.warning(self, "Compare Plans", "Please select a plan to use as reference.")
            return
        
        # Import here to avoid circular imports
        from quangtps.ui.plan_comparison_dialog import PlanComparisonDialog
        
        # Create and show the dialog
        dialog = PlanComparisonDialog(self.current_plan, self)
        dialog.exec_()

    def _create_toolbar(self):
        # Create toolbar
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(32, 32))
        
        # Add plan management actions
        self.new_plan_action = QAction(QIcon("quangtps/ui/icons/new_plan.png"), "New Plan", self)
        self.new_plan_action.setStatusTip("Create a new treatment plan")
        self.new_plan_action.triggered.connect(self._on_new_plan)
        self.toolbar.addAction(self.new_plan_action)
        
        self.open_plan_action = QAction(QIcon("quangtps/ui/icons/open_plan.png"), "Open Plan", self)
        self.open_plan_action.setStatusTip("Open an existing treatment plan")
        self.open_plan_action.triggered.connect(self._on_open_plan)
        self.toolbar.addAction(self.open_plan_action)
        
        self.save_plan_action = QAction(QIcon("quangtps/ui/icons/save_plan.png"), "Save Plan", self)
        self.save_plan_action.setStatusTip("Save the current treatment plan")
        self.save_plan_action.triggered.connect(self._on_save_plan)
        self.toolbar.addAction(self.save_plan_action)
        
        self.toolbar.addSeparator()
        
        # Add calculation actions
        self.calculate_dose_action = QAction(QIcon("quangtps/ui/icons/calculate.png"), "Calculate Dose", self)
        self.calculate_dose_action.setStatusTip("Calculate dose for the current plan")
        self.calculate_dose_action.triggered.connect(self._on_calculate_dose)
        self.toolbar.addAction(self.calculate_dose_action)
        
        self.toolbar.addSeparator()
        
        # Add evaluate actions
        self.evaluate_plan_action = QAction(QIcon("quangtps/ui/icons/evaluate.png"), "Evaluate Plan", self)
        self.evaluate_plan_action.setStatusTip("Evaluate the current plan")
        self.evaluate_plan_action.triggered.connect(self._on_evaluate_plan)
        self.toolbar.addAction(self.evaluate_plan_action)
        
        # Add plan comparison action
        self.compare_plans_action = QAction(QIcon("quangtps/ui/comparison.svg"), "Compare Plans", self)
        self.compare_plans_action.setStatusTip("Compare multiple treatment plans")
        self.compare_plans_action.triggered.connect(self._on_compare_plans)
        self.toolbar.addAction(self.compare_plans_action)
        
        self.toolbar.addSeparator()
        
        # Add some vertical space above the toolbar
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        
        return layout
