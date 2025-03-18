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
    QFormLayout, QTextEdit, QDateEdit, QDialog, QButtonGroup
)

from quangtps.planning.plan import Plan, PlanStatus, PlanType
from quangtps.planning.beam import BeamArrangement
from quangtps.database.plan_db import PlanDB
from quangtps.ui.dialogs.beam_dialog import BeamDialog
from quangtps.planning.prescription import Prescription
from quangtps.planning.optimization import OptimizationSettings
from quangtps.treatment.beams.beam import Beam

logger = logging.getLogger(__name__)


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
        
        # Thiết lập giao diện
        self._init_ui()
        
        logger.info("Khởi tạo tab lập kế hoạch hoàn tất")
        
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QHBoxLayout(self)
        
        # Splitter chính
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # Panel bên trái (cấu trúc và thông tin kế hoạch)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Tab widget cho panel bên trái
        self.left_tabs = QTabWidget()
        self.left_layout.addWidget(self.left_tabs)
        
        # Tab kế hoạch
        self.plan_info_widget = QWidget()
        self.plan_info_layout = QVBoxLayout(self.plan_info_widget)
        
        # Nhóm thông tin kế hoạch
        self.plan_group = QGroupBox("Thông tin kế hoạch")
        self.plan_form = QFormLayout(self.plan_group)
        
        self.plan_name_field = QLineEdit()
        self.plan_form.addRow("Tên kế hoạch:", self.plan_name_field)
        
        self.plan_description_field = QTextEdit()
        self.plan_description_field.setMaximumHeight(100)
        self.plan_form.addRow("Mô tả:", self.plan_description_field)
        
        self.plan_date_field = QDateEdit()
        self.plan_date_field.setDisplayFormat("dd/MM/yyyy")
        self.plan_date_field.setCalendarPopup(True)
        self.plan_date_field.setDate(QDate.currentDate())
        self.plan_form.addRow("Ngày tạo:", self.plan_date_field)
        
        self.plan_intent_field = QComboBox()
        self.plan_intent_field.addItems(["Điều trị triệt căn", "Điều trị triệu chứng", "Điều trị bổ trợ", "Khác"])
        self.plan_form.addRow("Mục đích:", self.plan_intent_field)
        
        self.plan_status_field = QComboBox()
        self.plan_status_field.addItems(["Đang dự thảo", "Đang xem xét", "Đã phê duyệt", "Hoàn thành", "Hủy bỏ"])
        self.plan_form.addRow("Trạng thái:", self.plan_status_field)
        
        self.plan_info_layout.addWidget(self.plan_group)
        
        # Nhóm liều lượng
        self.dose_group = QGroupBox("Liều lượng")
        self.dose_form = QFormLayout(self.dose_group)
        
        self.prescribed_dose_field = QLineEdit()
        self.dose_form.addRow("Liều chỉ định:", self.prescribed_dose_field)
        
        self.fractions_field = QLineEdit()
        self.dose_form.addRow("Số phân liều:", self.fractions_field)
        
        self.dose_per_fraction_field = QLineEdit()
        self.dose_form.addRow("Liều/phân liều:", self.dose_per_fraction_field)
        
        self.plan_info_layout.addWidget(self.dose_group)
        
        # Nút lưu kế hoạch
        self.save_plan_button = QPushButton("Lưu kế hoạch")
        self.save_plan_button.clicked.connect(self._save_plan)
        self.plan_info_layout.addWidget(self.save_plan_button, alignment=Qt.AlignRight)
        
        # Thêm tab kế hoạch
        self.left_tabs.addTab(self.plan_info_widget, "Kế hoạch")
        
        # Tab cấu trúc
        self.structures_widget = QWidget()
        self.structures_layout = QVBoxLayout(self.structures_widget)
        
        # Hiển thị cấu trúc
        self.structure_view = QLabel("Cấu trúc")
        self.structures_layout.addWidget(self.structure_view)
        
        # Thêm tab cấu trúc
        self.left_tabs.addTab(self.structures_widget, "Cấu trúc")
        
        # Tab ràng buộc
        self.constraints_widget = QWidget()
        self.constraints_layout = QVBoxLayout(self.constraints_widget)
        
        # Bảng ràng buộc
        self.constraints_table = QTableWidget(0, 4)
        self.constraints_table.setHorizontalHeaderLabels(["Cấu trúc", "Loại", "Giá trị", "Mức độ ưu tiên"])
        self.constraints_table.horizontalHeader().setStretchLastSection(True)
        self.constraints_layout.addWidget(self.constraints_table)
        
        # Nút thêm ràng buộc
        self.add_constraint_button = QPushButton("Thêm ràng buộc")
        self.add_constraint_button.clicked.connect(self._add_constraint)
        self.constraints_layout.addWidget(self.add_constraint_button, alignment=Qt.AlignRight)
        
        # Thêm tab ràng buộc
        self.left_tabs.addTab(self.constraints_widget, "Ràng buộc")
        
        # Panel bên phải (thiết lập kỹ thuật và chùm tia)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # Tab widget cho panel bên phải
        self.right_tabs = QTabWidget()
        self.right_layout.addWidget(self.right_tabs)
        
        # Tab kỹ thuật
        self.technique_widget = QWidget()
        self.technique_layout = QVBoxLayout(self.technique_widget)
        
        # Nhóm lựa chọn kỹ thuật
        self.technique_group = QGroupBox("Kỹ thuật xạ trị")
        self.technique_button_layout = QVBoxLayout(self.technique_group)
        
        # Radio buttons cho các kỹ thuật
        self.technique_button_group = QButtonGroup(self)
        
        self.dcat_button = QRadioButton("DCAT - Dynamic Conformal Arc Therapy")
        self.technique_button_group.addButton(self.dcat_button)
        self.technique_button_layout.addWidget(self.dcat_button)
        
        self.imrt_button = QRadioButton("IMRT - Intensity Modulated Radiation Therapy")
        self.technique_button_group.addButton(self.imrt_button)
        self.technique_button_layout.addWidget(self.imrt_button)
        
        self.vmat_button = QRadioButton("VMAT - Volumetric Modulated Arc Therapy")
        self.technique_button_group.addButton(self.vmat_button)
        self.technique_button_layout.addWidget(self.vmat_button)
        
        self.srs_button = QRadioButton("SRS - Stereotactic Radiosurgery")
        self.technique_button_group.addButton(self.srs_button)
        self.technique_button_layout.addWidget(self.srs_button)
        
        self.sbrt_button = QRadioButton("SBRT - Stereotactic Body Radiation Therapy")
        self.technique_button_group.addButton(self.sbrt_button)
        self.technique_button_layout.addWidget(self.sbrt_button)
        
        # Kết nối sự kiện thay đổi lựa chọn
        self.technique_button_group.buttonClicked.connect(self._technique_selected)
        
        self.technique_layout.addWidget(self.technique_group)
        
        # Nhóm tính toán độ phù hợp
        self.suitability_group = QGroupBox("Độ phù hợp của kỹ thuật")
        self.suitability_layout = QVBoxLayout(self.suitability_group)
        
        # Label hiển thị mức độ phù hợp
        self.suitability_label = QLabel("Vui lòng nhập thông tin bệnh nhân và kế hoạch để tính toán độ phù hợp")
        self.suitability_label.setWordWrap(True)
        self.suitability_layout.addWidget(self.suitability_label)
        
        # Nút tính toán độ phù hợp
        self.calculate_suitability_button = QPushButton("Tính toán độ phù hợp")
        self.calculate_suitability_button.clicked.connect(self._calculate_technique_suitability)
        self.suitability_layout.addWidget(self.calculate_suitability_button)
        
        self.technique_layout.addWidget(self.suitability_group)
        
        # Thêm tab kỹ thuật
        self.right_tabs.addTab(self.technique_widget, "Kỹ thuật")
        
        # Tab chùm tia
        self.beams_widget = QWidget()
        self.beams_layout = QVBoxLayout(self.beams_widget)
        
        # Bảng chùm tia
        self.beams_table = QTableWidget(0, 7)
        self.beams_table.setHorizontalHeaderLabels(["ID", "Tên", "Góc gantry", "Góc collimator", "Góc bàn", "MU", "Trạng thái"])
        self.beams_table.horizontalHeader().setStretchLastSection(True)
        self.beams_layout.addWidget(self.beams_table)
        
        # Nút thêm chùm tia
        self.beam_buttons_layout = QHBoxLayout()
        
        self.add_beam_button = QPushButton("Thêm chùm tia")
        self.add_beam_button.clicked.connect(self._add_beam)
        self.beam_buttons_layout.addWidget(self.add_beam_button)
        
        self.add_arc_button = QPushButton("Thêm cung")
        self.add_arc_button.clicked.connect(self._add_arc)
        self.beam_buttons_layout.addWidget(self.add_arc_button)
        
        self.edit_beam_button = QPushButton("Chỉnh sửa")
        self.edit_beam_button.clicked.connect(self._edit_beam)
        self.beam_buttons_layout.addWidget(self.edit_beam_button)
        
        self.delete_beam_button = QPushButton("Xóa")
        self.delete_beam_button.clicked.connect(self._delete_beam)
        self.beam_buttons_layout.addWidget(self.delete_beam_button)
        
        self.beams_layout.addLayout(self.beam_buttons_layout)
        
        # Thêm tab chùm tia
        self.right_tabs.addTab(self.beams_widget, "Chùm tia")
        
        # Tab tối ưu hóa
        self.optimization_widget = QWidget()
        self.optimization_layout = QVBoxLayout(self.optimization_widget)
        
        # Nhóm thiết lập tối ưu hóa
        self.optimization_group = QGroupBox("Thiết lập tối ưu hóa")
        self.optimization_form = QFormLayout(self.optimization_group)
        
        self.opt_algorithm_field = QComboBox()
        self.opt_algorithm_field.addItems(["Simulated Annealing", "Genetic Algorithm", "Gradient Descent", "IPOPT"])
        self.optimization_form.addRow("Thuật toán:", self.opt_algorithm_field)
        
        self.opt_iterations_field = QLineEdit()
        self.optimization_form.addRow("Số lần lặp:", self.opt_iterations_field)
        
        self.opt_convergence_field = QLineEdit()
        self.optimization_form.addRow("Ngưỡng hội tụ:", self.opt_convergence_field)
        
        self.optimization_layout.addWidget(self.optimization_group)
        
        # Nút tối ưu hóa
        self.run_optimization_button = QPushButton("Chạy tối ưu hóa")
        self.run_optimization_button.clicked.connect(self._run_optimization)
        self.optimization_layout.addWidget(self.run_optimization_button, alignment=Qt.AlignRight)
        
        # Thêm tab tối ưu hóa
        self.right_tabs.addTab(self.optimization_widget, "Tối ưu hóa")
        
        # Thêm các panel vào splitter
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_panel)
        
        # Thiết lập kích thước ban đầu
        self.main_splitter.setSizes([400, 600])
        
        # Vô hiệu hóa các tab liên quan đến kỹ thuật khi chưa có kế hoạch
        self.right_tabs.setEnabled(False)
    
    def set_plan(self, plan):
        """Thiết lập kế hoạch hiện tại và cập nhật giao diện.
        
        Args:
            plan: Đối tượng kế hoạch
        """
        self.current_plan = plan
        if plan:
            self._populate_plan_data()
            self.right_tabs.setEnabled(True)
        else:
            self._clear_plan_data()
            self.right_tabs.setEnabled(False)
    
    def set_patient(self, patient_id):
        """
        Thiết lập ID bệnh nhân hiện tại.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        """
        self.current_patient_id = patient_id
        # Kích hoạt nút tạo kế hoạch nếu có ID bệnh nhân
        if patient_id:
            self.save_plan_button.setEnabled(True)
        else:
            self.save_plan_button.setEnabled(False)
    
    def _populate_plan_data(self):
        """Điền thông tin kế hoạch vào giao diện."""
        if not self.current_plan:
            return
            
        # Thông tin cơ bản
        self.plan_name_field.setText(self.current_plan.plan_name)
        self.plan_description_field.setText(self.current_plan.description)
        
        # Thiết lập ngày tạo
        if self.current_plan.created_date:
            qdate = QDate(
                self.current_plan.created_date.year,
                self.current_plan.created_date.month,
                self.current_plan.created_date.day
            )
            self.plan_date_field.setDate(qdate)
        
        # Thiết lập mục đích và trạng thái
        self.plan_intent_field.setCurrentText(str(self.current_plan.plan_type))
        self.plan_status_field.setCurrentText(str(self.current_plan.status))
        
        # Thông tin liều lượng
        if self.current_plan.prescription:
            self.prescribed_dose_field.setText(str(self.current_plan.prescription.total_dose))
            self.fractions_field.setText(str(self.current_plan.prescription.fractions))
        
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
        plan_name = self.plan_name_field.text().strip()
        if not plan_name:
            QMessageBox.warning(
                self, 
                "Thiếu tên kế hoạch", 
                "Vui lòng nhập tên kế hoạch."
            )
            return
        
        # Kiểm tra và lấy các giá trị từ giao diện
        description = self.plan_description_field.text().strip()
        plan_date = self.plan_date_field.date().toPyDate()
        plan_type = self.plan_intent_field.currentText()
        plan_status = self.plan_status_field.currentText()
        
        # Thông tin liều lượng
        prescribed_dose = self.prescribed_dose_field.text().strip()
        fractions = self.fractions_field.text().strip()
        
        # Tạo hoặc cập nhật kế hoạch
        if self.current_plan:
            # Cập nhật kế hoạch hiện tại
            self.current_plan.plan_name = plan_name
            self.current_plan.description = description
            self.current_plan.created_date = plan_date
            self.current_plan.plan_type = PlanType(plan_type)
            self.current_plan.status = PlanStatus(plan_status)
            
            # Cập nhật đơn thuốc
            if not self.current_plan.prescription:
                self.current_plan.prescription = Prescription()
                
            self.current_plan.prescription.total_dose = float(prescribed_dose)
            self.current_plan.prescription.fractions = int(fractions)
            
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
                float(prescribed_dose), 
                int(fractions)
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
            new_plan.plan_name = name
            new_plan.description = description
            new_plan.created_date = date
            new_plan.plan_type = PlanType(plan_type)
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
            if self.imrt_button.isChecked():
                new_plan.technique = "IMRT"
            elif self.vmat_button.isChecked():
                new_plan.technique = "VMAT"
            elif self.dcat_button.isChecked():
                new_plan.technique = "DCAT"
            elif self.srs_button.isChecked():
                new_plan.technique = "SRS"
            elif self.sbrt_button.isChecked():
                new_plan.technique = "SBRT"
            else:
                new_plan.technique = "3DCRT"  # Mặc định
            
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
            self.right_tabs.setEnabled(True)
            
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
        self.plan_name_field.clear()
        self.plan_description_field.clear()
        self.plan_date_field.setDate(QDate.currentDate())
        self.plan_intent_field.setCurrentIndex(0)
        self.plan_status_field.setCurrentIndex(0)
        
        # Xóa thông tin liều lượng
        self.prescribed_dose_field.clear()
        self.fractions_field.clear()
        self.dose_per_fraction_field.clear()
        
        # Xóa bảng ràng buộc
        self.constraints_table.setRowCount(0)
        
        # Xóa lựa chọn kỹ thuật
        self.technique_button_group.setExclusive(False)
        for button in self.technique_button_group.buttons():
            button.setChecked(False)
        self.technique_button_group.setExclusive(True)
        
        # Xóa bảng chùm tia
        self.beams_table.setRowCount(0)
    
    def _add_constraint(self):
        """Thêm ràng buộc mới."""
        logger.info("Thêm ràng buộc")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _technique_selected(self, button):
        """
        Xử lý sự kiện khi một kỹ thuật được chọn.
        
        Parameters
        ----------
        button : QRadioButton
            Nút radio được chọn
        """
        logger.info(f"Kỹ thuật được chọn: {button.text()}")
        
        # Xác định kỹ thuật dựa trên nút được chọn
        if button is self.dcat_button:
            self.current_technique = "DCAT"
        elif button is self.imrt_button:
            self.current_technique = "IMRT"
        elif button is self.vmat_button:
            self.current_technique = "VMAT"
        elif button is self.srs_button:
            self.current_technique = "SRS"
        elif button is self.sbrt_button:
            self.current_technique = "SBRT"
        
        # Cập nhật giao diện dựa trên kỹ thuật được chọn
        self._update_beam_controls()
    
    def _update_beam_controls(self):
        """Cập nhật các điều khiển chùm tia dựa trên kỹ thuật được chọn."""
        # Kích hoạt/vô hiệu hóa các nút dựa trên kỹ thuật
        if self.current_technique in ["DCAT", "VMAT", "SRS", "SBRT"]:
            self.add_arc_button.setEnabled(True)
        else:
            self.add_arc_button.setEnabled(False)
        
        # Xóa bảng chùm tia
        self.beams_table.setRowCount(0)
    
    def _calculate_technique_suitability(self):
        """Tính toán độ phù hợp của các kỹ thuật."""
        logger.info("Tính toán độ phù hợp của kỹ thuật")
        
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
        # Hiện tại chỉ mô phỏng kết quả
        
        site = self.plan_intent_field.currentText()
        
        # Mô phỏng tính toán độ phù hợp
        suitability_text = (
            "Độ phù hợp của các kỹ thuật (thang điểm 1-10):\n"
            "- DCAT: 7/10 (Phù hợp cho hầu hết các trường hợp)\n"
            "- IMRT: 9/10 (Rất phù hợp cho các trường hợp phức tạp)\n"
            "- VMAT: 8/10 (Phù hợp cho các trường hợp cần phân bố liều đồng đều)\n"
            "- SRS: 5/10 (Phù hợp cho các tổn thương nhỏ trong não)\n"
            "- SBRT: 6/10 (Phù hợp cho các tổn thương ngoài não)"
        )
        
        self.suitability_label.setText(suitability_text)
    
    def _add_beam(self):
        """Thêm chùm tia mới vào kế hoạch điều trị hiện tại."""
        plan = self._get_current_plan()
        if not plan:
            QMessageBox.warning(
                self,
                "Không thể thêm chùm tia",
                "Vui lòng tạo kế hoạch điều trị trước."
            )
            return
        
        dialog = BeamDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            result = dialog.get_beam_data()
            beam_setup = result['beam_setup']
            
            if not plan.beam_arrangement:
                plan.beam_arrangement = BeamArrangement()
            
            plan.beam_arrangement.add_beam_setup(beam_setup)
            self._update_beams_table()
            
            # Log và thông báo
            logger.info(f"Thêm chùm tia {beam_setup.name} vào kế hoạch {plan.plan_name}")
            self.statusBar().showMessage(f"Đã thêm chùm tia: {beam_setup.name}", 3000)
    
    def _add_arc(self):
        """Thêm chùm tia dạng cung (arc) vào kế hoạch điều trị hiện tại."""
        plan = self._get_current_plan()
        if not plan:
            QMessageBox.warning(
                self,
                "Không thể thêm cung",
                "Vui lòng tạo kế hoạch điều trị trước."
            )
            return
        
        dialog = BeamDialog(self, is_arc=True)
        if dialog.exec_() == QDialog.Accepted:
            result = dialog.get_beam_data()
            beam_setup = result['beam_setup']
            
            # Thiết lập các thông số đặc trưng cho cung
            beam_setup.is_arc = True
            beam_setup.arc_start_angle = result.get('arc_start_angle', 0)
            beam_setup.arc_stop_angle = result.get('arc_stop_angle', 359)
            beam_setup.arc_direction = result.get('arc_direction', 'CW')  # CW: clockwise, CCW: counter-clockwise
            
            if not plan.beam_arrangement:
                plan.beam_arrangement = BeamArrangement()
            
            plan.beam_arrangement.add_beam_setup(beam_setup)
            
            # Cập nhật bảng
            self._update_beam_table()
            
            # Thông báo thay đổi
            self.plan_updated.emit(plan)
            
    def _edit_beam(self):
        """Chỉnh sửa chùm tia đã chọn."""
        plan = self._get_current_plan()
        if not plan or not plan.beam_arrangement:
            return
            
        # Lấy chùm tia đã chọn
        selected_indexes = self.beams_table.selectedIndexes()
        if not selected_indexes:
            QMessageBox.warning(
                self,
                "Không thể chỉnh sửa",
                "Vui lòng chọn một chùm tia để chỉnh sửa."
            )
            return
        
        # Lấy index của dòng đang chọn (chỉ lấy row đầu tiên nếu có nhiều dòng được chọn)
        row = selected_indexes[0].row()
        
        # Lấy beam_id từ bảng
        beam_id = self.beams_table.item(row, 0).data(Qt.UserRole)
        
        # Tìm beam_setup tương ứng
        beam_setup = plan.beam_arrangement.get_beam_setup_by_id(beam_id)
        if not beam_setup:
            QMessageBox.warning(
                self,
                "Lỗi",
                "Không tìm thấy chùm tia đã chọn."
            )
            return
        
        # Mở hộp thoại chỉnh sửa
        dialog = BeamDialog(self, beam_setup)
        if dialog.exec_() == QDialog.Accepted:
            result = dialog.get_beam_data()
            updated_beam_setup = result['beam_setup']
            
            # Cập nhật beam_setup
            # Các thuộc tính đã được cập nhật trong BeamDialog
            
            self._update_beam_table()
            
            # Thông báo thay đổi
            self.plan_updated.emit(plan)
            
    def _delete_beam(self):
        """Xóa chùm tia đã chọn khỏi kế hoạch điều trị hiện tại."""
        plan = self._get_current_plan()
        if not plan or not plan.beam_arrangement:
            return
        
        # Lấy chùm tia đã chọn
        selected_indexes = self.beams_table.selectedIndexes()
        if not selected_indexes:
            QMessageBox.warning(
                self,
                "Không thể xóa",
                "Vui lòng chọn một chùm tia để xóa."
            )
            return
        
        # Lấy index của dòng đang chọn (chỉ lấy row đầu tiên nếu có nhiều dòng được chọn)
        row = selected_indexes[0].row()
        
        # Lấy beam_id từ bảng
        beam_id = self.beams_table.item(row, 0).data(Qt.UserRole)
        beam_name = self.beams_table.item(row, 1).text()
        
        # Hiển thị hộp thoại xác nhận
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa chùm tia '{beam_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Xóa chùm tia khỏi beam_arrangement
            plan.beam_arrangement.remove_beam_setup(beam_id)
            
            # Cập nhật bảng
            self._update_beam_table()
            
            # Thông báo thay đổi
            self.plan_updated.emit(plan)
            
            logger.info(f"Đã xóa chùm tia {beam_name}")
            self.statusBar().showMessage(f"Đã xóa chùm tia: {beam_name}", 3000)
            
    def _update_beams_table(self):
        """Cập nhật bảng hiển thị chùm tia."""
        self.beams_table.setRowCount(0)  # Xóa tất cả dòng
        
        plan = self._get_current_plan()
        if not plan or not plan.beam_arrangement:
            return
        
        # Lấy danh sách các beam_setup
        beam_setups = plan.beam_arrangement.beam_setups
        
        # Thêm dòng cho mỗi chùm tia
        for i, beam_setup in enumerate(beam_setups):
            self.beams_table.insertRow(i)
            
            # ID (ẩn, lưu trong UserRole)
            id_item = QTableWidgetItem()
            id_item.setData(Qt.UserRole, beam_setup.beam_id)
            
            # Tên
            name_item = QTableWidgetItem(beam_setup.name)
            
            # Góc gantry
            gantry_angle = beam_setup.beam_geometry.gantry_angle if beam_setup.beam_geometry else 0
            gantry_item = QTableWidgetItem(f"{gantry_angle:.1f}°")
            
            # Góc collimator
            collimator_angle = beam_setup.beam_geometry.collimator_angle if beam_setup.beam_geometry else 0
            collimator_item = QTableWidgetItem(f"{collimator_angle:.1f}°")
            
            # Góc bàn
            couch_angle = beam_setup.beam_geometry.couch_angle if beam_setup.beam_geometry else 0
            couch_item = QTableWidgetItem(f"{couch_angle:.1f}°")
            
            # MU
            mu_item = QTableWidgetItem(f"{beam_setup.monitor_units:.1f}")
            
            # Trạng thái
            status = beam_setup.metadata.get('status', 'planning')
            status_display = {
                'planning': 'Đang lập kế hoạch',
                'approved': 'Đã phê duyệt',
                'delivered': 'Đã gửi',
                'processed': 'Đã xử lý'
            }
            status_item = QTableWidgetItem(status_display.get(status, 'Không xác định'))
            
            # Đặt các mục vào bảng
            self.beams_table.setItem(i, 0, id_item)
            self.beams_table.setItem(i, 1, name_item)
            self.beams_table.setItem(i, 2, gantry_item)
            self.beams_table.setItem(i, 3, collimator_item)
            self.beams_table.setItem(i, 4, couch_item)
            self.beams_table.setItem(i, 5, mu_item)
            self.beams_table.setItem(i, 6, status_item)
    
    def _get_current_plan(self):
        """Lấy kế hoạch hiện tại."""
        return self.current_plan
    
    def _update_plan(self):
        """Cập nhật kế hoạch hiện tại lên cơ sở dữ liệu."""
        if not self.current_plan:
            return
            
        try:
            # Cập nhật kế hoạch lên cơ sở dữ liệu
            self.plan_db.update_plan(self.current_plan)
            
            # Phát tín hiệu cập nhật
            self.plan_updated.emit(self.current_plan)
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Lỗi", 
                f"Không thể cập nhật kế hoạch: {str(e)}"
            )
            logger.error(f"Lỗi cập nhật kế hoạch: {str(e)}")

    def _save_plan(self):
        """Lưu kế hoạch hiện tại."""
        if not self.current_patient_id:
            QMessageBox.warning(
                self, 
                "Không thể lưu kế hoạch", 
                "Vui lòng chọn một bệnh nhân trước khi lưu kế hoạch."
            )
            return
        
        # Lấy thông tin từ form
        name = self.plan_name_field.text().strip()
        description = self.plan_description_field.toPlainText().strip()
        date = self.plan_date_field.date().toPyDate()
        plan_type = self.plan_type_field.currentText()
        status = self.plan_status_field.currentText()
        dose = self.plan_dose_field.text().strip()
        fractions = self.plan_fractions_field.text().strip()
        
        # Kiểm tra tên kế hoạch
        if not name:
            QMessageBox.warning(
                self, 
                "Tên không hợp lệ", 
                "Vui lòng nhập tên kế hoạch."
            )
            return
            
        try:
            if self.current_plan:
                # Cập nhật kế hoạch hiện tại
                self.current_plan.plan_name = name
                self.current_plan.description = description
                self.current_plan.created_date = date
                self.current_plan.plan_type = PlanType(plan_type)
                self.current_plan.status = PlanStatus(status)
                
                # Cập nhật đơn thuốc
                if not self.current_plan.prescription:
                    self.current_plan.prescription = Prescription()
                self.current_plan.prescription.total_dose = float(dose)
                self.current_plan.prescription.fractions = int(fractions)
                
                # Thiết lập kỹ thuật điều trị
                if self.imrt_button.isChecked():
                    self.current_plan.technique = "IMRT"
                elif self.vmat_button.isChecked():
                    self.current_plan.technique = "VMAT"
                elif self.dcat_button.isChecked():
                    self.current_plan.technique = "DCAT"
                elif self.srs_button.isChecked():
                    self.current_plan.technique = "SRS"
                elif self.sbrt_button.isChecked():
                    self.current_plan.technique = "SBRT"
                else:
                    self.current_plan.technique = "3DCRT"  # Mặc định
                
                # Cập nhật lên DB
                self.plan_db.update_plan(self.current_plan)
                
                # Thông báo
                QMessageBox.information(
                    self, 
                    "Cập nhật kế hoạch", 
                    f"Đã cập nhật kế hoạch {name} thành công."
                )
                
                # Phát tín hiệu cập nhật
                self.plan_updated.emit(self.current_plan)
                logger.info(f"Đã cập nhật kế hoạch ID={self.current_plan.plan_id}")
                
            else:
                # Tạo kế hoạch mới
                self._create_new_plan(name, description, date, plan_type, status, float(dose), int(fractions))
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Lỗi", 
                f"Không thể lưu kế hoạch: {str(e)}"
            )
            logger.error(f"Lỗi lưu kế hoạch: {str(e)}")

    def _init_right_panel(self):
        """Khởi tạo panel bên phải."""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tab widget
        self.right_tabs = QTabWidget()
        self.right_tabs.setEnabled(False)  # Bắt đầu với trạng thái không hoạt động
        
        # Các tab
        self.right_tabs.addTab(self._init_prescription_tab(), "Đơn thuốc")
        self.right_tabs.addTab(self._init_beam_tab(), "Chùm tia")
        self.right_tabs.addTab(self._init_objectives_tab(), "Ràng buộc")
        self.right_tabs.addTab(self._init_optimization_tab(), "Tối ưu")
        self.right_tabs.addTab(self._init_results_tab(), "Kết quả")
        
        right_layout.addWidget(self.right_tabs)
        
        return right_panel

    def _populate_prescription_fields(self):
        """Điền thông tin đơn thuốc vào form."""
        if not self.current_plan or not self.current_plan.prescription:
            return
            
        # Lấy thông tin đơn thuốc
        prescription = self.current_plan.prescription
        
        # Điền vào form
        self.prescribed_dose_field.setText(str(prescription.total_dose))
        self.fractions_field.setText(str(prescription.fractions))
        
        # Tính liều mỗi phân liều
        if prescription.fractions > 0:
            self.dose_per_fraction_field.setText(str(prescription.total_dose / prescription.fractions))

    def _update_prescribed_dose(self):
        """Cập nhật liều chỉ định."""
        if not self.current_plan or not self.current_plan.prescription:
            return
            
        # Lấy giá trị từ form
        total_dose = self.prescribed_dose_field.text().strip()
        fractions = self.fractions_field.text().strip()
        
        # Cập nhật đơn thuốc
        self.current_plan.prescription.total_dose = float(total_dose)
        self.current_plan.prescription.fractions = int(fractions)
        
        # Tính liều mỗi phân liều
        if fractions > 0:
            dose_per_fraction = float(total_dose) / int(fractions)
            self.dose_per_fraction_field.setText(str(dose_per_fraction))
        
        # Lưu thay đổi
        self._update_plan()

    def _populate_plan_data(self):
        """Điền thông tin kế hoạch vào form."""
        if not self.current_plan:
            self._clear_form()
            return
            
        # Điền thông tin kế hoạch vào form
        self.plan_name_field.setText(self.current_plan.plan_name)
        self.plan_description_field.setText(self.current_plan.description)
        
        # Thiết lập ngày tạo
        if self.current_plan.created_date:
            date = QDate.fromString(self.current_plan.created_date.strftime("%Y-%m-%d"), "yyyy-MM-dd")
            self.plan_date_field.setDate(date)
        
        # Thiết lập loại kế hoạch
        index = self.plan_type_field.findText(self.current_plan.plan_type.value)
        if index >= 0:
            self.plan_type_field.setCurrentIndex(index)
            
        # Thiết lập trạng thái
        index = self.plan_status_field.findText(self.current_plan.status.value)
        if index >= 0:
            self.plan_status_field.setCurrentIndex(index)
            
        # Thiết lập kỹ thuật điều trị
        if self.current_plan.technique == "IMRT":
            self.imrt_button.setChecked(True)
        elif self.current_plan.technique == "VMAT":
            self.vmat_button.setChecked(True)
        elif self.current_plan.technique == "DCAT":
            self.dcat_button.setChecked(True)
        elif self.current_plan.technique == "SRS":
            self.srs_button.setChecked(True)
        elif self.current_plan.technique == "SBRT":
            self.sbrt_button.setChecked(True)
            
        # Điền đơn thuốc
        if self.current_plan.prescription:
            self.plan_dose_field.setText(str(self.current_plan.prescription.total_dose))
            self.plan_fractions_field.setText(str(self.current_plan.prescription.fractions))
            
        # Điền bảng chùm tia
        self._populate_beam_table()
        
        # Bật tab
        self.right_tabs.setEnabled(True)

    def _populate_beam_table(self):
        """Điền danh sách chùm tia vào bảng."""
        # Xóa dữ liệu cũ
        self.beam_table.setRowCount(0)
        
        if not self.current_plan or not self.current_plan.beam_arrangement:
            return
        
        # Điền dữ liệu mới
        for i, beam in enumerate(self.current_plan.beam_arrangement.beams):
            self.beam_table.insertRow(i)
            
            # Thiết lập các cột
            self.beam_table.setItem(i, 0, QTableWidgetItem(beam.name))
            self.beam_table.setItem(i, 1, QTableWidgetItem(f"{beam.gantry_angle:.1f}°"))
            self.beam_table.setItem(i, 2, QTableWidgetItem(f"{beam.couch_angle:.1f}°"))
            self.beam_table.setItem(i, 3, QTableWidgetItem(f"{beam.collimator_angle:.1f}°"))
            self.beam_table.setItem(i, 4, QTableWidgetItem(f"{beam.field_size_x:.1f} cm"))
            self.beam_table.setItem(i, 5, QTableWidgetItem(f"{beam.field_size_y:.1f} cm"))
            self.beam_table.setItem(i, 6, QTableWidgetItem(f"{beam.monitor_units:.1f}"))
            self.beam_table.setItem(i, 7, QTableWidgetItem(beam.status))
        
        # Điều chỉnh kích thước cột
        self.beam_table.resizeColumnsToContents()

    def _add_beam(self):
        """Thêm chùm tia mới vào kế hoạch."""
        if not self.current_plan:
            QMessageBox.warning(
                self, 
                "Không có kế hoạch", 
                "Vui lòng chọn hoặc tạo một kế hoạch trước khi thêm chùm tia."
            )
            return
        
        # Tạo hộp thoại chùm tia mới
        dialog = BeamDialog(self)
        
        # Tạo tên mặc định cho chùm tia mới
        beam_count = len(self.current_plan.beam_arrangement.beams) + 1
        default_beam = Beam()
        default_beam.name = f"Beam {beam_count}"
        
        # Thiết lập góc gantry để đều đặn các chùm tia
        if beam_count > 1:
            default_beam.gantry_angle = (beam_count - 1) * (360.0 / beam_count) % 360
        
        # Điền thông tin mặc định vào form
        dialog.beam = default_beam
        dialog._populate_beam_data()
        
        # Hiển thị hộp thoại
        if dialog.exec_() == QDialog.Accepted:
            # Lấy thông tin chùm tia từ form
            new_beam = dialog.get_beam_data()
            
            # Thêm chùm tia vào kế hoạch
            self.current_plan.beam_arrangement.beams.append(new_beam)
            
            # Cập nhật bảng
            self._populate_beam_table()
            
            # Cập nhật kế hoạch
            self._update_plan()
            
            logger.info(f"Đã thêm chùm tia {new_beam.name} vào kế hoạch {self.current_plan.plan_name}")

    def _edit_beam(self):
        """Chỉnh sửa chùm tia được chọn."""
        if not self.current_plan or not self.current_plan.beam_arrangement:
            return
            
        # Lấy chỉ số chùm tia được chọn
        selected_row = self.beam_table.currentRow()
        if selected_row < 0 or selected_row >= len(self.current_plan.beam_arrangement.beams):
            return
            
        # Lấy đối tượng chùm tia
        beam = self.current_plan.beam_arrangement.beams[selected_row]
        
        # Tạo hộp thoại chỉnh sửa
        dialog = BeamDialog(self, beam)
        
        # Hiển thị hộp thoại
        if dialog.exec_() == QDialog.Accepted:
            # Cập nhật thông tin chùm tia
            updated_beam = dialog.get_beam_data()
            self.current_plan.beam_arrangement.beams[selected_row] = updated_beam
            
            # Cập nhật bảng
            self._populate_beam_table()
            
            # Cập nhật kế hoạch
            self._update_plan()
            
            logger.info(f"Đã cập nhật chùm tia {updated_beam.name} trong kế hoạch {self.current_plan.plan_name}")

    def _delete_beam(self):
        """Xóa chùm tia được chọn."""
        if not self.current_plan or not self.current_plan.beam_arrangement:
            return
            
        # Lấy chỉ số chùm tia được chọn
        selected_row = self.beam_table.currentRow()
        if selected_row < 0 or selected_row >= len(self.current_plan.beam_arrangement.beams):
            return
            
        # Lấy tên chùm tia để hiển thị
        beam_name = self.current_plan.beam_arrangement.beams[selected_row].name
        
        # Xác nhận xóa
        confirm = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa chùm tia {beam_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Xóa chùm tia
            del self.current_plan.beam_arrangement.beams[selected_row]
            
            # Cập nhật bảng
            self._populate_beam_table()
            
            # Cập nhật kế hoạch
            self._update_plan()
            
            logger.info(f"Đã xóa chùm tia {beam_name} khỏi kế hoạch {self.current_plan.plan_name}")

    def _update_plan(self):
        """Cập nhật kế hoạch hiện tại lên cơ sở dữ liệu."""
        if not self.current_plan:
            return
            
        try:
            # Cập nhật kế hoạch lên cơ sở dữ liệu
            self.plan_db.update_plan(self.current_plan)
            
            # Phát tín hiệu cập nhật
            self.plan_updated.emit(self.current_plan)
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Lỗi", 
                f"Không thể cập nhật kế hoạch: {str(e)}"
            )
            logger.error(f"Lỗi cập nhật kế hoạch: {str(e)}")

    def _save_plan(self):
        """Lưu kế hoạch hiện tại."""
        if not self.current_patient_id:
            QMessageBox.warning(
                self, 
                "Không thể lưu kế hoạch", 
                "Vui lòng chọn một bệnh nhân trước khi lưu kế hoạch."
            )
            return
        
        # Lấy thông tin từ form
        name = self.plan_name_field.text().strip()
        description = self.plan_description_field.toPlainText().strip()
        date = self.plan_date_field.date().toPyDate()
        plan_type = self.plan_type_field.currentText()
        status = self.plan_status_field.currentText()
        dose = self.plan_dose_field.text().strip()
        fractions = self.plan_fractions_field.text().strip()
        
        # Kiểm tra tên kế hoạch
        if not name:
            QMessageBox.warning(
                self, 
                "Tên không hợp lệ", 
                "Vui lòng nhập tên kế hoạch."
            )
            return
            
        try:
            if self.current_plan:
                # Cập nhật kế hoạch hiện tại
                self.current_plan.plan_name = name
                self.current_plan.description = description
                self.current_plan.created_date = date
                self.current_plan.plan_type = PlanType(plan_type)
                self.current_plan.status = PlanStatus(status)
                
                # Cập nhật đơn thuốc
                if not self.current_plan.prescription:
                    self.current_plan.prescription = Prescription()
                self.current_plan.prescription.total_dose = float(dose)
                self.current_plan.prescription.fractions = int(fractions)
                
                # Thiết lập kỹ thuật điều trị
                if self.imrt_button.isChecked():
                    self.current_plan.technique = "IMRT"
                elif self.vmat_button.isChecked():
                    self.current_plan.technique = "VMAT"
                elif self.dcat_button.isChecked():
                    self.current_plan.technique = "DCAT"
                elif self.srs_button.isChecked():
                    self.current_plan.technique = "SRS"
                elif self.sbrt_button.isChecked():
                    self.current_plan.technique = "SBRT"
                else:
                    self.current_plan.technique = "3DCRT"  # Mặc định
                
                # Cập nhật lên DB
                self.plan_db.update_plan(self.current_plan)
                
                # Thông báo
                QMessageBox.information(
                    self, 
                    "Cập nhật kế hoạch", 
                    f"Đã cập nhật kế hoạch {name} thành công."
                )
                
                # Phát tín hiệu cập nhật
                self.plan_updated.emit(self.current_plan)
                logger.info(f"Đã cập nhật kế hoạch ID={self.current_plan.plan_id}")
                
            else:
                # Tạo kế hoạch mới
                self._create_new_plan(name, description, date, plan_type, status, float(dose), int(fractions))
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Lỗi", 
                f"Không thể lưu kế hoạch: {str(e)}"
            )
            logger.error(f"Lỗi lưu kế hoạch: {str(e)}")

    def _init_right_panel(self):
        """Khởi tạo panel bên phải."""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tab widget
        self.right_tabs = QTabWidget()
        self.right_tabs.setEnabled(False)  # Bắt đầu với trạng thái không hoạt động
        
        # Các tab
        self.right_tabs.addTab(self._init_prescription_tab(), "Đơn thuốc")
        self.right_tabs.addTab(self._init_beam_tab(), "Chùm tia")
        self.right_tabs.addTab(self._init_objectives_tab(), "Ràng buộc")
        self.right_tabs.addTab(self._init_optimization_tab(), "Tối ưu")
        self.right_tabs.addTab(self._init_results_tab(), "Kết quả")
        
        right_layout.addWidget(self.right_tabs)
        
        return right_panel

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
            # Thu thập mục tiêu và ràng buộc
            from quangtps.optimization.objectives import ObjectiveCollection
            from quangtps.optimization.constraints import ConstraintCollection
            from quangtps.optimization.optimization_engine import OptimizationEngine, OptimizationParameters
            
            # Tạo các mục tiêu và ràng buộc từ kế hoạch
            objectives = ObjectiveCollection()
            constraints = ConstraintCollection()
            
            # Thêm mục tiêu cho các cấu trúc đích
            for target in plan.targets:
                if target.prescription and target.prescription.dose > 0:
                    objectives.add_uniform_dose(
                        structure_id=target.structure_id,
                        structure_name=target.name,
                        target_dose=target.prescription.dose,
                        weight=100.0
                    )
            
            # Thêm ràng buộc liều cho các cấu trúc nguy cấp
            for oar in plan.oars:
                if oar.dose_constraints:
                    for constraint in oar.dose_constraints:
                        constraints.add_dose_volume_constraint(
                            structure_id=oar.structure_id,
                            structure_name=oar.name,
                            dose_volume_type=constraint.type,
                            dose_value=constraint.dose,
                            volume_value=constraint.volume,
                            priority=constraint.priority
                        )
            
            # Tạo các tham số tối ưu hóa
            parameters = OptimizationParameters(
                max_iterations=100,
                convergence_threshold=0.001,
                step_size=0.1
            )
            
            # Tạo động cơ tối ưu hóa
            engine = OptimizationEngine(
                objectives=objectives,
                constraints=constraints,
                parameters=parameters,
                solver_name="gradient_descent"
            )
            
            # Chuẩn bị dữ liệu ban đầu
            from quangtps.dose.dose_grid import DoseGrid
            import numpy as np
            
            # Tạo lưới liều ban đầu
            dose_grid = DoseGrid.create_from_image(plan.image_series)
            
            # Tạo từ điển cấu trúc
            structures = {}
            for target in plan.targets:
                structures[target.structure_id] = target.get_mask()
            for oar in plan.oars:
                structures[oar.structure_id] = oar.get_mask()
                
            # Thiết lập trạng thái ban đầu
            engine.set_initial_state(
                dose_grid=dose_grid,
                structures=structures
            )
            
            # Đăng ký callback để cập nhật giao diện
            from quangtps.optimization.optimization_engine import OptimizationEvent
            
            def update_progress(context):
                iteration = context.get('iteration', 0)
                total = context.get('total_iterations', 100)
                value = context.get('objective_value', 0)
                msg.setText(f"Tối ưu hóa kế hoạch...\nLặp: {iteration}/{total}\nGiá trị: {value:.4f}")
                QApplication.processEvents()
                
            engine.register_callback(OptimizationEvent.ITERATION_COMPLETED, update_progress)
            
            # Chạy tối ưu hóa
            results = engine.optimize()
            
            # Cập nhật kế hoạch với kết quả
            plan.optimization_results = results
            
            # Cập nhật liều tính toán
            plan.calculated_dose = results.final_dose_grid
            
            # Cập nhật DVH
            plan.update_dvh()
            
            # Cập nhật giao diện
            self._update_plan(plan)
            
            # Hiển thị kết quả
            QMessageBox.information(
                self,
                "Tối ưu hóa hoàn tất",
                f"Tối ưu hóa kế hoạch đã hoàn tất.\n\n"
                f"Số lặp: {results.num_iterations}\n"
                f"Thời gian: {results.elapsed_time:.2f} giây\n"
                f"Cải thiện: {results.get_improvement_percentage():.2f}%"
            )
            
            # Cập nhật kế hoạch
            self._update_plan()
            
            # Log
            logger.info(f"Tối ưu hóa kế hoạch {plan.plan_name} hoàn tất sau {results.num_iterations} lần lặp")
            
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
        """Cập nhật hiển thị thông tin kế hoạch sau khi tối ưu hóa."""
        plan = self._get_current_plan()
        if not plan:
            return
            
        # Cập nhật thông tin hiển thị
        self._populate_plan_data()
        self._populate_beam_table()
        
        # Phát tín hiệu cập nhật
        self.plan_updated.emit(plan)
        
        # Thông báo
        self.statusBar().showMessage(f"Đã cập nhật hiển thị kế hoạch: {plan.plan_name}", 3000)
        logger.info(f"Đã cập nhật hiển thị kế hoạch: {plan.plan_name}")
