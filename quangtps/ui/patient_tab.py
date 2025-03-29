#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab thông tin bệnh nhân (Patient Tab) cho QuangTPS.

Module này cung cấp giao diện để hiển thị và chỉnh sửa thông tin bệnh nhân,
bao gồm thông tin cá nhân, lịch sử bệnh, và các hồ sơ y tế liên quan.
"""

import logging
import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import os
import shutil
import uuid

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QComboBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QScrollArea, QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QFileDialog, QDialog, QHeaderView, QProgressBar,
    QInputDialog, QFrame, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QDate, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

from quangtps.database.patient_db import PatientDB
from quangtps.dicom.dicom_importer import DicomImporter
from quangtps.dicom.dicom_exporter import DicomExporter
from quangtps.core.patient import Patient
from quangtps.ui.patient_creation_dialog import PatientCreationDialog
from quangtps.ui.patient_search_result_dialog import PatientSearchResultDialog
from quangtps.ui.new_patient_dialog import NewPatientDialog

logger = logging.getLogger(__name__)


class PatientTab(QWidget):
    """
    Tab hiển thị và chỉnh sửa thông tin bệnh nhân.
    
    Tab này bao gồm các phần thông tin cơ bản, lịch sử bệnh,
    kết quả khám lâm sàng, hình ảnh y tế, và các dữ liệu khác
    liên quan đến bệnh nhân.
    """
    
    # Tín hiệu để thông báo khi cập nhật dữ liệu bệnh nhân
    patient_updated = pyqtSignal(str)  # patient_id
    patient_created = pyqtSignal(str)
    patient_deleted = pyqtSignal(str)  # patient_id
    
    def __init__(self, parent=None):
        """
        Khởi tạo tab thông tin bệnh nhân.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Trạng thái
        self.current_patient = None
        self.patient_db = PatientDB()
        self.dicom_importer = DicomImporter()
        self.dicom_exporter = DicomExporter()
        
        # Thiết lập giao diện
        self._init_ui()
        
        logger.info("Khởi tạo tab thông tin bệnh nhân hoàn tất")
    
    def _init_ui(self):
        """Khởi tạo giao diện tab bệnh nhân"""
        layout = QVBoxLayout(self)

        # Phần đầu: Tìm kiếm bệnh nhân và tạo mới
        top_layout = QHBoxLayout()

        # Phần tìm kiếm
        search_group = QGroupBox("Tìm kiếm bệnh nhân")
        search_layout = QHBoxLayout(search_group)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Nhập tên hoặc mã bệnh nhân")
        self.search_input.returnPressed.connect(self._search_patients)

        self.search_button = QPushButton("Tìm")
        self.search_button.clicked.connect(self._search_patients)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)

        # Phần nút tạo mới
        button_layout = QVBoxLayout()

        self.create_button = QPushButton("Tạo bệnh nhân mới")
        self.create_button.clicked.connect(self._create_new_patient)

        self.delete_button = QPushButton("Xóa bệnh nhân")
        self.delete_button.clicked.connect(self._delete_current_patient)
        # Vô hiệu hóa cho đến khi chọn bệnh nhân
        self.delete_button.setEnabled(False)

        import_export_layout = QHBoxLayout()

        self.import_button = QPushButton("Nhập dữ liệu")
        self.import_button.clicked.connect(self._import_patient_data)
        self.import_button.setEnabled(False)

        self.export_button = QPushButton("Xuất dữ liệu")
        self.export_button.clicked.connect(self._export_patient_data)
        self.export_button.setEnabled(False)

        import_export_layout.addWidget(self.import_button)
        import_export_layout.addWidget(self.export_button)

        button_layout.addWidget(self.create_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addLayout(import_export_layout)

        top_layout.addWidget(search_group, 3)
        top_layout.addLayout(button_layout, 1)

        layout.addLayout(top_layout)

        # Phần chính: thông tin chi tiết về bệnh nhân
        self.stacked_widget = QTabWidget()

        # Tab thông tin cơ bản
        self.basic_info_tab = QWidget()
        basic_info_layout = QVBoxLayout(self.basic_info_tab)

        # Tạo scroll area để có thể cuộn khi có nhiều thông tin
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Form nhập thông tin cơ bản
        form_group = QGroupBox("Thông tin cơ bản")
        form_layout = QFormLayout(form_group)

        self.name_edit = QLineEdit()
        self.name_edit.setReadOnly(True)

        self.birth_date_edit = QDateEdit()
        self.birth_date_edit.setReadOnly(True)
        self.birth_date_edit.setCalendarPopup(True)

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Nam", "Nữ", "Khác"])
        self.gender_combo.setEnabled(False)
        
        # Thêm các trường thông tin liên hệ
        self.address_edit = QTextEdit()
        self.address_edit.setReadOnly(True)
        self.address_edit.setMaximumHeight(60)
        
        self.phone_edit = QLineEdit()
        self.phone_edit.setReadOnly(True)
        
        self.email_edit = QLineEdit()
        self.email_edit.setReadOnly(True)

        self.notes_edit = QTextEdit()
        self.notes_edit.setReadOnly(True)
        self.notes_edit.setMaximumHeight(80)

        form_layout.addRow("Họ tên:", self.name_edit)
        form_layout.addRow("Ngày sinh:", self.birth_date_edit)
        form_layout.addRow("Giới tính:", self.gender_combo)
        form_layout.addRow("Địa chỉ:", self.address_edit)
        form_layout.addRow("Điện thoại:", self.phone_edit)
        form_layout.addRow("Email:", self.email_edit)
        form_layout.addRow("Ghi chú:", self.notes_edit)
        
        # Thêm form group vào scroll layout
        scroll_layout.addWidget(form_group)

        # Nút chỉnh sửa / lưu
        edit_layout = QHBoxLayout()

        self.edit_button = QPushButton("Chỉnh sửa")
        self.edit_button.clicked.connect(lambda: self._toggle_edit_mode(True))
        self.edit_button.setEnabled(False)

        self.save_button = QPushButton("Lưu")
        self.save_button.clicked.connect(self._save_patient_info)
        self.save_button.setEnabled(False)

        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.clicked.connect(
            lambda: self._toggle_edit_mode(False))
        self.cancel_button.setEnabled(False)

        edit_layout.addWidget(self.edit_button)
        edit_layout.addWidget(self.save_button)
        edit_layout.addWidget(self.cancel_button)
        edit_layout.addStretch()

        scroll_layout.addLayout(edit_layout)
        scroll_area.setWidget(scroll_content)
        basic_info_layout.addWidget(scroll_area)

        # Tab thông tin y tế
        self.medical_info_tab = QWidget()
        self._init_medical_info_tab()
        
        # Tab thông tin xạ trị
        self.rt_info_tab = QWidget()
        self._init_rt_info_tab()

        # Tab lịch sử bệnh
        self.medical_history_tab = QWidget()
        medical_history_layout = QVBoxLayout(self.medical_history_tab)

        # Bảng lịch sử bệnh
        history_group = QGroupBox("Lịch sử bệnh")
        history_layout = QVBoxLayout(history_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(
            ["Ngày", "Chẩn đoán", "Bác sĩ", "Ghi chú"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        history_button_layout = QHBoxLayout()

        self.add_history_button = QPushButton("Thêm")
        self.add_history_button.clicked.connect(self._add_medical_history)
        self.add_history_button.setEnabled(False)

        history_button_layout.addWidget(self.add_history_button)
        history_button_layout.addStretch()

        history_layout.addWidget(self.history_table)
        history_layout.addLayout(history_button_layout)

        medical_history_layout.addWidget(history_group)

        # Tab dữ liệu y tế
        self.medical_data_tab = QWidget()
        medical_data_layout = QVBoxLayout(self.medical_data_tab)

        # Bảng dữ liệu ảnh y tế
        images_group = QGroupBox("Dữ liệu y tế")
        images_layout = QVBoxLayout(images_group)

        # Tạo splitter để chia bảng studies và series
        splitter = QSplitter(Qt.Vertical)

        # Bảng nghiên cứu
        studies_widget = QWidget()
        studies_layout = QVBoxLayout(studies_widget)
        studies_layout.setContentsMargins(0, 0, 0, 0)

        studies_label = QLabel("Danh sách nghiên cứu")
        studies_label.setStyleSheet("font-weight: bold;")

        self.studies_table = QTableWidget()
        self.studies_table.setColumnCount(4)
        self.studies_table.setHorizontalHeaderLabels(
            ["Ngày", "Mô tả", "Loại dữ liệu", "Số lượng series"])
        self.studies_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.studies_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.studies_table.setSelectionMode(QTableWidget.SingleSelection)
        self.studies_table.itemSelectionChanged.connect(
            self._on_study_selected)

        studies_layout.addWidget(studies_label)
        studies_layout.addWidget(self.studies_table)

        # Bảng series
        series_widget = QWidget()
        series_layout = QVBoxLayout(series_widget)
        series_layout.setContentsMargins(0, 0, 0, 0)

        series_label = QLabel("Danh sách series")
        series_label.setStyleSheet("font-weight: bold;")

        self.series_table = QTableWidget()
        self.series_table.setColumnCount(3)
        self.series_table.setHorizontalHeaderLabels(
            ["Mô tả", "Loại dữ liệu", "Số lượng file"])
        self.series_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.series_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.series_table.setSelectionMode(QTableWidget.SingleSelection)

        series_layout.addWidget(series_label)
        series_layout.addWidget(self.series_table)

        # Thêm widgets vào splitter
        splitter.addWidget(studies_widget)
        splitter.addWidget(series_widget)
        splitter.setSizes([200, 150])

        # Nút thêm dữ liệu y tế
        images_button_layout = QHBoxLayout()

        self.import_images_button = QPushButton("Nhập DICOM")
        self.import_images_button.clicked.connect(self._import_medical_images)
        self.import_images_button.setEnabled(False)

        self.view_images_button = QPushButton("Xem ảnh")
        self.view_images_button.clicked.connect(self._view_medical_images)
        self.view_images_button.setEnabled(False)

        images_button_layout.addWidget(self.import_images_button)
        images_button_layout.addWidget(self.view_images_button)
        images_button_layout.addStretch()

        images_layout.addWidget(splitter)
        images_layout.addLayout(images_button_layout)

        medical_data_layout.addWidget(images_group)

        # Thêm các tab vào stacked widget
        self.stacked_widget.addTab(self.basic_info_tab, "Thông tin cơ bản")
        self.stacked_widget.addTab(self.medical_info_tab, "Thông tin y tế")
        self.stacked_widget.addTab(self.rt_info_tab, "Thông tin xạ trị")
        self.stacked_widget.addTab(self.medical_history_tab, "Lịch sử bệnh")
        self.stacked_widget.addTab(self.medical_data_tab, "Dữ liệu y tế")

        layout.addWidget(self.stacked_widget)
        
        # Vô hiệu hóa các widget khi không có bệnh nhân được chọn
        self._toggle_edit_mode(False)
        
        # Kết nối tín hiệu
        self.name_edit.textChanged.connect(self._on_patient_data_changed)
        
        logger.info("Khởi tạo giao diện tab bệnh nhân hoàn tất")
    
    def _init_medical_info_tab(self):
        """Khởi tạo tab thông tin y tế."""
        layout = QVBoxLayout(self.medical_info_tab)
        
        # Tạo scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Thông tin hành chính y tế
        admin_group = QGroupBox("Thông tin hành chính")
        admin_layout = QFormLayout(admin_group)
        
        # Mã số bệnh án
        self.mrn_edit = QLineEdit()
        self.mrn_edit.setReadOnly(True)
        admin_layout.addRow("Mã số bệnh án:", self.mrn_edit)
        
        # Bác sĩ chính
        self.primary_physician_edit = QLineEdit()
        self.primary_physician_edit.setReadOnly(True)
        admin_layout.addRow("Bác sĩ chính:", self.primary_physician_edit)
        
        # Bác sĩ giới thiệu
        self.referring_physician_edit = QLineEdit()
        self.referring_physician_edit.setReadOnly(True)
        admin_layout.addRow("Bác sĩ giới thiệu:", self.referring_physician_edit)
        
        # Mã bệnh viện
        self.hospital_id_edit = QLineEdit()
        self.hospital_id_edit.setReadOnly(True)
        admin_layout.addRow("Mã bệnh viện:", self.hospital_id_edit)
        
        # Mã bảo hiểm
        self.insurance_id_edit = QLineEdit()
        self.insurance_id_edit.setReadOnly(True)
        admin_layout.addRow("Mã bảo hiểm:", self.insurance_id_edit)
        
        # Thông tin thể chất
        physical_group = QGroupBox("Thông tin thể chất")
        physical_layout = QFormLayout(physical_group)
        
        # Chiều cao
        self.height_edit = QDoubleSpinBox()
        self.height_edit.setRange(0, 250)
        self.height_edit.setDecimals(1)
        self.height_edit.setSuffix(" cm")
        self.height_edit.setReadOnly(True)
        self.height_edit.setButtonSymbols(QDoubleSpinBox.NoButtons)
        physical_layout.addRow("Chiều cao:", self.height_edit)
        
        # Cân nặng
        self.weight_edit = QDoubleSpinBox()
        self.weight_edit.setRange(0, 250)
        self.weight_edit.setDecimals(1)
        self.weight_edit.setSuffix(" kg")
        self.weight_edit.setReadOnly(True)
        self.weight_edit.setButtonSymbols(QDoubleSpinBox.NoButtons)
        physical_layout.addRow("Cân nặng:", self.weight_edit)
        
        # Hiển thị BMI và BSA
        self.bmi_label = QLabel("BMI: ...")
        self.bsa_label = QLabel("BSA: ...")
        
        bmi_bsa_layout = QHBoxLayout()
        bmi_bsa_layout.addWidget(self.bmi_label)
        bmi_bsa_layout.addWidget(self.bsa_label)
        physical_layout.addRow("Chỉ số:", bmi_bsa_layout)
        
        # Dị ứng
        self.allergies_edit = QTextEdit()
        self.allergies_edit.setReadOnly(True)
        self.allergies_edit.setMaximumHeight(80)
        physical_layout.addRow("Dị ứng:", self.allergies_edit)
        
        # Thêm các nhóm vào layout
        scroll_layout.addWidget(admin_group)
        scroll_layout.addWidget(physical_group)
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
    
    def _init_rt_info_tab(self):
        """Khởi tạo tab thông tin xạ trị."""
        layout = QVBoxLayout(self.rt_info_tab)
        
        # Tạo scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Thông tin chẩn đoán
        diagnosis_group = QGroupBox("Thông tin chẩn đoán")
        diagnosis_layout = QFormLayout(diagnosis_group)
        
        # Mã chẩn đoán ICD-10
        self.diagnosis_code_edit = QLineEdit()
        self.diagnosis_code_edit.setReadOnly(True)
        diagnosis_layout.addRow("Mã ICD-10:", self.diagnosis_code_edit)
        
        # Chẩn đoán chi tiết
        self.diagnosis_edit = QTextEdit()
        self.diagnosis_edit.setReadOnly(True)
        self.diagnosis_edit.setMaximumHeight(80)
        diagnosis_layout.addRow("Chẩn đoán:", self.diagnosis_edit)
        
        # Thông tin kế hoạch xạ trị
        planning_group = QGroupBox("Thông tin kế hoạch xạ trị")
        planning_layout = QFormLayout(planning_group)
        
        # Vị trí điều trị
        self.site_combo = QComboBox()
        self.site_combo.addItems([
            "", "Não", "Đầu cổ", "Phổi", "Vú", "Thực quản", "Gan", 
            "Tụy", "Tuyến tiền liệt", "Trực tràng", "Cổ tử cung", "Hạch bạch huyết", "Khác"
        ])
        self.site_combo.setEnabled(False)
        planning_layout.addRow("Vị trí điều trị:", self.site_combo)
        
        # Kỹ thuật xạ trị
        self.technique_combo = QComboBox()
        self.technique_combo.addItems([
            "", "3D-CRT", "IMRT", "VMAT", "SBRT", "SRS", 
            "Electron", "IORT", "Brachytherapy", "Proton", "Carbon ion", "Khác"
        ])
        self.technique_combo.setEnabled(False)
        planning_layout.addRow("Kỹ thuật xạ trị:", self.technique_combo)
        
        # Mục đích điều trị
        self.treatment_intent_combo = QComboBox()
        self.treatment_intent_combo.addItems([
            "", "Điều trị triệt căn (Curative)", "Điều trị giảm nhẹ (Palliative)", 
            "Điều trị bổ trợ (Adjuvant)", "Điều trị tân bổ trợ (Neoadjuvant)", "Khác"
        ])
        self.treatment_intent_combo.setEnabled(False)
        planning_layout.addRow("Mục đích điều trị:", self.treatment_intent_combo)
        
        # Thêm các nhóm vào layout
        scroll_layout.addWidget(diagnosis_group)
        scroll_layout.addWidget(planning_group)
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
    
    def _toggle_edit_mode(self, enabled: bool):
        """
        Bật/tắt chế độ chỉnh sửa thông tin bệnh nhân.

        Parameters
        ----------
        enabled : bool
            True để bật chế độ chỉnh sửa, False để tắt
        """
        if enabled and self.current_patient:
            # Sử dụng dialog chỉnh sửa thay vì chỉnh sửa trực tiếp trên form
            dialog = PatientCreationDialog(
                parent=self,
                patient_id=self.current_patient['id'],
                edit_mode=True,
                patient_data=self.current_patient
            )
            
            if dialog.exec_() == QDialog.Accepted:
                # Lấy dữ liệu bệnh nhân sau khi chỉnh sửa
                updated_data = dialog.get_patient_data()
                
                # Cập nhật thông tin bệnh nhân hiện tại
                self.current_patient = self.patient_db.get_patient(updated_data['id'])
                
                # Cập nhật lại giao diện
                self._populate_patient_data()
                
                # Kích hoạt tín hiệu cập nhật
                self.patient_updated.emit(self.current_patient['id'])
                
                # Ghi log
                logger.info(f"Đã cập nhật thông tin bệnh nhân: {self.current_patient['id']}")
            
            # Luôn luôn ở chế độ đọc sau khi mở dialog chỉnh sửa
            self._update_edit_mode_ui(False)
            return
            
        # Nếu là tắt chế độ chỉnh sửa hoặc không có patient hiện tại
        self._update_edit_mode_ui(enabled)
            
    def _update_edit_mode_ui(self, enabled: bool):
        """
        Cập nhật giao diện theo trạng thái chỉnh sửa.
        
        Parameters
        ----------
        enabled : bool
            True để bật chế độ chỉnh sửa, False để tắt
        """
        # Thiết lập trạng thái của các trường nhập liệu thông tin cơ bản
        self.name_edit.setReadOnly(not enabled)
        self.birth_date_edit.setReadOnly(not enabled)
        self.gender_combo.setEnabled(enabled)
        self.address_edit.setReadOnly(not enabled)
        self.phone_edit.setReadOnly(not enabled)
        self.email_edit.setReadOnly(not enabled)
        self.notes_edit.setReadOnly(not enabled)
        
        # Thiết lập trạng thái của các trường nhập liệu thông tin y tế
        self.mrn_edit.setReadOnly(not enabled)
        self.primary_physician_edit.setReadOnly(not enabled)
        self.referring_physician_edit.setReadOnly(not enabled)
        self.hospital_id_edit.setReadOnly(not enabled)
        self.insurance_id_edit.setReadOnly(not enabled)
        
        self.height_edit.setReadOnly(not enabled)
        self.height_edit.setButtonSymbols(QDoubleSpinBox.NoButtons if not enabled else QDoubleSpinBox.UpDownArrows)
        
        self.weight_edit.setReadOnly(not enabled)
        self.weight_edit.setButtonSymbols(QDoubleSpinBox.NoButtons if not enabled else QDoubleSpinBox.UpDownArrows)
        
        self.allergies_edit.setReadOnly(not enabled)
        
        # Thiết lập trạng thái của các trường nhập liệu thông tin xạ trị
        self.diagnosis_code_edit.setReadOnly(not enabled)
        self.diagnosis_edit.setReadOnly(not enabled)
        self.site_combo.setEnabled(enabled)
        self.technique_combo.setEnabled(enabled)
        self.treatment_intent_combo.setEnabled(enabled)

        # Cập nhật trạng thái các nút
        self.edit_button.setEnabled(not enabled)
        self.save_button.setEnabled(enabled)
        self.cancel_button.setEnabled(enabled)
        
        # Nếu bật chế độ chỉnh sửa, cập nhật các chỉ số BMI và BSA
        if enabled:
            self._update_physical_metrics()
            
            # Kết nối tín hiệu cho các trường sẽ ảnh hưởng đến BMI và BSA
            self.height_edit.valueChanged.connect(self._update_physical_metrics)
            self.weight_edit.valueChanged.connect(self._update_physical_metrics)
        else:
            # Ngắt kết nối khi không ở chế độ chỉnh sửa
            try:
                self.height_edit.valueChanged.disconnect(self._update_physical_metrics)
                self.weight_edit.valueChanged.disconnect(self._update_physical_metrics)
            except TypeError:
                # Bỏ qua lỗi nếu chưa được kết nối
                pass
    
    def _on_patient_data_changed(self):
        """Xử lý khi dữ liệu bệnh nhân thay đổi."""
        # Đổi tiêu đề tab để hiển thị trạng thái chưa lưu
        if self.current_patient and not self.windowTitle().endswith("*"):
            self.setWindowTitle(f"{self.windowTitle()} *")
    
    def set_patient(self, patient_id: str):
        """
        Thiết lập bệnh nhân hiện tại và hiển thị thông tin.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        """
        # Xóa dữ liệu bệnh nhân hiện tại
        self._clear_patient_data()
        
        try:
            # Lấy thông tin bệnh nhân
            patient_data = self.patient_db.get_patient(patient_id)
            
            if not patient_data:
                # Kiểm tra xem có phải ID hợp lệ không
                try:
                    uuid.UUID(patient_id)  # Kiểm tra định dạng UUID
                    
                    # Hiển thị hộp thoại xác nhận để tạo bệnh nhân mới
                    reply = QMessageBox.question(
                        self, 
                        "Tạo bệnh nhân mới",
                        f"Không tìm thấy bệnh nhân với ID: {patient_id}\nBạn có muốn tạo bệnh nhân mới với ID này không?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    
                    if reply == QMessageBox.Yes:
                        # Tạo dialog để nhập thông tin bệnh nhân mới
                        dialog = PatientCreationDialog(self)
                        # Thiết lập ID cố định
                        dialog.patient_id_edit.setText(patient_id)
                        dialog.patient_id_edit.setReadOnly(True)
                        
                        if dialog.exec_() == QDialog.Accepted:
                            # Lấy lại thông tin bệnh nhân sau khi tạo
                            patient_data = self.patient_db.get_patient(patient_id)
                            if not patient_data:
                                QMessageBox.warning(
                                    self, "Cảnh báo", f"Không thể lấy thông tin của bệnh nhân vừa tạo: {patient_id}")
                                return
                        else:
                            # Người dùng hủy tạo mới
                            return
                    else:
                        # Không muốn tạo bệnh nhân mới
                        return
                except (ValueError, TypeError):
                    # Không phải UUID hợp lệ
                    QMessageBox.warning(
                        self, "Cảnh báo", f"Không tìm thấy bệnh nhân với ID: {patient_id}")
                return
            
            # Lưu bệnh nhân hiện tại
            self.current_patient = patient_data

            # Hiển thị thông tin
            self._populate_patient_data()
            self._populate_medical_history()
            self._populate_medical_images()

            # Bật các nút và widget
            self.delete_button.setEnabled(True)
            self.edit_button.setEnabled(True)
            self.import_button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.import_images_button.setEnabled(True)
            self.add_history_button.setEnabled(True)

            # Ghi log
            logger.info(f"Đã hiển thị thông tin bệnh nhân: {patient_id}")

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể hiển thị thông tin bệnh nhân: {str(e)}")
            logger.error(
                f"Lỗi khi thiết lập bệnh nhân {patient_id}: {str(e)}", exc_info=True)
    
    def _populate_patient_data(self):
        """
        Hiển thị thông tin bệnh nhân lên giao diện.
        """
        if not self.current_patient:
            return
        
        try:
            # Lấy dữ liệu từ đối tượng bệnh nhân hiện tại
            patient = self.current_patient

            # Hiển thị thông tin cơ bản
            self.name_edit.setText(patient.get('name', ''))

            # Ngày sinh - ưu tiên trường 'dob' trước, nếu không có thì dùng 'birth_date'
            birth_date = patient.get('dob', '') or patient.get('birth_date', '')
            if birth_date:
                try:
                    # Chuyển đổi chuỗi ngày tháng thành QDate
                    date_parts = birth_date.split('-')
                    if len(date_parts) == 3:
                        year, month, day = map(int, date_parts)
                        self.birth_date_edit.setDate(QDate(year, month, day))
                except Exception as e:
                    logger.warning(
                        f"Không thể phân tích ngày sinh: {birth_date}, lỗi: {str(e)}")

            # Giới tính
            gender_map = {'male': 'Nam', 'female': 'Nữ', 'other': 'Khác'}
            gender = gender_map.get(patient.get('gender', ''), 'Khác')
            self.gender_combo.setCurrentText(gender)

            # Thông tin liên hệ
            self.address_edit.setText(patient.get('address', ''))
            self.phone_edit.setText(patient.get('phone', ''))
            self.email_edit.setText(patient.get('email', ''))
            
            # Ghi chú
            self.notes_edit.setText(patient.get('notes', ''))

            # Thông tin y tế
            self.mrn_edit.setText(patient.get('mrn', ''))
            self.primary_physician_edit.setText(patient.get('primary_physician', ''))
            self.referring_physician_edit.setText(patient.get('referring_physician', ''))
            self.hospital_id_edit.setText(patient.get('hospital_id', ''))
            self.insurance_id_edit.setText(patient.get('insurance_id', ''))
            
            # Thông tin thể chất
            try:
                height = float(patient.get('height_cm', 0))
                self.height_edit.setValue(height)
            except (ValueError, TypeError):
                self.height_edit.setValue(0)
                
            try:
                weight = float(patient.get('weight_kg', 0))
                self.weight_edit.setValue(weight)
            except (ValueError, TypeError):
                self.weight_edit.setValue(0)
                
            # Dị ứng
            self.allergies_edit.setText(patient.get('allergies', ''))
            
            # Thông tin xạ trị
            self.diagnosis_code_edit.setText(patient.get('diagnosis_code', ''))
            self.diagnosis_edit.setText(patient.get('diagnosis', ''))
            
            # Chọn giá trị trong combobox dựa trên dữ liệu bệnh nhân
            site = patient.get('site', '')
            if site:
                index = self.site_combo.findText(site, Qt.MatchFixedString)
                if index >= 0:
                    self.site_combo.setCurrentIndex(index)
            
            technique = patient.get('technique', '')
            if technique:
                index = self.technique_combo.findText(technique, Qt.MatchFixedString)
                if index >= 0:
                    self.technique_combo.setCurrentIndex(index)
                    
            intent = patient.get('treatment_intent', '')
            if intent:
                index = self.treatment_intent_combo.findText(intent, Qt.MatchContains)
                if index >= 0:
                    self.treatment_intent_combo.setCurrentIndex(index)
                    
            # Cập nhật các chỉ số BMI và BSA
            self._update_physical_metrics()
            
            # Kiểm tra và hiển thị metadata nếu có
            if 'metadata' in patient and patient['metadata']:
                metadata = patient['metadata']

                # Hiển thị thêm thông tin từ metadata nếu có
                additional_info = ""

                if 'dicom_id' in metadata and metadata['dicom_id']:
                    additional_info += f"DICOM ID: {metadata['dicom_id']}\n"

                if 'external_id' in metadata and metadata['external_id']:
                    additional_info += f"ID ngoài: {metadata['external_id']}\n"

                if additional_info:
                    current_text = self.notes_edit.toPlainText()
                    if current_text:
                        self.notes_edit.setText(
                            f"{current_text}\n\n{additional_info}")
                    else:
                        self.notes_edit.setText(additional_info)

        except Exception as e:
            logger.error(
                f"Lỗi khi hiển thị thông tin bệnh nhân: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Lỗi", f"Không thể hiển thị thông tin bệnh nhân: {str(e)}")
    
    def _update_physical_metrics(self):
        """Cập nhật chỉ số BMI và BSA."""
        height_cm = self.height_edit.value() 
        weight_kg = self.weight_edit.value()
        
        height_m = height_cm / 100.0  # Chuyển từ cm sang m
        
        # Tính BMI
        bmi = 0
        if height_m > 0 and weight_kg > 0:
            bmi = weight_kg / (height_m * height_m)
        
        # Tính BSA (Diện tích bề mặt cơ thể) theo công thức Mosteller
        bsa = 0
        if height_cm > 0 and weight_kg > 0:
            bsa = ((height_cm * weight_kg) / 3600) ** 0.5
        
        # Cập nhật nhãn
        self.bmi_label.setText(f"BMI: {bmi:.1f} kg/m²")
        self.bsa_label.setText(f"BSA: {bsa:.2f} m²")
        
        # Thêm màu sắc cho BMI để dễ nhận biết trạng thái
        if bmi < 18.5:
            self.bmi_label.setStyleSheet("color: orange;")  # Thiếu cân
        elif 18.5 <= bmi < 25:
            self.bmi_label.setStyleSheet("color: green;")   # Bình thường
        elif 25 <= bmi < 30:
            self.bmi_label.setStyleSheet("color: orange;")  # Thừa cân
        else:
            self.bmi_label.setStyleSheet("color: red;")     # Béo phì
    
    def _save_patient_info(self):
        """
        Lưu thông tin bệnh nhân vào cơ sở dữ liệu.
        """
        if not self.current_patient:
            QMessageBox.warning(
                self, "Cảnh báo", "Không có bệnh nhân nào được chọn")
            return
        
        try:
            # Lấy dữ liệu từ giao diện - thông tin cơ bản
            name = self.name_edit.text().strip()

            if not name:
                QMessageBox.warning(
                    self, "Cảnh báo", "Vui lòng nhập tên bệnh nhân")
                self.name_edit.setFocus()
                return
            
            birth_date = self.birth_date_edit.date().toString("yyyy-MM-dd")
            gender_map = {"Nam": "male", "Nữ": "female", "Khác": "other"}
            gender = gender_map[self.gender_combo.currentText()]
            
            address = self.address_edit.toPlainText().strip()
            phone = self.phone_edit.text().strip()
            email = self.email_edit.text().strip()
            notes = self.notes_edit.toPlainText().strip()
            
            # Thông tin y tế
            mrn = self.mrn_edit.text().strip()
            primary_physician = self.primary_physician_edit.text().strip()
            referring_physician = self.referring_physician_edit.text().strip()
            hospital_id = self.hospital_id_edit.text().strip()
            insurance_id = self.insurance_id_edit.text().strip()
            
            height_cm = self.height_edit.value()
            weight_kg = self.weight_edit.value()
            allergies = self.allergies_edit.toPlainText().strip()
            
            # Thông tin xạ trị
            diagnosis_code = self.diagnosis_code_edit.text().strip()
            diagnosis = self.diagnosis_edit.toPlainText().strip()
            
            site = self.site_combo.currentText() if self.site_combo.currentIndex() > 0 else ""
            technique = self.technique_combo.currentText() if self.technique_combo.currentIndex() > 0 else ""
            treatment_intent = self.treatment_intent_combo.currentText() if self.treatment_intent_combo.currentIndex() > 0 else ""

            # Lấy metadata hiện tại
            metadata = self.current_patient.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Cập nhật patient data
            patient_data = {
                'id': self.current_patient['id'],
                'name': name,
                'dob': birth_date,
                'birth_date': birth_date,  # Để tương thích với cả hai trường
                'gender': gender,
                'address': address,
                'phone': phone,
                'email': email,
                'notes': notes,
                
                # Thông tin y tế
                'mrn': mrn,
                'primary_physician': primary_physician,
                'referring_physician': referring_physician,
                'hospital_id': hospital_id,
                'insurance_id': insurance_id,
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'allergies': allergies,
                
                # Thông tin xạ trị
                'diagnosis_code': diagnosis_code,
                'diagnosis': diagnosis,
                'site': site,
                'technique': technique,
                'treatment_intent': treatment_intent,
                
                # Giữ metadata
                'metadata': metadata,
                
                # Cập nhật thời gian
                'updated_at': datetime.now().isoformat()
            }

            # Lưu vào cơ sở dữ liệu
            success = self.patient_db.update_patient(self.current_patient['id'], patient_data)
            
            if success:
                # Cập nhật thông tin bệnh nhân hiện tại
                updated_patient = self.patient_db.get_patient(
                    self.current_patient['id'])
                if updated_patient:
                    self.current_patient = updated_patient

                # Chuyển về chế độ đọc
                self._toggle_edit_mode(False)

                # Thông báo
                QMessageBox.information(
                    self, "Thành công", "Đã lưu thông tin bệnh nhân")

                # Kích hoạt tín hiệu cập nhật
                self.patient_updated.emit(self.current_patient['id'])

                # Ghi log
                logger.info(
                    f"Đã cập nhật thông tin bệnh nhân: {self.current_patient['id']}")
            else:
                QMessageBox.critical(
                    self, "Lỗi", "Không thể cập nhật thông tin bệnh nhân")
            
        except Exception as e:
            logger.error(
                f"Lỗi khi lưu thông tin bệnh nhân: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Lỗi", f"Không thể lưu thông tin bệnh nhân: {str(e)}")
    
    def _clear_patient_data(self):
        """
        Xóa dữ liệu bệnh nhân hiện tại và đặt lại giao diện.
        """
        # Đặt lại trạng thái
        self.current_patient = None
        
        # Xóa dữ liệu trên giao diện - thông tin cơ bản
        self.name_edit.clear()
        self.name_edit.setReadOnly(True)

        self.birth_date_edit.setDate(QDate.currentDate())
        self.birth_date_edit.setReadOnly(True)

        self.gender_combo.setCurrentIndex(0)
        self.gender_combo.setEnabled(False)
        
        self.address_edit.clear()
        self.address_edit.setReadOnly(True)
        
        self.phone_edit.clear()
        self.phone_edit.setReadOnly(True)
        
        self.email_edit.clear()
        self.email_edit.setReadOnly(True)

        self.notes_edit.clear()
        self.notes_edit.setReadOnly(True)
        
        # Xóa dữ liệu y tế
        self.mrn_edit.clear()
        self.mrn_edit.setReadOnly(True)
        
        self.primary_physician_edit.clear()
        self.primary_physician_edit.setReadOnly(True)
        
        self.referring_physician_edit.clear()
        self.referring_physician_edit.setReadOnly(True)
        
        self.hospital_id_edit.clear()
        self.hospital_id_edit.setReadOnly(True)
        
        self.insurance_id_edit.clear()
        self.insurance_id_edit.setReadOnly(True)
        
        self.height_edit.setValue(0)
        self.height_edit.setReadOnly(True)
        
        self.weight_edit.setValue(0)
        self.weight_edit.setReadOnly(True)
        
        self.allergies_edit.clear()
        self.allergies_edit.setReadOnly(True)
        
        # Đặt lại chỉ số BMI và BSA
        self.bmi_label.setText("BMI: ...")
        self.bsa_label.setText("BSA: ...")
        
        # Xóa dữ liệu xạ trị
        self.diagnosis_code_edit.clear()
        self.diagnosis_code_edit.setReadOnly(True)
        
        self.diagnosis_edit.clear()
        self.diagnosis_edit.setReadOnly(True)
        
        self.site_combo.setCurrentIndex(0)
        self.site_combo.setEnabled(False)
        
        self.technique_combo.setCurrentIndex(0)
        self.technique_combo.setEnabled(False)
        
        self.treatment_intent_combo.setCurrentIndex(0)
        self.treatment_intent_combo.setEnabled(False)

        # Vô hiệu hóa các nút
        self.delete_button.setEnabled(False)
        self.edit_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.import_images_button.setEnabled(False)
        self.view_images_button.setEnabled(False)
        self.add_history_button.setEnabled(False)

        # Xóa dữ liệu bảng
        self.history_table.setRowCount(0)
        self.studies_table.setRowCount(0)
        self.series_table.setRowCount(0)
        
        logger.info("Đã xóa dữ liệu bệnh nhân khỏi giao diện")

    def _populate_medical_history(self, history_data: List[Dict] = None):
        """Điền dữ liệu vào bảng lịch sử y tế."""
        self.history_table.setRowCount(0)
        
        if not history_data:
            history_data = []
            # Thử lấy lịch sử y tế từ bệnh nhân hiện tại nếu có
            if self.current_patient and 'medical_history' in self.current_patient:
                history_data = self.current_patient.get('medical_history', [])
        
        for entry in history_data:
            row_position = self.history_table.rowCount()
            self.history_table.insertRow(row_position)
            
            date_item = QTableWidgetItem(entry.get("date", ""))
            type_item = QTableWidgetItem(entry.get("type", ""))
            desc_item = QTableWidgetItem(entry.get("description", ""))
            doctor_item = QTableWidgetItem(entry.get("doctor", ""))
            
            self.history_table.setItem(row_position, 0, date_item)
            self.history_table.setItem(row_position, 1, type_item)
            self.history_table.setItem(row_position, 2, desc_item)
            self.history_table.setItem(row_position, 3, doctor_item)
    
    def _populate_medical_images(self):
        """Điền dữ liệu hình ảnh y tế vào các bảng."""
        if not self.current_patient:
            self.studies_table.setRowCount(0)
            self.series_table.setRowCount(0)
            return
        
        try:
            # Lấy danh sách nghiên cứu của bệnh nhân
            studies = self.patient_db.get_patient_studies(self.current_patient['id'])
            
            # Kiểm tra nếu không có nghiên cứu nào
            if not studies:
                self.studies_table.setRowCount(0)
                logger.info(f"Không có nghiên cứu nào cho bệnh nhân {self.current_patient['id']}")
            return
        
            # Điền dữ liệu nghiên cứu vào bảng
            self.studies_table.setRowCount(len(studies))
            
            for i, study in enumerate(studies):
                # Lấy số lượng series cho mỗi nghiên cứu
                series_count = 0
                try:
                    series_list = self.patient_db.get_study_series(study['id'])
                    series_count = len(series_list) if series_list else 0
                except Exception as e:
                    logger.warning(f"Không thể lấy danh sách series cho nghiên cứu {study['id']}: {str(e)}")
                
                # Thêm dữ liệu vào bảng
                for j, column in enumerate(['Mô tả', 'Ngày', 'Loại', 'Số lượng series']):
                    item = QtWidgets.QTableWidgetItem()
                    
                    if j == 0:
                        # Cột mô tả
                        value = study.get('description', 'Không có mô tả')
                        item.setText(value)
                        # Lưu ID nghiên cứu vào item để sử dụng sau này
                        item.setData(QtCore.Qt.UserRole, study['id'])
                    elif j == 1:
                        # Cột ngày
                        item.setText(study.get('date', 'N/A'))
                    elif j == 2:
                        # Cột loại
                        item.setText(study.get('modality', 'N/A'))
                    elif j == 3:
                        # Cột số lượng series
                        item.setText(str(series_count))
                        
                    self.studies_table.setItem(i, j, item)
                    
            # Điều chỉnh độ rộng các cột
            self.studies_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            for i in range(1, self.studies_table.columnCount()):
                self.studies_table.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
                
            # Chọn nghiên cứu đầu tiên
            if self.studies_table.rowCount() > 0:
                self.studies_table.selectRow(0)
                self._on_study_selected()
                
        except Exception as e:
            logger.error(f"Lỗi khi điền dữ liệu hình ảnh y tế: {str(e)}", exc_info=True)

    def _import_patient_data(self):
        """
        Nhập dữ liệu DICOM cho bệnh nhân hiện tại.
        """
        if not self.current_patient:
            QMessageBox.warning(
                self, "Lỗi", "Vui lòng chọn bệnh nhân trước khi nhập dữ liệu.")
            return

        # Mở dialog chọn thư mục
        directory = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục chứa dữ liệu DICOM", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not directory:
            return

        # Hiển thị dialog tiến trình
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Đang nhập dữ liệu DICOM")
        progress_dialog.setFixedSize(400, 100)

        progress_layout = QVBoxLayout(progress_dialog)
        progress_label = QLabel("Đang xử lý dữ liệu DICOM, vui lòng chờ...")
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # Chế độ không xác định

        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(progress_bar)

        progress_dialog.show()

        try:
            # Nhập dữ liệu DICOM
            patient_id = self.current_patient['id']
            self.dicom_importer.import_for_patient(directory, patient_id)

            # Cập nhật dữ liệu hiển thị
            self._populate_medical_images()

            QMessageBox.information(
                self, "Thành công", "Đã nhập dữ liệu DICOM thành công.")

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Lỗi khi nhập dữ liệu DICOM: {str(e)}")

        finally:
            progress_dialog.close()

    def _export_patient_data(self):
        """Xuất dữ liệu bệnh nhân ra một thư mục."""
        if not self.current_patient:
            QtWidgets.QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn bệnh nhân trước khi xuất dữ liệu.")
            return

        # Kiểm tra xem có dữ liệu để xuất không
        patient_id = self.current_patient['id']
        studies = self.patient_db.get_patient_studies(
            patient_id, include_series=True)

        if not studies:
            QtWidgets.QMessageBox.information(self, "Thông báo", 
                                            f"Bệnh nhân {self.current_patient['name']} không có dữ liệu y tế để xuất.")
            return

        # Chọn thư mục đích
        export_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Chọn thư mục xuất dữ liệu", 
            os.path.expanduser("~"),
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
            
        if not export_dir:
            return

        # Tạo thư mục cho bệnh nhân
        patient_dir = os.path.join(export_dir, f"{patient_id}_{self.current_patient['name'].replace(' ', '_')}")
        os.makedirs(patient_dir, exist_ok=True)
            
        # Xuất thông tin bệnh nhân
        patient_info_path = os.path.join(patient_dir, "patient_info.json")
        with open(patient_info_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_patient, f, ensure_ascii=False, indent=4)
            
        # Hiển thị dialog tiến trình
        progress = QtWidgets.QProgressDialog("Đang xuất dữ liệu bệnh nhân...", "Hủy", 0, 100, self)
        progress.setWindowTitle("Xuất dữ liệu")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()
            
        # Đếm tổng số series để tính tiến trình
        total_series = sum(len(study.get('series', [])) for study in studies)
        if total_series == 0:
            progress.setValue(100)
            QtWidgets.QMessageBox.information(self, "Thông báo", 
                                            f"Đã xuất thông tin bệnh nhân ra: {patient_info_path}")
            return
            
        # Xuất các file DICOM
        try:
            series_processed = 0
            
            for study in studies:
                study_dir = os.path.join(patient_dir, study.get('description', f"Study_{study['id']}").replace(' ', '_'))
                os.makedirs(study_dir, exist_ok=True)

                # Xuất thông tin nghiên cứu
                study_info_path = os.path.join(study_dir, "study_info.json")
                with open(study_info_path, 'w', encoding='utf-8') as f:
                    json.dump(study, f, ensure_ascii=False, indent=4)
                
                for series in study.get('series', []):
                    series_processed += 1
                    progress.setValue(int(series_processed * 100 / total_series))
                    
                    if progress.wasCanceled():
                        break
                        
                    # Tạo thư mục cho series
                    series_dir = os.path.join(study_dir, series.get('description', f"Series_{series['id']}").replace(' ', '_'))
                    os.makedirs(series_dir, exist_ok=True)

                    # Lấy danh sách các file trong series
                    files = self.patient_db.get_series_files(series['id'])
                    
                    for file_data in files:
                        file_path = file_data.get('file_path')
                        if file_path and os.path.exists(file_path):
                            # Sao chép file vào thư mục đích
                            dest_path = os.path.join(series_dir, os.path.basename(file_path))
                            shutil.copy2(file_path, dest_path)

                if progress.wasCanceled():
                    break
                    
            progress.setValue(100)
            QtWidgets.QMessageBox.information(self, "Thông báo", 
                                            f"Đã xuất dữ liệu bệnh nhân ra: {patient_dir}")

        except Exception as e:
            progress.cancel()
            logger.error(f"Lỗi khi xuất dữ liệu bệnh nhân: {str(e)}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Không thể xuất dữ liệu bệnh nhân: {str(e)}")

    def _import_medical_images(self):
        """
        Nhập dữ liệu DICOM cho bệnh nhân hiện tại.
        """
        self._import_patient_data()

    def _on_study_selected(self):
        """
        Xử lý khi người dùng chọn một nghiên cứu.
        """
        self._update_series_table()

    def _update_series_table(self):
        """Cập nhật bảng series dựa trên nghiên cứu đã chọn."""
        selected_rows = self.studies_table.selectionModel().selectedRows()
        
        if not selected_rows:
            self.series_table.setRowCount(0)
            return

        # Lấy study_id của nghiên cứu được chọn
        study_row = selected_rows[0].row()
        study_id = self.studies_table.item(study_row, 0).data(QtCore.Qt.UserRole)

        try:
            # Lấy danh sách series
            series_list = self.patient_db.get_study_series(study_id)
            
            if not series_list:
                logger.info(f"Không có series nào cho nghiên cứu {study_id}")
                self.series_table.setRowCount(0)
                return

            # Điền dữ liệu vào bảng series
            self.series_table.setRowCount(len(series_list))
            
            for i, series in enumerate(series_list):
                # Thêm dữ liệu vào bảng
                for j, column in enumerate(['Mô tả', 'Loại', 'Số series', 'Vùng cơ thể']):
                    item = QtWidgets.QTableWidgetItem()
                    
                    if j == 0:
                        # Cột mô tả
                        value = series.get('description', 'Không có mô tả')
                        item.setText(value)
                        # Lưu ID series vào item để sử dụng sau này
                        item.setData(QtCore.Qt.UserRole, series['id'])
                    elif j == 1:
                        # Cột loại
                        item.setText(series.get('modality', 'N/A'))
                    elif j == 2:
                        # Cột số series
                        item.setText(str(series.get('series_number', 'N/A')))
                    elif j == 3:
                        # Cột vùng cơ thể
                        item.setText(series.get('body_part', 'N/A'))
                        
                    self.series_table.setItem(i, j, item)
                    
            # Điều chỉnh độ rộng các cột
            self.series_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            for i in range(1, self.series_table.columnCount()):
                self.series_table.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
                
            # Chọn series đầu tiên
            if self.series_table.rowCount() > 0:
                self.series_table.selectRow(0)
            
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật bảng series: {str(e)}", exc_info=True)

    def _view_medical_images(self):
        """Mở cửa sổ xem hình ảnh cho series được chọn."""
        # Kiểm tra xem có series nào được chọn không
        selected_rows = self.series_table.selectionModel().selectedRows()
        if not selected_rows:
            QtWidgets.QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn series để xem hình ảnh.")
            return

        # Lấy ID series
        series_row = selected_rows[0].row()
        series_id = self.series_table.item(series_row, 0).data(QtCore.Qt.UserRole)
        
        # Lấy ID nghiên cứu
        study_row = self.studies_table.selectionModel().selectedRows()[0].row()
        study_id = None
            
        if self.studies_table.item(study_row, 0):
            study_id = self.studies_table.item(study_row, 0).data(QtCore.Qt.UserRole)

            studies = self.patient_db.get_patient_studies(self.current_patient['id'], include_series=True)
            
            for study in studies:
                if study_id and study['id'] != study_id:
                    continue
                    
                for series in study.get('series', []):
                    if series['id'] == series_id:
                        # Lấy danh sách file trong series
                        files = self.patient_db.get_series_files(series_id)
                        
                        if not files:
                            QtWidgets.QMessageBox.warning(self, "Cảnh báo", 
                                                       f"Không tìm thấy file nào trong series {series.get('description', series_id)}.")
                        return
                        
                    # Tạo danh sách đường dẫn file
                    file_paths = []
                    for file_data in files:
                        file_path = file_data.get('file_path')
                        if file_path and os.path.exists(file_path):
                            file_paths.append(file_path)
                            
                    if not file_paths:
                        QtWidgets.QMessageBox.warning(self, "Cảnh báo", 
                                                   f"Không tìm thấy file nào trong series {series.get('description', series_id)}.")
                        return

                    # Tạo signal để yêu cầu main window hiển thị hình ảnh
                    modality = series.get('modality', 'Unknown')
                    description = series.get('description', f"Series {series_id}")
                    
                    try:
                        # Tìm parent widget là main window
                        main_window = None
                        parent = self.parent()
                        while parent is not None:
                            if hasattr(parent, 'show_medical_images'):
                                main_window = parent
                                break
                            parent = parent.parent()
                            
                        if main_window is not None:
                            main_window.show_medical_images(
                                file_paths, modality, description, 
                                patient_name=self.current_patient['name'],
                                patient_id=self.current_patient['id']
                            )
                        else:
                            QtWidgets.QMessageBox.warning(self, "Cảnh báo", 
                                                     "Không thể mở cửa sổ xem hình ảnh.")
                
                    except Exception as e:
                        logger.error(f"Lỗi khi mở cửa sổ xem hình ảnh: {str(e)}", exc_info=True)
                        QtWidgets.QMessageBox.critical(self, "Lỗi", f"Không thể mở cửa sổ xem hình ảnh: {str(e)}")
                        
                    return
                    
        QtWidgets.QMessageBox.warning(self, "Cảnh báo", f"Không tìm thấy series với ID {series_id}.")

    def _search_patients(self):
        """Tìm kiếm bệnh nhân theo tên hoặc ID"""
        search_text = self.search_input.text().strip()
        
        if not search_text:
            QMessageBox.warning(self, "Thông báo", "Vui lòng nhập tên hoặc mã bệnh nhân")
            return

        try:
            patients = self.patient_db.search_patients(search_text)

            if not patients:
                QMessageBox.information(self, "Kết quả tìm kiếm", "Không tìm thấy bệnh nhân phù hợp")
                return
                
            # Hiển thị danh sách bệnh nhân nếu có nhiều kết quả
            if len(patients) > 1:
                dialog = PatientSearchResultDialog(patients, self)
                if dialog.exec_() == QDialog.Accepted and dialog.selected_patient_id:
                    self.set_patient(dialog.selected_patient_id)
            else:
                # Nếu chỉ có một kết quả, hiển thị ngay
                self.set_patient(patients[0]['id'])
                
        except Exception as e:
            logger.exception("Error during patient search: %s", str(e))
            QMessageBox.critical(self, "Lỗi", f"Không thể tìm kiếm bệnh nhân: {str(e)}")

    def _create_new_patient(self):
        """Tạo bệnh nhân mới"""
        try:
            dialog = NewPatientDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                patient_data = dialog.get_patient_data()
                
                # Thêm thông tin thời gian
                now = datetime.now().isoformat()
                patient_data.update({
                    'created_at': now,
                    'updated_at': now
                })
                
                # Tạo ID mới cho bệnh nhân
                patient_id = str(uuid.uuid4())
                patient_data['id'] = patient_id
                
                # Lưu vào cơ sở dữ liệu
                self.patient_db.add_patient(patient_data)
                
                # Cập nhật UI
                self.set_patient(patient_id)
                
                # Emit tín hiệu tạo bệnh nhân mới
                self.patient_created.emit(patient_id)
                
                QMessageBox.information(self, "Thành công", "Đã tạo bệnh nhân mới")
        except Exception as e:
            logger.exception("Error creating new patient: %s", str(e))
            QMessageBox.critical(self, "Lỗi", f"Không thể tạo bệnh nhân mới: {str(e)}")

    def _delete_current_patient(self):
        """Xóa bệnh nhân hiện tại"""
        if not self.current_patient:
            QMessageBox.warning(self, "Cảnh báo", "Không có bệnh nhân nào được chọn.")
            return
            
        # Xác nhận xóa
        confirm = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa bệnh nhân này không?\nToàn bộ dữ liệu liên quan sẽ bị xóa vĩnh viễn.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm != QMessageBox.Yes:
            return
            
        try:
            # Lưu ID trước khi xóa
            patient_id = self.current_patient['id']
            
            # Xóa bệnh nhân
            success = self.patient_db.delete_patient(self.current_patient['id'])
            
            if success:
                # Làm mới UI
                self._clear_patient_data()
                self.current_patient = None
                
                # Emit tín hiệu đã xóa bệnh nhân
                self.patient_deleted.emit(patient_id)
                
                # Thông báo thành công
                QMessageBox.information(self, "Thành công", "Đã xóa bệnh nhân.")
            else:
                QMessageBox.critical(self, "Lỗi", "Không thể xóa bệnh nhân.")

        except Exception as e:
            logger.exception("Error deleting patient: %s", str(e))
            QMessageBox.critical(self, "Lỗi", f"Không thể xóa bệnh nhân: {str(e)}")

    def _add_medical_history(self):
        """Thêm mục lịch sử bệnh mới"""
        if not self.current_patient:
            return
            
        # Tạo dialog đơn giản để nhập thông tin
        dialog = QDialog(self)
        dialog.setWindowTitle("Thêm lịch sử bệnh")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        
        # Ngày
        date_edit = QDateEdit()
        date_edit.setCalendarPopup(True)
        date_edit.setDate(QDate.currentDate())
        form_layout.addRow("Ngày:", date_edit)
        
        # Chẩn đoán
        diagnosis_edit = QLineEdit()
        form_layout.addRow("Chẩn đoán:", diagnosis_edit)
        
        # Bác sĩ
        physician_edit = QLineEdit()
        form_layout.addRow("Bác sĩ:", physician_edit)
        
        # Ghi chú
        notes_edit = QTextEdit()
        notes_edit.setMaximumHeight(100)
        form_layout.addRow("Ghi chú:", notes_edit)
        
        layout.addLayout(form_layout)
        
        # Nút OK/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                # Lấy thông tin từ dialog
                history_entry = {
                    "date": date_edit.date().toString("yyyy-MM-dd"),
                    "diagnosis": diagnosis_edit.text(),
                    "physician": physician_edit.text(),
                    "notes": notes_edit.toPlainText()
                }
                
                # Thêm vào dữ liệu bệnh nhân
                patient_data = self.current_patient.copy()
                
                if "medical_history" not in patient_data:
                    patient_data["medical_history"] = []
                
                patient_data["medical_history"].append(history_entry)
                
                # Cập nhật bệnh nhân
                success = self.patient_db.update_patient(
                    patient_data["id"], 
                    {"medical_history": patient_data["medical_history"]}
                )
                
                if success:
                    # Cập nhật hiển thị
                    self._populate_medical_history(patient_data["medical_history"])
                    QMessageBox.information(self, "Thành công", "Đã thêm lịch sử bệnh mới.")
                else:
                    QMessageBox.critical(self, "Lỗi", "Không thể cập nhật thông tin bệnh nhân.")
            
            except Exception as e:
                logger.exception("Error adding medical history: %s", str(e))
                QMessageBox.critical(self, "Lỗi", f"Không thể thêm lịch sử bệnh: {str(e)}")

