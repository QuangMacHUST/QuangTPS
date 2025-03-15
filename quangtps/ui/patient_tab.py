#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab thông tin bệnh nhân (Patient Tab) cho QuangTPS.

Module này cung cấp giao diện để hiển thị và chỉnh sửa thông tin bệnh nhân,
bao gồm thông tin cá nhân, lịch sử bệnh, và các hồ sơ y tế liên quan.
"""

import logging
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QComboBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QScrollArea, QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


class PatientTab(QWidget):
    """
    Tab hiển thị và chỉnh sửa thông tin bệnh nhân.
    
    Tab này bao gồm các phần thông tin cá nhân, lịch sử bệnh,
    kết quả khám lâm sàng, hình ảnh y tế, và các dữ liệu khác
    liên quan đến bệnh nhân.
    """
    
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
        
        # Thiết lập giao diện
        self._init_ui()
        
        logger.info("Khởi tạo tab thông tin bệnh nhân hoàn tất")
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Tab widget cho các phần thông tin khác nhau
        self.info_tabs = QTabWidget()
        self.main_layout.addWidget(self.info_tabs)
        
        # Tab thông tin cơ bản
        self.basic_info_widget = QWidget()
        self.basic_info_layout = QVBoxLayout(self.basic_info_widget)
        
        # Nhóm thông tin cá nhân
        self.personal_group = QGroupBox("Thông tin cá nhân")
        self.personal_layout = QFormLayout(self.personal_group)
        
        # Các trường thông tin
        self.patient_id_field = QLineEdit()
        self.patient_id_field.setReadOnly(True)
        self.personal_layout.addRow("Mã bệnh nhân:", self.patient_id_field)
        
        self.full_name_field = QLineEdit()
        self.personal_layout.addRow("Họ và tên:", self.full_name_field)
        
        self.dob_field = QDateEdit()
        self.dob_field.setDisplayFormat("dd/MM/yyyy")
        self.dob_field.setCalendarPopup(True)
        self.personal_layout.addRow("Ngày sinh:", self.dob_field)
        
        self.gender_field = QComboBox()
        self.gender_field.addItems(["Nam", "Nữ", "Khác"])
        self.personal_layout.addRow("Giới tính:", self.gender_field)
        
        self.id_number_field = QLineEdit()
        self.personal_layout.addRow("Số CMND/CCCD:", self.id_number_field)
        
        self.phone_field = QLineEdit()
        self.personal_layout.addRow("Điện thoại:", self.phone_field)
        
        self.email_field = QLineEdit()
        self.personal_layout.addRow("Email:", self.email_field)
        
        self.address_field = QLineEdit()
        self.personal_layout.addRow("Địa chỉ:", self.address_field)
        
        # Thêm nhóm thông tin cá nhân vào layout
        self.basic_info_layout.addWidget(self.personal_group)
        
        # Nhóm thông tin y tế
        self.medical_group = QGroupBox("Thông tin y tế")
        self.medical_layout = QFormLayout(self.medical_group)
        
        self.blood_type_field = QComboBox()
        self.blood_type_field.addItems(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Không biết"])
        self.medical_layout.addRow("Nhóm máu:", self.blood_type_field)
        
        self.allergies_field = QLineEdit()
        self.medical_layout.addRow("Dị ứng:", self.allergies_field)
        
        self.height_field = QDoubleSpinBox()
        self.height_field.setRange(0, 250)
        self.height_field.setSuffix(" cm")
        self.medical_layout.addRow("Chiều cao:", self.height_field)
        
        self.weight_field = QDoubleSpinBox()
        self.weight_field.setRange(0, 300)
        self.weight_field.setSuffix(" kg")
        self.medical_layout.addRow("Cân nặng:", self.weight_field)
        
        # Thêm nhóm thông tin y tế vào layout
        self.basic_info_layout.addWidget(self.medical_group)
        
        # Nút lưu thông tin
        self.save_button = QPushButton("Lưu thông tin")
        self.save_button.clicked.connect(self._save_patient_info)
        self.basic_info_layout.addWidget(self.save_button, alignment=Qt.AlignRight)
        
        # Thêm tab thông tin cơ bản
        self.info_tabs.addTab(self.basic_info_widget, "Thông tin cơ bản")
        
        # Tab lịch sử bệnh
        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        
        # Nhóm chẩn đoán
        self.diagnosis_group = QGroupBox("Chẩn đoán")
        self.diagnosis_layout = QFormLayout(self.diagnosis_group)
        
        self.disease_field = QLineEdit()
        self.diagnosis_layout.addRow("Bệnh chẩn đoán:", self.disease_field)
        
        self.site_field = QComboBox()
        self.site_field.addItems(["Đầu & Cổ", "Não", "Ngực", "Phổi", "Tuyến tiền liệt", "Gan", "Khác"])
        self.diagnosis_layout.addRow("Vị trí:", self.site_field)
        
        self.stage_field = QComboBox()
        self.stage_field.addItems(["I", "II", "III", "IV", "Không xác định"])
        self.diagnosis_layout.addRow("Giai đoạn:", self.stage_field)
        
        self.diagnosis_date_field = QDateEdit()
        self.diagnosis_date_field.setDisplayFormat("dd/MM/yyyy")
        self.diagnosis_date_field.setCalendarPopup(True)
        self.diagnosis_layout.addRow("Ngày chẩn đoán:", self.diagnosis_date_field)
        
        self.history_layout.addWidget(self.diagnosis_group)
        
        # Nhóm tiền sử
        self.history_group = QGroupBox("Tiền sử")
        self.history_layout_form = QFormLayout(self.history_group)
        
        self.medical_history_field = QTextEdit()
        self.history_layout_form.addRow("Tiền sử bệnh:", self.medical_history_field)
        
        self.family_history_field = QTextEdit()
        self.history_layout_form.addRow("Tiền sử gia đình:", self.family_history_field)
        
        self.history_layout.addWidget(self.history_group)
        
        # Nhóm điều trị trước đó
        self.previous_group = QGroupBox("Điều trị trước đó")
        self.previous_layout = QFormLayout(self.previous_group)
        
        self.surgery_field = QTextEdit()
        self.previous_layout.addRow("Phẫu thuật:", self.surgery_field)
        
        self.chemo_field = QTextEdit()
        self.previous_layout.addRow("Hóa trị:", self.chemo_field)
        
        self.radio_field = QTextEdit()
        self.previous_layout.addRow("Xạ trị:", self.radio_field)
        
        self.history_layout.addWidget(self.previous_group)
        
        # Nút lưu lịch sử
        self.save_history_button = QPushButton("Lưu lịch sử")
        self.save_history_button.clicked.connect(self._save_patient_history)
        self.history_layout.addWidget(self.save_history_button, alignment=Qt.AlignRight)
        
        # Thêm tab lịch sử bệnh
        self.info_tabs.addTab(self.history_widget, "Lịch sử bệnh")
        
        # Tab kết quả xét nghiệm
        self.lab_results_widget = QWidget()
        self.lab_results_layout = QVBoxLayout(self.lab_results_widget)
        
        # Bảng kết quả xét nghiệm
        self.lab_results_table = QTableWidget(0, 4)
        self.lab_results_table.setHorizontalHeaderLabels(["Ngày", "Loại xét nghiệm", "Kết quả", "Ghi chú"])
        self.lab_results_table.horizontalHeader().setStretchLastSection(True)
        self.lab_results_layout.addWidget(self.lab_results_table)
        
        # Nút thêm kết quả xét nghiệm
        self.add_lab_button = QPushButton("Thêm kết quả")
        self.add_lab_button.clicked.connect(self._add_lab_result)
        self.lab_results_layout.addWidget(self.add_lab_button, alignment=Qt.AlignRight)
        
        # Thêm tab kết quả xét nghiệm
        self.info_tabs.addTab(self.lab_results_widget, "Kết quả xét nghiệm")
        
        # Tab hình ảnh
        self.images_widget = QWidget()
        self.images_layout = QVBoxLayout(self.images_widget)
        
        # Bảng danh sách hình ảnh
        self.images_table = QTableWidget(0, 3)
        self.images_table.setHorizontalHeaderLabels(["Ngày", "Loại hình ảnh", "Mô tả"])
        self.images_table.horizontalHeader().setStretchLastSection(True)
        self.images_layout.addWidget(self.images_table)
        
        # Nút thêm hình ảnh
        self.add_image_button = QPushButton("Thêm hình ảnh")
        self.add_image_button.clicked.connect(self._add_image)
        self.images_layout.addWidget(self.add_image_button, alignment=Qt.AlignRight)
        
        # Thêm tab hình ảnh
        self.info_tabs.addTab(self.images_widget, "Hình ảnh")
    
    def set_patient(self, patient):
        """
        Thiết lập bệnh nhân hiện tại và cập nhật giao diện.
        
        Parameters
        ----------
        patient : Any
            Đối tượng bệnh nhân
        """
        self.current_patient = patient
        if patient:
            self._populate_patient_data()
        else:
            self._clear_patient_data()
    
    def _populate_patient_data(self):
        """Điền thông tin bệnh nhân vào giao diện."""
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
        pass
    
    def _clear_patient_data(self):
        """Xóa thông tin bệnh nhân khỏi giao diện."""
        # Xóa thông tin cá nhân
        self.patient_id_field.clear()
        self.full_name_field.clear()
        self.dob_field.setDate(QDate.currentDate())
        self.gender_field.setCurrentIndex(0)
        self.id_number_field.clear()
        self.phone_field.clear()
        self.email_field.clear()
        self.address_field.clear()
        
        # Xóa thông tin y tế
        self.blood_type_field.setCurrentIndex(0)
        self.allergies_field.clear()
        self.height_field.setValue(0)
        self.weight_field.setValue(0)
        
        # Xóa thông tin chẩn đoán
        self.disease_field.clear()
        self.site_field.setCurrentIndex(0)
        self.stage_field.setCurrentIndex(0)
        self.diagnosis_date_field.setDate(QDate.currentDate())
        
        # Xóa tiền sử
        self.medical_history_field.clear()
        self.family_history_field.clear()
        
        # Xóa thông tin điều trị trước đó
        self.surgery_field.clear()
        self.chemo_field.clear()
        self.radio_field.clear()
        
        # Xóa bảng kết quả xét nghiệm
        self.lab_results_table.setRowCount(0)
        
        # Xóa bảng hình ảnh
        self.images_table.setRowCount(0)
    
    def _save_patient_info(self):
        """Lưu thông tin cá nhân của bệnh nhân."""
        logger.info("Lưu thông tin bệnh nhân")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _save_patient_history(self):
        """Lưu lịch sử bệnh của bệnh nhân."""
        logger.info("Lưu lịch sử bệnh")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _add_lab_result(self):
        """Thêm kết quả xét nghiệm mới."""
        logger.info("Thêm kết quả xét nghiệm")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _add_image(self):
        """Thêm hình ảnh mới."""
        logger.info("Thêm hình ảnh")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
