#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab thông tin bệnh nhân (Patient Tab) cho QuangTPS.

Module này cung cấp giao diện để hiển thị và chỉnh sửa thông tin bệnh nhân,
bao gồm thông tin cá nhân, lịch sử bệnh, và các hồ sơ y tế liên quan.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import shutil

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QComboBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QScrollArea, QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QFileDialog, QDialog, QHeaderView, QProgressBar,
    QInputDialog
)
from PyQt5.QtCore import Qt, QDate, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

from quangtps.database.patient_db import PatientDatabase
from quangtps.dicom.dicom_importer import DicomImporter
from quangtps.dicom.dicom_exporter import DicomExporter
from quangtps.core.patient import Patient

logger = logging.getLogger(__name__)


class NewPatientDialog(QDialog):
    """
    Dialog nhập thông tin bệnh nhân mới.
    """

    def __init__(self, parent=None):
        """Khởi tạo dialog."""
        super().__init__(parent)
        self.setWindowTitle("Tạo bệnh nhân mới")
        self.setMinimumWidth(500)
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện của dialog."""
        layout = QVBoxLayout(self)

        # Form nhập thông tin
        form_group = QGroupBox("Thông tin bệnh nhân")
        form_layout = QFormLayout(form_group)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Nhập họ tên bệnh nhân")

        self.birth_date_edit = QDateEdit()
        self.birth_date_edit.setDate(QDate.currentDate())
        self.birth_date_edit.setCalendarPopup(True)

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Nam", "Nữ", "Khác"])

        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("Nhập mã bệnh nhân (nếu có)")

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Ghi chú thêm về bệnh nhân")
        self.notes_edit.setMaximumHeight(100)

        form_layout.addRow("Họ tên (*)", self.name_edit)
        form_layout.addRow("Ngày sinh", self.birth_date_edit)
        form_layout.addRow("Giới tính", self.gender_combo)
        form_layout.addRow("Mã bệnh nhân", self.id_edit)
        form_layout.addRow("Ghi chú", self.notes_edit)

        layout.addWidget(form_group)

        # Nút bấm
        button_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Hủy")
        self.cancel_btn.clicked.connect(self.reject)

        self.create_btn = QPushButton("Tạo")
        self.create_btn.setDefault(True)
        self.create_btn.clicked.connect(self._validate_and_accept)

        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.create_btn)

        layout.addLayout(button_layout)

    def _validate_and_accept(self):
        """Kiểm tra dữ liệu trước khi tạo bệnh nhân mới."""
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập họ tên bệnh nhân.")
            self.name_edit.setFocus()
            return

        self.accept()

    def get_patient_data(self):
        """
        Lấy dữ liệu bệnh nhân từ các trường nhập.

        Returns:
            dict: Dữ liệu bệnh nhân
        """
        gender_map = {"Nam": "male", "Nữ": "female", "Khác": "other"}

        metadata = {
            "notes": self.notes_edit.toPlainText(),
            "external_id": self.id_edit.text().strip()
        }

        return {
            "name": self.name_edit.text().strip(),
            "birth_date": self.birth_date_edit.date().toString("yyyy-MM-dd"),
            "gender": gender_map[self.gender_combo.currentText()],
            "metadata": metadata
        }


class PatientTab(QWidget):
    """
    Tab hiển thị và chỉnh sửa thông tin bệnh nhân.
    
    Tab này bao gồm các phần thông tin cá nhân, lịch sử bệnh,
    kết quả khám lâm sàng, hình ảnh y tế, và các dữ liệu khác
    liên quan đến bệnh nhân.
    """
    
    # Tín hiệu để thông báo khi cập nhật dữ liệu bệnh nhân
    patient_updated = pyqtSignal(str)  # patient_id
    patient_created = pyqtSignal(str)
    
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
        self.patient_db = PatientDatabase()
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

        self.notes_edit = QTextEdit()
        self.notes_edit.setReadOnly(True)

        form_layout.addRow("Họ tên:", self.name_edit)
        form_layout.addRow("Ngày sinh:", self.birth_date_edit)
        form_layout.addRow("Giới tính:", self.gender_combo)
        form_layout.addRow("Ghi chú:", self.notes_edit)

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

        basic_info_layout.addWidget(form_group)
        basic_info_layout.addLayout(edit_layout)

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
        self.stacked_widget.addTab(self.medical_history_tab, "Lịch sử bệnh")
        self.stacked_widget.addTab(self.medical_data_tab, "Dữ liệu y tế")

        layout.addWidget(self.stacked_widget)
        
        # Vô hiệu hóa các widget khi không có bệnh nhân được chọn
        self._toggle_edit_mode(False)
        
        # Kết nối tín hiệu
        self.name_edit.textChanged.connect(self._on_patient_data_changed)
        
        logger.info("Khởi tạo giao diện tab bệnh nhân hoàn tất")
    
    def _toggle_edit_mode(self, enabled: bool):
        """
        Bật/tắt chế độ chỉnh sửa thông tin bệnh nhân.

        Parameters
        ----------
        enabled : bool
            True để bật chế độ chỉnh sửa, False để tắt
        """
        # Thiết lập trạng thái của các trường nhập liệu
        self.name_edit.setReadOnly(not enabled)
        self.birth_date_edit.setReadOnly(not enabled)
        self.gender_combo.setEnabled(enabled)
        self.notes_edit.setReadOnly(not enabled)

        # Cập nhật trạng thái các nút
        self.edit_button.setEnabled(not enabled)
        self.save_button.setEnabled(enabled)
        self.cancel_button.setEnabled(enabled)
    
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
            ID của bệnh nhân cần hiển thị
        """
        try:
            # Lấy thông tin bệnh nhân
            patient_data = self.patient_db.get_patient(patient_id)

            if not patient_data:
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
    
    def _clear_patient_data(self):
        """
        Xóa dữ liệu bệnh nhân hiện tại và đặt lại giao diện.
        """
        # Đặt lại trạng thái
        self.current_patient = None
        
        # Xóa dữ liệu trên giao diện
        self.name_edit.clear()
        self.name_edit.setReadOnly(True)

        self.birth_date_edit.setDate(QDate.currentDate())
        self.birth_date_edit.setReadOnly(True)

        self.gender_combo.setCurrentIndex(0)
        self.gender_combo.setEnabled(False)

        self.notes_edit.clear()
        self.notes_edit.setReadOnly(True)

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

            # Ngày sinh
            birth_date = patient.get('birth_date', '')
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

            # Ghi chú từ metadata
            if 'metadata' in patient and patient['metadata']:
                metadata = patient['metadata']
                notes = metadata.get('notes', '')
                self.notes_edit.setText(notes)

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
    
    def _populate_medical_history(self, history_data: List[Dict]):
        """Điền dữ liệu vào bảng lịch sử y tế."""
        self.history_table.setRowCount(0)
        
        if not history_data:
            return
        
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
    
    def _populate_medical_images(self, images_data: List[Dict]):
        """Điền dữ liệu hình ảnh y tế."""
        # Xóa tất cả widget con hiện tại
        while self.images_list_layout.count():
            item = self.images_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not images_data:
            return
        
        for image_data in images_data:
            # Tạo widget hiển thị hình ảnh
            image_widget = QWidget()
            image_layout = QVBoxLayout(image_widget)
            
            # Label hiển thị hình ảnh
            image_label = QLabel()
            image_label.setFixedSize(150, 150)
            image_label.setScaledContents(True)
            
            # TODO: Cần cải thiện cách hiển thị hình ảnh thực tế
            # Hiện tại chỉ hiển thị icon mẫu
            image_label.setPixmap(QPixmap(":/icons/image.png"))
            
            # Label hiển thị thông tin hình ảnh
            info_label = QLabel(
                f"{image_data.get('type', 'Không rõ')} - {image_data.get('date', '')}")
            
            image_layout.addWidget(image_label)
            image_layout.addWidget(info_label)
            
            self.images_list_layout.addWidget(image_widget)
    
    def _save_patient_info(self):
        """
        Lưu thông tin bệnh nhân vào cơ sở dữ liệu.
        """
        if not self.current_patient:
            QMessageBox.warning(
                self, "Cảnh báo", "Không có bệnh nhân nào được chọn")
            return
        
        try:
            # Lấy dữ liệu từ giao diện
            name = self.name_edit.text().strip()

            if not name:
                QMessageBox.warning(
                    self, "Cảnh báo", "Vui lòng nhập tên bệnh nhân")
                self.name_edit.setFocus()
                return
            
            birth_date = self.birth_date_edit.date().toString("yyyy-MM-dd")
            gender_map = {"Nam": "male", "Nữ": "female", "Khác": "other"}
            gender = gender_map[self.gender_combo.currentText()]
            notes = self.notes_edit.toPlainText().strip()

            # Lấy metadata hiện tại
            metadata = self.current_patient.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Cập nhật notes trong metadata
            metadata['notes'] = notes

            # Lưu vào cơ sở dữ liệu
            success = self.patient_db.update_patient(
                self.current_patient['id'],
                name=name,
                birth_date=birth_date,
                gender=gender,
                metadata=metadata
            )
            
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
    
    def _get_medical_history_data(self) -> List[Dict]:
        """Lấy dữ liệu từ bảng lịch sử y tế."""
        history_data = []
        
        for row in range(self.history_table.rowCount()):
            entry = {
                "date": self.history_table.item(row, 0).text() if self.history_table.item(row, 0) else "",
                "type": self.history_table.item(row, 1).text() if self.history_table.item(row, 1) else "",
                "description": self.history_table.item(row, 2).text() if self.history_table.item(row, 2) else "",
                "doctor": self.history_table.item(row, 3).text() if self.history_table.item(row, 3) else ""
            }
            history_data.append(entry)
        
        return history_data
    
    def _get_medical_images_data(self) -> List[Dict]:
        """Lấy dữ liệu hình ảnh y tế."""
        # TODO: Triển khai việc lấy dữ liệu từ các widget hình ảnh
        # Hiện tại sẽ trả về dữ liệu trống hoặc giữ nguyên dữ liệu cũ nếu có
        if self.current_patient and isinstance(self.current_patient.get("metadata"), dict):
            return self.current_patient.get("metadata", {}).get("medical_images", [])
        return []
    
    def _add_medical_history(self):
        """Thêm mục mới vào lịch sử y tế."""
        row_position = self.history_table.rowCount()
        self.history_table.insertRow(row_position)
        
        # Thiết lập mục mặc định cho hàng mới
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.history_table.setItem(
            row_position, 0, QTableWidgetItem(current_date))
        self.history_table.setItem(row_position, 1, QTableWidgetItem("Khám"))
        self.history_table.setItem(row_position, 2, QTableWidgetItem(""))
        self.history_table.setItem(row_position, 3, QTableWidgetItem(""))
        
        # Kích hoạt chế độ chỉnh sửa cho ô mô tả
        self.history_table.editItem(self.history_table.item(row_position, 2))
        
        # Đánh dấu là có thay đổi chưa lưu
        self._on_patient_data_changed()
    
    def _add_medical_image(self):
        """Thêm hình ảnh mới."""
        # Mở hộp thoại chọn tệp tin
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn hình ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_path:
            return
        
        # Tạo widget hiển thị hình ảnh
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        
        # Label hiển thị hình ảnh
        image_label = QLabel()
        image_label.setFixedSize(150, 150)
        image_label.setScaledContents(True)
        
        # Tải hình ảnh
        pixmap = QPixmap(file_path)
        image_label.setPixmap(pixmap)
        
        # Label hiển thị thông tin hình ảnh
        current_date = datetime.now().strftime("%Y-%m-%d")
        info_label = QLabel(f"Hình ảnh - {current_date}")
        
        image_layout.addWidget(image_label)
        image_layout.addWidget(info_label)
        
        self.images_list_layout.addWidget(image_widget)
        
        # Đánh dấu là có thay đổi chưa lưu
        self._on_patient_data_changed()
    
    def _create_new_patient(self):
        """Tạo một bệnh nhân mới."""
        try:
            # Hiển thị dialog nhập thông tin
            dialog = NewPatientDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                patient_data = dialog.get_patient_data()

                # Lưu bệnh nhân vào cơ sở dữ liệu
                patient_id = self.patient_db.create_patient(
                    name=patient_data['name'],
                    birth_date=patient_data['birth_date'],
                    gender=patient_data['gender'],
                    metadata=patient_data.get('metadata', {})
                )

                # Cập nhật tab với bệnh nhân mới
                self.set_patient(patient_id)
                
                # Phát tín hiệu cho biết bệnh nhân mới đã được tạo
                self.patient_created.emit(patient_id)
                
                QMessageBox.information(
                    self, 
                    "Thành công", 
                    f"Đã tạo bệnh nhân mới: {patient_data['name']}"
                )

        except Exception as e:
            logger.error("Lỗi khi tạo bệnh nhân mới: %s",
                         str(e), exc_info=True)
            QMessageBox.critical(
                self, 
                "Lỗi", 
                f"Không thể tạo bệnh nhân mới: {str(e)}"
            )
    
    def _delete_current_patient(self):
        """Xóa bệnh nhân hiện tại."""
        if not self.current_patient:
            return
        
        patient_id = self.current_patient.get("id")
        patient_name = self.current_patient.get("name")
        
        # Hiển thị hộp thoại xác nhận
        reply = QMessageBox.question(
            self, 
            "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa bệnh nhân '{patient_name}' (ID: {patient_id}) không?\n\nHành động này không thể hoàn tác!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # Xóa bệnh nhân khỏi cơ sở dữ liệu
            success = self.patient_db.delete_patient(patient_id)
            
            if not success:
                raise ValueError(
                    f"Không thể xóa bệnh nhân với ID: {patient_id}")
            
            # Xóa dữ liệu khỏi giao diện
            self._clear_patient_data()
            self._toggle_edit_mode(False)
            
            # Phát tín hiệu thông báo đã xóa (có thể phát với patient_id=None)
            self.patient_updated.emit(patient_id)
            
            QMessageBox.information(
                self, "Thành công", f"Đã xóa bệnh nhân '{patient_name}' thành công")
            logger.info(f"Đã xóa bệnh nhân: {patient_id}")
        except Exception as e:
            logger.exception(f"Lỗi khi xóa bệnh nhân {patient_id}: {str(e)}")
            QMessageBox.critical(
                self, "Lỗi", f"Không thể xóa bệnh nhân: {str(e)}")

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
        """
        Xuất dữ liệu DICOM cho bệnh nhân hiện tại.
        """
        if not self.current_patient:
            QMessageBox.warning(
                self, "Lỗi", "Vui lòng chọn bệnh nhân trước khi xuất dữ liệu.")
            return

        # Kiểm tra xem có dữ liệu để xuất không
        patient_id = self.current_patient['id']
        studies = self.patient_db.get_patient_studies(
            patient_id, include_series=True)

        if not studies:
            QMessageBox.warning(
                self, "Lỗi", "Bệnh nhân không có dữ liệu để xuất.")
            return

        # Mở dialog chọn thư mục đích
        directory = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục lưu dữ liệu DICOM", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not directory:
            return

        # Hiển thị dialog tiến trình
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Đang xuất dữ liệu DICOM")
        progress_dialog.setFixedSize(400, 100)

        progress_layout = QVBoxLayout(progress_dialog)
        progress_label = QLabel("Đang xuất dữ liệu DICOM, vui lòng chờ...")
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # Chế độ không xác định

        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(progress_bar)

        progress_dialog.show()

        try:
            # Tạo thư mục cho bệnh nhân
            patient_dir = os.path.join(
                directory, f"patient_{self.current_patient['id']}")
            os.makedirs(patient_dir, exist_ok=True)

            # Xuất từng study
            for study in studies:
                # Tạo thư mục cho study
                study_dir = os.path.join(patient_dir, f"study_{study['id']}")
                os.makedirs(study_dir, exist_ok=True)

                # Xuất từng series
                for series in study.get('series', []):
                    # Tạo thư mục cho series
                    series_dir = os.path.join(
                        study_dir, f"{series['modality']}_{series['id']}")
                    os.makedirs(series_dir, exist_ok=True)

                    # Sao chép các file DICOM
                    for file_path in series.get('file_paths', []):
                        if os.path.exists(file_path):
                            dest_path = os.path.join(
                                series_dir, os.path.basename(file_path))
                            shutil.copy2(file_path, dest_path)

            QMessageBox.information(
                self, "Thành công", f"Đã xuất dữ liệu DICOM thành công.\nĐường dẫn: {patient_dir}")

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Lỗi khi xuất dữ liệu DICOM: {str(e)}")

        finally:
            progress_dialog.close()

    def _import_medical_images(self):
        """
        Nhập dữ liệu DICOM cho bệnh nhân hiện tại.
        """
        self._import_patient_data()

    def _populate_medical_images(self):
        """
        Hiển thị danh sách dữ liệu y tế của bệnh nhân.
        """
        # Xóa dữ liệu hiện tại
        self.studies_table.setRowCount(0)
        self.series_table.setRowCount(0)

        if not self.current_patient:
            return

        try:
            # Lấy danh sách nghiên cứu của bệnh nhân
            studies = self.patient_db.get_patient_studies(
                self.current_patient['id'])

            # Thêm dữ liệu vào bảng
            for i, study in enumerate(studies):
                self.studies_table.insertRow(i)

                # Lấy số lượng series
                series_count = len(
                    self.patient_db.get_study_series(study['id']))

                # Thêm thông tin study
                self.studies_table.setItem(
                    i, 0, QTableWidgetItem(study.get('date', '')))
                self.studies_table.setItem(
                    i, 1, QTableWidgetItem(study.get('description', '')))

                # Xác định loại dữ liệu
                study_type = "Khác"
                if 'metadata' in study and 'type' in study['metadata']:
                    study_type = study['metadata']['type']

                self.studies_table.setItem(i, 2, QTableWidgetItem(study_type))
                self.studies_table.setItem(
                    i, 3, QTableWidgetItem(str(series_count)))

                # Lưu ID study vào item
                self.studies_table.item(i, 0).setData(
                    Qt.UserRole, study.get('id', ''))

        except Exception as e:
            logger.error(
                f"Lỗi khi hiển thị dữ liệu y tế: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Lỗi", f"Không thể hiển thị dữ liệu y tế: {str(e)}")

        # Cập nhật trạng thái nút
        self.import_images_button.setEnabled(self.current_patient is not None)

    def _on_study_selected(self):
        """
        Xử lý khi người dùng chọn một nghiên cứu.
        """
        self._update_series_table()

    def _update_series_table(self):
        """
        Cập nhật bảng series dựa trên nghiên cứu được chọn.
        """
        # Xóa dữ liệu hiện tại
        self.series_table.setRowCount(0)

        # Lấy nghiên cứu được chọn
        selected_rows = self.studies_table.selectedItems()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        study_id = self.studies_table.item(row, 0).data(Qt.UserRole)

        # Lấy danh sách series
        series_list = self.patient_db.get_study_series(study_id)

        # Thêm dữ liệu vào bảng
        for i, series in enumerate(series_list):
            self.series_table.insertRow(i)

            # Thêm thông tin series
            self.series_table.setItem(
                i, 0, QTableWidgetItem(series.get('description', '')))
            self.series_table.setItem(
                i, 1, QTableWidgetItem(series.get('modality', '')))
            self.series_table.setItem(i, 2, QTableWidgetItem(
                str(len(series.get('file_paths', [])))))

            # Lưu ID series vào item
            self.series_table.item(i, 0).setData(
                Qt.UserRole, series.get('id', ''))

        # Cập nhật trạng thái nút
        self.view_images_button.setEnabled(self.series_table.rowCount() > 0)

    def _view_medical_images(self):
        """
        Mở công cụ xem ảnh y tế.
        """
        # Kiểm tra xem có series nào được chọn không
        selected_rows = self.series_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(
                self, "Lỗi", "Vui lòng chọn một series để xem.")
            return

        row = selected_rows[0].row()
        series_id = self.series_table.item(row, 0).data(Qt.UserRole)

        try:
            # Lấy thông tin series
            series = None
            for study in self.patient_db.get_patient_studies(self.current_patient['id'], include_series=True):
                for s in study.get('series', []):
                    if s['id'] == series_id:
                        series = s
                        break
                if series:
                    break

            if not series:
                QMessageBox.warning(
                    self, "Lỗi", "Không tìm thấy thông tin series.")
                return

            # Kiểm tra loại dữ liệu
            modality = series.get('modality', '')

            # Mở công cụ xem ảnh tương ứng
            if modality in ['CT', 'MR', 'PT']:
                # Gửi sự kiện để mở tab xem ảnh với dữ liệu của series
                self.patient_updated.emit({
                    'type': 'view_images',
                    'data': {
                        'patient_id': self.current_patient['id'],
                        'series_id': series_id
                    }
                })
            elif modality in ['RTSTRUCT']:
                # Gửi sự kiện để mở tab xem cấu trúc
                self.patient_updated.emit({
                    'type': 'view_structures',
                    'data': {
                        'patient_id': self.current_patient['id'],
                        'series_id': series_id
                    }
                })
            elif modality in ['RTPLAN']:
                # Gửi sự kiện để mở tab xem kế hoạch
                self.patient_updated.emit({
                    'type': 'view_plan',
                    'data': {
                        'patient_id': self.current_patient['id'],
                        'series_id': series_id
                    }
                })
            elif modality in ['RTDOSE']:
                # Gửi sự kiện để mở tab xem liều
                self.patient_updated.emit({
                    'type': 'view_dose',
                    'data': {
                        'patient_id': self.current_patient['id'],
                        'series_id': series_id
                    }
                })
            else:
                QMessageBox.information(
                    self, "Thông báo", f"Chưa hỗ trợ xem dữ liệu loại {modality}.")

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Lỗi khi mở công cụ xem ảnh: {str(e)}")

    def _search_patients(self):
        """
        Tìm kiếm bệnh nhân dựa trên từ khóa nhập vào.
        """
        search_text = self.search_input.text().strip()
        if not search_text:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập từ khóa tìm kiếm.")
            return

        try:
            # Tìm kiếm bệnh nhân
            search_query = {
                'name': search_text
            }

            # Thử truy vấn dạng DICOM ID trong metadata
            search_query['metadata'] = {'dicom_id': search_text}

            patients = self.patient_db.search_patients(search_query)

            if not patients:
                QMessageBox.information(
                    self, "Thông báo", "Không tìm thấy bệnh nhân phù hợp.")
                return

            # Hiển thị danh sách bệnh nhân tìm thấy
            if len(patients) == 1:
                # Nếu chỉ có 1 kết quả, hiển thị ngay
                self.set_patient(patients[0]['id'])
            else:
                # Nếu có nhiều kết quả, hiển thị dialog chọn
                patient_list = []
                for patient in patients:
                    patient_info = f"{patient['name']} ({patient['birth_date'] or 'N/A'})"
                    if 'metadata' in patient and patient['metadata'] and 'dicom_id' in patient['metadata']:
                        patient_info += f" - DICOM ID: {patient['metadata']['dicom_id']}"
                    patient_list.append((patient_info, patient['id']))

                selected_patient, ok = QInputDialog.getItem(
                    self, "Chọn bệnh nhân", "Chọn bệnh nhân từ danh sách:",
                    [p[0] for p in patient_list], 0, False
                )

                if ok and selected_patient:
                    # Lấy ID của bệnh nhân được chọn
                    selected_index = [p[0]
                                      for p in patient_list].index(selected_patient)
                    selected_id = patient_list[selected_index][1]
                    self.set_patient(selected_id)

        except Exception as e:
            logger.error(
                f"Lỗi khi tìm kiếm bệnh nhân: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Lỗi", f"Không thể tìm kiếm bệnh nhân: {str(e)}")

