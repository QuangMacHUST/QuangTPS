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

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QComboBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QScrollArea, QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QDate, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

from quangtps.database.patient_db import PatientDatabase

logger = logging.getLogger(__name__)


class PatientTab(QWidget):
    """
    Tab hiển thị và chỉnh sửa thông tin bệnh nhân.
    
    Tab này bao gồm các phần thông tin cá nhân, lịch sử bệnh,
    kết quả khám lâm sàng, hình ảnh y tế, và các dữ liệu khác
    liên quan đến bệnh nhân.
    """
    
    # Tín hiệu để thông báo khi cập nhật dữ liệu bệnh nhân
    patient_updated = pyqtSignal(object)
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
        
        # Thiết lập giao diện
        self._init_ui()
        
        logger.info("Khởi tạo tab thông tin bệnh nhân hoàn tất")
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Thanh công cụ
        toolbar_layout = QHBoxLayout()
        
        # Nút tạo bệnh nhân mới
        self.new_patient_btn = QPushButton("Bệnh nhân mới")
        self.new_patient_btn.clicked.connect(self._create_new_patient)
        toolbar_layout.addWidget(self.new_patient_btn)
        
        # Nút xóa bệnh nhân hiện tại
        self.delete_patient_btn = QPushButton("Xóa bệnh nhân")
        self.delete_patient_btn.clicked.connect(self._delete_current_patient)
        self.delete_patient_btn.setEnabled(False)
        toolbar_layout.addWidget(self.delete_patient_btn)
        
        toolbar_layout.addStretch()
        
        self.main_layout.addLayout(toolbar_layout)
        
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
        
        # Tab lịch sử y tế
        self.medical_history_widget = QWidget()
        self.medical_history_layout = QVBoxLayout(self.medical_history_widget)
        
        # Bảng lịch sử y tế
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Ngày", "Loại", "Mô tả", "Bác sĩ"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        
        # Nút thêm mục lịch sử mới
        self.add_history_btn = QPushButton("Thêm mục")
        self.add_history_btn.clicked.connect(self._add_medical_history)
        
        self.medical_history_layout.addWidget(self.history_table)
        self.medical_history_layout.addWidget(self.add_history_btn, alignment=Qt.AlignRight)
        
        # Thêm tab lịch sử y tế
        self.info_tabs.addTab(self.medical_history_widget, "Lịch sử y tế")
        
        # Tab hình ảnh y tế
        self.medical_images_widget = QWidget()
        self.medical_images_layout = QVBoxLayout(self.medical_images_widget)
        
        # Layout cho danh sách hình ảnh
        self.images_list_layout = QHBoxLayout()
        
        # Nút thêm hình ảnh mới
        self.add_image_btn = QPushButton("Thêm hình ảnh")
        self.add_image_btn.clicked.connect(self._add_medical_image)
        
        self.medical_images_layout.addLayout(self.images_list_layout)
        self.medical_images_layout.addWidget(self.add_image_btn, alignment=Qt.AlignRight)
        
        # Thêm tab hình ảnh y tế
        self.info_tabs.addTab(self.medical_images_widget, "Hình ảnh y tế")
        
        # Vô hiệu hóa các widget khi không có bệnh nhân được chọn
        self._toggle_edit_mode(False)
        
        # Kết nối tín hiệu
        self.full_name_field.textChanged.connect(self._on_patient_data_changed)
        
        logger.info("Khởi tạo giao diện tab bệnh nhân hoàn tất")
    
    def _toggle_edit_mode(self, enabled: bool):
        """Bật/tắt chế độ chỉnh sửa cho các widget."""
        # Các widget trong tab thông tin cơ bản
        self.full_name_field.setEnabled(enabled)
        self.dob_field.setEnabled(enabled)
        self.gender_field.setEnabled(enabled)
        self.id_number_field.setEnabled(enabled)
        self.phone_field.setEnabled(enabled)
        self.email_field.setEnabled(enabled)
        self.address_field.setEnabled(enabled)
        self.blood_type_field.setEnabled(enabled)
        self.allergies_field.setEnabled(enabled)
        self.height_field.setEnabled(enabled)
        self.weight_field.setEnabled(enabled)
        
        # Các nút trong tab lịch sử y tế và hình ảnh
        self.add_history_btn.setEnabled(enabled)
        self.add_image_btn.setEnabled(enabled)
        
        # Nút lưu và xóa
        self.save_button.setEnabled(enabled)
        self.delete_patient_btn.setEnabled(enabled)
    
    def _on_patient_data_changed(self):
        """Xử lý khi dữ liệu bệnh nhân thay đổi."""
        # Đổi tiêu đề tab để hiển thị trạng thái chưa lưu
        if self.current_patient and not self.windowTitle().endswith("*"):
            self.setWindowTitle(f"{self.windowTitle()} *")
    
    def set_patient(self, patient_id: str):
        """
        Thiết lập bệnh nhân hiện tại và cập nhật giao diện.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân cần hiển thị
        """
        if not patient_id:
            self._clear_patient_data()
            self._toggle_edit_mode(False)
            return
        
        try:
            patient = self.patient_db.get_patient(patient_id)
            if not patient:
                logger.error(f"Không tìm thấy bệnh nhân với ID: {patient_id}")
                QMessageBox.warning(self, "Lỗi", f"Không tìm thấy bệnh nhân với ID: {patient_id}")
                return
            
            self.current_patient = patient
            self._populate_patient_data()
            self._toggle_edit_mode(True)
            
            logger.info(f"Đã tải bệnh nhân: {patient_id}")
        except Exception as e:
            logger.exception(f"Lỗi khi tải bệnh nhân {patient_id}: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Không thể tải thông tin bệnh nhân: {str(e)}")
    
    def _clear_patient_data(self):
        """Xóa tất cả dữ liệu bệnh nhân khỏi giao diện."""
        self.current_patient = None
        
        # Xóa các trường thông tin cơ bản
        self.patient_id_field.clear()
        self.full_name_field.clear()
        self.dob_field.setDate(QDate.currentDate())
        self.gender_field.setCurrentIndex(0)
        self.id_number_field.clear()
        self.phone_field.clear()
        self.email_field.clear()
        self.address_field.clear()
        self.blood_type_field.setCurrentIndex(0)
        self.allergies_field.clear()
        self.height_field.setValue(0)
        self.weight_field.setValue(0)
        
        # Xóa bảng lịch sử y tế
        self.history_table.setRowCount(0)
        
        # Xóa hình ảnh y tế
        # Xóa tất cả widget con trong images_list_layout
        while self.images_list_layout.count():
            item = self.images_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        logger.info("Đã xóa dữ liệu bệnh nhân khỏi giao diện")
    
    def _populate_patient_data(self):
        """Điền dữ liệu bệnh nhân vào giao diện."""
        if not self.current_patient:
            return
        
        # Điền các trường thông tin cơ bản
        self.patient_id_field.setText(self.current_patient.get("id", ""))
        self.full_name_field.setText(self.current_patient.get("name", ""))
        
        # Xử lý ngày sinh
        birth_date = self.current_patient.get("birth_date")
        if birth_date:
            if isinstance(birth_date, str):
                try:
                    date_obj = datetime.strptime(birth_date, "%Y-%m-%d")
                    self.dob_field.setDate(QDate(date_obj.year, date_obj.month, date_obj.day))
                except ValueError:
                    logger.warning(f"Không thể chuyển đổi ngày sinh: {birth_date}")
                    self.dob_field.setDate(QDate.currentDate())
            elif isinstance(birth_date, datetime):
                self.dob_field.setDate(QDate(birth_date.year, birth_date.month, birth_date.day))
        else:
            self.dob_field.setDate(QDate.currentDate())
        
        # Thiết lập giới tính
        gender = self.current_patient.get("gender", "")
        if gender == "male":
            self.gender_field.setCurrentIndex(0)  # Nam
        elif gender == "female":
            self.gender_field.setCurrentIndex(1)  # Nữ
        else:
            self.gender_field.setCurrentIndex(2)  # Khác
        
        # Lấy thông tin bổ sung từ metadata
        metadata = self.current_patient.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Không thể parse metadata: {metadata}")
                metadata = {}
        
        self.id_number_field.setText(metadata.get("id_number", ""))
        self.phone_field.setText(metadata.get("phone", ""))
        self.email_field.setText(metadata.get("email", ""))
        self.address_field.setText(metadata.get("address", ""))
        
        # Thiết lập thông tin y tế
        blood_type = metadata.get("blood_type", "Không biết")
        blood_type_index = self.blood_type_field.findText(blood_type)
        if blood_type_index >= 0:
            self.blood_type_field.setCurrentIndex(blood_type_index)
        else:
            self.blood_type_field.setCurrentIndex(8)  # "Không biết"
        
        self.allergies_field.setText(metadata.get("allergies", ""))
        self.height_field.setValue(metadata.get("height", 0))
        self.weight_field.setValue(metadata.get("weight", 0))
        
        # Điền dữ liệu lịch sử y tế
        self._populate_medical_history(metadata.get("medical_history", []))
        
        # Điền dữ liệu hình ảnh y tế
        self._populate_medical_images(metadata.get("medical_images", []))
        
        logger.info(f"Đã điền dữ liệu cho bệnh nhân: {self.current_patient.get('id')}")
    
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
            info_label = QLabel(f"{image_data.get('type', 'Không rõ')} - {image_data.get('date', '')}")
            
            image_layout.addWidget(image_label)
            image_layout.addWidget(info_label)
            
            self.images_list_layout.addWidget(image_widget)
    
    def _save_patient_info(self):
        """Lưu thông tin bệnh nhân hiện tại."""
        if not self.current_patient:
            logger.warning("Không có bệnh nhân nào được chọn để lưu")
            return
        
        try:
            # Thu thập dữ liệu từ giao diện
            name = self.full_name_field.text().strip()
            if not name:
                QMessageBox.warning(self, "Lỗi", "Họ tên bệnh nhân không được để trống")
                return
            
            # Lấy ngày sinh
            dob = self.dob_field.date()
            birth_date = datetime(dob.year(), dob.month(), dob.day())
            
            # Lấy giới tính
            gender_map = {0: "male", 1: "female", 2: "other"}
            gender = gender_map.get(self.gender_field.currentIndex(), "other")
            
            # Thu thập metadata
            metadata = {
                "id_number": self.id_number_field.text().strip(),
                "phone": self.phone_field.text().strip(),
                "email": self.email_field.text().strip(),
                "address": self.address_field.text().strip(),
                "blood_type": self.blood_type_field.currentText(),
                "allergies": self.allergies_field.text().strip(),
                "height": self.height_field.value(),
                "weight": self.weight_field.value(),
                "medical_history": self._get_medical_history_data(),
                "medical_images": self._get_medical_images_data()
            }
            
            # Cập nhật bệnh nhân trong cơ sở dữ liệu
            patient_id = self.current_patient.get("id")
            self.patient_db.update_patient(
                patient_id,
                name=name,
                birth_date=birth_date,
                gender=gender,
                metadata=metadata
            )
            
            # Cập nhật current_patient với dữ liệu mới
            self.current_patient = self.patient_db.get_patient(patient_id)
            
            # Phát tín hiệu thông báo cập nhật
            self.patient_updated.emit(self.current_patient)
            
            # Cập nhật tiêu đề tab
            if self.windowTitle().endswith("*"):
                self.setWindowTitle(self.windowTitle()[:-2])
            
            QMessageBox.information(self, "Thành công", "Đã lưu thông tin bệnh nhân thành công")
            logger.info(f"Đã lưu thông tin bệnh nhân: {patient_id}")
            
        except Exception as e:
            logger.exception(f"Lỗi khi lưu thông tin bệnh nhân: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu thông tin bệnh nhân: {str(e)}")
    
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
        self.history_table.setItem(row_position, 0, QTableWidgetItem(current_date))
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
        """Tạo bệnh nhân mới."""
        try:
            # Tạo bệnh nhân mới với thông tin mặc định
            patient_name = "Bệnh nhân mới"
            patient_id = self.patient_db.create_patient(
                name=patient_name,
                birth_date=datetime.now(),
                gender="other"
            )
            
            if not patient_id:
                raise ValueError("Không thể tạo bệnh nhân mới")
            
            # Tải bệnh nhân mới vào giao diện
            self.set_patient(patient_id)
            
            # Phát tín hiệu thông báo tạo mới
            self.patient_created.emit(patient_id)
            
            # Focus vào trường họ tên để người dùng nhập
            self.full_name_field.setFocus()
            self.full_name_field.selectAll()
            
            logger.info(f"Đã tạo bệnh nhân mới với ID: {patient_id}")
        except Exception as e:
            logger.exception(f"Lỗi khi tạo bệnh nhân mới: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Không thể tạo bệnh nhân mới: {str(e)}")
    
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
                raise ValueError(f"Không thể xóa bệnh nhân với ID: {patient_id}")
            
            # Xóa dữ liệu khỏi giao diện
            self._clear_patient_data()
            self._toggle_edit_mode(False)
            
            # Phát tín hiệu thông báo đã xóa (có thể phát với patient_id=None)
            self.patient_updated.emit({"id": None})
            
            QMessageBox.information(self, "Thành công", f"Đã xóa bệnh nhân '{patient_name}' thành công")
            logger.info(f"Đã xóa bệnh nhân: {patient_id}")
        except Exception as e:
            logger.exception(f"Lỗi khi xóa bệnh nhân {patient_id}: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Không thể xóa bệnh nhân: {str(e)}")
