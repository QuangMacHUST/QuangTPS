#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog tạo bệnh nhân mới cho QuangTPS.
"""

import logging
import uuid
from datetime import datetime

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QComboBox, QGroupBox, QFormLayout,
    QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt, QDate

from quangtps.database.patient_db import PatientDatabase

logger = logging.getLogger(__name__)


class PatientCreationDialog(QDialog):
    """
    Dialog tạo mới bệnh nhân.
    """
    
    def __init__(self, parent=None):
        """Khởi tạo dialog."""
        super().__init__(parent)
        self.setWindowTitle("Tạo bệnh nhân mới")
        self.setMinimumWidth(500)
        
        self.patient_db = PatientDatabase()
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện."""
        layout = QVBoxLayout(self)
        
        # Form nhập thông tin
        form_group = QGroupBox("Thông tin bệnh nhân")
        form_layout = QFormLayout(form_group)
        
        # ID bệnh nhân
        self.patient_id_edit = QLineEdit()
        self.patient_id_edit.setPlaceholderText("Nhập ID bệnh nhân (hoặc để trống để tạo tự động)")
        form_layout.addRow("ID bệnh nhân:", self.patient_id_edit)
        
        # Tên bệnh nhân
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Nhập họ tên bệnh nhân")
        form_layout.addRow("Họ tên (*):", self.name_edit)
        
        # Ngày sinh
        self.birth_date_edit = QDateEdit()
        self.birth_date_edit.setDate(QDate.currentDate())
        self.birth_date_edit.setCalendarPopup(True)
        form_layout.addRow("Ngày sinh:", self.birth_date_edit)
        
        # Giới tính
        self.gender_combobox = QComboBox()
        self.gender_combobox.addItems(["Nam", "Nữ", "Khác"])
        form_layout.addRow("Giới tính:", self.gender_combobox)
        
        # Ghi chú
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Ghi chú thêm về bệnh nhân")
        self.notes_edit.setMaximumHeight(100)
        form_layout.addRow("Ghi chú:", self.notes_edit)
        
        layout.addWidget(form_group)
        
        # Nút bấm
        button_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Hủy")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.create_btn = QPushButton("Tạo")
        self.create_btn.setDefault(True)
        self.create_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.create_btn)
        
        layout.addLayout(button_layout)
    
    def accept(self):
        """Lưu bệnh nhân mới và đóng dialog."""
        try:
            # Kiểm tra và lấy các giá trị nhập
            patient_id = self.patient_id_edit.text().strip()
            name = self.name_edit.text().strip()
            birth_date = self.birth_date_edit.date().toString("yyyy-MM-dd")
            gender = self.gender_combobox.currentText()
            
            # Nếu không nhập ID, tạo ID mới
            if not patient_id:
                patient_id = str(uuid.uuid4())
            
            # Kiểm tra tên bệnh nhân
            if not name:
                QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập tên bệnh nhân.")
                return
            
            # Kiểm tra ID đã tồn tại chưa
            try:
                if self.patient_db.patient_exists(patient_id):
                    QMessageBox.warning(self, "Cảnh báo", f"ID bệnh nhân '{patient_id}' đã tồn tại.")
                    return
            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra ID bệnh nhân: {str(e)}", exc_info=True)
            
            # Chuẩn bị metadata
            metadata = {
                "notes": self.notes_edit.toPlainText().strip(),
                "created_at": datetime.now().isoformat()
            }
            
            # Tạo đối tượng bệnh nhân
            patient = {
                'id': patient_id,
                'name': name,
                'dob': birth_date,
                'gender': gender.lower(),
                'metadata': metadata
            }
            
            # Lưu bệnh nhân
            try:
                if self.patient_db.add_patient(patient):
                    logger.info(f"Đã tạo bệnh nhân mới: {patient_id}")
                    super().accept()
                else:
                    QMessageBox.critical(self, "Lỗi", "Không thể tạo bệnh nhân. Vui lòng kiểm tra lại.")
            except Exception as save_error:
                logger.error(f"Lỗi khi lưu bệnh nhân: {str(save_error)}", exc_info=True)
                QMessageBox.critical(self, "Lỗi khi lưu", f"Không thể lưu bệnh nhân: {str(save_error)}")
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo bệnh nhân mới: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Lỗi", f"Đã xảy ra lỗi: {str(e)}")
    
    def reject(self):
        """Đóng dialog mà không lưu."""
        super().reject()
    
    def get_patient_data(self):
        """
        Lấy dữ liệu bệnh nhân từ dialog.
        
        Returns:
            dict: Dữ liệu bệnh nhân
        """
        gender_map = {"Nam": "male", "Nữ": "female", "Khác": "other"}
        
        metadata = {
            "notes": self.notes_edit.toPlainText(),
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "id": self.patient_id_edit.text().strip() or str(uuid.uuid4()),
            "name": self.name_edit.text().strip(),
            "dob": self.birth_date_edit.date().toString("yyyy-MM-dd"),
            "birth_date": self.birth_date_edit.date().toString("yyyy-MM-dd"),
            "gender": gender_map[self.gender_combobox.currentText()],
            "metadata": metadata
        } 