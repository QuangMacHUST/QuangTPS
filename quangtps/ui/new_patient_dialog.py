#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog để tạo bệnh nhân mới
"""

import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QComboBox,
    QDateEdit, QTextEdit, QDialogButtonBox, QLabel, QGroupBox
)
from PyQt5.QtCore import Qt, QDate

logger = logging.getLogger(__name__)

class NewPatientDialog(QDialog):
    """
    Dialog tạo bệnh nhân mới.
    """
    
    def __init__(self, parent=None):
        """
        Khởi tạo dialog tạo bệnh nhân mới.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        self.setWindowTitle("Tạo bệnh nhân mới")
        self.setMinimumWidth(400)
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện dialog"""
        layout = QVBoxLayout(self)
        
        # Thông tin cơ bản
        basic_group = QGroupBox("Thông tin cơ bản")
        form_layout = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        form_layout.addRow("Họ tên:", self.name_edit)
        
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Nam", "Nữ", "Khác"])
        form_layout.addRow("Giới tính:", self.gender_combo)
        
        self.birth_date_edit = QDateEdit()
        self.birth_date_edit.setDisplayFormat("dd/MM/yyyy")
        self.birth_date_edit.setCalendarPopup(True)
        self.birth_date_edit.setDate(QDate.currentDate())
        form_layout.addRow("Ngày sinh:", self.birth_date_edit)
        
        self.mrn_edit = QLineEdit()
        form_layout.addRow("Mã bệnh án:", self.mrn_edit)
        
        self.phone_edit = QLineEdit()
        form_layout.addRow("Điện thoại:", self.phone_edit)
        
        self.email_edit = QLineEdit()
        form_layout.addRow("Email:", self.email_edit)
        
        self.address_edit = QTextEdit()
        self.address_edit.setMaximumHeight(60)
        form_layout.addRow("Địa chỉ:", self.address_edit)
        
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(60)
        form_layout.addRow("Ghi chú:", self.notes_edit)
        
        layout.addWidget(basic_group)
        
        # Nút OK/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)

    def get_patient_data(self):
        """
        Lấy dữ liệu bệnh nhân đã nhập.
        
        Returns
        -------
        dict
            Dữ liệu bệnh nhân
        """
        return {
            'name': self.name_edit.text(),
            'gender': self.gender_combo.currentText(),
            'birth_date': self.birth_date_edit.date().toString("yyyy-MM-dd"),
            'medical_record_num': self.mrn_edit.text(),
            'phone': self.phone_edit.text(),
            'email': self.email_edit.text(),
            'address': self.address_edit.toPlainText(),
            'notes': self.notes_edit.toPlainText(),
            'metadata': {}
        } 