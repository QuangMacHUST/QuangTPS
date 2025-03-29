#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog for displaying patient search results and selecting a patient.
"""

import logging
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QDialogButtonBox, QLabel, QHeaderView
)
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)

class PatientSearchResultDialog(QDialog):
    """
    Dialog to display patient search results and allow the user to select one.
    """
    
    def __init__(self, patients, parent=None):
        """
        Initialize the patient search result dialog.
        
        Parameters
        ----------
        patients : list
            List of patient dictionaries
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Kết quả tìm kiếm bệnh nhân")
        self.setMinimumSize(600, 400)
        self.patients = patients
        self.selected_patient_id = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Header
        layout.addWidget(QLabel(f"Tìm thấy {len(self.patients)} bệnh nhân:"))
        
        # Patient table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Họ tên", "Mã bệnh án", "Giới tính", "Ngày sinh", "Điện thoại"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.doubleClicked.connect(self._on_table_double_clicked)
        
        self._populate_table()
        
        layout.addWidget(self.table)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accepted)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
    
    def _populate_table(self):
        """Populate the table with patient data"""
        self.table.setRowCount(len(self.patients))
        
        for row, patient in enumerate(self.patients):
            self.table.setItem(row, 0, QTableWidgetItem(patient.get('name', '')))
            self.table.setItem(row, 1, QTableWidgetItem(patient.get('medical_record_num', '')))
            self.table.setItem(row, 2, QTableWidgetItem(patient.get('gender', '')))
            self.table.setItem(row, 3, QTableWidgetItem(patient.get('birth_date', '')))
            self.table.setItem(row, 4, QTableWidgetItem(patient.get('phone', '')))
            
            # Store the patient ID in the first column as user data
            self.table.item(row, 0).setData(Qt.UserRole, patient.get('id', ''))
    
    def _on_table_double_clicked(self, index):
        """Handle double click on table row"""
        row = index.row()
        self._select_row(row)
        self.accept()
    
    def _on_accepted(self):
        """Handle dialog acceptance"""
        selected_rows = self.table.selectionModel().selectedRows()
        
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        self._select_row(row)
        self.accept()
    
    def _select_row(self, row):
        """Select a patient from the specified row"""
        self.selected_patient_id = self.table.item(row, 0).data(Qt.UserRole) 