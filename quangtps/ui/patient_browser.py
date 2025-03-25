#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý trình duyệt bệnh nhân.

Module này cung cấp giao diện để xem và quản lý danh sách bệnh nhân
cùng với các kế hoạch điều trị liên quan.
"""

import logging
import os
import json
from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal, QDate
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTreeWidget,
    QTreeWidgetItem, QMenu, QAction, QMessageBox, QInputDialog, QLineEdit,
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QDateEdit, QTextEdit,
    QHeaderView, QSplitter, QGroupBox
)
from PyQt5.QtGui import QIcon, QColor

from quangtps.database.patient_db import PatientDatabase
from quangtps.database.plan_db import PlanDB
from quangtps.core.patient import Patient
from quangtps.ui.patient_creation_dialog import PatientCreationDialog

logger = logging.getLogger(__name__)

class PatientBrowser(QWidget):
    """Widget để duyệt và quản lý bệnh nhân."""
    
    # Tín hiệu khi chọn bệnh nhân hoặc kế hoạch
    patient_selected = pyqtSignal(str)  # patient_id
    plan_selected = pyqtSignal(str, str)  # patient_id, plan_id
    
    def __init__(self, parent=None):
        """Khởi tạo PatientBrowser."""
        super().__init__(parent)
        self.patient_db = PatientDatabase()
        self._init_ui()
        self._load_patients()
    
    def _init_ui(self):
        """Khởi tạo giao diện."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Thanh công cụ
        toolbar_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Làm mới")
        refresh_btn.clicked.connect(self._load_patients)
        refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        
        new_patient_btn = QPushButton("Bệnh nhân mới")
        new_patient_btn.clicked.connect(self._create_new_patient)
        new_patient_btn.setIcon(QIcon.fromTheme("user-new"))
        
        # Thêm các nút vào thanh công cụ
        toolbar_layout.addWidget(refresh_btn)
        toolbar_layout.addWidget(new_patient_btn)
        toolbar_layout.addStretch()
        
        main_layout.addLayout(toolbar_layout)
        
        # Trình duyệt bệnh nhân (dạng cây)
        self.patient_tree = QTreeWidget()
        self.patient_tree.setHeaderLabels(["Tên", "ID", "Ngày sinh", "Giới tính"])
        self.patient_tree.setColumnWidth(0, 200)
        self.patient_tree.setColumnWidth(1, 80)
        self.patient_tree.setColumnWidth(2, 100)
        self.patient_tree.setColumnWidth(3, 80)
        self.patient_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.patient_tree.customContextMenuRequested.connect(self._show_context_menu)
        self.patient_tree.itemSelectionChanged.connect(self._on_selection_changed)
        
        main_layout.addWidget(self.patient_tree)
        
        # Thanh tìm kiếm
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Tìm kiếm:"))
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Nhập tên hoặc ID bệnh nhân...")
        self.search_input.textChanged.connect(self._apply_search)
        
        search_layout.addWidget(self.search_input)
        
        # Combobox lọc theo giới tính
        search_layout.addWidget(QLabel("Giới tính:"))
        self.gender_filter = QComboBox()
        self.gender_filter.addItems(["Tất cả", "Nam", "Nữ", "Khác"])
        self.gender_filter.currentTextChanged.connect(self._apply_search)
        search_layout.addWidget(self.gender_filter)
        
        main_layout.addLayout(search_layout)
    
    def _load_patients(self):
        """Tải danh sách bệnh nhân và các kế hoạch liên quan."""
        self.patient_tree.clear()
        
        try:
            patients = self.patient_db.get_all_patients()
            
            for patient in patients:
                patient_item = QTreeWidgetItem(self.patient_tree)
                patient_item.setText(0, patient.get('name', 'Không tên'))
                patient_item.setText(1, patient.get('id', ''))
                
                # Hiển thị ngày sinh nếu có
                birth_date = patient.get('birth_date', '')
                patient_item.setText(2, birth_date if birth_date else "")
                
                # Hiển thị giới tính
                gender = patient.get('gender', '')
                patient_item.setText(3, gender if gender else "")
                
                patient_item.setData(0, Qt.UserRole, {"type": "patient", "data": patient})
                
                # Thêm các kế hoạch của bệnh nhân nếu có
                try:
                    plans = self.patient_db.get_patient_plans(patient.get('id', ''))
                    if plans:
                        for plan in plans:
                            plan_item = QTreeWidgetItem(patient_item)
                            plan_item.setText(0, plan.get('name', 'Không tên'))
                            plan_item.setText(1, plan.get('id', ''))
                            
                            # Hiển thị ngày tạo
                            created_date = plan.get('created_at', '')
                            if created_date:
                                # Hiển thị dạng ngày
                                plan_item.setText(2, created_date.split('T')[0] if 'T' in created_date else created_date)
                            
                            plan_item.setData(0, Qt.UserRole, {"type": "plan", "data": plan, "patient_id": patient.get('id', '')})
                except Exception as e:
                    logger.warning("Không thể tải kế hoạch cho bệnh nhân %s: %s", patient.get('id', ''), str(e))
                    
        except Exception as e:
            logger.error("Lỗi khi tải danh sách bệnh nhân: %s", str(e))
            QMessageBox.critical(self, "Lỗi", f"Không thể tải danh sách bệnh nhân: {str(e)}")
        
        # Tự động mở rộng cây
        self.patient_tree.expandAll()
    
    def refresh_patients(self):
        """
        Làm mới danh sách bệnh nhân.
        Phương thức này là bí danh công khai cho _load_patients.
        """
        self._load_patients()
    
    def select_patient(self, patient_id):
        """
        Chọn một bệnh nhân trong danh sách theo ID.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân cần chọn
        """
        if not patient_id:
            return
            
        # Tìm bệnh nhân trong danh sách
        for i in range(self.patient_tree.topLevelItemCount()):
            item = self.patient_tree.topLevelItem(i)
            item_data = item.data(0, Qt.UserRole)
            
            if item_data and item_data.get("type") == "patient":
                patient_data = item_data.get("data", {})
                if patient_data and patient_data.get("id") == patient_id:
                    # Chọn item
                    self.patient_tree.setCurrentItem(item)
                    # Gọi phương thức xử lý sự kiện chọn
                    self._on_selection_changed()
                    return
                
        logger.warning("Không tìm thấy bệnh nhân có ID: %s", patient_id)
    
    def _apply_search(self):
        """Áp dụng bộ lọc tìm kiếm cho cây bệnh nhân."""
        search_text = self.search_input.text().lower()
        gender_filter = self.gender_filter.currentText()
        
        # Ẩn hiện từng hàng dựa trên các điều kiện lọc
        for i in range(self.patient_tree.topLevelItemCount()):
            patient_item = self.patient_tree.topLevelItem(i)
            
            patient_name = patient_item.text(0).lower()
            patient_id = patient_item.text(1).lower()
            patient_gender = patient_item.text(3)
            
            # Kiểm tra điều kiện tìm kiếm văn bản
            text_match = not search_text or search_text in patient_name or search_text in patient_id
            
            # Kiểm tra điều kiện lọc giới tính
            gender_match = gender_filter == "Tất cả" or patient_gender == gender_filter
            
            # Ẩn/hiện item dựa trên kết quả lọc
            patient_item.setHidden(not (text_match and gender_match))
    
    def _on_selection_changed(self):
        """Xử lý sự kiện khi lựa chọn thay đổi."""
        selected_items = self.patient_tree.selectedItems()
        if not selected_items:
            return
            
        item = selected_items[0]
        item_data = item.data(0, Qt.UserRole)
        
        if not item_data:
            return
            
        item_type = item_data.get("type")
        if item_type == "patient":
            patient_data = item_data.get("data")
            if patient_data:
                self.patient_selected.emit(patient_data.get("id", ""))
        elif item_type == "plan":
            patient_id = item_data.get("patient_id", "")
            plan_id = item_data.get("data", {}).get("id", "")
            if patient_id and plan_id:
                self.plan_selected.emit(patient_id, plan_id)
    
    def _show_context_menu(self, position):
        """Hiển thị menu ngữ cảnh cho item được chọn."""
        item = self.patient_tree.itemAt(position)
        if not item:
            return
            
        item_data = item.data(0, Qt.UserRole)
        if not item_data:
            return
            
        context_menu = QMenu(self)
        
        if item_data["type"] == "patient":
            # Menu ngữ cảnh cho bệnh nhân
            edit_patient_action = QAction("Chỉnh sửa bệnh nhân", self)
            edit_patient_action.triggered.connect(
                lambda: self._edit_patient(item_data["data"])
            )
            
            delete_patient_action = QAction("Xóa bệnh nhân", self)
            delete_patient_action.triggered.connect(
                lambda: self._delete_patient(item_data["data"].get('id'))
            )
            
            new_plan_action = QAction("Tạo kế hoạch mới", self)
            new_plan_action.triggered.connect(
                lambda: self._create_new_plan(item_data["data"].get('id'))
            )
            
            context_menu.addAction(edit_patient_action)
            context_menu.addAction(delete_patient_action)
            context_menu.addSeparator()
            context_menu.addAction(new_plan_action)
            
        elif item_data["type"] == "plan":
            # Menu ngữ cảnh cho kế hoạch
            view_plan_action = QAction("Xem kế hoạch", self)
            view_plan_action.triggered.connect(
                lambda: self._view_plan(item_data["patient_id"], item_data["id"])
            )
            
            edit_plan_action = QAction("Chỉnh sửa kế hoạch", self)
            edit_plan_action.triggered.connect(
                lambda: self._edit_plan(item_data["patient_id"], item_data["id"])
            )
            
            delete_plan_action = QAction("Xóa kế hoạch", self)
            delete_plan_action.triggered.connect(
                lambda: self._delete_plan(item_data["patient_id"], item_data["id"])
            )
            
            context_menu.addAction(view_plan_action)
            context_menu.addAction(edit_plan_action)
            context_menu.addAction(delete_plan_action)
        
        context_menu.exec_(self.patient_tree.mapToGlobal(position))
    
    def _create_new_patient(self):
        """Tạo bệnh nhân mới."""
        try:
            # Sử dụng PatientCreationDialog thay vì PatientDialog để đồng nhất với PatientTab
            dialog = PatientCreationDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                # Lấy thông tin bệnh nhân từ dialog
                patient_data = dialog.get_patient_data()
                
                # Làm mới danh sách
                self._load_patients()
                
                # Chọn bệnh nhân mới tạo
                self.select_patient(patient_data["id"])
                
                QMessageBox.information(
                    self,
                    "Thành công",
                    f"Đã tạo bệnh nhân mới: {patient_data['name']}"
                )
                
                logger.info(f"Đã tạo bệnh nhân mới từ PatientBrowser: {patient_data['id']}")
                
        except Exception as e:
            logger.error("Lỗi khi tạo bệnh nhân mới: %s", str(e))
            
            # Kiểm tra lỗi cụ thể 
            error_msg = str(e)
            if "table patients has no column named dob" in error_msg:
                QMessageBox.critical(
                    self,
                    "Lỗi cấu trúc cơ sở dữ liệu",
                    "Cơ sở dữ liệu thiếu cột 'dob' cho bảng patients.\n\n"
                    "Để khắc phục, vui lòng chạy lệnh sau:\n"
                    "python scripts/update_database.py\n\n"
                    "Lệnh này sẽ cập nhật cấu trúc cơ sở dữ liệu để phù hợp với phiên bản hiện tại."
                )
            else:
                QMessageBox.critical(
                    self,
                    "Lỗi",
                    f"Không thể tạo bệnh nhân mới: {str(e)}"
                )
    
    def _edit_patient(self, patient_data):
        """Chỉnh sửa thông tin bệnh nhân."""
        dialog = PatientCreationDialog(self, patient_data)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            updated_data = dialog.get_patient_data()
            try:
                # Cập nhật bệnh nhân
                self.patient_db.update_patient(
                    patient_id=patient_data.get('id'),
                    name=updated_data["name"],
                    birth_date=updated_data["dob"],  # Sử dụng dob thay vì birth_date
                    gender=updated_data["gender"],
                    metadata=updated_data["metadata"]
                )
                
                QMessageBox.information(
                    self,
                    "Thành công",
                    f"Đã cập nhật thông tin bệnh nhân: {updated_data['name']}"
                )
                
                # Làm mới danh sách
                self._load_patients()
                
            except Exception as e:
                logger.error("Lỗi khi cập nhật bệnh nhân: %s", str(e))
                QMessageBox.critical(
                    self,
                    "Lỗi",
                    f"Không thể cập nhật bệnh nhân: {str(e)}"
                )
    
    def _delete_patient(self, patient_id):
        """Xóa bệnh nhân."""
        confirm = QMessageBox.question(
            self,
            "Xác nhận xóa",
            "Bạn có chắc muốn xóa bệnh nhân này và tất cả dữ liệu liên quan không?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Xóa bệnh nhân
                self.patient_db.delete_patient(patient_id)
                
                QMessageBox.information(
                    self,
                    "Thành công",
                    "Đã xóa bệnh nhân thành công"
                )
                
                # Làm mới danh sách
                self._load_patients()
                
            except Exception as e:
                logger.error("Lỗi khi xóa bệnh nhân: %s", str(e))
                QMessageBox.critical(
                    self,
                    "Lỗi",
                    f"Không thể xóa bệnh nhân: {str(e)}"
                )
    
    def _create_new_plan(self, patient_id):
        """Tạo kế hoạch điều trị mới cho bệnh nhân."""
        # Lấy thông tin bệnh nhân
        patient = self.patient_db.get_patient(patient_id)
        if not patient:
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy bệnh nhân")
            return
        
        plan_name, ok = QInputDialog.getText(
            self,
            "Tạo kế hoạch mới",
            f"Nhập tên kế hoạch điều trị cho bệnh nhân {patient.get('name')}:"
        )
        
        if ok and plan_name:
            # Triển khai tạo kế hoạch mới
            # Đây là phần sẽ kết nối với module kế hoạch
            QMessageBox.information(
                self,
                "Thành công",
                f"Đã tạo kế hoạch mới: {plan_name}"
            )
            
            # Làm mới danh sách
            self._load_patients()
    
    def _view_plan(self, patient_id, plan_id):
        """Xem chi tiết kế hoạch điều trị."""
        self.plan_selected.emit(patient_id, plan_id)
    
    def _edit_plan(self, patient_id, plan_id):
        """Chỉnh sửa kế hoạch điều trị."""
        # Lấy thông tin kế hoạch
        # Mở giao diện chỉnh sửa kế hoạch
        self.plan_selected.emit(patient_id, plan_id)
    
    def _delete_plan(self, patient_id, plan_id):
        """Xóa kế hoạch điều trị."""
        confirm = QMessageBox.question(
            self,
            "Xác nhận xóa",
            "Bạn có chắc muốn xóa kế hoạch điều trị này không?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Triển khai xóa kế hoạch
            # Đây là phần sẽ kết nối với module kế hoạch
            QMessageBox.information(
                self,
                "Thành công",
                "Đã xóa kế hoạch điều trị"
            )
            
            # Làm mới danh sách
            self._load_patients()
    
    def refresh(self):
        """Làm mới danh sách bệnh nhân."""
        self._load_patients()
