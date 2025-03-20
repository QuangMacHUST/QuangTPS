#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module giao diện người dùng cho việc quản lý cấu trúc.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QListWidgetItem, QSplitter, QDialog, QColorDialog, QComboBox, 
    QLineEdit, QFormLayout, QMessageBox, QFileDialog, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QProgressDialog, QMenu, QAction
)
from PyQt5.QtGui import QColor, QIcon, QBrush, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QSize

from quangtps.structure.structure import Structure, StructureSet, Contour, Point
from quangtps.database.structure_db import StructureDatabase
from quangtps.imaging.image import Image
from quangtps.database.patient_db import PatientDatabase
from quangtps.core.exceptions import StructureError
from quangtps.core.logging import get_logger
from quangtps.ui.widgets.viewport_3d import Viewport3D
from quangtps.ui.widgets.slice_viewer import SliceViewer

logger = get_logger(__name__)


class StructureDialog(QDialog):
    """Hộp thoại để tạo/chỉnh sửa cấu trúc."""
    
    def __init__(self, parent=None, structure: Structure = None):
        """
        Khởi tạo hộp thoại tạo/chỉnh sửa cấu trúc.
        
        Args:
            parent: Widget cha
            structure: Cấu trúc cần chỉnh sửa (None nếu tạo mới)
        """
        super().__init__(parent)
        self.structure = structure
        self.setup_ui()
        
    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        self.setWindowTitle("Structure Properties" if self.structure else "Create Structure")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Form layout
        form_layout = QFormLayout()
        
        # Tên cấu trúc
        self.name_edit = QLineEdit()
        if self.structure:
            self.name_edit.setText(self.structure.name)
        form_layout.addRow("Name:", self.name_edit)
        
        # Loại cấu trúc
        self.type_combo = QComboBox()
        structure_types = ["PTV", "CTV", "GTV", "OAR", "BODY", "EXTERNAL", "OTHER"]
        self.type_combo.addItems(structure_types)
        if self.structure and self.structure.type:
            index = self.type_combo.findText(self.structure.type)
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
        form_layout.addRow("Type:", self.type_combo)
        
        # Chọn màu
        color_layout = QHBoxLayout()
        self.color_button = QPushButton()
        self.color_button.setMinimumWidth(80)
        self.color_button.clicked.connect(self.select_color)
        
        color = QColor("#FF0000")  # Màu mặc định
        if self.structure and self.structure.color:
            color = QColor(self.structure.color)
            
        # Cập nhật màu nền của nút
        self.set_button_color(color)
        
        color_layout.addWidget(self.color_button)
        form_layout.addRow("Color:", color_layout)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
    def select_color(self):
        """Mở hộp thoại chọn màu."""
        current_color = QColor(self.color_button.property("color") or "#FF0000")
        color = QColorDialog.getColor(current_color, self, "Select Color")
        
        if color.isValid():
            self.set_button_color(color)
            
    def set_button_color(self, color: QColor):
        """Cập nhật màu nền của nút."""
        self.color_button.setStyleSheet(
            f"background-color: {color.name()}; color: {'white' if color.lightness() < 128 else 'black'};"
        )
        self.color_button.setText(color.name())
        self.color_button.setProperty("color", color.name())
        
    def get_structure_data(self) -> Dict:
        """Lấy dữ liệu cấu trúc từ form."""
        return {
            'name': self.name_edit.text(),
            'type': self.type_combo.currentText(),
            'color': self.color_button.property("color") or "#FF0000"
        }


class StructureEditorWidget(QWidget):
    """Widget quản lý cấu trúc."""
    
    structureChanged = pyqtSignal(Structure)
    structureSetChanged = pyqtSignal(StructureSet)
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget quản lý cấu trúc.
        
        Args:
            parent: Widget cha
        """
        super().__init__(parent)
        
        self.structure_db = StructureDatabase()
        self.patient_db = PatientDatabase()
        
        self.current_patient_id = None
        self.current_structure_set = None
        self.current_structure = None
        self.current_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        main_layout = QVBoxLayout(self)
        
        # Label thông tin bệnh nhân
        self.patient_label = QLabel("No patient selected")
        main_layout.addWidget(self.patient_label)
        
        # Tạo QSplitter để chia màn hình
        splitter = QSplitter(Qt.Horizontal)
        
        # Panel bên trái
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Danh sách cấu trúc
        structure_label = QLabel("Structures:")
        self.structure_list = QListWidget()
        self.structure_list.setSelectionMode(QListWidget.SingleSelection)
        self.structure_list.itemSelectionChanged.connect(self.structure_selected)
        
        # Nút quản lý cấu trúc
        structure_button_layout = QHBoxLayout()
        self.add_structure_button = QPushButton("Add")
        self.add_structure_button.clicked.connect(self.add_structure)
        self.edit_structure_button = QPushButton("Edit")
        self.edit_structure_button.clicked.connect(self.edit_structure)
        self.delete_structure_button = QPushButton("Delete")
        self.delete_structure_button.clicked.connect(self.delete_structure)
        
        structure_button_layout.addWidget(self.add_structure_button)
        structure_button_layout.addWidget(self.edit_structure_button)
        structure_button_layout.addWidget(self.delete_structure_button)
        
        left_layout.addWidget(structure_label)
        left_layout.addWidget(self.structure_list)
        left_layout.addLayout(structure_button_layout)
        
        # Panel bên phải
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # Tab thông tin
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        
        # Thông tin cấu trúc
        self.structure_info = QTreeWidget()
        self.structure_info.setHeaderLabels(["Property", "Value"])
        self.structure_info.setColumnCount(2)
        self.structure_info.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.structure_info.header().setSectionResizeMode(1, QHeaderView.Stretch)
        
        info_layout.addWidget(self.structure_info)
        
        # Tab hiển thị
        viewer_tab = QWidget()
        viewer_layout = QVBoxLayout(viewer_tab)
        
        # Placeholder cho viewport 3D và slice viewer
        self.viewer_placeholder = QLabel("3D viewer will be shown here")
        self.viewer_placeholder.setAlignment(Qt.AlignCenter)
        self.viewer_placeholder.setStyleSheet("background-color: #f0f0f0;")
        
        viewer_layout.addWidget(self.viewer_placeholder)
        
        # Thêm các tab
        self.tabs.addTab(info_tab, "Information")
        self.tabs.addTab(viewer_tab, "Viewer")
        
        right_layout.addWidget(self.tabs)
        
        # Thêm các panel vào splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Đặt kích thước ban đầu cho splitter
        splitter.setSizes([200, 600])
        
        main_layout.addWidget(splitter)
        
        # Cập nhật trạng thái các nút
        self.update_ui_state()
        
    def set_patient(self, patient_id: str):
        """
        Đặt bệnh nhân hiện tại và nạp các cấu trúc.
        
        Args:
            patient_id: ID của bệnh nhân
        """
        if not patient_id:
            return
            
        self.current_patient_id = patient_id
        
        # Nạp thông tin bệnh nhân
        patient = self.patient_db.get_patient(patient_id)
        if patient:
            self.patient_label.setText(f"Patient: {patient.get('name', '')} (ID: {patient_id})")
        else:
            self.patient_label.setText(f"Patient ID: {patient_id}")
            
        # Nạp tập hợp cấu trúc
        self.load_structure_set()
        
    def set_image(self, image: Image):
        """
        Đặt hình ảnh hiện tại.
        
        Args:
            image: Đối tượng Image
        """
        self.current_image = image
        
    def load_structure_set(self):
        """Nạp tập hợp cấu trúc của bệnh nhân hiện tại."""
        if not self.current_patient_id:
            return
            
        try:
            # Nạp tập hợp cấu trúc
            self.current_structure_set = self.structure_db.load_structure_set(self.current_patient_id)
            
            # Cập nhật danh sách cấu trúc
            self.update_structure_list()
            
            # Phát tín hiệu thông báo thay đổi
            self.structureSetChanged.emit(self.current_structure_set)
            
        except Exception as e:
            logger.error("Lỗi khi nạp tập hợp cấu trúc: %s", str(e), exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load structures: {str(e)}")
            
    def update_structure_list(self):
        """Cập nhật danh sách cấu trúc."""
        self.structure_list.clear()
        
        if not self.current_structure_set:
            return
            
        # Thêm các cấu trúc vào danh sách
        for structure in self.current_structure_set.structures:
            item = QListWidgetItem(structure.name)
            item.setData(Qt.UserRole, structure.id)
            
            # Đặt màu cho item
            if structure.color:
                color = QColor(structure.color)
                item.setForeground(QBrush(color))
                
            self.structure_list.addItem(item)
            
        # Cập nhật trạng thái UI
        self.update_ui_state()
        
    def structure_selected(self):
        """Xử lý sự kiện khi chọn cấu trúc."""
        selected_items = self.structure_list.selectedItems()
        
        if not selected_items:
            self.current_structure = None
            self.update_structure_info()
            self.update_ui_state()
            return
            
        # Lấy cấu trúc được chọn
        structure_id = selected_items[0].data(Qt.UserRole)
        
        if self.current_structure_set:
            self.current_structure = self.current_structure_set.get_structure(structure_id)
            
        # Cập nhật thông tin và UI
        self.update_structure_info()
        self.update_ui_state()
        
        # Phát tín hiệu thông báo thay đổi
        if self.current_structure:
            self.structureChanged.emit(self.current_structure)
            
    def update_structure_info(self):
        """Cập nhật thông tin cấu trúc."""
        self.structure_info.clear()
        
        if not self.current_structure:
            return
            
        # Thêm thông tin cơ bản
        self.add_info_item("ID", self.current_structure.id)
        self.add_info_item("Name", self.current_structure.name)
        self.add_info_item("Type", self.current_structure.type)
        self.add_info_item("Color", self.current_structure.color)
        self.add_info_item("Number of Contours", str(len(self.current_structure.contours)))
        
        # Thêm thông tin về thể tích
        volume = self.current_structure.volume()
        self.add_info_item("Volume", f"{volume:.2f} cm³")
        
        # Thêm metadata
        if self.current_structure.metadata:
            metadata_item = QTreeWidgetItem(["Metadata", ""])
            self.structure_info.addTopLevelItem(metadata_item)
            
            for key, value in self.current_structure.metadata.items():
                QTreeWidgetItem(metadata_item, [key, str(value)])
                
        # Mở rộng tất cả các mục
        self.structure_info.expandAll()
        
    def add_info_item(self, property_name: str, value: str):
        """Thêm một mục thông tin vào tree widget."""
        item = QTreeWidgetItem([property_name, value])
        self.structure_info.addTopLevelItem(item)
        
    def update_ui_state(self):
        """Cập nhật trạng thái các điều khiển."""
        has_patient = self.current_patient_id is not None
        has_structure = self.current_structure is not None
        
        # Cập nhật trạng thái các nút
        self.add_structure_button.setEnabled(has_patient)
        self.edit_structure_button.setEnabled(has_structure)
        self.delete_structure_button.setEnabled(has_structure)
        
    def add_structure(self):
        """Thêm cấu trúc mới."""
        if not self.current_patient_id:
            return
            
        dialog = StructureDialog(self)
        
        if dialog.exec_() == QDialog.Accepted:
            structure_data = dialog.get_structure_data()
            
            try:
                # Tạo cấu trúc mới
                structure = Structure(
                    name=structure_data['name'],
                    type=structure_data['type'],
                    color=structure_data['color']
                )
                
                # Thêm vào tập hợp cấu trúc
                self.current_structure_set.add_structure(structure)
                
                # Lưu vào cơ sở dữ liệu
                self.structure_db.save_structure(structure, self.current_patient_id)
                
                # Cập nhật danh sách
                self.update_structure_list()
                
                # Chọn cấu trúc mới tạo
                for i in range(self.structure_list.count()):
                    item = self.structure_list.item(i)
                    if item.data(Qt.UserRole) == structure.id:
                        self.structure_list.setCurrentItem(item)
                        break
                
            except Exception as e:
                logger.error("Lỗi khi thêm cấu trúc: %s", str(e), exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to add structure: {str(e)}")
                
    def edit_structure(self):
        """Chỉnh sửa cấu trúc được chọn."""
        if not self.current_structure:
            return
            
        dialog = StructureDialog(self, self.current_structure)
        
        if dialog.exec_() == QDialog.Accepted:
            structure_data = dialog.get_structure_data()
            
            try:
                # Cập nhật cấu trúc
                self.current_structure.name = structure_data['name']
                self.current_structure.type = structure_data['type']
                self.current_structure.color = structure_data['color']
                
                # Lưu vào cơ sở dữ liệu
                self.structure_db.save_structure(self.current_structure, self.current_patient_id)
                
                # Cập nhật danh sách và thông tin
                self.update_structure_list()
                self.update_structure_info()
                
                # Phát tín hiệu thông báo thay đổi
                self.structureChanged.emit(self.current_structure)
                
            except Exception as e:
                logger.error("Lỗi khi chỉnh sửa cấu trúc: %s", str(e), exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to update structure: {str(e)}")
                
    def delete_structure(self):
        """Xóa cấu trúc được chọn."""
        if not self.current_structure:
            return
            
        # Hiển thị hộp thoại xác nhận
        reply = QMessageBox.question(
            self, "Delete Structure", 
            f"Are you sure you want to delete '{self.current_structure.name}'?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            structure_id = self.current_structure.id
            
            # Xóa khỏi cơ sở dữ liệu
            self.structure_db.delete_structure(structure_id)
            
            # Xóa khỏi tập hợp cấu trúc
            self.current_structure_set.remove_structure(structure_id)
            
            # Đặt lại cấu trúc hiện tại
            self.current_structure = None
            
            # Cập nhật UI
            self.update_structure_list()
            self.update_structure_info()
            
            # Phát tín hiệu thông báo thay đổi
            self.structureSetChanged.emit(self.current_structure_set)
            
        except Exception as e:
            logger.error("Lỗi khi xóa cấu trúc: %s", str(e), exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to delete structure: {str(e)}")
