#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module hiển thị và quản lý cấu trúc.

Module này cung cấp giao diện để xem và quản lý các cấu trúc giải phẫu
được tạo thông qua phân đoạn hình ảnh.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Set

from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTreeWidget,
    QTreeWidgetItem, QMenu, QAction, QMessageBox, QInputDialog, QLineEdit,
    QComboBox, QHeaderView, QSizePolicy, QSplitter, QFrame, QToolButton,
    QDialog, QDialogButtonBox, QFormLayout, QDateEdit, QTextEdit, QColorDialog,
    QCheckBox, QSlider, QSpinBox, QGroupBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap, QBrush

from quangtps.core.logging import get_logger
from quangtps.imaging.structures import Structure, StructureSet
from quangtps.imaging.contour import Contour, ContourCollection

logger = get_logger(__name__)

class StructureView(QWidget):
    """Widget hiển thị và quản lý các cấu trúc giải phẫu."""
    
    # Tín hiệu khi có thay đổi cấu trúc
    structure_visibility_changed = pyqtSignal(str, bool)  # structure_id, visible
    structure_selected = pyqtSignal(str)  # structure_id
    structure_color_changed = pyqtSignal(str, QColor)  # structure_id, color
    structure_opacity_changed = pyqtSignal(str, float)  # structure_id, opacity
    
    def __init__(self, parent=None):
        """Khởi tạo StructureView."""
        super().__init__(parent)
        self.structures = {}  # Dict[str, Structure]
        self.structure_set = None  # StructureSet hiện tại
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Tiêu đề
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<b>Cấu trúc giải phẫu</b>"))
        
        # Nút tạo cấu trúc mới
        new_structure_btn = QPushButton("Tạo mới")
        new_structure_btn.clicked.connect(self._create_new_structure)
        header_layout.addWidget(new_structure_btn)
        
        main_layout.addLayout(header_layout)
        
        # Bảng hiển thị cấu trúc
        self.structure_table = QTableWidget()
        self.structure_table.setColumnCount(4)
        self.structure_table.setHorizontalHeaderLabels(["", "Tên", "Loại", "Thể tích"])
        self.structure_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.structure_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.structure_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.structure_table.setSelectionMode(QTableWidget.SingleSelection)
        self.structure_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.structure_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.structure_table.customContextMenuRequested.connect(self._show_context_menu)
        self.structure_table.itemSelectionChanged.connect(self._on_selection_changed)
        
        main_layout.addWidget(self.structure_table)
        
        # Khu vực thuộc tính của cấu trúc
        properties_group = QGroupBox("Thuộc tính")
        properties_layout = QFormLayout()
        
        # Độ mờ
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(70)  # Mặc định 70%
        self.opacity_slider.setEnabled(False)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        
        self.opacity_value = QLabel("70%")
        
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_value)
        
        properties_layout.addRow("Độ mờ:", opacity_layout)
        
        # Màu sắc
        color_layout = QHBoxLayout()
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(24, 24)
        self.color_btn.setEnabled(False)
        self.color_btn.clicked.connect(self._change_color)
        
        color_layout.addWidget(self.color_btn)
        color_layout.addStretch()
        
        properties_layout.addRow("Màu sắc:", color_layout)
        
        properties_group.setLayout(properties_layout)
        main_layout.addWidget(properties_group)
        
        # Thông tin thống kê
        stats_group = QGroupBox("Thống kê")
        stats_layout = QFormLayout()
        
        self.volume_label = QLabel("-")
        self.centroid_label = QLabel("-")
        self.min_hu_label = QLabel("-")
        self.max_hu_label = QLabel("-")
        self.mean_hu_label = QLabel("-")
        
        stats_layout.addRow("Thể tích:", self.volume_label)
        stats_layout.addRow("Tâm:", self.centroid_label)
        stats_layout.addRow("Min HU:", self.min_hu_label)
        stats_layout.addRow("Max HU:", self.max_hu_label)
        stats_layout.addRow("Mean HU:", self.mean_hu_label)
        
        stats_group.setLayout(stats_layout)
        main_layout.addWidget(stats_group)
        
        # Nút điều khiển
        control_layout = QHBoxLayout()
        
        delete_btn = QPushButton("Xóa cấu trúc")
        delete_btn.clicked.connect(self._delete_selected_structure)
        
        show_all_btn = QPushButton("Hiện tất cả")
        show_all_btn.clicked.connect(self._show_all_structures)
        
        hide_all_btn = QPushButton("Ẩn tất cả")
        hide_all_btn.clicked.connect(self._hide_all_structures)
        
        control_layout.addWidget(delete_btn)
        control_layout.addWidget(show_all_btn)
        control_layout.addWidget(hide_all_btn)
        
        main_layout.addLayout(control_layout)
    
    def load_structure_set(self, structure_set: StructureSet):
        """Tải một tập hợp cấu trúc."""
        self.structure_set = structure_set
        self.structures = {s.structure_id: s for s in structure_set.structures}
        self._update_structure_table()
    
    def add_structure(self, structure: Structure):
        """Thêm một cấu trúc mới."""
        if not self.structure_set:
            logger.warning("Không có structure set nào được tải.")
            return
            
        # Thêm cấu trúc vào tập hợp hiện tại
        self.structure_set.add_structure(structure)
        self.structures[structure.structure_id] = structure
        
        # Cập nhật bảng
        self._update_structure_table()
        
        # Chọn cấu trúc mới
        self._select_structure(structure.structure_id)
    
    def update_structure(self, structure: Structure):
        """Cập nhật một cấu trúc đã tồn tại."""
        if structure.structure_id not in self.structures:
            logger.warning(f"Cấu trúc {structure.structure_id} không tồn tại.")
            return
            
        self.structures[structure.structure_id] = structure
        
        # Cập nhật bảng
        self._update_structure_table()
    
    def remove_structure(self, structure_id: str):
        """Xóa một cấu trúc."""
        if structure_id not in self.structures:
            logger.warning(f"Cấu trúc {structure_id} không tồn tại.")
            return
            
        # Xóa cấu trúc khỏi tập hợp hiện tại
        if self.structure_set:
            self.structure_set.remove_structure(structure_id)
            
        # Xóa khỏi từ điển cục bộ
        del self.structures[structure_id]
        
        # Cập nhật bảng
        self._update_structure_table()
        
        # Xóa thông tin hiển thị
        self._clear_structure_info()
    
    def get_selected_structure(self) -> Optional[Structure]:
        """Lấy cấu trúc đang được chọn."""
        selected_items = self.structure_table.selectedItems()
        if not selected_items:
            return None
            
        row = selected_items[0].row()
        structure_id = self.structure_table.item(row, 1).data(Qt.UserRole)
        
        return self.structures.get(structure_id)
    
    def _update_structure_table(self):
        """Cập nhật bảng cấu trúc."""
        self.structure_table.setRowCount(0)
        
        if not self.structures:
            return
            
        for i, (structure_id, structure) in enumerate(self.structures.items()):
            self.structure_table.insertRow(i)
            
            # Checkbox hiển thị
            check_item = QTableWidgetItem()
            check_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            check_item.setCheckState(Qt.Checked if structure.visible else Qt.Unchecked)
            self.structure_table.setItem(i, 0, check_item)
            
            # Tên cấu trúc
            name_item = QTableWidgetItem(structure.name)
            name_item.setData(Qt.UserRole, structure.structure_id)
            
            # Hiển thị màu sắc
            if hasattr(structure, 'color') and structure.color:
                name_item.setForeground(QBrush(QColor(structure.color)))
                
            self.structure_table.setItem(i, 1, name_item)
            
            # Loại cấu trúc
            self.structure_table.setItem(i, 2, QTableWidgetItem(structure.type))
            
            # Thể tích
            volume_str = f"{structure.volume:.2f} cc" if hasattr(structure, 'volume') and structure.volume else "-"
            self.structure_table.setItem(i, 3, QTableWidgetItem(volume_str))
    
    def _on_selection_changed(self):
        """Xử lý sự kiện khi chọn cấu trúc thay đổi."""
        structure = self.get_selected_structure()
        if not structure:
            self._clear_structure_info()
            return
            
        # Phát tín hiệu
        self.structure_selected.emit(structure.structure_id)
        
        # Cập nhật thông tin hiển thị
        self._update_structure_info(structure)
    
    def _update_structure_info(self, structure: Structure):
        """Cập nhật thông tin hiển thị của cấu trúc."""
        # Cập nhật điều khiển thuộc tính
        self.opacity_slider.setEnabled(True)
        self.opacity_slider.setValue(int(structure.opacity * 100) if hasattr(structure, 'opacity') else 70)
        self.opacity_value.setText(f"{self.opacity_slider.value()}%")
        
        self.color_btn.setEnabled(True)
        if hasattr(structure, 'color') and structure.color:
            self.color_btn.setStyleSheet(f"background-color: {structure.color};")
        else:
            self.color_btn.setStyleSheet("")
        
        # Cập nhật thông tin thống kê
        self.volume_label.setText(f"{structure.volume:.2f} cc" if hasattr(structure, 'volume') and structure.volume else "-")
        
        if hasattr(structure, 'centroid') and structure.centroid is not None:
            centroid = structure.centroid
            self.centroid_label.setText(f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
        else:
            self.centroid_label.setText("-")
        
        # Cập nhật thông tin HU (nếu có)
        if hasattr(structure, 'min_hu') and structure.min_hu is not None:
            self.min_hu_label.setText(f"{structure.min_hu}")
        else:
            self.min_hu_label.setText("-")
            
        if hasattr(structure, 'max_hu') and structure.max_hu is not None:
            self.max_hu_label.setText(f"{structure.max_hu}")
        else:
            self.max_hu_label.setText("-")
            
        if hasattr(structure, 'mean_hu') and structure.mean_hu is not None:
            self.mean_hu_label.setText(f"{structure.mean_hu:.1f}")
        else:
            self.mean_hu_label.setText("-")
    
    def _clear_structure_info(self):
        """Xóa thông tin hiển thị."""
        self.opacity_slider.setEnabled(False)
        self.opacity_slider.setValue(70)
        self.opacity_value.setText("70%")
        
        self.color_btn.setEnabled(False)
        self.color_btn.setStyleSheet("")
        
        self.volume_label.setText("-")
        self.centroid_label.setText("-")
        self.min_hu_label.setText("-")
        self.max_hu_label.setText("-")
        self.mean_hu_label.setText("-")
    
    def _show_context_menu(self, position):
        """Hiển thị menu ngữ cảnh cho cấu trúc được chọn."""
        selected_items = self.structure_table.selectedItems()
        if not selected_items:
            return
            
        row = selected_items[0].row()
        structure_id = self.structure_table.item(row, 1).data(Qt.UserRole)
        structure = self.structures.get(structure_id)
        
        if not structure:
            return
            
        context_menu = QMenu(self)
        
        # Các hành động chính
        rename_action = QAction("Đổi tên", self)
        rename_action.triggered.connect(lambda: self._rename_structure(structure_id))
        
        delete_action = QAction("Xóa", self)
        delete_action.triggered.connect(lambda: self._delete_structure(structure_id))
        
        # Hành động về màu sắc
        color_action = QAction("Đổi màu", self)
        color_action.triggered.connect(lambda: self._change_color(structure_id))
        
        # Thêm hành động vào menu
        context_menu.addAction(rename_action)
        context_menu.addAction(delete_action)
        context_menu.addSeparator()
        context_menu.addAction(color_action)
        
        context_menu.exec_(self.structure_table.mapToGlobal(position))
    
    def _create_new_structure(self):
        """Tạo một cấu trúc mới."""
        if not self.structure_set:
            QMessageBox.warning(self, "Lỗi", "Không có structure set nào được tải.")
            return
            
        # Hiển thị dialog để nhập thông tin
        dialog = StructureDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
            
        structure_data = dialog.get_structure_data()
        
        try:
            # Tạo cấu trúc mới
            structure = Structure(
                name=structure_data["name"],
                type=structure_data["type"],
                color=structure_data["color"]
            )
            
            # Thêm vào tập hợp
            self.add_structure(structure)
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo cấu trúc mới: {e}")
            QMessageBox.warning(self, "Lỗi", f"Không thể tạo cấu trúc mới: {e}")
    
    def _rename_structure(self, structure_id: str):
        """Đổi tên cấu trúc."""
        structure = self.structures.get(structure_id)
        if not structure:
            return
            
        new_name, ok = QInputDialog.getText(
            self, "Đổi tên cấu trúc", "Nhập tên mới:",
            QLineEdit.Normal, structure.name
        )
        
        if not ok or not new_name.strip():
            return
            
        # Cập nhật tên
        structure.name = new_name.strip()
        self.update_structure(structure)
    
    def _delete_structure(self, structure_id: str):
        """Xóa cấu trúc."""
        structure = self.structures.get(structure_id)
        if not structure:
            return
            
        reply = QMessageBox.question(
            self, 
            "Xác nhận xóa", 
            f"Bạn có chắc chắn muốn xóa cấu trúc '{structure.name}'?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        self.remove_structure(structure_id)
    
    def _delete_selected_structure(self):
        """Xóa cấu trúc đang chọn."""
        structure = self.get_selected_structure()
        if not structure:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một cấu trúc để xóa.")
            return
            
        self._delete_structure(structure.structure_id)
    
    def _change_color(self, structure_id=None):
        """Thay đổi màu sắc của cấu trúc."""
        if structure_id is None:
            structure = self.get_selected_structure()
            if not structure:
                return
            structure_id = structure.structure_id
        
        structure = self.structures.get(structure_id)
        if not structure:
            return
            
        current_color = QColor(structure.color) if hasattr(structure, 'color') and structure.color else QColor(255, 0, 0)
        
        color = QColorDialog.getColor(current_color, self, "Chọn màu")
        if not color.isValid():
            return
            
        # Cập nhật màu
        structure.color = color.name()
        self.update_structure(structure)
        
        # Cập nhật nút màu
        if structure_id == self.get_selected_structure().structure_id:
            self.color_btn.setStyleSheet(f"background-color: {color.name()};")
            
        # Phát tín hiệu
        self.structure_color_changed.emit(structure_id, color)
    
    def _on_opacity_changed(self):
        """Xử lý sự kiện khi độ mờ thay đổi."""
        structure = self.get_selected_structure()
        if not structure:
            return
            
        opacity = self.opacity_slider.value() / 100.0
        self.opacity_value.setText(f"{self.opacity_slider.value()}%")
        
        # Cập nhật độ mờ
        structure.opacity = opacity
        self.update_structure(structure)
        
        # Phát tín hiệu
        self.structure_opacity_changed.emit(structure.structure_id, opacity)
    
    def _select_structure(self, structure_id: str):
        """Chọn một cấu trúc cụ thể."""
        if structure_id not in self.structures:
            return
            
        # Tìm hàng chứa cấu trúc
        for row in range(self.structure_table.rowCount()):
            item = self.structure_table.item(row, 1)
            if item and item.data(Qt.UserRole) == structure_id:
                self.structure_table.selectRow(row)
                break
    
    def _show_all_structures(self):
        """Hiển thị tất cả các cấu trúc."""
        if not self.structures:
            return
            
        for structure_id, structure in self.structures.items():
            structure.visible = True
            self.structure_visibility_changed.emit(structure_id, True)
            
        self._update_structure_table()
    
    def _hide_all_structures(self):
        """Ẩn tất cả các cấu trúc."""
        if not self.structures:
            return
            
        for structure_id, structure in self.structures.items():
            structure.visible = False
            self.structure_visibility_changed.emit(structure_id, False)
            
        self._update_structure_table()


class StructureDialog(QDialog):
    """Dialog để tạo một cấu trúc mới."""
    
    def __init__(self, parent=None):
        """Khởi tạo dialog."""
        super().__init__(parent)
        self.setWindowTitle("Tạo cấu trúc mới")
        self.setMinimumWidth(400)
        
        self.color = "#FF0000"  # Màu mặc định là đỏ
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện dialog."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        form_layout = QFormLayout()
        
        # Trường nhập tên
        self.name_input = QLineEdit()
        form_layout.addRow("Tên cấu trúc:", self.name_input)
        
        # Trường chọn loại
        self.type_combo = QComboBox()
        self.type_combo.addItem("PTV", "PTV")
        self.type_combo.addItem("CTV", "CTV")
        self.type_combo.addItem("GTV", "GTV")
        self.type_combo.addItem("Cơ quan nguy cấp", "OAR")
        self.type_combo.addItem("Tham chiếu", "REFERENCE")
        self.type_combo.addItem("Khác", "OTHER")
        form_layout.addRow("Loại cấu trúc:", self.type_combo)
        
        # Trường chọn màu
        color_layout = QHBoxLayout()
        
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(24, 24)
        self.color_btn.setStyleSheet(f"background-color: {self.color};")
        self.color_btn.clicked.connect(self._select_color)
        
        color_layout.addWidget(self.color_btn)
        color_layout.addStretch()
        
        form_layout.addRow("Màu sắc:", color_layout)
        
        layout.addLayout(form_layout)
        
        # Nút điều khiển
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addSpacing(10)
        layout.addWidget(button_box)
    
    def _select_color(self):
        """Chọn màu sắc cho cấu trúc."""
        color = QColorDialog.getColor(QColor(self.color), self, "Chọn màu")
        if color.isValid():
            self.color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {self.color};")
    
    def get_structure_data(self) -> Dict[str, Any]:
        """Lấy dữ liệu cấu trúc từ dialog."""
        return {
            "name": self.name_input.text().strip(),
            "type": self.type_combo.currentData(),
            "color": self.color
        }
        
    def accept(self):
        """Xác thực dữ liệu trước khi chấp nhận dialog."""
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên cấu trúc.")
            return
            
        super().accept()
