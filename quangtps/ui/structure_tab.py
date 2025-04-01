#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab Structure/Contour cho QuangTPS.

Module này triển khai giao diện Structure tab tương tự Eclipse của Varian,
cho phép người dùng vẽ, quản lý và chỉnh sửa structure và contour.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any, Set

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QListWidgetItem, QSplitter, QDialog, QColorDialog, QComboBox, 
    QLineEdit, QFormLayout, QMessageBox, QFileDialog, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QProgressDialog, QMenu, QAction,
    QToolBar, QGroupBox, QRadioButton, QButtonGroup, QCheckBox, QSlider,
    QSpinBox, QDoubleSpinBox, QToolButton, QFrame, QScrollArea, QGridLayout,
    QInputDialog
)
from PyQt5.QtGui import QColor, QIcon, QBrush, QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect

try:
    from quangtps.segmentation.structures.structure import (
        Structure, StructureType, StructurePriority
    )
    from quangtps.segmentation.structures.structure_set import StructureSet
    from quangtps.segmentation.contour.contour_manager import ContourManager
    from quangtps.segmentation.contour.polygon_tool import PolygonTool
    from quangtps.segmentation.contour.margin import MarginTool
    from quangtps.segmentation.contour.boolean_operations import BooleanOperator
    from quangtps.segmentation.contour.interpolation import ContourInterpolator
    from quangtps.segmentation.contour.advanced_editing import AdvancedContourEditor
    from quangtps.segmentation.auto_segmentation.semi_automatic import SemiAutomaticSegmentation
    from quangtps.segmentation.auto.engine import AutoSegmentationEngine
    from quangtps.ui.image_display import ImageDisplayWidget
    from quangtps.imaging.image import Image
    from quangtps.core.patient import Patient
    from quangtps.core.services import ServiceRegistry
    from quangtps.ui.mpr_viewer import MPRViewer
except ImportError as e:
    logging.error(f"Error importing structure modules: {e}")

logger = logging.getLogger(__name__)

class StructureListWidget(QWidget):
    """
    Widget hiển thị danh sách các cấu trúc và cho phép tương tác với chúng.
    
    Widget này hiển thị danh sách cấu trúc có thể tìm kiếm, chọn, hiển thị/ẩn,
    và quản lý các thuộc tính khác của cấu trúc.
    """
    
    structureSelected = pyqtSignal(object)  # Tín hiệu khi một cấu trúc được chọn
    structureChanged = pyqtSignal()         # Tín hiệu khi một cấu trúc thay đổi (tên, màu, v.v.)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.structure_set = None
        self.current_structure = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout(self)
        
        # Thanh tiêu đề
        title_layout = QHBoxLayout()
        title_label = QLabel("Danh sách cấu trúc")
        title_label.setStyleSheet("font-weight: bold; color: white; background-color: #2c3e50;")
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        main_layout.addLayout(title_layout)
        
        # Thanh tìm kiếm
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Tìm kiếm:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Nhập tên cấu trúc...")
        self.search_edit.textChanged.connect(self._filter_structures)
        search_layout.addWidget(self.search_edit)
        main_layout.addLayout(search_layout)
        
        # TreeWidget để hiển thị danh sách cấu trúc
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Name", "Type", "Visible"])
        self.tree_widget.setColumnCount(3)
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree_widget.setRootIsDecorated(False)
        self.tree_widget.itemSelectionChanged.connect(self._on_structure_selected)
        self.tree_widget.itemDoubleClicked.connect(self._on_structure_double_clicked)
        
        header = self.tree_widget.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        main_layout.addWidget(self.tree_widget)
        
        # Nút thao tác với cấu trúc
        buttons_layout = QHBoxLayout()
        
        # Nút tạo cấu trúc mới
        self.new_btn = QPushButton("Tạo mới")
        self.new_btn.clicked.connect(self._on_new_structure)
        buttons_layout.addWidget(self.new_btn)
        
        # Nút chỉnh sửa thuộc tính
        self.edit_btn = QPushButton("Chỉnh sửa")
        self.edit_btn.clicked.connect(self._on_edit_structure)
        buttons_layout.addWidget(self.edit_btn)
        
        # Nút xóa cấu trúc
        self.delete_btn = QPushButton("Xóa")
        self.delete_btn.clicked.connect(self._on_delete_structure)
        buttons_layout.addWidget(self.delete_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # Vùng thuộc tính của cấu trúc đã chọn
        self.properties_group = QGroupBox("Thuộc tính")
        properties_layout = QFormLayout()
        
        # Tên cấu trúc
        name_layout = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._on_name_changed)
        name_layout.addWidget(self.name_edit)
        properties_layout.addRow("Tên:", name_layout)
        
        # Loại cấu trúc
        type_layout = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["PTV", "CTV", "GTV", "ORGAN", "EXTERNAL", "OTHER"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)
        properties_layout.addRow("Loại:", type_layout)
        
        # Màu sắc
        color_layout = QHBoxLayout()
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(24, 24)
        self.color_btn.clicked.connect(self._on_color_clicked)
        color_layout.addWidget(self.color_btn)
        self.color_label = QLabel("RGB: 255, 0, 0")
        color_layout.addWidget(self.color_label)
        properties_layout.addRow("Màu:", color_layout)
        
        # Độ mờ
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("100%")
        opacity_layout.addWidget(self.opacity_label)
        properties_layout.addRow("Độ mờ:", opacity_layout)
        
        # Hiển thị/ẩn
        visible_layout = QHBoxLayout()
        self.visible_check = QCheckBox("Hiển thị")
        self.visible_check.setChecked(True)
        self.visible_check.stateChanged.connect(self._on_visible_changed)
        visible_layout.addWidget(self.visible_check)
        properties_layout.addRow("", visible_layout)
        
        self.properties_group.setLayout(properties_layout)
        main_layout.addWidget(self.properties_group)
        
        # Vô hiệu hóa vùng thuộc tính ban đầu
        self.properties_group.setEnabled(False)
    
    def set_structure_set(self, structure_set):
        """Thiết lập bộ cấu trúc để hiển thị."""
        self.structure_set = structure_set
        self._refresh_tree()
        
        # Xóa lựa chọn hiện tại
        self.current_structure = None
        self.properties_group.setEnabled(False)
    
    def _refresh_tree(self):
        """Làm mới cây cấu trúc với dữ liệu mới nhất."""
        self.tree_widget.clear()
        
        if not self.structure_set or not hasattr(self.structure_set, 'structures'):
            return
        
        try:
            for structure in self.structure_set.structures:
                item = QTreeWidgetItem(self.tree_widget)
                item.setText(0, structure.name if hasattr(structure, 'name') else "Unknown")
                item.setText(1, structure.type if hasattr(structure, 'type') else "OTHER")
                
                # Thiết lập cờ hiển thị
                is_visible = structure.visible if hasattr(structure, 'visible') else True
                item.setText(2, "✓" if is_visible else "")
                
                # Lưu trữ cấu trúc trong item
                item.setData(0, Qt.UserRole, structure)
                
                # Thiết lập màu sắc nếu có
                if hasattr(structure, 'color'):
                    color = QColor(*structure.color) if isinstance(structure.color, (list, tuple)) else QColor(structure.color)
                    item.setForeground(0, QBrush(color))
            
            # Sắp xếp theo tên
            self.tree_widget.sortItems(0, Qt.AscendingOrder)
            
        except Exception as e:
            logger.error(f"Lỗi khi làm mới danh sách cấu trúc: {e}")
    
    def _filter_structures(self):
        """Lọc danh sách cấu trúc theo văn bản tìm kiếm."""
        search_text = self.search_edit.text().lower()
        
        for i in range(self.tree_widget.topLevelItemCount()):
            item = self.tree_widget.topLevelItem(i)
            name = item.text(0).lower()
            item.setHidden(search_text and search_text not in name)
    
    def _on_structure_selected(self):
        """Xử lý khi một cấu trúc được chọn."""
        selected_items = self.tree_widget.selectedItems()
        
        if selected_items:
            item = selected_items[0]
            structure = item.data(0, Qt.UserRole)
            self.current_structure = structure
            
            # Cập nhật UI thuộc tính
            self._update_properties_ui()
            
            # Kích hoạt vùng thuộc tính
            self.properties_group.setEnabled(True)
            
            # Phát tín hiệu
            self.structureSelected.emit(structure)
        else:
            self.current_structure = None
            self.properties_group.setEnabled(False)
    
    def _on_structure_double_clicked(self, item, column):
        """Xử lý khi cấu trúc được nhấp đúp - chuyển đổi hiển thị/ẩn."""
        if column == 2:  # Cột "Visible"
            structure = item.data(0, Qt.UserRole)
            if structure:
                # Đảo ngược trạng thái hiển thị
                is_visible = not (structure.visible if hasattr(structure, 'visible') else True)
                structure.visible = is_visible
                
                # Cập nhật hiển thị
                item.setText(2, "✓" if is_visible else "")
                
                if structure == self.current_structure:
                    self.visible_check.setChecked(is_visible)
                
                # Phát tín hiệu
                self.structureChanged.emit()
    
    def _update_properties_ui(self):
        """Cập nhật UI thuộc tính với cấu trúc đã chọn."""
        if not self.current_structure:
            return
        
        try:
            # Cập nhật tên
            self.name_edit.setText(self.current_structure.name if hasattr(self.current_structure, 'name') else "")
            
            # Cập nhật loại
            type_index = self.type_combo.findText(
                self.current_structure.type if hasattr(self.current_structure, 'type') else "OTHER"
            )
            self.type_combo.setCurrentIndex(max(0, type_index))
            
            # Cập nhật màu sắc
            if hasattr(self.current_structure, 'color'):
                color = self.current_structure.color
                if isinstance(color, (list, tuple)) and len(color) >= 3:
                    qcolor = QColor(color[0], color[1], color[2])
                    self.color_btn.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]});")
                    self.color_label.setText(f"RGB: {color[0]}, {color[1]}, {color[2]}")
                else:
                    qcolor = QColor(color)
                    self.color_btn.setStyleSheet(f"background-color: {qcolor.name()};")
                    self.color_label.setText(f"RGB: {qcolor.red()}, {qcolor.green()}, {qcolor.blue()}")
            
            # Cập nhật độ mờ
            opacity = self.current_structure.opacity * 100 if hasattr(self.current_structure, 'opacity') else 100
            self.opacity_slider.setValue(int(opacity))
            self.opacity_label.setText(f"{int(opacity)}%")
            
            # Cập nhật hiển thị
            is_visible = self.current_structure.visible if hasattr(self.current_structure, 'visible') else True
            self.visible_check.setChecked(is_visible)
        
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật UI thuộc tính: {e}")
    
    def _on_name_changed(self, text):
        """Xử lý khi tên cấu trúc thay đổi."""
        if not self.current_structure:
            return
        
        # Cập nhật tên cấu trúc
        self.current_structure.name = text
        
        # Cập nhật item trong tree
        selected_items = self.tree_widget.selectedItems()
        if selected_items:
            selected_items[0].setText(0, text)
        
        # Phát tín hiệu
        self.structureChanged.emit()
    
    def _on_type_changed(self, text):
        """Xử lý khi loại cấu trúc thay đổi."""
        if not self.current_structure:
            return
        
        # Cập nhật loại cấu trúc
        self.current_structure.type = text
        
        # Cập nhật item trong tree
        selected_items = self.tree_widget.selectedItems()
        if selected_items:
            selected_items[0].setText(1, text)
        
        # Phát tín hiệu
        self.structureChanged.emit()
    
    def _on_color_clicked(self):
        """Xử lý khi nút màu sắc được nhấp."""
        if not self.current_structure:
            return
        
        # Lấy màu hiện tại
        current_color = QColor(255, 0, 0)  # Mặc định đỏ
        if hasattr(self.current_structure, 'color'):
            color = self.current_structure.color
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                current_color = QColor(color[0], color[1], color[2])
            else:
                current_color = QColor(color)
        
        # Hiển thị hộp thoại chọn màu
        color_dialog = QColorDialog(current_color, self)
        if color_dialog.exec_():
            new_color = color_dialog.selectedColor()
            
            # Cập nhật màu sắc cấu trúc
            self.current_structure.color = (new_color.red(), new_color.green(), new_color.blue())
            
            # Cập nhật UI
            self.color_btn.setStyleSheet(f"background-color: {new_color.name()};")
            self.color_label.setText(f"RGB: {new_color.red()}, {new_color.green()}, {new_color.blue()}")
            
            # Cập nhật màu trong tree
            selected_items = self.tree_widget.selectedItems()
            if selected_items:
                selected_items[0].setForeground(0, QBrush(new_color))
            
            # Phát tín hiệu
            self.structureChanged.emit()
    
    def _on_opacity_changed(self, value):
        """Xử lý khi độ mờ thay đổi."""
        if not self.current_structure:
            return
        
        # Cập nhật nhãn
        self.opacity_label.setText(f"{value}%")
        
        # Cập nhật độ mờ cấu trúc
        self.current_structure.opacity = value / 100.0
        
        # Phát tín hiệu
        self.structureChanged.emit()
    
    def _on_visible_changed(self, state):
        """Xử lý khi trạng thái hiển thị thay đổi."""
        if not self.current_structure:
            return
        
        # Cập nhật trạng thái hiển thị
        is_visible = state == Qt.Checked
        self.current_structure.visible = is_visible
        
        # Cập nhật hiển thị trong tree
        selected_items = self.tree_widget.selectedItems()
        if selected_items:
            selected_items[0].setText(2, "✓" if is_visible else "")
        
        # Phát tín hiệu
        self.structureChanged.emit()
    
    def _on_new_structure(self):
        """Xử lý khi nút tạo cấu trúc mới được nhấp."""
        if not self.structure_set:
            return
        
        name, ok = QInputDialog.getText(
            self, "Tạo cấu trúc mới", "Tên cấu trúc:",
            QLineEdit.Normal, "New Structure"
        )
        
        if ok and name:
            try:
                # Import Structure class
                from quangtps.segmentation.structures.structure import Structure
                
                # Tạo cấu trúc mới
                new_structure = Structure(name=name)
                new_structure.id = f"STRUCT_{len(self.structure_set.structures) + 1:03d}"
                new_structure.type = "OTHER"
                new_structure.color = (255, 0, 0)  # Mặc định đỏ
                new_structure.visible = True
                
                # Thêm vào bộ cấu trúc
                self.structure_set.structures.append(new_structure)
                
                # Làm mới tree
                self._refresh_tree()
                
                # Chọn cấu trúc mới
                for i in range(self.tree_widget.topLevelItemCount()):
                    item = self.tree_widget.topLevelItem(i)
                    if item.data(0, Qt.UserRole) == new_structure:
                        self.tree_widget.setCurrentItem(item)
                        break
                
                # Phát tín hiệu
                self.structureChanged.emit()
            
            except Exception as e:
                QMessageBox.critical(
                    self, "Lỗi",
                    f"Không thể tạo cấu trúc mới: {e}",
                    QMessageBox.Ok
                )
    
    def _on_edit_structure(self):
        """Xử lý khi nút chỉnh sửa cấu trúc được nhấp."""
        if not self.current_structure:
            return
        
        # Focus vào vùng thuộc tính để chỉnh sửa
        self.name_edit.setFocus()
    
    def _on_delete_structure(self):
        """Xử lý khi nút xóa cấu trúc được nhấp."""
        if not self.current_structure or not self.structure_set:
            return
        
        # Xác nhận xóa
        reply = QMessageBox.question(
            self, "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa cấu trúc '{self.current_structure.name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Xóa khỏi bộ cấu trúc
                self.structure_set.structures.remove(self.current_structure)
                
                # Làm mới tree
                self._refresh_tree()
                
                # Xóa lựa chọn hiện tại
                self.current_structure = None
                self.properties_group.setEnabled(False)
                
                # Phát tín hiệu
                self.structureChanged.emit()
            
            except Exception as e:
                QMessageBox.critical(
                    self, "Lỗi",
                    f"Không thể xóa cấu trúc: {e}",
                    QMessageBox.Ok
                )
    
    def get_selected_structure(self):
        """Lấy cấu trúc đang được chọn."""
        return self.current_structure

class StructureDialog(QDialog):
    """Hộp thoại để tạo/chỉnh sửa cấu trúc."""
    
    def __init__(self, parent=None, structure=None):
        super().__init__(parent)
        self.structure = structure
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện hộp thoại."""
        self.setWindowTitle("Structure Properties" if self.structure else "New Structure")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Form layout
        form_layout = QFormLayout()
        
        # ID cấu trúc (chỉ đọc nếu đang chỉnh sửa)
        self.id_edit = QLineEdit()
        if self.structure and hasattr(self.structure, 'id'):
            self.id_edit.setText(self.structure.id)
            self.id_edit.setReadOnly(True)
        form_layout.addRow("ID:", self.id_edit)
        
        # Tên cấu trúc
        self.name_edit = QLineEdit()
        if self.structure and hasattr(self.structure, 'name'):
            self.name_edit.setText(self.structure.name)
        form_layout.addRow("Name:", self.name_edit)
        
        # Loại cấu trúc
        self.type_combo = QComboBox()
        structure_types = ["PTV", "CTV", "GTV", "OAR", "BODY", "EXTERNAL", "OTHER"]
        self.type_combo.addItems(structure_types)
        if self.structure and hasattr(self.structure, 'type'):
            index = self.type_combo.findText(self.structure.type)
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
        form_layout.addRow("Type:", self.type_combo)
        
        # Màu sắc
        color_layout = QHBoxLayout()
        self.color_button = QPushButton()
        self.color_button.setMinimumWidth(80)
        self.color_button.clicked.connect(self.select_color)
        
        color = QColor("#FF0000")  # Màu mặc định
        if self.structure and hasattr(self.structure, 'color') and self.structure.color:
            color = QColor(self.structure.color)
            
        # Cập nhật màu nền của nút
        self.set_button_color(color)
        
        color_layout.addWidget(self.color_button)
        form_layout.addRow("Color:", color_layout)
        
        # Độ ưu tiên
        self.priority_combo = QComboBox()
        priorities = ["HIGH", "MEDIUM", "LOW"]
        self.priority_combo.addItems(priorities)
        if self.structure and hasattr(self.structure, 'priority'):
            index = self.priority_combo.findText(str(self.structure.priority))
            if index >= 0:
                self.priority_combo.setCurrentIndex(index)
        form_layout.addRow("Priority:", self.priority_combo)
        
        # Mô tả
        self.description_edit = QLineEdit()
        if self.structure and hasattr(self.structure, 'description'):
            self.description_edit.setText(self.structure.description)
        form_layout.addRow("Description:", self.description_edit)
        
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
    
    def set_button_color(self, color):
        """Cập nhật màu nền của nút."""
        self.color_button.setStyleSheet(
            f"background-color: {color.name()}; color: {'white' if color.lightness() < 128 else 'black'};"
        )
        self.color_button.setText(color.name())
        self.color_button.setProperty("color", color.name())
    
    def get_structure_data(self):
        """Lấy dữ liệu cấu trúc từ form."""
        return {
            'id': self.id_edit.text(),
            'name': self.name_edit.text(),
            'type': self.type_combo.currentText(),
            'color': self.color_button.property("color") or "#FF0000",
            'priority': self.priority_combo.currentText(),
            'description': self.description_edit.text()
        }

class DrawingToolsWidget(QWidget):
    """
    Widget cung cấp các công cụ vẽ cấu trúc.
    
    Widget này chứa các nút và tùy chọn để vẽ, chỉnh sửa và xóa cấu trúc,
    tương tự như các công cụ trong Eclipse.
    """
    
    toolSelected = pyqtSignal(str, dict)  # Tín hiệu khi công cụ được chọn
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_tool = None
        self.tool_options = {}
        
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout(self)
        
        # Label tiêu đề
        title_label = QLabel("Công cụ vẽ")
        title_label.setStyleSheet("font-weight: bold; color: white; background-color: #2c3e50;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Grid layout cho các nút công cụ
        tools_grid = QGridLayout()
        tools_grid.setSpacing(5)
        
        # Nút vẽ bằng tay
        self.brush_btn = QPushButton("Brush")
        self.brush_btn.setCheckable(True)
        self.brush_btn.clicked.connect(lambda: self._select_tool("Brush"))
        tools_grid.addWidget(self.brush_btn, 0, 0)
        
        # Nút vẽ Smart Brush
        self.smart_brush_btn = QPushButton("Smart Brush")
        self.smart_brush_btn.setCheckable(True)
        self.smart_brush_btn.clicked.connect(lambda: self._select_tool("SmartBrush"))
        tools_grid.addWidget(self.smart_brush_btn, 0, 1)
        
        # Nút Thresholding (phân ngưỡng)
        self.threshold_btn = QPushButton("Threshold")
        self.threshold_btn.setCheckable(True)
        self.threshold_btn.clicked.connect(lambda: self._select_tool("Threshold"))
        tools_grid.addWidget(self.threshold_btn, 1, 0)
        
        # Nút xóa
        self.eraser_btn = QPushButton("Eraser")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(lambda: self._select_tool("Eraser"))
        tools_grid.addWidget(self.eraser_btn, 1, 1)
        
        # Nút tạo điểm mốc
        self.point_btn = QPushButton("Point")
        self.point_btn.setCheckable(True)
        self.point_btn.clicked.connect(lambda: self._select_tool("Point"))
        tools_grid.addWidget(self.point_btn, 2, 0)
        
        # Nút tạo công cụ
        self.grow_btn = QPushButton("Region Grow")
        self.grow_btn.setCheckable(True)
        self.grow_btn.clicked.connect(lambda: self._select_tool("RegionGrow"))
        tools_grid.addWidget(self.grow_btn, 2, 1)
        
        main_layout.addLayout(tools_grid)
        
        # Thiết lập tùy chọn công cụ
        self.options_group = QGroupBox("Tùy chọn công cụ")
        options_layout = QVBoxLayout()
        
        # Kích thước brush
        brush_size_layout = QHBoxLayout()
        brush_size_layout.addWidget(QLabel("Kích thước:"))
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(30)
        self.brush_size_slider.setValue(5)
        self.brush_size_slider.valueChanged.connect(self._update_options)
        brush_size_layout.addWidget(self.brush_size_slider)
        self.brush_size_label = QLabel("5")
        brush_size_layout.addWidget(self.brush_size_label)
        options_layout.addLayout(brush_size_layout)
        
        # Độ mờ
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Độ mờ:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._update_options)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("50%")
        opacity_layout.addWidget(self.opacity_label)
        options_layout.addLayout(opacity_layout)
        
        # Ngưỡng thấp và cao cho thresholding
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Ngưỡng:"))
        self.threshold_min_slider = QSlider(Qt.Horizontal)
        self.threshold_min_slider.setMinimum(0)
        self.threshold_min_slider.setMaximum(255)
        self.threshold_min_slider.setValue(100)
        self.threshold_min_slider.valueChanged.connect(self._update_options)
        threshold_layout.addWidget(self.threshold_min_slider)
        threshold_layout.addWidget(QLabel("-"))
        self.threshold_max_slider = QSlider(Qt.Horizontal)
        self.threshold_max_slider.setMinimum(0)
        self.threshold_max_slider.setMaximum(255)
        self.threshold_max_slider.setValue(200)
        self.threshold_max_slider.valueChanged.connect(self._update_options)
        threshold_layout.addWidget(self.threshold_max_slider)
        options_layout.addLayout(threshold_layout)
        
        # Nút chuyển đổi 2D/3D
        mode_layout = QHBoxLayout()
        self.mode_2d_radio = QRadioButton("2D")
        self.mode_2d_radio.setChecked(True)
        self.mode_2d_radio.toggled.connect(self._update_options)
        self.mode_3d_radio = QRadioButton("3D")
        mode_layout.addWidget(self.mode_2d_radio)
        mode_layout.addWidget(self.mode_3d_radio)
        options_layout.addLayout(mode_layout)
        
        # Nút áp dụng
        self.apply_btn = QPushButton("Áp dụng")
        self.apply_btn.clicked.connect(self._apply_tool)
        options_layout.addWidget(self.apply_btn)
        
        self.options_group.setLayout(options_layout)
        main_layout.addWidget(self.options_group)
        
        # Vùng trống
        main_layout.addStretch()
    
    def _select_tool(self, tool_name):
        """Chọn công cụ và cập nhật UI."""
        # Bỏ chọn tất cả các nút khác
        buttons = [
            self.brush_btn, self.smart_brush_btn, self.threshold_btn,
            self.eraser_btn, self.point_btn, self.grow_btn
        ]
        
        for btn in buttons:
            if btn.text() != tool_name:
                btn.setChecked(False)
        
        # Cập nhật công cụ hiện tại
        self.current_tool = tool_name
        self._update_options()
        
        # Gửi tín hiệu
        self.toolSelected.emit(tool_name, self.tool_options)
        
        logger.info(f"Đã chọn công cụ vẽ: {tool_name}")
    
    def _update_options(self):
        """Cập nhật các tùy chọn dựa trên công cụ hiện tại."""
        # Cập nhật nhãn
        self.brush_size_label.setText(str(self.brush_size_slider.value()))
        self.opacity_label.setText(f"{self.opacity_slider.value()}%")
        
        # Các tùy chọn chung
        self.tool_options = {
            'brush_size': self.brush_size_slider.value(),
            'opacity': self.opacity_slider.value() / 100.0,
            'mode': '2D' if self.mode_2d_radio.isChecked() else '3D'
        }
        
        # Các tùy chọn riêng cho từng công cụ
        if self.current_tool == "Threshold":
            self.tool_options.update({
                'min_threshold': self.threshold_min_slider.value(),
                'max_threshold': self.threshold_max_slider.value()
            })
        
        # Gửi lại tín hiệu với tùy chọn đã cập nhật
        if self.current_tool:
            self.toolSelected.emit(self.current_tool, self.tool_options)
    
    def _apply_tool(self):
        """Áp dụng công cụ hiện tại với các tùy chọn đã chọn."""
        if not self.current_tool:
            return
        
        logger.info(f"Đang áp dụng công cụ {self.current_tool} với các tùy chọn: {self.tool_options}")
        # Thông thường đây sẽ kích hoạt một hành động trong MPR viewer
        # hoặc trong slice view để áp dụng thay đổi vào structure
        
    def set_active_structure(self, structure):
        """Thiết lập cấu trúc đang hoạt động."""
        # Cập nhật UI để phản ánh các thuộc tính của cấu trúc hiện tại
        if structure:
            logger.info(f"Đã đặt cấu trúc hoạt động cho công cụ vẽ: {structure.name if hasattr(structure, 'name') else 'Unknown'}")
            
            # Có thể cập nhật màu sắc hoặc các tùy chọn khác dựa trên structure
    
    def get_current_tool(self):
        """Lấy công cụ hiện tại và các tùy chọn của nó."""
        return self.current_tool, self.tool_options.copy() if self.tool_options else {}

class MPRViewWidget(QWidget):
    """
    Widget hiển thị MPR (MultiPlanar Reconstruction).
    
    Widget này hiển thị ba mặt cắt: Axial, Sagittal, Coronal của dữ liệu hình ảnh 3D,
    và cho phép hiển thị/vẽ/chỉnh sửa các cấu trúc trên mỗi mặt cắt.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.current_structure_set = None
        self.current_tool = None
        self.current_tool_options = {}
        self.current_plane = "Axial"  # Mặt cắt mặc định
        
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện MPR view với ba mặt cắt."""
        main_layout = QVBoxLayout(self)
        
        # Toolbar cho các nút điều khiển
        toolbar = QToolBar()
        
        # Buttons cho các mặt cắt
        self.axial_btn = QToolButton()
        self.axial_btn.setText("Axial")
        self.axial_btn.setCheckable(True)
        self.axial_btn.setChecked(True)
        self.axial_btn.clicked.connect(lambda: self.change_plane("Axial"))
        
        self.sagittal_btn = QToolButton()
        self.sagittal_btn.setText("Sagittal")
        self.sagittal_btn.setCheckable(True)
        self.sagittal_btn.clicked.connect(lambda: self.change_plane("Sagittal"))
        
        self.coronal_btn = QToolButton()
        self.coronal_btn.setText("Coronal")
        self.coronal_btn.setCheckable(True)
        self.coronal_btn.clicked.connect(lambda: self.change_plane("Coronal"))
        
        # Nút chọn slice
        self.slice_label = QLabel("Slice:")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(99)  # Giá trị mặc định
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self.change_slice)
        
        # Nút Window/Level
        self.window_level_btn = QToolButton()
        self.window_level_btn.setText("W/L")
        self.window_level_btn.setCheckable(True)
        self.window_level_btn.clicked.connect(lambda: self.set_tool("WindowLevel"))
        
        # Nút Zoom
        self.zoom_btn = QToolButton()
        self.zoom_btn.setText("Zoom")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.clicked.connect(lambda: self.set_tool("Zoom"))
        
        # Nút đo lường
        self.measure_btn = QToolButton()
        self.measure_btn.setText("Measure")
        self.measure_btn.setCheckable(True)
        self.measure_btn.clicked.connect(lambda: self.set_tool("Measure"))
        
        # Thêm các nút vào toolbar
        toolbar.addWidget(self.axial_btn)
        toolbar.addWidget(self.sagittal_btn)
        toolbar.addWidget(self.coronal_btn)
        toolbar.addSeparator()
        toolbar.addWidget(self.slice_label)
        toolbar.addWidget(self.slice_slider)
        toolbar.addSeparator()
        toolbar.addWidget(self.window_level_btn)
        toolbar.addWidget(self.zoom_btn)
        toolbar.addWidget(self.measure_btn)
        
        # Import và tạo MPRViewer
        from quangtps.ui.mpr_viewer import MPRViewer
        self.mpr_viewer = MPRViewer()
        
        # Thêm vào layout
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.mpr_viewer)
        
        # Kết nối tín hiệu từ MPRViewer
        if hasattr(self.mpr_viewer, 'sliceChanged'):
            self.mpr_viewer.sliceChanged.connect(self.on_slice_changed)
    
    def set_image(self, image):
        """Thiết lập hình ảnh để hiển thị."""
        self.current_image = image
        if image:
            try:
                if hasattr(image, 'data') and image.data is not None:
                    self.mpr_viewer.set_image_data(image.data)
                elif hasattr(image, 'get_numpy_array'):
                    data = image.get_numpy_array()
                    self.mpr_viewer.set_image_data(data)
                elif hasattr(image, 'pixel_data'):
                    self.mpr_viewer.set_image_data(image.pixel_data)
                else:
                    logger.warning("Không thể trích xuất dữ liệu từ hình ảnh để hiển thị")
                
                # Cập nhật slider sau khi đặt hình ảnh
                if hasattr(self.mpr_viewer, 'get_max_slice'):
                    max_slice = self.mpr_viewer.get_max_slice()
                    self.slice_slider.setMaximum(max_slice)
                    self.slice_slider.setValue(max_slice // 2)
            
            except Exception as e:
                logger.error(f"Lỗi khi thiết lập hình ảnh cho MPR viewer: {e}")
    
    def set_structure_set(self, structure_set):
        """Thiết lập bộ cấu trúc để hiển thị."""
        self.current_structure_set = structure_set
        if structure_set and hasattr(self.mpr_viewer, 'set_structure_set'):
            try:
                self.mpr_viewer.set_structure_set(structure_set)
            except Exception as e:
                logger.error(f"Lỗi khi thiết lập cấu trúc cho MPR viewer: {e}")
    
    def set_tool(self, tool_name, options=None):
        """Thiết lập công cụ hiện tại (vẽ, đo, v.v.)."""
        self.current_tool = tool_name
        self.current_tool_options = options or {}
        
        # Cập nhật trạng thái nút
        self.window_level_btn.setChecked(tool_name == "WindowLevel")
        self.zoom_btn.setChecked(tool_name == "Zoom")
        self.measure_btn.setChecked(tool_name == "Measure")
        
        # Thông báo cho MPRViewer
        if hasattr(self.mpr_viewer, 'set_tool'):
            try:
                self.mpr_viewer.set_tool(tool_name, self.current_tool_options)
            except Exception as e:
                logger.error(f"Lỗi khi đặt công cụ cho MPR viewer: {e}")
    
    def change_plane(self, plane):
        """Thay đổi mặt cắt hiển thị (Axial, Sagittal, Coronal)."""
        self.current_plane = plane
        
        # Cập nhật trạng thái nút
        self.axial_btn.setChecked(plane == "Axial")
        self.sagittal_btn.setChecked(plane == "Sagittal")
        self.coronal_btn.setChecked(plane == "Coronal")
        
        # Thông báo cho MPRViewer
        if hasattr(self.mpr_viewer, 'set_active_plane'):
            try:
                self.mpr_viewer.set_active_plane(plane)
                
                # Cập nhật slider
                if hasattr(self.mpr_viewer, 'get_max_slice'):
                    max_slice = self.mpr_viewer.get_max_slice(plane)
                    current_slice = self.mpr_viewer.get_current_slice(plane)
                    self.slice_slider.setMaximum(max_slice)
                    self.slice_slider.setValue(current_slice)
            except Exception as e:
                logger.error(f"Lỗi khi thay đổi mặt cắt trong MPR viewer: {e}")
    
    def change_slice(self, slice_index):
        """Thay đổi slice hiển thị."""
        if hasattr(self.mpr_viewer, 'set_slice'):
            try:
                self.mpr_viewer.set_slice(slice_index, self.current_plane)
            except Exception as e:
                logger.error(f"Lỗi khi thay đổi slice trong MPR viewer: {e}")
    
    def on_slice_changed(self, plane, slice_index):
        """Xử lý khi slice thay đổi trong MPRViewer."""
        if plane == self.current_plane:
            # Cập nhật slider mà không gây ra vòng lặp
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(slice_index)
            self.slice_slider.blockSignals(False)

class StructureTab(QWidget):
    """
    Tab Structure dùng để vẽ và quản lý structure/contour.
    
    Tab này cung cấp các công cụ và giao diện tương tự như Eclipse Structure tab,
    cho phép người dùng vẽ, chỉnh sửa, và quản lý các structure.
    """
    
    structureChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_patient = None
        self.current_image = None
        self.current_structure_set = None
        
        # Khởi tạo dịch vụ
        try:
            self.service_registry = ServiceRegistry()
            self.structure_db = self.service_registry.get_service("StructureDB")
            self.patient_db = self.service_registry.get_service("PatientDB")
            self.image_db = self.service_registry.get_service("ImageDB")
        except Exception as e:
            logger.error(f"Không thể khởi tạo dịch vụ: {e}")
            self.structure_db = None
            self.patient_db = None
            self.image_db = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện Structure tab."""
        main_layout = QHBoxLayout(self)
        
        # Splitter chính chia khu vực danh sách cấu trúc và khu vực vẽ
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Panel bên trái chứa danh sách cấu trúc và công cụ vẽ
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Danh sách cấu trúc
        self.structure_list_widget = StructureListWidget()
        self.structure_list_widget.structureSelected.connect(self.on_structure_selected)
        self.structure_list_widget.structureChanged.connect(self.on_structure_changed)
        left_layout.addWidget(self.structure_list_widget, 2)
        
        # Công cụ vẽ
        self.drawing_tools_widget = DrawingToolsWidget()
        self.drawing_tools_widget.toolSelected.connect(self.on_drawing_tool_selected)
        left_layout.addWidget(self.drawing_tools_widget, 1)
        
        # Panel bên phải chứa MPR viewer với các mặt cắt
        self.mpr_view_widget = MPRViewWidget()
        
        # Thêm các panel vào splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(self.mpr_view_widget)
        
        # Thiết lập kích thước khởi tạo cho các panel (30% bên trái, 70% bên phải)
        main_splitter.setSizes([300, 700])
        
        main_layout.addWidget(main_splitter)
        
        # Tạo dữ liệu mẫu cho kiểm thử
        self._create_sample_data()
    
    def _create_sample_data(self):
        """Tạo dữ liệu mẫu để hiển thị khi không có dữ liệu thực."""
        try:
            # Tạo dữ liệu hình ảnh mẫu cho MPR viewer
            from quangtps.ui.mpr_viewer import create_sample_data
            sample_data = create_sample_data(100)
            self.mpr_view_widget.mpr_viewer.set_image_data(sample_data)
            
            # Tạo cấu trúc mẫu
            if hasattr(self, 'structure_db') and self.structure_db:
                # Dữ liệu thực từ cơ sở dữ liệu
                pass
            else:
                # Dữ liệu mẫu nếu không có cơ sở dữ liệu
                try:
                    from quangtps.segmentation.structures.structure import Structure
                    from quangtps.segmentation.structures.structure_set import StructureSet
                    
                    # Tạo một bộ cấu trúc mẫu với các cấu trúc cơ bản
                    structures = []
                    
                    ptv = Structure(name="PTV")
                    ptv.id = "STRUCT_001"
                    ptv.type = "PTV"
                    ptv.color = (255, 0, 0)
                    ptv.visible = True
                    structures.append(ptv)
                    
                    ctv = Structure(name="CTV")
                    ctv.id = "STRUCT_002"
                    ctv.type = "CTV"
                    ctv.color = (255, 165, 0)
                    ctv.visible = True
                    structures.append(ctv)
                    
                    heart = Structure(name="Heart")
                    heart.id = "STRUCT_003"
                    heart.type = "ORGAN"
                    heart.color = (0, 0, 255)
                    heart.visible = True
                    structures.append(heart)
                    
                    lung_right = Structure(name="Lung_Right")
                    lung_right.id = "STRUCT_004"
                    lung_right.type = "ORGAN"
                    lung_right.color = (0, 255, 0)
                    lung_right.visible = True
                    structures.append(lung_right)
                    
                    lung_left = Structure(name="Lung_Left")
                    lung_left.id = "STRUCT_005"
                    lung_left.type = "ORGAN"
                    lung_left.color = (0, 255, 128)
                    lung_left.visible = True
                    structures.append(lung_left)
                    
                    spinal_cord = Structure(name="SpinalCord")
                    spinal_cord.id = "STRUCT_006"
                    spinal_cord.type = "ORGAN"
                    spinal_cord.color = (255, 255, 0)
                    spinal_cord.visible = True
                    structures.append(spinal_cord)
                    
                    body = Structure(name="Body")
                    body.id = "STRUCT_007"
                    body.type = "EXTERNAL"
                    body.color = (192, 192, 192)
                    body.visible = True
                    structures.append(body)
                    
                    # Tạo structure set
                    structure_set = StructureSet()
                    structure_set.id = "STRUCTSET_SAMPLE"
                    structure_set.structures = structures
                    
                    # Thiết lập vào list widget
                    self.structure_list_widget.set_structure_set(structure_set)
                    self.current_structure_set = structure_set
                    
                except Exception as e:
                    logger.error(f"Không thể tạo dữ liệu mẫu cho cấu trúc: {e}")
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo dữ liệu mẫu: {e}")
    
    def set_patient(self, patient):
        """Thiết lập bệnh nhân hiện tại và tải dữ liệu liên quan."""
        if patient and (not self.current_patient or patient.id != self.current_patient.id):
            self.current_patient = patient
            logger.info(f"Đã tải bệnh nhân trong Structure tab: {patient.id}")
            
            # Tải hình ảnh và cấu trúc liên quan
            try:
                if self.image_db:
                    images = self.image_db.get_images_by_patient_id(patient.id)
                    if images:
                        # Chọn hình ảnh mới nhất hoặc hình ảnh mặc định
                        self.set_image(images[0])
                
                if self.structure_db:
                    structure_sets = self.structure_db.get_structure_sets_by_patient_id(patient.id)
                    if structure_sets:
                        # Chọn bộ cấu trúc mới nhất hoặc bộ mặc định
                        self.set_structure_set(structure_sets[0])
            except Exception as e:
                logger.error(f"Lỗi khi tải dữ liệu bệnh nhân trong Structure tab: {e}")
    
    def set_image(self, image):
        """Thiết lập hình ảnh hiện tại để hiển thị."""
        self.current_image = image
        if image:
            logger.info(f"Đã tải hình ảnh trong Structure tab: {image.id if hasattr(image, 'id') else 'Unknown'}")
            self.mpr_view_widget.set_image(image)
    
    def set_structure_set(self, structure_set):
        """Thiết lập bộ cấu trúc hiện tại."""
        self.current_structure_set = structure_set
        if structure_set:
            logger.info(f"Đã tải bộ cấu trúc: {structure_set.id if hasattr(structure_set, 'id') else 'Unknown'}")
            self.structure_list_widget.set_structure_set(structure_set)
            self.mpr_view_widget.set_structure_set(structure_set)
    
    def on_structure_selected(self, structure):
        """Xử lý khi người dùng chọn một cấu trúc từ danh sách."""
        logger.info(f"Đã chọn cấu trúc: {structure.name if hasattr(structure, 'name') else 'Unknown'}")
        # Thêm xử lý khi cần thiết
    
    def on_structure_changed(self):
        """Xử lý khi có thay đổi trong bộ cấu trúc."""
        logger.info("Cấu trúc đã thay đổi")
        # Cập nhật hiển thị MPR viewer
        if self.current_structure_set:
            self.mpr_view_widget.set_structure_set(self.current_structure_set)
        
        # Gửi tín hiệu thay đổi
        self.structureChanged.emit()
    
    def on_drawing_tool_selected(self, tool_name, options):
        """Xử lý khi người dùng chọn công cụ vẽ."""
        logger.info(f"Đã chọn công cụ vẽ: {tool_name}")
        self.mpr_view_widget.set_tool(tool_name, options) 