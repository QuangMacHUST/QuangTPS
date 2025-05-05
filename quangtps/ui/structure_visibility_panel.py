#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module StructureVisibilityPanel cho QuangTPS.

Panel cho phép người dùng điều chỉnh hiển thị (visibility) của các cấu trúc trong giao diện 3D.
Cho phép điều chỉnh độ trong suốt, hiển thị/ẩn và tùy chỉnh màu sắc cấu trúc.
"""

import os
import logging
import random  # Thêm import random cho phần test
from typing import Dict, List, Tuple, Optional, Any, Set

# Import PyQt5 components with try/except
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QColorDialog,
        QCheckBox,
        QComboBox,
        QGroupBox,
        QFrame,
        QSplitter,
        QTreeWidget,
        QTreeWidgetItem,
        QMenu,
        QAction,
        QInputDialog,
        QMessageBox,
        QSizePolicy,
        QApplication,  # Thêm QApplication cho test standalone
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtGui import QColor, QBrush, QIcon, QFont

    # Đánh dấu rằng PyQt5 đã được import thành công
    PYQT5_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import PyQt5: {e}")
    PYQT5_AVAILABLE = False

    # Tạo các lớp giả cho type checking
    class QWidget:
        pass

    class pyqtSignal:
        def __init__(self, *args):
            pass


# Import QuangTPS modules
try:
    from quangtps.structures.structure import Structure
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure_utils.colors import get_default_color_for_type

    STRUCTURES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import các module structure: {e}")
    STRUCTURES_AVAILABLE = False

    # Hàm thay thế nếu không có module
    def get_default_color_for_type(structure_type):
        """Hàm thay thế tạo màu mặc định theo loại cấu trúc."""
        if "PTV" in structure_type:
            return (1.0, 0.0, 0.0)  # Đỏ cho PTV
        elif "GTV" in structure_type:
            return (1.0, 0.5, 0.0)  # Cam cho GTV
        elif "CTV" in structure_type:
            return (0.0, 1.0, 0.0)  # Xanh lá cho CTV
        elif "OAR" in structure_type or "Organ" in structure_type:
            return (0.0, 0.0, 1.0)  # Xanh dương cho OAR
        else:
            # Random color cho các loại khác
            return (random.random(), random.random(), random.random())


from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class StructureVisibilityPanel(QWidget):
    """
    Panel cho phép người dùng quản lý hiển thị các cấu trúc trong giao diện 3D.

    Cung cấp giao diện để điều chỉnh độ trong suốt, hiển thị/ẩn và tùy chỉnh màu sắc
    của các cấu trúc trong StructureSet.
    """

    # Tín hiệu phát ra khi có thay đổi về hiển thị cấu trúc
    visibilityChanged = pyqtSignal(
        dict
    )  # Phát ra dict {structure_id: (visible, opacity)}
    colorChanged = pyqtSignal(str, tuple)  # Structure ID, new color (r,g,b)

    def __init__(self, parent=None):
        """Khởi tạo StructureVisibilityPanel widget."""
        super(StructureVisibilityPanel, self).__init__(parent)

        # Lưu trữ dữ liệu
        self.structures = {}  # Dict mapping structure_id to Structure object
        self.visibility_data = {}  # Dict mapping structure_id to (visible, opacity)
        self.color_data = {}  # Dict mapping structure_id to color (r,g,b)
        self.structure_set = None

        # Thiết lập giao diện
        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)

        # Label tiêu đề
        title_label = QLabel("Hiển thị Cấu trúc")
        title_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(title_label)

        # Khung điều khiển chung
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(0, 0, 0, 0)

        # Nút hiển thị/ẩn tất cả
        self.show_all_btn = QPushButton("Hiện tất cả")
        self.show_all_btn.setToolTip("Hiển thị tất cả các cấu trúc")
        self.show_all_btn.clicked.connect(self.show_all_structures)
        control_layout.addWidget(self.show_all_btn)

        self.hide_all_btn = QPushButton("Ẩn tất cả")
        self.hide_all_btn.setToolTip("Ẩn tất cả các cấu trúc")
        self.hide_all_btn.clicked.connect(self.hide_all_structures)
        control_layout.addWidget(self.hide_all_btn)

        main_layout.addWidget(control_frame)

        # Combobox cho các preset structure group
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Nhóm:"))

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            ["Tất cả", "PTV", "OAR", "ITV", "GTV", "CTV", "Khác"]
        )
        self.preset_combo.currentTextChanged.connect(self.filter_by_group)
        preset_layout.addWidget(self.preset_combo)

        main_layout.addLayout(preset_layout)

        # Bảng cấu trúc
        self.structure_table = QTableWidget()
        self.structure_table.setColumnCount(3)
        self.structure_table.setHorizontalHeaderLabels(
            ["Cấu trúc", "Hiển thị", "Độ trong suốt"]
        )
        self.structure_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.structure_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Fixed
        )
        self.structure_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Fixed
        )
        self.structure_table.setColumnWidth(1, 50)
        self.structure_table.setColumnWidth(2, 100)
        self.structure_table.verticalHeader().setVisible(False)
        self.structure_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.structure_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.structure_table.customContextMenuRequested.connect(self._show_context_menu)

        main_layout.addWidget(self.structure_table)

        # Slider cho global opacity
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Độ trong suốt chung:"))

        self.global_opacity_slider = QSlider(Qt.Horizontal)
        self.global_opacity_slider.setRange(0, 100)
        self.global_opacity_slider.setValue(70)  # 70% opacity by default
        self.global_opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.global_opacity_slider.setTickInterval(10)
        self.global_opacity_slider.valueChanged.connect(self.update_global_opacity)
        opacity_layout.addWidget(self.global_opacity_slider)

        opacity_label = QLabel("70%")
        self.global_opacity_slider.valueChanged.connect(
            lambda value: opacity_label.setText(f"{value}%")
        )
        opacity_layout.addWidget(opacity_label)

        main_layout.addLayout(opacity_layout)

    def _show_context_menu(self, position):
        """Hiển thị menu ngữ cảnh khi nhấp chuột phải vào bảng."""
        selected_rows = self.structure_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        menu = QMenu(self)

        # Các action trong menu
        change_color_action = QAction("Thay đổi màu", self)
        change_color_action.triggered.connect(self._change_selected_color)
        menu.addAction(change_color_action)

        show_action = QAction("Hiển thị", self)
        show_action.triggered.connect(lambda: self._set_selected_visibility(True))
        menu.addAction(show_action)

        hide_action = QAction("Ẩn", self)
        hide_action.triggered.connect(lambda: self._set_selected_visibility(False))
        menu.addAction(hide_action)

        # Hiển thị menu tại vị trí chuột
        menu.exec_(self.structure_table.mapToGlobal(position))

    def _change_selected_color(self):
        """Thay đổi màu cho các cấu trúc đã chọn."""
        selected_rows = self.structure_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        # Chỉ thay đổi màu của cấu trúc đầu tiên được chọn
        row = selected_rows[0].row()
        structure_item = self.structure_table.item(row, 0)

        if not structure_item:
            return

        structure_id = structure_item.data(Qt.UserRole)
        current_color = self.color_data.get(structure_id)

        if not current_color:
            return

        # Hiển thị hộp thoại chọn màu
        current_qcolor = QColor(
            int(current_color[0] * 255),
            int(current_color[1] * 255),
            int(current_color[2] * 255),
        )

        color_dialog = QColorDialog(self)
        color_dialog.setCurrentColor(current_qcolor)

        if color_dialog.exec_():
            qcolor = color_dialog.selectedColor()
            new_color = (
                qcolor.red() / 255.0,
                qcolor.green() / 255.0,
                qcolor.blue() / 255.0,
            )

            # Cập nhật màu
            self.color_data[structure_id] = new_color

            # Cập nhật hiển thị trong bảng
            structure_item.setForeground(QBrush(qcolor))

            # Phát tín hiệu thay đổi
            self.colorChanged.emit(structure_id, new_color)

    def _set_selected_visibility(self, visible):
        """
        Đặt trạng thái hiển thị cho các cấu trúc đã chọn.

        Parameters:
        -----------
        visible : bool
            True để hiển thị, False để ẩn
        """
        selected_rows = self.structure_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        # Cập nhật trạng thái hiển thị cho từng cấu trúc đã chọn
        changes = {}

        for model_index in selected_rows:
            row = model_index.row()
            structure_item = self.structure_table.item(row, 0)

            if not structure_item:
                continue

            structure_id = structure_item.data(Qt.UserRole)

            if structure_id in self.visibility_data:
                _, opacity = self.visibility_data[structure_id]
                self.visibility_data[structure_id] = (visible, opacity)

                # Cập nhật checkbox
                checkbox = self.structure_table.cellWidget(row, 1)
                if checkbox:
                    checkbox.setChecked(visible)

                changes[structure_id] = (visible, opacity)

        # Phát tín hiệu nếu có thay đổi
        if changes:
            self.visibilityChanged.emit(changes)

    def set_structure_set(self, structure_set):
        """
        Đặt structure set và cập nhật hiển thị.

        Parameters:
        -----------
        structure_set : StructureSet
            Structure set cần hiển thị
        """
        self.structure_set = structure_set
        self.structures = {}
        self.visibility_data = {}
        self.color_data = {}

        # Nếu structure_set là None, xóa bảng và thoát
        if not structure_set:
            self.structure_table.setRowCount(0)
            return

        # Cập nhật danh sách cấu trúc
        self._update_structure_list()

    def _update_structure_list(self):
        """Cập nhật danh sách cấu trúc từ structure set."""
        # Xóa bảng hiện tại
        self.structure_table.setRowCount(0)

        if not self.structure_set:
            return

        # Lấy danh sách cấu trúc từ structure set
        structures = getattr(self.structure_set, "structures", {})

        # Tùy thuộc vào preset đã chọn, có thể lọc danh sách cấu trúc
        group_filter = self.preset_combo.currentText()

        # Thêm từng cấu trúc vào bảng
        for structure_id, structure in structures.items():
            # Áp dụng bộ lọc nếu cần
            if group_filter != "Tất cả":
                structure_type = getattr(structure, "type", "")

                if group_filter == "PTV" and not structure_type.startswith("PTV"):
                    continue
                elif group_filter == "OAR" and not (
                    structure_type.startswith("Organ")
                    or structure_type.startswith("OAR")
                ):
                    continue
                elif group_filter == "ITV" and not structure_type.startswith("ITV"):
                    continue
                elif group_filter == "GTV" and not structure_type.startswith("GTV"):
                    continue
                elif group_filter == "CTV" and not structure_type.startswith("CTV"):
                    continue
                elif group_filter == "Khác" and any(
                    structure_type.startswith(prefix)
                    for prefix in ["PTV", "OAR", "Organ", "ITV", "GTV", "CTV"]
                ):
                    continue

            # Lưu cấu trúc
            self.structures[structure_id] = structure

            # Thiết lập giá trị mặc định nếu chưa tồn tại
            if structure_id not in self.visibility_data:
                self.visibility_data[structure_id] = (True, 0.7)  # (visible, opacity)

            if structure_id not in self.color_data:
                # Lấy màu từ cấu trúc hoặc tạo màu mặc định
                default_color = get_default_color_for_type(
                    getattr(structure, "type", "Unknown")
                )
                structure_color = getattr(structure, "color", default_color)
                self.color_data[structure_id] = structure_color

            # Thêm vào bảng
            self._add_structure_to_table(structure_id, structure)

    def _add_structure_to_table(self, structure_id, structure):
        """
        Thêm một cấu trúc vào bảng.

        Parameters:
        -----------
        structure_id : str
            ID của cấu trúc
        structure : Structure
            Đối tượng cấu trúc
        """
        row = self.structure_table.rowCount()
        self.structure_table.insertRow(row)

        # Tên cấu trúc
        name = getattr(structure, "name", f"Structure {structure_id}")
        structure_item = QTableWidgetItem(name)
        structure_item.setData(Qt.UserRole, structure_id)

        # Đặt màu text theo màu cấu trúc
        color = self.color_data[structure_id]
        qcolor = QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        structure_item.setForeground(QBrush(qcolor))

        self.structure_table.setItem(row, 0, structure_item)

        # Checkbox hiển thị
        visible, _ = self.visibility_data[structure_id]
        checkbox = QCheckBox()
        checkbox.setChecked(visible)
        checkbox.stateChanged.connect(
            lambda state, sid=structure_id, r=row: self._toggle_visibility(
                sid, r, state
            )
        )

        # Đặt checkbox vào ô
        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_container)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.setAlignment(Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.structure_table.setCellWidget(row, 1, checkbox_container)

        # Slider độ trong suốt
        _, opacity = self.visibility_data[structure_id]
        opacity_percent = int(opacity * 100)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(opacity_percent)
        slider.valueChanged.connect(
            lambda value, sid=structure_id: self._update_opacity(sid, value / 100.0)
        )

        # Đặt slider vào ô
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.addWidget(slider)
        slider_layout.setContentsMargins(4, 0, 4, 0)
        self.structure_table.setCellWidget(row, 2, slider_container)

    def _toggle_visibility(self, structure_id, row, state):
        """
        Bật/tắt hiển thị một cấu trúc.

        Parameters:
        -----------
        structure_id : str
            ID của cấu trúc
        row : int
            Chỉ số hàng trong bảng
        state : int
            Trạng thái của checkbox (Qt.Checked hoặc Qt.Unchecked)
        """
        if structure_id not in self.visibility_data:
            return

        visible = state == Qt.Checked
        _, opacity = self.visibility_data[structure_id]
        self.visibility_data[structure_id] = (visible, opacity)

        # Phát tín hiệu thay đổi
        self.visibilityChanged.emit({structure_id: (visible, opacity)})

    def _update_opacity(self, structure_id, opacity):
        """
        Cập nhật độ trong suốt của một cấu trúc.

        Parameters:
        -----------
        structure_id : str
            ID của cấu trúc
        opacity : float
            Giá trị độ trong suốt (0.0 - 1.0)
        """
        if structure_id not in self.visibility_data:
            return

        visible, _ = self.visibility_data[structure_id]
        self.visibility_data[structure_id] = (visible, opacity)

        # Phát tín hiệu thay đổi
        if visible:  # Chỉ cập nhật nếu đang hiển thị
            self.visibilityChanged.emit({structure_id: (visible, opacity)})

    def update_global_opacity(self, value):
        """
        Cập nhật độ trong suốt cho tất cả các cấu trúc.

        Parameters:
        -----------
        value : int
            Phần trăm độ trong suốt (0-100)
        """
        opacity = value / 100.0
        changes = {}

        # Cập nhật từng slider trong bảng
        for row in range(self.structure_table.rowCount()):
            structure_item = self.structure_table.item(row, 0)

            if not structure_item:
                continue

            structure_id = structure_item.data(Qt.UserRole)

            if structure_id in self.visibility_data:
                visible, _ = self.visibility_data[structure_id]
                self.visibility_data[structure_id] = (visible, opacity)

                # Cập nhật slider
                slider_container = self.structure_table.cellWidget(row, 2)
                if slider_container:
                    slider = slider_container.findChild(QSlider)
                    if slider:
                        slider.blockSignals(True)
                        slider.setValue(int(opacity * 100))
                        slider.blockSignals(False)

                if visible:
                    changes[structure_id] = (visible, opacity)

        # Phát tín hiệu nếu có thay đổi
        if changes:
            self.visibilityChanged.emit(changes)

    def show_all_structures(self):
        """Hiển thị tất cả các cấu trúc."""
        changes = {}

        # Cập nhật trạng thái hiển thị
        for row in range(self.structure_table.rowCount()):
            structure_item = self.structure_table.item(row, 0)

            if not structure_item:
                continue

            structure_id = structure_item.data(Qt.UserRole)

            if structure_id in self.visibility_data:
                _, opacity = self.visibility_data[structure_id]
                self.visibility_data[structure_id] = (True, opacity)

                # Cập nhật checkbox
                checkbox_container = self.structure_table.cellWidget(row, 1)
                if checkbox_container:
                    checkbox = checkbox_container.findChild(QCheckBox)
                    if checkbox:
                        checkbox.blockSignals(True)
                        checkbox.setChecked(True)
                        checkbox.blockSignals(False)

                changes[structure_id] = (True, opacity)

        # Phát tín hiệu nếu có thay đổi
        if changes:
            self.visibilityChanged.emit(changes)

    def hide_all_structures(self):
        """Ẩn tất cả các cấu trúc."""
        changes = {}

        # Cập nhật trạng thái hiển thị
        for row in range(self.structure_table.rowCount()):
            structure_item = self.structure_table.item(row, 0)

            if not structure_item:
                continue

            structure_id = structure_item.data(Qt.UserRole)

            if structure_id in self.visibility_data:
                _, opacity = self.visibility_data[structure_id]
                self.visibility_data[structure_id] = (False, opacity)

                # Cập nhật checkbox
                checkbox_container = self.structure_table.cellWidget(row, 1)
                if checkbox_container:
                    checkbox = checkbox_container.findChild(QCheckBox)
                    if checkbox:
                        checkbox.blockSignals(True)
                        checkbox.setChecked(False)
                        checkbox.blockSignals(False)

                changes[structure_id] = (False, opacity)

        # Phát tín hiệu nếu có thay đổi
        if changes:
            self.visibilityChanged.emit(changes)

    def filter_by_group(self, group_name):
        """
        Lọc danh sách cấu trúc theo nhóm.

        Parameters:
        -----------
        group_name : str
            Tên nhóm cần lọc
        """
        # Cập nhật lại danh sách cấu trúc theo bộ lọc mới
        self._update_structure_list()

    def refresh(self):
        """Làm mới hiển thị."""
        if self.structure_set:
            self._update_structure_list()

    def get_visibility_data(self):
        """
        Lấy dữ liệu hiển thị hiện tại.

        Returns:
        --------
        dict
            Dictionary mapping structure_id to (visible, opacity)
        """
        return self.visibility_data.copy()

    def get_color_data(self):
        """
        Lấy dữ liệu màu sắc hiện tại.

        Returns:
        --------
        dict
            Dictionary mapping structure_id to color (r,g,b)
        """
        return self.color_data.copy()


# Test standalone
if __name__ == "__main__":
    import sys

    # Kiểm tra xem PyQt5 có sẵn không
    if not PYQT5_AVAILABLE:
        print("PyQt5 không khả dụng, không thể chạy test standalone")
        sys.exit(1)

    app = QApplication(sys.argv)

    # Tạo Structure và StructureSet giả để test
    class MockStructure:
        def __init__(self, name, structure_type, color=None):
            self.id = name.lower().replace(" ", "_")
            self.name = name
            self.type = structure_type
            self.color = color or (random.random(), random.random(), random.random())

    class MockStructureSet:
        def __init__(self):
            self.structures = {
                "ptv_1": MockStructure("PTV 54Gy", "PTV", (1.0, 0.0, 0.0)),  # Đỏ
                "ptv_2": MockStructure("PTV 60Gy", "PTV", (1.0, 0.5, 0.0)),  # Cam
                "spinal_cord": MockStructure(
                    "Spinal Cord", "OAR", (0.0, 1.0, 0.0)
                ),  # Xanh lá
                "lungs": MockStructure("Lungs", "OAR", (0.0, 0.0, 1.0)),  # Xanh dương
                "heart": MockStructure("Heart", "OAR", (1.0, 0.0, 1.0)),  # Tím
                "gtv": MockStructure("GTV", "GTV", (1.0, 1.0, 0.0)),  # Vàng
            }

    # Tạo widget và set dữ liệu
    widget = StructureVisibilityPanel()
    widget.setWindowTitle("Structure Visibility Panel")
    widget.resize(400, 500)

    # Kết nối tín hiệu để debug
    widget.visibilityChanged.connect(
        lambda changes: print(f"Visibility changed: {changes}")
    )
    widget.colorChanged.connect(
        lambda sid, color: print(f"Color changed for {sid}: {color}")
    )

    # Thiết lập dữ liệu giả
    widget.set_structure_set(MockStructureSet())

    widget.show()

    sys.exit(app.exec_())
