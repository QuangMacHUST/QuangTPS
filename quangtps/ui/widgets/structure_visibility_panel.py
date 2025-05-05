#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Widget cho phép quản lý hiển thị các cấu trúc trong hiển thị liều 3D.
Người dùng có thể bật/tắt hiển thị từng cấu trúc, thay đổi màu sắc và
thêm các tính năng tương tác khác.
"""

import logging
import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QColorDialog,
    QLabel,
    QHeaderView,
    QDoubleSpinBox,
    QAbstractItemView,
    QMessageBox,
    QMenu,
    QAction,
    QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QPixmap, QIcon

try:
    from quangtps.ui.utils.color_map import ColorMap
except ImportError:
    logging.warning("Không thể import color_map, sử dụng lớp ColorMap mặc định")

    class ColorMap:
        """Lớp giả để đáp ứng các tính năng cơ bản của ColorMap"""

        @staticmethod
        def rgb_to_hex(rgb):
            """Chuyển đổi RGB (0-1) sang mã hex"""
            r, g, b = [int(x * 255) for x in rgb]
            return f"#{r:02x}{g:02x}{b:02x}"

        @staticmethod
        def hex_to_rgb(hex_color):
            """Chuyển đổi mã hex sang RGB (0-1)"""
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


class StructureVisibilityPanel(QWidget):
    """
    Widget cung cấp bảng điều khiển hiển thị các cấu trúc.
    Cho phép bật/tắt cấu trúc và thay đổi màu sắc.

    Signals:
        visibility_changed: Phát ra khi thay đổi hiển thị của một cấu trúc.
            (str, bool): ID của cấu trúc và trạng thái hiển thị mới.
        color_changed: Phát ra khi thay đổi màu sắc của một cấu trúc.
            (str, tuple): ID của cấu trúc và màu RGB mới (0-1).
    """

    visibility_changed = pyqtSignal(str, bool)
    color_changed = pyqtSignal(str, tuple)

    def __init__(self, parent=None):
        """
        Khởi tạo StructureVisibilityPanel.

        Parameters:
        -----------
        parent : QWidget, optional
            Widget cha, mặc định là None.
        """
        super(StructureVisibilityPanel, self).__init__(parent)

        # Dữ liệu
        self._structures = {}  # {id: {'name': str, 'color': tuple, 'visible': bool}}

        # Thiết lập giao diện
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Bảng cấu trúc
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Hiển thị", "Cấu trúc", "Màu"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.horizontalHeader().resizeSection(0, 50)
        self.table.horizontalHeader().resizeSection(2, 50)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.table)

        # Control buttons
        button_layout = QHBoxLayout()

        self.show_all_button = QPushButton("Hiện tất cả")
        self.show_all_button.clicked.connect(self._show_all)
        button_layout.addWidget(self.show_all_button)

        self.hide_all_button = QPushButton("Ẩn tất cả")
        self.hide_all_button.clicked.connect(self._hide_all)
        button_layout.addWidget(self.hide_all_button)

        layout.addLayout(button_layout)

    def add_structure(self, struct_id, name, color, visible=True):
        """
        Thêm một cấu trúc vào bảng.

        Parameters:
        -----------
        struct_id : str
            ID của cấu trúc.
        name : str
            Tên hiển thị của cấu trúc.
        color : tuple
            Màu RGB (0-1) của cấu trúc.
        visible : bool, optional
            Trạng thái hiển thị, mặc định là True.
        """
        # Kiểm tra xem cấu trúc đã tồn tại chưa
        if struct_id in self._structures:
            return

        # Thêm vào dictionary
        self._structures[struct_id] = {"name": name, "color": color, "visible": visible}

        # Thêm vào bảng
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Checkbox hiển thị
        checkbox = QCheckBox()
        checkbox.setChecked(visible)
        checkbox.stateChanged.connect(
            lambda state, s_id=struct_id: self._toggle_visibility(s_id, state)
        )
        self.table.setCellWidget(row, 0, checkbox)

        # Tên cấu trúc
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.UserRole, struct_id)  # Lưu ID
        self.table.setItem(row, 1, name_item)

        # Màu sắc
        color_item = QTableWidgetItem()
        rgb = [int(c * 255) for c in color]
        color_qcolor = QColor(*rgb)
        color_item.setBackground(QBrush(color_qcolor))
        self.table.setItem(row, 2, color_item)

    def _toggle_visibility(self, struct_id, state):
        """
        Thay đổi trạng thái hiển thị của cấu trúc.

        Parameters:
        -----------
        struct_id : str
            ID của cấu trúc.
        state : int
            Trạng thái mới (Qt.Checked hoặc Qt.Unchecked).
        """
        if struct_id not in self._structures:
            return

        visible = state == Qt.Checked
        self._structures[struct_id]["visible"] = visible

        # Phát tín hiệu
        self.visibility_changed.emit(struct_id, visible)

    def _show_context_menu(self, pos):
        """
        Hiển thị menu ngữ cảnh khi nhấp chuột phải vào bảng.

        Parameters:
        -----------
        pos : QPoint
            Vị trí con trỏ chuột.
        """
        row = self.table.rowAt(pos.y())
        if row < 0:
            return

        name_item = self.table.item(row, 1)
        struct_id = name_item.data(Qt.UserRole)

        menu = QMenu(self)

        # Đổi màu
        change_color_action = QAction("Đổi màu", self)
        change_color_action.triggered.connect(lambda: self._change_color(struct_id))
        menu.addAction(change_color_action)

        # Chỉ hiển thị cấu trúc này
        solo_action = QAction("Chỉ hiển thị cấu trúc này", self)
        solo_action.triggered.connect(lambda: self._solo_structure(struct_id))
        menu.addAction(solo_action)

        menu.exec_(self.table.mapToGlobal(pos))

    def _change_color(self, struct_id):
        """
        Thay đổi màu sắc của cấu trúc.

        Parameters:
        -----------
        struct_id : str
            ID của cấu trúc.
        """
        if struct_id not in self._structures:
            return

        struct = self._structures[struct_id]
        current_color = struct["color"]
        current_qcolor = QColor(
            int(current_color[0] * 255),
            int(current_color[1] * 255),
            int(current_color[2] * 255),
        )

        color_dialog = QColorDialog(current_qcolor, self)
        if color_dialog.exec_():
            new_qcolor = color_dialog.selectedColor()
            new_color = (
                new_qcolor.red() / 255.0,
                new_qcolor.green() / 255.0,
                new_qcolor.blue() / 255.0,
            )

            # Cập nhật màu mới
            self._structures[struct_id]["color"] = new_color

            # Cập nhật UI
            for row in range(self.table.rowCount()):
                name_item = self.table.item(row, 1)
                if name_item and name_item.data(Qt.UserRole) == struct_id:
                    color_item = self.table.item(row, 2)
                    color_item.setBackground(QBrush(new_qcolor))
                    break

            # Phát tín hiệu
            self.color_changed.emit(struct_id, new_color)

    def _solo_structure(self, struct_id):
        """
        Chỉ hiển thị một cấu trúc, ẩn tất cả các cấu trúc khác.

        Parameters:
        -----------
        struct_id : str
            ID của cấu trúc cần hiển thị.
        """
        if struct_id not in self._structures:
            return

        # Ẩn tất cả cấu trúc khác
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 1)
            if name_item:
                current_id = name_item.data(Qt.UserRole)
                checkbox = self.table.cellWidget(row, 0)

                if current_id == struct_id:
                    # Hiển thị cấu trúc này
                    checkbox.setChecked(True)
                    self._structures[current_id]["visible"] = True
                    self.visibility_changed.emit(current_id, True)
                else:
                    # Ẩn cấu trúc khác
                    checkbox.setChecked(False)
                    self._structures[current_id]["visible"] = False
                    self.visibility_changed.emit(current_id, False)

    def _show_all(self):
        """Hiển thị tất cả các cấu trúc."""
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 1)
            if name_item:
                struct_id = name_item.data(Qt.UserRole)
                checkbox = self.table.cellWidget(row, 0)
                checkbox.setChecked(True)
                self._structures[struct_id]["visible"] = True
                self.visibility_changed.emit(struct_id, True)

    def _hide_all(self):
        """Ẩn tất cả các cấu trúc."""
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 1)
            if name_item:
                struct_id = name_item.data(Qt.UserRole)
                checkbox = self.table.cellWidget(row, 0)
                checkbox.setChecked(False)
                self._structures[struct_id]["visible"] = False
                self.visibility_changed.emit(struct_id, False)

    def clear(self):
        """Xóa tất cả các cấu trúc khỏi bảng."""
        self.table.setRowCount(0)
        self._structures = {}

    def get_structure_ids(self):
        """Lấy danh sách các ID cấu trúc hiện có."""
        return list(self._structures.keys())

    def get_structure_visibility(self, struct_id):
        """
        Lấy trạng thái hiển thị của một cấu trúc.

        Parameters:
        -----------
        struct_id : str
            ID của cấu trúc.

        Returns:
        --------
        bool
            True nếu cấu trúc đang hiển thị, False nếu không.
        """
        if struct_id in self._structures:
            return self._structures[struct_id]["visible"]
        return False

    def set_structure_visibility(self, struct_id, visible):
        """
        Đặt trạng thái hiển thị cho một cấu trúc.

        Parameters:
        -----------
        struct_id : str
            ID của cấu trúc.
        visible : bool
            Trạng thái hiển thị mới.
        """
        if struct_id not in self._structures:
            return

        # Cập nhật trạng thái
        self._structures[struct_id]["visible"] = visible

        # Cập nhật UI
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 1)
            if name_item and name_item.data(Qt.UserRole) == struct_id:
                checkbox = self.table.cellWidget(row, 0)
                checkbox.setChecked(visible)
                break


# Test standalone
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    window = StructureVisibilityPanel()
    window.setWindowTitle("Structure Visibility Panel")
    window.resize(400, 300)

    # Thêm một số cấu trúc mẫu
    window.add_structure("PTV", "PTV", (1.0, 0.0, 0.0))  # Đỏ
    window.add_structure("BRAIN", "Brain", (0.0, 0.5, 1.0))  # Xanh dương nhạt
    window.add_structure("OAR1", "Brainstem", (0.8, 0.8, 0.0))  # Vàng đậm
    window.add_structure("OAR2", "Spinal Cord", (1.0, 0.5, 0.0))  # Cam
    window.add_structure("OAR3", "Optic Chiasm", (0.0, 0.8, 0.0))  # Xanh lá

    # Kết nối tín hiệu để xem kết quả
    window.visibility_changed.connect(
        lambda id, visible: print(f"Visibility changed: {id} -> {visible}")
    )
    window.color_changed.connect(
        lambda id, color: print(f"Color changed: {id} -> {color}")
    )

    window.show()

    sys.exit(app.exec_())
