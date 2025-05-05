#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Widget cho việc lựa chọn và quản lý các mức isodose và màu sắc tương ứng
cho hiển thị 3D trong QuangTPS.
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
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QPixmap, QIcon

try:
    from quangtps.ui.utils.color_map import get_eclipse_colormap, ColorMap
except ImportError:
    logging.warning("Không thể import color_map, sử dụng bản đồ màu mặc định")

    def get_eclipse_colormap():
        """Cung cấp bản đồ màu mặc định trong trường hợp không thể import color_map"""
        return {
            100: (1.0, 0.0, 0.0),  # Đỏ
            95: (1.0, 0.5, 0.0),  # Cam
            90: (1.0, 1.0, 0.0),  # Vàng
            80: (0.0, 1.0, 0.0),  # Lục
            70: (0.0, 1.0, 1.0),  # Xanh ngọc
            60: (0.0, 0.5, 1.0),  # Xanh dương nhạt
            50: (0.0, 0.0, 1.0),  # Xanh dương
            40: (0.5, 0.0, 1.0),  # Tím nhạt
            30: (1.0, 0.0, 1.0),  # Hồng
            20: (0.7, 0.7, 0.7),  # Xám nhạt
            10: (0.5, 0.5, 0.5),  # Xám đậm
        }

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


class IsodoseSelector(QWidget):
    """
    Widget cho phép người dùng quản lý các mức isodose và màu sắc tương ứng.

    Signals:
        isodose_levels_changed: Phát ra khi có thay đổi các mức isodose hoặc màu sắc.
    """

    isodose_levels_changed = pyqtSignal(list, dict)

    def __init__(self, isodose_levels=None, isodose_colors=None, parent=None):
        """
        Khởi tạo widget IsodoseSelector.

        Parameters:
        -----------
        isodose_levels : list, optional
            Danh sách các mức isodose, mặc định là None.
        isodose_colors : dict, optional
            Dictionary ánh xạ mức isodose -> màu RGB, mặc định là None.
        parent : QWidget, optional
            Widget cha, mặc định là None.
        """
        super(IsodoseSelector, self).__init__(parent)

        # Dữ liệu
        self._isodose_levels = isodose_levels or [
            100,
            95,
            90,
            80,
            70,
            60,
            50,
            40,
            30,
            20,
            10,
        ]
        self._isodose_colors = isodose_colors or get_eclipse_colormap()

        # Kiểm tra và đảm bảo tất cả các mức isodose có màu
        for level in self._isodose_levels:
            if level not in self._isodose_colors:
                # Gán màu mặc định nếu không có
                self._isodose_colors[level] = (0.5, 0.5, 0.5)  # Xám

        # Thiết lập giao diện
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Table widget
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["% isodose", "Màu"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.horizontalHeader().resizeSection(1, 80)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.table)

        # Control buttons
        button_layout = QHBoxLayout()

        self.add_button = QPushButton("Thêm")
        self.add_button.clicked.connect(self._add_isodose)
        button_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("Xóa")
        self.remove_button.clicked.connect(self._remove_isodose)
        button_layout.addWidget(self.remove_button)

        self.reset_button = QPushButton("Mặc định")
        self.reset_button.clicked.connect(self._reset_isodose)
        button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)

        # Tạo các mục từ dữ liệu ban đầu
        self._update_table()

    def _update_table(self):
        """Cập nhật bảng isodose từ dữ liệu hiện tại."""
        self.table.setRowCount(0)  # Xóa tất cả các hàng

        # Sắp xếp các mức isodose theo thứ tự giảm dần
        sorted_levels = sorted(self._isodose_levels, reverse=True)

        # Thêm các hàng mới
        for i, level in enumerate(sorted_levels):
            self.table.insertRow(i)

            # Mức isodose
            level_item = QTableWidgetItem(f"{level:.1f}")
            level_item.setData(Qt.UserRole, level)  # Lưu giá trị thật
            self.table.setItem(i, 0, level_item)

            # Màu sắc
            color = self._isodose_colors.get(level, (0.5, 0.5, 0.5))
            color_item = QTableWidgetItem()

            # Tạo một ô màu
            rgb = [int(c * 255) for c in color]
            color_qcolor = QColor(*rgb)
            color_item.setBackground(QBrush(color_qcolor))

            self.table.setItem(i, 1, color_item)

    def _show_context_menu(self, pos):
        """Hiển thị menu ngữ cảnh khi nhấp chuột phải vào bảng."""
        row = self.table.rowAt(pos.y())
        if row >= 0:
            menu = QMenu(self)

            change_color_action = QAction("Đổi màu", self)
            change_color_action.triggered.connect(lambda: self._change_color(row))
            menu.addAction(change_color_action)

            edit_level_action = QAction("Sửa mức isodose", self)
            edit_level_action.triggered.connect(lambda: self._edit_isodose(row))
            menu.addAction(edit_level_action)

            menu.addSeparator()

            remove_action = QAction("Xóa", self)
            remove_action.triggered.connect(lambda: self._remove_isodose(row))
            menu.addAction(remove_action)

            menu.exec_(self.table.mapToGlobal(pos))

    def _change_color(self, row=None):
        """Thay đổi màu sắc cho isodose đã chọn."""
        if row is None:
            row = self.table.currentRow()

        if row >= 0:
            level_item = self.table.item(row, 0)
            level = level_item.data(Qt.UserRole)

            current_color = self._isodose_colors.get(level, (0.5, 0.5, 0.5))
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
                self._isodose_colors[level] = new_color

                # Cập nhật UI
                color_item = self.table.item(row, 1)
                color_item.setBackground(QBrush(new_qcolor))

                # Phát tín hiệu thay đổi
                self.isodose_levels_changed.emit(
                    self._isodose_levels, self._isodose_colors
                )

    def _add_isodose(self):
        """Thêm một mức isodose mới."""
        # Tìm một giá trị mặc định hợp lý cho isodose mới
        existing_levels = set(self._isodose_levels)
        for level in [85, 75, 65, 55, 45, 35, 25, 15, 5]:
            if level not in existing_levels:
                default_level = level
                break
        else:
            default_level = 50

        # Tạo và cấu hình spin box
        spin_box = QDoubleSpinBox(self)
        spin_box.setRange(0.1, 150.0)
        spin_box.setValue(default_level)
        spin_box.setDecimals(1)
        spin_box.selectAll()

        # Tạo layout
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(QLabel("Chọn mức isodose (%)"))
        dialog_layout.addWidget(spin_box)

        # Tạo dialog
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Thêm isodose")
        dialog.setText("Chọn mức isodose mới:")
        dialog.setLayout(dialog_layout)
        dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        # Thiết lập dialog
        dialog_widget = QWidget(self)
        dialog_widget.setLayout(dialog_layout)
        dialog.layout().addWidget(dialog_widget, 1, 1, 1, dialog.layout().columnCount())

        if dialog.exec_() == QMessageBox.Ok:
            new_level = spin_box.value()

            # Kiểm tra xem mức mới đã tồn tại chưa
            if new_level in self._isodose_levels:
                QMessageBox.warning(
                    self, "Lỗi", f"Mức isodose {new_level}% đã tồn tại!"
                )
                return

            # Thêm mức mới
            self._isodose_levels.append(new_level)

            # Gán màu tự động
            if new_level >= 95:
                self._isodose_colors[new_level] = (1.0, 0.0, 0.0)  # Đỏ
            elif new_level >= 90:
                self._isodose_colors[new_level] = (1.0, 0.5, 0.0)  # Cam
            elif new_level >= 80:
                self._isodose_colors[new_level] = (1.0, 1.0, 0.0)  # Vàng
            elif new_level >= 70:
                self._isodose_colors[new_level] = (0.0, 1.0, 0.0)  # Lục
            elif new_level >= 60:
                self._isodose_colors[new_level] = (0.0, 1.0, 1.0)  # Xanh ngọc
            elif new_level >= 50:
                self._isodose_colors[new_level] = (0.0, 0.5, 1.0)  # Xanh dương nhạt
            elif new_level >= 40:
                self._isodose_colors[new_level] = (0.0, 0.0, 1.0)  # Xanh dương
            elif new_level >= 30:
                self._isodose_colors[new_level] = (0.5, 0.0, 1.0)  # Tím nhạt
            elif new_level >= 20:
                self._isodose_colors[new_level] = (1.0, 0.0, 1.0)  # Hồng
            else:
                self._isodose_colors[new_level] = (0.5, 0.5, 0.5)  # Xám

            # Cập nhật UI
            self._update_table()

            # Phát tín hiệu thay đổi
            self.isodose_levels_changed.emit(self._isodose_levels, self._isodose_colors)

    def _edit_isodose(self, row=None):
        """Sửa giá trị mức isodose."""
        if row is None:
            row = self.table.currentRow()

        if row >= 0:
            level_item = self.table.item(row, 0)
            old_level = level_item.data(Qt.UserRole)

            # Tạo và cấu hình spin box
            spin_box = QDoubleSpinBox(self)
            spin_box.setRange(0.1, 150.0)
            spin_box.setValue(old_level)
            spin_box.setDecimals(1)
            spin_box.selectAll()

            # Tạo layout
            dialog_layout = QVBoxLayout()
            dialog_layout.addWidget(QLabel("Sửa mức isodose (%)"))
            dialog_layout.addWidget(spin_box)

            # Tạo dialog
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Sửa isodose")
            dialog.setText(f"Mức isodose hiện tại: {old_level}%")
            dialog.setLayout(dialog_layout)
            dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

            # Thiết lập dialog
            dialog_widget = QWidget(self)
            dialog_widget.setLayout(dialog_layout)
            dialog.layout().addWidget(
                dialog_widget, 1, 1, 1, dialog.layout().columnCount()
            )

            if dialog.exec_() == QMessageBox.Ok:
                new_level = spin_box.value()

                # Kiểm tra xem mức mới đã tồn tại chưa
                if new_level != old_level and new_level in self._isodose_levels:
                    QMessageBox.warning(
                        self, "Lỗi", f"Mức isodose {new_level}% đã tồn tại!"
                    )
                    return

                # Cập nhật mức
                self._isodose_levels.remove(old_level)
                self._isodose_levels.append(new_level)

                # Cập nhật màu
                self._isodose_colors[new_level] = self._isodose_colors.pop(old_level)

                # Cập nhật UI
                self._update_table()

                # Phát tín hiệu thay đổi
                self.isodose_levels_changed.emit(
                    self._isodose_levels, self._isodose_colors
                )

    def _remove_isodose(self, row=None):
        """Xóa mức isodose đã chọn."""
        if row is None:
            row = self.table.currentRow()

        if row >= 0:
            level_item = self.table.item(row, 0)
            level = level_item.data(Qt.UserRole)

            confirm = QMessageBox.question(
                self,
                "Xác nhận xóa",
                f"Bạn có chắc muốn xóa mức isodose {level}%?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if confirm == QMessageBox.Yes:
                # Xóa khỏi danh sách
                self._isodose_levels.remove(level)
                # Xóa khỏi dictionary màu
                if level in self._isodose_colors:
                    del self._isodose_colors[level]

                # Cập nhật UI
                self._update_table()

                # Phát tín hiệu thay đổi
                self.isodose_levels_changed.emit(
                    self._isodose_levels, self._isodose_colors
                )

    def _reset_isodose(self):
        """Đặt lại các mức và màu sắc isodose về mặc định."""
        confirm = QMessageBox.question(
            self,
            "Xác nhận đặt lại",
            "Bạn có chắc muốn đặt lại tất cả các mức và màu sắc isodose về mặc định?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if confirm == QMessageBox.Yes:
            # Đặt lại về mặc định
            self._isodose_levels = [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
            self._isodose_colors = get_eclipse_colormap()

            # Cập nhật UI
            self._update_table()

            # Phát tín hiệu thay đổi
            self.isodose_levels_changed.emit(self._isodose_levels, self._isodose_colors)

    def get_isodose_levels(self):
        """Lấy danh sách các mức isodose."""
        return sorted(self._isodose_levels, reverse=True)

    def get_isodose_colors(self):
        """Lấy dictionary các màu isodose."""
        return self._isodose_colors

    def set_isodose_levels(self, levels):
        """Đặt danh sách các mức isodose."""
        if not levels:
            return

        self._isodose_levels = list(levels)

        # Đảm bảo tất cả các mức đều có màu
        for level in self._isodose_levels:
            if level not in self._isodose_colors:
                # Gán màu mặc định
                if level >= 95:
                    self._isodose_colors[level] = (1.0, 0.0, 0.0)  # Đỏ
                elif level >= 90:
                    self._isodose_colors[level] = (1.0, 0.5, 0.0)  # Cam
                elif level >= 80:
                    self._isodose_colors[level] = (1.0, 1.0, 0.0)  # Vàng
                elif level >= 70:
                    self._isodose_colors[level] = (0.0, 1.0, 0.0)  # Lục
                elif level >= 60:
                    self._isodose_colors[level] = (0.0, 1.0, 1.0)  # Xanh ngọc
                elif level >= 50:
                    self._isodose_colors[level] = (0.0, 0.5, 1.0)  # Xanh dương nhạt
                elif level >= 40:
                    self._isodose_colors[level] = (0.0, 0.0, 1.0)  # Xanh dương
                elif level >= 30:
                    self._isodose_colors[level] = (0.5, 0.0, 1.0)  # Tím nhạt
                elif level >= 20:
                    self._isodose_colors[level] = (1.0, 0.0, 1.0)  # Hồng
                else:
                    self._isodose_colors[level] = (0.5, 0.5, 0.5)  # Xám

        # Cập nhật UI
        self._update_table()

        # Phát tín hiệu thay đổi
        self.isodose_levels_changed.emit(self._isodose_levels, self._isodose_colors)


# Test standalone
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    window = IsodoseSelector()
    window.setWindowTitle("Isodose Selector")
    window.resize(400, 300)
    window.show()

    sys.exit(app.exec_())
