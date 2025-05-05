#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module dialog ColorMapEditorDialog cho QuangTPS.

Dialog cho phép người dùng chỉnh sửa chi tiết các điểm màu trong colormap tùy chỉnh.
"""

import os
import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# Import PyQt5 components with try/except
try:
    from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QColorDialog,
        QMessageBox,
        QDialogButtonBox,
        QGroupBox,
        QDoubleSpinBox,
        QFrame,
        QGridLayout,
        QMenu,
        QAction,
        QApplication,  # Thêm QApplication cho test standalone
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint
    from PyQt5.QtGui import QColor, QBrush, QIcon, QPainter, QPixmap, QImage, QPen

    # Đánh dấu rằng PyQt5 đã được import thành công
    PYQT5_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import PyQt5: {e}")
    PYQT5_AVAILABLE = False

# Try to import matplotlib
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import cm
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.error("Không thể import matplotlib, dialog sẽ không hoạt động đầy đủ")
    MATPLOTLIB_AVAILABLE = False

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class ColorMapEditorDialog(QDialog):
    """
    Dialog cho phép chỉnh sửa chi tiết colormap tùy chỉnh.

    Hỗ trợ thêm/xóa/sửa các điểm màu và xem trước kết quả.
    """

    def __init__(self, colormap_data, parent=None):
        """
        Khởi tạo dialog.

        Parameters:
        -----------
        colormap_data : dict
            Dữ liệu colormap dạng {"name": "name", "colors": [(pos, (r,g,b)), ...]}
        parent : QWidget
            Widget cha
        """
        super(ColorMapEditorDialog, self).__init__(parent)

        # Lưu trữ dữ liệu
        self.colormap_name = colormap_data["name"]
        self.colormap_colors = colormap_data["colors"].copy()
        self.colormap = None

        # Thiết lập giao diện
        self.setup_ui()

        # Hiển thị dữ liệu
        self._display_data()

        # Cập nhật preview
        self._update_preview()

        # Thiết lập tiêu đề
        self.setWindowTitle(f"Chỉnh sửa ColorMap: {self.colormap_name}")

        # Đặt kích thước
        self.resize(550, 450)

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)

        # Group dữ liệu
        data_group = QGroupBox("Điểm màu")
        data_layout = QVBoxLayout(data_group)

        # Bảng điểm màu
        self.color_table = QTableWidget()
        self.color_table.setColumnCount(3)
        self.color_table.setHorizontalHeaderLabels(["Vị trí", "Màu sắc", "Giá trị RGB"])
        self.color_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.color_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.color_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.color_table.verticalHeader().setVisible(False)
        self.color_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.color_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.color_table.customContextMenuRequested.connect(self._show_context_menu)
        self.color_table.itemDoubleClicked.connect(self._on_item_double_clicked)

        data_layout.addWidget(self.color_table)

        # Nút thêm/xóa
        button_layout = QHBoxLayout()

        self.add_btn = QPushButton("Thêm")
        self.add_btn.clicked.connect(self._add_color_point)
        button_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("Xóa")
        self.remove_btn.clicked.connect(self._remove_color_point)
        button_layout.addWidget(self.remove_btn)

        self.normalize_btn = QPushButton("Chuẩn hóa")
        self.normalize_btn.setToolTip("Đặt lại các điểm để phân phối đều từ 0 đến 1")
        self.normalize_btn.clicked.connect(self._normalize_positions)
        button_layout.addWidget(self.normalize_btn)

        data_layout.addLayout(button_layout)

        main_layout.addWidget(data_group)

        # Group xem trước
        preview_group = QGroupBox("Xem trước")
        preview_layout = QVBoxLayout(preview_group)

        if MATPLOTLIB_AVAILABLE:
            # Tạo matplotlib figure
            self.figure, self.ax = plt.subplots(figsize=(5, 2), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumHeight(100)
            preview_layout.addWidget(self.canvas)
        else:
            # Fallback nếu không có matplotlib
            preview_label = QLabel(
                "Matplotlib không khả dụng. Không thể hiển thị xem trước."
            )
            preview_layout.addWidget(preview_label)

        main_layout.addWidget(preview_group)

        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def _display_data(self):
        """Hiển thị dữ liệu màu lên bảng."""
        self.color_table.setRowCount(0)

        # Sắp xếp các điểm theo vị trí
        sorted_colors = sorted(self.colormap_colors, key=lambda x: x[0])

        for pos, color in sorted_colors:
            row = self.color_table.rowCount()
            self.color_table.insertRow(row)

            # Cột vị trí
            pos_item = QTableWidgetItem(f"{pos:.3f}")
            pos_item.setData(Qt.UserRole, pos)
            self.color_table.setItem(row, 0, pos_item)

            # Cột màu
            color_item = QTableWidgetItem()
            qcolor = QColor(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            color_item.setBackground(QBrush(qcolor))
            color_item.setData(Qt.UserRole, color)
            self.color_table.setItem(row, 1, color_item)

            # Cột giá trị RGB
            rgb_text = f"R: {color[0]:.3f}, G: {color[1]:.3f}, B: {color[2]:.3f}"
            rgb_item = QTableWidgetItem(rgb_text)
            self.color_table.setItem(row, 2, rgb_item)

    def _on_item_double_clicked(self, item):
        """
        Xử lý sự kiện khi nhấp đúp vào một ô.

        Parameters:
        -----------
        item : QTableWidgetItem
            Item được nhấp đúp
        """
        row = item.row()
        col = item.column()

        if col == 0:  # Cột vị trí
            self._edit_position(row)
        elif col == 1:  # Cột màu
            self._edit_color(row)

    def _show_context_menu(self, position):
        """
        Hiển thị menu ngữ cảnh.

        Parameters:
        -----------
        position : QPoint
            Vị trí chuột
        """
        if not self.color_table.selectedItems():
            return

        menu = QMenu(self)

        edit_pos_action = QAction("Sửa vị trí", self)
        edit_pos_action.triggered.connect(
            lambda: self._edit_position(self.color_table.currentRow())
        )
        menu.addAction(edit_pos_action)

        edit_color_action = QAction("Sửa màu", self)
        edit_color_action.triggered.connect(
            lambda: self._edit_color(self.color_table.currentRow())
        )
        menu.addAction(edit_color_action)

        menu.addSeparator()

        delete_action = QAction("Xóa", self)
        delete_action.triggered.connect(lambda: self._remove_color_point())
        menu.addAction(delete_action)

        menu.exec_(self.color_table.mapToGlobal(position))

    def _edit_position(self, row):
        """
        Sửa vị trí của một điểm màu.

        Parameters:
        -----------
        row : int
            Chỉ số hàng
        """
        if row < 0 or row >= len(self.colormap_colors):
            return

        # Lấy giá trị hiện tại
        pos_item = self.color_table.item(row, 0)
        if not pos_item:
            return

        current_pos = pos_item.data(Qt.UserRole)

        # Tạo dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Sửa vị trí")

        layout = QVBoxLayout(dialog)

        label = QLabel("Vị trí (0.0-1.0):")
        layout.addWidget(label)

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setDecimals(3)
        spin.setValue(current_pos)
        layout.addWidget(spin)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_():
            new_pos = spin.value()

            # Cập nhật dữ liệu
            color = self.colormap_colors[row][1]
            self.colormap_colors[row] = (new_pos, color)

            # Cập nhật UI
            self._display_data()
            self._update_preview()

    def _edit_color(self, row):
        """
        Sửa màu của một điểm.

        Parameters:
        -----------
        row : int
            Chỉ số hàng
        """
        if row < 0 or row >= len(self.colormap_colors):
            return

        # Lấy giá trị hiện tại
        color_item = self.color_table.item(row, 1)
        if not color_item:
            return

        current_color = color_item.data(Qt.UserRole)

        # Hiển thị color dialog
        qcolor = QColor(
            int(current_color[0] * 255),
            int(current_color[1] * 255),
            int(current_color[2] * 255),
        )

        color_dialog = QColorDialog(self)
        color_dialog.setCurrentColor(qcolor)

        if color_dialog.exec_():
            selected_color = color_dialog.selectedColor()
            new_color = (
                selected_color.red() / 255.0,
                selected_color.green() / 255.0,
                selected_color.blue() / 255.0,
            )

            # Cập nhật dữ liệu
            pos = self.colormap_colors[row][0]
            self.colormap_colors[row] = (pos, new_color)

            # Cập nhật UI
            self._display_data()
            self._update_preview()

    def _add_color_point(self):
        """Thêm một điểm màu mới."""
        # Tìm vị trí tốt nhất để thêm
        positions = [pos for pos, _ in self.colormap_colors]
        if not positions:
            # Nếu chưa có điểm nào, thêm tại 0.5
            new_pos = 0.5
        else:
            # Tìm khoảng trống lớn nhất
            sorted_positions = sorted(positions)

            # Thêm các điểm biên nếu cần
            if sorted_positions[0] > 0.0:
                sorted_positions.insert(0, 0.0)
            if sorted_positions[-1] < 1.0:
                sorted_positions.append(1.0)

            max_gap = 0
            max_gap_pos = 0.5

            for i in range(len(sorted_positions) - 1):
                gap = sorted_positions[i + 1] - sorted_positions[i]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_pos = sorted_positions[i] + gap / 2

            new_pos = max_gap_pos

        # Tạo màu mặc định (trắng)
        new_color = (1.0, 1.0, 1.0)

        # Hiển thị dialog chọn màu
        color_dialog = QColorDialog(self)

        if color_dialog.exec_():
            selected_color = color_dialog.selectedColor()
            new_color = (
                selected_color.red() / 255.0,
                selected_color.green() / 255.0,
                selected_color.blue() / 255.0,
            )

        # Thêm vào danh sách
        self.colormap_colors.append((new_pos, new_color))

        # Cập nhật UI
        self._display_data()
        self._update_preview()

    def _remove_color_point(self):
        """Xóa điểm màu đã chọn."""
        selected_rows = [
            index.row() for index in self.color_table.selectionModel().selectedRows()
        ]

        if not selected_rows:
            QMessageBox.information(
                self, "Chọn điểm màu", "Vui lòng chọn điểm màu cần xóa."
            )
            return

        # Kiểm tra số lượng điểm tối thiểu
        if len(self.colormap_colors) - len(selected_rows) < 2:
            QMessageBox.warning(
                self, "Không thể xóa", "ColorMap phải có ít nhất 2 điểm màu."
            )
            return

        # Xác nhận xóa
        answer = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa {len(selected_rows)} điểm màu?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        # Xóa theo thứ tự giảm dần
        for row in sorted(selected_rows, reverse=True):
            if 0 <= row < len(self.colormap_colors):
                self.colormap_colors.pop(row)

        # Cập nhật UI
        self._display_data()
        self._update_preview()

    def _normalize_positions(self):
        """Chuẩn hóa vị trí các điểm để phân phối đều từ 0 đến 1."""
        if len(self.colormap_colors) < 2:
            return

        # Xác nhận
        answer = QMessageBox.question(
            self,
            "Xác nhận chuẩn hóa",
            "Bạn có chắc chắn muốn đặt lại các vị trí để phân phối đều từ 0 đến 1?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        # Sắp xếp theo vị trí
        sorted_colors = sorted(self.colormap_colors, key=lambda x: x[0])

        # Tính các vị trí mới
        n = len(sorted_colors)
        for i in range(n):
            new_pos = i / (n - 1) if n > 1 else 0.5
            sorted_colors[i] = (new_pos, sorted_colors[i][1])

        # Cập nhật dữ liệu
        self.colormap_colors = sorted_colors

        # Cập nhật UI
        self._display_data()
        self._update_preview()

    def _update_preview(self):
        """Cập nhật xem trước colormap."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, "ax"):
            return

        # Xóa dữ liệu cũ
        self.ax.clear()

        # Tạo colormap từ dữ liệu
        sorted_colors = sorted(self.colormap_colors, key=lambda x: x[0])
        positions = [pos for pos, _ in sorted_colors]
        rgb_values = [rgb for _, rgb in sorted_colors]

        # Tạo colormap
        cdict = {
            "red": [(pos, rgb[0], rgb[0]) for pos, rgb in zip(positions, rgb_values)],
            "green": [(pos, rgb[1], rgb[1]) for pos, rgb in zip(positions, rgb_values)],
            "blue": [(pos, rgb[2], rgb[2]) for pos, rgb in zip(positions, rgb_values)],
        }

        self.colormap = LinearSegmentedColormap(self.colormap_name, cdict)

        # Tạo gradient
        gradient = np.linspace(0, 1, 100)
        gradient = np.vstack((gradient, gradient))

        # Vẽ colormap
        self.ax.imshow(gradient, aspect="auto", cmap=self.colormap)
        self.ax.set_title("ColorMap Preview")
        self.ax.set_yticks([])
        self.ax.set_xticks([0, 99])
        self.ax.set_xticklabels(["0.0", "1.0"])

        # Vẽ các điểm màu
        y_pos = 0.5
        for pos, rgb in sorted_colors:
            x_pos = pos * 99  # Vị trí trên trục x
            self.ax.plot(x_pos, y_pos, "o", color="white", markersize=8)
            self.ax.plot(
                x_pos, y_pos, "o", color="black", markersize=6, fillstyle="none"
            )

        # Cập nhật canvas
        self.canvas.draw()

    def get_colormap_data(self):
        """
        Lấy dữ liệu colormap đã chỉnh sửa.

        Returns:
        --------
        dict
            Dữ liệu colormap dạng {"name": "name", "colors": [(pos, (r,g,b)), ...]}
        """
        return {"name": self.colormap_name, "colors": self.colormap_colors}

    def get_colormap(self):
        """
        Lấy đối tượng colormap đã tạo.

        Returns:
        --------
        LinearSegmentedColormap or None
            Đối tượng colormap hoặc None nếu chưa được tạo
        """
        return self.colormap


# Test standalone
if __name__ == "__main__":
    import sys

    # Kiểm tra xem PyQt5 và matplotlib có sẵn không
    if not PYQT5_AVAILABLE:
        print("PyQt5 không khả dụng, không thể chạy test standalone")
        sys.exit(1)

    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib không khả dụng, không thể chạy test standalone")
        sys.exit(1)

    app = QApplication(sys.argv)

    # Dữ liệu colormap mẫu
    sample_colormap = {
        "name": "Test ColorMap",
        "colors": [
            (0.0, (0.0, 0.0, 0.0)),  # Black
            (0.5, (1.0, 0.0, 0.0)),  # Red
            (1.0, (1.0, 1.0, 0.0)),  # Yellow
        ],
    }

    dialog = ColorMapEditorDialog(sample_colormap)
    if dialog.exec_():
        print("Colormap updated:", dialog.get_colormap_data())
    else:
        print("Dialog canceled")
