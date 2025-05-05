#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module IsodoseSelector cho QuangTPS.

Widget cho phép quản lý và tùy chỉnh các mức isodose hiển thị trong giao diện 3D.
Có thể điều chỉnh màu sắc, hiển thị/ẩn và thêm/xóa các mức isodose.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional

# Import PyQt5 components with try/except để bắt lỗi nếu không có PyQt5
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
        QInputDialog,
        QMessageBox,
        QCheckBox,
        QSpinBox,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QFrame,
        QApplication,  # Thêm QApplication cho phần test standalone
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QColor, QBrush, QIcon

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


from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class IsodoseSelector(QWidget):
    """
    Widget cho phép người dùng quản lý các mức isodose hiển thị.

    Cung cấp giao diện để thêm, xóa, điều chỉnh màu sắc và hiển thị/ẩn các mức isodose.
    Phát tín hiệu khi có thay đổi để các thành phần khác có thể cập nhật hiển thị.
    """

    # Tín hiệu phát ra khi isodose thay đổi (thêm/xóa/điều chỉnh)
    dataChanged = pyqtSignal()
    # Thêm tín hiệu mới đồng bộ với DoseVisualization3D
    isodose_levels_changed = pyqtSignal(dict)  # Dict mapping level -> IsodoseLevel

    def __init__(self, parent=None):
        """Khởi tạo IsodoseSelector widget."""
        super(IsodoseSelector, self).__init__(parent)

        # Lưu trữ dữ liệu isodose (mức, màu, hiển thị)
        self.isodose_levels = {}  # Dict mapping level (float) to (color, visible)

        # Khởi tạo giao diện
        self.setup_ui()

        # Màu mặc định và mức prescription
        self.default_colors = [
            (1.0, 0.0, 0.0),  # Đỏ - 100%
            (1.0, 0.5, 0.0),  # Cam - 95%
            (1.0, 1.0, 0.0),  # Vàng - 90%
            (0.0, 1.0, 0.0),  # Xanh lá - 80%
            (0.0, 1.0, 0.5),  # Ngọc - 70%
            (0.0, 1.0, 1.0),  # Xanh nhạt - 50%
            (0.0, 0.5, 1.0),  # Xanh dương nhạt - 30%
            (0.0, 0.0, 1.0),  # Xanh dương - 20%
            (0.5, 0.0, 1.0),  # Tím - 10%
        ]
        self.default_percentages = [100, 95, 90, 80, 70, 50, 30, 20, 10]
        self.prescription_dose = 60.0  # Gy

        # Thiết lập ban đầu
        self._init_default_levels()

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Frame cho prescription dose
        prescription_frame = QFrame()
        prescription_layout = QHBoxLayout(prescription_frame)
        prescription_layout.setContentsMargins(2, 2, 2, 2)

        # Label và SpinBox cho prescription dose
        prescription_layout.addWidget(QLabel("Liều kê toa:"))
        self.prescription_spinbox = QDoubleSpinBox()
        self.prescription_spinbox.setRange(0.1, 1000.0)
        self.prescription_spinbox.setValue(self.prescription_dose)
        self.prescription_spinbox.setSuffix(" Gy")
        self.prescription_spinbox.setDecimals(1)
        self.prescription_spinbox.setToolTip("Liều kê toa dùng để tính % isodose")
        self.prescription_spinbox.valueChanged.connect(self._update_prescription)
        prescription_layout.addWidget(self.prescription_spinbox)

        main_layout.addWidget(prescription_frame)

        # Bảng isodose
        self.isodose_table = QTableWidget()
        self.isodose_table.setColumnCount(3)
        self.isodose_table.setHorizontalHeaderLabels(["Mức", "Màu", "Hiển thị"])
        self.isodose_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.isodose_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.isodose_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.isodose_table.setColumnWidth(1, 40)
        self.isodose_table.setColumnWidth(2, 40)
        self.isodose_table.verticalHeader().setVisible(False)
        self.isodose_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.isodose_table.itemDoubleClicked.connect(self._on_item_double_clicked)

        main_layout.addWidget(self.isodose_table)

        # Nút thêm/xóa/reset
        button_layout = QHBoxLayout()

        self.add_button = QPushButton("Thêm")
        self.add_button.setToolTip("Thêm một mức isodose mới")
        self.add_button.clicked.connect(self._add_isodose)
        button_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("Xóa")
        self.remove_button.setToolTip("Xóa mức isodose đã chọn")
        self.remove_button.clicked.connect(self._remove_isodose)
        button_layout.addWidget(self.remove_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Khôi phục về mức mặc định")
        self.reset_button.clicked.connect(self._reset_isodose)
        button_layout.addWidget(self.reset_button)

        main_layout.addLayout(button_layout)

    def _init_default_levels(self):
        """Khởi tạo các mức isodose mặc định."""
        self.isodose_levels.clear()
        self.isodose_table.setRowCount(0)

        # Thêm các mức isodose mặc định
        for i, percent in enumerate(self.default_percentages):
            level = self.prescription_dose * percent / 100.0
            if i < len(self.default_colors):
                color = self.default_colors[i]
            else:
                # Tạo màu ngẫu nhiên nếu hết màu mặc định
                import random

                color = (random.random(), random.random(), random.random())

            self.isodose_levels[level] = (color, True)
            self._add_level_to_table(level, color, True)

    def _add_level_to_table(self, level, color, visible=True):
        """
        Thêm một mức isodose vào bảng.

        Parameters:
        -----------
        level : float
            Mức liều (Gy)
        color : tuple
            Màu RGB (các giá trị từ 0.0 đến 1.0)
        visible : bool
            Trạng thái hiển thị
        """
        row = self.isodose_table.rowCount()
        self.isodose_table.insertRow(row)

        # Cột mức liều
        level_item = QTableWidgetItem(f"{level:.1f} Gy")
        level_item.setData(Qt.UserRole, level)
        self.isodose_table.setItem(row, 0, level_item)

        # Cột màu
        color_item = QTableWidgetItem()
        qcolor = QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        color_item.setBackground(QBrush(qcolor))
        color_item.setData(Qt.UserRole, color)
        self.isodose_table.setItem(row, 1, color_item)

        # Cột hiển thị
        checkbox = QCheckBox()
        checkbox.setChecked(visible)
        checkbox.stateChanged.connect(
            lambda state, r=row: self._toggle_visibility(r, state)
        )

        # Đặt checkbox vào ô
        self.isodose_table.setCellWidget(row, 2, checkbox)

    def _on_item_double_clicked(self, item):
        """Xử lý sự kiện khi người dùng nhấp đúp vào một ô trong bảng."""
        row = item.row()
        col = item.column()

        if col == 1:  # Cột màu
            self._change_color(row)
        elif col == 0:  # Cột mức liều
            self._change_level(row)

    def _change_color(self, row):
        """
        Thay đổi màu của mức isodose.

        Parameters:
        -----------
        row : int
            Chỉ số hàng trong bảng
        """
        if row < 0 or row >= self.isodose_table.rowCount():
            return

        # Lấy thông tin hiện tại
        color_item = self.isodose_table.item(row, 1)
        level_item = self.isodose_table.item(row, 0)

        if not color_item or not level_item:
            return

        level = level_item.data(Qt.UserRole)
        current_color = color_item.data(Qt.UserRole)

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

            # Cập nhật màu trong bảng
            color_item.setBackground(QBrush(qcolor))
            color_item.setData(Qt.UserRole, new_color)

            # Cập nhật màu trong dữ liệu
            visible = self.isodose_levels[level][1]
            self.isodose_levels[level] = (new_color, visible)

            # Phát tín hiệu thay đổi
            self.dataChanged.emit()

            # Phát tín hiệu mới cho DoseVisualization3D
            self.isodose_levels_changed.emit(self.get_isodose_levels())

    def _change_level(self, row):
        """
        Thay đổi mức liều của isodose.

        Parameters:
        -----------
        row : int
            Chỉ số hàng trong bảng
        """
        if row < 0 or row >= self.isodose_table.rowCount():
            return

        # Lấy thông tin hiện tại
        level_item = self.isodose_table.item(row, 0)

        if not level_item:
            return

        old_level = level_item.data(Qt.UserRole)

        # Hiển thị hộp thoại nhập giá trị mới
        value, ok = QInputDialog.getDouble(
            self,
            "Thay đổi mức isodose",
            "Mức isodose mới (Gy):",
            old_level,
            0.1,
            1000.0,
            1,
        )

        if ok and value != old_level:
            # Kiểm tra mức này đã tồn tại chưa
            if value in self.isodose_levels and value != old_level:
                QMessageBox.warning(
                    self,
                    "Mức isodose đã tồn tại",
                    f"Mức isodose {value:.1f} Gy đã tồn tại trong danh sách.",
                )
                return

            # Lưu thông tin cũ
            color, visible = self.isodose_levels[old_level]

            # Xóa mức cũ
            del self.isodose_levels[old_level]

            # Thêm mức mới
            self.isodose_levels[value] = (color, visible)

            # Cập nhật lại hiển thị
            level_item.setText(f"{value:.1f} Gy")
            level_item.setData(Qt.UserRole, value)

            # Phát tín hiệu thay đổi
            self.dataChanged.emit()

    def _toggle_visibility(self, row, state):
        """
        Bật/tắt hiển thị một mức isodose.

        Parameters:
        -----------
        row : int
            Chỉ số hàng trong bảng
        state : int
            Trạng thái của checkbox (Qt.Checked hoặc Qt.Unchecked)
        """
        if row < 0 or row >= self.isodose_table.rowCount():
            return

        # Lấy mức isodose
        level_item = self.isodose_table.item(row, 0)

        if not level_item:
            return

        level = level_item.data(Qt.UserRole)

        # Cập nhật trạng thái hiển thị
        if level in self.isodose_levels:
            color = self.isodose_levels[level][0]
            visible = state == Qt.Checked
            self.isodose_levels[level] = (color, visible)

            # Phát tín hiệu thay đổi
            self.dataChanged.emit()

            # Phát tín hiệu mới cho DoseVisualization3D
            self.isodose_levels_changed.emit(self.get_isodose_levels())

    def _add_isodose(self):
        """Thêm một mức isodose mới."""
        # Hỏi phần trăm của liều kê toa
        percent, ok = QInputDialog.getDouble(
            self,
            "Thêm mức isodose",
            "Phần trăm của liều kê toa (%):",
            50.0,
            0.1,
            200.0,
            1,
        )

        if not ok:
            return

        # Tính mức theo Gy
        level = self.prescription_dose * percent / 100.0

        # Kiểm tra mức này đã tồn tại chưa
        if level in self.isodose_levels:
            QMessageBox.warning(
                self,
                "Mức isodose đã tồn tại",
                f"Mức isodose {level:.1f} Gy đã tồn tại trong danh sách.",
            )
            return

        # Chọn màu
        color_dialog = QColorDialog(self)

        if color_dialog.exec_():
            qcolor = color_dialog.selectedColor()
            color = (
                qcolor.red() / 255.0,
                qcolor.green() / 255.0,
                qcolor.blue() / 255.0,
            )

            # Thêm vào dữ liệu
            self.isodose_levels[level] = (color, True)

            # Thêm vào bảng
            self._add_level_to_table(level, color, True)

            # Sắp xếp lại bảng theo thứ tự giảm dần
            self._sort_table()

            # Phát tín hiệu thay đổi
            self.dataChanged.emit()

            # Phát tín hiệu mới cho DoseVisualization3D
            self.isodose_levels_changed.emit(self.get_isodose_levels())

    def _remove_isodose(self):
        """Xóa mức isodose đã chọn."""
        # Lấy hàng được chọn
        selected_rows = self.isodose_table.selectionModel().selectedRows()

        if not selected_rows:
            QMessageBox.information(
                self, "Chưa chọn mức", "Vui lòng chọn một mức isodose để xóa."
            )
            return

        # Xác nhận xóa
        answer = QMessageBox.question(
            self,
            "Xác nhận xóa",
            "Bạn có chắc chắn muốn xóa các mức isodose đã chọn?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        # Xóa từng mức đã chọn
        for index in sorted([i.row() for i in selected_rows], reverse=True):
            level_item = self.isodose_table.item(index, 0)

            if level_item:
                level = level_item.data(Qt.UserRole)

                # Xóa khỏi dữ liệu
                if level in self.isodose_levels:
                    del self.isodose_levels[level]

                # Xóa khỏi bảng
                self.isodose_table.removeRow(index)

        # Phát tín hiệu thay đổi
        self.dataChanged.emit()

        # Phát tín hiệu mới cho DoseVisualization3D
        self.isodose_levels_changed.emit(self.get_isodose_levels())

    def _reset_isodose(self):
        """Khôi phục về các mức isodose mặc định."""
        # Xác nhận reset
        answer = QMessageBox.question(
            self,
            "Xác nhận khôi phục",
            "Bạn có chắc chắn muốn khôi phục về các mức isodose mặc định?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        # Khởi tạo lại
        self._init_default_levels()

        # Phát tín hiệu thay đổi
        self.dataChanged.emit()

        # Phát tín hiệu mới cho DoseVisualization3D
        self.isodose_levels_changed.emit(self.get_isodose_levels())

    def _update_prescription(self, value):
        """
        Cập nhật liều kê toa và điều chỉnh các mức isodose tương ứng.

        Parameters:
        -----------
        value : float
            Giá trị liều kê toa mới (Gy)
        """
        # Lưu giá trị cũ
        old_value = self.prescription_dose
        self.prescription_dose = value

        # Tính tỷ lệ
        ratio = value / old_value if old_value > 0 else 1.0

        # Cập nhật các mức isodose
        new_levels = {}
        for level, (color, visible) in self.isodose_levels.items():
            new_level = level * ratio
            new_levels[new_level] = (color, visible)

        self.isodose_levels = new_levels

        # Cập nhật bảng
        self.isodose_table.setRowCount(0)
        for level, (color, visible) in sorted(
            self.isodose_levels.items(), key=lambda x: x[0], reverse=True
        ):
            self._add_level_to_table(level, color, visible)

        # Phát tín hiệu thay đổi
        self.dataChanged.emit()

        # Phát tín hiệu mới cho DoseVisualization3D
        self.isodose_levels_changed.emit(self.get_isodose_levels())

    def _sort_table(self):
        """Sắp xếp bảng theo thứ tự giảm dần của mức isodose."""
        # Tạo danh sách các mục để sắp xếp
        items = []
        for row in range(self.isodose_table.rowCount()):
            level_item = self.isodose_table.item(row, 0)
            if level_item:
                level = level_item.data(Qt.UserRole)
                color_item = self.isodose_table.item(row, 1)
                checkbox = self.isodose_table.cellWidget(row, 2)

                if color_item and checkbox:
                    color = color_item.data(Qt.UserRole)
                    visible = checkbox.isChecked()
                    items.append((level, color, visible))

        # Sắp xếp giảm dần
        items.sort(reverse=True)

        # Cập nhật lại bảng
        self.isodose_table.setRowCount(0)
        for level, color, visible in items:
            self._add_level_to_table(level, color, visible)

    def get_active_levels(self):
        """
        Lấy danh sách các mức isodose đang hoạt động.

        Returns:
        --------
        list
            Danh sách các tuple (level, color, opacity) cho các mức đang hiển thị
        """
        active_levels = []
        for level, (color, visible) in sorted(
            self.isodose_levels.items(), key=lambda x: x[0], reverse=True
        ):
            if visible:
                # Mức, màu và độ đục mặc định
                active_levels.append((level, color, 0.7))

        return active_levels

    def get_isodose_levels(self):
        """
        Lấy dictionary các mức isodose với định dạng phù hợp cho DoseVisualization3D.

        Returns:
        --------
        dict
            Dictionary mapping level -> IsodoseLevel objects
        """
        from quangtps.ui.dose_visualization_3d import IsodoseLevel

        result = {}
        for level, (color, visible) in self.isodose_levels.items():
            result[level] = IsodoseLevel(level, color, visible)

        return result

    def set_isodose_levels(self, isodose_dict):
        """
        Cập nhật các mức isodose từ đối tượng bên ngoài.

        Parameters:
        -----------
        isodose_dict : dict
            Dictionary mapping level -> IsodoseLevel object or (color, visible) tuple
        """
        if not isodose_dict:
            return

        # Xóa dữ liệu hiện tại
        self.isodose_levels.clear()
        self.isodose_table.setRowCount(0)

        # Thêm các mức mới
        for level, value in isodose_dict.items():
            if hasattr(value, "color") and hasattr(value, "visible"):
                # IsodoseLevel object
                color = value.color
                visible = value.visible
            elif isinstance(value, tuple) and len(value) >= 2:
                # (color, visible) tuple
                color, visible = value
            else:
                logger.warning(f"Không thể xác định dữ liệu isodose cho mức {level}")
                continue

            self.isodose_levels[level] = (color, visible)
            self._add_level_to_table(level, color, visible)

        # Sắp xếp bảng
        self._sort_table()

    def set_prescription_dose(self, dose):
        """
        Đặt liều kê toa.

        Parameters:
        -----------
        dose : float
            Liều kê toa mới (Gy)
        """
        if dose > 0:
            self.prescription_spinbox.setValue(dose)

    def clear(self):
        """Xóa tất cả các mức isodose."""
        self.isodose_levels.clear()
        self.isodose_table.setRowCount(0)
        self.dataChanged.emit()


# Test standalone
if __name__ == "__main__":
    import sys

    # Kiểm tra xem PyQt5 có sẵn không
    if not PYQT5_AVAILABLE:
        print("PyQt5 không khả dụng, không thể chạy test standalone")
        sys.exit(1)

    app = QApplication(sys.argv)
    widget = IsodoseSelector()
    widget.setWindowTitle("Isodose Selector")
    widget.resize(300, 400)
    widget.show()

    sys.exit(app.exec_())
