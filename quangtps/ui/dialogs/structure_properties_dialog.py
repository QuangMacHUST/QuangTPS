#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure Properties Dialog
==========================

Dialog để xem và chỉnh sửa thuộc tính của cấu trúc giải phẫu.
"""

import logging
from typing import Optional, Dict, Any, Tuple

try:
    from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QFormLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QComboBox,
        QTextEdit,
        QGroupBox,
        QCheckBox,
        QDialogButtonBox,
        QMessageBox,
        QColorDialog,
        QSpinBox,
        QDoubleSpinBox,
    )
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtGui import QIcon, QColor, QPixmap

    HAS_PYQT = True
except ImportError:
    logging.warning("PyQt5 không khả dụng. Sử dụng lớp giả mạch.")
    HAS_PYQT = False

# Import các module của hệ thống QuangTPS
try:
    from quangtps.structures.structure import Structure, StructureType
    from quangtps.ui.styles.eclipse_style_theme import apply_eclipse_theme_to_widget
except ImportError:
    logging.warning(
        "Không thể import các module QuangTPS cần thiết. Sử dụng lớp giả mạch."
    )

    class StructureType:
        """Lớp giả mạch cho StructureType khi không thể import."""

        PTV = "PTV"
        CTV = "CTV"
        GTV = "GTV"
        OAR = "OAR"
        EXTERNAL = "EXTERNAL"
        BOLUS = "BOLUS"
        OTHER = "OTHER"

    class Structure:
        """Lớp giả mạch cho Structure khi không thể import."""

        def __init__(self, name="", structure_type=None, color=(255, 0, 0)):
            self.name = name
            self.structure_type = structure_type or StructureType.OTHER
            self.color = color
            self.visible = True
            self.description = ""
            self.volume = 0.0


class ColorButton(QPushButton):
    """
    Button hiển thị và cho phép chọn màu sắc.
    """

    def __init__(self, color=None, parent=None):
        """
        Khởi tạo button màu sắc.

        Args:
            color: Màu sắc ban đầu dưới dạng tuple (r, g, b)
            parent: Widget cha
        """
        super().__init__(parent)
        self.setFixedSize(40, 25)
        self._color = color or (255, 0, 0)  # Mặc định là màu đỏ
        self._update_button_color()

        self.clicked.connect(self._choose_color)

    def _update_button_color(self):
        """Cập nhật màu sắc hiển thị trên button."""
        if not HAS_PYQT:
            return

        r, g, b = self._color
        color = QColor(r, g, b)

        # Tạo một pixmap với màu sắc đã chọn
        pixmap = QPixmap(self.size())
        pixmap.fill(color)

        # Thiết lập icon cho button
        self.setIcon(QIcon(pixmap))
        self.setIconSize(self.size() - QSize(10, 10))

        # Thiết lập stylesheet
        self.setStyleSheet(f"QPushButton {{ background-color: rgb({r}, {g}, {b}); }}")

    def _choose_color(self):
        """Mở dialog chọn màu sắc."""
        if not HAS_PYQT:
            return

        r, g, b = self._color
        initial_color = QColor(r, g, b)

        color = QColorDialog.getColor(initial_color, self, "Chọn màu sắc cho cấu trúc")

        if color.isValid():
            self._color = (color.red(), color.green(), color.blue())
            self._update_button_color()

    def get_color(self) -> Tuple[int, int, int]:
        """Lấy màu sắc hiện tại."""
        return self._color

    def set_color(self, color: Tuple[int, int, int]):
        """Thiết lập màu sắc mới."""
        self._color = color
        self._update_button_color()


class StructurePropertiesDialog(QDialog):
    """
    Dialog để xem và chỉnh sửa thuộc tính của cấu trúc giải phẫu.
    """

    def __init__(self, structure: Structure, parent=None):
        """
        Khởi tạo dialog thuộc tính cấu trúc.

        Args:
            structure: Cấu trúc giải phẫu cần chỉnh sửa thuộc tính
            parent: Widget cha
        """
        if not HAS_PYQT:
            logging.warning(
                "PyQt5 không khả dụng. StructurePropertiesDialog sẽ không hoạt động."
            )
            return

        super().__init__(parent)
        self.setWindowTitle("Thuộc tính cấu trúc")
        self.setMinimumSize(500, 400)

        self.structure = structure
        self._init_ui()
        self._load_structure_data()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout(self)

        # Form chính
        form_layout = QFormLayout()

        # Tên cấu trúc
        self.name_edit = QLineEdit()
        form_layout.addRow("Tên cấu trúc:", self.name_edit)

        # Loại cấu trúc
        self.type_combo = QComboBox()
        self.type_combo.addItems(
            ["PTV", "CTV", "GTV", "OAR", "EXTERNAL", "BOLUS", "OTHER"]
        )
        form_layout.addRow("Loại cấu trúc:", self.type_combo)

        # Màu sắc
        self.color_button = ColorButton()
        form_layout.addRow("Màu sắc:", self.color_button)

        # Hiển thị
        self.visible_checkbox = QCheckBox("Hiển thị cấu trúc")
        form_layout.addRow("", self.visible_checkbox)

        # Mô tả
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        form_layout.addRow("Mô tả:", self.description_edit)

        # Nhóm thông tin thể tích
        volume_group = QGroupBox("Thông tin thể tích")
        volume_layout = QFormLayout(volume_group)

        self.volume_label = QLabel("0.0 cm³")
        volume_layout.addRow("Thể tích:", self.volume_label)

        # Margin
        margin_group = QGroupBox("Tạo cấu trúc margin")
        margin_layout = QVBoxLayout(margin_group)

        margin_form = QFormLayout()
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(-10.0, 10.0)
        self.margin_spin.setSingleStep(0.1)
        self.margin_spin.setSuffix(" cm")
        margin_form.addRow("Margin:", self.margin_spin)

        self.create_margin_button = QPushButton("Tạo cấu trúc với margin")

        margin_layout.addLayout(margin_form)
        margin_layout.addWidget(self.create_margin_button)

        # Thêm các widget vào layout chính
        main_layout.addLayout(form_layout)
        main_layout.addWidget(volume_group)
        main_layout.addWidget(margin_group)

        # Thêm các nút
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._save_changes)
        button_box.rejected.connect(self.reject)

        main_layout.addStretch()
        main_layout.addWidget(button_box)

        # Kết nối tín hiệu
        self.create_margin_button.clicked.connect(self._create_margin_structure)

        # Áp dụng theme Eclipse nếu có
        try:
            apply_eclipse_theme_to_widget(self)
        except:
            pass

    def _load_structure_data(self):
        """Tải dữ liệu từ cấu trúc vào các trường nhập liệu."""
        if not self.structure:
            return

        # Tải các thuộc tính cơ bản
        self.name_edit.setText(
            self.structure.name if hasattr(self.structure, "name") else ""
        )

        # Loại cấu trúc
        structure_type = (
            self.structure.structure_type
            if hasattr(self.structure, "structure_type")
            else None
        )
        type_index = 6  # Mặc định là "OTHER"

        if hasattr(structure_type, "__str__"):
            type_str = str(structure_type)
            if "PTV" in type_str:
                type_index = 0
            elif "CTV" in type_str:
                type_index = 1
            elif "GTV" in type_str:
                type_index = 2
            elif "OAR" in type_str:
                type_index = 3
            elif "EXTERNAL" in type_str:
                type_index = 4
            elif "BOLUS" in type_str:
                type_index = 5

        self.type_combo.setCurrentIndex(type_index)

        # Màu sắc
        color = (
            self.structure.color if hasattr(self.structure, "color") else (255, 0, 0)
        )
        self.color_button.set_color(color)

        # Hiển thị
        visible = self.structure.visible if hasattr(self.structure, "visible") else True
        self.visible_checkbox.setChecked(visible)

        # Mô tả
        description = (
            self.structure.description if hasattr(self.structure, "description") else ""
        )
        self.description_edit.setText(description)

        # Thể tích
        volume = self.structure.volume if hasattr(self.structure, "volume") else 0.0
        self.volume_label.setText(f"{volume:.2f} cm³")

    def _save_changes(self):
        """Lưu các thay đổi vào đối tượng cấu trúc."""
        try:
            # Cập nhật các thuộc tính cơ bản
            self.structure.name = self.name_edit.text()

            # Loại cấu trúc
            type_map = {
                0: StructureType.PTV if hasattr(StructureType, "PTV") else "PTV",
                1: StructureType.CTV if hasattr(StructureType, "CTV") else "CTV",
                2: StructureType.GTV if hasattr(StructureType, "GTV") else "GTV",
                3: StructureType.OAR if hasattr(StructureType, "OAR") else "OAR",
                4: StructureType.EXTERNAL
                if hasattr(StructureType, "EXTERNAL")
                else "EXTERNAL",
                5: StructureType.BOLUS if hasattr(StructureType, "BOLUS") else "BOLUS",
                6: StructureType.OTHER if hasattr(StructureType, "OTHER") else "OTHER",
            }
            self.structure.structure_type = type_map[self.type_combo.currentIndex()]

            # Màu sắc
            self.structure.color = self.color_button.get_color()

            # Hiển thị
            self.structure.visible = self.visible_checkbox.isChecked()

            # Mô tả
            self.structure.description = self.description_edit.toPlainText()

            self.accept()

        except Exception as e:
            logging.error(f"Lỗi khi lưu thay đổi cấu trúc: {e}")
            QMessageBox.warning(
                self, "Lỗi", f"Đã xảy ra lỗi khi lưu thay đổi: {str(e)}"
            )

    def _create_margin_structure(self):
        """Tạo cấu trúc mới với margin từ cấu trúc hiện tại."""
        try:
            margin = self.margin_spin.value()

            # Thông báo tính năng chưa được triển khai đầy đủ
            QMessageBox.information(
                self,
                "Thông báo",
                f"Tính năng tạo cấu trúc với margin {margin} cm sẽ được triển khai trong phiên bản tiếp theo.",
            )

        except Exception as e:
            logging.error(f"Lỗi khi tạo cấu trúc margin: {e}")
            QMessageBox.warning(
                self, "Lỗi", f"Đã xảy ra lỗi khi tạo cấu trúc margin: {str(e)}"
            )


# Test code
if __name__ == "__main__" and HAS_PYQT:
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Tạo cấu trúc giả
    class DummyStructure:
        def __init__(self):
            self.name = "PTV"
            self.structure_type = "PTV"
            self.color = (255, 0, 0)
            self.visible = True
            self.description = "This is a test structure for demonstration purposes."
            self.volume = 125.7

    structure = DummyStructure()

    # Tạo dialog
    dialog = StructurePropertiesDialog(structure)
    if dialog.exec_():
        print("Structure properties updated:")
        print(f"Name: {structure.name}")
        print(f"Type: {structure.structure_type}")
        print(f"Color: {structure.color}")
        print(f"Visible: {structure.visible}")
        print(f"Description: {structure.description}")

    sys.exit(app.exec_())
