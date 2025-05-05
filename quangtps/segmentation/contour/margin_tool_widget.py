#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module margin tool widget cho QuangTPS.

Module này cung cấp giao diện người dùng hiện đại cho công cụ tạo margin,
cho phép người dùng áp dụng nhiều loại margin khác nhau cho cấu trúc.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

# Thêm xử lý exception khi import PyQt5
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QComboBox,
        QDoubleSpinBox,
        QGroupBox,
        QFormLayout,
        QRadioButton,
        QButtonGroup,
        QLineEdit,
        QDialog,
        QDialogButtonBox,
        QCheckBox,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QScrollArea,
        QSplitter,
        QFrame,
        QTabWidget,
        QSpinBox,
        QSlider,
        QToolButton,
        QColorDialog,
        QSizePolicy,
        QProgressBar,
    )
    from PyQt5.QtGui import QColor, QIcon, QPixmap, QPainter, QBrush, QPen
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRect, QPoint, QTimer

    PYQT_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import PyQt5: {e}")
    PYQT_AVAILABLE = False

    # Tạo các lớp giả để tránh lỗi cú pháp khi không có PyQt5
    class DummyQtClass:
        """Dummy class to replace Qt classes when PyQt5 is not available."""

        pass

    QWidget = QVBoxLayout = QHBoxLayout = QPushButton = QLabel = QComboBox = (
        DummyQtClass
    )
    QDoubleSpinBox = QGroupBox = QFormLayout = QRadioButton = QButtonGroup = (
        QLineEdit
    ) = DummyQtClass
    QDialog = QDialogButtonBox = QCheckBox = QListWidget = QListWidgetItem = (
        QMessageBox
    ) = DummyQtClass
    QScrollArea = QSplitter = QFrame = QTabWidget = QSpinBox = QSlider = QToolButton = (
        DummyQtClass
    )
    QColorDialog = QSizePolicy = DummyQtClass
    QColor = QIcon = QPixmap = QPainter = QBrush = QPen = DummyQtClass
    Qt = QSize = QRect = QPoint = QTimer = DummyQtClass

    class pyqtSignal:
        """Dummy signal class when PyQt5 is not available."""

        def __init__(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass


try:
    from quangtps.segmentation.contour.margin import MarginType, MarginTool
    from quangtps.segmentation.structures.structure import Structure, StructureType
    from quangtps.segmentation.structures.structure_set import StructureSet
except ImportError as e:
    logging.error(f"Không thể import các module cần thiết: {e}")

    # Tạo các lớp giả
    class MarginType:
        UNIFORM = "UNIFORM"
        ANISOTROPIC = "ANISOTROPIC"
        RING = "RING"
        SURFACE = "SURFACE"

    class MarginTool:
        def margin_by_type(self, *args, **kwargs):
            return []

    class Structure:
        pass

    class StructureType:
        pass

    class StructureSet:
        pass


logger = logging.getLogger(__name__)


class MarginStructureItem:
    """Đại diện cho một mục cấu trúc trong danh sách kết cấu."""

    def __init__(self, structure, selected=False):
        """Khởi tạo một mục cấu trúc."""
        self.structure = structure
        self.selected = selected

        # Các thuộc tính thêm
        self.name = structure.name if hasattr(structure, "name") else "Unknown"
        self.id = structure.id if hasattr(structure, "id") else "unknown_id"
        self.color = structure.color if hasattr(structure, "color") else (255, 0, 0)
        self.type = structure.type if hasattr(structure, "type") else None
        self.volume = None

        # Tính thể tích nếu có dữ liệu
        if hasattr(structure, "get_volume"):
            self.volume = structure.get_volume()

    def __str__(self):
        """Biểu diễn chuỗi của mục cấu trúc."""
        return self.name


class MarginToolWidget(QWidget):
    """Widget để tạo và áp dụng margin cho cấu trúc với giao diện hiện đại."""

    # Signals
    marginApplied = pyqtSignal(object, object)  # Structure cũ, Structure mới

    def __init__(self, parent=None):
        """Khởi tạo widget công cụ margin."""
        super().__init__(parent)

        self.parent = parent
        self.structure_set = None
        self.selected_structure = None
        self.margin_tool = MarginTool()
        self.pixel_spacing = (1.0, 1.0)  # Mặc định spacing 1mm

        # Lưu trữ tham số đã lưu cho mỗi loại margin
        self.saved_parameters = {
            MarginType.UNIFORM: {"margin_mm": 5.0},
            MarginType.ANISOTROPIC: {
                "margins_mm": {
                    "ANTERIOR": 5.0,
                    "POSTERIOR": 5.0,
                    "LEFT": 5.0,
                    "RIGHT": 5.0,
                }
            },
            MarginType.RING: {
                "inner_margin_mm": 0.0,
                "outer_margin_mm": 5.0,
            },
            MarginType.SURFACE: {"thickness_mm": 3.0},
        }

        self._init_ui()
        self.setup_connections()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Tiêu đề
        title_label = QLabel("Công cụ tạo Margin")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0078D7;")
        main_layout.addWidget(title_label)

        # Chia giao diện thành 2 phần: trái (danh sách cấu trúc) và phải (tham số margin)
        splitter = QSplitter(Qt.Horizontal)

        # Phần trái: Danh sách cấu trúc
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel_layout = QVBoxLayout(left_panel)

        # Tiêu đề danh sách cấu trúc
        structure_list_label = QLabel("Cấu trúc:")
        structure_list_label.setStyleSheet("font-weight: bold;")
        left_panel_layout.addWidget(structure_list_label)

        # Danh sách cấu trúc
        self.structure_list = QListWidget()
        self.structure_list.setMinimumWidth(200)
        self.structure_list.setSelectionMode(QListWidget.SingleSelection)
        left_panel_layout.addWidget(self.structure_list)

        splitter.addWidget(left_panel)

        # Phần phải: Tham số margin
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel_layout = QVBoxLayout(right_panel)

        # Loại margin
        margin_type_group = QGroupBox("Loại Margin")
        margin_type_layout = QVBoxLayout()

        self.margin_type_combo = QComboBox()
        self.margin_type_combo.addItem("Đồng đều (Uniform)", MarginType.UNIFORM)
        self.margin_type_combo.addItem(
            "Không đồng đều (Anisotropic)", MarginType.ANISOTROPIC
        )
        self.margin_type_combo.addItem("Vòng (Ring)", MarginType.RING)
        self.margin_type_combo.addItem("Bề mặt (Surface)", MarginType.SURFACE)

        margin_type_layout.addWidget(self.margin_type_combo)
        margin_type_group.setLayout(margin_type_layout)
        right_panel_layout.addWidget(margin_type_group)

        # Tham số margin
        self.parameter_container = QWidget()
        self.parameter_layout = QVBoxLayout(self.parameter_container)

        # 1. Uniform parameters
        self.uniform_parameters = QGroupBox("Tham số đồng đều")
        uniform_layout = QFormLayout()

        self.uniform_margin = QDoubleSpinBox()
        self.uniform_margin.setRange(-50, 50)
        self.uniform_margin.setSingleStep(0.5)
        self.uniform_margin.setValue(5.0)
        self.uniform_margin.setSuffix(" mm")

        uniform_layout.addRow("Margin:", self.uniform_margin)
        self.uniform_parameters.setLayout(uniform_layout)
        self.parameter_layout.addWidget(self.uniform_parameters)

        # 2. Anisotropic parameters
        self.anisotropic_parameters = QGroupBox("Tham số không đồng đều")
        aniso_layout = QFormLayout()

        self.aniso_anterior = QDoubleSpinBox()
        self.aniso_anterior.setRange(-50, 50)
        self.aniso_anterior.setSingleStep(0.5)
        self.aniso_anterior.setValue(5.0)
        self.aniso_anterior.setSuffix(" mm")
        aniso_layout.addRow("Phía trước (Anterior):", self.aniso_anterior)

        self.aniso_posterior = QDoubleSpinBox()
        self.aniso_posterior.setRange(-50, 50)
        self.aniso_posterior.setSingleStep(0.5)
        self.aniso_posterior.setValue(5.0)
        self.aniso_posterior.setSuffix(" mm")
        aniso_layout.addRow("Phía sau (Posterior):", self.aniso_posterior)

        self.aniso_left = QDoubleSpinBox()
        self.aniso_left.setRange(-50, 50)
        self.aniso_left.setSingleStep(0.5)
        self.aniso_left.setValue(5.0)
        self.aniso_left.setSuffix(" mm")
        aniso_layout.addRow("Bên trái (Left):", self.aniso_left)

        self.aniso_right = QDoubleSpinBox()
        self.aniso_right.setRange(-50, 50)
        self.aniso_right.setSingleStep(0.5)
        self.aniso_right.setValue(5.0)
        self.aniso_right.setSuffix(" mm")
        aniso_layout.addRow("Bên phải (Right):", self.aniso_right)

        self.anisotropic_parameters.setLayout(aniso_layout)
        self.parameter_layout.addWidget(self.anisotropic_parameters)

        # 3. Ring parameters
        self.ring_parameters = QGroupBox("Tham số vòng")
        ring_layout = QFormLayout()

        self.inner_margin = QDoubleSpinBox()
        self.inner_margin.setRange(-50, 50)
        self.inner_margin.setSingleStep(0.5)
        self.inner_margin.setValue(0.0)
        self.inner_margin.setSuffix(" mm")
        ring_layout.addRow("Margin trong:", self.inner_margin)

        self.outer_margin = QDoubleSpinBox()
        self.outer_margin.setRange(0, 50)
        self.outer_margin.setSingleStep(0.5)
        self.outer_margin.setValue(5.0)
        self.outer_margin.setSuffix(" mm")
        ring_layout.addRow("Margin ngoài:", self.outer_margin)

        self.ring_parameters.setLayout(ring_layout)
        self.parameter_layout.addWidget(self.ring_parameters)

        # 4. Surface parameters
        self.surface_parameters = QGroupBox("Tham số bề mặt")
        surface_layout = QFormLayout()

        self.thickness = QDoubleSpinBox()
        self.thickness.setRange(0.1, 20)
        self.thickness.setSingleStep(0.1)
        self.thickness.setValue(3.0)
        self.thickness.setSuffix(" mm")
        surface_layout.addRow("Độ dày:", self.thickness)

        self.surface_parameters.setLayout(surface_layout)
        self.parameter_layout.addWidget(self.surface_parameters)

        right_panel_layout.addWidget(self.parameter_container)

        # Tùy chọn đầu ra
        output_group = QGroupBox("Tùy chọn đầu ra")
        output_layout = QVBoxLayout()

        self.create_new_rb = QRadioButton("Tạo cấu trúc mới")
        self.create_new_rb.setChecked(True)
        self.replace_rb = QRadioButton("Thay thế cấu trúc hiện có")

        output_layout.addWidget(self.create_new_rb)
        output_layout.addWidget(self.replace_rb)

        name_layout = QFormLayout()
        self.new_name = QLineEdit()
        self.new_name.setText("")  # Sẽ được cập nhật khi chọn cấu trúc
        name_layout.addRow("Tên cấu trúc mới:", self.new_name)
        output_layout.addLayout(name_layout)

        output_group.setLayout(output_layout)
        right_panel_layout.addWidget(output_group)

        # Nút thao tác
        buttons_layout = QHBoxLayout()

        self.preview_button = QPushButton("Xem trước")
        self.preview_button.setIcon(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "preview.png"))
        )
        buttons_layout.addWidget(self.preview_button)

        self.apply_button = QPushButton("Áp dụng")
        self.apply_button.setIcon(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "apply.png"))
        )
        buttons_layout.addWidget(self.apply_button)

        right_panel_layout.addLayout(buttons_layout)

        # Thêm stretch để đẩy các widget lên trên
        right_panel_layout.addStretch(1)

        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)

        # Ẩn tất cả các nhóm tham số
        self.uniform_parameters.hide()
        self.anisotropic_parameters.hide()
        self.ring_parameters.hide()
        self.surface_parameters.hide()

        # Hiển thị nhóm tham số mặc định (uniform)
        self.uniform_parameters.show()

        # Áp dụng stylesheet
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #f0f0f0, stop: 1 #e0e0e0);
            }

            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
            }

            QPushButton:hover {
                background-color: #005A9E;
            }

            QPushButton:pressed {
                background-color: #004578;
            }

            QListWidget {
                border: 1px solid #cccccc;
                border-radius: 3px;
            }

            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #f0f0f0;
            }

            QListWidget::item:selected {
                background-color: #0078D7;
                color: white;
            }

            QComboBox {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 3px 5px;
            }

            QDoubleSpinBox, QLineEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 3px 5px;
            }
        """)

    def setup_connections(self):
        """Thiết lập các kết nối signal/slot."""
        # Kết nối thay đổi loại margin
        self.margin_type_combo.currentIndexChanged.connect(self.on_margin_type_changed)

        # Kết nối thay đổi tùy chọn đầu ra
        self.create_new_rb.toggled.connect(self.on_output_option_changed)

        # Kết nối thay đổi cấu trúc được chọn
        self.structure_list.currentItemChanged.connect(
            self.on_structure_selection_changed
        )

        # Kết nối nút xem trước và áp dụng
        self.preview_button.clicked.connect(self.preview_margin)
        self.apply_button.clicked.connect(self.apply_margin)

        # Kết nối các điều khiển thay đổi tham số
        self.uniform_margin.valueChanged.connect(self.save_margin_parameters)
        self.aniso_anterior.valueChanged.connect(self.save_margin_parameters)
        self.aniso_posterior.valueChanged.connect(self.save_margin_parameters)
        self.aniso_left.valueChanged.connect(self.save_margin_parameters)
        self.aniso_right.valueChanged.connect(self.save_margin_parameters)
        self.inner_margin.valueChanged.connect(self.save_margin_parameters)
        self.outer_margin.valueChanged.connect(self.save_margin_parameters)
        self.thickness.valueChanged.connect(self.save_margin_parameters)

    def set_structure_set(self, structure_set, pixel_spacing=(1.0, 1.0)):
        """
        Đặt tập hợp cấu trúc và spacing pixel.

        Parameters:
            structure_set: Tập hợp cấu trúc
            pixel_spacing: Khoảng cách giữa các pixel theo x và y (mm)
        """
        self.structure_set = structure_set
        self.pixel_spacing = pixel_spacing

        # Xóa danh sách cấu trúc
        self.structure_list.clear()

        # Cập nhật danh sách cấu trúc
        if structure_set and hasattr(structure_set, "structures"):
            for structure in structure_set.structures:
                self.add_structure_to_list(structure)

        # Cập nhật trạng thái các nút
        has_structures = self.structure_list.count() > 0
        self.apply_button.setEnabled(has_structures)
        self.preview_button.setEnabled(has_structures)

        # Chọn cấu trúc đầu tiên nếu có
        if has_structures:
            self.structure_list.setCurrentRow(0)

    def add_structure_to_list(self, structure):
        """
        Thêm cấu trúc vào danh sách.

        Parameters:
            structure: Cấu trúc cần thêm
        """
        if not structure:
            return

        # Tạo item cho danh sách
        item = QListWidgetItem(structure.name)

        # Đặt biểu tượng màu
        if hasattr(structure, "color"):
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(*structure.color))
            item.setIcon(QIcon(pixmap))

        # Lưu trữ reference đến cấu trúc
        item.setData(Qt.UserRole, structure)

        # Thêm vào danh sách
        self.structure_list.addItem(item)

    def on_structure_selection_changed(self, current, previous):
        """
        Xử lý khi lựa chọn cấu trúc thay đổi.

        Parameters:
            current: Item hiện tại
            previous: Item trước đó
        """
        if not current:
            self.selected_structure = None
            self.apply_button.setEnabled(False)
            self.preview_button.setEnabled(False)
            return

        # Lấy cấu trúc từ item
        self.selected_structure = current.data(Qt.UserRole)

        # Cập nhật tên cấu trúc mới mặc định
        default_name = f"{self.selected_structure.name}_margin"
        self.new_name.setText(default_name)

        # Kích hoạt các nút
        self.apply_button.setEnabled(True)
        self.preview_button.setEnabled(True)

    def on_margin_type_changed(self, index):
        """
        Xử lý khi loại margin thay đổi.

        Parameters:
            index: Chỉ số của loại margin được chọn
        """
        # Ẩn tất cả các nhóm tham số
        self.uniform_parameters.hide()
        self.anisotropic_parameters.hide()
        self.ring_parameters.hide()
        self.surface_parameters.hide()

        # Hiển thị nhóm tham số tương ứng
        margin_type = self.margin_type_combo.currentData()
        self.update_margin_parameters()

        if margin_type == MarginType.UNIFORM:
            self.uniform_parameters.show()
        elif margin_type == MarginType.ANISOTROPIC:
            self.anisotropic_parameters.show()
        elif margin_type == MarginType.RING:
            self.ring_parameters.show()
        elif margin_type == MarginType.SURFACE:
            self.surface_parameters.show()

    def on_output_option_changed(self, checked):
        """
        Xử lý khi tùy chọn đầu ra thay đổi.

        Parameters:
            checked: Trạng thái của radio button
        """
        # Bật/tắt ô nhập tên mới dựa trên tùy chọn
        self.new_name.setEnabled(self.create_new_rb.isChecked())

    def update_margin_parameters(self):
        """Cập nhật giá trị các tham số margin từ tham số đã lưu."""
        margin_type = self.margin_type_combo.currentData()

        if margin_type == MarginType.UNIFORM:
            self.uniform_margin.setValue(
                self.saved_parameters[MarginType.UNIFORM].get("margin_mm", 5.0)
            )

        elif margin_type == MarginType.ANISOTROPIC:
            margins = self.saved_parameters[MarginType.ANISOTROPIC].get(
                "margins_mm", {}
            )
            self.aniso_anterior.setValue(margins.get("ANTERIOR", 5.0))
            self.aniso_posterior.setValue(margins.get("POSTERIOR", 5.0))
            self.aniso_left.setValue(margins.get("LEFT", 5.0))
            self.aniso_right.setValue(margins.get("RIGHT", 5.0))

        elif margin_type == MarginType.RING:
            self.inner_margin.setValue(
                self.saved_parameters[MarginType.RING].get("inner_margin_mm", 0.0)
            )
            self.outer_margin.setValue(
                self.saved_parameters[MarginType.RING].get("outer_margin_mm", 5.0)
            )

        elif margin_type == MarginType.SURFACE:
            self.thickness.setValue(
                self.saved_parameters[MarginType.SURFACE].get("thickness_mm", 3.0)
            )

    def save_margin_parameters(self):
        """Lưu các tham số margin hiện tại."""
        margin_type = self.margin_type_combo.currentData()

        if margin_type == MarginType.UNIFORM:
            self.saved_parameters[MarginType.UNIFORM]["margin_mm"] = (
                self.uniform_margin.value()
            )

        elif margin_type == MarginType.ANISOTROPIC:
            margins = {
                "ANTERIOR": self.aniso_anterior.value(),
                "POSTERIOR": self.aniso_posterior.value(),
                "LEFT": self.aniso_left.value(),
                "RIGHT": self.aniso_right.value(),
            }
            self.saved_parameters[MarginType.ANISOTROPIC]["margins_mm"] = margins

        elif margin_type == MarginType.RING:
            self.saved_parameters[MarginType.RING]["inner_margin_mm"] = (
                self.inner_margin.value()
            )
            self.saved_parameters[MarginType.RING]["outer_margin_mm"] = (
                self.outer_margin.value()
            )

        elif margin_type == MarginType.SURFACE:
            self.saved_parameters[MarginType.SURFACE]["thickness_mm"] = (
                self.thickness.value()
            )

    def get_margin_parameters(self):
        """
        Lấy tham số margin hiện tại.

        Returns:
            Tuple chứa (margin_type, margin_params)
        """
        margin_type = self.margin_type_combo.currentData()
        return (margin_type, self.saved_parameters[margin_type])

    def apply_margin(self):
        """Áp dụng margin cho cấu trúc được chọn."""
        if not self.selected_structure or not self.structure_set:
            return

        try:
            # Lấy thông tin margin
            margin_type, margin_params = self.get_margin_parameters()
            create_new = self.create_new_rb.isChecked()
            new_structure_name = (
                self.new_name.text() if create_new else self.selected_structure.name
            )

            # Kiểm tra tên hợp lệ
            if create_new and not new_structure_name:
                QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên cho cấu trúc mới.")
                return

            # Kiểm tra tên trùng
            if create_new and hasattr(self.structure_set, "structures"):
                for structure in self.structure_set.structures:
                    if (
                        hasattr(structure, "name")
                        and structure.name == new_structure_name
                    ):
                        result = QMessageBox.question(
                            self,
                            "Cấu trúc đã tồn tại",
                            f"Cấu trúc '{new_structure_name}' đã tồn tại. Bạn có muốn ghi đè không?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No,
                        )
                        if result != QMessageBox.Yes:
                            return

            # Lấy cấu trúc nguồn
            source_structure = self.selected_structure
            target_structure = None

            # Áp dụng margin cho mỗi contour trong cấu trúc
            if create_new:
                # Tạo cấu trúc mới
                if hasattr(self.structure_set, "create_structure"):
                    # Nếu có phương thức tạo cấu trúc
                    target_structure = self.structure_set.create_structure(
                        name=new_structure_name,
                        structure_type=source_structure.type
                        if hasattr(source_structure, "type")
                        else None,
                        color=source_structure.color
                        if hasattr(source_structure, "color")
                        else (255, 0, 0),
                    )
                else:
                    # Tạo cấu trúc theo cách thông thường
                    try:
                        # Thử tạo structure với constructor khác nhau
                        if (
                            hasattr(Structure, "__init__")
                            and Structure.__init__.__code__.co_argcount >= 3
                        ):
                            target_structure = Structure(
                                name=new_structure_name,
                                structure_type=source_structure.type
                                if hasattr(source_structure, "type")
                                else None,
                            )
                        else:
                            target_structure = Structure()
                            if hasattr(target_structure, "name"):
                                target_structure.name = new_structure_name
                    except Exception as e:
                        logger.error(f"Lỗi khi tạo cấu trúc mới: {e}")
                        QMessageBox.critical(
                            self, "Lỗi", f"Không thể tạo cấu trúc mới: {e}"
                        )
                        return

                # Sao chép các thuộc tính từ cấu trúc nguồn
                if target_structure:
                    if hasattr(source_structure, "color") and hasattr(
                        target_structure, "color"
                    ):
                        target_structure.color = source_structure.color
                    if hasattr(source_structure, "type") and hasattr(
                        target_structure, "type"
                    ):
                        target_structure.type = source_structure.type
                    if hasattr(source_structure, "priority") and hasattr(
                        target_structure, "priority"
                    ):
                        target_structure.priority = source_structure.priority

                    # Thêm cấu trúc mới vào tập hợp cấu trúc
                    if hasattr(self.structure_set, "add_structure"):
                        self.structure_set.add_structure(target_structure)
                    elif hasattr(self.structure_set, "structures"):
                        self.structure_set.structures.append(target_structure)
            else:
                # Sử dụng cấu trúc hiện có
                target_structure = source_structure

            # Áp dụng margin cho từng contour
            if hasattr(source_structure, "contours"):
                new_contours = {}

                for slice_num, contours in source_structure.contours.items():
                    if contours:
                        # Áp dụng margin với loại tương ứng
                        margin_contours = self.margin_tool.margin_by_type(
                            contours, margin_type, margin_params, self.pixel_spacing
                        )
                        if margin_contours:
                            new_contours[slice_num] = margin_contours

                # Cập nhật contour cho cấu trúc mới
                if hasattr(target_structure, "set_contours"):
                    for slice_num, contours in new_contours.items():
                        target_structure.set_contours(slice_num, contours)
                elif hasattr(target_structure, "contours"):
                    target_structure.contours = new_contours

            # Nếu thành công, emit signal và thông báo
            self.marginApplied.emit(source_structure, target_structure)

            # Thêm cấu trúc mới vào danh sách nếu cần
            if create_new and target_structure:
                self.add_structure_to_list(target_structure)

                # Chọn cấu trúc mới
                for i in range(self.structure_list.count()):
                    item = self.structure_list.item(i)
                    if item.data(Qt.UserRole) == target_structure:
                        self.structure_list.setCurrentItem(item)
                        break

            QMessageBox.information(
                self,
                "Hoàn tất",
                f"Đã áp dụng margin thành công cho cấu trúc '{source_structure.name}'!",
            )

        except Exception as e:
            # Hiển thị thông báo lỗi
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi áp dụng margin: {str(e)}")
            logger.error(f"Lỗi khi áp dụng margin: {str(e)}")

    def preview_margin(self):
        """Xem trước kết quả margin."""
        if not self.selected_structure:
            return

        try:
            # Lấy thông tin margin
            margin_type, margin_params = self.get_margin_parameters()

            # Hiển thị hộp thoại xem trước đơn giản
            QMessageBox.information(
                self,
                "Xem trước Margin",
                f"Loại margin: {margin_type.value}\n"
                f"Tham số: {margin_params}\n\n"
                "Tính năng xem trước hình ảnh sẽ được phát triển trong phiên bản sau.",
            )

        except Exception as e:
            # Hiển thị thông báo lỗi
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi xem trước margin: {str(e)}")
            logger.error(f"Lỗi khi xem trước margin: {str(e)}")

    def set_pixel_spacing(self, spacing):
        """
        Đặt khoảng cách pixel.

        Parameters:
            spacing: Tuple (dx, dy) khoảng cách pixel theo mm
        """
        self.pixel_spacing = spacing


def show_margin_tool_dialog(parent=None, structure_set=None, pixel_spacing=(1.0, 1.0)):
    """
    Hiển thị hộp thoại công cụ margin.

    Parameters:
        parent: Widget cha
        structure_set: Tập hợp cấu trúc
        pixel_spacing: Khoảng cách pixel

    Returns:
        MarginToolWidget instance
    """
    if not PYQT_AVAILABLE:
        logger.error("Không thể hiển thị hộp thoại margin: PyQt5 không khả dụng")
        return None

    dialog = QDialog(parent)
    dialog.setWindowTitle("Công cụ Margin")
    dialog.setMinimumSize(800, 600)

    layout = QVBoxLayout(dialog)

    # Tạo widget margin
    margin_widget = MarginToolWidget(dialog)
    if structure_set:
        margin_widget.set_structure_set(structure_set, pixel_spacing)

    layout.addWidget(margin_widget)

    # Buttons
    button_box = QDialogButtonBox(QDialogButtonBox.Close)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)

    # Hiển thị hộp thoại
    dialog.show()

    return margin_widget
