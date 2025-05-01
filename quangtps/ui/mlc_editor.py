#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp giao diện chỉnh sửa MLC (Multi-Leaf Collimator).

Giao diện này cho phép người dùng thiết kế và chỉnh sửa hình dạng MLC
cho các chùm tia xạ trị, bao gồm cả việc tạo hình dạng từ các cấu trúc và
các hình dạng cơ bản như hình chữ nhật, hình tròn.
"""

import os
import logging
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QRadioButton,
    QFormLayout,
    QDoubleSpinBox,
    QSlider,
    QTabWidget,
    QToolBar,
    QAction,
    QCheckBox,
    QMessageBox,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon

from quangtps.planning.mlc import (
    MLC,
    MLCLeaf,
    MLCSequence,
    MLC_CONFIGURATIONS,
    create_shape_based_mlc,
)
from quangtps.planning.beam_configurator import BeamConfigurator
from quangtps.imaging.structures import Structure
from quangtps.core.config import Config
from quangtps.common.paths import get_icon_path

logger = logging.getLogger(__name__)


class MLCCanvas(FigureCanvas):
    """Widget hiển thị MLC dựa trên Matplotlib."""

    leaf_position_changed = pyqtSignal(int, float)  # (leaf_index, position)

    def __init__(self, parent=None, width=6, height=6, dpi=100):
        """
        Khởi tạo canvas MLC.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        width : int, optional
            Chiều rộng của hình (inch)
        height : int, optional
            Chiều cao của hình (inch)
        dpi : int, optional
            Độ phân giải hình (dots per inch)
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_aspect("equal")

        super().__init__(self.fig)
        self.setParent(parent)

        self.mlc = None
        self.drag_leaf = None
        self.drag_bank = None
        self.field_size = 40.0
        self.show_leaf_numbers = False
        self.selected_leaf = None

        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)
        self.mpl_connect("motion_notify_event", self.on_motion)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def set_mlc(self, mlc):
        """Thiết lập MLC để hiển thị."""
        self.mlc = mlc
        self.update_display()

    def update_display(self):
        """Cập nhật hiển thị của MLC."""
        if self.mlc is None:
            return

        self.axes.clear()

        # Thiết lập giới hạn trục
        half_size = self.field_size / 2
        self.axes.set_xlim(-half_size, half_size)
        self.axes.set_ylim(-half_size, half_size)

        # Tên trục
        self.axes.set_xlabel("X (cm)")
        self.axes.set_ylabel("Y (cm)")

        # Vẽ các lá MLC
        for leaf in self.mlc.leaves:
            y_min, y_max, x_pos, bank_factor = leaf.get_physical_coordinates()
            leaf_width = leaf.width

            # Xác định màu dựa trên ngân hàng lá
            color = "lightblue" if bank_factor > 0 else "lightcoral"

            # Đánh dấu lá được chọn
            if self.selected_leaf == leaf.index:
                color = "yellow" if bank_factor > 0 else "orange"

            # Vẽ hình chữ nhật đại diện cho lá
            if bank_factor > 0:  # Bank A (Left)
                rect = Rectangle(
                    (-half_size, y_min),
                    half_size + x_pos,
                    leaf_width,
                    facecolor=color,
                    edgecolor="black",
                    alpha=0.7,
                )
            else:  # Bank B (Right)
                rect = Rectangle(
                    (x_pos, y_min),
                    half_size - x_pos,
                    leaf_width,
                    facecolor=color,
                    edgecolor="black",
                    alpha=0.7,
                )

            self.axes.add_patch(rect)

            # Thêm số lá nếu được yêu cầu
            if self.show_leaf_numbers and y_max - y_min >= 0.5:
                if bank_factor > 0:
                    self.axes.text(
                        -half_size + 0.5,
                        (y_min + y_max) / 2,
                        str(leaf.index),
                        ha="left",
                        va="center",
                        fontsize=8,
                    )
                else:
                    self.axes.text(
                        half_size - 0.5,
                        (y_min + y_max) / 2,
                        str(leaf.index),
                        ha="right",
                        va="center",
                        fontsize=8,
                    )

        # Vẽ hệ tọa độ
        self.axes.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        self.axes.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

        # Tính toán và hiển thị phần trăm mở của trường
        if self.mlc.leaves:
            field_area = self.field_size * self.field_size
            transmission_map = self.mlc.get_transmission_map(resolution=100)
            open_area = (
                np.sum(transmission_map > 0.5) / transmission_map.size * field_area
            )
            open_percent = (open_area / field_area) * 100

            self.axes.set_title(f"MLC Field - {open_percent:.1f}% Open")

        self.fig.tight_layout()
        self.draw()

    def on_press(self, event):
        """Xử lý sự kiện nhấn chuột."""
        if event.inaxes != self.axes or self.mlc is None:
            return

        # Chuyển đổi tọa độ chuột thành tọa độ MLC
        x, y = event.xdata, event.ydata

        # Kiểm tra xem đã nhấn vào lá nào
        for leaf in self.mlc.leaves:
            y_min, y_max, x_pos, bank_factor = leaf.get_physical_coordinates()

            if y_min <= y <= y_max:
                if (bank_factor > 0 and abs(x_pos - x) < 0.5) or (
                    bank_factor < 0 and abs(x_pos - x) < 0.5
                ):
                    self.drag_leaf = leaf.index
                    self.drag_bank = leaf.bank
                    self.selected_leaf = leaf.index
                    self.update_display()
                    break

    def on_release(self, event):
        """Xử lý sự kiện thả chuột."""
        self.drag_leaf = None
        self.drag_bank = None

    def on_motion(self, event):
        """Xử lý sự kiện di chuyển chuột."""
        if event.inaxes != self.axes or self.drag_leaf is None or self.mlc is None:
            return

        # Chuyển đổi tọa độ chuột thành tọa độ MLC
        x, y = event.xdata, event.ydata

        for leaf in self.mlc.leaves:
            if leaf.index == self.drag_leaf and leaf.bank == self.drag_bank:
                # Tính toán vị trí mới cho lá
                if leaf.bank == "A":  # Bank A (bên trái)
                    # Đảm bảo lá không vượt quá giới hạn
                    new_position = min(
                        max(x, -self.field_size / 2), self.field_size / 2
                    )
                    # Đảm bảo lá không vượt quá lá đối diện
                    paired_leaf = self.mlc.get_leaf(self.drag_leaf, "B")
                    if paired_leaf and new_position >= paired_leaf.position - 0.1:
                        new_position = paired_leaf.position - 0.1
                else:  # Bank B (bên phải)
                    # Đảm bảo lá không vượt quá giới hạn
                    new_position = min(
                        max(x, -self.field_size / 2), self.field_size / 2
                    )
                    # Đảm bảo lá không vượt quá lá đối diện
                    paired_leaf = self.mlc.get_leaf(self.drag_leaf, "A")
                    if paired_leaf and new_position <= paired_leaf.position + 0.1:
                        new_position = paired_leaf.position + 0.1

                # Cập nhật vị trí lá
                leaf.position = new_position
                self.leaf_position_changed.emit(leaf.index, new_position)
                self.update_display()
                break

    def fit_to_structure(self, structure, margin=0.0):
        """
        Tự động điều chỉnh lá MLC để phù hợp với cấu trúc từ góc nhìn beam's eye view.

        Parameters
        ----------
        structure : Structure
            Cấu trúc để tạo hình dạng
        margin : float, optional
            Lề bổ sung (cm) để thêm xung quanh cấu trúc
        """
        if self.mlc is None or structure is None:
            return

        try:
            # Tạo hình dạng MLC từ cấu trúc
            self.mlc = create_shape_based_mlc(
                structure=structure, mlc_type=self.mlc.model_name, margin=margin
            )

            logger.info(
                f"Tạo hình dạng MLC từ cấu trúc {structure.name} với lề {margin} cm"
            )
            self.update_display()

        except Exception as e:
            logger.error(f"Lỗi khi tạo hình dạng MLC từ cấu trúc: {str(e)}")
            QMessageBox.warning(
                self,
                "Lỗi Tạo Hình MLC",
                f"Không thể tạo hình dạng MLC từ cấu trúc: {str(e)}",
            )

    def optimize_leaf_positions(self, target_structure, oar_structures=None):
        """
        Tối ưu hóa vị trí lá MLC để tối đa hóa bao phủ mục tiêu và giảm thiểu
        liều lượng đến các cơ quan nguy cấp.

        Parameters
        ----------
        target_structure : Structure
            Cấu trúc mục tiêu cần bao phủ
        oar_structures : list of Structure, optional
            Danh sách các cơ quan nguy cấp cần bảo vệ
        """
        if self.mlc is None or target_structure is None:
            return

        try:
            from quangtps.optimization.mlc_optimization import optimize_mlc_shape

            # Tạo danh sách cơ quan nguy cấp nếu không được cung cấp
            if oar_structures is None:
                oar_structures = []

            # Tối ưu hóa hình dạng MLC
            optimized_mlc = optimize_mlc_shape(
                original_mlc=self.mlc,
                target=target_structure,
                oars=oar_structures,
                field_size=self.field_size,
            )

            # Cập nhật MLC với kết quả tối ưu hóa
            self.mlc = optimized_mlc
            self.update_display()

            logger.info(
                f"Tối ưu hóa MLC cho mục tiêu {target_structure.name} và {len(oar_structures)} cơ quan nguy cấp"
            )

        except ImportError:
            logger.error("Không thể nhập mô-đun tối ưu hóa MLC")
            QMessageBox.warning(
                self,
                "Tính năng không khả dụng",
                "Mô-đun tối ưu hóa MLC không khả dụng.",
            )
        except Exception as e:
            logger.error(f"Lỗi khi tối ưu hóa vị trí lá MLC: {str(e)}")
            QMessageBox.warning(
                self, "Lỗi tối ưu hóa", f"Không thể tối ưu hóa vị trí lá MLC: {str(e)}"
            )


class MLCEditor(QWidget):
    """
    Widget chỉnh sửa MLC cho kế hoạch xạ trị.
    """

    mlc_changed = pyqtSignal(MLC)  # Phát khi MLC thay đổi

    def __init__(self, parent=None, structures=None):
        """
        Khởi tạo trình soạn thảo MLC.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        structures : list of Structure, optional
            Danh sách các cấu trúc có sẵn cho việc tạo hình MLC
        """
        super().__init__(parent)
        self.mlc = None
        self.structures = structures or []
        self.target_structure = None
        self.oar_structures = []

        self._init_ui()
        self._create_default_mlc()
        self._update_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QHBoxLayout()

        # Panel trái - Điều khiển
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Cấu hình MLC
        config_group = QGroupBox("Cấu hình MLC")
        config_layout = QFormLayout()

        self.mlc_type_combo = QComboBox()
        for config in MLC_CONFIGURATIONS:
            self.mlc_type_combo.addItem(
                f"{config['name']} ({config['num_leaves']} leaves)"
            )
        self.mlc_type_combo.currentIndexChanged.connect(self._on_mlc_type_changed)

        self.field_size_spin = QDoubleSpinBox()
        self.field_size_spin.setRange(1.0, 40.0)
        self.field_size_spin.setSingleStep(1.0)
        self.field_size_spin.setValue(40.0)
        self.field_size_spin.setSuffix(" cm")
        self.field_size_spin.valueChanged.connect(self._update_field_size)

        config_layout.addRow("Loại MLC:", self.mlc_type_combo)
        config_layout.addRow("Kích thước trường:", self.field_size_spin)
        config_group.setLayout(config_layout)

        # Các hình dạng chuẩn
        shapes_group = QGroupBox("Các hình dạng chuẩn")
        shapes_layout = QVBoxLayout()

        self.rect_button = QPushButton("Tạo hình chữ nhật")
        self.rect_button.clicked.connect(self._create_rectangular_field)

        self.circle_button = QPushButton("Tạo hình tròn")
        self.circle_button.clicked.connect(self._create_circular_field)

        self.clear_button = QPushButton("Xóa trường")
        self.clear_button.clicked.connect(self._clear_field)

        self.close_button = QPushButton("Đóng trường")
        self.close_button.clicked.connect(self._close_field)

        shapes_layout.addWidget(self.rect_button)
        shapes_layout.addWidget(self.circle_button)
        shapes_layout.addWidget(self.clear_button)
        shapes_layout.addWidget(self.close_button)
        shapes_group.setLayout(shapes_layout)

        # Nhóm tạo hình dựa trên cấu trúc
        structure_group = QGroupBox("Tạo hình từ cấu trúc")
        structure_layout = QVBoxLayout()

        structure_form = QFormLayout()
        self.structure_combo = QComboBox()
        self.structure_combo.setToolTip("Chọn cấu trúc để tạo hình MLC")

        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(-1.0, 2.0)
        self.margin_spin.setSingleStep(0.1)
        self.margin_spin.setValue(0.3)
        self.margin_spin.setSuffix(" cm")
        self.margin_spin.setToolTip("Lề bổ sung xung quanh cấu trúc")

        structure_form.addRow("Cấu trúc:", self.structure_combo)
        structure_form.addRow("Lề:", self.margin_spin)

        self.fit_structure_button = QPushButton("Tạo hình từ cấu trúc")
        self.fit_structure_button.clicked.connect(self._fit_to_selected_structure)

        structure_layout.addLayout(structure_form)
        structure_layout.addWidget(self.fit_structure_button)
        structure_group.setLayout(structure_layout)

        # Nhóm tối ưu hóa
        optimization_group = QGroupBox("Tối ưu hóa MLC")
        optimization_layout = QVBoxLayout()

        optimization_form = QFormLayout()
        self.target_combo = QComboBox()
        self.target_combo.setToolTip("Chọn cấu trúc mục tiêu để tối ưu hóa")

        optimization_form.addRow("Mục tiêu:", self.target_combo)

        self.oar_list = QTableWidget()
        self.oar_list.setColumnCount(2)
        self.oar_list.setHorizontalHeaderLabels(["Cấu trúc", "Ưu tiên"])
        self.oar_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.oar_list.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.oar_list.setSelectionBehavior(QTableWidget.SelectRows)
        self.oar_list.setSelectionMode(QTableWidget.SingleSelection)

        oar_buttons_layout = QHBoxLayout()
        self.add_oar_button = QPushButton("Thêm")
        self.add_oar_button.clicked.connect(self._add_oar_to_list)
        self.remove_oar_button = QPushButton("Xóa")
        self.remove_oar_button.clicked.connect(self._remove_oar_from_list)

        oar_buttons_layout.addWidget(self.add_oar_button)
        oar_buttons_layout.addWidget(self.remove_oar_button)

        self.optimize_button = QPushButton("Tối ưu hóa MLC")
        self.optimize_button.clicked.connect(self._optimize_mlc)
        self.optimize_button.setToolTip(
            "Tối ưu hóa vị trí lá MLC để bao phủ mục tiêu tốt nhất và tránh cơ quan nguy cấp"
        )

        optimization_layout.addLayout(optimization_form)
        optimization_layout.addWidget(QLabel("Cơ quan nguy cấp:"))
        optimization_layout.addWidget(self.oar_list)
        optimization_layout.addLayout(oar_buttons_layout)
        optimization_layout.addWidget(self.optimize_button)
        optimization_group.setLayout(optimization_layout)

        # Bảng vị trí lá
        leaves_group = QGroupBox("Vị trí lá")
        leaves_layout = QVBoxLayout()

        self.leaf_table = QTableWidget()
        self.leaf_table.setColumnCount(3)
        self.leaf_table.setHorizontalHeaderLabels(["#", "Bank A (cm)", "Bank B (cm)"])
        header = self.leaf_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        self.show_numbers_check = QCheckBox("Hiển thị số thứ tự lá")
        self.show_numbers_check.stateChanged.connect(self._toggle_leaf_numbers)

        leaves_layout.addWidget(self.leaf_table)
        leaves_layout.addWidget(self.show_numbers_check)
        leaves_group.setLayout(leaves_layout)

        # Thêm các nhóm điều khiển vào panel trái
        left_layout.addWidget(config_group)
        left_layout.addWidget(shapes_group)
        left_layout.addWidget(structure_group)
        left_layout.addWidget(optimization_group)
        left_layout.addWidget(leaves_group)
        left_layout.addStretch()
        left_panel.setLayout(left_layout)

        # Panel phải - Hiển thị
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        self.mlc_canvas = MLCCanvas(self)
        self.mlc_canvas.leaf_position_changed.connect(
            self._on_canvas_leaf_position_changed
        )

        right_layout.addWidget(self.mlc_canvas)
        right_panel.setLayout(right_layout)

        # Thêm panels vào layout chính
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Cập nhật danh sách cấu trúc
        self._update_structure_lists()

    def _update_structure_lists(self):
        """Cập nhật danh sách cấu trúc trong các combobox."""
        self.structure_combo.clear()
        self.target_combo.clear()

        for structure in self.structures:
            self.structure_combo.addItem(structure.name, structure)
            self.target_combo.addItem(structure.name, structure)

    def _add_oar_to_list(self):
        """Thêm cơ quan nguy cấp vào danh sách tối ưu hóa."""
        selected_index = self.target_combo.currentIndex()
        if selected_index < 0 or selected_index >= len(self.structures):
            return

        structure = self.structures[selected_index]

        # Kiểm tra xem cấu trúc đã có trong danh sách chưa
        for row in range(self.oar_list.rowCount()):
            if self.oar_list.item(row, 0).data(Qt.UserRole) == structure:
                return

        row = self.oar_list.rowCount()
        self.oar_list.insertRow(row)

        name_item = QTableWidgetItem(structure.name)
        name_item.setData(Qt.UserRole, structure)
        self.oar_list.setItem(row, 0, name_item)

        priority_combo = QComboBox()
        priority_combo.addItems(["Cao", "Trung bình", "Thấp"])
        priority_combo.setCurrentIndex(1)  # Mặc định là trung bình
        self.oar_list.setCellWidget(row, 1, priority_combo)

    def _remove_oar_from_list(self):
        """Xóa cơ quan nguy cấp đã chọn khỏi danh sách."""
        selected_rows = self.oar_list.selectionModel().selectedRows()
        if not selected_rows:
            return

        for row in sorted([index.row() for index in selected_rows], reverse=True):
            self.oar_list.removeRow(row)

    def _fit_to_selected_structure(self):
        """Điều chỉnh MLC để phù hợp với cấu trúc đã chọn."""
        selected_index = self.structure_combo.currentIndex()
        if selected_index < 0 or selected_index >= len(self.structures):
            return

        structure = self.structures[selected_index]
        margin = self.margin_spin.value()

        self.mlc_canvas.fit_to_structure(structure, margin)

        # Cập nhật bảng vị trí lá sau khi thay đổi
        self._update_ui()

        # Phát tín hiệu thay đổi
        self.mlc_changed.emit(self.mlc_canvas.get_mlc())

    def _optimize_mlc(self):
        """Tối ưu hóa vị trí lá MLC dựa trên mục tiêu và cơ quan nguy cấp."""
        # Lấy cấu trúc mục tiêu
        target_index = self.target_combo.currentIndex()
        if target_index < 0 or target_index >= len(self.structures):
            QMessageBox.warning(
                self, "Thiếu mục tiêu", "Vui lòng chọn cấu trúc mục tiêu để tối ưu hóa."
            )
            return

        target = self.structures[target_index]

        # Lấy danh sách cơ quan nguy cấp
        oars = []
        for row in range(self.oar_list.rowCount()):
            structure = self.oar_list.item(row, 0).data(Qt.UserRole)
            priority_combo = self.oar_list.cellWidget(row, 1)
            priority = priority_combo.currentIndex()  # 0: Cao, 1: Trung bình, 2: Thấp

            oars.append({"structure": structure, "priority": priority})

        # Tối ưu hóa MLC
        self.mlc_canvas.optimize_leaf_positions(
            target, [item["structure"] for item in oars]
        )

        # Cập nhật UI
        self._update_ui()

        # Phát tín hiệu thay đổi
        self.mlc_changed.emit(self.mlc_canvas.get_mlc())

    def _update_field_size(self, value):
        """Cập nhật kích thước trường trong canvas."""
        self.mlc_canvas.field_size = value
        self.mlc_canvas.update_display()

    def _create_default_mlc(self):
        """Tạo MLC mặc định."""
        self.mlc = MLC(self.current_mlc_type)
        self._update_ui()
        self.mlc_changed.emit(self.mlc)

    def _update_ui(self):
        """Cập nhật giao diện người dùng với MLC hiện tại."""
        if self.mlc is None:
            return

        # Cập nhật canvas
        self.mlc_canvas.set_mlc(self.mlc)

        # Cập nhật bảng lá
        self.leaf_table.setRowCount(0)
        self.leaf_table.blockSignals(True)

        for leaf in self.mlc.leaves:
            row = self.leaf_table.rowCount()
            self.leaf_table.insertRow(row)

            self.leaf_table.setItem(row, 0, QTableWidgetItem(str(leaf.index)))
            self.leaf_table.setItem(row, 1, QTableWidgetItem(leaf.bank))

            position_item = QTableWidgetItem(f"{leaf.position:.2f}")
            self.leaf_table.setItem(row, 2, position_item)

        self.leaf_table.blockSignals(False)

    def _on_mlc_type_changed(self, index):
        """Xử lý khi loại MLC thay đổi."""
        mlc_type = self.mlc_type_combo.currentData()
        if mlc_type != self.current_mlc_type:
            self.current_mlc_type = mlc_type
            self.mlc = MLC(self.current_mlc_type)
            self._update_ui()
            self.mlc_changed.emit(self.mlc)

    def _create_rectangular_field(self):
        """Tạo trường hình chữ nhật."""
        if self.mlc is None:
            return

        width = self.rect_width_spin.value()
        height = self.rect_height_spin.value()

        # Chuyển đổi từ kích thước sang tọa độ
        x1 = -width / 2
        x2 = width / 2
        y1 = -height / 2
        y2 = height / 2

        # Thiết lập trường hình chữ nhật
        self.mlc.set_rectangular_field(x1, x2, y1, y2)
        self._update_ui()
        self.mlc_changed.emit(self.mlc)

    def _create_circular_field(self):
        """Tạo trường hình tròn."""
        if self.mlc is None:
            return

        radius = self.circle_radius_spin.value()

        # Thiết lập trường hình tròn
        self.mlc.set_circular_field(0, 0, radius)
        self._update_ui()
        self.mlc_changed.emit(self.mlc)

    def _clear_field(self):
        """Mở toàn bộ trường (tất cả các lá đều mở hết mức)."""
        if self.mlc is None:
            return

        for leaf in self.mlc.leaves:
            if leaf.bank == "A":
                self.mlc.set_leaf_position(leaf.index, -20.0)
            else:
                self.mlc.set_leaf_position(leaf.index, 20.0)

        self._update_ui()
        self.mlc_changed.emit(self.mlc)

    def _close_field(self):
        """Đóng toàn bộ trường (tất cả các lá đều ở giữa)."""
        if self.mlc is None:
            return

        for leaf in self.mlc.leaves:
            self.mlc.set_leaf_position(leaf.index, 0.0)

        self._update_ui()
        self.mlc_changed.emit(self.mlc)

    def _on_leaf_position_edited(self, item):
        """Xử lý khi vị trí lá được chỉnh sửa trong bảng."""
        if item.column() != 2 or self.mlc is None:
            return

        row = item.row()
        leaf_index = int(self.leaf_table.item(row, 0).text())

        try:
            position = float(item.text())
            if self.mlc.set_leaf_position(leaf_index, position):
                self.mlc_canvas.update_display()
                self.mlc_changed.emit(self.mlc)
        except ValueError:
            # Khôi phục giá trị cũ
            leaf = self.mlc.get_leaf(leaf_index)
            if leaf:
                item.setText(f"{leaf.position:.2f}")

    def _on_canvas_leaf_position_changed(self, leaf_index, position):
        """Xử lý khi vị trí lá thay đổi từ canvas."""
        # Cập nhật giá trị trong bảng
        for row in range(self.leaf_table.rowCount()):
            if int(self.leaf_table.item(row, 0).text()) == leaf_index:
                self.leaf_table.blockSignals(True)
                self.leaf_table.item(row, 2).setText(f"{position:.2f}")
                self.leaf_table.blockSignals(False)
                break

        self.mlc_changed.emit(self.mlc)

    def _toggle_leaf_numbers(self, state):
        """Bật/tắt hiển thị số lá."""
        self.mlc_canvas.show_leaf_numbers = state == Qt.Checked
        self.mlc_canvas.update_display()

    def set_mlc(self, mlc):
        """Thiết lập MLC từ bên ngoài."""
        self.mlc = mlc

        # Cập nhật combo loại MLC
        index = self.mlc_type_combo.findData(mlc.mlc_type)
        if index >= 0:
            self.mlc_type_combo.setCurrentIndex(index)

        self._update_ui()

    def get_mlc(self):
        """Lấy MLC hiện tại."""
        return self.mlc


if __name__ == "__main__":
    # Chạy giao diện để kiểm tra
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    window = MLCEditor()
    window.setWindowTitle("MLC Editor")
    window.resize(1000, 600)
    window.show()

    sys.exit(app.exec_())
