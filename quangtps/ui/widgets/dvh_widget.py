#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module DVHWidget cho QuangTPS.

Module này cung cấp giao diện người dùng để hiển thị biểu đồ liều-thể tích (DVH),
cho phép so sánh nhiều kế hoạch xạ trị và phân tích tham số liều.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import enum
from functools import partial

# Thêm xử lý exception khi import PyQt5 và matplotlib
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QComboBox,
        QCheckBox,
        QGroupBox,
        QFormLayout,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QTabWidget,
        QRadioButton,
        QButtonGroup,
        QMenu,
        QAction,
        QToolBar,
        QFileDialog,
        QMessageBox,
        QScrollArea,
        QFrame,
        QSizePolicy,
        QColorDialog,
        QApplication,
        QToolButton,
        QSlider,
        QDialog,
        QDialogButtonBox,
    )
    from PyQt5.QtGui import QIcon, QColor, QPixmap, QPainter, QPen, QBrush, QFontMetrics
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPointF

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
    QCheckBox = QGroupBox = QFormLayout = QSplitter = QTableWidget = (
        QTableWidgetItem
    ) = DummyQtClass
    QHeaderView = QTabWidget = QRadioButton = QButtonGroup = QMenu = QAction = (
        QToolBar
    ) = DummyQtClass
    QFileDialog = QMessageBox = QScrollArea = QFrame = QSizePolicy = QColorDialog = (
        DummyQtClass
    )
    QApplication = QToolButton = QSlider = DummyQtClass
    QIcon = QColor = QPixmap = QPainter = QPen = QBrush = QFontMetrics = DummyQtClass
    Qt = QSize = QPointF = DummyQtClass

    class pyqtSignal:
        """Dummy signal class when PyQt5 is not available."""

        def __init__(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass


try:
    import matplotlib

    # Đặt backend cho matplotlib
    if PYQT_AVAILABLE:
        matplotlib.use("Qt5Agg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import matplotlib: {e}")
    MATPLOTLIB_AVAILABLE = False

    # Tạo các lớp giả
    class Figure:
        def add_subplot(self, *args, **kwargs):
            return DummySubplot()

    class DummySubplot:
        def plot(self, *args, **kwargs):
            pass

        def set_xlabel(self, *args, **kwargs):
            pass

        def set_ylabel(self, *args, **kwargs):
            pass

        def set_title(self, *args, **kwargs):
            pass

        def legend(self, *args, **kwargs):
            pass

        def grid(self, *args, **kwargs):
            pass

        def set_xlim(self, *args, **kwargs):
            pass

        def set_ylim(self, *args, **kwargs):
            pass

    class FigureCanvas:
        def __init__(self, figure):
            self.figure = figure

        def draw(self):
            pass

    class NavigationToolbar:
        def __init__(self, canvas, parent):
            self.canvas = canvas
            self.parent = parent


try:
    # Import các module từ QuangTPS
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure import Structure, StructureType
    from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator, DVHType
    from quangtps.planning.plan import Plan
except ImportError as e:
    logging.error(f"Không thể import các module QuangTPS: {e}")

    # Tạo các lớp giả
    class DoseGrid:
        def get_dose_at_point(self, *args, **kwargs):
            return 0.0

        def get_max_dose(self):
            return 0.0

    class StructureSet:
        def __init__(self):
            self.structures = []

    class Structure:
        def __init__(self, name="", structure_type=None, color=(255, 0, 0)):
            self.name = name
            self.type = structure_type
            self.color = color
            self.contours = {}

        def get_volume(self):
            return 0.0

    class StructureType(enum.Enum):
        PTV = "PTV"
        OAR = "OAR"
        OTHER = "OTHER"

    class DVHCalculator:
        def __init__(self):
            pass

        def calculate_dvh(self, structure, dose_grid, dvh_type, resolution=0.1):
            return {
                "dose": np.linspace(0, 70, 701),
                "volume": np.exp(-np.linspace(0, 7, 701)),
            }

    class DVHType(enum.Enum):
        CUMULATIVE = "CUMULATIVE"
        DIFFERENTIAL = "DIFFERENTIAL"

    class Plan:
        def __init__(self):
            self.dose_grid = DoseGrid()
            self.structure_set = StructureSet()
            self.name = "Default Plan"


# Thiết lập logging
logger = logging.getLogger(__name__)


class DVHWidget(QWidget):
    """Widget hiển thị biểu đồ liều-thể tích (DVH) và thống kê liều."""

    # Signals
    planChanged = pyqtSignal(object)  # Khi kế hoạch thay đổi
    structureSelectionChanged = pyqtSignal(list)  # Khi lựa chọn cấu trúc thay đổi

    def __init__(self, parent=None):
        """Khởi tạo widget DVH."""
        super().__init__(parent)

        self.parent = parent

        # Các tham số
        self.plans = {}  # Dict[name, Plan]
        self.current_plan_name = None
        self.selected_structures = set()  # Tên các cấu trúc được chọn
        self.display_volumes = (
            True  # Hiển thị thể tích tương đối (%) hoặc tuyệt đối (cc)
        )
        self.dvh_type = DVHType.CUMULATIVE  # Loại DVH (tích lũy hoặc vi phân)
        self.normalization_value = 100.0  # Giá trị chuẩn hóa (%)
        self.dose_unit = "Gy"  # Đơn vị liều
        self.calculated_dvhs = {}  # Dict[(plan_name, structure_name), dvh_data]
        self.structure_colors = {}  # Dict[structure_name, color]
        self.plan_line_styles = {  # Style cho các kế hoạch khác nhau
            0: "-",  # Solid
            1: "--",  # Dashed
            2: "-.",  # Dash-dot
            3: ":",  # Dotted
            4: (0, (3, 1, 1, 1)),  # Dash-dot-dot
            5: (0, (5, 1)),  # Dense dashed
        }
        self.dvh_calculator = DVHCalculator()

        # Các thành phần UI
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout tổng thể
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Thanh công cụ
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Nút xuất và in
        export_action = QAction(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "export.png")),
            "Xuất",
            self,
        )
        export_action.triggered.connect(self.export_dvh)
        toolbar.addAction(export_action)

        print_action = QAction(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "print.png")),
            "In",
            self,
        )
        print_action.triggered.connect(self.print_dvh)
        toolbar.addAction(print_action)

        toolbar.addSeparator()

        # Loại DVH
        dvh_type_label = QLabel("Loại DVH:")
        toolbar.addWidget(dvh_type_label)

        self.dvh_type_combo = QComboBox()
        self.dvh_type_combo.addItem("Tích lũy", DVHType.CUMULATIVE)
        self.dvh_type_combo.addItem("Vi phân", DVHType.DIFFERENTIAL)
        toolbar.addWidget(self.dvh_type_combo)

        toolbar.addSeparator()

        # Hiển thị thể tích
        volume_label = QLabel("Thể tích:")
        toolbar.addWidget(volume_label)

        self.volume_combo = QComboBox()
        self.volume_combo.addItem("Tương đối (%)", True)
        self.volume_combo.addItem("Tuyệt đối (cc)", False)
        toolbar.addWidget(self.volume_combo)

        toolbar.addSeparator()

        # Chuẩn hóa liều
        norm_label = QLabel("Chuẩn hóa:")
        toolbar.addWidget(norm_label)

        self.norm_combo = QComboBox()
        self.norm_combo.setEditable(True)
        self.norm_combo.addItem("100%")
        self.norm_combo.addItem("95%")
        self.norm_combo.addItem("90%")
        self.norm_combo.addItem("80%")
        toolbar.addWidget(self.norm_combo)

        # Nút làm mới
        refresh_action = QAction(
            QIcon(os.path.join(os.path.dirname(__file__), "icons", "refresh.png")),
            "Làm mới",
            self,
        )
        refresh_action.triggered.connect(self.refresh_dvh)
        toolbar.addAction(refresh_action)

        main_layout.addWidget(toolbar)

        # Tạo splitter chính (chia màn hình thành 2 phần)
        main_splitter = QSplitter(Qt.Horizontal)

        # Phần trái: Danh sách kế hoạch và cấu trúc
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Danh sách kế hoạch
        plan_group = QGroupBox("Kế hoạch")
        plan_layout = QVBoxLayout()

        self.plan_combo = QComboBox()
        self.plan_combo.setMaximumWidth(200)
        plan_layout.addWidget(self.plan_combo)

        plan_group.setLayout(plan_layout)
        left_layout.addWidget(plan_group)

        # Danh sách cấu trúc
        structure_group = QGroupBox("Cấu trúc")
        structure_layout = QVBoxLayout()

        self.ptv_group = QGroupBox("PTV")
        self.ptv_layout = QVBoxLayout()
        self.ptv_group.setLayout(self.ptv_layout)
        structure_layout.addWidget(self.ptv_group)

        self.oar_group = QGroupBox("OAR")
        self.oar_layout = QVBoxLayout()
        self.oar_group.setLayout(self.oar_layout)
        structure_layout.addWidget(self.oar_group)

        self.other_group = QGroupBox("Khác")
        self.other_layout = QVBoxLayout()
        self.other_group.setLayout(self.other_layout)
        structure_layout.addWidget(self.other_group)

        # Nút chọn tất cả / bỏ chọn tất cả
        buttons_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Chọn tất cả")
        self.select_all_button.clicked.connect(self.select_all_structures)
        buttons_layout.addWidget(self.select_all_button)

        self.deselect_all_button = QPushButton("Bỏ chọn tất cả")
        self.deselect_all_button.clicked.connect(self.deselect_all_structures)
        buttons_layout.addWidget(self.deselect_all_button)

        structure_layout.addLayout(buttons_layout)

        structure_group.setLayout(structure_layout)
        left_layout.addWidget(structure_group)

        # Thêm stretch để đẩy các widget lên trên
        left_layout.addStretch(1)

        # Phần phải: Biểu đồ DVH và bảng thống kê
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Biểu đồ DVH
        dvh_group = QGroupBox("Biểu đồ Liều-Thể tích (DVH)")
        dvh_layout = QVBoxLayout()

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.ax = self.figure.add_subplot(111)

            dvh_layout.addWidget(self.toolbar)
            dvh_layout.addWidget(self.canvas)
            else:
            # Hiển thị thông báo nếu matplotlib không khả dụng
            dvh_label = QLabel(
                "Matplotlib không khả dụng. Không thể hiển thị biểu đồ DVH."
            )
            dvh_label.setStyleSheet("color: red;")
            dvh_layout.addWidget(dvh_label)

        dvh_group.setLayout(dvh_layout)
        right_layout.addWidget(dvh_group)

        # Bảng thống kê
        stats_group = QGroupBox("Thống kê liều lượng")
        stats_layout = QVBoxLayout()

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(7)
        self.stats_table.setHorizontalHeaderLabels(
            ["Cấu trúc", "Dmin", "Dmax", "Dmean", "D95", "V20Gy", "V30Gy"]
        )
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        stats_layout.addWidget(self.stats_table)

        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        # Thêm các panel vào splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([200, 600])  # Thiết lập kích thước ban đầu

        main_layout.addWidget(main_splitter)

        # Thêm stylesheet
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

            QCheckBox {
                spacing: 5px;
            }

            QComboBox {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 3px 5px;
            }

            QTableWidget {
                border: 1px solid #cccccc;
                gridline-color: #e0e0e0;
            }

            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1px solid #e0e0e0;
                padding: 4px;
            }

            QToolBar {
                border: none;
                background-color: #f5f5f5;
            }
        """)

        # Kích hoạt các control
        self.set_controls_enabled(False)  # Tắt các control cho đến khi có dữ liệu

    def _connect_signals(self):
        """Kết nối các signals và slots."""
        if not PYQT_AVAILABLE:
            return

        # Kết nối các sự kiện thay đổi
        self.plan_combo.currentIndexChanged.connect(self.on_plan_changed)
        self.dvh_type_combo.currentIndexChanged.connect(self.on_dvh_type_changed)
        self.volume_combo.currentIndexChanged.connect(self.on_volume_display_changed)
        self.norm_combo.currentTextChanged.connect(self.on_normalization_changed)

    def set_controls_enabled(self, enabled):
        """Bật/tắt các điều khiển UI."""
        self.dvh_type_combo.setEnabled(enabled)
        self.volume_combo.setEnabled(enabled)
        self.norm_combo.setEnabled(enabled)
        self.select_all_button.setEnabled(enabled)
        self.deselect_all_button.setEnabled(enabled)

    def add_plan(self, plan, name=None):
        """
        Thêm kế hoạch mới để hiển thị DVH.

        Parameters:
            plan: Đối tượng Plan
            name: Tên hiển thị cho kế hoạch (nếu None, sẽ sử dụng plan.name)
        """
        if not plan:
            return

        # Lấy tên kế hoạch
        plan_name = (
            name if name else getattr(plan, "name", f"Plan {len(self.plans) + 1}")
        )

        # Thêm vào danh sách kế hoạch
        self.plans[plan_name] = plan

        # Cập nhật UI
        self.plan_combo.blockSignals(True)
        self.plan_combo.addItem(plan_name)
        self.plan_combo.blockSignals(False)

        # Nếu đây là kế hoạch đầu tiên, chọn nó
        if len(self.plans) == 1:
            self.plan_combo.setCurrentText(plan_name)
            self.on_plan_changed(0)

    def remove_plan(self, plan_name):
        """
        Xóa kế hoạch khỏi danh sách.

        Parameters:
            plan_name: Tên kế hoạch cần xóa
        """
        if plan_name not in self.plans:
            return

        # Xóa khỏi danh sách
        del self.plans[plan_name]

        # Cập nhật UI
        index = self.plan_combo.findText(plan_name)
        if index >= 0:
            self.plan_combo.removeItem(index)

        # Xóa DVH đã tính toán
        keys_to_remove = []
        for key in self.calculated_dvhs:
            if key[0] == plan_name:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.calculated_dvhs[key]

        # Cập nhật biểu đồ nếu cần
        self.refresh_dvh()

    def clear_plans(self):
        """Xóa tất cả kế hoạch."""
        self.plans.clear()
        self.calculated_dvhs.clear()
        self.selected_structures.clear()

        # Cập nhật UI
        self.plan_combo.clear()
        self.clear_structure_lists()

        # Xóa biểu đồ
        if MATPLOTLIB_AVAILABLE:
            self.ax.clear()
            self.canvas.draw()

        # Xóa bảng thống kê
        self.stats_table.setRowCount(0)

        # Tắt các điều khiển
        self.set_controls_enabled(False)

    def clear_structure_lists(self):
        """Xóa danh sách cấu trúc."""
        # Xóa các check box trong danh sách cấu trúc
        for layout in [self.ptv_layout, self.oar_layout, self.other_layout]:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def update_structure_lists(self):
        """Cập nhật danh sách cấu trúc từ kế hoạch hiện tại."""
        # Xóa danh sách cũ
        self.clear_structure_lists()

        if not self.current_plan_name or self.current_plan_name not in self.plans:
            return

        plan = self.plans[self.current_plan_name]
        if not plan or not hasattr(plan, "structure_set") or not plan.structure_set:
            return

        # Lấy các cấu trúc từ structure_set
        structures = []
        if hasattr(plan.structure_set, "structures"):
            structures = plan.structure_set.structures

        # Thêm checkbox cho mỗi cấu trúc
        for structure in structures:
            # Tạo checkbox
            checkbox = QCheckBox(structure.name)

            # Đặt màu cho checkbox (sử dụng màu cấu trúc)
            if hasattr(structure, "color"):
                color = self.get_structure_color(structure)
                self.structure_colors[structure.name] = color

                # Tạo một ô màu nhỏ bên cạnh checkbox
                icon_size = 16
                pixmap = QPixmap(icon_size, icon_size)
                pixmap.fill(QColor(*color))
                checkbox.setIcon(QIcon(pixmap))

            # Kết nối sự kiện thay đổi trạng thái
            checkbox.stateChanged.connect(
                partial(self.on_structure_selection_changed, structure.name)
            )

            # Thêm vào nhóm phù hợp dựa trên loại cấu trúc
            if hasattr(structure, "type"):
                if structure.type == StructureType.PTV:
                    self.ptv_layout.addWidget(checkbox)
                elif structure.type == StructureType.OAR:
                    self.oar_layout.addWidget(checkbox)
        else:
                    self.other_layout.addWidget(checkbox)
            else:
                self.other_layout.addWidget(checkbox)

    def get_structure_color(self, structure):
        """
        Lấy màu cấu trúc để hiển thị trên biểu đồ.

        Parameters:
            structure: Đối tượng Structure

        Returns:
            Tuple (r, g, b) hoặc (r, g, b, a)
        """
        if hasattr(structure, "color"):
            color = structure.color
            # Đảm bảo màu là rgb hoặc rgba
            if len(color) == 3:
                return color
            elif len(color) == 4:
                return color
            else:
                # Tạo màu ngẫu nhiên nếu không đúng định dạng
                return (
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                )
        else:
            # Tạo màu ngẫu nhiên nếu không có thuộc tính color
            return (
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256),
            )

    def on_plan_changed(self, index):
        """
        Xử lý khi kế hoạch được chọn thay đổi.

        Parameters:
            index: Chỉ số của kế hoạch được chọn
        """
        if index < 0 or self.plan_combo.count() == 0:
            return

        # Lấy tên kế hoạch mới
        new_plan_name = self.plan_combo.itemText(index)
        if new_plan_name not in self.plans:
            return

        # Cập nhật kế hoạch hiện tại
        self.current_plan_name = new_plan_name

        # Cập nhật danh sách cấu trúc
        self.update_structure_lists()

        # Bật các điều khiển
        self.set_controls_enabled(True)

        # Thông báo thay đổi kế hoạch
        self.planChanged.emit(self.plans[new_plan_name])

    def on_structure_selection_changed(self, structure_name, state):
        """
        Xử lý khi lựa chọn cấu trúc thay đổi.

        Parameters:
            structure_name: Tên cấu trúc
            state: Trạng thái checkbox (Qt.Checked hoặc Qt.Unchecked)
        """
        if state == Qt.Checked:
            self.selected_structures.add(structure_name)
        else:
            if structure_name in self.selected_structures:
                self.selected_structures.remove(structure_name)

        # Thông báo thay đổi lựa chọn cấu trúc
        self.structureSelectionChanged.emit(list(self.selected_structures))

        # Cập nhật biểu đồ
        self.update_dvh_plot()

        # Cập nhật bảng thống kê
        self.update_stats_table()

    def on_dvh_type_changed(self, index):
        """
        Xử lý khi loại DVH thay đổi.

        Parameters:
            index: Chỉ số của loại DVH được chọn
        """
        self.dvh_type = self.dvh_type_combo.itemData(index)
        self.update_dvh_plot()

    def on_volume_display_changed(self, index):
        """
        Xử lý khi kiểu hiển thị thể tích thay đổi.

        Parameters:
            index: Chỉ số của loại hiển thị được chọn
        """
        self.display_volumes = self.volume_combo.itemData(index)
        self.update_dvh_plot()

    def on_normalization_changed(self, text):
        """
        Xử lý khi giá trị chuẩn hóa thay đổi.

        Parameters:
            text: Giá trị chuẩn hóa mới (dạng chuỗi, ví dụ "100%")
        """
        try:
            # Lấy giá trị số từ chuỗi (loại bỏ %)
            value_str = text.strip().rstrip("%")
            value = float(value_str)

            # Cập nhật giá trị chuẩn hóa
            self.normalization_value = value

            # Cập nhật biểu đồ
            self.update_dvh_plot()
        except ValueError:
            # Bỏ qua nếu không chuyển đổi được thành số
            pass

    def select_all_structures(self):
        """Chọn tất cả cấu trúc."""
        self._update_all_checkboxes(True)

    def deselect_all_structures(self):
        """Bỏ chọn tất cả cấu trúc."""
        self._update_all_checkboxes(False)

    def _update_all_checkboxes(self, checked):
        """
        Cập nhật trạng thái tất cả checkbox.

        Parameters:
            checked: Trạng thái mới (True: chọn, False: bỏ chọn)
        """
        # Lặp qua tất cả checkbox trong các layout
        for layout in [self.ptv_layout, self.oar_layout, self.other_layout]:
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if isinstance(widget, QCheckBox):
                    # Chặn signals để tránh gọi lại on_structure_selection_changed nhiều lần
                    widget.blockSignals(True)
                    widget.setChecked(checked)
                    widget.blockSignals(False)

                    # Cập nhật danh sách cấu trúc được chọn
                    if checked:
                        self.selected_structures.add(widget.text())
                    else:
                        if widget.text() in self.selected_structures:
                            self.selected_structures.remove(widget.text())

        # Thông báo thay đổi lựa chọn cấu trúc
        self.structureSelectionChanged.emit(list(self.selected_structures))

        # Cập nhật biểu đồ
        self.update_dvh_plot()

        # Cập nhật bảng thống kê
        self.update_stats_table()

    def calculate_dvh(self, plan_name, structure_name):
        """
        Tính toán DVH cho cấu trúc trong kế hoạch.

        Parameters:
            plan_name: Tên kế hoạch
            structure_name: Tên cấu trúc

        Returns:
            Dữ liệu DVH hoặc None nếu không thể tính toán
        """
        key = (plan_name, structure_name)

        # Kiểm tra xem DVH đã được tính toán chưa
        if key in self.calculated_dvhs:
            return self.calculated_dvhs[key]

        # Lấy kế hoạch
        if plan_name not in self.plans:
            return None

        plan = self.plans[plan_name]
        if (
            not plan
            or not hasattr(plan, "dose_grid")
            or not hasattr(plan, "structure_set")
        ):
            return None

        # Lấy structure_set và dose_grid
        structure_set = plan.structure_set
        dose_grid = plan.dose_grid

        if not structure_set or not dose_grid:
            return None

        # Tìm cấu trúc
        structure = None
        if hasattr(structure_set, "get_structure_by_name"):
            structure = structure_set.get_structure_by_name(structure_name)
        elif hasattr(structure_set, "structures"):
            for s in structure_set.structures:
                if hasattr(s, "name") and s.name == structure_name:
                    structure = s
                    break

        if not structure:
            return None

        try:
            # Tính toán DVH
            dvh_data = self.dvh_calculator.calculate_dvh(
                structure, dose_grid, self.dvh_type
            )

            # Lưu kết quả
            self.calculated_dvhs[key] = dvh_data

            return dvh_data
        except Exception as e:
            logger.error(f"Lỗi khi tính toán DVH: {e}")
            return None

    def update_dvh_plot(self):
        """Cập nhật biểu đồ DVH với dữ liệu mới nhất."""
        if not MATPLOTLIB_AVAILABLE:
            return

        if not self.current_plan_name or not self.selected_structures:
            # Xóa biểu đồ nếu không có dữ liệu
        self.ax.clear()
            self.canvas.draw()
            return

        # Xóa biểu đồ cũ
        self.ax.clear()

        # Danh sách để lưu trữ các đường và nhãn cho legend
        lines = []
        labels = []

        # Tính toán và vẽ DVH cho mỗi cấu trúc được chọn
        for structure_name in self.selected_structures:
            # Tính toán DVH cho kế hoạch hiện tại
            dvh_data = self.calculate_dvh(self.current_plan_name, structure_name)
            if not dvh_data:
                continue

            # Lấy màu cấu trúc
            color = self.structure_colors.get(
                structure_name, (0, 0, 255)
            )  # Mặc định là xanh lam

            # Chuẩn bị màu cho matplotlib (chuyển từ 0-255 sang 0-1)
            mpl_color = tuple(c / 255.0 for c in color[:3])

            # Lấy dữ liệu từ dvh_data
            dose_values = dvh_data.get("dose", [])
            volume_values = dvh_data.get("volume", [])

            if len(dose_values) == 0 or len(volume_values) == 0:
                continue

            # Chuẩn hóa liều nếu cần
            if self.normalization_value != 100.0:
                dose_values = np.array(dose_values) * (self.normalization_value / 100.0)

            # Kiểm tra xem hiển thị thể tích tương đối hay tuyệt đối
            if self.display_volumes and "relative_volume" in dvh_data:
                volume_values = dvh_data["relative_volume"]
            elif not self.display_volumes and "absolute_volume" in dvh_data:
                volume_values = dvh_data["absolute_volume"]

            # Vẽ đường DVH
            (line,) = self.ax.plot(
                dose_values,
                volume_values,
                color=mpl_color,
                linestyle=self.plan_line_styles.get(0, "-"),  # Kiểu đường mặc định
                label=structure_name,
                linewidth=2,
            )

            lines.append(line)
            labels.append(structure_name)

        # Đặt nhãn trục
        self.ax.set_xlabel(f"Liều ({self.dose_unit})")
        if self.display_volumes:
            self.ax.set_ylabel("Thể tích (%)")
        else:
            self.ax.set_ylabel("Thể tích (cc)")

        # Đặt tiêu đề
        self.ax.set_title(f"DVH - {self.current_plan_name}")

        # Hiển thị lưới
        self.ax.grid(True, linestyle="--", alpha=0.7)

        # Đặt giới hạn trục x và y
        self.ax.set_xlim(0, None)  # Bắt đầu từ 0, kết thúc tự động
        self.ax.set_ylim(
            0, 105 if self.display_volumes else None
        )  # Giới hạn y tùy thuộc vào kiểu hiển thị

        # Hiển thị legend
        if lines:
            self.ax.legend(handles=lines, labels=labels, loc="upper right")

        # Cập nhật canvas
        self.canvas.draw()

    def update_stats_table(self):
        """Cập nhật bảng thống kê với dữ liệu mới nhất."""
        if not self.current_plan_name or not self.selected_structures:
            # Xóa bảng nếu không có dữ liệu
            self.stats_table.setRowCount(0)
            return

        # Chuẩn bị dữ liệu cho bảng
        table_data = []

        # Tính toán thống kê cho mỗi cấu trúc được chọn
        for structure_name in self.selected_structures:
            # Tính toán DVH cho kế hoạch hiện tại
            dvh_data = self.calculate_dvh(self.current_plan_name, structure_name)
            if not dvh_data:
                continue

            # Lấy dữ liệu từ dvh_data
            dose_values = dvh_data.get("dose", [])
            volume_values = dvh_data.get("volume", [])

            if len(dose_values) == 0 or len(volume_values) == 0:
                continue

            # Tính các chỉ số thống kê
            dmin = np.min(dose_values) if len(dose_values) > 0 else 0
            dmax = np.max(dose_values) if len(dose_values) > 0 else 0

            # Dmean (tính bằng cách lấy trung bình có trọng số)
            if "differential_volume" in dvh_data and len(
                dvh_data["differential_volume"]
            ) == len(dose_values):
                diff_volume = dvh_data["differential_volume"]
                dmean = (
                    np.sum(dose_values * diff_volume) / np.sum(diff_volume)
                    if np.sum(diff_volume) > 0
                    else 0
                )
            else:
                dmean = np.mean(dose_values) if len(dose_values) > 0 else 0

            # D95 (liều nhận bởi 95% thể tích)
            d95 = 0
            if self.dvh_type == DVHType.CUMULATIVE:
                # Tìm vị trí gần nhất với 95%
                idx_95 = (
                    np.abs(volume_values - 95).argmin()
                    if self.display_volumes
                    else None
                )
                if idx_95 is not None and idx_95 < len(dose_values):
                    d95 = dose_values[idx_95]

            # V20Gy và V30Gy (% thể tích nhận liều >= 20Gy và 30Gy)
            v20gy = 0
            v30gy = 0
            if self.dvh_type == DVHType.CUMULATIVE:
                # Tìm vị trí gần nhất với 20Gy và 30Gy
                idx_20gy = np.abs(dose_values - 20).argmin()
                idx_30gy = np.abs(dose_values - 30).argmin()

                if idx_20gy < len(volume_values):
                    v20gy = volume_values[idx_20gy]

                if idx_30gy < len(volume_values):
                    v30gy = volume_values[idx_30gy]

            # Thêm vào dữ liệu bảng
            table_data.append(
                {
                    "structure_name": structure_name,
                    "dmin": dmin,
                    "dmax": dmax,
                    "dmean": dmean,
                    "d95": d95,
                    "v20gy": v20gy,
                    "v30gy": v30gy,
                }
            )

        # Cập nhật bảng
        self.stats_table.setRowCount(len(table_data))

        for i, data in enumerate(table_data):
            # Cấu trúc
            item = QTableWidgetItem(data["structure_name"])
            # Đặt màu cho item
            color = self.structure_colors.get(data["structure_name"], (0, 0, 255))
            item.setForeground(QBrush(QColor(*color)))
            self.stats_table.setItem(i, 0, item)

            # Các chỉ số thống kê
            self.stats_table.setItem(
                i, 1, QTableWidgetItem(f"{data['dmin']:.2f} {self.dose_unit}")
            )
            self.stats_table.setItem(
                i, 2, QTableWidgetItem(f"{data['dmax']:.2f} {self.dose_unit}")
            )
            self.stats_table.setItem(
                i, 3, QTableWidgetItem(f"{data['dmean']:.2f} {self.dose_unit}")
            )
            self.stats_table.setItem(
                i, 4, QTableWidgetItem(f"{data['d95']:.2f} {self.dose_unit}")
            )

            # V20Gy và V30Gy
            v20gy_unit = "%" if self.display_volumes else "cc"
            v30gy_unit = "%" if self.display_volumes else "cc"
            self.stats_table.setItem(
                i, 5, QTableWidgetItem(f"{data['v20gy']:.2f} {v20gy_unit}")
            )
            self.stats_table.setItem(
                i, 6, QTableWidgetItem(f"{data['v30gy']:.2f} {v30gy_unit}")
            )

    def refresh_dvh(self):
        """Làm mới biểu đồ DVH và bảng thống kê."""
        # Xóa DVH đã tính toán
        self.calculated_dvhs.clear()

        # Cập nhật biểu đồ
        self.update_dvh_plot()

        # Cập nhật bảng thống kê
        self.update_stats_table()

    def export_dvh(self):
        """Xuất biểu đồ DVH ra file."""
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(
                self, "Lỗi", "Matplotlib không khả dụng. Không thể xuất biểu đồ DVH."
            )
            return

        if not self.current_plan_name or not self.selected_structures:
            QMessageBox.warning(self, "Lỗi", "Không có dữ liệu để xuất.")
            return

        # Hiển thị hộp thoại chọn file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất biểu đồ DVH",
            "",
            "Hình ảnh PNG (*.png);;Hình ảnh PDF (*.pdf);;Hình ảnh SVG (*.svg)",
        )

        if not file_path:
            return

        try:
            # Lưu biểu đồ ra file
            self.figure.savefig(file_path, dpi=300, bbox_inches="tight")
            QMessageBox.information(
                self, "Thành công", f"Đã xuất biểu đồ DVH ra file {file_path}."
            )
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xuất biểu đồ DVH: {str(e)}")

    def print_dvh(self):
        """In biểu đồ DVH."""
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(
                self, "Lỗi", "Matplotlib không khả dụng. Không thể in biểu đồ DVH."
            )
            return

        if not self.current_plan_name or not self.selected_structures:
            QMessageBox.warning(self, "Lỗi", "Không có dữ liệu để in.")
            return

        try:
            # Hiển thị hộp thoại để chọn file PDF đầu ra
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Xuất biểu đồ DVH để in",
                "",
                "Tệp PDF (*.pdf)",
            )

            if file_path:
                # Lưu biểu đồ trực tiếp ra PDF với độ phân giải cao
                self.figure.savefig(
                    file_path, format="pdf", dpi=300, bbox_inches="tight"
                )
                QMessageBox.information(
                    self,
                    "Thành công",
                    f"Đã xuất biểu đồ DVH ra file {file_path} để in.",
                )
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể in biểu đồ DVH: {str(e)}")


def show_dvh_dialog(parent=None, plan=None):
    """
    Hiển thị hộp thoại DVH.

    Parameters:
        parent: Widget cha
        plan: Kế hoạch xạ trị

    Returns:
        DVHWidget instance
    """
    if not PYQT_AVAILABLE:
        logging.error("Không thể hiển thị hộp thoại DVH: PyQt5 không khả dụng")
        return None

    try:
        # Sử dụng các class đã import ở đầu file
        dialog = QDialog(parent) if "QDialog" in globals() else None

        # Nếu không có QDialog, tạo một widget thay thế
        if dialog is None:
            dialog = QWidget(parent)
            dialog.setWindowTitle = lambda x: None
            dialog.resize = lambda x, y: None
            dialog.show = lambda: None
            dialog.reject = lambda: None

        dialog.setWindowTitle("Biểu đồ Liều-Thể tích (DVH)")
        dialog.resize(1000, 700)

        layout = QVBoxLayout(dialog)

        # Tạo widget DVH
        dvh_widget = DVHWidget(dialog)
        if plan:
            dvh_widget.add_plan(plan)

        layout.addWidget(dvh_widget)

        # Buttons
        button_box = (
            QDialogButtonBox(QDialogButtonBox.Close)
            if "QDialogButtonBox" in globals()
            else QPushButton("Đóng")
        )
        if isinstance(button_box, QDialogButtonBox):
            button_box.rejected.connect(dialog.reject)
        else:
            button_box.clicked.connect(dialog.close)
        layout.addWidget(button_box)

        # Hiển thị hộp thoại
        dialog.show()

        return dvh_widget
    except Exception as e:
        logging.error(f"Lỗi khi hiển thị hộp thoại DVH: {e}")
        return None
