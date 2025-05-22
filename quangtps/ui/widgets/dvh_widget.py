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


# Kiểm tra Eclipse theme và import create_eclipse_widget_style
HAS_ECLIPSE_THEME = False
try:
    from quangtps.ui.eclipse_style_theme import create_eclipse_widget_style

    HAS_ECLIPSE_THEME = True
except ImportError:
    logging.debug("Không tìm thấy eclipse_style_theme, sử dụng style mặc định")

    # Hàm giả để tránh lỗi khi gọi
    def create_eclipse_widget_style(*args, **kwargs):
        return ""


try:
    # Import các module từ QuangTPS
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure import Structure, StructureType
    from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh
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

    # Hàm giả để tránh lỗi khi calculate_dvh không khả dụng
    def calculate_dvh(*args, **kwargs):
        logging.error("Hàm calculate_dvh không khả dụng!")
            return {
            "dose_bins": np.array([0]),
            "differential_volume": np.array([0]),
            "cumulative_volume": np.array([0]),
            "min_dose": 0,
            "max_dose": 0,
            "mean_dose": 0,
            "median_dose": 0,
            "std_dose": 0,
            "volume": 0,
            }

    class DVHType(enum.Enum):
        CUMULATIVE = "CUMULATIVE"
        DIFFERENTIAL = "DIFFERENTIAL"

    class DVHCalculator:
        def __init__(self):
            pass

        def calculate_dvh(self, structure, dose_grid, dvh_type, **kwargs):
            # Trả về dữ liệu DVH giả
            return {
                "dose": np.linspace(0, 70, 100),
                "volume": np.exp(-np.linspace(0, 7, 100)),
                "min_dose": 0,
                "max_dose": 70,
                "mean_dose": 20,
                "differential_volume": np.exp(-np.linspace(0, 7, 100)) * 0.1,
                "metrics": {
                    "D95": 40,
                    "D50": 60,
                    "D5": 69,
                },
            }

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

        # Thêm hỗ trợ cho DVH bands từ phân tích độ bền vững
        self.robustness_results = {}  # Dict[structure_name, robustness_result]
        self.show_robustness_bands = False  # Có hiển thị dải DVH không
        self.robustness_alpha = 0.3  # Độ trong suốt của dải DVH

        # Các thành phần UI
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        if not PYQT_AVAILABLE:
            layout = QVBoxLayout(self)
            label = QLabel("PyQt không khả dụng, không thể hiển thị DVH")
            layout.addWidget(label)
            return

        # Tạo layout chính
        main_layout = QVBoxLayout(self)

        # Tạo splitter để chia màn hình
        splitter = QSplitter(Qt.Horizontal)

        # Panel bên trái: Điều khiển và danh sách cấu trúc
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Tạo group box cho các điều khiển
        controls_group = QGroupBox("Điều khiển DVH")
        controls_layout = QFormLayout(controls_group)

        # Chọn kế hoạch
        self.plan_combo = QComboBox()
        controls_layout.addRow("Kế hoạch:", self.plan_combo)

        # Loại DVH (tích lũy/vi phân)
        self.dvh_type_combo = QComboBox()
        self.dvh_type_combo.addItems(["Tích lũy", "Vi phân"])
        controls_layout.addRow("Loại DVH:", self.dvh_type_combo)

        # Hiển thị thể tích (% hoặc cc)
        self.volume_combo = QComboBox()
        self.volume_combo.addItems(["Tương đối (%)", "Tuyệt đối (cc)"])
        controls_layout.addRow("Thể tích:", self.volume_combo)

        # Chuẩn hóa liều
        self.norm_combo = QComboBox()
        self.norm_combo.setEditable(True)
        self.norm_combo.addItems(["100", "95", "90", "80", "50"])
        controls_layout.addRow("Chuẩn hóa (%):", self.norm_combo)

        # Thêm checkbox hiển thị dải DVH từ phân tích độ bền vững
        self.show_bands_checkbox = QCheckBox("Hiển thị dải DVH")
        self.show_bands_checkbox.setToolTip("Hiển thị dải DVH từ phân tích độ bền vững")
        self.show_bands_checkbox.setChecked(False)
        controls_layout.addRow("Độ bền vững:", self.show_bands_checkbox)

        left_layout.addWidget(controls_group)

        # Tạo group box cho danh sách cấu trúc
        structures_group = QGroupBox("Cấu trúc")
        structures_layout = QVBoxLayout(structures_group)

        # Các nút chọn tất cả/bỏ chọn tất cả
        buttons_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Chọn tất cả")
        buttons_layout.addWidget(self.select_all_button)

        self.deselect_all_button = QPushButton("Bỏ chọn tất cả")
        buttons_layout.addWidget(self.deselect_all_button)
        structures_layout.addLayout(buttons_layout)

        # Tạo scroll area cho danh sách cấu trúc
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        self.structures_widget = QWidget()
        self.structures_layout = QVBoxLayout(self.structures_widget)
        self.structures_layout.setContentsMargins(0, 0, 0, 0)
        self.structures_layout.setSpacing(2)
        self.structures_layout.addStretch()

        scroll_area.setWidget(self.structures_widget)
        structures_layout.addWidget(scroll_area)

        left_layout.addWidget(structures_group)
        splitter.addWidget(left_panel)

        # Panel bên phải: Biểu đồ DVH và bảng thống kê
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Tạo tab widget cho biểu đồ và bảng thống kê
        tab_widget = QTabWidget()

        # Tab biểu đồ DVH
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)

        # Tạo biểu đồ matplotlib
        self.figure = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel(f"Liều ({self.dose_unit})")
        self.ax.set_ylabel("Thể tích (%)")
        self.ax.set_title("Biểu đồ liều-thể tích (DVH)")
        self.ax.grid(True)

        # Thêm thanh công cụ matplotlib
        self.toolbar = NavigationToolbar(self.canvas, self)
            dvh_layout.addWidget(self.toolbar)
            dvh_layout.addWidget(self.canvas)

        tab_widget.addTab(dvh_tab, "Biểu đồ DVH")

        # Tab thống kê liều
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(6)
        self.stats_table.setHorizontalHeaderLabels(
            ["Cấu trúc", "Min (Gy)", "Max (Gy)", "Mean (Gy)", "D95 (Gy)", "V20 (%)"]
        )
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(self.stats_table)

        tab_widget.addTab(stats_tab, "Thống kê liều")

        right_layout.addWidget(tab_widget)

        # Thêm các nút xuất và in DVH
        buttons_layout = QHBoxLayout()
        self.export_button = QPushButton("Xuất DVH")
        buttons_layout.addWidget(self.export_button)

        self.print_button = QPushButton("In DVH")
        buttons_layout.addWidget(self.print_button)

        right_layout.addLayout(buttons_layout)

        splitter.addWidget(right_panel)
        splitter.setSizes([200, 600])  # Thiết lập kích thước ban đầu

        main_layout.addWidget(splitter)

        # Thiết lập style Eclipse nếu có
        if HAS_ECLIPSE_THEME and "create_eclipse_widget_style" in globals():
            try:
                # Sửa lỗi quá nhiều tham số, chỉ cần truyền kiểu widget
                self.setStyleSheet(create_eclipse_widget_style("table"))
            except Exception as e:
                logger.debug(f"Không thể thiết lập Eclipse style cho bảng: {e}")

    def _connect_signals(self):
        """Kết nối các tín hiệu với các slot."""
        self.plan_combo.currentIndexChanged.connect(self.on_plan_changed)
        self.dvh_type_combo.currentIndexChanged.connect(self.on_dvh_type_changed)
        self.volume_combo.currentIndexChanged.connect(self.on_volume_display_changed)
        self.norm_combo.currentTextChanged.connect(self.on_normalization_changed)
        self.select_all_button.clicked.connect(self.select_all_structures)
        self.deselect_all_button.clicked.connect(self.deselect_all_structures)
        self.export_button.clicked.connect(self.export_dvh)
        self.print_button.clicked.connect(self.print_dvh)

        # Kết nối checkbox hiển thị dải DVH
        self.show_bands_checkbox.toggled.connect(self.on_show_bands_toggled)

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
        for layout in [self.structures_layout]:
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
                    self.structures_layout.addWidget(checkbox)
                elif structure.type == StructureType.OAR:
                    self.structures_layout.addWidget(checkbox)
                else:
                    self.structures_layout.addWidget(checkbox)
            else:
                self.structures_layout.addWidget(checkbox)

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
        Xử lý khi chọn hiển thị thể tích tương đối hoặc tuyệt đối.

        Parameters:
            index: Chỉ số lựa chọn từ combo box
        """
        # Lưu lại chế độ hiển thị
        self.display_volumes = index == 0  # 0: Tương đối (%), 1: Tuyệt đối (cc)

        # Cập nhật hiển thị đơn vị thể tích trong header của bảng thống kê
        if self.display_volumes:
            # Cập nhật header cho hiển thị phần trăm
            self.stats_table.horizontalHeaderItem(5).setText("V20 (%)")
            self.stats_table.horizontalHeaderItem(6).setText("V30 (%)")
        else:
            # Cập nhật header cho hiển thị thể tích tuyệt đối
            self.stats_table.horizontalHeaderItem(5).setText("V20 (cc)")
            self.stats_table.horizontalHeaderItem(6).setText("V30 (cc)")

        # Cập nhật biểu đồ DVH
        self.update_dvh_plot()

        # Cập nhật bảng thống kê liều
        self.update_stats_table()

        logger.debug(
            f"Chế độ hiển thị thể tích: {'Tương đối (%)' if self.display_volumes else 'Tuyệt đối (cc)'}"
        )

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
        for layout in [self.structures_layout]:
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
        """Cập nhật biểu đồ DVH với dữ liệu hiện tại."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib không khả dụng, không thể vẽ biểu đồ DVH")
            return

        # Xóa biểu đồ cũ
            self.ax.clear()

        # Thiết lập tiêu đề và nhãn trục
        volume_label = "Thể tích (%)" if self.display_volumes else "Thể tích (cc)"
        dvh_type_label = (
            "Tích lũy" if self.dvh_type == DVHType.CUMULATIVE else "Vi phân"
        )

        self.ax.set_xlabel(f"Liều ({self.dose_unit})")
        self.ax.set_ylabel(volume_label)
        self.ax.set_title(f"Biểu đồ {dvh_type_label} liều-thể tích (DVH)")
        self.ax.grid(True)

        # Kiểm tra xem có kế hoạch và cấu trúc được chọn không
        if not self.current_plan_name or not self.selected_structures:
            self.canvas.draw()
            return

        # Vẽ DVH cho từng cấu trúc được chọn
        # Sắp xếp để các TARGET (PTV) hiển thị trước, sau đó là OAR, cuối cùng là các cấu trúc khác
        structure_types = {}

        # Phân loại các cấu trúc theo loại
        for structure_name in self.selected_structures:
            plan = self.plans.get(self.current_plan_name)
            if not plan:
                continue

            # Lấy thông tin cấu trúc từ kế hoạch
            structure_set = getattr(plan, "structure_set", None)
            if not structure_set:
                continue

            # Tìm cấu trúc trong tập cấu trúc
            structure = None
            for s in getattr(structure_set, "structures", []):
                if getattr(s, "name", "") == structure_name:
                    structure = s
                    break

            if structure:
                structure_type = getattr(structure, "structure_type", None)
                if structure_type in (StructureType.PTV, "PTV", "TARGET", "GTV", "CTV"):
                    structure_types[structure_name] = "TARGET"
                elif structure_type in (StructureType.OAR, "OAR", "ORGAN"):
                    structure_types[structure_name] = "OAR"
                else:
                    structure_types[structure_name] = "OTHER"
            else:
                # Thử phỏng đoán loại cấu trúc từ tên
                if any(
                    kw in structure_name.upper()
                    for kw in ["PTV", "GTV", "CTV", "TARGET"]
                ):
                    structure_types[structure_name] = "TARGET"
                elif any(kw in structure_name.upper() for kw in ["OAR", "ORGAN"]):
                    structure_types[structure_name] = "OAR"
                else:
                    structure_types[structure_name] = "OTHER"

        # Sắp xếp cấu trúc theo loại: TARGET trước, sau đó OAR, cuối cùng là OTHER
        sorted_structures = sorted(
            self.selected_structures,
            key=lambda x: (
                0
                if structure_types.get(x) == "TARGET"
                else 1
                if structure_types.get(x) == "OAR"
                else 2,
                x,
            ),
        )

        # Đếm số lượng mỗi loại để lựa chọn style line phù hợp
        target_count = sum(
            1 for x in sorted_structures if structure_types.get(x) == "TARGET"
        )
        oar_count = sum(1 for x in sorted_structures if structure_types.get(x) == "OAR")

        # Chuẩn bị danh sách style cho các loại cấu trúc khác nhau
        target_line_styles = ["-", "--", "-.", ":"]
        oar_line_styles = ["-", "--", "-.", ":"]
        target_line_width = 2.5
        oar_line_width = 2.0
        other_line_width = 1.5

        target_index = 0
        oar_index = 0
        other_index = 0

        for structure_name in sorted_structures:
            # Lấy dữ liệu DVH
            key = (self.current_plan_name, structure_name)
            if key not in self.calculated_dvhs:
                try:
                    self.calculate_dvh(self.current_plan_name, structure_name)
                except Exception as e:
                    logger.error(f"Lỗi khi tính toán DVH cho {structure_name}: {e}")
                continue

            dvh_data = self.calculated_dvhs.get(key)
            if not dvh_data:
                continue

            # Lấy màu cho cấu trúc
            color = self.get_structure_color(structure_name)
            structure_type = structure_types.get(structure_name, "OTHER")

            # Quyết định style line dựa trên loại cấu trúc
            line_style = "-"  # Mặc định
            line_width = other_line_width

            if structure_type == "TARGET":
                line_style = target_line_styles[target_index % len(target_line_styles)]
                line_width = target_line_width
                target_index += 1
            elif structure_type == "OAR":
                line_style = oar_line_styles[oar_index % len(oar_line_styles)]
                line_width = oar_line_width
                oar_index += 1
            else:
                line_style = ":"  # Đường chấm cho các cấu trúc khác
                line_width = other_line_width
                other_index += 1

            # Vẽ đường DVH chính
            dose = dvh_data.get("dose", [])
            volume = dvh_data.get("volume", [])

            # Chuyển đổi thể tích nếu cần
            if not self.display_volumes and "absolute_volume" in dvh_data:
                # Sử dụng thể tích tuyệt đối (cc) thay vì phần trăm
                volume = dvh_data.get("absolute_volume", [])

            # Thêm thông tin cấu trúc vào nhãn
            label = f"{structure_name}"
            if "volume_cc" in dvh_data:
                total_volume = dvh_data.get("volume_cc", 0)
                label = f"{structure_name} ({total_volume:.1f}cc)"

            self.ax.plot(
                dose,
                volume,
                color=color,
                label=label,
                linewidth=line_width,
                linestyle=line_style,
            )

            # Vẽ dải DVH nếu được bật và có dữ liệu
            if self.show_robustness_bands and structure_name in self.robustness_results:
                rob_data = self.robustness_results[structure_name]

                # Trích xuất dữ liệu dải DVH
                min_dvh = rob_data.get("min_dvh", {})
                max_dvh = rob_data.get("max_dvh", {})

                if min_dvh and max_dvh:
                    min_dose = min_dvh.get("dose", [])
                    min_volume = min_dvh.get("volume", [])
                    max_dose = max_dvh.get("dose", [])
                    max_volume = max_dvh.get("volume", [])

                    # Chuyển đổi thể tích nếu hiển thị thể tích tuyệt đối
                    if not self.display_volumes:
                        if "absolute_volume" in min_dvh:
                            min_volume = min_dvh.get("absolute_volume", [])
                        if "absolute_volume" in max_dvh:
                            max_volume = max_dvh.get("absolute_volume", [])

                    # Vẽ vùng giữa min và max
                    self.ax.fill_between(
                        min_dose,
                        min_volume,
                        max_volume,
                        color=color,
                        alpha=self.robustness_alpha,
                        label=f"{structure_name} (độ bền vững)",
                    )

        # Thêm legend với vị trí tự động
        if self.selected_structures:
            self.ax.legend(
                loc="upper right", bbox_to_anchor=(1.02, 1), fontsize="small"
            )

        # Thiết lập giới hạn trục y phù hợp
        if self.display_volumes:
            # Với thể tích tương đối, mặc định từ 0-105%
            self.ax.set_ylim(0, 105)
        else:
            # Với thể tích tuyệt đối, để tự động điều chỉnh dựa trên dữ liệu
            self.ax.set_ylim(bottom=0)  # Bắt đầu từ 0, để giới hạn trên tự động

        # Thiết lập giới hạn trục x, luôn bắt đầu từ 0
        self.ax.set_xlim(0, None)  # Bắt đầu từ 0, để giới hạn trên tự động

        # Cập nhật biểu đồ
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

    def set_robustness_result(self, result):
        """
        Thiết lập kết quả phân tích độ bền vững để hiển thị dải DVH.

        Parameters
        ----------
        result : RobustnessResult
            Kết quả phân tích độ bền vững
        """
        if result is None:
            self.robustness_results = {}
            self.show_robustness_bands = False
            return

        self.robustness_results = {}

        try:
            # Lấy dữ liệu DVH cho từng cấu trúc
            for structure_name, dvh_data in result.get_structure_dvhs().items():
                self.robustness_results[structure_name] = dvh_data

            # Bật hiển thị dải DVH
            self.show_robustness_bands = True

            # Cập nhật biểu đồ
            self.update_dvh_plot()
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập kết quả phân tích độ bền vững: {e}")
            self.show_robustness_bands = False

    def on_show_bands_toggled(self, checked):
        """Xử lý khi bật/tắt hiển thị dải DVH."""
        self.show_robustness_bands = checked
        self.update_dvh_plot()

    def _create_sample_dvh(self, structure):
        """
        Tạo dữ liệu DVH mẫu khi không có dữ liệu thực tế.

        Parameters
        ----------
        structure : Dict[str, Any]
            Thông tin cấu trúc

        Returns
        -------
        Dict[str, Any]
            Dữ liệu DVH mẫu
        """
        try:
            structure_name = structure.get("name", "Unknown")
            structure_type = self._get_structure_type(structure_name)

            # Số lượng điểm dữ liệu
            num_points = 100

            # Tạo mảng liều từ 0 đến 80 Gy
            dose = np.linspace(0, 80, num_points)

            # Tham số cho đường cong mẫu dựa vào loại cấu trúc
            if structure_type == "TARGET":
                # Tạo đường cong cho PTV (hình chữ nhật với vai phải)
                volume = np.ones(num_points) * 100
                # Liều theo toa từ 45-78 Gy
                prescription = np.random.uniform(45, 78)
                # Tìm chỉ số của liều theo toa
                idx = np.argmin(np.abs(dose - prescription))
                # Tạo vai phải với dropout từ 100% xuống 0%
                drop_idx = idx + np.random.randint(3, 10)  # Thêm biến động
                if drop_idx < num_points:
                    # Tạo sự suy giảm dần
                    volume[idx:drop_idx] = np.linspace(100, 5, drop_idx - idx)
                    volume[drop_idx:] = 0
            elif structure_type == "OAR":
                # Tạo đường cong cho OAR (hàm mũ giảm dần)
                mean_dose = np.random.uniform(
                    5, 30
                )  # Liều trung bình ngẫu nhiên từ 5-30 Gy

                # Tham số alpha cho tốc độ suy giảm (nhỏ hơn cho các OAR song song, lớn hơn cho các OAR nối tiếp)
                alpha = np.random.uniform(0.05, 0.15)

                # Tạo đường cong mũ giảm dần
                volume = 100 * np.exp(-alpha * dose)

                # Thêm nhiễu để tạo tính thực tế
                noise = np.random.normal(0, 2, num_points)
                volume = volume + noise
                volume = np.clip(volume, 0, 100)  # Đảm bảo giá trị trong khoảng 0-100%
        else:
                # Cấu trúc khác - tạo đường cong ngẫu nhiên
                volume = 100 * np.exp(-0.1 * dose) + np.random.normal(0, 5, num_points)
                volume = np.clip(volume, 0, 100)

            # Làm trơn đường cong
            from scipy.ndimage import gaussian_filter1d

            volume = gaussian_filter1d(volume, sigma=1.5)

            # Tạo một số chỉ số thống kê phổ biến
            d90 = np.interp(90, np.flip(volume), np.flip(dose))
            d50 = np.interp(50, np.flip(volume), np.flip(dose))
            d10 = np.interp(10, np.flip(volume), np.flip(dose))

            v20 = np.interp(20, dose, volume) if structure_type != "TARGET" else 100
            v10 = np.interp(10, dose, volume) if structure_type != "TARGET" else 100

            mean_dose = np.sum(dose * np.diff(np.append(volume, [0])) * -1) / 100

            # Hoàn thiện dictionary DVH
            dvh_data = {
                "dose": dose.tolist(),
                "volume": volume.tolist(),
                "metrics": {
                    "Dmin": dose[volume > 0].min() if np.any(volume > 0) else 0,
                    "Dmax": dose[volume > 0].max() if np.any(volume > 0) else 0,
                    "Dmean": mean_dose,
                    "D98": np.interp(98, np.flip(volume), np.flip(dose)),
                    "D95": np.interp(95, np.flip(volume), np.flip(dose)),
                    "D90": d90,
                    "D50": d50,
                    "D2": np.interp(2, np.flip(volume), np.flip(dose)),
                    "V20Gy": v20,
                    "V10Gy": v10,
                    "volume_cc": np.random.uniform(5, 200),  # Thể tích giả định
                },
            }

            # Lưu dữ liệu vào cấu trúc để sử dụng lại
            structure["dvh"] = dvh_data
            structure["is_sample_dvh"] = True  # Đánh dấu là dữ liệu mẫu

            # Thông báo debug thành công
            logger.debug(
                f"Đã tạo dữ liệu DVH mẫu cho {structure_name} (loại: {structure_type})"
            )

            return dvh_data
    except Exception as e:
            logger.error(f"Lỗi khi tạo dữ liệu DVH mẫu: {e}")
            import traceback

            logger.debug(traceback.format_exc())

            # Trả về dữ liệu mẫu đơn giản nếu lỗi
            return {
                "dose": list(range(0, 81)),
                "volume": [100] * 41 + [0] * 40,
                "metrics": {
                    "Dmin": 0,
                    "Dmax": 40,
                    "Dmean": 20,
                    "D98": 38,
                    "D95": 39,
                    "D50": 40,
                    "D2": 40,
                    "V20Gy": 100,
                    "V10Gy": 100,
                },
            }
