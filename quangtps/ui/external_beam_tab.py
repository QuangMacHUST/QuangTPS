#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
External Beam Planning tab cho QuangTPS.

Tab này cung cấp giao diện và chức năng cho lập kế hoạch xạ trị chùm tia ngoài
với tích hợp hiển thị 3D và các công cụ tối ưu hóa theo kiểu Eclipse.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)

# Thử import PyQt5
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QSplitter,
        QTabWidget,
        QGroupBox,
        QLabel,
        QPushButton,
        QComboBox,
        QCheckBox,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QMenu,
        QAction,
        QToolBar,
        QStatusBar,
        QFrame,
        QSpacerItem,
        QSizePolicy,
        QToolButton,
        QFileDialog,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QObject
    from PyQt5.QtGui import QIcon, QColor, QFont, QPalette

    HAS_PYQT = True
except ImportError:
    logger.warning("PyQt5 không khả dụng. Không thể tạo External Beam Planning tab.")
    HAS_PYQT = False

# Import module 3D visualization nếu có
try:
    from quangtps.ui.visualization_3d import (
        Visualization3DWidget,
        DisplayMode,
        ViewOrientation,
        VisualizationMode,
        create_3d_visualization_widget,
    )

    HAS_3D_VIS = True
except ImportError:
    logger.warning(
        "Module visualization_3d không khả dụng. Chức năng hiển thị 3D sẽ bị hạn chế."
    )
    HAS_3D_VIS = False

# Import các module khác của QuangTPS
try:
    from quangtps.ui.dose_volume_histogram import DVHWidget
    from quangtps.ui.beam_configuration_widget import BeamConfigWidget
    from quangtps.ui.structure_selection_widget import StructureSelectionWidget
    from quangtps.ui.dosimetric_table import DosimetricTableWidget
    from quangtps.ui.eclipse_style_theme import get_eclipse_colormap

    HAS_UI_MODULES = True
except ImportError:
    logger.warning(
        "Không thể import đầy đủ các module UI. Một số chức năng sẽ bị hạn chế."
    )
    HAS_UI_MODULES = False


class BeamPlanningMode(Enum):
    """Enum cho các chế độ lập kế hoạch chùm tia."""

    FORWARD = "forward"  # Lập kế hoạch thuận
    INVERSE = "inverse"  # Lập kế hoạch ngược
    MULTI_CRITERIA = "mco"  # Tối ưu hóa đa tiêu chí


class ExternalBeamPlanningTab(QWidget):
    """
    Tab lập kế hoạch xạ trị chùm tia ngoài.

    Cung cấp giao diện đồ họa cho việc lập kế hoạch và hiển thị kết quả.
    """

    # Tín hiệu
    plan_changed = pyqtSignal(object)  # Phát khi kế hoạch thay đổi
    dose_calculated = pyqtSignal(np.ndarray)  # Phát khi phân bố liều được tính toán
    optimization_started = pyqtSignal()  # Phát khi bắt đầu tối ưu hóa
    optimization_progress = pyqtSignal(int, str)  # Phát khi tiến độ tối ưu hóa thay đổi
    optimization_finished = pyqtSignal(bool, str)  # Phát khi tối ưu hóa kết thúc

    def __init__(self, parent=None):
        """
        Khởi tạo tab lập kế hoạch chùm tia ngoài.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, mặc định là None
        """
        if not HAS_PYQT:
            logger.error(
                "PyQt5 không khả dụng. Không thể tạo External Beam Planning tab."
            )
            return None

        super().__init__(parent)

        # Dữ liệu nội bộ
        self.current_plan = None  # Kế hoạch hiện tại
        self.current_beam_set = None  # Tập chùm tia hiện tại
        self.current_structures = []  # Danh sách các cấu trúc hiện tại
        self.current_dose_grid = None  # Lưới liều hiện tại
        self.current_mode = BeamPlanningMode.FORWARD  # Chế độ lập kế hoạch mặc định

        # Thiết lập giao diện người dùng
        self._setup_ui()

        # Kết nối các tín hiệu và khe
        self._connect_signals()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng cho tab."""
        try:
            # Layout chính là VBoxLayout
            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(2, 2, 2, 2)
            main_layout.setSpacing(2)

            # Tạo toolbar
            toolbar = QToolBar("External Beam Planning Toolbar")
            main_layout.addWidget(toolbar)

            # Tạo splitter dọc
            self.main_splitter = QSplitter(Qt.Vertical)
            main_layout.addWidget(self.main_splitter, 1)  # Stretch = 1

            # Tạo splitter ngang cho phần trên
            self.top_splitter = QSplitter(Qt.Horizontal)
            self.main_splitter.addWidget(self.top_splitter)

            # Phần bên trái: Cấu hình và danh sách chùm tia
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            left_layout.setContentsMargins(0, 0, 0, 0)

            # Tạo tab widget cho các tùy chọn
            self.left_tab_widget = QTabWidget()
            left_layout.addWidget(self.left_tab_widget)

            # Tab cấu hình chùm tia
            if HAS_UI_MODULES:
                # Sử dụng widget có sẵn nếu có thể
                self.beam_config_widget = BeamConfigWidget()
                self.left_tab_widget.addTab(
                    self.beam_config_widget, "Beam Configuration"
                )
            else:
                # Tạo placeholder nếu không có module
                beam_placeholder = QWidget()
                beam_layout = QVBoxLayout(beam_placeholder)
                beam_layout.addWidget(QLabel("Beam Configuration module not available"))
                self.left_tab_widget.addTab(beam_placeholder, "Beam Configuration")

            # Tab lựa chọn cấu trúc
            if HAS_UI_MODULES:
                # Sử dụng widget có sẵn nếu có thể
                self.structure_widget = StructureSelectionWidget()
                self.left_tab_widget.addTab(self.structure_widget, "Structures")
            else:
                # Tạo placeholder nếu không có module
                structure_placeholder = QWidget()
                structure_layout = QVBoxLayout(structure_placeholder)
                structure_layout.addWidget(
                    QLabel("Structure Selection module not available")
                )
                self.left_tab_widget.addTab(structure_placeholder, "Structures")

            # Tab mục tiêu và ràng buộc
            objectives_widget = self._create_objectives_widget()
            self.left_tab_widget.addTab(objectives_widget, "Objectives")

            self.top_splitter.addWidget(left_widget)

            # Phần chính giữa: Hiển thị 3D và 2D
            center_widget = QWidget()
            center_layout = QVBoxLayout(center_widget)
            center_layout.setContentsMargins(0, 0, 0, 0)

            # Tạo tab widget cho các chế độ hiển thị
            self.view_tab_widget = QTabWidget()
            center_layout.addWidget(self.view_tab_widget)

            # Tab hiển thị 3D
            if HAS_3D_VIS:
                # Sử dụng widget 3D visualization nếu có thể
                self.vis_3d_widget = create_3d_visualization_widget()
                if self.vis_3d_widget:
                    self.view_tab_widget.addTab(self.vis_3d_widget, "3D View")
                else:
                    # Tạo placeholder nếu không thể tạo widget
                    vis3d_placeholder = QWidget()
                    vis3d_layout = QVBoxLayout(vis3d_placeholder)
                    vis3d_layout.addWidget(
                        QLabel("3D Visualization could not be created")
                    )
                    self.view_tab_widget.addTab(vis3d_placeholder, "3D View")
            else:
                # Tạo placeholder nếu không có module
                vis3d_placeholder = QWidget()
                vis3d_layout = QVBoxLayout(vis3d_placeholder)
                vis3d_layout.addWidget(QLabel("3D Visualization module not available"))
                self.view_tab_widget.addTab(vis3d_placeholder, "3D View")

            # Tab hiển thị 2D
            view_2d_widget = QWidget()
            view_2d_layout = QVBoxLayout(view_2d_widget)
            view_2d_layout.addWidget(QLabel("2D View - Coming soon"))
            self.view_tab_widget.addTab(view_2d_widget, "2D View")

            self.top_splitter.addWidget(center_widget)

            # Widget phía dưới: DVH và bảng thông số liều
            bottom_widget = QWidget()
            bottom_layout = QVBoxLayout(bottom_widget)
            bottom_layout.setContentsMargins(0, 0, 0, 0)

            # Tạo tab widget cho các chức năng đánh giá
            self.bottom_tab_widget = QTabWidget()
            bottom_layout.addWidget(self.bottom_tab_widget)

            # Tab DVH
            if HAS_UI_MODULES:
                # Sử dụng widget DVH nếu có thể
                self.dvh_widget = DVHWidget()
                self.bottom_tab_widget.addTab(self.dvh_widget, "Dose Volume Histogram")
            else:
                # Tạo placeholder nếu không có module
                dvh_placeholder = QWidget()
                dvh_layout = QVBoxLayout(dvh_placeholder)
                dvh_layout.addWidget(QLabel("DVH module not available"))
                self.bottom_tab_widget.addTab(dvh_placeholder, "Dose Volume Histogram")

            # Tab bảng thông số liều
            if HAS_UI_MODULES:
                # Sử dụng widget bảng liều nếu có thể
                self.dose_table = DosimetricTableWidget()
                self.bottom_tab_widget.addTab(self.dose_table, "Dose Statistics")
            else:
                # Tạo placeholder nếu không có module
                dose_table_placeholder = QWidget()
                dose_table_layout = QVBoxLayout(dose_table_placeholder)
                dose_table_layout.addWidget(QLabel("Dose Table module not available"))
                self.bottom_tab_widget.addTab(dose_table_placeholder, "Dose Statistics")

            self.main_splitter.addWidget(bottom_widget)

            # Thêm thanh trạng thái
            self.status_bar = QStatusBar()
            self.status_bar.setSizeGripEnabled(False)
            main_layout.addWidget(self.status_bar)

            # Thêm các action cho toolbar
            self._setup_toolbar_actions(toolbar)

            # Thiết lập kích thước splitter
            self.top_splitter.setSizes(
                [int(self.width() * 0.3), int(self.width() * 0.7)]
            )
            self.main_splitter.setSizes(
                [int(self.height() * 0.7), int(self.height() * 0.3)]
            )

            logger.info("Đã thiết lập thành công giao diện External Beam Planning tab")

        except Exception as e:
            logger.error(
                f"Lỗi khi thiết lập giao diện External Beam Planning tab: {str(e)}"
            )

    def _setup_toolbar_actions(self, toolbar):
        """
        Thiết lập các action cho toolbar.

        Parameters
        ----------
        toolbar : QToolBar
            Toolbar cần thêm các action
        """
        try:
            # Action tạo kế hoạch mới
            new_plan_action = QAction("New Plan", self)
            new_plan_action.triggered.connect(self._on_new_plan)
            toolbar.addAction(new_plan_action)

            # Action tính toán liều
            calculate_dose_action = QAction("Calculate Dose", self)
            calculate_dose_action.triggered.connect(self._on_calculate_dose)
            toolbar.addAction(calculate_dose_action)

            # Action tối ưu hóa
            optimize_action = QAction("Optimize", self)
            optimize_action.triggered.connect(self._on_optimize)
            toolbar.addAction(optimize_action)

            # Thêm separator
            toolbar.addSeparator()

            # Combobox chế độ lập kế hoạch
            self.mode_combo = QComboBox()
            self.mode_combo.addItem("Forward Planning", BeamPlanningMode.FORWARD.value)
            self.mode_combo.addItem("Inverse Planning", BeamPlanningMode.INVERSE.value)
            self.mode_combo.addItem(
                "Multi-Criteria Optimization", BeamPlanningMode.MULTI_CRITERIA.value
            )
            self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

            toolbar.addWidget(QLabel("Planning Mode: "))
            toolbar.addWidget(self.mode_combo)

            # Thêm separator
            toolbar.addSeparator()

            # Action lưu kế hoạch
            save_plan_action = QAction("Save Plan", self)
            save_plan_action.triggered.connect(self._on_save_plan)
            toolbar.addAction(save_plan_action)

            # Action xuất báo cáo
            export_report_action = QAction("Export Report", self)
            export_report_action.triggered.connect(self._on_export_report)
            toolbar.addAction(export_report_action)

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập actions cho toolbar: {str(e)}")

    def _create_objectives_widget(self):
        """
        Tạo widget mục tiêu và ràng buộc cho tối ưu hóa.

        Returns
        -------
        QWidget
            Widget chứa các tùy chọn mục tiêu và ràng buộc
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Bảng mục tiêu
        obj_group = QGroupBox("Clinical Objectives")
        obj_layout = QVBoxLayout(obj_group)

        # Tạo bảng mục tiêu
        self.objectives_table = QTableWidget(0, 5)  # 0 hàng, 5 cột
        self.objectives_table.setHorizontalHeaderLabels(
            ["Structure", "Type", "Dose (%)", "Volume (%)", "Weight"]
        )
        self.objectives_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        obj_layout.addWidget(self.objectives_table)

        # Các nút để thêm/xóa mục tiêu
        buttons_layout = QHBoxLayout()

        add_objective_button = QPushButton("Add Objective")
        add_objective_button.clicked.connect(self._on_add_objective)
        buttons_layout.addWidget(add_objective_button)

        remove_objective_button = QPushButton("Remove Objective")
        remove_objective_button.clicked.connect(self._on_remove_objective)
        buttons_layout.addWidget(remove_objective_button)

        obj_layout.addLayout(buttons_layout)
        layout.addWidget(obj_group)

        # Tùy chọn tối ưu hóa
        opt_group = QGroupBox("Optimization Options")
        opt_layout = QVBoxLayout(opt_group)

        # Các tùy chọn tối ưu hóa
        max_iter_layout = QHBoxLayout()
        max_iter_layout.addWidget(QLabel("Max Iterations:"))
        self.max_iter_spin = QComboBox()
        self.max_iter_spin.addItems(["50", "100", "200", "500", "1000"])
        self.max_iter_spin.setCurrentIndex(1)  # Mặc định 100
        max_iter_layout.addWidget(self.max_iter_spin)
        opt_layout.addLayout(max_iter_layout)

        convergence_layout = QHBoxLayout()
        convergence_layout.addWidget(QLabel("Convergence:"))
        self.convergence_combo = QComboBox()
        self.convergence_combo.addItems(["0.1%", "0.01%", "0.001%"])
        self.convergence_combo.setCurrentIndex(1)  # Mặc định 0.01%
        convergence_layout.addWidget(self.convergence_combo)
        opt_layout.addLayout(convergence_layout)

        # Checkbox cho các tùy chọn khác
        self.normalize_checkbox = QCheckBox("Auto-normalize to prescription")
        self.normalize_checkbox.setChecked(True)
        opt_layout.addWidget(self.normalize_checkbox)

        self.hot_spot_checkbox = QCheckBox("Control hot spots")
        self.hot_spot_checkbox.setChecked(True)
        opt_layout.addWidget(self.hot_spot_checkbox)

        layout.addWidget(opt_group)

        # Thêm spacer
        layout.addStretch(1)

        return widget

    def _connect_signals(self):
        """Kết nối các tín hiệu và khe."""
        try:
            # Kết nối tín hiệu từ các widget con
            if hasattr(self, "beam_config_widget") and self.beam_config_widget:
                # Kết nối tín hiệu beam_changed
                pass

            if hasattr(self, "structure_widget") and self.structure_widget:
                # Kết nối tín hiệu structure_selection_changed
                pass

            if hasattr(self, "dvh_widget") and self.dvh_widget:
                # Kết nối tín hiệu structures_selected
                pass

            if hasattr(self, "vis_3d_widget") and self.vis_3d_widget:
                # Kết nối tín hiệu view_changed và display_mode_changed
                pass

        except Exception as e:
            logger.error(f"Lỗi khi kết nối tín hiệu: {str(e)}")

    def _on_new_plan(self):
        """Xử lý khi người dùng yêu cầu tạo kế hoạch mới."""
        logger.info("Tạo kế hoạch mới")
        self.status_bar.showMessage("Creating new plan...")

        # TODO: Thêm mã để hiển thị hộp thoại tạo kế hoạch mới
        # và xử lý việc tạo kế hoạch

    def _on_calculate_dose(self):
        """Xử lý khi người dùng yêu cầu tính toán liều."""
        logger.info("Bắt đầu tính toán liều")
        self.status_bar.showMessage("Calculating dose...")

        # TODO: Thêm mã để hiển thị hộp thoại tính toán liều
        # và xử lý việc tính toán liều

    def _on_optimize(self):
        """Xử lý khi người dùng yêu cầu tối ưu hóa."""
        logger.info("Bắt đầu tối ưu hóa kế hoạch")
        self.status_bar.showMessage("Optimizing plan...")

        # TODO: Thêm mã để thực hiện tối ưu hóa kế hoạch
        # và cập nhật giao diện người dùng

    def _on_mode_changed(self, index):
        """
        Xử lý khi chế độ lập kế hoạch thay đổi.

        Parameters
        ----------
        index : int
            Chỉ mục của chế độ mới trong combobox
        """
        mode_value = self.mode_combo.itemData(index)
        try:
            self.current_mode = BeamPlanningMode(mode_value)
            logger.info(f"Chế độ lập kế hoạch chuyển sang: {self.current_mode.name}")

            # Cập nhật giao diện người dùng dựa trên chế độ mới
            if self.current_mode == BeamPlanningMode.FORWARD:
                # Hiển thị các tùy chọn lập kế hoạch thuận
                pass
            elif self.current_mode == BeamPlanningMode.INVERSE:
                # Hiển thị các tùy chọn lập kế hoạch ngược
                pass
            elif self.current_mode == BeamPlanningMode.MULTI_CRITERIA:
                # Hiển thị các tùy chọn tối ưu hóa đa tiêu chí
                pass

        except ValueError as e:
            logger.error(f"Giá trị chế độ không hợp lệ: {str(e)}")

    def _on_save_plan(self):
        """Xử lý khi người dùng yêu cầu lưu kế hoạch."""
        logger.info("Lưu kế hoạch")
        self.status_bar.showMessage("Saving plan...")

        # TODO: Thêm mã để hiển thị hộp thoại lưu kế hoạch
        # và xử lý việc lưu kế hoạch

    def _on_export_report(self):
        """Xử lý khi người dùng yêu cầu xuất báo cáo."""
        logger.info("Xuất báo cáo kế hoạch")
        self.status_bar.showMessage("Exporting report...")

        # TODO: Thêm mã để hiển thị hộp thoại xuất báo cáo
        # và xử lý việc xuất báo cáo

    def _on_add_objective(self):
        """Xử lý khi người dùng yêu cầu thêm mục tiêu."""
        # Thêm hàng mới vào bảng mục tiêu
        row_position = self.objectives_table.rowCount()
        self.objectives_table.insertRow(row_position)

        # Thêm combobox cho cấu trúc
        structure_combo = QComboBox()
        if hasattr(self, "current_structures") and self.current_structures:
            for structure in self.current_structures:
                structure_combo.addItem(structure.name, structure.id)
        self.objectives_table.setCellWidget(row_position, 0, structure_combo)

        # Thêm combobox cho loại mục tiêu
        type_combo = QComboBox()
        type_combo.addItems(
            [
                "Maximum Dose",
                "Minimum Dose",
                "Mean Dose",
                "Maximum DVH",
                "Minimum DVH",
                "Fall-off",
            ]
        )
        self.objectives_table.setCellWidget(row_position, 1, type_combo)

        # Thêm các giá trị mặc định
        dose_item = QTableWidgetItem("100.0")
        self.objectives_table.setItem(row_position, 2, dose_item)

        volume_item = QTableWidgetItem("95.0")
        self.objectives_table.setItem(row_position, 3, volume_item)

        weight_item = QTableWidgetItem("100.0")
        self.objectives_table.setItem(row_position, 4, weight_item)

    def _on_remove_objective(self):
        """Xử lý khi người dùng yêu cầu xóa mục tiêu."""
        # Lấy các hàng được chọn
        selected_rows = set()
        for item in self.objectives_table.selectedItems():
            selected_rows.add(item.row())

        # Xóa các hàng được chọn, từ dưới lên để tránh thay đổi chỉ mục
        for row in sorted(selected_rows, reverse=True):
            self.objectives_table.removeRow(row)

    def set_plan(self, plan):
        """
        Thiết lập kế hoạch hiện tại.

        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị cần hiển thị
        """
        self.current_plan = plan
        # TODO: Cập nhật giao diện cho kế hoạch mới

        # Phát tín hiệu kế hoạch đã thay đổi
        self.plan_changed.emit(plan)

    def set_structures(self, structures):
        """
        Thiết lập danh sách cấu trúc.

        Parameters
        ----------
        structures : List[Structure]
            Danh sách các cấu trúc cần hiển thị
        """
        self.current_structures = structures
        # TODO: Cập nhật giao diện cho danh sách cấu trúc mới

        # Hiển thị cấu trúc trong 3D view nếu có
        if hasattr(self, "vis_3d_widget") and self.vis_3d_widget:
            # TODO: Hiển thị cấu trúc trong 3D view
            pass

    def set_dose_grid(self, dose_grid, spacing=None, origin=None):
        """
        Thiết lập phân bố liều hiện tại.

        Parameters
        ----------
        dose_grid : np.ndarray
            Mảng 3D chứa dữ liệu phân bố liều
        spacing : Tuple[float, float, float], optional
            Khoảng cách giữa các điểm liều (mm), mặc định là None
        origin : Tuple[float, float, float], optional
            Điểm gốc của phân bố liều, mặc định là None
        """
        self.current_dose_grid = dose_grid

        # Cập nhật hiển thị liều
        # TODO: Cập nhật các hiển thị và đồ thị với phân bố liều mới

        # Hiển thị phân bố liều trong 3D view nếu có
        if (
            hasattr(self, "vis_3d_widget")
            and self.vis_3d_widget
            and dose_grid is not None
        ):
            if spacing is None:
                spacing = (1.0, 1.0, 1.0)  # Mặc định 1mm
            if origin is None:
                origin = (0.0, 0.0, 0.0)  # Mặc định (0,0,0)

            self.vis_3d_widget.set_dose_data(dose_grid, spacing, origin)

        # Phát tín hiệu phân bố liều đã được tính toán
        self.dose_calculated.emit(dose_grid)
