#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab External Beam Planning cho QuangTPS.

Module này triển khai giao diện Eclipse-like External Beam Planning,
tích hợp các tính năng lập kế hoạch và tính liều vào một tab duy nhất.
Đây là sự kết hợp các tính năng của planning_tab.py và dose_tab.py,
với cải tiến giao diện mô phỏng theo phần mềm Eclipse TPS của Varian.
"""

import os
import sys
import logging
import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import time
from enum import Enum

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QDialog,
    QColorDialog,
    QComboBox,
    QLineEdit,
    QFormLayout,
    QMessageBox,
    QFileDialog,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QProgressDialog,
    QMenu,
    QAction,
    QToolBar,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QToolButton,
    QFrame,
    QScrollArea,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QDateEdit,
    QInputDialog,
    QSizePolicy,
    QProgressBar,
)
from PyQt5.QtGui import QColor, QIcon, QBrush, QPixmap, QImage, QPainter, QPen, QCursor
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QDate

# Import matplotlib for visualization if available
try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib not available for DVH visualization")
    MATPLOTLIB_AVAILABLE = False

# Import QuangTPS modules
try:
    # Import core modules
    from quangtps.core.services import ServiceRegistry
    from quangtps.core.patient import Patient
    from quangtps.database.patient_db import PatientDB
    from quangtps.database.plan_db import PlanDB
    from quangtps.database.structure_db import StructureDB
    from quangtps.database.beam_db import BeamDB
    from quangtps.database.dose_db import DoseDB
    from quangtps.planning.plan import Plan, PlanStatus, PlanType
    from quangtps.planning.beam import Beam
    from quangtps.planning.prescription import Prescription

    # Import planning modules
    from quangtps.treatment.techniques.crt_manager import CRTManager
    from quangtps.treatment.techniques.imrt import IMRT
    from quangtps.treatment.techniques.vmat import VMAT
    from quangtps.treatment.techniques.treatment_technique import TreatmentTechnique

    # Import dose calculation modules
    from quangtps.dose.dose_calculator import DoseCalculator
    from quangtps.dose.dose_grid import DoseGrid

    # Import optimization modules
    from quangtps.optimization.optimization_engine import (
        OptimizationEngine,
        OptimizationParameters,
    )
    from quangtps.optimization.objectives import ObjectiveCollection
    from quangtps.optimization.constraints import ConstraintCollection

    # Import evaluation modules
    from quangtps.evaluation.plan_evaluation import PlanEvaluation
    from quangtps.evaluation.dvh.dvh_calculation import (
        calculate_dvh,
        calculate_dvh_metrics,
    )
    from quangtps.evaluation.dvh.dvh_visualization import plot_dvh

    # Import UI modules
    from quangtps.ui.dialogs.beam_dialog import BeamDialog
    from quangtps.ui.beam_visualization_panel import BeamVisualizationPanel
    from quangtps.ui.dose_visualization_3d import DoseVisualization3D

    # Import MCO-related modules
    from quangtps.optimization.mco.mco_engine import MCOEngine

    # Import additional modules
    from quangtps.ui.visualization_3d import (
        create_3d_visualization_widget,
        DisplayMode,
        ViewOrientation,
    )
    from quangtps.ui.dvh_widget import create_dvh_widget
    from quangtps.ui.eclipse_style_theme import (
        apply_eclipse_theme,
        create_eclipse_widget_style,
    )
    from quangtps.ui import get_colormap_for_display

    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logging.error(f"Error importing QuangTPS modules: {e}")

logger = logging.getLogger(__name__)


class BeamPlanningMode(Enum):
    """Enum cho các chế độ lập kế hoạch chùm tia."""

    FORWARD = "forward"  # Lập kế hoạch thuận
    INVERSE = "inverse"  # Lập kế hoạch ngược
    MULTI_CRITERIA = "mco"  # Tối ưu hóa đa tiêu chí


class ExternalBeamPlanningTab(QWidget):
    """
    Tab External Beam Planning cho QuangTPS với giao diện kiểu Eclipse.

    Tab này tích hợp các tính năng lập kế hoạch và tính liều trong một giao diện
    thống nhất, tương tự như Eclipse TPS của Varian. Bao gồm các tính năng:
    - Quản lý kế hoạch và chùm tia
    - Thiết lập kỹ thuật điều trị (3D CRT, IMRT, VMAT)
    - Tối ưu hóa kế hoạch (Multi-Criteria Optimization)
    - Tính toán liều
    - Phân tích và đánh giá kế hoạch (DVH, metrics, dose visualization)
    """

    # Tín hiệu
    plan_created = pyqtSignal(object)
    plan_updated = pyqtSignal(object)
    plan_deleted = pyqtSignal(str)
    patient_loaded = pyqtSignal(object)
    calculation_started = pyqtSignal()
    calculation_finished = pyqtSignal()
    plan_changed = pyqtSignal(object)  # Phát khi kế hoạch thay đổi
    dose_calculated = pyqtSignal(np.ndarray)  # Phát khi phân bố liều được tính toán
    optimization_started = pyqtSignal()  # Phát khi bắt đầu tối ưu hóa
    optimization_progress = pyqtSignal(int, str)  # Phát khi tiến độ tối ưu hóa thay đổi
    optimization_finished = pyqtSignal(bool, str)  # Phát khi tối ưu hóa kết thúc

    def __init__(self, parent=None):
        """
        Khởi tạo tab External Beam Planning.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        if not HAS_PYQT:
            logger.error(
                "PyQt5 không khả dụng. Không thể khởi tạo ExternalBeamPlanningTab."
            )
            return

        super().__init__(parent)

        # Khởi tạo trạng thái
        self.current_patient = None
        self.current_plan = None
        self.current_beam = None
        self.current_image = None
        self.current_structure_set = None
        self.current_dose_grid = None
        self.structures = {}
        self.dose_grid = None
        self.dose_spacing = None
        self.dose_origin = None
        self.planning_mode = BeamPlanningMode.INVERSE  # Chế độ mặc định

        # Initialize services
        self.service_registry = ServiceRegistry()
        self.plan_db = self.service_registry.get_service("PlanDB")
        self.patient_db = self.service_registry.get_service("PatientDB")
        self.structure_db = self.service_registry.get_service("StructureDB")
        self.beam_db = self.service_registry.get_service("BeamDB")
        self.dose_db = self.service_registry.get_service("DoseDB")

        self.dose_calculator = self.service_registry.get_service("DoseCalculator")
        self.optimization_engine = self.service_registry.get_service(
            "OptimizationEngine"
        )

        # Khởi tạo các managers
        self.crt_manager = CRTManager() if MODULES_AVAILABLE else None

        # Thuật toán tính liều và tối ưu hóa
        self.dose_algorithm = None
        self.optimizer = None

        # Thiết lập UI
        self._init_ui()

        # Kết nối tín hiệu
        self._connect_signals()

    def _init_ui(self):
        """Khởi tạo giao diện tab External Beam Planning."""
        main_layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QToolBar("External Beam Planning Toolbar")
        main_layout.addWidget(toolbar)
        self._setup_toolbar_actions(toolbar)

        # Mode selection (Forward vs Inverse vs MCO)
        mode_layout = QHBoxLayout()
        mode_group = QGroupBox("Chế độ lập kế hoạch")
        mode_layout.addWidget(mode_group)

        mode_group_layout = QHBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["Lập kế hoạch thuận", "Lập kế hoạch ngược", "Tối ưu hóa đa tiêu chí"]
        )
        self.mode_combo.setCurrentIndex(1)  # Inverse planning là mặc định
        mode_group_layout.addWidget(self.mode_combo)

        main_layout.addLayout(mode_layout)

        # Main splitter (chia đôi màn hình)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter, 1)  # Stretch factor = 1

        # Phần bên trái - Cấu hình kế hoạch
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Beam configuration
        beam_config_group = QGroupBox("Cấu hình chùm tia")
        beam_config_layout = QVBoxLayout(beam_config_group)

        # Beam list
        self.beam_table = QTableWidget()
        self.beam_table.setColumnCount(4)
        self.beam_table.setHorizontalHeaderLabels(
            ["Chùm tia", "Góc", "Trọng số", "MLC"]
        )
        self.beam_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        beam_config_layout.addWidget(self.beam_table)

        # Beam action buttons
        beam_actions = QHBoxLayout()
        self.add_beam_btn = QPushButton("Thêm chùm")
        self.edit_beam_btn = QPushButton("Sửa chùm")
        self.remove_beam_btn = QPushButton("Xóa chùm")

        beam_actions.addWidget(self.add_beam_btn)
        beam_actions.addWidget(self.edit_beam_btn)
        beam_actions.addWidget(self.remove_beam_btn)
        beam_config_layout.addLayout(beam_actions)

        left_layout.addWidget(beam_config_group)

        # Structure selection
        structure_group = QGroupBox("Cấu trúc")
        structure_layout = QVBoxLayout(structure_group)

        self.structure_table = QTableWidget()
        self.structure_table.setColumnCount(3)
        self.structure_table.setHorizontalHeaderLabels(["Tên", "Loại", "Hiển thị"])
        self.structure_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        structure_layout.addWidget(self.structure_table)

        left_layout.addWidget(structure_group)

        # Optimization objectives
        objectives_group = QGroupBox("Mục tiêu tối ưu hóa")
        objectives_layout = QVBoxLayout(objectives_group)

        self.objectives_widget = self._create_objectives_widget()
        objectives_layout.addWidget(self.objectives_widget)

        left_layout.addWidget(objectives_group)

        main_splitter.addWidget(left_widget)

        # Phần bên phải - Hiển thị 3D và DVH
        right_widget = QTabWidget()

        # Tab 3D Visualization
        self.vis3d_widget = None
        try:
            self.vis3d_widget = create_3d_visualization_widget()
            if self.vis3d_widget:
                right_widget.addTab(self.vis3d_widget, "3D")
        except Exception as e:
            logger.error(f"Lỗi khi tạo widget hiển thị 3D: {str(e)}")
            self.vis3d_widget = QLabel("Không thể hiển thị 3D")
            right_widget.addTab(self.vis3d_widget, "3D")

        # Tab DVH
        self.dvh_widget = None
        try:
            self.dvh_widget = create_dvh_widget()
            if self.dvh_widget:
                right_widget.addTab(self.dvh_widget, "DVH")
        except Exception as e:
            logger.error(f"Lỗi khi tạo widget DVH: {str(e)}")
            self.dvh_widget = QLabel("Không thể hiển thị DVH")
            right_widget.addTab(self.dvh_widget, "DVH")

        # Tab 2D Views
        slices_widget = QWidget()
        slices_layout = QGridLayout(slices_widget)

        # Placeholder cho slice views
        axial_label = QLabel("Axial View (coming soon)")
        axial_label.setAlignment(Qt.AlignCenter)
        axial_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")

        sagittal_label = QLabel("Sagittal View (coming soon)")
        sagittal_label.setAlignment(Qt.AlignCenter)
        sagittal_label.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ccc;"
        )

        coronal_label = QLabel("Coronal View (coming soon)")
        coronal_label.setAlignment(Qt.AlignCenter)
        coronal_label.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ccc;"
        )

        slices_layout.addWidget(axial_label, 0, 0)
        slices_layout.addWidget(sagittal_label, 0, 1)
        slices_layout.addWidget(coronal_label, 1, 0, 1, 2)

        right_widget.addTab(slices_widget, "2D Views")

        # Tab Plan Evaluation
        evaluation_widget = QWidget()
        evaluation_layout = QVBoxLayout(evaluation_widget)

        # Placeholder cho plan evaluation
        evaluation_label = QLabel("Plan Evaluation (coming soon)")
        evaluation_label.setAlignment(Qt.AlignCenter)
        evaluation_layout.addWidget(evaluation_label)

        right_widget.addTab(evaluation_widget, "Đánh giá kế hoạch")

        main_splitter.addWidget(right_widget)

        # Thiết lập kích thước ban đầu
        main_splitter.setSizes([400, 600])

        # Status bar
        status_bar = QFrame()
        status_bar.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        status_bar_layout = QHBoxLayout(status_bar)
        status_bar_layout.setContentsMargins(5, 2, 5, 2)

        self.status_label = QLabel("Sẵn sàng")
        status_bar_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        status_bar_layout.addWidget(self.progress_bar)

        main_layout.addWidget(status_bar)

        # Áp dụng Eclipse style nếu có thể
        if HAS_QUANGTPS_MODULES:
            try:
                self.setStyleSheet(create_eclipse_widget_style("tab"))
            except:
                pass

    def _setup_toolbar_actions(self, toolbar):
        """Thiết lập các action cho toolbar."""
        # New Plan
        new_plan_action = QAction("Kế hoạch mới", self)
        new_plan_action.triggered.connect(self._on_new_plan)
        toolbar.addAction(new_plan_action)

        # Save Plan
        save_plan_action = QAction("Lưu kế hoạch", self)
        save_plan_action.triggered.connect(self._on_save_plan)
        toolbar.addAction(save_plan_action)

        toolbar.addSeparator()

        # Calculate Dose
        calc_dose_action = QAction("Tính toán liều", self)
        calc_dose_action.triggered.connect(self._on_calculate_dose)
        toolbar.addAction(calc_dose_action)

        # Optimize
        optimize_action = QAction("Tối ưu hóa", self)
        optimize_action.triggered.connect(self._on_optimize)
        toolbar.addAction(optimize_action)

        toolbar.addSeparator()

        # Algorithm selection
        self.algorithm_combo = QComboBox()
        if HAS_QUANGTPS_MODULES:
            try:
                from quangtps.dose.algorithms import get_algorithm_display_names

                self.algorithm_combo.addItems(get_algorithm_display_names())
            except:
                self.algorithm_combo.addItems(
                    ["Monte Carlo", "Pencil Beam", "Collapsed Cone"]
                )
        else:
            self.algorithm_combo.addItems(
                ["Monte Carlo", "Pencil Beam", "Collapsed Cone"]
            )

        toolbar.addWidget(QLabel("Thuật toán: "))
        toolbar.addWidget(self.algorithm_combo)

        toolbar.addSeparator()

        # Export Report
        export_report_action = QAction("Xuất báo cáo", self)
        export_report_action.triggered.connect(self._on_export_report)
        toolbar.addAction(export_report_action)

    def _create_objectives_widget(self):
        """
        Tạo widget chứa danh sách và chỉnh sửa mục tiêu tối ưu hóa.

        Returns
        -------
        QWidget
            Widget chứa bảng mục tiêu và các nút điều khiển
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Bảng mục tiêu
        self.objectives_table = QTableWidget()
        self.objectives_table.setColumnCount(5)
        self.objectives_table.setHorizontalHeaderLabels(
            ["Cấu trúc", "Loại", "Liều/Thể tích", "Giá trị", "Trọng số"]
        )
        self.objectives_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        layout.addWidget(self.objectives_table)

        # Nút điều khiển
        controls_layout = QHBoxLayout()

        self.add_objective_btn = QPushButton("Thêm")
        self.add_objective_btn.clicked.connect(self._on_add_objective)
        controls_layout.addWidget(self.add_objective_btn)

        self.edit_objective_btn = QPushButton("Sửa")
        controls_layout.addWidget(self.edit_objective_btn)

        self.remove_objective_btn = QPushButton("Xóa")
        self.remove_objective_btn.clicked.connect(self._on_remove_objective)
        controls_layout.addWidget(self.remove_objective_btn)

        layout.addLayout(controls_layout)

        # Trạng thái tối ưu hóa
        status_layout = QHBoxLayout()

        self.clear_objectives_btn = QPushButton("Xóa tất cả")
        status_layout.addWidget(self.clear_objectives_btn)

        status_layout.addStretch()

        self.load_protocol_btn = QPushButton("Tải protocol")
        status_layout.addWidget(self.load_protocol_btn)

        layout.addLayout(status_layout)

        return widget

    def _connect_signals(self):
        """Kết nối các tín hiệu và slots."""
        # Kết nối mode combo
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        # Kết nối các tín hiệu tối ưu hóa với UI
        self.optimization_started.connect(lambda: self.progress_bar.setVisible(True))
        self.optimization_progress.connect(
            lambda value, text: self._update_optimization_progress(value, text)
        )
        self.optimization_finished.connect(
            lambda success, msg: self._on_optimization_finished(success, msg)
        )

        # Kết nối với DVH widget nếu có
        if hasattr(self, "dvh_widget") and self.dvh_widget:
            self.dose_calculated.connect(lambda: self._update_dvh_display())

    def _on_new_plan(self):
        """Xử lý khi người dùng tạo kế hoạch mới."""
        # TODO: Implement new plan dialog
        QMessageBox.information(
            self, "Thông báo", "Chức năng tạo kế hoạch mới sẽ được bổ sung sau."
        )

    def _on_calculate_dose(self):
        """Xử lý khi người dùng tính toán phân bố liều."""
        if not self.current_plan:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có kế hoạch nào được tạo.")
            return

        # TODO: Implement dose calculation
        QMessageBox.information(
            self, "Thông báo", "Chức năng tính toán liều đang được phát triển."
        )

    def _on_optimize(self):
        """Xử lý khi người dùng tối ưu hóa kế hoạch."""
        if not self.current_plan:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có kế hoạch nào được tạo.")
            return

        # TODO: Implement optimization
        QMessageBox.information(
            self, "Thông báo", "Chức năng tối ưu hóa đang được phát triển."
        )

        # Hiển thị giả tiến độ tối ưu hóa (cho demo)
        self._fake_optimization_progress()

    def _on_mode_changed(self, index):
        """
        Xử lý khi chế độ lập kế hoạch thay đổi.

        Parameters
        ----------
        index : int
            Chỉ mục của chế độ mới trong combo box
        """
        # Cập nhật chế độ kế hoạch
        if index == 0:
            self.planning_mode = BeamPlanningMode.FORWARD
        elif index == 1:
            self.planning_mode = BeamPlanningMode.INVERSE
        elif index == 2:
            self.planning_mode = BeamPlanningMode.MULTI_CRITERIA
        else:
            return

        # Cập nhật UI dựa trên chế độ mới
        self._update_ui_for_mode()

    def _update_ui_for_mode(self):
        """Cập nhật UI dựa trên chế độ lập kế hoạch hiện tại."""
        if self.planning_mode == BeamPlanningMode.FORWARD:
            # Trong chế độ forward planning, ẩn bảng mục tiêu tối ưu hóa
            if hasattr(self, "objectives_group"):
                self.objectives_group.setVisible(False)
        elif self.planning_mode == BeamPlanningMode.INVERSE:
            # Trong chế độ inverse planning, hiện bảng mục tiêu tối ưu hóa
            if hasattr(self, "objectives_group"):
                self.objectives_group.setVisible(True)
        elif self.planning_mode == BeamPlanningMode.MULTI_CRITERIA:
            # Trong chế độ MCO, hiện bảng mục tiêu tối ưu hóa và nút MCO Navigator
            if hasattr(self, "objectives_group"):
                self.objectives_group.setVisible(True)
            # TODO: Hiển thị nút MCO Navigator

    def _update_optimization_progress(self, value, text):
        """
        Cập nhật hiển thị tiến độ tối ưu hóa.

        Parameters
        ----------
        value : int
            Giá trị tiến độ (0-100)
        text : str
            Mô tả trạng thái
        """
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}% - {text}")
        self.status_label.setText(text)
        QApplication.processEvents()  # Cập nhật UI ngay lập tức

    def _on_optimization_finished(self, success, message):
        """
        Xử lý khi tối ưu hóa kết thúc.

        Parameters
        ----------
        success : bool
            True nếu tối ưu hóa thành công, False nếu thất bại
        message : str
            Thông báo kết quả
        """
        self.progress_bar.setVisible(False)

        if success:
            self.status_label.setText(f"Tối ưu hóa thành công: {message}")

            # Cập nhật hiển thị DVH và 3D
            self._update_dvh_display()

            QMessageBox.information(
                self, "Tối ưu hóa", f"Đã hoàn tất tối ưu hóa: {message}"
            )
        else:
            self.status_label.setText(f"Tối ưu hóa thất bại: {message}")
            QMessageBox.warning(self, "Tối ưu hóa", f"Lỗi khi tối ưu hóa: {message}")

    def _on_save_plan(self):
        """Xử lý khi người dùng lưu kế hoạch hiện tại."""
        # TODO: Implement save plan functionality
        QMessageBox.information(
            self, "Thông báo", "Chức năng lưu kế hoạch sẽ được bổ sung sau."
        )

    def _on_export_report(self):
        """Xử lý khi người dùng xuất báo cáo kế hoạch."""
        # TODO: Implement export report functionality
        QMessageBox.information(
            self, "Thông báo", "Chức năng xuất báo cáo đang được phát triển."
        )

    def _on_add_objective(self):
        """Xử lý khi người dùng thêm mục tiêu tối ưu hóa mới."""
        # Đảm bảo có cấu trúc
        if not self.structures:
            QMessageBox.warning(
                self, "Cảnh báo", "Cần tải cấu trúc trước khi thêm mục tiêu tối ưu."
            )
            return

        # Demo: Thêm một mục tiêu mẫu
        row_count = self.objectives_table.rowCount()
        self.objectives_table.insertRow(row_count)

        # Giả sử có ít nhất 1 cấu trúc
        structure_names = list(self.structures.keys())
        first_structure = structure_names[0] if structure_names else "PTV"

        self.objectives_table.setItem(row_count, 0, QTableWidgetItem(first_structure))
        self.objectives_table.setItem(row_count, 1, QTableWidgetItem("Min Dose"))
        self.objectives_table.setItem(row_count, 2, QTableWidgetItem("Dose"))
        self.objectives_table.setItem(row_count, 3, QTableWidgetItem("50 Gy"))
        self.objectives_table.setItem(row_count, 4, QTableWidgetItem("100"))

    def _on_remove_objective(self):
        """Xử lý khi người dùng xóa mục tiêu tối ưu hóa."""
        # Lấy hàng được chọn
        selected_rows = self.objectives_table.selectedItems()
        if not selected_rows:
            return

        selected_row = selected_rows[0].row()
        self.objectives_table.removeRow(selected_row)

    def set_plan(self, plan):
        """
        Thiết lập kế hoạch cho tab.

        Parameters
        ----------
        plan : Plan
            Đối tượng kế hoạch
        """
        self.current_plan = plan

        # Cập nhật UI
        if plan:
            self.status_label.setText(f"Đã tải kế hoạch: {plan.name}")

            # Cập nhật thông tin kế hoạch trong UI
            self._update_plan_display()

    def set_structures(self, structures):
        """
        Thiết lập danh sách cấu trúc.

        Parameters
        ----------
        structures : Dict[str, Any]
            Dict với khóa là ID cấu trúc và giá trị là đối tượng Structure
        """
        self.structures = structures

        if not hasattr(self, "structure_table"):
            return

        # Xóa bảng cấu trúc hiện tại
        self.structure_table.setRowCount(0)

        # Thêm cấu trúc vào bảng
        for structure_id, structure in structures.items():
            row = self.structure_table.rowCount()
            self.structure_table.insertRow(row)

            # Tên
            self.structure_table.setItem(row, 0, QTableWidgetItem(structure.name))

            # Loại (Target, OAR...)
            structure_type = (
                "Target"
                if "PTV" in structure.name
                or "GTV" in structure.name
                or "CTV" in structure.name
                else "OAR"
            )
            self.structure_table.setItem(row, 1, QTableWidgetItem(structure_type))

            # Checkbox hiển thị
            show_cb = QCheckBox()
            show_cb.setChecked(True)
            self.structure_table.setCellWidget(row, 2, show_cb)

        # Cập nhật widget DVH nếu có
        if hasattr(self, "dvh_widget") and self.dvh_widget:
            self.dvh_widget.set_structures(structures)

        # Cập nhật hiển thị 3D nếu có
        # TODO: Add structures to 3D view

    def set_dose_grid(self, dose_grid, spacing=None, origin=None):
        """
        Thiết lập lưới liều.

        Parameters
        ----------
        dose_grid : np.ndarray
            Mảng 3D chứa dữ liệu liều
        spacing : tuple, optional
            Khoảng cách voxel (mm)
        origin : tuple, optional
            Tọa độ gốc (mm)
        """
        self.dose_grid = dose_grid
        self.dose_spacing = spacing
        self.dose_origin = origin

        # Phát tín hiệu liều đã được tính toán
        self.dose_calculated.emit(dose_grid)

        # Cập nhật hiển thị 3D
        if hasattr(self, "vis3d_widget") and self.vis3d_widget:
            self.vis3d_widget.set_dose_data(dose_grid, spacing, origin)

        # Cập nhật DVH
        self._update_dvh_display()

    def _update_plan_display(self):
        """Cập nhật hiển thị thông tin kế hoạch."""
        if not self.current_plan:
            return

        # Cập nhật bảng chùm tia
        if hasattr(self, "beam_table"):
            self.beam_table.setRowCount(0)

            # Thêm thông tin các chùm tia
            if hasattr(self.current_plan, "beams"):
                for i, beam in enumerate(self.current_plan.beams):
                    row = self.beam_table.rowCount()
                    self.beam_table.insertRow(row)

                    self.beam_table.setItem(row, 0, QTableWidgetItem(f"Beam {i + 1}"))
                    self.beam_table.setItem(
                        row, 1, QTableWidgetItem(f"{beam.gantry_angle:.1f}°")
                    )
                    self.beam_table.setItem(
                        row, 2, QTableWidgetItem(f"{beam.weight:.2f}")
                    )
                    self.beam_table.setItem(
                        row,
                        3,
                        QTableWidgetItem("Yes" if hasattr(beam, "mlc") else "No"),
                    )

    def _update_dvh_display(self):
        """Cập nhật hiển thị DVH sau khi tính liều hoặc tối ưu hóa."""
        if (
            not hasattr(self, "dvh_widget")
            or not self.dvh_widget
            or not self.dose_grid is not None
        ):
            return

        # Cập nhật DVH cho tất cả cấu trúc
        self.dvh_widget.set_dose_grid(
            self.dose_grid, self.dose_spacing, self.dose_origin
        )
        self.dvh_widget.calculate_and_display_dvh()

    def _fake_optimization_progress(self):
        """Giả tiến độ tối ưu hóa cho mục đích demo."""
        # Thông báo bắt đầu
        self.optimization_started.emit()

        # Cập nhật tiến độ
        for i in range(101):
            if i < 20:
                message = "Đang khởi tạo tối ưu hóa..."
            elif i < 40:
                message = "Đang tính toán ma trận liều..."
            elif i < 70:
                message = "Đang tối ưu hóa trọng số chùm tia..."
            elif i < 90:
                message = "Đang tinh chỉnh kết quả..."
            else:
                message = "Đang hoàn tất tính toán..."

            self.optimization_progress.emit(i, message)
            QApplication.processEvents()
            time.sleep(0.05)  # Giả lập thời gian tính toán

        # Kết thúc tối ưu hóa
        self.optimization_finished.emit(True, "Đã tối ưu hóa kế hoạch thành công")

        # Tạo dữ liệu giả cho hiển thị kết quả
        self._create_fake_dose_grid()

    def _create_fake_dose_grid(self):
        """Tạo dữ liệu phân bố liều giả cho mục đích demo."""
        # Tạo mảng 3D đơn giản (100x100x100)
        grid_size = 100
        dose_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

        # Tạo phân bố liều giả dạng Gaussian
        x = np.linspace(-3, 3, grid_size)
        y = np.linspace(-3, 3, grid_size)
        z = np.linspace(-3, 3, grid_size)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Tạo 2 chùm tia đối diện
        # Chùm 1: Từ trục X dương
        beam1 = np.exp(-(Y**2 + Z**2) / 0.5) * (X > -2)

        # Chùm 2: Từ trục X âm
        beam2 = np.exp(-(Y**2 + Z**2) / 0.5) * (X < 2)

        # Kết hợp các chùm
        dose_grid = (beam1 + beam2) * 70.0  # Liều tối đa 70Gy

        # Thiết lập thông tin không gian
        spacing = (2.0, 2.0, 2.0)  # mm
        origin = (-100.0, -100.0, -100.0)  # mm

        # Cập nhật liều
        self.set_dose_grid(dose_grid, spacing, origin)
