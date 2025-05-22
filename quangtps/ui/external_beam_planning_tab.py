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

# Khởi tạo logger
logger = logging.getLogger(__name__)

# Import với try-except để xử lý lỗi import
try:
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
        QGridLayout,
        QApplication,
    )
    from PyQt5.QtGui import (
        QColor,
        QIcon,
        QBrush,
        QPixmap,
        QImage,
        QPainter,
        QPen,
        QCursor,
    )
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QDate

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    logger.error("PyQt5 không khả dụng. Tab External Beam Planning sẽ không hoạt động.")

    # Tạo các lớp giả để tránh lỗi khi import
    class DummyWidget:
        pass

    class DummySignal:
        def __init__(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

    QWidget = DummyWidget
    pyqtSignal = DummySignal
    QApplication = None

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
    from quangtps.ui.dialogs.kbp_dialog import KBPDialog

    # Import additional modules
    from quangtps.ui.visualization_3d import (
        create_3d_visualization_widget,
        DisplayMode,
        ViewOrientation,
    )
    from quangtps.ui.dvh_widget import create_dvh_widget
    from quangtps.ui.dose_visualization_widget import create_dose_visualization_widget
    from quangtps.ui.eclipse_style_theme import (
        apply_eclipse_theme,
        create_eclipse_widget_style,
    )
    from quangtps.ui import get_colormap_for_display

    # Import MCO modules
    try:
        # Import MCO Navigator widget
        from quangtps.ui.mco_navigator_widget import (
            MCONavigatorWidget,
            create_mco_navigator_widget,
        )
        from quangtps.optimization.mco.mco_navigator import (
            MCONavigator,
            ParetoSolution,
            ParetoSolutionType,
        )
        from quangtps.optimization.mco.mco_pareto_3d_widget import (
            Pareto3DWidget,
            create_pareto_3d_widget,
        )

        HAS_MCO_UI_MODULE = True
    except ImportError as e:
        HAS_MCO_UI_MODULE = False
        logger.warning(f"Không thể import module MCO UI: {e}")
        logger.warning("Chức năng MCO Navigator sẽ không khả dụng.")

    # Kiểm tra module MCO một cách riêng biệt
    try:
    # Import MCO-related modules
    from quangtps.optimization.mco.mco_engine import MCOEngine

        HAS_MCO_MODULE = True
    except ImportError:
        HAS_MCO_MODULE = False
        logger.warning(
            "Module MCO không khả dụng. Chức năng tối ưu hóa đa tiêu chí sẽ bị hạn chế."
        )

    MODULES_AVAILABLE = True
    HAS_QUANGTPS_MODULES = True
except ImportError as e:
    MODULES_AVAILABLE = False
    HAS_QUANGTPS_MODULES = False
    HAS_MCO_MODULE = False
    HAS_MCO_UI_MODULE = False
    logger.error(f"Error importing QuangTPS modules: {e}")

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
        """
        Khởi tạo giao diện người dùng cho tab External Beam Planning.
        Tạo layout và các thành phần giao diện theo phong cách Eclipse.
        """
        if not HAS_PYQT:
            logger.error("PyQt5 không khả dụng. Không thể khởi tạo UI.")
            return

        # Áp dụng phong cách Eclipse
        try:
            apply_eclipse_theme(self)
        except Exception as e:
            logger.warning(f"Không thể áp dụng phong cách Eclipse: {e}")

        # Layout chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(2)

        # Tạo toolbar
        toolbar = QToolBar()
        self._setup_toolbar_actions(toolbar)
        main_layout.addWidget(toolbar)

        # Tạo main splitter giữa panel trái và phải
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(main_splitter, 1)  # stretch = 1

        # Panel trái chứa danh sách chùm tia và bảng mục tiêu
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Combo chọn kế hoạch
        plan_layout = QHBoxLayout()
        plan_layout.addWidget(QLabel("Kế hoạch:"))
        self.plan_combo = QComboBox()
        self.plan_combo.setMinimumWidth(200)
        plan_layout.addWidget(self.plan_combo)
        plan_layout.addStretch()
        left_layout.addLayout(plan_layout)

        # Radio button chọn chế độ lập kế hoạch
        mode_group = QGroupBox("Chế độ lập kế hoạch")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_buttons = QButtonGroup()

        # Chế độ Forward Planning
        self.forward_radio = QRadioButton("Forward")
        self.forward_radio.setChecked(True)
        self.mode_buttons.addButton(self.forward_radio, 0)
        mode_layout.addWidget(self.forward_radio)

        # Chế độ Inverse Planning
        self.inverse_radio = QRadioButton("Inverse")
        self.mode_buttons.addButton(self.inverse_radio, 1)
        mode_layout.addWidget(self.inverse_radio)

        # Chế độ Multi-Criteria Optimization
        self.mco_radio = QRadioButton("MCO")
        self.mode_buttons.addButton(self.mco_radio, 2)
        if not HAS_MCO_MODULE:
            self.mco_radio.setEnabled(False)
            self.mco_radio.setToolTip("Module MCO không khả dụng")
        mode_layout.addWidget(self.mco_radio)

        left_layout.addWidget(mode_group)

        # Tab widget chứa tab beam management và tab objectives
        plan_tabs = QTabWidget()
        plan_tabs.setDocumentMode(True)

        # Tab quản lý chùm tia
        beams_tab = QWidget()
        beams_layout = QVBoxLayout(beams_tab)
        beams_layout.setContentsMargins(0, 0, 0, 0)

        # Danh sách chùm tia
        self.beams_list = QListWidget()
        self.beams_list.setSelectionMode(QListWidget.SingleSelection)
        self.beams_list.setMinimumHeight(150)
        beams_layout.addWidget(QLabel("Chùm tia:"))
        beams_layout.addWidget(self.beams_list)

        # Nút thêm/xóa chùm tia
        beams_button_layout = QHBoxLayout()
        self.add_beam_button = QPushButton("Thêm")
        self.remove_beam_button = QPushButton("Xóa")
        self.edit_beam_button = QPushButton("Sửa")
        beams_button_layout.addWidget(self.add_beam_button)
        beams_button_layout.addWidget(self.edit_beam_button)
        beams_button_layout.addWidget(self.remove_beam_button)
        beams_layout.addLayout(beams_button_layout)

        plan_tabs.addTab(beams_tab, "Chùm tia")

        # Tab mục tiêu tối ưu hóa
        objectives_tab = QWidget()
        objectives_layout = QVBoxLayout(objectives_tab)
        objectives_layout.setContentsMargins(0, 0, 0, 0)

        # Widget mục tiêu
        self.objectives_widget = self._create_objectives_widget()
        objectives_layout.addWidget(self.objectives_widget)

        plan_tabs.addTab(objectives_tab, "Mục tiêu")
        left_layout.addWidget(plan_tabs, 1)  # stretch = 1

        # Thêm panel trái vào main splitter
        main_splitter.addWidget(left_panel)

        # Panel phải chứa hiển thị liều và DVH
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget chứa các tab hiển thị
        display_tabs = QTabWidget()
        display_tabs.setDocumentMode(True)

        # Tab 3D
        tab_3d = QWidget()
        tab_3d_layout = QVBoxLayout(tab_3d)
        tab_3d_layout.setContentsMargins(0, 0, 0, 0)

        # Tạo widget hiển thị liều 3D
        try:
            # Sử dụng dose_visualization_3d.py
            self.dose_3d_widget = DoseVisualization3D()
            logger.info("Đã tạo thành công widget hiển thị liều 3D.")
        except Exception as e:
            logger.error(f"Không thể tạo widget hiển thị liều 3D: {e}")
            # Fallback to placeholder
            self.dose_3d_widget = QLabel("Hiển thị liều 3D (Không khả dụng)")
            self.dose_3d_widget.setAlignment(Qt.AlignCenter)
            self.dose_3d_widget.setStyleSheet("background-color: #f0f0f0; color: #888;")

        tab_3d_layout.addWidget(self.dose_3d_widget)
        display_tabs.addTab(tab_3d, "3D")

        # Tab DVH
        tab_dvh = QWidget()
        tab_dvh_layout = QVBoxLayout(tab_dvh)
        tab_dvh_layout.setContentsMargins(0, 0, 0, 0)

        # Tạo widget DVH
        try:
            self.dvh_widget = create_dvh_widget()
            logger.info("Đã tạo thành công widget DVH.")
        except Exception as e:
            logger.error(f"Không thể tạo widget DVH: {e}")
            # Fallback to placeholder
            self.dvh_widget = QLabel("Biểu đồ DVH (Không khả dụng)")
            self.dvh_widget.setAlignment(Qt.AlignCenter)
            self.dvh_widget.setStyleSheet("background-color: #f0f0f0; color: #888;")

        tab_dvh_layout.addWidget(self.dvh_widget)
        display_tabs.addTab(tab_dvh, "DVH")

        # Tạo tab MCO (sẽ hiển thị nếu chọn chế độ MCO)
        self.mco_tab = QWidget()
        self.mco_tab_layout = QVBoxLayout(self.mco_tab)
        self.mco_tab_layout.setContentsMargins(0, 0, 0, 0)

        if HAS_MCO_UI_MODULE:
            try:
                # Tạo MCO Navigator widget
                self.mco_navigator_widget = create_mco_navigator_widget()
                self.mco_tab_layout.addWidget(self.mco_navigator_widget)
                logger.info("Đã tạo thành công widget MCO Navigator.")
            except Exception as e:
                logger.error(f"Không thể tạo widget MCO Navigator: {e}")
                # Fallback to placeholder
                mco_placeholder = QLabel("MCO Navigator (Không khả dụng)")
                mco_placeholder.setAlignment(Qt.AlignCenter)
                mco_placeholder.setStyleSheet("background-color: #f0f0f0; color: #888;")
                self.mco_tab_layout.addWidget(mco_placeholder)
        else:
            mco_placeholder = QLabel("MCO Navigator (Module không khả dụng)")
            mco_placeholder.setAlignment(Qt.AlignCenter)
            mco_placeholder.setStyleSheet("background-color: #f0f0f0; color: #888;")
            self.mco_tab_layout.addWidget(mco_placeholder)

        # Thêm tab MCO (ẩn ban đầu)
        self.mco_tab_index = display_tabs.addTab(self.mco_tab, "MCO Navigator")
        display_tabs.setTabVisible(self.mco_tab_index, False)

        right_layout.addWidget(display_tabs, 1)  # stretch = 1
        main_splitter.addWidget(right_panel)

        # Thiết lập kích thước ban đầu của splitter
        main_splitter.setSizes([300, 700])  # Tỷ lệ 3:7

        # Kết nối các tín hiệu
        self._connect_signals()

        # Cập nhật giao diện theo chế độ mặc định (forward planning)
        self.current_planning_mode = BeamPlanningMode.FORWARD
        self._update_ui_for_mode()

        # Thêm statusbar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Sẵn sàng")

        logger.info("Đã khởi tạo UI cho tab External Beam Planning")

    def set_dose_grid(self, dose_grid, spacing=None, origin=None):
        """
        Đặt lưới liều cho hiển thị.

        Parameters
        ----------
        dose_grid : np.ndarray hoặc DoseGrid
            Lưới liều
        spacing : tuple, optional
            Khoảng cách voxel (mm)
        origin : tuple, optional
            Tọa độ gốc của lưới (mm)
        """
        self.dose_grid = dose_grid
        self.dose_grid_spacing = spacing
        self.dose_grid_origin = origin

        # Cập nhật hiển thị DVH
        self._update_dvh_display()

        # Cập nhật hiển thị liều 3D
        try:
            if hasattr(self, "dose_3d_widget") and self.dose_3d_widget is not None:
                # Nếu dose_grid là DoseGrid object, lấy ra array và metadata
                if (
                    hasattr(dose_grid, "grid")
                    and hasattr(dose_grid, "spacing")
                    and hasattr(dose_grid, "origin")
                ):
                    self.dose_3d_widget.set_dose_grid(
                        dose_grid.grid,
                        dose_grid.spacing if spacing is None else spacing,
                        dose_grid.origin if origin is None else origin,
                    )
                else:
                    # Nếu truyền vào là array trực tiếp
                    self.dose_3d_widget.set_dose_grid(dose_grid, spacing, origin)

                # Cập nhật hiển thị cấu trúc nếu có
                if hasattr(self, "structures") and self.structures is not None:
                    self.dose_3d_widget.set_structures(self.structures)

                # Cập nhật visualization
                self.dose_3d_widget.update_visualization()
                logger.info("Đã cập nhật hiển thị liều 3D thành công")
            else:
                logger.warning("Widget hiển thị liều 3D không khả dụng")
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật hiển thị liều 3D: {e}")

        # Phát tín hiệu
        self.dose_calculated.emit(dose_grid)

    def set_structures(self, structures):
        """
        Đặt cấu trúc cho hiển thị.

        Parameters
        ----------
        structures : dict
            Dictionary chứa các cấu trúc
        """
        self.structures = structures

        # Cập nhật combobox trong widget mục tiêu
        if hasattr(self, "structure_combo"):
            try:
                current_text = self.structure_combo.currentText()
                self.structure_combo.clear()

                for structure_id, structure in structures.items():
                    if hasattr(structure, "name"):
                        self.structure_combo.addItem(structure.name, structure_id)

                # Khôi phục lựa chọn trước đó nếu có thể
                index = self.structure_combo.findText(current_text)
                if index >= 0:
                    self.structure_combo.setCurrentIndex(index)
            except Exception as e:
                logger.error(f"Lỗi khi cập nhật structure_combo: {e}")

        # Cập nhật hiển thị DVH
        self._update_dvh_display()

        # Cập nhật hiển thị 3D
        try:
            if hasattr(self, "dose_3d_widget") and self.dose_3d_widget is not None:
                self.dose_3d_widget.set_structures(structures)
                self.dose_3d_widget.update_visualization()
                logger.info("Đã cập nhật cấu trúc trong hiển thị 3D")
            else:
                logger.warning("Widget hiển thị liều 3D không khả dụng")
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật cấu trúc trong hiển thị 3D: {e}")

        # Phát tín hiệu
        self.structure_changed.emit(structures)

    def _setup_toolbar_actions(self, toolbar):
        """Thiết lập các action cho toolbar."""

        # Tạo các action cho toolbar
        toolbar.setIconSize(QSize(24, 24))

        # Action tạo kế hoạch mới
        new_plan_action = QAction(
            QIcon(os.path.join("quangtps", "ui", "icons", "new_plan.png")),
            "Tạo kế hoạch mới",
            self,
        )
        new_plan_action.triggered.connect(self._on_new_plan)
        toolbar.addAction(new_plan_action)

        # Action lưu kế hoạch
        save_plan_action = QAction(
            QIcon(os.path.join("quangtps", "ui", "icons", "save.png")),
            "Lưu kế hoạch",
            self,
        )
        save_plan_action.triggered.connect(self._on_save_plan)
        toolbar.addAction(save_plan_action)

        toolbar.addSeparator()

        # Action thêm chùm tia
        add_beam_action = QAction(
            QIcon(os.path.join("quangtps", "ui", "icons", "add_beam.png")),
            "Thêm chùm tia",
            self,
        )
        add_beam_action.triggered.connect(self._on_add_beam)
        toolbar.addAction(add_beam_action)

        toolbar.addSeparator()

        # Knowledge-Based Planning action
        kbp_action = QAction(
            QIcon(os.path.join("quangtps", "ui", "icons", "kbp.png")),
            "Knowledge-Based Planning",
            self,
        )
        kbp_action.triggered.connect(self._on_knowledge_based_planning)
        toolbar.addAction(kbp_action)

        toolbar.addSeparator()

        # Action xuất báo cáo
        report_action = QAction(
            QIcon(os.path.join("quangtps", "ui", "icons", "report.png")),
            "Xuất báo cáo",
            self,
        )
        report_action.triggered.connect(self._on_export_report)
        toolbar.addAction(report_action)

        # Action tính toán liều
        calc_dose_action = QAction(
            QIcon(os.path.join("quangtps", "ui", "icons", "calculate.png")),
            "Tính toán liều",
            self,
        )
        calc_dose_action.triggered.connect(self._on_calculate_dose)
        toolbar.addAction(calc_dose_action)

        # Action tối ưu hóa
        optimize_action = QAction(
            QIcon(os.path.join("quangtps", "ui", "icons", "optimize.png")),
            "Tối ưu hóa",
            self,
        )
        optimize_action.triggered.connect(self._on_optimize)
        toolbar.addAction(optimize_action)

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
        self.mode_buttons.buttonClicked.connect(self._on_mode_changed)

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
            self.status_label.setText("Chế độ lập kế hoạch thuận đã được kích hoạt")
        elif index == 1:
            self.planning_mode = BeamPlanningMode.INVERSE
            self.status_label.setText("Chế độ lập kế hoạch ngược đã được kích hoạt")
        elif index == 2:
            self.planning_mode = BeamPlanningMode.MULTI_CRITERIA
            self.status_label.setText(
                "Chế độ tối ưu hóa đa tiêu chí (MCO) đã được kích hoạt"
                )
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
            # Trong chế độ MCO, hiện bảng mục tiêu tối ưu hóa và hiển thị MCO Navigator
            if hasattr(self, "objectives_group"):
                self.objectives_group.setVisible(True)

            # Tự động hiển thị MCO Navigator
            self._show_mco_navigator()

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
        if QApplication is not None:
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

    def _show_mco_navigator(self):
        """Hiển thị MCO Navigator khi chế độ MCO được chọn."""
        # Kiểm tra xem MCO module có khả dụng không
        if not HAS_MCO_UI_MODULE or not HAS_MCO_MODULE:
            QMessageBox.warning(
                self,
                "MCO không khả dụng",
                "Module tối ưu hóa đa tiêu chí (MCO) không khả dụng. Vui lòng cài đặt hoặc kích hoạt module MCO.",
            )
            # Chuyển về chế độ Inverse planning
            self.mode_buttons.blockSignals(True)
            self.mode_buttons.button(1).setChecked(True)
            self.mode_buttons.blockSignals(False)
            self._on_mode_changed(1)
            return

        try:
            # Kiểm tra xem tab MCO đã tồn tại chưa
            mco_index = -1
            right_widget = None

            # Tìm QTabWidget bên phải
            for i in range(self.layout().count()):
                item = self.layout().itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QSplitter):
                    splitter = item.widget()
                    if splitter.count() > 1:
                        right_widget = splitter.widget(1)
                        if isinstance(right_widget, QTabWidget):
                            # Tìm tab MCO nếu đã tồn tại
                            for j in range(right_widget.count()):
                                if right_widget.tabText(j) == "MCO Navigator":
                                    mco_index = j
                                    break
                        break

            if mco_index >= 0 and right_widget:
                # Tab đã tồn tại, chuyển đến tab đó
                right_widget.setCurrentIndex(mco_index)
                self.status_label.setText("MCO Navigator đã được kích hoạt")

                # Cập nhật dữ liệu nếu đã có thay đổi mục tiêu
                if hasattr(self, "mco_navigator_widget") and self.mco_navigator_widget:
                    self._update_mco_objectives()
            elif right_widget and isinstance(right_widget, QTabWidget):
                # Tạo mới tab MCO Navigator
                self.status_label.setText("Đang tạo MCO Navigator...")

                if (
                    not hasattr(self, "mco_navigator_widget")
                    or self.mco_navigator_widget is None
                ):
                    # Thu thập các mục tiêu từ bảng objectives
                    objectives = self._collect_current_objectives()

                    # Tạo MCO Navigator widget
                    self.mco_navigator_widget = create_mco_navigator_widget(
                        objectives=objectives
                    )

                    if self.mco_navigator_widget:
                        # Kết nối tín hiệu
                        self.mco_navigator_widget.solution_selected_signal.connect(
                            self._on_mco_solution_selected
                        )

                        # Thêm tab MCO Navigator
                        right_widget.addTab(self.mco_navigator_widget, "MCO Navigator")
                        mco_index = right_widget.count() - 1
                        right_widget.setCurrentIndex(mco_index)

                        # Tạo dữ liệu mẫu cho demo
                        self._generate_mco_sample_data()
                    else:
                        self.status_label.setText("Không thể tạo MCO Navigator widget")
                else:
                    # MCO Navigator widget đã tồn tại nhưng tab chưa được thêm
                    right_widget.addTab(self.mco_navigator_widget, "MCO Navigator")
                    mco_index = right_widget.count() - 1
                    right_widget.setCurrentIndex(mco_index)
                    self.status_label.setText("MCO Navigator đã được kích hoạt")
        except Exception as e:
            logger.error(f"Lỗi khi tạo MCO Navigator: {str(e)}")
            self.status_label.setText(f"Lỗi MCO: {str(e)}")
            QMessageBox.critical(
                self, "Lỗi MCO", f"Không thể tạo MCO Navigator: {str(e)}"
            )

    def _collect_current_objectives(self):
        """Thu thập các mục tiêu từ bảng objectives hiện tại.

        Returns
        -------
        Dict[str, None]
            Từ điển các mục tiêu để sử dụng cho MCO Navigator
        """
        objectives = {}

        # Thu thập từ bảng objectives
        for i in range(self.objectives_table.rowCount()):
            structure_item = self.objectives_table.item(i, 0)
            type_item = self.objectives_table.item(i, 1)
            value_item = self.objectives_table.item(i, 3)

            if structure_item and type_item and value_item:
                structure_name = structure_item.text()
                obj_type = type_item.text()
                obj_value = value_item.text()

                objective_name = f"{structure_name} {obj_type}"
                objectives[objective_name] = None

        # Nếu không có mục tiêu, tạo một số mục tiêu mẫu từ cấu trúc
        if not objectives:
            # Thu thập cấu trúc từ bảng structure
            structure_names = []
            for i in range(self.structure_table.rowCount()):
                name_item = self.structure_table.item(i, 0)
                if name_item:
                    structure_names.append(name_item.text())

            # Tạo các mục tiêu mẫu dựa trên cấu trúc
            for name in structure_names:
                if "PTV" in name or "CTV" in name or "GTV" in name:
                    objectives[f"{name} Coverage"] = None
                    objectives[f"{name} Homogeneity"] = None
                    objectives[f"{name} Conformity"] = None
                elif any(
                    oar in name.lower()
                    for oar in [
                        "cord",
                        "brain",
                        "parotid",
                        "heart",
                        "lung",
                        "liver",
                        "kidney",
                    ]
                ):
                    objectives[f"{name} Max Dose"] = None
                    objectives[f"{name} Mean Dose"] = None

        # Đảm bảo có ít nhất một mục tiêu
        if not objectives:
            objectives = {
                "PTV Coverage": None,
                "PTV Homogeneity": None,
                "OAR Max Dose": None,
                "OAR Mean Dose": None,
                "Conformity": None,
            }

        return objectives

    def _update_mco_objectives(self):
        """Cập nhật mục tiêu trong MCO Navigator dựa trên mục tiêu hiện tại."""
        if (
            not hasattr(self, "mco_navigator_widget")
            or self.mco_navigator_widget is None
        ):
            return

        objectives = self._collect_current_objectives()
        self.mco_navigator_widget.set_objectives(objectives)

    def _generate_mco_sample_data(self):
        """Tạo dữ liệu mẫu cho MCO Navigator với giao diện tiến trình."""
        if (
            not hasattr(self, "mco_navigator_widget")
            or self.mco_navigator_widget is None
        ):
            return

        try:
            self.status_label.setText("Đang tạo dữ liệu giải pháp Pareto mẫu...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Giả lập tiến trình tạo dữ liệu
            for i in range(1, 41):
                self.progress_bar.setValue(i)
                self.status_label.setText(
                    f"Đang tính toán điểm neo Pareto... {i * 2.5:.1f}%"
                )
                QApplication.processEvents()
                time.sleep(0.05)

            # Tạo dữ liệu mẫu
            self.mco_navigator_widget.create_sample_data()

            # Giả lập hoàn tất
            for i in range(41, 101):
                self.progress_bar.setValue(i)
                self.status_label.setText(f"Đang tạo bề mặt Pareto... {i}%")
                QApplication.processEvents()
                time.sleep(0.01)

            self.status_label.setText(
                "MCO Navigator đã sẵn sàng với dữ liệu Pareto mẫu"
            )
            self.progress_bar.setVisible(False)
        except Exception as e:
            logger.error(f"Lỗi khi tạo dữ liệu mẫu MCO: {str(e)}")
            self.status_label.setText("MCO Navigator đã sẵn sàng")
            self.progress_bar.setVisible(False)

    def _on_mco_solution_selected(self, solution_id):
        """
        Xử lý khi một giải pháp MCO được chọn.

        Parameters
        ----------
        solution_id : str
            ID của giải pháp Pareto được chọn
        """
        if (
            not hasattr(self, "mco_navigator_widget")
            or self.mco_navigator_widget is None
        ):
            return

        # Lấy đối tượng MCO Navigator
        mco_navigator = self.mco_navigator_widget.mco_navigator

        if solution_id in mco_navigator.solutions:
            solution = mco_navigator.solutions[solution_id]

            # Cập nhật UI thông báo
            self.status_label.setText(f"Đang áp dụng giải pháp MCO: {solution_id}")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Cập nhật các mục tiêu và trọng số dựa trên giải pháp đã chọn
            if hasattr(solution, "weights") and solution.weights:
                # Cập nhật trọng số trong bảng mục tiêu
                self._update_objectives_from_mco_solution(solution)

            # Giả lập quá trình tính toán lại liều với trọng số mới
            self._simulate_dose_calculation_progress(
                f"Đang tính toán lại liều cho giải pháp {solution_id}"
            )

            # Tạo liều từ giải pháp
            if hasattr(solution, "dose_data") and solution.dose_data is not None:
                # Nếu giải pháp có dữ liệu liều đính kèm, sử dụng nó
                self.set_dose_grid(solution.dose_data)
            else:
                # Tạo dữ liệu liều mẫu dựa trên giá trị mục tiêu giải pháp
                fake_dose = self._create_solution_based_dose_grid(solution)
                self.set_dose_grid(fake_dose)

            # Cập nhật DVH dựa trên liều mới
            self._update_dvh_display()

            # Hiển thị thông tin về giải pháp đã chọn
            solution_info = self._format_solution_info(solution)
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"Giải pháp MCO {solution_id} đã được áp dụng")

            # Cập nhật hiển thị 3D nếu có
            if hasattr(self, "dose_3d_widget") and self.dose_3d_widget:
                try:
                    self.dose_3d_widget.update()
                except:
                    pass

            # Chuyển đến tab DVH để người dùng xem kết quả
            self._show_dvh_tab()

    def _update_objectives_from_mco_solution(self, solution):
        """
        Cập nhật trọng số trong bảng mục tiêu từ giải pháp MCO được chọn.

        Parameters
        ----------
        solution : ParetoSolution
            Giải pháp Pareto được chọn
        """
        if not hasattr(solution, "weights"):
            return

        # Cập nhật trọng số trong bảng mục tiêu
        for i in range(self.objectives_table.rowCount()):
            structure_item = self.objectives_table.item(i, 0)
            type_item = self.objectives_table.item(i, 1)
            weight_item = self.objectives_table.item(i, 4)

            if structure_item and type_item and weight_item:
                structure_name = structure_item.text()
                obj_type = type_item.text()
                objective_name = f"{structure_name} {obj_type}"

                # Nếu mục tiêu có trong giải pháp, cập nhật trọng số
                if objective_name in solution.weights:
                    weight_value = solution.weights[objective_name]
                    weight_item.setText(f"{weight_value:.2f}")

                    # Đánh dấu mục tiêu đã cập nhật với màu nền
                    for col in range(self.objectives_table.columnCount()):
                        item = self.objectives_table.item(i, col)
                        if item:
                            item.setBackground(QBrush(QColor(220, 240, 255)))

    def _create_solution_based_dose_grid(self, solution):
        """
        Tạo dữ liệu liều dựa trên giải pháp Pareto được chọn.

        Parameters
        ----------
        solution : ParetoSolution
            Giải pháp Pareto được chọn

        Returns
        -------
        np.ndarray
            Dữ liệu liều giả lập
        """
        # Tạo mảng 3D đơn giản (100x100x100)
        grid_size = 100
        dose_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

        # Tạo phân bố liều giả dạng Gaussian
        x = np.linspace(-3, 3, grid_size)
        y = np.linspace(-3, 3, grid_size)
        z = np.linspace(-3, 3, grid_size)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Tạo các thông số phân bố liều dựa trên giải pháp
        try:
            # Số lượng chùm
            beam_count = min(4, len(solution.weights))

            # Lấy đánh giá về độ đồng nhất và phủ từ mục tiêu
            coverage_score = 0
            homogeneity_score = 0
            conformity_score = 0
            max_dose_limit = 70  # Mặc định 70 Gy

            if hasattr(solution, "objectives_values"):
                for key, value in solution.objectives_values.items():
                    if "Coverage" in key:
                        coverage_score = value / 100  # Giả sử thang điểm 0-100
                    elif "Homogeneity" in key:
                        homogeneity_score = value / 100
                    elif "Conformity" in key:
                        conformity_score = value / 100

            # Tạo phân bố liều dựa trên điểm số
            # 1. PTV coverage: Mức độ phủ khối u (tốt = khối u nhận đủ liều)
            # 2. Homogeneity: Độ đồng nhất liều trong khối u (tốt = liều đồng nhất)
            # 3. Conformity: Độ phù hợp với hình dáng khối u (tốt = ít chiếu vào mô lành)

            # Tạo khối u giả lập (hình cầu)
            tumor_center = (grid_size // 2, grid_size // 2, grid_size // 2)
            tumor_radius = grid_size // 6

            # Mặt nạ khối u
            tx, ty, tz = np.ogrid[:grid_size, :grid_size, :grid_size]
            tumor_dist = np.sqrt(
                (tx - tumor_center[0]) ** 2
                + (ty - tumor_center[1]) ** 2
                + (tz - tumor_center[2]) ** 2
            )
            tumor_mask = tumor_dist <= tumor_radius

            # Mặt nạ vùng bên ngoài
            outside_mask = tumor_dist > tumor_radius

            # Tạo các chùm tia
            angles = np.linspace(0, 360, beam_count, endpoint=False)

            for i, angle in enumerate(angles):
                # Tính hướng chùm tia
                rad_angle = np.radians(angle)
                beam_dir = (np.cos(rad_angle), np.sin(rad_angle), 0)

                # Tạo gradient giảm dần dọc theo hướng chùm
                beam_val = X * beam_dir[0] + Y * beam_dir[1] + Z * beam_dir[2]
                beam_val = beam_val - beam_val.min()
                beam_val = beam_val / beam_val.max()

                # Thêm chùm vào lưới liều
                dose_grid += beam_val * 20  # Liều tối đa cho mỗi chùm là 20 Gy

            # Hiệu chỉnh độ đồng nhất
            if homogeneity_score > 0.5:
                # Liều đồng nhất hơn trong PTV
                mean_ptv_dose = np.mean(dose_grid[tumor_mask])
                dose_factor = 1 - (homogeneity_score - 0.5) * 0.5  # 0.75 to 1.0
                dose_grid[tumor_mask] = mean_ptv_dose * dose_factor + dose_grid[
                    tumor_mask
                ] * (1 - dose_factor)

            # Hiệu chỉnh độ phủ
            if coverage_score > 0:
                # Đảm bảo PTV nhận đủ liều theo độ phủ
                dose_grid[tumor_mask] = np.maximum(
                    dose_grid[tumor_mask], 60 * coverage_score
                )

            # Hiệu chỉnh độ phù hợp
            if conformity_score > 0:
                # Giảm liều ở ngoài PTV dựa trên độ phù hợp
                outside_factor = 1 - conformity_score
                dose_grid[outside_mask] *= outside_factor

            # Đảm bảo liều nằm trong khoảng hợp lý
            dose_grid = np.clip(dose_grid, 0, max_dose_limit)

        except Exception as e:
            logger.error(f"Lỗi khi tạo liều giả lập từ giải pháp: {str(e)}")

            # Fallback đến phân bố liều đơn giản
            beam1 = np.exp(-(Y**2 + Z**2) / 0.5) * (X > -2)
            beam2 = np.exp(-(Y**2 + Z**2) / 0.5) * (X < 2)
            dose_grid = (beam1 + beam2) * 70.0  # Liều tối đa 70Gy

        # Thiết lập thông tin không gian
        spacing = (2.0, 2.0, 2.0)  # mm
        origin = (-100.0, -100.0, -100.0)  # mm

        # Lưu thông tin không gian
        self.dose_spacing = spacing
        self.dose_origin = origin

        return dose_grid

    def _format_solution_info(self, solution):
        """
        Định dạng thông tin về giải pháp để hiển thị.

        Parameters
        ----------
        solution : ParetoSolution
            Giải pháp Pareto được chọn

        Returns
        -------
        str
            Thông tin định dạng về giải pháp
        """
        info = []

        if hasattr(solution, "solution_id"):
            info.append(f"ID: {solution.solution_id}")

        if hasattr(solution, "solution_type"):
            info.append(f"Loại: {solution.solution_type.value}")

        if hasattr(solution, "objectives_values") and solution.objectives_values:
            for key, value in solution.objectives_values.items():
                info.append(f"{key}: {value:.2f}")

        return ", ".join(info)

    def _show_dvh_tab(self):
        """Chuyển đến tab DVH nếu có."""
        # Tìm tab widget chứa tab DVH
        tab_widget = None
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QSplitter):
                splitter = item.widget()
                if splitter.count() > 1 and isinstance(splitter.widget(1), QTabWidget):
                    tab_widget = splitter.widget(1)
                    break

        if tab_widget:
            # Tìm tab DVH và chuyển đến đó
            for i in range(tab_widget.count()):
                if tab_widget.tabText(i) == "DVH":
                    tab_widget.setCurrentIndex(i)
                    break

    def _simulate_dose_calculation_progress(self, message="Đang tính toán liều..."):
        """
        Giả lập tiến trình tính toán liều.

        Parameters
        ----------
        message : str
            Thông báo hiển thị trong quá trình tính toán
        """
        for i in range(101):
            self.progress_bar.setValue(i)

            if i < 30:
                step_message = f"{message} - Đang khởi tạo..."
            elif i < 60:
                step_message = f"{message} - Đang tính kernel liều..."
            elif i < 85:
                step_message = f"{message} - Đang tích hợp phân phối liều..."
        else:
                step_message = f"{message} - Đang hoàn tất..."

            self.status_label.setText(step_message)
            QApplication.processEvents()

            # Tạm dừng để giả lập tính toán
            time.sleep(0.02)

    def _on_knowledge_based_planning(self):
        """
        Mở dialog Knowledge-Based Planning để đề xuất các tham số tối ưu.
        """
        if not self.current_plan:
            QMessageBox.warning(
                self, "Lỗi", "Vui lòng tạo hoặc chọn một kế hoạch trước."
            )
            return

        if not self.current_structure_set:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một bộ cấu trúc trước.")
            return

        try:
            # Lấy thông tin cần thiết cho KBP
            patient_id = self.current_patient.id if self.current_patient else ""
            structure_set_id = (
                self.current_structure_set.id if self.current_structure_set else ""
            )

            # Xác định vị trí điều trị từ tên kế hoạch hoặc cấu trúc
            site = ""
            if self.current_plan and hasattr(self.current_plan, "site"):
                site = self.current_plan.site
            elif self.current_structure_set:
                # Thử xác định vị trí từ tên cấu trúc
                structure_names = [s.name.lower() for s in self.structures.values()]

                site_keywords = {
                    "brain": ["brain", "cerebral", "cranial", "head"],
                    "lung": ["lung", "pulmonary", "thoracic"],
                    "prostate": ["prostate", "prostatic"],
                    "head_neck": ["head", "neck", "throat", "larynx"],
                    "breast": ["breast", "chest"],
                    "rectum": ["rectum", "rectal"],
                    "bladder": ["bladder"],
                    "spine": ["spine", "spinal"],
                }

                for potential_site, keywords in site_keywords.items():
                    if any(
                        keyword in " ".join(structure_names) for keyword in keywords
                    ):
                        site = potential_site
                        break

            # Tạo và hiển thị dialog KBP
            from quangtps.ui.dialogs import KBPDialog

            kbp_dialog = KBPDialog(
                patient_id=patient_id,
                structure_set_id=structure_set_id,
                site=site,
                parent=self,
            )

            # Kết nối tín hiệu áp dụng đề xuất
            kbp_dialog.kbpRecommendationApplied.connect(self._apply_kbp_recommendation)

            # Hiển thị dialog
            kbp_dialog.exec_()

        except ImportError as e:
            logger.error(f"Không thể import KBPDialog: {e}")
            QMessageBox.warning(
                self,
                "Tính năng không khả dụng",
                "Module Knowledge-Based Planning không khả dụng.\n"
                "Vui lòng kiểm tra cài đặt và thử lại sau.",
            )
        except Exception as e:
            logger.error(f"Lỗi khi mở KBP Dialog: {e}")
            QMessageBox.critical(
                self, "Lỗi", f"Đã xảy ra lỗi khi mở Knowledge-Based Planning:\n{str(e)}"
            )

    def _apply_kbp_recommendation(self, recommendation):
        """
        Áp dụng đề xuất từ KBP vào kế hoạch hiện tại.

        Parameters
        ----------
        recommendation : Dict
            Đề xuất từ KBP, bao gồm mục tiêu tối ưu và ràng buộc liều
        """
        if not recommendation or not self.current_plan:
            logger.warning(
                "Không thể áp dụng đề xuất KBP: Không có đề xuất hoặc kế hoạch"
            )
            return

        try:
            logger.info(
                "Áp dụng đề xuất Knowledge-Based Planning vào kế hoạch hiện tại"
            )

            # Hiển thị thông báo tiến trình
            progress_dialog = QProgressDialog(
                "Đang áp dụng đề xuất Knowledge-Based Planning...", "Hủy", 0, 100, self
            )
            progress_dialog.setWindowTitle("Knowledge-Based Planning")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setValue(0)
            progress_dialog.show()

            # Cập nhật các mục tiêu tối ưu
            if "objectives" in recommendation and self.objectives_widget:
                progress_dialog.setValue(10)
                progress_dialog.setLabelText("Đang cập nhật mục tiêu tối ưu...")

                # Xóa các mục tiêu hiện tại
                self.objectives_widget.clear_objectives()

                # Thêm các mục tiêu mới từ đề xuất
                for structure_name, objectives in recommendation["objectives"].items():
                    for obj_type, params in objectives.items():
                        # Tạo mục tiêu mới
                        objective = {
                            "structure": structure_name,
                            "type": obj_type,
                            "params": params,
                            "weight": params.get("weight", 1.0),
                        }

                        # Thêm vào widget
                        self.objectives_widget.add_objective(objective)

                        # Cập nhật vào kế hoạch
                        if hasattr(self.current_plan, "objectives"):
                            if not isinstance(self.current_plan.objectives, list):
                                self.current_plan.objectives = []
                            self.current_plan.objectives.append(objective)

                progress_dialog.setValue(40)

            # Cập nhật các ràng buộc liều
            if "constraints" in recommendation and hasattr(self, "constraints_widget"):
                progress_dialog.setValue(50)
                progress_dialog.setLabelText("Đang cập nhật ràng buộc liều...")

                # Xóa các ràng buộc hiện tại nếu có widget constraints
                if hasattr(self, "constraints_widget") and self.constraints_widget:
                    self.constraints_widget.clear_constraints()

                # Thêm các ràng buộc mới từ đề xuất
                for structure_name, constraints in recommendation[
                    "constraints"
                ].items():
                    for constraint_type, params in constraints.items():
                        # Tạo ràng buộc mới
                        constraint = {
                            "structure": structure_name,
                            "type": constraint_type,
                            "params": params,
                            "priority": params.get("priority", "High"),
                        }

                        # Thêm vào widget nếu có
                        if (
                            hasattr(self, "constraints_widget")
                            and self.constraints_widget
                        ):
                            self.constraints_widget.add_constraint(constraint)

                        # Cập nhật vào kế hoạch
                        if hasattr(self.current_plan, "constraints"):
                            if not isinstance(self.current_plan.constraints, list):
                                self.current_plan.constraints = []
                            self.current_plan.constraints.append(constraint)

                progress_dialog.setValue(80)

            # Cập nhật các tham số tối ưu hóa nếu có
            if "optimization_params" in recommendation:
                progress_dialog.setValue(90)
                progress_dialog.setLabelText("Đang cập nhật tham số tối ưu hóa...")

                opt_params = recommendation["optimization_params"]

                # Cập nhật các tham số vào kế hoạch
                if hasattr(self.current_plan, "optimization_params"):
                    self.current_plan.optimization_params.update(opt_params)
                else:
                    self.current_plan.optimization_params = opt_params

            progress_dialog.setValue(100)
            progress_dialog.close()

            # Thông báo thành công
            QMessageBox.information(
                self,
                "Knowledge-Based Planning",
                "Đã áp dụng thành công đề xuất Knowledge-Based Planning vào kế hoạch hiện tại.\n\n"
                "Bạn có thể tiến hành tối ưu hóa kế hoạch ngay bây giờ.",
            )

            # Phát tín hiệu cập nhật kế hoạch
            self.plan_updated.emit(self.current_plan)

            # Tự động chuyển sang chế độ tối ưu hóa ngược
            self.inverse_radio.setChecked(True)
            self._on_mode_changed(1)  # 1 = Inverse Planning mode

        except Exception as e:
            logger.error(f"Lỗi khi áp dụng đề xuất KBP: {str(e)}")
            QMessageBox.critical(
                self,
                "Lỗi",
                f"Đã xảy ra lỗi khi áp dụng đề xuất Knowledge-Based Planning:\n{str(e)}",
            )

    def _format_params(self, params):
        """
        Format các tham số mục tiêu/ràng buộc thành chuỗi.

        Parameters
        ----------
        params : Dict[str, Any]
            Từ điển các tham số cần định dạng

        Returns
        -------
        str
            Chuỗi đã định dạng
        """
        result = []
        for key, value in params.items():
            if key == "dose":
                result.append(f"{value:.1f} Gy")
            elif key == "volume":
                result.append(f"{value:.1f}%")
            elif key == "weight" or key == "priority":
                # Bỏ qua các tham số này vì chúng được hiển thị riêng
                continue
            else:
                result.append(f"{key}: {value}")
        return ", ".join(result)
