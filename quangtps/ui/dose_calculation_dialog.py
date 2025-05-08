#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module DoseCalculationDialog cho QuangTPS

Cung cấp giao diện tương tác hiện đại và dễ sử dụng để cấu hình và thực hiện tính toán liều,
tương tự như giao diện tính toán liều của Eclipse của Varian.
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np

# Thêm xử lý ngoại lệ khi import PyQt5
try:
from PyQt5.QtWidgets import (
        QDialog,
        QPushButton,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QComboBox,
        QProgressBar,
        QGroupBox,
        QCheckBox,
        QSpinBox,
        QDoubleSpinBox,
        QDialogButtonBox,
        QFrame,
        QTabWidget,
        QWidget,
        QFormLayout,
        QRadioButton,
        QButtonGroup,
        QScrollArea,
        QSplitter,
        QLineEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QHeaderView,
        QMessageBox,
        QToolTip,
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer
    from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QCursor

    PYQT_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import PyQt5: {e}")
    PYQT_AVAILABLE = False

    # Tạo các lớp giả để tránh lỗi cú pháp khi không có PyQt5
    class DummyQtClass:
        """Dummy class to replace Qt classes when PyQt5 is not available."""

        pass

    # Tạo các lớp Widget cơ bản
    QDialog = QVBoxLayout = QHBoxLayout = QPushButton = QLabel = QComboBox = (
        QProgressBar
    ) = DummyQtClass
    QGroupBox = QCheckBox = QSpinBox = QDoubleSpinBox = QDialogButtonBox = QFrame = (
        QTabWidget
    ) = DummyQtClass
    QWidget = QFormLayout = QRadioButton = QButtonGroup = QScrollArea = QSplitter = (
        QLineEdit
    ) = DummyQtClass
    QTreeWidget = QTreeWidgetItem = QHeaderView = QMessageBox = QToolTip = DummyQtClass

    # Tạo các lớp Core
    Qt = QThread = QSize = QTimer = DummyQtClass

    # Tạo lớp signal
    class pyqtSignal:
        """Dummy signal class when PyQt5 is not available."""

        def __init__(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

    pyqtSlot = lambda *args, **kwargs: lambda func: func

    # Tạo các lớp Gui
    QFont = QIcon = QPalette = QColor = QCursor = DummyQtClass


try:
    from quangtps.dose.dose_engine import DoseCalculationAlgorithm
    from quangtps.dose.dose_calculation import (
        DoseCalculator,
        DoseCalculationStatus,
        DoseCalculationResult,
        DoseRegionOfInterest,
    )
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.ui.widgets.dose_viewer_widget import DoseViewerWidget
    from quangtps.ui.widgets.dvh_widget import DVHWidget

    # Thử import từ thư mục styles, với xử lý ngoại lệ
    try:
        from quangtps.ui.styles.theme import get_icon
    except ImportError:
        # Tạo function thay thế đơn giản
        def get_icon(name):
            return QIcon()

except ImportError as e:
    logging.error(f"Error importing dose modules: {e}")

logger = logging.getLogger(__name__)


class DoseCalculationThread(QThread):
    """Thread riêng để thực hiện tính toán liều."""

    progress_updated = pyqtSignal(DoseCalculationStatus)
    calculation_finished = pyqtSignal(object, Exception)

    def __init__(
        self, calculator: DoseCalculator, patient_ct, structures, beams, roi, parameters
    ):
        """
        Khởi tạo thread tính toán liều.

        Parameters:
            calculator (DoseCalculator): Bộ tính toán liều
            patient_ct: Hình ảnh CT bệnh nhân
            structures: Dict các cấu trúc
            beams: Danh sách các chùm tia
            roi (DoseRegionOfInterest): Vùng tính toán
            parameters (dict): Các tham số tính toán
        """
        super().__init__()
        self.calculator = calculator
        self.patient_ct = patient_ct
        self.structures = structures
        self.beams = beams
        self.roi = roi
        self.parameters = parameters

    def run(self):
        """Chạy thread tính toán liều."""
        result = None
        error = None

        # Đặt callback để báo tiến độ
        def update_progress(status):
            self.progress_updated.emit(status)

        self.calculator.set_callback(update_progress)

        try:
            # Thực hiện tính toán liều
            result = self.calculator.calculate_dose(
                patient_ct=self.patient_ct,
                structures=self.structures,
                beams=self.beams,
                roi=self.roi,
                parameters=self.parameters,
            )

        except Exception as e:
            logger.error(f"Error during dose calculation: {str(e)}")
            error = e

        # Thông báo hoàn thành
        self.calculation_finished.emit(result, error)


class DoseCalculationDialog(QDialog):
    """
    Hộp thoại tính toán liều xạ trị.

    Dialog này cho phép người dùng cấu hình và thực hiện tính toán phân bố liều,
    cũng như xem kết quả tính toán.
    """

    def __init__(self, parent=None):
        """
        Khởi tạo hộp thoại tính toán liều.

        Parameters:
            parent (QWidget, optional): Widget cha
        """
        super().__init__(parent)

        self.setWindowTitle("Tính toán liều")
        self.resize(900, 700)
        self.setMinimumSize(800, 600)

        # Dữ liệu và công cụ tính toán
        self.patient_ct = None
        self.structures = {}
        self.beams = []
        self.dose_calculator = DoseCalculator()
        self.calculation_thread = None
        self.roi = None
        self.dose_result = None

        # Tạo UI
        self.setup_ui()

        # Kết nối các tín hiệu
        self.connect_signals()

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        main_layout = QVBoxLayout()

        # Tạo QSplitter để chia màn hình
        splitter = QSplitter(Qt.Horizontal)

        # Bên trái: Cấu hình tính toán
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Các tab cấu hình
        config_tabs = QTabWidget()

        # Tab 1: Cấu hình cơ bản
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)

        # GroupBox thuật toán
        algorithm_group = QGroupBox("Thuật toán tính toán")
        algorithm_layout = QFormLayout(algorithm_group)

        self.algorithm_combo = QComboBox()
        for algo in DoseCalculationAlgorithm:
            self.algorithm_combo.addItem(algo.value, algo.name)

        algorithm_layout.addRow("Thuật toán:", self.algorithm_combo)

        # Thêm mô tả thuật toán
        self.algorithm_description = QLabel()
        self.algorithm_description.setWordWrap(True)
        self.algorithm_description.setStyleSheet("font-style: italic; color: #666;")
        algorithm_layout.addRow(self.algorithm_description)

        basic_layout.addWidget(algorithm_group)

        # Vùng tính toán
        roi_group = QGroupBox("Vùng tính toán")
        roi_layout = QFormLayout(roi_group)

        self.roi_type_combo = QComboBox()
        self.roi_type_combo.addItem("Toàn bộ hình ảnh", "FULL")
        self.roi_type_combo.addItem("Vùng cấu trúc", "STRUCTURE")

        roi_layout.addRow("Loại vùng:", self.roi_type_combo)

        self.structure_combo = QComboBox()
        roi_layout.addRow("Cấu trúc:", self.structure_combo)

        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0, 50)
        self.margin_spin.setValue(5.0)
        self.margin_spin.setSuffix(" mm")
        roi_layout.addRow("Mở rộng:", self.margin_spin)

        basic_layout.addWidget(roi_group)

        # Độ phân giải
        resolution_group = QGroupBox("Độ phân giải")
        resolution_layout = QFormLayout(resolution_group)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("Thô (5 mm)", 5.0)
        self.resolution_combo.addItem("Chuẩn (3 mm)", 3.0)
        self.resolution_combo.addItem("Mịn (2 mm)", 2.0)
        self.resolution_combo.addItem("Rất mịn (1 mm)", 1.0)
        self.resolution_combo.setCurrentIndex(1)  # Chọn chuẩn

        resolution_layout.addRow("Độ phân giải:", self.resolution_combo)

        basic_layout.addWidget(resolution_group)

        # Thêm vào tab cơ bản
        config_tabs.addTab(basic_tab, "Cơ bản")

        # Tab 2: Cấu hình nâng cao
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)

        # GroupBox các tham số tính toán
        params_group = QGroupBox("Tham số tính toán")
        params_layout = QFormLayout(params_group)

        self.heterogeneity_check = QCheckBox("Kích hoạt")
        self.heterogeneity_check.setChecked(True)
        params_layout.addRow("Hiệu chỉnh không đồng nhất:", self.heterogeneity_check)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 32)
        self.threads_spin.setValue(4)
        params_layout.addRow("Số luồng tính toán:", self.threads_spin)

        self.output_correction_check = QCheckBox("Kích hoạt")
        self.output_correction_check.setChecked(True)
        params_layout.addRow("Hiệu chỉnh hệ số đầu ra:", self.output_correction_check)

        # Thêm tùy chọn GPU
        self.use_gpu_check = QCheckBox("Kích hoạt")
        self.use_gpu_check.setChecked(True)
        self.use_gpu_check.setToolTip(
            "Sử dụng GPU để tăng tốc tính toán liều nếu thuật toán hỗ trợ và phần cứng khả dụng"
        )
        params_layout.addRow("Tính toán với GPU:", self.use_gpu_check)

        # Thêm thông tin về GPU
        self.gpu_info_label = QLabel("Đang kiểm tra thông tin GPU...")
        self.gpu_info_label.setStyleSheet("font-style: italic; color: #888;")
        params_layout.addRow(self.gpu_info_label)

        # Thêm combobox chọn GPU nếu có nhiều GPU
        self.gpu_device_combo = QComboBox()
        self.gpu_device_combo.setEnabled(False)
        params_layout.addRow("Thiết bị GPU:", self.gpu_device_combo)

        # Tùy chọn độ chính xác
        self.gpu_precision_combo = QComboBox()
        self.gpu_precision_combo.addItem("Đơn (nhanh hơn)", "single")
        self.gpu_precision_combo.addItem("Kép (chính xác hơn)", "double")
        self.gpu_precision_combo.setEnabled(False)
        params_layout.addRow("Độ chính xác:", self.gpu_precision_combo)

        self.custom_params_tree = QTreeWidget()
        self.custom_params_tree.setHeaderLabels(["Tham số", "Giá trị"])
        self.custom_params_tree.setRootIsDecorated(False)
        self.custom_params_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )

        params_layout.addRow(self.custom_params_tree)

        advanced_layout.addWidget(params_group)

        # Tùy chọn nâng cao
        advanced_options_group = QGroupBox("Tùy chọn nâng cao")
        advanced_options_layout = QFormLayout(advanced_options_group)

        self.verbose_logging_check = QCheckBox("Kích hoạt")
        advanced_options_layout.addRow(
            "Ghi nhật ký chi tiết:", self.verbose_logging_check
        )

        self.save_intermediate_check = QCheckBox("Kích hoạt")
        advanced_options_layout.addRow(
            "Lưu kết quả trung gian:", self.save_intermediate_check
        )

        advanced_layout.addWidget(advanced_options_group)

        # Thêm vào tab nâng cao
        config_tabs.addTab(advanced_tab, "Nâng cao")

        # Tab 3: Chùm tia
        beams_tab = QWidget()
        beams_layout = QVBoxLayout(beams_tab)

        self.beams_tree = QTreeWidget()
        self.beams_tree.setHeaderLabels(["Tên", "Năng lượng", "Liều", "Góc"])
        self.beams_tree.setRootIsDecorated(False)

        beams_layout.addWidget(self.beams_tree)

        # Thêm vào tab chùm tia
        config_tabs.addTab(beams_tab, "Chùm tia")

        # Thêm tab vào layout bên trái
        left_layout.addWidget(config_tabs)

        # Thanh tiến trình
        progress_group = QGroupBox("Tiến độ tính toán")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Chưa bắt đầu tính toán")
        progress_layout.addWidget(self.status_label)

        left_layout.addWidget(progress_group)

        # Các nút điều khiển
        button_layout = QHBoxLayout()

        self.calculate_button = QPushButton("Bắt đầu tính toán")
        self.calculate_button.setIcon(get_icon("calculate"))

        self.cancel_button = QPushButton("Hủy tính toán")
        self.cancel_button.setIcon(get_icon("cancel"))
        self.cancel_button.setEnabled(False)

        self.close_button = QPushButton("Đóng")
        self.close_button.setIcon(get_icon("close"))

        button_layout.addWidget(self.calculate_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        left_layout.addLayout(button_layout)

        # Bên phải: Hiển thị kết quả
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.result_tabs = QTabWidget()

        # Tab hiển thị liều
        self.dose_viewer = DoseViewerWidget()
        self.result_tabs.addTab(self.dose_viewer, "Hiển thị liều")

        # Tab DVH
        self.dvh_widget = DVHWidget()
        self.result_tabs.addTab(self.dvh_widget, "Biểu đồ DVH")

        # Tab thống kê
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        self.stats_tree = QTreeWidget()
        self.stats_tree.setHeaderLabels(["Thông số", "Giá trị"])
        self.stats_tree.setRootIsDecorated(False)
        stats_layout.addWidget(self.stats_tree)

        self.result_tabs.addTab(stats_tab, "Thống kê")

        right_layout.addWidget(self.result_tabs)

        # Thêm các panel vào splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # Đặt tỉ lệ ban đầu (40/60)
        splitter.setSizes([350, 550])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Khởi tạo giá trị ban đầu
        self.update_algorithm_description()
        self.update_structure_combo()

    def connect_signals(self):
        """Kết nối các tín hiệu và slots."""
        # Cập nhật mô tả thuật toán khi thay đổi
        self.algorithm_combo.currentIndexChanged.connect(
            self.update_algorithm_description
        )

        # Cập nhật danh sách tham số tùy chỉnh khi thay đổi thuật toán
        self.algorithm_combo.currentIndexChanged.connect(self.update_custom_parameters)

        # Kích hoạt/vô hiệu hóa các điều khiển ROI
        self.roi_type_combo.currentIndexChanged.connect(self.update_roi_controls)

        # Kích hoạt/vô hiệu hóa các tùy chọn GPU dựa trên checkbox
        self.use_gpu_check.toggled.connect(self.update_gpu_controls)

        # Kiểm tra thông tin GPU khi khởi động
        QTimer.singleShot(500, self.check_gpu_availability)

        # Xử lý nút tính toán
        self.calculate_button.clicked.connect(self.start_calculation)

        # Xử lý nút hủy
        self.cancel_button.clicked.connect(self.cancel_calculation)

        # Xử lý nút đóng
        self.close_button.clicked.connect(self.close)

    def check_gpu_availability(self):
        """Kiểm tra và hiển thị thông tin về GPU có sẵn"""
        try:
            # Thử import CuPy để kiểm tra GPU
            import cupy as cp

            # Lấy thông tin GPU
            gpu_count = cp.cuda.runtime.getDeviceCount()

            if gpu_count > 0:
                # Có GPU khả dụng, hiển thị thông tin
                current_device = cp.cuda.runtime.getDevice()
                gpu_props = cp.cuda.runtime.getDeviceProperties(current_device)

                # Hiển thị tên GPU và bộ nhớ
                mem_total = gpu_props["totalGlobalMem"] / (1024**3)  # Convert to GB
                gpu_info = f"Đã tìm thấy: {gpu_props['name']} ({mem_total:.1f} GB)"
                self.gpu_info_label.setText(gpu_info)

                # Thêm các GPU vào combobox
                self.gpu_device_combo.clear()
                for i in range(gpu_count):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    self.gpu_device_combo.addItem(f"{props['name']}", i)

                # Kích hoạt các điều khiển GPU
                self.gpu_device_combo.setEnabled(True)
                self.gpu_precision_combo.setEnabled(True)
                self.use_gpu_check.setEnabled(True)
            else:
                # Không có GPU khả dụng
                self.gpu_info_label.setText("Không tìm thấy GPU hỗ trợ CUDA")
                self.use_gpu_check.setChecked(False)
                self.use_gpu_check.setEnabled(False)
                self.gpu_device_combo.setEnabled(False)
                self.gpu_precision_combo.setEnabled(False)

        except ImportError:
            # CuPy không được cài đặt hoặc không tìm thấy GPU
            self.gpu_info_label.setText("Không tìm thấy thư viện CuPy hoặc GPU CUDA")
            self.use_gpu_check.setChecked(False)
            self.use_gpu_check.setEnabled(False)
            self.gpu_device_combo.setEnabled(False)
            self.gpu_precision_combo.setEnabled(False)
        except Exception as e:
            # Lỗi khác
            self.gpu_info_label.setText(f"Lỗi kiểm tra GPU: {str(e)}")
            self.use_gpu_check.setChecked(False)
            self.use_gpu_check.setEnabled(False)
            self.gpu_device_combo.setEnabled(False)
            self.gpu_precision_combo.setEnabled(False)

    def update_gpu_controls(self, enabled):
        """Cập nhật trạng thái các điều khiển GPU dựa trên checkbox"""
        self.gpu_device_combo.setEnabled(enabled)
        self.gpu_precision_combo.setEnabled(enabled)

    def update_algorithm_description(self):
        """Cập nhật mô tả thuật toán."""
        algorithm_name = self.algorithm_combo.currentData()

        if algorithm_name == DoseCalculationAlgorithm.CCC.name:
            description = (
                "Collapsed Cone Convolution sử dụng kỹ thuật tích chập với các kernel "
                "để mô phỏng quá trình vật lý của tán xạ bức xạ trong các vật liệu. "
                "Thuật toán này cung cấp cân bằng tốt giữa tốc độ và độ chính xác."
            )
        elif algorithm_name == DoseCalculationAlgorithm.PENCIL_BEAM.name:
            description = (
                "Pencil Beam là thuật toán đơn giản, hiệu quả sử dụng các chùm tia "
                "hình bút chì để mô phỏng quá trình vật lý. Thích hợp cho các trường "
                "hợp đơn giản và tính toán nhanh, nhưng không chính xác cho các vùng "
                "không đồng nhất phức tạp."
            )
        elif algorithm_name == DoseCalculationAlgorithm.AAA.name:
            description = (
                "Analytical Anisotropic Algorithm (AAA) là thuật toán tính toán liều 3D "
                "kết hợp mô hình phân tán anisotropic với mô hình tương tác electron-photon. "
                "Nó cung cấp độ chính xác cao, đặc biệt là trong các vùng không đồng nhất."
            )
        elif (
            algorithm_name == DoseCalculationAlgorithm.ACUROS.name
            or algorithm_name == "ACUROS_XB"
        ):
            description = (
                "Acuros XB giải phương trình vận chuyển bức xạ tuyến tính Boltzmann, "
                "mang lại độ chính xác tương đương Monte Carlo với tốc độ nhanh hơn đáng kể. "
                "Đặc biệt chính xác trong các môi trường có mật độ electron không đồng nhất cao. "
                "Thuật toán này hỗ trợ tính toán GPU để tăng tốc đáng kể."
            )

            # Kiểm tra xem thuật toán này có hỗ trợ GPU không
            try:
                from quangtps.dose.dose_engine import is_gpu_supported

                if hasattr(DoseCalculationAlgorithm, "ACUROS_XB") and is_gpu_supported(
                    DoseCalculationAlgorithm.ACUROS_XB
                ):
                    # Hiển thị thông tin GPU trong mô tả
                    description += "\n\nThuật toán này hỗ trợ tính toán trên GPU để tăng tốc đáng kể."
            except ImportError:
                pass

        elif algorithm_name == DoseCalculationAlgorithm.CONV_SUPERPOSITION.name:
            description = (
                "Convolution Superposition tích hợp convolution của năng lượng được giải phóng "
                "với kernel đáp ứng, cho phép xem xét tính không đồng nhất của vật liệu. "
                "Phù hợp cho hầu hết các ứng dụng lâm sàng."
            )
        elif algorithm_name == DoseCalculationAlgorithm.MONTE_CARLO.name:
            description = (
                "Monte Carlo là thuật toán chính xác nhất, sử dụng mô phỏng xác suất các quá trình "
                "vật lý của bức xạ ion hóa. Rất chính xác cho tất cả các loại vùng không đồng nhất "
                "nhưng đòi hỏi thời gian tính toán lâu hơn."
            )

            # Kiểm tra xem thuật toán này có hỗ trợ GPU không
            try:
                from quangtps.dose.dose_engine import is_gpu_supported

                if is_gpu_supported(DoseCalculationAlgorithm.MONTE_CARLO):
                    # Hiển thị thông tin GPU trong mô tả
                    description += "\n\nThuật toán này hỗ trợ tính toán trên GPU để tăng tốc đáng kể."
            except ImportError:
                pass
        else:
            description = ""

        self.algorithm_description.setText(description)

        # Cập nhật trạng thái điều khiển GPU
        try:
            from quangtps.dose.dose_engine import is_gpu_supported

            if hasattr(DoseCalculationAlgorithm, algorithm_name):
                algo_enum = getattr(DoseCalculationAlgorithm, algorithm_name)
                gpu_supported = is_gpu_supported(algo_enum)
                self.use_gpu_check.setEnabled(gpu_supported)
                if not gpu_supported:
                    self.use_gpu_check.setChecked(False)
                    self.gpu_device_combo.setEnabled(False)
                    self.gpu_precision_combo.setEnabled(False)
        except (ImportError, AttributeError):
            # Nếu không thể kiểm tra, mặc định vô hiệu hóa điều khiển
            pass

        # Cập nhật danh sách tham số
        self.update_custom_parameters()

    def update_custom_parameters(self):
        """Cập nhật danh sách tham số tùy chỉnh dựa trên thuật toán được chọn."""
        self.custom_params_tree.clear()

        algorithm_name = self.algorithm_combo.currentData()

        # Tham số chung
        common_params = [
            ("normalization_value", "Giá trị chuẩn hóa", 100.0, "%"),
            ("normalization_point", "Điểm chuẩn hóa", "isocenter", ""),
            ("dose_grid_margin", "Mở rộng lưới liều", 5.0, "mm"),
            ("use_beam_limiting_devices", "Sử dụng MLC/Jaw", True, ""),
        ]

        # Tham số riêng cho từng thuật toán
        if algorithm_name == DoseCalculationAlgorithm.CCC.name:
            algo_params = [
                ("num_cones", "Số cone", 32, ""),
                ("scatter_kernel_size", "Kích thước kernel tán xạ", 10.0, "cm"),
            ]
        elif algorithm_name == DoseCalculationAlgorithm.PENCIL_BEAM.name:
            algo_params = [
                ("kernel_width", "Độ rộng kernel", 7, ""),
                ("use_precalculated_kernels", "Dùng kernel có sẵn", True, ""),
            ]
        elif algorithm_name == DoseCalculationAlgorithm.AAA.name:
            algo_params = [
                ("scatter_kernel_resolution", "Độ phân giải kernel tán xạ", 1.0, "mm"),
                ("output_factor_correction", "Hiệu chỉnh hệ số đầu ra", True, ""),
            ]
        elif algorithm_name == DoseCalculationAlgorithm.ACUROS.name:
            algo_params = [
                ("grid_size", "Kích thước lưới", 2.5, "mm"),
                ("energy_cutoff", "Ngưỡng cắt năng lượng", 0.01, "MeV"),
            ]
        elif algorithm_name == DoseCalculationAlgorithm.CONV_SUPERPOSITION.name:
            algo_params = [
                ("kernel_mode", "Chế độ kernel", "isotropic", ""),
                ("scatter_components", "Thành phần tán xạ", 3, ""),
            ]
        elif algorithm_name == DoseCalculationAlgorithm.MONTE_CARLO.name:
            algo_params = [
                ("num_histories", "Số lịch sử hạt", 1000000, ""),
                ("statistical_uncertainty", "Bất định thống kê", 2.0, "%"),
                ("variance_reduction", "Giảm phương sai", True, ""),
            ]
        else:
            algo_params = []

        # Thêm tham số vào tree
        for param_id, param_name, default_value, unit in common_params + algo_params:
            item = QTreeWidgetItem()
            item.setText(0, param_name)

            # Thêm control tùy thuộc vào loại tham số
            if isinstance(default_value, bool):
                checkbox = QCheckBox()
                checkbox.setChecked(default_value)
                self.custom_params_tree.setItemWidget(item, 1, checkbox)
            elif isinstance(default_value, int):
                spinbox = QSpinBox()
                spinbox.setValue(default_value)
                if "histories" in param_id:
                    spinbox.setRange(1000, 100000000)
                    spinbox.setSingleStep(1000)
                else:
                    spinbox.setRange(1, 1000)
                spinbox.setSuffix(f" {unit}" if unit else "")
                self.custom_params_tree.setItemWidget(item, 1, spinbox)
            elif isinstance(default_value, float):
                spinbox = QDoubleSpinBox()
                spinbox.setValue(default_value)
                spinbox.setDecimals(2)
                spinbox.setSingleStep(0.1)
                spinbox.setRange(0, 100)
                spinbox.setSuffix(f" {unit}" if unit else "")
                self.custom_params_tree.setItemWidget(item, 1, spinbox)
            elif isinstance(default_value, str):
                combobox = QComboBox()

                if param_id == "normalization_point":
                    combobox.addItems(["isocenter", "max_dose", "custom_point"])
                elif param_id == "kernel_mode":
                    combobox.addItems(["isotropic", "anisotropic", "hybrid"])

                combobox.setCurrentText(default_value)
                self.custom_params_tree.setItemWidget(item, 1, combobox)
            else:
                item.setText(1, str(default_value))

            item.setData(0, Qt.UserRole, param_id)
            self.custom_params_tree.addTopLevelItem(item)

        self.custom_params_tree.expandAll()

    def update_structure_combo(self):
        """Cập nhật danh sách cấu trúc."""
        self.structure_combo.clear()

        # Thêm cấu trúc giả cho demo
        demo_structures = [
            "PTV",
            "CTV",
            "GTV",
            "BODY",
            "HEART",
            "LUNG_LEFT",
            "LUNG_RIGHT",
            "SPINAL_CORD",
        ]

        for structure in demo_structures:
            self.structure_combo.addItem(structure)

        # Thêm cấu trúc thực từ dataset nếu có
        if self.structures:
            self.structure_combo.clear()
            for name in self.structures.keys():
                self.structure_combo.addItem(name)

    def update_roi_controls(self):
        """Kích hoạt/vô hiệu hóa các điều khiển ROI dựa trên loại vùng."""
        roi_type = self.roi_type_combo.currentData()

        # Kích hoạt hoặc vô hiệu hóa các điều khiển liên quan
        self.structure_combo.setEnabled(roi_type == "STRUCTURE")
        self.margin_spin.setEnabled(roi_type == "STRUCTURE")

    def set_patient_data(self, patient_ct, structures=None, beams=None):
        """
        Đặt dữ liệu bệnh nhân.

        Parameters:
            patient_ct: Hình ảnh CT bệnh nhân
            structures (dict, optional): Dict các cấu trúc
            beams (list, optional): Danh sách các chùm tia
        """
        self.patient_ct = patient_ct

        if structures:
            self.structures = structures
            self.update_structure_combo()

        if beams:
            self.beams = beams
            self.update_beams_tree()

    def update_beams_tree(self):
        """Cập nhật danh sách chùm tia."""
        self.beams_tree.clear()

        for i, beam in enumerate(self.beams):
            item = QTreeWidgetItem()

            # Tên chùm tia
            beam_name = beam.get("name", f"Beam {i + 1}")
            item.setText(0, beam_name)

            # Năng lượng
            energy = beam.get("energy", 6.0)
            energy_text = f"{energy} MV" if energy >= 1.0 else f"{energy * 1000:.0f} kV"
            item.setText(1, energy_text)

            # Liều (MU)
            mu = beam.get("mu", 100.0)
            item.setText(2, f"{mu:.1f} MU")

            # Góc (gantry)
            gantry_angle = beam.get("gantry_angle", 0.0)
            item.setText(3, f"{gantry_angle:.1f}°")

            self.beams_tree.addTopLevelItem(item)

        # Nếu không có chùm tia, thêm các chùm tia mẫu
        if not self.beams:
            demo_beams = [
                {"name": "AP", "energy": 6.0, "mu": 100.0, "gantry_angle": 0.0},
                {"name": "PA", "energy": 6.0, "mu": 100.0, "gantry_angle": 180.0},
                {"name": "LT", "energy": 6.0, "mu": 80.0, "gantry_angle": 270.0},
                {"name": "RT", "energy": 6.0, "mu": 80.0, "gantry_angle": 90.0},
            ]

            self.beams = demo_beams
            self.update_beams_tree()

    def get_selected_algorithm(self):
        """
        Lấy thuật toán được chọn.

        Returns:
            DoseCalculationAlgorithm: Thuật toán được chọn
        """
        algorithm_name = self.algorithm_combo.currentData()
        return DoseCalculationAlgorithm.from_string(algorithm_name)

    def get_calculation_parameters(self):
        """
        Lấy các tham số tính toán từ giao diện.

        Returns:
            dict: Tham số tính toán
        """
        params = {}

        # Lấy từ các control cơ bản
        params["algorithm"] = self.get_selected_algorithm()
        params["resolution"] = self.resolution_combo.currentData()
        params["heterogeneity_correction"] = self.heterogeneity_check.isChecked()
        params["threads"] = self.threads_spin.value()
        params["output_correction"] = self.output_correction_check.isChecked()

        # Thêm tham số GPU
        params["use_gpu"] = (
            self.use_gpu_check.isChecked() and self.use_gpu_check.isEnabled()
        )
        if params["use_gpu"]:
            params["gpu_device"] = self.gpu_device_combo.currentData()
            params["gpu_precision"] = self.gpu_precision_combo.currentData()

        # Thêm tham số từ tree tùy chỉnh
        for i in range(self.custom_params_tree.topLevelItemCount()):
            item = self.custom_params_tree.topLevelItem(i)
            param_id = item.data(0, Qt.UserRole)

            # Lấy giá trị từ các widget tùy chỉnh
            widget = self.custom_params_tree.itemWidget(item, 1)

            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
            else:
                value = item.text(1)

            params[param_id] = value

        return params

    def get_selected_roi(self):
        """
        Lấy vùng tính toán (ROI) được chọn.

        Returns:
            DoseRegionOfInterest: Vùng tính toán
        """
        roi_type = self.roi_type_combo.currentData()

        if roi_type == "STRUCTURE":
            # Lấy tên cấu trúc và margin
            structure_name = self.structure_combo.currentText()
            margin_mm = self.margin_spin.value()

            return DoseRegionOfInterest(
                name=f"{structure_name}+{margin_mm}mm",
                structure_name=structure_name,
                margin_mm=margin_mm,
            )
        else:
            # Toàn bộ hình ảnh
            return DoseRegionOfInterest(name="Whole image")

    def start_calculation(self):
        """Bắt đầu tính toán liều."""
        # Kiểm tra dữ liệu đầu vào
        if not self.patient_ct:
            QMessageBox.warning(
                self, "Thiếu dữ liệu", "Cần hình ảnh CT bệnh nhân để tính toán liều."
            )
            return

        if not self.beams:
            QMessageBox.warning(
                self, "Thiếu dữ liệu", "Cần ít nhất một chùm tia để tính toán liều."
            )
            return

        # Lấy tham số tính toán
        parameters = self.get_calculation_parameters()

        # Lấy vùng tính toán
        self.roi = self.get_selected_roi()

        # Đặt thuật toán
        self.dose_calculator.set_algorithm(parameters["algorithm"])

        # Cập nhật UI
        self.progress_bar.setValue(0)
        self.status_label.setText("Đang khởi tạo tính toán...")
        self.calculate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        # Tạo và khởi động thread tính toán
        self.calculation_thread = DoseCalculationThread(
            calculator=self.dose_calculator,
            patient_ct=self.patient_ct,
            structures=self.structures,
            beams=self.beams,
            roi=self.roi,
            parameters=parameters,
        )

        # Kết nối tín hiệu
        self.calculation_thread.progress_updated.connect(
            self.update_calculation_progress
        )
        self.calculation_thread.calculation_finished.connect(
            self.handle_calculation_result
        )

        # Bắt đầu tính toán
        self.calculation_thread.start()

    def cancel_calculation(self):
        """Hủy tính toán liều đang chạy."""
        if self.calculation_thread and self.calculation_thread.isRunning():
            # Cập nhật UI
            self.status_label.setText("Đang hủy tính toán...")

            # Dừng thread
            self.calculation_thread.terminate()
            self.calculation_thread.wait()

            # Đặt lại UI
            self.status_label.setText("Tính toán đã bị hủy")
            self.calculate_button.setEnabled(True)
            self.cancel_button.setEnabled(False)

    @pyqtSlot(DoseCalculationStatus)
    def update_calculation_progress(self, status: DoseCalculationStatus):
        """
        Cập nhật tiến độ tính toán.

        Parameters:
            status (DoseCalculationStatus): Trạng thái tính toán
        """
        # Cập nhật thanh tiến độ
        progress_percent = int(status.progress * 100)
        self.progress_bar.setValue(progress_percent)

        # Cập nhật nhãn trạng thái
        if status.status:
            elapsed_time = status.get_elapsed_time()
            self.status_label.setText(f"{status.status} - {elapsed_time:.1f}s")

    @pyqtSlot(object, Exception)
    def handle_calculation_result(self, result, error):
        """
        Xử lý kết quả tính toán.

        Parameters:
            result (DoseGrid): Kết quả tính toán liều
            error (Exception): Lỗi nếu có
        """
        # Đặt lại UI
        self.calculate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        if error:
            # Hiển thị lỗi
            self.status_label.setText(f"Lỗi: {str(error)}")
            QMessageBox.critical(
                self, "Lỗi tính toán", f"Lỗi khi tính toán liều:\n{str(error)}"
            )
            return

        # Lưu kết quả
        self.dose_result = result

        # Cập nhật UI
        self.status_label.setText("Tính toán hoàn thành")
        self.progress_bar.setValue(100)

        # Hiển thị kết quả
        self.display_calculation_result()

    def display_calculation_result(self):
        """Hiển thị kết quả tính toán."""
        if not self.dose_result:
            return

        # Hiển thị phân bố liều
        self.dose_viewer.set_dose_data(self.dose_result.data)

        # TODO: Hiển thị DVH
        # TODO: Hiển thị thống kê

        # Chuyển sang tab hiển thị kết quả
        self.result_tabs.setCurrentIndex(0)

    def closeEvent(self, event):
        """
        Xử lý sự kiện đóng hộp thoại.

        Parameters:
            event: Sự kiện đóng
        """
        # Hủy tính toán nếu đang chạy
        if self.calculation_thread and self.calculation_thread.isRunning():
            self.calculation_thread.terminate()
            self.calculation_thread.wait()

        event.accept()
