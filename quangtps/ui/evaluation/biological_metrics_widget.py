#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Widget hiển thị các chỉ số sinh học cho đánh giá kế hoạch xạ trị.

Module này cung cấp các widget để hiển thị và tương tác với các chỉ số sinh học
như TCP, NTCP, EUD, và BED.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Import các module PyQt
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QPushButton,
        QComboBox,
        QDoubleSpinBox,
        QTabWidget,
        QSplitter,
        QGroupBox,
        QFormLayout,
        QHeaderView,
        QMessageBox,
        QScrollArea,
        QApplication,
        QGridLayout,
        QSpinBox,
        QAbstractItemView,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QColor, QBrush, QPainter
    from PyQt5.QtChart import (
        QChart,
        QChartView,
        QBarSet,
        QBarSeries,
        QValueAxis,
        QBarCategoryAxis,
    )

    HAS_PYQT = True
    logger = logging.getLogger(__name__)
    logger.info("PyQt5 đã được import thành công cho biological_metrics_widget")
except ImportError as e:
    HAS_PYQT = False
    logger = logging.getLogger(__name__)
    logger.error(f"Không thể import PyQt5: {str(e)}")

    # Tạo các lớp giả khi không có PyQt
    class QWidget:
        def __init__(self, *args, **kwargs):
            pass

    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass

    class QColor:
        def __init__(self, *args, **kwargs):
            pass


# Import matplotlib với xử lý lỗi
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError as e:
    HAS_MATPLOTLIB = False
    logger.error(f"Không thể import matplotlib: {str(e)}")

    # Tạo lớp giả cho matplotlib
    class FigureCanvas:
        def __init__(self, *args, **kwargs):
            pass

    class Figure:
        def __init__(self, *args, **kwargs):
            pass


# Import các module sinh học
try:
    from quangtps.evaluation.biological.biological_models import (
        calculate_eud,
        calculate_tcp,
        calculate_ntcp,
        calculate_biological_metrics,
        get_organ_specific_parameters,
    )

    HAS_BIO_MODELS = True
except ImportError:
    logging.warning(
        "Không thể import module biological_models. Một số tính năng sẽ không khả dụng."
    )
    HAS_BIO_MODELS = False

# Cập nhật phần import để sử dụng module biological_evaluation mới
try:
    from quangtps.evaluation.biological.biological_evaluation import (
        BiologicalEvaluation,
        create_biological_evaluation,
    )

    HAS_BIO_EVALUATION = True
except ImportError:
    HAS_BIO_EVALUATION = False
    logger.error("Không thể import module biological_evaluation.")

logger = logging.getLogger(__name__)


class BiologicalMetricsWidget(QWidget):
    """
    Widget hiển thị các chỉ số sinh học cho đánh giá kế hoạch xạ trị.

    Widget này hiển thị các chỉ số sinh học như TCP, NTCP, EUD, và BED
    cho từng cấu trúc trong kế hoạch xạ trị.
    """

    # Tín hiệu khi người dùng thay đổi tham số
    parameters_changed = pyqtSignal()

    def __init__(self, parent=None):
        """
        Khởi tạo BiologicalMetricsWidget.

        Args:
            parent: Widget cha
        """
        super().__init__(parent)

        # Dữ liệu
        self.dvh_data = {}
        self.structure_names = []
        self.biological_metrics = {}
        self.structure_types = {}
        self.structure_organ_mapping = {}
        self.detailed_results = {}

        # Module sinh học
        self.bio_evaluator = (
            create_biological_evaluation() if HAS_BIO_EVALUATION else None
        )

        # Tham số mặc định
        self.parameters = {
            "alpha_beta_tumor": 10.0,  # Tỷ lệ alpha/beta cho khối u (Gy)
            "alpha_beta_normal": 3.0,  # Tỷ lệ alpha/beta cho mô lành (Gy)
            "fraction_size": 2.0,  # Kích thước phân liều (Gy)
            "num_fractions": 30,  # Số phân liều
            "show_eud": True,  # Hiển thị EUD
            "show_tcp": True,  # Hiển thị TCP
            "show_ntcp": True,  # Hiển thị NTCP
            "show_bed": True,  # Hiển thị BED
            "auto_update": True,  # Tự động cập nhật khi thay đổi tham số
            "tcp_model": "poisson",  # Mô hình TCP mặc định
            "ntcp_model": "lkb",  # Mô hình NTCP mặc định
            "use_alternate_models": True,  # Sử dụng các mô hình thay thế khi cần
            "auto_detect_type": True,  # Tự động nhận diện loại cấu trúc
            "show_warnings": True,  # Hiển thị cảnh báo trong đánh giá
            "show_radar_chart": True,  # Hiển thị biểu đồ radar
            "show_sensitivity": True,  # Hiển thị phân tích độ nhạy
        }

        # Trạng thái UI
        self.selected_structure = None
        self.is_updating = False

        # Khởi tạo giao diện
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout(self)

        # Tạo tabs
        self.tabs = QTabWidget()

        # Tab 1: Bảng chỉ số sinh học
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab)

        # Bảng chỉ số sinh học
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Cấu trúc", "Loại", "EUD (Gy)", "TCP (%)", "NTCP (%)", "Đánh giá"]
        )

        # Kết nối sự kiện chọn dòng để hiển thị chi tiết
        self.metrics_table.itemSelectionChanged.connect(
            lambda: self._update_detail_display()
        )

        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        metrics_layout.addWidget(self.metrics_table)

        # Tab 2: Biểu đồ TCP/NTCP
        self.chart_tab = QWidget()
        chart_layout = QVBoxLayout(self.chart_tab)

        # Tạo biểu đồ
        self.chart = QChart()
        self.chart.setTitle("TCP/NTCP theo cấu trúc")
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        chart_layout.addWidget(self.chart_view)

        # Tab 3: Biểu đồ Radar
        self.radar_tab = QWidget()
        self.radar_layout = QVBoxLayout(self.radar_tab)

        if HAS_MATPLOTLIB:
            # Phân chia không gian biểu đồ radar
            self.radar_splitter = QSplitter(Qt.Horizontal)

            # Panel trái (danh sách cấu trúc)
            self.structure_list_widget = QWidget()
            structure_layout = QVBoxLayout(self.structure_list_widget)
            structure_layout.addWidget(QLabel("Cấu trúc:"))

            self.structures_combo = QComboBox()
            self.structures_combo.currentIndexChanged.connect(self._update_radar_chart)
            structure_layout.addWidget(self.structures_combo)

            structure_layout.addStretch()

            # Panel phải (biểu đồ radar)
            self.radar_widget = QWidget()
            radar_widget_layout = QVBoxLayout(self.radar_widget)

            # Tạo figure và canvas cho biểu đồ radar
            self.radar_figure = Figure(figsize=(6, 6))
            self.radar_canvas = FigureCanvas(self.radar_figure)
            radar_widget_layout.addWidget(self.radar_canvas)

            # Thêm các widget vào splitter
            self.radar_splitter.addWidget(self.structure_list_widget)
            self.radar_splitter.addWidget(self.radar_widget)
            self.radar_splitter.setSizes([100, 500])  # Kích thước tương đối

            self.radar_layout.addWidget(self.radar_splitter)
        else:
            self.radar_layout.addWidget(
                QLabel("Matplotlib không khả dụng. Không thể hiển thị biểu đồ radar.")
            )

        # Tab 4: Phân tích độ nhạy
        self.sensitivity_tab = QWidget()
        self.sensitivity_layout = QVBoxLayout(self.sensitivity_tab)

        if HAS_MATPLOTLIB:
            # Panel điều khiển ở trên cùng
            control_panel = QWidget()
            control_layout = QHBoxLayout(control_panel)

            # Widget chọn cấu trúc
            structure_widget = QWidget()
            structure_layout = QVBoxLayout(structure_widget)
            structure_layout.addWidget(QLabel("Cấu trúc:"))
            self.sensitivity_structure_combo = QComboBox()
            self.sensitivity_structure_combo.currentIndexChanged.connect(
                self._update_sensitivity_params
            )
            structure_layout.addWidget(self.sensitivity_structure_combo)
            control_layout.addWidget(structure_widget)

            # Widget chọn tham số
            param_widget = QWidget()
            param_layout = QVBoxLayout(param_widget)
            param_layout.addWidget(QLabel("Tham số:"))
            self.sensitivity_param_combo = QComboBox()
            param_layout.addWidget(self.sensitivity_param_combo)
            control_layout.addWidget(param_widget)

            # Widget chọn phạm vi
            range_widget = QWidget()
            range_layout = QGridLayout(range_widget)
            range_layout.addWidget(QLabel("Phạm vi (%):"), 0, 0)

            # Min range
            min_layout = QHBoxLayout()
            min_layout.addWidget(QLabel("Từ:"))
            self.sensitivity_range_min = QSpinBox()
            self.sensitivity_range_min.setRange(-50, 0)
            self.sensitivity_range_min.setValue(-20)
            self.sensitivity_range_min.setSingleStep(5)
            self.sensitivity_range_min.setSuffix("%")
            min_layout.addWidget(self.sensitivity_range_min)
            range_layout.addLayout(min_layout, 0, 1)

            # Max range
            max_layout = QHBoxLayout()
            max_layout.addWidget(QLabel("Đến:"))
            self.sensitivity_range_max = QSpinBox()
            self.sensitivity_range_max.setRange(0, 50)
            self.sensitivity_range_max.setValue(20)
            self.sensitivity_range_max.setSingleStep(5)
            self.sensitivity_range_max.setSuffix("%")
            max_layout.addWidget(self.sensitivity_range_max)
            range_layout.addLayout(max_layout, 0, 2)

            # Số điểm dữ liệu
            points_layout = QHBoxLayout()
            points_layout.addWidget(QLabel("Số điểm:"))
            self.sensitivity_points = QSpinBox()
            self.sensitivity_points.setRange(3, 15)
            self.sensitivity_points.setValue(7)
            self.sensitivity_points.setSingleStep(2)
            points_layout.addWidget(self.sensitivity_points)
            range_layout.addLayout(points_layout, 1, 2)

            control_layout.addWidget(range_widget)

            # Nút phân tích
            analyze_btn = QPushButton("Phân tích")
            analyze_btn.clicked.connect(self._run_sensitivity_analysis)
            control_layout.addWidget(analyze_btn)

            # Thêm panel điều khiển vào layout
            self.sensitivity_layout.addWidget(control_panel)

            # Widget tách
            splitter = QSplitter(Qt.Vertical)

            # Panel hiển thị biểu đồ
            self.sensitivity_chart_panel = QWidget()
            chart_layout = QVBoxLayout(self.sensitivity_chart_panel)

            # Tạo figure cho biểu đồ phân tích độ nhạy
            self.sensitivity_figure = Figure(figsize=(8, 5))
            self.sensitivity_canvas = FigureCanvas(self.sensitivity_figure)
            chart_layout.addWidget(self.sensitivity_canvas)

            # Panel hiển thị bảng kết quả
            self.sensitivity_table_panel = QWidget()
            table_layout = QVBoxLayout(self.sensitivity_table_panel)

            # Bảng kết quả
            self.sensitivity_table = QTableWidget()
            self.sensitivity_table.setColumnCount(3)
            self.sensitivity_table.setHorizontalHeaderLabels(
                ["Giá trị tham số", "Kết quả", "Thay đổi (%)"]
            )
            self.sensitivity_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch
            )
            table_layout.addWidget(self.sensitivity_table)

            # Thêm các panel vào splitter
            splitter.addWidget(self.sensitivity_chart_panel)
            splitter.addWidget(self.sensitivity_table_panel)
            splitter.setSizes([600, 300])  # Thiết lập kích thước ban đầu

            # Thêm splitter vào layout
            self.sensitivity_layout.addWidget(splitter)

            # Thêm tab vào widget chính
            self.tabs.addTab(self.sensitivity_tab, "Độ nhạy")

        # Tab 5: Chi tiết
        self.details_tab = QWidget()
        details_layout = QVBoxLayout(self.details_tab)

        # Khởi tạo panel chi tiết
        self._init_details_panel()
        details_layout.addWidget(self.details_scroll_area)

        # Thêm các tab vào tabwidget
        self.tabs.addTab(self.metrics_tab, "Chỉ số")
        self.tabs.addTab(self.chart_tab, "Biểu đồ")
        self.tabs.addTab(self.radar_tab, "Radar")
        self.tabs.addTab(self.sensitivity_tab, "Độ nhạy")
        self.tabs.addTab(self.details_tab, "Chi tiết")

        # Kết nối sự kiện thay đổi tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Thêm phần tham số
        params_group = QGroupBox("Tham số sinh học")
        params_layout = QFormLayout(params_group)

        # Tham số alpha/beta cho khối u
        self.alpha_beta_tumor_spin = QDoubleSpinBox()
        self.alpha_beta_tumor_spin.setRange(0.1, 50.0)
        self.alpha_beta_tumor_spin.setValue(self.parameters["alpha_beta_tumor"])
        self.alpha_beta_tumor_spin.setSingleStep(0.5)
        self.alpha_beta_tumor_spin.valueChanged.connect(self._on_parameter_changed)
        params_layout.addRow("α/β khối u (Gy):", self.alpha_beta_tumor_spin)

        # Tham số alpha/beta cho mô lành
        self.alpha_beta_normal_spin = QDoubleSpinBox()
        self.alpha_beta_normal_spin.setRange(0.1, 50.0)
        self.alpha_beta_normal_spin.setValue(self.parameters["alpha_beta_normal"])
        self.alpha_beta_normal_spin.setSingleStep(0.5)
        self.alpha_beta_normal_spin.valueChanged.connect(self._on_parameter_changed)
        params_layout.addRow("α/β mô lành (Gy):", self.alpha_beta_normal_spin)

        # Kích thước phân liều
        self.fraction_size_spin = QDoubleSpinBox()
        self.fraction_size_spin.setRange(0.1, 20.0)
        self.fraction_size_spin.setValue(self.parameters["fraction_size"])
        self.fraction_size_spin.setSingleStep(0.1)
        self.fraction_size_spin.valueChanged.connect(self._on_parameter_changed)
        params_layout.addRow("Kích thước phân liều (Gy):", self.fraction_size_spin)

        # Số phân liều
        self.num_fractions_spin = QDoubleSpinBox()
        self.num_fractions_spin.setRange(1, 100)
        self.num_fractions_spin.setValue(self.parameters["num_fractions"])
        self.num_fractions_spin.setDecimals(0)
        self.num_fractions_spin.setSingleStep(1)
        self.num_fractions_spin.valueChanged.connect(self._on_parameter_changed)
        params_layout.addRow("Số phân liều:", self.num_fractions_spin)

        # Mô hình TCP
        self.tcp_model_combo = QComboBox()
        self.tcp_model_combo.addItems(["poisson", "niemierko", "logistic", "webb"])
        self.tcp_model_combo.setCurrentText(self.parameters["tcp_model"])
        self.tcp_model_combo.currentTextChanged.connect(self._on_parameter_changed)
        params_layout.addRow("Mô hình TCP:", self.tcp_model_combo)

        # Mô hình NTCP
        self.ntcp_model_combo = QComboBox()
        self.ntcp_model_combo.addItems(
            ["lkb", "relative_seriality", "logit", "poisson"]
        )
        self.ntcp_model_combo.setCurrentText(self.parameters["ntcp_model"])
        self.ntcp_model_combo.currentTextChanged.connect(self._on_parameter_changed)
        params_layout.addRow("Mô hình NTCP:", self.ntcp_model_combo)

        # Nút cập nhật
        buttons_layout = QHBoxLayout()
        update_button = QPushButton("Cập nhật chỉ số")
        update_button.clicked.connect(self.update_metrics)
        buttons_layout.addWidget(update_button)

        export_button = QPushButton("Xuất báo cáo")
        export_button.clicked.connect(self._export_report)
        buttons_layout.addWidget(export_button)

        # Thêm vào layout chính
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(params_group)
        main_layout.addLayout(buttons_layout)

        # Khởi tạo chi tiết
        self._init_details_panel()

    def _init_details_panel(self):
        """Khởi tạo panel hiển thị chi tiết cấu trúc."""
        # Tạo widget cho chi tiết
        self.detail_widget = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_widget)

        # Tiêu đề
        self.detail_title = QLabel("Chi tiết đánh giá sinh học")
        self.detail_title.setAlignment(Qt.AlignCenter)
        self.detail_title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.detail_layout.addWidget(self.detail_title)

        # Thông tin cơ bản
        self.basic_info_group = QGroupBox("Thông tin cơ bản")
        basic_info_layout = QFormLayout(self.basic_info_group)

        self.detail_structure_name = QLabel("")
        basic_info_layout.addRow("Cấu trúc:", self.detail_structure_name)

        self.detail_structure_type = QLabel("")
        basic_info_layout.addRow("Loại:", self.detail_structure_type)

        self.detail_organ_type = QLabel("")
        basic_info_layout.addRow("Cơ quan:", self.detail_organ_type)

        self.detail_layout.addWidget(self.basic_info_group)

        # Chỉ số sinh học
        self.bio_metrics_group = QGroupBox("Chỉ số sinh học")
        bio_metrics_layout = QFormLayout(self.bio_metrics_group)

        self.detail_eud = QLabel("")
        bio_metrics_layout.addRow("EUD:", self.detail_eud)

        self.detail_tcp = QLabel("")
        bio_metrics_layout.addRow("TCP:", self.detail_tcp)

        self.detail_ntcp = QLabel("")
        bio_metrics_layout.addRow("NTCP:", self.detail_ntcp)

        self.detail_bed = QLabel("")
        bio_metrics_layout.addRow("BED:", self.detail_bed)

        self.detail_layout.addWidget(self.bio_metrics_group)

        # Đánh giá & khuyến nghị
        self.evaluation_group = QGroupBox("Đánh giá & khuyến nghị")
        evaluation_layout = QVBoxLayout(self.evaluation_group)

        self.detail_status = QLabel("")
        evaluation_layout.addWidget(self.detail_status)

        self.detail_concerns = QLabel("")
        self.detail_concerns.setWordWrap(True)
        evaluation_layout.addWidget(self.detail_concerns)

        self.detail_recommendations = QLabel("")
        self.detail_recommendations.setWordWrap(True)
        evaluation_layout.addWidget(self.detail_recommendations)

        self.detail_layout.addWidget(self.evaluation_group)

        # Mô hình thay thế
        self.alt_models_group = QGroupBox("Các mô hình thay thế")
        alt_models_layout = QVBoxLayout(self.alt_models_group)

        self.alt_models_table = QTableWidget()
        self.alt_models_table.setColumnCount(2)
        self.alt_models_table.setHorizontalHeaderLabels(["Mô hình", "Giá trị"])
        self.alt_models_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        alt_models_layout.addWidget(self.alt_models_table)

        self.detail_layout.addWidget(self.alt_models_group)

        # Thêm vào layout chi tiết chính
        self.details_content_layout.addWidget(self.detail_widget)
        self.details_content_layout.addStretch()

    def _on_tab_changed(self, index):
        """Xử lý khi người dùng chuyển tab."""
        if index == 2:  # Radar chart tab
            self._update_radar_chart()
        elif index == 3:  # Sensitivity tab
            self._update_sensitivity_analysis()
        elif index == 4:  # Details tab
            self._update_detail_display()

    def update_metrics(self):
        """Cập nhật tính toán các chỉ số sinh học."""
        if not self.dvh_data:
            logger.warning("Không có dữ liệu DVH, không thể tính toán chỉ số sinh học.")
            return

        # Lấy tham số từ UI
        self._update_parameters_from_ui()

        try:
            if self.bio_evaluator is None:
                # Nếu không có module sinh học, tạo mới
                if HAS_BIO_EVALUATION:
                    self.bio_evaluator = create_biological_evaluation()
                else:
                    logger.error("Không thể tạo module đánh giá sinh học.")
                    QMessageBox.warning(
                        self, "Lỗi", "Module đánh giá sinh học không khả dụng."
                    )
                    return

            # Thiết lập tham số cho module sinh học
            self.bio_evaluator.set_parameters(
                {
                    "tcp_model": self.parameters["tcp_model"],
                    "ntcp_model": self.parameters["ntcp_model"],
                    "alpha_beta_tumor": self.parameters["alpha_beta_tumor"],
                    "alpha_beta_normal": self.parameters["alpha_beta_normal"],
                    "fraction_size": self.parameters["fraction_size"],
                    "num_fractions": int(self.parameters["num_fractions"]),
                    "auto_detect_structure_type": self.parameters["auto_detect_type"],
                }
            )

            # Tính toán các chỉ số sinh học
            self.biological_metrics = self.bio_evaluator.calculate_metrics(
                self.dvh_data,
                num_fractions=int(self.parameters["num_fractions"]),
                dose_per_fraction=self.parameters["fraction_size"],
                organ_mapping=self.structure_organ_mapping,
            )

            self.detailed_results = self.biological_metrics.copy()

            # Cập nhật giao diện
            self._update_table()
            self._update_chart()

            # Cập nhật dữ liệu cấu trúc trong các combo box
            self._update_structure_combos()

            # Nếu đang ở tab radar chart, cập nhật radar chart
            if self.tabs.currentIndex() == 2:
                self._update_radar_chart()
            elif self.tabs.currentIndex() == 3:
                self._update_sensitivity_analysis()
            elif self.tabs.currentIndex() == 4:
                self._update_detail_display()

        except Exception as e:
            logger.error(f"Lỗi khi tính toán chỉ số sinh học: {str(e)}")
            import traceback

            logger.debug(traceback.format_exc())
            QMessageBox.warning(
                self, "Lỗi", f"Lỗi khi tính toán chỉ số sinh học: {str(e)}"
            )

    def _update_structure_combos(self):
        """Cập nhật danh sách cấu trúc trong các combo box."""
        # Lưu lựa chọn hiện tại
        current_structure = (
            self.structures_combo.currentText()
            if self.structures_combo.count() > 0
            else ""
        )
        current_sensitivity_structure = (
            self.sensitivity_structure_combo.currentText()
            if self.sensitivity_structure_combo.count() > 0
            else ""
        )

        # Cập nhật combo box cấu trúc cho radar chart
        self.structures_combo.blockSignals(True)
        self.structures_combo.clear()
        self.structures_combo.addItems(sorted(self.biological_metrics.keys()))
        if current_structure and current_structure in self.biological_metrics:
            index = self.structures_combo.findText(current_structure)
            if index >= 0:
                self.structures_combo.setCurrentIndex(index)
        self.structures_combo.blockSignals(False)

        # Cập nhật combo box cấu trúc cho phân tích độ nhạy
        self.sensitivity_structure_combo.blockSignals(True)
        self.sensitivity_structure_combo.clear()
        self.sensitivity_structure_combo.addItems(
            sorted(self.biological_metrics.keys())
        )
        if (
            current_sensitivity_structure
            and current_sensitivity_structure in self.biological_metrics
        ):
            index = self.sensitivity_structure_combo.findText(
                current_sensitivity_structure
            )
            if index >= 0:
                self.sensitivity_structure_combo.setCurrentIndex(index)
        self.sensitivity_structure_combo.blockSignals(False)

    def _update_table(self):
        """Cập nhật bảng chỉ số sinh học."""
        # Xóa dữ liệu cũ
        self.metrics_table.setRowCount(0)

        if not self.biological_metrics:
            return

        # Thêm dữ liệu mới
        self.metrics_table.setRowCount(len(self.biological_metrics))

        for i, (structure_name, metrics) in enumerate(self.biological_metrics.items()):
            # Tên cấu trúc
            name_item = QTableWidgetItem(structure_name)
            self.metrics_table.setItem(i, 0, name_item)

            # Loại cấu trúc
            # EUD
            eud_item = QTableWidgetItem(f"{metrics.get('eud', 0):.2f}")
            self.metrics_table.setItem(i, 1, eud_item)

            # TCP
            tcp_item = QTableWidgetItem(f"{metrics.get('tcp', 0) * 100:.2f}")
            self.metrics_table.setItem(i, 2, tcp_item)

            # NTCP
            ntcp_item = QTableWidgetItem(f"{metrics.get('ntcp', 0) * 100:.2f}")
            self.metrics_table.setItem(i, 3, ntcp_item)

            # BED
            bed_item = QTableWidgetItem(f"{metrics.get('bed', 0):.2f}")
            self.metrics_table.setItem(i, 4, bed_item)

            # Màu sắc dựa trên loại cấu trúc
            if (
                "PTV" in structure_name
                or "GTV" in structure_name
                or "CTV" in structure_name
            ):
                # Mục tiêu: màu xanh lá cây cho TCP cao
                tcp = metrics.get("tcp", 0)
                if tcp > 0.95:  # TCP > 95%
                    self._set_row_color(i, QColor(200, 255, 200))  # Xanh lá nhạt
                elif tcp > 0.9:  # TCP > 90%
                    self._set_row_color(i, QColor(255, 255, 200))  # Vàng nhạt
                else:
                    self._set_row_color(i, QColor(255, 200, 200))  # Đỏ nhạt
            else:
                # Cơ quan nguy cấp: màu xanh lá cây cho NTCP thấp
                ntcp = metrics.get("ntcp", 0)
                if ntcp < 0.05:  # NTCP < 5%
                    self._set_row_color(i, QColor(200, 255, 200))  # Xanh lá nhạt
                elif ntcp < 0.1:  # NTCP < 10%
                    self._set_row_color(i, QColor(255, 255, 200))  # Vàng nhạt
                else:
                    self._set_row_color(i, QColor(255, 200, 200))  # Đỏ nhạt

    def _set_row_color(self, row: int, color: QColor):
        """Thiết lập màu sắc cho một hàng trong bảng."""
        for col in range(self.metrics_table.columnCount()):
            item = self.metrics_table.item(row, col)
            if item:
                item.setBackground(color)

    def _update_chart(self):
        """Cập nhật biểu đồ TCP/NTCP."""
        # Xóa dữ liệu cũ
        self.chart.removeAllSeries()

        if not self.biological_metrics:
            return

        # Tạo dữ liệu cho biểu đồ
        tcp_set = QBarSet("TCP")
        ntcp_set = QBarSet("NTCP")
        categories = []

        for structure_name, metrics in self.biological_metrics.items():
            tcp = metrics.get("tcp", 0) * 100  # Chuyển sang phần trăm
            ntcp = metrics.get("ntcp", 0) * 100  # Chuyển sang phần trăm

            tcp_set.append(tcp)
            ntcp_set.append(ntcp)
            categories.append(structure_name)

        # Tạo series
        series = QBarSeries()
        series.append(tcp_set)
        series.append(ntcp_set)

        # Thêm series vào biểu đồ
        self.chart.addSeries(series)

        # Tạo trục
        axisX = QBarCategoryAxis()
        axisX.append(categories)
        self.chart.addAxis(axisX, Qt.AlignBottom)
        series.attachAxis(axisX)

        axisY = QValueAxis()
        axisY.setRange(0, 100)
        axisY.setTitleText("Phần trăm (%)")
        self.chart.addAxis(axisY, Qt.AlignLeft)
        series.attachAxis(axisY)

        # Cập nhật tiêu đề
        self.chart.setTitle("TCP/NTCP theo cấu trúc")
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)

    def _update_radar_chart(self):
        """Cập nhật biểu đồ radar cho các chỉ số sinh học."""
        if not HAS_MATPLOTLIB:
            logger.warning("Không thể tạo biểu đồ radar vì thiếu matplotlib")
            return

        if not self.biological_metrics:
            return

        # Xóa biểu đồ cũ nếu có
        try:
            if hasattr(self, "radar_canvas") and self.radar_canvas:
                self.radar_layout.removeWidget(self.radar_canvas)
                self.radar_canvas.deleteLater()
        except Exception as e:
            logger.error(f"Lỗi khi xóa biểu đồ radar cũ: {str(e)}")

        try:
            # Lấy các cấu trúc
            if hasattr(self, "structure_combo") and self.structure_combo:
                selected_structure = self.structure_combo.currentText()
            else:
                selected_structure = next(iter(self.biological_metrics.keys()), None)

            if (
                not selected_structure
                or selected_structure not in self.biological_metrics
            ):
                return

            # Lấy dữ liệu cho biểu đồ radar
            metrics = self.biological_metrics[selected_structure]

            # Xác định loại cấu trúc để hiển thị thông tin phù hợp
            structure_type = self.structure_types.get(selected_structure, "OAR")

            # Lọc các chỉ số sinh học phù hợp với loại cấu trúc
            data = {}

            if structure_type == "TARGET":
                # Cho cấu trúc đích, tập trung vào TCP và EUD
                if "TCP" in metrics:
                    data["TCP (%)"] = metrics["TCP"]
                if "EUD" in metrics:
                    data["EUD (Gy)"] = metrics["EUD"]
                if "BED" in metrics:
                    data["BED (Gy)"] = metrics["BED"]
                # Thêm chỉ số sinh học đặc thù cho PTV
                if "TCP_logistic" in metrics:
                    data["TCP-L (%)"] = metrics["TCP_logistic"]
                if "TCP_poisson" in metrics:
                    data["TCP-P (%)"] = metrics["TCP_poisson"]
                if "TCP_niemierko" in metrics:
                    data["TCP-N (%)"] = metrics["TCP_niemierko"]
                if "g_EUD" in metrics:
                    data["gEUD (Gy)"] = metrics["g_EUD"]
            else:
                # Cho cơ quan nguy cấp, tập trung vào NTCP và EUD
                if "NTCP" in metrics:
                    data["NTCP (%)"] = metrics["NTCP"]
                if "EUD" in metrics:
                    data["EUD (Gy)"] = metrics["EUD"]
                if "BED" in metrics:
                    data["BED (Gy)"] = metrics["BED"]
                # Thêm chỉ số sinh học đặc thù cho OAR
                if "NTCP_lkb" in metrics:
                    data["NTCP-LKB (%)"] = metrics["NTCP_lkb"]
                if "NTCP_logit" in metrics:
                    data["NTCP-Logit (%)"] = metrics["NTCP_logit"]
                if "NTCP_poisson" in metrics:
                    data["NTCP-Poisson (%)"] = metrics["NTCP_poisson"]
                if "NTCP_rs" in metrics:
                    data["NTCP-RS (%)"] = metrics["NTCP_rs"]
                if "g_EUD" in metrics:
                    data["gEUD (Gy)"] = metrics["g_EUD"]

            # Vẽ biểu đồ radar
            self._draw_radar_chart(selected_structure, data, structure_type)
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật biểu đồ radar: {str(e)}")

    def _draw_radar_chart(self, structure_name, metrics_data, structure_type="OAR"):
        """
        Vẽ biểu đồ radar để hiển thị các chỉ số sinh học.

        Args:
            structure_name (str): Tên cấu trúc
            metrics_data (dict): Dữ liệu chỉ số sinh học
            structure_type (str): Loại cấu trúc ('TARGET' hoặc 'OAR')
        """
        if not HAS_MATPLOTLIB or not metrics_data:
            return

        # Xóa nội dung cũ trên figure
        self.radar_figure.clear()

        # Tạo trục radar
        ax = self.radar_figure.add_subplot(111, projection="polar")

        # Chuẩn bị dữ liệu
        categories = []
        values = []
        normalized_values = []  # Giá trị chuẩn hóa để hiển thị trên cùng biểu đồ
        colors = []
        optimal_directions = []  # Hướng tối ưu (max hoặc min) cho mỗi chỉ số

        # Chuẩn bị thang chuẩn hóa để hiển thị các giá trị trên cùng biểu đồ
        normalization_ranges = {
            "TCP (%)": (0, 100),
            "NTCP (%)": (0, 100),
            "EUD (Gy)": (0, 80),
            "gEUD (Gy)": (0, 80),
            "BED (Gy)": (0, 120),
            "Max Dose (Gy)": (0, 100),
            "Mean Dose (Gy)": (0, 80),
            "CI": (0, 1),
            "HI": (0, 2),
        }

        # Tạo dict chỉ số theo loại cấu trúc
        if structure_type == "TARGET":
            metric_info = {
                "TCP (%)": {"optimal": "max", "weight": 1.0, "color": "#1a9641"},
                "EUD (Gy)": {"optimal": "max", "weight": 0.8, "color": "#91cf60"},
                "BED (Gy)": {"optimal": "max", "weight": 0.7, "color": "#a6d96a"},
                "CI": {"optimal": "max", "weight": 0.6, "color": "#ffffbf"},
                "HI": {"optimal": "min", "weight": 0.6, "color": "#fc8d59"},
            }
        else:  # OAR
            metric_info = {
                "NTCP (%)": {"optimal": "min", "weight": 1.0, "color": "#d7191c"},
                "EUD (Gy)": {"optimal": "min", "weight": 0.8, "color": "#fdae61"},
                "BED (Gy)": {"optimal": "min", "weight": 0.7, "color": "#fee08b"},
                "Max Dose (Gy)": {"optimal": "min", "weight": 0.9, "color": "#e6f598"},
                "Mean Dose (Gy)": {"optimal": "min", "weight": 0.9, "color": "#abdda4"},
            }

        # Lọc các chỉ số có trong dữ liệu
        available_metrics = {}
        for metric_name, info in metric_info.items():
            base_name = metric_name.split(" ")[0]  # Lấy phần đầu của tên chỉ số
            for key in metrics_data:
                if base_name.lower() in key.lower():
                    if isinstance(metrics_data[key], dict):
                        value = metrics_data[key].get("value")
                    else:
                        value = metrics_data[key]

                    if value is not None:
                        available_metrics[metric_name] = {
                            "value": value,
                            "optimal": info["optimal"],
                            "weight": info["weight"],
                            "color": info["color"],
                        }
                    break

        # Nếu không có dữ liệu
        if not available_metrics:
            ax.text(
                0,
                0,
                f"Không có dữ liệu cho {structure_name}",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.radar_figure.tight_layout()
            self.radar_canvas.draw()
            return

        # Thu thập dữ liệu và chuẩn hóa
        for metric_name, info in available_metrics.items():
            categories.append(metric_name)
            raw_value = info["value"]
            values.append(raw_value)

            # Chuẩn hóa giá trị để hiển thị trên cùng biểu đồ
            min_val, max_val = normalization_ranges.get(metric_name, (0, 1))
            normalized_value = (
                (raw_value - min_val) / (max_val - min_val)
                if max_val > min_val
                else 0.5
            )

            # Đảo ngược giá trị nếu tối ưu là min (số càng thấp càng tốt)
            if info["optimal"] == "min":
                normalized_value = 1 - normalized_value

            normalized_values.append(normalized_value)
            colors.append(info["color"])
            optimal_directions.append(info["optimal"])

        # Số lượng chỉ số
        N = len(categories)
        if N < 3:
            # Biểu đồ radar cần ít nhất 3 chiều, thêm chiều giả nếu cần
            dummy_count = 3 - N
            categories.extend([""] * dummy_count)
            values.extend([0] * dummy_count)
            normalized_values.extend([0] * dummy_count)
            colors.extend(["#cccccc"] * dummy_count)
            optimal_directions.extend(["max"] * dummy_count)
            N = 3

        # Tạo các góc cho biểu đồ radar
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

        # Khép kín biểu đồ
        categories = categories + [categories[0]]
        normalized_values = normalized_values + [normalized_values[0]]
        values = values + [values[0]]
        angles = angles + [angles[0]]

        # Vẽ biểu đồ radar với giá trị chuẩn hóa
        ax.plot(
            angles, normalized_values, "o-", linewidth=2, color="#5c9dc2", alpha=0.8
        )
        ax.fill(angles, normalized_values, color="#5c9dc2", alpha=0.2)

        # Đặt nhãn cho các trục
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])

        # Thêm thông tin giá trị thực tế lên đồ thị
        for i, (angle, value, normalized, category) in enumerate(
            zip(angles[:-1], values[:-1], normalized_values[:-1], categories[:-1])
        ):
            if category:  # Chỉ hiển thị cho các trục có tên
                # Tính toán vị trí văn bản dựa trên góc
                ha = "left" if -np.pi / 2 <= angle <= np.pi / 2 else "right"
                va = "center"
                offset = 0.1
                text_angle = angle

                # Hiển thị giá trị thực tế
                if "%" in category:
                    value_text = f"{value:.1f}%"
                elif "Gy" in category:
                    value_text = f"{value:.1f} Gy"
                else:
                    value_text = f"{value:.2f}"

                ax.text(
                    text_angle,
                    normalized + offset,
                    value_text,
                    ha=ha,
                    va=va,
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"),
                )

        # Vẽ vòng tròn nền
        self._draw_background_circles(ax)

        # Cài đặt giới hạn trục và tiêu đề
        ax.set_ylim(0, 1.2)  # Giới hạn cho giá trị chuẩn hóa
        ax.set_title(f"Radar Chart: {structure_name} ({structure_type})", fontsize=12)
        self.radar_figure.tight_layout()

        # Hiển thị biểu đồ
        self.radar_canvas.draw()

    def _draw_background_circles(self, ax):
        """
        Vẽ các vòng tròn nền cho biểu đồ radar.

        Args:
            ax: Trục matplotlib để vẽ vòng tròn
        """
        # Vẽ vòng tròn nền
        r_max = 1.0
        for i in range(5):
            r = r_max * i / 4
            circle = plt.Circle(
                (0, 0),
                r,
                transform=ax.transData._b,
                fill=False,
                edgecolor="gray",
                alpha=0.3,
                linestyle=":",
            )
            ax.add_artist(circle)

            # Thêm nhãn giá trị cho vòng tròn
            if i > 0:  # Không hiển thị giá trị 0
                # Hiển thị giá trị dưới dạng phần trăm
                label = f"{i / 4 * 100:.0f}%"
                ax.text(
                    0,
                    r,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="gray",
                    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.1"),
                )

    def _update_detail_display(self, all_metrics=None):
        """Cập nhật hiển thị chi tiết thông số cho cấu trúc đã chọn."""
        # Sử dụng tham số đã truyền vào hoặc metrics đã tính
        if all_metrics is None:
            all_metrics = self.biological_metrics

        if not all_metrics:
            return

        try:
            # Lấy tên cấu trúc đã chọn
            selected_rows = self.metrics_table.selectedItems()
            if not selected_rows:
                return

            row = selected_rows[0].row()
            structure_name = self.metrics_table.item(row, 0).text()

            if structure_name not in all_metrics:
                return

            metrics = all_metrics[structure_name]

            # Tạo chi tiết để hiển thị
            detail_text = f"<h3>Chi tiết cho cấu trúc: {structure_name}</h3>"
            detail_text += (
                f"<p><b>Loại cấu trúc:</b> {metrics.get('type', 'Không xác định')}</p>"
            )

            # Thêm các thông số tính toán
            detail_text += "<h4>Chỉ số sinh học:</h4>"
            detail_text += "<ul>"

            for key, value in metrics.items():
                if key not in ["name", "type"]:
                    if key in ["TCP", "NTCP"]:
                        detail_text += f"<li><b>{key}:</b> {value:.2f}%</li>"
                    else:
                        detail_text += f"<li><b>{key}:</b> {value:.2f} Gy</li>"

            detail_text += "</ul>"

            # Hiển thị thông số tham số nếu có
            if hasattr(self, "bio_eval") and self.bio_eval:
                detail_text += "<h4>Tham số tính toán:</h4>"
                detail_text += "<ul>"

                params = self.parameters
                detail_text += f"<li><b>Số phân liều:</b> {params.get('num_fractions', 'N/A')}</li>"
                detail_text += f"<li><b>Liều mỗi phân liều:</b> {params.get('fraction_size', 'N/A')} Gy</li>"
                detail_text += f"<li><b>Tỷ lệ α/β:</b> {params.get('alpha_beta_ratio', 'N/A')} Gy</li>"

                detail_text += "</ul>"

            # Hiển thị chi tiết nếu chưa có QLabel chi tiết, tạo mới
            if not hasattr(self, "detail_label"):
                # Tạo tab mới cho chi tiết
                self.detail_tab = QWidget()
                detail_layout = QVBoxLayout(self.detail_tab)

                # Tạo widget cuộn cho chi tiết
                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)

                # Tạo widget nội dung
                content_widget = QWidget()
                content_layout = QVBoxLayout(content_widget)

                # Tạo label hiển thị chi tiết
                self.detail_label = QLabel()
                self.detail_label.setTextFormat(Qt.RichText)
                self.detail_label.setWordWrap(True)
                self.detail_label.setText(detail_text)

                content_layout.addWidget(self.detail_label)
                scroll_area.setWidget(content_widget)
                detail_layout.addWidget(scroll_area)

                # Thêm tab vào widget chính
                self.tabs.addTab(self.detail_tab, "Chi tiết")

                # Chuyển đến tab chi tiết
                self.tabs.setCurrentWidget(self.detail_tab)
            else:
                # Cập nhật nội dung
                self.detail_label.setText(detail_text)
                # Chuyển đến tab chi tiết
                self.tabs.setCurrentWidget(self.detail_tab)

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật hiển thị chi tiết: {str(e)}")

    def _on_parameter_changed(self):
        """Xử lý khi người dùng thay đổi tham số."""
        # Phát tín hiệu thông báo tham số đã thay đổi
        self.parameters_changed.emit()

    def _run_sensitivity_analysis(self):
        """
        Thực hiện phân tích độ nhạy cho tham số được chọn trên cấu trúc được chọn.
        """
        if not HAS_MATPLOTLIB:
            return

        structure = self.sensitivity_structure_combo.currentText()
        parameter = self.sensitivity_param_combo.currentText()
        min_range = (
            self.sensitivity_range_min.value() / 100.0
        )  # Convert from percentage
        max_range = (
            self.sensitivity_range_max.value() / 100.0
        )  # Convert from percentage
        num_points = self.sensitivity_points.value()

        if not structure or structure not in self.dvh_data:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một cấu trúc hợp lệ!")
            return

        logger.info(f"Thực hiện phân tích độ nhạy cho {structure}, tham số {parameter}")

        try:
            # Xác định loại cấu trúc (TARGET/OAR)
            structure_type = (
                "TARGET"
                if any(
                    kw in structure.upper() for kw in ["PTV", "CTV", "GTV", "TARGET"]
                )
                else "OAR"
            )

            # Tìm giá trị tham số gốc từ dữ liệu
            base_value = self._get_base_parameter_value(structure, parameter)
            if base_value is None:
                QMessageBox.warning(
                    self,
                    "Cảnh báo",
                    f"Không thể xác định giá trị cơ sở cho tham số {parameter}!",
                )
                return

            # Tạo danh sách các giá trị tham số để phân tích
            delta = max_range - min_range
            step = delta / (num_points - 1) if num_points > 1 else delta
            parameter_values = [
                base_value * (1 + min_range + i * step) for i in range(num_points)
            ]

            # Thực hiện phân tích độ nhạy
            results = {}

            # Xác định đúng loại chỉ số cần tính
            if structure_type == "TARGET":
                metric = "TCP"
            else:
                metric = "NTCP"

            # Lấy giá trị gốc
            base_metric_value = self._get_structure_metric_value(structure, metric)
            if base_metric_value is None:
                QMessageBox.warning(
                    self, "Cảnh báo", f"Không thể xác định giá trị {metric} cơ sở!"
                )
                return

            # Tính toán các giá trị cho từng điểm tham số
            # Đây là mô phỏng tính toán, trong thực tế cần gọi BiologicalEvaluation
            for param_value in parameter_values:
                # Tỷ lệ thay đổi tham số so với giá trị cơ sở
                param_ratio = param_value / base_value

                # Mô phỏng tính toán chỉ số sinh học dựa vào tham số mới
                # Công thức dưới đây chỉ là ước lượng, cần thay bằng công thức chính xác cho từng chỉ số
                if parameter == "alpha" or parameter == "alpha/beta":
                    if structure_type == "TARGET":
                        # TCP tỷ lệ thuận với alpha và alpha/beta
                        metric_value = base_metric_value * param_ratio**0.5
                    else:
                        # NTCP tỷ lệ thuận với alpha và alpha/beta (cho các mô phản ứng sớm)
                        metric_value = base_metric_value * param_ratio**0.5
                elif (
                    parameter == "gamma50"
                    or parameter == "TCD50"
                    or parameter == "TD50"
                ):
                    if structure_type == "TARGET":
                        # TCP tỷ lệ nghịch với TCD50
                        metric_value = base_metric_value / param_ratio**0.7
                    else:
                        # NTCP tỷ lệ nghịch với TD50
                        metric_value = base_metric_value / param_ratio**0.7
                elif parameter == "n" or parameter == "m":
                    # NTCP tỷ lệ thuận với n hoặc m
                    metric_value = base_metric_value * param_ratio**0.3
                else:
                    # Tham số không xác định
                    metric_value = base_metric_value

                # Đảm bảo giá trị nằm trong khoảng 0-100%
                metric_value = (
                    max(0, min(100, metric_value * 100))
                    if metric in ["TCP", "NTCP"]
                    else metric_value
                )

                # Lưu kết quả
                results[param_value] = metric_value

            # Hiển thị kết quả trong biểu đồ và bảng
            self._display_sensitivity_results(structure, parameter, base_value, results)

        except Exception as e:
            logger.error(f"Lỗi khi phân tích độ nhạy: {str(e)}")
            import traceback

            logger.debug(traceback.format_exc())
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi phân tích độ nhạy: {str(e)}")

    def _get_base_parameter_value(self, structure, parameter):
        """
        Lấy giá trị cơ sở của tham số cho cấu trúc.

        Args:
            structure: Tên cấu trúc
            parameter: Tên tham số

        Returns:
            float: Giá trị cơ sở của tham số
        """
        # Giá trị mặc định cho các tham số
        structure_type = (
            "TARGET"
            if any(kw in structure.upper() for kw in ["PTV", "CTV", "GTV", "TARGET"])
            else "OAR"
        )

        if structure_type == "TARGET":
            default_values = {
                "alpha": 0.35,
                "alpha/beta": 10.0,
                "gamma50": 2.0,
                "TCD50": 50.0,
            }
        else:
            default_values = {"n": 0.1, "m": 0.2, "TD50": 50.0, "gamma50": 2.0}

        # TODO: Thay thế bằng việc lấy giá trị từ dữ liệu thực tế
        # từ đối tượng BiologicalEvaluation
        return default_values.get(parameter, 1.0)

    def _get_structure_metric_value(self, structure, metric):
        """
        Lấy giá trị chỉ số sinh học cho cấu trúc.

        Args:
            structure: Tên cấu trúc
            metric: Tên chỉ số (TCP/NTCP)

        Returns:
            float: Giá trị chỉ số sinh học
        """
        # Lấy từ dữ liệu có sẵn nếu có
        if hasattr(self, "all_metrics") and structure in self.all_metrics:
            structure_metrics = self.all_metrics.get(structure, {})

            # Tìm chỉ số phù hợp
            for metric_key, metric_data in structure_metrics.items():
                if metric.lower() in metric_key.lower():
                    if isinstance(metric_data, dict) and "value" in metric_data:
                        return metric_data["value"]
                    elif isinstance(metric_data, (int, float)):
                        return metric_data

        # Giá trị mặc định
        structure_type = (
            "TARGET"
            if any(kw in structure.upper() for kw in ["PTV", "CTV", "GTV", "TARGET"])
            else "OAR"
        )
        if structure_type == "TARGET" and metric == "TCP":
            return 0.85  # 85% TCP là một giá trị khá tốt cho TARGET
        elif structure_type == "OAR" and metric == "NTCP":
            return 0.05  # 5% NTCP là một giá trị hợp lý cho OAR
        else:
            return 0.5  # Giá trị mặc định

    def _display_sensitivity_results(
        self, structure_name, param_name, base_value, results
    ):
        """
        Hiển thị kết quả phân tích độ nhạy.

        Args:
            structure_name: Tên cấu trúc
            param_name: Tên tham số
            base_value: Giá trị cơ sở của tham số
            results: Dict với phần tử {param_value: metric_value}
        """
        # Xóa nội dung cũ của biểu đồ
        self.sensitivity_figure.clear()

        # Xác định loại cấu trúc
        structure_type = (
            "TARGET"
            if any(
                kw in structure_name.upper() for kw in ["PTV", "CTV", "GTV", "TARGET"]
            )
            else "OAR"
        )

        # Xác định tên chỉ số
        metric = "TCP" if structure_type == "TARGET" else "NTCP"

        # Chuẩn bị dữ liệu cho biểu đồ
        param_values = sorted(results.keys())
        metric_values = [results[p] for p in param_values]

        # Tính % thay đổi so với giá trị cơ sở
        base_index = None
        for i, val in enumerate(param_values):
            if abs(val - base_value) < 1e-6:
                base_index = i
                break

        base_metric = (
            results[base_value]
            if base_value in results
            else metric_values[len(metric_values) // 2]
        )
        percent_changes = [
            (val - base_metric) / base_metric * 100 if base_metric != 0 else 0
            for val in metric_values
        ]

        # Tạo biểu đồ
        ax = self.sensitivity_figure.add_subplot(111)

        # Vẽ đường biểu đồ
        (line,) = ax.plot(
            param_values,
            metric_values,
            "o-",
            linewidth=2,
            color="#1f77b4",
            markersize=8,
        )

        # Đánh dấu điểm cơ sở
        if base_index is not None:
            ax.plot(
                base_value,
                metric_values[base_index],
                "o",
                markersize=10,
                markerfacecolor="red",
                markeredgecolor="black",
                label="Giá trị cơ sở",
            )

        # Thêm nhãn và tiêu đề
        ax.set_xlabel(f"Giá trị {param_name}")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_title(
            f"Phân tích độ nhạy cho {structure_name}: {param_name} vs {metric}"
        )

        # Thêm grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # Thêm chú thích
        ax.legend()

        # Thêm chú thích bổ sung
        text_info = (
            f"Cấu trúc: {structure_name} ({structure_type})\n"
            f"Tham số: {param_name}\n"
            f"Giá trị cơ sở: {base_value:.3f}\n"
            f"{metric} cơ sở: {base_metric:.2f}%"
        )

        # Đặt chú thích ở góc trên bên trái
        ax.text(
            0.02,
            0.98,
            text_info,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        )

        # Hiển thị gradients để chỉ ra chiều tốt/xấu
        y_range = ax.get_ylim()
        y_height = y_range[1] - y_range[0]

        # Thiết lập gradient màu dựa trên loại chỉ số
        if (structure_type == "TARGET" and metric == "TCP") or (
            structure_type == "OAR" and metric == "NTCP"
        ):
            # Cho TARGET: TCP cao hơn là tốt hơn
            # Cho OAR: NTCP thấp hơn là tốt hơn
            gradient_colors = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
        else:
            # Ngược lại
            gradient_colors = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]

        # Vẽ arrow với text chỉ thị "Tốt hơn" và "Kém hơn"
        if structure_type == "TARGET" and metric == "TCP":
            ax.annotate(
                "Tốt hơn",
                xy=(0.85, 0.95),
                xytext=(0.85, 0.85),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(facecolor="green", shrink=0.05, width=2),
                horizontalalignment="center",
                verticalalignment="center",
            )
        elif structure_type == "OAR" and metric == "NTCP":
            ax.annotate(
                "Tốt hơn",
                xy=(0.85, 0.05),
                xytext=(0.85, 0.15),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(facecolor="green", shrink=0.05, width=2),
                horizontalalignment="center",
                verticalalignment="center",
            )

        # Căn chỉnh biểu đồ
        self.sensitivity_figure.tight_layout()

        # Hiển thị biểu đồ
        self.sensitivity_canvas.draw()

        # Cập nhật bảng kết quả
        self._update_sensitivity_table(param_name, base_value, results)

    def _update_sensitivity_table(self, param_name, base_value, results):
        """
        Cập nhật bảng thông tin với kết quả phân tích độ nhạy.

        Args:
            param_name: Tên tham số
            base_value: Giá trị tham số cơ sở
            results: Dict với phần tử {param_value: metric_value}
        """
        # Xóa dữ liệu cũ
        self.sensitivity_table.setRowCount(0)

        if not results:
            return

        # Sắp xếp các giá trị tham số
        param_values = sorted(results.keys())

        # Tìm giá trị cơ sở trong danh sách
        base_index = -1
        for i, val in enumerate(param_values):
            if abs(val - base_value) < 1e-6:
                base_index = i
                break

        base_metric = (
            results[base_value]
            if base_value in results
            else results[param_values[len(param_values) // 2]]
        )

        # Thêm dữ liệu vào bảng
        for i, param_value in enumerate(param_values):
            row = self.sensitivity_table.rowCount()
            self.sensitivity_table.insertRow(row)

            # Giá trị tham số
            param_item = QTableWidgetItem(f"{param_value:.4f}")
            if i == base_index:
                # Đánh dấu giá trị cơ sở bằng bold
                font = param_item.font()
                font.setBold(True)
                param_item.setFont(font)
                param_item.setBackground(QColor(230, 230, 255))  # Màu nền nhẹ

            self.sensitivity_table.setItem(row, 0, param_item)

            # Giá trị kết quả
            metric_value = results[param_value]
            metric_item = QTableWidgetItem(f"{metric_value:.2f}")

            if i == base_index:
                font = metric_item.font()
                font.setBold(True)
                metric_item.setFont(font)
                metric_item.setBackground(QColor(230, 230, 255))

            self.sensitivity_table.setItem(row, 1, metric_item)

            # Phần trăm thay đổi so với giá trị cơ sở
            if base_metric != 0:
                percent_change = (metric_value - base_metric) / base_metric * 100
                change_text = f"{percent_change:+.2f}%"
            else:
                percent_change = 0
                change_text = "N/A"

            change_item = QTableWidgetItem(change_text)

            # Định dạng màu sắc cho % thay đổi
            if i != base_index:  # Không áp dụng màu cho giá trị cơ sở
                if abs(percent_change) < 5:
                    # Thay đổi nhỏ - màu trung tính
                    change_item.setForeground(QColor(0, 0, 0))  # Đen
                elif percent_change > 0:
                    # Thay đổi tích cực - màu xanh
                    change_item.setForeground(QColor(0, 120, 0))  # Xanh lá
                else:
                    # Thay đổi tiêu cực - màu đỏ
                    change_item.setForeground(QColor(200, 0, 0))  # Đỏ

            if i == base_index:
                font = change_item.font()
                font.setBold(True)
                change_item.setFont(font)
                change_item.setBackground(QColor(230, 230, 255))

            self.sensitivity_table.setItem(row, 2, change_item)

        # Điều chỉnh kích thước cột cho phù hợp
        self.sensitivity_table.resizeColumnsToContents()

        # Cuộn đến giá trị cơ sở
        if base_index >= 0:
            self.sensitivity_table.scrollToItem(
                self.sensitivity_table.item(base_index, 0),
                QAbstractItemView.PositionAtCenter,
            )


def create_biological_metrics_widget(parent=None):
    """
    Tạo widget hiển thị chỉ số sinh học.

    Args:
        parent: Widget cha

    Returns:
        BiologicalMetricsWidget hoặc QWidget trống nếu không có module sinh học
    """
    if HAS_BIO_MODELS:
        return BiologicalMetricsWidget(parent)
    else:
        # Tạo widget giả nếu không có module sinh học
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        label = QLabel("Module phân tích sinh học không khả dụng")
        layout.addWidget(label)
        return widget


if __name__ == "__main__":
    # Ví dụ sử dụng
    import sys

    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QPainter

        HAS_PYQT = True
    except ImportError as e:
        logger.error(f"Không thể import các thành phần PyQt5 cần thiết: {e}")
        print(f"Không thể import các thành phần PyQt5 cần thiết: {e}")
        HAS_PYQT = False
        sys.exit(1)

    if HAS_PYQT:
        app = QApplication(sys.argv)

        # Tạo dữ liệu DVH giả
        dvh_data = {
            "PTV": {
                "dose_bins": np.linspace(0, 80, 100),
                "volume_bins": np.array([100] * 70 + [0] * 30),
            },
            "Parotid_L": {
                "dose_bins": np.linspace(0, 80, 100),
                "volume_bins": np.array(
                    [100] * 20 + list(np.linspace(100, 0, 50)) + [0] * 30
                ),
            },
            "Spinal_Cord": {
                "dose_bins": np.linspace(0, 80, 100),
                "volume_bins": np.array(
                    [100] * 10 + list(np.linspace(100, 0, 30)) + [0] * 60
                ),
            },
        }

        # Tạo widget
        widget = create_biological_metrics_widget()
        widget.set_dvh_data(dvh_data)
        widget.resize(800, 600)
        widget.show()

        sys.exit(app.exec_())
