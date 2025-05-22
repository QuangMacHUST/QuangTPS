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
        QListWidget,
        QFrame,
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
            min_layout.addWidget(self.sensitivity_range_min)
            range_layout.addLayout(min_layout, 0, 1)

            # Max range
            max_layout = QHBoxLayout()
            max_layout.addWidget(QLabel("Đến:"))
            self.sensitivity_range_max = QSpinBox()
            self.sensitivity_range_max.setRange(0, 50)
            self.sensitivity_range_max.setValue(20)
            max_layout.addWidget(self.sensitivity_range_max)
            range_layout.addLayout(max_layout, 0, 2)

            control_layout.addWidget(range_widget)

            # Nút phân tích
            analyze_button = QPushButton("Phân tích")
            analyze_button.clicked.connect(self._run_sensitivity_analysis)
            control_layout.addWidget(analyze_button)

            self.sensitivity_layout.addWidget(control_panel)

            # Thêm vùng hiển thị biểu đồ độ nhạy
            self.sensitivity_figure = Figure(figsize=(8, 6))
            self.sensitivity_canvas = FigureCanvas(self.sensitivity_figure)
            self.sensitivity_layout.addWidget(self.sensitivity_canvas, 3)

            # Thêm bảng kết quả
            self.sensitivity_results_table = QTableWidget()
            self.sensitivity_results_table.setColumnCount(3)
            self.sensitivity_results_table.setHorizontalHeaderLabels(
                ["Thay đổi tham số (%)", "Giá trị mới", "Thay đổi kết quả (%)"]
            )
            self.sensitivity_layout.addWidget(self.sensitivity_results_table, 1)
        else:
            self.sensitivity_layout.addWidget(
                QLabel(
                    "Matplotlib không khả dụng. Không thể hiển thị phân tích độ nhạy."
                )
            )

        # Tab 5: Chi tiết
        self.details_tab = QWidget()
        self.details_layout = QVBoxLayout(self.details_tab)

        # Thêm panel chi tiết
        self._init_details_panel()

        # Thêm các tab vào widget
        self.tabs.addTab(self.metrics_tab, "Bảng chỉ số")
        self.tabs.addTab(self.chart_tab, "Biểu đồ")
        self.tabs.addTab(self.radar_tab, "Biểu đồ Radar")
        self.tabs.addTab(self.sensitivity_tab, "Phân tích độ nhạy")
        self.tabs.addTab(self.details_tab, "Chi tiết")

        # Kết nối sự kiện thay đổi tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Thêm các widget vào layout chính
        main_layout.addWidget(self.tabs)

    def _init_details_panel(self):
        """Khởi tạo panel hiển thị chi tiết về đánh giá sinh học."""
        # Tạo layout chính cho tab chi tiết
        details_layout = QVBoxLayout()

        # Tạo scroll area để có thể cuộn khi có nhiều thông tin
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Tạo form layout cho các trường thông tin
        form_layout = QFormLayout()

        # Thông tin cơ bản về cấu trúc
        self.detail_structure_name = QLabel("Chưa chọn")
        self.detail_structure_name.setStyleSheet("font-weight: bold; font-size: 14px;")
        form_layout.addRow("Cấu trúc:", self.detail_structure_name)

        self.detail_structure_type = QLabel("")
        form_layout.addRow("Loại cấu trúc:", self.detail_structure_type)

        self.detail_organ_type = QLabel("")
        form_layout.addRow("Loại cơ quan:", self.detail_organ_type)

        # Tạo đường kẻ phân cách
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        # Tạo group box cho các chỉ số sinh học
        metrics_group = QGroupBox("Chỉ số sinh học")
        metrics_layout = QFormLayout(metrics_group)

        self.detail_eud = QLabel("")
        metrics_layout.addRow("EUD:", self.detail_eud)

        self.detail_tcp = QLabel("")
        metrics_layout.addRow("TCP:", self.detail_tcp)

        self.detail_ntcp = QLabel("")
        metrics_layout.addRow("NTCP:", self.detail_ntcp)

        self.detail_bed = QLabel("")
        metrics_layout.addRow("BED:", self.detail_bed)

        # Tạo group box cho đánh giá
        evaluation_group = QGroupBox("Đánh giá")
        evaluation_layout = QVBoxLayout(evaluation_group)

        self.detail_status = QLabel("")
        self.detail_status.setStyleSheet("font-weight: bold; font-size: 12px;")
        evaluation_layout.addWidget(self.detail_status)

        # Tạo danh sách khuyến nghị
        evaluation_layout.addWidget(QLabel("Khuyến nghị:"))
        self.detail_recommendations = QListWidget()
        self.detail_recommendations.setMaximumHeight(100)
        evaluation_layout.addWidget(self.detail_recommendations)

        # Tạo group box cho tham số tính toán
        params_group = QGroupBox("Tham số tính toán")
        params_layout = QFormLayout(params_group)

        self.detail_num_fractions = QLabel(f"{self.parameters.get('num_fractions', 0)}")
        params_layout.addRow("Số phân liều:", self.detail_num_fractions)

        self.detail_fraction_size = QLabel(
            f"{self.parameters.get('fraction_size', 0)} Gy"
        )
        params_layout.addRow("Liều mỗi phân liều:", self.detail_fraction_size)

        self.detail_alpha_beta = QLabel("")
        params_layout.addRow("Tỷ lệ α/β:", self.detail_alpha_beta)

        self.detail_model = QLabel("")
        params_layout.addRow("Mô hình sử dụng:", self.detail_model)

        # Thêm các widget vào layout chính
        scroll_layout.addLayout(form_layout)
        scroll_layout.addWidget(line)
        scroll_layout.addWidget(metrics_group)
        scroll_layout.addWidget(evaluation_group)
        scroll_layout.addWidget(params_group)
        scroll_layout.addStretch()

        # Thiết lập scroll area
        scroll_area.setWidget(scroll_content)

        # Thêm scroll area vào layout chính
        details_layout.addWidget(scroll_area)

        # Thiết lập layout cho tab chi tiết
        self.details_layout.addLayout(details_layout)

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
        """Cập nhật biểu đồ radar."""
        if not HAS_MATPLOTLIB:
            return

        # Lấy cấu trúc được chọn
        selected_structure = self.structures_combo.currentText()
        if not selected_structure or not self.biological_metrics:
            return

        # Xóa biểu đồ cũ
        self.radar_figure.clear()

        # Lấy thông tin cấu trúc
        structure_type = self.structure_types.get(selected_structure, "OAR")

        # Vẽ biểu đồ radar mới
        self._draw_radar_chart(
            selected_structure, self.biological_metrics, structure_type
        )

        # Cập nhật canvas
        self.radar_canvas.draw()

    def _draw_radar_chart(self, structure_name, metrics_data, structure_type="OAR"):
        """
        Vẽ biểu đồ radar cho một cấu trúc.

        Args:
            structure_name: Tên cấu trúc
            metrics_data: Dữ liệu chỉ số sinh học
            structure_type: Loại cấu trúc (TARGET/OAR)
        """
        if not HAS_MATPLOTLIB or structure_name not in metrics_data:
            return

        # Lấy dữ liệu cho cấu trúc
        metrics = metrics_data[structure_name]

        # Tạo subplot với tọa độ cực
        ax = self.radar_figure.add_subplot(111, polar=True)

        # Xác định các chỉ số sẽ hiển thị tùy theo loại cấu trúc
        if structure_type == "TARGET":
            # Các chỉ số cho cấu trúc mục tiêu
            categories = ["TCP", "EUD", "CI", "HI", "Coverage"]

            # Lấy giá trị cho từng chỉ số
            tcp = metrics.get("tcp", 0) * 100  # Chuyển sang phần trăm
            eud = min(
                metrics.get("eud", 0) / 80, 1.0
            )  # Chuẩn hóa EUD (giả sử max 80Gy)
            ci = min(
                1.0 / max(metrics.get("conformity_index", 1), 0.5), 1.0
            )  # CI càng gần 1 càng tốt
            hi = min(
                1.0 / max(metrics.get("homogeneity_index", 1), 0.5), 1.0
            )  # HI càng gần 1 càng tốt
            coverage = metrics.get("coverage", 0) * 100  # Chuyển sang phần trăm

            values = [tcp / 100, eud, ci, hi, coverage / 100]  # Chuẩn hóa về thang 0-1

            # Màu sắc cho TARGET (đỏ)
            color = "red"

            # Giá trị hiển thị
            display_values = [
                f"{tcp:.1f}%",
                f"{metrics.get('eud', 0):.1f}Gy",
                f"{metrics.get('conformity_index', 1):.2f}",
                f"{metrics.get('homogeneity_index', 1):.2f}",
                f"{coverage:.1f}%",
            ]
        else:
            # Các chỉ số cho cơ quan nguy cấp
            categories = ["NTCP", "Mean Dose", "Max Dose", "EUD", "Sparing"]

            # Lấy giá trị cho từng chỉ số
            ntcp = metrics.get("ntcp", 0) * 100  # Chuyển sang phần trăm
            mean_dose = min(
                metrics.get("mean_dose", 0) / 50, 1.0
            )  # Chuẩn hóa mean dose (giả sử max 50Gy)
            max_dose = min(
                metrics.get("max_dose", 0) / 80, 1.0
            )  # Chuẩn hóa max dose (giả sử max 80Gy)
            eud = min(
                metrics.get("eud", 0) / 50, 1.0
            )  # Chuẩn hóa EUD (giả sử max 50Gy)
            sparing = 1.0 - metrics.get(
                "percent_volume_threshold", 0
            )  # Phần thể tích được bảo vệ

            # Đảo ngược giá trị NTCP (thấp là tốt)
            ntcp_inv = 1.0 - (ntcp / 100)
            mean_dose_inv = 1.0 - mean_dose
            max_dose_inv = 1.0 - max_dose
            eud_inv = 1.0 - eud

            values = [
                ntcp_inv,
                mean_dose_inv,
                max_dose_inv,
                eud_inv,
                sparing,
            ]  # Chuẩn hóa về thang 0-1

            # Màu sắc cho OAR (xanh lam)
            color = "blue"

            # Giá trị hiển thị
            display_values = [
                f"{ntcp:.1f}%",
                f"{metrics.get('mean_dose', 0):.1f}Gy",
                f"{metrics.get('max_dose', 0):.1f}Gy",
                f"{metrics.get('eud', 0):.1f}Gy",
                f"{sparing * 100:.1f}%",
            ]

        # Số lượng biến
        N = len(categories)

        # Góc cho mỗi trục
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Đóng đường

        # Thêm giá trị đầu tiên vào cuối để đóng đường
        values += values[:1]

        # Vẽ các đường tròn nền
        self._draw_background_circles(ax)

        # Vẽ đường biểu đồ
        ax.plot(angles, values, linewidth=2, linestyle="solid", color=color)
        ax.fill(angles, values, color=color, alpha=0.25)

        # Thiết lập các trục
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Thêm nhãn giá trị
        for i, (angle, value, display) in enumerate(
            zip(angles[:-1], values[:-1], display_values)
        ):
            # Tính toán vị trí để hiển thị giá trị
            ha = "left" if 0 <= angle < np.pi else "right"
            offset = 0.1 if 0 <= angle < np.pi else -0.1

            # Hiển thị giá trị thực tế (không chuẩn hóa)
            ax.text(angle, value + 0.1, display, ha=ha, va="center", fontsize=9)

        # Thêm tiêu đề
        title = f"Đánh giá sinh học: {structure_name}"
        ax.set_title(title, fontsize=12, fontweight="bold", y=1.08)

        # Đặt giới hạn trục r từ 0 đến 1
        ax.set_ylim(0, 1)

        # Ẩn nhãn giá trị trên trục r
        ax.set_yticklabels([])

    def _draw_background_circles(self, ax):
        """
        Vẽ các đường tròn nền cho biểu đồ radar.

        Args:
            ax: Trục biểu đồ
        """
        # Vẽ các đường tròn đồng tâm
        circles = [0.2, 0.4, 0.6, 0.8, 1.0]
        for circle in circles:
            ax.plot(
                np.linspace(0, 2 * np.pi, 100),
                [circle] * 100,
                color="gray",
                linestyle="--",
                linewidth=0.5,
                alpha=0.7,
            )

        # Tạo lưới từ tâm ra ngoài
        ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

        # Thêm nhãn cho các đường tròn
        labels = ["20%", "40%", "60%", "80%", "100%"]
        for circle, label in zip(circles, labels):
            ax.text(
                0, circle, label, ha="center", va="bottom", color="gray", fontsize=8
            )

    def _update_detail_display(self, all_metrics=None):
        """
        Cập nhật hiển thị chi tiết cho cấu trúc được chọn.

        Args:
            all_metrics: Từ điển chứa tất cả các chỉ số sinh học
        """
        # Nếu không có all_metrics, sử dụng dữ liệu hiện tại
        if all_metrics is None:
            all_metrics = self.biological_metrics

        # Lấy cấu trúc được chọn
        selected_row = self.metrics_table.currentRow()
        if selected_row < 0 or selected_row >= self.metrics_table.rowCount():
            return

        structure_name = self.metrics_table.item(selected_row, 0).text()

        if structure_name not in all_metrics:
            return

        metrics = all_metrics[structure_name]
        structure_type = metrics.get("type", "OAR")

        # Cập nhật thông tin cơ bản
        self.detail_structure_name.setText(structure_name)
        self.detail_structure_type.setText(structure_type)

        # Cập nhật thông tin cơ quan
        organ_type = metrics.get("organ_type", "unknown")
        self.detail_organ_type.setText(organ_type)

        # Cập nhật chỉ số sinh học
        eud = metrics.get("eud", 0)
        self.detail_eud.setText(f"{eud:.2f} Gy")

        # Cập nhật TCP/NTCP dựa vào loại cấu trúc
        if structure_type == "TARGET":
            tcp = metrics.get("tcp", 0) * 100  # Chuyển sang phần trăm
            self.detail_tcp.setText(f"{tcp:.2f}%")

            # Đặt màu sắc dựa vào giá trị TCP
            if tcp >= 95:
                self.detail_tcp.setStyleSheet("color: green; font-weight: bold;")
            elif tcp >= 90:
                self.detail_tcp.setStyleSheet("color: lightgreen; font-weight: bold;")
            elif tcp >= 80:
                self.detail_tcp.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.detail_tcp.setStyleSheet("color: red; font-weight: bold;")

            # Ẩn NTCP cho TARGET
            self.detail_ntcp.setText("N/A")
            self.detail_ntcp.setStyleSheet("")
        else:
            ntcp = metrics.get("ntcp", 0) * 100  # Chuyển sang phần trăm
            self.detail_ntcp.setText(f"{ntcp:.2f}%")

            # Đặt màu sắc dựa vào giá trị NTCP
            if ntcp <= 1:
                self.detail_ntcp.setStyleSheet("color: green; font-weight: bold;")
            elif ntcp <= 5:
                self.detail_ntcp.setStyleSheet("color: lightgreen; font-weight: bold;")
            elif ntcp <= 10:
                self.detail_ntcp.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.detail_ntcp.setStyleSheet("color: red; font-weight: bold;")

            # Ẩn TCP cho OAR
            self.detail_tcp.setText("N/A")
            self.detail_tcp.setStyleSheet("")

        # Cập nhật BED
        bed = metrics.get("bed", 0)
        self.detail_bed.setText(f"{bed:.2f} Gy")

        # Cập nhật đánh giá
        evaluation = ""
        recommendations = []

        if structure_type == "TARGET":
            tcp = metrics.get("tcp", 0)
            if tcp >= 0.95:
                evaluation = "Rất tốt - TCP cao"
                self.detail_status.setStyleSheet("color: green; font-weight: bold;")
            elif tcp >= 0.9:
                evaluation = "Tốt - TCP đạt yêu cầu"
                self.detail_status.setStyleSheet(
                    "color: lightgreen; font-weight: bold;"
                )
            elif tcp >= 0.8:
                evaluation = "Chấp nhận được - TCP khá"
                self.detail_status.setStyleSheet("color: orange; font-weight: bold;")
                recommendations.append("Xem xét tăng liều để cải thiện TCP")
            else:
                evaluation = "Cần cải thiện - TCP thấp"
                self.detail_status.setStyleSheet("color: red; font-weight: bold;")
                recommendations.append("Cần tăng liều đáng kể để cải thiện TCP")
                recommendations.append("Kiểm tra lại kế hoạch điều trị")
        else:
            ntcp = metrics.get("ntcp", 0)
            if ntcp <= 0.01:
                evaluation = "Rất tốt - NTCP rất thấp"
                self.detail_status.setStyleSheet("color: green; font-weight: bold;")
            elif ntcp <= 0.05:
                evaluation = "Tốt - NTCP thấp"
                self.detail_status.setStyleSheet(
                    "color: lightgreen; font-weight: bold;"
                )
            elif ntcp <= 0.1:
                evaluation = "Chấp nhận được - NTCP trung bình"
                self.detail_status.setStyleSheet("color: orange; font-weight: bold;")
                recommendations.append("Xem xét giảm liều để cải thiện NTCP")
            else:
                evaluation = "Cần cải thiện - NTCP cao"
                self.detail_status.setStyleSheet("color: red; font-weight: bold;")
                recommendations.append("Cần giảm liều đáng kể để cải thiện NTCP")
                recommendations.append("Kiểm tra lại kế hoạch điều trị")

        self.detail_status.setText(evaluation)

        # Hiển thị khuyến nghị
        self.detail_recommendations.clear()
        for recommendation in recommendations:
            self.detail_recommendations.addItem(recommendation)

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
        self.sensitivity_results_table.setRowCount(0)

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
            row = self.sensitivity_results_table.rowCount()
            self.sensitivity_results_table.insertRow(row)

            # Giá trị tham số
            param_item = QTableWidgetItem(f"{param_value:.4f}")
            if i == base_index:
                # Đánh dấu giá trị cơ sở bằng bold
                font = param_item.font()
                font.setBold(True)
                param_item.setFont(font)
                param_item.setBackground(QColor(230, 230, 255))  # Màu nền nhẹ

            self.sensitivity_results_table.setItem(row, 0, param_item)

            # Giá trị kết quả
            metric_value = results[param_value]
            metric_item = QTableWidgetItem(f"{metric_value:.2f}")

            if i == base_index:
                font = metric_item.font()
                font.setBold(True)
                metric_item.setFont(font)
                metric_item.setBackground(QColor(230, 230, 255))

            self.sensitivity_results_table.setItem(row, 1, metric_item)

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

            self.sensitivity_results_table.setItem(row, 2, change_item)

        # Điều chỉnh kích thước cột cho phù hợp
        self.sensitivity_results_table.resizeColumnsToContents()

        # Cuộn đến giá trị cơ sở
        if base_index >= 0:
            self.sensitivity_results_table.scrollToItem(
                self.sensitivity_results_table.item(base_index, 0),
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
