#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Widget hiển thị các chỉ số sinh học cho đánh giá kế hoạch xạ trị.

Widget này hiển thị các chỉ số sinh học như TCP, NTCP, EUD từ dữ liệu DVH
của kế hoạch xạ trị, giúp đánh giá kế hoạch từ góc độ sinh học thay vì chỉ
dựa vào các chỉ số vật lý đơn thuần.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

# Biến toàn cục để kiểm tra PyQt5 khả dụng
HAS_PYQT = False

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QComboBox,
        QPushButton,
        QGroupBox,
        QFormLayout,
        QSplitter,
        QFrame,
        QCheckBox,
        QTabWidget,
        QSpinBox,
        QDoubleSpinBox,
        QApplication,
        QScrollArea,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QFont, QColor, QPainter

    HAS_PYQT = True
    HAS_QT = True
except ImportError:
    logging.error("PyQt5 không khả dụng. Widget chỉ số sinh học không hoạt động.")
    HAS_QT = False

    # Tạo các lớp giả cho IDE
    class QWidget:
        pass

    class pyqtSignal:
        pass


try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    HAS_MPL = True
except ImportError:
    logging.error("matplotlib không khả dụng. Biểu đồ chỉ số sinh học không hoạt động.")
    HAS_MPL = False

try:
    from quangtps.evaluation.biological.biological_models import (
        calculate_tcp,
        calculate_ntcp,
        calculate_eud,
        calculate_biological_metrics,
        get_organ_specific_parameters,
    )

    HAS_BIO_MODULE = True
except ImportError:
    logging.error("Module phân tích sinh học không khả dụng.")
    HAS_BIO_MODULE = False

logger = logging.getLogger(__name__)


class BiologicalMetricsTable(QTableWidget):
    """Bảng hiển thị các chỉ số sinh học."""

    def __init__(self, parent=None):
        super().__init__(0, 6, parent)
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện bảng chỉ số sinh học."""
        self.setHorizontalHeaderLabels(
            ["Cấu trúc", "EUD (Gy)", "TCP (%)", "NTCP (%)", "gEUD (Gy)", "BED (Gy)"]
        )

        # Thiết lập độ rộng cột
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)

        # Thiết lập các thuộc tính khác
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)

    def clear_data(self):
        """Xóa dữ liệu hiện có trong bảng."""
        self.setRowCount(0)

    def update_metrics(self, metrics_data: Dict[str, Dict[str, Any]]):
        """
        Cập nhật bảng với dữ liệu chỉ số sinh học mới.

        Parameters
        ----------
        metrics_data : Dict[str, Dict[str, Any]]
            Từ điển chỉ số sinh học, với format:
            {
                structure_name: {
                    'EUD': float,
                    'TCP': float,
                    'NTCP': float,
                    'gEUD': float,
                    'BED': float
                }
            }
        """
        self.clear_data()

        if not metrics_data:
            return

        row_idx = 0
        for struct_name, metrics in metrics_data.items():
            self.insertRow(row_idx)

            # Tên cấu trúc
            self.setItem(row_idx, 0, QTableWidgetItem(struct_name))

            # EUD
            eud_item = QTableWidgetItem(f"{metrics.get('EUD', 0.0):.2f}")
            self.setItem(row_idx, 1, eud_item)

            # TCP
            tcp = metrics.get("TCP", 0.0)
            tcp_item = QTableWidgetItem(f"{tcp:.2f}")
            if tcp > 0.0:
                # Màu sắc dựa trên giá trị TCP
                if tcp >= 95.0:
                    tcp_item.setBackground(QColor(200, 255, 200))  # Xanh lá nhạt
                elif tcp >= 80.0:
                    tcp_item.setBackground(QColor(255, 255, 200))  # Vàng nhạt
                else:
                    tcp_item.setBackground(QColor(255, 200, 200))  # Đỏ nhạt
            self.setItem(row_idx, 2, tcp_item)

            # NTCP
            ntcp = metrics.get("NTCP", 0.0)
            ntcp_item = QTableWidgetItem(f"{ntcp:.2f}")
            if ntcp > 0.0:
                # Màu sắc dựa trên giá trị NTCP
                if ntcp <= 5.0:
                    ntcp_item.setBackground(QColor(200, 255, 200))  # Xanh lá nhạt
                elif ntcp <= 15.0:
                    ntcp_item.setBackground(QColor(255, 255, 200))  # Vàng nhạt
                else:
                    ntcp_item.setBackground(QColor(255, 200, 200))  # Đỏ nhạt
            self.setItem(row_idx, 3, ntcp_item)

            # gEUD
            geud_item = QTableWidgetItem(f"{metrics.get('gEUD', 0.0):.2f}")
            self.setItem(row_idx, 4, geud_item)

            # BED
            bed_item = QTableWidgetItem(f"{metrics.get('BED', 0.0):.2f}")
            self.setItem(row_idx, 5, bed_item)

            row_idx += 1

        self.resizeRowsToContents()


class TCPNTCPPlot(QWidget):
    """Widget hiển thị đồ thị TCP/NTCP."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = None
        self.canvas = None
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện plot TCP/NTCP."""
        layout = QVBoxLayout(self)

        if HAS_MPL:
            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            layout.addWidget(
                QLabel("matplotlib không khả dụng. Không thể hiển thị đồ thị.")
            )

    def plot_data(self, structures_data: Dict[str, Dict[str, Any]]):
        """
        Vẽ đồ thị TCP/NTCP cho các cấu trúc.

        Parameters
        ----------
        structures_data : Dict[str, Dict[str, Any]]
            Dữ liệu các cấu trúc, với format:
            {
                structure_name: {
                    'type': 'TARGET'|'OAR',
                    'TCP': float,
                    'NTCP': float,
                    'value': float  # TCP hoặc NTCP tùy theo loại cấu trúc
                }
            }
        """
        if not HAS_MPL or not structures_data:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Phân tách các cấu trúc thành TARGET và OAR
        targets = {}
        oars = {}

        for name, data in structures_data.items():
            if data.get("type", "").upper() == "TARGET":
                targets[name] = data.get("TCP", 0.0)
            else:
                oars[name] = data.get("NTCP", 0.0)

        # Vẽ biểu đồ cột cho TCP của các TARGET
        if targets:
            names = list(targets.keys())
            values = list(targets.values())
            x_pos = np.arange(len(names))
            bars1 = ax.bar(x_pos, values, width=0.4, color="b", alpha=0.7, label="TCP")

            # Thêm nhãn giá trị
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    1.01 * height,
                    f"{values[i]:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Vẽ biểu đồ cột cho NTCP của các OAR
        if oars:
            names = list(oars.keys())
            values = list(oars.values())
            x_pos = np.arange(len(names)) + (len(targets) + 1 if targets else 0)
            bars2 = ax.bar(x_pos, values, width=0.4, color="r", alpha=0.7, label="NTCP")

            # Thêm nhãn giá trị
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    1.01 * height,
                    f"{values[i]:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Thiết lập trục và nhãn
        all_names = (
            list(targets.keys())
            + ([""] if targets and oars else [])
            + list(oars.keys())
        )
        x_pos_all = np.arange(len(all_names))
        ax.set_xticks(x_pos_all)
        ax.set_xticklabels(all_names, rotation=45, ha="right")
        ax.set_ylabel("Probability (%)")
        ax.set_title("TCP và NTCP cho các cấu trúc")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_ylim(0, 105)  # Giới hạn y-axis từ 0 đến 105%

        self.figure.tight_layout()
        self.canvas.draw()


class ModelParametersWidget(QWidget):
    """Widget cho phép điều chỉnh các tham số của mô hình sinh học."""

    parametersChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện điều chỉnh tham số."""
        main_layout = QVBoxLayout(self)

        # Nhóm tham số TCP
        tcp_group = QGroupBox("Tham số TCP")
        tcp_form = QFormLayout()

        self.tcd50_spin = QDoubleSpinBox()
        self.tcd50_spin.setRange(30.0, 100.0)
        self.tcd50_spin.setValue(50.0)
        self.tcd50_spin.setDecimals(1)
        self.tcd50_spin.setSingleStep(1.0)
        self.tcd50_spin.setSuffix(" Gy")
        self.tcd50_spin.valueChanged.connect(self._on_parameter_changed)
        tcp_form.addRow("TCD50:", self.tcd50_spin)

        self.gamma50_tcp_spin = QDoubleSpinBox()
        self.gamma50_tcp_spin.setRange(0.1, 10.0)
        self.gamma50_tcp_spin.setValue(2.0)
        self.gamma50_tcp_spin.setDecimals(1)
        self.gamma50_tcp_spin.setSingleStep(0.1)
        self.gamma50_tcp_spin.valueChanged.connect(self._on_parameter_changed)
        tcp_form.addRow("Gamma50:", self.gamma50_tcp_spin)

        self.alpha_beta_tumor_spin = QDoubleSpinBox()
        self.alpha_beta_tumor_spin.setRange(1.0, 20.0)
        self.alpha_beta_tumor_spin.setValue(10.0)
        self.alpha_beta_tumor_spin.setDecimals(1)
        self.alpha_beta_tumor_spin.setSingleStep(1.0)
        self.alpha_beta_tumor_spin.setSuffix(" Gy")
        self.alpha_beta_tumor_spin.valueChanged.connect(self._on_parameter_changed)
        tcp_form.addRow("α/β (Khối u):", self.alpha_beta_tumor_spin)

        tcp_group.setLayout(tcp_form)

        # Nhóm tham số NTCP
        ntcp_group = QGroupBox("Tham số NTCP")
        ntcp_form = QFormLayout()

        self.td50_spin = QDoubleSpinBox()
        self.td50_spin.setRange(20.0, 100.0)
        self.td50_spin.setValue(80.0)
        self.td50_spin.setDecimals(1)
        self.td50_spin.setSingleStep(1.0)
        self.td50_spin.setSuffix(" Gy")
        self.td50_spin.valueChanged.connect(self._on_parameter_changed)
        ntcp_form.addRow("TD50:", self.td50_spin)

        self.n_spin = QDoubleSpinBox()
        self.n_spin.setRange(0.01, 2.0)
        self.n_spin.setValue(0.1)
        self.n_spin.setDecimals(2)
        self.n_spin.setSingleStep(0.05)
        self.n_spin.valueChanged.connect(self._on_parameter_changed)
        ntcp_form.addRow("n:", self.n_spin)

        self.m_spin = QDoubleSpinBox()
        self.m_spin.setRange(0.01, 1.0)
        self.m_spin.setValue(0.1)
        self.m_spin.setDecimals(2)
        self.m_spin.setSingleStep(0.05)
        self.m_spin.valueChanged.connect(self._on_parameter_changed)
        ntcp_form.addRow("m:", self.m_spin)

        self.alpha_beta_normal_spin = QDoubleSpinBox()
        self.alpha_beta_normal_spin.setRange(0.5, 10.0)
        self.alpha_beta_normal_spin.setValue(3.0)
        self.alpha_beta_normal_spin.setDecimals(1)
        self.alpha_beta_normal_spin.setSingleStep(0.5)
        self.alpha_beta_normal_spin.setSuffix(" Gy")
        self.alpha_beta_normal_spin.valueChanged.connect(self._on_parameter_changed)
        ntcp_form.addRow("α/β (OAR):", self.alpha_beta_normal_spin)

        ntcp_group.setLayout(ntcp_form)

        # Nhóm tham số khác
        other_group = QGroupBox("Tham số khác")
        other_form = QFormLayout()

        self.fraction_size_spin = QDoubleSpinBox()
        self.fraction_size_spin.setRange(0.5, 10.0)
        self.fraction_size_spin.setValue(2.0)
        self.fraction_size_spin.setDecimals(1)
        self.fraction_size_spin.setSingleStep(0.5)
        self.fraction_size_spin.setSuffix(" Gy")
        self.fraction_size_spin.valueChanged.connect(self._on_parameter_changed)
        other_form.addRow("Kích thước phân liều:", self.fraction_size_spin)

        self.use_organ_params_cb = QCheckBox("Sử dụng tham số cơ quan đặc trưng")
        self.use_organ_params_cb.setChecked(True)
        self.use_organ_params_cb.stateChanged.connect(self._on_parameter_changed)
        other_form.addRow(self.use_organ_params_cb)

        other_group.setLayout(other_form)

        # Nút cập nhật
        update_button = QPushButton("Cập nhật tính toán")
        update_button.clicked.connect(self._on_update_clicked)

        # Thêm các nhóm vào layout chính
        main_layout.addWidget(tcp_group)
        main_layout.addWidget(ntcp_group)
        main_layout.addWidget(other_group)
        main_layout.addWidget(update_button)
        main_layout.addStretch()

    def _on_parameter_changed(self):
        """Xử lý khi thay đổi tham số."""
        # Cập nhật trạng thái kích hoạt/vô hiệu hóa các control dựa vào checkbox
        use_organ_params = self.use_organ_params_cb.isChecked()
        self.tcd50_spin.setEnabled(not use_organ_params)
        self.gamma50_tcp_spin.setEnabled(not use_organ_params)
        self.alpha_beta_tumor_spin.setEnabled(not use_organ_params)
        self.td50_spin.setEnabled(not use_organ_params)
        self.n_spin.setEnabled(not use_organ_params)
        self.m_spin.setEnabled(not use_organ_params)
        self.alpha_beta_normal_spin.setEnabled(not use_organ_params)

    def _on_update_clicked(self):
        """Xử lý khi nhấn nút cập nhật."""
        # Thu thập các tham số hiện tại
        params = {
            "use_organ_params": self.use_organ_params_cb.isChecked(),
            "tcd50": self.tcd50_spin.value(),
            "gamma50": self.gamma50_tcp_spin.value(),
            "alpha_beta_tumor": self.alpha_beta_tumor_spin.value(),
            "td50": self.td50_spin.value(),
            "n": self.n_spin.value(),
            "m": self.m_spin.value(),
            "alpha_beta_normal": self.alpha_beta_normal_spin.value(),
            "fraction_size": self.fraction_size_spin.value(),
        }

        # Phát tín hiệu với tham số mới
        self.parametersChanged.emit(params)

    def load_organ_parameters(self, organ_name: str, is_target: bool):
        """
        Tải tham số đặc trưng cho cơ quan.

        Parameters
        ----------
        organ_name : str
            Tên cơ quan
        is_target : bool
            Có phải là mục tiêu (TARGET) không
        """
        if HAS_BIO_MODULE and self.use_organ_params_cb.isChecked():
            try:
                params = get_organ_specific_parameters(organ_name)

                if is_target:
                    self.tcd50_spin.setValue(params.get("tcd50", 50.0))
                    self.gamma50_tcp_spin.setValue(params.get("gamma50", 2.0))
                    self.alpha_beta_tumor_spin.setValue(params.get("alpha_beta", 10.0))
                else:
                    self.td50_spin.setValue(params.get("td50", 80.0))
                    self.n_spin.setValue(params.get("n", 0.1))
                    self.m_spin.setValue(params.get("m", 0.1))
                    self.alpha_beta_normal_spin.setValue(params.get("alpha_beta", 3.0))
            except Exception as e:
                logger.error(f"Lỗi khi tải tham số cho {organ_name}: {str(e)}")

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Lấy các tham số hiện tại.

        Returns
        -------
        Dict[str, Any]
            Từ điển các tham số hiện tại
        """
        return {
            "use_organ_params": self.use_organ_params_cb.isChecked(),
            "tcd50": self.tcd50_spin.value(),
            "gamma50": self.gamma50_tcp_spin.value(),
            "alpha_beta_tumor": self.alpha_beta_tumor_spin.value(),
            "td50": self.td50_spin.value(),
            "n": self.n_spin.value(),
            "m": self.m_spin.value(),
            "alpha_beta_normal": self.alpha_beta_normal_spin.value(),
            "fraction_size": self.fraction_size_spin.value(),
        }


class BiologicalMetricsWidget(QWidget):
    """Widget chính hiển thị các chỉ số sinh học."""

    structureSelectionChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        """Khởi tạo BiologicalMetricsWidget."""
        super().__init__(parent)
        self.dvh_data = {}
        self.structure_types = {}
        self.current_structure = None
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện widget."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Tiêu đề
        title_label = QLabel("Phân tích sinh học")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # Tạo splitter chính để chia màn hình
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(2)
        main_splitter.setChildrenCollapsible(False)

        # Panel trái: Bảng điều khiển và thông số
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Nhóm điều khiển cấu trúc
        structure_group = QGroupBox("Chọn cấu trúc")
        structure_layout = QVBoxLayout(structure_group)

        # Combo box cho phép chọn cấu trúc
        self.structure_combo = QComboBox()
        self.structure_combo.currentIndexChanged.connect(self._on_structure_changed)
        structure_layout.addWidget(self.structure_combo)

        # Thêm checkbox sử dụng tham số mặc định
        self.use_default_params = QCheckBox("Sử dụng tham số mặc định theo cơ quan")
        self.use_default_params.setChecked(True)
        self.use_default_params.stateChanged.connect(self._on_default_params_toggled)
        structure_layout.addWidget(self.use_default_params)

        left_layout.addWidget(structure_group)

        # Thêm widget tham số mô hình
        self.params_widget = ModelParametersWidget()
        self.params_widget.parametersChanged.connect(self._on_parameters_changed)
        left_layout.addWidget(self.params_widget)

        # Thêm nút cập nhật
        update_button = QPushButton("Cập nhật phân tích")
        update_button.clicked.connect(lambda: self._calculate_all_metrics())
        left_layout.addWidget(update_button)

        # Thêm khoảng trống co giãn ở dưới
        left_layout.addStretch(1)

        # Panel phải: Tab hiển thị kết quả
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget cho các loại hiển thị kết quả
        self.results_tabs = QTabWidget()

        # Tab bảng chỉ số
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        self.metrics_table = BiologicalMetricsTable()
        table_layout.addWidget(self.metrics_table)
        self.results_tabs.addTab(table_tab, "Bảng chỉ số")

        # Tab biểu đồ TCP/NTCP
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.tcp_ntcp_plot = TCPNTCPPlot()
        plot_layout.addWidget(self.tcp_ntcp_plot)
        self.results_tabs.addTab(plot_tab, "Biểu đồ TCP/NTCP")

        # Tab chi tiết cho cấu trúc hiện tại
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)

        # Nhóm thông tin chi tiết
        details_group = QGroupBox("Chi tiết chỉ số sinh học")
        details_form = QFormLayout(details_group)

        self.eud_label = QLabel("0.00 Gy")
        self.tcp_label = QLabel("0.00%")
        self.ntcp_label = QLabel("0.00%")
        self.geud_label = QLabel("0.00 Gy")
        self.bed_label = QLabel("0.00 Gy")

        details_form.addRow("EUD (Equivalent Uniform Dose):", self.eud_label)
        details_form.addRow("TCP (Tumor Control Probability):", self.tcp_label)
        details_form.addRow(
            "NTCP (Normal Tissue Complication Probability):", self.ntcp_label
        )
        details_form.addRow("gEUD (Generalized EUD):", self.geud_label)
        details_form.addRow("BED (Biologically Effective Dose):", self.bed_label)

        details_layout.addWidget(details_group)

        # Nhóm thông tin tham số
        params_group = QGroupBox("Tham số đang sử dụng")
        params_form = QFormLayout(params_group)

        self.a_param_label = QLabel("0.0")
        self.n_param_label = QLabel("0.0")
        self.m_param_label = QLabel("0.0")
        self.td50_label = QLabel("0.0 Gy")
        self.gamma50_label = QLabel("0.0")
        self.alpha_beta_label = QLabel("0.0 Gy")

        params_form.addRow("Tham số a:", self.a_param_label)
        params_form.addRow("Tham số n:", self.n_param_label)
        params_form.addRow("Tham số m:", self.m_param_label)
        params_form.addRow("TD50:", self.td50_label)
        params_form.addRow("γ50:", self.gamma50_label)
        params_form.addRow("α/β:", self.alpha_beta_label)

        details_layout.addWidget(params_group)

        # Thêm khoảng trống co giãn ở dưới
        details_layout.addStretch(1)

        self.results_tabs.addTab(details_tab, "Chi tiết")

        # Tab so sánh - nếu có nhiều kế hoạch
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)
        comparison_layout.addWidget(
            QLabel("Tính năng so sánh kế hoạch sẽ có trong phiên bản tới.")
        )
        self.results_tabs.addTab(comparison_tab, "So sánh")

        right_layout.addWidget(self.results_tabs)

        # Thêm các panel vào splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)

        # Thiết lập kích thước ban đầu cho các panel
        main_splitter.setSizes([int(self.width() * 0.3), int(self.width() * 0.7)])

        main_layout.addWidget(main_splitter)

        # Thêm panel thông tin trạng thái ở dưới cùng
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Sẵn sàng")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)

        # Thêm nhãn phiên bản phía bên phải
        version_label = QLabel("v2.0")
        version_label.setAlignment(Qt.AlignRight)
        status_layout.addWidget(version_label)

        main_layout.addLayout(status_layout)

    def set_dvh_data(
        self,
        dvh_data: Dict[str, Dict[str, List[float]]],
        structure_types: Dict[str, str],
    ):
        """
        Thiết lập dữ liệu DVH cho tính toán chỉ số sinh học.

        Parameters
        ----------
        dvh_data : Dict[str, Dict[str, List[float]]]
            Dữ liệu DVH, với format:
            {
                structure_name: {
                    'dose_bins': List[float],  # Liều (Gy)
                    'volume_bins': List[float],  # Thể tích (cm³)
                    'cum_dvh': List[float]  # DVH tích lũy (%)
                }
            }
        structure_types : Dict[str, str]
            Từ điển map tên cấu trúc với loại: 'TARGET' hoặc 'OAR'
        """
        if not dvh_data:
            self.clear_data()
            return

        self.dvh_data = dvh_data
        self.structure_types = structure_types

        # Cập nhật combobox cấu trúc
        self.structure_combo.blockSignals(True)
        self.structure_combo.clear()

        # Sắp xếp cấu trúc: targets trước, OARs sau
        targets = [
            (name, "TARGET")
            for name, type_ in structure_types.items()
            if type_.upper() == "TARGET"
        ]
        oars = [
            (name, "OAR")
            for name, type_ in structure_types.items()
            if type_.upper() != "TARGET"
        ]

        # Thêm vào combo box với phân nhóm rõ ràng
        if targets:
            self.structure_combo.addItem("--- MỤC TIÊU (TARGETS) ---", None)
            for name, _ in sorted(targets, key=lambda x: x[0]):
                self.structure_combo.addItem(name, name)

        if oars:
            self.structure_combo.addItem("--- CƠ QUAN NGUY CẤP (OARs) ---", None)
            for name, _ in sorted(oars, key=lambda x: x[0]):
                self.structure_combo.addItem(name, name)

        self.structure_combo.blockSignals(False)

        # Chọn cấu trúc đầu tiên (bỏ qua các mục nhóm)
        for i in range(self.structure_combo.count()):
            data = self.structure_combo.itemData(i)
            if data is not None:
                self.structure_combo.setCurrentIndex(i)
                break

        # Tính toán các chỉ số sinh học cho tất cả cấu trúc
        self._calculate_all_metrics()

    def _on_structure_changed(self, index):
        """Xử lý khi người dùng thay đổi lựa chọn cấu trúc."""
        # Bỏ qua nếu đây là một đề mục nhóm
        data = self.structure_combo.itemData(index)
        if data is None:
            # Tìm index đầu tiên có dữ liệu
            for i in range(self.structure_combo.count()):
                if self.structure_combo.itemData(i) is not None:
                    self.structure_combo.setCurrentIndex(i)
                    return
            return

        # Lưu cấu trúc hiện tại
        self.current_structure = data

        # Kiểm tra loại cấu trúc
        is_target = False
        if self.current_structure in self.structure_types:
            is_target = self.structure_types[self.current_structure].upper() == "TARGET"

        # Cập nhật tham số dựa trên cấu trúc nếu đang sử dụng tham số mặc định
        if self.use_default_params.isChecked():
            self.params_widget.load_organ_parameters(self.current_structure, is_target)

        # Cập nhật hiển thị chi tiết
        self._update_detail_display()

        # Phát tín hiệu thay đổi cấu trúc
        self.structureSelectionChanged.emit(self.current_structure)

    def _on_parameters_changed(self, params):
        """Xử lý khi người dùng thay đổi tham số mô hình."""
        # Cập nhật hiển thị chi tiết
        self._update_detail_display()

    def _on_default_params_toggled(self, state):
        """Xử lý khi người dùng bật/tắt sử dụng tham số mặc định."""
        use_default = state == Qt.Checked
        self.params_widget.setEnabled(not use_default)

        # Nếu bật tham số mặc định, cập nhật tham số theo cấu trúc hiện tại
        if use_default and self.current_structure:
            is_target = False
            if self.current_structure in self.structure_types:
                is_target = (
                    self.structure_types[self.current_structure].upper() == "TARGET"
                )
            self.params_widget.load_organ_parameters(self.current_structure, is_target)

    def _calculate_all_metrics(self, params=None):
        """
        Tính toán các chỉ số sinh học cho tất cả cấu trúc.

        Parameters
        ----------
        params : Dict, optional
            Tham số sinh học, nếu không cung cấp sẽ sử dụng tham số hiện tại
        """
        if not HAS_BIO_MODULE or not self.dvh_data:
            self.status_label.setText(
                "Không thể tính toán chỉ số sinh học. Module sinh học không khả dụng."
            )
            return

        self.status_label.setText("Đang tính toán chỉ số sinh học...")

        # Lấy tham số từ widget nếu không được cung cấp
        if params is None:
            params = self.params_widget.get_current_parameters()

        try:
            # Tính toán chỉ số sinh học cho từng cấu trúc
            all_metrics = {}

            for struct_name, dvh in self.dvh_data.items():
                # Xác định loại cấu trúc
                is_target = (
                    struct_name in self.structure_types
                    and self.structure_types[struct_name].upper() == "TARGET"
                )

                # Lấy tham số đặc thù cho cơ quan nếu sử dụng tham số mặc định
                struct_params = params
                if self.use_default_params.isChecked():
                    struct_params = get_organ_specific_parameters(
                        struct_name, is_target
                    )

                # Lấy dữ liệu liều và thể tích
                dose_bins = dvh.get("dose_bins", [])
                cum_dvh = dvh.get("cum_dvh", [])

                if not dose_bins or not cum_dvh or len(dose_bins) != len(cum_dvh):
                    logger.warning(
                        f"Dữ liệu DVH không hợp lệ cho cấu trúc {struct_name}"
                    )
                    continue

                # Tính toán các chỉ số sinh học
                metrics = calculate_biological_metrics(
                    dose_bins, cum_dvh, struct_params, is_target=is_target
                )

                # Thêm thông tin loại cấu trúc
                metrics["type"] = "TARGET" if is_target else "OAR"

                # Thêm vào từ điển kết quả
                all_metrics[struct_name] = metrics

            # Cập nhật bảng chỉ số
            self.metrics_table.update_metrics(all_metrics)

            # Cập nhật biểu đồ TCP/NTCP
            self.tcp_ntcp_plot.plot_data(all_metrics)

            # Cập nhật hiển thị chi tiết cho cấu trúc hiện tại
            self._update_detail_display(all_metrics)

            self.status_label.setText(
                f"Hoàn tất tính toán cho {len(all_metrics)} cấu trúc"
            )

        except Exception as e:
            logger.error(f"Lỗi khi tính toán chỉ số sinh học: {str(e)}")
            self.status_label.setText(f"Lỗi: {str(e)}")

    def _update_detail_display(self, all_metrics=None):
        """
        Cập nhật hiển thị chi tiết cho cấu trúc hiện tại.

        Parameters
        ----------
        all_metrics : Dict[str, Dict[str, Any]], optional
            Từ điển chứa các chỉ số sinh học đã tính, nếu không cung cấp thì sẽ tính lại
        """
        if not self.current_structure:
            return

        # Nếu không có metrics sẵn, thực hiện tính toán
        if all_metrics is None:
            params = self.params_widget.get_current_parameters()
            struct_dvh = self.dvh_data.get(self.current_structure, {})

            # Xác định loại cấu trúc
            is_target = (
                self.current_structure in self.structure_types
                and self.structure_types[self.current_structure].upper() == "TARGET"
            )

            # Lấy tham số đặc thù cho cơ quan nếu sử dụng tham số mặc định
            if self.use_default_params.isChecked():
                params = get_organ_specific_parameters(
                    self.current_structure, is_target
                )

            # Lấy dữ liệu liều và thể tích
            dose_bins = struct_dvh.get("dose_bins", [])
            cum_dvh = struct_dvh.get("cum_dvh", [])

            if not dose_bins or not cum_dvh or len(dose_bins) != len(cum_dvh):
                return

            # Tính toán các chỉ số sinh học
            metrics = calculate_biological_metrics(
                dose_bins, cum_dvh, params, is_target=is_target
            )
        else:
            # Sử dụng metrics đã có
            metrics = all_metrics.get(self.current_structure, {})

        if not metrics:
            return

        # Cập nhật nhãn chi tiết
        self.eud_label.setText(f"{metrics.get('EUD', 0.0):.2f} Gy")
        self.tcp_label.setText(f"{metrics.get('TCP', 0.0):.2f}%")
        self.ntcp_label.setText(f"{metrics.get('NTCP', 0.0):.2f}%")
        self.geud_label.setText(f"{metrics.get('gEUD', 0.0):.2f} Gy")
        self.bed_label.setText(f"{metrics.get('BED', 0.0):.2f} Gy")

        # Thêm màu sắc cho TCP/NTCP dựa trên giá trị
        tcp = metrics.get("TCP", 0.0)
        ntcp = metrics.get("NTCP", 0.0)

        # Màu sắc cho TCP (cao là tốt)
        if tcp >= 95.0:
            self.tcp_label.setStyleSheet("color: green; font-weight: bold;")
        elif tcp >= 80.0:
            self.tcp_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.tcp_label.setStyleSheet("color: red; font-weight: bold;")

        # Màu sắc cho NTCP (thấp là tốt)
        if ntcp <= 5.0:
            self.ntcp_label.setStyleSheet("color: green; font-weight: bold;")
        elif ntcp <= 15.0:
            self.ntcp_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.ntcp_label.setStyleSheet("color: red; font-weight: bold;")

        # Cập nhật nhãn tham số
        params = self.params_widget.get_current_parameters()
        self.a_param_label.setText(f"{params.get('a', 0.0)}")
        self.n_param_label.setText(f"{params.get('n', 0.0)}")
        self.m_param_label.setText(f"{params.get('m', 0.0)}")
        self.td50_label.setText(f"{params.get('td50', 0.0):.2f} Gy")
        self.gamma50_label.setText(f"{params.get('gamma50', 0.0):.2f}")
        self.alpha_beta_label.setText(f"{params.get('alpha_beta', 0.0):.2f} Gy")

    def clear_data(self):
        """Xóa tất cả dữ liệu hiển thị."""
        self.dvh_data = {}
        self.structure_types = {}
        self.current_structure = None

        # Xóa combobox cấu trúc
        self.structure_combo.clear()

        # Xóa bảng metrics
        self.metrics_table.clear_data()

        # Xóa biểu đồ
        if HAS_MPL:
            self.tcp_ntcp_plot.figure.clear()
            if hasattr(self.tcp_ntcp_plot, "canvas"):
                self.tcp_ntcp_plot.canvas.draw()

        # Đặt lại nhãn chi tiết
        self.eud_label.setText("0.00 Gy")
        self.tcp_label.setText("0.00%")
        self.ntcp_label.setText("0.00%")
        self.geud_label.setText("0.00 Gy")
        self.bed_label.setText("0.00 Gy")

        self.a_param_label.setText("0.0")
        self.n_param_label.setText("0.0")
        self.m_param_label.setText("0.0")
        self.td50_label.setText("0.0 Gy")
        self.gamma50_label.setText("0.0")
        self.alpha_beta_label.setText("0.0 Gy")

        self.tcp_label.setStyleSheet("")
        self.ntcp_label.setStyleSheet("")

        self.status_label.setText("Không có dữ liệu")


# Hàm tiện ích để tạo widget
def create_biological_metrics_widget(parent=None) -> Optional[BiologicalMetricsWidget]:
    """
    Tạo và trả về widget chỉ số sinh học.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha

    Returns
    -------
    Optional[BiologicalMetricsWidget]
        Widget chỉ số sinh học, hoặc None nếu có lỗi
    """
    try:
        return BiologicalMetricsWidget(parent)
    except Exception as e:
        logger.error(f"Lỗi khi tạo BiologicalMetricsWidget: {str(e)}")
        return None
