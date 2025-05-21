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


class RadarChart(QWidget):
    """Widget hiển thị biểu đồ radar (còn gọi là lưới nhện) để trực quan hóa nhiều chỉ số sinh học cùng lúc."""

    def __init__(self, parent=None):
        if not HAS_MPL:
            logging.error("Matplotlib không khả dụng. RadarChart không hoạt động.")
            return

        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện biểu đồ radar."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.canvas)

        # Thiết lập giá trị mặc định
        self.data = {}
        self.metrics = []
        self.structure_types = {}

    def plot_radar(
        self, metrics_data: Dict[str, Dict[str, Any]], structure_types: Dict[str, str]
    ):
        """
        Vẽ biểu đồ radar từ dữ liệu chỉ số sinh học.

        Parameters
        ----------
        metrics_data : Dict[str, Dict[str, Any]]
            Từ điển chỉ số sinh học cho mỗi cấu trúc
        structure_types : Dict[str, str]
            Từ điển loại cấu trúc (TARGET/OAR)
        """
        self.figure.clear()
        self.data = metrics_data
        self.structure_types = structure_types

        if not metrics_data:
            self.canvas.draw()
            return

        # Lọc các chỉ số cần hiển thị
        display_metrics = ["TCP", "NTCP_inv", "EUD_norm", "CI", "HI"]
        metric_labels = [
            "TCP",
            "Độ an toàn",
            "Liều phân phối",
            "Chỉ số bao phủ",
            "Độ đồng nhất",
        ]

        # Thiết lập dữ liệu radar
        targets = []
        target_values = []
        oars = []
        oar_values = []

        for name, metrics in metrics_data.items():
            # Chuẩn hóa và chuyển đổi dữ liệu
            radar_data = []

            # TCP - đã ở dạng %
            tcp = metrics.get("TCP", 0.0)

            # NTCP đảo ngược (100 - NTCP) để giá trị cao = tốt
            ntcp = metrics.get("NTCP", 0.0)
            ntcp_inv = max(0, 100 - ntcp)

            # Chuẩn hóa EUD (0-100%)
            eud = metrics.get("EUD", 0.0)
            eud_norm = min(100, eud / 80.0 * 100) if eud > 0 else 0

            # Thêm các chỉ số khác nếu có
            ci = metrics.get("CI", 85.0)  # Conformity Index
            hi = metrics.get("HI", 85.0)  # Homogeneity Index

            # Tạo mảng dữ liệu radar
            values = [tcp, ntcp_inv, eud_norm, ci, hi]

            # Phân loại dữ liệu theo loại cấu trúc
            structure_type = structure_types.get(name, "OAR")
            if (
                "TARGET" in structure_type
                or "PTV" in structure_type
                or "CTV" in structure_type
            ):
                targets.append(name)
                target_values.append(values)
            else:
                oars.append(name)
                oar_values.append(values)

        # Vẽ biểu đồ radar
        ax = self.figure.add_subplot(111, polar=True)

        # Thiết lập góc cho các trục (categories)
        angles = np.linspace(
            0, 2 * np.pi, len(display_metrics), endpoint=False
        ).tolist()
        angles += angles[:1]  # Đóng vòng tròn

        # Thiết lập nhãn cho các trục
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)

        # Thiết lập các vòng giá trị
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"])
        ax.set_ylim(0, 100)

        # Vẽ các TARGET
        for i, (name, values) in enumerate(zip(targets, target_values)):
            values += values[:1]  # Đóng vòng tròn
            ax.plot(
                angles, values, "o-", linewidth=2, label=f"{name} (TARGET)", alpha=0.8
            )
            ax.fill(angles, values, alpha=0.1)

        # Vẽ các OAR
        for i, (name, values) in enumerate(zip(oars, oar_values)):
            values += values[:1]  # Đóng vòng tròn
            ax.plot(angles, values, "o-", linewidth=1, label=f"{name} (OAR)", alpha=0.6)

        # Thêm legend
        if targets or oars:
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))

        # Vẽ các vòng tròn đồng tâm cho thang đánh giá
        self._draw_background_circles(ax)

        self.canvas.draw()

    def _draw_background_circles(self, ax):
        """Vẽ các vòng tròn đồng tâm với mã màu đánh giá."""
        # Các mức đánh giá và màu tương ứng
        evaluation_ranges = [
            (0, 20, "#ffcccc"),  # Rất kém - đỏ nhạt
            (20, 40, "#ffeecc"),  # Kém - cam nhạt
            (40, 60, "#ffffcc"),  # Trung bình - vàng nhạt
            (60, 80, "#ccffcc"),  # Tốt - xanh lá nhạt
            (80, 100, "#ccffee"),  # Rất tốt - xanh lục nhạt
        ]

        # Vẽ các vòng tròn fill màu
        for min_val, max_val, color in evaluation_ranges:
            # Tạo mảng dữ liệu để vẽ vòng tròn fill
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.fill_between(theta, min_val, max_val, color=color, alpha=0.2)

        # Vẽ các đường vòng tròn
        for radius in [20, 40, 60, 80, 100]:
            ax.plot(np.linspace(0, 2 * np.pi, 100), [radius] * 100, "k-", alpha=0.1)


class SensitivityAnalysisTab(QWidget):
    """Tab hiển thị phân tích độ nhạy của các tham số mô hình sinh học."""

    parameterChanged = pyqtSignal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện tab phân tích độ nhạy."""
        layout = QVBoxLayout(self)

        # Panel điều khiển
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # Chọn cấu trúc
        structure_group = QGroupBox("Chọn cấu trúc")
        structure_layout = QVBoxLayout(structure_group)
        self.structure_combo = QComboBox()
        structure_layout.addWidget(self.structure_combo)

        # Chọn tham số để phân tích
        param_group = QGroupBox("Chọn tham số phân tích")
        param_layout = QVBoxLayout(param_group)
        self.parameter_combo = QComboBox()
        self.parameter_combo.addItems(
            [
                "a (TCP)",
                "TCD50 (TCP)",
                "gamma50 (TCP)",
                "n (NTCP)",
                "m (NTCP)",
                "TD50 (NTCP)",
                "a/b (BED)",
            ]
        )
        param_layout.addWidget(self.parameter_combo)

        # Thanh trượt điều chỉnh tham số
        slider_group = QGroupBox("Điều chỉnh giá trị")
        slider_layout = QFormLayout(slider_group)
        self.value_slider = QDoubleSpinBox()
        self.value_slider.setRange(0.1, 100.0)
        self.value_slider.setSingleStep(0.1)
        self.value_slider.setValue(10.0)
        slider_layout.addRow("Giá trị:", self.value_slider)

        # Nút phân tích
        analyze_button = QPushButton("Phân tích")
        analyze_button.clicked.connect(self._on_analyze_clicked)

        control_layout.addWidget(structure_group, 1)
        control_layout.addWidget(param_group, 1)
        control_layout.addWidget(slider_group, 1)
        control_layout.addWidget(analyze_button)

        # Khu vực biểu đồ
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        # Biểu đồ chính
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        # Kết quả tính toán
        result_panel = QGroupBox("Kết quả phân tích độ nhạy")
        result_layout = QVBoxLayout(result_panel)

        self.result_text = QLabel("Chọn cấu trúc và tham số để phân tích độ nhạy")
        self.result_text.setWordWrap(True)
        result_layout.addWidget(self.result_text)

        # Thêm các panel vào layout chính
        layout.addWidget(control_panel)
        layout.addWidget(plot_panel, 3)
        layout.addWidget(result_panel)

        # Kết nối tín hiệu
        self.structure_combo.currentTextChanged.connect(self._on_structure_changed)
        self.parameter_combo.currentTextChanged.connect(self._on_parameter_changed)
        self.value_slider.valueChanged.connect(self._on_value_changed)

        # Biến lưu trữ dữ liệu
        self.dvh_data = {}
        self.structure_types = {}
        self.current_parameters = {}
        self.default_parameters = {}
        self.sensitivity_results = {}

    def set_dvh_data(
        self,
        dvh_data: Dict[str, Dict[str, List[float]]],
        structure_types: Dict[str, str],
    ):
        """
        Đặt dữ liệu DVH cho phân tích.

        Parameters
        ----------
        dvh_data : Dict[str, Dict[str, List[float]]]
            Dữ liệu DVH cho mỗi cấu trúc
        structure_types : Dict[str, str]
            Loại của mỗi cấu trúc (TARGET/OAR)
        """
        self.dvh_data = dvh_data
        self.structure_types = structure_types

        # Cập nhật combobox cấu trúc
        self.structure_combo.clear()
        if dvh_data:
            self.structure_combo.addItems(sorted(dvh_data.keys()))

    def _on_structure_changed(self, structure_name):
        """Xử lý khi lựa chọn cấu trúc thay đổi."""
        if not structure_name or not HAS_BIO_MODULE:
            return

        # Xác định loại cấu trúc (TARGET/OAR)
        structure_type = self.structure_types.get(structure_name, "OAR")
        is_target = (
            "TARGET" in structure_type
            or "PTV" in structure_type
            or "CTV" in structure_type
        )

        # Tải tham số cho cơ quan
        self.default_parameters = get_organ_specific_parameters(
            structure_name, is_target
        )
        self.current_parameters = self.default_parameters.copy()

        # Cập nhật giá trị tham số hiện tại
        param_key = self._get_parameter_key()
        if param_key in self.default_parameters:
            self.value_slider.setValue(self.default_parameters[param_key])

    def _on_parameter_changed(self, parameter_name):
        """Xử lý khi lựa chọn tham số phân tích thay đổi."""
        param_key = self._get_parameter_key()
        if param_key in self.default_parameters:
            self.value_slider.setValue(self.default_parameters[param_key])

    def _on_value_changed(self, value):
        """Xử lý khi giá trị tham số thay đổi."""
        param_key = self._get_parameter_key()
        if param_key:
            self.current_parameters[param_key] = value

    def _get_parameter_key(self) -> str:
        """Lấy khóa tham số từ lựa chọn trong combobox."""
        param_text = self.parameter_combo.currentText()

        # Mapping từ text hiển thị sang khóa tham số
        mapping = {
            "a (TCP)": "a",
            "TCD50 (TCP)": "tcd50",
            "gamma50 (TCP)": "gamma50",
            "n (NTCP)": "n",
            "m (NTCP)": "m",
            "TD50 (NTCP)": "td50",
            "a/b (BED)": "alpha_beta",
        }

        return mapping.get(param_text, "")

    def _on_analyze_clicked(self):
        """Thực hiện phân tích độ nhạy."""
        if not HAS_BIO_MODULE:
            self.result_text.setText("Module phân tích sinh học không khả dụng.")
            return

        structure_name = self.structure_combo.currentText()
        if not structure_name or structure_name not in self.dvh_data:
            self.result_text.setText("Vui lòng chọn cấu trúc hợp lệ.")
            return

        param_key = self._get_parameter_key()
        if not param_key:
            self.result_text.setText("Vui lòng chọn tham số cần phân tích.")
            return

        # Lấy dữ liệu DVH của cấu trúc
        dvh = self.dvh_data.get(structure_name, {})
        if not dvh:
            self.result_text.setText(
                f"Không có dữ liệu DVH cho cấu trúc {structure_name}."
            )
            return

        # Thực hiện phân tích độ nhạy
        self._perform_sensitivity_analysis(structure_name, param_key)

    def _perform_sensitivity_analysis(self, structure_name, param_key):
        """
        Thực hiện phân tích độ nhạy cho tham số được chọn.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        param_key : str
            Khóa tham số cần phân tích
        """
        # Lấy loại cấu trúc
        structure_type = self.structure_types.get(structure_name, "OAR")
        is_target = (
            "TARGET" in structure_type
            or "PTV" in structure_type
            or "CTV" in structure_type
        )

        # Lấy dữ liệu DVH
        dvh_data = self.dvh_data[structure_name]
        doses = np.array(dvh_data.get("doses", []))
        volumes = np.array(dvh_data.get("volumes", []))

        if len(doses) == 0 or len(volumes) == 0:
            self.result_text.setText(f"Dữ liệu DVH không hợp lệ cho {structure_name}.")
            return

        # Tạo dải giá trị tham số để phân tích
        base_value = self.default_parameters.get(param_key, 1.0)
        variation = 0.5  # +/- 50%

        param_values = np.linspace(
            base_value * (1 - variation), base_value * (1 + variation), 20
        )

        # Tính toán các giá trị chỉ số với mỗi giá trị tham số
        results = []
        labels = []

        # Xác định chỉ số cần phân tích dựa trên loại tham số
        if param_key in ["a", "tcd50", "gamma50"]:
            metric = "TCP"
        elif param_key in ["n", "m", "td50"]:
            metric = "NTCP"
        else:
            metric = "BED"

        for value in param_values:
            # Tạo bộ tham số mới với giá trị thay đổi
            params = self.default_parameters.copy()
            params[param_key] = value

            # Tính toán chỉ số sinh học
            metrics = calculate_biological_metrics(
                doses, volumes, structure_name, is_target, params
            )

            # Lưu kết quả
            results.append(metrics.get(metric, 0.0))
            labels.append(f"{value:.2f}")

        # Vẽ biểu đồ
        self._plot_sensitivity_results(param_key, param_values, results, metric)

        # Hiển thị kết quả
        base_result = results[len(results) // 2]  # Giá trị cơ sở ở giữa
        min_result = min(results)
        max_result = max(results)

        result_text = (
            f"<b>Phân tích độ nhạy cho {structure_name}</b><br>"
            f"Tham số: <b>{param_key}</b> (Giá trị cơ sở: {base_value:.2f})<br>"
            f"Chỉ số phân tích: <b>{metric}</b><br>"
            f"Giá trị cơ sở: {base_result:.2f}<br>"
            f"Phạm vi biến thiên: {min_result:.2f} - {max_result:.2f}<br>"
            f"Độ biến thiên tương đối: {(max_result - min_result) / base_result * 100:.1f}%<br>"
            f"Kết luận: Độ nhạy {'<span style="color:red;">CAO</span>' if (max_result - min_result) / base_result > 0.2 else '<span style="color:green;">THẤP</span>'}"
        )

        self.result_text.setText(result_text)

    def _plot_sensitivity_results(self, param_key, param_values, results, metric):
        """
        Vẽ biểu đồ kết quả phân tích độ nhạy.

        Parameters
        ----------
        param_key : str
            Khóa tham số
        param_values : List[float]
            Các giá trị tham số
        results : List[float]
            Các giá trị chỉ số tính được
        metric : str
            Tên chỉ số đang phân tích
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Vẽ biểu đồ đường
        ax.plot(param_values, results, "o-", linewidth=2)

        # Đánh dấu giá trị cơ sở
        base_idx = len(param_values) // 2
        ax.plot([param_values[base_idx]], [results[base_idx]], "ro", markersize=8)

        # Thiết lập nhãn trục
        ax.set_xlabel(f"Giá trị tham số ({param_key})")
        ax.set_ylabel(f"{metric}")
        ax.set_title(f"Phân tích độ nhạy: Ảnh hưởng của {param_key} lên {metric}")

        # Thêm lưới
        ax.grid(True, linestyle="--", alpha=0.7)

        self.canvas.draw()


class BiologicalMetricsWidget(QWidget):
    """Widget chính hiển thị các chỉ số sinh học."""

    structureSelectionChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        if not HAS_QT:
            return
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)

        # TabWidget chính
        self.tab_widget = QTabWidget()

        # Tab 1: Chỉ số chính
        self.metrics_tab = QWidget()
        self._setup_metrics_tab()
        self.tab_widget.addTab(self.metrics_tab, "Chỉ số chính")

        # Tab 2: Biểu đồ
        self.chart_tab = QWidget()
        self._setup_chart_tab()
        self.tab_widget.addTab(self.chart_tab, "Biểu đồ")

        # Tab 3: Biểu đồ radar
        self.radar_tab = QWidget()
        self._setup_radar_tab()
        self.tab_widget.addTab(self.radar_tab, "Radar")

        # Tab 4: Phân tích độ nhạy
        self.sensitivity_tab = SensitivityAnalysisTab()
        self.tab_widget.addTab(self.sensitivity_tab, "Độ nhạy")

        # Tab 5: Chi tiết
        self.detail_tab = QWidget()
        self._setup_detail_tab()
        self.tab_widget.addTab(self.detail_tab, "Chi tiết")

        # Thêm TabWidget vào layout chính
        main_layout.addWidget(self.tab_widget)

    def _setup_metrics_tab(self):
        """Thiết lập tab hiển thị các chỉ số sinh học chính."""
        layout = QVBoxLayout(self.metrics_tab)

        # Tạo splitter
        splitter = QSplitter(Qt.Vertical)

        # Widget chọn cấu trúc và các tham số
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)

        # Phần bên trái: Chọn cấu trúc
        left_panel = QGroupBox("Cấu trúc")
        left_layout = QVBoxLayout(left_panel)

        self.structure_combo = QComboBox()
        self.structure_combo.currentIndexChanged.connect(self._on_structure_changed)
        left_layout.addWidget(QLabel("Chọn cấu trúc:"))
        left_layout.addWidget(self.structure_combo)

        # Phần bên phải: Tham số mô hình
        right_panel = QGroupBox("Tham số mô hình")
        right_layout = QVBoxLayout(right_panel)

        self.use_default_params = QCheckBox("Sử dụng tham số mặc định")
        self.use_default_params.setChecked(True)
        self.use_default_params.stateChanged.connect(self._on_default_params_toggled)
        right_layout.addWidget(self.use_default_params)

        # Thêm panel tham số mô hình
        self.parameter_widget = ModelParametersWidget()
        self.parameter_widget.setEnabled(False)  # Mặc định sử dụng tham số mặc định
        self.parameter_widget.parametersChanged.connect(self._on_parameters_changed)
        right_layout.addWidget(self.parameter_widget)

        # Thêm các panel vào layout
        top_layout.addWidget(left_panel)
        top_layout.addWidget(right_panel)

        # Widget hiển thị kết quả
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)

        self.metrics_table = BiologicalMetricsTable()
        bottom_layout.addWidget(self.metrics_table)

        # Thêm widgets vào splitter
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        splitter.setSizes([200, 400])  # Thiết lập kích thước ban đầu

        layout.addWidget(splitter)

    def _setup_chart_tab(self):
        """Thiết lập tab hiển thị các biểu đồ."""
        layout = QVBoxLayout(self.chart_tab)

        # Tạo widget hiển thị đồ thị TCP/NTCP
        self.tcp_ntcp_plot = TCPNTCPPlot()
        layout.addWidget(self.tcp_ntcp_plot)

    def _setup_radar_tab(self):
        """Thiết lập tab hiển thị biểu đồ radar."""
        layout = QVBoxLayout(self.radar_tab)

        # Tạo biểu đồ radar
        self.radar_chart = RadarChart()
        layout.addWidget(self.radar_chart)

        # Thêm ghi chú giải thích
        note_label = QLabel(
            "<b>Biểu đồ radar</b> hiển thị trực quan đánh giá toàn diện kế hoạch xạ trị dựa trên "
            "nhiều chỉ số sinh học và vật lý. Các vòng tròn đồng tâm biểu thị mức đánh giá từ kém "
            "(bên trong) đến xuất sắc (bên ngoài)."
        )
        note_label.setWordWrap(True)
        layout.addWidget(note_label)

    def _setup_detail_tab(self):
        """Thiết lập tab hiển thị thông tin chi tiết."""
        layout = QVBoxLayout(self.detail_tab)

        # Tạo widget hiển thị chi tiết
        self.detail_display = QLabel("Chọn một cấu trúc để xem thông tin chi tiết.")
        self.detail_display.setWordWrap(True)
        self.detail_display.setTextFormat(Qt.RichText)

        # Đặt trong ScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.detail_display)

        layout.addWidget(scroll_area)

    def refresh_all_views(self):
        """Làm mới tất cả các view."""
        # Tính toán lại tất cả các chỉ số
        all_metrics = self._calculate_all_metrics()

        # Cập nhật bảng
        self.metrics_table.update_metrics(all_metrics)

        # Cập nhật biểu đồ
        self.tcp_ntcp_plot.plot_data(all_metrics)

        # Cập nhật biểu đồ radar
        self.radar_chart.plot_radar(all_metrics, self.structure_types)

        # Cập nhật hiển thị chi tiết
        self._update_detail_display(all_metrics)

    def _calculate_all_metrics(self):
        """Tính toán các chỉ số sinh học cho tất cả cấu trúc."""
        if not HAS_BIO_MODULE or not self.dvh_data:
            return {}

        all_metrics = {}
        for struct_name, dvh in self.dvh_data.items():
            # Xác định loại cấu trúc
            is_target = (
                struct_name in self.structure_types
                and self.structure_types[struct_name].upper() == "TARGET"
            )

            # Lấy tham số đặc thù cho cơ quan nếu sử dụng tham số mặc định
            struct_params = self.parameter_widget.get_current_parameters()
            if self.use_default_params.isChecked():
                struct_params = get_organ_specific_parameters(struct_name, is_target)

            # Lấy dữ liệu liều và thể tích
            dose_bins = dvh.get("dose_bins", [])
            cum_dvh = dvh.get("cum_dvh", [])

            if not dose_bins or not cum_dvh or len(dose_bins) != len(cum_dvh):
                logger.warning(f"Dữ liệu DVH không hợp lệ cho cấu trúc {struct_name}")
                continue

            # Tính toán các chỉ số sinh học
            metrics = calculate_biological_metrics(
                dose_bins, cum_dvh, struct_params, is_target=is_target
            )

            # Thêm thông tin loại cấu trúc
            metrics["type"] = "TARGET" if is_target else "OAR"

            # Thêm vào từ điển kết quả
            all_metrics[struct_name] = metrics

        return all_metrics

    def _update_detail_display(self, all_metrics):
        """Cập nhật hiển thị chi tiết cho cấu trúc hiện tại."""
        if not self.structure_combo.currentText():
            return

        metrics = all_metrics.get(self.structure_combo.currentText(), {})
        if not metrics:
            return

        self.detail_display.setText(
            f"EUD: {metrics.get('EUD', 0.0):.2f} Gy<br>"
            f"TCP: {metrics.get('TCP', 0.0):.2f}%<br>"
            f"NTCP: {metrics.get('NTCP', 0.0):.2f}%<br>"
            f"gEUD: {metrics.get('gEUD', 0.0):.2f} Gy<br>"
            f"BED: {metrics.get('BED', 0.0):.2f} Gy"
        )

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
            self.parameter_widget.load_organ_parameters(
                self.current_structure, is_target
            )

        # Cập nhật hiển thị chi tiết
        self._update_detail_display(self._calculate_all_metrics())

        # Phát tín hiệu thay đổi cấu trúc
        self.structureSelectionChanged.emit(self.current_structure)

    def _on_parameters_changed(self, params):
        """Xử lý khi người dùng thay đổi tham số mô hình."""
        # Cập nhật hiển thị chi tiết
        self._update_detail_display(self._calculate_all_metrics())

    def _on_default_params_toggled(self, state):
        """Xử lý khi người dùng bật/tắt sử dụng tham số mặc định."""
        use_default = state == Qt.Checked
        self.parameter_widget.setEnabled(not use_default)

        # Nếu bật tham số mặc định, cập nhật tham số theo cấu trúc hiện tại
        if use_default and self.current_structure:
            is_target = False
            if self.current_structure in self.structure_types:
                is_target = (
                    self.structure_types[self.current_structure].upper() == "TARGET"
                )
            self.parameter_widget.load_organ_parameters(
                self.current_structure, is_target
            )


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
