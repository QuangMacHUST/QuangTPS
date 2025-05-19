#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Widget hiển thị kết quả phân tích độ bền vững trong QuangTPS.

Module này cung cấp một widget độc lập để hiển thị kết quả
phân tích độ bền vững của kế hoạch, có thể được tái sử dụng
trong các tab hoặc cửa sổ khác nhau.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple

try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QCheckBox,
        QTabWidget,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
    )
    from PyQt5.QtGui import QColor
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_PYQT = True
except ImportError:
    # Fallback for when PyQt is not available
    from PySide2.QtCore import Qt, Signal as pyqtSignal
    from PySide2.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QCheckBox,
        QTabWidget,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
    )
    from PySide2.QtGui import QColor
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_PYQT = True
except Exception:
    HAS_PYQT = False
    import warnings

    warnings.warn(
        "Không thể import PyQt hoặc PySide, RobustAnalysisWidget sẽ là placeholder"
    )

logger = logging.getLogger(__name__)


class RobustAnalysisWidget(QWidget):
    """
    Widget hiển thị kết quả phân tích độ bền vững.

    Widget này hiển thị DVH bands, biểu đồ độ phủ mục tiêu, bảng chỉ số đánh giá
    và phân tích không gian cho kết quả phân tích độ bền vững, với giao diện
    phong cách Eclipse.
    """

    # Signals
    exportRequested = pyqtSignal(object)  # Phát khi người dùng yêu cầu xuất kết quả

    def __init__(self, parent=None):
        """
        Khởi tạo widget hiển thị kết quả phân tích độ bền vững.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, by default None
        """
        super().__init__(parent)

        # Dữ liệu nội bộ
        self.robustness_result = None

        # Thiết lập UI
        if HAS_PYQT:
            self._setup_ui()
        else:
            # Placeholder khi không có PyQt
            layout = QVBoxLayout(self)
            label = QLabel("PyQt hoặc PySide là bắt buộc cho RobustAnalysisWidget")
            layout.addWidget(label)

    def _setup_ui(self):
        """Thiết lập giao diện người dùng với phong cách Eclipse."""
        main_layout = QVBoxLayout(self)

        # Tiêu đề
        title_layout = QHBoxLayout()
        title_label = QLabel("Kết quả phân tích độ bền vững")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #003366;")
        title_layout.addWidget(title_label)

        # Thêm combobox lựa chọn
        title_layout.addStretch()
        title_layout.addWidget(QLabel("Hiển thị:"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["DVH Bands", "Coverage", "Metrics", "All"])
        self.display_mode_combo.currentIndexChanged.connect(self._update_display_mode)
        title_layout.addWidget(self.display_mode_combo)

        main_layout.addLayout(title_layout)

        # Tab widget
        self.tab_widget = QTabWidget()

        # DVH Tab
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)

        # Lựa chọn cấu trúc
        dvh_controls_layout = QHBoxLayout()
        dvh_controls_layout.addWidget(QLabel("Cấu trúc:"))
        self.structure_combo = QComboBox()
        self.structure_combo.currentIndexChanged.connect(self._update_dvh_plot)
        dvh_controls_layout.addWidget(self.structure_combo)

        dvh_controls_layout.addWidget(QLabel("Hiển thị:"))
        self.show_nominal_check = QCheckBox("Nominal")
        self.show_nominal_check.setChecked(True)
        self.show_nominal_check.stateChanged.connect(self._update_dvh_plot)
        dvh_controls_layout.addWidget(self.show_nominal_check)

        self.show_worst_check = QCheckBox("Worst Case")
        self.show_worst_check.setChecked(True)
        self.show_worst_check.stateChanged.connect(self._update_dvh_plot)
        dvh_controls_layout.addWidget(self.show_worst_check)

        self.show_all_check = QCheckBox("All Scenarios")
        self.show_all_check.stateChanged.connect(self._update_dvh_plot)
        dvh_controls_layout.addWidget(self.show_all_check)

        dvh_layout.addLayout(dvh_controls_layout)

        # DVH Plot
        dvh_figure = Figure(figsize=(8, 6), dpi=100)
        self.dvh_canvas = FigureCanvas(dvh_figure)
        dvh_layout.addWidget(self.dvh_canvas)

        self.tab_widget.addTab(dvh_tab, "DVH Bands")

        # Coverage Tab
        coverage_tab = QWidget()
        coverage_layout = QVBoxLayout(coverage_tab)

        # Lựa chọn chỉ số
        coverage_controls_layout = QHBoxLayout()
        coverage_controls_layout.addWidget(QLabel("Chỉ số:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["D95", "D98", "V95", "Homogeneity Index"])
        self.metric_combo.currentIndexChanged.connect(self._update_coverage_plot)
        coverage_controls_layout.addWidget(self.metric_combo)

        coverage_layout.addLayout(coverage_controls_layout)

        # Coverage Plot
        coverage_figure = Figure(figsize=(8, 6), dpi=100)
        self.coverage_canvas = FigureCanvas(coverage_figure)
        coverage_layout.addWidget(self.coverage_canvas)

        self.tab_widget.addTab(coverage_tab, "Target Coverage")

        # Metrics Tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)

        # Bảng chỉ số
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Cấu trúc", "Chỉ số", "Nominal", "Min", "Max", "Range"]
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        metrics_layout.addWidget(self.metrics_table)

        self.tab_widget.addTab(metrics_tab, "Metrics")

        # Thêm tab widget vào layout chính
        main_layout.addWidget(self.tab_widget)

        # Các nút điều khiển
        buttons_layout = QHBoxLayout()

        self.export_button = QPushButton("Xuất kết quả")
        self.export_button.clicked.connect(self._request_export)
        buttons_layout.addWidget(self.export_button)

        buttons_layout.addStretch()

        main_layout.addLayout(buttons_layout)

        # Khởi tạo plots
        self._init_plots()

    def _init_plots(self):
        """Khởi tạo các biểu đồ trống."""
        # DVH plot
        self.dvh_ax = self.dvh_canvas.figure.add_subplot(111)
        self.dvh_ax.set_xlabel("Liều (Gy)")
        self.dvh_ax.set_ylabel("Thể tích (%)")
        self.dvh_ax.set_title("DVH Robustness Bands")
        self.dvh_ax.set_xlim([0, 80])
        self.dvh_ax.set_ylim([0, 105])
        self.dvh_ax.grid(True)
        self.dvh_canvas.figure.tight_layout()
        self.dvh_canvas.draw()

        # Coverage plot
        self.coverage_ax = self.coverage_canvas.figure.add_subplot(111)
        self.coverage_ax.set_xlabel("Cấu trúc mục tiêu")
        self.coverage_ax.set_ylabel("Liều (Gy)")
        self.coverage_ax.set_title("Độ phủ mục tiêu (D95)")
        self.coverage_ax.grid(True)
        self.coverage_canvas.figure.tight_layout()
        self.coverage_canvas.draw()

    def set_robustness_result(self, result: Any):
        """
        Đặt kết quả phân tích độ bền vững để hiển thị.

        Parameters
        ----------
        result : Any
            Đối tượng kết quả phân tích độ bền vững
        """
        self.robustness_result = result

        # Cập nhật danh sách cấu trúc
        self._update_structure_list()

        # Cập nhật tất cả tab hiển thị
        self._update_dvh_plot()
        self._update_coverage_plot()
        self._update_metrics_table()

    def _update_structure_list(self):
        """Cập nhật danh sách cấu trúc từ kết quả phân tích."""
        if not self.robustness_result:
            return

        # Lưu cấu trúc đang chọn
        current_structure = self.structure_combo.currentText()

        # Xóa và cập nhật combo
        self.structure_combo.clear()

        try:
            # Lấy danh sách cấu trúc từ kết quả
            structure_names = self.robustness_result.get_structure_names()

            if structure_names:
                self.structure_combo.addItems(structure_names)

                # Chọn lại cấu trúc trước đó nếu có
                if current_structure and current_structure in structure_names:
                    index = self.structure_combo.findText(current_structure)
                    if index >= 0:
                        self.structure_combo.setCurrentIndex(index)

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật danh sách cấu trúc: {str(e)}")

    def _update_display_mode(self):
        """Cập nhật chế độ hiển thị dựa trên lựa chọn người dùng."""
        mode = self.display_mode_combo.currentText()

        if mode == "DVH Bands":
            self.tab_widget.setCurrentIndex(0)
        elif mode == "Coverage":
            self.tab_widget.setCurrentIndex(1)
        elif mode == "Metrics":
            self.tab_widget.setCurrentIndex(2)
        # Mode "All" just keeps current tab

    def _update_dvh_plot(self):
        """Cập nhật biểu đồ DVH cho cấu trúc đang được chọn."""
        if not self.robustness_result:
            return

        structure_name = self.structure_combo.currentText()
        if not structure_name:
            return

        # Xóa biểu đồ hiện tại
        self.dvh_ax.clear()

        try:
            # Thiết lập lại tiêu đề và nhãn
            self.dvh_ax.set_xlabel("Liều (Gy)")
            self.dvh_ax.set_ylabel("Thể tích (%)")
            self.dvh_ax.set_title(f"DVH Robustness Band: {structure_name}")
            self.dvh_ax.set_xlim([0, 80])
            self.dvh_ax.set_ylim([0, 105])
            self.dvh_ax.grid(True)

            # Lấy dữ liệu DVH từ kết quả phân tích
            structure_dvhs = self.robustness_result.get_structure_dvhs(structure_name)

            if structure_dvhs:
                # Vẽ nominal DVH nếu được chọn
                if self.show_nominal_check.isChecked() and "nominal" in structure_dvhs:
                    nominal_dvh = structure_dvhs["nominal"]
                    self.dvh_ax.plot(
                        nominal_dvh["dose"],
                        nominal_dvh["volume"],
                        "k-",
                        linewidth=2,
                        label="Nominal",
                    )

                # Vẽ worst case DVH nếu được chọn
                if self.show_worst_check.isChecked() and "worst" in structure_dvhs:
                    worst_dvh = structure_dvhs["worst"]
                    self.dvh_ax.plot(
                        worst_dvh["dose"],
                        worst_dvh["volume"],
                        "r--",
                        linewidth=2,
                        label="Worst Case",
                    )

                # Vẽ tất cả các scenarios nếu được chọn
                if self.show_all_check.isChecked() and "scenarios" in structure_dvhs:
                    scenarios = structure_dvhs["scenarios"]
                    for i, scenario_dvh in enumerate(scenarios):
                        self.dvh_ax.plot(
                            scenario_dvh["dose"],
                            scenario_dvh["volume"],
                            "-",
                            color="gray",
                            alpha=0.3,
                            linewidth=1,
                        )
                    # Thêm một đường mẫu cho legend
                    self.dvh_ax.plot(
                        [],
                        [],
                        "-",
                        color="gray",
                        alpha=0.5,
                        linewidth=1,
                        label="Scenarios",
                    )

                # Vẽ band (min-max) nếu có dữ liệu
                if "min" in structure_dvhs and "max" in structure_dvhs:
                    min_dvh = structure_dvhs["min"]
                    max_dvh = structure_dvhs["max"]

                    # Điều chỉnh mảng liều để có cùng kích thước
                    min_dose = min_dvh["dose"]
                    max_dose = max_dvh["dose"]
                    min_vol = min_dvh["volume"]
                    max_vol = max_dvh["volume"]

                    self.dvh_ax.fill_between(
                        min_dose,
                        min_vol,
                        max_vol,
                        color="lightblue",
                        alpha=0.5,
                        label="Min-Max Band",
                    )

                self.dvh_ax.legend(loc="best")
            else:
                self.dvh_ax.text(
                    0.5,
                    0.5,
                    f"Không có dữ liệu DVH cho {structure_name}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=self.dvh_ax.transAxes,
                )

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật biểu đồ DVH: {str(e)}")
            self.dvh_ax.text(
                0.5,
                0.5,
                f"Lỗi: {str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.dvh_ax.transAxes,
            )

        # Cập nhật canvas
        self.dvh_canvas.figure.tight_layout()
        self.dvh_canvas.draw()

    def _update_coverage_plot(self):
        """Cập nhật biểu đồ độ phủ cho các cấu trúc mục tiêu."""
        if not self.robustness_result:
            return

        # Xóa biểu đồ hiện tại
        self.coverage_ax.clear()

        try:
            # Lấy loại chỉ số được chọn
            metric_type = self.metric_combo.currentText()

            # Thiết lập lại tiêu đề và nhãn
            self.coverage_ax.set_xlabel("Cấu trúc mục tiêu")
            self.coverage_ax.set_ylabel(
                f"{metric_type} (Gy)" if metric_type != "V95" else "V95 (%)"
            )
            self.coverage_ax.set_title(f"Độ phủ mục tiêu: {metric_type}")
            self.coverage_ax.grid(True)

            # Lấy dữ liệu độ phủ từ kết quả phân tích
            coverage_data = self.robustness_result.get_target_coverage_data(metric_type)

            if coverage_data and coverage_data["target_names"]:
                target_names = coverage_data["target_names"]
                nominal_values = coverage_data["nominal"]
                min_values = coverage_data["min"]
                max_values = coverage_data["max"]

                # Thiết lập trục x
                x = np.arange(len(target_names))
                width = 0.6  # Độ rộng của bar

                # Vẽ dải min-max
                self.coverage_ax.bar(
                    x,
                    np.array(max_values) - np.array(min_values),
                    bottom=min_values,
                    width=width,
                    alpha=0.5,
                    color="lightblue",
                    label="Min-Max Range",
                )

                # Vẽ giá trị nominal
                self.coverage_ax.scatter(
                    x, nominal_values, color="red", s=50, zorder=3, label="Nominal"
                )

                # Thiết lập các nhãn trục x
                self.coverage_ax.set_xticks(x)
                self.coverage_ax.set_xticklabels(target_names)

                self.coverage_ax.legend(loc="best")
            else:
                self.coverage_ax.text(
                    0.5,
                    0.5,
                    f"Không có dữ liệu độ phủ cho {metric_type}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=self.coverage_ax.transAxes,
                )

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật biểu đồ độ phủ: {str(e)}")
            self.coverage_ax.text(
                0.5,
                0.5,
                f"Lỗi: {str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.coverage_ax.transAxes,
            )

        # Cập nhật canvas
        self.coverage_canvas.figure.tight_layout()
        self.coverage_canvas.draw()

    def _update_metrics_table(self):
        """Cập nhật bảng chỉ số đánh giá với dữ liệu phân tích."""
        if not self.robustness_result:
            return

        try:
            # Xóa bảng hiện tại
            self.metrics_table.setRowCount(0)

            # Lấy dữ liệu các chỉ số từ kết quả phân tích
            metrics = self.robustness_result.get_evaluation_metrics()

            if not metrics:
                return

            # Thiết lập số hàng
            row_count = sum(len(data) for data in metrics.values())
            self.metrics_table.setRowCount(row_count)

            # Điền dữ liệu vào bảng
            current_row = 0

            # Thêm chỉ số cho targets
            for target_name, target_metrics in metrics.get("targets", {}).items():
                for metric_name, values in target_metrics.items():
                    self._add_metric_to_table(
                        current_row, target_name, metric_name, values
                    )
                    current_row += 1

            # Thêm chỉ số cho OARs
            for oar_name, oar_metrics in metrics.get("oars", {}).items():
                for metric_name, values in oar_metrics.items():
                    self._add_metric_to_table(
                        current_row, oar_name, metric_name, values
                    )
                    current_row += 1

            # Điều chỉnh kích thước bảng
            self.metrics_table.resizeColumnsToContents()
            self.metrics_table.resizeRowsToContents()

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật bảng chỉ số: {str(e)}")
            self.metrics_table.setRowCount(1)
            self.metrics_table.setItem(0, 0, QTableWidgetItem(f"Lỗi: {str(e)}"))

    def _add_metric_to_table(self, row, structure_name, metric_name, values):
        """
        Thêm một chỉ số vào bảng metrics.

        Parameters
        ----------
        row : int
            Số hàng cần thêm
        structure_name : str
            Tên cấu trúc
        metric_name : str
            Tên chỉ số
        values : Dict
            Các giá trị của chỉ số (nominal, min, max)
        """
        # Cột 0: Tên cấu trúc
        self.metrics_table.setItem(row, 0, QTableWidgetItem(structure_name))

        # Cột 1: Tên chỉ số
        self.metrics_table.setItem(row, 1, QTableWidgetItem(metric_name))

        # Cột 2-5: Các giá trị
        nominal = values.get("nominal", 0)
        min_val = values.get("min", 0)
        max_val = values.get("max", 0)
        range_val = max_val - min_val

        self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{nominal:.2f}"))
        self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{min_val:.2f}"))
        self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{max_val:.2f}"))
        self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{range_val:.2f}"))

        # Đặt màu nền dựa trên phạm vi
        normalized_range = min(1.0, range_val / nominal if nominal > 0 else 0)
        bg_color = QColor(
            int(255 * normalized_range),  # Red
            int(255 * (1 - normalized_range)),  # Green
            0,  # Blue
        )

        for col in range(3, 6):
            self.metrics_table.item(row, col).setBackground(bg_color)

    def _request_export(self):
        """Phát tín hiệu yêu cầu xuất kết quả."""
        if self.robustness_result:
            self.exportRequested.emit(self.robustness_result)
