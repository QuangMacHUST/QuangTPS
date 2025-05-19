#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tab phân tích độ bền vững trong QuangTPS.

Module này cung cấp giao diện người dùng để phân tích độ bền vững
của kế hoạch điều trị với phong cách Eclipse, cho phép đánh giá sự thay đổi
của phân phối liều khi có sự thay đổi về vị trí bệnh nhân hoặc độ không
chắc chắn về phạm vi.
"""

import logging
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union

try:
    from PyQt5.QtCore import Qt, QSize, pyqtSignal, QMetaObject, Q_ARG
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QSpinBox,
        QDoubleSpinBox,
        QCheckBox,
        QTabWidget,
        QGroupBox,
        QScrollArea,
        QGridLayout,
        QFileDialog,
        QMessageBox,
        QProgressBar,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
    )
    from PyQt5.QtGui import QIcon, QColor, QPalette
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_PYQT = True
except ImportError:
    # Fallback for when PyQt is not available
    from PySide2.QtCore import Qt, QSize, Signal as pyqtSignal
    from PySide2.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QSpinBox,
        QDoubleSpinBox,
        QCheckBox,
        QTabWidget,
        QGroupBox,
        QScrollArea,
        QGridLayout,
        QFileDialog,
        QMessageBox,
        QProgressBar,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
    )
    from PySide2.QtGui import QIcon, QColor, QPalette
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_PYQT = True
except Exception:
    HAS_PYQT = False
    import warnings

    warnings.warn(
        "Không thể import PyQt hoặc PySide, RobustAnalysisTab sẽ là placeholder"
    )

logger = logging.getLogger(__name__)


class RobustAnalysisTab(QWidget):
    """
    Tab phân tích độ bền vững theo phong cách Eclipse.

    Tab này cung cấp giao diện người dùng để phân tích và hiển thị kết quả
    đánh giá độ bền vững của kế hoạch điều trị, bao gồm DVH bands,
    biểu đồ độ phủ và các chỉ số định lượng.
    """

    # Signals
    planUpdated = pyqtSignal(object)  # Phát khi kế hoạch được cập nhật
    analysisComplete = pyqtSignal(object)  # Phát khi phân tích hoàn tất

    def __init__(self, parent=None):
        """
        Khởi tạo tab phân tích độ bền vững.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, by default None
        """
        super().__init__(parent)

        # Dữ liệu nội bộ
        self.plan = None
        self.structures = {}
        self.dose_grid = None
        self.dose_calculator = None
        self.robustness_result = None
        self.analyzing = False

        # Thiết lập UI
        if HAS_PYQT:
            self._setup_ui()
            self._init_plots()
        else:
            # Placeholder khi không có PyQt
            layout = QVBoxLayout(self)
            label = QLabel("PyQt hoặc PySide là bắt buộc cho RobustAnalysisTab")
            layout.addWidget(label)

    def set_plan(self, plan):
        """
        Thiết lập kế hoạch để phân tích.

        Parameters
        ----------
        plan : Any
            Đối tượng kế hoạch xạ trị
        """
        self.plan = plan

        # Cập nhật UI nếu có sẵn
        if hasattr(self, "analyze_button"):
            self.analyze_button.setEnabled(plan is not None)

        # Cập nhật danh sách cấu trúc
        self._update_structures()

        # Đặt lại các phân tích trước đó
        self.robustness_result = None
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(False)

    def _setup_ui(self):
        """Thiết lập giao diện người dùng với phong cách Eclipse."""
        main_layout = QVBoxLayout(self)

        # Tạo splitter chính để chia màn hình thành 2 phần: cấu hình và kết quả
        main_splitter = QSplitter(Qt.Horizontal)

        # Phần cấu hình (bên trái)
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        # Tiêu đề
        title_label = QLabel("Phân tích độ bền vững")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #003366;")
        config_layout.addWidget(title_label)

        # Thiết lập thông số độ không chắc chắn
        uncertainty_group = QGroupBox("Thông số độ không chắc chắn")
        uncertainty_layout = QGridLayout(uncertainty_group)

        # Độ không chắc chắn về vị trí
        uncertainty_layout.addWidget(QLabel("Vị trí (mm):"), 0, 0)
        self.setup_uncertainty_spin = QDoubleSpinBox()
        self.setup_uncertainty_spin.setRange(0, 10)
        self.setup_uncertainty_spin.setValue(3.0)
        self.setup_uncertainty_spin.setSingleStep(0.5)
        uncertainty_layout.addWidget(self.setup_uncertainty_spin, 0, 1)

        # Độ không chắc chắn về phạm vi (cho proton)
        uncertainty_layout.addWidget(QLabel("Phạm vi (%):"), 1, 0)
        self.range_uncertainty_spin = QDoubleSpinBox()
        self.range_uncertainty_spin.setRange(0, 5)
        self.range_uncertainty_spin.setValue(3.0)
        self.range_uncertainty_spin.setSingleStep(0.5)
        uncertainty_layout.addWidget(self.range_uncertainty_spin, 1, 1)

        # Số lượng scenarios để phân tích
        uncertainty_layout.addWidget(QLabel("Số kịch bản:"), 2, 0)
        self.scenarios_spin = QSpinBox()
        self.scenarios_spin.setRange(5, 30)
        self.scenarios_spin.setValue(10)
        uncertainty_layout.addWidget(self.scenarios_spin, 2, 1)

        config_layout.addWidget(uncertainty_group)

        # Nhóm lựa chọn cấu trúc
        structure_group = QWidget()
        structure_layout = QVBoxLayout(structure_group)
        structure_layout.addWidget(QLabel("Lựa chọn cấu trúc:"))

        # Targets group
        targets_group = QGroupBox("Cấu trúc mục tiêu (Targets)")
        targets_layout = QVBoxLayout(targets_group)

        self.target_checkboxes = {}  # Sẽ được điền sau
        targets_layout.addWidget(QLabel("Không có cấu trúc mục tiêu"))

        targets_group.setLayout(targets_layout)
        structure_layout.addWidget(targets_group)

        # OARs group
        oars_group = QGroupBox("Cơ quan nguy cấp (OARs)")
        oars_layout = QVBoxLayout(oars_group)

        self.oar_checkboxes = {}  # Sẽ được điền sau
        oars_layout.addWidget(QLabel("Không có cơ quan nguy cấp"))

        oars_group.setLayout(oars_layout)
        structure_layout.addWidget(oars_group)

        # Thêm scroll area cho danh sách cấu trúc
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(structure_group)

        config_layout.addWidget(scroll_area)

        # Các nút cho phần cấu hình
        buttons_layout = QHBoxLayout()

        self.analyze_button = QPushButton("Phân tích độ bền vững")
        self.analyze_button.setIcon(QIcon("quangtps/ui/icons/analyze_icon.png"))
        self.analyze_button.clicked.connect(self._analyze_robustness)
        buttons_layout.addWidget(self.analyze_button)

        self.optimize_button = QPushButton("Tối ưu hóa kế hoạch")
        self.optimize_button.setIcon(QIcon("quangtps/ui/icons/optimize_icon.png"))
        self.optimize_button.clicked.connect(self._optimize_robustness)
        buttons_layout.addWidget(self.optimize_button)

        config_layout.addLayout(buttons_layout)

        # Thêm widget cấu hình vào splitter
        main_splitter.addWidget(config_widget)

        # Phần kết quả (bên phải)
        results_widget = QWidget()

    def _init_plots(self):
        """Khởi tạo các biểu đồ trống."""
        # Khởi tạo biểu đồ DVH
        self.dvh_ax = self.dvh_canvas.figure.add_subplot(111)
        self.dvh_ax.set_xlabel("Liều (Gy)")
        self.dvh_ax.set_ylabel("Thể tích (%)")
        self.dvh_ax.set_title("DVH Robustness Bands")
        self.dvh_ax.set_xlim([0, 80])
        self.dvh_ax.set_ylim([0, 105])
        self.dvh_ax.grid(True)
        self.dvh_canvas.figure.tight_layout()
        self.dvh_canvas.draw()

        # Khởi tạo biểu đồ độ phủ
        self.coverage_ax = self.coverage_canvas.figure.add_subplot(111)
        self.coverage_ax.set_xlabel("Cấu trúc mục tiêu")
        self.coverage_ax.set_ylabel("Liều (Gy)")
        self.coverage_ax.set_title("Độ phủ mục tiêu (D95)")
        self.coverage_ax.grid(True)
        self.coverage_canvas.figure.tight_layout()
        self.coverage_canvas.draw()

        # Khởi tạo biểu đồ phân tích không gian
        self.spatial_ax = self.spatial_canvas.figure.add_subplot(111)
        self.spatial_ax.set_title("Phân tích không gian độ bền vững")
        self.spatial_ax.set_xlabel("X (mm)")
        self.spatial_ax.set_ylabel("Y (mm)")
        self.spatial_ax.grid(True)
        self.spatial_canvas.figure.tight_layout()
        self.spatial_canvas.draw()

    def _update_structures(self):
        """Cập nhật danh sách cấu trúc từ kế hoạch hiện tại."""
        self.structures = {}
        if not self.plan or not hasattr(self.plan, "structures"):
            return

        try:
            # Lấy cấu trúc từ kế hoạch
            self.structures = self.plan.structures

            # Phân loại cấu trúc thành targets và OARs
            targets = {}
            oars = {}

            for name, structure in self.structures.items():
                if name.lower().startswith(("ptv", "ctv", "gtv")):
                    targets[name] = structure
                else:
                    oars[name] = structure

            # Cập nhật UI với cấu trúc mới
            self._update_structure_ui(targets, oars)

            # Cập nhật combobox cho DVH
            self.dvh_structure_combo.clear()
            structure_names = list(self.structures.keys())
            self.dvh_structure_combo.addItems(structure_names)

            # Cập nhật trạng thái nút phân tích
            self.analyze_button.setEnabled(len(self.structures) > 0)

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật cấu trúc: {str(e)}")
            self.analyze_button.setEnabled(False)

    def _update_structure_ui(self, targets, oars):
        """
        Cập nhật UI hiển thị cấu trúc mục tiêu và cơ quan nguy cấp.

        Parameters
        ----------
        targets : Dict
            Dictionary các cấu trúc mục tiêu
        oars : Dict
            Dictionary các cơ quan nguy cấp
        """
        # Cập nhật checkboxes cho targets
        if hasattr(self, "target_checkboxes"):
            # Xóa layout cũ
            for i in reversed(range(self.targets_layout.count())):
                widget = self.targets_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            self.target_checkboxes.clear()

            if targets:
                for name in targets:
                    checkbox = QCheckBox(name)
                    checkbox.setChecked(True)
                    self.target_checkboxes[name] = checkbox
                    self.targets_layout.addWidget(checkbox)
            else:
                self.targets_layout.addWidget(QLabel("Không có cấu trúc mục tiêu"))

        # Cập nhật checkboxes cho OARs
        if hasattr(self, "oar_checkboxes"):
            # Xóa layout cũ
            for i in reversed(range(self.oars_layout.count())):
                widget = self.oars_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            self.oar_checkboxes.clear()

            if oars:
                for name in oars:
                    checkbox = QCheckBox(name)
                    checkbox.setChecked(True)
                    self.oar_checkboxes[name] = checkbox
                    self.oars_layout.addWidget(checkbox)
            else:
                self.oars_layout.addWidget(QLabel("Không có cơ quan nguy cấp"))

    def _analyze_robustness(self):
        """Phân tích độ bền vững của kế hoạch hiện tại."""
        if self.analyzing:
            return

        if not self.plan:
            QMessageBox.warning(
                self,
                "Không có kế hoạch",
                "Vui lòng tải kế hoạch trước khi phân tích độ bền vững.",
            )
            return

        if not hasattr(self.plan, "dose_grid") or self.plan.dose_grid is None:
            QMessageBox.warning(
                self,
                "Không có dữ liệu liều",
                "Cần tính toán liều trước khi phân tích độ bền vững.",
            )
            return

        # Lấy cấu trúc được lựa chọn
        selected_structures = {}
        for name, checkbox in self.target_checkboxes.items():
            if checkbox.isChecked():
                selected_structures[name] = self.structures[name]

        for name, checkbox in self.oar_checkboxes.items():
            if checkbox.isChecked():
                selected_structures[name] = self.structures[name]

        if not selected_structures:
            QMessageBox.warning(
                self,
                "Không có cấu trúc nào được chọn",
                "Vui lòng chọn ít nhất một cấu trúc để phân tích.",
            )
            return

        # Lấy thông số
        setup_uncertainty = self.setup_uncertainty_spin.value()
        range_uncertainty = self.range_uncertainty_spin.value()
        num_scenarios = self.scenarios_spin.value()

        # Hiển thị thanh tiến trình
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Đang phân tích độ bền vững...")
        self.analyze_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.analyzing = True

        # Thực hiện phân tích trong thread riêng
        def analysis_thread():
            try:
                # Import phân tích độ bền vững
                from quangtps.evaluation.robustness.robustness_analyzer import (
                    RobustnessAnalyzer,
                )

                # Tạo analyzer
                analyzer = RobustnessAnalyzer(
                    plan=self.plan,
                    structures=selected_structures,
                    setup_uncertainty=setup_uncertainty,
                    range_uncertainty=range_uncertainty,
                    num_scenarios=num_scenarios,
                )

                # Phân tích
                self.robustness_result = analyzer.analyze()

                # Cập nhật UI trong main thread
                if HAS_PYQT:
                    QMetaObject.invokeMethod(
                        self, "_analysis_complete", Qt.QueuedConnection
                    )
            except Exception as e:
                logger.error(f"Lỗi khi phân tích độ bền vững: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())

                # Cập nhật UI trong main thread
                if HAS_PYQT:
                    QMetaObject.invokeMethod(
                        self, "_analysis_error", Qt.QueuedConnection, Q_ARG(str, str(e))
                    )

        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()

    def _analysis_complete(self):
        """Xử lý khi phân tích hoàn tất."""
        self.progress_bar.setValue(100)
        self.status_label.setText("Phân tích hoàn tất")
        self.analyze_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.report_button.setEnabled(True)
        self.analyzing = False

        # Cập nhật UI với kết quả
        self._update_results_ui()

        # Phát tín hiệu
        self.analysisComplete.emit(self.robustness_result)

    def _analysis_error(self, error_message):
        """Xử lý khi phân tích gặp lỗi."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Phân tích thất bại")
        self.analyze_button.setEnabled(True)
        self.optimize_button.setEnabled(False)
        self.analyzing = False

        QMessageBox.critical(
            self,
            "Lỗi phân tích",
            f"Lỗi trong quá trình phân tích độ bền vững: {error_message}",
        )

    def _update_results_ui(self):
        """Cập nhật UI với kết quả phân tích hiện tại."""
        if not self.robustness_result:
            return

        # Cập nhật biểu đồ DVH cho cấu trúc đang chọn
        self._update_dvh_plot()

        # Cập nhật biểu đồ độ phủ
        self._update_coverage_plot()

        # Cập nhật bảng chỉ số đánh giá
        self._update_metrics_table()

        # Cập nhật phân tích không gian
        self._update_spatial_plot()

    def _update_dvh_plot(self):
        """Cập nhật biểu đồ DVH với dữ liệu cho cấu trúc đang được chọn."""
        if not self.robustness_result:
            return

        structure_name = self.dvh_structure_combo.currentText()
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
                if (
                    self.show_all_scenarios_check.isChecked()
                    and "scenarios" in structure_dvhs
                ):
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
            metric_type = self.coverage_metric_combo.currentText()

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

                # Thêm prescription line nếu có
                if hasattr(self.plan, "prescription") and self.plan.prescription:
                    try:
                        rx_dose = self.plan.prescription.dose
                        self.coverage_ax.axhline(
                            y=rx_dose,
                            color="green",
                            linestyle="--",
                            linewidth=2,
                            label="Prescription",
                        )
                    except:
                        pass

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
        # Màu xanh lá cho phạm vi nhỏ, màu đỏ cho phạm vi lớn
        normalized_range = min(1.0, range_val / nominal if nominal > 0 else 0)
        bg_color = QColor(
            int(255 * normalized_range),  # Red
            int(255 * (1 - normalized_range)),  # Green
            0,  # Blue
        )

        for col in range(3, 6):
            self.metrics_table.item(row, col).setBackground(bg_color)

    def _update_spatial_plot(self):
        """Cập nhật biểu đồ phân tích không gian."""
        if not self.robustness_result:
            return

        # Xóa biểu đồ hiện tại
        self.spatial_ax.clear()

        try:
            # Lấy loại hiển thị được chọn
            display_type = self.spatial_display_combo.currentText()

            # Thiết lập lại tiêu đề và nhãn
            self.spatial_ax.set_title(f"Phân tích không gian: {display_type}")
            self.spatial_ax.set_xlabel("X (mm)")
            self.spatial_ax.set_ylabel("Y (mm)")
            self.spatial_ax.grid(True)

            # Lấy dữ liệu không gian từ kết quả phân tích
            spatial_data = self.robustness_result.get_spatial_analysis_data(
                display_type
            )

            if spatial_data and "data" in spatial_data:
                # Tạo heatmap
                im = self.spatial_ax.imshow(
                    spatial_data["data"],
                    cmap=spatial_data.get("colormap", "viridis"),
                    aspect="equal",
                    origin="lower",
                    extent=spatial_data.get("extent", [0, 100, 0, 100]),
                )

                # Thêm thanh màu
                self.spatial_canvas.figure.colorbar(im, ax=self.spatial_ax)

                # Thêm đường bao cấu trúc nếu có
                if "contours" in spatial_data:
                    for name, contour in spatial_data["contours"].items():
                        self.spatial_ax.plot(
                            contour["x"], contour["y"], "k-", linewidth=1, label=name
                        )

                    self.spatial_ax.legend(loc="best")
            else:
                self.spatial_ax.text(
                    0.5,
                    0.5,
                    f"Không có dữ liệu không gian cho {display_type}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=self.spatial_ax.transAxes,
                )

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật biểu đồ không gian: {str(e)}")
            self.spatial_ax.text(
                0.5,
                0.5,
                f"Lỗi: {str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.spatial_ax.transAxes,
            )

        # Cập nhật canvas
        self.spatial_canvas.figure.tight_layout()
        self.spatial_canvas.draw()

    def _export_results(self):
        """Xuất kết quả phân tích ra file."""
        if not self.robustness_result:
            return

        # Hỏi file xuất
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất kết quả phân tích độ bền vững",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx)",
        )

        if not filename:
            return

        try:
            # Xuất kết quả theo định dạng file
            if filename.endswith(".csv"):
                self.robustness_result.export_to_csv(filename)
            elif filename.endswith(".xlsx"):
                self.robustness_result.export_to_excel(filename)

            QMessageBox.information(
                self,
                "Xuất kết quả thành công",
                f"Kết quả phân tích đã được xuất ra file {filename}.",
            )

        except Exception as e:
            logger.error(f"Lỗi khi xuất kết quả: {str(e)}")
            QMessageBox.critical(
                self, "Lỗi xuất kết quả", f"Không thể xuất kết quả: {str(e)}"
            )

    def _create_report(self):
        """Tạo báo cáo phân tích độ bền vững."""
        if not self.robustness_result:
            return

        # Hỏi file báo cáo
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Tạo báo cáo phân tích độ bền vững",
            "",
            "PDF Files (*.pdf);;HTML Files (*.html)",
        )

        if not filename:
            return

        try:
            # Tạo báo cáo theo định dạng file
            if filename.endswith(".pdf"):
                self.robustness_result.create_pdf_report(filename, plan=self.plan)
            elif filename.endswith(".html"):
                self.robustness_result.create_html_report(filename, plan=self.plan)

            QMessageBox.information(
                self,
                "Tạo báo cáo thành công",
                f"Báo cáo phân tích đã được tạo tại {filename}.",
            )

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo: {str(e)}")
            QMessageBox.critical(
                self, "Lỗi tạo báo cáo", f"Không thể tạo báo cáo: {str(e)}"
            )

    def _optimize_robustness(self):
        """Tối ưu hóa kế hoạch cho độ bền vững."""
        QMessageBox.information(
            self,
            "Tính năng đang phát triển",
            "Tính năng tối ưu hóa độ bền vững đang được phát triển và sẽ có sẵn trong phiên bản tới.",
        )
