#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog phân tích độ bền vững (Robustness Analysis) theo phong cách Eclipse.

Dialog này cho phép người dùng phân tích độ bền vững của kế hoạch xạ trị,
với khả năng thiết lập các tham số không chắc chắn và hiển thị kết quả trực quan.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import numpy as np
import time
from datetime import datetime

try:
    from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QTabWidget,
        QWidget,
        QGroupBox,
        QFormLayout,
        QCheckBox,
        QProgressBar,
        QMessageBox,
        QSpinBox,
        QDoubleSpinBox,
        QSplitter,
        QFrame,
        QApplication,
        QTreeWidget,
        QTreeWidgetItem,
        QRadioButton,
        QButtonGroup,
        QFileDialog,
    )
    from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QFont, QColor, QIcon
except ImportError:
    logging.error(
        "PyQt5 không khả dụng. Dialog phân tích độ bền vững sẽ không hoạt động."
    )

    # Tạo các lớp giả cho IDE
    class QDialog:
        pass

    class QWidget:
        pass

    class pyqtSignal:
        pass


try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_pdf import PdfPages

    HAS_MPL = True
except ImportError:
    logging.error("matplotlib không khả dụng. Các biểu đồ sẽ không hiển thị được.")
    HAS_MPL = False

try:
    from quangtps.evaluation.robustness import RobustnessAnalyzer, RobustnessResult
    from quangtps.utils.ui_utils import create_eclipse_icon

    HAS_ROBUSTNESS_MODULE = True
except ImportError:
    logging.error("Module phân tích độ bền vững không khả dụng.")
    HAS_ROBUSTNESS_MODULE = False

logger = logging.getLogger(__name__)


class RobustnessDVHWidget(QWidget):
    """Widget hiển thị DVH với dải biến động (DVH bands) cho phân tích độ bền vững."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = None
        self.canvas = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        if HAS_MPL:
            self.figure = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)

            # Thêm một checkbox để bật/tắt hiển thị dải DVH
            self.show_bands_cb = QCheckBox("Hiển thị dải DVH (độ dao động)")
            self.show_bands_cb.setChecked(True)
            self.show_bands_cb.toggled.connect(self._update_plot)
            layout.addWidget(self.show_bands_cb)

            # Thêm các radio button để chọn hiển thị các cấu trúc
            self.structure_group = QGroupBox("Cấu trúc")
            self.structure_layout = QVBoxLayout()
            self.structure_group.setLayout(self.structure_layout)
            self.structure_buttons = QButtonGroup()
            self.structure_buttons.buttonClicked.connect(self._update_plot)
            layout.addWidget(self.structure_group)
        else:
            layout.addWidget(
                QLabel("matplotlib không khả dụng. Không thể hiển thị DVH.")
            )

    def set_robustness_result(self, result):
        """Thiết lập kết quả phân tích độ bền vững và cập nhật biểu đồ."""
        if not HAS_MPL:
            return

        self.result = result

        # Xóa các radio button cũ
        for i in reversed(range(self.structure_layout.count())):
            widget = self.structure_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Thêm radio button cho mỗi cấu trúc
        structures = result.get_structures() if result else []
        for i, structure in enumerate(structures):
            rb = QRadioButton(structure)
            if i == 0:
                rb.setChecked(True)
            self.structure_buttons.addButton(rb, i)
            self.structure_layout.addWidget(rb)

        self._update_plot()

    def _update_plot(self):
        """Cập nhật biểu đồ DVH với dải biến động."""
        if not HAS_MPL or not hasattr(self, "result") or not self.result:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Xác định cấu trúc được chọn
        selected_button = self.structure_buttons.checkedButton()
        if not selected_button:
            return

        structure_name = selected_button.text()
        show_bands = self.show_bands_cb.isChecked()

        # Lấy dữ liệu DVH cho cấu trúc
        dvh_data = self.result.get_structure_dvhs(structure_name)

        if not dvh_data:
            return

        # Vẽ DVH danh nghĩa (đường thường)
        doses = dvh_data.get("nominal", {}).get("doses", [])
        volumes = dvh_data.get("nominal", {}).get("volumes", [])

        if len(doses) > 0 and len(volumes) > 0:
            ax.plot(doses, volumes, "b-", linewidth=2, label="Kế hoạch danh nghĩa")

        # Vẽ dải DVH nếu được yêu cầu
        if show_bands:
            min_volumes = dvh_data.get("min", {}).get("volumes", [])
            max_volumes = dvh_data.get("max", {}).get("volumes", [])

            if len(doses) > 0 and len(min_volumes) > 0 and len(max_volumes) > 0:
                ax.fill_between(
                    doses,
                    min_volumes,
                    max_volumes,
                    alpha=0.3,
                    color="b",
                    label="Độ dao động độ bền vững",
                )

        ax.set_xlabel("Liều (Gy)")
        ax.set_ylabel("Thể tích (%)")
        ax.set_title(f"Biểu đồ DVH cho {structure_name} với phân tích độ bền vững")
        ax.grid(True)
        ax.legend()

        self.canvas.draw()


class RobustnessMetricsTable(QTableWidget):
    """Bảng hiển thị các chỉ số đánh giá độ bền vững."""

    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self._init_ui()

    def _init_ui(self):
        self.setHorizontalHeaderLabels(
            ["Cấu trúc", "Chỉ số", "Danh nghĩa", "Tồi nhất", "Biên độ"]
        )
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)

    def set_metrics(self, metrics):
        """Thiết lập các chỉ số đánh giá độ bền vững."""
        self.clearContents()
        self.setRowCount(0)

        if not metrics:
            return

        row = 0
        for structure_name, structure_metrics in metrics.items():
            for metric_name, metric_values in structure_metrics.items():
                self.insertRow(row)

                # Cấu trúc
                self.setItem(row, 0, QTableWidgetItem(structure_name))

                # Chỉ số
                self.setItem(row, 1, QTableWidgetItem(metric_name))

                # Giá trị danh nghĩa
                nominal = metric_values.get("nominal", "N/A")
                nominal_item = QTableWidgetItem(
                    f"{nominal:.2f}"
                    if isinstance(nominal, (int, float))
                    else str(nominal)
                )
                self.setItem(row, 2, nominal_item)

                # Giá trị tồi nhất
                worst = metric_values.get("worst", "N/A")
                worst_item = QTableWidgetItem(
                    f"{worst:.2f}" if isinstance(worst, (int, float)) else str(worst)
                )
                self.setItem(row, 3, worst_item)

                # Biên độ
                if isinstance(nominal, (int, float)) and isinstance(
                    worst, (int, float)
                ):
                    amplitude = abs(worst - nominal)
                    amplitude_item = QTableWidgetItem(f"{amplitude:.2f}")

                    # Màu sắc dựa trên biên độ
                    if amplitude > 5.0:
                        amplitude_item.setBackground(QColor(255, 200, 200))  # Đỏ nhạt
                    elif amplitude > 2.0:
                        amplitude_item.setBackground(QColor(255, 255, 200))  # Vàng nhạt
                    else:
                        amplitude_item.setBackground(QColor(200, 255, 200))  # Xanh nhạt
                else:
                    amplitude_item = QTableWidgetItem("N/A")

                self.setItem(row, 4, amplitude_item)

                row += 1


class RobustnessAnalysisDialog(QDialog):
    """
    Dialog phân tích độ bền vững (Robustness Analysis) theo phong cách Eclipse.

    Dialog này cho phép người dùng thiết lập các tham số không chắc chắn
    và phân tích độ bền vững của kế hoạch xạ trị, với khả năng hiển thị
    kết quả trực quan qua DVH bands và các chỉ số đánh giá.
    """

    # Signal khi hoàn tất phân tích
    analysisCompleted = pyqtSignal(object)

    def __init__(self, plan=None, parent=None):
        super().__init__(parent)
        self.plan = plan
        self.result = None
        self.setWindowTitle("Phân tích độ bền vững (Robustness Analysis)")
        self.resize(900, 600)

        # Kiểm tra module phân tích độ bền vững
        if not HAS_ROBUSTNESS_MODULE:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Module phân tích độ bền vững không khả dụng."))
            button = QPushButton("Đóng")
            button.clicked.connect(self.reject)
            layout.addWidget(button)
            return

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # Splitter chính chia màn hình thành phần thiết lập và phần kết quả
        main_splitter = QSplitter(Qt.Horizontal)

        # 1. Panel thiết lập tham số
        setup_panel = QWidget()
        setup_layout = QVBoxLayout(setup_panel)

        # 1.1. Thiết lập độ không chắc chắn
        uncertainty_group = QGroupBox("Thiết lập độ không chắc chắn")
        uncertainty_form = QFormLayout()

        self.setup_uncertainty_spin = QDoubleSpinBox()
        self.setup_uncertainty_spin.setRange(0, 10)
        self.setup_uncertainty_spin.setValue(3.0)
        self.setup_uncertainty_spin.setSingleStep(0.5)
        self.setup_uncertainty_spin.setDecimals(1)
        self.setup_uncertainty_spin.setSuffix(" mm")
        uncertainty_form.addRow(
            "Độ không chắc chắn vị trí (setup):", self.setup_uncertainty_spin
        )

        self.range_uncertainty_spin = QDoubleSpinBox()
        self.range_uncertainty_spin.setRange(0, 5)
        self.range_uncertainty_spin.setValue(3.0)
        self.range_uncertainty_spin.setSingleStep(0.5)
        self.range_uncertainty_spin.setDecimals(1)
        self.range_uncertainty_spin.setSuffix(" %")
        uncertainty_form.addRow(
            "Độ không chắc chắn phạm vi (range):", self.range_uncertainty_spin
        )

        self.num_scenarios_spin = QSpinBox()
        self.num_scenarios_spin.setRange(1, 20)
        self.num_scenarios_spin.setValue(8)
        uncertainty_form.addRow("Số lượng kịch bản phân tích:", self.num_scenarios_spin)

        uncertainty_group.setLayout(uncertainty_form)
        setup_layout.addWidget(uncertainty_group)

        # 1.2. Lựa chọn cấu trúc quan trọng
        structures_group = QGroupBox("Cấu trúc quan trọng")
        structures_layout = QVBoxLayout()

        if (
            self.plan
            and hasattr(self.plan, "structure_set")
            and self.plan.structure_set
        ):
            self.structures_table = QTableWidget(0, 2)
            self.structures_table.setHorizontalHeaderLabels(["Cấu trúc", "Phân tích"])
            self.structures_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.Stretch
            )
            self.structures_table.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeToContents
            )

            # Thêm các cấu trúc hiện có
            for i, structure in enumerate(self.plan.structure_set.structures):
                self.structures_table.insertRow(i)
                self.structures_table.setItem(i, 0, QTableWidgetItem(structure.name))

                # Checkbox cho việc chọn cấu trúc để phân tích
                checkbox = QCheckBox()
                # Tự động chọn PTV và OAR quan trọng
                if (
                    "PTV" in structure.name
                    or "Cord" in structure.name
                    or "Brainstem" in structure.name
                ):
                    checkbox.setChecked(True)
                self.structures_table.setCellWidget(i, 1, checkbox)
        else:
            self.structures_table = QLabel(
                "Không có cấu trúc nào trong kế hoạch hiện tại"
            )

        structures_layout.addWidget(self.structures_table)
        structures_group.setLayout(structures_layout)
        setup_layout.addWidget(structures_group)

        # 1.3. Nút chạy phân tích
        analyze_button = QPushButton("Phân tích độ bền vững")
        analyze_button.setIcon(create_eclipse_icon("analyze"))
        analyze_button.clicked.connect(self._run_analysis)
        setup_layout.addWidget(analyze_button)

        # Thêm khoảng trống co giãn
        setup_layout.addStretch()

        # 1.4. Nút xuất kết quả
        export_group = QGroupBox("Xuất kết quả")
        export_layout = QHBoxLayout()

        export_pdf_button = QPushButton("Xuất PDF")
        export_pdf_button.setIcon(create_eclipse_icon("pdf"))
        export_pdf_button.clicked.connect(self._export_pdf)
        export_layout.addWidget(export_pdf_button)

        export_csv_button = QPushButton("Xuất CSV")
        export_csv_button.setIcon(create_eclipse_icon("csv"))
        export_csv_button.clicked.connect(self._export_csv)
        export_layout.addWidget(export_csv_button)

        export_group.setLayout(export_layout)
        setup_layout.addWidget(export_group)

        # 2. Panel kết quả
        results_panel = QTabWidget()

        # 2.1. Tab DVH với dải biến động
        self.dvh_widget = RobustnessDVHWidget()
        results_panel.addTab(self.dvh_widget, "DVH với dải biến động")

        # 2.2. Tab các chỉ số đánh giá
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        self.metrics_table = RobustnessMetricsTable()
        metrics_layout.addWidget(self.metrics_table)
        results_panel.addTab(metrics_tab, "Chỉ số đánh giá")

        # 2.3. Tab phân tích mục tiêu
        target_tab = QWidget()
        target_layout = QVBoxLayout(target_tab)

        # Chuẩn bị widget cho biểu đồ độ phủ mục tiêu
        if HAS_MPL:
            self.target_figure = Figure(figsize=(6, 4), dpi=100)
            self.target_canvas = FigureCanvas(self.target_figure)
            target_layout.addWidget(self.target_canvas)
        else:
            target_layout.addWidget(
                QLabel("matplotlib không khả dụng. Không thể hiển thị biểu đồ.")
            )

        results_panel.addTab(target_tab, "Độ phủ mục tiêu")

        # Thêm panel thiết lập và kết quả vào splitter
        main_splitter.addWidget(setup_panel)
        main_splitter.addWidget(results_panel)
        main_splitter.setSizes([300, 600])  # Thiết lập kích thước ban đầu

        main_layout.addWidget(main_splitter)

        # Thanh trạng thái
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Nút điều khiển
        buttons_layout = QHBoxLayout()
        self.close_button = QPushButton("Đóng")
        self.close_button.clicked.connect(self.reject)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.close_button)
        main_layout.addLayout(buttons_layout)

    def _run_analysis(self):
        """Chạy phân tích độ bền vững cho kế hoạch hiện tại."""
        # Kiểm tra xem có kế hoạch và dose grid hay không
        if not self.plan:
            QMessageBox.warning(
                self,
                "Lỗi phân tích",
                "Không tìm thấy kế hoạch. Vui lòng tải kế hoạch trước khi phân tích.",
            )
            return False

        # Kiểm tra dose grid
        if not hasattr(self.plan, "dose_grid") or self.plan.dose_grid is None:
            QMessageBox.warning(
                self,
                "Lỗi phân tích",
                "Không tìm thấy dose grid. Vui lòng tính toán liều trước khi phân tích độ bền vững.",
            )
            return False

        # Lấy các cấu trúc đã chọn
        selected_structures = {}
        for i in range(self.structures_table.rowCount()):
            checkbox = self.structures_table.cellWidget(i, 1)
            if checkbox and checkbox.isChecked():
                structure_name = self.structures_table.item(i, 0).text()
                structure = self.plan.structure_set.structures[i]
                # Kiểm tra cấu trúc có mask hay không
                if not hasattr(structure, "mask") or structure.mask is None:
                    QMessageBox.warning(
                        self,
                        "Lỗi cấu trúc",
                        f"Cấu trúc {structure_name} không có mask. Vui lòng kiểm tra lại cấu trúc.",
                    )
                    return False
                selected_structures[structure_name] = structure

        # Kiểm tra xem có cấu trúc nào được chọn không
        if not selected_structures:
            QMessageBox.warning(
                self,
                "Lỗi phân tích",
                "Không có cấu trúc nào được chọn. Vui lòng chọn ít nhất một cấu trúc để phân tích.",
            )
            return False

        # Lấy các tham số độ bền vững
        setup_uncertainty = self.setup_uncertainty_spin.value()
        range_uncertainty = self.range_uncertainty_spin.value()

        # Hiển thị thanh tiến trình
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(
            f"Phân tích độ bền vững với sai số thiết lập {setup_uncertainty}mm và sai số phạm vi {range_uncertainty}%..."
        )

        # Tạo đối tượng phân tích độ bền vững
        if HAS_ROBUSTNESS_MODULE:
            try:
                analyzer = RobustnessAnalyzer()

                # Kiểm tra và thiết lập các thuộc tính cần thiết
                if hasattr(analyzer, "set_structures"):
                    analyzer.set_structures(selected_structures)
                else:
                    logger.warning(
                        "Phương thức set_structures không có trong RobustnessAnalyzer"
                    )
                    # Fallback: Thử thiết lập trực tiếp
                    analyzer.structures = selected_structures

                if hasattr(analyzer, "set_dose_grid"):
                    analyzer.set_dose_grid(self.plan.dose_grid)
                else:
                    logger.warning(
                        "Phương thức set_dose_grid không có trong RobustnessAnalyzer"
                    )
                    # Fallback: Thử thiết lập trực tiếp
                    analyzer.dose_grid = self.plan.dose_grid

                # Thiết lập các tham số không chắc chắn
                if hasattr(analyzer, "set_setup_uncertainty"):
                    analyzer.set_setup_uncertainty(setup_uncertainty)
                else:
                    logger.warning(
                        "Phương thức set_setup_uncertainty không có trong RobustnessAnalyzer"
                    )
                    # Fallback
                    analyzer.setup_uncertainty = setup_uncertainty

                if hasattr(analyzer, "set_range_uncertainty"):
                    analyzer.set_range_uncertainty(range_uncertainty)
                else:
                    logger.warning(
                        "Phương thức set_range_uncertainty không có trong RobustnessAnalyzer"
                    )
                    # Fallback
                    analyzer.range_uncertainty = range_uncertainty

                # Thiết lập callback tiến trình
                def progress_callback(progress_percent):
                    """Cập nhật thanh tiến trình."""
                    self.progress_bar.setValue(int(progress_percent))
                    QApplication.processEvents()  # Đảm bảo UI được cập nhật

                if hasattr(analyzer, "set_progress_callback"):
                    analyzer.set_progress_callback(progress_callback)

                # Chạy phân tích
                if hasattr(analyzer, "analyze"):
                    self.result = analyzer.analyze()
                else:
                    logger.warning(
                        "Phương thức analyze không có trong RobustnessAnalyzer"
                    )
                    # Fallback: Sử dụng phân tích mô phỏng
                    self.result = self._simulate_analysis(analyzer)

                # Kiểm tra kết quả
                if self.result:
                    self.progress_bar.setValue(100)
                    self.status_label.setText(
                        f"Hoàn thành phân tích độ bền vững cho {len(selected_structures)} cấu trúc."
                    )
                    self._update_results_ui()
                    self.export_pdf_button.setEnabled(True)
                    self.export_csv_button.setEnabled(True)
                    self.analysisCompleted.emit(self.result)
                    return True
                else:
                    self.progress_bar.setValue(0)
                    self.status_label.setText("Phân tích không thành công.")
                    QMessageBox.warning(
                        self,
                        "Lỗi phân tích",
                        "Không thể tạo kết quả phân tích. Vui lòng kiểm tra dữ liệu đầu vào.",
                    )
                    return False

            except Exception as e:
                import traceback

                logger.error(f"Lỗi khi phân tích độ bền vững: {e}")
                logger.error(traceback.format_exc())
                self.progress_bar.setValue(0)
                self.status_label.setText("Phân tích không thành công.")
                QMessageBox.critical(
                    self, "Lỗi phân tích", f"Lỗi khi phân tích độ bền vững: {str(e)}"
                )
                return False
        else:
            # Mô phỏng phân tích nếu không có module
            self.status_label.setText("Mô phỏng phân tích độ bền vững...")

            # Mô phỏng tiến trình
            for i in range(101):
                self.progress_bar.setValue(i)
                time.sleep(0.01)  # Tạm dừng để hiệu ứng tiến trình
                QApplication.processEvents()  # Đảm bảo UI được cập nhật

            self.result = self._create_mock_result()
            self.status_label.setText(
                f"Hoàn thành mô phỏng phân tích độ bền vững cho {len(selected_structures)} cấu trúc."
            )
            self._update_results_ui()
            self.export_pdf_button.setEnabled(True)
            self.export_csv_button.setEnabled(True)
            self.analysisCompleted.emit(self.result)
            return True

    def _simulate_analysis(self, analyzer):
        """Giả lập quá trình phân tích độ bền vững (cho mục đích demo)."""
        # Trong triển khai thật, bạn sẽ gọi analyzer.analyze(self.plan) tại đây

        # Giả lập tiến trình
        for i in range(101):
            self.progress_bar.setValue(i)

            if i == 100:
                # Tạo kết quả giả lập
                self.result = self._create_mock_result()

                # Cập nhật giao diện với kết quả
                self._update_results_ui()

                # Phát signal hoàn tất phân tích
                self.analysisCompleted.emit(self.result)

                # Hiển thị thông báo hoàn tất
                QMessageBox.information(
                    self,
                    "Hoàn tất",
                    "Phân tích độ bền vững đã hoàn tất. Xem kết quả trong các tab.",
                )

            QApplication.processEvents()
            time.sleep(0.02)  # Tạm dừng để giả lập tính toán

        self.progress_bar.setVisible(False)

    def _create_mock_result(self):
        """Tạo kết quả giả lập cho mục đích demo."""
        # Trong triển khai thật, kết quả này sẽ đến từ analyzer.analyze(self.plan)

        class MockResult:
            def get_structures(self):
                return ["PTV", "Cord", "Parotid_L", "Parotid_R"]

            def get_structure_dvhs(self, structure_name):
                # Tạo dữ liệu DVH giả lập
                doses = np.linspace(0, 70, 100)

                if structure_name == "PTV":
                    nominal = np.concatenate(
                        [np.ones(80) * 100, np.linspace(100, 0, 20)]
                    )
                    min_vals = nominal - np.random.uniform(0, 10, size=100)
                    min_vals = np.clip(min_vals, 0, 100)
                    max_vals = np.minimum(
                        nominal + np.random.uniform(0, 5, size=100), 100
                    )
                elif "Parotid" in structure_name:
                    nominal = np.exp(-doses / 30) * 100
                    min_vals = nominal - np.random.uniform(0, 5, size=100)
                    min_vals = np.clip(min_vals, 0, 100)
                    max_vals = np.minimum(
                        nominal + np.random.uniform(0, 10, size=100), 100
                    )
                else:  # OAR
                    nominal = np.exp(-doses / 20) * 100
                    min_vals = nominal - np.random.uniform(0, 3, size=100)
                    min_vals = np.clip(min_vals, 0, 100)
                    max_vals = np.minimum(
                        nominal + np.random.uniform(0, 8, size=100), 100
                    )

                return {
                    "nominal": {"doses": doses.tolist(), "volumes": nominal.tolist()},
                    "min": {"doses": doses.tolist(), "volumes": min_vals.tolist()},
                    "max": {"doses": doses.tolist(), "volumes": max_vals.tolist()},
                }

            def get_evaluation_metrics(self):
                # Tạo các chỉ số đánh giá giả lập
                return {
                    "PTV": {
                        "D95 (Gy)": {"nominal": 63.5, "worst": 60.2},
                        "V95% (%)": {"nominal": 99.2, "worst": 96.5},
                        "HI": {"nominal": 1.05, "worst": 1.12},
                        "CI": {"nominal": 0.98, "worst": 0.92},
                    },
                    "Cord": {
                        "Dmax (Gy)": {"nominal": 42.1, "worst": 45.8},
                        "D0.1cc (Gy)": {"nominal": 40.5, "worst": 43.2},
                    },
                    "Parotid_L": {
                        "Dmean (Gy)": {"nominal": 25.3, "worst": 28.9},
                        "V30Gy (%)": {"nominal": 45.2, "worst": 51.7},
                    },
                    "Parotid_R": {
                        "Dmean (Gy)": {"nominal": 23.7, "worst": 26.1},
                        "V30Gy (%)": {"nominal": 42.6, "worst": 48.3},
                    },
                }

            def get_target_coverage_data(self):
                # Tạo dữ liệu độ phủ mục tiêu giả lập
                dose_levels = np.linspace(80, 110, 8)  # % của liều kê toa
                nominal_coverage = np.array([100, 100, 99.8, 96.5, 78.3, 45.2, 12.5, 0])
                min_coverage = np.array([100, 99.5, 97.1, 92.8, 71.6, 38.9, 8.2, 0])
                max_coverage = np.array([100, 100, 100, 99.2, 82.5, 49.6, 15.7, 1.2])

                return {
                    "dose_levels": dose_levels.tolist(),
                    "nominal": nominal_coverage.tolist(),
                    "min": min_coverage.tolist(),
                    "max": max_coverage.tolist(),
                }

        return MockResult()

    def _update_results_ui(self):
        """Cập nhật giao diện với kết quả phân tích."""
        if not self.result:
            return

        # Cập nhật widget DVH
        self.dvh_widget.set_robustness_result(self.result)

        # Cập nhật bảng chỉ số đánh giá
        self.metrics_table.set_metrics(self.result.get_evaluation_metrics())

        # Cập nhật biểu đồ độ phủ mục tiêu
        if HAS_MPL:
            self.target_figure.clear()
            ax = self.target_figure.add_subplot(111)

            target_data = self.result.get_target_coverage_data()
            dose_levels = target_data.get("dose_levels", [])
            nominal = target_data.get("nominal", [])
            min_vals = target_data.get("min", [])
            max_vals = target_data.get("max", [])

            ax.plot(dose_levels, nominal, "b-", linewidth=2, label="Danh nghĩa")
            ax.fill_between(
                dose_levels,
                min_vals,
                max_vals,
                alpha=0.3,
                color="b",
                label="Biên độ dao động",
            )

            # Thêm đường tham chiếu tại 95% độ phủ
            ax.axhline(y=95, color="r", linestyle="--", label="95% độ phủ")

            ax.set_xlabel("Liều (% liều kê toa)")
            ax.set_ylabel("Độ phủ mục tiêu (%)")
            ax.set_title("Độ phủ mục tiêu với phân tích độ bền vững")
            ax.grid(True)
            ax.legend()

            self.target_canvas.draw()

    def _export_pdf(self):
        """Xuất kết quả phân tích ra file PDF."""
        if not self.result:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có kết quả phân tích để xuất.")
            return

        try:
            # Hỏi người dùng vị trí lưu file
            export_dir = QFileDialog.getExistingDirectory(
                self, "Chọn thư mục lưu báo cáo PDF", ""
            )

            if not export_dir:
                return

            # Tạo tên file với timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if hasattr(self.plan, "id"):
                plan_id = self.plan.id
            else:
                plan_id = "plan"

            filename = os.path.join(
                export_dir, f"robustness_analysis_{plan_id}_{timestamp}.pdf"
            )

            # Tạo báo cáo PDF
            with PdfPages(filename) as pdf:
                # Trang tiêu đề
                plt.figure(figsize=(10, 12))
                plt.axis("off")
                title_text = f"Báo cáo phân tích độ bền vững\n"
                title_text += f"Kế hoạch: {plan_id}\n"
                title_text += (
                    f"Ngày: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
                )

                # Thêm thông tin thiết lập
                title_text += f"Độ không chắc chắn thiết lập (setup): {self.setup_uncertainty_spin.value()} mm\n"
                title_text += f"Độ không chắc chắn phạm vi (range): {self.range_uncertainty_spin.value()} %\n"
                title_text += f"Số lượng kịch bản: {self.num_scenarios_spin.value()}"

                plt.text(0.5, 0.5, title_text, ha="center", va="center", fontsize=14)
                pdf.savefig()
                plt.close()

                # Trang DVH cho từng cấu trúc
                for structure_name in self.result.get_structures():
                    plt.figure(figsize=(10, 8))
                    ax = plt.subplot(111)

                    dvh_data = self.result.get_structure_dvhs(structure_name)
                    if not dvh_data:
                        continue

                    # Vẽ DVH danh nghĩa và dải biên
                    doses = dvh_data.get("nominal", {}).get("doses", [])
                    volumes = dvh_data.get("nominal", {}).get("volumes", [])
                    min_volumes = dvh_data.get("min", {}).get("volumes", [])
                    max_volumes = dvh_data.get("max", {}).get("volumes", [])

                    if len(doses) > 0 and len(volumes) > 0:
                        ax.plot(
                            doses,
                            volumes,
                            "b-",
                            linewidth=2,
                            label="Kế hoạch danh nghĩa",
                        )

                        if len(min_volumes) > 0 and len(max_volumes) > 0:
                            ax.fill_between(
                                doses,
                                min_volumes,
                                max_volumes,
                                alpha=0.3,
                                color="b",
                                label="Độ dao động độ bền vững",
                            )

                    ax.set_xlabel("Liều (Gy)")
                    ax.set_ylabel("Thể tích (%)")
                    ax.set_title(f"DVH cho {structure_name} với phân tích độ bền vững")
                    ax.grid(True)
                    ax.legend()

                    pdf.savefig()
                    plt.close()

                # Trang chỉ số đánh giá
                metrics = self.result.get_evaluation_metrics()
                if metrics:
                    plt.figure(figsize=(12, len(metrics) * 1.5 + 2))
                    ax = plt.subplot(111)
                    ax.axis("off")

                    table_data = []
                    table_colors = []
                    headers = [
                        "Cấu trúc",
                        "Chỉ số",
                        "Danh nghĩa",
                        "Tồi nhất",
                        "Biên độ",
                    ]

                    for structure_name, structure_metrics in metrics.items():
                        for metric_name, metric_values in structure_metrics.items():
                            nominal = metric_values.get("nominal", "N/A")
                            worst = metric_values.get("worst", "N/A")

                            if isinstance(nominal, (int, float)) and isinstance(
                                worst, (int, float)
                            ):
                                amplitude = abs(worst - nominal)
                                row = [
                                    structure_name,
                                    metric_name,
                                    f"{nominal:.2f}",
                                    f"{worst:.2f}",
                                    f"{amplitude:.2f}",
                                ]

                                # Màu sắc dựa trên biên độ
                                if amplitude > 5.0:
                                    row_color = [
                                        "white",
                                        "white",
                                        "white",
                                        "white",
                                        (1.0, 0.8, 0.8),
                                    ]  # Đỏ nhạt
                                elif amplitude > 2.0:
                                    row_color = [
                                        "white",
                                        "white",
                                        "white",
                                        "white",
                                        (1.0, 1.0, 0.8),
                                    ]  # Vàng nhạt
                                else:
                                    row_color = [
                                        "white",
                                        "white",
                                        "white",
                                        "white",
                                        (0.8, 1.0, 0.8),
                                    ]  # Xanh nhạt
                            else:
                                row = [
                                    structure_name,
                                    metric_name,
                                    str(nominal),
                                    str(worst),
                                    "N/A",
                                ]
                                row_color = ["white"] * 5

                            table_data.append(row)
                            table_colors.append(row_color)

                    table = ax.table(
                        cellText=table_data,
                        cellColours=table_colors,
                        colLabels=headers,
                        loc="center",
                        cellLoc="center",
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)

                    plt.title("Chỉ số đánh giá độ bền vững")
                    pdf.savefig()
                    plt.close()

                # Trang phân tích độ phủ mục tiêu
                target_data = self.result.get_target_coverage_data()
                if target_data:
                    plt.figure(figsize=(10, 8))
                    ax = plt.subplot(111)

                    dose_levels = target_data.get("dose_levels", [])
                    nominal = target_data.get("nominal", [])
                    min_vals = target_data.get("min", [])
                    max_vals = target_data.get("max", [])

                    if len(dose_levels) > 0 and len(nominal) > 0:
                        ax.plot(
                            dose_levels, nominal, "b-", linewidth=2, label="Danh nghĩa"
                        )

                        if len(min_vals) > 0 and len(max_vals) > 0:
                            ax.fill_between(
                                dose_levels,
                                min_vals,
                                max_vals,
                                alpha=0.3,
                                color="b",
                                label="Biên độ dao động",
                            )

                        # Thêm đường tham chiếu tại 95% độ phủ
                        ax.axhline(y=95, color="r", linestyle="--", label="95% độ phủ")

                    ax.set_xlabel("Liều (% liều kê toa)")
                    ax.set_ylabel("Độ phủ mục tiêu (%)")
                    ax.set_title("Độ phủ mục tiêu với phân tích độ bền vững")
                    ax.grid(True)
                    ax.legend()

                    pdf.savefig()
                    plt.close()

            QMessageBox.information(
                self,
                "Xuất PDF thành công",
                f"Báo cáo phân tích độ bền vững đã được lưu tại:\n{filename}",
            )

        except Exception as e:
            logger.error(f"Lỗi khi xuất PDF: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Lỗi xuất PDF",
                f"Không thể xuất báo cáo PDF: {str(e)}",
            )

    def _export_csv(self):
        """Xuất kết quả phân tích ra file CSV."""
        if not self.result:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có kết quả phân tích để xuất.")
            return

        try:
            import csv
            import os
            from datetime import datetime

            # Hỏi người dùng vị trí lưu file
            export_dir = QFileDialog.getExistingDirectory(
                self, "Chọn thư mục lưu báo cáo CSV", ""
            )

            if not export_dir:
                return

            # Tạo tên file với timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if hasattr(self.plan, "id"):
                plan_id = self.plan.id
            else:
                plan_id = "plan"

            # File cho chỉ số đánh giá
            metrics_filename = os.path.join(
                export_dir, f"robustness_metrics_{plan_id}_{timestamp}.csv"
            )

            # Xuất các chỉ số đánh giá
            metrics = self.result.get_evaluation_metrics()
            if metrics:
                with open(metrics_filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["Cấu trúc", "Chỉ số", "Danh nghĩa", "Tồi nhất", "Biên độ"]
                    )

                    for structure_name, structure_metrics in metrics.items():
                        for metric_name, metric_values in structure_metrics.items():
                            nominal = metric_values.get("nominal", "N/A")
                            worst = metric_values.get("worst", "N/A")

                            if isinstance(nominal, (int, float)) and isinstance(
                                worst, (int, float)
                            ):
                                amplitude = abs(worst - nominal)
                                row = [
                                    structure_name,
                                    metric_name,
                                    f"{nominal:.2f}",
                                    f"{worst:.2f}",
                                    f"{amplitude:.2f}",
                                ]
                            else:
                                row = [
                                    structure_name,
                                    metric_name,
                                    str(nominal),
                                    str(worst),
                                    "N/A",
                                ]

                            writer.writerow(row)

            # File cho dữ liệu DVH
            for structure_name in self.result.get_structures():
                dvh_filename = os.path.join(
                    export_dir,
                    f"robustness_dvh_{structure_name}_{plan_id}_{timestamp}.csv",
                )

                dvh_data = self.result.get_structure_dvhs(structure_name)
                if not dvh_data:
                    continue

                doses = dvh_data.get("nominal", {}).get("doses", [])
                volumes_nominal = dvh_data.get("nominal", {}).get("volumes", [])
                volumes_min = dvh_data.get("min", {}).get("volumes", [])
                volumes_max = dvh_data.get("max", {}).get("volumes", [])

                if len(doses) > 0 and len(volumes_nominal) > 0:
                    with open(dvh_filename, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "Liều (Gy)",
                                "Thể tích (%) - Danh nghĩa",
                                "Thể tích (%) - Tối thiểu",
                                "Thể tích (%) - Tối đa",
                            ]
                        )

                        for i in range(len(doses)):
                            row = [doses[i], volumes_nominal[i]]

                            if i < len(volumes_min):
                                row.append(volumes_min[i])
                            else:
                                row.append("")

                            if i < len(volumes_max):
                                row.append(volumes_max[i])
                            else:
                                row.append("")

                            writer.writerow(row)

            # File cho dữ liệu độ phủ mục tiêu
            target_filename = os.path.join(
                export_dir, f"robustness_target_coverage_{plan_id}_{timestamp}.csv"
            )
            target_data = self.result.get_target_coverage_data()

            if target_data:
                dose_levels = target_data.get("dose_levels", [])
                nominal = target_data.get("nominal", [])
                min_vals = target_data.get("min", [])
                max_vals = target_data.get("max", [])

                if len(dose_levels) > 0 and len(nominal) > 0:
                    with open(target_filename, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "Liều (%)",
                                "Độ phủ (%) - Danh nghĩa",
                                "Độ phủ (%) - Tối thiểu",
                                "Độ phủ (%) - Tối đa",
                            ]
                        )

                        for i in range(len(dose_levels)):
                            row = [dose_levels[i], nominal[i]]

                            if i < len(min_vals):
                                row.append(min_vals[i])
                            else:
                                row.append("")

                            if i < len(max_vals):
                                row.append(max_vals[i])
                            else:
                                row.append("")

                            writer.writerow(row)

            QMessageBox.information(
                self,
                "Xuất CSV thành công",
                f"Dữ liệu phân tích độ bền vững đã được xuất ra các file CSV trong thư mục:\n{export_dir}",
            )

        except Exception as e:
            logger.error(f"Lỗi khi xuất CSV: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Lỗi xuất CSV",
                f"Không thể xuất dữ liệu CSV: {str(e)}",
            )


# Hàm tiện ích để tạo dialog
def create_robustness_analysis_dialog(plan=None, parent=None):
    """Tạo và trả về dialog phân tích độ bền vững."""
    try:
        return RobustnessAnalysisDialog(plan, parent)
    except Exception as e:
        logger.error(f"Lỗi khi tạo dialog phân tích độ bền vững: {str(e)}")
        if parent:
            QMessageBox.critical(
                parent, "Lỗi", f"Không thể tạo dialog phân tích độ bền vững: {str(e)}"
            )
        return None
