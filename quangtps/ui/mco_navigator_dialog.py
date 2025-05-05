#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module giao diện người dùng cho Multi-Criteria Optimization (MCO) Navigator.

Module này triển khai giao diện người dùng cho công cụ MCO Navigator,
mô phỏng theo giao diện MCO của Eclipse. Cho phép người dùng khám phá
không gian lời giải Pareto và tương tác trực quan với các lời giải.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Union

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
    QDialogButtonBox,
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
    QTableWidget,
    QTableWidgetItem,
    QApplication,
    QSizePolicy,
    QGridLayout,
)
from PyQt5.QtGui import QColor, QIcon, QPixmap, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRectF, QTimer

# Import matplotlib for visualization
try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available, using simplified visualization")

# Import from QuangTPS modules
from quangtps.core.services import ServiceRegistry
from quangtps.planning.plan import Plan
from quangtps.optimization.mco.mco_interface import (
    MCOEngine,
    MCOSolution,
    MCOObjectiveSpace,
    MCONavigator,
    calculate_mco_metrics,
)
from quangtps.evaluation.dvh.dvh_calculation import DVHCalculator
from quangtps.evaluation.dvh.dvh_visualization import plot_dvh

logger = logging.getLogger(__name__)


class ObjectiveSlider(QWidget):
    """
    Widget slider cho điều chỉnh giá trị của một hàm mục tiêu.
    """

    valueChanged = pyqtSignal(str, float)

    def __init__(
        self, objective_name, min_value, max_value, current_value, parent=None
    ):
        super().__init__(parent)

        self.objective_name = objective_name
        self.min_value = min_value
        self.max_value = max_value

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title
        self.title_label = QLabel(objective_name)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.title_label)

        # Slider layout
        slider_layout = QHBoxLayout()

        # Min label
        self.min_label = QLabel(f"{min_value:.1f}")
        slider_layout.addWidget(self.min_label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(self._value_to_slider(current_value))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        slider_layout.addWidget(self.slider)

        # Max label
        self.max_label = QLabel(f"{max_value:.1f}")
        slider_layout.addWidget(self.max_label)

        layout.addLayout(slider_layout)

        # Value display
        self.value_label = QLabel(f"Current: {current_value:.2f}")
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)

        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_value_changed)

    def _value_to_slider(self, value):
        """Chuyển đổi giá trị thực tế sang giá trị trên thanh trượt"""
        # Normalize to 0-100 range
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        return int(normalized * 100)

    def _slider_to_value(self, slider_value):
        """Chuyển đổi giá trị thanh trượt sang giá trị thực tế"""
        # Convert from 0-100 range to actual value
        normalized = slider_value / 100
        return self.min_value + normalized * (self.max_value - self.min_value)

    def _on_slider_value_changed(self, slider_value):
        value = self._slider_to_value(slider_value)
        self.value_label.setText(f"Current: {value:.2f}")
        self.valueChanged.emit(self.objective_name, value)

    def set_value(self, value):
        """Thiết lập giá trị cho thanh trượt"""
        self.slider.setValue(self._value_to_slider(value))

    def get_value(self):
        """Lấy giá trị hiện tại của thanh trượt"""
        return self._slider_to_value(self.slider.value())


class DVHComparisonWidget(QWidget):
    """
    Widget so sánh DVH giữa các lời giải.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure
            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_xlabel("Dose (Gy)")
            self.ax.set_ylabel("Volume (%)")
            self.ax.set_title("DVH Comparison")
            self.ax.grid(True)

            layout.addWidget(self.canvas)
        else:
            layout.addWidget(
                QLabel("Matplotlib is not available. Cannot display DVH comparison.")
            )

        # Controls
        controls_layout = QHBoxLayout()

        self.structure_combo = QComboBox()
        self.structure_combo.setMinimumWidth(150)

        self.reference_combo = QComboBox()
        self.reference_combo.addItem("No Reference")
        self.reference_combo.setMinimumWidth(150)

        controls_layout.addWidget(QLabel("Structure:"))
        controls_layout.addWidget(self.structure_combo)
        controls_layout.addWidget(QLabel("Reference:"))
        controls_layout.addWidget(self.reference_combo)

        layout.addLayout(controls_layout)

        # Connect signals
        if MATPLOTLIB_AVAILABLE:
            self.structure_combo.currentIndexChanged.connect(self.update_plot)
            self.reference_combo.currentIndexChanged.connect(self.update_plot)

    def clear(self):
        """Clear the plot"""
        if MATPLOTLIB_AVAILABLE:
            self.ax.clear()
            self.ax.set_xlabel("Dose (Gy)")
            self.ax.set_ylabel("Volume (%)")
            self.ax.set_title("DVH Comparison")
            self.ax.grid(True)
            self.canvas.draw()

    def update_structures(self, structures):
        """Update the structure list"""
        self.structure_combo.clear()
        for structure in structures:
            self.structure_combo.addItem(structure.name)

    def update_references(self, references):
        """Update reference plan list"""
        current_text = self.reference_combo.currentText()

        self.reference_combo.clear()
        self.reference_combo.addItem("No Reference")

        for ref in references:
            self.reference_combo.addItem(ref)

        # Try to restore the previous selection
        index = self.reference_combo.findText(current_text)
        if index >= 0:
            self.reference_combo.setCurrentIndex(index)

    def update_plot(self):
        """Update the DVH plot"""
        if not MATPLOTLIB_AVAILABLE:
            return

        # This would be implemented to update the DVH plot
        # based on the current solution and reference
        pass

    def plot_dvh_comparison(
        self, current_solution, reference_solution=None, structure_name=None
    ):
        """Plot DVH comparison between solutions"""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Clear the plot
        self.ax.clear()

        # If no structure specified, use the currently selected one
        if structure_name is None:
            if self.structure_combo.count() > 0:
                structure_name = self.structure_combo.currentText()
            else:
                return

        # Find the structure
        current_plan = current_solution.plan
        structure = None
        for s in current_plan.get_structures():
            if s.name == structure_name:
                structure = s
                break

        if structure is None:
            return

        # Plot current solution DVH
        try:
            dvh_data = DVHCalculator.calculate_dvh(current_plan, structure)
            self.ax.plot(
                dvh_data.doses, dvh_data.volumes, "b-", label=f"Current Solution"
            )

            # Plot reference DVH if available
            if reference_solution:
                ref_plan = reference_solution.plan
                ref_dvh_data = DVHCalculator.calculate_dvh(ref_plan, structure)
                self.ax.plot(
                    ref_dvh_data.doses, ref_dvh_data.volumes, "r--", label=f"Reference"
                )

            self.ax.set_xlabel("Dose (Gy)")
            self.ax.set_ylabel("Volume (%)")
            self.ax.set_title(f"DVH Comparison - {structure_name}")
            self.ax.grid(True)
            self.ax.legend()

            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error plotting DVH comparison: {e}")


class ParetoFrontWidget(QWidget):
    """
    Widget hiển thị không gian Pareto và cho phép người dùng chọn lời giải.
    """

    solutionSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure
            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_xlabel("Objective 1")
            self.ax.set_ylabel("Objective 2")
            self.ax.set_title("Pareto Front")
            self.ax.grid(True)

            layout.addWidget(self.canvas)

            # Connect click event
            self.canvas.mpl_connect("button_press_event", self._on_click)
        else:
            layout.addWidget(
                QLabel("Matplotlib is not available. Cannot display Pareto front.")
            )

        # Controls
        controls_layout = QHBoxLayout()

        self.x_axis_combo = QComboBox()
        self.y_axis_combo = QComboBox()

        controls_layout.addWidget(QLabel("X Axis:"))
        controls_layout.addWidget(self.x_axis_combo)
        controls_layout.addWidget(QLabel("Y Axis:"))
        controls_layout.addWidget(self.y_axis_combo)

        layout.addLayout(controls_layout)

        # Connect signals
        if MATPLOTLIB_AVAILABLE:
            self.x_axis_combo.currentIndexChanged.connect(self.update_plot)
            self.y_axis_combo.currentIndexChanged.connect(self.update_plot)

        self.objective_space = None
        self.points = []  # Store the points for mouse click detection

    def set_objective_space(self, objective_space):
        """Set the objective space"""
        self.objective_space = objective_space

        # Update objective lists
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()

        objective_names = list(objective_space.objectives.keys())

        for name in objective_names:
            self.x_axis_combo.addItem(name)
            self.y_axis_combo.addItem(name)

        # Set default selections if possible
        if len(objective_names) >= 2:
            self.x_axis_combo.setCurrentIndex(0)
            self.y_axis_combo.setCurrentIndex(1)

        self.update_plot()

    def update_plot(self):
        """Update the Pareto front plot"""
        if not MATPLOTLIB_AVAILABLE or self.objective_space is None:
            return

        if self.x_axis_combo.count() == 0 or self.y_axis_combo.count() == 0:
            return

        x_objective = self.x_axis_combo.currentText()
        y_objective = self.y_axis_combo.currentText()

        # Use the objective space's plotting method
        self.ax.clear()
        self.objective_space.plot_pareto_front(x_objective, y_objective, self.ax)

        self.canvas.draw()

    def _on_click(self, event):
        """Handle mouse click events on the plot"""
        if not self.objective_space:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Find the closest solution point
        closest_index = None
        min_distance = float("inf")

        x_objective = self.x_axis_combo.currentText()
        y_objective = self.y_axis_combo.currentText()

        for i, solution in enumerate(self.objective_space.solutions):
            x = solution.get_objective_value(x_objective)
            y = solution.get_objective_value(y_objective)

            # Calculate distance in data coordinates
            distance = np.sqrt((x - event.xdata) ** 2 + (y - event.ydata) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_index = i

        if closest_index is not None:
            self.solutionSelected.emit(closest_index)
            self.update_plot()  # Refresh to show the new current solution


class SolutionMetricsWidget(QWidget):
    """
    Widget hiển thị các chỉ số của lời giải hiện tại.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )

        layout.addWidget(self.metrics_table)

    def update_metrics(self, solution):
        """Update the metrics display for a solution"""
        self.metrics_table.setRowCount(0)

        if solution is None:
            return

        # Calculate metrics
        metrics = calculate_mco_metrics(solution)

        # Add objective values
        for name, value in solution.objectives.items():
            self.add_metric(name, value)

        # Add calculated metrics
        for name, value in metrics.items():
            if name == "CI":
                self.add_metric("Conformity Index", value, "")
            elif name == "HI":
                self.add_metric("Homogeneity Index", value, "")
            elif name == "GI":
                self.add_metric("Gradient Index", value, "")

    def add_metric(self, name, value, unit=""):
        """Add a metric to the table"""
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)

        self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{value:.3f}"))
        self.metrics_table.setItem(row, 2, QTableWidgetItem(unit))


class SolutionComparisonWidget(QWidget):
    """
    Widget so sánh hai hoặc nhiều lời giải MCO.

    Cho phép người dùng so sánh các chỉ số DVH, thống kê liều và các tiêu chí lâm sàng
    giữa các kế hoạch trên mặt Pareto.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.solutions = []
        self.current_solution_index = -1

        self._setup_ui()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng"""
        main_layout = QVBoxLayout(self)

        # Toolbar cho các tùy chọn so sánh
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))

        self.add_solution_action = QAction("Add Current Solution", self)
        self.add_solution_action.triggered.connect(self._add_current_solution)
        toolbar.addAction(self.add_solution_action)

        self.clear_action = QAction("Clear All", self)
        self.clear_action.triggered.connect(self._clear_solutions)
        toolbar.addAction(self.clear_action)

        main_layout.addWidget(toolbar)

        # Bảng so sánh
        self.comparison_table = QTableWidget(0, 0)
        self.comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.comparison_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

        main_layout.addWidget(self.comparison_table, 1)

        # DVH comparison plot
        self.dvh_widget = DVHComparisonWidget()
        main_layout.addWidget(self.dvh_widget, 1)

    def set_current_solution(self, solution, index):
        """Thiết lập lời giải hiện tại"""
        self.current_solution = solution
        self.current_solution_index = index

        # Cập nhật DVH nếu có thể
        self._update_dvh()

    def _add_current_solution(self):
        """Thêm lời giải hiện tại vào so sánh"""
        if not hasattr(self, "current_solution") or self.current_solution is None:
            return

        # Kiểm tra xem lời giải đã có trong danh sách chưa
        for sol in self.solutions:
            if sol.solution_id == self.current_solution.solution_id:
                QMessageBox.information(
                    self, "Already Added", "This solution is already in the comparison."
                )
                return

        # Thêm lời giải vào danh sách
        self.solutions.append(self.current_solution)

        # Cập nhật bảng so sánh
        self._update_comparison_table()

        # Cập nhật DVH
        self._update_dvh()

        # Cập nhật danh sách tham chiếu
        self.dvh_widget.update_references(
            [f"Solution {i + 1}" for i in range(len(self.solutions))]
        )

    def _clear_solutions(self):
        """Xóa tất cả các lời giải khỏi so sánh"""
        self.solutions = []
        self._update_comparison_table()
        self._update_dvh()
        self.dvh_widget.update_references([])

    def _update_comparison_table(self):
        """Cập nhật bảng so sánh với dữ liệu mới nhất"""
        self.comparison_table.clear()

        if not self.solutions:
            self.comparison_table.setRowCount(0)
            self.comparison_table.setColumnCount(0)
            return

        # Tạo cột cho mỗi lời giải
        self.comparison_table.setColumnCount(len(self.solutions) + 1)

        headers = ["Metric"]
        for i in range(len(self.solutions)):
            headers.append(f"Solution {i + 1}")

        self.comparison_table.setHorizontalHeaderLabels(headers)

        # Thu thập các thông số cho so sánh
        structures = {}
        metrics = set()

        for solution in self.solutions:
            plan = solution.plan

            # Thu thập các cấu trúc và chỉ số
            for structure in plan.structure_set.structures:
                if structure.id not in structures:
                    structures[structure.id] = structure.name

                # Thêm các chỉ số cho cấu trúc này
                metrics.add(f"{structure.name} - Min Dose")
                metrics.add(f"{structure.name} - Max Dose")
                metrics.add(f"{structure.name} - Mean Dose")

                # Các chỉ số DVH phổ biến
                if "PTV" in structure.name.upper():
                    metrics.add(f"{structure.name} - D95%")
                    metrics.add(f"{structure.name} - D98%")
                    metrics.add(f"{structure.name} - V95%")
                elif "OAR" in structure.name.upper() or any(
                    oar in structure.name.upper()
                    for oar in ["CORD", "HEART", "LUNG", "KIDNEY", "LIVER"]
                ):
                    metrics.add(f"{structure.name} - V20Gy")
                    metrics.add(f"{structure.name} - V30Gy")
                    metrics.add(f"{structure.name} - Mean Dose")

        # Thêm chỉ số tổng quát
        metrics.add("Conformity Index")
        metrics.add("Homogeneity Index")
        metrics.add("Total MU")

        # Thêm hàng cho mỗi thông số
        metrics = sorted(metrics)
        self.comparison_table.setRowCount(len(metrics))

        for i, metric in enumerate(metrics):
            # Thêm tên chỉ số
            self.comparison_table.setItem(i, 0, QTableWidgetItem(metric))

            # Thêm giá trị cho mỗi lời giải
            for j, solution in enumerate(self.solutions):
                plan = solution.plan

                # Tính giá trị chỉ số
                value = self._calculate_metric_value(metric, plan)

                # Tạo item với giá trị đã định dạng
                item = QTableWidgetItem(f"{value:.2f}")

                # Set alignment to center
                item.setTextAlignment(Qt.AlignCenter)

                self.comparison_table.setItem(i, j + 1, item)

    def _calculate_metric_value(self, metric_name, plan):
        """Tính giá trị của một chỉ số cho một kế hoạch"""
        try:
            # Phân tích tên chỉ số
            if " - " in metric_name:
                structure_name, metric_type = metric_name.split(" - ", 1)

                # Tìm cấu trúc
                structure = None
                for s in plan.structure_set.structures:
                    if s.name == structure_name:
                        structure = s
                        break

                if structure is None:
                    return 0.0

                # DVH Calculator
                from quangtps.evaluation.dvh.dvh_calculation import (
                    calculate_dvh_metrics,
                )

                # Tính các thông số DVH
                metrics = calculate_dvh_metrics(structure, plan.dose)

                # Trả về giá trị thích hợp
                if metric_type == "Min Dose":
                    return metrics.get("min_dose", 0.0)
                elif metric_type == "Max Dose":
                    return metrics.get("max_dose", 0.0)
                elif metric_type == "Mean Dose":
                    return metrics.get("mean_dose", 0.0)
                elif metric_type == "D95%":
                    return metrics.get("D95", 0.0)
                elif metric_type == "D98%":
                    return metrics.get("D98", 0.0)
                elif metric_type == "V95%":
                    return metrics.get("V95", 0.0)
                elif metric_type == "V20Gy":
                    return metrics.get("V20", 0.0)
                elif metric_type == "V30Gy":
                    return metrics.get("V30", 0.0)
                else:
                    return 0.0
            elif metric_name == "Conformity Index":
                # Tính CI (đơn giản hóa)
                return 1.0  # Placeholder
            elif metric_name == "Homogeneity Index":
                # Tính HI (đơn giản hóa)
                return 1.0  # Placeholder
            elif metric_name == "Total MU":
                # Lấy tổng MU
                return plan.total_mu if hasattr(plan, "total_mu") else 0.0
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error calculating metric {metric_name}: {str(e)}")
            return 0.0

    def _update_dvh(self):
        """Cập nhật đồ thị DVH từ các lời giải đã chọn"""
        # Cập nhật danh sách cấu trúc
        if self.solutions:
            plan = self.solutions[0].plan
            self.dvh_widget.update_structures(plan.structure_set.structures)

            # Vẽ DVH cho tất cả các lời giải
            if hasattr(self, "current_solution") and self.current_solution is not None:
                self.dvh_widget.plot_dvh_comparison(self.current_solution)


class MCONavigatorDialog(QDialog):
    """
    Hộp thoại chính cho Multi-Criteria Optimization Navigator.

    Cung cấp giao diện người dùng tương tác để khám phá không gian lời giải Pareto,
    so sánh các lời giải, và chọn lời giải tối ưu dựa trên ưu tiên lâm sàng.
    """

    solutionAccepted = pyqtSignal(MCOSolution)

    def __init__(self, plan, parent=None):
        super().__init__(parent)

        self.plan = plan
        self.setWindowTitle("Multi-Criteria Optimization Navigator")
        self.resize(1200, 800)

        # MCO Navigator
        self.mco_navigator = None
        self.current_solution = None
        self.current_solution_index = -1

        # History for undo/redo
        self.history = []
        self.history_index = -1

        # Setup UI
        self._setup_ui()

        # Initialize MCO
        self._initialize_mco()

        # Fetch solutions
        self._initialize_solutions()

        # Setup objective sliders
        self._setup_objectives()

        # Update UI
        self._update_ui_for_current_solution()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng"""
        main_layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))

        # Action: Generate
        self.generate_action = QAction("Generate Solutions", self)
        self.generate_action.setToolTip("Generate Pareto surface solutions")
        self.generate_action.triggered.connect(self._generate_solutions)
        toolbar.addAction(self.generate_action)

        toolbar.addSeparator()

        # Action: Undo
        self.undo_action = QAction("Undo", self)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self._undo)
        toolbar.addAction(self.undo_action)

        # Action: Redo
        self.redo_action = QAction("Redo", self)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self._redo)
        toolbar.addAction(self.redo_action)

        toolbar.addSeparator()

        # Action: Save
        self.save_action = QAction("Save Current", self)
        self.save_action.setToolTip("Save current solution")
        self.save_action.triggered.connect(self._save_current_solution)
        toolbar.addAction(self.save_action)

        main_layout.addWidget(toolbar)

        # Main content splitter
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setHandleWidth(2)

        # Left panel: Controls & Solutions
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Tabs for different controls
        control_tabs = QTabWidget()

        # Tab 1: Objectives tab
        objectives_widget = QWidget()
        objectives_layout = QVBoxLayout(objectives_widget)
        objectives_layout.setContentsMargins(10, 10, 10, 10)

        # Intro text
        objectives_layout.addWidget(
            QLabel("<b>Adjust objective weights to navigate the solution space:</b>")
        )

        # Scrollable area for objectives
        objectives_scroll = QScrollArea()
        objectives_scroll.setWidgetResizable(True)

        self.objectives_content = QWidget()
        self.objectives_layout = QVBoxLayout(self.objectives_content)
        objectives_scroll.setWidget(self.objectives_content)

        objectives_layout.addWidget(objectives_scroll)

        # Navigate button
        self.navigate_button = QPushButton("Navigate to Weights")
        self.navigate_button.clicked.connect(self._navigate_to_weights)
        objectives_layout.addWidget(self.navigate_button)

        control_tabs.addTab(objectives_widget, "Objectives")

        # Tab 2: Navigation tab
        solutions_widget = QWidget()
        solutions_layout = QVBoxLayout(solutions_widget)

        self.pareto_front_widget = ParetoFrontWidget()
        self.pareto_front_widget.solutionSelected.connect(self._on_solution_selected)
        solutions_layout.addWidget(self.pareto_front_widget)

        # List of solutions
        solutions_layout.addWidget(QLabel("<b>Available Solutions:</b>"))

        self.solution_list = QListWidget()
        self.solution_list.currentRowChanged.connect(self._on_solution_selected)
        solutions_layout.addWidget(self.solution_list)

        control_tabs.addTab(solutions_widget, "Navigation")

        # Tab 3: Comparison
        self.comparison_widget = SolutionComparisonWidget()
        control_tabs.addTab(self.comparison_widget, "Comparison")

        left_layout.addWidget(control_tabs, 1)

        # Add the left panel to the splitter
        content_splitter.addWidget(left_panel)

        # Right panel: Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Tabs for visualization
        viz_tabs = QTabWidget()

        # Tab 1: DVH
        self.dvh_widget = DVHComparisonWidget()
        viz_tabs.addTab(self.dvh_widget, "DVH")

        # Tab 2: Metrics
        self.metrics_widget = SolutionMetricsWidget()
        viz_tabs.addTab(self.metrics_widget, "Metrics")

        right_layout.addWidget(viz_tabs)

        # Add right panel to splitter
        content_splitter.addWidget(right_panel)

        # Set initial sizes
        content_splitter.setSizes([300, 700])

        main_layout.addWidget(content_splitter, 1)

        # Bottom row with button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(button_box)

    def _initialize_mco(self):
        """Khởi tạo MCO Navigator"""
        try:
            # Tạo MCO Navigator
            from quangtps.optimization.mco.mco_interface import create_mco_navigator

            self.mco_navigator = create_mco_navigator(self.plan)

            if self.mco_navigator:
                logger.info("MCO Navigator initialized successfully")
            else:
                logger.error("Failed to create MCO Navigator")
                QMessageBox.critical(
                    self, "Error", "Failed to initialize MCO Navigator"
                )
        except Exception as e:
            logger.error(f"Error initializing MCO Navigator: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to initialize MCO Navigator: {str(e)}"
            )

    def _initialize_solutions(self):
        """Khởi tạo các lời giải có sẵn"""
        if not self.mco_navigator:
            return

        # Get initial solutions
        solutions = self.mco_navigator.get_solutions()

        # Populate solution list
        for i, (solution_id, solution) in enumerate(solutions.items()):
            self.solution_list.addItem(f"Solution {i + 1}: {solution_id}")

        # Select first solution if available
        if self.solution_list.count() > 0:
            self.solution_list.setCurrentRow(0)

        # Update Pareto front visualization
        objective_space = self.mco_navigator.get_objective_space()
        self.pareto_front_widget.set_objective_space(objective_space)

    def _setup_objectives(self):
        """Thiết lập các hàm mục tiêu"""
        if not self.mco_navigator:
            return

        # Clear existing content
        while self.objectives_layout.count():
            item = self.objectives_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get objectives
        objectives = self.mco_navigator.get_objectives()

        if not objectives:
            self.objectives_layout.addWidget(QLabel("No objectives available"))
            return

        # Create sliders for each objective
        self._create_objective_sliders()

    def _create_objective_sliders(self):
        """Tạo các thanh trượt cho từng hàm mục tiêu"""
        if not self.mco_navigator:
            return

        # Get trade-off ranges
        trade_off_ranges = self.mco_navigator.get_trade_off_ranges()

        # Container for sliders
        self.objective_sliders = {}

        # Objectives by category
        targets = []
        oars = []
        others = []

        for objective_id, objective in self.mco_navigator.get_objectives().items():
            # Skip inactive objectives
            if not objective.is_active:
                continue

            # Determine category
            if (
                "PTV" in objective.structure_name.upper()
                or "CTV" in objective.structure_name.upper()
            ):
                targets.append(objective)
            elif "OAR" in objective.structure_name.upper() or any(
                organ in objective.structure_name.upper()
                for organ in [
                    "CORD",
                    "HEART",
                    "LUNG",
                    "LIVER",
                    "KIDNEY",
                    "BLADDER",
                    "RECTUM",
                ]
            ):
                oars.append(objective)
            else:
                others.append(objective)

        # Add section for targets
        if targets:
            target_group = QGroupBox("Target Structures")
            target_layout = QVBoxLayout(target_group)

            for objective in targets:
                min_val, max_val = trade_off_ranges.get(objective.objective_id, (0, 1))
                current_val = objective.value

                # Create slider
                slider = ObjectiveSlider(
                    f"{objective.structure_name}: {objective.objective_type}",
                    min_val,
                    max_val,
                    current_val,
                )
                slider.valueChanged.connect(self._on_slider_value_changed)

                target_layout.addWidget(slider)
                self.objective_sliders[objective.objective_id] = slider

            self.objectives_layout.addWidget(target_group)

        # Add section for OARs
        if oars:
            oar_group = QGroupBox("Organs at Risk")
            oar_layout = QVBoxLayout(oar_group)

            for objective in oars:
                min_val, max_val = trade_off_ranges.get(objective.objective_id, (0, 1))
                current_val = objective.value

                # Create slider
                slider = ObjectiveSlider(
                    f"{objective.structure_name}: {objective.objective_type}",
                    min_val,
                    max_val,
                    current_val,
                )
                slider.valueChanged.connect(self._on_slider_value_changed)

                oar_layout.addWidget(slider)
                self.objective_sliders[objective.objective_id] = slider

            self.objectives_layout.addWidget(oar_group)

        # Add section for other objectives
        if others:
            other_group = QGroupBox("Other Structures")
            other_layout = QVBoxLayout(other_group)

            for objective in others:
                min_val, max_val = trade_off_ranges.get(objective.objective_id, (0, 1))
                current_val = objective.value

                # Create slider
                slider = ObjectiveSlider(
                    f"{objective.structure_name}: {objective.objective_type}",
                    min_val,
                    max_val,
                    current_val,
                )
                slider.valueChanged.connect(self._on_slider_value_changed)

                other_layout.addWidget(slider)
                self.objective_sliders[objective.objective_id] = slider

            self.objectives_layout.addWidget(other_group)

        # Add stretch to push all widgets to the top
        self.objectives_layout.addStretch(1)

    def _update_ui_for_current_solution(self):
        """Cập nhật giao diện với lời giải hiện tại"""
        if not self.current_solution:
            return

        # Update sliders to reflect current solution values
        objectives = self.mco_navigator.get_objectives()

        for objective_id, objective in objectives.items():
            if objective_id in self.objective_sliders:
                value = self.current_solution.objective_values.get(
                    objective_id, objective.value
                )
                self.objective_sliders[objective_id].set_value(value)

        # Update metrics widget
        self.metrics_widget.update_metrics(self.current_solution)

        # Update DVH widget
        self.dvh_widget.plot_dvh_comparison(self.current_solution)

        # Update comparison widget
        self.comparison_widget.set_current_solution(
            self.current_solution, self.current_solution_index
        )

        # Update history controls
        self.undo_action.setEnabled(self.history_index > 0)
        self.redo_action.setEnabled(self.history_index < len(self.history) - 1)

    def _generate_solutions(self):
        """Sinh các lời giải trên mặt Pareto"""
        if not self.mco_navigator:
            return

        try:
            # Show progress dialog
            progress = QProgressDialog(
                "Generating Pareto solutions...", "Cancel", 0, 100, self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()

            def update_progress(value, message):
                progress.setValue(int(value * 100))
                progress.setLabelText(message)
                QApplication.processEvents()
                return not progress.wasCanceled()

            # Generate solutions
            self.mco_navigator.generate_pareto_solutions(10, update_progress)

            # Update UI
            self._initialize_solutions()

            # Update sliders
            self._setup_objectives()

            # Close progress dialog
            progress.close()

            QMessageBox.information(
                self, "Success", "Pareto solutions generated successfully"
            )

        except Exception as e:
            logger.error(f"Error generating solutions: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to generate solutions: {str(e)}"
            )

    def _on_solution_selected(self, index):
        """Handle selection of a solution"""
        if index < 0:
            return

        solutions = list(self.mco_navigator.get_solutions().values())
        if index >= len(solutions):
            return

        self.current_solution = solutions[index]
        self.current_solution_index = index

        # Add to history
        if (self.history_index < 0) or (
            self.history_index < len(self.history)
            and self.history[self.history_index] != self.current_solution
        ):
            # Truncate history if we're not at the end
            if self.history_index < len(self.history) - 1:
                self.history = self.history[: self.history_index + 1]

            self.history.append(self.current_solution)
            self.history_index = len(self.history) - 1

        # Update UI
        self._update_ui_for_current_solution()

    def _on_slider_value_changed(self, objective_name, target_value):
        """Handle slider value change"""
        # Find objective ID from name
        objective_id = None

        for obj_id, obj in self.mco_navigator.get_objectives().items():
            if f"{obj.structure_name}: {obj.objective_type}" == objective_name:
                objective_id = obj_id
                break

        if not objective_id:
            return

        # Update slider weights
        weights = {}

        for obj_id, slider in self.objective_sliders.items():
            weights[obj_id] = slider.get_value()

        # Navigate to new solution based on weights
        try:
            solution = self.mco_navigator.navigate_to_values(weights)

            if solution:
                # Add it to the solution list if not there yet
                solutions = list(self.mco_navigator.get_solutions().values())
                if solution not in solutions:
                    solutions.append(solution)
                    self.solution_list.addItem(
                        f"Solution {len(solutions)}: {solution.solution_id}"
                    )

                # Update current solution
                self.current_solution = solution
                self.current_solution_index = solutions.index(solution)

                # Select in the list
                self.solution_list.setCurrentRow(self.current_solution_index)

                # Add to history
                if (self.history_index < 0) or (
                    self.history_index < len(self.history)
                    and self.history[self.history_index] != self.current_solution
                ):
                    # Truncate history if we're not at the end
                    if self.history_index < len(self.history) - 1:
                        self.history = self.history[: self.history_index + 1]

                    self.history.append(self.current_solution)
                    self.history_index = len(self.history) - 1

                # Update UI
                self._update_ui_for_current_solution()
        except Exception as e:
            logger.error(f"Error navigating to weights: {str(e)}")

    def _navigate_to_weights(self):
        """Navigate to the solution with the current weights"""
        if not self.mco_navigator:
            return

        # Get weights from sliders
        weights = {}

        for obj_id, slider in self.objective_sliders.items():
            weights[obj_id] = slider.get_value()

        # Navigate to solution
        try:
            solution = self.mco_navigator.navigate_to_values(weights)

            if solution:
                # Add it to the solution list if not there yet
                solutions = list(self.mco_navigator.get_solutions().values())
                if solution not in solutions:
                    solutions.append(solution)
                    self.solution_list.addItem(
                        f"Solution {len(solutions)}: {solution.solution_id}"
                    )

                # Update current solution
                self.current_solution = solution
                self.current_solution_index = solutions.index(solution)

                # Select in the list
                self.solution_list.setCurrentRow(self.current_solution_index)

                # Add to history
                if (self.history_index < 0) or (
                    self.history_index < len(self.history)
                    and self.history[self.history_index] != self.current_solution
                ):
                    # Truncate history if we're not at the end
                    if self.history_index < len(self.history) - 1:
                        self.history = self.history[: self.history_index + 1]

                    self.history.append(self.current_solution)
                    self.history_index = len(self.history) - 1

                # Update UI
                self._update_ui_for_current_solution()
        except Exception as e:
            logger.error(f"Error navigating to weights: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to navigate to weights: {str(e)}"
            )

    def _save_current_solution(self):
        """Save the current solution to disk"""
        if not self.current_solution:
            return

        try:
            # Save to MCO navigator
            if self.mco_navigator.save_solution(self.current_solution):
                QMessageBox.information(self, "Success", "Solution saved successfully")
            else:
                QMessageBox.warning(self, "Warning", "Failed to save solution")
        except Exception as e:
            logger.error(f"Error saving solution: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save solution: {str(e)}")

    def _undo(self):
        """Undo last navigation"""
        if self.history_index <= 0:
            return

        self.history_index -= 1
        self.current_solution = self.history[self.history_index]

        # Find solution in the list
        solutions = list(self.mco_navigator.get_solutions().values())
        if self.current_solution in solutions:
            self.current_solution_index = solutions.index(self.current_solution)
            self.solution_list.setCurrentRow(self.current_solution_index)

        # Update UI
        self._update_ui_for_current_solution()

    def _redo(self):
        """Redo last undone navigation"""
        if self.history_index >= len(self.history) - 1:
            return

        self.history_index += 1
        self.current_solution = self.history[self.history_index]

        # Find solution in the list
        solutions = list(self.mco_navigator.get_solutions().values())
        if self.current_solution in solutions:
            self.current_solution_index = solutions.index(self.current_solution)
            self.solution_list.setCurrentRow(self.current_solution_index)

        # Update UI
        self._update_ui_for_current_solution()

    def _on_accept(self):
        """Accept the current solution"""
        if not self.current_solution:
            QMessageBox.warning(self, "No Solution", "Please select a solution first")
            return

        # Emit signal
        self.solutionAccepted.emit(self.current_solution)

        # Close dialog
        self.accept()


if __name__ == "__main__":
    """
    Demo standalone mode for MCO Navigator Dialog.
    """
    # This would be used for testing the dialog independently
    app = QApplication(sys.argv)

    # Create a mock plan for testing
    from quangtps.planning.plan import Plan

    mock_plan = Plan()

    dialog = MCONavigatorDialog(mock_plan)
    dialog.show()

    sys.exit(app.exec_())
