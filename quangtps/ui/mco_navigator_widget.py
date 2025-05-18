#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Widget điều hướng Pareto cho tối ưu hóa đa tiêu chí.

Module này cung cấp một giao diện người dùng đồ họa để khám phá và
điều hướng không gian giải pháp Pareto trong quy trình tối ưu hóa
đa tiêu chí (MCO).
"""

import os
import logging
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from typing import Dict, List, Tuple, Optional, Any, Callable

try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QLabel,
        QPushButton,
        QSlider,
        QGroupBox,
        QComboBox,
        QCheckBox,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QFileDialog,
        QMessageBox,
        QSplitter,
        QFrame,
        QDialog,
        QDialogButtonBox,
        QListWidget,
        QListWidgetItem,
    )
    from PyQt5.QtGui import QColor
except ImportError:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QLabel,
        QPushButton,
        QSlider,
        QGroupBox,
        QComboBox,
        QCheckBox,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QFileDialog,
        QMessageBox,
        QSplitter,
        QFrame,
        QDialog,
        QDialogButtonBox,
        QListWidget,
        QListWidgetItem,
    )
    from PyQt6.QtGui import QColor

from quangtps.optimization.mco.pareto_navigator import ParetoNavigator, ParetoSolution
from quangtps.core.planning import Plan

logger = logging.getLogger(__name__)


class ParetoFigureCanvas(FigureCanvasQTAgg):
    """Canvas để hiển thị đồ thị Pareto."""

    clicked_point = pyqtSignal(str)  # Phát tín hiệu ID của điểm được chọn

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection="3d")
        super().__init__(self.fig)
        self.setParent(parent)
        self.solutions = {}
        self.highlighted_solution = None
        self.selected_objectives = []

        # Thêm các thuộc tính mới
        self.color_by_objective = None  # Mục tiêu để tô màu
        self.solution_points = None  # Lưu trữ các điểm để cập nhật màu
        self.colorbar = None  # Thanh màu

        # Đặt màu nền cho Eclipse-like style
        self.fig.patch.set_facecolor("#f0f0f0")
        self.axes.set_facecolor("#ffffff")

        # Kết nối sự kiện click chuột
        self.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event):
        """Xử lý khi người dùng nhấp vào đồ thị."""
        if event.inaxes != self.axes:
            return

        if not self.solutions:
            return

        # Tìm điểm gần nhất
        min_dist = float("inf")
        closest_id = None

        if len(self.selected_objectives) < 2:
            return

        x_obj, y_obj = self.selected_objectives[:2]
        z_obj = (
            self.selected_objectives[2] if len(self.selected_objectives) > 2 else None
        )

        for sol_id, sol in self.solutions.items():
            if x_obj not in sol.objective_values or y_obj not in sol.objective_values:
                continue

            x = sol.objective_values[x_obj]
            y = sol.objective_values[y_obj]

            if z_obj and z_obj in sol.objective_values:
                z = sol.objective_values[z_obj]
                dist = np.sqrt(
                    (x - event.xdata) ** 2
                    + (y - event.ydata) ** 2
                    + (z - event.zdata) ** 2
                )
            else:
                dist = np.sqrt((x - event.xdata) ** 2 + (y - event.ydata) ** 2)

            if dist < min_dist:
                min_dist = dist
                closest_id = sol_id

        if closest_id and min_dist < 0.1:  # Ngưỡng khoảng cách
            self.clicked_point.emit(closest_id)

    def highlight_solution(self, solution_id):
        """
        Tô sáng một giải pháp cụ thể trên đồ thị.

        Args:
            solution_id (str): ID của giải pháp cần tô sáng
        """
        self.highlighted_solution = solution_id
        self.update_plot()

    def set_color_by_objective(self, objective_name):
        """
        Thiết lập mục tiêu dùng để tô màu các điểm.

        Args:
            objective_name (str): Tên mục tiêu hoặc None để không tô màu
        """
        self.color_by_objective = objective_name
        self.update_plot()

    def update_plot(self):
        """Cập nhật đồ thị với cấu hình hiện tại mà không vẽ lại toàn bộ."""
        if (
            not self.solutions
            or not self.selected_objectives
            or not self.solution_points
        ):
            return

        # Cập nhật màu sắc dựa trên mục tiêu được chọn
        if (
            self.color_by_objective
            and self.color_by_objective in self.selected_objectives
        ):
            values = []
            for sol_id in self.solutions:
                if sol_id in self.solutions:
                    sol = self.solutions[sol_id]
                    if self.color_by_objective in sol.objective_values:
                        values.append(sol.objective_values[self.color_by_objective])
                    else:
                        values.append(0)

            if values:
                self.solution_points.set_array(np.array(values))

                # Cập nhật hoặc tạo thanh màu
                if self.colorbar:
                    self.colorbar.update_normal(self.solution_points)
                else:
                    self.colorbar = self.fig.colorbar(
                        self.solution_points, ax=self.axes
                    )
                    self.colorbar.set_label(self.color_by_objective)

        # Tô sáng giải pháp được chọn
        if self.highlighted_solution and self.highlighted_solution in self.solutions:
            sol = self.solutions[self.highlighted_solution]
            x_obj, y_obj = self.selected_objectives[:2]
            z_obj = (
                self.selected_objectives[2]
                if len(self.selected_objectives) > 2
                else None
            )

            if x_obj in sol.objective_values and y_obj in sol.objective_values:
                x = sol.objective_values[x_obj]
                y = sol.objective_values[y_obj]

                if z_obj and z_obj in sol.objective_values:
                    z = sol.objective_values[z_obj]
                    self.axes.scatter(
                        [x], [y], [z], c="red", s=100, marker="*", zorder=10
                    )
                else:
                    self.axes.scatter(
                        [x], [y], [0], c="red", s=100, marker="*", zorder=10
                    )

        self.draw()

    def clear(self):
        """Xóa đồ thị và đặt lại các tham số."""
        self.axes.clear()
        self.highlighted_solution = None
        self.solution_points = None
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
        self.draw()


class ObjectiveWeightEditor(QWidget):
    """Widget để hiệu chỉnh trọng số cho các mục tiêu tối ưu hóa."""

    weights_changed = pyqtSignal(dict)  # Phát tín hiệu khi trọng số thay đổi

    def __init__(self, parent=None):
        super().__init__(parent)
        self.objectives = []
        self.sliders = {}
        self.weight_labels = {}
        self.is_updating = False

        # Thiết lập giao diện
        layout = QVBoxLayout(self)

        # Nhãn tiêu đề
        title = QLabel("Trọng số mục tiêu")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Layout cho sliders
        self.sliders_layout = QVBoxLayout()
        layout.addLayout(self.sliders_layout)

        # Nút reset
        reset_btn = QPushButton("Đặt lại trọng số")
        reset_btn.clicked.connect(self._reset_weights)
        layout.addWidget(reset_btn)

        layout.addStretch()

    def set_objectives(
        self, objectives: List[str], default_weights: Optional[Dict[str, float]] = None
    ):
        """Thiết lập danh sách mục tiêu và trọng số mặc định."""
        self.is_updating = True
        self.objectives = objectives

        # Xóa sliders cũ
        for slider in self.sliders.values():
            slider.setParent(None)
        self.sliders = {}
        self.weight_labels = {}

        # Xóa layout
        while self.sliders_layout.count():
            item = self.sliders_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Trọng số mặc định bằng nhau
        if default_weights is None:
            weight = 1.0 / len(objectives) if objectives else 0.0
            default_weights = {obj: weight for obj in objectives}

        # Tạo sliders mới
        for obj in objectives:
            weight = default_weights.get(obj, 0.0)

            # Container widget for each slider row
            container = QWidget()
            row_layout = QHBoxLayout(container)
            row_layout.setContentsMargins(0, 0, 0, 0)

            # Label for objective name
            label = QLabel(obj)
            label.setMinimumWidth(100)
            row_layout.addWidget(label)

            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(weight * 100))
            slider.valueChanged.connect(lambda v, o=obj: self._on_slider_changed(o, v))
            row_layout.addWidget(slider)

            # Weight label
            weight_label = QLabel(f"{weight:.2f}")
            weight_label.setMinimumWidth(40)
            row_layout.addWidget(weight_label)

            self.sliders_layout.addWidget(container)
            self.sliders[obj] = slider
            self.weight_labels[obj] = weight_label

        self.is_updating = False
        self._normalize_weights()

    def _on_slider_changed(self, obj_name: str, value: int):
        """Xử lý khi một slider thay đổi giá trị."""
        if self.is_updating:
            return

        # Cập nhật nhãn
        weight = value / 100.0
        self.weight_labels[obj_name].setText(f"{weight:.2f}")

        # Chuẩn hóa trọng số
        self._normalize_weights()

        # Phát tín hiệu trọng số đã thay đổi
        weights = {obj: self.sliders[obj].value() / 100.0 for obj in self.sliders}
        self.weights_changed.emit(weights)

    def _reset_weights(self):
        """Đặt lại tất cả trọng số về giá trị mặc định bằng nhau."""
        if not self.objectives:
            return

        self.is_updating = True
        weight = 1.0 / len(self.objectives)

        for obj in self.objectives:
            self.sliders[obj].setValue(int(weight * 100))
            self.weight_labels[obj].setText(f"{weight:.2f}")

        self.is_updating = False

        # Phát tín hiệu trọng số đã thay đổi
        weights = {obj: weight for obj in self.objectives}
        self.weights_changed.emit(weights)

    def _normalize_weights(self):
        """Chuẩn hóa trọng số để tổng bằng 1."""
        if self.is_updating or not self.sliders:
            return

        self.is_updating = True

        # Lấy tổng trọng số hiện tại
        total = sum(slider.value() for slider in self.sliders.values())

        if total > 0:
            # Chuẩn hóa
            for obj, slider in self.sliders.items():
                normalized = slider.value() / total
                slider.setValue(int(normalized * 100))
                self.weight_labels[obj].setText(f"{normalized:.2f}")

        self.is_updating = False


class SolutionDetailsPanel(QWidget):
    """Panel hiển thị chi tiết về giải pháp Pareto đã chọn."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # Tiêu đề
        title = QLabel("Chi tiết giải pháp")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Bảng mục tiêu
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Mục tiêu", "Giá trị", "Trọng số"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # Thông tin tóm tắt
        self.summary = QLabel()
        layout.addWidget(self.summary)

        # Nút áp dụng
        self.apply_btn = QPushButton("Áp dụng giải pháp này")
        layout.addWidget(self.apply_btn)

    def display_solution(self, solution: Optional[ParetoSolution]):
        """Hiển thị thông tin chi tiết về giải pháp đã chọn."""
        # Xóa dữ liệu cũ
        self.table.setRowCount(0)
        self.summary.setText("")

        if not solution:
            self.apply_btn.setEnabled(False)
            return

        self.apply_btn.setEnabled(True)

        # Hiển thị thông tin mục tiêu
        self.table.setRowCount(len(solution.objective_values))

        for i, (obj_name, value) in enumerate(solution.objective_values.items()):
            # Tên mục tiêu
            self.table.setItem(i, 0, QTableWidgetItem(obj_name))

            # Giá trị
            self.table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))

            # Trọng số
            weight = solution.weights.get(obj_name, 0.0)
            self.table.setItem(i, 2, QTableWidgetItem(f"{weight:.4f}"))

        # Hiển thị tóm tắt
        total_score = sum(
            v * solution.weights.get(k, 0.0)
            for k, v in solution.objective_values.items()
        )
        self.summary.setText(f"Tổng điểm: {total_score:.4f}\nID: {solution.id}")


class ParetoNavigatorWidget(QWidget):
    """
    Widget điều hướng Pareto cho tối ưu hóa đa tiêu chí (MCO).

    Widget này cung cấp giao diện người dùng để khám phá và điều hướng
    không gian giải pháp Pareto, cho phép người dùng chọn giải pháp tối ưu
    dựa trên sự đánh đổi giữa các mục tiêu lâm sàng khác nhau.
    """

    plan_created = pyqtSignal(Plan)  # Phát tín hiệu khi tạo kế hoạch từ giải pháp

    def __init__(self, parent=None):
        super().__init__(parent)
        self.navigator = None
        self.current_solution = None

        self._setup_ui()

    def set_navigator(self, navigator: ParetoNavigator):
        """Thiết lập đối tượng ParetoNavigator để điều hướng không gian Pareto."""
        self.navigator = navigator

        if navigator:
            # Lấy danh sách mục tiêu
            objectives = []
            if hasattr(navigator.pareto_surface, "objectives"):
                objectives = list(navigator.pareto_surface.objectives.keys())

            # Thiết lập danh sách mục tiêu cho các thành phần UI
            self.weight_editor.set_objectives(objectives)
            self.obj_x_combo.clear()
            self.obj_x_combo.addItems(objectives)
            self.obj_y_combo.clear()
            self.obj_y_combo.addItems(objectives)
            self.obj_z_combo.clear()
            self.obj_z_combo.addItem("Không")
            self.obj_z_combo.addItems(objectives)

            # Thiết lập giá trị mặc định cho combos
            if len(objectives) > 0:
                self.obj_x_combo.setCurrentIndex(0)
            if len(objectives) > 1:
                self.obj_y_combo.setCurrentIndex(1)

            # Vẽ bề mặt Pareto
            self._draw_pareto_surface()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        main_layout = QVBoxLayout(self)

        # Tiêu đề
        title = QLabel("Điều hướng Pareto")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        main_layout.addWidget(title)

        # Splitter chính
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Panel bên trái
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Widget chọn trọng số
        self.weight_editor = ObjectiveWeightEditor()
        left_layout.addWidget(self.weight_editor)

        # Nút chọn theo trọng số
        select_btn = QPushButton("Chọn theo trọng số")
        select_btn.clicked.connect(self._select_by_weights)
        left_layout.addWidget(select_btn)

        # Nút hiển thị giải pháp lân cận
        neighbors_btn = QPushButton("Xem giải pháp lân cận")
        neighbors_btn.clicked.connect(self._show_neighboring_solutions)
        left_layout.addWidget(neighbors_btn)

        # Nút tạo kế hoạch
        create_plan_btn = QPushButton("Tạo kế hoạch")
        create_plan_btn.clicked.connect(self._create_plan)
        left_layout.addWidget(create_plan_btn)

        # Nút lưu phiên điều hướng
        save_btn = QPushButton("Lưu phiên điều hướng")
        save_btn.clicked.connect(self._save_session)
        left_layout.addWidget(save_btn)

        main_splitter.addWidget(left_panel)

        # Panel bên phải (trực quan hóa)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Controls for visualization
        viz_control = QWidget()
        viz_layout = QHBoxLayout(viz_control)
        viz_layout.setContentsMargins(0, 0, 0, 0)

        # X-axis objective
        viz_layout.addWidget(QLabel("Trục X:"))
        self.obj_x_combo = QComboBox()
        self.obj_x_combo.currentIndexChanged.connect(self._draw_pareto_surface)
        viz_layout.addWidget(self.obj_x_combo)

        # Y-axis objective
        viz_layout.addWidget(QLabel("Trục Y:"))
        self.obj_y_combo = QComboBox()
        self.obj_y_combo.currentIndexChanged.connect(self._draw_pareto_surface)
        viz_layout.addWidget(self.obj_y_combo)

        # Z-axis objective (optional)
        viz_layout.addWidget(QLabel("Trục Z:"))
        self.obj_z_combo = QComboBox()
        self.obj_z_combo.currentIndexChanged.connect(self._draw_pareto_surface)
        viz_layout.addWidget(self.obj_z_combo)

        right_layout.addWidget(viz_control)

        # Pareto surface figure
        self.figure_canvas = ParetoFigureCanvas(self)
        right_layout.addWidget(self.figure_canvas)

        main_splitter.addWidget(right_panel)

        # Panel chi tiết giải pháp
        self.details_panel = SolutionDetailsPanel()
        self.details_panel.apply_btn.clicked.connect(self._create_plan)
        main_splitter.addWidget(self.details_panel)

        # Thiết lập kích thước tương đối
        main_splitter.setSizes([200, 500, 300])

        # Kết nối các tín hiệu
        self._connect_signals()

    def _connect_signals(self):
        """Kết nối các tín hiệu và slots."""
        self.weight_editor.weights_changed.connect(self._on_weights_changed)
        self.figure_canvas.clicked_point.connect(self._on_solution_selected)

    def _draw_pareto_surface(self):
        """Vẽ bề mặt Pareto dựa trên các mục tiêu đã chọn."""
        if not self.navigator:
            return

        # Xóa đồ thị
        self.figure_canvas.clear()

        # Lấy các mục tiêu đã chọn
        if self.obj_x_combo.currentIndex() < 0 or self.obj_y_combo.currentIndex() < 0:
            return

        x_obj = self.obj_x_combo.currentText()
        y_obj = self.obj_y_combo.currentText()

        use_3d = self.obj_z_combo.currentIndex() > 0
        z_obj = (
            self.obj_z_combo.currentText()
            if use_3d and self.obj_z_combo.currentText() != "Không"
            else None
        )

        # Lưu danh sách mục tiêu được chọn
        selected_objectives = [x_obj, y_obj]
        if z_obj:
            selected_objectives.append(z_obj)
        self.figure_canvas.selected_objectives = selected_objectives

        # Lấy tất cả giải pháp từ navigator
        if not hasattr(self.navigator.pareto_surface, "solutions"):
            return

        solutions = self.navigator.pareto_surface.solutions
        self.figure_canvas.solutions = {sol.id: sol for sol in solutions}

        x_values = []
        y_values = []
        z_values = []
        colors = []  # Giá trị màu dựa trên mục tiêu được chọn

        # Xác định mục tiêu để tô màu (mục tiêu thứ 3 hoặc đầu tiên nếu không có mục tiêu thứ 3)
        color_objective = z_obj if z_obj else x_obj
        self.figure_canvas.color_by_objective = color_objective

        for sol in solutions:
            if x_obj in sol.objective_values and y_obj in sol.objective_values:
                x_values.append(sol.objective_values[x_obj])
                y_values.append(sol.objective_values[y_obj])

                if z_obj and z_obj in sol.objective_values:
                    z_values.append(sol.objective_values[z_obj])
                else:
                    z_values.append(0)

                # Thêm giá trị cho tô màu
                if color_objective in sol.objective_values:
                    colors.append(sol.objective_values[color_objective])
                else:
                    colors.append(0)

        # Vẽ bề mặt Pareto
        if not x_values:
            return

        if use_3d and z_obj:
            # Mặt 3D
            scatter = self.figure_canvas.axes.scatter(
                x_values,
                y_values,
                z_values,
                c=colors,
                cmap="viridis",
                marker="o",
                alpha=0.7,
            )
            self.figure_canvas.axes.set_xlabel(x_obj)
            self.figure_canvas.axes.set_ylabel(y_obj)
            self.figure_canvas.axes.set_zlabel(z_obj)

            # Cập nhật giới hạn trục
            x_range = max(x_values) - min(x_values)
            y_range = max(y_values) - min(y_values)
            z_range = max(z_values) - min(z_values)

            self.figure_canvas.axes.set_xlim(
                [min(x_values) - 0.1 * x_range, max(x_values) + 0.1 * x_range]
            )
            self.figure_canvas.axes.set_ylim(
                [min(y_values) - 0.1 * y_range, max(y_values) + 0.1 * y_range]
            )
            self.figure_canvas.axes.set_zlim(
                [min(z_values) - 0.1 * z_range, max(z_values) + 0.1 * z_range]
            )

        else:
            # Mặt 2D
            scatter = self.figure_canvas.axes.scatter(
                x_values, y_values, c=colors, cmap="viridis", marker="o", alpha=0.7
            )
            self.figure_canvas.axes.set_xlabel(x_obj)
            self.figure_canvas.axes.set_ylabel(y_obj)

            # Cập nhật giới hạn trục
            x_range = max(x_values) - min(x_values)
            y_range = max(y_values) - min(y_values)

            self.figure_canvas.axes.set_xlim(
                [min(x_values) - 0.1 * x_range, max(x_values) + 0.1 * x_range]
            )
            self.figure_canvas.axes.set_ylim(
                [min(y_values) - 0.1 * y_range, max(y_values) + 0.1 * y_range]
            )

            # Ẩn trục Z trong chế độ 2D
            self.figure_canvas.axes.set_zticks([])

        # Thêm tiêu đề
        self.figure_canvas.axes.set_title("Bề mặt Pareto")

        # Lưu đối tượng scatter để cập nhật sau này
        self.figure_canvas.solution_points = scatter

        # Cập nhật thanh màu và tô sáng giải pháp được chọn
        self.figure_canvas.colorbar = self.figure_canvas.fig.colorbar(
            scatter, ax=self.figure_canvas.axes
        )
        self.figure_canvas.colorbar.set_label(color_objective)

        # Nếu có giải pháp hiện tại, tô sáng nó
        if self.current_solution:
            self.figure_canvas.highlight_solution(self.current_solution.id)

        # Thêm lưới và cải thiện hiển thị
        self.figure_canvas.axes.grid(True, alpha=0.3)

        # Vẽ lại canvas
        self.figure_canvas.draw()

    def _on_weights_changed(self, weights: Dict[str, float]):
        """
        Xử lý khi trọng số được thay đổi thông qua giao diện sliders.

        Args:
            weights: Dict[str, float] - Từ điển chứa tên mục tiêu và trọng số tương ứng
        """
        if not self.navigator or not weights:
            return

        try:
            # Hiển thị thông tin trạng thái
            status_msg = "Đang tìm giải pháp tối ưu với trọng số mới..."
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Tìm giải pháp phù hợp nhất với các trọng số mới
            solution = self.navigator.find_solution_by_weights(weights)

            # Hiển thị giải pháp được tìm thấy
            if solution:
                self.current_solution = solution

                # Cập nhật hiển thị giải pháp
                self.details_panel.display_solution(solution)

                # Tô sáng giải pháp trên đồ thị
                self.figure_canvas.highlight_solution(solution.id)

                status_msg = "Đã tìm thấy giải pháp phù hợp với trọng số được chọn."
            else:
                status_msg = "Không tìm thấy giải pháp phù hợp với trọng số được chọn."

            # Reset con trỏ chuột
            QApplication.restoreOverrideCursor()

            # Hiển thị thông báo trạng thái tạm thời ở góc phải dưới
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage(status_msg, 3000)

        except Exception as e:
            # Hiển thị thông báo lỗi
            QMessageBox.warning(
                self,
                "Lỗi tìm kiếm giải pháp",
                f"Không thể tìm giải pháp với trọng số đã chọn: {str(e)}",
            )
            logger.error(f"Error finding solution by weights: {str(e)}")

            # Reset con trỏ chuột
            QApplication.restoreOverrideCursor()

    def _select_by_weights(self):
        """
        Chọn giải pháp tốt nhất dựa trên trọng số hiện tại.

        Chức năng này tìm kiếm giải pháp phù hợp nhất với trọng số hiện tại
        hoặc nội suy một giải pháp mới nếu không có giải pháp trực tiếp.
        """
        if not self.navigator:
            QMessageBox.warning(self, "Lỗi", "Chưa khởi tạo ParetoNavigator.")
            return

        try:
            # Lấy trọng số hiện tại từ widget điều chỉnh trọng số
            weights = {}
            for obj in self.navigator.pareto_surface.objectives.keys():
                if obj in self.weight_editor.sliders:
                    weights[obj] = self.weight_editor.sliders[obj].value() / 100.0

            if not weights:
                QMessageBox.warning(
                    self, "Lỗi", "Không có mục tiêu nào được thiết lập."
                )
                return

            # Hiển thị thông tin trạng thái
            wait_dialog = QMessageBox(self)
            wait_dialog.setWindowTitle("Đang xử lý")
            wait_dialog.setText("Đang tìm giải pháp tối ưu với trọng số mới...")
            wait_dialog.setStandardButtons(QMessageBox.NoButton)
            QTimer.singleShot(
                100, wait_dialog.close
            )  # Chỉ hiển thị trong thời gian ngắn
            wait_dialog.show()
            QApplication.processEvents()
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Tìm giải pháp phù hợp với trọng số hoặc nội suy một giải pháp mới
            solution = self.navigator.navigate_to_weights(weights)

            QApplication.restoreOverrideCursor()

            if solution:
                # Cập nhật giải pháp hiện tại
                self.current_solution = solution

                # Cập nhật hiển thị giải pháp
                self.details_panel.display_solution(solution)

                # Tô sáng giải pháp trên đồ thị
                self.figure_canvas.highlight_solution(solution.id)

                # Thông báo thành công
                QMessageBox.information(
                    self,
                    "Thành công",
                    "Đã tìm thấy giải pháp tối ưu với trọng số đã chọn.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Thông báo",
                    "Không thể tìm thấy hoặc nội suy giải pháp với trọng số đã chọn.",
                )

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi tìm kiếm giải pháp: {str(e)}")
            logger.error(f"Error in _select_by_weights: {str(e)}", exc_info=True)

    def _show_neighboring_solutions(self):
        """Hiển thị các giải pháp lân cận của giải pháp hiện tại."""
        if not self.navigator or not self.current_solution:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một giải pháp trước.")
            return

        # Lấy các giải pháp lân cận
        neighbors = self.navigator.get_neighboring_solutions(num_neighbors=5)

        if not neighbors:
            QMessageBox.information(
                self, "Thông báo", "Không tìm thấy giải pháp lân cận."
            )
            return

        # Hiển thị dialog chọn giải pháp lân cận
        dialog = QDialog(self)
        dialog.setWindowTitle("Giải pháp lân cận")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)

        # Hướng dẫn
        layout.addWidget(QLabel("Chọn một giải pháp lân cận để điều hướng:"))

        # Danh sách giải pháp
        solution_list = QListWidget()
        layout.addWidget(solution_list)

        # Thêm giải pháp hiện tại
        current_item = QListWidgetItem(
            f"Giải pháp hiện tại (ID: {self.current_solution.id})"
        )
        current_item.setData(Qt.UserRole, self.current_solution.id)
        solution_list.addItem(current_item)

        # Thêm các giải pháp lân cận
        for i, sol in enumerate(neighbors):
            # Tính tổng điểm
            score = sum(
                v * sol.weights.get(k, 0.0) for k, v in sol.objective_values.items()
            )

            # Tạo mô tả
            description = f"Lân cận {i + 1} (ID: {sol.id}) - Điểm: {score:.4f}"

            item = QListWidgetItem(description)
            item.setData(Qt.UserRole, sol.id)
            solution_list.addItem(item)

        # Nút điều khiển
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # Hiển thị dialog
        if dialog.exec() == QDialog.Accepted:
            selected_items = solution_list.selectedItems()
            if selected_items:
                sol_id = selected_items[0].data(Qt.UserRole)
                self._on_solution_selected(sol_id)

    def _create_plan(self):
        """Tạo kế hoạch xạ trị từ giải pháp hiện tại."""
        if not self.navigator or not self.current_solution:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một giải pháp trước.")
            return

        # Tạo kế hoạch
        plan = self.navigator.create_plan_from_current_solution()

        if plan:
            # Phát tín hiệu với kế hoạch đã tạo
            self.plan_created.emit(plan)
            QMessageBox.information(
                self, "Thành công", "Đã tạo kế hoạch từ giải pháp đã chọn."
            )
        else:
            QMessageBox.warning(
                self,
                "Lỗi",
                "Không thể tạo kế hoạch. Vui lòng kiểm tra xem generator đã được thiết lập chưa.",
            )

    def _save_session(self):
        """Lưu phiên điều hướng hiện tại."""
        if not self.navigator:
            QMessageBox.warning(
                self, "Lỗi", "Không có phiên điều hướng nào được thiết lập."
            )
            return

        # Hiển thị dialog để chọn file
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Lưu phiên điều hướng", "", "JSON Files (*.json)"
        )

        if not filepath:
            return

        # Thêm phần mở rộng .json nếu cần
        if not filepath.endswith(".json"):
            filepath += ".json"

        # Lưu phiên
        success = self.navigator.save_navigation_session(filepath)

        if success:
            QMessageBox.information(
                self, "Thành công", f"Đã lưu phiên điều hướng vào {filepath}"
            )
        else:
            QMessageBox.warning(
                self, "Lỗi", "Không thể lưu phiên điều hướng. Vui lòng thử lại."
            )

    def _on_solution_selected(self, solution_id: str):
        """
        Xử lý khi người dùng chọn một giải pháp từ đồ thị.

        Args:
            solution_id (str): ID của giải pháp được chọn
        """
        if not self.navigator:
            return

        try:
            # Lấy giải pháp theo ID
            solution = None
            for sol in self.navigator.pareto_surface.solutions:
                if sol.id == solution_id:
                    solution = sol
                    break

            if not solution:
                logger.warning(f"Không tìm thấy giải pháp với ID: {solution_id}")
                return

            # Cập nhật giải pháp hiện tại
            self.current_solution = solution

            # Cập nhật hiển thị giải pháp
            self.details_panel.display_solution(solution)

            # Tô sáng giải pháp trên đồ thị
            self.figure_canvas.highlight_solution(solution.id)

            # Cập nhật trọng số trong widget điều chỉnh trọng số
            if hasattr(solution, "weights") and solution.weights:
                # Tránh gọi lại hàm xử lý thay đổi trọng số
                self.weight_editor.is_updating = True

                for obj, weight in solution.weights.items():
                    if obj in self.weight_editor.sliders:
                        self.weight_editor.sliders[obj].setValue(int(weight * 100))
                        self.weight_editor.weight_labels[obj].setText(f"{weight:.2f}")

                self.weight_editor.is_updating = False

            # Tạo hiệu ứng nhấp nháy nhẹ để thu hút sự chú ý
            old_border = self.details_panel.styleSheet()
            self.details_panel.setStyleSheet(
                "border: 2px solid #3498db; border-radius: 5px;"
            )

            # Sau 500ms, quay lại kiểu cũ
            QTimer.singleShot(500, lambda: self.details_panel.setStyleSheet(old_border))

            # Hiển thị các mục tiêu chính của giải pháp này
            if hasattr(solution, "objective_values") and solution.objective_values:
                values_text = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in solution.objective_values.items()]
                )
                if hasattr(self, "statusBar") and self.statusBar():
                    self.statusBar().showMessage(
                        f"Đã chọn giải pháp: {values_text}", 5000
                    )

        except Exception as e:
            logger.error(f"Lỗi khi chọn giải pháp: {str(e)}", exc_info=True)
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage(f"Lỗi khi chọn giải pháp: {str(e)}", 3000)


def create_pareto_navigator_widget(
    navigator: ParetoNavigator = None,
) -> ParetoNavigatorWidget:
    """Tạo một widget điều hướng Pareto mới."""
    widget = ParetoNavigatorWidget()

    if navigator:
        widget.set_navigator(navigator)

    return widget
