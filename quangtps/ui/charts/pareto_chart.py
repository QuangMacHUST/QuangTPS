#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module định nghĩa các đồ thị Pareto cho tối ưu hóa đa tiêu chí.

Module này cung cấp các lớp để hiển thị mặt Pareto 2D và 3D, với khả năng
tương tác để lựa chọn các giải pháp trên mặt Pareto.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Callable

try:
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QComboBox,
        QToolBar,
        QAction,
        QSizePolicy,
        QCheckBox,
        QTabWidget,
    )
    from PyQt5.QtGui import QIcon, QPixmap
except ImportError:
    from PyQt6.QtCore import Qt, pyqtSignal, QSize
    from PyQt6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QComboBox,
        QToolBar,
        QAction,
        QSizePolicy,
        QCheckBox,
        QTabWidget,
    )
    from PyQt6.QtGui import QIcon, QPixmap

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from quangtps.optimization.mco.pareto_navigator import ParetoSolution

logger = logging.getLogger(__name__)


class ParetoChart(QWidget):
    """
    Widget đồ thị trực quan hóa mặt Pareto.

    Hiển thị mặt Pareto 2D hoặc 3D cho phép người dùng khám phá và chọn
    các giải pháp Pareto-optimal.
    """

    # Tín hiệu khi người dùng chọn một giải pháp
    solutionSelected = pyqtSignal(str)  # Phát ra ID của giải pháp được chọn

    def __init__(self, parent=None):
        super().__init__(parent)
        self.solutions: Dict[str, ParetoSolution] = {}
        self.highlighted_id: Optional[str] = None
        self.history_ids: Set[str] = set()
        self.selected_objectives: List[str] = []
        self.picked_points = []
        self.current_mode = "2D"  # "2D" hoặc "3D"

        self._setup_ui()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Thanh công cụ
        toolbar_layout = QHBoxLayout()

        # Combobox chọn mục tiêu
        self.x_combo = QComboBox()
        self.x_combo.setMinimumWidth(150)
        self.y_combo = QComboBox()
        self.y_combo.setMinimumWidth(150)
        self.z_combo = QComboBox()
        self.z_combo.setMinimumWidth(150)
        self.z_combo.setVisible(False)

        toolbar_layout.addWidget(QLabel("X:"))
        toolbar_layout.addWidget(self.x_combo)
        toolbar_layout.addWidget(QLabel("Y:"))
        toolbar_layout.addWidget(self.y_combo)
        toolbar_layout.addWidget(QLabel("Z:"))
        toolbar_layout.addWidget(self.z_combo)

        # Checkbox cho chế độ 3D
        self.mode_3d = QCheckBox("Chế độ 3D")
        self.mode_3d.stateChanged.connect(self._toggle_3d_mode)
        toolbar_layout.addWidget(self.mode_3d)

        # Checkbox hiển thị lịch sử
        self.show_history = QCheckBox("Hiển thị lịch sử")
        self.show_history.setChecked(True)
        self.show_history.stateChanged.connect(self._update_plot)
        toolbar_layout.addWidget(self.show_history)

        toolbar_layout.addStretch()

        layout.addLayout(toolbar_layout)

        # Figure và canvas cho đồ thị Matplotlib
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.mpl_connect("button_press_event", self._on_plot_clicked)

        # Thanh công cụ Matplotlib
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.canvas)
        layout.addWidget(self.mpl_toolbar)

        # Kết nối signals
        self.x_combo.currentIndexChanged.connect(self._update_plot)
        self.y_combo.currentIndexChanged.connect(self._update_plot)
        self.z_combo.currentIndexChanged.connect(self._update_plot)

    def _toggle_3d_mode(self, state):
        """Chuyển đổi giữa chế độ hiển thị 2D và 3D."""
        self.current_mode = "3D" if state else "2D"
        self.z_combo.setVisible(state)
        self._update_plot()

    def _on_plot_clicked(self, event):
        """Xử lý khi người dùng click vào đồ thị."""
        if event.inaxes is None or not self.solutions:
            return

        # Lấy tọa độ điểm được click
        x, y = event.xdata, event.ydata

        # Xác định trục nào đang được sử dụng
        if self.x_combo.count() == 0 or self.y_combo.count() == 0:
            return

        x_obj = self.x_combo.currentText()
        y_obj = self.y_combo.currentText()

        # Tìm giải pháp gần nhất với điểm được click
        closest_id = None
        min_dist = float("inf")

        for sol_id, sol in self.solutions.items():
            obj_values = sol.objective_values
            if x_obj in obj_values and y_obj in obj_values:
                sol_x = obj_values[x_obj]
                sol_y = obj_values[y_obj]

                # Tính khoảng cách Euclidean
                dist = np.sqrt((sol_x - x) ** 2 + (sol_y - y) ** 2)

                if dist < min_dist:
                    min_dist = dist
                    closest_id = sol_id

        # Nếu tìm thấy giải pháp gần nhất
        if closest_id and min_dist < 0.05 * (
            self.figure.get_figwidth() + self.figure.get_figheight()
        ):
            self.highlighted_id = closest_id
            self.solutionSelected.emit(closest_id)
            self._update_plot()

    def set_solutions(self, solutions: Dict[str, ParetoSolution] = None):
        """Thiết lập danh sách các giải pháp Pareto."""
        if solutions is None:
            solutions = {}

        self.solutions = solutions

        # Cập nhật danh sách các mục tiêu
        all_objectives = set()
        for sol in self.solutions.values():
            all_objectives.update(sol.objective_values.keys())

        self.x_combo.clear()
        self.y_combo.clear()
        self.z_combo.clear()

        objective_list = sorted(list(all_objectives))
        self.x_combo.addItems(objective_list)
        self.y_combo.addItems(objective_list)
        self.z_combo.addItems(objective_list)

        # Chọn mục tiêu mặc định
        if len(objective_list) >= 3:
            self.x_combo.setCurrentIndex(0)
            self.y_combo.setCurrentIndex(1)
            self.z_combo.setCurrentIndex(2)
        elif len(objective_list) >= 2:
            self.x_combo.setCurrentIndex(0)
            self.y_combo.setCurrentIndex(1)

        self._update_plot()

    def set_solutions_list(self, solutions_list: List[ParetoSolution] = None):
        """Thiết lập danh sách các giải pháp Pareto từ list."""
        if solutions_list is None:
            solutions_list = []

        solutions_dict = {sol.id: sol for sol in solutions_list}
        self.set_solutions(solutions_dict)

    def highlight_solution(self, solution_id: str):
        """Đánh dấu một giải pháp cụ thể trên đồ thị."""
        if solution_id in self.solutions:
            self.highlighted_id = solution_id
            self.history_ids.add(solution_id)
            self._update_plot()

    def set_history(self, history_ids: List[str]):
        """Thiết lập lịch sử điều hướng Pareto."""
        self.history_ids = set(id for id in history_ids if id in self.solutions)
        self._update_plot()

    def _update_plot(self):
        """Cập nhật đồ thị Pareto."""
        if not self.solutions:
            self.figure.clear()
            self.canvas.draw()
            return

        self.figure.clear()

        if self.current_mode == "3D" and self.z_combo.count() > 0:
            self._draw_3d_plot()
        else:
            self._draw_2d_plot()

        self.canvas.draw()

    def _draw_2d_plot(self):
        """Vẽ đồ thị Pareto 2D."""
        if self.x_combo.count() == 0 or self.y_combo.count() == 0:
            return

        ax = self.figure.add_subplot(111)

        x_obj = self.x_combo.currentText()
        y_obj = self.y_combo.currentText()

        # Thu thập dữ liệu
        x_values = []
        y_values = []
        ids = []

        for sol_id, sol in self.solutions.items():
            obj_values = sol.objective_values
            if x_obj in obj_values and y_obj in obj_values:
                x_values.append(obj_values[x_obj])
                y_values.append(obj_values[y_obj])
                ids.append(sol_id)

        # Vẽ tất cả các điểm
        if x_values and y_values:
            sc = ax.scatter(x_values, y_values, c="blue", marker="o", alpha=0.7)

            # Đánh dấu điểm được highlight
            if self.highlighted_id in self.solutions:
                sol = self.solutions[self.highlighted_id]
                obj_values = sol.objective_values
                if x_obj in obj_values and y_obj in obj_values:
                    ax.scatter(
                        [obj_values[x_obj]],
                        [obj_values[y_obj]],
                        c="red",
                        marker="*",
                        s=200,
                        alpha=1.0,
                    )

            # Vẽ lịch sử nếu được yêu cầu
            if self.show_history.isChecked() and self.history_ids:
                history_x = []
                history_y = []

                # Thu thập các điểm trong lịch sử theo thứ tự
                for sol_id in self.history_ids:
                    if sol_id in self.solutions:
                        sol = self.solutions[sol_id]
                        obj_values = sol.objective_values
                        if x_obj in obj_values and y_obj in obj_values:
                            history_x.append(obj_values[x_obj])
                            history_y.append(obj_values[y_obj])

                if history_x and history_y:
                    ax.plot(history_x, history_y, "r--", lw=1.5, alpha=0.7)
                    ax.scatter(
                        history_x, history_y, c="green", marker="o", s=80, alpha=0.5
                    )

        # Thiết lập labels và tiêu đề
        ax.set_xlabel(x_obj)
        ax.set_ylabel(y_obj)
        ax.set_title("Mặt Pareto 2D")
        ax.grid(True, linestyle="--", alpha=0.7)

    def _draw_3d_plot(self):
        """Vẽ đồ thị Pareto 3D."""
        if (
            self.x_combo.count() == 0
            or self.y_combo.count() == 0
            or self.z_combo.count() == 0
        ):
            return

        ax = self.figure.add_subplot(111, projection="3d")

        x_obj = self.x_combo.currentText()
        y_obj = self.y_combo.currentText()
        z_obj = self.z_combo.currentText()

        # Thu thập dữ liệu
        x_values = []
        y_values = []
        z_values = []
        ids = []

        for sol_id, sol in self.solutions.items():
            obj_values = sol.objective_values
            if x_obj in obj_values and y_obj in obj_values and z_obj in obj_values:
                x_values.append(obj_values[x_obj])
                y_values.append(obj_values[y_obj])
                z_values.append(obj_values[z_obj])
                ids.append(sol_id)

        # Vẽ tất cả các điểm
        if x_values and y_values and z_values:
            ax.scatter(x_values, y_values, z_values, c="blue", marker="o", alpha=0.7)

            # Đánh dấu điểm được highlight
            if self.highlighted_id in self.solutions:
                sol = self.solutions[self.highlighted_id]
                obj_values = sol.objective_values
                if x_obj in obj_values and y_obj in obj_values and z_obj in obj_values:
                    ax.scatter(
                        [obj_values[x_obj]],
                        [obj_values[y_obj]],
                        [obj_values[z_obj]],
                        c="red",
                        marker="*",
                        s=200,
                        alpha=1.0,
                    )

            # Vẽ lịch sử nếu được yêu cầu
            if self.show_history.isChecked() and self.history_ids:
                history_x = []
                history_y = []
                history_z = []

                # Thu thập các điểm trong lịch sử theo thứ tự
                for sol_id in self.history_ids:
                    if sol_id in self.solutions:
                        sol = self.solutions[sol_id]
                        obj_values = sol.objective_values
                        if (
                            x_obj in obj_values
                            and y_obj in obj_values
                            and z_obj in obj_values
                        ):
                            history_x.append(obj_values[x_obj])
                            history_y.append(obj_values[y_obj])
                            history_z.append(obj_values[z_obj])

                if history_x and history_y and history_z:
                    ax.plot(history_x, history_y, history_z, "r--", lw=1.5, alpha=0.7)
                    ax.scatter(
                        history_x,
                        history_y,
                        history_z,
                        c="green",
                        marker="o",
                        s=80,
                        alpha=0.5,
                    )

        # Thiết lập labels và tiêu đề
        ax.set_xlabel(x_obj)
        ax.set_ylabel(y_obj)
        ax.set_zlabel(z_obj)
        ax.set_title("Mặt Pareto 3D")
        ax.grid(True, linestyle="--", alpha=0.7)

    def clear(self):
        """Xóa tất cả dữ liệu và làm mới đồ thị."""
        self.solutions = {}
        self.highlighted_id = None
        self.history_ids = set()
        self.figure.clear()
        self.canvas.draw()

        self.x_combo.clear()
        self.y_combo.clear()
        self.z_combo.clear()

    def save_figure(self, filepath: str):
        """Lưu hình ảnh đồ thị hiện tại."""
        if filepath:
            self.figure.savefig(
                filepath, dpi=300, bbox_inches="tight", format=filepath.split(".")[-1]
            )
            logger.info(f"Đã lưu hình ảnh đồ thị Pareto vào {filepath}")


class ParetoRadarChart(QWidget):
    """
    Biểu đồ radar để trực quan hóa các giải pháp Pareto theo nhiều chiều.

    Biểu đồ này hiển thị các giá trị của nhiều mục tiêu khác nhau
    trên một biểu đồ radar (spider chart), giúp trực quan hóa
    sự cân bằng và đánh đổi giữa các mục tiêu.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.solution = None
        self.reference_solution = None
        self.objectives = []
        self.values = []
        self.reference_values = []
        self.normalized = True

        self._setup_ui()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Thanh công cụ
        toolbar_layout = QHBoxLayout()

        # Checkbox chuẩn hóa dữ liệu
        self.normalize_cb = QCheckBox("Chuẩn hóa dữ liệu")
        self.normalize_cb.setChecked(True)
        self.normalize_cb.stateChanged.connect(self._update_plot)
        toolbar_layout.addWidget(self.normalize_cb)

        # Checkbox hiển thị tham chiếu
        self.show_reference_cb = QCheckBox("Hiển thị tham chiếu")
        self.show_reference_cb.setEnabled(False)
        self.show_reference_cb.stateChanged.connect(self._update_plot)
        toolbar_layout.addWidget(self.show_reference_cb)

        toolbar_layout.addStretch()

        layout.addLayout(toolbar_layout)

        # Figure và canvas cho đồ thị Matplotlib
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Thanh công cụ Matplotlib
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.canvas)
        layout.addWidget(self.mpl_toolbar)

    def set_solution(self, solution: ParetoSolution):
        """Thiết lập giải pháp để hiển thị."""
        self.solution = solution

        if solution:
            # Lấy danh sách các mục tiêu và giá trị
            self.objectives = list(solution.objective_values.keys())
            self.values = [solution.objective_values[obj] for obj in self.objectives]

            self._update_plot()

    def set_reference_solution(self, reference: ParetoSolution):
        """Thiết lập giải pháp tham chiếu để so sánh."""
        self.reference_solution = reference

        if reference:
            # Lấy giá trị cho các mục tiêu
            self.reference_values = []
            for obj in self.objectives:
                if obj in reference.objective_values:
                    self.reference_values.append(reference.objective_values[obj])
                else:
                    self.reference_values.append(0.0)

            self.show_reference_cb.setEnabled(True)
        else:
            self.reference_values = []
            self.show_reference_cb.setEnabled(False)

        self._update_plot()

    def _update_plot(self):
        """Cập nhật đồ thị radar."""
        if not self.solution or not self.objectives:
            self.figure.clear()
            self.canvas.draw()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111, polar=True)

        # Chuẩn bị dữ liệu
        num_vars = len(self.objectives)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Đóng vòng tròn
        angles += angles[:1]

        # Chuẩn hóa dữ liệu nếu cần
        if self.normalize_cb.isChecked():
            # Tìm min/max cho mỗi mục tiêu
            values_min = []
            values_max = []

            for i, obj in enumerate(self.objectives):
                all_values = [self.values[i]]
                if (
                    self.reference_solution
                    and self.show_reference_cb.isChecked()
                    and i < len(self.reference_values)
                ):
                    all_values.append(self.reference_values[i])

                values_min.append(min(all_values))
                values_max.append(max(all_values))

            # Chuẩn hóa giá trị
            normalized_values = []
            for i, val in enumerate(self.values):
                if values_max[i] != values_min[i]:
                    norm_val = (val - values_min[i]) / (values_max[i] - values_min[i])
                else:
                    norm_val = 0.5  # Giá trị mặc định nếu min = max
                normalized_values.append(norm_val)

            # Chuẩn hóa giá trị tham chiếu nếu có
            normalized_ref_values = []
            if self.reference_values:
                for i, val in enumerate(self.reference_values):
                    if i < len(values_max) and values_max[i] != values_min[i]:
                        norm_val = (val - values_min[i]) / (
                            values_max[i] - values_min[i]
                        )
                    else:
                        norm_val = 0.5
                    normalized_ref_values.append(norm_val)

            # Đóng vòng tròn
            normalized_values += normalized_values[:1]
            if normalized_ref_values:
                normalized_ref_values += normalized_ref_values[:1]
        else:
            # Sử dụng giá trị nguyên
            normalized_values = self.values + self.values[:1]
            if self.reference_values:
                normalized_ref_values = (
                    self.reference_values + self.reference_values[:1]
                )
            else:
                normalized_ref_values = []

        # Đóng vòng tròn cho các góc
        names = self.objectives + [self.objectives[0]]

        # Vẽ biểu đồ radar
        ax.fill(angles, normalized_values, color="red", alpha=0.2)
        ax.plot(angles, normalized_values, color="red", linewidth=2, linestyle="solid")
        ax.scatter(angles, normalized_values, color="red", s=100)

        # Vẽ giải pháp tham chiếu nếu có
        if (
            self.reference_solution
            and self.show_reference_cb.isChecked()
            and normalized_ref_values
        ):
            ax.fill(angles, normalized_ref_values, color="blue", alpha=0.1)
            ax.plot(
                angles,
                normalized_ref_values,
                color="blue",
                linewidth=2,
                linestyle="dashed",
            )
            ax.scatter(angles, normalized_ref_values, color="blue", s=100)

        # Thiết lập các tham số biểu đồ
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names[:-1])

        # Tiêu đề
        solution_name = getattr(self.solution, "name", None) or getattr(
            self.solution, "id", "Solution"
        )
        self.figure.suptitle(f"Giải pháp: {solution_name}", fontsize=16)

        self.canvas.draw()

    def clear(self):
        """Xóa tất cả dữ liệu và làm mới đồ thị."""
        self.solution = None
        self.reference_solution = None
        self.objectives = []
        self.values = []
        self.reference_values = []

        self.figure.clear()
        self.canvas.draw()

    def save_figure(self, filepath: str):
        """Lưu hình ảnh đồ thị hiện tại."""
        if filepath:
            self.figure.savefig(
                filepath, dpi=300, bbox_inches="tight", format=filepath.split(".")[-1]
            )
            logger.info(f"Đã lưu hình ảnh đồ thị Radar vào {filepath}")
