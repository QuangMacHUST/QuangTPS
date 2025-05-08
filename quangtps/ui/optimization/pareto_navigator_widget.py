#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Widget điều hướng Pareto nhẹ cho tối ưu hóa đa tiêu chí.

Module này cung cấp một giao diện người dùng đồ họa nhẹ và hiệu quả để khám phá
và điều hướng không gian giải pháp Pareto trong quy trình tối ưu hóa đa tiêu chí (MCO).
Được thiết kế để tích hợp dễ dàng với các ứng dụng hiện có và hỗ trợ nhiều loại
tối ưu hóa phức tạp.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable

try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QGroupBox,
        QComboBox,
        QTabWidget,
        QSplitter,
        QMessageBox,
        QDialog,
        QDialogButtonBox,
        QListWidget,
        QListWidgetItem,
    )
except ImportError:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QGroupBox,
        QComboBox,
        QTabWidget,
        QSplitter,
        QMessageBox,
        QDialog,
        QDialogButtonBox,
        QListWidget,
        QListWidgetItem,
    )

from quangtps.optimization.mco.pareto_navigator import ParetoNavigator, ParetoSolution
from quangtps.core.planning import Plan
from quangtps.ui.mco_navigator_widget import (
    ParetoFigureCanvas,
    ObjectiveWeightEditor,
    SolutionDetailsPanel,
)

logger = logging.getLogger(__name__)


class ParetoNavigatorLightWidget(QWidget):
    """
    Lightweight widget cho điều hướng Pareto trong tối ưu hóa đa tiêu chí.

    Widget này cung cấp một giao diện đơn giản và hiệu quả để khám phá
    không gian giải pháp Pareto, tập trung vào trải nghiệm người dùng.
    """

    plan_created = pyqtSignal(Plan)  # Phát tín hiệu khi tạo kế hoạch từ giải pháp

    def __init__(self, parent=None):
        super().__init__(parent)
        self.navigator = None
        self.current_solution = None
        self._setup_ui()

    def set_navigator(self, navigator: ParetoNavigator):
        """Thiết lập đối tượng ParetoNavigator để sử dụng."""
        self.navigator = navigator

        if navigator:
            # Lấy danh sách mục tiêu
            objectives = []
            if hasattr(navigator.pareto_surface, "objectives"):
                objectives = list(navigator.pareto_surface.objectives.keys())

            # Thiết lập trọng số
            weights = {}
            if objectives:
                equal_weight = 1.0 / len(objectives)
                weights = {obj: equal_weight for obj in objectives}

            # Cập nhật giao diện
            self.weight_editor.set_objectives(objectives, weights)
            self.figure_canvas.selected_objectives = (
                objectives[:2] if len(objectives) >= 2 else objectives
            )

            # Hiển thị bề mặt Pareto
            self._update_pareto_view()

            logger.info(f"Đã thiết lập ParetoNavigator với {len(objectives)} mục tiêu")

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Tạo splitter chính
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)

        # Panel bên trái
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Widget điều chỉnh trọng số
        self.weight_editor = ObjectiveWeightEditor()
        left_layout.addWidget(self.weight_editor)

        # Thêm các nút điều khiển
        button_layout = QVBoxLayout()

        # Nút chọn theo trọng số
        find_btn = QPushButton("Tìm giải pháp tối ưu")
        find_btn.clicked.connect(self._find_solution)
        button_layout.addWidget(find_btn)

        # Nút hiển thị giải pháp lân cận
        neighbors_btn = QPushButton("Xem giải pháp lân cận")
        neighbors_btn.clicked.connect(self._show_neighbors)
        button_layout.addWidget(neighbors_btn)

        # Nút áp dụng giải pháp
        apply_btn = QPushButton("Áp dụng giải pháp")
        apply_btn.clicked.connect(self._apply_solution)
        button_layout.addWidget(apply_btn)

        left_layout.addLayout(button_layout)
        left_layout.addStretch()

        # Panel bên phải
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Canvas hiển thị mặt Pareto
        self.figure_canvas = ParetoFigureCanvas()
        right_layout.addWidget(self.figure_canvas)

        # Chi tiết giải pháp
        self.details_panel = SolutionDetailsPanel()
        self.details_panel.apply_btn.clicked.connect(self._apply_solution)

        # Thiết lập các panel trong splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.addWidget(self.details_panel)

        # Thiết lập kích thước tương đối
        main_splitter.setSizes([200, 400, 200])

        # Kết nối tín hiệu
        self._connect_signals()

    def _connect_signals(self):
        """Kết nối các tín hiệu với slots."""
        self.weight_editor.weights_changed.connect(self._on_weights_changed)
        self.figure_canvas.clicked_point.connect(self._on_solution_selected)

    def _on_weights_changed(self, weights: Dict[str, float]):
        """Xử lý khi trọng số thay đổi."""
        if self.navigator:
            self.navigator.set_objective_weights(weights)

    def _find_solution(self):
        """Tìm giải pháp tối ưu dựa trên trọng số hiện tại."""
        if not self.navigator:
            QMessageBox.warning(self, "Lỗi", "Chưa thiết lập ParetoNavigator")
            return

        # Chọn giải pháp dựa trên trọng số hiện tại
        solution = self.navigator.select_solution_by_weights()

        if solution:
            self.current_solution = solution
            self.details_panel.display_solution(solution)
            self._update_pareto_view()
        else:
            QMessageBox.warning(
                self,
                "Không tìm thấy",
                "Không tìm thấy giải pháp phù hợp với trọng số đã cho",
            )

    def _on_solution_selected(self, solution_id: str):
        """Xử lý khi người dùng chọn một giải pháp từ đồ thị."""
        if not self.navigator:
            return

        solution = self.navigator.select_solution_by_id(solution_id)

        if solution:
            self.current_solution = solution
            self.details_panel.display_solution(solution)

            # Cập nhật trọng số
            if solution.weights:
                self.weight_editor.is_updating = True

                for obj, weight in solution.weights.items():
                    if obj in self.weight_editor.sliders:
                        self.weight_editor.sliders[obj].setValue(int(weight * 100))
                        self.weight_editor.weight_labels[obj].setText(f"{weight:.2f}")

                self.weight_editor.is_updating = False

    def _show_neighbors(self):
        """Hiển thị các giải pháp lân cận với giải pháp hiện tại."""
        if not self.navigator or not self.current_solution:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một giải pháp trước")
            return

        # Lấy các giải pháp lân cận
        neighbors = self.navigator.get_neighboring_solutions(5)

        if not neighbors:
            QMessageBox.information(
                self, "Thông báo", "Không tìm thấy giải pháp lân cận"
            )
            return

        # Hiển thị dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Giải pháp lân cận")

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Chọn một giải pháp lân cận:"))

        # Danh sách giải pháp
        solutions_list = QListWidget()

        # Thêm giải pháp hiện tại
        current_item = QListWidgetItem(
            f"Giải pháp hiện tại - ID: {self.current_solution.id}"
        )
        current_item.setData(Qt.UserRole, self.current_solution.id)
        solutions_list.addItem(current_item)

        # Thêm các giải pháp lân cận
        for i, neighbor in enumerate(neighbors):
            # Tính điểm tổng hợp
            score = sum(
                value * neighbor.weights.get(obj, 0.0)
                for obj, value in neighbor.objective_values.items()
            )

            item = QListWidgetItem(
                f"Lân cận {i + 1} - ID: {neighbor.id} (Điểm: {score:.4f})"
            )
            item.setData(Qt.UserRole, neighbor.id)
            solutions_list.addItem(item)

        layout.addWidget(solutions_list)

        # Nút
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # Hiển thị dialog
        if dialog.exec() == QDialog.Accepted:
            selected = solutions_list.selectedItems()
            if selected:
                solution_id = selected[0].data(Qt.UserRole)
                self._on_solution_selected(solution_id)

    def _apply_solution(self):
        """Áp dụng giải pháp hiện tại để tạo kế hoạch."""
        if not self.navigator or not self.current_solution:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một giải pháp trước")
            return

        # Tạo kế hoạch
        plan = self.navigator.create_plan_from_current_solution()

        if plan:
            self.plan_created.emit(plan)
            QMessageBox.information(
                self, "Thành công", "Đã tạo kế hoạch từ giải pháp đã chọn"
            )
        else:
            QMessageBox.warning(
                self, "Lỗi", "Không thể tạo kế hoạch từ giải pháp đã chọn"
            )

    def _update_pareto_view(self):
        """Cập nhật hiển thị mặt Pareto."""
        if not self.navigator or not hasattr(
            self.navigator.pareto_surface, "solutions"
        ):
            return

        # Lấy tất cả giải pháp
        solutions = self.navigator.pareto_surface.solutions
        self.figure_canvas.solutions = {sol.id: sol for sol in solutions}

        # Vẽ mặt Pareto
        # (Sử dụng phương thức có sẵn trong ParetoFigureCanvas)
        self.figure_canvas.axes.clear()

        # Chọn hai mục tiêu đầu tiên nếu có
        objectives = self.figure_canvas.selected_objectives
        if len(objectives) < 2:
            return

        x_obj, y_obj = objectives[:2]

        # Thu thập dữ liệu
        x_values = []
        y_values = []

        for sol in solutions:
            if x_obj in sol.objective_values and y_obj in sol.objective_values:
                x_values.append(sol.objective_values[x_obj])
                y_values.append(sol.objective_values[y_obj])

        # Vẽ mặt Pareto
        self.figure_canvas.axes.scatter(x_values, y_values, c="b", marker="o")

        # Đánh dấu giải pháp hiện tại
        if self.current_solution:
            if (
                x_obj in self.current_solution.objective_values
                and y_obj in self.current_solution.objective_values
            ):
                x = self.current_solution.objective_values[x_obj]
                y = self.current_solution.objective_values[y_obj]
                self.figure_canvas.axes.scatter([x], [y], c="r", marker="*", s=100)

        # Cập nhật trục
        self.figure_canvas.axes.set_xlabel(x_obj)
        self.figure_canvas.axes.set_ylabel(y_obj)

        # Cập nhật canvas
        self.figure_canvas.fig.tight_layout()
        self.figure_canvas.draw()


def create_pareto_navigator_light_widget(
    navigator: Optional[ParetoNavigator] = None,
) -> ParetoNavigatorLightWidget:
    """Tạo widget điều hướng Pareto nhẹ."""
    widget = ParetoNavigatorLightWidget()

    if navigator:
        widget.set_navigator(navigator)

    return widget
