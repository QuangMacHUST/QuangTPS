#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Widget MCO Navigator cho giao diện Eclipse-style.

Module này cung cấp một wrapper cho MCONavigator để tích hợp vào giao diện
Eclipse-style của QuangTPS. Widget này tích hợp cả bảng giải pháp và biểu đồ
Pareto 3D trong một giao diện thống nhất, với phong cách thiết kế giống Eclipse.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Set

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSplitter,
        QGroupBox,
        QTableWidget,
        QTableWidgetItem,
        QFrame,
        QComboBox,
        QCheckBox,
        QHeaderView,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QColor, QBrush

    HAS_PYQT = True
except ImportError:
    logger.warning("PyQt5 không khả dụng, MCO Navigator Widget sẽ bị tắt")
    HAS_PYQT = False

# Thử import PySide6 nếu PyQt5 không khả dụng
if not HAS_PYQT:
    try:
        from PySide6.QtWidgets import (
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSplitter,
            QGroupBox,
            QTableWidget,
            QTableWidgetItem,
            QFrame,
            QComboBox,
            QCheckBox,
            QHeaderView,
        )
        from PySide6.QtCore import Qt, Signal as pyqtSignal, Slot as pyqtSlot
        from PySide6.QtGui import QColor, QBrush

        HAS_PYQT = True
        logger.info("Sử dụng PySide6 thay thế cho PyQt5")
    except ImportError:
        logger.warning("PySide6 cũng không khả dụng")

try:
    from quangtps.optimization.mco.mco_navigator import (
        MCONavigator,
        ParetoSolution,
        ParetoSolutionType,
    )
    from quangtps.optimization.mco.mco_pareto_3d_widget import (
        Pareto3DWidget,
        create_pareto_3d_widget,
    )

    HAS_MCO_MODULES = True
except ImportError:
    logger.warning("Không thể import các module MCO, widget sẽ bị tắt")
    HAS_MCO_MODULES = False


class MCONavigatorWidget(QWidget):
    """
    Widget giao diện Eclipse-style cho MCO Navigator.

    Một widget tích hợp cả bảng giải pháp và biểu đồ Pareto 3D trong một giao diện
    thống nhất, với phong cách thiết kế giống Eclipse. Widget này sử dụng splitter
    để cho phép người dùng điều chỉnh kích thước các phần khác nhau của giao diện.

    Attributes
    ----------
    solution_selected_signal : pyqtSignal
        Tín hiệu phát ra khi người dùng chọn một giải pháp
    """

    solution_selected_signal = pyqtSignal(str)  # Phát ra solution_id khi chọn

    def __init__(self, parent=None, mco_navigator=None):
        """
        Khởi tạo widget MCO Navigator.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, mặc định là None
        mco_navigator : MCONavigator, optional
            Đối tượng MCO Navigator, nếu None sẽ tạo mới
        """
        if not HAS_PYQT or not HAS_MCO_MODULES:
            return

        super().__init__(parent)

        self.mco_navigator = mco_navigator or MCONavigator()
        self.pareto_3d_widget = None

        self._setup_ui()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Tạo splitter dọc chính
        vertical_splitter = QSplitter(Qt.Vertical)

        # Phần trên: Bảng giải pháp
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(5, 5, 5, 5)

        # Tiêu đề và các nút điều khiển
        header_layout = QHBoxLayout()
        title_label = QLabel("<b>Pareto Solutions Explorer</b>")
        title_label.setStyleSheet("font-size: 12px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # Thêm nút điều khiển
        self.compute_btn = QPushButton("Calculate Anchor Plans")
        self.compute_btn.setToolTip("Calculate anchor plans for each objective")
        self.compute_btn.clicked.connect(self.on_compute_anchor_points)
        header_layout.addWidget(self.compute_btn)

        self.save_btn = QPushButton("Save Solution")
        self.save_btn.setToolTip("Save current solution")
        self.save_btn.clicked.connect(self.on_save_current_solution)
        header_layout.addWidget(self.save_btn)

        top_layout.addLayout(header_layout)

        # Bảng giải pháp
        self.solutions_table = QTableWidget()
        self.solutions_table.setColumnCount(4)
        self.solutions_table.setHorizontalHeaderLabels(
            ["ID", "Type", "Score", "Details"]
        )
        self.solutions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.solutions_table.setSelectionMode(QTableWidget.SingleSelection)
        self.solutions_table.itemSelectionChanged.connect(self.on_solution_selected)
        # Tự động điều chỉnh kích thước cột
        self.solutions_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )

        top_layout.addWidget(self.solutions_table)

        # Khung tùy chọn hiển thị
        options_layout = QHBoxLayout()

        self.show_pareto_only_checkbox = QCheckBox("Show Pareto optimal solutions only")
        self.show_pareto_only_checkbox.setChecked(True)
        self.show_pareto_only_checkbox.toggled.connect(self.update_solutions_table)
        options_layout.addWidget(self.show_pareto_only_checkbox)

        self.show_history_checkbox = QCheckBox("Show exploration history")
        self.show_history_checkbox.setChecked(True)
        self.show_history_checkbox.toggled.connect(self.update_3d_view)
        options_layout.addWidget(self.show_history_checkbox)

        options_layout.addStretch()

        top_layout.addLayout(options_layout)

        # Phần dưới: Biểu đồ Pareto 3D
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(5, 5, 5, 5)

        # Thêm Pareto 3D widget nếu có
        try:
            self.pareto_3d_widget = create_pareto_3d_widget()
            if self.pareto_3d_widget:
                self.pareto_3d_widget.point_selected_signal.connect(
                    self.on_pareto_point_selected
                )
                bottom_layout.addWidget(self.pareto_3d_widget)
            else:
                bottom_layout.addWidget(QLabel("Pareto 3D widget unavailable"))
        except Exception as e:
            logger.error(f"Error creating Pareto 3D widget: {e}")
            bottom_layout.addWidget(QLabel(f"Error: {str(e)}"))

        # Thêm các widget vào splitter
        vertical_splitter.addWidget(top_widget)
        vertical_splitter.addWidget(bottom_widget)
        vertical_splitter.setStretchFactor(0, 2)
        vertical_splitter.setStretchFactor(1, 3)

        main_layout.addWidget(vertical_splitter)

        # Thanh trạng thái
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        self.apply_btn = QPushButton("Apply Selected Solution")
        self.apply_btn.clicked.connect(self.on_apply_selected_solution)
        status_layout.addWidget(self.apply_btn)

        main_layout.addLayout(status_layout)

        # Thiết lập kích thước mặc định
        self.setMinimumSize(800, 600)

        # Cập nhật UI ban đầu
        self.update_solutions_table()

    def on_compute_anchor_points(self):
        """Tính toán các điểm neo Pareto."""
        self.status_label.setText("Calculating anchor points...")

        try:
            self.mco_navigator.compute_anchor_points()
            self.update_solutions_table()
            self.update_3d_view()
            self.status_label.setText("Anchor points calculated successfully")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            logger.error(f"Error computing anchor points: {e}")

    def on_save_current_solution(self):
        """Lưu giải pháp hiện tại."""
        if not self.mco_navigator.current_solution:
            self.status_label.setText("No current solution to save")
            return

        solution_id = self.mco_navigator.current_solution.solution_id
        if self.mco_navigator.save_solution(self.mco_navigator.current_solution):
            self.update_solutions_table()
            self.update_3d_view()
            self.status_label.setText(f"Solution {solution_id} saved")
        else:
            self.status_label.setText("Failed to save solution")

    def on_apply_selected_solution(self):
        """Áp dụng giải pháp đã chọn."""
        selected_items = self.solutions_table.selectedItems()
        if not selected_items:
            self.status_label.setText("No solution selected")
            return

        row = selected_items[0].row()
        solution_id = self.solutions_table.item(row, 0).text()

        if self.mco_navigator.apply_solution(solution_id):
            self.status_label.setText(f"Solution {solution_id} applied")
            # Phát tín hiệu để cập nhật kế hoạch
            self.solution_selected_signal.emit(solution_id)
        else:
            self.status_label.setText(f"Failed to apply solution {solution_id}")

    def on_solution_selected(self):
        """Xử lý khi người dùng chọn một giải pháp từ bảng."""
        selected_items = self.solutions_table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        solution_id = self.solutions_table.item(row, 0).text()

        if solution_id in self.mco_navigator.solutions:
            # Cập nhật giải pháp hiện tại
            self.mco_navigator.apply_solution(solution_id)

            # Cập nhật hiển thị 3D
            if self.pareto_3d_widget:
                self.pareto_3d_widget.set_current_solution(solution_id)

            self.status_label.setText(f"Solution {solution_id} selected")

    def on_pareto_point_selected(self, solution_id):
        """Xử lý khi một điểm trên biểu đồ Pareto 3D được chọn."""
        # Cập nhật lựa chọn trong bảng
        for row in range(self.solutions_table.rowCount()):
            if self.solutions_table.item(row, 0).text() == solution_id:
                self.solutions_table.selectRow(row)
                break

        # Áp dụng giải pháp
        if solution_id in self.mco_navigator.solutions:
            self.mco_navigator.apply_solution(solution_id)
            self.status_label.setText(f"Pareto point {solution_id} selected")

    def update_solutions_table(self):
        """Cập nhật bảng giải pháp với dữ liệu mới nhất."""
        self.solutions_table.clearContents()

        # Lọc các giải pháp hiển thị
        solutions = self.mco_navigator.solutions
        if self.show_pareto_only_checkbox.isChecked():
            # TODO: Lọc thật sự các giải pháp Pareto tối ưu
            # Hiện tại lọc đơn giản dựa trên loại giải pháp
            solutions = {
                k: v
                for k, v in solutions.items()
                if v.solution_type
                in [ParetoSolutionType.ANCHOR, ParetoSolutionType.BALANCED]
            }

        # Cập nhật bảng
        self.solutions_table.setRowCount(len(solutions))

        for i, (solution_id, solution) in enumerate(solutions.items()):
            # ID
            id_item = QTableWidgetItem(solution_id)
            self.solutions_table.setItem(i, 0, id_item)

            # Loại
            type_item = QTableWidgetItem(solution.solution_type.value)
            self.solutions_table.setItem(i, 1, type_item)

            # Điểm số tổng hợp (giả lập)
            score = sum(solution.objectives_values.values()) / len(
                solution.objectives_values
            )
            score_item = QTableWidgetItem(f"{score:.2f}")
            self.solutions_table.setItem(i, 2, score_item)

            # Chi tiết mục tiêu
            details = "; ".join(
                f"{k}: {v:.2f}" for k, v in solution.objectives_values.items()
            )
            details_item = QTableWidgetItem(details)
            self.solutions_table.setItem(i, 3, details_item)

            # Tô màu cho giải pháp hiện tại
            if (
                self.mco_navigator.current_solution
                and solution_id == self.mco_navigator.current_solution.solution_id
            ):
                for col in range(self.solutions_table.columnCount()):
                    cell_item = self.solutions_table.item(i, col)
                    if cell_item:
                        cell_item.setBackground(QBrush(QColor(200, 230, 250)))

        self.solutions_table.resizeColumnsToContents()
        # Đảm bảo cột chi tiết không quá rộng
        self.solutions_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )

    def update_3d_view(self):
        """Cập nhật hiển thị 3D Pareto."""
        if not self.pareto_3d_widget:
            return

        # Cập nhật cài đặt hiển thị
        self.pareto_3d_widget.show_history_checkbox.setChecked(
            self.show_history_checkbox.isChecked()
        )
        self.pareto_3d_widget.show_pareto_only_checkbox.setChecked(
            self.show_pareto_only_checkbox.isChecked()
        )

        # Cập nhật dữ liệu
        solutions_dict = {}
        pareto_optimal = {}

        for solution_id, solution in self.mco_navigator.solutions.items():
            solution_data = {
                "objectives": solution.objectives_values,
                "weights": solution.weights,
                "type": solution.solution_type.value,
            }

            solutions_dict[solution_id] = solution_data

            # Xác định giải pháp Pareto tối ưu
            if solution.solution_type in [
                ParetoSolutionType.ANCHOR,
                ParetoSolutionType.BALANCED,
            ]:
                pareto_optimal[solution_id] = solution_data

        self.pareto_3d_widget.set_data(solutions_dict, pareto_optimal)

        # Đặt giải pháp hiện tại
        if self.mco_navigator.current_solution:
            self.pareto_3d_widget.set_current_solution(
                self.mco_navigator.current_solution.solution_id
            )

    def set_mco_navigator(self, mco_navigator):
        """Đặt đối tượng MCO Navigator mới.

        Parameters
        ----------
        mco_navigator : MCONavigator
            Đối tượng MCO Navigator mới
        """
        self.mco_navigator = mco_navigator
        self.update_solutions_table()
        self.update_3d_view()

    def set_objectives(self, objectives):
        """Đặt danh sách mục tiêu mới.

        Parameters
        ----------
        objectives : Dict[str, ObjectiveFunction]
            Từ điển các hàm mục tiêu
        """
        self.mco_navigator.objectives = objectives
        # Cập nhật trên Pareto 3D widget nếu có
        if self.pareto_3d_widget:
            self.pareto_3d_widget.objective_names = list(objectives.keys())
            self.pareto_3d_widget._update_objective_combos()

    def create_sample_data(self):
        """
        Tạo dữ liệu mẫu cho MCO Navigator khi module tính toán không khả dụng.

        Phương thức này tạo ra các giải pháp Pareto mẫu với các giá trị mục tiêu đa dạng
        để mô phỏng một bề mặt Pareto thực tế. Các giải pháp bao gồm:
        - Các điểm neo (anchor points) tối ưu hóa cho từng mục tiêu riêng lẻ
        - Các giải pháp Pareto tối ưu với sự cân bằng khác nhau
        - Các giải pháp không tối ưu Pareto để minh họa quá trình khám phá
        """
        if not self.mco_navigator or not hasattr(self.mco_navigator, "objectives"):
            self.status_label.setText(
                "No objectives defined, cannot create sample data"
            )
            return

        # Xóa dữ liệu hiện có
        self.mco_navigator.solutions.clear()
        self.mco_navigator.current_solution = None

        # Lấy danh sách mục tiêu
        objectives = list(self.mco_navigator.objectives.keys())

        if not objectives:
            self.status_label.setText(
                "No objectives defined, cannot create sample data"
            )
            return

        # Đảm bảo có ít nhất 3 mục tiêu để tạo bề mặt Pareto 3D
        while len(objectives) < 3:
            objectives.append(f"Objective {len(objectives) + 1}")

        # Tạo các điểm neo (anchor points) - tối ưu cho từng mục tiêu riêng lẻ
        for i, obj in enumerate(objectives[:3]):
            solution_id = f"A{i + 1}"

            # Giá trị mục tiêu: tối ưu cho mục tiêu hiện tại, kém cho các mục tiêu khác
            obj_values = {}
            weights = {}

            for j, other_obj in enumerate(objectives[:3]):
                if other_obj == obj:
                    obj_values[other_obj] = 95.0  # Giá trị cao cho mục tiêu được tối ưu
                    weights[other_obj] = 10.0  # Trọng số cao cho mục tiêu được tối ưu
                else:
                    # Giá trị thấp hơn cho các mục tiêu khác
                    obj_values[other_obj] = 40.0 + (j * 5)
                    weights[other_obj] = 1.0

            # Thêm các mục tiêu khác nếu có
            for other_obj in objectives[3:]:
                obj_values[other_obj] = 50.0
                weights[other_obj] = 1.0

            # Tạo giải pháp
            solution = ParetoSolution(
                solution_id=solution_id,
                solution_type=ParetoSolutionType.ANCHOR,
                objectives_values=obj_values,
                weights=weights,
                is_pareto_optimal=True,
                metadata={
                    "description": f"Anchor point optimizing {obj}",
                    "creation_time": "2023-08-05 10:00:00",
                    "computation_time": "5.2 seconds",
                },
            )

            # Thêm vào danh sách giải pháp
            self.mco_navigator.solutions[solution_id] = solution

        # Tạo các giải pháp Pareto tối ưu với sự cân bằng khác nhau
        for i in range(15):
            solution_id = f"P{i + 1}"

            # Tạo giá trị mục tiêu và trọng số ngẫu nhiên nhưng vẫn đảm bảo tính Pareto
            obj_values = {}
            weights = {}

            # Tạo một cân bằng khác nhau giữa các mục tiêu
            balance_factor = i / 14.0  # 0.0 đến 1.0

            # Tính giá trị cho 3 mục tiêu đầu tiên để tạo bề mặt Pareto
            if len(objectives) >= 3:
                # Mục tiêu 1: giảm dần từ 95 xuống 60
                obj_values[objectives[0]] = 95.0 - (balance_factor * 35.0)
                weights[objectives[0]] = 10.0 - (balance_factor * 8.0)

                # Mục tiêu 2: tăng dần từ 60 lên 90
                obj_values[objectives[1]] = 60.0 + (balance_factor * 30.0)
                weights[objectives[1]] = 2.0 + (balance_factor * 8.0)

                # Mục tiêu 3: hình parabol, cao ở giữa
                parabola_factor = 4.0 * (balance_factor - 0.5) ** 2
                obj_values[objectives[2]] = 85.0 - (parabola_factor * 25.0)
                weights[objectives[2]] = 5.0

            # Thêm các mục tiêu khác nếu có
            for j, obj in enumerate(objectives[3:], start=3):
                obj_values[obj] = 50.0 + (balance_factor * 20.0) + (j * 2.0)
                weights[obj] = 3.0

            # Tạo giải pháp
            solution = ParetoSolution(
                solution_id=solution_id,
                solution_type=ParetoSolutionType.PARETO_OPTIMAL,
                objectives_values=obj_values,
                weights=weights,
                is_pareto_optimal=True,
                metadata={
                    "description": f"Balanced solution {i + 1}",
                    "creation_time": f"2023-08-05 {10 + i // 2}:{(i % 2) * 30:02d}:00",
                    "computation_time": f"{2.0 + i / 5:.1f} seconds",
                    "balance_factor": balance_factor,
                },
            )

            # Thêm vào danh sách giải pháp
            self.mco_navigator.solutions[solution_id] = solution

        # Tạo một số giải pháp không tối ưu Pareto để minh họa quá trình khám phá
        for i in range(8):
            solution_id = f"N{i + 1}"

            # Giá trị mục tiêu kém hơn các giải pháp Pareto
            obj_values = {}
            weights = {}

            for j, obj in enumerate(objectives[:3]):
                # Giá trị thấp hơn 10-20% so với giải pháp Pareto
                obj_values[obj] = 50.0 + (i * 5.0) - (j * 3.0)
                weights[obj] = 3.0 + (i % 3)

            # Thêm các mục tiêu khác nếu có
            for obj in objectives[3:]:
                obj_values[obj] = 40.0 + (i * 3.0)
                weights[obj] = 2.0

            # Tạo giải pháp
            solution = ParetoSolution(
                solution_id=solution_id,
                solution_type=ParetoSolutionType.INTERMEDIATE,
                objectives_values=obj_values,
                weights=weights,
                is_pareto_optimal=False,
                metadata={
                    "description": f"Non-optimal solution {i + 1}",
                    "creation_time": f"2023-08-05 09:{i * 5:02d}:00",
                    "computation_time": f"{1.5 + i / 10:.1f} seconds",
                },
            )

            # Thêm vào danh sách giải pháp
            self.mco_navigator.solutions[solution_id] = solution

        # Tạo giải pháp hiện tại (current solution)
        if self.mco_navigator.solutions:
            # Chọn một giải pháp Pareto làm giải pháp hiện tại
            pareto_solutions = [
                s
                for s in self.mco_navigator.solutions.values()
                if s.is_pareto_optimal
                and s.solution_type == ParetoSolutionType.PARETO_OPTIMAL
            ]
            if pareto_solutions:
                self.mco_navigator.current_solution = pareto_solutions[
                    len(pareto_solutions) // 2
                ]

        # Cập nhật giao diện
        self.update_solutions_table()

        # Cập nhật biểu đồ Pareto 3D nếu có
        if self.pareto_3d_widget:
            try:
                self.pareto_3d_widget.set_solutions(self.mco_navigator.solutions)
                self.pareto_3d_widget.set_current_solution(
                    self.mco_navigator.current_solution
                )
                self.update_3d_view()
            except Exception as e:
                logger.error(f"Error updating Pareto 3D view: {e}")

        self.status_label.setText("Sample data created successfully")


# Hàm tiện ích để tạo widget
def create_mco_navigator_widget(parent=None, **kwargs):
    """
    Tạo và trả về widget MCO Navigator.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha, mặc định là None
    **kwargs :
        Tham số truyền cho MCONavigator

    Returns
    -------
    MCONavigatorWidget or None
        Widget MCO Navigator hoặc None nếu không thể tạo
    """
    if not HAS_PYQT or not HAS_MCO_MODULES:
        logger.error("Cannot create MCO Navigator Widget: missing dependencies")
        return None

    try:
        from quangtps.optimization.mco.mco_navigator import MCONavigator

        mco_navigator = MCONavigator(**kwargs)
        widget = MCONavigatorWidget(parent=parent, mco_navigator=mco_navigator)
        return widget
    except Exception as e:
        logger.error(f"Error creating MCO Navigator Widget: {e}")
        return None


# Test code khi chạy trực tiếp
if __name__ == "__main__":
    import sys
    import numpy as np

    if not HAS_PYQT or not HAS_MCO_MODULES:
        print("Required modules not available. Test cannot run.")
        sys.exit(1)

    from PyQt5.QtWidgets import QApplication
    from quangtps.optimization.mco.mco_navigator import (
        MCONavigator,
        ParetoSolution,
        ParetoSolutionType,
    )

    app = QApplication(sys.argv)

    # Tạo dữ liệu mẫu
    objectives = {
        "PTV Coverage": None,
        "Brainstem Max": None,
        "Parotid Mean": None,
        "Spinal Cord Max": None,
        "Conformity": None,
    }

    # Tạo MCO Navigator
    mco_navigator = MCONavigator(objectives=objectives)

    # Tạo điểm neo mẫu
    for obj_name in objectives:
        # Trọng số
        weights = {o: 0.01 for o in objectives}
        weights[obj_name] = 1.0

        # Giá trị mục tiêu
        obj_values = {o: np.random.random() * 100 for o in objectives}
        obj_values[obj_name] = np.random.random() * 20  # Tốt hơn cho mục tiêu này

        solution = ParetoSolution(
            solution_id=f"anchor_{obj_name}",
            objectives_values=obj_values,
            weights=weights,
            solution_type=ParetoSolutionType.ANCHOR,
        )

        mco_navigator.solutions[solution.solution_id] = solution

    # Tạo điểm cân bằng mẫu
    weights = {o: 1.0 / len(objectives) for o in objectives}
    obj_values = {o: np.random.random() * 50 + 25 for o in objectives}

    solution = ParetoSolution(
        solution_id="balanced",
        objectives_values=obj_values,
        weights=weights,
        solution_type=ParetoSolutionType.BALANCED,
    )

    mco_navigator.solutions[solution.solution_id] = solution
    mco_navigator.current_solution = solution

    # Tạo và hiển thị widget
    widget = MCONavigatorWidget(mco_navigator=mco_navigator)
    widget.show()

    # Tạo dữ liệu mẫu
    widget.create_sample_data()

    sys.exit(app.exec_())
