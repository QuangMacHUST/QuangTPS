#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cung cấp widget hiển thị 3D cho bề mặt Pareto trong MCO Navigator.

Module này triển khai widget hiển thị biểu đồ 3D tương tác cho bề mặt Pareto,
tương tự như Eclipse MCO Navigator của Varian. Người dùng có thể xoay, zoom
và chọn điểm trên bề mặt Pareto để khám phá các lựa chọn tối ưu hóa khác nhau.
"""

import logging
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QComboBox,
        QLabel,
        QPushButton,
        QCheckBox,
        QFrame,
        QSizePolicy,
        QSplitter,
    )
    from PyQt5.QtCore import Qt, pyqtSignal

    HAS_PYQT = True
except ImportError:
    logger.warning("PyQt5 không khả dụng, widget Pareto 3D sẽ bị tắt")
    HAS_PYQT = False

# Thử import PySide6 nếu PyQt5 không khả dụng
if not HAS_PYQT:
    try:
        from PySide6.QtWidgets import (
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QComboBox,
            QLabel,
            QPushButton,
            QCheckBox,
            QFrame,
            QSizePolicy,
            QSplitter,
        )
        from PySide6.QtCore import Qt, Signal as pyqtSignal

        HAS_PYQT = True
        logger.info("Sử dụng PySide6 thay thế cho PyQt5")
    except ImportError:
        logger.warning("PySide6 cũng không khả dụng")


class Pareto3DWidget(QWidget):
    """
    Widget hiển thị bề mặt Pareto 3D tương tác cho MCO Navigator.

    Widget này hiển thị một biểu đồ 3D tương tác cho bề mặt Pareto, cho phép
    khám phá các kế hoạch tối ưu Pareto trong không gian đa mục tiêu. Tính năng
    bao gồm:
    - Biểu đồ 3D tương tác với quay, zoom và pan
    - Tùy chọn chọn 3 trục cho các mục tiêu khác nhau
    - Hiển thị điểm hiện tại trên bề mặt Pareto
    - Hiển thị lịch sử di chuyển trên bề mặt Pareto

    Attributes
    ----------
    point_selected_signal : pyqtSignal
        Tín hiệu phát ra khi một điểm trên bề mặt Pareto được chọn
    """

    point_selected_signal = pyqtSignal(str)  # Phát ra solution_id khi chọn

    def __init__(
        self, parent=None, objective_names: List[str] = None, colormap: str = "viridis"
    ):
        """
        Khởi tạo widget biểu đồ Pareto 3D.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, mặc định là None
        objective_names : List[str], optional
            Danh sách tên các mục tiêu, mặc định là None
        colormap : str, optional
            Tên của colormap matplotlib, mặc định là "viridis"
        """
        if not HAS_PYQT:
            return

        super().__init__(parent)

        self.objective_names = objective_names or []
        self.colormap = colormap
        self.solutions = {}  # Dict lưu các giải pháp Pareto
        self.pareto_optimal_solutions = {}  # Dict lưu các giải pháp Pareto tối ưu
        self.current_solution_id = None
        self.history = []  # Lịch sử các giải pháp đã xem

        # Các trục đang hiển thị
        self.selected_objectives = {
            "x": None,
            "y": None,
            "z": None,
            "color": None,  # Mục tiêu biểu thị bằng màu sắc
        }

        self._setup_ui()
        self._setup_empty_plot()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng cho widget."""
        main_layout = QVBoxLayout(self)

        # Tạo ComboBox để chọn các trục
        axis_layout = QHBoxLayout()

        # Trục X
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_combo = QComboBox()
        x_layout.addWidget(self.x_combo)
        axis_layout.addLayout(x_layout)

        # Trục Y
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_combo = QComboBox()
        y_layout.addWidget(self.y_combo)
        axis_layout.addLayout(y_layout)

        # Trục Z
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.z_combo = QComboBox()
        z_layout.addWidget(self.z_combo)
        axis_layout.addLayout(z_layout)

        # Trục màu (mục tiêu biểu thị bằng màu)
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Màu:"))
        self.color_combo = QComboBox()
        color_layout.addWidget(self.color_combo)
        axis_layout.addLayout(color_layout)

        # Thêm layout chọn trục vào layout chính
        main_layout.addLayout(axis_layout)

        # Tạo biểu đồ matplotlib
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Kết nối sự kiện
        self.x_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.y_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.z_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.color_combo.currentIndexChanged.connect(self._on_axis_changed)

        # Thêm các tùy chọn hiển thị
        options_layout = QHBoxLayout()

        self.show_history_checkbox = QCheckBox("Hiển thị lịch sử")
        self.show_history_checkbox.setChecked(True)
        self.show_history_checkbox.toggled.connect(self._update_plot)
        options_layout.addWidget(self.show_history_checkbox)

        self.show_pareto_only_checkbox = QCheckBox("Chỉ hiện Pareto tối ưu")
        self.show_pareto_only_checkbox.setChecked(True)
        self.show_pareto_only_checkbox.toggled.connect(self._update_plot)
        options_layout.addWidget(self.show_pareto_only_checkbox)

        self.show_labels_checkbox = QCheckBox("Hiển thị nhãn")
        self.show_labels_checkbox.setChecked(False)
        self.show_labels_checkbox.toggled.connect(self._update_plot)
        options_layout.addWidget(self.show_labels_checkbox)

        options_layout.addStretch()

        # Nút làm mới biểu đồ
        refresh_button = QPushButton("Làm mới")
        refresh_button.clicked.connect(self._update_plot)
        options_layout.addWidget(refresh_button)

        # Thêm layout tùy chọn vào layout chính
        main_layout.addLayout(options_layout)

        # Thêm biểu đồ và thanh công cụ
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        # Kết nối sự kiện click chuột trên biểu đồ
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        # Cập nhật danh sách mục tiêu trong combobox
        self._update_objective_combos()

    def _setup_empty_plot(self):
        """Thiết lập biểu đồ trống ban đầu."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Bề mặt Pareto 3D")
        self.canvas.draw()

    def _update_objective_combos(self):
        """Cập nhật danh sách mục tiêu trong các combo box."""
        if not self.objective_names:
            return

        # Lưu lại các lựa chọn hiện tại
        current_x = self.x_combo.currentText() if self.x_combo.count() > 0 else None
        current_y = self.y_combo.currentText() if self.y_combo.count() > 0 else None
        current_z = self.z_combo.currentText() if self.z_combo.count() > 0 else None
        current_color = (
            self.color_combo.currentText() if self.color_combo.count() > 0 else None
        )

        # Xóa và thêm lại các mục
        self.x_combo.clear()
        self.y_combo.clear()
        self.z_combo.clear()
        self.color_combo.clear()

        # Thêm mục "Không có" cho trục màu
        self.color_combo.addItem("Không có")

        # Thêm các mục tiêu
        for obj_name in self.objective_names:
            self.x_combo.addItem(obj_name)
            self.y_combo.addItem(obj_name)
            self.z_combo.addItem(obj_name)
            self.color_combo.addItem(obj_name)

        # Khôi phục lựa chọn hoặc đặt mặc định
        if current_x and current_x in self.objective_names:
            self.x_combo.setCurrentText(current_x)
        elif len(self.objective_names) > 0:
            self.x_combo.setCurrentIndex(0)

        if current_y and current_y in self.objective_names:
            self.y_combo.setCurrentText(current_y)
        elif len(self.objective_names) > 1:
            self.y_combo.setCurrentIndex(1)
        elif len(self.objective_names) > 0:
            self.y_combo.setCurrentIndex(0)

        if current_z and current_z in self.objective_names:
            self.z_combo.setCurrentText(current_z)
        elif len(self.objective_names) > 2:
            self.z_combo.setCurrentIndex(2)
        elif len(self.objective_names) > 0:
            self.z_combo.setCurrentIndex(0)

        if current_color and current_color in self.objective_names:
            self.color_combo.setCurrentText(current_color)
        else:
            self.color_combo.setCurrentIndex(0)  # "Không có"

        # Cập nhật các trục đã chọn
        self._update_selected_objectives()

    def _update_selected_objectives(self):
        """Cập nhật các mục tiêu đã chọn cho mỗi trục."""
        if self.x_combo.count() > 0:
            self.selected_objectives["x"] = self.x_combo.currentText()

        if self.y_combo.count() > 0:
            self.selected_objectives["y"] = self.y_combo.currentText()

        if self.z_combo.count() > 0:
            self.selected_objectives["z"] = self.z_combo.currentText()

        if self.color_combo.count() > 0:
            color_text = self.color_combo.currentText()
            self.selected_objectives["color"] = (
                color_text if color_text != "Không có" else None
            )

    def _on_axis_changed(self):
        """Xử lý khi người dùng thay đổi lựa chọn trục."""
        self._update_selected_objectives()
        self._update_plot()

    def _on_plot_click(self, event):
        """Xử lý khi người dùng click vào biểu đồ."""
        if not event.inaxes or not isinstance(event.inaxes, Axes3D):
            return

        if not self.solutions:
            return

        # Lấy tọa độ điểm click
        x_click, y_click = event.xdata, event.ydata
        z_click = event.zdata if hasattr(event, "zdata") else None

        # Tìm điểm gần nhất
        closest_solution_id = None
        min_distance = float("inf")

        solutions_dict = (
            self.pareto_optimal_solutions
            if self.show_pareto_only_checkbox.isChecked()
            else self.solutions
        )

        x_obj = self.selected_objectives["x"]
        y_obj = self.selected_objectives["y"]
        z_obj = self.selected_objectives["z"]

        for solution_id, solution in solutions_dict.items():
            obj_values = solution.get("objectives", {})

            if x_obj in obj_values and y_obj in obj_values:
                x_sol = obj_values[x_obj]
                y_sol = obj_values[y_obj]

                # Tính khoảng cách 2D hoặc 3D tùy thuộc vào dữ liệu
                if z_obj in obj_values and z_click is not None:
                    z_sol = obj_values[z_obj]
                    distance = np.sqrt(
                        (x_sol - x_click) ** 2
                        + (y_sol - y_click) ** 2
                        + (z_sol - z_click) ** 2
                    )
                else:
                    distance = np.sqrt((x_sol - x_click) ** 2 + (y_sol - y_click) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_solution_id = solution_id

        if closest_solution_id is not None:
            # Cập nhật giải pháp hiện tại và phát tín hiệu
            self.set_current_solution(closest_solution_id)
            self.point_selected_signal.emit(closest_solution_id)

    def set_data(
        self,
        solutions: Dict[str, Dict],
        pareto_optimal_solutions: Dict[str, Dict] = None,
    ):
        """
        Đặt dữ liệu cho biểu đồ Pareto 3D.

        Parameters
        ----------
        solutions : Dict[str, Dict]
            Dictionary các giải pháp, với key là solution_id và value là dict chứa
            thông tin về giải pháp, bao gồm "objectives" là dict các giá trị mục tiêu.
        pareto_optimal_solutions : Dict[str, Dict], optional
            Dictionary các giải pháp Pareto tối ưu, định dạng giống solutions.
            Nếu None, tất cả các giải pháp được coi là Pareto tối ưu.
        """
        self.solutions = solutions
        self.pareto_optimal_solutions = pareto_optimal_solutions or solutions

        # Cập nhật danh sách tên mục tiêu từ dữ liệu
        if solutions and len(solutions) > 0:
            first_solution = next(iter(solutions.values()))
            objectives = first_solution.get("objectives", {})
            self.objective_names = list(objectives.keys())
            self._update_objective_combos()

        # Cập nhật biểu đồ
        self._update_plot()

    def set_current_solution(self, solution_id: str):
        """
        Đặt giải pháp hiện tại.

        Parameters
        ----------
        solution_id : str
            ID của giải pháp cần đặt làm hiện tại
        """
        if solution_id in self.solutions:
            # Thêm vào lịch sử nếu khác giải pháp hiện tại
            if solution_id != self.current_solution_id:
                if self.current_solution_id is not None:
                    self.history.append(self.current_solution_id)
                self.current_solution_id = solution_id

            # Giới hạn kích thước lịch sử
            if len(self.history) > 20:
                self.history = self.history[-20:]

            # Cập nhật biểu đồ
            self._update_plot()

    def add_solution(
        self, solution_id: str, solution_data: Dict, is_pareto_optimal: bool = False
    ):
        """
        Thêm một giải pháp mới vào biểu đồ.

        Parameters
        ----------
        solution_id : str
            ID của giải pháp
        solution_data : Dict
            Dữ liệu của giải pháp, bao gồm "objectives" là dict các giá trị mục tiêu
        is_pareto_optimal : bool, optional
            Có phải giải pháp Pareto tối ưu không, mặc định là False
        """
        self.solutions[solution_id] = solution_data

        if is_pareto_optimal:
            self.pareto_optimal_solutions[solution_id] = solution_data

        # Cập nhật danh sách tên mục tiêu nếu cần
        if not self.objective_names and "objectives" in solution_data:
            self.objective_names = list(solution_data["objectives"].keys())
            self._update_objective_combos()

        # Cập nhật biểu đồ
        self._update_plot()

    def clear_data(self):
        """Xóa tất cả dữ liệu và làm mới biểu đồ."""
        self.solutions = {}
        self.pareto_optimal_solutions = {}
        self.current_solution_id = None
        self.history = []
        self._setup_empty_plot()

    def _update_plot(self):
        """Cập nhật biểu đồ Pareto 3D với dữ liệu hiện tại."""
        if not self.solutions:
            self._setup_empty_plot()
            return

        # Lấy các mục tiêu đã chọn
        x_obj = self.selected_objectives["x"]
        y_obj = self.selected_objectives["y"]
        z_obj = self.selected_objectives["z"]
        color_obj = self.selected_objectives["color"]

        if not x_obj or not y_obj:
            return

        # Xóa biểu đồ hiện tại
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection="3d")

        # Quyết định hiển thị tất cả hoặc chỉ Pareto tối ưu
        solutions_dict = (
            self.pareto_optimal_solutions
            if self.show_pareto_only_checkbox.isChecked()
            else self.solutions
        )

        # Thu thập dữ liệu
        x_values = []
        y_values = []
        z_values = []
        color_values = []
        sizes = []
        labels = []

        for solution_id, solution in solutions_dict.items():
            obj_values = solution.get("objectives", {})

            if (
                x_obj in obj_values
                and y_obj in obj_values
                and (z_obj is None or z_obj in obj_values)
            ):
                x_values.append(obj_values[x_obj])
                y_values.append(obj_values[y_obj])

                # Giá trị Z có thể là hằng số nếu không có mục tiêu z_obj
                if z_obj is not None and z_obj in obj_values:
                    z_values.append(obj_values[z_obj])
                else:
                    z_values.append(0)  # Giá trị mặc định

                # Giá trị màu
                if color_obj is not None and color_obj in obj_values:
                    color_values.append(obj_values[color_obj])
                else:
                    color_values.append(0)  # Giá trị mặc định

                # Kích thước điểm - lớn hơn cho giải pháp hiện tại
                sizes.append(100 if solution_id == self.current_solution_id else 30)

                # Nhãn
                labels.append(solution_id)

        if not x_values or not y_values or not z_values:
            return

        # Chuẩn bị colormap
        try:
            cmap = getattr(cm, self.colormap, cm.viridis)
        except:
            cmap = cm.viridis

        # Tạo biểu đồ scatter
        if color_obj is not None and color_values:
            # Chuẩn hóa giá trị màu
            norm = mpl.colors.Normalize(vmin=min(color_values), vmax=max(color_values))
            scatter = self.ax.scatter(
                x_values,
                y_values,
                z_values,
                c=color_values,
                cmap=cmap,
                norm=norm,
                s=sizes,
                alpha=0.7,
            )
            # Thêm colorbar
            cbar = self.figure.colorbar(scatter)
            cbar.set_label(color_obj)
        else:
            scatter = self.ax.scatter(x_values, y_values, z_values, s=sizes, alpha=0.7)

        # Thêm nhãn nếu cần
        if self.show_labels_checkbox.isChecked():
            for i, label in enumerate(labels):
                self.ax.text(x_values[i], y_values[i], z_values[i], label, size=8)

        # Đánh dấu giải pháp hiện tại với màu sắc đặc biệt
        if self.current_solution_id in solutions_dict:
            current_solution = solutions_dict[self.current_solution_id]
            obj_values = current_solution.get("objectives", {})

            if x_obj in obj_values and y_obj in obj_values:
                x_curr = obj_values[x_obj]
                y_curr = obj_values[y_obj]
                z_curr = obj_values[z_obj] if z_obj in obj_values else 0

                self.ax.scatter(
                    [x_curr], [y_curr], [z_curr], color="red", s=200, marker="*"
                )

        # Hiển thị lịch sử nếu cần
        if self.show_history_checkbox.isChecked() and self.history:
            hist_x = []
            hist_y = []
            hist_z = []

            for hist_id in self.history:
                if hist_id in self.solutions:
                    hist_solution = self.solutions[hist_id]
                    obj_values = hist_solution.get("objectives", {})

                    if x_obj in obj_values and y_obj in obj_values:
                        hist_x.append(obj_values[x_obj])
                        hist_y.append(obj_values[y_obj])
                        hist_z.append(obj_values[z_obj] if z_obj in obj_values else 0)

            if hist_x and hist_y and hist_z:
                # Vẽ đường lịch sử
                self.ax.plot(hist_x, hist_y, hist_z, "r-", alpha=0.5, linewidth=2)

                # Đánh dấu các điểm lịch sử
                self.ax.scatter(hist_x, hist_y, hist_z, color="orange", s=50, alpha=0.5)

        # Đặt nhãn và tiêu đề
        self.ax.set_xlabel(x_obj)
        self.ax.set_ylabel(y_obj)
        self.ax.set_zlabel(z_obj if z_obj else "Z")
        self.ax.set_title("Bề mặt Pareto 3D")

        # Vẽ lại canvas
        self.canvas.draw()


def create_pareto_3d_widget(parent=None, **kwargs):
    """
    Hàm tiện ích để tạo widget Pareto 3D.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha, mặc định là None
    **kwargs : dict
        Các tham số khác cho Pareto3DWidget

    Returns
    -------
    Pareto3DWidget
        Widget Pareto 3D đã được cấu hình
    """
    if not HAS_PYQT:
        logger.error(
            "PyQt5 hoặc PySide6 không khả dụng, không thể tạo widget Pareto 3D"
        )
        return None

    try:
        widget = Pareto3DWidget(parent=parent, **kwargs)
        return widget
    except Exception as e:
        logger.error(f"Lỗi khi tạo widget Pareto 3D: {str(e)}")
        return None


# Code kiểm thử chỉ chạy khi script được thực thi trực tiếp
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    # Khởi tạo QApplication
    app = QApplication(sys.argv)

    # Tạo dữ liệu mẫu
    objective_names = [
        "PTV Coverage",
        "Brainstem Max",
        "Parotid Mean",
        "Spinal Cord Max",
        "Conformity",
    ]

    # Tạo các giải pháp mẫu
    solutions = {}
    for i in range(20):
        solution_id = f"solution_{i}"
        solutions[solution_id] = {
            "objectives": {
                "PTV Coverage": np.random.random() * 100,
                "Brainstem Max": np.random.random() * 50,
                "Parotid Mean": np.random.random() * 30,
                "Spinal Cord Max": np.random.random() * 40,
                "Conformity": np.random.random() * 1.5,
            },
            "weights": {
                "PTV Coverage": np.random.random(),
                "Brainstem Max": np.random.random(),
                "Parotid Mean": np.random.random(),
                "Spinal Cord Max": np.random.random(),
                "Conformity": np.random.random(),
            },
        }

    # Tạo giải pháp Pareto tối ưu mẫu
    pareto_optimal = {}
    for i in range(5):
        solution_id = f"solution_{i}"
        pareto_optimal[solution_id] = solutions[solution_id]

    # Tạo và hiển thị widget
    widget = Pareto3DWidget(objective_names=objective_names)
    widget.set_data(solutions, pareto_optimal)
    widget.set_current_solution("solution_0")
    widget.show()

    sys.exit(app.exec_())
