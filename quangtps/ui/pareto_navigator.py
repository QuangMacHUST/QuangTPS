#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cho Navigator Pareto trong tối ưu hóa đa tiêu chí.

Module này cung cấp các thành phần giao diện người dùng để điều hướng
và trực quan hóa bề mặt Pareto trong tối ưu hóa đa tiêu chí (MCO).
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import time

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QSlider,
        QLabel,
        QPushButton,
        QComboBox,
        QCheckBox,
        QGridLayout,
        QGroupBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
    from PyQt5.QtGui import QColor, QPalette

    has_pyqt5 = True
except ImportError:
    has_pyqt5 = False

    # Định nghĩa các lớp giả mạo khi không có PyQt5
    class QWidget:
        pass

    class pyqtSignal:
        def __init__(self, *args):
            pass


try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    import matplotlib.pyplot as plt

    has_mpl = True
except ImportError:
    has_mpl = False

    # Lớp giả mạo khi không có matplotlib
    class FigureCanvas:
        pass


logger = logging.getLogger(__name__)


class ParetoFigureCanvas(FigureCanvas):
    """Canvas hiển thị đồ thị Pareto."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """
        Khởi tạo canvas hiển thị đồ thị Pareto.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha.
        width : int, optional
            Chiều rộng inch, mặc định là 8.
        height : int, optional
            Chiều cao inch, mặc định là 6.
        dpi : int, optional
            Số điểm ảnh trên inch, mặc định là 100.
        """
        if not has_mpl:
            logger.error("Không thể khởi tạo ParetoFigureCanvas: thiếu matplotlib")
            return

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.pareto_points = []
        self.objective_names = []
        self.highlighted_point = None
        self.colorbar = None
        self.is_3d = False

        # Thiết lập đồ thị
        self.setup_figure()

    def setup_figure(self):
        """Thiết lập đồ thị Pareto."""
        try:
            self.fig.clear()
            if self.is_3d:
                self.axes = self.fig.add_subplot(111, projection="3d")
            else:
                self.axes = self.fig.add_subplot(111)

            self.axes.set_title("Trực quan hóa Pareto")
            self.axes.grid(True, linestyle="--", alpha=0.7)

            # Thiết lập Eclipse-like style
            self.fig.patch.set_facecolor("#F0F0F0")
            self.axes.set_facecolor("#FFFFFF")

            # Thiết lập màu và style
            plt.rcParams["axes.edgecolor"] = "#555555"
            plt.rcParams["axes.linewidth"] = 1.5

            self.draw()
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập figure: {str(e)}")

    def set_pareto_data(self, pareto_points, objective_names=None):
        """
        Thiết lập dữ liệu cho đồ thị Pareto.

        Parameters
        ----------
        pareto_points : List[List[float]]
            Danh sách các điểm Pareto. Mỗi điểm là một danh sách các giá trị mục tiêu.
        objective_names : List[str], optional
            Tên các mục tiêu tối ưu hóa.
        """
        self.pareto_points = pareto_points

        if objective_names:
            self.objective_names = objective_names
        else:
            # Tạo tên mặc định nếu không cung cấp
            self.objective_names = [
                f"Objective {i + 1}" for i in range(len(pareto_points[0]))
            ]

        # Cập nhật đồ thị
        self.update_plot()

    def set_3d_mode(self, enabled):
        """
        Chuyển đổi giữa chế độ hiển thị 2D và 3D.

        Parameters
        ----------
        enabled : bool
            True để hiển thị 3D, False để hiển thị 2D.
        """
        if self.is_3d != enabled:
            self.is_3d = enabled
            self.setup_figure()
            self.update_plot()

    def highlight_solution(self, index):
        """
        Tô sáng giải pháp được chọn trong không gian Pareto.

        Parameters
        ----------
        index : int
            Chỉ số của giải pháp cần tô sáng.
        """
        self.highlighted_point = index
        self.update_plot()

    def set_color_by_objective(self, objective_index):
        """
        Thiết lập màu sắc điểm dựa trên giá trị của một mục tiêu cụ thể.

        Parameters
        ----------
        objective_index : int
            Chỉ số của mục tiêu dùng để xác định màu.
        """
        self.color_by = objective_index
        self.update_plot()

    def update_plot(self):
        """Cập nhật đồ thị Pareto với dữ liệu mới nhất."""
        if not self.pareto_points or not has_mpl:
            return

        try:
            self.axes.clear()

            # Chuyển đổi danh sách điểm thành mảng numpy để vẽ
            points = np.array(self.pareto_points)

            # Xác định số chiều có thể hiển thị
            num_objectives = min(3, points.shape[1])

            if self.is_3d and num_objectives >= 3:
                # Hiển thị 3D Pareto
                scatter = self.axes.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    c=points[:, 0],
                    cmap="viridis",
                    s=50,
                    alpha=0.8,
                    edgecolors="w",
                )

                self.axes.set_xlabel(self.objective_names[0])
                self.axes.set_ylabel(self.objective_names[1])
                self.axes.set_zlabel(self.objective_names[2])

                # Thêm thanh màu
                if self.colorbar:
                    self.colorbar.remove()
                self.colorbar = self.fig.colorbar(scatter, ax=self.axes, pad=0.1)
                self.colorbar.set_label(self.objective_names[0])

            else:
                # Hiển thị 2D Pareto
                scatter = self.axes.scatter(
                    points[:, 0],
                    points[:, 1],
                    c=points[:, 0],
                    cmap="viridis",
                    s=50,
                    alpha=0.8,
                    edgecolors="w",
                )

                self.axes.set_xlabel(self.objective_names[0])
                self.axes.set_ylabel(self.objective_names[1])

                # Thêm thanh màu
                if hasattr(self, "colorbar") and self.colorbar:
                    self.colorbar.remove()
                self.colorbar = self.fig.colorbar(scatter, ax=self.axes, pad=0.1)
                self.colorbar.set_label(self.objective_names[0])

            # Tô sáng điểm được chọn
            if self.highlighted_point is not None and 0 <= self.highlighted_point < len(
                self.pareto_points
            ):
                point = self.pareto_points[self.highlighted_point]
                if self.is_3d and num_objectives >= 3:
                    self.axes.scatter(
                        [point[0]],
                        [point[1]],
                        [point[2]],
                        color="red",
                        s=100,
                        edgecolors="k",
                        linewidth=2,
                        zorder=10,
                    )
                else:
                    self.axes.scatter(
                        [point[0]],
                        [point[1]],
                        color="red",
                        s=100,
                        edgecolors="k",
                        linewidth=2,
                        zorder=10,
                    )

            # Thiết lập tiêu đề
            if self.highlighted_point is not None:
                self.axes.set_title("Trực quan hóa Pareto (Đã chọn giải pháp)")
            else:
                self.axes.set_title("Trực quan hóa Pareto")

            # Đặt lưới
            self.axes.grid(True, linestyle="--", alpha=0.7)

            # Cập nhật canvas
            self.fig.tight_layout()
            self.draw()

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật đồ thị Pareto: {str(e)}")


class ParetoNavigator(QWidget):
    """
    Widget điều hướng Pareto cho tối ưu hóa đa tiêu chí.

    Widget này cho phép người dùng điều hướng qua không gian Pareto
    của các giải pháp tối ưu hóa đa tiêu chí (MCO) và chọn giải pháp
    tối ưu phù hợp với mong muốn.
    """

    # Tín hiệu để thông báo khi người dùng chọn một giải pháp
    solutionSelected = pyqtSignal(int)  # Chỉ số của giải pháp được chọn
    weightsChanged = pyqtSignal(list)  # Danh sách trọng số mới

    def __init__(self, parent=None):
        """
        Khởi tạo widget điều hướng Pareto.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha.
        """
        if not has_pyqt5:
            logger.error("Không thể khởi tạo ParetoNavigator: thiếu PyQt5")
            return

        super().__init__(parent)

        self.pareto_points = []
        self.objective_names = []
        self.objective_weights = []

        self.selected_solution_index = None
        self.current_view_mode = "2D"

        # Thiết lập giao diện người dùng
        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        try:
            # Layout chính: chia đôi màn hình
            self.main_layout = QHBoxLayout(self)
            self.splitter = QSplitter(Qt.Horizontal)

            # Panel bên trái: đồ thị và điều khiển đồ thị
            self.left_panel = QWidget()
            self.left_layout = QVBoxLayout(self.left_panel)

            # Panel bên phải: thanh trượt trọng số và thông tin giải pháp
            self.right_panel = QWidget()
            self.right_layout = QVBoxLayout(self.right_panel)

            # Canvas Pareto
            self.pareto_canvas = ParetoFigureCanvas(self, width=6, height=5, dpi=100)
            self.pareto_toolbar = NavigationToolbar(self.pareto_canvas, self)

            self.left_layout.addWidget(self.pareto_canvas)
            self.left_layout.addWidget(self.pareto_toolbar)

            # Điều khiển đồ thị
            self.view_controls = QGroupBox("Điều khiển hiển thị")
            self.view_layout = QHBoxLayout()

            self.view_mode_combo = QComboBox()
            self.view_mode_combo.addItems(["2D", "3D"])
            self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)

            self.color_by_combo = QComboBox()
            self.color_by_combo.currentIndexChanged.connect(self._on_color_by_changed)

            self.view_layout.addWidget(QLabel("Chế độ xem:"))
            self.view_layout.addWidget(self.view_mode_combo)
            self.view_layout.addWidget(QLabel("Màu theo:"))
            self.view_layout.addWidget(self.color_by_combo)

            self.view_controls.setLayout(self.view_layout)
            self.left_layout.addWidget(self.view_controls)

            # Trọng số mục tiêu (bên phải)
            self.weights_group = QGroupBox("Trọng số mục tiêu")
            self.weights_layout = QVBoxLayout()
            self.weights_group.setLayout(self.weights_layout)
            self.right_layout.addWidget(self.weights_group)

            # Thông tin giải pháp được chọn
            self.solution_info = QGroupBox("Thông tin giải pháp")
            self.solution_layout = QVBoxLayout()

            self.solution_table = QTableWidget()
            self.solution_table.setColumnCount(2)
            self.solution_table.setHorizontalHeaderLabels(["Mục tiêu", "Giá trị"])
            self.solution_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch
            )

            self.solution_layout.addWidget(self.solution_table)
            self.solution_info.setLayout(self.solution_layout)
            self.right_layout.addWidget(self.solution_info)

            # Các nút hành động
            self.buttons_layout = QHBoxLayout()

            self.select_button = QPushButton("Chọn giải pháp này")
            self.select_button.clicked.connect(self._on_select_button_clicked)

            self.reset_button = QPushButton("Đặt lại trọng số")
            self.reset_button.clicked.connect(self._on_reset_button_clicked)

            self.buttons_layout.addWidget(self.select_button)
            self.buttons_layout.addWidget(self.reset_button)

            self.right_layout.addLayout(self.buttons_layout)

            # Thêm panel vào splitter
            self.splitter.addWidget(self.left_panel)
            self.splitter.addWidget(self.right_panel)

            # Thiết lập kích thước ban đầu
            self.splitter.setSizes([int(self.width() * 0.7), int(self.width() * 0.3)])

            # Thêm splitter vào layout chính
            self.main_layout.addWidget(self.splitter)

            # Thiết lập style giống Eclipse
            self.setup_eclipse_style()

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập UI cho ParetoNavigator: {str(e)}")

    def setup_eclipse_style(self):
        """Thiết lập style giống Eclipse cho widget."""
        try:
            style = """
                QWidget {
                    background-color: #F0F0F0;
                    color: #333333;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #CCCCCC;
                    border-radius: 3px;
                    margin-top: 1ex;
                    padding-top: 1ex;
                }
                QPushButton {
                    background-color: #4A90E2;
                    color: white;
                    border-radius: 3px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #5A9AE8;
                }
                QSlider::groove:horizontal {
                    height: 8px;
                    background: #CCCCCC;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #4A90E2;
                    border-radius: 6px;
                    width: 12px;
                    margin: -4px 0;
                }
            """
            self.setStyleSheet(style)
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập Eclipse style: {str(e)}")

    def set_pareto_data(self, pareto_points, objective_names=None):
        """
        Thiết lập dữ liệu Pareto cho widget.

        Parameters
        ----------
        pareto_points : List[List[float]]
            Danh sách các điểm Pareto. Mỗi điểm là một danh sách các giá trị mục tiêu.
        objective_names : List[str], optional
            Tên các mục tiêu tối ưu hóa.
        """
        self.pareto_points = pareto_points

        if objective_names:
            self.objective_names = objective_names
        else:
            # Tạo tên mặc định nếu không cung cấp
            self.objective_names = [
                f"Mục tiêu {i + 1}" for i in range(len(pareto_points[0]))
            ]

        # Thiết lập trọng số ban đầu (đồng đều)
        num_objectives = len(self.objective_names)
        self.objective_weights = [1.0 / num_objectives] * num_objectives

        # Cập nhật UI
        self._update_weights_ui()
        self._update_color_by_combo()

        # Thiết lập dữ liệu cho canvas
        self.pareto_canvas.set_pareto_data(self.pareto_points, self.objective_names)

        # Ban đầu không chọn giải pháp nào
        self.selected_solution_index = None
        self._update_solution_info()

    def _update_weights_ui(self):
        """Cập nhật UI cho phần trọng số."""
        try:
            # Xóa layout cũ
            for i in reversed(range(self.weights_layout.count())):
                widget = self.weights_layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()

            # Tạo slider cho mỗi mục tiêu
            self.weight_sliders = []
            self.weight_labels = []

            for i, name in enumerate(self.objective_names):
                # Layout cho mỗi mục tiêu
                obj_layout = QHBoxLayout()

                # Label cho tên mục tiêu
                name_label = QLabel(name)
                obj_layout.addWidget(name_label, 1)

                # Slider cho trọng số
                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 100)
                slider.setValue(int(self.objective_weights[i] * 100))
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(10)
                slider.valueChanged.connect(
                    lambda val, idx=i: self._on_slider_changed(idx, val)
                )
                obj_layout.addWidget(slider, 3)

                # Label cho giá trị trọng số
                value_label = QLabel(f"{self.objective_weights[i]:.2f}")
                obj_layout.addWidget(value_label, 1)

                self.weight_sliders.append(slider)
                self.weight_labels.append(value_label)

                self.weights_layout.addLayout(obj_layout)

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật UI trọng số: {str(e)}")

    def _update_color_by_combo(self):
        """Cập nhật combobox chọn màu theo mục tiêu."""
        try:
            self.color_by_combo.clear()
            self.color_by_combo.addItems(self.objective_names)
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật color_by_combo: {str(e)}")

    def _update_solution_info(self):
        """Cập nhật thông tin về giải pháp được chọn."""
        try:
            self.solution_table.setRowCount(len(self.objective_names))

            for i, name in enumerate(self.objective_names):
                # Tên mục tiêu
                self.solution_table.setItem(i, 0, QTableWidgetItem(name))

                # Giá trị mục tiêu
                if self.selected_solution_index is not None:
                    value = self.pareto_points[self.selected_solution_index][i]
                    self.solution_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
                else:
                    self.solution_table.setItem(i, 1, QTableWidgetItem("N/A"))

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật thông tin giải pháp: {str(e)}")

    def _on_view_mode_changed(self, mode):
        """
        Xử lý khi người dùng thay đổi chế độ xem (2D/3D).

        Parameters
        ----------
        mode : str
            Chế độ xem mới ("2D" hoặc "3D").
        """
        try:
            self.current_view_mode = mode
            self.pareto_canvas.set_3d_mode(mode == "3D")
        except Exception as e:
            logger.error(f"Lỗi khi thay đổi chế độ xem: {str(e)}")

    def _on_color_by_changed(self, index):
        """
        Xử lý khi người dùng thay đổi mục tiêu dùng để xác định màu.

        Parameters
        ----------
        index : int
            Chỉ số của mục tiêu được chọn.
        """
        try:
            if 0 <= index < len(self.objective_names):
                self.pareto_canvas.set_color_by_objective(index)
        except Exception as e:
            logger.error(f"Lỗi khi thay đổi color_by: {str(e)}")

    def _on_slider_changed(self, index, value):
        """
        Xử lý khi người dùng thay đổi trọng số.

        Parameters
        ----------
        index : int
            Chỉ số của mục tiêu có trọng số thay đổi.
        value : int
            Giá trị mới của slider (0-100).
        """
        try:
            # Cập nhật trọng số
            self.objective_weights[index] = value / 100.0

            # Chuẩn hóa tổng trọng số về 1
            total_weight = sum(self.objective_weights)
            if total_weight > 0:
                self.objective_weights = [
                    w / total_weight for w in self.objective_weights
                ]

            # Cập nhật UI
            for i, (slider, label) in enumerate(
                zip(self.weight_sliders, self.weight_labels)
            ):
                # Tránh gọi lại sự kiện valueChanged
                slider.blockSignals(True)
                slider.setValue(int(self.objective_weights[i] * 100))
                slider.blockSignals(False)

                label.setText(f"{self.objective_weights[i]:.2f}")

            # Phát tín hiệu khi trọng số thay đổi
            self.weightsChanged.emit(self.objective_weights)

            # Tìm và chọn giải pháp phù hợp với trọng số
            self._select_by_weights()

        except Exception as e:
            logger.error(f"Lỗi khi thay đổi slider: {str(e)}")

    def _select_by_weights(self):
        """Tìm và chọn giải pháp phù hợp nhất với trọng số hiện tại."""
        try:
            if not self.pareto_points or not self.objective_weights:
                return

            # Tính weighted sum cho mỗi giải pháp
            best_index = -1
            best_score = float("inf")  # Giá trị nhỏ hơn là tốt hơn

            for i, point in enumerate(self.pareto_points):
                # Tính tổng có trọng số (weighted sum)
                score = sum(
                    point[j] * self.objective_weights[j] for j in range(len(point))
                )

                if score < best_score:
                    best_score = score
                    best_index = i

            if best_index >= 0:
                self.selected_solution_index = best_index
                self._on_solution_selected(best_index)

        except Exception as e:
            logger.error(f"Lỗi khi chọn giải pháp theo trọng số: {str(e)}")

    def _on_solution_selected(self, index):
        """
        Xử lý khi một giải pháp được chọn.

        Parameters
        ----------
        index : int
            Chỉ số của giải pháp được chọn.
        """
        try:
            self.selected_solution_index = index

            # Tô sáng giải pháp trên đồ thị
            self.pareto_canvas.highlight_solution(index)

            # Cập nhật thông tin giải pháp
            self._update_solution_info()

        except Exception as e:
            logger.error(f"Lỗi khi chọn giải pháp: {str(e)}")

    def _on_select_button_clicked(self):
        """Xử lý khi người dùng nhấn nút chọn giải pháp."""
        try:
            if self.selected_solution_index is not None:
                self.solutionSelected.emit(self.selected_solution_index)
        except Exception as e:
            logger.error(f"Lỗi khi nhấn nút chọn: {str(e)}")

    def _on_reset_button_clicked(self):
        """Xử lý khi người dùng nhấn nút đặt lại trọng số."""
        try:
            # Đặt lại trọng số về đồng đều
            num_objectives = len(self.objective_names)
            self.objective_weights = [1.0 / num_objectives] * num_objectives

            # Cập nhật UI
            self._update_weights_ui()

            # Phát tín hiệu trọng số thay đổi
            self.weightsChanged.emit(self.objective_weights)

            # Tìm và chọn giải pháp phù hợp với trọng số
            self._select_by_weights()

        except Exception as e:
            logger.error(f"Lỗi khi đặt lại trọng số: {str(e)}")


# Dạng Function để dễ sử dụng từ bên ngoài
def show_pareto_navigator(pareto_points, objective_names=None, parent=None):
    """
    Hiển thị cửa sổ điều hướng Pareto.

    Function này tạo và hiển thị một cửa sổ điều hướng Pareto độc lập.

    Parameters
    ----------
    pareto_points : List[List[float]]
        Danh sách các điểm Pareto.
    objective_names : List[str], optional
        Tên các mục tiêu.
    parent : QWidget, optional
        Widget cha.

    Returns
    -------
    ParetoNavigator
        Đối tượng điều hướng Pareto đã tạo.
    """
    from PyQt5.QtWidgets import QDialog, QVBoxLayout

    dialog = QDialog(parent)
    dialog.setWindowTitle("Eclipse-style Pareto Navigator")
    dialog.resize(1000, 700)

    layout = QVBoxLayout(dialog)
    navigator = ParetoNavigator(dialog)
    layout.addWidget(navigator)

    navigator.set_pareto_data(pareto_points, objective_names)
    dialog.show()

    return navigator
