#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCO Navigator Widget for QuangTPS.

Widget cho multi-criteria optimization với Pareto navigation.
"""

import logging
import numpy as np
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# Import matplotlib với fallback
try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    logger.warning("Matplotlib không khả dụng cho ParetoFigureCanvas")
    HAS_MATPLOTLIB = False

    # Tạo fallback classes
    class Figure:
        def __init__(self, *args, **kwargs):
            pass

        def add_subplot(self, *args, **kwargs):
            return FallbackAxes()

        def clear(self):
            pass

    class FigureCanvas:
        def __init__(self, figure):
            self.figure = figure

        def draw(self):
            pass

    class FallbackAxes:
        def scatter(self, *args, **kwargs):
            pass

        def plot(self, *args, **kwargs):
            pass

        def set_xlabel(self, *args, **kwargs):
            pass

        def set_ylabel(self, *args, **kwargs):
            pass

        def set_title(self, *args, **kwargs):
            pass

        def grid(self, *args, **kwargs):
            pass


# Import PyQt5 với fallback
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QSlider,
        QTextEdit,
    )
    from PyQt5.QtCore import pyqtSignal, Qt

    HAS_PYQT = True
except ImportError:
    logger.warning("PyQt5 không khả dụng cho MCO Navigator Widget")
    HAS_PYQT = False

    # Fallback classes
    class QWidget:
        def __init__(self, parent=None):
            self.parent = parent

    class pyqtSignal:
        def __init__(self, *args):
            pass

    # Fallback Qt constants
    class Qt:
        Horizontal = 1


class ParetoFigureCanvas(FigureCanvas):
    """
    Canvas để hiển thị bề mặt Pareto 3D cho MCO.

    Cung cấp visualization cho multi-criteria optimization với khả năng
    tương tác và khám phá các giải pháp Pareto optimal.
    """

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """
        Khởi tạo Pareto Figure Canvas.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        width : float
            Chiều rộng figure (inches)
        height : float
            Chiều cao figure (inches)
        dpi : int
            Độ phân giải (dots per inch)
        """
        if not HAS_MATPLOTLIB:
            logger.error("Không thể khởi tạo ParetoFigureCanvas: thiếu matplotlib")
            # Khởi tạo với fallback
            super().__init__(Figure())
            return

        # Tạo matplotlib figure
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)

        if parent and HAS_PYQT:
            self.setParent(parent)

        # Thiết lập subplot cho hiển thị Pareto
        self.axes = self.figure.add_subplot(
            111, projection="3d" if self._has_3d_support() else None
        )

        # Data storage
        self.pareto_points = []
        self.objective_names = []
        self.selected_point = None

        # Styling
        self.figure.patch.set_facecolor("#2B2B2B")  # Eclipse dark theme
        self.axes.set_facecolor("#3C3C3C")

        logger.info("ParetoFigureCanvas khởi tạo thành công")

    def _has_3d_support(self) -> bool:
        """Kiểm tra hỗ trợ hiển thị 3D."""
        try:
            from mpl_toolkits.mplot3d import Axes3D

            return True
        except ImportError:
            logger.warning("Không hỗ trợ hiển thị 3D, sử dụng 2D")
            return False

    def set_pareto_data(
        self, points: List[Dict[str, float]], objective_names: List[str]
    ) -> None:
        """
        Cập nhật dữ liệu Pareto points.

        Parameters
        ----------
        points : List[Dict[str, float]]
            Danh sách các điểm Pareto với giá trị objective
        objective_names : List[str]
            Tên các objective functions
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Không thể cập nhật dữ liệu: thiếu matplotlib")
            return

        self.pareto_points = points
        self.objective_names = objective_names

        logger.info(
            f"Cập nhật {len(points)} điểm Pareto với {len(objective_names)} objectives"
        )
        self.update_plot()

    def update_plot(self) -> None:
        """Cập nhật hiển thị biểu đồ Pareto."""
        if not HAS_MATPLOTLIB or not self.pareto_points:
            return

        self.axes.clear()

        try:
            if len(self.objective_names) >= 3:
                self._plot_3d()
            elif len(self.objective_names) == 2:
                self._plot_2d()
            else:
                self._plot_1d()

        except Exception as e:
            logger.error(f"Lỗi cập nhật plot: {e}")
            self._plot_fallback()

        self.draw()

    def _plot_3d(self) -> None:
        """Hiển thị Pareto surface 3D."""
        if len(self.objective_names) < 3:
            return

        obj1_name, obj2_name, obj3_name = self.objective_names[:3]

        x_values = [point.get(obj1_name, 0) for point in self.pareto_points]
        y_values = [point.get(obj2_name, 0) for point in self.pareto_points]
        z_values = [point.get(obj3_name, 0) for point in self.pareto_points]

        # Scatter plot với màu gradient
        colors = np.linspace(0, 1, len(self.pareto_points))
        scatter = self.axes.scatter(
            x_values, y_values, z_values, c=colors, cmap="viridis", s=50, alpha=0.7
        )

        # Labels và title
        self.axes.set_xlabel(obj1_name, color="white")
        self.axes.set_ylabel(obj2_name, color="white")
        self.axes.set_zlabel(obj3_name, color="white")
        self.axes.set_title("Pareto Surface (3D)", color="white")

        # Grid
        self.axes.grid(True, alpha=0.3)

    def _plot_2d(self) -> None:
        """Hiển thị Pareto front 2D."""
        if len(self.objective_names) < 2:
            return

        obj1_name, obj2_name = self.objective_names[:2]

        x_values = [point.get(obj1_name, 0) for point in self.pareto_points]
        y_values = [point.get(obj2_name, 0) for point in self.pareto_points]

        # Scatter plot
        self.axes.scatter(x_values, y_values, c="#4A90E2", s=50, alpha=0.7)

        # Connect points to show Pareto front
        if len(x_values) > 1:
            # Sort by first objective for proper line connection
            sorted_pairs = sorted(zip(x_values, y_values))
            sorted_x, sorted_y = zip(*sorted_pairs)
            self.axes.plot(sorted_x, sorted_y, "r-", alpha=0.5, linewidth=2)

        # Labels và title
        self.axes.set_xlabel(obj1_name, color="white")
        self.axes.set_ylabel(obj2_name, color="white")
        self.axes.set_title("Pareto Front (2D)", color="white")

        # Grid
        self.axes.grid(True, alpha=0.3)

    def _plot_1d(self) -> None:
        """Hiển thị single objective."""
        if not self.objective_names:
            return

        obj_name = self.objective_names[0]
        values = [point.get(obj_name, 0) for point in self.pareto_points]

        # Bar plot cho single objective
        indices = range(len(values))
        self.axes.bar(indices, values, color="#4A90E2", alpha=0.7)

        # Labels
        self.axes.set_xlabel("Solution Index", color="white")
        self.axes.set_ylabel(obj_name, color="white")
        self.axes.set_title(f"Objective Values: {obj_name}", color="white")

        # Grid
        self.axes.grid(True, alpha=0.3)

    def _plot_fallback(self) -> None:
        """Hiển thị fallback khi có lỗi."""
        self.axes.text(
            0.5,
            0.5,
            "Pareto Visualization\n(Data Error)",
            transform=self.axes.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=14,
        )

        self.axes.set_title("MCO Pareto Navigator", color="white")

    def highlight_point(self, point_index: int) -> None:
        """Làm nổi bật một điểm Pareto cụ thể."""
        if not HAS_MATPLOTLIB or point_index >= len(self.pareto_points):
            return

        self.selected_point = point_index
        logger.info(f"Highlighted Pareto point {point_index}")

        # Redraw với điểm được highlight
        self.update_plot()

        # Thêm marker cho điểm được chọn
        try:
            if len(self.objective_names) >= 2:
                point = self.pareto_points[point_index]
                obj1_name = self.objective_names[0]
                obj2_name = self.objective_names[1]

                x = point.get(obj1_name, 0)
                y = point.get(obj2_name, 0)

                if len(self.objective_names) >= 3:
                    obj3_name = self.objective_names[2]
                    z = point.get(obj3_name, 0)
                    self.axes.scatter([x], [y], [z], c="red", s=100, marker="*")
                else:
                    self.axes.scatter([x], [y], c="red", s=100, marker="*")

            self.draw()

        except Exception as e:
            logger.error(f"Lỗi highlight point: {e}")

    def clear_plot(self) -> None:
        """Xóa biểu đồ."""
        if not HAS_MATPLOTLIB:
            return

        self.axes.clear()
        self.axes.set_facecolor("#3C3C3C")
        self.draw()

        logger.info("Cleared Pareto plot")


class MCONavigatorWidget(QWidget):
    """
    Widget chính cho MCO Navigator.

    Cung cấp interface để khám phá các giải pháp Pareto optimal
    và điều chỉnh trọng số các objective functions.
    """

    # Signals
    solution_selected = pyqtSignal(int) if HAS_PYQT else None
    weights_changed = pyqtSignal(dict) if HAS_PYQT else None

    def __init__(self, parent=None):
        """Khởi tạo MCO Navigator Widget."""
        super().__init__(parent)

        self.pareto_canvas = None
        self.current_weights = {}

        if HAS_PYQT and HAS_MATPLOTLIB:
            self.setup_ui()
        else:
            self.setup_fallback_ui()

        logger.info("MCONavigatorWidget khởi tạo thành công")

    def setup_ui(self) -> None:
        """Thiết lập giao diện người dùng."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Multi-Criteria Optimization Navigator")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Pareto canvas
        self.pareto_canvas = ParetoFigureCanvas(self, width=8, height=6)
        layout.addWidget(self.pareto_canvas)

        # Control buttons
        button_layout = QHBoxLayout()

        self.compute_btn = QPushButton("Compute Pareto")
        self.compute_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5BA0F2;
            }
        """)
        button_layout.addWidget(self.compute_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #666666;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #777777;
            }
        """)
        button_layout.addWidget(self.reset_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals
        self.compute_btn.clicked.connect(self.compute_pareto)
        self.reset_btn.clicked.connect(self.reset_view)

    def setup_fallback_ui(self) -> None:
        """Thiết lập UI fallback khi thiếu dependencies."""
        if HAS_PYQT:
            layout = QVBoxLayout()

            error_label = QLabel(
                "MCO Navigator không khả dụng\n(Thiếu matplotlib hoặc PyQt5)"
            )
            error_label.setStyleSheet("color: #F5A623; font-size: 14px;")
            layout.addWidget(error_label)

            self.setLayout(layout)

    def set_objectives(self, objectives: List[str]) -> None:
        """Thiết lập danh sách objectives."""
        if self.pareto_canvas:
            self.pareto_canvas.objective_names = objectives
            logger.info(f"Set objectives: {objectives}")

    def update_pareto_data(self, points: List[Dict[str, float]]) -> None:
        """Cập nhật dữ liệu Pareto points."""
        if self.pareto_canvas:
            self.pareto_canvas.set_pareto_data(
                points, self.pareto_canvas.objective_names
            )

    def compute_pareto(self) -> None:
        """Tính toán Pareto surface."""
        logger.info("Computing Pareto surface...")

        # Generate mock data for demo
        mock_points = []
        objectives = ["Objective 1", "Objective 2", "Objective 3"]

        for i in range(20):
            point = {
                "Objective 1": np.random.uniform(0, 100),
                "Objective 2": np.random.uniform(0, 100),
                "Objective 3": np.random.uniform(0, 100),
            }
            mock_points.append(point)

        if self.pareto_canvas:
            self.pareto_canvas.set_pareto_data(mock_points, objectives)

    def reset_view(self) -> None:
        """Reset hiển thị."""
        if self.pareto_canvas:
            self.pareto_canvas.clear_plot()

        logger.info("MCO Navigator view reset")

    def select_pareto_point(self, point_index: int) -> None:
        """Chọn một điểm Pareto."""
        if self.pareto_canvas:
            self.pareto_canvas.highlight_point(point_index)

        if self.solution_selected:
            self.solution_selected.emit(point_index)


# Factory function
def create_mco_navigator_widget(
    parent=None, objectives=None, **kwargs
) -> MCONavigatorWidget:
    """
    Tạo MCO Navigator Widget.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha
    objectives : List[str] or Dict[str, Any], optional
        Danh sách hoặc dictionary các objective functions
    **kwargs : Any
        Các tham số khác

    Returns
    -------
    MCONavigatorWidget
        Widget MCO Navigator
    """
    try:
        # Khởi tạo widget
        widget = MCONavigatorWidget(parent)

        # Xử lý objectives
        if objectives is not None:
            if isinstance(objectives, dict):
                objective_names = list(objectives.keys())
            elif isinstance(objectives, (list, tuple)):
                objective_names = list(objectives)
            else:
                logger.warning(f"Loại objectives không hợp lệ: {type(objectives)}")
                objective_names = ["Objective 1", "Objective 2", "Objective 3"]

            # Thiết lập objectives cho widget
            if hasattr(widget, "set_objectives"):
                widget.set_objectives(objective_names)

            # Tạo dữ liệu Pareto giả lập nếu có ít nhất 2 objectives
            if len(objective_names) >= 2 and hasattr(widget, "pareto_canvas"):
                # Tạo dữ liệu Pareto giả lập
                import numpy as np

                num_points = 20
                points = []

                for i in range(num_points):
                    point = {}
                    for obj in objective_names:
                        # Giá trị ngẫu nhiên từ 0 đến 1
                        point[obj] = np.random.rand()
                    points.append(point)

                # Cập nhật dữ liệu Pareto
                if hasattr(widget.pareto_canvas, "set_pareto_data"):
                    widget.pareto_canvas.set_pareto_data(points, objective_names)
        else:
            logger.warning("Không có objectives được cung cấp cho MCO Navigator Widget")
            # Thiết lập objectives mặc định
            default_objectives = ["Target Coverage", "OAR Sparing", "Conformity"]
            if hasattr(widget, "set_objectives"):
                widget.set_objectives(default_objectives)

        # Xử lý các tham số khác
        if "title" in kwargs and hasattr(widget, "setWindowTitle"):
            widget.setWindowTitle(kwargs["title"])

        if "size" in kwargs and hasattr(widget, "resize"):
            widget.resize(*kwargs["size"])

        logger.info("Tạo MCO Navigator Widget thành công")
        return widget
    except Exception as e:
        logger.error(f"Lỗi tạo MCO Navigator Widget: {e}")
        # Tạo widget fallback đơn giản
        fallback_widget = MCONavigatorWidget(parent)
        if hasattr(fallback_widget, "setup_fallback_ui"):
            fallback_widget.setup_fallback_ui()
        return fallback_widget


# Export classes
__all__ = ["ParetoFigureCanvas", "MCONavigatorWidget", "create_mco_navigator_widget"]


# Thêm các class còn thiếu
class ObjectiveWeightEditor(QWidget if HAS_PYQT else object):
    """
    Widget cho phép người dùng điều chỉnh trọng số các objective functions.
    """

    weights_changed = pyqtSignal(dict) if HAS_PYQT else None

    def __init__(self, parent=None):
        """Khởi tạo ObjectiveWeightEditor."""
        if HAS_PYQT:
            super().__init__(parent)
            self.sliders = {}
            self.weight_labels = {}
            self.is_updating = False
            self.setup_ui()
        else:
            logger.warning("ObjectiveWeightEditor yêu cầu PyQt5")

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        if not HAS_PYQT:
            return

        layout = QVBoxLayout()

        # Title
        title = QLabel("Objective Weights")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        self.weights_layout = QVBoxLayout()
        layout.addLayout(self.weights_layout)

        self.setLayout(layout)

    def set_objectives(self, objectives: List[str], weights: Dict[str, float] = None):
        """Thiết lập danh sách objectives và trọng số ban đầu."""
        if not HAS_PYQT:
            return

        # Xóa widgets cũ
        for i in reversed(range(self.weights_layout.count())):
            child = self.weights_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        self.sliders.clear()
        self.weight_labels.clear()

        if not objectives:
            return

        # Thiết lập trọng số mặc định
        if weights is None:
            equal_weight = 1.0 / len(objectives)
            weights = {obj: equal_weight for obj in objectives}

        # Tạo slider cho mỗi objective
        for obj in objectives:
            obj_layout = QHBoxLayout()

            # Label tên objective
            obj_label = QLabel(f"{obj}:")
            obj_label.setStyleSheet("color: white;")
            obj_layout.addWidget(obj_label)

            # Slider
            slider = QSlider(Qt.Horizontal) if HAS_PYQT else None
            if slider:
                slider.setMinimum(0)
                slider.setMaximum(100)
                slider.setValue(int(weights.get(obj, 0.5) * 100))
                slider.valueChanged.connect(self._on_slider_changed)
                obj_layout.addWidget(slider)
                self.sliders[obj] = slider

            # Label hiển thị giá trị
            value_label = QLabel(f"{weights.get(obj, 0.5):.2f}")
            value_label.setStyleSheet("color: white;")
            obj_layout.addWidget(value_label)
            self.weight_labels[obj] = value_label

            self.weights_layout.addLayout(obj_layout)

        logger.info(f"Thiết lập {len(objectives)} objectives trong weight editor")

    def _on_slider_changed(self):
        """Xử lý khi slider thay đổi."""
        if self.is_updating or not HAS_PYQT:
            return

        # Tính toán trọng số mới
        weights = {}
        total = 0

        for obj, slider in self.sliders.items():
            value = slider.value() / 100.0
            weights[obj] = value
            total += value

        # Normalize weights
        if total > 0:
            for obj in weights:
                weights[obj] /= total
                self.weight_labels[obj].setText(f"{weights[obj]:.2f}")

        # Emit signal
        if self.weights_changed:
            self.weights_changed.emit(weights)


class SolutionDetailsPanel(QWidget if HAS_PYQT else object):
    """
    Panel hiển thị chi tiết của một giải pháp Pareto.
    """

    def __init__(self, parent=None):
        """Khởi tạo SolutionDetailsPanel."""
        if HAS_PYQT:
            super().__init__(parent)
            self.current_solution = None
            self.setup_ui()
        else:
            logger.warning("SolutionDetailsPanel yêu cầu PyQt5")

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        if not HAS_PYQT:
            return

        layout = QVBoxLayout()

        # Title
        title = QLabel("Solution Details")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # Details area
        self.details_text = QTextEdit()
        self.details_text.setStyleSheet("""
            QTextEdit {
                background-color: #3C3C3C;
                color: white;
                border: 1px solid #555555;
                font-family: 'Courier New', monospace;
            }
        """)
        self.details_text.setReadOnly(True)
        layout.addWidget(self.details_text)

        # Apply button
        self.apply_btn = QPushButton("Apply Solution")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5BA0F2;
            }
        """)
        layout.addWidget(self.apply_btn)

        self.setLayout(layout)

    def display_solution(self, solution):
        """Hiển thị chi tiết của một giải pháp."""
        if not HAS_PYQT:
            return

        self.current_solution = solution

        if not solution:
            self.details_text.setText("Không có giải pháp để hiển thị")
            return

        # Format thông tin giải pháp
        details = []
        details.append(f"Solution ID: {getattr(solution, 'id', 'N/A')}")
        details.append(f"Rank: {getattr(solution, 'rank', 'N/A')}")
        details.append("")

        # Objective values
        if hasattr(solution, "objective_values"):
            details.append("Objective Values:")
            for obj, value in solution.objective_values.items():
                details.append(f"  {obj}: {value:.4f}")
            details.append("")

        # Parameters
        if hasattr(solution, "parameters"):
            details.append("Parameters:")
            for param, value in solution.parameters.items():
                details.append(f"  {param}: {value:.4f}")
            details.append("")

        # Constraints
        if hasattr(solution, "constraint_violations"):
            details.append("Constraint Violations:")
            for constraint, violation in solution.constraint_violations.items():
                status = "PASS" if violation <= 0 else f"FAIL ({violation:.4f})"
                details.append(f"  {constraint}: {status}")

        self.details_text.setText("\n".join(details))
        logger.info(f"Hiển thị chi tiết solution {getattr(solution, 'id', 'unknown')}")


# Cập nhật __all__ với các class mới
__all__ = [
    "ParetoFigureCanvas",
    "MCONavigatorWidget",
    "ObjectiveWeightEditor",
    "SolutionDetailsPanel",
    "create_mco_navigator_widget",
]
