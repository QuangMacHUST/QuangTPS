#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module ColorMapSelector cho QuangTPS.

Cung cấp widget cho việc lựa chọn và tùy chỉnh colormap
cho hiển thị phân phối liều trong hệ thống kế hoạch xạ trị.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Try to import matplotlib for colormaps
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    MATPLOTLIB_AVAILABLE = True

    # Kiểm tra xem có hàm register_cmap hay không
    if not hasattr(plt.cm, "register_cmap"):
        # Tạo hàm giả
        def register_cmap(name, cmap):
            plt.cm._cmap_registry[name] = cmap
            plt.cm.__dict__[name] = cmap

        plt.register_cmap = register_cmap
    else:
        plt.register_cmap = plt.cm.register_cmap

except ImportError as e:
    logging.error(f"Không thể import matplotlib: {e}")
    MATPLOTLIB_AVAILABLE = False

    # Tạo lớp giả cho matplotlib
    class plt:
        @staticmethod
        def register_cmap(name, cmap):
            pass

        @staticmethod
        def colormaps():
            return []

        @staticmethod
        def get_cmap(name):
            return None

    class LinearSegmentedColormap:
        @staticmethod
        def from_list(name, colors, N=256):
            return None

        def reversed(self):
            return self


# Try to import PyQt5
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QSlider,
        QCheckBox,
        QComboBox,
        QGroupBox,
        QFrame,
        QSplitter,
        QSpinBox,
        QDoubleSpinBox,
        QTabWidget,
        QMessageBox,
        QSizePolicy,
        QStackedWidget,
        QGridLayout,
        QLineEdit,
        QToolButton,
        QColorDialog,
        QDialog,
        QDialogButtonBox,
        QApplication,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, pyqtSlot
    from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap

    PYQT5_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import PyQt5: {e}")
    PYQT5_AVAILABLE = False

    # Tạo các lớp giả cho type checking
    class QWidget:
        pass

    class pyqtSignal:
        def __init__(self, *args):
            pass

    class QApplication:
        pass


try:
    from quangtps.common.geometry.colors import ColorMap, ColorUtils

    QUANGTPS_COLOR_MODULES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import các module màu sắc của QuangTPS: {e}")
    QUANGTPS_COLOR_MODULES_AVAILABLE = False

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class ColorMapSelector(QWidget):
    """
    Widget cho phép người dùng lựa chọn và tùy chỉnh colormap
    cho hiển thị phân phối liều xạ trị.

    Hỗ trợ các colormap tiêu chuẩn từ matplotlib và đồng bộ hóa
    thay đổi với DoseVisualization3D qua tín hiệu colormap_changed.
    """

    # Tín hiệu khi người dùng thay đổi colormap
    colormap_changed = pyqtSignal(object)  # Emit the colormap object

    # Các colormap mặc định
    DEFAULT_COLORMAPS = [
        "jet",  # Standard choice for dose display
        "rainbow",  # Alternative
        "hot",  # Heat map
        "coolwarm",  # Diverging blue-red
        "viridis",  # Perceptually uniform
        "plasma",  # Perceptually uniform
        "cividis",  # Color-vision-deficiency friendly
        "RdYlBu",  # Red-Yellow-Blue
        "eclipse",  # Custom Eclipse-like colormap
    ]

    # Eclipse colormap
    ECLIPSE_COLORS = [
        (0.0, (0.0, 0.0, 0.8)),  # Deep blue (low dose)
        (0.2, (0.0, 0.5, 1.0)),  # Light blue
        (0.4, (0.0, 1.0, 0.0)),  # Green
        (0.6, (1.0, 1.0, 0.0)),  # Yellow
        (0.8, (1.0, 0.5, 0.0)),  # Orange
        (1.0, (1.0, 0.0, 0.0)),  # Red (high dose)
    ]

    def __init__(self, parent=None):
        """
        Khởi tạo widget ColorMapSelector.

        Parameters:
        -----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)

        # Initialize values
        self.current_colormap_name = "jet"
        self.current_colormap = None
        self.invert_colormap = False
        self.custom_colormap_colors = []
        self.alpha = 1.0  # Transparency value (0-1)
        self.colormap_preview_width = 200
        self.colormap_preview_height = 30

        # Create custom colormaps
        self._create_custom_colormaps()

        # Setup UI
        self.setup_ui()

        # Initialize with default colormap
        self.set_colormap("jet")

    def _create_custom_colormaps(self):
        """Tạo colormap tùy chỉnh mô phỏng Eclipse."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Create Eclipse-like colormap
        eclipse_cmap = LinearSegmentedColormap.from_list(
            "eclipse", [color for _, color in self.ECLIPSE_COLORS], N=256
        )

        # Register custom colormaps
        plt.register_cmap(name="eclipse", cmap=eclipse_cmap)

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        if not PYQT5_AVAILABLE:
            self._setup_fallback_ui()
            return

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Group box for colormap selection
        colormap_group = QGroupBox("Colormap")
        colormap_layout = QVBoxLayout(colormap_group)

        # Colormap selection combobox
        colormap_layout.addWidget(QLabel("Colormap Type:"))
        self.colormap_combo = QComboBox()

        # Add available colormaps
        for cmap_name in self.DEFAULT_COLORMAPS:
            if not MATPLOTLIB_AVAILABLE and cmap_name != "eclipse":
                # Skip matplotlib colormaps if not available
                continue
            self.colormap_combo.addItem(cmap_name)

        self.colormap_combo.currentTextChanged.connect(self.on_colormap_selected)
        colormap_layout.addWidget(self.colormap_combo)

        # Invert colormap checkbox
        self.invert_checkbox = QCheckBox("Invert Colormap")
        self.invert_checkbox.toggled.connect(self.on_invert_toggled)
        colormap_layout.addWidget(self.invert_checkbox)

        # Alpha (transparency) slider
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Opacity:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(self.alpha * 100))
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        alpha_layout.addWidget(self.alpha_slider)
        self.alpha_value_label = QLabel(f"{int(self.alpha * 100)}%")
        alpha_layout.addWidget(self.alpha_value_label)
        colormap_layout.addLayout(alpha_layout)

        # Colormap preview
        colormap_layout.addWidget(QLabel("Preview:"))
        self.preview_frame = QFrame()
        self.preview_frame.setFrameShape(QFrame.StyledPanel)
        self.preview_frame.setMinimumHeight(self.colormap_preview_height)
        self.preview_frame.setMinimumWidth(self.colormap_preview_width)
        colormap_layout.addWidget(self.preview_frame)

        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure for colormap preview
            self.preview_figure = Figure(
                figsize=(
                    self.colormap_preview_width / 100,
                    self.colormap_preview_height / 100,
                ),
                dpi=100,
            )
            self.preview_canvas = FigureCanvas(self.preview_figure)
            self.preview_canvas.setMinimumHeight(self.colormap_preview_height)
            self.preview_canvas.setMaximumHeight(self.colormap_preview_height)
            self.preview_ax = self.preview_figure.add_subplot(111)
            self.preview_ax.set_position([0, 0, 1, 1])  # Fill entire figure
            colormap_layout.addWidget(self.preview_canvas)

        # Custom colormap button
        self.custom_colormap_button = QPushButton("Create Custom Colormap")
        self.custom_colormap_button.clicked.connect(self.show_custom_colormap_dialog)
        colormap_layout.addWidget(self.custom_colormap_button)

        # Apply button
        self.apply_button = QPushButton("Apply Colormap")
        self.apply_button.clicked.connect(self.apply_colormap)
        colormap_layout.addWidget(self.apply_button)

        # Add colormap group to main layout
        main_layout.addWidget(colormap_group)

        # Add stretch to align widgets to top
        main_layout.addStretch()

        # Initial colormap preview update
        self.update_colormap_preview()

    def _setup_fallback_ui(self):
        """Set up fallback UI when required dependencies are not available."""
        layout = QVBoxLayout(self)

        error_label = QLabel(
            "Cannot initialize ColorMapSelector due to missing dependencies."
        )
        error_label.setStyleSheet("color: red; font-weight: bold;")

        dependencies_label = QLabel(
            "Please ensure the following dependencies are installed:\n"
            "- PyQt5\n"
            "- Matplotlib (for advanced colormaps)"
        )

        layout.addWidget(error_label)
        layout.addWidget(dependencies_label)
        layout.addStretch()

    def set_colormap(self, colormap_name):
        """
        Thiết lập colormap theo tên.

        Parameters:
        -----------
        colormap_name : str
            Tên của colormap
        """
        if not MATPLOTLIB_AVAILABLE and colormap_name not in ["eclipse"]:
            logger.warning(
                f"Matplotlib is not available, cannot set colormap {colormap_name}"
            )
            return

        self.current_colormap_name = colormap_name

        # Update combobox if it exists
        if hasattr(self, "colormap_combo"):
            index = self.colormap_combo.findText(colormap_name)
            if index >= 0:
                self.colormap_combo.setCurrentIndex(index)

        # Get actual colormap object
        if MATPLOTLIB_AVAILABLE:
            if colormap_name in plt.colormaps():
                self.current_colormap = plt.get_cmap(colormap_name)
            else:
                logger.warning(f"Colormap {colormap_name} not found, using default")
                self.current_colormap = plt.get_cmap("jet")
        else:
            # Fallback for non-matplotlib environments
            self.current_colormap = self._create_fallback_colormap(colormap_name)

        # Apply inversion if needed
        if self.invert_colormap and MATPLOTLIB_AVAILABLE:
            self.current_colormap = self.current_colormap.reversed()

        # Update preview
        self.update_colormap_preview()

        # Emit signal with the colormap object
        self.colormap_changed.emit(self.current_colormap)

    def _create_fallback_colormap(self, name):
        """
        Tạo colormap giả khi không có matplotlib.

        Parameters:
        -----------
        name : str
            Tên colormap

        Returns:
        --------
        object
            Đối tượng colormap đơn giản
        """
        # Simple dictionary-based colormap
        if name == "eclipse":
            return {"name": "eclipse", "colors": self.ECLIPSE_COLORS}
        else:
            # Default fallback
            return {
                "name": "default",
                "colors": [
                    (0.0, (0.0, 0.0, 1.0)),
                    (0.5, (0.0, 1.0, 0.0)),
                    (1.0, (1.0, 0.0, 0.0)),
                ],
            }

    def update_colormap_preview(self):
        """Cập nhật hiển thị preview của colormap hiện tại."""
        if not hasattr(self, "preview_frame"):
            return

        if MATPLOTLIB_AVAILABLE and hasattr(self, "preview_ax"):
            # Clear previous content
            self.preview_ax.clear()

            # Create gradient data
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))

            # Display the colormap
            self.preview_ax.imshow(gradient, aspect="auto", cmap=self.current_colormap)
            self.preview_ax.set_axis_off()

            # Update canvas
            self.preview_canvas.draw()
        else:
            # Fallback: use colored frame
            if self.current_colormap_name == "eclipse":
                # Use blue to red gradient for frame background
                self.preview_frame.setStyleSheet(
                    "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                    "stop:0 #0000CC, stop:0.25 #0080FF, stop:0.5 #00FF00, "
                    "stop:0.75 #FFFF00, stop:1 #FF0000);"
                )
            else:
                # Simple blue to red gradient
                self.preview_frame.setStyleSheet(
                    "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                    "stop:0 blue, stop:0.5 green, stop:1 red);"
                )

    def on_colormap_selected(self, colormap_name):
        """
        Xử lý sự kiện khi người dùng chọn colormap mới.

        Parameters:
        -----------
        colormap_name : str
            Tên của colormap được chọn
        """
        self.set_colormap(colormap_name)

    def on_invert_toggled(self, checked):
        """
        Xử lý sự kiện khi người dùng chọn đảo ngược colormap.

        Parameters:
        -----------
        checked : bool
            Trạng thái của checkbox
        """
        self.invert_colormap = checked

        # Re-apply current colormap with inversion
        self.set_colormap(self.current_colormap_name)

    def on_alpha_changed(self, value):
        """
        Xử lý sự kiện khi người dùng thay đổi độ trong suốt.

        Parameters:
        -----------
        value : int
            Giá trị từ slider (0-100)
        """
        self.alpha = value / 100.0
        self.alpha_value_label.setText(f"{value}%")

        # Emit signal that colormap has changed
        self.colormap_changed.emit(self.current_colormap)

    def apply_colormap(self):
        """Áp dụng colormap hiện tại và thông báo cho các thành phần khác."""
        # Emit signal with current settings
        self.colormap_changed.emit(self.current_colormap)

    def show_custom_colormap_dialog(self):
        """Hiển thị hộp thoại tạo colormap tùy chỉnh."""
        if not PYQT5_AVAILABLE:
            return

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Custom Colormap")
        dialog.setMinimumWidth(400)

        # Dialog layout
        layout = QVBoxLayout(dialog)

        # Instruction label
        layout.addWidget(QLabel("Define color points for custom colormap:"))

        # Color point grid
        color_grid = QGridLayout()
        color_grid.addWidget(QLabel("Position (0-1)"), 0, 0)
        color_grid.addWidget(QLabel("Color"), 0, 1)
        color_grid.addWidget(QLabel("Actions"), 0, 2)

        # Initial color points (at least 2)
        color_points = []
        if not self.custom_colormap_colors:
            # Default: blue to red gradient
            color_points = [
                {"position": 0.0, "color": QColor(0, 0, 255)},
                {"position": 1.0, "color": QColor(255, 0, 0)},
            ]
        else:
            # Use existing custom colors
            for pos, (r, g, b) in self.custom_colormap_colors:
                color_points.append(
                    {
                        "position": pos,
                        "color": QColor(int(r * 255), int(g * 255), int(b * 255)),
                    }
                )

        # Create widgets for each color point
        position_inputs = []
        color_buttons = []

        def update_color_row(index):
            position = float(position_inputs[index].text())
            color = color_points[index]["color"]
            color_buttons[index].setStyleSheet(
                f"background-color: {color.name()}; min-width: 24px; min-height: 24px;"
            )
            color_points[index]["position"] = position

        def add_color_row(position, color, row_index):
            # Position input
            position_input = QLineEdit(str(position))
            position_input.setMaximumWidth(80)
            position_inputs.append(position_input)

            # Color button
            color_button = QToolButton()
            color_button.setMinimumSize(24, 24)
            color_button.setStyleSheet(
                f"background-color: {color.name()}; min-width: 24px; min-height: 24px;"
            )
            color_buttons.append(color_button)

            # Add/Remove buttons
            add_button = QToolButton()
            add_button.setText("+")
            add_button.setToolTip("Add color point")

            remove_button = QToolButton()
            remove_button.setText("-")
            remove_button.setToolTip("Remove color point")

            # Add widgets to grid
            color_grid.addWidget(position_input, row_index + 1, 0)
            color_grid.addWidget(color_button, row_index + 1, 1)

            button_layout = QHBoxLayout()
            button_layout.addWidget(add_button)
            button_layout.addWidget(remove_button)
            color_grid.addLayout(button_layout, row_index + 1, 2)

            # Connect signals
            current_index = row_index

            def open_color_dialog():
                color_dialog = QColorDialog(
                    color_points[current_index]["color"], dialog
                )
                if color_dialog.exec_():
                    new_color = color_dialog.selectedColor()
                    color_points[current_index]["color"] = new_color
                    update_color_row(current_index)

            color_button.clicked.connect(open_color_dialog)
            position_input.textChanged.connect(lambda: update_color_row(current_index))

            def add_new_point():
                # Find position for new point
                current_pos = color_points[current_index]["position"]
                if current_index < len(color_points) - 1:
                    next_pos = color_points[current_index + 1]["position"]
                    new_pos = (current_pos + next_pos) / 2
                else:
                    new_pos = min(current_pos + 0.1, 1.0)

                # Interpolate color
                current_color = color_points[current_index]["color"]
                if current_index < len(color_points) - 1:
                    next_color = color_points[current_index + 1]["color"]
                    new_color = QColor(
                        (current_color.red() + next_color.red()) // 2,
                        (current_color.green() + next_color.green()) // 2,
                        (current_color.blue() + next_color.blue()) // 2,
                    )
                else:
                    new_color = current_color

                # Insert new point
                color_points.insert(
                    current_index + 1, {"position": new_pos, "color": new_color}
                )

                # Recreate the dialog with updated points
                dialog.reject()
                self.show_custom_colormap_dialog()

            def remove_point():
                if len(color_points) > 2:  # Keep at least 2 points
                    del color_points[current_index]
                    dialog.reject()
                    self.show_custom_colormap_dialog()
                else:
                    QMessageBox.warning(
                        dialog,
                        "Cannot Remove",
                        "A colormap needs at least 2 color points.",
                    )

            add_button.clicked.connect(add_new_point)
            remove_button.clicked.connect(remove_point)

        # Add rows for each color point
        for i, point in enumerate(color_points):
            add_color_row(point["position"], point["color"], i)

        layout.addLayout(color_grid)

        # Preview
        layout.addWidget(QLabel("Preview:"))
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.StyledPanel)
        preview_frame.setMinimumHeight(30)
        layout.addWidget(preview_frame)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Show dialog and process result
        if dialog.exec_():
            # Convert to format needed for colormap
            self.custom_colormap_colors = []
            for point in color_points:
                pos = point["position"]
                color = point["color"]
                self.custom_colormap_colors.append(
                    (pos, (color.red() / 255, color.green() / 255, color.blue() / 255))
                )

            # Sort by position
            self.custom_colormap_colors.sort(key=lambda x: x[0])

            # Create and register custom colormap
            if MATPLOTLIB_AVAILABLE:
                custom_cmap = LinearSegmentedColormap.from_list(
                    "custom",
                    [(pos, color) for pos, color in self.custom_colormap_colors],
                    N=256,
                )
                plt.register_cmap(name="custom", cmap=custom_cmap)

                # Update combobox if "custom" not already present
                if self.colormap_combo.findText("custom") == -1:
                    self.colormap_combo.addItem("custom")

                # Select the custom colormap
                self.set_colormap("custom")

    def get_colormap_data(self):
        """
        Get the current colormap data for use with other visualization tools.

        Returns:
        --------
        dict
            Dictionary with colormap information
        """
        data = {
            "name": self.current_colormap_name,
            "inverted": self.invert_colormap,
            "alpha": self.alpha,
        }

        # Add colors if it's a custom colormap
        if self.current_colormap_name == "custom" and self.custom_colormap_colors:
            data["colors"] = self.custom_colormap_colors

        return data

    def set_from_colormap_data(self, data):
        """
        Set colormap from saved data.

        Parameters:
        -----------
        data : dict
            Dictionary with colormap information
        """
        if "name" in data:
            # Set inverted flag before setting colormap
            self.invert_colormap = data.get("inverted", False)
            if hasattr(self, "invert_checkbox"):
                self.invert_checkbox.setChecked(self.invert_colormap)

            # Set alpha
            if "alpha" in data:
                self.alpha = data["alpha"]
                if hasattr(self, "alpha_slider"):
                    self.alpha_slider.setValue(int(self.alpha * 100))

            # Set custom colors if available
            if "colors" in data and data["name"] == "custom":
                self.custom_colormap_colors = data["colors"]

                # Create custom colormap if matplotlib is available
                if MATPLOTLIB_AVAILABLE:
                    custom_cmap = LinearSegmentedColormap.from_list(
                        "custom",
                        [(pos, color) for pos, color in self.custom_colormap_colors],
                        N=256,
                    )
                    plt.register_cmap(name="custom", cmap=custom_cmap)

                    # Add to combobox if needed
                    if (
                        hasattr(self, "colormap_combo")
                        and self.colormap_combo.findText("custom") == -1
                    ):
                        self.colormap_combo.addItem("custom")

            # Set the colormap
            self.set_colormap(data["name"])


# Para pruebas independientes
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Create test window
    selector = ColorMapSelector()
    selector.resize(300, 400)
    selector.show()

    # Run application
    sys.exit(app.exec_())
