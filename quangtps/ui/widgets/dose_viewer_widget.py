#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module DoseViewerWidget cho QuangTPS

Widget hiển thị phân bố liều cho phép hiển thị và tương tác với phân bố liều 2D,
tương tự như trong Eclipse của Varian.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

# Thêm xử lý ngoại lệ khi import PyQt5
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QComboBox,
        QSlider,
        QSpinBox,
        QDoubleSpinBox,
        QCheckBox,
        QPushButton,
        QGroupBox,
        QRadioButton,
        QButtonGroup,
        QFrame,
        QSplitter,
        QMenu,
        QToolButton,
        QAction,
        QToolBar,
        QStyle,
        QStyleOption,
        QFormLayout,
        QSizePolicy,
        QColorDialog,
        QMessageBox,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
    from PyQt5.QtGui import (
        QColor,
        QPainter,
        QFont,
        QPixmap,
        QIcon,
        QImage,
        QCursor,
        QPen,
        QBrush,
        QLinearGradient,
        QGradient,
    )

    PYQT_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import PyQt5: {e}")
    PYQT_AVAILABLE = False

    # Tạo các lớp giả để tránh lỗi cú pháp khi không có PyQt5
    class DummyQtClass:
        """Dummy class to replace Qt classes when PyQt5 is not available."""

        pass

    # Tạo các lớp Widget cơ bản
    QWidget = QVBoxLayout = QHBoxLayout = QLabel = QComboBox = QSlider = QSpinBox = (
        DummyQtClass
    )
    QDoubleSpinBox = QCheckBox = QPushButton = QGroupBox = QRadioButton = (
        QButtonGroup
    ) = DummyQtClass
    QFrame = QSplitter = QMenu = QToolButton = QAction = QToolBar = QStyle = (
        QStyleOption
    ) = DummyQtClass
    QFormLayout = QSizePolicy = QColorDialog = QMessageBox = DummyQtClass

    # Tạo các lớp Core
    Qt = QSize = DummyQtClass

    # Tạo lớp signal
    class pyqtSignal:
        """Dummy signal class when PyQt5 is not available."""

        def __init__(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

    pyqtSlot = lambda *args, **kwargs: lambda func: func

    # Tạo các lớp Gui
    QColor = QPainter = QFont = QPixmap = QIcon = QImage = QCursor = QPen = QBrush = (
        DummyQtClass
    )
    QLinearGradient = QGradient = DummyQtClass

# Thêm xử lý ngoại lệ khi import matplotlib
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.figure as figure
    from matplotlib.figure import Figure
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    # Sử dụng Qt5Agg backend
    matplotlib.use("Qt5Agg")

    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import matplotlib: {e}")
    MATPLOTLIB_AVAILABLE = False

    # Tạo các lớp giả để tránh lỗi cú pháp khi không có matplotlib
    class DummyMatplotlibClass:
        """Dummy class to replace matplotlib classes when matplotlib is not available."""

        pass

    matplotlib = DummyMatplotlibClass()
    plt = DummyMatplotlibClass()
    figure = DummyMatplotlibClass()
    Figure = DummyMatplotlibClass
    mcolors = DummyMatplotlibClass()
    LinearSegmentedColormap = DummyMatplotlibClass
    gridspec = DummyMatplotlibClass()

    class FigureCanvas(DummyMatplotlibClass):
        """Dummy FigureCanvas class."""

        pass


# Import các thành phần phụ thuộc với xử lý ngoại lệ
try:
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.imaging.image_utils import apply_window_level, overlay_dose_on_image
    from quangtps.ui.styles.color_maps import get_available_colormaps, get_colormap
except ImportError as e:
    logging.error(f"Error importing dose viewer dependencies: {e}")

logger = logging.getLogger(__name__)


class ColorMapSelector(QWidget):
    """Widget cho phép chọn và tùy chỉnh colormap."""

    colormap_changed = pyqtSignal(str, float, float, float)

    def __init__(self, parent=None):
        """
        Khởi tạo ColorMapSelector.

        Parameters:
            parent (QWidget, optional): Widget cha
        """
        super().__init__(parent)

        # Dữ liệu
        self.colormaps = {
            "Eclipse": {
                "colors": [
                    (0, 0, 0.5),
                    (0, 0, 1),
                    (0, 1, 1),
                    (0, 1, 0),
                    (1, 1, 0),
                    (1, 0, 0),
                ],
                "positions": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            },
            "Đỏ-Xanh": {
                "colors": [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)],
                "positions": [0, 0.25, 0.5, 0.75, 1.0],
            },
            "Hot": {
                "colors": [
                    (0, 0, 0),
                    (0.5, 0, 0),
                    (1, 0, 0),
                    (1, 0.5, 0),
                    (1, 1, 0),
                    (1, 1, 1),
                ],
                "positions": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            },
            "Cầu vồng": {
                "colors": [
                    (0.5, 0, 0.5),
                    (0, 0, 1),
                    (0, 1, 1),
                    (0, 1, 0),
                    (1, 1, 0),
                    (1, 0, 0),
                ],
                "positions": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            },
            "Grayscale": {
                "colors": [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)],
                "positions": [0, 0.5, 1.0],
            },
        }

        self.selected_colormap = "Eclipse"
        self.min_value = 0.0
        self.max_value = 100.0
        self.alpha = 0.8

        # Tạo layout
        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        main_layout = QVBoxLayout(self)

        # Phần trên: Chọn colormap
        top_layout = QHBoxLayout()

        # Label và combo box chọn colormap
        top_layout.addWidget(QLabel("Colormap:"))

        self.colormap_combo = QComboBox()
        for cmap_name in self.colormaps.keys():
            self.colormap_combo.addItem(cmap_name)
        self.colormap_combo.setCurrentText(self.selected_colormap)
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)

        top_layout.addWidget(self.colormap_combo)
        top_layout.addStretch()

        main_layout.addLayout(top_layout)

        # Phần giữa: Preview colormap
        self.preview = ColorMapPreview()
        self.preview.set_colormap(
            self.selected_colormap, self.colormaps[self.selected_colormap]
        )
        main_layout.addWidget(self.preview)

        # Phần dưới: Điều chỉnh min/max và độ trong suốt
        controls_layout = QFormLayout()

        # Thang giá trị tối thiểu
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(0, 1000)
        self.min_spin.setValue(self.min_value)
        self.min_spin.setSingleStep(1.0)
        self.min_spin.setSuffix(" %")
        self.min_spin.valueChanged.connect(self.on_value_changed)
        controls_layout.addRow("Tối thiểu:", self.min_spin)

        # Thang giá trị tối đa
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(0, 1000)
        self.max_spin.setValue(self.max_value)
        self.max_spin.setSingleStep(1.0)
        self.max_spin.setSuffix(" %")
        self.max_spin.valueChanged.connect(self.on_value_changed)
        controls_layout.addRow("Tối đa:", self.max_spin)

        # Độ trong suốt
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(self.alpha * 100))
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(self.alpha_slider)
        self.alpha_label = QLabel(f"{self.alpha:.2f}")
        alpha_layout.addWidget(self.alpha_label)

        controls_layout.addRow("Độ trong suốt:", alpha_layout)

        main_layout.addLayout(controls_layout)

    def on_colormap_changed(self, colormap_name):
        """
        Xử lý khi colormap được thay đổi.

        Parameters:
            colormap_name (str): Tên colormap mới
        """
        self.selected_colormap = colormap_name
        self.preview.set_colormap(colormap_name, self.colormaps[colormap_name])
        self.emit_change()

    def on_value_changed(self):
        """Xử lý khi giá trị min/max thay đổi."""
        self.min_value = self.min_spin.value()
        self.max_value = self.max_spin.value()

        # Đảm bảo min < max
        if self.min_value >= self.max_value:
            if self.sender() == self.min_spin:
                self.min_value = self.max_value - 1.0
                self.min_spin.setValue(self.min_value)
            else:
                self.max_value = self.min_value + 1.0
                self.max_spin.setValue(self.max_value)

        self.emit_change()

    def on_alpha_changed(self, value):
        """
        Xử lý khi độ trong suốt thay đổi.

        Parameters:
            value (int): Giá trị mới (0-100)
        """
        self.alpha = value / 100.0
        self.alpha_label.setText(f"{self.alpha:.2f}")
        self.emit_change()

    def emit_change(self):
        """Phát tín hiệu thay đổi colormap."""
        self.colormap_changed.emit(
            self.selected_colormap, self.min_value, self.max_value, self.alpha
        )

    def set_value_range(self, min_value, max_value):
        """
        Đặt dải giá trị.

        Parameters:
            min_value (float): Giá trị tối thiểu
            max_value (float): Giá trị tối đa
        """
        self.min_spin.setValue(min_value)
        self.max_spin.setValue(max_value)


class ColorMapPreview(QWidget):
    """Widget hiển thị preview của colormap."""

    def __init__(self, parent=None):
        """
        Khởi tạo ColorMapPreview.

        Parameters:
            parent (QWidget, optional): Widget cha
        """
        super().__init__(parent)

        self.colormap_name = ""
        self.colormap_data = None

        # Đặt kích thước tối thiểu
        self.setMinimumSize(200, 30)

    def set_colormap(self, name, data):
        """
        Đặt colormap để hiển thị.

        Parameters:
            name (str): Tên colormap
            data (dict): Dữ liệu colormap
        """
        self.colormap_name = name
        self.colormap_data = data
        self.update()

    def paintEvent(self, event):
        """
        Vẽ preview colormap.

        Parameters:
            event: Sự kiện paint
        """
        if not self.colormap_data:
            return

        # Thiết lập painter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Tạo gradient từ dữ liệu colormap
        gradient = QLinearGradient(0, 0, self.width(), 0)

        for pos, color in zip(
            self.colormap_data["positions"], self.colormap_data["colors"]
        ):
            r, g, b = color
            gradient.setColorAt(pos, QColor(int(r * 255), int(g * 255), int(b * 255)))

        # Vẽ gradient
        painter.fillRect(0, 0, self.width(), self.height(), gradient)

        # Vẽ viền
        painter.setPen(Qt.black)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)


class DoseViewerWidget(QWidget):
    """
    Widget hiển thị phân bố liều.

    Widget này hiển thị phân bố liều trong các mặt cắt Axial, Sagittal, Coronal
    và cung cấp các công cụ điều khiển hiển thị.
    """

    def __init__(self, parent=None):
        """
        Khởi tạo DoseViewerWidget.

        Parameters:
            parent (QWidget, optional): Widget cha
        """
        super().__init__(parent)

        # Dữ liệu
        self.dose_data = None
        self.image_data = None
        self.current_slice = {"axial": 0, "sagittal": 0, "coronal": 0}
        self.colormap_name = "Eclipse"
        self.min_value = 0.0
        self.max_value = 100.0
        self.alpha = 0.8
        self.show_isodose = True
        self.isodose_levels = [10, 30, 50, 70, 80, 90, 95, 100]

        # Tạo layout
        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        main_layout = QVBoxLayout(self)

        # Toolbar với các nút điều khiển
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))

        # Các nút hiển thị
        self.axial_action = QAction("Axial", self)
        self.axial_action.setCheckable(True)
        self.axial_action.setChecked(True)
        toolbar.addAction(self.axial_action)

        self.sagittal_action = QAction("Sagittal", self)
        self.sagittal_action.setCheckable(True)
        self.sagittal_action.setChecked(True)
        toolbar.addAction(self.sagittal_action)

        self.coronal_action = QAction("Coronal", self)
        self.coronal_action.setCheckable(True)
        self.coronal_action.setChecked(True)
        toolbar.addAction(self.coronal_action)

        toolbar.addSeparator()

        # Các nút hiển thị
        self.isodose_action = QAction("Isodose", self)
        self.isodose_action.setCheckable(True)
        self.isodose_action.setChecked(self.show_isodose)
        self.isodose_action.toggled.connect(self.toggle_isodose)
        toolbar.addAction(self.isodose_action)

        # Nút cấu hình isodose
        configure_isodose_button = QToolButton()
        configure_isodose_button.setText("Cấu hình")
        configure_isodose_button.clicked.connect(self.configure_isodose)
        toolbar.addWidget(configure_isodose_button)

        main_layout.addWidget(toolbar)

        # Container chính chứa màn hình và điều khiển
        main_container = QSplitter(Qt.Horizontal)

        # Bên trái: Màn hình hiển thị
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(0, 0, 0, 0)

        # Tạo các trục matplotlib
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        gs = gridspec.GridSpec(2, 2)
        self.axial_ax = self.fig.add_subplot(gs[0, 0])
        self.sagittal_ax = self.fig.add_subplot(gs[0, 1])
        self.coronal_ax = self.fig.add_subplot(gs[1, 0])
        self.colorbar_ax = self.fig.add_subplot(gs[1, 1])

        # Thiết lập các trục
        self.axial_ax.set_title("Axial")
        self.sagittal_ax.set_title("Sagittal")
        self.coronal_ax.set_title("Coronal")

        # Thêm canvas vào layout
        display_layout.addWidget(self.canvas)

        # Thanh trượt điều chỉnh lát cắt
        slice_control_layout = QHBoxLayout()

        # Axial
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setRange(0, 0)
        self.axial_slider.valueChanged.connect(lambda v: self.change_slice("axial", v))

        axial_layout = QVBoxLayout()
        axial_layout.addWidget(QLabel("Axial:"))
        axial_layout.addWidget(self.axial_slider)

        # Sagittal
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setRange(0, 0)
        self.sagittal_slider.valueChanged.connect(
            lambda v: self.change_slice("sagittal", v)
        )

        sagittal_layout = QVBoxLayout()
        sagittal_layout.addWidget(QLabel("Sagittal:"))
        sagittal_layout.addWidget(self.sagittal_slider)

        # Coronal
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setRange(0, 0)
        self.coronal_slider.valueChanged.connect(
            lambda v: self.change_slice("coronal", v)
        )

        coronal_layout = QVBoxLayout()
        coronal_layout.addWidget(QLabel("Coronal:"))
        coronal_layout.addWidget(self.coronal_slider)

        slice_control_layout.addLayout(axial_layout)
        slice_control_layout.addLayout(sagittal_layout)
        slice_control_layout.addLayout(coronal_layout)

        display_layout.addLayout(slice_control_layout)

        # Bên phải: Điều khiển hiển thị
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # GroupBox colormap
        colormap_group = QGroupBox("Colormap")
        colormap_layout = QVBoxLayout(colormap_group)

        self.colormap_selector = ColorMapSelector()
        self.colormap_selector.colormap_changed.connect(self.change_colormap)

        colormap_layout.addWidget(self.colormap_selector)
        control_layout.addWidget(colormap_group)

        # GroupBox isodose
        isodose_group = QGroupBox("Đường đồng liều")
        isodose_layout = QVBoxLayout(isodose_group)

        self.isodose_check = QCheckBox("Hiển thị đường đồng liều")
        self.isodose_check.setChecked(self.show_isodose)
        self.isodose_check.toggled.connect(self.toggle_isodose)
        isodose_layout.addWidget(self.isodose_check)

        # Nút cấu hình isodose
        isodose_button = QPushButton("Cấu hình đường đồng liều")
        isodose_button.clicked.connect(self.configure_isodose)
        isodose_layout.addWidget(isodose_button)

        control_layout.addWidget(isodose_group)

        # Thêm stretch để đẩy các controls lên trên
        control_layout.addStretch()

        # Thêm các widgets vào splitter
        main_container.addWidget(display_widget)
        main_container.addWidget(control_widget)

        # Thiết lập kích thước ban đầu (70% hiển thị, 30% điều khiển)
        main_container.setSizes([700, 300])

        main_layout.addWidget(main_container)

        # Khởi tạo hiển thị
        self.update_display()

    def set_dose_data(self, dose_data, image_data=None):
        """
        Đặt dữ liệu liều để hiển thị.

        Parameters:
            dose_data (np.ndarray): Dữ liệu liều 3D
            image_data (np.ndarray, optional): Dữ liệu hình ảnh CT/MRI
        """
        self.dose_data = dose_data

        if image_data is not None:
            self.image_data = image_data

        if self.dose_data is not None:
            # Cập nhật thang giá trị
            max_dose = np.max(self.dose_data)
            self.colormap_selector.set_value_range(0, max_dose * 100)

            # Cập nhật thanh trượt lát cắt
            depth, height, width = self.dose_data.shape
            self.axial_slider.setRange(0, depth - 1)
            self.sagittal_slider.setRange(0, width - 1)
            self.coronal_slider.setRange(0, height - 1)

            # Đặt lát cắt ban đầu ở giữa
            self.current_slice = {
                "axial": depth // 2,
                "sagittal": width // 2,
                "coronal": height // 2,
            }

            self.axial_slider.setValue(self.current_slice["axial"])
            self.sagittal_slider.setValue(self.current_slice["sagittal"])
            self.coronal_slider.setValue(self.current_slice["coronal"])

        # Cập nhật hiển thị
        self.update_display()

    def update_display(self):
        """Cập nhật hiển thị phân bố liều."""
        # Xóa các trục
        self.axial_ax.clear()
        self.sagittal_ax.clear()
        self.coronal_ax.clear()
        self.colorbar_ax.clear()

        if self.dose_data is None:
            # Hiển thị thông báo nếu không có dữ liệu
            for ax in [self.axial_ax, self.sagittal_ax, self.coronal_ax]:
                ax.text(
                    0.5,
                    0.5,
                    "Không có dữ liệu liều",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            self.canvas.draw()
            return

        # Lấy dữ liệu từng lát cắt
        axial_slice = self.dose_data[self.current_slice["axial"], :, :]
        sagittal_slice = self.dose_data[:, :, self.current_slice["sagittal"]]
        coronal_slice = self.dose_data[:, self.current_slice["coronal"], :]

        # Tạo colormap từ tên
        if self.colormap_name in self.colormap_selector.colormaps:
            cmap_data = self.colormap_selector.colormaps[self.colormap_name]
            cmap = LinearSegmentedColormap.from_list(
                self.colormap_name, cmap_data["colors"], N=256
            )
        else:
            cmap = plt.get_cmap("jet")  # Fallback

        # Thiết lập giá trị min/max
        vmin = self.min_value / 100.0 * np.max(self.dose_data)
        vmax = self.max_value / 100.0 * np.max(self.dose_data)

        # Hiển thị từng lát cắt
        im_axial = self.axial_ax.imshow(
            axial_slice, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal"
        )
        im_sagittal = self.sagittal_ax.imshow(
            sagittal_slice, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal"
        )
        im_coronal = self.coronal_ax.imshow(
            coronal_slice, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal"
        )

        # Hiển thị đường đồng liều nếu được bật
        if self.show_isodose:
            max_dose = np.max(self.dose_data)
            for level in self.isodose_levels:
                # Tính giá trị ngưỡng
                threshold = level / 100.0 * max_dose

                # Vẽ contour cho từng mặt cắt
                self.axial_ax.contour(
                    axial_slice, levels=[threshold], colors="white", linewidths=1
                )
                self.sagittal_ax.contour(
                    sagittal_slice, levels=[threshold], colors="white", linewidths=1
                )
                self.coronal_ax.contour(
                    coronal_slice, levels=[threshold], colors="white", linewidths=1
                )

        # Hiển thị colorbar
        cbar = plt.colorbar(im_axial, cax=self.colorbar_ax)
        cbar.set_label("Dose [%]")

        # Cài đặt tiêu đề cho mỗi mặt cắt
        self.axial_ax.set_title(f"Axial - Slice {self.current_slice['axial']}")
        self.sagittal_ax.set_title(f"Sagittal - Slice {self.current_slice['sagittal']}")
        self.coronal_ax.set_title(f"Coronal - Slice {self.current_slice['coronal']}")

        # Tắt hiển thị các trục
        for ax in [self.axial_ax, self.sagittal_ax, self.coronal_ax]:
            ax.axis("off")

        # Vẽ lại canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def change_slice(self, view, value):
        """
        Thay đổi lát cắt hiển thị.

        Parameters:
            view (str): Tên góc nhìn ("axial", "sagittal", "coronal")
            value (int): Chỉ số lát cắt
        """
        self.current_slice[view] = value
        self.update_display()

    def change_colormap(self, name, min_value, max_value, alpha):
        """
        Thay đổi colormap.

        Parameters:
            name (str): Tên colormap
            min_value (float): Giá trị tối thiểu
            max_value (float): Giá trị tối đa
            alpha (float): Độ trong suốt
        """
        self.colormap_name = name
        self.min_value = min_value
        self.max_value = max_value
        self.alpha = alpha
        self.update_display()

    def toggle_isodose(self, show):
        """
        Bật/tắt hiển thị đường đồng liều.

        Parameters:
            show (bool): Trạng thái hiển thị
        """
        # Đảm bảo cả checkbox và action đều cùng trạng thái
        self.show_isodose = show
        self.isodose_check.setChecked(show)
        self.isodose_action.setChecked(show)
        self.update_display()

    def configure_isodose(self):
        """Mở hộp thoại cấu hình đường đồng liều."""
        # TODO: Thêm hộp thoại cấu hình đường đồng liều
        QMessageBox.information(
            self, "Cấu hình đường đồng liều", "Chức năng đang được phát triển!"
        )

    def export_image(self, filename, dpi=300):
        """
        Xuất hình ảnh hiển thị ra file.

        Parameters:
            filename (str): Đường dẫn file đầu ra
            dpi (int, optional): Độ phân giải (DPI)
        """
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def keyPressEvent(self, event):
        """
        Xử lý các phím tắt.

        Parameters:
            event: Sự kiện bàn phím
        """
        # TODO: Thêm xử lý phím tắt
        super().keyPressEvent(event)
