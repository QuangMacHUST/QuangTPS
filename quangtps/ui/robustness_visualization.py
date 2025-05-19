#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module trực quan hóa kết quả phân tích độ bền vững.

Module này cung cấp các công cụ trực quan hóa nâng cao để hiển thị
kết quả phân tích độ bền vững của kế hoạch xạ trị, với các biểu đồ
và heatmap 3D phong cách Eclipse.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union, Tuple

try:
    from PyQt5.QtCore import Qt, pyqtSignal, QObject
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QCheckBox,
        QSlider,
        QSplitter,
        QTabWidget,
        QColorDialog,
        QFileDialog,
        QMessageBox,
    )
    from PyQt5.QtGui import QColor, QPixmap, QPainter, QIcon
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_PYQT = True
except ImportError:
    try:
        # Fallback for when PyQt is not available
        from PySide2.QtCore import Qt, Signal as pyqtSignal, QObject
        from PySide2.QtWidgets import (
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QComboBox,
            QCheckBox,
            QSlider,
            QSplitter,
            QTabWidget,
            QColorDialog,
            QFileDialog,
            QMessageBox,
        )
        from PySide2.QtGui import QColor, QPixmap, QPainter, QIcon
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        HAS_PYQT = True
    except ImportError:
        HAS_PYQT = False

# Try to import VTK for 3D visualization
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    HAS_VTK = True
except ImportError:
    HAS_VTK = False

# Try to import DoseColormap utility
try:
    from quangtps.ui.utils.dose_colormap import DoseColormap

    HAS_DOSE_COLORMAP = True
except ImportError:
    HAS_DOSE_COLORMAP = False

logger = logging.getLogger(__name__)


class RobustnessVisualization(QWidget):
    """
    Widget trực quan hóa kết quả phân tích độ bền vững.

    Widget này hiển thị kết quả phân tích độ bền vững với các biểu đồ
    nâng cao, heatmap 3D, và các công cụ tương tác cho phép người dùng
    khám phá sự thay đổi của phân phối liều trong các kịch bản khác nhau.
    """

    # Signals
    viewUpdated = pyqtSignal()  # Phát khi view được cập nhật

    def __init__(self, parent=None):
        """
        Khởi tạo widget trực quan hóa độ bền vững.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, by default None
        """
        super().__init__(parent)

        # Dữ liệu nội bộ
        self.robustness_result = None
        self.dose_grid = None
        self.structure_masks = {}
        self.current_view = "axial"  # axial, sagittal, coronal, 3d
        self.current_slice = 0
        self.current_scenario = 0
        self.display_mode = "difference"  # difference, gamma, std_dev, min_max

        # Thiết lập UI
        if HAS_PYQT:
            self._setup_ui()
        else:
            # Placeholder khi không có PyQt
            layout = QVBoxLayout(self)
            label = QLabel("PyQt hoặc PySide là bắt buộc cho RobustnessVisualization")
            layout.addWidget(label)

    def _setup_ui(self):
        """Thiết lập giao diện người dùng cho trực quan hóa độ bền vững."""
        main_layout = QVBoxLayout(self)

        # Tiêu đề
        title_layout = QHBoxLayout()
        title_label = QLabel("Trực quan hóa độ bền vững")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #003366;")
        title_layout.addWidget(title_label)

        # Điều khiển hiển thị
        title_layout.addStretch()
        title_layout.addWidget(QLabel("Hiển thị:"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(
            ["Dose Difference", "Gamma Index", "Standard Deviation", "Min-Max Range"]
        )
        self.display_mode_combo.currentIndexChanged.connect(self._update_display_mode)
        title_layout.addWidget(self.display_mode_combo)

        main_layout.addLayout(title_layout)

        # Splitter chính để chia màn hình
        main_splitter = QSplitter(Qt.Horizontal)

        # Panel bên trái: Điều khiển và thông tin
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Lựa chọn View
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Axial", "Sagittal", "Coronal", "3D"])
        self.view_combo.currentIndexChanged.connect(self._update_view)
        view_layout.addWidget(self.view_combo)

        control_layout.addLayout(view_layout)

        # Slider chọn lát cắt
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(99)  # Sẽ được cập nhật sau
        self.slice_slider.valueChanged.connect(self._update_slice)
        slice_layout.addWidget(self.slice_slider)
        self.slice_value_label = QLabel("0")
        slice_layout.addWidget(self.slice_value_label)

        control_layout.addLayout(slice_layout)

        # Điều khiển kịch bản (scenario)
        scenario_layout = QHBoxLayout()
        scenario_layout.addWidget(QLabel("Scenario:"))
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItem("Nominal")
        scenario_layout.addWidget(self.scenario_combo)

        control_layout.addLayout(scenario_layout)

        # Checkbox hiển thị
        display_options_layout = QVBoxLayout()

        self.show_structures_check = QCheckBox("Hiển thị cấu trúc")
        self.show_structures_check.setChecked(True)
        self.show_structures_check.stateChanged.connect(self._update_display)
        display_options_layout.addWidget(self.show_structures_check)

        self.show_isodose_check = QCheckBox("Hiển thị đường đồng liều")
        self.show_isodose_check.setChecked(True)
        self.show_isodose_check.stateChanged.connect(self._update_display)
        display_options_layout.addWidget(self.show_isodose_check)

        self.show_colorbar_check = QCheckBox("Hiển thị thanh màu")
        self.show_colorbar_check.setChecked(True)
        self.show_colorbar_check.stateChanged.connect(self._update_display)
        display_options_layout.addWidget(self.show_colorbar_check)

        control_layout.addLayout(display_options_layout)

        # Thông tin giá trị
        info_layout = QVBoxLayout()

        self.value_info_label = QLabel("Giá trị: N/A")
        info_layout.addWidget(self.value_info_label)

        self.position_info_label = QLabel("Vị trí: N/A")
        info_layout.addWidget(self.position_info_label)

        self.structure_info_label = QLabel("Cấu trúc: N/A")
        info_layout.addWidget(self.structure_info_label)

        control_layout.addLayout(info_layout)

        # Thêm stretch để đẩy nút điều khiển xuống dưới
        control_layout.addStretch()

        # Các nút điều khiển
        buttons_layout = QHBoxLayout()

        self.save_image_button = QPushButton("Lưu ảnh")
        self.save_image_button.clicked.connect(self._save_current_view)
        buttons_layout.addWidget(self.save_image_button)

        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.clicked.connect(self._reset_view)
        buttons_layout.addWidget(self.reset_view_button)

        control_layout.addLayout(buttons_layout)

        # Thêm panel điều khiển vào splitter
        main_splitter.addWidget(control_widget)

        # Panel bên phải: Hiển thị chính
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)

        # Tạo figure cho hiển thị 2D
        fig = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(fig)
        self.ax = fig.add_subplot(111)
        display_layout.addWidget(self.canvas)

        # Tạo VTK viewer (nếu có) cho hiển thị 3D
        if HAS_VTK:
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            self.vtk_renderer = vtk.vtkRenderer()
            self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)
            self.vtk_interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

            # Thêm vào layout nhưng ẩn đi ban đầu
            display_layout.addWidget(self.vtk_widget)
            self.vtk_widget.hide()

        # Thêm panel hiển thị vào splitter
        main_splitter.addWidget(display_widget)

        # Thiết lập tỉ lệ ban đầu cho splitter
        main_splitter.setSizes([int(self.width() * 0.3), int(self.width() * 0.7)])

        # Thêm splitter vào layout chính
        main_layout.addWidget(main_splitter)

        # Khởi tạo hiển thị
        self._init_display()

    def _init_display(self):
        """Khởi tạo hiển thị ban đầu."""
        # Xóa biểu đồ hiện tại
        self.ax.clear()

        # Thiết lập layout cho figure
        self.ax.set_title("Phân tích độ bền vững")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.grid(False)

        # Hiển thị placeholder
        self.ax.text(
            0.5,
            0.5,
            "Không có dữ liệu phân tích độ bền vững",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.ax.transAxes,
            fontsize=14,
        )

        # Cập nhật canvas
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def set_data(self, robustness_result, dose_grid=None, structure_masks=None):
        """
        Đặt dữ liệu kết quả phân tích độ bền vững để hiển thị.

        Parameters
        ----------
        robustness_result : Any
            Đối tượng kết quả phân tích độ bền vững
        dose_grid : numpy.ndarray, optional
            Mảng 3D lưới liều, by default None
        structure_masks : Dict[str, numpy.ndarray], optional
            Dictionary masks của các cấu trúc, by default None
        """
        self.robustness_result = robustness_result
        self.dose_grid = dose_grid

        if structure_masks:
            self.structure_masks = structure_masks

        # Cập nhật danh sách kịch bản
        self._update_scenario_list()

        # Cập nhật slider lát cắt
        self._update_slice_range()

        # Cập nhật hiển thị
        self._update_display()

    def _update_scenario_list(self):
        """Cập nhật danh sách kịch bản từ kết quả phân tích."""
        if not self.robustness_result:
            return

        # Lưu kịch bản đang chọn
        current_scenario = self.scenario_combo.currentText()

        # Xóa và cập nhật combo
        self.scenario_combo.clear()
        self.scenario_combo.addItem("Nominal")

        try:
            # Lấy danh sách kịch bản từ kết quả
            scenarios = self.robustness_result.get_scenario_names()

            if scenarios:
                for scenario in scenarios:
                    if scenario != "Nominal":
                        self.scenario_combo.addItem(scenario)

                # Chọn lại kịch bản trước đó nếu có
                if current_scenario:
                    index = self.scenario_combo.findText(current_scenario)
                    if index >= 0:
                        self.scenario_combo.setCurrentIndex(index)
                    else:
                        self.scenario_combo.setCurrentIndex(0)  # Nominal

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật danh sách kịch bản: {str(e)}")

    def _update_slice_range(self):
        """Cập nhật phạm vi của slider lát cắt dựa trên dữ liệu liều."""
        if self.dose_grid is not None:
            # Lấy kích thước lưới liều
            z_size, y_size, x_size = self.dose_grid.shape

            # Cập nhật max của slider dựa trên view hiện tại
            if self.current_view == "axial":
                self.slice_slider.setMaximum(z_size - 1)
            elif self.current_view == "sagittal":
                self.slice_slider.setMaximum(x_size - 1)
            elif self.current_view == "coronal":
                self.slice_slider.setMaximum(y_size - 1)

            # Đảm bảo giá trị hiện tại hợp lệ
            if self.current_slice >= self.slice_slider.maximum():
                self.current_slice = self.slice_slider.maximum() // 2
                self.slice_slider.setValue(self.current_slice)

            # Cập nhật label
            self.slice_value_label.setText(str(self.current_slice))

    def _update_view(self):
        """Cập nhật view hiển thị khi người dùng thay đổi combobox."""
        view_text = self.view_combo.currentText().lower()

        # Chuyển đổi từ văn bản sang mã view
        if view_text == "3d":
            self.current_view = "3d"

            # Ẩn hiển thị 2D và hiện hiển thị 3D
            if HAS_VTK:
                self.canvas.hide()
                self.vtk_widget.show()

                # Vô hiệu hóa slider lát cắt
                self.slice_slider.setEnabled(False)

                # Khởi tạo view 3D
                self._setup_3d_visualization()
        else:
            # Ẩn hiển thị 3D (nếu có) và hiện hiển thị 2D
            if HAS_VTK and self.vtk_widget.isVisible():
                self.vtk_widget.hide()
                self.canvas.show()

            # Kích hoạt slider lát cắt
            self.slice_slider.setEnabled(True)

            # Cập nhật view hiện tại
            self.current_view = view_text

            # Cập nhật phạm vi slider
            self._update_slice_range()

            # Cập nhật hiển thị
            self._update_display()

    def _update_slice(self):
        """Cập nhật lát cắt hiển thị khi người dùng thay đổi slider."""
        self.current_slice = self.slice_slider.value()
        self.slice_value_label.setText(str(self.current_slice))

        # Chỉ cập nhật hiển thị nếu đang ở chế độ 2D
        if self.current_view != "3d":
            self._update_display()

    def _update_display_mode(self):
        """Cập nhật chế độ hiển thị khi người dùng thay đổi combobox."""
        display_text = self.display_mode_combo.currentText()

        # Chuyển đổi từ văn bản sang mã chế độ hiển thị
        if display_text == "Dose Difference":
            self.display_mode = "difference"
        elif display_text == "Gamma Index":
            self.display_mode = "gamma"
        elif display_text == "Standard Deviation":
            self.display_mode = "std_dev"
        elif display_text == "Min-Max Range":
            self.display_mode = "min_max"

        # Cập nhật hiển thị
        self._update_display()

    def _update_display(self):
        """Cập nhật hiển thị dựa trên các thiết lập hiện tại."""
        if not self.robustness_result or self.dose_grid is None:
            return

        # Lấy kịch bản hiện tại
        current_scenario_text = self.scenario_combo.currentText()

        try:
            # Hiển thị dựa trên view
            if self.current_view == "3d":
                self._update_3d_display(current_scenario_text)
            else:
                self._update_2d_display(current_scenario_text)

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật hiển thị: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

    def _update_2d_display(self, scenario_name):
        """
        Cập nhật hiển thị 2D cho lát cắt hiện tại.

        Parameters
        ----------
        scenario_name : str
            Tên kịch bản đang hiển thị
        """
        # Xóa biểu đồ hiện tại
        self.ax.clear()

        try:
            # Lấy dữ liệu phân tích cho kịch bản
            scenario_data = self.robustness_result.get_scenario_data(scenario_name)

            if not scenario_data:
                self.ax.text(
                    0.5,
                    0.5,
                    f"Không có dữ liệu cho kịch bản {scenario_name}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=self.ax.transAxes,
                )
                self.canvas.draw()
                return

            # Lấy dữ liệu hiển thị phù hợp với chế độ
            if self.display_mode == "difference":
                display_data = scenario_data.get("difference")
                cmap = "RdBu_r"
                title_suffix = "Chênh lệch liều (Gy)"
            elif self.display_mode == "gamma":
                display_data = scenario_data.get("gamma")
                cmap = "viridis"
                title_suffix = "Chỉ số Gamma"
            elif self.display_mode == "std_dev":
                display_data = scenario_data.get("std_dev")
                cmap = "hot"
                title_suffix = "Độ lệch chuẩn (Gy)"
            elif self.display_mode == "min_max":
                display_data = scenario_data.get("range")
                cmap = "plasma"
                title_suffix = "Phạm vi Min-Max (Gy)"

            # Nếu không có dữ liệu hiển thị, hiển thị thông báo
            if display_data is None:
                self.ax.text(
                    0.5,
                    0.5,
                    f"Không có dữ liệu {self.display_mode} cho kịch bản {scenario_name}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=self.ax.transAxes,
                )
                self.canvas.draw()
                return

            # Lấy lát cắt dựa trên view
            if self.current_view == "axial":
                slice_data = display_data[self.current_slice, :, :]
                x_label, y_label = "X (mm)", "Y (mm)"
            elif self.current_view == "sagittal":
                slice_data = display_data[:, :, self.current_slice]
                x_label, y_label = "Y (mm)", "Z (mm)"
            elif self.current_view == "coronal":
                slice_data = display_data[:, self.current_slice, :]
                x_label, y_label = "X (mm)", "Z (mm)"

            # Nếu dùng dose colormap
            if HAS_DOSE_COLORMAP and self.display_mode in [
                "difference",
                "std_dev",
                "min_max",
            ]:
                colormap = DoseColormap().get_colormap(self.display_mode)
                if colormap:
                    cmap = colormap

            # Hiển thị heatmap
            im = self.ax.imshow(
                slice_data, cmap=cmap, origin="lower", interpolation="bilinear"
            )

            # Thêm thanh màu nếu được chọn
            if self.show_colorbar_check.isChecked():
                self.canvas.figure.colorbar(im, ax=self.ax, label=title_suffix)

            # Hiển thị đường đồng liều nếu được chọn
            if self.show_isodose_check.isChecked():
                # Tính phạm vi giá trị để tạo mức đồng liều
                vmin, vmax = np.nanmin(slice_data), np.nanmax(slice_data)

                if vmin < vmax:
                    # Tạo các mức đồng liều phù hợp với dữ liệu
                    if self.display_mode == "gamma":
                        levels = [0.5, 1.0, 1.5]  # Mức đặc biệt cho gamma index
                    else:
                        # Tạo 5 mức đồng liều từ min đến max
                        levels = np.linspace(vmin, vmax, 5)

                    # Vẽ các đường đồng liều
                    contours = self.ax.contour(
                        slice_data,
                        levels=levels,
                        colors="white",
                        linewidths=0.5,
                        alpha=0.7,
                    )

                    # Thêm labels cho contours
                    self.ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

            # Hiển thị cấu trúc nếu được chọn
            if self.show_structures_check.isChecked() and self.structure_masks:
                for name, mask in self.structure_masks.items():
                    # Lấy mask cho lát cắt hiện tại
                    if self.current_view == "axial":
                        structure_slice = mask[self.current_slice, :, :]
                    elif self.current_view == "sagittal":
                        structure_slice = mask[:, :, self.current_slice]
                    elif self.current_view == "coronal":
                        structure_slice = mask[:, self.current_slice, :]

                    # Vẽ đường viền cho cấu trúc
                    self.ax.contour(
                        structure_slice,
                        levels=[0.5],
                        colors=[
                            "r"
                            if name.lower().startswith(("ptv", "ctv", "gtv"))
                            else "g"
                        ],
                        linewidths=1,
                    )

            # Thiết lập labels và tiêu đề
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label)
            self.ax.set_title(
                f"{scenario_name} - {title_suffix} - {self.current_view.capitalize()} View"
            )

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật hiển thị 2D: {str(e)}")
            self.ax.text(
                0.5,
                0.5,
                f"Lỗi: {str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.ax.transAxes,
            )

        # Cập nhật canvas
        self.canvas.figure.tight_layout()
        self.canvas.draw()

        # Phát tín hiệu cập nhật
        self.viewUpdated.emit()

    def _setup_3d_visualization(self):
        """Thiết lập hiển thị 3D với VTK."""
        if not HAS_VTK:
            logger.warning("VTK không khả dụng, không thể tạo hiển thị 3D")
            return

        # Xóa tất cả các actors hiện tại
        self.vtk_renderer.RemoveAllViewProps()

        # Tạo axes
        axes = vtk.vtkAxesActor()
        self.vtk_renderer.AddActor(axes)

        # Thiết lập camera
        self.vtk_renderer.ResetCamera()
        self.vtk_renderer.GetActiveCamera().Elevation(30)
        self.vtk_renderer.GetActiveCamera().Azimuth(30)

        # Cập nhật renderer
        self.vtk_widget.GetRenderWindow().Render()

    def _update_3d_display(self, scenario_name):
        """
        Cập nhật hiển thị 3D cho kịch bản hiện tại.

        Parameters
        ----------
        scenario_name : str
            Tên kịch bản đang hiển thị
        """
        if not HAS_VTK:
            logger.warning("VTK không khả dụng, không thể cập nhật hiển thị 3D")
            return

        try:
            # Lấy dữ liệu phân tích cho kịch bản
            scenario_data = self.robustness_result.get_scenario_data(scenario_name)

            if not scenario_data:
                return

            # Lấy dữ liệu hiển thị phù hợp với chế độ
            if self.display_mode == "difference":
                display_data = scenario_data.get("difference")
            elif self.display_mode == "gamma":
                display_data = scenario_data.get("gamma")
            elif self.display_mode == "std_dev":
                display_data = scenario_data.get("std_dev")
            elif self.display_mode == "min_max":
                display_data = scenario_data.get("range")

            # Nếu không có dữ liệu hiển thị, thoát
            if display_data is None:
                return

            # Xóa tất cả các actors hiện tại (trừ axes)
            actors = self.vtk_renderer.GetActors()
            actors.InitTraversal()
            actor = actors.GetNextActor()
            while actor:
                if not isinstance(actor, vtk.vtkAxesActor):
                    self.vtk_renderer.RemoveActor(actor)
                actor = actors.GetNextActor()

            # Tạo nguồn dữ liệu VTK từ numpy array
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(
                display_data.shape[2], display_data.shape[1], display_data.shape[0]
            )
            vtk_data.AllocateScalars(vtk.VTK_FLOAT, 1)

            # Copy dữ liệu từ numpy sang vtk
            for z in range(display_data.shape[0]):
                for y in range(display_data.shape[1]):
                    for x in range(display_data.shape[2]):
                        vtk_data.SetScalarComponentFromFloat(
                            x, y, z, 0, display_data[z, y, x]
                        )

            # Tạo isosurface với marching cubes
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(vtk_data)

            # Tính phạm vi giá trị để tạo mức isosurface
            vmin, vmax = np.nanmin(display_data), np.nanmax(display_data)

            if self.display_mode == "gamma":
                contour.SetValue(0, 1.0)  # Gamma = 1.0 là ngưỡng quan trọng
            else:
                # Chia range thành 3 phần, lấy giá trị ở 1/3 và 2/3
                contour.SetValue(0, vmin + (vmax - vmin) / 3)
                contour.SetValue(1, vmin + 2 * (vmax - vmin) / 3)

            # Tạo mapper và actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            mapper.ScalarVisibilityOn()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Tô màu cho actor
            if self.display_mode == "difference":
                actor.GetProperty().SetColor(1, 0, 0)  # Đỏ
            elif self.display_mode == "gamma":
                actor.GetProperty().SetColor(0, 1, 0)  # Xanh lá
            elif self.display_mode == "std_dev":
                actor.GetProperty().SetColor(1, 0.5, 0)  # Cam
            elif self.display_mode == "min_max":
                actor.GetProperty().SetColor(0, 0, 1)  # Xanh dương

            actor.GetProperty().SetOpacity(0.7)

            # Thêm actor vào renderer
            self.vtk_renderer.AddActor(actor)

            # Thêm cấu trúc nếu được chọn
            if self.show_structures_check.isChecked() and self.structure_masks:
                for name, mask in self.structure_masks.items():
                    # Tạo isosurface cho cấu trúc
                    vtk_structure_data = vtk.vtkImageData()
                    vtk_structure_data.SetDimensions(
                        mask.shape[2], mask.shape[1], mask.shape[0]
                    )
                    vtk_structure_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

                    # Copy dữ liệu từ numpy sang vtk
                    for z in range(mask.shape[0]):
                        for y in range(mask.shape[1]):
                            for x in range(mask.shape[2]):
                                value = 1 if mask[z, y, x] > 0.5 else 0
                                vtk_structure_data.SetScalarComponentFromFloat(
                                    x, y, z, 0, value
                                )

                    # Tạo isosurface với marching cubes
                    structure_contour = vtk.vtkMarchingCubes()
                    structure_contour.SetInputData(vtk_structure_data)
                    structure_contour.SetValue(0, 0.5)

                    # Tạo mapper và actor
                    structure_mapper = vtk.vtkPolyDataMapper()
                    structure_mapper.SetInputConnection(
                        structure_contour.GetOutputPort()
                    )

                    structure_actor = vtk.vtkActor()
                    structure_actor.SetMapper(structure_mapper)

                    # Tô màu cho actor dựa vào loại cấu trúc
                    if name.lower().startswith(("ptv", "ctv", "gtv")):
                        structure_actor.GetProperty().SetColor(
                            1, 0, 0
                        )  # Đỏ cho targets
                    else:
                        structure_actor.GetProperty().SetColor(
                            0, 1, 0
                        )  # Xanh lá cho OARs

                    structure_actor.GetProperty().SetOpacity(0.3)

                    # Thêm actor vào renderer
                    self.vtk_renderer.AddActor(structure_actor)

            # Cập nhật renderer
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            # Bắt đầu interactor nếu cần
            if not self.vtk_interactor.GetInitialized():
                self.vtk_interactor.Initialize()
                self.vtk_interactor.Start()

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật hiển thị 3D: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

        # Phát tín hiệu cập nhật
        self.viewUpdated.emit()

    def _save_current_view(self):
        """Lưu view hiện tại thành file ảnh."""
        # Hỏi file lưu
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu ảnh hiển thị",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;TIFF Files (*.tiff)",
        )

        if not filename:
            return

        try:
            if self.current_view == "3d" and HAS_VTK:
                # Lưu ảnh từ VTK renderer
                renderWindow = self.vtk_widget.GetRenderWindow()

                # Tạo bộ lọc lưu ảnh
                windowToImage = vtk.vtkWindowToImageFilter()
                windowToImage.SetInput(renderWindow)
                windowToImage.Update()

                # Lưu ảnh
                writer = vtk.vtkPNGWriter()
                if filename.endswith(".jpg"):
                    writer = vtk.vtkJPEGWriter()
                elif filename.endswith(".tiff"):
                    writer = vtk.vtkTIFFWriter()

                writer.SetFileName(filename)
                writer.SetInputConnection(windowToImage.GetOutputPort())
                writer.Write()
            else:
                # Lưu ảnh từ matplotlib
                self.canvas.figure.savefig(filename, dpi=300, bbox_inches="tight")

            QMessageBox.information(
                self, "Lưu ảnh thành công", f"Ảnh đã được lưu tại {filename}."
            )

        except Exception as e:
            logger.error(f"Lỗi khi lưu ảnh: {str(e)}")
            QMessageBox.critical(self, "Lỗi lưu ảnh", f"Không thể lưu ảnh: {str(e)}")

    def _reset_view(self):
        """Reset view về trạng thái mặc định."""
        # Reset view 2D
        if self.current_view != "3d":
            self.current_slice = self.slice_slider.maximum() // 2
            self.slice_slider.setValue(self.current_slice)
            self._update_display()

        # Reset view 3D
        elif HAS_VTK:
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
