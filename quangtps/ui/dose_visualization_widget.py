#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module hiển thị phân bố liều 3D nâng cao cho QuangTPS.

Module này cung cấp widget hiển thị phân bố liều 3D với các tính năng tương tự Eclipse của Varian:
- Hiển thị iso-dose surfaces với các mức liều có thể tùy chỉnh
- Chế độ volume rendering với độ trong suốt và chuyển đổi transfer function
- Hiển thị cấu trúc 3D với các tùy chọn về độ trong suốt và màu sắc
- Hiển thị chùm tia và góc chiếu
- Tích hợp với DVH widget để hiển thị liều tương ứng với các cấu trúc đã chọn
- Các góc nhìn tiêu chuẩn (anterior, posterior, left, right, superior, inferior)
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

# Khởi tạo logger
logger = logging.getLogger(__name__)

# Import PyQt5 với try-except
try:
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QComboBox,
        QCheckBox,
        QSlider,
        QToolBar,
        QAction,
        QSpinBox,
        QDoubleSpinBox,
        QColorDialog,
        QSplitter,
        QTabWidget,
        QGroupBox,
        QFrame,
        QToolButton,
        QMenu,
        QSizePolicy,
    )
    from PyQt5.QtGui import QColor, QIcon

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    logger.warning(
        "PyQt5 không khả dụng. Widget hiển thị liều 3D sẽ hoạt động ở chế độ hạn chế."
    )

    # Tạo lớp giả
    class QWidget:
        pass

    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass


# Thử import VTK
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    logger.warning(
        "VTK không khả dụng. Sẽ sử dụng các phương pháp khác cho hiển thị 3D."
    )

# Thử import các module visualizaton thay thế
try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib không khả dụng. Khả năng hiển thị 3D sẽ bị hạn chế.")

# Import từ quangtps
try:
    from quangtps.ui.eclipse_style_theme import (
        get_eclipse_colormap,
        create_eclipse_widget_style,
    )
    from quangtps.ui import get_colormap_for_display

    HAS_QUANGTPS_UI = True
except ImportError:
    HAS_QUANGTPS_UI = False
    logger.warning("Không thể import module UI của QuangTPS.")


class DoseVisualizationWidget(QWidget):
    """
    Widget hiển thị phân bố liều 3D nâng cao với các tính năng tương tự Eclipse.

    Tín hiệu:
    ---------
    structure_selected : pyqtSignal(str)
        Phát khi người dùng chọn một cấu trúc trong cảnh 3D
    isodose_level_selected : pyqtSignal(float)
        Phát khi người dùng chọn một mức liều
    view_changed : pyqtSignal(str)
        Phát khi góc nhìn thay đổi
    """

    # Tín hiệu
    structure_selected = pyqtSignal(str)  # Phát khi chọn cấu trúc
    isodose_level_selected = pyqtSignal(float)  # Phát khi chọn mức liều
    view_changed = pyqtSignal(str)  # Phát khi thay đổi góc nhìn

    def __init__(self, parent=None):
        """
        Khởi tạo widget hiển thị liều 3D.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        if not HAS_PYQT:
            logger.error(
                "PyQt5 không khả dụng. Không thể khởi tạo DoseVisualizationWidget."
            )
            return

        super().__init__(parent)

        # Khởi tạo dữ liệu
        self.dose_grid = None
        self.dose_spacing = None
        self.dose_origin = None
        self.structures = {}
        self.isodose_levels = [
            107,
            105,
            100,
            98,
            95,
            90,
            80,
            70,
            60,
            50,
            40,
            30,
            20,
            10,
        ]
        self.isodose_colors = {}
        self.prescription_dose = 70.0  # Gy
        self.selected_structure = None
        self.display_mode = "surface"  # "surface", "volume", "mip" hoặc "xray"
        self.current_view = "anterior"

        # Các giá trị hiển thị
        self.show_structures = True
        self.show_isodoses = True
        self.show_beams = True
        self.structure_opacity = 0.5
        self.dose_opacity = 0.7

        # Khởi tạo giao diện
        self._setup_ui()

        # Khởi tạo màu cho các mức liều
        self._initialize_isodose_colors()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        if not HAS_PYQT:
            return

        # Layout chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        main_layout.addWidget(toolbar)

        # Chế độ hiển thị
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(
            ["Surface", "Volume Rendering", "MIP", "X-Ray"]
        )
        self.display_mode_combo.setCurrentIndex(0)
        self.display_mode_combo.setToolTip("Chọn chế độ hiển thị 3D")
        toolbar.addWidget(QLabel("Chế độ:"))
        toolbar.addWidget(self.display_mode_combo)

        toolbar.addSeparator()

        # Góc nhìn
        view_button = QToolButton()
        view_button.setText("Góc nhìn")
        view_button.setPopupMode(QToolButton.InstantPopup)
        view_menu = QMenu()

        view_actions = {}
        for view_name in [
            "Anterior",
            "Posterior",
            "Left",
            "Right",
            "Superior",
            "Inferior",
            "Isometric",
        ]:
            action = QAction(view_name, self)
            view_menu.addAction(action)
            view_actions[view_name.lower()] = action

        view_button.setMenu(view_menu)
        toolbar.addWidget(view_button)

        toolbar.addSeparator()

        # Checkbox hiển thị
        self.show_structures_cb = QCheckBox("Cấu trúc")
        self.show_structures_cb.setChecked(True)
        toolbar.addWidget(self.show_structures_cb)

        self.show_isodoses_cb = QCheckBox("Isodose")
        self.show_isodoses_cb.setChecked(True)
        toolbar.addWidget(self.show_isodoses_cb)

        self.show_beams_cb = QCheckBox("Chùm tia")
        self.show_beams_cb.setChecked(True)
        toolbar.addWidget(self.show_beams_cb)

        toolbar.addSeparator()

        # Chỉnh độ trong suốt
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Độ trong suốt:"))

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.dose_opacity * 100))
        self.opacity_slider.setFixedWidth(100)
        opacity_layout.addWidget(self.opacity_slider)

        opacity_widget = QWidget()
        opacity_widget.setLayout(opacity_layout)
        toolbar.addWidget(opacity_widget)

        # Widget chính cho hiển thị 3D
        self.visualization_widget = None
        if HAS_VTK:
            # Sử dụng VTK nếu có (ưu tiên nhất)
            self.visualization_widget = QVTKRenderWindowInteractor(self)
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.1, 0.1)  # Màu nền tối
            self.visualization_widget.GetRenderWindow().AddRenderer(self.renderer)
            self.interactor = (
                self.visualization_widget.GetRenderWindow().GetInteractor()
            )

            # Thiết lập interactor style
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.interactor.SetInteractorStyle(style)

        elif HAS_MATPLOTLIB:
            # Fallback sang matplotlib nếu không có VTK
            figure = Figure(figsize=(5, 5), dpi=100)
            self.axes = figure.add_subplot(111, projection="3d")
            self.visualization_widget = FigureCanvasQTAgg(figure)

        else:
            # Fallback khi không có thư viện visualization nào
            self.visualization_widget = QLabel(
                "Không có thư viện visualization\n(VTK hoặc matplotlib) khả dụng."
            )
            self.visualization_widget.setAlignment(Qt.AlignCenter)
            self.visualization_widget.setStyleSheet(
                "background-color: #f0f0f0; border: 1px solid #ccc;"
            )

        main_layout.addWidget(self.visualization_widget, 1)  # Stretch factor 1

        # Điều khiển isodose
        isodose_group = QGroupBox("Mức Isodose")
        isodose_layout = QVBoxLayout(isodose_group)

        # Thiết lập checkbox và slider cho mỗi mức isodose
        self.isodose_widgets = {}
        for level in self.isodose_levels:
            level_layout = QHBoxLayout()

            # Checkbox
            checkbox = QCheckBox(f"{level}%")
            checkbox.setChecked(True)
            level_layout.addWidget(checkbox)

            # Color button (sẽ được thiết lập màu trong _initialize_isodose_colors())
            color_button = QPushButton()
            color_button.setFixedSize(16, 16)
            color_button.setStyleSheet("background-color: #FF0000;")
            color_button.clicked.connect(
                lambda _, l=level: self._select_isodose_color(l)
            )
            level_layout.addWidget(color_button)

            level_layout.addStretch(1)
            isodose_layout.addLayout(level_layout)

            self.isodose_widgets[level] = {
                "checkbox": checkbox,
                "color_button": color_button,
            }

        isodose_layout.addStretch(1)
        main_layout.addWidget(isodose_group)

        # Kết nối signals
        if HAS_PYQT:
            self.display_mode_combo.currentIndexChanged.connect(
                self._on_display_mode_changed
            )
            self.show_structures_cb.toggled.connect(self._on_show_structures_toggled)
            self.show_isodoses_cb.toggled.connect(self._on_show_isodoses_toggled)
            self.show_beams_cb.toggled.connect(self._on_show_beams_toggled)
            self.opacity_slider.valueChanged.connect(
                lambda v: self._set_dose_opacity(v / 100.0)
            )

            for view_name, action in view_actions.items():
                action.triggered.connect(lambda _, v=view_name: self._set_view(v))

        # Thiết lập style Eclipse nếu có thể
        if HAS_QUANGTPS_UI:
            try:
                self.setStyleSheet(create_eclipse_widget_style("visualization"))
            except:
                pass

    def _initialize_isodose_colors(self):
        """Thiết lập màu mặc định cho các mức isodose."""
        # Sử dụng hệ màu tương tự Eclipse
        self.isodose_colors = {
            107: QColor(220, 0, 0),  # Đỏ tươi (107%)
            105: QColor(255, 0, 0),  # Đỏ (105%)
            100: QColor(255, 100, 100),  # Đỏ nhạt (100%)
            98: QColor(255, 150, 150),  # Hồng (98%)
            95: QColor(255, 200, 0),  # Vàng (95%)
            90: QColor(255, 255, 0),  # Vàng nhạt (90%)
            80: QColor(0, 220, 0),  # Xanh lá (80%)
            70: QColor(0, 255, 127),  # Xanh lá nhạt (70%)
            60: QColor(0, 255, 255),  # Xanh dương nhạt (60%)
            50: QColor(0, 127, 255),  # Xanh dương (50%)
            40: QColor(0, 0, 255),  # Xanh dương đậm (40%)
            30: QColor(75, 0, 130),  # Chàm (30%)
            20: QColor(148, 0, 211),  # Tím (20%)
            10: QColor(200, 200, 200),  # Xám (10%)
        }

        # Cập nhật giao diện nếu có
        if hasattr(self, "isodose_widgets"):
            for level, widgets in self.isodose_widgets.items():
                if level in self.isodose_colors:
                    color = self.isodose_colors[level]
                    widgets["color_button"].setStyleSheet(
                        f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
                    )

    def _select_isodose_color(self, level):
        """
        Hiển thị color dialog để chọn màu cho mức isodose.

        Parameters
        ----------
        level : float
            Mức isodose cần thay đổi màu (%)
        """
        if not HAS_PYQT:
            return

        current_color = self.isodose_colors.get(level, QColor(255, 0, 0))
        color = QColorDialog.getColor(
            current_color, self, f"Chọn màu cho mức isodose {level}%"
        )

        if color.isValid():
            self.isodose_colors[level] = color
            self.isodose_widgets[level]["color_button"].setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            )
            self._update_dose_display()

    def _on_display_mode_changed(self, index):
        """Xử lý khi chế độ hiển thị thay đổi."""
        mode_map = ["surface", "volume", "mip", "xray"]
        if index < len(mode_map):
            self.display_mode = mode_map[index]
            self._update_visualization()

    def _on_show_structures_toggled(self, checked):
        """Xử lý khi toggle hiển thị cấu trúc."""
        self.show_structures = checked
        self._update_structures_display()

    def _on_show_isodoses_toggled(self, checked):
        """Xử lý khi toggle hiển thị isodose."""
        self.show_isodoses = checked
        self._update_dose_display()

    def _on_show_beams_toggled(self, checked):
        """Xử lý khi toggle hiển thị chùm tia."""
        self.show_beams = checked
        self._update_beams_display()

    def _set_dose_opacity(self, opacity):
        """
        Thiết lập độ trong suốt cho phân bố liều.

        Parameters
        ----------
        opacity : float
            Độ trong suốt từ 0.0 (hoàn toàn trong suốt) đến 1.0 (hoàn toàn đục)
        """
        self.dose_opacity = opacity
        self._update_dose_display()

    def _set_view(self, view_name):
        """
        Thiết lập góc nhìn.

        Parameters
        ----------
        view_name : str
            Tên góc nhìn: anterior, posterior, left, right, superior, inferior, isometric
        """
        self.current_view = view_name

        if HAS_VTK and hasattr(self, "renderer"):
            # Reset camera vị trí
            self.renderer.ResetCamera()
            camera = self.renderer.GetActiveCamera()

            # Thiết lập góc nhìn
            if view_name == "anterior":
                camera.SetPosition(0, -1000, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "posterior":
                camera.SetPosition(0, 1000, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "left":
                camera.SetPosition(-1000, 0, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "right":
                camera.SetPosition(1000, 0, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "superior":
                camera.SetPosition(0, 0, 1000)
                camera.SetViewUp(0, 1, 0)
            elif view_name == "inferior":
                camera.SetPosition(0, 0, -1000)
                camera.SetViewUp(0, 1, 0)
            elif view_name == "isometric":
                camera.SetPosition(500, -500, 500)
                camera.SetViewUp(0, 0, 1)

            # Render lại scene
            self.renderer.ResetCameraClippingRange()
            self.visualization_widget.GetRenderWindow().Render()

        elif HAS_MATPLOTLIB and hasattr(self, "axes"):
            # Thiết lập góc nhìn cho matplotlib
            views = {
                "anterior": (0, -90),  # (elev, azim)
                "posterior": (0, 90),
                "left": (0, 0),
                "right": (0, 180),
                "superior": (90, 0),
                "inferior": (-90, 0),
                "isometric": (30, -45),
            }

            if view_name in views:
                elev, azim = views[view_name]
                self.axes.view_init(elev=elev, azim=azim)
                self.visualization_widget.draw()

        # Phát tín hiệu
        self.view_changed.emit(view_name)

    def set_dose_grid(self, dose_grid, spacing=None, origin=None):
        """
        Thiết lập dữ liệu phân bố liều.

        Parameters
        ----------
        dose_grid : np.ndarray
            Mảng 3D chứa dữ liệu phân bố liều
        spacing : tuple, optional
            Khoảng cách voxel (dx, dy, dz) (mm)
        origin : tuple, optional
            Tọa độ gốc (x0, y0, z0) (mm)
        """
        self.dose_grid = dose_grid
        self.dose_spacing = spacing or (1.0, 1.0, 1.0)
        self.dose_origin = origin or (0.0, 0.0, 0.0)

        # Cập nhật hiển thị
        self._update_visualization()

    def set_structures(self, structures):
        """
        Thiết lập dữ liệu cấu trúc.

        Parameters
        ----------
        structures : Dict[str, Any]
            Dict với khóa là ID cấu trúc và giá trị là đối tượng Structure
        """
        self.structures = structures
        self._update_structures_display()

    def set_prescription_dose(self, dose):
        """
        Thiết lập liều kê toa (để chuẩn hóa các mức isodose).

        Parameters
        ----------
        dose : float
            Liều kê toa (Gy)
        """
        self.prescription_dose = dose
        self._update_dose_display()

    def _update_visualization(self):
        """Cập nhật toàn bộ hiển thị 3D."""
        # Cập nhật cả 3 thành phần
        self._update_dose_display()
        self._update_structures_display()
        self._update_beams_display()

    def _update_dose_display(self):
        """Cập nhật hiển thị phân bố liều."""
        if not self.dose_grid is not None or not self.show_isodoses:
            return

        if HAS_VTK and hasattr(self, "renderer"):
            # Xóa các actor isodose cũ
            for actor in self.renderer.GetActors():
                if hasattr(actor, "isodose_level"):
                    self.renderer.RemoveActor(actor)

            # Tạo isodose surfaces mới
            for level in self.isodose_levels:
                # Kiểm tra checkbox
                if hasattr(self, "isodose_widgets") and level in self.isodose_widgets:
                    checkbox = self.isodose_widgets[level]["checkbox"]
                    if not checkbox.isChecked():
                        continue

                # Convert từ phần trăm sang giá trị liều
                dose_value = level / 100.0 * self.prescription_dose

                # Tạo isosurface
                try:
                    # Tạo vtkImageData từ numpy array
                    image_data = vtk.vtkImageData()
                    image_data.SetDimensions(self.dose_grid.shape)
                    image_data.SetSpacing(self.dose_spacing)
                    image_data.SetOrigin(self.dose_origin)

                    # Chuyển đổi numpy array sang vtkDoubleArray
                    flat_data = self.dose_grid.flatten("F").astype("float32")
                    vtk_data = vtk.vtkFloatArray()
                    vtk_data.SetNumberOfComponents(1)
                    vtk_data.SetNumberOfValues(flat_data.size)
                    for i, val in enumerate(flat_data):
                        vtk_data.SetValue(i, val)

                    image_data.GetPointData().SetScalars(vtk_data)

                    # Tạo isosurface
                    contour = vtk.vtkMarchingCubes()
                    contour.SetInputData(image_data)
                    contour.SetValue(0, dose_value)
                    contour.Update()

                    # Smoothing
                    smoother = vtk.vtkWindowedSincPolyDataFilter()
                    smoother.SetInputData(contour.GetOutput())
                    smoother.SetNumberOfIterations(10)
                    smoother.SetPassBand(0.1)
                    smoother.NonManifoldSmoothingOn()
                    smoother.NormalizeCoordinatesOn()
                    smoother.Update()

                    # Mapper và Actor
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(smoother.GetOutput())

                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)

                    # Thiết lập màu
                    color = self.isodose_colors.get(level, QColor(255, 0, 0))
                    actor.GetProperty().SetColor(
                        color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0
                    )
                    actor.GetProperty().SetOpacity(self.dose_opacity)

                    # Lưu thông tin level với actor
                    actor.isodose_level = level

                    self.renderer.AddActor(actor)
                except Exception as e:
                    logger.error(f"Lỗi khi tạo isodose {level}%: {str(e)}")

            # Render lại scene
            self.renderer.ResetCameraClippingRange()
            self.visualization_widget.GetRenderWindow().Render()

        elif HAS_MATPLOTLIB and hasattr(self, "axes"):
            # Cài đặt matplotlib hiển thị
            # Lưu ý: Hiển thị 3D với matplotlib khá hạn chế so với VTK
            self.axes.clear()

            # Tính toán các giá trị isodose grid
            x, y, z = np.indices(self.dose_grid.shape)
            x = x * self.dose_spacing[0] + self.dose_origin[0]
            y = y * self.dose_spacing[1] + self.dose_origin[1]
            z = z * self.dose_spacing[2] + self.dose_origin[2]

            # Chế độ hiển thị
            if self.display_mode == "surface":
                # Đơn giản hóa bằng cách chỉ hiển thị một vài mặt cắt
                mid_x = self.dose_grid.shape[0] // 2
                mid_y = self.dose_grid.shape[1] // 2
                mid_z = self.dose_grid.shape[2] // 2

                self.axes.contourf(
                    y[:, :, mid_z],
                    x[:, :, mid_z],
                    self.dose_grid[:, :, mid_z],
                    levels=10,
                    cmap="jet",
                    alpha=0.7,
                )
                self.axes.contourf(
                    y[:, mid_y, :],
                    z[:, mid_y, :],
                    self.dose_grid[:, mid_y, :],
                    levels=10,
                    cmap="jet",
                    alpha=0.7,
                )
                self.axes.contourf(
                    x[mid_x, :, :],
                    z[mid_x, :, :],
                    self.dose_grid[mid_x, :, :],
                    levels=10,
                    cmap="jet",
                    alpha=0.7,
                )

            elif self.display_mode == "volume":
                # Hiển thị đơn giản bằng scatter plot với độ trong suốt
                mask = self.dose_grid > (
                    self.prescription_dose * 0.1
                )  # Chỉ hiển thị vùng > 10%
                points = np.column_stack((x[mask], y[mask], z[mask]))
                values = self.dose_grid[mask]

                # Subsample để tránh quá nhiều điểm
                max_points = 5000
                if len(points) > max_points:
                    idx = np.random.choice(len(points), max_points, replace=False)
                    points = points[idx]
                    values = values[idx]

                # Normalize giá trị từ 0-1 cho màu
                norm_values = values / self.prescription_dose

                # Scatter plot
                sc = self.axes.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    c=norm_values,
                    cmap="jet",
                    alpha=0.5,
                    s=10,
                )  # s là kích thước điểm

            # Thiết lập các thuộc tính
            self.axes.set_xlabel("X (mm)")
            self.axes.set_ylabel("Y (mm)")
            self.axes.set_zlabel("Z (mm)")
            self.axes.set_title("Dose Distribution")

            # Cập nhật view angle
            self._set_view(self.current_view)

            # Render
            self.visualization_widget.draw()

    def _update_structures_display(self):
        """Cập nhật hiển thị cấu trúc."""
        if not self.structures or not self.show_structures:
            return

        if HAS_VTK and hasattr(self, "renderer"):
            # Xóa các actor cấu trúc cũ
            for actor in self.renderer.GetActors():
                if hasattr(actor, "structure_id"):
                    self.renderer.RemoveActor(actor)

            # Tạo surface mới cho mỗi cấu trúc
            for structure_id, structure in self.structures.items():
                try:
                    if hasattr(structure, "mesh_points") and hasattr(
                        structure, "mesh_faces"
                    ):
                        # Nếu có sẵn mesh, dùng mesh đó
                        points = structure.mesh_points
                        faces = structure.mesh_faces
                    elif hasattr(structure, "mask") and structure.mask is not None:
                        # Nếu có mask, tạo mesh từ mask
                        # (Sử dụng VTK vtkMarchingCubes)
                        mask = structure.mask
                        spacing = getattr(structure, "spacing", (1.0, 1.0, 1.0))
                        origin = getattr(structure, "origin", (0.0, 0.0, 0.0))

                        # Tạo vtkImageData từ numpy array
                        image_data = vtk.vtkImageData()
                        image_data.SetDimensions(mask.shape)
                        image_data.SetSpacing(spacing)
                        image_data.SetOrigin(origin)

                        # Chuyển đổi numpy array sang vtkUnsignedCharArray
                        flat_data = mask.flatten("F").astype("uint8")
                        vtk_data = vtk.vtkUnsignedCharArray()
                        vtk_data.SetNumberOfComponents(1)
                        vtk_data.SetNumberOfValues(flat_data.size)
                        for i, val in enumerate(flat_data):
                            vtk_data.SetValue(i, val)

                        image_data.GetPointData().SetScalars(vtk_data)

                        # Tạo surface từ mask
                        contour = vtk.vtkMarchingCubes()
                        contour.SetInputData(image_data)
                        contour.SetValue(0, 0.5)  # Ngưỡng
                        contour.Update()

                        # Smoothing
                        smoother = vtk.vtkWindowedSincPolyDataFilter()
                        smoother.SetInputData(contour.GetOutput())
                        smoother.SetNumberOfIterations(15)
                        smoother.SetPassBand(0.1)
                        smoother.NonManifoldSmoothingOn()
                        smoother.NormalizeCoordinatesOn()
                        smoother.Update()

                        # Mapper và Actor
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputData(smoother.GetOutput())

                    else:
                        # Không có đủ dữ liệu để tạo mesh
                        logging.warning(
                            f"Không đủ dữ liệu để hiển thị cấu trúc {structure_id}"
                        )
                        continue

                    # Tạo actor
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)

                    # Thiết lập màu
                    if hasattr(structure, "color") and structure.color:
                        r, g, b = (
                            structure.color
                            if len(structure.color) >= 3
                            else (1.0, 0.0, 0.0)
                        )
                    else:
                        # Màu mặc định dựa trên tên
                        if any(
                            target_name in structure.name
                            for target_name in ["PTV", "CTV", "GTV"]
                        ):
                            r, g, b = 1.0, 0.0, 0.0  # Đỏ
                        elif "CORD" in structure.name:
                            r, g, b = 1.0, 1.0, 0.0  # Vàng
                        elif "LUNG" in structure.name:
                            r, g, b = 0.0, 0.0, 1.0  # Xanh dương
                        else:
                            r, g, b = 0.0, 1.0, 0.0  # Xanh lá

                    actor.GetProperty().SetColor(r, g, b)
                    actor.GetProperty().SetOpacity(self.structure_opacity)

                    # Lưu thông tin structure với actor
                    actor.structure_id = structure_id

                    self.renderer.AddActor(actor)
                except Exception as e:
                    logger.error(f"Lỗi khi hiển thị cấu trúc {structure_id}: {str(e)}")

            # Render lại scene
            self.renderer.ResetCameraClippingRange()
            self.visualization_widget.GetRenderWindow().Render()

        elif HAS_MATPLOTLIB and hasattr(self, "axes"):
            # Cài đặt matplotlib hiển thị
            # Lưu ý: Hiển thị 3D phức tạp với matplotlib, ở đây chỉ hiển thị đơn giản
            # Có thể cần cải tiến thêm

            # TODO: Cải thiện hiển thị cấu trúc với matplotlib
            pass

    def _update_beams_display(self):
        """Cập nhật hiển thị chùm tia."""
        # TODO: Cài đặt hiển thị chùm tia trong 3D
        pass


def create_dose_visualization_widget(parent=None):
    """
    Tạo widget hiển thị liều mới.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha

    Returns
    -------
    DoseVisualizationWidget
        Widget hiển thị liều mới
    """
    return DoseVisualizationWidget(parent)
