#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module 3D Visualization cho QuangTPS.

Module này cung cấp các chức năng hiển thị 3D cho phân bố liều và cấu trúc,
hỗ trợ mô phỏng giao diện tương tự Eclipse TPS.
"""

import os
import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

# Thử import các thư viện cần thiết
try:
    import vtk
    from vtkmodules.vtkCommonCore import vtkCommand
    from vtkmodules.vtkRenderingCore import vtkVolume, vtkRenderer

    HAS_VTK = True
except ImportError:
    logger.warning("VTK không khả dụng. Chức năng hiển thị 3D sẽ bị hạn chế.")
    HAS_VTK = False

try:
    from PyQt5.QtCore import Qt, pyqtSignal, QObject
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QSplitter,
        QFrame,
        QToolBar,
    )
    from PyQt5.QtGui import QColor

    HAS_PYQT = True

    # Thử import VTK-Qt integration
    if HAS_VTK:
        try:
            from vtkmodules.qt.QVTKRenderWindowInteractor import (
                QVTKRenderWindowInteractor,
            )

            HAS_VTK_QT = True
        except ImportError:
            logger.warning(
                "VTK-Qt integration không khả dụng. Sử dụng tích hợp thủ công."
            )
            HAS_VTK_QT = False
    else:
        HAS_VTK_QT = False

except ImportError:
    logger.warning("PyQt5 không khả dụng. Chức năng hiển thị 3D sẽ bị hạn chế.")
    HAS_PYQT = False
    HAS_VTK_QT = False


# Enum cho các chế độ hiển thị
class DisplayMode(Enum):
    STRUCTURES = "structures"  # Chỉ hiển thị cấu trúc
    DOSE = "dose"  # Chỉ hiển thị phân bố liều
    COMBINED = "combined"  # Hiển thị cả liều và cấu trúc


class VisualizationMode(Enum):
    VOLUME_RENDERING = "volume_rendering"  # Hiển thị thể tích
    ISODOSE_SURFACES = "isodose_surfaces"  # Hiển thị đường đẳng liều
    CONTOUR_LINES = "contour_lines"  # Hiển thị đường viền


class ViewOrientation(Enum):
    AXIAL = "axial"  # Góc nhìn trục
    SAGITTAL = "sagittal"  # Góc nhìn dọc
    CORONAL = "coronal"  # Góc nhìn vành
    OBLIQUE = "oblique"  # Góc nhìn xiên
    THREE_D = "3d"  # Góc nhìn 3D tự do


class Visualization3D:
    """
    Lớp cơ sở cho hiển thị 3D trong QuangTPS.

    Cung cấp các chức năng cơ bản để hiển thị phân bố liều và cấu trúc
    dưới dạng 3D sử dụng VTK.
    """

    def __init__(self):
        """Khởi tạo đối tượng hiển thị 3D."""
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.camera = None

        # Các cấu trúc và đối tượng hiển thị
        self.structures = {}  # Dict cấu trúc: id -> vtk_actor
        self.dose_actor = None  # Actor cho phân bố liều
        self.isodose_actors = {}  # Dict cho các đường đẳng liều

        # Trạng thái hiển thị
        self.display_mode = DisplayMode.COMBINED
        self.visualization_mode = VisualizationMode.VOLUME_RENDERING
        self.view_orientation = ViewOrientation.THREE_D

        # Dữ liệu
        self.dose_array = None  # Numpy array chứa dữ liệu liều
        self.dose_spacing = None  # Khoảng cách giữa các điểm liều (mm)
        self.dose_origin = None  # Điểm gốc của phân bố liều

        # Các giá trị và màu sắc đẳng liều
        self.isodose_values = [100, 95, 90, 80, 70, 50, 30, 10]  # Phần trăm liều tối đa
        self.isodose_colors = {
            100: (1.0, 0.0, 0.0),  # Đỏ
            95: (1.0, 0.5, 0.0),  # Cam
            90: (1.0, 1.0, 0.0),  # Vàng
            80: (0.0, 1.0, 0.0),  # Xanh lá
            70: (0.0, 1.0, 1.0),  # Xanh lơ
            50: (0.0, 0.0, 1.0),  # Xanh dương
            30: (0.0, 0.0, 0.7),  # Xanh dương đậm
            10: (0.5, 0.0, 0.5),  # Tím
        }

        # Khởi tạo VTK nếu khả dụng
        if HAS_VTK:
            self._initialize_vtk()

    def _initialize_vtk(self):
        """
        Khởi tạo các thành phần VTK cần thiết.
        """
        try:
            # Tạo renderer
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.1, 0.1)  # Nền tối

            # Tạo render window
            self.render_window = vtk.vtkRenderWindow()
            self.render_window.AddRenderer(self.renderer)

            # Tạo interactor
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.render_window)

            # Thiết lập kiểu tương tác
            interactor_style = vtk.vtkInteractorStyleTrackballCamera()
            self.interactor.SetInteractorStyle(interactor_style)

            # Lấy camera
            self.camera = self.renderer.GetActiveCamera()
            self.camera.SetViewUp(0, 0, 1)
            self.camera.SetPosition(0, -1, 0)
            self.camera.SetFocalPoint(0, 0, 0)

            logger.info("Đã khởi tạo thành công các thành phần VTK.")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo VTK: {str(e)}")

    def set_dose_data(
        self,
        dose_array: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ):
        """
        Thiết lập dữ liệu phân bố liều.

        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        spacing : Tuple[float, float, float]
            Khoảng cách giữa các điểm liều (mm)
        origin : Tuple[float, float, float]
            Điểm gốc của phân bố liều
        """
        if not HAS_VTK:
            logger.warning("VTK không khả dụng. Không thể hiển thị dữ liệu liều.")
            return

        try:
            self.dose_array = dose_array
            self.dose_spacing = spacing
            self.dose_origin = origin

            # Xóa các hiển thị liều cũ
            if self.dose_actor:
                self.renderer.RemoveActor(self.dose_actor)

            for actor in self.isodose_actors.values():
                self.renderer.RemoveActor(actor)

            self.isodose_actors = {}

            # Tạo lại hiển thị liều dựa trên chế độ hiển thị
            if self.visualization_mode == VisualizationMode.VOLUME_RENDERING:
                self._create_volume_rendering()
            elif self.visualization_mode == VisualizationMode.ISODOSE_SURFACES:
                self._create_isodose_surfaces()

            # Render lại scene
            if self.interactor:
                self.interactor.Render()

            logger.info("Đã thiết lập thành công dữ liệu phân bố liều.")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập dữ liệu liều: {str(e)}")

    def add_structure(
        self,
        structure_id: str,
        contour_points: List[List[List[float]]],
        color: Tuple[float, float, float] = None,
    ):
        """
        Thêm một cấu trúc để hiển thị.

        Parameters
        ----------
        structure_id : str
            ID duy nhất của cấu trúc
        contour_points : List[List[List[float]]]
            Danh sách các contour, mỗi contour là một danh sách các điểm 3D
        color : Tuple[float, float, float], optional
            Màu RGB của cấu trúc, mặc định là None (sẽ dùng màu ngẫu nhiên)
        """
        if not HAS_VTK:
            logger.warning("VTK không khả dụng. Không thể hiển thị cấu trúc.")
            return

        try:
            # Xóa cấu trúc cũ nếu có
            if structure_id in self.structures:
                self.renderer.RemoveActor(self.structures[structure_id])

            # Tạo màu nếu không được cung cấp
            if color is None:
                import random

                color = (random.random(), random.random(), random.random())

            # Tạo các đối tượng VTK cần thiết
            points = vtk.vtkPoints()
            polys = vtk.vtkCellArray()

            # Xử lý từng contour
            for contour in contour_points:
                # Bắt đầu đa giác mới
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(len(contour))

                # Thêm các điểm vào contour
                point_offset = points.GetNumberOfPoints()
                for i, point in enumerate(contour):
                    points.InsertNextPoint(point)
                    polygon.GetPointIds().SetId(i, point_offset + i)

                # Thêm polygon vào danh sách
                polys.InsertNextCell(polygon)

            # Tạo polydata
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)

            # Tạo lưới tam giác
            triangulator = vtk.vtkTriangleFilter()
            triangulator.SetInputData(polydata)
            triangulator.Update()

            # Tạo bề mặt
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(triangulator.GetOutput())
            normals.SetFeatureAngle(60.0)
            normals.Update()

            # Tạo mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(normals.GetOutput())

            # Tạo actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(0.7)

            # Thêm vào renderer
            self.renderer.AddActor(actor)
            self.structures[structure_id] = actor

            # Render lại scene
            if self.interactor:
                self.interactor.Render()

            logger.info(f"Đã thêm thành công cấu trúc {structure_id}.")
        except Exception as e:
            logger.error(f"Lỗi khi thêm cấu trúc {structure_id}: {str(e)}")

    def remove_structure(self, structure_id: str):
        """
        Xóa một cấu trúc khỏi hiển thị.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc cần xóa
        """
        if not HAS_VTK:
            return

        if structure_id in self.structures:
            self.renderer.RemoveActor(self.structures[structure_id])
            del self.structures[structure_id]

            # Render lại scene
            if self.interactor:
                self.interactor.Render()

    def set_display_mode(self, mode: DisplayMode):
        """
        Thiết lập chế độ hiển thị.

        Parameters
        ----------
        mode : DisplayMode
            Chế độ hiển thị mới
        """
        self.display_mode = mode

        if not HAS_VTK:
            return

        # Cập nhật hiển thị
        if mode == DisplayMode.STRUCTURES:
            # Hiển thị chỉ cấu trúc
            if self.dose_actor:
                self.dose_actor.SetVisibility(False)
            for actor in self.isodose_actors.values():
                actor.SetVisibility(False)

            for actor in self.structures.values():
                actor.SetVisibility(True)

        elif mode == DisplayMode.DOSE:
            # Hiển thị chỉ liều
            if self.dose_actor:
                self.dose_actor.SetVisibility(True)
            for actor in self.isodose_actors.values():
                actor.SetVisibility(True)

            for actor in self.structures.values():
                actor.SetVisibility(False)

        elif mode == DisplayMode.COMBINED:
            # Hiển thị cả hai
            if self.dose_actor:
                self.dose_actor.SetVisibility(True)
            for actor in self.isodose_actors.values():
                actor.SetVisibility(True)

            for actor in self.structures.values():
                actor.SetVisibility(True)

        # Render lại scene
        if self.interactor:
            self.interactor.Render()

    def set_view_orientation(self, orientation: ViewOrientation):
        """
        Thiết lập hướng nhìn.

        Parameters
        ----------
        orientation : ViewOrientation
            Hướng nhìn mới
        """
        self.view_orientation = orientation

        if not HAS_VTK or not self.camera:
            return

        # Đặt lại camera dựa trên hướng nhìn
        if orientation == ViewOrientation.AXIAL:
            self.camera.SetViewUp(0, 1, 0)
            self.camera.SetPosition(0, 0, 1)
            self.camera.SetFocalPoint(0, 0, 0)

        elif orientation == ViewOrientation.SAGITTAL:
            self.camera.SetViewUp(0, 0, 1)
            self.camera.SetPosition(1, 0, 0)
            self.camera.SetFocalPoint(0, 0, 0)

        elif orientation == ViewOrientation.CORONAL:
            self.camera.SetViewUp(0, 0, 1)
            self.camera.SetPosition(0, 1, 0)
            self.camera.SetFocalPoint(0, 0, 0)

        elif orientation == ViewOrientation.THREE_D:
            self.camera.SetViewUp(0, 0, 1)
            self.camera.SetPosition(1, 1, 1)
            self.camera.SetFocalPoint(0, 0, 0)

        # Reset camera để hiển thị toàn bộ scene
        self.renderer.ResetCamera()

        # Render lại scene
        if self.interactor:
            self.interactor.Render()

    def _create_volume_rendering(self):
        """Tạo hiển thị phân bố liều dưới dạng volume rendering."""
        if not HAS_VTK or self.dose_array is None:
            return

        try:
            # Chuyển đổi từ numpy array sang VTK
            vtk_data = vtk.vtkDoubleArray()
            vtk_data.SetNumberOfComponents(1)

            # Chuẩn hóa dữ liệu về khoảng [0, 1]
            dose_min = np.min(self.dose_array)
            dose_max = np.max(self.dose_array)
            normalized_array = (
                (self.dose_array - dose_min) / (dose_max - dose_min)
                if dose_max > dose_min
                else np.zeros_like(self.dose_array)
            )

            # Đóng gói dữ liệu
            for value in normalized_array.flatten():
                vtk_data.InsertNextValue(value)

            # Tạo image data
            shape = self.dose_array.shape
            image_data = vtk.vtkImageData()
            image_data.SetDimensions(shape[0], shape[1], shape[2])
            image_data.SetSpacing(self.dose_spacing)
            image_data.SetOrigin(self.dose_origin)
            image_data.GetPointData().SetScalars(vtk_data)

            # Tạo hàm truyền (transfer function)
            color_function = vtk.vtkColorTransferFunction()
            opacity_function = vtk.vtkPiecewiseFunction()

            # Tạo color mapping dựa trên isodose_colors
            for dose_percent, color in self.isodose_colors.items():
                normalized_value = dose_percent / 100
                color_function.AddRGBPoint(
                    normalized_value, color[0], color[1], color[2]
                )
                opacity_function.AddPoint(
                    normalized_value, normalized_value * 0.8
                )  # Độ mờ tỷ lệ với giá trị

            # Tạo các giá trị transparent cho phần liều thấp
            opacity_function.AddPoint(0, 0.0)  # Trong suốt ở giá trị 0

            # Tạo volume property
            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetColor(color_function)
            volume_property.SetScalarOpacity(opacity_function)
            volume_property.SetInterpolationTypeToLinear()
            volume_property.ShadeOn()

            # Tạo mapper
            if hasattr(vtk, "vtkGPUVolumeRayCastMapper"):
                # Sử dụng GPU nếu có thể
                mapper = vtk.vtkGPUVolumeRayCastMapper()
            else:
                mapper = vtk.vtkFixedPointVolumeRayCastMapper()

            mapper.SetInputData(image_data)

            # Tạo volume
            volume = vtk.vtkVolume()
            volume.SetMapper(mapper)
            volume.SetProperty(volume_property)

            # Thêm vào renderer
            self.renderer.AddVolume(volume)
            self.dose_actor = volume

            logger.info("Đã tạo thành công hiển thị volume rendering.")
        except Exception as e:
            logger.error(f"Lỗi khi tạo volume rendering: {str(e)}")

    def _create_isodose_surfaces(self):
        """Tạo hiển thị phân bố liều dưới dạng đường đẳng liều."""
        if not HAS_VTK or self.dose_array is None:
            return

        try:
            # Chuyển đổi từ numpy array sang VTK
            vtk_data = vtk.vtkDoubleArray()
            vtk_data.SetNumberOfComponents(1)

            # Lấy giá trị liều tối đa
            dose_max = np.max(self.dose_array)

            # Đóng gói dữ liệu
            for value in self.dose_array.flatten():
                vtk_data.InsertNextValue(value)

            # Tạo image data
            shape = self.dose_array.shape
            image_data = vtk.vtkImageData()
            image_data.SetDimensions(shape[0], shape[1], shape[2])
            image_data.SetSpacing(self.dose_spacing)
            image_data.SetOrigin(self.dose_origin)
            image_data.GetPointData().SetScalars(vtk_data)

            # Tạo các bề mặt đẳng liều
            for isodose_value in self.isodose_values:
                # Tính giá trị đẳng liều thực tế
                actual_value = (isodose_value / 100) * dose_max

                # Tạo đường đẳng liều
                contour = vtk.vtkMarchingCubes()
                contour.SetInputData(image_data)
                contour.SetValue(0, actual_value)
                contour.Update()

                # Tạo mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(contour.GetOutput())

                # Tạo actor
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                # Thiết lập màu và độ mờ
                color = self.isodose_colors.get(isodose_value, (0.5, 0.5, 0.5))
                actor.GetProperty().SetColor(color)
                actor.GetProperty().SetOpacity(0.4)

                # Thêm vào renderer
                self.renderer.AddActor(actor)
                self.isodose_actors[isodose_value] = actor

            logger.info("Đã tạo thành công các bề mặt đẳng liều.")
        except Exception as e:
            logger.error(f"Lỗi khi tạo bề mặt đẳng liều: {str(e)}")


class Visualization3DWidget(QWidget):
    """
    Widget hiển thị 3D cho QuangTPS.

    Widget tích hợp hiển thị 3D vào giao diện PyQt.
    """

    # Tín hiệu
    view_changed = pyqtSignal(ViewOrientation)
    display_mode_changed = pyqtSignal(DisplayMode)

    def __init__(self, parent=None):
        """
        Khởi tạo widget hiển thị 3D.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, mặc định là None
        """
        if not HAS_PYQT:
            logger.error("PyQt5 không khả dụng. Không thể tạo widget 3D.")
            return

        super().__init__(parent)

        # Visualization3D instance
        self.vis3d = Visualization3D()

        # Thiết lập layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tạo toolbar
        self.toolbar = QToolBar()
        layout.addWidget(self.toolbar)

        # Tạo main content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Tạo frame cho renderer VTK
        self.vtk_frame = QFrame()
        self.vtk_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        content_layout.addWidget(self.vtk_frame)

        # Thêm nội dung vào layout chính
        layout.addWidget(content_widget)

        # Tạo VTK widget nếu có thể
        if HAS_VTK_QT:
            vtk_layout = QVBoxLayout(self.vtk_frame)
            vtk_layout.setContentsMargins(0, 0, 0, 0)

            self.vtk_widget = QVTKRenderWindowInteractor(self.vtk_frame)
            vtk_layout.addWidget(self.vtk_widget)

            # Kết nối vtk_widget với visualization
            self.vtk_widget.GetRenderWindow().AddRenderer(self.vis3d.renderer)
            self.vis3d.render_window = self.vtk_widget.GetRenderWindow()
            self.vis3d.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

            # Thiết lập interactor style
            interactor_style = vtk.vtkInteractorStyleTrackballCamera()
            self.vis3d.interactor.SetInteractorStyle(interactor_style)

            # Khởi tạo interactor
            self.vis3d.interactor.Initialize()
        else:
            # Tạo widget placeholder nếu không có VTK-Qt
            from PyQt5.QtWidgets import QLabel

            placeholder = QLabel(
                "VTK-Qt không khả dụng.\nKhông thể hiển thị 3D.", self.vtk_frame
            )
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: red; font-size: 14px;")

            vtk_layout = QVBoxLayout(self.vtk_frame)
            vtk_layout.addWidget(placeholder)

        # Thiết lập các action cho toolbar
        self._create_toolbar_actions()

    def _create_toolbar_actions(self):
        """Tạo các action cho toolbar."""
        if not hasattr(self, "toolbar"):
            return

        # Các action cho hướng nhìn
        try:
            from PyQt5.QtWidgets import QAction
            from PyQt5.QtGui import QIcon

            # Action cho góc nhìn Axial
            axial_action = QAction("Axial", self)
            axial_action.triggered.connect(
                lambda: self._change_view(ViewOrientation.AXIAL)
            )
            self.toolbar.addAction(axial_action)

            # Action cho góc nhìn Sagittal
            sagittal_action = QAction("Sagittal", self)
            sagittal_action.triggered.connect(
                lambda: self._change_view(ViewOrientation.SAGITTAL)
            )
            self.toolbar.addAction(sagittal_action)

            # Action cho góc nhìn Coronal
            coronal_action = QAction("Coronal", self)
            coronal_action.triggered.connect(
                lambda: self._change_view(ViewOrientation.CORONAL)
            )
            self.toolbar.addAction(coronal_action)

            # Action cho góc nhìn 3D
            three_d_action = QAction("3D", self)
            three_d_action.triggered.connect(
                lambda: self._change_view(ViewOrientation.THREE_D)
            )
            self.toolbar.addAction(three_d_action)

            self.toolbar.addSeparator()

            # Action cho chế độ hiển thị
            structures_action = QAction("Structures", self)
            structures_action.triggered.connect(
                lambda: self._change_display_mode(DisplayMode.STRUCTURES)
            )
            self.toolbar.addAction(structures_action)

            dose_action = QAction("Dose", self)
            dose_action.triggered.connect(
                lambda: self._change_display_mode(DisplayMode.DOSE)
            )
            self.toolbar.addAction(dose_action)

            combined_action = QAction("Combined", self)
            combined_action.triggered.connect(
                lambda: self._change_display_mode(DisplayMode.COMBINED)
            )
            self.toolbar.addAction(combined_action)

        except Exception as e:
            logger.error(f"Lỗi khi tạo toolbar actions: {str(e)}")

    def _change_view(self, orientation: ViewOrientation):
        """
        Thay đổi hướng nhìn.

        Parameters
        ----------
        orientation : ViewOrientation
            Hướng nhìn mới
        """
        self.vis3d.set_view_orientation(orientation)
        self.view_changed.emit(orientation)

    def _change_display_mode(self, mode: DisplayMode):
        """
        Thay đổi chế độ hiển thị.

        Parameters
        ----------
        mode : DisplayMode
            Chế độ hiển thị mới
        """
        self.vis3d.set_display_mode(mode)
        self.display_mode_changed.emit(mode)

    def set_dose_data(
        self,
        dose_array: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ):
        """
        Thiết lập dữ liệu phân bố liều.

        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        spacing : Tuple[float, float, float]
            Khoảng cách giữa các điểm liều (mm)
        origin : Tuple[float, float, float]
            Điểm gốc của phân bố liều
        """
        self.vis3d.set_dose_data(dose_array, spacing, origin)

    def add_structure(
        self,
        structure_id: str,
        contour_points: List[List[List[float]]],
        color: Tuple[float, float, float] = None,
    ):
        """
        Thêm một cấu trúc để hiển thị.

        Parameters
        ----------
        structure_id : str
            ID duy nhất của cấu trúc
        contour_points : List[List[List[float]]]
            Danh sách các contour, mỗi contour là một danh sách các điểm 3D
        color : Tuple[float, float, float], optional
            Màu RGB của cấu trúc, mặc định là None (sẽ dùng màu ngẫu nhiên)
        """
        self.vis3d.add_structure(structure_id, contour_points, color)

    def remove_structure(self, structure_id: str):
        """
        Xóa một cấu trúc khỏi hiển thị.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc cần xóa
        """
        self.vis3d.remove_structure(structure_id)

    def showEvent(self, event):
        """Xử lý khi widget được hiển thị."""
        super().showEvent(event)
        if hasattr(self, "vtk_widget"):
            self.vis3d.interactor.Initialize()

    def closeEvent(self, event):
        """Xử lý khi widget được đóng."""
        super().closeEvent(event)
        if hasattr(self, "vtk_widget"):
            self.vtk_widget.Finalize()


def create_3d_visualization_widget(parent=None) -> Optional[Visualization3DWidget]:
    """
    Tạo widget hiển thị 3D nếu có thể.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha, mặc định là None

    Returns
    -------
    Optional[Visualization3DWidget]
        Widget hiển thị 3D hoặc None nếu không thể tạo
    """
    if not HAS_PYQT:
        logger.error("PyQt5 không khả dụng. Không thể tạo widget 3D.")
        return None

    try:
        widget = Visualization3DWidget(parent)
        return widget
    except Exception as e:
        logger.error(f"Lỗi khi tạo widget hiển thị 3D: {str(e)}")
        return None
