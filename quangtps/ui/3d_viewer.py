#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module 3D Viewer cho QuangTPS.

Module này cung cấp khả năng hiển thị và tương tác với dữ liệu y tế 3D,
bao gồm ảnh CT, MRI, cấu trúc và phân phối liều.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QToolBar,
        QAction,
        QSlider,
        QLabel,
        QSpinBox,
        QDoubleSpinBox,
        QCheckBox,
        QComboBox,
        QPushButton,
        QGroupBox,
        QGridLayout,
        QTabWidget,
        QSizePolicy,
        QSplitter,
        QFrame,
        QMessageBox,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QTimer
    from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    logger.warning("PyQt5 không có sẵn - 3D Viewer sẽ bị giới hạn")

try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    HAS_VTK = True
    logger.info("VTK đã được import thành công cho 3D viewer")

    # Định nghĩa hàm helper để truy cập thuộc tính vtk an toàn
    def get_vtk_attribute(attr_name, default_value=None):
        """Truy cập thuộc tính của vtk một cách an toàn"""
        return getattr(vtk, attr_name, default_value)

    # Kiểm tra version để xử lý tương thích
    vtk_version = get_vtk_attribute("vtkVersion")
    if vtk_version is not None:
        vtk_version = vtk_version.GetVTKVersion()
        logger.info(f"VTK version: {vtk_version}")
    else:
        logger.warning("Không thể xác định phiên bản VTK")

    # Chuẩn bị ánh xạ kiểu dữ liệu numpy sang VTK
    NUMPY_TO_VTK_TYPE_MAP = {
        np.uint8: get_vtk_attribute("VTK_UNSIGNED_CHAR", 3),
        np.int8: get_vtk_attribute("VTK_CHAR", 2),
        np.uint16: get_vtk_attribute("VTK_UNSIGNED_SHORT", 5),
        np.int16: get_vtk_attribute("VTK_SHORT", 4),
        np.uint32: get_vtk_attribute("VTK_UNSIGNED_INT", 7),
        np.int32: get_vtk_attribute("VTK_INT", 6),
        np.float32: get_vtk_attribute("VTK_FLOAT", 10),
        np.float64: get_vtk_attribute("VTK_DOUBLE", 11),
    }

    # Kiểm tra sẵn có của các class VTK cần thiết
    VTK_CLASSES = [
        "vtkRenderer",
        "vtkImageData",
        "vtkGPUVolumeRayCastMapper",
        "vtkVolume",
        "vtkVolumeProperty",
        "vtkColorTransferFunction",
        "vtkPiecewiseFunction",
        "vtkImageMapToColors",
        "vtkImageSliceMapper",
        "vtkImageActor",
        "vtkUnsignedCharArray",
        "vtkMarchingCubes",
        "vtkPolyDataMapper",
        "vtkActor",
        "vtkDataArray",
    ]

    VTK_CLASSES_AVAILABLE = {}
    for cls_name in VTK_CLASSES:
        VTK_CLASSES_AVAILABLE[cls_name] = hasattr(vtk, cls_name)
        if not VTK_CLASSES_AVAILABLE[cls_name]:
            logger.warning(f"VTK class {cls_name} không khả dụng")

    # Kiểm tra nếu thiếu quá nhiều class cần thiết
    if sum(VTK_CLASSES_AVAILABLE.values()) < len(VTK_CLASSES) * 0.8:
        logger.warning(
            "Nhiều class VTK cần thiết không khả dụng, chuyển sang chế độ fallback"
        )
        HAS_VTK = False

except ImportError as e:
    HAS_VTK = False
    logger.warning(f"VTK không khả dụng ({str(e)}). Sử dụng Matplotlib fallback.")
    NUMPY_TO_VTK_TYPE_MAP = {}  # Dummy map
    VTK_CLASSES_AVAILABLE = {}

    def get_vtk_attribute(attr_name, default_value=None):
        """Dummy function khi VTK không khả dụng"""
        return default_value

except Exception as e:
    HAS_VTK = False
    logger.error(f"Lỗi không mong đợi khi import VTK: {str(e)}")
    NUMPY_TO_VTK_TYPE_MAP = {}  # Dummy map
    VTK_CLASSES_AVAILABLE = {}

    def get_vtk_attribute(attr_name, default_value=None):
        """Dummy function khi VTK không khả dụng"""
        return default_value


try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.patches as patches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib không có sẵn - 2D plotting sẽ bị giới hạn")


class ViewMode(str, Enum):
    """Chế độ xem."""

    AXIAL = "axial"  # Cắt ngang
    SAGITTAL = "sagittal"  # Cắt dọc
    CORONAL = "coronal"  # Cắt trán
    VOLUME_3D = "3d"  # Hiển thị 3D


class RenderingMode(str, Enum):
    """Chế độ render."""

    VOLUME_RENDERING = "volume"  # Volume rendering
    SURFACE_RENDERING = "surface"  # Surface rendering
    SLICE_RENDERING = "slice"  # Slice rendering
    COMBINED = "combined"  # Kết hợp


@dataclass
class ViewportSettings:
    """Cài đặt viewport."""

    window_width: float = 400.0  # Độ rộng cửa sổ
    window_level: float = 40.0  # Mức cửa sổ
    zoom: float = 1.0  # Tỷ lệ phóng to
    pan_x: float = 0.0  # Di chuyển X
    pan_y: float = 0.0  # Di chuyển Y
    rotation_x: float = 0.0  # Xoay X
    rotation_y: float = 0.0  # Xoay Y
    rotation_z: float = 0.0  # Xoay Z


@dataclass
class StructureDisplaySettings:
    """Cài đặt hiển thị cấu trúc."""

    visible: bool = True
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # RGB
    opacity: float = 0.3
    wireframe: bool = False
    outline_width: float = 1.0


if HAS_PYQT and HAS_VTK:

    class VTK3DViewer(QWidget):
        """Widget hiển thị 3D sử dụng VTK."""

        # Signals
        slice_changed = pyqtSignal(int, str)  # slice_index, view_mode
        window_level_changed = pyqtSignal(float, float)  # window, level
        structure_selected = pyqtSignal(str)  # structure_name

        def __init__(self, parent=None):
            """
            Khởi tạo VTK3DViewer.

            Parameters
            ----------
            parent : QWidget, optional
                Widget cha
            """
            super().__init__(parent)

            # Khởi tạo biến trạng thái
            self._vtk_available = (
                HAS_VTK  # Theo mặc định, trạng thái dựa trên việc import VTK
            )
            self.structures = {}  # Dictionary lưu trữ structure masks
            self.structure_controls = {}  # Dictionary lưu trữ structure control settings
            self.structure_actors = {}  # Dictionary lưu trữ structure actors
            self.slice_actors = {}  # Dictionary lưu trữ slice actors

            # Khởi tạo các biến VTK
            self.renderer = None
            self.render_window = None
            self.interactor = None
            self.volume_actor = None
            self.volume_mapper = None
            self.slice_mapper = None
            self.slice_actor = None

            # Khởi tạo biến dữ liệu
            self.image_data = None
            self.image_spacing = (1.0, 1.0, 1.0)
            self.dose_data = None

            # Chế độ xem
            self.current_view_mode = ViewMode.AXIAL
            self.current_render_mode = RenderingMode.VOLUME_RENDERING

            # Thiết lập UI
            self.init_ui()
            self.setup_vtk()

        def init_ui(self):
            """Khởi tạo giao diện người dùng."""
            layout = QVBoxLayout(self)

            # Toolbar
            toolbar = self.create_toolbar()
            layout.addWidget(toolbar)

            # Main splitter
            splitter = QSplitter(Qt.Horizontal)
            layout.addWidget(splitter)

            # VTK widget
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            self.vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            splitter.addWidget(self.vtk_widget)

            # Control panel
            control_panel = self.create_control_panel()
            splitter.addWidget(control_panel)

            # Set splitter sizes
            splitter.setSizes([800, 300])

        def create_toolbar(self):
            """Tạo toolbar."""
            toolbar = QToolBar("3D Viewer Controls")

            # View mode selection
            self.view_mode_combo = QComboBox()
            self.view_mode_combo.addItems(["Axial", "Sagittal", "Coronal", "3D Volume"])
            self.view_mode_combo.currentTextChanged.connect(self.on_view_mode_changed)
            toolbar.addWidget(QLabel("View:"))
            toolbar.addWidget(self.view_mode_combo)

            toolbar.addSeparator()

            # Rendering mode
            self.render_mode_combo = QComboBox()
            self.render_mode_combo.addItems(
                ["Volume Rendering", "Surface Rendering", "Slice View", "Combined"]
            )
            self.render_mode_combo.currentTextChanged.connect(
                self.on_render_mode_changed
            )
            toolbar.addWidget(QLabel("Render:"))
            toolbar.addWidget(self.render_mode_combo)

            toolbar.addSeparator()

            # Reset view button
            reset_action = QAction("Reset View", self)
            reset_action.triggered.connect(self.reset_view)
            toolbar.addAction(reset_action)

            # Screenshot button
            screenshot_action = QAction("Screenshot", self)
            screenshot_action.triggered.connect(self.take_screenshot)
            toolbar.addAction(screenshot_action)

            return toolbar

        def create_control_panel(self):
            """Tạo panel điều khiển."""
            panel = QWidget()
            layout = QVBoxLayout(panel)

            # Image controls
            image_group = QGroupBox("Image Controls")
            image_layout = QGridLayout(image_group)

            # Slice selector
            image_layout.addWidget(QLabel("Slice:"), 0, 0)
            self.slice_slider = QSlider(Qt.Horizontal)
            self.slice_slider.valueChanged.connect(self.on_slice_changed)
            image_layout.addWidget(self.slice_slider, 0, 1)

            self.slice_spinbox = QSpinBox()
            self.slice_spinbox.valueChanged.connect(self.on_slice_spinbox_changed)
            image_layout.addWidget(self.slice_spinbox, 0, 2)

            # Window/Level controls
            image_layout.addWidget(QLabel("Window:"), 1, 0)
            self.window_spinbox = QDoubleSpinBox()
            self.window_spinbox.setRange(1, 4000)
            self.window_spinbox.setValue(400)
            self.window_spinbox.valueChanged.connect(self.on_window_level_changed)
            image_layout.addWidget(self.window_spinbox, 1, 1, 1, 2)

            image_layout.addWidget(QLabel("Level:"), 2, 0)
            self.level_spinbox = QDoubleSpinBox()
            self.level_spinbox.setRange(-1000, 3000)
            self.level_spinbox.setValue(40)
            self.level_spinbox.valueChanged.connect(self.on_window_level_changed)
            image_layout.addWidget(self.level_spinbox, 2, 1, 1, 2)

            layout.addWidget(image_group)

            # Structure controls
            structure_group = QGroupBox("Structures")
            structure_layout = QVBoxLayout(structure_group)

            # Add scrollable area for structures
            self.structure_controls_widget = QWidget()
            self.structure_controls_layout = QVBoxLayout(self.structure_controls_widget)
            structure_layout.addWidget(self.structure_controls_widget)

            layout.addWidget(structure_group)

            # Dose controls
            dose_group = QGroupBox("Dose Display")
            dose_layout = QGridLayout(dose_group)

            self.dose_visible_cb = QCheckBox("Show Dose")
            self.dose_visible_cb.toggled.connect(self.on_dose_visibility_changed)
            dose_layout.addWidget(self.dose_visible_cb, 0, 0)

            dose_layout.addWidget(QLabel("Opacity:"), 1, 0)
            self.dose_opacity_slider = QSlider(Qt.Horizontal)
            self.dose_opacity_slider.setRange(0, 100)
            self.dose_opacity_slider.setValue(50)
            self.dose_opacity_slider.valueChanged.connect(self.on_dose_opacity_changed)
            dose_layout.addWidget(self.dose_opacity_slider, 1, 1)

            layout.addWidget(dose_group)

            layout.addStretch()
            return panel

        def setup_vtk(self):
            """Thiết lập VTK rendering."""
            try:
                # Create renderer
                if VTK_CLASSES_AVAILABLE.get("vtkRenderer", False):
                    self.renderer = getattr(vtk, "vtkRenderer")()
                    self.renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background

                    # Create render window
                    self.render_window = self.vtk_widget.GetRenderWindow()
                    self.render_window.AddRenderer(self.renderer)

                    # Create interactor
                    self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

                    # Set up camera
                    camera = self.renderer.GetActiveCamera()
                    camera.SetPosition(0, 0, 300)
                    camera.SetFocalPoint(0, 0, 0)
                    camera.SetViewUp(0, 1, 0)

                    # Start interactor
                    self.interactor.Initialize()
                    self.interactor.Start()

                    logger.info("VTK rendering setup thành công")
                else:
                    raise ValueError("Class vtkRenderer không khả dụng")
            except Exception as e:
                logger.error(f"Lỗi khi thiết lập VTK rendering: {str(e)}")
                # Hiển thị thông báo lỗi cho người dùng
                if HAS_PYQT:
                    QMessageBox.warning(
                        self,
                        "VTK Error",
                        f"Không thể khởi tạo VTK renderer: {str(e)}\n"
                        "Viewer sẽ chuyển sang chế độ giới hạn.",
                    )
                # Đánh dấu là không khả dụng để sử dụng fallback
                self._vtk_available = False
                self._setup_fallback_display()

        def set_image_data(
            self,
            image_data: np.ndarray,
            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ):
            """
            Thiết lập dữ liệu hình ảnh.

            Parameters
            ----------
            image_data : np.ndarray
                Dữ liệu hình ảnh 3D
            spacing : Tuple[float, float, float]
                Khoảng cách giữa các voxel theo mm
            """
            try:
                if image_data is None:
                    logger.error(
                        "Không thể thiết lập dữ liệu hình ảnh: image_data là None"
                    )
                    return

                if not isinstance(image_data, np.ndarray):
                    logger.warning(
                        f"image_data không phải là numpy array, đang chuyển đổi từ {type(image_data)}"
                    )
                    try:
                        image_data = np.array(image_data)
                    except Exception as e:
                        logger.error(
                            f"Không thể chuyển đổi image_data thành numpy array: {str(e)}"
                        )
                        return

                # Lưu trữ dữ liệu
                self.image_data = image_data
                self.image_spacing = spacing

                # Điều chỉnh giá trị và kích thước sliders
                shape = image_data.shape
                self.slice_slider.setMaximum(shape[2] - 1)
                self.slice_slider.setValue(shape[2] // 2)
                self.slice_spinbox.setMaximum(shape[2] - 1)
                self.slice_spinbox.setValue(shape[2] // 2)

                # Kiểm tra nếu VTK không khả dụng hoặc bị lỗi trước đó
                if not HAS_VTK or not self._vtk_available:
                    logger.warning(
                        "VTK không khả dụng, sử dụng chế độ hiển thị giới hạn"
                    )
                    self._setup_fallback_display()
                    return

                # Get VTK classes
                vtkImageData = get_vtk_attribute("vtkImageData")
                vtkGPUVolumeRayCastMapper = get_vtk_attribute(
                    "vtkGPUVolumeRayCastMapper"
                )
                vtkVolume = get_vtk_attribute("vtkVolume")

                if None in [vtkImageData, vtkGPUVolumeRayCastMapper, vtkVolume]:
                    logger.error("Một hoặc nhiều class VTK cần thiết không khả dụng")
                    self._setup_fallback_display()
                    return

                # Tạo vtk ImageData
                vtk_image = vtkImageData()
                vtk_image.SetDimensions(shape[0], shape[1], shape[2])
                vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
                vtk_image.SetOrigin(0, 0, 0)

                # Xác định kiểu dữ liệu cho VTK dựa vào kiểu dữ liệu numpy
                vtk_data_type = NUMPY_TO_VTK_TYPE_MAP.get(image_data.dtype.type)

                if vtk_data_type is None:
                    logger.warning(
                        f"Kiểu dữ liệu {image_data.dtype} không được hỗ trợ trực tiếp, chuyển sang float32"
                    )
                    image_data = image_data.astype(np.float32)
                    vtk_data_type = get_vtk_attribute("VTK_FLOAT", 10)

                vtk_image.AllocateScalars(vtk_data_type, 1)

                # Sử dụng hàm tiện ích để chuyển đổi an toàn
                vtk_array = numpy_to_vtk_array(image_data, vtk_data_type)

                if vtk_array is None:
                    logger.error("Không thể chuyển đổi dữ liệu sang VTK array")
                    self._setup_fallback_display()
                    return

                # Đặt tên và gán vào point data
                vtk_array.SetName("ImageScalars")
                vtk_image.GetPointData().SetScalars(vtk_array)

                # Tạo các actors
                if self.renderer is None:
                    self.setup_vtk()

                # Kiểm tra lại nếu setup_vtk thất bại
                if not self._vtk_available:
                    self._setup_fallback_display()
                    return

                # Volume Rendering
                self.volume_mapper = vtkGPUVolumeRayCastMapper()
                self.volume_mapper.SetInputData(vtk_image)

                self.volume_actor = vtkVolume()
                self.volume_actor.SetMapper(self.volume_mapper)

                # Thiết lập thuộc tính hiển thị
                self.setup_volume_properties()

                # Thêm vào renderer
                self.renderer.AddViewProp(self.volume_actor)

                # Reset view để hiển thị toàn bộ dữ liệu
                self.renderer.ResetCamera()
                self.render_window.Render()

                logger.info(
                    f"Đã thiết lập dữ liệu hình ảnh {image_data.shape} thành công"
                )

            except Exception as e:
                logger.error(f"Lỗi khi thiết lập dữ liệu hình ảnh: {str(e)}")
                import traceback

                logger.debug(traceback.format_exc())

                # Chuyển sang fallback mode
                self._setup_fallback_display()

                # Thông báo cho người dùng
                if HAS_PYQT:
                    QMessageBox.warning(
                        self,
                        "Rendering Error",
                        f"Không thể render dữ liệu 3D: {str(e)}\n"
                        "Hiển thị 2D đơn giản sẽ được sử dụng thay thế.",
                    )

        def setup_volume_properties(self):
            """Thiết lập thuộc tính volume."""
            try:
                if not self._vtk_available:
                    logger.warning(
                        "VTK không khả dụng, không thể thiết lập volume properties"
                    )
                    return

                # Get VTK classes
                vtkVolumeProperty = get_vtk_attribute("vtkVolumeProperty")
                vtkColorTransferFunction = get_vtk_attribute("vtkColorTransferFunction")
                vtkPiecewiseFunction = get_vtk_attribute("vtkPiecewiseFunction")

                if None in [
                    vtkVolumeProperty,
                    vtkColorTransferFunction,
                    vtkPiecewiseFunction,
                ]:
                    logger.error("Một hoặc nhiều class VTK cần thiết không khả dụng")
                    return

                volume_property = vtkVolumeProperty()

                # Color transfer function
                color_tf = vtkColorTransferFunction()
                color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)  # Air - black
                color_tf.AddRGBPoint(-300, 0.4, 0.2, 0.0)  # Fat - brown
                color_tf.AddRGBPoint(0, 0.8, 0.6, 0.4)  # Soft tissue - beige
                color_tf.AddRGBPoint(400, 1.0, 0.8, 0.6)  # Bone - white-ish
                color_tf.AddRGBPoint(1000, 1.0, 1.0, 1.0)  # Dense bone - white

                # Opacity transfer function
                opacity_tf = vtkPiecewiseFunction()
                opacity_tf.AddPoint(-1000, 0.0)  # Air - transparent
                opacity_tf.AddPoint(-300, 0.1)  # Fat - slightly visible
                opacity_tf.AddPoint(0, 0.3)  # Soft tissue - visible
                opacity_tf.AddPoint(400, 0.5)  # Bone - more visible
                opacity_tf.AddPoint(1000, 0.8)  # Dense bone - most visible

                # Gradient opacity
                gradient_tf = vtkPiecewiseFunction()
                gradient_tf.AddPoint(0, 0.0)
                gradient_tf.AddPoint(90, 0.5)
                gradient_tf.AddPoint(100, 1.0)

                # Set volume properties
                volume_property.SetColor(color_tf)
                volume_property.SetScalarOpacity(opacity_tf)
                volume_property.SetGradientOpacity(gradient_tf)
                volume_property.SetInterpolationTypeToLinear()
                volume_property.ShadeOn()

                if self.volume_actor is not None:
                    self.volume_actor.SetProperty(volume_property)

            except Exception as e:
                logger.error(f"Lỗi khi thiết lập volume properties: {str(e)}")

        def add_structure(
            self,
            name: str,
            mask: np.ndarray,
            color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        ):
            """
            Thêm cấu trúc để hiển thị.

            Parameters
            ----------
            name : str
                Tên cấu trúc
            mask : np.ndarray
                Mảng 3D chứa mask của cấu trúc
            color : Tuple[float, float, float], optional
                Màu sắc (RGB) của cấu trúc, mặc định là (1.0, 0.0, 0.0) (đỏ)
            """
            try:
                if not self._vtk_available:
                    logger.warning("VTK không khả dụng, không thể thêm structure")
                    return

                if mask is None or not np.any(mask):
                    logger.warning(f"Structure mask '{name}' trống hoặc không hợp lệ")
                    return

                # Lưu mask
                self.structures[name] = mask

                # Tạo settings mặc định
                settings = StructureDisplaySettings(
                    visible=True,
                    color=color,
                    opacity=0.7,
                    wireframe=False,
                    outline_width=1.0,
                )
                self.structure_controls[name] = settings

                # Tạo actor cho cấu trúc
                if self.renderer is not None:
                    actor = self.create_structure_actor(name, mask, settings)
                    if actor is not None:
                        self.structure_actors[name] = actor
                        self.renderer.AddActor(actor)
                        actor.SetVisibility(settings.visible)

                        # Cập nhật view
                        if self.render_window is not None:
                            self.render_window.Render()
                    else:
                        logger.warning(f"Không thể tạo actor cho structure '{name}'")

                # Thêm control cho cấu trúc vào panel
                self.add_structure_control(name, settings)

            except Exception as e:
                logger.error(f"Lỗi khi thêm structure '{name}': {str(e)}")

        def create_structure_actor(
            self, name: str, mask: np.ndarray, settings: StructureDisplaySettings
        ):
            """
            Tạo actor VTK cho cấu trúc.

            Parameters
            ----------
            name : str
                Tên cấu trúc
            mask : np.ndarray
                Mảng 3D chứa mask của cấu trúc
            settings : StructureDisplaySettings
                Cài đặt hiển thị cho cấu trúc

            Returns
            -------
            vtk.vtkActor
                Actor VTK cho cấu trúc
            """
            try:
                if not self._vtk_available:
                    logger.warning("VTK không khả dụng, không thể tạo structure actor")
                    return None

                # Create VTK image data from mask
                vtkImageData = get_vtk_attribute("vtkImageData")
                if vtkImageData is None:
                    logger.error("vtkImageData không khả dụng")
                    return None

                vtk_mask = vtkImageData()
                vtk_mask.SetDimensions(mask.shape)
                vtk_mask.SetSpacing(1.0, 1.0, 1.0)

                # Convert mask to VTK
                flat_mask = mask.astype(np.uint8).flatten(order="F")

                vtkUnsignedCharArray = get_vtk_attribute("vtkUnsignedCharArray")
                if vtkUnsignedCharArray is None:
                    logger.error("vtkUnsignedCharArray không khả dụng")
                    return None

                vtk_array = vtkUnsignedCharArray()
                vtk_array.SetName("MaskArray")
                vtk_array.SetNumberOfTuples(len(flat_mask))

                # Manually set the values (safer than SetArray which is not always available)
                for i in range(len(flat_mask)):
                    vtk_array.SetValue(i, flat_mask[i])

                vtk_mask.GetPointData().SetScalars(vtk_array)

                # Create contour filter
                vtkMarchingCubes = get_vtk_attribute("vtkMarchingCubes")
                if vtkMarchingCubes is None:
                    logger.error("vtkMarchingCubes không khả dụng")
                    return None

                contour = vtkMarchingCubes()
                contour.SetInputData(vtk_mask)
                contour.SetValue(0, 0.5)
                contour.Update()

                # Create mapper
                vtkPolyDataMapper = get_vtk_attribute("vtkPolyDataMapper")
                if vtkPolyDataMapper is None:
                    logger.error("vtkPolyDataMapper không khả dụng")
                    return None

                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(contour.GetOutputPort())
                mapper.ScalarVisibilityOff()

                # Create actor
                vtkActor = get_vtk_attribute("vtkActor")
                if vtkActor is None:
                    logger.error("vtkActor không khả dụng")
                    return None

                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(settings.color)
                actor.GetProperty().SetOpacity(settings.opacity)

                if settings.wireframe:
                    actor.GetProperty().SetRepresentationToWireframe()
                    actor.GetProperty().SetLineWidth(settings.outline_width)

                return actor
            except Exception as e:
                logger.error(f"Lỗi khi tạo structure actor: {str(e)}")
                return None

        def add_structure_control(self, name: str, settings: StructureDisplaySettings):
            """Thêm điều khiển cho cấu trúc."""
            control_widget = QWidget()
            layout = QHBoxLayout(control_widget)

            # Visibility checkbox
            visible_cb = QCheckBox(name)
            visible_cb.setChecked(settings.visible)
            visible_cb.toggled.connect(
                lambda checked: self.on_structure_visibility_changed(name, checked)
            )
            layout.addWidget(visible_cb)

            # Opacity slider
            opacity_slider = QSlider(Qt.Horizontal)
            opacity_slider.setRange(0, 100)
            opacity_slider.setValue(int(settings.opacity * 100))
            opacity_slider.valueChanged.connect(
                lambda value: self.on_structure_opacity_changed(name, value / 100.0)
            )
            layout.addWidget(opacity_slider)

            # Color button
            color_btn = QPushButton()
            color_btn.setFixedSize(20, 20)
            color_btn.setStyleSheet(
                f"background-color: rgb({int(settings.color[0] * 255)}, {int(settings.color[1] * 255)}, {int(settings.color[2] * 255)})"
            )
            color_btn.clicked.connect(lambda: self.change_structure_color(name))
            layout.addWidget(color_btn)

            self.structure_controls_layout.addWidget(control_widget)

        def set_dose_data(self, dose_data: np.ndarray):
            """Thiết lập dữ liệu liều."""
            self.dose_data = dose_data
            # TODO: Implement dose visualization

        def on_view_mode_changed(self, mode: str):
            """
            Xử lý khi chế độ xem thay đổi.

            Parameters
            ----------
            mode : str
                Chế độ xem mới
            """
            mode = mode.lower()

            # Ẩn tất cả các slice actors
            for actor in self.slice_actors.values():
                self.renderer.RemoveActor(actor)

            # Hiển thị actor phù hợp với chế độ xem
            if mode == "axial" and "axial" in self.slice_actors:
                self.renderer.AddActor(self.slice_actors["axial"])
            elif mode == "sagittal" and "sagittal" in self.slice_actors:
                self.renderer.AddActor(self.slice_actors["sagittal"])
            elif mode == "coronal" and "coronal" in self.slice_actors:
                self.renderer.AddActor(self.slice_actors["coronal"])
            elif mode == "3d volume" or mode == "3d":
                # Không hiển thị slice actors trong chế độ 3D
                pass

            # Cập nhật viewport
            self.render_window.Render()

            # Gọi lại on_slice_changed để cập nhật slice hiện tại
            self.on_slice_changed(self.slice_slider.value())

        def on_render_mode_changed(self, mode: str):
            """Xử lý thay đổi chế độ render."""
            # TODO: Implement rendering mode switching
            pass

        def on_slice_changed(self, slice_index: int):
            """
            Xử lý khi slice thay đổi.

            Parameters
            ----------
            slice_index : int
                Chỉ số slice mới
            """
            try:
                # Đồng bộ giá trị với spinbox
                if self.slice_spinbox.value() != slice_index:
                    self.slice_spinbox.setValue(slice_index)

                if not self._vtk_available:
                    # Cập nhật hiển thị fallback
                    self._update_fallback_display()
                    return

                # Cập nhật slice mapper dựa trên chế độ xem hiện tại
                view_mode = self.view_mode_combo.currentText().lower()

                if view_mode == "axial" and "axial" in self.slice_actors:
                    mapper = self.slice_actors["axial"].GetMapper()
                    mapper.SetSliceNumber(slice_index)
                elif view_mode == "sagittal" and "sagittal" in self.slice_actors:
                    mapper = self.slice_actors["sagittal"].GetMapper()
                    mapper.SetSliceNumber(slice_index)
                elif view_mode == "coronal" and "coronal" in self.slice_actors:
                    mapper = self.slice_actors["coronal"].GetMapper()
                    mapper.SetSliceNumber(slice_index)

                # Render lại scene
                self.render_window.Render()

                # Phát signal
                self.slice_changed.emit(slice_index, view_mode)
            except Exception as e:
                logger.error(f"Lỗi khi thay đổi slice: {str(e)}")

        def on_slice_spinbox_changed(self, slice_index: int):
            """Xử lý thay đổi từ spinbox."""
            self.slice_slider.setValue(slice_index)

        def on_window_level_changed(self):
            """Xử lý thay đổi window/level."""
            window = self.window_spinbox.value()
            level = self.level_spinbox.value()
            self.window_level_changed.emit(window, level)
            # TODO: Update volume rendering

        def on_structure_visibility_changed(self, name: str, visible: bool):
            """Xử lý thay đổi hiển thị cấu trúc."""
            if name in self.structure_actors:
                self.structure_actors[name].SetVisibility(visible)
                self.structure_controls[name].visible = visible
                self.render_window.Render()

        def on_structure_opacity_changed(self, name: str, opacity: float):
            """Xử lý thay đổi độ trong suốt cấu trúc."""
            if name in self.structure_actors:
                self.structure_actors[name].GetProperty().SetOpacity(opacity)
                self.structure_controls[name].opacity = opacity
                self.render_window.Render()

        def change_structure_color(self, name: str):
            """Thay đổi màu cấu trúc."""
            # TODO: Implement color picker dialog
            pass

        def on_dose_visibility_changed(self, visible: bool):
            """Xử lý thay đổi hiển thị liều."""
            # TODO: Implement dose visibility
            pass

        def on_dose_opacity_changed(self, opacity: int):
            """Xử lý thay đổi độ trong suốt liều."""
            # TODO: Implement dose opacity
            pass

        def reset_view(self):
            """Reset về view mặc định."""
            self.renderer.ResetCamera()
            self.render_window.Render()

        def take_screenshot(self):
            """Chụp màn hình."""
            # TODO: Implement screenshot functionality
            pass

        def create_slice_actors(self, vtk_image):
            """
            Tạo các actors cho hiển thị slice 2D.

            Parameters
            ----------
            vtk_image : vtkImageData
                Dữ liệu hình ảnh VTK
            """
            try:
                if not self._vtk_available or self.renderer is None:
                    logger.warning(
                        "VTK không khả dụng hoặc renderer chưa được khởi tạo"
                    )
                    return

                # Get VTK classes
                vtkImageMapToColors = get_vtk_attribute("vtkImageMapToColors")
                vtkImageSliceMapper = get_vtk_attribute("vtkImageSliceMapper")
                vtkImageActor = get_vtk_attribute("vtkImageActor")

                if None in [vtkImageMapToColors, vtkImageSliceMapper, vtkImageActor]:
                    logger.error("Một hoặc nhiều class VTK cần thiết không khả dụng")
                    return

                # Lấy kích thước dữ liệu
                dimensions = vtk_image.GetDimensions()

                # Xóa các slice actors cũ nếu có
                for actor in self.slice_actors.values():
                    self.renderer.RemoveActor(actor)
                self.slice_actors = {}

                # Tạo slice actors cho 3 mặt phẳng
                # 1. Axial (XY plane)
                axial_colors = vtkImageMapToColors()
                axial_colors.SetInputData(vtk_image)
                if (
                    self.volume_actor is not None
                    and self.volume_actor.GetProperty() is not None
                ):
                    axial_colors.SetLookupTable(
                        self.volume_actor.GetProperty().GetRGBTransferFunction()
                    )

                axial_mapper = vtkImageSliceMapper()
                axial_mapper.SetInputConnection(axial_colors.GetOutputPort())
                axial_mapper.SetOrientationToZ()
                axial_mapper.SetSliceNumber(dimensions[2] // 2)

                axial_actor = vtkImageActor()
                axial_actor.SetMapper(axial_mapper)
                self.slice_actors["axial"] = axial_actor

                # 2. Sagittal (YZ plane)
                sagittal_colors = vtkImageMapToColors()
                sagittal_colors.SetInputData(vtk_image)
                if (
                    self.volume_actor is not None
                    and self.volume_actor.GetProperty() is not None
                ):
                    sagittal_colors.SetLookupTable(
                        self.volume_actor.GetProperty().GetRGBTransferFunction()
                    )

                sagittal_mapper = vtkImageSliceMapper()
                sagittal_mapper.SetInputConnection(sagittal_colors.GetOutputPort())
                sagittal_mapper.SetOrientationToX()
                sagittal_mapper.SetSliceNumber(dimensions[0] // 2)

                sagittal_actor = vtkImageActor()
                sagittal_actor.SetMapper(sagittal_mapper)
                self.slice_actors["sagittal"] = sagittal_actor

                # 3. Coronal (XZ plane)
                coronal_colors = vtkImageMapToColors()
                coronal_colors.SetInputData(vtk_image)
                if (
                    self.volume_actor is not None
                    and self.volume_actor.GetProperty() is not None
                ):
                    coronal_colors.SetLookupTable(
                        self.volume_actor.GetProperty().GetRGBTransferFunction()
                    )

                coronal_mapper = vtkImageSliceMapper()
                coronal_mapper.SetInputConnection(coronal_colors.GetOutputPort())
                coronal_mapper.SetOrientationToY()
                coronal_mapper.SetSliceNumber(dimensions[1] // 2)

                coronal_actor = vtkImageActor()
                coronal_actor.SetMapper(coronal_mapper)
                self.slice_actors["coronal"] = coronal_actor

                # Thêm actors vào renderer nếu ở chế độ slice
                if self.current_render_mode == RenderingMode.SLICE_RENDERING:
                    if self.current_view_mode == ViewMode.AXIAL:
                        self.renderer.AddActor(self.slice_actors["axial"])
                    elif self.current_view_mode == ViewMode.SAGITTAL:
                        self.renderer.AddActor(self.slice_actors["sagittal"])
                    elif self.current_view_mode == ViewMode.CORONAL:
                        self.renderer.AddActor(self.slice_actors["coronal"])

                logger.info("Đã tạo slice actors thành công")

            except Exception as e:
                logger.error(f"Lỗi khi tạo slice actors: {str(e)}")
                import traceback

                logger.debug(traceback.format_exc())

        def _setup_fallback_display(self):
            """Thiết lập hiển thị 2D đơn giản khi VTK không khả dụng."""
            try:
                if not HAS_MATPLOTLIB:
                    logger.warning(
                        "Cả VTK và Matplotlib đều không khả dụng, không thể hiển thị dữ liệu"
                    )
                    return

                # Xóa layout hiện tại
                if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
                    self.vtk_widget.setParent(None)

                # Tạo matplotlib widget
                self.figure = Figure(figsize=(8, 6), dpi=100)
                self.canvas = FigureCanvas(self.figure)

                # Thay thế vtk_widget bằng matplotlib canvas
                self.main_layout.insertWidget(0, self.canvas)

                # Hiển thị slice hiện tại
                self._update_fallback_display()

                logger.info("Đã thiết lập fallback display với Matplotlib")
            except Exception as e:
                logger.error(f"Lỗi khi thiết lập fallback display: {str(e)}")

        def _update_fallback_display(self):
            """Cập nhật hiển thị 2D khi ở chế độ fallback."""
            try:
                if not hasattr(self, "figure") or not hasattr(self, "canvas"):
                    return

                if not hasattr(self, "image_data") or self.image_data is None:
                    return

                # Lấy slice hiện tại
                current_slice = self.slice_slider.value()
                if current_slice >= self.image_data.shape[2]:
                    current_slice = self.image_data.shape[2] - 1

                # Hiển thị slice
                self.figure.clear()
                ax = self.figure.add_subplot(111)

                # Hiển thị CT data
                if self.image_data is not None:
                    slice_data = self.image_data[:, :, current_slice]
                    ax.imshow(slice_data.T, cmap="gray", aspect="auto")

                # Hiển thị cấu trúc nếu có
                for name, data in self.structures.items():
                    if data["visible"] and data["mask"].shape[2] > current_slice:
                        mask_slice = data["mask"][:, :, current_slice]
                        if np.any(mask_slice):
                            mask_display = np.ma.masked_where(
                                mask_slice.T == 0, mask_slice.T
                            )
                            ax.imshow(mask_display, alpha=0.3, cmap="jet")

                ax.set_title(f"Slice {current_slice}")
                ax.set_axis_off()
                self.canvas.draw()

            except Exception as e:
                logger.error(f"Lỗi khi cập nhật fallback display: {str(e)}")

elif HAS_PYQT and HAS_MATPLOTLIB:

    class Matplotlib3DViewer(QWidget):
        """Widget hiển thị 3D sử dụng Matplotlib (fallback)."""

        # Signals
        slice_changed = pyqtSignal(int, str)
        window_level_changed = pyqtSignal(float, float)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.image_data = None
            self.dose_data = None
            self.structures = {}

            self.init_ui()

        def init_ui(self):
            """Khởi tạo giao diện."""
            layout = QVBoxLayout(self)

            # Create matplotlib figure
            self.figure = Figure(figsize=(12, 8))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)

            # Control panel
            control_panel = self.create_control_panel()
            layout.addWidget(control_panel)

        def create_control_panel(self):
            """Tạo panel điều khiển."""
            panel = QWidget()
            layout = QHBoxLayout(panel)

            # Slice controls
            layout.addWidget(QLabel("Slice:"))
            self.slice_slider = QSlider(Qt.Horizontal)
            self.slice_slider.valueChanged.connect(self.update_display)
            layout.addWidget(self.slice_slider)

            # Window/Level
            layout.addWidget(QLabel("Window:"))
            self.window_spinbox = QSpinBox()
            self.window_spinbox.setRange(1, 4000)
            self.window_spinbox.setValue(400)
            self.window_spinbox.valueChanged.connect(self.update_display)
            layout.addWidget(self.window_spinbox)

            layout.addWidget(QLabel("Level:"))
            self.level_spinbox = QSpinBox()
            self.level_spinbox.setRange(-1000, 3000)
            self.level_spinbox.setValue(40)
            self.level_spinbox.valueChanged.connect(self.update_display)
            layout.addWidget(self.level_spinbox)

            return panel

        def set_image_data(
            self,
            image_data: np.ndarray,
            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ):
            """Thiết lập dữ liệu ảnh."""
            self.image_data = image_data

            if image_data is not None:
                self.slice_slider.setRange(0, image_data.shape[2] - 1)
                self.slice_slider.setValue(image_data.shape[2] // 2)
                self.update_display()

        def add_structure(
            self,
            name: str,
            mask: np.ndarray,
            color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        ):
            """Thêm cấu trúc."""
            self.structures[name] = {"mask": mask, "color": color, "visible": True}
            self.update_display()

        def set_dose_data(self, dose_data: np.ndarray):
            """Thiết lập dữ liệu liều."""
            self.dose_data = dose_data
            self.update_display()

        def update_display(self):
            """Cập nhật hiển thị."""
            self.figure.clear()

            if self.image_data is None:
                self.canvas.draw()
                return

            # Create 2x2 subplot layout
            gs = self.figure.add_gridspec(2, 2)

            slice_idx = self.slice_slider.value()
            window = self.window_spinbox.value()
            level = self.level_spinbox.value()

            # Axial view
            ax1 = self.figure.add_subplot(gs[0, 0])
            axial_slice = self.image_data[:, :, slice_idx]
            vmin = level - window / 2
            vmax = level + window / 2
            ax1.imshow(axial_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax1.set_title(f"Axial (Slice {slice_idx})")
            ax1.axis("off")

            # Sagittal view
            ax2 = self.figure.add_subplot(gs[0, 1])
            sag_slice = self.image_data[slice_idx, :, :]
            ax2.imshow(sag_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax2.set_title(f"Sagittal (Slice {slice_idx})")
            ax2.axis("off")

            # Coronal view
            ax3 = self.figure.add_subplot(gs[1, 0])
            cor_slice = self.image_data[:, slice_idx, :]
            ax3.imshow(cor_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax3.set_title(f"Coronal (Slice {slice_idx})")
            ax3.axis("off")

            # 3D view (simple projection)
            ax4 = self.figure.add_subplot(gs[1, 1])
            mip = np.max(self.image_data, axis=2)
            ax4.imshow(mip, cmap="gray", origin="lower")
            ax4.set_title("Maximum Intensity Projection")
            ax4.axis("off")

            # Overlay structures
            for name, struct_data in self.structures.items():
                if struct_data["visible"]:
                    mask = struct_data["mask"]
                    color = struct_data["color"]

                    # Overlay on each view
                    if mask.shape == self.image_data.shape:
                        # Axial
                        mask_slice = mask[:, :, slice_idx]
                        if np.any(mask_slice):
                            contours = plt.contour(
                                mask_slice, levels=[0.5], colors=[color], linewidths=1.5
                            )
                            ax1.contour(
                                mask_slice, levels=[0.5], colors=[color], linewidths=1.5
                            )

                        # Sagittal
                        mask_sag = mask[slice_idx, :, :]
                        if np.any(mask_sag):
                            ax2.contour(
                                mask_sag, levels=[0.5], colors=[color], linewidths=1.5
                            )

                        # Coronal
                        mask_cor = mask[:, slice_idx, :]
                        if np.any(mask_cor):
                            ax3.contour(
                                mask_cor, levels=[0.5], colors=[color], linewidths=1.5
                            )

            plt.tight_layout()
            self.canvas.draw()

else:

    class Dummy3DViewer(QWidget):
        """Dummy viewer khi không có thư viện cần thiết."""

        def __init__(self, parent=None):
            super().__init__(parent) if HAS_PYQT else None
            logger.error("Không có thư viện cần thiết để tạo 3D Viewer")

        def set_image_data(self, *args, **kwargs):
            logger.error("3D Viewer không khả dụng")

        def add_structure(self, *args, **kwargs):
            logger.error("3D Viewer không khả dụng")

        def set_dose_data(self, *args, **kwargs):
            logger.error("3D Viewer không khả dụng")


# Export appropriate viewer based on available libraries
if HAS_PYQT and HAS_VTK:
    Viewer3D = VTK3DViewer
    logger.info("Sử dụng VTK3DViewer")
elif HAS_PYQT and HAS_MATPLOTLIB:
    Viewer3D = Matplotlib3DViewer
    logger.info("Sử dụng Matplotlib3DViewer (fallback)")
else:
    Viewer3D = Dummy3DViewer
    logger.warning("Không có viewer 3D nào khả dụng")


def create_3d_viewer(parent=None) -> QWidget:
    """Factory function để tạo 3D viewer phù hợp."""
    return Viewer3D(parent)


# Function để chuyển đổi numpy array sang VTK array một cách an toàn
def numpy_to_vtk_array(np_array, vtk_data_type=None):
    """
    Chuyển đổi numpy array sang VTK array một cách an toàn.

    Parameters
    ----------
    np_array : np.ndarray
        Mảng numpy cần chuyển đổi
    vtk_data_type : int, optional
        Kiểu dữ liệu VTK, nếu None sẽ tự động xác định

    Returns
    -------
    vtk.vtkDataArray
        VTK array đã chuyển đổi hoặc None nếu gặp lỗi
    """
    if not HAS_VTK:
        logger.warning("VTK không khả dụng, không thể chuyển đổi array")
        return None

    try:
        # Xác định kiểu dữ liệu VTK tự động nếu không được chỉ định
        if vtk_data_type is None:
            # Sử dụng get_vtk_attribute thay vì truy cập trực tiếp vtk.VTK_FLOAT
            vtk_data_type = NUMPY_TO_VTK_TYPE_MAP.get(
                np_array.dtype.type, get_vtk_attribute("VTK_FLOAT", 10)
            )

        # Đảm bảo dữ liệu ở định dạng C-contiguous để xử lý đúng trong VTK
        if not np_array.flags["C_CONTIGUOUS"]:
            logger.warning("Chuyển đổi dữ liệu sang C-contiguous format")
            np_array = np.ascontiguousarray(np_array)

        # Sử dụng phương pháp thay thế dựa vào kiểu dữ liệu
        vtk_array = None

        # Chọn loại array phù hợp với kiểu dữ liệu numpy
        if np_array.dtype.type == np.float32:
            vtkFloatArray = get_vtk_attribute("vtkFloatArray")
            if vtkFloatArray is not None:
                vtk_array = vtkFloatArray()
            else:
                logger.error("vtkFloatArray không khả dụng")
                return None
        elif np_array.dtype.type == np.float64:
            vtkDoubleArray = get_vtk_attribute("vtkDoubleArray")
            if vtkDoubleArray is not None:
                vtk_array = vtkDoubleArray()
            else:
                logger.error("vtkDoubleArray không khả dụng")
                return None
        elif np_array.dtype.type == np.int32:
            vtkIntArray = get_vtk_attribute("vtkIntArray")
            if vtkIntArray is not None:
                vtk_array = vtkIntArray()
            else:
                logger.error("vtkIntArray không khả dụng")
                return None
        elif np_array.dtype.type == np.uint8:
            vtkUnsignedCharArray = get_vtk_attribute("vtkUnsignedCharArray")
            if vtkUnsignedCharArray is not None:
                vtk_array = vtkUnsignedCharArray()
            else:
                logger.error("vtkUnsignedCharArray không khả dụng")
                return None
        else:
            # Fallback to float if specific type not available
            vtkFloatArray = get_vtk_attribute("vtkFloatArray")
            if vtkFloatArray is not None:
                vtk_array = vtkFloatArray()
                # Convert data to float32
                np_array = np_array.astype(np.float32)
            else:
                logger.error("Không có loại VTK array nào khả dụng")
                return None

        if vtk_array is None:
            logger.error("Không thể tạo VTK array phù hợp")
            return None

        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetNumberOfTuples(np_array.size)

        # Phương pháp 1: Sử dụng SetArray
        success = False
        try:
            if hasattr(vtk_array, "SetArray"):
                vtk_array.SetArray(np_array.ravel(), np_array.size, 1)
                success = True
        except Exception as e:
            logger.warning(f"Lỗi khi sử dụng SetArray: {str(e)}")
            success = False

        # Phương pháp 2: Thủ công set từng giá trị nếu phương pháp 1 thất bại
        if not success:
            flat_data = np_array.ravel()
            for i in range(len(flat_data)):
                vtk_array.SetValue(i, float(flat_data[i]))

        return vtk_array

    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi numpy array sang VTK array: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())
        return None
