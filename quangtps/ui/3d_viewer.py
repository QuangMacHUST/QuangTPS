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
except ImportError:
    HAS_VTK = False
    logger.warning("VTK không có sẵn - 3D rendering sẽ bị giới hạn")

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
            super().__init__(parent)
            self.image_data = None
            self.dose_data = None
            self.structures = {}  # Dict[str, np.ndarray]
            self.structure_settings = {}  # Dict[str, StructureDisplaySettings]

            # VTK objects
            self.renderer = None
            self.render_window = None
            self.interactor = None
            self.volume_mapper = None
            self.volume_actor = None
            self.slice_actors = {}
            self.structure_actors = {}

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
            self.structure_controls = QWidget()
            self.structure_controls_layout = QVBoxLayout(self.structure_controls)
            structure_layout.addWidget(self.structure_controls)

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
            # Create renderer
            self.renderer = vtk.vtkRenderer()
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

        def set_image_data(
            self,
            image_data: np.ndarray,
            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ):
            """Thiết lập dữ liệu ảnh."""
            self.image_data = image_data

            if image_data is None:
                return

            # Update slice controls
            self.slice_slider.setRange(0, image_data.shape[2] - 1)
            self.slice_spinbox.setRange(0, image_data.shape[2] - 1)
            self.slice_slider.setValue(image_data.shape[2] // 2)

            # Create VTK image data
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(image_data.shape)
            vtk_image.SetSpacing(spacing)
            vtk_image.SetOrigin(0, 0, 0)

            # Convert numpy array to VTK
            flat_array = image_data.flatten(order="F").astype(
                np.float32
            )  # Đảm bảo kiểu float32
            vtk_array = vtk.vtkFloatArray()
            vtk_array.SetName("ImageArray")
            vtk_array.SetNumberOfComponents(1)
            vtk_array.SetNumberOfTuples(len(flat_array))

            # Sử dụng SetData thay vì SetArray để tránh lỗi
            try:
                # Thử cách mới trước
                vtk_array.SetData(vtk.numpy_to_vtk(flat_array))
            except (AttributeError, TypeError):
                # Fallback cho VTK cũ
                try:
                    for i, value in enumerate(flat_array):
                        vtk_array.SetValue(i, float(value))
                except Exception as e:
                    logger.error(f"Không thể thiết lập VTK array: {e}")
                    return

            vtk_image.GetPointData().SetScalars(vtk_array)

            # Create volume mapper
            self.volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
            self.volume_mapper.SetInputData(vtk_image)

            # Create volume actor
            self.volume_actor = vtk.vtkVolume()
            self.volume_actor.SetMapper(self.volume_mapper)

            # Set up volume properties
            self.setup_volume_properties()

            # Add to renderer
            self.renderer.AddVolume(self.volume_actor)
            self.render_window.Render()

        def setup_volume_properties(self):
            """Thiết lập thuộc tính volume."""
            volume_property = vtk.vtkVolumeProperty()

            # Color transfer function
            color_tf = vtk.vtkColorTransferFunction()
            color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)  # Air - black
            color_tf.AddRGBPoint(-300, 0.4, 0.2, 0.0)  # Fat - brown
            color_tf.AddRGBPoint(0, 0.8, 0.6, 0.4)  # Soft tissue - beige
            color_tf.AddRGBPoint(300, 1.0, 1.0, 1.0)  # Bone - white
            color_tf.AddRGBPoint(3000, 1.0, 1.0, 1.0)  # Dense bone - white

            # Opacity transfer function
            opacity_tf = vtk.vtkPiecewiseFunction()
            opacity_tf.AddPoint(-1000, 0.0)  # Air - transparent
            opacity_tf.AddPoint(-300, 0.1)  # Fat - slightly visible
            opacity_tf.AddPoint(0, 0.3)  # Soft tissue - visible
            opacity_tf.AddPoint(300, 0.8)  # Bone - opaque
            opacity_tf.AddPoint(3000, 1.0)  # Dense bone - opaque

            # Gradient opacity
            gradient_tf = vtk.vtkPiecewiseFunction()
            gradient_tf.AddPoint(0, 0.0)
            gradient_tf.AddPoint(90, 0.5)
            gradient_tf.AddPoint(100, 1.0)

            volume_property.SetColor(color_tf)
            volume_property.SetScalarOpacity(opacity_tf)
            volume_property.SetGradientOpacity(gradient_tf)
            volume_property.SetInterpolationTypeToLinear()
            volume_property.ShadeOn()
            volume_property.SetAmbient(0.4)
            volume_property.SetDiffuse(0.6)
            volume_property.SetSpecular(0.2)

            self.volume_actor.SetProperty(volume_property)

        def add_structure(
            self,
            name: str,
            mask: np.ndarray,
            color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        ):
            """Thêm cấu trúc."""
            self.structures[name] = mask

            settings = StructureDisplaySettings(visible=True, color=color, opacity=0.3)
            self.structure_settings[name] = settings

            # Create structure actor
            self.create_structure_actor(name, mask, settings)

            # Add to control panel
            self.add_structure_control(name, settings)

        def create_structure_actor(
            self, name: str, mask: np.ndarray, settings: StructureDisplaySettings
        ):
            """Tạo VTK actor cho cấu trúc."""
            if mask is None or not np.any(mask):
                return

            # Create VTK image data from mask
            vtk_mask = vtk.vtkImageData()
            vtk_mask.SetDimensions(mask.shape)
            vtk_mask.SetSpacing(1.0, 1.0, 1.0)

            # Convert mask to VTK
            flat_mask = mask.astype(np.uint8).flatten(order="F")
            vtk_array = vtk.vtkUnsignedCharArray()
            vtk_array.SetName("MaskArray")
            vtk_array.SetNumberOfTuples(len(flat_mask))

            # Sử dụng SetData thay vì SetArray để tránh lỗi
            try:
                # Thử cách mới trước
                vtk_array.SetData(vtk.numpy_to_vtk(flat_mask))
            except (AttributeError, TypeError):
                # Fallback cho VTK cũ
                try:
                    for i, value in enumerate(flat_mask):
                        vtk_array.SetValue(i, int(value))
                except Exception as e:
                    logger.error(f"Không thể thiết lập VTK mask array: {e}")
                    return

            vtk_mask.GetPointData().SetScalars(vtk_array)

            # Create contour filter
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(vtk_mask)
            contour.SetValue(0, 0.5)
            contour.Update()

            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            mapper.ScalarVisibilityOff()

            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(settings.color)
            actor.GetProperty().SetOpacity(settings.opacity)
            actor.SetVisibility(settings.visible)

            if settings.wireframe:
                actor.GetProperty().SetRepresentationToWireframe()
            else:
                actor.GetProperty().SetRepresentationToSurface()

            self.structure_actors[name] = actor
            self.renderer.AddActor(actor)

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
            """Xử lý thay đổi chế độ xem."""
            mode_map = {
                "Axial": ViewMode.AXIAL,
                "Sagittal": ViewMode.SAGITTAL,
                "Coronal": ViewMode.CORONAL,
                "3D Volume": ViewMode.VOLUME_3D,
            }

            view_mode = mode_map.get(mode, ViewMode.AXIAL)
            # TODO: Implement view mode switching

        def on_render_mode_changed(self, mode: str):
            """Xử lý thay đổi chế độ render."""
            # TODO: Implement rendering mode switching
            pass

        def on_slice_changed(self, slice_index: int):
            """Xử lý thay đổi slice."""
            self.slice_spinbox.setValue(slice_index)
            self.slice_changed.emit(slice_index, "axial")

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
                self.structure_settings[name].visible = visible
                self.render_window.Render()

        def on_structure_opacity_changed(self, name: str, opacity: float):
            """Xử lý thay đổi độ trong suốt cấu trúc."""
            if name in self.structure_actors:
                self.structure_actors[name].GetProperty().SetOpacity(opacity)
                self.structure_settings[name].opacity = opacity
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
