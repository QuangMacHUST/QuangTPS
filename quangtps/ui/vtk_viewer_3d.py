#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module VTKViewer3D cho QuangTPS.

Cung cấp giao diện trực quan 3D tổng hợp sử dụng thư viện VTK.
Được thiết kế để tích hợp chặt chẽ với các module của QuangTPS như
dose visualization, structure visualization, và beam visualization.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import math
import time
from datetime import datetime

# Thử import các thư viện phụ thuộc
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    VTK_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import VTK: {e}")
    VTK_AVAILABLE = False

    # Tạo module vtk giả để tránh lỗi linter
    class vtk:
        # Tạo các lớp giả cho VTK
        class vtkRenderer:
            def SetBackground(self, *args):
                pass

            def SetGradientBackground(self, *args):
                pass

            def SetBackground2(self, *args):
                pass

            def AddActor(self, *args):
                pass

            def RemoveActor(self, *args):
                pass

            def ResetCamera(self, *args):
                pass

            def GetActiveCamera(self, *args):
                return None

            def SetUseFXAA(self, *args):
                pass

        class vtkRenderWindow:
            def AddRenderer(self, *args):
                pass

            def SetSize(self, *args):
                pass

            def SetMultiSamples(self, *args):
                pass

            def Render(self, *args):
                pass

            def GetInteractor(self, *args):
                return None

        class vtkRenderWindowInteractor:
            def SetInteractorStyle(self, *args):
                pass

            def RemoveObservers(self, *args):
                pass

        class vtkInteractorStyleTrackballCamera:
            pass

        class vtkInteractorStyleJoystickCamera:
            pass

        class vtkCamera:
            def SetPosition(self, *args):
                pass

            def SetFocalPoint(self, *args):
                pass

            def SetViewUp(self, *args):
                pass

            def GetPosition(self, *args):
                return (0, 0, 0)

            def GetFocalPoint(self, *args):
                return (0, 0, 0)

            def GetViewUp(self, *args):
                return (0, 0, 1)

            def Azimuth(self, *args):
                pass

            def Elevation(self, *args):
                pass

            def Roll(self, *args):
                pass

            def Zoom(self, *args):
                pass

        class vtkActor:
            def SetMapper(self, *args):
                pass

            def GetProperty(self, *args):
                class Property:
                    def SetColor(self, *args):
                        pass

                    def SetOpacity(self, *args):
                        pass

                return Property()

        class vtkProperty:
            def SetColor(self, *args):
                pass

            def SetOpacity(self, *args):
                pass

        class vtkOrientationMarkerWidget:
            def SetOrientationMarker(self, *args):
                pass

            def SetInteractor(self, *args):
                pass

            def SetViewport(self, *args):
                pass

            def SetEnabled(self, *args):
                pass

            def InteractiveOff(self, *args):
                pass

        class vtkAxesActor:
            pass

        class vtkTextActor:
            def SetInput(self, *args):
                pass

            def SetPosition(self, *args):
                pass

            def GetTextProperty(self, *args):
                class TextProperty:
                    def SetColor(self, *args):
                        pass

                    def SetFontSize(self, *args):
                        pass

                return TextProperty()

        class vtkCaptionActor2D:
            pass

        class vtkTextProperty:
            pass

        class vtkWindowToImageFilter:
            def SetInput(self, *args):
                pass

            def SetScale(self, *args):
                pass

            def SetInputBufferTypeToRGB(self, *args):
                pass

            def ReadFrontBufferOff(self, *args):
                pass

            def Update(self, *args):
                pass

            def GetOutput(self, *args):
                class Output:
                    def GetDimensions(self, *args):
                        return (0, 0, 0)

                    def GetPointData(self, *args):
                        class PointData:
                            def GetScalars(self, *args):
                                class Scalars:
                                    def GetNumberOfComponents(self, *args):
                                        return 3

                                    def GetTuple(self, *args):
                                        return (0, 0, 0)

                                return Scalars()

                        return PointData()

                return Output()

        class vtkSphereSource:
            def SetRadius(self, *args):
                pass

            def SetThetaResolution(self, *args):
                pass

            def SetPhiResolution(self, *args):
                pass

            def Update(self, *args):
                pass

            def GetOutput(self, *args):
                return None

        class vtkPolyDataMapper:
            def SetInputData(self, *args):
                pass

        class vtkCubeSource:
            def SetXLength(self, *args):
                pass

            def SetYLength(self, *args):
                pass

            def SetZLength(self, *args):
                pass

            def SetCenter(self, *args):
                pass

            def Update(self, *args):
                pass

            def GetOutput(self, *args):
                return None


# Khai báo các class VTK để tránh lỗi linter
if VTK_AVAILABLE:
    vtkRenderer = vtk.vtkRenderer
    vtkRenderWindow = vtk.vtkRenderWindow
    vtkRenderWindowInteractor = vtk.vtkRenderWindowInteractor
    vtkInteractorStyleTrackballCamera = vtk.vtkInteractorStyleTrackballCamera
    vtkCamera = vtk.vtkCamera
    vtkActor = vtk.vtkActor
    vtkProperty = vtk.vtkProperty
    vtkOrientationMarkerWidget = vtk.vtkOrientationMarkerWidget
    vtkAxesActor = vtk.vtkAxesActor
    vtkTextActor = vtk.vtkTextActor
    vtkCaptionActor2D = vtk.vtkCaptionActor2D
    vtkTextProperty = vtk.vtkTextProperty
else:
    vtkRenderer = vtk.vtkRenderer
    vtkRenderWindow = vtk.vtkRenderWindow
    vtkRenderWindowInteractor = vtk.vtkRenderWindowInteractor
    vtkInteractorStyleTrackballCamera = vtk.vtkInteractorStyleTrackballCamera
    vtkCamera = vtk.vtkCamera
    vtkActor = vtk.vtkActor
    vtkProperty = vtk.vtkProperty
    vtkOrientationMarkerWidget = vtk.vtkOrientationMarkerWidget
    vtkAxesActor = vtk.vtkAxesActor
    vtkTextActor = vtk.vtkTextActor
    vtkCaptionActor2D = vtk.vtkCaptionActor2D
    vtkTextProperty = vtk.vtkTextProperty

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
        QToolBar,
        QAction,
        QActionGroup,
        QApplication,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
    from PyQt5.QtGui import QColor, QFont, QIcon

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

    class QApplicationFallback:
        def __init__(self, argv):
            pass

        def exec_(self):
            return 0

    QApplication = QApplicationFallback


from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class CameraController:
    """
    Điều khiển camera VTK với các chức năng nâng cao.

    Quản lý các góc nhìn, các chế độ camera, và các hoạt ảnh chuyển đổi
    giữa các góc nhìn khác nhau. Hỗ trợ các góc nhìn chuẩn như axial,
    sagittal, coronal và các góc tùy chỉnh.
    """

    def __init__(self, renderer):
        """
        Khởi tạo camera controller.

        Parameters:
        -----------
        renderer : vtkRenderer
            VTK renderer to control
        """
        self.renderer = renderer
        self.camera = renderer.GetActiveCamera() if renderer else None
        self.last_update_time = time.time()
        self.animation_duration = 0.5  # seconds
        self.animating = False
        self.start_position = None
        self.start_focal_point = None
        self.start_view_up = None
        self.target_position = None
        self.target_focal_point = None
        self.target_view_up = None
        self.animation_start_time = None

        # Standard view presets
        self.standard_views = {
            "axial": {
                "position": (0, 0, 1000),
                "focal_point": (0, 0, 0),
                "view_up": (0, 1, 0),
            },
            "sagittal": {
                "position": (1000, 0, 0),
                "focal_point": (0, 0, 0),
                "view_up": (0, 0, 1),
            },
            "coronal": {
                "position": (0, 1000, 0),
                "focal_point": (0, 0, 0),
                "view_up": (0, 0, 1),
            },
            "3d": {
                "position": (500, 500, 500),
                "focal_point": (0, 0, 0),
                "view_up": (0, 0, 1),
            },
        }

    def set_view(self, view_name, animate=True):
        """
        Thiết lập camera đến một góc nhìn tiêu chuẩn.

        Parameters:
        -----------
        view_name : str
            Name of the standard view ("axial", "sagittal", "coronal", "3d")
        animate : bool
            Whether to animate the transition
        """
        if not self.camera or view_name not in self.standard_views:
            return False

        view = self.standard_views[view_name]

        if animate:
            self._start_animation(
                view["position"], view["focal_point"], view["view_up"]
            )
        else:
            self.camera.SetPosition(*view["position"])
            self.camera.SetFocalPoint(*view["focal_point"])
            self.camera.SetViewUp(*view["view_up"])

        return True

    def set_azimuth(self, angle):
        """
        Xoay camera theo góc azimuth.

        Parameters:
        -----------
        angle : float
            Góc azimuth trong độ (0-360)
        """
        if not self.camera:
            return

        self.camera.Azimuth(angle)

    def set_elevation(self, angle):
        """
        Xoay camera theo góc elevation.

        Parameters:
        -----------
        angle : float
            Góc elevation trong độ (-90 đến 90)
        """
        if not self.camera:
            return

        self.camera.Elevation(angle)

    def set_roll(self, angle):
        """
        Xoay camera theo góc roll.

        Parameters:
        -----------
        angle : float
            Góc roll trong độ (0-360)
        """
        if not self.camera:
            return

        self.camera.Roll(angle)

    def zoom(self, factor):
        """
        Phóng to hoặc thu nhỏ góc nhìn.

        Parameters:
        -----------
        factor : float
            Hệ số zoom. Giá trị > 1 là phóng to, < 1 là thu nhỏ.
        """
        if not self.camera:
            return

        self.camera.Zoom(factor)

    def update(self):
        """Cập nhật camera khi đang có hoạt ảnh."""
        if not self.animating:
            return False

        current_time = time.time()
        elapsed = current_time - self.animation_start_time

        if elapsed >= self.animation_duration:
            # Animation complete
            self.camera.SetPosition(*self.target_position)
            self.camera.SetFocalPoint(*self.target_focal_point)
            self.camera.SetViewUp(*self.target_view_up)
            self.animating = False
            return True

        # Calculate interpolation factor (0-1)
        t = elapsed / self.animation_duration

        # Apply easing function (ease in/out)
        t = self._ease_in_out(t)

        # Interpolate position
        pos = self._interpolate(self.start_position, self.target_position, t)
        fp = self._interpolate(self.start_focal_point, self.target_focal_point, t)
        vu = self._interpolate(self.start_view_up, self.target_view_up, t)

        # Set camera properties
        self.camera.SetPosition(*pos)
        self.camera.SetFocalPoint(*fp)
        self.camera.SetViewUp(*vu)

        return True

    def _start_animation(self, target_position, target_focal_point, target_view_up):
        """
        Bắt đầu hoạt ảnh chuyển đổi camera.

        Parameters:
        -----------
        target_position : tuple
            Target position (x, y, z)
        target_focal_point : tuple
            Target focal point (x, y, z)
        target_view_up : tuple
            Target view up vector (x, y, z)
        """
        if not self.camera:
            return

        self.start_position = self.camera.GetPosition()
        self.start_focal_point = self.camera.GetFocalPoint()
        self.start_view_up = self.camera.GetViewUp()

        self.target_position = target_position
        self.target_focal_point = target_focal_point
        self.target_view_up = target_view_up

        self.animation_start_time = time.time()
        self.animating = True

    def _interpolate(self, start, end, t):
        """
        Nội suy tuyến tính giữa 2 điểm 3D.

        Parameters:
        -----------
        start : tuple
            Starting point (x, y, z)
        end : tuple
            Ending point (x, y, z)
        t : float
            Interpolation factor (0-1)

        Returns:
        --------
        tuple
            Interpolated point (x, y, z)
        """
        return (
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
            start[2] + (end[2] - start[2]) * t,
        )

    def _ease_in_out(self, t):
        """
        Hàm easing để làm mịn chuyển động.

        Parameters:
        -----------
        t : float
            Input time factor (0-1)

        Returns:
        --------
        float
            Eased time factor (0-1)
        """
        # Cubic easing
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t


class VTKViewer3D(QWidget):
    """
    Enhanced VTK-based 3D viewer for QuangTPS.

    This widget provides a 3D visualization platform that can display
    various types of medical data including patient anatomy, radiation dose,
    beam arrangements, and more. It integrates with VTK for high-performance
    3D rendering and provides an intuitive API for managing the displayed objects.

    The component is designed to handle memory efficiently with large datasets
    and provides hooks for responsive rendering during user interaction.
    """

    # Signals
    view_changed = pyqtSignal(str)  # Emitted when view changes (3D, axial, etc.)
    render_complete = pyqtSignal()  # Emitted when rendering is complete
    object_selected = pyqtSignal(str, object)  # Emitted when object is selected

    def __init__(self, parent=None, memory_threshold_mb=2000, quality_level="high"):
        """
        Initialize the 3D VTK viewer.

        Parameters:
        -----------
        parent : QWidget, optional
            Parent widget
        memory_threshold_mb : int, optional
            Memory threshold in MB for downsampling
        quality_level : str, optional
            Rendering quality level ("low", "medium", "high")
        """
        super().__init__(parent)

        # Initialize members
        self.actors = {}  # Map of actor name to vtk actor object
        self._memory_threshold_mb = memory_threshold_mb
        self._quality_level = quality_level
        self._last_render_time = time.time()
        self._render_fps = 30
        self._interaction_quality = "medium"  # Quality during interaction
        self._current_view = "3d"  # Current view mode
        self._enable_orientation_marker = True
        self._enable_axes = True
        self._enable_annotations = True
        self._adaptive_rendering = True
        self._renderer = None
        self._camera_controller = None

        # Set up UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface components."""
        if not PYQT5_AVAILABLE or not VTK_AVAILABLE:
            self._setup_fallback_ui()
            return

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create VTK rendering widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create renderer and render window
        self._renderer = vtkRenderer()
        self._renderer.SetBackground(0.2, 0.2, 0.2)  # Dark gray background
        self._renderer.SetGradientBackground(True)
        self._renderer.SetBackground2(0.0, 0.0, 0.0)  # Black at the top

        # Configure render window
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self._renderer)
        render_window.SetSize(600, 600)
        render_window.SetMultiSamples(4)  # Anti-aliasing for better quality

        # Configure interactor
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor_style = vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(interactor_style)

        # Create camera controller
        self._camera_controller = CameraController(self._renderer)
        self._camera_controller.set_view("3d", animate=False)

        # Add orientation marker
        if self._enable_orientation_marker:
            axes = vtkAxesActor()
            self.marker_widget = vtkOrientationMarkerWidget()
            self.marker_widget.SetOrientationMarker(axes)
            self.marker_widget.SetInteractor(interactor)
            self.marker_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
            self.marker_widget.SetEnabled(1)
            self.marker_widget.InteractiveOff()

        # Add VTK widget to layout
        layout.addWidget(self.vtk_widget)

        # Create view control toolbar
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))

        # Create view buttons
        self._create_view_buttons(toolbar)

        # Add toolbar to layout
        layout.addWidget(toolbar)

        # Initialize the interactor
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        interactor.RemoveObservers("LeftButtonPressEvent")

        # Set up render timer for smoother updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._timed_render)
        self.render_timer.start(1000 // self._render_fps)

    def _setup_fallback_ui(self):
        """Set up fallback UI when required dependencies are not available."""
        layout = QVBoxLayout(self)

        error_label = QLabel(
            "Cannot initialize VTKViewer3D due to missing dependencies."
        )
        error_label.setStyleSheet("color: red; font-weight: bold;")

        dependencies_label = QLabel(
            "Please ensure the following dependencies are installed:\n- VTK\n- PyQt5"
        )

        layout.addWidget(error_label)
        layout.addWidget(dependencies_label)
        layout.addStretch()

    def _create_view_buttons(self, toolbar):
        """Create buttons for different view orientations."""
        # Create view actions
        view_group = QActionGroup(self)

        # 3D view
        view_3d = QAction("3D", view_group)
        view_3d.setCheckable(True)
        view_3d.setChecked(True)
        view_3d.triggered.connect(lambda: self.set_view("3d"))

        # Axial view
        view_axial = QAction("Axial", view_group)
        view_axial.setCheckable(True)
        view_axial.triggered.connect(lambda: self.set_view("axial"))

        # Sagittal view
        view_sagittal = QAction("Sagittal", view_group)
        view_sagittal.setCheckable(True)
        view_sagittal.triggered.connect(lambda: self.set_view("sagittal"))

        # Coronal view
        view_coronal = QAction("Coronal", view_group)
        view_coronal.setCheckable(True)
        view_coronal.triggered.connect(lambda: self.set_view("coronal"))

        # Add actions to toolbar
        toolbar.addAction(view_3d)
        toolbar.addAction(view_axial)
        toolbar.addAction(view_sagittal)
        toolbar.addAction(view_coronal)
        toolbar.addSeparator()

        # Add quality control
        quality_label = QLabel("Quality:")
        toolbar.addWidget(quality_label)

        quality_combo = QComboBox()
        quality_combo.addItems(["Low", "Medium", "High"])
        quality_combo.setCurrentText(self._quality_level.capitalize())
        quality_combo.currentTextChanged.connect(
            lambda text: self.set_quality(text.lower())
        )
        toolbar.addWidget(quality_combo)

    def add_actor(self, name, actor):
        """
        Add a VTK actor to the scene.

        Parameters:
        -----------
        name : str
            Name to identify the actor
        actor : vtkActor
            VTK actor to add
        """
        if not VTK_AVAILABLE or self._renderer is None:
            logger.warning("VTK is not available or renderer is not initialized.")
            return

        if name in self.actors:
            self.remove_actor(name)

        self.actors[name] = actor
        self._renderer.AddActor(actor)
        self.render()

    def remove_actor(self, name):
        """
        Remove a VTK actor from the scene.

        Parameters:
        -----------
        name : str
            Name of the actor to remove
        """
        if not VTK_AVAILABLE or self._renderer is None:
            return

        if name in self.actors:
            self._renderer.RemoveActor(self.actors[name])
            del self.actors[name]
            self.render()

    def clear_scene(self):
        """Remove all actors from the scene."""
        if not VTK_AVAILABLE or self._renderer is None:
            return

        for actor in self.actors.values():
            self._renderer.RemoveActor(actor)

        self.actors.clear()
        self.render()

    def reset_camera(self):
        """Reset the camera to fit all actors in view."""
        if not VTK_AVAILABLE or self._renderer is None:
            return

        self._renderer.ResetCamera()
        self.render()

    def render(self):
        """Render the scene."""
        if not VTK_AVAILABLE or not hasattr(self, "vtk_widget"):
            return

        # Update the camera if needed
        if self._camera_controller and self._camera_controller.update():
            # Camera was updated by animation
            pass

        # Only render at most 30 FPS
        current_time = time.time()
        time_since_last_render = current_time - self._last_render_time

        if time_since_last_render >= 1.0 / self._render_fps:
            self.vtk_widget.GetRenderWindow().Render()
            self._last_render_time = current_time
            self.render_complete.emit()

    def set_view(self, view_name):
        """
        Set the camera to a standard view.

        Parameters:
        -----------
        view_name : str
            View name ("3d", "axial", "sagittal", "coronal")
        """
        if not VTK_AVAILABLE or self._camera_controller is None:
            return

        if self._camera_controller.set_view(view_name):
            self._current_view = view_name
            self.view_changed.emit(view_name)
            self.render()

    def set_quality(self, quality_level):
        """
        Set rendering quality level.

        Parameters:
        -----------
        quality_level : str
            Quality level ("low", "medium", "high")
        """
        if quality_level not in ["low", "medium", "high"]:
            logger.warning(f"Invalid quality level: {quality_level}")
            return

        self._quality_level = quality_level

        # Apply quality settings to renderer
        if VTK_AVAILABLE and self._renderer is not None:
            if quality_level == "low":
                self.vtk_widget.GetRenderWindow().SetMultiSamples(0)
                self._renderer.SetUseFXAA(False)
            elif quality_level == "medium":
                self.vtk_widget.GetRenderWindow().SetMultiSamples(2)
                self._renderer.SetUseFXAA(True)
            else:  # high
                self.vtk_widget.GetRenderWindow().SetMultiSamples(8)
                self._renderer.SetUseFXAA(True)

            self.render()

    def _timed_render(self):
        """Handle timed rendering."""
        if not VTK_AVAILABLE or not hasattr(self, "vtk_widget"):
            return

        # Check if camera animation is in progress
        if self._camera_controller and self._camera_controller.animating:
            self.render()

    def set_background_color(self, color):
        """
        Set background color of the renderer.

        Parameters:
        -----------
        color : tuple
            RGB color tuple (r, g, b) with values 0-1
        """
        if not VTK_AVAILABLE or self._renderer is None:
            return

        self._renderer.SetBackground(*color)
        self.render()

    def add_text_annotation(self, text, position=(10, 10), color=(1, 1, 1)):
        """
        Add text annotation to the viewer.

        Parameters:
        -----------
        text : str
            Text to display
        position : tuple, optional
            Screen position (x, y) in pixels
        color : tuple, optional
            RGB color tuple (r, g, b) with values 0-1
        """
        if not VTK_AVAILABLE or self._renderer is None:
            return

        text_actor = vtkTextActor()
        text_actor.SetInput(text)
        text_actor.SetPosition(position)
        text_actor.GetTextProperty().SetColor(color)
        text_actor.GetTextProperty().SetFontSize(14)

        self.add_actor(f"annotation_{text}", text_actor)

    def set_interaction_mode(self, mode):
        """
        Set interaction mode.

        Parameters:
        -----------
        mode : str
            Interaction mode (e.g., "trackball", "joystick")
        """
        if not VTK_AVAILABLE or not hasattr(self, "vtk_widget"):
            return

        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        if mode == "trackball":
            interactor_style = vtkInteractorStyleTrackballCamera()
        elif mode == "joystick":
            interactor_style = vtk.vtkInteractorStyleJoystickCamera()
        else:
            logger.warning(f"Unsupported interaction mode: {mode}")
            return

        interactor.SetInteractorStyle(interactor_style)

    def get_screenshot(self, width=None, height=None):
        """
        Capture a screenshot of the current view.

        Parameters:
        -----------
        width : int, optional
            Screenshot width in pixels
        height : int, optional
            Screenshot height in pixels

        Returns:
        --------
        numpy.ndarray
            RGB image as a numpy array
        """
        if not VTK_AVAILABLE or not hasattr(self, "vtk_widget"):
            return None

        window = self.vtk_widget.GetRenderWindow()

        # Use current size if not specified
        if width is None or height is None:
            width, height = window.GetSize()

        # Create VTK window to image filter
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(window)
        window_to_image.SetScale(1)
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()
        window_to_image.Update()

        # Convert to numpy array
        vtk_image = window_to_image.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()

        # Create numpy array
        image = np.zeros((height, width, components), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                pixel = vtk_array.GetTuple(i * width + j)
                image[height - i - 1, j, :] = pixel

        return image

    def adapt_quality_to_memory(self):
        """Adapt rendering quality based on available memory."""
        if not VTK_AVAILABLE:
            return

        try:
            # Try to get available memory (platform dependent)
            import psutil

            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)

            # Adjust quality based on available memory
            if available_memory_mb < self._memory_threshold_mb * 0.3:
                self.set_quality("low")
            elif available_memory_mb < self._memory_threshold_mb * 0.7:
                self.set_quality("medium")
            else:
                self.set_quality("high")

        except ImportError:
            # psutil not available, use default quality
            pass


# Test function for standalone testing
if __name__ == "__main__":
    import sys

    # Try to import PyQt5, fallback if not available
    try:
        from PyQt5.QtWidgets import QApplication

        HAS_PYQT_MAIN = True
    except ImportError:
        try:
            from PyQt6.QtWidgets import QApplication

            HAS_PYQT_MAIN = True
        except ImportError:
            # Create minimal fallback for testing
            def app_init(self, argv):
                pass

            def app_exec(self):
                return 0

            QApplication = type(
                "QApplication", (), {"__init__": app_init, "exec_": app_exec}
            )
            HAS_PYQT_MAIN = False
            print("⚠ PyQt not available. Running in test mode only.")

    app = QApplication(sys.argv)

    if HAS_PYQT_MAIN:
        # Create test window
        viewer = VTKViewer3D()
        viewer.resize(800, 600)
        viewer.show()
        viewer.set_view("3d")

        # Create some test actors for visualization
        if VTK_AVAILABLE:
            # Create a sphere
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetRadius(50.0)
            sphere_source.SetThetaResolution(30)
            sphere_source.SetPhiResolution(30)
            sphere_source.Update()

            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputData(sphere_source.GetOutput())

            sphere_actor = vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red

            # Add to viewer
            viewer.add_actor("sphere", sphere_actor)

            # Create a cube
            cube_source = vtk.vtkCubeSource()
            cube_source.SetXLength(80.0)
            cube_source.SetYLength(80.0)
            cube_source.SetZLength(80.0)
            cube_source.SetCenter(100, 0, 0)
            cube_source.Update()

            cube_mapper = vtk.vtkPolyDataMapper()
            cube_mapper.SetInputData(cube_source.GetOutput())

            cube_actor = vtkActor()
            cube_actor.SetMapper(cube_mapper)
            cube_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # Blue

            # Add to viewer
            viewer.add_actor("cube", cube_actor)

            # Add text annotation
            viewer.add_text_annotation("QuangTPS 3D Viewer", (10, 10), (1, 1, 1))

            # Reset camera to show all objects
            viewer.reset_camera()

        # Run the application
        sys.exit(app.exec_())
    else:
        print("✓ VTKViewer3D module test completed without Qt widgets")
        sys.exit(0)
