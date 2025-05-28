#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Visualization Module
======================

Module này cung cấp các widget để hiển thị dữ liệu y tế 3D sử dụng VTK.
Hỗ trợ hiển thị hình ảnh, cấu trúc giải phẫu và phân phối liều.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt
from enum import Enum

logger = logging.getLogger(__name__)


# DisplayMode enum for external_beam_planning_tab compatibility
class DisplayMode(Enum):
    """Display modes cho 3D visualization."""

    STRUCTURE_ONLY = "structure_only"
    DOSE_ONLY = "dose_only"
    COMBINED = "combined"
    OVERLAY = "overlay"


# ViewOrientation enum for external_beam_planning_tab compatibility
class ViewOrientation(Enum):
    """View orientations cho 3D visualization."""

    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    OBLIQUE = "oblique"
    ANTERIOR = "anterior"
    POSTERIOR = "posterior"
    LEFT = "left"
    RIGHT = "right"


# Safe VTK import với fallback mechanism
HAS_VTK = False
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    HAS_VTK = True
    logger.info("Đã khởi tạo thành công các thành phần VTK.")
except ImportError as e:
    logger.warning(f"VTK không khả dụng ({e}). 3D visualization sẽ bị giới hạn.")
    vtk = None
    QVTKRenderWindowInteractor = None


class VTKWidget(QWidget):
    """Widget hiển thị VTK với fallback mechanism."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        if HAS_VTK:
            self._init_vtk()
        else:
            self._init_fallback()

    def _init_vtk(self):
        """Khởi tạo VTK renderer."""
        try:
            layout = QVBoxLayout()

            # Create VTK widget
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            layout.addWidget(self.vtk_widget)

            # Setup renderer
            self.renderer = vtk.vtkRenderer()
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
            self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

            # Set background color
            self.renderer.SetBackground(0.1, 0.1, 0.1)

            # Initialize interactor
            self.interactor.Initialize()

            self.setLayout(layout)
            logger.info("VTKWidget khởi tạo thành công")

        except Exception as e:
            logger.error(f"Lỗi khởi tạo VTK: {e}")
            self._init_fallback()

    def _init_fallback(self):
        """Khởi tạo fallback UI khi VTK không khả dụng."""
        layout = QVBoxLayout()

        label = QLabel("3D Visualization không khả dụng\n(Yêu cầu VTK)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 14px;
                padding: 20px;
                border: 2px dashed #555;
                border-radius: 10px;
            }
        """)

        layout.addWidget(label)
        self.setLayout(layout)


class StructureViewer3D(VTKWidget):
    """Widget hiển thị cấu trúc giải phẫu 3D."""

    structureSelected = pyqtSignal(str)  # structure name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.structures = {}  # name -> actor
        self.structure_colors = {}
        self.structure_visibility = {}

        if HAS_VTK:
            self._setup_camera()

    def _setup_camera(self):
        """Thiết lập camera mặc định."""
        if not HAS_VTK:
            return

        try:
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(0, -500, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            self.renderer.ResetCameraClippingRange()
        except Exception as e:
            logger.error(f"Lỗi thiết lập camera: {e}")

    def add_structure(
        self,
        name: str,
        mask: np.ndarray,
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        opacity: float = 0.7,
    ):
        """Thêm cấu trúc vào scene."""
        if not HAS_VTK:
            logger.warning("VTK không khả dụng, không thể hiển thị cấu trúc")
            return

        try:
            # Remove existing structure if present
            if name in self.structures:
                self.remove_structure(name)

            # Create VTK image data
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(mask.shape)
            vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

            # Convert numpy array to VTK
            points = vtk_data.GetPointData().GetScalars()
            points.SetVoidArray(mask.astype(np.uint8), mask.size, 1)

            # Create contour filter
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(vtk_data)
            contour.SetValue(0, 0.5)
            contour.Update()

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(opacity)

            # Add to renderer
            self.renderer.AddActor(actor)

            # Store references
            self.structures[name] = actor
            self.structure_colors[name] = color
            self.structure_visibility[name] = True

            # Update display
            self.interactor.Render()

            logger.info(f"Đã thêm cấu trúc '{name}' vào 3D viewer")

        except Exception as e:
            logger.error(f"Lỗi thêm cấu trúc '{name}': {e}")

    def remove_structure(self, name: str):
        """Xóa cấu trúc khỏi scene."""
        if not HAS_VTK or name not in self.structures:
            return

        try:
            actor = self.structures[name]
            self.renderer.RemoveActor(actor)

            del self.structures[name]
            del self.structure_colors[name]
            del self.structure_visibility[name]

            self.interactor.Render()
            logger.info(f"Đã xóa cấu trúc '{name}' khỏi 3D viewer")

        except Exception as e:
            logger.error(f"Lỗi xóa cấu trúc '{name}': {e}")

    def set_structure_visibility(self, name: str, visible: bool):
        """Thiết lập hiển thị/ẩn cấu trúc."""
        if not HAS_VTK or name not in self.structures:
            return

        try:
            actor = self.structures[name]
            actor.SetVisibility(visible)
            self.structure_visibility[name] = visible
            self.interactor.Render()

        except Exception as e:
            logger.error(f"Lỗi thiết lập visibility cho '{name}': {e}")

    def set_structure_color(self, name: str, color: Tuple[float, float, float]):
        """Thiết lập màu cấu trúc."""
        if not HAS_VTK or name not in self.structures:
            return

        try:
            actor = self.structures[name]
            actor.GetProperty().SetColor(color)
            self.structure_colors[name] = color
            self.interactor.Render()

        except Exception as e:
            logger.error(f"Lỗi thiết lập màu cho '{name}': {e}")

    def set_structure_opacity(self, name: str, opacity: float):
        """Thiết lập độ trong suốt cấu trúc."""
        if not HAS_VTK or name not in self.structures:
            return

        try:
            actor = self.structures[name]
            actor.GetProperty().SetOpacity(opacity)
            self.interactor.Render()

        except Exception as e:
            logger.error(f"Lỗi thiết lập opacity cho '{name}': {e}")

    def clear_all_structures(self):
        """Xóa tất cả cấu trúc."""
        if not HAS_VTK:
            return

        try:
            for name in list(self.structures.keys()):
                self.remove_structure(name)

        except Exception as e:
            logger.error(f"Lỗi xóa tất cả cấu trúc: {e}")

    def reset_camera(self):
        """Reset camera về vị trí mặc định."""
        if not HAS_VTK:
            return

        try:
            self.renderer.ResetCamera()
            self.interactor.Render()

        except Exception as e:
            logger.error(f"Lỗi reset camera: {e}")


class DoseViewer3D(VTKWidget):
    """Widget hiển thị phân phối liều 3D."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dose_actor = None
        self.isodose_actors = {}

        if HAS_VTK:
            self._setup_camera()

    def _setup_camera(self):
        """Thiết lập camera mặc định."""
        if not HAS_VTK:
            return

        try:
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(0, -500, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            self.renderer.ResetCameraClippingRange()
        except Exception as e:
            logger.error(f"Lỗi thiết lập camera: {e}")

    def set_dose_grid(
        self,
        dose_data: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Thiết lập dữ liệu liều."""
        if not HAS_VTK:
            return

        try:
            # Create VTK image data
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(dose_data.shape)
            vtk_data.SetSpacing(spacing)
            vtk_data.SetOrigin(origin)
            vtk_data.AllocateScalars(vtk.VTK_FLOAT, 1)

            # Convert numpy array to VTK
            points = vtk_data.GetPointData().GetScalars()
            points.SetVoidArray(dose_data.astype(np.float32), dose_data.size, 1)

            self.dose_data = vtk_data
            logger.info("Đã thiết lập dữ liệu liều cho 3D viewer")

        except Exception as e:
            logger.error(f"Lỗi thiết lập dose grid: {e}")

    def add_isodose_surface(
        self,
        dose_level: float,
        color: Tuple[float, float, float] = (1.0, 1.0, 0.0),
        opacity: float = 0.5,
    ):
        """Thêm bề mặt isodose."""
        if not HAS_VTK or not hasattr(self, "dose_data"):
            return

        try:
            # Create contour filter
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(self.dose_data)
            contour.SetValue(0, dose_level)
            contour.Update()

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(opacity)

            # Add to renderer
            self.renderer.AddActor(actor)

            # Store reference
            self.isodose_actors[dose_level] = actor

            # Update display
            self.interactor.Render()

            logger.info(f"Đã thêm isodose surface {dose_level} Gy")

        except Exception as e:
            logger.error(f"Lỗi thêm isodose surface {dose_level}: {e}")

    def remove_isodose_surface(self, dose_level: float):
        """Xóa bề mặt isodose."""
        if not HAS_VTK or dose_level not in self.isodose_actors:
            return

        try:
            actor = self.isodose_actors[dose_level]
            self.renderer.RemoveActor(actor)
            del self.isodose_actors[dose_level]
            self.interactor.Render()

        except Exception as e:
            logger.error(f"Lỗi xóa isodose surface {dose_level}: {e}")

    def clear_isodose_surfaces(self):
        """Xóa tất cả bề mặt isodose."""
        if not HAS_VTK:
            return

        try:
            for dose_level in list(self.isodose_actors.keys()):
                self.remove_isodose_surface(dose_level)

        except Exception as e:
            logger.error(f"Lỗi xóa tất cả isodose surfaces: {e}")


class CombinedViewer3D(VTKWidget):
    """Widget hiển thị kết hợp cấu trúc và liều 3D."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.structure_viewer = StructureViewer3D()
        self.dose_viewer = DoseViewer3D()

        if HAS_VTK:
            # Share the same renderer
            self.structure_viewer.renderer = self.renderer
            self.structure_viewer.interactor = self.interactor
            self.dose_viewer.renderer = self.renderer
            self.dose_viewer.interactor = self.interactor

    def add_structure(
        self,
        name: str,
        mask: np.ndarray,
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        opacity: float = 0.7,
    ):
        """Thêm cấu trúc vào scene."""
        if hasattr(self.structure_viewer, "add_structure"):
            self.structure_viewer.add_structure(name, mask, color, opacity)

    def set_dose_grid(
        self,
        dose_data: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Thiết lập dữ liệu liều."""
        if hasattr(self.dose_viewer, "set_dose_grid"):
            self.dose_viewer.set_dose_grid(dose_data, spacing, origin)

    def add_isodose_surface(
        self,
        dose_level: float,
        color: Tuple[float, float, float] = (1.0, 1.0, 0.0),
        opacity: float = 0.5,
    ):
        """Thêm bề mặt isodose."""
        if hasattr(self.dose_viewer, "add_isodose_surface"):
            self.dose_viewer.add_isodose_surface(dose_level, color, opacity)


# Factory function để tạo 3D viewer phù hợp
def create_3d_viewer(viewer_type: str = "combined", parent=None):
    """
    Tạo 3D viewer phù hợp.

    Args:
        viewer_type: "structure", "dose", hoặc "combined"
        parent: Parent widget

    Returns:
        3D viewer widget
    """
    if viewer_type == "structure":
        return StructureViewer3D(parent)
    elif viewer_type == "dose":
        return DoseViewer3D(parent)
    else:
        return CombinedViewer3D(parent)


# Alias function for compatibility
def create_3d_visualization_widget(parent=None):
    """
    Tạo widget hiển thị 3D tương thích với external_beam_planning_tab.

    Args:
        parent: Parent widget

    Returns:
        3D visualization widget
    """
    return create_3d_viewer("combined", parent)


# Safe VTK import check function
def safe_vtk_import():
    """Kiểm tra VTK có khả dụng không."""
    return HAS_VTK


def check_vtk_class_availability(class_name: str) -> bool:
    """
    Kiểm tra một VTK class có khả dụng không.

    Args:
        class_name: Tên class VTK (ví dụ: 'vtkRenderer')

    Returns:
        True nếu class khả dụng
    """
    if not HAS_VTK:
        return False

    try:
        return hasattr(vtk, class_name)
    except:
        return False
