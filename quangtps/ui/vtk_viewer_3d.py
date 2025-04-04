"""
VTK-based 3D medical imaging viewer for QuangTPS.

This module implements a VTK-based 3D viewer for visualizing medical imaging data
and structures in 3D, similar to Eclipse treatment planning system.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import SimpleITK as sitk

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSplitter, QSlider, QFrame, QGridLayout, QSpinBox, QSizePolicy,
    QToolBar, QAction, QComboBox, QCheckBox, QToolButton, QMenu
)
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QSize

logger = logging.getLogger(__name__)

class VTKViewer3D(QWidget):
    """
    Viewer dùng VTK cho hiển thị dữ liệu 3D y tế.
    
    Cung cấp chức năng hiển thị dữ liệu 3D từ CT, MRI, cấu trúc bề mặt
    và phân phối liều trong không gian 3D.
    """
    
    view_changed = pyqtSignal()  # Emitted when the view changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        
        # Tạo layout và widget VTK
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Thêm toolbar
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(16, 16))
        
        # Các hành động toolbar
        self.action_reset_view = QAction(QIcon("quangtps/ui/icons/new_icons/reset_view.png"), "Đặt lại góc nhìn", self)
        self.action_reset_view.triggered.connect(self.reset_view)
        self.toolbar.addAction(self.action_reset_view)
        
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Volume Rendering", "Surface Rendering", "MIP", "X-Ray"])
        self.view_mode.currentIndexChanged.connect(self.set_rendering_mode)
        self.toolbar.addWidget(QLabel("Mode:"))
        self.toolbar.addWidget(self.view_mode)
        
        self.toolbar.addSeparator()
        
        self.show_structures = QCheckBox("Show Structures")
        self.show_structures.setChecked(True)
        self.show_structures.stateChanged.connect(self.toggle_structures)
        self.toolbar.addWidget(self.show_structures)
        
        self.show_dose = QCheckBox("Show Dose")
        self.show_dose.setChecked(False)
        self.show_dose.stateChanged.connect(self.toggle_dose)
        self.toolbar.addWidget(self.show_dose)
        
        self.main_layout.addWidget(self.toolbar)
        
        # Tạo widget VTK
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.main_layout.addWidget(self.vtk_widget)
        
        # Tạo renderer và camera
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)  # Nền tối
        
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # Setup style
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.style)
        
        # Setup camera
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetViewUp(0, 0, 1)
        self.camera.SetPosition(0, -500, 0)
        self.camera.SetFocalPoint(0, 0, 0)
        
        # Các biến thành viên
        self.volume = None
        self.volume_mapper = vtk.vtkSmartVolumeMapper()
        self.volume_property = vtk.vtkVolumeProperty()
        self.color_function = vtk.vtkColorTransferFunction()
        self.opacity_function = vtk.vtkPiecewiseFunction()
        self.structure_actors = {}
        self.dose_actor = None
        self.rendering_mode = 0  # 0: Volume, 1: Surface, 2: MIP, 3: X-Ray
        
        # Khởi tạo
        self._setup_rendering_pipeline()
        self.renderer.ResetCamera()
        self.interactor.Initialize()
    
    def _setup_rendering_pipeline(self):
        """Thiết lập pipeline cho volume rendering."""
        # Thiết lập thuộc tính volume
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.ShadeOn()
        self.volume_property.SetAmbient(0.1)
        self.volume_property.SetDiffuse(0.9)
        self.volume_property.SetSpecular(0.2)
        self.volume_property.SetSpecularPower(10.0)
        
        # Thiết lập các hàm chuyển đổi mặc định cho CT
        self.setup_ct_transfer_functions()
    
    def setup_ct_transfer_functions(self):
        """Thiết lập hàm chuyển đổi màu và độ trong suốt cho dữ liệu CT."""
        # Clear previous transfer functions
        self.color_function.RemoveAllPoints()
        self.opacity_function.RemoveAllPoints()
        
        # Color function for CT
        self.color_function.AddRGBPoint(-1000, 0.0, 0.0, 0.0)   # Air
        self.color_function.AddRGBPoint(-600, 0.0, 0.0, 0.0)    # Lung
        self.color_function.AddRGBPoint(-400, 0.15, 0.15, 0.15) # Lung tissue
        self.color_function.AddRGBPoint(-100, 0.3, 0.3, 0.45)   # Soft tissue
        self.color_function.AddRGBPoint(40, 0.7, 0.7, 0.7)      # Soft tissue
        self.color_function.AddRGBPoint(400, 1.0, 1.0, 0.9)     # Bone
        self.color_function.AddRGBPoint(3000, 1.0, 1.0, 1.0)    # Bone/metal
        
        # Opacity function for CT
        self.opacity_function.AddPoint(-1000, 0.0)
        self.opacity_function.AddPoint(-600, 0.0)
        self.opacity_function.AddPoint(-400, 0.05)
        self.opacity_function.AddPoint(-100, 0.1)
        self.opacity_function.AddPoint(40, 0.2)
        self.opacity_function.AddPoint(400, 0.3)
        self.opacity_function.AddPoint(3000, 0.3)
        
        # Set to volume property
        self.volume_property.SetColor(self.color_function)
        self.volume_property.SetScalarOpacity(self.opacity_function)
        
        # Update if volume exists
        if self.volume:
            self.vtk_widget.GetRenderWindow().Render()
    
    def setup_mri_transfer_functions(self):
        """Thiết lập hàm chuyển đổi màu và độ trong suốt cho dữ liệu MRI."""
        # Clear previous transfer functions
        self.color_function.RemoveAllPoints()
        self.opacity_function.RemoveAllPoints()
        
        # Color function for MRI (T1/T2)
        self.color_function.AddRGBPoint(0, 0.0, 0.0, 0.0)      # Background
        self.color_function.AddRGBPoint(50, 0.1, 0.1, 0.1)     # Dark tissue
        self.color_function.AddRGBPoint(100, 0.2, 0.2, 0.2)    # Gray matter
        self.color_function.AddRGBPoint(200, 0.4, 0.4, 0.5)    # White matter
        self.color_function.AddRGBPoint(300, 0.7, 0.7, 0.8)    # CSF
        self.color_function.AddRGBPoint(500, 1.0, 1.0, 1.0)    # Bright tissue
        
        # Opacity function for MRI
        self.opacity_function.AddPoint(0, 0.0)
        self.opacity_function.AddPoint(50, 0.05)
        self.opacity_function.AddPoint(100, 0.1)
        self.opacity_function.AddPoint(200, 0.2)
        self.opacity_function.AddPoint(300, 0.3)
        self.opacity_function.AddPoint(500, 0.4)
        
        # Set to volume property
        self.volume_property.SetColor(self.color_function)
        self.volume_property.SetScalarOpacity(self.opacity_function)
        
        # Update if volume exists
        if self.volume:
            self.vtk_widget.GetRenderWindow().Render()
    
    def set_rendering_mode(self, mode_index):
        """Thay đổi chế độ rendering."""
        self.rendering_mode = mode_index
        if not self.volume_mapper:
            return
            
        if mode_index == 0:  # Volume Rendering
            self.volume_mapper.SetBlendModeToComposite()
            self.volume_property.ShadeOn()
        elif mode_index == 1:  # Surface Rendering
            self.volume_mapper.SetBlendModeToComposite()
            self.volume_property.SetAmbient(0.3)
            self.volume_property.SetDiffuse(0.9)
            self.volume_property.SetSpecular(0.2)
            self.volume_property.ShadeOn()
        elif mode_index == 2:  # MIP
            self.volume_mapper.SetBlendModeToMaximumIntensity()
            self.volume_property.ShadeOff()
        elif mode_index == 3:  # X-Ray
            self.volume_mapper.SetBlendModeToAverageIntensity()
            self.volume_property.ShadeOff()
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def set_image_data(self, image_data, spacing=None, origin=None):
        """Thiết lập dữ liệu hình ảnh 3D."""
        if image_data is None:
            return
            
        self.clear_image()
        
        # Create VTK image data
        vtk_image = vtk.vtkImageData()
        
        if isinstance(image_data, sitk.Image):
            # Handle SimpleITK image
            array = sitk.GetArrayFromImage(image_data)
            size = image_data.GetSize()
            vtk_image.SetDimensions(size[0], size[1], size[2])
            vtk_image.SetSpacing(image_data.GetSpacing())
            vtk_image.SetOrigin(image_data.GetOrigin())
        else:
            # Handle numpy array
            array = image_data
            if array.ndim != 3:
                logger.error("Chỉ hỗ trợ dữ liệu 3D")
                return
                
            # Note: VTK uses different ordering than numpy
            vtk_image.SetDimensions(array.shape[2], array.shape[1], array.shape[0])
            
            if spacing:
                vtk_image.SetSpacing(spacing)
            else:
                vtk_image.SetSpacing(1.0, 1.0, 1.0)
                
            if origin:
                vtk_image.SetOrigin(origin)
            else:
                vtk_image.SetOrigin(0.0, 0.0, 0.0)
            
            # Swap axes for VTK ordering
            array = np.swapaxes(array, 0, 2)
        
        # Create scalar data
        flat_array = array.ravel().astype(np.float32)
        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetNumberOfValues(flat_array.size)
        for i, val in enumerate(flat_array):
            vtk_array.SetValue(i, val)
        
        vtk_image.GetPointData().SetScalars(vtk_array)
        
        # Set up volume rendering pipeline
        self.volume_mapper.SetInputData(vtk_image)
        self.volume_mapper.SetRequestedRenderModeToGPU()
        self.volume_mapper.SetSampleDistance(1.0)
        
        # Create volume if it doesn't exist
        if not self.volume:
            self.volume = vtk.vtkVolume()
            self.volume.SetMapper(self.volume_mapper)
            self.volume.SetProperty(self.volume_property)
            self.renderer.AddVolume(self.volume)
        
        # Apply the current rendering mode
        self.set_rendering_mode(self.rendering_mode)
        
        # Reset view
        self.reset_view()
    
    def add_structure(self, structure_id, mask, color=(1.0, 0.0, 0.0), opacity=0.5, name=None):
        """Thêm cấu trúc (contour) vào hiển thị 3D."""
        if structure_id in self.structure_actors:
            self.renderer.RemoveActor(self.structure_actors[structure_id]['actor'])
            del self.structure_actors[structure_id]
        
        if mask is None or not np.any(mask):
            return
        
        # Create VTK image data for the structure mask
        vtk_mask = vtk.vtkImageData()
        vtk_mask.SetDimensions(mask.shape[2], mask.shape[1], mask.shape[0])
        vtk_mask.SetSpacing(1.0, 1.0, 1.0)  # Use appropriate spacing
        vtk_mask.SetOrigin(0.0, 0.0, 0.0)   # Use appropriate origin
        
        # Convert boolean mask to uint8
        mask_array = (mask * 255).astype(np.uint8).ravel()
        vtk_array = vtk.vtkUnsignedCharArray()
        vtk_array.SetNumberOfValues(mask_array.size)
        for i, val in enumerate(mask_array):
            vtk_array.SetValue(i, val)
        
        vtk_mask.GetPointData().SetScalars(vtk_array)
        
        # Create iso-surface using marching cubes
        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(vtk_mask)
        contour.SetValue(0, 127.5)  # Threshold for binary mask
        
        # Smooth the surface
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(15)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()
        
        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        
        # Store structure info
        self.structure_actors[structure_id] = {
            'actor': actor,
            'name': name or structure_id,
            'color': color,
            'opacity': opacity,
            'visible': True
        }
        
        self.renderer.AddActor(actor)
        self.vtk_widget.GetRenderWindow().Render()
    
    def set_structure_visibility(self, structure_id, visible):
        """Thay đổi khả năng hiển thị của cấu trúc."""
        if structure_id in self.structure_actors:
            self.structure_actors[structure_id]['actor'].SetVisibility(visible)
            self.structure_actors[structure_id]['visible'] = visible
            self.vtk_widget.GetRenderWindow().Render()
    
    def set_structure_opacity(self, structure_id, opacity):
        """Thay đổi độ trong suốt của cấu trúc."""
        if structure_id in self.structure_actors:
            self.structure_actors[structure_id]['actor'].GetProperty().SetOpacity(opacity)
            self.structure_actors[structure_id]['opacity'] = opacity
            self.vtk_widget.GetRenderWindow().Render()
    
    def set_structure_color(self, structure_id, color):
        """Thay đổi màu sắc của cấu trúc."""
        if structure_id in self.structure_actors:
            self.structure_actors[structure_id]['actor'].GetProperty().SetColor(color[0], color[1], color[2])
            self.structure_actors[structure_id]['color'] = color
            self.vtk_widget.GetRenderWindow().Render()
    
    def remove_structure(self, structure_id):
        """Xóa cấu trúc khỏi hiển thị 3D."""
        if structure_id in self.structure_actors:
            self.renderer.RemoveActor(self.structure_actors[structure_id]['actor'])
            del self.structure_actors[structure_id]
            self.vtk_widget.GetRenderWindow().Render()
    
    def add_dose(self, dose_grid, isodose_levels=None):
        """Thêm hiển thị phân phối liều."""
        if dose_grid is None or not hasattr(dose_grid, 'data') or dose_grid.data is None:
            return
            
        # Remove existing dose display
        if self.dose_actor:
            self.renderer.RemoveActor(self.dose_actor)
            self.dose_actor = None
            
        # Default isodose levels if not provided
        if isodose_levels is None:
            isodose_levels = [95, 80, 60, 40, 20]
        
        # Create VTK image data
        vtk_dose = vtk.vtkImageData()
        vtk_dose.SetDimensions(dose_grid.data.shape[2], dose_grid.data.shape[1], dose_grid.data.shape[0])
        
        if hasattr(dose_grid, 'spacing'):
            vtk_dose.SetSpacing(dose_grid.spacing)
        else:
            vtk_dose.SetSpacing(1.0, 1.0, 1.0)
            
        if hasattr(dose_grid, 'origin'):
            vtk_dose.SetOrigin(dose_grid.origin)
        else:
            vtk_dose.SetOrigin(0.0, 0.0, 0.0)
        
        # Create scalar data
        flat_array = dose_grid.data.ravel().astype(np.float32)
        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetNumberOfValues(flat_array.size)
        for i, val in enumerate(flat_array):
            vtk_array.SetValue(i, val)
        
        vtk_dose.GetPointData().SetScalars(vtk_array)
        
        # Create isosurfaces for different dose levels
        append = vtk.vtkAppendPolyData()
        
        # Get dose statistics
        dose_max = np.max(dose_grid.data)
        
        for level in isodose_levels:
            # Calculate absolute dose value
            dose_value = dose_max * level / 100.0
            
            # Create isosurface
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(vtk_dose)
            contour.SetValue(0, dose_value)
            contour.Update()
            
            # Add to append filter
            append.AddInputData(contour.GetOutput())
        
        append.Update()
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(append.GetOutputPort())
        mapper.ScalarVisibilityOff()
        
        self.dose_actor = vtk.vtkActor()
        self.dose_actor.SetMapper(mapper)
        self.dose_actor.GetProperty().SetColor(0.8, 0.2, 0.2)  # Red for dose
        self.dose_actor.GetProperty().SetOpacity(0.3)
        self.dose_actor.SetVisibility(self.show_dose.isChecked())
        
        self.renderer.AddActor(self.dose_actor)
        self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_structures(self, state):
        """Bật/tắt hiển thị cấu trúc."""
        for structure_id, structure_info in self.structure_actors.items():
            structure_info['actor'].SetVisibility(state)
        self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_dose(self, state):
        """Bật/tắt hiển thị phân phối liều."""
        if self.dose_actor:
            self.dose_actor.SetVisibility(state)
            self.vtk_widget.GetRenderWindow().Render()
    
    def clear_image(self):
        """Xóa hiển thị hình ảnh."""
        if self.volume:
            self.renderer.RemoveVolume(self.volume)
            self.volume = None
    
    def clear_structures(self):
        """Xóa tất cả cấu trúc khỏi hiển thị 3D."""
        for structure_id, structure_info in self.structure_actors.items():
            self.renderer.RemoveActor(structure_info['actor'])
        self.structure_actors = {}
        self.vtk_widget.GetRenderWindow().Render()
    
    def clear_all(self):
        """Xóa tất cả hiển thị."""
        self.clear_image()
        self.clear_structures()
        if self.dose_actor:
            self.renderer.RemoveActor(self.dose_actor)
            self.dose_actor = None
        self.vtk_widget.GetRenderWindow().Render()
    
    def reset_view(self):
        """Đặt lại góc nhìn camera."""
        if self.renderer:
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
    
    def take_screenshot(self, filename=None):
        """Chụp ảnh màn hình hiển thị 3D."""
        if filename is None:
            import tempfile
            filename = os.path.join(tempfile.gettempdir(), "quangtps_screenshot.png")
            
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.vtk_widget.GetRenderWindow())
        window_to_image.Update()
        
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
        
        return filename

# Test function
def test():
    """Test VTKViewer3D widget with sample data."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create sample image data
    image_size = [100, 100, 100]
    image_data = np.zeros(image_size)
    
    # Add a sphere
    center = np.array(image_size) / 2
    radius = min(image_size) / 4
    
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            for z in range(image_size[2]):
                dist = np.sqrt(((np.array([x, y, z]) - center)**2).sum())
                if dist < radius:
                    image_data[x, y, z] = 1000  # Bone-like value
                else:
                    image_data[x, y, z] = -1000  # Air-like value
    
    # Create sample structure
    structure_mask = np.zeros_like(image_data, dtype=bool)
    structure_radius = radius * 0.8
    
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            for z in range(image_size[2]):
                dist = np.sqrt(((np.array([x, y, z]) - center)**2).sum())
                if dist < structure_radius:
                    structure_mask[x, y, z] = True
    
    # Create and show viewer
    viewer = VTKViewer3D()
    viewer.set_image_data(image_data)
    viewer.add_structure("Test Structure", structure_mask, color=(1.0, 0.0, 0.0), opacity=0.5)
    viewer.show()
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(test()) 