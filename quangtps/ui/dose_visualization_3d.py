#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module DoseVisualization3D cho QuangTPS.

Cung cấp khả năng hiển thị 3D phân phối liều với các tính năng nâng cao
như điều chỉnh isodose, colormap, và vùng hiển thị. Được thiết kế để tích hợp
với tab External Beam Planning.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Sử dụng try/except cho các import không đảm bảo
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    VTK_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import VTK: {e}")
    VTK_AVAILABLE = False

    # Tạo lớp giả cho VTK
    class vtk:
        class vtkMarchingCubes:
            def SetInputData(self, *args):
                pass

            def SetValue(self, *args):
                pass

            def Update(self, *args):
                pass

            def GetOutput(self, *args):
                return None

        class vtkSmoothPolyDataFilter:
            def SetInputData(self, *args):
                pass

            def SetNumberOfIterations(self, *args):
                pass

            def SetRelaxationFactor(self, *args):
                pass

            def FeatureEdgeSmoothingOff(self, *args):
                pass

            def BoundarySmoothingOn(self, *args):
                pass

            def Update(self, *args):
                pass

            def GetOutput(self, *args):
                return None

        class vtkPolyDataMapper:
            def SetInputData(self, *args):
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

        class vtkImageData:
            def SetDimensions(self, *args):
                pass

            def SetSpacing(self, *args):
                pass

            def SetOrigin(self, *args):
                pass

            def AllocateScalars(self, *args):
                pass

            def SetScalarComponentFromFloat(self, *args):
                pass

        class vtkSmartVolumeMapper:
            def SetInputData(self, *args):
                pass

        class vtkVolumeProperty:
            def ShadeOff(self, *args):
                pass

            def SetInterpolationTypeToLinear(self, *args):
                pass

            def SetColor(self, *args):
                pass

            def SetScalarOpacity(self, *args):
                pass

        class vtkColorTransferFunction:
            def AddRGBPoint(self, *args):
                pass

        class vtkPiecewiseFunction:
            def AddPoint(self, *args):
                pass

        class vtkVolume:
            def SetMapper(self, *args):
                pass

            def SetProperty(self, *args):
                pass

        VTK_FLOAT = 10


# Khai báo các class VTK để tránh lỗi linter nếu VTK có sẵn
if VTK_AVAILABLE:
    vtkMarchingCubes = vtk.vtkMarchingCubes
    vtkSmoothPolyDataFilter = vtk.vtkSmoothPolyDataFilter
    vtkPolyDataMapper = vtk.vtkPolyDataMapper
    vtkActor = vtk.vtkActor
    vtkImageData = vtk.vtkImageData
    vtkFloatArray = vtk.vtkFloatArray
    vtkSmartVolumeMapper = vtk.vtkSmartVolumeMapper
    vtkVolumeProperty = vtk.vtkVolumeProperty
    vtkColorTransferFunction = vtk.vtkColorTransferFunction
    vtkPiecewiseFunction = vtk.vtkPiecewiseFunction
    vtkVolume = vtk.vtkVolume
    VTK_FLOAT = vtk.VTK_FLOAT
else:
    vtkMarchingCubes = vtk.vtkMarchingCubes
    vtkSmoothPolyDataFilter = vtk.vtkSmoothPolyDataFilter
    vtkPolyDataMapper = vtk.vtkPolyDataMapper
    vtkActor = vtk.vtkActor
    vtkImageData = vtk.vtkImageData
    vtkFloatArray = type("vtkFloatArray", (), {})
    vtkSmartVolumeMapper = vtk.vtkSmartVolumeMapper
    vtkVolumeProperty = vtk.vtkVolumeProperty
    vtkColorTransferFunction = vtk.vtkColorTransferFunction
    vtkPiecewiseFunction = vtk.vtkPiecewiseFunction
    vtkVolume = vtk.vtkVolume
    VTK_FLOAT = vtk.VTK_FLOAT

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
        QApplication,  # Thêm QApplication cho test standalone
        QDialog,
        QFileDialog,
        QToolBar,
        QAction,
        QActionGroup,
        QDialogButtonBox,
        QInputDialog,
        QTableWidget,
        QTableWidgetItem,
        QProgressBar,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer, QThread, pyqtSlot, QPoint
    from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap, QBrush, QCursor, QPainter

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


# Import các module nội bộ của QuangTPS
try:
    from quangtps.ui.vtk_viewer_3d import VTKViewer3D
    from quangtps.ui.isodose_selector import IsodoseSelector
    from quangtps.ui.structure_visibility_panel import StructureVisibilityPanel
    from quangtps.ui.colormap_selector import ColorMapSelector
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.structures.structure_set import StructureSet

    QUANGTPS_MODULES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import các module QuangTPS: {e}")
    QUANGTPS_MODULES_AVAILABLE = False

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class IsodoseLevel:
    """
    Đại diện cho một mức isodose và thuộc tính hiển thị của nó.

    Attributes:
        level: float
            Mức liều (Gy)
        color: tuple
            Tuple (r, g, b) thể hiện màu sắc
        visible: bool
            Trạng thái hiển thị
    """

    def __init__(self, level, color, visible=True):
        self.level = level
        self.color = color
        self.visible = visible
        self.actor = None  # VTK actor representing this isodose


class DoseVisualization3D(QWidget):
    """
    Enhanced 3D dose visualization component for QuangTPS.

    This widget provides advanced 3D visualization of dose distributions
    with interactive controls for isodose levels, color mapping, and
    transparency. It is designed to be integrated into the External
    Beam Planning tab to provide Eclipse-like visualization capabilities.
    """

    dose_visualization_updated = pyqtSignal()

    # Các chế độ hiển thị
    VOLUME_MODE = "VOLUME"
    SURFACE_MODE = "SURFACE"
    CONTOUR_MODE = "CONTOUR"

    def __init__(self, parent=None):
        """Initialize the 3D dose visualization component."""
        super().__init__(parent)

        # Initialize members
        self.dose_grid = None
        self.prescription_dose = None
        self.isodose_levels = {}  # Dict mapping dose level to IsodoseLevel objects
        self.current_view_mode = "3D"  # "3D", "Axial", "Sagittal", "Coronal"
        self.isodose_actors = {}
        self.structure_actors = {}
        self.structures = []
        self._current_mode = "surface"  # surface, volume, contour
        self._enable_memory_management = True
        self._memory_threshold_mb = 2000  # 2GB
        self._downsampling_enabled = False
        self._adaptive_quality = True

        # Default isodose levels as percentages of prescription dose
        self.default_level_percentages = [100, 95, 90, 80, 70, 50, 30, 20, 10]
        self.default_colors = [
            (1.0, 0.0, 0.0),  # Red - 100%
            (1.0, 0.5, 0.0),  # Orange - 95%
            (1.0, 1.0, 0.0),  # Yellow - 90%
            (0.0, 1.0, 0.0),  # Green - 80%
            (0.0, 1.0, 0.5),  # Teal - 70%
            (0.0, 1.0, 1.0),  # Cyan - 50%
            (0.0, 0.5, 1.0),  # Light Blue - 30%
            (0.0, 0.0, 1.0),  # Blue - 20%
            (0.5, 0.0, 1.0),  # Purple - 10%
        ]

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface components."""
        # Kiểm tra các dependency
        if not PYQT5_AVAILABLE or not VTK_AVAILABLE or not QUANGTPS_MODULES_AVAILABLE:
            self._setup_fallback_ui()
            return

        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create VTK viewer for 3D visualization
        self.vtk_viewer = VTKViewer3D(
            parent=self, memory_threshold_mb=self._memory_threshold_mb
        )
        self.vtk_viewer.setMinimumWidth(600)

        # Create control panel
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_panel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        control_layout = QVBoxLayout(control_panel)

        # Create isodose selector
        self.isodose_selector = IsodoseSelector()
        self.isodose_selector.isodose_levels_changed.connect(
            self.on_isodose_levels_changed
        )

        # Create structure visibility panel
        self.structure_panel = StructureVisibilityPanel()
        self.structure_panel.structure_visibility_changed.connect(
            self.on_structure_visibility_changed
        )

        # Create colormap selector
        self.colormap_selector = ColorMapSelector()
        self.colormap_selector.colormap_changed.connect(self.on_colormap_changed)

        # Create display mode controls
        display_mode_group = QGroupBox("Display Mode")
        display_mode_layout = QVBoxLayout(display_mode_group)

        self.mode_surface = QCheckBox("Surface")
        self.mode_volume = QCheckBox("Volume")
        self.mode_contour = QCheckBox("Contour")

        self.mode_surface.setChecked(self._current_mode == "surface")
        self.mode_volume.setChecked(self._current_mode == "volume")
        self.mode_contour.setChecked(self._current_mode == "contour")

        self.mode_surface.toggled.connect(
            lambda checked: self.set_display_mode("surface") if checked else None
        )
        self.mode_volume.toggled.connect(
            lambda checked: self.set_display_mode("volume") if checked else None
        )
        self.mode_contour.toggled.connect(
            lambda checked: self.set_display_mode("contour") if checked else None
        )

        display_mode_layout.addWidget(self.mode_surface)
        display_mode_layout.addWidget(self.mode_volume)
        display_mode_layout.addWidget(self.mode_contour)

        # Create tabs for organization
        tabs = QTabWidget()

        # Isodose tab
        isodose_tab = QWidget()
        isodose_tab_layout = QVBoxLayout(isodose_tab)
        isodose_tab_layout.addWidget(self.isodose_selector)
        isodose_tab_layout.addStretch()

        # Structures tab
        structures_tab = QWidget()
        structures_tab_layout = QVBoxLayout(structures_tab)
        structures_tab_layout.addWidget(self.structure_panel)
        structures_tab_layout.addStretch()

        # Display tab
        display_tab = QWidget()
        display_tab_layout = QVBoxLayout(display_tab)
        display_tab_layout.addWidget(display_mode_group)
        display_tab_layout.addWidget(self.colormap_selector)
        display_tab_layout.addStretch()

        # Add tabs
        tabs.addTab(isodose_tab, "Isodose")
        tabs.addTab(structures_tab, "Structures")
        tabs.addTab(display_tab, "Display")

        # Add tabs to control panel
        control_layout.addWidget(tabs)

        # Add update button
        self.update_button = QPushButton("Update Visualization")
        self.update_button.clicked.connect(self.update_visualization)
        control_layout.addWidget(self.update_button)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.vtk_viewer)
        splitter.addWidget(control_panel)
        splitter.setStretchFactor(0, 3)  # VTK viewer gets more space
        splitter.setStretchFactor(1, 1)

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # Initialize visualization timer for performance
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_visualization)

    def _setup_fallback_ui(self):
        """Set up fallback UI when required dependencies are not available."""
        layout = QVBoxLayout(self)

        error_label = QLabel(
            "Cannot initialize DoseVisualization3D due to missing dependencies."
        )
        error_label.setStyleSheet("color: red; font-weight: bold;")

        dependencies_label = QLabel(
            "Please ensure the following dependencies are installed:\n"
            "- VTK\n"
            "- PyQt5\n"
            "- QuangTPS modules"
        )

        layout.addWidget(error_label)
        layout.addWidget(dependencies_label)
        layout.addStretch()

    def set_dose_grid(self, dose_grid, prescription_dose=None):
        """
        Set the dose grid to visualize.

        Parameters:
        -----------
        dose_grid : DoseGrid
            The dose grid to visualize
        prescription_dose : float, optional
            The prescription dose in Gy
        """
        if not VTK_AVAILABLE:
            logger.warning("VTK is not available, cannot set dose grid.")
            return

        self.dose_grid = dose_grid

        if prescription_dose is not None:
            self.prescription_dose = prescription_dose
        elif self.dose_grid is not None:
            # Default to maximum dose if no prescription provided
            self.prescription_dose = dose_grid.get_max_dose()

        # Generate default isodose levels
        self.create_default_isodose_levels()

        # Update isodose selector
        if hasattr(self, "isodose_selector"):
            self.isodose_selector.set_prescription_dose(self.prescription_dose)

        # Update visualization
        self.update_visualization()

    def create_default_isodose_levels(self):
        """Create default isodose levels based on prescription dose."""
        if self.prescription_dose is None:
            logger.warning(
                "Cannot create isodose levels: prescription dose is not set."
            )
            return

        self.isodose_levels.clear()

        for i, percentage in enumerate(self.default_level_percentages):
            if i < len(self.default_colors):
                level_dose = self.prescription_dose * percentage / 100.0
                color = self.default_colors[i]

                self.isodose_levels[level_dose] = IsodoseLevel(level_dose, color)

        # Update isodose selector with new levels
        if hasattr(self, "isodose_selector"):
            self.isodose_selector.set_isodose_levels(self.isodose_levels)

    def set_structures(self, structures):
        """
        Set the structures to visualize.

        Parameters:
        -----------
        structures : list
            List of structure objects to visualize
        """
        if not VTK_AVAILABLE:
            logger.warning("VTK is not available, cannot set structures.")
            return

        self.structures = structures

        # Update structure panel
        if hasattr(self, "structure_panel"):
            self.structure_panel.set_structures(structures)

        # Update visualization
        self.update_visualization()

    def update_visualization(self):
        """Update the 3D visualization with current settings."""
        if not VTK_AVAILABLE or not hasattr(self, "vtk_viewer"):
            logger.warning("VTK is not available or VTK viewer is not initialized.")
            return

        if self.dose_grid is None:
            logger.warning("No dose grid to visualize.")
            return

        logger.info("Updating dose visualization...")

        try:
            # Clear existing visualization
            self.vtk_viewer.clear_scene()

            # Get visible isodose levels
            visible_isodose_levels = [
                level for level in self.isodose_levels.values() if level.visible
            ]

            # Create visualization based on current mode
            if self._current_mode == "surface":
                self._create_isodose_surfaces(visible_isodose_levels)
            elif self._current_mode == "volume":
                self._create_isodose_volume()
            elif self._current_mode == "contour":
                self._create_isodose_contours(visible_isodose_levels)

            # Create structure visualization if structures are available
            if self.structures and hasattr(self, "structure_panel"):
                visible_structures = [
                    s
                    for s in self.structures
                    if self.structure_panel.is_structure_visible(s)
                ]
                self._create_structure_models(visible_structures)

            # Refresh the view
            self.vtk_viewer.reset_camera()
            self.vtk_viewer.render()

            # Emit signal that visualization was updated
            self.dose_visualization_updated.emit()

            logger.info("Dose visualization updated successfully.")

        except Exception as e:
            logger.error(f"Error updating dose visualization: {e}")

    def _create_isodose_surfaces(self, isodose_levels):
        """Create isodose surfaces for the given levels."""
        if not isodose_levels:
            logger.info("No visible isodose levels to display.")
            return

        logger.info(f"Creating isodose surfaces for {len(isodose_levels)} levels.")

        try:
            # Convert dose grid to VTK image data
            dose_vtk_data = self._convert_dose_grid_to_vtk()

            if dose_vtk_data is None:
                logger.error("Failed to convert dose grid to VTK data.")
                return

            # Create isosurface for each level
            for level in isodose_levels:
                iso_value = level.level
                color = level.color

                # Create contour filter
                contour = vtkMarchingCubes()
                contour.SetInputData(dose_vtk_data)
                contour.SetValue(0, iso_value)
                contour.Update()

                # Apply smoothing to get better visual quality
                smoother = vtkSmoothPolyDataFilter()
                smoother.SetInputData(contour.GetOutput())
                smoother.SetNumberOfIterations(15)
                smoother.SetRelaxationFactor(0.1)
                smoother.FeatureEdgeSmoothingOff()
                smoother.BoundarySmoothingOn()
                smoother.Update()

                # Create mapper
                mapper = vtkPolyDataMapper()
                mapper.SetInputData(smoother.GetOutput())

                # Create actor
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(color)
                actor.GetProperty().SetOpacity(0.7)  # Semi-transparent

                # Store actor reference
                level.actor = actor
                self.isodose_actors[iso_value] = actor

                # Add to viewer
                self.vtk_viewer.add_actor(str(iso_value), actor)

                logger.debug(f"Created isodose surface for level {iso_value} Gy")

        except Exception as e:
            logger.error(f"Error creating isodose surfaces: {e}")

    def _convert_dose_grid_to_vtk(self):
        """Convert dose grid to VTK image data."""
        if not VTK_AVAILABLE or self.dose_grid is None:
            return None

        try:
            # Get dose grid data
            dose_array = self.dose_grid.get_dose_array()
            spacing = self.dose_grid.get_spacing()
            origin = self.dose_grid.get_origin()

            # Create VTK image data
            vtk_image = vtkImageData()
            vtk_image.SetDimensions(dose_array.shape)
            vtk_image.SetSpacing(spacing)
            vtk_image.SetOrigin(origin)
            vtk_image.AllocateScalars(VTK_FLOAT, 1)

            # Copy dose data to VTK image
            for i in range(dose_array.shape[0]):
                for j in range(dose_array.shape[1]):
                    for k in range(dose_array.shape[2]):
                        vtk_image.SetScalarComponentFromFloat(
                            i, j, k, 0, dose_array[i, j, k]
                        )

            return vtk_image

        except Exception as e:
            logger.error(f"Error converting dose grid to VTK: {e}")
            return None

    def _create_isodose_volume(self):
        """Create volume rendering of dose distribution."""
        if not VTK_AVAILABLE or self.dose_grid is None:
            return

        try:
            # Convert dose grid to VTK image data
            dose_vtk_data = self._convert_dose_grid_to_vtk()

            if dose_vtk_data is None:
                return

            # Setup volume mapper
            mapper = vtkSmartVolumeMapper()
            mapper.SetInputData(dose_vtk_data)

            # Setup volume properties
            volume_property = vtkVolumeProperty()
            volume_property.ShadeOff()
            volume_property.SetInterpolationTypeToLinear()

            # Setup color and opacity transfer functions
            color_tf = vtkColorTransferFunction()
            opacity_tf = vtkPiecewiseFunction()

            # Get dose range
            max_dose = self.dose_grid.get_max_dose()

            # Set up transfer functions
            # Add transfer function points for each isodose level
            for level in sorted(self.isodose_levels.keys()):
                isodose = self.isodose_levels[level]
                if isodose.visible:
                    color_tf.AddRGBPoint(level, *isodose.color)
                    opacity_tf.AddPoint(level, 0.7 * (level / max_dose))

            # Ensure we have at least entry for 0 dose
            color_tf.AddRGBPoint(0, 0.0, 0.0, 0.0)  # Black for 0 dose
            opacity_tf.AddPoint(0, 0.0)  # Transparent for 0 dose

            # Set transfer functions
            volume_property.SetColor(color_tf)
            volume_property.SetScalarOpacity(opacity_tf)

            # Create volume
            volume = vtkVolume()
            volume.SetMapper(mapper)
            volume.SetProperty(volume_property)

            # Add to viewer
            self.vtk_viewer.add_actor("dose_volume", volume)

            logger.info("Created dose volume visualization.")

        except Exception as e:
            logger.error(f"Error creating dose volume: {e}")

    def _create_isodose_contours(self, isodose_levels):
        """Create isodose contours for each orthogonal plane."""
        if not isodose_levels or self.dose_grid is None:
            return

        logger.info("Creating isodose contours is not fully implemented yet.")
        # This would require extracting contours from each plane
        # Implementation would depend on specific requirements

    def _create_structure_models(self, structures):
        """Create 3D models for the given structures."""
        if not structures:
            return

        logger.info(f"Creating structure models for {len(structures)} structures.")

        # Actual implementation would depend on structure representation
        # Placeholder for future implementation

    # Event handlers
    def on_isodose_levels_changed(self, isodose_levels):
        """Handle changes to isodose levels from the selector."""
        self.isodose_levels = isodose_levels
        self.update_timer.start(200)  # Debounce updates

    def on_structure_visibility_changed(self):
        """Handle changes to structure visibility."""
        self.update_timer.start(200)  # Debounce updates

    def on_colormap_changed(self):
        """Handle changes to colormap."""
        self.update_timer.start(200)  # Debounce updates

    def set_display_mode(self, mode):
        """Set the display mode (surface, volume, contour)."""
        if mode not in ["surface", "volume", "contour"]:
            logger.warning(f"Invalid display mode: {mode}")
            return

        logger.info(f"Setting display mode to {mode}")
        self._current_mode = mode

        # Update mode checkboxes
        if hasattr(self, "mode_surface"):
            self.mode_surface.setChecked(mode == "surface")
        if hasattr(self, "mode_volume"):
            self.mode_volume.setChecked(mode == "volume")
        if hasattr(self, "mode_contour"):
            self.mode_contour.setChecked(mode == "contour")

        # Update visualization
        self.update_visualization()


# For standalone testing
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # Create test window
    viewer = DoseVisualization3D()
    viewer.resize(1200, 800)
    viewer.show()

    # Run application
    sys.exit(app.exec_())
