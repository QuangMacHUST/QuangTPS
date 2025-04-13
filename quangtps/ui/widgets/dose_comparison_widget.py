"""
Dose Comparison Widget

This module provides a widget for comparing dose distributions between plans,
including dose subtraction visualization and gamma analysis.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QComboBox, QGroupBox, QSizePolicy, QFrame,
    QCheckBox, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
    QSplitter, QTabWidget, QScrollArea, QButtonGroup, QRadioButton,
    QGridLayout, QToolBar, QAction, QMenu, QToolButton
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QIcon, QPalette

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.cm as cm

from quangtps.core.plan import Plan
from quangtps.core.dose import DoseGrid
from quangtps.core.logging import get_logger
from quangtps.analysis.gamma_analysis import calculate_gamma_index
from quangtps.common.paths import get_icon_path

logger = get_logger(__name__)


class DoseComparisonWidget(QWidget):
    """
    Widget for comparing dose distributions between plans.
    
    This widget provides views for:
    - Dose subtraction (difference)
    - Gamma analysis
    - Isodose overlays
    """
    
    # Signals
    dose_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the dose comparison widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Initialize state
        self.reference_plan = None
        self.comparison_plan = None
        self.display_mode = "subtraction"  # or "gamma"
        self.current_slice_index = 0
        self.current_orientation = "axial"  # "sagittal", "coronal"
        self.gamma_criteria = {
            "dose_threshold": 3.0,  # percent
            "distance_threshold": 3.0,  # mm
            "local_normalization": True
        }
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # View controls
        view_group = QGroupBox("View")
        view_layout = QHBoxLayout(view_group)
        
        # Orientation selection
        orientation_layout = QHBoxLayout()
        orientation_layout.addWidget(QLabel("Orientation:"))
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Axial", "Sagittal", "Coronal"])
        self.orientation_combo.currentTextChanged.connect(
            lambda text: self._set_orientation(text.lower())
        )
        orientation_layout.addWidget(self.orientation_combo)
        view_layout.addLayout(orientation_layout)
        
        # Slice controls
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("Slice:"))
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(99)  # Will be updated when plans are set
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        slice_layout.addWidget(self.slice_slider)
        
        self.slice_spinbox = QSpinBox()
        self.slice_spinbox.setMinimum(0)
        self.slice_spinbox.setMaximum(99)  # Will be updated when plans are set
        self.slice_spinbox.valueChanged.connect(self._on_slice_spinbox_changed)
        slice_layout.addWidget(self.slice_spinbox)
        
        view_layout.addLayout(slice_layout)
        controls_layout.addWidget(view_group)
        
        # Gamma criteria controls (only shown when gamma display is active)
        self.gamma_group = QGroupBox("Gamma Criteria")
        gamma_layout = QHBoxLayout(self.gamma_group)
        
        # Dose threshold
        dose_layout = QHBoxLayout()
        dose_layout.addWidget(QLabel("Dose:"))
        self.dose_threshold_spinner = QDoubleSpinBox()
        self.dose_threshold_spinner.setMinimum(0.1)
        self.dose_threshold_spinner.setMaximum(10.0)
        self.dose_threshold_spinner.setValue(self.gamma_criteria["dose_threshold"])
        self.dose_threshold_spinner.setSuffix(" %")
        self.dose_threshold_spinner.valueChanged.connect(self._on_gamma_criteria_changed)
        dose_layout.addWidget(self.dose_threshold_spinner)
        gamma_layout.addLayout(dose_layout)
        
        # Distance threshold
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Distance:"))
        self.distance_threshold_spinner = QDoubleSpinBox()
        self.distance_threshold_spinner.setMinimum(0.1)
        self.distance_threshold_spinner.setMaximum(10.0)
        self.distance_threshold_spinner.setValue(self.gamma_criteria["distance_threshold"])
        self.distance_threshold_spinner.setSuffix(" mm")
        self.distance_threshold_spinner.valueChanged.connect(self._on_gamma_criteria_changed)
        distance_layout.addWidget(self.distance_threshold_spinner)
        gamma_layout.addLayout(distance_layout)
        
        # Local/global normalization
        self.local_norm_checkbox = QCheckBox("Local Normalization")
        self.local_norm_checkbox.setChecked(self.gamma_criteria["local_normalization"])
        self.local_norm_checkbox.stateChanged.connect(self._on_gamma_criteria_changed)
        gamma_layout.addWidget(self.local_norm_checkbox)
        
        # Run gamma button
        self.run_gamma_button = QPushButton("Calculate Gamma")
        self.run_gamma_button.clicked.connect(self._calculate_gamma)
        gamma_layout.addWidget(self.run_gamma_button)
        
        controls_layout.addWidget(self.gamma_group)
        
        # Add controls to main layout
        main_layout.addLayout(controls_layout)
        
        # Add display area with matplotlib
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        display_layout = QVBoxLayout()
        display_layout.addWidget(self.toolbar)
        display_layout.addWidget(self.canvas)
        
        main_layout.addLayout(display_layout)
        
        # Set initial state
        self.gamma_group.setVisible(False)
    
    def set_plans(self, reference_plan: Plan, comparison_plan: Plan, display_mode: str = "subtraction"):
        """
        Set the plans to compare.
        
        Args:
            reference_plan: The reference plan
            comparison_plan: The plan to compare against the reference
            display_mode: The display mode (subtraction, gamma)
        """
        self.reference_plan = reference_plan
        self.comparison_plan = comparison_plan
        self.display_mode = display_mode.lower()
        
        # Check dose grids
        if not reference_plan.has_dose():
            logger.warning(f"Reference plan {reference_plan.id} has no dose grid")
            return
            
        if not comparison_plan.has_dose():
            logger.warning(f"Comparison plan {comparison_plan.id} has no dose grid")
            return
        
        # Reset view to axial and center slice
        self.current_orientation = "axial"
        self.orientation_combo.setCurrentText("Axial")
        
        # Calculate dose differences
        self._calculate_dose_difference()
        
        # Set visibility based on display mode
        self.gamma_group.setVisible(self.display_mode == "gamma")
        
        # Set slice range based on the dose grid dimensions
        self._update_slice_range()
        
        # Update the display
        self._update_display()
    
    def _set_orientation(self, orientation: str):
        """
        Set the current orientation.
        
        Args:
            orientation: The new orientation (axial, sagittal, coronal)
        """
        self.current_orientation = orientation.lower()
        self._update_slice_range()
        self._update_display()
    
    def _update_slice_range(self):
        """Update the slice slider and spinbox based on current orientation."""
        if not self.reference_plan or not self.reference_plan.has_dose():
            return
            
        dose_grid = self.reference_plan.get_dose_grid()
        dimensions = dose_grid.data.shape
        
        # Set slider range based on orientation
        if self.current_orientation == "axial":
            max_slice = dimensions[2] - 1
        elif self.current_orientation == "sagittal":
            max_slice = dimensions[0] - 1
        elif self.current_orientation == "coronal":
            max_slice = dimensions[1] - 1
        else:
            max_slice = 0
        
        # Update slider
        self.slice_slider.setMaximum(max_slice)
        self.slice_spinbox.setMaximum(max_slice)
        
        # Reset current slice to middle
        self.current_slice_index = max_slice // 2
        self.slice_slider.setValue(self.current_slice_index)
        self.slice_spinbox.setValue(self.current_slice_index)
    
    def _on_slice_changed(self, value: int):
        """
        Handle slice slider value change.
        
        Args:
            value: New slice index
        """
        self.current_slice_index = value
        self.slice_spinbox.setValue(value)
        self._update_display()
    
    def _on_slice_spinbox_changed(self, value: int):
        """
        Handle slice spinbox value change.
        
        Args:
            value: New slice index
        """
        self.current_slice_index = value
        self.slice_slider.setValue(value)
        self._update_display()
    
    def _on_gamma_criteria_changed(self):
        """Handle changes to gamma criteria."""
        self.gamma_criteria["dose_threshold"] = self.dose_threshold_spinner.value()
        self.gamma_criteria["distance_threshold"] = self.distance_threshold_spinner.value()
        self.gamma_criteria["local_normalization"] = self.local_norm_checkbox.isChecked()
    
    def _calculate_dose_difference(self):
        """Calculate dose difference between the plans."""
        if not self.reference_plan or not self.comparison_plan:
            return
            
        if not self.reference_plan.has_dose() or not self.comparison_plan.has_dose():
            return
        
        # Get dose grids
        ref_dose_grid = self.reference_plan.get_dose_grid()
        comp_dose_grid = self.comparison_plan.get_dose_grid()
        
        # Check if grids are compatible for direct subtraction
        if ref_dose_grid.data.shape != comp_dose_grid.data.shape:
            logger.warning("Dose grids have different dimensions, resampling needed")
            # Implementation of dose grid resampling would be here
            # For now we'll just return
            return
        
        # Calculate difference (comp - ref)
        self.dose_diff = comp_dose_grid.data - ref_dose_grid.data
        
        # Store reference grid info for display
        self.ref_grid_info = {
            "dimensions": ref_dose_grid.data.shape,
            "spacing": ref_dose_grid.voxel_size,
            "origin": ref_dose_grid.origin
        }
    
    def _calculate_gamma(self):
        """Calculate gamma index between the plans."""
        if not self.reference_plan or not self.comparison_plan:
            return
            
        if not self.reference_plan.has_dose() or not self.comparison_plan.has_dose():
            return
        
        # Get dose grids
        ref_dose_grid = self.reference_plan.get_dose_grid()
        comp_dose_grid = self.comparison_plan.get_dose_grid()
        
        # Check if grids are compatible
        if ref_dose_grid.data.shape != comp_dose_grid.data.shape:
            logger.warning("Dose grids have different dimensions, resampling needed")
            return
        
        # Calculate gamma index
        try:
            self.gamma_index = calculate_gamma_index(
                reference_dose=ref_dose_grid.data,
                evaluation_dose=comp_dose_grid.data,
                distance_threshold_mm=self.gamma_criteria["distance_threshold"],
                dose_threshold_percent=self.gamma_criteria["dose_threshold"],
                voxel_size_mm=ref_dose_grid.voxel_size,
                local_normalization=self.gamma_criteria["local_normalization"]
            )
            
            # Update display
            self._update_display()
            
            # Calculate passing rate
            passing_voxels = np.sum(self.gamma_index <= 1.0)
            total_voxels = np.prod(self.gamma_index.shape)
            passing_rate = (passing_voxels / total_voxels) * 100
            
            logger.info(f"Gamma passing rate: {passing_rate:.2f}%")
            
            # Show passing rate in the run button
            self.run_gamma_button.setText(f"Passing Rate: {passing_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"Error calculating gamma index: {str(e)}")
    
    def _update_display(self):
        """Update the display with current data."""
        if not self.reference_plan or not self.comparison_plan:
            self.figure.clear()
            self.canvas.draw()
            return
        
        # Clear figure
        self.figure.clear()
        
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Determine display data based on mode
        if self.display_mode == "subtraction":
            if not hasattr(self, "dose_diff"):
                ax.text(0.5, 0.5, "No dose difference data available", 
                       ha='center', va='center', transform=ax.transAxes)
                self.canvas.draw()
                return
                
            # Get slice data
            if self.current_orientation == "axial":
                slice_data = self.dose_diff[:, :, self.current_slice_index]
                title = f"Axial Dose Difference - Slice {self.current_slice_index}"
            elif self.current_orientation == "sagittal":
                slice_data = self.dose_diff[self.current_slice_index, :, :]
                title = f"Sagittal Dose Difference - Slice {self.current_slice_index}"
            elif self.current_orientation == "coronal":
                slice_data = self.dose_diff[:, self.current_slice_index, :]
                title = f"Coronal Dose Difference - Slice {self.current_slice_index}"
            
            # Transpose for correct orientation
            if self.current_orientation != "axial":
                slice_data = slice_data.T
            
            # Display dose difference
            im = ax.imshow(slice_data, cmap='coolwarm')
            cbar = self.figure.colorbar(im, ax=ax)
            cbar.set_label('Dose Difference (Gy)')
            
            # Set title
            ax.set_title(title)
            
        elif self.display_mode == "gamma":
            if not hasattr(self, "gamma_index"):
                ax.text(0.5, 0.5, "Gamma index not calculated", 
                       ha='center', va='center', transform=ax.transAxes)
                self.canvas.draw()
                return
                
            # Get slice data
            if self.current_orientation == "axial":
                slice_data = self.gamma_index[:, :, self.current_slice_index]
                title = f"Axial Gamma Index - Slice {self.current_slice_index}"
            elif self.current_orientation == "sagittal":
                slice_data = self.gamma_index[self.current_slice_index, :, :]
                title = f"Sagittal Gamma Index - Slice {self.current_slice_index}"
            elif self.current_orientation == "coronal":
                slice_data = self.gamma_index[:, self.current_slice_index, :]
                title = f"Coronal Gamma Index - Slice {self.current_slice_index}"
            
            # Transpose for correct orientation
            if self.current_orientation != "axial":
                slice_data = slice_data.T
            
            # Clip gamma values for better visualization
            display_data = np.clip(slice_data, 0, 2)
            
            # Custom colormap for gamma (red < 1, blue > 1)
            cmap = plt.cm.RdBu_r
            
            # Display gamma index
            im = ax.imshow(display_data, cmap=cmap, vmin=0, vmax=2)
            cbar = self.figure.colorbar(im, ax=ax)
            cbar.set_label('Gamma Index')
            
            # Set title
            criteria_text = f"{self.gamma_criteria['dose_threshold']}%/{self.gamma_criteria['distance_threshold']}mm"
            if self.gamma_criteria["local_normalization"]:
                criteria_text += " (Local)"
            else:
                criteria_text += " (Global)"
                
            ax.set_title(f"{title} - {criteria_text}")
        
        # Draw
        self.canvas.draw()
    
    def refresh(self):
        """Refresh the display."""
        self._update_display()