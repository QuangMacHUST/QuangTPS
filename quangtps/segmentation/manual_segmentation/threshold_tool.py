#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Threshold-based Contouring Tool Module
======================================

This module provides Eclipse-like threshold-based contouring tools for QuangTPS.
"""

import numpy as np
import logging
from enum import Enum
from collections import deque
from typing import List, Tuple, Dict, Optional, Any, Union

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QComboBox, QGroupBox, QRadioButton,
    QButtonGroup, QSpinBox, QCheckBox, QGridLayout,
    QFormLayout, QSizePolicy
)
from PyQt5.QtGui import QIcon, QColor, QCursor

logger = logging.getLogger(__name__)

class ThresholdOperation(Enum):
    """Enum for the different threshold operations."""
    INSIDE_RANGE = 1    # Include values inside the range
    OUTSIDE_RANGE = 2   # Include values outside the range
    ABOVE_VALUE = 3     # Include values above a threshold
    BELOW_VALUE = 4     # Include values below a threshold

class ThresholdMode(Enum):
    """Enum for the different threshold modes."""
    PREVIEW = 1     # Show preview of threshold
    DRAW = 2        # Draw threshold to create contour
    EDIT = 3        # Edit existing contour with threshold

class ThresholdContourTool:
    """
    Tool for creating contours based on image intensity thresholds.
    
    This class implements threshold-based segmentation for creating and editing
    contours based on image intensity values, similar to Eclipse's threshold tool.
    """
    
    def __init__(self, mode=ThresholdMode.PREVIEW, operation=ThresholdOperation.INSIDE_RANGE):
        """Initialize the threshold contouring tool."""
        self.mode = mode
        self.operation = operation
        self.image_data = None
        self.slice_index = None
        self.orientation = None
        self.lower_threshold = 0
        self.upper_threshold = 100
        self.seed_point = None
        self.preview_mask = None
        self.structure = None
        self.region_growing_enabled = True
        self.smooth_contours = True
        self.use_3d_threshold = False
        self.preview_opacity = 0.5
        self.preview_color = QColor(0, 255, 255, 128)  # Cyan with alpha
    
    def set_image_data(self, image_data):
        """Set the image data for thresholding."""
        self.image_data = image_data
        
        # Automatically set reasonable threshold values based on image data range
        if image_data is not None and hasattr(image_data, 'data') and image_data.data is not None:
            min_val = np.min(image_data.data)
            max_val = np.max(image_data.data)
            
            # Set thresholds to a reasonable range within the data
            self.lower_threshold = min_val + (max_val - min_val) * 0.25
            self.upper_threshold = min_val + (max_val - min_val) * 0.75
    
    def set_seed_point(self, point, slice_index, orientation):
        """Set the seed point for region growing."""
        self.seed_point = (point[0], point[1])
        self.slice_index = slice_index
        self.orientation = orientation
        
        # Calculate the threshold mask
        if self.mode == ThresholdMode.PREVIEW:
            self._calculate_threshold_mask()
    
    def set_thresholds(self, lower_threshold, upper_threshold):
        """Set the lower and upper threshold values."""
        if self.lower_threshold != lower_threshold or self.upper_threshold != upper_threshold:
            self.lower_threshold = lower_threshold
            self.upper_threshold = upper_threshold
            
            # Recalculate the threshold mask for preview
            if self.mode == ThresholdMode.PREVIEW and self.seed_point is not None:
                self._calculate_threshold_mask()
    
    def set_operation(self, operation):
        """Set the threshold operation."""
        if self.operation != operation:
            self.operation = operation
            
            # Recalculate the threshold mask for preview
            if self.mode == ThresholdMode.PREVIEW and self.seed_point is not None:
                self._calculate_threshold_mask()
    
    def set_mode(self, mode):
        """Set the thresholding mode."""
        if self.mode != mode:
            self.mode = mode
            
            # Calculate or clear preview mask as needed
            if mode == ThresholdMode.PREVIEW and self.seed_point is not None:
                self._calculate_threshold_mask()
            else:
                self.preview_mask = None
    
    def get_preview_mask(self):
        """Get the preview mask for display."""
        return self.preview_mask
    
    def apply_threshold(self):
        """Apply the current threshold to create or edit a contour."""
        if self.structure is None or self.seed_point is None or self.image_data is None:
            return None
        
        # Calculate the threshold mask if not already done
        if self.preview_mask is None:
            self._calculate_threshold_mask()
            
        if self.preview_mask is None:
            return None
            
        # Create contour points from the mask
        contour_points = []
        
        # Extract contours from the mask using OpenCV if available
        try:
            import cv2
            
            # Convert mask to 8-bit unsigned integer format for OpenCV
            mask_uint8 = (self.preview_mask * 255).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Convert contours to points format expected by structure
            for contour in contours:
                # Simplify contour if smooth_contours is enabled
                if self.smooth_contours and len(contour) > 10:
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    points = [tuple(point[0]) for point in approx]
                else:
                    points = [tuple(point[0]) for point in contour]
                
                # Only add contours with at least 3 points
                if len(points) >= 3:
                    contour_points.append(points)
                    
        except ImportError:
            logger.warning("OpenCV not available. Using simple contour extraction.")
            
            # Simple contour extraction using connected components
            from scipy import ndimage
            
            # Label connected components
            labeled_mask, num_features = ndimage.label(self.preview_mask)
            
            for i in range(1, num_features + 1):
                # Create binary mask for this component
                component_mask = (labeled_mask == i)
                
                # Extract boundary points (simple approach)
                y_indices, x_indices = np.where(component_mask)
                
                if len(x_indices) > 0:
                    # Create a simple outline by finding boundary pixels
                    # This is a simplified approach and won't create optimal contours
                    boundary_mask = component_mask & (~ndimage.binary_erosion(component_mask))
                    bound_y, bound_x = np.where(boundary_mask)
                    
                    # Order boundary points (simple approach, not perfect)
                    if len(bound_x) >= 3:
                        ordered_points = []
                        current_x, current_y = bound_x[0], bound_y[0]
                        ordered_points.append((int(current_x), int(current_y)))
                        
                        # Mark this point as visited
                        visited = set([(current_x, current_y)])
                        
                        # Iterate until we've added all boundary points or can't find more
                        while len(ordered_points) < len(bound_x):
                            # Find closest unvisited point
                            min_dist = float('inf')
                            next_point = None
                            
                            for i in range(len(bound_x)):
                                x, y = bound_x[i], bound_y[i]
                                if (x, y) not in visited:
                                    dist = (x - current_x) ** 2 + (y - current_y) ** 2
                                    if dist < min_dist:
                                        min_dist = dist
                                        next_point = (int(x), int(y))
                            
                            if next_point is None:
                                break
                                
                            ordered_points.append(next_point)
                            visited.add(next_point)
                            current_x, current_y = next_point
                        
                        contour_points.append(ordered_points)
        
        return contour_points, self.slice_index, self.orientation
    
    def get_cursor(self):
        """Get the cursor for the threshold tool."""
        return QCursor(Qt.CrossCursor)
    
    def _calculate_threshold_mask(self):
        """Calculate the threshold mask based on current settings."""
        if self.image_data is None or self.seed_point is None:
            self.preview_mask = None
            return
        
        try:
            # Extract the slice based on orientation
            if self.orientation == "axial":
                slice_data = self.image_data[self.slice_index, :, :]
            elif self.orientation == "sagittal":
                slice_data = self.image_data[:, :, self.slice_index]
                slice_data = slice_data.transpose()
            elif self.orientation == "coronal":
                slice_data = self.image_data[:, self.slice_index, :]
            else:
                # Default to axial if orientation is not specified
                slice_data = self.image_data[self.slice_index, :, :]
            
            # Create binary mask based on threshold operation
            if self.operation == ThresholdOperation.INSIDE_RANGE:
                mask = (slice_data >= self.lower_threshold) & (slice_data <= self.upper_threshold)
            elif self.operation == ThresholdOperation.OUTSIDE_RANGE:
                mask = (slice_data < self.lower_threshold) | (slice_data > self.upper_threshold)
            elif self.operation == ThresholdOperation.ABOVE_VALUE:
                mask = (slice_data >= self.lower_threshold)
            elif self.operation == ThresholdOperation.BELOW_VALUE:
                mask = (slice_data <= self.upper_threshold)
            else:
                mask = np.zeros_like(slice_data, dtype=bool)
            
            # Apply region growing if enabled
            if self.region_growing_enabled and self.seed_point is not None:
                seed_x, seed_y = int(self.seed_point[0]), int(self.seed_point[1])
                
                # Check if seed point is within image bounds
                if 0 <= seed_x < mask.shape[1] and 0 <= seed_y < mask.shape[0]:
                    # Check if seed point satisfies threshold
                    if mask[seed_y, seed_x]:
                        # Perform region growing from seed point
                        region_mask = self._region_growing(seed_y, seed_x, mask)
                        mask = region_mask
                    else:
                        # Seed point doesn't satisfy threshold
                        mask = np.zeros_like(mask, dtype=bool)
            
            # Store mask for preview
            self.preview_mask = mask.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error calculating threshold mask: {e}")
            self.preview_mask = None
    
    def _region_growing(self, seed_y, seed_x, threshold_mask=None):
        """Perform region growing from a seed point."""
        if threshold_mask is None:
            return np.zeros_like(self.image_data[0], dtype=bool)
        
        # Create a mask the same size as the threshold mask
        visited = np.zeros_like(threshold_mask, dtype=bool)
        result_mask = np.zeros_like(threshold_mask, dtype=bool)
        
        # Queue for breadth-first search
        queue = deque([(seed_y, seed_x)])
        visited[seed_y, seed_x] = True
        result_mask[seed_y, seed_x] = True
        
        # Define 4-connected neighborhood
        dy = [-1, 0, 1, 0]  # up, right, down, left
        dx = [0, 1, 0, -1]
        
        height, width = threshold_mask.shape
        
        # Perform breadth-first region growing
        while queue:
            y, x = queue.popleft()
            
            for i in range(4):
                ny, nx = y + dy[i], x + dx[i]
                
                # Check if within bounds
                if 0 <= ny < height and 0 <= nx < width:
                    # Check if unvisited and meets threshold criteria
                    if not visited[ny, nx] and threshold_mask[ny, nx]:
                        visited[ny, nx] = True
                        result_mask[ny, nx] = True
                        queue.append((ny, nx))
        
        return result_mask

class ThresholdToolWidget(QWidget):
    """
    Widget for configuring threshold contouring tool.
    
    This class provides a UI for configuring the threshold contouring tool,
    allowing the user to set threshold values and operations.
    """
    
    # Signals
    toolChanged = pyqtSignal(dict)  # Emitted when tool settings change
    applyThreshold = pyqtSignal()   # Emitted when the apply button is clicked
    
    def __init__(self, parent=None):
        """Initialize the threshold tool widget."""
        super().__init__(parent)
        
        # Default values
        self.min_value = -1000
        self.max_value = 3000
        
        # Setup UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Threshold operation selection
        operation_group = QGroupBox("Threshold Operation")
        operation_layout = QVBoxLayout(operation_group)
        
        self.operation_buttons = QButtonGroup(self)
        
        self.inside_range_btn = QRadioButton("Inside Range")
        self.inside_range_btn.setChecked(True)
        self.operation_buttons.addButton(self.inside_range_btn, ThresholdOperation.INSIDE_RANGE.value)
        operation_layout.addWidget(self.inside_range_btn)
        
        self.outside_range_btn = QRadioButton("Outside Range")
        self.operation_buttons.addButton(self.outside_range_btn, ThresholdOperation.OUTSIDE_RANGE.value)
        operation_layout.addWidget(self.outside_range_btn)
        
        self.above_value_btn = QRadioButton("Above Value")
        self.operation_buttons.addButton(self.above_value_btn, ThresholdOperation.ABOVE_VALUE.value)
        operation_layout.addWidget(self.above_value_btn)
        
        self.below_value_btn = QRadioButton("Below Value")
        self.operation_buttons.addButton(self.below_value_btn, ThresholdOperation.BELOW_VALUE.value)
        operation_layout.addWidget(self.below_value_btn)
        
        self.operation_buttons.buttonClicked.connect(self._on_operation_changed)
        
        main_layout.addWidget(operation_group)
        
        # Threshold values
        threshold_group = QGroupBox("Threshold Values")
        threshold_layout = QGridLayout(threshold_group)
        
        # Lower threshold
        threshold_layout.addWidget(QLabel("Lower:"), 0, 0)
        self.lower_threshold_spin = QSpinBox()
        self.lower_threshold_spin.setRange(self.min_value, self.max_value)
        self.lower_threshold_spin.setValue(0)
        self.lower_threshold_spin.valueChanged.connect(self.on_lower_threshold_changed)
        threshold_layout.addWidget(self.lower_threshold_spin, 0, 1)
        
        self.lower_threshold_slider = QSlider(Qt.Horizontal)
        self.lower_threshold_slider.setRange(self.min_value, self.max_value)
        self.lower_threshold_slider.setValue(0)
        self.lower_threshold_slider.valueChanged.connect(self.lower_threshold_spin.setValue)
        threshold_layout.addWidget(self.lower_threshold_slider, 1, 0, 1, 2)
        
        # Upper threshold
        threshold_layout.addWidget(QLabel("Upper:"), 2, 0)
        self.upper_threshold_spin = QSpinBox()
        self.upper_threshold_spin.setRange(self.min_value, self.max_value)
        self.upper_threshold_spin.setValue(100)
        self.upper_threshold_spin.valueChanged.connect(self.on_upper_threshold_changed)
        threshold_layout.addWidget(self.upper_threshold_spin, 2, 1)
        
        self.upper_threshold_slider = QSlider(Qt.Horizontal)
        self.upper_threshold_slider.setRange(self.min_value, self.max_value)
        self.upper_threshold_slider.setValue(100)
        self.upper_threshold_slider.valueChanged.connect(self.upper_threshold_spin.setValue)
        threshold_layout.addWidget(self.upper_threshold_slider, 3, 0, 1, 2)
        
        main_layout.addWidget(threshold_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self.region_growing_check = QCheckBox("Use Region Growing")
        self.region_growing_check.setChecked(True)
        self.region_growing_check.stateChanged.connect(self._on_settings_changed)
        options_layout.addWidget(self.region_growing_check)
        
        self.smooth_contours_check = QCheckBox("Smooth Contours")
        self.smooth_contours_check.setChecked(True)
        self.smooth_contours_check.stateChanged.connect(self._on_settings_changed)
        options_layout.addWidget(self.smooth_contours_check)
        
        self.preview_check = QCheckBox("Show Preview")
        self.preview_check.setChecked(True)
        self.preview_check.stateChanged.connect(self._on_preview_changed)
        options_layout.addWidget(self.preview_check)
        
        main_layout.addWidget(options_group)
        
        # Mode selection
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_buttons = QButtonGroup(self)
        
        self.preview_mode_btn = QRadioButton("Preview")
        self.preview_mode_btn.setChecked(True)
        self.mode_buttons.addButton(self.preview_mode_btn, ThresholdMode.PREVIEW.value)
        mode_layout.addWidget(self.preview_mode_btn)
        
        self.draw_mode_btn = QRadioButton("Draw")
        self.mode_buttons.addButton(self.draw_mode_btn, ThresholdMode.DRAW.value)
        mode_layout.addWidget(self.draw_mode_btn)
        
        self.edit_mode_btn = QRadioButton("Edit")
        self.mode_buttons.addButton(self.edit_mode_btn, ThresholdMode.EDIT.value)
        mode_layout.addWidget(self.edit_mode_btn)
        
        self.mode_buttons.buttonClicked.connect(self._on_mode_changed)
        
        main_layout.addWidget(mode_group)
        
        # Apply button
        self.apply_button = QPushButton("Apply Threshold")
        self.apply_button.clicked.connect(self.on_apply_threshold)
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #2070c0;
                color: white;
                border-radius: 3px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3080d0;
            }
        """)
        main_layout.addWidget(self.apply_button)
        
        # Add stretch to push everything to the top
        main_layout.addStretch(1)
        
        # Apply Eclipse-like styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QRadioButton, QCheckBox {
                spacing: 5px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #cccccc;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #2070c0;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
    
    def set_operation(self, operation):
        """Set the threshold operation."""
        if operation == ThresholdOperation.INSIDE_RANGE:
            self.inside_range_btn.setChecked(True)
        elif operation == ThresholdOperation.OUTSIDE_RANGE:
            self.outside_range_btn.setChecked(True)
        elif operation == ThresholdOperation.ABOVE_VALUE:
            self.above_value_btn.setChecked(True)
        elif operation == ThresholdOperation.BELOW_VALUE:
            self.below_value_btn.setChecked(True)
        
        # Update UI state based on operation
        self._update_ui_for_operation(operation)
        
        # Emit settings changed signal
        self._on_settings_changed()
    
    def set_mode(self, mode):
        """Set the thresholding mode."""
        if mode == ThresholdMode.PREVIEW:
            self.preview_mode_btn.setChecked(True)
        elif mode == ThresholdMode.DRAW:
            self.draw_mode_btn.setChecked(True)
        elif mode == ThresholdMode.EDIT:
            self.edit_mode_btn.setChecked(True)
        
        # Emit settings changed signal
        self._on_mode_changed()
    
    def _update_ui_for_operation(self, operation):
        """Update UI elements based on the selected operation."""
        # Show/hide threshold controls based on operation
        if operation == ThresholdOperation.INSIDE_RANGE or operation == ThresholdOperation.OUTSIDE_RANGE:
            # Both thresholds needed
            self.lower_threshold_spin.setEnabled(True)
            self.lower_threshold_slider.setEnabled(True)
            self.upper_threshold_spin.setEnabled(True)
            self.upper_threshold_slider.setEnabled(True)
        elif operation == ThresholdOperation.ABOVE_VALUE:
            # Only lower threshold needed
            self.lower_threshold_spin.setEnabled(True)
            self.lower_threshold_slider.setEnabled(True)
            self.upper_threshold_spin.setEnabled(False)
            self.upper_threshold_slider.setEnabled(False)
        elif operation == ThresholdOperation.BELOW_VALUE:
            # Only upper threshold needed
            self.lower_threshold_spin.setEnabled(False)
            self.lower_threshold_slider.setEnabled(False)
            self.upper_threshold_spin.setEnabled(True)
            self.upper_threshold_slider.setEnabled(True)
    
    def on_lower_threshold_changed(self, value):
        """Handle changes to the lower threshold."""
        # Ensure lower threshold <= upper threshold
        if value > self.upper_threshold_spin.value():
            self.upper_threshold_spin.setValue(value)
            self.upper_threshold_slider.setValue(value)
        
        self.lower_threshold_slider.setValue(value)
        self._on_settings_changed()
    
    def on_upper_threshold_changed(self, value):
        """Handle changes to the upper threshold."""
        # Ensure upper threshold >= lower threshold
        if value < self.lower_threshold_spin.value():
            self.lower_threshold_spin.setValue(value)
            self.lower_threshold_slider.setValue(value)
        
        self.upper_threshold_slider.setValue(value)
        self._on_settings_changed()
    
    def on_apply_threshold(self):
        """Handle apply threshold button click."""
        self.applyThreshold.emit()
    
    def get_options(self):
        """Get the current tool options."""
        # Get the selected operation
        operation_id = self.operation_buttons.checkedId()
        operation = ThresholdOperation(operation_id) if operation_id > 0 else ThresholdOperation.INSIDE_RANGE
        
        # Get the selected mode
        mode_id = self.mode_buttons.checkedId()
        mode = ThresholdMode(mode_id) if mode_id > 0 else ThresholdMode.PREVIEW
        
        # Compile options
        options = {
            'operation': operation,
            'mode': mode,
            'lower_threshold': self.lower_threshold_spin.value(),
            'upper_threshold': self.upper_threshold_spin.value(),
            'region_growing': self.region_growing_check.isChecked(),
            'smooth_contours': self.smooth_contours_check.isChecked(),
            'preview': self.preview_check.isChecked(),
        }
        
        return options
    
    def set_image_range(self, min_value, max_value):
        """Set the range of values for the image."""
        self.min_value = min_value
        self.max_value = max_value
        
        # Update ranges for sliders and spinboxes
        self.lower_threshold_spin.setRange(min_value, max_value)
        self.lower_threshold_slider.setRange(min_value, max_value)
        self.upper_threshold_spin.setRange(min_value, max_value)
        self.upper_threshold_slider.setRange(min_value, max_value)
        
        # Set default values to a reasonable range within the data
        default_lower = min_value + (max_value - min_value) * 0.25
        default_upper = min_value + (max_value - min_value) * 0.75
        
        self.lower_threshold_spin.setValue(int(default_lower))
        self.lower_threshold_slider.setValue(int(default_lower))
        self.upper_threshold_spin.setValue(int(default_upper))
        self.upper_threshold_slider.setValue(int(default_upper))
    
    def _on_operation_changed(self):
        """Handle operation selection change."""
        operation_id = self.operation_buttons.checkedId()
        if operation_id > 0:
            operation = ThresholdOperation(operation_id)
            self._update_ui_for_operation(operation)
            self._on_settings_changed()
    
    def _on_mode_changed(self):
        """Handle mode selection change."""
        self._on_settings_changed()
    
    def _on_preview_changed(self):
        """Handle preview checkbox state change."""
        preview_enabled = self.preview_check.isChecked()
        if preview_enabled:
            self.preview_mode_btn.setEnabled(True)
            if self.preview_mode_btn.isChecked():
                self._on_settings_changed()
        else:
            if self.preview_mode_btn.isChecked():
                self.draw_mode_btn.setChecked(True)
            self.preview_mode_btn.setEnabled(False)
            self._on_settings_changed()
    
    def _on_settings_changed(self):
        """Handle settings changes and emit signal."""
        options = self.get_options()
        self.toolChanged.emit(options) 