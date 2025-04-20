#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image control widget for QuangTPS.

This module provides a widget for controlling image display parameters
such as window/level, brightness/contrast, zoom, and pan.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QComboBox,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon

logger = logging.getLogger(__name__)


class ImageControlWidget(QWidget):
    """Widget for controlling image display parameters."""

    # Tín hiệu (signals)
    window_changed = pyqtSignal(int, int)  # window width, window level
    slice_changed = pyqtSignal(int)
    view_changed = pyqtSignal(str)  # "axial", "sagittal", "coronal"
    zoom_changed = pyqtSignal(float)
    pan_changed = pyqtSignal(int, int)  # dx, dy
    brightness_changed = pyqtSignal(int)
    contrast_changed = pyqtSignal(int)
    overlay_opacity_changed = pyqtSignal(float)
    preset_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialize the image control widget."""
        super().__init__(parent)

        # Initialize properties
        self.window_width = 400
        self.window_level = 40
        self.current_slice = 0
        self.max_slice = 100
        self.current_view = "axial"
        self.zoom = 1.0
        self.brightness = 0
        self.contrast = 1
        self.overlay_opacity = 0.7

        # Define presets
        self.window_presets = {
            "CT Bone": (2000, 500),
            "CT Lung": (1500, -600),
            "CT Abdomen": (400, 40),
            "CT Brain": (80, 40),
            "CT Angio": (600, 160),
            "MR T1": (500, 300),
            "MR T2": (1000, 600),
            "PET": (20000, 10000),
        }

        # Initialize the UI
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setMinimumWidth(250)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # View selection group
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)

        # View combo box
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Axial", "Sagittal", "Coronal"])
        self.view_combo.setCurrentText(self.current_view.capitalize())
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        view_layout.addWidget(self.view_combo)

        # Slice control
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("Slice:"))

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, self.max_slice)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        slice_layout.addWidget(self.slice_slider, 1)

        self.slice_spinbox = QSpinBox()
        self.slice_spinbox.setRange(0, self.max_slice)
        self.slice_spinbox.setValue(self.current_slice)
        self.slice_spinbox.valueChanged.connect(self._on_slice_changed)
        slice_layout.addWidget(self.slice_spinbox)

        view_layout.addLayout(slice_layout)
        main_layout.addWidget(view_group)

        # Window/level group
        window_group = QGroupBox("Window/Level")
        window_layout = QFormLayout(window_group)

        # Window width control
        self.window_width_slider = QSlider(Qt.Horizontal)
        self.window_width_slider.setRange(1, 4000)
        self.window_width_slider.setValue(self.window_width)
        self.window_width_slider.valueChanged.connect(self._on_window_width_changed)
        window_layout.addRow("Width:", self.window_width_slider)

        self.window_width_spinbox = QSpinBox()
        self.window_width_spinbox.setRange(1, 4000)
        self.window_width_spinbox.setValue(self.window_width)
        self.window_width_spinbox.valueChanged.connect(self._on_window_width_changed)
        window_layout.addRow("", self.window_width_spinbox)

        # Window level control
        self.window_level_slider = QSlider(Qt.Horizontal)
        self.window_level_slider.setRange(-1000, 3000)
        self.window_level_slider.setValue(self.window_level)
        self.window_level_slider.valueChanged.connect(self._on_window_level_changed)
        window_layout.addRow("Level:", self.window_level_slider)

        self.window_level_spinbox = QSpinBox()
        self.window_level_spinbox.setRange(-1000, 3000)
        self.window_level_spinbox.setValue(self.window_level)
        self.window_level_spinbox.valueChanged.connect(self._on_window_level_changed)
        window_layout.addRow("", self.window_level_spinbox)

        # Brightness/Contrast group
        bc_group = QGroupBox("Brightness/Contrast")
        bc_layout = QFormLayout(bc_group)

        # Brightness control
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(self.brightness)
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        bc_layout.addRow("Brightness:", self.brightness_slider)

        # Contrast control
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(int(self.contrast * 100))
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        bc_layout.addRow("Contrast:", self.contrast_slider)

        # Presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.window_presets.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo, 1)

        window_layout.addRow(preset_layout)
        main_layout.addWidget(window_group)
        main_layout.addWidget(bc_group)

        # Overlay opacity
        overlay_group = QGroupBox("Overlay")
        overlay_layout = QFormLayout(overlay_group)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.overlay_opacity * 100))
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        overlay_layout.addRow("Opacity:", self.opacity_slider)

        main_layout.addWidget(overlay_group)

        # Add stretch to push all controls to the top
        main_layout.addStretch(1)

    def _on_view_changed(self, view_text):
        """Handle view change."""
        view = view_text.lower()
        if view != self.current_view:
            self.current_view = view
            self.view_changed.emit(view)

    def _on_slice_changed(self, value):
        """Handle slice change."""
        if value != self.current_slice:
            self.current_slice = value

            # Update both controls to stay in sync
            if self.sender() == self.slice_slider:
                self.slice_spinbox.setValue(value)
            else:
                self.slice_slider.setValue(value)

            self.slice_changed.emit(value)

    def _on_window_width_changed(self, value):
        """Handle window width change."""
        if value != self.window_width:
            self.window_width = value

            # Update both controls to stay in sync
            if self.sender() == self.window_width_slider:
                self.window_width_spinbox.setValue(value)
            else:
                self.window_width_slider.setValue(value)

            self.window_changed.emit(self.window_width, self.window_level)

    def _on_window_level_changed(self, value):
        """Handle window level change."""
        if value != self.window_level:
            self.window_level = value

            # Update both controls to stay in sync
            if self.sender() == self.window_level_slider:
                self.window_level_spinbox.setValue(value)
            else:
                self.window_level_slider.setValue(value)

            self.window_changed.emit(self.window_width, self.window_level)

    def _on_brightness_changed(self, value):
        """Handle brightness change."""
        if value != self.brightness:
            self.brightness = value
            self.brightness_changed.emit(value)

    def _on_contrast_changed(self, value):
        """Handle contrast change."""
        contrast = value / 100.0
        if contrast != self.contrast:
            self.contrast = contrast
            self.contrast_changed.emit(value)

    def _on_opacity_changed(self, value):
        """Handle opacity change."""
        opacity = value / 100.0
        if opacity != self.overlay_opacity:
            self.overlay_opacity = opacity
            self.overlay_opacity_changed.emit(opacity)

    def _on_preset_changed(self, preset_name):
        """Handle preset selection."""
        if preset_name in self.window_presets:
            width, level = self.window_presets[preset_name]
            self.window_width = width
            self.window_level = level

            # Update controls
            self.window_width_slider.setValue(width)
            self.window_width_spinbox.setValue(width)
            self.window_level_slider.setValue(level)
            self.window_level_spinbox.setValue(level)

            # Emit signals
            self.window_changed.emit(width, level)
            self.preset_selected.emit(preset_name)

    def set_max_slice(self, max_slice):
        """Set the maximum slice value."""
        self.max_slice = max(0, max_slice)
        self.slice_slider.setRange(0, self.max_slice)
        self.slice_spinbox.setRange(0, self.max_slice)

    def set_current_slice(self, slice_idx):
        """Set the current slice index."""
        slice_idx = max(0, min(slice_idx, self.max_slice))
        if slice_idx != self.current_slice:
            self.current_slice = slice_idx
            self.slice_slider.setValue(slice_idx)
            self.slice_spinbox.setValue(slice_idx)

    def set_window(self, width, level):
        """Set the window width and level."""
        if width != self.window_width or level != self.window_level:
            self.window_width = width
            self.window_level = level

            # Update controls
            self.window_width_slider.setValue(width)
            self.window_width_spinbox.setValue(width)
            self.window_level_slider.setValue(level)
            self.window_level_spinbox.setValue(level)
