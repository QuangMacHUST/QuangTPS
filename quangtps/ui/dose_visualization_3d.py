#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Dose Visualization Component for QuangTPS

This module provides enhanced 3D dose visualization capabilities with interactive
isodose control, color mapping, and real-time visualization updates.
It integrates with the External Beam Planning tab to provide Eclipse-like
3D dose visualization.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSplitter,
    QSlider,
    QFrame,
    QGridLayout,
    QSpinBox,
    QSizePolicy,
    QToolBar,
    QAction,
    QComboBox,
    QCheckBox,
    QToolButton,
    QMenu,
    QGroupBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QColorDialog,
    QInputDialog,
    QFileDialog,
)
from PyQt5.QtGui import QColor, QIcon, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, QSize

from quangtps.ui.vtk_viewer_3d import VTKViewer3D
from quangtps.core.logging import get_logger
from quangtps.dose.dose_grid import DoseGrid

logger = get_logger(__name__)


class IsodoseLevel:
    """Class representing an isodose level with color and visibility settings."""

    def __init__(
        self, level: float, color: Tuple[float, float, float], visible: bool = True
    ):
        """
        Initialize an isodose level.

        Parameters
        ----------
        level : float
            Dose level in Gy
        color : tuple
            RGB color tuple (values between 0 and 1)
        visible : bool
            Whether the isodose level is visible
        """
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

    def __init__(self, parent=None):
        """Initialize the 3D dose visualization component."""
        super().__init__(parent)

        # Initialize members
        self.dose_grid = None
        self.prescription_dose = None
        self.isodose_levels = {}  # Dict mapping dose level to IsodoseLevel objects
        self.current_view_mode = "3D"  # "3D", "Axial", "Sagittal", "Coronal"

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
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Add toolbar
        self.toolbar = QToolBar("Dose Visualization Controls")
        self.toolbar.setIconSize(QSize(16, 16))

        # Toolbar actions
        self.action_reset_view = QAction(
            QIcon("quangtps/ui/icons/reset_view.png"), "Reset View", self
        )
        self.action_reset_view.triggered.connect(self._reset_view)
        self.toolbar.addAction(self.action_reset_view)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(
            ["3D View", "Axial View", "Sagittal View", "Coronal View"]
        )
        self.view_mode_combo.currentIndexChanged.connect(self._change_view_mode)
        self.toolbar.addWidget(QLabel("View:"))
        self.toolbar.addWidget(self.view_mode_combo)

        self.toolbar.addSeparator()

        self.toolbar.addWidget(QLabel("Prescription:"))
        self.prescription_spinbox = QDoubleSpinBox()
        self.prescription_spinbox.setRange(0.1, 100.0)
        self.prescription_spinbox.setValue(2.0)
        self.prescription_spinbox.setSuffix(" Gy")
        self.prescription_spinbox.setDecimals(1)
        self.prescription_spinbox.valueChanged.connect(self._update_prescription)
        self.toolbar.addWidget(self.prescription_spinbox)

        # Screenshot button
        self.screenshot_btn = QPushButton("Screenshot")
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        self.toolbar.addWidget(self.screenshot_btn)

        main_layout.addWidget(self.toolbar)

        # Main splitter for 3D view and controls
        splitter = QSplitter(Qt.Horizontal)

        # 3D viewer
        self.vtk_viewer = VTKViewer3D()
        splitter.addWidget(self.vtk_viewer)

        # Right side controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Isodose controls group
        isodose_group = QGroupBox("Isodose Controls")
        isodose_layout = QVBoxLayout(isodose_group)

        # Isodose table
        self.isodose_table = QTableWidget(0, 3)
        self.isodose_table.setHorizontalHeaderLabels(["Level", "Color", "Visible"])
        self.isodose_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        isodose_layout.addWidget(self.isodose_table)

        # Isodose buttons
        isodose_buttons_layout = QHBoxLayout()
        self.add_isodose_btn = QPushButton("Add")
        self.add_isodose_btn.clicked.connect(self._add_isodose_level)

        self.remove_isodose_btn = QPushButton("Remove")
        self.remove_isodose_btn.clicked.connect(self._remove_isodose_level)

        self.reset_isodose_btn = QPushButton("Reset Default")
        self.reset_isodose_btn.clicked.connect(self._reset_isodose_levels)

        isodose_buttons_layout.addWidget(self.add_isodose_btn)
        isodose_buttons_layout.addWidget(self.remove_isodose_btn)
        isodose_buttons_layout.addWidget(self.reset_isodose_btn)
        isodose_layout.addLayout(isodose_buttons_layout)

        controls_layout.addWidget(isodose_group)

        # Visualization controls group
        vis_group = QGroupBox("Visualization Controls")
        vis_layout = QVBoxLayout(vis_group)

        # Transparency slider
        transparency_layout = QHBoxLayout()
        transparency_layout.addWidget(QLabel("Transparency:"))
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(30)
        self.transparency_slider.valueChanged.connect(self._update_transparency)
        transparency_layout.addWidget(self.transparency_slider)
        vis_layout.addLayout(transparency_layout)

        # Display mode
        display_layout = QHBoxLayout()
        display_layout.addWidget(QLabel("Display:"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["Surface", "Volume", "Contour"])
        self.display_mode_combo.currentIndexChanged.connect(self._update_display_mode)
        display_layout.addWidget(self.display_mode_combo)
        vis_layout.addLayout(display_layout)

        # Show/hide structures
        self.show_structures_check = QCheckBox("Show Structures")
        self.show_structures_check.setChecked(True)
        self.show_structures_check.stateChanged.connect(self._toggle_structures)
        vis_layout.addWidget(self.show_structures_check)

        # Show/hide DVH
        self.show_dvh_check = QCheckBox("Show DVH")
        self.show_dvh_check.setChecked(False)
        self.show_dvh_check.stateChanged.connect(self._toggle_dvh)
        vis_layout.addWidget(self.show_dvh_check)

        controls_layout.addWidget(vis_group)

        # Add legend/info
        info_group = QGroupBox("Dose Information")
        info_layout = QVBoxLayout(info_group)

        # Min, max, mean dose
        self.dose_info_label = QLabel("Min: N/A  Max: N/A  Mean: N/A")
        info_layout.addWidget(self.dose_info_label)

        # Statistics
        self.stats_label = QLabel("No dose data available")
        info_layout.addWidget(self.stats_label)

        controls_layout.addWidget(info_group)

        # Add a stretch to ensure controls stay at the top
        controls_layout.addStretch()

        # Add controls to splitter
        splitter.addWidget(controls_widget)

        # Set initial sizes (70% viewer, 30% controls)
        splitter.setSizes([700, 300])

        main_layout.addWidget(splitter)

        # Initialize isodose levels
        self._init_default_isodose_levels()

    def _init_default_isodose_levels(self):
        """Initialize default isodose levels based on prescription dose."""
        self.isodose_levels.clear()
        self.isodose_table.setRowCount(0)

        prescription = self.prescription_spinbox.value()

        for i, percent in enumerate(self.default_level_percentages):
            level = prescription * percent / 100.0
            if i < len(self.default_colors):
                color = self.default_colors[i]
            else:
                # Generate random color if we run out of defaults
                color = (np.random.random(), np.random.random(), np.random.random())

            self._add_isodose_to_table(level, color)

    def _add_isodose_to_table(self, level: float, color: Tuple[float, float, float]):
        """Add an isodose level to the table and internal storage."""
        # Create IsodoseLevel object
        isodose = IsodoseLevel(level, color)
        self.isodose_levels[level] = isodose

        # Add to table
        row = self.isodose_table.rowCount()
        self.isodose_table.insertRow(row)

        # Level
        level_item = QTableWidgetItem(f"{level:.1f} Gy")
        level_item.setData(Qt.UserRole, level)
        self.isodose_table.setItem(row, 0, level_item)

        # Color
        color_item = QTableWidgetItem()
        color_item.setBackground(
            QBrush(
                QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            )
        )
        color_item.setData(Qt.UserRole, color)
        self.isodose_table.setItem(row, 1, color_item)

        # Visibility checkbox
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(
            lambda state, l=level: self._toggle_isodose_visibility(l, state)
        )
        self.isodose_table.setCellWidget(row, 2, checkbox)

    def _add_isodose_level(self):
        """Add a new isodose level."""
        prescription = self.prescription_spinbox.value()

        # Ask for percentage of prescription
        percent, ok = QInputDialog.getDouble(
            self,
            "Add Isodose Level",
            "Enter percentage of prescription dose:",
            value=50.0,
            min=1.0,
            max=200.0,
            decimals=1,
        )

        if not ok:
            return

        level = prescription * percent / 100.0

        # Check if level already exists
        if level in self.isodose_levels:
            return

        # Ask for color
        color_dialog = QColorDialog(self)
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)

        if color_dialog.exec_():
            qcolor = color_dialog.selectedColor()
            color = (
                qcolor.red() / 255.0,
                qcolor.green() / 255.0,
                qcolor.blue() / 255.0,
            )

            # Add to table
            self._add_isodose_to_table(level, color)

            # Update visualization if dose grid exists
            if self.dose_grid is not None:
                self._update_dose_visualization()

    def _remove_isodose_level(self):
        """Remove the selected isodose level."""
        selected_rows = self.isodose_table.selectedIndexes()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        level_item = self.isodose_table.item(row, 0)
        level = level_item.data(Qt.UserRole)

        # Remove from storage
        if level in self.isodose_levels:
            # Remove actor from renderer if it exists
            isodose = self.isodose_levels[level]
            if isodose.actor and self.vtk_viewer.renderer:
                self.vtk_viewer.renderer.RemoveActor(isodose.actor)

            del self.isodose_levels[level]

        # Remove from table
        self.isodose_table.removeRow(row)

        # Update visualization
        self.vtk_viewer.vtk_widget.GetRenderWindow().Render()

    def _reset_isodose_levels(self):
        """Reset isodose levels to defaults."""
        # Clear existing isodose actors
        for level, isodose in self.isodose_levels.items():
            if isodose.actor and self.vtk_viewer.renderer:
                self.vtk_viewer.renderer.RemoveActor(isodose.actor)

        # Reinitialize default levels
        self._init_default_isodose_levels()

        # Update visualization if dose grid exists
        if self.dose_grid is not None:
            self._update_dose_visualization()

    def _toggle_isodose_visibility(self, level: float, state: int):
        """Toggle visibility of an isodose level."""
        if level not in self.isodose_levels:
            return

        isodose = self.isodose_levels[level]
        isodose.visible = state == Qt.Checked

        # Update actor visibility if it exists
        if isodose.actor:
            isodose.actor.SetVisibility(isodose.visible)
            self.vtk_viewer.vtk_widget.GetRenderWindow().Render()

    def _update_prescription(self, value: float):
        """Update isodose levels when prescription changes."""
        # Store old prescription
        old_prescription = self.prescription_dose or value
        self.prescription_dose = value

        # Calculate ratio
        ratio = value / old_prescription if old_prescription > 0 else 1.0

        # Update levels
        new_levels = {}
        for old_level, isodose in self.isodose_levels.items():
            new_level = old_level * ratio
            isodose.level = new_level
            new_levels[new_level] = isodose

        self.isodose_levels = new_levels

        # Update table
        self.isodose_table.setRowCount(0)
        for level, isodose in sorted(self.isodose_levels.items(), reverse=True):
            self._add_isodose_to_table(level, isodose.color)

        # Update visualization if dose grid exists
        if self.dose_grid is not None:
            self._update_dose_visualization()

    def _update_transparency(self, opacity_percent):
        """
        Update the transparency of the dose visualization.

        Args:
            opacity_percent (int): The opacity percentage (0-100)
        """
        opacity = max(0.0, min(1.0, opacity_percent / 100.0))

        # Update opacity for all isodose levels
        for level, isodose in self.isodose_levels.items():
            if isodose.actor:
                isodose.actor.GetProperty().SetOpacity(opacity)

        # Render
        self.vtk_viewer.vtk_widget.GetRenderWindow().Render()

    def _update_display_mode(self, index: int):
        """
        Cập nhật chế độ hiển thị cho phân phối liều.

        Parameters
        ----------
        index : int
            Chỉ số của chế độ hiển thị, 0: Surface, 1: Volume, 2: Contour
        """
        mode_names = ["Surface", "Volume", "Contour"]
        mode = mode_names[index] if index < len(mode_names) else "Surface"

        logger.info(f"Đang thay đổi chế độ hiển thị isodose sang: {mode}")

        # Nếu không có dữ liệu liều, thoát
        if self.dose_grid is None:
            return

        # Lấy actor hiện tại
        for level, isodose in self.isodose_levels.items():
            if isodose.actor is None:
                continue

            # Áp dụng chế độ hiển thị phù hợp
            if mode == "Surface":
                # Chế độ bề mặt isodose
                isodose.actor.GetProperty().SetRepresentationToSurface()
                isodose.actor.GetProperty().SetOpacity(
                    self.transparency_slider.value() / 100.0
                )

            elif mode == "Volume":
                # Chế độ khối lượng với độ trong suốt
                isodose.actor.GetProperty().SetRepresentationToSurface()
                isodose.actor.GetProperty().SetOpacity(
                    min(0.7, self.transparency_slider.value() / 100.0)
                )

            elif mode == "Contour":
                # Chế độ đường viền
                isodose.actor.GetProperty().SetRepresentationToWireframe()
                isodose.actor.GetProperty().SetLineWidth(2.0)
                isodose.actor.GetProperty().SetOpacity(1.0)  # Đường viền luôn đặc

        # Cập nhật hiển thị
        if hasattr(self.vtk_viewer, "vtk_widget") and self.vtk_viewer.vtk_widget:
            self.vtk_viewer.vtk_widget.GetRenderWindow().Render()

        # Phát tín hiệu thông báo đã cập nhật
        self.dose_visualization_updated.emit()

    def _toggle_structures(self, state: int):
        """Toggle visibility of structures."""
        visible = state == Qt.Checked
        self.vtk_viewer.toggle_structures(visible)

    def _toggle_dvh(self, state: int):
        """Toggle DVH display."""
        # Not implemented yet - would show a pop-up DVH window
        pass

    def _reset_view(self):
        """Reset the 3D view."""
        self.vtk_viewer.reset_view()

    def _change_view_mode(self, index: int):
        """Change the view mode (3D, Axial, Sagittal, Coronal)."""
        mode_names = ["3D", "Axial", "Sagittal", "Coronal"]
        if index < len(mode_names):
            self.current_view_mode = mode_names[index]

            camera = self.vtk_viewer.camera
            if self.current_view_mode == "3D":
                camera.SetPosition(0, -500, 0)
                camera.SetViewUp(0, 0, 1)
                camera.SetFocalPoint(0, 0, 0)
            elif self.current_view_mode == "Axial":
                camera.SetPosition(0, 0, 500)
                camera.SetViewUp(0, 1, 0)
                camera.SetFocalPoint(0, 0, 0)
            elif self.current_view_mode == "Sagittal":
                camera.SetPosition(500, 0, 0)
                camera.SetViewUp(0, 0, 1)
                camera.SetFocalPoint(0, 0, 0)
            elif self.current_view_mode == "Coronal":
                camera.SetPosition(0, 500, 0)
                camera.SetViewUp(0, 0, 1)
                camera.SetFocalPoint(0, 0, 0)

            self.vtk_viewer.vtk_widget.GetRenderWindow().Render()

    def _take_screenshot(self):
        """Chụp và lưu ảnh màn hình của hiển thị 3D."""
        try:
            if (
                not hasattr(self.vtk_viewer, "vtk_widget")
                or not self.vtk_viewer.vtk_widget
            ):
                logger.error("Không thể chụp màn hình: VTK widget không khởi tạo")
                return

            # Hiển thị hộp thoại lưu tệp
            try:
                from PyQt5.QtWidgets import QFileDialog
            except ImportError as e:
                logger.error(f"Không thể import QFileDialog: {e}")
                return

            # Tạo tên tệp mặc định dựa trên thời gian
            import datetime

            default_filename = f"dose_visualization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu ảnh chụp màn hình",
                default_filename,
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
            )

            if not filename:
                return  # Người dùng đã hủy

            # Thêm phần mở rộng .png nếu không có phần mở rộng
            if not any(
                filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]
            ):
                filename += ".png"

            # Tạo bộ lọc cửa sổ VTK để lưu ảnh chụp màn hình
            window_to_image_filter = vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(
                self.vtk_viewer.vtk_widget.GetRenderWindow()
            )
            window_to_image_filter.SetInputBufferTypeToRGB()
            window_to_image_filter.ReadFrontBufferOff()
            window_to_image_filter.Update()

            # Xác định bộ ghi tệp dựa trên phần mở rộng
            if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
                writer = vtk.vtkJPEGWriter()
            else:
                writer = vtk.vtkPNGWriter()

            writer.SetFileName(filename)
            writer.SetInputConnection(window_to_image_filter.GetOutputPort())
            writer.Write()

            logger.info(f"Đã lưu ảnh chụp màn hình thành công vào: {filename}")

        except Exception as e:
            logger.error(f"Lỗi khi lưu ảnh chụp màn hình: {str(e)}")
            from PyQt5.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Lỗi khi lưu ảnh chụp",
                f"Không thể lưu ảnh chụp màn hình: {str(e)}",
            )

    def set_dose_grid(self, dose_grid: DoseGrid):
        """
        Set the dose grid for visualization.

        Parameters
        ----------
        dose_grid : DoseGrid
            The dose grid to visualize
        """
        self.dose_grid = dose_grid

        if dose_grid is None:
            self._clear_dose()
            return

        # Update dose info
        if hasattr(dose_grid, "get_statistics"):
            min_dose, max_dose, mean_dose = dose_grid.get_statistics()
            self.dose_info_label.setText(
                f"Min: {min_dose:.2f} Gy  Max: {max_dose:.2f} Gy  Mean: {mean_dose:.2f} Gy"
            )
            self.stats_label.setText(f"Grid size: {dose_grid.shape}")

        # Update visualization
        self._update_dose_visualization()

    def _update_dose_visualization(self):
        """Update the 3D dose visualization based on current settings."""
        if self.dose_grid is None:
            return

        # Clear existing isodose surfaces
        self._clear_dose()

        # Get dose data as numpy array
        dose_array = getattr(self.dose_grid, "dose_array", None)

        if dose_array is None and hasattr(self.dose_grid, "get_array"):
            dose_array = self.dose_grid.get_array()

        if dose_array is None:
            logger.error("Could not get dose array from dose grid")
            return

        # Get spacing and origin from dose grid
        spacing = getattr(self.dose_grid, "spacing", (1.0, 1.0, 1.0))
        origin = getattr(self.dose_grid, "origin", (0.0, 0.0, 0.0))

        # Add dose to VTK viewer with current isodose levels
        isodose_levels = [
            (level, isodose.color) for level, isodose in self.isodose_levels.items()
        ]
        self.vtk_viewer.add_dose(self.dose_grid, isodose_levels)

        # Store actors for each isodose level
        if hasattr(self.vtk_viewer, "dose_actors") and self.vtk_viewer.dose_actors:
            for i, (level, _) in enumerate(isodose_levels):
                if i < len(self.vtk_viewer.dose_actors):
                    self.isodose_levels[level].actor = self.vtk_viewer.dose_actors[i]

        # Apply transparency
        self._update_transparency(self.transparency_slider.value())

        # Apply visibility settings
        for level, isodose in self.isodose_levels.items():
            if isodose.actor:
                isodose.actor.SetVisibility(isodose.visible)

        # Signal that visualization has been updated
        self.dose_visualization_updated.emit()

    def _clear_dose(self):
        """Clear all dose visualization."""
        # Clear isodose actors from renderer
        for level, isodose in self.isodose_levels.items():
            if isodose.actor and self.vtk_viewer.renderer:
                self.vtk_viewer.renderer.RemoveActor(isodose.actor)
                isodose.actor = None

        # Call VTK viewer's clear method
        if hasattr(self.vtk_viewer, "clear_dose"):
            self.vtk_viewer.clear_dose()

        # Reset dose info
        self.dose_info_label.setText("Min: N/A  Max: N/A  Mean: N/A")
        self.stats_label.setText("No dose data available")

    def set_image_data(self, image_data, spacing=None, origin=None):
        """Set the underlying image data for proper dose overlay."""
        self.vtk_viewer.set_image_data(image_data, spacing, origin)

    def add_structure(
        self, structure_id, mask, color=(1.0, 0.0, 0.0), opacity=0.5, name=None
    ):
        """Add a structure for visualization along with dose."""
        self.vtk_viewer.add_structure(structure_id, mask, color, opacity, name)

    def clear_all(self):
        """Clear all visualizations."""
        self._clear_dose()
        self.vtk_viewer.clear_all()

    def set_dose_threshold(self, dose_value):
        """
        Set the minimum dose threshold for visualization.

        Args:
            dose_value (float): The minimum dose value to display in Gy
        """
        self.dose_threshold = dose_value

        # Update the isodose display
        if hasattr(self, "dose_array") and self.dose_array is not None:
            self._update_isodose_display()

        # Log the change
        logger.info(f"Dose threshold set to {dose_value} Gy")

        # Request a refresh of the display
        if hasattr(self, "vtk_viewer") and hasattr(self.vtk_viewer, "render"):
            self.vtk_viewer.render()

    def _update_isodose_display(self):
        """Update the isodose display based on current settings."""
        if not hasattr(self, "dose_array") or self.dose_array is None:
            return

        # Clear existing isodose actors
        for level, isodose in self.isodose_levels.items():
            if isodose.actor and self.vtk_viewer.renderer:
                self.vtk_viewer.renderer.RemoveActor(isodose.actor)

        self.isodose_levels.clear()

        # Re-create isodose contours
        for level in self.isodose_levels:
            # Skip levels below threshold
            if level < self.dose_threshold:
                continue

            # Create contour at this level
            color = self.isodose_levels.get(level, (1, 1, 1))
            isodose = IsodoseLevel(level, color)
            self.isodose_levels[level] = isodose

            if isodose.actor:
                self.vtk_viewer.renderer.AddActor(isodose.actor)

        # Request a refresh
        if hasattr(self, "vtk_viewer") and hasattr(self.vtk_viewer, "render"):
            self.vtk_viewer.render()


# Test function
def test():
    """Test function for standalone testing."""
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = DoseVisualization3D()
    widget.resize(1200, 800)
    widget.show()

    # Create test dose grid with spherical dose distribution
    dose_array = np.zeros((100, 100, 100), dtype=np.float32)
    center = np.array([50, 50, 50])

    # Create spherical dose distribution
    for x in range(100):
        for y in range(100):
            for z in range(100):
                dist = np.sqrt(((np.array([x, y, z]) - center) ** 2).sum())
                if dist < 40:
                    dose_array[x, y, z] = max(0, 5.0 * (1.0 - dist / 40.0))

    # Create dummy dose grid
    class DummyDoseGrid:
        def __init__(self, dose_array):
            self.dose_array = dose_array
            self.shape = dose_array.shape
            self.spacing = (1.0, 1.0, 1.0)
            self.origin = (0.0, 0.0, 0.0)

        def get_statistics(self):
            return (
                np.min(self.dose_array),
                np.max(self.dose_array),
                np.mean(self.dose_array),
            )

    # Set test dose grid
    widget.set_dose_grid(DummyDoseGrid(dose_array))

    sys.exit(app.exec_())


if __name__ == "__main__":
    test()
