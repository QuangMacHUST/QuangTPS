#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module containing image widget implementations for QuangTPS.

This module provides the widgets for displaying and interacting with medical images
in different orientations and formats.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QRubberBand,
    QMenu,
    QAction,
)
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QMouseEvent,
    QWheelEvent,
    QKeyEvent,
    QCursor,
)
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal, QRectF, QPointF


class ImageSliceWidget(QWidget):
    """
    Widget for displaying a 2D image slice with interaction capabilities.

    This widget handles display of medical image slices with proper window/level,
    overlays for structures, dose, and interaction with mouse and keyboard for
    image manipulation and structure drawing.

    Signals:
    --------
    mouse_pressed : pyqtSignal
        Emitted when mouse is pressed on the image
    mouse_moved : pyqtSignal
        Emitted when mouse is moved over the image
    mouse_released : pyqtSignal
        Emitted when mouse is released
    key_pressed : pyqtSignal
        Emitted when a key is pressed while the widget has focus
    key_released : pyqtSignal
        Emitted when a key is released
    slice_changed : pyqtSignal
        Emitted when the displayed slice changes
    """

    # Define the required signals
    mouse_pressed = pyqtSignal(QMouseEvent)
    mouse_moved = pyqtSignal(QMouseEvent)
    mouse_released = pyqtSignal(QMouseEvent)
    key_pressed = pyqtSignal(QKeyEvent)
    key_released = pyqtSignal(QKeyEvent)
    slice_changed = pyqtSignal(int)  # Slice index

    def __init__(self, parent=None, orientation="axial"):
        """
        Initialize the image slice widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        orientation : str, optional
            Orientation of the slice ('axial', 'sagittal', or 'coronal')
        """
        super().__init__(parent)

        # Widget properties
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus
        self.setMouseTracking(True)  # Track mouse movement

        # Image display properties
        self.orientation = orientation
        self.image_data = None
        self.displayed_image = QImage()
        self.pixmap = QPixmap()
        self.overlay_pixmap = QPixmap()

        # View properties
        self.window_width = 400  # Default window width
        self.window_level = 40  # Default window level
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)

        # Interaction state
        self.is_panning = False
        self.last_pan_pos = QPoint()
        self.is_drawing = False
        self.cursor_position = QPoint()

        # Slice properties
        self.current_slice_idx = 0
        self.max_slices = 0

        # Overlay data
        self.structure_overlays = {}  # Map of structure name to overlay data
        self.dose_overlay = None
        self.show_structures = True
        self.show_dose = True
        self.overlay_opacity = 0.5

        # Additional UI elements
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.rubber_band_origin = QPoint()

        # Set background color
        self.setStyleSheet("background-color: black;")

    def set_image_data(self, image_data: np.ndarray, slice_idx: int = 0):
        """
        Set the image data to display.

        Parameters
        ----------
        image_data : np.ndarray
            3D array of image data
        slice_idx : int, optional
            Index of the slice to display
        """
        if image_data is None or image_data.size == 0:
            return

        self.image_data = image_data
        self.max_slices = image_data.shape[0] if len(image_data.shape) > 2 else 1

        # Set current slice within valid range
        self.set_slice(min(slice_idx, self.max_slices - 1))

        # Reset view
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)

        # Update the display
        self.update()

    def set_slice(self, slice_idx: int):
        """
        Set the current slice to display.

        Parameters
        ----------
        slice_idx : int
            Index of the slice to display
        """
        if self.image_data is None:
            return

        if 0 <= slice_idx < self.max_slices:
            self.current_slice_idx = slice_idx
            self._update_displayed_image()
            self.slice_changed.emit(slice_idx)
            self.update()

    def set_window_level(self, window: int, level: int):
        """
        Set the window/level values for image display.

        Parameters
        ----------
        window : int
            Window width value
        level : int
            Window level value
        """
        self.window_width = max(1, window)
        self.window_level = level
        self._update_displayed_image()
        self.update()

    def set_brightness(self, brightness: int):
        """
        Set the brightness (window level) of the displayed image.

        Parameters
        ----------
        brightness : int
            Brightness value (window level)
        """
        self.set_window_level(self.window_width, brightness)

    def set_contrast(self, contrast: int):
        """
        Set the contrast (window width) of the displayed image.

        Parameters
        ----------
        contrast : int
            Contrast value (window width)
        """
        self.set_window_level(contrast, self.window_level)

    def set_background_data(self, data: np.ndarray):
        """
        Set background image data.

        Parameters
        ----------
        data : np.ndarray
            2D array of background image data
        """
        if data is None or data.size == 0:
            return

        if len(data.shape) == 2:
            # Single 2D slice
            self.image_data = data.reshape(1, data.shape[0], data.shape[1])
            self.max_slices = 1
            self.current_slice_idx = 0
            self._update_displayed_image()
            self.update()

    def add_structure_overlay(
        self, structure_name: str, contour_data: Dict, color: QColor
    ):
        """
        Add structure contour overlay.

        Parameters
        ----------
        structure_name : str
            Name of the structure
        contour_data : Dict
            Dictionary mapping slice indices to contour point lists
        color : QColor
            Color to use for the structure overlay
        """
        self.structure_overlays[structure_name] = {
            "data": contour_data,
            "color": color,
            "visible": True,
        }
        self.update()

    def set_structure_visibility(self, structure_name: str, visible: bool):
        """
        Set the visibility of a structure overlay.

        Parameters
        ----------
        structure_name : str
            Name of the structure
        visible : bool
            Whether the structure should be visible
        """
        if structure_name in self.structure_overlays:
            self.structure_overlays[structure_name]["visible"] = visible
            self.update()

    def set_dose_overlay(self, dose_data: np.ndarray, colormap: str = "jet"):
        """
        Set dose overlay data.

        Parameters
        ----------
        dose_data : np.ndarray
            3D array of dose data
        colormap : str, optional
            Colormap to use for dose display
        """
        self.dose_overlay = {"data": dose_data, "colormap": colormap, "visible": True}
        self.update()

    def _update_displayed_image(self):
        """Update the displayed image with current window/level settings."""
        if self.image_data is None:
            return

        # Get current slice
        if len(self.image_data.shape) > 2:
            slice_data = self.image_data[self.current_slice_idx]
        else:
            slice_data = self.image_data

        # Apply window/level
        min_val = self.window_level - self.window_width / 2
        max_val = self.window_level + self.window_width / 2

        # Clip and scale to 0-255
        normalized = np.clip(slice_data, min_val, max_val)
        normalized = (normalized - min_val) / (max_val - min_val) * 255
        display_data = normalized.astype(np.uint8)

        # Create QImage
        height, width = display_data.shape
        bytes_per_line = width
        self.displayed_image = QImage(
            display_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8
        )

        # Create pixmap
        self.pixmap = QPixmap.fromImage(self.displayed_image)

    def _draw_overlays(self, painter: QPainter):
        """
        Draw structure and dose overlays.

        Parameters
        ----------
        painter : QPainter
            Painter to use for drawing
        """
        if not self.pixmap or self.pixmap.isNull():
            return

        # Draw dose overlay if available
        if self.dose_overlay and self.dose_overlay["visible"] and self.show_dose:
            self._draw_dose_overlay(painter)

        # Draw structure overlays
        if self.show_structures:
            for name, structure in self.structure_overlays.items():
                if structure["visible"]:
                    self._draw_structure(painter, name, structure)

    def _draw_structure(self, painter: QPainter, name: str, structure: Dict):
        """
        Draw a single structure overlay.

        Parameters
        ----------
        painter : QPainter
            Painter to use for drawing
        name : str
            Name of the structure
        structure : Dict
            Structure data dictionary
        """
        contour_data = structure["data"]
        color = structure["color"]

        # Check if we have contour data for current slice
        if self.current_slice_idx not in contour_data:
            return

        # Set up pen for contour drawing
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)

        # Set up brush with translucent fill
        fill_color = QColor(color)
        fill_color.setAlpha(int(80))  # 30% opacity
        painter.setBrush(QBrush(fill_color))

        # Draw each contour in the current slice
        contours = contour_data[self.current_slice_idx]
        for contour in contours:
            points = []
            for point in contour:
                x, y = point
                # Apply zoom and pan
                x = int(x * self.zoom_factor) + self.pan_offset.x()
                y = int(y * self.zoom_factor) + self.pan_offset.y()
                points.append(QPoint(x, y))

            # Draw polygon
            if len(points) > 2:
                painter.drawPolygon(points)

    def _draw_dose_overlay(self, painter: QPainter):
        """
        Draw the dose overlay.

        Parameters
        ----------
        painter : QPainter
            Painter to use for drawing
        """
        # Implementation would depend on dose visualization method
        # Simplified placeholder
        if self.dose_overlay and "data" in self.dose_overlay:
            dose_data = self.dose_overlay["data"]
            if len(dose_data.shape) > 2:
                dose_slice = dose_data[self.current_slice_idx]
            else:
                dose_slice = dose_data

            # Very basic colorization - would be more complex in real implementation
            painter.setOpacity(0.5)
            painter.drawText(10, 30, "Dose Overlay Active")
            painter.setOpacity(1.0)

    def paintEvent(self, event):
        """
        Handle paint events.

        Parameters
        ----------
        event : QPaintEvent
            Paint event
        """
        if not self.pixmap or self.pixmap.isNull():
            return

        painter = QPainter(self)

        # Draw background image
        scaled_pixmap = self.pixmap.scaled(
            int(self.pixmap.width() * self.zoom_factor),
            int(self.pixmap.height() * self.zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # Calculate position to center the image
        x = (self.width() - scaled_pixmap.width()) // 2 + self.pan_offset.x()
        y = (self.height() - scaled_pixmap.height()) // 2 + self.pan_offset.y()

        # Draw the image
        painter.drawPixmap(x, y, scaled_pixmap)

        # Draw overlays
        self._draw_overlays(painter)

        # Draw orientation label
        painter.setPen(Qt.white)
        painter.drawText(10, 20, f"Orientation: {self.orientation.capitalize()}")
        painter.drawText(
            10, 40, f"Slice: {self.current_slice_idx + 1}/{self.max_slices}"
        )

        # End painting
        painter.end()

    def mousePressEvent(self, event):
        """
        Handle mouse press events.

        Parameters
        ----------
        event : QMouseEvent
            Mouse event
        """
        if event.button() == Qt.MiddleButton:
            # Start panning
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            # For drawing or selection operations
            self.is_drawing = True

            # Start rubber band for region selection if needed
            if event.modifiers() & Qt.ControlModifier:
                self.rubber_band_origin = event.pos()
                self.rubber_band.setGeometry(QRect(self.rubber_band_origin, QSize()))
                self.rubber_band.show()

        # Emit signal for other components to handle
        self.mouse_pressed.emit(event)

        # Call parent handler
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Handle mouse move events.

        Parameters
        ----------
        event : QMouseEvent
            Mouse event
        """
        self.cursor_position = event.pos()

        if self.is_panning:
            # Calculate pan delta
            delta = event.pos() - self.last_pan_pos
            self.pan_offset += delta
            self.last_pan_pos = event.pos()
            self.update()
        elif self.is_drawing and event.buttons() & Qt.LeftButton:
            # Handle drawing operations
            if event.modifiers() & Qt.ControlModifier and self.rubber_band.isVisible():
                # Update rubber band for selection
                self.rubber_band.setGeometry(
                    QRect(self.rubber_band_origin, event.pos()).normalized()
                )

        # Emit signal for other components to handle
        self.mouse_moved.emit(event)

        # Call parent handler
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Handle mouse release events.

        Parameters
        ----------
        event : QMouseEvent
            Mouse event
        """
        if event.button() == Qt.MiddleButton:
            # End panning
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton:
            # End drawing or selection
            self.is_drawing = False

            # Handle rubber band selection
            if self.rubber_band.isVisible():
                selection_rect = self.rubber_band.geometry()
                self.rubber_band.hide()
                # Process the selection...

        # Emit signal for other components to handle
        self.mouse_released.emit(event)

        # Call parent handler
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """
        Handle mouse wheel events.

        Parameters
        ----------
        event : QWheelEvent
            Wheel event
        """
        delta = event.angleDelta().y()

        if event.modifiers() & Qt.ControlModifier:
            # Zoom with Ctrl+Wheel
            zoom_factor = 1.1 if delta > 0 else 0.9
            self.zoom_factor *= zoom_factor
            self.zoom_factor = max(0.2, min(10.0, self.zoom_factor))  # Limit zoom range
        else:
            # Change slice with wheel
            new_slice = self.current_slice_idx + (-1 if delta > 0 else 1)
            if 0 <= new_slice < self.max_slices:
                self.set_slice(new_slice)

        self.update()

        # Call parent handler
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        """
        Handle key press events.

        Parameters
        ----------
        event : QKeyEvent
            Key event
        """
        # Emit signal for other components to handle
        self.key_pressed.emit(event)

        # Handle navigation keys
        if event.key() == Qt.Key_Up or event.key() == Qt.Key_Down:
            # Up/Down: change slice
            delta = -1 if event.key() == Qt.Key_Up else 1
            new_slice = self.current_slice_idx + delta
            if 0 <= new_slice < self.max_slices:
                self.set_slice(new_slice)
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # Plus: zoom in
            self.zoom_factor *= 1.1
            self.zoom_factor = min(10.0, self.zoom_factor)
            self.update()
        elif event.key() == Qt.Key_Minus:
            # Minus: zoom out
            self.zoom_factor *= 0.9
            self.zoom_factor = max(0.2, self.zoom_factor)
            self.update()
        elif event.key() == Qt.Key_R:
            # R: reset view
            self.zoom_factor = 1.0
            self.pan_offset = QPoint(0, 0)
            self.update()

        # Call parent handler
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """
        Handle key release events.

        Parameters
        ----------
        event : QKeyEvent
            Key event
        """
        # Emit signal for other components to handle
        self.key_released.emit(event)

        # Call parent handler
        super().keyReleaseEvent(event)


class ImageViewer(QWidget):
    """Widget for displaying multi-planar (MPR) image views."""

    # Define signals
    slice_changed = pyqtSignal(str, int)  # Orientation, slice index

    def __init__(self, parent=None):
        """Initialize the MPR image viewer."""
        super().__init__(parent)

        # Image data
        self.image_data = None
        self.current_position = [0, 0, 0]  # [z, y, x]

        # Create the layout
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create horizontal layout for the three views
        views_layout = QHBoxLayout()

        # Create the three slice views
        self.axial_view = ImageSliceWidget(self, orientation="axial")
        self.sagittal_view = ImageSliceWidget(self, orientation="sagittal")
        self.coronal_view = ImageSliceWidget(self, orientation="coronal")

        # Add views to layout
        views_layout.addWidget(self.axial_view)
        views_layout.addWidget(self.sagittal_view)
        views_layout.addWidget(self.coronal_view)

        # Add views layout to main layout
        layout.addLayout(views_layout)

        # Connect signals
        self.axial_view.slice_changed.connect(
            lambda idx: self._on_slice_changed("axial", idx)
        )
        self.sagittal_view.slice_changed.connect(
            lambda idx: self._on_slice_changed("sagittal", idx)
        )
        self.coronal_view.slice_changed.connect(
            lambda idx: self._on_slice_changed("coronal", idx)
        )

    def set_image_data(self, image_data):
        """
        Set the 3D image data to display.

        Parameters
        ----------
        image_data : np.ndarray
            3D array of image data
        """
        if image_data is None or image_data.size == 0:
            return

        self.image_data = image_data

        # Get dimensions
        if len(image_data.shape) == 3:
            z, y, x = image_data.shape
        else:
            # Handle 2D image
            y, x = image_data.shape
            z = 1

        # Initialize current position
        self.current_position = [z // 2, y // 2, x // 2]

        # Update all views
        self._update_displays()

    def _on_slice_changed(self, orientation, slice_idx):
        """
        Handle slice change in one of the views.

        Parameters
        ----------
        orientation : str
            Orientation of the view ('axial', 'sagittal', or 'coronal')
        slice_idx : int
            New slice index
        """
        # Update current position based on orientation
        if orientation == "axial":
            self.current_position[0] = slice_idx
        elif orientation == "sagittal":
            self.current_position[2] = slice_idx
        elif orientation == "coronal":
            self.current_position[1] = slice_idx

        # Update other views
        self._update_displays()

        # Emit signal
        self.slice_changed.emit(orientation, slice_idx)

    def _update_displays(self):
        """Update all three views with current position."""
        if self.image_data is None:
            return

        # Prepare slices for each view
        if len(self.image_data.shape) == 3:
            z, y, x = self.image_data.shape

            # Get current position
            curr_z, curr_y, curr_x = self.current_position

            # Ensure position is within bounds
            curr_z = max(0, min(curr_z, z - 1))
            curr_y = max(0, min(curr_y, y - 1))
            curr_x = max(0, min(curr_x, x - 1))

            # Update current position
            self.current_position = [curr_z, curr_y, curr_x]

            # Extract slices
            axial_slice = self.image_data[curr_z, :, :]
            sagittal_slice = self.image_data[:, :, curr_x]
            coronal_slice = self.image_data[:, curr_y, :]

            # Update views
            self.axial_view.set_image_data(axial_slice)
            self.sagittal_view.set_image_data(sagittal_slice)
            self.coronal_view.set_image_data(coronal_slice)
        else:
            # Handle 2D image - only update axial view
            self.axial_view.set_image_data(self.image_data)

    def set_window_level(self, window, level):
        """
        Set window/level for all views.

        Parameters
        ----------
        window : int
            Window width
        level : int
            Window level
        """
        self.axial_view.set_window_level(window, level)
        self.sagittal_view.set_window_level(window, level)
        self.coronal_view.set_window_level(window, level)

    def add_structure_overlay(self, structure_name, contour_data, color):
        """
        Add structure overlay to all views.

        Parameters
        ----------
        structure_name : str
            Name of the structure
        contour_data : Dict
            Contour data for each slice
        color : QColor
            Color for the structure
        """
        self.axial_view.add_structure_overlay(structure_name, contour_data, color)
        self.sagittal_view.add_structure_overlay(structure_name, contour_data, color)
        self.coronal_view.add_structure_overlay(structure_name, contour_data, color)

    def set_structure_visibility(self, structure_name, visible):
        """
        Set visibility of a structure in all views.

        Parameters
        ----------
        structure_name : str
            Name of the structure
        visible : bool
            Whether the structure should be visible
        """
        self.axial_view.set_structure_visibility(structure_name, visible)
        self.sagittal_view.set_structure_visibility(structure_name, visible)
        self.coronal_view.set_structure_visibility(structure_name, visible)

    def get_current_slice_indices(self):
        """
        Get current slice indices for all views.

        Returns
        -------
        Tuple[int, int, int]
            Current slice indices (axial, sagittal, coronal)
        """
        return tuple(self.current_position)
