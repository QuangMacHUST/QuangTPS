#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MPR Viewer Module
================

This module provides a multi-planar reconstruction (MPR) viewer for displaying
medical images in axial, sagittal, and coronal orientations with structure
overlays and measurement tools.
"""

import os
import logging
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
    QSplitter, QScrollArea, QFrame, QToolBar, QAction, 
    QSlider, QSpinBox, QComboBox, QToolButton, QMenu,
    QSizePolicy
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, 
    QIcon, QCursor, QMouseEvent, QWheelEvent, QPalette,
    QFont, QTransform
)
from PyQt5.QtCore import (
    Qt, QSize, QPoint, QRect, QRectF, pyqtSignal, QObject, 
    QTimer, QEventLoop
)

from quangtps.core.types import ImageOrientation, PatientOrientation
from quangtps.imaging.image import Image

logger = logging.getLogger(__name__)

class ViewOrientation(Enum):
    """Enum for MPR view orientations."""
    AXIAL = 1
    SAGITTAL = 2
    CORONAL = 3

class MPRView(QWidget):
    """
    Single MPR view widget for displaying a specific orientation.
    
    Displays one orientation (axial, sagittal, or coronal) of a volume,
    with overlays for structures, measurements, and annotations.
    """
    
    # Signals
    sliceChanged = pyqtSignal(int)
    windowLevelChanged = pyqtSignal(int, int)  # window, level
    zoomChanged = pyqtSignal(float)
    mousePressed = pyqtSignal(QPoint, QPoint)  # view pos, image pos
    mouseMoved = pyqtSignal(QPoint, QPoint)    # view pos, image pos
    mouseReleased = pyqtSignal(QPoint, QPoint) # view pos, image pos
    mouseDoubleClicked = pyqtSignal(QPoint, QPoint) # view pos, image pos
    
    def __init__(self, orientation=ViewOrientation.AXIAL, parent=None):
        """Initialize MPR view with specified orientation."""
        super().__init__(parent)
        
        # View settings
        self.orientation = orientation
        self.title = str(orientation.name).capitalize()
        self.slice_index = 0
        self.max_slice = 0
        
        # Image properties
        self.image_data = None
        self.slice_data = None
        self.pixel_spacing = (1.0, 1.0, 1.0)  # x, y, z in mm
        self.window_width = 400
        self.window_level = 40
        
        # Display properties
        self.zoom = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_mouse_pos = QPoint()
        
        # Rendering properties
        self.pixmap = QPixmap()
        self.display_image = QImage()
        
        # Overlays
        self.structure_overlays = {}  # id -> (structure, color, is_selected)
        self.temp_overlays = []       # Temporary for drawing operations
        self.measurement_overlays = []
        self.annotation_overlays = []
        
        # Configure widget
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Create title label
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            font-weight: bold;
            background-color: #2070c0;
            color: white;
            padding: 2px;
        """)
        self.layout.addWidget(self.title_label)
        
        # Create view frame
        self.view_frame = QFrame()
        self.view_frame.setFrameShape(QFrame.StyledPanel)
        self.view_frame.setLineWidth(1)
        self.view_layout = QVBoxLayout(self.view_frame)
        self.view_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add view frame to layout
        self.layout.addWidget(self.view_frame, 1)
        
        # Create slice slider
        self.slice_controls = QWidget()
        self.slice_layout = QHBoxLayout(self.slice_controls)
        self.slice_layout.setContentsMargins(3, 0, 3, 0)
        
        self.slice_label = QLabel("Slice:")
        self.slice_layout.addWidget(self.slice_label)
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self.set_slice_index)
        self.slice_layout.addWidget(self.slice_slider, 1)
        
        self.slice_spinbox = QSpinBox()
        self.slice_spinbox.setMinimum(0)
        self.slice_spinbox.setMaximum(0)
        self.slice_spinbox.valueChanged.connect(self.set_slice_index)
        self.slice_layout.addWidget(self.slice_spinbox)
        
        # Add slice controls to layout
        self.layout.addWidget(self.slice_controls)
        
        # Create position info label
        self.position_label = QLabel("X: -- Y: -- Value: --")
        self.position_label.setAlignment(Qt.AlignCenter)
        self.position_label.setStyleSheet("font-size: 9pt; padding: 2px; color: #505050;")
        self.layout.addWidget(self.position_label)
        
        # Apply styling
        self.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
            }
        """)
    
    def set_image_data(self, image_data):
        """Set the image data for this view."""
        if image_data is None:
            return
        
        self.image_data = image_data
        
        # Update slice range based on orientation
        if self.orientation == ViewOrientation.AXIAL:
            self.max_slice = image_data.shape[0] - 1
        elif self.orientation == ViewOrientation.SAGITTAL:
            self.max_slice = image_data.shape[2] - 1
        elif self.orientation == ViewOrientation.CORONAL:
            self.max_slice = image_data.shape[1] - 1
        
        # Update slider range
        self.slice_slider.setMaximum(self.max_slice)
        self.slice_spinbox.setMaximum(self.max_slice)
        
        # Set slice to middle by default
        self.set_slice_index(self.max_slice // 2)
    
    def set_slice_index(self, index):
        """Set the current slice index."""
        if not self.image_data is not None:
            return
        
        # Ensure index is within bounds
        index = max(0, min(index, self.max_slice))
        
        if index != self.slice_index:
            self.slice_index = index
            
            # Update slider and spinbox if they don't match
            if self.slice_slider.value() != index:
                self.slice_slider.setValue(index)
            if self.slice_spinbox.value() != index:
                self.slice_spinbox.setValue(index)
            
            # Extract slice data
            self.update_slice_data()
            
            # Emit signal
            self.sliceChanged.emit(index)
        
        # Always update display
        self.update_display()
    
    def update_slice_data(self):
        """Update the current slice data based on orientation."""
        if self.image_data is None:
            return
        
        try:
            if self.orientation == ViewOrientation.AXIAL:
                self.slice_data = self.image_data[self.slice_index, :, :]
            elif self.orientation == ViewOrientation.SAGITTAL:
                self.slice_data = self.image_data[:, :, self.slice_index]
            elif self.orientation == ViewOrientation.CORONAL:
                self.slice_data = self.image_data[:, self.slice_index, :]
        except IndexError:
            logger.error(f"Index error accessing slice {self.slice_index} in orientation {self.orientation}")
            self.slice_data = np.zeros((10, 10), dtype=np.int16)
    
    def set_window_level(self, window, level):
        """Set the window width and level."""
        self.window_width = max(1, window)
        self.window_level = level
        
        # Update display
        self.update_display()
        
        # Emit signal
        self.windowLevelChanged.emit(window, level)
    
    def set_zoom(self, zoom):
        """Set the zoom level."""
        self.zoom = max(0.1, zoom)
        
        # Update display
        self.update_display()
        
        # Emit signal
        self.zoomChanged.emit(zoom)
    
    def update_display(self):
        """Update the display image and repaint."""
        if self.slice_data is None:
            return
        
        # Convert slice data to display image with windowing
        self.display_image = self.apply_window_level()
        
        # Create pixmap from display image
        self.pixmap = QPixmap.fromImage(self.display_image)
        
        # Request repaint
        self.update()
    
    def apply_window_level(self):
        """Apply window/level to the slice data and return a QImage."""
        if self.slice_data is None:
            return QImage()
        
        # Apply window/level
        low = self.window_level - self.window_width // 2
        high = self.window_level + self.window_width // 2
        
        # Clip values
        display_data = np.clip(self.slice_data, low, high)
        
        # Normalize to 0-255
        if high > low:
            display_data = ((display_data - low) / (high - low) * 255).astype(np.uint8)
        else:
            display_data = np.zeros_like(self.slice_data, dtype=np.uint8)
        
        # Create QImage
        height, width = display_data.shape
        bytes_per_line = width
        
        # Convert to RGBA format (grayscale with alpha channel)
        rgba_data = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_data[..., 0] = display_data  # R
        rgba_data[..., 1] = display_data  # G
        rgba_data[..., 2] = display_data  # B
        rgba_data[..., 3] = 255           # A
        
        qimage = QImage(rgba_data.data, width, height, width * 4, QImage.Format_RGBA8888)
        
        return qimage
    
    def add_structure_overlay(self, structure_id, structure, color, is_selected=False):
        """Add a structure overlay to the view."""
        self.structure_overlays[structure_id] = (structure, color, is_selected)
        self.update()
    
    def remove_structure_overlay(self, structure_id):
        """Remove a structure overlay from the view."""
        if structure_id in self.structure_overlays:
            del self.structure_overlays[structure_id]
            self.update()
    
    def add_temp_overlay(self, overlay):
        """Add a temporary overlay (e.g., for drawing operations)."""
        self.temp_overlays.append(overlay)
        self.update()
    
    def clear_temp_overlays(self):
        """Clear all temporary overlays."""
        self.temp_overlays.clear()
        self.update()
    
    def clear_all_overlays(self):
        """Clear all overlays."""
        self.structure_overlays.clear()
        self.temp_overlays.clear()
        self.measurement_overlays.clear()
        self.annotation_overlays.clear()
        self.update()
    
    def draw_structure_overlays(self, painter):
        """Draw structure overlays on the view."""
        if not self.structure_overlays:
            return
        
        for structure_id, (structure, color, is_selected) in self.structure_overlays.items():
            if not structure.visible:
                continue
            
            # Get contours for current slice and orientation
            contours = structure.get_contours_for_slice(self.slice_index, self.orientation_to_int())
            
            if not contours:
                continue
            
            # Set up painter for this structure
            line_width = 2 if is_selected else 1
            alpha = 128  # Semi-transparent
            
            # Draw filled contours
            painter.setPen(Qt.NoPen)
            fill_color = QColor(color)
            fill_color.setAlpha(40)  # Very transparent for fill
            painter.setBrush(QBrush(fill_color))
            
            for contour in contours:
                # Convert contour points to view coordinates
                points = [self.image_to_view(QPoint(int(x), int(y))) for x, y in contour]
                
                # Draw polygon
                painter.drawPolygon(points)
            
            # Draw contour outlines
            painter.setBrush(Qt.NoBrush)
            outline_color = QColor(color)
            outline_color.setAlpha(alpha)
            painter.setPen(QPen(outline_color, line_width))
            
            for contour in contours:
                # Convert contour points to view coordinates
                points = [self.image_to_view(QPoint(int(x), int(y))) for x, y in contour]
                
                # Draw polyline (closed)
                painter.drawPolygon(points)
    
    def draw_temp_overlays(self, painter):
        """Draw temporary overlays on the view."""
        if not self.temp_overlays:
            return
        
        # Draw each temporary overlay
        for overlay in self.temp_overlays:
            # Overlay might be a mask or a list of points, handle accordingly
            if isinstance(overlay, np.ndarray):
                # Convert mask to RGBA image
                height, width = overlay.shape
                rgba_data = np.zeros((height, width, 4), dtype=np.uint8)
                
                # Highlight pixels where mask is True
                mask_color = QColor(255, 0, 0, 128)  # Semi-transparent red
                rgba_data[overlay > 0, 0] = mask_color.red()
                rgba_data[overlay > 0, 1] = mask_color.green()
                rgba_data[overlay > 0, 2] = mask_color.blue()
                rgba_data[overlay > 0, 3] = mask_color.alpha()
                
                # Create QImage and draw it
                overlay_image = QImage(rgba_data.data, width, height, width * 4, QImage.Format_RGBA8888)
                
                # Scale and position to match the view
                dest_rect = self.get_image_rect()
                painter.drawImage(dest_rect, overlay_image)
            
            elif isinstance(overlay, list):
                # Assume it's a list of points to draw
                if not overlay:
                    continue
                
                # Convert points to view coordinates
                points = [self.image_to_view(QPoint(int(x), int(y))) for x, y in overlay]
                
                # Draw polyline
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawPolyline(points)
    
    def paintEvent(self, event):
        """Handle paint events to render the view."""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # Draw image
        if not self.pixmap.isNull():
            # Calculate scaled image rectangle
            img_rect = self.get_image_rect()
            
            # Draw the image
            painter.drawPixmap(img_rect, self.pixmap)
            
            # Draw structure overlays
            self.draw_structure_overlays(painter)
            
            # Draw temporary overlays
            self.draw_temp_overlays(painter)
            
            # Draw crosshairs
            self.draw_crosshairs(painter)
        else:
            # Draw a placeholder
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image data")
    
    def draw_crosshairs(self, painter):
        """Draw crosshairs on the view."""
        # Get center of view in view coordinates
        center = self.rect().center()
        
        # Draw crosshairs
        painter.setPen(QPen(QColor(0, 255, 0), 1, Qt.DashLine))
        
        # Horizontal line
        painter.drawLine(0, center.y(), self.width(), center.y())
        
        # Vertical line
        painter.drawLine(center.x(), 0, center.x(), self.height())
    
    def get_image_rect(self):
        """Get the rectangle for displaying the image with current zoom and pan."""
        if self.display_image.isNull():
            return QRect()
        
        # Calculate scaled image size
        img_width = int(self.display_image.width() * self.zoom)
        img_height = int(self.display_image.height() * self.zoom)
        
        # Center image in view
        x = (self.view_frame.width() - img_width) // 2 + self.pan_offset.x()
        y = (self.view_frame.height() - img_height) // 2 + self.pan_offset.y()
        
        return QRect(x, y, img_width, img_height)
    
    def image_to_view(self, image_pos):
        """Convert image coordinates to view coordinates."""
        img_rect = self.get_image_rect()
        
        # Calculate scaling factor
        if self.display_image.width() > 0 and self.display_image.height() > 0:
            scale_x = img_rect.width() / self.display_image.width()
            scale_y = img_rect.height() / self.display_image.height()
            
            # Convert coordinates
            view_x = img_rect.x() + int(image_pos.x() * scale_x)
            view_y = img_rect.y() + int(image_pos.y() * scale_y)
            
            return QPoint(view_x, view_y)
        
        return QPoint(0, 0)
    
    def view_to_image(self, view_pos):
        """Convert view coordinates to image coordinates."""
        img_rect = self.get_image_rect()
        
        # Check if point is within image bounds
        if not img_rect.contains(view_pos):
            return QPoint(-1, -1)
        
        # Calculate image coordinates
        if img_rect.width() > 0 and img_rect.height() > 0:
            img_x = int((view_pos.x() - img_rect.x()) * self.display_image.width() / img_rect.width())
            img_y = int((view_pos.y() - img_rect.y()) * self.display_image.height() / img_rect.height())
            
            return QPoint(img_x, img_y)
        
        return QPoint(-1, -1)
    
    def orientation_to_int(self):
        """Convert ViewOrientation to integer for structure methods."""
        if self.orientation == ViewOrientation.AXIAL:
            return 0
        elif self.orientation == ViewOrientation.SAGITTAL:
            return 1
        elif self.orientation == ViewOrientation.CORONAL:
            return 2
        return 0
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        super().mousePressEvent(event)
        
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier):
            # Start panning
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            # Convert to image coordinates
            img_pos = self.view_to_image(event.pos())
            
            # Emit signal
            self.mousePressed.emit(event.pos(), img_pos)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        super().mouseMoveEvent(event)
        
        # Handle panning
        if self.is_panning:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.pos()
            self.update()
        
        # Update position display
        if self.display_image.isNull():
            return
        
        img_pos = self.view_to_image(event.pos())
        
        if img_pos.x() >= 0 and img_pos.y() >= 0 and img_pos.x() < self.display_image.width() and img_pos.y() < self.display_image.height():
            # Get value at position
            value = 0
            if self.slice_data is not None:
                try:
                    value = self.slice_data[img_pos.y(), img_pos.x()]
                except IndexError:
                    pass
            
            # Update position label
            self.position_label.setText(f"X: {img_pos.x()} Y: {img_pos.y()} Value: {value}")
            
            # Emit signal
            self.mouseMoved.emit(event.pos(), img_pos)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        super().mouseReleaseEvent(event)
        
        if self.is_panning and (event.button() == Qt.MiddleButton or event.button() == Qt.LeftButton):
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        
        # Convert to image coordinates
        img_pos = self.view_to_image(event.pos())
        
        # Emit signal
        self.mouseReleased.emit(event.pos(), img_pos)
    
    def wheelEvent(self, event):
        """Handle mouse wheel events."""
        super().wheelEvent(event)
        
        if event.modifiers() & Qt.ControlModifier:
            # Zoom
            delta = event.angleDelta().y() / 120.0
            new_zoom = self.zoom * (1.0 + delta * 0.1)
            self.set_zoom(new_zoom)
        else:
            # Change slice
            delta = event.angleDelta().y() / 120.0
            new_index = int(self.slice_index - delta)
            self.set_slice_index(new_index)
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        super().keyPressEvent(event)
        
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # Zoom in
            self.set_zoom(self.zoom * 1.1)
        elif event.key() == Qt.Key_Minus:
            # Zoom out
            self.set_zoom(self.zoom / 1.1)
        elif event.key() == Qt.Key_Left:
            # Previous slice
            self.set_slice_index(self.slice_index - 1)
        elif event.key() == Qt.Key_Right:
            # Next slice
            self.set_slice_index(self.slice_index + 1)
        elif event.key() == Qt.Key_R:
            # Reset view
            self.pan_offset = QPoint(0, 0)
            self.zoom = 1.0
            self.update()
        elif event.key() == Qt.Key_Space:
            # Reset window/level to defaults
            self.set_window_level(400, 40)

class MPRViewer(QWidget):
    """
    Multi-planar reconstruction viewer widget.
    
    Displays axial, sagittal, and coronal views of a 3D volume with
    synchronized navigation and structure overlays.
    """
    
    # Signals
    sliceChanged = pyqtSignal(int, ViewOrientation)
    orientationChanged = pyqtSignal(ViewOrientation)
    zoomChanged = pyqtSignal(float)
    windowLevelChanged = pyqtSignal(int, int)  # window, level
    mousePressed = pyqtSignal(int, QPoint, QPoint)  # view id, view pos, image pos
    mouseMoved = pyqtSignal(int, QPoint, QPoint)    # view id, view pos, image pos
    mouseReleased = pyqtSignal(int, QPoint, QPoint) # view id, view pos, image pos
    
    def __init__(self, parent=None):
        """Initialize MPR viewer with three views."""
        super().__init__(parent)
        
        # Viewer settings
        self.image = None
        self.views = {}
        self.current_view_id = 0
        self.current_orientation = ViewOrientation.AXIAL
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Create toolbar
        self.create_toolbar()
        layout.addWidget(self.toolbar)
        
        # Create views container
        views_container = QWidget()
        views_layout = QGridLayout(views_container)
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.setSpacing(5)
        
        # Create views
        self.views[0] = MPRView(ViewOrientation.AXIAL)
        self.views[1] = MPRView(ViewOrientation.CORONAL)
        self.views[2] = MPRView(ViewOrientation.SAGITTAL)
        
        # Add views to grid layout
        views_layout.addWidget(self.views[0], 0, 0)
        views_layout.addWidget(self.views[1], 0, 1)
        views_layout.addWidget(self.views[2], 1, 0)
        
        # Add view container to layout
        layout.addWidget(views_container, 1)
        
        # Set up connections
        self.setup_connections()
        
        # Apply styling
        self.apply_styling()
    
    def create_toolbar(self):
        """Create the toolbar with MPR controls."""
        self.toolbar = QToolBar("MPR Controls")
        self.toolbar.setIconSize(QSize(20, 20))
        
        # Window/level controls
        self.toolbar.addWidget(QLabel("W:"))
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setRange(1, 4000)
        self.window_spinbox.setValue(400)
        self.window_spinbox.setSingleStep(10)
        self.window_spinbox.valueChanged.connect(self.on_window_changed)
        self.toolbar.addWidget(self.window_spinbox)
        
        self.toolbar.addWidget(QLabel("L:"))
        self.level_spinbox = QSpinBox()
        self.level_spinbox.setRange(-1000, 3000)
        self.level_spinbox.setValue(40)
        self.level_spinbox.setSingleStep(10)
        self.level_spinbox.valueChanged.connect(self.on_level_changed)
        self.toolbar.addWidget(self.level_spinbox)
        
        # Preset window/level menu
        self.preset_button = QToolButton()
        self.preset_button.setText("Presets")
        self.preset_button.setPopupMode(QToolButton.InstantPopup)
        
        preset_menu = QMenu()
        presets = [
            ("Default", 400, 40),
            ("CT Lung", 1500, -600),
            ("CT Bone", 2000, 500),
            ("CT Abdomen", 400, 50),
            ("CT Brain", 80, 40),
            ("MR T1", 500, 300),
            ("MR T2", 1000, 500),
            ("PET", 10000, 5000)
        ]
        
        for name, window, level in presets:
            action = preset_menu.addAction(name)
            action.setData((window, level))
        
        preset_menu.triggered.connect(self.on_preset_selected)
        self.preset_button.setMenu(preset_menu)
        self.toolbar.addWidget(self.preset_button)
        
        # Add spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        
        # Zoom controls
        self.toolbar.addWidget(QLabel("Zoom:"))
        self.zoom_spinbox = QDoubleSpinBox()
        self.zoom_spinbox.setRange(0.1, 10.0)
        self.zoom_spinbox.setValue(1.0)
        self.zoom_spinbox.setSingleStep(0.1)
        self.zoom_spinbox.valueChanged.connect(self.on_zoom_changed)
        self.toolbar.addWidget(self.zoom_spinbox)
        
        # Reset view button
        reset_button = QToolButton()
        reset_button.setText("Reset")
        reset_button.clicked.connect(self.reset_view)
        self.toolbar.addWidget(reset_button)
    
    def setup_connections(self):
        """Set up signal-slot connections between components."""
        # Connect view signals to MPR viewer signals
        for view_id, view in self.views.items():
            # Forward mouse events with view_id
            view.mousePressed.connect(lambda view_pos, img_pos, id=view_id: 
                                    self.mousePressed.emit(id, view_pos, img_pos))
            view.mouseMoved.connect(lambda view_pos, img_pos, id=view_id: 
                                   self.mouseMoved.emit(id, view_pos, img_pos))
            view.mouseReleased.connect(lambda view_pos, img_pos, id=view_id: 
                                      self.mouseReleased.emit(id, view_pos, img_pos))
            
            # Connect slice change signals
            view.sliceChanged.connect(lambda slice_idx, orientation=view.orientation: 
                                     self.sliceChanged.emit(slice_idx, orientation))
            
            # Connect window/level signals
            view.windowLevelChanged.connect(self.windowLevelChanged)
            
            # Connect zoom signals
            view.zoomChanged.connect(self.zoomChanged)
    
    def apply_styling(self):
        """Apply styling to the MPR viewer."""
        self.setStyleSheet("""
            QToolBar {
                background-color: #f0f0f0;
                border: none;
                padding: 2px;
            }
            
            QLabel {
                font-size: 10pt;
            }
            
            QSpinBox, QDoubleSpinBox {
                min-width: 70px;
                max-width: 100px;
            }
            
            QToolButton {
                background-color: #e0e0e0;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 3px;
            }
            
            QToolButton:hover {
                background-color: #d0d0d0;
            }
            
            QToolButton:pressed {
                background-color: #c0c0c0;
            }
        """)
    
    def set_image(self, image):
        """Set the image data for all views."""
        self.image = image
        
        if image is None or image.data is None:
            return
        
        # Set image data for each view
        for view in self.views.values():
            view.set_image_data(image.data)
        
        # Reset view settings
        self.reset_view()
    
    def on_window_changed(self, value):
        """Handle window width spinbox changes."""
        window = value
        level = self.level_spinbox.value()
        
        # Update all views
        for view in self.views.values():
            view.set_window_level(window, level)
    
    def on_level_changed(self, value):
        """Handle window level spinbox changes."""
        window = self.window_spinbox.value()
        level = value
        
        # Update all views
        for view in self.views.values():
            view.set_window_level(window, level)
    
    def on_preset_selected(self, action):
        """Handle window/level preset selection."""
        window, level = action.data()
        
        # Update spinboxes
        self.window_spinbox.setValue(window)
        self.level_spinbox.setValue(level)
        
        # Update all views
        for view in self.views.values():
            view.set_window_level(window, level)
    
    def on_zoom_changed(self, value):
        """Handle zoom spinbox changes."""
        # Update all views
        for view in self.views.values():
            view.set_zoom(value)
    
    def reset_view(self):
        """Reset all view parameters to defaults."""
        # Reset window/level
        self.window_spinbox.setValue(400)
        self.level_spinbox.setValue(40)
        
        # Reset zoom
        self.zoom_spinbox.setValue(1.0)
        
        # Reset pan for all views
        for view in self.views.values():
            view.pan_offset = QPoint(0, 0)
            view.update()
    
    def set_current_orientation(self, orientation):
        """Set the current active orientation."""
        self.current_orientation = orientation
        self.orientationChanged.emit(orientation)
    
    def get_current_slice_index(self):
        """Get the slice index for the current orientation."""
        for view in self.views.values():
            if view.orientation == self.current_orientation:
                return view.slice_index
        return 0
    
    def add_structure_overlay(self, structure_id, structure, color, is_selected=False):
        """Add a structure overlay to all views."""
        for view in self.views.values():
            view.add_structure_overlay(structure_id, structure, color, is_selected)
    
    def remove_structure_overlay(self, structure_id):
        """Remove a structure overlay from all views."""
        for view in self.views.values():
            view.remove_structure_overlay(structure_id)
    
    def add_temp_overlay(self, overlay):
        """Add a temporary overlay to the current view."""
        # Find view with current orientation
        for view in self.views.values():
            if view.orientation == self.current_orientation:
                view.add_temp_overlay(overlay)
                break
    
    def clear_temp_overlays(self):
        """Clear all temporary overlays."""
        for view in self.views.values():
            view.clear_temp_overlays()
    
    def clear_all_overlays(self):
        """Clear all overlays from all views."""
        for view in self.views.values():
            view.clear_all_overlays()
    
    def update_view(self, view_id):
        """Update a specific view."""
        if view_id in self.views:
            self.views[view_id].update()
    
    def update_all_views(self):
        """Update all views."""
        for view in self.views.values():
            view.update()
    
    def set_cursor(self, cursor):
        """Set the cursor for all views."""
        for view in self.views.values():
            view.setCursor(cursor)

def test_mpr_viewer():
    """Test function for the MPR viewer."""
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    
    # Create test data
    class TestImage:
        def __init__(self):
            # Create a simple test volume
            self.shape = (100, 512, 512)
            self.data = np.zeros(self.shape, dtype=np.float32)
            
            # Add some shapes for visualization
            # Sphere
            center = np.array([50, 256, 256])
            radius = 100
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[2]):
                        dist = np.sqrt(np.sum(np.square(np.array([i, j, k]) - center)))
                        if dist < radius:
                            self.data[i, j, k] = 1000 - dist * 2
                            
            # Add some noise
            self.data += np.random.normal(0, 10, self.shape)
            
    class TestStructure:
        def __init__(self, name, color):
            self.name = name
            self.color = color
            self.visible = True
            self.type = "ORGAN"
            
    class TestStructureSet:
        def __init__(self):
            self.structures = [
                TestStructure("PTV", "#FF0000"),
                TestStructure("CTV", "#00FF00"),
                TestStructure("OAR", "#0000FF")
            ]
    
    # Create main window
    main_window = QMainWindow()
    
    # Create test data
    test_image = TestImage()
    test_structure_set = TestStructureSet()
    
    # Create MPR viewer
    mpr_viewer = MPRViewer()
    mpr_viewer.set_image(test_image)
    
    # Set as central widget
    main_window.setCentralWidget(mpr_viewer)
    main_window.setWindowTitle("QuangTPS - MPR Viewer")
    main_window.resize(1200, 800)
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_mpr_viewer() 