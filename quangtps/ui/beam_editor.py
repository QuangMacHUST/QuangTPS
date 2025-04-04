#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beam Editor Module
================

This module provides a widget for editing beam parameters in radiotherapy
treatment planning, with functionality similar to Eclipse's field editor.
"""

import os
import logging
import math
from typing import List, Dict, Optional, Any, Tuple, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QComboBox,
    QTabWidget, QSplitter, QFrame, QCheckBox, QSlider, QToolButton,
    QGridLayout, QSizePolicy
)
from PyQt5.QtGui import QColor, QIcon, QFont, QPixmap, QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRect, QPointF, QPainterPath

# Import local modules if they exist
try:
    from quangtps.beam.beam import Beam
    from quangtps.beam.beam_modifiers import Wedge, Block, MLC
    from quangtps.planning.beam_set import BeamSet
except ImportError:
    logging.warning("Failed to import QuangTPS beam modules in beam editor")

logger = logging.getLogger(__name__)


class BeamDiagramWidget(QWidget):
    """Widget for displaying a 2D diagram of the beam's eye view."""
    
    def __init__(self, parent=None):
        """Initialize the beam diagram widget."""
        super().__init__(parent)
        
        # Initialize variables
        self.beam = None
        self.scale_factor = 1.0
        self.show_mlc = True
        self.show_blocks = True
        self.show_contours = True
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(palette)
    
    def set_beam(self, beam):
        """Set the beam to display."""
        self.beam = beam
        self.update()
    
    def set_show_mlc(self, show):
        """Set whether to show MLC leaves."""
        self.show_mlc = show
        self.update()
    
    def set_show_blocks(self, show):
        """Set whether to show blocks."""
        self.show_blocks = show
        self.update()
    
    def set_show_contours(self, show):
        """Set whether to show structure contours."""
        self.show_contours = show
        self.update()
    
    def paintEvent(self, event):
        """Paint the beam diagram."""
        if not self.beam:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set up coordinate system
        # Origin at center, X to the right, Y upward
        width, height = self.width(), self.height()
        center_x, center_y = width // 2, height // 2
        
        # Scale factor (pixels per cm)
        max_field_dim = max(self.beam.x_field_size, self.beam.y_field_size)
        if max_field_dim > 0:
            self.scale_factor = min(width, height) * 0.8 / max_field_dim
        else:
            self.scale_factor = 10  # Default scale factor
        
        # Transform coordinates
        painter.translate(center_x, center_y)
        painter.scale(self.scale_factor, -self.scale_factor)  # Flip Y axis
        
        # Draw field outline
        x_half = self.beam.x_field_size / 2
        y_half = self.beam.y_field_size / 2
        field_rect = QRectF(
            -x_half, -y_half,
            self.beam.x_field_size, self.beam.y_field_size
        )
        
        painter.setPen(QPen(Qt.black, 0.1))
        painter.setBrush(QBrush(QColor(240, 240, 240)))
        painter.drawRect(field_rect)
        
        # Draw MLC if enabled and available
        if self.show_mlc:
            self._draw_mlc(painter)
        
        # Draw blocks if enabled and available
        if self.show_blocks:
            self._draw_blocks(painter)
        
        # Draw field boundary
        painter.setPen(QPen(Qt.black, 0.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(field_rect)
        
        # Draw central axis
        painter.setPen(QPen(Qt.red, 0.1))
        painter.drawLine(-x_half, 0, x_half, 0)  # X axis
        painter.drawLine(0, -y_half, 0, y_half)  # Y axis
        
        # Draw isocenter marker
        painter.setPen(QPen(Qt.red, 0.1))
        painter.setBrush(QBrush(Qt.red))
        painter.drawEllipse(QPointF(0, 0), 0.2, 0.2)  # 4mm diameter
    
    def _draw_mlc(self, painter):
        """Draw MLC leaves in the beam diagram."""
        # Check if beam has MLC
        mlc = self.beam.mlc if hasattr(self.beam, 'mlc') else None
        if not mlc:
            return
        
        # Draw MLC leaves
        painter.setPen(QPen(Qt.darkGray, 0.05))
        painter.setBrush(QBrush(QColor(200, 200, 200, 180)))
        
        # Simplified MLC drawing - would be replaced with actual MLC geometry
        leaf_positions = getattr(mlc, 'leaf_positions', [])
        if leaf_positions:
            leaf_width = self.beam.y_field_size / len(leaf_positions)
            y_start = -self.beam.y_field_size / 2
            
            for i, (left, right) in enumerate(leaf_positions):
                y = y_start + i * leaf_width
                
                # Draw left leaf
                if left > -self.beam.x_field_size / 2:
                    painter.drawRect(
                        -self.beam.x_field_size / 2, y,
                        left + self.beam.x_field_size / 2, leaf_width
                    )
                
                # Draw right leaf
                if right < self.beam.x_field_size / 2:
                    painter.drawRect(
                        right, y,
                        self.beam.x_field_size / 2 - right, leaf_width
                    )
    
    def _draw_blocks(self, painter):
        """Draw blocks in the beam diagram."""
        # Check if beam has blocks
        blocks = self.beam.blocks if hasattr(self.beam, 'blocks') else []
        if not blocks:
            return
        
        # Draw each block
        painter.setPen(QPen(Qt.black, 0.1))
        painter.setBrush(QBrush(QColor(100, 100, 100, 200)))
        
        for block in blocks:
            # Simplified block drawing - would be replaced with actual block geometry
            points = getattr(block, 'points', [])
            if points and len(points) >= 3:
                path = QPainterPath()
                path.moveTo(points[0][0], points[0][1])
                for x, y in points[1:]:
                    path.lineTo(x, y)
                path.closeSubpath()
                painter.drawPath(path)


class BeamEditor(QWidget):
    """
    Widget for editing beam parameters.
    
    This widget provides a UI for editing all aspects of a beam, including
    geometry, modifiers (wedges, MLCs, etc.), and dosimetric properties.
    """
    
    # Signals
    beamChanged = pyqtSignal(object)
    
    def __init__(self, parent=None):
        """Initialize the beam editor widget."""
        super().__init__(parent)
        
        # Initialize variables
        self.beam = None
        self.beam_set = None
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Beam name and energy header
        header_layout = QHBoxLayout()
        
        self.beam_name_label = QLabel("No Beam Selected")
        self.beam_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.beam_energy_label = QLabel("")
        
        header_layout.addWidget(self.beam_name_label)
        header_layout.addStretch()
        header_layout.addWidget(self.beam_energy_label)
        
        main_layout.addLayout(header_layout)
        
        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)
        
        # Main content splitter (diagram + parameters)
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Beam diagram
        diagram_container = QWidget()
        diagram_layout = QVBoxLayout(diagram_container)
        diagram_layout.setContentsMargins(0, 0, 0, 0)
        
        self.diagram_widget = BeamDiagramWidget()
        
        # Diagram controls
        diagram_controls = QHBoxLayout()
        
        self.show_mlc_checkbox = QCheckBox("Show MLC")
        self.show_mlc_checkbox.setChecked(True)
        self.show_mlc_checkbox.toggled.connect(
            self.diagram_widget.set_show_mlc
        )
        
        self.show_blocks_checkbox = QCheckBox("Show Blocks")
        self.show_blocks_checkbox.setChecked(True)
        self.show_blocks_checkbox.toggled.connect(
            self.diagram_widget.set_show_blocks
        )
        
        self.show_contours_checkbox = QCheckBox("Show Contours")
        self.show_contours_checkbox.setChecked(True)
        self.show_contours_checkbox.toggled.connect(
            self.diagram_widget.set_show_contours
        )
        
        diagram_controls.addWidget(self.show_mlc_checkbox)
        diagram_controls.addWidget(self.show_blocks_checkbox)
        diagram_controls.addWidget(self.show_contours_checkbox)
        
        diagram_layout.addWidget(self.diagram_widget)
        diagram_layout.addLayout(diagram_controls)
        
        # Right side: Parameters
        params_container = QWidget()
        params_layout = QVBoxLayout(params_container)
        params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Parameters tabs
        params_tabs = QTabWidget()
        
        # Geometry tab
        geometry_tab = QWidget()
        geometry_layout = QFormLayout(geometry_tab)
        
        # Basic geometry parameters
        self.gantry_angle_spin = QDoubleSpinBox()
        self.gantry_angle_spin.setRange(0, 360)
        self.gantry_angle_spin.setDecimals(1)
        self.gantry_angle_spin.setSingleStep(10)
        self.gantry_angle_spin.setToolTip("Gantry angle in degrees")
        self.gantry_angle_spin.valueChanged.connect(self.on_gantry_angle_changed)
        
        self.collimator_angle_spin = QDoubleSpinBox()
        self.collimator_angle_spin.setRange(0, 360)
        self.collimator_angle_spin.setDecimals(1)
        self.collimator_angle_spin.setSingleStep(10)
        self.collimator_angle_spin.setToolTip("Collimator angle in degrees")
        self.collimator_angle_spin.valueChanged.connect(self.on_collimator_angle_changed)
        
        self.couch_angle_spin = QDoubleSpinBox()
        self.couch_angle_spin.setRange(0, 360)
        self.couch_angle_spin.setDecimals(1)
        self.couch_angle_spin.setSingleStep(10)
        self.couch_angle_spin.setToolTip("Couch angle in degrees")
        self.couch_angle_spin.valueChanged.connect(self.on_couch_angle_changed)
        
        self.x_field_size_spin = QDoubleSpinBox()
        self.x_field_size_spin.setRange(0, 40)
        self.x_field_size_spin.setDecimals(1)
        self.x_field_size_spin.setSingleStep(1)
        self.x_field_size_spin.setToolTip("Field size in X direction (cm)")
        self.x_field_size_spin.valueChanged.connect(self.on_x_field_size_changed)
        
        self.y_field_size_spin = QDoubleSpinBox()
        self.y_field_size_spin.setRange(0, 40)
        self.y_field_size_spin.setDecimals(1)
        self.y_field_size_spin.setSingleStep(1)
        self.y_field_size_spin.setToolTip("Field size in Y direction (cm)")
        self.y_field_size_spin.valueChanged.connect(self.on_y_field_size_changed)
        
        self.ssd_spin = QDoubleSpinBox()
        self.ssd_spin.setRange(70, 150)
        self.ssd_spin.setDecimals(1)
        self.ssd_spin.setSingleStep(1)
        self.ssd_spin.setToolTip("Source-to-Surface Distance (cm)")
        self.ssd_spin.valueChanged.connect(self.on_ssd_changed)
        
        geometry_layout.addRow("Gantry Angle (°):", self.gantry_angle_spin)
        geometry_layout.addRow("Collimator Angle (°):", self.collimator_angle_spin)
        geometry_layout.addRow("Couch Angle (°):", self.couch_angle_spin)
        geometry_layout.addRow("X Field Size (cm):", self.x_field_size_spin)
        geometry_layout.addRow("Y Field Size (cm):", self.y_field_size_spin)
        geometry_layout.addRow("SSD (cm):", self.ssd_spin)
        
        # Dose tab
        dose_tab = QWidget()
        dose_layout = QFormLayout(dose_tab)
        
        self.beam_weight_spin = QDoubleSpinBox()
        self.beam_weight_spin.setRange(0, 1000)
        self.beam_weight_spin.setDecimals(1)
        self.beam_weight_spin.setSingleStep(10)
        self.beam_weight_spin.setToolTip("Beam weight in monitor units (MU)")
        self.beam_weight_spin.valueChanged.connect(self.on_beam_weight_changed)
        
        self.isocenter_dose_spin = QDoubleSpinBox()
        self.isocenter_dose_spin.setRange(0, 1000)
        self.isocenter_dose_spin.setDecimals(2)
        self.isocenter_dose_spin.setSingleStep(1)
        self.isocenter_dose_spin.setToolTip("Dose at isocenter (Gy)")
        self.isocenter_dose_spin.valueChanged.connect(self.on_isocenter_dose_changed)
        
        dose_layout.addRow("Beam Weight (MU):", self.beam_weight_spin)
        dose_layout.addRow("Isocenter Dose (Gy):", self.isocenter_dose_spin)
        
        # Modifiers tab
        modifiers_tab = QWidget()
        modifiers_layout = QVBoxLayout(modifiers_tab)
        
        # Wedge section
        wedge_group = QGroupBox("Wedge")
        wedge_layout = QFormLayout(wedge_group)
        
        self.wedge_checkbox = QCheckBox("Use Wedge")
        self.wedge_checkbox.toggled.connect(self.on_wedge_toggled)
        
        self.wedge_angle_spin = QDoubleSpinBox()
        self.wedge_angle_spin.setRange(0, 60)
        self.wedge_angle_spin.setDecimals(0)
        self.wedge_angle_spin.setSingleStep(15)
        self.wedge_angle_spin.setToolTip("Wedge angle in degrees")
        self.wedge_angle_spin.valueChanged.connect(self.on_wedge_angle_changed)
        
        self.wedge_direction_combo = QComboBox()
        self.wedge_direction_combo.addItems(["IN", "OUT", "LEFT", "RIGHT"])
        self.wedge_direction_combo.setToolTip("Wedge direction")
        self.wedge_direction_combo.currentTextChanged.connect(self.on_wedge_direction_changed)
        
        wedge_layout.addRow(self.wedge_checkbox)
        wedge_layout.addRow("Angle (°):", self.wedge_angle_spin)
        wedge_layout.addRow("Direction:", self.wedge_direction_combo)
        
        modifiers_layout.addWidget(wedge_group)
        
        # MLC section
        mlc_group = QGroupBox("MLC")
        mlc_layout = QFormLayout(mlc_group)
        
        self.mlc_checkbox = QCheckBox("Use MLC")
        self.mlc_checkbox.toggled.connect(self.on_mlc_toggled)
        
        self.mlc_prescribe_button = QPushButton("Set MLC to PTV")
        self.mlc_prescribe_button.clicked.connect(self.on_mlc_prescribe)
        
        self.mlc_edit_button = QPushButton("Edit MLC...")
        self.mlc_edit_button.clicked.connect(self.on_mlc_edit)
        
        mlc_layout.addRow(self.mlc_checkbox)
        mlc_layout.addRow(self.mlc_prescribe_button)
        mlc_layout.addRow(self.mlc_edit_button)
        
        modifiers_layout.addWidget(mlc_group)
        
        # Add tabs to tab widget
        params_tabs.addTab(geometry_tab, "Geometry")
        params_tabs.addTab(dose_tab, "Dose")
        params_tabs.addTab(modifiers_tab, "Modifiers")
        
        params_layout.addWidget(params_tabs)
        
        # Add widgets to splitter
        content_splitter.addWidget(diagram_container)
        content_splitter.addWidget(params_container)
        
        # Set initial splitter sizes
        content_splitter.setSizes([400, 300])
        
        main_layout.addWidget(content_splitter)
        
        # Apply styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 8px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 3px;
                background-color: #f0f0f0;
            }
            
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 4px;
            }
            
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            
            QDoubleSpinBox, QSpinBox, QComboBox {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 2px;
                min-width: 70px;
            }
            
            QFrame[frameShape="4"] {
                color: #cccccc;
                margin: 5px 0;
            }
        """)
        
        # Initialize UI state
        self.update_ui_state()
    
    def set_beam(self, beam: 'Beam'):
        """
        Set the beam to edit.
        
        Args:
            beam: The beam to edit
        """
        self.beam = beam
        
        # Update UI with beam parameters
        self.update_from_beam()
        
        # Update diagram
        self.diagram_widget.set_beam(beam)
        
        # Update UI state
        self.update_ui_state()
    
    def set_beam_set(self, beam_set: 'BeamSet'):
        """
        Set the beam set.
        
        Args:
            beam_set: The beam set containing the beam
        """
        self.beam_set = beam_set
    
    def update_from_beam(self):
        """Update UI controls from the current beam."""
        if not self.beam:
            return
        
        # Block signals during update
        self._block_signals(True)
        
        # Update header
        self.beam_name_label.setText(self.beam.name if hasattr(self.beam, 'name') else "Unnamed Beam")
        
        energy_str = ""
        if hasattr(self.beam, 'energy') and self.beam.energy:
            energy_str = f"{self.beam.energy:.1f} MV"
        self.beam_energy_label.setText(energy_str)
        
        # Update geometry parameters
        self.gantry_angle_spin.setValue(
            self.beam.gantry_angle if hasattr(self.beam, 'gantry_angle') else 0
        )
        self.collimator_angle_spin.setValue(
            self.beam.collimator_angle if hasattr(self.beam, 'collimator_angle') else 0
        )
        self.couch_angle_spin.setValue(
            self.beam.couch_angle if hasattr(self.beam, 'couch_angle') else 0
        )
        self.x_field_size_spin.setValue(
            self.beam.x_field_size if hasattr(self.beam, 'x_field_size') else 10
        )
        self.y_field_size_spin.setValue(
            self.beam.y_field_size if hasattr(self.beam, 'y_field_size') else 10
        )
        self.ssd_spin.setValue(
            self.beam.ssd if hasattr(self.beam, 'ssd') else 100
        )
        
        # Update dose parameters
        self.beam_weight_spin.setValue(
            self.beam.weight if hasattr(self.beam, 'weight') else 100
        )
        
        isocenter_dose = 0
        if hasattr(self.beam, 'isocenter_dose'):
            isocenter_dose = self.beam.isocenter_dose
        self.isocenter_dose_spin.setValue(isocenter_dose)
        
        # Update modifiers
        # Wedge
        has_wedge = hasattr(self.beam, 'wedge') and self.beam.wedge is not None
        self.wedge_checkbox.setChecked(has_wedge)
        
        if has_wedge:
            wedge = self.beam.wedge
            self.wedge_angle_spin.setValue(
                wedge.angle if hasattr(wedge, 'angle') else 0
            )
            
            direction = wedge.direction if hasattr(wedge, 'direction') else "IN"
            index = self.wedge_direction_combo.findText(direction)
            if index >= 0:
                self.wedge_direction_combo.setCurrentIndex(index)
        
        # MLC
        has_mlc = hasattr(self.beam, 'mlc') and self.beam.mlc is not None
        self.mlc_checkbox.setChecked(has_mlc)
        
        # Unblock signals
        self._block_signals(False)
    
    def _block_signals(self, block):
        """
        Block or unblock signals from all UI controls.
        
        Args:
            block: Whether to block signals
        """
        for control in [
            self.gantry_angle_spin, self.collimator_angle_spin, self.couch_angle_spin,
            self.x_field_size_spin, self.y_field_size_spin, self.ssd_spin,
            self.beam_weight_spin, self.isocenter_dose_spin,
            self.wedge_checkbox, self.wedge_angle_spin, self.wedge_direction_combo,
            self.mlc_checkbox
        ]:
            control.blockSignals(block)
    
    def update_ui_state(self):
        """Update UI control states based on current conditions."""
        has_beam = self.beam is not None
        
        # Enable/disable all controls based on beam availability
        for control in [
            self.gantry_angle_spin, self.collimator_angle_spin, self.couch_angle_spin,
            self.x_field_size_spin, self.y_field_size_spin, self.ssd_spin,
            self.beam_weight_spin, self.isocenter_dose_spin,
            self.wedge_checkbox, self.mlc_checkbox
        ]:
            control.setEnabled(has_beam)
        
        # Wedge controls
        has_wedge = has_beam and self.wedge_checkbox.isChecked()
        self.wedge_angle_spin.setEnabled(has_wedge)
        self.wedge_direction_combo.setEnabled(has_wedge)
        
        # MLC controls
        has_mlc = has_beam and self.mlc_checkbox.isChecked()
        self.mlc_prescribe_button.setEnabled(has_mlc)
        self.mlc_edit_button.setEnabled(has_mlc)
    
    def on_gantry_angle_changed(self, value):
        """Handle gantry angle changes."""
        if not self.beam:
            return
        
        self.beam.gantry_angle = value
        self.beamChanged.emit(self.beam)
    
    def on_collimator_angle_changed(self, value):
        """Handle collimator angle changes."""
        if not self.beam:
            return
        
        self.beam.collimator_angle = value
        self.beamChanged.emit(self.beam)
    
    def on_couch_angle_changed(self, value):
        """Handle couch angle changes."""
        if not self.beam:
            return
        
        self.beam.couch_angle = value
        self.beamChanged.emit(self.beam)
    
    def on_x_field_size_changed(self, value):
        """Handle X field size changes."""
        if not self.beam:
            return
        
        self.beam.x_field_size = value
        self.diagram_widget.update()
        self.beamChanged.emit(self.beam)
    
    def on_y_field_size_changed(self, value):
        """Handle Y field size changes."""
        if not self.beam:
            return
        
        self.beam.y_field_size = value
        self.diagram_widget.update()
        self.beamChanged.emit(self.beam)
    
    def on_ssd_changed(self, value):
        """Handle SSD changes."""
        if not self.beam:
            return
        
        self.beam.ssd = value
        self.beamChanged.emit(self.beam)
    
    def on_beam_weight_changed(self, value):
        """Handle beam weight changes."""
        if not self.beam:
            return
        
        self.beam.weight = value
        self.beamChanged.emit(self.beam)
    
    def on_isocenter_dose_changed(self, value):
        """Handle isocenter dose changes."""
        if not self.beam:
            return
        
        self.beam.isocenter_dose = value
        self.beamChanged.emit(self.beam)
    
    def on_wedge_toggled(self, checked):
        """Handle wedge checkbox toggle."""
        if not self.beam:
            return
        
        if checked:
            # Create a new wedge if not present
            if not hasattr(self.beam, 'wedge') or self.beam.wedge is None:
                self.beam.wedge = Wedge()
                self.beam.wedge.angle = self.wedge_angle_spin.value()
                self.beam.wedge.direction = self.wedge_direction_combo.currentText()
        else:
            # Remove the wedge
            self.beam.wedge = None
        
        self.update_ui_state()
        self.diagram_widget.update()
        self.beamChanged.emit(self.beam)
    
    def on_wedge_angle_changed(self, value):
        """Handle wedge angle changes."""
        if not self.beam or not hasattr(self.beam, 'wedge') or self.beam.wedge is None:
            return
        
        self.beam.wedge.angle = value
        self.diagram_widget.update()
        self.beamChanged.emit(self.beam)
    
    def on_wedge_direction_changed(self, direction):
        """Handle wedge direction changes."""
        if not self.beam or not hasattr(self.beam, 'wedge') or self.beam.wedge is None:
            return
        
        self.beam.wedge.direction = direction
        self.diagram_widget.update()
        self.beamChanged.emit(self.beam)
    
    def on_mlc_toggled(self, checked):
        """Handle MLC checkbox toggle."""
        if not self.beam:
            return
        
        if checked:
            # Create a new MLC if not present
            if not hasattr(self.beam, 'mlc') or self.beam.mlc is None:
                self.beam.mlc = MLC()
                # Initialize with default leaf positions matching field size
                x_half = self.beam.x_field_size / 2
                y_half = self.beam.y_field_size / 2
                num_leaves = 60  # Default number of leaf pairs
                
                leaf_positions = []
                for i in range(num_leaves):
                    leaf_positions.append((-x_half, x_half))
                
                self.beam.mlc.leaf_positions = leaf_positions
        else:
            # Remove the MLC
            self.beam.mlc = None
        
        self.update_ui_state()
        self.diagram_widget.update()
        self.beamChanged.emit(self.beam)
    
    def on_mlc_prescribe(self):
        """Handle MLC prescribe button click."""
        # This would typically shape the MLC to the PTV in the beam's eye view
        # For now, we'll just show a placeholder message
        logging.info("MLC prescribed to PTV - placeholder")
    
    def on_mlc_edit(self):
        """Handle MLC edit button click."""
        # This would typically open an MLC editor dialog
        # For now, we'll just show a placeholder message
        logging.info("MLC edit dialog - placeholder")


def test_beam_editor():
    """Test function for the beam editor widget."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    # Create a dummy beam for testing
    class DummyBeam:
        def __init__(self):
            self.name = "Test Beam"
            self.energy = 6.0
            self.gantry_angle = 0.0
            self.collimator_angle = 0.0
            self.couch_angle = 0.0
            self.x_field_size = 10.0
            self.y_field_size = 10.0
            self.ssd = 100.0
            self.weight = 100.0
            self.isocenter_dose = 1.0
            self.mlc = None
            self.wedge = None
            self.blocks = []
    
    class DummyWedge:
        def __init__(self):
            self.angle = 30.0
            self.direction = "IN"
    
    class DummyMLC:
        def __init__(self):
            # Create some sample leaf positions
            self.leaf_positions = []
            for i in range(60):
                # Position leaves to form a circle
                r = 5.0  # 5cm radius
                y = -5.0 + i * 10.0 / 60
                x = math.sqrt(r*r - y*y) if abs(y) < r else 0
                self.leaf_positions.append((-x, x))
    
    app = QApplication(sys.argv)
    
    widget = BeamEditor()
    
    # Create a sample beam
    beam = DummyBeam()
    
    # Add a wedge
    beam.wedge = DummyWedge()
    
    # Add an MLC
    beam.mlc = DummyMLC()
    
    widget.set_beam(beam)
    widget.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    test_beam_editor() 