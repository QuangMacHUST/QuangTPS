#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eclipse-like collision detection dialog for QuangTPS.

This module provides a user interface for detecting and visualizing
potential collisions between the gantry, couch, and patient during
treatment delivery.
"""

import os
import sys
import logging
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
import importlib.util

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QGroupBox, QComboBox, QTabWidget, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QMessageBox,
    QDialogButtonBox, QRadioButton, QButtonGroup, QWidget
)
from PyQt5.QtGui import QColor, QPalette, QBrush, QFont, QPixmap, QPainter, QPen

# Check for PyVista and related dependencies
PYVISTA_AVAILABLE = importlib.util.find_spec("pyvista") is not None
PYVISTAQT_AVAILABLE = importlib.util.find_spec("pyvistaqt") is not None
VTK_AVAILABLE = importlib.util.find_spec("vtk") is not None

if PYVISTA_AVAILABLE and PYVISTAQT_AVAILABLE and VTK_AVAILABLE:
    import pyvista as pv
    from pyvistaqt import QtInteractor, BackgroundPlotter
    import vtk
else:
    pv = None
    QtInteractor = None
    BackgroundPlotter = None
    vtk = None

from quangtps.core.types import Plan, Patient, Structure, Beam
from quangtps.planning.collision_detection import CollisionDetector, create_collision_detector
from quangtps.core.logging import get_logger
from quangtps.ui.widgets.beam_angle_widget import BeamAngleWidget
from quangtps.ui.widgets.couch_widget import CouchWidget
from quangtps.ui.styles import Colors, get_icon
from quangtps.ui.dependency_installer import check_and_install_feature_dependencies

logger = get_logger(__name__)


class CollisionVisualization(QWidget):
    """
    Modern Eclipse-like 3D visualization of treatment room with collision detection.
    This widget uses PyVista for realistic 3D visualization of treatment machine,
    couch, and patient with accurate collision detection capabilities.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize visualization properties
        self.gantry_angle = 0.0
        self.couch_angle = 0.0
        self.collimator_angle = 0.0
        self.collision_detected = False
        self.warning_detected = False
        self.colliding_components = []
        self.beam_field_size = (100, 100)  # Default field size in mm
        self.machine_type = "TrueBeam"  # Default machine
        
        # Treatment machine parameters
        self.sad = 1000.0  # Source-to-axis distance (mm)
        self.sid = 1500.0  # Source-to-imager distance (mm)
        self.gantry_radius = 1000.0  # Gantry rotation radius (mm)
        
        # PyVista availability
        self.pyvista_available = PYVISTA_AVAILABLE and PYVISTAQT_AVAILABLE and VTK_AVAILABLE
        
        # Setup UI
        self._setup_ui()
        
        # Initialize PyVista plotter if available
        self._init_plotter()
        
    def _setup_ui(self):
        """Set up the user interface."""
        # Set minimum size
        self.setMinimumSize(500, 500)
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Container for 3D visualization
        self.plotter_container = QWidget(self)
        self.plotter_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, 
                                          QtWidgets.QSizePolicy.Expanding)
        self.plotter_layout = QVBoxLayout(self.plotter_container)
        self.plotter_layout.setContentsMargins(0, 0, 0, 0)
        
        self.main_layout.addWidget(self.plotter_container)
        
        # Status bar for collision information
        self.status_bar = QLabel("No collision detected")
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setStyleSheet("font-weight: bold; padding: 5px;")
        self.status_bar.setFixedHeight(30)
        self.main_layout.addWidget(self.status_bar)
        
        # If PyVista not available, show install button
        if not self.pyvista_available:
            self._show_dependency_message()
            
    def _init_plotter(self):
        """Initialize the PyVista 3D visualization plotter if dependencies are available."""
        if not self.pyvista_available:
            return
            
        try:
            # Create the PyVista QT interactor
            self.plotter = QtInteractor(self.plotter_container)
            self.plotter_layout.addWidget(self.plotter)
            
            # Configure plotter
            self.plotter.set_background("black")
            self.plotter.add_axes()
            self.plotter.show_grid()
            
            # Add lighting
            self.plotter.enable_shadows()
            light = pv.Light(position=(0, 0, 1), focal_point=(0, 0, 0), 
                           color=[1, 1, 1], intensity=0.8)
            self.plotter.add_light(light)
            
            # Set initial camera position
            self.plotter.camera_position = 'xy'  # Default anterior view
            
            # Create treatment room and machine geometries
            self._create_room_geometry()
            self._create_machine_geometry()
            self._create_patient_geometry()
            
            # Update visualization
            self.update_visualization()
            
        except Exception as e:
            logger.error(f"Error initializing PyVista plotter: {str(e)}")
            self.pyvista_available = False
            self._show_dependency_message()
    
    def _show_dependency_message(self):
        """Show a message about missing dependencies."""
        # Remove any existing widgets
        for i in reversed(range(self.plotter_layout.count())):
            item = self.plotter_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        
        # Create a message widget
        message_widget = QWidget()
        message_layout = QVBoxLayout(message_widget)
        
        label = QLabel("3D visualization requires additional dependencies")
        label.setAlignment(Qt.AlignCenter)
        
        install_button = QPushButton("Install 3D Visualization Dependencies")
        install_button.clicked.connect(self._install_dependencies)
        
        message_layout.addStretch(1)
        message_layout.addWidget(label)
        message_layout.addWidget(install_button)
        message_layout.addStretch(1)
        
        self.plotter_layout.addWidget(message_widget)
        
    def _install_dependencies(self):
        """Attempt to install PyVista and dependencies."""
        requirements = ["pyvista", "pyvistaqt", "vtk"]
        
        try:
            result = check_and_install_feature_dependencies("3d_visualization", self)
            if result:
                QMessageBox.information(
                    self, 
                    "Dependencies Installed", 
                    "3D visualization dependencies have been installed successfully. "
                    "Please restart the application to use the 3D visualization features."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Installation Failed",
                    "Failed to install 3D visualization dependencies. "
                    "Please try to install manually: pip install pyvista pyvistaqt vtk"
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Installation Error",
                f"An error occurred while installing dependencies: {str(e)}"
            )
            
    def _create_room_geometry(self):
        """Create the treatment room geometry."""
        if not self.pyvista_available:
            return
            
        # Create room floor
        floor = pv.Plane(center=(0, -700, 0), direction=(0, 1, 0), i_size=4000, j_size=4000)
        self.plotter.add_mesh(floor, color='#555555', opacity=0.3, name="floor")
        
        # Create ceiling
        ceiling = pv.Plane(center=(0, 1200, 0), direction=(0, -1, 0), i_size=4000, j_size=4000)
        self.plotter.add_mesh(ceiling, color='#555555', opacity=0.1, name="ceiling")
        
        # Create walls
        wall1 = pv.Plane(center=(2000, 0, 0), direction=(-1, 0, 0), i_size=2000, j_size=4000)
        self.plotter.add_mesh(wall1, color='#555555', opacity=0.1, name="wall1")
        
        wall2 = pv.Plane(center=(-2000, 0, 0), direction=(1, 0, 0), i_size=2000, j_size=4000)
        self.plotter.add_mesh(wall2, color='#555555', opacity=0.1, name="wall2")
        
        wall3 = pv.Plane(center=(0, 0, 2000), direction=(0, 0, -1), i_size=4000, j_size=2000)
        self.plotter.add_mesh(wall3, color='#555555', opacity=0.1, name="wall3")
        
        wall4 = pv.Plane(center=(0, 0, -2000), direction=(0, 0, 1), i_size=4000, j_size=2000)
        self.plotter.add_mesh(wall4, color='#555555', opacity=0.1, name="wall4")
        
        # Create isocenter marker
        isocenter = pv.Sphere(center=(0, 0, 0), radius=10)
        self.plotter.add_mesh(isocenter, color='red', name="isocenter")
        
    def _create_machine_geometry(self):
        """Create the treatment machine geometry."""
        if not self.pyvista_available:
            return
            
        # Remove existing machine geometries if they exist
        self.plotter.remove_actor("gantry", render=False)
        self.plotter.remove_actor("gantry_arm", render=False)
        self.plotter.remove_actor("treatment_head", render=False)
        self.plotter.remove_actor("collimator", render=False)
        self.plotter.remove_actor("beam", render=False)
        self.plotter.remove_actor("couch_top", render=False)
        self.plotter.remove_actor("couch_base", render=False)
        
        # Gantry ring
        gantry_radius = self.gantry_radius
        gantry_thickness = 100
        gantry_width = 200
        gantry_ring = pv.Tube(center=(0, 0, 0), 
                             direction=(0, 1, 0), 
                             radius=gantry_radius, 
                             thickness=gantry_thickness)
        gantry_ring.clip_box(factor=0.5, invert=False)  # Half ring
        
        # Gantry arm
        angle_rad = np.radians(self.gantry_angle)
        arm_length = gantry_radius
        arm_width = 100
        arm_height = 100
        arm_start = np.array([
            gantry_radius * np.sin(angle_rad),
            0,
            -gantry_radius * np.cos(angle_rad)
        ])
        arm_direction = -np.array([np.sin(angle_rad), 0, -np.cos(angle_rad)])
        gantry_arm = pv.Cylinder(center=arm_start - arm_direction * arm_length/2, 
                               direction=arm_direction,
                               radius=arm_width/2, 
                               height=arm_length)
        
        # Treatment head
        head_size = 150
        head_length = 300
        head_position = arm_start - arm_direction * arm_length
        treatment_head = pv.Cylinder(center=head_position - arm_direction * head_length/2,
                                   direction=arm_direction,
                                   radius=head_size/2, 
                                   height=head_length)
        
        # Collimator
        collimator_size = 120
        collimator_height = 80
        collimator_position = head_position - arm_direction * (head_length + collimator_height/2)
        
        # Apply collimator rotation
        collimator_direction = arm_direction.copy()
        # Adjust direction with collimator angle if needed
        
        collimator = pv.Cylinder(center=collimator_position,
                               direction=collimator_direction,
                               radius=collimator_size/2,
                               height=collimator_height)
        
        # Treatment beam visualization
        beam_length = self.sid  # Length from source to detector
        beam_width = self.beam_field_size[0]
        beam_height = self.beam_field_size[1]
        
        # Beam source position (in treatment head)
        source_position = head_position - arm_direction * (head_length * 0.25)
        
        # Create a pyramid to represent the beam
        points = np.array([
            source_position,  # Apex (source)
            # Base points (at isocenter plane)
            source_position + beam_length * arm_direction + np.array([beam_width/2, 0, beam_height/2]),
            source_position + beam_length * arm_direction + np.array([beam_width/2, 0, -beam_height/2]),
            source_position + beam_length * arm_direction + np.array([-beam_width/2, 0, -beam_height/2]),
            source_position + beam_length * arm_direction + np.array([-beam_width/2, 0, beam_height/2]),
        ])
        
        # Define the faces of the pyramid
        faces = np.array([
            4, 0, 1, 2, 3,  # Base
            3, 0, 1, 4,     # Side face 1
            3, 0, 4, 3,     # Side face 2
            3, 0, 3, 2,     # Side face 3
            3, 0, 2, 1,     # Side face 4
        ])
        
        beam = pv.PolyData(points, faces)
        
        # Couch - create based on couch angle
        couch_angle_rad = np.radians(self.couch_angle)
        
        # Couch top
        couch_length = 2200
        couch_width = 550
        couch_height = 70
        couch_y_pos = -400  # Below isocenter
        
        # Create couch top as a box
        couch_top = pv.Box(bounds=[
            -couch_width/2, couch_width/2,
            couch_y_pos - couch_height/2, couch_y_pos + couch_height/2,
            -couch_length/2, couch_length/2
        ])
        
        # Rotate couch top based on couch angle
        couch_top = couch_top.rotate_z(self.couch_angle)
        
        # Couch base/pedestal
        base_height = 800
        base_width = 300
        base_length = 600
        base_y_pos = couch_y_pos - base_height/2 - couch_height/2
        
        couch_base = pv.Box(bounds=[
            -base_width/2, base_width/2,
            base_y_pos - base_height/2, base_y_pos + base_height/2,
            -base_length/2, base_length/2
        ])
        
        # Add meshes to the plotter
        gantry_color = [0.7, 0.7, 0.7]  # Light gray
        colliding = any(comp[0] == "gantry" for comp in self.colliding_components)
        if colliding:
            gantry_color = [1.0, 0.0, 0.0]  # Red if colliding
            
        couch_color = [0.3, 0.3, 0.8]  # Blue-ish
        if any(comp[0] == "couch" for comp in self.colliding_components):
            couch_color = [1.0, 0.0, 0.0]  # Red if colliding
            
        self.plotter.add_mesh(gantry_ring, color=gantry_color, name="gantry", render=False)
        self.plotter.add_mesh(gantry_arm, color=gantry_color, name="gantry_arm", render=False)
        self.plotter.add_mesh(treatment_head, color=gantry_color, name="treatment_head", render=False)
        self.plotter.add_mesh(collimator, color=gantry_color, name="collimator", render=False)
        self.plotter.add_mesh(beam, color='yellow', opacity=0.3, name="beam", render=False)
        self.plotter.add_mesh(couch_top, color=couch_color, name="couch_top", render=False)
        self.plotter.add_mesh(couch_base, color=couch_color, name="couch_base", render=False)
        
    def _create_patient_geometry(self):
        """Create a simplified patient geometry."""
        if not self.pyvista_available:
            return
            
        # Remove existing patient geometry if it exists
        self.plotter.remove_actor("patient_body", render=False)
        self.plotter.remove_actor("patient_head", render=False)
        
        # Create simplified patient body as ellipsoid
        patient_length = 1800
        patient_width = 350
        patient_height = 250
        patient_y_pos = -150  # Slightly below isocenter
        
        body = pv.Cylinder(center=(0, patient_y_pos, 0), 
                         direction=(0, 0, 1), 
                         radius=patient_width/2, 
                         height=patient_length)
        body = body.scale([1.0, patient_height/patient_width, 1.0])
        
        # Create head as a sphere
        head_size = 100
        head_position = [0, patient_y_pos, -patient_length/2 - head_size/2]
        head = pv.Sphere(center=head_position, radius=head_size)
        
        # Rotate patient based on couch angle
        body = body.rotate_z(self.couch_angle)
        head = head.rotate_z(self.couch_angle)
        
        # Add meshes to plotter
        patient_color = [0.9, 0.7, 0.5]  # Skin tone
        if any(comp[1] == "patient" for comp in self.colliding_components):
            patient_color = [1.0, 0.5, 0.0]  # Orange if colliding
            
        self.plotter.add_mesh(body, color=patient_color, opacity=0.7, name="patient_body", render=False)
        self.plotter.add_mesh(head, color=patient_color, opacity=0.7, name="patient_head", render=False)
        
    def set_angles(self, gantry_angle: float, couch_angle: float, collimator_angle: float = 0.0):
        """Set the angles for visualization."""
        self.gantry_angle = gantry_angle
        self.couch_angle = couch_angle
        self.collimator_angle = collimator_angle
        self.update_visualization()
        
    def set_collision_result(self, result: Dict):
        """Set the collision detection result."""
        self.collision_detected = result.get("collision_detected", False)
        self.warning_detected = result.get("warning", False)
        self.colliding_components = result.get("colliding_components", [])
        
        # Update status bar
        if self.collision_detected:
            self.status_bar.setText("COLLISION DETECTED")
            self.status_bar.setStyleSheet("color: white; background-color: red; font-weight: bold; padding: 5px;")
        elif self.warning_detected:
            self.status_bar.setText("COLLISION WARNING - Clearance less than safety margin")
            self.status_bar.setStyleSheet("color: black; background-color: yellow; font-weight: bold; padding: 5px;")
        else:
            self.status_bar.setText("No collision detected")
            self.status_bar.setStyleSheet("color: white; background-color: green; font-weight: bold; padding: 5px;")
            
        self.update_visualization()
        
    def set_field_size(self, width: float, height: float):
        """Set the beam field size in mm."""
        self.beam_field_size = (width, height)
        self.update_visualization()
        
    def set_machine_type(self, machine_type: str):
        """Set the machine type."""
        self.machine_type = machine_type
        # Adjust parameters based on machine type
        if machine_type == "TrueBeam":
            self.sad = 1000.0
            self.sid = 1500.0 
            self.gantry_radius = 1000.0
        elif machine_type == "Halcyon":
            self.sad = 1000.0
            self.sid = 1500.0
            self.gantry_radius = 900.0
        elif machine_type == "Ethos":
            self.sad = 1000.0
            self.sid = 1500.0
            self.gantry_radius = 950.0
        self.update_visualization()
        
    def update_visualization(self):
        """Update the 3D visualization."""
        if not self.pyvista_available:
            self.update()  # Fallback to 2D painting if PyVista not available
            return
            
        # Update machine and patient geometries
        self._create_machine_geometry()
        self._create_patient_geometry()
        
        # Render the scene
        self.plotter.render()
        
    def paintEvent(self, event):
        """Fallback 2D visualization if PyVista is not available."""
        if self.pyvista_available:
            return
            
        # Traditional QPainter implementation as fallback
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        
        # Set scale factor for drawing
        scale = min(width, height) * 0.4
        
        # Draw room background (floor)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(80, 80, 80)))
        painter.drawRect(0, 0, width, height)
        
        # Draw a coordinate system
        painter.setPen(QPen(QColor(150, 150, 150), 1, Qt.DashLine))
        painter.drawLine(center_x, 0, center_x, height)
        painter.drawLine(0, center_y, width, center_y)
        
        # Draw labels for coordinate system
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawText(center_x + 5, 15, "Head")
        painter.drawText(center_x + 5, height - 5, "Feet")
        painter.drawText(5, center_y - 5, "Right")
        painter.drawText(width - 40, center_y - 5, "Left")
        
        # Draw patient outline (simplified)
        painter.setPen(QPen(QColor(100, 150, 255), 2))
        painter.setBrush(QBrush(QColor(100, 150, 255, 100)))
        patient_width = scale * 0.3
        patient_height = scale * 0.8
        painter.drawEllipse(center_x - patient_width/2, center_y - patient_height/2, 
                          patient_width, patient_height)
        
        # Calculate gantry position based on gantry angle
        gantry_radius = scale * 0.9
        rad_angle = math.radians(self.gantry_angle)
        gantry_x = center_x + gantry_radius * math.sin(rad_angle)
        gantry_y = center_y - gantry_radius * math.cos(rad_angle)
        
        # Draw couch
        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(self.couch_angle)
        
        couch_color = QColor(150, 150, 150)
        if any(comp[0] == "couch" for comp in self.colliding_components):
            couch_color = QColor(255, 50, 50)
        elif any(comp[0] == "couch" and comp[1] == "patient" for comp in self.colliding_components):
            couch_color = QColor(255, 150, 0)
        
        painter.setPen(QPen(couch_color, 2))
        painter.setBrush(QBrush(QColor(couch_color.red(), couch_color.green(), couch_color.blue(), 100)))
        
        couch_width = scale * 0.5
        couch_height = scale * 1.2
        painter.drawRect(-couch_width/2, -couch_height/2, couch_width, couch_height)
        
        painter.restore()
        
        # Draw gantry
        gantry_color = QColor(200, 200, 200)
        if any(comp[0] == "gantry" for comp in self.colliding_components):
            gantry_color = QColor(255, 50, 50)
        elif any(comp[0] == "gantry" and comp[1] == "patient" for comp in self.colliding_components):
            gantry_color = QColor(255, 150, 0)
        
        painter.setPen(QPen(gantry_color, 3))
        
        # Draw gantry circle
        painter.drawEllipse(center_x - gantry_radius, center_y - gantry_radius, 
                          2 * gantry_radius, 2 * gantry_radius)
        
        # Draw gantry head
        painter.setBrush(QBrush(QColor(gantry_color.red(), gantry_color.green(), gantry_color.blue(), 150)))
        gantry_head_size = scale * 0.1
        painter.drawEllipse(gantry_x - gantry_head_size/2, gantry_y - gantry_head_size/2, 
                          gantry_head_size, gantry_head_size)
        
        # Draw beam direction from gantry head
        beam_length = scale * 0.2
        beam_end_x = gantry_x - beam_length * math.sin(rad_angle)
        beam_end_y = gantry_y + beam_length * math.cos(rad_angle)
        
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.drawLine(gantry_x, gantry_y, beam_end_x, beam_end_y)
        
        # Draw isocenter
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        isocenter_size = 6
        painter.drawEllipse(center_x - isocenter_size/2, center_y - isocenter_size/2, 
                          isocenter_size, isocenter_size)
        
        # Draw collision warning if needed
        if self.collision_detected:
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            painter.drawText(10, 30, "COLLISION DETECTED")
        elif self.warning_detected:
            painter.setPen(QPen(QColor(255, 150, 0), 3))
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            painter.drawText(10, 30, "COLLISION WARNING")
        
        # Draw angle information
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(10, height - 60, f"Gantry: {self.gantry_angle:.1f}°")
        painter.drawText(10, height - 45, f"Couch: {self.couch_angle:.1f}°")
        painter.drawText(10, height - 30, f"Collimator: {self.collimator_angle:.1f}°")
        
        # End painting
        painter.end()

    def resizeEvent(self, event):
        """Handle window resize event."""
        super().resizeEvent(event)
        if self.pyvista_available and hasattr(self, 'plotter'):
            self.plotter.update_scalar_bar_range()
            
    def closeEvent(self, event):
        """Handle widget close event."""
        if self.pyvista_available and hasattr(self, 'plotter'):
            self.plotter.close()
        super().closeEvent(event)


class CollisionResultTable(QTableWidget):
    """
    Table for displaying collision detection results.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup table
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels([
            "Beam", "Control Point", "Gantry", "Couch", "Collimator", 
            "Minimum Distance", "Status"
        ])
        
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        
        self.results = {}
    
    def update_results(self, results: Dict):
        """
        Update the table with collision detection results.
        
        Args:
            results: Dictionary of beam collision results
        """
        self.results = results
        self.setRowCount(0)
        
        row = 0
        for beam_id, beam_results in results.items():
            for cp_result in beam_results:
                self.insertRow(row)
                
                # Beam ID
                self.setItem(row, 0, QTableWidgetItem(beam_id))
                
                # Control point
                self.setItem(row, 1, QTableWidgetItem(str(cp_result["control_point"])))
                
                # Gantry angle
                self.setItem(row, 2, QTableWidgetItem(f"{cp_result['gantry_angle']:.1f}°"))
                
                # Couch angle
                self.setItem(row, 3, QTableWidgetItem(f"{cp_result['couch_angle']:.1f}°"))
                
                # Collimator angle
                self.setItem(row, 4, QTableWidgetItem(f"{cp_result['collimator_angle']:.1f}°"))
                
                # Minimum distance
                min_distance = cp_result["result"]["min_distance"]
                distance_item = QTableWidgetItem(f"{min_distance:.1f} mm")
                self.setItem(row, 5, distance_item)
                
                # Status
                result = cp_result["result"]
                if result["collision_detected"]:
                    status = "COLLISION"
                    status_color = QColor(255, 0, 0)
                elif result["warning"]:
                    status = "WARNING"
                    status_color = QColor(255, 150, 0)
                else:
                    status = "OK"
                    status_color = QColor(0, 200, 0)
                
                status_item = QTableWidgetItem(status)
                status_item.setForeground(QBrush(status_color))
                status_item.setFont(QFont("Arial", 10, QFont.Bold))
                self.setItem(row, 6, status_item)
                
                # Color the row based on status
                if result["collision_detected"]:
                    for col in range(self.columnCount()):
                        self.item(row, col).setBackground(QBrush(QColor(255, 200, 200)))
                elif result["warning"]:
                    for col in range(self.columnCount()):
                        self.item(row, col).setBackground(QBrush(QColor(255, 240, 200)))
                
                row += 1
        
        self.resizeColumnsToContents()


class CollisionDetailsTable(QTableWidget):
    """
    Table for displaying detailed collision information.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup table
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Component 1", "Component 2", "Distance (mm)"])
        
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
    
    def update_details(self, result: Dict):
        """
        Update the table with detailed collision information.
        
        Args:
            result: Collision detection result dictionary
        """
        self.setRowCount(0)
        
        components = result.get("colliding_components", [])
        min_distance = result.get("min_distance", float('inf'))
        
        self.setRowCount(len(components))
        
        for row, (comp1, comp2, distance) in enumerate(components):
            # Component 1
            self.setItem(row, 0, QTableWidgetItem(comp1.capitalize()))
            
            # Component 2
            self.setItem(row, 1, QTableWidgetItem(comp2.capitalize()))
            
            # Distance
            distance_item = QTableWidgetItem(f"{distance:.1f}")
            if distance < 50.0:
                distance_item.setForeground(QBrush(QColor(255, 0, 0)))
                distance_item.setFont(QFont("Arial", 10, QFont.Bold))
            elif distance < 100.0:
                distance_item.setForeground(QBrush(QColor(255, 150, 0)))
                distance_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.setItem(row, 2, distance_item)


class CollisionDetectionDialog(QDialog):
    """
    Dialog for detecting and visualizing potential collisions.
    """
    
    def __init__(self, plan: Plan, patient: Patient, parent=None):
        super().__init__(parent)
        
        self.plan = plan
        self.patient = patient
        
        # Initialize collision detector
        self.detector = create_collision_detector(plan, patient)
        
        # Initialize dialog
        self.setWindowTitle("Collision Detection")
        self.setMinimumSize(800, 600)
        
        # Set up UI
        self._setup_ui()
        
        # Check plan collisions
        self._check_plan_collisions()
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        
        # Create top control panel
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_layout = QHBoxLayout(control_panel)
        
        # Machine selection
        machine_group = QGroupBox("Machine")
        machine_layout = QVBoxLayout(machine_group)
        
        self.machine_combo = QComboBox()
        self.machine_combo.addItems(["TrueBeam", "VitalBeam", "Halcyon"])
        self.machine_combo.currentTextChanged.connect(self._on_machine_changed)
        machine_layout.addWidget(self.machine_combo)
        
        control_layout.addWidget(machine_group)
        
        # Structure selection
        structure_group = QGroupBox("Patient Structure")
        structure_layout = QVBoxLayout(structure_group)
        
        self.structure_combo = QComboBox()
        for structure in self.patient.get_structures():
            self.structure_combo.addItem(structure.name)
        
        # Set default to BODY or External if available
        default_index = self.structure_combo.findText("BODY")
        if default_index == -1:
            default_index = self.structure_combo.findText("External")
        if default_index != -1:
            self.structure_combo.setCurrentIndex(default_index)
        
        self.structure_combo.currentTextChanged.connect(self._on_structure_changed)
        structure_layout.addWidget(self.structure_combo)
        
        control_layout.addWidget(structure_group)
        
        # Thresholds
        threshold_group = QGroupBox("Thresholds")
        threshold_layout = QVBoxLayout(threshold_group)
        
        collision_layout = QHBoxLayout()
        collision_layout.addWidget(QLabel("Collision:"))
        self.collision_spin = QDoubleSpinBox()
        self.collision_spin.setRange(10.0, 200.0)
        self.collision_spin.setValue(50.0)
        self.collision_spin.setSuffix(" mm")
        self.collision_spin.valueChanged.connect(self._on_threshold_changed)
        collision_layout.addWidget(self.collision_spin)
        threshold_layout.addLayout(collision_layout)
        
        warning_layout = QHBoxLayout()
        warning_layout.addWidget(QLabel("Warning:"))
        self.warning_spin = QDoubleSpinBox()
        self.warning_spin.setRange(20.0, 300.0)
        self.warning_spin.setValue(100.0)
        self.warning_spin.setSuffix(" mm")
        self.warning_spin.valueChanged.connect(self._on_threshold_changed)
        warning_layout.addWidget(self.warning_spin)
        threshold_layout.addLayout(warning_layout)
        
        control_layout.addWidget(threshold_group)
        
        # Add manual check controls
        manual_group = QGroupBox("Manual Check")
        manual_layout = QHBoxLayout(manual_group)
        
        # Angle input group
        angle_layout = QVBoxLayout()
        
        gantry_layout = QHBoxLayout()
        gantry_layout.addWidget(QLabel("Gantry:"))
        self.gantry_spin = QDoubleSpinBox()
        self.gantry_spin.setRange(0.0, 359.9)
        self.gantry_spin.setValue(0.0)
        self.gantry_spin.setSuffix("°")
        self.gantry_spin.valueChanged.connect(self._on_manual_angle_changed)
        gantry_layout.addWidget(self.gantry_spin)
        angle_layout.addLayout(gantry_layout)
        
        couch_layout = QHBoxLayout()
        couch_layout.addWidget(QLabel("Couch:"))
        self.couch_spin = QDoubleSpinBox()
        self.couch_spin.setRange(0.0, 359.9)
        self.couch_spin.setValue(0.0)
        self.couch_spin.setSuffix("°")
        self.couch_spin.valueChanged.connect(self._on_manual_angle_changed)
        couch_layout.addWidget(self.couch_spin)
        angle_layout.addLayout(couch_layout)
        
        collimator_layout = QHBoxLayout()
        collimator_layout.addWidget(QLabel("Collimator:"))
        self.collimator_spin = QDoubleSpinBox()
        self.collimator_spin.setRange(0.0, 359.9)
        self.collimator_spin.setValue(0.0)
        self.collimator_spin.setSuffix("°")
        self.collimator_spin.valueChanged.connect(self._on_manual_angle_changed)
        collimator_layout.addWidget(self.collimator_spin)
        angle_layout.addLayout(collimator_layout)
        
        manual_layout.addLayout(angle_layout)
        
        # Check button
        self.check_button = QPushButton("Check")
        self.check_button.clicked.connect(self._on_manual_check)
        manual_layout.addWidget(self.check_button)
        
        control_layout.addWidget(manual_group)
        
        main_layout.addWidget(control_panel)
        
        # Create main content area with splitter
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - visualization
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        
        # Add visualization widget
        self.visualization = CollisionVisualization()
        left_layout.addWidget(self.visualization)
        
        # Add details table
        self.details_table = CollisionDetailsTable()
        left_layout.addWidget(self.details_table)
        
        self.splitter.addWidget(left_panel)
        
        # Right panel - results table
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        
        # Add results table
        self.results_table = CollisionResultTable()
        self.results_table.selectionModel().selectionChanged.connect(self._on_result_selected)
        right_layout.addWidget(self.results_table)
        
        # Add controls for finding collision-free angles
        free_angles_group = QGroupBox("Find Collision-Free Angles")
        free_angles_layout = QVBoxLayout(free_angles_group)
        
        fixed_couch_layout = QHBoxLayout()
        fixed_couch_layout.addWidget(QLabel("Fixed Couch Angle:"))
        self.fixed_couch_spin = QDoubleSpinBox()
        self.fixed_couch_spin.setRange(0.0, 359.9)
        self.fixed_couch_spin.setValue(0.0)
        self.fixed_couch_spin.setSuffix("°")
        fixed_couch_layout.addWidget(self.fixed_couch_spin)
        
        self.find_angles_button = QPushButton("Find Angles")
        self.find_angles_button.clicked.connect(self._on_find_free_angles)
        fixed_couch_layout.addWidget(self.find_angles_button)
        
        free_angles_layout.addLayout(fixed_couch_layout)
        
        # List of collision-free angles
        self.free_angles_label = QLabel("Collision-Free Gantry Angles:")
        free_angles_layout.addWidget(self.free_angles_label)
        
        right_layout.addWidget(free_angles_group)
        
        self.splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        self.splitter.setSizes([400, 400])
        
        main_layout.addWidget(self.splitter, 1)  # 1 = stretch factor
        
        # Add button box
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
    
    def _check_plan_collisions(self):
        """Check collisions for the entire plan."""
        # Show a busy cursor
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        
        try:
            # Run collision detection
            results = self.detector.check_plan_collisions(self.plan)
            
            # Update the results table
            self.results_table.update_results(results)
            
            # Select the first row if available
            if self.results_table.rowCount() > 0:
                self.results_table.selectRow(0)
        finally:
            # Restore the cursor
            QtWidgets.QApplication.restoreOverrideCursor()
    
    def _on_machine_changed(self, machine_type: str):
        """
        Handle machine type change.
        
        Args:
            machine_type: New machine type
        """
        # Reinitialize the detector with the new machine type
        self.detector = create_collision_detector(self.plan, self.patient, machine_type)
        
        # Update collision thresholds
        self.detector.collision_threshold = self.collision_spin.value()
        self.detector.warning_threshold = self.warning_spin.value()
        
        # Re-check plan collisions
        self._check_plan_collisions()
    
    def _on_structure_changed(self, structure_name: str):
        """
        Handle patient structure change.
        
        Args:
            structure_name: Name of the selected structure
        """
        # Update the detector with the new structure
        self.detector.set_patient(self.patient, structure_name)
        
        # Re-check plan collisions
        self._check_plan_collisions()
    
    def _on_threshold_changed(self):
        """Handle threshold value changes."""
        self.detector.collision_threshold = self.collision_spin.value()
        self.detector.warning_threshold = self.warning_spin.value()
        
        # Re-check plan collisions
        self._check_plan_collisions()
    
    def _on_manual_angle_changed(self):
        """Handle manual angle input changes."""
        # Update visualization
        self.visualization.set_angles(
            self.gantry_spin.value(),
            self.couch_spin.value(),
            self.collimator_spin.value()
        )
    
    def _on_manual_check(self):
        """Handle manual collision check button click."""
        # Get current angles
        gantry_angle = self.gantry_spin.value()
        couch_angle = self.couch_spin.value()
        collimator_angle = self.collimator_spin.value()
        
        # Check collision
        result = self.detector.check_collision(gantry_angle, couch_angle, collimator_angle)
        
        # Update visualization
        self.visualization.set_angles(gantry_angle, couch_angle, collimator_angle)
        self.visualization.set_collision_result(result)
        
        # Update details table
        self.details_table.update_details(result)
    
    def _on_result_selected(self, selected, deselected):
        """
        Handle selection change in the results table.
        
        Args:
            selected: Selected indexes
            deselected: Deselected indexes
        """
        if not selected.indexes():
            return
        
        # Get the selected row
        row = selected.indexes()[0].row()
        
        # Get angles from the table
        gantry_text = self.results_table.item(row, 2).text()
        couch_text = self.results_table.item(row, 3).text()
        collimator_text = self.results_table.item(row, 4).text()
        
        # Parse angles
        gantry_angle = float(gantry_text.replace('°', ''))
        couch_angle = float(couch_text.replace('°', ''))
        collimator_angle = float(collimator_text.replace('°', ''))
        
        # Update spin boxes
        self.gantry_spin.setValue(gantry_angle)
        self.couch_spin.setValue(couch_angle)
        self.collimator_spin.setValue(collimator_angle)
        
        # Get beam ID and control point
        beam_id = self.results_table.item(row, 0).text()
        cp_index = int(self.results_table.item(row, 1).text())
        
        # Get result from stored data
        result = self.results_table.results[beam_id][cp_index]["result"]
        
        # Update visualization
        self.visualization.set_angles(gantry_angle, couch_angle, collimator_angle)
        self.visualization.set_collision_result(result)
        
        # Update details table
        self.details_table.update_details(result)
    
    def _on_find_free_angles(self):
        """Find and display collision-free angles."""
        # Show a busy cursor
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        
        try:
            # Get fixed couch angle
            couch_angle = self.fixed_couch_spin.value()
            
            # Find collision-free angles
            free_angles = self.detector.get_collision_free_angles(couch_angle, 10.0)
            
            # Format the angles as a string
            if free_angles:
                angles_str = ", ".join([f"{angle:.1f}°" for angle in free_angles])
                self.free_angles_label.setText(f"Collision-Free Gantry Angles: {angles_str}")
            else:
                self.free_angles_label.setText("No collision-free gantry angles found")
        finally:
            # Restore the cursor
            QtWidgets.QApplication.restoreOverrideCursor() 