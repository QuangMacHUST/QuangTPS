#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Beam Visualization Module

This module provides interactive 3D visualization of radiotherapy treatment beams,
patient anatomy, and dose distributions using PyVista.
"""

import os
import sys
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any

import importlib.util

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

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QCheckBox, QGroupBox, QTabWidget, QToolBar,
    QSplitter, QFrame, QSizePolicy, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QColor, QIcon

from quangtps.core.logging import get_logger
from quangtps.imaging.image import Image
from quangtps.imaging.structures import Structure, StructureSet
from quangtps.planning.beam import Beam
from quangtps.treatment.plan import Plan
from quangtps.dose.dose_grid import DoseGrid
from quangtps.ui.dependency_installer import check_and_install_feature_dependencies

logger = get_logger(__name__)

class Beam3DVisualization(QWidget):
    """
    Interactive 3D visualization for radiotherapy beams and patient anatomy.
    
    This widget provides a comprehensive 3D visualization of the treatment setup,
    including the patient anatomy, beam arrangement, and dose distributions using PyVista.
    """
    
    # Signals
    beam_selected = pyqtSignal(object)  # Emitted when a beam is selected in 3D view
    view_updated = pyqtSignal()  # Emitted when the view is updated
    
    def __init__(self, parent=None):
        """Initialize the 3D beam visualization widget."""
        super().__init__(parent)
        
        # State variables
        self.patient_image = None
        self.structure_set = None
        self.plan = None
        self.beams = []
        self.dose_grid = None
        self.selected_beam = None
        
        # Visualization options
        self.show_patient = True
        self.show_structures = True
        self.show_beams = True
        self.show_dose = False
        self.show_machine = True
        self.show_axes = True
        self.structure_opacity = 0.5
        self.dose_opacity = 0.7
        self.isodose_levels = [95, 80, 70, 50, 30, 10]  # % of max dose
        
        # Machine parameters
        self.sad = 1000.0  # Source-to-Axis Distance in mm
        self.linac_length = 600.0  # Length of the linac head in mm
        
        # Check for dependencies
        self._check_dependencies()
        
        # Initialize UI
        self._init_ui()
        
        # Initialize PyVista plotter if available
        self._init_plotter()
    
    def _check_dependencies(self):
        """Check if the required dependencies for 3D visualization are available."""
        if not all([PYVISTA_AVAILABLE, PYVISTAQT_AVAILABLE, VTK_AVAILABLE]):
            logger.warning("3D visualization dependencies are not available")
            self.dependencies_available = False
        else:
            self.dependencies_available = True
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar for visualization controls
        toolbar = QToolBar("3D Visualization Controls")
        
        # View presets
        view_label = QLabel("View:")
        toolbar.addWidget(view_label)
        
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Anterior", "Posterior", "Left", "Right", "Superior", "Inferior", "BEV"])
        self.view_combo.setCurrentIndex(0)
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        toolbar.addWidget(self.view_combo)
        toolbar.addSeparator()
        
        # Visualization checkboxes
        self.patient_check = QCheckBox("Patient")
        self.patient_check.setChecked(self.show_patient)
        self.patient_check.toggled.connect(self._on_patient_toggled)
        toolbar.addWidget(self.patient_check)
        
        self.structures_check = QCheckBox("Structures")
        self.structures_check.setChecked(self.show_structures)
        self.structures_check.toggled.connect(self._on_structures_toggled)
        toolbar.addWidget(self.structures_check)
        
        self.beams_check = QCheckBox("Beams")
        self.beams_check.setChecked(self.show_beams)
        self.beams_check.toggled.connect(self._on_beams_toggled)
        toolbar.addWidget(self.beams_check)
        
        self.dose_check = QCheckBox("Dose")
        self.dose_check.setChecked(self.show_dose)
        self.dose_check.toggled.connect(self._on_dose_toggled)
        toolbar.addWidget(self.dose_check)
        
        self.machine_check = QCheckBox("Machine")
        self.machine_check.setChecked(self.show_machine)
        self.machine_check.toggled.connect(self._on_machine_toggled)
        toolbar.addWidget(self.machine_check)
        
        self.axes_check = QCheckBox("Axes")
        self.axes_check.setChecked(self.show_axes)
        self.axes_check.toggled.connect(self._on_axes_toggled)
        toolbar.addWidget(self.axes_check)
        
        main_layout.addWidget(toolbar)
        
        # Container for PyVista plotter
        self.plotter_container = QWidget()
        self.plotter_container.setMinimumSize(500, 400)
        self.plotter_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Placeholder layout for the plotter
        self.plotter_layout = QVBoxLayout(self.plotter_container)
        self.plotter_layout.setContentsMargins(0, 0, 0, 0)
        
        main_layout.addWidget(self.plotter_container, 1)
        
        # If dependencies are not available, show an info message
        if not self.dependencies_available:
            self._show_dependency_message()
    
    def _init_plotter(self):
        """Initialize the PyVista plotter if dependencies are available."""
        if not self.dependencies_available:
            return
        
        try:
            # Set up the PyVista plotter within the Qt container
            self.plotter = QtInteractor(self.plotter_container)
            self.plotter_layout.addWidget(self.plotter)
            
            # Set up initial plotter properties
            self.plotter.set_background("black")
            if self.show_axes:
                self.plotter.add_axes()
            
            # Add a callback for when an actor is clicked
            self.plotter.enable_point_picking(callback=self._on_point_picked, 
                                             show_message=False,
                                             left_clicking=True)
            
            # Store actors for later reference
            self.actors = {
                "patient": None,
                "structures": {},
                "beams": {},
                "dose": {},
                "machine": {},
            }
            
            # Set up lighting
            self.plotter.enable_shadows()
            light = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), 
                            color=[1, 1, 1], intensity=0.8)
            self.plotter.add_light(light)
            
            # Set up initial camera position
            self.plotter.camera_position = 'xy'  # Default to anterior view
            
            # Update the plotter
            self.plotter.update()
            
        except Exception as e:
            logger.error(f"Error initializing PyVista plotter: {str(e)}")
            self.dependencies_available = False
            self._show_dependency_message()
    
    def _show_dependency_message(self):
        """Show a message about missing dependencies in the plotter container."""
        # Clear any existing widgets in the plotter container
        for i in reversed(range(self.plotter_layout.count())): 
            self.plotter_layout.itemAt(i).widget().setParent(None)
        
        # Add a message about missing dependencies
        message_label = QLabel("3D visualization requires additional dependencies.")
        message_label.setAlignment(Qt.AlignCenter)
        
        install_button = QPushButton("Install Dependencies")
        install_button.clicked.connect(self._install_dependencies)
        
        layout = QVBoxLayout()
        layout.addStretch()
        layout.addWidget(message_label)
        layout.addWidget(install_button)
        layout.addStretch()
        
        message_widget = QWidget()
        message_widget.setLayout(layout)
        
        self.plotter_layout.addWidget(message_widget)
    
    def _install_dependencies(self):
        """Install the required dependencies for 3D visualization."""
        result = check_and_install_feature_dependencies("3d_visualization", self)
        if result:
            QMessageBox.information(
                self,
                "Dependencies Installed",
                "3D visualization dependencies were installed successfully. "
                "Please restart the application to use the 3D visualization features."
            )
        else:
            QMessageBox.warning(
                self,
                "Installation Failed",
                "Failed to install dependencies. Please try again or install manually."
            )
    
    def set_patient_data(self, image: Image, structures: StructureSet = None):
        """
        Set the patient image and structure set data.
        
        Parameters
        ----------
        image : Image
            The patient CT or MR image
        structures : StructureSet, optional
            The structure set containing contours
        """
        self.patient_image = image
        self.structure_set = structures
        self._update_visualization()
    
    def set_plan(self, plan: Plan):
        """
        Set the treatment plan data.
        
        Parameters
        ----------
        plan : Plan
            The treatment plan containing beams
        """
        self.plan = plan
        if plan and hasattr(plan, 'beams'):
            self.beams = plan.beams
        else:
            self.beams = []
        self._update_visualization()
    
    def set_beams(self, beams: List[Beam]):
        """
        Set the beam data directly.
        
        Parameters
        ----------
        beams : List[Beam]
            List of beam objects
        """
        self.beams = beams
        self._update_visualization()
    
    def set_dose_grid(self, dose_grid: DoseGrid):
        """
        Set the dose grid data for visualization.
        
        Parameters
        ----------
        dose_grid : DoseGrid
            The dose grid to visualize
        """
        self.dose_grid = dose_grid
        self._update_visualization()
    
    def _update_visualization(self):
        """Update the 3D visualization with current data."""
        if not self.dependencies_available:
            return
        
        # Clear existing actors
        self.plotter.clear()
        
        # Reset actor dictionaries
        self.actors = {
            "patient": None,
            "structures": {},
            "beams": {},
            "dose": {},
            "machine": {},
        }
        
        # Add patient outline if available
        if self.show_patient and self.patient_image is not None:
            self._add_patient_visualization()
        
        # Add structures if available
        if self.show_structures and self.structure_set is not None:
            self._add_structures_visualization()
        
        # Add beams if available
        if self.show_beams and self.beams:
            self._add_beams_visualization()
        
        # Add dose visualization if available
        if self.show_dose and self.dose_grid is not None:
            self._add_dose_visualization()
        
        # Add axes if enabled
        if self.show_axes:
            self.plotter.add_axes()
        
        # Update the view
        self.plotter.reset_camera()
        self.plotter.update()
        
        # Emit signal that view was updated
        self.view_updated.emit()
    
    def _add_patient_visualization(self):
        """Add the patient outline to the visualization."""
        if not self.patient_image or not hasattr(self.patient_image, 'data') or self.patient_image.data is None:
            return
            
        try:
            # Create a simple surface from the patient external contour
            # or use a thresholded surface from the CT data
            data = self.patient_image.data
            
            # Convert to PyVista grid
            grid = pv.UniformGrid()
            grid.dimensions = np.array(data.shape) + 1
            grid.origin = (0, 0, 0)  # Or use actual patient coordinates
            grid.spacing = (1, 1, 1)  # Or use actual pixel spacing
            grid.cell_data["values"] = data.flatten(order="F")
            
            # Extract a contour at a threshold (e.g., air-tissue boundary)
            contour = grid.contour([0])
            contour.compute_normals(inplace=True)
            
            # Add the contour to the visualization
            self.actors["patient"] = self.plotter.add_mesh(
                contour,
                color="white",
                opacity=0.2,
                pickable=False
            )
            
        except Exception as e:
            logger.error(f"Error adding patient visualization: {str(e)}")
    
    def _add_structures_visualization(self):
        """Add structure contours to the visualization."""
        if not self.structure_set or not hasattr(self.structure_set, 'structures'):
            return
            
        try:
            for structure in self.structure_set.structures:
                if not structure.mask.any():
                    continue
                
                # Convert structure mask to mesh
                mask = structure.mask
                
                # Create a PyVista grid from the mask
                grid = pv.UniformGrid()
                grid.dimensions = np.array(mask.shape) + 1
                grid.origin = (0, 0, 0)  # Or use actual patient coordinates
                grid.spacing = (1, 1, 1)  # Or use actual pixel spacing
                grid.cell_data["values"] = mask.flatten(order="F")
                
                # Extract a contour at threshold 0.5 (binary mask)
                contour = grid.contour([0.5])
                contour.compute_normals(inplace=True)
                
                # Determine structure color
                color = structure.color if hasattr(structure, 'color') else [1, 0, 0]
                
                # Add the structure to the visualization
                self.actors["structures"][structure.name] = self.plotter.add_mesh(
                    contour,
                    color=color,
                    opacity=self.structure_opacity,
                    pickable=True,
                    name=structure.name
                )
                
        except Exception as e:
            logger.error(f"Error adding structures visualization: {str(e)}")
    
    def _add_beams_visualization(self):
        """Add beam visualization to the scene."""
        if not self.beams:
            return
            
        try:
            for i, beam in enumerate(self.beams):
                # Get beam parameters
                gantry_angle = beam.gantry_angle if hasattr(beam, 'gantry_angle') else 0
                collimator_angle = beam.collimator_angle if hasattr(beam, 'collimator_angle') else 0
                couch_angle = beam.couch_angle if hasattr(beam, 'couch_angle') else 0
                field_size = beam.field_size if hasattr(beam, 'field_size') else (100, 100)
                isocenter = beam.isocenter if hasattr(beam, 'isocenter') else (0, 0, 0)
                
                # Create beam geometry
                beam_actors = self._create_beam_geometry(
                    gantry_angle, collimator_angle, couch_angle,
                    field_size, isocenter, i
                )
                
                # Add beam to scene with unique IDs
                beam_group = {}
                for key, actor in beam_actors.items():
                    actor_key = f"beam_{i}_{key}"
                    beam_group[key] = self.plotter.add_mesh(
                        actor, 
                        name=actor_key,
                        pickable=True
                    )
                
                # Store the beam actors for later reference
                self.actors["beams"][i] = beam_group
                
                # Add machine geometry if enabled
                if self.show_machine:
                    machine_actor = self._create_machine_geometry(
                        gantry_angle, collimator_angle, isocenter
                    )
                    self.actors["machine"][i] = self.plotter.add_mesh(
                        machine_actor,
                        color="silver",
                        opacity=0.7,
                        pickable=False,
                        name=f"machine_{i}"
                    )
                
        except Exception as e:
            logger.error(f"Error adding beams visualization: {str(e)}")
    
    def _create_beam_geometry(self, gantry_angle, collimator_angle, couch_angle, 
                             field_size, isocenter, beam_index):
        """
        Create the 3D geometry for a beam.
        
        Parameters
        ----------
        gantry_angle : float
            The gantry angle in degrees
        collimator_angle : float
            The collimator angle in degrees
        couch_angle : float
            The couch angle in degrees
        field_size : tuple
            The field size (width, height) in mm
        isocenter : tuple
            The isocenter position (x, y, z) in mm
        beam_index : int
            The index of the beam for color selection
            
        Returns
        -------
        dict
            Dictionary of PyVista meshes representing beam components
        """
        # Define beam colors based on index
        colors = [
            [0.8, 0.1, 0.1],  # Red
            [0.1, 0.8, 0.1],  # Green
            [0.1, 0.1, 0.8],  # Blue
            [0.8, 0.8, 0.1],  # Yellow
            [0.8, 0.1, 0.8],  # Magenta
            [0.1, 0.8, 0.8],  # Cyan
            [0.8, 0.5, 0.2],  # Orange
            [0.5, 0.2, 0.8],  # Purple
        ]
        color = colors[beam_index % len(colors)]
        
        # Convert angles to radians
        gantry_rad = np.radians(gantry_angle)
        collimator_rad = np.radians(collimator_angle)
        couch_rad = np.radians(couch_angle)
        
        # Create beam components
        beam_meshes = {}
        
        # Create source point
        source_position = np.array([
            isocenter[0] + self.sad * np.sin(gantry_rad) * np.cos(couch_rad),
            isocenter[1] + self.sad * np.sin(couch_rad),
            isocenter[2] - self.sad * np.cos(gantry_rad) * np.cos(couch_rad)
        ])
        source = pv.Sphere(radius=10, center=source_position)
        beam_meshes["source"] = source
        
        # Create beam central axis
        central_axis = pv.Line(source_position, isocenter)
        beam_meshes["central_axis"] = central_axis
        
        # Create beam field at isocenter
        width, height = field_size[0], field_size[1]
        
        # First create field in standard position
        x_half = width / 2
        y_half = height / 2
        field_points = np.array([
            [-x_half, -y_half, 0],
            [x_half, -y_half, 0],
            [x_half, y_half, 0],
            [-x_half, y_half, 0]
        ])
        
        # Create transformation matrix for field
        # Collimator rotation
        coll_transform = np.array([
            [np.cos(collimator_rad), -np.sin(collimator_rad), 0],
            [np.sin(collimator_rad), np.cos(collimator_rad), 0],
            [0, 0, 1]
        ])
        
        # Apply collimator rotation
        field_points = np.array([np.dot(pt, coll_transform) for pt in field_points])
        
        # Gantry rotation matrix
        gantry_transform = np.array([
            [np.cos(gantry_rad), 0, np.sin(gantry_rad)],
            [0, 1, 0],
            [-np.sin(gantry_rad), 0, np.cos(gantry_rad)]
        ])
        
        # Apply gantry rotation
        field_points = np.array([np.dot(pt, gantry_transform) for pt in field_points])
        
        # Couch rotation matrix
        couch_transform = np.array([
            [np.cos(couch_rad), np.sin(couch_rad), 0],
            [-np.sin(couch_rad), np.cos(couch_rad), 0],
            [0, 0, 1]
        ])
        
        # Apply couch rotation
        field_points = np.array([np.dot(pt, couch_transform) for pt in field_points])
        
        # Move field to isocenter
        field_points += np.array(isocenter)
        
        # Create beam field
        field_poly = pv.PolyData()
        field_poly.points = field_points
        field_poly.faces = np.array([4, 0, 1, 2, 3])
        beam_meshes["field"] = field_poly
        
        # Create beam pyramid (from source to field)
        pyramid_vertices = np.vstack((source_position, field_points))
        pyramid_faces = np.array([
            4, 0, 1, 2, 3,  # base (field)
            3, 0, 1, 4,     # side 1
            3, 1, 2, 4,     # side 2
            3, 2, 3, 4,     # side 3
            3, 3, 0, 4      # side 4
        ])
        pyramid = pv.PolyData(pyramid_vertices, pyramid_faces)
        beam_meshes["pyramid"] = pyramid
        
        return beam_meshes
    
    def _create_machine_geometry(self, gantry_angle, collimator_angle, isocenter):
        """
        Create the 3D geometry for a treatment machine.
        
        Parameters
        ----------
        gantry_angle : float
            The gantry angle in degrees
        collimator_angle : float
            The collimator angle in degrees
        isocenter : tuple
            The isocenter position (x, y, z) in mm
            
        Returns
        -------
        pv.PolyData
            PyVista mesh representing the treatment machine
        """
        # Convert angles to radians
        gantry_rad = np.radians(gantry_angle)
        
        # Create simplified gantry geometry
        gantry_radius = self.sad + 100  # Add some offset for visualization
        
        # Create the gantry ring
        gantry_ring = pv.Circle(radius=gantry_radius, resolution=36)
        gantry_ring.rotate_x(90, inplace=True)
        
        # Create the linac head
        source_position = np.array([
            isocenter[0] + self.sad * np.sin(gantry_rad),
            isocenter[1],
            isocenter[2] - self.sad * np.cos(gantry_rad)
        ])
        
        # Create a cylinder for the linac head
        linac_head = pv.Cylinder(
            radius=50,
            height=self.linac_length,
            direction=(source_position - np.array(isocenter)) / self.sad,
            center=source_position - (source_position - np.array(isocenter)) * 0.5 * self.linac_length / self.sad
        )
        
        # Combine the parts
        machine_parts = [gantry_ring, linac_head]
        machine = linac_head  # Use the linac head as the base
        
        return machine
    
    def _add_dose_visualization(self):
        """Add dose visualization to the scene."""
        if not self.dose_grid or not hasattr(self.dose_grid, 'data') or self.dose_grid.data is None:
            return
            
        try:
            # Get dose data
            dose_data = self.dose_grid.data
            max_dose = np.max(dose_data)
            
            # Create PyVista grid from dose data
            grid = pv.UniformGrid()
            grid.dimensions = np.array(dose_data.shape) + 1
            grid.origin = (0, 0, 0)  # Or use actual patient coordinates
            grid.spacing = (1, 1, 1)  # Or use actual dose grid spacing
            grid.cell_data["values"] = dose_data.flatten(order="F")
            
            # Create isodose contours
            for level in self.isodose_levels:
                dose_value = level * max_dose / 100.0
                contour = grid.contour([dose_value])
                
                # Skip if contour is empty
                if contour.n_points == 0:
                    continue
                
                # Determine color for isodose level
                # Red for high dose, blue for low dose
                t = level / 100.0
                r = min(1.0, 2.0 * t)
                b = min(1.0, 2.0 * (1.0 - t))
                g = min(r, b)
                
                # Add the contour to the visualization
                self.actors["dose"][level] = self.plotter.add_mesh(
                    contour,
                    color=(r, g, b),
                    opacity=self.dose_opacity,
                    pickable=False,
                    name=f"isodose_{level}"
                )
                
        except Exception as e:
            logger.error(f"Error adding dose visualization: {str(e)}")
    
    def _on_view_changed(self, index):
        """
        Handle changes to the view preset.
        
        Parameters
        ----------
        index : int
            Index of the selected view in the combo box
        """
        if not self.dependencies_available:
            return
            
        view_name = self.view_combo.currentText()
        
        if view_name == "Anterior":
            self.plotter.view_xy()
        elif view_name == "Posterior":
            self.plotter.view_xy(negative=True)
        elif view_name == "Left":
            self.plotter.view_yz()
        elif view_name == "Right":
            self.plotter.view_yz(negative=True)
        elif view_name == "Superior":
            self.plotter.view_zx()
        elif view_name == "Inferior":
            self.plotter.view_zx(negative=True)
        elif view_name == "BEV" and self.selected_beam is not None:
            self._set_beam_eye_view(self.selected_beam)
    
    def _set_beam_eye_view(self, beam_index):
        """
        Set the camera to a beam's eye view.
        
        Parameters
        ----------
        beam_index : int
            Index of the beam to view from
        """
        if not self.dependencies_available or beam_index is None:
            return
            
        try:
            beam = self.beams[beam_index]
            
            # Get beam parameters
            gantry_angle = beam.gantry_angle if hasattr(beam, 'gantry_angle') else 0
            couch_angle = beam.couch_angle if hasattr(beam, 'couch_angle') else 0
            isocenter = beam.isocenter if hasattr(beam, 'isocenter') else (0, 0, 0)
            
            # Convert angles to radians
            gantry_rad = np.radians(gantry_angle)
            couch_rad = np.radians(couch_angle)
            
            # Calculate source position
            source_position = np.array([
                isocenter[0] + self.sad * np.sin(gantry_rad) * np.cos(couch_rad),
                isocenter[1] + self.sad * np.sin(couch_rad),
                isocenter[2] - self.sad * np.cos(gantry_rad) * np.cos(couch_rad)
            ])
            
            # Set camera to source position, looking at isocenter
            self.plotter.camera.position = source_position
            self.plotter.camera.focal_point = isocenter
            self.plotter.camera.view_up = [0, 1, 0]  # Set view up direction
            
            # Update the view
            self.plotter.update()
            
        except Exception as e:
            logger.error(f"Error setting beam's eye view: {str(e)}")
    
    def _on_patient_toggled(self, checked):
        """
        Handle toggling of patient visibility.
        
        Parameters
        ----------
        checked : bool
            Whether the patient should be visible
        """
        self.show_patient = checked
        self._update_visualization()
    
    def _on_structures_toggled(self, checked):
        """
        Handle toggling of structures visibility.
        
        Parameters
        ----------
        checked : bool
            Whether structures should be visible
        """
        self.show_structures = checked
        self._update_visualization()
    
    def _on_beams_toggled(self, checked):
        """
        Handle toggling of beams visibility.
        
        Parameters
        ----------
        checked : bool
            Whether beams should be visible
        """
        self.show_beams = checked
        self._update_visualization()
    
    def _on_dose_toggled(self, checked):
        """
        Handle toggling of dose visibility.
        
        Parameters
        ----------
        checked : bool
            Whether dose should be visible
        """
        self.show_dose = checked
        self._update_visualization()
    
    def _on_machine_toggled(self, checked):
        """
        Handle toggling of machine visibility.
        
        Parameters
        ----------
        checked : bool
            Whether the machine should be visible
        """
        self.show_machine = checked
        self._update_visualization()
    
    def _on_axes_toggled(self, checked):
        """
        Handle toggling of axes visibility.
        
        Parameters
        ----------
        checked : bool
            Whether axes should be visible
        """
        self.show_axes = checked
        if self.dependencies_available:
            if checked:
                self.plotter.add_axes()
            else:
                self.plotter.remove_bounds_axes()
            self.plotter.update()
    
    def _on_point_picked(self, point, actor=None):
        """
        Handle picking of points in the 3D view.
        
        Parameters
        ----------
        point : numpy.ndarray
            The picked point coordinates
        actor : vtkActor, optional
            The actor that was picked
        """
        if actor is None:
            return
        
        # Get the actor name
        actor_name = actor.name
        
        # Check if it's a beam actor
        if actor_name and actor_name.startswith("beam_"):
            # Extract beam index from actor name
            parts = actor_name.split("_")
            if len(parts) > 1:
                try:
                    beam_index = int(parts[1])
                    self.selected_beam = beam_index
                    
                    # Highlight the selected beam
                    self._highlight_selected_beam()
                    
                    # Emit signal that a beam was selected
                    if beam_index < len(self.beams):
                        self.beam_selected.emit(self.beams[beam_index])
                        
                except (ValueError, IndexError):
                    pass
    
    def _highlight_selected_beam(self):
        """Highlight the currently selected beam."""
        if not self.dependencies_available or self.selected_beam is None:
            return
            
        # Reset all beam colors
        for beam_index, beam_actors in self.actors["beams"].items():
            for part_name, actor in beam_actors.items():
                # Set normal color
                color = [0.8, 0.8, 0.8]  # Default grey
                
                # Use predefined colors based on beam index
                colors = [
                    [0.8, 0.1, 0.1],  # Red
                    [0.1, 0.8, 0.1],  # Green
                    [0.1, 0.1, 0.8],  # Blue
                    [0.8, 0.8, 0.1],  # Yellow
                    [0.8, 0.1, 0.8],  # Magenta
                    [0.1, 0.8, 0.8],  # Cyan
                    [0.8, 0.5, 0.2],  # Orange
                    [0.5, 0.2, 0.8],  # Purple
                ]
                
                if beam_index < len(colors):
                    color = colors[beam_index]
                
                # If this is the selected beam, increase brightness
                if beam_index == self.selected_beam:
                    color = [min(1.0, c * 1.5) for c in color]
                    
                # Update actor color
                actor.GetProperty().SetColor(color)
        
        # Update the view
        self.plotter.update()
    
    def set_structure_opacity(self, opacity):
        """
        Set the opacity for structure visualization.
        
        Parameters
        ----------
        opacity : float
            Opacity value between 0 and 1
        """
        self.structure_opacity = max(0.0, min(1.0, opacity))
        self._update_visualization()
    
    def set_dose_opacity(self, opacity):
        """
        Set the opacity for dose visualization.
        
        Parameters
        ----------
        opacity : float
            Opacity value between 0 and 1
        """
        self.dose_opacity = max(0.0, min(1.0, opacity))
        self._update_visualization()
    
    def set_isodose_levels(self, levels):
        """
        Set the isodose levels for dose visualization.
        
        Parameters
        ----------
        levels : list of float
            List of isodose levels as percentages of max dose
        """
        self.isodose_levels = sorted(levels, reverse=True)
        self._update_visualization()
    
    def take_screenshot(self, filename=None):
        """
        Take a screenshot of the current 3D view.
        
        Parameters
        ----------
        filename : str, optional
            Filename to save the screenshot to. If None, a default name is used.
            
        Returns
        -------
        str
            Path to the saved screenshot file
        """
        if not self.dependencies_available:
            return None
            
        if filename is None:
            # Generate a default filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quangtps_3d_view_{timestamp}.png"
        
        try:
            # Save screenshot
            self.plotter.screenshot(filename)
            logger.info(f"Screenshot saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            return None
    
    def resizeEvent(self, event):
        """Handle resize events for the widget."""
        super().resizeEvent(event)
        if self.dependencies_available:
            self.plotter.update_bounds_axes()
    
    def closeEvent(self, event):
        """Handle close events for the widget."""
        if self.dependencies_available:
            self.plotter.close()
        super().closeEvent(event)

# For testing purposes
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    viewer = Beam3DVisualization()
    viewer.show()
    sys.exit(app.exec_()) 