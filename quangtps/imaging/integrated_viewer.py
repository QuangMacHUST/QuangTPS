#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for integrated visualization in QuangTPS.

This module provides a comprehensive integrated viewer that combines:
1. 2D Multi-Planar Reconstruction views (axial, sagittal, coronal)
2. 3D volume rendering with contour visualization
3. Treatment beam visualization
4. DVH (Dose-Volume Histogram) panel
5. Real-time planning controls
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import SimpleITK as sitk

# VTK imports
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, 
    QCheckBox, QSlider, QGroupBox, QFormLayout, QColorDialog, QSpinBox,
    QDoubleSpinBox, QMessageBox, QFrame, QSplitter, QTabWidget, QScrollArea,
    QSizePolicy, QStackedWidget, QToolBar, QAction, QMenu, QToolButton, 
    QFileDialog, QDockWidget, QDialog, QGridLayout, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize, QTimer, QThread
from PyQt5.QtGui import QColor, QIcon, QPixmap, QFont, QMouseEvent, QKeyEvent

from quangtps.imaging.image_viewer import ImageViewer, MPRViewer
from quangtps.imaging.volume_renderer import VolumeRenderingWidget
from quangtps.imaging.contour import Contour, ContourCollection
from quangtps.imaging.structures import Structure
from quangtps.planning.beam import BeamSetup
from quangtps.planning.plan import Plan
from quangtps.planning.treatment_planner import TreatmentPlanner
from quangtps.evaluation.dvh.dvh_visualization import plot_dvh
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator

logger = logging.getLogger(__name__)


class BeamVisualizationWidget(QWidget):
    """Widget for visualizing treatment beams on image planes and in 3D."""
    
    def __init__(self, parent=None):
        """Initialize the beam visualization widget."""
        super().__init__(parent)
        self.beams = []
        self.selected_beam_index = -1
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        
        # Beam selection
        self.beam_combo = QComboBox()
        self.beam_combo.currentIndexChanged.connect(self._on_beam_selected)
        self.layout.addWidget(self.beam_combo)
        
        # Beam properties
        props_group = QGroupBox("Beam Properties")
        props_layout = QFormLayout(props_group)
        
        # Beam angle controls
        angle_layout = QHBoxLayout()
        self.gantry_spin = QDoubleSpinBox()
        self.gantry_spin.setRange(0, 360)
        self.gantry_spin.setSingleStep(1)
        self.gantry_spin.valueChanged.connect(self._on_gantry_changed)
        angle_layout.addWidget(QLabel("Gantry:"))
        angle_layout.addWidget(self.gantry_spin)
        
        self.collimator_spin = QDoubleSpinBox()
        self.collimator_spin.setRange(0, 360)
        self.collimator_spin.setSingleStep(1)
        self.collimator_spin.valueChanged.connect(self._on_collimator_changed)
        angle_layout.addWidget(QLabel("Collimator:"))
        angle_layout.addWidget(self.collimator_spin)
        
        self.couch_spin = QDoubleSpinBox()
        self.couch_spin.setRange(0, 360)
        self.couch_spin.setSingleStep(1)
        self.couch_spin.valueChanged.connect(self._on_couch_changed)
        angle_layout.addWidget(QLabel("Couch:"))
        angle_layout.addWidget(self.couch_spin)
        
        props_layout.addRow("Angles:", angle_layout)
        
        # Beam energy
        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["6MV", "10MV", "15MV", "6FFF", "10FFF"])
        props_layout.addRow("Energy:", self.energy_combo)
        
        # MLC visualization checkbox
        self.show_mlc = QCheckBox("Show MLC")
        self.show_mlc.setChecked(True)
        props_layout.addRow("", self.show_mlc)
        
        # Add beam properties group to layout
        self.layout.addWidget(props_group)
        
        # Buttons for beam modification
        buttons_layout = QHBoxLayout()
        
        self.add_beam_btn = QPushButton("Add Beam")
        self.add_beam_btn.clicked.connect(self._add_beam)
        buttons_layout.addWidget(self.add_beam_btn)
        
        self.delete_beam_btn = QPushButton("Delete Beam")
        self.delete_beam_btn.clicked.connect(self._delete_beam)
        buttons_layout.addWidget(self.delete_beam_btn)
        
        self.layout.addLayout(buttons_layout)
        
        # MLC editor
        mlc_group = QGroupBox("MLC Editor")
        mlc_layout = QVBoxLayout(mlc_group)
        
        # This will be a placeholder for a future MLC editor component
        mlc_label = QLabel("MLC editing capabilities will be available here")
        mlc_layout.addWidget(mlc_label)
        
        self.layout.addWidget(mlc_group)
        
        # Stretch to fill available space
        self.layout.addStretch()
    
    def set_beams(self, beams):
        """
        Set the beams to visualize.
        
        Parameters
        ----------
        beams : list
            List of beam objects
        """
        self.beams = beams
        self._update_beam_list()
    
    def _update_beam_list(self):
        """Update the beam selection dropdown."""
        self.beam_combo.clear()
        
        if not self.beams:
            return
            
        for i, beam in enumerate(self.beams):
            self.beam_combo.addItem(f"Beam {i+1}: {beam.name}")
    
    def _on_beam_selected(self, index):
        """
        Handle beam selection.
        
        Parameters
        ----------
        index : int
            Index of the selected beam
        """
        if index < 0 or index >= len(self.beams):
            return
            
        self.selected_beam_index = index
        beam = self.beams[index]
        
        # Update UI with beam properties
        self.gantry_spin.setValue(beam.gantry_angle)
        self.collimator_spin.setValue(beam.collimator_angle)
        self.couch_spin.setValue(beam.couch_angle)
        
        # Update energy selection
        energy_index = self.energy_combo.findText(beam.energy)
        if energy_index >= 0:
            self.energy_combo.setCurrentIndex(energy_index)
    
    def _on_gantry_changed(self, value):
        """
        Handle gantry angle change.
        
        Parameters
        ----------
        value : float
            New gantry angle
        """
        if self.selected_beam_index < 0:
            return
            
        self.beams[self.selected_beam_index].gantry_angle = value
        # Emit signal for updating 3D view
    
    def _on_collimator_changed(self, value):
        """
        Handle collimator angle change.
        
        Parameters
        ----------
        value : float
            New collimator angle
        """
        if self.selected_beam_index < 0:
            return
            
        self.beams[self.selected_beam_index].collimator_angle = value
        # Emit signal for updating 3D view
    
    def _on_couch_changed(self, value):
        """
        Handle couch angle change.
        
        Parameters
        ----------
        value : float
            New couch angle
        """
        if self.selected_beam_index < 0:
            return
            
        self.beams[self.selected_beam_index].couch_angle = value
        # Emit signal for updating 3D view
    
    def _add_beam(self):
        """Add a new beam."""
        # Implement beam creation logic
        pass
    
    def _delete_beam(self):
        """Delete the selected beam."""
        if self.selected_beam_index < 0:
            return
            
        # Implement beam deletion logic
        pass


class DVHWidget(QWidget):
    """Widget for displaying Dose-Volume Histograms."""
    
    def __init__(self, parent=None):
        """Initialize the DVH widget."""
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Controls for DVH display
        controls_layout = QHBoxLayout()
        
        # Structure selection
        self.structure_combo = QComboBox()
        controls_layout.addWidget(QLabel("Structure:"))
        controls_layout.addWidget(self.structure_combo)
        
        # DVH type
        self.dvh_type_combo = QComboBox()
        self.dvh_type_combo.addItems(["Cumulative", "Differential"])
        controls_layout.addWidget(QLabel("DVH Type:"))
        controls_layout.addWidget(self.dvh_type_combo)
        
        # Normalization
        self.normalize_check = QCheckBox("Normalize to Prescription")
        controls_layout.addWidget(self.normalize_check)
        
        # Update button
        self.update_btn = QPushButton("Update")
        self.update_btn.clicked.connect(self._update_dvh)
        controls_layout.addWidget(self.update_btn)
        
        self.layout.addLayout(controls_layout)
        
        # Create initial empty plot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Dose-Volume Histogram")
        self.ax.set_xlabel("Dose (Gy)")
        self.ax.set_ylabel("Volume (%)")
        self.ax.grid(True)
        self.canvas.draw()
    
    def set_structures(self, structures):
        """
        Set available structures for DVH calculation.
        
        Parameters
        ----------
        structures : list
            List of structure objects
        """
        self.structure_combo.clear()
        
        for structure in structures:
            self.structure_combo.addItem(structure.name)
    
    def _update_dvh(self):
        """Update the DVH display."""
        # Implement DVH update logic based on selected structures and settings
        pass
    
    def plot_dvh_data(self, dvh_data, structure_name, color=None):
        """
        Plot DVH data for a structure.
        
        Parameters
        ----------
        dvh_data : dict
            DVH data to plot
        structure_name : str
            Name of the structure
        color : str, optional
            Color for the plot
        """
        is_cumulative = self.dvh_type_combo.currentText() == "Cumulative"
        
        # Clear previous plot if this is the first structure
        if len(self.ax.lines) == 0:
            self.ax.clear()
            self.ax.set_title("Dose-Volume Histogram")
            self.ax.set_xlabel("Dose (Gy)")
            self.ax.set_ylabel("Volume (%)")
            self.ax.grid(True)
            
            # Invert y-axis for cumulative DVH
            if is_cumulative:
                self.ax.invert_yaxis()
        
        # Plot the DVH
        dvh_type = "cumulative" if is_cumulative else "differential"
        plot_dvh(
            dvh_data,
            structure_name=structure_name,
            dvh_type=dvh_type,
            ax=self.ax,
            color=color,
            show_metrics=True,
            metrics_to_show=["D95", "D50", "D5", "V20", "V10"]
        )
        
        self.canvas.draw()


class ContourOverlayWidget(QWidget):
    """Widget for controlling contour overlay on images."""
    
    def __init__(self, parent=None):
        """Initialize the contour overlay widget."""
        super().__init__(parent)
        self.structures = []
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        
        # Structure list
        self.structure_list = QTableWidget()
        self.structure_list.setColumnCount(3)
        self.structure_list.setHorizontalHeaderLabels(["Structure", "Type", "Visible"])
        self.structure_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.structure_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.structure_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.layout.addWidget(self.structure_list)
        
        # Controls for contour display
        controls_group = QGroupBox("Display Controls")
        controls_layout = QFormLayout(controls_group)
        
        # Opacity slider
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        controls_layout.addRow("Opacity:", self.opacity_slider)
        
        # Line width
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 5.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setValue(1.5)
        controls_layout.addRow("Line Width:", self.line_width_spin)
        
        # Fill contours
        self.fill_check = QCheckBox("Fill Contours")
        self.fill_check.setChecked(True)
        controls_layout.addRow("", self.fill_check)
        
        # Show in 3D
        self.show_3d_check = QCheckBox("Show in 3D View")
        self.show_3d_check.setChecked(True)
        controls_layout.addRow("", self.show_3d_check)
        
        # Add controls group to layout
        self.layout.addWidget(controls_group)
        
        # Buttons for structure management
        buttons_layout = QHBoxLayout()
        
        self.add_structure_btn = QPushButton("Add Structure")
        buttons_layout.addWidget(self.add_structure_btn)
        
        self.edit_structure_btn = QPushButton("Edit Structure")
        buttons_layout.addWidget(self.edit_structure_btn)
        
        self.delete_structure_btn = QPushButton("Delete Structure")
        buttons_layout.addWidget(self.delete_structure_btn)
        
        self.layout.addLayout(buttons_layout)
        
        # Stretch to fill available space
        self.layout.addStretch()
    
    def set_structures(self, structures):
        """
        Set the structures to display.
        
        Parameters
        ----------
        structures : list
            List of structure objects
        """
        self.structures = structures
        self._update_structure_list()
    
    def _update_structure_list(self):
        """Update the structure list display."""
        self.structure_list.setRowCount(0)
        
        for i, structure in enumerate(self.structures):
            self.structure_list.insertRow(i)
            
            name_item = QTableWidgetItem(structure.name)
            type_item = QTableWidgetItem(structure.type if hasattr(structure, 'type') else "")
            
            visible_checkbox = QCheckBox()
            visible_checkbox.setChecked(True)
            
            self.structure_list.setItem(i, 0, name_item)
            self.structure_list.setItem(i, 1, type_item)
            self.structure_list.setCellWidget(i, 2, visible_checkbox)


class IntegratedViewer(QWidget):
    """
    Integrated viewer combining MPR views, 3D volume rendering, contours, beams and DVH.
    
    This widget provides a comprehensive visualization environment for radiotherapy
    treatment planning, allowing simultaneous viewing of images, contours, dose
    distributions, and treatment beams.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the integrated viewer.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Check VTK availability
        if not VTK_AVAILABLE:
            logger.warning("VTK not available. 3D visualization will be disabled.")
        
        # Internal data storage
        self.patient_id = None
        self.image_data = None
        self.structures = []
        self.beams = []
        self.dose_data = None
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        
        # Toolbar for main controls
        self.toolbar = QToolBar()
        self.main_layout.addWidget(self.toolbar)
        
        # Add toolbar actions
        self.load_patient_action = QAction("Load Patient", self)
        self.load_patient_action.triggered.connect(self._load_patient)
        self.toolbar.addAction(self.load_patient_action)
        
        self.load_image_action = QAction("Load Image", self)
        self.load_image_action.triggered.connect(self._load_image)
        self.toolbar.addAction(self.load_image_action)
        
        self.save_plan_action = QAction("Save Plan", self)
        self.save_plan_action.triggered.connect(self._save_plan)
        self.toolbar.addAction(self.save_plan_action)
        
        self.toolbar.addSeparator()
        
        self.calculate_dose_action = QAction("Calculate Dose", self)
        self.calculate_dose_action.triggered.connect(self._calculate_dose)
        self.toolbar.addAction(self.calculate_dose_action)
        
        self.optimize_plan_action = QAction("Optimize Plan", self)
        self.optimize_plan_action.triggered.connect(self._optimize_plan)
        self.toolbar.addAction(self.optimize_plan_action)
        
        # Main splitter for image views and controls
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # Left side: Image views (MPR + 3D)
        self.views_widget = QWidget()
        self.views_layout = QVBoxLayout(self.views_widget)
        
        # Create layout for 2D views and 3D view
        self.views_splitter = QSplitter(Qt.Vertical)
        
        # 2D MPR views
        self.mpr_viewer = MPRViewer()
        self.views_splitter.addWidget(self.mpr_viewer)
        
        # 3D view
        if VTK_AVAILABLE:
            self.volume_viewer = VolumeRenderingWidget()
            self.views_splitter.addWidget(self.volume_viewer)
        else:
            self.volume_viewer = QLabel("3D visualization requires VTK")
            self.volume_viewer.setAlignment(Qt.AlignCenter)
            self.views_splitter.addWidget(self.volume_viewer)
        
        # Add views to layout
        self.views_layout.addWidget(self.views_splitter)
        
        # Add views widget to main splitter
        self.main_splitter.addWidget(self.views_widget)
        
        # Right side: Controls (tabs for contours, beams, DVH)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        
        # Tab widget for different controls
        self.control_tabs = QTabWidget()
        
        # Contour tab
        self.contour_widget = ContourOverlayWidget()
        self.control_tabs.addTab(self.contour_widget, "Contours")
        
        # Beam tab
        self.beam_widget = BeamVisualizationWidget()
        self.control_tabs.addTab(self.beam_widget, "Beams")
        
        # DVH tab
        self.dvh_widget = DVHWidget()
        self.control_tabs.addTab(self.dvh_widget, "DVH")
        
        # Add tabs to controls layout
        self.controls_layout.addWidget(self.control_tabs)
        
        # Add controls widget to main splitter
        self.main_splitter.addWidget(self.controls_widget)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([700, 300])
        self.views_splitter.setSizes([400, 300])
    
    def set_patient(self, patient_id):
        """
        Set the active patient.
        
        Parameters
        ----------
        patient_id : str
            ID of the patient
        """
        self.patient_id = patient_id
        # Load patient data
    
    def set_image_data(self, image_data, metadata=None):
        """
        Set the image data to visualize.
        
        Parameters
        ----------
        image_data : numpy.ndarray or sitk.Image
            3D image data
        metadata : dict, optional
            Metadata for the image
        """
        self.image_data = image_data
        
        # Update MPR viewer
        if isinstance(image_data, sitk.Image):
            self.mpr_viewer.set_sitk_image(image_data)
        else:
            self.mpr_viewer.set_volume(image_data, metadata)
        
        # Update 3D viewer if available
        if VTK_AVAILABLE and hasattr(self.volume_viewer, 'set_image_data'):
            if isinstance(image_data, sitk.Image):
                self.volume_viewer.set_image_data(image_data)
            else:
                spacing = metadata.get('spacing', (1.0, 1.0, 1.0)) if metadata else (1.0, 1.0, 1.0)
                origin = metadata.get('origin', (0.0, 0.0, 0.0)) if metadata else (0.0, 0.0, 0.0)
                self.volume_viewer.set_image_data(image_data, spacing, origin)
    
    def set_structures(self, structures):
        """
        Set the structures to visualize.
        
        Parameters
        ----------
        structures : list
            List of structure objects
        """
        self.structures = structures
        
        # Update contour widget
        self.contour_widget.set_structures(structures)
        
        # Update DVH widget
        self.dvh_widget.set_structures(structures)
        
        # Update structure visualization in MPR and 3D views
        # (implementation depends on how contours are visualized in those widgets)
    
    def set_beams(self, beams):
        """
        Set the treatment beams to visualize.
        
        Parameters
        ----------
        beams : list
            List of beam objects
        """
        self.beams = beams
        
        # Update beam widget
        self.beam_widget.set_beams(beams)
        
        # Update beam visualization in MPR and 3D views
        # (implementation depends on how beams are visualized in those widgets)
    
    def set_dose_data(self, dose_data, metadata=None):
        """
        Set the dose distribution to visualize.
        
        Parameters
        ----------
        dose_data : numpy.ndarray or sitk.Image
            3D dose distribution
        metadata : dict, optional
            Metadata for the dose
        """
        self.dose_data = dose_data
        
        # Update dose visualization in MPR and 3D views
        # (implementation depends on how dose is visualized in those widgets)
        
        # Update DVH if structures are available
        if self.structures and self.dose_data is not None:
            # Calculate DVH for each structure
            pass
    
    def _load_patient(self):
        """Load patient data."""
        # Implement patient selection and loading logic
        pass
    
    def _load_image(self):
        """Load image data."""
        # Implement image loading logic
        pass
    
    def _save_plan(self):
        """Save the current treatment plan."""
        # Implement plan saving logic
        pass
    
    def _calculate_dose(self):
        """Calculate dose for the current plan."""
        # Implement dose calculation logic
        pass
    
    def _optimize_plan(self):
        """Optimize the current treatment plan."""
        # Implement plan optimization logic
        pass 