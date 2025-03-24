#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Treatment planning tab module for QuangTPS.

This module provides the treatment planning tab for the QuangTPS application,
integrating all planning-related functions into a single interface.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import SimpleITK as sitk

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QMessageBox, QFileDialog, QProgressBar,
    QSplitter, QTabWidget, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

from quangtps.imaging.integrated_viewer import IntegratedViewer
from quangtps.database.patient_db import PatientDatabase
from quangtps.database.structure_db import StructureDatabase
from quangtps.database.dose_db import DoseDB
from quangtps.database.plan_db import PlanDB
from quangtps.planning.treatment_planner import TreatmentPlanner
from quangtps.planning.plan import Plan, PlanType, PlanStatus
from quangtps.dicom.dicom_importer import DicomImporter
from quangtps.dose.dose_engine import DoseEngine

logger = logging.getLogger(__name__)


class TreatmentPlanningTab(QWidget):
    """
    Treatment planning tab for QuangTPS.
    
    This tab provides a comprehensive interface for radiotherapy treatment planning,
    including image visualization, contour editing, beam configuration, dose calculation,
    plan optimization, and evaluation.
    """
    
    # Signal emitted when a plan is saved
    plan_updated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Initialize the treatment planning tab.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Initialize databases
        self.patient_db = PatientDatabase()
        self.structure_db = StructureDatabase()
        self.dose_db = DoseDB()
        self.plan_db = PlanDB()
        
        # Initialize treatment planner
        self.current_patient_id = None
        self.current_plan_id = None
        self.treatment_planner = None
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        self.layout = QVBoxLayout(self)
        
        # Patient and plan selection
        selection_layout = QHBoxLayout()
        
        # Patient selection
        patient_group = QGroupBox("Patient")
        patient_layout = QFormLayout(patient_group)
        
        self.patient_combo = QComboBox()
        self.patient_combo.currentIndexChanged.connect(self._on_patient_selected)
        patient_layout.addRow("Patient:", self.patient_combo)
        
        self.load_patient_btn = QPushButton("Load")
        self.load_patient_btn.clicked.connect(self._on_load_patient)
        patient_layout.addRow("", self.load_patient_btn)
        
        selection_layout.addWidget(patient_group)
        
        # Plan selection
        plan_group = QGroupBox("Treatment Plan")
        plan_layout = QFormLayout(plan_group)
        
        self.plan_combo = QComboBox()
        self.plan_combo.currentIndexChanged.connect(self._on_plan_selected)
        plan_layout.addRow("Plan:", self.plan_combo)
        
        plan_buttons = QHBoxLayout()
        
        self.new_plan_btn = QPushButton("New")
        self.new_plan_btn.clicked.connect(self._on_new_plan)
        plan_buttons.addWidget(self.new_plan_btn)
        
        self.delete_plan_btn = QPushButton("Delete")
        self.delete_plan_btn.clicked.connect(self._on_delete_plan)
        plan_buttons.addWidget(self.delete_plan_btn)
        
        plan_layout.addRow("", plan_buttons)
        
        selection_layout.addWidget(plan_group)
        
        # Progress bar for operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        selection_layout.addWidget(self.progress_bar)
        
        # Add selection controls to main layout
        self.layout.addLayout(selection_layout)
        
        # Integrated viewer
        self.integrated_viewer = IntegratedViewer()
        self.layout.addWidget(self.integrated_viewer)
        
        # Populate patient list
        self._populate_patients()
    
    def _populate_patients(self):
        """Populate the patient selection dropdown."""
        self.patient_combo.clear()
        self.patient_combo.addItem("Select Patient", None)
        
        try:
            patients = self.patient_db.get_all_patients()
            for patient in patients:
                patient_name = patient["name"]
                patient_id = patient["id"]
                self.patient_combo.addItem(f"{patient_name}", patient_id)
        except Exception as e:
            logger.error(f"Error loading patients: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load patients: {str(e)}")
    
    def _populate_plans(self):
        """Populate the plan selection dropdown."""
        self.plan_combo.clear()
        self.plan_combo.addItem("Select Plan", None)
        
        if not self.current_patient_id:
            return
            
        try:
            plans = self.plan_db.get_patient_plans(self.current_patient_id)
            for plan in plans:
                plan_name = plan["name"]
                plan_id = plan["id"]
                self.plan_combo.addItem(f"{plan_name}", plan_id)
        except Exception as e:
            logger.error(f"Error loading plans: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load plans: {str(e)}")
    
    def _on_patient_selected(self, index):
        """
        Handle patient selection.
        
        Parameters
        ----------
        index : int
            Index of the selected patient in the combo box
        """
        if index <= 0:  # "Select Patient" item
            self.current_patient_id = None
            self.plan_combo.clear()
            self.plan_combo.setEnabled(False)
            self.new_plan_btn.setEnabled(False)
            self.delete_plan_btn.setEnabled(False)
            return
            
        # Get selected patient ID
        self.current_patient_id = self.patient_combo.currentData()
        
        # Initialize treatment planner for this patient
        self.treatment_planner = TreatmentPlanner(self.current_patient_id)
        
        # Enable plan controls
        self.plan_combo.setEnabled(True)
        self.new_plan_btn.setEnabled(True)
        
        # Update plan list
        self._populate_plans()
        
        # Load patient image data
        self._load_patient_data()
    
    def _on_plan_selected(self, index):
        """
        Handle plan selection.
        
        Parameters
        ----------
        index : int
            Index of the selected plan in the combo box
        """
        if index <= 0:  # "Select Plan" item
            self.current_plan_id = None
            self.delete_plan_btn.setEnabled(False)
            return
            
        # Get selected plan ID
        self.current_plan_id = self.plan_combo.currentData()
        
        # Enable plan deletion
        self.delete_plan_btn.setEnabled(True)
        
        # Load plan data
        self._load_plan_data()
    
    def _on_load_patient(self):
        """Handle load patient button click."""
        # This would typically open a patient browser dialog
        # For now, just refresh the patient list
        self._populate_patients()
    
    def _on_new_plan(self):
        """Handle new plan button click."""
        if not self.current_patient_id:
            QMessageBox.warning(self, "Warning", "Please select a patient first.")
            return
            
        # This would typically open a dialog to create a new plan
        # For now, just create a simple plan
        try:
            plan_name = f"Plan {self.plan_combo.count()}"
            plan_type = PlanType.DEFINITIVE
            
            # Create a basic plan
            plan = Plan(plan_name, self.current_patient_id, plan_type=plan_type)
            
            # Save to database
            self.plan_db.create_plan(plan)
            
            # Update plan list
            self._populate_plans()
            
            # Select the new plan
            for i in range(self.plan_combo.count()):
                if self.plan_combo.itemData(i) == plan.plan_id:
                    self.plan_combo.setCurrentIndex(i)
                    break
                    
        except Exception as e:
            logger.error(f"Error creating new plan: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to create new plan: {str(e)}")
    
    def _on_delete_plan(self):
        """Handle delete plan button click."""
        if not self.current_plan_id:
            return
            
        # Confirm deletion
        result = QMessageBox.question(
            self, 
            "Confirm Deletion",
            f"Are you sure you want to delete the selected plan?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            try:
                # Delete plan from database
                self.plan_db.delete_plan(self.current_plan_id)
                
                # Update plan list
                self._populate_plans()
                
                # Clear the current plan
                self.current_plan_id = None
                
            except Exception as e:
                logger.error(f"Error deleting plan: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to delete plan: {str(e)}")
    
    def _load_patient_data(self):
        """Load patient image and structure data."""
        if not self.current_patient_id:
            return
            
        try:
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Get patient image data
            self.progress_bar.setValue(25)
            
            # Load primary image (usually CT)
            patient_studies = self.patient_db.get_patient_studies(self.current_patient_id, include_series=True)
            
            ct_series = None
            for study in patient_studies:
                for series in study.get('series', []):
                    if series.get('modality') == 'CT':
                        ct_series = series
                        break
                if ct_series:
                    break
            
            if ct_series:
                # Load CT image
                # This would typically load the actual image data from files
                # For now, just create a dummy image for demonstration
                image_shape = (128, 128, 128)
                image_data = np.zeros(image_shape, dtype=np.float32)
                
                # Create a SimpleITK image
                sitk_image = sitk.GetImageFromArray(image_data)
                sitk_image.SetSpacing((1.0, 1.0, 1.0))
                sitk_image.SetOrigin((0.0, 0.0, 0.0))
                
                # Set image in the viewer
                self.integrated_viewer.set_image_data(sitk_image)
            
            self.progress_bar.setValue(50)
            
            # Load structures
            structures = self.structure_db.get_patient_structures(self.current_patient_id)
            
            # Convert to structure objects the viewer can use
            # This is a placeholder - actual implementation would create proper structure objects
            viewer_structures = []
            for structure_data in structures:
                structure = type('Structure', (), {
                    'name': structure_data.get('name', 'Unknown'),
                    'type': structure_data.get('type', ''),
                    'color': structure_data.get('color', '#FF0000')
                })
                viewer_structures.append(structure)
            
            # Set structures in the viewer
            self.integrated_viewer.set_structures(viewer_structures)
            
            self.progress_bar.setValue(100)
            
        except Exception as e:
            logger.error(f"Error loading patient data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load patient data: {str(e)}")
        finally:
            # Hide progress
            self.progress_bar.setVisible(False)
    
    def _load_plan_data(self):
        """Load plan data including beams and dose."""
        if not self.current_plan_id:
            return
            
        try:
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Get plan data
            plan_data = self.plan_db.get_plan(self.current_plan_id)
            
            self.progress_bar.setValue(25)
            
            # Load beams
            beams = self.plan_db.get_plan_beams(self.current_plan_id)
            
            # Convert to beam objects the viewer can use
            # This is a placeholder - actual implementation would create proper beam objects
            viewer_beams = []
            for beam_data in beams:
                beam = type('Beam', (), {
                    'name': beam_data.get('name', 'Unknown'),
                    'gantry_angle': beam_data.get('gantry_angle', 0),
                    'collimator_angle': beam_data.get('collimator_angle', 0),
                    'couch_angle': beam_data.get('couch_angle', 0),
                    'energy': beam_data.get('energy', '6MV')
                })
                viewer_beams.append(beam)
            
            # Set beams in the viewer
            self.integrated_viewer.set_beams(viewer_beams)
            
            self.progress_bar.setValue(50)
            
            # Load dose data if available
            dose_distributions = self.dose_db.get_plan_doses(self.current_plan_id)
            
            if dose_distributions:
                # Load the first dose distribution
                dose_data = dose_distributions[0]
                
                # This would typically load the actual dose data from files
                # For now, just create a dummy dose for demonstration
                dose_shape = (128, 128, 128)
                dose_array = np.zeros(dose_shape, dtype=np.float32)
                
                # Create a SimpleITK image
                dose_image = sitk.GetImageFromArray(dose_array)
                dose_image.SetSpacing((1.0, 1.0, 1.0))
                dose_image.SetOrigin((0.0, 0.0, 0.0))
                
                # Set dose in the viewer
                self.integrated_viewer.set_dose_data(dose_image)
            
            self.progress_bar.setValue(100)
            
        except Exception as e:
            logger.error(f"Error loading plan data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load plan data: {str(e)}")
        finally:
            # Hide progress
            self.progress_bar.setVisible(False)
    
    def set_patient(self, patient_id):
        """
        Set the active patient.
        
        Parameters
        ----------
        patient_id : str
            ID of the patient to set as active
        """
        # Find the patient in the combo box
        for i in range(self.patient_combo.count()):
            if self.patient_combo.itemData(i) == patient_id:
                self.patient_combo.setCurrentIndex(i)
                return
        
        # If patient not found, refresh list and try again
        self._populate_patients()
        for i in range(self.patient_combo.count()):
            if self.patient_combo.itemData(i) == patient_id:
                self.patient_combo.setCurrentIndex(i)
                return 