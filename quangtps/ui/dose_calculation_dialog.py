#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog for dose calculation settings and execution.

This module provides a dialog for configuring and executing dose
calculation using various algorithms.
"""

import os
import logging
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QFileDialog, QMessageBox, QTabWidget,
    QWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from quangtps.core.exceptions import DoseCalculationError
from quangtps.dose.dose_calculator import DoseCalculator
from quangtps.dose.algorithms import AVAILABLE_ALGORITHMS
from quangtps.planning.beam import Beam
from quangtps.planning.plan import TreatmentPlan
from quangtps.imaging.image import Image

logger = logging.getLogger(__name__)

class DoseCalculationWorker(QThread):
    """
    Worker thread for dose calculation to prevent UI freezing.
    """
    
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, calculator, plan, ct_image, params):
        """
        Initialize the worker.
        
        Parameters
        ----------
        calculator : DoseCalculator
            The dose calculator object
        plan : TreatmentPlan
            The treatment plan to calculate dose for
        ct_image : Image
            The CT image for dose calculation
        params : dict
            Additional parameters for the calculation
        """
        super().__init__()
        self.calculator = calculator
        self.plan = plan
        self.ct_image = ct_image
        self.params = params
    
    def run(self):
        """Run the dose calculation."""
        try:
            # Set algorithm-specific parameters
            algorithm = self.calculator.algorithm
            for param_name, value in self.params.items():
                if hasattr(algorithm, f"set_{param_name}"):
                    getattr(algorithm, f"set_{param_name}")(value)
                elif hasattr(algorithm, "set_parameter"):
                    algorithm.set_parameter(param_name, value)
            
            # Report progress for preparation
            self.progress_signal.emit(5)
            
            # Calculate dose for each beam
            num_beams = len(self.plan.beams)
            beam_progress_step = 90 / max(1, num_beams)
            
            # If calculating for individual beams
            if self.params.get("calculate_individual_beams", False):
                beam_doses = []
                for i, beam in enumerate(self.plan.beams):
                    # Calculate dose for single beam
                    beam_dose = self.calculator.calculate_dose_for_beam(beam, self.ct_image)
                    beam_doses.append(beam_dose)
                    
                    # Update progress
                    progress = 5 + int((i + 1) * beam_progress_step)
                    self.progress_signal.emit(progress)
                
                # Sum beam doses to get total plan dose
                total_dose = Image(
                    data=np.zeros_like(beam_doses[0].data),
                    spacing=beam_doses[0].spacing,
                    origin=beam_doses[0].origin,
                    direction=beam_doses[0].direction
                )
                
                # Apply weights and sum
                for i, beam_dose in enumerate(beam_doses):
                    weight = self.plan.beams[i].weight
                    total_dose.data += beam_dose.data * weight
                
                # Normalize if needed
                if self.plan.normalization_value is not None:
                    normalization_factor = self.plan.normalization_value / np.max(total_dose.data)
                    total_dose.data *= normalization_factor
            else:
                # Calculate dose for entire plan directly
                total_dose = self.calculator.calculate_dose_for_plan(self.plan, self.ct_image)
            
            # Set final properties
            total_dose.modality = "RTDOSE"
            total_dose.description = f"Plan dose for {self.plan.name} calculated with {self.calculator.algorithm_name}"
            
            # Final progress
            self.progress_signal.emit(100)
            
            # Emit result
            self.finished_signal.emit(total_dose)
            
        except Exception as e:
            logger.error(f"Error in dose calculation: {str(e)}")
            self.error_signal.emit(str(e))

class DoseCalculationDialog(QDialog):
    """
    Dialog for configuring and executing dose calculation.
    """
    
    def __init__(self, parent=None, plan=None, ct_image=None):
        """
        Initialize the dialog.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        plan : TreatmentPlan, optional
            Treatment plan to calculate dose for
        ct_image : Image, optional
            CT image for dose calculation
        """
        super().__init__(parent)
        
        self.plan = plan
        self.ct_image = ct_image
        self.calculator = DoseCalculator()
        self.result_dose = None
        
        self.setWindowTitle("Dose Calculation")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Main tabs
        tab_widget = QTabWidget()
        
        # Basic settings tab
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # Algorithm selection group
        algorithm_group = QGroupBox("Calculation Algorithm")
        algorithm_layout = QFormLayout(algorithm_group)
        
        self.algorithm_combo = QComboBox()
        for algo_name in sorted(AVAILABLE_ALGORITHMS.keys()):
            self.algorithm_combo.addItem(algo_name)
        self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        algorithm_layout.addRow("Algorithm:", self.algorithm_combo)
        
        self.machine_combo = QComboBox()
        self.machine_combo.addItems(["Generic", "Truebeam", "Halcyon", "VitalBeam"])
        algorithm_layout.addRow("Machine:", self.machine_combo)
        
        basic_layout.addWidget(algorithm_group)
        
        # Algorithm parameters group
        self.params_group = QGroupBox("Algorithm Parameters")
        self.params_layout = QFormLayout(self.params_group)
        
        # Common parameters
        self.heterogeneity_checkbox = QCheckBox("Enable heterogeneity correction")
        self.heterogeneity_checkbox.setChecked(True)
        self.params_layout.addRow("", self.heterogeneity_checkbox)
        
        self.grid_size_spin = QDoubleSpinBox()
        self.grid_size_spin.setRange(0.1, 1.0)
        self.grid_size_spin.setSingleStep(0.05)
        self.grid_size_spin.setValue(0.25)
        self.grid_size_spin.setSuffix(" cm")
        self.params_layout.addRow("Grid size:", self.grid_size_spin)
        
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(8)
        self.params_layout.addRow("Threads:", self.threads_spin)
        
        # Monte Carlo specific parameters (hidden by default)
        self.mc_group = QGroupBox("Monte Carlo Settings")
        self.mc_layout = QFormLayout(self.mc_group)
        
        self.histories_spin = QSpinBox()
        self.histories_spin.setRange(10000, 100000000)
        self.histories_spin.setSingleStep(10000)
        self.histories_spin.setValue(1000000)
        self.mc_layout.addRow("Histories:", self.histories_spin)
        
        self.uncertainty_spin = QDoubleSpinBox()
        self.uncertainty_spin.setRange(0.1, 10.0)
        self.uncertainty_spin.setSingleStep(0.1)
        self.uncertainty_spin.setValue(2.0)
        self.uncertainty_spin.setSuffix(" %")
        self.mc_layout.addRow("Statistical uncertainty:", self.uncertainty_spin)
        
        self.use_gpu_checkbox = QCheckBox("Use GPU acceleration (if available)")
        self.use_gpu_checkbox.setChecked(False)
        self.mc_layout.addRow("", self.use_gpu_checkbox)
        
        # Hide MC settings by default
        self.mc_group.setVisible(False)
        
        basic_layout.addWidget(self.params_group)
        basic_layout.addWidget(self.mc_group)
        
        # Beam settings group
        beam_group = QGroupBox("Beam Settings")
        beam_layout = QFormLayout(beam_group)
        
        self.individual_beams_checkbox = QCheckBox("Calculate individual beam doses")
        self.individual_beams_checkbox.setChecked(False)
        beam_layout.addRow("", self.individual_beams_checkbox)
        
        if self.plan:
            beam_info = QLabel(f"Plan: {self.plan.name}, {len(self.plan.beams)} beams")
        else:
            beam_info = QLabel("No plan selected")
        beam_layout.addRow("Plan info:", beam_info)
        
        basic_layout.addWidget(beam_group)
        
        # Add stretch to push everything to the top
        basic_layout.addStretch()
        
        # Advanced settings tab
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        
        # Beam model settings
        model_group = QGroupBox("Beam Model Settings")
        model_layout = QFormLayout(model_group)
        
        self.model_dir_edit = QLabel("Default")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_model_dir)
        
        model_dir_layout = QHBoxLayout()
        model_dir_layout.addWidget(self.model_dir_edit)
        model_dir_layout.addWidget(browse_button)
        model_layout.addRow("Model directory:", model_dir_layout)
        
        # Show available models
        self.available_models_label = QLabel("No beam models loaded")
        self.reload_models_button = QPushButton("Reload Models")
        self.reload_models_button.clicked.connect(self._reload_models)
        
        model_buttons_layout = QHBoxLayout()
        model_buttons_layout.addWidget(self.available_models_label)
        model_buttons_layout.addWidget(self.reload_models_button)
        model_layout.addRow("Status:", model_buttons_layout)
        
        advanced_layout.addWidget(model_group)
        
        # Add output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)
        
        self.save_dose_checkbox = QCheckBox("Save dose to file after calculation")
        self.save_dose_checkbox.setChecked(False)
        output_layout.addRow("", self.save_dose_checkbox)
        
        advanced_layout.addWidget(output_group)
        
        # Add stretch to push everything to the top
        advanced_layout.addStretch()
        
        # Add tabs to widget
        tab_widget.addTab(basic_tab, "Basic Settings")
        tab_widget.addTab(advanced_tab, "Advanced Settings")
        
        layout.addWidget(tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self._calculate_dose)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        button_layout.addWidget(self.calculate_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Initial setup
        self._on_algorithm_changed(self.algorithm_combo.currentText())
        self._reload_models()
    
    def _on_algorithm_changed(self, algorithm_name):
        """
        Handle algorithm selection change.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the selected algorithm
        """
        # Set the calculator algorithm
        try:
            self.calculator.set_algorithm(algorithm_name)
        except ValueError as e:
            logger.error(f"Error setting algorithm: {str(e)}")
            QMessageBox.warning(self, "Algorithm Error", f"Error setting algorithm: {str(e)}")
        
        # Show/hide algorithm-specific settings
        self.mc_group.setVisible(algorithm_name == "MONTE_CARLO")
    
    def _browse_model_dir(self):
        """Browse for beam model directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Beam Model Directory")
        if directory:
            self.model_dir_edit.setText(directory)
            self.calculator = DoseCalculator(
                algorithm=self.algorithm_combo.currentText(),
                beam_model_dir=directory
            )
            self._reload_models()
    
    def _reload_models(self):
        """Reload available beam models."""
        try:
            available_models = self.calculator.get_available_beam_models()
            
            # Count total models
            total_models = 0
            for machine, beam_types in available_models.items():
                for beam_type, energies in beam_types.items():
                    total_models += len(energies)
            
            self.available_models_label.setText(f"{total_models} models loaded")
            
            # Enable machine selection based on available models
            current_machine = self.machine_combo.currentText()
            self.machine_combo.clear()
            
            # Always add Generic
            self.machine_combo.addItem("Generic")
            
            # Add other machines if models are available
            for machine in available_models.keys():
                if machine.lower() != "generic":
                    self.machine_combo.addItem(machine.capitalize())
            
            # Try to restore previous selection
            index = self.machine_combo.findText(current_machine, Qt.MatchExactly)
            if index >= 0:
                self.machine_combo.setCurrentIndex(index)
            
        except Exception as e:
            logger.error(f"Error loading beam models: {str(e)}")
            self.available_models_label.setText("Error loading models")
    
    def _calculate_dose(self):
        """Execute dose calculation."""
        if not self.plan or not self.ct_image:
            QMessageBox.warning(self, "Missing Data", "Treatment plan or CT image is missing")
            return
        
        # Collect parameters
        params = {
            "heterogeneity_correction": self.heterogeneity_checkbox.isChecked(),
            "grid_size": self.grid_size_spin.value(),
            "threads": self.threads_spin.value(),
            "calculate_individual_beams": self.individual_beams_checkbox.isChecked()
        }
        
        if self.algorithm_combo.currentText() == "MONTE_CARLO":
            params.update({
                "num_histories": self.histories_spin.value(),
                "statistical_uncertainty": self.uncertainty_spin.value(),
                "use_gpu": self.use_gpu_checkbox.isChecked()
            })
        
        # Disable UI during calculation
        self._set_ui_enabled(False)
        
        # Create and start worker thread
        self.worker = DoseCalculationWorker(self.calculator, self.plan, self.ct_image, params)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self._calculation_finished)
        self.worker.error_signal.connect(self._calculation_error)
        self.worker.start()
    
    def _calculation_finished(self, dose_image):
        """
        Handle completion of dose calculation.
        
        Parameters
        ----------
        dose_image : Image
            The calculated dose image
        """
        self.result_dose = dose_image
        
        # Save dose if requested
        if self.save_dose_checkbox.isChecked():
            # Ask for file location
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Dose File", "", "DICOM Files (*.dcm);;All Files (*)"
            )
            if file_path:
                try:
                    # Save to file
                    dose_image.save(file_path)
                    logger.info(f"Dose saved to {file_path}")
                except Exception as e:
                    logger.error(f"Error saving dose: {str(e)}")
                    QMessageBox.warning(self, "Save Error", f"Error saving dose: {str(e)}")
        
        # Show success message
        QMessageBox.information(self, "Calculation Complete", 
                               "Dose calculation completed successfully")
        
        # Re-enable UI
        self._set_ui_enabled(True)
    
    def _calculation_error(self, error_message):
        """
        Handle error in dose calculation.
        
        Parameters
        ----------
        error_message : str
            Error message
        """
        logger.error(f"Dose calculation error: {error_message}")
        QMessageBox.critical(self, "Calculation Error", 
                            f"Error during dose calculation:\n{error_message}")
        
        # Re-enable UI
        self._set_ui_enabled(True)
    
    def _set_ui_enabled(self, enabled):
        """
        Enable or disable UI elements during calculation.
        
        Parameters
        ----------
        enabled : bool
            Whether UI should be enabled
        """
        self.algorithm_combo.setEnabled(enabled)
        self.machine_combo.setEnabled(enabled)
        self.params_group.setEnabled(enabled)
        self.mc_group.setEnabled(enabled)
        self.calculate_button.setEnabled(enabled)
        self.close_button.setEnabled(enabled) 