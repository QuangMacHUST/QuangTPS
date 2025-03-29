#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dose calculation interface module.

This module provides a user interface for performing dose calculations
using various algorithms available in QuangTPS.
"""

import os
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QProgressBar, QSplitter, QFileDialog,
    QMessageBox, QRadioButton, QButtonGroup, QScrollArea, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QPixmap, QIcon, QPalette

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from quangtps.core.exceptions import DoseCalculationError
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.dose.algorithms import (
    DoseCalculationAlgorithm, 
    PencilBeamAlgorithm, 
    CollapsedConeAlgorithm, 
    MonteCarloAlgorithm
)
from quangtps.dose.beam_data_processor import BeamModel
from quangtps.ui.utils import get_icon_path, create_info_icon

logger = logging.getLogger(__name__)

class DoseCalculationThread(QThread):
    """Thread for running dose calculation without blocking the UI."""
    
    # Signal emitted when progress updates
    progress_updated = pyqtSignal(int)
    
    # Signal emitted when calculation finishes
    calculation_finished = pyqtSignal(object)
    
    # Signal emitted when an error occurs
    error_occurred = pyqtSignal(str)
    
    def __init__(self, algorithm: DoseCalculationAlgorithm, beam: Beam, 
                 ct_image: Image, parent=None):
        """Initialize thread with calculation parameters."""
        super().__init__(parent)
        self.algorithm = algorithm
        self.beam = beam
        self.ct_image = ct_image
        self.result = None
        
        # Flag to indicate if the thread should continue running
        self.keep_running = True
    
    def run(self):
        """Run the dose calculation in a separate thread."""
        try:
            start_time = time.time()
            
            # Emit initial progress
            self.progress_updated.emit(0)
            
            # Run calculation
            self.result = self.algorithm.calculate(self.ct_image, self.beam)
            
            # Emit final progress if successful
            self.progress_updated.emit(100)
            
            # Emit result
            self.calculation_finished.emit(self.result)
            
            logger.info(f"Dose calculation completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in dose calculation thread: {str(e)}")
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """Stop the thread."""
        self.keep_running = False
        self.wait()

class DoseAlgorithmWidget(QWidget):
    """Widget for displaying and configuring a dose calculation algorithm."""
    
    def __init__(self, algorithm: DoseCalculationAlgorithm, parent=None):
        """Initialize widget with an algorithm."""
        super().__init__(parent)
        self.algorithm = algorithm
        
        # Store controls for parameters
        self.parameter_controls = {}
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout()
        
        # Header with algorithm name and version
        header_layout = QHBoxLayout()
        name_label = QLabel(f"<b>{self.algorithm.name}</b> (v{self.algorithm.version})")
        name_label.setStyleSheet("font-size: 14px;")
        header_layout.addWidget(name_label)
        
        # Add info icon with tooltip description
        info_icon = create_info_icon(self.algorithm.get_description())
        header_layout.addWidget(info_icon)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Divider line
        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #cccccc;")
        layout.addWidget(divider)
        
        # Parameter group box
        param_group = QGroupBox("Algorithm Parameters")
        param_layout = QVBoxLayout()
        
        # Get parameter info
        param_info = self.algorithm.get_parameters_info()
        
        # Create controls for each parameter
        for param_name, param_data in param_info.items():
            param_layout.addLayout(self._create_parameter_control(param_name, param_data))
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Add reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_parameters)
        layout.addWidget(reset_button)
        
        # Add stretch at the end
        layout.addStretch()
        
        self.setLayout(layout)
    
    def _create_parameter_control(self, param_name: str, param_data: Dict[str, Any]) -> QHBoxLayout:
        """Create a control for a parameter."""
        # Layout for this parameter
        param_layout = QHBoxLayout()
        
        # Parameter name label with tooltip
        name_label = QLabel(param_name.replace('_', ' ').title())
        name_label.setToolTip(param_data.get('description', ''))
        param_layout.addWidget(name_label, 1)
        
        # Create appropriate control based on parameter type
        param_type = param_data.get('type', 'float')
        default_value = param_data.get('default', 0)
        current_value = self.algorithm.parameters.get(param_name, default_value)
        
        if param_type == 'bool':
            # Boolean parameter (checkbox)
            control = QCheckBox()
            control.setChecked(current_value)
            control.stateChanged.connect(
                lambda state, name=param_name: self._update_parameter(name, bool(state))
            )
        
        elif param_type == 'int':
            # Integer parameter (spin box)
            control = QSpinBox()
            min_val = param_data.get('range', [0, 100])[0]
            max_val = param_data.get('range', [0, 100])[1]
            control.setRange(min_val, max_val)
            control.setValue(current_value)
            control.valueChanged.connect(
                lambda value, name=param_name: self._update_parameter(name, value)
            )
        
        elif param_type == 'float':
            # Float parameter (double spin box)
            control = QDoubleSpinBox()
            min_val = param_data.get('range', [0.0, 10.0])[0]
            max_val = param_data.get('range', [0.0, 10.0])[1]
            control.setRange(min_val, max_val)
            control.setDecimals(3)
            control.setSingleStep(0.1)
            control.setValue(current_value)
            control.valueChanged.connect(
                lambda value, name=param_name: self._update_parameter(name, value)
            )
        
        elif param_type == 'str' and 'options' in param_data:
            # String with options (combo box)
            control = QComboBox()
            control.addItems(param_data['options'])
            index = param_data['options'].index(current_value) if current_value in param_data['options'] else 0
            control.setCurrentIndex(index)
            control.currentTextChanged.connect(
                lambda text, name=param_name: self._update_parameter(name, text)
            )
        
        else:
            # Fallback to label for unsupported types
            control = QLabel(str(current_value))
        
        # Store control for later access
        self.parameter_controls[param_name] = control
        
        param_layout.addWidget(control, 2)
        
        return param_layout
    
    def _update_parameter(self, name: str, value: Any):
        """Update a parameter in the algorithm."""
        self.algorithm.parameters[name] = value
        logger.debug(f"Updated parameter {name} = {value}")
    
    def _reset_parameters(self):
        """Reset all parameters to their default values."""
        param_info = self.algorithm.get_parameters_info()
        
        for param_name, param_data in param_info.items():
            default_value = param_data.get('default')
            
            if default_value is not None:
                # Update algorithm parameter
                self.algorithm.parameters[param_name] = default_value
                
                # Update control
                control = self.parameter_controls.get(param_name)
                if control:
                    if isinstance(control, QCheckBox):
                        control.setChecked(default_value)
                    elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                        control.setValue(default_value)
                    elif isinstance(control, QComboBox):
                        try:
                            index = param_data['options'].index(default_value)
                            control.setCurrentIndex(index)
                        except (ValueError, KeyError):
                            pass
        
        logger.info("Reset parameters to defaults")

class DoseCalculatorWidget(QWidget):
    """Main widget for dose calculation interface."""
    
    def __init__(self, parent=None):
        """Initialize the dose calculator widget."""
        super().__init__(parent)
        
        # Initialize algorithms
        self.algorithms = {
            'pencil_beam': PencilBeamAlgorithm(),
            'collapsed_cone': CollapsedConeAlgorithm(),
            'monte_carlo': MonteCarloAlgorithm()
        }
        
        # Current state
        self.current_beam = None
        self.current_ct = None
        self.current_plan = None
        self.beam_model = None
        self.calculation_thread = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Add algorithm selection
        algorithm_group = QGroupBox("Dose Calculation Algorithm")
        algorithm_layout = QVBoxLayout()
        
        self.algorithm_combo = QComboBox()
        for alg_id, algorithm in self.algorithms.items():
            self.algorithm_combo.addItem(algorithm.name, alg_id)
        
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        algorithm_layout.addWidget(self.algorithm_combo)
        
        algorithm_group.setLayout(algorithm_layout)
        main_layout.addWidget(algorithm_group)
        
        # Create tab widget for algorithm parameters and results
        self.tab_widget = QTabWidget()
        
        # Create and add parameter widgets for each algorithm
        self.algorithm_widgets = {}
        for alg_id, algorithm in self.algorithms.items():
            widget = DoseAlgorithmWidget(algorithm)
            self.algorithm_widgets[alg_id] = widget
            
            # Add to a scroll area to handle overflow
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            
            self.tab_widget.addTab(scroll, "Parameters")
        
        # Add results tab
        self.results_widget = QWidget()
        results_layout = QVBoxLayout()
        
        # Add plot for dose visualization
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        results_layout.addWidget(self.toolbar)
        results_layout.addWidget(self.canvas)
        
        # Add statistics table for dose results
        self.stats_tree = QTreeWidget()
        self.stats_tree.setHeaderLabels(["Metric", "Value"])
        self.stats_tree.setMinimumHeight(150)
        results_layout.addWidget(self.stats_tree)
        
        self.results_widget.setLayout(results_layout)
        self.tab_widget.addTab(self.results_widget, "Results")
        
        main_layout.addWidget(self.tab_widget)
        
        # Add calculation controls
        control_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        self.calculate_button = QPushButton("Calculate Dose")
        self.calculate_button.clicked.connect(self._on_calculate)
        self.calculate_button.setEnabled(False)  # Disabled until beam and CT are set
        control_layout.addWidget(self.calculate_button)
        
        main_layout.addLayout(control_layout)
        
        self.setLayout(main_layout)
        
        # Update UI to show current algorithm
        self._on_algorithm_changed(self.algorithm_combo.currentIndex())
    
    def _on_algorithm_changed(self, index: int):
        """Handle algorithm selection change."""
        if index < 0:
            return
        
        alg_id = self.algorithm_combo.itemData(index)
        
        # Replace the parameters tab with the appropriate widget
        self.tab_widget.removeTab(0)
        
        # Add to a scroll area to handle overflow
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.algorithm_widgets[alg_id])
        
        self.tab_widget.insertTab(0, scroll, "Parameters")
        self.tab_widget.setCurrentIndex(0)
        
        logger.info(f"Selected algorithm: {self.algorithms[alg_id].name}")
    
    def _on_calculate(self):
        """Handle calculate button click."""
        # Get current algorithm
        alg_id = self.algorithm_combo.itemData(self.algorithm_combo.currentIndex())
        algorithm = self.algorithms[alg_id]
        
        # Validate that we have required data
        if self.current_beam is None or self.current_ct is None:
            QMessageBox.warning(
                self, 
                "Calculation Error", 
                "Beam and CT image must be set before calculation"
            )
            return
        
        # Check if calculation is already running
        if self.calculation_thread and self.calculation_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Calculation In Progress",
                "A calculation is already running. Do you want to stop it and start a new one?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.calculation_thread.stop()
            else:
                return
        
        # Update UI
        self.calculate_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Set beam model if available
        if self.beam_model:
            algorithm.set_beam_model(self.beam_model)
        
        # Create and start calculation thread
        self.calculation_thread = DoseCalculationThread(
            algorithm=algorithm,
            beam=self.current_beam,
            ct_image=self.current_ct,
            parent=self
        )
        
        # Connect signals
        self.calculation_thread.progress_updated.connect(self.progress_bar.setValue)
        self.calculation_thread.calculation_finished.connect(self._on_calculation_finished)
        self.calculation_thread.error_occurred.connect(self._on_calculation_error)
        
        # Start calculation
        self.calculation_thread.start()
        
        logger.info(f"Started dose calculation with {algorithm.name}")
    
    def _on_calculation_finished(self, result):
        """Handle calculation completion."""
        # Update UI
        self.calculate_button.setEnabled(True)
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(1)
        
        # Plot results
        self._plot_dose_results(result)
        
        # Update statistics
        self._update_statistics(result)
        
        logger.info("Dose calculation completed successfully")
    
    def _on_calculation_error(self, error_msg):
        """Handle calculation error."""
        # Update UI
        self.calculate_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Show error message
        QMessageBox.critical(
            self,
            "Calculation Error",
            f"Error during dose calculation:\n{error_msg}"
        )
        
        logger.error(f"Dose calculation error: {error_msg}")
    
    def _plot_dose_results(self, result):
        """Plot dose calculation results."""
        # Clear figure
        self.figure.clear()
        
        # Extract dose array
        dose_data = result.dose_data
        
        # Get middle slices
        nx, ny, nz = dose_data.shape
        mid_x = nx // 2
        mid_y = ny // 2
        mid_z = nz // 2
        
        # Create 3 subplots for different views
        axial = self.figure.add_subplot(221)
        sagittal = self.figure.add_subplot(222)
        coronal = self.figure.add_subplot(223)
        colorbar_ax = self.figure.add_subplot(224)
        
        # Plot axial view
        im_axial = axial.imshow(
            dose_data[:, :, mid_z].T,
            cmap='jet',
            vmin=0,
            vmax=100
        )
        axial.set_title('Axial (XY)')
        axial.set_xlabel('X')
        axial.set_ylabel('Y')
        
        # Plot sagittal view
        im_sagittal = sagittal.imshow(
            dose_data[mid_x, :, :].T,
            cmap='jet',
            vmin=0,
            vmax=100
        )
        sagittal.set_title('Sagittal (YZ)')
        sagittal.set_xlabel('Y')
        sagittal.set_ylabel('Z')
        
        # Plot coronal view
        im_coronal = coronal.imshow(
            dose_data[:, mid_y, :].T,
            cmap='jet',
            vmin=0,
            vmax=100
        )
        coronal.set_title('Coronal (XZ)')
        coronal.set_xlabel('X')
        coronal.set_ylabel('Z')
        
        # Add colorbar
        self.figure.colorbar(im_axial, cax=colorbar_ax, label='Dose (%)')
        
        # Add algorithm info
        algorithm_info = f"{result.algorithm} v{result.version}\nCalc time: {result.calculation_time:.2f}s"
        self.figure.text(0.5, 0.95, algorithm_info, ha='center', va='top')
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _update_statistics(self, result):
        """Update statistics table with dose results."""
        # Clear existing items
        self.stats_tree.clear()
        
        # Extract dose array
        dose_data = result.dose_data
        
        # Basic statistics
        stats = {
            "Calculation Information": {
                "Algorithm": result.algorithm,
                "Version": result.version,
                "Calculation Time": f"{result.calculation_time:.2f} seconds"
            },
            "Dose Statistics": {
                "Maximum Dose": f"{np.max(dose_data):.2f}%",
                "Minimum Dose": f"{np.min(dose_data):.2f}%",
                "Mean Dose": f"{np.mean(dose_data):.2f}%",
                "Standard Deviation": f"{np.std(dose_data):.2f}%"
            },
            "Beam Information": {
                "Name": result.beam_info.get('name', 'Unknown'),
                "Energy": result.beam_info.get('energy', 'Unknown'),
                "Gantry Angle": f"{result.beam_info.get('gantry_angle', 0):.1f}°",
                "Field Size": f"{result.beam_info.get('field_size', (0, 0))[0]}×{result.beam_info.get('field_size', (0, 0))[1]} cm²"
            },
            "Volume Metrics": {
                "V90": f"{self._calculate_volume_coverage(dose_data, 90):.2f}%",
                "V80": f"{self._calculate_volume_coverage(dose_data, 80):.2f}%",
                "V50": f"{self._calculate_volume_coverage(dose_data, 50):.2f}%",
                "V20": f"{self._calculate_volume_coverage(dose_data, 20):.2f}%"
            }
        }
        
        # Add algorithm parameters
        param_group = {"Algorithm Parameters": {}}
        for param_name, param_value in result.parameters.items():
            param_group["Algorithm Parameters"][param_name] = str(param_value)
        
        stats.update(param_group)
        
        # Add any additional data
        if hasattr(result, 'additional_data') and result.additional_data:
            add_group = {"Additional Data": {}}
            for key, value in result.additional_data.items():
                if isinstance(value, np.ndarray):
                    add_group["Additional Data"][key] = "Array data"
                else:
                    add_group["Additional Data"][key] = str(value)
            
            stats.update(add_group)
        
        # Populate tree
        for group_name, items in stats.items():
            group_item = QTreeWidgetItem(self.stats_tree, [group_name, ""])
            
            for key, value in items.items():
                QTreeWidgetItem(group_item, [key, str(value)])
            
            group_item.setExpanded(True)
        
        # Resize columns to content
        self.stats_tree.resizeColumnToContents(0)
        self.stats_tree.resizeColumnToContents(1)
    
    def _calculate_volume_coverage(self, dose_data, threshold):
        """Calculate percentage of volume receiving at least threshold dose."""
        total_voxels = np.sum(dose_data > 0)
        if total_voxels == 0:
            return 0.0
        
        covered_voxels = np.sum(dose_data >= threshold)
        return (covered_voxels / total_voxels) * 100
    
    def set_beam(self, beam: Beam):
        """Set the current beam for dose calculation."""
        self.current_beam = beam
        self._update_calculation_button()
        logger.info(f"Set beam: {beam.name if beam else 'None'}")
    
    def set_ct_image(self, ct_image: Image):
        """Set the current CT image for dose calculation."""
        self.current_ct = ct_image
        self._update_calculation_button()
        logger.info(f"Set CT image: {'Valid image' if ct_image else 'None'}")
    
    def set_beam_model(self, beam_model: BeamModel):
        """Set the beam model for dose calculation."""
        self.beam_model = beam_model
        
        # Apply to all algorithms
        for algorithm in self.algorithms.values():
            algorithm.set_beam_model(beam_model)
        
        logger.info(f"Set beam model: {beam_model.name if beam_model else 'None'}")
    
    def set_plan(self, plan: Plan):
        """Set the current plan for dose calculation."""
        self.current_plan = plan
        logger.info(f"Set plan: {plan.name if plan else 'None'}")
    
    def _update_calculation_button(self):
        """Update the state of the calculate button."""
        self.calculate_button.setEnabled(
            self.current_beam is not None and 
            self.current_ct is not None
        )

def main():
    """Run the dose calculator widget as a standalone application."""
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show widget
    widget = DoseCalculatorWidget()
    widget.setWindowTitle("QuangTPS Dose Calculator")
    widget.resize(1000, 800)
    widget.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 