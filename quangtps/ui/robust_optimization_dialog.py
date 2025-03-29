#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust optimization dialog for radiation therapy treatment planning.

This module provides a dialog for configuring and executing
robust optimization of treatment plans, accounting for setup
uncertainties and range uncertainties.
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union, Any

from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer, QMetaObject, Q_ARG
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPainter, QPen, QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QCheckBox, QGroupBox, 
    QPushButton, QComboBox, QTabWidget, QWidget,
    QSpinBox, QDoubleSpinBox, QLineEdit, QFrame,
    QScrollArea, QSplitter, QFileDialog, QMessageBox,
    QProgressBar, QRadioButton, QButtonGroup
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from quangtps.core.plan import Plan
from quangtps.core.structure import Structure
from quangtps.core.types import DoseGrid
from quangtps.core.logging import get_logger
from quangtps.dose.calculation import DoseCalculator
from quangtps.optimization.objectives import PlanningObjectives, ObjectiveFunction
from quangtps.optimization.constraints import ConstraintFunction

# Import the new robust optimization modules
try:
    from quangtps.evaluation.robustness.robust_optimizer import (
        RobustOptimizer, optimize_robust_plan
    )
    from quangtps.evaluation.robustness.robustness_analyzer import (
        RobustnessAnalyzer, RobustnessResult, analyze_plan_robustness
    )
except ImportError:
    # Fallback to direct imports
    from quangtps.optimization.methods.robust_optimizer import (
        RobustOptimizer as OptRobustOptimizer,
        optimize_robust_plan as opt_optimize_robust_plan
    )
    
    # Create alias for compatibility
    RobustOptimizer = OptRobustOptimizer
    optimize_robust_plan = opt_optimize_robust_plan
    
    # Create stub classes if robustness_analyzer not available
    class RobustnessResult:
        """Stub class for robustness result."""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            
        def plot_dvh_band(self, structure_name, ax=None):
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "DVH Band plotting not available", 
                   ha='center', va='center', transform=ax.transAxes)
            return ax
    
    def analyze_plan_robustness(plan, structures, dose_grid, **kwargs):
        """Stub function for analyzing plan robustness."""
        return RobustnessResult(
            nominal_scenario=None, 
            scenarios=[], 
            target_coverage_range={}, 
            oar_dose_range={}
        )
    
    class RobustnessAnalyzer:
        """Stub class for robustness analyzer."""
        def __init__(self, plan, structures, dose_grid):
            self.plan = plan
            self.structures = structures
            self.dose_grid = dose_grid
            
        def analyze(self):
            return analyze_plan_robustness(self.plan, self.structures, self.dose_grid)

logger = get_logger(__name__)


class RobustOptimizationDialog(QDialog):
    """
    Dialog for configuring and executing robust optimization.
    
    This dialog provides controls for setting up robust optimization
    parameters and visualizing the results of the optimization.
    """
    
    planOptimized = pyqtSignal(Plan)
    
    def __init__(self, plan: Plan, structures: Dict[str, Structure], dose_grid: Optional[DoseGrid] = None, parent=None):
        """
        Initialize the robust optimization dialog.
        
        Args:
            plan: Treatment plan to optimize
            structures: Dictionary of available structures
            dose_grid: Dose grid (if available) for analysis
            parent: Parent widget
        """
        super().__init__(parent)
        self.plan = plan
        self.structures = structures
        self.dose_grid = dose_grid
        self.robustness_result = None
        self.target_structures = []
        self.oar_structures = []
        
        # Initialize dose calculator if we have it
        self.dose_calculator = None
        if hasattr(plan, 'dose_calculator'):
            self.dose_calculator = plan.dose_calculator
        
        # Identify targets and OARs
        for name, structure in structures.items():
            if name.lower().startswith(('ptv', 'ctv', 'target')):
                self.target_structures.append(name)
            else:
                self.oar_structures.append(name)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        # Set window properties
        self.setWindowTitle("Robust Optimization")
        self.setMinimumSize(1000, 800)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create tabbed interface
        tab_widget = QTabWidget()
        
        # Create configuration tab
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # Uncertainty parameters group
        uncertainty_group = QGroupBox("Uncertainty Parameters")
        uncertainty_layout = QGridLayout(uncertainty_group)
        
        # Setup uncertainty
        uncertainty_layout.addWidget(QLabel("Setup Uncertainty (mm):"), 0, 0)
        self.setup_uncertainty_spin = QDoubleSpinBox()
        self.setup_uncertainty_spin.setRange(0.0, 10.0)
        self.setup_uncertainty_spin.setValue(3.0)
        self.setup_uncertainty_spin.setSingleStep(0.5)
        uncertainty_layout.addWidget(self.setup_uncertainty_spin, 0, 1)
        
        # Range uncertainty
        uncertainty_layout.addWidget(QLabel("Range Uncertainty (%):"), 1, 0)
        self.range_uncertainty_spin = QDoubleSpinBox()
        self.range_uncertainty_spin.setRange(0.0, 10.0)
        self.range_uncertainty_spin.setValue(3.5)
        self.range_uncertainty_spin.setSingleStep(0.5)
        uncertainty_layout.addWidget(self.range_uncertainty_spin, 1, 1)
        
        # Specific range uncertainty controls for particle therapy
        is_particle = hasattr(self.plan, "technique") and self.plan.technique in ["PROTON", "CARBON", "PBS"]
        self.range_uncertainty_spin.setEnabled(is_particle)
        
        config_layout.addWidget(uncertainty_group)
        
        # Optimization parameters group
        optimization_group = QGroupBox("Optimization Parameters")
        optimization_layout = QGridLayout(optimization_group)
        
        # Max iterations
        optimization_layout.addWidget(QLabel("Max Iterations:"), 0, 0)
        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(10, 500)
        self.max_iterations_spin.setValue(100)
        self.max_iterations_spin.setSingleStep(10)
        optimization_layout.addWidget(self.max_iterations_spin, 0, 1)
        
        # Convergence threshold
        optimization_layout.addWidget(QLabel("Convergence Threshold:"), 1, 0)
        self.convergence_spin = QDoubleSpinBox()
        self.convergence_spin.setRange(0.0001, 0.01)
        self.convergence_spin.setValue(0.001)
        self.convergence_spin.setSingleStep(0.0001)
        self.convergence_spin.setDecimals(4)
        optimization_layout.addWidget(self.convergence_spin, 1, 1)
        
        # Learning rate
        optimization_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.01, 1.0)
        self.learning_rate_spin.setValue(0.1)
        self.learning_rate_spin.setSingleStep(0.01)
        optimization_layout.addWidget(self.learning_rate_spin, 2, 1)
        
        config_layout.addWidget(optimization_group)
        
        # Structure selection group
        structure_group = QGroupBox("Structures")
        structure_layout = QVBoxLayout(structure_group)
        
        # Targets group
        targets_group = QGroupBox("Targets")
        targets_layout = QVBoxLayout(targets_group)
        
        self.target_checkboxes = {}
        for name in self.target_structures:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            self.target_checkboxes[name] = checkbox
            targets_layout.addWidget(checkbox)
        
        targets_group.setLayout(targets_layout)
        structure_layout.addWidget(targets_group)
        
        # OARs group
        oars_group = QGroupBox("Organs at Risk")
        oars_layout = QVBoxLayout(oars_group)
        
        self.oar_checkboxes = {}
        for name in self.oar_structures:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            self.oar_checkboxes[name] = checkbox
            oars_layout.addWidget(checkbox)
        
        oars_group.setLayout(oars_layout)
        structure_layout.addWidget(oars_group)
        
        # Structure selection scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(structure_group)
        
        config_layout.addWidget(scroll_area)
        
        # Buttons for configuration tab
        config_buttons_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("Analyze Robustness")
        self.analyze_button.clicked.connect(self._analyze_robustness)
        config_buttons_layout.addWidget(self.analyze_button)
        
        self.optimize_button = QPushButton("Optimize Plan")
        self.optimize_button.clicked.connect(self._optimize_plan)
        config_buttons_layout.addWidget(self.optimize_button)
        
        config_layout.addLayout(config_buttons_layout)
        
        # Add configuration tab
        tab_widget.addTab(config_tab, "Configuration")
        
        # Create results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        # Create tabbed results view
        results_tab_widget = QTabWidget()
        
        # Create DVH tab
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)
        
        # Structure selection for DVH
        dvh_structure_layout = QHBoxLayout()
        dvh_layout.addWidget(QLabel("Select Structure:"))
        self.dvh_structure_combo = QComboBox()
        dvh_structure_layout.addWidget(self.dvh_structure_combo)
        
        dvh_layout.addLayout(dvh_structure_layout)
        
        # DVH plot
        dvh_figure = Figure(figsize=(8, 6), dpi=100)
        self.dvh_canvas = FigureCanvas(dvh_figure)
        dvh_layout.addWidget(self.dvh_canvas)
        
        results_tab_widget.addTab(dvh_tab, "DVH Bands")
        
        # Create coverage tab
        coverage_tab = QWidget()
        coverage_layout = QVBoxLayout(coverage_tab)
        
        # Coverage plot
        coverage_figure = Figure(figsize=(8, 6), dpi=100)
        self.coverage_canvas = FigureCanvas(coverage_figure)
        coverage_layout.addWidget(self.coverage_canvas)
        
        results_tab_widget.addTab(coverage_tab, "Target Coverage")
        
        # Create dose metrics tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        # Create metrics grid
        self.metrics_grid = QGridLayout()
        metrics_layout.addLayout(self.metrics_grid)
        
        results_tab_widget.addTab(metrics_tab, "Dose Metrics")
        
        # Add results tab widget to results tab
        results_layout.addWidget(results_tab_widget)
        
        # Buttons for results tab
        results_buttons_layout = QHBoxLayout()
        
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self._export_results)
        self.export_button.setEnabled(False)
        results_buttons_layout.addWidget(self.export_button)
        
        self.accept_button = QPushButton("Accept Plan")
        self.accept_button.clicked.connect(self._accept_plan)
        self.accept_button.setEnabled(False)
        results_buttons_layout.addWidget(self.accept_button)
        
        results_layout.addLayout(results_buttons_layout)
        
        # Add results tab
        tab_widget.addTab(results_tab, "Results")
        
        # Progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        # Status label
        self.status_label = QLabel("Ready")
        
        # Main dialog buttons
        button_layout = QHBoxLayout()
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        button_layout.addWidget(self.close_button)
        
        # Add components to main layout
        main_layout.addWidget(tab_widget)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(button_layout)
        
        # Setup structure combobox
        self._populate_structure_combo()
        self.dvh_structure_combo.currentTextChanged.connect(self._update_dvh_plot)
        
        # Initialize plots
        self._init_plots()
    
    def _populate_structure_combo(self):
        """Populate the structure combobox."""
        self.dvh_structure_combo.clear()
        
        # Add target structures first
        for name in self.target_structures:
            self.dvh_structure_combo.addItem(name)
        
        # Add OAR structures
        for name in self.oar_structures:
            self.dvh_structure_combo.addItem(name)
    
    def _init_plots(self):
        """Initialize the plot canvases."""
        # Initialize DVH plot
        self.dvh_ax = self.dvh_canvas.figure.add_subplot(111)
        self.dvh_ax.set_xlabel('Dose (Gy)')
        self.dvh_ax.set_ylabel('Volume (%)')
        self.dvh_ax.set_title('DVH Robustness Band')
        self.dvh_ax.grid(True)
        self.dvh_canvas.figure.tight_layout()
        self.dvh_canvas.draw()
        
        # Initialize coverage plot
        self.coverage_ax = self.coverage_canvas.figure.add_subplot(111)
        self.coverage_ax.set_title('Target Coverage Robustness')
        self.coverage_ax.grid(True)
        self.coverage_canvas.figure.tight_layout()
        self.coverage_canvas.draw()
    
    def _analyze_robustness(self):
        """Analyze the robustness of the current plan."""
        if self.dose_grid is None:
            QMessageBox.warning(
                self,
                "Missing Dose Grid",
                "Dose calculation is required before robustness analysis. Please calculate dose first."
            )
            return
        
        # Get selected structures
        selected_structures = {}
        for name, checkbox in self.target_checkboxes.items():
            if checkbox.isChecked():
                selected_structures[name] = self.structures[name]
                
        for name, checkbox in self.oar_checkboxes.items():
            if checkbox.isChecked():
                selected_structures[name] = self.structures[name]
        
        # Check if any structures selected
        if not selected_structures:
            QMessageBox.warning(
                self,
                "No Structures Selected",
                "Please select at least one structure for robustness analysis."
            )
            return
        
        # Get uncertainty parameters
        setup_uncertainty = self.setup_uncertainty_spin.value()
        range_uncertainty = self.range_uncertainty_spin.value()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analyzing robustness...")
        self.analyze_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        
        # Run analysis in a separate thread
        def analysis_thread():
            try:
                # Analyze robustness
                self.robustness_result = analyze_plan_robustness(
                    self.plan,
                    selected_structures,
                    self.dose_grid,
                    setup_uncertainty,
                    range_uncertainty
                )
                
                # Update UI in main thread
                QMetaObject.invokeMethod(
                    self,
                    "_analysis_complete",
                    Qt.QueuedConnection
                )
            except Exception as e:
                # Handle errors
                import traceback
                logger.error(f"Error during robustness analysis: {e}")
                logger.error(traceback.format_exc())
                
                QMetaObject.invokeMethod(
                    self,
                    "_analysis_error",
                    Qt.QueuedConnection,
                    Q_ARG(str, str(e))
                )
        
        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()
    
    def _analysis_complete(self):
        """Handle completion of robustness analysis."""
        self.progress_bar.setValue(100)
        self.status_label.setText("Analysis complete")
        self.analyze_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        self.export_button.setEnabled(True)
        
        # Update UI with results
        self._update_results_ui()
    
    def _analysis_error(self, error_message):
        """Handle errors during robustness analysis."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis failed")
        self.analyze_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        
        QMessageBox.critical(
            self,
            "Analysis Error",
            f"Error during robustness analysis: {error_message}"
        )
    
    def _optimize_plan(self):
        """Optimize the plan for robustness."""
        # Get selected structures
        selected_structures = {}
        for name, checkbox in self.target_checkboxes.items():
            if checkbox.isChecked():
                selected_structures[name] = self.structures[name]
                
        for name, checkbox in self.oar_checkboxes.items():
            if checkbox.isChecked():
                selected_structures[name] = self.structures[name]
        
        # Check if any structures selected
        if not selected_structures:
            QMessageBox.warning(
                self,
                "No Structures Selected",
                "Please select at least one structure for robust optimization."
            )
            return
        
        # Get uncertainty parameters
        setup_uncertainty = self.setup_uncertainty_spin.value()
        range_uncertainty = self.range_uncertainty_spin.value()
        
        # Get optimization parameters
        max_iterations = self.max_iterations_spin.value()
        convergence_threshold = self.convergence_spin.value()
        learning_rate = self.learning_rate_spin.value()
        
        # Check if we have a dose calculator
        if not self.dose_calculator:
            QMessageBox.critical(
                self,
                "Dose Calculator Required",
                "A dose calculator is required for robust optimization. Please ensure your plan has a dose calculator."
            )
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Optimizing plan...")
        self.analyze_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        
        # Create objectives from the plan
        # For simplicity, we'll use the plan's existing objectives
        # In a real implementation, you might want to let the user configure these
        objectives = PlanningObjectives()
        if hasattr(self.plan, 'objectives'):
            objectives = self.plan.objectives
        
        # Run optimization in a separate thread
        def optimization_thread():
            try:
                # Create optimizer using the new module
                optimizer = RobustOptimizer(
                    plan=self.plan,
                    objectives=objectives,
                    dose_calculator=self.dose_calculator,
                    structures=selected_structures
                )
                
                # Set parameters
                optimizer.set_parameter('max_iterations', max_iterations)
                optimizer.set_parameter('convergence_threshold', convergence_threshold)
                optimizer.set_parameter('step_size', learning_rate)
                
                # Set up progress callback
                def progress_callback(iteration):
                    QMetaObject.invokeMethod(
                        self,
                        "_update_progress",
                        Qt.QueuedConnection,
                        Q_ARG(int, iteration)
                    )
                
                optimizer.set_progress_callback(progress_callback)
                
                # Generate scenarios
                optimizer.generate_standard_scenarios(
                    setup_uncertainty=setup_uncertainty,
                    range_uncertainty=range_uncertainty
                )
                
                # Optimize
                optimized_plan, robustness_result = optimizer.optimize()
                
                # Store results
                self.optimized_plan = optimized_plan
                self.robustness_result = robustness_result
                
                # Update UI in main thread
                QMetaObject.invokeMethod(
                    self,
                    "_optimization_complete",
                    Qt.QueuedConnection
                )
            except Exception as e:
                # Handle errors
                import traceback
                logger.error(f"Error during optimization: {e}")
                logger.error(traceback.format_exc())
                
                QMetaObject.invokeMethod(
                    self,
                    "_optimization_error",
                    Qt.QueuedConnection,
                    Q_ARG(str, str(e))
                )
        
        thread = threading.Thread(target=optimization_thread)
        thread.daemon = True
        thread.start()
    
    def _update_progress(self, iteration):
        """Update progress bar during optimization."""
        max_iterations = self.max_iterations_spin.value()
        
        if iteration <= max_iterations:
            progress = int((iteration / max_iterations) * 100)
            self.progress_bar.setValue(progress)
            self.status_label.setText(f"Optimizing plan... Iteration {iteration}/{max_iterations}")
    
    def _optimization_complete(self):
        """Handle completion of robust optimization."""
        self.progress_bar.setValue(100)
        self.status_label.setText("Optimization complete")
        self.analyze_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        self.accept_button.setEnabled(True)
        
        # Update plan
        if hasattr(self, 'optimized_plan') and self.optimized_plan:
            self.plan = self.optimized_plan
            self.planOptimized.emit(self.optimized_plan)
        
        # Update results UI if we have robustness results
        if self.robustness_result:
            self._update_results_ui()
        
        # Show message
        QMessageBox.information(
            self,
            "Optimization Complete",
            "Robust optimization has completed successfully."
        )
    
    def _optimization_error(self, error_message):
        """Handle errors during robust optimization."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Optimization failed")
        self.analyze_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        
        QMessageBox.critical(
            self,
            "Optimization Error",
            f"Error during robust optimization: {error_message}"
        )
    
    def _analyze_optimized_plan(self):
        """Analyze the robustness of the optimized plan."""
        # In a real implementation, you would calculate dose for the optimized plan
        # and then analyze its robustness
        pass
    
    def _update_results_ui(self):
        """Update the results UI with current analysis data."""
        if not self.robustness_result:
            return
        
        # Update DVH plot for currently selected structure
        self._update_dvh_plot()
        
        # Update coverage plot
        self._update_coverage_plot()
        
        # Update metrics
        self._update_metrics()
    
    def _update_dvh_plot(self):
        """Update the DVH plot with data for the selected structure."""
        if not self.robustness_result:
            return
            
        structure_name = self.dvh_structure_combo.currentText()
        if not structure_name:
            return
        
        # Clear plot
        self.dvh_ax.clear()
        
        try:
            # Use the robustness result's built-in plotting method
            self.robustness_result.plot_dvh_band(structure_name, ax=self.dvh_ax)
        except Exception as e:
            logger.error(f"Error plotting DVH band: {e}")
            
        # Update canvas
        self.dvh_canvas.figure.tight_layout()
        self.dvh_canvas.draw()
    
    def _update_coverage_plot(self):
        """Update the target coverage plot."""
        if not self.robustness_result:
            return
            
        # Clear plot
        self.coverage_ax.clear()
        
        try:
            # Plot coverage range for targets
            target_names = []
            min_values = []
            max_values = []
            nominal_values = []
            
            for name, (min_val, max_val) in self.robustness_result.target_coverage_range.items():
                target_names.append(name)
                min_values.append(min_val)
                max_values.append(max_val)
                
                # Find nominal value
                if name in self.robustness_result.nominal_scenario.dvh_data:
                    dvh = self.robustness_result.nominal_scenario.dvh_data[name]
                    d95 = np.interp(95, dvh['volume_percent'][::-1], dvh['dose'][::-1])
                    nominal_values.append(d95)
                else:
                    nominal_values.append((min_val + max_val) / 2)
            
            # Setup positions
            x = np.arange(len(target_names))
            
            # Plot min-max ranges as bars
            self.coverage_ax.bar(x, np.array(max_values) - np.array(min_values), 
                                 bottom=min_values, width=0.5, alpha=0.5, 
                                 color='blue', label='Range')
            
            # Plot nominal values as points
            self.coverage_ax.scatter(x, nominal_values, color='red', 
                                     marker='o', s=50, label='Nominal')
            
            # Set labels
            self.coverage_ax.set_title('Target Coverage Robustness (D95)')
            self.coverage_ax.set_xlabel('Target')
            self.coverage_ax.set_ylabel('Dose (Gy)')
            self.coverage_ax.set_xticks(x)
            self.coverage_ax.set_xticklabels(target_names)
            self.coverage_ax.legend()
            
        except Exception as e:
            logger.error(f"Error plotting coverage: {e}")
            
        # Update canvas
        self.coverage_canvas.figure.tight_layout()
        self.coverage_canvas.draw()
    
    def _update_metrics(self):
        """Update the dose metrics grid."""
        if not self.robustness_result:
            return
            
        # Clear existing grid
        for i in reversed(range(self.metrics_grid.count())): 
            self.metrics_grid.itemAt(i).widget().setParent(None)
        
        # Add header
        self.metrics_grid.addWidget(QLabel("<b>Structure</b>"), 0, 0)
        self.metrics_grid.addWidget(QLabel("<b>Metric</b>"), 0, 1)
        self.metrics_grid.addWidget(QLabel("<b>Nominal</b>"), 0, 2)
        self.metrics_grid.addWidget(QLabel("<b>Min</b>"), 0, 3)
        self.metrics_grid.addWidget(QLabel("<b>Max</b>"), 0, 4)
        self.metrics_grid.addWidget(QLabel("<b>Range</b>"), 0, 5)
        
        # Add target coverage metrics
        row = 1
        for name, (min_val, max_val) in self.robustness_result.target_coverage_range.items():
            self.metrics_grid.addWidget(QLabel(name), row, 0)
            self.metrics_grid.addWidget(QLabel("D95"), row, 1)
            
            # Find nominal value
            nominal_value = 0
            if name in self.robustness_result.nominal_scenario.dvh_data:
                dvh = self.robustness_result.nominal_scenario.dvh_data[name]
                nominal_value = np.interp(95, dvh['volume_percent'][::-1], dvh['dose'][::-1])
            
            self.metrics_grid.addWidget(QLabel(f"{nominal_value:.2f} Gy"), row, 2)
            self.metrics_grid.addWidget(QLabel(f"{min_val:.2f} Gy"), row, 3)
            self.metrics_grid.addWidget(QLabel(f"{max_val:.2f} Gy"), row, 4)
            self.metrics_grid.addWidget(QLabel(f"{max_val - min_val:.2f} Gy"), row, 5)
            
            row += 1
        
        # Add OAR metrics
        for name, (min_val, max_val) in self.robustness_result.oar_dose_range.items():
            self.metrics_grid.addWidget(QLabel(name), row, 0)
            self.metrics_grid.addWidget(QLabel("Mean"), row, 1)
            
            # Find nominal value
            nominal_value = (min_val + max_val) / 2
            
            self.metrics_grid.addWidget(QLabel(f"{nominal_value:.2f} Gy"), row, 2)
            self.metrics_grid.addWidget(QLabel(f"{min_val:.2f} Gy"), row, 3)
            self.metrics_grid.addWidget(QLabel(f"{max_val:.2f} Gy"), row, 4)
            self.metrics_grid.addWidget(QLabel(f"{max_val - min_val:.2f} Gy"), row, 5)
            
            row += 1
    
    def _export_results(self):
        """Export robustness analysis results."""
        if not self.robustness_result:
            return
        
        # Ask for export file
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Robustness Results",
            "",
            "CSV Files (*.csv);;PDF Reports (*.pdf)"
        )
        
        if not filename:
            return
        
        if filename.endswith('.csv'):
            self._export_to_csv(filename)
        elif filename.endswith('.pdf'):
            self._export_to_pdf(filename)
    
    def _export_to_csv(self, filename):
        """Export results to CSV file."""
        try:
            with open(filename, 'w') as f:
                # Write header
                f.write("Structure,Metric,Nominal,Min,Max,Range\n")
                
                # Write target coverage metrics
                for name, (min_val, max_val) in self.robustness_result.target_coverage_range.items():
                    # Find nominal value
                    nominal_value = 0
                    if name in self.robustness_result.nominal_scenario.dvh_data:
                        dvh = self.robustness_result.nominal_scenario.dvh_data[name]
                        nominal_value = np.interp(95, dvh['volume_percent'][::-1], dvh['dose'][::-1])
                    
                    f.write(f"{name},D95,{nominal_value:.2f},{min_val:.2f},{max_val:.2f},{max_val - min_val:.2f}\n")
                
                # Write OAR metrics
                for name, (min_val, max_val) in self.robustness_result.oar_dose_range.items():
                    # Find nominal value
                    nominal_value = (min_val + max_val) / 2
                    
                    f.write(f"{name},Mean,{nominal_value:.2f},{min_val:.2f},{max_val:.2f},{max_val - min_val:.2f}\n")
            
            # Show success message
            QMessageBox.information(
                self,
                "Export Complete",
                f"Results exported to {filename}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting results: {e}"
            )
    
    def _export_to_pdf(self, filename):
        """Export results to PDF report."""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            
            with PdfPages(filename) as pdf:
                # Create DVH plots for all structures
                for name in self.target_structures + self.oar_structures:
                    if name in self.robustness_result.nominal_scenario.dvh_data:
                        # Create plot
                        fig = plt.figure(figsize=(8, 6))
                        ax = fig.add_subplot(111)
                        
                        # Plot the band
                        self.robustness_result.plot_dvh_band(name, ax=ax)
                        
                        # Save to PDF
                        pdf.savefig(fig)
                        plt.close(fig)
                
                # Create coverage plot
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                
                # Plot coverage range for targets
                target_names = []
                min_values = []
                max_values = []
                nominal_values = []
                
                for name, (min_val, max_val) in self.robustness_result.target_coverage_range.items():
                    target_names.append(name)
                    min_values.append(min_val)
                    max_values.append(max_val)
                    
                    # Find nominal value
                    if name in self.robustness_result.nominal_scenario.dvh_data:
                        dvh = self.robustness_result.nominal_scenario.dvh_data[name]
                        d95 = np.interp(95, dvh['volume_percent'][::-1], dvh['dose'][::-1])
                        nominal_values.append(d95)
                    else:
                        nominal_values.append((min_val + max_val) / 2)
                
                # Setup positions
                x = np.arange(len(target_names))
                
                # Plot min-max ranges as bars
                ax.bar(x, np.array(max_values) - np.array(min_values), 
                       bottom=min_values, width=0.5, alpha=0.5, 
                       color='blue', label='Range')
                
                # Plot nominal values as points
                ax.scatter(x, nominal_values, color='red', 
                           marker='o', s=50, label='Nominal')
                
                # Set labels
                ax.set_title('Target Coverage Robustness (D95)')
                ax.set_xlabel('Target')
                ax.set_ylabel('Dose (Gy)')
                ax.set_xticks(x)
                ax.set_xticklabels(target_names)
                ax.legend()
                
                # Save to PDF
                pdf.savefig(fig)
                plt.close(fig)
                
                # Create metadata page
                from matplotlib.backends.backend_pdf import PdfPages
                
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                # Add plan details
                text = f"Robustness Analysis Report\n\n"
                text += f"Plan: {self.plan.name}\n"
                text += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                text += f"Setup Uncertainty: {self.setup_uncertainty_spin.value()} mm\n"
                text += f"Range Uncertainty: {self.range_uncertainty_spin.value()} %\n\n"
                
                text += "Target Coverage (D95):\n"
                for name, (min_val, max_val) in self.robustness_result.target_coverage_range.items():
                    nominal_value = 0
                    if name in self.robustness_result.nominal_scenario.dvh_data:
                        dvh = self.robustness_result.nominal_scenario.dvh_data[name]
                        nominal_value = np.interp(95, dvh['volume_percent'][::-1], dvh['dose'][::-1])
                        
                    text += f"  {name}: {nominal_value:.2f} Gy (nominal), range: {min_val:.2f} - {max_val:.2f} Gy\n"
                
                text += "\nOAR Doses (Mean):\n"
                for name, (min_val, max_val) in self.robustness_result.oar_dose_range.items():
                    nominal_value = (min_val + max_val) / 2
                    text += f"  {name}: {nominal_value:.2f} Gy (nominal), range: {min_val:.2f} - {max_val:.2f} Gy\n"
                
                ax.text(0.1, 0.9, text, va='top', fontsize=10, 
                        fontfamily='monospace')
                
                # Save to PDF
                pdf.savefig(fig)
                plt.close(fig)
            
            # Show success message
            QMessageBox.information(
                self,
                "Export Complete",
                f"Report exported to {filename}"
            )
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting report: {e}"
            )
    
    def _accept_plan(self):
        """Accept the optimized plan and close dialog."""
        if hasattr(self, 'optimized_plan'):
            # Emit signal with optimized plan
            self.planOptimized.emit(self.optimized_plan)
            self.accept()
        else:
            QMessageBox.warning(
                self,
                "No Optimized Plan",
                "There is no optimized plan to accept. Please optimize the plan first."
            )


def show_robust_optimization_dialog(plan: Plan, structures: Dict[str, Structure], dose_grid: Optional[DoseGrid] = None, parent=None):
    """
    Show the robust optimization dialog.
    
    Args:
        plan: Treatment plan to optimize
        structures: Dictionary of available structures
        dose_grid: Dose grid (if available) for analysis
        parent: Parent widget
        
    Returns:
        The dialog result
    """
    dialog = RobustOptimizationDialog(plan, structures, dose_grid, parent)
    return dialog.exec_() 