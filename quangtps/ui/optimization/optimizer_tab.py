#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimizer Tab Module
===================

This module provides an Eclipse-like interface for plan optimization,
integrating the optimization objective panel with visualization and
controls for the optimization process.
"""

import logging
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, 
    QTabWidget, QGroupBox, QFrame, QSplitter, QToolButton,
    QSlider, QProgressBar, QCheckBox, QRadioButton,
    QButtonGroup, QFileDialog, QMessageBox, QMenu,
    QAction, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QColor, QBrush, QIcon, QPixmap, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QThread, QTimer

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Try to import QuangTPS modules
try:
    from quangtps.ui.optimization.optimization_objective_panel import OptimizationObjectivePanel
    from quangtps.ui.mpr_view import MPRViewer
    from quangtps.ui.widgets.dvh_widget import DVHWidget
    from quangtps.dose.dose_calculator import DoseCalculator
    from quangtps.structures.structure import Structure
    from quangtps.structures.structure_set import StructureSet
    from quangtps.planning.plan import Plan
    from quangtps.planning.beam_set import BeamSet
    from quangtps.planning.prescription import Prescription
    from quangtps.optimization.optimizer import PlanOptimizer
except ImportError:
    logging.warning("Failed to import QuangTPS optimizer modules")
    
    # Create placeholder classes if needed
    class PlanOptimizer:
        """Placeholder for PlanOptimizer if actual implementation is not available."""
        def __init__(self, *args, **kwargs):
            pass
        
        def set_objectives(self, objectives):
            pass
        
        def run_optimization(self, max_iterations=100):
            pass
        
        def stop_optimization(self):
            pass

logger = logging.getLogger(__name__)

class OptimizationCostGraph(FigureCanvas):
    """
    Canvas for displaying optimization cost function graph.
    """
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """Initialize the optimization cost graph."""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        # Initialize variables
        self.iteration_data = []
        self.cost_data = []
        self.objective_data = {}
        
        # Set up figure
        self.setup_figure()
    
    def setup_figure(self):
        """Set up the figure with appropriate styling."""
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
        
        # Set up axes
        self.axes.set_xlabel('Iteration')
        self.axes.set_ylabel('Cost')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_title('Optimization Progress')
    
    def add_cost_point(self, iteration, cost, objective_values=None):
        """Add a new data point to the cost function graph."""
        self.iteration_data.append(iteration)
        self.cost_data.append(cost)
        
        # Store individual objective costs if provided
        if objective_values:
            for obj_name, value in objective_values.items():
                if obj_name not in self.objective_data:
                    self.objective_data[obj_name] = []
                self.objective_data[obj_name].append(value)
        
        # Update plot
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with current data."""
        self.axes.clear()
        
        # Plot main cost function
        if self.iteration_data and self.cost_data:
            self.axes.plot(
                self.iteration_data, 
                self.cost_data, 
                'b-', 
                linewidth=2, 
                label='Total Cost'
            )
            
            # Plot individual objectives
            for obj_name, values in self.objective_data.items():
                if len(values) == len(self.iteration_data):
                    self.axes.plot(
                        self.iteration_data,
                        values,
                        '--',
                        alpha=0.7,
                        linewidth=1,
                        label=obj_name
                    )
        
        # Set up axes
        self.axes.set_xlabel('Iteration')
        self.axes.set_ylabel('Cost')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend if we have individual objectives
        if self.objective_data:
            self.axes.legend(
                loc='upper right',
                fontsize='small',
                framealpha=0.7
            )
        
        # Set y axis to log scale if range is large
        if self.cost_data and max(self.cost_data) / (min(self.cost_data) + 1e-10) > 100:
            self.axes.set_yscale('log')
        else:
            self.axes.set_yscale('linear')
        
        # Draw
        self.fig.canvas.draw()
    
    def clear(self):
        """Clear all data and reset the plot."""
        self.iteration_data = []
        self.cost_data = []
        self.objective_data = {}
        
        self.axes.clear()
        self.setup_figure()
        self.fig.canvas.draw()

class OptimizationStatus(QWidget):
    """
    Widget for displaying optimization status and controls.
    """
    
    # Signals
    startOptimizationRequested = pyqtSignal()
    stopOptimizationRequested = pyqtSignal()
    resumeOptimizationRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the optimization status widget."""
        super().__init__(parent)
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create progress group
        progress_group = QGroupBox("Optimization Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Create status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        progress_layout.addWidget(self.status_label)
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        # Create metrics grid
        metrics_layout = QHBoxLayout()
        
        # Add metrics
        self.iteration_label = QLabel("Iteration: 0")
        metrics_layout.addWidget(self.iteration_label)
        
        self.cost_label = QLabel("Cost: 0.0")
        metrics_layout.addWidget(self.cost_label)
        
        self.time_label = QLabel("Time: 0s")
        metrics_layout.addWidget(self.time_label)
        
        progress_layout.addLayout(metrics_layout)
        
        # Create parameters group
        params_group = QGroupBox("Optimization Parameters")
        params_layout = QFormLayout(params_group)
        
        # Create max iterations control
        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(1, 1000)
        self.max_iterations_spin.setValue(100)
        params_layout.addRow("Max Iterations:", self.max_iterations_spin)
        
        # Create convergence tolerance control
        self.convergence_spin = QDoubleSpinBox()
        self.convergence_spin.setRange(1e-6, 1e-2)
        self.convergence_spin.setValue(1e-4)
        self.convergence_spin.setDecimals(6)
        self.convergence_spin.setSingleStep(1e-5)
        params_layout.addRow("Convergence Tolerance:", self.convergence_spin)
        
        # Create optimization mode controls
        self.optimization_mode_group = QButtonGroup(self)
        
        mode_layout = QHBoxLayout()
        
        self.normal_mode_radio = QRadioButton("Normal")
        self.normal_mode_radio.setChecked(True)
        self.optimization_mode_group.addButton(self.normal_mode_radio)
        mode_layout.addWidget(self.normal_mode_radio)
        
        self.fast_mode_radio = QRadioButton("Fast")
        self.optimization_mode_group.addButton(self.fast_mode_radio)
        mode_layout.addWidget(self.fast_mode_radio)
        
        self.accurate_mode_radio = QRadioButton("Accurate")
        self.optimization_mode_group.addButton(self.accurate_mode_radio)
        mode_layout.addWidget(self.accurate_mode_radio)
        
        params_layout.addRow("Mode:", mode_layout)
        
        # Create controls layout
        controls_layout = QHBoxLayout()
        
        # Create start button
        self.start_button = QPushButton("Start Optimization")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_button.clicked.connect(self._on_start_clicked)
        controls_layout.addWidget(self.start_button)
        
        # Create stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        controls_layout.addWidget(self.stop_button)
        
        # Create pause button
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._on_pause_clicked)
        controls_layout.addWidget(self.pause_button)
        
        # Add widgets to main layout
        main_layout.addWidget(progress_group)
        main_layout.addWidget(params_group)
        main_layout.addSpacing(10)
        main_layout.addLayout(controls_layout)
        main_layout.addStretch(1)
        
        # Set initial state
        self._update_ui_state(is_running=False, is_paused=False)
    
    def set_progress(self, iteration, max_iterations, cost, elapsed_time):
        """Update the progress display with current optimization status."""
        # Update progress bar
        progress = (iteration / max_iterations) * 100
        self.progress_bar.setValue(int(progress))
        
        # Update labels
        self.iteration_label.setText(f"Iteration: {iteration}")
        self.cost_label.setText(f"Cost: {cost:.3f}")
        self.time_label.setText(f"Time: {elapsed_time:.1f}s")
        
        # Update status label
        self.status_label.setText("Optimizing...")
    
    def set_status(self, status):
        """Set the status text."""
        self.status_label.setText(status)
    
    def _on_start_clicked(self):
        """Handle start button clicked."""
        # Get optimization parameters
        max_iterations = self.max_iterations_spin.value()
        convergence = self.convergence_spin.value()
        
        # Get optimization mode
        if self.fast_mode_radio.isChecked():
            mode = "fast"
        elif self.accurate_mode_radio.isChecked():
            mode = "accurate"
        else:
            mode = "normal"
        
        # Update UI state
        self._update_ui_state(is_running=True, is_paused=False)
        
        # Reset progress display
        self.progress_bar.setValue(0)
        self.iteration_label.setText("Iteration: 0")
        self.cost_label.setText("Cost: 0.0")
        self.time_label.setText("Time: 0s")
        
        # Emit signal
        self.startOptimizationRequested.emit()
    
    def _on_stop_clicked(self):
        """Handle stop button clicked."""
        # Update UI state
        self._update_ui_state(is_running=False, is_paused=False)
        
        # Emit signal
        self.stopOptimizationRequested.emit()
    
    def _on_pause_clicked(self):
        """Handle pause/resume button clicked."""
        # Check if paused or running
        is_paused = self.pause_button.text() == "Resume"
        
        # Update UI state
        self._update_ui_state(is_running=True, is_paused=not is_paused)
        
        # Emit signal
        if is_paused:
            self.resumeOptimizationRequested.emit()
        else:
            self.stopOptimizationRequested.emit()
    
    def _update_ui_state(self, is_running, is_paused):
        """Update UI control states based on optimization state."""
        # Update button states
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)
        self.pause_button.setEnabled(is_running)
        
        # Update pause button text
        self.pause_button.setText("Resume" if is_paused else "Pause")
        
        # Update parameter controls
        self.max_iterations_spin.setEnabled(not is_running)
        self.convergence_spin.setEnabled(not is_running)
        self.normal_mode_radio.setEnabled(not is_running)
        self.fast_mode_radio.setEnabled(not is_running)
        self.accurate_mode_radio.setEnabled(not is_running)
        
        # Update status label
        if not is_running:
            self.status_label.setText("Ready")
        elif is_paused:
            self.status_label.setText("Paused")
        else:
            self.status_label.setText("Optimizing...")

class OptimizerTab(QWidget):
    """
    Tab for IMRT/VMAT plan optimization.
    
    This tab provides an Eclipse-like interface for configuring optimization
    objectives, running the optimization process, and visualizing the results.
    """
    
    # Signals
    optimizationCompleted = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the optimizer tab."""
        super().__init__(parent)
        
        # Initialize variables
        self.plan = None
        self.dose_calculator = None
        self.optimizer = None
        self.optimization_thread = None
        self.optimization_timer = QTimer()
        self.optimization_timer.timeout.connect(self._update_optimization_status)
        self.optimization_start_time = 0
        self.current_iteration = 0
        self.current_cost = 0
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create objectives panel
        self.objectives_panel = OptimizationObjectivePanel()
        self.objectives_panel.objectivesChanged.connect(self._on_objectives_changed)
        self.objectives_panel.startOptimizationRequested.connect(self._on_start_optimization)
        left_layout.addWidget(self.objectives_panel)
        
        # Create right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget for visualization
        self.viz_tabs = QTabWidget()
        
        # Create cost function tab
        cost_tab = QWidget()
        cost_layout = QVBoxLayout(cost_tab)
        
        # Create cost function graph
        self.cost_graph = OptimizationCostGraph()
        cost_layout.addWidget(self.cost_graph)
        
        self.viz_tabs.addTab(cost_tab, "Cost Function")
        
        # Create DVH tab
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)
        
        # Create DVH widget
        self.dvh_widget = DVHWidget() if 'DVHWidget' in locals() else QLabel("DVH Widget Placeholder")
        dvh_layout.addWidget(self.dvh_widget)
        
        self.viz_tabs.addTab(dvh_tab, "DVH")
        
        # Create dose display tab
        dose_tab = QWidget()
        dose_layout = QVBoxLayout(dose_tab)
        
        # Create MPR viewer
        self.mpr_viewer = MPRViewer() if 'MPRViewer' in locals() else QLabel("MPR Viewer Placeholder")
        dose_layout.addWidget(self.mpr_viewer)
        
        self.viz_tabs.addTab(dose_tab, "Dose Display")
        
        # Create fluence tab
        fluence_tab = QWidget()
        fluence_layout = QVBoxLayout(fluence_tab)
        
        # Create fluence display (placeholder)
        self.fluence_display = QLabel("Fluence Maps (Not Implemented)")
        self.fluence_display.setAlignment(Qt.AlignCenter)
        fluence_layout.addWidget(self.fluence_display)
        
        self.viz_tabs.addTab(fluence_tab, "Fluence Maps")
        
        # Add tab widget to right panel
        right_layout.addWidget(self.viz_tabs)
        
        # Create optimization status widget
        self.optimization_status = OptimizationStatus()
        self.optimization_status.startOptimizationRequested.connect(self._on_start_optimization)
        self.optimization_status.stopOptimizationRequested.connect(self._on_stop_optimization)
        self.optimization_status.resumeOptimizationRequested.connect(self._on_resume_optimization)
        right_layout.addWidget(self.optimization_status)
        
        # Add panels to main splitter
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([400, 600])
        
        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter)
        
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
                background-color: #f5f5f5;
            }
            
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 5px 10px;
                margin-right: 2px;
            }
            
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            
            QTabBar::tab:hover {
                background-color: #e0e0e0;
            }
        """)
    
    def set_plan(self, plan):
        """Set the treatment plan for optimization."""
        self.plan = plan
        
        # Update UI with plan data
        self._update_from_plan()
    
    def set_dose_calculator(self, calculator):
        """Set the dose calculator for optimization."""
        self.dose_calculator = calculator
        
        # Create optimizer with dose calculator
        self._create_optimizer()
    
    def _update_from_plan(self):
        """Update UI components with plan data."""
        if not self.plan:
            return
        
        # Set structure set in objectives panel
        if hasattr(self.plan, 'structure_set'):
            self.objectives_panel.set_structure_set(self.plan.structure_set)
        
        # Set prescription in objectives panel
        if hasattr(self.plan, 'prescription'):
            self.objectives_panel.set_prescription(self.plan.prescription)
        
        # Update optimization parameters from plan
        # (This would be plan-specific settings)
        
        # Create optimizer with plan
        self._create_optimizer()
    
    def _create_optimizer(self):
        """Create the optimizer with current plan and dose calculator."""
        # Only create if we have both plan and dose calculator
        if not self.plan or not self.dose_calculator:
            return
        
        # Try to create optimizer
        try:
            self.optimizer = PlanOptimizer(
                plan=self.plan,
                dose_calculator=self.dose_calculator
            )
            
            # Set objectives
            self._on_objectives_changed()
            
            # Enable optimization UI
            self._update_ui_state(enable=True)
            
        except Exception as e:
            logger.error(f"Failed to create optimizer: {e}")
            self.optimizer = None
            
            # Disable optimization UI
            self._update_ui_state(enable=False)
    
    def _update_ui_state(self, enable=True):
        """Update UI enabled states based on optimizer availability."""
        # Enable/disable objectives panel
        self.objectives_panel.setEnabled(enable)
        
        # Enable/disable optimization status
        self.optimization_status.setEnabled(enable)
        
        if not enable:
            # Clear visualization
            self.cost_graph.clear()
    
    def _on_objectives_changed(self):
        """Handle changes to optimization objectives."""
        if not self.optimizer:
            return
        
        # Get objectives from panel
        objectives = self.objectives_panel.get_objectives()
        
        # Set objectives in optimizer
        try:
            self.optimizer.set_objectives(objectives)
        except Exception as e:
            logger.error(f"Failed to set objectives: {e}")
    
    def _on_start_optimization(self):
        """Start the optimization process."""
        if not self.optimizer:
            return
        
        # Get optimization parameters
        max_iterations = self.optimization_status.max_iterations_spin.value()
        convergence_tolerance = self.optimization_status.convergence_spin.value()
        
        # Get optimization mode
        if self.optimization_status.fast_mode_radio.isChecked():
            mode = "fast"
        elif self.optimization_status.accurate_mode_radio.isChecked():
            mode = "accurate"
        else:
            mode = "normal"
        
        # Reset cost graph
        self.cost_graph.clear()
        
        # Reset status
        self.current_iteration = 0
        self.current_cost = 0
        
        # Run optimization in a background thread
        self._run_optimization_in_thread(
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            mode=mode
        )
    
    def _on_stop_optimization(self):
        """Stop the optimization process."""
        if not self.optimization_thread:
            return
        
        # Stop the optimization
        try:
            self.optimizer.stop_optimization()
        except Exception as e:
            logger.error(f"Failed to stop optimization: {e}")
        
        # Stop the timer
        self.optimization_timer.stop()
        
        # Update status
        self.optimization_status.set_status("Stopped")
    
    def _on_resume_optimization(self):
        """Resume the optimization process."""
        # Not implemented - would require storing state
        # For now, just start a new optimization
        self._on_start_optimization()
    
    def _run_optimization_in_thread(self, max_iterations, convergence_tolerance, mode):
        """Run the optimization process in a background thread."""
        # Create a QThread to run the optimization
        self.optimization_thread = QThread()
        
        # Set up a timer to update the UI with progress
        self.optimization_start_time = 0
        self.optimization_timer.start(100)  # Update every 100ms
        
        # In a real implementation, we would move the optimizer to the thread
        # and connect its signals/slots - for now, just simulate the process
        
        # Simulate optimization with a timer
        # In a real implementation, we would run the optimizer in a thread
        self._simulate_optimization(max_iterations)
    
    def _simulate_optimization(self, max_iterations):
        """Simulate the optimization process for demonstration purposes."""
        # Create a timer to simulate iterations
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(
            lambda: self._simulate_iteration(max_iterations)
        )
        self.sim_timer.start(200)  # Simulate an iteration every 200ms
    
    def _simulate_iteration(self, max_iterations):
        """Simulate an optimization iteration."""
        # Increment iteration
        self.current_iteration += 1
        
        # Simulate cost function (decreasing exponentially with noise)
        base_cost = 100 * np.exp(-0.05 * self.current_iteration)
        noise = 5 * np.random.normal() * np.exp(-0.01 * self.current_iteration)
        self.current_cost = base_cost + noise
        
        # Simulate objective values
        objective_values = {
            "PTV": 70 * np.exp(-0.05 * self.current_iteration) + np.random.normal() * 2,
            "OAR1": 20 * np.exp(-0.07 * self.current_iteration) + np.random.normal() * 1,
            "OAR2": 10 * np.exp(-0.03 * self.current_iteration) + np.random.normal() * 0.5
        }
        
        # Add data point to cost graph
        self.cost_graph.add_cost_point(
            iteration=self.current_iteration,
            cost=self.current_cost,
            objective_values=objective_values
        )
        
        # Stop if reached max iterations
        if self.current_iteration >= max_iterations:
            self.sim_timer.stop()
            self.optimization_timer.stop()
            
            # Update status
            self.optimization_status.set_status("Completed")
            
            # Update UI state
            self.optimization_status._update_ui_state(
                is_running=False,
                is_paused=False
            )
            
            # Emit completed signal
            self.optimizationCompleted.emit()
    
    def _update_optimization_status(self):
        """Update the optimization status display."""
        # Calculate elapsed time
        if self.optimization_start_time == 0:
            self.optimization_start_time = np.datetime64('now')
            elapsed_time = 0
        else:
            current_time = np.datetime64('now')
            elapsed_time = (current_time - self.optimization_start_time) / np.timedelta64(1, 's')
        
        # Update status display
        self.optimization_status.set_progress(
            iteration=self.current_iteration,
            max_iterations=self.optimization_status.max_iterations_spin.value(),
            cost=self.current_cost,
            elapsed_time=elapsed_time
        )

def test_optimizer_tab():
    """Test function for the optimizer tab."""
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("QuangTPS Optimizer")
    window.resize(1200, 800)
    
    # Create test structures and structure set
    class TestStructure:
        def __init__(self, name, id=None):
            self.name = name
            self.id = id or name
    
    class TestStructureSet:
        def __init__(self):
            self.structures = [
                TestStructure("PTV"),
                TestStructure("OAR1_Parotid_L"),
                TestStructure("OAR2_Parotid_R"),
                TestStructure("OAR3_SpinalCord"),
                TestStructure("OAR4_Brainstem"),
                TestStructure("BODY")
            ]
    
    # Create test prescription
    class TestPrescription:
        def __init__(self):
            self.targets = []
    
    class TestTarget:
        def __init__(self, name, dose):
            self.name = name
            self.dose_level = TestDoseLevel(dose)
    
    class TestDoseLevel:
        def __init__(self, dose):
            self.dose = dose
    
    # Create test plan
    class TestPlan:
        def __init__(self):
            self.structure_set = TestStructureSet()
            self.prescription = TestPrescription()
            self.prescription.targets.append(TestTarget("PTV", 70.0))
    
    # Create test dose calculator
    class TestDoseCalculator:
        def __init__(self):
            pass
        
        def calculate_dose(self, *args, **kwargs):
            return np.zeros((10, 10, 10))
    
    # Create optimizer tab
    optimizer_tab = OptimizerTab()
    optimizer_tab.set_plan(TestPlan())
    optimizer_tab.set_dose_calculator(TestDoseCalculator())
    
    # Set as central widget
    window.setCentralWidget(optimizer_tab)
    window.show()
    
    return app.exec_()

if __name__ == "__main__":
    test_optimizer_tab() 