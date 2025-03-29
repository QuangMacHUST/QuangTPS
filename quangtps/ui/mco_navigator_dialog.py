#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-criteria optimization (MCO) navigator dialog.

This module provides a dialog for navigating the trade-off space of
Pareto-optimal treatment plans. It allows users to interactively
explore different treatment plans by adjusting weights for different
clinical objectives.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPainter, QPen, QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QCheckBox, QGroupBox, 
    QPushButton, QComboBox, QTabWidget, QWidget,
    QSpinBox, QDoubleSpinBox, QLineEdit, QFrame,
    QScrollArea, QSplitter, QFileDialog, QMessageBox
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from quangtps.optimization.mco.mco_engine import MCOEngine, ParetoSolution
from quangtps.core.types import Plan, DoseGrid
from quangtps.planning.evaluation import PlanEvaluator
from quangtps.ui.dose_tab import DVHWidget
from quangtps.evaluation.dvh.dvh_visualization import DVHPlot

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class ObjectiveSlider(QWidget):
    """
    Custom slider widget for adjusting objective weights.
    
    This widget includes a slider, label, and value display.
    """
    
    valueChanged = pyqtSignal(str, float)
    
    def __init__(self, objective_name: str, label: str, initial_value: float = 0.0, parent=None):
        """Initialize the objective slider."""
        super().__init__(parent)
        self.objective_name = objective_name
        self.label_text = label
        self.initial_value = initial_value
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI for the slider."""
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Create label
        self.label = QLabel(self.label_text)
        self.label.setMinimumWidth(150)
        layout.addWidget(self.label)
        
        # Create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)  # 0.0 to 1.0 with 0.01 steps
        self.slider.setValue(int(self.initial_value * 100))
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 1)  # 1 is the stretch factor
        
        # Create value display
        self.value_label = QLabel(f"{self.initial_value:.2f}")
        self.value_label.setMinimumWidth(40)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.value_label)
        
        self.setLayout(layout)
    
    def _on_slider_changed(self, value: int):
        """Handle slider value changes."""
        # Convert slider value (0-100) to objective weight (0.0-1.0)
        weight = value / 100.0
        self.value_label.setText(f"{weight:.2f}")
        self.valueChanged.emit(self.objective_name, weight)
    
    def get_value(self) -> float:
        """Get the current slider value as a weight (0.0-1.0)."""
        return self.slider.value() / 100.0
    
    def set_value(self, value: float):
        """Set the slider value from a weight (0.0-1.0)."""
        self.slider.setValue(int(value * 100))


class MCONavigatorDialog(QDialog):
    """
    Dialog for navigating the trade-off space of Pareto-optimal plans.
    
    This dialog allows users to interactively explore different treatment
    plans by adjusting weights for different clinical objectives.
    """
    
    planAccepted = pyqtSignal(Plan)
    
    def __init__(self, mco_engine: MCOEngine, parent=None):
        """Initialize the MCO navigator dialog."""
        super().__init__(parent)
        self.mco_engine = mco_engine
        self.plan = mco_engine.plan
        self.current_solution = None
        self.sliders = {}
        self.updating_sliders = False
        self.dvh_plot = None
        self.pareto_plot = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._delayed_update)
        
        self._setup_ui()
        self._initialize_pareto_surface()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        # Set window properties
        self.setWindowTitle("Multi-Criteria Optimization Navigator")
        self.setMinimumSize(1200, 800)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create splitter for top and bottom sections
        splitter = QSplitter(Qt.Vertical)
        
        # Create top widget (objectives and visualization)
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        # Create left panel (objectives)
        left_panel = QGroupBox("Objectives")
        left_layout = QVBoxLayout(left_panel)
        
        # Create scroll area for sliders
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        slider_widget = QWidget()
        self.sliders_layout = QVBoxLayout(slider_widget)
        
        # Add objective sliders
        self._create_objective_sliders()
        
        # Complete left panel
        scroll_area.setWidget(slider_widget)
        left_layout.addWidget(scroll_area)
        
        # Add buttons for presets
        presets_layout = QHBoxLayout()
        
        reset_button = QPushButton("Reset Weights")
        reset_button.clicked.connect(self._reset_weights)
        presets_layout.addWidget(reset_button)
        
        balanced_button = QPushButton("Balanced Plan")
        balanced_button.clicked.connect(self._set_balanced_weights)
        presets_layout.addWidget(balanced_button)
        
        left_layout.addLayout(presets_layout)
        
        # Create right panel (visualization)
        right_panel = QTabWidget()
        
        # Create DVH tab
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)
        self.dvh_widget = DVHWidget()
        dvh_layout.addWidget(self.dvh_widget)
        right_panel.addTab(dvh_tab, "DVH")
        
        # Create Pareto front tab
        pareto_tab = QWidget()
        pareto_layout = QVBoxLayout(pareto_tab)
        
        # matplotlib figure for Pareto front
        fig = Figure(figsize=(8, 6), dpi=100)
        self.pareto_canvas = FigureCanvas(fig)
        pareto_layout.addWidget(self.pareto_canvas)
        
        # Controls for Pareto plot
        pareto_controls = QHBoxLayout()
        
        self.x_objective_combo = QComboBox()
        pareto_controls.addWidget(QLabel("X-Axis:"))
        pareto_controls.addWidget(self.x_objective_combo)
        
        self.y_objective_combo = QComboBox()
        pareto_controls.addWidget(QLabel("Y-Axis:"))
        pareto_controls.addWidget(self.y_objective_combo)
        
        update_plot_button = QPushButton("Update Plot")
        update_plot_button.clicked.connect(self._update_pareto_plot)
        pareto_controls.addWidget(update_plot_button)
        
        pareto_layout.addLayout(pareto_controls)
        right_panel.addTab(pareto_tab, "Pareto Front")
        
        # Create Dose tab
        dose_tab = QWidget()
        dose_layout = QVBoxLayout(dose_tab)
        # Placeholder for dose display (should be replaced with actual dose viewer)
        dose_label = QLabel("Dose visualization will be displayed here.")
        dose_label.setAlignment(Qt.AlignCenter)
        dose_layout.addWidget(dose_label)
        right_panel.addTab(dose_tab, "Dose")
        
        # Create Statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_text = QLabel("Generate plans to see statistics.")
        self.stats_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        stats_layout.addWidget(self.stats_text)
        right_panel.addTab(stats_tab, "Statistics")
        
        # Add panels to top layout
        top_layout.addWidget(left_panel, 1)  # 1 is the stretch factor
        top_layout.addWidget(right_panel, 3)  # 3 is the stretch factor
        
        # Create bottom widget (controls)
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        
        # Create generation controls
        generation_group = QGroupBox("Pareto Surface Generation")
        generation_layout = QGridLayout()
        
        generation_layout.addWidget(QLabel("Number of Plans:"), 0, 0)
        self.num_plans_spin = QSpinBox()
        self.num_plans_spin.setMinimum(5)
        self.num_plans_spin.setMaximum(100)
        self.num_plans_spin.setValue(10)
        generation_layout.addWidget(self.num_plans_spin, 0, 1)
        
        generation_layout.addWidget(QLabel("Method:"), 1, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Weight Sampling", "Constraint Sampling", "Normal Constraint"])
        generation_layout.addWidget(self.method_combo, 1, 1)
        
        generate_button = QPushButton("Generate Pareto Surface")
        generate_button.clicked.connect(self._generate_pareto_surface)
        generation_layout.addWidget(generate_button, 2, 0, 1, 2)
        
        generation_group.setLayout(generation_layout)
        bottom_layout.addWidget(generation_group)
        
        # Create navigation controls
        navigation_group = QGroupBox("Navigation Controls")
        navigation_layout = QVBoxLayout()
        
        nav_buttons_layout = QHBoxLayout()
        
        update_button = QPushButton("Update Plan")
        update_button.clicked.connect(self._update_plan)
        nav_buttons_layout.addWidget(update_button)
        
        interpolate_check = QCheckBox("Auto-Interpolate")
        interpolate_check.setChecked(True)
        interpolate_check.stateChanged.connect(self._on_auto_interpolate_changed)
        nav_buttons_layout.addWidget(interpolate_check)
        
        navigation_layout.addLayout(nav_buttons_layout)
        
        # Show status
        self.status_label = QLabel("Ready.")
        navigation_layout.addWidget(self.status_label)
        
        navigation_group.setLayout(navigation_layout)
        bottom_layout.addWidget(navigation_group)
        
        # Create action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        
        accept_button = QPushButton("Accept Plan")
        accept_button.clicked.connect(self._accept_plan)
        action_layout.addWidget(accept_button)
        
        save_button = QPushButton("Save Solutions")
        save_button.clicked.connect(self._save_solutions)
        action_layout.addWidget(save_button)
        
        action_group.setLayout(action_layout)
        bottom_layout.addWidget(action_group)
        
        # Add widgets to splitter
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        splitter.setSizes([600, 200])  # Initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        main_layout.addWidget(close_button)
        
        self.setLayout(main_layout)
        
        # Fill objective comboboxes
        self._update_objective_combos()
    
    def _create_objective_sliders(self):
        """Create sliders for each objective."""
        # Clear existing sliders
        while self.sliders_layout.count():
            item = self.sliders_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.sliders = {}
        
        # Get objective names and add sliders
        for name, objective in self.mco_engine.objectives.items():
            # Create a friendly label from the objective name
            label = name.replace("_", " ").title()
            
            # Add the objective type to the label
            objective_type = objective.objective_type.name.replace("_", " ").title()
            label = f"{label} ({objective_type})"
            
            # Create the slider
            slider = ObjectiveSlider(name, label)
            slider.valueChanged.connect(self._on_slider_changed)
            self.sliders_layout.addWidget(slider)
            self.sliders[name] = slider
        
        # Add stretch to push sliders to the top
        self.sliders_layout.addStretch()
    
    def _update_objective_combos(self):
        """Update the objective comboboxes for the Pareto plot."""
        self.x_objective_combo.clear()
        self.y_objective_combo.clear()
        
        objectives = list(self.mco_engine.objectives.keys())
        
        for name in objectives:
            label = name.replace("_", " ").title()
            self.x_objective_combo.addItem(label, name)
            self.y_objective_combo.addItem(label, name)
        
        # Set defaults if we have at least 2 objectives
        if len(objectives) >= 2:
            self.x_objective_combo.setCurrentIndex(0)
            self.y_objective_combo.setCurrentIndex(1)
    
    def _initialize_pareto_surface(self):
        """Initialize the Pareto surface by generating anchor plans."""
        self.status_label.setText("Generating anchor plans...")
        QTimer.singleShot(100, self._generate_anchor_plans)
    
    def _generate_anchor_plans(self):
        """Generate anchor plans for each objective."""
        try:
            # Prepare the MCO engine
            if not self.mco_engine.prepare():
                self.status_label.setText("Failed to prepare MCO engine.")
                QMessageBox.warning(self, "Error", "Failed to prepare MCO engine.")
                return
            
            # Generate anchor plans
            self.mco_engine.generate_anchor_plans()
            
            # Generate a balanced plan
            self.mco_engine.generate_balanced_plan()
            
            # Update status
            self.status_label.setText(f"Generated {len(self.mco_engine.solutions)} plans.")
            
            # Set the balanced plan as current
            if self.mco_engine.solutions:
                # Find the balanced plan (the one with most equal weights)
                balanced_index = self._find_balanced_plan()
                if balanced_index >= 0:
                    self.current_solution = self.mco_engine.solutions[balanced_index]
                    self._update_slider_values_from_solution(self.current_solution)
                    self._update_displays()
        except Exception as e:
            logger.error(f"Error generating anchor plans: {e}", exc_info=True)
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to generate plans: {str(e)}")
    
    def _find_balanced_plan(self) -> int:
        """Find the index of the most balanced plan."""
        if not self.mco_engine.solutions:
            return -1
        
        # Find the plan with the most balanced weights
        min_std = float('inf')
        min_idx = -1
        
        for i, solution in enumerate(self.mco_engine.solutions):
            weights = list(solution.weights.values())
            if not weights:
                continue
            
            # Calculate standard deviation of weights
            std = np.std(weights)
            if std < min_std:
                min_std = std
                min_idx = i
        
        return min_idx
    
    def _generate_pareto_surface(self):
        """Generate additional Pareto-optimal plans."""
        try:
            # Get parameters
            num_plans = self.num_plans_spin.value()
            method_idx = self.method_combo.currentIndex()
            method = ['weight_sampling', 'constraint_sampling', 'normal_constraint'][method_idx]
            
            # Update status
            self.status_label.setText(f"Generating {num_plans} Pareto-optimal plans...")
            
            # Generate plans
            self.mco_engine.generate_pareto_surface(num_plans, method)
            
            # Update status
            self.status_label.setText(f"Generated {len(self.mco_engine.solutions)} plans total.")
            
            # Update displays
            self._update_pareto_plot()
        except Exception as e:
            logger.error(f"Error generating Pareto surface: {e}", exc_info=True)
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to generate Pareto surface: {str(e)}")
    
    def _on_slider_changed(self, objective_name: str, weight: float):
        """Handle slider value changes."""
        if self.updating_sliders:
            return
        
        # If auto-interpolate is checked, update after a short delay
        interpolate_check = self.findChild(QCheckBox, "Auto-Interpolate")
        if interpolate_check and interpolate_check.isChecked():
            # Reset timer to prevent multiple rapid updates
            self.timer.stop()
            self.timer.start(300)  # 300 ms delay
    
    def _delayed_update(self):
        """Update the plan after a delay (for auto-interpolate)."""
        self.timer.stop()
        self._update_plan()
    
    def _on_auto_interpolate_changed(self, state: int):
        """Handle auto-interpolate checkbox changes."""
        # Nothing to do here - the checking happens in _on_slider_changed
        pass
    
    def _update_plan(self):
        """Update the plan based on current slider values."""
        try:
            # Get weights from sliders
            weights = {name: slider.get_value() for name, slider in self.sliders.items()}
            
            # Check if we have any non-zero weights
            if sum(weights.values()) == 0:
                self.status_label.setText("Error: All weights are zero.")
                return
            
            # Update status
            self.status_label.setText("Updating plan...")
            
            # Navigate to the new solution
            solution = self.mco_engine.navigate(weights)
            
            if solution:
                self.current_solution = solution
                # Update displays
                self._update_displays()
                
                # Update status
                if 'interpolated' in solution.metadata and solution.metadata['interpolated']:
                    self.status_label.setText("Updated plan (interpolated).")
                else:
                    self.status_label.setText("Updated plan (re-optimized).")
            else:
                self.status_label.setText("Failed to update plan.")
        except Exception as e:
            logger.error(f"Error updating plan: {e}", exc_info=True)
            self.status_label.setText(f"Error: {str(e)}")
    
    def _update_slider_values_from_solution(self, solution: ParetoSolution):
        """Update slider values from a solution."""
        # Set flag to prevent recursive updates
        self.updating_sliders = True
        
        try:
            # Get total sum of weights for normalization
            total_weight = sum(solution.weights.values())
            if total_weight == 0:
                return
            
            # Update sliders
            for name, slider in self.sliders.items():
                weight = solution.weights.get(name, 0.0) / total_weight
                slider.set_value(weight)
        finally:
            # Clear flag
            self.updating_sliders = False
    
    def _update_displays(self):
        """Update all displays with the current solution."""
        if not self.current_solution:
            return
        
        try:
            # Update DVH
            if self.current_solution.dose_grid:
                structures = self.plan.structure_set.get_structure_names()
                self.dvh_widget.update_dvh(self.current_solution.dose_grid, structures)
            
            # Update statistics
            self._update_statistics()
            
            # Update Pareto plot (if open)
            if self.pareto_canvas.isVisible():
                self._update_pareto_plot()
        except Exception as e:
            logger.error(f"Error updating displays: {e}", exc_info=True)
    
    def _update_statistics(self):
        """Update the statistics display."""
        if not self.current_solution:
            return
        
        try:
            # Create a string with all stats
            stats = []
            
            # Add plan info
            stats.append("<b>Plan Information:</b>")
            stats.append(f"Plan: {self.plan.name}" if hasattr(self.plan, 'name') else "Unnamed Plan")
            
            # Add objective values
            stats.append("<br><b>Objective Values:</b>")
            for name, value in self.current_solution.objective_values.items():
                name_formatted = name.replace("_", " ").title()
                stats.append(f"{name_formatted}: {value:.4f}")
            
            # Add metadata
            stats.append("<br><b>Solution Information:</b>")
            if 'optimization_time' in self.current_solution.metadata:
                stats.append(f"Optimization Time: {self.current_solution.metadata['optimization_time']:.2f} seconds")
            if 'iterations' in self.current_solution.metadata:
                stats.append(f"Iterations: {self.current_solution.metadata['iterations']}")
            if 'interpolated' in self.current_solution.metadata and self.current_solution.metadata['interpolated']:
                stats.append("Solution Type: Interpolated")
            else:
                stats.append("Solution Type: Optimized")
            
            # Update the stats text
            self.stats_text.setText("<p>" + "<br>".join(stats) + "</p>")
        except Exception as e:
            logger.error(f"Error updating statistics: {e}", exc_info=True)
    
    def _update_pareto_plot(self):
        """Update the Pareto front plot."""
        if not self.mco_engine.solutions:
            return
        
        try:
            # Get selected objectives
            x_idx = self.x_objective_combo.currentIndex()
            y_idx = self.y_objective_combo.currentIndex()
            
            if x_idx < 0 or y_idx < 0:
                return
            
            x_name = self.x_objective_combo.itemData(x_idx)
            y_name = self.y_objective_combo.itemData(y_idx)
            
            # Generate plot
            plot = self.mco_engine.plot_pareto_front(x_name, y_name, True)
            
            if plot:
                # Clear current plot
                self.pareto_canvas.figure.clear()
                
                # Draw the new plot on the canvas
                ax = self.pareto_canvas.figure.add_subplot(111)
                
                # Extract data from the Pareto solutions
                x_values = []
                y_values = []
                
                for solution in self.mco_engine.solutions:
                    if x_name in solution.objective_values and y_name in solution.objective_values:
                        x_values.append(solution.objective_values[x_name])
                        y_values.append(solution.objective_values[y_name])
                
                # Plot points
                ax.scatter(x_values, y_values, c='blue', s=50, label='Pareto Solutions')
                
                # Highlight current solution if available
                if self.current_solution and x_name in self.current_solution.objective_values and y_name in self.current_solution.objective_values:
                    current_x = self.current_solution.objective_values[x_name]
                    current_y = self.current_solution.objective_values[y_name]
                    ax.scatter([current_x], [current_y], c='red', s=100, label='Current Solution')
                
                # Add labels and title
                x_label = x_name.replace("_", " ").title()
                y_label = y_name.replace("_", " ").title()
                ax.set_xlabel(f"{x_label}")
                ax.set_ylabel(f"{y_label}")
                ax.set_title(f"Pareto Front: {x_label} vs {y_label}")
                ax.grid(True)
                ax.legend()
                
                # Refresh canvas
                self.pareto_canvas.draw()
        except Exception as e:
            logger.error(f"Error updating Pareto plot: {e}", exc_info=True)
    
    def _reset_weights(self):
        """Reset all weights to zero."""
        for slider in self.sliders.values():
            slider.set_value(0.0)
    
    def _set_balanced_weights(self):
        """Set equal weights for all objectives."""
        if not self.sliders:
            return
        
        weight = 1.0 / len(self.sliders)
        for slider in self.sliders.values():
            slider.set_value(weight)
        
        # Update plan
        self._update_plan()
    
    def _accept_plan(self):
        """Accept the current plan and close the dialog."""
        if not self.current_solution:
            QMessageBox.warning(self, "No Solution", "No solution available to accept.")
            return
        
        try:
            # Apply the current solution to the plan
            updated_plan = self.mco_engine.accept_current_solution()
            
            if updated_plan:
                # Emit the plan accepted signal
                self.planAccepted.emit(updated_plan)
                
                # Close the dialog
                self.accept()
            else:
                QMessageBox.warning(self, "Error", "Failed to apply solution to plan.")
        except Exception as e:
            logger.error(f"Error accepting plan: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to accept plan: {str(e)}")
    
    def _save_solutions(self):
        """Save the generated solutions to a file."""
        if not self.mco_engine.solutions:
            QMessageBox.warning(self, "No Solutions", "No solutions available to save.")
            return
        
        try:
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save MCO Solutions", "", "JSON Files (*.json)"
            )
            
            if not file_path:
                return
            
            # Add .json extension if missing
            if not file_path.lower().endswith('.json'):
                file_path += '.json'
            
            # Save solutions
            success = self.mco_engine.save_solutions(file_path)
            
            if success:
                QMessageBox.information(
                    self, "Success", f"Saved {len(self.mco_engine.solutions)} solutions to {file_path}"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to save solutions.")
        except Exception as e:
            logger.error(f"Error saving solutions: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to save solutions: {str(e)}")


def show_mco_navigator(plan: Plan, parent=None) -> Optional[Plan]:
    """
    Show the MCO navigator dialog for the given plan.
    
    Args:
        plan: The plan to optimize
        parent: Parent widget
    
    Returns:
        The updated plan if accepted, None otherwise
    """
    # Get objectives and constraints from the plan
    from quangtps.optimization.objectives import get_objectives_from_plan
    from quangtps.optimization.constraints import get_constraints_from_plan
    
    objectives = get_objectives_from_plan(plan)
    constraints = get_constraints_from_plan(plan)
    
    if not objectives:
        QMessageBox.warning(
            parent, "No Objectives", 
            "This plan has no optimization objectives. Please define objectives first."
        )
        return None
    
    # Create MCO engine
    from quangtps.optimization.mco.mco_engine import create_mco_engine
    mco_engine = create_mco_engine(plan, objectives, constraints)
    
    # Create and show dialog
    dialog = MCONavigatorDialog(mco_engine, parent)
    
    # Return plan if accepted
    if dialog.exec_() == QDialog.Accepted:
        return dialog.mco_engine.plan
    
    return None


if __name__ == "__main__":
    # Test code
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create a dummy plan
    class DummyPlan:
        def __init__(self):
            self.name = "Test Plan"
            self.id = "test_plan_1"
        
        def clone(self):
            return DummyPlan()
    
    # Create a dummy objective
    class DummyObjective:
        def __init__(self, name):
            self.name = name
            self.objective_type = type('ObjectiveType', (), {'name': 'MIN_DOSE'})
        
        def evaluate(self, fluence):
            # Mock ObjectiveResult
            result = type('ObjectiveResult', (), {'value': 0.5})
            return result
        
        def to_dict(self):
            return {'name': self.name, 'type': 'MIN_DOSE'}
    
    # Create a dummy MCO engine
    class DummyMCOEngine:
        def __init__(self):
            self.plan = DummyPlan()
            self.objectives = {
                'ptv_coverage': DummyObjective('ptv_coverage'),
                'oar_sparing': DummyObjective('oar_sparing'),
                'conformity': DummyObjective('conformity'),
                'homogeneity': DummyObjective('homogeneity')
            }
            self.constraints = []
            self.solutions = []
            self.current_solution = None
        
        def prepare(self):
            return True
        
        def generate_anchor_plans(self, num_anchors=None):
            return []
        
        def generate_balanced_plan(self):
            return None
        
        def generate_pareto_surface(self, num_points=10, method='weight_sampling'):
            return []
        
        def navigate(self, slider_values):
            return None
        
        def accept_current_solution(self):
            return self.plan
        
        def save_solutions(self, filename):
            return True
    
    # Show the dialog
    mco_engine = DummyMCOEngine()
    dialog = MCONavigatorDialog(mco_engine)
    dialog.show()
    
    sys.exit(app.exec_()) 