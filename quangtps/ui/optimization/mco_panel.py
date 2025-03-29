#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-criteria optimization (MCO) panel.

This module provides a user interface for multi-criteria optimization,
allowing users to navigate the Pareto front and select a preferred plan.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QGroupBox, QComboBox, QTabWidget, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QMessageBox
)
from PyQt5.QtGui import QColor, QPalette, QBrush, QFont

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from quangtps.core.types import Plan, Structure
from quangtps.optimization.mco import MCOEngine, ParetoSolution, create_mco_engine
from quangtps.optimization.objectives import Objective
from quangtps.optimization.constraints import Constraint
from quangtps.core.logging import get_logger
from quangtps.ui.widgets.dvh_widget import DVHWidget
from quangtps.ui.widgets.plan_eval_widget import PlanEvaluationWidget
from quangtps.ui.dialogs.progress_dialog import ProgressDialog
from quangtps.ui.widgets.objective_editor import ObjectiveEditorWidget
from quangtps.ui.styles import Colors, get_icon

logger = get_logger(__name__)


class SliderWithLabel(QWidget):
    """A slider with a name label and a value label."""
    
    valueChanged = pyqtSignal(str, float)
    
    def __init__(self, name: str, min_val: float = 0.0, max_val: float = 1.0, 
                 initial_val: float = 0.5, parent=None):
        super().__init__(parent)
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.name_label = QLabel(name)
        self.name_label.setMinimumWidth(150)
        layout.addWidget(self.name_label)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(int((initial_val - min_val) / (max_val - min_val) * 100))
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 1)
        
        self.value_label = QLabel(f"{initial_val:.2f}")
        self.value_label.setMinimumWidth(50)
        layout.addWidget(self.value_label)
    
    def _on_slider_changed(self, value: int):
        # Convert slider value (0-100) to actual value (min_val to max_val)
        actual_value = self.min_val + (self.max_val - self.min_val) * (value / 100.0)
        self.value_label.setText(f"{actual_value:.2f}")
        self.valueChanged.emit(self.name, actual_value)
    
    def set_value(self, value: float):
        slider_value = int((value - self.min_val) / (max(self.max_val - self.min_val, 0.001)) * 100)
        self.slider.setValue(slider_value)
    
    def get_value(self) -> float:
        slider_value = self.slider.value()
        actual_value = self.min_val + (self.max_val - self.min_val) * (slider_value / 100.0)
        return actual_value


class ObjectiveValueWidget(QWidget):
    """
    Widget for displaying objective values for different solutions.
    """
    
    solutionSelected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        # Create a table for the objective values
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Solution", "Total Score", "Details"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table)
        
        # Add a button layout at the bottom
        button_layout = QHBoxLayout()
        
        self.add_solution_btn = QPushButton("Add Solution")
        self.add_solution_btn.clicked.connect(self._on_add_solution)
        button_layout.addWidget(self.add_solution_btn)
        
        self.delete_solution_btn = QPushButton("Delete Solution")
        self.delete_solution_btn.clicked.connect(self._on_delete_solution)
        self.delete_solution_btn.setEnabled(False)
        button_layout.addWidget(self.delete_solution_btn)
        
        self.save_solutions_btn = QPushButton("Save Solutions")
        self.save_solutions_btn.clicked.connect(self._on_save_solutions)
        button_layout.addWidget(self.save_solutions_btn)
        
        self.load_solutions_btn = QPushButton("Load Solutions")
        self.load_solutions_btn.clicked.connect(self._on_load_solutions)
        button_layout.addWidget(self.load_solutions_btn)
        
        layout.addLayout(button_layout)
        
        # Initialize data
        self.solutions: List[ParetoSolution] = []
    
    def _on_selection_changed(self, selected, deselected):
        self.delete_solution_btn.setEnabled(len(selected.indexes()) > 0)
        if len(selected.indexes()) > 0:
            row = selected.indexes()[0].row()
            self.solutionSelected.emit(row)
    
    def _on_add_solution(self):
        # Placeholder - will be connected to MCO engine
        pass
    
    def _on_delete_solution(self):
        selected_rows = set(index.row() for index in self.table.selectedIndexes())
        if not selected_rows:
            return
        
        # Sort rows in descending order to avoid index shifting when removing
        for row in sorted(selected_rows, reverse=True):
            self.table.removeRow(row)
            if 0 <= row < len(self.solutions):
                self.solutions.pop(row)
    
    def _on_save_solutions(self):
        # Placeholder - will save to a file
        pass
    
    def _on_load_solutions(self):
        # Placeholder - will load from a file
        pass
    
    def update_solutions(self, solutions: List[ParetoSolution]):
        self.solutions = solutions
        self.table.setRowCount(len(solutions))
        
        for i, solution in enumerate(solutions):
            # Solution number
            self.table.setItem(i, 0, QTableWidgetItem(f"Solution {i+1}"))
            
            # Total score (sum of weighted objective values)
            total_score = 0.0
            for name, value in solution.objective_values.items():
                if name in solution.weights:
                    total_score += value * solution.weights[name]
            self.table.setItem(i, 1, QTableWidgetItem(f"{total_score:.2f}"))
            
            # Details
            details = []
            for name, value in solution.objective_values.items():
                weight = solution.weights.get(name, 0.0)
                details.append(f"{name}: {value:.2f} (w={weight:.2f})")
            details_str = ", ".join(details)
            self.table.setItem(i, 2, QTableWidgetItem(details_str))
    
    def get_selected_solution_index(self) -> Optional[int]:
        selected_rows = set(index.row() for index in self.table.selectedIndexes())
        if not selected_rows:
            return None
        return min(selected_rows)  # Just return the first selected row


class TradingWidget(QWidget):
    """
    Widget for trading off between different objectives using sliders.
    """
    
    weightsChanged = pyqtSignal(dict)
    generateSolutionRequested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        # Group box for sliders
        self.sliders_group = QGroupBox("Objective Weights")
        self.sliders_layout = QVBoxLayout(self.sliders_group)
        layout.addWidget(self.sliders_group)
        
        # Add buttons for generating a solution and resetting weights
        button_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Solution")
        self.generate_btn.clicked.connect(self._on_generate_solution)
        button_layout.addWidget(self.generate_btn)
        
        self.reset_btn = QPushButton("Reset Weights")
        self.reset_btn.clicked.connect(self._on_reset_weights)
        button_layout.addWidget(self.reset_btn)
        
        layout.addLayout(button_layout)
        
        # Initialize data
        self.sliders: Dict[str, SliderWithLabel] = {}
        self.weights: Dict[str, float] = {}
        self.objectives: Dict[str, Objective] = {}
    
    def _on_slider_changed(self, name: str, value: float):
        self.weights[name] = value
        self.weightsChanged.emit(self.weights)
    
    def _on_generate_solution(self):
        self.generateSolutionRequested.emit(self.weights)
    
    def _on_reset_weights(self):
        # Reset all sliders to equal weights
        if not self.sliders:
            return
        
        weight = 1.0 / len(self.sliders)
        for slider in self.sliders.values():
            slider.set_value(weight)
    
    def set_objectives(self, objectives: Dict[str, Objective]):
        # Clear existing sliders
        while self.sliders_layout.count():
            item = self.sliders_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        self.sliders = {}
        self.objectives = objectives
        
        # Create sliders for each objective
        equal_weight = 1.0 / len(objectives) if objectives else 0.0
        self.weights = {name: equal_weight for name in objectives}
        
        # Sort objectives by structure type (PTV first, then OAR)
        for name, objective in sorted(objectives.items(), 
                                    key=lambda x: 0 if "PTV" in x[0] else 1):
            slider = SliderWithLabel(name, initial_val=equal_weight)
            slider.valueChanged.connect(self._on_slider_changed)
            self.sliders_layout.addWidget(slider)
            self.sliders[name] = slider
    
    def set_weights(self, weights: Dict[str, float]):
        self.weights = weights.copy()
        for name, weight in weights.items():
            if name in self.sliders:
                self.sliders[name].set_value(weight)


class ParetoPlotWidget(QWidget):
    """
    Widget for visualizing the Pareto front.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        # Matplotlib figure for the Pareto front
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Controls for the plot
        control_layout = QHBoxLayout()
        
        self.x_axis_combo = QComboBox()
        control_layout.addWidget(QLabel("X-Axis:"))
        control_layout.addWidget(self.x_axis_combo)
        
        self.y_axis_combo = QComboBox()
        control_layout.addWidget(QLabel("Y-Axis:"))
        control_layout.addWidget(self.y_axis_combo)
        
        self.plot_btn = QPushButton("Update Plot")
        self.plot_btn.clicked.connect(self._update_plot)
        control_layout.addWidget(self.plot_btn)
        
        layout.addLayout(control_layout)
        
        # Initialize data
        self.solutions: List[ParetoSolution] = []
        self.selected_index: Optional[int] = None
    
    def set_objectives(self, objectives: Dict[str, Objective]):
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        
        # Add objectives to the combo boxes
        for name in objectives:
            self.x_axis_combo.addItem(name)
            self.y_axis_combo.addItem(name)
        
        # Set default selections
        if self.x_axis_combo.count() > 0:
            self.x_axis_combo.setCurrentIndex(0)
        if self.y_axis_combo.count() > 1:
            self.y_axis_combo.setCurrentIndex(1)
    
    def set_solutions(self, solutions: List[ParetoSolution]):
        self.solutions = solutions
        self._update_plot()
    
    def set_selected_solution(self, index: int):
        self.selected_index = index
        self._update_plot()
    
    def _update_plot(self):
        if not self.solutions:
            return
        
        x_name = self.x_axis_combo.currentText()
        y_name = self.y_axis_combo.currentText()
        
        if not x_name or not y_name:
            return
        
        # Extract values for the selected objectives
        x_values = []
        y_values = []
        for solution in self.solutions:
            if x_name in solution.objective_values and y_name in solution.objective_values:
                x_values.append(solution.objective_values[x_name])
                y_values.append(solution.objective_values[y_name])
        
        # Clear the figure
        self.figure.clear()
        
        # Create a subplot
        ax = self.figure.add_subplot(111)
        
        # Plot the Pareto front
        ax.scatter(x_values, y_values, color='blue', marker='o')
        
        # Highlight the selected solution if any
        if self.selected_index is not None and 0 <= self.selected_index < len(self.solutions):
            selected_solution = self.solutions[self.selected_index]
            if x_name in selected_solution.objective_values and y_name in selected_solution.objective_values:
                x_val = selected_solution.objective_values[x_name]
                y_val = selected_solution.objective_values[y_name]
                ax.scatter([x_val], [y_val], color='red', marker='o', s=100)
        
        # Set labels
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title("Pareto Front")
        
        # Draw grid
        ax.grid(True)
        
        # Update the canvas
        self.canvas.draw()


class MCOPanel(QWidget):
    """
    Panel for multi-criteria optimization.
    
    This panel allows users to navigate the Pareto front and select a preferred plan.
    """
    
    planUpdated = pyqtSignal(Plan)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize data
        self.plan: Optional[Plan] = None
        self.mco_engine: Optional[MCOEngine] = None
        self.objectives: Dict[str, Objective] = {}
        self.constraints: List[Constraint] = []
        
        # Set up UI
        self._setup_ui()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Create a splitter for the left and right panes
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left pane - contains objectives, trading, and Pareto plot
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabs for the left pane
        left_tabs = QTabWidget()
        
        # Tab for objectives
        self.objective_editor = ObjectiveEditorWidget()
        left_tabs.addTab(self.objective_editor, "Objectives")
        
        # Tab for trading
        self.trading_widget = TradingWidget()
        self.trading_widget.generateSolutionRequested.connect(self._on_generate_solution)
        left_tabs.addTab(self.trading_widget, "Trading")
        
        # Tab for Pareto visualization
        self.pareto_plot = ParetoPlotWidget()
        left_tabs.addTab(self.pareto_plot, "Pareto Plot")
        
        left_layout.addWidget(left_tabs)
        
        # Add a widget for solution comparison
        self.solution_widget = ObjectiveValueWidget()
        self.solution_widget.solutionSelected.connect(self._on_solution_selected)
        left_layout.addWidget(self.solution_widget)
        
        self.splitter.addWidget(left_widget)
        
        # Right pane - contains plan evaluation
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabs for the right pane
        right_tabs = QTabWidget()
        
        # Tab for DVH
        self.dvh_widget = DVHWidget()
        right_tabs.addTab(self.dvh_widget, "DVH")
        
        # Tab for plan evaluation
        self.plan_eval = PlanEvaluationWidget()
        right_tabs.addTab(self.plan_eval, "Evaluation")
        
        right_layout.addWidget(right_tabs)
        
        # Add buttons for accepting a plan
        button_layout = QHBoxLayout()
        
        self.accept_btn = QPushButton("Accept Plan")
        self.accept_btn.clicked.connect(self._on_accept_plan)
        button_layout.addWidget(self.accept_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        button_layout.addWidget(self.reset_btn)
        
        right_layout.addLayout(button_layout)
        
        self.splitter.addWidget(right_widget)
        
        # Set initial sizes
        self.splitter.setSizes([400, 600])
        
        main_layout.addWidget(self.splitter)
    
    def set_plan(self, plan: Plan):
        """Set the plan to optimize."""
        self.plan = plan
        if plan:
            self.objective_editor.set_plan(plan)
            self.dvh_widget.set_plan(plan)
            self.plan_eval.set_plan(plan)
            
            # Disable UI elements until objectives are set
            self.accept_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
    
    def initialize_mco(self):
        """Initialize the MCO engine with the current objectives and constraints."""
        if not self.plan:
            return False
        
        # Get objectives and constraints from the editor
        self.objectives = self.objective_editor.get_objectives()
        self.constraints = self.objective_editor.get_constraints()
        
        if not self.objectives:
            QMessageBox.warning(self, "Warning", "No objectives defined")
            return False
        
        # Create and prepare the MCO engine
        try:
            self.mco_engine = create_mco_engine(self.plan, self.objectives, self.constraints)
            
            # Show a progress dialog during preparation
            progress_dialog = ProgressDialog("Preparing MCO Engine", "Initializing...", self)
            progress_dialog.show()
            
            # Use a QTimer to update the progress periodically
            def update_progress():
                progress_dialog.set_progress(50)  # Just a placeholder
            
            timer = QTimer(self)
            timer.timeout.connect(update_progress)
            timer.start(100)
            
            # Prepare the engine
            success = self.mco_engine.prepare()
            
            # Stop the timer and close the dialog
            timer.stop()
            progress_dialog.close()
            
            if not success:
                QMessageBox.critical(self, "Error", "Failed to prepare MCO engine")
                return False
            
            # Update UI components with the objectives
            self.trading_widget.set_objectives(self.objectives)
            self.pareto_plot.set_objectives(self.objectives)
            
            # Generate a balanced plan to start with
            solution = self.mco_engine.generate_balanced_plan()
            if solution:
                self.solution_widget.update_solutions([solution])
                self.pareto_plot.set_solutions([solution])
                self._update_plan_display(solution)
            
            # Enable UI elements
            self.accept_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            
            return True
        except Exception as e:
            logger.error(f"Error initializing MCO: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error initializing MCO: {str(e)}")
            return False
    
    def _on_generate_solution(self, weights: Dict[str, float]):
        """Generate a new solution with the given weights."""
        if not self.mco_engine:
            QMessageBox.warning(self, "Warning", "MCO engine not initialized")
            return
        
        # Show a progress dialog during optimization
        progress_dialog = ProgressDialog("Generating Solution", "Optimizing...", self)
        progress_dialog.show()
        
        # Use a QTimer to update the progress periodically
        def update_progress():
            progress_dialog.set_progress(progress_dialog.progress() + 1)
        
        timer = QTimer(self)
        timer.timeout.connect(update_progress)
        timer.start(100)
        
        try:
            # Generate a solution with the given weights
            solution = self.mco_engine._optimize_with_weights(weights)
            
            # Stop the timer and close the dialog
            timer.stop()
            progress_dialog.close()
            
            if solution:
                # Add the solution to the list
                self.mco_engine.solutions.append(solution)
                
                # Update the UI
                self.solution_widget.update_solutions(self.mco_engine.solutions)
                self.pareto_plot.set_solutions(self.mco_engine.solutions)
                
                # Select the new solution
                idx = len(self.mco_engine.solutions) - 1
                self.solution_widget.table.selectRow(idx)
                self._update_plan_display(solution)
            else:
                QMessageBox.warning(self, "Warning", "Failed to generate solution")
        except Exception as e:
            timer.stop()
            progress_dialog.close()
            logger.error(f"Error generating solution: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error generating solution: {str(e)}")
    
    def _on_solution_selected(self, index: int):
        """Handle selection of a solution."""
        if not self.mco_engine or index < 0 or index >= len(self.mco_engine.solutions):
            return
        
        # Get the selected solution
        solution = self.mco_engine.solutions[index]
        
        # Update the trading widget with the weights
        self.trading_widget.set_weights(solution.weights)
        
        # Update the Pareto plot
        self.pareto_plot.set_selected_solution(index)
        
        # Update the plan display
        self._update_plan_display(solution)
    
    def _update_plan_display(self, solution: ParetoSolution):
        """Update the plan display with the given solution."""
        if not self.plan:
            return
        
        # Create a temporary plan with the solution
        temp_plan = self.plan.copy()
        
        # Update the dose grid
        if solution.dose_grid is not None:
            temp_plan.set_dose_grid(solution.dose_grid)
        
        # Update the DVH and evaluation displays
        self.dvh_widget.set_plan(temp_plan)
        self.plan_eval.set_plan(temp_plan)
    
    def _on_accept_plan(self):
        """Accept the current solution and apply it to the plan."""
        if not self.mco_engine:
            return
        
        # Get the selected solution index
        index = self.solution_widget.get_selected_solution_index()
        if index is None:
            QMessageBox.warning(self, "Warning", "No solution selected")
            return
        
        # Get the solution
        solution = self.mco_engine.solutions[index]
        
        # Apply the solution to the plan
        try:
            updated_plan = self.mco_engine.accept_current_solution()
            if updated_plan:
                self.plan = updated_plan
                self.planUpdated.emit(updated_plan)
                QMessageBox.information(self, "Success", "Plan updated successfully")
            else:
                QMessageBox.warning(self, "Warning", "Failed to update plan")
        except Exception as e:
            logger.error(f"Error accepting plan: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error accepting plan: {str(e)}")
    
    def _on_reset(self):
        """Reset the MCO engine."""
        if not self.mco_engine:
            return
        
        # Confirm with the user
        reply = QMessageBox.question(
            self, "Confirm Reset", 
            "Are you sure you want to reset the MCO engine? This will discard all solutions.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Reset the engine
        self.mco_engine.reset()
        
        # Update the UI
        self.solution_widget.update_solutions([])
        self.pareto_plot.set_solutions([])
        
        # Reset the trading widget
        self.trading_widget.set_objectives(self.objectives)
        
        # Reset the plan display
        if self.plan:
            self.dvh_widget.set_plan(self.plan)
            self.plan_eval.set_plan(self.plan) 