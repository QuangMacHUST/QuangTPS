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
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QSize, QMargins
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QGroupBox, QComboBox, QTabWidget, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QMessageBox,
    QScrollArea, QDialog, QToolBar, QAction, QGridLayout, QStackedWidget,
    QSizePolicy
)
from PyQt5.QtGui import QColor, QPalette, QBrush, QFont, QPainter, QPen, QLinearGradient, QIcon, QPixmap

from quangtps.core.types import Plan, Structure
from quangtps.optimization.mco.mco_navigator import MCONavigator
from quangtps.optimization.mco.pareto_surface import ParetoSolution
from quangtps.optimization.objectives import Objective, ObjectiveFunction, ObjectiveType
from quangtps.optimization.constraints import Constraint
from quangtps.core.logging import get_logger
from quangtps.ui.widgets.dvh_widget import DVHWidget
from quangtps.ui.widgets.chart_widgets import PlanEvaluationWidget
from quangtps.ui.dialogs.progress_dialog import ProgressDialog
from quangtps.ui.widgets.objective_widget import ObjectiveEditorWidget
from quangtps.ui.styles import Colors, get_icon
from quangtps.evaluation.dvh.dvh_calculation import DVHCalculator
from quangtps.ui.charts.pareto_chart import ParetoChart
from quangtps.ui.charts.radar_chart import RadarChart
from quangtps.core.services import ServiceRegistry
from quangtps.evaluation.dvh.dvh_data import DVHData, DVHCurve
from quangtps.ui.charts.dvh_plot import DVHPlot
from quangtps.planning.plan import Plan

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
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.value_label)
    
    def _on_slider_changed(self, value: int):
        # Convert slider value (0-100) to actual value (min_val to max_val)
        actual_value = self.min_val + (value / 100.0) * (self.max_val - self.min_val)
        self.value_label.setText(f"{actual_value:.2f}")
        self.valueChanged.emit(self.name, actual_value)
    
    def set_value(self, value: float):
        normalized_value = int((value - self.min_val) / (self.max_val - self.min_val) * 100)
        self.slider.setValue(normalized_value)
    
    def get_value(self) -> float:
        return self.min_val + (self.slider.value() / 100.0) * (self.max_val - self.min_val)


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


class ObjectiveSlider(QWidget):
    """
    Custom slider widget for MCO objective weight adjustment.
    
    Shows an objective name, slider, and value label.
    """
    valueChanged = pyqtSignal(str, float)
    
    def __init__(self, objective_name: str, min_val: float, max_val: float, parent=None):
        super().__init__(parent)
        
        self.objective_name = objective_name
        self.min_val = min_val
        self.max_val = max_val
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Objective name label
        self.name_label = QLabel(self.objective_name)
        self.name_label.setMinimumWidth(150)
        layout.addWidget(self.name_label)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        layout.addWidget(self.slider, 1)
        
        # Value label
        self.value_label = QLabel(self._format_value(self.get_value()))
        self.value_label.setMinimumWidth(80)
        layout.addWidget(self.value_label)
        
        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_value_changed)
    
    def _format_value(self, value: float) -> str:
        """Format the value for display in the label."""
        return f"{value:.2f}"
    
    def _on_slider_value_changed(self, _):
        """Handle slider value changes."""
        value = self.get_value()
        self.value_label.setText(self._format_value(value))
        self.valueChanged.emit(self.objective_name, value)
    
    def get_value(self) -> float:
        """Get the current value mapped from slider position to value range."""
        slider_pos = self.slider.value() / 100.0  # 0 to 1
        return self.min_val + slider_pos * (self.max_val - self.min_val)
    
    def set_value(self, value: float):
        """Set the slider position based on value."""
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        self.slider.setValue(int(normalized * 100))
    
    def get_name(self) -> str:
        """Get the objective name."""
        return self.objective_name


class WeightSlider(QWidget):
    """
    Slider for adjusting interpolation weights between Pareto-optimal plans.
    """
    valueChanged = pyqtSignal(int, float)
    
    def __init__(self, solution_index: int, solution_name: str, parent=None):
        super().__init__(parent)
        
        self.solution_index = solution_index
        self.solution_name = solution_name
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Solution name label
        self.name_label = QLabel(f"Plan {self.solution_index+1}: {self.solution_name}")
        self.name_label.setMinimumWidth(150)
        layout.addWidget(self.name_label)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        layout.addWidget(self.slider, 1)
        
        # Value label
        self.value_label = QLabel("0.00")
        self.value_label.setMinimumWidth(60)
        layout.addWidget(self.value_label)
        
        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_value_changed)
    
    def _on_slider_value_changed(self, _):
        """Handle slider value changes."""
        value = self.get_value()
        self.value_label.setText(f"{value:.2f}")
        self.valueChanged.emit(self.solution_index, value)
    
    def get_value(self) -> float:
        """Get the current weight value (0-1)."""
        return self.slider.value() / 100.0
    
    def set_value(self, value: float):
        """Set the slider position based on weight value (0-1)."""
        self.slider.setValue(int(value * 100))
    
    def get_index(self) -> int:
        """Get the solution index."""
        return self.solution_index


class MCOPanel(QWidget):
    """
    Multi-Criteria Optimization (MCO) panel for navigating the Pareto surface.
    
    This panel contains:
    1. Controls for generating Pareto-optimal plans
    2. Sliders for navigating between plans
    3. Visualization of plan trade-offs
    4. Tools for selecting the final plan
    """
    
    planSelected = pyqtSignal(Plan)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self.mco_navigator: Optional[MCONavigator] = None
        self.current_plan: Optional[Plan] = None
        self.objective_sliders: List[ObjectiveSlider] = []
        self.weight_sliders: List[WeightSlider] = []
        self.is_updating_sliders = False
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Create splitter for left panel and visualization
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # MCO Generation Group
        generation_group = QGroupBox("MCO Generation")
        generation_layout = QVBoxLayout(generation_group)
        
        # Number of plans
        plans_layout = QHBoxLayout()
        plans_layout.addWidget(QLabel("Number of plans:"))
        self.num_plans_spin = QSpinBox()
        self.num_plans_spin.setMinimum(3)
        self.num_plans_spin.setMaximum(20)
        self.num_plans_spin.setValue(7)
        plans_layout.addWidget(self.num_plans_spin)
        generation_layout.addLayout(plans_layout)
        
        # Generate button
        self.generate_button = QPushButton("Generate Pareto Plans")
        self.generate_button.clicked.connect(self._on_generate_clicked)
        generation_layout.addWidget(self.generate_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        generation_layout.addWidget(self.progress_bar)
        
        left_layout.addWidget(generation_group)
        
        # Navigation Group
        navigation_group = QGroupBox("Plan Navigation")
        navigation_layout = QVBoxLayout(navigation_group)
        
        # Tabs for navigation methods
        self.nav_tabs = QTabWidget()
        
        # Tab 1: Objective Sliders
        self.objective_sliders_widget = QWidget()
        self.objective_sliders_layout = QVBoxLayout(self.objective_sliders_widget)
        objective_scroll = QScrollArea()
        objective_scroll.setWidgetResizable(True)
        objective_scroll.setWidget(self.objective_sliders_widget)
        self.nav_tabs.addTab(objective_scroll, "Objectives")
        
        # Tab 2: Plan Interpolation
        self.weight_sliders_widget = QWidget()
        self.weight_sliders_layout = QVBoxLayout(self.weight_sliders_widget)
        self.weight_sliders_layout.setAlignment(Qt.AlignTop)
        weight_scroll = QScrollArea()
        weight_scroll.setWidgetResizable(True)
        weight_scroll.setWidget(self.weight_sliders_widget)
        self.nav_tabs.addTab(weight_scroll, "Interpolation")
        
        navigation_layout.addWidget(self.nav_tabs)
        
        # Normalize weights checkbox
        self.normalize_weights = QCheckBox("Normalize weights")
        self.normalize_weights.setChecked(True)
        self.normalize_weights.toggled.connect(self._on_normalize_changed)
        navigation_layout.addWidget(self.normalize_weights)
        
        # Apply button
        self.apply_button = QPushButton("Apply Current Plan")
        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.apply_button.setEnabled(False)
        navigation_layout.addWidget(self.apply_button)
        
        left_layout.addWidget(navigation_group)
        left_layout.addStretch()
        
        # Right panel (visualization)
        visualization_panel = QWidget()
        visualization_layout = QVBoxLayout(visualization_panel)
        visualization_layout.setContentsMargins(5, 5, 5, 5)
        
        # Visualization tabs
        viz_tabs = QTabWidget()
        
        # Tab 1: DVH
        self.dvh_plot = DVHPlot()
        viz_tabs.addTab(self.dvh_plot, "DVH")
        
        # Tab 2: Radar Chart
        self.radar_chart = RadarChart()
        viz_tabs.addTab(self.radar_chart, "Objectives")
        
        # Tab 3: Pareto Chart
        self.pareto_chart = ParetoChart()
        viz_tabs.addTab(self.pareto_chart, "Pareto Surface")
        
        visualization_layout.addWidget(viz_tabs)
        
        # Add panel and visualization to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(visualization_panel)
        splitter.setSizes([300, 700])
        
        main_layout.addWidget(splitter)
    
    def set_plan(self, plan: Plan):
        """
        Set the current plan for MCO.
        
        Args:
            plan: The plan to use for MCO
        """
        self.current_plan = plan
        
        # Create or reset MCO navigator
        self.mco_navigator = MCONavigator(plan)
        
        # Reset UI
        self._reset_ui()
        
        logger.info(f"MCO panel set to plan: {plan.name}")
        
        # Enable the generate button
        self.generate_button.setEnabled(True)
        self.apply_button.setEnabled(False)
    
    def _reset_ui(self):
        """Reset the UI to initial state."""
        # Clear objective sliders
        self._clear_objective_sliders()
        
        # Clear weight sliders
        self._clear_weight_sliders()
        
        # Reset plots
        self.dvh_plot.clear()
        self.radar_chart.clear()
        self.pareto_chart.clear()
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
    
    def _clear_objective_sliders(self):
        """Clear all objective sliders."""
        self.objective_sliders = []
        
        # Clear layout
        while self.objective_sliders_layout.count():
            item = self.objective_sliders_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def _clear_weight_sliders(self):
        """Clear all weight sliders."""
        self.weight_sliders = []
        
        # Clear layout
        while self.weight_sliders_layout.count():
            item = self.weight_sliders_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def _on_generate_clicked(self):
        """Handle generate button click."""
        if not self.mco_navigator or not self.current_plan:
            QMessageBox.warning(self, "Warning", "No plan selected for MCO")
            return
        
        # Check if objectives exist
        optimization_service = ServiceRegistry.get_service("OptimizationService")
        if optimization_service:
            objectives = optimization_service.get_objectives_for_plan(self.current_plan)
            
            if not objectives or len(objectives) < 2:
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    "At least two objectives are required for MCO.\n"
                    "Please add objectives in the optimizer tab first."
                )
            return
        
            # Set objectives in the MCO navigator
            self.mco_navigator.set_objectives(objectives)
            
            # Get number of plans to generate
            num_plans = self.num_plans_spin.value()
            
            # Show progress bar
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Start generation in a separate thread
            self._start_generation(num_plans)
    
    def _start_generation(self, num_plans: int):
        """
        Start generating Pareto-optimal plans.
        
        In a real implementation, this would be done in a separate thread.
        For simplicity, we do it directly here and use a timer to simulate progress.
        """
        # Disable UI during generation
        self.generate_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        
        # Start generation
        if self.mco_navigator:
            success = self.mco_navigator.generate_pareto_plans(num_plans)
            
            if success:
                # Update UI after generation
                self._update_after_generation()
            else:
                QMessageBox.critical(self, "Error", "Failed to generate Pareto-optimal plans")
                self.progress_bar.setVisible(False)
                self.generate_button.setEnabled(True)
    
    def _update_after_generation(self):
        """Update UI after Pareto plan generation."""
        if not self.mco_navigator:
            return
        
        # Update progress
        self.progress_bar.setValue(100)
        
        # Create objective sliders
        self._create_objective_sliders()
        
        # Create weight sliders
        self._create_weight_sliders()
        
        # Update visualization
        self._update_visualization()
        
        # Enable the apply button
        self.apply_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        
        # Hide progress bar after a delay
        QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
    
    def _create_objective_sliders(self):
        """Create sliders for each objective."""
        if not self.mco_navigator:
            return
        
        # Clear existing sliders
        self._clear_objective_sliders()
        
        # Add a header label
        header = QLabel("Adjust objectives to navigate the Pareto surface:")
        header.setStyleSheet("font-weight: bold;")
        self.objective_sliders_layout.addWidget(header)
        
        # Add a slider for each objective
        for obj_name in self.mco_navigator.pareto_surface.objective_names:
            # Get range of this objective
            min_val, max_val = self.mco_navigator.get_objective_range(obj_name)
            
            # Create slider
            slider = ObjectiveSlider(obj_name, min_val, max_val)
            slider.valueChanged.connect(self._on_objective_slider_changed)
            
            # Add to layout and list
            self.objective_sliders_layout.addWidget(slider)
            self.objective_sliders.append(slider)
        
        # Add stretch at the end
        self.objective_sliders_layout.addStretch()
    
    def _create_weight_sliders(self):
        """Create sliders for interpolation weights."""
        if not self.mco_navigator:
            return
        
        # Clear existing sliders
        self._clear_weight_sliders()
        
        # Add a header label
        header = QLabel("Adjust weights to interpolate between plans:")
        header.setStyleSheet("font-weight: bold;")
        self.weight_sliders_layout.addWidget(header)
        
        # Add a slider for each solution
        for i, solution in enumerate(self.mco_navigator.solutions):
            # Skip if solution doesn't have a plan
            if not solution.plan:
                continue
                
            # Create slider
            name = solution.plan.name if solution.plan.name else f"Solution {i+1}"
            slider = WeightSlider(i, name)
            slider.valueChanged.connect(self._on_weight_slider_changed)
            
            # If this is the first slider, set it to 1.0
            if i == 0:
                slider.set_value(1.0)
                
            # Add to layout and list
            self.weight_sliders_layout.addWidget(slider)
            self.weight_sliders.append(slider)
        
        # Add stretch at the end
        self.weight_sliders_layout.addStretch()
    
    def _on_objective_slider_changed(self, objective_name: str, value: float):
        """
        Handle changes to objective sliders.
        
        Args:
            objective_name: Name of the objective
            value: New value
        """
        if self.is_updating_sliders or not self.mco_navigator:
            return
        
        # Find closest solution to current objective values
        objective_values = {}
        
        for slider in self.objective_sliders:
            objective_values[slider.get_name()] = slider.get_value()
            
        closest_idx = self.mco_navigator.pareto_surface.find_closest_solution(objective_values)
        
        if closest_idx >= 0:
            # Update weight sliders to show this solution
            self.is_updating_sliders = True
            
            for slider in self.weight_sliders:
                if slider.get_index() == closest_idx:
                    slider.set_value(1.0)
            else:
                    slider.set_value(0.0)
                    
            self.is_updating_sliders = False
            
            # Update visualization
            self._update_visualization()
    
    def _on_weight_slider_changed(self, solution_index: int, value: float):
        """
        Handle changes to weight sliders.
        
        Args:
            solution_index: Index of the solution
            value: New weight value
        """
        if self.is_updating_sliders or not self.mco_navigator:
            return
        
        # If normalize is checked, adjust other sliders to make weights sum to 1.0
        if self.normalize_weights.isChecked():
            self._normalize_weights(solution_index)
            
        # Get all weights
        weights = {}
        for slider in self.weight_sliders:
            weight = slider.get_value()
            if weight > 0:
                weights[slider.get_index()] = weight
                
        # If we have weights, interpolate
        if weights:
            # Normalize weights to sum to 1.0
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                normalized_weights = {idx: w / weight_sum for idx, w in weights.items()}
                
                # Interpolate plan
                interpolated_plan = self.mco_navigator.interpolate(normalized_weights)
                
                if interpolated_plan:
                    # Update objective sliders
                    self._update_objective_sliders()
                    
                    # Update visualization
                    self._update_visualization()
    
    def _normalize_weights(self, changed_index: int):
        """
        Normalize weights to sum to 1.0 after a slider change.
        
        Args:
            changed_index: Index of the slider that changed
        """
        self.is_updating_sliders = True
        
        # Get all weights
        weights = {}
        for slider in self.weight_sliders:
            idx = slider.get_index()
            weights[idx] = slider.get_value()
            
        # Calculate sum of weights
        weight_sum = sum(weights.values())
        
        # If sum is greater than 1.0, reduce other weights proportionally
        if weight_sum > 1.0:
            # Calculate how much to reduce
            excess = weight_sum - 1.0
            
            # If the changed slider's weight is >= 1.0, set others to 0
            if weights[changed_index] >= 1.0:
                for slider in self.weight_sliders:
                    if slider.get_index() != changed_index:
                        slider.set_value(0.0)
                    else:
                        slider.set_value(1.0)
            else:
                # Calculate total weight of other sliders
                other_weights_sum = sum(w for i, w in weights.items() if i != changed_index)
                
                if other_weights_sum > 0:
                    # Reduce each slider proportionally
                    for slider in self.weight_sliders:
                        idx = slider.get_index()
                        if idx != changed_index:
                            new_weight = max(0, weights[idx] - (excess * weights[idx] / other_weights_sum))
                            slider.set_value(new_weight)
        
        self.is_updating_sliders = False
    
    def _on_normalize_changed(self, checked: bool):
        """
        Handle changes to the normalize checkbox.
        
        Args:
            checked: Whether the checkbox is checked
        """
        if checked:
            # Normalize weights
            weight_sum = sum(slider.get_value() for slider in self.weight_sliders)
            
            if weight_sum > 0:
                self.is_updating_sliders = True
                
                # Normalize all sliders
                for slider in self.weight_sliders:
                    slider.set_value(slider.get_value() / weight_sum)
                    
                self.is_updating_sliders = False
                
                # Update visualization
                self._on_weight_slider_changed(-1, 0)
    
    def _update_objective_sliders(self):
        """Update objective sliders based on current solution."""
        if not self.mco_navigator or not self.mco_navigator.current_solution:
            return
        
        self.is_updating_sliders = True
        
        for slider in self.objective_sliders:
            obj_name = slider.get_name()
            value = self.mco_navigator.current_solution.get_objective_value(obj_name)
            slider.set_value(value)
            
        self.is_updating_sliders = False
    
    def _update_visualization(self):
        """Update all visualization components."""
        if not self.mco_navigator or not self.mco_navigator.current_solution:
            return
            
        # Update DVH plot
        if self.mco_navigator.current_solution.plan:
            self.dvh_plot.set_plan(self.mco_navigator.current_solution.plan)
            
        # Update radar chart
        if self.mco_navigator.current_solution and self.mco_navigator.solutions:
            self.radar_chart.clear()
            
            # Add all solutions
            for i, sol in enumerate(self.mco_navigator.solutions):
                if i == 0:  # Base solution
                    self.radar_chart.add_solution(sol.objective_values, "Base", QColor(0, 0, 255))
                else:
                    self.radar_chart.add_solution(sol.objective_values, f"Plan {i+1}", QColor(200, 200, 200))
                    
            # Add current solution last so it's on top
            self.radar_chart.add_solution(
                self.mco_navigator.current_solution.objective_values,
                "Current",
                QColor(255, 0, 0)
            )
            
            # Update the chart
            self.radar_chart.update_chart()
            
        # Update Pareto chart
        if self.mco_navigator.pareto_surface and not self.mco_navigator.pareto_surface.is_empty():
            self.pareto_chart.set_pareto_surface(self.mco_navigator.pareto_surface)
            if self.mco_navigator.current_solution:
                self.pareto_chart.highlight_solution(self.mco_navigator.current_solution)
    
    def _on_apply_clicked(self):
        """Handle apply button click."""
        if not self.mco_navigator or not self.mco_navigator.current_solution:
            return
            
        # Apply current solution to create a final plan
        success = self.mco_navigator.apply_current_solution()
        
        if success:
            # Emit planSelected signal with the final plan
            final_plan = self.mco_navigator.base_plan
            self.planSelected.emit(final_plan)
            
            QMessageBox.information(
                self,
                "MCO Complete",
                f"The MCO plan '{final_plan.name}' has been created successfully."
            )
        else:
            QMessageBox.critical(
                self,
                "Error",
                "Failed to create the final MCO plan."
            ) 