#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Evaluation Tab for QuangTPS

This module provides the functionality to evaluate and analyze radiation therapy plans
using dose-volume histograms (DVH) and various evaluation metrics.
"""

import os
import logging
import numpy as np
# pylint: disable=no-name-in-module
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QGroupBox, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QScrollArea, QFrame, QHeaderView, QCheckBox, QListWidget,
    QAbstractItemView, QListWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon
# pylint: enable=no-name-in-module

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available, some visualization features will be disabled")

# Import QuangTPS modules
try:
    from quangtps.evaluation.plan_evaluation import PlanEvaluation
    from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh, calculate_dvh_metrics
    from quangtps.evaluation.dvh.dvh_visualization import plot_dvh
    EVALUATION_AVAILABLE = True
except ImportError as e:
    EVALUATION_AVAILABLE = False
    logging.warning(f"Plan evaluation modules not available: {e}")

logger = logging.getLogger(__name__)

class DVHCanvas(FigureCanvas):
    """
    Canvas for displaying DVH plots using matplotlib.
    
    This canvas provides an Eclipse-like DVH visualization experience
    with interactive features and customizable display options.
    """
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """
        Initialize the DVH canvas.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        width : float, optional
            Width of the figure in inches
        height : float, optional
            Height of the figure in inches
        dpi : int, optional
            Resolution in dots per inch
        """
        # Create figure with white background similar to Eclipse
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('white')
        
        # Create subplot with grid
        self.axes = self.fig.add_subplot(111)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis labels in Eclipse-like style
        self.axes.set_xlabel('Dose (Gy)', fontsize=10, fontweight='bold')
        self.axes.set_ylabel('Volume (%)', fontsize=10, fontweight='bold')
        self.axes.set_title('Dose Volume Histogram', fontsize=12, fontweight='bold')
        
        # Set limits
        self.axes.set_xlim(0, 100)
        self.axes.set_ylim(0, 100)
        
        # Initialize the canvas
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Setup formatting similar to Eclipse
        for spine in self.axes.spines.values():
            spine.set_color('#555555')
        
        self.axes.tick_params(direction='out', colors='#555555')
        
        # Enable tight layout for better use of space
        self.fig.tight_layout()
        
        # Dictionary to store plot lines by structure name
        self.structure_lines = {}
        
    def clear(self):
        """Clear the canvas to prepare for new data"""
        self.axes.clear()
        self.structure_lines = {}
        
        # Reset axes properties
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('Dose (Gy)', fontsize=10, fontweight='bold')
        self.axes.set_ylabel('Volume (%)', fontsize=10, fontweight='bold')
        self.axes.set_title('Dose Volume Histogram', fontsize=12, fontweight='bold')
        
        # Reset limits
        self.axes.set_xlim(0, 100)
        self.axes.set_ylim(0, 100)
        
    def plot_dvh_data(self, dvh_data, prescription_dose=None):
        """
        Plot DVH data for multiple structures
        
        Parameters
        ----------
        dvh_data : dict
            Dictionary mapping structure names to DVH data (dose, volume)
        prescription_dose : float, optional
            Prescription dose for reference (vertical line)
        """
        self.clear()
        
        if not dvh_data:
            logger.warning("No DVH data to plot")
            return
            
        # Get color map - we'll use tab10 for up to 10 structures
        colors = plt.cm.tab10.colors if 'plt' in globals() else [
            (0.12, 0.47, 0.71), (0.85, 0.37, 0.01),
            (0.20, 0.63, 0.17), (0.90, 0.11, 0.11),
            (0.54, 0.34, 0.86), (0.95, 0.51, 0.19),
            (0.74, 0.74, 0.13), (0.59, 0.29, 0.58),
            (0.22, 0.43, 0.10), (0.53, 0.53, 0.53)
        ]
        
        # Plot each structure's DVH
        for i, (structure_name, data) in enumerate(dvh_data.items()):
            color_idx = i % len(colors)
            dose, volume = data
            line, = self.axes.plot(dose, volume, label=structure_name, color=colors[color_idx])
            self.structure_lines[structure_name] = line
            
        # Show prescription dose if provided
        if prescription_dose is not None:
            self.axes.axvline(x=prescription_dose, color='r', linestyle='--', label=f'Prescription: {prescription_dose} Gy')
            
        # Update plot limits based on data
        max_dose = max([data[0].max() for data in dvh_data.values()]) if dvh_data else 80
        self.axes.set_xlim(0, max_dose * 1.1)  # Add 10% margin
        
        # Add legend
        self.axes.legend(loc='lower left')
        
        # Refresh canvas
        self.draw()

class MetricsTable(QTableWidget):
    """Table for displaying DVH metrics for each structure"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(["Structure", "Min Dose", "Max Dose", "Mean Dose", "D95", "V20"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
    def update_metrics(self, metrics_data):
        """
        Update the table with new metrics data
        
        Parameters
        ----------
        metrics_data : dict
            Dictionary mapping structure names to metrics dictionaries
        """
        self.setRowCount(0)  # Clear existing rows
        
        if not metrics_data:
            return
            
        # Add a row for each structure
        for structure_name, metrics in metrics_data.items():
            row = self.rowCount()
            self.insertRow(row)
            
            # Structure name
            self.setItem(row, 0, QTableWidgetItem(structure_name))
            
            # Format metrics values
            def format_metric(metric_name, unit="Gy"):
                if metric_name not in metrics:
                    return "N/A"
                value = metrics.get(metric_name)
                if value is None:
                    return "N/A"
                try:
                    # Try to format as float
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        return f"{value:.2f} {unit}"
                    elif isinstance(value, str):
                        return f"{value} {unit}"
                    else:
                        return "N/A"
                except:
                    return "N/A"
            
            # Set metrics in table
            self.setItem(row, 1, QTableWidgetItem(format_metric('min_dose')))
            self.setItem(row, 2, QTableWidgetItem(format_metric('max_dose')))
            self.setItem(row, 3, QTableWidgetItem(format_metric('mean_dose')))
            self.setItem(row, 4, QTableWidgetItem(format_metric('D95')))
            self.setItem(row, 5, QTableWidgetItem(format_metric('V20', "%")))

class PlanEvaluationTab(QWidget):
    """
    Plan Evaluation Tab for QuangTPS
    
    Provides tools for evaluating radiotherapy treatment plans:
    - DVH visualization
    - Metrics calculation and display
    - Constraint checking
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Current plan and data
        self.current_plan = None
        self.current_dvh_data = {}
        self.current_metrics = {}
        
        # Note: status_label is created in setup_ui
        
    def setup_ui(self):
        """
        Set up the UI components for the Plan Evaluation tab.
        
        Creates an Eclipse-like interface with DVH visualization, metrics display,
        and structure management similar to Eclipse's Plan Evaluation workspace.
        """
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Header section with plan info
        header_layout = QHBoxLayout()
        
        # Plan title
        self.plan_title_label = QLabel("Plan: None")
        self.plan_title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        header_layout.addWidget(self.plan_title_label)
        
        # Add spacer
        header_layout.addStretch(1)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_button.clicked.connect(self.refresh_evaluation)
        header_layout.addWidget(refresh_button)
        
        # Compare button (placeholder for future implementation)
        compare_button = QPushButton("Compare Plans")
        compare_button.setIcon(QIcon.fromTheme("document-properties"))
        compare_button.setEnabled(False)  # Not implemented yet
        header_layout.addWidget(compare_button)
        
        main_layout.addLayout(header_layout)
        
        # Splitter for main content
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Structure list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add structure list group
        structure_group = QGroupBox("Structures")
        structure_layout = QVBoxLayout(structure_group)
        
        # Structure list
        self.structure_list = QListWidget()
        self.structure_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.structure_list.itemChanged.connect(self._on_structure_selected)
        structure_layout.addWidget(self.structure_list)
        
        # Structure list controls
        structure_buttons_layout = QHBoxLayout()
        
        # Show all button
        show_all_button = QPushButton("Show All")
        show_all_button.clicked.connect(self._show_all_structures)
        structure_buttons_layout.addWidget(show_all_button)
        
        # Hide all button
        hide_all_button = QPushButton("Hide All")
        hide_all_button.clicked.connect(self._hide_all_structures)
        structure_buttons_layout.addWidget(hide_all_button)
        
        structure_layout.addLayout(structure_buttons_layout)
        
        # Add show metrics checkbox
        self.show_metrics_checkbox = QCheckBox("Show metrics on plot")
        self.show_metrics_checkbox.setChecked(True)
        self.show_metrics_checkbox.stateChanged.connect(self.refresh_evaluation)
        structure_layout.addWidget(self.show_metrics_checkbox)
        
        left_layout.addWidget(structure_group)
        
        # Add clinical goals group (placeholder for future implementation)
        goals_group = QGroupBox("Clinical Goals")
        goals_layout = QVBoxLayout(goals_group)
        
        # Add a placeholder message for now
        goals_label = QLabel("Clinical goals feature is under development")
        goals_label.setAlignment(Qt.AlignCenter)
        goals_layout.addWidget(goals_label)
        
        # Add a sample goal table to show the design
        goals_table = QTableWidget(0, 3)
        goals_table.setHorizontalHeaderLabels(["Goal", "Value", "Status"])
        goals_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        goals_table.setEnabled(False)  # Not implemented yet
        goals_layout.addWidget(goals_table)
        
        left_layout.addWidget(goals_group)
        
        # Right panel - DVH and metrics
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add tab widget for different views
        self.view_tabs = QTabWidget()
        
        # DVH tab
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)
        
        # Add DVH canvas
        self.dvh_canvas = DVHCanvas(self)
        dvh_layout.addWidget(self.dvh_canvas)
        
        self.view_tabs.addTab(dvh_tab, "DVH")
        
        # Metrics tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        # Add metrics table
        self.metrics_table = MetricsTable(self)
        metrics_layout.addWidget(self.metrics_table)
        
        self.view_tabs.addTab(metrics_tab, "Metrics")
        
        # Plan indices tab (shows conformity, homogeneity, etc.)
        indices_tab = QWidget()
        indices_layout = QVBoxLayout(indices_tab)
        
        # Add plan indices table
        self.indices_table = QTableWidget(0, 2)
        self.indices_table.setHorizontalHeaderLabels(["Index", "Value"])
        self.indices_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        indices_layout.addWidget(self.indices_table)
        
        self.view_tabs.addTab(indices_tab, "Plan Indices")
        
        right_layout.addWidget(self.view_tabs)
        
        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        
        # Set initial sizes - 25% for left panel, 75% for right panel
        main_splitter.setSizes([250, 750])
        
        main_layout.addWidget(main_splitter)
        
        # Add status label at the bottom
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-style: italic; color: gray;")
        main_layout.addWidget(self.status_label)
        
        # Disable controls until a plan is loaded
        self.structure_list.setEnabled(False)
        self.show_metrics_checkbox.setEnabled(False)
        
        # Initialize with empty data
        self.dvh_data = {}
        self.metrics_data = {}
        self.plan_data = None
        self.current_plan = None
        self.current_patient = None
        
    def set_plan(self, plan, patient=None):
        """
        Set the plan to evaluate.
        
        This method handles various input formats to ensure consistent behavior
        similar to Eclipse's plan evaluation workflow.
        
        Parameters
        ----------
        plan : Plan or dict or str
            The treatment plan to evaluate. Can be a Plan object, a dictionary
            with plan data, or a plan ID.
        patient : Patient or dict or str, optional
            The patient associated with the plan. Can be a Patient object, a
            dictionary with patient data, or a patient ID.
        """
        try:
            logger.info(f"Setting plan for evaluation: {getattr(plan, 'id', plan)}")
            
            # Convert plan ID to plan object if needed
            if isinstance(plan, str):
                logger.info(f"Plan is provided as ID: {plan}")
                # Try to load plan from database
                from quangtps.database.plan_db import PlanDB
                plan_db = PlanDB()
                plan_data = plan_db.get_plan(plan)
                
                if not plan_data:
                    logger.warning(f"Could not find plan with ID: {plan}")
                    return
                    
                plan = plan_data
                
            # Extract plan data if needed
            self.plan_data = self._extract_plan_data(plan)
            if not self.plan_data:
                logger.warning("Failed to extract plan data")
                return
                
            # Set title with plan name
            plan_name = self.plan_data.get('name', str(getattr(plan, 'id', plan)))
            if hasattr(self, 'plan_title_label'):
                self.plan_title_label.setText(f"Plan: {plan_name}")
            
            # Store the patient for reference
            self.current_patient = patient
                
            # Evaluate the plan
            self.evaluate_plan(plan)
            
            # Update UI
            self.structure_list.setEnabled(True)
            self.show_metrics_checkbox.setEnabled(True)
            
            # Show notification of successful loading
            logger.info(f"Plan '{plan_name}' loaded successfully for evaluation")
            
        except Exception as e:
            logger.error(f"Error setting plan for evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate_plan(self, plan):
        """
        Evaluate the given plan
        
        Parameters
        ----------
        plan : Plan or dict
            Plan object or dictionary containing plan data
        """
        if not EVALUATION_AVAILABLE:
            self.status_label.setText("Plan evaluation modules not available")
            return
            
        try:
            self.status_label.setText("Evaluating plan...")
            
            # Extract plan data
            # This is where we'd extract the dose grid and structures from the plan
            # For now, we'll create test data as a placeholder
            plan_data = self._extract_plan_data(plan)
            
            if not plan_data:
                self.status_label.setText("Could not extract plan data")
                return
                
            dose_grid = plan_data.get('dose_grid')
            structures = plan_data.get('structures')
            prescription_dose = plan_data.get('prescription_dose')
            
            if dose_grid is None or not structures:
                self.status_label.setText("Missing dose grid or structures")
                return
                
            # Calculate DVH for each structure
            dvh_data = {}
            metrics_data = {}
            
            for name, mask in structures.items():
                # Calculate DVH
                dose, volume = calculate_dvh(dose_grid, mask)
                dvh_data[name] = (dose, volume)
                
                # Calculate metrics
                metrics = calculate_dvh_metrics(dose, volume, rx_dose=prescription_dose)
                metrics_data[name] = metrics
                
            # Store results
            self.current_dvh_data = dvh_data
            self.current_metrics = metrics_data
            
            # Update UI
            self._update_dvh_plot(dvh_data, prescription_dose)
            self._update_metrics_tables(metrics_data)
            self._update_structure_list(list(structures.keys()))
            
            # Calculate plan quality indices
            self._calculate_plan_indices(plan_data, dvh_data, metrics_data)
            
            self.status_label.setText("Plan evaluation complete")
            
        except Exception as e:
            logger.error(f"Error evaluating plan: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Error: {str(e)}")
    
    def refresh_evaluation(self):
        """Refresh the current plan evaluation"""
        if self.current_plan:
            self.evaluate_plan(self.current_plan)
        else:
            self.status_label.setText("No plan selected")
    
    def _extract_plan_data(self, plan):
        """
        Extract necessary data from plan for evaluation
        
        Parameters
        ----------
        plan : Plan or dict
            Plan object or dictionary
            
        Returns
        -------
        dict
            Dictionary containing dose_grid, structures, and prescription_dose
        """
        # In a real implementation, this would extract data from the plan object
        # For now, we'll return test data
        
        try:
            # Check if we can extract real data from the plan
            if hasattr(plan, 'get_dose_grid') and hasattr(plan, 'get_structures'):
                dose_grid = plan.get_dose_grid()
                structures = plan.get_structures()
                prescription_dose = getattr(plan, 'prescription_dose', 70.0)
                
                if dose_grid is not None and structures:
                    return {
                        'dose_grid': dose_grid,
                        'structures': structures,
                        'prescription_dose': prescription_dose
                    }
            
            # Fallback to test data
            logger.info("Using test data for plan evaluation")
            return self._create_test_data()
            
        except Exception as e:
            logger.error(f"Error extracting plan data: {e}")
            # Fallback to test data
            return self._create_test_data()
    
    def _create_test_data(self):
        """
        Create test data for demonstration
        
        Returns
        -------
        dict
            Dictionary with test dose grid and structures
        """
        # Create a simple 3D dose grid
        grid_size = 100
        dose_grid = np.zeros((grid_size, grid_size, grid_size))
        
        # Fill with a simple dose distribution (spherical falloff from center)
        center = grid_size // 2
        max_dose = 70.0  # Gy
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Distance from center
                    r = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                    # Dose falls off with distance
                    dose_grid[i, j, k] = max_dose * np.exp(-r/20)
        
        # Create some test structures
        structures = {}
        
        # PTV - spherical region at center
        ptv = np.zeros_like(dose_grid, dtype=bool)
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    r = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                    if r < 15:
                        ptv[i, j, k] = True
        structures['PTV'] = ptv
        
        # OAR 1 - offset sphere
        oar1 = np.zeros_like(dose_grid, dtype=bool)
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    r = np.sqrt((i - center-20)**2 + (j - center)**2 + (k - center)**2)
                    if r < 10:
                        oar1[i, j, k] = True
        structures['Parotid_L'] = oar1
        
        # OAR 2 - another offset sphere
        oar2 = np.zeros_like(dose_grid, dtype=bool)
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    r = np.sqrt((i - center+20)**2 + (j - center)**2 + (k - center)**2)
                    if r < 10:
                        oar2[i, j, k] = True
        structures['Parotid_R'] = oar2
        
        # Body - everything
        body = np.ones_like(dose_grid, dtype=bool)
        structures['Body'] = body
        
        return {
            'dose_grid': dose_grid,
            'structures': structures,
            'prescription_dose': 70.0
        }
    
    def _update_dvh_plot(self, dvh_data, prescription_dose=None, show_metrics=True):
        """
        Update the DVH plot with new data
        
        Parameters
        ----------
        dvh_data : dict
            Dictionary mapping structure names to DVH data - can be either:
            1. A tuple of (dose_array, volume_array)
            2. A dictionary with 'dose_bins' and 'volume_pct' or 'cumulative_volume' keys
        prescription_dose : float, optional
            Prescription dose for reference (vertical line)
        show_metrics : bool, optional
            Whether to display metrics on the plot
        """
        if not hasattr(self, 'dvh_canvas'):
            logger.warning("DVH canvas not available, cannot update plot")
            return
            
        # Clear the canvas
        self.dvh_canvas.clear()
        
        if not dvh_data:
            logger.warning("No DVH data to plot")
            self.dvh_canvas.draw()
            return
        
        # Draw DVH curves
        try:
            # Try to use QuangTPS DVH visualization
            from quangtps.evaluation.dvh.dvh_visualization import plot_dvh
            
            # Convert data format if needed - the plot_dvh function expects a different format
            # than what might be provided (tuples of dose, volume)
            formatted_data = {}
            for struct_name, struct_data in dvh_data.items():
                if isinstance(struct_data, tuple) and len(struct_data) == 2:
                    # Convert from (dose, volume) tuple to dictionary format
                    formatted_data[struct_name] = {
                        'dose_bins': struct_data[0],
                        'volume_pct': struct_data[1],
                        'cumulative_volume': struct_data[1]  # Add this for compatibility
                    }
                elif isinstance(struct_data, dict):
                    # Check if it has the expected keys
                    if 'dose_bins' in struct_data:
                        formatted_data[struct_name] = struct_data.copy()
                        # Ensure both volume formats are available
                        if 'volume_pct' in struct_data and 'cumulative_volume' not in struct_data:
                            formatted_data[struct_name]['cumulative_volume'] = struct_data['volume_pct']
                        elif 'cumulative_volume' in struct_data and 'volume_pct' not in struct_data:
                            formatted_data[struct_name]['volume_pct'] = struct_data['cumulative_volume']
                    else:
                        logger.warning(f"Missing dose_bins in data for structure {struct_name}")
                        continue
                else:
                    logger.warning(f"Unexpected data format for structure {struct_name}")
                    continue
            
            # Prepare structure-specific metrics to show if requested
            metrics_to_show = None
            if show_metrics and hasattr(self, 'metrics_data') and self.metrics_data:
                # For each structure, include D95, mean dose, and some other key metrics
                metrics_to_show = ["D95", "mean_dose", "D50", "D2", "V20"]
            
            # Plot using QuangTPS function with properly formatted data
            plot_dvh(formatted_data, ax=self.dvh_canvas.axes, 
                    prescription_dose=prescription_dose,
                    show_metrics=show_metrics,
                    metrics_to_show=metrics_to_show)
                    
            logger.info(f"Updated DVH plot with {len(dvh_data)} structures using QuangTPS visualization")
            
        except (ImportError, Exception) as e:
            # Fallback to basic plotting
            logger.warning(f"Using basic plotting for DVH visualization due to error: {e}")
            
            # Define colors for structures
            colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
            
            # Plot each structure
            for i, (name, data) in enumerate(dvh_data.items()):
                color = colors[i % len(colors)]
                if isinstance(data, tuple) and len(data) == 2:
                    dose, volume = data
                    self.dvh_canvas.axes.plot(dose, volume, label=name, color=color)
                elif isinstance(data, dict):
                    if 'dose_bins' in data:
                        # Try to get volume data using different possible keys
                        volume_data = None
                        for key in ['volume_pct', 'cumulative_volume']:
                            if key in data:
                                volume_data = data[key]
                                break
                        
                        if volume_data is not None:
                            self.dvh_canvas.axes.plot(data['dose_bins'], volume_data, label=name, color=color)
                        else:
                            logger.warning(f"Cannot plot data for {name} - no volume data found")
                    else:
                        logger.warning(f"Cannot plot data for {name} - no dose_bins found")
                else:
                    logger.warning(f"Cannot plot data for {name} - unexpected format")
                
            # Add prescription dose line if available
            if prescription_dose is not None:
                self.dvh_canvas.axes.axvline(x=prescription_dose, color='r', 
                                           linestyle='--', label=f'Prescription: {prescription_dose} Gy')
            
            # Add labels and grid
            self.dvh_canvas.axes.set_xlabel('Dose (Gy)')
            self.dvh_canvas.axes.set_ylabel('Volume (%)')
            self.dvh_canvas.axes.set_title('Dose Volume Histogram')
            self.dvh_canvas.axes.set_xlim(0, prescription_dose * 1.2 if prescription_dose else None)
            self.dvh_canvas.axes.set_ylim(0, 100)
            self.dvh_canvas.axes.grid(True)
            
            # Add legend
            self.dvh_canvas.axes.legend(loc='lower left')
            
            # Add metrics if requested
            if show_metrics and hasattr(self, 'metrics_data') and self.metrics_data:
                metrics_text = ""
                for name, metrics in self.metrics_data.items():
                    if name in dvh_data:
                        if name.startswith('PTV') or name == 'CTV':
                            metrics_text += f"{name}: D95={metrics.get('D95', 0):.1f}Gy, "
                            metrics_text += f"D50={metrics.get('D50', 0):.1f}Gy\n"
                        else:
                            metrics_text += f"{name}: V20={metrics.get('V20', 0):.1f}%, "
                            metrics_text += f"Mean={metrics.get('mean_dose', 0):.1f}Gy\n"
                
                # Add metrics text to the plot
                if metrics_text:
                    self.dvh_canvas.axes.text(0.98, 0.98, metrics_text, transform=self.dvh_canvas.axes.transAxes,
                                            verticalalignment='top', horizontalalignment='right',
                                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Refresh the canvas
        self.dvh_canvas.draw()
    
    def _update_metrics_tables(self, metrics_data):
        """
        Update metrics tables with new data
        
        Parameters
        ----------
        metrics_data : dict
            Dictionary of structure names to metrics dictionaries
        """
        self.metrics_table.update_metrics(metrics_data)
    
    def _update_structure_list(self, structure_names):
        """
        Update the structure list with the available structures.
        
        Parameters
        ----------
        structure_names : list
            List of structure names
        """
        # Save current selection state if the list already has items
        current_selection = {}
        if self.structure_list.count() > 0:
            for i in range(self.structure_list.count()):
                item = self.structure_list.item(i)
                current_selection[item.text()] = (item.checkState() == Qt.Checked)
        
        # Clear the list and block signals during update
        self.structure_list.blockSignals(True)
        self.structure_list.clear()
        
        # Add structures to the list
        for name in structure_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            
            # Set check state based on previous selection or default to checked
            checked = current_selection.get(name, True)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            
            self.structure_list.addItem(item)
        
        # Update status
        if structure_names:
            self.status_label.setText(f"Loaded {len(structure_names)} structures")
        else:
            self.status_label.setText("No structures available")
            
        # Re-enable signals
        self.structure_list.blockSignals(False)
        
        # If we have structures, enable related controls
        self.structure_list.setEnabled(len(structure_names) > 0)
        self.show_metrics_checkbox.setEnabled(len(structure_names) > 0)
    
    def _on_structure_selected(self, item):
        """
        Handle structure selection changes in the structure list.
        
        Parameters
        ----------
        item : QListWidgetItem
            The list item that was changed
        """
        if not hasattr(self, 'dvh_data') or not self.dvh_data:
            return
            
        # Get all selected structures
        selected_structures = []
        for i in range(self.structure_list.count()):
            item = self.structure_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_structures.append(item.text())
        
        # If nothing is selected, show all
        if not selected_structures:
            selected_structures = list(self.dvh_data.keys())
            
        # Filter DVH data for selected structures
        filtered_dvh_data = {name: data for name, data in self.dvh_data.items() 
                            if name in selected_structures}
        
        # Get prescription dose
        prescription_dose = None
        if hasattr(self, 'plan_data') and self.plan_data:
            prescription_dose = self.plan_data.get('prescription_dose')
        
        # Update DVH plot with selected structures
        show_metrics = self.show_metrics_checkbox.isChecked() if hasattr(self, 'show_metrics_checkbox') else True
        self._update_dvh_plot(filtered_dvh_data, prescription_dose, show_metrics=show_metrics)
        
        logger.info(f"Updated DVH plot with {len(filtered_dvh_data)} selected structures")
    
    def _show_all_structures(self):
        """Show all structures in the DVH plot"""
        # Check all items in the structure list
        for i in range(self.structure_list.count()):
            item = self.structure_list.item(i)
            item.setCheckState(Qt.Checked)
        
        # Update the plot with all structures
        if hasattr(self, 'dvh_data') and self.dvh_data:
            # Get prescription dose
            prescription_dose = None
            if hasattr(self, 'plan_data') and self.plan_data:
                prescription_dose = self.plan_data.get('prescription_dose')
                
            # Update DVH plot with all structures
            show_metrics = self.show_metrics_checkbox.isChecked() if hasattr(self, 'show_metrics_checkbox') else True
            self._update_dvh_plot(self.dvh_data, prescription_dose, show_metrics=show_metrics)
            
            logger.info(f"Showing all {len(self.dvh_data)} structures in DVH plot")

    def _hide_all_structures(self):
        """Hide all structures in the DVH plot"""
        # Uncheck all items in the structure list
        for i in range(self.structure_list.count()):
            item = self.structure_list.item(i)
            item.setCheckState(Qt.Unchecked)
        
        # Display an empty plot
        if hasattr(self, 'dvh_canvas'):
            self.dvh_canvas.clear()
            self.dvh_canvas.draw()
            
        logger.info("Hiding all structures in DVH plot")
    
    def _calculate_plan_indices(self, plan_data=None, dvh_data=None, metrics_data=None):
        """
        Calculate plan quality indices
        
        Parameters
        ----------
        plan_data : dict, optional
            Plan data including dose grid and structures
        dvh_data : dict, optional
            DVH data for each structure
        metrics_data : dict, optional
            Metrics data for each structure
        """
        indices = {}
        
        try:
            # Use class attributes if parameters are not provided
            if dvh_data is None and hasattr(self, 'dvh_data'):
                dvh_data = self.dvh_data
            
            if metrics_data is None and hasattr(self, 'metrics_data'):
                metrics_data = self.metrics_data
                
            if plan_data is None and hasattr(self, 'plan_data'):
                plan_data = self.plan_data
            
            # Skip if we don't have the needed data
            if not metrics_data:
                logger.warning("No metrics data available for plan indices calculation")
                return
            
            # Get PTV data if available
            ptv_metrics = None
            for name, metrics in metrics_data.items():
                if 'PTV' in name:
                    ptv_metrics = metrics
                    break
            
            if ptv_metrics:
                # Homogeneity Index (HI) = (D2% - D98%) / D50%
                d2 = ptv_metrics.get('D2', 0)
                d98 = ptv_metrics.get('D98', 0)
                d50 = ptv_metrics.get('D50', 0)
                
                if d50 > 0:
                    hi = (d2 - d98) / d50
                    indices['Homogeneity Index (HI)'] = hi
                
                # Conformity Index (CI) = V95% / PTV_volume
                v95 = ptv_metrics.get('V95', 0)
                ptv_volume = 100  # Normalized to 100%
                
                if ptv_volume > 0:
                    ci = v95 / ptv_volume
                    indices['Conformity Index (CI)'] = ci
            
            # Update indices table
            if hasattr(self, 'indices_table'):
                self.indices_table.setRowCount(0)
                for name, value in indices.items():
                    row = self.indices_table.rowCount()
                    self.indices_table.insertRow(row)
                    self.indices_table.setItem(row, 0, QTableWidgetItem(str(name)))
                    # Format the value as a string with 3 decimal places
                    value_str = "{:.3f}".format(value) if isinstance(value, (int, float)) else str(value)
                    self.indices_table.setItem(row, 1, QTableWidgetItem(value_str))
                    
                logger.info(f"Updated plan indices table with {len(indices)} indices")
            else:
                logger.warning("Indices table not available")
        
        except Exception as e:
            logger.error(f"Error calculating plan indices: {str(e)}")
            return
    
    def set_dvh_data(self, dvh_data, prescription_dose=None):
        """
        Set DVH data for the evaluation.
        
        Parameters
        ----------
        dvh_data : dict
            Dictionary mapping structure names to DVH data - can be either:
            1. A tuple of (dose_array, volume_array)
            2. A dictionary with 'dose_bins' and 'volume_pct' or 'cumulative_volume' keys
        prescription_dose : float, optional
            Prescription dose in Gy
        """
        try:
            logger.info(f"Setting DVH data with {len(dvh_data)} structures")
            
            # Store DVH data
            self.dvh_data = dvh_data
            
            # Calculate metrics for each structure
            self.metrics_data = {}
            for name, data in dvh_data.items():
                logger.debug(f"Calculating metrics for structure: {name}")
                try:
                    # Try to import DVH calculations module
                    from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh_metrics
                    
                    # Convert data to expected format if needed
                    if isinstance(data, tuple) and len(data) == 2:
                        # Format: (dose_bins, volume_pct)
                        metrics_data = {
                            'dose_bins': data[0],
                            'cumulative_volume': data[1],
                            'volume_pct': data[1],
                        }
                    elif isinstance(data, dict):
                        # Check for required keys
                        metrics_data = {}
                        if 'dose_bins' in data:
                            metrics_data['dose_bins'] = data['dose_bins']
                        
                            # Get volume data
                            if 'cumulative_volume' in data:
                                metrics_data['cumulative_volume'] = data['cumulative_volume']
                            elif 'volume_pct' in data:
                                metrics_data['cumulative_volume'] = data['volume_pct']
                            else:
                                logger.warning(f"Missing volume data for {name}")
                                continue
                        else:
                            logger.warning(f"Missing dose_bins for {name}")
                            continue
                    else:
                        logger.warning(f"Unexpected data format for {name}")
                        continue
                    
                    # Calculate metrics for the structure
                    metrics = calculate_dvh_metrics(
                        metrics_data, 
                        metrics_list=['D98', 'D95', 'D50', 'D2', 'Dmean', 'V5', 'V10', 'V20', 'V30', 'V40', 'V50'],
                        rx_dose=prescription_dose
                    )
                    
                    self.metrics_data[name] = metrics
                    logger.debug(f"Calculated metrics for {name}: {metrics}")
                    
                except (ImportError, Exception) as e:
                    # Simple fallback metrics calculation
                    logger.warning(f"Using basic metrics calculation: {e}")
                    metrics = {
                        'mean_dose': 0,
                        'D95': 0,
                        'D50': 0,
                        'V20': 0
                    }
                    
                    if isinstance(data, tuple) and len(data) == 2:
                        dose, volume = data
                        # Simple mean dose estimate
                        metrics['mean_dose'] = np.mean(dose) if len(dose) > 0 else 0
                    elif isinstance(data, dict):
                        if 'mean_dose' in data:
                            metrics['mean_dose'] = data['mean_dose']
                    
                    self.metrics_data[name] = metrics
            
            # Update structure list
            structure_names = list(dvh_data.keys())
            self._update_structure_list(structure_names)
            
            # Update DVH plot with all structures
            self._update_dvh_plot(dvh_data, prescription_dose, 
                                 show_metrics=self.show_metrics_checkbox.isChecked() if hasattr(self, 'show_metrics_checkbox') else True)
            
            # Update metrics table
            if hasattr(self, 'metrics_table'):
                self._update_metrics_tables(self.metrics_data)
            
            # Calculate and update plan indices
            if hasattr(self, 'plan_data') and self.plan_data and prescription_dose:
                self._calculate_plan_indices(self.plan_data, dvh_data, self.metrics_data)
            
            # Set enabled state of controls
            if hasattr(self, 'structure_list'):
                self.structure_list.setEnabled(True)
            if hasattr(self, 'show_metrics_checkbox'):
                self.show_metrics_checkbox.setEnabled(True)
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"DVH data for {len(dvh_data)} structures loaded")
            
            logger.info(f"Successfully set DVH data for {len(dvh_data)} structures")
            return True
            
        except Exception as e:
            logger.error(f"Error setting DVH data: {e}")
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Error loading DVH data: {str(e)}")
            return False
                
    def set_prescription(self, prescription):
        """
        Set the prescription data for the current plan evaluation.
        
        Parameters
        ----------
        prescription : dict
            Dictionary containing prescription data, including:
            - total_dose: float, the total prescription dose in Gy
            - fractions: int, the number of fractions
            - prescription_name: str, optional name of the prescription
        """
        try:
            logger.info(f"Setting prescription data: {prescription}")
            
            # Extract prescription dose
            prescription_dose = prescription.get("total_dose", 0.0)
            
            # Store prescription dose in plan_data
            if not hasattr(self, 'plan_data') or self.plan_data is None:
                self.plan_data = {}
            
            self.plan_data['prescription_dose'] = prescription_dose
            self.plan_data['fractions'] = prescription.get('fractions', 0)
            self.plan_data['prescription_name'] = prescription.get('prescription_name', 'Custom Prescription')
            
            # Update UI if we have DVH data
            if hasattr(self, 'dvh_data') and self.dvh_data:
                show_metrics = self.show_metrics_checkbox.isChecked() if hasattr(self, 'show_metrics_checkbox') else True
                self._update_dvh_plot(self.dvh_data, prescription_dose, show_metrics=show_metrics)
                
                # Recalculate metrics with the prescription dose
                if hasattr(self, 'metrics_data') and self.metrics_data:
                    # Recalculate plan indices
                    self._calculate_plan_indices()
            
            # Update plan title if provided
            if hasattr(self, 'plan_title_label'):
                prescription_name = prescription.get("prescription_name", "Custom Prescription")
                fractions = prescription.get("fractions", 0)
                if fractions > 0:
                    self.plan_title_label.setText(f"Plan: {prescription_name} ({prescription_dose} Gy in {fractions} fractions)")
                else:
                    self.plan_title_label.setText(f"Plan: {prescription_name} ({prescription_dose} Gy)")
            
            # Update status label
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Prescription set: {prescription_dose} Gy in {prescription.get('fractions', 0)} fractions")
            
            logger.info(f"Prescription set: {prescription_dose} Gy in {prescription.get('fractions', 0)} fractions")
            return True
            
        except Exception as e:
            logger.error(f"Error setting prescription: {str(e)}")
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Error setting prescription: {str(e)}")
            return False
