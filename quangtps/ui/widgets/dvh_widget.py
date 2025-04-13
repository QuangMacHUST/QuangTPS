#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DVH Widget

This module provides a widget for displaying Dose-Volume Histogram (DVH) curves
for radiotherapy treatment plan evaluation.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.lines import Line2D

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, 
                            QComboBox, QLabel, QPushButton, QFrame, QGroupBox,
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QSplitter, QTabWidget, QToolButton, QMenu, QAction,
                            QToolBar, QSizePolicy, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QIcon, QFont, QPalette

from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh_metrics
from quangtps.evaluation.dvh.dvh_visualization import get_structure_color
from quangtps.ui.styles import Colors
from quangtps.evaluation.dvh.dvh_data import DVHData, DVHCurve
from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class DVHPlot(FigureCanvas):
    """
    Interactive DVH plot for displaying dose-volume histograms.
    
    This class provides an interactive matplotlib-based plot for visualizing
    dose-volume histograms with features like hover tooltips, data selection,
    and customizable styling.
    """
    
    pointSelected = pyqtSignal(str, float, float)  # Structure name, dose, volume
    
    def __init__(self, width=6, height=5, dpi=100):
        """
        Initialize the DVH plot.
        
        Parameters
        ----------
        width : int, optional
            Width of the figure in inches
        height : int, optional
            Height of the figure in inches
        dpi : int, optional
            DPI of the figure
        """
        # Create figure and axis
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # Initialize base class
        super().__init__(self.fig)
        
        # Set up the plot
        self.axes.set_xlabel('Dose (Gy)')
        self.axes.set_ylabel('Volume (%)')
        self.axes.set_title('Cumulative Dose-Volume Histogram')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlim(0, 100)
        self.axes.set_ylim(0, 105)
        
        # Store data
        self.dvh_data = {}
        self.structure_lines = {}
        self.current_mode = 'cumulative'
        self.normalization = 'absolute'  # 'absolute', 'relative', or 'prescription'
        self.prescription_dose = None
        self.selected_structures = {}
        
        # Connect events
        self.mpl_connect('motion_notify_event', self.on_hover)
        self.mpl_connect('button_press_event', self.on_click)
        
        # Hover annotation
        self.hover_annotation = self.axes.annotate(
            '', xy=(0, 0), xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='black'),
            visible=False
        )
        
        # Set tight layout
        self.fig.tight_layout()
    
    def update_dvh(self, dvh_data, selected_structures=None):
        """
        Update the DVH plot with new data.
        
        Parameters
        ----------
        dvh_data : Dict[str, Dict]
            Dictionary containing DVH data for each structure
        selected_structures : Dict[str, bool], optional
            Dictionary mapping structure names to visibility state
        """
        # Clear current plot
        self.axes.clear()
        self.structure_lines = {}
        
        # Store data
        self.dvh_data = dvh_data
        
        # Create a dictionary of structure visibility if not provided
        if selected_structures is None:
            self.selected_structures = {
                name: True for name in dvh_data.keys()
            }
        else:
            self.selected_structures = selected_structures
        
        # Plot each structure
        for name, data in dvh_data.items():
            # Skip if not selected
            if not self.selected_structures.get(name, True):
                continue
            
            # Get volume and dose data based on current mode
            if self.current_mode == 'cumulative':
                volume = data['cumulative_volume']
            else:
                volume = data['differential_volume']
            
            dose_bins = data['dose_bins']
            
            # Apply normalization
            if self.normalization == 'relative' and np.max(dose_bins) > 0:
                dose_bins = dose_bins / np.max(dose_bins) * 100
            elif (self.normalization == 'prescription' and 
                  self.prescription_dose is not None and 
                  self.prescription_dose > 0):
                dose_bins = dose_bins / self.prescription_dose * 100
            
            # Plot the line
            color = get_structure_color(name)
            line, = self.axes.plot(
                dose_bins, volume, 
                label=name,
                linewidth=2,
                color=color
            )
            
            # Store the line reference
            self.structure_lines[name] = line
        
        # Set labels
        self.axes.set_ylabel('Volume (%)')
        
        if self.normalization == 'absolute':
            self.axes.set_xlabel('Dose (Gy)')
        elif self.normalization == 'relative':
            self.axes.set_xlabel('Dose (% of max)')
        elif self.normalization == 'prescription':
            self.axes.set_xlabel('Dose (% of prescription)')
        
        # Set title
        mode_text = 'Cumulative' if self.current_mode == 'cumulative' else 'Differential'
        self.axes.set_title(f'{mode_text} Dose-Volume Histogram')
        
        # Set grid and limits
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlim(0, None)
        self.axes.set_ylim(0, 105)
        
        # Add legend if there are multiple structures
        if len(self.structure_lines) > 1:
            self.axes.legend(loc='best')
        
        # Update the figure
        self.fig.tight_layout()
        self.draw()
    
    def toggle_structure(self, structure_name, visible):
        """
        Toggle visibility of a structure in the DVH plot.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure to toggle
        visible : bool
            Whether the structure should be visible
        """
        self.selected_structures[structure_name] = visible
        self.update_dvh(self.dvh_data, self.selected_structures)
    
    def on_hover(self, event):
        """
        Handle mouse hover events on the plot.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event containing cursor position
        """
        if event.inaxes != self.axes:
            self.hover_annotation.set_visible(False)
            self.draw_idle()
            return
        
        # Get X and Y data
        x, y = event.xdata, event.ydata
        
        # Find the closest line and point
        min_dist = float('inf')
        closest_line = None
        closest_idx = None
        closest_name = None
        
        for name, line in self.structure_lines.items():
            # Skip if invisible
            if not self.selected_structures.get(name, True):
                continue
            
            # Get line data
            line_x, line_y = line.get_data()
            
            # Find closest point
            for i, (xi, yi) in enumerate(zip(line_x, line_y)):
                dist = np.sqrt((x - xi)**2 + (y - yi)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_line = line
                    closest_idx = i
                    closest_name = name
        
        # Show annotation if close enough
        if closest_line and min_dist < 2:
            # Get the exact point
            x_point = closest_line.get_xdata()[closest_idx]
            y_point = closest_line.get_ydata()[closest_idx]
            
            # Update annotation
            self.hover_annotation.xy = (x_point, y_point)
            
            # Set annotation text based on normalization
            if self.normalization == 'absolute':
                self.hover_annotation.set_text(
                    f'{closest_name}\nDose: {x_point:.1f} Gy\nVolume: {y_point:.1f}%'
                )
            else:
                self.hover_annotation.set_text(
                    f'{closest_name}\nDose: {x_point:.1f}%\nVolume: {y_point:.1f}%'
                )
            
            # Show the annotation
            self.hover_annotation.set_visible(True)
        else:
            self.hover_annotation.set_visible(False)
        
        self.draw_idle()
    
    def on_click(self, event):
        """
        Handle mouse click events on the plot.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event containing cursor position and button information
        """
        if event.inaxes != self.axes or event.button != 1:  # Left click only
            return
        
        # Get X and Y data
        x, y = event.xdata, event.ydata
        
        # Find the closest line and point
        min_dist = float('inf')
        closest_name = None
        closest_x = None
        closest_y = None
        
        for name, line in self.structure_lines.items():
            # Skip if invisible
            if not self.selected_structures.get(name, True):
                continue
            
            # Get line data
            line_x, line_y = line.get_data()
            
            # Find closest point
            for i, (xi, yi) in enumerate(zip(line_x, line_y)):
                dist = np.sqrt((x - xi)**2 + (y - yi)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_name = name
                    closest_x = xi
                    closest_y = yi
        
        # Emit signal if close enough
        if closest_name and min_dist < 3:
            self.pointSelected.emit(closest_name, closest_x, closest_y)
    
    def _get_max_dose(self):
        """
        Get the maximum dose across all structures.
        
        Returns
        -------
        float
            Maximum dose in Gy
        """
        max_dose = 0
        for data in self.dvh_data.values():
            if np.max(data['dose_bins']) > max_dose:
                max_dose = np.max(data['dose_bins'])
        return max_dose


class DVHMetricsTable(QTableWidget):
    """Table widget for displaying DVH metrics."""
    
    def __init__(self, parent=None):
        """
        Initialize the DVH metrics table.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Set up table properties
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Set up headers
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        
        # Set minimum size
        self.setMinimumHeight(100)
        
        # Initialize with empty headers
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels(['Structure'])
    
    def update_metrics(self, dvh_data):
        """
        Update the table with metrics from DVH data.
        
        Parameters
        ----------
        dvh_data : Dict[str, Dict]
            Dictionary containing DVH data for each structure
        """
        if not dvh_data:
            self.setRowCount(0)
            return
        
        # Define metrics to display
        metrics = [
            'D2', 'D5', 'D50', 'D95', 'D98', 
            'V5', 'V10', 'V20', 'V30', 'V40', 'V50', 
            'Dmean', 'Dmax', 'Dmin'
        ]
        
        # Set up columns
        self.setColumnCount(len(metrics) + 1)
        headers = ['Structure'] + metrics
        self.setHorizontalHeaderLabels(headers)
        
        # Set up rows
        structure_names = list(dvh_data.keys())
        self.setRowCount(len(structure_names))
        
        # Fill table data
        for row, name in enumerate(structure_names):
            # Set structure name
            item = QTableWidgetItem(name)
            item.setBackground(QColor(get_structure_color(name)))
            item.setForeground(QColor('white'))
            font = QFont()
            font.setBold(True)
            item.setFont(font)
            self.setItem(row, 0, item)
            
            # Calculate metrics
            metrics_values = calculate_dvh_metrics(dvh_data[name])
            
            # Set metric values
            for col, metric in enumerate(metrics, 1):
                value = metrics_values.get(metric, '-')
                if isinstance(value, (int, float)):
                    if metric.startswith('D'):
                        text = f"{value:.1f} Gy"
                    else:
                        text = f"{value:.1f}%"
                else:
                    text = str(value)
                
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(row, col, item)


class DVHWidget(QWidget):
    """Widget for displaying DVH curves."""
    
    def __init__(self, parent=None):
        """
        Initialize the DVH widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Initialize parameters
        self.dvh_curves = {}  # Dictionary of DVH data keyed by structure ID
        self.curve_names = {}  # Dictionary of curve names keyed by structure ID
        self.curve_colors = {}  # Dictionary of curve colors keyed by structure ID
        
        # Display settings
        self.dvh_type = "cumulative"  # "cumulative" or "differential"
        self.volume_type = "relative"  # "relative" or "absolute"
        self.dose_type = "absolute"  # "relative" or "absolute"
        
        # Reference values
        self.reference_dose = 1.0  # Reference dose for normalization in Gy
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create figure and canvas for matplotlib
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create subplot
        self.ax = self.figure.add_subplot(111)
        
        # Initialize plot
        self._setup_plot()
        
        # Create toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add to layout
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
    
    def _setup_plot(self):
        """Set up the plot with appropriate labels and grid."""
        self.ax.clear()
        
        # Set labels based on current display settings
        if self.volume_type == "relative":
            self.ax.set_ylabel("Volume (%)")
        else:
            self.ax.set_ylabel("Volume (cc)")
        
        if self.dose_type == "relative":
            self.ax.set_xlabel("Dose (%)")
        else:
            self.ax.set_xlabel("Dose (Gy)")
        
        # Set title based on DVH type
        if self.dvh_type == "cumulative":
            self.ax.set_title("Cumulative Dose-Volume Histogram")
        else:
            self.ax.set_title("Differential Dose-Volume Histogram")
        
        # Set grid
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set axes limits
        if self.volume_type == "relative":
            self.ax.set_ylim(0, 105)  # 0-105% for relative volume
        
        if self.dose_type == "relative":
            self.ax.set_xlim(0, 105)  # 0-105% for relative dose
        
        # Enable legend
        self.ax.legend()
    
    def add_dvh_curve(self, dvh_data: DVHData, name: str, color: Tuple[float, float, float, float] = None):
        """
        Add a DVH curve to the plot.
        
        Args:
            dvh_data: DVH data object
            name: Name for the curve (usually structure name)
            color: RGBA color tuple for the curve
        """
        if dvh_data is None:
            logger.warning(f"Cannot add DVH curve for {name}: DVH data is None")
            return
        
        # Store the data
        structure_id = dvh_data.structure_id
        self.dvh_curves[structure_id] = dvh_data
        self.curve_names[structure_id] = name
        
        # Use provided color or generate one
        if color is None:
            # Generate a random color if none provided
            import random
            color = (random.random(), random.random(), random.random(), 1.0)
        
        self.curve_colors[structure_id] = color
    
    def clear(self):
        """Clear all DVH curves."""
        self.dvh_curves.clear()
        self.curve_names.clear()
        self.curve_colors.clear()
        
        # Reset plot
        self._setup_plot()
        self.canvas.draw()
    
    def refresh(self):
        """Refresh the DVH display."""
        # Reset plot
        self._setup_plot()
        
        # Plot each curve
        for structure_id, dvh_data in self.dvh_curves.items():
            self._plot_dvh_curve(structure_id)
        
        # Draw legend if we have curves
        if self.dvh_curves:
            self.ax.legend(loc='upper right')
        
        # Redraw canvas
        self.canvas.draw()
    
    def _plot_dvh_curve(self, structure_id: str):
        """
        Plot a DVH curve for the given structure ID.
        
        Args:
            structure_id: ID of the structure to plot
        """
        if structure_id not in self.dvh_curves:
            return
        
        dvh_data = self.dvh_curves[structure_id]
        name = self.curve_names.get(structure_id, structure_id)
        color = self.curve_colors.get(structure_id, (0.5, 0.5, 0.5, 1.0))
        
        # Get dose bins
        dose_bins = np.array(dvh_data.dose_bins)
        
        # Convert dose if needed
        if self.dose_type == "relative" and dvh_data.dose_unit == "Gy":
            if self.reference_dose > 0:
                dose_bins = dose_bins / self.reference_dose * 100.0
        
        # Get appropriate volume data
        if self.dvh_type == "cumulative":
            volume_data = np.array(dvh_data.cumulative_volume)
        else:  # differential
            volume_data = np.array(dvh_data.differential_volume)
        
        # Convert volume if needed
        if self.volume_type == "relative" and dvh_data.volume_unit == "cc":
            if dvh_data.total_volume > 0:
                volume_data = volume_data / dvh_data.total_volume * 100.0
        elif self.volume_type == "absolute" and dvh_data.volume_unit == "%":
            if dvh_data.total_volume > 0:
                volume_data = volume_data * dvh_data.total_volume / 100.0
        
        # Plot the curve
        self.ax.plot(dose_bins, volume_data, label=name, color=color, linewidth=2)
    
    def set_dvh_type(self, dvh_type: str):
        """
        Set the DVH display type.
        
        Args:
            dvh_type: "cumulative" or "differential"
        """
        if dvh_type not in ["cumulative", "differential"]:
            logger.warning(f"Invalid DVH type: {dvh_type}")
            return
            
        self.dvh_type = dvh_type
    
    def set_volume_type(self, volume_type: str):
        """
        Set the volume display type.
        
        Args:
            volume_type: "relative" or "absolute"
        """
        if volume_type not in ["relative", "absolute"]:
            logger.warning(f"Invalid volume type: {volume_type}")
            return
            
        self.volume_type = volume_type
    
    def set_dose_type(self, dose_type: str):
        """
        Set the dose display type.
        
        Args:
            dose_type: "relative" or "absolute"
        """
        if dose_type not in ["relative", "absolute"]:
            logger.warning(f"Invalid dose type: {dose_type}")
            return
            
        self.dose_type = dose_type
    
    def set_reference_dose(self, dose: float):
        """
        Set the reference dose for normalization.
        
        Args:
            dose: Reference dose in Gy
        """
        if dose <= 0:
            logger.warning(f"Invalid reference dose: {dose}")
            return
            
        self.reference_dose = dose
    
    def save_figure(self, file_path: str):
        """
        Save the current figure to a file.
        
        Args:
            file_path: Path to save the figure
        """
        self.figure.savefig(file_path, dpi=300, bbox_inches='tight') 