#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DVH (Dose-Volume Histogram) widget for QuangTPS.

This module provides a widget for displaying and interacting with DVH data
in the treatment planning system.
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
                            QToolBar)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QIcon, QFont, QPalette

from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh_metrics
from quangtps.evaluation.dvh.dvh_visualization import get_structure_color
from quangtps.ui.styles import Colors
from quangtps.evaluation.dvh.dvh_data import DVHData, DVHCurve

logger = logging.getLogger(__name__)


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
    """
    Widget for displaying Dose-Volume Histograms (DVH).
    Provides interactive visualization of DVH data with options for:
    - Showing/hiding individual structures
    - Displaying dose statistics
    - Exporting data
    - Customizing display options
    """
    
    selection_changed = pyqtSignal(str)  # structure_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.dvh_data: Optional[DVHData] = None
        self.selected_structure_id: Optional[str] = None
        self.structure_visibility: Dict[str, bool] = {}  # structure_id -> visibility
        self.structure_colors: Dict[str, QColor] = {}  # structure_id -> color
        
        # Display options
        self.display_mode = "cumulative"  # "cumulative" or "differential"
        self.show_grid = True
        self.show_legend = True
        self.x_axis_unit = "Gy"  # "Gy" or "%"
        self.y_axis_unit = "%"  # "%" or "cc" (cubic centimeters)
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar for options
        self.toolbar = QToolBar()
        
        # DVH Type combobox
        self.toolbar.addWidget(QLabel("DVH Type:"))
        self.dvh_type_combo = QComboBox()
        self.dvh_type_combo.addItems(["Cumulative", "Differential"])
        self.dvh_type_combo.setCurrentIndex(0)
        self.dvh_type_combo.currentIndexChanged.connect(self._on_dvh_type_changed)
        self.toolbar.addWidget(self.dvh_type_combo)
        
        self.toolbar.addSeparator()
        
        # X-axis units
        self.toolbar.addWidget(QLabel("Dose Units:"))
        self.x_units_combo = QComboBox()
        self.x_units_combo.addItems(["Gy", "%"])
        self.x_units_combo.setCurrentIndex(0)
        self.x_units_combo.currentIndexChanged.connect(self._on_x_units_changed)
        self.toolbar.addWidget(self.x_units_combo)
        
        self.toolbar.addSeparator()
        
        # Y-axis units
        self.toolbar.addWidget(QLabel("Volume Units:"))
        self.y_units_combo = QComboBox()
        self.y_units_combo.addItems(["%", "cc"])
        self.y_units_combo.setCurrentIndex(0)
        self.y_units_combo.currentIndexChanged.connect(self._on_y_units_changed)
        self.toolbar.addWidget(self.y_units_combo)
        
        self.toolbar.addSeparator()
        
        # Display options
        self.grid_checkbox = QCheckBox("Grid")
        self.grid_checkbox.setChecked(self.show_grid)
        self.grid_checkbox.toggled.connect(self._on_grid_toggled)
        self.toolbar.addWidget(self.grid_checkbox)
        
        self.legend_checkbox = QCheckBox("Legend")
        self.legend_checkbox.setChecked(self.show_legend)
        self.legend_checkbox.toggled.connect(self._on_legend_toggled)
        self.toolbar.addWidget(self.legend_checkbox)
        
        main_layout.addWidget(self.toolbar)
        
        # Matplotlib figure and canvas for DVH plot
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)
        
        # Add matplotlib toolbar
        nav_toolbar = NavigationToolbar(self.canvas, self)
        
        main_layout.addWidget(nav_toolbar)
        main_layout.addWidget(self.canvas, stretch=1)
        
        # Initialize the plot
        self.ax = self.figure.add_subplot(111)
        self._setup_axes()
        
        # Connect canvas click event
        self.canvas.mpl_connect('pick_event', self._on_curve_picked)
    
    def _setup_axes(self):
        """Setup the axes with proper labels and grid"""
        self.ax.clear()
        
        # Set labels based on selected units
        x_label = f"Dose [{self.x_axis_unit}]"
        y_label = f"Volume [{self.y_axis_unit}]"
        
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.grid(self.show_grid)
        
        if self.display_mode == "cumulative":
            self.ax.set_title("Cumulative Dose Volume Histogram")
            # For cumulative DVH, y-axis ranges from 0 to 100%
            if self.y_axis_unit == "%":
                self.ax.set_ylim(0, 100)
        else:
            self.ax.set_title("Differential Dose Volume Histogram")
        
        # Set x-axis range based on unit
        if self.x_axis_unit == "Gy":
            self.ax.set_xlim(0, 80)  # Typical range for radiotherapy in Gy
        else:
            self.ax.set_xlim(0, 120)  # Percentage can go over 100%
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def set_dvh_data(self, dvh_data: DVHData):
        """Set DVH data and update the plot"""
        self.dvh_data = dvh_data
        
        # Initialize visibility and colors
        structure_ids = dvh_data.get_structure_ids()
        for struct_id in structure_ids:
            if struct_id not in self.structure_visibility:
                self.structure_visibility[struct_id] = True
            
            if struct_id not in self.structure_colors:
                # Assign color based on structure type
                structure = dvh_data.get_structure(struct_id)
                if structure:
                    if "PTV" in structure.name or "CTV" in structure.name or "GTV" in structure.name:
                        self.structure_colors[struct_id] = QColor(255, 0, 0)  # Red for targets
                    elif any(oar in structure.name for oar in ["Lung", "Heart", "Liver", "Kidney", "Spinal", "Brain", "Cord"]):
                        self.structure_colors[struct_id] = QColor(0, 0, 255)  # Blue for OARs
                    else:
                        self.structure_colors[struct_id] = QColor(0, 180, 0)  # Green for other structures
        
        # Update the plot
        self._update_plot()
    
    def clear(self):
        """Clear the plot"""
        self.dvh_data = None
        self.selected_structure_id = None
        
        # Clear the plot
        self.ax.clear()
        self._setup_axes()
    
    def _update_plot(self):
        """Update the DVH plot with current data and settings"""
        if not self.dvh_data:
            return
        
        # Clear the current plot
        self.ax.clear()
        self._setup_axes()
        
        # Get all structure IDs
        structure_ids = self.dvh_data.get_structure_ids()
        
        # Plot each visible structure
        legend_entries = []
        for struct_id in structure_ids:
            if not self.structure_visibility.get(struct_id, True):
                continue
            
            # Get the structure and its DVH curve
            structure = self.dvh_data.get_structure(struct_id)
            curve = self.dvh_data.get_curve(struct_id)
            
            if not structure or not curve:
                continue
            
            # Get the color
            color = self.structure_colors.get(struct_id)
            matplotlib_color = None
            if color:
                matplotlib_color = (color.red()/255.0, color.green()/255.0, color.blue()/255.0)
            
            # Set line styles
            line_style = '-'
            line_width = 2.0
            
            # Highlight selected structure
            if struct_id == self.selected_structure_id:
                line_width = 3.0
            
            # Get the appropriate data based on display mode and units
            x_data, y_data = self._get_plot_data(curve)
            
            # Plot the curve
            line, = self.ax.plot(
                x_data, y_data, 
                linestyle=line_style, 
                linewidth=line_width,
                color=matplotlib_color,
                label=structure.name,
                picker=5  # Make the line pickable
            )
            
            # Store for legend
            legend_entries.append(line)
        
        # Add legend if enabled
        if self.show_legend and legend_entries:
            self.ax.legend()
        
        # Update the canvas
        self.canvas.draw()
    
    def _get_plot_data(self, curve: DVHCurve) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the appropriate x and y data for plotting based on the current settings.
        
        Args:
            curve: The DVH curve to extract data from
            
        Returns:
            Tuple of (x_data, y_data) as numpy arrays
        """
        # Get dose and volume data
        dose_data = curve.dose_data.copy()
        volume_data = curve.volume_data.copy()
        
        # Transform data based on display mode
        if self.display_mode == "differential":
            # Convert cumulative to differential if needed
            if curve.is_cumulative:
                # Simple approach: diff between adjacent points
                # In a real implementation, proper differentiation should be used
                differential_volume = np.zeros_like(volume_data)
                differential_volume[1:] = np.diff(volume_data)
                differential_volume[0] = volume_data[0]
                volume_data = differential_volume
        else:  # cumulative
            # Convert differential to cumulative if needed
            if not curve.is_cumulative:
                # Simple approach: running sum
                # In a real implementation, proper integration should be used
                volume_data = np.cumsum(volume_data)
                # Normalize
                if np.max(volume_data) > 0:
                    volume_data = 100.0 * volume_data / np.max(volume_data)
        
        # Transform x-axis (dose) based on units
        if self.x_axis_unit == "%":
            # Convert dose to percentage of prescription
            prescription_dose = self.dvh_data.prescription_dose
            if prescription_dose > 0:
                dose_data = 100.0 * dose_data / prescription_dose
        
        # Transform y-axis (volume) based on units
        if self.y_axis_unit == "cc":
            # Convert percentage to absolute volume
            total_volume = curve.total_volume
            volume_data = volume_data * total_volume / 100.0
        
        return dose_data, volume_data
    
    def _on_dvh_type_changed(self, index):
        """Handle DVH type combobox changes"""
        if index == 0:
            self.display_mode = "cumulative"
        else:
            self.display_mode = "differential"
        
        self._update_plot()
    
    def _on_x_units_changed(self, index):
        """Handle X-axis units combobox changes"""
        if index == 0:
            self.x_axis_unit = "Gy"
        else:
            self.x_axis_unit = "%"
        
        self._update_plot()
    
    def _on_y_units_changed(self, index):
        """Handle Y-axis units combobox changes"""
        if index == 0:
            self.y_axis_unit = "%"
        else:
            self.y_axis_unit = "cc"
        
        self._update_plot()
    
    def _on_grid_toggled(self, checked):
        """Handle grid checkbox changes"""
        self.show_grid = checked
        self._update_plot()
    
    def _on_legend_toggled(self, checked):
        """Handle legend checkbox changes"""
        self.show_legend = checked
        self._update_plot()
    
    def _on_curve_picked(self, event):
        """Handle curve picking (clicking) in the plot"""
        # Get the artist (line) that was picked
        artist = event.artist
        
        # Find the corresponding structure ID
        structure_ids = self.dvh_data.get_structure_ids()
        for struct_id in structure_ids:
            structure = self.dvh_data.get_structure(struct_id)
            if structure and structure.name == artist.get_label():
                # Set as selected
                self.selected_structure_id = struct_id
                
                # Emit signal
                self.selection_changed.emit(struct_id)
                
                # Update plot to highlight selected curve
                self._update_plot()
                break 