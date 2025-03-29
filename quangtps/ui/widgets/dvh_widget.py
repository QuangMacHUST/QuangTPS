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

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.lines import Line2D

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, 
                            QComboBox, QLabel, QPushButton, QFrame, QGroupBox,
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QSplitter, QTabWidget, QToolButton, QMenu, QAction)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QIcon, QFont

from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh_metrics
from quangtps.evaluation.dvh.dvh_visualization import get_structure_color
from quangtps.ui.styles import Colors

logger = logging.getLogger(__name__)


class DVHPlot(FigureCanvasQTAgg):
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
    Widget for displaying and analyzing dose-volume histograms.
    
    This widget provides a comprehensive interface for visualizing and 
    analyzing dose-volume histograms, including an interactive plot and
    a table of metrics.
    """
    
    structureSelected = pyqtSignal(str)  # Emitted when a structure is selected
    
    def __init__(self, parent=None):
        """
        Initialize the DVH widget.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Initialize data
        self.dvh_data = {}
        
        # Set up UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create top toolbar
        toolbar_layout = QHBoxLayout()
        
        # Add display mode selection
        display_group = QGroupBox("Display Mode")
        display_layout = QHBoxLayout(display_group)
        
        self.rb_cumulative = QCheckBox("Cumulative")
        self.rb_cumulative.setChecked(True)
        self.rb_differential = QCheckBox("Differential")
        
        # Make the checkboxes mutually exclusive
        self.rb_cumulative.clicked.connect(lambda checked: self._on_display_mode_changed(checked))
        self.rb_differential.clicked.connect(lambda checked: self._on_display_mode_changed(checked))
        
        display_layout.addWidget(self.rb_cumulative)
        display_layout.addWidget(self.rb_differential)
        toolbar_layout.addWidget(display_group)
        
        # Add normalization selection
        norm_group = QGroupBox("Normalization")
        norm_layout = QHBoxLayout(norm_group)
        
        self.combo_normalization = QComboBox()
        self.combo_normalization.addItem("Absolute (Gy)")
        self.combo_normalization.addItem("Relative to Max (%)")
        self.combo_normalization.addItem("Relative to Prescription (%)")
        self.combo_normalization.currentIndexChanged.connect(self._on_normalization_changed)
        
        norm_layout.addWidget(self.combo_normalization)
        toolbar_layout.addWidget(norm_group)
        
        # Add export button
        self.btn_export = QPushButton("Export DVH")
        self.btn_export.setIcon(QIcon.fromTheme("document-save"))
        toolbar_layout.addWidget(self.btn_export)
        
        # Add toolbar to main layout
        main_layout.addLayout(toolbar_layout)
        
        # Create splitter for plot and metrics
        splitter = QSplitter(Qt.Vertical)
        
        # Add plot
        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)
        
        self.dvh_plot = DVHPlot(width=8, height=5)
        self.dvh_plot.pointSelected.connect(self._on_point_selected)
        
        # Add matplotlib toolbar
        nav_toolbar = NavigationToolbar(self.dvh_plot, self)
        
        plot_layout.addWidget(nav_toolbar)
        plot_layout.addWidget(self.dvh_plot)
        
        splitter.addWidget(plot_frame)
        
        # Add metrics table
        metrics_frame = QFrame()
        metrics_layout = QVBoxLayout(metrics_frame)
        
        metrics_label = QLabel("DVH Metrics")
        metrics_label.setAlignment(Qt.AlignCenter)
        metrics_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        
        self.metrics_table = DVHMetricsTable()
        
        metrics_layout.addWidget(metrics_label)
        metrics_layout.addWidget(self.metrics_table)
        
        splitter.addWidget(metrics_frame)
        
        # Set splitter sizes (70% plot, 30% metrics)
        splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter, 1)
    
    def update_dvh(self, dvh_data):
        """
        Update the widget with DVH data.
        
        Parameters
        ----------
        dvh_data : Dict[str, Dict]
            Dictionary containing DVH data for each structure
        """
        self.dvh_data = dvh_data
        
        # Update plot
        self.dvh_plot.update_dvh(dvh_data)
        
        # Update metrics table
        self.metrics_table.update_metrics(dvh_data)
    
    def _on_structure_selected(self, structure_name):
        """
        Handle selection of a structure.
        
        Parameters
        ----------
        structure_name : str
            Name of the selected structure
        """
        # Select in the metrics table
        for row in range(self.metrics_table.rowCount()):
            if self.metrics_table.item(row, 0).text() == structure_name:
                self.metrics_table.selectRow(row)
                break
        
        # Emit signal
        self.structureSelected.emit(structure_name)
    
    def _on_display_mode_changed(self, checked):
        """
        Handle change in display mode.
        
        Parameters
        ----------
        checked : bool
            Whether the button is checked
        """
        # Determine which button was clicked
        if self.sender() == self.rb_cumulative and checked:
            self.rb_differential.setChecked(False)
            self._update_display_mode(True)
        elif self.sender() == self.rb_differential and checked:
            self.rb_cumulative.setChecked(False)
            self._update_display_mode(False)
    
    def _update_display_mode(self, cumulative=True):
        """
        Update the display mode.
        
        Parameters
        ----------
        cumulative : bool, optional
            Whether to show cumulative DVH
        """
        self.dvh_plot.current_mode = 'cumulative' if cumulative else 'differential'
        self.dvh_plot.update_dvh(self.dvh_data, self.dvh_plot.selected_structures)
    
    def _on_normalization_changed(self, normalization_mode):
        """
        Handle change in normalization mode.
        
        Parameters
        ----------
        normalization_mode : int
            Index of the selected normalization mode
        """
        # Map index to normalization mode
        mode_map = {
            0: 'absolute',
            1: 'relative',
            2: 'prescription'
        }
        
        self.dvh_plot.normalization = mode_map.get(normalization_mode, 'absolute')
        self.dvh_plot.update_dvh(self.dvh_data, self.dvh_plot.selected_structures)
    
    def _on_point_selected(self, structure_name, dose, volume):
        """
        Handle selection of a point on the DVH curve.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        dose : float
            Dose value at the selected point
        volume : float
            Volume value at the selected point
        """
        self._on_structure_selected(structure_name) 