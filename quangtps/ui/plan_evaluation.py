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
from typing import Dict, List, Optional, Set, Tuple, Any, Union, cast
import random
import sys

# Import PyQt - use PyQt6 if available, otherwise fallback to PyQt5
try:
    from PyQt6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QComboBox,
        QGroupBox,
        QPushButton,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QSplitter,
        QScrollArea,
        QFrame,
        QHeaderView,
        QCheckBox,
        QListWidget,
        QAbstractItemView,
        QListWidgetItem,
        QRadioButton,
        QButtonGroup,
        QMenu,
        QAction,
        QToolBar,
        QSizePolicy,
        QFileDialog,
        QMessageBox,
        QSlider,
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPoint
    from PyQt6.QtGui import QColor, QIcon, QBrush, QImage, QPixmap

    USING_QT6 = True
except ImportError:
    # Fallback to PyQt5
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QComboBox,
        QGroupBox,
        QPushButton,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QSplitter,
        QScrollArea,
        QFrame,
        QHeaderView,
        QCheckBox,
        QListWidget,
        QAbstractItemView,
        QListWidgetItem,
        QRadioButton,
        QButtonGroup,
        QMenu,
        QAction,
        QToolBar,
        QSizePolicy,
        QFileDialog,
        QMessageBox,
        QSlider,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint
    from PyQt5.QtGui import QColor, QIcon, QBrush, QImage, QPixmap

    USING_QT6 = False

# Import OpenCV for contour drawing if available
try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available. Structure contour display will be limited.")

# Import matplotlib for DVH plotting
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib

    # Set the backend based on PyQt version
    if USING_QT6:
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    else:
        matplotlib.use("Qt5Agg")
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    from matplotlib.figure import Figure
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib not available. DVH display will be disabled.")

# Import QuangTPS modules
from quangtps.core.plan import Plan
from quangtps.core.structures import Structure, StructureType
from quangtps.evaluation.dvh.dvh_data import DVHData as DVHDataOriginal
from quangtps.evaluation.dvh.dvh_calculator import calculate_dvh
from quangtps.common.paths import get_icon_path, get_temp_dir
from quangtps.evaluation.clinical_goals import (
    ClinicalGoal,
    GoalResult,
    GoalType,
    GoalOperator,
)
from quangtps.evaluation.clinical_protocols import ClinicalProtocol
from quangtps.evaluation.protocol_manager import ProtocolManager
from quangtps.ui.widgets.dvh_widget import DVHWidget
from quangtps.ui.widgets.metrics_table import MetricsTable
from quangtps.core.logging import get_logger

# Try to import PlanMetric if available
try:
    from quangtps.evaluation.metrics import PlanMetric
except ImportError:
    # Create a placeholder if not available to avoid import errors
    class PlanMetric:
        """Placeholder class for PlanMetric if module not available"""

        pass

    logging.warning(
        "PlanMetric module not available, some evaluation features will be disabled"
    )

logger = get_logger(__name__)


class DVHCanvas(FigureCanvasQTAgg):
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
        self.fig.set_facecolor("white")

        # Create subplot with grid
        self.axes = self.fig.add_subplot(111)
        self.axes.grid(True, linestyle="--", alpha=0.7)

        # Set axis labels in Eclipse-like style
        self.axes.set_xlabel("Dose (Gy)", fontsize=10, fontweight="bold")
        self.axes.set_ylabel("Volume (%)", fontsize=10, fontweight="bold")
        self.axes.set_title("Dose Volume Histogram", fontsize=12, fontweight="bold")

        # Set limits
        self.axes.set_xlim(0, 100)
        self.axes.set_ylim(0, 100)

        # Initialize the canvas
        super().__init__(self.fig)
        self.setParent(parent)

        # Setup formatting similar to Eclipse
        for spine in self.axes.spines.values():
            spine.set_color("#555555")

        self.axes.tick_params(direction="out", colors="#555555")

        # Enable tight layout for better use of space
        self.fig.tight_layout()

        # Dictionary to store plot lines by structure name
        self.structure_lines = {}

    def clear(self):
        """Clear the canvas to prepare for new data"""
        self.axes.clear()
        self.structure_lines = {}

        # Reset axes properties
        self.axes.grid(True, linestyle="--", alpha=0.7)
        self.axes.set_xlabel("Dose (Gy)", fontsize=10, fontweight="bold")
        self.axes.set_ylabel("Volume (%)", fontsize=10, fontweight="bold")
        self.axes.set_title("Dose Volume Histogram", fontsize=12, fontweight="bold")

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
            Prescription dose in Gy, used for normalization if provided
        """
        if not dvh_data:
            return

        # Clear any existing plots
        self.axes.clear()
        self.structure_lines = {}

        # Reset grid and labels
        self.axes.grid(True, linestyle="--", alpha=0.7)
        self.axes.set_xlabel("Dose (Gy)", fontsize=10, fontweight="bold")
        self.axes.set_ylabel("Volume (%)", fontsize=10, fontweight="bold")
        self.axes.set_title("Dose Volume Histogram", fontsize=12, fontweight="bold")

        # Track max dose for setting x-axis limits
        max_dose = 0.0

        # Colors for different structures
        colormap = plt.get_cmap("tab10")
        color_list = [colormap(i) for i in range(10)]  # Get 10 colors from the colormap

        # Plot each structure's DVH
        for i, (structure_name, data) in enumerate(dvh_data.items()):
            if isinstance(data, tuple) and len(data) == 2:
                # Handle the case where data is a tuple of (dose_bins, volume_bins)
                dose_bins, volume_bins = data

                # Skip if empty data
                if len(dose_bins) == 0 or len(volume_bins) == 0:
                    continue

                # Plot the DVH curve
                color = color_list[i % len(color_list)]
                (line,) = self.axes.plot(
                    dose_bins,
                    volume_bins,
                    "-",
                    linewidth=2,
                    color=color,
                    label=structure_name,
                )

                # Store the line for later reference
                self.structure_lines[structure_name] = line

                # Update max dose for axis limits
                if len(dose_bins) > 0:
                    max_dose = max(max_dose, np.max(dose_bins))

            elif (
                isinstance(data, dict) and "dose_bins" in data and "volume_bins" in data
            ):
                # Handle the case where data is a dict with dose_bins and volume_bins
                dose_bins = data["dose_bins"]
                volume_bins = data["volume_bins"]

                # Skip if empty data
                if len(dose_bins) == 0 or len(volume_bins) == 0:
                    continue

                # Plot the DVH curve
                color = color_list[i % len(color_list)]
                (line,) = self.axes.plot(
                    dose_bins,
                    volume_bins,
                    "-",
                    linewidth=2,
                    color=color,
                    label=structure_name,
                )

                # Store the line for later reference
                self.structure_lines[structure_name] = line

                # Update max dose for axis limits
                if len(dose_bins) > 0:
                    max_dose = max(max_dose, np.max(dose_bins))

        # Set x-axis limits based on max dose
        if max_dose > 0:
            # Add 5% margin to max dose
            self.axes.set_xlim(0, max_dose * 1.05)
        else:
            # Default limits if no valid data
            self.axes.set_xlim(0, 100)

        # Set y-axis limits for percentage volume
        self.axes.set_ylim(0, 100)

        # Add prescription dose line if provided
        if prescription_dose is not None and prescription_dose > 0:
            self.axes.axvline(
                x=prescription_dose,
                color="k",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="Prescription",
            )

        # Add legend with small font size for better readability
        self.axes.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0.7,
            ncol=2 if len(dvh_data) > 5 else 1,  # Use two columns for many structures
        )

        # Refresh the canvas
        self.draw()


class MetricsTable(QTableWidget):
    """Table for displaying DVH metrics for each structure"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(
            ["Structure", "Min Dose", "Max Dose", "Mean Dose", "D95", "V20"]
        )
        header = self.horizontalHeader()
        if header:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

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
            self.setItem(row, 1, QTableWidgetItem(format_metric("min_dose")))
            self.setItem(row, 2, QTableWidgetItem(format_metric("max_dose")))
            self.setItem(row, 3, QTableWidgetItem(format_metric("mean_dose")))
            self.setItem(row, 4, QTableWidgetItem(format_metric("D95")))
            self.setItem(row, 5, QTableWidgetItem(format_metric("V20", "%")))


class PlanEvaluationTab(QWidget):
    """
    Tab for evaluating radiotherapy treatment plans with a focus on DVH analysis
    and clinical goals evaluation.
    """

    # Signals
    plan_changed = pyqtSignal(Plan)

    def __init__(self, parent=None):
        """Initialize the plan evaluation tab."""
        super().__init__(parent)

        # Current plan and data
        self.plan = None
        self.dvh_data = {}
        self.current_metrics = {}
        self._selected_structures = []  # Add this line to ensure attribute exists

        # DVH calculation parameters
        self.dvh_type = "cumulative"  # or "differential"
        self.volume_type = "relative"  # or "absolute"
        self.dose_type = "relative"  # or "absolute"

        # Protocol manager
        self.protocol_manager = ProtocolManager()
        self.current_protocol = None

        # Initialize UI
        self._init_ui()

        # Set initial state
        self._update_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)

        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left side - Structure list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Structure list group
        structure_group = QGroupBox("Structures")
        structure_layout = QVBoxLayout(structure_group)

        self.structure_list = QListWidget()
        self.structure_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.structure_list.selectionChanged.connect(
            self._on_structure_selection_changed
        )
        structure_layout.addWidget(self.structure_list)

        # Structure list buttons
        button_layout = QHBoxLayout()

        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self._on_select_all)
        button_layout.addWidget(self.select_all_button)

        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.clicked.connect(self._on_deselect_all)
        button_layout.addWidget(self.deselect_all_button)

        structure_layout.addLayout(button_layout)
        left_layout.addWidget(structure_group)

        # Protocol group
        protocol_group = QGroupBox("Clinical Protocol")
        protocol_layout = QVBoxLayout(protocol_group)

        # Protocol selection
        protocol_select_layout = QHBoxLayout()
        protocol_select_layout.addWidget(QLabel("Protocol:"))

        self.protocol_combo = QComboBox()
        self.protocol_combo.currentTextChanged.connect(self._on_protocol_changed)
        protocol_select_layout.addWidget(self.protocol_combo)

        protocol_layout.addLayout(protocol_select_layout)

        # Apply protocol button
        self.apply_protocol_button = QPushButton("Apply Protocol")
        self.apply_protocol_button.clicked.connect(self._on_apply_protocol)
        protocol_layout.addWidget(self.apply_protocol_button)

        # Protocol goals list
        self.goals_list = QTableWidget()
        self.goals_list.setColumnCount(3)
        self.goals_list.setHorizontalHeaderLabels(["Goal", "Target", "Achieved"])
        self.goals_list.setEditTriggers(QTableWidget.NoEditTriggers)
        self.goals_list.setSelectionBehavior(QTableWidget.SelectRows)
        self.goals_list.horizontalHeader().setStretchLastSection(True)
        self.goals_list.verticalHeader().setVisible(False)
        protocol_layout.addWidget(self.goals_list)

        left_layout.addWidget(protocol_group)

        # DVH options group
        dvh_options_group = QGroupBox("DVH Options")
        dvh_options_layout = QVBoxLayout(dvh_options_group)

        # DVH type
        dvh_type_layout = QHBoxLayout()
        dvh_type_layout.addWidget(QLabel("DVH Type:"))

        self.dvh_type_combo = QComboBox()
        self.dvh_type_combo.addItems(["Cumulative", "Differential"])
        self.dvh_type_combo.currentIndexChanged.connect(self._on_dvh_type_changed)
        dvh_type_layout.addWidget(self.dvh_type_combo)

        dvh_options_layout.addLayout(dvh_type_layout)

        # Volume type
        volume_type_layout = QHBoxLayout()
        volume_type_layout.addWidget(QLabel("Volume:"))

        self.volume_type_combo = QComboBox()
        self.volume_type_combo.addItems(["Relative (%)", "Absolute (cc)"])
        self.volume_type_combo.currentIndexChanged.connect(self._on_volume_type_changed)
        volume_type_layout.addWidget(self.volume_type_combo)

        dvh_options_layout.addLayout(volume_type_layout)

        # Dose type
        dose_type_layout = QHBoxLayout()
        dose_type_layout.addWidget(QLabel("Dose:"))

        self.dose_type_combo = QComboBox()
        self.dose_type_combo.addItems(["Relative (%)", "Absolute (Gy)"])
        self.dose_type_combo.currentIndexChanged.connect(self._on_dose_type_changed)
        dose_type_layout.addWidget(self.dose_type_combo)

        dvh_options_layout.addLayout(dose_type_layout)

        # Export button
        self.export_button = QPushButton("Export DVH")
        self.export_button.clicked.connect(self._on_export_dvh)
        dvh_options_layout.addWidget(self.export_button)

        left_layout.addWidget(dvh_options_group)

        # Add left panel to splitter
        main_splitter.addWidget(left_panel)

        # Right side - DVH display and metrics
        right_panel = QTabWidget()

        # DVH tab
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)

        # DVH plot
        self.dvh_canvas = DVHCanvas(self)
        dvh_layout.addWidget(self.dvh_canvas)

        # Status label
        self.status_label = QLabel("Ready")
        dvh_layout.addWidget(self.status_label)

        right_panel.addTab(dvh_tab, "DVH")

        # Metrics tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)

        self.metrics_table = MetricsTable()
        metrics_layout.addWidget(self.metrics_table)

        right_panel.addTab(metrics_tab, "Metrics")

        # Dose Display tab
        dose_tab = QWidget()
        dose_layout = QVBoxLayout(dose_tab)

        # Create a proper dose display widget
        self.dose_display = DoseDisplayWidget()
        dose_layout.addWidget(self.dose_display)

        right_panel.addTab(dose_tab, "Dose Display")

        # Plan Quality tab (linked to protocol)
        plan_quality_tab = QWidget()
        plan_quality_layout = QVBoxLayout(plan_quality_tab)

        # Will add plan quality widget here when implemented
        self.plan_quality_widget = QLabel(
            "Plan Quality evaluation will be available in a future update"
        )
        plan_quality_layout.addWidget(self.plan_quality_widget)

        right_panel.addTab(plan_quality_tab, "Plan Quality")

        # Add right panel to splitter
        main_splitter.addWidget(right_panel)

        # Set splitter sizes
        main_splitter.setSizes([300, 700])

        # Add splitter to main layout
        main_layout.addWidget(main_splitter)

    def set_plan(self, plan: Optional[Plan]):
        """
        Set the current plan for evaluation.

        Args:
            plan: Plan object or None
        """
        logger.debug(f"Setting plan for evaluation: {plan.name if plan else None}")

        # Update plan
        self.plan = plan

        # Reset data
        self.dvh_data = {}
        self.current_metrics = {}

        # Calculate DVH data if plan is available
        if plan:
            try:
                self._calculate_dvh_data()
                self.status_label.setText(f"Loaded plan: {plan.name}")
            except Exception as e:
                # Handle error
                logger.error(f"Error calculating DVH data: {str(e)}")
                self.status_label.setText(f"Error calculating DVH data: {str(e)}")
        else:
            self.status_label.setText("No plan loaded")

        # Update UI
        self._update_ui()

        # Emit signal
        self.plan_changed.emit(plan)

    def _update_ui(self):
        """Update UI elements based on current state."""
        # Update structure list
        self._update_structure_list()

        # Update DVH display
        self._update_dvh()

        # Update metrics display
        self._update_metrics()

        # Update goals display based on protocol
        self._update_goals()

        # Enable/disable controls based on plan availability
        has_plan = self.plan is not None
        self.structure_list.setEnabled(has_plan)
        self.select_all_button.setEnabled(has_plan)
        self.deselect_all_button.setEnabled(has_plan)
        self.apply_protocol_button.setEnabled(
            has_plan and self.current_protocol is not None
        )
        self.export_button.setEnabled(has_plan and bool(self.dvh_data))

    def _update_structure_list(self):
        """Update the structure list with structures from the plan."""
        self.structure_list.clear()

        if not self.plan:
            return

        # Get structures from plan
        structures = []
        if hasattr(self.plan, "structures"):
            structures = self.plan.structures
        elif hasattr(self.plan, "structure_set") and self.plan.structure_set:
            structures = self.plan.structure_set.structures

        # Add structures to list
        for structure in structures:
            item = QListWidgetItem(structure.name)

            # Set color
            if hasattr(structure, "color"):
                color = structure.color
                # Handle different color formats
                if isinstance(color, QColor):
                    item.setForeground(color)
                elif isinstance(color, (list, tuple)) and len(color) >= 3:
                    # Convert RGB[A] list to QColor
                    if len(color) == 3:
                        qcolor = QColor(
                            int(color[0] * 255),
                            int(color[1] * 255),
                            int(color[2] * 255),
                        )
                    else:  # RGBA
                        qcolor = QColor(
                            int(color[0] * 255),
                            int(color[1] * 255),
                            int(color[2] * 255),
                            int(color[3] * 255),
                        )
                    item.setForeground(qcolor)
                elif isinstance(color, str):
                    # Handle hex or named colors
                    item.setForeground(QColor(color))

            # Store structure data
            item.setData(Qt.ItemDataRole.UserRole, structure)

            # Add to list and select by default
            self.structure_list.addItem(item)
            item.setSelected(True)

    def _update_dvh(self):
        """Update the DVH display with current data."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Get DVH data
        dvh_data = self._calculate_dvh_data()
        if not dvh_data:
            # Clear the DVH display
            if hasattr(self, "dvh_canvas"):
                self.dvh_canvas.clear()
                self.dvh_canvas.draw()
            return

        # Plot the DVH
        if hasattr(self, "dvh_canvas"):
            prescription_dose = None
            if self.plan and hasattr(self.plan, "prescription_dose"):
                prescription_dose = self.plan.prescription_dose
            self.dvh_canvas.plot_dvh_data(dvh_data, prescription_dose=prescription_dose)

        # Update metrics table based on DVH data
        self._dvh_data = dvh_data
        self._update_metrics()

    def _update_metrics(self):
        """Update the metrics table with DVH-derived metrics."""
        if not self.plan or not hasattr(self, "_dvh_data") or not self._dvh_data:
            # Clear metrics table
            if hasattr(self, "metrics_table"):
                self.metrics_table.update_metrics({})
            return

        # Prepare metrics data for each structure
        metrics_data = {}

        for structure_name, data in self._dvh_data.items():
            # Skip if not valid data
            if not isinstance(data, tuple) or len(data) != 2:
                continue

            dose_bins, volume_bins = data
            if len(dose_bins) == 0 or len(volume_bins) == 0:
                continue

            # Calculate basic metrics
            max_dose = np.max(dose_bins) if len(dose_bins) > 0 else 0

            # Find min dose (excluding zero)
            non_zero_doses = (
                dose_bins[dose_bins > 0] if len(dose_bins) > 0 else np.array([])
            )
            min_dose = np.min(non_zero_doses) if len(non_zero_doses) > 0 else 0

            # Calculate mean dose (approximately)
            # For cumulative DVH, can approximate by taking area under curve
            volume_diff = np.diff(np.append(volume_bins, [0]))
            mean_dose = (
                np.sum(dose_bins * -volume_diff) / 100.0 if len(dose_bins) > 0 else 0
            )

            # Calculate D95 (dose to 95% of volume)
            # For cumulative DVH, find dose at volume=95
            d95 = 0
            if len(dose_bins) > 0 and len(volume_bins) > 0:
                # Interpolate to find dose at 95% volume
                try:
                    # Depending on how volume_bins is ordered
                    if volume_bins[0] > volume_bins[-1]:  # Descending
                        d95 = np.interp(95, volume_bins[::-1], dose_bins[::-1])
                    else:  # Ascending
                        d95 = np.interp(95, volume_bins, dose_bins)
                except:
                    d95 = 0

            # Calculate V20 (volume receiving 20 Gy)
            v20 = 0
            if len(dose_bins) > 0 and len(volume_bins) > 0:
                # Interpolate to find volume at 20 Gy
                try:
                    # Depending on how volume_bins is ordered
                    if (
                        volume_bins[0] > volume_bins[-1]
                    ):  # Descending (typical for cumulative DVH)
                        v20 = np.interp(20, dose_bins, volume_bins)
                    else:  # Ascending
                        # Find closest index where dose >= 20
                        idx = np.searchsorted(dose_bins, 20)
                        if idx < len(dose_bins):
                            v20 = volume_bins[idx]
                except:
                    v20 = 0

            # Add metrics for this structure
            metrics_data[structure_name] = {
                "min_dose": min_dose,
                "max_dose": max_dose,
                "mean_dose": mean_dose,
                "D95": d95,
                "V20": v20,
            }

        # Update the metrics table
        if hasattr(self, "metrics_table"):
            self.metrics_table.update_metrics(metrics_data)

    def _update_goals(self):
        """Update clinical goals display based on protocol."""
        # Clear existing goals
        self.goals_list.setRowCount(0)

        if not self.plan or not self.current_protocol:
            return

        # Evaluate goals against current plan
        goal_results = self._evaluate_protocol()

        # Add results to table
        for i, (goal, result) in enumerate(goal_results):
            # Add row
            self.goals_list.insertRow(i)

            # Goal description
            goal_text = f"{goal.structure_name}: {goal.description}"
            goal_item = QTableWidgetItem(goal_text)
            self.goals_list.setItem(i, 0, goal_item)

            # Target value
            target_text = f"{goal.value}{goal.unit}"
            target_item = QTableWidgetItem(target_text)
            self.goals_list.setItem(i, 1, target_item)

            # Achieved value
            achieved_text = f"{result.value:.2f}{goal.unit}"
            achieved_item = QTableWidgetItem(achieved_text)

            # Color based on result
            if result.passed:
                achieved_item.setBackground(QColor(200, 255, 200))  # Light green
            else:
                achieved_item.setBackground(QColor(255, 200, 200))  # Light red

            self.goals_list.setItem(i, 2, achieved_item)

        # Resize columns to content
        self.goals_list.resizeColumnsToContents()

    def _calculate_dvh_data(self):
        """Calculate DVH data for the current plan and selected structures."""
        if (
            not self.plan
            or not hasattr(self.plan, "has_dose")
            or not self.plan.has_dose()
        ):
            return {}

        # Get dose data
        if not hasattr(self.plan, "get_dose_grid"):
            return {}

        dose_grid = self.plan.get_dose_grid()
        if dose_grid is None:
            logger.warning("No dose grid available for DVH calculation")
            return {}

        # Get voxel dimensions
        voxel_size = (
            dose_grid.get_voxel_size() if hasattr(dose_grid, "get_voxel_size") else None
        )
        if not voxel_size:
            logger.warning("Invalid voxel size for DVH calculation")
            return {}

        # Calculate DVH for each selected structure
        dvh_data = {}
        if not hasattr(self.plan, "structure_set") or self.plan.structure_set is None:
            return {}

        structure_set = self.plan.structure_set

        # Get selected structures
        selected_structures = []
        for i in range(self.structure_list.count()):
            item = self.structure_list.item(i)
            if item and item.isSelected():
                structure = item.data(Qt.ItemDataRole.UserRole)
                if structure and hasattr(structure, "id"):
                    selected_structures.append(structure.id)

        self._selected_structures = selected_structures

        for structure_id in self._selected_structures:
            structure = structure_set.get_structure(structure_id)
            if not structure:
                continue

            # Get structure mask aligned with dose grid
            structure_mask = structure_set.get_structure_mask(
                structure_id, dose_grid.shape
            )
            if structure_mask is None:
                logger.warning(f"Cannot get mask for structure {structure.name}")
                continue

            # Get structure volume in cc
            structure_volume = (
                structure.get_volume() if hasattr(structure, "get_volume") else 0
            )
            if structure_volume <= 0:
                logger.warning(f"Invalid volume for structure {structure.name}")
                continue

            try:
                # Use the calculate_dvh function from dvh_calculator
                dose_bins, volume_bins = calculate_dvh(
                    dose_grid=dose_grid.data,
                    structure_mask=structure_mask,
                    num_bins=100,
                )

                # Store the result
                if len(dose_bins) > 0 and len(volume_bins) > 0:
                    # We're storing tuples for simplicity here, but could create DVHData objects
                    dvh_data[structure.name] = {
                        "dose_bins": dose_bins,
                        "volume_bins": volume_bins,
                    }
            except Exception as e:
                logger.error(f"Error calculating DVH for {structure.name}: {str(e)}")

        return dvh_data

    def _evaluate_protocol(self) -> List[Tuple[ClinicalGoal, GoalResult]]:
        """
        Evaluate clinical protocol goals against the current plan.

        Returns:
            List of tuples with (goal, result) pairs
        """
        results = []

        if not self.plan or not self.current_protocol:
            return results

        # Get DVH data
        if not self.dvh_data:
            try:
                self._calculate_dvh_data()
            except Exception as e:
                logger.error(
                    f"Error calculating DVH data for protocol evaluation: {str(e)}"
                )
                return results

        # Evaluate each goal
        for goal in self.current_protocol.goals:
            # Find structure by name
            structure_id = None
            for i in range(self.structure_list.count()):
                item = self.structure_list.item(i)
                structure = item.data(Qt.ItemDataRole.UserRole)
                if structure.name.lower() == goal.structure_name.lower():
                    structure_id = structure.id
                    break

            if not structure_id or structure_id not in self.dvh_data:
                # Structure not found or no DVH data
                result = GoalResult(goal, False, 0.0)
            else:
                # Evaluate goal against DVH data
                try:
                    # This is a simplified evaluation - replace with actual implementation
                    dvh = self.dvh_data[structure_id]

                    value = 0.0
                    if goal.type == GoalType.D_X:
                        # Dose to X% of volume
                        if hasattr(dvh, "get_dose_at_volume"):
                            value = dvh.get_dose_at_volume(goal.parameter)
                    elif goal.type == GoalType.V_X:
                        # Volume receiving X Gy
                        if hasattr(dvh, "get_volume_at_dose"):
                            value = dvh.get_volume_at_dose(goal.parameter)
                    elif goal.type == GoalType.MAX_DOSE:
                        # Maximum dose
                        if hasattr(dvh, "get_max_dose"):
                            value = dvh.get_max_dose()
                    elif goal.type == GoalType.MEAN_DOSE:
                        # Mean dose
                        if hasattr(dvh, "get_mean_dose"):
                            value = dvh.get_mean_dose()

                    # Create result
                    result = GoalResult(
                        goal,
                        value <= goal.value
                        if goal.direction == "upper"
                        else value >= goal.value,
                        value,
                    )

                except Exception as e:
                    logger.error(f"Error evaluating goal {goal.description}: {str(e)}")
                    result = GoalResult(goal, False, 0.0)

            results.append((goal, result))

        return results

    def _populate_protocols(self):
        """Populate the protocol selector with available protocols."""
        self.protocol_combo.clear()

        # Add empty option
        self.protocol_combo.addItem("None")

        # Get protocols from manager
        for protocol_name in self.protocol_manager.get_protocol_names():
            self.protocol_combo.addItem(protocol_name)

    def _get_structure_color(
        self, structure_id: str
    ) -> Tuple[float, float, float, float]:
        """
        Get the color for a structure.

        Args:
            structure_id: Structure identifier

        Returns:
            RGBA color tuple
        """
        # Default color
        default_color = (1.0, 0.0, 0.0, 1.0)  # Red

        if not self.plan:
            return default_color

        # Find structure in plan
        structure = None
        structures = []

        if hasattr(self.plan, "structures"):
            structures = self.plan.structures
        elif hasattr(self.plan, "structure_set") and self.plan.structure_set:
            structures = self.plan.structure_set.structures

        for s in structures:
            if s.id == structure_id:
                structure = s
                break

        if not structure:
            return default_color

        # Get color from structure
        if hasattr(structure, "color"):
            color = structure.color

            # Handle different color formats
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                if len(color) == 3:
                    return (color[0], color[1], color[2], 1.0)
                else:
                    return tuple(color)
            elif isinstance(color, QColor):
                return (color.redF(), color.greenF(), color.blueF(), color.alphaF())
            elif isinstance(color, str):
                qcolor = QColor(color)
                return (qcolor.redF(), qcolor.greenF(), qcolor.blueF(), qcolor.alphaF())

        return default_color

    def _on_structure_selection_changed(self):
        """Handle structure selection changes."""
        # Update DVH display with selected structures
        self._update_dvh()

        # Update metrics display
        self._update_metrics()

    def _on_select_all(self):
        """Handle select all button click."""
        for i in range(self.structure_list.count()):
            item = self.structure_list.item(i)
            item.setSelected(True)

    def _on_deselect_all(self):
        """Handle deselect all button click."""
        for i in range(self.structure_list.count()):
            item = self.structure_list.item(i)
            item.setSelected(False)

    def _on_protocol_changed(self, protocol_name: str):
        """
        Handle protocol selection change.

        Args:
            protocol_name: Name of the selected protocol
        """
        if protocol_name == "None":
            self.current_protocol = None
        else:
            # Get protocol from manager
            self.current_protocol = self.protocol_manager.get_protocol(protocol_name)

        # Update apply button state
        self.apply_protocol_button.setEnabled(
            self.plan is not None and self.current_protocol is not None
        )

        # Update UI
        self._update_goals()

    def _on_apply_protocol(self):
        """Handle apply protocol button click."""
        if not self.plan or not self.current_protocol:
            return

        # Evaluate protocol
        goal_results = self._evaluate_protocol()

        # Update goals display
        self._update_goals()

        # Update status
        passed_count = sum(1 for _, result in goal_results if result.passed)
        total_count = len(goal_results)
        self.status_label.setText(
            f"Protocol '{self.current_protocol.name}' applied: "
            f"{passed_count}/{total_count} goals passed"
        )

    def _on_dvh_type_changed(self, dvh_type: int):
        """
        Handle DVH type change.

        Args:
            dvh_type: Index of selected DVH type (0=Cumulative, 1=Differential)
        """
        self.dvh_type = "cumulative" if dvh_type == 0 else "differential"

        # Update DVH display
        self._update_dvh()

        # Update status
        self.status_label.setText(f"DVH type changed to {self.dvh_type}")

    def _on_volume_type_changed(self, index: int):
        """
        Handle volume type change.

        Args:
            index: Index of selected volume type (0=Relative, 1=Absolute)
        """
        self.volume_type = "relative" if index == 0 else "absolute"

        # Update DVH display
        self._update_dvh()

        # Update status
        self.status_label.setText(f"Volume type changed to {self.volume_type}")

    def _on_dose_type_changed(self, index: int):
        """
        Handle dose type change.

        Args:
            index: Index of selected dose type (0=Relative, 1=Absolute)
        """
        self.dose_type = "relative" if index == 0 else "absolute"

        # Update DVH display
        self._update_dvh()

        # Update status
        self.status_label.setText(f"Dose type changed to {self.dose_type}")

    def _on_export_dvh(self):
        """Handle export DVH button click."""
        if not self.plan or not self.dvh_data:
            return

        # Show export options
        menu = QMenu(self)

        # Export image action
        export_image_action = QAction("Export as Image", self)
        export_image_action.triggered.connect(self._export_dvh_image)
        menu.addAction(export_image_action)

        # Export CSV action
        export_csv_action = QAction("Export as CSV", self)
        export_csv_action.triggered.connect(self._export_dvh_csv)
        menu.addAction(export_csv_action)

        # Show menu at button
        menu.exec_(
            self.export_button.mapToGlobal(QPoint(0, self.export_button.height()))
        )

    def _export_dvh_image(self):
        """Export DVH as image."""
        if not self.plan:
            return

        # Get save filename
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export DVH as Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
        )

        if not filename:
            return

        try:
            # Save figure
            self.dvh_canvas.figure.savefig(filename, dpi=300, bbox_inches="tight")
            self.status_label.setText(f"DVH exported to {filename}")

            # Show success message
            QMessageBox.information(
                self, "Export Successful", f"DVH exported to {filename}"
            )

        except Exception as e:
            logger.error(f"Error exporting DVH image: {str(e)}")
            self.status_label.setText(f"Error exporting DVH image: {str(e)}")

            # Show error message
            QMessageBox.critical(
                self, "Export Error", f"Error exporting DVH image: {str(e)}"
            )

    def _export_dvh_csv(self):
        """Export DVH as CSV."""
        if not self.plan or not self.dvh_data:
            return

        # Get save filename
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export DVH as CSV", "", "CSV Files (*.csv);;All Files (*)"
        )

        if not filename:
            return

        try:
            # Get selected structures
            selected_structures = []
            structure_names = {}

            for i in range(self.structure_list.count()):
                item = self.structure_list.item(i)
                if item.isSelected():
                    structure = item.data(Qt.ItemDataRole.UserRole)
                    selected_structures.append(structure)
                    structure_names[structure.id] = structure.name

            # Open CSV file
            with open(filename, "w") as f:
                # Write header
                f.write("Dose")
                for structure in selected_structures:
                    if structure.id in self.dvh_data:
                        f.write(f",{structure_names[structure.id]}")
                f.write("\n")

                # Write data
                # This is a simplified example - replace with actual DVH data format
                dose_bins = range(0, 101)  # 0-100% in 1% increments

                for dose in dose_bins:
                    f.write(f"{dose}")
                    for structure in selected_structures:
                        if structure.id in self.dvh_data:
                            dvh = self.dvh_data[structure.id]

                            # Get volume at dose
                            if hasattr(dvh, "get_volume_at_dose"):
                                volume = dvh.get_volume_at_dose(dose)
                                f.write(f",{volume}")
                            else:
                                f.write(",0")
                    f.write("\n")

            self.status_label.setText(f"DVH exported to {filename}")

            # Show success message
            QMessageBox.information(
                self, "Export Successful", f"DVH exported to {filename}"
            )

        except Exception as e:
            logger.error(f"Error exporting DVH CSV: {str(e)}")
            self.status_label.setText(f"Error exporting DVH CSV: {str(e)}")

            # Show error message
            QMessageBox.critical(
                self, "Export Error", f"Error exporting DVH CSV: {str(e)}"
            )


class DoseDisplayWidget(QWidget):
    """
    Widget for displaying dose distribution overlaid on anatomical images.

    This widget displays a slice of patient anatomy with dose overlay,
    allowing for visualization of dose distribution relative to structures.
    """

    def __init__(self, parent=None):
        """Initialize the dose display widget."""
        super().__init__(parent)

        # Display data
        self.anatomy_data = None
        self.dose_data = None
        self.structure_overlays = {}
        self.current_background = None
        self.current_dose = None  # Add missing attribute initialization

        # Display properties
        self.window_width = 400
        self.window_level = 40
        self.colormap = "jet"  # For dose display
        self.dose_opacity = 0.6
        self.show_colorbar = True
        self.dose_range = [0, 100]  # % of prescription

        # Slice navigation
        self.current_slice = 0
        self.orientation = "axial"  # axial, sagittal, coronal

        # Initialize UI
        self._init_ui()

        # Create sample data for testing
        self._create_sample_data()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Image display area
        self.image_widget = QLabel()
        self.image_widget.setMinimumSize(300, 300)
        self.image_widget.setAlignment(Qt.AlignCenter)
        self.image_widget.setStyleSheet("background-color: black;")
        layout.addWidget(self.image_widget)

        # Controls for navigation and display settings
        controls_layout = QHBoxLayout()

        # Slice slider
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("Slice:"))

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(20)  # Will be updated with data
        self.slice_slider.setValue(10)
        self.slice_slider.valueChanged.connect(self.set_slice_data)
        slice_layout.addWidget(self.slice_slider)

        self.slice_label = QLabel("10/20")
        slice_layout.addWidget(self.slice_label)

        controls_layout.addLayout(slice_layout)

        # Window/level controls
        window_level_layout = QHBoxLayout()
        window_level_layout.addWidget(QLabel("W/L:"))

        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setMinimum(1)
        self.window_slider.setMaximum(4000)
        self.window_slider.setValue(self.window_width)
        self.window_slider.valueChanged.connect(self.set_window)
        window_level_layout.addWidget(self.window_slider)

        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setMinimum(-1000)
        self.level_slider.setMaximum(3000)
        self.level_slider.setValue(self.window_level)
        self.level_slider.valueChanged.connect(self.set_level)
        window_level_layout.addWidget(self.level_slider)

        controls_layout.addLayout(window_level_layout)

        # Orientation selector
        orientation_layout = QHBoxLayout()
        orientation_layout.addWidget(QLabel("View:"))

        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Axial", "Sagittal", "Coronal"])
        self.orientation_combo.currentTextChanged.connect(self._on_orientation_changed)
        orientation_layout.addWidget(self.orientation_combo)

        controls_layout.addLayout(orientation_layout)

        # Add dose display controls
        dose_layout = QVBoxLayout()
        dose_controls = QHBoxLayout()

        # Colormap selection
        dose_controls.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["Jet", "Hot", "Viridis", "Plasma"])
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        dose_controls.addWidget(self.colormap_combo)

        # Dose opacity slider
        dose_controls.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self.dose_opacity * 100))
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        dose_controls.addWidget(self.opacity_slider)

        dose_layout.addLayout(dose_controls)

        # Structure visibility
        structures_group = QGroupBox("Structures")
        self.structures_layout = QVBoxLayout(structures_group)
        self.structure_checkboxes = {}  # Will be populated with structure data

        dose_layout.addWidget(structures_group)

        # Add layouts to main layout
        layout.addLayout(controls_layout)
        layout.addLayout(dose_layout)

    def _create_sample_data(self):
        """Create sample data for testing."""
        try:
            # Create sample anatomy image (CT-like)
            size = (20, 256, 256)  # z, y, x
            self.anatomy_data = np.ones(size, dtype=np.float32) * 40  # Base CT value

            # Add some basic structures
            center_z, center_y, center_x = 10, 128, 128
            radius_y, radius_x = 100, 100

            # Create a phantom with body and target
            for z in range(size[0]):
                for y in range(size[1]):
                    for x in range(size[2]):
                        # Calculate distance from center
                        dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

                        # Body (water)
                        if dist < radius_y:
                            self.anatomy_data[z, y, x] = 0  # Water = 0 HU

                            # Add some bone structures
                            if 70 < dist < 90 and abs(x - center_x) > 50:
                                self.anatomy_data[z, y, x] = 700  # Bone ~ 700 HU

                            # Add a target structure
                            if dist < 30 and abs(z - center_z) < 5:
                                self.anatomy_data[z, y, x] = 10  # Soft tissue ~ 10 HU

            # Create sample dose data
            self.dose_data = np.zeros(size, dtype=np.float32)

            # Simple dose distribution - higher in the target
            for z in range(size[0]):
                for y in range(size[1]):
                    for x in range(size[2]):
                        # Calculate distance from center
                        dist = np.sqrt(
                            (y - center_y) ** 2
                            + (x - center_x) ** 2
                            + 2 * (z - center_z) ** 2
                        )

                        # Dose falls off with distance from target center
                        if dist < 100:  # Only inside "body"
                            falloff = 1.0 - (dist / 100.0) ** 1.5  # Dose falloff
                            self.dose_data[z, y, x] = max(
                                0, falloff * 100
                            )  # Max 100% dose

        except Exception as e:
            print(f"Error creating sample data: {e}")
            import traceback

            traceback.print_exc()

    def _add_sample_data(self):
        """Add sample data to the display."""
        if self.anatomy_data is not None and self.dose_data is not None:
            self.set_slice_data(10)  # Display a middle slice

    def set_slice_data(self, slice_index):
        """
        Set the current slice to display.

        Parameters
        ----------
        slice_index : int
            Index of the slice to display
        """
        if self.anatomy_data is None:
            return

        # Update current slice index
        self.current_slice = slice_index
        self.slice_slider.setValue(slice_index)
        self.slice_label.setText(f"{slice_index}/{self.anatomy_data.shape[0] - 1}")

        # Extract the current slice based on orientation
        if self.orientation == "axial":
            anatomy_slice = self.anatomy_data[slice_index, :, :]
            dose_slice = (
                self.dose_data[slice_index, :, :]
                if self.dose_data is not None
                else None
            )
        elif self.orientation == "sagittal":
            anatomy_slice = self.anatomy_data[:, :, slice_index]
            dose_slice = (
                self.dose_data[:, :, slice_index]
                if self.dose_data is not None
                else None
            )
        elif self.orientation == "coronal":
            anatomy_slice = self.anatomy_data[:, slice_index, :]
            dose_slice = (
                self.dose_data[:, slice_index, :]
                if self.dose_data is not None
                else None
            )

        # Display the slice
        self.set_background_data(anatomy_slice)
        if dose_slice is not None:
            self.set_dose_overlay(dose_slice)

        # Update the display
        self.update_display()

    def set_background_data(self, image_data):
        """
        Set background image data.

        Parameters
        ----------
        image_data : np.ndarray
            2D array of background image data
        """
        if image_data is None or image_data.size == 0:
            return

        # Store the data
        self.current_background = image_data

        # Update the display
        self.update_display()

    def set_dose_overlay(self, dose_data):
        """
        Set dose overlay data.

        Parameters
        ----------
        dose_data : np.ndarray
            2D array of dose data
        """
        if dose_data is None or dose_data.size == 0:
            return

        # Store the data
        self.current_dose = dose_data

        # Update the display
        self.update_display()

    def set_window(self, value):
        """Set the window width."""
        self.window_width = value
        self.update_display()

    def set_level(self, value):
        """Set the window level."""
        self.window_level = value
        self.update_display()

    def set_brightness(self, value):
        """Set the brightness (window level)."""
        self.window_level = value
        self.update_display()

    def set_contrast(self, value):
        """Set the contrast (window width)."""
        self.window_width = value
        self.update_display()

    def update_display(self):
        """Update the display with current data and settings."""
        if not hasattr(self, "current_background") or self.current_background is None:
            return

        try:
            # Apply window/level to anatomy image
            min_val = self.window_level - self.window_width / 2
            max_val = self.window_level + self.window_width / 2

            normalized = np.clip(self.current_background, min_val, max_val)
            normalized = (normalized - min_val) / (max_val - min_val) * 255
            display_data = normalized.astype(np.uint8)

            # Convert to RGB image
            rgb_image = np.stack([display_data, display_data, display_data], axis=2)

            # Add dose overlay if available
            if hasattr(self, "current_dose") and self.current_dose is not None:
                # Normalize dose to 0-1 range
                norm_dose = (
                    np.clip(self.current_dose, 0, self.dose_range[1])
                    / self.dose_range[1]
                )

                # Create colormap
                if self.colormap == "jet":
                    # Simple jet-like colormap (blue to red)
                    r = np.clip(np.interp(norm_dose, [0.5, 1.0], [0, 255]), 0, 255)
                    g = np.clip(np.interp(norm_dose, [0.25, 0.75], [0, 255]), 0, 255)
                    b = np.clip(np.interp(norm_dose, [0.0, 0.5], [255, 0]), 0, 255)
                elif self.colormap == "hot":
                    # Hot colormap (black -> red -> yellow -> white)
                    r = np.clip(np.interp(norm_dose, [0.0, 0.33], [0, 255]), 0, 255)
                    g = np.clip(np.interp(norm_dose, [0.33, 0.66], [0, 255]), 0, 255)
                    b = np.clip(np.interp(norm_dose, [0.66, 1.0], [0, 255]), 0, 255)
                elif self.colormap == "viridis":
                    # Simplified viridis (purple -> green -> yellow)
                    r = np.clip(
                        np.interp(norm_dose, [0.0, 0.5, 1.0], [68, 33, 253]), 0, 255
                    )
                    g = np.clip(
                        np.interp(norm_dose, [0.0, 0.5, 1.0], [1, 145, 231]), 0, 255
                    )
                    b = np.clip(
                        np.interp(norm_dose, [0.0, 0.5, 1.0], [84, 140, 37]), 0, 255
                    )
                elif self.colormap == "plasma":
                    # Simplified plasma (dark purple -> red -> yellow)
                    r = np.clip(
                        np.interp(norm_dose, [0.0, 0.5, 1.0], [13, 240, 252]), 0, 255
                    )
                    g = np.clip(
                        np.interp(norm_dose, [0.0, 0.5, 1.0], [8, 39, 234]), 0, 255
                    )
                    b = np.clip(
                        np.interp(norm_dose, [0.0, 0.5, 1.0], [135, 110, 38]), 0, 255
                    )
                else:
                    # Default to grayscale
                    r = g = b = norm_dose * 255

                # Create RGBA dose overlay
                dose_rgba = np.stack(
                    [r, g, b, norm_dose * 255 * self.dose_opacity], axis=2
                ).astype(np.uint8)

                # Resize dose overlay if needed
                if dose_rgba.shape[:2] != rgb_image.shape[:2]:
                    # Simple resizing - in practice would use proper interpolation
                    from scipy.ndimage import zoom

                    zoom_factors = (
                        rgb_image.shape[0] / dose_rgba.shape[0],
                        rgb_image.shape[1] / dose_rgba.shape[1],
                        1,
                    )
                    dose_rgba = zoom(dose_rgba, zoom_factors, order=1)

                # Alpha blend dose and anatomy
                alpha = dose_rgba[:, :, 3:4] / 255.0
                rgb_image = rgb_image * (1 - alpha) + dose_rgba[:, :, :3] * alpha

            # Add structure overlays
            if hasattr(self, "structure_overlays") and self.structure_overlays:
                # Get current slice and orientation
                slice_idx = self.current_slice

                # Create a mask for the current slice
                struct_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

                # Draw structures on the current slice
                for struct_id, struct_info in self.structure_overlays.items():
                    if not struct_info["visible"]:
                        continue

                    contours = struct_info["data"]
                    if contours is None:
                        continue

                    color = struct_info["color"]

                    # Extract contours for current slice based on orientation
                    slice_contours = []
                    if self.orientation == "axial" and slice_idx in contours:
                        slice_contours = contours[slice_idx]
                    elif self.orientation == "sagittal":
                        # Extract sagittal contours (may need additional processing)
                        for z, z_contours in contours.items():
                            for contour in z_contours:
                                # Filter points that lie on this sagittal slice
                                sagittal_points = [
                                    pt for pt in contour if int(pt[0]) == slice_idx
                                ]
                                if sagittal_points:
                                    slice_contours.append(sagittal_points)
                    elif self.orientation == "coronal":
                        # Extract coronal contours
                        for z, z_contours in contours.items():
                            for contour in z_contours:
                                # Filter points that lie on this coronal slice
                                coronal_points = [
                                    pt for pt in contour if int(pt[1]) == slice_idx
                                ]
                                if coronal_points:
                                    slice_contours.append(coronal_points)

                    # Draw contours on the mask
                    if slice_contours:
                        try:
                            if OPENCV_AVAILABLE:
                                for contour in slice_contours:
                                    # Convert contour to appropriate format for cv2
                                    if (
                                        len(contour) > 2
                                    ):  # Need at least 3 points for a contour
                                        contour_pts = np.array(contour).astype(np.int32)

                                        # Transform coordinates based on orientation
                                        if self.orientation == "axial":
                                            # Use y, x coords for axial view
                                            points = np.zeros(
                                                (len(contour), 2), dtype=np.int32
                                            )
                                            for i, pt in enumerate(contour):
                                                points[i, 0] = int(pt[2])  # x
                                                points[i, 1] = int(pt[1])  # y
                                        elif self.orientation == "sagittal":
                                            # Use z, y coords for sagittal view
                                            points = np.zeros(
                                                (len(contour), 2), dtype=np.int32
                                            )
                                            for i, pt in enumerate(contour):
                                                points[i, 0] = int(pt[1])  # y
                                                points[i, 1] = int(pt[0])  # z
                                        elif self.orientation == "coronal":
                                            # Use z, x coords for coronal view
                                            points = np.zeros(
                                                (len(contour), 2), dtype=np.int32
                                            )
                                            for i, pt in enumerate(contour):
                                                points[i, 0] = int(pt[2])  # x
                                                points[i, 1] = int(pt[0])  # z

                                        # Make sure points are within image bounds
                                        points[:, 0] = np.clip(
                                            points[:, 0], 0, struct_mask.shape[1] - 1
                                        )
                                        points[:, 1] = np.clip(
                                            points[:, 1], 0, struct_mask.shape[0] - 1
                                        )

                                        # Draw the contour
                                        cv2.drawContours(
                                            struct_mask, [points], -1, 255, 1
                                        )
                            else:
                                # Basic contour drawing without OpenCV
                                for contour in slice_contours:
                                    if len(contour) > 2:
                                        for i in range(len(contour)):
                                            p1 = contour[i]
                                            p2 = contour[
                                                (i + 1) % len(contour)
                                            ]  # Loop back to first point

                                            if self.orientation == "axial":
                                                x1, y1 = int(p1[2]), int(p1[1])
                                                x2, y2 = int(p2[2]), int(p2[1])
                                            elif self.orientation == "sagittal":
                                                x1, y1 = int(p1[1]), int(p1[0])
                                                x2, y2 = int(p2[1]), int(p2[0])
                                            elif self.orientation == "coronal":
                                                x1, y1 = int(p1[2]), int(p1[0])
                                                x2, y2 = int(p2[2]), int(p2[0])

                                            # Clip coordinates to image bounds
                                            x1 = max(
                                                0, min(x1, struct_mask.shape[1] - 1)
                                            )
                                            y1 = max(
                                                0, min(y1, struct_mask.shape[0] - 1)
                                            )
                                            x2 = max(
                                                0, min(x2, struct_mask.shape[1] - 1)
                                            )
                                            y2 = max(
                                                0, min(y2, struct_mask.shape[0] - 1)
                                            )

                                            # Simple line drawing using Bresenham's line algorithm
                                            # Draw a line between p1 and p2
                                            dx = abs(x2 - x1)
                                            dy = abs(y2 - y1)
                                            sx = 1 if x1 < x2 else -1
                                            sy = 1 if y1 < y2 else -1
                                            err = dx - dy

                                            while x1 != x2 or y1 != y2:
                                                struct_mask[y1, x1] = 255
                                                e2 = 2 * err
                                                if e2 > -dy:
                                                    err -= dy
                                                    x1 += sx
                                                if e2 < dx:
                                                    err += dx
                                                    y1 += sy

                                            struct_mask[y2, x2] = (
                                                255  # Ensure last point is drawn
                                            )
                        except Exception as e:
                            print(f"Error drawing contour: {e}")
                            import traceback

                            traceback.print_exc()

                # Overlay structure contours on the image
                for struct_id, struct_info in self.structure_overlays.items():
                    if not struct_info["visible"]:
                        continue

                    color = struct_info["color"]
                    if isinstance(color, tuple):
                        r, g, b = color[:3]
                    else:
                        r, g, b = 255, 0, 0  # Default to red

                    # Create colored mask for this structure
                    struct_overlay = np.zeros_like(rgb_image)
                    struct_overlay[struct_mask > 0] = [r, g, b]

                    # Add structure overlay to image
                    rgb_image = np.where(
                        np.expand_dims(struct_mask > 0, axis=2),
                        struct_overlay,
                        rgb_image,
                    )

            # Convert to QImage and display
            height, width = rgb_image.shape[:2]
            qimage = QImage(
                rgb_image.astype(np.uint8),
                width,
                height,
                width * 3,
                QImage.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(qimage)

            # Scale to fit widget while maintaining aspect ratio
            self.image_widget.setPixmap(
                pixmap.scaled(
                    self.image_widget.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        except Exception as e:
            print(f"Error updating display: {e}")
            import traceback

            traceback.print_exc()

    def set_plan(self, plan):
        """
        Set the current plan data for display.

        Parameters
        ----------
        plan : Plan
            Plan object containing image, dose, and structure data
        """
        if plan is None:
            return

        # Reset display
        self.anatomy_data = None
        self.dose_data = None
        self.structure_overlays = {}

        # Extract plan data
        try:
            # Extract anatomy data from plan
            if hasattr(plan, "get_image_data") and callable(plan.get_image_data):
                self.anatomy_data = plan.get_image_data()

            # Extract dose data from plan
            if hasattr(plan, "get_dose_data") and callable(plan.get_dose_data):
                self.dose_data = plan.get_dose_data()

            # Extract structure data from plan
            if hasattr(plan, "get_structures") and callable(plan.get_structures):
                structures = plan.get_structures()
                for structure in structures:
                    self.structure_overlays[structure.id] = {
                        "data": structure.get_contours()
                        if hasattr(structure, "get_contours")
                        else None,
                        "color": structure.color
                        if hasattr(structure, "color")
                        else (255, 0, 0),
                        "visible": True,
                    }

                # Update structure checkboxes in UI
                self._update_structure_display()

            # If we have data, update the display
            if self.anatomy_data is not None:
                # Update slice slider range
                max_slice = self.anatomy_data.shape[0] - 1
                self.slice_slider.setMaximum(max_slice)
                self.slice_slider.setValue(max_slice // 2)

                # Update the display
                self.set_slice_data(max_slice // 2)
            else:
                # If no plan data, use sample data
                self._create_sample_data()
                self._add_sample_data()

        except Exception as e:
            print(f"Error loading plan data: {e}")
            import traceback

            traceback.print_exc()

            # Use sample data as fallback
            self._create_sample_data()
            self._add_sample_data()

    def _update_structure_display(self):
        """Update the structure display list."""
        # Clear existing checkboxes
        while self.structures_layout.count():
            item = self.structures_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self.structure_checkboxes = {}

        # Add checkbox for each structure
        for struct_id, struct_info in self.structure_overlays.items():
            checkbox = QCheckBox(struct_id)
            checkbox.setChecked(struct_info["visible"])

            # Set color indicator
            color = struct_info["color"]
            if isinstance(color, tuple):
                r, g, b = color[:3]
                style = f"QCheckBox::indicator {{ background-color: rgb({r}, {g}, {b}); border: 1px solid black; }}"
                checkbox.setStyleSheet(style)

            # Connect signal
            checkbox.stateChanged.connect(
                lambda state, sid=struct_id: self._on_structure_visibility_changed(
                    sid, state
                )
            )

            self.structures_layout.addWidget(checkbox)
            self.structure_checkboxes[struct_id] = checkbox

    def _on_colormap_changed(self, colormap_name):
        """Handle colormap change."""
        self.colormap = colormap_name.lower()
        self.update_display()

    def _on_opacity_changed(self, value):
        """Handle opacity change."""
        self.dose_opacity = value / 100.0
        self.update_display()

    def _on_structure_visibility_changed(self, structure_id, state):
        """Handle structure visibility change."""
        if structure_id in self.structure_overlays:
            self.structure_overlays[structure_id]["visible"] = state == Qt.Checked
            self.update_display()

    def _on_orientation_changed(self, orientation):
        """Handle orientation change."""
        orientation = orientation.lower()
        if orientation in ["axial", "sagittal", "coronal"]:
            self.orientation = orientation

            # Update slice slider range based on orientation
            if self.anatomy_data is not None:
                if orientation == "axial":
                    max_slice = self.anatomy_data.shape[0] - 1
                elif orientation == "sagittal":
                    max_slice = self.anatomy_data.shape[2] - 1
                elif orientation == "coronal":
                    max_slice = self.anatomy_data.shape[1] - 1

                self.slice_slider.setMaximum(max_slice)
                self.slice_slider.setValue(max_slice // 2)

                # Update display
                self.set_slice_data(max_slice // 2)

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update_display()  # Refresh display when widget is resized
