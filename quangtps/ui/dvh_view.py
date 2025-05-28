#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DVH View for QuangTPS.

This module provides visualization of dose-volume histograms for
radiotherapy treatment plans, including metrics calculation and structure selection.
"""

import os
import logging
import numpy as np
import csv
from typing import Dict, List, Optional, Set, Any, Tuple

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QScrollArea,
    QGroupBox,
    QSplitter,
    QMessageBox,
    QSizePolicy,
)
from PyQt5.QtGui import QColor, QBrush, QFont

# Fix matplotlib imports
import matplotlib

matplotlib.use("Qt5Agg")
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
except ImportError:
    # Fallback for older versions of matplotlib
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
from matplotlib.figure import Figure

# Fix imports for QuangTPS modules - use conditional imports
try:
    from quangtps.common.signals import SignalManager
except ImportError:
    # Create a simple signal manager if the actual one isn't available
    class SignalManager:
        def __init__(self):
            self.plan_loaded = pyqtSignal(object)

        @classmethod
        def get_instance(cls):
            if not hasattr(cls, "_instance"):
                cls._instance = SignalManager()
            return cls._instance


try:
    from quangtps.common.config import Config
except ImportError:
    # Simple config placeholder
    class Config:
        @classmethod
        def get_instance(cls):
            if not hasattr(cls, "_instance"):
                cls._instance = Config()
            return cls._instance

        def get_value(self, key, default=None):
            return default


from quangtps.planning.treatment_plan import TreatmentPlan
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
from quangtps.evaluation.dvh.dvh_data import DVHData
from quangtps.evaluation.dvh.dvh_metrics import DVHMetrics

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DVH_TYPES = ["Cumulative", "Differential"]
DOSE_UNITS = ["Gy", "cGy", "%"]
VOLUME_UNITS = ["%", "cc"]


class StructureCheckbox(QWidget):
    """
    Widget for displaying a structure checkbox with structure color.
    Used in the structure selection panel.
    """

    def __init__(self, structure_name: str, color: Tuple[int, int, int], parent=None):
        """
        Initialize structure checkbox widget.

        Parameters
        ----------
        structure_name : str
            Name of the structure
        color : Tuple[int, int, int]
            RGB color tuple for the structure
        parent : QWidget, optional
            Parent widget, by default None
        """
        super().__init__(parent)

        self.structure_name = structure_name
        self.color = color

        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)

        # Color indicator
        self.color_frame = QFrame()
        self.color_frame.setFixedSize(16, 16)
        self.color_frame.setStyleSheet(
            f"background-color: rgb{color}; border: 1px solid gray;"
        )
        layout.addWidget(self.color_frame)

        # Checkbox
        self.checkbox = QCheckBox(structure_name)
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox)

        self.setLayout(layout)

    def is_checked(self) -> bool:
        """Return the checked state of the checkbox."""
        return self.checkbox.isChecked()

    def set_checked(self, checked: bool):
        """Set the checked state of the checkbox."""
        self.checkbox.setChecked(checked)


class DVHView(QWidget):
    """
    Widget for displaying dose-volume histograms and related analysis.

    DVHView provides tools for visualizing and analyzing DVH data
    for treatment plans, including metrics tables and structure selection.
    """

    signal_plan_changed = pyqtSignal(TreatmentPlan)

    def __init__(self, parent=None):
        """
        Initialize DVH view widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None
        """
        super().__init__(parent)

        # Initialize variables
        self.plan = None
        self.dvh_data = {}
        self.structure_checkboxes = {}
        self.calculator = DVHCalculator()
        self.figure = None
        self.canvas = None
        self.axes = None

        # Structure colors (in case structure doesn't define its own color)
        self.default_colors = [
            (255, 0, 0),  # Red
            (0, 0, 255),  # Blue
            (0, 255, 0),  # Green
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (255, 192, 203),  # Pink
            (255, 255, 0),  # Yellow
            (165, 42, 42),  # Brown
            (0, 255, 255),  # Cyan
        ]

        # Initialize UI
        self._init_ui()

        # Connect to the signal manager
        try:
            SignalManager().plan_loaded.connect(self.set_plan)
        except (AttributeError, TypeError):
            logger.warning("Could not connect to SignalManager plan_loaded signal")

    def _init_ui(self):
        """Initialize user interface."""
        main_layout = QVBoxLayout()

        # Top toolbar with controls
        toolbar_layout = QHBoxLayout()

        # DVH type selector
        toolbar_layout.addWidget(QLabel("DVH Type:"))
        self.dvh_type_combo = QComboBox()
        self.dvh_type_combo.addItems(DVH_TYPES)
        self.dvh_type_combo.setCurrentIndex(0)  # Default to Cumulative
        self.dvh_type_combo.currentIndexChanged.connect(self._update_plot)
        toolbar_layout.addWidget(self.dvh_type_combo)

        # Dose unit selector
        toolbar_layout.addWidget(QLabel("Dose:"))
        self.dose_unit_combo = QComboBox()
        self.dose_unit_combo.addItems(DOSE_UNITS)
        self.dose_unit_combo.setCurrentIndex(0)  # Default to Gy
        self.dose_unit_combo.currentIndexChanged.connect(self._update_plot)
        toolbar_layout.addWidget(self.dose_unit_combo)

        # Volume unit selector
        toolbar_layout.addWidget(QLabel("Volume:"))
        self.volume_unit_combo = QComboBox()
        self.volume_unit_combo.addItems(VOLUME_UNITS)
        self.volume_unit_combo.setCurrentIndex(0)  # Default to %
        self.volume_unit_combo.currentIndexChanged.connect(self._update_plot)
        toolbar_layout.addWidget(self.volume_unit_combo)

        # Export buttons
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self._export_to_csv)
        toolbar_layout.addWidget(self.export_btn)

        self.save_image_btn = QPushButton("Save Image")
        self.save_image_btn.clicked.connect(self._save_image)
        toolbar_layout.addWidget(self.save_image_btn)

        toolbar_layout.addStretch()

        main_layout.addLayout(toolbar_layout)

        # Main content area with plot and structure selection
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Structure selection
        structures_group = QGroupBox("Structures")
        structures_layout = QVBoxLayout()

        # Structure selection controls
        selection_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all_structures)
        selection_layout.addWidget(select_all_btn)

        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self._select_no_structures)
        selection_layout.addWidget(select_none_btn)
        structures_layout.addLayout(selection_layout)

        # Structure checkboxes in a scrollable area
        self.structures_scroll = QScrollArea()
        self.structures_scroll.setWidgetResizable(True)
        self.structures_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.structures_content = QWidget()
        self.structures_content_layout = QVBoxLayout(self.structures_content)
        self.structures_content_layout.setAlignment(Qt.AlignTop)
        self.structures_scroll.setWidget(self.structures_content)
        structures_layout.addWidget(self.structures_scroll)

        structures_group.setLayout(structures_layout)
        splitter.addWidget(structures_group)

        # Right panel: DVH plot and metrics
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Create matplotlib figure for DVH plot
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel("Dose (Gy)")
        self.axes.set_ylabel("Volume (%)")
        self.axes.set_title("Dose-Volume Histogram")
        self.axes.grid(True)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setMinimumHeight(150)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.metrics_table)

        splitter.addWidget(right_panel)
        splitter.setSizes([200, 600])  # Set initial sizes

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def set_plan(self, plan: TreatmentPlan):
        """
        Set the current treatment plan and update the DVH view.

        Parameters
        ----------
        plan : TreatmentPlan
            Treatment plan to display
        """
        if plan is None:
            logger.warning("Attempted to set None as the current plan")
            return

        self.plan = plan
        logger.info(f"Setting plan in DVH view: {plan.name}")

        # Check if plan has a dose grid
        if not hasattr(plan, "dose") or plan.dose is None:
            logger.warning(f"Plan {plan.name} has no dose grid, cannot calculate DVH")
            self._show_placeholder_dvh()
            return

        # Check if plan has a structure set
        if not hasattr(plan, "structure_set") or plan.structure_set is None:
            logger.warning(
                f"Plan {plan.name} has no structure set, cannot calculate DVH"
            )
            self._show_placeholder_dvh()
            return

        # Calculate DVH for each structure
        self._calculate_dvh()
        self._update_structure_list()
        self._update_plot()
        self._update_metrics_table()

        self.signal_plan_changed.emit(plan)

    def _clear_structures(self):
        """Clear the structure list in the UI."""
        self.structure_checkboxes = {}
        # Clear the layout
        while self.structures_content_layout.count():
            item = self.structures_content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _update_structure_list(self):
        """Update the list of structures in the UI."""
        self._clear_structures()

        if (
            not self.plan
            or not hasattr(self.plan, "structure_set")
            or not self.plan.structure_set
        ):
            logger.warning("No structures available to display")
            return

        # Add checkboxes for each structure with DVH data
        color_index = 0
        for struct_name, dvh in self.dvh_data.items():
            if dvh is None:
                continue

            # Get structure color if available, otherwise use default
            color = self.default_colors[color_index % len(self.default_colors)]
            color_index += 1

            # Try to get color from structure
            try:
                struct = next(
                    (
                        s
                        for s in self.plan.structure_set.structures
                        if s.name == struct_name
                    ),
                    None,
                )
                if struct and hasattr(struct, "color") and struct.color:
                    color = struct.color
            except (AttributeError, IndexError):
                pass

            # Create checkbox widget
            checkbox = StructureCheckbox(struct_name, color)
            checkbox.checkbox.stateChanged.connect(self._update_plot)
            self.structure_checkboxes[struct_name] = checkbox
            self.structures_content_layout.addWidget(checkbox)

    def _update_plot(self):
        """Update the DVH plot with current settings."""
        if not self.plan or not self.dvh_data:
            self._show_placeholder_dvh()
            return

        # Clear the plot
        self.axes.clear()

        # Get current settings
        is_cumulative = self.dvh_type_combo.currentText() == "Cumulative"
        dose_unit = self.dose_unit_combo.currentText()
        volume_unit = self.volume_unit_combo.currentText()

        # Dose conversion factors
        dose_factor = 1.0
        if dose_unit == "cGy":
            dose_factor = 100.0
        elif dose_unit == "%":
            # Use prescription dose as reference if available
            prescription_dose = 1.0
            if hasattr(self.plan, "prescription") and self.plan.prescription:
                try:
                    prescription_dose = float(self.plan.prescription.dose)
                except (AttributeError, ValueError):
                    pass
            dose_factor = 100.0 / prescription_dose

        # Plot DVH for each selected structure
        max_dose = 0.0
        for struct_name, checkbox in self.structure_checkboxes.items():
            if not checkbox.is_checked() or struct_name not in self.dvh_data:
                continue

            dvh_data = self.dvh_data[struct_name]
            if dvh_data is None:
                continue

            # Get dose and volume data
            dose_bins = dvh_data.dose_bins * dose_factor
            volume_bins = dvh_data.volume_bins

            # Convert volume to requested units
            if volume_unit == "cc" and hasattr(dvh_data, "structure_volume"):
                volume_bins = volume_bins * dvh_data.structure_volume / 100.0

            # If the DVH type doesn't match the requested type, convert it
            if is_cumulative != dvh_data.is_cumulative:
                if is_cumulative:
                    # Convert differential to cumulative
                    volume_bins = np.array(
                        [np.sum(volume_bins[i:]) for i in range(len(volume_bins))]
                    )
                else:
                    # Convert cumulative to differential (approximate by taking differences)
                    volume_bins = np.diff(
                        np.append(volume_bins, [0]), prepend=[volume_bins[0]]
                    )

            # Plot the data
            rgb_color = np.array(checkbox.color) / 255.0
            label = struct_name

            self.axes.plot(
                dose_bins, volume_bins, label=label, color=rgb_color, linewidth=2
            )

            # Track maximum dose for axis scaling
            max_dose = max(max_dose, np.max(dose_bins))

        # Set up plot labels and legend
        self.axes.set_xlabel(f"Dose ({dose_unit})")
        self.axes.set_ylabel(f"Volume ({volume_unit})")
        self.axes.set_title(
            f"{'Cumulative' if is_cumulative else 'Differential'} Dose-Volume Histogram"
        )

        # Set plot limits
        if is_cumulative:
            self.axes.set_ylim(0, 105 if volume_unit == "%" else None)
        self.axes.set_xlim(0, max_dose * 1.05)

        # Add grid and legend
        self.axes.grid(True)
        if self.axes.get_legend_handles_labels()[
            0
        ]:  # Only add legend if there are items
            self.axes.legend(loc="best")

        # Refresh canvas
        self.canvas.draw()

    def _update_metrics_table(self):
        """Update the metrics table with current DVH data."""
        if not self.plan or not self.dvh_data:
            self.metrics_table.setRowCount(0)
            self.metrics_table.setColumnCount(0)
            return

        # Set up table columns
        metrics = [
            "Structure",
            "Volume (cc)",
            "Min (Gy)",
            "Mean (Gy)",
            "Max (Gy)",
            "D95%",
            "D90%",
            "D50%",
            "V95%",
            "V90%",
            "V50%",
        ]

        self.metrics_table.setColumnCount(len(metrics))
        self.metrics_table.setHorizontalHeaderLabels(metrics)

        # Count selected structures
        selected_structures = [
            name
            for name, checkbox in self.structure_checkboxes.items()
            if checkbox.is_checked()
        ]

        self.metrics_table.setRowCount(len(selected_structures))

        # Populate the table
        row_index = 0
        for struct_name in selected_structures:
            if struct_name not in self.dvh_data:
                continue

            dvh_data = self.dvh_data[struct_name]
            if dvh_data is None:
                continue

            # Get prescription dose if available
            prescription_dose = 0.0
            if hasattr(self.plan, "prescription") and self.plan.prescription:
                try:
                    prescription_dose = float(self.plan.prescription.dose)
                except (AttributeError, ValueError):
                    pass

            # Structure name
            self.metrics_table.setItem(row_index, 0, QTableWidgetItem(struct_name))

            # Volume
            vol_item = QTableWidgetItem(f"{dvh_data.structure_volume:.2f}")
            vol_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(row_index, 1, vol_item)

            # Min dose
            min_item = QTableWidgetItem(f"{dvh_data.min_dose:.2f}")
            min_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(row_index, 2, min_item)

            # Mean dose
            mean_item = QTableWidgetItem(f"{dvh_data.mean_dose:.2f}")
            mean_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(row_index, 3, mean_item)

            # Max dose
            max_item = QTableWidgetItem(f"{dvh_data.max_dose:.2f}")
            max_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(row_index, 4, max_item)

            # D95%
            d95_item = QTableWidgetItem(f"{dvh_data.get_dx(95):.2f}")
            d95_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(row_index, 5, d95_item)

            # D90%
            d90_item = QTableWidgetItem(f"{dvh_data.get_dx(90):.2f}")
            d90_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(row_index, 6, d90_item)

            # D50%
            d50_item = QTableWidgetItem(f"{dvh_data.get_dx(50):.2f}")
            d50_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(row_index, 7, d50_item)

            # V95%
            v95_dose = (
                prescription_dose * 0.95
                if prescription_dose > 0
                else dvh_data.max_dose * 0.95
            )
            v95_item = QTableWidgetItem(f"{dvh_data.get_vx(v95_dose):.2f}")
            v95_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(row_index, 8, v95_item)

            # V90%
            v90_dose = (
                prescription_dose * 0.90
                if prescription_dose > 0
                else dvh_data.max_dose * 0.90
            )
            v90_item = QTableWidgetItem(f"{dvh_data.get_vx(v90_dose):.2f}")
            v90_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(row_index, 9, v90_item)

            # V50%
            v50_dose = (
                prescription_dose * 0.50
                if prescription_dose > 0
                else dvh_data.max_dose * 0.50
            )
            v50_item = QTableWidgetItem(f"{dvh_data.get_vx(v50_dose):.2f}")
            v50_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(row_index, 10, v50_item)

            # Apply structure color to name cell
            if struct_name in self.structure_checkboxes:
                rgb_color = self.structure_checkboxes[struct_name].color
                color = QColor(*rgb_color)
                color.setAlpha(100)  # Semi-transparent
                self.metrics_table.item(row_index, 0).setBackground(QBrush(color))

            row_index += 1

    def _select_all_structures(self):
        """Select all structures in the list."""
        for checkbox in self.structure_checkboxes.values():
            checkbox.set_checked(True)
        self._update_plot()

    def _select_no_structures(self):
        """Deselect all structures in the list."""
        for checkbox in self.structure_checkboxes.values():
            checkbox.set_checked(False)
        self._update_plot()

    def _export_to_csv(self):
        """Export DVH data to CSV file."""
        if not self.plan or not self.dvh_data:
            QMessageBox.warning(
                self, "Export Failed", "No DVH data available to export."
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save DVH Data", "", "CSV Files (*.csv);;All Files (*)"
        )

        if not filename:
            return

        try:
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                header = ["Dose (Gy)"]
                for struct_name, checkbox in self.structure_checkboxes.items():
                    if checkbox.is_checked() and struct_name in self.dvh_data:
                        header.append(f"{struct_name} (Volume %)")
                writer.writerow(header)

                # Get maximum number of dose bins
                max_bins = 0
                for struct_name, checkbox in self.structure_checkboxes.items():
                    if checkbox.is_checked() and struct_name in self.dvh_data:
                        dvh_data = self.dvh_data[struct_name]
                        if dvh_data is not None:
                            max_bins = max(max_bins, len(dvh_data.dose_bins))

                # Write data rows
                for i in range(max_bins):
                    row = []

                    # First column is dose bin
                    for struct_name, checkbox in self.structure_checkboxes.items():
                        if checkbox.is_checked() and struct_name in self.dvh_data:
                            dvh_data = self.dvh_data[struct_name]
                            if dvh_data is not None and i < len(dvh_data.dose_bins):
                                row.append(dvh_data.dose_bins[i])
                                break

                    # Skip if no data for this row
                    if not row:
                        continue

                    # Add volume data for each structure
                    for struct_name, checkbox in self.structure_checkboxes.items():
                        if checkbox.is_checked() and struct_name in self.dvh_data:
                            dvh_data = self.dvh_data[struct_name]
                            if dvh_data is not None and i < len(dvh_data.volume_bins):
                                row.append(dvh_data.volume_bins[i])
                            else:
                                row.append("")

                    writer.writerow(row)

            QMessageBox.information(
                self, "Export Successful", f"DVH data exported to {filename}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Failed", f"Failed to export DVH data: {str(e)}"
            )
            logger.error(f"DVH export error: {str(e)}")

    def _save_image(self):
        """Save the DVH plot as an image file."""
        if not self.figure:
            QMessageBox.warning(self, "Save Failed", "No DVH plot available to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save DVH Plot", "", "PNG Files (*.png);;All Files (*)"
        )

        if not filename:
            return

        try:
            self.figure.savefig(filename, dpi=300, bbox_inches="tight")
            QMessageBox.information(
                self, "Save Successful", f"DVH plot saved to {filename}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Save Failed", f"Failed to save DVH plot: {str(e)}"
            )
            logger.error(f"DVH image save error: {str(e)}")

    def _calculate_dvh(self):
        """Calculate DVH for each structure in the plan."""
        if not self.plan or not hasattr(self.plan, "dose") or not self.plan.dose:
            logger.warning("Cannot calculate DVH: No dose grid available")
            return

        if not hasattr(self.plan, "structure_set") or not self.plan.structure_set:
            logger.warning("Cannot calculate DVH: No structure set available")
            return

        self.dvh_data = {}

        try:
            # Get dose grid as SimpleITK image
            dose_image = self.plan.dose.get_sitk_image()

            # Calculate DVH for each structure
            for structure in self.plan.structure_set.structures:
                try:
                    # Skip if structure has no mask
                    if not hasattr(structure, "mask") or structure.mask is None:
                        logger.warning(
                            f"Structure {structure.name} has no mask, skipping DVH calculation"
                        )
                        continue

                    # Get structure mask as SimpleITK image
                    structure_mask = structure.get_sitk_mask()

                    # Calculate DVH
                    dvh_data = self.calculator.calculate_dvh_data(
                        dose_image, structure_mask, structure.name, cumulative=True
                    )

                    self.dvh_data[structure.name] = dvh_data

                except Exception as e:
                    logger.error(
                        f"Error calculating DVH for structure {structure.name}: {str(e)}"
                    )

        except Exception as e:
            logger.error(f"Error calculating DVH: {str(e)}")

    def _show_placeholder_dvh(self):
        """Show a placeholder message when no DVH data is available."""
        self.axes.clear()
        self.axes.text(
            0.5,
            0.5,
            "No DVH data available.\nCalculate dose or load a plan with dose.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.axes.transAxes,
        )
        self.axes.set_xlabel("Dose (Gy)")
        self.axes.set_ylabel("Volume (%)")
        self.axes.set_title("Dose-Volume Histogram")
        self.canvas.draw()

        # Clear metrics table
        self.metrics_table.setRowCount(0)
        self.metrics_table.setColumnCount(0)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    window = DVHView()
    window.setWindowTitle("DVH View")
    window.setGeometry(100, 100, 1000, 600)
    window.show()

    sys.exit(app.exec_())
