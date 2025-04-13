"""
Metrics Table Widget

This module provides a widget for displaying dose metrics for structures
in a radiotherapy treatment plan.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
import logging

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QLabel, QComboBox, QPushButton, QHBoxLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush, QFont

from quangtps.core.plan import Plan
from quangtps.core.structures import Structure
from quangtps.evaluation.dvh.dvh_data import DVHData
from quangtps.evaluation.metrics import calculate_d_metric, calculate_v_metric
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class MetricsTable(QWidget):
    """Widget for displaying dose metrics for structures."""
    
    # Default metrics to display
    DEFAULT_D_METRICS = [95, 98, 50, 2]  # D95, D98, D50, D2
    DEFAULT_V_METRICS = [95, 100, 105]   # V95, V100, V105
    
    def __init__(self, parent=None):
        """
        Initialize the metrics table widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Initialize state
        self.current_plan = None
        self.selected_structures = set()
        self.reference_dose = 1.0  # Reference dose in Gy for normalization
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Add controls for customizing metrics
        metrics_label = QLabel("Metrics:")
        controls_layout.addWidget(metrics_label)
        
        # D metrics selection
        self.d_metrics_combo = QComboBox()
        self.d_metrics_combo.setEditable(True)
        self.d_metrics_combo.setToolTip("Enter D metrics to display (comma-separated)")
        self.d_metrics_combo.addItem("D95, D98, D50, D2")
        self.d_metrics_combo.addItem("D90, D95, D50, D5")
        self.d_metrics_combo.addItem("D99, D95, D50, D1")
        controls_layout.addWidget(self.d_metrics_combo)
        
        # V metrics selection
        self.v_metrics_combo = QComboBox()
        self.v_metrics_combo.setEditable(True)
        self.v_metrics_combo.setToolTip("Enter V metrics to display (comma-separated)")
        self.v_metrics_combo.addItem("V95, V100, V105")
        self.v_metrics_combo.addItem("V90, V95, V100")
        self.v_metrics_combo.addItem("V80, V90, V100, V110")
        controls_layout.addWidget(self.v_metrics_combo)
        
        # Apply button
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply_metrics)
        controls_layout.addWidget(self.apply_btn)
        
        # Reference dose selection
        dose_layout = QHBoxLayout()
        
        dose_label = QLabel("Reference Dose (Gy):")
        dose_layout.addWidget(dose_label)
        
        self.dose_combo = QComboBox()
        self.dose_combo.setEditable(True)
        self.dose_combo.addItems(["1.0", "1.8", "2.0", "5.0", "10.0"])
        dose_layout.addWidget(self.dose_combo)
        
        self.dose_apply_btn = QPushButton("Set")
        self.dose_apply_btn.clicked.connect(self._on_set_reference_dose)
        dose_layout.addWidget(self.dose_apply_btn)
        
        dose_layout.addStretch()
        
        # Add controls to main layout
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(dose_layout)
        
        # Create table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(0)
        self.metrics_table.setRowCount(0)
        self.metrics_table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.metrics_table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        main_layout.addWidget(self.metrics_table)
        
        # Initialize with default metrics
        self._init_metrics_display()
    
    def _init_metrics_display(self):
        """Initialize the metrics display with default values."""
        self.d_metrics = self.DEFAULT_D_METRICS.copy()
        self.v_metrics = self.DEFAULT_V_METRICS.copy()
        
        # Set up table columns
        self._setup_table_columns()
    
    def _setup_table_columns(self):
        """Set up the table columns based on current metrics."""
        # Clear table
        self.metrics_table.clear()
        self.metrics_table.setRowCount(0)
        
        # Calculate total columns needed
        d_metric_count = len(self.d_metrics)
        v_metric_count = len(self.v_metrics)
        total_columns = 1 + d_metric_count + v_metric_count  # Name column + D metrics + V metrics
        
        self.metrics_table.setColumnCount(total_columns)
        
        # Set headers
        headers = ["Structure"]
        
        # Add D metric headers
        for d_value in self.d_metrics:
            headers.append(f"D{d_value} (Gy)")
        
        # Add V metric headers
        for v_value in self.v_metrics:
            headers.append(f"V{v_value} (%)")
        
        self.metrics_table.setHorizontalHeaderLabels(headers)
    
    def clear(self):
        """Clear all data from the table."""
        self.metrics_table.setRowCount(0)
        self.selected_structures.clear()
        self.current_plan = None
    
    def add_plan(self, plan: Plan, set_as_current: bool = False):
        """
        Add a plan for metrics calculation.
        
        Args:
            plan: The plan to add
            set_as_current: Whether to set this as the current plan
        """
        if set_as_current:
            self.current_plan = plan
        
        if not plan:
            return
            
        if not plan.has_dose():
            logger.warning(f"Plan {plan.name} has no dose data")
            return
        
        # Set reference dose if not already set
        if self.reference_dose <= 0 and plan.prescription_dose > 0:
            self.reference_dose = plan.prescription_dose
            # Update dose display
            self.dose_combo.setCurrentText(f"{self.reference_dose:.1f}")
    
    def add_structure(self, structure_id: str):
        """
        Add a structure to the metrics table.
        
        Args:
            structure_id: ID of the structure to add
        """
        if not self.current_plan:
            logger.warning("Cannot add structure: No current plan")
            return
            
        # Check if structure exists in plan
        structure = self.current_plan.structure_set.get_structure(structure_id)
        if not structure:
            logger.warning(f"Structure {structure_id} not found in plan {self.current_plan.name}")
            return
        
        # Add to selected structures
        self.selected_structures.add(structure_id)
    
    def refresh(self):
        """Refresh the metrics display."""
        if not self.current_plan or not self.selected_structures:
            return
            
        # Clear table content but keep headers
        while self.metrics_table.rowCount() > 0:
            self.metrics_table.removeRow(0)
        
        # Add rows for each structure
        for structure_id in self.selected_structures:
            structure = self.current_plan.structure_set.get_structure(structure_id)
            if not structure:
                continue
                
            # Get DVH data
            dvh_data = self.current_plan.get_dvh_data(structure_id)
            if not dvh_data:
                logger.warning(f"No DVH data for structure {structure.name}")
                continue
            
            # Add row for this structure
            self._add_structure_row(structure, dvh_data)
    
    def _add_structure_row(self, structure: Structure, dvh_data: DVHData):
        """
        Add a row for the given structure.
        
        Args:
            structure: The structure
            dvh_data: DVH data for the structure
        """
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        
        # Add structure name
        name_item = QTableWidgetItem(structure.name)
        name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.metrics_table.setItem(row, 0, name_item)
        
        # Calculate and add D metrics
        col = 1
        for d_value in self.d_metrics:
            try:
                # Calculate Dx in absolute dose (Gy)
                d_metric = calculate_d_metric(dvh_data, d_value)
                
                # Add to table
                value_item = QTableWidgetItem(f"{d_metric:.2f}")
                value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.metrics_table.setItem(row, col, value_item)
                
                # Highlight targets for important metrics
                if structure.is_target() and d_value in [95, 98, 99]:
                    # Check prescription coverage
                    if self.reference_dose > 0:
                        coverage_ratio = d_metric / self.reference_dose
                        if coverage_ratio >= 1.0:
                            # Good coverage
                            value_item.setBackground(QBrush(QColor(200, 255, 200)))  # Light green
                        elif coverage_ratio >= 0.95:
                            # Acceptable coverage
                            value_item.setBackground(QBrush(QColor(255, 255, 200)))  # Light yellow
                        else:
                            # Poor coverage
                            value_item.setBackground(QBrush(QColor(255, 200, 200)))  # Light red
            except Exception as e:
                logger.error(f"Error calculating D{d_value} for {structure.name}: {str(e)}")
                value_item = QTableWidgetItem("N/A")
                value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.metrics_table.setItem(row, col, value_item)
            
            col += 1
        
        # Calculate and add V metrics
        for v_value in self.v_metrics:
            try:
                # Calculate Vx as percentage of structure volume
                if self.reference_dose > 0:
                    # Convert percentage to absolute dose
                    dose_value = v_value / 100.0 * self.reference_dose
                    v_metric = calculate_v_metric(dvh_data, dose_value)
                    
                    # Add to table
                    value_item = QTableWidgetItem(f"{v_metric:.1f}")
                    value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.metrics_table.setItem(row, col, value_item)
                    
                    # Highlight targets for important metrics
                    if structure.is_target() and v_value in [95, 100]:
                        if v_metric >= 98.0:
                            # Good coverage
                            value_item.setBackground(QBrush(QColor(200, 255, 200)))  # Light green
                        elif v_metric >= 95.0:
                            # Acceptable coverage
                            value_item.setBackground(QBrush(QColor(255, 255, 200)))  # Light yellow
                        else:
                            # Poor coverage
                            value_item.setBackground(QBrush(QColor(255, 200, 200)))  # Light red
                else:
                    value_item = QTableWidgetItem("N/A")
                    value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.metrics_table.setItem(row, col, value_item)
            except Exception as e:
                logger.error(f"Error calculating V{v_value} for {structure.name}: {str(e)}")
                value_item = QTableWidgetItem("N/A")
                value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.metrics_table.setItem(row, col, value_item)
            
            col += 1
    
    def _on_apply_metrics(self):
        """Apply the metrics selection."""
        try:
            # Parse D metrics
            d_metrics_text = self.d_metrics_combo.currentText()
            d_metrics_values = [int(d.strip().replace('D', '')) for d in d_metrics_text.split(',') if d.strip()]
            
            # Parse V metrics
            v_metrics_text = self.v_metrics_combo.currentText()
            v_metrics_values = [int(v.strip().replace('V', '')) for v in v_metrics_text.split(',') if v.strip()]
            
            # Update metrics
            self.d_metrics = d_metrics_values
            self.v_metrics = v_metrics_values
            
            # Update table
            self._setup_table_columns()
            self.refresh()
            
        except ValueError as e:
            logger.error(f"Error parsing metrics: {str(e)}")
    
    def _on_set_reference_dose(self):
        """Set the reference dose for normalization."""
        try:
            dose_text = self.dose_combo.currentText()
            dose_value = float(dose_text)
            
            if dose_value <= 0:
                logger.warning(f"Invalid reference dose: {dose_value}")
                return
                
            self.reference_dose = dose_value
            
            # Refresh display
            self.refresh()
            
        except ValueError as e:
            logger.error(f"Error parsing reference dose: {str(e)}") 