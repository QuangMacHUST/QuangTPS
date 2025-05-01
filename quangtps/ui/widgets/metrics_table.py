#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics Table Widget for QuangTPS

This module provides a table widget for displaying radiotherapy plan metrics
such as min/max/mean dose and DVH metrics.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any

from PyQt5.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class MetricsTable(QTableWidget):
    """
    Table for displaying DVH metrics for each structure
    """

    def __init__(self, parent=None):
        """Initialize the metrics table widget"""
        super().__init__(parent)

        # Set up the table
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(
            ["Structure", "Min Dose", "Max Dose", "Mean Dose", "D95", "V20"]
        )

        # Configure the header
        header = self.horizontalHeader()
        if header:
            header.setSectionResizeMode(QHeaderView.Stretch)

        # Set selection behavior
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)

        # Store metrics data
        self._metrics_data = {}

    def update_metrics(self, metrics_data: Dict[str, Dict[str, Any]]):
        """
        Update the table with new metrics data

        Parameters
        ----------
        metrics_data : dict
            Dictionary mapping structure names to metrics dictionaries
        """
        self.setRowCount(0)  # Clear existing rows
        self._metrics_data = metrics_data.copy() if metrics_data else {}

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

    def highlight_structure(self, structure_name: str, highlight: bool = True):
        """
        Highlight a specific structure in the table

        Parameters
        ----------
        structure_name : str
            Name of the structure to highlight
        highlight : bool
            Whether to highlight (True) or remove highlight (False)
        """
        # Find the row with this structure
        for row in range(self.rowCount()):
            name_item = self.item(row, 0)
            if name_item and name_item.text() == structure_name:
                # Apply or remove highlight
                color = QColor(230, 240, 255) if highlight else None
                for col in range(self.columnCount()):
                    item = self.item(row, col)
                    if item:
                        if highlight:
                            item.setBackground(QBrush(color))
                        else:
                            item.setBackground(QBrush())
                break

    def get_structure_names(self) -> List[str]:
        """
        Get the list of structure names in the table

        Returns
        -------
        List[str]
            List of structure names
        """
        names = []
        for row in range(self.rowCount()):
            name_item = self.item(row, 0)
            if name_item:
                names.append(name_item.text())
        return names

    def clear(self):
        """Clear the table contents"""
        self.setRowCount(0)
        self._metrics_data = {}
