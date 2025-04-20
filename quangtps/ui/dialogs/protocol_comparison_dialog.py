#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Protocol Comparison Dialog

This module provides a dialog for comparing multiple clinical protocols
against a treatment plan or multiple plans against a single protocol.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QMessageBox,
    QSplitter,
    QFrame,
    QFileDialog,
    QScrollArea,
    QTabWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QIcon, QFont, QPixmap

from quangtps.core.plan import Plan
from quangtps.evaluation.clinical_goals import (
    ClinicalGoal,
    GoalResult,
    GoalType,
    GoalOperator,
    GoalPriority,
)
from quangtps.evaluation.clinical_protocols import ClinicalProtocol
from quangtps.evaluation.protocol_manager import ProtocolManager
from quangtps.common.paths import get_icon_path
from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class ProtocolComparisonDialog(QDialog):
    """
    A dialog for comparing multiple clinical protocols against a treatment plan
    or multiple plans against a single protocol.
    """

    def __init__(self, plan=None, parent=None):
        """
        Initialize the protocol comparison dialog.

        Parameters
        ----------
        plan : Plan, optional
            Initial plan to evaluate
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self.setWindowTitle("Protocol Comparison")
        self.resize(900, 700)

        # Initialize data
        self.plans: List[Plan] = []
        if plan:
            self.plans.append(plan)

        self.protocol_manager = ProtocolManager()
        self.selected_protocols: List[str] = []
        self.comparison_mode = "protocol"  # "protocol" or "plan"

        # Initialize UI
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)

        # Mode selection
        mode_group = QGroupBox("Comparison Mode")
        mode_layout = QHBoxLayout(mode_group)

        self.protocol_mode_radio = QRadioButton("Compare Protocols (single plan)")
        self.plan_mode_radio = QRadioButton("Compare Plans (single protocol)")

        self.protocol_mode_radio.setChecked(True)

        mode_layout.addWidget(self.protocol_mode_radio)
        mode_layout.addWidget(self.plan_mode_radio)

        # Connect mode selection signals
        self.protocol_mode_radio.toggled.connect(self._on_mode_changed)

        main_layout.addWidget(mode_group)

        # Splitter for selection and results
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(5)

        # Selection panel
        selection_frame = QFrame()
        selection_frame.setFrameShape(QFrame.StyledPanel)
        selection_layout = QVBoxLayout(selection_frame)

        # Plan selection
        self.plan_group = QGroupBox("Treatment Plan")
        plan_layout = QHBoxLayout(self.plan_group)

        self.plan_combo = QComboBox()
        self.add_plan_button = QPushButton("Add Plan")

        if self.plans:
            self.plan_combo.addItem(self.plans[0].name)

        plan_layout.addWidget(self.plan_combo)
        plan_layout.addWidget(self.add_plan_button)

        selection_layout.addWidget(self.plan_group)

        # Protocol selection
        self.protocol_group = QGroupBox("Clinical Protocols")
        protocol_layout = QVBoxLayout(self.protocol_group)

        self.protocol_list = QTableWidget()
        self.protocol_list.setColumnCount(2)
        self.protocol_list.setHorizontalHeaderLabels(["Protocol", "Treatment Site"])
        self.protocol_list.setSelectionBehavior(QTableWidget.SelectRows)
        self.protocol_list.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.protocol_list.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.protocol_list.verticalHeader().setVisible(False)

        protocol_layout.addWidget(self.protocol_list)

        # Protocol buttons
        protocol_button_layout = QHBoxLayout()
        self.select_all_protocols_button = QPushButton("Select All")
        self.deselect_all_protocols_button = QPushButton("Deselect All")

        protocol_button_layout.addWidget(self.select_all_protocols_button)
        protocol_button_layout.addWidget(self.deselect_all_protocols_button)
        protocol_layout.addLayout(protocol_button_layout)

        selection_layout.addWidget(self.protocol_group)

        # Results panel
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.StyledPanel)
        results_layout = QVBoxLayout(results_frame)

        # Results table
        results_group = QGroupBox("Comparison Results")
        results_table_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

        results_table_layout.addWidget(self.results_table)

        # Add result stats
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Pass rate: N/A")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()

        # Export button
        self.export_button = QPushButton("Export Results")
        stats_layout.addWidget(self.export_button)

        results_table_layout.addLayout(stats_layout)

        results_layout.addWidget(results_group)

        # Add widgets to splitter
        splitter.addWidget(selection_frame)
        splitter.addWidget(results_frame)
        splitter.setSizes([300, 400])

        main_layout.addWidget(splitter)

        # Bottom buttons
        buttons_layout = QHBoxLayout()
        self.compare_button = QPushButton("Compare")
        self.close_button = QPushButton("Close")

        buttons_layout.addStretch()
        buttons_layout.addWidget(self.compare_button)
        buttons_layout.addWidget(self.close_button)

        main_layout.addLayout(buttons_layout)

        # Connect signals
        self.add_plan_button.clicked.connect(self._on_add_plan)
        self.select_all_protocols_button.clicked.connect(self._on_select_all_protocols)
        self.deselect_all_protocols_button.clicked.connect(
            self._on_deselect_all_protocols
        )
        self.compare_button.clicked.connect(self._on_compare)
        self.close_button.clicked.connect(self.close)
        self.export_button.clicked.connect(self._on_export_results)

        # Initialize protocol list
        self._populate_protocols()

        # Update UI based on current mode
        self._update_ui_for_mode()

    def _populate_protocols(self):
        """Populate the protocol list."""
        self.protocol_list.setRowCount(0)

        protocols = self.protocol_manager.get_available_protocols()

        for i, protocol in enumerate(protocols):
            self.protocol_list.insertRow(i)

            # Create a checkbox item for the protocol name
            checkbox = QTableWidgetItem(protocol.name)
            checkbox.setFlags(
                Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
            )
            checkbox.setCheckState(Qt.Unchecked)

            self.protocol_list.setItem(i, 0, checkbox)
            self.protocol_list.setItem(i, 1, QTableWidgetItem(protocol.site))

    def _on_mode_changed(self):
        """Handle comparison mode change."""
        if self.protocol_mode_radio.isChecked():
            self.comparison_mode = "protocol"
        else:
            self.comparison_mode = "plan"

        self._update_ui_for_mode()

    def _update_ui_for_mode(self):
        """Update UI elements based on current comparison mode."""
        if self.comparison_mode == "protocol":
            self.plan_group.setTitle("Treatment Plan")
            self.protocol_group.setTitle("Select Clinical Protocols to Compare")
            self.plan_combo.setEnabled(True)
            self.add_plan_button.setEnabled(False)  # Only one plan in protocol mode
        else:
            self.plan_group.setTitle("Select Treatment Plans to Compare")
            self.protocol_group.setTitle("Clinical Protocol")
            self.plan_combo.setEnabled(True)
            self.add_plan_button.setEnabled(True)

    def _on_add_plan(self):
        """Handle adding a new plan for comparison."""
        # In a real implementation, this would open a dialog to select a plan
        # For now, we'll just add a dummy plan name
        plan_name = f"Plan {self.plan_combo.count() + 1}"
        self.plan_combo.addItem(plan_name)

    def _on_select_all_protocols(self):
        """Select all protocols in the list."""
        for i in range(self.protocol_list.rowCount()):
            item = self.protocol_list.item(i, 0)
            item.setCheckState(Qt.Checked)

    def _on_deselect_all_protocols(self):
        """Deselect all protocols in the list."""
        for i in range(self.protocol_list.rowCount()):
            item = self.protocol_list.item(i, 0)
            item.setCheckState(Qt.Unchecked)

    def _get_selected_protocols(self) -> List[str]:
        """Get names of selected protocols."""
        selected = []
        for i in range(self.protocol_list.rowCount()):
            item = self.protocol_list.item(i, 0)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())

        return selected

    def _on_compare(self):
        """Perform the comparison based on selected mode and items."""
        # Reset results table
        self.results_table.clear()

        # Get selected protocols and plans
        selected_protocols = self._get_selected_protocols()

        if not selected_protocols:
            QMessageBox.warning(
                self,
                "Selection Required",
                "Please select at least one protocol to compare.",
            )
            return

        self.selected_protocols = selected_protocols

        if self.comparison_mode == "protocol":
            # Compare multiple protocols against a single plan
            self._compare_protocols()
        else:
            # Compare multiple plans against a single protocol
            self._compare_plans()

    def _compare_protocols(self):
        """Compare multiple protocols against a single plan."""
        if not self.plans:
            QMessageBox.warning(
                self, "No Plan", "No treatment plan available for comparison."
            )
            return

        current_plan = self.plans[0]  # In protocol mode, we use the first plan

        # Set up results table
        self.results_table.setColumnCount(len(self.selected_protocols) + 1)

        # Set header labels
        headers = ["Clinical Goal"]
        headers.extend(self.selected_protocols)
        self.results_table.setHorizontalHeaderLabels(headers)

        # Get all unique goals from selected protocols
        all_goals = []
        for protocol_name in self.selected_protocols:
            protocol = self.protocol_manager.get_protocol(protocol_name)
            if protocol:
                for goal in protocol.goals:
                    # Check if we already have this goal
                    if not any(
                        g.structure_name == goal.structure_name
                        and g._get_type_str() == goal._get_type_str()
                        for g in all_goals
                    ):
                        all_goals.append(goal)

        # Sort goals by structure name
        all_goals.sort(key=lambda g: g.structure_name)

        # Set up rows
        self.results_table.setRowCount(len(all_goals))

        # Evaluate each goal for each protocol
        for i, goal in enumerate(all_goals):
            # Set goal description
            goal_item = QTableWidgetItem(
                f"{goal.structure_name}: {goal._get_type_str()} {goal._get_operator_str()} {goal.value}"
            )
            self.results_table.setItem(i, 0, goal_item)

            # Evaluate for each protocol
            for j, protocol_name in enumerate(self.selected_protocols):
                protocol = self.protocol_manager.get_protocol(protocol_name)
                if not protocol:
                    continue

                # Find matching goal in protocol
                matching_goal = None
                for p_goal in protocol.goals:
                    if (
                        p_goal.structure_name == goal.structure_name
                        and p_goal._get_type_str() == goal._get_type_str()
                    ):
                        matching_goal = p_goal
                        break

                if not matching_goal:
                    # Goal not in this protocol
                    result_item = QTableWidgetItem("N/A")
                    result_item.setTextAlignment(Qt.AlignCenter)
                    self.results_table.setItem(i, j + 1, result_item)
                else:
                    # Evaluate goal (simplified for demo)
                    # In a real implementation, this would use the DVH data
                    from random import random

                    passed = random() > 0.3  # 70% chance of passing

                    result_item = QTableWidgetItem("PASS" if passed else "FAIL")
                    result_item.setTextAlignment(Qt.AlignCenter)

                    # Set background color
                    if passed:
                        result_item.setBackground(QColor(200, 255, 200))  # Light green
                    else:
                        result_item.setBackground(QColor(255, 200, 200))  # Light red

                    self.results_table.setItem(i, j + 1, result_item)

        # Resize columns to content
        self.results_table.resizeColumnsToContents()

        # Update stats
        self._update_comparison_stats()

    def _compare_plans(self):
        """Compare multiple plans against a single protocol."""
        if not self.selected_protocols:
            QMessageBox.warning(
                self, "No Protocol", "Please select a protocol for comparison."
            )
            return

        # In plan mode, we use just the first selected protocol
        protocol_name = self.selected_protocols[0]
        protocol = self.protocol_manager.get_protocol(protocol_name)

        if not protocol:
            QMessageBox.warning(
                self, "Protocol Not Found", f"Protocol '{protocol_name}' not found."
            )
            return

        # Set up results table
        self.results_table.setColumnCount(self.plan_combo.count() + 1)

        # Set header labels
        headers = ["Clinical Goal"]
        for i in range(self.plan_combo.count()):
            headers.append(self.plan_combo.itemText(i))
        self.results_table.setHorizontalHeaderLabels(headers)

        # Set up rows
        self.results_table.setRowCount(len(protocol.goals))

        # Evaluate each goal for each plan
        for i, goal in enumerate(protocol.goals):
            # Set goal description
            goal_item = QTableWidgetItem(
                f"{goal.structure_name}: {goal._get_type_str()} {goal._get_operator_str()} {goal.value}"
            )
            self.results_table.setItem(i, 0, goal_item)

            # Evaluate for each plan (simplified for demo)
            for j in range(self.plan_combo.count()):
                # In a real implementation, this would evaluate the goal against the plan
                from random import random

                passed = random() > 0.3  # 70% chance of passing

                result_item = QTableWidgetItem("PASS" if passed else "FAIL")
                result_item.setTextAlignment(Qt.AlignCenter)

                # Set background color
                if passed:
                    result_item.setBackground(QColor(200, 255, 200))  # Light green
                else:
                    result_item.setBackground(QColor(255, 200, 200))  # Light red

                self.results_table.setItem(i, j + 1, result_item)

        # Resize columns to content
        self.results_table.resizeColumnsToContents()

        # Update stats
        self._update_comparison_stats()

    def _update_comparison_stats(self):
        """Update the comparison statistics."""
        total_evaluations = 0
        passed_evaluations = 0

        # Count total and passed evaluations
        for i in range(self.results_table.rowCount()):
            for j in range(1, self.results_table.columnCount()):
                item = self.results_table.item(i, j)
                if item and item.text() != "N/A":
                    total_evaluations += 1
                    if item.text() == "PASS":
                        passed_evaluations += 1

        # Calculate pass rate
        pass_rate = (
            0
            if total_evaluations == 0
            else (passed_evaluations / total_evaluations) * 100
        )

        # Update stats label
        self.stats_label.setText(
            f"Pass rate: {pass_rate:.1f}% ({passed_evaluations}/{total_evaluations})"
        )

    def _on_export_results(self):
        """Export the comparison results to a file."""
        if self.results_table.rowCount() == 0:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return

        # Get file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "HTML Files (*.html);;CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".html"):
                self._export_results_html(file_path)
            elif file_path.endswith(".csv"):
                self._export_results_csv(file_path)
            else:
                # Default to HTML if no extension specified
                if "." not in os.path.basename(file_path):
                    file_path += ".html"
                self._export_results_html(file_path)

            QMessageBox.information(
                self, "Export Successful", f"Results exported to {file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Failed", f"Failed to export results: {str(e)}"
            )

    def _export_results_html(self, file_path: str):
        """Export results to HTML file."""
        # Generate HTML content
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Protocol Comparison Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .pass { background-color: #c8ffc8; }
                .fail { background-color: #ffc8c8; }
                .na { background-color: #f0f0f0; }
                h1, h2 { color: #333; }
                .summary { margin: 20px 0; padding: 10px; background-color: #f8f8f8; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>Protocol Comparison Results</h1>
        """

        # Add comparison info
        if self.comparison_mode == "protocol":
            html += f"<p>Plan: {self.plan_combo.currentText()}</p>"
            html += (
                "<p>Protocols compared: " + ", ".join(self.selected_protocols) + "</p>"
            )
        else:
            plans = [
                self.plan_combo.itemText(i) for i in range(self.plan_combo.count())
            ]
            html += "<p>Plans compared: " + ", ".join(plans) + "</p>"
            html += f"<p>Protocol: {self.selected_protocols[0]}</p>"

        # Add summary
        html += f"<div class='summary'>{self.stats_label.text()}</div>"

        # Add results table
        html += "<table>"

        # Add header
        html += "<tr>"
        for j in range(self.results_table.columnCount()):
            html += f"<th>{self.results_table.horizontalHeaderItem(j).text()}</th>"
        html += "</tr>"

        # Add rows
        for i in range(self.results_table.rowCount()):
            html += "<tr>"
            for j in range(self.results_table.columnCount()):
                item = self.results_table.item(i, j)
                if j == 0:  # Goal description
                    html += f"<td>{item.text()}</td>"
                else:  # Result
                    css_class = ""
                    if item.text() == "PASS":
                        css_class = "pass"
                    elif item.text() == "FAIL":
                        css_class = "fail"
                    elif item.text() == "N/A":
                        css_class = "na"

                    html += f"<td class='{css_class}'>{item.text()}</td>"
            html += "</tr>"

        html += """
        </table>
        <p><i>Generated by QuangTPS Protocol Comparison Tool</i></p>
        </body>
        </html>
        """

        # Write to file
        with open(file_path, "w") as f:
            f.write(html)

    def _export_results_csv(self, file_path: str):
        """Export results to CSV file."""
        import csv

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            header = []
            for j in range(self.results_table.columnCount()):
                header.append(self.results_table.horizontalHeaderItem(j).text())
            writer.writerow(header)

            # Write rows
            for i in range(self.results_table.rowCount()):
                row = []
                for j in range(self.results_table.columnCount()):
                    item = self.results_table.item(i, j)
                    row.append(item.text())
                writer.writerow(row)

            # Add summary
            writer.writerow([])
            writer.writerow([self.stats_label.text()])


# For testing
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = ProtocolComparisonDialog()
    dialog.show()
    sys.exit(app.exec_())
