#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Quality Widget Module for QuangTPS.

This module provides a widget for displaying plan quality evaluation results
in an Eclipse-like style for the evaluation tab.
"""

import os
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QPushButton, QDialog, QFileDialog, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPalette

from quangtps.evaluation.plan_quality import PlanQualityEvaluator, GoalResultType
from quangtps.evaluation.plan_evaluation import PlanEvaluation
from quangtps.ui.dialogs.protocol_dialog import ClinicalProtocolDialog

logger = logging.getLogger(__name__)


class GoalProgressBar(QProgressBar):
    """
    Custom progress bar for displaying clinical goal achievement.
    
    This widget shows a colored progress bar indicating the level of
    achievement for clinical goals (pass, acceptable, fail).
    """
    
    def __init__(self, parent=None):
        """Initialize the progress bar."""
        super().__init__(parent)
        self.setTextVisible(True)
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(0)
        self.setFixedHeight(20)
        
    def setProgress(self, value: float, result_type: str):
        """
        Set the progress value and style based on result type.
        
        Args:
            value: Progress value (0-100)
            result_type: Result type (PASSED, ACCEPTABLE, FAILED, NOT_EVALUATED)
        """
        self.setValue(int(value))
        
        # Set color based on result type
        style = "QProgressBar {"
        style += "border: 1px solid #CCCCCC;"
        style += "border-radius: 3px;"
        style += "text-align: center;"
        style += "}"
        
        style += "QProgressBar::chunk {"
        
        if result_type == "PASSED":
            style += "background-color: #4CAF50;"  # Green
        elif result_type == "ACCEPTABLE":
            style += "background-color: #FFC107;"  # Amber
        elif result_type == "FAILED":
            style += "background-color: #F44336;"  # Red
        else:  # NOT_EVALUATED
            style += "background-color: #9E9E9E;"  # Gray
            
        style += "}"
        
        self.setStyleSheet(style)
        
    def paintEvent(self, event):
        """
        Customize the appearance of the progress bar.
        
        Args:
            event: Paint event
        """
        # Call the base class paint event
        super().paintEvent(event)
        
        # Add custom painting if needed
        painter = QPainter(self)
        painter.setPen(Qt.black)
        
        # Draw text
        text = f"{self.value()}%"
        painter.drawText(self.rect(), Qt.AlignCenter, text)


class PlanQualityWidget(QWidget):
    """
    Widget for displaying plan quality evaluation results.
    
    This widget provides an interface for evaluating plan quality against
    clinical protocols and displaying the results in an Eclipse-like style.
    """
    
    # Signal emitted when a clinical goal is selected
    goalSelected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """Initialize the plan quality widget."""
        super().__init__(parent)
        self.plan_evaluator = PlanQualityEvaluator()
        self.plan_evaluation = None
        self.protocols = []
        self.evaluation_results = None
        self.current_protocol = None
        
        self.initUI()
        self.loadProtocols()
        
    def initUI(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Protocol selection
        protocol_layout = QHBoxLayout()
        protocol_layout.addWidget(QLabel("Clinical Protocol:"))
        
        self.protocol_combo = QComboBox()
        self.protocol_combo.setMinimumWidth(200)
        self.protocol_combo.currentIndexChanged.connect(self.onProtocolChanged)
        protocol_layout.addWidget(self.protocol_combo)
        
        self.select_protocol_btn = QPushButton("Select...")
        self.select_protocol_btn.clicked.connect(self.openProtocolDialog)
        protocol_layout.addWidget(self.select_protocol_btn)
        
        self.evaluate_btn = QPushButton("Evaluate")
        self.evaluate_btn.clicked.connect(self.evaluatePlanQuality)
        protocol_layout.addWidget(self.evaluate_btn)
        
        protocol_layout.addStretch()
        main_layout.addLayout(protocol_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)
        
        # Scores section
        scores_layout = QHBoxLayout()
        
        # Overall score
        overall_score_layout = QVBoxLayout()
        overall_score_layout.addWidget(QLabel("Overall Score"))
        self.overall_progress = GoalProgressBar()
        overall_score_layout.addWidget(self.overall_progress)
        scores_layout.addLayout(overall_score_layout)
        
        # Target score
        target_score_layout = QVBoxLayout()
        target_score_layout.addWidget(QLabel("Target Score"))
        self.target_progress = GoalProgressBar()
        target_score_layout.addWidget(self.target_progress)
        scores_layout.addLayout(target_score_layout)
        
        # OAR score
        oar_score_layout = QVBoxLayout()
        oar_score_layout.addWidget(QLabel("OAR Score"))
        self.oar_progress = GoalProgressBar()
        oar_score_layout.addWidget(self.oar_progress)
        scores_layout.addLayout(oar_score_layout)
        
        main_layout.addLayout(scores_layout)
        
        # Clinical goals table
        main_layout.addWidget(QLabel("Clinical Goals"))
        
        self.goals_table = QTableWidget()
        self.goals_table.setColumnCount(6)
        self.goals_table.setHorizontalHeaderLabels([
            "Structure", "Goal", "Priority", "Result", "Value", "Status"
        ])
        
        # Set table properties
        self.goals_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.goals_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.goals_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.goals_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.goals_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.goals_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        self.goals_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.goals_table.setSelectionMode(QTableWidget.SingleSelection)
        self.goals_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.goals_table.setAlternatingRowColors(True)
        
        self.goals_table.itemSelectionChanged.connect(self.onGoalSelected)
        
        main_layout.addWidget(self.goals_table)
        
        # Status section
        status_layout = QHBoxLayout()
        
        self.passed_label = QLabel("Passed: 0")
        status_layout.addWidget(self.passed_label)
        
        self.acceptable_label = QLabel("Acceptable: 0")
        status_layout.addWidget(self.acceptable_label)
        
        self.failed_label = QLabel("Failed: 0")
        status_layout.addWidget(self.failed_label)
        
        self.not_evaluated_label = QLabel("Not Evaluated: 0")
        status_layout.addWidget(self.not_evaluated_label)
        
        status_layout.addStretch()
        
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self.exportReport)
        status_layout.addWidget(self.export_btn)
        
        main_layout.addLayout(status_layout)
        
        # Set the main layout
        self.setLayout(main_layout)
        
    def loadProtocols(self):
        """Load available clinical protocols."""
        try:
            # Clear and update the combo box
            self.protocol_combo.clear()
            self.protocols = self.plan_evaluator.get_available_protocols()
            
            # Add protocols to combo box
            self.protocol_combo.addItem("Select Protocol...", "")
            
            for protocol in self.protocols:
                name = protocol["name"]
                count = protocol["goals_count"]
                self.protocol_combo.addItem(f"{name} ({count} goals)", protocol["path"])
                
            # Disable evaluation if no protocols
            self.evaluate_btn.setEnabled(len(self.protocols) > 0)
            
            logger.info(f"Loaded {len(self.protocols)} clinical protocols")
            
        except Exception as e:
            logger.error(f"Error loading protocols: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to load protocols: {str(e)}")
            
    def openProtocolDialog(self):
        """Open the protocol selection dialog."""
        try:
            dialog = ClinicalProtocolDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                protocol_info = dialog.get_selected_protocol()
                if protocol_info:
                    # Find and select the protocol in the combo box
                    for i in range(self.protocol_combo.count()):
                        if self.protocol_combo.itemData(i) == protocol_info["path"]:
                            self.protocol_combo.setCurrentIndex(i)
                            break
                    else:
                        # Protocol not in combo, load and add it
                        self.loadProtocols()
                        # Try to find again
                        for i in range(self.protocol_combo.count()):
                            if self.protocol_combo.itemData(i) == protocol_info["path"]:
                                self.protocol_combo.setCurrentIndex(i)
                                break
                            
        except Exception as e:
            logger.error(f"Error opening protocol dialog: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to open protocol dialog: {str(e)}")
            
    def setPlanEvaluation(self, plan_evaluation: PlanEvaluation):
        """
        Set the plan evaluation for quality assessment.
        
        Args:
            plan_evaluation: PlanEvaluation object containing dose data
        """
        self.plan_evaluation = plan_evaluation
        self.plan_evaluator.set_plan_evaluation(plan_evaluation)
        
        # Clear existing results
        self.clearResults()
        
        # Enable evaluation if a protocol is selected
        self.evaluate_btn.setEnabled(
            self.plan_evaluation is not None and 
            self.protocol_combo.currentIndex() > 0
        )
        
    def onProtocolChanged(self):
        """Handle protocol selection change."""
        # Get selected protocol path
        index = self.protocol_combo.currentIndex()
        if index <= 0:
            self.current_protocol = None
            self.evaluate_btn.setEnabled(False)
            return
            
        protocol_path = self.protocol_combo.itemData(index)
        
        # Load protocol
        if protocol_path:
            success = self.plan_evaluator.load_protocol(protocol_path)
            if success:
                self.current_protocol = self.plan_evaluator.get_protocol_summary()
                self.evaluate_btn.setEnabled(self.plan_evaluation is not None)
                logger.info(f"Loaded protocol: {self.current_protocol['name']}")
            else:
                self.current_protocol = None
                self.evaluate_btn.setEnabled(False)
                QMessageBox.warning(
                    self, 
                    "Protocol Error", 
                    f"Failed to load protocol from {protocol_path}"
                )
                
    def evaluatePlanQuality(self):
        """Evaluate plan quality against the selected protocol."""
        if not self.plan_evaluation:
            QMessageBox.warning(self, "Warning", "No plan evaluation available")
            return
            
        if not self.current_protocol:
            QMessageBox.warning(self, "Warning", "No protocol selected")
            return
            
        try:
            # Evaluate plan quality
            results = self.plan_evaluator.evaluate_plan_quality()
            
            if "error" in results:
                QMessageBox.warning(self, "Evaluation Error", results["error"])
                return
                
            # Store results
            self.evaluation_results = results
            
            # Update UI with results
            self.updateResultsDisplay()
            
        except Exception as e:
            logger.error(f"Error evaluating plan quality: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to evaluate plan quality: {str(e)}")
            
    def updateResultsDisplay(self):
        """Update the UI with evaluation results."""
        if not self.evaluation_results:
            return
            
        # Update progress bars
        overall_score = self.evaluation_results.get("overall_score", 0.0)
        target_score = self.evaluation_results.get("target_score", 0.0)
        oar_score = self.evaluation_results.get("oar_score", 0.0)
        
        result_type = "PASSED"
        if overall_score < 70:
            result_type = "FAILED"
        elif overall_score < 90:
            result_type = "ACCEPTABLE"
            
        self.overall_progress.setProgress(overall_score, result_type)
        
        target_result_type = "PASSED"
        if target_score < 70:
            target_result_type = "FAILED"
        elif target_score < 90:
            target_result_type = "ACCEPTABLE"
            
        self.target_progress.setProgress(target_score, target_result_type)
        
        oar_result_type = "PASSED"
        if oar_score < 70:
            oar_result_type = "FAILED"
        elif oar_score < 90:
            oar_result_type = "ACCEPTABLE"
            
        self.oar_progress.setProgress(oar_score, oar_result_type)
        
        # Update status labels
        passed = self.evaluation_results.get("goals_passed", 0)
        acceptable = self.evaluation_results.get("goals_acceptable", 0)
        failed = self.evaluation_results.get("goals_failed", 0)
        not_evaluated = self.evaluation_results.get("goals_not_evaluated", 0)
        
        self.passed_label.setText(f"Passed: {passed}")
        self.acceptable_label.setText(f"Acceptable: {acceptable}")
        self.failed_label.setText(f"Failed: {failed}")
        self.not_evaluated_label.setText(f"Not Evaluated: {not_evaluated}")
        
        # Update goals table
        self.updateGoalsTable()
        
    def updateGoalsTable(self):
        """Update the clinical goals table with results."""
        if not self.evaluation_results:
            return
            
        # Get goals details
        goals = self.evaluation_results.get("goals_details", [])
        
        # Clear table
        self.goals_table.setRowCount(0)
        
        # Add goals to table
        self.goals_table.setRowCount(len(goals))
        
        for row, goal in enumerate(goals):
            # Structure name
            structure_item = QTableWidgetItem(goal["structure_name"])
            self.goals_table.setItem(row, 0, structure_item)
            
            # Goal description
            description_item = QTableWidgetItem(goal["description"])
            self.goals_table.setItem(row, 1, description_item)
            
            # Priority
            priority_item = QTableWidgetItem(goal["priority"])
            self.goals_table.setItem(row, 2, priority_item)
            
            # Result
            result_item = QTableWidgetItem(goal["result"] if goal["result"] else "Not evaluated")
            self.goals_table.setItem(row, 3, result_item)
            
            # Value
            value = goal["achieved_value"]
            value_text = f"{value:.2f}" if value is not None else "N/A"
            value_item = QTableWidgetItem(value_text)
            self.goals_table.setItem(row, 4, value_item)
            
            # Status
            status_item = QTableWidgetItem(goal["result_type"])
            
            # Set status color
            if goal["result_type"] == "PASSED":
                status_item.setBackground(QColor("#CCFFCC"))  # Light green
            elif goal["result_type"] == "ACCEPTABLE":
                status_item.setBackground(QColor("#FFFFCC"))  # Light yellow
            elif goal["result_type"] == "FAILED":
                status_item.setBackground(QColor("#FFCCCC"))  # Light red
                
            self.goals_table.setItem(row, 5, status_item)
            
        # Resize columns
        self.goals_table.resizeColumnsToContents()
        
        # Enable export button
        self.export_btn.setEnabled(True)
        
    def onGoalSelected(self):
        """Handle goal selection in the table."""
        selected_rows = self.goals_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        
        if self.evaluation_results and "goals_details" in self.evaluation_results:
            goals = self.evaluation_results["goals_details"]
            if row < len(goals):
                # Emit signal with the selected goal
                self.goalSelected.emit(goals[row])
                
    def exportReport(self):
        """Export the plan quality report."""
        if not self.evaluation_results:
            QMessageBox.warning(self, "Warning", "No evaluation results to export")
            return
            
        try:
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Report",
                "",
                "HTML Files (*.html);;Text Files (*.txt);;All Files (*.*)"
            )
            
            if not file_path:
                return
                
            # Determine file type and export
            if file_path.lower().endswith(".html"):
                self.exportHtmlReport(file_path)
            else:
                self.exportTextReport(file_path)
                
            QMessageBox.information(
                self, 
                "Export Successful", 
                f"Report saved to {file_path}"
            )
            
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to export report: {str(e)}")
            
    def exportHtmlReport(self, file_path: str):
        """
        Export a report in HTML format.
        
        Args:
            file_path: Path to save the HTML report
        """
        # This would be implemented with HTML templates in a real system
        # Simple implementation for now
        html = "<html><head><title>Plan Quality Report</title>"
        html += "<style>"
        html += "body { font-family: Arial, sans-serif; margin: 20px; }"
        html += "h1, h2 { color: #2c3e50; }"
        html += "table { border-collapse: collapse; width: 100%; }"
        html += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }"
        html += "th { background-color: #f2f2f2; }"
        html += ".passed { background-color: #d5f5e3; }"
        html += ".acceptable { background-color: #fcf3cf; }"
        html += ".failed { background-color: #f5b7b1; }"
        html += ".not-evaluated { background-color: #f2f3f4; }"
        html += ".progress-bar { height: 20px; background-color: #eee; border-radius: 3px; }"
        html += ".progress-value { height: 100%; border-radius: 3px; text-align: center; line-height: 20px; color: white; }"
        html += ".progress-passed { background-color: #2ecc71; }"
        html += ".progress-acceptable { background-color: #f1c40f; }"
        html += ".progress-failed { background-color: #e74c3c; }"
        html += "</style></head><body>"
        
        # Protocol and Plan info
        protocol_name = self.evaluation_results.get("protocol_name", "Unknown")
        html += f"<h1>Plan Quality Report</h1>"
        html += f"<p><strong>Protocol:</strong> {protocol_name}</p>"
        
        # Scores
        overall_score = self.evaluation_results.get("overall_score", 0.0)
        target_score = self.evaluation_results.get("target_score", 0.0)
        oar_score = self.evaluation_results.get("oar_score", 0.0)
        
        html += "<h2>Quality Scores</h2>"
        
        # Overall score
        score_class = "progress-passed"
        if overall_score < 70:
            score_class = "progress-failed"
        elif overall_score < 90:
            score_class = "progress-acceptable"
            
        html += "<p><strong>Overall Score:</strong></p>"
        html += f'<div class="progress-bar"><div class="progress-value {score_class}" style="width: {overall_score}%;">{overall_score:.1f}%</div></div>'
        
        # Target score
        score_class = "progress-passed"
        if target_score < 70:
            score_class = "progress-failed"
        elif target_score < 90:
            score_class = "progress-acceptable"
            
        html += "<p><strong>Target Score:</strong></p>"
        html += f'<div class="progress-bar"><div class="progress-value {score_class}" style="width: {target_score}%;">{target_score:.1f}%</div></div>'
        
        # OAR score
        score_class = "progress-passed"
        if oar_score < 70:
            score_class = "progress-failed"
        elif oar_score < 90:
            score_class = "progress-acceptable"
            
        html += "<p><strong>OAR Score:</strong></p>"
        html += f'<div class="progress-bar"><div class="progress-value {score_class}" style="width: {oar_score}%;">{oar_score:.1f}%</div></div>'
        
        # Goal counts
        passed = self.evaluation_results.get("goals_passed", 0)
        acceptable = self.evaluation_results.get("goals_acceptable", 0)
        failed = self.evaluation_results.get("goals_failed", 0)
        not_evaluated = self.evaluation_results.get("goals_not_evaluated", 0)
        
        html += "<h2>Goals Summary</h2>"
        html += f"<p>Passed: {passed} | Acceptable: {acceptable} | Failed: {failed} | Not Evaluated: {not_evaluated}</p>"
        
        # Clinical goals
        html += "<h2>Clinical Goals</h2>"
        html += "<table>"
        html += "<tr><th>Structure</th><th>Goal</th><th>Priority</th><th>Result</th><th>Value</th><th>Status</th></tr>"
        
        goals = self.evaluation_results.get("goals_details", [])
        for goal in goals:
            # Determine class based on result type
            css_class = "not-evaluated"
            if goal["result_type"] == "PASSED":
                css_class = "passed"
            elif goal["result_type"] == "ACCEPTABLE":
                css_class = "acceptable"
            elif goal["result_type"] == "FAILED":
                css_class = "failed"
                
            html += f'<tr class="{css_class}">'
            html += f'<td>{goal["structure_name"]}</td>'
            html += f'<td>{goal["description"]}</td>'
            html += f'<td>{goal["priority"]}</td>'
            
            result_text = goal["result"] if goal["result"] else "Not evaluated"
            html += f'<td>{result_text}</td>'
            
            value = goal["achieved_value"]
            value_text = f"{value:.2f}" if value is not None else "N/A"
            html += f'<td>{value_text}</td>'
            
            html += f'<td>{goal["result_type"]}</td>'
            html += '</tr>'
            
        html += "</table>"
        
        html += "</body></html>"
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(html)
            
    def exportTextReport(self, file_path: str):
        """
        Export a report in text format.
        
        Args:
            file_path: Path to save the text report
        """
        # Simple text report
        text = "Plan Quality Report\n"
        text += "===================\n\n"
        
        # Protocol info
        protocol_name = self.evaluation_results.get("protocol_name", "Unknown")
        text += f"Protocol: {protocol_name}\n\n"
        
        # Scores
        overall_score = self.evaluation_results.get("overall_score", 0.0)
        target_score = self.evaluation_results.get("target_score", 0.0)
        oar_score = self.evaluation_results.get("oar_score", 0.0)
        
        text += "Quality Scores\n"
        text += "--------------\n"
        
        result_type = "PASSED"
        if overall_score < 70:
            result_type = "FAILED"
        elif overall_score < 90:
            result_type = "ACCEPTABLE"
            
        text += f"Overall Score: {overall_score:.1f}% ({result_type})\n"
        
        target_result_type = "PASSED"
        if target_score < 70:
            target_result_type = "FAILED"
        elif target_score < 90:
            target_result_type = "ACCEPTABLE"
            
        text += f"Target Score: {target_score:.1f}% ({target_result_type})\n"
        
        oar_result_type = "PASSED"
        if oar_score < 70:
            oar_result_type = "FAILED"
        elif oar_score < 90:
            oar_result_type = "ACCEPTABLE"
            
        text += f"OAR Score: {oar_score:.1f}% ({oar_result_type})\n\n"
        
        # Goal counts
        passed = self.evaluation_results.get("goals_passed", 0)
        acceptable = self.evaluation_results.get("goals_acceptable", 0)
        failed = self.evaluation_results.get("goals_failed", 0)
        not_evaluated = self.evaluation_results.get("goals_not_evaluated", 0)
        
        text += "Goals Summary\n"
        text += "-------------\n"
        text += f"Passed: {passed}\n"
        text += f"Acceptable: {acceptable}\n"
        text += f"Failed: {failed}\n"
        text += f"Not Evaluated: {not_evaluated}\n\n"
        
        # Clinical goals
        text += "Clinical Goals\n"
        text += "--------------\n"
        
        goals = self.evaluation_results.get("goals_details", [])
        for goal in sorted(goals, key=lambda g: g["result_type"]):
            text += f"Structure: {goal['structure_name']}\n"
            text += f"Goal: {goal['description']}\n"
            text += f"Priority: {goal['priority']}\n"
            
            result_text = goal["result"] if goal["result"] else "Not evaluated"
            text += f"Result: {result_text}\n"
            
            value = goal["achieved_value"]
            value_text = f"{value:.2f}" if value is not None else "N/A"
            text += f"Value: {value_text}\n"
            
            text += f"Status: {goal['result_type']}\n"
            text += "--------------\n"
            
        # Write to file
        with open(file_path, 'w') as f:
            f.write(text)
            
    def clearResults(self):
        """Clear all evaluation results."""
        self.evaluation_results = None
        
        # Reset progress bars
        self.overall_progress.setProgress(0, "NOT_EVALUATED")
        self.target_progress.setProgress(0, "NOT_EVALUATED")
        self.oar_progress.setProgress(0, "NOT_EVALUATED")
        
        # Reset status labels
        self.passed_label.setText("Passed: 0")
        self.acceptable_label.setText("Acceptable: 0")
        self.failed_label.setText("Failed: 0")
        self.not_evaluated_label.setText("Not Evaluated: 0")
        
        # Clear goals table
        self.goals_table.setRowCount(0)
        
        # Disable export button
        self.export_btn.setEnabled(False) 