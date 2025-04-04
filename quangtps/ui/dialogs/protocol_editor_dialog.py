#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Protocol Editor Dialog Module
===========================

This module provides a dialog for editing clinical protocols, similar to
the protocol editing interface in Eclipse.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QListWidget, QListWidgetItem, QTextEdit, QGroupBox,
    QFormLayout, QDialogButtonBox, QMessageBox, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox, QDoubleSpinBox,
    QCheckBox, QMenu, QAction, QAbstractItemView, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QColor, QContextMenuEvent

try:
    from quangtps.evaluation.clinical_goals import (
        ClinicalGoal, GoalType, GoalOperator, GoalPriority, GoalResult
    )
    from quangtps.evaluation.clinical_protocols import ClinicalProtocol, ProtocolManager
except ImportError:
    logging.warning("Failed to import protocol management modules")
    # Define placeholder classes for type hints
    class ClinicalGoal:
        pass
    class ClinicalProtocol:
        pass
    class ProtocolManager:
        pass
    class GoalType:
        pass
    class GoalOperator:
        pass
    class GoalPriority:
        pass
    class GoalResult:
        pass

logger = logging.getLogger(__name__)

class GoalEditDialog(QDialog):
    """Dialog for editing a clinical goal."""
    
    def __init__(self, parent=None, goal=None, structures=None):
        """
        Initialize the goal edit dialog.
        
        Parameters:
        -----------
        parent : QWidget, optional
            Parent widget
        goal : ClinicalGoal, optional
            Goal to edit, or None for a new goal
        structures : List[str], optional
            List of available structure names
        """
        super().__init__(parent)
        self.setWindowTitle("Edit Clinical Goal")
        self.resize(500, 400)
        
        self.goal = goal
        self.structures = structures or []
        
        self.init_ui()
        self.load_goal()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Form layout for goal properties
        form_layout = QFormLayout()
        
        # Structure
        self.structure_combo = QComboBox()
        self.structure_combo.addItems(self.structures)
        self.structure_combo.setEditable(True)
        form_layout.addRow("Structure:", self.structure_combo)
        
        # Goal type
        self.type_combo = QComboBox()
        self.type_combo.addItem("Volume at Dose (V20Gy)", GoalType.VOLUME_AT_DOSE)
        self.type_combo.addItem("Dose at Volume (D95%)", GoalType.DOSE_AT_VOLUME)
        self.type_combo.addItem("Maximum Dose", GoalType.MAX_DOSE)
        self.type_combo.addItem("Minimum Dose", GoalType.MIN_DOSE)
        self.type_combo.addItem("Mean Dose", GoalType.MEAN_DOSE)
        self.type_combo.addItem("Conformity Index", GoalType.CI)
        self.type_combo.addItem("Homogeneity Index", GoalType.HI)
        self.type_combo.addItem("Gradient Index", GoalType.GI)
        self.type_combo.currentIndexChanged.connect(self.on_goal_type_changed)
        form_layout.addRow("Goal Type:", self.type_combo)
        
        # Dose level (for Volume at Dose)
        self.dose_level_layout = QHBoxLayout()
        self.dose_level_spin = QDoubleSpinBox()
        self.dose_level_spin.setRange(0, 200)
        self.dose_level_spin.setSuffix(" Gy")
        self.dose_level_spin.setDecimals(1)
        self.dose_level_layout.addWidget(self.dose_level_spin)
        self.dose_level_widget = QWidget()
        self.dose_level_widget.setLayout(self.dose_level_layout)
        form_layout.addRow("Dose Level:", self.dose_level_widget)
        
        # Volume level (for Dose at Volume)
        self.volume_level_layout = QHBoxLayout()
        self.volume_level_spin = QDoubleSpinBox()
        self.volume_level_spin.setRange(0, 100)
        self.volume_level_spin.setSuffix(" %")
        self.volume_level_spin.setDecimals(1)
        self.volume_level_layout.addWidget(self.volume_level_spin)
        self.volume_level_widget = QWidget()
        self.volume_level_widget.setLayout(self.volume_level_layout)
        form_layout.addRow("Volume Level:", self.volume_level_widget)
        
        # Operator
        self.operator_combo = QComboBox()
        self.operator_combo.addItem("<", GoalOperator.LESS_THAN)
        self.operator_combo.addItem("<=", GoalOperator.LESS_THAN_OR_EQUAL)
        self.operator_combo.addItem(">", GoalOperator.GREATER_THAN)
        self.operator_combo.addItem(">=", GoalOperator.GREATER_THAN_OR_EQUAL)
        self.operator_combo.addItem("=", GoalOperator.EQUAL)
        form_layout.addRow("Operator:", self.operator_combo)
        
        # Value
        self.value_layout = QHBoxLayout()
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(0, 200)
        self.value_spin.setDecimals(1)
        self.value_layout.addWidget(self.value_spin)
        self.value_units_label = QLabel("%")
        self.value_layout.addWidget(self.value_units_label)
        self.value_widget = QWidget()
        self.value_widget.setLayout(self.value_layout)
        form_layout.addRow("Target Value:", self.value_widget)
        
        # Priority
        self.priority_combo = QComboBox()
        self.priority_combo.addItem("Critical", GoalPriority.CRITICAL)
        self.priority_combo.addItem("Major", GoalPriority.MAJOR)
        self.priority_combo.addItem("Minor", GoalPriority.MINOR)
        form_layout.addRow("Priority:", self.priority_combo)
        
        # Notes
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        form_layout.addRow("Notes:", self.notes_edit)
        
        layout.addLayout(form_layout)
        
        # Button box
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
        # Initial setup based on goal type
        self.on_goal_type_changed()
    
    def load_goal(self):
        """Load goal data into the form if editing an existing goal."""
        if not self.goal:
            return
            
        # Set structure
        index = self.structure_combo.findText(self.goal.structure_name)
        if index >= 0:
            self.structure_combo.setCurrentIndex(index)
        else:
            self.structure_combo.setEditText(self.goal.structure_name)
            
        # Set goal type
        for i in range(self.type_combo.count()):
            if self.type_combo.itemData(i) == self.goal.goal_type:
                self.type_combo.setCurrentIndex(i)
                break
                
        # Set dose and volume levels
        if self.goal.dose_level is not None:
            self.dose_level_spin.setValue(self.goal.dose_level)
        if self.goal.volume_level is not None:
            self.volume_level_spin.setValue(self.goal.volume_level)
            
        # Set operator
        for i in range(self.operator_combo.count()):
            if self.operator_combo.itemData(i) == self.goal.operator:
                self.operator_combo.setCurrentIndex(i)
                break
                
        # Set value
        self.value_spin.setValue(self.goal.value)
        
        # Set priority
        for i in range(self.priority_combo.count()):
            if self.priority_combo.itemData(i) == self.goal.priority:
                self.priority_combo.setCurrentIndex(i)
                break
                
        # Set notes
        self.notes_edit.setText(self.goal.notes)
    
    def on_goal_type_changed(self):
        """Handle goal type changes and update UI accordingly."""
        goal_type = self.type_combo.currentData()
        
        # Show/hide dose level based on goal type
        if goal_type == GoalType.VOLUME_AT_DOSE:
            self.dose_level_widget.setVisible(True)
            self.value_units_label.setText("%")
            self.value_spin.setRange(0, 100)
            self.value_spin.setSuffix("")
        else:
            self.dose_level_widget.setVisible(False)
            
        # Show/hide volume level based on goal type
        if goal_type == GoalType.DOSE_AT_VOLUME:
            self.volume_level_widget.setVisible(True)
            self.value_units_label.setText("Gy")
            self.value_spin.setRange(0, 200)
            self.value_spin.setSuffix("")
        else:
            self.volume_level_widget.setVisible(False)
            
        # Set units for value based on goal type
        if goal_type in [GoalType.MAX_DOSE, GoalType.MIN_DOSE, GoalType.MEAN_DOSE]:
            self.value_units_label.setText("Gy")
            self.value_spin.setRange(0, 200)
            self.value_spin.setSuffix("")
        elif goal_type in [GoalType.CI, GoalType.HI, GoalType.GI]:
            self.value_units_label.setText("")
            self.value_spin.setRange(0, 10)
            self.value_spin.setSuffix("")
    
    def get_goal(self):
        """
        Get the edited goal.
        
        Returns:
        --------
        ClinicalGoal
            The edited clinical goal
        """
        structure_name = self.structure_combo.currentText()
        structure_id = structure_name.replace(" ", "")  # Simple ID generation
        goal_type = self.type_combo.currentData()
        operator = self.operator_combo.currentData()
        value = self.value_spin.value()
        priority = self.priority_combo.currentData()
        notes = self.notes_edit.toPlainText()
        
        # Dose and volume levels
        dose_level = None
        volume_level = None
        
        if goal_type == GoalType.VOLUME_AT_DOSE:
            dose_level = self.dose_level_spin.value()
        elif goal_type == GoalType.DOSE_AT_VOLUME:
            volume_level = self.volume_level_spin.value()
            
        # Create new goal
        goal = ClinicalGoal(
            structure_id=structure_id,
            structure_name=structure_name,
            goal_type=goal_type,
            operator=operator,
            value=value,
            priority=priority,
            dose_level=dose_level,
            volume_level=volume_level,
            notes=notes
        )
        
        return goal
    
    def accept(self):
        """Handle dialog acceptance with validation."""
        # Basic validation
        if not self.structure_combo.currentText():
            QMessageBox.warning(self, "Validation Error", "Structure name is required")
            return
        
        super().accept()


class ProtocolEditorDialog(QDialog):
    """
    Dialog for editing clinical protocols.
    
    This dialog allows users to create or edit clinical protocols for
    plan evaluation, similar to the protocol editing interface in Eclipse.
    """
    
    protocolSaved = pyqtSignal(object)
    
    def __init__(self, parent=None, protocol=None, protocol_manager=None):
        """
        Initialize the protocol editor dialog.
        
        Parameters:
        -----------
        parent : QWidget, optional
            Parent widget
        protocol : ClinicalProtocol, optional
            Protocol to edit, or None for a new protocol
        protocol_manager : ProtocolManager, optional
            Manager for clinical protocols
        """
        super().__init__(parent)
        self.setWindowTitle("Edit Clinical Protocol")
        self.resize(900, 700)
        
        self.protocol = protocol
        self.protocol_manager = protocol_manager
        
        if self.protocol_manager is None:
            from quangtps.evaluation.clinical_protocols import ProtocolManager
            self.protocol_manager = ProtocolManager()
            
        self.init_ui()
        self.load_protocol()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Protocol tab
        self.protocol_tab = QWidget()
        protocol_layout = QVBoxLayout(self.protocol_tab)
        
        # Protocol info group
        info_group = QGroupBox("Protocol Information")
        info_layout = QFormLayout(info_group)
        
        # Protocol name
        self.name_edit = QLineEdit()
        info_layout.addRow("Name:", self.name_edit)
        
        # Treatment site
        self.site_combo = QComboBox()
        self.site_combo.setEditable(True)
        self.site_combo.addItems([
            "Breast", "Prostate", "Head and Neck", "Lung", "Brain", 
            "Esophagus", "Liver", "Pancreas", "Rectum", "Gynecological"
        ])
        info_layout.addRow("Treatment Site:", self.site_combo)
        
        # Description
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        info_layout.addRow("Description:", self.description_edit)
        
        # Author
        self.author_edit = QLineEdit()
        info_layout.addRow("Author:", self.author_edit)
        
        # Version
        self.version_edit = QLineEdit()
        self.version_edit.setText("1.0")
        info_layout.addRow("Version:", self.version_edit)
        
        protocol_layout.addWidget(info_group)
        
        # Goals group
        goals_group = QGroupBox("Clinical Goals")
        goals_layout = QVBoxLayout(goals_group)
        
        # Goals table
        self.goals_table = QTableWidget()
        self.goals_table.setColumnCount(7)
        self.goals_table.setHorizontalHeaderLabels([
            "Structure", "Type", "Operator", "Value", "Priority", "Notes", ""
        ])
        self.goals_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.goals_table.customContextMenuRequested.connect(self.show_context_menu)
        self.goals_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.goals_table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # Set column widths
        header = self.goals_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Structure
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Operator
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Value
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Priority
        header.setSectionResizeMode(5, QHeaderView.Stretch)  # Notes
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Edit button
        
        goals_layout.addWidget(self.goals_table)
        
        # Goals toolbar
        buttons_layout = QHBoxLayout()
        
        self.add_goal_button = QPushButton("Add Goal")
        self.add_goal_button.clicked.connect(self.add_goal)
        buttons_layout.addWidget(self.add_goal_button)
        
        self.edit_goal_button = QPushButton("Edit Goal")
        self.edit_goal_button.clicked.connect(self.edit_goal)
        buttons_layout.addWidget(self.edit_goal_button)
        
        self.remove_goal_button = QPushButton("Remove Goal")
        self.remove_goal_button.clicked.connect(self.remove_goal)
        buttons_layout.addWidget(self.remove_goal_button)
        
        buttons_layout.addStretch()
        goals_layout.addLayout(buttons_layout)
        
        protocol_layout.addWidget(goals_group)
        
        # Add protocol tab to tab widget
        self.tab_widget.addTab(self.protocol_tab, "Protocol")
        
        # Add preview tab (future enhancement)
        self.preview_tab = QWidget()
        preview_layout = QVBoxLayout(self.preview_tab)
        
        self.preview_label = QLabel("Protocol Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        
        self.tab_widget.addTab(self.preview_tab, "Preview")
        
        layout.addWidget(self.tab_widget)
        
        # Button box
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self.button_box.button(QDialogButtonBox.Save).setText("Save Protocol")
        self.button_box.accepted.connect(self.save_protocol)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
    
    def load_protocol(self):
        """Load protocol data into the form if editing an existing protocol."""
        if not self.protocol:
            return
            
        # Load protocol information
        self.name_edit.setText(self.protocol.name)
        
        # Set site
        index = self.site_combo.findText(self.protocol.site)
        if index >= 0:
            self.site_combo.setCurrentIndex(index)
        else:
            self.site_combo.setEditText(self.protocol.site)
            
        self.description_edit.setText(self.protocol.description)
        self.author_edit.setText(self.protocol.author)
        self.version_edit.setText(self.protocol.version)
        
        # Load goals
        self.update_goals_table()
    
    def update_goals_table(self):
        """Update the goals table with current protocol goals."""
        self.goals_table.setRowCount(0)
        
        if not self.protocol:
            return
            
        # Add goals to table
        for i, goal in enumerate(self.protocol.goals):
            self.goals_table.insertRow(i)
            
            # Structure
            self.goals_table.setItem(i, 0, QTableWidgetItem(goal.structure_name))
            
            # Type
            type_item = QTableWidgetItem(self._get_goal_type_display(goal.goal_type))
            if goal.dose_level is not None:
                type_item.setText(f"V{goal.dose_level:.1f}Gy")
            elif goal.volume_level is not None:
                type_item.setText(f"D{goal.volume_level:.1f}%")
            self.goals_table.setItem(i, 1, type_item)
            
            # Operator
            op_item = QTableWidgetItem(self._get_operator_display(goal.operator))
            self.goals_table.setItem(i, 2, op_item)
            
            # Value
            value_text = f"{goal.value:.1f}"
            if goal.goal_type == GoalType.VOLUME_AT_DOSE:
                value_text += " %"
            elif goal.goal_type in [GoalType.DOSE_AT_VOLUME, GoalType.MAX_DOSE, GoalType.MIN_DOSE, GoalType.MEAN_DOSE]:
                value_text += " Gy"
            self.goals_table.setItem(i, 3, QTableWidgetItem(value_text))
            
            # Priority
            priority_item = QTableWidgetItem(self._get_priority_display(goal.priority))
            
            # Set priority color
            priority_color = {
                GoalPriority.CRITICAL: QColor("#FF4500"),  # OrangeRed
                GoalPriority.MAJOR: QColor("#FFA500"),     # Orange
                GoalPriority.MINOR: QColor("#FFD700")      # Gold
            }.get(goal.priority, QColor("#000000"))
            
            priority_item.setForeground(priority_color)
            priority_item.setFont(QFont("", -1, QFont.Bold))
            self.goals_table.setItem(i, 4, priority_item)
            
            # Notes
            self.goals_table.setItem(i, 5, QTableWidgetItem(goal.notes))
            
            # Edit button
            edit_button = QPushButton("Edit")
            edit_button.clicked.connect(lambda checked, row=i: self.edit_goal(row))
            self.goals_table.setCellWidget(i, 6, edit_button)
    
    def _get_goal_type_display(self, goal_type):
        """Get display text for a goal type."""
        return {
            GoalType.VOLUME_AT_DOSE: "Volume at Dose",
            GoalType.DOSE_AT_VOLUME: "Dose at Volume",
            GoalType.MAX_DOSE: "Maximum Dose",
            GoalType.MIN_DOSE: "Minimum Dose",
            GoalType.MEAN_DOSE: "Mean Dose",
            GoalType.CI: "Conformity Index",
            GoalType.HI: "Homogeneity Index",
            GoalType.GI: "Gradient Index"
        }.get(goal_type, "Unknown")
    
    def _get_operator_display(self, operator):
        """Get display text for an operator."""
        return {
            GoalOperator.LESS_THAN: "<",
            GoalOperator.LESS_THAN_OR_EQUAL: "≤",
            GoalOperator.GREATER_THAN: ">",
            GoalOperator.GREATER_THAN_OR_EQUAL: "≥",
            GoalOperator.EQUAL: "="
        }.get(operator, "Unknown")
    
    def _get_priority_display(self, priority):
        """Get display text for a priority."""
        return {
            GoalPriority.CRITICAL: "Critical",
            GoalPriority.MAJOR: "Major",
            GoalPriority.MINOR: "Minor"
        }.get(priority, "Unknown")
    
    def show_context_menu(self, position):
        """Show context menu for goals table."""
        menu = QMenu()
        
        # Only show context menu if a row is selected
        if not self.goals_table.selectedItems():
            return
            
        row = self.goals_table.currentRow()
        
        edit_action = QAction("Edit Goal", self)
        edit_action.triggered.connect(lambda: self.edit_goal(row))
        menu.addAction(edit_action)
        
        remove_action = QAction("Remove Goal", self)
        remove_action.triggered.connect(lambda: self.remove_goal(row))
        menu.addAction(remove_action)
        
        menu.exec_(self.goals_table.mapToGlobal(position))
    
    def add_goal(self):
        """Add a new clinical goal."""
        # Get list of structures from existing goals
        structures = []
        if self.protocol:
            structures = list(set(goal.structure_name for goal in self.protocol.goals))
        
        # Create and show dialog
        dialog = GoalEditDialog(self, structures=structures)
        if dialog.exec_() == QDialog.Accepted:
            goal = dialog.get_goal()
            
            # Add goal to protocol
            if not self.protocol:
                # Create new protocol if editing a new protocol
                self.protocol = ClinicalProtocol(
                    name=self.name_edit.text() or "New Protocol",
                    site=self.site_combo.currentText() or "Unknown",
                    description=self.description_edit.toPlainText(),
                    author=self.author_edit.text(),
                    version=self.version_edit.text() or "1.0"
                )
                
            self.protocol.add_goal(goal)
            self.update_goals_table()
    
    def edit_goal(self, row=None):
        """
        Edit a clinical goal.
        
        Parameters:
        -----------
        row : int, optional
            Row index of the goal to edit, or None to use the currently selected row
        """
        if row is None and self.goals_table.selectedItems():
            row = self.goals_table.currentRow()
            
        if row is None or row < 0 or not self.protocol or row >= len(self.protocol.goals):
            return
            
        # Get goal to edit
        goal = self.protocol.goals[row]
        
        # Get list of structures from existing goals
        structures = list(set(g.structure_name for g in self.protocol.goals))
        
        # Create and show dialog
        dialog = GoalEditDialog(self, goal=goal, structures=structures)
        if dialog.exec_() == QDialog.Accepted:
            # Replace goal
            self.protocol.goals[row] = dialog.get_goal()
            self.update_goals_table()
    
    def remove_goal(self, row=None):
        """
        Remove a clinical goal.
        
        Parameters:
        -----------
        row : int, optional
            Row index of the goal to remove, or None to use the currently selected row
        """
        if row is None and self.goals_table.selectedItems():
            row = self.goals_table.currentRow()
            
        if row is None or row < 0 or not self.protocol or row >= len(self.protocol.goals):
            return
            
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            "Are you sure you want to remove this clinical goal?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove goal
            self.protocol.remove_goal(row)
            self.update_goals_table()
    
    def validate_protocol(self):
        """
        Validate the protocol data.
        
        Returns:
        --------
        bool
            True if valid, False otherwise
        str
            Error message if invalid, empty string if valid
        """
        # Protocol name is required
        name = self.name_edit.text()
        if not name:
            return False, "Protocol name is required"
            
        # Treatment site is required
        site = self.site_combo.currentText()
        if not site:
            return False, "Treatment site is required"
            
        # At least one goal is required
        if self.protocol and not self.protocol.goals:
            return False, "At least one clinical goal is required"
            
        return True, ""
    
    def save_protocol(self):
        """Save the protocol."""
        # Validate protocol
        valid, error = self.validate_protocol()
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
            
        try:
            # Create new protocol or update existing one
            if not self.protocol:
                self.protocol = ClinicalProtocol(
                    name=self.name_edit.text(),
                    site=self.site_combo.currentText(),
                    description=self.description_edit.toPlainText(),
                    author=self.author_edit.text(),
                    version=self.version_edit.text() or "1.0"
                )
            else:
                # Update protocol information
                self.protocol.name = self.name_edit.text()
                self.protocol.site = self.site_combo.currentText()
                self.protocol.description = self.description_edit.toPlainText()
                self.protocol.author = self.author_edit.text()
                self.protocol.version = self.version_edit.text() or "1.0"
                
            # Save protocol
            success = self.protocol_manager.save_protocol(self.protocol)
            
            if success:
                self.protocolSaved.emit(self.protocol)
                self.accept()
            else:
                QMessageBox.warning(self, "Save Error", "Failed to save protocol")
                
        except Exception as e:
            logger.error(f"Error saving protocol: {e}")
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the protocol: {str(e)}")


# For testing purposes
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample protocol for testing
    class SampleGoal:
        def __init__(self, structure_name, goal_type, operator, value, priority, 
                    dose_level=None, volume_level=None, notes=""):
            self.structure_id = structure_name.replace(" ", "")
            self.structure_name = structure_name
            self.goal_type = goal_type
            self.operator = operator
            self.value = value
            self.priority = priority
            self.dose_level = dose_level
            self.volume_level = volume_level
            self.notes = notes
    
    class SampleProtocol:
        def __init__(self, name, site, description="", author="", version="1.0"):
            self.name = name
            self.site = site
            self.description = description
            self.author = author
            self.version = version
            self.goals = []
        
        def add_goal(self, goal):
            self.goals.append(goal)
        
        def remove_goal(self, index):
            if 0 <= index < len(self.goals):
                del self.goals[index]
    
    class SampleProtocolManager:
        def save_protocol(self, protocol):
            print(f"Saving protocol: {protocol.name}")
            print(f"Site: {protocol.site}")
            print(f"Description: {protocol.description}")
            print(f"Author: {protocol.author}")
            print(f"Version: {protocol.version}")
            print(f"Goals: {len(protocol.goals)}")
            for goal in protocol.goals:
                print(f"  {goal.structure_name}: {goal.goal_type} {goal.operator} {goal.value}")
            return True
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create mock data for testing
    protocol = SampleProtocol("Test Protocol", "Head and Neck", 
                            "A test protocol for head and neck cancer",
                            "Test User", "1.0")
    
    # Add some goals
    protocol.add_goal(SampleGoal("PTV", GoalType.DOSE_AT_VOLUME, GoalOperator.GREATER_THAN, 
                                95.0, GoalPriority.CRITICAL, None, 95.0, 
                                "Target coverage goal"))
    
    protocol.add_goal(SampleGoal("Spinal Cord", GoalType.MAX_DOSE, GoalOperator.LESS_THAN, 
                                45.0, GoalPriority.CRITICAL, None, None, 
                                "Critical OAR constraint"))
    
    protocol.add_goal(SampleGoal("Parotid L", GoalType.MEAN_DOSE, GoalOperator.LESS_THAN, 
                                26.0, GoalPriority.MAJOR, None, None, 
                                "Left parotid constraint"))
    
    # Create dialog for editing an existing protocol
    dialog = ProtocolEditorDialog(protocol=protocol, protocol_manager=SampleProtocolManager())
    
    # Connect to saved signal
    dialog.protocolSaved.connect(lambda p: print(f"Protocol saved: {p.name}"))