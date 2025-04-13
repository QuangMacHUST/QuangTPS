"""
Plan Comparison Dialog

This module provides a dialog for comparing multiple radiotherapy treatment plans,
similar to Eclipse's plan comparison functionality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget, QTreeWidget, 
    QTreeWidgetItem, QLabel, QComboBox, QPushButton, QFileDialog, QMessageBox,
    QGroupBox, QScrollArea, QWidget, QCheckBox, QRadioButton, QButtonGroup,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout, QSpinBox,
    QDoubleSpinBox, QApplication, QMenu, QAction, QToolBar, QToolButton, QStatusBar
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QIcon, QPalette, QFont, QPixmap, QImage

from quangtps.core.plan import Plan
from quangtps.ui.widgets.dvh_widget import DVHWidget
from quangtps.ui.widgets.metrics_table import MetricsTable
from quangtps.ui.widgets.dose_comparison_widget import DoseComparisonWidget
from quangtps.evaluation.plan_comparison import PlanComparison
from quangtps.evaluation.dvh.dvh_data import DVHData
from quangtps.evaluation.clinical_protocols import ClinicalProtocol
from quangtps.common.widgets import IconButton, CollapsibleGroup
from quangtps.common.paths import get_icon_path, get_protocols_directory

logger = logging.getLogger(__name__)

class StructureTreeItem(QTreeWidgetItem):
    """
    Tree widget item for a structure in the plan comparison dialog.
    
    This class represents a structure in the tree view, with checkable state
    to control visibility in the DVH plot.
    """
    
    def __init__(self, structure_id: str, structure_name: str, parent=None):
        """
        Initialize a structure tree item.
        
        Args:
            structure_id: ID of the structure
            structure_name: Display name of the structure
            parent: Parent tree widget item
        """
        super().__init__(parent)
        self.structure_id = structure_id
        self.setText(0, structure_name)
        self.setCheckState(0, Qt.Checked)
        
        # Add icon based on structure type if available
        if "ptv" in structure_id.lower() or "target" in structure_id.lower():
            self.setIcon(0, QIcon(get_icon_path("target")))
        elif "oar" in structure_id.lower() or "organ" in structure_id.lower():
            self.setIcon(0, QIcon(get_icon_path("organ")))
        else:
            self.setIcon(0, QIcon(get_icon_path("structure")))

class PlanComparisonDialog(QDialog):
    """
    Dialog for comparing multiple radiotherapy treatment plans.
    
    This dialog provides an Eclipse-like interface for comparing multiple plans,
    including DVH curves, metrics, and dose distributions. It allows visualizing
    differences between plans and evaluating clinical goals.
    """
    
    def __init__(self, reference_plan: Plan, parent=None):
        """
        Initialize a plan comparison dialog.
        
        Args:
            reference_plan: The reference plan for comparison
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up core data
        self.reference_plan = reference_plan
        self.comparison = PlanComparison(reference_plan)
        
        # GUI state
        self.current_plan_id = reference_plan.id
        self.current_structure_id = None
        self.dvh_display_type = "cumulative"
        self.volume_type = "relative"
        self.dose_type = "absolute"
        self.dose_diff_ref_plan_id = reference_plan.id
        self.dose_diff_eval_plan_id = None
        
        # Initialize UI
        self.setWindowTitle(f"Plan Comparison - {reference_plan.name}")
        self.resize(1200, 800)
        self.setWindowIcon(QIcon(get_icon_path("comparison")))
        
        self._init_ui()
        
        # Initial update
        self._update_ui()
    
    def _init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # Toolbar
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        
        # Add plan button
        add_plan_action = QAction(QIcon(get_icon_path("add")), "Add Plan", self)
        add_plan_action.triggered.connect(self._on_add_plan)
        toolbar.addAction(add_plan_action)
        
        # Remove plan button
        remove_plan_action = QAction(QIcon(get_icon_path("remove")), "Remove Plan", self)
        remove_plan_action.triggered.connect(self._on_remove_plan)
        toolbar.addAction(remove_plan_action)
        
        toolbar.addSeparator()
        
        # Select protocol button
        protocol_action = QAction(QIcon(get_icon_path("protocol")), "Select Protocol", self)
        protocol_action.triggered.connect(self._on_select_protocol)
        toolbar.addAction(protocol_action)
        
        toolbar.addSeparator()
        
        # Export report button
        export_action = QAction(QIcon(get_icon_path("export")), "Export Report", self)
        export_action.triggered.connect(self._on_export_report)
        toolbar.addAction(export_action)
        
        main_layout.addWidget(toolbar)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - structures tree
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Plans group
        plans_group = QGroupBox("Plans")
        plans_layout = QVBoxLayout(plans_group)
        
        self.plans_table = QTableWidget()
        self.plans_table.setColumnCount(3)
        self.plans_table.setHorizontalHeaderLabels(["Name", "Type", "Reference"])
        self.plans_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.plans_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.plans_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        plans_layout.addWidget(self.plans_table)
        left_layout.addWidget(plans_group)
        
        # Structures group
        structures_group = QGroupBox("Structures")
        structures_layout = QVBoxLayout(structures_group)
        
        self.structures_tree = QTreeWidget()
        self.structures_tree.setHeaderLabels(["Name"])
        self.structures_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.structures_tree.itemChanged.connect(self._on_structure_selection_changed)
        self.structures_tree.itemSelectionChanged.connect(self._on_structure_selection_changed)
        
        structures_layout.addWidget(self.structures_tree)
        left_layout.addWidget(structures_group)
        
        # Add to splitter
        splitter.addWidget(left_panel)
        
        # Right panel - tabs for DVH, metrics, dose diff, etc.
        right_panel = QTabWidget()
        
        # DVH tab
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)
        
        # DVH display options
        dvh_options_layout = QHBoxLayout()
        
        # DVH type
        dvh_type_layout = QHBoxLayout()
        dvh_type_layout.addWidget(QLabel("DVH Type:"))
        self.dvh_type_combo = QComboBox()
        self.dvh_type_combo.addItems(["Cumulative", "Differential"])
        self.dvh_type_combo.currentTextChanged.connect(
            lambda text: self._on_dvh_type_changed(text.lower())
        )
        dvh_type_layout.addWidget(self.dvh_type_combo)
        dvh_options_layout.addLayout(dvh_type_layout)
        
        # Volume type
        volume_type_layout = QHBoxLayout()
        volume_type_layout.addWidget(QLabel("Volume:"))
        self.volume_type_combo = QComboBox()
        self.volume_type_combo.addItems(["Relative (%)", "Absolute (cc)"])
        self.volume_type_combo.currentTextChanged.connect(
            lambda text: self._on_volume_type_changed("relative" if "%" in text else "absolute")
        )
        volume_type_layout.addWidget(self.volume_type_combo)
        dvh_options_layout.addLayout(volume_type_layout)
        
        # Dose type
        dose_type_layout = QHBoxLayout()
        dose_type_layout.addWidget(QLabel("Dose:"))
        self.dose_type_combo = QComboBox()
        self.dose_type_combo.addItems(["Absolute (Gy)", "Relative (%)"])
        self.dose_type_combo.currentTextChanged.connect(
            lambda text: self._on_dose_type_changed("absolute" if "Gy" in text else "relative")
        )
        dose_type_layout.addWidget(self.dose_type_combo)
        dvh_options_layout.addLayout(dose_type_layout)
        
        dvh_layout.addLayout(dvh_options_layout)
        
        # DVH widget
        self.dvh_widget = DVHWidget()
        dvh_layout.addWidget(self.dvh_widget)
        
        right_panel.addTab(dvh_tab, "DVH")
        
        # Metrics tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        self.metrics_table = MetricsTable()
        metrics_layout.addWidget(self.metrics_table)
        
        right_panel.addTab(metrics_tab, "Metrics")
        
        # Dose comparison tab
        dose_diff_tab = QWidget()
        dose_diff_layout = QVBoxLayout(dose_diff_tab)
        
        # Dose diff controls
        dose_diff_controls = QHBoxLayout()
        
        # Reference plan selection
        dose_diff_controls.addWidget(QLabel("Reference Plan:"))
        self.dose_diff_ref_combo = QComboBox()
        self.dose_diff_ref_combo.currentIndexChanged.connect(self._on_dose_diff_plan_changed)
        dose_diff_controls.addWidget(self.dose_diff_ref_combo)
        
        # Evaluation plan selection
        dose_diff_controls.addWidget(QLabel("Evaluation Plan:"))
        self.dose_diff_eval_combo = QComboBox()
        self.dose_diff_eval_combo.currentIndexChanged.connect(self._on_dose_diff_plan_changed)
        dose_diff_controls.addWidget(self.dose_diff_eval_combo)
        
        # Display type
        dose_diff_controls.addWidget(QLabel("Display:"))
        self.dose_diff_display_combo = QComboBox()
        self.dose_diff_display_combo.addItems(["Subtraction", "Gamma Index"])
        self.dose_diff_display_combo.currentTextChanged.connect(self._update_dose_diff_view)
        dose_diff_controls.addWidget(self.dose_diff_display_combo)
        
        dose_diff_layout.addLayout(dose_diff_controls)
        
        # Dose comparison widget
        self.dose_comparison_widget = DoseComparisonWidget()
        dose_diff_layout.addWidget(self.dose_comparison_widget)
        
        right_panel.addTab(dose_diff_tab, "Dose Comparison")
        
        # Goals tab
        goals_tab = QWidget()
        goals_layout = QVBoxLayout(goals_tab)
        
        # Select protocol widget
        protocol_layout = QHBoxLayout()
        protocol_layout.addWidget(QLabel("Protocol:"))
        self.protocol_combo = QComboBox()
        self.protocol_combo.setMinimumWidth(200)
        protocol_layout.addWidget(self.protocol_combo)
        
        # Apply protocol button
        self.apply_protocol_button = QPushButton("Apply Protocol")
        self.apply_protocol_button.clicked.connect(self._on_apply_protocol)
        protocol_layout.addWidget(self.apply_protocol_button)
        
        protocol_layout.addStretch()
        goals_layout.addLayout(protocol_layout)
        
        # Goals table
        self.goals_table = QTableWidget()
        self.goals_table.setColumnCount(5)  # Goal, Structure, Plan1, Plan2, etc.
        self.goals_table.setHorizontalHeaderLabels(["Goal", "Structure", "Reference", "Comparison", "Difference"])
        self.goals_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.goals_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.goals_table.setAlternatingRowColors(True)
        goals_layout.addWidget(self.goals_table)
        
        right_panel.addTab(goals_tab, "Clinical Goals")
        
        # Add to splitter
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])  # Default sizes
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        
        # Find protocols
        self._populate_protocols()
    
    def _update_ui(self):
        """Update all UI components with current data."""
        self._update_plan_table()
        self._update_structure_list()
        self._update_dvh()
        self._update_metrics_table()
        self._update_dose_diff_view()
        self._update_goals_table()
        
        # Update status bar
        n_plans = len(self.comparison.comparison_plans) + 1
        n_structures = len(self.comparison.get_structure_ids())
        self.status_bar.showMessage(f"Comparing {n_plans} plans with {n_structures} common structures")
    
    def _update_plan_table(self):
        """Update the plans table with current plan data."""
        self.plans_table.setRowCount(0)
        
        # Add reference plan
        row = self.plans_table.rowCount()
        self.plans_table.insertRow(row)
        
        name_item = QTableWidgetItem(self.reference_plan.name)
        type_item = QTableWidgetItem(str(self.reference_plan.technique) if hasattr(self.reference_plan, "technique") else "Unknown")
        ref_item = QTableWidgetItem("Yes")
        ref_item.setTextAlignment(Qt.AlignCenter)
        ref_item.setFlags(ref_item.flags() & ~Qt.ItemIsEnabled)  # Disable editing
        
        self.plans_table.setItem(row, 0, name_item)
        self.plans_table.setItem(row, 1, type_item)
        self.plans_table.setItem(row, 2, ref_item)
        
        # Add data to the item for identification
        name_item.setData(Qt.UserRole, self.reference_plan.id)
        
        # Add comparison plans
        for plan_id, plan in self.comparison.comparison_plans.items():
            row = self.plans_table.rowCount()
            self.plans_table.insertRow(row)
            
            name_item = QTableWidgetItem(plan.name)
            type_item = QTableWidgetItem(str(plan.technique) if hasattr(plan, "technique") else "Unknown")
            ref_item = QTableWidgetItem("No")
            ref_item.setTextAlignment(Qt.AlignCenter)
            
            self.plans_table.setItem(row, 0, name_item)
            self.plans_table.setItem(row, 1, type_item)
            self.plans_table.setItem(row, 2, ref_item)
            
            # Add data to the item for identification
            name_item.setData(Qt.UserRole, plan_id)
        
        # Update dose diff plan combos
        self.dose_diff_ref_combo.clear()
        self.dose_diff_eval_combo.clear()
        
        self.dose_diff_ref_combo.addItem(self.reference_plan.name, self.reference_plan.id)
        self.dose_diff_eval_combo.addItem(self.reference_plan.name, self.reference_plan.id)
        
        for plan_id, plan in self.comparison.comparison_plans.items():
            self.dose_diff_ref_combo.addItem(plan.name, plan_id)
            self.dose_diff_eval_combo.addItem(plan.name, plan_id)
        
        # Set current selections
        ref_index = self.dose_diff_ref_combo.findData(self.dose_diff_ref_plan_id)
        if ref_index >= 0:
            self.dose_diff_ref_combo.setCurrentIndex(ref_index)
        
        if self.dose_diff_eval_plan_id:
            eval_index = self.dose_diff_eval_combo.findData(self.dose_diff_eval_plan_id)
            if eval_index >= 0:
                self.dose_diff_eval_combo.setCurrentIndex(eval_index)
        elif self.comparison.comparison_plans:
            # Set first comparison plan as default
            first_plan_id = next(iter(self.comparison.comparison_plans.keys()))
            eval_index = self.dose_diff_eval_combo.findData(first_plan_id)
            if eval_index >= 0:
                self.dose_diff_eval_combo.setCurrentIndex(eval_index)
                self.dose_diff_eval_plan_id = first_plan_id
    
    def _update_structure_list(self):
        """Update the structures tree with current data."""
        self.structures_tree.clear()
        
        structure_ids = self.comparison.get_structure_ids()
        for structure_id in structure_ids:
            structure = self.reference_plan.get_structure(structure_id)
            if structure:
                item = StructureTreeItem(structure_id, structure.name)
                self.structures_tree.addTopLevelItem(item)
                
                # Select first structure as default if none selected
                if not self.current_structure_id:
                    self.current_structure_id = structure_id
                    item.setSelected(True)
    
    def _update_dvh(self):
        """Update the DVH widget with current data."""
        self.dvh_widget.clear_all_curves()
        
        # Set DVH display options
        self.dvh_widget.set_display_type(
            display_type=self.dvh_display_type,
            volume_type=self.volume_type,
            dose_type=self.dose_type
        )
        
        # Get selected structures
        selected_structures = []
        for i in range(self.structures_tree.topLevelItemCount()):
            item = self.structures_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                selected_structures.append(item.structure_id)
        
        if not selected_structures:
            return
        
        # Ensure consistent colors across plans
        structure_colors = {}
        for i, structure_id in enumerate(selected_structures):
            # Get color from predefined list or generate one
            color = self._get_structure_color(structure_id, i)
            structure_colors[structure_id] = color
        
        # Add curves for reference plan
        for structure_id in selected_structures:
            dvh_data = self.comparison.get_dvh_data(self.reference_plan.id, structure_id)
            if dvh_data:
                structure = self.reference_plan.get_structure(structure_id)
                label = f"{structure.name} ({self.reference_plan.name})"
                self.dvh_widget.add_dvh_curve(
                    dvh_data=dvh_data,
                    label=label,
                    plan_id=self.reference_plan.id,
                    color=structure_colors[structure_id],
                    dashed=False
                )
        
        # Add curves for comparison plans
        for plan_id, plan in self.comparison.comparison_plans.items():
            for structure_id in selected_structures:
                dvh_data = self.comparison.get_dvh_data(plan_id, structure_id)
                if dvh_data:
                    structure = plan.get_structure(structure_id)
                    label = f"{structure.name} ({plan.name})"
                    self.dvh_widget.add_dvh_curve(
                        dvh_data=dvh_data,
                        label=label,
                        plan_id=plan_id,
                        color=structure_colors[structure_id],
                        dashed=True
                    )
        
        # Set prescription dose if available
        if hasattr(self.reference_plan, "prescription") and self.reference_plan.prescription:
            self.dvh_widget.set_prescription_dose(self.reference_plan.prescription.dose)
    
    def _update_metrics_table(self):
        """Update the metrics table with current data."""
        self.metrics_table.clear()
        
        # Add reference plan
        self.metrics_table.add_plan(self.reference_plan, is_reference=True)
        
        # Add comparison plans
        for plan_id, plan in self.comparison.comparison_plans.items():
            self.metrics_table.add_plan(plan)
        
        # Add structures
        structure_ids = self.comparison.get_structure_ids()
        for structure_id in structure_ids:
            structure = self.reference_plan.get_structure(structure_id)
            if structure:
                self.metrics_table.add_structure(structure_id)
        
        # Refresh to show data
        self.metrics_table.refresh()
    
    def _update_dose_diff_view(self):
        """Update the dose difference view."""
        display_mode = self.dose_diff_display_combo.currentText().lower()
        
        # Check if both plans are selected
        if not self.dose_diff_ref_plan_id or not self.dose_diff_eval_plan_id:
            return
        
        # Get plans
        if self.dose_diff_ref_plan_id == self.reference_plan.id:
            ref_plan = self.reference_plan
        else:
            ref_plan = self.comparison.comparison_plans.get(self.dose_diff_ref_plan_id)
        
        if self.dose_diff_eval_plan_id == self.reference_plan.id:
            eval_plan = self.reference_plan
        else:
            eval_plan = self.comparison.comparison_plans.get(self.dose_diff_eval_plan_id)
        
        if not ref_plan or not eval_plan:
            return
        
        # Set plans in dose comparison widget
        self.dose_comparison_widget.set_plans(
            reference_plan=ref_plan,
            comparison_plan=eval_plan,
            display_mode=display_mode
        )
    
    def _update_goals_table(self):
        """Update the clinical goals table."""
        self.goals_table.setRowCount(0)
        
        # Get protocol
        protocol = self.comparison.protocol
        if not protocol:
            self.goals_table.setEnabled(False)
            return
            
        self.goals_table.setEnabled(True)
        
        # Set up columns
        self.goals_table.setColumnCount(0)
        self.goals_table.setColumnCount(3 + len(self.comparison.comparison_plans))
        
        headers = ["Goal", "Structure"]
        headers.append(self.reference_plan.name)
        
        for plan_id, plan in self.comparison.comparison_plans.items():
            headers.append(plan.name)
        
        self.goals_table.setHorizontalHeaderLabels(headers)
        
        # Get all goals
        for goal in protocol.goals:
            row = self.goals_table.rowCount()
            self.goals_table.insertRow(row)
            
            # Goal description
            goal_item = QTableWidgetItem(goal.description)
            self.goals_table.setItem(row, 0, goal_item)
            
            # Structure name
            structure = self.reference_plan.get_structure(goal.structure_id)
            if structure:
                structure_item = QTableWidgetItem(structure.name)
            else:
                structure_item = QTableWidgetItem(goal.structure_id)
            self.goals_table.setItem(row, 1, structure_item)
            
            # Reference plan result
            ref_results = self.comparison.get_goal_results(self.reference_plan.id)
            if goal.id in ref_results:
                result = ref_results[goal.id]
                value_item = QTableWidgetItem(f"{result.value:.2f} {result.unit}")
                
                # Set color based on pass/fail
                if result.passed:
                    value_item.setBackground(QColor(200, 255, 200))  # Light green
                else:
                    value_item.setBackground(QColor(255, 200, 200))  # Light red
            else:
                value_item = QTableWidgetItem("N/A")
            
            self.goals_table.setItem(row, 2, value_item)
            
            # Comparison plan results
            col = 3
            for plan_id, plan in self.comparison.comparison_plans.items():
                plan_results = self.comparison.get_goal_results(plan_id)
                
                if goal.id in plan_results:
                    result = plan_results[goal.id]
                    value_item = QTableWidgetItem(f"{result.value:.2f} {result.unit}")
                    
                    # Set color based on pass/fail
                    if result.passed:
                        value_item.setBackground(QColor(200, 255, 200))  # Light green
                    else:
                        value_item.setBackground(QColor(255, 200, 200))  # Light red
                    
                    # Compare with reference plan
                    if goal.id in ref_results:
                        ref_result = ref_results[goal.id]
                        if abs(ref_result.value) > 1e-6:  # Avoid division by zero
                            diff_percent = ((result.value - ref_result.value) / abs(ref_result.value)) * 100
                            diff_item = QTableWidgetItem(f"{diff_percent:+.1f}%")
                            
                            # Set color based on improvement or not
                            is_improvement = (diff_percent > 0 and goal.is_higher_better()) or \
                                            (diff_percent < 0 and not goal.is_higher_better())
                            
                            if is_improvement:
                                diff_item.setBackground(QColor(200, 255, 200))  # Light green
                            elif diff_percent != 0:
                                diff_item.setBackground(QColor(255, 200, 200))  # Light red
                            
                            self.goals_table.setItem(row, col + 1, diff_item)
                else:
                    value_item = QTableWidgetItem("N/A")
                
                self.goals_table.setItem(row, col, value_item)
                col += 1
    
    def _on_structure_selection_changed(self):
        """Handle structure selection change in the tree."""
        self._update_dvh()
    
    def _on_dvh_type_changed(self, dvh_type):
        """Handle change of DVH type (cumulative/differential)."""
        self.dvh_display_type = dvh_type
        self._update_dvh()
    
    def _on_volume_type_changed(self, volume_type):
        """Handle change of volume type (relative/absolute)."""
        self.volume_type = volume_type
        self._update_dvh()
    
    def _on_dose_type_changed(self, dose_type):
        """Handle change of dose type (absolute/relative)."""
        self.dose_type = dose_type
        self._update_dvh()
    
    def _on_dose_diff_plan_changed(self):
        """Handle change of plans in dose difference view."""
        self.dose_diff_ref_plan_id = self.dose_diff_ref_combo.currentData()
        self.dose_diff_eval_plan_id = self.dose_diff_eval_combo.currentData()
        self._update_dose_diff_view()
    
    def _on_add_plan(self):
        """Handle add plan button click."""
        # This would typically open a dialog to select a plan
        # For now, let's just show a message
        QMessageBox.information(
            self,
            "Add Plan",
            "To add a plan, this would open a plan selection dialog.\n"
            "In this demo, please use the add_comparison_plan() method."
        )
    
    def _on_remove_plan(self):
        """Handle remove plan button click."""
        # Get selected plan
        selected_items = self.plans_table.selectedItems()
        if not selected_items:
            return
        
        # Get plan ID from the first column of the selected row
        row = selected_items[0].row()
        plan_id = self.plans_table.item(row, 0).data(Qt.UserRole)
        
        # Cannot remove reference plan
        if plan_id == self.reference_plan.id:
            QMessageBox.warning(
                self,
                "Cannot Remove Reference Plan",
                "The reference plan cannot be removed from the comparison."
            )
            return
        
        # Remove the plan
        self.comparison.remove_comparison_plan(plan_id)
        
        # Update UI
        self._update_ui()
    
    def _on_select_protocol(self):
        """Handle protocol selection button click."""
        if not self.protocol_combo.count():
            QMessageBox.warning(
                self,
                "No Protocols Available",
                "No clinical protocols are available. Please add protocols to the protocols directory."
            )
            return
        
        # Get selected protocol name
        protocol_name = self.protocol_combo.currentText()
        
        # Load protocol
        try:
            from quangtps.evaluation.protocol_manager import ProtocolManager
            manager = ProtocolManager()
            protocol = manager.load_protocol(protocol_name)
            
            if protocol:
                self.comparison.set_clinical_protocol(protocol)
                self._update_goals_table()
                
                QMessageBox.information(
                    self,
                    "Protocol Applied",
                    f"The protocol '{protocol_name}' has been applied to all plans."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Protocol Not Found",
                    f"The protocol '{protocol_name}' could not be loaded."
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Protocol Error",
                f"Error loading protocol: {str(e)}"
            )
    
    def _on_apply_protocol(self):
        """Handle apply protocol button click."""
        self._on_select_protocol()
    
    def _on_export_report(self):
        """Handle export report button click."""
        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report",
            os.path.expanduser("~/plan_comparison_report.pdf"),
            "PDF Files (*.pdf)"
        )
        
        if not file_path:
            return
        
        # Generate report
        report_path = self.comparison.generate_comparison_report(file_path)
        
        if report_path:
            QMessageBox.information(
                self,
                "Report Generated",
                f"Report has been saved to:\n{report_path}"
            )
        else:
            QMessageBox.warning(
                self,
                "Report Error",
                "There was an error generating the report."
            )
    
    def _get_structure_color(self, structure_id, index):
        """
        Get a color for a structure.
        
        Args:
            structure_id: ID of the structure
            index: Index for fallback color selection
            
        Returns:
            Color as hex string
        """
        # Define colors for common structure types
        structure_colors = {
            "ptv": "#FF0000",  # Red
            "ctv": "#FFA500",  # Orange
            "gtv": "#FF4500",  # OrangeRed
            "lung": "#ADD8E6",  # Light blue
            "heart": "#FF69B4",  # Pink
            "cord": "#FFFF00",  # Yellow
            "esophagus": "#9370DB",  # Medium purple
            "liver": "#8B4513",  # SaddleBrown
            "kidney": "#20B2AA",  # Light sea green
            "brain": "#D8BFD8",  # Thistle
            "brainstem": "#DDA0DD",  # Plum
        }
        
        # Try to match structure ID with known types
        for key, color in structure_colors.items():
            if key in structure_id.lower():
                return color
        
        # Fallback to color list
        color_list = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        
        return color_list[index % len(color_list)]
    
    def _populate_protocols(self):
        """Populate the protocols combo box with available protocols."""
        self.protocol_combo.clear()
        
        try:
            from quangtps.evaluation.protocol_manager import ProtocolManager
            manager = ProtocolManager()
            protocols = manager.get_available_protocols()
            
            for protocol_name in protocols:
                self.protocol_combo.addItem(protocol_name)
        except Exception as e:
            logger.error(f"Error loading protocols: {str(e)}")
    
    def add_comparison_plan(self, plan: Plan):
        """
        Add a plan to the comparison.
        
        Args:
            plan: The plan to add
        """
        success = self.comparison.add_comparison_plan(plan)
        
        if success:
            # Update UI
            self._update_ui()
            
            # Set as evaluation plan for dose difference
            self.dose_diff_eval_plan_id = plan.id
            eval_index = self.dose_diff_eval_combo.findData(plan.id)
            if eval_index >= 0:
                self.dose_diff_eval_combo.setCurrentIndex(eval_index) 