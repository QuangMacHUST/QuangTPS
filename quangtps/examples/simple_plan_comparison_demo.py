"""
Simple Plan Comparison Demo

A self-contained demo that demonstrates plan comparison functionality
without relying on other QuangTPS modules.
"""

import sys
import os
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTabWidget, QWidget, QMessageBox,
    QSplitter, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QColor

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


# Simple DVH Data classes
class DVHCurve:
    """Class representing a DVH curve."""
    
    def __init__(self, dose_bins, volume_bins, is_cumulative=True):
        self.dose_bins = dose_bins
        self.volume_bins = volume_bins
        self.is_cumulative = is_cumulative


class DVHData:
    """Class representing DVH data for a structure."""
    
    def __init__(self, structure_id, structure_name, structure_volume, max_dose, mean_dose, min_dose):
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.structure_volume = structure_volume
        self.max_dose = max_dose
        self.mean_dose = mean_dose
        self.min_dose = min_dose
        
        # Create sample DVH curves
        self.cumulative = self._create_sample_curve(is_cumulative=True)
        self.differential = self._create_sample_curve(is_cumulative=False)
    
    def _create_sample_curve(self, is_cumulative):
        """Create a sample curve."""
        dose_bins = np.linspace(0, 60, 100)
        
        if "PTV" in self.structure_name:
            # Create a PTV-like curve
            if is_cumulative:
                volume_bins = 100 * np.exp(-0.1 * dose_bins)
            else:
                volume_bins = np.zeros_like(dose_bins)
                volume_bins[30:60] = 100 * np.exp(-0.5 * (dose_bins[30:60] - 50)**2 / 100)
        else:
            # Create an OAR-like curve
            if is_cumulative:
                volume_bins = 100 * np.exp(-0.2 * dose_bins)
            else:
                volume_bins = np.zeros_like(dose_bins)
                volume_bins[0:30] = 100 * np.exp(-0.5 * (dose_bins[0:30] - 10)**2 / 50)
        
        return DVHCurve(dose_bins, volume_bins, is_cumulative)
    
    def get_volume_at_dose(self, dose):
        """Get the volume percentage at a specified dose."""
        # Simple lookup with linear interpolation
        dose_bins = self.cumulative.dose_bins
        volume_bins = self.cumulative.volume_bins
        
        # Find dose index
        for i in range(len(dose_bins)):
            if dose_bins[i] >= dose:
                if i == 0:
                    return volume_bins[0]
                else:
                    # Linear interpolation
                    d0, d1 = dose_bins[i-1], dose_bins[i]
                    v0, v1 = volume_bins[i-1], volume_bins[i]
                    return v0 + (v1 - v0) * (dose - d0) / (d1 - d0)
        
        return 0.0
    
    def get_dose_at_volume(self, volume):
        """Get the dose at a specified volume percentage."""
        # Simple lookup with linear interpolation
        dose_bins = self.cumulative.dose_bins
        volume_bins = self.cumulative.volume_bins
        
        # Find volume index (DVH curve is descending in volume)
        for i in range(len(volume_bins)):
            if volume_bins[i] <= volume:
                if i == 0:
                    return dose_bins[0]
                else:
                    # Linear interpolation
                    v0, v1 = volume_bins[i-1], volume_bins[i]
                    d0, d1 = dose_bins[i-1], dose_bins[i]
                    return d0 + (d1 - d0) * (volume - v0) / (v1 - v0)
        
        return dose_bins[-1]


# Simple Structure class
class Structure:
    """Class representing an anatomical structure."""
    
    def __init__(self, id, name, type="PTV"):
        self.id = id
        self.name = name
        self.type = type
        
        # Assign some reasonable defaults
        if "PTV" in name:
            self.volume = 100.0
            self.color = (1.0, 0.0, 0.0)  # Red
        elif "Lung" in name:
            self.volume = 1500.0
            self.color = (0.0, 0.7, 1.0)  # Light blue
        elif "Heart" in name:
            self.volume = 800.0
            self.color = (1.0, 0.0, 0.5)  # Pink
        elif "Cord" in name:
            self.volume = 80.0
            self.color = (1.0, 1.0, 0.0)  # Yellow
        else:
            self.volume = 200.0
            self.color = (0.0, 0.5, 0.0)  # Green


# Simple Plan class
class Plan:
    """Class representing a treatment plan."""
    
    def __init__(self, id, name, prescription_dose=60.0):
        self.id = id
        self.name = name
        self.structures = {}
        
        # Simple prescription object
        self.prescription = SimpleObject()
        self.prescription.dose = prescription_dose
        self.prescription.fractions = 30
    
    def add_structure(self, structure):
        """Add a structure to the plan."""
        self.structures[structure.id] = structure
    
    def get_structure(self, structure_id):
        """Get a structure by ID."""
        return self.structures.get(structure_id)
    
    def get_structures(self):
        """Get all structures."""
        return list(self.structures.values())
    
    def get_dvh_data(self, structure_id):
        """Get DVH data for a structure."""
        structure = self.get_structure(structure_id)
        if not structure:
            return None
        
        # Create DVH data with reasonable values based on structure type
        if "PTV" in structure.name:
            max_dose = 1.05 * self.prescription.dose
            mean_dose = 1.00 * self.prescription.dose
            min_dose = 0.95 * self.prescription.dose
        elif "Lung" in structure.name:
            max_dose = 0.8 * self.prescription.dose
            mean_dose = 0.3 * self.prescription.dose
            min_dose = 0.1 * self.prescription.dose
        elif "Heart" in structure.name:
            max_dose = 0.7 * self.prescription.dose
            mean_dose = 0.2 * self.prescription.dose
            min_dose = 0.1 * self.prescription.dose
        elif "Cord" in structure.name:
            max_dose = 0.6 * self.prescription.dose
            mean_dose = 0.25 * self.prescription.dose
            min_dose = 0.05 * self.prescription.dose
        else:
            max_dose = 0.9 * self.prescription.dose
            mean_dose = 0.5 * self.prescription.dose
            min_dose = 0.1 * self.prescription.dose
        
        return DVHData(
            structure_id=structure.id,
            structure_name=structure.name,
            structure_volume=structure.volume,
            max_dose=max_dose,
            mean_dose=mean_dose,
            min_dose=min_dose
        )
    
    def get_dose(self):
        """Get the 3D dose distribution."""
        # Create a simple 3D dose distribution (50x50x30)
        dose = np.zeros((50, 50, 30))
        
        # Create a spherical high-dose region in the center
        center = np.array([25, 25, 15])
        for x in range(dose.shape[0]):
            for y in range(dose.shape[1]):
                for z in range(dose.shape[2]):
                    # Distance from center
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                    
                    # Dose falls off with distance
                    dose[x, y, z] = self.prescription.dose * np.exp(-0.1 * dist)
        
        return dose


# Simple object for holding attributes
class SimpleObject:
    pass


# DVH Widget for displaying DVH curves
class DVHWidget(QWidget):
    """Widget for displaying DVH curves."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup state
        self.curves = {}  # structure_id -> curve_info
        
        # Initialize UI
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Add to layout
        layout.addWidget(self.canvas)
        
        # Setup axes
        self.ax.set_xlabel("Dose (Gy)")
        self.ax.set_ylabel("Volume (%)")
        self.ax.set_xlim([0, 70])
        self.ax.set_ylim([0, 105])
        self.ax.grid(True)
    
    def add_curve(self, dvh_data, label=None, color=None, linestyle='-'):
        """Add a DVH curve."""
        if not label:
            label = dvh_data.structure_name
        
        # Store in dictionary
        curve_id = dvh_data.structure_id
        self.curves[curve_id] = {
            'dvh_data': dvh_data,
            'label': label,
            'color': color,
            'linestyle': linestyle
        }
    
    def clear(self):
        """Clear all curves."""
        self.curves.clear()
        self.ax.clear()
        self.ax.set_xlabel("Dose (Gy)")
        self.ax.set_ylabel("Volume (%)")
        self.ax.set_xlim([0, 70])
        self.ax.set_ylim([0, 105])
        self.ax.grid(True)
        self.canvas.draw()
    
    def refresh(self):
        """Refresh the plot."""
        self.ax.clear()
        
        # Plot each curve
        for curve_id, curve_info in self.curves.items():
            dvh_data = curve_info['dvh_data']
            label = curve_info['label']
            color = curve_info['color']
            linestyle = curve_info['linestyle']
            
            # Get data from cumulative curve
            dose = dvh_data.cumulative.dose_bins
            volume = dvh_data.cumulative.volume_bins
            
            # Plot
            self.ax.plot(dose, volume, label=label, color=color, linestyle=linestyle)
        
        # Update axes
        self.ax.set_xlabel("Dose (Gy)")
        self.ax.set_ylabel("Volume (%)")
        self.ax.set_xlim([0, 70])
        self.ax.set_ylim([0, 105])
        self.ax.grid(True)
        
        # Add legend if we have curves
        if self.curves:
            self.ax.legend()
        
        # Redraw
        self.canvas.draw()


# Plan Comparison Dialog
class PlanComparisonDialog(QDialog):
    """Dialog for comparing multiple plans."""
    
    def __init__(self, reference_plan, parent=None):
        super().__init__(parent)
        
        # Setup state
        self.reference_plan = reference_plan
        self.comparison_plans = {}
        
        # Set window properties
        self.setWindowTitle("Plan Comparison")
        self.resize(900, 600)
        
        # Create UI
        self._create_ui()
        
        # Add reference plan
        self._add_plan(reference_plan, is_reference=True)
    
    def _create_ui(self):
        """Create the UI."""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Add a splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left side - tree view of plans and structures
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Plans & Structures")
        self.tree.setMinimumWidth(200)
        splitter.addWidget(self.tree)
        
        # Create plan and structure roots
        self.plans_root = QTreeWidgetItem(self.tree, ["Plans"])
        self.structures_root = QTreeWidgetItem(self.tree, ["Structures"])
        
        # Right side - tab widget
        tab_widget = QTabWidget()
        splitter.addWidget(tab_widget)
        
        # DVH tab
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)
        
        # Add DVH widget
        self.dvh_widget = DVHWidget()
        dvh_layout.addWidget(self.dvh_widget)
        
        tab_widget.addTab(dvh_tab, "DVH Comparison")
        
        # Set splitter proportions
        splitter.setSizes([200, 700])
        
        # Add close button
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # Connect tree signals
        self.tree.itemChanged.connect(self._on_tree_item_changed)
    
    def _add_plan(self, plan, is_reference=False):
        """Add a plan to the comparison."""
        # Skip if already added
        if plan.id in self.comparison_plans and not is_reference:
            return
        
        # Add to comparison plans
        if not is_reference:
            self.comparison_plans[plan.id] = plan
        
        # Add to UI
        plan_item = QTreeWidgetItem(self.plans_root, [plan.name])
        plan_item.setData(0, Qt.UserRole, plan.id)
        plan_item.setCheckState(0, Qt.Checked)
        
        # Add structures if first plan or reference
        if not self.structures_root.childCount() or is_reference:
            # Add structures
            for structure in plan.get_structures():
                struct_item = QTreeWidgetItem(self.structures_root, [structure.name])
                struct_item.setData(0, Qt.UserRole, structure.id)
                struct_item.setCheckState(0, Qt.Checked)
        
        # Expand tree items
        self.plans_root.setExpanded(True)
        self.structures_root.setExpanded(True)
        
        # Update DVH
        self._update_dvh()
    
    def _on_tree_item_changed(self, item, column):
        """Handle tree item changes."""
        # Update DVH when item selection changes
        self._update_dvh()
    
    def _update_dvh(self):
        """Update the DVH display."""
        # Clear existing data
        self.dvh_widget.clear()
        
        # Get checked structures
        structure_ids = []
        for i in range(self.structures_root.childCount()):
            struct_item = self.structures_root.child(i)
            if struct_item.checkState(0) == Qt.Checked:
                structure_ids.append(struct_item.data(0, Qt.UserRole))
        
        # Get checked plans
        plan_ids = []
        for i in range(self.plans_root.childCount()):
            plan_item = self.plans_root.child(i)
            if plan_item.checkState(0) == Qt.Checked:
                plan_ids.append(plan_item.data(0, Qt.UserRole))
        
        # Add curves for reference plan if checked
        if self.reference_plan.id in plan_ids:
            for structure_id in structure_ids:
                dvh_data = self.reference_plan.get_dvh_data(structure_id)
                if dvh_data:
                    # Add curve
                    structure = self.reference_plan.get_structure(structure_id)
                    self.dvh_widget.add_curve(
                        dvh_data=dvh_data,
                        label=f"{structure.name} ({self.reference_plan.name})",
                        color=structure.color,
                        linestyle='-'  # Solid line for reference
                    )
        
        # Add curves for comparison plans if checked
        for plan_id in plan_ids:
            if plan_id == self.reference_plan.id:
                continue  # Skip reference plan (already added)
                
            plan = self.comparison_plans.get(plan_id)
            if not plan:
                continue
                
            for structure_id in structure_ids:
                dvh_data = plan.get_dvh_data(structure_id)
                if dvh_data:
                    # Add curve
                    structure = plan.get_structure(structure_id)
                    self.dvh_widget.add_curve(
                        dvh_data=dvh_data,
                        label=f"{structure.name} ({plan.name})",
                        color=structure.color,
                        linestyle='--'  # Dashed line for comparison
                    )
        
        # Refresh display
        self.dvh_widget.refresh()


def create_sample_plan(id, name, prescription_dose, num_beams=4):
    """Create a sample plan for testing."""
    plan = Plan(id, name, prescription_dose)
    
    # Add structures
    ptv = Structure("PTV", "PTV")
    lung_left = Structure("LUNG_L", "Left Lung", "OAR")
    lung_right = Structure("LUNG_R", "Right Lung", "OAR")
    heart = Structure("HEART", "Heart", "OAR")
    cord = Structure("CORD", "Spinal Cord", "OAR")
    
    plan.add_structure(ptv)
    plan.add_structure(lung_left)
    plan.add_structure(lung_right)
    plan.add_structure(heart)
    plan.add_structure(cord)
    
    return plan


def main():
    """Main function."""
    # Create application
    app = QApplication(sys.argv)
    
    # Create sample plans
    plan1 = create_sample_plan("PLAN1", "3D Plan", 50.0)
    plan2 = create_sample_plan("PLAN2", "IMRT Plan", 60.0)
    plan3 = create_sample_plan("PLAN3", "VMAT Plan", 54.0)
    
    # Create and show dialog with VMAT as reference
    dialog = PlanComparisonDialog(plan3)
    
    # Add other plans for comparison
    dialog._add_plan(plan1)
    dialog._add_plan(plan2)
    
    dialog.exec_()


if __name__ == "__main__":
    main() 