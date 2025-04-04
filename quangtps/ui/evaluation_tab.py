"""
Plan Evaluation Tab Module

This module implements a tab for DVH-based evaluation of treatment plans
and integrates plan quality assessment against clinical protocols.
"""

import os
import logging
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
        QSplitter, QTabWidget, QFrame, QPushButton, QToolBar,
        QAction, QComboBox, QSlider, QCheckBox, QSpinBox,
        QTreeWidget, QTreeWidgetItem, QMenu, QHeaderView,
        QTableWidget, QTableWidgetItem, QFileDialog, QGroupBox,
        QMessageBox, QListWidget, QAbstractItemView, QFormLayout,
        QDoubleSpinBox, QProgressBar, QSizePolicy, QStatusBar,
        QToolButton
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtGui import QIcon, QColor, QBrush, QCursor
except ImportError as e:
    logging.error(f"Unable to import PyQt5 components: {e}")
    # Define placeholder classes if needed for type checking
    class QWidget: pass
    class QVBoxLayout: pass
    class QHBoxLayout: pass
    class QSplitter: pass
    # And other required classes...

try:
    from quangtps.imaging.image import Image
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure import Structure
    from quangtps.beams.beam_set import BeamSet
except ImportError as e:
    logging.warning(f"Unable to import core components: {e}")
    # Define placeholders if needed

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from quangtps.dose.dose_calculator import DoseCalculator
else:
    # Define a placeholder for type hints
    class DoseCalculator:
        """Placeholder for DoseCalculator class."""
        def get_structure_set(self): pass
        def get_beam_set(self): pass
        def get_prescription_dose(self): pass

try:
    from quangtps.evaluation.plan_evaluation import PlanEvaluation, DVHCalculator
except ImportError as e:
    logging.warning(f"Unable to import PlanEvaluation: {e}")
    DVHCalculator = None
    PlanEvaluation = None

# Try to import plan quality widget
try:
    from quangtps.ui.plan_quality_widget import PlanQualityWidget
except ImportError:
    logging.warning("Could not import PlanQualityWidget")
    # Create a placeholder if import fails
    class PlanQualityWidget(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Plan Quality module not available"))
            
        def setPlanEvaluation(self, plan_evaluation):
            pass

# Try to import protocol dialog
try:
    from quangtps.ui.dialogs.protocol_dialog import ClinicalProtocolDialog
except ImportError as e:
    logging.warning(f"Unable to import ClinicalProtocolDialog, protocol selection will be disabled: {e}")
    ClinicalProtocolDialog = None

logger = logging.getLogger(__name__)

class DVHCanvas(FigureCanvas):
    """Matplotlib canvas for displaying DVH plots."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """Initialize the canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Setup
        self.axes.set_xlabel('Dose (Gy)')
        self.axes.set_ylabel('Volume (%)')
        self.axes.set_title('Cumulative Dose-Volume Histogram')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Set initial limits
        self.axes.set_xlim(0, 80)
        self.axes.set_ylim(0, 105)
        
        self.fig.tight_layout()
        
    def clear_plot(self):
        """Clear all plots on the canvas."""
        self.axes.clear()
        
        # Reset labels and title
        self.axes.set_xlabel('Dose (Gy)')
        self.axes.set_ylabel('Volume (%)')
        self.axes.set_title('Cumulative Dose-Volume Histogram')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        self.draw()
        
    def plot_dvh(self, dose_bins, volume_values, structure_name, color):
        """Plot a DVH curve."""
        self.axes.plot(dose_bins, volume_values, label=structure_name, color=color)
        self.axes.legend(loc='upper right')
        self.draw()
        
    def set_prescription_line(self, prescription):
        """Add a vertical line at the prescription dose."""
        if prescription > 0:
            self.axes.axvline(x=prescription, color='r', linestyle='--', linewidth=1)
            self.draw()
        
    def save_figure(self, filename):
        """Save the figure to a file."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        
class EvaluationTab(QWidget):
    """
    Tab for displaying plan evaluation, including DVH and structure statistics.
    """
    
    def __init__(self, parent=None):
        """Initialize the evaluation tab."""
        super().__init__(parent)
        
        # Initialize data
        self.image = None
        self.structure_set = None
        self.beam_set = None
        self.dose_calculator = None
        self.dvh_calculator = DVHCalculator()
        self.plan_evaluation = PlanEvaluation()
        
        # Initialize UI
        self.init_ui()
        
        logger.info("Evaluation tab initialized")
        
    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create splitter for resizable sections
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Structure selection and DVH settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Plan selection
        plan_group = QGroupBox("Plan")
        plan_layout = QVBoxLayout(plan_group)
        
        # Combo boxes for patient, image and structure set
        self.plan_combo = QComboBox()
        self.plan_combo.currentIndexChanged.connect(self.on_plan_changed)
        plan_layout.addWidget(self.plan_combo)
        
        left_layout.addWidget(plan_group)
        
        # Structure selection
        structure_group = QGroupBox("Structures")
        structure_layout = QVBoxLayout(structure_group)
        
        self.structure_list = QListWidget()
        self.structure_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.structure_list.itemSelectionChanged.connect(self.on_structure_selection_changed)
        structure_layout.addWidget(self.structure_list)
        
        # Add all button
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.on_select_all_clicked)
        structure_layout.addWidget(self.select_all_button)
        
        left_layout.addWidget(structure_group)
        
        # DVH settings
        dvh_group = QGroupBox("DVH Settings")
        dvh_layout = QFormLayout(dvh_group)
        
        # Normalization point
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(["None", "Prescription", "Max Dose", "Mean Dose"])
        self.normalization_combo.currentIndexChanged.connect(self.on_normalization_changed)
        dvh_layout.addRow("Normalization:", self.normalization_combo)
        
        # Prescription dose
        self.prescription_spinbox = QDoubleSpinBox()
        self.prescription_spinbox.setRange(0, 10000)
        self.prescription_spinbox.setSuffix(" cGy")
        self.prescription_spinbox.setValue(7000)
        self.prescription_spinbox.valueChanged.connect(self.on_prescription_changed)
        dvh_layout.addRow("Prescription:", self.prescription_spinbox)
        
        # Export button
        self.export_button = QPushButton("Export DVH Data")
        self.export_button.clicked.connect(self.on_export_dvh)
        dvh_layout.addWidget(self.export_button)
        
        left_layout.addWidget(dvh_group)
        
        # Report button
        self.report_button = QPushButton("Generate Report")
        self.report_button.clicked.connect(self.on_generate_report)
        left_layout.addWidget(self.report_button)
        
        # Add spacer to push everything up
        left_layout.addStretch()
        
        # Right side of splitter (multiple tabs)
        right_panel = QTabWidget()
        
        # Tab 1: DVH with Statistics
        dvh_tab = QSplitter(Qt.Vertical)
        
        # Upper part - DVH plot
        self.dvh_canvas = DVHCanvas(self)
        dvh_tab.addWidget(self.dvh_canvas)
        
        # Lower part - Statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(6)
        self.stats_table.setHorizontalHeaderLabels(["Structure", "Min", "Max", "Mean", "D95", "V95"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        dvh_tab.addWidget(self.stats_table)
        
        # Set default sizes for the splitter
        dvh_tab.setSizes([400, 200])
        
        # Tab 2: Plan Quality
        self.plan_quality_tab = QWidget()
        plan_quality_layout = QVBoxLayout(self.plan_quality_tab)
        
        # Protocol selection and management
        protocol_group = QGroupBox("Clinical Protocol")
        protocol_layout = QHBoxLayout(protocol_group)
        
        self.protocol_combo = QComboBox()
        protocol_layout.addWidget(self.protocol_combo, 1)
        
        self.select_protocol_button = QPushButton("Select Protocol")
        self.select_protocol_button.clicked.connect(self.on_select_protocol)
        protocol_layout.addWidget(self.select_protocol_button)
        
        plan_quality_layout.addWidget(protocol_group)
        
        # Plan quality widget
        self.plan_quality_widget = PlanQualityWidget()
        plan_quality_layout.addWidget(self.plan_quality_widget)
        
        # Add the tabs
        right_panel.addTab(dvh_tab, "DVH Analysis")
        right_panel.addTab(self.plan_quality_tab, "Plan Quality")
        
        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        
        # Set default sizes
        main_splitter.setSizes([200, 800])
        
        # Add splitter to main layout
        layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        
        # Initialize empty data
        self.current_plan = None
        self.current_image = None
        self.current_structure_set = None
        self.current_dose = None
        self.plan_evaluator = None
        
        # Set initial status
        self.status_bar.showMessage("Ready")
        
        # Load available protocols
        self.load_protocols()
        
    def set_dose_calculator(self, dose_calculator: "DoseCalculator"):
        """
        Set the dose calculator for evaluation.
        
        Args:
            dose_calculator: Dose calculator containing calculated dose
        """
        if dose_calculator is None:
            return
            
        self.dose_calculator = dose_calculator
        self.structure_set = dose_calculator.get_structure_set()
        self.beam_set = dose_calculator.get_beam_set()
        
        # Update the plan evaluation with the dose calculator
        self.plan_evaluation.set_dose_calculator(dose_calculator)
        
        # Also update the plan quality widget
        self.plan_quality_widget.setPlanEvaluation(self.plan_evaluation)
        
        # Update the UI
        self._update_structure_list()
        
        logger.info("Dose calculator set in evaluation tab")
        
    def _update_structure_list(self):
        """Update the structure list in the UI."""
        self.structure_list.clear()
        
        if not self.structure_set:
            return
            
        # Group structures by type
        structures_by_type = {}
        
        for structure in self.structure_set.structures:
            structure_type = getattr(structure, 'type', 'Other')
            
            if structure_type not in structures_by_type:
                structures_by_type[structure_type] = []
                
            structures_by_type[structure_type].append(structure)
            
        # Add to structure list
        for type_name, structures in structures_by_type.items():
            self.structure_list.addItem(type_name)
            
            for structure in structures:
                self.structure_list.addItem(structure.name)
                
        logger.info(f"Updated structure list with {len(self.structure_set.structures)} structures")
        
    def on_structure_selection_changed(self):
        """Handle structure selection changes."""
        self._update_dvh()
        self._update_structure_stats()
        
    def _get_selected_structures(self) -> List[Structure]:
        """Get the currently selected structures."""
        selected_structures = []
        
        for index in self.structure_list.selectedIndexes():
            structure_name = self.structure_list.item(index.row()).text()
            structure = self.structure_set.get_structure_by_name(structure_name)
            if structure:
                selected_structures.append(structure)
                
        return selected_structures
        
    def _update_dvh(self):
        """Update the DVH display."""
        selected_structures = self._get_selected_structures()
        
        if not selected_structures or not self.dose_calculator:
            return
            
        # Clear current plot
        self.dvh_canvas.clear_plot()
        
        # Get DVH settings
        relative = self.relative_cb.isChecked()
        normalize = self.normalize_cb.isChecked()
        
        # Get prescription dose for normalization
        prescription = 0
        if normalize and self.beam_set:
            prescription = self.dose_calculator.get_prescription_dose()
            
        # Set max dose limit
        max_dose = self.dose_limit_spin.value()
        self.dvh_canvas.axes.set_xlim(0, max_dose)
        
        # Plot DVH for each selected structure
        for i, structure in enumerate(selected_structures):
            try:
                # Get structure color or use default colors
                if hasattr(structure, 'color'):
                    color = [c/255 for c in structure.color[:3]]  # Convert to matplotlib 0-1 range
                else:
                    # Cycle through default colors
                    colors = ['r', 'g', 'b', 'c', 'm', 'y']
                    color = colors[i % len(colors)]
                    
                # Calculate DVH
                dvh = self.plan_evaluation.calculate_cumulative_dvh(structure, relative=relative)
                
                if dvh is not None:
                    dose_bins, volume_values = dvh
                    
                    # Normalize if needed
                    if normalize and prescription > 0:
                        dose_bins = [d / prescription * 100 for d in dose_bins]
                        
                    # Plot
                    self.dvh_canvas.plot_dvh(dose_bins, volume_values, structure.name, color)
                    
            except Exception as e:
                logger.error(f"Error plotting DVH for {structure.name}: {str(e)}")
                
        # Add prescription line if not normalizing
        if not normalize and self.beam_set:
            prescription = self.dose_calculator.get_prescription_dose()
            self.dvh_canvas.set_prescription_line(prescription)
            
        # Update axis labels for normalization
        if normalize:
            self.dvh_canvas.axes.set_xlabel('Dose (% of Prescription)')
        else:
            self.dvh_canvas.axes.set_xlabel('Dose (Gy)')
            
        # Redraw
        self.dvh_canvas.draw()
        
    def _update_structure_stats(self):
        """Update the structure statistics table."""
        selected_structures = self._get_selected_structures()
        
        if not selected_structures or not self.dose_calculator:
            return
            
        # Clear table
        self.stats_table.setRowCount(0)
        
        # Add stats for each selected structure
        for structure in selected_structures:
            try:
                stats = self.plan_evaluation.get_structure_metrics(structure)
                
                if stats:
                    # Add a new row
                    row = self.stats_table.rowCount()
                    self.stats_table.insertRow(row)
                    
                    # Structure name
                    self.stats_table.setItem(row, 0, QTableWidgetItem(structure.name))
                    
                    # Min dose
                    min_dose = stats.get('min_dose', 0)
                    self.stats_table.setItem(row, 1, QTableWidgetItem(f"{min_dose:.2f}"))
                    
                    # Max dose
                    max_dose = stats.get('max_dose', 0)
                    self.stats_table.setItem(row, 2, QTableWidgetItem(f"{max_dose:.2f}"))
                    
                    # Mean dose
                    mean_dose = stats.get('mean_dose', 0)
                    self.stats_table.setItem(row, 3, QTableWidgetItem(f"{mean_dose:.2f}"))
                    
                    # D95
                    d95 = stats.get('d95', 0)
                    self.stats_table.setItem(row, 4, QTableWidgetItem(f"{d95:.2f}"))
                    
                    # V95
                    v95 = stats.get('v95', 0)
                    self.stats_table.setItem(row, 5, QTableWidgetItem(f"{v95:.2f}"))
                    
            except Exception as e:
                logger.error(f"Error getting stats for {structure.name}: {str(e)}")
                
    def on_plan_changed(self, index):
        """Handle plan selection change."""
        if index < 0:
            self.current_plan = None
            self.current_image = None
            self.current_structure_set = None
            self.current_dose = None
            self.plan_evaluator = None
            
            # Clear the structure list and DVH plot
            self.structure_list.clear()
            self.dvh_canvas.clear()
            self.stats_table.setRowCount(0)
            
            return
            
        # Get the selected plan
        self.current_plan = self.plan_combo.currentData()
        
        # Get the associated image and structure set
        if self.current_plan:
            # Get image and structure set
            self.current_image = self.current_plan.get_image()
            self.current_structure_set = self.current_plan.get_structure_set()
            self.current_dose = self.current_plan.get_dose_grid()
            
            # Set prescription dose if available
            if self.current_plan.get_prescription():
                self.prescription_spinbox.setValue(self.current_plan.get_prescription().get_dose())
            
        try:
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save DVH Plot",
                os.path.expanduser("~/dvh_plot.png"),
                "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*.*)"
            )
            
            if file_path:
                self.dvh_canvas.save_figure(file_path)
                QMessageBox.information(self, "Save Successful", f"DVH plot saved to {file_path}")
                logger.info(f"DVH plot saved to {file_path}")
                
        except Exception as e:
            logger.error(f"Error saving DVH plot: {str(e)}")
            QMessageBox.warning(self, "Save Error", f"Error saving DVH plot: {str(e)}")
            
    def _generate_report(self):
        """Generate a plan evaluation report."""
        if not self.plan_evaluation or not self.dose_calculator:
            QMessageBox.warning(self, "No Data", "No dose data available for report generation.")
            return
            
        try:
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Evaluation Report",
                os.path.expanduser("~/plan_evaluation.html"),
                "HTML Files (*.html);;All Files (*.*)"
            )
            
            if not file_path:
                return
                
            # Generate report
            report_html = self.plan_evaluation.generate_html_report()
            
            if report_html:
                with open(file_path, 'w') as f:
                    f.write(report_html)
                    
                QMessageBox.information(self, "Report Generated", f"Evaluation report saved to {file_path}")
                logger.info(f"Evaluation report saved to {file_path}")
                
                # Try to open the report in the default browser
                import webbrowser
                webbrowser.open(file_path)
            else:
                QMessageBox.warning(self, "Report Error", "Failed to generate report content.")
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            QMessageBox.warning(self, "Report Error", f"Error generating report: {str(e)}")
            
    def _evaluate_plan(self):
        """Evaluate the plan against clinical protocols."""
        if not self.plan_evaluation or not self.dose_calculator:
            QMessageBox.warning(self, "No Data", "No dose data available for plan evaluation.")
            return
            
        # Switch to the plan quality tab
        self.eval_tabs.setCurrentWidget(self.quality_tab)
        
        try:
            # Use the protocol dialog to select a protocol
            if ClinicalProtocolDialog:
                dialog = ClinicalProtocolDialog(self)
                if dialog.exec_():
                    protocol = dialog.get_selected_protocol()
                    if protocol:
                        # The plan quality widget will load and evaluate the protocol
                        self.plan_quality_widget.setPlanEvaluation(self.plan_evaluation)
                        
                        # TODO: Add logic to trigger protocol evaluation
                        # This would depend on the exact API of the PlanQualityWidget
                        
            else:
                QMessageBox.warning(self, "Feature Unavailable", "Protocol selection dialog is not available.")
        except Exception as e:
            logger.error(f"Error evaluating plan: {str(e)}")
            QMessageBox.warning(self, "Evaluation Error", f"Error evaluating plan: {str(e)}")
            
    def _on_goal_selected(self, goal_info):
        """
        Handle selection of a clinical goal.
        
        Args:
            goal_info: Information about the selected goal
        """
        # Find the structure in the tree and select it
        structure_name = goal_info.get("structure_name")
        if structure_name:
            self._select_structure_by_name(structure_name)
            
    def _select_structure_by_name(self, structure_name):
        """
        Select a structure in the tree by name.
        
        Args:
            structure_name: Name of the structure to select
        """
        # Deselect all current selections
        self.structure_tree.clearSelection()
        
        # Find the structure in the tree
        for i in range(self.structure_tree.topLevelItemCount()):
            type_item = self.structure_tree.topLevelItem(i)
            
            for j in range(type_item.childCount()):
                structure_item = type_item.child(j)
                item_text = structure_item.text(0)
                
                if item_text == structure_name:
                    # Select this item
                    structure_item.setSelected(True)
                    # Ensure it's visible
                    self.structure_tree.scrollToItem(structure_item)
                    break

def test_evaluation_tab():
    """Test the evaluation tab with sample data."""
    import sys
    from PyQt5.QtWidgets import QApplication
    import numpy as np
    from quangtps.imaging.image import Image
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure import Structure
    from quangtps.beams.beam import BeamSet
    from quangtps.dose.dose_calculator import DoseCalculator
    
    class MockDoseCalculator(DoseCalculator):
        """Mock dose calculator for testing."""
        
        def __init__(self):
            """Initialize the mock dose calculator."""
            self.structure_set = StructureSet()
            self.structure_set.structures = [
                Structure(name="PTV", id="1", type="Target"),
                Structure(name="Brainstem", id="2", type="OAR"),
                Structure(name="Spinal Cord", id="3", type="OAR"),
                Structure(name="Parotid L", id="4", type="OAR"),
                Structure(name="Parotid R", id="5", type="OAR")
            ]
            
        def get_structure_set(self):
            """Get the structure set."""
            return self.structure_set
            
        def get_prescription_dose(self):
            """Get the prescription dose."""
            return 60.0
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create evaluation tab
    tab = EvaluationTab()
    
    # Set mock dose calculator
    tab.set_dose_calculator(MockDoseCalculator())
    
    # Show tab
    tab.show()
    
    # Run application
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    test_evaluation_tab() 