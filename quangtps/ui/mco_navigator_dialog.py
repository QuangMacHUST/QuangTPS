#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module giao diện người dùng cho Multi-Criteria Optimization (MCO) Navigator.

Module này triển khai giao diện người dùng cho công cụ MCO Navigator,
mô phỏng theo giao diện MCO của Eclipse. Cho phép người dùng khám phá 
không gian lời giải Pareto và tương tác trực quan với các lời giải.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, 
    QListWidgetItem, QSplitter, QDialog, QDialogButtonBox, QComboBox, 
    QLineEdit, QFormLayout, QMessageBox, QFileDialog, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QProgressDialog, QMenu, QAction,
    QToolBar, QGroupBox, QRadioButton, QButtonGroup, QCheckBox, QSlider,
    QSpinBox, QDoubleSpinBox, QToolButton, QFrame, QScrollArea, QTableWidget,
    QTableWidgetItem, QApplication, QSizePolicy, QGridLayout
)
from PyQt5.QtGui import QColor, QIcon, QPixmap, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRectF, QTimer

# Import matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available, using simplified visualization")

# Import from QuangTPS modules
from quangtps.core.services import ServiceRegistry
from quangtps.planning.plan import Plan
from quangtps.optimization.mco.mco_interface import (
    MCOEngine, MCOSolution, MCOObjectiveSpace, MCONavigator, calculate_mco_metrics
)
from quangtps.evaluation.dvh.dvh_calculation import DVHCalculator
from quangtps.evaluation.dvh.dvh_visualization import plot_dvh

logger = logging.getLogger(__name__)

class ObjectiveSlider(QWidget):
    """
    Widget slider cho điều chỉnh giá trị của một hàm mục tiêu.
    """
    
    valueChanged = pyqtSignal(str, float)
    
    def __init__(self, objective_name, min_value, max_value, current_value, parent=None):
        super().__init__(parent)
        
        self.objective_name = objective_name
        self.min_value = min_value
        self.max_value = max_value
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        self.title_label = QLabel(objective_name)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.title_label)
        
        # Slider layout
        slider_layout = QHBoxLayout()
        
        # Min label
        self.min_label = QLabel(f"{min_value:.1f}")
        slider_layout.addWidget(self.min_label)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(self._value_to_slider(current_value))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        slider_layout.addWidget(self.slider)
        
        # Max label
        self.max_label = QLabel(f"{max_value:.1f}")
        slider_layout.addWidget(self.max_label)
        
        layout.addLayout(slider_layout)
        
        # Value display
        self.value_label = QLabel(f"Current: {current_value:.2f}")
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        
        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_value_changed)
    
    def _value_to_slider(self, value):
        """Chuyển đổi giá trị thực tế sang giá trị trên thanh trượt"""
        # Normalize to 0-100 range
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        return int(normalized * 100)
    
    def _slider_to_value(self, slider_value):
        """Chuyển đổi giá trị thanh trượt sang giá trị thực tế"""
        # Convert from 0-100 range to actual value
        normalized = slider_value / 100
        return self.min_value + normalized * (self.max_value - self.min_value)
    
    def _on_slider_value_changed(self, slider_value):
        value = self._slider_to_value(slider_value)
        self.value_label.setText(f"Current: {value:.2f}")
        self.valueChanged.emit(self.objective_name, value)
    
    def set_value(self, value):
        """Thiết lập giá trị cho thanh trượt"""
        self.slider.setValue(self._value_to_slider(value))
    
    def get_value(self):
        """Lấy giá trị hiện tại của thanh trượt"""
        return self._slider_to_value(self.slider.value())


class DVHComparisonWidget(QWidget):
    """
    Widget so sánh DVH giữa các lời giải.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure
            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_xlabel('Dose (Gy)')
            self.ax.set_ylabel('Volume (%)')
            self.ax.set_title('DVH Comparison')
            self.ax.grid(True)
            
            layout.addWidget(self.canvas)
        else:
            layout.addWidget(QLabel("Matplotlib is not available. Cannot display DVH comparison."))
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.structure_combo = QComboBox()
        self.structure_combo.setMinimumWidth(150)
        
        self.reference_combo = QComboBox()
        self.reference_combo.addItem("No Reference")
        self.reference_combo.setMinimumWidth(150)
        
        controls_layout.addWidget(QLabel("Structure:"))
        controls_layout.addWidget(self.structure_combo)
        controls_layout.addWidget(QLabel("Reference:"))
        controls_layout.addWidget(self.reference_combo)
        
        layout.addLayout(controls_layout)
        
        # Connect signals
        if MATPLOTLIB_AVAILABLE:
            self.structure_combo.currentIndexChanged.connect(self.update_plot)
            self.reference_combo.currentIndexChanged.connect(self.update_plot)
    
    def clear(self):
        """Clear the plot"""
        if MATPLOTLIB_AVAILABLE:
            self.ax.clear()
            self.ax.set_xlabel('Dose (Gy)')
            self.ax.set_ylabel('Volume (%)')
            self.ax.set_title('DVH Comparison')
            self.ax.grid(True)
            self.canvas.draw()
    
    def update_structures(self, structures):
        """Update the structure list"""
        self.structure_combo.clear()
        for structure in structures:
            self.structure_combo.addItem(structure.name)
    
    def update_references(self, references):
        """Update reference plan list"""
        current_text = self.reference_combo.currentText()
        
        self.reference_combo.clear()
        self.reference_combo.addItem("No Reference")
        
        for ref in references:
            self.reference_combo.addItem(ref)
        
        # Try to restore the previous selection
        index = self.reference_combo.findText(current_text)
        if index >= 0:
            self.reference_combo.setCurrentIndex(index)
    
    def update_plot(self):
        """Update the DVH plot"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # This would be implemented to update the DVH plot
        # based on the current solution and reference
        pass
    
    def plot_dvh_comparison(self, current_solution, reference_solution=None, structure_name=None):
        """Plot DVH comparison between solutions"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Clear the plot
        self.ax.clear()
        
        # If no structure specified, use the currently selected one
        if structure_name is None:
            if self.structure_combo.count() > 0:
                structure_name = self.structure_combo.currentText()
            else:
                return
        
        # Find the structure
        current_plan = current_solution.plan
        structure = None
        for s in current_plan.get_structures():
            if s.name == structure_name:
                structure = s
                break
        
        if structure is None:
            return
        
        # Plot current solution DVH
        try:
            dvh_data = DVHCalculator.calculate_dvh(current_plan, structure)
            self.ax.plot(dvh_data.doses, dvh_data.volumes, 'b-', label=f"Current Solution")
            
            # Plot reference DVH if available
            if reference_solution:
                ref_plan = reference_solution.plan
                ref_dvh_data = DVHCalculator.calculate_dvh(ref_plan, structure)
                self.ax.plot(ref_dvh_data.doses, ref_dvh_data.volumes, 'r--', label=f"Reference")
            
            self.ax.set_xlabel('Dose (Gy)')
            self.ax.set_ylabel('Volume (%)')
            self.ax.set_title(f'DVH Comparison - {structure_name}')
            self.ax.grid(True)
            self.ax.legend()
            
            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error plotting DVH comparison: {e}")


class ParetoFrontWidget(QWidget):
    """
    Widget hiển thị không gian Pareto và cho phép người dùng chọn lời giải.
    """
    
    solutionSelected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure
            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_xlabel('Objective 1')
            self.ax.set_ylabel('Objective 2')
            self.ax.set_title('Pareto Front')
            self.ax.grid(True)
            
            layout.addWidget(self.canvas)
            
            # Connect click event
            self.canvas.mpl_connect('button_press_event', self._on_click)
        else:
            layout.addWidget(QLabel("Matplotlib is not available. Cannot display Pareto front."))
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.x_axis_combo = QComboBox()
        self.y_axis_combo = QComboBox()
        
        controls_layout.addWidget(QLabel("X Axis:"))
        controls_layout.addWidget(self.x_axis_combo)
        controls_layout.addWidget(QLabel("Y Axis:"))
        controls_layout.addWidget(self.y_axis_combo)
        
        layout.addLayout(controls_layout)
        
        # Connect signals
        if MATPLOTLIB_AVAILABLE:
            self.x_axis_combo.currentIndexChanged.connect(self.update_plot)
            self.y_axis_combo.currentIndexChanged.connect(self.update_plot)
        
        self.objective_space = None
        self.points = []  # Store the points for mouse click detection
    
    def set_objective_space(self, objective_space):
        """Set the objective space"""
        self.objective_space = objective_space
        
        # Update objective lists
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        
        objective_names = list(objective_space.objectives.keys())
        
        for name in objective_names:
            self.x_axis_combo.addItem(name)
            self.y_axis_combo.addItem(name)
        
        # Set default selections if possible
        if len(objective_names) >= 2:
            self.x_axis_combo.setCurrentIndex(0)
            self.y_axis_combo.setCurrentIndex(1)
        
        self.update_plot()
    
    def update_plot(self):
        """Update the Pareto front plot"""
        if not MATPLOTLIB_AVAILABLE or self.objective_space is None:
            return
        
        if self.x_axis_combo.count() == 0 or self.y_axis_combo.count() == 0:
            return
        
        x_objective = self.x_axis_combo.currentText()
        y_objective = self.y_axis_combo.currentText()
        
        # Use the objective space's plotting method
        self.ax.clear()
        self.objective_space.plot_pareto_front(x_objective, y_objective, self.ax)
        
        self.canvas.draw()
    
    def _on_click(self, event):
        """Handle mouse click events on the plot"""
        if not self.objective_space:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        # Find the closest solution point
        closest_index = None
        min_distance = float('inf')
        
        x_objective = self.x_axis_combo.currentText()
        y_objective = self.y_axis_combo.currentText()
        
        for i, solution in enumerate(self.objective_space.solutions):
            x = solution.get_objective_value(x_objective)
            y = solution.get_objective_value(y_objective)
            
            # Calculate distance in data coordinates
            distance = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        if closest_index is not None:
            self.solutionSelected.emit(closest_index)
            self.update_plot()  # Refresh to show the new current solution


class SolutionMetricsWidget(QWidget):
    """
    Widget hiển thị các chỉ số của lời giải hiện tại.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.metrics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.metrics_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        
        layout.addWidget(self.metrics_table)
    
    def update_metrics(self, solution):
        """Update the metrics display for a solution"""
        self.metrics_table.setRowCount(0)
        
        if solution is None:
            return
        
        # Calculate metrics
        metrics = calculate_mco_metrics(solution)
        
        # Add objective values
        for name, value in solution.objectives.items():
            self.add_metric(name, value)
        
        # Add calculated metrics
        for name, value in metrics.items():
            if name == 'CI':
                self.add_metric("Conformity Index", value, "")
            elif name == 'HI':
                self.add_metric("Homogeneity Index", value, "")
            elif name == 'GI':
                self.add_metric("Gradient Index", value, "")
    
    def add_metric(self, name, value, unit=""):
        """Add a metric to the table"""
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        
        self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{value:.3f}"))
        self.metrics_table.setItem(row, 2, QTableWidgetItem(unit))


class MCONavigatorDialog(QDialog):
    """
    Hộp thoại chính cho Multi-Criteria Optimization Navigator.
    
    Cung cấp giao diện người dùng tương tác để khám phá không gian lời giải Pareto,
    so sánh các lời giải, và chọn lời giải tối ưu dựa trên ưu tiên lâm sàng.
    """
    
    solutionAccepted = pyqtSignal(MCOSolution)
    
    def __init__(self, plan, parent=None):
        """
        Khởi tạo hộp thoại MCO Navigator.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị cần tối ưu hóa
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        self.plan = plan
        self.setWindowTitle("Multi-Criteria Optimization Navigator")
        self.resize(1200, 800)
        
        # Initialize MCO Engine
        self.mco_engine = MCOEngine(plan)
        self.solution_sliders = {}
        
        # Setup UI
        self._setup_ui()
        
        # Generate initial solutions
        QTimer.singleShot(100, self._initialize_solutions)
    
    def _setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout(self)
        
        # Main splitter
        self.main_splitter = QSplitter(Qt.Vertical)
        
        # Upper section (visual navigation)
        self.upper_widget = QWidget()
        upper_layout = QHBoxLayout(self.upper_widget)
        
        # Pareto front visualization
        self.pareto_widget = ParetoFrontWidget()
        upper_layout.addWidget(self.pareto_widget)
        
        # DVH comparison
        self.dvh_widget = DVHComparisonWidget()
        upper_layout.addWidget(self.dvh_widget)
        
        self.main_splitter.addWidget(self.upper_widget)
        
        # Lower section (sliders and metrics)
        self.lower_widget = QWidget()
        lower_layout = QHBoxLayout(self.lower_widget)
        
        # Sliders for objectives
        self.sliders_widget = QWidget()
        self.sliders_layout = QVBoxLayout(self.sliders_widget)
        
        sliders_scroll = QScrollArea()
        sliders_scroll.setWidgetResizable(True)
        sliders_scroll.setWidget(self.sliders_widget)
        
        # Metrics
        self.metrics_widget = SolutionMetricsWidget()
        
        lower_layout.addWidget(sliders_scroll)
        lower_layout.addWidget(self.metrics_widget)
        
        self.main_splitter.addWidget(self.lower_widget)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([500, 300])
        
        main_layout.addWidget(self.main_splitter)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Solutions")
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        
        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.undo_btn)
        button_layout.addWidget(self.redo_btn)
        button_layout.addStretch()
        button_layout.addWidget(button_box)
        
        main_layout.addLayout(button_layout)
        
        # Connect signals
        self.generate_btn.clicked.connect(self._generate_solutions)
        self.undo_btn.clicked.connect(self._undo)
        self.redo_btn.clicked.connect(self._redo)
        
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        
        self.pareto_widget.solutionSelected.connect(self._on_solution_selected)
    
    def _initialize_solutions(self):
        """Initialize the MCO solutions"""
        try:
            # Add objectives based on the plan
            self._setup_objectives()
            
            # Generate initial solutions
            progress = QProgressDialog("Generating initial solutions...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Generate solutions
            solutions = self.mco_engine.generate_initial_solutions(10)
            
            # Update UI
            self.pareto_widget.set_objective_space(self.mco_engine.objective_space)
            
            # Select the first solution
            if solutions:
                self.mco_engine.navigator.select_solution(0)
                self._update_ui_for_current_solution()
            
            progress.close()
            
        except Exception as e:
            logger.error(f"Error initializing MCO solutions: {e}")
            QMessageBox.critical(self, "Error", f"Could not initialize MCO solutions: {str(e)}")
    
    def _setup_objectives(self):
        """Setup objectives based on the plan"""
        # This would be implemented to add objectives based on the plan's structures
        # For example, adding objectives for PTV coverage, OAR sparing, etc.
        
        # For demonstration, add some sample objectives
        from quangtps.optimization.objectives import DVHObjective, MeanDoseObjective
        
        # Find PTV and OARs
        ptv = None
        oars = []
        
        for structure in self.plan.get_structures():
            if "PTV" in structure.name.upper():
                ptv = structure
            elif any(oar_name in structure.name.upper() for oar_name in ["HEART", "LUNG", "SPINAL", "CORD", "LIVER", "KIDNEY"]):
                oars.append(structure)
        
        # Add PTV coverage objective
        if ptv:
            ptv_obj = DVHObjective(structure=ptv, dose=self.plan.prescription.dose * 0.95, 
                                 volume=95, direction="greater", weight=100)
            self.mco_engine.add_objective(ptv_obj, 100, "PTV Coverage")
        
        # Add OAR sparing objectives
        for oar in oars:
            if "HEART" in oar.name.upper():
                heart_obj = MeanDoseObjective(structure=oar, dose=15, weight=50)
                self.mco_engine.add_objective(heart_obj, 50, "Heart Mean Dose")
            elif "LUNG" in oar.name.upper():
                lung_obj = DVHObjective(structure=oar, dose=20, volume=30, 
                                      direction="less", weight=50)
                self.mco_engine.add_objective(lung_obj, 50, "Lung V20")
            elif "SPINAL" in oar.name.upper() or "CORD" in oar.name.upper():
                cord_obj = DVHObjective(structure=oar, dose=45, volume=0, 
                                      direction="less", weight=80)
                self.mco_engine.add_objective(cord_obj, 80, "Cord Max Dose")
    
    def _create_objective_sliders(self):
        """Create sliders for adjusting objective values"""
        # Clear existing sliders
        for i in reversed(range(self.sliders_layout.count())):
            widget = self.sliders_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        self.solution_sliders = {}
        
        # Get current solution to determine value ranges
        current_solution = self.mco_engine.objective_space.get_current_solution()
        if not current_solution:
            return
        
        # Create a slider for each objective
        for name in self.mco_engine.objective_space.objectives.keys():
            # Get min/max values across all solutions
            min_value = float('inf')
            max_value = float('-inf')
            
            for solution in self.mco_engine.objective_space.solutions:
                value = solution.get_objective_value(name)
                min_value = min(min_value, value)
                max_value = max(max_value, value)
            
            # Add some padding to the range
            range_padding = 0.1 * (max_value - min_value)
            min_value -= range_padding
            max_value += range_padding
            
            # Create the slider
            current_value = current_solution.get_objective_value(name)
            slider = ObjectiveSlider(name, min_value, max_value, current_value)
            slider.valueChanged.connect(self._on_slider_value_changed)
            
            self.sliders_layout.addWidget(slider)
            self.solution_sliders[name] = slider
        
        # Add a stretch at the end
        self.sliders_layout.addStretch()
    
    def _update_ui_for_current_solution(self):
        """Update all UI components for the current solution"""
        current_solution = self.mco_engine.objective_space.get_current_solution()
        if not current_solution:
            return
        
        # Create sliders if not already created
        if not self.solution_sliders:
            self._create_objective_sliders()
        
        # Update slider values
        for name, slider in self.solution_sliders.items():
            value = current_solution.get_objective_value(name)
            slider.set_value(value)
        
        # Update metrics display
        self.metrics_widget.update_metrics(current_solution)
        
        # Update DVH display
        structures = self.plan.get_structures()
        self.dvh_widget.update_structures(structures)
        
        # Update references
        references = ["Original Plan"]
        for i, solution in enumerate(self.mco_engine.objective_space.solutions):
            if solution is not current_solution:
                references.append(f"Solution {i+1}")
        
        self.dvh_widget.update_references(references)
        
        # Update DVH plot
        self.dvh_widget.plot_dvh_comparison(current_solution)
        
        # Update Pareto plot
        self.pareto_widget.update_plot()
    
    def _generate_solutions(self):
        """Generate a new set of solutions"""
        try:
            progress = QProgressDialog("Generating solutions...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Generate solutions
            solutions = self.mco_engine.generate_initial_solutions(10)
            
            # Update UI
            self.pareto_widget.update_plot()
            
            # Select the first solution
            if solutions:
                self.mco_engine.navigator.select_solution(0)
                self._update_ui_for_current_solution()
            
            progress.close()
            
        except Exception as e:
            logger.error(f"Error generating MCO solutions: {e}")
            QMessageBox.critical(self, "Error", f"Could not generate MCO solutions: {str(e)}")
    
    def _on_solution_selected(self, index):
        """Handle selection of a solution from the Pareto plot"""
        try:
            self.mco_engine.navigator.select_solution(index)
            self._update_ui_for_current_solution()
        except Exception as e:
            logger.error(f"Error selecting solution: {e}")
    
    def _on_slider_value_changed(self, objective_name, target_value):
        """Handle changes in objective sliders"""
        try:
            # For now, just find the closest solution
            closest_solution_index = None
            min_difference = float('inf')
            
            for i, solution in enumerate(self.mco_engine.objective_space.solutions):
                value = solution.get_objective_value(objective_name)
                difference = abs(value - target_value)
                
                if difference < min_difference:
                    min_difference = difference
                    closest_solution_index = i
            
            if closest_solution_index is not None:
                self.mco_engine.navigator.select_solution(closest_solution_index)
                self._update_ui_for_current_solution()
        except Exception as e:
            logger.error(f"Error handling slider change: {e}")
    
    def _undo(self):
        """Undo the last navigation action"""
        previous_index = self.mco_engine.navigator.undo()
        if previous_index is not None:
            self._update_ui_for_current_solution()
    
    def _redo(self):
        """Redo the last undone navigation action"""
        next_index = self.mco_engine.navigator.redo()
        if next_index is not None:
            self._update_ui_for_current_solution()
    
    def _on_accept(self):
        """Handle dialog acceptance"""
        current_solution = self.mco_engine.objective_space.get_current_solution()
        if current_solution:
            self.solutionAccepted.emit(current_solution)
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "No solution is currently selected.")


if __name__ == "__main__":
    """
    Demo standalone mode for MCO Navigator Dialog.
    """
    # This would be used for testing the dialog independently
    app = QApplication(sys.argv)
    
    # Create a mock plan for testing
    from quangtps.planning.plan import Plan
    mock_plan = Plan()
    
    dialog = MCONavigatorDialog(mock_plan)
    dialog.show()
    
    sys.exit(app.exec_()) 