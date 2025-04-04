import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QBrush, QPainter, QPainterPath

# Import matplotlib for plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from quangtps.optimization.mco.pareto_surface import ParetoSurface, ParetoSolution

logger = logging.getLogger(__name__)

class ParetoChart(QWidget):
    """
    Chart for visualizing the Pareto surface.
    
    This chart shows the Pareto front in 2D by plotting pairs of objectives.
    It allows the user to select which objectives to display on the x and y axes.
    """
    
    solutionSelected = pyqtSignal(int)  # Signal emitted when a solution is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self.pareto_surface: Optional[ParetoSurface] = None
        self.current_solution: Optional[ParetoSolution] = None
        self.solution_points = []
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Controls for selecting axes
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_layout = QHBoxLayout(controls_frame)
        
        # X-axis selector
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X-axis:"))
        self.x_axis_combo = QComboBox()
        x_layout.addWidget(self.x_axis_combo)
        controls_layout.addLayout(x_layout)
        
        # Y-axis selector
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y-axis:"))
        self.y_axis_combo = QComboBox()
        y_layout.addWidget(self.y_axis_combo)
        controls_layout.addLayout(y_layout)
        
        main_layout.addWidget(controls_frame)
        
        # Matplotlib figure for plotting
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        main_layout.addWidget(self.canvas, 1)
        
        # Connect signals
        self.x_axis_combo.currentIndexChanged.connect(self._update_chart)
        self.y_axis_combo.currentIndexChanged.connect(self._update_chart)
        self.canvas.mpl_connect('pick_event', self._on_pick)
    
    def set_pareto_surface(self, pareto_surface: ParetoSurface):
        """
        Set the Pareto surface to display.
        
        Args:
            pareto_surface: The Pareto surface
        """
        self.pareto_surface = pareto_surface
        
        # Update axis selectors
        self._update_axis_selectors()
        
        # Update chart
        self._update_chart()
    
    def highlight_solution(self, solution: ParetoSolution):
        """
        Highlight a specific solution on the chart.
        
        Args:
            solution: The solution to highlight
        """
        self.current_solution = solution
        
        # Update the chart to show the highlight
        self._update_chart()
    
    def clear(self):
        """Clear the chart."""
        self.pareto_surface = None
        self.current_solution = None
        self.solution_points = []
        
        # Clear the plot
        if hasattr(self, 'axes'):
            self.axes.clear()
            self.canvas.draw()
            
        # Clear the combo boxes
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
    
    def _update_axis_selectors(self):
        """Update the axis selector combo boxes with available objectives."""
        if not self.pareto_surface:
            return
            
        # Block signals to prevent triggering updates
        self.x_axis_combo.blockSignals(True)
        self.y_axis_combo.blockSignals(True)
        
        # Clear existing items
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        
        # Add objectives
        objectives = self.pareto_surface.objective_names
        for obj in objectives:
            self.x_axis_combo.addItem(obj)
            self.y_axis_combo.addItem(obj)
            
        # Select default axes
        if len(objectives) >= 2:
            self.x_axis_combo.setCurrentIndex(0)
            self.y_axis_combo.setCurrentIndex(1)
        elif len(objectives) == 1:
            self.x_axis_combo.setCurrentIndex(0)
            self.y_axis_combo.setCurrentIndex(0)
            
        # Unblock signals
        self.x_axis_combo.blockSignals(False)
        self.y_axis_combo.blockSignals(False)
    
    def _update_chart(self):
        """Update the chart based on the selected axes and Pareto surface."""
        if not self.pareto_surface or self.pareto_surface.is_empty():
            return
            
        # Get selected objectives
        if self.x_axis_combo.count() == 0 or self.y_axis_combo.count() == 0:
            return
            
        x_obj = self.x_axis_combo.currentText()
        y_obj = self.y_axis_combo.currentText()
        
        # Clear the plot
        self.axes.clear()
        
        # Set labels
        self.axes.set_xlabel(x_obj)
        self.axes.set_ylabel(y_obj)
        self.axes.set_title("Pareto Surface")
        
        # Get objective ranges
        x_range = self.pareto_surface.get_objective_range(x_obj)
        y_range = self.pareto_surface.get_objective_range(y_obj)
        
        # Calculate axis margins (5% padding)
        x_margin = (x_range[1] - x_range[0]) * 0.05 if x_range[1] > x_range[0] else 0.1
        y_margin = (y_range[1] - y_range[0]) * 0.05 if y_range[1] > y_range[0] else 0.1
        
        # Set axis limits with margins
        self.axes.set_xlim(x_range[0] - x_margin, x_range[1] + x_margin)
        self.axes.set_ylim(y_range[0] - y_margin, y_range[1] + y_margin)
        
        # Plot solutions
        x_values = []
        y_values = []
        self.solution_points = []
        
        for i, solution in enumerate(self.pareto_surface.solutions):
            x_val = solution.get_objective_value(x_obj)
            y_val = solution.get_objective_value(y_obj)
            
            x_values.append(x_val)
            y_values.append(y_val)
            self.solution_points.append((i, x_val, y_val))
        
        # Plot all solutions
        scatter = self.axes.scatter(
            x_values, y_values,
            c='#7EB9FF',  # Light blue color
            s=80,         # Point size
            alpha=0.7,    # Transparency
            edgecolors='#1F77B4',  # Darker blue edge color
            linewidths=1,
            picker=5      # Make points pickable
        )
        
        # If we have a current solution, highlight it
        if self.current_solution:
            current_x = self.current_solution.get_objective_value(x_obj)
            current_y = self.current_solution.get_objective_value(y_obj)
            
            # Mark current solution with a red circle
            self.axes.scatter(
                [current_x], [current_y],
                c='#FF5050',  # Light red
                s=100,        # Slightly larger point
                alpha=0.9,    # More opaque
                edgecolors='#CC0000',  # Darker red edge
                linewidths=2,
                zorder=10     # Ensure it's on top
            )
            
            # Also draw a larger circle around it
            circle = Circle(
                (current_x, current_y),
                radius=max(x_margin/3, y_margin/3),
                fill=False,
                edgecolor='#CC0000',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                zorder=9
            )
            self.axes.add_patch(circle)
        
        # Draw Pareto front if we have more than 1 solution
        if len(x_values) > 1 and x_obj != y_obj:
            # Find the Pareto front points
            # (for minimization problems, we want points that are not dominated)
            pareto_indices = []
            for i in range(len(x_values)):
                dominated = False
                for j in range(len(x_values)):
                    if i != j:
                        if (x_values[j] <= x_values[i] and y_values[j] <= y_values[i] and
                            (x_values[j] < x_values[i] or y_values[j] < y_values[i])):
                            dominated = True
                            break
                if not dominated:
                    pareto_indices.append(i)
            
            # Sort indices by x value
            pareto_indices.sort(key=lambda i: x_values[i])
            
            # Draw the Pareto front line
            pareto_x = [x_values[i] for i in pareto_indices]
            pareto_y = [y_values[i] for i in pareto_indices]
            
            self.axes.plot(
                pareto_x, pareto_y,
                'r--',
                linewidth=1.5,
                alpha=0.7
            )
        
        # Grid and tight layout
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.figure.tight_layout()
        
        # Redraw the canvas
        self.canvas.draw()
    
    def _on_pick(self, event):
        """
        Handle picking a point on the chart.
        
        Args:
            event: The pick event
        """
        # Only process scatter plot picks
        if not hasattr(event, 'ind') or len(event.ind) == 0:
            return
            
        # Get the index of the picked point
        index = event.ind[0]
        
        if 0 <= index < len(self.solution_points):
            solution_index = self.solution_points[index][0]
            
            # Emit the signal with the solution index
            self.solutionSelected.emit(solution_index) 