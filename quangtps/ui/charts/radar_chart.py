import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

# Import matplotlib for plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class RadarChart(QWidget):
    """
    Radar chart for visualizing objective values of different solutions.
    
    This chart displays objective values on radial axes, making it easy
    to compare multiple solutions across different objectives.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.solutions: List[Dict] = []  # List of dictionaries with objective values
        self.labels: List[str] = []      # List of solution labels
        self.colors: List[QColor] = []   # List of solution colors
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        
        # Create subplot with polar projection
        self.axes = self.figure.add_subplot(111, polar=True)
        
        # Initial empty plot
        self.canvas.draw()
    
    def add_solution(self, objective_values: Dict[str, float], label: str, color: QColor):
        """
        Add a solution to the radar chart.
        
        Args:
            objective_values: Dictionary mapping objective names to values
            label: Label for the solution in the legend
            color: Color for the solution
        """
        self.solutions.append(objective_values)
        self.labels.append(label)
        self.colors.append(color)
    
    def clear(self):
        """Clear the chart."""
        self.solutions = []
        self.labels = []
        self.colors = []
        
        # Clear the plot
        if hasattr(self, 'axes'):
            self.axes.clear()
            self.canvas.draw()
    
    def update_chart(self):
        """Update the chart with the current data."""
        if not self.solutions:
            return
        
        # Clear the plot
        self.axes.clear()
        
        # Get all objective names from solutions
        all_objectives = set()
        for solution in self.solutions:
            all_objectives.update(solution.keys())
        
        # Sort objectives for consistent display
        objectives = sorted(list(all_objectives))
        n_objectives = len(objectives)
        
        if n_objectives < 3:
            # Need at least 3 objectives for a radar chart
            self.axes.text(0, 0, "Need at least 3 objectives for radar chart",
                         ha='center', va='center', fontsize=12)
            self.canvas.draw()
            return
        
        # Angle of each axis
        angles = np.linspace(0, 2*np.pi, n_objectives, endpoint=False).tolist()
        
        # Close the polygon by repeating the first angle
        angles.append(angles[0])
        
        # Set the labels for each axis
        self.axes.set_xticks(angles[:-1])
        self.axes.set_xticklabels(objectives, fontsize=9)
        
        # Get the range of each objective
        ranges = {}
        for obj in objectives:
            values = [sol.get(obj, 0) for sol in self.solutions if obj in sol]
            if values:
                ranges[obj] = (min(values), max(values))
            else:
                ranges[obj] = (0, 1)
        
        # Plot each solution
        for i, (solution, label, color) in enumerate(zip(self.solutions, self.labels, self.colors)):
            # Normalize the values to [0, 1]
            normalized_values = []
            for obj in objectives:
                value = solution.get(obj, 0)
                obj_min, obj_max = ranges[obj]
                
                # Avoid division by zero
                if obj_max == obj_min:
                    normalized = 0.5
                else:
                    normalized = (value - obj_min) / (obj_max - obj_min)
                
                normalized_values.append(normalized)
            
            # Close the polygon by repeating the first value
            normalized_values.append(normalized_values[0])
            
            # Plot the values
            rgba = [color.red()/255, color.green()/255, color.blue()/255, color.alpha()/255]
            self.axes.plot(angles, normalized_values, linewidth=2, linestyle='solid', 
                          color=rgba, label=label)
            self.axes.fill(angles, normalized_values, color=rgba, alpha=0.1)
        
        # Set y-ticks (optional)
        self.axes.set_yticks([0.2, 0.4, 0.6, 0.8])
        self.axes.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8)
        self.axes.set_rlabel_position(0)
        
        # Add legend
        self.axes.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Adjust the starting angle
        self.axes.set_theta_offset(np.pi / 2)
        self.axes.set_theta_direction(-1)
        
        # Add grid lines
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Set title
        self.axes.set_title("Objective Values Comparison", size=12, y=1.1)
        
        # Tight layout
        self.figure.tight_layout()
        
        # Draw the chart
        self.canvas.draw() 