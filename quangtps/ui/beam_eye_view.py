#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beam's Eye View (BEV) Visualizer
==============================

This module provides a visualization tool for viewing structures and fields
from a beam's perspective, similar to the BEV view in Eclipse.
"""

import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QCheckBox, QGroupBox, QSlider, QToolBar, QAction,
    QToolButton, QSizePolicy, QStyle, QFrame, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPixmap, QColor

logger = logging.getLogger(__name__)

class BEVCanvas(FigureCanvas):
    """Canvas for displaying beam's eye view."""
    
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        """Initialize the BEV canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='black')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Setup figure
        self.setup_figure()
        
        # Initialize variables
        self.beam = None
        self.structures = []
        self.current_sad = 1000.0  # Source-to-axis distance in mm
        self.isocenter = np.array([0, 0, 0])
        self.field_size = [100, 100]  # mm
        self.mlc_positions = None
        self.jaw_positions = None
        self.show_structures = True
        self.show_field = True
        self.show_mlc = True
        self.show_jaws = True
        self.show_grid = True
        self.show_rulers = True
        self.structure_colors = {}
        
        # Set default colors for structures
        self.default_colors = {
            'PTV': 'red',
            'CTV': 'pink',
            'GTV': 'maroon',
            'BODY': 'blue',
            'LUNG': 'yellow',
            'HEART': 'red',
            'CORD': 'green',
            'ESOPHAGUS': 'orange',
            'LIVER': 'brown',
            'KIDNEY': 'purple',
            'BRAIN': 'lightblue',
            'LENS': 'cyan',
            'PAROTID': 'magenta'
        }
    
    def setup_figure(self):
        """Setup the figure appearance."""
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Set axes properties
        self.axes.set_aspect('equal')
        self.axes.set_facecolor('black')
        
        # Set grid
        self.axes.grid(True, color='gray', linestyle='-', linewidth=0.2, alpha=0.5)
        
        # Set labels
        self.axes.set_xlabel('X (mm)', color='white')
        self.axes.set_ylabel('Y (mm)', color='white')
        
        # Set tick colors
        self.axes.tick_params(colors='white')
        
        # Set spines color
        for spine in self.axes.spines.values():
            spine.set_color('white')
    
    def set_beam(self, beam):
        """
        Set the beam for visualization.
        
        Parameters
        ----------
        beam : Beam
            The beam to visualize
        """
        self.beam = beam
        
        # Extract beam parameters
        if hasattr(beam, 'sad'):
            self.current_sad = beam.sad
        
        if hasattr(beam, 'isocenter'):
            self.isocenter = np.array(beam.isocenter)
        
        if hasattr(beam, 'field_size'):
            self.field_size = beam.field_size
        
        if hasattr(beam, 'mlc_positions'):
            self.mlc_positions = beam.mlc_positions
        
        if hasattr(beam, 'jaw_positions'):
            self.jaw_positions = beam.jaw_positions
        
        # Update the view
        self.update_view()
    
    def set_structures(self, structures):
        """
        Set the structures for visualization.
        
        Parameters
        ----------
        structures : list
            List of structures to visualize
        """
        self.structures = structures
        
        # Assign colors to structures
        for structure in structures:
            if structure.name in self.default_colors:
                self.structure_colors[structure.id] = self.default_colors[structure.name]
            else:
                # Assign a random color for structures not in default_colors
                color = np.random.rand(3)
                self.structure_colors[structure.id] = mcolors.to_hex(color)
        
        # Update the view
        self.update_view()
    
    def set_sad(self, sad):
        """
        Set the source-to-axis distance.
        
        Parameters
        ----------
        sad : float
            Source-to-axis distance in mm
        """
        self.current_sad = sad
        self.update_view()
    
    def update_view(self):
        """Update the beam's eye view."""
        # Clear the axes
        self.axes.clear()
        
        # Setup axes
        self.setup_figure()
        
        # Draw structures if enabled
        if self.show_structures and self.structures:
            self._draw_structures()
        
        # Draw field if enabled
        if self.show_field:
            self._draw_field()
        
        # Draw MLC if enabled
        if self.show_mlc and self.mlc_positions is not None:
            self._draw_mlc()
        
        # Draw jaws if enabled
        if self.show_jaws and self.jaw_positions is not None:
            self._draw_jaws()
        
        # Set limits based on field size or default
        field_margin = 20  # mm
        if self.field_size:
            xlim = max(200, self.field_size[0] + field_margin)
            ylim = max(200, self.field_size[1] + field_margin)
            self.axes.set_xlim(-xlim/2, xlim/2)
            self.axes.set_ylim(-ylim/2, ylim/2)
        else:
            self.axes.set_xlim(-150, 150)
            self.axes.set_ylim(-150, 150)
        
        # Draw grid if enabled
        if self.show_grid:
            self.axes.grid(True, color='gray', linestyle='-', linewidth=0.2, alpha=0.5)
        else:
            self.axes.grid(False)
        
        # Draw rulers if enabled
        if self.show_rulers:
            self._draw_rulers()
        
        # Draw a center crosshair
        self._draw_crosshair()
        
        # Draw legend
        if self.show_structures and self.structures:
            self._draw_legend()
        
        # Update canvas
        self.fig.canvas.draw()
    
    def _draw_structures(self):
        """Draw the structures in beam's eye view."""
        if not self.structures:
            return
        
        if not self.beam:
            return
        
        # Transform structures to beam's eye view
        for structure in self.structures:
            # Skip if structure doesn't have contours
            if not hasattr(structure, 'contours') or not structure.contours:
                continue
            
            # Get color for structure
            color = self.structure_colors.get(structure.id, 'white')
            
            # Project contours to beam's eye view
            for contour in structure.contours:
                # Transform contour points to beam's coordinate system
                if hasattr(self.beam, 'get_bev_coordinates'):
                    # Use beam's method if available
                    bev_points = self.beam.get_bev_coordinates(contour)
                else:
                    # Otherwise use simple projection
                    bev_points = self._project_to_bev(contour)
                
                # Draw contour
                if len(bev_points) > 2:
                    polygon = Polygon(bev_points, closed=True, fill=True, 
                                      color=color, alpha=0.3, edgecolor=color)
                    self.axes.add_patch(polygon)
    
    def _draw_field(self):
        """Draw the treatment field."""
        if not self.field_size:
            return
        
        # Draw rectangular field
        width, height = self.field_size
        x, y = -width/2, -height/2
        
        rect = Rectangle((x, y), width, height, fill=False, 
                          edgecolor='yellow', linewidth=2)
        self.axes.add_patch(rect)
    
    def _draw_mlc(self):
        """Draw the multi-leaf collimator (MLC)."""
        if not self.mlc_positions:
            return
        
        # Example MLC drawing - would need real MLC data
        # This simulates basic MLC leaves
        leaf_width = 5  # mm
        num_leaves = 60
        leaf_length = 200  # mm
        
        # Draw leaves in a simple pattern
        for i in range(num_leaves//2):
            # Top bank leaf
            y_pos = i * leaf_width
            x_pos = -leaf_length/2 + np.random.rand() * 50  # Random position for illustration
            
            top_leaf = Rectangle((x_pos, y_pos), leaf_length/2 - x_pos, leaf_width, 
                                 fill=True, color='gray', alpha=0.7)
            self.axes.add_patch(top_leaf)
            
            # Bottom bank leaf
            y_pos = -i * leaf_width - leaf_width
            x_pos = -leaf_length/2 + np.random.rand() * 50  # Random position for illustration
            
            bottom_leaf = Rectangle((x_pos, y_pos), leaf_length/2 - x_pos, leaf_width, 
                                   fill=True, color='gray', alpha=0.7)
            self.axes.add_patch(bottom_leaf)
            
            # Opposing leaves (right side)
            y_pos = i * leaf_width
            x_pos = np.random.rand() * 50  # Random position for illustration
            
            top_leaf_right = Rectangle((x_pos, y_pos), leaf_length/2 - x_pos, leaf_width, 
                                       fill=True, color='gray', alpha=0.7)
            self.axes.add_patch(top_leaf_right)
            
            y_pos = -i * leaf_width - leaf_width
            x_pos = np.random.rand() * 50  # Random position for illustration
            
            bottom_leaf_right = Rectangle((x_pos, y_pos), leaf_length/2 - x_pos, leaf_width, 
                                         fill=True, color='gray', alpha=0.7)
            self.axes.add_patch(bottom_leaf_right)
    
    def _draw_jaws(self):
        """Draw the collimator jaws."""
        if not self.jaw_positions:
            return
        
        # Example jaw drawing - would need real jaw data
        field_width, field_height = self.field_size
        
        # Draw jaws as semi-transparent rectangles outside the field
        # X1 jaw (left)
        x1_jaw = Rectangle((-200, -200), 200 - field_width/2, 400, 
                         fill=True, color='darkgray', alpha=0.5)
        self.axes.add_patch(x1_jaw)
        
        # X2 jaw (right)
        x2_jaw = Rectangle((field_width/2, -200), 200 - field_width/2, 400, 
                         fill=True, color='darkgray', alpha=0.5)
        self.axes.add_patch(x2_jaw)
        
        # Y1 jaw (bottom)
        y1_jaw = Rectangle((-200, -200), 400, 200 - field_height/2, 
                         fill=True, color='darkgray', alpha=0.5)
        self.axes.add_patch(y1_jaw)
        
        # Y2 jaw (top)
        y2_jaw = Rectangle((-200, field_height/2), 400, 200 - field_height/2, 
                         fill=True, color='darkgray', alpha=0.5)
        self.axes.add_patch(y2_jaw)
    
    def _draw_crosshair(self):
        """Draw a center crosshair."""
        # Draw horizontal line
        self.axes.axhline(y=0, color='white', linestyle='--', alpha=0.8)
        
        # Draw vertical line
        self.axes.axvline(x=0, color='white', linestyle='--', alpha=0.8)
        
        # Draw central circle
        circle = Circle((0, 0), radius=2, fill=True, color='white')
        self.axes.add_patch(circle)
    
    def _draw_rulers(self):
        """Draw rulers for scale reference."""
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        
        # Draw tick marks every 10mm
        tick_interval = 10
        tick_size = 2
        
        # X-axis ticks
        for x in range(int(xlim[0]), int(xlim[1]) + 1, tick_interval):
            if x == 0:
                continue  # Skip origin as it has the crosshair
            self.axes.plot([x, x], [-tick_size, tick_size], color='white', linewidth=0.5)
            if x % 50 == 0:  # Label every 50mm
                self.axes.text(x, -10, f"{x}", color='white', 
                             ha='center', va='top', fontsize=8)
        
        # Y-axis ticks
        for y in range(int(ylim[0]), int(ylim[1]) + 1, tick_interval):
            if y == 0:
                continue  # Skip origin as it has the crosshair
            self.axes.plot([-tick_size, tick_size], [y, y], color='white', linewidth=0.5)
            if y % 50 == 0:  # Label every 50mm
                self.axes.text(-10, y, f"{y}", color='white', 
                             ha='right', va='center', fontsize=8)
    
    def _draw_legend(self):
        """Draw a legend for structures."""
        # Create legend patches and labels
        patches = []
        labels = []
        
        for structure in self.structures:
            if structure.id in self.structure_colors:
                color = self.structure_colors[structure.id]
                patch = Rectangle((0, 0), 1, 1, color=color, alpha=0.5)
                patches.append(patch)
                labels.append(structure.name)
        
        # Add legend to plot
        if patches:
            self.axes.legend(patches, labels, loc='lower right', 
                           facecolor='darkgray', framealpha=0.7, 
                           fontsize=8, labelcolor='white')
    
    def _project_to_bev(self, points):
        """
        Project 3D points to beam's eye view coordinates.
        
        Parameters
        ----------
        points : array-like
            3D points to project
        
        Returns
        -------
        array
            2D points in beam's eye view
        """
        # Simple parallel projection for illustration
        # In a real implementation, this would use beam direction and gantry/collimator angles
        points = np.array(points)
        
        # Default to Z projection if no beam is set
        if not self.beam:
            return points[:, :2]  # Just take X and Y coordinates
        
        # Simple projection for illustration - would need proper coordinate transform
        # based on gantry angle, collimator angle, etc.
        if hasattr(self.beam, 'gantry_angle') and hasattr(self.beam, 'collimator_angle'):
            gantry_angle = np.radians(self.beam.gantry_angle)
            collimator_angle = np.radians(self.beam.collimator_angle)
            
            # Transform to beam's eye view
            # This is a simplified projection and would need proper implementation
            # for accurate BEV coordinates based on treatment planning system conventions
            x = points[:, 0] * np.cos(gantry_angle) - points[:, 2] * np.sin(gantry_angle)
            y = points[:, 1]
            z = points[:, 0] * np.sin(gantry_angle) + points[:, 2] * np.cos(gantry_angle)
            
            # Apply collimator rotation
            x_bev = x * np.cos(collimator_angle) - y * np.sin(collimator_angle)
            y_bev = x * np.sin(collimator_angle) + y * np.cos(collimator_angle)
            
            return np.column_stack((x_bev, y_bev))
        else:
            # Default projection
            return points[:, :2]
    
    def toggle_structures(self, show):
        """Toggle visibility of structures."""
        self.show_structures = show
        self.update_view()
    
    def toggle_field(self, show):
        """Toggle visibility of treatment field."""
        self.show_field = show
        self.update_view()
    
    def toggle_mlc(self, show):
        """Toggle visibility of MLC."""
        self.show_mlc = show
        self.update_view()
    
    def toggle_jaws(self, show):
        """Toggle visibility of jaws."""
        self.show_jaws = show
        self.update_view()
    
    def toggle_grid(self, show):
        """Toggle visibility of grid."""
        self.show_grid = show
        self.update_view()
    
    def toggle_rulers(self, show):
        """Toggle visibility of rulers."""
        self.show_rulers = show
        self.update_view()
    
    def set_field_size(self, width, height):
        """
        Set the field size.
        
        Parameters
        ----------
        width : float
            Field width in mm
        height : float
            Field height in mm
        """
        self.field_size = [width, height]
        self.update_view()
    
    def export_view(self, filename, dpi=300):
        """
        Export the current view to an image file.
        
        Parameters
        ----------
        filename : str
            Output filename
        dpi : int, optional
            Resolution in dots per inch
        """
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')


class BeamEyeView(QWidget):
    """Widget for displaying and interacting with beam's eye view."""
    
    # Signals
    fieldSizeChanged = pyqtSignal(float, float)
    
    def __init__(self, parent=None):
        """Initialize the beam's eye view widget."""
        super().__init__(parent)
        
        # Create layout
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create toolbar
        toolbar = QToolBar()
        
        # Add toolbar actions
        self.struct_action = QAction("Structures", self)
        self.struct_action.setCheckable(True)
        self.struct_action.setChecked(True)
        self.struct_action.triggered.connect(self._toggle_structures)
        toolbar.addAction(self.struct_action)
        
        self.field_action = QAction("Field", self)
        self.field_action.setCheckable(True)
        self.field_action.setChecked(True)
        self.field_action.triggered.connect(self._toggle_field)
        toolbar.addAction(self.field_action)
        
        self.mlc_action = QAction("MLC", self)
        self.mlc_action.setCheckable(True)
        self.mlc_action.setChecked(True)
        self.mlc_action.triggered.connect(self._toggle_mlc)
        toolbar.addAction(self.mlc_action)
        
        self.jaws_action = QAction("Jaws", self)
        self.jaws_action.setCheckable(True)
        self.jaws_action.setChecked(True)
        self.jaws_action.triggered.connect(self._toggle_jaws)
        toolbar.addAction(self.jaws_action)
        
        toolbar.addSeparator()
        
        self.grid_action = QAction("Grid", self)
        self.grid_action.setCheckable(True)
        self.grid_action.setChecked(True)
        self.grid_action.triggered.connect(self._toggle_grid)
        toolbar.addAction(self.grid_action)
        
        self.rulers_action = QAction("Rulers", self)
        self.rulers_action.setCheckable(True)
        self.rulers_action.setChecked(True)
        self.rulers_action.triggered.connect(self._toggle_rulers)
        toolbar.addAction(self.rulers_action)
        
        toolbar.addSeparator()
        
        self.export_action = QAction("Export", self)
        self.export_action.triggered.connect(self._export_view)
        toolbar.addAction(self.export_action)
        
        # Add toolbar to layout
        main_layout.addWidget(toolbar)
        
        # Create BEV canvas
        self.bev_canvas = BEVCanvas(self)
        main_layout.addWidget(self.bev_canvas)
        
        # Create control panel
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        controls_layout = QHBoxLayout(controls_frame)
        
        # Field size controls
        field_group = QGroupBox("Field Size")
        field_layout = QFormLayout(field_group)
        
        # Width control
        width_layout = QHBoxLayout()
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(5, 400)
        self.width_slider.setValue(100)
        self.width_slider.valueChanged.connect(self._on_field_size_changed)
        width_layout.addWidget(self.width_slider, 1)
        
        self.width_label = QLabel("100 mm")
        width_layout.addWidget(self.width_label)
        
        field_layout.addRow("Width:", width_layout)
        
        # Height control
        height_layout = QHBoxLayout()
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(5, 400)
        self.height_slider.setValue(100)
        self.height_slider.valueChanged.connect(self._on_field_size_changed)
        height_layout.addWidget(self.height_slider, 1)
        
        self.height_label = QLabel("100 mm")
        height_layout.addWidget(self.height_label)
        
        field_layout.addRow("Height:", height_layout)
        
        controls_layout.addWidget(field_group)
        
        # SAD control
        sad_group = QGroupBox("Source-to-Axis Distance")
        sad_layout = QHBoxLayout(sad_group)
        
        self.sad_slider = QSlider(Qt.Horizontal)
        self.sad_slider.setRange(500, 2000)
        self.sad_slider.setValue(1000)
        self.sad_slider.valueChanged.connect(self._on_sad_changed)
        sad_layout.addWidget(self.sad_slider, 1)
        
        self.sad_label = QLabel("1000 mm")
        sad_layout.addWidget(self.sad_label)
        
        controls_layout.addWidget(sad_group)
        
        # Add controls to main layout
        main_layout.addWidget(controls_frame)
        
        # Set minimum size
        self.setMinimumSize(600, 600)
    
    def set_beam(self, beam):
        """
        Set the beam for visualization.
        
        Parameters
        ----------
        beam : Beam
            The beam to visualize
        """
        self.bev_canvas.set_beam(beam)
        
        # Update UI controls with beam parameters
        if hasattr(beam, 'field_size'):
            width, height = beam.field_size
            self.width_slider.setValue(width)
            self.height_slider.setValue(height)
            self.width_label.setText(f"{width} mm")
            self.height_label.setText(f"{height} mm")
        
        if hasattr(beam, 'sad'):
            self.sad_slider.setValue(beam.sad)
            self.sad_label.setText(f"{beam.sad} mm")
    
    def set_structures(self, structures):
        """
        Set the structures for visualization.
        
        Parameters
        ----------
        structures : list
            List of structures to visualize
        """
        self.bev_canvas.set_structures(structures)
    
    def _toggle_structures(self, checked):
        """Toggle visibility of structures."""
        self.bev_canvas.toggle_structures(checked)
    
    def _toggle_field(self, checked):
        """Toggle visibility of treatment field."""
        self.bev_canvas.toggle_field(checked)
    
    def _toggle_mlc(self, checked):
        """Toggle visibility of MLC."""
        self.bev_canvas.toggle_mlc(checked)
    
    def _toggle_jaws(self, checked):
        """Toggle visibility of jaws."""
        self.bev_canvas.toggle_jaws(checked)
    
    def _toggle_grid(self, checked):
        """Toggle visibility of grid."""
        self.bev_canvas.toggle_grid(checked)
    
    def _toggle_rulers(self, checked):
        """Toggle visibility of rulers."""
        self.bev_canvas.toggle_rulers(checked)
    
    def _on_field_size_changed(self):
        """Handle field size changes."""
        width = self.width_slider.value()
        height = self.height_slider.value()
        
        # Update labels
        self.width_label.setText(f"{width} mm")
        self.height_label.setText(f"{height} mm")
        
        # Update BEV canvas
        self.bev_canvas.set_field_size(width, height)
        
        # Emit signal
        self.fieldSizeChanged.emit(width, height)
    
    def _on_sad_changed(self, value):
        """Handle SAD changes."""
        # Update label
        self.sad_label.setText(f"{value} mm")
        
        # Update BEV canvas
        self.bev_canvas.set_sad(value)
    
    def _export_view(self):
        """Export the current view to an image file."""
        from PyQt5.QtWidgets import QFileDialog
        
        # Show file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Beam's Eye View", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if filename:
            # Add extension if needed
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename += '.png'
            
            # Export view
            self.bev_canvas.export_view(filename)

def test_beam_eye_view():
    """Test function for the beam eye view widget."""
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Beam's Eye View Test")
    window.resize(800, 800)
    
    # Create BEV widget
    bev_widget = BeamEyeView()
    
    # Create test beam
    class TestBeam:
        def __init__(self):
            self.gantry_angle = 0
            self.collimator_angle = 0
            self.sad = 1000
            self.isocenter = [0, 0, 0]
            self.field_size = [100, 100]
            self.mlc_positions = None
            self.jaw_positions = None
    
    # Create test structures
    class TestStructure:
        def __init__(self, name, id, contours=None):
            self.name = name
            self.id = id
            self.contours = contours or []
    
    # Create test data
    beam = TestBeam()
    
    # Create some test contours (simplified for example)
    ptv_contours = [
        np.array([
            [20, 30, 0],
            [20, -30, 0],
            [-20, -30, 0],
            [-20, 30, 0]
        ])
    ]
    
    cord_contours = [
        np.array([
            [5, 10, 20],
            [5, -10, 20],
            [-5, -10, 20],
            [-5, 10, 20]
        ])
    ]
    
    body_contours = [
        np.array([
            [100, 100, 0],
            [100, -100, 0],
            [-100, -100, 0],
            [-100, 100, 0]
        ])
    ]
    
    structures = [
        TestStructure("PTV", "ptv1", ptv_contours),
        TestStructure("Spinal Cord", "cord", cord_contours),
        TestStructure("BODY", "body", body_contours)
    ]
    
    # Set data in widget
    bev_widget.set_beam(beam)
    bev_widget.set_structures(structures)
    
    # Set as central widget
    window.setCentralWidget(bev_widget)
    
    window.show()
    return app.exec_()

if __name__ == "__main__":
    test_beam_eye_view() 