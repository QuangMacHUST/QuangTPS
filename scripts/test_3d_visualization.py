#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for 3D beam visualization.

This script creates a simple test case with a water phantom, treatment beams,
and dose distribution to demonstrate the 3D visualization capabilities.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import from quangtps
script_dir = Path(__file__).parent.absolute()
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
    from PyQt5.QtCore import Qt
except ImportError:
    print("Error: PyQt5 is required for this script.")
    sys.exit(1)

# Import required QuangTPS modules (after path is set)
try:
    from quangtps.ui.beam_3d_visualization import Beam3DVisualization
    from quangtps.imaging.image import Image
    from quangtps.imaging.structures import Structure, StructureSet
    from quangtps.planning.beam import Beam
    from quangtps.treatment.plan import Plan
    from quangtps.dose.dose_grid import DoseGrid
except ImportError as e:
    print(f"Error importing QuangTPS modules: {e}")
    sys.exit(1)

def create_water_phantom(size=(300, 300, 300), spacing=(2, 2, 2)):
    """
    Create a simple water phantom.
    
    Parameters
    ----------
    size : tuple
        Size of phantom in mm (width, height, depth)
    spacing : tuple
        Voxel spacing in mm (dx, dy, dz)
        
    Returns
    -------
    Image
        Image object containing the phantom data
    """
    # Calculate dimensions
    dims = tuple(int(s / sp) for s, sp in zip(size, spacing))
    
    # Create a homogeneous water phantom (HU = 0)
    data = np.zeros(dims, dtype=np.float32)
    
    # Add some heterogeneities
    # Bone (HU ~ 1000)
    center = tuple(d // 2 for d in dims)
    radius = min(dims) // 10
    x, y, z = np.indices(dims)
    bone_sphere = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
    data[bone_sphere] = 1000
    
    # Air cavity (HU ~ -1000)
    center2 = (center[0], center[1], center[2] + radius * 2)
    radius2 = radius // 2
    air_sphere = (x - center2[0])**2 + (y - center2[1])**2 + (z - center2[2])**2 <= radius2**2
    data[air_sphere] = -1000
    
    # Create image object
    image = Image()
    image.data = data
    image.spacing = spacing
    image.modality = "CT"
    
    return image

def create_structures(image):
    """
    Create test structures for the phantom.
    
    Parameters
    ----------
    image : Image
        The phantom image
        
    Returns
    -------
    StructureSet
        Structure set containing test structures
    """
    # Get image dimensions
    dims = image.data.shape
    
    # Create structure set
    structure_set = StructureSet()
    structure_set.structures = []
    
    # Create external contour (body)
    body = Structure()
    body.name = "Body"
    body.color = [0.2, 0.8, 0.2]  # Green
    body.mask = np.ones(dims, dtype=bool)
    margin = 2  # 2 voxel margin
    body.mask[:margin, :, :] = False
    body.mask[-margin:, :, :] = False
    body.mask[:, :margin, :] = False
    body.mask[:, -margin:, :] = False
    body.mask[:, :, :margin] = False
    body.mask[:, :, -margin:] = False
    structure_set.structures.append(body)
    
    # Create a target structure (PTV)
    ptv = Structure()
    ptv.name = "PTV"
    ptv.color = [0.8, 0.2, 0.2]  # Red
    ptv.mask = np.zeros(dims, dtype=bool)
    center = tuple(d // 2 for d in dims)
    radius = min(dims) // 6
    x, y, z = np.indices(dims)
    ptv.mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
    structure_set.structures.append(ptv)
    
    # Create an OAR
    oar = Structure()
    oar.name = "OAR"
    oar.color = [0.2, 0.2, 0.8]  # Blue
    oar.mask = np.zeros(dims, dtype=bool)
    center2 = (center[0] + radius, center[1], center[2])
    radius2 = radius // 2
    oar.mask = (x - center2[0])**2 + (y - center2[1])**2 + (z - center2[2])**2 <= radius2**2
    structure_set.structures.append(oar)
    
    return structure_set

def create_beams(isocenter=(150, 150, 150)):
    """
    Create test beams for the phantom.
    
    Parameters
    ----------
    isocenter : tuple
        Isocenter position in mm
        
    Returns
    -------
    list
        List of Beam objects
    """
    beams = []
    
    # Create a set of test beams
    beam_params = [
        {"name": "AP", "gantry_angle": 0, "collimator_angle": 0, "couch_angle": 0, "field_size": (100, 100)},
        {"name": "PA", "gantry_angle": 180, "collimator_angle": 0, "couch_angle": 0, "field_size": (100, 100)},
        {"name": "LLAT", "gantry_angle": 270, "collimator_angle": 0, "couch_angle": 0, "field_size": (100, 100)},
        {"name": "RLAT", "gantry_angle": 90, "collimator_angle": 0, "couch_angle": 0, "field_size": (100, 100)},
        {"name": "RAO", "gantry_angle": 45, "collimator_angle": 0, "couch_angle": 0, "field_size": (100, 100)},
        {"name": "RPO", "gantry_angle": 315, "collimator_angle": 0, "couch_angle": 0, "field_size": (100, 100)},
    ]
    
    for i, params in enumerate(beam_params):
        beam = Beam()
        beam.name = params["name"]
        beam.gantry_angle = params["gantry_angle"]
        beam.collimator_angle = params["collimator_angle"]
        beam.couch_angle = params["couch_angle"]
        beam.field_size = params["field_size"]
        beam.isocenter = isocenter
        beam.energy = 6  # 6 MV
        beam.ssd = 900  # 900 mm SSD
        beam.weight = 1.0  # Equal weight
        beam.mu = 100  # 100 monitor units
        beams.append(beam)
    
    return beams

def create_dose_grid(image, beams):
    """
    Create a simple dose distribution.
    
    Parameters
    ----------
    image : Image
        The phantom image
    beams : list
        List of Beam objects
        
    Returns
    -------
    DoseGrid
        Dose grid containing a simple dose distribution
    """
    # Create a dose grid with the same dimensions as the image
    dose_grid = DoseGrid()
    dose_grid.data = np.zeros_like(image.data)
    dose_grid.spacing = image.spacing
    
    # Get image dimensions
    dims = image.data.shape
    center = tuple(d // 2 for d in dims)
    
    # Create a simple dose distribution for each beam
    for beam in beams:
        # Convert angles to radians
        gantry_rad = np.radians(beam.gantry_angle)
        
        # Create a simple dose falloff along the beam direction
        x, y, z = np.indices(dims)
        
        # Calculate distance from each point to the beam central axis
        # Simplified for demonstration - in reality, this would use ray tracing
        # and actual beam modeling
        
        # Define beam direction vector based on gantry angle
        dx = np.sin(gantry_rad)
        dy = 0
        dz = -np.cos(gantry_rad)
        
        # Calculate perpendicular distance to beam axis
        # This is a simplification - real dose would be more complex
        beam_origin = np.array(beam.isocenter) - np.array([dx, dy, dz]) * 1000  # Point 1 m along beam axis
        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
        
        # Calculate distance from each point to the line
        v = np.array([dx, dy, dz])
        a = points - beam_origin
        dist = np.linalg.norm(a - np.dot(a, v).reshape(-1, 1) * v, axis=1)
        dist = dist.reshape(dims)
        
        # Create exponential falloff from beam axis
        sigma = 50  # Spread parameter
        beam_dose = np.exp(-dist**2 / (2 * sigma**2))
        
        # Add depth dose falloff
        depth = np.abs((x - center[0]) * dx + (y - center[1]) * dy + (z - center[2]) * dz)
        depth_factor = np.exp(-depth / 200)  # Simple exponential attenuation
        
        beam_dose = beam_dose * depth_factor
        
        # Add to total dose (weighted by beam weight)
        dose_grid.data += beam_dose * beam.weight
    
    # Normalize to a max dose of 100%
    if np.max(dose_grid.data) > 0:
        dose_grid.data = dose_grid.data / np.max(dose_grid.data) * 100
    
    return dose_grid

def create_plan(image, structures, beams, dose_grid):
    """
    Create a test treatment plan.
    
    Parameters
    ----------
    image : Image
        The phantom image
    structures : StructureSet
        Structure set containing contours
    beams : list
        List of Beam objects
    dose_grid : DoseGrid
        Dose grid containing the dose distribution
        
    Returns
    -------
    Plan
        Treatment plan object
    """
    plan = Plan()
    plan.name = "Test 3D-CRT Plan"
    plan.description = "Test plan for 3D visualization"
    plan.technique = "3D-CRT"
    plan.beams = beams
    plan.image = image
    plan.structures = structures
    plan.dose_grid = dose_grid
    
    return plan

class TestWindow(QMainWindow):
    """Main window for testing the 3D visualization."""
    
    def __init__(self):
        """Initialize the test window."""
        super().__init__()
        
        self.setWindowTitle("QuangTPS 3D Visualization Test")
        self.resize(1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create phantom and test data
        self.image = create_water_phantom()
        self.structures = create_structures(self.image)
        self.beams = create_beams()
        self.dose_grid = create_dose_grid(self.image, self.beams)
        self.plan = create_plan(self.image, self.structures, self.beams, self.dose_grid)
        
        # Create label for information
        info_label = QLabel("Test data: Water phantom with 6 beams (AP, PA, LLAT, RLAT, RAO, RPO)")
        layout.addWidget(info_label)
        
        # Create the 3D visualization widget
        try:
            self.visualization = Beam3DVisualization()
            
            # Set data
            self.visualization.set_patient_data(self.image, self.structures)
            self.visualization.set_beams(self.beams)
            self.visualization.set_dose_grid(self.dose_grid)
            
            layout.addWidget(self.visualization)
            
        except Exception as e:
            error_label = QLabel(f"Error initializing 3D visualization: {str(e)}\n\nPlease make sure PyVista, PyVistaQt, and VTK are installed.")
            error_label.setStyleSheet("color: red;")
            layout.addWidget(error_label)
        
        # Add a screenshot button
        screenshot_button = QPushButton("Take Screenshot")
        screenshot_button.clicked.connect(self.take_screenshot)
        layout.addWidget(screenshot_button)
    
    def take_screenshot(self):
        """Take a screenshot of the 3D visualization."""
        try:
            if hasattr(self, 'visualization'):
                filename = self.visualization.take_screenshot()
                if filename:
                    print(f"Screenshot saved to: {filename}")
        except Exception as e:
            print(f"Error taking screenshot: {str(e)}")

def main():
    """Main function."""
    # Check for dependencies
    try:
        import pyvista
        import pyvistaqt
        import vtk
    except ImportError:
        print("Warning: 3D visualization dependencies (PyVista, PyVistaQt, VTK) are not installed.")
        print("You can install them with: pip install pyvista pyvistaqt vtk")
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = TestWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 