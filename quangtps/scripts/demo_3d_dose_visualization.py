#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script for 3D dose visualization in QuangTPS.

This script demonstrates the enhanced 3D dose visualization capabilities
in QuangTPS, including interactive isodose control, structure visualization,
and multiple viewing modes.
"""

import os
import sys
import logging
import numpy as np
import SimpleITK as sitk

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from quangtps.ui.dose_visualization_3d import DoseVisualization3D
from quangtps.dose.dose_grid import DoseGrid
from quangtps.core.structures import Structure, StructureSet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo_3d_dose")


def create_synthetic_dose_grid():
    """Create a synthetic dose grid for demonstration."""
    # Create a 3D array for dose
    size = 128
    dose_array = np.zeros((size, size, size), dtype=np.float32)
    center = np.array([size // 2, size // 2, size // 2])

    # Create a dose distribution that looks like a treatment plan
    # Main target area
    for x in range(size):
        for y in range(size):
            for z in range(size):
                pos = np.array([x, y, z])
                # Create a primary beam path from one direction
                if abs(pos[0] - center[0]) < 20 and pos[1] < center[1]:
                    falloff = 1.0 - 0.01 * abs(pos[1] - center[1]) / center[1]
                    dose_array[x, y, z] += max(0, 3.0 * falloff)

                # Create another beam path at an angle
                if (
                    abs((pos[0] - center[0]) * 0.7 + (pos[1] - center[1]) * 0.7) < 20
                    and pos[1] > center[1]
                ):
                    falloff = 1.0 - 0.01 * abs(pos[1] - center[1]) / center[1]
                    dose_array[x, y, z] += max(0, 3.0 * falloff)

                # Central high dose region (target)
                dist = np.sqrt(((pos - center) ** 2).sum())
                if dist < 20:
                    dose_array[x, y, z] = max(
                        dose_array[x, y, z], 5.0 * (1.0 - 0.8 * dist / 20.0)
                    )

    # Create DoseGrid object
    dose_grid = DoseGrid(dose_array)
    dose_grid.spacing = (1.0, 1.0, 1.0)  # 1mm spacing
    dose_grid.origin = (-size // 2, -size // 2, -size // 2)  # Center at origin
    dose_grid.prescription_dose = 5.0  # Gy

    return dose_grid


def create_synthetic_structures():
    """Create synthetic structures for demonstration."""
    size = 128
    center = np.array([size // 2, size // 2, size // 2])

    # Create a StructureSet
    structure_set = StructureSet()

    # Create PTV structure
    ptv_mask = np.zeros((size, size, size), dtype=bool)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                pos = np.array([x, y, z])
                dist = np.sqrt(((pos - center) ** 2).sum())
                if dist < 20:
                    ptv_mask[x, y, z] = True

    ptv = Structure(
        id="PTV",
        name="PTV",
        mask=ptv_mask,
        color=(1.0, 0.0, 0.0),  # Red
        type="PTV",
    )
    structure_set.add_structure(ptv)

    # Create OAR structure 1 (e.g., spinal cord)
    oar1_mask = np.zeros((size, size, size), dtype=bool)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Create a cylindrical structure behind the target
                if (x - center[0]) ** 2 + (z - center[2]) ** 2 < 10**2 and y > center[
                    1
                ] + 25:
                    oar1_mask[x, y, z] = True

    oar1 = Structure(
        id="SpinalCord",
        name="Spinal Cord",
        mask=oar1_mask,
        color=(0.0, 0.0, 1.0),  # Blue
        type="OAR",
    )
    structure_set.add_structure(oar1)

    # Create OAR structure 2 (e.g., parotid)
    oar2_mask = np.zeros((size, size, size), dtype=bool)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Create a spherical structure to the side of the target
                dist = np.sqrt(
                    ((np.array([x, y, z]) - (center + np.array([30, 0, 0]))) ** 2).sum()
                )
                if dist < 15:
                    oar2_mask[x, y, z] = True

    oar2 = Structure(
        id="Parotid",
        name="Parotid",
        mask=oar2_mask,
        color=(0.0, 1.0, 0.0),  # Green
        type="OAR",
    )
    structure_set.add_structure(oar2)

    # Create body contour
    body_mask = np.zeros((size, size, size), dtype=bool)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Create an ellipsoidal body contour
                normalized_pos = np.array(
                    [
                        (x - center[0]) / (size * 0.4),
                        (y - center[1]) / (size * 0.4),
                        (z - center[2]) / (size * 0.4),
                    ]
                )
                if np.sum(normalized_pos**2) < 1.0:
                    body_mask[x, y, z] = True

    body = Structure(
        id="Body",
        name="Body",
        mask=body_mask,
        color=(0.8, 0.8, 0.6),  # Tan
        type="EXTERNAL",
    )
    structure_set.add_structure(body)

    return structure_set


def create_synthetic_image():
    """Create a synthetic CT image for demonstration."""
    size = 128
    spacing = (1.0, 1.0, 1.0)  # 1mm spacing
    origin = (-size // 2, -size // 2, -size // 2)  # Center at origin

    # Create an array with water density (0 HU)
    image_array = np.zeros((size, size, size), dtype=np.float32)
    center = np.array([size // 2, size // 2, size // 2])

    # Add body profile (soft tissue)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Create an ellipsoidal body
                normalized_pos = np.array(
                    [
                        (x - center[0]) / (size * 0.4),
                        (y - center[1]) / (size * 0.4),
                        (z - center[2]) / (size * 0.4),
                    ]
                )
                if np.sum(normalized_pos**2) < 1.0:
                    image_array[x, y, z] = 40  # Soft tissue (40 HU)

    # Add bone structure
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Spine
                if (x - center[0]) ** 2 + (z - center[2]) ** 2 < 8**2 and y > center[
                    1
                ] + 25:
                    image_array[x, y, z] = 400  # Bone (400 HU)

                # Skull-like structure
                dist_from_center = np.sqrt(((np.array([x, y, z]) - center) ** 2).sum())
                if (
                    dist_from_center > 35
                    and dist_from_center < 40
                    and y < center[1] - 10
                ):
                    image_array[x, y, z] = 400  # Bone (400 HU)

    # Add air cavities
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Air cavity in the center of the head
                if ((x - center[0]) ** 2 + (z - center[2]) ** 2) < 12**2 and y < center[
                    1
                ] - 25:
                    image_array[x, y, z] = -1000  # Air (-1000 HU)

    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)

    return sitk_image


def run_demo():
    """Run the 3D dose visualization demo."""
    logger.info("Starting 3D dose visualization demo")

    # Create QApplication
    app = QApplication(sys.argv)

    # Create demo window
    widget = DoseVisualization3D()
    widget.setWindowTitle("QuangTPS 3D Dose Visualization Demo")
    widget.resize(1200, 800)

    # Create synthetic data
    logger.info("Creating synthetic data")
    dose_grid = create_synthetic_dose_grid()
    structure_set = create_synthetic_structures()
    image = create_synthetic_image()

    # Set prescription dose
    if hasattr(widget, "prescription_spinbox"):
        widget.prescription_spinbox.setValue(5.0)  # 5 Gy prescription

    # Set image data
    logger.info("Setting image data")
    image_array = sitk.GetArrayFromImage(image)
    widget.set_image_data(image_array)

    # Add structures
    logger.info("Adding structures")
    for structure in structure_set.structures:
        widget.add_structure(
            structure.id,
            structure.mask,
            color=structure.color,
            opacity=0.5,
            name=structure.name,
        )

    # Set dose grid
    logger.info("Setting dose grid")
    widget.set_dose_grid(dose_grid)

    # Show the widget
    widget.show()

    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_demo()
