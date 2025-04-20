"""
Structure utility functions for QuangTPS.

This module provides utility functions for manipulating and analyzing structures
in radiotherapy treatment planning.
"""

from typing import List, Tuple, Optional, Dict, Any, cast
import numpy as np
import logging
from scipy import ndimage

from quangtps.structures.structure import Structure
from quangtps.structures.structure_set import StructureSet

logger = logging.getLogger(__name__)


def create_structure_from_mask(
    mask: np.ndarray,
    name: str,
    structure_type: str = "ORGAN",
    color: Tuple[int, int, int] = (255, 0, 0),
    opacity: float = 0.5,
) -> Structure:
    """
    Create a structure from a binary mask.

    Args:
        mask: 3D binary numpy array representing the structure
        name: Name of the structure
        structure_type: Type of structure (e.g., 'PTV', 'OAR', 'ORGAN')
        color: RGB color tuple for structure visualization
        opacity: Opacity for structure visualization (0.0-1.0)

    Returns:
        Structure object initialized with the provided mask
    """
    structure = Structure(name=name, structure_type=structure_type)
    structure.set_mask(mask)
    structure.color = color
    structure.opacity = opacity
    return structure


def get_structure_volume(
    structure: Structure, voxel_spacing: Tuple[float, float, float]
) -> float:
    """
    Calculate the volume of a structure in cubic centimeters.

    Args:
        structure: Structure object
        voxel_spacing: Spacing of voxels in mm (x, y, z)

    Returns:
        Volume of the structure in cubic centimeters
    """
    if structure.mask is None:
        return 0.0

    # Convert voxel spacing from mm to cm
    spacing_cm = tuple(s / 10.0 for s in voxel_spacing)

    # Calculate voxel volume in cubic cm
    voxel_volume = spacing_cm[0] * spacing_cm[1] * spacing_cm[2]

    # Count non-zero voxels and multiply by voxel volume
    return np.count_nonzero(structure.mask) * voxel_volume


def get_structure_centroid(
    structure: Structure,
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate the centroid (center of mass) of a structure in voxel coordinates.

    Args:
        structure: Structure object

    Returns:
        Centroid coordinates (x, y, z) in voxel space, or None if mask is empty
    """
    if structure.mask is None or np.count_nonzero(structure.mask) == 0:
        return None

    # Get center of mass and return as a tuple of floats
    center = ndimage.center_of_mass(structure.mask)
    return (float(center[0]), float(center[1]), float(center[2]))


def expand_structure(
    structure: Structure, margin_mm: float, voxel_spacing: Tuple[float, float, float]
) -> Structure:
    """
    Create a new structure by expanding/contracting an existing structure.

    Args:
        structure: Structure to expand
        margin_mm: Margin to expand by in millimeters (negative for contraction)
        voxel_spacing: Spacing of voxels in mm (x, y, z)

    Returns:
        New expanded Structure object
    """
    if structure.mask is None:
        raise ValueError("Structure has no mask")

    # Convert margin from mm to voxels in each dimension
    margin_voxels = [margin_mm / spacing for spacing in voxel_spacing]

    # Use binary dilation/erosion based on margin sign
    if margin_mm >= 0:
        # Create structuring element for dilation
        struct_elem = ndimage.generate_binary_structure(3, 1)
        # Calculate number of iterations based on margin
        iterations = int(max(margin_voxels))
        new_mask = ndimage.binary_dilation(
            structure.mask, structure=struct_elem, iterations=iterations
        ).astype(np.uint8)
    else:
        # Create structuring element for erosion
        struct_elem = ndimage.generate_binary_structure(3, 1)
        # Calculate number of iterations based on margin (absolute value)
        iterations = int(max(abs(val) for val in margin_voxels))
        new_mask = ndimage.binary_erosion(
            structure.mask, structure=struct_elem, iterations=iterations
        ).astype(np.uint8)

    # Create new structure with expanded/contracted mask
    new_name = (
        f"{structure.name}+{margin_mm}mm"
        if margin_mm >= 0
        else f"{structure.name}{margin_mm}mm"
    )

    # Create new structure
    new_structure = create_structure_from_mask(
        new_mask,
        name=new_name,
        structure_type=structure.type,
        color=structure.color,
        opacity=structure.opacity,
    )

    return new_structure


def create_boolean_structure(
    structure_a: Structure,
    structure_b: Structure,
    operation: str,
    name: Optional[str] = None,
) -> Structure:
    """
    Create a new structure by performing a boolean operation between two structures.

    Args:
        structure_a: First structure
        structure_b: Second structure
        operation: Boolean operation ('AND', 'OR', 'SUB' for A-B, 'SUB_REV' for B-A, 'XOR')
        name: Name for the new structure (default: auto-generated based on operation)

    Returns:
        New Structure resulting from the boolean operation
    """
    if structure_a.mask is None or structure_b.mask is None:
        raise ValueError("Both structures must have masks")

    # Check mask dimensions match
    if structure_a.mask.shape != structure_b.mask.shape:
        raise ValueError(
            f"Structure masks have different shapes: {structure_a.mask.shape} vs {structure_b.mask.shape}"
        )

    # Perform boolean operation
    mask_a = structure_a.mask.astype(bool)
    mask_b = structure_b.mask.astype(bool)

    if operation.upper() == "AND":
        result_mask = np.logical_and(mask_a, mask_b)
        default_name = f"{structure_a.name}_AND_{structure_b.name}"
    elif operation.upper() == "OR":
        result_mask = np.logical_or(mask_a, mask_b)
        default_name = f"{structure_a.name}_OR_{structure_b.name}"
    elif operation.upper() == "SUB":
        result_mask = np.logical_and(mask_a, np.logical_not(mask_b))
        default_name = f"{structure_a.name}_SUB_{structure_b.name}"
    elif operation.upper() == "SUB_REV":
        result_mask = np.logical_and(np.logical_not(mask_a), mask_b)
        default_name = f"{structure_b.name}_SUB_{structure_a.name}"
    elif operation.upper() == "XOR":
        result_mask = np.logical_xor(mask_a, mask_b)
        default_name = f"{structure_a.name}_XOR_{structure_b.name}"
    else:
        raise ValueError(
            f"Invalid operation: {operation}. Must be one of: AND, OR, SUB, SUB_REV, XOR"
        )

    # Convert boolean mask to uint8
    result_mask = result_mask.astype(np.uint8)

    # Set name for the new structure
    if name is None:
        name = default_name

    # Create new structure
    new_structure = create_structure_from_mask(
        result_mask,
        name=name,
        structure_type="DERIVED",
        color=structure_a.color,
        opacity=structure_a.opacity,
    )

    return new_structure
