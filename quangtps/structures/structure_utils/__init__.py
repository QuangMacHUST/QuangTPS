#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure utilities module for QuangTPS.

This module provides various utility functions for working with anatomical structures,
including volume calculation, overlap detection, and spatial metrics.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


def calculate_volume(
    structure_mask: np.ndarray, voxel_size: Tuple[float, float, float]
) -> float:
    """
    Calculate the volume of a structure based on its mask and voxel size.

    Parameters
    ----------
    structure_mask : np.ndarray
        3D binary mask representing the structure
    voxel_size : Tuple[float, float, float]
        Size of each voxel in mm (dx, dy, dz)

    Returns
    -------
    float
        Volume in cubic centimeters (cc)
    """
    if structure_mask is None or not isinstance(structure_mask, np.ndarray):
        logger.warning("Invalid structure mask for volume calculation")
        return 0.0

    # Count non-zero voxels (voxels inside the structure)
    voxel_count = np.count_nonzero(structure_mask)

    # Calculate voxel volume in cubic mm
    voxel_volume_mm3 = voxel_size[0] * voxel_size[1] * voxel_size[2]

    # Convert to cc (1 cc = 1000 mm³)
    volume_cc = voxel_count * voxel_volume_mm3 / 1000.0

    return volume_cc


def calculate_centroid(structure_mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate the centroid (center of mass) of a structure.

    Parameters
    ----------
    structure_mask : np.ndarray
        3D binary mask representing the structure

    Returns
    -------
    Tuple[float, float, float]
        Coordinates of the centroid (x, y, z)
    """
    if structure_mask is None or not isinstance(structure_mask, np.ndarray):
        logger.warning("Invalid structure mask for centroid calculation")
        return (0.0, 0.0, 0.0)

    if np.count_nonzero(structure_mask) == 0:
        logger.warning("Empty structure mask for centroid calculation")
        return (0.0, 0.0, 0.0)

    # Get indices of non-zero elements
    indices = np.where(structure_mask > 0)

    # Calculate mean position and convert numpy types to float
    x_centroid = float(np.mean(indices[0])) if len(indices[0]) > 0 else 0.0
    y_centroid = float(np.mean(indices[1])) if len(indices[1]) > 0 else 0.0
    z_centroid = float(np.mean(indices[2])) if len(indices[2]) > 0 else 0.0

    return (x_centroid, y_centroid, z_centroid)


def calculate_overlap(
    structure1_mask: np.ndarray, structure2_mask: np.ndarray
) -> float:
    """
    Calculate the overlap volume between two structures.

    Parameters
    ----------
    structure1_mask : np.ndarray
        3D binary mask representing the first structure
    structure2_mask : np.ndarray
        3D binary mask representing the second structure

    Returns
    -------
    float
        Overlap volume in voxels
    """
    if (
        structure1_mask is None
        or not isinstance(structure1_mask, np.ndarray)
        or structure2_mask is None
        or not isinstance(structure2_mask, np.ndarray)
    ):
        logger.warning("Invalid structure masks for overlap calculation")
        return 0.0

    if structure1_mask.shape != structure2_mask.shape:
        logger.warning(
            f"Structure masks have different shapes: {structure1_mask.shape} vs {structure2_mask.shape}"
        )
        return 0.0

    # Calculate intersection
    intersection = np.logical_and(structure1_mask > 0, structure2_mask > 0)

    # Count voxels in the intersection
    overlap_volume = np.count_nonzero(intersection)

    return float(overlap_volume)


def calculate_distance(
    point1: Tuple[float, float, float], point2: Tuple[float, float, float]
) -> float:
    """
    Calculate the Euclidean distance between two points in 3D space.

    Parameters
    ----------
    point1 : Tuple[float, float, float]
        Coordinates of the first point (x, y, z)
    point2 : Tuple[float, float, float]
        Coordinates of the second point (x, y, z)

    Returns
    -------
    float
        Euclidean distance between the points
    """
    return np.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
        + (point1[2] - point2[2]) ** 2
    )


def find_closest_structure(
    target_structure_centroid: Tuple[float, float, float],
    structure_centroids: Dict[str, Tuple[float, float, float]],
) -> str:
    """
    Find the closest structure to a target structure based on centroid distances.

    Parameters
    ----------
    target_structure_centroid : Tuple[float, float, float]
        Centroid of the target structure
    structure_centroids : Dict[str, Tuple[float, float, float]]
        Dictionary mapping structure IDs to their centroids

    Returns
    -------
    str
        ID of the closest structure
    """
    if not structure_centroids:
        return ""

    closest_structure = ""
    min_distance = float("inf")

    for structure_id, centroid in structure_centroids.items():
        distance = calculate_distance(target_structure_centroid, centroid)
        if distance < min_distance:
            min_distance = distance
            closest_structure = structure_id

    return closest_structure


def calculate_surface_area(
    structure_mask: np.ndarray, voxel_size: Tuple[float, float, float]
) -> float:
    """
    Calculate the surface area of a structure.

    Parameters
    ----------
    structure_mask : np.ndarray
        3D binary mask representing the structure
    voxel_size : Tuple[float, float, float]
        Size of each voxel in mm (dx, dy, dz)

    Returns
    -------
    float
        Surface area in square centimeters (cm²)
    """
    if structure_mask is None or not isinstance(structure_mask, np.ndarray):
        logger.warning("Invalid structure mask for surface area calculation")
        return 0.0

    # Implementation using the marching cubes algorithm would be ideal
    # This is a simplified approximation using edge detection
    from scipy import ndimage

    # Apply a 3D edge detection filter
    edges = ndimage.binary_erosion(structure_mask) ^ structure_mask

    # Count edge voxels
    edge_voxel_count = np.count_nonzero(edges)

    # Approximate surface area - this is a rough approximation
    # A more accurate method would use marching cubes or similar
    avg_voxel_side = (voxel_size[0] + voxel_size[1] + voxel_size[2]) / 3
    surface_area_mm2 = edge_voxel_count * avg_voxel_side * avg_voxel_side

    # Convert to cm²
    surface_area_cm2 = surface_area_mm2 / 100.0

    return surface_area_cm2


def calculate_dice_coefficient(
    structure1_mask: np.ndarray, structure2_mask: np.ndarray
) -> float:
    """
    Calculate the Dice similarity coefficient between two structures.

    Parameters
    ----------
    structure1_mask : np.ndarray
        3D binary mask representing the first structure
    structure2_mask : np.ndarray
        3D binary mask representing the second structure

    Returns
    -------
    float
        Dice coefficient (0.0 to 1.0)
    """
    if (
        structure1_mask is None
        or not isinstance(structure1_mask, np.ndarray)
        or structure2_mask is None
        or not isinstance(structure2_mask, np.ndarray)
    ):
        logger.warning("Invalid structure masks for Dice coefficient calculation")
        return 0.0

    if structure1_mask.shape != structure2_mask.shape:
        logger.warning(
            f"Structure masks have different shapes: {structure1_mask.shape} vs {structure2_mask.shape}"
        )
        return 0.0

    # Calculate intersection
    intersection = np.logical_and(structure1_mask > 0, structure2_mask > 0)
    intersection_size = np.count_nonzero(intersection)

    # Calculate sizes of both structures
    size1 = np.count_nonzero(structure1_mask)
    size2 = np.count_nonzero(structure2_mask)

    # Handle edge case to avoid division by zero
    if size1 + size2 == 0:
        return 0.0

    # Calculate Dice coefficient: 2*|X∩Y| / (|X|+|Y|)
    dice = 2.0 * intersection_size / (size1 + size2)

    return dice
