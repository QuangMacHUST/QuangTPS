#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for contour measurements and statistics.

This module provides functionality for calculating various metrics and statistics
of contours, including area, perimeter, volume, and DVH (Dose Volume Histogram)
calculations for radiotherapy treatment planning.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from enum import Enum
from scipy import interpolate, ndimage
from skimage import measure

logger = logging.getLogger(__name__)


class MeasurementType(str, Enum):
    """Enum for different measurement types."""
    AREA = "AREA"  # Area of contour
    PERIMETER = "PERIMETER"  # Perimeter of contour
    VOLUME = "VOLUME"  # Volume of structure
    MEAN_DOSE = "MEAN_DOSE"  # Mean dose within structure
    MIN_DOSE = "MIN_DOSE"  # Minimum dose within structure
    MAX_DOSE = "MAX_DOSE"  # Maximum dose within structure
    INTEGRAL_DOSE = "INTEGRAL_DOSE"  # Integral dose within structure
    HOMOGENEITY_INDEX = "HOMOGENEITY_INDEX"  # Homogeneity index
    CONFORMITY_INDEX = "CONFORMITY_INDEX"  # Conformity index
    GRADIENT_INDEX = "GRADIENT_INDEX"  # Gradient index


class ContourMeasurements:
    """
    Class for calculating and managing contour measurements.
    
    This class provides methods for calculating various metrics for 2D contours
    and 3D structures, as well as for generating visualizations of these metrics.
    """
    
    def __init__(self):
        """Initialize contour measurements."""
        self.measurement_cache = {}  # Cache for measurement results
    
    def clear_cache(self):
        """Clear measurement cache."""
        self.measurement_cache = {}
        logger.info("Cleared measurement cache")
    
    def calculate_area(self, contour: np.ndarray, 
                     pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> float:
        """
        Calculate the area of a 2D contour.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
        pixel_spacing : Tuple[float, float], optional
            Pixel spacing in mm
            
        Returns
        -------
        float
            Area in mm²
        """
        # Check if contour is valid
        if contour is None or len(contour) < 3:
            return 0.0
        
        # Calculate area using shoelace formula
        x = contour[:, 0] * pixel_spacing[0]
        y = contour[:, 1] * pixel_spacing[1]
        
        # Shoelace formula for area
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        return area
    
    def calculate_perimeter(self, contour: np.ndarray,
                         pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> float:
        """
        Calculate the perimeter of a 2D contour.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
        pixel_spacing : Tuple[float, float], optional
            Pixel spacing in mm
            
        Returns
        -------
        float
            Perimeter in mm
        """
        # Check if contour is valid
        if contour is None or len(contour) < 3:
            return 0.0
        
        # Scale coordinates by pixel spacing
        x = contour[:, 0] * pixel_spacing[0]
        y = contour[:, 1] * pixel_spacing[1]
        
        # Calculate distances between consecutive points
        dx = np.diff(np.append(x, x[0]))
        dy = np.diff(np.append(y, y[0]))
        
        # Sum of Euclidean distances
        perimeter = np.sum(np.sqrt(dx**2 + dy**2))
        
        return perimeter
    
    def calculate_centroid(self, contour: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the centroid of a 2D contour.
        
        Parameters
        ----------
        contour : np.ndarray
            Contour points as nx2 array
            
        Returns
        -------
        Tuple[float, float]
            Centroid coordinates (x, y)
        """
        # Check if contour is valid
        if contour is None or len(contour) < 3:
            return (0.0, 0.0)
        
        # Calculate centroid as mean of all points
        centroid_x = np.mean(contour[:, 0])
        centroid_y = np.mean(contour[:, 1])
        
        return (centroid_x, centroid_y)
    
    def calculate_volume(self, contours: Dict[int, np.ndarray], 
                       slice_thickness: float,
                       pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> float:
        """
        Calculate the volume of a 3D structure from contours on slices.
        
        Parameters
        ----------
        contours : Dict[int, np.ndarray]
            Dictionary mapping slice indices to contour points
        slice_thickness : float
            Thickness of each slice in mm
        pixel_spacing : Tuple[float, float], optional
            Pixel spacing in mm
            
        Returns
        -------
        float
            Volume in mm³
        """
        # Check if there are enough contours
        if not contours or len(contours) < 1:
            return 0.0
        
        # Calculate area for each slice
        areas = {}
        for slice_idx, contour in contours.items():
            areas[slice_idx] = self.calculate_area(contour, pixel_spacing)
        
        # Sort slice indices
        slice_indices = sorted(areas.keys())
        
        if len(slice_indices) == 1:
            # Single slice: use area * slice thickness
            return areas[slice_indices[0]] * slice_thickness
        
        # Multiple slices: integrate areas
        volume = 0.0
        
        for i in range(len(slice_indices) - 1):
            idx1 = slice_indices[i]
            idx2 = slice_indices[i + 1]
            
            area1 = areas[idx1]
            area2 = areas[idx2]
            
            # Distance between slices
            z_distance = (idx2 - idx1) * slice_thickness
            
            # Trapezoidal rule for integration
            volume += 0.5 * (area1 + area2) * z_distance
        
        return volume
    
    def calculate_structure_statistics(self, contours: Dict[int, np.ndarray],
                                    dose_data: Optional[np.ndarray] = None,
                                    slice_thickness: float = 1.0,
                                    pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> Dict:
        """
        Calculate comprehensive statistics for a structure.
        
        Parameters
        ----------
        contours : Dict[int, np.ndarray]
            Dictionary mapping slice indices to contour points
        dose_data : np.ndarray, optional
            3D dose data array
        slice_thickness : float, optional
            Thickness of each slice in mm
        pixel_spacing : Tuple[float, float], optional
            Pixel spacing in mm
            
        Returns
        -------
        Dict
            Dictionary of calculated statistics
        """
        # Basic structure statistics
        volume = self.calculate_volume(contours, slice_thickness, pixel_spacing)
        
        # Initialize statistics dictionary
        stats = {
            "volume_mm3": volume,
            "volume_cc": volume / 1000.0,  # Convert to cc
            "num_slices": len(contours),
            "slice_indices": sorted(contours.keys())
        }
        
        # Add per-slice statistics
        stats["slice_stats"] = {}
        
        for slice_idx, contour in contours.items():
            area = self.calculate_area(contour, pixel_spacing)
            perimeter = self.calculate_perimeter(contour, pixel_spacing)
            centroid = self.calculate_centroid(contour)
            
            stats["slice_stats"][slice_idx] = {
                "area_mm2": area,
                "perimeter_mm": perimeter,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "num_points": len(contour)
            }
        
        # If dose data is provided, calculate dose statistics
        if dose_data is not None:
            dose_stats = self.calculate_dose_statistics(contours, dose_data, 
                                                     slice_thickness, pixel_spacing)
            stats.update(dose_stats)
        
        return stats
    
    def calculate_dose_statistics(self, contours: Dict[int, np.ndarray],
                              dose_data: np.ndarray,
                              slice_thickness: float = 1.0,
                              pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> Dict:
        """
        Calculate dose statistics for a structure.
        
        Parameters
        ----------
        contours : Dict[int, np.ndarray]
            Dictionary mapping slice indices to contour points
        dose_data : np.ndarray
            3D dose data array
        slice_thickness : float, optional
            Thickness of each slice in mm
        pixel_spacing : Tuple[float, float], optional
            Pixel spacing in mm
            
        Returns
        -------
        Dict
            Dictionary of dose statistics
        """
        # Check if contours and dose data are valid
        if not contours or dose_data is None:
            return {}
        
        # Create a mask for the structure
        mask = self.create_structure_mask(contours, dose_data.shape)
        
        # Extract dose values within the structure
        dose_values = dose_data[mask > 0]
        
        if len(dose_values) == 0:
            return {
                "dose_stats": {
                    "min_dose": 0.0,
                    "max_dose": 0.0,
                    "mean_dose": 0.0,
                    "median_dose": 0.0,
                    "d95": 0.0,
                    "d90": 0.0,
                    "d50": 0.0,
                    "v95": 0.0,
                    "v90": 0.0,
                    "v50": 0.0
                }
            }
        
        # Calculate basic dose statistics
        min_dose = np.min(dose_values)
        max_dose = np.max(dose_values)
        mean_dose = np.mean(dose_values)
        median_dose = np.median(dose_values)
        
        # Calculate dose-volume metrics
        sorted_dose = np.sort(dose_values)
        num_voxels = len(dose_values)
        
        # Dose received by X% of volume (D95, D90, D50)
        d95_idx = int(0.05 * num_voxels)  # Dose received by 95% of volume
        d90_idx = int(0.10 * num_voxels)  # Dose received by 90% of volume
        d50_idx = int(0.50 * num_voxels)  # Dose received by 50% of volume
        
        d95 = sorted_dose[d95_idx] if d95_idx < num_voxels else min_dose
        d90 = sorted_dose[d90_idx] if d90_idx < num_voxels else min_dose
        d50 = sorted_dose[d50_idx] if d50_idx < num_voxels else min_dose
        
        # Percentage of volume receiving X% of prescription dose
        # Assuming max_dose is the prescription dose for this example
        v95 = np.sum(dose_values >= 0.95 * max_dose) / num_voxels * 100.0
        v90 = np.sum(dose_values >= 0.90 * max_dose) / num_voxels * 100.0
        v50 = np.sum(dose_values >= 0.50 * max_dose) / num_voxels * 100.0
        
        # Homogeneity index (D5 - D95) / D50
        d5_idx = int(0.95 * num_voxels)  # Dose received by 5% of volume
        d5 = sorted_dose[d5_idx] if d5_idx < num_voxels else max_dose
        homogeneity_index = (d5 - d95) / d50 if d50 > 0 else 0.0
        
        # Return dose statistics
        return {
            "dose_stats": {
                "min_dose": float(min_dose),
                "max_dose": float(max_dose),
                "mean_dose": float(mean_dose),
                "median_dose": float(median_dose),
                "d95": float(d95),
                "d90": float(d90),
                "d50": float(d50),
                "v95": float(v95),
                "v90": float(v90),
                "v50": float(v50),
                "homogeneity_index": float(homogeneity_index)
            }
        }
    
    def create_structure_mask(self, contours: Dict[int, np.ndarray], 
                           shape: Tuple[int, ...]) -> np.ndarray:
        """
        Create a binary mask for a structure.
        
        Parameters
        ----------
        contours : Dict[int, np.ndarray]
            Dictionary mapping slice indices to contour points
        shape : Tuple[int, ...]
            Shape of the output mask
            
        Returns
        -------
        np.ndarray
            Binary mask for the structure
        """
        # Create empty mask
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Fill mask for each slice
        for slice_idx, contour in contours.items():
            if slice_idx < 0 or slice_idx >= shape[0]:
                continue
            
            # Create a 2D binary mask from contour
            slice_mask = np.zeros((shape[1], shape[2]), dtype=np.uint8)
            
            # Convert contour to polygon for rasterization
            # First ensure contour is closed
            if np.any(contour[0] != contour[-1]):
                closed_contour = np.vstack([contour, contour[0]])
            else:
                closed_contour = contour
            
            # Round to integer coordinates
            polygon = np.round(closed_contour).astype(int)
            
            # Create rasterized polygon
            rr, cc = measure.grid_points_in_poly((shape[1], shape[2]), polygon)
            
            # Ensure coordinates are within bounds
            valid_idx = (rr >= 0) & (rr < shape[1]) & (cc >= 0) & (cc < shape[2])
            if np.any(valid_idx):
                slice_mask[rr[valid_idx], cc[valid_idx]] = 1
            
            # Add to 3D mask
            mask[slice_idx] = slice_mask
        
        return mask
    
    def calculate_dvh(self, contours: Dict[int, np.ndarray],
                   dose_data: np.ndarray,
                   dose_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Dose Volume Histogram (DVH) for a structure.
        
        Parameters
        ----------
        contours : Dict[int, np.ndarray]
            Dictionary mapping slice indices to contour points
        dose_data : np.ndarray
            3D dose data array
        dose_bins : int, optional
            Number of bins for the histogram
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Dose values and corresponding volume percentages
        """
        # Create a mask for the structure
        mask = self.create_structure_mask(contours, dose_data.shape)
        
        # Extract dose values within the structure
        dose_values = dose_data[mask > 0]
        
        if len(dose_values) == 0:
            return np.array([]), np.array([])
        
        # Calculate dose range
        min_dose = np.min(dose_values)
        max_dose = np.max(dose_values)
        
        # Create dose bins
        dose_range = np.linspace(min_dose, max_dose, dose_bins)
        
        # Calculate histogram
        hist, _ = np.histogram(dose_values, bins=dose_range)
        
        # Convert to cumulative histogram (DVH)
        dvh = np.cumsum(hist[::-1])[::-1]
        
        # Normalize to percentage of total volume
        dvh = dvh / len(dose_values) * 100.0
        
        # Return dose values and corresponding volume percentages
        return dose_range[:-1], dvh
    
    def plot_dvh(self, contours: Dict[int, np.ndarray],
              dose_data: np.ndarray,
              structure_name: str = "Structure",
              color: str = "blue",
              dose_bins: int = 100) -> plt.Figure:
        """
        Plot the Dose Volume Histogram (DVH) for a structure.
        
        Parameters
        ----------
        contours : Dict[int, np.ndarray]
            Dictionary mapping slice indices to contour points
        dose_data : np.ndarray
            3D dose data array
        structure_name : str, optional
            Name of the structure
        color : str, optional
            Color for the plot
        dose_bins : int, optional
            Number of bins for the histogram
            
        Returns
        -------
        plt.Figure
            Matplotlib figure with the DVH plot
        """
        # Calculate DVH
        dose_values, volume_pct = self.calculate_dvh(contours, dose_data, dose_bins)
        
        if len(dose_values) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlabel("Dose (Gy)")
            ax.set_ylabel("Volume (%)")
            ax.set_title(f"DVH for {structure_name}")
            ax.text(0.5, 0.5, "No dose data available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dose_values, volume_pct, color=color, linewidth=2, label=structure_name)
        
        # Add labels and title
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.set_title(f"DVH for {structure_name}")
        ax.grid(True)
        ax.legend()
        
        # Add key dose metrics as vertical lines
        dose_stats = self.calculate_dose_statistics(contours, dose_data)
        if "dose_stats" in dose_stats:
            d95 = dose_stats["dose_stats"]["d95"]
            ax.axvline(x=d95, color=color, linestyle='--', 
                      label=f"D95 = {d95:.2f} Gy")
        
        # Set y-axis from 0 to 100%
        ax.set_ylim(0, 100)
        
        return fig


class ComparisonMetrics:
    """
    Class for comparing contours and calculating comparison metrics.
    
    This class provides methods for calculating various metrics for comparing
    contours, such as Dice coefficient, Hausdorff distance, and overlap metrics.
    """
    
    def __init__(self):
        """Initialize comparison metrics."""
        pass
    
    def calculate_dice_coefficient(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate the Dice coefficient between two binary masks.
        
        Parameters
        ----------
        mask1 : np.ndarray
            First binary mask
        mask2 : np.ndarray
            Second binary mask
            
        Returns
        -------
        float
            Dice coefficient (0 to 1)
        """
        # Ensure masks are binary
        mask1_binary = mask1 > 0
        mask2_binary = mask2 > 0
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1_binary, mask2_binary).sum()
        total = mask1_binary.sum() + mask2_binary.sum()
        
        # Calculate Dice coefficient
        if total == 0:
            return 0.0
        
        dice = 2.0 * intersection / total
        
        return float(dice)
    
    def calculate_hausdorff_distance(self, contour1: np.ndarray, contour2: np.ndarray,
                                  pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> float:
        """
        Calculate the Hausdorff distance between two contours.
        
        Parameters
        ----------
        contour1 : np.ndarray
            First contour as nx2 array
        contour2 : np.ndarray
            Second contour as nx2 array
        pixel_spacing : Tuple[float, float], optional
            Pixel spacing in mm
            
        Returns
        -------
        float
            Hausdorff distance in mm
        """
        from scipy.spatial.distance import cdist
        
        # Check if contours are valid
        if contour1 is None or len(contour1) < 3 or contour2 is None or len(contour2) < 3:
            return float('inf')
        
        # Scale coordinates by pixel spacing
        contour1_scaled = contour1.copy()
        contour1_scaled[:, 0] *= pixel_spacing[0]
        contour1_scaled[:, 1] *= pixel_spacing[1]
        
        contour2_scaled = contour2.copy()
        contour2_scaled[:, 0] *= pixel_spacing[0]
        contour2_scaled[:, 1] *= pixel_spacing[1]
        
        # Calculate pairwise distances
        distances = cdist(contour1_scaled, contour2_scaled)
        
        # Calculate Hausdorff distance
        h1 = np.max(np.min(distances, axis=1))
        h2 = np.max(np.min(distances, axis=0))
        
        hausdorff = max(h1, h2)
        
        return float(hausdorff)
    
    def compare_contours(self, contour1: np.ndarray, contour2: np.ndarray,
                       shape: Tuple[int, int],
                       pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> Dict:
        """
        Compare two contours and calculate various metrics.
        
        Parameters
        ----------
        contour1 : np.ndarray
            First contour as nx2 array
        contour2 : np.ndarray
            Second contour as nx2 array
        shape : Tuple[int, int]
            Shape of the image
        pixel_spacing : Tuple[float, float], optional
            Pixel spacing in mm
            
        Returns
        -------
        Dict
            Dictionary of comparison metrics
        """
        # Create masks from contours
        measurements = ContourMeasurements()
        mask1 = np.zeros(shape, dtype=np.uint8)
        mask2 = np.zeros(shape, dtype=np.uint8)
        
        # Create binary masks using rasterization
        # Assuming create_structure_mask can be adapted for 2D
        slice_contours1 = {0: contour1}
        slice_contours2 = {0: contour2}
        
        mask1_3d = measurements.create_structure_mask(slice_contours1, (1, shape[0], shape[1]))
        mask2_3d = measurements.create_structure_mask(slice_contours2, (1, shape[0], shape[1]))
        
        mask1 = mask1_3d[0]
        mask2 = mask2_3d[0]
        
        # Calculate area overlap
        area1 = measurements.calculate_area(contour1, pixel_spacing)
        area2 = measurements.calculate_area(contour2, pixel_spacing)
        
        # Calculate metrics
        dice = self.calculate_dice_coefficient(mask1, mask2)
        hausdorff = self.calculate_hausdorff_distance(contour1, contour2, pixel_spacing)
        
        # Calculate Jaccard index
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        jaccard = float(intersection / union) if union > 0 else 0.0
        
        # Calculate sensitivity and specificity
        true_positive = intersection
        false_positive = (mask2 > 0).sum() - true_positive
        false_negative = (mask1 > 0).sum() - true_positive
        
        sensitivity = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0.0
        precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) > 0 else 0.0
        
        # Return metrics
        return {
            "dice_coefficient": dice,
            "hausdorff_distance_mm": hausdorff,
            "jaccard_index": jaccard,
            "sensitivity": sensitivity,
            "precision": precision,
            "area1_mm2": area1,
            "area2_mm2": area2,
            "area_difference_mm2": abs(area1 - area2),
            "area_difference_percent": 100.0 * abs(area1 - area2) / area1 if area1 > 0 else float('inf')
        }
    
    def compare_structures(self, contours1: Dict[int, np.ndarray], 
                       contours2: Dict[int, np.ndarray],
                       shape: Tuple[int, int, int],
                       slice_thickness: float = 1.0,
                       pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> Dict:
        """
        Compare two 3D structures and calculate various metrics.
        
        Parameters
        ----------
        contours1 : Dict[int, np.ndarray]
            First structure contours
        contours2 : Dict[int, np.ndarray]
            Second structure contours
        shape : Tuple[int, int, int]
            Shape of the volume
        slice_thickness : float, optional
            Thickness of each slice in mm
        pixel_spacing : Tuple[float, float], optional
            Pixel spacing in mm
            
        Returns
        -------
        Dict
            Dictionary of comparison metrics
        """
        # Create masks from contours
        measurements = ContourMeasurements()
        mask1 = measurements.create_structure_mask(contours1, shape)
        mask2 = measurements.create_structure_mask(contours2, shape)
        
        # Calculate volume overlap
        volume1 = measurements.calculate_volume(contours1, slice_thickness, pixel_spacing)
        volume2 = measurements.calculate_volume(contours2, slice_thickness, pixel_spacing)
        
        # Calculate metrics
        dice = self.calculate_dice_coefficient(mask1, mask2)
        
        # Calculate Jaccard index
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        jaccard = float(intersection / union) if union > 0 else 0.0
        
        # Calculate average Hausdorff distance
        common_slices = set(contours1.keys()).intersection(set(contours2.keys()))
        hausdorff_distances = []
        
        for slice_idx in common_slices:
            hausdorff = self.calculate_hausdorff_distance(
                contours1[slice_idx], contours2[slice_idx], pixel_spacing
            )
            hausdorff_distances.append(hausdorff)
        
        avg_hausdorff = np.mean(hausdorff_distances) if hausdorff_distances else float('inf')
        max_hausdorff = np.max(hausdorff_distances) if hausdorff_distances else float('inf')
        
        # Calculate sensitivity and specificity
        true_positive = intersection
        false_positive = (mask2 > 0).sum() - true_positive
        false_negative = (mask1 > 0).sum() - true_positive
        
        sensitivity = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0.0
        precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) > 0 else 0.0
        
        # Return metrics
        return {
            "dice_coefficient": dice,
            "jaccard_index": jaccard,
            "avg_hausdorff_distance_mm": avg_hausdorff,
            "max_hausdorff_distance_mm": max_hausdorff,
            "sensitivity": sensitivity,
            "precision": precision,
            "volume1_mm3": volume1,
            "volume1_cc": volume1 / 1000.0,
            "volume2_mm3": volume2,
            "volume2_cc": volume2 / 1000.0,
            "volume_difference_mm3": abs(volume1 - volume2),
            "volume_difference_cc": abs(volume1 - volume2) / 1000.0,
            "volume_difference_percent": 100.0 * abs(volume1 - volume2) / volume1 if volume1 > 0 else float('inf'),
            "common_slices": len(common_slices),
            "slices1": len(contours1),
            "slices2": len(contours2)
        }
