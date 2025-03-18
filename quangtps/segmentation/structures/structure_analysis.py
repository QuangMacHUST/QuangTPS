#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure Analysis Module for QuangTPS.

This module provides functionality for analyzing anatomical structures and tumors
in radiotherapy treatment planning, including volume calculation, shape analysis,
and dosimetric statistics.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from scipy import ndimage
import SimpleITK as sitk

from quangtps.core.exceptions import ValidationError
from quangtps.segmentation.structures.structure_set import StructureSet
from quangtps.segmentation.structures.structure import Structure

logger = logging.getLogger(__name__)

class StructureAnalyzer:
    """
    Class for analyzing segmented structures.
    
    This class provides methods to calculate various metrics and statistics 
    for anatomical structures and tumors, such as volume, center of mass,
    dimensions, shape descriptors, and histogram analysis.
    """
    
    def __init__(self):
        """Initialize structure analyzer."""
        pass
    
    def calculate_volume(self, structure: Structure, spacing: Tuple[float, float, float] = None) -> float:
        """
        Calculate the volume of a structure.
        
        Parameters
        ----------
        structure : Structure
            The structure to analyze
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm (x, y, z)
            
        Returns
        -------
        float
            Volume in cubic centimeters (cc)
        """
        try:
            # Get binary mask
            mask = structure.get_mask()
            
            if mask is None or np.sum(mask) == 0:
                return 0.0
            
            # Use structure's spacing if not provided
            if spacing is None:
                spacing = structure.spacing if hasattr(structure, 'spacing') else (1.0, 1.0, 1.0)
            
            # Calculate volume (convert from mm³ to cm³)
            voxel_volume = spacing[0] * spacing[1] * spacing[2] / 1000.0  # mm³ to cm³
            volume_cc = np.sum(mask) * voxel_volume
            
            return volume_cc
            
        except Exception as e:
            logger.error(f"Error calculating structure volume: {str(e)}")
            raise ValidationError(f"Error calculating structure volume: {str(e)}")
    
    def calculate_center_of_mass(self, structure: Structure) -> Tuple[float, float, float]:
        """
        Calculate the center of mass of a structure.
        
        Parameters
        ----------
        structure : Structure
            The structure to analyze
            
        Returns
        -------
        Tuple[float, float, float]
            Center of mass coordinates (x, y, z)
        """
        try:
            # Get binary mask
            mask = structure.get_mask()
            
            if mask is None or np.sum(mask) == 0:
                return (0.0, 0.0, 0.0)
            
            # Calculate center of mass
            center_of_mass = ndimage.center_of_mass(mask)
            
            # Convert to physical coordinates if spacing is available
            if hasattr(structure, 'spacing') and hasattr(structure, 'origin'):
                spacing = structure.spacing
                origin = structure.origin
                com_physical = (
                    origin[0] + center_of_mass[0] * spacing[0],
                    origin[1] + center_of_mass[1] * spacing[1],
                    origin[2] + center_of_mass[2] * spacing[2]
                )
                return com_physical
            
            return center_of_mass
            
        except Exception as e:
            logger.error(f"Error calculating center of mass: {str(e)}")
            raise ValidationError(f"Error calculating center of mass: {str(e)}")
    
    def calculate_dimensions(self, structure: Structure, spacing: Tuple[float, float, float] = None) -> Dict[str, float]:
        """
        Calculate the dimensions (width, height, depth) of a structure.
        
        Parameters
        ----------
        structure : Structure
            The structure to analyze
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm (x, y, z)
            
        Returns
        -------
        Dict[str, float]
            Dictionary with width, height, and depth in mm
        """
        try:
            # Get binary mask
            mask = structure.get_mask()
            
            if mask is None or np.sum(mask) == 0:
                return {"width": 0.0, "height": 0.0, "depth": 0.0}
            
            # Use structure's spacing if not provided
            if spacing is None:
                spacing = structure.spacing if hasattr(structure, 'spacing') else (1.0, 1.0, 1.0)
            
            # Find non-zero voxel indices
            indices = np.where(mask > 0)
            
            # Calculate min and max along each axis
            min_x, max_x = np.min(indices[0]), np.max(indices[0])
            min_y, max_y = np.min(indices[1]), np.max(indices[1])
            min_z, max_z = np.min(indices[2]), np.max(indices[2])
            
            # Calculate dimensions in mm
            width = (max_x - min_x + 1) * spacing[0]
            height = (max_y - min_y + 1) * spacing[1]
            depth = (max_z - min_z + 1) * spacing[2]
            
            return {
                "width": width,
                "height": height,
                "depth": depth
            }
            
        except Exception as e:
            logger.error(f"Error calculating structure dimensions: {str(e)}")
            raise ValidationError(f"Error calculating structure dimensions: {str(e)}")
    
    def calculate_surface_area(self, structure: Structure, spacing: Tuple[float, float, float] = None) -> float:
        """
        Calculate the surface area of a structure.
        
        Parameters
        ----------
        structure : Structure
            The structure to analyze
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm (x, y, z)
            
        Returns
        -------
        float
            Surface area in square centimeters (cm²)
        """
        try:
            # Get binary mask
            mask = structure.get_mask()
            
            if mask is None or np.sum(mask) == 0:
                return 0.0
            
            # Use structure's spacing if not provided
            if spacing is None:
                spacing = structure.spacing if hasattr(structure, 'spacing') else (1.0, 1.0, 1.0)
            
            # Convert to SimpleITK image for surface area calculation
            sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
            sitk_mask.SetSpacing(spacing)
            
            # Calculate surface area using SimpleITK
            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            label_shape_filter.Execute(sitk_mask)
            
            # Convert from mm² to cm²
            surface_area_cm2 = label_shape_filter.GetPerimeter(1) / 100.0
            
            return surface_area_cm2
            
        except Exception as e:
            logger.error(f"Error calculating surface area: {str(e)}")
            raise ValidationError(f"Error calculating surface area: {str(e)}")
    
    def calculate_mean_density(self, structure: Structure, image_data: np.ndarray) -> float:
        """
        Calculate the mean density (HU value) within a structure.
        
        Parameters
        ----------
        structure : Structure
            The structure to analyze
        image_data : np.ndarray
            CT or other image data in Hounsfield Units or other intensity values
            
        Returns
        -------
        float
            Mean density value
        """
        try:
            # Get binary mask
            mask = structure.get_mask()
            
            if mask is None or np.sum(mask) == 0:
                return 0.0
            
            # Check image data dimensions match mask
            if mask.shape != image_data.shape:
                raise ValidationError(f"Mask shape {mask.shape} does not match image data shape {image_data.shape}")
            
            # Calculate mean density in the masked region
            masked_image = image_data * mask
            mean_density = np.sum(masked_image) / np.sum(mask)
            
            return mean_density
            
        except Exception as e:
            logger.error(f"Error calculating mean density: {str(e)}")
            raise ValidationError(f"Error calculating mean density: {str(e)}")
    
    def calculate_histogram(self, structure: Structure, image_data: np.ndarray, 
                          bins: int = 100, range_min: float = -1000, range_max: float = 3000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the histogram of pixel values within a structure.
        
        Parameters
        ----------
        structure : Structure
            The structure to analyze
        image_data : np.ndarray
            CT or other image data
        bins : int, optional
            Number of histogram bins
        range_min : float, optional
            Minimum value for histogram range
        range_max : float, optional
            Maximum value for histogram range
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Histogram values and bin edges
        """
        try:
            # Get binary mask
            mask = structure.get_mask()
            
            if mask is None or np.sum(mask) == 0:
                return np.zeros(bins), np.linspace(range_min, range_max, bins + 1)
            
            # Check image data dimensions match mask
            if mask.shape != image_data.shape:
                raise ValidationError(f"Mask shape {mask.shape} does not match image data shape {image_data.shape}")
            
            # Extract values within the mask
            values = image_data[mask > 0]
            
            # Calculate histogram
            hist, bin_edges = np.histogram(values, bins=bins, range=(range_min, range_max))
            
            return hist, bin_edges
            
        except Exception as e:
            logger.error(f"Error calculating histogram: {str(e)}")
            raise ValidationError(f"Error calculating histogram: {str(e)}")
    
    def calculate_shape_metrics(self, structure: Structure, spacing: Tuple[float, float, float] = None) -> Dict[str, float]:
        """
        Calculate shape metrics for a structure (sphericity, compactness, etc.).
        
        Parameters
        ----------
        structure : Structure
            The structure to analyze
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm (x, y, z)
            
        Returns
        -------
        Dict[str, float]
            Dictionary of shape metrics
        """
        try:
            # Get binary mask
            mask = structure.get_mask()
            
            if mask is None or np.sum(mask) == 0:
                return {
                    "sphericity": 0.0,
                    "compactness": 0.0,
                    "elongation": 0.0
                }
            
            # Use structure's spacing if not provided
            if spacing is None:
                spacing = structure.spacing if hasattr(structure, 'spacing') else (1.0, 1.0, 1.0)
            
            # Convert to SimpleITK image
            sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
            sitk_mask.SetSpacing(spacing)
            
            # Calculate shape metrics using SimpleITK
            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            label_shape_filter.Execute(sitk_mask)
            
            # Volume in mm³
            volume_mm3 = label_shape_filter.GetPhysicalSize(1)
            
            # Surface area in mm²
            surface_area_mm2 = label_shape_filter.GetPerimeter(1)
            
            # Calculate sphericity: ratio of the surface area of a sphere with the same volume to the actual surface area
            # Sphericity = 1 for a perfect sphere, < 1 for all other shapes
            if surface_area_mm2 > 0:
                sphere_radius = ((3 * volume_mm3) / (4 * np.pi)) ** (1/3)
                sphere_surface = 4 * np.pi * (sphere_radius ** 2)
                sphericity = sphere_surface / surface_area_mm2
            else:
                sphericity = 0.0
            
            # Calculate compactness: ratio of volume to surface area (normalized)
            if surface_area_mm2 > 0:
                compactness = (volume_mm3 ** (2/3)) / surface_area_mm2
            else:
                compactness = 0.0
            
            # Calculate elongation using principal components analysis
            # Get principal axes
            principal_axes = label_shape_filter.GetPrincipalAxes(1)
            
            # Convert to numpy array for easier manipulation
            axes_lengths = np.array([
                np.linalg.norm(principal_axes[0:3]),
                np.linalg.norm(principal_axes[3:6]),
                np.linalg.norm(principal_axes[6:9])
            ])
            
            # Sort axes lengths
            axes_lengths = np.sort(axes_lengths)
            
            # Calculate elongation (ratio of longest to shortest axis)
            if axes_lengths[0] > 0:
                elongation = axes_lengths[2] / axes_lengths[0]
            else:
                elongation = 0.0
            
            return {
                "sphericity": sphericity,
                "compactness": compactness,
                "elongation": elongation
            }
            
        except Exception as e:
            logger.error(f"Error calculating shape metrics: {str(e)}")
            raise ValidationError(f"Error calculating shape metrics: {str(e)}")

class DoseVolumeAnalyzer:
    """
    Class for analyzing dose-volume relationships for structures.
    
    This class provides methods to calculate dose-volume histograms (DVHs) and
    related metrics for evaluating radiation treatment plans.
    """
    
    def __init__(self):
        """Initialize dose-volume analyzer."""
        pass
    
    def calculate_dvh(self, structure: Structure, dose_data: np.ndarray, 
                     max_dose: float = None, bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Dose-Volume Histogram (DVH) for a structure.
        
        Parameters
        ----------
        structure : Structure
            The structure to analyze
        dose_data : np.ndarray
            3D dose distribution data
        max_dose : float, optional
            Maximum dose value for binning
        bins : int, optional
            Number of histogram bins
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Dose bins and corresponding volume percentages
        """
        try:
            # Get binary mask
            mask = structure.get_mask()
            
            if mask is None or np.sum(mask) == 0:
                return np.zeros(bins), np.zeros(bins)
            
            # Check dose data dimensions match mask
            if mask.shape != dose_data.shape:
                raise ValidationError(f"Mask shape {mask.shape} does not match dose data shape {dose_data.shape}")
            
            # Extract dose values within the structure
            structure_dose = dose_data[mask > 0]
            
            # Determine max dose if not provided
            if max_dose is None:
                max_dose = np.max(structure_dose) * 1.1  # Add 10% margin
            
            # Create dose bins
            dose_bins = np.linspace(0, max_dose, bins + 1)
            bin_centers = (dose_bins[:-1] + dose_bins[1:]) / 2
            
            # Calculate histogram
            hist, _ = np.histogram(structure_dose, bins=dose_bins)
            
            # Convert to cumulative histogram (fraction of volume receiving at least dose D)
            total_voxels = len(structure_dose)
            cumulative_volume = np.cumsum(hist[::-1])[::-1] / total_voxels * 100  # As percentage
            
            return bin_centers, cumulative_volume
            
        except Exception as e:
            logger.error(f"Error calculating DVH: {str(e)}")
            raise ValidationError(f"Error calculating DVH: {str(e)}")
    
    def calculate_dose_metrics(self, structure: Structure, dose_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate dose metrics for a structure.
        
        Parameters
        ----------
        structure : Structure
            The structure to analyze
        dose_data : np.ndarray
            3D dose distribution data
            
        Returns
        -------
        Dict[str, float]
            Dictionary of dose metrics including min, max, mean, median doses
        """
        try:
            # Get binary mask
            mask = structure.get_mask()
            
            if mask is None or np.sum(mask) == 0:
                return {
                    "min_dose": 0.0,
                    "max_dose": 0.0,
                    "mean_dose": 0.0,
                    "median_dose": 0.0
                }
            
            # Check dose data dimensions match mask
            if mask.shape != dose_data.shape:
                raise ValidationError(f"Mask shape {mask.shape} does not match dose data shape {dose_data.shape}")
            
            # Extract dose values within the structure
            structure_dose = dose_data[mask > 0]
            
            # Calculate dose metrics
            min_dose = np.min(structure_dose)
            max_dose = np.max(structure_dose)
            mean_dose = np.mean(structure_dose)
            median_dose = np.median(structure_dose)
            
            return {
                "min_dose": min_dose,
                "max_dose": max_dose,
                "mean_dose": mean_dose,
                "median_dose": median_dose
            }
            
        except Exception as e:
            logger.error(f"Error calculating dose metrics: {str(e)}")
            raise ValidationError(f"Error calculating dose metrics: {str(e)}")
    
    def calculate_dose_volume_metrics(self, structure: Structure, dose_data: np.ndarray, 
                                    dose_thresholds: List[float] = None) -> Dict[str, float]:
        """
        Calculate dose-volume metrics (e.g., V20Gy, D95) for a structure.
        
        Parameters
        ----------
        structure : Structure
            The structure to analyze
        dose_data : np.ndarray
            3D dose distribution data
        dose_thresholds : List[float], optional
            List of dose thresholds in Gy for V metrics
            
        Returns
        -------
        Dict[str, float]
            Dictionary of dose-volume metrics
        """
        try:
            # Get binary mask
            mask = structure.get_mask()
            
            if mask is None or np.sum(mask) == 0:
                return {}
            
            # Default dose thresholds if not provided
            if dose_thresholds is None:
                dose_thresholds = [5, 10, 20, 30, 40, 50, 60, 70, 80]
                
            # Check dose data dimensions match mask
            if mask.shape != dose_data.shape:
                raise ValidationError(f"Mask shape {mask.shape} does not match dose data shape {dose_data.shape}")
            
            # Extract dose values within the structure
            structure_dose = dose_data[mask > 0]
            total_voxels = len(structure_dose)
            
            # Calculate V metrics (percentage of volume receiving at least X Gy)
            v_metrics = {}
            for threshold in dose_thresholds:
                volume_percentage = np.sum(structure_dose >= threshold) / total_voxels * 100
                v_metrics[f"V{threshold}Gy"] = volume_percentage
            
            # Calculate D metrics (minimum dose to the hottest X% of the volume)
            d_metrics = {}
            percentiles = [95, 90, 50, 5, 2]
            sorted_doses = np.sort(structure_dose)
            for percentile in percentiles:
                index = int(np.ceil((100 - percentile) / 100 * total_voxels)) - 1
                if index < 0:
                    index = 0
                d_metrics[f"D{percentile}"] = sorted_doses[index]
            
            # Combine metrics
            metrics = {**v_metrics, **d_metrics}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating dose-volume metrics: {str(e)}")
            raise ValidationError(f"Error calculating dose-volume metrics: {str(e)}")
    
    def calculate_conformity_index(self, target_structure: Structure, dose_data: np.ndarray, 
                                 prescription_dose: float) -> float:
        """
        Calculate the Conformity Index (CI) for a target structure.
        
        CI = (Volume receiving prescription dose) / (Target volume)
        
        Parameters
        ----------
        target_structure : Structure
            The target structure (e.g., PTV)
        dose_data : np.ndarray
            3D dose distribution data
        prescription_dose : float
            Prescription dose in Gy
            
        Returns
        -------
        float
            Conformity Index
        """
        try:
            # Get target binary mask
            target_mask = target_structure.get_mask()
            
            if target_mask is None or np.sum(target_mask) == 0:
                return 0.0
            
            # Check dose data dimensions match mask
            if target_mask.shape != dose_data.shape:
                raise ValidationError(f"Mask shape {target_mask.shape} does not match dose data shape {dose_data.shape}")
            
            # Calculate volumes
            target_volume = np.sum(target_mask)
            prescription_volume = np.sum(dose_data >= prescription_dose)
            
            # Calculate CI
            if target_volume > 0:
                ci = prescription_volume / target_volume
            else:
                ci = 0.0
            
            return ci
            
        except Exception as e:
            logger.error(f"Error calculating conformity index: {str(e)}")
            raise ValidationError(f"Error calculating conformity index: {str(e)}")
    
    def calculate_homogeneity_index(self, target_structure: Structure, dose_data: np.ndarray, 
                                  prescription_dose: float) -> float:
        """
        Calculate the Homogeneity Index (HI) for a target structure.
        
        HI = (D2% - D98%) / D50%
        
        Parameters
        ----------
        target_structure : Structure
            The target structure (e.g., PTV)
        dose_data : np.ndarray
            3D dose distribution data
        prescription_dose : float
            Prescription dose in Gy
            
        Returns
        -------
        float
            Homogeneity Index
        """
        try:
            # Get target binary mask
            target_mask = target_structure.get_mask()
            
            if target_mask is None or np.sum(target_mask) == 0:
                return 0.0
            
            # Check dose data dimensions match mask
            if target_mask.shape != dose_data.shape:
                raise ValidationError(f"Mask shape {target_mask.shape} does not match dose data shape {dose_data.shape}")
            
            # Extract dose values within the target
            target_dose = dose_data[target_mask > 0]
            total_voxels = len(target_dose)
            
            if total_voxels == 0:
                return 0.0
            
            # Calculate D2%, D98%, and D50%
            sorted_doses = np.sort(target_dose)
            d2 = sorted_doses[int(np.ceil(0.98 * total_voxels)) - 1]
            d98 = sorted_doses[int(np.ceil(0.02 * total_voxels)) - 1]
            d50 = sorted_doses[int(np.ceil(0.5 * total_voxels)) - 1]
            
            # Calculate HI
            if d50 > 0:
                hi = (d2 - d98) / d50
            else:
                hi = 0.0
            
            return hi
            
        except Exception as e:
            logger.error(f"Error calculating homogeneity index: {str(e)}")
            raise ValidationError(f"Error calculating homogeneity index: {str(e)}")
    
    def calculate_gradient_index(self, target_structure: Structure, dose_data: np.ndarray, 
                               prescription_dose: float, reference_dose_fraction: float = 0.5) -> float:
        """
        Calculate the Gradient Index (GI) for a target structure.
        
        GI = (Volume receiving ref_dose) / (Volume receiving prescription_dose)
        
        Parameters
        ----------
        target_structure : Structure
            The target structure (e.g., PTV)
        dose_data : np.ndarray
            3D dose distribution data
        prescription_dose : float
            Prescription dose in Gy
        reference_dose_fraction : float, optional
            Fraction of prescription dose for reference isodose
            
        Returns
        -------
        float
            Gradient Index
        """
        try:
            # Calculate reference dose
            reference_dose = prescription_dose * reference_dose_fraction
            
            # Calculate volumes
            prescription_volume = np.sum(dose_data >= prescription_dose)
            reference_volume = np.sum(dose_data >= reference_dose)
            
            # Calculate GI
            if prescription_volume > 0:
                gi = reference_volume / prescription_volume
            else:
                gi = 0.0
            
            return gi
            
        except Exception as e:
            logger.error(f"Error calculating gradient index: {str(e)}")
            raise ValidationError(f"Error calculating gradient index: {str(e)}")
    
    def plot_dvh(self, structures: List[Structure], dose_data: np.ndarray, 
               max_dose: float = None, bins: int = 100, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot Dose-Volume Histograms for multiple structures.
        
        Parameters
        ----------
        structures : List[Structure]
            List of structures to include in the DVH
        dose_data : np.ndarray
            3D dose distribution data
        max_dose : float, optional
            Maximum dose for x-axis
        bins : int, optional
            Number of histogram bins
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns
        -------
        plt.Figure
            Matplotlib figure with DVH plot
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Colors for different structures
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            # Calculate and plot DVH for each structure
            for i, structure in enumerate(structures):
                # Get structure name
                name = structure.name if hasattr(structure, 'name') else f"Structure {i+1}"
                
                # Calculate DVH
                dose_bins, volume_percent = self.calculate_dvh(structure, dose_data, max_dose, bins)
                
                # Plot DVH
                color = colors[i % len(colors)]
                ax.plot(dose_bins, volume_percent, label=name, color=color, linewidth=2)
            
            # Set plot labels and title
            ax.set_xlabel('Dose (Gy)', fontsize=12)
            ax.set_ylabel('Volume (%)', fontsize=12)
            ax.set_title('Dose-Volume Histogram (DVH)', fontsize=14)
            
            # Set y-axis limits
            ax.set_ylim(0, 105)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            ax.legend(loc='upper right', fontsize=10)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting DVH: {str(e)}")
            raise ValidationError(f"Error plotting DVH: {str(e)}")

class TreatmentPlanEvaluator:
    """
    Class for evaluating radiation treatment plans based on structure analysis.
    
    This class combines structure and dose-volume analysis to evaluate 
    treatment plans against clinical goals and constraints.
    """
    
    def __init__(self):
        """Initialize treatment plan evaluator."""
        self.structure_analyzer = StructureAnalyzer()
        self.dvh_analyzer = DoseVolumeAnalyzer()
    
    def evaluate_plan(self, structure_set: StructureSet, dose_data: np.ndarray, 
                      clinical_goals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a treatment plan against clinical goals.
        
        Parameters
        ----------
        structure_set : StructureSet
            Set of structures in the plan
        dose_data : np.ndarray
            3D dose distribution data
        clinical_goals : Dict[str, Dict[str, Any]]
            Dictionary of clinical goals for each structure
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Evaluation results for each structure and goal
        """
        try:
            results = {}
            
            # Evaluate each structure
            for structure_id, goals in clinical_goals.items():
                # Get structure
                structure = structure_set.get_structure(structure_id)
                
                if structure is None:
                    logger.warning(f"Structure '{structure_id}' not found in structure set")
                    continue
                
                structure_results = {}
                
                # Evaluate each goal for the structure
                for goal_name, goal_params in goals.items():
                    goal_type = goal_params.get('type')
                    
                    if goal_type == 'volume_at_dose':
                        # Volume receiving at least specified dose (V20Gy < 30%)
                        dose_threshold = goal_params.get('dose_threshold')
                        volume_limit = goal_params.get('limit')
                        comparator = goal_params.get('comparator', '<')
                        
                        # Calculate actual value
                        metrics = self.dvh_analyzer.calculate_dose_volume_metrics(
                            structure, dose_data, [dose_threshold])
                        actual_value = metrics.get(f"V{dose_threshold}Gy", 0.0)
                        
                        # Check if goal is met
                        if comparator == '<':
                            is_met = actual_value < volume_limit
                        elif comparator == '<=':
                            is_met = actual_value <= volume_limit
                        elif comparator == '>':
                            is_met = actual_value > volume_limit
                        elif comparator == '>=':
                            is_met = actual_value >= volume_limit
                        else:
                            is_met = False
                        
                        structure_results[goal_name] = {
                            'goal': f"V{dose_threshold}Gy {comparator} {volume_limit}%",
                            'actual': actual_value,
                            'is_met': is_met
                        }
                        
                    elif goal_type == 'dose_at_volume':
                        # Dose to specified volume (D95% > 95% of prescription)
                        volume_point = goal_params.get('volume_point')
                        dose_limit = goal_params.get('limit')
                        comparator = goal_params.get('comparator', '>')
                        
                        # Calculate actual value
                        metrics = self.dvh_analyzer.calculate_dose_volume_metrics(structure, dose_data)
                        actual_value = metrics.get(f"D{volume_point}", 0.0)
                        
                        # Check if goal is met
                        if comparator == '<':
                            is_met = actual_value < dose_limit
                        elif comparator == '<=':
                            is_met = actual_value <= dose_limit
                        elif comparator == '>':
                            is_met = actual_value > dose_limit
                        elif comparator == '>=':
                            is_met = actual_value >= dose_limit
                        else:
                            is_met = False
                        
                        structure_results[goal_name] = {
                            'goal': f"D{volume_point}% {comparator} {dose_limit} Gy",
                            'actual': actual_value,
                            'is_met': is_met
                        }
                        
                    elif goal_type == 'max_dose':
                        # Maximum dose (Dmax < 107% of prescription)
                        dose_limit = goal_params.get('limit')
                        comparator = goal_params.get('comparator', '<')
                        
                        # Calculate actual value
                        metrics = self.dvh_analyzer.calculate_dose_metrics(structure, dose_data)
                        actual_value = metrics.get('max_dose', 0.0)
                        
                        # Check if goal is met
                        if comparator == '<':
                            is_met = actual_value < dose_limit
                        elif comparator == '<=':
                            is_met = actual_value <= dose_limit
                        elif comparator == '>':
                            is_met = actual_value > dose_limit
                        elif comparator == '>=':
                            is_met = actual_value >= dose_limit
                        else:
                            is_met = False
                        
                        structure_results[goal_name] = {
                            'goal': f"Max dose {comparator} {dose_limit} Gy",
                            'actual': actual_value,
                            'is_met': is_met
                        }
                        
                    elif goal_type == 'mean_dose':
                        # Mean dose (Dmean < 26Gy)
                        dose_limit = goal_params.get('limit')
                        comparator = goal_params.get('comparator', '<')
                        
                        # Calculate actual value
                        metrics = self.dvh_analyzer.calculate_dose_metrics(structure, dose_data)
                        actual_value = metrics.get('mean_dose', 0.0)
                        
                        # Check if goal is met
                        if comparator == '<':
                            is_met = actual_value < dose_limit
                        elif comparator == '<=':
                            is_met = actual_value <= dose_limit
                        elif comparator == '>':
                            is_met = actual_value > dose_limit
                        elif comparator == '>=':
                            is_met = actual_value >= dose_limit
                        else:
                            is_met = False
                        
                        structure_results[goal_name] = {
                            'goal': f"Mean dose {comparator} {dose_limit} Gy",
                            'actual': actual_value,
                            'is_met': is_met
                        }
                
                # Add results for this structure
                results[structure_id] = structure_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating treatment plan: {str(e)}")
            raise ValidationError(f"Error evaluating treatment plan: {str(e)}")
    
    def generate_plan_evaluation_report(self, structure_set: StructureSet, dose_data: np.ndarray, 
                                      clinical_goals: Dict[str, Dict[str, Any]], 
                                      prescription_dose: float) -> Dict[str, Any]:
        """
        Generate a comprehensive treatment plan evaluation report.
        
        Parameters
        ----------
        structure_set : StructureSet
            Set of structures in the plan
        dose_data : np.ndarray
            3D dose distribution data
        clinical_goals : Dict[str, Dict[str, Any]]
            Dictionary of clinical goals for each structure
        prescription_dose : float
            Prescription dose in Gy
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive evaluation report
        """
        try:
            report = {
                'prescription_dose': prescription_dose,
                'structures': {},
                'targets': {},
                'oars': {},
                'metrics': {},
                'goals_evaluation': {},
                'overall_assessment': {}
            }
            
            # Identify targets and OARs
            targets = []
            oars = []
            
            for structure_id in clinical_goals.keys():
                structure = structure_set.get_structure(structure_id)
                
                if structure is None:
                    continue
                    
                structure_type = getattr(structure, 'type', '').lower()
                
                if structure_type in ['ptv', 'ctv', 'gtv', 'target']:
                    targets.append(structure)
                else:
                    oars.append(structure)
            
            # Calculate basic metrics for all structures
            for structure in targets + oars:
                structure_id = structure.id
                structure_name = getattr(structure, 'name', structure_id)
                
                # Calculate structure metrics
                volume = self.structure_analyzer.calculate_volume(structure)
                dose_metrics = self.dvh_analyzer.calculate_dose_metrics(structure, dose_data)
                
                report['structures'][structure_id] = {
                    'name': structure_name,
                    'volume_cc': volume,
                    'min_dose': dose_metrics['min_dose'],
                    'max_dose': dose_metrics['max_dose'],
                    'mean_dose': dose_metrics['mean_dose'],
                    'median_dose': dose_metrics['median_dose']
                }
            
            # Calculate target-specific metrics
            for target in targets:
                target_id = target.id
                
                # Calculate conformity and homogeneity indices
                ci = self.dvh_analyzer.calculate_conformity_index(target, dose_data, prescription_dose)
                hi = self.dvh_analyzer.calculate_homogeneity_index(target, dose_data, prescription_dose)
                gi = self.dvh_analyzer.calculate_gradient_index(target, dose_data, prescription_dose)
                
                report['targets'][target_id] = {
                    'conformity_index': ci,
                    'homogeneity_index': hi,
                    'gradient_index': gi
                }
            
            # Calculate OAR-specific metrics
            for oar in oars:
                oar_id = oar.id
                
                # Get dose-volume metrics
                dv_metrics = self.dvh_analyzer.calculate_dose_volume_metrics(oar, dose_data)
                
                report['oars'][oar_id] = dv_metrics
            
            # Evaluate clinical goals
            report['goals_evaluation'] = self.evaluate_plan(structure_set, dose_data, clinical_goals)
            
            # Calculate overall plan metrics
            if targets:
                # Use first target as primary target
                primary_target = targets[0]
                
                # Calculate overall plan quality metrics
                report['metrics'] = {
                    'conformity_index': self.dvh_analyzer.calculate_conformity_index(
                        primary_target, dose_data, prescription_dose),
                    'homogeneity_index': self.dvh_analyzer.calculate_homogeneity_index(
                        primary_target, dose_data, prescription_dose),
                    'gradient_index': self.dvh_analyzer.calculate_gradient_index(
                        primary_target, dose_data, prescription_dose)
                }
            
            # Overall assessment
            goals_met = 0
            total_goals = 0
            
            for structure_results in report['goals_evaluation'].values():
                for goal_result in structure_results.values():
                    total_goals += 1
                    if goal_result.get('is_met', False):
                        goals_met += 1
            
            if total_goals > 0:
                goals_met_percentage = (goals_met / total_goals) * 100
                
                if goals_met_percentage == 100:
                    quality_assessment = "Excellent - All clinical goals met"
                elif goals_met_percentage >= 90:
                    quality_assessment = "Very Good - Most clinical goals met"
                elif goals_met_percentage >= 75:
                    quality_assessment = "Good - Majority of clinical goals met"
                elif goals_met_percentage >= 50:
                    quality_assessment = "Fair - Some clinical goals met"
                else:
                    quality_assessment = "Poor - Few clinical goals met"
            else:
                goals_met_percentage = 0
                quality_assessment = "N/A - No clinical goals defined"
            
            report['overall_assessment'] = {
                'goals_met_percentage': goals_met_percentage,
                'quality_assessment': quality_assessment
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating plan evaluation report: {str(e)}")
            raise ValidationError(f"Error generating plan evaluation report: {str(e)}")
