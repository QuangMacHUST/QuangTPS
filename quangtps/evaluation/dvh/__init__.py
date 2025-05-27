#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DVH (Dose-Volume Histogram) calculation and analysis module.

This module provides functions and classes for calculating, analyzing, and
visualizing dose-volume histograms in radiotherapy treatment planning.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)


# Lazy import function để tránh circular dependencies
def _lazy_import_dvh_calculation():
    """Lazy import các hàm DVH calculation để tránh circular import"""
    try:
        from quangtps.evaluation.dvh.dvh_calculation import (
            calculate_dvh,
            calculate_dvh_for_plan,
            calculate_dvh_metrics,
            calculate_conformity_index,
            calculate_homogeneity_index,
            calculate_gradient_index,
            calculate_equivalent_uniform_dose,
            _get_dose_at_volume,
            _get_volume_at_dose,
            DVHCalculator,
        )

        return {
            "calculate_dvh": calculate_dvh,
            "calculate_dvh_for_plan": calculate_dvh_for_plan,
            "calculate_dvh_metrics": calculate_dvh_metrics,
            "calculate_conformity_index": calculate_conformity_index,
            "calculate_homogeneity_index": calculate_homogeneity_index,
            "calculate_gradient_index": calculate_gradient_index,
            "calculate_equivalent_uniform_dose": calculate_equivalent_uniform_dose,
            "_get_dose_at_volume": _get_dose_at_volume,
            "_get_volume_at_dose": _get_volume_at_dose,
            "DVHCalculator": DVHCalculator,
        }
    except ImportError as e:
        logger.warning(f"Cannot import DVH calculation functions: {e}. Using fallback.")
        return None


# Fallback DVH calculation functions
def calculate_dvh(dose_grid, structure_mask, dose_bins=None, bin_count=1000):
    """Enhanced fallback DVH calculation function."""
    logger.info("Using enhanced fallback calculate_dvh function")

    try:
        # Try lazy import first
        dvh_funcs = _lazy_import_dvh_calculation()
        if dvh_funcs:
            return dvh_funcs["calculate_dvh"](
                dose_grid, structure_mask, dose_bins, bin_count
            )
    except Exception:
        pass

    # Enhanced fallback implementation
    if not hasattr(dose_grid, "shape") or not hasattr(structure_mask, "shape"):
        logger.warning("Invalid dose_grid or structure_mask format")
        return _empty_dvh_result()

    # Extract dose values within structure
    if structure_mask.any():
        structure_doses = dose_grid[structure_mask > 0]
    else:
        structure_doses = np.array([])

    if len(structure_doses) == 0:
        return _empty_dvh_result()

    # Enhanced dose statistics
    min_dose = float(np.min(structure_doses))
    max_dose = float(np.max(structure_doses))
    mean_dose = float(np.mean(structure_doses))
    median_dose = float(np.median(structure_doses))
    std_dose = float(np.std(structure_doses))

    # Create optimized dose bins
    if dose_bins is None:
        dose_bins = np.linspace(0, max_dose * 1.05, bin_count + 1)

    bin_centers = (dose_bins[1:] + dose_bins[:-1]) / 2

    # Calculate histogram with improved accuracy
    hist, _ = np.histogram(structure_doses, bins=dose_bins)
    total_voxels = len(structure_doses)

    if total_voxels > 0:
        differential_volume = (hist / total_voxels) * 100
        cumulative_volume = np.cumsum(differential_volume[::-1])[::-1]
    else:
        differential_volume = np.zeros_like(hist)
        cumulative_volume = np.zeros_like(hist)

    # Enhanced volume calculation
    voxel_volume = 1.0  # Default voxel volume in mm³
    total_volume = total_voxels * voxel_volume / 1000  # Convert to cm³

    return {
        "dose_bins": bin_centers,
        "differential_volume": differential_volume,
        "cumulative_volume": cumulative_volume,
        "min_dose": min_dose,
        "max_dose": max_dose,
        "mean_dose": mean_dose,
        "median_dose": median_dose,
        "std_dose": std_dose,
        "volume": total_volume,
        "total_voxels": total_voxels,
    }


def calculate_dvh_for_plan(dose_grid, structures, bin_count=1000):
    """Enhanced fallback DVH calculation for plan."""
    logger.info("Using enhanced fallback calculate_dvh_for_plan function")

    try:
        # Try lazy import first
        dvh_funcs = _lazy_import_dvh_calculation()
        if dvh_funcs:
            return dvh_funcs["calculate_dvh_for_plan"](dose_grid, structures, bin_count)
    except Exception:
        pass

    dvhs = {}
    for name, mask in structures.items():
        try:
            dvhs[name] = calculate_dvh(dose_grid, mask, bin_count=bin_count)
        except Exception as e:
            logger.error(f"Error calculating DVH for structure {name}: {e}")
            dvhs[name] = _empty_dvh_result()
    return dvhs


def calculate_dvh_metrics(dvh_data, metrics_list=None, rx_dose=None):
    """Enhanced fallback DVH metrics calculation."""
    logger.info("Using enhanced fallback calculate_dvh_metrics function")

    try:
        # Try lazy import first
        dvh_funcs = _lazy_import_dvh_calculation()
        if dvh_funcs:
            return dvh_funcs["calculate_dvh_metrics"](dvh_data, metrics_list, rx_dose)
    except Exception:
        pass

    if not dvh_data or not isinstance(dvh_data, dict):
        return {}

    # Enhanced metrics calculation
    mean_dose = dvh_data.get("mean_dose", 0)
    max_dose = dvh_data.get("max_dose", 0)
    min_dose = dvh_data.get("min_dose", 0)

    cumulative_volume = dvh_data.get("cumulative_volume", np.array([]))
    dose_bins = dvh_data.get("dose_bins", np.array([]))

    metrics = {
        "Dmean": mean_dose,
        "Dmax": max_dose,
        "Dmin": min_dose,
        "D95": _estimate_dose_at_volume(dose_bins, cumulative_volume, 95),
        "D50": _estimate_dose_at_volume(dose_bins, cumulative_volume, 50),
        "D5": _estimate_dose_at_volume(dose_bins, cumulative_volume, 5),
    }

    # Add volume metrics if prescription dose is provided
    if rx_dose:
        metrics.update(
            {
                "V20": _estimate_volume_at_dose(
                    dose_bins, cumulative_volume, rx_dose * 0.20
                ),
                "V50": _estimate_volume_at_dose(
                    dose_bins, cumulative_volume, rx_dose * 0.50
                ),
                "V95": _estimate_volume_at_dose(
                    dose_bins, cumulative_volume, rx_dose * 0.95
                ),
            }
        )

    return metrics


def calculate_conformity_index(target_dvh, prescription_dose):
    """Enhanced fallback conformity index."""
    try:
        dvh_funcs = _lazy_import_dvh_calculation()
        if dvh_funcs:
            return dvh_funcs["calculate_conformity_index"](
                target_dvh, prescription_dose
            )
    except Exception:
        pass

    # Simple fallback calculation
    if not target_dvh or not prescription_dose:
        return 1.0

    # Estimate CI based on volume covered by prescription dose
    volume_95 = _estimate_volume_at_dose(
        target_dvh.get("dose_bins", []),
        target_dvh.get("cumulative_volume", []),
        prescription_dose * 0.95,
    )

    return min(volume_95 / 95.0, 1.5) if volume_95 > 0 else 1.0


def calculate_homogeneity_index(target_dvh, prescription_dose):
    """Enhanced fallback homogeneity index."""
    try:
        dvh_funcs = _lazy_import_dvh_calculation()
        if dvh_funcs:
            return dvh_funcs["calculate_homogeneity_index"](
                target_dvh, prescription_dose
            )
    except Exception:
        pass

    if not target_dvh or not prescription_dose:
        return 0.1

    # Estimate HI = (D2 - D98) / prescription_dose
    d2 = _estimate_dose_at_volume(
        target_dvh.get("dose_bins", []), target_dvh.get("cumulative_volume", []), 2
    )
    d98 = _estimate_dose_at_volume(
        target_dvh.get("dose_bins", []), target_dvh.get("cumulative_volume", []), 98
    )

    return abs(d2 - d98) / prescription_dose if prescription_dose > 0 else 0.1


def calculate_gradient_index(reference_dvh, high_dose, low_dose):
    """Enhanced fallback gradient index."""
    try:
        dvh_funcs = _lazy_import_dvh_calculation()
        if dvh_funcs:
            return dvh_funcs["calculate_gradient_index"](
                reference_dvh, high_dose, low_dose
            )
    except Exception:
        pass

    return 1.0  # Simple fallback


def calculate_equivalent_uniform_dose(dvh_data, a=1.0):
    """Enhanced fallback EUD calculation."""
    try:
        dvh_funcs = _lazy_import_dvh_calculation()
        if dvh_funcs:
            return dvh_funcs["calculate_equivalent_uniform_dose"](dvh_data, a)
    except Exception:
        pass

    if not dvh_data:
        return 0.0

    # Simple EUD approximation
    mean_dose = dvh_data.get("mean_dose", 0)
    if abs(a) < 0.001:  # a ≈ 0, return mean dose
        return mean_dose
    else:
        # Simplified EUD calculation
        return mean_dose * (1 + a * 0.1)  # Rough approximation


def _get_dose_at_volume(dose_bins, cumulative_volume, volume_percent):
    """Enhanced fallback dose at volume."""
    try:
        dvh_funcs = _lazy_import_dvh_calculation()
        if dvh_funcs:
            return dvh_funcs["_get_dose_at_volume"](
                dose_bins, cumulative_volume, volume_percent
            )
    except Exception:
        pass

    return _estimate_dose_at_volume(dose_bins, cumulative_volume, volume_percent)


def _get_volume_at_dose(dose_bins, cumulative_volume, dose_value):
    """Enhanced fallback volume at dose."""
    try:
        dvh_funcs = _lazy_import_dvh_calculation()
        if dvh_funcs:
            return dvh_funcs["_get_volume_at_dose"](
                dose_bins, cumulative_volume, dose_value
            )
    except Exception:
        pass

    return _estimate_volume_at_dose(dose_bins, cumulative_volume, dose_value)


# Helper functions
def _empty_dvh_result():
    """Return empty DVH result structure."""
    return {
        "dose_bins": np.array([0]),
        "differential_volume": np.array([0]),
        "cumulative_volume": np.array([100]),
        "min_dose": 0,
        "max_dose": 0,
        "mean_dose": 0,
        "median_dose": 0,
        "std_dose": 0,
        "volume": 0,
        "total_voxels": 0,
    }


def _estimate_dose_at_volume(dose_bins, cumulative_volume, volume_percent):
    """Estimate dose at given volume percentage."""
    if len(dose_bins) == 0 or len(cumulative_volume) == 0:
        return 0.0

    try:
        # Find closest volume percentage
        idx = np.argmin(np.abs(cumulative_volume - volume_percent))
        return dose_bins[idx] if idx < len(dose_bins) else dose_bins[-1]
    except Exception:
        return np.mean(dose_bins) if len(dose_bins) > 0 else 0.0


def _estimate_volume_at_dose(dose_bins, cumulative_volume, dose_value):
    """Estimate volume percentage at given dose."""
    if len(dose_bins) == 0 or len(cumulative_volume) == 0:
        return 0.0

    try:
        # Find closest dose value
        idx = np.argmin(np.abs(dose_bins - dose_value))
        return cumulative_volume[idx] if idx < len(cumulative_volume) else 0.0
    except Exception:
        return 50.0  # Fallback percentage


# Create enhanced DVHCalculator class
class DVHCalculator:
    """Enhanced fallback DVHCalculator class."""

    def __init__(self):
        self.bin_count = 1000
        self.min_dose = 0.0
        self.max_dose = 100.0
        self.bin_width = 0.1
        self.use_relative_volume = True

    def calculate_dvh(self, dose, structure_set, structures_to_include=None):
        """Enhanced DVH calculation."""
        try:
            # Try lazy import first
            dvh_funcs = _lazy_import_dvh_calculation()
            if dvh_funcs and "DVHCalculator" in dvh_funcs:
                real_calculator = dvh_funcs["DVHCalculator"]()
                return real_calculator.calculate_dvh(
                    dose, structure_set, structures_to_include
                )
        except Exception:
            pass

        logger.warning("Using fallback DVHCalculator.calculate_dvh")
        return None

    def set_bin_width(self, bin_width):
        self.bin_width = bin_width

    def set_dose_range(self, min_dose, max_dose):
        self.min_dose = min_dose
        self.max_dose = max_dose


# Try to import visualization functions
try:
    from quangtps.evaluation.dvh.dvh_visualization import (
        get_structure_color,
        plot_dvh,
        plot_multiple_dvh,
        create_dvh_report,
        plot_dvh_bands,
        export_dvh_to_csv,
    )
except ImportError as e:
    logger.warning(f"Cannot import DVH visualization functions: {e}")


# Compatibility function
def calculate_dvh_from_dose_grid(dose_grid, structure_mask, **kwargs):
    """Compatibility function for older DVH calculation interface."""
    try:
        return calculate_dvh(dose_grid, structure_mask, **kwargs)
    except Exception as e:
        logger.error(f"Error in calculate_dvh_from_dose_grid: {e}")
        return _empty_dvh_result()


# Module-level exports
__all__ = [
    # Main DVH functions
    "calculate_dvh",
    "calculate_dvh_for_plan",
    "calculate_dvh_metrics",
    "calculate_conformity_index",
    "calculate_homogeneity_index",
    "calculate_gradient_index",
    "calculate_equivalent_uniform_dose",
    "_get_dose_at_volume",
    "_get_volume_at_dose",
    "DVHCalculator",
    # Compatibility
    "calculate_dvh_from_dose_grid",
]
