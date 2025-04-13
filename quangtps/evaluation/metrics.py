"""
Plan Evaluation Metrics

This module provides functions for calculating common dose metrics
used in radiotherapy treatment plan evaluation.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging

from quangtps.evaluation.dvh.dvh_data import DVHData
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

def calculate_d_metric(dvh_data: DVHData, volume_percent: float) -> float:
    """
    Calculate the dose (in Gy) received by a given volume percentage of the structure.
    
    Args:
        dvh_data: DVH data for the structure
        volume_percent: Volume percentage (0-100)
        
    Returns:
        Dose in Gy received by the specified volume percentage
    """
    if dvh_data is None:
        return 0.0
        
    return dvh_data.get_dose_at_volume(volume_percent)

def calculate_v_metric(dvh_data: DVHData, dose: float) -> float:
    """
    Calculate the volume percentage receiving at least the given dose.
    
    Args:
        dvh_data: DVH data for the structure
        dose: Dose value in Gy
        
    Returns:
        Volume percentage receiving at least the specified dose
    """
    if dvh_data is None:
        return 0.0
        
    return dvh_data.get_volume_at_dose(dose)

def calculate_mean_dose(dvh_data: DVHData) -> float:
    """
    Get the mean dose to the structure.
    
    Args:
        dvh_data: DVH data for the structure
        
    Returns:
        Mean dose in Gy
    """
    if dvh_data is None:
        return 0.0
        
    return dvh_data.mean_dose

def calculate_max_dose(dvh_data: DVHData, volume_percent: float = 0.03) -> float:
    """
    Calculate the maximum dose to the structure, defined as D0.03% by default.
    
    Args:
        dvh_data: DVH data for the structure
        volume_percent: Volume percentage threshold (default 0.03%)
        
    Returns:
        Maximum dose in Gy
    """
    if dvh_data is None:
        return 0.0
        
    # Use D0.03% as max dose by default
    return calculate_d_metric(dvh_data, volume_percent)

def calculate_min_dose(dvh_data: DVHData, volume_percent: float = 99.0) -> float:
    """
    Calculate the minimum dose to the structure, defined as D99% by default.
    
    Args:
        dvh_data: DVH data for the structure
        volume_percent: Volume percentage threshold (default 99%)
        
    Returns:
        Minimum dose in Gy
    """
    if dvh_data is None:
        return 0.0
        
    # Use D99% as min dose by default
    return calculate_d_metric(dvh_data, volume_percent)

def calculate_homogeneity_index(dvh_data: DVHData) -> float:
    """
    Calculate the homogeneity index of the dose distribution.
    HI = (D2% - D98%) / D50%
    
    Args:
        dvh_data: DVH data for the structure
        
    Returns:
        Homogeneity index (lower is better)
    """
    if dvh_data is None:
        return 0.0
        
    d2 = calculate_d_metric(dvh_data, 2.0)
    d98 = calculate_d_metric(dvh_data, 98.0)
    d50 = calculate_d_metric(dvh_data, 50.0)
    
    if d50 == 0.0:
        return 0.0
        
    return (d2 - d98) / d50

def calculate_conformity_index(target_dvh: DVHData, isodose_dvh: DVHData, 
                               reference_dose: float) -> float:
    """
    Calculate the conformity index of the dose distribution.
    CI = (V100% isodose) / (Target volume)
    
    Args:
        target_dvh: DVH data for the target structure
        isodose_dvh: DVH data for the isodose volume
        reference_dose: Reference dose in Gy (100% isodose)
        
    Returns:
        Conformity index (closer to 1.0 is better)
    """
    if target_dvh is None or isodose_dvh is None or reference_dose <= 0:
        return 0.0
        
    # Calculate volume of reference isodose in cc
    isodose_volume = isodose_dvh.total_volume * calculate_v_metric(isodose_dvh, reference_dose) / 100.0
    
    # Get target volume in cc
    target_volume = target_dvh.total_volume
    
    if target_volume == 0.0:
        return 0.0
        
    return isodose_volume / target_volume

def calculate_gradient_index(isodose_dvh: DVHData, reference_dose: float, 
                            lower_percent: float = 50.0) -> float:
    """
    Calculate the gradient index of the dose distribution.
    GI = (Vx% isodose) / (V100% isodose)
    Default is 50% isodose (x=50).
    
    Args:
        isodose_dvh: DVH data for the isodose volume
        reference_dose: Reference dose in Gy (100% isodose)
        lower_percent: Lower percentage for gradient evaluation (default 50%)
        
    Returns:
        Gradient index (closer to 1.0 indicates sharper dose fall-off)
    """
    if isodose_dvh is None or reference_dose <= 0:
        return 0.0
        
    # Calculate volume of reference isodose (V100%)
    v100 = calculate_v_metric(isodose_dvh, reference_dose)
    
    # Calculate volume of lower isodose (Vx%)
    vx = calculate_v_metric(isodose_dvh, reference_dose * lower_percent / 100.0)
    
    if v100 == 0.0:
        return 0.0
        
    return vx / v100

def calculate_conformation_number(target_dvh: DVHData, isodose_dvh: DVHData, 
                                 reference_dose: float) -> float:
    """
    Calculate the conformation number.
    CN = (Target volume covered by reference isodose)² / (Target volume * Reference isodose volume)
    
    Args:
        target_dvh: DVH data for the target structure
        isodose_dvh: DVH data for the isodose volume
        reference_dose: Reference dose in Gy
        
    Returns:
        Conformation number (0-1, where 1 is perfect)
    """
    if target_dvh is None or isodose_dvh is None or reference_dose <= 0:
        return 0.0
        
    # Target volume in cc
    target_volume = target_dvh.total_volume
    
    # Calculate volume of reference isodose in cc
    isodose_volume = isodose_dvh.total_volume * calculate_v_metric(isodose_dvh, reference_dose) / 100.0
    
    # Calculate target volume covered by reference isodose in cc
    target_coverage_percent = calculate_v_metric(target_dvh, reference_dose)
    target_coverage_volume = target_volume * target_coverage_percent / 100.0
    
    if target_volume == 0.0 or isodose_volume == 0.0:
        return 0.0
        
    return (target_coverage_volume * target_coverage_volume) / (target_volume * isodose_volume)

def calculate_dxx_metrics(dvh_data: DVHData, volume_percentages: List[float] = None) -> Dict[str, float]:
    """
    Calculate multiple Dxx metrics at once.
    
    Args:
        dvh_data: DVH data for the structure
        volume_percentages: List of volume percentages to calculate
            (default [1, 2, 5, 50, 95, 98, 99])
        
    Returns:
        Dictionary mapping metric names (e.g., "D95") to values
    """
    if dvh_data is None:
        return {}
        
    if volume_percentages is None:
        volume_percentages = [1, 2, 5, 50, 95, 98, 99]
    
    results = {}
    
    for vol in volume_percentages:
        metric_name = f"D{vol}"
        results[metric_name] = calculate_d_metric(dvh_data, vol)
    
    return results

def calculate_vxx_metrics(dvh_data: DVHData, dose_levels: List[float] = None, 
                         reference_dose: float = None) -> Dict[str, float]:
    """
    Calculate multiple Vxx metrics at once.
    
    Args:
        dvh_data: DVH data for the structure
        dose_levels: List of absolute dose levels in Gy to calculate
        reference_dose: Reference dose in Gy for relative dose levels
            If provided, dose_levels are interpreted as percentages
        
    Returns:
        Dictionary mapping metric names (e.g., "V20Gy" or "V95%") to values
    """
    if dvh_data is None:
        return {}
    
    results = {}
    
    if reference_dose is not None and reference_dose > 0:
        # Use relative dose levels (percentages)
        if dose_levels is None:
            dose_levels = [50, 80, 90, 95, 100, 105, 110]
        
        for pct in dose_levels:
            metric_name = f"V{pct}%"
            dose_value = reference_dose * pct / 100.0
            results[metric_name] = calculate_v_metric(dvh_data, dose_value)
    else:
        # Use absolute dose levels (Gy)
        if dose_levels is None:
            dose_levels = [5, 10, 20, 30, 40, 50]
        
        for dose in dose_levels:
            metric_name = f"V{dose}Gy"
            results[metric_name] = calculate_v_metric(dvh_data, dose)
    
    return results

def calculate_all_metrics(dvh_data: DVHData, reference_dose: float = None) -> Dict[str, Any]:
    """
    Calculate a comprehensive set of metrics for a structure.
    
    Args:
        dvh_data: DVH data for the structure
        reference_dose: Reference dose in Gy (for relative metrics)
        
    Returns:
        Dictionary containing all calculated metrics
    """
    if dvh_data is None:
        return {}
    
    metrics = {}
    
    # Basic statistics
    metrics["mean_dose"] = dvh_data.mean_dose
    metrics["median_dose"] = dvh_data.median_dose
    metrics["min_dose"] = dvh_data.min_dose
    metrics["max_dose"] = dvh_data.max_dose
    metrics["volume"] = dvh_data.total_volume
    
    # Dxx metrics
    dxx_metrics = calculate_dxx_metrics(dvh_data)
    metrics.update(dxx_metrics)
    
    # Vxx metrics (absolute)
    vxx_abs_metrics = calculate_vxx_metrics(dvh_data)
    metrics.update(vxx_abs_metrics)
    
    # Vxx metrics (relative to reference dose)
    if reference_dose is not None and reference_dose > 0:
        vxx_rel_metrics = calculate_vxx_metrics(dvh_data, reference_dose=reference_dose)
        metrics.update(vxx_rel_metrics)
    
    # Homogeneity index
    metrics["homogeneity_index"] = calculate_homogeneity_index(dvh_data)
    
    return metrics 