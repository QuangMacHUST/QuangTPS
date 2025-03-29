#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DVH (Dose-Volume Histogram) calculation and analysis module.

This module provides functions and classes for calculating, analyzing, and
visualizing dose-volume histograms in radiotherapy treatment planning.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)

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
        _get_volume_at_dose
    )
except ImportError as e:
    logger.error(f"Error importing DVH calculation functions: {e}")

try:
    from quangtps.evaluation.dvh.dvh_visualization import (
        get_structure_color,
        plot_dvh,
        plot_multiple_dvh,
        create_dvh_report,
        plot_dvh_bands,
        export_dvh_to_csv
    )
except ImportError as e:
    logger.error(f"Error importing DVH visualization functions: {e}")

# Define a compatibility function for code that might still use old interfaces
def calculate_dvh_from_dose_grid(dose_grid, structure_mask, **kwargs):
    """
    Compatibility function for older code that uses the old DVH calculation interface.
    
    Parameters
    ----------
    dose_grid : array-like
        3D dose grid with dose values
    structure_mask : array-like
        Binary mask of the structure
    **kwargs : dict
        Additional parameters to pass to calculate_dvh
        
    Returns
    -------
    dict
        Dictionary containing DVH data
    """
    try:
        return calculate_dvh(dose_grid, structure_mask, **kwargs)
    except Exception as e:
        logger.error(f"Error in calculate_dvh_from_dose_grid: {e}")
        # Return empty DVH data
        return {
            'dose_bins': [],
            'differential_volume': [],
            'cumulative_volume': [],
            'min_dose': 0,
            'max_dose': 0,
            'mean_dose': 0,
            'median_dose': 0,
            'std_dose': 0,
            'volume': 0
        }

__all__ = [
    'calculate_dvh',
    'calculate_dvh_for_plan',
    'calculate_dvh_metrics',
    'calculate_conformity_index',
    'calculate_homogeneity_index',
    'calculate_gradient_index',
    'calculate_equivalent_uniform_dose',
    'calculate_dvh_from_dose_grid',
    'get_structure_color',
    'plot_dvh',
    'plot_multiple_dvh',
    'create_dvh_report',
    'plot_dvh_bands',
    'export_dvh_to_csv'
]
