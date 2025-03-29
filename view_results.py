#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DVH Viewer - Visualizer for Dose-Volume Histogram results
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import tempfile
from scipy import interpolate

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import DVH functions
try:
    from quangtps.evaluation.dvh import (
        calculate_dvh,
        calculate_dvh_for_plan,
        calculate_dvh_metrics,
        calculate_conformity_index,
        calculate_homogeneity_index,
        plot_dvh,
        plot_multiple_dvh,
        create_dvh_report
    )
    use_system_dvh = True
    logger.info("Using system DVH module")
except ImportError as e:
    logger.warning(f"Could not import QuangTPS DVH module: {e}")
    logger.warning("Using built-in DVH calculations instead")
    use_system_dvh = False

def create_test_data():
    """Create test data for DVH calculation."""
    # Create a dose grid
    grid_size = 100
    dose_grid = np.zeros((grid_size, grid_size, grid_size))
    
    # Create a gaussian dose distribution
    x, y, z = np.meshgrid(
        np.linspace(-10, 10, grid_size),
        np.linspace(-10, 10, grid_size),
        np.linspace(-10, 10, grid_size)
    )
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Create structures
    ptv_mask = r < 3
    oar1_mask = (x > 0) & (y > 0) & (z > 0) & (r < 6)
    oar2_mask = (x < -3) & (y < -3) & (z < -3) & (r < 8)
    oar3_mask = (r > 3.5) & (r < 5)
    body_mask = r < 10
    
    # Fill dose grid with realistic distribution
    dose_grid = 60 * np.exp(-(r**2) / (2*4**2))
    
    # Add some hotspots
    hotspot1_center = (grid_size // 4, grid_size // 4, grid_size // 2)
    hotspot_r = np.sqrt(
        (np.arange(grid_size)[:, np.newaxis, np.newaxis] - hotspot1_center[0])**2 +
        (np.arange(grid_size)[np.newaxis, :, np.newaxis] - hotspot1_center[1])**2 +
        (np.arange(grid_size)[np.newaxis, np.newaxis, :] - hotspot1_center[2])**2
    )
    dose_grid += 10 * np.exp(-(hotspot_r**2) / (2*2**2))
    
    # Create a dictionary of structures
    structures = {
        'PTV': ptv_mask,
        'OAR1': oar1_mask,
        'OAR2': oar2_mask,
        'OAR3': oar3_mask,
        'Body': body_mask
    }
    
    return dose_grid, structures

def calculate_dvh_metrics_direct(dvh_data):
    """
    Calculate DVH metrics directly from DVH data (for testing).
    
    Parameters
    ----------
    dvh_data : dict
        DVH data dictionary with dose_bins and cumulative_volume
        
    Returns
    -------
    dict
        Dictionary of DVH metrics
    """
    # Extract arrays from DVH data
    dose_bins = dvh_data.get('dose_bins', np.array([0]))
    cumulative_volume = dvh_data.get('cumulative_volume', np.array([0]))
    
    # Basic metrics directly from DVH data
    metrics = {
        'Dmin': dvh_data.get('min_dose', 0),
        'Dmax': dvh_data.get('max_dose', 0),
        'Dmean': dvh_data.get('mean_dose', 0),
        'Dmedian': dvh_data.get('median_dose', 0),
        'volume': dvh_data.get('volume', 0)
    }
    
    # Check if we have valid data
    if len(dose_bins) <= 1 or len(cumulative_volume) <= 1:
        return metrics
    
    # Create interpolation functions
    try:
        interp_func_dose = interpolate.interp1d(
            cumulative_volume, dose_bins, 
            bounds_error=False, 
            fill_value=(dose_bins[-1], dose_bins[0])
        )
        
        interp_func_volume = interpolate.interp1d(
            dose_bins, cumulative_volume, 
            bounds_error=False, 
            fill_value=(100.0, 0.0)
        )
        
        # Calculate Dx metrics (dose to x% of volume)
        dx_values = [2, 5, 50, 95, 98]
        for x in dx_values:
            metrics[f'D{x}'] = float(interp_func_dose(x))
        
        # Calculate Vx metrics (volume receiving x Gy)
        vx_values = [5, 10, 20, 30, 40, 50]
        for x in vx_values:
            metrics[f'V{x}'] = float(interp_func_volume(x))
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    
    return metrics

def calculate_dvh_custom(dose_grid, structure_mask, num_bins=100):
    """
    Calculate DVH manually if system module is not available.
    
    Parameters
    ----------
    dose_grid : array
        3D dose grid
    structure_mask : array
        3D structure mask
    num_bins : int
        Number of bins for histogram
        
    Returns
    -------
    dict
        DVH data dictionary
    """
    # Extract dose values within the structure
    doses_in_structure = dose_grid[structure_mask > 0]
    
    if len(doses_in_structure) == 0:
        logger.warning("No voxels in structure!")
        return {
            'dose_bins': np.array([0]),
            'differential_volume': np.array([0]),
            'cumulative_volume': np.array([0]),
            'min_dose': 0,
            'max_dose': 0,
            'mean_dose': 0,
            'median_dose': 0,
            'volume': 0
        }
    
    # Calculate statistics
    min_dose = np.min(doses_in_structure)
    max_dose = np.max(doses_in_structure)
    mean_dose = np.mean(doses_in_structure)
    median_dose = np.median(doses_in_structure)
    
    # Create dose bins
    dose_bins = np.linspace(0, max_dose * 1.05, num_bins + 1)
    bin_centers = (dose_bins[1:] + dose_bins[:-1]) / 2
    
    # Calculate histogram
    hist, _ = np.histogram(doses_in_structure, bins=dose_bins)
    
    # Normalize to percentage
    total_voxels = len(doses_in_structure)
    differential_volume = (hist / total_voxels) * 100
    
    # Calculate cumulative DVH
    cumulative_volume = np.cumsum(differential_volume[::-1])[::-1]
    
    # Calculate volume (assume 1mm³ voxels = 0.001 cm³)
    volume = total_voxels * 0.001
    
    return {
        'dose_bins': bin_centers,
        'differential_volume': differential_volume,
        'cumulative_volume': cumulative_volume,
        'min_dose': min_dose,
        'max_dose': max_dose,
        'mean_dose': mean_dose,
        'median_dose': median_dose,
        'volume': volume
    }

def calculate_dvh_for_all_structures(dose_grid, structures):
    """
    Calculate DVH for all structures using appropriate method.
    
    Parameters
    ----------
    dose_grid : array
        3D dose grid
    structures : dict
        Dictionary of structure masks
        
    Returns
    -------
    dict
        Dictionary of DVH data for each structure
    """
    dvh_data = {}
    
    if use_system_dvh:
        # Use system DVH module
        dvh_data = calculate_dvh_for_plan(dose_grid, structures)
    else:
        # Use manual calculation
        for name, mask in structures.items():
            dvh_data[name] = calculate_dvh_custom(dose_grid, mask)
    
    return dvh_data

def main():
    """
    Main function to run the DVH viewer.
    """
    logger.info("Starting DVH Viewer")
    
    # Create test data
    logger.info("Creating test data...")
    dose_grid, structures = create_test_data()
    
    logger.info(f"Dose grid shape: {dose_grid.shape}")
    logger.info(f"Dose range: {dose_grid.min():.2f} - {dose_grid.max():.2f} Gy")
    
    for name, mask in structures.items():
        logger.info(f"Structure {name}: {np.sum(mask)} voxels")
    
    # Calculate DVH for all structures
    logger.info("Calculating DVH for all structures...")
    dvh_data = calculate_dvh_for_all_structures(dose_grid, structures)
    
    # Create figure for cumulative DVH
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Cumulative Dose-Volume Histogram")
    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume (%)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot DVH for each structure
    structure_colors = {
        'PTV': 'red',
        'OAR1': 'blue',
        'OAR2': 'green',
        'OAR3': 'purple',
        'Body': 'gray'
    }
    
    # Add metrics to display
    metrics_to_show = ['D95', 'D50', 'Dmean', 'V20']
    
    for name, color in structure_colors.items():
        if name in dvh_data:
            # Plot the DVH curve
            ax.plot(
                dvh_data[name]['dose_bins'],
                dvh_data[name]['cumulative_volume'],
                label=name,
                color=color,
                linewidth=2
            )
            
            # Calculate and show metrics
            metrics = calculate_dvh_metrics_direct(dvh_data[name])
            
            # Format metrics text
            metrics_text = []
            for metric in metrics_to_show:
                if metric in metrics:
                    if metric.startswith('D'):
                        metrics_text.append(f"{metric}: {metrics[metric]:.1f} Gy")
                    elif metric.startswith('V'):
                        metrics_text.append(f"{metric}: {metrics[metric]:.1f}%")
            
            # Position text with an offset based on structure
            offset = list(structure_colors.keys()).index(name) * 15
            ax.annotate(
                f"{name}: " + ", ".join(metrics_text),
                xy=(0.5, 15 + offset),
                xycoords='data',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
                fontsize=8
            )
    
    ax.legend(loc='lower left')
    ax.set_ylim(0, 105)
    ax.set_xlim(0, dose_grid.max() * 1.05)

    # Calculate and display plan evaluation indices
    prescription_dose = 60.0  # Gy
    
    # Calculate metrics directly
    ptv_metrics = calculate_dvh_metrics_direct(dvh_data['PTV'])
    
    # Conformity Index
    v_rx = np.interp(prescription_dose, 
                     dvh_data['PTV']['dose_bins'], 
                     dvh_data['PTV']['cumulative_volume'])
    ci = v_rx / 100.0 if v_rx > 0 else 0
    
    # Homogeneity Index
    d2 = ptv_metrics.get('D2', 0)
    d98 = ptv_metrics.get('D98', 0)
    hi = (d2 - d98) / prescription_dose if prescription_dose > 0 else 0
    
    # Add indices to the plot
    indices_text = f"Prescription: {prescription_dose:.1f} Gy\n" \
                  f"Conformity Index: {ci:.4f}\n" \
                  f"Homogeneity Index: {hi:.4f}"
    
    ax.text(
        0.02, 0.02, indices_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
        fontsize=9,
        verticalalignment='bottom'
    )
    
    # Save the figure
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"dvh_results_{timestamp}.png"
    plt.savefig(output_path, dpi=300)
    logger.info(f"DVH plot saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    # If system DVH module is available, use it to create a report
    if use_system_dvh:
        try:
            # Try to create a comprehensive report
            logger.info("Creating comprehensive DVH report...")
            report_path = f"dvh_report_{timestamp}.png"
            
            create_dvh_report(
                dvh_list=[dvh_data],
                structure_names=list(structures.keys()),
                plan_names=["Test Plan"],
                prescription_doses={'PTV': prescription_dose},
                output_path=report_path,
                show_statistics=True
            )
            
            logger.info(f"DVH report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error creating DVH report: {e}")
    
    logger.info("DVH visualization complete")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 