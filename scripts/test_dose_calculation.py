#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dose Calculation Test Script

This script creates a simple water phantom, defines beams, and calculates 
the dose using different algorithms to compare their results.

Usage:
    python test_dose_calculation.py [--algorithm ALGORITHM] [--output OUTPUT_DIR]
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path so we can import QuangTPS modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.dose.beam_data_processor import BeamModel
from quangtps.dose.algorithms import get_available_algorithms, get_algorithm_instance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DoseCalculationTest")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test dose calculation algorithms")
    
    parser.add_argument(
        "--algorithm", "-a", 
        choices=["pencil_beam", "collapsed_cone", "monte_carlo", "all"], 
        default="all",
        help="Dose calculation algorithm to test"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dose_test_results",
        help="Output directory for results"
    )
    
    return parser.parse_args()

def create_water_phantom():
    """
    Create a simple water phantom image.
    
    Returns
    -------
    Image
        Water phantom image
    """
    # Create a 30x30x30 cm water phantom with 5mm voxels
    phantom = Image()
    voxel_size = 0.5  # cm
    phantom_size = 60  # voxels (30 cm)
    
    # Create array of all ones (water)
    phantom.data = np.ones((phantom_size, phantom_size, phantom_size), dtype=np.float32)
    
    # Set image properties
    phantom.pixel_spacing = [voxel_size, voxel_size, voxel_size]
    phantom.origin = [-15, -15, -15]  # cm
    
    # Add some heterogeneities
    
    # Bone (relative electron density ~ 1.8)
    # 5x5x5 cm cube at (2, 2, 0) cm
    x_start = int((2 - phantom.origin[0]) / voxel_size)
    y_start = int((2 - phantom.origin[1]) / voxel_size)
    z_start = int((0 - phantom.origin[2]) / voxel_size)
    
    cube_size = int(5 / voxel_size)
    phantom.data[x_start:x_start+cube_size, y_start:y_start+cube_size, z_start:z_start+cube_size] = 1.8
    
    # Air (relative electron density ~ 0.001)
    # 5x5x5 cm cube at (-7, -7, 0) cm
    x_start = int((-7 - phantom.origin[0]) / voxel_size)
    y_start = int((-7 - phantom.origin[1]) / voxel_size)
    z_start = int((0 - phantom.origin[2]) / voxel_size)
    
    phantom.data[x_start:x_start+cube_size, y_start:y_start+cube_size, z_start:z_start+cube_size] = 0.001
    
    logger.info(f"Created water phantom: {phantom_size}x{phantom_size}x{phantom_size} voxels, {voxel_size} cm voxel size")
    
    return phantom

def create_beams():
    """
    Create test beams for dose calculation.
    
    Returns
    -------
    list of Beam
        List of test beams
    """
    beams = []
    
    # Create a simple AP beam
    ap_beam = Beam()
    ap_beam.name = "AP"
    ap_beam.energy = 6  # 6 MV
    ap_beam.gantry_angle = 0.0
    ap_beam.collimator_angle = 0.0
    ap_beam.field_size = (10.0, 10.0)  # 10x10 cm field
    ap_beam.sad = 100.0  # 100 cm SAD
    
    beams.append(ap_beam)
    
    # Create a lateral beam
    lat_beam = Beam()
    lat_beam.name = "Lateral"
    lat_beam.energy = 6  # 6 MV
    lat_beam.gantry_angle = 90.0
    lat_beam.collimator_angle = 0.0
    lat_beam.field_size = (10.0, 10.0)  # 10x10 cm field
    lat_beam.sad = 100.0  # 100 cm SAD
    
    beams.append(lat_beam)
    
    logger.info(f"Created {len(beams)} test beams")
    
    return beams

def calculate_dose(phantom, beams, algorithm_name):
    """
    Calculate dose using the specified algorithm.
    
    Parameters
    ----------
    phantom : Image
        Phantom image
    beams : list of Beam
        List of beams
    algorithm_name : str
        Name of the algorithm to use
    
    Returns
    -------
    tuple
        Tuple containing (dose_result, calculation_time)
    """
    # Create a beam model
    beam_model = BeamModel()
    beam_model.name = "Test 6MV"
    beam_model.energy = 6
    
    # Get the algorithm
    algorithm = get_algorithm_instance(algorithm_name)
    
    # Set beam model
    algorithm.set_beam_model(beam_model)
    
    # Set algorithm parameters for a faster test
    if algorithm_name == "pencil_beam":
        algorithm.set_parameters(grid_size=0.5, threads=4)
    elif algorithm_name == "collapsed_cone":
        algorithm.set_parameters(grid_size=0.5, threads=4, num_cones=16)
    elif algorithm_name == "monte_carlo":
        algorithm.set_parameters(
            num_histories=10000,  # Low number for faster test
            grid_size=0.5,
            threads=4,
            use_gpu=False  # Set to True if you have a GPU and CUDA
        )
    
    # Calculate dose
    logger.info(f"Calculating dose using {algorithm_name}...")
    start_time = time.time()
    
    try:
        result = algorithm.calculate(phantom, beams)
        calculation_time = time.time() - start_time
        
        logger.info(f"Dose calculation completed in {calculation_time:.2f} seconds")
        return result, calculation_time
    except Exception as e:
        logger.error(f"Error calculating dose with {algorithm_name}: {e}")
        return None, time.time() - start_time

def plot_results(results, output_dir):
    """
    Plot dose calculation results.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm names to (result, time) tuples
    output_dir : str
        Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot central slices for each algorithm
    plt.figure(figsize=(15, 5))
    
    for i, (algo_name, (result, calc_time)) in enumerate(results.items()):
        if result is None:
            continue
        
        # Get the central slice
        dose_grid = result.dose_grid
        central_slice = dose_grid[:, :, dose_grid.shape[2] // 2]
        
        # Normalize to maximum dose
        max_dose = np.max(dose_grid)
        if max_dose > 0:
            normalized_slice = central_slice / max_dose * 100.0
        else:
            normalized_slice = central_slice
        
        # Plot
        plt.subplot(1, len(results), i + 1)
        plt.imshow(normalized_slice.T, cmap='jet', origin='lower')
        plt.colorbar(label='Dose (%)')
        plt.title(f"{algo_name}\n{calc_time:.2f} seconds")
        plt.xlabel('X')
        plt.ylabel('Y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dose_comparison.png"), dpi=300)
    logger.info(f"Saved dose comparison plot to {output_dir}/dose_comparison.png")
    
    # Plot profiles through isocenter
    plt.figure(figsize=(10, 6))
    
    for algo_name, (result, _) in results.items():
        if result is None:
            continue
        
        # Get the dose grid
        dose_grid = result.dose_grid
        
        # Get profile through isocenter
        isocenter_idx = dose_grid.shape // 2
        profile_x = dose_grid[:, isocenter_idx[1], isocenter_idx[2]]
        
        # Normalize to maximum dose
        max_dose = np.max(dose_grid)
        if max_dose > 0:
            profile_x = profile_x / max_dose * 100.0
        
        # Plot
        plt.plot(profile_x, label=algo_name)
    
    plt.xlabel('X position (voxel)')
    plt.ylabel('Dose (%)')
    plt.title('Dose Profile Through Isocenter')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "dose_profile.png"), dpi=300)
    logger.info(f"Saved dose profile plot to {output_dir}/dose_profile.png")
    
    # Create comparison tables
    with open(os.path.join(output_dir, "results_summary.txt"), 'w') as f:
        f.write("Dose Calculation Results Summary\n")
        f.write("================================\n\n")
        
        # Table of calculation times
        f.write("Calculation Times:\n")
        f.write("-----------------\n")
        for algo_name, (_, calc_time) in results.items():
            f.write(f"{algo_name}: {calc_time:.2f} seconds\n")
        
        f.write("\n")
        
        # Table of dose statistics
        f.write("Dose Statistics:\n")
        f.write("--------------\n")
        f.write(f"{'Algorithm':<15} {'Min Dose':<10} {'Max Dose':<10} {'Mean Dose':<10}\n")
        
        for algo_name, (result, _) in results.items():
            if result is None:
                continue
            
            dose_grid = result.dose_grid
            min_dose = np.min(dose_grid)
            max_dose = np.max(dose_grid)
            mean_dose = np.mean(dose_grid)
            
            f.write(f"{algo_name:<15} {min_dose:<10.4f} {max_dose:<10.4f} {mean_dose:<10.4f}\n")
    
    logger.info(f"Saved results summary to {output_dir}/results_summary.txt")

def main():
    """Main function."""
    args = parse_args()
    
    # Create test phantom
    phantom = create_water_phantom()
    
    # Create test beams
    beams = create_beams()
    
    # Run dose calculations
    results = {}
    
    # Determine which algorithms to run
    if args.algorithm == "all":
        algorithms = ["pencil_beam", "collapsed_cone", "monte_carlo"]
    else:
        algorithms = [args.algorithm]
    
    # Calculate dose for each algorithm
    for algo_name in algorithms:
        result, calc_time = calculate_dose(phantom, beams, algo_name)
        results[algo_name] = (result, calc_time)
    
    # Plot and save results
    plot_results(results, args.output)
    
    logger.info("Dose calculation test completed!")

if __name__ == "__main__":
    main() 