#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Plan Evaluation Script

This script demonstrates the use of the QuangTPS plan evaluation module
to evaluate a test radiotherapy treatment plan.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import QuangTPS modules
try:
    from quangtps.evaluation import (
        evaluate_plan,
        PlanEvaluation,
        calculate_dvh,
        calculate_dvh_for_plan
    )
    use_system_modules = True
    logger.info("Using QuangTPS evaluation modules")
except ImportError as e:
    logger.warning(f"Error importing QuangTPS modules: {e}")
    logger.warning("Using built-in test data generation only")
    use_system_modules = False

def create_test_data():
    """
    Create test data for plan evaluation.
    
    Returns
    -------
    tuple
        (dose_grid, structures, prescription_doses, structure_types)
    """
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
    structures = {}
    
    # Planning Target Volume (PTV)
    ptv_mask = r < 3
    structures['PTV'] = ptv_mask
    
    # Clinical Target Volume (CTV)
    ctv_mask = r < 2.5
    structures['CTV'] = ctv_mask
    
    # Organs at Risk (OARs)
    # Lung
    lung_mask = ((x > 0) & (y > 0) & (z > 0) & (r < 6))
    structures['Lung'] = lung_mask
    
    # Heart
    heart_center = np.array([-3, 2, 1])
    heart_r = np.sqrt(
        (x - heart_center[0])**2 + 
        (y - heart_center[1])**2 + 
        (z - heart_center[2])**2
    )
    heart_mask = heart_r < 2.5
    structures['Heart'] = heart_mask
    
    # Spinal Cord
    cord_center = np.array([0, -4, 0])
    cord_r = np.sqrt(
        (x - cord_center[0])**2 + 
        (y - cord_center[1])**2
    )
    cord_mask = (cord_r < 1) & (np.abs(z) < 8)
    structures['SpinalCord'] = cord_mask
    
    # Body contour
    body_mask = r < 10
    structures['Body'] = body_mask
    
    # Fill dose grid with realistic distribution
    # Primary target dose
    dose_grid = 60 * np.exp(-(r**2) / (2*3**2))
    
    # Add some hotspots
    hotspot_center = (grid_size // 4, grid_size // 4, grid_size // 2)
    hotspot_r = np.sqrt(
        (np.arange(grid_size)[:, np.newaxis, np.newaxis] - hotspot_center[0])**2 +
        (np.arange(grid_size)[np.newaxis, :, np.newaxis] - hotspot_center[1])**2 +
        (np.arange(grid_size)[np.newaxis, np.newaxis, :] - hotspot_center[2])**2
    )
    dose_grid += 10 * np.exp(-(hotspot_r**2) / (2*2**2))
    
    # Define prescription doses
    prescription_doses = {
        'PTV': 60.0,  # Gy
        'CTV': 66.0   # Gy
    }
    
    # Define structure types
    structure_types = {
        'PTV': 'PTV',
        'CTV': 'CTV',
        'Lung': 'OAR',
        'Heart': 'OAR',
        'SpinalCord': 'OAR',
        'Body': 'EXTERNAL'
    }
    
    return dose_grid, structures, prescription_doses, structure_types

def define_clinical_constraints():
    """
    Define clinical constraints for plan evaluation.
    
    Returns
    -------
    dict
        Dictionary of constraints for each structure
    """
    constraints = {
        'PTV': [
            {'type': 'D95', 'goal': 57.0, 'relation': '>', 'priority': 'HIGH', 'unit': 'Gy'},
            {'type': 'D2', 'goal': 63.0, 'relation': '<', 'priority': 'MEDIUM', 'unit': 'Gy'},
            {'type': 'D98', 'goal': 55.8, 'relation': '>', 'priority': 'MEDIUM', 'unit': 'Gy'}
        ],
        'CTV': [
            {'type': 'D98', 'goal': 64.7, 'relation': '>', 'priority': 'HIGH', 'unit': 'Gy'},
            {'type': 'D50', 'goal': 66.0, 'relation': '=', 'priority': 'MEDIUM', 'unit': 'Gy'}
        ],
        'Lung': [
            {'type': 'V20', 'goal': 30.0, 'relation': '<', 'priority': 'HIGH', 'unit': '%'},
            {'type': 'V5', 'goal': 60.0, 'relation': '<', 'priority': 'MEDIUM', 'unit': '%'},
            {'type': 'Dmean', 'goal': 15.0, 'relation': '<', 'priority': 'MEDIUM', 'unit': 'Gy'}
        ],
        'Heart': [
            {'type': 'V30', 'goal': 30.0, 'relation': '<', 'priority': 'HIGH', 'unit': '%'},
            {'type': 'Dmean', 'goal': 20.0, 'relation': '<', 'priority': 'MEDIUM', 'unit': 'Gy'}
        ],
        'SpinalCord': [
            {'type': 'Dmax', 'goal': 45.0, 'relation': '<', 'priority': 'HIGH', 'unit': 'Gy'},
            {'type': 'D2', 'goal': 40.0, 'relation': '<', 'priority': 'MEDIUM', 'unit': 'Gy'}
        ]
    }
    
    return constraints

def run_basic_evaluation():
    """
    Run a basic evaluation of a test plan.
    """
    logger.info("Starting basic plan evaluation")
    
    # Create test data
    logger.info("Creating test data...")
    dose_grid, structures, prescription_doses, structure_types = create_test_data()
    
    logger.info(f"Created dose grid with shape {dose_grid.shape}")
    logger.info(f"Dose range: {dose_grid.min():.2f} - {dose_grid.max():.2f} Gy")
    
    for name, mask in structures.items():
        logger.info(f"Structure {name}: {np.sum(mask)} voxels")
    
    # If QuangTPS modules are available, use them
    if use_system_modules:
        # Evaluate the plan
        logger.info("Evaluating plan using QuangTPS modules...")
        plan_name = "Test Plan"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"plan_evaluation_{timestamp}.png"
        
        evaluator = evaluate_plan(
            dose_grid=dose_grid,
            structures=structures,
            prescription_doses=prescription_doses,
            structure_types=structure_types,
            plan_name=plan_name,
            output_path=output_path
        )
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics()
        
        # Calculate plan indices
        indices = evaluator.calculate_plan_indices()
        
        # Define and evaluate constraints
        constraints = define_clinical_constraints()
        constraint_results = evaluator.evaluate_constraints(constraints)
        
        # Print results
        logger.info("\n--- Plan Evaluation Results ---")
        
        logger.info("\nPlan Indices:")
        for name, value in indices.items():
            logger.info(f"  {name}: {value:.4f}")
        
        logger.info("\nDVH Metrics:")
        for structure, structure_metrics in metrics.items():
            logger.info(f"\n  {structure}:")
            for metric, value in structure_metrics.items():
                if metric.startswith('D'):
                    logger.info(f"    {metric}: {value:.2f} Gy")
                elif metric.startswith('V'):
                    logger.info(f"    {metric}: {value:.2f}%")
                else:
                    logger.info(f"    {metric}: {value:.2f}")
        
        logger.info("\nConstraint Evaluation:")
        for structure, structure_constraints in constraint_results.items():
            logger.info(f"\n  {structure}:")
            for constraint in structure_constraints:
                result = constraint['result']
                actual = constraint['actual']
                goal = constraint['goal']
                relation = constraint['relation']
                type_ = constraint['type']
                unit = constraint.get('unit', '')
                
                result_color = ''
                if result == 'PASS':
                    result_color = 'PASS'
                elif result == 'FAIL':
                    result_color = 'FAIL'
                else:
                    result_color = 'BORDERLINE'
                
                logger.info(f"    {type_} {relation} {goal} {unit}: {actual:.2f} {unit} [{result_color}]")
        
        # Plot DVH
        logger.info("\nCreating DVH plot...")
        fig, ax = evaluator.plot_dvh(show_metrics=True)
        dvh_path = f"dvh_plot_{timestamp}.png"
        fig.savefig(dvh_path, dpi=300, bbox_inches='tight')
        logger.info(f"DVH plot saved to {dvh_path}")
        
        # Create comprehensive report
        logger.info("\nCreating comprehensive report...")
        report_path = f"plan_report_{timestamp}.png"
        report = evaluator.create_evaluation_report(
            output_path=report_path,
            include_metrics=True,
            include_constraints=True,
            constraints=constraints,
            include_biological=True
        )
        logger.info(f"Report saved to {report_path}")
        
        # Show the plot
        plt.show()
    else:
        # Basic evaluation without QuangTPS modules
        logger.info("QuangTPS modules not available, running basic evaluation only...")
        
        # Calculate basic metrics
        logger.info("Calculating basic metrics...")
        
        # Create simple DVH plot
        logger.info("Creating basic DVH plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Dose-Volume Histogram")
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        
        # Plot simple DVH curves using numpy histogram
        colors = {
            'PTV': 'red',
            'CTV': 'orange',
            'Lung': 'blue',
            'Heart': 'pink',
            'SpinalCord': 'yellow',
            'Body': 'gray'
        }
        
        for name, mask in structures.items():
            if name in colors:
                # Extract doses in structure
                structure_doses = dose_grid[mask > 0]
                
                if len(structure_doses) > 0:
                    # Create histogram
                    hist, bins = np.histogram(structure_doses, bins=100, density=True)
                    cum_hist = np.cumsum(hist) / np.sum(hist) * 100
                    cum_hist = 100 - cum_hist  # Convert to volume receiving at least X dose
                    
                    # Plot
                    bin_centers = (bins[1:] + bins[:-1]) / 2
                    ax.plot(bin_centers, cum_hist, label=name, color=colors[name], linewidth=2)
                    
                    # Calculate basic metrics
                    d95 = np.percentile(structure_doses, 5)  # D95
                    d50 = np.median(structure_doses)  # D50
                    dmax = np.max(structure_doses)  # Dmax
                    dmean = np.mean(structure_doses)  # Dmean
                    
                    # Add metrics text
                    metrics_text = f"{name}:\nD95: {d95:.1f} Gy\nD50: {d50:.1f} Gy\nDmax: {dmax:.1f} Gy"
                    ax.text(0.7 * dmax, 90 - list(colors.keys()).index(name) * 15, 
                           metrics_text, fontsize=8, color=colors[name],
                           bbox=dict(facecolor='white', alpha=0.7))
        
        ax.grid(True)
        ax.legend(loc='lower left')
        ax.set_ylim(0, 105)
        
        # Save and show the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"basic_dvh_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        logger.info(f"Basic DVH plot saved to {output_path}")
        plt.show()
    
    logger.info("Plan evaluation complete")
    return 0

if __name__ == "__main__":
    sys.exit(run_basic_evaluation()) 