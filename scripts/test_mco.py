#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the Multi-Criteria Optimization (MCO) module in QuangTPS.

This script demonstrates the functionality of the MCO module, including:
- Creating objectives and constraints
- Generating Pareto-optimal solutions
- Navigating the solution space
- Visualizing the Pareto front and DVH comparisons

Usage:
    python test_mco.py

Author: QuangTPS Development Team
Date: 2025-04-15
"""

import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path to import QuangTPS modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quangtps.core.services import ServiceRegistry
from quangtps.planning.plan import Plan
from quangtps.segmentation.structures.structure import Structure
from quangtps.planning.prescription import Prescription, FractionSchedule
from quangtps.optimization.objectives import DVHObjective, MeanDoseObjective, MaxDoseObjective
from quangtps.optimization.constraints import DVHConstraint
from quangtps.optimization.mco.mco_interface import MCOEngine, MCOSolution, MCOObjectiveSpace, MCONavigator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_plan():
    """Create a test plan with structures for demonstration."""
    # Create a new plan
    plan = Plan(plan_name="MCO_Test_Plan", patient_id="TEST_PATIENT_001")
    plan.id = "TEST_PLAN_001"
    
    # Create a test prescription
    prescription = Prescription()
    prescription.dose = 70.0  # Gy
    prescription.fraction_schedule = FractionSchedule(num_fractions=35, dose_per_fraction=2.0)
    prescription.target_structure_name = "PTV"
    plan.prescription = prescription
    
    # Create test structures
    # PTV
    ptv = Structure()
    ptv.id = "STRUCT_001"
    ptv.name = "PTV"
    ptv.type = "PTV"
    ptv.color = (255, 0, 0)
    
    # OARs
    heart = Structure()
    heart.id = "STRUCT_002"
    heart.name = "Heart"
    heart.type = "ORGAN"
    heart.color = (0, 0, 255)
    
    lung_right = Structure()
    lung_right.id = "STRUCT_003"
    lung_right.name = "Lung_Right"
    lung_right.type = "ORGAN"
    lung_right.color = (0, 255, 0)
    
    lung_left = Structure()
    lung_left.id = "STRUCT_004"
    lung_left.name = "Lung_Left"
    lung_left.type = "ORGAN"
    lung_left.color = (0, 255, 128)
    
    spinal_cord = Structure()
    spinal_cord.id = "STRUCT_005"
    spinal_cord.name = "SpinalCord"
    spinal_cord.type = "ORGAN"
    spinal_cord.color = (255, 255, 0)
    
    # Mock methods for testing
    def get_structures():
        return [ptv, heart, lung_right, lung_left, spinal_cord]
    
    plan.get_structures = get_structures
    
    return plan

def mock_optimized_plan(plan, weights):
    """Create a mock 'optimized' plan with the given weights."""
    # Clone the plan
    optimized_plan = Plan(plan_name=plan.name, patient_id=plan.patient_id)
    optimized_plan.prescription = plan.prescription
    optimized_plan.get_structures = plan.get_structures
    
    # This would normally come from actual optimization
    # but for this test we're just simulating results
    return optimized_plan

def mock_objective_values(weights):
    """Calculate mock objective values based on weights."""
    # Simulate objective values
    # In a real case, these would come from evaluating the optimized plan
    values = {}
    
    # PTV Coverage (higher is better, but we're minimizing negative)
    # Lower weight means more emphasis, so lower weight = better coverage
    values["PTV Coverage"] = -95 + 15 * weights.get("PTV Coverage", 0) / 100
    
    # Heart Mean Dose (lower is better)
    # Higher weight = lower heart dose
    values["Heart Mean Dose"] = 25 - 15 * weights.get("Heart Mean Dose", 0) / 100
    
    # Lung V20 (lower is better)
    # Higher weight = lower lung V20
    values["Lung V20"] = 35 - 20 * weights.get("Lung V20", 0) / 100
    
    # Spinal Cord Max Dose (lower is better)
    # Higher weight = lower cord dose
    values["Cord Max Dose"] = 50 - 30 * weights.get("Cord Max Dose", 0) / 100
    
    # Add some random variation to simulate optimization variability
    for key in values:
        values[key] += np.random.normal(0, 1)
    
    return values

def test_mco_core_functionality():
    """Test the core functionality of the MCO module."""
    logger.info("Testing MCO core functionality...")
    
    # Create a test plan
    plan = create_test_plan()
    
    # Create MCO Engine
    mco_engine = MCOEngine(plan)
    
    # Add objectives
    for structure in plan.get_structures():
        if structure.name == "PTV":
            ptv_obj = DVHObjective(structure=structure, dose=plan.prescription.dose * 0.95, 
                                volume=95, direction="greater", weight=100)
            mco_engine.add_objective(ptv_obj, 100, "PTV Coverage")
        elif structure.name == "Heart":
            heart_obj = MeanDoseObjective(structure=structure, dose=15, weight=50)
            mco_engine.add_objective(heart_obj, 50, "Heart Mean Dose")
        elif "Lung" in structure.name:
            lung_obj = DVHObjective(structure=structure, dose=20, volume=30, 
                                   direction="less", weight=50)
            mco_engine.add_objective(lung_obj, 50, "Lung V20")
        elif structure.name == "SpinalCord":
            cord_obj = MaxDoseObjective(structure=structure, dose=45, weight=80)
            mco_engine.add_objective(cord_obj, 80, "Cord Max Dose")
    
    # Add constraints (optional)
    for structure in plan.get_structures():
        if structure.name == "SpinalCord":
            cord_constraint = DVHConstraint(structure=structure, dose=50, volume=0, 
                                           direction="less")
            mco_engine.add_constraint(cord_constraint)
    
    # Mock the optimization engine's behavior
    # In a real case, this would use the actual optimization engine
    def mock_generate_solution(weight_vector):
        # Create a clone of the plan
        optimized_plan = mock_optimized_plan(plan, weight_vector)
        
        # Calculate mock objective values
        objective_values = mock_objective_values(weight_vector)
        
        # Create MCO solution
        solution = MCOSolution(
            plan=optimized_plan,
            objectives=objective_values,
            constraints_satisfied=True,
            metadata={"weights": weight_vector}
        )
        
        return solution
    
    # Replace the _generate_solution method with our mock
    mco_engine._generate_solution = mock_generate_solution
    
    # Generate initial solutions
    logger.info("Generating initial solutions...")
    solutions = mco_engine.generate_initial_solutions(10)
    
    logger.info(f"Generated {len(solutions)} solutions")
    
    # Test navigator
    logger.info("Testing MCO Navigator...")
    navigator = mco_engine.navigator
    
    # Select first solution
    navigator.select_solution(0)
    current = mco_engine.objective_space.get_current_solution()
    logger.info(f"Current solution objectives: {current.objectives}")
    
    # Select another solution
    navigator.select_solution(2)
    current = mco_engine.objective_space.get_current_solution()
    logger.info(f"Current solution objectives: {current.objectives}")
    
    # Test undo/redo
    navigator.undo()
    current = mco_engine.objective_space.get_current_solution()
    logger.info(f"After undo, current solution objectives: {current.objectives}")
    
    navigator.redo()
    current = mco_engine.objective_space.get_current_solution()
    logger.info(f"After redo, current solution objectives: {current.objectives}")
    
    # Test plotting
    logger.info("Testing Pareto front visualization...")
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    
    # Choose two objectives to plot
    mco_engine.objective_space.plot_pareto_front("PTV Coverage", "Heart Mean Dose", ax)
    
    plt.savefig("pareto_front_test.png")
    logger.info("Saved Pareto front visualization to 'pareto_front_test.png'")
    
    return mco_engine

def main():
    parser = argparse.ArgumentParser(description="Test the MCO module in QuangTPS")
    parser.add_argument('--visualize', action='store_true', help="Show visualizations")
    args = parser.parse_args()
    
    logger.info("Starting MCO module test")
    
    # Test core functionality
    mco_engine = test_mco_core_functionality()
    
    # Show visualizations if requested
    if args.visualize:
        logger.info("Displaying visualizations...")
        plt.show()
    
    logger.info("MCO module test completed successfully")

if __name__ == "__main__":
    main() 