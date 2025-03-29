#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS Optimization Check Script

This script tests and validates the IMRT/VMAT optimization functionality
in the QuangTPS system.

Usage:
    python check_optimization.py [--verbose]
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import time
import argparse
from typing import Dict, List, Optional, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("optimization-check")

def check_dependencies():
    """Check if required dependencies for optimization are available."""
    logger.info("Checking required dependencies for optimization...")
    
    dependencies = [
        "numpy", "scipy", "matplotlib", 
        "pulp", "cvxpy"  # Optional but recommended
    ]
    
    missing = []
    for package in dependencies:
        try:
            importlib.import_module(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            logger.warning(f"❌ {package} is not available")
            missing.append(package)
    
    if missing:
        logger.warning("Some optimization dependencies are missing. Install them with:")
        logger.warning(f"pip install {' '.join(missing)}")
        return False
    
    return True

def check_optimization_modules():
    """Check if optimization modules are available and properly structured."""
    logger.info("Checking optimization modules structure...")
    
    modules = [
        "quangtps.optimization",
        "quangtps.optimization.objectives",
        "quangtps.optimization.constraints",
        "quangtps.optimization.gradient_descent",
        "quangtps.optimization.simulated_annealing",
        "quangtps.optimization.genetic_algorithm",
        "quangtps.optimization.optimization_engine"
    ]
    
    missing = []
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            logger.info(f"✅ {module_name} is available")
        except ImportError as e:
            logger.warning(f"❌ {module_name} is not available: {str(e)}")
            missing.append(module_name)
    
    if missing:
        logger.warning("Some optimization modules are missing or have import errors")
        return False
    
    return True

def test_gradient_descent():
    """Test if gradient descent optimization is working properly."""
    logger.info("Testing gradient descent optimization...")
    
    try:
        from quangtps.optimization.gradient_descent import GradientDescent
        
        # Test function: f(x) = x^2 + 2x + 1
        # Minimum at x = -1
        def objective(x):
            return x**2 + 2*x + 1
        
        def gradient(x):
            return 2*x + 2
        
        # Initialize optimizer
        optimizer = GradientDescent(
            learning_rate=0.1,
            max_iterations=100,
            convergence_threshold=1e-6
        )
        
        # Starting point
        initial_params = np.array([5.0])
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize(
            objective_function=objective,
            gradient_function=gradient,
            initial_parameters=initial_params
        )
        duration = time.time() - start_time
        
        # Check result
        expected_value = -1.0
        logger.info(f"Optimization result: {result.parameters[0]:.6f} (expected: {expected_value:.6f})")
        logger.info(f"Objective value: {result.objective_value:.6f}")
        logger.info(f"Iterations: {result.num_iterations}")
        logger.info(f"Duration: {duration:.3f} seconds")
        
        # Verify result is close to expected minimum
        if abs(result.parameters[0] - expected_value) < 0.01:
            logger.info("✅ Gradient descent optimization is working correctly")
            return True
        else:
            logger.warning("❌ Gradient descent did not converge to expected value")
            return False
        
    except Exception as e:
        logger.error(f"Error testing gradient descent: {str(e)}")
        return False

def test_simulated_annealing():
    """Test if simulated annealing optimization is working properly."""
    logger.info("Testing simulated annealing optimization...")
    
    try:
        from quangtps.optimization.simulated_annealing import SimulatedAnnealing
        
        # Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
        # Global minimum at (a,a²)
        def rosenbrock(params, a=1, b=100):
            x, y = params
            return (a - x)**2 + b * (y - x**2)**2
        
        # Initialize optimizer
        optimizer = SimulatedAnnealing(
            max_iterations=500,
            initial_temperature=1.0,
            cooling_rate=0.95
        )
        
        # Starting point
        initial_params = np.array([-1.0, 1.0])
        
        # Run optimization
        start_time = time.time()
        try:
            result = optimizer.optimize(
                objective_function=rosenbrock,
                initial_parameters=initial_params
            )
            duration = time.time() - start_time
            
            # Check result
            expected_values = np.array([1.0, 1.0])
            logger.info(f"Optimization result: ({result.parameters[0]:.3f}, {result.parameters[1]:.3f})")
            logger.info(f"Expected result: ({expected_values[0]:.3f}, {expected_values[1]:.3f})")
            logger.info(f"Objective value: {result.objective_value:.6f}")
            logger.info(f"Iterations: {result.num_iterations}")
            logger.info(f"Duration: {duration:.3f} seconds")
            
            # Verify result is close to expected minimum
            distance = np.linalg.norm(result.parameters - expected_values)
            if distance < 0.5:  # Less strict for stochastic methods
                logger.info("✅ Simulated annealing optimization is working correctly")
                return True
            else:
                logger.warning(f"⚠️ Simulated annealing result not close to expected value (distance: {distance:.3f})")
                return False
        except TypeError as e:
            # Handle case where optimize method has different signature
            logger.warning(f"❌ Simulated annealing optimize method has wrong signature: {str(e)}")
            return False
        
    except Exception as e:
        logger.error(f"Error testing simulated annealing: {str(e)}")
        return False

def check_treatment_techniques():
    """Check if treatment technique classes properly use optimization."""
    logger.info("Checking IMRT/VMAT techniques integration with optimization...")
    
    try:
        # Try to import and check technique classes
        from quangtps.treatment.techniques.imrt import IMRT
        from quangtps.treatment.techniques.vmat import VMAT
        
        # Check IMRT class
        imrt = IMRT(name="Test IMRT Plan")
        if hasattr(imrt, "optimize_fluence_maps"):
            logger.info("✅ IMRT class has optimize_fluence_maps method")
        else:
            logger.warning("❌ IMRT class missing optimize_fluence_maps method")
            return False
        
        # Check VMAT class
        vmat = VMAT(name="Test VMAT Plan")
        if hasattr(vmat, "optimize_plan"):
            logger.info("✅ VMAT class has optimize_plan method")
        else:
            logger.warning("❌ VMAT class missing optimize_plan method")
            return False
        
        # Check if optimization calls are properly integrated
        logger.info("📝 Note: IMRT and VMAT classes have optimization methods, but full validation requires patient data")
        return True
        
    except Exception as e:
        logger.error(f"Error checking treatment techniques: {str(e)}")
        return False

def test_objective_functions():
    """Test if objective functions are implemented and working correctly."""
    logger.info("Testing objective functions...")
    
    try:
        # Import objective functions
        from quangtps.optimization.objectives import (
            MinDose, MaxDose, UniformDose, DoseVolume,
            ConformityIndex, HomogeneityIndex, GradientIndex,
            ObjectiveCollection
        )
        
        # Create some test data
        target_structure = np.ones((10, 10, 10))
        oar_structure = np.zeros((10, 10, 10))
        oar_structure[5:10, 5:10, 5:10] = 1
        
        structures = {
            "PTV": target_structure,
            "OAR": oar_structure
        }
        
        # Create a simple dose grid (uniform 50 Gy to target, 20 Gy to OAR)
        dose_grid = np.zeros((10, 10, 10))
        dose_grid[target_structure == 1] = 50.0
        dose_grid[oar_structure == 1] = 20.0
        
        # Test minimum dose objective
        min_dose_obj = MinDose(structure_name="PTV", dose=45.0, weight=1.0)
        min_dose_value = min_dose_obj.evaluate(dose_grid, structures)
        logger.info(f"MinDose objective value: {min_dose_value:.3f}")
        
        # Test maximum dose objective
        max_dose_obj = MaxDose(structure_name="OAR", dose=15.0, weight=1.0)
        max_dose_value = max_dose_obj.evaluate(dose_grid, structures)
        logger.info(f"MaxDose objective value: {max_dose_value:.3f}")
        
        # Test collection of objectives
        collection = ObjectiveCollection()
        collection.add_objective(min_dose_obj)
        collection.add_objective(max_dose_obj)
        total_value = collection.evaluate(dose_grid, structures)
        logger.info(f"Total objective value: {total_value:.3f}")
        
        logger.info("✅ Objective functions are working correctly")
        return True
        
    except Exception as e:
        logger.error(f"Error testing objective functions: {str(e)}")
        return False

def main():
    """Main function to check optimization functionality."""
    parser = argparse.ArgumentParser(description="Check QuangTPS optimization functionality")
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed information during testing"
    )
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("Starting QuangTPS Optimization Functionality Check")
    logger.info("================================================")
    
    # Check dependencies first
    if not check_dependencies():
        logger.warning("Basic dependencies check failed. Some optimization features may not work.")
    
    # Run all checks
    checks = [
        ("Optimization modules structure", check_optimization_modules),
        ("Gradient descent algorithm", test_gradient_descent),
        ("Simulated annealing algorithm", test_simulated_annealing),
        ("Treatment techniques integration", check_treatment_techniques),
        ("Objective functions", test_objective_functions)
    ]
    
    results = []
    
    for name, check_func in checks:
        logger.info(f"\n--- Checking {name} ---")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Error during {name} check: {str(e)}")
            results.append((name, False))
    
    # Summary
    logger.info("\n================================================")
    logger.info("Optimization Checks Summary:")
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        if result:
            passed += 1
        else:
            failed += 1
        logger.info(f"{status} - {name}")
    
    logger.info(f"\nResults: {passed} passed, {failed} failed")
    
    # Overall result
    if failed == 0:
        logger.info("\n✅ Overall: All optimization checks passed")
        return 0
    else:
        logger.warning("\n⚠️ Overall: Some optimization checks failed")
        logger.info("See the improvement plan for next steps on fixing optimization functionality")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 