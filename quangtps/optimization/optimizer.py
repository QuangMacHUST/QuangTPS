#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Optimizer Module
==================

This module provides a higher-level interface for optimizing IMRT and VMAT treatment plans,
integrating with the underlying optimization engine to implement optimization algorithms.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import threading

# Import QuangTPS modules
try:
    from quangtps.planning.plan import Plan
    from quangtps.planning.beam_set import BeamSet
    from quangtps.planning.beam import Beam
    from quangtps.dose.dose_calculator import DoseCalculator
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.optimization.optimization_engine import (
        OptimizationEngine, OptimizationParameters, OptimizationStatus,
        OptimizationEvent, OptimizationResults, create_engine
    )
    from quangtps.optimization.objectives import (
        ObjectiveFunction, ObjectiveCollection, 
        DoseObjective, DVHObjective,
        LowerDoseObjective, UpperDoseObjective,
        MeanDoseObjective, MaxDoseObjective, MinDoseObjective,
        LowerDVHObjective, UpperDVHObjective,
        ConformityObjective, HomogeneityObjective
    )
    from quangtps.optimization.constraints import ConstraintCollection
except ImportError:
    logging.warning("Failed to import QuangTPS optimization modules")

logger = logging.getLogger(__name__)

class PlanOptimizer:
    """
    High-level interface for optimizing treatment plans.
    
    This class provides a simplified interface for using the optimization engine,
    handling the interaction between the treatment plan, dose calculation, and
    optimization process.
    """
    
    def __init__(self, plan, dose_calculator):
        """
        Initialize the plan optimizer.
        
        Parameters
        ----------
        plan : Plan
            The treatment plan to optimize
        dose_calculator : DoseCalculator
            The dose calculator to use for dose computation
        """
        self.plan = plan
        self.dose_calculator = dose_calculator
        self.objectives = ObjectiveCollection()
        self.constraints = ConstraintCollection()
        self.parameters = OptimizationParameters()
        self.engine = None
        self.result = None
        self.status = OptimizationStatus.READY
        self.stop_requested = False
        self.callback = None
        self.lock = threading.Lock()
    
    def set_objectives(self, objectives):
        """
        Set the optimization objectives.
        
        Parameters
        ----------
        objectives : list
            List of objective functions or OptimizationRow objects from the UI
        """
        self.objectives = ObjectiveCollection()
        
        # Convert objectives to appropriate format for the engine
        for obj in objectives:
            # Handle UI OptimizationRow objects
            if hasattr(obj, "objective_type") and hasattr(obj, "structure"):
                structure = obj.structure
                structure_id = obj.structure_id
                objective_type = obj.objective_type
                
                # Create appropriate objective based on type
                # The exact implementation depends on how ObjectiveFunction types are defined
                try:
                    if objective_type == 1 or "LOWER_DOSE" in str(objective_type):  # LOWER_DOSE
                        objective = LowerDoseObjective(
                            structure_id=structure_id,
                            structure_name=obj.structure_name,
                            dose=obj.dose,
                            weight=obj.weight,
                            priority=obj.priority
                        )
                    elif objective_type == 2 or "UPPER_DOSE" in str(objective_type):  # UPPER_DOSE
                        objective = UpperDoseObjective(
                            structure_id=structure_id,
                            structure_name=obj.structure_name,
                            dose=obj.dose,
                            weight=obj.weight,
                            priority=obj.priority
                        )
                    elif objective_type == 3 or "MEAN_DOSE" in str(objective_type):  # MEAN_DOSE
                        objective = MeanDoseObjective(
                            structure_id=structure_id,
                            structure_name=obj.structure_name,
                            dose=obj.dose,
                            weight=obj.weight,
                            priority=obj.priority
                        )
                    elif objective_type == 4 or "MAX_DOSE" in str(objective_type):  # MAX_DOSE
                        objective = MaxDoseObjective(
                            structure_id=structure_id,
                            structure_name=obj.structure_name,
                            dose=obj.dose,
                            weight=obj.weight,
                            priority=obj.priority
                        )
                    elif objective_type == 5 or "MIN_DOSE" in str(objective_type):  # MIN_DOSE
                        objective = MinDoseObjective(
                            structure_id=structure_id,
                            structure_name=obj.structure_name,
                            dose=obj.dose,
                            weight=obj.weight,
                            priority=obj.priority
                        )
                    elif objective_type == 6 or "LOWER_DVH" in str(objective_type):  # LOWER_DVH
                        objective = LowerDVHObjective(
                            structure_id=structure_id,
                            structure_name=obj.structure_name,
                            dose=obj.dose,
                            volume=obj.volume,
                            weight=obj.weight,
                            priority=obj.priority
                        )
                    elif objective_type == 7 or "UPPER_DVH" in str(objective_type):  # UPPER_DVH
                        objective = UpperDVHObjective(
                            structure_id=structure_id,
                            structure_name=obj.structure_name,
                            dose=obj.dose,
                            volume=obj.volume,
                            weight=obj.weight,
                            priority=obj.priority
                        )
                    elif objective_type == 8 or "CONFORMITY" in str(objective_type):  # CONFORMITY
                        objective = ConformityObjective(
                            structure_id=structure_id,
                            structure_name=obj.structure_name,
                            dose=obj.dose,
                            weight=obj.weight,
                            priority=obj.priority
                        )
                    elif objective_type == 9 or "HOMOGENEITY" in str(objective_type):  # HOMOGENEITY
                        objective = HomogeneityObjective(
                            structure_id=structure_id,
                            structure_name=obj.structure_name,
                            dose=obj.dose,
                            weight=obj.weight,
                            priority=obj.priority
                        )
                    else:
                        logger.warning(f"Unknown objective type: {objective_type}")
                        continue
                
                    self.objectives.add_objective(objective)
                    
                except Exception as e:
                    logger.error(f"Failed to create objective: {e}")
            
            # Handle ObjectiveFunction instances
            elif isinstance(obj, ObjectiveFunction):
                self.objectives.add_objective(obj)
            
            else:
                logger.warning(f"Unsupported objective type: {type(obj)}")
        
        logger.info(f"Set {len(self.objectives)} objectives for optimization")
    
    def set_parameters(self, **kwargs):
        """
        Set optimization parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for optimization parameters
        """
        for key, value in kwargs.items():
            if hasattr(self.parameters, key):
                setattr(self.parameters, key, value)
            else:
                logger.warning(f"Unknown optimization parameter: {key}")
    
    def run_optimization(self, max_iterations=100, convergence_tolerance=1e-4, 
                         mode="normal", callback=None):
        """
        Run the optimization process.
        
        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations
        convergence_tolerance : float, optional
            Convergence tolerance
        mode : str, optional
            Optimization mode ("normal", "fast", or "accurate")
        callback : callable, optional
            Callback function for progress updates
        
        Returns
        -------
        OptimizationResults
            The optimization results
        """
        # Set optimization parameters
        self.parameters.max_iterations = max_iterations
        self.parameters.convergence_threshold = convergence_tolerance
        
        # Adjust parameters based on mode
        if mode == "fast":
            self.parameters.learning_rate *= 1.5
            self.parameters.convergence_threshold *= 2.0
        elif mode == "accurate":
            self.parameters.learning_rate *= 0.8
            self.parameters.convergence_threshold *= 0.5
        
        # Store callback
        self.callback = callback
        
        # Reset stop flag
        self.stop_requested = False
        
        # Create engine
        solver_name = "lbfgs" if self.parameters.use_lbfgs else "gradient_descent"
        self.engine = create_engine(
            objectives=self.objectives,
            constraints=self.constraints,
            parameters=self.parameters,
            solver_name=solver_name
        )
        
        # Set initial state
        try:
            # Calculate initial dose
            initial_dose = self._calculate_initial_dose()
            
            # Extract structures
            structures = self._extract_structures()
            
            # Set initial state in engine
            self.engine.set_initial_state(
                dose_grid=initial_dose,
                structures=structures
            )
            
            # Register callbacks for progress updates
            self.engine.register_callback(
                OptimizationEvent.ITERATION_COMPLETED,
                self._on_iteration_completed
            )
            
            # Run optimization
            self.result = self.engine.optimize()
            
            # Update plan with optimized results
            self._update_plan_with_results()
            
            return self.result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self.status = OptimizationStatus.FAILED
            raise
    
    def stop_optimization(self):
        """Stop the optimization process."""
        self.stop_requested = True
        if self.engine:
            self.engine.stop()
    
    def _calculate_initial_dose(self):
        """
        Calculate the initial dose distribution.
        
        Returns
        -------
        DoseGrid
            The initial dose distribution
        """
        try:
            # Calculate dose using the dose calculator
            dose = self.dose_calculator.calculate_dose(
                beams=self.plan.beam_set.beams,
                image=self.plan.image if hasattr(self.plan, 'image') else None,
                structure_set=self.plan.structure_set if hasattr(self.plan, 'structure_set') else None
            )
            
            # Convert to DoseGrid if necessary
            if not isinstance(dose, DoseGrid):
                # Create a DoseGrid from the numpy array
                if hasattr(self.plan, 'image') and self.plan.image:
                    spacing = self.plan.image.spacing
                    origin = self.plan.image.origin
                else:
                    spacing = (1.0, 1.0, 1.0)
                    origin = (0.0, 0.0, 0.0)
                
                dose_grid = DoseGrid(
                    dose_array=dose,
                    spacing=spacing,
                    origin=origin
                )
                return dose_grid
            
            return dose
            
        except Exception as e:
            logger.error(f"Failed to calculate initial dose: {e}")
            # Return dummy dose grid
            return DoseGrid(
                dose_array=np.zeros((10, 10, 10)),
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0)
            )
    
    def _extract_structures(self):
        """
        Extract structure masks from the plan.
        
        Returns
        -------
        dict
            Dictionary mapping structure IDs to masks
        """
        structures = {}
        
        if hasattr(self.plan, 'structure_set') and self.plan.structure_set:
            for structure in self.plan.structure_set.structures:
                if hasattr(structure, 'get_mask'):
                    mask = structure.get_mask()
                    if mask is not None:
                        structures[structure.id] = mask
        
        return structures
    
    def _on_iteration_completed(self, context):
        """
        Handle iteration completed event.
        
        Parameters
        ----------
        context : dict
            Event context with iteration information
        """
        iteration = context.get('iteration', 0)
        cost = context.get('cost', float('inf'))
        objective_values = context.get('objective_values', {})
        elapsed_time = context.get('elapsed_time', 0.0)
        
        # Update status
        self.status = context.get('status', OptimizationStatus.RUNNING)
        
        # Call callback if provided
        if self.callback:
            try:
                self.callback(iteration, cost, objective_values)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
        
        # Check if optimization should be stopped
        if self.stop_requested:
            return False  # Stop optimization
        
        return True  # Continue optimization
    
    def _update_plan_with_results(self):
        """Update the plan with the optimization results."""
        if not self.result:
            return
        
        try:
            # Update dose in plan
            if hasattr(self.plan, 'set_dose') and self.result.final_dose_grid:
                self.plan.set_dose(self.result.final_dose_grid)
            
            # The rest depends on how the plan and beams store their parameters
            # This might involve setting MLC positions, jaw positions, etc.
            
            logger.info("Updated plan with optimization results")
            
        except Exception as e:
            logger.error(f"Failed to update plan with results: {e}")
    
    def get_optimization_status(self):
        """
        Get the current optimization status.
        
        Returns
        -------
        tuple
            (status, iteration, cost, elapsed_time)
        """
        if self.engine:
            status = self.engine.status
            iteration = self.engine.current_iteration
            cost = self.engine.current_objective_value
            elapsed_time = self.engine.elapsed_time
        else:
            status = self.status
            iteration = 0
            cost = float('inf')
            elapsed_time = 0.0
        
        return (status, iteration, cost, elapsed_time)
    
    def generate_report(self):
        """
        Generate a report of the optimization results.
        
        Returns
        -------
        dict
            Report data
        """
        if not self.result:
            return {"status": "No optimization results available"}
        
        # Get summary from results
        summary = self.result.get_summary()
        
        # Add more plan-specific information
        report = {
            **summary,
            "plan_id": getattr(self.plan, 'id', 'unknown'),
            "plan_name": getattr(self.plan, 'name', 'Unknown Plan'),
            "num_beams": len(self.plan.beam_set.beams) if hasattr(self.plan, 'beam_set') else 0,
            "num_objectives": len(self.objectives) if self.objectives else 0,
            "num_constraints": len(self.constraints) if self.constraints else 0,
            "optimization_mode": getattr(self.parameters, 'solver_name', 'unknown')
        }
        
        return report

def test_plan_optimizer():
    """Test function for the plan optimizer."""
    # This would be implemented with dummy test objects
    # similar to the test in optimizer.py, but integrated with
    # the optimization engine
    pass

if __name__ == "__main__":
    test_plan_optimizer() 