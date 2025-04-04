#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-criteria optimization (MCO) engine.

This module provides the backend functionality for multi-criteria optimization,
including generation of Pareto-optimal plans, interpolation between plans,
and navigation of the Pareto surface.
"""

import os
import json
import time
import logging
import copy
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from quangtps.core.types import Plan, DoseGrid, Treatment, Structure
from quangtps.planning.optimization import PlanOptimizer
from quangtps.optimization.objectives import Objective, ObjectiveResult, ObjectiveFunction, ObjectiveType
from quangtps.optimization.constraints import Constraint
from quangtps.optimization.optimizer_factory import create_optimizer
from quangtps.treatment.techniques.imrt import IMRTTreatment
from quangtps.treatment.techniques.vmat import VMATTreatment
from quangtps.core.logging import get_logger
from quangtps.core.patient import Patient
from quangtps.evaluation.dvh.dvh_calculation import DVHCalculator, calculate_dvh_metrics
from quangtps.planning.plan import Plan
from quangtps.optimization.optimizers.optimizer_factory import OptimizerFactory
from quangtps.dose.dose_calculation import DoseEngine
from quangtps.optimization.optimizer import Optimizer, OptimizationParameters
from quangtps.optimization.objectives.objective_factory import ObjectiveFactory

logger = get_logger(__name__)

@dataclass
class MCOTradeoffObjective:
    """
    Defines an objective for MCO trade-off exploration.
    
    This represents a single objective function that can be used in multi-criteria
    optimization to generate and navigate the Pareto front.
    """
    objective_id: str
    structure_id: str
    structure_name: str
    objective_type: str  # e.g., "MinDose", "MaxDose", "MeanDose", "DoseAtVolume", "VolumeAtDose"
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Priority (1-3, with 1 being highest)
    weight: float = 1.0  # Current weight for weighted-sum optimization
    value: float = 0.0  # Current value of this objective 
    min_value: float = 0.0  # Minimum achievable value
    max_value: float = 0.0  # Maximum achievable value
    is_active: bool = True  # Whether this objective is active in the current trade-off
    
    def to_optimizer_objective(self) -> Objective:
        """Convert to an optimizer objective."""
        factory = ObjectiveFactory()
        return factory.create_objective(
            objective_type=self.objective_type,
            structure_id=self.structure_id,
            parameters=self.parameters,
            weight=self.weight
        )
    
    def calculate_value(self, plan: Plan) -> float:
        """
        Calculate the current value of this objective for a given plan.
        
        Args:
            plan: The plan to evaluate
            
        Returns:
            The current value of this objective
        """
        structure = None
        for s in plan.structure_set.structures:
            if s.id == self.structure_id:
                structure = s
                break
        
        if not structure:
            logger.warning(f"Structure {self.structure_id} not found for objective {self.objective_id}")
            return 0.0
        
        metrics = calculate_dvh_metrics(structure, plan.dose)
        
        if self.objective_type == "MinDose":
            return metrics.get("min_dose", 0.0)
        elif self.objective_type == "MaxDose":
            return metrics.get("max_dose", 0.0)
        elif self.objective_type == "MeanDose":
            return metrics.get("mean_dose", 0.0)
        elif self.objective_type == "DoseAtVolume":
            volume = self.parameters.get("volume", 50)
            return metrics.get(f"D{volume}", 0.0)
        elif self.objective_type == "VolumeAtDose":
            dose = self.parameters.get("dose", 50)
            return metrics.get(f"V{dose}", 0.0)
        else:
            logger.warning(f"Unknown objective type {self.objective_type} for {self.objective_id}")
            return 0.0
    
    def update_value(self, plan: Plan):
        """Update the value of this objective based on the given plan."""
        self.value = self.calculate_value(plan)


@dataclass
class MCOSolution:
    """
    Represents a single solution in the MCO Pareto front.
    
    This includes the plan, objective values, and weights used to generate this solution.
    """
    plan: Plan
    objective_values: Dict[str, float]  # objective_id -> value
    weights: Dict[str, float]  # objective_id -> weight
    solution_id: str = ""  # Unique identifier for this solution
    
    def __post_init__(self):
        if not self.solution_id:
            self.solution_id = f"sol_{id(self)}"


class MCOEngine:
    """
    Multi-Criteria Optimization (MCO) engine.
    
    This class provides the capability to generate and navigate Pareto-optimal
    treatment plans using multi-criteria optimization, similar to Eclipse's MCO.
    """
    
    def __init__(self, base_plan: Plan, optimizer: Optimizer):
        """
        Initialize the MCO engine.
        
        Args:
            base_plan: The base plan to start optimization from
            optimizer: The optimizer to use for generating plans
        """
        self.base_plan = base_plan
        self.optimizer = optimizer
        
        # Define objectives
        self.objectives: Dict[str, MCOTradeoffObjective] = {}
        
        # Pareto front
        self.solutions: Dict[str, MCOSolution] = {}
        self.anchor_solutions: Dict[str, MCOSolution] = {}  # objective_id -> solution
        
        # Current solution
        self.current_solution: Optional[MCOSolution] = None
        
        # Status
        self.generation_status = "Not Started"
        self.is_generating = False
        self.generation_progress = 0.0
        
        # Callbacks
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        self.solution_callback: Optional[Callable[[MCOSolution], None]] = None
    
    def add_objective(self, objective: MCOTradeoffObjective):
        """
        Add an objective to the MCO problem.
        
        Args:
            objective: The objective to add
        """
        self.objectives[objective.objective_id] = objective
    
    def add_objective_from_params(
        self, 
        structure_id: str,
        structure_name: str,
        objective_type: str,
        parameters: Dict[str, Any] = None,
        priority: int = 1
    ) -> str:
        """
        Add an objective to the MCO problem from parameters.
        
        Args:
            structure_id: ID of the structure
            structure_name: Name of the structure
            objective_type: Type of objective
            parameters: Parameters for the objective
            priority: Priority level
            
        Returns:
            ID of the added objective
        """
        objective_id = f"{objective_type}_{structure_id}_{id(self)}"
        
        objective = MCOTradeoffObjective(
            objective_id=objective_id,
            structure_id=structure_id,
            structure_name=structure_name,
            objective_type=objective_type,
            parameters=parameters or {},
            priority=priority
        )
        
        self.add_objective(objective)
        return objective_id
    
    def remove_objective(self, objective_id: str) -> bool:
        """
        Remove an objective from the MCO problem.
        
        Args:
            objective_id: ID of the objective to remove
            
        Returns:
            True if the objective was removed, False otherwise
        """
        if objective_id in self.objectives:
            del self.objectives[objective_id]
            return True
        return False
    
    def set_objective_weight(self, objective_id: str, weight: float):
        """
        Set the weight of an objective.
        
        Args:
            objective_id: ID of the objective
            weight: New weight
        """
        if objective_id in self.objectives:
            self.objectives[objective_id].weight = weight
    
    def generate_anchor_plans(self, callback: Callable[[float, str], None] = None):
        """
        Generate anchor plans for each objective.
        
        An anchor plan is a plan that optimizes a single objective to its extreme,
        representing a corner of the Pareto front.
        
        Args:
            callback: Callback function for progress updates
        """
        self.is_generating = True
        self.generation_status = "Generating anchor plans"
        self.generation_progress = 0.0
        
        if callback:
            self.progress_callback = callback
        
        # For each objective, generate an anchor plan
        for i, (objective_id, objective) in enumerate(self.objectives.items()):
            if not objective.is_active:
                continue
                
            # Update progress
            progress = i / len(self.objectives)
            self.generation_progress = progress
            if self.progress_callback:
                self.progress_callback(progress, f"Generating anchor plan for {objective.structure_name}")
            
            # Create optimization parameters with only this objective
            opt_params = OptimizationParameters()
            
            for other_id, other_obj in self.objectives.items():
                if other_id == objective_id:
                    # Set very high weight for this objective
                    other_obj.weight = 1000.0
                else:
                    # Set very low weight for other objectives
                    other_obj.weight = 0.001
                
                # Add to optimizer
                opt_params.add_objective(other_obj.to_optimizer_objective())
            
            # Run optimization
            anchor_plan = self.optimizer.optimize(self.base_plan, opt_params)
            
            # Update objective values
            objective_values = {}
            for obj_id, obj in self.objectives.items():
                value = obj.calculate_value(anchor_plan)
                objective_values[obj_id] = value
                obj.update_value(anchor_plan)
                
                # Update min/max values
                if obj_id == objective_id:
                    obj.min_value = value  # This is the best possible value for this objective
            
            # Create solution
            weights = {obj_id: obj.weight for obj_id, obj in self.objectives.items()}
            solution = MCOSolution(
                plan=anchor_plan,
                objective_values=objective_values,
                weights=weights,
                solution_id=f"anchor_{objective_id}"
            )
            
            # Store as anchor solution
            self.anchor_solutions[objective_id] = solution
            self.solutions[solution.solution_id] = solution
        
        # Find max values for each objective
        for objective_id, objective in self.objectives.items():
            max_value = float('-inf')
            for anchor_id, anchor in self.anchor_solutions.items():
                if anchor_id != objective_id:  # Look at other anchor solutions
                    value = anchor.objective_values.get(objective_id, 0.0)
                    max_value = max(max_value, value)
            
            if max_value != float('-inf'):
                objective.max_value = max_value
        
        # Generate balanced solution
        self.generation_progress = 0.95
        if self.progress_callback:
            self.progress_callback(0.95, "Generating balanced solution")
        
        balanced_plan = self.generate_balanced_plan()
        
        self.generation_status = "Generation complete"
        self.generation_progress = 1.0
        if self.progress_callback:
            self.progress_callback(1.0, "Generation complete")
        
        self.is_generating = False
        self.current_solution = self.solutions.get("balanced", None)
    
    def generate_balanced_plan(self) -> Plan:
        """
        Generate a balanced plan that represents a compromise between all objectives.
        
        Returns:
            The balanced plan
        """
        # Reset weights to balanced values
        for objective in self.objectives.values():
            objective.weight = 1.0
        
        # Create optimization parameters
        opt_params = OptimizationParameters()
        for objective in self.objectives.values():
            if objective.is_active:
                opt_params.add_objective(objective.to_optimizer_objective())
        
        # Run optimization
        balanced_plan = self.optimizer.optimize(self.base_plan, opt_params)
        
        # Calculate objective values
        objective_values = {}
        for obj_id, obj in self.objectives.items():
            value = obj.calculate_value(balanced_plan)
            objective_values[obj_id] = value
        
        # Create solution
        weights = {obj_id: obj.weight for obj_id, obj in self.objectives.items()}
        solution = MCOSolution(
            plan=balanced_plan,
            objective_values=objective_values,
            weights=weights,
            solution_id="balanced"
        )
        
        # Store as solution
        self.solutions["balanced"] = solution
        
        return balanced_plan
    
    def generate_navigated_plan(self, weights: Dict[str, float]) -> MCOSolution:
        """
        Generate a navigated plan based on the given weights.
        
        Args:
            weights: Dict mapping objective_id to weight
            
        Returns:
            The generated solution
        """
        # Update weights
        for objective_id, weight in weights.items():
            if objective_id in self.objectives:
                self.objectives[objective_id].weight = weight
        
        # Create optimization parameters
        opt_params = OptimizationParameters()
        for objective in self.objectives.values():
            if objective.is_active:
                opt_params.add_objective(objective.to_optimizer_objective())
        
        # Run optimization
        navigated_plan = self.optimizer.optimize(self.base_plan, opt_params)
        
        # Calculate objective values
        objective_values = {}
        for obj_id, obj in self.objectives.items():
            value = obj.calculate_value(navigated_plan)
            objective_values[obj_id] = value
        
        # Create solution
        solution = MCOSolution(
            plan=navigated_plan,
            objective_values=objective_values,
            weights=weights.copy(),
            solution_id=f"nav_{int(time.time())}"
        )
        
        # Store as solution
        self.solutions[solution.solution_id] = solution
        self.current_solution = solution
        
        # Call solution callback if provided
        if self.solution_callback:
            self.solution_callback(solution)
        
        return solution
    
    def get_trade_off_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the range of possible values for each objective.
        
        Returns:
            Dict mapping objective_id to (min_value, max_value)
        """
        return {obj_id: (obj.min_value, obj.max_value) for obj_id, obj in self.objectives.items()}
    
    def save_current_solution(self) -> bool:
        """
        Save the current solution in the navigation history.
        
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.current_solution:
            return False
            
        # Generate a new solution ID
        solution_id = f"saved_{int(time.time())}"
        
        # Create a copy of the current solution with the new ID
        solution = MCOSolution(
            plan=self.current_solution.plan,
            objective_values=self.current_solution.objective_values.copy(),
            weights=self.current_solution.weights.copy(),
            solution_id=solution_id
        )
        
        # Store as solution
        self.solutions[solution_id] = solution
        
        return True
    
    def load_solution(self, solution_id: str) -> bool:
        """
        Load a solution from the navigation history.
        
        Args:
            solution_id: ID of the solution to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if solution_id not in self.solutions:
            return False
            
        self.current_solution = self.solutions[solution_id]
        
        # Update weights
        for objective_id, weight in self.current_solution.weights.items():
            if objective_id in self.objectives:
                self.objectives[objective_id].weight = weight
        
        # Call solution callback if provided
        if self.solution_callback and self.current_solution:
            self.solution_callback(self.current_solution)
        
        return True


def create_mco_engine(base_plan: Plan, optimizer: Optimizer) -> MCOEngine:
    """
    Create and initialize an MCO engine for a given plan.
    
    Args:
        base_plan: Base plan to start from
        optimizer: Optimizer to use
        
    Returns:
        Initialized MCO engine
    """
    engine = MCOEngine(base_plan, optimizer)
    
    # Add common clinical objectives
    if base_plan.structure_set:
        # Find targets
        targets = []
        oars = []
        
        for structure in base_plan.structure_set.structures:
            if structure.structure_type == "PTV" or "PTV" in structure.name:
                targets.append(structure)
            elif structure.structure_type == "OAR" or any(oar in structure.name.upper() for oar in 
                             ["SPINAL", "CORD", "HEART", "LUNG", "LIVER", "KIDNEY", "BOWEL", "BLADDER", "RECTUM"]):
                oars.append(structure)
        
        # Add objectives for targets
        for target in targets:
            # Minimum dose to target
            engine.add_objective_from_params(
                structure_id=target.id,
                structure_name=target.name,
                objective_type="MinDose",
                priority=1
            )
            
            # Maximum dose to target
            engine.add_objective_from_params(
                structure_id=target.id,
                structure_name=target.name,
                objective_type="MaxDose",
                priority=1
            )
            
            # Dose homogeneity
            engine.add_objective_from_params(
                structure_id=target.id,
                structure_name=target.name,
                objective_type="DoseAtVolume",
                parameters={"volume": 95},
                priority=1
            )
        
        # Add objectives for OARs
        for oar in oars:
            # Maximum dose to OAR
            engine.add_objective_from_params(
                structure_id=oar.id,
                structure_name=oar.name,
                objective_type="MaxDose",
                priority=2
            )
            
            # Mean dose to OAR
            engine.add_objective_from_params(
                structure_id=oar.id,
                structure_name=oar.name,
                objective_type="MeanDose",
                priority=2
            )
    
    return engine


if __name__ == "__main__":
    # Test code
    import sys
    import matplotlib.pyplot as plt
    
    # Create a dummy plan and objectives
    class DummyPlan:
        def __init__(self):
            self.name = "Test Plan"
            self.id = "test_plan_1"
        
        def clone(self):
            return DummyPlan()
    
    class DummyObjective:
        def __init__(self, name):
            self.name = name
        
        def evaluate(self, fluence):
            # Return a dummy result
            return 0.5
    
    # Create a test engine
    plan = DummyPlan()
    objectives = {
        'ptv_coverage': DummyObjective('ptv_coverage'),
        'oar_sparing': DummyObjective('oar_sparing')
    }
    
    engine = MCOEngine()
    
    # Test saving and loading
    engine.solutions = {
        "Solution_1": MCOSolution(id="Solution_1", objective_values={'ptv_coverage': 1.0, 'oar_sparing': 0.0}, plan=plan, weight_vector={'ptv_coverage': 1.0, 'oar_sparing': 0.0}),
        "Solution_2": MCOSolution(id="Solution_2", objective_values={'ptv_coverage': 0.0, 'oar_sparing': 1.0}, plan=plan, weight_vector={'ptv_coverage': 0.0, 'oar_sparing': 1.0})
    }
    
    engine.save_current_solution("Test Solution 1")
    print("Saved current solution")
    
    engine.current_solution_id = None
    engine.load_solutions("test_solutions.json")
    print(f"Loaded {len(engine.solutions)} solutions") 