"""
Robust optimization implementation for handling setup and range uncertainties.

This module provides classes and functions for robust treatment plan optimization,
which takes into account positional setup uncertainties and range uncertainties
for particle therapy to ensure treatment plans are resilient to various uncertainties.
"""

import os
import time
import logging
import numpy as np
import scipy.optimize
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

from quangtps.treatment.plan import Plan
from quangtps.treatment.beams import Beam
from quangtps.treatment.patient import Patient
from quangtps.segmentation.structures import Structure
from quangtps.dose import Dose
from quangtps.imaging import Image
from quangtps.dose.calculation import DoseCalculator
from quangtps.objectives import ObjectiveFunction, PlanningObjectives
from quangtps.constraints import ConstraintFunction
from quangtps.core.logging import get_logger
from quangtps.core.types import DoseGrid
from quangtps.core.services import ServiceManager
from quangtps.planning.beam import Beam
from quangtps.optimization.methods.objective_based import ObjectiveBasedOptimizer

# Configure logging
logger = get_logger(__name__)

@dataclass
class UncertaintyScenario:
    """
    Represents a scenario for robust optimization.
    
    Each scenario contains shift parameters and weight for optimization.
    """
    name: str
    parameters: Dict[str, float]
    weight: float = 1.0
    dose_grid: Optional[DoseGrid] = None
    
    def is_nominal(self) -> bool:
        """Return True if this is the nominal scenario (no shifts)."""
        return all(abs(v) < 1e-6 for v in self.parameters.values())
    
    def __str__(self) -> str:
        """String representation of the scenario."""
        param_strs = []
        for k, v in self.parameters.items():
            if k.startswith('shift_'):
                axis = k.split('_')[1].upper()
                param_strs.append(f"{axis}{'+' if v > 0 else ''}{v:.1f}mm")
            elif k == 'range':
                param_strs.append(f"Range{'+' if v > 0 else ''}{v:.1f}%")
        
        return f"{self.name}: {', '.join(param_strs)} (w={self.weight:.2f})"


class RobustOptimizer:
    """
    Optimizer that accounts for setup and range uncertainties.
    
    This optimizer creates multiple scenarios with different uncertainty parameters
    and optimizes the plan to be robust against all these scenarios.
    """
    
    def __init__(self, plan: Plan, objectives: PlanningObjectives, dose_calculator: DoseCalculator):
        """
        Initialize robust optimizer.
        
        Parameters
        ----------
        plan : Plan
            The treatment plan to optimize
        objectives : PlanningObjectives
            Planning objectives
        dose_calculator : DoseCalculator
            Dose calculation engine
        """
        self.plan = plan
        self.objectives = objectives
        self.dose_calculator = dose_calculator
        
        # Default parameters
        self.parameters = {
            'max_iterations': 100,
            'convergence_threshold': 0.001,
            'learning_rate': 0.1,
            'setup_uncertainty': 3.0,  # mm
            'range_uncertainty': 3.5,  # percent
            'scenario_sampling': 'corners',  # 'corners', 'random', 'hybrid'
            'scenario_count': 9,  # for random sampling
            'nominal_weight': 2.0,  # weight for nominal scenario
            'voxel_sampling': 0.3,  # fraction of voxels to use (speedup)
            'worst_case': False,  # use worst case (min/max) over all scenarios
            'iteration_callback': None
        }
        
        # Scenarios
        self.scenarios = []
        self.nominal_scenario = UncertaintyScenario(
            name="Nominal",
            parameters={},
            weight=self.parameters['nominal_weight']
        )
        
        # Optimization state
        self.current_iteration = 0
        self.objective_values = []
        self.best_value = float('inf')
        self.best_fluence = None
        self.convergence_count = 0
        
        # Register with service manager for progress tracking
        self.service_manager = ServiceManager()
        
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set optimizer parameter.
        
        Parameters
        ----------
        name : str
            Parameter name
        value : Any
            Parameter value
        """
        if name in self.parameters:
            self.parameters[name] = value
            
            # Special case for setup and range uncertainty
            if name == 'setup_uncertainty' or name == 'range_uncertainty':
                # Clear scenarios to force re-generation
                self.scenarios = []
        else:
            logger.warning(f"Unknown parameter: {name}")
    
    def generate_standard_scenarios(self, setup_uncertainty: float = None, range_uncertainty: float = None) -> None:
        """
        Generate standard uncertainty scenarios.
        
        Parameters
        ----------
        setup_uncertainty : float, optional
            Setup uncertainty magnitude in mm
        range_uncertainty : float, optional
            Range uncertainty magnitude in percent
        """
        if setup_uncertainty is not None:
            self.parameters['setup_uncertainty'] = setup_uncertainty
        
        if range_uncertainty is not None:
            self.parameters['range_uncertainty'] = range_uncertainty
            
        setup_unc = self.parameters['setup_uncertainty']
        range_unc = self.parameters['range_uncertainty']
        sampling = self.parameters['scenario_sampling']
        
        self.scenarios = []
        
        # Add nominal scenario
        self.nominal_scenario = UncertaintyScenario(
            name="Nominal",
            parameters={},
            weight=self.parameters['nominal_weight']
        )
        
        # Different sampling strategies
        if sampling == 'corners':
            # Systematic shifts - corners of the uncertainty space
            for x_shift in [-setup_unc, 0, setup_unc]:
                for y_shift in [-setup_unc, 0, setup_unc]:
                    for z_shift in [-setup_unc, 0, setup_unc]:
                        # Skip nominal (already added)
                        if x_shift == 0 and y_shift == 0 and z_shift == 0:
                            continue
                            
                        scenario_params = {}
                        if x_shift != 0:
                            scenario_params['shift_x'] = x_shift
                        if y_shift != 0:
                            scenario_params['shift_y'] = y_shift
                        if z_shift != 0:
                            scenario_params['shift_z'] = z_shift
                            
                        # Only add range uncertainty for particle therapy
                        use_range = False
                        for beam in self.plan.beams:
                            if beam.is_particle_beam():
                                use_range = True
                                break
                                
                        if use_range:
                            for r_shift in [-range_unc, range_unc]:
                                r_params = scenario_params.copy()
                                r_params['range'] = r_shift
                                name = self._make_scenario_name(r_params)
                                self.scenarios.append(UncertaintyScenario(
                                    name=name,
                                    parameters=r_params,
                                    weight=1.0
                                ))
                        else:
                            name = self._make_scenario_name(scenario_params)
                            self.scenarios.append(UncertaintyScenario(
                                name=name,
                                parameters=scenario_params,
                                weight=1.0
                            ))
                            
        elif sampling == 'random':
            # Random sampling of the uncertainty space
            n_scenarios = self.parameters['scenario_count']
            rng = np.random.default_rng(42)  # Seed for reproducibility
            
            for i in range(n_scenarios):
                x_shift = rng.uniform(-setup_unc, setup_unc)
                y_shift = rng.uniform(-setup_unc, setup_unc)
                z_shift = rng.uniform(-setup_unc, setup_unc)
                
                scenario_params = {
                    'shift_x': x_shift,
                    'shift_y': y_shift,
                    'shift_z': z_shift
                }
                
                # Only add range uncertainty for particle therapy
                use_range = False
                for beam in self.plan.beams:
                    if beam.is_particle_beam():
                        use_range = True
                        break
                        
                if use_range:
                    r_shift = rng.uniform(-range_unc, range_unc)
                    scenario_params['range'] = r_shift
                    
                name = f"Random_{i+1}"
                self.scenarios.append(UncertaintyScenario(
                    name=name,
                    parameters=scenario_params,
                    weight=1.0
                ))
                
        elif sampling == 'hybrid':
            # Combination of corners and random sampling
            
            # Add corners for setup uncertainty
            for x_shift in [-setup_unc, setup_unc]:
                for y_shift in [-setup_unc, setup_unc]:
                    for z_shift in [-setup_unc, setup_unc]:
                        scenario_params = {
                            'shift_x': x_shift,
                            'shift_y': y_shift,
                            'shift_z': z_shift
                        }
                        name = self._make_scenario_name(scenario_params)
                        self.scenarios.append(UncertaintyScenario(
                            name=name,
                            parameters=scenario_params,
                            weight=1.0
                        ))
            
            # Only add range uncertainty for particle therapy
            use_range = False
            for beam in self.plan.beams:
                if beam.is_particle_beam():
                    use_range = True
                    break
                    
            if use_range:
                # Add range uncertainty scenarios
                for r_shift in [-range_unc, range_unc]:
                    scenario_params = {'range': r_shift}
                    name = f"Range{'+' if r_shift > 0 else ''}{r_shift:.1f}%"
                    self.scenarios.append(UncertaintyScenario(
                        name=name,
                        parameters=scenario_params,
                        weight=1.0
                    ))
                    
        logger.info(f"Generated {len(self.scenarios)} uncertainty scenarios")
        for i, scenario in enumerate(self.scenarios):
            logger.debug(f"Scenario {i+1}: {scenario}")
            
    def _make_scenario_name(self, params: Dict[str, float]) -> str:
        """Generate a descriptive name for a scenario."""
        parts = []
        
        if 'shift_x' in params:
            parts.append(f"X{'+' if params['shift_x'] > 0 else ''}{params['shift_x']:.1f}")
        if 'shift_y' in params:
            parts.append(f"Y{'+' if params['shift_y'] > 0 else ''}{params['shift_y']:.1f}")
        if 'shift_z' in params:
            parts.append(f"Z{'+' if params['shift_z'] > 0 else ''}{params['shift_z']:.1f}")
        if 'range' in params:
            parts.append(f"R{'+' if params['range'] > 0 else ''}{params['range']:.1f}%")
            
        return "_".join(parts)
    
    def calculate_scenario_doses(self) -> Dict[str, DoseGrid]:
        """
        Calculate doses for all scenarios.
        
        Returns
        -------
        Dict[str, DoseGrid]
            Dictionary mapping scenario names to dose grids
        """
        # Calculate dose for nominal scenario
        logger.info("Calculating dose for nominal scenario")
        try:
            nominal_dose = self.dose_calculator.calculate_dose(self.plan)
            self.nominal_scenario.dose_grid = nominal_dose
        except Exception as e:
            logger.error(f"Error calculating nominal dose: {e}")
            return {}
            
        # Calculate doses for all other scenarios
        scenario_doses = {self.nominal_scenario.name: self.nominal_scenario.dose_grid}
        
        for scenario in self.scenarios:
            logger.info(f"Calculating dose for scenario: {scenario}")
            
            # Create a modified plan with shifted beams
            modified_plan = self._create_scenario_plan(self.plan, scenario.parameters)
            
            try:
                # Calculate dose
                scenario_dose = self.dose_calculator.calculate_dose(modified_plan)
                scenario.dose_grid = scenario_dose
                scenario_doses[scenario.name] = scenario_dose
            except Exception as e:
                logger.error(f"Error calculating dose for scenario {scenario.name}: {e}")
                
        return scenario_doses
    
    def _create_scenario_plan(self, original_plan: Plan, scenario_params: Dict[str, float]) -> Plan:
        """
        Create a modified plan for an uncertainty scenario.
        
        Parameters
        ----------
        original_plan : Plan
            The original treatment plan
        scenario_params : Dict[str, float]
            Scenario parameters (shifts and range)
            
        Returns
        -------
        Plan
            The modified plan
        """
        # Create a deep copy of the plan
        modified_plan = Plan(
            plan_id=original_plan.plan_id,
            plan_name=original_plan.name,
            patient_id=original_plan.patient_id
        )
        
        # Copy attributes
        modified_plan.description = original_plan.description
        modified_plan.status = original_plan.status
        modified_plan.plan_type = original_plan.plan_type
        modified_plan.prescription = original_plan.prescription
        modified_plan.dose_grid = original_plan.dose_grid
        
        # Create shifted beams
        x_shift = scenario_params.get('shift_x', 0)
        y_shift = scenario_params.get('shift_y', 0)
        z_shift = scenario_params.get('shift_z', 0)
        range_shift = scenario_params.get('range', 0)
        
        for beam in original_plan.beams:
            # Create a copy of the beam
            shifted_beam = beam.copy()
            
            # Apply isocenter shifts (opposite direction)
            if x_shift != 0 or y_shift != 0 or z_shift != 0:
                shifted_beam.shift_isocenter(-x_shift, -y_shift, -z_shift)
                
            # Apply range shift for particle beams
            if range_shift != 0 and beam.is_particle_beam():
                shifted_beam.apply_range_shift(range_shift)
                
            # Add to modified plan
            modified_plan.add_beam(shifted_beam)
            
        return modified_plan
        
    def optimize(self) -> Dict:
        """
        Run robust optimization.
        
        Returns
        -------
        Dict
            Optimization results
        """
        # Generate scenarios if not already done
        if not self.scenarios:
            self.generate_standard_scenarios()
            
        # Initialize progress tracking
        self.service_manager.publish('optimization_started', {
            'plan_id': self.plan.plan_id,
            'max_iterations': self.parameters['max_iterations']
        })
        
        # Calculate doses for all scenarios
        scenario_doses = self.calculate_scenario_doses()
        if not scenario_doses:
            logger.error("No scenario doses calculated, cannot optimize")
            return {'success': False, 'error': 'No scenario doses calculated'}
            
        # Create base optimizer for each scenario
        scenario_optimizers = {}
        
        # Nominal scenario optimizer
        nominal_opt = ObjectiveBasedOptimizer(self.plan, self.objectives, self.dose_calculator)
        nominal_opt.set_parameter('max_iterations', self.parameters['max_iterations'])
        nominal_opt.set_parameter('convergence_threshold', self.parameters['convergence_threshold'])
        nominal_opt.set_parameter('learning_rate', self.parameters['learning_rate'])
        
        # Initialize optimization variables
        best_objective_value = float('inf')
        best_fluence_map = None
        convergence_count = 0
        
        # Main optimization loop
        for iteration in range(self.parameters['max_iterations']):
            self.current_iteration = iteration
            
            # Run one iteration of optimization for nominal scenario
            nominal_result = nominal_opt.optimize_step()
            
            # Update fluence maps for all scenarios
            for scenario_name, dose_grid in scenario_doses.items():
                if scenario_name == self.nominal_scenario.name:
                    continue
                    
                # Create scenario-specific optimizer and update fluence
                if scenario_name not in scenario_optimizers:
                    scenario_plan = self._create_scenario_plan(
                        self.plan, 
                        next(s.parameters for s in self.scenarios if s.name == scenario_name)
                    )
                    scenario_opt = ObjectiveBasedOptimizer(scenario_plan, self.objectives, self.dose_calculator)
                    scenario_optimizers[scenario_name] = scenario_opt
                
                # Apply nominal fluence to this scenario
                scenario_optimizers[scenario_name].apply_fluence(nominal_opt.get_current_fluence())
                
            # Calculate composite objective value across all scenarios
            composite_value = self._calculate_composite_objective(
                nominal_opt, scenario_optimizers
            )
            
            # Store objective value
            self.objective_values.append(composite_value)
            
            # Check if this is the best result so far
            if composite_value < best_objective_value:
                best_objective_value = composite_value
                best_fluence_map = nominal_opt.get_current_fluence()
                convergence_count = 0
            else:
                convergence_count += 1
                
            # Report progress
            if self.parameters['iteration_callback'] is not None:
                self.parameters['iteration_callback'](iteration, composite_value)
                
            self.service_manager.publish('optimization_progress', {
                'plan_id': self.plan.plan_id,
                'iteration': iteration,
                'objective_value': composite_value,
                'convergence_count': convergence_count
            })
                
            # Check for convergence
            if convergence_count >= 5:
                logger.info(f"Converged after {iteration+1} iterations")
                break
                
        # Apply best fluence to final plan
        if best_fluence_map is not None:
            nominal_opt.apply_fluence(best_fluence_map)
            
        # Final dose calculation
        final_plan = nominal_opt.get_current_plan()
        try:
            final_dose = self.dose_calculator.calculate_dose(final_plan)
            final_plan.dose_grid = final_dose
        except Exception as e:
            logger.error(f"Error calculating final dose: {e}")
            
        # Finalize progress
        self.service_manager.publish('optimization_completed', {
            'plan_id': self.plan.plan_id,
            'iterations': self.current_iteration + 1,
            'final_objective_value': best_objective_value
        })
            
        # Return results
        return {
            'plan': final_plan,
            'objective_values': self.objective_values,
            'final_objective_value': best_objective_value,
            'iterations': self.current_iteration + 1,
            'success': True
        }
    
    def _calculate_composite_objective(self, 
                                      nominal_optimizer: ObjectiveBasedOptimizer,
                                      scenario_optimizers: Dict[str, ObjectiveBasedOptimizer]) -> float:
        """
        Calculate composite objective value across all scenarios.
        
        Parameters
        ----------
        nominal_optimizer : ObjectiveBasedOptimizer
            Optimizer for nominal scenario
        scenario_optimizers : Dict[str, ObjectiveBasedOptimizer]
            Optimizers for all other scenarios
            
        Returns
        -------
        float
            Composite objective value
        """
        # Get objective value for nominal scenario
        nominal_value = nominal_optimizer.calculate_objective_value()
        
        # Get objective values for all other scenarios
        scenario_values = {}
        for name, optimizer in scenario_optimizers.items():
            scenario_values[name] = optimizer.calculate_objective_value()
            
        # Calculate composite value based on strategy
        if self.parameters['worst_case']:
            # Worst-case optimization
            return max([nominal_value * self.nominal_scenario.weight] + 
                      [v * self.scenarios[i].weight for i, (_, v) in enumerate(scenario_values.items())])
        else:
            # Weighted average
            total_weight = self.nominal_scenario.weight + sum(s.weight for s in self.scenarios)
            weighted_sum = nominal_value * self.nominal_scenario.weight
            
            for i, (_, value) in enumerate(scenario_values.items()):
                weighted_sum += value * self.scenarios[i].weight
                
            return weighted_sum / total_weight
    
    def add_constraint(self, constraint: ConstraintFunction) -> None:
        """
        Add constraint to optimization.
        
        Parameters
        ----------
        constraint : ConstraintFunction
            Constraint function
        """
        # Add to objectives
        self.objectives.add_constraint(constraint)
        
    def get_robustness_metrics(self) -> Dict:
        """
        Get robustness metrics for current plan.
        
        Returns
        -------
        Dict
            Dictionary of robustness metrics
        """
        metrics = {}
        
        # Check if we have scenario doses
        if not self.scenarios or not hasattr(self.scenarios[0], 'dose_grid') or self.scenarios[0].dose_grid is None:
            logger.warning("No scenario doses available for robustness metrics")
            return metrics
            
        # Get target structures
        target_names = []
        for obj in self.objectives.objectives:
            if obj.structure_name.lower().startswith(('ptv', 'ctv', 'target')):
                target_names.append(obj.structure_name)
                
        target_names = list(set(target_names))  # Remove duplicates
        
        # Calculate metrics for each target
        for target in target_names:
            # Get D95 for each scenario
            d95_values = []
            
            # Nominal scenario
            if hasattr(self.nominal_scenario, 'dose_grid') and self.nominal_scenario.dose_grid is not None:
                try:
                    d95 = self._calculate_d95(target, self.nominal_scenario.dose_grid)
                    d95_values.append(d95)
                except Exception as e:
                    logger.warning(f"Error calculating D95 for nominal scenario: {e}")
            
            # Other scenarios
            for scenario in self.scenarios:
                if hasattr(scenario, 'dose_grid') and scenario.dose_grid is not None:
                    try:
                        d95 = self._calculate_d95(target, scenario.dose_grid)
                        d95_values.append(d95)
                    except Exception as e:
                        logger.warning(f"Error calculating D95 for scenario {scenario.name}: {e}")
            
            if d95_values:
                # Calculate min, max, and range of D95
                min_d95 = min(d95_values)
                max_d95 = max(d95_values)
                d95_range = max_d95 - min_d95
                
                metrics[f"{target}_min_D95"] = min_d95
                metrics[f"{target}_max_D95"] = max_d95
                metrics[f"{target}_D95_range"] = d95_range
                
                # Robustness score (smaller is better)
                robustness_score = d95_range / max_d95 if max_d95 > 0 else 0
                metrics[f"{target}_robustness_score"] = robustness_score
                
        return metrics
    
    def _calculate_d95(self, structure_name: str, dose_grid: DoseGrid) -> float:
        """
        Calculate D95 for a structure.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        dose_grid : DoseGrid
            Dose grid to calculate from
            
        Returns
        -------
        float
            D95 value in Gy
        """
        # This is a simplified implementation
        # In a real system, you would use the DVH module
        return 0.0  # Placeholder
        
    def get_current_plan(self) -> Plan:
        """
        Get the current plan.
        
        Returns
        -------
        Plan
            The current plan
        """
        return self.plan
        
    def plot_objective_history(self, figsize=(10, 6)):
        """
        Plot the history of objective values.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, len(self.objective_values) + 1), self.objective_values, 'o-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Composite Objective Value')
        ax.set_title('Robust Optimization Progress')
        ax.grid(True)
        
        # Add best value marker
        if self.objective_values:
            best_idx = np.argmin(self.objective_values)
            best_value = self.objective_values[best_idx]
            ax.plot(best_idx + 1, best_value, 'ro', markersize=10, label=f'Best: {best_value:.4f}')
            ax.legend()
            
        return fig


def create_robust_objective(base_objective: ObjectiveFunction, 
                           weight_factor: float = 1.0,
                           priority: int = 1) -> ObjectiveFunction:
    """
    Create a robust version of an objective function.
    
    Parameters
    ----------
    base_objective : ObjectiveFunction
        Base objective function
    weight_factor : float, optional
        Factor to apply to weight, by default 1.0
    priority : int, optional
        Priority level, by default 1
        
    Returns
    -------
    ObjectiveFunction
        Robust objective function
    """
    # Clone the objective
    robust_obj = base_objective.copy()
    
    # Update weight and priority
    robust_obj.weight *= weight_factor
    robust_obj.priority = priority
    
    # Add robustness flag
    robust_obj.is_robust = True
    
    return robust_obj


def optimize_robust_plan(plan: Plan, 
                         objectives: PlanningObjectives,
                         dose_calculator: DoseCalculator,
                         setup_uncertainty: float = 3.0,
                         range_uncertainty: float = 3.5,
                         max_iterations: int = 100,
                         scenario_sampling: str = 'corners') -> Dict:
    """
    Optimize a plan with robustness considerations.
    
    Parameters
    ----------
    plan : Plan
        Treatment plan to optimize
    objectives : PlanningObjectives
        Planning objectives
    dose_calculator : DoseCalculator
        Dose calculation engine
    setup_uncertainty : float, optional
        Setup uncertainty in mm, by default 3.0
    range_uncertainty : float, optional
        Range uncertainty in percent, by default 3.5
    max_iterations : int, optional
        Maximum number of iterations, by default 100
    scenario_sampling : str, optional
        Scenario sampling strategy ('corners', 'random', 'hybrid'), by default 'corners'
        
    Returns
    -------
    Dict
        Optimization results
    """
    # Create optimizer
    optimizer = RobustOptimizer(plan, objectives, dose_calculator)
    
    # Set parameters
    optimizer.set_parameter('max_iterations', max_iterations)
    optimizer.set_parameter('setup_uncertainty', setup_uncertainty)
    optimizer.set_parameter('range_uncertainty', range_uncertainty)
    optimizer.set_parameter('scenario_sampling', scenario_sampling)
    
    # Run optimization
    return optimizer.optimize() 