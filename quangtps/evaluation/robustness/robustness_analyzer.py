#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robustness analysis module for evaluating treatment plan robustness.

This module provides tools to evaluate the robustness of treatment plans against
uncertainties such as patient setup errors and range uncertainties for particle therapy.
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import time

from quangtps.core.logging import get_logger
from quangtps.core.types import DoseGrid
from quangtps.core.structure import Structure
from quangtps.core.plan import Plan
from quangtps.evaluation.dvh.dvh_calculator import calculate_dvh
from quangtps.dose.dose_calculator import DoseCalculator

logger = get_logger(__name__)


class UncertaintyType:
    """Enumeration of uncertainty types for robust analysis."""
    SETUP = "SETUP"          # Positional setup uncertainties
    RANGE = "RANGE"          # Range uncertainties for particle therapy
    DEFORMATION = "DEFORMATION"  # Anatomical deformation uncertainties
    BREATHING = "BREATHING"  # Breathing motion uncertainties


@dataclass
class ScenarioResult:
    """Results of a single robustness scenario."""
    scenario_name: str
    uncertainty_parameters: Dict[str, float]
    dose_grid: DoseGrid
    dvh_data: Dict[str, Dict[str, np.ndarray]]


@dataclass
class RobustnessResult:
    """Results of a robustness analysis."""
    nominal_scenario: ScenarioResult
    scenarios: List[ScenarioResult]
    target_coverage_range: Dict[str, Tuple[float, float]]
    oar_dose_range: Dict[str, Tuple[float, float]]
    
    def get_scenario_by_name(self, name: str) -> Optional[ScenarioResult]:
        """
        Get a scenario by name.
        
        Args:
            name: The name of the scenario
            
        Returns:
            ScenarioResult if found, None otherwise
        """
        if self.nominal_scenario.scenario_name == name:
            return self.nominal_scenario
            
        for scenario in self.scenarios:
            if scenario.scenario_name == name:
                return scenario
                
        return None
    
    def get_dvh_for_structure(self, structure_name: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get DVH data for a structure across all scenarios.
        
        Args:
            structure_name: The name of the structure
            
        Returns:
            Dict mapping scenario names to DVH data
        """
        result = {}
        
        # Add nominal scenario
        if structure_name in self.nominal_scenario.dvh_data:
            result[self.nominal_scenario.scenario_name] = self.nominal_scenario.dvh_data[structure_name]
            
        # Add other scenarios
        for scenario in self.scenarios:
            if structure_name in scenario.dvh_data:
                result[scenario.scenario_name] = scenario.dvh_data[structure_name]
                
        return result
    
    def get_band_dvh(self, structure_name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get the lower and upper bounds of DVH for all scenarios.
        
        Args:
            structure_name: The name of the structure
            
        Returns:
            Tuple[Dict, Dict]: (Lower bound DVH, Upper bound DVH)
        """
        if structure_name not in self.nominal_scenario.dvh_data:
            raise ValueError(f"Structure '{structure_name}' does not exist")
        
        # Combine all DVHs
        all_doses = []
        all_volumes = []
        
        nominal_dvh = self.nominal_scenario.dvh_data[structure_name]
        all_doses.append(nominal_dvh['dose'])
        all_volumes.append(nominal_dvh['volume_percent'])
        
        for scenario in self.scenarios:
            if structure_name in scenario.dvh_data:
                dvh = scenario.dvh_data[structure_name]
                all_doses.append(dvh['dose'])
                all_volumes.append(dvh['volume_percent'])
        
        # Create common dose grid
        min_dose = min(np.min(d) for d in all_doses)
        max_dose = max(np.max(d) for d in all_doses)
        common_dose = np.linspace(min_dose, max_dose, 100)
        
        # Interpolate volume at each dose point
        interp_volumes = []
        for i, dose in enumerate(all_doses):
            volume = all_volumes[i]
            interp_vol = np.interp(common_dose, dose, volume, left=100, right=0)
            interp_volumes.append(interp_vol)
        
        # Calculate lower and upper bounds
        lower_bound = np.min(interp_volumes, axis=0)
        upper_bound = np.max(interp_volumes, axis=0)
        
        # Return DVH format
        lower_dvh = {
            'dose': common_dose,
            'volume_percent': lower_bound
        }
        
        upper_dvh = {
            'dose': common_dose,
            'volume_percent': upper_bound
        }
        
        return lower_dvh, upper_dvh
    
    def plot_dvh_band(self, structure_name: str, ax=None):
        """
        Plot DVH band for a structure.
        
        Args:
            structure_name: The name of the structure
            ax: Matplotlib axis to plot on (creates a new one if None)
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get DVH band
        lower_dvh, upper_dvh = self.get_band_dvh(structure_name)
        
        # Plot the band
        ax.fill_between(
            lower_dvh['dose'], 
            lower_dvh['volume_percent'], 
            upper_dvh['volume_percent'],
            alpha=0.3, color='blue'
        )
        
        # Plot the nominal curve
        nominal_dvh = self.nominal_scenario.dvh_data[structure_name]
        ax.plot(
            nominal_dvh['dose'], 
            nominal_dvh['volume_percent'],
            color='blue', linewidth=2, label=f"{structure_name} (nominal)"
        )
        
        # Set labels and title
        ax.set_xlabel('Dose (Gy)')
        ax.set_ylabel('Volume (%)')
        ax.set_title(f'DVH Robustness Band for {structure_name}')
        ax.grid(True)
        ax.legend()
        
        return ax
    
    def plot_all_structures(self, structures: List[str], figsize=(12, 8)):
        """
        Plot DVH bands for multiple structures.
        
        Args:
            structures: List of structure names to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10.colors
        for i, structure in enumerate(structures):
            if structure not in self.nominal_scenario.dvh_data:
                continue
                
            color = colors[i % len(colors)]
            
            # Get DVH band
            try:
                lower_dvh, upper_dvh = self.get_band_dvh(structure)
                
                # Plot the band
                ax.fill_between(
                    lower_dvh['dose'], 
                    lower_dvh['volume_percent'], 
                    upper_dvh['volume_percent'],
                    alpha=0.2, color=color
                )
                
                # Plot the nominal curve
                nominal_dvh = self.nominal_scenario.dvh_data[structure]
                ax.plot(
                    nominal_dvh['dose'], 
                    nominal_dvh['volume_percent'],
                    color=color, linewidth=2, label=structure
                )
            except Exception as e:
                logger.error(f"Error plotting {structure}: {e}")
        
        # Set labels and title
        ax.set_xlabel('Dose (Gy)')
        ax.set_ylabel('Volume (%)')
        ax.set_title('DVH Robustness Bands')
        ax.grid(True)
        ax.legend()
        
        return fig

    def plot_all_dvh_bands(self, targets: List[str] = None, oars: List[str] = None, figsize=(12, 8)):
        """
        Plot DVH bands for targets and OARs in separate subplots.
        
        Args:
            targets: List of target structure names to plot (defaults to all targets)
            oars: List of OAR structure names to plot (defaults to all OARs)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if targets is None:
            targets = [s for s in self.nominal_scenario.dvh_data.keys() 
                       if s.lower().startswith(('ptv', 'ctv', 'target'))]
        
        if oars is None:
            oars = [s for s in self.nominal_scenario.dvh_data.keys() 
                    if not s.lower().startswith(('ptv', 'ctv', 'target'))]
        
        has_targets = len(targets) > 0
        has_oars = len(oars) > 0
        
        if not (has_targets or has_oars):
            # No structures to plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No structures to plot", ha='center', va='center')
            return fig
        
        if has_targets and has_oars:
            # Both targets and OARs
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Plot targets
            self._plot_structure_group(targets, ax1, "Targets")
            
            # Plot OARs
            self._plot_structure_group(oars, ax2, "Organs at Risk")
        elif has_targets:
            # Only targets
            fig, ax = plt.subplots(figsize=figsize)
            self._plot_structure_group(targets, ax, "Targets")
        else:
            # Only OARs
            fig, ax = plt.subplots(figsize=figsize)
            self._plot_structure_group(oars, ax, "Organs at Risk")
        
        fig.tight_layout()
        return fig
    
    def _plot_structure_group(self, structures: List[str], ax, title: str):
        """
        Plot a group of structures on a single axis.
        
        Args:
            structures: List of structure names to plot
            ax: Matplotlib axis to plot on
            title: Title for the plot
        """
        colors = plt.cm.tab10.colors
        for i, structure in enumerate(structures):
            if structure not in self.nominal_scenario.dvh_data:
                continue
                
            color = colors[i % len(colors)]
            
            try:
                # Get DVH band
                lower_dvh, upper_dvh = self.get_band_dvh(structure)
                
                # Plot the band
                ax.fill_between(
                    lower_dvh['dose'], 
                    lower_dvh['volume_percent'], 
                    upper_dvh['volume_percent'],
                    alpha=0.2, color=color
                )
                
                # Plot the nominal curve
                nominal_dvh = self.nominal_scenario.dvh_data[structure]
                ax.plot(
                    nominal_dvh['dose'], 
                    nominal_dvh['volume_percent'],
                    color=color, linewidth=2, label=structure
                )
            except Exception as e:
                logger.error(f"Error plotting {structure}: {e}")
        
        # Set labels and title
        ax.set_xlabel('Dose (Gy)')
        ax.set_ylabel('Volume (%)')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()


class RobustnessAnalyzer:
    """
    Analyzer for evaluating treatment plan robustness against uncertainties.
    
    This class evaluates how a treatment plan behaves under various uncertainty
    scenarios, such as setup errors and range uncertainties.
    """
    
    def __init__(self, plan: Plan, structures: Dict[str, Structure], dose_grid: DoseGrid):
        """
        Initialize the robustness analyzer.
        
        Parameters
        ----------
        plan : Plan
            The treatment plan to analyze
        structures : Dict[str, Structure]
            Dictionary of structures
        dose_grid : DoseGrid
            The dose grid of the plan
        """
        self.plan = plan
        self.structures = structures
        self.dose_grid = dose_grid
        self.dose_calculator = None
        
        # Default parameters
        self.setup_uncertainty = 3.0  # mm
        self.range_uncertainty = 3.5  # percent
        self.scenarios = []
        
        # For progress tracking
        self.progress_callback = None
        
    def set_setup_uncertainty(self, uncertainty_mm: float):
        """
        Set the setup uncertainty magnitude in mm.
        
        Parameters
        ----------
        uncertainty_mm : float
            Setup uncertainty in mm
        """
        self.setup_uncertainty = uncertainty_mm
        self.scenarios = []  # Clear existing scenarios
        
    def set_range_uncertainty(self, uncertainty_percent: float):
        """
        Set the range uncertainty magnitude in percent.
        
        Parameters
        ----------
        uncertainty_percent : float
            Range uncertainty in percent
        """
        self.range_uncertainty = uncertainty_percent
        self.scenarios = []  # Clear existing scenarios
        
    def generate_scenarios(self) -> List[Dict[str, float]]:
        """
        Generate standard uncertainty scenarios.
        
        Returns
        -------
        List[Dict[str, float]]
            List of scenario parameter dictionaries
        """
        scenarios = []
        
        # Add nominal scenario (no shifts)
        scenarios.append({})
        
        # Setup uncertainty scenarios
        for x_shift in [-self.setup_uncertainty, 0, self.setup_uncertainty]:
            for y_shift in [-self.setup_uncertainty, 0, self.setup_uncertainty]:
                for z_shift in [-self.setup_uncertainty, 0, self.setup_uncertainty]:
                    # Skip nominal (already added)
                    if x_shift == 0 and y_shift == 0 and z_shift == 0:
                        continue
                        
                    scenario = {}
                    if x_shift != 0:
                        scenario['shift_x'] = x_shift
                    if y_shift != 0:
                        scenario['shift_y'] = y_shift
                    if z_shift != 0:
                        scenario['shift_z'] = z_shift
                        
                    scenarios.append(scenario)
        
        # Range uncertainty scenarios (only for particle beams)
        has_particle_beams = False
        for beam in self.plan.beams:
            if hasattr(beam, 'is_particle_beam') and beam.is_particle_beam():
                has_particle_beams = True
                break
                
        if has_particle_beams:
            for r_shift in [-self.range_uncertainty, self.range_uncertainty]:
                scenarios.append({'range': r_shift})
                
        return scenarios
        
    def _simulate_scenario(self, scenario: Dict[str, float]) -> Tuple[Dict[str, float], DoseGrid, Dict[str, Dict]]:
        """
        Simulate a single uncertainty scenario.
        
        Parameters
        ----------
        scenario : Dict[str, float]
            Scenario parameters
            
        Returns
        -------
        Tuple[Dict[str, float], DoseGrid, Dict[str, Dict]]
            Tuple of (scenario parameters, dose grid, DVH data)
        """
        if self.dose_calculator is None:
            from quangtps.dose.dose_calculator import DoseCalculator
            self.dose_calculator = DoseCalculator()
            
        # Create a scenario plan with shifts
        modified_plan = self._create_scenario_plan(self.plan, scenario)
        
        # Calculate dose for the scenario
        try:
            dose_grid = self.dose_calculator.calculate_dose(modified_plan)
        except Exception as e:
            logger.error(f"Error calculating dose for scenario {scenario}: {e}")
            # Return empty results
            return scenario, None, {}
            
        # Calculate DVH for each structure
        dvh_data = {}
        for name, structure in self.structures.items():
            try:
                dvh = calculate_dvh(dose_grid, structure)
                dvh_data[name] = dvh
            except Exception as e:
                logger.error(f"Error calculating DVH for structure {name} in scenario {scenario}: {e}")
                
        return scenario, dose_grid, dvh_data
        
    def _create_scenario_plan(self, original_plan: Plan, scenario_params: Dict[str, float]) -> Plan:
        """
        Create a modified plan for a scenario with specified shifts.
        
        Parameters
        ----------
        original_plan : Plan
            Original treatment plan
        scenario_params : Dict[str, float]
            Scenario parameters with shifts
            
        Returns
        -------
        Plan
            Modified plan
        """
        # This is similar to what's in RobustOptimizer._create_scenario_plan
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
        
        # Get shift parameters
        x_shift = scenario_params.get('shift_x', 0)
        y_shift = scenario_params.get('shift_y', 0)
        z_shift = scenario_params.get('shift_z', 0)
        range_shift = scenario_params.get('range', 0)
        
        # Create shifted beams
        for beam in original_plan.beams:
            # Create a copy of the beam
            shifted_beam = beam.copy()
            
            # Apply isocenter shifts (opposite direction for patient shifts)
            if x_shift != 0 or y_shift != 0 or z_shift != 0:
                shifted_beam.shift_isocenter(-x_shift, -y_shift, -z_shift)
                
            # Apply range shift for particle beams
            if range_shift != 0 and hasattr(beam, 'is_particle_beam') and beam.is_particle_beam():
                shifted_beam.apply_range_shift(range_shift)
                
            # Add beam to plan
            modified_plan.add_beam(shifted_beam)
            
        return modified_plan
        
    def _make_scenario_name(self, params: Dict[str, float]) -> str:
        """
        Create a descriptive name for a scenario.
        
        Parameters
        ----------
        params : Dict[str, float]
            Scenario parameters
            
        Returns
        -------
        str
            Scenario name
        """
        # If empty parameters, it's the nominal scenario
        if not params:
            return "Nominal"
            
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
        
    def analyze(self, max_workers: int = 4) -> RobustnessResult:
        """
        Analyze plan robustness by evaluating multiple uncertainty scenarios.
        
        Parameters
        ----------
        max_workers : int, optional
            Maximum number of parallel workers, by default 4
            
        Returns
        -------
        RobustnessResult
            Results of the robustness analysis
        """
        # Generate scenarios if not already done
        if not self.scenarios:
            self.scenarios = self.generate_scenarios()
            
        logger.info(f"Analyzing plan robustness with {len(self.scenarios)} scenarios")
        
        # Run scenarios in parallel
        start_time = time.time()
        scenario_results = []
        progress_total = len(self.scenarios)
        progress_current = 0
        
        # Process the nominal scenario first
        nominal_params = {}
        nominal_name = self._make_scenario_name(nominal_params)
        try:
            nominal_params, nominal_dose, nominal_dvh = self._simulate_scenario(nominal_params)
            
            if nominal_dose is None:
                logger.error("Failed to calculate nominal dose")
                return None
                
            progress_current += 1
            if self.progress_callback:
                self.progress_callback(progress_current / progress_total)
                
        except Exception as e:
            logger.error(f"Error simulating nominal scenario: {e}")
            return None
            
        # Process other scenarios in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scenarios except nominal
            future_to_scenario = {}
            for scenario in self.scenarios[1:]:  # Skip nominal (first scenario)
                future = executor.submit(self._simulate_scenario, scenario)
                future_to_scenario[future] = scenario
                
            # Process results as they complete
            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                
                try:
                    params, dose_grid, dvh_data = future.result()
                    
                    # Skip if dose calculation failed
                    if dose_grid is None:
                        logger.warning(f"Skipping scenario {scenario} due to failed dose calculation")
                        continue
                        
                    name = self._make_scenario_name(params)
                    scenario_results.append(ScenarioResult(
                        scenario_name=name,
                        uncertainty_parameters=params,
                        dose_grid=dose_grid,
                        dvh_data=dvh_data
                    ))
                    
                except Exception as e:
                    logger.error(f"Error processing scenario {scenario}: {e}")
                
                # Update progress
                progress_current += 1
                if self.progress_callback:
                    self.progress_callback(progress_current / progress_total)
        
        # Create result for nominal scenario
        nominal_result = ScenarioResult(
            scenario_name="Nominal",
            uncertainty_parameters={},
            dose_grid=nominal_dose,
            dvh_data=nominal_dvh
        )
        
        # Calculate robustness metrics
        target_coverage_range = self._calculate_target_coverage_range(nominal_result, scenario_results)
        oar_dose_range = self._calculate_oar_dose_range(nominal_result, scenario_results)
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Robustness analysis completed in {duration:.2f} seconds")
        
        # Create and return final result
        result = RobustnessResult(
            nominal_scenario=nominal_result,
            scenarios=scenario_results,
            target_coverage_range=target_coverage_range,
            oar_dose_range=oar_dose_range
        )
        
        return result
        
    def _calculate_target_coverage_range(self, 
                                        nominal: ScenarioResult, 
                                        scenarios: List[ScenarioResult]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate the range of target coverage across scenarios.
        
        Parameters
        ----------
        nominal : ScenarioResult
            Nominal scenario results
        scenarios : List[ScenarioResult]
            List of scenario results
            
        Returns
        -------
        Dict[str, Tuple[float, float]]
            Dictionary mapping target names to (min, max) D95 values
        """
        result = {}
        
        # Identify target structures
        target_names = []
        for name in nominal.dvh_data.keys():
            if name.lower().startswith(('ptv', 'ctv', 'target')):
                target_names.append(name)
                
        # Calculate D95 range for each target
        for target in target_names:
            if target not in nominal.dvh_data:
                continue
                
            d95_values = []
            
            # Calculate nominal D95
            nominal_d95 = self._calculate_d95(nominal.dvh_data[target])
            d95_values.append(nominal_d95)
            
            # Calculate D95 for each scenario
            for scenario in scenarios:
                if target in scenario.dvh_data:
                    d95 = self._calculate_d95(scenario.dvh_data[target])
                    d95_values.append(d95)
            
            if d95_values:
                result[target] = (min(d95_values), max(d95_values))
                
        return result
        
    def _calculate_oar_dose_range(self, 
                                 nominal: ScenarioResult, 
                                 scenarios: List[ScenarioResult]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate the range of OAR doses across scenarios.
        
        Parameters
        ----------
        nominal : ScenarioResult
            Nominal scenario results
        scenarios : List[ScenarioResult]
            List of scenario results
            
        Returns
        -------
        Dict[str, Tuple[float, float]]
            Dictionary mapping OAR names to (min, max) D1cc values
        """
        result = {}
        
        # Identify OAR structures (non-targets)
        oar_names = []
        for name in nominal.dvh_data.keys():
            if not name.lower().startswith(('ptv', 'ctv', 'target')):
                oar_names.append(name)
                
        # Calculate D1cc range for each OAR
        for oar in oar_names:
            if oar not in nominal.dvh_data:
                continue
                
            d1cc_values = []
            
            # Calculate nominal D1cc
            nominal_d1cc = self._calculate_d1cc(nominal.dvh_data[oar])
            d1cc_values.append(nominal_d1cc)
            
            # Calculate D1cc for each scenario
            for scenario in scenarios:
                if oar in scenario.dvh_data:
                    d1cc = self._calculate_d1cc(scenario.dvh_data[oar])
                    d1cc_values.append(d1cc)
            
            if d1cc_values:
                result[oar] = (min(d1cc_values), max(d1cc_values))
                
        return result
        
    def _calculate_d95(self, dvh: Dict[str, np.ndarray]) -> float:
        """
        Calculate D95 (dose to 95% of volume) from a DVH.
        
        Parameters
        ----------
        dvh : Dict[str, np.ndarray]
            DVH data with 'dose' and 'volume_percent' keys
            
        Returns
        -------
        float
            D95 value in Gy
        """
        dose = dvh['dose']
        volume = dvh['volume_percent']
        
        # Interpolate to find the dose at which volume = 95%
        d95 = np.interp(95, volume[::-1], dose[::-1])
        
        return d95
        
    def _calculate_d1cc(self, dvh: Dict[str, np.ndarray]) -> float:
        """
        Calculate D1cc (dose to 1cc volume) from a DVH.
        
        Parameters
        ----------
        dvh : Dict[str, np.ndarray]
            DVH data with 'dose' and 'volume_percent' keys
            
        Returns
        -------
        float
            D1cc value in Gy
        """
        # This is a simplified implementation
        # In a real implementation, you would convert volume_percent to absolute volume
        # and then find the dose at 1cc
        
        dose = dvh['dose']
        volume = dvh['volume_percent']
        
        # Assume the structure volume is 100cc for this example
        structure_volume = 100  # cc
        
        # Convert 1cc to percent
        percent_1cc = 1 / structure_volume * 100
        
        # Interpolate to find the dose at which volume = percent_1cc
        d1cc = np.interp(percent_1cc, volume[::-1], dose[::-1])
        
        return d1cc


def analyze_plan_robustness(plan: Plan, structures: Dict[str, Structure], dose_grid: DoseGrid, 
                          setup_uncertainty: float = 3.0, range_uncertainty: float = 3.5) -> RobustnessResult:
    """
    Analyze the robustness of a treatment plan.
    
    This is a convenience function that uses RobustnessAnalyzer to evaluate
    plan robustness against setup and range uncertainties.
    
    Parameters
    ----------
    plan : Plan
        The treatment plan to analyze
    structures : Dict[str, Structure]
        Dictionary of structures
    dose_grid : DoseGrid
        The dose grid of the plan
    setup_uncertainty : float, optional
        Setup uncertainty in mm, by default 3.0
    range_uncertainty : float, optional
        Range uncertainty in percent, by default 3.5
    
    Returns
    -------
    RobustnessResult
        Results of the robustness analysis
    """
    analyzer = RobustnessAnalyzer(plan, structures, dose_grid)
    analyzer.set_setup_uncertainty(setup_uncertainty)
    analyzer.set_range_uncertainty(range_uncertainty)
    
    return analyzer.analyze()
