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

    SETUP = "SETUP"  # Positional setup uncertainties
    RANGE = "RANGE"  # Range uncertainties for particle therapy
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

    def get_dvh_for_structure(
        self, structure_name: str
    ) -> Dict[str, Dict[str, np.ndarray]]:
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
            result[self.nominal_scenario.scenario_name] = (
                self.nominal_scenario.dvh_data[structure_name]
            )

        # Add other scenarios
        for scenario in self.scenarios:
            if structure_name in scenario.dvh_data:
                result[scenario.scenario_name] = scenario.dvh_data[structure_name]

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical summary of robustness analysis.

        Returns:
            Dict containing statistics about robustness analysis
        """
        stats = {
            "target_coverage_range": self.target_coverage_range,
            "oar_dose_range": self.oar_dose_range,
            "num_scenarios": len(self.scenarios) + 1,  # Include nominal scenario
            "scenario_names": [self.nominal_scenario.scenario_name]
            + [s.scenario_name for s in self.scenarios],
        }

        # Calculate additional statistics
        if self.target_coverage_range:
            # Calculate coefficient of variation for target coverage
            target_cv = {}
            for target, (min_val, max_val) in self.target_coverage_range.items():
                mean = (min_val + max_val) / 2
                if mean > 0:
                    std = (max_val - min_val) / (2 * 1.96)  # Assuming 95% CI
                    target_cv[target] = std / mean
                else:
                    target_cv[target] = 0

            stats["target_coverage_cv"] = target_cv

        return stats

    def plot_dvh_band(
        self, structure_name, ax=None, color=None, alpha=0.3, nominal_linestyle="-"
    ):
        """
        Plot DVH band showing the range of DVH curves across all scenarios.

        Args:
            structure_name: Name of the structure to plot
            ax: Matplotlib axis to plot on, if None a new figure is created
            color: Color for the DVH band
            alpha: Alpha value for the band
            nominal_linestyle: Line style for the nominal scenario

        Returns:
            The matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        dvh_data = self.get_dvh_for_structure(structure_name)

        if not dvh_data:
            ax.text(
                0.5,
                0.5,
                f"No DVH data for {structure_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        return ax

        # Set default color if not provided
        if color is None:
            color = "blue"

        # Get all volume and dose arrays
        all_volumes = []
        all_doses = []

        for scenario_name, data in dvh_data.items():
            if "volume" in data and "dose" in data:
                all_volumes.append(data["volume"])
                all_doses.append(data["dose"])

        # Create common dose axis for interpolation
        if all_doses:
            # Find min and max dose values across all scenarios
            min_dose = min([d.min() for d in all_doses if len(d) > 0])
            max_dose = max([d.max() for d in all_doses if len(d) > 0])

            # Create common dose axis
            common_dose = np.linspace(min_dose, max_dose, 100)

            # Interpolate volumes for each scenario
            interpolated_volumes = []

            for volumes, doses in zip(all_volumes, all_doses):
                if len(doses) > 1:  # Need at least 2 points for interpolation
                    try:
                        interp_vol = np.interp(
                            common_dose,
                            doses,
                            volumes,
                            left=volumes[0],
                            right=volumes[-1],
                        )
                        interpolated_volumes.append(interp_vol)
                    except Exception as e:
                        logger.warning(
                            f"Error interpolating DVH for {structure_name}: {e}"
                        )

            # Calculate min and max volumes at each dose point
            if interpolated_volumes:
                min_volumes = np.min(interpolated_volumes, axis=0)
                max_volumes = np.max(interpolated_volumes, axis=0)

                # Plot the band
                ax.fill_between(
                    common_dose, min_volumes, max_volumes, color=color, alpha=alpha
                )

                # Plot nominal scenario
                nominal_data = dvh_data.get(self.nominal_scenario.scenario_name)
                if nominal_data and "volume" in nominal_data and "dose" in nominal_data:
                ax.plot(
                        nominal_data["dose"],
                        nominal_data["volume"],
                        color=color,
                        linestyle=nominal_linestyle,
                        label=f"{structure_name} (nominal)",
                    )

                # Configure axis
                ax.set_xlabel("Dose (Gy)")
                ax.set_ylabel("Volume (%)")
                ax.set_title(f"DVH Band for {structure_name}")
        ax.grid(True)
        ax.legend()

        return ax


class RobustnessAnalyzer:
    """
    Analyzer for treatment plan robustness against uncertainties.

    This class provides methods to evaluate the robustness of treatment plans
    against various uncertainties such as setup errors and range uncertainties.
    """

    def __init__(
        self, plan: Plan, structures: Dict[str, Structure], dose_grid: DoseGrid
    ):
        """
        Initialize robustness analyzer.

        Parameters
        ----------
        plan : Plan
            Treatment plan to analyze
        structures : Dict[str, Structure]
            Dictionary of structures
        dose_grid : DoseGrid
            Dose grid of the plan
        """
        self.plan = plan
        self.structures = structures
        self.dose_grid = dose_grid
        self.dose_calculator = DoseCalculator()

        # Setup and range uncertainty parameters
        self.setup_uncertainty = 3.0  # mm
        self.range_uncertainty = 3.5  # percentage

        # Dictionary of target structure names
        self.target_names = []
        self.oar_names = []

        # Initialize target and OAR lists based on structure types
        self._initialize_structure_lists()

        # Scenario name format
        self.scenario_name_format = "{type}_{direction}_{magnitude}"

    def _initialize_structure_lists(self):
        """Initialize target and OAR lists based on structure types."""
        for name, structure in self.structures.items():
            if structure.type.lower() in ["ptv", "ctv", "gtv", "target"]:
                self.target_names.append(name)
            elif structure.type.lower() in ["oar", "organ", "normal"]:
                self.oar_names.append(name)

        logger.info(
            f"Initialized with {len(self.target_names)} targets and {len(self.oar_names)} OARs"
        )

    def set_setup_uncertainty(self, uncertainty_mm: float):
        """
        Set setup uncertainty value.

        Parameters
        ----------
        uncertainty_mm : float
            Setup uncertainty in millimeters
        """
        self.setup_uncertainty = uncertainty_mm

    def set_range_uncertainty(self, uncertainty_percent: float):
        """
        Set range uncertainty value.

        Parameters
        ----------
        uncertainty_percent : float
            Range uncertainty in percentage
        """
        self.range_uncertainty = uncertainty_percent

    def set_structure_lists(self, target_names: List[str], oar_names: List[str]):
        """
        Manually set target and OAR structure lists.

        Parameters
        ----------
        target_names : List[str]
            List of target structure names
        oar_names : List[str]
            List of OAR structure names
        """
        self.target_names = target_names
        self.oar_names = oar_names

    def _generate_scenarios(self) -> Dict[str, Dict[str, float]]:
        """
        Generate uncertainty scenarios for analysis.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping scenario names to uncertainty parameters
        """
        scenarios = {}

        # Nominal scenario
        scenarios["nominal"] = {}

        # Setup uncertainty scenarios
        for axis in ["x", "y", "z"]:
            for direction in ["+", "-"]:
                shift = (
                    self.setup_uncertainty
                    if direction == "+"
                    else -self.setup_uncertainty
                )

                scenario_name = self.scenario_name_format.format(
                    type="setup",
                    direction=f"{direction}{axis}",
                    magnitude=f"{abs(shift):.1f}",
                )

                scenarios[scenario_name] = {f"setup_{axis}": shift}

        # Range uncertainty scenarios (if applicable)
        is_particle = hasattr(self.plan, "modality") and getattr(
            self.plan, "modality", ""
        ).lower() in ["proton", "carbon"]

        if is_particle and self.range_uncertainty > 0:
            for direction in ["+", "-"]:
                shift = (
                    self.range_uncertainty
                    if direction == "+"
                    else -self.range_uncertainty
                )

                scenario_name = self.scenario_name_format.format(
                    type="range", direction=direction, magnitude=f"{abs(shift):.1f}"
                )

                scenarios[scenario_name] = {"range": shift}

        logger.info(f"Generated {len(scenarios)} scenarios for robustness analysis")

        return scenarios

    def _calculate_dose_for_scenario(
        self, scenario_name: str, params: Dict[str, float]
    ) -> DoseGrid:
        """
        Calculate dose for a specific scenario.

        Parameters
        ----------
        scenario_name : str
            Scenario name
        params : Dict[str, float]
            Uncertainty parameters

        Returns
        -------
        DoseGrid
            Dose grid for the scenario
        """
        logger.info(f"Calculating dose for scenario: {scenario_name}")

        # Create a copy of the plan for this scenario
        scenario_plan = self.plan.copy()

        # Apply setup shifts
        if any(key.startswith("setup_") for key in params):
            x_shift = params.get("setup_x", 0)
            y_shift = params.get("setup_y", 0)
            z_shift = params.get("setup_z", 0)

            # Shift isocenter in opposite direction (shift patient = -shift isocenter)
            if hasattr(scenario_plan, "isocenter"):
                current_iso = scenario_plan.isocenter
                new_iso = [
                    current_iso[0] - x_shift,
                    current_iso[1] - y_shift,
                    current_iso[2] - z_shift,
                ]
                scenario_plan.isocenter = new_iso

        # Apply range uncertainty
        if "range" in params:
            range_shift = params["range"]

            # Scale energy for all beams
            range_factor = 1.0 + range_shift / 100.0

            if hasattr(scenario_plan, "beams"):
                for beam in scenario_plan.beams:
                    if hasattr(beam, "energy"):
                        beam.energy *= range_factor

        # Calculate dose
        try:
            scenario_dose = self.dose_calculator.calculate_dose(scenario_plan)
            return scenario_dose
        except Exception as e:
            logger.error(f"Error calculating dose for scenario {scenario_name}: {e}")
            # Return nominal dose as fallback
            return self.dose_grid

    def _calculate_dvh_for_scenario(
        self, dose_grid: DoseGrid
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate DVH for all structures using a specific dose grid.

        Parameters
        ----------
        dose_grid : DoseGrid
            Dose grid

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Dictionary mapping structure names to DVH data
        """
        dvh_data = {}

        # Calculate DVH for targets
        for target_name in self.target_names:
            if target_name in self.structures:
                structure = self.structures[target_name]
                try:
                    dvh_data[target_name] = calculate_dvh(structure, dose_grid)
                except Exception as e:
                    logger.error(f"Error calculating DVH for target {target_name}: {e}")

        # Calculate DVH for OARs
        for oar_name in self.oar_names:
            if oar_name in self.structures:
                structure = self.structures[oar_name]
                try:
                    dvh_data[oar_name] = calculate_dvh(structure, dose_grid)
            except Exception as e:
                    logger.error(f"Error calculating DVH for OAR {oar_name}: {e}")

        return dvh_data

    def _calculate_target_coverage(
        self, dvh_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """
        Calculate coverage metrics for targets.

        Parameters
        ----------
        dvh_data : Dict[str, Dict[str, np.ndarray]]
            DVH data for all structures

        Returns
        -------
        Dict[str, float]
            Dictionary mapping target names to coverage values
        """
        coverage = {}

        for target_name in self.target_names:
            if target_name in dvh_data:
                target_dvh = dvh_data[target_name]

                # Extract prescription dose (simplified)
                rx_dose = 0
                if hasattr(self.plan, "prescriptions"):
                    for rx in self.plan.prescriptions:
                        if (
                            rx.structure_id == target_name
                            or rx.structure_name == target_name
                        ):
                            rx_dose = rx.dose
                            break

                if rx_dose <= 0:
                    # Default to D95 if no prescription found
                    coverage[target_name] = self._calculate_d95(target_dvh)
                else:
                    # Calculate V100% (volume receiving prescription dose)
                    coverage[target_name] = self._calculate_v_dose(target_dvh, rx_dose)

        return coverage

    def _calculate_oar_doses(
        self, dvh_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """
        Calculate dose metrics for OARs.

        Parameters
        ----------
        dvh_data : Dict[str, Dict[str, np.ndarray]]
            DVH data for all structures

        Returns
        -------
        Dict[str, float]
            Dictionary mapping OAR names to dose values
        """
        doses = {}

        for oar_name in self.oar_names:
            if oar_name in dvh_data:
                oar_dvh = dvh_data[oar_name]

                # Calculate D0.1cc or Dmax
                doses[oar_name] = self._calculate_dmax(oar_dvh)

        return doses

    def _calculate_d95(self, dvh_data: Dict[str, np.ndarray]) -> float:
        """
        Calculate D95 from DVH data.

        Parameters
        ----------
        dvh_data : Dict[str, np.ndarray]
            DVH data

        Returns
        -------
        float
            D95 value in Gy
        """
        try:
            volume = dvh_data.get("volume", [])
            dose = dvh_data.get("dose", [])

            if len(volume) > 0 and len(dose) > 0:
                # Find dose at 95% volume
                d95 = np.interp(95, volume[::-1], dose[::-1])
                return d95
        except Exception as e:
            logger.error(f"Error calculating D95: {e}")

        return 0.0

    def _calculate_v_dose(self, dvh_data: Dict[str, np.ndarray], dose: float) -> float:
        """
        Calculate volume receiving at least specified dose.

        Parameters
        ----------
        dvh_data : Dict[str, np.ndarray]
            DVH data
        dose : float
            Dose threshold in Gy

        Returns
        -------
        float
            Volume percentage
        """
        try:
            volume = dvh_data.get("volume", [])
            dose_values = dvh_data.get("dose", [])

            if len(volume) > 0 and len(dose_values) > 0:
                # Find volume at specified dose
                v_dose = np.interp(dose, dose_values, volume)
                return v_dose
        except Exception as e:
            logger.error(f"Error calculating V{dose:.1f}: {e}")

        return 0.0

    def _calculate_dmax(self, dvh_data: Dict[str, np.ndarray]) -> float:
        """
        Calculate maximum dose (D0.1cc or similar).

        Parameters
        ----------
        dvh_data : Dict[str, np.ndarray]
            DVH data

        Returns
        -------
        float
            Maximum dose in Gy
        """
        try:
            volume = dvh_data.get("volume", [])
            dose = dvh_data.get("dose", [])

            if len(volume) > 0 and len(dose) > 0:
                # Find dose at 0.1% volume
                dmax = np.interp(0.1, volume[::-1], dose[::-1])
                return dmax
        except Exception as e:
            logger.error(f"Error calculating Dmax: {e}")

        return 0.0

    def analyze(self) -> RobustnessResult:
        """
        Analyze plan robustness against uncertainties.

        Returns
        -------
        RobustnessResult
            Results of robustness analysis
        """
        logger.info("Starting robustness analysis")
        start_time = time.time()

        # Generate scenarios
        scenarios = self._generate_scenarios()

        # Initialize results containers
        scenario_results = []
        nominal_result = None

        # Process nominal scenario first
        nominal_params = scenarios.pop("nominal")
        nominal_dvh = self._calculate_dvh_for_scenario(self.dose_grid)

        nominal_result = ScenarioResult(
            scenario_name="nominal",
            uncertainty_parameters=nominal_params,
            dose_grid=self.dose_grid,
            dvh_data=nominal_dvh,
        )

        # Calculate target coverage and OAR doses for nominal scenario
        nominal_coverage = self._calculate_target_coverage(nominal_dvh)
        nominal_oar_doses = self._calculate_oar_doses(nominal_dvh)

        # Initialize min/max values with nominal values
        min_coverage = {t: v for t, v in nominal_coverage.items()}
        max_coverage = {t: v for t, v in nominal_coverage.items()}
        min_oar_doses = {o: v for o, v in nominal_oar_doses.items()}
        max_oar_doses = {o: v for o, v in nominal_oar_doses.items()}

        # Process all other scenarios
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit dose calculation tasks
            future_to_scenario = {}
            for scenario_name, params in scenarios.items():
                future = executor.submit(
                    self._calculate_dose_for_scenario, scenario_name, params
                )
                future_to_scenario[future] = (scenario_name, params)

            # Process results as they complete
            for future in as_completed(future_to_scenario):
                scenario_name, params = future_to_scenario[future]
                try:
                    scenario_dose = future.result()

                    # Calculate DVH
                    scenario_dvh = self._calculate_dvh_for_scenario(scenario_dose)

                    # Create scenario result
                    scenario_result = ScenarioResult(
                        scenario_name=scenario_name,
                        uncertainty_parameters=params,
                        dose_grid=scenario_dose,
                        dvh_data=scenario_dvh,
                    )

                    # Calculate metrics
                    scenario_coverage = self._calculate_target_coverage(scenario_dvh)
                    scenario_oar_doses = self._calculate_oar_doses(scenario_dvh)

                    # Update min/max values
                    for target, coverage in scenario_coverage.items():
                        min_coverage[target] = min(
                            min_coverage.get(target, float("inf")), coverage
                        )
                        max_coverage[target] = max(
                            max_coverage.get(target, float("-inf")), coverage
                        )

                    for oar, dose in scenario_oar_doses.items():
                        min_oar_doses[oar] = min(
                            min_oar_doses.get(oar, float("inf")), dose
                        )
                        max_oar_doses[oar] = max(
                            max_oar_doses.get(oar, float("-inf")), dose
                        )

                    # Store result
                    scenario_results.append(scenario_result)

                except Exception as e:
                    logger.error(f"Error processing scenario {scenario_name}: {e}")

        # Compute final ranges
        target_coverage_range = {
            t: (min_coverage[t], max_coverage[t]) for t in min_coverage
        }
        oar_dose_range = {
            o: (min_oar_doses[o], max_oar_doses[o]) for o in min_oar_doses
        }

        # Create final result
        result = RobustnessResult(
            nominal_scenario=nominal_result,
            scenarios=scenario_results,
            target_coverage_range=target_coverage_range,
            oar_dose_range=oar_dose_range,
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Robustness analysis completed in {elapsed_time:.2f} seconds")

        return result

    def plot_dvh_bands(self, figsize=(12, 8)):
        """
        Analyze plan and plot DVH bands for targets and OARs.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 8)

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing DVH bands
        """
        # Run analysis if not already done
        result = self.analyze()

        # Create figure
        fig = plt.figure(figsize=figsize)

        # Determine number of subplots based on number of structures
        n_structures = len(self.target_names) + len(self.oar_names)
        n_rows = int(np.ceil(n_structures / 2))

        # Create color map
        target_color = "red"
        oar_colors = plt.cm.viridis(np.linspace(0, 1, len(self.oar_names)))

        # Plot targets
        for i, target_name in enumerate(self.target_names):
            ax = fig.add_subplot(n_rows, 2, i + 1)
            result.plot_dvh_band(target_name, ax=ax, color=target_color)

        # Plot OARs
        for i, oar_name in enumerate(self.oar_names):
            ax = fig.add_subplot(n_rows, 2, i + len(self.target_names) + 1)
            result.plot_dvh_band(oar_name, ax=ax, color=oar_colors[i])

        plt.tight_layout()
        return fig


def analyze_plan_robustness(
    plan: Plan,
    structures: Dict[str, Structure],
    dose_grid: DoseGrid,
    setup_uncertainty: float = 3.0,
    range_uncertainty: float = 3.5,
) -> RobustnessResult:
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
