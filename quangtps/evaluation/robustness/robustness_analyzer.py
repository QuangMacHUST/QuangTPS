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
import os

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

    def get_structure_dvhs(
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

    def get_target_coverage_data(self, metric_type: str = "D95") -> Dict[str, Any]:
        """
        Get target coverage data for visualization.

        Args:
            metric_type: Type of metric to return (D95, V95, etc.)

        Returns:
            Dictionary with target coverage data
        """
        coverage_data = {
            "target_names": [],
            "nominal_values": [],
            "min_values": [],
            "max_values": [],
            "all_values": {},
            "metric_type": metric_type,
        }

        # Only process targets with coverage data
        for target, (min_val, max_val) in self.target_coverage_range.items():
            coverage_data["target_names"].append(target)

            # Get nominal value for this target
            nominal_value = 0.0
            if metric_type == "D95":
                nominal_dvh = self.nominal_scenario.dvh_data.get(target, {})
                if nominal_dvh:
                    try:
                        # Calculate D95 from DVH
                        volume = nominal_dvh.get("volume", [])
                        dose = nominal_dvh.get("dose", [])
                        if len(volume) > 0 and len(dose) > 0:
                            # Find dose at 95% volume
                            nominal_value = np.interp(95, volume[::-1], dose[::-1])
                    except Exception as e:
                        logger.warning(
                            f"Error calculating {metric_type} for {target}: {e}"
                        )

            coverage_data["nominal_values"].append(nominal_value)
            coverage_data["min_values"].append(min_val)
            coverage_data["max_values"].append(max_val)

            # Collect values from all scenarios
            all_scenario_values = []
            for scenario in [self.nominal_scenario] + self.scenarios:
                scenario_dvh = scenario.dvh_data.get(target, {})
                if scenario_dvh:
                    try:
                        # Calculate metric from DVH
                        volume = scenario_dvh.get("volume", [])
                        dose = scenario_dvh.get("dose", [])
                        if len(volume) > 0 and len(dose) > 0:
                            # Find dose at 95% volume for D95
                            if metric_type == "D95":
                                value = np.interp(95, volume[::-1], dose[::-1])
                            # Add other metrics as needed
                            all_scenario_values.append(value)
                    except Exception as e:
                        logger.warning(
                            f"Error calculating {metric_type} for {target} in scenario {scenario.scenario_name}: {e}"
                        )

            coverage_data["all_values"][target] = all_scenario_values

        return coverage_data

    def get_evaluation_metrics(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get comprehensive evaluation metrics for targets and OARs.

        Returns:
            Dictionary with metrics for each structure
        """
        metrics = {"targets": {}, "oars": {}}

        # Target metrics
        for target, (min_val, max_val) in self.target_coverage_range.items():
            # Calculate nominal value (from nominal scenario)
            nominal_dvh = self.nominal_scenario.dvh_data.get(target, {})
            d95 = 0.0
            if nominal_dvh:
                try:
                    # Calculate D95 from DVH
                    volume = nominal_dvh.get("volume", [])
                    dose = nominal_dvh.get("dose", [])
                    if len(volume) > 0 and len(dose) > 0:
                        # Find dose at 95% volume
                        d95 = np.interp(95, volume[::-1], dose[::-1])
                except Exception:
                    pass

            # Get all D95 values for variation calculation
            all_d95 = []
            for scenario in [self.nominal_scenario] + self.scenarios:
                scenario_dvh = scenario.dvh_data.get(target, {})
                if scenario_dvh:
                    try:
                        volume = scenario_dvh.get("volume", [])
                        dose = scenario_dvh.get("dose", [])
                        if len(volume) > 0 and len(dose) > 0:
                            # Calculate D95
                            value = np.interp(95, volume[::-1], dose[::-1])
                            all_d95.append(value)
                    except Exception:
                        pass

            # Calculate statistics
            metrics["targets"][target] = {
                "D95": {
                    "nominal": d95,
                    "min": min_val,
                    "max": max_val,
                    "variation": max_val - min_val
                    if min_val is not None and max_val is not None
                    else None,
                    "variation_percent": ((max_val - min_val) / d95 * 100)
                    if d95 > 0 and min_val is not None and max_val is not None
                    else None,
                    "values": all_d95,
                }
            }

            # Add more metrics like conformity index, homogeneity index, etc.

        # OAR metrics
        for oar, (min_val, max_val) in self.oar_dose_range.items():
            # Calculate nominal Dmax
            nominal_dvh = self.nominal_scenario.dvh_data.get(oar, {})
            dmax = 0.0
            if nominal_dvh:
                try:
                    dose = nominal_dvh.get("dose", [])
                    if len(dose) > 0:
                        dmax = np.max(dose)
                except Exception:
                    pass

            # Get all Dmax values
            all_dmax = []
            for scenario in [self.nominal_scenario] + self.scenarios:
                scenario_dvh = scenario.dvh_data.get(oar, {})
                if scenario_dvh:
                    try:
                        dose = scenario_dvh.get("dose", [])
                        if len(dose) > 0:
                            value = np.max(dose)
                            all_dmax.append(value)
                    except Exception:
                        pass

            # Calculate statistics
            metrics["oars"][oar] = {
                "Dmax": {
                    "nominal": dmax,
                    "min": min_val,
                    "max": max_val,
                    "variation": max_val - min_val
                    if min_val is not None and max_val is not None
                    else None,
                    "variation_percent": ((max_val - min_val) / dmax * 100)
                    if dmax > 0 and min_val is not None and max_val is not None
                    else None,
                    "values": all_dmax,
                }
            }

            # Add more metrics like mean dose, D1cc, etc.

        return metrics

    def get_spatial_analysis_data(
        self, display_type: str = "dose_difference"
    ) -> Dict[str, Any]:
        """
        Get spatial analysis data for visualization.

        Args:
            display_type: Type of data to display:
                        - "dose_difference": Difference between scenarios
                        - "uncertainty_map": Uncertainty map
                        - "worst_case": Worst case scenario

        Returns:
            Dictionary with spatial analysis data
        """
        # Initialize result
        result = {
            "type": display_type,
            "data": None,
            "colormap": "RdBu_r",  # Default colormap
            "title": "",
            "contours": {},
        }

        try:
            # Get reference dose grid
            nominal_dose = self.nominal_scenario.dose_grid

            if display_type == "dose_difference":
                # Find maximum difference at each voxel
                max_diff = np.zeros_like(nominal_dose)
                min_diff = np.zeros_like(nominal_dose)

                for scenario in self.scenarios:
                    diff = scenario.dose_grid - nominal_dose
                    max_diff = np.maximum(max_diff, diff)
                    min_diff = np.minimum(min_diff, diff)

                # Use the larger of min or max difference (absolute)
                abs_max_diff = np.maximum(np.abs(min_diff), np.abs(max_diff))

                # Create a 2D slice for display (middle slice)
                slice_idx = abs_max_diff.shape[2] // 2
                result["data"] = abs_max_diff[:, :, slice_idx]
                result["title"] = "Maximum Dose Difference (Gy)"

            elif display_type == "uncertainty_map":
                # Calculate standard deviation at each voxel
                all_doses = np.zeros((len(self.scenarios) + 1,) + nominal_dose.shape)
                all_doses[0] = nominal_dose

                for i, scenario in enumerate(self.scenarios):
                    all_doses[i + 1] = scenario.dose_grid

                # Calculate standard deviation
                std_dev = np.std(all_doses, axis=0)

                # Create a 2D slice for display
                slice_idx = std_dev.shape[2] // 2
                result["data"] = std_dev[:, :, slice_idx]
                result["title"] = "Dose Uncertainty (Gy)"
                result["colormap"] = "viridis"

            elif display_type == "worst_case":
                # Find the minimum dose at each voxel for targets
                # and maximum dose for OARs
                worst_case = np.copy(nominal_dose)

                # TODO: This would be more sophisticated with actual
                # structure masks to differentiate targets from OARs

                # For now, just show the minimum dose at each voxel
                for scenario in self.scenarios:
                    worst_case = np.minimum(worst_case, scenario.dose_grid)

                # Create a 2D slice for display
                slice_idx = worst_case.shape[2] // 2
                result["data"] = worst_case[:, :, slice_idx]
                result["title"] = "Worst Case Dose (Gy)"
                result["colormap"] = "jet"

            # Add structure contours for context if available
            # (This is a placeholder - actual implementation would depend on how structures are stored)

            return result

        except Exception as e:
            logger.error(f"Error generating spatial analysis data: {e}")
            return {"type": display_type, "error": str(e), "data": None}

    def export_to_csv(self, filename: str) -> bool:
        """
        Export robustness analysis results to CSV file.

        Args:
            filename: Path to the output CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            import csv

            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(["QuangTPS Robustness Analysis Results"])
                writer.writerow([])

                # Write analysis parameters
                writer.writerow(["Number of scenarios", len(self.scenarios) + 1])
                writer.writerow([])

                # Write target coverage results
                writer.writerow(["Target Coverage Results"])
                writer.writerow(
                    [
                        "Target",
                        "D95 (Nominal)",
                        "D95 (Min)",
                        "D95 (Max)",
                        "Variation",
                        "Variation (%)",
                    ]
                )

                metrics = self.get_evaluation_metrics()
                for target, metric_data in metrics["targets"].items():
                    d95_data = metric_data.get("D95", {})
                    writer.writerow(
                        [
                            target,
                            f"{d95_data.get('nominal', 0):.2f}",
                            f"{d95_data.get('min', 0):.2f}",
                            f"{d95_data.get('max', 0):.2f}",
                            f"{d95_data.get('variation', 0):.2f}",
                            f"{d95_data.get('variation_percent', 0):.2f}%",
                        ]
                    )

                writer.writerow([])

                # Write OAR results
                writer.writerow(["OAR Dose Results"])
                writer.writerow(
                    [
                        "OAR",
                        "Dmax (Nominal)",
                        "Dmax (Min)",
                        "Dmax (Max)",
                        "Variation",
                        "Variation (%)",
                    ]
                )

                for oar, metric_data in metrics["oars"].items():
                    dmax_data = metric_data.get("Dmax", {})
                    writer.writerow(
                        [
                            oar,
                            f"{dmax_data.get('nominal', 0):.2f}",
                            f"{dmax_data.get('min', 0):.2f}",
                            f"{dmax_data.get('max', 0):.2f}",
                            f"{dmax_data.get('variation', 0):.2f}",
                            f"{dmax_data.get('variation_percent', 0):.2f}%",
                        ]
                    )

            logger.info(
                f"Successfully exported robustness analysis results to {filename}"
            )
            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def export_to_excel(self, filename: str) -> bool:
        """
        Export robustness analysis results to Excel file.

        Args:
            filename: Path to the output Excel file

        Returns:
            True if successful, False otherwise
        """
        try:
            import pandas as pd

            # Create a Pandas Excel writer using XlsxWriter as the engine
            with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                # Get metrics
                metrics = self.get_evaluation_metrics()

                # Create summary sheet
                summary_data = {
                    "Parameter": [
                        "Number of Scenarios",
                        "Setup Uncertainty",
                        "Range Uncertainty",
                    ],
                    "Value": [
                        len(self.scenarios) + 1,
                        "N/A",
                        "N/A",
                    ],  # Placeholder values
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                # Create target metrics sheet
                target_data = []
                for target, metrics_dict in metrics["targets"].items():
                    d95 = metrics_dict.get("D95", {})
                    target_data.append(
                        {
                            "Target": target,
                            "D95 (Nominal)": d95.get("nominal", 0),
                            "D95 (Min)": d95.get("min", 0),
                            "D95 (Max)": d95.get("max", 0),
                            "Variation": d95.get("variation", 0),
                            "Variation (%)": d95.get("variation_percent", 0),
                        }
                    )

                if target_data:
                    target_df = pd.DataFrame(target_data)
                    target_df.to_excel(
                        writer, sheet_name="Target Coverage", index=False
                    )

                # Create OAR metrics sheet
                oar_data = []
                for oar, metrics_dict in metrics["oars"].items():
                    dmax = metrics_dict.get("Dmax", {})
                    oar_data.append(
                        {
                            "OAR": oar,
                            "Dmax (Nominal)": dmax.get("nominal", 0),
                            "Dmax (Min)": dmax.get("min", 0),
                            "Dmax (Max)": dmax.get("max", 0),
                            "Variation": dmax.get("variation", 0),
                            "Variation (%)": dmax.get("variation_percent", 0),
                        }
                    )

                if oar_data:
                    oar_df = pd.DataFrame(oar_data)
                    oar_df.to_excel(writer, sheet_name="OAR Doses", index=False)

                # Create scenario details sheet
                scenario_data = []
                for scenario in [self.nominal_scenario] + self.scenarios:
                    row = {"Scenario": scenario.scenario_name}
                    # Add parameters
                    for param, value in scenario.uncertainty_parameters.items():
                        row[param] = value
                    scenario_data.append(row)

                if scenario_data:
                    scenario_df = pd.DataFrame(scenario_data)
                    scenario_df.to_excel(writer, sheet_name="Scenarios", index=False)

            logger.info(
                f"Successfully exported robustness analysis results to {filename}"
            )
            return True

        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return False

    def create_pdf_report(self, filename: str, plan: Optional[Plan] = None) -> bool:
        """
        Create a PDF report of the robustness analysis.

        Args:
            filename: Path to the output PDF file
            plan: Optional plan info to include in the report

        Returns:
            True if successful, False otherwise
        """
        try:
            from fpdf import FPDF
            import matplotlib.pyplot as plt
            import tempfile
            import os

            # Create PDF object
            pdf = FPDF()
            pdf.add_page()

            # Add header
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "QuangTPS Robustness Analysis Report", 0, 1, "C")
            pdf.ln(4)

            # Add plan info if available
            if plan:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Plan Information", 0, 1, "L")

                pdf.set_font("Arial", "", 10)
                pdf.cell(
                    0,
                    6,
                    f"Plan Name: {plan.name if hasattr(plan, 'name') else 'N/A'}",
                    0,
                    1,
                )
                pdf.cell(
                    0,
                    6,
                    f"Patient ID: {plan.patient_id if hasattr(plan, 'patient_id') else 'N/A'}",
                    0,
                    1,
                )
                pdf.cell(
                    0,
                    6,
                    f"Number of Beams: {len(plan.beams) if hasattr(plan, 'beams') else 'N/A'}",
                    0,
                    1,
                )
                pdf.ln(4)

            # Add robustness parameters
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Robustness Analysis Parameters", 0, 1, "L")

            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"Number of Scenarios: {len(self.scenarios) + 1}", 0, 1)
            # Add more parameters here
            pdf.ln(4)

            # Add target coverage results
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Target Coverage Results", 0, 1, "L")

            metrics = self.get_evaluation_metrics()

            # Create a table
            pdf.set_font("Arial", "B", 9)
            col_width = 30
            pdf.cell(col_width, 7, "Target", 1, 0, "C")
            pdf.cell(col_width, 7, "D95 (Nominal)", 1, 0, "C")
            pdf.cell(col_width, 7, "D95 (Min)", 1, 0, "C")
            pdf.cell(col_width, 7, "D95 (Max)", 1, 0, "C")
            pdf.cell(col_width, 7, "Variation (%)", 1, 1, "C")

            pdf.set_font("Arial", "", 9)
            for target, metric_data in metrics["targets"].items():
                d95_data = metric_data.get("D95", {})
                pdf.cell(col_width, 7, target, 1, 0)
                pdf.cell(col_width, 7, f"{d95_data.get('nominal', 0):.2f}", 1, 0, "C")
                pdf.cell(col_width, 7, f"{d95_data.get('min', 0):.2f}", 1, 0, "C")
                pdf.cell(col_width, 7, f"{d95_data.get('max', 0):.2f}", 1, 0, "C")
                pdf.cell(
                    col_width,
                    7,
                    f"{d95_data.get('variation_percent', 0):.2f}%",
                    1,
                    1,
                    "C",
                )

            pdf.ln(4)

            # Add OAR results
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "OAR Dose Results", 0, 1, "L")

            # Create a table
            pdf.set_font("Arial", "B", 9)
            pdf.cell(col_width, 7, "OAR", 1, 0, "C")
            pdf.cell(col_width, 7, "Dmax (Nominal)", 1, 0, "C")
            pdf.cell(col_width, 7, "Dmax (Min)", 1, 0, "C")
            pdf.cell(col_width, 7, "Dmax (Max)", 1, 0, "C")
            pdf.cell(col_width, 7, "Variation (%)", 1, 1, "C")

            pdf.set_font("Arial", "", 9)
            for oar, metric_data in metrics["oars"].items():
                dmax_data = metric_data.get("Dmax", {})
                pdf.cell(col_width, 7, oar, 1, 0)
                pdf.cell(col_width, 7, f"{dmax_data.get('nominal', 0):.2f}", 1, 0, "C")
                pdf.cell(col_width, 7, f"{dmax_data.get('min', 0):.2f}", 1, 0, "C")
                pdf.cell(col_width, 7, f"{dmax_data.get('max', 0):.2f}", 1, 0, "C")
                pdf.cell(
                    col_width,
                    7,
                    f"{dmax_data.get('variation_percent', 0):.2f}%",
                    1,
                    1,
                    "C",
                )

            # Add DVH plots
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Create and save DVH plots
                for target in metrics["targets"]:
                    plt.figure(figsize=(8, 6))
                    self.plot_dvh_band(target)
                    plot_path = os.path.join(tmpdirname, f"{target}_dvh.png")
                    plt.savefig(plot_path)
                    plt.close()

                    # Add new page for each plot
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, f"DVH Band for {target}", 0, 1, "C")
                    pdf.image(plot_path, x=10, y=30, w=180)

            # Save PDF
            pdf.output(filename)

            logger.info(f"Successfully created PDF report at {filename}")
            return True

        except Exception as e:
            logger.error(f"Error creating PDF report: {e}")
            return False

    def create_html_report(self, filename: str, plan: Optional[Plan] = None) -> bool:
        """
        Create an HTML report of the robustness analysis.

        Args:
            filename: Path to the output HTML file
            plan: Optional plan info to include in the report

        Returns:
            True if successful, False otherwise
        """
        try:
            import base64
            import io
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

            metrics = self.get_evaluation_metrics()

            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>QuangTPS Robustness Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #f2f2f2; }}
                    .plot-container {{ margin: 20px 0; }}
                    .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                <h1>QuangTPS Robustness Analysis Report</h1>
            """

            # Add plan info if available
            if plan:
                html_content += f"""
                <div class="summary">
                    <h2>Plan Information</h2>
                    <p><strong>Plan Name:</strong> {plan.name if hasattr(plan, "name") else "N/A"}</p>
                    <p><strong>Patient ID:</strong> {plan.patient_id if hasattr(plan, "patient_id") else "N/A"}</p>
                    <p><strong>Number of Beams:</strong> {len(plan.beams) if hasattr(plan, "beams") else "N/A"}</p>
                </div>
                """

            # Add robustness parameters
            html_content += f"""
            <div class="summary">
                <h2>Robustness Analysis Parameters</h2>
                <p><strong>Number of Scenarios:</strong> {len(self.scenarios) + 1}</p>
                <!-- Add more parameters here -->
            </div>
            """

            # Add target coverage results
            html_content += """
            <h2>Target Coverage Results</h2>
            <table>
                <tr>
                    <th>Target</th>
                    <th>D95 (Nominal)</th>
                    <th>D95 (Min)</th>
                    <th>D95 (Max)</th>
                    <th>Variation</th>
                    <th>Variation (%)</th>
                </tr>
            """

            for target, metric_data in metrics["targets"].items():
                d95_data = metric_data.get("D95", {})
                html_content += f"""
                <tr>
                    <td>{target}</td>
                    <td>{d95_data.get("nominal", 0):.2f}</td>
                    <td>{d95_data.get("min", 0):.2f}</td>
                    <td>{d95_data.get("max", 0):.2f}</td>
                    <td>{d95_data.get("variation", 0):.2f}</td>
                    <td>{d95_data.get("variation_percent", 0):.2f}%</td>
                </tr>
                """

            html_content += """
            </table>
            """

            # Add OAR results
            html_content += """
            <h2>OAR Dose Results</h2>
            <table>
                <tr>
                    <th>OAR</th>
                    <th>Dmax (Nominal)</th>
                    <th>Dmax (Min)</th>
                    <th>Dmax (Max)</th>
                    <th>Variation</th>
                    <th>Variation (%)</th>
                </tr>
            """

            for oar, metric_data in metrics["oars"].items():
                dmax_data = metric_data.get("Dmax", {})
                html_content += f"""
                <tr>
                    <td>{oar}</td>
                    <td>{dmax_data.get("nominal", 0):.2f}</td>
                    <td>{dmax_data.get("min", 0):.2f}</td>
                    <td>{dmax_data.get("max", 0):.2f}</td>
                    <td>{dmax_data.get("variation", 0):.2f}</td>
                    <td>{dmax_data.get("variation_percent", 0):.2f}%</td>
                </tr>
                """

            html_content += """
            </table>
            """

            # Add DVH plots
            html_content += "<h2>DVH Bands</h2>"

            for target in metrics["targets"]:
                # Create plot and convert to base64 for embedding
                fig = plt.figure(figsize=(10, 6))
                self.plot_dvh_band(target)

                buf = io.BytesIO()
                FigureCanvas(fig).print_png(buf)
                img_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                plt.close(fig)

                html_content += f"""
                <div class="plot-container">
                    <h3>DVH Band for {target}</h3>
                    <img src="data:image/png;base64,{img_data}" alt="DVH Band for {target}" width="800">
                </div>
                """

            # Close HTML document
            html_content += """
            </body>
            </html>
            """

            # Write to file
            with open(filename, "w") as f:
                f.write(html_content)

            logger.info(f"Successfully created HTML report at {filename}")
            return True

        except Exception as e:
            logger.error(f"Error creating HTML report: {e}")
            return False


class RobustnessAnalyzer:
    """
    Analyzer for treatment plan robustness against uncertainties.

    This class provides methods to evaluate the robustness of treatment plans
    against various uncertainties such as setup errors and range uncertainties.
    """

    def __init__(
        self,
        plan: Plan,
        structures: Dict[str, Structure],
        dose_grid: Optional[DoseGrid] = None,
        setup_uncertainty: float = 3.0,
        range_uncertainty: float = 3.5,
        num_scenarios: int = 7,
    ):
        """
        Initialize robustness analyzer.

        Args:
            plan: The treatment plan to analyze
            structures: Dictionary of structures (name -> Structure object)
            dose_grid: Optional dose grid for the plan (will be calculated if not provided)
            setup_uncertainty: Setup uncertainty in mm
            range_uncertainty: Range uncertainty in percent
            num_scenarios: Number of scenarios to generate
        """
        self.plan = plan
        self.structures = structures
        self.dose_grid = dose_grid
        self.setup_uncertainty = setup_uncertainty
        self.range_uncertainty = range_uncertainty
        self.num_scenarios = num_scenarios

        # Initialize target and OAR lists
        self.target_names = []
        self.oar_names = []
        self._initialize_structure_lists()

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

        self.dose_calculator = DoseCalculator()

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
        Phân tích độ bền vững của kế hoạch đối với các yếu tố không chắc chắn.

        Phương thức này phân tích sự ảnh hưởng của các yếu tố không chắc chắn (setup và range)
        lên kế hoạch điều trị bằng cách tạo và đánh giá nhiều kịch bản khác nhau. Kết quả cho thấy
        khả năng kế hoạch duy trì hiệu quả khi có sai số về setup hoặc range.

        Returns
        -------
        RobustnessResult
            Kết quả phân tích độ bền vững, bao gồm thông tin về tất cả các kịch bản đã phân tích
        """
        logger.info("Bắt đầu phân tích độ bền vững cho kế hoạch")
        start_time = time.time()

        try:
            # Tạo các kịch bản để phân tích
            scenarios = self._generate_scenarios()
            logger.info(f"Đã tạo {len(scenarios)} kịch bản để phân tích")

            # Khởi tạo containers cho kết quả
            scenario_results = []
            nominal_result = None

            # Xử lý kịch bản danh nghĩa (nominal) trước
            try:
                nominal_params = scenarios.pop("nominal")
                nominal_dvh = self._calculate_dvh_for_scenario(self.dose_grid)

                nominal_result = ScenarioResult(
                    scenario_name="nominal",
                    uncertainty_parameters=nominal_params,
                    dose_grid=self.dose_grid,
                    dvh_data=nominal_dvh,
                )

                # Tính toán độ bao phủ mục tiêu và liều OAR cho kịch bản danh nghĩa
                nominal_coverage = self._calculate_target_coverage(nominal_dvh)
                nominal_oar_doses = self._calculate_oar_doses(nominal_dvh)

                # Khởi tạo giá trị min/max bằng giá trị danh nghĩa
                min_coverage = {t: v for t, v in nominal_coverage.items()}
                max_coverage = {t: v for t, v in nominal_coverage.items()}
                min_oar_doses = {o: v for o, v in nominal_oar_doses.items()}
                max_oar_doses = {o: v for o, v in nominal_oar_doses.items()}

                logger.info(
                    f"Đã xử lý kịch bản danh nghĩa: Target coverage = {nominal_coverage}"
                )
            except Exception as e:
                logger.error(f"Lỗi khi xử lý kịch bản danh nghĩa: {str(e)}")
                # Tiếp tục với các kịch bản khác nếu kịch bản danh nghĩa thất bại

            # Xử lý các kịch bản phân tích độ bền vững
            successful_scenarios = 0
            failed_scenarios = 0

            # Sử dụng ThreadPoolExecutor để tính toán song song (nếu có thể)
            with ThreadPoolExecutor(
                max_workers=min(8, os.cpu_count() or 4)
            ) as executor:
                # Submit các tác vụ tính toán và theo dõi kết quả
                future_to_scenario = {}
                for scenario_name, params in scenarios.items():
                    future = executor.submit(
                        self._process_scenario, scenario_name, params
                    )
                    future_to_scenario[future] = scenario_name

                # Xử lý kết quả khi hoàn thành
                for future in as_completed(future_to_scenario):
                    scenario_name = future_to_scenario[future]
                    try:
                        result = future.result()
                        if result:
                            scenario_results.append(result)

                            # Cập nhật min/max coverage và liều OAR
                            scenario_coverage = self._calculate_target_coverage(
                                result.dvh_data
                            )
                            scenario_oar_doses = self._calculate_oar_doses(
                                result.dvh_data
                            )

                            for target, coverage in scenario_coverage.items():
                                if target in min_coverage:
                                    min_coverage[target] = min(
                                        min_coverage[target], coverage
                                    )
                                    max_coverage[target] = max(
                                        max_coverage[target], coverage
                                    )
                                else:
                                    min_coverage[target] = coverage
                                    max_coverage[target] = coverage

                            for oar, dose in scenario_oar_doses.items():
                                if oar in min_oar_doses:
                                    min_oar_doses[oar] = min(min_oar_doses[oar], dose)
                                    max_oar_doses[oar] = max(max_oar_doses[oar], dose)
                                else:
                                    min_oar_doses[oar] = dose
                                    max_oar_doses[oar] = dose

                            successful_scenarios += 1
                            logger.debug(
                                f"Kịch bản {scenario_name} đã hoàn thành thành công"
                            )
                        else:
                            failed_scenarios += 1
                            logger.warning(
                                f"Kịch bản {scenario_name} không trả về kết quả"
                            )
                    except Exception as e:
                        failed_scenarios += 1
                        logger.error(
                            f"Lỗi khi xử lý kịch bản {scenario_name}: {str(e)}"
                        )

            # Tạo kết quả tổng hợp
            elapsed_time = time.time() - start_time
            logger.info(
                f"Phân tích độ bền vững hoàn thành trong {elapsed_time:.2f} giây"
            )
            logger.info(
                f"Số kịch bản thành công: {successful_scenarios}, thất bại: {failed_scenarios}"
            )

            if not nominal_result and not scenario_results:
                logger.error(
                    "Không thể hoàn thành bất kỳ kịch bản nào, không có kết quả phân tích"
                )
                return RobustnessResult(
                    nominal=None,
                    scenarios=[],
                    min_target_coverage={},
                    max_target_coverage={},
                    min_oar_doses={},
                    max_oar_doses={},
                )

            result = RobustnessResult(
                nominal=nominal_result,
                scenarios=scenario_results,
                min_target_coverage=min_coverage,
                max_target_coverage=max_coverage,
                min_oar_doses=min_oar_doses,
                max_oar_doses=max_oar_doses,
            )

            # Tạo tóm tắt kết quả
            self._log_robustness_summary(result)

            return result

        except Exception as e:
            logger.error(f"Lỗi khi phân tích độ bền vững: {str(e)}")
            # Trả về kết quả trống trong trường hợp lỗi
            return RobustnessResult(
                nominal=None,
                scenarios=[],
                min_target_coverage={},
                max_target_coverage={},
                min_oar_doses={},
                max_oar_doses={},
            )

    def _process_scenario(self, scenario_name, params):
        """
        Xử lý một kịch bản đánh giá độ bền vững.

        Parameters
        ----------
        scenario_name : str
            Tên kịch bản
        params : dict
            Tham số kịch bản

        Returns
        -------
        ScenarioResult hoặc None
            Kết quả kịch bản hoặc None nếu có lỗi
        """
        try:
            logger.debug(f"Đang xử lý kịch bản: {scenario_name}")

            # Tính liều cho kịch bản
            dose_grid = self._calculate_dose_for_scenario(scenario_name, params)

            if dose_grid is None:
                logger.warning(f"Không thể tính toán liều cho kịch bản {scenario_name}")
                return None

            # Tính DVH cho kịch bản
            dvh_data = self._calculate_dvh_for_scenario(dose_grid)

            # Tạo kết quả kịch bản
            result = ScenarioResult(
                scenario_name=scenario_name,
                uncertainty_parameters=params,
                dose_grid=dose_grid,
                dvh_data=dvh_data,
            )

            return result

        except Exception as e:
            logger.error(f"Lỗi trong kịch bản {scenario_name}: {str(e)}")
            return None

    def _log_robustness_summary(self, result):
        """
        Tạo và ghi log tóm tắt kết quả phân tích độ bền vững.

        Parameters
        ----------
        result : RobustnessResult
            Kết quả phân tích độ bền vững
        """
        logger.info("=== TÓM TẮT PHÂN TÍCH ĐỘ BỀN VỮNG ===")

        # Tóm tắt độ bao phủ mục tiêu
        logger.info("Độ bao phủ mục tiêu:")
        for target, min_cov in result.min_target_coverage.items():
            max_cov = result.max_target_coverage.get(target, min_cov)
            nominal_cov = 0
            if result.nominal and target in result.nominal.dvh_data:
                nominal_dvh = result.nominal.dvh_data[target]
                nominal_cov = self._calculate_d95(nominal_dvh)

            logger.info(
                f"  {target}: {min_cov:.2f}% - {nominal_cov:.2f}% - {max_cov:.2f}% (Min-Nominal-Max)"
            )

        # Tóm tắt liều OAR
        logger.info("Liều OAR (D1cc):")
        for oar, min_dose in result.min_oar_doses.items():
            max_dose = result.max_oar_doses.get(oar, min_dose)
            nominal_dose = 0
            if result.nominal and oar in result.nominal.dvh_data:
                nominal_dvh = result.nominal.dvh_data[oar]
                nominal_dose = self._calculate_d1cc(nominal_dvh)

            logger.info(
                f"  {oar}: {min_dose:.2f} - {nominal_dose:.2f} - {max_dose:.2f} Gy (Min-Nominal-Max)"
            )

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
