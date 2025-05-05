#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Quality Module
==================

This module provides functionality for evaluating the quality of treatment plans
based on clinical goals, DVH metrics, and other evaluation criteria. It implements
features similar to Eclipse's Plan Evaluation tools.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import time
from datetime import datetime

# Import QuangTPS modules
try:
    from quangtps.evaluation.clinical_protocols import ClinicalProtocol
    from quangtps.evaluation.protocol_manager import ProtocolManager
    from quangtps.evaluation.dvh.dvh_analysis import DVHAnalyzer
    from quangtps.evaluation.metrics.conformity import ConformityIndex
    from quangtps.evaluation.metrics.homogeneity import HomogeneityIndex
    from quangtps.evaluation.metrics.gradient import GradientIndex
    from quangtps.planning.plan import Plan
    from quangtps.evaluation.clinical_goals import (
        ClinicalGoal,
        ClinicalGoalCollection,
        ClinicalGoalTemplate,
        ClinicalGoalManager,
        GoalType,
        GoalOperator,
        GoalPriority,
        GoalResult,
    )

    EVALUATION_MODULES_AVAILABLE = True
except ImportError:
    logging.warning("Failed to import QuangTPS evaluation modules")
    EVALUATION_MODULES_AVAILABLE = False

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class PlanQualityMetric(Enum):
    """Types of plan quality metrics."""

    CONFORMITY_INDEX = 1
    HOMOGENEITY_INDEX = 2
    GRADIENT_INDEX = 3
    COVERAGE = 4
    HOTSPOT = 5
    COLD_SPOT = 6
    INTEGRAL_DOSE = 7
    NORMAL_TISSUE_SPARING = 8


class PlanQualityScore(Enum):
    """Scoring levels for plan quality assessment."""

    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    UNACCEPTABLE = 1
    NOT_APPLICABLE = 0


class PlanQualityEvaluator:
    """
    Evaluator for comprehensive plan quality assessment.

    This class provides methods for evaluating various aspects of plan quality
    and generating a comprehensive report, similar to Eclipse's evaluation tools.
    """

    def __init__(self, plan=None, protocol=None, dvh_analyzer=None):
        """
        Initialize the plan quality evaluator.

        Parameters
        ----------
        plan : Plan, optional
            Treatment plan to evaluate
        protocol : ClinicalProtocol, optional
            Clinical protocol with goals
        dvh_analyzer : DVHAnalyzer, optional
            DVH analyzer for the plan
        """
        self.plan = plan
        self.protocol = protocol
        self.dvh_analyzer = dvh_analyzer
        self.metrics = {}
        self.scores = {}
        self.results = {}
        self.overall_score = None
        self.evaluation_time = None
        self.evaluation_timestamp = None
        self.clinical_goals = ClinicalGoalCollection()
        self._protocol_manager = None

        # Initialize from protocol if provided
        if protocol and hasattr(protocol, "goals"):
            self.clinical_goals.add_goals(protocol.goals)

    def set_plan(self, plan):
        """
        Set the treatment plan for evaluation.

        Parameters
        ----------
        plan : Plan
            Treatment plan
        """
        self.plan = plan

    def set_protocol(self, protocol):
        """
        Set the clinical protocol for evaluation.

        Parameters
        ----------
        protocol : ClinicalProtocol or str
            Clinical protocol or protocol name
        """
        # Handle protocol by name
        if isinstance(protocol, str):
            if self._protocol_manager is None:
                self._protocol_manager = ProtocolManager()

            protocol_obj = self._protocol_manager.get_protocol(protocol)
            if protocol_obj is None:
                logger.error(f"Protocol '{protocol}' not found")
                return
            protocol = protocol_obj

        self.protocol = protocol

        # Update clinical goals from protocol
        if hasattr(protocol, "goals"):
            self.clinical_goals = ClinicalGoalCollection()
            self.clinical_goals.add_goals(protocol.goals)

    def set_dvh_analyzer(self, dvh_analyzer):
        """
        Set the DVH analyzer for evaluation.

        Parameters
        ----------
        dvh_analyzer : DVHAnalyzer
            DVH analyzer for the plan
        """
        self.dvh_analyzer = dvh_analyzer

    def evaluate(self):
        """
        Perform comprehensive plan quality evaluation.

        Returns
        -------
        dict
            Evaluation results
        """
        if not self.plan:
            logger.error("No plan set for evaluation")
            return None

        if not self.dvh_analyzer:
            logger.error("No DVH analyzer set for evaluation")
            return None

        # Record evaluation start time
        start_time = time.time()

        # Reset results
        self.metrics = {}
        self.scores = {}
        self.results = {}
        self.overall_score = None

        # Evaluate clinical goals
        self._evaluate_clinical_goals()

        # Evaluate PTV coverage and conformity
        self._evaluate_target_coverage()

        # Evaluate hotspots
        self._evaluate_hotspots()

        # Evaluate normal tissue sparing
        self._evaluate_normal_tissue_sparing()

        # Calculate overall score
        self._calculate_overall_score()

        # Compile results
        self._compile_results()

        # Record evaluation time and timestamp
        self.evaluation_time = time.time() - start_time
        self.evaluation_timestamp = datetime.now()

        # Add evaluation metadata
        self.results["evaluation"] = {
            "timestamp": self.evaluation_timestamp.isoformat(),
            "execution_time": self.evaluation_time,
            "protocol": self.protocol.name if self.protocol else None,
        }

        logger.info(
            f"Plan quality evaluation completed in {self.evaluation_time:.2f} seconds"
        )
        return self.results

    def _evaluate_clinical_goals(self):
        """Evaluate clinical goals."""
        if (
            not self.clinical_goals
            or not hasattr(self.clinical_goals, "goals")
            or not self.clinical_goals.goals
        ):
            logger.warning("No clinical goals set for evaluation")
            self.scores["clinical_goals"] = PlanQualityScore.NOT_APPLICABLE
            self.results["clinical_goals"] = {
                "total": 0,
                "passed": 0,
                "warning": 0,
                "failed": 0,
                "not_applicable": 0,
                "goals": [],
            }
            return

        try:
            # Evaluate goals using DVH analyzer
            goal_results = self.clinical_goals.evaluate(self.dvh_analyzer)

            # Store results
            self.results["clinical_goals"] = goal_results

            # Calculate score based on goal compliance
            total_goals = goal_results["total"]
            if total_goals == 0:
                self.scores["clinical_goals"] = PlanQualityScore.NOT_APPLICABLE
                return

            # Weight by priority
            weighted_score = 0
            total_weight = 0

            for goal_result in goal_results["goals"]:
                priority = goal_result["priority"]
                result = goal_result["result"]

                # Assign weight based on priority
                if priority == GoalPriority.CRITICAL.value:
                    weight = 3.0
                elif priority == GoalPriority.MAJOR.value:
                    weight = 2.0
                else:  # MINOR
                    weight = 1.0

                # Assign score based on result
                if result == GoalResult.PASSED.value:
                    score = 5  # Excellent
                elif result == GoalResult.WARNING.value:
                    score = 3  # Acceptable
                elif result == GoalResult.FAILED.value:
                    score = 1  # Unacceptable
                else:  # NOT_APPLICABLE
                    continue  # Skip in calculation

                weighted_score += score * weight
                total_weight += weight

            if total_weight > 0:
                average_score = weighted_score / total_weight

                # Map average score to quality score
                if average_score >= 4.5:
                    self.scores["clinical_goals"] = PlanQualityScore.EXCELLENT
                elif average_score >= 3.5:
                    self.scores["clinical_goals"] = PlanQualityScore.GOOD
                elif average_score >= 2.5:
                    self.scores["clinical_goals"] = PlanQualityScore.ACCEPTABLE
                elif average_score >= 1.5:
                    self.scores["clinical_goals"] = PlanQualityScore.POOR
                else:
                    self.scores["clinical_goals"] = PlanQualityScore.UNACCEPTABLE
            else:
                self.scores["clinical_goals"] = PlanQualityScore.NOT_APPLICABLE

            # Cache the average score for detailed reporting
            self.metrics["clinical_goals_score"] = (
                average_score if total_weight > 0 else 0
            )

        except Exception as e:
            logger.error(f"Error evaluating clinical goals: {e}")
            self.scores["clinical_goals"] = PlanQualityScore.NOT_APPLICABLE

    def _evaluate_target_coverage(self):
        """Evaluate PTV coverage metrics."""
        try:
            # Get prescription and targets
            prescription = (
                self.plan.prescription if hasattr(self.plan, "prescription") else None
            )
            if (
                not prescription
                or not hasattr(prescription, "targets")
                or not prescription.targets
            ):
                logger.warning("No prescription or targets found in plan")
                self.scores["target_coverage"] = PlanQualityScore.NOT_APPLICABLE
                return

            target_metrics = {}
            target_scores = {}

            for target in prescription.targets:
                target_id = target.structure_id
                target_name = target.structure_name
                rx_dose = target.dose_level.dose

                # Skip if target is not in DVH data
                if not self.dvh_analyzer.has_structure(target_id):
                    continue

                # Calculate coverage metrics
                try:
                    # D95
                    d95 = self.dvh_analyzer.get_dose_at_volume(target_id, 95)
                    d95_percent = (d95 / rx_dose) * 100 if rx_dose > 0 else 0

                    # V95
                    v95 = self.dvh_analyzer.get_volume_at_dose(
                        target_id, 0.95 * rx_dose
                    )

                    # Conformity index
                    ci = self.dvh_analyzer.get_conformity_index(target_id, rx_dose)

                    # Homogeneity index
                    hi = self.dvh_analyzer.get_homogeneity_index(target_id)

                    # Gradient index
                    gi = self.dvh_analyzer.get_gradient_index(target_id, rx_dose)

                    # Maximum dose
                    dmax = self.dvh_analyzer.get_max_dose(target_id)
                    dmax_percent = (dmax / rx_dose) * 100 if rx_dose > 0 else 0

                    # Minimum dose
                    dmin = self.dvh_analyzer.get_min_dose(target_id)
                    dmin_percent = (dmin / rx_dose) * 100 if rx_dose > 0 else 0

                    # Store metrics
                    target_metrics[target_id] = {
                        "name": target_name,
                        "rx_dose": rx_dose,
                        "d95": d95,
                        "d95_percent": d95_percent,
                        "v95": v95,
                        "conformity_index": ci,
                        "homogeneity_index": hi,
                        "gradient_index": gi,
                        "dmax": dmax,
                        "dmax_percent": dmax_percent,
                        "dmin": dmin,
                        "dmin_percent": dmin_percent,
                    }

                    # Calculate score for this target
                    coverage_score = self._score_target_coverage(
                        d95_percent, v95, ci, hi, gi
                    )
                    target_scores[target_id] = coverage_score

                except Exception as e:
                    logger.error(
                        f"Error calculating metrics for target {target_id}: {e}"
                    )
                    target_scores[target_id] = PlanQualityScore.NOT_APPLICABLE

            # Store metrics and scores
            self.metrics["target_coverage"] = target_metrics

            # Calculate overall target coverage score
            if target_scores:
                # Weight scores by target volume
                weighted_score = 0
                total_weight = 0

                for target_id, score in target_scores.items():
                    if score == PlanQualityScore.NOT_APPLICABLE:
                        continue

                    # Use volume as weight
                    volume = self.dvh_analyzer.get_structure_volume(target_id)
                    if volume <= 0:
                        volume = 1  # Default weight

                    weighted_score += score.value * volume
                    total_weight += volume

                if total_weight > 0:
                    avg_score = weighted_score / total_weight

                    # Map to score enum
                    if avg_score >= 4.5:
                        self.scores["target_coverage"] = PlanQualityScore.EXCELLENT
                    elif avg_score >= 3.5:
                        self.scores["target_coverage"] = PlanQualityScore.GOOD
                    elif avg_score >= 2.5:
                        self.scores["target_coverage"] = PlanQualityScore.ACCEPTABLE
                    elif avg_score >= 1.5:
                        self.scores["target_coverage"] = PlanQualityScore.POOR
                    else:
                        self.scores["target_coverage"] = PlanQualityScore.UNACCEPTABLE
                else:
                    self.scores["target_coverage"] = PlanQualityScore.NOT_APPLICABLE
            else:
                self.scores["target_coverage"] = PlanQualityScore.NOT_APPLICABLE

        except Exception as e:
            logger.error(f"Error evaluating target coverage: {e}")
            self.scores["target_coverage"] = PlanQualityScore.NOT_APPLICABLE

    def _score_target_coverage(self, d95_percent, v95, ci, hi, gi):
        """
        Score target coverage based on key metrics.

        Parameters
        ----------
        d95_percent : float
            D95 as percentage of prescription dose
        v95 : float
            Volume receiving 95% of prescription dose
        ci : float
            Conformity index
        hi : float
            Homogeneity index
        gi : float
            Gradient index

        Returns
        -------
        PlanQualityScore
            Score for target coverage
        """
        # Calculate sub-scores (0-5 scale)
        d95_score = (
            5
            if d95_percent >= 99
            else (
                4
                if d95_percent >= 97
                else (3 if d95_percent >= 95 else (2 if d95_percent >= 90 else 1))
            )
        )

        v95_score = (
            5
            if v95 >= 99
            else (4 if v95 >= 98 else (3 if v95 >= 95 else (2 if v95 >= 90 else 1)))
        )

        ci_score = (
            5
            if 0.95 <= ci <= 1.05
            else (
                4
                if 0.9 <= ci <= 1.1
                else (3 if 0.85 <= ci <= 1.15 else (2 if 0.8 <= ci <= 1.2 else 1))
            )
        )

        hi_score = (
            5
            if hi >= 0.95
            else (4 if hi >= 0.9 else (3 if hi >= 0.85 else (2 if hi >= 0.8 else 1)))
        )

        # Calculate weighted average
        weights = {"d95": 0.35, "v95": 0.35, "ci": 0.2, "hi": 0.1}

        avg_score = (
            d95_score * weights["d95"]
            + v95_score * weights["v95"]
            + ci_score * weights["ci"]
            + hi_score * weights["hi"]
        )

        # Map to score enum
        if avg_score >= 4.5:
            return PlanQualityScore.EXCELLENT
        elif avg_score >= 3.5:
            return PlanQualityScore.GOOD
        elif avg_score >= 2.5:
            return PlanQualityScore.ACCEPTABLE
        elif avg_score >= 1.5:
            return PlanQualityScore.POOR
        else:
            return PlanQualityScore.UNACCEPTABLE

    def _evaluate_hotspots(self):
        """Evaluate dose hotspots."""
        try:
            # Get prescription and targets
            prescription = (
                self.plan.prescription if hasattr(self.plan, "prescription") else None
            )
            if (
                not prescription
                or not hasattr(prescription, "targets")
                or not prescription.targets
            ):
                logger.warning("No prescription or targets found in plan")
                self.scores["hotspots"] = PlanQualityScore.NOT_APPLICABLE
                return

            # Get main target and its prescription dose
            target = prescription.targets[0]
            rx_dose = target.dose_level.dose

            # Get body structure
            body_id = None
            if hasattr(self.plan, "structure_set") and self.plan.structure_set:
                for structure in self.plan.structure_set.structures:
                    if structure.name.lower() in ["body", "patient", "external"]:
                        body_id = structure.id
                        break

            if not body_id or not self.dvh_analyzer.has_structure(body_id):
                logger.warning("No body structure found for hotspot evaluation")
                self.scores["hotspots"] = PlanQualityScore.NOT_APPLICABLE
                return

            # Calculate maximum dose in body
            max_dose = self.dvh_analyzer.get_max_dose(body_id)

            # Calculate global hotspot as percentage of Rx
            global_hotspot = max_dose / rx_dose if rx_dose > 0 else 0

            # Store metric
            self.metrics["hotspots"] = {
                "global_hotspot": global_hotspot,
                "max_dose": max_dose,
                "rx_dose": rx_dose,
            }

            # Score hotspot (lower is better)
            if global_hotspot <= 1.05:
                self.scores["hotspots"] = PlanQualityScore.EXCELLENT
            elif global_hotspot <= 1.10:
                self.scores["hotspots"] = PlanQualityScore.GOOD
            elif global_hotspot <= 1.15:
                self.scores["hotspots"] = PlanQualityScore.ACCEPTABLE
            elif global_hotspot <= 1.20:
                self.scores["hotspots"] = PlanQualityScore.POOR
            else:
                self.scores["hotspots"] = PlanQualityScore.UNACCEPTABLE

        except Exception as e:
            logger.error(f"Error evaluating hotspots: {e}")
            self.scores["hotspots"] = PlanQualityScore.NOT_APPLICABLE

    def _evaluate_normal_tissue_sparing(self):
        """Evaluate normal tissue sparing."""
        try:
            # Get prescription and targets
            prescription = (
                self.plan.prescription if hasattr(self.plan, "prescription") else None
            )
            if (
                not prescription
                or not hasattr(prescription, "targets")
                or not prescription.targets
            ):
                logger.warning("No prescription or targets found in plan")
                self.scores["normal_tissue"] = PlanQualityScore.NOT_APPLICABLE
                return

            # Get main target and its prescription dose
            target = prescription.targets[0]
            target_id = target.structure_id
            rx_dose = target.dose_level.dose

            # Get body structure
            body_id = None
            if hasattr(self.plan, "structure_set") and self.plan.structure_set:
                for structure in self.plan.structure_set.structures:
                    if structure.name.lower() in ["body", "patient", "external"]:
                        body_id = structure.id
                        break

            if (
                not body_id
                or not self.dvh_analyzer.has_structure(body_id)
                or not self.dvh_analyzer.has_structure(target_id)
            ):
                logger.warning(
                    "Body or target structure not found for normal tissue evaluation"
                )
                self.scores["normal_tissue"] = PlanQualityScore.NOT_APPLICABLE
                return

            # Create normal tissue structure (body minus target)
            # In practice, we'll estimate this from the DVH data

            # Calculate integral dose to body
            body_mean_dose = self.dvh_analyzer.get_mean_dose(body_id)

            # Calculate target volume and mean dose
            target_volume = self.dvh_analyzer.get_structure_volume(target_id)
            target_mean_dose = self.dvh_analyzer.get_mean_dose(target_id)

            # Calculate body volume
            body_volume = self.dvh_analyzer.get_structure_volume(body_id)

            # Estimate normal tissue volume and integral dose
            normal_volume = body_volume - target_volume
            if normal_volume <= 0:
                logger.warning("Invalid normal tissue volume calculation")
                self.scores["normal_tissue"] = PlanQualityScore.NOT_APPLICABLE
                return

            # Estimate normal tissue mean dose (from integral dose)
            body_integral = body_mean_dose * body_volume
            target_integral = target_mean_dose * target_volume
            normal_integral = body_integral - target_integral
            normal_mean_dose = normal_integral / normal_volume

            # Calculate normal tissue dose ratio (relative to Rx)
            normal_dose_ratio = normal_mean_dose / rx_dose if rx_dose > 0 else 0

            # Store metrics
            self.metrics["normal_tissue"] = {
                "normal_mean_dose": normal_mean_dose,
                "normal_dose_ratio": normal_dose_ratio,
                "normal_volume": normal_volume,
                "rx_dose": rx_dose,
            }

            # Score normal tissue sparing (lower is better)
            if normal_dose_ratio <= 0.05:
                self.scores["normal_tissue"] = PlanQualityScore.EXCELLENT
            elif normal_dose_ratio <= 0.10:
                self.scores["normal_tissue"] = PlanQualityScore.GOOD
            elif normal_dose_ratio <= 0.15:
                self.scores["normal_tissue"] = PlanQualityScore.ACCEPTABLE
            elif normal_dose_ratio <= 0.20:
                self.scores["normal_tissue"] = PlanQualityScore.POOR
            else:
                self.scores["normal_tissue"] = PlanQualityScore.UNACCEPTABLE

        except Exception as e:
            logger.error(f"Error evaluating normal tissue sparing: {e}")
            self.scores["normal_tissue"] = PlanQualityScore.NOT_APPLICABLE

    def _calculate_overall_score(self):
        """Calculate overall plan quality score."""
        # Make sure all key metrics have been evaluated
        required_metrics = [
            "clinical_goals",
            "target_coverage",
            "hotspots",
            "normal_tissue",
        ]

        if not all(metric in self.scores for metric in required_metrics):
            logger.warning("Not all required metrics have been evaluated")
            self.overall_score = PlanQualityScore.NOT_APPLICABLE
            return

        # Calculate weighted average score
        weights = {
            "clinical_goals": 0.4,
            "target_coverage": 0.3,
            "hotspots": 0.15,
            "normal_tissue": 0.15,
        }

        weighted_sum = 0
        total_weight = 0

        for metric, weight in weights.items():
            if (
                metric in self.scores
                and self.scores[metric] != PlanQualityScore.NOT_APPLICABLE
            ):
                weighted_sum += self.scores[metric].value * weight
                total_weight += weight

        if total_weight > 0:
            average_score = weighted_sum / total_weight

            # Map to quality score
            if average_score >= 4.5:
                self.overall_score = PlanQualityScore.EXCELLENT
            elif average_score >= 3.5:
                self.overall_score = PlanQualityScore.GOOD
            elif average_score >= 2.5:
                self.overall_score = PlanQualityScore.ACCEPTABLE
            elif average_score >= 1.5:
                self.overall_score = PlanQualityScore.POOR
            else:
                self.overall_score = PlanQualityScore.UNACCEPTABLE
        else:
            self.overall_score = PlanQualityScore.NOT_APPLICABLE

    def _compile_results(self):
        """Compile all evaluation results into a report."""
        self.results = {
            "plan_name": self.plan.name
            if hasattr(self.plan, "name")
            else "Unknown Plan",
            "plan_id": self.plan.id if hasattr(self.plan, "id") else "Unknown ID",
            "metrics": self.metrics,
            "scores": {k: v.name for k, v in self.scores.items()},
            "overall_score": self.overall_score.name
            if self.overall_score
            else "NOT_APPLICABLE",
            "clinical_goals": self.results.get("clinical_goals", {}),
        }

    def get_score_color(self, score):
        """
        Get a color associated with a quality score.

        Parameters
        ----------
        score : PlanQualityScore
            Quality score

        Returns
        -------
        tuple
            RGB color tuple
        """
        color_map = {
            PlanQualityScore.EXCELLENT: (0.0, 0.8, 0.0),  # Green
            PlanQualityScore.GOOD: (0.5, 0.8, 0.0),  # Light Green
            PlanQualityScore.ACCEPTABLE: (1.0, 0.8, 0.0),  # Yellow
            PlanQualityScore.POOR: (1.0, 0.4, 0.0),  # Orange
            PlanQualityScore.UNACCEPTABLE: (0.8, 0.0, 0.0),  # Red
            PlanQualityScore.NOT_APPLICABLE: (0.5, 0.5, 0.5),  # Gray
        }

        return color_map.get(score, (0.5, 0.5, 0.5))

    def plot_quality_radar(self, figsize=(8, 8)):
        """
        Create a radar plot of plan quality scores.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
            Radar plot figure
        """
        # Check if evaluation has been performed
        if not self.scores:
            logger.warning("No scores available for plotting")
            return None

        # Prepare data for radar plot
        categories = ["Clinical Goals", "Target Coverage", "Hotspots", "Normal Tissue"]
        scores = [
            self.scores.get("clinical_goals", PlanQualityScore.NOT_APPLICABLE).value,
            self.scores.get("target_coverage", PlanQualityScore.NOT_APPLICABLE).value,
            self.scores.get("hotspots", PlanQualityScore.NOT_APPLICABLE).value,
            self.scores.get("normal_tissue", PlanQualityScore.NOT_APPLICABLE).value,
        ]

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)

        # Number of variables
        N = len(categories)

        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Add the scores
        scores += scores[:1]  # Close the loop

        # Plot data
        ax.plot(angles, scores, linewidth=2, linestyle="solid")

        # Fill area
        ax.fill(angles, scores, alpha=0.3)

        # Set category labels
        plt.xticks(angles[:-1], categories)

        # Set y-axis limits
        ax.set_ylim(0, 5)

        # Add score labels
        for i, score in enumerate(scores[:-1]):
            score_text = PlanQualityScore(score).name if score > 0 else "N/A"
            angle = angles[i]
            x = 1.3 * np.cos(angle)
            y = 1.3 * np.sin(angle)
            ax.text(
                angle,
                score + 0.2,
                score_text,
                color=self.get_score_color(
                    PlanQualityScore(score)
                    if score > 0
                    else PlanQualityScore.NOT_APPLICABLE
                ),
                fontweight="bold",
            )

        # Add title
        ax.set_title(f"Plan Quality Evaluation: {self.overall_score.name}")

        return fig

    def generate_report(self):
        """
        Generate a comprehensive text report of plan quality.

        Returns
        -------
        str
            Report text
        """
        if not self.results:
            return "No evaluation results available."

        report = [
            f"Plan Quality Report for {self.results['plan_name']}",
            "=" * 40,
            f"Overall Quality: {self.results['overall_score']}",
            "",
            "Quality Scores by Category:",
            "-" * 30,
        ]

        for category, score in self.results["scores"].items():
            report.append(f"{category.replace('_', ' ').title()}: {score}")

        report.extend(
            [
                "",
                "Clinical Goals Summary:",
                "-" * 30,
                f"Total Goals: {self.results['clinical_goals'].get('total', 0)}",
                f"Passed: {self.results['clinical_goals'].get('passed', 0)}",
                f"Failed: {self.results['clinical_goals'].get('failed', 0)}",
                f"Warning: {self.results['clinical_goals'].get('warning', 0)}",
                f"Not Applicable: {self.results['clinical_goals'].get('not_applicable', 0)}",
                "",
                "Detailed Metrics:",
                "-" * 30,
            ]
        )

        for category, metrics in self.metrics.items():
            report.append(f"{category.replace('_', ' ').title()}:")

            if isinstance(metrics, dict):
                if all(isinstance(v, dict) for v in metrics.values()):
                    # Nested dictionaries (e.g., per target)
                    for name, values in metrics.items():
                        report.append(f"  {name}:")
                        for key, value in values.items():
                            if isinstance(value, float):
                                report.append(
                                    f"    {key.replace('_', ' ').title()}: {value:.3f}"
                                )
                            else:
                                report.append(
                                    f"    {key.replace('_', ' ').title()}: {value}"
                                )
                else:
                    # Single level dictionary
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            report.append(
                                f"  {key.replace('_', ' ').title()}: {value:.3f}"
                            )
                        else:
                            report.append(f"  {key.replace('_', ' ').title()}: {value}")

            report.append("")

        report.extend(["Clinical Goals Details:", "-" * 30])

        for goal in self.results["clinical_goals"].get("goals", []):
            result_str = {
                1: "PASSED",
                2: "FAILED",
                3: "WARNING",
                4: "NOT_APPLICABLE",
            }.get(goal.get("result", 4), "UNKNOWN")

            report.append(
                f"{goal.get('description', 'Unknown Goal')}: {result_str} "
                f"(Achieved: {goal.get('achieved_value', 'N/A'):.2f}, "
                f"Target: {goal.get('target_value', 'N/A'):.2f})"
            )

        return "\n".join(report)


def test_plan_quality_evaluator():
    """Test function for the plan quality evaluator."""
    import sys

    # Create some test data
    class TestDVHCalculator:
        def __init__(self):
            self.data = {
                "ptv": {
                    "volume": 100.0,
                    "mean_dose": 72.0,
                    "max_dose": 78.0,
                    "min_dose": 65.0,
                    "d95": 70.0,
                    "d98": 68.0,
                    "d2": 76.0,
                    "v70": 95.0,
                },
                "body": {
                    "volume": 5000.0,
                    "mean_dose": 5.0,
                    "max_dose": 78.0,
                    "min_dose": 0.1,
                },
                "cord": {
                    "volume": 50.0,
                    "mean_dose": 20.0,
                    "max_dose": 40.0,
                    "v45": 0.0,
                },
                "parotid_l": {"volume": 30.0, "mean_dose": 25.0},
                "parotid_r": {"volume": 28.0, "mean_dose": 24.0},
            }

        def has_structure(self, structure_id):
            return structure_id in self.data

        def get_volume_at_dose(self, structure_id, dose):
            if structure_id == "ptv" and dose == 70:
                return 95.0
            elif structure_id == "cord" and dose == 45:
                return 0.0
            return 0.0

        def get_dose_at_volume(self, structure_id, volume):
            if structure_id == "ptv":
                if volume == 95:
                    return 70.0
                elif volume == 98:
                    return 68.0
                elif volume == 2:
                    return 76.0
            return 0.0

        def get_max_dose(self, structure_id):
            return self.data.get(structure_id, {}).get("max_dose", 0.0)

        def get_min_dose(self, structure_id):
            return self.data.get(structure_id, {}).get("min_dose", 0.0)

        def get_mean_dose(self, structure_id):
            return self.data.get(structure_id, {}).get("mean_dose", 0.0)

        def get_structure_volume(self, structure_id):
            return self.data.get(structure_id, {}).get("volume", 0.0)

        def get_conformity_index(self, structure_id, reference_dose=None):
            if structure_id == "ptv":
                return 0.85
            return 0.0

        def get_homogeneity_index(self, structure_id):
            if structure_id == "ptv":
                return 0.08
            return 0.0

        def get_gradient_index(self, structure_id, reference_dose=None):
            if structure_id == "ptv":
                return 3.5
            return 0.0

    class TestPlan:
        def __init__(self):
            self.id = "test_plan_1"
            self.name = "Test Plan"
            self.prescription = TestPrescription()
            self.structure_set = TestStructureSet()

    class TestPrescription:
        def __init__(self):
            self.targets = [TestTarget()]

    class TestTarget:
        def __init__(self):
            self.structure_id = "ptv"
            self.structure_name = "PTV"
            self.dose_level = TestDoseLevel()

    class TestDoseLevel:
        def __init__(self):
            self.dose = 70.0

    class TestStructureSet:
        def __init__(self):
            self.structures = [
                TestStructure("ptv", "PTV"),
                TestStructure("body", "Body"),
                TestStructure("cord", "Spinal Cord"),
                TestStructure("parotid_l", "Parotid L"),
                TestStructure("parotid_r", "Parotid R"),
            ]

    class TestStructure:
        def __init__(self, id, name):
            self.id = id
            self.name = name

    # Create clinical goals
    goals = ClinicalGoalCollection("Test Goals")

    ptv_goal = ClinicalGoal(
        structure_id="ptv",
        structure_name="PTV",
        goal_type=GoalType.DOSE_AT_VOLUME,
        operator=GoalOperator.GREATER_THAN_OR_EQUAL,
        value=70.0,
        volume_level=95.0,
        priority=GoalPriority.CRITICAL,
    )
    goals.add_goal(ptv_goal)

    cord_goal = ClinicalGoal(
        structure_id="cord",
        structure_name="Spinal Cord",
        goal_type=GoalType.MAX_DOSE,
        operator=GoalOperator.LESS_THAN,
        value=45.0,
        priority=GoalPriority.CRITICAL,
    )
    goals.add_goal(cord_goal)

    parotid_l_goal = ClinicalGoal(
        structure_id="parotid_l",
        structure_name="Parotid L",
        goal_type=GoalType.MEAN_DOSE,
        operator=GoalOperator.LESS_THAN,
        value=26.0,
        priority=GoalPriority.MAJOR,
    )
    goals.add_goal(parotid_l_goal)

    # Create plan quality evaluator
    evaluator = PlanQualityEvaluator(
        plan=TestPlan(), protocol=TestProtocol(), dvh_analyzer=TestDVHCalculator()
    )

    # Evaluate plan quality
    results = evaluator.evaluate()

    # Generate report
    report = evaluator.generate_report()
    print(report)

    # Create radar plot
    fig = evaluator.plot_quality_radar()
    if fig:
        plt.show()


if __name__ == "__main__":
    test_plan_quality_evaluator()
