"""
Plan Comparison Module

This module provides functionality for comparing multiple radiotherapy treatment plans.
It includes features for comparing DVHs, dose distributions, and clinical metrics
similar to Eclipse's plan comparison capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import logging
from datetime import datetime

from quangtps.core.types import Structure
from quangtps.planning.plan import Plan
from quangtps.evaluation.dvh.dvh_data import DVHData, DVHCurve
from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh, calculate_dvh_metrics
from quangtps.evaluation.clinical_goals import ClinicalGoal, GoalResult
from quangtps.evaluation.clinical_protocols import ClinicalProtocol
from quangtps.evaluation.plan_quality import PlanQualityEvaluator
from quangtps.evaluation.dose_analysis import analyze_dose_distribution
from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class PlanComparison:
    """
    Class for comparing multiple treatment plans.
    
    This class provides functionality similar to Eclipse's plan comparison features,
    allowing for side-by-side comparison of DVHs, dose distributions, and metrics.
    """
    
    def __init__(self, reference_plan: Plan):
        """
        Initialize the plan comparison with a reference plan.
        
        Args:
            reference_plan: The primary plan to compare others against
        """
        self.reference_plan = reference_plan
        self.comparison_plans: Dict[str, Plan] = {}
        self.dvh_data: Dict[str, Dict[str, DVHData]] = {}
        self.metric_data: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.structure_names: Dict[str, str] = {}
        self.protocol: Optional[ClinicalProtocol] = None
        self.goal_results: Dict[str, Dict[str, GoalResult]] = {}
        
        # Calculate DVH and metrics for reference plan
        self._calculate_plan_data(reference_plan)
    
    def add_comparison_plan(self, plan: Plan) -> bool:
        """
        Add a plan to be compared against the reference plan.
        
        Args:
            plan: The plan to add for comparison
            
        Returns:
            True if the plan was successfully added, False otherwise
        """
        if plan.id == self.reference_plan.id:
            logger.warning(f"Cannot add reference plan {plan.id} as comparison plan")
            return False
            
        if plan.id in self.comparison_plans:
            logger.warning(f"Plan {plan.id} already exists in comparison")
            return False
            
        # Store the plan
        self.comparison_plans[plan.id] = plan
        
        # Calculate DVH and metrics for the plan
        self._calculate_plan_data(plan)
        
        return True
    
    def remove_comparison_plan(self, plan_id: str) -> bool:
        """
        Remove a plan from the comparison.
        
        Args:
            plan_id: ID of the plan to remove
            
        Returns:
            True if the plan was removed, False if it wasn't found
        """
        if plan_id not in self.comparison_plans:
            logger.warning(f"Plan {plan_id} not found in comparison")
            return False
            
        del self.comparison_plans[plan_id]
        
        # Remove the plan's data
        if plan_id in self.dvh_data:
            del self.dvh_data[plan_id]
            
        if plan_id in self.metric_data:
            del self.metric_data[plan_id]
            
        if plan_id in self.goal_results:
            del self.goal_results[plan_id]
            
        return True
    
    def set_clinical_protocol(self, protocol: ClinicalProtocol):
        """
        Set a clinical protocol for evaluating all plans.
        
        Args:
            protocol: The clinical protocol to use for evaluation
        """
        self.protocol = protocol
        
        # Evaluate all plans with the protocol
        self._evaluate_with_protocol(self.reference_plan)
        for plan_id, plan in self.comparison_plans.items():
            self._evaluate_with_protocol(plan)
    
    def get_plan_names(self) -> List[str]:
        """
        Get the list of all plan names in the comparison.
        
        Returns:
            List of plan names, with reference plan first
        """
        names = [self.reference_plan.name]
        for plan_id in self.comparison_plans:
            names.append(self.comparison_plans[plan_id].name)
        return names
    
    def get_structure_ids(self) -> List[str]:
        """
        Get the list of all structure IDs across all plans.
        
        Returns:
            List of all unique structure IDs
        """
        structure_ids = set()
        
        # Add structures from reference plan
        for structure in self.reference_plan.structure_set.structures:
            structure_ids.add(structure.id)
            self.structure_names[structure.id] = structure.name
        
        # Add structures from comparison plans
        for plan_id, plan in self.comparison_plans.items():
            for structure in plan.structure_set.structures:
                structure_ids.add(structure.id)
                self.structure_names[structure.id] = structure.name
        
        return list(structure_ids)
    
    def get_dvh_data(self, plan_id: str, structure_id: str) -> Optional[DVHData]:
        """
        Get DVH data for a specific plan and structure.
        
        Args:
            plan_id: ID of the plan
            structure_id: ID of the structure
            
        Returns:
            DVH data if found, None otherwise
        """
        if plan_id not in self.dvh_data:
            return None
            
        return self.dvh_data[plan_id].get(structure_id)
    
    def get_metric(self, plan_id: str, structure_id: str, metric_name: str) -> Optional[float]:
        """
        Get a specific metric value for a plan and structure.
        
        Args:
            plan_id: ID of the plan
            structure_id: ID of the structure
            metric_name: Name of the metric (e.g., "D95", "V20")
            
        Returns:
            Metric value if found, None otherwise
        """
        if plan_id not in self.metric_data:
            return None
            
        if structure_id not in self.metric_data[plan_id]:
            return None
            
        return self.metric_data[plan_id][structure_id].get(metric_name)
    
    def get_all_metrics(self, structure_id: str, metric_name: str) -> Dict[str, float]:
        """
        Get a specific metric across all plans for a structure.
        
        Args:
            structure_id: ID of the structure
            metric_name: Name of the metric
            
        Returns:
            Dictionary mapping plan IDs to metric values
        """
        results = {}
        
        # Reference plan
        if structure_id in self.metric_data.get(self.reference_plan.id, {}):
            if metric_name in self.metric_data[self.reference_plan.id][structure_id]:
                results[self.reference_plan.id] = self.metric_data[self.reference_plan.id][structure_id][metric_name]
        
        # Comparison plans
        for plan_id in self.comparison_plans:
            if structure_id in self.metric_data.get(plan_id, {}):
                if metric_name in self.metric_data[plan_id][structure_id]:
                    results[plan_id] = self.metric_data[plan_id][structure_id][metric_name]
        
        return results
    
    def get_goal_results(self, plan_id: str) -> Dict[str, GoalResult]:
        """
        Get all goal results for a specific plan.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            Dictionary mapping goal IDs to goal results
        """
        return self.goal_results.get(plan_id, {})
    
    def compare_specific_metrics(self, metric_names: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare specific metrics across all plans.
        
        Args:
            metric_names: List of metric names to compare
            
        Returns:
            Dictionary mapping structure IDs to plan IDs to metric values
        """
        results = {}
        structure_ids = self.get_structure_ids()
        
        for structure_id in structure_ids:
            results[structure_id] = {}
            
            for metric_name in metric_names:
                metric_results = self.get_all_metrics(structure_id, metric_name)
                
                if structure_id not in results:
                    results[structure_id] = {}
                    
                results[structure_id][metric_name] = metric_results
        
        return results
    
    def generate_comparison_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a report comparing all plans in HTML format.
        
        Args:
            output_file: Path to save the report (optional)
            
        Returns:
            HTML string containing the report
        """
        # Import here to avoid circular imports
        from quangtps.reporting.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        report = generator.generate_plan_comparison_report(
            reference_plan=self.reference_plan,
            comparison_plans=list(self.comparison_plans.values()),
            dvh_data=self.dvh_data,
            metric_data=self.metric_data,
            goal_results=self.goal_results
        )
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def _calculate_plan_data(self, plan: Plan):
        """
        Calculate DVH and metrics data for a plan.
        
        Args:
            plan: The plan to analyze
        """
        if not plan.dose:
            logger.warning(f"Plan {plan.id} has no dose data, skipping analysis")
            return
            
        # Store DVH data
        if plan.id not in self.dvh_data:
            self.dvh_data[plan.id] = {}
            
        # Store metric data
        if plan.id not in self.metric_data:
            self.metric_data[plan.id] = {}
        
        # Calculate DVH and metrics for each structure
        for structure in plan.structure_set.structures:
            # Skip empty structures
            if structure.is_empty():
                continue
                
            # Calculate DVH
            dvh = calculate_dvh(structure, plan.dose)
            self.dvh_data[plan.id][structure.id] = dvh
            
            # Calculate metrics
            metrics = calculate_dvh_metrics(structure, plan.dose)
            self.metric_data[plan.id][structure.id] = metrics
            
            # Store structure name
            self.structure_names[structure.id] = structure.name
        
        # Evaluate the plan with protocol if set
        if self.protocol:
            self._evaluate_with_protocol(plan)
    
    def _evaluate_with_protocol(self, plan: Plan):
        """
        Evaluate a plan with the current protocol.
        
        Args:
            plan: The plan to evaluate
        """
        if not self.protocol:
            return
            
        # Use PlanQualityEvaluator to evaluate the plan
        evaluator = PlanQualityEvaluator()
        evaluator.set_protocol(self.protocol)
        evaluator.evaluate_plan(plan)
        
        # Store goal results
        if plan.id not in self.goal_results:
            self.goal_results[plan.id] = {}
            
        # Get goal results
        for goal_id, result in evaluator.get_goal_results().items():
            self.goal_results[plan.id][goal_id] = result
    
    def calculate_gamma_index(self, reference_plan_id: str, evaluation_plan_id: str, 
                             dose_threshold: float = 3.0, distance_threshold: float = 3.0) -> np.ndarray:
        """
        Calculate the gamma index between two plans.
        
        Args:
            reference_plan_id: ID of the reference plan
            evaluation_plan_id: ID of the evaluation plan
            dose_threshold: Dose difference threshold in percent
            distance_threshold: Distance to agreement threshold in mm
            
        Returns:
            3D numpy array of gamma index values
        """
        from quangtps.evaluation.qa.gamma_analysis import calculate_gamma_index
        
        # Get the reference plan
        if reference_plan_id == self.reference_plan.id:
            reference_plan = self.reference_plan
        else:
            reference_plan = self.comparison_plans.get(reference_plan_id)
            
        # Get the evaluation plan
        if evaluation_plan_id == self.reference_plan.id:
            evaluation_plan = self.reference_plan
        else:
            evaluation_plan = self.comparison_plans.get(evaluation_plan_id)
            
        if not reference_plan or not evaluation_plan:
            logger.error(f"Plan not found for gamma index calculation")
            return np.zeros((1, 1, 1))  # Return empty array
            
        if not reference_plan.dose or not evaluation_plan.dose:
            logger.error(f"Dose data missing for gamma index calculation")
            return np.zeros((1, 1, 1))  # Return empty array
            
        # Calculate gamma index
        gamma = calculate_gamma_index(
            reference_dose=reference_plan.dose,
            evaluation_dose=evaluation_plan.dose,
            dose_threshold=dose_threshold,
            distance_threshold=distance_threshold
        )
        
        return gamma
    
    def calculate_dose_difference(self, reference_plan_id: str, evaluation_plan_id: str) -> np.ndarray:
        """
        Calculate the dose difference between two plans.
        
        Args:
            reference_plan_id: ID of the reference plan
            evaluation_plan_id: ID of the evaluation plan
            
        Returns:
            3D numpy array of dose difference values
        """
        # Get the reference plan
        if reference_plan_id == self.reference_plan.id:
            reference_plan = self.reference_plan
        else:
            reference_plan = self.comparison_plans.get(reference_plan_id)
            
        # Get the evaluation plan
        if evaluation_plan_id == self.reference_plan.id:
            evaluation_plan = self.reference_plan
        else:
            evaluation_plan = self.comparison_plans.get(evaluation_plan_id)
            
        if not reference_plan or not evaluation_plan:
            logger.error(f"Plan not found for dose difference calculation")
            return np.zeros((1, 1, 1))  # Return empty array
            
        if not reference_plan.dose or not evaluation_plan.dose:
            logger.error(f"Dose data missing for dose difference calculation")
            return np.zeros((1, 1, 1))  # Return empty array
            
        # Get dose grids
        ref_dose = reference_plan.dose.get_dose_grid()
        eval_dose = evaluation_plan.dose.get_dose_grid()
        
        # Ensure grids are the same size
        if ref_dose.shape != eval_dose.shape:
            logger.error(f"Dose grid sizes do not match: {ref_dose.shape} vs {eval_dose.shape}")
            return np.zeros((1, 1, 1))  # Return empty array
            
        # Calculate difference
        difference = eval_dose - ref_dose
        
        return difference
