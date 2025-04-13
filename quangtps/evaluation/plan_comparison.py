"""
Plan Comparison Module

This module provides functionality for comparing multiple radiotherapy treatment plans
to support decision making in the planning process.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from copy import deepcopy

from quangtps.core.plan import Plan
from quangtps.core.structures import Structure, StructureType
from quangtps.evaluation.dvh.dvh_data import DVHData, DVHCurve
from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh, calculate_dvh_metrics
from quangtps.evaluation.clinical_goals import ClinicalGoal, GoalResult, GoalType, GoalOperator
from quangtps.evaluation.clinical_protocols import ClinicalProtocol
from quangtps.evaluation.plan_quality import PlanQualityEvaluator
from quangtps.evaluation.dose_analysis import analyze_dose_distribution
from quangtps.core.logging import get_logger
from quangtps.common.paths import get_temp_dir
from quangtps.evaluation.metrics import conformity_index, homogeneity_index, gradient_index
from quangtps.evaluation.qa.gamma_analysis import calculate_gamma_index_3d

logger = get_logger(__name__)


class PlanComparison:
    """
    Class for comparing multiple radiotherapy treatment plans.
    
    This class provides functionality for comparing DVH data, dose statistics,
    clinical goals achievement, and other metrics between multiple plans.
    
    Attributes:
        reference_plan: The primary plan against which others are compared
        comparison_plans: Dictionary of additional plans to compare
        protocol: Optional clinical protocol for evaluating all plans
        plan_data: Cached plan data including DVH and metrics
    """
    
    def __init__(self, reference_plan: Plan):
        """
        Initialize a plan comparison object.
        
        Args:
            reference_plan: The primary plan against which others are compared
        """
        self.reference_plan = reference_plan
        self.comparison_plans = {}
        self.protocol = None
        
        # Cache plan data
        self.plan_data = {}
        self._calculate_plan_data(reference_plan)
        
        logger.info(f"Created plan comparison with reference plan: {reference_plan.name}")
    
    def add_comparison_plan(self, plan: Plan) -> bool:
        """
        Add a plan to compare against the reference plan.
        
        Args:
            plan: The plan to add for comparison
            
        Returns:
            True if the plan was successfully added, False otherwise
        """
        # Skip if plan ID already exists
        if plan.id in self.comparison_plans:
            logger.warning(f"Plan {plan.name} already exists in comparison")
            return False
        
        # Skip if it's the reference plan
        if plan.id == self.reference_plan.id:
            logger.warning(f"Cannot add reference plan as comparison plan")
            return False
        
        self.comparison_plans[plan.id] = plan
        self._calculate_plan_data(plan)
        
        logger.info(f"Added comparison plan: {plan.name}")
        return True
    
    def remove_comparison_plan(self, plan_id: str) -> bool:
        """
        Remove a plan from the comparison.
        
        Args:
            plan_id: ID of the plan to remove
            
        Returns:
            True if the plan was successfully removed, False otherwise
        """
        if plan_id not in self.comparison_plans:
            logger.warning(f"Plan {plan_id} not found in comparison")
            return False
        
        # Remove the plan and its data
        del self.comparison_plans[plan_id]
        if plan_id in self.plan_data:
            del self.plan_data[plan_id]
        
        logger.info(f"Removed comparison plan: {plan_id}")
        return True
    
    def set_clinical_protocol(self, protocol: ClinicalProtocol):
        """
        Set a clinical protocol for evaluating all plans.
        
        Args:
            protocol: The clinical protocol to use
        """
        self.protocol = protocol
        
        # Re-evaluate all plans with the new protocol
        self._evaluate_with_protocol(self.reference_plan)
        for plan_id, plan in self.comparison_plans.items():
            self._evaluate_with_protocol(plan)
        
        logger.info(f"Set clinical protocol: {protocol.name}")
    
    def get_plan_names(self) -> List[str]:
        """
        Get a list of all plan names in the comparison.
        
        Returns:
            List of plan names
        """
        names = [self.reference_plan.name]
        for plan in self.comparison_plans.values():
            names.append(plan.name)
        return names
    
    def get_structure_ids(self) -> List[str]:
        """
        Get a list of structure IDs that exist in all plans.
        
        Returns:
            List of structure IDs
        """
        # Get structures from reference plan
        reference_structures = {
            s.id for s in self.reference_plan.structure_set.structures
        }
        
        # Find intersection with comparison plans
        for plan in self.comparison_plans.values():
            plan_structures = {s.id for s in plan.structure_set.structures}
            reference_structures &= plan_structures
        
        return list(reference_structures)
    
    def get_dvh_data(self, plan_id: str, structure_id: str) -> Optional[DVHData]:
        """
        Get DVH data for a specific structure in a plan.
        
        Args:
            plan_id: ID of the plan
            structure_id: ID of the structure
            
        Returns:
            DVH data for the specified structure and plan, or None if not found
        """
        if plan_id == self.reference_plan.id:
            plan_data = self.plan_data.get(self.reference_plan.id, {})
        else:
            plan_data = self.plan_data.get(plan_id, {})
        
        return plan_data.get('dvh_data', {}).get(structure_id)
    
    def get_metric(self, plan_id: str, structure_id: str, metric_name: str) -> Optional[float]:
        """
        Get a specific metric value for a structure in a plan.
        
        Args:
            plan_id: ID of the plan
            structure_id: ID of the structure
            metric_name: Name of the metric (e.g., "D95", "V20", "mean_dose")
            
        Returns:
            Value of the metric, or None if not available
        """
        if plan_id == self.reference_plan.id:
            plan_data = self.plan_data.get(self.reference_plan.id, {})
        else:
            plan_data = self.plan_data.get(plan_id, {})
        
        metrics = plan_data.get('metrics', {}).get(structure_id, {})
        return metrics.get(metric_name)
    
    def get_all_metrics(self, structure_id: str, metric_name: str) -> Dict[str, float]:
        """
        Get a specific metric across all plans for a structure.
        
        Args:
            structure_id: ID of the structure
            metric_name: Name of the metric
        
        Returns:
            Dictionary mapping plan IDs to metric values
        """
        result = {}
        
        # Get from reference plan
        ref_value = self.get_metric(self.reference_plan.id, structure_id, metric_name)
        if ref_value is not None:
            result[self.reference_plan.id] = ref_value
        
        # Get from comparison plans
        for plan_id in self.comparison_plans:
            value = self.get_metric(plan_id, structure_id, metric_name)
            if value is not None:
                result[plan_id] = value
        
        return result
    
    def get_goal_results(self, plan_id: str) -> Dict[str, GoalResult]:
        """
        Get the results of clinical goals for a plan.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            Dictionary mapping goal IDs to results
        """
        if plan_id == self.reference_plan.id:
            plan_data = self.plan_data.get(self.reference_plan.id, {})
        else:
            plan_data = self.plan_data.get(plan_id, {})
        
        return plan_data.get('goal_results', {})
    
    def compare_specific_metrics(self, metric_names: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare specific metrics across all plans for all common structures.
        
        Args:
            metric_names: List of metric names to compare
        
        Returns:
            Nested dictionary with structure IDs as keys, then metric names, then plan IDs
        """
        result = {}
        
        # Get common structures
        structure_ids = self.get_structure_ids()
        
        for structure_id in structure_ids:
            result[structure_id] = {}
            
            for metric_name in metric_names:
                # Get metric values for all plans
                metric_values = self.get_all_metrics(structure_id, metric_name)
                
                if metric_values:
                    result[structure_id][metric_name] = metric_values
        
        return result
    
    def generate_comparison_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a detailed PDF report of the plan comparison.
        
        Args:
            output_file: Optional path for saving the report. If None, a temporary file is created.
        
        Returns:
            Path to the generated report file
        """
        try:
            from quangtps.reporting.report_generator import ReportGenerator
            
            # Create report generator
            generator = ReportGenerator()
            
            # Add plan comparison data
            generator.add_section("Plan Comparison")
            generator.add_plan_comparison(self)
            
            # Generate report
            if output_file is None:
                output_file = os.path.join(get_temp_dir(), "plan_comparison_report.pdf")
            
            generator.generate_pdf(output_file)
            logger.info(f"Generated comparison report: {output_file}")
            
            return output_file
        
        except ImportError:
            logger.error("Cannot generate report - reporting module not available")
            return ""
    
    def _calculate_plan_data(self, plan: Plan):
        """
        Calculate and cache DVH data and metrics for a plan.
        
        Args:
            plan: The plan to calculate data for
        """
        plan_data = {
            'dvh_data': {},
            'metrics': {},
            'goal_results': {},
        }
        
        # Get all structures
        structures = plan.structure_set.structures
        
        for structure in structures:
            # Get DVH data
            dvh_data = plan.get_dvh_data(structure.id)
            if dvh_data:
                plan_data['dvh_data'][structure.id] = dvh_data
                
                # Calculate metrics
                structure_metrics = {}
                
                # D metrics (dose at volume)
                structure_metrics['D95'] = dvh_data.get_dose_at_volume(95)
                structure_metrics['D90'] = dvh_data.get_dose_at_volume(90)
                structure_metrics['D50'] = dvh_data.get_dose_at_volume(50)
                structure_metrics['D5'] = dvh_data.get_dose_at_volume(5)
                structure_metrics['D2'] = dvh_data.get_dose_at_volume(2)
                
                # V metrics (volume at dose) - using prescription dose from plan
                prescription = plan.prescription
                if prescription:
                    prescription_dose = prescription.dose
                    structure_metrics[f'V{int(prescription_dose)}'] = dvh_data.get_volume_at_dose(prescription_dose)
                    structure_metrics[f'V{int(prescription_dose*0.95)}'] = dvh_data.get_volume_at_dose(prescription_dose * 0.95)
                    structure_metrics[f'V{int(prescription_dose*0.9)}'] = dvh_data.get_volume_at_dose(prescription_dose * 0.9)
                    structure_metrics[f'V{int(prescription_dose*0.5)}'] = dvh_data.get_volume_at_dose(prescription_dose * 0.5)
                
                # Common V metrics for OARs
                structure_metrics['V5'] = dvh_data.get_volume_at_dose(5)
                structure_metrics['V10'] = dvh_data.get_volume_at_dose(10)
                structure_metrics['V20'] = dvh_data.get_volume_at_dose(20)
                structure_metrics['V30'] = dvh_data.get_volume_at_dose(30)
                structure_metrics['V40'] = dvh_data.get_volume_at_dose(40)
                structure_metrics['V50'] = dvh_data.get_volume_at_dose(50)
                
                # Other common metrics
                structure_metrics['mean_dose'] = dvh_data.mean_dose
                structure_metrics['max_dose'] = dvh_data.max_dose
                structure_metrics['min_dose'] = dvh_data.min_dose
                
                plan_data['metrics'][structure.id] = structure_metrics
        
        self.plan_data[plan.id] = plan_data
        
        # Evaluate with protocol if available
        if self.protocol:
            self._evaluate_with_protocol(plan)
    
    def _evaluate_with_protocol(self, plan: Plan):
        """
        Evaluate a plan against the clinical protocol.
        
        Args:
            plan: The plan to evaluate
        """
        if not self.protocol:
            return
        
        goal_results = {}
        
        for goal in self.protocol.goals:
            # Skip goals for structures not in this plan
            structure = plan.structure_set.get_structure(goal.structure_id)
            if not structure:
                continue
            
            # Get DVH data
            dvh_data = plan.get_dvh_data(goal.structure_id)
            if not dvh_data:
                continue
            
            # Evaluate goal
            result = goal.evaluate(dvh_data)
            if result:
                goal_results[goal.id] = result
        
        # Store results
        if plan.id in self.plan_data:
            self.plan_data[plan.id]['goal_results'] = goal_results
    
    def calculate_gamma_index(self, reference_plan_id: str, evaluation_plan_id: str, 
                             dose_threshold: float = 3.0, distance_threshold: float = 3.0) -> np.ndarray:
        """
        Calculate the gamma index between two plans.
        
        The gamma index is a metric for comparing dose distributions that combines
        dose difference and distance-to-agreement criteria.
        
        Args:
            reference_plan_id: ID of the reference plan
            evaluation_plan_id: ID of the plan to evaluate
            dose_threshold: Dose difference threshold in percent
            distance_threshold: Distance-to-agreement threshold in mm
            
        Returns:
            3D numpy array of gamma index values
        """
        # Get reference plan
        if reference_plan_id == self.reference_plan.id:
            ref_plan = self.reference_plan
        else:
            ref_plan = self.comparison_plans.get(reference_plan_id)
        
        # Get evaluation plan
        if evaluation_plan_id == self.reference_plan.id:
            eval_plan = self.reference_plan
        else:
            eval_plan = self.comparison_plans.get(evaluation_plan_id)
        
        if not ref_plan or not eval_plan:
            logger.error(f"Cannot calculate gamma index - plan not found")
            return np.array([])
        
        # Get dose distributions
        ref_dose = ref_plan.dose.get_dose_grid()
        eval_dose = eval_plan.dose.get_dose_grid()
        
        if ref_dose is None or eval_dose is None:
            logger.error(f"Cannot calculate gamma index - dose not available")
            return np.array([])
        
        try:
            # Calculate gamma index
            gamma = calculate_gamma_index_3d(
                ref_dose,
                eval_dose,
                ref_dose.spacing,
                eval_dose.spacing,
                dose_threshold=dose_threshold,
                distance_threshold=distance_threshold
            )
            return gamma
        except Exception as e:
            logger.error(f"Error calculating gamma index: {str(e)}")
            return np.array([])
    
    def calculate_dose_difference(self, reference_plan_id: str, evaluation_plan_id: str) -> np.ndarray:
        """
        Calculate the dose difference between two plans.
        
        Args:
            reference_plan_id: ID of the reference plan
            evaluation_plan_id: ID of the plan to evaluate
        
        Returns:
            3D numpy array of dose differences (evaluation - reference)
        """
        # Get reference plan
        if reference_plan_id == self.reference_plan.id:
            ref_plan = self.reference_plan
        else:
            ref_plan = self.comparison_plans.get(reference_plan_id)
        
        # Get evaluation plan
        if evaluation_plan_id == self.reference_plan.id:
            eval_plan = self.reference_plan
        else:
            eval_plan = self.comparison_plans.get(evaluation_plan_id)
        
        if not ref_plan or not eval_plan:
            logger.error(f"Cannot calculate dose difference - plan not found")
            return np.array([])
        
        # Get dose distributions
        ref_dose = ref_plan.dose.get_dose_grid()
        eval_dose = eval_plan.dose.get_dose_grid()
        
        if ref_dose is None or eval_dose is None:
            logger.error(f"Cannot calculate dose difference - dose not available")
            return np.array([])
        
        try:
            # Ensure same dimensions
            if ref_dose.shape != eval_dose.shape:
                logger.error(f"Cannot calculate dose difference - dose dimensions do not match")
                return np.array([])
            
            # Calculate difference (eval - ref)
            difference = eval_dose - ref_dose
            return difference
        except Exception as e:
            logger.error(f"Error calculating dose difference: {str(e)}")
            return np.array([])
