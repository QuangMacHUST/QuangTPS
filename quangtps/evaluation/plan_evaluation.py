#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Evaluation Module for QuangTPS.

This module provides functions for evaluating treatment plans based on
dose distribution metrics, DVH analysis, and biological models.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# Import DVH-related functions
from quangtps.evaluation.dvh import (
    calculate_dvh,
    calculate_dvh_for_plan,
    calculate_dvh_metrics,
    calculate_conformity_index,
    calculate_homogeneity_index,
    calculate_gradient_index,
    calculate_equivalent_uniform_dose,
    plot_dvh,
    plot_multiple_dvh,
    create_dvh_report
)

# Import biological models if available
try:
    from quangtps.evaluation.biological.tcp import (
        calculate_tcp_lq_poisson,
        calculate_tcp_lq_poisson_dvh
    )
    from quangtps.evaluation.biological.ntcp import (
        calculate_ntcp_lkb,
        calculate_ntcp_for_dvh
    )
    BIOLOGICAL_MODELS_AVAILABLE = True
except ImportError:
    BIOLOGICAL_MODELS_AVAILABLE = False
    logging.warning("Biological models not available, TCP/NTCP calculation disabled")

logger = logging.getLogger(__name__)

class PlanEvaluation:
    """
    Class for evaluating a radiotherapy treatment plan.
    
    This class provides methods for analyzing dose distributions,
    calculating DVH metrics, and evaluating plan quality based on
    various metrics and criteria.
    """
    
    def __init__(
        self,
        dose_grid: np.ndarray,
        structures: Dict[str, np.ndarray],
        prescription_doses: Dict[str, float] = None,
        structure_types: Dict[str, str] = None,
        plan_name: str = "Plan"
    ):
        """
        Initialize the plan evaluation with dose grid and structures.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid with dose values in Gy
        structures : Dict[str, np.ndarray]
            Dictionary mapping structure names to binary 3D masks
        prescription_doses : Dict[str, float], optional
            Dictionary mapping target structure names to prescription doses
        structure_types : Dict[str, str], optional
            Dictionary mapping structure names to structure types (PTV, OAR, etc.)
        plan_name : str, optional
            Name of the plan for reporting
        """
        self.dose_grid = dose_grid
        self.structures = structures
        self.prescription_doses = prescription_doses or {}
        self.structure_types = structure_types or {}
        self.plan_name = plan_name
        
        # Calculate DVH data
        self.dvh_data = calculate_dvh_for_plan(dose_grid, structures)
        
        # Store common metrics
        self.metrics = {}
        
    def calculate_metrics(
        self,
        metrics_list: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate DVH metrics for all structures.
        
        Parameters
        ----------
        metrics_list : List[str], optional
            List of metrics to calculate, e.g. ['D95', 'V20', 'Dmax']
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping structure names to dictionaries of metrics
        """
        if metrics_list is None:
            metrics_list = [
                'D98', 'D95', 'D90', 'D50', 'D2',
                'V5', 'V10', 'V20', 'V30', 'V40', 'V50',
                'Dmin', 'Dmax', 'Dmean'
            ]
        
        self.metrics = {}
        
        for structure_name, dvh in self.dvh_data.items():
            # Get prescription dose if available
            rx_dose = self.prescription_doses.get(structure_name)
            
            # Calculate metrics
            structure_metrics = calculate_dvh_metrics(dvh, metrics_list, rx_dose)
            
            self.metrics[structure_name] = structure_metrics
            
        return self.metrics
    
    def calculate_plan_indices(self) -> Dict[str, float]:
        """
        Calculate plan quality indices such as CI, HI, GI.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of plan indices
        """
        indices = {}
        
        # Identify target structures (PTVs)
        targets = [name for name, type_ in self.structure_types.items() 
                 if type_.upper() in ('PTV', 'TARGET', 'CTV', 'GTV')]
        
        # If no targets specified, try to find them by name
        if not targets:
            targets = [name for name in self.structures.keys() 
                     if any(t in name.upper() for t in ('PTV', 'TARGET', 'GTV', 'CTV'))]
        
        # Calculate indices for each target
        for target_name in targets:
            if target_name not in self.dvh_data:
                continue
                
            # Get prescription dose
            rx_dose = self.prescription_doses.get(target_name)
            if rx_dose is None:
                logger.warning(f"No prescription dose for {target_name}, skipping indices")
                continue
            
            # Get target DVH data
            target_dvh = self.dvh_data[target_name]
            
            # Calculate conformity index
            ci = calculate_conformity_index(target_dvh, rx_dose)
            indices[f"{target_name}_CI"] = ci
            
            # Calculate homogeneity index
            hi = calculate_homogeneity_index(target_dvh, rx_dose)
            indices[f"{target_name}_HI"] = hi
            
            # Calculate gradient index if body/external contour exists
            body_names = ['BODY', 'EXTERNAL', 'PATIENT']
            body_name = next((name for name in self.structures.keys() 
                           if name.upper() in body_names), None)
            
            if body_name and body_name in self.dvh_data:
                half_rx = rx_dose / 2.0
                gi = calculate_gradient_index(self.dvh_data[body_name], rx_dose, half_rx)
                indices[f"{target_name}_GI"] = gi
        
        return indices
    
    def evaluate_constraints(
        self,
        constraints: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Evaluate plan against clinical constraints.
        
        Parameters
        ----------
        constraints : Dict[str, List[Dict[str, Any]]]
            Dictionary mapping structure names to lists of constraint dictionaries.
            Each constraint has keys:
                - 'type': str, e.g. 'D95', 'V20'
                - 'goal': float, goal value
                - 'relation': str, '<', '>', '='
                - 'priority': str, optional, 'MUST', 'HIGH', 'MEDIUM', 'LOW'
                - 'unit': str, optional, 'Gy', '%', etc.
        
        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary mapping structure names to lists of evaluated constraints
            with additional keys:
                - 'actual': float, actual value
                - 'result': str, 'PASS', 'FAIL', 'BORDERLINE'
                - 'deviation': float, deviation from goal
        """
        if not self.metrics:
            self.calculate_metrics()
        
        results = {}
        
        for structure_name, structure_constraints in constraints.items():
            if structure_name not in self.metrics:
                logger.warning(f"Structure {structure_name} not found in metrics")
                continue
                
            structure_results = []
            
            for constraint in structure_constraints:
                constraint_type = constraint.get('type')
                goal = constraint.get('goal')
                relation = constraint.get('relation')
                
                if not constraint_type or goal is None or not relation:
                    logger.warning(f"Invalid constraint: {constraint}")
                    continue
                
                # Get actual value from metrics
                if constraint_type in self.metrics[structure_name]:
                    actual = self.metrics[structure_name][constraint_type]
                else:
                    # Try to calculate if not available
                    dvh = self.dvh_data[structure_name]
                    
                    if constraint_type.startswith('D') and constraint_type[1:].replace('.', '', 1).isdigit():
                        # Dx constraint
                        percent = float(constraint_type[1:])
                        actual = calculate_dvh_metrics(dvh, [constraint_type])[constraint_type]
                    elif constraint_type.startswith('V') and constraint_type[1:].replace('.', '', 1).isdigit():
                        # Vx constraint
                        dose = float(constraint_type[1:])
                        actual = calculate_dvh_metrics(dvh, [constraint_type])[constraint_type]
                    else:
                        logger.warning(f"Cannot calculate {constraint_type}")
                        continue
                
                # Evaluate constraint
                deviation = actual - goal if relation != '=' else abs(actual - goal)
                tolerance = constraint.get('tolerance', 1.0)
                
                if relation == '<':
                    if actual <= goal:
                        result = 'PASS'
                    elif actual <= goal + tolerance:
                        result = 'BORDERLINE'
                    else:
                        result = 'FAIL'
                elif relation == '>':
                    if actual >= goal:
                        result = 'PASS'
                    elif actual >= goal - tolerance:
                        result = 'BORDERLINE'
                    else:
                        result = 'FAIL'
                else:  # relation == '='
                    if abs(actual - goal) <= tolerance:
                        result = 'PASS'
                    elif abs(actual - goal) <= tolerance * 2:
                        result = 'BORDERLINE'
                    else:
                        result = 'FAIL'
                
                # Create result dictionary
                result_dict = {**constraint, 'actual': actual, 'result': result, 'deviation': deviation}
                structure_results.append(result_dict)
            
            results[structure_name] = structure_results
        
        return results
    
    def calculate_tcp_ntcp(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate TCP and NTCP for structures if models are available.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping structure names to dictionaries of TCP/NTCP values
        """
        if not BIOLOGICAL_MODELS_AVAILABLE:
            logger.warning("Biological models not available, skipping TCP/NTCP")
            return {}
        
        bio_metrics = {}
        
        for structure_name, dvh in self.dvh_data.items():
            structure_bio = {}
            
            # Determine structure type
            struct_type = self.structure_types.get(structure_name, '').upper()
            
            if 'PTV' in struct_type or 'GTV' in struct_type or 'CTV' in struct_type or 'TARGET' in struct_type:
                # Calculate TCP for targets
                try:
                    # TCP parameters - could be loaded from a database
                    params = {
                        'alpha': 0.3,  # Gy^-1
                        'beta': 0.03,  # Gy^-2
                        'n': 1e7,      # clonogenic cell number
                        'alpha_beta': 10.0,  # Gy
                        'fraction_size': 2.0  # Gy
                    }
                    
                    tcp = calculate_tcp_lq_poisson_dvh(dvh, params)
                    structure_bio['TCP'] = tcp
                    
                except Exception as e:
                    logger.error(f"Error calculating TCP for {structure_name}: {e}")
            
            elif 'OAR' in struct_type or any(oar in structure_name.upper() for oar in 
                                           ['LUNG', 'HEART', 'SPINAL', 'CORD', 'LIVER', 'KIDNEY']):
                # Calculate NTCP for OARs
                try:
                    # NTCP parameters - should be based on specific organ model
                    params = {
                        'n': 0.1,    # volume effect parameter
                        'm': 0.15,   # slope parameter
                        'TD50': 50.0  # dose for 50% complication probability
                    }
                    
                    ntcp = calculate_ntcp_lkb(dvh, params)
                    structure_bio['NTCP'] = ntcp
                    
                except Exception as e:
                    logger.error(f"Error calculating NTCP for {structure_name}: {e}")
            
            if structure_bio:
                bio_metrics[structure_name] = structure_bio
        
        return bio_metrics
    
    def plot_dvh(
        self,
        structures: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
        show_metrics: bool = True,
        metrics_to_show: Optional[List[str]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot DVH curves for specified structures.
        
        Parameters
        ----------
        structures : List[str], optional
            List of structures to include, if None, include all
        ax : plt.Axes, optional
            Axes to plot on, if None, create new figure
        show_metrics : bool, optional
            Whether to show metrics on the plot
        metrics_to_show : List[str], optional
            List of metrics to show
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        if structures is None:
            structures = list(self.dvh_data.keys())
        
        # Filter to only structures that exist
        valid_structures = [s for s in structures if s in self.dvh_data]
        
        # Create filtered DVH data dictionary
        filtered_dvh = {s: self.dvh_data[s] for s in valid_structures}
        
        # Default metrics to show
        if metrics_to_show is None:
            metrics_to_show = ['D95', 'Dmean', 'V20']
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Create a subplot for each valid structure
        for struct_name in valid_structures:
            # Get prescription dose if available
            rx_dose = self.prescription_doses.get(struct_name)
            
            # Plot the DVH
            plot_dvh(
                dvh_data={struct_name: filtered_dvh[struct_name]},
                structure_name=struct_name,
                ax=ax,
                show_metrics=show_metrics,
                metrics_to_show=metrics_to_show,
                prescription_dose=rx_dose
            )
        
        # Add plan indices to the plot if targets exist
        indices = self.calculate_plan_indices()
        
        if indices:
            indices_text = "Plan Indices:\n"
            for name, value in indices.items():
                indices_text += f"{name}: {value:.3f}\n"
            
            # Add text to bottom left
            ax.text(
                0.02, 0.02, indices_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
                fontsize=8,
                verticalalignment='bottom'
            )
        
        # Set title
        ax.set_title(f"DVH for {self.plan_name}")
        
        return fig, ax
    
    def create_evaluation_report(
        self,
        output_path: Optional[str] = None,
        include_metrics: bool = True,
        include_constraints: bool = False,
        constraints: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        include_biological: bool = False
    ) -> Dict[str, Any]:
        """
        Create a comprehensive plan evaluation report.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the report
        include_metrics : bool, optional
            Whether to include DVH metrics
        include_constraints : bool, optional
            Whether to include constraint evaluation
        constraints : Dict[str, List[Dict[str, Any]]], optional
            Dictionary of constraints to evaluate
        include_biological : bool, optional
            Whether to include TCP/NTCP calculations
            
        Returns
        -------
        Dict[str, Any]
            Report data including figures and metrics
        """
        # Initialize report data
        report_data = {
            'plan_name': self.plan_name,
            'structures': list(self.structures.keys()),
            'prescription_doses': self.prescription_doses.copy(),
            'structure_types': self.structure_types.copy()
        }
        
        # Calculate metrics
        if include_metrics:
            self.calculate_metrics()
            report_data['metrics'] = self.metrics
        
        # Calculate plan indices
        indices = self.calculate_plan_indices()
        report_data['indices'] = indices
        
        # Evaluate constraints
        if include_constraints and constraints:
            constraint_results = self.evaluate_constraints(constraints)
            report_data['constraints'] = constraint_results
        
        # Calculate biological metrics
        if include_biological and BIOLOGICAL_MODELS_AVAILABLE:
            bio_metrics = self.calculate_tcp_ntcp()
            report_data['biological'] = bio_metrics
        
        # Create DVH report
        try:
            dvh_report = create_dvh_report(
                dvh_list=[self.dvh_data],
                structure_names=list(self.structures.keys()),
                plan_names=[self.plan_name],
                prescription_doses=self.prescription_doses,
                structure_types=self.structure_types,
                output_path=output_path,
                show_statistics=include_metrics
            )
            
            report_data['dvh_figure'] = dvh_report.get('figure')
            
            if output_path:
                logger.info(f"DVH report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error creating DVH report: {e}")
        
        return report_data

# Standalone function for easier usage
def evaluate_plan(
    dose_grid: np.ndarray,
    structures: Dict[str, np.ndarray],
    prescription_doses: Dict[str, float] = None,
    structure_types: Dict[str, str] = None,
    plan_name: str = "Plan",
    output_path: Optional[str] = None
) -> PlanEvaluation:
    """
    Evaluate a radiotherapy treatment plan.
    
    Parameters
    ----------
    dose_grid : np.ndarray
        3D dose grid with dose values in Gy
    structures : Dict[str, np.ndarray]
        Dictionary mapping structure names to binary 3D masks
    prescription_doses : Dict[str, float], optional
        Dictionary mapping target structure names to prescription doses
    structure_types : Dict[str, str], optional
        Dictionary mapping structure names to structure types (PTV, OAR, etc.)
    plan_name : str, optional
        Name of the plan for reporting
    output_path : str, optional
        Path to save the evaluation report
        
    Returns
    -------
    PlanEvaluation
        PlanEvaluation object with calculated metrics and reports
    """
    evaluator = PlanEvaluation(
        dose_grid=dose_grid,
        structures=structures,
        prescription_doses=prescription_doses,
        structure_types=structure_types,
        plan_name=plan_name
    )
    
    # Calculate metrics
    evaluator.calculate_metrics()
    
    # Calculate plan indices
    evaluator.calculate_plan_indices()
    
    # Create report if output path provided
    if output_path:
        evaluator.create_evaluation_report(output_path=output_path)
    
    return evaluator 