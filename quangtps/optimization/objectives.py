#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for optimization objectives and constraints in QuangTPS.

This module provides classes and functions to define dose-based objectives
and constraints for radiotherapy treatment planning optimization.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import DVH functions from the correct location
from quangtps.evaluation.dvh import (
    calculate_dvh_from_dose_grid, 
    calculate_dvh_metrics
)
from quangtps.evaluation.dvh.dvh_calculation import (
    _get_dose_at_volume,
    _get_volume_at_dose
)

logger = logging.getLogger(__name__)


class ObjectiveFunction:
    """
    Base class for all objective functions.
    
    This class defines the interface for all objective functions used in the 
    optimization process. Subclasses must implement the __call__ method to 
    compute the objective value and the gradient method to compute the gradient.
    """
    
    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        """
        Initialize the objective function.
        
        Parameters
        ----------
        weight : float, optional
            Weight of the objective in the total cost function
        name : str, optional
            Name of the objective function
        """
        self.weight = weight
        self.name = name or self.__class__.__name__
    
    def __call__(self, dose_grid, structures=None, **kwargs) -> float:
        """
        Evaluate the objective function.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid
        structures : Dict[str, np.ndarray], optional
            Dictionary mapping structure names to binary masks
        **kwargs : Any
            Additional parameters
            
        Returns
        -------
        float
            Objective function value
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def gradient(self, dose_grid, structures=None, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the objective function.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid
        structures : Dict[str, np.ndarray], optional
            Dictionary mapping structure names to binary masks
        **kwargs : Any
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Gradient of the objective function
        """
        raise NotImplementedError("Subclasses must implement this method")


class DoseBasedObjective(ObjectiveFunction):
    """
    Base class for dose-based objective functions.
    
    Dose-based objectives depend only on the dose distribution, such as
    minimizing the total dose.
    """
    
    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        """
        Initialize the dose-based objective.
        
        Parameters
        ----------
        weight : float, optional
            Weight of the objective in the total cost function
        name : str, optional
            Name of the objective function
        """
        super().__init__(weight, name)


class StructureBasedObjective(ObjectiveFunction):
    """
    Base class for structure-based objective functions.
    
    Structure-based objectives depend on both the dose distribution and 
    the structures, such as minimizing the dose to an organ at risk.
    """
    
    def __init__(self, structure_name: str, weight: float = 1.0, name: Optional[str] = None):
        """
        Initialize the structure-based objective.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        weight : float, optional
            Weight of the objective in the total cost function
        name : str, optional
            Name of the objective function
        """
        super().__init__(weight, name)
        self.structure_name = structure_name


class MinDose(StructureBasedObjective):
    """
    Minimize the minimum dose to a structure.
    
    This objective function penalizes doses below the prescribed minimum dose.
    It is typically used for target structures.
    """
    
    def __init__(self, structure_name: str, min_dose: float, 
                 weight: float = 1.0, name: Optional[str] = None):
        """
        Initialize the minimum dose objective.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        min_dose : float
            Minimum dose in Gy
        weight : float, optional
            Weight of the objective in the total cost function
        name : str, optional
            Name of the objective function
        """
        super().__init__(structure_name, weight, name or f"MinDose_{structure_name}")
        self.min_dose = min_dose
    
    def __call__(self, dose_grid, structures=None, **kwargs) -> float:
        """
        Evaluate the minimum dose objective.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid
        structures : Dict[str, np.ndarray], optional
            Dictionary mapping structure names to binary masks
        **kwargs : Any
            Additional parameters
            
        Returns
        -------
        float
            Objective function value
        """
        if structures is None or self.structure_name not in structures:
            return 0.0
        
        structure_mask = structures[self.structure_name]
        doses = dose_grid[structure_mask > 0]
        
        if len(doses) == 0:
            return 0.0
        
        min_dose = np.min(doses)
        if min_dose >= self.min_dose:
            return 0.0
        
        return self.weight * (self.min_dose - min_dose)**2
    
    def gradient(self, dose_grid, structures=None, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the minimum dose objective.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid
        structures : Dict[str, np.ndarray], optional
            Dictionary mapping structure names to binary masks
        **kwargs : Any
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Gradient of the objective function
        """
        if structures is None or self.structure_name not in structures:
            return np.zeros_like(dose_grid)
        
        structure_mask = structures[self.structure_name]
        
        # Find voxel with minimum dose
        doses = dose_grid * structure_mask
        flat_indices = np.argwhere(structure_mask > 0)
        
        if len(flat_indices) == 0:
            return np.zeros_like(dose_grid)
        
        voxel_doses = doses[structure_mask > 0]
        min_dose_idx = np.argmin(voxel_doses)
        min_dose = voxel_doses[min_dose_idx]
        
        if min_dose >= self.min_dose:
            return np.zeros_like(dose_grid)
        
        # Get coordinates of minimum dose voxel
        min_voxel = tuple(flat_indices[min_dose_idx])
        
        # Initialize gradient
        gradient = np.zeros_like(dose_grid)
        
        # Update gradient at the minimum dose voxel
        gradient[min_voxel] = -2.0 * self.weight * (self.min_dose - min_dose)
        
        return gradient


class MaxDose(StructureBasedObjective):
    """
    Minimize the maximum dose to a structure.
    
    This objective function penalizes doses above the prescribed maximum dose.
    It is typically used for organs at risk.
    """
    
    def __init__(self, structure_name: str, max_dose: float, 
                 weight: float = 1.0, name: Optional[str] = None):
        """
        Initialize the maximum dose objective.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        max_dose : float
            Maximum dose in Gy
        weight : float, optional
            Weight of the objective in the total cost function
        name : str, optional
            Name of the objective function
        """
        super().__init__(structure_name, weight, name or f"MaxDose_{structure_name}")
        self.max_dose = max_dose
    
    def __call__(self, dose_grid, structures=None, **kwargs) -> float:
        """
        Evaluate the maximum dose objective.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid
        structures : Dict[str, np.ndarray], optional
            Dictionary mapping structure names to binary masks
        **kwargs : Any
            Additional parameters
            
        Returns
        -------
        float
            Objective function value
        """
        if structures is None or self.structure_name not in structures:
            return 0.0
        
        structure_mask = structures[self.structure_name]
        doses = dose_grid[structure_mask > 0]
        
        if len(doses) == 0:
            return 0.0
        
        max_dose = np.max(doses)
        if max_dose <= self.max_dose:
            return 0.0
        
        return self.weight * (max_dose - self.max_dose)**2
    
    def gradient(self, dose_grid, structures=None, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the maximum dose objective.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid
        structures : Dict[str, np.ndarray], optional
            Dictionary mapping structure names to binary masks
        **kwargs : Any
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Gradient of the objective function
        """
        if structures is None or self.structure_name not in structures:
            return np.zeros_like(dose_grid)
        
        structure_mask = structures[self.structure_name]
        
        # Find voxel with maximum dose
        doses = dose_grid * structure_mask
        flat_indices = np.argwhere(structure_mask > 0)
        
        if len(flat_indices) == 0:
            return np.zeros_like(dose_grid)
        
        voxel_doses = doses[structure_mask > 0]
        max_dose_idx = np.argmax(voxel_doses)
        max_dose = voxel_doses[max_dose_idx]
        
        if max_dose <= self.max_dose:
            return np.zeros_like(dose_grid)
        
        # Get coordinates of maximum dose voxel
        max_voxel = tuple(flat_indices[max_dose_idx])
        
        # Initialize gradient
        gradient = np.zeros_like(dose_grid)
        
        # Update gradient at the maximum dose voxel
        gradient[max_voxel] = 2.0 * self.weight * (max_dose - self.max_dose)
        
        return gradient


class MeanDose(StructureBasedObjective):
    """
    Minimize the mean dose to a structure.
    
    This objective function penalizes the mean dose above the prescribed dose.
    It is typically used for organs at risk.
    """
    
    def __init__(self, structure_name: str, target_dose: float, 
                 weight: float = 1.0, name: Optional[str] = None):
        """
        Initialize the mean dose objective.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        target_dose : float
            Target mean dose in Gy
        weight : float, optional
            Weight of the objective in the total cost function
        name : str, optional
            Name of the objective function
        """
        super().__init__(structure_name, weight, name or f"MeanDose_{structure_name}")
        self.target_dose = target_dose
    
    def __call__(self, dose_grid, structures=None, **kwargs) -> float:
        """
        Evaluate the mean dose objective.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid
        structures : Dict[str, np.ndarray], optional
            Dictionary mapping structure names to binary masks
        **kwargs : Any
            Additional parameters
            
        Returns
        -------
        float
            Objective function value
        """
        if structures is None or self.structure_name not in structures:
            return 0.0
        
        structure_mask = structures[self.structure_name]
        doses = dose_grid[structure_mask > 0]
        
        if len(doses) == 0:
            return 0.0
        
        mean_dose = np.mean(doses)
        if mean_dose <= self.target_dose:
            return 0.0
        
        return self.weight * (mean_dose - self.target_dose)**2
    
    def gradient(self, dose_grid, structures=None, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the mean dose objective.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid
        structures : Dict[str, np.ndarray], optional
            Dictionary mapping structure names to binary masks
        **kwargs : Any
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Gradient of the objective function
        """
        if structures is None or self.structure_name not in structures:
            return np.zeros_like(dose_grid)
        
        structure_mask = structures[self.structure_name]
        doses = dose_grid[structure_mask > 0]
        
        if len(doses) == 0:
            return np.zeros_like(dose_grid)
        
        mean_dose = np.mean(doses)
        if mean_dose <= self.target_dose:
            return np.zeros_like(dose_grid)
        
        # Calculate gradient
        gradient = np.zeros_like(dose_grid)
        voxel_count = np.sum(structure_mask > 0)
        
        # Update gradient for all voxels in the structure
        gradient[structure_mask > 0] = 2.0 * self.weight * (mean_dose - self.target_dose) / voxel_count
        
        return gradient


class UniformDose(StructureBasedObjective):
    """
    Minimize dose non-uniformity in a structure.
    
    This objective function penalizes the variance of the dose distribution
    in a structure. It is typically used for target structures to achieve
    a uniform dose distribution.
    """
    
    def __init__(self, structure_name: str, target_dose: float, 
                 weight: float = 1.0, name: Optional[str] = None):
        """
        Initialize the uniform dose objective.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        target_dose : float
            Target dose in Gy
        weight : float, optional
            Weight of the objective in the total cost function
        name : str, optional
            Name of the objective function
        """
        super().__init__(structure_name, weight, name or f"UniformDose_{structure_name}")
        self.target_dose = target_dose
    
    def __call__(self, dose_grid, structures=None, **kwargs) -> float:
        """
        Evaluate the uniform dose objective.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid
        structures : Dict[str, np.ndarray], optional
            Dictionary mapping structure names to binary masks
        **kwargs : Any
            Additional parameters
            
        Returns
        -------
        float
            Objective function value
        """
        if structures is None or self.structure_name not in structures:
            return 0.0
        
        structure_mask = structures[self.structure_name]
        doses = dose_grid[structure_mask > 0]
        
        if len(doses) == 0:
            return 0.0
        
        return self.weight * np.sum((doses - self.target_dose)**2) / len(doses)
    
    def gradient(self, dose_grid, structures=None, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the uniform dose objective.
        
        Parameters
        ----------
        dose_grid : np.ndarray
            3D dose grid
        structures : Dict[str, np.ndarray], optional
            Dictionary mapping structure names to binary masks
        **kwargs : Any
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Gradient of the objective function
        """
        if structures is None or self.structure_name not in structures:
            return np.zeros_like(dose_grid)
        
        structure_mask = structures[self.structure_name]
        
        # Initialize gradient
        gradient = np.zeros_like(dose_grid)
        
        # Update gradient for all voxels in the structure
        voxel_count = np.sum(structure_mask > 0)
        if voxel_count > 0:
            gradient[structure_mask > 0] = 2.0 * self.weight * (dose_grid[structure_mask > 0] - self.target_dose) / voxel_count
        
        return gradient


class DoseVolumeObjective(StructureBasedObjective):
    """
    Base class for dose-volume based objectives.
    
    Dose-volume objectives depend on the dose-volume histogram (DVH) of a structure.
    These objectives are typically used to constrain the volume of a structure
    receiving a certain dose.
    """
    
    def __init__(self, structure_name: str, dose: float, volume: float,
                 weight: float = 1.0, name: Optional[str] = None):
        """
        Initialize the dose-volume objective.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        dose : float
            Dose threshold in Gy
        volume : float
            Volume threshold as a percentage (0-100)
        weight : float, optional
            Weight of the objective in the total cost function
        name : str, optional
            Name of the objective function
        """
        super().__init__(structure_name, weight, name)
        self.dose = dose
        self.volume = volume


class DVHObjective(DoseVolumeObjective):
    """
    Dose-volume histogram based objective.
    
    This objective uses DVH metrics for optimization.
    """
    
    def __init__(self, structure_name: str, metric_type: str, metric_value: float,
                 target_value: float, constraint: str = "<",
                 weight: float = 1.0, name: Optional[str] = None):
        """
        Initialize the DVH objective.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure to apply the objective to
        metric_type : str
            Type of DVH metric ('Dx' or 'Vx')
        metric_value : float
            Value for the metric (x in Dx or Vx)
        target_value : float
            Target value for the objective
        constraint : str
            Constraint type ('<', '>', '=')
        weight : float
            Objective weight
        name : str, optional
            Name of the objective
        """
        super().__init__(structure_name, 0, 0, weight, name)
        self.metric_type = metric_type.upper()
        self.metric_value = metric_value
        self.target_value = target_value
        self.constraint = constraint

        # Validate inputs
        if not (self.metric_type.startswith('D') or self.metric_type.startswith('V')):
            raise ValueError(f"Metric type '{metric_type}' not supported. Use 'Dx' or 'Vx'.")
        
        # Set name if not provided
        if not name:
            self.name = f"{self.metric_type}{self.metric_value}{self.constraint}{self.target_value}"

    def __call__(self, dose_grid, structures=None, **kwargs) -> float:
        """
        Calculate the objective value.
        
        Parameters
        ----------
        dose_grid : array-like
            3D dose grid
        structures : dict, optional
            Dictionary of structure masks
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        float
            Objective value
        """
        if not structures or self.structure_name not in structures:
            logger.warning(f"Structure '{self.structure_name}' not found.")
            return 0.0
        
        # Get structure mask
        structure_mask = structures[self.structure_name]
        
        # Calculate DVH
        dvh_data = calculate_dvh_from_dose_grid(dose_grid, structure_mask)
        
        # Get actual value based on metric type
        actual_value = 0.0
        
        if self.metric_type.startswith('D'):
            # We need to adjust the dose
            dose_bins = dvh_data.get('dose_bins', [])
            volume_bins = dvh_data.get('cumulative_volume', [])
            
            if len(dose_bins) > 0 and len(volume_bins) > 0:
                actual_value = _get_dose_at_volume(
                    dose_bins, 
                    volume_bins, 
                    self.metric_value
                )
            
        elif self.metric_type.startswith('V'):
            # We need to adjust the volume
            dose_bins = dvh_data.get('dose_bins', [])
            volume_bins = dvh_data.get('cumulative_volume', [])
            
            if len(dose_bins) > 0 and len(volume_bins) > 0:
                actual_value = _get_volume_at_dose(
                    dose_bins, 
                    volume_bins, 
                    self.metric_value
                )
        
        # Calculate penalty based on constraint
        penalty = 0.0
        
        if self.constraint == '<':
            if actual_value > self.target_value:
                penalty = (actual_value - self.target_value) ** 2
        elif self.constraint == '>':
            if actual_value < self.target_value:
                penalty = (self.target_value - actual_value) ** 2
        else:  # constraint == '='
            penalty = (actual_value - self.target_value) ** 2
        
        return self.weight * penalty

    def gradient(self, dose_grid, structures=None, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the DVH objective.
        
        Parameters
        ----------
        dose_grid : array-like
            3D dose grid
        structures : dict, optional
            Dictionary of structure masks
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Gradient of the objective function
        """
        if not structures or self.structure_name not in structures:
            logger.warning(f"Structure '{self.structure_name}' not found.")
            return np.zeros_like(dose_grid)
        
        structure_mask = structures[self.structure_name]
        
        # Calculate current objective value
        current_value = self.__call__(dose_grid, structures, **kwargs)
        
        # If constraint is already satisfied, gradient is zero
        if current_value == 0.0:
            return np.zeros_like(dose_grid)
        
        # For simplicity, use a uniform gradient inside the structure
        # This is a rough approximation and not optimal for all cases
        gradient = np.zeros_like(dose_grid)
        
        if self.metric_type.startswith('D'):
            # For D metrics, we want to increase/decrease the dose
            if self.constraint == '<':
                # We want to decrease the dose
                gradient[structure_mask > 0] = 2.0 * self.weight
            elif self.constraint == '>':
                # We want to increase the dose
                gradient[structure_mask > 0] = -2.0 * self.weight
            else:  # self.constraint == '='
                # We need to adjust the dose
                dvh_data = calculate_dvh_from_dose_grid(dose_grid, structure_mask)
                actual_value = _get_dose_at_volume(
                    dvh_data.get('dose_bins', []), 
                    dvh_data.get('cumulative_volume', []), 
                    self.metric_value
                )
            
                if actual_value < self.target_value:
                    gradient[structure_mask > 0] = -2.0 * self.weight
                else:
                    gradient[structure_mask > 0] = 2.0 * self.weight
        else:  # self.metric_type.startswith('V')
            # For V metrics, we want to adjust the volume receiving a dose
            if self.constraint == '<':
                # We want to decrease the volume
                gradient[structure_mask > 0] = 2.0 * self.weight
            elif self.constraint == '>':
                # We want to increase the volume
                gradient[structure_mask > 0] = -2.0 * self.weight
            else:  # self.constraint == '='
                # We need to adjust the volume
                dvh_data = calculate_dvh_from_dose_grid(dose_grid, structure_mask)
                actual_value = _get_volume_at_dose(
                    dvh_data.get('dose_bins', []), 
                    dvh_data.get('cumulative_volume', []), 
                    self.metric_value
                )
            
                if actual_value < self.target_value:
                    gradient[structure_mask > 0] = -2.0 * self.weight
                else:
                    gradient[structure_mask > 0] = 2.0 * self.weight
        
        return gradient


def get_objective_result(objective: Union[ObjectiveFunction, Dict],
                        dose_grid, structures=None, **kwargs) -> Dict:
    """
    Evaluate an objective function and return detailed results.
    
    Parameters
    ----------
    objective : Union[ObjectiveFunction, Dict]
        Objective function or a dictionary describing it
    dose_grid : np.ndarray
        3D dose grid
    structures : Dict[str, np.ndarray], optional
        Dictionary mapping structure names to binary masks
    **kwargs : Any
        Additional parameters
        
    Returns
    -------
    Dict
        Dictionary containing objective function details and result
    """
    if isinstance(objective, dict):
        # Create objective function from dictionary
        obj_type = objective.get('type', 'UniformDose')
        structure_name = objective.get('structure', '')
        weight = objective.get('weight', 1.0)
        name = objective.get('name', None)
        
        if obj_type == 'MinDose':
            min_dose = objective.get('min_dose', 0.0)
            obj = MinDose(structure_name, min_dose, weight, name)
        elif obj_type == 'MaxDose':
            max_dose = objective.get('max_dose', 0.0)
            obj = MaxDose(structure_name, max_dose, weight, name)
        elif obj_type == 'MeanDose':
            target_dose = objective.get('target_dose', 0.0)
            obj = MeanDose(structure_name, target_dose, weight, name)
        elif obj_type == 'UniformDose':
            target_dose = objective.get('target_dose', 0.0)
            obj = UniformDose(structure_name, target_dose, weight, name)
        elif obj_type == 'DVH':
            metric_type = objective.get('metric_type', 'D')
            metric_value = objective.get('metric_value', 0.0)
            target_value = objective.get('target_value', 0.0)
            constraint = objective.get('constraint', '<')
            obj = DVHObjective(structure_name, metric_type, metric_value, 
                              target_value, constraint, weight, name)
        else:
            # Default to uniform dose
            target_dose = objective.get('target_dose', 0.0)
            obj = UniformDose(structure_name, target_dose, weight, name)
    else:
        # Use provided objective function
        obj = objective
    
    # Evaluate objective function
    value = obj(dose_grid, structures, **kwargs)
    
    # Return result
    return {
        'name': obj.name,
        'type': obj.__class__.__name__,
        'weight': obj.weight,
        'value': value,
        'weighted_value': value,
        'structure': getattr(obj, 'structure_name', None)
    }
