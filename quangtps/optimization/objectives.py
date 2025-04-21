#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimization Objectives Module
==============================

This module defines the objective functions used in IMRT/VMAT optimization,
matching the objectives available in the Eclipse treatment planning system.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ObjectiveFunction(ABC):
    """
    Abstract base class for all optimization objective functions.
    """

    def __init__(
        self,
        structure_id: str,
        structure_name: str,
        weight: float = 1.0,
        priority: int = 100,
        enabled: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize the objective function.

        Parameters
        ----------
        structure_id : str
            ID of the structure
        structure_name : str
            Name of the structure
        weight : float, optional
            Weight of the objective
        priority : int, optional
            Priority of the objective (100 is default)
        enabled : bool, optional
            Whether the objective is enabled
        normalize : bool, optional
            Whether to normalize the objective value
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.weight = weight
        self.priority = priority
        self.enabled = enabled
        self.normalize = normalize

    @abstractmethod
    def evaluate(self, dose, structure_mask) -> float:
        """
        Evaluate the objective function.

        Parameters
        ----------
        dose : ndarray
            Dose distribution
        structure_mask : ndarray
            Binary mask of the structure

        Returns
        -------
        float
            Objective function value
        """
        pass

    @abstractmethod
    def get_gradient(self, dose, structure_mask) -> np.ndarray:
        """
        Calculate the gradient of the objective function.

        Parameters
        ----------
        dose : ndarray
            Dose distribution
        structure_mask : ndarray
            Binary mask of the structure

        Returns
        -------
        ndarray
            Gradient of the objective function
        """
        pass

    def to_dict(self) -> Dict:
        """
        Convert the objective to a dictionary.

        Returns
        -------
        Dict
            Dictionary representation of the objective
        """
        return {
            "type": self.__class__.__name__,
            "structure_id": self.structure_id,
            "structure_name": self.structure_name,
            "weight": self.weight,
            "priority": self.priority,
            "enabled": self.enabled,
            "normalize": self.normalize,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ObjectiveFunction":
        """
        Create an objective from a dictionary.

        Parameters
        ----------
        data : Dict
            Dictionary with objective parameters

        Returns
        -------
        ObjectiveFunction
            Created objective function
        """
        obj_type = data.pop("type", cls.__name__)

        # Find the appropriate subclass based on type
        for subclass in ObjectiveFunction.__subclasses__():
            if subclass.__name__ == obj_type:
                return subclass(**data)

        raise ValueError(f"Unknown objective type: {obj_type}")


class DoseObjective(ObjectiveFunction):
    """Base class for dose-based objectives."""

    def __init__(self, structure_id: str, structure_name: str, dose: float, **kwargs):
        """
        Initialize a dose-based objective.

        Parameters
        ----------
        structure_id : str
            ID of the structure
        structure_name : str
            Name of the structure
        dose : float
            Reference dose in Gy
        **kwargs : dict
            Additional parameters for the base class
        """
        super().__init__(structure_id, structure_name, **kwargs)
        self.dose = dose

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = super().to_dict()
        data["dose"] = self.dose
        return data


class DVHObjective(DoseObjective):
    """Base class for DVH-based objectives."""

    def __init__(
        self,
        structure_id: str,
        structure_name: str,
        dose: float,
        volume: float,
        **kwargs,
    ):
        """
        Initialize a DVH-based objective.

        Parameters
        ----------
        structure_id : str
            ID of the structure
        structure_name : str
            Name of the structure
        dose : float
            Reference dose in Gy
        volume : float
            Reference volume percentage (0-100)
        **kwargs : dict
            Additional parameters for the base class
        """
        super().__init__(structure_id, structure_name, dose, **kwargs)
        self.volume = volume

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = super().to_dict()
        data["volume"] = self.volume
        return data


class LowerDoseObjective(DoseObjective):
    """
    Lower dose objective to ensure dose is above a threshold.
    Corresponds to Eclipse's 'Lower Dose' objective.
    """

    def evaluate(self, dose, structure_mask):
        """Evaluate the objective."""
        if np.sum(structure_mask) == 0:
            return 0.0

        # Get masked dose values
        masked_dose = dose[structure_mask > 0]

        # Calculate how much dose is below target
        below_target = np.maximum(0, self.dose - masked_dose)

        # Calculate mean squared deviation
        msd = np.mean(below_target**2)

        return self.weight * msd

    def get_gradient(self, dose, structure_mask):
        """Calculate gradient."""
        gradient = np.zeros_like(dose)
        if np.sum(structure_mask) == 0:
            return gradient

        masked_indices = structure_mask > 0
        masked_dose = dose[masked_indices]

        # Calculate gradient: -2 * (dose - target) for voxels below target
        below_target = masked_dose < self.dose
        voxel_gradients = np.zeros_like(masked_dose)
        voxel_gradients[below_target] = -2 * (masked_dose[below_target] - self.dose)

        # Apply to full gradient array
        gradient[masked_indices] = voxel_gradients * self.weight

        return gradient


class UpperDoseObjective(DoseObjective):
    """
    Upper dose objective to ensure dose is below a threshold.
    Corresponds to Eclipse's 'Upper Dose' objective.
    """

    def evaluate(self, dose, structure_mask):
        """Evaluate the objective."""
        if np.sum(structure_mask) == 0:
            return 0.0

        # Get masked dose values
        masked_dose = dose[structure_mask > 0]

        # Calculate how much dose is above target
        above_target = np.maximum(0, masked_dose - self.dose)

        # Calculate mean squared deviation
        msd = np.mean(above_target**2)

        return self.weight * msd

    def get_gradient(self, dose, structure_mask):
        """Calculate gradient."""
        gradient = np.zeros_like(dose)
        if np.sum(structure_mask) == 0:
            return gradient

        masked_indices = structure_mask > 0
        masked_dose = dose[masked_indices]

        # Calculate gradient: 2 * (dose - target) for voxels above target
        above_target = masked_dose > self.dose
        voxel_gradients = np.zeros_like(masked_dose)
        voxel_gradients[above_target] = 2 * (masked_dose[above_target] - self.dose)

        # Apply to full gradient array
        gradient[masked_indices] = voxel_gradients * self.weight

        return gradient


class MeanDoseObjective(DoseObjective):
    """
    Mean dose objective to achieve a target mean dose.
    Corresponds to Eclipse's 'Mean Dose' objective.
    """

    def evaluate(self, dose, structure_mask):
        """Evaluate the objective."""
        if np.sum(structure_mask) == 0:
            return 0.0

        # Get masked dose values
        masked_dose = dose[structure_mask > 0]

        # Calculate mean dose
        mean_dose = np.mean(masked_dose)

        # Calculate squared deviation from target
        deviation = (mean_dose - self.dose) ** 2

        return self.weight * deviation

    def get_gradient(self, dose, structure_mask):
        """Calculate gradient."""
        gradient = np.zeros_like(dose)
        if np.sum(structure_mask) == 0:
            return gradient

        masked_indices = structure_mask > 0
        masked_dose = dose[masked_indices]

        # Calculate mean dose
        mean_dose = np.mean(masked_dose)

        # Calculate gradient: 2 * (mean - target) / num_voxels
        num_voxels = np.sum(masked_indices)
        gradient_value = 2 * (mean_dose - self.dose) / num_voxels

        # Apply to full gradient array
        gradient[masked_indices] = gradient_value * self.weight

        return gradient


class MaxDoseObjective(DoseObjective):
    """
    Maximum dose objective to limit maximum dose.
    Corresponds to Eclipse's 'Maximum Dose' objective.
    """

    def evaluate(self, dose, structure_mask):
        """Evaluate the objective."""
        if np.sum(structure_mask) == 0:
            return 0.0

        # Get masked dose values
        masked_dose = dose[structure_mask > 0]

        # Calculate maximum dose
        max_dose = np.max(masked_dose)

        # Calculate how much the max is above target
        above_target = np.maximum(0, max_dose - self.dose)

        # Calculate squared deviation
        deviation = above_target**2

        return self.weight * deviation

    def get_gradient(self, dose, structure_mask):
        """Calculate gradient."""
        gradient = np.zeros_like(dose)
        if np.sum(structure_mask) == 0:
            return gradient

        masked_indices = structure_mask > 0
        masked_dose = dose[masked_indices]

        # Find the maximum dose value and its index
        max_dose = np.max(masked_dose)
        if max_dose <= self.dose:
            return gradient

        # Only apply gradient to the voxel(s) with maximum dose
        max_indices = np.where(dose == max_dose)
        gradient[max_indices] = 2 * (max_dose - self.dose) * self.weight

        return gradient


class MinDoseObjective(DoseObjective):
    """
    Minimum dose objective to ensure minimum dose.
    Corresponds to Eclipse's 'Minimum Dose' objective.
    """

    def evaluate(self, dose, structure_mask):
        """Evaluate the objective."""
        if np.sum(structure_mask) == 0:
            return 0.0

        # Get masked dose values
        masked_dose = dose[structure_mask > 0]

        # Calculate minimum dose
        min_dose = np.min(masked_dose)

        # Calculate how much the min is below target
        below_target = np.maximum(0, self.dose - min_dose)

        # Calculate squared deviation
        deviation = below_target**2

        return self.weight * deviation

    def get_gradient(self, dose, structure_mask):
        """Calculate gradient."""
        gradient = np.zeros_like(dose)
        if np.sum(structure_mask) == 0:
            return gradient

        masked_indices = structure_mask > 0
        masked_dose = dose[masked_indices]

        # Find the minimum dose value and its index
        min_dose = np.min(masked_dose)
        if min_dose >= self.dose:
            return gradient

        # Only apply gradient to the voxel(s) with minimum dose
        min_indices = np.where(dose == min_dose)
        gradient[min_indices] = -2 * (self.dose - min_dose) * self.weight

        return gradient


class LowerDVHObjective(DVHObjective):
    """
    Lower DVH objective to ensure a minimum dose to a volume.
    Corresponds to Eclipse's 'Lower DVH' objective.
    """

    def evaluate(self, dose, structure_mask):
        """Evaluate the objective."""
        if np.sum(structure_mask) == 0:
            return 0.0

        # Get masked dose values
        masked_dose = dose[structure_mask > 0]

        # Calculate DVH
        hist, bins = np.histogram(
            masked_dose, bins=100, range=(0, np.max(masked_dose) * 1.1)
        )
        cum_hist = np.cumsum(hist) / np.sum(hist) * 100

        # Find bin index for the specified volume percentage
        volume_index = np.searchsorted(cum_hist, self.volume)
        if volume_index >= len(bins) - 1:
            volume_index = len(bins) - 2

        # Get dose at the specified volume
        dose_at_volume = bins[volume_index]

        # Calculate how much the dose is below target
        below_target = np.maximum(0, self.dose - dose_at_volume)

        # Calculate squared deviation
        deviation = below_target**2

        return self.weight * deviation

    def get_gradient(self, dose, structure_mask):
        """
        Calculate gradient.
        This is more complex for DVH constraints and requires approximation.
        """
        gradient = np.zeros_like(dose)
        if np.sum(structure_mask) == 0:
            return gradient

        masked_indices = structure_mask > 0
        masked_dose = dose[masked_indices]

        # Sort doses to find the dose threshold
        sorted_dose = np.sort(masked_dose)
        index = int(np.floor(len(sorted_dose) * (100 - self.volume) / 100))
        if index >= len(sorted_dose):
            index = len(sorted_dose) - 1

        dose_threshold = sorted_dose[index]

        # If the dose at volume is already above target, no gradient
        if dose_threshold >= self.dose:
            return gradient

        # Apply gradient to voxels near the threshold
        window = 0.05 * (np.max(masked_dose) - np.min(masked_dose))

        # Create a window of influence around the threshold
        influence = np.zeros_like(masked_dose)
        influence[masked_dose > (dose_threshold - window)] = 1.0
        influence[masked_dose > (dose_threshold + window)] = 0.0

        # Apply the influence-weighted gradient
        voxel_gradients = -2 * (self.dose - dose_threshold) * influence

        # Scale by number of influenced voxels to maintain magnitude
        num_influenced = np.sum(influence > 0)
        if num_influenced > 0:
            voxel_gradients = voxel_gradients * (len(masked_dose) / num_influenced)

        # Apply to full gradient array
        gradient[masked_indices] = voxel_gradients * self.weight

        return gradient


class UpperDVHObjective(DVHObjective):
    """
    Upper DVH objective to limit dose to a volume.
    Corresponds to Eclipse's 'Upper DVH' objective.
    """

    def evaluate(self, dose, structure_mask):
        """Evaluate the objective."""
        if np.sum(structure_mask) == 0:
            return 0.0

        # Get masked dose values
        masked_dose = dose[structure_mask > 0]

        # Calculate DVH
        hist, bins = np.histogram(
            masked_dose, bins=100, range=(0, np.max(masked_dose) * 1.1)
        )
        cum_hist = np.cumsum(hist[::-1])[::-1] / np.sum(hist) * 100

        # Find bin index for the specified volume percentage
        volume_index = np.searchsorted(cum_hist, self.volume)
        if volume_index >= len(bins) - 1:
            volume_index = len(bins) - 2

        # Get dose at the specified volume
        dose_at_volume = bins[volume_index]

        # Calculate how much the dose is above target
        above_target = np.maximum(0, dose_at_volume - self.dose)

        # Calculate squared deviation
        deviation = above_target**2

        return self.weight * deviation

    def get_gradient(self, dose, structure_mask):
        """
        Calculate gradient.
        This is more complex for DVH constraints and requires approximation.
        """
        gradient = np.zeros_like(dose)
        if np.sum(structure_mask) == 0:
            return gradient

        masked_indices = structure_mask > 0
        masked_dose = dose[masked_indices]

        # Sort doses to find the dose threshold
        sorted_dose = np.sort(masked_dose)[::-1]  # Descending
        index = int(np.floor(len(sorted_dose) * self.volume / 100))
        if index >= len(sorted_dose):
            index = len(sorted_dose) - 1

        dose_threshold = sorted_dose[index]

        # If the dose at volume is already below target, no gradient
        if dose_threshold <= self.dose:
            return gradient

        # Apply gradient to voxels near the threshold
        window = 0.05 * (np.max(masked_dose) - np.min(masked_dose))

        # Create a window of influence around the threshold
        influence = np.zeros_like(masked_dose)
        influence[masked_dose < (dose_threshold + window)] = 1.0
        influence[masked_dose < (dose_threshold - window)] = 0.0

        # Apply the influence-weighted gradient
        voxel_gradients = 2 * (dose_threshold - self.dose) * influence

        # Scale by number of influenced voxels to maintain magnitude
        num_influenced = np.sum(influence > 0)
        if num_influenced > 0:
            voxel_gradients = voxel_gradients * (len(masked_dose) / num_influenced)

        # Apply to full gradient array
        gradient[masked_indices] = voxel_gradients * self.weight

        return gradient


class ConformityObjective(DoseObjective):
    """
    Conformity objective to ensure dose conforms to target.
    Corresponds to Eclipse's 'Conformity' objective.
    """

    def evaluate(self, dose, structure_mask):
        """Evaluate the objective."""
        if np.sum(structure_mask) == 0:
            return 0.0

        # Calculate conformity index (CI)
        # CI = (Volume of target receiving at least reference dose) / (Target volume)
        target_volume = np.sum(structure_mask)
        target_covered = np.sum((structure_mask > 0) & (dose >= self.dose))

        if target_volume == 0:
            return 0.0

        ci = target_covered / target_volume

        # Penalty is proportional to deviation from ideal CI of 1.0
        deviation = (1.0 - ci) ** 2

        return self.weight * deviation

    def get_gradient(self, dose, structure_mask):
        """Calculate gradient."""
        gradient = np.zeros_like(dose)
        if np.sum(structure_mask) == 0:
            return gradient

        # Identify voxels just below and just above the threshold
        masked_indices = structure_mask > 0
        masked_dose = dose[masked_indices]

        # Window around threshold for gradient influence
        window = 0.05 * np.max(masked_dose)

        # Create influence weights for voxels near the threshold
        influence = np.zeros_like(masked_dose)
        influence[
            (masked_dose > self.dose - window) & (masked_dose < self.dose + window)
        ] = 1.0

        # Calculate direction: negative for voxels below threshold
        direction = np.ones_like(masked_dose)
        direction[masked_dose < self.dose] = -1.0

        # Apply gradient
        voxel_gradients = direction * influence

        # Scale gradient
        target_volume = np.sum(structure_mask)
        voxel_gradients = voxel_gradients * (2.0 / target_volume)

        # Apply to full gradient array
        gradient[masked_indices] = voxel_gradients * self.weight

        return gradient


class HomogeneityObjective(DoseObjective):
    """
    Homogeneity objective to ensure uniform dose in target.
    Corresponds to Eclipse's 'Homogeneity' objective.
    """

    def evaluate(self, dose, structure_mask):
        """Evaluate the objective."""
        if np.sum(structure_mask) == 0:
            return 0.0

        # Get masked dose values
        masked_dose = dose[structure_mask > 0]

        # Calculate homogeneity index (HI)
        # HI = (D2% - D98%) / D50%
        if len(masked_dose) < 3:
            return 0.0

        sorted_dose = np.sort(masked_dose)
        d2_index = int(np.ceil(0.98 * len(sorted_dose)))
        d98_index = int(np.floor(0.02 * len(sorted_dose)))
        d50_index = int(0.5 * len(sorted_dose))

        d2 = sorted_dose[d2_index]
        d98 = sorted_dose[d98_index]
        d50 = sorted_dose[d50_index]

        if d50 == 0:
            return 0.0

        hi = (d2 - d98) / d50

        # Penalty is proportional to HI (lower is better)
        penalty = hi**2

        return self.weight * penalty

    def get_gradient(self, dose, structure_mask):
        """Calculate gradient."""
        gradient = np.zeros_like(dose)
        if np.sum(structure_mask) == 0:
            return gradient

        # Get masked dose values
        masked_indices = structure_mask > 0
        masked_dose = dose[masked_indices]

        # Find D2%, D98%, D50%
        sorted_indices = np.argsort(masked_dose)
        d2_index = int(np.ceil(0.98 * len(sorted_indices)))
        d98_index = int(np.floor(0.02 * len(sorted_indices)))
        d50_index = int(0.5 * len(sorted_indices))

        if (
            d2_index >= len(sorted_indices)
            or d98_index >= len(sorted_indices)
            or d50_index >= len(sorted_indices)
        ):
            return gradient

        d2_idx = sorted_indices[d2_index]
        d98_idx = sorted_indices[d98_index]
        d50_idx = sorted_indices[d50_index]

        d2 = masked_dose[d2_idx]
        d98 = masked_dose[d98_idx]
        d50 = masked_dose[d50_idx]

        if d50 == 0:
            return gradient

        # Create sparse gradient
        voxel_gradients = np.zeros_like(masked_dose)

        # Gradient for D2%: decrease
        voxel_gradients[d2_idx] = 2.0 * (d2 - d98) / (d50**2)

        # Gradient for D98%: increase
        voxel_gradients[d98_idx] = -2.0 * (d2 - d98) / (d50**2)

        # Gradient for D50%: adjust to minimize HI
        voxel_gradients[d50_idx] = -2.0 * (d2 - d98) ** 2 / (d50**3)

        # Apply to full gradient array
        gradient[masked_indices] = voxel_gradients * self.weight

        return gradient


class ObjectiveCollection:
    """Collection of objective functions."""

    def __init__(self):
        """Initialize an empty collection."""
        self.objectives = []

    def add_objective(self, objective):
        """
        Add an objective to the collection.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective to add
        """
        if not isinstance(objective, ObjectiveFunction):
            raise TypeError("Objective must be an instance of ObjectiveFunction")

        self.objectives.append(objective)

    def remove_objective(self, index):
        """
        Remove an objective from the collection.

        Parameters
        ----------
        index : int
            Index of the objective to remove
        """
        if 0 <= index < len(self.objectives):
            del self.objectives[index]

    def clear(self):
        """Clear all objectives."""
        self.objectives = []

    def evaluate(self, dose, structures):
        """
        Evaluate all objectives.

        Parameters
        ----------
        dose : ndarray
            Dose distribution
        structures : dict
            Dictionary mapping structure IDs to masks

        Returns
        -------
        float
            Total objective function value
        dict
            Dictionary with individual objective values
        """
        total_value = 0.0
        objective_values = {}

        for i, obj in enumerate(self.objectives):
            if not obj.enabled:
                continue

            # Skip if structure mask is not available
            if obj.structure_id not in structures:
                continue

            structure_mask = structures[obj.structure_id]
            value = obj.evaluate(dose, structure_mask)

            # Store value with some identifier
            obj_name = f"{obj.__class__.__name__}_{obj.structure_name}_{i}"
            objective_values[obj_name] = value

            # Add to total
            total_value += value

        return total_value, objective_values

    def get_gradient(self, dose, structures):
        """
        Calculate the gradient of all objectives.

        Parameters
        ----------
        dose : ndarray
            Dose distribution
        structures : dict
            Dictionary mapping structure IDs to masks

        Returns
        -------
        ndarray
            Total gradient
        """
        gradient = np.zeros_like(dose)

        for obj in self.objectives:
            if not obj.enabled:
                continue

            # Skip if structure mask is not available
            if obj.structure_id not in structures:
                continue

            structure_mask = structures[obj.structure_id]
            obj_gradient = obj.get_gradient(dose, structure_mask)

            # Add to total gradient
            gradient += obj_gradient

        return gradient

    def to_dict(self):
        """
        Convert the collection to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the collection
        """
        return {"objectives": [obj.to_dict() for obj in self.objectives]}

    @classmethod
    def from_dict(cls, data):
        """
            Create a collection from a dictionary.

        Parameters
        ----------
            data : dict
                Dictionary with collection data

        Returns
        -------
            ObjectiveCollection
                Created collection
        """
        collection = cls()

        for obj_data in data.get("objectives", []):
            obj = ObjectiveFunction.from_dict(obj_data)
            collection.add_objective(obj)

        return collection

    def __len__(self):
        """Get the number of objectives."""
        return len(self.objectives)

    def __getitem__(self, index):
        """Get an objective by index."""
        return self.objectives[index]


def create_objective(
    objective_type: str, structure_id: str, structure_name: str, **kwargs
) -> ObjectiveFunction:
    """
    Create an objective function of the specified type.

    Parameters
    ----------
    objective_type : str
        Type of objective to create (e.g., 'UpperDoseObjective', 'LowerDVHObjective')
    structure_id : str
        ID of the structure
    structure_name : str
        Name of the structure
    **kwargs : dict
        Additional parameters for the objective function

    Returns
    -------
    ObjectiveFunction
        Created objective function

    Raises
    ------
    ValueError
        If the objective type is unknown
    """
    # Map string names to classes
    objective_types = {
        "LowerDoseObjective": LowerDoseObjective,
        "UpperDoseObjective": UpperDoseObjective,
        "MeanDoseObjective": MeanDoseObjective,
        "MaxDoseObjective": MaxDoseObjective,
        "MinDoseObjective": MinDoseObjective,
        "LowerDVHObjective": LowerDVHObjective,
        "UpperDVHObjective": UpperDVHObjective,
        "ConformityObjective": ConformityObjective,
        "HomogeneityObjective": HomogeneityObjective,
    }

    # Check if objective type exists
    if objective_type not in objective_types:
        available_types = ", ".join(objective_types.keys())
        raise ValueError(
            f"Unknown objective type: {objective_type}. Available types: {available_types}"
        )

    # Get the class
    objective_class = objective_types[objective_type]

    # Create the objective
    return objective_class(
        structure_id=structure_id, structure_name=structure_name, **kwargs
    )
