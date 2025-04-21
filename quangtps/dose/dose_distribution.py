"""
Module for dose distribution management in QuangTPS.

This module provides classes for storing, manipulating, and evaluating
radiation dose distributions for radiotherapy treatment planning.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import SimpleITK as sitk

from quangtps.dose.dose_grid import DoseGrid, DoseUnit
from quangtps.core.utils import create_unique_id

logger = logging.getLogger(__name__)


class DoseDistribution:
    """
    Class representing a dose distribution for radiotherapy treatment planning.

    This class encapsulates a 3D dose grid and provides methods for manipulating
    and evaluating the dose distribution, such as calculating DVHs, dose statistics,
    and isodose information.

    Attributes:
        id (str): Unique identifier for the dose distribution
        name (str): Human-readable name for the dose distribution
        dose_grid (DoseGrid): 3D dose grid containing dose values
        prescription_dose (float): Prescription dose in Gy
        plan_name (str): Name of the associated treatment plan
        plan_id (str): ID of the associated treatment plan
        grid_size (tuple): Dimensions of the dose grid (dx, dy, dz)
        normalization_value (float): Factor used to normalize the dose distribution
        isodose_levels (list): List of isodose levels (in % of prescription dose)
    """

    def __init__(
        self, name: str = "Dose Distribution", dose_grid: Optional[DoseGrid] = None
    ):
        """
        Initialize a new dose distribution.

        Args:
            name: Human-readable name for the dose distribution
            dose_grid: 3D dose grid containing dose values (optional)
        """
        self.id = create_unique_id()
        self.name = name
        self.dose_grid = dose_grid or DoseGrid()
        self.prescription_dose = 0.0
        self.plan_name = ""
        self.plan_id = ""
        self.grid_size = (0.0, 0.0, 0.0)
        self.normalization_value = 1.0
        self.isodose_levels = [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]

        # Set default grid size based on dose grid spacing
        if dose_grid:
            self.grid_size = dose_grid.spacing

    def set_dose_grid(self, dose_grid: DoseGrid):
        """
        Set the dose grid for this distribution.

        Args:
            dose_grid: 3D dose grid containing dose values
        """
        self.dose_grid = dose_grid
        self.grid_size = dose_grid.spacing

    def get_dose_grid(self) -> DoseGrid:
        """
        Get the dose grid.

        Returns:
            The dose grid for this distribution
        """
        return self.dose_grid

    def set_prescription_dose(self, dose_gy: float):
        """
        Set the prescription dose.

        Args:
            dose_gy: Prescription dose in Gy
        """
        self.prescription_dose = dose_gy

    def set_plan_info(self, plan_name: str, plan_id: str):
        """
        Set plan information for this dose distribution.

        Args:
            plan_name: Name of the associated treatment plan
            plan_id: ID of the associated treatment plan
        """
        self.plan_name = plan_name
        self.plan_id = plan_id

    def set_normalization(self, normalization_value: float):
        """
        Set normalization factor and apply it to the dose grid.

        Args:
            normalization_value: Factor to normalize the dose distribution
        """
        if normalization_value <= 0:
            logger.warning(
                f"Invalid normalization value: {normalization_value}. Using 1.0 instead."
            )
            normalization_value = 1.0

        # Apply normalization by scaling the dose grid
        scale_factor = normalization_value / self.normalization_value
        if self.dose_grid.grid_data is not None:
            self.dose_grid.grid_data *= scale_factor

            # Update SimpleITK image
            self.dose_grid.sitk_image = sitk.GetImageFromArray(self.dose_grid.grid_data)
            self.dose_grid.sitk_image.SetOrigin(self.dose_grid.origin)
            self.dose_grid.sitk_image.SetSpacing(self.dose_grid.spacing)
            self.dose_grid.sitk_image.SetDirection(self.dose_grid.direction)

        # Update normalization value
        self.normalization_value = normalization_value

    def get_dose_at_point(self, point: Tuple[float, float, float]) -> float:
        """
        Get dose value at a specific point in 3D space.

        Args:
            point: (x, y, z) coordinates in the patient coordinate system

        Returns:
            Dose value in Gy at the specified point
        """
        if self.dose_grid is None:
            return 0.0
        return self.dose_grid.get_dose_at_point(point)

    def get_dose_at_index(self, index: Tuple[int, int, int]) -> float:
        """
        Get dose value at a specific grid index.

        Args:
            index: (i, j, k) indices in the dose grid

        Returns:
            Dose value in Gy at the specified index
        """
        if self.dose_grid is None:
            return 0.0
        return self.dose_grid.get_dose_at_index(index)

    def get_dvh(
        self, structure_mask: np.ndarray, bins: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate dose-volume histogram (DVH) for a structure.

        Args:
            structure_mask: Binary 3D array with same dimensions as dose grid
            bins: Number of dose bins in the histogram

        Returns:
            Tuple of (dose_bins, volume_bins)
        """
        if self.dose_grid is None or self.dose_grid.grid_data is None:
            return np.array([]), np.array([])

        return self.dose_grid.get_dose_volume_histogram(structure_mask, bins)

    def get_min_dose(self, structure_mask: Optional[np.ndarray] = None) -> float:
        """
        Get minimum dose in a structure or in the entire distribution.

        Args:
            structure_mask: Binary 3D array with same dimensions as dose grid (optional)

        Returns:
            Minimum dose in Gy
        """
        if self.dose_grid is None or self.dose_grid.grid_data is None:
            return 0.0

        return self.dose_grid.get_min_dose(structure_mask)

    def get_max_dose(self, structure_mask: Optional[np.ndarray] = None) -> float:
        """
        Get maximum dose in a structure or in the entire distribution.

        Args:
            structure_mask: Binary 3D array with same dimensions as dose grid (optional)

        Returns:
            Maximum dose in Gy
        """
        if self.dose_grid is None or self.dose_grid.grid_data is None:
            return 0.0

        return self.dose_grid.get_max_dose(structure_mask)

    def get_mean_dose(self, structure_mask: np.ndarray) -> float:
        """
        Get mean dose in a structure.

        Args:
            structure_mask: Binary 3D array with same dimensions as dose grid

        Returns:
            Mean dose in Gy
        """
        if self.dose_grid is None or self.dose_grid.grid_data is None:
            return 0.0

        return self.dose_grid.get_mean_dose(structure_mask)

    def get_dose_at_volume(
        self, structure_mask: np.ndarray, volume_percent: float
    ) -> float:
        """
        Get dose that covers a specific volume percentage of a structure (Dx).

        Args:
            structure_mask: Binary 3D array with same dimensions as dose grid
            volume_percent: Volume percentage (0-100)

        Returns:
            Dose in Gy
        """
        if self.dose_grid is None or self.dose_grid.grid_data is None:
            return 0.0

        return self.dose_grid.get_dose_at_volume(structure_mask, volume_percent)

    def get_volume_at_dose(
        self, structure_mask: np.ndarray, dose_value: float
    ) -> float:
        """
        Get volume percentage receiving at least a specific dose (Vx).

        Args:
            structure_mask: Binary 3D array with same dimensions as dose grid
            dose_value: Dose threshold in Gy

        Returns:
            Volume percentage (0-100)
        """
        if self.dose_grid is None or self.dose_grid.grid_data is None:
            return 0.0

        return self.dose_grid.get_volume_at_dose(structure_mask, dose_value)

    def get_isodose_surface(self, iso_level: float) -> np.ndarray:
        """
        Get binary mask of voxels receiving at least the specified dose.

        Args:
            iso_level: Isodose level in Gy

        Returns:
            Binary 3D array with same dimensions as dose grid
        """
        if self.dose_grid is None or self.dose_grid.grid_data is None:
            return np.array([])

        return self.dose_grid.grid_data >= iso_level

    def get_isodose_percentage(self, percentage: float) -> float:
        """
        Get dose value corresponding to a percentage of prescription dose.

        Args:
            percentage: Percentage of prescription dose (0-100)

        Returns:
            Dose in Gy
        """
        if self.prescription_dose <= 0:
            logger.warning(
                "Prescription dose is not set. Using max dose for percentage calculation."
            )
            if self.dose_grid and self.dose_grid.grid_data is not None:
                max_dose = np.max(self.dose_grid.grid_data)
                return max_dose * percentage / 100.0
            return 0.0

        return self.prescription_dose * percentage / 100.0

    def get_conformity_index(
        self, target_mask: np.ndarray, reference_dose_percent: float = 95.0
    ) -> float:
        """
        Calculate the Conformity Index (CI) for a target structure.

        CI = (V_ref / V_target) * (V_ref / V_body_ref), where:
        - V_ref is the volume of the target receiving at least the reference dose
        - V_target is the total volume of the target
        - V_body_ref is the volume of the body receiving at least the reference dose

        In this simplified version, V_body_ref is approximated as the volume of
        the reference isodose in the entire dose distribution.

        Args:
            target_mask: Binary 3D array representing the target structure
            reference_dose_percent: Reference dose as a percentage of prescription dose

        Returns:
            Conformity Index value
        """
        if self.dose_grid is None or self.dose_grid.grid_data is None:
            return 0.0

        # Convert percentage to absolute dose
        reference_dose = self.get_isodose_percentage(reference_dose_percent)

        # Calculate total target volume
        target_volume = np.sum(target_mask)
        if target_volume == 0:
            return 0.0

        # Calculate volume of target receiving at least the reference dose
        target_vol_ref = np.sum(
            np.logical_and(target_mask, self.dose_grid.grid_data >= reference_dose)
        )

        # Calculate volume of entire body receiving at least the reference dose
        body_vol_ref = np.sum(self.dose_grid.grid_data >= reference_dose)

        if body_vol_ref == 0:
            return 0.0

        # Calculate CI
        ci = (target_vol_ref / target_volume) * (target_vol_ref / body_vol_ref)

        return ci

    def get_homogeneity_index(self, target_mask: np.ndarray) -> float:
        """
        Calculate the Homogeneity Index (HI) for a target structure.

        HI = (D2 - D98) / D50, where:
        - D2 is the dose received by 2% of the target volume
        - D98 is the dose received by 98% of the target volume
        - D50 is the dose received by 50% of the target volume

        Args:
            target_mask: Binary 3D array representing the target structure

        Returns:
            Homogeneity Index value
        """
        if self.dose_grid is None or self.dose_grid.grid_data is None:
            return 0.0

        # Calculate doses at specific volumes
        d2 = self.get_dose_at_volume(target_mask, 2.0)
        d98 = self.get_dose_at_volume(target_mask, 98.0)
        d50 = self.get_dose_at_volume(target_mask, 50.0)

        if d50 == 0:
            return 0.0

        # Calculate HI
        hi = (d2 - d98) / d50

        return hi

    def copy(self) -> "DoseDistribution":
        """
        Create a deep copy of this dose distribution.

        Returns:
            A new DoseDistribution object with the same attributes
        """
        # Create a copy of the dose grid
        if self.dose_grid and self.dose_grid.grid_data is not None:
            new_grid = DoseGrid(
                grid_data=np.copy(self.dose_grid.grid_data),
                origin=self.dose_grid.origin,
                spacing=self.dose_grid.spacing,
                direction=self.dose_grid.direction,
            )
        else:
            new_grid = DoseGrid()

        # Create a new dose distribution
        new_dist = DoseDistribution(name=self.name, dose_grid=new_grid)
        new_dist.prescription_dose = self.prescription_dose
        new_dist.plan_name = self.plan_name
        new_dist.plan_id = self.plan_id
        new_dist.grid_size = self.grid_size
        new_dist.normalization_value = self.normalization_value
        new_dist.isodose_levels = self.isodose_levels.copy()

        return new_dist
