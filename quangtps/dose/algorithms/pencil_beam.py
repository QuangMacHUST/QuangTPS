#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of the Pencil Beam dose calculation algorithm.

This module provides a class for calculating dose distributions using
the Pencil Beam algorithm for radiotherapy treatment planning.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

from quangtps.core.exceptions import DoseCalculationError
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.dose.beam_data_processor import BeamModel, BeamModelParameter

logger = logging.getLogger(__name__)

class PencilBeamAlgorithm:
    """
    Implementation of the Pencil Beam dose calculation algorithm.
    
    This class provides methods for calculating 3D dose distributions
    using the Pencil Beam algorithm for radiotherapy treatment planning.
    """
    
    def __init__(self):
        """
        Initialize the Pencil Beam algorithm.
        """
        self.beam_model = None
        self.calculate_heterogeneity = True
        self.grid_size = 0.25  # Calculation grid size in cm
        self.threads = 8  # Number of parallel threads
        logger.info("Initialized Pencil Beam algorithm")
    
    def set_beam_model(self, beam_model: BeamModel):
        """
        Set the beam model for dose calculation.
        
        Parameters
        ----------
        beam_model : BeamModel
            The beam model containing beam data for dose calculation
        """
        self.beam_model = beam_model
        logger.info(f"Set beam model: {beam_model.name}")
    
    def set_heterogeneity_correction(self, enabled: bool):
        """
        Enable or disable heterogeneity correction.
        
        Parameters
        ----------
        enabled : bool
            Flag to enable or disable heterogeneity correction
        """
        self.calculate_heterogeneity = enabled
        status = "enabled" if enabled else "disabled"
        logger.info(f"Heterogeneity correction {status}")
    
    def set_calculation_parameters(self, grid_size: float = 0.25, threads: int = 8):
        """
        Set calculation parameters.
        
        Parameters
        ----------
        grid_size : float
            Calculation grid size in cm
        threads : int
            Number of parallel threads for calculation
        """
        self.grid_size = grid_size
        self.threads = threads
        logger.info(f"Set calculation parameters: grid_size={grid_size}cm, threads={threads}")
    
    def calculate_beam_dose(self, beam: Beam, ct_image: Image) -> Image:
        """
        Calculate dose for a single beam.
        
        Parameters
        ----------
        beam : Beam
            The beam to calculate dose for
        ct_image : Image
            The CT image for dose calculation
            
        Returns
        -------
        Image
            The calculated dose image
        
        Raises
        ------
        DoseCalculationError
            If dose calculation fails
        """
        if self.beam_model is None:
            logger.error("No beam model set for dose calculation")
            raise DoseCalculationError("No beam model set for dose calculation")
        
        try:
            logger.info(f"Calculating dose for beam: {beam.name}")
            
            # Convert CT to electron density
            electron_density = self._convert_ct_to_density(ct_image)
            
            # Create dose grid
            dose_grid = self._initialize_dose_grid(ct_image)
            
            # Get beam parameters
            source_position = beam.get_source_position()
            isocenter = beam.isocenter
            field_size = beam.field_size
            gantry_angle = beam.gantry_angle
            collimator_angle = beam.collimator_angle
            
            logger.info(f"Beam parameters: gantry={gantry_angle}°, field={field_size[0]}x{field_size[1]}cm, " 
                        f"collimator={collimator_angle}°")
            
            # Calculate TERMA (Total Energy Released per unit MAss)
            terma_grid = self._calculate_terma(
                electron_density, 
                dose_grid.shape, 
                source_position, 
                isocenter, 
                field_size,
                gantry_angle,
                collimator_angle
            )
            
            # Convolve TERMA with dose deposition kernel
            dose_data = self._convolve_with_kernel(terma_grid, electron_density)
            
            # Create dose image
            dose_image = Image(
                data=dose_data,
                spacing=ct_image.spacing,
                origin=ct_image.origin,
                direction=ct_image.direction
            )
            
            # Normalize to isocenter
            self._normalize_to_isocenter(dose_image, isocenter)
            
            return dose_image
            
        except Exception as e:
            error_msg = f"Error in Pencil Beam dose calculation: {str(e)}"
            logger.error(error_msg)
            raise DoseCalculationError(error_msg) from e
    
    def create_generic_beam_model(self, energy: str) -> BeamModel:
        """
        Create a generic beam model for the specified energy.
        
        Parameters
        ----------
        energy : str
            The beam energy (e.g., "6MV", "10MV")
            
        Returns
        -------
        BeamModel
            A generic beam model
        """
        logger.info(f"Creating generic beam model for energy: {energy}")
        
        # Create basic beam model
        model = BeamModel(
            name=f"Generic {energy}",
            energy=energy,
            beam_type="PHOTON"
        )
        
        # Add PDD data for 10x10 field
        depths = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        
        # Different PDD values based on energy
        if energy == "6MV":
            pdd_values = np.array([100.0, 99.5, 98.0, 95.0, 90.0, 85.0, 80.0, 60.0, 40.0, 30.0, 20.0, 15.0])
        elif energy == "10MV":
            pdd_values = np.array([100.0, 99.8, 99.0, 97.0, 94.0, 90.0, 87.0, 70.0, 50.0, 35.0, 25.0, 20.0])
        else:  # Default to 15MV
            pdd_values = np.array([100.0, 100.0, 99.5, 98.0, 96.0, 94.0, 91.0, 75.0, 55.0, 40.0, 30.0, 25.0])
        
        pdd_parameter = BeamModelParameter(
            name="pdd_10x10",
            value_grid=pdd_values,
            dimensions=["depth"],
            units=["cm"],
            dimension_values=[depths],
            interpolation_method="cubic"
        )
        model.add_parameter(pdd_parameter)
        
        # Add profile data for 10x10 field at 10cm depth
        positions = np.array([-20, -15, -10, -8, -6, -5, -4, -3, -2, -1, 0, 
                              1, 2, 3, 4, 5, 6, 8, 10, 15, 20])
        
        # Generic profile (symmetric)
        profile_values = np.array([2, 3, 5, 10, 40, 70, 90, 95, 98, 99.5, 100, 
                                   99.5, 98, 95, 90, 70, 40, 10, 5, 3, 2])
        
        profile_parameter = BeamModelParameter(
            name="profile_10x10_10cm",
            value_grid=profile_values,
            dimensions=["off_axis"],
            units=["cm"],
            dimension_values=[positions],
            interpolation_method="cubic"
        )
        model.add_parameter(profile_parameter)
        
        # Add output factors for various field sizes
        field_sizes_x = np.array([3, 5, 10, 15, 20, 30, 40])
        field_sizes_y = np.array([3, 5, 10, 15, 20, 30, 40])
        
        # Generic output factors (example values)
        of_grid = np.array([
            [0.85, 0.87, 0.90, 0.92, 0.93, 0.95, 0.96],  # 3x3, 3x5, 3x10, etc.
            [0.87, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98],  # 5x3, 5x5, 5x10, etc.
            [0.90, 0.93, 1.00, 1.02, 1.03, 1.05, 1.06],  # 10x3, 10x5, 10x10, etc.
            [0.92, 0.95, 1.02, 1.04, 1.05, 1.07, 1.08],
            [0.93, 0.96, 1.03, 1.05, 1.06, 1.08, 1.09],
            [0.95, 0.97, 1.05, 1.07, 1.08, 1.10, 1.11],
            [0.96, 0.98, 1.06, 1.08, 1.09, 1.11, 1.12]
        ])
        
        of_parameter = BeamModelParameter(
            name="output_factors",
            value_grid=of_grid,
            dimensions=["field_size_y", "field_size_x"],
            units=["cm", "cm"],
            dimension_values=[field_sizes_y, field_sizes_x],
            interpolation_method="linear"
        )
        model.add_parameter(of_parameter)
        
        return model
    
    def _convert_ct_to_density(self, ct_image: Image) -> np.ndarray:
        """
        Convert CT image (in HU) to electron density relative to water.
        
        Parameters
        ----------
        ct_image : Image
            The CT image in Hounsfield Units
            
        Returns
        -------
        np.ndarray
            Electron density relative to water
        """
        # Simple linear conversion from HU to relative electron density
        # For a more accurate calculation, a proper CT calibration curve should be used
        hu_values = ct_image.data
        
        # Basic conversion: 
        # - Air (-1000 HU) to ~0 density
        # - Water (0 HU) to 1.0 density
        # - Bone (1000 HU) to ~1.8 density
        rel_e_density = 1.0 + hu_values / 1000.0
        
        # Set minimum to a small positive number to avoid division by zero
        rel_e_density = np.maximum(rel_e_density, 0.001)
        
        return rel_e_density
    
    def _initialize_dose_grid(self, ct_image: Image) -> np.ndarray:
        """
        Initialize an empty dose grid with the same dimensions as the CT image.
        
        Parameters
        ----------
        ct_image : Image
            The CT image
            
        Returns
        -------
        np.ndarray
            Empty dose grid
        """
        return np.zeros_like(ct_image.data)
    
    def _calculate_terma(self, 
                         electron_density: np.ndarray, 
                         grid_shape: Tuple[int, int, int],
                         source_position: np.ndarray,
                         isocenter: np.ndarray,
                         field_size: Tuple[float, float],
                         gantry_angle: float,
                         collimator_angle: float) -> np.ndarray:
        """
        Calculate TERMA (Total Energy Released per unit MAss) grid.
        
        Parameters
        ----------
        electron_density : np.ndarray
            Electron density grid
        grid_shape : Tuple[int, int, int]
            Shape of the output grid
        source_position : np.ndarray
            Source position in world coordinates
        isocenter : np.ndarray
            Isocenter position in world coordinates
        field_size : Tuple[float, float]
            Field size at isocenter in cm
        gantry_angle : float
            Gantry angle in degrees
        collimator_angle : float
            Collimator angle in degrees
            
        Returns
        -------
        np.ndarray
            TERMA grid
        """
        # This is a simplified implementation for demonstration
        # A full implementation would include:
        # - Ray tracing from source through each voxel
        # - Attenuation calculation based on electron density
        # - Field shape and size consideration
        # - Off-axis factors
        
        # Create empty TERMA grid
        terma = np.zeros(grid_shape)
        
        # Simplified model: exponential attenuation from entrance
        # This is an approximation - real implementation would be more complex
        
        # Get beam direction based on gantry angle
        gantry_rad = np.radians(gantry_angle)
        beam_dir = np.array([
            np.sin(gantry_rad),
            0,
            -np.cos(gantry_rad)
        ])
        
        # Calculate a simple planar TERMA
        # This is just a placeholder for the real calculation
        for z in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                for x in range(grid_shape[2]):
                    # Calculate position relative to isocenter
                    rel_pos = np.array([z, y, x]) - isocenter
                    
                    # Project onto beam direction
                    depth = np.dot(rel_pos, beam_dir)
                    
                    # Get perpendicular distance to central axis
                    perpendicular = rel_pos - depth * beam_dir
                    dist_to_axis = np.linalg.norm(perpendicular)
                    
                    # Simple field check (rectangular field)
                    if dist_to_axis < field_size[0] / 2:
                        # Depth-based attenuation
                        if depth > 0:  # Only calculate forward from source
                            # Simplified exponential attenuation
                            mu_water = 0.05  # Approximate attenuation coefficient for water (1/cm)
                            terma[z, y, x] = 100.0 * np.exp(-mu_water * depth * electron_density[z, y, x])
        
        return terma
    
    def _convolve_with_kernel(self, terma: np.ndarray, electron_density: np.ndarray) -> np.ndarray:
        """
        Convolve TERMA with dose deposition kernel to get dose.
        
        Parameters
        ----------
        terma : np.ndarray
            TERMA grid
        electron_density : np.ndarray
            Electron density grid
            
        Returns
        -------
        np.ndarray
            Dose grid
        """
        # This is a simplified implementation
        # A full implementation would include a proper dose deposition kernel
        # (e.g., point kernel, pencil beam kernel) and proper convolution
        
        # For simplicity, we'll use a simple 3D Gaussian kernel as an approximation
        kernel_size = 5
        sigma = 1.0
        
        # Create a simple Gaussian kernel
        kernel = np.zeros((kernel_size, kernel_size, kernel_size))
        center = kernel_size // 2
        
        for z in range(kernel_size):
            for y in range(kernel_size):
                for x in range(kernel_size):
                    dist_sq = (z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2
                    kernel[z, y, x] = np.exp(-dist_sq / (2 * sigma ** 2))
        
        # Normalize the kernel
        kernel = kernel / np.sum(kernel)
        
        # Apply convolution
        from scipy.ndimage import convolve
        dose = convolve(terma, kernel, mode='constant', cval=0.0)
        
        # Apply density scaling if heterogeneity correction is enabled
        if self.calculate_heterogeneity:
            dose = dose / np.maximum(electron_density, 0.001)
        
        return dose
    
    def _normalize_to_isocenter(self, dose_image: Image, isocenter: np.ndarray):
        """
        Normalize dose so that the isocenter receives 100% dose.
        
        Parameters
        ----------
        dose_image : Image
            Dose image to normalize
        isocenter : np.ndarray
            Isocenter position in voxel coordinates
        """
        # Convert isocenter to voxel indices
        iso_indices = dose_image.world_to_voxel(isocenter)
        
        # Get dose at isocenter
        iso_z, iso_y, iso_x = np.round(iso_indices).astype(int)
        
        # Ensure within bounds
        iso_z = np.clip(iso_z, 0, dose_image.data.shape[0]-1)
        iso_y = np.clip(iso_y, 0, dose_image.data.shape[1]-1)
        iso_x = np.clip(iso_x, 0, dose_image.data.shape[2]-1)
        
        iso_dose = dose_image.data[iso_z, iso_y, iso_x]
        
        if iso_dose > 0:
            # Normalize to isocenter
            dose_image.data = dose_image.data * (100.0 / iso_dose)
            logger.info(f"Normalized dose to isocenter. Original value: {iso_dose:.2f}")
        else:
            logger.warning("Zero dose at isocenter, cannot normalize")
