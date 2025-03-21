#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of the Collapsed Cone Convolution/Superposition algorithm.

This module provides a class for calculating dose distributions using
the Collapsed Cone Convolution/Superposition algorithm for radiotherapy 
treatment planning.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

from quangtps.core.exceptions import DoseCalculationError
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.dose.beam_data_processor import BeamModel, BeamModelParameter

logger = logging.getLogger(__name__)

class CollapsedConeAlgorithm:
    """
    Implementation of the Collapsed Cone Convolution/Superposition algorithm.
    
    This class provides methods for calculating 3D dose distributions
    using the CCC algorithm for radiotherapy treatment planning.
    """
    
    def __init__(self):
        """
        Initialize the Collapsed Cone Convolution algorithm.
        """
        self.beam_model = None
        self.calculate_heterogeneity = True
        self.grid_size = 0.25  # Calculation grid size in cm
        self.threads = 8  # Number of parallel threads
        self.num_cones = 32  # Number of cones in the discrete approximation
        
        # Angular distribution of cones as (theta, phi) pairs
        self._initialize_cone_directions()
        
        logger.info("Initialized Collapsed Cone Convolution algorithm")
    
    def _initialize_cone_directions(self):
        """
        Initialize the discrete cone directions for collapsed cone calculation.
        """
        # This creates an approximate uniform distribution of directions over a sphere
        # using golden spiral method
        n = self.num_cones
        self.cone_directions = np.zeros((n, 3))
        
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(n):
            # Golden spiral distribution
            z = 1 - (2 * i + 1) / n
            radius = np.sqrt(1 - z * z)
            
            # Golden angle increment
            theta = 2 * np.pi * i / golden_ratio
            
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            
            # Store the direction
            self.cone_directions[i] = [x, y, z]
            
        # Normalize the directions
        for i in range(n):
            self.cone_directions[i] = self.cone_directions[i] / np.linalg.norm(self.cone_directions[i])
    
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
    
    def set_calculation_parameters(self, grid_size: float = 0.25, threads: int = 8, num_cones: int = 32):
        """
        Set calculation parameters.
        
        Parameters
        ----------
        grid_size : float
            Calculation grid size in cm
        threads : int
            Number of parallel threads for calculation
        num_cones : int
            Number of discrete cones for the collapsed cone approximation
        """
        self.grid_size = grid_size
        self.threads = threads
        
        if num_cones != self.num_cones:
            self.num_cones = num_cones
            self._initialize_cone_directions()
            
        logger.info(f"Set calculation parameters: grid_size={grid_size}cm, threads={threads}, num_cones={num_cones}")
    
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
            
            # Calculate primary fluence
            primary_fluence = self._calculate_primary_fluence(
                ct_image.shape,
                ct_image.spacing,
                source_position,
                isocenter, 
                field_size,
                gantry_angle,
                collimator_angle
            )
            
            # Calculate TERMA (Total Energy Released per unit MAss)
            terma_grid = self._calculate_terma(
                primary_fluence,
                electron_density,
                source_position
            )
            
            # Apply collapsed cone convolution
            dose_data = self._collapsed_cone_convolution(terma_grid, electron_density, ct_image.spacing)
            
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
            error_msg = f"Error in Collapsed Cone dose calculation: {str(e)}"
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
        
        # Add energy spectrum for CCC algorithm
        energies = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Different spectrum based on beam energy
        if energy == "6MV":
            spectrum = np.array([0.01, 0.05, 0.15, 0.25, 0.30, 0.15, 0.06, 0.02, 0.01])
        elif energy == "10MV":
            spectrum = np.array([0.01, 0.04, 0.10, 0.20, 0.25, 0.20, 0.10, 0.06, 0.04])
        else:  # Default to 15MV
            spectrum = np.array([0.01, 0.03, 0.07, 0.15, 0.20, 0.20, 0.15, 0.10, 0.09])
        
        # Normalize spectrum
        spectrum = spectrum / np.sum(spectrum)
        
        spectrum_parameter = BeamModelParameter(
            name="energy_spectrum",
            value_grid=spectrum,
            dimensions=["energy"],
            units=["MeV"],
            dimension_values=[energies],
            interpolation_method="linear"
        )
        model.add_parameter(spectrum_parameter)
        
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
    
    def _calculate_primary_fluence(self,
                                  grid_shape: Tuple[int, int, int],
                                  grid_spacing: Tuple[float, float, float],
                                  source_position: np.ndarray,
                                  isocenter: np.ndarray,
                                  field_size: Tuple[float, float],
                                  gantry_angle: float,
                                  collimator_angle: float) -> np.ndarray:
        """
        Calculate primary fluence distribution at each voxel.
        
        Parameters
        ----------
        grid_shape : Tuple[int, int, int]
            Shape of the output grid
        grid_spacing : Tuple[float, float, float]
            Spacing of the grid in cm
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
            Primary fluence grid
        """
        # Create empty fluence grid
        fluence = np.zeros(grid_shape)
        
        # Get beam direction based on gantry angle
        gantry_rad = np.radians(gantry_angle)
        collimator_rad = np.radians(collimator_angle)
        
        # Main beam direction
        beam_dir = np.array([
            np.sin(gantry_rad),
            0,
            -np.cos(gantry_rad)
        ])
        
        # Perpendicular directions (accounting for collimator rotation)
        perp1 = np.array([
            np.cos(gantry_rad) * np.cos(collimator_rad),
            np.sin(collimator_rad),
            np.sin(gantry_rad) * np.cos(collimator_rad)
        ])
        
        perp2 = np.array([
            -np.cos(gantry_rad) * np.sin(collimator_rad),
            np.cos(collimator_rad),
            -np.sin(gantry_rad) * np.sin(collimator_rad)
        ])
        
        # Half field size in cm
        half_field_x = field_size[0] / 2
        half_field_y = field_size[1] / 2
        
        # Source to isocenter distance (typically 100 cm)
        SID = np.linalg.norm(source_position - isocenter)
        
        # Calculate a planar fluence
        nz, ny, nx = grid_shape
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    # Calculate position relative to source
                    pos = np.array([z, y, x]) * grid_spacing
                    rel_pos = pos - source_position
                    dist = np.linalg.norm(rel_pos)
                    
                    if dist > 0:
                        # Normalize direction
                        direction = rel_pos / dist
                        
                        # Project onto beam direction
                        cosine = np.dot(direction, beam_dir)
                        
                        if cosine > 0:  # Forward direction only
                            # Calculate perpendicular distance to beam axis at isocenter plane
                            # This is effectively beam's eye view coordinates
                            depth = np.dot(pos - source_position, beam_dir)
                            scale = depth / SID  # Scale factor for divergence
                            
                            # Project onto perpendicular directions
                            x_bev = np.dot(pos - source_position, perp1) / scale
                            y_bev = np.dot(pos - source_position, perp2) / scale
                            
                            # Check if within field
                            if abs(x_bev) <= half_field_x and abs(y_bev) <= half_field_y:
                                # Apply inverse square law for distance
                                inv_square = (SID / dist) ** 2
                                
                                # Apply off-axis factors from beam profile
                                # This is a simplified version - real implementation would use the beam model
                                off_axis_factor = 1.0
                                if self.beam_model is not None:
                                    # Get profile from beam model if available
                                    profile_name = f"profile_10x10_10cm"  # Simplified - would select based on field size
                                    if self.beam_model.has_parameter(profile_name):
                                        profile_param = self.beam_model.get_parameter(profile_name)
                                        # Use radial distance for simple approximation
                                        radial_dist = np.sqrt(x_bev ** 2 + y_bev ** 2)
                                        off_axis_factor = profile_param.interpolate([radial_dist])
                                
                                # Apply fluence
                                fluence[z, y, x] = 100.0 * inv_square * off_axis_factor * cosine
        
        return fluence
    
    def _calculate_terma(self, 
                         primary_fluence: np.ndarray, 
                         electron_density: np.ndarray,
                         source_position: np.ndarray) -> np.ndarray:
        """
        Calculate TERMA (Total Energy Released per unit MAss) grid.
        
        Parameters
        ----------
        primary_fluence : np.ndarray
            Primary fluence grid
        electron_density : np.ndarray
            Electron density grid
        source_position : np.ndarray
            Source position in world coordinates
            
        Returns
        -------
        np.ndarray
            TERMA grid
        """
        # Initialize TERMA grid
        terma = np.zeros_like(primary_fluence)
        
        # Calculate effective attenuation coefficient based on energy spectrum
        # This is a simplified version - real implementation would use energy spectrum from beam model
        mu_water = 0.05  # Approximate average attenuation coefficient for water (1/cm)
        
        # Grid shape
        nz, ny, nx = primary_fluence.shape
        
        # Calculate TERMA for each voxel
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if primary_fluence[z, y, x] > 0:
                        # Apply attenuation based on electron density
                        # This is a simplified calculation - real implementation would account for beam hardening
                        terma[z, y, x] = primary_fluence[z, y, x] * mu_water * electron_density[z, y, x]
        
        return terma
    
    def _collapsed_cone_convolution(self, 
                                   terma: np.ndarray, 
                                   electron_density: np.ndarray,
                                   grid_spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Apply collapsed cone convolution to calculate dose from TERMA.
        
        Parameters
        ----------
        terma : np.ndarray
            TERMA grid
        electron_density : np.ndarray
            Electron density grid
        grid_spacing : Tuple[float, float, float]
            Spacing of the grid in cm
            
        Returns
        -------
        np.ndarray
            Dose grid
        """
        # Initialize dose grid
        dose = np.zeros_like(terma)
        
        # Grid shape
        nz, ny, nx = terma.shape
        
        # Spacing
        dz, dy, dx = grid_spacing
        
        # Energy deposition kernel parameters (simplified)
        # These parameters would typically come from Monte Carlo simulations or measurements
        # Here we use a simplified exponential kernel
        a_prim = 0.6  # Primary component fraction
        a_scat = 0.4  # Scatter component fraction
        mu_prim = 0.08  # Primary attenuation coefficient (1/cm)
        mu_scat = 0.04  # Scatter attenuation coefficient (1/cm)
        
        # Loop through all voxels
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if terma[z, y, x] > 0:
                        # For each TERMA voxel, distribute energy along each cone
                        for cone_idx in range(self.num_cones):
                            direction = self.cone_directions[cone_idx]
                            
                            # Trace along cone and deposit energy
                            self._trace_cone(
                                dose, terma, electron_density,
                                z, y, x, direction,
                                a_prim, a_scat, mu_prim, mu_scat,
                                dz, dy, dx
                            )
        
        return dose
    
    def _trace_cone(self, 
                   dose: np.ndarray, 
                   terma: np.ndarray, 
                   electron_density: np.ndarray,
                   z0: int, y0: int, x0: int, 
                   direction: np.ndarray,
                   a_prim: float, a_scat: float, 
                   mu_prim: float, mu_scat: float,
                   dz: float, dy: float, dx: float):
        """
        Trace a single cone from a source voxel and deposit dose.
        
        Parameters
        ----------
        dose : np.ndarray
            Dose grid to update
        terma : np.ndarray
            TERMA grid
        electron_density : np.ndarray
            Electron density grid
        z0, y0, x0 : int
            Source voxel indices
        direction : np.ndarray
            Direction of the cone
        a_prim, a_scat : float
            Relative weights of primary and scatter components
        mu_prim, mu_scat : float
            Attenuation coefficients for primary and scatter
        dz, dy, dx : float
            Grid spacing in each dimension
        """
        # Grid shape
        nz, ny, nx = terma.shape
        
        # Normalize spacing to make step size approximately equal to the smallest voxel dimension
        min_spacing = min(dz, dy, dx)
        step_size = min_spacing
        
        # Convert direction to step sizes in grid indices
        dz_step = direction[0] * step_size / dz
        dy_step = direction[1] * step_size / dy
        dx_step = direction[2] * step_size / dx
        
        # Initial terma at source voxel
        terma_source = terma[z0, y0, x0]
        
        # Current position (starting at the center of source voxel)
        z_pos = z0 + 0.5
        y_pos = y0 + 0.5
        x_pos = x0 + 0.5
        
        # Initialize radiological path length
        path_length = 0.0
        
        # Trace the cone with small steps
        max_steps = int(max(nz, ny, nx) * 2)  # Limit number of steps to avoid infinite loops
        
        for step in range(max_steps):
            # Update position
            z_pos += dz_step
            y_pos += dy_step
            x_pos += dx_step
            
            # Convert to integer indices
            z_idx = int(z_pos)
            y_idx = int(y_pos)
            x_idx = int(x_pos)
            
            # Check if we're outside the grid
            if (z_idx < 0 or z_idx >= nz or 
                y_idx < 0 or y_idx >= ny or 
                x_idx < 0 or x_idx >= nx):
                break
            
            # Calculate geometric distance from source
            geo_dist = step_size * (step + 1)
            
            # Update radiological path length
            path_length += step_size * electron_density[z_idx, y_idx, x_idx]
            
            # Calculate solid angle factor (inverse square)
            omega = 1.0 / (geo_dist * geo_dist)
            
            # Calculate energy deposition kernel
            kernel_value = (
                a_prim * np.exp(-mu_prim * path_length) + 
                a_scat * np.exp(-mu_scat * path_length)
            )
            
            # Deposit dose
            if self.calculate_heterogeneity:
                # Scale by local electron density for heterogeneity correction
                dose_contrib = terma_source * kernel_value * omega / electron_density[z_idx, y_idx, x_idx]
            else:
                dose_contrib = terma_source * kernel_value * omega
            
            # Add to dose grid
            dose[z_idx, y_idx, x_idx] += dose_contrib
    
    def _normalize_to_isocenter(self, dose_image: Image, isocenter: np.ndarray):
        """
        Normalize dose so that the isocenter receives 100% dose.
        
        Parameters
        ----------
        dose_image : Image
            Dose image to normalize
        isocenter : np.ndarray
            Isocenter position in world coordinates
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