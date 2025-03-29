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
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor

from quangtps.core.exceptions import DoseCalculationError, ValidationError
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.dose.beam_data_processor import BeamModel, BeamProfileData
from quangtps.dose.algorithms.base import DoseCalculationAlgorithm, DoseCalculationResult
from quangtps.dose.physics.terma import calculate_terma

logger = logging.getLogger(__name__)

class CollapsedConeAlgorithm(DoseCalculationAlgorithm):
    """
    Implementation of the Collapsed Cone Convolution/Superposition algorithm.
    
    This class provides methods for calculating 3D dose distributions
    using the Collapsed Cone Convolution/Superposition algorithm for 
    radiotherapy treatment planning, which accounts for heterogeneities
    in the patient and handles tissue scattering more accurately.
    """
    
    def __init__(self):
        """
        Initialize the Collapsed Cone algorithm.
        """
        super().__init__("Collapsed Cone")
        self.version = "1.1"
        
        # Default parameters
        self.parameters.update({
            'grid_size': 0.3,  # Calculation grid size in cm
            'threads': 8,  # Number of parallel threads
            'num_cones': 32,  # Number of angular cones (higher value = more accurate but slower)
            'max_scatter_radius': 30,  # Maximum radius in cm for scatter consideration
            'use_heterogeneity_correction': True,  # Whether to account for tissue heterogeneity
            'use_adaptive_grid': True,  # Whether to use variable grid spacing
            'use_gpu': False,  # Whether to use GPU acceleration
            'photon_cutoff': 0.01,  # Energy cutoff for photon transport
            'electron_cutoff': 0.001,  # Energy cutoff for electron transport
            'density_threshold': 0.01,  # Density threshold for considering a voxel
            'normalization_depth': 10.0,  # Depth in cm for dose normalization
        })
        
        # Initialize cone directions based on spherical coordinates (theta, phi)
        # with uniform distribution over the unit sphere
        self._initialize_cone_directions()
        
        logger.info(f"Initialized {self.name} algorithm version {self.version}")
        
        self.beam_model = None
        self.kernel_table = None
    
    def _initialize_cone_directions(self):
        """Initialize the collapsed cone directions based on the number of cones."""
        num_cones = self.parameters['num_cones']
        
        # Generate approximately evenly distributed points on a unit sphere
        # using the Fibonacci lattice method
        golden_ratio = (1 + 5**0.5) / 2
        i = np.arange(0, num_cones)
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / num_cones)
        
        # Convert spherical to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        # Store directions as unit vectors
        self.cone_directions = np.column_stack((x, y, z))
        
        # Calculate solid angles for each cone (approximate)
        self.cone_solid_angles = np.full(num_cones, 4 * np.pi / num_cones)
        
        logger.debug(f"Initialized {num_cones} cone directions for collapsed cone algorithm")
    
    def set_beam_model(self, beam_model: BeamModel):
        """
        Set the beam model for dose calculation.
        
        Parameters
        ----------
        beam_model : BeamModel
            The beam model containing beam data for dose calculation
        """
        self.beam_model = beam_model
        
        # Prepare the kernel data from the beam model for efficiency
        self._prepare_kernel_table()
        
        logger.info(f"Set beam model: {beam_model.name}")
    
    def _prepare_kernel_table(self):
        """
        Prepare the convolution kernel table based on the beam model.
        This precomputes the scatter kernel for different distances and angles.
        """
        if not self.beam_model:
            logger.warning("Cannot prepare kernel table: No beam model set")
            return
            
        try:
            # Create a radial distance grid (in cm)
            max_radius = self.parameters['max_scatter_radius']
            num_radial_bins = 100
            radial_distances = np.linspace(0.1, max_radius, num_radial_bins)
            
            # Create an angular grid (in radians)
            num_angular_bins = 36
            angles = np.linspace(0, np.pi, num_angular_bins)
            
            # Create a 2D kernel table for [radius, angle]
            kernel_table = np.zeros((num_radial_bins, num_angular_bins))
            
            # Fill the kernel table with appropriate scatter values
            # This is a simplified model - in real implementation, 
            # this would use measured or Monte Carlo simulated data
            for i, r in enumerate(radial_distances):
                for j, angle in enumerate(angles):
                    # Simplified function - a combination of primary and scatter components
                    # Primary component (decreases with inverse square of distance)
                    primary = np.exp(-0.097 * r) / (r**2)
                    
                    # Scatter component (broader, less directional)
                    scatter = np.exp(-0.097 * r * (1 - 0.5 * np.cos(angle))) / (r**2)
                    
                    # Combine with angle-dependent weighting
                    kernel_table[i, j] = primary * (0.75 + 0.25 * np.cos(angle)) + scatter * 0.5
            
            # Normalize the kernel
            kernel_table /= np.sum(kernel_table)
            
            self.kernel_table = {
                'radial_distances': radial_distances,
                'angles': angles,
                'values': kernel_table
            }
            
            logger.info(f"Prepared kernel table with {num_radial_bins}x{num_angular_bins} bins")
            
        except Exception as e:
            logger.error(f"Error preparing kernel table: {str(e)}")
            self.kernel_table = None
    
    def _get_kernel_value(self, distance: float, angle: float) -> float:
        """
        Get the kernel value for a given distance and angle through interpolation.
        
        Parameters
        ----------
        distance : float
            Distance in cm
        angle : float
            Angle in radians
            
        Returns
        -------
        float
            Interpolated kernel value
        """
        if self.kernel_table is None:
            # Fallback when the kernel table isn't available
            return np.exp(-0.097 * distance) / (distance**2 + 0.1)
        
        # Ensure angle is within [0, pi]
        angle = min(max(angle, 0), np.pi)
        
        # Get the kernel table data
        radial_distances = self.kernel_table['radial_distances']
        angles = self.kernel_table['angles']
        values = self.kernel_table['values']
        
        # Find indices for interpolation
        if distance <= radial_distances[0]:
            r_idx1, r_idx2 = 0, 0
            r_weight = 1.0
        elif distance >= radial_distances[-1]:
            r_idx1, r_idx2 = len(radial_distances)-1, len(radial_distances)-1
            r_weight = 1.0
        else:
            for i in range(len(radial_distances)-1):
                if radial_distances[i] <= distance <= radial_distances[i+1]:
                    r_idx1, r_idx2 = i, i+1
                    r_weight = (distance - radial_distances[i]) / (radial_distances[i+1] - radial_distances[i])
                    break
            else:
                # Should not happen, but just in case
                r_idx1, r_idx2 = 0, 0
                r_weight = 1.0
        
        # Same for angle
        if angle <= angles[0]:
            a_idx1, a_idx2 = 0, 0
            a_weight = 1.0
        elif angle >= angles[-1]:
            a_idx1, a_idx2 = len(angles)-1, len(angles)-1
            a_weight = 1.0
        else:
            for i in range(len(angles)-1):
                if angles[i] <= angle <= angles[i+1]:
                    a_idx1, a_idx2 = i, i+1
                    a_weight = (angle - angles[i]) / (angles[i+1] - angles[i])
                    break
            else:
                a_idx1, a_idx2 = 0, 0
                a_weight = 1.0
        
        # Bilinear interpolation
        v1 = values[r_idx1, a_idx1] * (1-r_weight) + values[r_idx2, a_idx1] * r_weight
        v2 = values[r_idx1, a_idx2] * (1-r_weight) + values[r_idx2, a_idx2] * r_weight
        
        return v1 * (1-a_weight) + v2 * a_weight
    
    def _calculate_dose_for_cone(self, terma: np.ndarray, density: np.ndarray, 
                               direction: np.ndarray, solid_angle: float,
                               voxel_size: Tuple[float, float, float]) -> np.ndarray:
        """
        Calculate dose contribution for a single cone direction.
        
        Parameters
        ----------
        terma : np.ndarray
            TERMA (Total Energy Released per unit MAss) array
        density : np.ndarray
            Electron density relative to water
        direction : np.ndarray
            Direction vector of the cone
        solid_angle : float
            Solid angle of the cone
        voxel_size : Tuple[float, float, float]
            Size of voxels in cm
            
        Returns
        -------
        np.ndarray
            Dose contribution for this cone
        """
        # Create a dose array the same size as the terma array
        dose = np.zeros_like(terma)
        
        # Get array dimensions
        nx, ny, nz = terma.shape
        
        # Get voxel sizes in cm
        dx, dy, dz = voxel_size
        
        # Normalize direction vector
        direction = direction / np.linalg.norm(direction)
        
        # Use heterogeneity correction if enabled
        use_heterogeneity = self.parameters['use_heterogeneity_correction']
        
        # Determine step sizes along the ray path based on direction
        # This ensures we step across voxel boundaries appropriately
        step_sizes = np.abs(np.array([dx, dy, dz]) / (np.abs(direction) + 1e-10))
        step_size = 0.5 * np.min(step_sizes)  # Half of smallest step needed to cross a voxel
        
        # Process each ray (from each voxel) along the cone direction
        # This is very computationally intensive, so we'll use a simplified approach
        
        # Process the array in the most efficient direction
        # based on which component of the direction vector is largest
        # This ensures we trace the rays efficiently
        max_dim = np.argmax(np.abs(direction))
        
        if max_dim == 0:  # X direction is dominant
            if direction[0] > 0:
                range_i = range(nx)
            else:
                range_i = range(nx-1, -1, -1)
                
            for i in range_i:
                for j in range(ny):
                    for k in range(nz):
                        if terma[i, j, k] > 0 and density[i, j, k] > self.parameters['density_threshold']:
                            self._trace_ray(terma, density, dose, (i, j, k), direction, solid_angle, 
                                          step_size, use_heterogeneity, voxel_size)
        
        elif max_dim == 1:  # Y direction is dominant
            if direction[1] > 0:
                range_j = range(ny)
            else:
                range_j = range(ny-1, -1, -1)
                
            for j in range_j:
                for i in range(nx):
                    for k in range(nz):
                        if terma[i, j, k] > 0 and density[i, j, k] > self.parameters['density_threshold']:
                            self._trace_ray(terma, density, dose, (i, j, k), direction, solid_angle, 
                                          step_size, use_heterogeneity, voxel_size)
        
        else:  # Z direction is dominant
            if direction[2] > 0:
                range_k = range(nz)
            else:
                range_k = range(nz-1, -1, -1)
                
            for k in range_k:
                for i in range(nx):
                    for j in range(ny):
                        if terma[i, j, k] > 0 and density[i, j, k] > self.parameters['density_threshold']:
                            self._trace_ray(terma, density, dose, (i, j, k), direction, solid_angle, 
                                          step_size, use_heterogeneity, voxel_size)
        
        return dose
    
    def _trace_ray(self, terma: np.ndarray, density: np.ndarray, dose: np.ndarray, 
                 start_idx: Tuple[int, int, int], direction: np.ndarray, solid_angle: float,
                 step_size: float, use_heterogeneity: bool, voxel_size: Tuple[float, float, float]):
        """
        Trace a ray from a source voxel along a given direction and accumulate dose.
        
        Parameters
        ----------
        terma : np.ndarray
            TERMA array
        density : np.ndarray
            Electron density relative to water
        dose : np.ndarray
            Dose array to accumulate into
        start_idx : Tuple[int, int, int]
            Starting voxel indices (i, j, k)
        direction : np.ndarray
            Direction vector
        solid_angle : float
            Solid angle of the cone
        step_size : float
            Step size in cm
        use_heterogeneity : bool
            Whether to use heterogeneity correction
        voxel_size : Tuple[float, float, float]
            Size of voxels in cm
        """
        # Get array dimensions
        nx, ny, nz = terma.shape
        
        # Get voxel sizes
        dx, dy, dz = voxel_size
        
        # Current position (in voxel indices, as floats)
        i, j, k = start_idx
        
        # Get the source terma
        source_terma = terma[i, j, k]
        
        # Skip if terma is negligible
        if source_terma < 1e-10:
            return
        
        # Maximum distance to trace in cm
        max_distance = self.parameters['max_scatter_radius']
        
        # Current distance traveled
        distance = 0.0
        
        # Reference density at source voxel
        ref_density = density[i, j, k]
        
        # Initialize variables for the ray tracing loop
        current_position = np.array([i * dx, j * dy, k * dz])
        
        # Track accumulated attenuation along the ray
        accumulated_attenuation = 0.0
        
        # Continue tracing until max distance is reached or we exit the volume
        while distance < max_distance:
            # Update position
            current_position += direction * step_size
            distance += step_size
            
            # Convert to voxel indices
            ii = int(current_position[0] / dx + 0.5)
            jj = int(current_position[1] / dy + 0.5)
            kk = int(current_position[2] / dz + 0.5)
            
            # Check if we're still in the volume
            if not (0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz):
                break
            
            # Get density at current position
            current_density = density[ii, jj, kk]
            
            # Skip if density is too low (air or outside patient)
            if current_density < self.parameters['density_threshold']:
                continue
            
            # Calculate angle between source and current position
            dx_vec = np.array([(ii - i) * dx, (jj - j) * dy, (kk - k) * dz])
            cos_angle = np.dot(dx_vec, direction) / (np.linalg.norm(dx_vec) + 1e-10)
            angle = np.arccos(min(max(cos_angle, -1.0), 1.0))
            
            # Get kernel value for this distance and angle
            kernel_value = self._get_kernel_value(distance, angle)
            
            # Update attenuation based on ray path
            if use_heterogeneity:
                # Simple exponential attenuation based on density
                # This should be enhanced with more accurate radiobiological models
                current_attenuation = np.exp(-0.03 * current_density * step_size)
                accumulated_attenuation += np.log(current_attenuation)
                attenuation = np.exp(accumulated_attenuation)
            else:
                # Simplified water-equivalent attenuation
                attenuation = np.exp(-0.03 * distance)
            
            # Calculate dose contribution to this voxel
            dose_contribution = source_terma * kernel_value * attenuation * solid_angle
            
            # Adjust for density (dose = terma * energy deposition kernel)
            dose_contribution *= current_density / ref_density
            
            # Add to dose array (atomic for thread safety)
            np.add.at(dose, (ii, jj, kk), dose_contribution)
    
    def set_calculation_parameters(self, grid_size: float = 0.3, threads: int = 8, 
                                 num_cones: int = 32, max_scatter_radius: float = 30):
        """
        Set calculation parameters.
        
        Parameters
        ----------
        grid_size : float
            Calculation grid size in cm
        threads : int
            Number of parallel threads for calculation
        num_cones : int
            Number of angular cones
        max_scatter_radius : float
            Maximum radius in cm for scatter consideration
        """
        self.parameters['grid_size'] = grid_size
        self.parameters['threads'] = threads
        self.parameters['num_cones'] = num_cones
        self.parameters['max_scatter_radius'] = max_scatter_radius
        
        # Update cone directions if num_cones changed
        if num_cones != len(self.cone_directions):
            self._initialize_cone_directions()
        
        logger.info(f"Set calculation parameters: grid_size={grid_size}cm, threads={threads}, "
                   f"num_cones={num_cones}, max_scatter_radius={max_scatter_radius}cm")
    
    def set_heterogeneity_correction(self, enabled: bool):
        """
        Enable or disable heterogeneity correction.
        
        Parameters
        ----------
        enabled : bool
            Flag to enable or disable heterogeneity correction
        """
        self.parameters['use_heterogeneity_correction'] = enabled
        status = "enabled" if enabled else "disabled"
        logger.info(f"Heterogeneity correction {status}")
        
    def calculate(self, ct_image: Image, beam: Beam) -> DoseCalculationResult:
        """
        Calculate dose distribution using Collapsed Cone algorithm.
        
        Parameters
        ----------
        ct_image : Image
            CT image for dose calculation
        beam : Beam
            Treatment beam
            
        Returns
        -------
        DoseCalculationResult
            Calculated dose and metadata
            
        Raises
        ------
        DoseCalculationError
            If dose calculation fails
        ValidationError
            If inputs are invalid
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self.validate_inputs(ct_image, beam)
            
            # Get calculation parameters
            use_heterogeneity = self.parameters['use_heterogeneity_correction']
            grid_size = self.parameters['grid_size']
            threads = self.parameters['threads']
            num_cones = self.parameters['num_cones']
            
            logger.info(f"Starting Collapsed Cone dose calculation for beam {beam.name}")
            logger.info(f"Parameters: grid_size={grid_size}cm, threads={threads}, "
                       f"heterogeneity={use_heterogeneity}, cones={num_cones}")
            
            # Convert CT to electron density
            electron_density = self._convert_ct_to_density(ct_image)
            
            # Calculate TERMA (Total Energy Released per unit MAss)
            terma = calculate_terma(ct_image, beam, self.beam_model)
            
            # Initialize dose grid
            dose_data = np.zeros_like(ct_image.data)
            
            # Get voxel sizes in cm
            voxel_size = tuple(x / 10.0 for x in ct_image.spacing)  # mm to cm
            
            # Parallel processing of cones
            if threads > 1:
                with ThreadPoolExecutor(max_workers=threads) as executor:
                    futures = []
                    
                    for i, direction in enumerate(self.cone_directions):
                        solid_angle = self.cone_solid_angles[i]
                        futures.append(
                            executor.submit(
                                self._calculate_dose_for_cone,
                                terma, electron_density, direction, solid_angle, voxel_size
                            )
                        )
                    
                    # Collect results
                    for future in futures:
                        dose_contribution = future.result()
                        dose_data += dose_contribution
            else:
                # Sequential processing
                for i, direction in enumerate(self.cone_directions):
                    solid_angle = self.cone_solid_angles[i]
                    dose_contribution = self._calculate_dose_for_cone(
                        terma, electron_density, direction, solid_angle, voxel_size
                    )
                    dose_data += dose_contribution
            
            # Normalize dose
            if np.max(dose_data) > 0:
                # Scale based on the reference normalization depth
                norm_depth_idx = int(self.parameters['normalization_depth'] / voxel_size[2])
                if 0 <= norm_depth_idx < ct_image.data.shape[2]:
                    # Find maximum dose at normalization depth
                    norm_slice = dose_data[:, :, norm_depth_idx]
                    if np.max(norm_slice) > 0:
                        dose_data = dose_data / np.max(norm_slice) * 100.0  # Normalize to 100% at reference depth
                else:
                    # Fallback normalization to global maximum
                    dose_data = dose_data / np.max(dose_data) * 100.0
            
            calculation_time = time.time() - start_time
            logger.info(f"Collapsed Cone dose calculation completed in {calculation_time:.2f} seconds")
            
            # Create and return result
            result = DoseCalculationResult(
                dose_data=dose_data,
                calculation_time=calculation_time,
                algorithm=self.name,
                version=self.version,
                parameters=self.parameters.copy(),
                beam_info={
                    'name': beam.name,
                    'energy': getattr(beam, 'energy', 'Unknown'),
                    'gantry_angle': getattr(beam, 'gantry_angle', 0),
                    'collimator_angle': getattr(beam, 'collimator_angle', 0),
                    'field_size': getattr(beam, 'field_size', (10, 10)),
                }
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error in Collapsed Cone dose calculation: {str(e)}"
            logger.error(error_msg)
            raise DoseCalculationError(error_msg) from e
    
    def _convert_ct_to_density(self, ct_image: Image) -> np.ndarray:
        """
        Convert CT image data to electron density relative to water.
        
        Parameters
        ----------
        ct_image : Image
            CT image
            
        Returns
        -------
        np.ndarray
            Electron density array
        """
        # This is a simplified conversion - real implementation would use a calibration curve
        # based on the specific CT scanner and protocol
        
        # CT values (Hounsfield Units) typically range from -1000 (air) to +1000 or more (bone)
        # We convert to relative electron density where water = 1.0
        
        # Simple linear mapping
        ct_data = ct_image.data.copy()
        
        # Air: -1000 HU -> 0.001 relative electron density
        # Water: 0 HU -> 1.0 relative electron density
        # Bone: +1000 HU -> 1.8 relative electron density
        electron_density = 1.0 + ct_data / 1000.0
        
        # Set minimum density threshold to avoid division by zero
        electron_density = np.maximum(electron_density, 0.001)
        
        return electron_density
    
    def calculate_beam_dose(self, beam: Beam, ct_image: Image) -> Image:
        """
        Calculate dose for a beam.
        
        Parameters
        ----------
        beam : Beam
            Treatment beam
        ct_image : Image
            CT image
            
        Returns
        -------
        Image
            Dose image
        """
        result = self.calculate(ct_image, beam)
        
        # Create a new image with the dose data
        dose_image = Image(
            data=result.dose_data,
            spacing=ct_image.spacing,
            origin=ct_image.origin,
            direction=ct_image.direction,
            modality="RTDOSE"
        )
        
        # Set additional metadata
        dose_image.metadata = {
            'algorithm': self.name,
            'version': self.version,
            'calculation_time': result.calculation_time,
            'beam_name': beam.name,
            'parameters': self.parameters
        }
        
        return dose_image
    
    def get_description(self) -> str:
        """
        Get a description of the algorithm.
        
        Returns
        -------
        str
            Algorithm description
        """
        return (f"{self.name} v{self.version} - A convolution/superposition algorithm "
                f"that models dose deposition using {self.parameters['num_cones']} directional cones "
                f"to approximate the radiation transport through the patient.")
    
    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Get information about the algorithm parameters.
        
        Returns
        -------
        Dict[str, Any]
            Parameter information
        """
        return {
            'grid_size': {
                'description': 'Calculation grid size in cm',
                'default': 0.3,
                'type': 'float',
                'range': [0.1, 1.0]
            },
            'threads': {
                'description': 'Number of parallel threads',
                'default': 8,
                'type': 'int',
                'range': [1, 64]
            },
            'num_cones': {
                'description': 'Number of angular cones',
                'default': 32,
                'type': 'int',
                'range': [8, 512]
            },
            'max_scatter_radius': {
                'description': 'Maximum radius in cm for scatter consideration',
                'default': 30,
                'type': 'float',
                'range': [5, 50]
            },
            'use_heterogeneity_correction': {
                'description': 'Whether to account for tissue heterogeneity',
                'default': True,
                'type': 'bool'
            },
            'use_adaptive_grid': {
                'description': 'Whether to use variable grid spacing',
                'default': True,
                'type': 'bool'
            },
            'photon_cutoff': {
                'description': 'Energy cutoff for photon transport',
                'default': 0.01,
                'type': 'float',
                'range': [0.001, 0.1]
            },
            'electron_cutoff': {
                'description': 'Energy cutoff for electron transport',
                'default': 0.001,
                'type': 'float',
                'range': [0.0001, 0.01]
            }
        } 