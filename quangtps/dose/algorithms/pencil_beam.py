#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of the Pencil Beam dose calculation algorithm.

This module provides a class for calculating dose distributions using
the Pencil Beam algorithm for radiotherapy treatment planning.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any

from quangtps.core.exceptions import DoseCalculationError, ValidationError
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.dose.beam_data_processor import BeamModel, BeamModelParameter
from quangtps.dose.algorithms.base import DoseCalculationAlgorithm, DoseCalculationResult
from quangtps.dose.physics.terma import calculate_terma_from_beam

logger = logging.getLogger(__name__)

class PencilBeamAlgorithm(DoseCalculationAlgorithm):
    """
    Implementation of the Pencil Beam dose calculation algorithm.
    
    This class provides methods for calculating 3D dose distributions
    using the Pencil Beam algorithm for radiotherapy treatment planning.
    """
    
    def __init__(self):
        """
        Initialize the Pencil Beam algorithm.
        """
        super().__init__("Pencil Beam")
        self.version = "1.0"
        
        # Default parameters
        self.parameters.update({
            'grid_size': 0.2,  # Calculation grid size in cm
            'threads': 4,  # Number of parallel threads
            'tissue_air_ratio_correction': True,  # Whether to apply TAR correction
            'use_gpu': False,  # Whether to use GPU acceleration
            'pencil_spacing': 0.1,  # Spacing between pencil beams in cm
            'integration_step': 0.5,  # Integration step size in cm for ray tracing
        })
        
        logger.info(f"Initialized {self.name} algorithm version {self.version}")
        
        self.beam_model = None
    
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
        self.get_parameter('heterogeneity_correction')
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
        self.get_parameter('grid_size')
        self.get_parameter('threads')
        logger.info(f"Set calculation parameters: grid_size={grid_size}cm, threads={threads}")
    
    def calculate(self, ct_image: Image, beam: Beam) -> DoseCalculationResult:
        """
        Calculate dose distribution using Pencil Beam algorithm.
        
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
            tissue_air_ratio_correction = self.get_parameter('tissue_air_ratio_correction')
            grid_size = self.get_parameter('grid_size')
            threads = self.get_parameter('threads')
            pencil_spacing = self.get_parameter('pencil_spacing')
            integration_step = self.get_parameter('integration_step')
            
            logger.info(f"Starting Pencil Beam dose calculation for beam {beam.name}")
            logger.info(f"Parameters: grid_size={grid_size}cm, threads={threads}, TAR={tissue_air_ratio_correction}")
            
            # Convert CT to electron density
            electron_density = self._convert_ct_to_density(ct_image)
            
            # Initialize dose grid
            dose_data = np.zeros_like(ct_image.data)
            
            # Get beam parameters
            field_size = beam.field_size  # in cm
            sad = beam.sad if hasattr(beam, 'sad') else 1000.0  # mm -> convert to cm later
            isocenter = np.array(beam.isocenter) / 10.0  # mm -> cm
            beam_direction = np.array(beam.get_direction())
            
            # Extract beam MLC configuration if available
            mlc_config = None
            if hasattr(beam, 'mlc') and beam.mlc is not None:
                mlc_config = beam.mlc.get_leaf_positions()
            
            # Calculate source position
            source_position = isocenter - beam_direction * (sad / 10.0)  # cm
            
            # Determine calculation grid
            nx, ny, nz = ct_image.data.shape
            dx, dy, dz = np.array(ct_image.spacing) / 10.0  # mm -> cm
            
            # Calculate the grid of pencil beam entry points
            # This is a simplified approach - in a real implementation, 
            # we would account for beam divergence and MLC configuration
            
            # Project the field onto the patient surface
            if mlc_config is not None:
                # Handle MLC-defined field
                pencil_beams = self._generate_mlc_pencil_beams(
                    mlc_config, 
                    pencil_spacing, 
                    source_position, 
                    isocenter, 
                    beam_direction
                )
            else:
                # Handle rectangular field
                pencil_beams = self._generate_rectangular_pencil_beams(
                    field_size, 
                    pencil_spacing, 
                    source_position, 
                    isocenter, 
                    beam_direction
                )
            
            logger.info(f"Generated {len(pencil_beams)} pencil beams")
            
            # For each pencil beam, trace through the patient and calculate dose
            for pb_index, pencil_beam in enumerate(pencil_beams):
                if pb_index % 100 == 0:
                    logger.debug(f"Processing pencil beam {pb_index}/{len(pencil_beams)}")
                    
                # Get pencil beam entry point and direction
                entry_point = pencil_beam['entry_point']
                direction = pencil_beam['direction']
                weight = pencil_beam['weight']
                
                # Trace ray through patient
                self._trace_pencil_beam(
                    entry_point, 
                    direction, 
                    weight,
                    electron_density, 
                    dose_data, 
                    ct_image,
                    integration_step, 
                    tissue_air_ratio_correction
                )
            
            # Validate results
            self._validate_calculation_completed(dose_data)
            
            # Create result object
            calculation_time = time.time() - start_time
            logger.info(f"Pencil Beam calculation completed in {calculation_time:.2f} seconds")
            
            dose_image = Image(
                data=dose_data,
                spacing=ct_image.spacing,
                origin=ct_image.origin,
                direction=ct_image.direction,
                modality="RTDOSE"
            )
            
            # Set dose description and additional metadata
            dose_image.description = f"Dose calculated with {self.name} algorithm"
            
            result = DoseCalculationResult(
                dose=dose_image,
                algorithm_name=self.name,
                calculation_time=calculation_time,
                additional_data={
                    'beam_name': beam.name,
                    'parameters': self.get_parameters()
                }
            )
            
            return result
            
        except ValidationError as e:
            logger.error(f"Validation error in {self.name} calculation: {str(e)}")
            raise
        
        except Exception as e:
            logger.error(f"Error in {self.name} calculation: {str(e)}")
            raise DoseCalculationError(f"{self.name} calculation failed: {str(e)}")
    
    def _generate_rectangular_pencil_beams(
        self, 
        field_size: List[float], 
        pencil_spacing: float, 
        source_position: np.ndarray, 
        isocenter: np.ndarray, 
        beam_direction: np.ndarray
    ) -> List[Dict]:
        """
        Generate pencil beams for a rectangular field.
        
        Parameters
        ----------
        field_size : List[float]
            Field size in cm at isocenter
        pencil_spacing : float
            Spacing between pencil beams in cm
        source_position : np.ndarray
            Source position in cm
        isocenter : np.ndarray
            Isocenter position in cm
        beam_direction : np.ndarray
            Beam direction vector
            
        Returns
        -------
        List[Dict]
            List of pencil beam definitions
        """
        pencil_beams = []
        
        # Calculate the number of pencil beams
        nx = int(field_size[0] / pencil_spacing) + 1
        ny = int(field_size[1] / pencil_spacing) + 1
        
        # Ensure odd number for symmetry
        if nx % 2 == 0:
            nx += 1
        if ny % 2 == 0:
            ny += 1
        
        # Calculate the half field size
        half_width = field_size[0] / 2
        half_height = field_size[1] / 2
        
        # Define orthogonal axes to the beam direction
        # This creates a coordinate system with the beam direction as one axis
        v1 = beam_direction
        
        # Find a vector perpendicular to v1
        if abs(v1[0]) < abs(v1[1]):
            v2 = np.array([1, 0, 0])
        else:
            v2 = np.array([0, 1, 0])
            
        # Make v2 orthogonal to v1
        v2 = v2 - np.dot(v2, v1) * v1
        v2 = v2 / np.linalg.norm(v2)
        
        # Create third orthogonal vector
        v3 = np.cross(v1, v2)
        
        # Spacing between pencil beams
        x_spacing = 2 * half_width / (nx - 1)
        y_spacing = 2 * half_height / (ny - 1)
        
        # Generate pencil beams
        for i in range(nx):
            for j in range(ny):
                # Calculate position at isocenter plane
                x = -half_width + i * x_spacing
                y = -half_height + j * y_spacing
                
                # Calculate position in 3D space
                position = isocenter + x * v2 + y * v3
                
                # Calculate direction from source to this position
                direction = position - source_position
                direction = direction / np.linalg.norm(direction)
                
                # For rectangular fields, we could apply a flat fluence profile
                # or model the penumbra with a slightly reduced weight at the edges
                weight = 1.0
                
                # Add to list
                pencil_beams.append({
                    'entry_point': position,
                    'direction': direction,
                    'weight': weight
                })
        
        return pencil_beams
    
    def _generate_mlc_pencil_beams(
        self, 
        mlc_config: Dict, 
        pencil_spacing: float, 
        source_position: np.ndarray, 
        isocenter: np.ndarray, 
        beam_direction: np.ndarray
    ) -> List[Dict]:
        """
        Generate pencil beams for an MLC-defined field.
        
        Parameters
        ----------
        mlc_config : Dict
            MLC leaf positions
        pencil_spacing : float
            Spacing between pencil beams in cm
        source_position : np.ndarray
            Source position in cm
        isocenter : np.ndarray
            Isocenter position in cm
        beam_direction : np.ndarray
            Beam direction vector
            
        Returns
        -------
        List[Dict]
            List of pencil beam definitions
        """
        pencil_beams = []
        
        # Extract MLC leaf positions
        leaf_positions = mlc_config
        
        # Define orthogonal axes to the beam direction
        # This creates a coordinate system with the beam direction as one axis
        v1 = beam_direction
        
        # Find a vector perpendicular to v1
        if abs(v1[0]) < abs(v1[1]):
            v2 = np.array([1, 0, 0])
        else:
            v2 = np.array([0, 1, 0])
            
        # Make v2 orthogonal to v1
        v2 = v2 - np.dot(v2, v1) * v1
        v2 = v2 / np.linalg.norm(v2)
        
        # Create third orthogonal vector
        v3 = np.cross(v1, v2)
        
        # Estimate field extent from MLC positions
        min_leaf_pos = min([min(leaf['left'], leaf['right']) for leaf in leaf_positions])
        max_leaf_pos = max([max(leaf['left'], leaf['right']) for leaf in leaf_positions])
        leaf_width = leaf_positions[0].get('width', 1.0)  # in cm at isocenter
        
        # Y positions from leaf centers
        y_positions = [leaf['center'] for leaf in leaf_positions]
        min_y = min(y_positions) - leaf_width/2
        max_y = max(y_positions) + leaf_width/2
        
        # Calculate number of pencil beams
        nx = int((max_leaf_pos - min_leaf_pos) / pencil_spacing) + 1
        ny = int((max_y - min_y) / pencil_spacing) + 1
        
        # Generate grid of potential pencil beam positions
        for i in range(nx):
            x = min_leaf_pos + i * pencil_spacing
            
            for j in range(ny):
                y = min_y + j * pencil_spacing
                
                # Check if this position is inside the MLC aperture
                inside_aperture = False
                
                # Find which leaf this y position corresponds to
                for leaf in leaf_positions:
                    leaf_center = leaf['center']
                    half_width = leaf['width'] / 2
                    
                    if (leaf_center - half_width) <= y <= (leaf_center + half_width):
                        # Inside this leaf's extent
                        if leaf['left'] <= x <= leaf['right']:
                            inside_aperture = True
                            break
                
                if inside_aperture:
                    # Calculate position in 3D space
                    position = isocenter + x * v2 + y * v3
                    
                    # Calculate direction from source to this position
                    direction = position - source_position
                    direction = direction / np.linalg.norm(direction)
                    
                    # For MLC fields, we could model leaf transmission
                    # with reduced weights for positions that are partially blocked
                    weight = 1.0
                    
                    # Add to list
                    pencil_beams.append({
                        'entry_point': position,
                        'direction': direction,
                        'weight': weight
                    })
        
        return pencil_beams
    
    def _trace_pencil_beam(
        self, 
        entry_point: np.ndarray, 
        direction: np.ndarray, 
        weight: float,
        density_grid: np.ndarray, 
        dose_grid: np.ndarray, 
        ct_image: Image,
        step_size: float, 
        apply_tar_correction: bool
    ):
        """
        Trace a pencil beam through the patient and deposit dose.
        
        Parameters
        ----------
        entry_point : np.ndarray
            Entry point of the pencil beam in cm
        direction : np.ndarray
            Direction vector of the pencil beam
        weight : float
            Weight of the pencil beam
        density_grid : np.ndarray
            Electron density grid
        dose_grid : np.ndarray
            Dose grid to update
        ct_image : Image
            CT image
        step_size : float
            Integration step size in cm
        apply_tar_correction : bool
            Whether to apply tissue-air ratio correction
        """
        # Convert entry point from cm to mm for world_to_voxel method
        entry_point_mm = entry_point * 10.0
        
        # Get grid dimensions
        nx, ny, nz = ct_image.data.shape
        
        # Get spacing in cm
        dx, dy, dz = np.array(ct_image.spacing) / 10.0  # mm -> cm
        
        # Convert entry point to voxel indices
        entry_indices = ct_image.world_to_voxel(entry_point_mm)
        current_pos = np.array([entry_indices[0], entry_indices[1], entry_indices[2]])
        
        # Initialize depth and accumulated density
        depth = 0.0
        rad_depth = 0.0  # Radiological depth
        
        # Get generic kernel data from beam model
        # In a real implementation, we would have these in the beam model
        # For this example, we'll use simple exponential falloff kernels
        
        # Loop until we exit the patient or reach a maximum depth
        max_depth = 50.0  # cm
        max_steps = int(max_depth / step_size)
        
        for step in range(max_steps):
            # Calculate current indices
            i, j, k = int(current_pos[0]), int(current_pos[1]), int(current_pos[2])
            
            # Check if we're inside the grid
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                # Get density at current position
                density = density_grid[i, j, k]
                
                # Update radiological depth
                rad_depth_step = density * step_size
                rad_depth += rad_depth_step
                
                # Calculate dose at current depth
                # In a real implementation, we would use PDD or TPR data from the beam model
                # For this example, we'll use a simple exponential falloff
                dose = weight * np.exp(-0.05 * rad_depth)
                
                # Apply lateral dose spread
                # In a real implementation, we would use appropriate kernel data
                # For this example, we'll apply a simple Gaussian kernel
                
                # Define kernel size based on depth
                kernel_radius = int(depth * 0.05 / dx) + 1  # Larger kernel at greater depths
                
                # Deposit dose with lateral spread
                for ki in range(max(0, i - kernel_radius), min(nx, i + kernel_radius + 1)):
                    for kj in range(max(0, j - kernel_radius), min(ny, j + kernel_radius + 1)):
                        for kk in range(max(0, k - kernel_radius), min(nz, k + kernel_radius + 1)):
                            # Calculate distance from central axis
                            di = (ki - i) * dx
                            dj = (kj - j) * dy
                            dk = (kk - k) * dz
                            r = np.sqrt(di*di + dj*dj + dk*dk)
                            
                            # Gaussian kernel
                            sigma = 0.3 * (1 + 0.1 * depth)  # Wider kernel at greater depths
                            kernel_value = np.exp(-r*r / (2 * sigma*sigma))
                            
                            # Deposit dose
                            dose_grid[ki, kj, kk] += dose * kernel_value
                
                # Update depth
                depth += step_size
                
                # Update position
                current_pos += direction * (step_size / dx)  # Scale by voxel size
            else:
                # We've left the patient
                break
    
    def calculate_beam_dose(self, beam: Beam, ct_image: Image) -> Image:
        """
        Calculate dose for a beam using Pencil Beam algorithm.
        
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
        return result.dose
    
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
            pdd_values = np.array([0, 97.0, 100.0, 97.0, 93.0, 89.0, 85.0, 65.0, 45.0, 30.0, 21.0, 15.0])
        elif energy == "10MV":
            pdd_values = np.array([0, 90.0, 100.0, 99.0, 95.0, 92.0, 88.0, 70.0, 52.0, 38.0, 27.0, 20.0])
        else:  # Default to 15MV
            pdd_values = np.array([0, 85.0, 100.0, 99.5, 97.0, 94.0, 91.0, 75.0, 57.0, 43.0, 32.0, 24.0])
        
        pdd_parameter = BeamModelParameter(
            name="pdd_10x10",
            value_grid=pdd_values,
            dimensions=["depth"],
            units=["cm"],
            dimension_values=[depths],
            interpolation_method="cubic"
        )
        model.add_parameter(pdd_parameter)
        
        # Create profiles for different depths and field sizes
        # This is a simplified example - real data would be more comprehensive
        
        # Profile data for 10x10 field at different depths
        x_positions = np.linspace(-10, 10, 21)  # -10 to 10 cm in 1 cm steps
        
        # Different profiles based on energy and depth
        # Depth: dmax
        if energy == "6MV":
            profile_dmax = np.ones_like(x_positions)
            # Add penumbra
            profile_dmax[0] = 0.2
            profile_dmax[1] = 0.6
            profile_dmax[-2] = 0.6
            profile_dmax[-1] = 0.2
        elif energy == "10MV":
            profile_dmax = np.ones_like(x_positions)
            # Sharper penumbra for higher energies
            profile_dmax[0] = 0.1
            profile_dmax[1] = 0.5
            profile_dmax[-2] = 0.5
            profile_dmax[-1] = 0.1
        else:  # 15MV
            profile_dmax = np.ones_like(x_positions)
            profile_dmax[0] = 0.05
            profile_dmax[1] = 0.4
            profile_dmax[-2] = 0.4
            profile_dmax[-1] = 0.05
        
        profile_parameter = BeamModelParameter(
            name="profile_10x10_dmax",
            value_grid=profile_dmax,
            dimensions=["x"],
            units=["cm"],
            dimension_values=[x_positions],
            interpolation_method="cubic"
        )
        model.add_parameter(profile_parameter)
        
        # Add more profiles for different depths
        # This is a simplified example
        
        return model
    
    def get_description(self) -> str:
        """Get algorithm description."""
        return (
            "Pencil Beam algorithm for dose calculation. "
            "This algorithm models the beam as a collection of narrow pencil beams "
            "and calculates the dose distribution by summing the contributions "
            "from each pencil beam."
        )
    
    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Get information about available parameters.
        
        Returns
        -------
        Dict[str, Any]
            Parameter information
        """
        return {
            'grid_size': {
                'description': 'Calculation grid size in cm',
                'type': 'float',
                'default': 0.2,
                'range': [0.1, 0.5]
            },
            'threads': {
                'description': 'Number of parallel threads',
                'type': 'int',
                'default': 4,
                'range': [1, 16]
            },
            'tissue_air_ratio_correction': {
                'description': 'Apply tissue-air ratio correction',
                'type': 'bool',
                'default': True
            },
            'use_gpu': {
                'description': 'Use GPU acceleration if available',
                'type': 'bool',
                'default': False
            },
            'pencil_spacing': {
                'description': 'Spacing between pencil beams in cm',
                'type': 'float',
                'default': 0.1,
                'range': [0.05, 0.5]
            },
            'integration_step': {
                'description': 'Integration step size in cm',
                'type': 'float',
                'default': 0.5,
                'range': [0.1, 1.0]
            }
        }
