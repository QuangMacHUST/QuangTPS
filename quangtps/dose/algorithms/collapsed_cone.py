#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of the Collapsed Cone Convolution dose calculation algorithm.

This module provides a class for calculating dose distributions using
the Collapsed Cone Convolution algorithm for radiotherapy treatment planning.
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

class CollapsedConeAlgorithm(DoseCalculationAlgorithm):
    """
    Implementation of the Collapsed Cone Convolution dose calculation algorithm.
    
    This class provides methods for calculating 3D dose distributions
    using the Collapsed Cone Convolution algorithm for radiotherapy treatment planning.
    """
    
    def __init__(self):
        """
        Initialize the Collapsed Cone algorithm.
        """
        super().__init__("Collapsed Cone Convolution")
        self.version = "1.0"
        
        # Default parameters
        self.parameters.update({
            'grid_size': 0.3,  # Calculation grid size in cm
            'threads': 8,  # Number of parallel threads
            'heterogeneity_correction': True,  # Whether to apply heterogeneity correction
            'use_gpu': False,  # Whether to use GPU acceleration
            'number_of_cones': 16,  # Number of cones in each solid angle
            'polyenergetic': True,  # Whether to use polyenergetic calculation
        })
        
        logger.info(f"Initialized {self.name} algorithm version {self.version}")
        
        self.beam_model = None
    
    def calculate(self, ct_image: Image, beam: Beam) -> DoseCalculationResult:
        """
        Calculate dose distribution using Collapsed Cone Convolution algorithm.
        
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
            heterogeneity_correction = self.get_parameter('heterogeneity_correction')
            grid_size = self.get_parameter('grid_size')
            threads = self.get_parameter('threads')
            number_of_cones = self.get_parameter('number_of_cones')
            polyenergetic = self.get_parameter('polyenergetic')
            
            logger.info(f"Starting dose calculation for beam {beam.name}")
            logger.info(f"Parameters: grid_size={grid_size}cm, threads={threads}, heterogeneity={heterogeneity_correction}")
            
            # Convert CT to electron density
            electron_density = self._convert_ct_to_density(ct_image)
            
            # Calculate TERMA
            logger.info("Calculating TERMA...")
            terma_grid = calculate_terma_from_beam(
                ct_image=ct_image,
                beam=beam,
                beam_model=self.beam_model
            )
            
            # Apply collapsed cone convolution
            logger.info(f"Applying collapsed cone convolution with {number_of_cones} cones...")
            dose_data = self._apply_collapsed_cone_convolution(
                terma_grid, 
                electron_density, 
                ct_image.spacing, 
                number_of_cones, 
                heterogeneity_correction
            )
            
            # Validate results
            self._validate_calculation_completed(dose_data)
            
            # Create result object
            calculation_time = time.time() - start_time
            logger.info(f"Dose calculation completed in {calculation_time:.2f} seconds")
            
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
    
    def _apply_collapsed_cone_convolution(
        self, 
        terma: np.ndarray, 
        density: np.ndarray, 
        spacing: Tuple[float, float, float], 
        number_of_cones: int,
        heterogeneity_correction: bool
    ) -> np.ndarray:
        """
        Apply collapsed cone convolution to get dose from TERMA.
        
        Parameters
        ----------
        terma : np.ndarray
            TERMA grid
        density : np.ndarray
            Electron density grid
        spacing : Tuple[float, float, float]
            Grid spacing in mm
        number_of_cones : int
            Number of cones to use in each solid angle
        heterogeneity_correction : bool
            Whether to apply tissue heterogeneity correction
            
        Returns
        -------
        np.ndarray
            Dose grid
        """
        # Initialize dose grid
        dose = np.zeros_like(terma)
        
        # For this implementation, we'll use a simplified approach
        # In a real implementation, this would be much more complex
        
        # Define cone directions based on icosahedron vertices
        # Here we just use a simple spherical sampling
        directions = []
        for i in range(number_of_cones):
            phi = np.arccos(1 - 2 * (i + 0.5) / number_of_cones)
            for j in range(number_of_cones):
                theta = 2 * np.pi * (j + 0.5) / number_of_cones
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                directions.append((x, y, z))
        
        # Apply convolution for each cone direction
        cone_weight = 1.0 / len(directions)
        
        for direction in directions:
            # Get normalized direction vector
            dx, dy, dz = direction
            
            # Trace rays through volume
            nx, ny, nz = terma.shape
            max_dim = max(nx, ny, nz)
            
            # Simplified ray tracing
            # In a real implementation, would use Siddon's algorithm or similar
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if terma[i, j, k] <= 0:
                            continue
                        
                        # Deposit dose along ray
                        for step in range(1, max_dim):
                            # Calculate position along ray
                            ii = int(i + dx * step)
                            jj = int(j + dy * step)
                            kk = int(k + dz * step)
                            
                            # Check if out of bounds
                            if (ii < 0 or ii >= nx or 
                                jj < 0 or jj >= ny or 
                                kk < 0 or kk >= nz):
                                break
                            
                            # Calculate distance in cm
                            dist_cm = step * np.sqrt(dx*dx + dy*dy + dz*dz) * spacing[0] / 10.0
                            
                            # Apply inverse square law and attenuation
                            local_density = density[ii, jj, kk] if heterogeneity_correction else 1.0
                            attenuation = np.exp(-0.05 * dist_cm * local_density)  # Simplified attenuation
                            
                            # Deposit dose
                            dose[ii, jj, kk] += terma[i, j, k] * attenuation * cone_weight / (dist_cm * dist_cm + 0.01)
        
        return dose
        
    def calculate_beam_dose(self, beam: Beam, ct_image: Image) -> Image:
        """
        Calculate dose for a beam using Collapsed Cone Convolution algorithm.
        
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
        
        # Add energetic spectrum
        energies = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0])
        
        # Generic spectrum based on energy
        if energy == "6MV":
            spectrum = np.array([0.01, 0.05, 0.15, 0.25, 0.30, 0.15, 0.05, 0.03, 0.01, 0.0, 0.0])
        elif energy == "10MV":
            spectrum = np.array([0.01, 0.03, 0.10, 0.20, 0.25, 0.20, 0.10, 0.05, 0.04, 0.02, 0.0])
        else:  # Default to 15MV
            spectrum = np.array([0.01, 0.02, 0.05, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05, 0.02])
        
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
    
    def get_description(self) -> str:
        """Get algorithm description."""
        return (
            "Collapsed Cone Convolution algorithm for dose calculation. "
            "This algorithm models the energy deposition by dividing the solid angle "
            "into cones and calculating dose deposition along these cones."
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
                'default': 0.3,
                'range': [0.1, 1.0]
            },
            'threads': {
                'description': 'Number of parallel threads',
                'type': 'int',
                'default': 8,
                'range': [1, 32]
            },
            'heterogeneity_correction': {
                'description': 'Apply tissue heterogeneity correction',
                'type': 'bool',
                'default': True
            },
            'use_gpu': {
                'description': 'Use GPU acceleration if available',
                'type': 'bool',
                'default': False
            },
            'number_of_cones': {
                'description': 'Number of cones in each solid angle',
                'type': 'int',
                'default': 16,
                'range': [8, 64]
            },
            'polyenergetic': {
                'description': 'Use polyenergetic calculation',
                'type': 'bool',
                'default': True
            }
        } 