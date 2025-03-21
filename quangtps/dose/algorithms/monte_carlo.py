#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of the Monte Carlo dose calculation algorithm.

This module provides a class for calculating dose distributions using
the Monte Carlo algorithm for radiotherapy treatment planning.
"""

import os
import numpy as np
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor

from quangtps.core.exceptions import DoseCalculationError
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.dose.beam_data_processor import BeamModel, BeamModelParameter

logger = logging.getLogger(__name__)

class MonteCarloAlgorithm:
    """
    Implementation of the Monte Carlo dose calculation algorithm.
    
    This class provides methods for calculating 3D dose distributions
    using the Monte Carlo algorithm for radiotherapy treatment planning.
    """
    
    DEFAULT_PARAMS = {
        "num_histories": 1000000,  # Number of particle histories to simulate
        "grid_size": 0.25,         # Calculation grid size in cm
        "energy_cutoff": 0.01,     # Energy cutoff for particle tracking in MeV
        "statistical_uncertainty": 2.0,  # Target statistical uncertainty in %
        "threads": 8,              # Number of CPU threads to use
        "use_gpu": False,          # Whether to use GPU acceleration if available
        "max_chunk_size": 100000   # Maximum chunk size for particle batches
    }
    
    def __init__(self):
        """
        Initialize the Monte Carlo algorithm with default parameters.
        """
        self.beam_model = None
        self.parameters = self.DEFAULT_PARAMS.copy()
        self.material_lookup = None  # Will store CT number to material conversion
        self.cross_section_data = None  # Will store material cross section data
        
        # Initialize random number generator
        self.rng = np.random.RandomState(seed=42)
        
        logger.info("Initialized Monte Carlo algorithm")
    
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
        
        # Load energy spectrum from beam model
        if self.beam_model.has_parameter("energy_spectrum"):
            self.energy_spectrum = self.beam_model.get_parameter("energy_spectrum")
            logger.info("Loaded energy spectrum from beam model")
        else:
            logger.warning("Beam model does not contain energy spectrum, using default")
    
    def set_parameter(self, name: str, value: Any):
        """
        Set a calculation parameter.
        
        Parameters
        ----------
        name : str
            Parameter name
        value : Any
            Parameter value
        """
        if name in self.parameters:
            self.parameters[name] = value
            logger.info(f"Set parameter {name} = {value}")
        else:
            logger.warning(f"Unknown parameter: {name}")
    
    def set_parameters(self, params: Dict[str, Any]):
        """
        Set multiple calculation parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameter name/value pairs
        """
        for name, value in params.items():
            self.set_parameter(name, value)
    
    def load_cross_section_data(self, data_file: str):
        """
        Load cross-section data for Monte Carlo simulation.
        
        Parameters
        ----------
        data_file : str
            Path to cross-section data file
        """
        try:
            with open(data_file, 'r') as f:
                self.cross_section_data = json.load(f)
            logger.info(f"Loaded cross-section data from {data_file}")
        except Exception as e:
            logger.error(f"Failed to load cross-section data: {str(e)}")
    
    def load_material_lookup(self, data_file: str):
        """
        Load CT number to material conversion table.
        
        Parameters
        ----------
        data_file : str
            Path to material lookup data file
        """
        try:
            with open(data_file, 'r') as f:
                self.material_lookup = json.load(f)
            logger.info(f"Loaded material lookup table from {data_file}")
        except Exception as e:
            logger.error(f"Failed to load material lookup table: {str(e)}")
    
    def calculate_beam_dose(self, beam: Beam, ct_image: Image) -> Image:
        """
        Calculate dose for a single beam using Monte Carlo simulation.
        
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
            # Start timing
            start_time = time.time()
            logger.info(f"Starting Monte Carlo calculation for beam: {beam.name}")
            logger.info(f"Using {self.parameters['num_histories']} histories with " 
                        f"{self.parameters['threads']} threads")
            
            # Convert CT to materials and densities
            materials, densities = self._convert_ct_to_materials(ct_image)
            
            # Initialize dose and uncertainty grids
            dose_grid = np.zeros_like(ct_image.data, dtype=np.float32)
            uncertainty_grid = np.zeros_like(ct_image.data, dtype=np.float32)
            
            # Get beam parameters
            source_position = beam.get_source_position()
            isocenter = beam.isocenter
            field_size = beam.field_size
            gantry_angle = beam.gantry_angle
            collimator_angle = beam.collimator_angle
            couch_angle = beam.couch_angle
            
            # Get energy spectrum
            if self.beam_model.has_parameter("energy_spectrum"):
                energy_spectrum = self.beam_model.get_parameter("energy_spectrum")
                energies = energy_spectrum.dimension_values[0]
                probabilities = energy_spectrum.value_grid
            else:
                # Default energy spectrum if not available
                energy_mean = float(beam.energy.replace("MV", "").replace("X", ""))
                energies, probabilities = self._create_default_spectrum(energy_mean)
            
            # Split calculation into chunks for parallelization
            num_histories = self.parameters["num_histories"]
            chunk_size = min(num_histories // self.parameters["threads"], 
                             self.parameters["max_chunk_size"])
            num_chunks = int(np.ceil(num_histories / chunk_size))
            
            logger.info(f"Splitting calculation into {num_chunks} chunks of " 
                        f"{chunk_size} histories each")
            
            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=self.parameters["threads"]) as executor:
                futures = []
                
                for i in range(num_chunks):
                    # Calculate chunk size (last chunk may be smaller)
                    actual_chunk_size = min(chunk_size, num_histories - i * chunk_size)
                    
                    # Submit chunk for processing
                    future = executor.submit(
                        self._simulate_particles,
                        actual_chunk_size,
                        ct_image.shape,
                        ct_image.spacing,
                        ct_image.origin,
                        materials,
                        densities,
                        source_position,
                        isocenter,
                        field_size,
                        gantry_angle,
                        collimator_angle,
                        couch_angle,
                        energies,
                        probabilities,
                        i  # Seed offset
                    )
                    futures.append(future)
                
                # Collect results from all chunks
                for i, future in enumerate(futures):
                    try:
                        chunk_dose, chunk_uncertainty = future.result()
                        
                        # Combine results (weighting by number of histories)
                        dose_grid += chunk_dose
                        uncertainty_grid += chunk_uncertainty
                        
                        logger.info(f"Completed chunk {i+1}/{num_chunks}")
                    except Exception as e:
                        logger.error(f"Error in chunk {i+1}: {str(e)}")
            
            # Normalize by total number of histories
            dose_grid /= num_histories
            
            # Calculate final statistical uncertainty
            valid_dose = dose_grid > 0
            if np.any(valid_dose):
                mean_uncertainty = np.mean(uncertainty_grid[valid_dose] / dose_grid[valid_dose]) * 100
                logger.info(f"Mean statistical uncertainty: {mean_uncertainty:.2f}%")
            
            # Create dose image
            dose_image = Image(
                data=dose_grid,
                spacing=ct_image.spacing,
                origin=ct_image.origin,
                direction=ct_image.direction
            )
            
            # Normalize to isocenter
            self._normalize_to_isocenter(dose_image, isocenter)
            
            # Calculate total time
            elapsed_time = time.time() - start_time
            logger.info(f"Monte Carlo calculation completed in {elapsed_time:.2f} seconds")
            
            return dose_image
            
        except Exception as e:
            error_msg = f"Error in Monte Carlo dose calculation: {str(e)}"
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
        
        # Add energy spectrum
        energy_mean = float(energy.replace("MV", "").replace("X", ""))
        energies, probabilities = self._create_default_spectrum(energy_mean)
        
        spectrum_parameter = BeamModelParameter(
            name="energy_spectrum",
            value_grid=probabilities,
            dimensions=["energy"],
            units=["MeV"],
            dimension_values=[energies],
            interpolation_method="linear"
        )
        model.add_parameter(spectrum_parameter)
        
        # Add fluence map (uniform)
        x_pos = np.linspace(-20, 20, 41)
        y_pos = np.linspace(-20, 20, 41)
        fluence_map = np.ones((len(y_pos), len(x_pos)))
        
        fluence_parameter = BeamModelParameter(
            name="fluence_map",
            value_grid=fluence_map,
            dimensions=["y", "x"],
            units=["cm", "cm"],
            dimension_values=[y_pos, x_pos],
            interpolation_method="linear"
        )
        model.add_parameter(fluence_parameter)
        
        # Add angular distribution (for particle direction sampling)
        # This is a simplified model - real implementation would include more details
        angles = np.linspace(0, 5, 11)  # Angles from 0 to 5 degrees
        distribution = np.exp(-angles**2 / 2)  # Approximately Gaussian
        
        # Normalize
        distribution = distribution / np.sum(distribution)
        
        angular_parameter = BeamModelParameter(
            name="angular_distribution",
            value_grid=distribution,
            dimensions=["angle"],
            units=["degree"],
            dimension_values=[angles],
            interpolation_method="linear"
        )
        model.add_parameter(angular_parameter)
        
        return model
    
    def _create_default_spectrum(self, nominal_energy: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a default energy spectrum for a given nominal energy.
        
        Parameters
        ----------
        nominal_energy : float
            Nominal beam energy in MV
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Energy values and corresponding probabilities
        """
        # Create a simple energy distribution
        # This is a very simplified model - real spectra are more complex
        
        # Create energy bins from 0 to the nominal energy
        energies = np.linspace(0.1, nominal_energy, 50)
        
        # Create probabilities (simplified model)
        # Shape is roughly based on typical photon spectra
        probabilities = (energies / nominal_energy) * np.exp(-(energies / nominal_energy)**2 * 3)
        
        # Add a peak at higher energy (bremsstrahlung peak)
        peak_pos = 0.8 * nominal_energy
        peak_idx = np.argmin(np.abs(energies - peak_pos))
        probabilities[peak_idx:] += 0.5 * np.exp(-((energies[peak_idx:] - peak_pos) / (0.1 * nominal_energy))**2)
        
        # Normalize
        probabilities /= np.sum(probabilities)
        
        return energies, probabilities
    
    def _convert_ct_to_materials(self, ct_image: Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert CT image to material indices and densities.
        
        Parameters
        ----------
        ct_image : Image
            The CT image in Hounsfield Units
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Material indices and densities
        """
        # Simple conversion from HU to material and density
        # In a real implementation, this would use the material lookup table
        
        hu_values = ct_image.data
        
        # Default material indices (simplified)
        # 0: Air, 1: Soft tissue, 2: Bone
        material_indices = np.zeros_like(hu_values, dtype=np.int32)
        
        # Set material based on HU value
        material_indices[(hu_values > -500) & (hu_values <= 100)] = 1  # Soft tissue
        material_indices[hu_values > 100] = 2  # Bone
        
        # Calculate density relative to water
        densities = np.ones_like(hu_values, dtype=np.float32)
        
        # Air region
        air_mask = hu_values <= -500
        densities[air_mask] = 0.00121 * (1 + hu_values[air_mask] / 1000)
        
        # Soft tissue region
        tissue_mask = (hu_values > -500) & (hu_values <= 100)
        densities[tissue_mask] = 1.0 + 0.001 * hu_values[tissue_mask]
        
        # Bone region
        bone_mask = hu_values > 100
        densities[bone_mask] = 1.0 + 0.001 * hu_values[bone_mask]
        
        return material_indices, densities
    
    def _simulate_particles(self, 
                           num_histories: int,
                           grid_shape: Tuple[int, int, int],
                           grid_spacing: Tuple[float, float, float],
                           grid_origin: Tuple[float, float, float],
                           materials: np.ndarray,
                           densities: np.ndarray,
                           source_position: np.ndarray,
                           isocenter: np.ndarray,
                           field_size: Tuple[float, float],
                           gantry_angle: float,
                           collimator_angle: float,
                           couch_angle: float,
                           energies: np.ndarray,
                           energy_probabilities: np.ndarray,
                           seed_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a batch of particles for Monte Carlo dose calculation.
        
        This is the core function that simulates individual particles through the CT geometry.
        
        Parameters
        ----------
        num_histories : int
            Number of particle histories to simulate
        grid_shape : Tuple[int, int, int]
            Shape of the CT/dose grid
        grid_spacing : Tuple[float, float, float]
            Spacing of the grid in cm
        grid_origin : Tuple[float, float, float]
            Origin of the grid in world coordinates
        materials : np.ndarray
            Material indices for each voxel
        densities : np.ndarray
            Density values for each voxel
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
        couch_angle : float
            Couch angle in degrees
        energies : np.ndarray
            Energy bins for the spectrum
        energy_probabilities : np.ndarray
            Probability distribution for the energy spectrum
        seed_offset : int
            Offset for the random seed
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Dose and uncertainty grids
        """
        # Initialize dose and uncertainty grids
        dose_grid = np.zeros(grid_shape, dtype=np.float32)
        squared_dose_grid = np.zeros(grid_shape, dtype=np.float32)  # For uncertainty calculation
        
        # Initialize random number generator with offset seed for reproducibility
        rng = np.random.RandomState(seed=42 + seed_offset)
        
        # Helper variables
        nz, ny, nx = grid_shape
        dz, dy, dx = grid_spacing
        
        # Get beam direction based on gantry and couch angles
        gantry_rad = np.radians(gantry_angle)
        couch_rad = np.radians(couch_angle)
        collimator_rad = np.radians(collimator_angle)
        
        # Main beam direction
        beam_dir = np.array([
            np.sin(gantry_rad) * np.cos(couch_rad),
            np.sin(couch_rad),
            -np.cos(gantry_rad) * np.cos(couch_rad)
        ])
        
        # Perpendicular directions (accounting for collimator rotation)
        perp1 = np.array([
            np.cos(gantry_rad) * np.cos(collimator_rad) + np.sin(gantry_rad) * np.sin(couch_rad) * np.sin(collimator_rad),
            -np.cos(couch_rad) * np.sin(collimator_rad),
            np.sin(gantry_rad) * np.cos(collimator_rad) - np.cos(gantry_rad) * np.sin(couch_rad) * np.sin(collimator_rad)
        ])
        
        perp2 = np.array([
            -np.cos(gantry_rad) * np.sin(collimator_rad) + np.sin(gantry_rad) * np.sin(couch_rad) * np.cos(collimator_rad),
            np.cos(couch_rad) * np.cos(collimator_rad),
            -np.sin(gantry_rad) * np.sin(collimator_rad) - np.cos(gantry_rad) * np.sin(couch_rad) * np.cos(collimator_rad)
        ])
        
        # Source to isocenter distance (typically 100 cm)
        SID = np.linalg.norm(source_position - isocenter)
        
        # Half field size in cm
        half_field_x = field_size[0] / 2
        half_field_y = field_size[1] / 2
        
        # Energy cutoff
        energy_cutoff = self.parameters["energy_cutoff"]
        
        # Simulate particles
        for i in range(num_histories):
            # Sample particle energy from spectrum
            energy = rng.choice(energies, p=energy_probabilities)
            
            # Sample initial particle position (at source, with appropriate field spread)
            # This is the starting point of the particle at the source
            initial_pos = source_position.copy()
            
            # Sample initial direction (with appropriate beam divergence)
            # First sample uniformly within field borders at isocenter plane
            x_iso = rng.uniform(-half_field_x, half_field_x)
            y_iso = rng.uniform(-half_field_y, half_field_y)
            
            # Convert to a direction from source to isocenter plane
            direction = beam_dir + (x_iso / SID) * perp1 + (y_iso / SID) * perp2
            direction = direction / np.linalg.norm(direction)
            
            # Add small random angular spread (simplified model of source size/angular distribution)
            spread = 0.01  # Radians
            theta = rng.normal(0, spread)
            phi = rng.uniform(0, 2 * np.pi)
            
            # Apply small rotation to direction
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            # Create orthogonal basis
            v1 = direction
            v2 = np.array([1, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1, 0])
            v2 = v2 - np.dot(v2, v1) * v1
            v2 = v2 / np.linalg.norm(v2)
            v3 = np.cross(v1, v2)
            
            # Apply rotation
            direction = cos_theta * v1 + sin_theta * cos_phi * v2 + sin_theta * sin_phi * v3
            
            # Track the particle through the geometry
            pos = initial_pos.copy()
            
            while energy > energy_cutoff:
                # Convert position to voxel indices
                z_idx = int((pos[0] - grid_origin[0]) / dz)
                y_idx = int((pos[1] - grid_origin[1]) / dy)
                x_idx = int((pos[2] - grid_origin[2]) / dx)
                
                # Check if we're inside the grid
                if (0 <= z_idx < nz and 0 <= y_idx < ny and 0 <= x_idx < nx):
                    # Get material and density at current position
                    material = materials[z_idx, y_idx, x_idx]
                    density = densities[z_idx, y_idx, x_idx]
                    
                    # Simplified interaction model
                    # In a real implementation, this would use cross-section data
                    
                    # Determine step size based on mean free path
                    # This is a simplified model - real implementation would be more complex
                    mfp = self._calculate_mean_free_path(energy, material, density)
                    step_size = -mfp * np.log(rng.uniform(0, 1))
                    
                    # Calculate energy deposition in this step
                    if material > 0:  # Not air
                        # Simplified energy deposition calculation
                        # Real implementation would account for different interaction processes
                        energy_dep = energy * 0.01 * density
                        
                        # Deposit energy in voxel
                        dose_grid[z_idx, y_idx, x_idx] += energy_dep
                        squared_dose_grid[z_idx, y_idx, x_idx] += energy_dep ** 2
                    
                    # Update energy (simplified)
                    energy_loss = energy * 0.01 * density
                    energy -= energy_loss
                    
                    # Update position
                    pos += direction * step_size
                    
                    # Small chance of scattering (simplified)
                    if rng.uniform(0, 1) < 0.1:
                        # Simplified isotropic scattering for demonstration
                        theta = np.arccos(1 - 2 * rng.uniform(0, 1))
                        phi = 2 * np.pi * rng.uniform(0, 1)
                        
                        # Apply same rotation method as above
                        sin_theta = np.sin(theta)
                        cos_theta = np.cos(theta)
                        sin_phi = np.sin(phi)
                        cos_phi = np.cos(phi)
                        
                        # Create orthogonal basis
                        v1 = direction
                        v2 = np.array([1, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1, 0])
                        v2 = v2 - np.dot(v2, v1) * v1
                        v2 = v2 / np.linalg.norm(v2)
                        v3 = np.cross(v1, v2)
                        
                        # Apply rotation
                        direction = cos_theta * v1 + sin_theta * cos_phi * v2 + sin_theta * sin_phi * v3
                else:
                    # Particle left the grid
                    break
        
        # Calculate uncertainty grid
        uncertainty_grid = np.sqrt(squared_dose_grid - (dose_grid ** 2) / num_histories)
        
        return dose_grid, uncertainty_grid
    
    def _calculate_mean_free_path(self, energy: float, material: int, density: float) -> float:
        """
        Calculate mean free path for a particle in a material.
        
        Parameters
        ----------
        energy : float
            Particle energy in MeV
        material : int
            Material index
        density : float
            Material density in g/cm³
            
        Returns
        -------
        float
            Mean free path in cm
        """
        # Simplified model - real implementation would use cross-section data
        # and account for different interaction processes
        
        # Base mean free path values at 1 MeV for different materials
        base_mfp = {
            0: 5000.0,  # Air (very long mfp)
            1: 15.0,    # Soft tissue
            2: 5.0      # Bone
        }
        
        # Energy scaling (simplified)
        # Higher energy typically means longer mfp
        energy_factor = energy ** 0.5
        
        # Density scaling
        # Higher density means shorter mfp
        density_factor = 1.0 / density if density > 0 else 1.0
        
        # Calculate mfp
        mfp = base_mfp.get(material, 10.0) * energy_factor * density_factor
        
        return mfp
    
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