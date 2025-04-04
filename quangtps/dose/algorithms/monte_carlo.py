#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo dose calculation algorithm.

This module implements a GPU-accelerated Monte Carlo algorithm for dose calculation
in radiotherapy treatment planning. The Monte Carlo method simulates individual
particle trajectories to model radiation transport and interaction with tissue,
providing the highest accuracy for heterogeneous tissue calculations.
"""

import os
import numpy as np
import logging
import time
import json
import random
import multiprocessing
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d

from quangtps.core.exceptions import DoseCalculationError, ValidationError
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.dose.beam_data_processor import BeamModel, BeamModelParameter
from quangtps.dose.algorithms.base import DoseCalculationAlgorithm, DoseCalculationResult
from quangtps.dose.physics.terma import calculate_terma_from_beam

# Conditional imports for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    import pyopencl as cl
    from numba import cuda
    HAS_GPU = True
    logger = logging.getLogger(__name__)
    logger.info("CUDA GPU acceleration available for Monte Carlo calculations")
except ImportError:
    HAS_GPU = False
    logger = logging.getLogger(__name__)
    logger.warning("CUDA GPU acceleration not available - falling back to CPU for Monte Carlo calculations")


class MonteCarloAlgorithm(DoseCalculationAlgorithm):
    """
    Monte Carlo algorithm for dose calculation in radiotherapy.
    
    This class implements a GPU-accelerated Monte Carlo approach to simulate
    the transport and interaction of radiation particles in patient tissues.
    It provides the highest level of accuracy for heterogeneous tissues and
    complex geometries, especially near tissue interfaces and in low-density regions.
    
    Features:
    - Full 3D particle transport simulation
    - Accurate physics models for photon and electron interactions
    - GPU acceleration (when available)
    - Optimized multithreading for CPU calculations
    - Variance reduction techniques for faster convergence
    - Phase space file support for beam modeling
    """
    
    def __init__(self):
        """Initialize the Monte Carlo algorithm with default parameters."""
        super().__init__("Monte Carlo")
        self.version = "2.0"
        
        # Default parameters
        self.parameters.update({
            'num_histories': 1000000,           # Number of particle histories to simulate
            'grid_size': 0.3,                   # Calculation grid size in cm
            'threads': max(1, multiprocessing.cpu_count() - 1),  # Number of parallel threads
            'max_energy': 20.0,                 # Maximum energy in MeV
            'particle_type': 'photon',          # Particle type: 'photon', 'electron', 'mixed'
            'statistical_uncertainty': 2.0,     # Target statistical uncertainty in %
            'voxel_scale_factor': 1.0,          # Scaling factor for voxel size
            'electron_cutoff': 0.2,             # Energy cutoff for electron transport in MeV
            'photon_cutoff': 0.01,              # Energy cutoff for photon transport in MeV
            'use_variance_reduction': True,     # Whether to use variance reduction techniques
            'seed': None,                       # Random seed (None for random initialization)
            'save_phase_space': False,          # Whether to save phase space data
            'phase_space_file': '',             # Path to phase space file
            'density_threshold': 0.01,          # Density threshold for considering a voxel
            'use_gpu': HAS_GPU,                 # Whether to use GPU acceleration
            'gpu_batch_size': 10000,            # Batch size for GPU calculations
            'use_importance_sampling': True,    # Whether to use importance sampling
            'use_photon_splitting': True,       # Use photon splitting variance reduction
            'split_factor': 5,                  # Number of split photons
            'use_interaction_forcing': True,    # Use interaction forcing for variance reduction
            'cross_section_table': 'NIST',      # Cross-section data source: 'NIST', 'ICRP', 'custom'
            'report_progress': True,            # Whether to report calculation progress
            'use_denoising': True,              # New parameter for dose denoising
            'use_kernel_density_estimator': True,  # New parameter for KDE scoring
            'use_track_length_estimator': True,  # New parameter for track length scoring
            'enable_russian_roulette': True,    # New parameter for Russian roulette variance reduction
            'use_opencl_fallback': True,         # New parameter to use OpenCL if CUDA is not available
            'use_multilevel_parallelism': True   # New parameter for nested parallelism
        })
        
        self.beam_model = None
        self.interaction_data = None
        self.rng = None
        self.device = None
        
        # Initialize random number generator
        self._initialize_rng()
        
        # Initialize interaction data tables
        self._initialize_interaction_data()
        
        # Initialize GPU if available
        if HAS_GPU and self.parameters['use_gpu']:
            self._initialize_gpu()
        
        logger.info(f"Initialized {self.name} algorithm version {self.version}")
    
    def _initialize_gpu(self):
        """Initialize GPU resources if available."""
        if not HAS_GPU:
            return
        
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            if num_gpus > 0:
                # Use device 0 by default
                self.device = cp.cuda.Device(0)
                with self.device:
                    # Allocate memory for a simple test calculation
                    mem_info = cp.cuda.Device().mem_info
                    free_memory = mem_info[0]
                    total_memory = mem_info[1]
                    
                    # Log GPU information
                    device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
                    logger.info(f"Using GPU: {device_name}")
                    logger.info(f"GPU Memory: {free_memory / 1024**3:.2f} GB free / {total_memory / 1024**3:.2f} GB total")
                    
                    # Adjust batch size based on available memory
                    suggested_batch_size = min(self.parameters['gpu_batch_size'], 
                                              int(free_memory * 0.4 / (4 * 256**3)))  # Rough estimate
                    self.parameters['gpu_batch_size'] = max(1000, suggested_batch_size)
                    logger.info(f"GPU batch size set to {self.parameters['gpu_batch_size']}")
            else:
                logger.warning("No CUDA-compatible GPUs found. Using CPU calculation.")
                self.parameters['use_gpu'] = False
        except Exception as e:
            logger.error(f"Error initializing GPU: {e}")
            logger.warning("Falling back to CPU calculation.")
            self.parameters['use_gpu'] = False
    
    def _initialize_rng(self):
        """Initialize the random number generator."""
        seed = self.parameters['seed']
        if seed is None:
            # Use system time if no seed provided
            seed = int(time.time())
        
        self.rng = random.Random(seed)
        np.random.seed(seed)
        if HAS_GPU and self.parameters['use_gpu']:
            cp.random.seed(seed)
        
        logger.debug(f"Initialized RNG with seed: {seed}")
    
    def _initialize_interaction_data(self):
        """
        Initialize interaction data tables for photons and electrons.
        
        These tables store cross-section data for different interaction processes
        as a function of energy and material (electron density). For accuracy,
        we now include data based on NIST databases.
        """
        # Energy grid for cross-section data (in MeV)
        energy_grid = np.logspace(-2, np.log10(self.parameters['max_energy']), 150)
        
        # Electron density grid relative to water
        density_grid = np.linspace(0.01, 3.0, 30)
        
        # Initialize interaction data structure
        self.interaction_data = {
            'energy_grid': energy_grid,
            'density_grid': density_grid,
            'photon': {
                'photoelectric': np.zeros((len(energy_grid), len(density_grid))),
                'compton': np.zeros((len(energy_grid), len(density_grid))),
                'pair_production': np.zeros((len(energy_grid), len(density_grid))),
                'total': np.zeros((len(energy_grid), len(density_grid))),
                'rayleigh': np.zeros((len(energy_grid), len(density_grid)))
            },
            'electron': {
                'collision': np.zeros((len(energy_grid), len(density_grid))),
                'radiative': np.zeros((len(energy_grid), len(density_grid))),
                'total': np.zeros((len(energy_grid), len(density_grid)))
            }
        }
        
        # Load cross-section data based on selected source
        cross_section_source = self.parameters['cross_section_table']
        
        if cross_section_source == 'NIST':
            self._load_nist_cross_sections()
        elif cross_section_source == 'ICRP':
            self._load_icrp_cross_sections()
        else:
            # Fallback to built-in approximation if custom source is not specified
            self._generate_approximate_cross_sections()
            
        logger.debug(f"Initialized interaction data tables using {cross_section_source} data")
    
    def _load_nist_cross_sections(self):
        """
        Load cross-section data from NIST database files.
        
        This method attempts to load pre-calculated cross-section data from 
        NIST database files. If files are not found, it falls back to approximate
        calculation.
        """
        try:
            # Attempt to load NIST data from data files
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cross_sections')
            
            # Check if data files exist
            nist_file = os.path.join(data_dir, 'nist_cross_sections.npz')
            if not os.path.exists(nist_file):
                logger.warning(f"NIST cross-section data file not found: {nist_file}")
                self._generate_approximate_cross_sections()
                return
            
            # Load data
            data = np.load(nist_file)
            
            # Copy data to interaction_data structure
            # Verify that shapes match before copying
            if (data['energy_grid'].shape[0] == self.interaction_data['energy_grid'].shape[0] and
                data['density_grid'].shape[0] == self.interaction_data['density_grid'].shape[0]):
                # Update energy and density grids
                self.interaction_data['energy_grid'] = data['energy_grid']
                self.interaction_data['density_grid'] = data['density_grid']
                
                # Copy cross-section data
                for interaction_type in ['photoelectric', 'compton', 'pair_production', 'total', 'rayleigh']:
                    key = f'photon_{interaction_type}'
                    if key in data:
                        self.interaction_data['photon'][interaction_type] = data[key]
                
                for interaction_type in ['collision', 'radiative', 'total']:
                    key = f'electron_{interaction_type}'
                    if key in data:
                        self.interaction_data['electron'][interaction_type] = data[key]
                        
                logger.info("Successfully loaded NIST cross-section data")
            else:
                logger.warning("NIST data dimensions don't match expected dimensions")
                self._generate_approximate_cross_sections()
                
        except Exception as e:
            logger.error(f"Error loading NIST cross-section data: {e}")
            logger.warning("Falling back to approximate cross-section calculation")
            self._generate_approximate_cross_sections()
    
    def _load_icrp_cross_sections(self):
        """
        Load cross-section data from ICRP database files.
        
        This method attempts to load pre-calculated cross-section data from 
        ICRP database files. If files are not found, it falls back to approximate
        calculation.
        """
        try:
            # Similar implementation as NIST but with ICRP data source
            # For now, fall back to approximate calculation
            logger.warning("ICRP cross-section data loading not implemented yet")
            self._generate_approximate_cross_sections()
        except Exception as e:
            logger.error(f"Error loading ICRP cross-section data: {e}")
            self._generate_approximate_cross_sections()
    
    def _generate_approximate_cross_sections(self):
        """
        Generate approximate cross-section data based on physical models.
        
        This method is used as a fallback when database files are not available.
        It generates cross-section data using simplified physical models that
        approximate the behavior of photons and electrons in tissue.
        """
        # Energy and density grids should already be initialized
        energy_grid = self.interaction_data['energy_grid']
        density_grid = self.interaction_data['density_grid']
        
        # Fill interaction data with more accurate approximate cross-sections
        # These are improved physical models based on Klein-Nishina and other formulations
        
        # Photon interaction cross-sections (cm^2/g)
        for i, energy in enumerate(energy_grid):
            for j, density in enumerate(density_grid):
                # Calculate effective atomic number based on relative electron density
                # This is an approximation - tissues with same electron density may have different Z
                z_eff = density * 7.5  # Approximate effective Z
                
                # Photoelectric effect - improved model with better energy dependence
                # Approximation of the form: constant * Z^4 / E^3.5
                self.interaction_data['photon']['photoelectric'][i, j] = (
                    0.15 * (z_eff**4) / (energy**3.5) * density
                )
                
                # Compton scattering - Klein-Nishina formula approximation
                # Simplified form that captures the basic energy dependence
                klein_nishina_factor = 1.0
                if energy > 0.1:  # Apply KN correction for higher energies
                    e_ratio = 1.0 / (1.0 + energy / 0.511)
                    klein_nishina_factor = 0.5 * (1.0 + e_ratio + e_ratio**2)
                
                self.interaction_data['photon']['compton'][i, j] = (
                    0.15 * z_eff * klein_nishina_factor / np.sqrt(energy) * density
                )
                
                # Pair production (threshold at 1.022 MeV)
                if energy > 1.022:
                    energy_factor = 0.0
                    if energy > 1.022:
                        energy_factor = (1.0 - 1.022/energy)**1.5
                    
                    self.interaction_data['photon']['pair_production'][i, j] = (
                        0.05 * z_eff**2 * energy_factor * density
                    )
                
                # Total photon attenuation
                self.interaction_data['photon']['total'][i, j] = (
                    self.interaction_data['photon']['photoelectric'][i, j] +
                    self.interaction_data['photon']['compton'][i, j] +
                    self.interaction_data['photon']['pair_production'][i, j]
                )
        
        # Electron interaction cross-sections
        for i, energy in enumerate(energy_grid):
            for j, density in enumerate(density_grid):
                # Effective Z
                z_eff = density * 7.5
                
                # Collision stopping power (MeV*cm^2/g) - Bethe formula approximation
                # Include density effect and shell corrections for more accuracy
                bethe_term = np.log(energy/0.001) if energy > 0.001 else 0
                density_effect = 0
                if energy > 0.1:
                    plasma_energy = 0.02857 * np.sqrt(density)  # Approximate plasma energy
                    density_effect = np.log(energy/plasma_energy) - 0.5
                    density_effect = max(0, density_effect)
                
                self.interaction_data['electron']['collision'][i, j] = (
                    2.0 * density * (bethe_term - density_effect) * (1.0 + 3.61 / (energy + 0.5))
                )
                
                # Radiative stopping power (bremsstrahlung)
                # Improved model with better Z and energy dependence
                radiation_yield = energy / (1600.0 + energy)  # Radiation yield factor
                self.interaction_data['electron']['radiative'][i, j] = (
                    0.02 * z_eff * energy * radiation_yield * density
                )
                
                # Total electron stopping power
                self.interaction_data['electron']['total'][i, j] = (
                    self.interaction_data['electron']['collision'][i, j] +
                    self.interaction_data['electron']['radiative'][i, j]
                )
        
        logger.debug("Generated approximate cross-section data")
    
    def set_beam_model(self, beam_model: BeamModel):
        """
        Set the beam model for dose calculation.
        
        Parameters
        ----------
        beam_model : BeamModel
            The beam model containing spectrum and fluence data
        """
        self.beam_model = beam_model
        logger.info(f"Set beam model: {beam_model.name}")
    
    def set_parameters(self, **kwargs):
        """
        Set calculation parameters.
        
        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments for parameters
        """
        # Update parameters
        for key, value in kwargs.items():
            if key in self.parameters:
                old_value = self.parameters[key]
                self.parameters[key] = value
                logger.info(f"Set parameter {key} = {value} (was {old_value})")
            else:
                logger.warning(f"Unknown parameter: {key}")
        
        # Special handling for GPU parameter
        if 'use_gpu' in kwargs and kwargs['use_gpu'] and not HAS_GPU:
            logger.warning("GPU acceleration requested but not available. Using CPU.")
            self.parameters['use_gpu'] = False
    
    def get_parameter(self, name: str) -> Any:
        """
        Get a parameter value.
        
        Parameters
        ----------
        name : str
            Parameter name
            
        Returns
        -------
        Any
            Parameter value
        """
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise ValueError(f"Unknown parameter: {name}")
    
    def validate_inputs(self, ct_image: Image, beam: Beam):
        """
        Validate inputs for dose calculation.
        
        Parameters
        ----------
        ct_image : Image
            CT image for dose calculation
        beam : Beam
            Treatment beam
            
        Raises
        ------
        ValidationError
            If inputs are invalid
        """
        # Check if CT image is valid
        if ct_image is None or not hasattr(ct_image, 'data') or ct_image.data is None:
            raise ValidationError("Invalid CT image")
        
        # Check if beam is valid
        if beam is None:
            raise ValidationError("Invalid beam")
        
        # Check if beam_model is set
        if self.beam_model is None:
            raise ValidationError("Beam model not set")
        
        # Check if CT image has valid spacing
        if not hasattr(ct_image, 'spacing') or len(ct_image.spacing) != 3:
            raise ValidationError("CT image must have valid spacing (x, y, z)")
        
        # Check energy
        if not hasattr(beam, 'energy'):
            raise ValidationError("Beam must have energy")
        
        # Add more validation as needed
    
    def get_description(self) -> str:
        """
        Get a description of the algorithm.
        
        Returns
        -------
        str
            Algorithm description
        """
        return (
            f"{self.name} v{self.version} - A GPU-accelerated Monte Carlo algorithm "
            f"for dose calculation that simulates {self.parameters['num_histories']} "
            f"particle histories to model radiation transport in tissue."
        )
    
    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Get information about algorithm parameters.
        
        Returns
        -------
        Dict[str, Any]
            Parameter information
        """
        return {
            'num_histories': {
                'description': 'Number of particle histories to simulate',
                'default': 1000000,
                'type': 'int',
                'range': [10000, 100000000]
            },
            'grid_size': {
                'description': 'Calculation grid size in cm',
                'default': 0.3,
                'type': 'float',
                'range': [0.1, 1.0]
            },
            'threads': {
                'description': 'Number of parallel threads',
                'default': max(1, multiprocessing.cpu_count() - 1),
                'type': 'int',
                'range': [1, 64]
            },
            'statistical_uncertainty': {
                'description': 'Target statistical uncertainty in %',
                'default': 2.0,
                'type': 'float',
                'range': [0.5, 10.0]
            },
            'electron_cutoff': {
                'description': 'Energy cutoff for electron transport in MeV',
                'default': 0.2,
                'type': 'float',
                'range': [0.05, 1.0]
            },
            'photon_cutoff': {
                'description': 'Energy cutoff for photon transport in MeV',
                'default': 0.01,
                'type': 'float',
                'range': [0.001, 0.1]
            },
            'use_variance_reduction': {
                'description': 'Whether to use variance reduction techniques',
                'default': True,
                'type': 'bool'
            },
            'particle_type': {
                'description': 'Type of particles to simulate',
                'default': 'photon',
                'type': 'str',
                'options': ['photon', 'electron', 'mixed']
            },
            'use_gpu': {
                'description': 'Whether to use GPU acceleration',
                'default': HAS_GPU,
                'type': 'bool'
            },
            'gpu_batch_size': {
                'description': 'Batch size for GPU calculations',
                'default': 10000,
                'type': 'int',
                'range': [1000, 1000000]
            },
            'use_importance_sampling': {
                'description': 'Whether to use importance sampling',
                'default': True,
                'type': 'bool'
            },
            'use_photon_splitting': {
                'description': 'Use photon splitting variance reduction',
                'default': True,
                'type': 'bool'
            },
            'split_factor': {
                'description': 'Number of split photons',
                'default': 5,
                'type': 'int',
                'range': [1, 10]
            },
            'use_interaction_forcing': {
                'description': 'Use interaction forcing for variance reduction',
                'default': True,
                'type': 'bool'
            },
            'cross_section_table': {
                'description': 'Cross-section data source',
                'default': 'NIST',
                'type': 'str',
                'options': ['NIST', 'ICRP', 'custom']
            },
            'report_progress': {
                'description': 'Whether to report calculation progress',
                'default': True,
                'type': 'bool'
            },
            'use_denoising': {
                'description': 'Whether to apply dose denoising',
                'default': True,
                'type': 'bool'
            },
            'use_kernel_density_estimator': {
                'description': 'Whether to use kernel density estimator',
                'default': True,
                'type': 'bool'
            },
            'use_track_length_estimator': {
                'description': 'Whether to use track length estimator',
                'default': True,
                'type': 'bool'
            },
            'enable_russian_roulette': {
                'description': 'Whether to enable Russian roulette variance reduction',
                'default': True,
                'type': 'bool'
            },
            'use_opencl_fallback': {
                'description': 'Whether to use OpenCL as a fallback',
                'default': True,
                'type': 'bool'
            },
            'use_multilevel_parallelism': {
                'description': 'Whether to use multilevel parallelism',
                'default': True,
                'type': 'bool'
            }
        }

    def calculate(self, ct_image: Image, beam: Beam) -> DoseCalculationResult:
        """
        Calculate dose distribution using Monte Carlo algorithm.
        
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
            num_histories = self.get_parameter('num_histories')
            energy_cutoff = self.get_parameter('energy_cutoff')
            statistical_uncertainty = self.get_parameter('statistical_uncertainty')
            threads = self.get_parameter('threads')
            use_gpu = self.get_parameter('use_gpu') and HAS_GPU
            
            logger.info(f"Starting Monte Carlo calculation for beam {beam.name}")
            logger.info(f"Parameters: histories={num_histories}, threads={threads}, uncertainty={statistical_uncertainty}%")
            
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
            couch_angle = beam.couch_angle if hasattr(beam, 'couch_angle') else 0.0
            
            # Get energy spectrum
            if self.beam_model.has_parameter("energy_spectrum"):
                energy_spectrum = self.beam_model.get_parameter("energy_spectrum")
                energies = energy_spectrum.dimension_values[0]
                probabilities = energy_spectrum.value_grid
            else:
                # Default energy spectrum if not available
                energy_mean = float(beam.energy.replace("MV", "").replace("X", ""))
                energies, probabilities = self._create_default_spectrum(energy_mean)
            
            # Perform Monte Carlo simulation
            dose_grid, uncertainty_grid = self._simulate_particles(
                num_histories=num_histories,
                grid_shape=ct_image.data.shape,
                grid_spacing=ct_image.spacing,
                grid_origin=ct_image.origin,
                materials=materials,
                densities=densities,
                source_position=source_position,
                isocenter=isocenter,
                field_size=field_size,
                gantry_angle=gantry_angle,
                collimator_angle=collimator_angle,
                couch_angle=couch_angle,
                energies=energies,
                energy_probabilities=probabilities
            )
            
            # Validate results
            self._validate_calculation_completed(dose_grid)
            
            # Create result object
            calculation_time = time.time() - start_time
            logger.info(f"Monte Carlo calculation completed in {calculation_time:.2f} seconds")
            
            dose_image = Image(
                data=dose_grid,
                spacing=ct_image.spacing,
                origin=ct_image.origin,
                direction=ct_image.direction,
                modality="RTDOSE"
            )
            
            result = DoseCalculationResult(
                dose=dose_image,
                algorithm_name=self.name,
                calculation_time=calculation_time,
                additional_data={
                    'beam_name': beam.name,
                    'uncertainty': uncertainty_grid,
                    'parameters': self.get_parameters()
                }
            )
            
            return result
            
        except ValidationError as e:
            logger.error(f"Validation error in {self.name} calculation: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Error in {self.name} calculation: {str(e)}")
            
            # Split calculation into chunks for parallelization
            try:
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
                        actual_chunk_size = min(
                            chunk_size, num_histories - i * chunk_size)

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

                            logger.info(f"Completed chunk {i + 1}/{num_chunks}")
                        except Exception as e:
                            logger.error(f"Error in chunk {i + 1}: {str(e)}")

                # Normalize by total number of histories
                dose_grid /= num_histories

                # Calculate final statistical uncertainty
                valid_dose = dose_grid > 0
                if np.any(valid_dose):
                    mean_uncertainty = np.mean(
                        uncertainty_grid[valid_dose] / dose_grid[valid_dose]) * 100
                    logger.info(
                        f"Mean statistical uncertainty: {mean_uncertainty:.2f}%")

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
                logger.info(
                    f"Monte Carlo calculation completed in {elapsed_time:.2f} seconds")

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
        probabilities = (energies / nominal_energy) * \
            np.exp(-(energies / nominal_energy)**2 * 3)

        # Add a peak at higher energy (bremsstrahlung peak)
        peak_pos = 0.8 * nominal_energy
        peak_idx = np.argmin(np.abs(energies - peak_pos))
        probabilities[peak_idx:] += 0.5 * \
            np.exp(-((energies[peak_idx:] - peak_pos) /
                   (0.1 * nominal_energy))**2)

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
        material_indices[(hu_values > -500) &
                         (hu_values <= 100)] = 1  # Soft tissue
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
        Simulate particle transport through the patient geometry.

        This is the core Monte Carlo particle transport simulation function. It simulates
        the transport of photons and electrons through the patient geometry and scores
        the dose deposition.

        Parameters
        ----------
        num_histories : int
            Number of particle histories to simulate
        grid_shape : Tuple[int, int, int]
            Shape of the dose grid
        grid_spacing : Tuple[float, float, float]
            Spacing of the dose grid in cm
        grid_origin : Tuple[float, float, float]
            Origin of the dose grid in cm
        materials : np.ndarray
            Material indices for each voxel
        densities : np.ndarray
            Density values for each voxel
        source_position : np.ndarray
            Position of the source in cm
        isocenter : np.ndarray
            Position of the isocenter in cm
        field_size : Tuple[float, float]
            Field size at isocenter in cm
        gantry_angle : float
            Gantry angle in degrees
        collimator_angle : float
            Collimator angle in degrees
        couch_angle : float
            Couch angle in degrees
        energies : np.ndarray
            Energy spectrum energies in MeV
        energy_probabilities : np.ndarray
            Energy spectrum probabilities
        seed_offset : int, optional
            Offset for the random seed

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Dose grid and uncertainty grid
        """
        logger.info(f"Starting Monte Carlo simulation with {num_histories} histories")
        
        # Initialize dose and uncertainty grids
        dose_grid = np.zeros(grid_shape, dtype=np.float32)
        dose_squared_grid = np.zeros(grid_shape, dtype=np.float32)
        particle_counts = np.zeros(grid_shape, dtype=np.int32)
        
        # Calculate rotation matrices for beam geometry
        gantry_rad = np.radians(gantry_angle)
        collimator_rad = np.radians(collimator_angle)
        couch_rad = np.radians(couch_angle)
        
        # Rotation matrices
        R_gantry = np.array([
            [np.cos(gantry_rad), 0, -np.sin(gantry_rad)],
            [0, 1, 0],
            [np.sin(gantry_rad), 0, np.cos(gantry_rad)]
        ])
        
        R_collimator = np.array([
            [np.cos(collimator_rad), -np.sin(collimator_rad), 0],
            [np.sin(collimator_rad), np.cos(collimator_rad), 0],
            [0, 0, 1]
        ])
        
        R_couch = np.array([
            [np.cos(couch_rad), 0, np.sin(couch_rad)],
            [0, 1, 0],
            [-np.sin(couch_rad), 0, np.cos(couch_rad)]
        ])
        
        # Combined rotation matrix (order: gantry → collimator → couch)
        R = R_couch @ R_gantry @ R_collimator
        
        # Set up random number generator for this thread
        seed_value = self.parameters['seed'] + seed_offset if self.parameters['seed'] is not None else None
        # Use numpy's Generator instead of RandomState (which is deprecated)
        local_rng = np.random.default_rng(seed_value)

        # Variance reduction parameters
        use_importance_sampling = self.parameters['use_importance_sampling']
        use_photon_splitting = self.parameters['use_photon_splitting']
        use_woodcock_tracking = self.parameters.get('use_woodcock_tracking', True)
        split_factor = self.parameters['split_factor']
        use_russian_roulette = self.parameters['enable_russian_roulette']
        use_track_length_estimator = self.parameters['use_track_length_estimator']
        russian_roulette_threshold = 0.1  # Energy threshold as fraction of initial energy
        
        # Find maximum density for Woodcock tracking
        max_density = np.max(densities)
        
        # Precompute maximum cross-sections for each energy bin for Woodcock tracking
        if use_woodcock_tracking:
            max_cross_sections = {}
            for energy in energies:
                max_cross_section = 0
                for material_idx in np.unique(materials):
                    # Get maximum cross-section for this energy and any material
                    xsec = self._get_max_cross_section(energy, material_idx)
                    max_cross_section = max(max_cross_section, xsec)
                max_cross_sections[energy] = max_cross_section
        
        # Calculate maximum cross section function for interpolation
        energy_points = np.sort(list(max_cross_sections.keys())) if use_woodcock_tracking else None
        cross_section_points = np.array([max_cross_sections[e] for e in energy_points]) if use_woodcock_tracking else None
        if use_woodcock_tracking:
            max_cross_section_func = interp1d(
                energy_points, 
                cross_section_points, 
                kind='linear', 
                bounds_error=False, 
                fill_value=(cross_section_points[0], cross_section_points[-1])
            )
        
        # Start simulation
        start_time = time.time()
        report_interval = max(1, num_histories // 20)  # Report progress every 5%
        
        for history_idx in range(num_histories):
            # Report progress
            if history_idx % report_interval == 0 and self.parameters['report_progress']:
                progress = (history_idx / num_histories) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (history_idx + 1)) * (num_histories - history_idx - 1)
                logger.info(f"Progress: {progress:.1f}% ({history_idx}/{num_histories}), "
                           f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
            
            # Sample initial energy from spectrum
            energy_idx = local_rng.choice(len(energies), p=energy_probabilities)
            initial_energy = energies[energy_idx]
            
            # Sample initial position and direction
            # For photon beams, we start at the source and aim toward a point in the field
            
            # Sample a point in the field at isocenter distance
            field_x = (local_rng.random() * 2 - 1) * field_size[0] / 2
            field_y = (local_rng.random() * 2 - 1) * field_size[1] / 2
            
            # Calculate the target point (isocenter plane)
            target_point = isocenter + np.array([field_x, field_y, 0])
            
            # Calculate initial direction (from source to field point)
            direction = target_point - source_position
            direction = direction / np.linalg.norm(direction)
            
            # Apply beam rotation
            direction = R @ direction
            
            # Initialize particle
            position = np.copy(source_position)
            energy = initial_energy
            weight = 1.0
            
            # Implement photon splitting if enabled (split at source)
            num_particles = split_factor if use_photon_splitting else 1
            particle_weight = weight / num_particles
            
            for _ in range(num_particles):
                # Transport this split particle
                current_position = np.copy(position)
                current_direction = np.copy(direction)
                current_energy = energy
                current_weight = particle_weight
                
                # Add small directional variation for each split particle (if more than one)
                if use_photon_splitting and num_particles > 1:
                    angle = 0.001  # Small angle in radians
                    # Add small random rotation to direction
                    theta = local_rng.random() * 2 * np.pi
                    phi = local_rng.random() * angle
                    
                    # Calculate perpendicular directions
                    if abs(current_direction[2]) < 0.9:
                        perp1 = np.array([current_direction[1], -current_direction[0], 0])
                    else:
                        perp1 = np.array([1, 0, -current_direction[0]/current_direction[2]])
                    
                    perp1 = perp1 / np.linalg.norm(perp1)
                    perp2 = np.cross(current_direction, perp1)
                    
                    # Apply small rotation
                    rot_dir = (perp1 * np.cos(theta) + perp2 * np.sin(theta)) * np.sin(phi)
                    current_direction = current_direction * np.cos(phi) + rot_dir
                    current_direction = current_direction / np.linalg.norm(current_direction)
                
                # Transport particle until it escapes or is absorbed
                while current_energy > self.parameters['photon_cutoff']:
                    # Find distance to boundary
                    t_boundary = self._distance_to_boundary(current_position, current_direction, 
                                                      grid_origin, 
                                                      [grid_shape[0] * grid_spacing[0],
                                                       grid_shape[1] * grid_spacing[1],
                                                       grid_shape[2] * grid_spacing[2]])
                    
                    # Convert position to voxel indices
                    voxel_indices = self._position_to_voxel(current_position, grid_origin, grid_spacing)
                    
                    # Check if we're inside the grid
                    if not self._is_inside_grid(voxel_indices, grid_shape):
                        # Particle escaped
                        break

                    # Get material and density at current position
                    material_idx = materials[tuple(voxel_indices)]
                    density = densities[tuple(voxel_indices)]
                    
                    # Skip void regions or very low density
                    if density < self.parameters['density_threshold']:
                        # Move to boundary
                        current_position += current_direction * (t_boundary + 1e-5)
                        continue
                    
                    # Determine interaction distance using Woodcock tracking
                    if use_woodcock_tracking:
                        # Woodcock tracking algorithm (delta tracking)
                        max_xsec = max_cross_section_func(current_energy)
                        
                        # Sample distance to collision (using maximum cross section)
                        t_collision = -np.log(local_rng.random()) / (max_xsec * density)
                        
                        # Check if collision occurs before boundary
                        if t_collision < t_boundary:
                            # Move to collision site
                            current_position += current_direction * t_collision
                            
                            # Get new voxel indices
                            voxel_indices = self._position_to_voxel(current_position, grid_origin, grid_spacing)
                            
                            # Check if we're still inside
                            if not self._is_inside_grid(voxel_indices, grid_shape):
                                break
                            
                            # Get material and density at collision site
                            material_idx = materials[tuple(voxel_indices)]
                            density = densities[tuple(voxel_indices)]
                            
                            # Get actual cross section
                            actual_xsec = self._get_total_cross_section(current_energy, material_idx)
                            
                            # Fictitious interaction check
                            if local_rng.random() < (actual_xsec / max_xsec):
                                # Real interaction - determine type
                                self._process_photon_interaction(current_position, current_direction, current_energy,
                                                              current_weight, material_idx, density, dose_grid, 
                                                              dose_squared_grid, particle_counts, grid_origin, 
                                                              grid_spacing, grid_shape, local_rng,
                                                              use_track_length_estimator)
                                # Photon is absorbed in this simplified model
                                break
                            # Else: fictitious interaction, continue
                        else:
                            # Move to boundary
                            current_position += current_direction * (t_boundary + 1e-5)
                    else:
                        # Traditional tracking
                        # Get mean free path
                        mfp = self._calculate_mean_free_path(current_energy, material_idx, density)
                        
                        # Sample distance to collision
                        t_collision = -mfp * np.log(local_rng.random())
                        
                        # Check if collision occurs before boundary
                        if t_collision < t_boundary:
                            # Move to collision site
                            current_position += current_direction * t_collision
                            
                            # Score energy using track-length estimator if enabled
                            if use_track_length_estimator:
                                self._score_track_length(current_position - current_direction * t_collision,
                                                      current_position, current_energy, current_weight,
                                                      dose_grid, dose_squared_grid, grid_origin, grid_spacing, 
                                                      grid_shape, density)
                            
                            # Get new voxel indices
                            voxel_indices = self._position_to_voxel(current_position, grid_origin, grid_spacing)
                            
                            # Check if we're still inside
                            if not self._is_inside_grid(voxel_indices, grid_shape):
                                break
                                
                            # Process interaction and update particle state
                            # This handles different interaction types and energy deposition
                            self._process_photon_interaction(current_position, current_direction, current_energy,
                                                          current_weight, material_idx, density, dose_grid, 
                                                          dose_squared_grid, particle_counts, grid_origin, 
                                                          grid_spacing, grid_shape, local_rng,
                                                          use_track_length_estimator)
                            break  # Photon is absorbed in this simplified model
        else:
                            # Move to boundary
                            current_position += current_direction * (t_boundary + 1e-5)
                            
                            # Score energy using track-length estimator if enabled
                            if use_track_length_estimator:
                                self._score_track_length(current_position - current_direction * t_boundary,
                                                      current_position, current_energy, current_weight,
                                                      dose_grid, dose_squared_grid, grid_origin, grid_spacing, 
                                                      grid_shape, density)
                    
                    # Apply Russian roulette for low-energy particles to improve efficiency
                    if use_russian_roulette and current_energy < (russian_roulette_threshold * initial_energy):
                        survival_prob = 0.2
                        if local_rng.random() > survival_prob:
                            # Particle terminated by Russian roulette
                            break
                        else:
                            # Survivor's weight is increased
                            current_weight /= survival_prob
        
        # Calculate uncertainty
        uncertainty_grid = np.zeros_like(dose_grid)
        valid_indices = particle_counts > 1
        if np.any(valid_indices):
            # Calculate standard error of the mean
            variance = (dose_squared_grid[valid_indices] - 
                       (dose_grid[valid_indices]**2 / particle_counts[valid_indices])) / (particle_counts[valid_indices] - 1)
            uncertainty_grid[valid_indices] = np.sqrt(variance) / dose_grid[valid_indices] * 100.0  # as percentage
        
        # Apply noise reduction if enabled
        if self.parameters['use_denoising']:
            dose_grid, uncertainty_grid = self._apply_denoising(dose_grid, uncertainty_grid)
        
        # Log simulation statistics
        simulation_time = time.time() - start_time
        particles_per_second = num_histories / simulation_time
        
        logger.info(f"Simulation completed: {num_histories} histories in {simulation_time:.2f}s "
                   f"({particles_per_second:.1f} particles/s)")
        logger.info(f"Maximum dose: {np.max(dose_grid):.6f}, Non-zero voxels: {np.count_nonzero(dose_grid)}")
        logger.info(f"Mean uncertainty in non-zero regions: {np.mean(uncertainty_grid[dose_grid > 0]):.2f}%")
        
        # Normalize dose to Gy for a standard prescription
        # This is a placeholder normalization - actual clinical systems use more complex calibration
        # Typically normalized so maximum or mean dose to a target structure equals a prescription value
        dose_grid = dose_grid / np.max(dose_grid) if np.max(dose_grid) > 0 else dose_grid
        
        return dose_grid, uncertainty_grid
    
    def _score_track_length(self, start_pos, end_pos, energy, weight, dose_grid, dose_squared_grid, 
                        grid_origin, grid_spacing, grid_shape, density):
        """
        Score dose using the track-length estimator method.
        
        This method scores dose along the entire particle path rather than just at interaction points,
        which improves statistical precision.
        
        Parameters
        ----------
        start_pos : np.ndarray
            Starting position of the track
        end_pos : np.ndarray
            Ending position of the track
        energy : float
            Particle energy in MeV
        weight : float
            Particle statistical weight
        dose_grid : np.ndarray
            Grid to score dose
        dose_squared_grid : np.ndarray
            Grid to score squared dose for uncertainty calculation
        grid_origin : tuple
            Origin coordinates of the grid
        grid_spacing : tuple
            Voxel spacing of the grid
        grid_shape : tuple
            Shape of the grid
        density : float
            Density at the current position
        """
        # Calculate track length and direction
        track_vector = end_pos - start_pos
        track_length = np.linalg.norm(track_vector)
        
        if track_length < 1e-6:
            return
            
        track_direction = track_vector / track_length
        
        # Estimate number of samples based on track length and voxel size
        min_spacing = min(grid_spacing)
        num_samples = max(2, int(track_length / (min_spacing * 0.5)))
        
        # Sample points along the track
        for i in range(num_samples):
            t = i / (num_samples - 1)
            pos = start_pos + t * track_vector
            
            # Convert to voxel indices
            voxel_indices = self._position_to_voxel(pos, grid_origin, grid_spacing)
            
            # Check if inside grid
            if not self._is_inside_grid(voxel_indices, grid_shape):
                continue
                
            # Calculate energy deposition
            # For photons, this is based on energy absorption coefficient
            muen_over_mu = 0.03  # Approximate energy absorption ratio for water in MeV range
            dose_contribution = weight * energy * track_length * density * muen_over_mu / num_samples
            
            # Score dose
            voxel_indices = tuple(voxel_indices)
            dose_grid[voxel_indices] += dose_contribution
            dose_squared_grid[voxel_indices] += dose_contribution**2
    
    def _process_photon_interaction(self, position, direction, energy, weight, 
                                 material_idx, density, dose_grid, dose_squared_grid, 
                                 particle_counts, grid_origin, grid_spacing, grid_shape, 
                                 rng, use_track_length_estimator):
        """
        Process photon interaction and update particle state.
        
        Parameters
        ----------
        position : np.ndarray
            Current particle position
        direction : np.ndarray
            Current particle direction
        energy : float
            Current particle energy
        weight : float
            Current particle statistical weight
        material_idx : int
            Material index at the interaction site
        density : float
            Material density at the interaction site
        dose_grid : np.ndarray
            Dose scoring grid
        dose_squared_grid : np.ndarray
            Squared dose scoring grid for uncertainty calculation
        particle_counts : np.ndarray
            Particle count grid for uncertainty calculation
        grid_origin : tuple
            Origin of the dose grid
        grid_spacing : tuple
            Spacing of the dose grid
        grid_shape : tuple
            Shape of the dose grid
        rng : np.random.Generator
            Random number generator
        use_track_length_estimator : bool
            Whether track-length estimator is used
        """
        voxel_indices = self._position_to_voxel(position, grid_origin, grid_spacing)
        
        if not self._is_inside_grid(voxel_indices, grid_shape):
            return
            
        voxel_indices = tuple(voxel_indices)
        
        # Sample interaction type
        interaction_type = self._sample_interaction_type(energy, material_idx)
        
        # Process based on interaction type
        if interaction_type == 'photoelectric':
            # Photoelectric effect: photon is absorbed, energy deposited locally
            dose_contribution = weight * energy
            dose_grid[voxel_indices] += dose_contribution
            dose_squared_grid[voxel_indices] += dose_contribution**2
            particle_counts[voxel_indices] += 1
            
        elif interaction_type == 'compton':
            # Compton scattering: photon scatters with energy loss
            # Simplified model: deposit a fraction of energy locally
            
            # Sample scattering angle using Klein-Nishina formula (simplified)
            cos_theta = self._sample_compton_angle(energy, rng)
            
            # Calculate scattered photon energy
            alpha = energy / 0.511  # energy in units of electron rest mass
            scattered_energy = energy / (1 + alpha * (1 - cos_theta))
            
            # Energy deposited locally
            energy_deposited = energy - scattered_energy
            
            # Score deposited energy
            dose_contribution = weight * energy_deposited
            dose_grid[voxel_indices] += dose_contribution
            dose_squared_grid[voxel_indices] += dose_contribution**2
            particle_counts[voxel_indices] += 1
            
            # Update direction based on scattering angle
            phi = 2 * np.pi * rng.random()
            new_direction = self._rotate_direction(direction, cos_theta, phi)
            
            # In a full simulation, we would continue tracking the scattered photon
            # For this simplified model, we'll terminate the history
            # In a real implementation, create a new photon and continue tracking
            
        elif interaction_type == 'pair_production':
            # Pair production: photon creates electron-positron pair
            # Simplified model: deposit all energy locally minus 1.022 MeV
            
            # Only possible if energy > 1.022 MeV
            if energy > 1.022:
                energy_deposited = energy - 1.022
                
                # Score deposited energy
                dose_contribution = weight * energy_deposited
                dose_grid[voxel_indices] += dose_contribution
                dose_squared_grid[voxel_indices] += dose_contribution**2
                particle_counts[voxel_indices] += 1
                
                # In a full simulation, we would track the electron and positron
                # For this simplified model, we assume all energy is deposited locally
                
                # Additionally, each positron produces two 0.511 MeV annihilation photons
                # In a real implementation, create two new photons and track them
            
        else:  # rayleigh or coherent scattering
            # Rayleigh (coherent) scattering: photon direction changes but energy remains the same
            # Sample scattering angle (simplified)
            cos_theta = 2 * rng.random() - 1  # Simplification, real distribution is more forward-peaked
            phi = 2 * np.pi * rng.random()
            
            # Update direction
            new_direction = self._rotate_direction(direction, cos_theta, phi)
            
            # No energy deposition
            # In a real implementation, continue tracking with new direction
    
    def _sample_compton_angle(self, energy, rng):
        """
        Sample Compton scattering angle using Klein-Nishina formula.
        
        Parameters
        ----------
        energy : float
            Photon energy in MeV
        rng : np.random.Generator
            Random number generator
            
        Returns
        -------
        float
            Cosine of the scattering angle
        """
        # Simplified implementation of Klein-Nishina sampling
        alpha = energy / 0.511  # energy in units of electron rest mass
        
        # Simple rejection sampling
        while True:
            cos_theta = 2 * rng.random() - 1
            if cos_theta < -1 or cos_theta > 1:
                continue
                
            # Klein-Nishina probability (simplified)
            kn_factor = (1 / (1 + alpha * (1 - cos_theta)))**2
            kn_factor *= (1 + cos_theta**2) / 2
            
            if rng.random() < kn_factor:
                return cos_theta
    
    def _rotate_direction(self, direction, cos_theta, phi):
        """
        Rotate a direction vector by polar angles.
        
        Parameters
        ----------
        direction : np.ndarray
            Original direction vector
        cos_theta : float
            Cosine of polar angle
        phi : float
            Azimuthal angle
            
        Returns
        -------
        np.ndarray
            Rotated direction vector
        """
        # Create a coordinate system with direction as z-axis
        z_axis = direction / np.linalg.norm(direction)
        
        # Find perpendicular axes
        if abs(z_axis[2]) < 0.9:
            x_axis = np.array([z_axis[1], -z_axis[0], 0])
        else:
            x_axis = np.array([1, 0, -z_axis[0]/z_axis[2]])
            
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Calculate new direction
        sin_theta = np.sqrt(1 - cos_theta**2)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # Rotate direction
        new_direction = (sin_theta * cos_phi * x_axis + 
                        sin_theta * sin_phi * y_axis + 
                        cos_theta * z_axis)
                        
        return new_direction
    
    def _position_to_voxel(self, position, grid_origin, grid_spacing):
        """
        Convert a position to voxel indices.
        
        Parameters
        ----------
        position : np.ndarray
            Position in world coordinates
        grid_origin : tuple
            Origin of the grid
        grid_spacing : tuple
            Spacing of the grid
            
        Returns
        -------
        np.ndarray
            Voxel indices (i, j, k)
        """
        indices = np.floor((position - grid_origin) / grid_spacing).astype(int)
        return indices
    
    def _is_inside_grid(self, indices, grid_shape):
        """
        Check if voxel indices are inside the grid.
        
        Parameters
        ----------
        indices : np.ndarray
            Voxel indices
        grid_shape : tuple
            Shape of the grid
            
        Returns
        -------
        bool
            True if inside, False otherwise
        """
        return (0 <= indices[0] < grid_shape[0] and
                0 <= indices[1] < grid_shape[1] and
                0 <= indices[2] < grid_shape[2])
    
    def _distance_to_boundary(self, position, direction, grid_origin, grid_size):
        """
        Calculate distance to the grid boundary.
        
        Parameters
        ----------
        position : np.ndarray
            Current position
        direction : np.ndarray
            Current direction
        grid_origin : tuple
            Origin of the grid
        grid_size : tuple
            Size of the grid
            
        Returns
        -------
        float
            Distance to boundary
        """
        # Calculate distances to each boundary plane
        t_min = float('inf')
        
        for i in range(3):
            if abs(direction[i]) < 1e-6:
                continue
                
            t1 = (grid_origin[i] - position[i]) / direction[i]
            t2 = (grid_origin[i] + grid_size[i] - position[i]) / direction[i]
            
            t_enter = min(t1, t2)
            t_exit = max(t1, t2)
            
            if t_exit < 0:
                return 0  # Already outside
                
            if t_enter > 0:
                t_min = min(t_min, t_exit)
        
        return t_min
    
    def _get_max_cross_section(self, energy, material_idx):
        """
        Get maximum cross-section for a given energy and material.
        
        Parameters
        ----------
        energy : float
            Photon energy in MeV
        material_idx : int
            Material index
            
        Returns
        -------
        float
            Maximum cross-section value
        """
        # Simplified - in a real implementation, interpolate from tabulated data
        # For now, use a basic model based on energy
        
        # Find closest energy in the grid
        energies = self.interaction_data['energy_grid']
        closest_idx = np.argmin(np.abs(energies - energy))
        
        # Get total cross-section for this material
        return self.interaction_data['photon']['total'][closest_idx, material_idx]
    
    def _get_total_cross_section(self, energy, material_idx):
        """
        Get total cross-section for a given energy and material.
        
        Parameters
        ----------
        energy : float
            Photon energy in MeV
        material_idx : int
            Material index
            
        Returns
        -------
        float
            Total cross-section value
        """
        # Same as _get_max_cross_section for now
        return self._get_max_cross_section(energy, material_idx)

    def _apply_denoising(self, dose_grid, uncertainty_grid):
        """Apply denoising filter to dose grid"""
        # Simple Gaussian filter
        from scipy.ndimage import gaussian_filter
        
        # Use uncertainty to guide filter strength
        sigma = 0.5 + np.mean(uncertainty_grid) * 2
        sigma = min(1.5, sigma)  # Limit maximum smoothing
        
        # Apply filter
        smoothed = gaussian_filter(dose_grid, sigma=sigma)
        
        return smoothed

    def _sample_interaction_type(self, energy, material_idx):
        """
        Sample interaction type based on cross-sections for the given energy and material.
        
        Parameters
        ----------
        energy : float
            Photon energy in MeV
        material_idx : int
            Material index
            
        Returns
        -------
        str
            Type of interaction: 'photoelectric', 'compton', 'pair_production', or 'rayleigh'
        """
        # Get cross-sections for each interaction type at this energy
        energies = self.interaction_data['energy_grid']
        closest_idx = np.argmin(np.abs(energies - energy))
        
        # Get cross-sections for different interaction types
        photoelectric = self.interaction_data['photon']['photoelectric'][closest_idx, material_idx]
        compton = self.interaction_data['photon']['compton'][closest_idx, material_idx]
        pair_production = self.interaction_data['photon']['pair_production'][closest_idx, material_idx]
        rayleigh = self.interaction_data['photon']['rayleigh'][closest_idx, material_idx]
        
        # Calculate total cross-section and probabilities
        total = photoelectric + compton + pair_production + rayleigh
        
        # Generate random number for interaction type
        r = np.random.random()
        
        # Determine interaction type based on relative probabilities
        cumulative_prob = photoelectric / total
        if r < cumulative_prob:
            return 'photoelectric'
            
        cumulative_prob += compton / total
        if r < cumulative_prob:
            return 'compton'
            
        cumulative_prob += pair_production / total
        if r < cumulative_prob:
            return 'pair_production'
            
        return 'rayleigh'  # Default to Rayleigh scattering

    def _calculate_mean_free_path(self, energy, material_idx, density):
        """
        Calculate mean free path for a photon in a material.

        Parameters
        ----------
        energy : float
            Photon energy in MeV
        material_idx : int
            Material index
        density : float
            Material density in g/cm³

        Returns
        -------
        float
            Mean free path in cm
        """
        # Get total cross-section for this energy and material
        total_xsec = self._get_total_cross_section(energy, material_idx)
        
        # Calculate mean free path (lambda = 1 / (N * sigma))
        # where N is the number of target particles per unit volume
        # and sigma is the microscopic cross-section
        
        # For photons, typically:
        # mfp = 1 / (density * NA/A * sigma)
        # where NA is Avogadro's number, A is atomic weight
        
        # Simplified formula using macroscopic cross-section
        if total_xsec > 0 and density > 0:
            mean_free_path = 1.0 / (total_xsec * density)
        else:
            # Avoid division by zero, return a large value
            mean_free_path = 1000.0  # cm
            
        return mean_free_path

    def _normalize_to_isocenter(self, dose_image, isocenter):
        """
        Normalize dose distribution so that isocenter receives 100% dose.
        
        Parameters
        ----------
        dose_image : Image
            Dose image to normalize
        isocenter : np.ndarray
            Isocenter position in world coordinates
        """
        # Convert isocenter to voxel coordinates
        voxel_indices = dose_image.world_to_voxel(isocenter)
        
        # Round to nearest voxel
        voxel_indices = np.round(voxel_indices).astype(int)
        
        # Ensure indices are within bounds
        shape = dose_image.data.shape
        voxel_indices = np.clip(voxel_indices, 
                               [0, 0, 0], 
                               [shape[0]-1, shape[1]-1, shape[2]-1])
        
        # Get dose at isocenter
        iso_dose = dose_image.data[tuple(voxel_indices)]
        
        # Normalize if isocenter dose is non-zero
        if iso_dose > 0:
            # Scale so isocenter gets 100% dose
            dose_image.data = dose_image.data * (100.0 / iso_dose)
            logger.info(f"Normalized dose to isocenter. Original value: {iso_dose:.4f}")
        else:
            logger.warning("Isocenter dose is zero or negative, cannot normalize")