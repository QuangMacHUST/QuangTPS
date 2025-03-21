"""
Source module for Monte Carlo simulations.

This module provides classes for creating and sampling radiation sources
for Monte Carlo dose calculations, including photon and electron sources.
"""

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Union, Any

from .particle import Particle, ParticleType
from quangtps.core.types import BeamParameters


class RadiationSource:
    """
    Base class for radiation sources in Monte Carlo simulations.
    
    A radiation source defines the initial state of primary particles
    (position, direction, energy) entering the patient.
    """
    
    def __init__(self, beam_parameters: BeamParameters):
        """
        Initialize a radiation source.
        
        Args:
            beam_parameters: Parameters describing the beam setup
        """
        self.beam_parameters = beam_parameters
        self.isocenter = np.array(beam_parameters.isocenter)
        self.gantry_angle = beam_parameters.gantry_angle
        self.collimator_angle = beam_parameters.collimator_angle
        self.couch_angle = beam_parameters.couch_angle
        self.sad = beam_parameters.sad or 1000.0  # Source-axis distance in mm
        self.energy_mev = beam_parameters.nominal_energy  # MeV
        
        # Set field size
        self.field_width = beam_parameters.field_size[0]  # mm
        self.field_height = beam_parameters.field_size[1]  # mm
        
        # Initialize beam direction
        self._setup_beam_geometry()
    
    def _setup_beam_geometry(self) -> None:
        """
        Set up beam geometry based on gantry, collimator, and couch angles.
        
        This sets up the coordinate system and direction vectors for the beam.
        """
        # Convert angles to radians
        gantry_rad = math.radians(self.gantry_angle)
        coll_rad = math.radians(self.collimator_angle)
        couch_rad = math.radians(self.couch_angle)
        
        # Gantry rotation (around y-axis)
        # At gantry=0, beam points in +z direction (toward gantry)
        self.beam_direction = np.array([
            math.sin(gantry_rad),
            0,
            math.cos(gantry_rad)
        ])
        
        # Determine source position (opposite of beam direction, at distance SAD)
        self.source_position = self.isocenter - self.beam_direction * self.sad
        
        # Set up beam coordinate system
        # beam_z points from source to isocenter
        self.beam_z = self.beam_direction
        
        # beam_x is perpendicular to beam_z, in the transverse plane
        # At gantry=0, collimator=0, beam_x points to patient's right
        self.beam_x = np.array([
            math.cos(gantry_rad) * math.cos(coll_rad),
            math.sin(coll_rad),
            -math.sin(gantry_rad) * math.cos(coll_rad)
        ])
        
        # Apply couch rotation
        if abs(couch_rad) > 1e-6:
            # Rotation around z-axis (perpendicular to couch)
            couch_cos = math.cos(couch_rad)
            couch_sin = math.sin(couch_rad)
            
            # Rotate beam_direction and beam_x
            self.beam_direction = np.array([
                self.beam_direction[0] * couch_cos - self.beam_direction[1] * couch_sin,
                self.beam_direction[0] * couch_sin + self.beam_direction[1] * couch_cos,
                self.beam_direction[2]
            ])
            
            self.beam_x = np.array([
                self.beam_x[0] * couch_cos - self.beam_x[1] * couch_sin,
                self.beam_x[0] * couch_sin + self.beam_x[1] * couch_cos,
                self.beam_x[2]
            ])
            
            # Recalculate source position
            self.source_position = self.isocenter - self.beam_direction * self.sad
        
        # beam_y is perpendicular to beam_z and beam_x
        self.beam_y = np.cross(self.beam_z, self.beam_x)
    
    def sample_particle(self, rng: random.Random) -> Particle:
        """
        Sample a particle from this source.
        
        Args:
            rng: Random number generator
            
        Returns:
            A new particle with position, direction, and energy
        """
        raise NotImplementedError("Subclasses must implement this method")


class PhotonSource(RadiationSource):
    """
    Photon beam source for Monte Carlo simulations.
    
    This class models a clinical photon beam from a linear accelerator,
    including beam spectrum, fluence distribution, and beam modifiers.
    """
    
    def __init__(self, beam_parameters: BeamParameters):
        """
        Initialize a photon beam source.
        
        Args:
            beam_parameters: Parameters describing the beam setup
        """
        super().__init__(beam_parameters)
        
        # Initialize photon beam specific parameters
        self.mean_energy = self.energy_mev
        self.energy_sigma = 0.03 * self.mean_energy  # 3% FWHM energy spread
        
        # Phase space data or beam model parameters
        # In a real implementation, these would be loaded from commissioned data
        self.spectrum = self._initialize_spectrum()
        self.fluence_map = None  # For IMRT/VMAT, this would be non-uniform
        
        # MLC configuration if available
        self.mlc_positions = beam_parameters.mlc_positions if hasattr(beam_parameters, 'mlc_positions') else None
    
    def _initialize_spectrum(self) -> Dict[float, float]:
        """
        Initialize the photon energy spectrum.
        
        In a real implementation, this would load spectrum data from a phase-space
        file or beam model for the specific beam energy.
        
        Returns:
            Dictionary mapping energy (MeV) to relative probability
        """
        # Simplified bremsstrahlung spectrum for clinical photon beams
        # In reality, this would be based on commissioned beam data
        spectrum = {}
        
        # Create a spectrum with 100 energy bins
        max_energy = self.energy_mev  # Max energy = nominal energy
        n_bins = 100
        
        for i in range(1, n_bins + 1):
            energy = i * max_energy / n_bins
            
            # Simplified bremsstrahlung shape
            # Higher probability at lower energies with peak around 1/3 of max energy
            if energy < 0.1:
                probability = 0  # Filter low energy photons
            else:
                # Shape with peak at ~Emax/3 and falling off toward Emax
                x = energy / max_energy
                probability = 20 * x * math.exp(-2.0 * x)
            
            spectrum[energy] = probability
        
        # Normalize to sum to 1
        total_prob = sum(spectrum.values())
        for energy in spectrum:
            spectrum[energy] /= total_prob
        
        return spectrum
    
    def sample_particle(self, rng: random.Random) -> Particle:
        """
        Sample a photon from the source.
        
        Args:
            rng: Random number generator
            
        Returns:
            A new photon particle
        """
        # Sample particle position based on field size
        # X and Y coordinates are sampled within the field size
        # Z coordinate is set at the source distance
        
        # Simplified: sample from a rectangular field (could be shaped by MLC)
        field_x = (rng.random() * 2 - 1) * self.field_width / 2
        field_y = (rng.random() * 2 - 1) * self.field_height / 2
        
        # Check if position is blocked by MLC or jaws
        if self.mlc_positions is not None:
            # In a real implementation, check if ray is blocked by MLC
            # This requires checking the projection of the ray against MLC leaf positions
            pass
        
        # Calculate position in global coordinates
        position = (self.source_position + 
                   field_x * self.beam_x + 
                   field_y * self.beam_y)
        
        # Calculate direction (from source to field point at isocenter plane)
        target_point = (self.isocenter + 
                       field_x * self.beam_x + 
                       field_y * self.beam_y)
        
        direction = target_point - position
        direction = direction / np.linalg.norm(direction)
        
        # Sample energy from spectrum
        # Simplified: use Gaussian around mean energy
        # In a real implementation, sample from the actual spectrum
        energy = rng.gauss(self.mean_energy, self.energy_sigma)
        energy = max(0.01, min(energy, self.energy_mev))  # Clamp to valid range
        
        # Create photon particle
        particle = Particle(
            position=position,
            direction=direction,
            energy=energy,
            type=ParticleType.PHOTON
        )
        
        return particle


class ElectronSource(RadiationSource):
    """
    Electron beam source for Monte Carlo simulations.
    
    This class models a clinical electron beam from a linear accelerator,
    including beam spectrum, fluence distribution, and beam modifiers.
    """
    
    def __init__(self, beam_parameters: BeamParameters):
        """
        Initialize an electron beam source.
        
        Args:
            beam_parameters: Parameters describing the beam setup
        """
        super().__init__(beam_parameters)
        
        # Initialize electron beam specific parameters
        self.mean_energy = self.energy_mev
        self.energy_sigma = 0.05 * self.mean_energy  # 5% energy spread
        
        # Electron beam angular spread parameter
        # Electrons have more lateral spread than photons
        # This is parameterized as the standard deviation of the angular distribution
        self.angular_sigma = 0.05  # radians, about 3 degrees
        
        # Applicator size
        self.applicator_size = beam_parameters.applicator_size if hasattr(beam_parameters, 'applicator_size') else self.field_width
    
    def sample_particle(self, rng: random.Random) -> Particle:
        """
        Sample an electron from the source.
        
        Args:
            rng: Random number generator
            
        Returns:
            A new electron particle
        """
        # Sample particle position based on applicator size
        # Electrons typically use applicators rather than MLCs
        
        # Sample position within circular applicator
        r = math.sqrt(rng.random()) * self.applicator_size / 2
        theta = rng.random() * 2 * math.pi
        
        field_x = r * math.cos(theta)
        field_y = r * math.sin(theta)
        
        # Calculate position in global coordinates
        position = (self.source_position + 
                   field_x * self.beam_x + 
                   field_y * self.beam_y)
        
        # Calculate direction (from source to field point at isocenter plane)
        target_point = (self.isocenter + 
                       field_x * self.beam_x + 
                       field_y * self.beam_y)
        
        base_direction = target_point - position
        base_direction = base_direction / np.linalg.norm(base_direction)
        
        # Apply angular spread to direction
        # Electrons have more spread than photons
        theta = rng.gauss(0, self.angular_sigma)
        phi = rng.random() * 2 * math.pi
        
        # Create a coordinate system with z-axis along the base direction
        z_axis = base_direction
        
        # Create x and y axes perpendicular to z
        if abs(z_axis[2]) < 0.999:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [0, 1, 0])
            
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # New direction with angular spread
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        
        direction = np.array([
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        ])
        
        # Transform back to original coordinate system
        direction = direction[0] * x_axis + direction[1] * y_axis + direction[2] * z_axis
        direction = direction / np.linalg.norm(direction)
        
        # Sample energy from spectrum (approximately Gaussian for electrons)
        energy = rng.gauss(self.mean_energy, self.energy_sigma)
        energy = max(0.1, min(energy, 1.1 * self.mean_energy))  # Clamp to valid range
        
        # Create electron particle
        particle = Particle(
            position=position,
            direction=direction,
            energy=energy,
            type=ParticleType.ELECTRON
        )
        
        return particle
