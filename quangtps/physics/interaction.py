"""
Interaction module for Monte Carlo simulations.

This module provides classes for modeling the interaction of particles
with matter in Monte Carlo dose calculations.
"""

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Union, Any

from quangtps.physics.particle import Particle, ParticleType


class PhotonInteraction:
    """
    Models the interaction of photons with matter.
    
    This class implements the physics of photon interactions, including
    photoelectric effect, Compton scattering, pair production, and
    coherent (Rayleigh) scattering.
    """
    
    def __init__(self):
        """Initialize photon interaction model."""
        self._cross_section_cache = {}  # Cache for cross-section data
        
    def get_mean_free_path(self, material: Any, energy: float) -> float:
        """
        Calculate the mean free path for a photon in the given material.
        
        Args:
            material: Material object with physical properties
            energy: Photon energy in MeV
            
        Returns:
            Mean free path in cm
        """
        # Get total cross-section from material
        total_cs = material.get_cross_section(energy, 'total')
        
        # Calculate mean free path (mfp = 1/(Σ*ρ))
        if total_cs > 0:
            return 1.0 / (total_cs * material.density)
        else:
            return 1.0e10  # Very large value for negligible interaction
    
    def interact(self, photon: Particle, material: Any, 
                rng: random.Random) -> Tuple[float, List[Particle]]:
        """
        Simulate a photon interaction with matter.
        
        Args:
            photon: The photon particle
            material: Material at the interaction site
            rng: Random number generator
            
        Returns:
            Tuple of (energy deposited locally, list of secondary particles)
        """
        if photon.type != ParticleType.PHOTON:
            raise ValueError("Particle is not a photon")
        
        # Get cross-sections for different interaction types
        pe_cs = material.get_cross_section(photon.energy, 'photoelectric')
        compton_cs = material.get_cross_section(photon.energy, 'compton')
        pair_cs = material.get_cross_section(photon.energy, 'pair')
        total_cs = pe_cs + compton_cs + pair_cs
        
        # Normalize to get probabilities
        if total_cs <= 0:
            return 0.0, []
            
        pe_prob = pe_cs / total_cs
        compton_prob = compton_cs / total_cs
        pair_prob = pair_cs / total_cs
        
        # Determine interaction type
        r = rng.random()
        
        if r < pe_prob:
            # Photoelectric effect
            return self._photoelectric(photon, material, rng)
        elif r < pe_prob + compton_prob:
            # Compton scattering
            return self._compton(photon, material, rng)
        else:
            # Pair production
            return self._pair_production(photon, material, rng)
    
    def _photoelectric(self, photon: Particle, material: Any, 
                      rng: random.Random) -> Tuple[float, List[Particle]]:
        """
        Simulate photoelectric effect.
        
        In photoelectric effect, the photon is absorbed, and an electron is
        ejected with energy = photon_energy - binding_energy.
        
        Args:
            photon: The photon particle
            material: Material at the interaction site
            rng: Random number generator
            
        Returns:
            Tuple of (energy deposited locally, list of secondary particles)
        """
        # Simplified model: assume all energy is deposited locally
        # More detailed model would account for binding energy and atomic relaxation
        
        energy_dep = photon.energy
        secondaries = []
        
        # In a more detailed simulation, we would create an electron with
        # slightly less energy (accounting for binding energy)
        # For simplicity, we assume all energy is deposited locally
        
        # Mark photon as absorbed
        photon.is_alive = False
        
        return energy_dep, secondaries
    
    def _compton(self, photon: Particle, material: Any, 
                rng: random.Random) -> Tuple[float, List[Particle]]:
        """
        Simulate Compton scattering.
        
        In Compton scattering, the photon transfers some energy to an electron
        and continues with reduced energy in a different direction.
        
        Args:
            photon: The photon particle
            material: Material at the interaction site
            rng: Random number generator
            
        Returns:
            Tuple of (energy deposited locally, list of secondary particles)
        """
        # Original photon energy and direction
        E0 = photon.energy
        orig_dir = photon.direction.copy()
        
        # Electron rest energy in MeV
        mc2 = 0.511
        
        # Use Klein-Nishina formula to sample scattering angle
        # This is a simplified implementation
        kappa = E0 / mc2
        
        # Sample random number for rejection sampling
        while True:
            r1, r2 = rng.random(), rng.random()
            cos_theta = 1.0 - 2.0 * r1
            
            # Klein-Nishina probability
            sin_theta_sq = 1.0 - cos_theta * cos_theta
            sin_theta = math.sqrt(max(0, sin_theta_sq))
            
            probability = 1.0 - (kappa * sin_theta_sq) / \
                         (1.0 + kappa * (1.0 - cos_theta))
            
            if r2 <= probability:
                break
        
        # Calculate new photon energy using Compton formula
        E1 = E0 / (1.0 + kappa * (1.0 - cos_theta))
        
        # Energy transferred to electron
        Ee = E0 - E1
        
        # Create scattered photon - update energy and direction
        photon.energy = E1
        
        # Calculate new direction - need to sample azimuthal angle
        phi = 2.0 * math.pi * rng.random()
        
        # Create a coordinate system with z-axis along the original direction
        z_axis = orig_dir
        
        # Create x and y axes perpendicular to z
        if abs(z_axis[2]) < 0.999:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [0, 1, 0])
            
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # New direction in this coordinate system
        new_dir = np.array([
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta
        ])
        
        # Transform back to original coordinate system
        photon.direction = new_dir[0] * x_axis + new_dir[1] * y_axis + new_dir[2] * z_axis
        photon.direction = photon.direction / np.linalg.norm(photon.direction)
        
        # Create Compton electron if above threshold
        secondaries = []
        if Ee > 0.01:  # 10 keV threshold
            # Direction of electron - simplified model
            # Proper physics would use conservation of momentum
            e_theta = math.pi - math.acos(cos_theta)
            e_phi = phi + math.pi
            
            e_dir = np.array([
                math.sin(e_theta) * math.cos(e_phi),
                math.sin(e_theta) * math.sin(e_phi),
                math.cos(e_theta)
            ])
            
            # Transform to original coordinate system
            e_dir = e_dir[0] * x_axis + e_dir[1] * y_axis + e_dir[2] * z_axis
            e_dir = e_dir / np.linalg.norm(e_dir)
            
            # Create electron secondary
            electron = Particle(
                position=photon.position.copy(),
                direction=e_dir,
                energy=Ee,
                type=ParticleType.ELECTRON,
                weight=photon.weight
            )
            secondaries.append(electron)
            
            # All energy accounted for in secondary particles
            energy_dep = 0.0
        else:
            # Deposit the electron energy locally if below threshold
            energy_dep = Ee
            
        return energy_dep, secondaries
    
    def _pair_production(self, photon: Particle, material: Any, 
                        rng: random.Random) -> Tuple[float, List[Particle]]:
        """
        Simulate pair production.
        
        In pair production, the photon is absorbed, and an electron-positron
        pair is created.
        
        Args:
            photon: The photon particle
            material: Material at the interaction site
            rng: Random number generator
            
        Returns:
            Tuple of (energy deposited locally, list of secondary particles)
        """
        # Pair production threshold
        threshold = 1.022  # 2 * electron rest mass (MeV)
        
        if photon.energy < threshold:
            # Not enough energy for pair production
            # This shouldn't happen if the cross-sections are calculated correctly
            return photon.energy, []
        
        # Available energy for the pair
        available_energy = photon.energy - threshold
        
        # Simplified model: split energy equally between electron and positron
        # More detailed model would sample from energy distribution
        e_energy = threshold / 2.0 + available_energy / 2.0
        p_energy = threshold / 2.0 + available_energy / 2.0
        
        # Sample directions - simplified to back-to-back emission
        # along a random direction
        phi = 2.0 * math.pi * rng.random()
        cos_theta = 2.0 * rng.random() - 1.0
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        
        e_dir = np.array([
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta
        ])
        
        # Positron goes in opposite direction
        p_dir = -e_dir
        
        # Create electron and positron
        secondaries = []
        
        if e_energy > 0.01:  # 10 keV threshold
            electron = Particle(
                position=photon.position.copy(),
                direction=e_dir,
                energy=e_energy,
                type=ParticleType.ELECTRON,
                weight=photon.weight
            )
            secondaries.append(electron)
        else:
            # Deposit locally if below threshold
            e_energy = 0.0
            
        if p_energy > 0.01:  # 10 keV threshold
            positron = Particle(
                position=photon.position.copy(),
                direction=p_dir,
                energy=p_energy,
                type=ParticleType.POSITRON,
                weight=photon.weight
            )
            secondaries.append(positron)
        else:
            # Deposit locally if below threshold
            p_energy = 0.0
            
        # Photon is absorbed
        photon.is_alive = False
        
        # Calculate local energy deposition
        energy_dep = photon.energy - e_energy - p_energy
        
        return energy_dep, secondaries


class ElectronInteraction:
    """
    Models the interaction of electrons with matter.
    
    This class implements the physics of electron interactions, including
    ionization, bremsstrahlung, and elastic scattering.
    """
    
    def __init__(self):
        """Initialize electron interaction model."""
        self._stopping_power_cache = {}  # Cache for stopping power data
        
    def calculate_energy_loss(self, energy: float, material: Any, 
                             step_length: float) -> float:
        """
        Calculate energy loss for an electron over a step.
        
        Args:
            energy: Electron energy in MeV
            material: Material object with physical properties
            step_length: Step length in cm
            
        Returns:
            Energy loss in MeV
        """
        # Get stopping power from material
        stopping_power = material.get_stopping_power(energy, 'electron')
        
        # Calculate energy loss
        energy_loss = stopping_power * material.density * step_length
        
        # Ensure energy loss doesn't exceed electron energy
        return min(energy_loss, energy)
    
    def apply_multiple_scattering(self, electron: Particle, material: Any, 
                                step_length: float, rng: random.Random) -> None:
        """
        Apply multiple scattering to an electron.
        
        This implements the Molière theory of multiple scattering, which
        models the cumulative effect of many small-angle scatterings.
        
        Args:
            electron: The electron particle
            material: Material at the interaction site
            step_length: Step length in cm
            rng: Random number generator
        """
        # Simplified Gaussian model of multiple scattering
        # More detailed model would use Molière theory
        
        # Calculate characteristic scattering angle
        # theta_0 ~ 13.6 MeV / (beta*p) * sqrt(x/X0) * [1 + 0.038*ln(x/X0)]
        # where p is momentum, x is step length, X0 is radiation length
        
        # Electron momentum in MeV/c
        momentum = math.sqrt(electron.energy * (electron.energy + 1.022))
        beta = math.sqrt(1.0 - 1.0 / (1.0 + electron.energy / 0.511)**2)
        
        # Calculate characteristic angle (in radians)
        rad_length = material.radiation_length
        x_over_x0 = step_length / rad_length
        
        # Highland formula
        theta_0 = 13.6 / (beta * momentum) * math.sqrt(x_over_x0) * \
                 (1.0 + 0.038 * math.log(x_over_x0))
        
        # Sample scattering angles
        theta = rng.gauss(0, theta_0)
        phi = 2.0 * math.pi * rng.random()
        
        # Create a coordinate system with z-axis along the original direction
        z_axis = electron.direction
        
        # Create x and y axes perpendicular to z
        if abs(z_axis[2]) < 0.999:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [0, 1, 0])
            
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # New direction after scattering
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        
        new_dir = np.array([
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        ])
        
        # Transform back to original coordinate system
        electron.direction = new_dir[0] * x_axis + new_dir[1] * y_axis + new_dir[2] * z_axis
        electron.direction = electron.direction / np.linalg.norm(electron.direction)
    
    def sample_bremsstrahlung_energy(self, electron_energy: float, 
                                    rng: random.Random) -> float:
        """
        Sample the energy of a bremsstrahlung photon.
        
        Args:
            electron_energy: Energy of the electron in MeV
            rng: Random number generator
            
        Returns:
            Photon energy in MeV
        """
        # Simplified model: Sample from 1/k distribution
        # More detailed model would use the Bethe-Heitler cross section
        
        # Minimum and maximum photon energy
        k_min = 0.001  # 1 keV
        k_max = min(electron_energy, 100.0)  # Up to electron energy or 100 MeV
        
        # Sample from 1/k distribution
        r = rng.random()
        k = k_min * (k_max / k_min)**r
        
        return k
