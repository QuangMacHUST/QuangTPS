"""
Particle module for Monte Carlo simulation.

This module defines the particle types, properties, and history tracking
for Monte Carlo radiation transport simulations.
"""

import numpy as np
import enum
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import deque


class ParticleType(enum.Enum):
    """Enumeration of particle types supported by the Monte Carlo simulation."""
    PHOTON = 1
    ELECTRON = 2
    POSITRON = 3
    PROTON = 4
    NEUTRON = 5
    CARBON_ION = 6
    ALPHA = 7


class Particle:
    """
    Represents a particle in the Monte Carlo simulation.
    
    This class stores the physical properties of a particle and tracks
    its state during transport through matter.
    """
    
    def __init__(self, position: np.ndarray, direction: np.ndarray, energy: float, 
                type: ParticleType, weight: float = 1.0, history_id: int = -1):
        """
        Initialize a new particle.
        
        Args:
            position: 3D position vector [x, y, z] in mm
            direction: Normalized 3D direction vector
            energy: Kinetic energy in MeV
            type: Particle type (photon, electron, etc.)
            weight: Statistical weight for variance reduction (default: 1.0)
            history_id: ID of the primary history this particle belongs to
        """
        self.position = np.array(position, dtype=np.float32)
        self.direction = np.array(direction, dtype=np.float32)
        # Normalize direction if necessary
        dir_norm = np.linalg.norm(self.direction)
        if abs(dir_norm - 1.0) > 1e-6:
            self.direction /= dir_norm
            
        self.energy = float(energy)
        self.type = type
        self.weight = float(weight)
        self.is_alive = True
        self.history_id = history_id
        self.steps_taken = 0
        self.creation_process = None  # Process that created this particle
        self.parent_id = -1  # ID of parent particle (if any)
        
    def copy(self) -> 'Particle':
        """Create a copy of this particle."""
        return Particle(
            position=self.position.copy(),
            direction=self.direction.copy(),
            energy=self.energy,
            type=self.type,
            weight=self.weight,
            history_id=self.history_id
        )
    
    def __repr__(self) -> str:
        """String representation of the particle."""
        return (f"Particle(type={self.type.name}, "
                f"energy={self.energy:.3f} MeV, "
                f"position=[{self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f}], "
                f"weight={self.weight:.3f})")


class ParticleHistory:
    """
    Tracks the history of a primary particle and its secondaries.
    
    This class manages the transport of a primary particle and all its
    secondary particles in the correct order, as required for accurate
    Monte Carlo transport.
    """
    
    def __init__(self, primary_particle: Particle):
        """
        Initialize a new particle history.
        
        Args:
            primary_particle: The primary particle that starts the history
        """
        self.primary = primary_particle
        self.primary.history_id = id(self)
        self._secondaries = deque()  # Queue of secondary particles to be processed
        self._active_count = 0  # Number of particles currently being transported
        self._secondary_count = 0  # Total number of secondaries created
        self._interactions = []  # List of interaction events
        
    def add_secondary(self, particle: Particle) -> None:
        """
        Add a secondary particle to this history.
        
        Args:
            particle: The secondary particle to add
        """
        particle.history_id = self.primary.history_id
        particle.parent_id = id(particle)
        self._secondaries.append(particle)
        self._secondary_count += 1
        self._active_count += 1
        
    def add_interaction(self, interaction_type: str, position: np.ndarray, 
                       energy_deposited: float, particle_id: int) -> None:
        """
        Record an interaction event.
        
        Args:
            interaction_type: Type of interaction (e.g., 'compton', 'photoelectric')
            position: 3D position of the interaction
            energy_deposited: Energy deposited in the medium (MeV)
            particle_id: ID of the particle involved
        """
        self._interactions.append({
            'type': interaction_type,
            'position': position.copy(),
            'energy': energy_deposited,
            'particle_id': particle_id,
        })
        
    def get_next_secondary(self) -> Optional[Particle]:
        """
        Get the next secondary particle to transport.
        
        Returns:
            The next secondary particle, or None if there are no more
        """
        if self._secondaries:
            return self._secondaries.popleft()
        return None
    
    def has_active_secondaries(self) -> bool:
        """
        Check if there are active secondary particles.
        
        Returns:
            True if there are active secondaries, False otherwise
        """
        return len(self._secondaries) > 0
    
    def mark_secondary_complete(self) -> None:
        """Mark a secondary particle as complete."""
        self._active_count -= 1
        
    def get_interaction_count(self) -> int:
        """
        Get the number of interactions in this history.
        
        Returns:
            Number of interactions
        """
        return len(self._interactions)
    
    def get_secondary_count(self) -> int:
        """
        Get the total number of secondary particles.
        
        Returns:
            Number of secondary particles
        """
        return self._secondary_count
    
    def is_complete(self) -> bool:
        """
        Check if this history is complete.
        
        Returns:
            True if all particles have been transported, False otherwise
        """
        return not self.has_active_secondaries() and self._active_count == 0
