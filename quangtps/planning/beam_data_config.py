"""
Beam data configuration for treatment planning.

This module contains configuration settings for treatment beams including 
energy levels, PDD data, beam profiles, and output factors for different 
treatment machines.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Default beam models
DEFAULT_MACHINE = "TrueBeam"
DEFAULT_PHOTON_ENERGIES = ["6X", "10X", "15X"]
DEFAULT_ELECTRON_ENERGIES = ["6E", "9E", "12E", "15E", "18E"]

# Machine types
MACHINE_TYPES = {
    "TrueBeam": {
        "manufacturer": "Varian",
        "description": "TrueBeam Linear Accelerator",
        "energies": {
            "photon": DEFAULT_PHOTON_ENERGIES,
            "electron": DEFAULT_ELECTRON_ENERGIES
        },
        "max_field_size": 40.0,  # cm
        "min_field_size": 1.0,   # cm
        "mlc_type": "HD120",     # Multi-leaf collimator type
        "max_gantry_speed": 6.0, # deg/sec
        "max_dose_rate": 1400    # MU/min
    },
    "Halcyon": {
        "manufacturer": "Varian",
        "description": "Halcyon Linear Accelerator",
        "energies": {
            "photon": ["6X", "6X FFF"],
            "electron": []
        },
        "max_field_size": 28.0,  # cm
        "min_field_size": 1.0,   # cm
        "mlc_type": "Dual-Layer",
        "max_gantry_speed": 4.0, # deg/sec
        "max_dose_rate": 800     # MU/min
    },
    "Elekta Versa HD": {
        "manufacturer": "Elekta",
        "description": "Versa HD Linear Accelerator",
        "energies": {
            "photon": ["6X", "10X", "15X", "6X FFF", "10X FFF"],
            "electron": ["4E", "6E", "8E", "10E", "12E", "15E", "18E"]
        },
        "max_field_size": 40.0,  # cm
        "min_field_size": 1.0,   # cm
        "mlc_type": "Agility",   # Multi-leaf collimator type
        "max_gantry_speed": 6.0, # deg/sec
        "max_dose_rate": 1400    # MU/min
    }
}

# Default beam data directory
default_beam_data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "data", 
    "beam_data"
)

class BeamDataManager:
    """Manages beam data for treatment planning."""
    
    def __init__(self, data_dir=None):
        """
        Initialize the beam data manager.
        
        Args:
            data_dir (str, optional): Directory containing beam data. Defaults to None.
        """
        self.data_dir = data_dir or default_beam_data_dir
        self.machines = {}
        self.load_beam_data()
    
    def load_beam_data(self):
        """Load beam data from the data directory."""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Beam data directory not found: {self.data_dir}")
            logger.info("Using default beam data configuration")
            self.machines = MACHINE_TYPES
            return
        
        try:
            # Load machine configurations from JSON files in the beam data directory
            for machine_file in Path(self.data_dir).glob("*.json"):
                with open(machine_file, 'r') as f:
                    machine_data = json.load(f)
                    machine_name = machine_data.get("name")
                    if machine_name:
                        self.machines[machine_name] = machine_data
                        logger.info(f"Loaded beam data for machine: {machine_name}")
            
            # If no machines were loaded, use the default configurations
            if not self.machines:
                logger.warning("No beam data files found, using default configurations")
                self.machines = MACHINE_TYPES
        
        except Exception as e:
            logger.error(f"Error loading beam data: {str(e)}")
            logger.info("Using default beam data configuration")
            self.machines = MACHINE_TYPES
    
    def get_machine_names(self):
        """Get the names of all available treatment machines."""
        return list(self.machines.keys())
    
    def get_machine(self, machine_name=None):
        """
        Get machine data for a specific machine.
        
        Args:
            machine_name (str, optional): Name of the machine. Defaults to DEFAULT_MACHINE.
        
        Returns:
            dict: Machine data or None if not found.
        """
        if machine_name is None:
            machine_name = DEFAULT_MACHINE
        
        return self.machines.get(machine_name)
    
    def get_photon_energies(self, machine_name=None):
        """
        Get available photon energies for a machine.
        
        Args:
            machine_name (str, optional): Name of the machine. Defaults to DEFAULT_MACHINE.
        
        Returns:
            list: Available photon energies.
        """
        machine = self.get_machine(machine_name)
        if machine and "energies" in machine and "photon" in machine["energies"]:
            return machine["energies"]["photon"]
        return DEFAULT_PHOTON_ENERGIES
    
    def get_electron_energies(self, machine_name=None):
        """
        Get available electron energies for a machine.
        
        Args:
            machine_name (str, optional): Name of the machine. Defaults to DEFAULT_MACHINE.
        
        Returns:
            list: Available electron energies.
        """
        machine = self.get_machine(machine_name)
        if machine and "energies" in machine and "electron" in machine["energies"]:
            return machine["energies"]["electron"]
        return DEFAULT_ELECTRON_ENERGIES
    
    def get_pdd_data(self, machine_name, energy, field_size=10.0):
        """
        Get Percentage Depth Dose data for a specific machine, energy, and field size.
        
        Args:
            machine_name (str): Name of the machine.
            energy (str): Energy value (e.g., "6X").
            field_size (float, optional): Field size in cm. Defaults to 10.0.
        
        Returns:
            tuple: (depths, values) where depths and values are numpy arrays.
        """
        # TODO: Implement loading actual PDD data from files
        # For now, return a simplified model based on exponential decay
        depths = np.arange(0, 30.1, 0.1)
        
        # Different parameters based on energy
        if energy.endswith("X"):  # Photon energy
            energy_value = float(energy.rstrip("X"))
            d_max = 1.5 if energy_value < 10 else 2.5
            mu = 0.04 + 0.003 * energy_value  # Attenuation coefficient
        else:  # Electron energy
            energy_value = float(energy.rstrip("E"))
            d_max = energy_value / 3
            mu = 0.1 + 0.01 * energy_value
        
        # Simple build-up and fall-off model
        values = np.zeros_like(depths)
        buildup_mask = depths < d_max
        falloff_mask = depths >= d_max
        
        # Build-up region (approximately quadratic)
        values[buildup_mask] = (depths[buildup_mask] / d_max) ** 2 * 100
        
        # Fall-off region (exponential decay)
        values[falloff_mask] = 100 * np.exp(-mu * (depths[falloff_mask] - d_max))
        
        return depths, values
    
    def get_profile_data(self, machine_name, energy, depth=10.0, field_size=10.0):
        """
        Get beam profile data for a specific machine, energy, depth, and field size.
        
        Args:
            machine_name (str): Name of the machine.
            energy (str): Energy value (e.g., "6X").
            depth (float, optional): Depth in cm. Defaults to 10.0.
            field_size (float, optional): Field size in cm. Defaults to 10.0.
        
        Returns:
            tuple: (positions, values) where positions and values are numpy arrays.
        """
        # TODO: Implement loading actual profile data from files
        # For now, return a simplified model
        half_field = field_size / 2
        positions = np.linspace(-half_field - 5, half_field + 5, 200)
        
        # Different parameters based on energy
        if energy.endswith("X"):  # Photon energy
            energy_value = float(energy.rstrip("X"))
            penumbra = 0.3 + 0.02 * energy_value  # Penumbra width increases with energy
        else:  # Electron energy
            energy_value = float(energy.rstrip("E"))
            penumbra = 0.8 + 0.05 * energy_value  # Electrons have wider penumbra
        
        # Create a profile with flat central region and penumbra on the edges
        values = np.zeros_like(positions)
        
        # Inside field (flat region with small horn effect)
        inside_field = (positions > -half_field) & (positions < half_field)
        values[inside_field] = 100 * (1 + 0.02 * np.cos(np.pi * positions[inside_field] / half_field))
        
        # Penumbra regions (error function transition)
        for edge, sign in [(-half_field, 1), (half_field, -1)]:
            distance_from_edge = sign * (positions - edge)
            values += 50 * (1 + sign * np.tanh(2 * distance_from_edge / penumbra)) * np.exp(-distance_from_edge**2 / (2 * penumbra**2))
        
        # Normalize to 100 at center
        values = values / np.max(values) * 100
        
        return positions, values
    
    def get_output_factor(self, machine_name, energy, field_size, depth=10.0):
        """
        Get output factor for a specific machine, energy, field size, and depth.
        
        Args:
            machine_name (str): Name of the machine.
            energy (str): Energy value (e.g., "6X").
            field_size (float): Field size in cm.
            depth (float, optional): Depth in cm. Defaults to 10.0.
        
        Returns:
            float: Output factor relative to 10x10 field.
        """
        # TODO: Implement loading actual output factor data from files
        # For now, return a simplified model based on field size
        reference_field_size = 10.0
        
        if field_size < 1.0:
            return 0.5  # Small fields have lower output
        
        # Simple model: OF = a + b*log(FS) where FS is field size
        if energy.endswith("X"):  # Photon energy
            a, b = 0.87, 0.05
        else:  # Electron energy
            a, b = 0.8, 0.07
        
        output_factor = a + b * np.log10(field_size / reference_field_size + 0.1)
        return min(max(output_factor, 0.5), 1.2)  # Limit the range
    
    def save_beam_data(self, machine_name, data):
        """
        Save beam data for a specific machine.
        
        Args:
            machine_name (str): Name of the machine.
            data (dict): Machine data to save.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        
        file_path = os.path.join(self.data_dir, f"{machine_name}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved beam data for machine: {machine_name}")
            
            # Update the in-memory data
            self.machines[machine_name] = data
        except Exception as e:
            logger.error(f"Error saving beam data for machine {machine_name}: {str(e)}")

# Singleton instance
_beam_data_manager = None

def get_beam_data_manager():
    """Get the singleton instance of BeamDataManager."""
    global _beam_data_manager
    if _beam_data_manager is None:
        _beam_data_manager = BeamDataManager()
    return _beam_data_manager 