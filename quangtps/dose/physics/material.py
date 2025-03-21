"""
Material properties module for Monte Carlo simulations.

This module provides classes and functions for representing material
properties and creating material maps from CT images for use in
Monte Carlo dose calculations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import SimpleITK as sitk
import logging

logger = logging.getLogger(__name__)


class Material:
    """
    Represents the physical properties of a material.
    
    This class stores the atomic composition, density, and other physical
    properties of a material that are relevant for radiation transport.
    """
    
    def __init__(self, name: str, density: float, electron_density: float = None, 
                effective_z: float = None, composition: Dict[str, float] = None):
        """
        Initialize a new material.
        
        Args:
            name: Material name (e.g., 'Water', 'Bone')
            density: Physical density in g/cm³
            electron_density: Electron density relative to water (optional)
            effective_z: Effective atomic number (optional)
            composition: Elemental composition as dict of element -> fraction by weight
        """
        self.name = name
        self.density = density
        self.electron_density = electron_density or 1.0
        self.effective_z = effective_z or 7.42  # Default to water if not specified
        self.composition = composition or {'H': 0.112, 'O': 0.888}  # Default to water
        
        # Radiation interaction properties (calculated or loaded from tables)
        self.radiation_length = self._calculate_radiation_length()
        self.stopping_powers = {}  # Energy -> stopping power (MeV·cm²/g)
        self.cross_sections = {}  # Energy -> cross-section (cm²/g)
        
    def _calculate_radiation_length(self) -> float:
        """
        Calculate radiation length based on composition.
        
        Returns:
            Radiation length in g/cm²
        """
        # Simplified calculation
        # Real implementation would use proper formulas based on composition
        if self.effective_z < 6:
            return 40.0  # Air-like
        elif self.effective_z < 8:
            return 36.1  # Water-like
        elif self.effective_z < 10:
            return 30.0  # Soft tissue-like
        elif self.effective_z < 14:
            return 24.0  # Bone-like
        else:
            return 12.0  # Metal-like
    
    def get_stopping_power(self, energy: float, particle_type: str = 'electron') -> float:
        """
        Get the mass stopping power for a particle at a given energy.
        
        Args:
            energy: Particle energy in MeV
            particle_type: Type of particle ('electron', 'photon', 'proton', etc.)
            
        Returns:
            Mass stopping power in MeV·cm²/g
        """
        # Use pre-calculated tables or compute with appropriate formulas
        # Simplified implementation for electrons using Bethe formula
        if particle_type.lower() == 'electron':
            if energy in self.stopping_powers:
                return self.stopping_powers[energy]
            
            # Simplified Bethe formula for electrons
            # More accurate implementation would use the full formula
            beta2 = 1.0 - 1.0 / (1.0 + energy / 0.511)**2  # Relativistic beta²
            
            # Simplified stopping power calculation
            stopping_power = 0.3071 * self.electron_density * self.effective_z / beta2 * \
                             (np.log(energy / (1 - beta2) / 0.0000115) - beta2)
            
            self.stopping_powers[energy] = stopping_power
            return stopping_power
        else:
            logger.warning(f"Stopping power for particle type {particle_type} not implemented")
            return 0.0
    
    def get_cross_section(self, energy: float, interaction_type: str = 'total') -> float:
        """
        Get the mass attenuation coefficient for a photon at a given energy.
        
        Args:
            energy: Photon energy in MeV
            interaction_type: Type of interaction ('total', 'photoelectric', 'compton', 'pair')
            
        Returns:
            Mass attenuation coefficient in cm²/g
        """
        key = f"{energy}_{interaction_type}"
        if key in self.cross_sections:
            return self.cross_sections[key]
        
        # Simplified implementation
        # For a real implementation, data would be loaded from NIST or similar sources
        if interaction_type.lower() == 'total':
            # Simplified total cross section model
            # In reality, this should use tabulated data or more complex formulas
            if energy < 0.1:
                # Low energy region where photoelectric effect dominates
                cross_section = 5.0 * self.effective_z**3 * energy**(-3.5)
            elif energy < 10.0:
                # Mid energy region where Compton scattering dominates
                cross_section = 0.5 * self.electron_density * (1.0 / energy)**0.5
            else:
                # High energy region where pair production becomes important
                cross_section = 0.05 * self.effective_z * np.log(energy)
        
        elif interaction_type.lower() == 'photoelectric':
            # Simplified photoelectric cross section
            if energy < 0.2:
                cross_section = 4.0 * self.effective_z**3 * energy**(-3.5)
            else:
                cross_section = 4.0 * self.effective_z**3 * energy**(-3.5) * \
                               (0.2 / energy)**0.5
        
        elif interaction_type.lower() == 'compton':
            # Simplified Compton cross section (Klein-Nishina)
            # Proportional to electron density
            cross_section = 0.5 * self.electron_density * (1.0 / energy)**0.5
        
        elif interaction_type.lower() == 'pair':
            # Simplified pair production cross section
            # Only happens above 1.022 MeV threshold
            if energy <= 1.022:
                cross_section = 0.0
            else:
                cross_section = 0.05 * self.effective_z * np.log(energy)
        
        else:
            logger.warning(f"Cross section for interaction type {interaction_type} not implemented")
            cross_section = 0.0
        
        self.cross_sections[key] = cross_section
        return cross_section
    
    def __repr__(self) -> str:
        """String representation of the material."""
        return f"Material(name='{self.name}', density={self.density:.3f} g/cm³, Z_eff={self.effective_z:.2f})"


class MaterialLibrary:
    """
    Library of materials for radiation transport simulations.
    
    This class provides a collection of predefined materials and
    methods for accessing them by name or material index.
    """
    
    def __init__(self):
        """Initialize the material library with a set of default materials."""
        self.materials_by_name = {}
        self.materials_by_index = {}
        self._initialize_default_materials()
    
    def _initialize_default_materials(self):
        """Initialize the default material library."""
        # Add standard materials
        # Air
        self.add_material(Material(
            name="air",
            density=0.001205,
            electron_density=0.001,
            effective_z=7.78,
            composition={'N': 0.755, 'O': 0.232, 'Ar': 0.013}
        ), index=0)
        
        # Water
        self.add_material(Material(
            name="water",
            density=1.0,
            electron_density=1.0,
            effective_z=7.42,
            composition={'H': 0.112, 'O': 0.888}
        ), index=1)
        
        # Soft Tissue
        self.add_material(Material(
            name="soft_tissue",
            density=1.05,
            electron_density=1.03,
            effective_z=7.4,
            composition={'H': 0.102, 'C': 0.143, 'N': 0.034, 'O': 0.708, 'Na': 0.002, 'P': 0.003, 'S': 0.003, 'Cl': 0.002, 'K': 0.003}
        ), index=2)
        
        # Muscle
        self.add_material(Material(
            name="muscle",
            density=1.05,
            electron_density=1.04,
            effective_z=7.64,
            composition={'H': 0.102, 'C': 0.143, 'N': 0.034, 'O': 0.71, 'Na': 0.001, 'P': 0.002, 'S': 0.003, 'Cl': 0.001, 'K': 0.004}
        ), index=3)
        
        # Adipose Tissue
        self.add_material(Material(
            name="adipose",
            density=0.92,
            electron_density=0.95,
            effective_z=6.33,
            composition={'H': 0.12, 'C': 0.64, 'N': 0.008, 'O': 0.232}
        ), index=4)
        
        # Lung Tissue
        self.add_material(Material(
            name="lung",
            density=0.25,
            electron_density=0.258,
            effective_z=7.4,
            composition={'H': 0.103, 'C': 0.105, 'N': 0.031, 'O': 0.749, 'Na': 0.002, 'P': 0.002, 'S': 0.003, 'Cl': 0.002, 'K': 0.003}
        ), index=5)
        
        # Bone (Cortical)
        self.add_material(Material(
            name="bone",
            density=1.85,
            electron_density=1.7,
            effective_z=13.8,
            composition={'H': 0.034, 'C': 0.155, 'N': 0.042, 'O': 0.435, 'P': 0.103, 'Ca': 0.225, 'Mg': 0.002, 'S': 0.003, 'Na': 0.001}
        ), index=6)
        
        # Titanium
        self.add_material(Material(
            name="titanium",
            density=4.54,
            electron_density=3.7,
            effective_z=22.0,
            composition={'Ti': 1.0}
        ), index=7)
    
    def add_material(self, material: Material, index: Optional[int] = None) -> int:
        """
        Add a material to the library.
        
        Parameters
        ----------
        material : Material
            The material to add
        index : Optional[int]
            Material index, auto-assigned if None
            
        Returns
        -------
        int
            The assigned material index
        """
        # Ensure material name is lowercase for case-insensitive lookup
        name = material.name.lower()
        
        # Add to name-based dictionary
        self.materials_by_name[name] = material
        
        # Assign index if not provided
        if index is None:
            index = len(self.materials_by_index)
        
        # Add to index-based dictionary
        self.materials_by_index[index] = material
        
        return index
    
    def get_material_by_name(self, name: str) -> Material:
        """
        Get a material by name.
        
        Parameters
        ----------
        name : str
            Material name (case-insensitive)
            
        Returns
        -------
        Material
            The requested material, or water if not found
        """
        name = name.lower()
        if name in self.materials_by_name:
            return self.materials_by_name[name]
        else:
            logger.warning(f"Material '{name}' not found, returning water")
            return self.materials_by_name["water"]
    
    def get_material_by_index(self, index: int) -> Material:
        """
        Get a material by index.
        
        Parameters
        ----------
        index : int
            Material index
            
        Returns
        -------
        Material
            The requested material, or water if not found
        """
        if index in self.materials_by_index:
            return self.materials_by_index[index]
        else:
            logger.warning(f"Material index {index} not found, returning water")
            return self.materials_by_name["water"]
    
    def create_material_map_from_ct(self, ct_image: np.ndarray) -> np.ndarray:
        """
        Create a material index map from a CT image.
        
        Parameters
        ----------
        ct_image : np.ndarray
            CT image in Hounsfield Units
            
        Returns
        -------
        np.ndarray
            Material index map with the same shape as the input
        """
        # Initialize material property converter
        props = MaterialProperties()
        
        # Create output array
        material_map = np.zeros_like(ct_image, dtype=np.int8)
        
        # Convert HU values to material indices
        for idx, hu_value in np.ndenumerate(ct_image):
            material_map[idx] = props.hu_to_material(hu_value)
        
        return material_map
    
    def list_materials(self) -> List[str]:
        """
        Get a list of all available material names.
        
        Returns
        -------
        List[str]
            List of material names
        """
        return list(self.materials_by_name.keys())
    
    def __len__(self) -> int:
        """Number of materials in the library."""
        return len(self.materials_by_name)


class MaterialProperties:
    """
    Manager for material properties and conversion between HU and materials.
    
    This class provides methods for converting Hounsfield Units (HU) to
    material properties and manages a database of predefined materials.
    """
    
    def __init__(self):
        """Initialize the material properties database."""
        self.materials = {}
        self.hu_to_material_map = {}
        self._initialize_default_materials()
        self._initialize_hu_conversion()
    
    def _initialize_default_materials(self) -> None:
        """Initialize the default materials database."""
        # Air
        self.materials[0] = Material(
            name="Air",
            density=0.001205,
            electron_density=0.001,
            effective_z=7.78,
            composition={'N': 0.755, 'O': 0.232, 'Ar': 0.013}
        )
        
        # Water
        self.materials[1] = Material(
            name="Water",
            density=1.0,
            electron_density=1.0,
            effective_z=7.42,
            composition={'H': 0.112, 'O': 0.888}
        )
        
        # Soft Tissue
        self.materials[2] = Material(
            name="Soft Tissue",
            density=1.05,
            electron_density=1.03,
            effective_z=7.4,
            composition={'H': 0.102, 'C': 0.143, 'N': 0.034, 'O': 0.708, 'Na': 0.002, 'P': 0.003, 'S': 0.003, 'Cl': 0.002, 'K': 0.003}
        )
        
        # Muscle
        self.materials[3] = Material(
            name="Muscle",
            density=1.05,
            electron_density=1.04,
            effective_z=7.64,
            composition={'H': 0.102, 'C': 0.143, 'N': 0.034, 'O': 0.71, 'Na': 0.001, 'P': 0.002, 'S': 0.003, 'Cl': 0.001, 'K': 0.004}
        )
        
        # Adipose Tissue
        self.materials[4] = Material(
            name="Adipose",
            density=0.92,
            electron_density=0.95,
            effective_z=6.33,
            composition={'H': 0.12, 'C': 0.64, 'N': 0.008, 'O': 0.232}
        )
        
        # Lung Tissue
        self.materials[5] = Material(
            name="Lung",
            density=0.25,
            electron_density=0.258,
            effective_z=7.4,
            composition={'H': 0.103, 'C': 0.105, 'N': 0.031, 'O': 0.749, 'Na': 0.002, 'P': 0.002, 'S': 0.003, 'Cl': 0.002, 'K': 0.003}
        )
        
        # Bone (Cortical)
        self.materials[6] = Material(
            name="Bone",
            density=1.85,
            electron_density=1.7,
            effective_z=13.8,
            composition={'H': 0.034, 'C': 0.155, 'N': 0.042, 'O': 0.435, 'P': 0.103, 'Ca': 0.225, 'Mg': 0.002, 'S': 0.003, 'Na': 0.001}
        )
        
        # Metal (Titanium)
        self.materials[7] = Material(
            name="Titanium",
            density=4.54,
            electron_density=3.7,
            effective_z=22.0,
            composition={'Ti': 1.0}
        )
    
    def _initialize_hu_conversion(self) -> None:
        """Initialize the HU to material conversion map."""
        # Standard HU ranges for different tissues
        # These would be calibrated based on the specific CT scanner
        self.hu_ranges = [
            (-1000, -950, 0),    # Air
            (-950, -200, 5),     # Lung
            (-200, -5, 4),       # Adipose
            (-5, 5, 1),          # Water
            (5, 40, 2),          # Soft Tissue
            (40, 200, 3),        # Muscle
            (200, 3000, 6),      # Bone
            (3000, 30000, 7)     # Metal
        ]
    
    def get_material(self, material_id: int) -> Material:
        """
        Get a material by ID.
        
        Args:
            material_id: Material identifier
            
        Returns:
            Material object
        """
        if material_id in self.materials:
            return self.materials[material_id]
        else:
            logger.warning(f"Material ID {material_id} not found. Using water as default.")
            return self.materials[1]  # Default to water
    
    def hu_to_material(self, hu: float) -> int:
        """
        Convert Hounsfield Units to a material ID.
        
        Args:
            hu: Hounsfield Units value
            
        Returns:
            Material identifier
        """
        for low, high, material_id in self.hu_ranges:
            if low <= hu < high:
                return material_id
        
        # Default to highest material (metal) for very high HU values
        return self.hu_ranges[-1][2]
    
    def hu_to_density(self, hu: float) -> float:
        """
        Convert Hounsfield Units to physical density (g/cm³).
        
        Args:
            hu: Hounsfield Units value
            
        Returns:
            Physical density in g/cm³
        """
        # Standard conversion formula
        # Different for HU < 0 (air, lung) and HU >= 0 (water, soft tissue, bone)
        if hu < 0:
            return 1.0 + hu / 1000.0
        else:
            return 1.0 + hu / 1950.0
    
    def add_material(self, material: Material, material_id: Optional[int] = None) -> int:
        """
        Add a new material to the database.
        
        Args:
            material: Material object
            material_id: Optional material ID (auto-generated if None)
            
        Returns:
            Material ID
        """
        if material_id is None:
            material_id = max(self.materials.keys()) + 1
            
        self.materials[material_id] = material
        return material_id


def create_material_map_from_ct(ct_array: np.ndarray) -> np.ndarray:
    """
    Create a material map from a CT image.
    
    This function converts a CT image in Hounsfield Units to a material index
    array for use in Monte Carlo simulations.
    
    Args:
        ct_array: CT image array in Hounsfield Units
        
    Returns:
        Array with material indices
    """
    material_props = MaterialProperties()
    
    # Initialize material map with same shape as CT array
    material_map = np.zeros_like(ct_array, dtype=np.int8)
    
    # Convert HU values to material indices
    for idx, hu_value in np.ndenumerate(ct_array):
        material_map[idx] = material_props.hu_to_material(hu_value)
    
    return material_map
