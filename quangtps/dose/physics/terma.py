#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TERMA calculation for radiotherapy dose calculation.

This module provides functions for calculating the TERMA (Total Energy Released
per unit MAss) distribution, which is the first step in convolution/superposition
dose calculation algorithms.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

def calculate_terma(ct_data: np.ndarray, 
                  density_map: np.ndarray, 
                  fluence: np.ndarray, 
                  spectrum: Dict[float, float],
                  spacing: Tuple[float, float, float]) -> np.ndarray:
    """
    Calculate the TERMA distribution.
    
    Parameters
    ----------
    ct_data : np.ndarray
        CT data in Hounsfield units
    density_map : np.ndarray
        Density map in g/cm^3
    fluence : np.ndarray
        Fluence distribution
    spectrum : Dict[float, float]
        Energy spectrum (energy -> relative intensity)
    spacing : Tuple[float, float, float]
        Grid spacing (mm)
        
    Returns
    -------
    np.ndarray
        TERMA distribution
    """
    logger.info("Calculating TERMA...")
    
    # Initialize TERMA grid
    terma = np.zeros_like(ct_data, dtype=np.float32)
    
    # Get grid dimensions
    nx, ny, nz = ct_data.shape
    
    # Convert spectrum to arrays for easier processing
    energies = np.array(list(spectrum.keys()))
    intensities = np.array(list(spectrum.values()))
    
    # Normalize spectrum
    intensities = intensities / np.sum(intensities)
    
    # Calculate the mass attenuation coefficients for each energy
    # and accumulate TERMA contributions
    for i, energy in enumerate(energies):
        intensity = intensities[i]
        
        # Skip very low intensities
        if intensity < 0.001:
            continue
        
        # Calculate mass attenuation coefficients for this energy
        mu_rho = calculate_mass_attenuation(energy, ct_data)
        
        # Calculate radiological depths
        rad_depths = calculate_radiological_depths(density_map, mu_rho, spacing)
        
        # Calculate TERMA contribution for this energy
        # TERMA = fluence * energy * (mu/rho)
        terma_contrib = fluence * energy * mu_rho * np.exp(-rad_depths) * intensity
        
        # Accumulate TERMA
        terma += terma_contrib
    
    logger.info("TERMA calculation completed.")
    return terma

def calculate_mass_attenuation(energy: float, ct_data: np.ndarray) -> np.ndarray:
    """
    Calculate mass attenuation coefficients from CT data.
    
    Parameters
    ----------
    energy : float
        Photon energy (MV)
    ct_data : np.ndarray
        CT data in Hounsfield units
        
    Returns
    -------
    np.ndarray
        Mass attenuation coefficients (cm²/g)
    """
    # Simplified model based on CT number
    # In a real implementation, this would use a more accurate model
    # based on material composition and ICRU or NIST data
    
    # Convert HU to relative electron density (approximate)
    rel_ed = 1.0 + ct_data * 0.001
    
    # Clip to physical range
    rel_ed = np.clip(rel_ed, 0.001, 3.0)
    
    # Calculate mass attenuation coefficient (simplified)
    # This is a very approximate model
    # The attenuation coefficient depends on energy and material atomic number
    if energy < 1.0:
        # Low energy: photoelectric effect dominates
        mu_rho = 0.05 * rel_ed * (1.0 / energy)**2.5
    elif energy < 5.0:
        # Medium energy: Compton effect dominates
        mu_rho = 0.03 * rel_ed * (1.0 / energy)**1.0
    else:
        # High energy: pair production increases
        mu_rho = 0.02 * rel_ed * (1.0 / energy)**0.5
    
    return mu_rho

def calculate_radiological_depths(density_map: np.ndarray, 
                                mu_rho: np.ndarray, 
                                spacing: Tuple[float, float, float]) -> np.ndarray:
    """
    Calculate radiological depths for TERMA calculation.
    
    Parameters
    ----------
    density_map : np.ndarray
        Density map in g/cm^3
    mu_rho : np.ndarray
        Mass attenuation coefficients (cm²/g)
    spacing : Tuple[float, float, float]
        Grid spacing (mm)
        
    Returns
    -------
    np.ndarray
        Radiological depths
    """
    # Get grid dimensions
    nx, ny, nz = density_map.shape
    
    # Initialize depths array
    depths = np.zeros_like(density_map)
    
    # Calculate linear attenuation coefficient
    mu = mu_rho * density_map
    
    # Calculate step size in cm (convert from mm)
    dx = spacing[0] / 10.0
    
    # Calculate depths along each ray (simplified)
    # Assuming beam direction is along the x-axis
    # A real implementation would account for beam angle and divergence
    
    # Accumulate depths
    for i in range(1, nx):
        depths[i, :, :] = depths[i-1, :, :] + mu[i-1, :, :] * dx
    
    return depths

def calculate_energy_fluence(fluence: np.ndarray, 
                           spectrum: Dict[float, float]) -> np.ndarray:
    """
    Calculate energy fluence from photon fluence and spectrum.
    
    Parameters
    ----------
    fluence : np.ndarray
        Photon fluence distribution
    spectrum : Dict[float, float]
        Energy spectrum (energy -> relative intensity)
        
    Returns
    -------
    np.ndarray
        Energy fluence distribution
    """
    # Convert spectrum to arrays for easier processing
    energies = np.array(list(spectrum.keys()))
    intensities = np.array(list(spectrum.values()))
    
    # Normalize spectrum
    intensities = intensities / np.sum(intensities)
    
    # Calculate mean energy
    mean_energy = np.sum(energies * intensities)
    
    # Calculate energy fluence
    energy_fluence = fluence * mean_energy
    
    return energy_fluence

def calculate_polyenergetic_terma(ct_data: np.ndarray, 
                                density_map: np.ndarray, 
                                fluence: np.ndarray, 
                                spectrum: Dict[float, float],
                                spacing: Tuple[float, float, float]) -> np.ndarray:
    """
    Calculate polyenergetic TERMA using spectrum splitting.
    
    This is a more efficient method for polyenergetic beams that
    groups similar energies together to reduce computation time.
    
    Parameters
    ----------
    ct_data : np.ndarray
        CT data in Hounsfield units
    density_map : np.ndarray
        Density map in g/cm^3
    fluence : np.ndarray
        Fluence distribution
    spectrum : Dict[float, float]
        Energy spectrum (energy -> relative intensity)
    spacing : Tuple[float, float, float]
        Grid spacing (mm)
        
    Returns
    -------
    np.ndarray
        TERMA distribution
    """
    logger.info("Calculating polyenergetic TERMA using spectrum splitting...")
    
    # Group spectrum into energy bins
    energy_bins = {
        "low": (0.0, 1.0),     # 0-1 MV
        "medium": (1.0, 4.0),  # 1-4 MV
        "high": (4.0, 25.0)    # 4+ MV
    }
    
    # Initialize TERMA grid
    terma = np.zeros_like(ct_data, dtype=np.float32)
    
    # Process each energy bin
    for bin_name, (e_min, e_max) in energy_bins.items():
        # Filter spectrum for this bin
        bin_spectrum = {
            energy: intensity for energy, intensity in spectrum.items()
            if e_min <= energy < e_max
        }
        
        # Skip if bin is empty
        if not bin_spectrum:
            continue
        
        # Calculate bin weight (sum of intensities)
        bin_weight = sum(bin_spectrum.values())
        
        # Skip if bin weight is negligible
        if bin_weight < 0.001:
            continue
        
        # Calculate mean energy for this bin
        energies = np.array(list(bin_spectrum.keys()))
        intensities = np.array(list(bin_spectrum.values()))
        mean_energy = np.sum(energies * intensities) / np.sum(intensities)
        
        # Calculate mass attenuation coefficients for mean energy
        mu_rho = calculate_mass_attenuation(mean_energy, ct_data)
        
        # Calculate radiological depths
        rad_depths = calculate_radiological_depths(density_map, mu_rho, spacing)
        
        # Calculate TERMA contribution for this bin
        terma_contrib = fluence * mean_energy * mu_rho * np.exp(-rad_depths) * bin_weight
        
        # Accumulate TERMA
        terma += terma_contrib
        
        logger.info(f"Processed {bin_name} energy bin with mean energy {mean_energy:.2f} MV")
    
    logger.info("Polyenergetic TERMA calculation completed.")
    return terma

def calculate_terma_from_beam(ct_image, beam, beam_model):
    """
    Calculate TERMA from CT image, beam and beam model directly.
    
    This is an adapter function that extracts the required data from
    higher-level objects and calls the base calculate_terma function.
    
    Parameters
    ----------
    ct_image : Image
        CT image for dose calculation
    beam : Beam
        Treatment beam
    beam_model : BeamModel
        Beam model containing spectrum data
        
    Returns
    -------
    np.ndarray
        TERMA distribution
    """
    logger.info(f"Calculating TERMA for beam {beam.name}...")
    
    # Extract CT data
    ct_data = ct_image.data
    
    # Calculate density map from CT data
    density_map = (ct_data + 1000) / 1000
    density_map = np.clip(density_map, 0.001, None)  # Ensure positive values
    
    # Get spacing in mm
    spacing = ct_image.spacing
    
    # Get beam spectrum
    if hasattr(beam_model, 'get_spectrum'):
        spectrum = beam_model.get_spectrum()
    else:
        # Create a generic spectrum if not available
        energy = float(beam.energy.replace('MV', '').strip())
        spectrum = {energy: 1.0}
        logger.warning(f"Using monoenergetic spectrum for beam {beam.name} with energy {energy} MV")
    
    # Calculate primary fluence
    fluence = _calculate_primary_fluence(beam, ct_data.shape, spacing)
    
    # Call base function
    return calculate_terma(ct_data, density_map, fluence, spectrum, spacing)

def _calculate_primary_fluence(beam, shape, spacing):
    """
    Calculate primary fluence for a beam.
    
    Parameters
    ----------
    beam : Beam
        Treatment beam
    shape : tuple
        Shape of the output grid
    spacing : tuple
        Grid spacing (mm)
        
    Returns
    -------
    np.ndarray
        Primary fluence grid
    """
    # Create empty fluence grid
    fluence = np.zeros(shape, dtype=np.float32)
    
    # Get beam direction and isocenter
    direction = beam.get_direction()
    isocenter = beam.isocenter
    
    # Get field size and SAD
    field_size = beam.field_size  # cm
    sad = beam.sad if hasattr(beam, 'sad') else 1000.0  # mm
    
    # Create a simplified beam's eye view fluence at isocenter
    i_iso, j_iso, k_iso = isocenter[0], isocenter[1], isocenter[2]
    
    # Convert to voxel coordinates
    i_vox = int(i_iso / spacing[0])
    j_vox = int(j_iso / spacing[1])
    k_vox = int(k_iso / spacing[2])
    
    # Field size in voxels at isocenter
    field_i = int(field_size[0] * 10 / spacing[0])  # convert cm to mm
    field_j = int(field_size[1] * 10 / spacing[1])
    
    # Apply a uniform fluence in the field area, with inverse square law
    nx, ny, nz = shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Convert to real-world coordinates
                x = i * spacing[0]
                y = j * spacing[1]
                z = k * spacing[2]
                
                # Calculate distance from source
                source_pos = [isocenter[0] - direction[0] * sad, 
                             isocenter[1] - direction[1] * sad,
                             isocenter[2] - direction[2] * sad]
                
                dx = x - source_pos[0]
                dy = y - source_pos[1]
                dz = z - source_pos[2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Skip if behind source
                if np.dot([dx, dy, dz], direction) <= 0:
                    continue
                
                # Project to beam's eye view at isocenter
                # This is a simplified calculation
                t = (isocenter[0] - source_pos[0]) / dx if dx != 0 else 0
                bev_y = source_pos[1] + t * dy
                bev_z = source_pos[2] + t * dz
                
                # Check if within field
                if (abs(bev_y - isocenter[1]) <= field_size[0]/2*10 and
                    abs(bev_z - isocenter[2]) <= field_size[1]/2*10):
                    # Apply inverse square law
                    fluence[i, j, k] = (sad / dist) ** 2
    
    return fluence
