#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TERMA (Total Energy Released per unit MAss) calculation module.

This module provides functions for calculating TERMA, which represents
the energy transferred from primary photons to secondary particles per 
unit mass of medium. TERMA is a critical component in dose calculation 
algorithms like Collapsed Cone and Convolution/Superposition.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

from quangtps.core.exceptions import DoseCalculationError
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.dose.beam_data_processor import BeamModel

logger = logging.getLogger(__name__)

def calculate_terma(ct_image: Image, beam: Beam, beam_model: BeamModel) -> np.ndarray:
    """
    Calculate TERMA (Total Energy Released per unit MAss) for a beam.
    
    This function computes the total energy released per unit mass at each
    point in the CT image when irradiated by the specified beam. TERMA
    represents the energy transferred from primary photons to electrons.
    
    Parameters
    ----------
    ct_image : Image
        CT image of the patient
    beam : Beam
        Radiation beam
    beam_model : BeamModel
        Beam model containing physical parameters
        
    Returns
    -------
    np.ndarray
        3D array of TERMA values
    
    Raises
    ------
    DoseCalculationError
        If TERMA calculation fails
    """
    try:
        logger.info(f"Calculating TERMA for beam {beam.name}")
        
        # Get CT data and convert to linear attenuation coefficients
        mu = convert_ct_to_attenuation(ct_image.data, beam_model)
        
        # Get beam parameters
        gantry_angle = getattr(beam, 'gantry_angle', 0.0)
        collimator_angle = getattr(beam, 'collimator_angle', 0.0)
        couch_angle = getattr(beam, 'couch_angle', 0.0)
        field_size = getattr(beam, 'field_size', (10, 10))
        energy = getattr(beam, 'energy', '6MV')
        isocenter = getattr(beam, 'isocenter', None)
        
        # If isocenter is not provided, use the center of the CT image
        if isocenter is None:
            isocenter = get_center_coordinates(ct_image)
            logger.warning(f"Isocenter not specified, using center of CT image: {isocenter}")
        
        # Extract CT dimensions and voxel spacing
        dimensions = ct_image.data.shape
        voxel_spacing = ct_image.spacing  # in mm
        
        # Convert voxel spacing to cm for dose calculation
        voxel_spacing_cm = [s / 10.0 for s in voxel_spacing]
        
        # Calculate source position based on gantry angle and SSD
        source_position = calculate_source_position(
            isocenter, gantry_angle, couch_angle, beam_model
        )
        
        # Calculate beam direction
        beam_direction = calculate_beam_direction(isocenter, source_position)
        
        # Calculate primary fluence (includes beam profile and MLC/jaw effects)
        fluence = calculate_primary_fluence(
            dimensions, voxel_spacing_cm, source_position, 
            beam_direction, field_size, collimator_angle, beam_model
        )
        
        # Calculate radiological path length for each voxel
        path_length = calculate_radiological_path_length(
            mu, source_position, dimensions, voxel_spacing_cm
        )
        
        # Calculate TERMA using the path length, attenuation, and fluence
        terma = calculate_terma_from_path_length(
            fluence, path_length, mu, beam_model, energy
        )
        
        logger.info(f"TERMA calculation completed for beam {beam.name}")
        
        return terma
        
    except Exception as e:
        error_msg = f"Error calculating TERMA: {str(e)}"
        logger.error(error_msg)
        raise DoseCalculationError(error_msg) from e

def convert_ct_to_attenuation(ct_data: np.ndarray, beam_model: BeamModel) -> np.ndarray:
    """
    Convert CT Hounsfield Units to linear attenuation coefficients.
    
    Parameters
    ----------
    ct_data : np.ndarray
        CT data in Hounsfield Units
    beam_model : BeamModel
        Beam model with energy-dependent properties
        
    Returns
    -------
    np.ndarray
        Linear attenuation coefficients in cm^-1
    """
    # Get the energy from the beam model
    energy = beam_model.energy if hasattr(beam_model, 'energy') else '6MV'
    
    # This is a simplified conversion - a more accurate implementation would use
    # a calibration curve specific to the CT scanner and energy spectrum
    
    # Convert HU to relative electron density
    # Water is defined as HU=0 and density=1.0
    # Air is approximately HU=-1000 and density~0.001
    # Bone ranges from ~HU=300 to 1500+ with density~1.3-2.0
    
    # Linear scaling from HU to relative electron density
    electron_density = 1.0 + ct_data / 1000.0
    
    # Apply minimum threshold to avoid negative values
    electron_density = np.maximum(electron_density, 0.001)
    
    # Convert electron density to linear attenuation coefficient
    # This is a simplified approach - in reality, this depends on the energy spectrum
    
    # Default attenuation coefficients (in cm^-1) for different energies at density=1.0
    if energy == '6MV':
        mu_water = 0.05  # Approximate attenuation coefficient for water at 6MV (average energy ~2MeV)
    elif energy == '10MV':
        mu_water = 0.03  # Approximate for 10MV (average energy ~3-4MeV)
    elif energy == '15MV':
        mu_water = 0.025  # Approximate for 15MV (average energy ~5-6MeV)
    elif energy == '18MV':
        mu_water = 0.022  # Approximate for 18MV
    else:
        # Default to 6MV if unknown energy
        mu_water = 0.05
        logger.warning(f"Unknown energy {energy}, using default attenuation coefficient for 6MV")
    
    # Scale by electron density to get attenuation coefficient for each voxel
    mu = electron_density * mu_water
    
    return mu

def get_center_coordinates(ct_image: Image) -> Tuple[float, float, float]:
    """
    Get the coordinates of the center of the CT image.
    
    Parameters
    ----------
    ct_image : Image
        CT image
        
    Returns
    -------
    Tuple[float, float, float]
        Coordinates of the center (x, y, z) in mm
    """
    dimensions = ct_image.data.shape
    spacing = ct_image.spacing  # in mm
    origin = ct_image.origin if hasattr(ct_image, 'origin') else (0, 0, 0)
    
    # Calculate center coordinates
    center_x = origin[0] + dimensions[0] * spacing[0] / 2
    center_y = origin[1] + dimensions[1] * spacing[1] / 2
    center_z = origin[2] + dimensions[2] * spacing[2] / 2
    
    return (center_x, center_y, center_z)

def calculate_source_position(
    isocenter: Tuple[float, float, float], 
    gantry_angle: float, 
    couch_angle: float,
    beam_model: BeamModel
) -> Tuple[float, float, float]:
    """
    Calculate the source position based on isocenter and angles.
    
    Parameters
    ----------
    isocenter : Tuple[float, float, float]
        Isocenter coordinates (x, y, z) in mm
    gantry_angle : float
        Gantry angle in degrees
    couch_angle : float
        Couch angle in degrees
    beam_model : BeamModel
        Beam model containing SAD information
        
    Returns
    -------
    Tuple[float, float, float]
        Source coordinates (x, y, z) in mm
    """
    # Get source-to-axis distance (SAD) from beam model, default to 1000mm (100cm)
    sad = 1000.0  # Default to 1000mm (100cm)
    if hasattr(beam_model, 'parameters') and 'sad' in beam_model.parameters:
        sad = beam_model.parameters['sad']
    
    # Convert angles to radians
    gantry_rad = np.radians(gantry_angle)
    couch_rad = np.radians(couch_angle)
    
    # Calculate source position
    # Start with gantry rotation in the y-z plane
    dx = 0
    dy = -np.sin(gantry_rad) * sad
    dz = -np.cos(gantry_rad) * sad
    
    # Apply couch rotation around z-axis
    dx_rot = dx * np.cos(couch_rad) - dy * np.sin(couch_rad)
    dy_rot = dx * np.sin(couch_rad) + dy * np.cos(couch_rad)
    
    # Source position = isocenter + offset
    source_x = isocenter[0] + dx_rot
    source_y = isocenter[1] + dy_rot
    source_z = isocenter[2] + dz
    
    return (source_x, source_y, source_z)

def calculate_beam_direction(
    isocenter: Tuple[float, float, float], 
    source_position: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Calculate the beam direction vector.
    
    Parameters
    ----------
    isocenter : Tuple[float, float, float]
        Isocenter coordinates (x, y, z) in mm
    source_position : Tuple[float, float, float]
        Source coordinates (x, y, z) in mm
        
    Returns
    -------
    Tuple[float, float, float]
        Normalized beam direction vector
    """
    # Calculate vector from source to isocenter
    dx = isocenter[0] - source_position[0]
    dy = isocenter[1] - source_position[1]
    dz = isocenter[2] - source_position[2]
    
    # Normalize the vector
    magnitude = np.sqrt(dx*dx + dy*dy + dz*dz)
    if magnitude > 0:
        direction = (dx/magnitude, dy/magnitude, dz/magnitude)
    else:
        # Default direction if source and isocenter are the same (which should not happen)
        direction = (0, 0, 1)
        logger.warning("Source and isocenter positions are the same, using default beam direction")
    
    return direction

def calculate_primary_fluence(
    dimensions: Tuple[int, int, int],
    voxel_spacing_cm: List[float],
    source_position: Tuple[float, float, float],
    beam_direction: Tuple[float, float, float],
    field_size: Tuple[float, float],
    collimator_angle: float,
    beam_model: BeamModel
) -> np.ndarray:
    """
    Calculate primary fluence at each voxel.
    
    Parameters
    ----------
    dimensions : Tuple[int, int, int]
        Dimensions of the CT volume
    voxel_spacing_cm : List[float]
        Voxel spacing in cm
    source_position : Tuple[float, float, float]
        Source coordinates in mm
    beam_direction : Tuple[float, float, float]
        Normalized beam direction vector
    field_size : Tuple[float, float]
        Field size in cm (width, height)
    collimator_angle : float
        Collimator angle in degrees
    beam_model : BeamModel
        Beam model with source parameters
        
    Returns
    -------
    np.ndarray
        Primary fluence at each voxel
    """
    # Initialize fluence array
    fluence = np.zeros(dimensions)
    
    # Convert source position from mm to cm for consistency with voxel_spacing_cm
    source_position_cm = (source_position[0]/10.0, source_position[1]/10.0, source_position[2]/10.0)
    
    # Convert collimator angle to radians
    collimator_rad = np.radians(collimator_angle)
    
    # Convert field size from width x height at isocenter to half-width angles
    # Field size is specified at isocenter, but we need to know the angular extent
    sad = 100.0  # Default SAD in cm
    if hasattr(beam_model, 'parameters') and 'sad' in beam_model.parameters:
        sad = beam_model.parameters['sad'] / 10.0  # Convert from mm to cm
    
    # Half width and height in cm at isocenter
    half_width = field_size[0] / 2.0
    half_height = field_size[1] / 2.0
    
    # Calculate half angles based on field size and SAD
    half_angle_x = np.arctan(half_width / sad)
    half_angle_y = np.arctan(half_height / sad)
    
    # Define beam profiles based on the beam model
    # In a real implementation, beam profiles would be obtained from the beam model
    # Here we use a simplified model - Gaussian fall-off at the field edges
    
    # Calculate coordinates for each voxel
    nx, ny, nz = dimensions
    
    # Precompute coordinates for efficiency
    x_coords = np.zeros(dimensions)
    y_coords = np.zeros(dimensions)
    z_coords = np.zeros(dimensions)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Voxel coordinates in cm
                x = i * voxel_spacing_cm[0]
                y = j * voxel_spacing_cm[1]
                z = k * voxel_spacing_cm[2]
                
                x_coords[i, j, k] = x
                y_coords[i, j, k] = y
                z_coords[i, j, k] = z
    
    # Calculate distance from each voxel to the source
    dx = x_coords - source_position_cm[0]
    dy = y_coords - source_position_cm[1]
    dz = z_coords - source_position_cm[2]
    
    # Distance from source to each voxel
    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Calculate the angle of each voxel with respect to the beam direction
    # Project the voxel-to-source vector onto the beam direction
    cos_theta = (dx * beam_direction[0] + dy * beam_direction[1] + dz * beam_direction[2]) / distance
    
    # Avoid division by zero and ensure cos_theta is in [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Angle between voxel-to-source vector and beam direction
    theta = np.arccos(cos_theta)
    
    # Calculate transverse coordinates in the beam's eye view
    # We need to define two orthogonal vectors in the plane perpendicular to the beam direction
    
    # First orthogonal vector - in the x-y plane if possible
    ortho1 = np.array([-beam_direction[1], beam_direction[0], 0])
    norm = np.linalg.norm(ortho1)
    if norm < 1e-10:
        # If beam_direction is along z-axis, use a different orthogonal vector
        ortho1 = np.array([1, 0, 0])
    else:
        ortho1 = ortho1 / norm
    
    # Second orthogonal vector - cross product of beam_direction and ortho1
    beam_dir_array = np.array(beam_direction)
    ortho2 = np.cross(beam_dir_array, ortho1)
    ortho2 = ortho2 / np.linalg.norm(ortho2)
    
    # Rotate these vectors by collimator angle
    cos_coll = np.cos(collimator_rad)
    sin_coll = np.sin(collimator_rad)
    
    ortho1_rot = cos_coll * ortho1 + sin_coll * ortho2
    ortho2_rot = -sin_coll * ortho1 + cos_coll * ortho2
    
    # Project each voxel onto these orthogonal axes to get beam's eye view coordinates
    x_bev = dx * ortho1_rot[0] + dy * ortho1_rot[1] + dz * ortho1_rot[2]
    y_bev = dx * ortho2_rot[0] + dy * ortho2_rot[1] + dz * ortho2_rot[2]
    
    # Calculate normalized coordinates in the beam's eye view (angle with respect to central axis)
    theta_x = np.arctan2(x_bev, distance)
    theta_y = np.arctan2(y_bev, distance)
    
    # Determine which voxels are within the field
    in_field = (np.abs(theta_x) <= half_angle_x) & (np.abs(theta_y) <= half_angle_y)
    
    # Apply inverse square law to all voxels
    fluence = 1.0 / (distance * distance)
    
    # Set fluence to zero outside the field
    fluence[~in_field] = 0.0
    
    # Apply beam profile - simplified model with flat field and penumbra
    # In a real implementation, this would use measured profiles from the beam model
    penumbra_width = 0.5  # Penumbra width in degrees
    
    # Normalized position within the field (0 at center, 1 at edge)
    x_norm = np.abs(theta_x) / half_angle_x
    y_norm = np.abs(theta_y) / half_angle_y
    
    # Find voxels in the penumbra region
    penumbra_x = (x_norm > 0.9) & (x_norm <= 1.1) & in_field
    penumbra_y = (y_norm > 0.9) & (y_norm <= 1.1) & in_field
    
    # Apply penumbra fall-off - simplified sigmoid function
    if np.any(penumbra_x):
        norm_pos = (x_norm[penumbra_x] - 0.9) / 0.2  # Normalize to [0, 1] within penumbra
        penumbra_factor = 0.5 * (1 + np.cos(norm_pos * np.pi))
        fluence[penumbra_x] *= penumbra_factor
    
    if np.any(penumbra_y):
        norm_pos = (y_norm[penumbra_y] - 0.9) / 0.2  # Normalize to [0, 1] within penumbra
        penumbra_factor = 0.5 * (1 + np.cos(norm_pos * np.pi))
        fluence[penumbra_y] *= penumbra_factor
    
    # Apply off-axis ratio if available in the beam model
    # This is a simplified approach - in a real implementation, 
    # this would use measured off-axis ratios from the beam model
    
    # Here we use a simple cosine function to model the off-axis softening
    off_axis_factor = np.cos(theta * 0.8)  # Simplified off-axis factor
    fluence *= off_axis_factor
    
    # Normalize the fluence (arbitrary units)
    if np.max(fluence) > 0:
        fluence = fluence / np.max(fluence)
    
    return fluence

def calculate_radiological_path_length(
    mu: np.ndarray, 
    source_position: Tuple[float, float, float],
    dimensions: Tuple[int, int, int],
    voxel_spacing_cm: List[float]
) -> np.ndarray:
    """
    Calculate radiological path length from source to each voxel.
    
    This uses a simplified raytracing algorithm to calculate the 
    radiological path length, accounting for the attenuation in each
    voxel along the ray path.
    
    Parameters
    ----------
    mu : np.ndarray
        Linear attenuation coefficients
    source_position : Tuple[float, float, float]
        Source coordinates in mm
    dimensions : Tuple[int, int, int]
        Dimensions of the CT volume
    voxel_spacing_cm : List[float]
        Voxel spacing in cm
        
    Returns
    -------
    np.ndarray
        Radiological path length to each voxel
    """
    # Initialize path length array
    path_length = np.zeros(dimensions)
    
    # Convert source position from mm to indices
    source_x_cm = source_position[0] / 10.0  # Convert from mm to cm
    source_y_cm = source_position[1] / 10.0
    source_z_cm = source_position[2] / 10.0
    
    # Convert source position to voxel indices
    source_i = source_x_cm / voxel_spacing_cm[0]
    source_j = source_y_cm / voxel_spacing_cm[1]
    source_k = source_z_cm / voxel_spacing_cm[2]
    
    # Get dimensions
    nx, ny, nz = dimensions
    
    # Calculate path length using Siddon's algorithm (simplified)
    # This is a computationally efficient approach for raytracing through a 3D volume
    
    # For each voxel, trace a ray from the source to the voxel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Calculate direction vector from source to voxel
                di = i - source_i
                dj = j - source_j
                dk = k - source_k
                
                # Length of the direction vector
                length = np.sqrt(di*di + dj*dj + dk*dk)
                
                # Normalize direction vector
                if length > 0:
                    di /= length
                    dj /= length
                    dk /= length
                else:
                    # Skip if source is in this voxel
                    continue
                
                # Determine step size based on the longest axis
                max_comp = max(abs(di), abs(dj), abs(dk))
                steps = int(length / max_comp * 2)  # Ensure we have enough steps
                
                # Ray tracing
                path = 0.0
                for step in range(steps):
                    # Parameter along the ray
                    t = step / steps
                    
                    # Calculate position along the ray
                    ray_i = source_i + t * di * length
                    ray_j = source_j + t * dj * length
                    ray_k = source_k + t * dk * length
                    
                    # Check if the position is within the volume
                    if (0 <= ray_i < nx and 0 <= ray_j < ny and 0 <= ray_k < nz):
                        # Interpolate attenuation coefficient at this point
                        # We use trilinear interpolation for accuracy
                        
                        # Find the eight surrounding voxels
                        i0 = int(ray_i)
                        j0 = int(ray_j)
                        k0 = int(ray_k)
                        
                        i1 = min(i0 + 1, nx - 1)
                        j1 = min(j0 + 1, ny - 1)
                        k1 = min(k0 + 1, nz - 1)
                        
                        # Calculate interpolation weights
                        wi = ray_i - i0
                        wj = ray_j - j0
                        wk = ray_k - k0
                        
                        # Trilinear interpolation of mu
                        mu_interp = (
                            mu[i0, j0, k0] * (1-wi) * (1-wj) * (1-wk) +
                            mu[i1, j0, k0] * wi * (1-wj) * (1-wk) +
                            mu[i0, j1, k0] * (1-wi) * wj * (1-wk) +
                            mu[i0, j0, k1] * (1-wi) * (1-wj) * wk +
                            mu[i1, j1, k0] * wi * wj * (1-wk) +
                            mu[i1, j0, k1] * wi * (1-wj) * wk +
                            mu[i0, j1, k1] * (1-wi) * wj * wk +
                            mu[i1, j1, k1] * wi * wj * wk
                        )
                        
                        # Step size along the ray in cm
                        step_size = length / steps
                        
                        # Accumulate path length
                        path += mu_interp * step_size
                
                path_length[i, j, k] = path
    
    return path_length

def calculate_terma_from_path_length(
    fluence: np.ndarray,
    path_length: np.ndarray,
    mu: np.ndarray,
    beam_model: BeamModel,
    energy: str
) -> np.ndarray:
    """
    Calculate TERMA using fluence, path length, and attenuation coefficients.
    
    Parameters
    ----------
    fluence : np.ndarray
        Primary fluence at each voxel
    path_length : np.ndarray
        Radiological path length to each voxel
    mu : np.ndarray
        Linear attenuation coefficients
    beam_model : BeamModel
        Beam model with energy-dependent properties
    energy : str
        Beam energy
        
    Returns
    -------
    np.ndarray
        TERMA at each voxel
    """
    # Calculate attenuation based on path length
    attenuation = np.exp(-path_length)
    
    # Calculate mean energy based on the beam energy
    # This is a simplified approach - in a real implementation,
    # mean energy would be obtained from the beam model
    if energy == '6MV':
        mean_energy = 2.0  # Average energy in MeV
    elif energy == '10MV':
        mean_energy = 3.5  # Average energy in MeV
    elif energy == '15MV':
        mean_energy = 5.5  # Average energy in MeV
    elif energy == '18MV':
        mean_energy = 6.5  # Average energy in MeV
    else:
        # Default to 6MV if unknown energy
        mean_energy = 2.0
        logger.warning(f"Unknown energy {energy}, using default mean energy for 6MV")
    
    # TERMA = fluence * attenuation * mu * energy
    terma = fluence * attenuation * mu * mean_energy
    
    # Apply normalization
    if np.max(terma) > 0:
        terma = terma / np.max(terma) * 100.0  # Normalize to 100 at maximum
    
    return terma
