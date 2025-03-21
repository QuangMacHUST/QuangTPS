#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Superposition Convolution algorithm for dose calculation in radiotherapy.

This module provides a class for calculating dose distributions using the
Convolution/Superposition algorithm for radiotherapy treatment planning.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any

from quangtps.core.exceptions import DoseCalculationError, ValidationError
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam
from quangtps.dose.physics.terma import calculate_terma
from quangtps.dose.physics.material import Material
from quangtps.dose.beam_data_processor import BeamModel, BeamModelParameter

logger = logging.getLogger(__name__)

class ConvolutionSuperpositionAlgorithm:
    """
    Implementation of the Convolution/Superposition algorithm.
    
    This class provides methods for calculating 3D dose distributions
    using the Convolution/Superposition algorithm for radiotherapy treatment planning.
    The algorithm is based on calculating the TERMA (Total Energy Released per unit MAss)
    and then convolving it with a kernel representing energy deposition.
    """
    
    def __init__(self):
        """Initialize the Convolution/Superposition algorithm."""
        self.name = "Convolution/Superposition"
        self.version = "1.0.0"
        self.supports_heterogeneity = True
        self.kernels = {}  # Dictionary to store precalculated kernels
        
    def calculate_dose(self, 
                      ct_image: Image, 
                      beam: Beam, 
                      beam_model: BeamModel, 
                      calculation_grid: Optional[Tuple[float, float, float]] = None,
                      parameters: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Calculate the dose distribution using the Convolution/Superposition method.
        
        Parameters
        ----------
        ct_image : Image
            CT image with density information
        beam : Beam
            Treatment beam parameters
        beam_model : BeamModel
            Beam model containing spectrum and kernel data
        calculation_grid : Optional[Tuple[float, float, float]]
            Grid spacing for dose calculation (mm), if None uses CT grid
        parameters : Optional[Dict[str, Any]]
            Additional parameters for the calculation
            
        Returns
        -------
        np.ndarray
            3D dose distribution (Gy)
        """
        logger.info(f"Calculating dose using {self.name} algorithm (version {self.version})")
        start_time = time.time()
        
        # Set default parameters if not provided
        if parameters is None:
            parameters = {}
        
        # Extract parameters
        use_gpu = parameters.get('use_gpu', False)
        num_angles = parameters.get('num_angles', 24)  # Number of angles for angular discretization
        delta_radius = parameters.get('delta_radius', 2.0)  # Radius step (mm)
        max_radius = parameters.get('max_radius', 300.0)  # Maximum radius for kernel (mm)
        cutoff = parameters.get('cutoff', 0.001)  # Energy cutoff value
        
        # Validate inputs
        self._validate_inputs(ct_image, beam, beam_model)
        
        # Get CT image data and properties
        ct_data = ct_image.data
        spacing = ct_image.spacing if calculation_grid is None else calculation_grid
        origin = ct_image.origin
        
        # Convert CT numbers to material properties
        density_map = self._get_density_map(ct_data)
        
        try:
            # Step 1: Calculate TERMA (Total Energy Released per unit MAss)
            logger.info("Calculating TERMA...")
            terma = self._calculate_terma(ct_data, density_map, beam, beam_model, spacing)
            
            # Step 2: Get or create convolution kernel
            logger.info("Preparing convolution kernel...")
            kernel = self._get_convolution_kernel(beam_model, beam.energy, spacing, num_angles, 
                                                delta_radius, max_radius)
            
            # Step 3: Perform the convolution/superposition
            logger.info("Performing convolution/superposition...")
            if use_gpu and self._is_gpu_available():
                dose = self._calculate_superposition_gpu(terma, density_map, kernel, spacing, cutoff)
            else:
                dose = self._calculate_superposition_cpu(terma, density_map, kernel, spacing, cutoff)
            
            # Step 4: Normalize the dose
            logger.info("Normalizing dose...")
            dose = self._normalize_dose(dose, beam)
            
            execution_time = time.time() - start_time
            logger.info(f"Dose calculation completed in {execution_time:.2f} seconds")
            
            return dose
            
        except Exception as e:
            logger.error(f"Error during dose calculation: {str(e)}")
            raise DoseCalculationError(f"Convolution/Superposition dose calculation failed: {str(e)}")
    
    def _validate_inputs(self, ct_image: Image, beam: Beam, beam_model: BeamModel):
        """
        Validate the inputs for dose calculation.
        
        Parameters
        ----------
        ct_image : Image
            CT image
        beam : Beam
            Treatment beam
        beam_model : BeamModel
            Beam model
            
        Raises
        ------
        ValidationError
            If inputs are invalid
        """
        # Check CT image
        if ct_image is None or ct_image.data is None:
            raise ValidationError("CT image is required for dose calculation")
        
        # Check beam
        if beam is None:
            raise ValidationError("Beam is required for dose calculation")
        
        # Check beam model
        if beam_model is None:
            raise ValidationError("Beam model is required for dose calculation")
        
        # Check if beam energy is supported by the beam model
        if not beam_model.supports_energy(beam.energy):
            raise ValidationError(f"Beam energy {beam.energy} is not supported by the beam model")
    
    def _get_density_map(self, ct_data: np.ndarray) -> np.ndarray:
        """
        Convert CT numbers to mass density.
        
        Parameters
        ----------
        ct_data : np.ndarray
            CT data in Hounsfield units
            
        Returns
        -------
        np.ndarray
            Density map in g/cm^3
        """
        # Simple conversion from HU to density
        # Typical formula: density = 1.0 + HU * 0.001 (simplified)
        density_map = 1.0 + ct_data * 0.001
        
        # Clip to physical range
        density_map = np.clip(density_map, 0.001, 3.0)
        
        return density_map
    
    def _calculate_terma(self, 
                       ct_data: np.ndarray, 
                       density_map: np.ndarray, 
                       beam: Beam, 
                       beam_model: BeamModel,
                       spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Calculate the TERMA distribution.
        
        Parameters
        ----------
        ct_data : np.ndarray
            CT data
        density_map : np.ndarray
            Density map
        beam : Beam
            Treatment beam
        beam_model : BeamModel
            Beam model
        spacing : Tuple[float, float, float]
            Grid spacing (mm)
            
        Returns
        -------
        np.ndarray
            TERMA distribution
        """
        # Get beam parameters
        energy = beam.energy
        sad = beam.sad
        
        # Get beam spectrum
        spectrum = beam_model.get_parameter(BeamModelParameter.SPECTRUM, energy)
        
        # Calculate primary fluence
        # This should account for beam geometry, MLC, field size, etc.
        fluence = self._calculate_primary_fluence(beam, ct_data.shape, spacing)
        
        # Calculate TERMA
        terma = calculate_terma(ct_data, density_map, fluence, spectrum, spacing)
        
        return terma
    
    def _calculate_primary_fluence(self, 
                                 beam: Beam, 
                                 shape: Tuple[int, int, int], 
                                 spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Calculate the primary fluence distribution.
        
        Parameters
        ----------
        beam : Beam
            Treatment beam
        shape : Tuple[int, int, int]
            Shape of the calculation grid
        spacing : Tuple[float, float, float]
            Grid spacing (mm)
            
        Returns
        -------
        np.ndarray
            Primary fluence distribution
        """
        # Initialize fluence grid
        fluence = np.zeros(shape, dtype=np.float32)
        
        # Get beam parameters
        field_size = beam.field_size  # (x, y) in cm
        isocenter = beam.isocenter    # (x, y, z) in mm
        gantry_angle = beam.gantry_angle
        collimator_angle = beam.collimator_angle
        
        # Convert field size from cm to mm
        field_size_mm = (field_size[0] * 10, field_size[1] * 10)
        
        # Create coordinate grids (in mm)
        nx, ny, nz = shape
        x = np.arange(0, nx) * spacing[0]
        y = np.arange(0, ny) * spacing[1]
        z = np.arange(0, nz) * spacing[2]
        
        # Adjust coordinates relative to isocenter
        x -= isocenter[0]
        y -= isocenter[1]
        z -= isocenter[2]
        
        # Create 3D coordinates
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Rotate coordinates based on gantry and collimator angles
        # This is a simplified rotation - a full implementation would use rotation matrices
        # Convert angles to radians
        gantry_rad = np.radians(gantry_angle)
        coll_rad = np.radians(collimator_angle)
        
        # Apply gantry rotation (around y-axis)
        X_g = X * np.cos(gantry_rad) + Z * np.sin(gantry_rad)
        Z_g = -X * np.sin(gantry_rad) + Z * np.cos(gantry_rad)
        
        # Apply collimator rotation (around z-axis)
        X_gc = X_g * np.cos(coll_rad) - Y * np.sin(coll_rad)
        Y_gc = X_g * np.sin(coll_rad) + Y * np.cos(coll_rad)
        
        # Determine which points are within the field
        half_width = field_size_mm[0] / 2
        half_height = field_size_mm[1] / 2
        
        # Create a mask for points inside the field
        field_mask = (
            (X_gc >= -half_width) & 
            (X_gc <= half_width) & 
            (Y_gc >= -half_height) & 
            (Y_gc <= half_height)
        )
        
        # Set fluence to 1.0 inside the field, account for inverse square law
        # Distance from source to each point (simplified - assumes source at 0,0,SAD)
        sad = beam.sad  # mm
        source_pos = (0, 0, -sad)  # Assuming beam axis is along z
        
        # Calculate distance from source to each point
        distance = np.sqrt(
            (X - source_pos[0])**2 + 
            (Y - source_pos[1])**2 + 
            (Z - source_pos[2])**2
        )
        
        # Apply inverse square law: fluence ~ 1/r²
        fluence = np.zeros_like(X, dtype=np.float32)
        fluence[field_mask] = (sad / distance[field_mask])**2
        
        # Apply off-axis factors (simplified)
        # In a real implementation, this would use measured off-axis ratios
        off_axis_distance = np.sqrt(X_gc**2 + Y_gc**2)
        off_axis_factor = np.exp(-0.5 * (off_axis_distance / (field_size_mm[0] / 2))**2)
        fluence *= off_axis_factor
        
        # Apply beam profile and modifiers (MLC, wedges, etc.)
        # This is simplified - real implementation would be more complex
        fluence = self._apply_beam_modifiers(fluence, beam)
        
        return fluence
    
    def _apply_beam_modifiers(self, fluence: np.ndarray, beam: Beam) -> np.ndarray:
        """
        Apply beam modifiers like MLC, wedges, etc.
        
        Parameters
        ----------
        fluence : np.ndarray
            Primary fluence
        beam : Beam
            Treatment beam
            
        Returns
        -------
        np.ndarray
            Modified fluence
        """
        # This is a placeholder - real implementation would be complex
        # Apply MLC effect
        if hasattr(beam, 'mlc') and beam.mlc is not None:
            # MLC attenuation - simplified
            mlc_transmission = 0.02  # 2% transmission through MLC
            # For real implementation, need to project MLC leaves onto the fluence grid
            # and calculate attenuation accordingly
            pass
        
        # Apply wedge effect
        if hasattr(beam, 'wedge') and beam.wedge is not None:
            wedge_angle = beam.wedge.angle
            wedge_direction = beam.wedge.direction
            
            # Simplified wedge model
            # For a real implementation, need to use the wedge profile
            shape = fluence.shape
            nx, ny, nz = shape
            
            # Create a linear gradient
            if wedge_direction == 'IN':
                gradient = np.linspace(1.0, np.cos(np.radians(wedge_angle)), nx)
                for i in range(nx):
                    fluence[i, :, :] *= gradient[i]
            elif wedge_direction == 'OUT':
                gradient = np.linspace(np.cos(np.radians(wedge_angle)), 1.0, nx)
                for i in range(nx):
                    fluence[i, :, :] *= gradient[i]
            elif wedge_direction == 'LEFT':
                gradient = np.linspace(1.0, np.cos(np.radians(wedge_angle)), ny)
                for j in range(ny):
                    fluence[:, j, :] *= gradient[j]
            elif wedge_direction == 'RIGHT':
                gradient = np.linspace(np.cos(np.radians(wedge_angle)), 1.0, ny)
                for j in range(ny):
                    fluence[:, j, :] *= gradient[j]
        
        # Apply other modifiers (blocks, compensators, etc.)
        
        return fluence
    
    def _get_convolution_kernel(self, 
                             beam_model: BeamModel, 
                             energy: float, 
                             spacing: Tuple[float, float, float],
                             num_angles: int,
                             delta_radius: float,
                             max_radius: float) -> Dict:
        """
        Get or create a convolution kernel.
        
        Parameters
        ----------
        beam_model : BeamModel
            Beam model
        energy : float
            Beam energy (MV)
        spacing : Tuple[float, float, float]
            Grid spacing (mm)
        num_angles : int
            Number of angles for angular discretization
        delta_radius : float
            Radius step (mm)
        max_radius : float
            Maximum radius for kernel (mm)
            
        Returns
        -------
        Dict
            Convolution kernel data
        """
        # Check if kernel already exists
        key = f"{energy}_{spacing[0]}_{spacing[1]}_{spacing[2]}_{num_angles}_{delta_radius}_{max_radius}"
        if key in self.kernels:
            return self.kernels[key]
        
        # Create a new kernel
        # Get kernel data from beam model or calculate it
        try:
            # Try to get from beam model
            kernel_data = beam_model.get_parameter(BeamModelParameter.KERNEL, energy)
        except:
            # Calculate if not available
            logger.info("Kernel not found in beam model, calculating...")
            kernel_data = self._calculate_kernel(energy, num_angles, delta_radius, max_radius)
        
        # Interpolate kernel to match calculation grid if needed
        kernel_data = self._interpolate_kernel(kernel_data, spacing)
        
        # Store kernel for future use
        self.kernels[key] = kernel_data
        
        return kernel_data
    
    def _calculate_kernel(self, 
                        energy: float, 
                        num_angles: int,
                        delta_radius: float,
                        max_radius: float) -> Dict:
        """
        Calculate a convolution kernel for a given energy.
        
        Parameters
        ----------
        energy : float
            Beam energy (MV)
        num_angles : int
            Number of angles for angular discretization
        delta_radius : float
            Radius step (mm)
        max_radius : float
            Maximum radius for kernel (mm)
            
        Returns
        -------
        Dict
            Convolution kernel data
        """
        logger.info(f"Calculating kernel for {energy} MV...")
        
        # Generate radial points
        num_radial_points = int(max_radius / delta_radius) + 1
        radii = np.linspace(0, max_radius, num_radial_points)
        
        # Generate angular points
        angles = np.linspace(0, np.pi, num_angles)  # Using half sphere (symmetry)
        
        # Initialize kernel arrays
        primary = np.zeros((num_radial_points, num_angles))
        scatter = np.zeros((num_radial_points, num_angles))
        
        # Calculate kernel values
        # This is a simplified model - real kernels are complex
        for i, r in enumerate(radii):
            if r == 0:
                # Avoid division by zero
                primary[i, :] = 1.0
                scatter[i, :] = 0.0
            else:
                # Primary dose: falls off with inverse square
                primary[i, :] = np.exp(-0.06 * energy * r) / (r*r)
                
                # Scatter dose: more complex behavior
                scatter_factor = 0.3 * energy * np.exp(-0.1 * r)
                
                for j, theta in enumerate(angles):
                    # Forward-peaked scatter
                    forward_factor = 0.5 * (1 + np.cos(theta))
                    scatter[i, j] = scatter_factor * forward_factor
        
        # Normalize kernel
        total = primary + scatter
        total /= np.sum(total)
        
        return {
            'radii': radii,
            'angles': angles,
            'primary': primary / np.sum(total),
            'scatter': scatter / np.sum(total),
            'total': total / np.sum(total),
            'energy': energy
        }
    
    def _interpolate_kernel(self, 
                          kernel_data: Dict, 
                          spacing: Tuple[float, float, float]) -> Dict:
        """
        Interpolate kernel to match calculation grid.
        
        Parameters
        ----------
        kernel_data : Dict
            Kernel data
        spacing : Tuple[float, float, float]
            Grid spacing (mm)
            
        Returns
        -------
        Dict
            Interpolated kernel data
        """
        # This is a simplified implementation
        # Real implementation would involve interpolating the kernel to match the grid
        
        # For now, just return the original kernel
        return kernel_data
    
    def _calculate_superposition_cpu(self, 
                                   terma: np.ndarray, 
                                   density_map: np.ndarray,
                                   kernel: Dict, 
                                   spacing: Tuple[float, float, float],
                                   cutoff: float) -> np.ndarray:
        """
        Perform the convolution/superposition on CPU.
        
        Parameters
        ----------
        terma : np.ndarray
            TERMA distribution
        density_map : np.ndarray
            Density map
        kernel : Dict
            Convolution kernel
        spacing : Tuple[float, float, float]
            Grid spacing (mm)
        cutoff : float
            Energy cutoff value
            
        Returns
        -------
        np.ndarray
            Dose distribution
        """
        # Get grid dimensions
        nx, ny, nz = terma.shape
        
        # Initialize dose grid
        dose = np.zeros_like(terma)
        
        # Get kernel data
        radii = kernel['radii']
        angles = kernel['angles']
        kernel_primary = kernel['primary']
        kernel_scatter = kernel['scatter']
        
        # Precompute sin and cos values for angles
        sin_theta = np.sin(angles)
        cos_theta = np.cos(angles)
        
        # Get kernel dimensions
        nr = len(radii)
        na = len(angles)
        
        # Get maximum kernel radius in voxels
        max_radius_voxels = int(np.ceil(radii[-1] / min(spacing)))
        
        # Convert spacing to array for vectorized operations
        spacing_arr = np.array(spacing)
        
        # Perform superposition
        # This is the most computationally intensive part
        # A real implementation would use more optimized methods
        
        # Loop over all voxels with non-zero TERMA
        # Using a mask to skip zero or near-zero TERMA voxels
        terma_mask = terma > cutoff
        terma_indices = np.nonzero(terma_mask)
        
        num_voxels = len(terma_indices[0])
        logger.info(f"Processing {num_voxels} voxels with non-zero TERMA")
        
        # Process voxels in batches to reduce memory usage
        batch_size = 1000
        num_batches = (num_voxels + batch_size - 1) // batch_size
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, num_voxels)
            
            logger.info(f"Processing batch {batch+1}/{num_batches} ({end_idx-start_idx} voxels)")
            
            for idx in range(start_idx, end_idx):
                i, j, k = terma_indices[0][idx], terma_indices[1][idx], terma_indices[2][idx]
                
                # TERMA at current voxel
                terma_value = terma[i, j, k]
                
                # Skip if TERMA is too small
                if terma_value <= cutoff:
                    continue
                
                # Density at current voxel
                density = density_map[i, j, k]
                
                # Loop over all kernel points
                for ir in range(1, nr):  # Skip r=0 to avoid division by zero
                    r = radii[ir]
                    radius_voxels = r / min(spacing)
                    
                    # Skip if radius is too large
                    if radius_voxels > max_radius_voxels:
                        continue
                    
                    for ia in range(na):
                        # Skip very small kernel values
                        kernel_value = kernel_primary[ir, ia] + kernel_scatter[ir, ia]
                        if kernel_value <= cutoff:
                            continue
                        
                        # Calculate vector components
                        # Using spherical coordinates (r, theta, phi)
                        # For simplicity, using full sphere (phi = 0..2π)
                        # In a real implementation, use symmetry to reduce computation
                        for phi in np.linspace(0, 2*np.pi, 8):  # Using 8 phi angles
                            sin_phi = np.sin(phi)
                            cos_phi = np.cos(phi)
                            
                            # Direction vector
                            dir_x = r * sin_theta[ia] * cos_phi
                            dir_y = r * sin_theta[ia] * sin_phi
                            dir_z = r * cos_theta[ia]
                            
                            # Convert to voxel increments
                            voxel_x = dir_x / spacing[0]
                            voxel_y = dir_y / spacing[1]
                            voxel_z = dir_z / spacing[2]
                            
                            # Find target voxel
                            target_i = int(i + voxel_x)
                            target_j = int(j + voxel_y)
                            target_k = int(k + voxel_z)
                            
                            # Check if target voxel is within grid
                            if (0 <= target_i < nx and 
                                0 <= target_j < ny and 
                                0 <= target_k < nz):
                                
                                # Density scaling (approximating radiological path)
                                # In a real implementation, use raytracing for accurate path
                                target_density = density_map[target_i, target_j, target_k]
                                density_ratio = density / target_density
                                
                                # Calculate dose contribution
                                dose_contrib = terma_value * kernel_value * density_ratio
                                
                                # Add contribution to dose
                                dose[target_i, target_j, target_k] += dose_contrib / 8  # divide by number of phi angles
        
        return dose
    
    def _calculate_superposition_gpu(self, 
                                   terma: np.ndarray, 
                                   density_map: np.ndarray,
                                   kernel: Dict, 
                                   spacing: Tuple[float, float, float],
                                   cutoff: float) -> np.ndarray:
        """
        Perform the convolution/superposition on GPU.
        
        Parameters
        ----------
        terma : np.ndarray
            TERMA distribution
        density_map : np.ndarray
            Density map
        kernel : Dict
            Convolution kernel
        spacing : Tuple[float, float, float]
            Grid spacing (mm)
        cutoff : float
            Energy cutoff value
            
        Returns
        -------
        np.ndarray
            Dose distribution
        """
        # Placeholder for GPU implementation
        logger.warning("GPU implementation not available, falling back to CPU")
        return self._calculate_superposition_cpu(terma, density_map, kernel, spacing, cutoff)
    
    def _is_gpu_available(self) -> bool:
        """
        Check if GPU is available for computation.
        
        Returns
        -------
        bool
            True if GPU is available, False otherwise
        """
        # Placeholder - real implementation would check for CUDA, OpenCL, etc.
        return False
    
    def _normalize_dose(self, dose: np.ndarray, beam: Beam) -> np.ndarray:
        """
        Normalize the dose distribution.
        
        Parameters
        ----------
        dose : np.ndarray
            Dose distribution
        beam : Beam
            Treatment beam
            
        Returns
        -------
        np.ndarray
            Normalized dose distribution
        """
        # Get monitor units
        mu = beam.mu
        
        # Get calibration factor (Gy/MU at reference conditions)
        # In a real implementation, this would come from beam data
        cal_factor = 0.01  # 1 cGy/MU (typical)
        
        # Scale dose by MU and calibration factor
        dose *= mu * cal_factor
        
        return dose


class ConvolutionAlgorithm(ConvolutionSuperpositionAlgorithm):
    """
    Standard convolution algorithm without density scaling during superposition.
    
    This is a simplified version of the superposition algorithm that doesn't
    account for density heterogeneities during the convolution process.
    """
    
    def __init__(self):
        """Initialize the Convolution algorithm."""
        super().__init__()
        self.name = "Convolution"
        self.supports_heterogeneity = False
    
    def _calculate_superposition_cpu(self, 
                                   terma: np.ndarray, 
                                   density_map: np.ndarray,
                                   kernel: Dict, 
                                   spacing: Tuple[float, float, float],
                                   cutoff: float) -> np.ndarray:
        """
        Perform the convolution on CPU without density scaling.
        
        This method overrides the superposition method to perform a simpler
        convolution without accounting for density heterogeneities during
        the convolution process.
        
        Parameters
        ----------
        terma : np.ndarray
            TERMA distribution
        density_map : np.ndarray
            Density map (not used in this method)
        kernel : Dict
            Convolution kernel
        spacing : Tuple[float, float, float]
            Grid spacing (mm)
        cutoff : float
            Energy cutoff value
            
        Returns
        -------
        np.ndarray
            Dose distribution
        """
        # For a simple convolution, we can use FFT-based convolution
        # Convert kernel to a 3D array
        kernel_3d = self._convert_kernel_to_3d(kernel, terma.shape, spacing)
        
        # Perform FFT convolution
        logger.info("Performing FFT convolution...")
        
        # Compute FFT of terma and kernel
        terma_fft = np.fft.fftn(terma)
        kernel_fft = np.fft.fftn(kernel_3d)
        
        # Multiply in frequency domain
        result_fft = terma_fft * kernel_fft
        
        # Inverse FFT to get dose
        dose = np.real(np.fft.ifftn(result_fft))
        
        return dose
    
    def _convert_kernel_to_3d(self, 
                            kernel: Dict, 
                            shape: Tuple[int, int, int],
                            spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Convert polar kernel to 3D Cartesian grid.
        
        Parameters
        ----------
        kernel : Dict
            Kernel data in polar coordinates
        shape : Tuple[int, int, int]
            Shape of the output grid
        spacing : Tuple[float, float, float]
            Grid spacing (mm)
            
        Returns
        -------
        np.ndarray
            3D kernel array
        """
        # Get grid dimensions
        nx, ny, nz = shape
        
        # Calculate center of kernel
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        
        # Initialize 3D kernel
        kernel_3d = np.zeros(shape, dtype=np.float32)
        
        # Get kernel data
        radii = kernel['radii']
        angles = kernel['angles']
        kernel_total = kernel['total']
        
        # Get maximum kernel radius in voxels
        max_radius_voxels = int(np.ceil(radii[-1] / min(spacing)))
        
        # Make sure the kernel fits within the grid
        if max_radius_voxels >= min(cx, cy, cz):
            logger.warning("Kernel radius exceeds grid size, truncating kernel")
            max_radius_voxels = min(cx, cy, cz) - 1
        
        # Create coordinate grids
        x = np.arange(-cx, nx-cx) * spacing[0]
        y = np.arange(-cy, ny-cy) * spacing[1]
        z = np.arange(-cz, nz-cz) * spacing[2]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate radial distance from center
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Calculate polar angle (theta)
        # Avoid division by zero at the origin
        R_flat = np.clip(R.flatten(), 1e-10, None)
        Z_flat = Z.flatten()
        Theta = np.arccos(np.clip(Z_flat / R_flat, -1.0, 1.0)).reshape(R.shape)
        
        # For each voxel, interpolate the kernel value
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    r = R[i, j, k]
                    
                    # Skip if outside kernel radius
                    if r > radii[-1]:
                        continue
                    
                    theta = Theta[i, j, k]
                    
                    # Interpolate kernel value
                    # Find radial bin
                    r_idx = np.searchsorted(radii, r) - 1
                    r_idx = max(0, min(r_idx, len(radii) - 2))
                    r1, r2 = radii[r_idx], radii[r_idx + 1]
                    r_frac = (r - r1) / (r2 - r1) if r2 > r1 else 0.0
                    
                    # Find angular bin
                    theta_idx = np.searchsorted(angles, theta) - 1
                    theta_idx = max(0, min(theta_idx, len(angles) - 2))
                    theta1, theta2 = angles[theta_idx], angles[theta_idx + 1]
                    theta_frac = (theta - theta1) / (theta2 - theta1) if theta2 > theta1 else 0.0
                    
                    # Bilinear interpolation
                    v1 = kernel_total[r_idx, theta_idx]
                    v2 = kernel_total[r_idx + 1, theta_idx]
                    v3 = kernel_total[r_idx, theta_idx + 1]
                    v4 = kernel_total[r_idx + 1, theta_idx + 1]
                    
                    v12 = v1 * (1 - r_frac) + v2 * r_frac
                    v34 = v3 * (1 - r_frac) + v4 * r_frac
                    
                    kernel_value = v12 * (1 - theta_frac) + v34 * theta_frac
                    
                    # Set kernel value
                    kernel_3d[i, j, k] = kernel_value
        
        # Normalize kernel
        kernel_3d /= np.sum(kernel_3d)
        
        return kernel_3d
