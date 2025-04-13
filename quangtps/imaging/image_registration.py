#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Registration Module
========================

This module provides functionality for registering (fusing) multiple medical images
in a manner similar to Eclipse's image registration functionality.
"""

import logging
import os
import numpy as np
import SimpleITK as sitk
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from enum import Enum
from dataclasses import dataclass

from quangtps.core.services import ServiceRegistry
from quangtps.imaging.image import Image
from quangtps.structures.structure import Structure
from quangtps.structures.structure_set import StructureSet

logger = logging.getLogger(__name__)

class RegistrationType(Enum):
    """Types of registration that can be performed."""
    RIGID = "rigid"
    AFFINE = "affine"
    DEFORMABLE = "deformable"
    HYBRID = "hybrid"

class RegistrationMetric(Enum):
    """Metrics used for evaluating registration quality."""
    MUTUAL_INFORMATION = "mutual_information"
    NORMALIZED_CORRELATION = "normalized_correlation"
    MEAN_SQUARES = "mean_squares"
    CORRELATION_RATIO = "correlation_ratio"

@dataclass
class RegistrationResult:
    """Result of a registration operation."""
    success: bool
    transform: Any  # SimpleITK transform
    metric_value: float
    fixed_image_id: str
    moving_image_id: str
    registration_type: RegistrationType
    parameters: Dict[str, Any]
    transform_parameters: List[float]
    center_of_rotation: List[float]
    registered_image: Optional[Image] = None

class ImageRegistration:
    """
    Eclipse-like image registration class for multi-modality fusion.
    
    This class provides functionality for registering medical images using various
    registration methods, similar to Eclipse's image registration capabilities.
    """
    
    def __init__(self):
        """Initialize the image registration module."""
        self.fixed_image = None
        self.moving_image = None
        self.registration_results = {}
        self.current_result = None
        
    def set_fixed_image(self, image: Image) -> None:
        """
        Set the fixed image for registration.
        
        Parameters
        ----------
        image : Image
            Fixed image (reference image)
        """
        self.fixed_image = image
        logger.info(f"Set fixed image: {image.id if hasattr(image, 'id') else 'Unknown'}")
        
    def set_moving_image(self, image: Image) -> None:
        """
        Set the moving image for registration.
        
        Parameters
        ----------
        image : Image
            Moving image (image to be registered)
        """
        self.moving_image = image
        logger.info(f"Set moving image: {image.id if hasattr(image, 'id') else 'Unknown'}")
        
    def register_images(self, 
                       registration_type: RegistrationType = RegistrationType.RIGID,
                       metric: RegistrationMetric = RegistrationMetric.MUTUAL_INFORMATION,
                       parameters: Dict[str, Any] = None,
                       callback: Callable[[float], None] = None) -> RegistrationResult:
        """
        Register moving image to fixed image.
        
        Parameters
        ----------
        registration_type : RegistrationType
            Type of registration to perform
        metric : RegistrationMetric
            Metric to use for registration
        parameters : Dict[str, Any]
            Additional parameters for registration
        callback : Callable[[float], None]
            Callback function for progress updates
            
        Returns
        -------
        RegistrationResult
            Result of registration
        """
        if self.fixed_image is None or self.moving_image is None:
            logger.error("Fixed or moving image not set")
            return RegistrationResult(
                success=False,
                transform=None,
                metric_value=0.0,
                fixed_image_id=self.fixed_image.id if self.fixed_image else "",
                moving_image_id=self.moving_image.id if self.moving_image else "",
                registration_type=registration_type,
                parameters=parameters or {},
                transform_parameters=[],
                center_of_rotation=[]
            )
            
        # Default parameters
        if parameters is None:
            parameters = {}
        
        fixed_sitk = self._image_to_sitk(self.fixed_image)
        moving_sitk = self._image_to_sitk(self.moving_image)
        
        # Setup registration method based on type
        if registration_type == RegistrationType.RIGID:
            result = self._perform_rigid_registration(fixed_sitk, moving_sitk, metric, parameters, callback)
        elif registration_type == RegistrationType.AFFINE:
            result = self._perform_affine_registration(fixed_sitk, moving_sitk, metric, parameters, callback)
        elif registration_type == RegistrationType.DEFORMABLE:
            result = self._perform_deformable_registration(fixed_sitk, moving_sitk, metric, parameters, callback)
        else:
            logger.error(f"Unsupported registration type: {registration_type}")
            return RegistrationResult(
                success=False,
                transform=None,
                metric_value=0.0,
                fixed_image_id=self.fixed_image.id,
                moving_image_id=self.moving_image.id,
                registration_type=registration_type,
                parameters=parameters,
                transform_parameters=[],
                center_of_rotation=[]
            )
            
        # Store result
        result_id = f"{self.fixed_image.id}_{self.moving_image.id}_{registration_type.value}"
        self.registration_results[result_id] = result
        self.current_result = result
        
        logger.info(f"Registration completed: {result.success}, metric value: {result.metric_value}")
        
        return result
        
    def _perform_rigid_registration(self, fixed_sitk, moving_sitk, metric, parameters, callback=None):
        """Perform rigid registration."""
        try:
            # Create registration method
            registration_method = sitk.ImageRegistrationMethod()
            
            # Set up metric
            if metric == RegistrationMetric.MUTUAL_INFORMATION:
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            elif metric == RegistrationMetric.NORMALIZED_CORRELATION:
                registration_method.SetMetricAsCorrelation()
            elif metric == RegistrationMetric.MEAN_SQUARES:
                registration_method.SetMetricAsMeanSquares()
            else:
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                
            # Set optimizer
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=parameters.get("learning_rate", 1.0),
                numberOfIterations=parameters.get("iterations", 100),
                convergenceMinimumValue=parameters.get("convergence_value", 1e-6),
                convergenceWindowSize=parameters.get("convergence_window", 10)
            )
            
            # Set interpolator
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Set initial transform
            transform = sitk.Euler3DTransform()
            if "initial_rotation" in parameters:
                transform.SetRotation(*parameters["initial_rotation"])
            if "initial_translation" in parameters:
                transform.SetTranslation(parameters["initial_translation"])
                
            registration_method.SetInitialTransform(transform)
            
            # Setup callback if provided
            if callback:
                def internal_callback():
                    callback(registration_method.GetOptimizerIteration() / 
                            parameters.get("iterations", 100))
                registration_method.AddCommand(sitk.sitkIterationEvent, internal_callback)
                
            # Perform registration
            final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
            
            # Apply transform to moving image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(final_transform)
            
            registered_sitk = resampler.Execute(moving_sitk)
            registered_image = self._sitk_to_image(registered_sitk, self.moving_image)
            
            # Get transformation parameters
            transform_params = list(final_transform.GetParameters())
            center_of_rotation = list(final_transform.GetCenter())
            
            return RegistrationResult(
                success=True,
                transform=final_transform,
                metric_value=registration_method.GetMetricValue(),
                fixed_image_id=self.fixed_image.id,
                moving_image_id=self.moving_image.id,
                registration_type=RegistrationType.RIGID,
                parameters=parameters,
                transform_parameters=transform_params,
                center_of_rotation=center_of_rotation,
                registered_image=registered_image
            )
            
        except Exception as e:
            logger.error(f"Error performing rigid registration: {e}")
            return RegistrationResult(
                success=False,
                transform=None,
                metric_value=0.0,
                fixed_image_id=self.fixed_image.id,
                moving_image_id=self.moving_image.id,
                registration_type=RegistrationType.RIGID,
                parameters=parameters,
                transform_parameters=[],
                center_of_rotation=[],
                registered_image=None
            )
    
    def _perform_affine_registration(self, fixed_sitk, moving_sitk, metric, parameters, callback=None):
        """Perform affine registration."""
        try:
            # Create registration method
            registration_method = sitk.ImageRegistrationMethod()
            
            # Set up metric
            if metric == RegistrationMetric.MUTUAL_INFORMATION:
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            elif metric == RegistrationMetric.NORMALIZED_CORRELATION:
                registration_method.SetMetricAsCorrelation()
            elif metric == RegistrationMetric.MEAN_SQUARES:
                registration_method.SetMetricAsMeanSquares()
            else:
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                
            # Set optimizer
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=parameters.get("learning_rate", 1.0),
                numberOfIterations=parameters.get("iterations", 100),
                convergenceMinimumValue=parameters.get("convergence_value", 1e-6),
                convergenceWindowSize=parameters.get("convergence_window", 10)
            )
            
            # Set interpolator
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Set initial transform
            transform = sitk.AffineTransform(3)
            if "initial_matrix" in parameters:
                transform.SetMatrix(parameters["initial_matrix"])
            if "initial_translation" in parameters:
                transform.SetTranslation(parameters["initial_translation"])
                
            registration_method.SetInitialTransform(transform)
            
            # Setup callback if provided
            if callback:
                def internal_callback():
                    callback(registration_method.GetOptimizerIteration() / 
                            parameters.get("iterations", 100))
                registration_method.AddCommand(sitk.sitkIterationEvent, internal_callback)
                
            # Perform registration
            final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
            
            # Apply transform to moving image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(final_transform)
            
            registered_sitk = resampler.Execute(moving_sitk)
            registered_image = self._sitk_to_image(registered_sitk, self.moving_image)
            
            # Get transformation parameters
            transform_params = list(final_transform.GetParameters())
            center_of_rotation = list(final_transform.GetCenter())
            
            return RegistrationResult(
                success=True,
                transform=final_transform,
                metric_value=registration_method.GetMetricValue(),
                fixed_image_id=self.fixed_image.id,
                moving_image_id=self.moving_image.id,
                registration_type=RegistrationType.AFFINE,
                parameters=parameters,
                transform_parameters=transform_params,
                center_of_rotation=center_of_rotation,
                registered_image=registered_image
            )
            
        except Exception as e:
            logger.error(f"Error performing affine registration: {e}")
            return RegistrationResult(
                success=False,
                transform=None,
                metric_value=0.0,
                fixed_image_id=self.fixed_image.id,
                moving_image_id=self.moving_image.id,
                registration_type=RegistrationType.AFFINE,
                parameters=parameters,
                transform_parameters=[],
                center_of_rotation=[],
                registered_image=None
            )
    
    def _perform_deformable_registration(self, fixed_sitk, moving_sitk, metric, parameters, callback=None):
        """Perform deformable registration."""
        try:
            # Create registration method
            registration_method = sitk.ImageRegistrationMethod()
            
            # Set up metric
            if metric == RegistrationMetric.MUTUAL_INFORMATION:
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            elif metric == RegistrationMetric.NORMALIZED_CORRELATION:
                registration_method.SetMetricAsCorrelation()
            elif metric == RegistrationMetric.MEAN_SQUARES:
                registration_method.SetMetricAsMeanSquares()
            else:
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                
            # Set optimizer
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=parameters.get("learning_rate", 1.0),
                numberOfIterations=parameters.get("iterations", 100),
                convergenceMinimumValue=parameters.get("convergence_value", 1e-6),
                convergenceWindowSize=parameters.get("convergence_window", 10)
            )
            
            # Set interpolator
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Perform initial rigid alignment
            initial_transform = sitk.Euler3DTransform()
            
            # Setup B-spline transform
            transform_domain_mesh_size = parameters.get("mesh_size", [10, 10, 10])
            transform = sitk.BSplineTransformInitializer(
                fixed_sitk, 
                transform_domain_mesh_size
            )
            
            registration_method.SetInitialTransform(transform)
            
            # Setup callback if provided
            if callback:
                def internal_callback():
                    callback(registration_method.GetOptimizerIteration() / 
                            parameters.get("iterations", 100))
                registration_method.AddCommand(sitk.sitkIterationEvent, internal_callback)
                
            # Perform registration
            final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
            
            # Apply transform to moving image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(final_transform)
            
            registered_sitk = resampler.Execute(moving_sitk)
            registered_image = self._sitk_to_image(registered_sitk, self.moving_image)
            
            # Get transformation parameters (for BSpline, these are the control points)
            transform_params = list(final_transform.GetParameters())
            center_of_rotation = [0, 0, 0]  # Not applicable for BSpline
            
            return RegistrationResult(
                success=True,
                transform=final_transform,
                metric_value=registration_method.GetMetricValue(),
                fixed_image_id=self.fixed_image.id,
                moving_image_id=self.moving_image.id,
                registration_type=RegistrationType.DEFORMABLE,
                parameters=parameters,
                transform_parameters=transform_params,
                center_of_rotation=center_of_rotation,
                registered_image=registered_image
            )
            
        except Exception as e:
            logger.error(f"Error performing deformable registration: {e}")
            return RegistrationResult(
                success=False,
                transform=None,
                metric_value=0.0,
                fixed_image_id=self.fixed_image.id,
                moving_image_id=self.moving_image.id,
                registration_type=RegistrationType.DEFORMABLE,
                parameters=parameters,
                transform_parameters=[],
                center_of_rotation=[],
                registered_image=None
            )
    
    def apply_transform_to_structure_set(self, structure_set: StructureSet, transform: Any) -> StructureSet:
        """
        Apply a transform to a structure set.
        
        Parameters
        ----------
        structure_set : StructureSet
            Structure set to transform
        transform : Any
            SimpleITK transform
            
        Returns
        -------
        StructureSet
            Transformed structure set
        """
        if transform is None:
            logger.error("Transform not provided")
            return None
            
        try:
            # Create a new structure set
            new_structure_set = StructureSet(
                id=f"{structure_set.id}_transformed",
                name=f"{structure_set.name}_transformed",
                description=f"Transformed from {structure_set.name}"
            )
            
            # Transform each structure
            for structure in structure_set.structures:
                # Convert structure to SimpleITK mask
                mask = self._structure_to_sitk_mask(structure, self.moving_image)
                
                # Apply transform
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(self._image_to_sitk(self.fixed_image))
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                resampler.SetTransform(transform)
                
                transformed_mask = resampler.Execute(mask)
                
                # Convert back to structure
                transformed_structure = self._sitk_mask_to_structure(
                    transformed_mask, 
                    structure.name, 
                    structure.color, 
                    self.fixed_image
                )
                
                # Add to new structure set
                new_structure_set.add_structure(transformed_structure)
                
            return new_structure_set
            
        except Exception as e:
            logger.error(f"Error applying transform to structure set: {e}")
            return None
    
    def manual_registration(self, 
                          translation: List[float] = None, 
                          rotation: List[float] = None,
                          scale: List[float] = None) -> RegistrationResult:
        """
        Perform manual registration by applying user-specified transformation.
        
        Parameters
        ----------
        translation : List[float]
            Translation in [x, y, z] mm
        rotation : List[float]
            Rotation in [x, y, z] degrees
        scale : List[float]
            Scale factors in [x, y, z]
            
        Returns
        -------
        RegistrationResult
            Result of registration
        """
        if self.fixed_image is None or self.moving_image is None:
            logger.error("Fixed or moving image not set")
            return RegistrationResult(
                success=False,
                transform=None,
                metric_value=0.0,
                fixed_image_id=self.fixed_image.id if self.fixed_image else "",
                moving_image_id=self.moving_image.id if self.moving_image else "",
                registration_type=RegistrationType.RIGID,
                parameters={},
                transform_parameters=[],
                center_of_rotation=[]
            )
            
        try:
            # Create transform
            transform = sitk.Similarity3DTransform()
            
            # Set center of rotation (center of fixed image)
            fixed_sitk = self._image_to_sitk(self.fixed_image)
            center = [x/2 for x in fixed_sitk.GetSize()]
            physical_center = fixed_sitk.TransformIndexToPhysicalPoint(center)
            transform.SetCenter(physical_center)
            
            # Set translation
            if translation:
                transform.SetTranslation(translation)
                
            # Set rotation (convert degrees to radians)
            if rotation:
                # Convert to radians
                rotation_rad = [np.deg2rad(r) for r in rotation]
                transform.SetRotation(*rotation_rad)
                
            # Set scale
            if scale:
                transform.SetScale(scale[0])  # Similarity transform uses isotropic scaling
                
            # Apply transform to moving image
            moving_sitk = self._image_to_sitk(self.moving_image)
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(transform)
            
            registered_sitk = resampler.Execute(moving_sitk)
            registered_image = self._sitk_to_image(registered_sitk, self.moving_image)
            
            # Get transformation parameters
            transform_params = list(transform.GetParameters())
            center_of_rotation = list(transform.GetCenter())
            
            # Create result
            result = RegistrationResult(
                success=True,
                transform=transform,
                metric_value=0.0,  # No metric value for manual registration
                fixed_image_id=self.fixed_image.id,
                moving_image_id=self.moving_image.id,
                registration_type=RegistrationType.RIGID,
                parameters={
                    "translation": translation,
                    "rotation": rotation,
                    "scale": scale
                },
                transform_parameters=transform_params,
                center_of_rotation=center_of_rotation,
                registered_image=registered_image
            )
            
            # Store result
            result_id = f"{self.fixed_image.id}_{self.moving_image.id}_manual"
            self.registration_results[result_id] = result
            self.current_result = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing manual registration: {e}")
            return RegistrationResult(
                success=False,
                transform=None,
                metric_value=0.0,
                fixed_image_id=self.fixed_image.id,
                moving_image_id=self.moving_image.id,
                registration_type=RegistrationType.RIGID,
                parameters={},
                transform_parameters=[],
                center_of_rotation=[]
            )
    
    def get_registration_result(self, fixed_image_id: str, moving_image_id: str, registration_type: RegistrationType = None) -> Optional[RegistrationResult]:
        """
        Get a registration result by image IDs.
        
        Parameters
        ----------
        fixed_image_id : str
            ID of fixed image
        moving_image_id : str
            ID of moving image
        registration_type : RegistrationType
            Type of registration
            
        Returns
        -------
        Optional[RegistrationResult]
            Registration result if found, None otherwise
        """
        if registration_type:
            result_id = f"{fixed_image_id}_{moving_image_id}_{registration_type.value}"
            return self.registration_results.get(result_id)
        else:
            # Return the most recent result for these images
            for key, result in self.registration_results.items():
                if result.fixed_image_id == fixed_image_id and result.moving_image_id == moving_image_id:
                    return result
        return None
    
    def _image_to_sitk(self, image: Image) -> sitk.Image:
        """Convert Image to SimpleITK image."""
        # This implementation depends on the specific Image class structure
        # Here's a generic approach
        try:
            # Assuming image.data is a numpy array
            sitk_image = sitk.GetImageFromArray(image.data)
            
            # Set spacing, origin, direction if available
            if hasattr(image, 'spacing'):
                sitk_image.SetSpacing(image.spacing)
            if hasattr(image, 'origin'):
                sitk_image.SetOrigin(image.origin)
            if hasattr(image, 'direction'):
                sitk_image.SetDirection(image.direction)
                
            return sitk_image
            
        except Exception as e:
            logger.error(f"Error converting Image to SimpleITK: {e}")
            return sitk.Image([1, 1, 1], sitk.sitkFloat32)
    
    def _sitk_to_image(self, sitk_image: sitk.Image, template_image: Image) -> Image:
        """Convert SimpleITK image to Image."""
        try:
            # Convert to numpy array
            array = sitk.GetArrayFromImage(sitk_image)
            
            # Create new Image object (implementation depends on Image class)
            # Here we're creating a simple class with the necessary attributes
            class SimpleImage:
                def __init__(self, data, id, spacing, origin, direction):
                    self.data = data
                    self.id = id
                    self.spacing = spacing
                    self.origin = origin
                    self.direction = direction
                    
            # Create a new ID
            new_id = f"{template_image.id}_registered"
            
            # Create Image with same properties as template but new data
            image = SimpleImage(
                data=array,
                id=new_id,
                spacing=sitk_image.GetSpacing(),
                origin=sitk_image.GetOrigin(),
                direction=sitk_image.GetDirection()
            )
            
            return image
            
        except Exception as e:
            logger.error(f"Error converting SimpleITK to Image: {e}")
            return None
    
    def _structure_to_sitk_mask(self, structure: Structure, reference_image: Image) -> sitk.Image:
        """Convert Structure to SimpleITK binary mask."""
        try:
            # Create an empty mask
            reference_sitk = self._image_to_sitk(reference_image)
            mask = sitk.Image(reference_sitk.GetSize(), sitk.sitkUInt8)
            mask.CopyInformation(reference_sitk)
            mask.Fill(0)
            
            # Fill mask with structure contours
            # Implementation depends on Structure class
            # This is a simplified example
            if hasattr(structure, 'mask') and structure.mask is not None:
                # If structure already has a mask array
                mask_array = structure.mask
                sitk_mask = sitk.GetImageFromArray(mask_array)
                sitk_mask.CopyInformation(reference_sitk)
                return sitk_mask
            else:
                # Need to create mask from contours
                # This is complex and depends on structure implementation
                # For this example, we'll just return an empty mask
                return mask
                
        except Exception as e:
            logger.error(f"Error converting Structure to SimpleITK mask: {e}")
            return sitk.Image([1, 1, 1], sitk.sitkUInt8)
    
    def _sitk_mask_to_structure(self, mask: sitk.Image, name: str, color: List[int], reference_image: Image) -> Structure:
        """Convert SimpleITK binary mask to Structure."""
        try:
            # Convert mask to numpy array
            mask_array = sitk.GetArrayFromImage(mask)
            
            # Create structure from mask
            # Implementation depends on Structure class
            # This is a simplified example
            structure = Structure(
                name=name,
                color=color
            )
            
            # Set mask
            if hasattr(structure, 'set_mask'):
                structure.set_mask(mask_array)
                
            return structure
            
        except Exception as e:
            logger.error(f"Error converting SimpleITK mask to Structure: {e}")
            return None
            
    def validate_registration(self, result: RegistrationResult) -> Dict[str, Any]:
        """
        Validate a registration result with various metrics.
        
        Parameters
        ----------
        result : RegistrationResult
            Registration result to validate
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of validation metrics
        """
        if not result or not result.success or not result.transform:
            return {"valid": False, "error": "Invalid registration result"}
            
        try:
            # Calculate validation metrics
            fixed_sitk = self._image_to_sitk(self.fixed_image)
            moving_sitk = self._image_to_sitk(self.moving_image)
            
            # Mutual information
            metric = sitk.ImageRegistrationMethod()
            metric.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            mi_value = metric.MetricEvaluate(fixed_sitk, moving_sitk)
            
            # Normalized correlation
            metric.SetMetricAsCorrelation()
            nc_value = metric.MetricEvaluate(fixed_sitk, moving_sitk)
            
            # Mean squares
            metric.SetMetricAsMeanSquares()
            ms_value = metric.MetricEvaluate(fixed_sitk, moving_sitk)
            
            return {
                "valid": True,
                "mutual_information": mi_value,
                "normalized_correlation": nc_value,
                "mean_squares": ms_value,
                "transform_type": result.registration_type.value,
                "parameters": result.transform_parameters
            }
            
        except Exception as e:
            logger.error(f"Error validating registration: {e}")
            return {"valid": False, "error": str(e)}

# Create an instance of the image registration class for use in the application
image_registration = ImageRegistration() 