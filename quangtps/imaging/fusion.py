#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image fusion module for radiotherapy treatment planning.

This module provides classes and functions for multi-modality image fusion
(CT-MRI, CT-PET, etc.) to aid in radiotherapy treatment planning.
"""

import numpy as np
import SimpleITK as sitk
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum, auto
import matplotlib.colors as mcolors

from quangtps.imaging.image import Image
from quangtps.core.exceptions import FusionError, ValidationError

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Enum defining various image fusion methods."""
    ALPHA_BLENDING = auto()
    CHECKERBOARD = auto()
    COLORMAP_OVERLAY = auto()
    WINDOW_LEVEL = auto()
    SUBTRACTION = auto()
    WINDOWED_SUBTRACTION = auto()


class RegistrationMethod(Enum):
    """Enum defining various image registration methods."""
    RIGID = auto()
    AFFINE = auto()
    DEFORMABLE = auto()
    LANDMARK_BASED = auto()
    MANUAL = auto()


class ImageFusion:
    """
    Class for managing fusion of different image modalities.
    
    This class provides methods for fusion of different imaging modalities
    such as CT-MRI, CT-PET, etc. using various fusion methods.
    """
    
    def __init__(self, primary_image: Optional[Image] = None):
        """
        Initialize the image fusion with an optional primary image.
        
        Parameters
        ----------
        primary_image : Optional[Image]
            Primary image (usually CT) for fusion
        """
        self.primary_image = primary_image
        self.secondary_images: Dict[str, Image] = {}
        self.registration_transforms: Dict[str, Any] = {}
        self.fusion_params: Dict[str, Dict[str, Any]] = {}
    
    def set_primary_image(self, image: Image) -> None:
        """
        Set the primary image for fusion.
        
        Parameters
        ----------
        image : Image
            Primary image (usually CT) for fusion
        """
        self.primary_image = image
    
    def add_secondary_image(self, image: Image, modality: str) -> None:
        """
        Add a secondary image for fusion.
        
        Parameters
        ----------
        image : Image
            Secondary image (MRI, PET, etc.)
        modality : str
            Modality of the secondary image
        """
        self.secondary_images[modality] = image
        
        # Initialize fusion parameters with defaults
        self.fusion_params[modality] = {
            'method': FusionMethod.ALPHA_BLENDING,
            'alpha': 0.5,
            'colormap': 'jet',
            'window_center': None,
            'window_width': None,
            'checkerboard_size': 10
        }
    
    def remove_secondary_image(self, modality: str) -> None:
        """
        Remove a secondary image from fusion.
        
        Parameters
        ----------
        modality : str
            Modality of the secondary image to remove
        """
        if modality in self.secondary_images:
            del self.secondary_images[modality]
            
        if modality in self.fusion_params:
            del self.fusion_params[modality]
            
        if modality in self.registration_transforms:
            del self.registration_transforms[modality]
    
    def set_fusion_parameters(self, modality: str, parameters: Dict[str, Any]) -> None:
        """
        Set fusion parameters for a specific modality.
        
        Parameters
        ----------
        modality : str
            Image modality
        parameters : Dict[str, Any]
            Fusion parameters
        """
        if modality not in self.secondary_images:
            raise ValueError(f"No secondary image with modality {modality}")
        
        # Update existing parameters
        self.fusion_params[modality].update(parameters)
    
    def register_images(self, 
                      modality: str, 
                      method: RegistrationMethod = RegistrationMethod.RIGID,
                      parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Register secondary image to primary image.
        
        Parameters
        ----------
        modality : str
            Secondary image modality
        method : RegistrationMethod
            Registration method
        parameters : Optional[Dict[str, Any]]
            Registration parameters
            
        Raises
        ------
        ValueError
            If primary or secondary image is not set
        FusionError
            If registration fails
        """
        if self.primary_image is None:
            raise ValueError("Primary image not set")
            
        if modality not in self.secondary_images:
            raise ValueError(f"No secondary image with modality {modality}")
        
        secondary_image = self.secondary_images[modality]
        
        # Initialize parameters if not provided
        if parameters is None:
            parameters = {}
        
        # Get SimpleITK images
        fixed_image = sitk.GetImageFromArray(self.primary_image.data)
        fixed_image.SetSpacing(self.primary_image.spacing)
        fixed_image.SetOrigin(self.primary_image.origin)
        
        moving_image = sitk.GetImageFromArray(secondary_image.data)
        moving_image.SetSpacing(secondary_image.spacing)
        moving_image.SetOrigin(secondary_image.origin)
        
        try:
            # Perform registration based on the selected method
            if method == RegistrationMethod.RIGID:
                transform = self._rigid_registration(fixed_image, moving_image, parameters)
            elif method == RegistrationMethod.AFFINE:
                transform = self._affine_registration(fixed_image, moving_image, parameters)
            elif method == RegistrationMethod.DEFORMABLE:
                transform = self._deformable_registration(fixed_image, moving_image, parameters)
            elif method == RegistrationMethod.LANDMARK_BASED:
                # This would require landmarks in the parameters
                if 'landmarks' not in parameters:
                    raise ValueError("Landmark-based registration requires landmarks")
                transform = self._landmark_registration(fixed_image, moving_image, parameters['landmarks'])
            elif method == RegistrationMethod.MANUAL:
                # Manual registration uses a provided transform matrix
                if 'transform_matrix' not in parameters:
                    raise ValueError("Manual registration requires a transform matrix")
                transform = parameters['transform_matrix']
            else:
                raise ValueError(f"Unsupported registration method: {method}")
            
            # Store the transform
            self.registration_transforms[modality] = transform
            
        except Exception as e:
            logger.error(f"Image registration failed: {str(e)}")
            raise FusionError(f"Image registration failed: {str(e)}")
    
    def _rigid_registration(self, 
                          fixed_image: sitk.Image, 
                          moving_image: sitk.Image,
                          parameters: Dict[str, Any]) -> sitk.Transform:
        """
        Perform rigid registration between two images.
        
        Parameters
        ----------
        fixed_image : sitk.Image
            Fixed image (primary)
        moving_image : sitk.Image
            Moving image (secondary)
        parameters : Dict[str, Any]
            Registration parameters
            
        Returns
        -------
        sitk.Transform
            Registration transform
        """
        # Get parameters
        max_iterations = parameters.get('max_iterations', 100)
        learn_rate = parameters.get('learning_rate', 0.01)
        
        # Initialize registration
        registration_method = sitk.ImageRegistrationMethod()
        
        # Set up similarity metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.2)
        
        # Set optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=learn_rate,
            numberOfIterations=max_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        
        # Set interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Set initial transform as rigid
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        # Execute the registration
        final_transform = registration_method.Execute(fixed_image, moving_image)
        
        return final_transform
    
    def _affine_registration(self, 
                           fixed_image: sitk.Image, 
                           moving_image: sitk.Image,
                           parameters: Dict[str, Any]) -> sitk.Transform:
        """
        Perform affine registration between two images.
        
        Parameters
        ----------
        fixed_image : sitk.Image
            Fixed image (primary)
        moving_image : sitk.Image
            Moving image (secondary)
        parameters : Dict[str, Any]
            Registration parameters
            
        Returns
        -------
        sitk.Transform
            Registration transform
        """
        # Similar to rigid registration but using AffineTransform
        max_iterations = parameters.get('max_iterations', 200)
        learn_rate = parameters.get('learning_rate', 0.1)
        
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.2)
        
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=learn_rate,
            numberOfIterations=max_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.AffineTransform(3), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        final_transform = registration_method.Execute(fixed_image, moving_image)
        
        return final_transform
    
    def _deformable_registration(self, 
                               fixed_image: sitk.Image, 
                               moving_image: sitk.Image,
                               parameters: Dict[str, Any]) -> sitk.Transform:
        """
        Perform deformable registration between two images.
        
        Parameters
        ----------
        fixed_image : sitk.Image
            Fixed image (primary)
        moving_image : sitk.Image
            Moving image (secondary)
        parameters : Dict[str, Any]
            Registration parameters
            
        Returns
        -------
        sitk.Transform
            Registration transform
        """
        # This is a simplified version - deformable registration is complex
        # First do a rigid registration to get a good initial alignment
        rigid_transform = self._rigid_registration(fixed_image, moving_image, parameters)
        
        # Apply rigid transform to get better starting point for deformable
        moving_resampled = sitk.Resample(
            moving_image, 
            fixed_image, 
            rigid_transform, 
            sitk.sitkLinear, 
            0.0, 
            moving_image.GetPixelID()
        )
        
        # Setup BSpline transform
        transform_domain_mesh_size = parameters.get('mesh_size', [8, 8, 8])
        transform_domain_physical_dimensions = [
            fixed_image.GetSize()[i] * fixed_image.GetSpacing()[i] for i in range(3)
        ]
        transform_domain_origin = fixed_image.GetOrigin()
        
        bspline_transform = sitk.BSplineTransformInitializer(
            fixed_image, 
            transform_domain_mesh_size
        )
        
        # Setup registration method
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.2)
        
        registration_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=100,
            maximumNumberOfCorrections=10,
            maximumNumberOfFunctionEvaluations=100,
            costFunctionConvergenceFactor=1e7
        )
        
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetInitialTransform(bspline_transform, inPlace=False)
        
        # Add regularization
        registration_method.SetOptimizerWeights([1, 0.1, 0.1])
        
        # Execute the registration
        final_transform = registration_method.Execute(fixed_image, moving_resampled)
        
        # Combine transforms (rigid + bspline)
        composite_transform = sitk.CompositeTransform(3)
        composite_transform.AddTransform(rigid_transform)
        composite_transform.AddTransform(final_transform)
        
        return composite_transform
    
    def _landmark_registration(self, 
                             fixed_image: sitk.Image, 
                             moving_image: sitk.Image,
                             landmarks: List[Tuple[List[float], List[float]]]) -> sitk.Transform:
        """
        Perform landmark-based registration.
        
        Parameters
        ----------
        fixed_image : sitk.Image
            Fixed image (primary)
        moving_image : sitk.Image
            Moving image (secondary)
        landmarks : List[Tuple[List[float], List[float]]]
            List of landmark point pairs [(fixed_point, moving_point), ...]
            
        Returns
        -------
        sitk.Transform
            Registration transform
        """
        if len(landmarks) < 3:
            raise ValueError("At least 3 landmark pairs are required for landmark-based registration")
        
        # Extract landmark points
        fixed_points = [p[0] for p in landmarks]
        moving_points = [p[1] for p in landmarks]
        
        # Convert to numpy arrays
        fixed_points_np = np.array(fixed_points)
        moving_points_np = np.array(moving_points)
        
        # Create landmark transform
        landmark_transform = sitk.LandmarkBasedTransformInitializer(
            sitk.AffineTransform(3), 
            fixed_points, 
            moving_points
        )
        
        return landmark_transform
    
    def apply_transform(self, modality: str) -> Image:
        """
        Apply registration transform to secondary image.
        
        Parameters
        ----------
        modality : str
            Secondary image modality
            
        Returns
        -------
        Image
            Transformed secondary image
            
        Raises
        ------
        ValueError
            If primary or secondary image is not set
            If no transform exists for the given modality
        """
        if self.primary_image is None:
            raise ValueError("Primary image not set")
            
        if modality not in self.secondary_images:
            raise ValueError(f"No secondary image with modality {modality}")
            
        if modality not in self.registration_transforms:
            raise ValueError(f"No transform available for modality {modality}")
        
        secondary_image = self.secondary_images[modality]
        transform = self.registration_transforms[modality]
        
        # Get SimpleITK images
        fixed_image = sitk.GetImageFromArray(self.primary_image.data)
        fixed_image.SetSpacing(self.primary_image.spacing)
        fixed_image.SetOrigin(self.primary_image.origin)
        
        moving_image = sitk.GetImageFromArray(secondary_image.data)
        moving_image.SetSpacing(secondary_image.spacing)
        moving_image.SetOrigin(secondary_image.origin)
        
        # Resample moving image using the transform
        resampled_image = sitk.Resample(
            moving_image, 
            fixed_image, 
            transform, 
            sitk.sitkLinear, 
            0.0, 
            moving_image.GetPixelID()
        )
        
        # Convert back to numpy array
        resampled_data = sitk.GetArrayFromImage(resampled_image)
        
        # Create new Image object
        result = Image(
            data=resampled_data,
            spacing=self.primary_image.spacing,
            origin=self.primary_image.origin,
            modality=secondary_image.modality,
            metadata=secondary_image.metadata.copy()
        )
        
        # Update metadata
        result.metadata.update({
            'registered': True,
            'primary_image_uid': self.primary_image.metadata.get('SOPInstanceUID', ''),
            'registration_method': 'transform_applied'
        })
        
        return result
    
    def create_fused_image(self, modality: str, slice_index: int = 0) -> np.ndarray:
        """
        Create a fused image for visualization.
        
        Parameters
        ----------
        modality : str
            Secondary image modality
        slice_index : int, optional
            Index of the slice to fuse (for 3D images)
            
        Returns
        -------
        np.ndarray
            Fused image data
            
        Raises
        ------
        ValueError
            If primary or secondary image is not set
        FusionError
            If fusion fails
        """
        if self.primary_image is None:
            raise ValueError("Primary image not set")
            
        if modality not in self.secondary_images:
            raise ValueError(f"No secondary image with modality {modality}")
        
        # Get registered secondary image
        if modality in self.registration_transforms:
            secondary_image = self.apply_transform(modality)
        else:
            secondary_image = self.secondary_images[modality]
            # Check if dimensions match
            if self.primary_image.data.shape != secondary_image.data.shape:
                raise FusionError("Primary and secondary images must have the same dimensions for fusion")
        
        # Get fusion parameters
        params = self.fusion_params[modality]
        method = params['method']
        
        # Extract slice from 3D volumes
        if self.primary_image.data.ndim == 3:
            primary_slice = self.primary_image.data[slice_index, :, :]
        else:
            primary_slice = self.primary_image.data
            
        if secondary_image.data.ndim == 3:
            secondary_slice = secondary_image.data[slice_index, :, :]
        else:
            secondary_slice = secondary_image.data
        
        # Normalize image data for visualization
        p_min, p_max = np.min(primary_slice), np.max(primary_slice)
        s_min, s_max = np.min(secondary_slice), np.max(secondary_slice)
        
        primary_normalized = (primary_slice - p_min) / (p_max - p_min) if p_max > p_min else primary_slice
        secondary_normalized = (secondary_slice - s_min) / (s_max - s_min) if s_max > s_min else secondary_slice
        
        try:
            # Apply fusion method
            if method == FusionMethod.ALPHA_BLENDING:
                alpha = params.get('alpha', 0.5)
                fused_image = self._alpha_blend(primary_normalized, secondary_normalized, alpha)
            
            elif method == FusionMethod.CHECKERBOARD:
                size = params.get('checkerboard_size', 10)
                fused_image = self._checkerboard(primary_normalized, secondary_normalized, size)
            
            elif method == FusionMethod.COLORMAP_OVERLAY:
                colormap = params.get('colormap', 'jet')
                alpha = params.get('alpha', 0.7)
                fused_image = self._colormap_overlay(primary_normalized, secondary_normalized, colormap, alpha)
            
            elif method == FusionMethod.WINDOW_LEVEL:
                window_center = params.get('window_center', 0.5)
                window_width = params.get('window_width', 1.0)
                fused_image = self._window_level(primary_normalized, secondary_normalized, window_center, window_width)
            
            elif method == FusionMethod.SUBTRACTION:
                fused_image = self._subtraction(primary_normalized, secondary_normalized)
            
            elif method == FusionMethod.WINDOWED_SUBTRACTION:
                window_center = params.get('window_center', 0.0)
                window_width = params.get('window_width', 0.5)
                fused_image = self._windowed_subtraction(primary_normalized, secondary_normalized, window_center, window_width)
            
            else:
                raise ValueError(f"Unsupported fusion method: {method}")
            
            return fused_image
            
        except Exception as e:
            logger.error(f"Image fusion failed: {str(e)}")
            raise FusionError(f"Image fusion failed: {str(e)}")
    
    def _alpha_blend(self, img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
        """
        Blend two images using alpha blending.
        
        Parameters
        ----------
        img1 : np.ndarray
            First image (normalized)
        img2 : np.ndarray
            Second image (normalized)
        alpha : float
            Blending factor (0-1)
            
        Returns
        -------
        np.ndarray
            Blended image
        """
        return (1 - alpha) * img1 + alpha * img2
    
    def _checkerboard(self, img1: np.ndarray, img2: np.ndarray, size: int) -> np.ndarray:
        """
        Create a checkerboard pattern from two images.
        
        Parameters
        ----------
        img1 : np.ndarray
            First image (normalized)
        img2 : np.ndarray
            Second image (normalized)
        size : int
            Size of checkerboard squares
            
        Returns
        -------
        np.ndarray
            Checkerboard image
        """
        result = np.zeros_like(img1)
        h, w = img1.shape
        
        for i in range(0, h, size):
            for j in range(0, w, size):
                if (i // size + j // size) % 2 == 0:
                    result[i:i+size, j:j+size] = img1[i:i+size, j:j+size]
                else:
                    result[i:i+size, j:j+size] = img2[i:i+size, j:j+size]
        
        return result
    
    def _colormap_overlay(self, img1: np.ndarray, img2: np.ndarray, 
                        colormap: str, alpha: float) -> np.ndarray:
        """
        Overlay a colormap of img2 onto img1.
        
        Parameters
        ----------
        img1 : np.ndarray
            First image (normalized, background)
        img2 : np.ndarray
            Second image (normalized, overlay)
        colormap : str
            Matplotlib colormap name
        alpha : float
            Opacity of the overlay (0-1)
            
        Returns
        -------
        np.ndarray
            Color overlay image (RGB)
        """
        # Create RGB version of background (grayscale)
        background = np.stack([img1] * 3, axis=-1)
        
        # Apply colormap to overlay
        cmap = mcolors.LinearSegmentedColormap(colormap, plt.get_cmap(colormap)._segmentdata)
        overlay_colored = cmap(img2)[:, :, :3]  # Remove alpha channel
        
        # Blend images
        result = (1 - alpha) * background + alpha * overlay_colored
        
        return result
    
    def _window_level(self, img1: np.ndarray, img2: np.ndarray, 
                    window_center: float, window_width: float) -> np.ndarray:
        """
        Apply windowing to blend two images.
        
        Parameters
        ----------
        img1 : np.ndarray
            First image (normalized)
        img2 : np.ndarray
            Second image (normalized)
        window_center : float
            Window center (0-1)
        window_width : float
            Window width (0-1)
            
        Returns
        -------
        np.ndarray
            Windowed image
        """
        # Calculate window boundaries
        lower = window_center - window_width/2
        upper = window_center + window_width/2
        
        # Create mask based on img1 values
        mask = np.zeros_like(img1, dtype=float)
        mask[img1 <= lower] = 0
        mask[img1 >= upper] = 1
        
        # Linear ramp for transition region
        mask[(img1 > lower) & (img1 < upper)] = (img1[(img1 > lower) & (img1 < upper)] - lower) / (upper - lower)
        
        # Blend images using the mask
        return (1 - mask) * img1 + mask * img2
    
    def _subtraction(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Subtract second image from first.
        
        Parameters
        ----------
        img1 : np.ndarray
            First image (normalized)
        img2 : np.ndarray
            Second image (normalized)
            
        Returns
        -------
        np.ndarray
            Subtraction image
        """
        # Simple subtraction with normalization
        diff = img1 - img2
        diff_min, diff_max = np.min(diff), np.max(diff)
        
        if diff_max > diff_min:
            result = (diff - diff_min) / (diff_max - diff_min)
        else:
            result = np.zeros_like(diff)
        
        return result
    
    def _windowed_subtraction(self, img1: np.ndarray, img2: np.ndarray, 
                            window_center: float, window_width: float) -> np.ndarray:
        """
        Apply windowing to subtraction image.
        
        Parameters
        ----------
        img1 : np.ndarray
            First image (normalized)
        img2 : np.ndarray
            Second image (normalized)
        window_center : float
            Window center (-1 to 1)
        window_width : float
            Window width (0-2)
            
        Returns
        -------
        np.ndarray
            Windowed subtraction image
        """
        # Calculate difference
        diff = img1 - img2
        
        # Apply windowing
        lower = window_center - window_width/2
        upper = window_center + window_width/2
        
        result = np.clip((diff - lower) / (upper - lower), 0, 1)
        
        return result


class MultiModalityViewer:
    """
    Viewer for displaying multi-modality images.
    
    This class provides functionality for visualizing multiple image
    modalities with various fusion options.
    """
    
    def __init__(self, fusion_manager: Optional[ImageFusion] = None):
        """
        Initialize the multi-modality viewer.
        
        Parameters
        ----------
        fusion_manager : Optional[ImageFusion]
            Image fusion manager
        """
        self.fusion_manager = fusion_manager or ImageFusion()
        self.current_slice = 0
        self.figures = {}
        self.axes = {}
        
    def set_fusion_manager(self, fusion_manager: ImageFusion) -> None:
        """
        Set the fusion manager.
        
        Parameters
        ----------
        fusion_manager : ImageFusion
            Image fusion manager
        """
        self.fusion_manager = fusion_manager
        
    def display_modality(self, modality: str, slice_index: Optional[int] = None) -> None:
        """
        Display a single modality.
        
        Parameters
        ----------
        modality : str
            Image modality
        slice_index : Optional[int]
            Index of the slice to display (for 3D images)
        """
        if slice_index is not None:
            self.current_slice = slice_index
            
        slice_idx = self.current_slice
        
        if modality == 'primary':
            if self.fusion_manager.primary_image is None:
                raise ValueError("Primary image not set")
                
            image = self.fusion_manager.primary_image
            title = f"Primary ({image.modality if hasattr(image, 'modality') else 'Unknown'})"
            
            if image.data.ndim == 3:
                data = image.data[slice_idx, :, :]
            else:
                data = image.data
        else:
            if modality not in self.fusion_manager.secondary_images:
                raise ValueError(f"No secondary image with modality {modality}")
                
            image = self.fusion_manager.secondary_images[modality]
            title = f"Secondary - {modality}"
            
            if image.data.ndim == 3:
                data = image.data[slice_idx, :, :]
            else:
                data = image.data
        
        try:
            import matplotlib.pyplot as plt
            
            # Create figure if it doesn't exist
            if modality not in self.figures:
                self.figures[modality] = plt.figure()
                self.axes[modality] = self.figures[modality].add_subplot(111)
                
            ax = self.axes[modality]
            fig = self.figures[modality]
            
            # Clear previous content
            ax.clear()
            
            # Display image
            im = ax.imshow(data, cmap='gray')
            ax.set_title(title)
            
            # Add colorbar
            if hasattr(fig, 'colorbar'):
                fig.colorbar.remove()
            fig.colorbar = fig.colorbar(im, ax=ax)
            
            # Update display
            fig.canvas.draw()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for display")
            
    def display_fusion(self, modality: str, slice_index: Optional[int] = None) -> None:
        """
        Display fusion of primary and secondary images.
        
        Parameters
        ----------
        modality : str
            Secondary image modality
        slice_index : Optional[int]
            Index of the slice to display (for 3D images)
        """
        if slice_index is not None:
            self.current_slice = slice_index
            
        # Create fused image
        fused_data = self.fusion_manager.create_fused_image(modality, self.current_slice)
        
        try:
            import matplotlib.pyplot as plt
            
            # Create figure if it doesn't exist
            key = f"fusion_{modality}"
            if key not in self.figures:
                self.figures[key] = plt.figure()
                self.axes[key] = self.figures[key].add_subplot(111)
                
            ax = self.axes[key]
            fig = self.figures[key]
            
            # Clear previous content
            ax.clear()
            
            # Display image
            if fused_data.ndim == 3:  # RGB image
                im = ax.imshow(fused_data)
            else:
                im = ax.imshow(fused_data, cmap='gray')
                
            fusion_method = self.fusion_manager.fusion_params[modality]['method'].name
            ax.set_title(f"Fusion: Primary + {modality} ({fusion_method})")
            
            # Add colorbar for grayscale images
            if fused_data.ndim != 3:
                if hasattr(fig, 'colorbar'):
                    fig.colorbar.remove()
                fig.colorbar = fig.colorbar(im, ax=ax)
            
            # Update display
            fig.canvas.draw()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for display")
            
    def display_linked_views(self, modalities: List[str], slice_index: Optional[int] = None) -> None:
        """
        Display linked views of multiple modalities.
        
        Parameters
        ----------
        modalities : List[str]
            List of modalities to display
        slice_index : Optional[int]
            Index of the slice to display (for 3D images)
        """
        if slice_index is not None:
            self.current_slice = slice_index
            
        try:
            import matplotlib.pyplot as plt
            
            # Create new figure
            fig, axes = plt.subplots(1, len(modalities) + 1, figsize=(5 * (len(modalities) + 1), 5))
            
            # Display primary image
            if self.fusion_manager.primary_image is None:
                raise ValueError("Primary image not set")
                
            primary_image = self.fusion_manager.primary_image
            
            if primary_image.data.ndim == 3:
                primary_data = primary_image.data[self.current_slice, :, :]
            else:
                primary_data = primary_image.data
                
            axes[0].imshow(primary_data, cmap='gray')
            axes[0].set_title("Primary")
            
            # Display secondary images and fusions
            for i, modality in enumerate(modalities):
                if modality not in self.fusion_manager.secondary_images:
                    logger.warning(f"Modality {modality} not found, skipping")
                    continue
                    
                fused_data = self.fusion_manager.create_fused_image(modality, self.current_slice)
                
                if fused_data.ndim == 3:  # RGB image
                    axes[i + 1].imshow(fused_data)
                else:
                    axes[i + 1].imshow(fused_data, cmap='gray')
                    
                fusion_method = self.fusion_manager.fusion_params[modality]['method'].name
                axes[i + 1].set_title(f"Fusion: {modality} ({fusion_method})")
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for display")
    
    def next_slice(self) -> None:
        """Move to the next slice."""
        # Check if primary image exists and is 3D
        if (self.fusion_manager.primary_image is not None and 
            self.fusion_manager.primary_image.data.ndim == 3):
            max_slice = self.fusion_manager.primary_image.data.shape[0] - 1
            self.current_slice = min(self.current_slice + 1, max_slice)
    
    def previous_slice(self) -> None:
        """Move to the previous slice."""
        # Check if primary image exists
        if self.fusion_manager.primary_image is not None:
            self.current_slice = max(self.current_slice - 1, 0)
    
    def goto_slice(self, slice_index: int) -> None:
        """
        Go to a specific slice.
        
        Parameters
        ----------
        slice_index : int
            Index of the slice to display
        """
        # Check if primary image exists and is 3D
        if (self.fusion_manager.primary_image is not None and 
            self.fusion_manager.primary_image.data.ndim == 3):
            max_slice = self.fusion_manager.primary_image.data.shape[0] - 1
            self.current_slice = max(0, min(slice_index, max_slice))

# Import at the end to avoid circular imports
import matplotlib.pyplot as plt
