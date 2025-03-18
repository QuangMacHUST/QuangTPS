#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual Segmentation Editor for QuangTPS.

This module provides a comprehensive editor for manual segmentation and contouring
of anatomical structures and tumors in radiotherapy treatment planning.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import SimpleITK as sitk
from scipy import ndimage

from quangtps.core.exceptions import ValidationError
from quangtps.segmentation.manual_segmentation.drawing_tools import DrawingToolManager
from quangtps.segmentation.structures.structure_set import StructureSet
from quangtps.segmentation.structures.structure import Structure
from quangtps.segmentation.contour.contour_tools import ContourTool

logger = logging.getLogger(__name__)


class ManualSegmentationEditor:
    """
    Editor for manual segmentation and contour editing.
    
    This class provides a comprehensive interface for manual segmentation
    and contouring, integrating drawing tools and structure management.
    """
    
    def __init__(self):
        """Initialize the manual segmentation editor."""
        self.drawing_tool_manager = DrawingToolManager()
        self.active_structure = None
        self.structure_set = None
        self.image_data = None
        self.spacing = None
        self.origin = None
        self.contour_tool = ContourTool()
        
        # Set up callback for all drawing tools
        self.drawing_tool_manager.set_callback_for_all_tools(self._on_contour_update)
    
    def set_image_data(self, image_data: np.ndarray, spacing: Tuple[float, float, float] = None,
                      origin: Tuple[float, float, float] = None):
        """
        Set the image data for segmentation.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data
        spacing : Tuple[float, float, float], optional
            Image spacing in mm
        origin : Tuple[float, float, float], optional
            Image origin coordinates
        """
        self.image_data = image_data
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.origin = origin if origin is not None else (0.0, 0.0, 0.0)
        
        # Initialize structure set if not already set
        if self.structure_set is None:
            self.structure_set = StructureSet()
    
    def set_structure_set(self, structure_set: StructureSet):
        """
        Set the structure set for editing.
        
        Parameters
        ----------
        structure_set : StructureSet
            Structure set to edit
        """
        self.structure_set = structure_set
    
    def create_new_structure(self, id: str, name: str, color: List[int] = None, 
                           type: str = "ORGAN") -> str:
        """
        Create a new empty structure.
        
        Parameters
        ----------
        id : str
            Structure ID
        name : str
            Structure name
        color : List[int], optional
            RGB color for the structure
        type : str, optional
            Structure type
            
        Returns
        -------
        str
            ID of the created structure
        """
        if self.image_data is None:
            raise ValidationError("Image data must be set before creating structures")
        
        if self.structure_set is None:
            self.structure_set = StructureSet()
        
        # Create an empty mask
        mask = np.zeros_like(self.image_data, dtype=np.uint8)
        
        # Set default color if not provided
        if color is None:
            color = [255, 0, 0]  # Red
        
        # Create the structure
        structure = Structure(
            id=id,
            name=name,
            mask=mask,
            color=color,
            type=type,
            spacing=self.spacing,
            origin=self.origin
        )
        
        # Add the structure to the set
        self.structure_set.add_structure(structure)
        
        # Set as active structure
        self.set_active_structure(id)
        
        return id
    
    def set_active_structure(self, structure_id: str) -> bool:
        """
        Set the active structure for editing.
        
        Parameters
        ----------
        structure_id : str
            ID of the structure to set as active
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.structure_set is None:
            logger.warning("No structure set available")
            return False
        
        # Get the structure from the set
        structure = self.structure_set.get_structure(structure_id)
        if structure is None:
            logger.warning(f"Structure with ID {structure_id} not found")
            return False
        
        # Set as active structure
        self.active_structure = structure
        logger.info(f"Set active structure to: {structure.name} (ID: {structure.id})")
        
        return True
    
    def get_active_structure(self) -> Optional[Structure]:
        """
        Get the currently active structure.
        
        Returns
        -------
        Structure or None
            The active structure, or None if no structure is active
        """
        return self.active_structure
    
    def get_structure_mask(self, structure_id: str) -> Optional[np.ndarray]:
        """
        Get the mask for a specific structure.
        
        Parameters
        ----------
        structure_id : str
            ID of the structure
            
        Returns
        -------
        np.ndarray or None
            Mask of the structure, or None if not found
        """
        if self.structure_set is None:
            return None
        
        structure = self.structure_set.get_structure(structure_id)
        if structure is None:
            return None
        
        return structure.mask
    
    def delete_structure(self, structure_id: str) -> bool:
        """
        Delete a structure from the structure set.
        
        Parameters
        ----------
        structure_id : str
            ID of the structure to delete
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.structure_set is None:
            return False
        
        # Remove the structure from the set
        result = self.structure_set.remove_structure(structure_id)
        
        # Reset active structure if it was deleted
        if result and self.active_structure and self.active_structure.id == structure_id:
            self.active_structure = None
        
        return result
    
    def activate_tool(self, tool_name: str) -> bool:
        """
        Activate a drawing tool by name.
        
        Parameters
        ----------
        tool_name : str
            Name of the tool to activate
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        return self.drawing_tool_manager.activate_tool(tool_name)
    
    def on_mouse_down(self, x: int, y: int, slice_idx: int):
        """
        Handle mouse down event.
        
        Parameters
        ----------
        x : int
            X coordinate
        y : int
            Y coordinate
        slice_idx : int
            Slice index
        """
        if self.active_structure is None:
            logger.warning("No active structure selected for editing")
            return
        
        self.drawing_tool_manager.on_mouse_down(x, y, slice_idx)
    
    def on_mouse_move(self, x: int, y: int, slice_idx: int):
        """
        Handle mouse move event.
        
        Parameters
        ----------
        x : int
            X coordinate
        y : int
            Y coordinate
        slice_idx : int
            Slice index
        """
        if self.active_structure is None:
            return
        
        self.drawing_tool_manager.on_mouse_move(x, y, slice_idx)
    
    def on_mouse_up(self, x: int, y: int, slice_idx: int):
        """
        Handle mouse up event.
        
        Parameters
        ----------
        x : int
            X coordinate
        y : int
            Y coordinate
        slice_idx : int
            Slice index
        """
        if self.active_structure is None:
            return
        
        self.drawing_tool_manager.on_mouse_up(x, y, slice_idx)
    
    def _on_contour_update(self, points: List[Tuple[int, int]], slice_idx: int):
        """
        Callback for when a contour is updated by a drawing tool.
        
        Parameters
        ----------
        points : List[Tuple[int, int]]
            List of contour points
        slice_idx : int
            Slice index
        """
        if self.active_structure is None or len(points) < 2:
            return
        
        # Get current structure mask
        mask = self.active_structure.mask.copy()
        
        # Get active tool
        active_tool = self.drawing_tool_manager.get_active_tool()
        if active_tool is None:
            return
        
        # Apply the tool to modify the mask
        updated_mask = active_tool.apply_to_mask(mask, slice_idx)
        
        # Update the structure with the modified mask
        self.active_structure.mask = updated_mask
        
        # Also update in the structure set
        self.structure_set.update_structure(self.active_structure)
    
    def interpolate_contours(self, start_slice: int, end_slice: int) -> bool:
        """
        Interpolate contours between two slices.
        
        Parameters
        ----------
        start_slice : int
            Starting slice index
        end_slice : int
            Ending slice index
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.active_structure is None:
            logger.warning("No active structure selected for interpolation")
            return False
        
        if start_slice == end_slice or abs(start_slice - end_slice) <= 1:
            logger.warning("Cannot interpolate between adjacent or same slices")
            return False
        
        # Get the structure mask
        mask = self.active_structure.mask
        
        # Check if the slices have contours
        if np.max(mask[start_slice]) == 0 or np.max(mask[end_slice]) == 0:
            logger.warning("Both start and end slices must have contours for interpolation")
            return False
        
        # Sort slices
        start = min(start_slice, end_slice)
        end = max(start_slice, end_slice)
        
        # Create a temporary 3D mask with just the start and end slices
        temp_mask = np.zeros_like(mask)
        temp_mask[start] = mask[start]
        temp_mask[end] = mask[end]
        
        # Perform 3D interpolation
        for i in range(start + 1, end):
            # Calculate interpolation weight
            weight = (i - start) / (end - start)
            
            # Linear interpolation between binary masks
            interp_slice = np.round(
                (1 - weight) * mask[start] + weight * mask[end]
            ).astype(np.uint8)
            
            # Use binary morphology to clean up the interpolated mask
            interp_slice = ndimage.binary_closing(interp_slice).astype(np.uint8)
            
            # Update the mask
            mask[i] = interp_slice
        
        # Update the structure with the modified mask
        self.active_structure.mask = mask
        
        # Also update in the structure set
        self.structure_set.update_structure(self.active_structure)
        
        return True
    
    def copy_to_slice(self, source_slice: int, target_slice: int) -> bool:
        """
        Copy contour from one slice to another.
        
        Parameters
        ----------
        source_slice : int
            Source slice index
        target_slice : int
            Target slice index
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.active_structure is None:
            logger.warning("No active structure selected for copying")
            return False
        
        # Get the structure mask
        mask = self.active_structure.mask
        
        # Check if the source slice has a contour
        if np.max(mask[source_slice]) == 0:
            logger.warning("Source slice has no contour to copy")
            return False
        
        # Copy the slice
        mask[target_slice] = mask[source_slice].copy()
        
        # Update the structure with the modified mask
        self.active_structure.mask = mask
        
        # Also update in the structure set
        self.structure_set.update_structure(self.active_structure)
        
        return True
    
    def clear_slice(self, slice_idx: int) -> bool:
        """
        Clear contour from a slice.
        
        Parameters
        ----------
        slice_idx : int
            Slice index to clear
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.active_structure is None:
            logger.warning("No active structure selected for clearing")
            return False
        
        # Get the structure mask
        mask = self.active_structure.mask
        
        # Clear the slice
        mask[slice_idx] = np.zeros_like(mask[slice_idx])
        
        # Update the structure with the modified mask
        self.active_structure.mask = mask
        
        # Also update in the structure set
        self.structure_set.update_structure(self.active_structure)
        
        return True
    
    def smooth_contour(self, slice_idx: int, iterations: int = 1) -> bool:
        """
        Smooth a contour on a slice.
        
        Parameters
        ----------
        slice_idx : int
            Slice index
        iterations : int, optional
            Number of smoothing iterations
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.active_structure is None:
            logger.warning("No active structure selected for smoothing")
            return False
        
        # Get the structure mask
        mask = self.active_structure.mask
        
        # Check if the slice has a contour
        if np.max(mask[slice_idx]) == 0:
            logger.warning("Slice has no contour to smooth")
            return False
        
        # Get the slice mask
        slice_mask = mask[slice_idx]
        
        # Apply morphological operations for smoothing
        smoothed_mask = ndimage.binary_closing(slice_mask, iterations=iterations)
        smoothed_mask = ndimage.binary_opening(smoothed_mask, iterations=iterations)
        
        # Gaussian filter for additional smoothing
        smoothed_mask = ndimage.gaussian_filter(smoothed_mask.astype(np.float32), sigma=0.5)
        smoothed_mask = (smoothed_mask > 0.5).astype(np.uint8)
        
        # Update the mask
        mask[slice_idx] = smoothed_mask
        
        # Update the structure with the modified mask
        self.active_structure.mask = mask
        
        # Also update in the structure set
        self.structure_set.update_structure(self.active_structure)
        
        return True
    
    def get_structure_set(self) -> Optional[StructureSet]:
        """
        Get the current structure set.
        
        Returns
        -------
        StructureSet or None
            The current structure set, or None if not set
        """
        return self.structure_set
    
    def extract_contours_from_mask(self, structure_id: str = None) -> Dict[int, List[np.ndarray]]:
        """
        Extract contour points from the mask for visualization.
        
        Parameters
        ----------
        structure_id : str, optional
            ID of the structure to extract contours from. If None, uses active structure.
            
        Returns
        -------
        Dict[int, List[np.ndarray]]
            Dictionary mapping slice indices to lists of contour points
        """
        # Get the structure
        structure = None
        if structure_id is not None:
            if self.structure_set is not None:
                structure = self.structure_set.get_structure(structure_id)
        else:
            structure = self.active_structure
        
        if structure is None:
            logger.warning("No structure specified for contour extraction")
            return {}
        
        # Get the mask
        mask = structure.mask
        
        # Extract contours for each slice
        contours_by_slice = {}
        
        for slice_idx in range(mask.shape[0]):
            slice_mask = mask[slice_idx]
            
            # Skip empty slices
            if np.max(slice_mask) == 0:
                continue
            
            # Extract contours using OpenCV
            import cv2
            contours, _ = cv2.findContours(
                slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Convert to list of numpy arrays
            contours_list = [contour.squeeze() for contour in contours if contour.size > 2]
            
            # Store in dictionary
            if contours_list:
                contours_by_slice[slice_idx] = contours_list
        
        return contours_by_slice
    
    def save_structure_set(self, filename: str) -> bool:
        """
        Save the structure set to a file.
        
        Parameters
        ----------
        filename : str
            Path to save the structure set
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.structure_set is None:
            logger.warning("No structure set to save")
            return False
        
        try:
            # Convert to SITK image
            structures_dict = {}
            for structure in self.structure_set.get_all_structures():
                # Create SITK image from mask
                mask_sitk = sitk.GetImageFromArray(structure.mask)
                mask_sitk.SetSpacing(structure.spacing)
                mask_sitk.SetOrigin(structure.origin)
                
                # Store structure info
                structures_dict[structure.id] = {
                    'image': mask_sitk,
                    'name': structure.name,
                    'color': structure.color,
                    'type': structure.type
                }
            
            # Save as SITK image series
            writer = sitk.ImageFileWriter()
            writer.SetFileName(filename)
            writer.Execute(sitk.Image3D(structures_dict))
            
            logger.info(f"Structure set saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving structure set: {str(e)}")
            return False
    
    def load_structure_set(self, filename: str) -> bool:
        """
        Load a structure set from a file.
        
        Parameters
        ----------
        filename : str
            Path to the structure set file
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Read SITK image
            reader = sitk.ImageFileReader()
            reader.SetFileName(filename)
            image = reader.Execute()
            
            # Create a new structure set
            structure_set = StructureSet()
            
            # Extract structures
            for i in range(image.GetSize()[3]):
                # Extract the 3D structure mask
                structure_image = sitk.Extract(
                    image, 
                    [image.GetSize()[0], image.GetSize()[1], image.GetSize()[2], 0], 
                    [0, 0, 0, i]
                )
                
                # Convert to numpy array
                mask = sitk.GetArrayFromImage(structure_image)
                
                # Get metadata
                structure_id = f"structure_{i}"
                structure_name = structure_id
                structure_color = [255, 0, 0]  # Default to red
                structure_type = "ORGAN"
                
                # Create structure
                structure = Structure(
                    id=structure_id,
                    name=structure_name,
                    mask=mask,
                    color=structure_color,
                    type=structure_type,
                    spacing=structure_image.GetSpacing(),
                    origin=structure_image.GetOrigin()
                )
                
                # Add to structure set
                structure_set.add_structure(structure)
            
            # Set as current structure set
            self.structure_set = structure_set
            
            # Set first structure as active if available
            structures = structure_set.get_all_structures()
            if structures:
                self.active_structure = structures[0]
            
            logger.info(f"Structure set loaded from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading structure set: {str(e)}")
            return False
    
    def convert_mask_to_contours(self, mask: np.ndarray, slice_idx: int) -> List[np.ndarray]:
        """
        Convert a binary mask to contour points for a specific slice.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask
        slice_idx : int
            Slice index
            
        Returns
        -------
        List[np.ndarray]
            List of contour points arrays
        """
        import cv2
        
        # Get the slice mask
        slice_mask = mask[slice_idx]
        
        # Find contours
        contours, _ = cv2.findContours(
            slice_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert to list of numpy arrays and simplify
        contours_list = []
        for contour in contours:
            if contour.size > 2:
                # Simplify contour if it has many points
                if contour.shape[0] > 100:
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    contours_list.append(approx.squeeze())
                else:
                    contours_list.append(contour.squeeze())
        
        return contours_list
    
    def convert_contours_to_mask(self, contours: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert contour points to a binary mask.
        
        Parameters
        ----------
        contours : List[np.ndarray]
            List of contour points arrays
        shape : Tuple[int, int]
            Shape of the output mask
            
        Returns
        -------
        np.ndarray
            Binary mask
        """
        import cv2
        
        # Create an empty mask
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Draw filled contours
        for contour in contours:
            # Ensure contour is in the right format for OpenCV
            if len(contour.shape) == 1:
                # Single point, can't draw a contour
                continue
            elif len(contour.shape) == 2:
                if contour.shape[0] < 3:
                    # Need at least 3 points for a contour
                    continue
                cv2.fillPoly(mask, [contour.astype(np.int32)], color=1)
            else:
                # Reshape if necessary
                reshaped = contour.reshape(-1, 2).astype(np.int32)
                if reshaped.shape[0] < 3:
                    # Need at least 3 points for a contour
                    continue
                cv2.fillPoly(mask, [reshaped], color=1)
        
        return mask
