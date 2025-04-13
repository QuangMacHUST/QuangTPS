#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cấu trúc giải phẫu cho QuangTPS.

Module này cung cấp các lớp và hàm để làm việc với các cấu trúc giải phẫu trong
hệ thống lập kế hoạch điều trị xạ trị QuangTPS.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Set, Union
import numpy as np
from enum import Enum, auto
from datetime import datetime
import uuid

# Import trực tiếp từ imaging.structures để tái sử dụng mã đã có
from quangtps.imaging.structures import (
    Structure as ImageStructure,
    StructureSet as ImageStructureSet,
    StructureType as ImageStructureType,
    StructureColor
)

logger = logging.getLogger(__name__)

# Định nghĩa lại Structure và StructureSet để sử dụng ở mức cao hơn
class Structure(ImageStructure):
    """
    Lớp Structure được kế thừa từ imaging.structures.Structure.
    
    Lớp này cung cấp tất cả các chức năng từ ImageStructure và có thể được mở rộng 
    thêm các chức năng cần thiết cho module core.
    """
    
    def __init__(self, 
                id: str, 
                name: str, 
                structure_type: ImageStructureType = ImageStructureType.UNDEFINED,
                color: Optional[Tuple[int, int, int]] = None, 
                parent_set_id: Optional[str] = None):
        """
        Khởi tạo cấu trúc.
        
        Parameters
        ----------
        id : str
            ID duy nhất của cấu trúc
        name : str
            Tên mô tả của cấu trúc
        structure_type : StructureType, optional
            Loại cấu trúc, mặc định là UNDEFINED
        color : Optional[Tuple[int, int, int]], optional
            Màu RGB của cấu trúc, mặc định dựa vào loại cấu trúc
        parent_set_id : Optional[str], optional
            ID của tập cấu trúc chứa cấu trúc này
        """
        super().__init__(id, name, structure_type, color, parent_set_id)

    @staticmethod
    def from_image_structure(image_structure: ImageStructure) -> 'Structure':
        """
        Tạo một đối tượng Structure từ một đối tượng ImageStructure.
        
        Parameters
        ----------
        image_structure : ImageStructure
            Đối tượng ImageStructure nguồn
            
        Returns
        -------
        Structure
            Đối tượng Structure mới
        """
        structure = Structure(
            id=image_structure.id,
            name=image_structure.name,
            structure_type=image_structure.structure_type,
            color=image_structure.color,
            parent_set_id=image_structure.parent_set_id
        )
        
        # Sao chép các thuộc tính khác
        structure.description = image_structure.description
        structure.creation_date = image_structure.creation_date
        structure.modification_date = image_structure.modification_date
        structure.created_by = image_structure.created_by
        structure.contours = image_structure.contours
        structure.volume_cc = image_structure.volume_cc
        structure.surface_area_cm2 = image_structure.surface_area_cm2
        structure.display_opacity = image_structure.display_opacity
        structure.is_visible = image_structure.is_visible
        structure.is_locked = image_structure.is_locked
        structure.tags = image_structure.tags.copy() if hasattr(image_structure, 'tags') else set()
        structure.metadata = image_structure.metadata.copy() if hasattr(image_structure, 'metadata') else {}
        
        return structure


class StructureSet(ImageStructureSet):
    """
    Lớp StructureSet được kế thừa từ imaging.structures.StructureSet.
    
    Lớp này cung cấp tất cả các chức năng từ ImageStructureSet và có thể được mở rộng 
    thêm các chức năng cần thiết cho module core.
    """
    
    def __init__(self, id: str, name: str, series_id: Optional[str] = None):
        """
        Khởi tạo tập cấu trúc.
        
        Parameters
        ----------
        id : str
            ID duy nhất của tập cấu trúc
        name : str
            Tên mô tả của tập cấu trúc
        series_id : Optional[str], optional
            ID của chuỗi hình ảnh liên kết
        """
        super().__init__(id, name, series_id)
        
    @staticmethod
    def from_image_structure_set(image_structure_set: ImageStructureSet) -> 'StructureSet':
        """
        Tạo một đối tượng StructureSet từ một đối tượng ImageStructureSet.
        
        Parameters
        ----------
        image_structure_set : ImageStructureSet
            Đối tượng ImageStructureSet nguồn
            
        Returns
        -------
        StructureSet
            Đối tượng StructureSet mới
        """
        structure_set = StructureSet(
            id=image_structure_set.id,
            name=image_structure_set.name,
            series_id=image_structure_set.series_id
        )
        
        # Sao chép các thuộc tính khác
        structure_set.description = image_structure_set.description
        structure_set.creation_date = image_structure_set.creation_date
        structure_set.modification_date = image_structure_set.modification_date
        structure_set.created_by = image_structure_set.created_by
        
        # Chuyển đổi các cấu trúc
        for structure in image_structure_set.structures:
            core_structure = Structure.from_image_structure(structure)
            structure_set.add_structure(core_structure)
        
        return structure_set


# Tái export các lớp và hằng từ imaging.structures để đảm bảo tính tương thích
StructureType = ImageStructureType

# Đảm bảo các lớp này có sẵn khi import từ core.structures
__all__ = [
    'Structure',
    'StructureSet',
    'StructureType',
    'StructureColor'
]

class StructureType(Enum):
    """Enumeration of structure types."""
    PTV = "PTV"
    CTV = "CTV"
    GTV = "GTV"
    OAR = "OAR"  # Organ at Risk
    EXTERNAL = "EXTERNAL"  # Body contour
    PRV = "PRV"  # Planning organ at Risk Volume
    CONTROL = "CONTROL"  # Control structure for optimization
    CUSTOM = "CUSTOM"  # Custom structure


class Structure:
    """
    Class representing an anatomical structure (contour).
    
    This includes target volumes (PTV, CTV, GTV), organs at risk (OARs),
    and other structure types used in radiotherapy planning.
    
    Attributes:
        id: Unique identifier for the structure
        name: Name of the structure
        type: Type of structure (PTV, OAR, etc.)
        color: Color used for displaying the structure
        volume: Volume of the structure in cc
        mesh: 3D mesh representation (for visualization)
        contours: Contour points by slice
    """
    
    def __init__(self, id: str, name: str, type: Union[StructureType, str] = StructureType.CUSTOM):
        """
        Initialize a structure.
        
        Args:
            id: Unique identifier for the structure
            name: Name of the structure
            type: Type of structure (PTV, OAR, etc.)
        """
        self.id = id
        self.name = name
        
        # Convert string type to enum if needed
        if isinstance(type, str):
            try:
                self.type = StructureType(type)
            except ValueError:
                self.type = StructureType.CUSTOM
        else:
            self.type = type
        
        # Assign a default color based on type
        self.color = self._get_default_color()
        
        # Physical properties
        self.volume = 0.0  # Volume in cc
        
        # Geometry
        self.mesh = None  # 3D mesh for visualization
        self.contours = {}  # Dictionary of slice -> list of contour points
    
    def _get_default_color(self) -> Tuple[float, float, float]:
        """
        Get the default color for this structure type.
        
        Returns:
            RGB color tuple with values from 0-1
        """
        if self.type == StructureType.PTV:
            return (1.0, 0.0, 0.0)  # Red
        elif self.type == StructureType.CTV:
            return (1.0, 0.5, 0.0)  # Orange
        elif self.type == StructureType.GTV:
            return (1.0, 0.0, 0.5)  # Pink
        elif self.type == StructureType.OAR:
            return (0.0, 0.0, 1.0)  # Blue
        elif self.type == StructureType.EXTERNAL:
            return (0.0, 1.0, 0.0)  # Green
        elif self.type == StructureType.PRV:
            return (0.5, 0.0, 0.5)  # Purple
        elif self.type == StructureType.CONTROL:
            return (0.0, 1.0, 1.0)  # Cyan
        else:
            return (0.5, 0.5, 0.5)  # Gray
    
    def set_contours(self, contours: Dict[int, List[List[Tuple[float, float]]]]):
        """
        Set contour data for the structure.
        
        Args:
            contours: Dictionary of {slice_index: [contour_points]}
                      where contour_points is a list of (x,y) tuples
        """
        self.contours = contours
        
        # Calculate volume based on contours
        self._calculate_volume()
    
    def _calculate_volume(self):
        """Calculate the volume of the structure based on contours."""
        # In a real implementation, this would calculate the volume
        # based on the contour data using a proper algorithm
        
        # For demonstration, we'll set some reasonable values
        if self.type == StructureType.PTV:
            self.volume = 100.0  # 100 cc for PTV
        elif self.type == StructureType.CTV:
            self.volume = 80.0  # 80 cc for CTV
        elif self.type == StructureType.GTV:
            self.volume = 50.0  # 50 cc for GTV
        elif self.type == StructureType.EXTERNAL:
            self.volume = 30000.0  # 30000 cc for body
        elif "Lung" in self.name:
            self.volume = 1500.0  # 1500 cc for lung
        elif "Heart" in self.name:
            self.volume = 800.0  # 800 cc for heart
        elif "Spinal" in self.name or "Cord" in self.name:
            self.volume = 80.0  # 80 cc for spinal cord
        elif "Esophagus" in self.name:
            self.volume = 40.0  # 40 cc for esophagus
        elif "Liver" in self.name:
            self.volume = 1500.0  # 1500 cc for liver
        elif "Kidney" in self.name:
            self.volume = 150.0  # 150 cc for kidney
        elif "Brain" in self.name:
            self.volume = 1400.0  # 1400 cc for brain
        else:
            # Default volume
            self.volume = 100.0
    
    def get_contours(self, view_index: int, slice_index: int) -> List[List[Tuple[float, float]]]:
        """
        Get contours for a specific view and slice.
        
        Args:
            view_index: View index (0=axial, 1=sagittal, 2=coronal)
            slice_index: Slice index
            
        Returns:
            List of contours, where each contour is a list of (x,y) point tuples
        """
        # In a real implementation, this would return the actual contours
        # for the requested view and slice
        
        # For demonstration, create some sample contours
        # Circle with radius based on structure type
        if view_index == 0:  # Axial
            # Only return contours for the middle slices
            if slice_index < 10 or slice_index > 40:
                return []
                
            radius = 0.0
            
            if self.type == StructureType.PTV:
                radius = 20.0
            elif self.type == StructureType.CTV:
                radius = 18.0
            elif self.type == StructureType.GTV:
                radius = 15.0
            elif self.type == StructureType.EXTERNAL:
                radius = 50.0
            elif "Lung" in self.name:
                radius = 40.0
                # Return two contours for lungs (left and right)
                return [
                    [(30 + 30 * np.cos(t), 40 + 20 * np.sin(t)) for t in np.linspace(0, 2*np.pi, 20)],
                    [(70 + 30 * np.cos(t), 40 + 20 * np.sin(t)) for t in np.linspace(0, 2*np.pi, 20)]
                ]
            elif "Heart" in self.name:
                radius = 15.0
                # Heart shape
                return [
                    [(50 + 15 * np.sin(t), 40 + 15 * np.cos(t) + 5 * np.abs(np.sin(t))) for t in np.linspace(0, 2*np.pi, 30)]
                ]
            elif "Spinal" in self.name or "Cord" in self.name:
                # Oval shape for spinal cord
                return [
                    [(50 + 5 * np.cos(t), 15 + 10 * np.sin(t)) for t in np.linspace(0, 2*np.pi, 20)]
                ]
            else:
                radius = 10.0
            
            # Create circular contour around center (50, 50)
            center_x, center_y = 50, 50
            n_points = 20
            
            contour = []
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                contour.append((x, y))
            
            return [contour]
        else:
            # For other views, return empty list for now
            return [] 