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