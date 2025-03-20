#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cầu nối giữa cấu trúc hình ảnh và phân đoạn.

Module này cung cấp các lớp và hàm để chuyển đổi giữa các đối tượng cấu trúc 
từ module imaging và segmentation, đảm bảo tính nhất quán trong toàn bộ hệ thống.
"""

import logging
import json
import uuid
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any, Union

from quangtps.imaging.structures import (
    Structure as ImagingStructure,
    StructureSet as ImagingStructureSet,
    StructureType as ImagingStructureType
)
from quangtps.segmentation.structures import (
    Structure, StructureSet, StructureType, StructurePriority,
    Point, Contour, StructureTemplate
)

logger = logging.getLogger(__name__)

# Ánh xạ từ kiểu cấu trúc imaging sang segmentation
_TYPE_MAPPING_IMAGING_TO_SEGMENTATION = {
    ImagingStructureType.EXTERNAL: StructureType.EXTERNAL,
    ImagingStructureType.PTV: StructureType.PTV,
    ImagingStructureType.CTV: StructureType.CTV,
    ImagingStructureType.GTV: StructureType.GTV,
    ImagingStructureType.ITV: StructureType.ITV,
    ImagingStructureType.OAR: StructureType.OAR,
    ImagingStructureType.ORGAN: StructureType.OTHER,
    ImagingStructureType.BOLUS: StructureType.SUPPORT,
    ImagingStructureType.AVOIDANCE: StructureType.OTHER,
    ImagingStructureType.SUPPORT: StructureType.SUPPORT,
    ImagingStructureType.MARKER: StructureType.MARKER,
    ImagingStructureType.REGISTRATION: StructureType.OTHER,
    ImagingStructureType.ISOCENTER: StructureType.OTHER,
    ImagingStructureType.UNDEFINED: StructureType.OTHER,
    ImagingStructureType.CUSTOM: StructureType.OTHER
}

# Ánh xạ từ kiểu cấu trúc segmentation sang imaging
_TYPE_MAPPING_SEGMENTATION_TO_IMAGING = {
    StructureType.EXTERNAL: ImagingStructureType.EXTERNAL,
    StructureType.PTV: ImagingStructureType.PTV,
    StructureType.CTV: ImagingStructureType.CTV,
    StructureType.GTV: ImagingStructureType.GTV,
    StructureType.ITV: ImagingStructureType.ITV,
    StructureType.OAR: ImagingStructureType.OAR,
    StructureType.SUPPORT: ImagingStructureType.SUPPORT,
    StructureType.MARKER: ImagingStructureType.MARKER,
    StructureType.OTHER: ImagingStructureType.UNDEFINED
}


def imaging_to_segmentation_structure(
    imaging_structure: ImagingStructure
) -> Structure:
    """
    Chuyển đổi Structure từ module imaging sang Structure của module segmentation.
    
    Parameters
    ----------
    imaging_structure : ImagingStructure
        Đối tượng Structure từ module imaging
        
    Returns
    -------
    Structure
        Đối tượng Structure tương ứng từ module segmentation
    """
    # Ánh xạ loại cấu trúc
    structure_type = _TYPE_MAPPING_IMAGING_TO_SEGMENTATION.get(
        imaging_structure.structure_type, StructureType.OTHER)
    
    # Tạo đối tượng Structure từ segmentation
    segmentation_structure = Structure(
        name=imaging_structure.name,
        id=imaging_structure.id if hasattr(imaging_structure, 'id') else str(uuid.uuid4())
    )
    
    # Chuyển đổi các thuộc tính cơ bản
    if hasattr(imaging_structure, 'type'):
        segmentation_structure.type = imaging_structure.type
    if hasattr(imaging_structure, 'color'):
        segmentation_structure.color = imaging_structure.color
    if hasattr(imaging_structure, 'priority'):
        segmentation_structure.priority = imaging_structure.priority
    if hasattr(imaging_structure, 'meta'):
        segmentation_structure.meta = imaging_structure.meta
    if hasattr(imaging_structure, 'creation_date'):
        segmentation_structure.creation_date = imaging_structure.creation_date
    
    # Chuyển đổi contours nếu có
    if hasattr(imaging_structure, 'contours') and imaging_structure.contours:
        for z_value, contour_list in imaging_structure.contours.items():
            for contour_data in contour_list:
                # Chuyển đổi contour_data thành đối tượng Contour của segmentation
                points = []
                for point_data in contour_data:
                    # Tạo đối tượng Point
                    if isinstance(point_data, (list, tuple, np.ndarray)):
                        # Nếu dữ liệu điểm là mảng/tuple
                        if len(point_data) >= 3:
                            point = Point(point_data[0], point_data[1], point_data[2])
                        else:
                            point = Point(point_data[0], point_data[1], z_value)
                    elif hasattr(point_data, 'x') and hasattr(point_data, 'y'):
                        # Nếu dữ liệu điểm là đối tượng có thuộc tính x, y
                        z = point_data.z if hasattr(point_data, 'z') else z_value
                        point = Point(point_data.x, point_data.y, z)
                    else:
                        # Trường hợp khác, bỏ qua
                        continue
                        
                    points.append(point)
                
                # Chỉ tạo contour nếu có đủ điểm
                if len(points) > 2:
                    contour = Contour(points, z_value)
                    segmentation_structure.add_contour(contour)
    
    return segmentation_structure


def segmentation_to_imaging_structure(
    segmentation_structure: Structure
) -> ImagingStructure:
    """
    Chuyển đổi Structure từ module segmentation sang Structure của module imaging.
    
    Parameters
    ----------
    segmentation_structure : Structure
        Đối tượng Structure từ module segmentation
        
    Returns
    -------
    ImagingStructure
        Đối tượng Structure tương ứng từ module imaging
    """
    # Ánh xạ loại cấu trúc
    structure_type = _TYPE_MAPPING_SEGMENTATION_TO_IMAGING.get(
        segmentation_structure.type, ImagingStructureType.UNDEFINED)
    
    # Tạo đối tượng Structure từ imaging
    imaging_structure = ImagingStructure(
        id=segmentation_structure.id,
        name=segmentation_structure.name,
        structure_type=structure_type,
        color=segmentation_structure.color
    )
    
    # Chuyển đổi các thuộc tính cơ bản
    if hasattr(segmentation_structure, 'priority'):
        imaging_structure.priority = segmentation_structure.priority
    if hasattr(segmentation_structure, 'meta'):
        imaging_structure.meta = segmentation_structure.meta
    if hasattr(segmentation_structure, 'creation_date'):
        imaging_structure.creation_date = segmentation_structure.creation_date
    
    # Chuyển đổi contours
    imaging_structure.contours = {}
    
    for contour in segmentation_structure.get_contours():
        z_value = contour.z
        
        # Tạo danh sách điểm
        points_data = []
        for point in contour.points:
            # Chuyển Point thành dạng tuple/list
            point_data = [point.x, point.y, point.z]
            points_data.append(point_data)
        
        # Thêm contour vào từng mặt phẳng z
        if z_value not in imaging_structure.contours:
            imaging_structure.contours[z_value] = []
        
        # Thêm contour vào danh sách
        imaging_structure.contours[z_value].append(points_data)
    
    return imaging_structure


def imaging_to_segmentation_structure_set(
    imaging_structure_set: ImagingStructureSet
) -> StructureSet:
    """
    Chuyển đổi StructureSet từ module imaging sang StructureSet của module segmentation.
    
    Parameters
    ----------
    imaging_structure_set : ImagingStructureSet
        Đối tượng StructureSet từ module imaging
        
    Returns
    -------
    StructureSet
        Đối tượng StructureSet tương ứng từ module segmentation
    """
    # Tạo đối tượng StructureSet từ segmentation
    segmentation_structure_set = StructureSet(
        name=imaging_structure_set.name if hasattr(imaging_structure_set, 'name') else "Converted StructureSet",
        id=imaging_structure_set.id if hasattr(imaging_structure_set, 'id') else str(uuid.uuid4())
    )
    
    # Chuyển đổi các thuộc tính cơ bản
    if hasattr(imaging_structure_set, 'meta'):
        segmentation_structure_set.meta = imaging_structure_set.meta
    if hasattr(imaging_structure_set, 'creation_date'):
        segmentation_structure_set.creation_date = imaging_structure_set.creation_date
    if hasattr(imaging_structure_set, 'modified_date'):
        segmentation_structure_set.modified_date = imaging_structure_set.modified_date
    if hasattr(imaging_structure_set, 'associated_image_id'):
        segmentation_structure_set.associated_image_id = imaging_structure_set.associated_image_id
    
    # Chuyển đổi từng cấu trúc
    if hasattr(imaging_structure_set, 'structures'):
        structures = imaging_structure_set.structures
        
        # Nếu structures là dict (key-value)
        if isinstance(structures, dict):
            for structure_id, imaging_structure in structures.items():
                segmentation_structure = imaging_to_segmentation_structure(imaging_structure)
                segmentation_structure_set.add_structure(segmentation_structure)
        
        # Nếu structures là list
        elif isinstance(structures, list):
            for imaging_structure in structures:
                segmentation_structure = imaging_to_segmentation_structure(imaging_structure)
                segmentation_structure_set.add_structure(segmentation_structure)
    
    return segmentation_structure_set


def segmentation_to_imaging_structure_set(
    segmentation_structure_set: StructureSet
) -> ImagingStructureSet:
    """
    Chuyển đổi StructureSet từ module segmentation sang StructureSet của module imaging.
    
    Parameters
    ----------
    segmentation_structure_set : StructureSet
        Đối tượng StructureSet từ module segmentation
        
    Returns
    -------
    ImagingStructureSet
        Đối tượng StructureSet tương ứng từ module imaging
    """
    # Tạo đối tượng StructureSet từ imaging
    imaging_structure_set = ImagingStructureSet(
        id=segmentation_structure_set.id,
        name=segmentation_structure_set.name
    )
    
    # Chuyển đổi các thuộc tính cơ bản
    if hasattr(segmentation_structure_set, 'meta'):
        imaging_structure_set.meta = segmentation_structure_set.meta
    if hasattr(segmentation_structure_set, 'creation_date'):
        imaging_structure_set.creation_date = segmentation_structure_set.creation_date
    if hasattr(segmentation_structure_set, 'modified_date'):
        imaging_structure_set.modified_date = segmentation_structure_set.modified_date
    if hasattr(segmentation_structure_set, 'associated_image_id'):
        imaging_structure_set.associated_image_id = segmentation_structure_set.associated_image_id
    
    # Chuyển đổi từng cấu trúc
    imaging_structure_set.structures = {}
    
    for structure_id, segmentation_structure in segmentation_structure_set.structures.items():
        imaging_structure = segmentation_to_imaging_structure(segmentation_structure)
        imaging_structure_set.structures[structure_id] = imaging_structure
    
    return imaging_structure_set


def convert_point(point_data: Union[List, Tuple, Dict, Any], z: Optional[float] = None) -> Point:
    """
    Chuyển đổi dữ liệu điểm từ nhiều định dạng khác nhau thành đối tượng Point.
    
    Args:
        point_data: Dữ liệu điểm (list, tuple, dict hoặc đối tượng có thuộc tính x, y, z)
        z: Giá trị z mặc định nếu không có trong dữ liệu
        
    Returns:
        Point: Đối tượng Point tương ứng
    """
    if isinstance(point_data, (list, tuple, np.ndarray)):
        # Nếu dữ liệu điểm là mảng/tuple
        if len(point_data) >= 3:
            return Point(point_data[0], point_data[1], point_data[2])
        else:
            return Point(point_data[0], point_data[1], z or 0.0)
    elif isinstance(point_data, dict):
        # Nếu dữ liệu điểm là dict
        x = point_data.get('x', 0.0)
        y = point_data.get('y', 0.0)
        z_val = point_data.get('z', z or 0.0)
        return Point(x, y, z_val)
    elif hasattr(point_data, 'x') and hasattr(point_data, 'y'):
        # Nếu dữ liệu điểm là đối tượng có thuộc tính x, y
        z_val = point_data.z if hasattr(point_data, 'z') else z or 0.0
        return Point(point_data.x, point_data.y, z_val)
    else:
        # Trường hợp không xác định, tạo điểm mặc định
        return Point(0.0, 0.0, z or 0.0)


def convert_contour(contour_data: Union[List, Dict, Any], z: Optional[float] = None) -> Contour:
    """
    Chuyển đổi dữ liệu contour từ nhiều định dạng khác nhau thành đối tượng Contour.
    
    Args:
        contour_data: Dữ liệu contour (list các điểm, dict hoặc đối tượng có thuộc tính points)
        z: Giá trị z mặc định cho contour
        
    Returns:
        Contour: Đối tượng Contour tương ứng
    """
    points = []
    
    if isinstance(contour_data, list):
        # Nếu dữ liệu contour là list các điểm
        for point_data in contour_data:
            point = convert_point(point_data, z)
            points.append(point)
    elif isinstance(contour_data, dict) and 'points' in contour_data:
        # Nếu dữ liệu contour là dict có khóa 'points'
        contour_z = contour_data.get('z', z)
        for point_data in contour_data['points']:
            point = convert_point(point_data, contour_z)
            points.append(point)
    elif hasattr(contour_data, 'points'):
        # Nếu dữ liệu contour là đối tượng có thuộc tính points
        contour_z = contour_data.z if hasattr(contour_data, 'z') else z
        for point_data in contour_data.points:
            point = convert_point(point_data, contour_z)
            points.append(point)
    
    # Chỉ tạo contour nếu có đủ điểm
    if len(points) > 2:
        contour_z = z
        if len(points) > 0 and all(hasattr(p, 'z') for p in points):
            # If all points have z values, use their average
            contour_z = sum(p.z for p in points) / len(points)
        return Contour(points, contour_z or 0.0)
    else:
        # Trường hợp không đủ điểm, tạo contour rỗng
        return Contour([], z or 0.0)
