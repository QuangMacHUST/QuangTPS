#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa các cấu trúc giải phẫu trong QuangTPS.

Module này cung cấp lớp Structure và các liệt kê liên quan để đại diện
cho các cấu trúc giải phẫu sử dụng trong kế hoạch điều trị xạ trị.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Set, Union
from enum import Enum, auto
import uuid
import logging
import cv2

from quangtps.segmentation.structures.geometry import Point, Contour

logger = logging.getLogger(__name__)

class StructureType(Enum):
    """Loại cấu trúc giải phẫu."""
    TARGET = auto()       # Mục tiêu điều trị (PTV, GTV, CTV)
    OAR = auto()          # Cơ quan nguy cấp (organs at risk)
    BODY = auto()         # Đường viền cơ thể 
    SUPPORT = auto()      # Cấu trúc hỗ trợ (bolus, ...)
    EXTERNAL = auto()     # Cấu trúc bên ngoài
    UNKNOWN = auto()      # Không xác định

class StructurePriority(Enum):
    """Mức độ ưu tiên của cấu trúc trong quá trình tối ưu hóa."""
    HIGH = 3       # Ưu tiên cao (các mục tiêu chính, OARs quan trọng)
    MEDIUM = 2     # Ưu tiên trung bình
    LOW = 1        # Ưu tiên thấp
    NONE = 0       # Không ưu tiên

class Structure:
    """
    Đối tượng đại diện cho một cấu trúc giải phẫu hoặc mục tiêu trong kế hoạch điều trị.
    
    Attributes:
        id (str): ID duy nhất của cấu trúc
        name (str): Tên của cấu trúc
        type (StructureType): Loại cấu trúc (TARGET, OAR, ...)
        priority (StructurePriority): Mức độ ưu tiên trong tối ưu hóa
        color (Tuple[int, int, int]): Màu RGB của cấu trúc
        contours (Dict[float, List[Contour]]): Từ điển các đường bao, với khóa là vị trí z
        visible (bool): Trạng thái hiển thị cấu trúc
        metadata (Dict[str, Any]): Thông tin bổ sung về cấu trúc
        is_empty (bool): Cấu trúc có trống không (không có đường bao)
        dose_constraints (List[Dict]): Các ràng buộc liều cho cấu trúc này
    """
    
    def __init__(self, name: str, 
                 structure_type: StructureType = StructureType.UNKNOWN,
                 priority: StructurePriority = StructurePriority.NONE,
                 color: Optional[Tuple[int, int, int]] = None,
                 contours: Optional[Dict[float, List[Contour]]] = None,
                 structure_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo cấu trúc.
        
        Parameters:
            name (str): Tên cấu trúc
            structure_type (StructureType): Loại cấu trúc
            priority (StructurePriority): Mức độ ưu tiên
            color (Optional[Tuple[int, int, int]]): Màu RGB (nếu None, tự động gán)
            contours (Optional[Dict[float, List[Contour]]]): Từ điển các đường bao theo z
            structure_id (Optional[str]): ID duy nhất (nếu None, tự động tạo)
            metadata (Optional[Dict[str, Any]]): Thông tin bổ sung
        """
        self.id = structure_id or str(uuid.uuid4())
        self.name = name
        self.type = structure_type
        self.priority = priority
        self.contours = contours or {}
        self.visible = True
        self.metadata = metadata or {}
        self.dose_constraints = []
        
        # Tự động chọn màu dựa trên loại cấu trúc nếu không cung cấp
        if color is None:
            if structure_type == StructureType.TARGET:
                self.color = (255, 0, 0)  # Đỏ cho PTV/CTV
            elif structure_type == StructureType.OAR:
                self.color = (0, 0, 255)  # Xanh dương cho OARs
            elif structure_type == StructureType.BODY:
                self.color = (0, 255, 0)  # Xanh lá cho thân thể
            else:
                self.color = (255, 165, 0)  # Cam cho các cấu trúc khác
        else:
            self.color = color
    
    @property
    def is_empty(self) -> bool:
        """Kiểm tra xem cấu trúc có trống không (không có đường bao)."""
        return len(self.contours) == 0 or all(len(contours) == 0 for contours in self.contours.values())
    
    def get_z_positions(self) -> List[float]:
        """
        Lấy danh sách các vị trí z (lát cắt) mà cấu trúc này có đường bao.
        
        Returns:
            List[float]: Danh sách các vị trí z, được sắp xếp
        """
        return sorted(self.contours.keys())
    
    def add_contour(self, contour: Contour) -> None:
        """
        Thêm một đường bao vào cấu trúc.
        
        Parameters:
            contour (Contour): Đường bao cần thêm
        """
        z = contour.z
        if z not in self.contours:
            self.contours[z] = []
        self.contours[z].append(contour)
    
    def remove_contour(self, z: float, contour_id: str) -> bool:
        """
        Xóa một đường bao khỏi cấu trúc.
        
        Parameters:
            z (float): Vị trí z của đường bao
            contour_id (str): ID của đường bao cần xóa
            
        Returns:
            bool: True nếu xóa thành công, False nếu không tìm thấy
        """
        if z not in self.contours:
            return False
        
        for i, contour in enumerate(self.contours[z]):
            if contour.id == contour_id:
                self.contours[z].pop(i)
                if not self.contours[z]:  # Nếu không còn đường bao nào ở z này
                    del self.contours[z]
                return True
        
        return False
    
    def clear_contours(self) -> None:
        """Xóa tất cả các đường bao của cấu trúc."""
        self.contours.clear()
    
    def remove_contours_at_z(self, z: float) -> bool:
        """
        Xóa tất cả các đường bao tại một vị trí z cụ thể.
        
        Parameters:
            z (float): Vị trí z của các đường bao cần xóa
            
        Returns:
            bool: True nếu có đường bao bị xóa, False nếu không
        """
        if z in self.contours:
            del self.contours[z]
            return True
        return False
    
    def calculate_volume(self) -> float:
        """
        Tính toán thể tích của cấu trúc.
        
        Returns:
            float: Thể tích tính bằng mm³
        """
        if self.is_empty:
            return 0.0
        
        total_volume = 0.0
        z_positions = self.get_z_positions()
        
        for i in range(len(z_positions) - 1):
            z1 = z_positions[i]
            z2 = z_positions[i+1]
            thickness = abs(z2 - z1)
            
            # Tính diện tích trung bình của 2 lát cắt liền kề
            area1 = sum(contour.get_area() for contour in self.contours[z1])
            area2 = sum(contour.get_area() for contour in self.contours[z2])
            avg_area = (area1 + area2) / 2
            
            # Thể tích = diện tích trung bình * độ dày
            total_volume += avg_area * thickness
        
        return total_volume
    
    def create_mask(self, shape: Tuple[int, int], pixel_spacing: Tuple[float, float], 
                    origin: Tuple[float, float], z: float) -> np.ndarray:
        """
        Tạo mặt nạ nhị phân từ các đường bao tại vị trí z.
        
        Parameters:
            shape (Tuple[int, int]): Kích thước của mặt nạ (height, width)
            pixel_spacing (Tuple[float, float]): Khoảng cách giữa các pixel (dx, dy)
            origin (Tuple[float, float]): Tọa độ gốc của hình ảnh (x0, y0)
            z (float): Vị trí z để tạo mặt nạ
            
        Returns:
            np.ndarray: Mặt nạ nhị phân, dtype=np.uint8, 0 là nền, 255 là cấu trúc
        """
        if z not in self.contours or not self.contours[z]:
            return np.zeros(shape, dtype=np.uint8)
        
        mask = np.zeros(shape, dtype=np.uint8)
        
        for contour in self.contours[z]:
            # Chuyển đổi từ tọa độ thế giới sang tọa độ pixel
            points_px = []
            for point in contour.points:
                px = int(round((point.x - origin[0]) / pixel_spacing[0]))
                py = int(round((point.y - origin[1]) / pixel_spacing[1]))
                points_px.append([px, py])
            
            # Vẽ contour lên mặt nạ
            if len(points_px) >= 3:  # Cần ít nhất 3 điểm để tạo đa giác
                points_array = np.array(points_px, dtype=np.int32)
                cv2.fillPoly(mask, [points_array], 255)
        
        return mask
    
    def add_dose_constraint(self, constraint_type: str, dose: float, volume: Optional[float] = None, 
                           priority: float = 1.0, description: Optional[str] = None) -> None:
        """
        Thêm ràng buộc liều cho cấu trúc.
        
        Parameters:
            constraint_type (str): Loại ràng buộc ('max', 'min', 'mean', 'D95', 'V20', ...)
            dose (float): Giá trị liều (Gy)
            volume (Optional[float]): Giá trị thể tích (%, chỉ dùng cho ràng buộc DVH)
            priority (float): Trọng số ưu tiên của ràng buộc này
            description (Optional[str]): Mô tả ràng buộc
        """
        constraint = {
            'type': constraint_type,
            'dose': dose,
            'priority': priority
        }
        
        if volume is not None:
            constraint['volume'] = volume
            
        if description:
            constraint['description'] = description
            
        self.dose_constraints.append(constraint)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi cấu trúc thành dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary mô tả cấu trúc
        """
        # Chuyển đổi contours thành định dạng có thể serialize
        contours_dict = {}
        for z, contours in self.contours.items():
            contours_dict[str(z)] = [c.to_dict() for c in contours]
        
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.name,
            'priority': self.priority.name,
            'color': self.color,
            'contours': contours_dict,
            'visible': self.visible,
            'metadata': self.metadata,
            'dose_constraints': self.dose_constraints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Structure':
        """
        Tạo đối tượng Structure từ dictionary.
        
        Parameters:
            data (Dict[str, Any]): Dictionary mô tả cấu trúc
            
        Returns:
            Structure: Đối tượng Structure mới
        """
        # Chuyển đổi contours từ định dạng serialized
        contours = {}
        for z_str, contour_list in data.get('contours', {}).items():
            z = float(z_str)
            contours[z] = [Contour.from_dict(c) for c in contour_list]
        
        # Chuyển đổi các trường enum
        structure_type = StructureType[data.get('type', 'UNKNOWN')]
        priority = StructurePriority[data.get('priority', 'NONE')]
        
        # Tạo Structure mới
        structure = cls(
            name=data['name'],
            structure_type=structure_type,
            priority=priority,
            color=tuple(data.get('color', (255, 165, 0))),
            contours=contours,
            structure_id=data.get('id'),
            metadata=data.get('metadata', {})
        )
        
        # Thiết lập các thuộc tính khác
        structure.visible = data.get('visible', True)
        structure.dose_constraints = data.get('dose_constraints', [])
        
        return structure
    
    def __str__(self) -> str:
        """String representation."""
        z_count = len(self.contours)
        contour_count = sum(len(contours) for contours in self.contours.values())
        return f"Structure(name='{self.name}', type={self.type.name}, {z_count} slices, {contour_count} contours)"
