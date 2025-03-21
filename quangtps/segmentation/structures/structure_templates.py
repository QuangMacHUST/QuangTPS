#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các template cấu trúc giải phẫu trong QuangTPS.

Module này cung cấp lớp StructureTemplate để định nghĩa và quản lý các 
mẫu cấu trúc giải phẫu được sử dụng trong kế hoạch điều trị xạ trị.
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import os
import uuid
import logging

from quangtps.segmentation.structures.structure import StructureType, StructurePriority

logger = logging.getLogger(__name__)

class StructureTemplate:
    """
    Đối tượng đại diện cho một mẫu cấu trúc giải phẫu tái sử dụng.
    
    Attributes:
        id (str): ID duy nhất của template
        name (str): Tên template
        description (str): Mô tả template
        type (StructureType): Loại cấu trúc
        priority (StructurePriority): Mức độ ưu tiên trong tối ưu hóa
        color (Tuple[int, int, int]): Màu RGB mặc định
        dose_constraints (List[Dict]): Các ràng buộc liều mặc định
        metadata (Dict[str, Any]): Thông tin bổ sung
    """
    
    def __init__(self, name: str, 
                description: str = "",
                structure_type: StructureType = StructureType.UNKNOWN,
                priority: StructurePriority = StructurePriority.NONE,
                color: Optional[Tuple[int, int, int]] = None,
                dose_constraints: Optional[List[Dict]] = None,
                template_id: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo template cấu trúc.
        
        Parameters:
            name (str): Tên template
            description (str): Mô tả template
            structure_type (StructureType): Loại cấu trúc
            priority (StructurePriority): Mức độ ưu tiên
            color (Optional[Tuple[int, int, int]]): Màu RGB mặc định
            dose_constraints (Optional[List[Dict]]): Các ràng buộc liều mặc định
            template_id (Optional[str]): ID duy nhất (nếu None, tự động tạo)
            metadata (Optional[Dict[str, Any]]): Thông tin bổ sung
        """
        self.id = template_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.type = structure_type
        self.priority = priority
        self.dose_constraints = dose_constraints or []
        self.metadata = metadata or {}
        
        # Chọn màu mặc định dựa trên loại cấu trúc
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
    
    def add_dose_constraint(self, constraint_type: str, dose: float, volume: Optional[float] = None, 
                          priority: float = 1.0, description: Optional[str] = None) -> None:
        """
        Thêm ràng buộc liều cho template.
        
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
        Chuyển đổi template thành dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary mô tả template
        """
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'type': self.type.name,
            'priority': self.priority.name,
            'color': self.color,
            'dose_constraints': self.dose_constraints,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructureTemplate':
        """
        Tạo đối tượng StructureTemplate từ dictionary.
        
        Parameters:
            data (Dict[str, Any]): Dictionary mô tả template
            
        Returns:
            StructureTemplate: Đối tượng StructureTemplate mới
        """
        # Chuyển đổi các trường enum
        structure_type = StructureType[data.get('type', 'UNKNOWN')]
        priority = StructurePriority[data.get('priority', 'NONE')]
        
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            structure_type=structure_type,
            priority=priority,
            color=tuple(data.get('color', (255, 165, 0))),
            dose_constraints=data.get('dose_constraints', []),
            template_id=data.get('id'),
            metadata=data.get('metadata', {})
        )
    
    def __str__(self) -> str:
        """String representation."""
        return f"StructureTemplate(name='{self.name}', type={self.type.name})"


class StructureTemplateLibrary:
    """
    Đối tượng quản lý thư viện các template cấu trúc giải phẫu.
    
    Cung cấp các phương thức để thêm, xóa, tìm kiếm và quản lý template.
    """
    
    def __init__(self):
        """Khởi tạo thư viện template."""
        self.templates: Dict[str, StructureTemplate] = {}
    
    def add_template(self, template: StructureTemplate) -> None:
        """
        Thêm template vào thư viện.
        
        Parameters:
            template (StructureTemplate): Template cần thêm
        """
        self.templates[template.id] = template
    
    def remove_template(self, template_id: str) -> bool:
        """
        Xóa template khỏi thư viện.
        
        Parameters:
            template_id (str): ID của template cần xóa
            
        Returns:
            bool: True nếu xóa thành công, False nếu không tìm thấy
        """
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False
    
    def get_template(self, template_id: str) -> Optional[StructureTemplate]:
        """
        Lấy template theo ID.
        
        Parameters:
            template_id (str): ID của template cần lấy
            
        Returns:
            Optional[StructureTemplate]: Template nếu tìm thấy, None nếu không
        """
        return self.templates.get(template_id)
    
    def get_template_by_name(self, name: str) -> Optional[StructureTemplate]:
        """
        Tìm template theo tên.
        
        Parameters:
            name (str): Tên template cần tìm
            
        Returns:
            Optional[StructureTemplate]: Template đầu tiên có tên phù hợp, None nếu không tìm thấy
        """
        for template in self.templates.values():
            if template.name == name:
                return template
        return None
    
    def get_templates_by_type(self, structure_type: StructureType) -> List[StructureTemplate]:
        """
        Lấy danh sách template theo loại cấu trúc.
        
        Parameters:
            structure_type (StructureType): Loại cấu trúc
            
        Returns:
            List[StructureTemplate]: Danh sách template thuộc loại đã cho
        """
        return [t for t in self.templates.values() if t.type == structure_type]
    
    def get_all_templates(self) -> List[StructureTemplate]:
        """
        Lấy tất cả template trong thư viện.
        
        Returns:
            List[StructureTemplate]: Danh sách tất cả template
        """
        return list(self.templates.values())
    
    def clear(self) -> None:
        """Xóa tất cả template trong thư viện."""
        self.templates.clear()
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Tải template từ file JSON.
        
        Parameters:
            filepath (str): Đường dẫn tới file JSON chứa template
            
        Returns:
            bool: True nếu tải thành công, False nếu có lỗi
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for template_data in data.get('templates', []):
                template = StructureTemplate.from_dict(template_data)
                self.add_template(template)
                
            return True
        except Exception as e:
            logger.error(f"Failed to load templates from file: {e}")
            return False
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Lưu thư viện template vào file JSON.
        
        Parameters:
            filepath (str): Đường dẫn tới file JSON
            
        Returns:
            bool: True nếu lưu thành công, False nếu có lỗi
        """
        try:
            data = {
                'templates': [t.to_dict() for t in self.templates.values()]
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            return True
        except Exception as e:
            logger.error(f"Failed to save templates to file: {e}")
            return False
    
    def create_default_templates(self) -> None:
        """Tạo các template mặc định cho các cấu trúc giải phẫu phổ biến."""
        # PTV
        ptv = StructureTemplate(
            name="PTV",
            description="Planning Target Volume",
            structure_type=StructureType.TARGET,
            priority=StructurePriority.HIGH,
            color=(255, 0, 0)
        )
        ptv.add_dose_constraint("min", 95.0, description="Tối thiểu 95% liều kê đơn")
        ptv.add_dose_constraint("D95", 95.0, description="95% thể tích cần đạt 95% liều kê đơn")
        self.add_template(ptv)
        
        # CTV
        ctv = StructureTemplate(
            name="CTV",
            description="Clinical Target Volume",
            structure_type=StructureType.TARGET,
            priority=StructurePriority.HIGH,
            color=(255, 50, 50)
        )
        ctv.add_dose_constraint("min", 98.0, description="Tối thiểu 98% liều kê đơn")
        self.add_template(ctv)
        
        # GTV
        gtv = StructureTemplate(
            name="GTV",
            description="Gross Tumor Volume",
            structure_type=StructureType.TARGET,
            priority=StructurePriority.HIGH,
            color=(255, 100, 100)
        )
        gtv.add_dose_constraint("min", 100.0, description="Tối thiểu 100% liều kê đơn")
        self.add_template(gtv)
        
        # Một số OARs phổ biến
        
        # Não
        brain = StructureTemplate(
            name="Brain",
            description="Não",
            structure_type=StructureType.OAR,
            priority=StructurePriority.HIGH,
            color=(0, 0, 255)
        )
        brain.add_dose_constraint("max", 45.0, description="Liều tối đa không quá 45 Gy")
        self.add_template(brain)
        
        # Thân não
        brainstem = StructureTemplate(
            name="Brainstem",
            description="Thân não",
            structure_type=StructureType.OAR,
            priority=StructurePriority.HIGH,
            color=(0, 50, 255)
        )
        brainstem.add_dose_constraint("max", 54.0, description="Liều tối đa không quá 54 Gy")
        self.add_template(brainstem)
        
        # Tim
        heart = StructureTemplate(
            name="Heart",
            description="Tim",
            structure_type=StructureType.OAR,
            priority=StructurePriority.HIGH,
            color=(0, 100, 255)
        )
        heart.add_dose_constraint("mean", 26.0, description="Liều trung bình không quá 26 Gy")
        heart.add_dose_constraint("V25", 10.0, description="Không quá 10% thể tích nhận liều 25 Gy")
        self.add_template(heart)
        
        # Phổi
        lung = StructureTemplate(
            name="Lung",
            description="Phổi",
            structure_type=StructureType.OAR,
            priority=StructurePriority.HIGH,
            color=(0, 150, 255)
        )
        lung.add_dose_constraint("mean", 20.0, description="Liều trung bình không quá 20 Gy")
        lung.add_dose_constraint("V20", 30.0, description="Không quá 30% thể tích nhận liều 20 Gy")
        self.add_template(lung)
        
        # Tủy sống
        spinalcord = StructureTemplate(
            name="Spinal Cord",
            description="Tủy sống",
            structure_type=StructureType.OAR,
            priority=StructurePriority.HIGH,
            color=(0, 200, 255)
        )
        spinalcord.add_dose_constraint("max", 45.0, description="Liều tối đa không quá 45 Gy")
        self.add_template(spinalcord)
        
        # Thân thể
        body = StructureTemplate(
            name="Body",
            description="Thân thể",
            structure_type=StructureType.BODY,
            priority=StructurePriority.NONE,
            color=(0, 255, 0)
        )
        self.add_template(body)


# Khởi tạo thư viện template toàn cục
template_library = StructureTemplateLibrary()