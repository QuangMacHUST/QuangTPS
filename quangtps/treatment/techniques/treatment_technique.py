#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa lớp kỹ thuật xạ trị cơ bản.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class TreatmentTechnique:
    """
    Lớp cơ sở cho các kỹ thuật xạ trị.
    
    Lớp này cung cấp cấu trúc và chức năng cơ bản cho tất cả các kỹ thuật xạ trị,
    cho phép các lớp con mở rộng với các tính năng và tham số cụ thể.
    """
    
    def __init__(self, technique_name: str):
        """
        Khởi tạo một kỹ thuật xạ trị.
        
        Parameters
        ----------
        technique_name : str
            Tên của kỹ thuật
        """
        self.technique_name = technique_name
        self.description = ""
        self.compatible_machines = []  # Danh sách các máy xạ trị tương thích
        self.metadata = {}
    
    def set_description(self, description: str):
        """
        Thiết lập mô tả cho kỹ thuật.
        
        Parameters
        ----------
        description : str
            Mô tả của kỹ thuật
        """
        self.description = description
    
    def add_compatible_machine(self, machine_type: str):
        """
        Thêm loại máy xạ trị tương thích.
        
        Parameters
        ----------
        machine_type : str
            Loại máy xạ trị
        """
        if machine_type not in self.compatible_machines:
            self.compatible_machines.append(machine_type)
    
    def is_compatible_with(self, machine_type: str) -> bool:
        """
        Kiểm tra xem kỹ thuật có tương thích với loại máy không.
        
        Parameters
        ----------
        machine_type : str
            Loại máy xạ trị
            
        Returns
        -------
        bool
            True nếu tương thích, False nếu không
        """
        return machine_type in self.compatible_machines
    
    def add_metadata(self, key: str, value: Any):
        """
        Thêm thông tin metadata cho kỹ thuật.
        
        Parameters
        ----------
        key : str
            Khóa metadata
        value : Any
            Giá trị metadata
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Lấy thông tin metadata theo khóa.
        
        Parameters
        ----------
        key : str
            Khóa metadata
        default : Any, optional
            Giá trị mặc định nếu không tìm thấy khóa
            
        Returns
        -------
        Any
            Giá trị metadata
        """
        return self.metadata.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin kỹ thuật thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kỹ thuật
        """
        return {
            "technique_name": self.technique_name,
            "description": self.description,
            "compatible_machines": self.compatible_machines,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreatmentTechnique':
        """
        Tạo đối tượng TreatmentTechnique từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin kỹ thuật
            
        Returns
        -------
        TreatmentTechnique
            Đối tượng TreatmentTechnique
        """
        technique = cls(data["technique_name"])
        technique.description = data.get("description", "")
        technique.compatible_machines = data.get("compatible_machines", [])
        technique.metadata = data.get("metadata", {})
        
        return technique
    
    def __str__(self) -> str:
        return f"{self.technique_name}"
    
    def __repr__(self) -> str:
        return f"TreatmentTechnique(technique_name='{self.technique_name}')" 