#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý thiết bị gia tốc (Accelerator).

Module này cung cấp các lớp cơ sở và phương thức để định nghĩa và quản lý
các loại máy gia tốc được sử dụng trong xạ trị.
"""

import uuid
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Accelerator:
    """
    Lớp cơ sở đại diện cho một thiết bị gia tốc.
    
    Lớp này đóng vai trò là lớp cơ sở cho tất cả các loại máy gia tốc
    như Linac, Cyclotron, Gamma knife, v.v.
    """
    
    def __init__(self, 
                machine_name: str, 
                manufacturer: str = "Generic", 
                machine_id: Optional[str] = None):
        """
        Khởi tạo một thiết bị gia tốc.
        
        Parameters
        ----------
        machine_name : str
            Tên của thiết bị
        manufacturer : str, optional
            Nhà sản xuất của thiết bị
        machine_id : str, optional
            ID duy nhất của thiết bị. Nếu không cung cấp, một ID mới sẽ được tạo.
        """
        self.machine_name = machine_name
        self.manufacturer = manufacturer
        self.machine_id = machine_id if machine_id else str(uuid.uuid4())
        
        # Loại gia tốc (sẽ được ghi đè bởi các lớp con)
        self.accelerator_type = "GENERIC"
        
        # Thông tin bổ sung
        self.description = ""
        self.metadata = {}
    
    def set_description(self, description: str):
        """
        Thiết lập mô tả cho thiết bị.
        
        Parameters
        ----------
        description : str
            Mô tả của thiết bị
        """
        self.description = description
    
    def add_metadata(self, key: str, value: Any):
        """
        Thêm thông tin metadata cho thiết bị.
        
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
        Chuyển đổi thông tin thiết bị thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin thiết bị
        """
        return {
            "machine_name": self.machine_name,
            "manufacturer": self.manufacturer,
            "machine_id": self.machine_id,
            "accelerator_type": self.accelerator_type,
            "description": self.description,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Accelerator':
        """
        Tạo đối tượng Accelerator từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin thiết bị
            
        Returns
        -------
        Accelerator
            Đối tượng Accelerator
        """
        accelerator = cls(
            machine_name=data["machine_name"],
            manufacturer=data["manufacturer"],
            machine_id=data["machine_id"]
        )
        
        accelerator.accelerator_type = data["accelerator_type"]
        accelerator.description = data["description"]
        accelerator.metadata = data["metadata"]
        
        return accelerator