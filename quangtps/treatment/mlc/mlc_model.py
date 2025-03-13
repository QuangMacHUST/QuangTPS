#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module mô hình MLC (Multi-Leaf Collimator).

Module này cung cấp các lớp và phương thức mô tả 
các kiểu và tham số kỹ thuật của các hệ thống MLC khác nhau.
"""

import logging
from typing import Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class MLCModel:
    """
    Lớp mô hình MLC (Multi-Leaf Collimator).
    
    Lớp này mô tả các thông số kỹ thuật và giới hạn
    cho các loại và mô hình MLC khác nhau sử dụng trong xạ trị.
    """
    
    # Constants for common MLC types
    TYPE_BINARY = "binary"
    TYPE_TERTIARY = "tertiary"
    TYPE_MULTILEAF = "multileaf"
    
    # Common manufacturers
    MANUFACTURER_VARIAN = "Varian"
    MANUFACTURER_ELEKTA = "Elekta"
    MANUFACTURER_SIEMENS = "Siemens"
    
    def __init__(self, 
                 model_name: str, 
                 manufacturer: str,
                 leaf_count: int,
                 leaf_width: float,
                 max_field_size: Tuple[float, float],
                 max_travel: float,
                 mlc_type: str = TYPE_MULTILEAF):
        """
        Khởi tạo một mô hình MLC.
        
        Parameters
        ----------
        model_name : str
            Tên mô hình MLC
        manufacturer : str
            Nhà sản xuất MLC
        leaf_count : int
            Số lượng lá trong MLC
        leaf_width : float
            Chiều rộng của mỗi lá (mm)
        max_field_size : Tuple[float, float]
            Kích thước trường tối đa (chiều rộng, chiều dài) (mm)
        max_travel : float
            Khoảng di chuyển tối đa của lá (mm)
        mlc_type : str, optional
            Loại MLC (binary, tertiary, multileaf)
        """
        self.model_name = model_name
        self.manufacturer = manufacturer
        self.leaf_count = leaf_count
        self.leaf_width = leaf_width
        self.max_field_size = max_field_size
        self.max_travel = max_travel
        self.mlc_type = mlc_type
        
        # Map leaf IDs (0-based) to their properties
        self.leaves = {}
        for i in range(leaf_count):
            self.leaves[i] = {
                "width": leaf_width,
                "max_travel": max_travel,
                "min_position": -max_travel/2,
                "max_position": max_travel/2
            }
    
    def get_model_name(self) -> str:
        """
        Lấy tên mô hình MLC.
        
        Returns
        -------
        str
            Tên mô hình MLC
        """
        return self.model_name
    
    def get_manufacturer(self) -> str:
        """
        Lấy tên nhà sản xuất MLC.
        
        Returns
        -------
        str
            Tên nhà sản xuất
        """
        return self.manufacturer
    
    def get_leaf_count(self) -> int:
        """
        Lấy số lượng lá trong MLC.
        
        Returns
        -------
        int
            Số lượng lá
        """
        return self.leaf_count
    
    def get_leaf_width(self) -> float:
        """
        Lấy chiều rộng của lá MLC.
        
        Returns
        -------
        float
            Chiều rộng của lá (mm)
        """
        return self.leaf_width
    
    def get_max_field_size(self) -> Tuple[float, float]:
        """
        Lấy kích thước trường tối đa.
        
        Returns
        -------
        Tuple[float, float]
            Kích thước trường tối đa (chiều rộng, chiều dài) (mm)
        """
        return self.max_field_size
    
    def get_max_travel(self) -> float:
        """
        Lấy khoảng di chuyển tối đa của lá MLC.
        
        Returns
        -------
        float
            Khoảng di chuyển tối đa (mm)
        """
        return self.max_travel
    
    def is_valid_position(self, leaf_id: int, position: float) -> bool:
        """
        Kiểm tra xem vị trí có hợp lệ cho lá MLC cụ thể không.
        
        Parameters
        ----------
        leaf_id : int
            ID của lá MLC (0-based)
        position : float
            Vị trí cần kiểm tra (mm)
            
        Returns
        -------
        bool
            True nếu vị trí hợp lệ, False nếu không
        """
        if leaf_id not in self.leaves:
            logger.warning(f"Invalid leaf ID: {leaf_id}")
            return False
        
        leaf = self.leaves[leaf_id]
        if position < leaf["min_position"] or position > leaf["max_position"]:
            logger.warning(f"Position {position} out of range for leaf {leaf_id}")
            return False
        
        return True
    
    def get_leaf_properties(self, leaf_id: int) -> Optional[Dict[str, Any]]:
        """
        Lấy thuộc tính của lá MLC cụ thể.
        
        Parameters
        ----------
        leaf_id : int
            ID của lá MLC (0-based)
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary chứa thuộc tính của lá, hoặc None nếu ID không hợp lệ
        """
        return self.leaves.get(leaf_id)
    
    def get_all_leaves(self) -> Dict[int, Dict[str, Any]]:
        """
        Lấy thuộc tính của tất cả các lá MLC.
        
        Returns
        -------
        Dict[int, Dict[str, Any]]
            Dictionary chứa thuộc tính của tất cả các lá
        """
        return self.leaves.copy()
    
    @classmethod
    def create_standard_model(cls, model_name: str) -> Optional['MLCModel']:
        """
        Tạo một mô hình MLC chuẩn dựa trên tên mô hình.
        
        Parameters
        ----------
        model_name : str
            Tên mô hình MLC chuẩn
            
        Returns
        -------
        Optional[MLCModel]
            Đối tượng MLCModel, hoặc None nếu không tìm thấy mô hình
        """
        standard_models = {
            "Varian_Millennium120": {
                "manufacturer": cls.MANUFACTURER_VARIAN,
                "leaf_count": 120,
                "leaf_width": 5.0,
                "max_field_size": (400.0, 400.0),
                "max_travel": 200.0,
                "mlc_type": cls.TYPE_MULTILEAF
            },
            "Varian_HD120": {
                "manufacturer": cls.MANUFACTURER_VARIAN,
                "leaf_count": 120,
                "leaf_width": 2.5,
                "max_field_size": (400.0, 400.0),
                "max_travel": 200.0,
                "mlc_type": cls.TYPE_MULTILEAF
            },
            "Elekta_Agility": {
                "manufacturer": cls.MANUFACTURER_ELEKTA,
                "leaf_count": 160,
                "leaf_width": 5.0,
                "max_field_size": (400.0, 400.0),
                "max_travel": 200.0,
                "mlc_type": cls.TYPE_MULTILEAF
            },
            "Siemens_160": {
                "manufacturer": cls.MANUFACTURER_SIEMENS,
                "leaf_count": 160,
                "leaf_width": 5.0,
                "max_field_size": (400.0, 400.0),
                "max_travel": 200.0,
                "mlc_type": cls.TYPE_MULTILEAF
            }
        }
        
        if model_name not in standard_models:
            logger.warning(f"Unknown MLC model: {model_name}")
            return None
        
        specs = standard_models[model_name]
        return cls(
            model_name=model_name,
            manufacturer=specs["manufacturer"],
            leaf_count=specs["leaf_count"],
            leaf_width=specs["leaf_width"],
            max_field_size=specs["max_field_size"],
            max_travel=specs["max_travel"],
            mlc_type=specs["mlc_type"]
        )