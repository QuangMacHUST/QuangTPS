#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module điều khiển hệ thống MLC (Multi-Leaf Collimator).

Module này cung cấp các lớp và phương thức để điều khiển 
hệ thống MLC trong liệu trình xạ trị.
"""

import logging
from typing import Dict, Any
from quangtps.treatment.mlc.mlc_model import MLCModel

logger = logging.getLogger(__name__)

class MLCController:
    """
    Lớp điều khiển hệ thống MLC (Multi-Leaf Collimator).
    
    Lớp này cung cấp các phương thức để điều khiển các lá MLC, cài đặt
    và kiểm tra các trường hình dạng, và tương tác với hệ thống máy xạ trị.
    """
    
    def __init__(self, mlc_model: MLCModel):
        """
        Khởi tạo một bộ điều khiển MLC.
        
        Parameters
        ----------
        mlc_model : MLCModel
            Mô hình MLC được sử dụng cho việc điều khiển
        """
        self.mlc_model = mlc_model
        self.current_positions = {}
        self.target_positions = {}
        self._is_initialized = False
        self._is_connected = False
        
    def initialize(self) -> bool:
        """
        Khởi tạo kết nối với hệ thống MLC.
        
        Returns
        -------
        bool
            True nếu khởi tạo thành công, False nếu có lỗi
        """
        try:
            # Mô phỏng việc khởi tạo kết nối với MLC
            self._is_initialized = True
            logger.info("MLC controller initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing MLC controller: {str(e)}")
            return False
        
    def connect(self) -> bool:
        """
        Kết nối với hệ thống MLC vật lý.
        
        Returns
        -------
        bool
            True nếu kết nối thành công, False nếu có lỗi
        """
        if not self._is_initialized:
            logger.warning("MLC controller not initialized")
            return False
        
        try:
            # Mô phỏng việc kết nối với MLC
            self._is_connected = True
            logger.info("Connected to MLC system")
            return True
        except Exception as e:
            logger.error(f"Error connecting to MLC system: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Ngắt kết nối với hệ thống MLC.
        
        Returns
        -------
        bool
            True nếu ngắt kết nối thành công, False nếu có lỗi
        """
        if not self._is_connected:
            logger.warning("MLC controller not connected")
            return True
        
        try:
            # Mô phỏng việc ngắt kết nối
            self._is_connected = False
            logger.info("Disconnected from MLC system")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from MLC system: {str(e)}")
            return False
    
    def set_leaf_positions(self, positions: Dict[int, float]) -> bool:
        """
        Cài đặt vị trí cho các lá MLC.
        
        Parameters
        ----------
        positions : Dict[int, float]
            Dictionary chứa vị trí của các lá, khóa là ID của lá, giá trị là vị trí (mm)
        
        Returns
        -------
        bool
            True nếu cài đặt thành công, False nếu có lỗi
        """
        if not self._is_connected:
            logger.warning("MLC controller not connected")
            return False
        
        try:
            # Kiểm tra tính hợp lệ của vị trí
            for leaf_id, position in positions.items():
                if not self.mlc_model.is_valid_position(leaf_id, position):
                    logger.warning(f"Invalid position {position} for leaf {leaf_id}")
                    return False
            
            self.target_positions.update(positions)
            logger.info("Leaf positions set successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting leaf positions: {str(e)}")
            return False
    
    def move_leaves(self) -> bool:
        """
        Di chuyển các lá MLC đến vị trí đích.
        
        Returns
        -------
        bool
            True nếu di chuyển thành công, False nếu có lỗi
        """
        if not self._is_connected:
            logger.warning("MLC controller not connected")
            return False
        
        try:
            # Mô phỏng việc di chuyển các lá
            self.current_positions = self.target_positions.copy()
            logger.info("Leaves moved to target positions")
            return True
        except Exception as e:
            logger.error(f"Error moving leaves: {str(e)}")
            return False
    
    def get_current_positions(self) -> Dict[int, float]:
        """
        Lấy vị trí hiện tại của các lá MLC.
        
        Returns
        -------
        Dict[int, float]
            Dictionary chứa vị trí hiện tại của các lá
        """
        return self.current_positions.copy()
    
    def verify_positions(self) -> bool:
        """
        Kiểm tra xem các lá MLC có ở đúng vị trí đích không.
        
        Returns
        -------
        bool
            True nếu các lá ở đúng vị trí, False nếu không
        """
        if not self._is_connected:
            logger.warning("MLC controller not connected")
            return False
        
        for leaf_id, target_pos in self.target_positions.items():
            current_pos = self.current_positions.get(leaf_id)
            if current_pos is None or abs(current_pos - target_pos) > 0.5:  # Dung sai 0.5mm
                return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của hệ thống MLC.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin trạng thái
        """
        return {
            "initialized": self._is_initialized,
            "connected": self._is_connected,
            "leaf_count": self.mlc_model.get_leaf_count(),
            "model": self.mlc_model.get_model_name()
        }