#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module thư viện mô hình MLC (Multi-Leaf Collimator).

Module này cung cấp tập hợp các mô hình MLC định nghĩa sẵn
cùng với các phương thức để truy cập, lấy và tạo mới mô hình.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from quangtps.treatment.mlc.mlc_model import MLCModel
from quangtps.common.paths import get_data_dir

logger = logging.getLogger(__name__)

class MLCModelLibrary:
    """
    Thư viện các mô hình MLC.
    
    Lớp này cung cấp một tập hợp các mô hình MLC định nghĩa sẵn
    và các phương thức để truy cập chúng.
    """
    
    # Dictionary lưu trữ các mô hình
    _models: Dict[str, MLCModel] = {}
    
    # Dictionary ánh xạ từ model_name sang manufacturer+model_name
    _name_to_id: Dict[str, str] = {}
    
    @classmethod
    def initialize(cls):
        """Khởi tạo thư viện với các mô hình định nghĩa sẵn."""
        
        if cls._models:
            return  # Đã khởi tạo rồi
        
        # Khởi tạo các mô hình Varian
        cls.add_model(MLCModel(
            model_name="Millennium120",
            manufacturer="Varian",
            leaf_count=120,
            leaf_width=5.0,  # mm
            max_field_size=(400.0, 400.0),  # mm
            max_travel=200.0  # mm
        ))
        
        cls.add_model(MLCModel(
            model_name="HD120",
            manufacturer="Varian",
            leaf_count=120,
            leaf_width=2.5,  # mm
            max_field_size=(400.0, 400.0),  # mm
            max_travel=200.0  # mm
        ))
        
        # Khởi tạo các mô hình Elekta
        cls.add_model(MLCModel(
            model_name="Agility",
            manufacturer="Elekta",
            leaf_count=160,
            leaf_width=5.0,  # mm
            max_field_size=(400.0, 400.0),  # mm
            max_travel=200.0  # mm
        ))
        
        cls.add_model(MLCModel(
            model_name="MLCi2",
            manufacturer="Elekta",
            leaf_count=80,
            leaf_width=10.0,  # mm
            max_field_size=(400.0, 400.0),  # mm
            max_travel=200.0  # mm
        ))
        
        # Khởi tạo các mô hình Siemens
        cls.add_model(MLCModel(
            model_name="160MLC",
            manufacturer="Siemens",
            leaf_count=160,
            leaf_width=5.0,  # mm
            max_field_size=(400.0, 400.0),  # mm
            max_travel=200.0  # mm
        ))
        
        # Tải các mô hình tùy chỉnh từ file cấu hình
        cls._load_custom_models()
        
    @classmethod
    def add_model(cls, model: MLCModel) -> bool:
        """
        Thêm một mô hình MLC vào thư viện.
        
        Args:
            model: Mô hình MLC cần thêm
            
        Returns:
            bool: True nếu thêm thành công, False nếu thất bại
        """
        model_id = f"{model.manufacturer}_{model.model_name}"
        cls._models[model_id] = model
        cls._name_to_id[model.model_name] = model_id
        return True
    
    @classmethod
    def get_model(cls, model_name: str, manufacturer: Optional[str] = None) -> Optional[MLCModel]:
        """
        Lấy một mô hình MLC theo tên và nhà sản xuất.
        
        Args:
            model_name: Tên mô hình
            manufacturer: Nhà sản xuất (không bắt buộc)
            
        Returns:
            Optional[MLCModel]: Mô hình MLC nếu tìm thấy, None nếu không
        """
        # Đảm bảo thư viện đã được khởi tạo
        if not cls._models:
            cls.initialize()
        
        # Nếu có thông tin manufacturer, tìm theo model_id
        if manufacturer:
            model_id = f"{manufacturer}_{model_name}"
            return cls._models.get(model_id)
        
        # Không có manufacturer, tìm theo tên
        model_id = cls._name_to_id.get(model_name)
        if model_id:
            return cls._models.get(model_id)
        
        # Tìm kiếm một cách không phân biệt hoa thường
        model_name_lower = model_name.lower()
        for mid, model in cls._models.items():
            if model.model_name.lower() == model_name_lower:
                return model
        
        # Không tìm thấy
        return None
    
    @classmethod
    def get_all_models(cls) -> List[MLCModel]:
        """
        Lấy danh sách tất cả các mô hình MLC có sẵn.
        
        Returns:
            List[MLCModel]: Danh sách các mô hình
        """
        # Đảm bảo thư viện đã được khởi tạo
        if not cls._models:
            cls.initialize()
            
        return list(cls._models.values())
    
    @classmethod
    def get_models_by_manufacturer(cls, manufacturer: str) -> List[MLCModel]:
        """
        Lấy danh sách các mô hình MLC theo nhà sản xuất.
        
        Args:
            manufacturer: Tên nhà sản xuất
            
        Returns:
            List[MLCModel]: Danh sách các mô hình của nhà sản xuất
        """
        # Đảm bảo thư viện đã được khởi tạo
        if not cls._models:
            cls.initialize()
            
        # Tìm tất cả các mô hình của nhà sản xuất
        manufacturer_lower = manufacturer.lower()
        return [model for model in cls._models.values() 
                if model.manufacturer.lower() == manufacturer_lower]
    
    @classmethod
    def _load_custom_models(cls):
        """Tải các mô hình tùy chỉnh từ file cấu hình."""
        
        try:
            # Tìm đường dẫn đến file cấu hình
            config_dir = Path(get_data_dir()) / "machine_data" / "mlc"
            config_file = config_dir / "custom_mlc_models.json"
            
            # Nếu file không tồn tại thì tạo file trống
            if not config_file.exists():
                config_dir.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump([], f)
                return
            
            # Đọc file cấu hình
            with open(config_file, 'r') as f:
                custom_models = json.load(f)
            
            # Tạo các mô hình từ dữ liệu đọc được
            for model_data in custom_models:
                try:
                    model = MLCModel(
                        model_name=model_data.get("model_name", "Unknown"),
                        manufacturer=model_data.get("manufacturer", "Custom"),
                        leaf_count=model_data.get("leaf_count", 120),
                        leaf_width=model_data.get("leaf_width", 5.0),
                        max_field_size=model_data.get("max_field_size", (400.0, 400.0)),
                        max_travel=model_data.get("max_travel", 200.0),
                        mlc_type=model_data.get("mlc_type", MLCModel.TYPE_MULTILEAF)
                    )
                    cls.add_model(model)
                except Exception as e:
                    logger.error(f"Lỗi khi tạo mô hình MLC tùy chỉnh: {e}")
        
        except Exception as e:
            logger.error(f"Lỗi khi tải các mô hình MLC tùy chỉnh: {e}")
    
    @classmethod
    def save_custom_model(cls, model: MLCModel) -> bool:
        """
        Lưu một mô hình MLC tùy chỉnh vào file cấu hình.
        
        Args:
            model: Mô hình MLC cần lưu
            
        Returns:
            bool: True nếu lưu thành công, False nếu thất bại
        """
        try:
            # Tìm đường dẫn đến file cấu hình
            config_dir = Path(get_data_dir()) / "machine_data" / "mlc"
            config_file = config_dir / "custom_mlc_models.json"
            
            # Tạo thư mục nếu chưa tồn tại
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Đọc các mô hình hiện có
            custom_models = []
            if config_file.exists():
                with open(config_file, 'r') as f:
                    custom_models = json.load(f)
            
            # Chuyển đổi mô hình thành dữ liệu JSON
            model_data = {
                "model_name": model.model_name,
                "manufacturer": model.manufacturer,
                "leaf_count": model.leaf_count,
                "leaf_width": model.leaf_width,
                "max_field_size": model.max_field_size,
                "max_travel": model.max_travel,
                "mlc_type": model.mlc_type
            }
            
            # Kiểm tra xem mô hình đã tồn tại chưa
            for i, existing_model in enumerate(custom_models):
                if (existing_model.get("model_name") == model.model_name and 
                    existing_model.get("manufacturer") == model.manufacturer):
                    # Cập nhật mô hình hiện có
                    custom_models[i] = model_data
                    break
            else:
                # Thêm mô hình mới
                custom_models.append(model_data)
            
            # Lưu lại file cấu hình
            with open(config_file, 'w') as f:
                json.dump(custom_models, f, indent=2)
                
            # Thêm mô hình vào thư viện
            cls.add_model(model)
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình MLC tùy chỉnh: {e}")
            return False 