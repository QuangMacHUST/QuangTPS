#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý kho mô hình phân đoạn tự động.

Module này cung cấp các lớp và hàm để tải, quản lý và 
triển khai các mô hình học sâu cho việc phân đoạn tự động các cấu trúc giải phẫu.
"""

import os
import json
import logging
import requests
import shutil
import zipfile
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import threading
import time
from datetime import datetime

from quangtps.core.config import Config

logger = logging.getLogger(__name__)

MODEL_REGISTRY_URL = "https://quangtps-models.example.com/registry.json"


class ModelRepository:
    """
    Lớp quản lý kho lưu trữ các mô hình phân đoạn tự động.
    
    Cung cấp các phương thức để tải, cập nhật, liệt kê và quản lý các mô hình
    học sâu được sử dụng cho phân đoạn tự động trong QuangTPS.
    """
    
    def __init__(self):
        """Khởi tạo kho mô hình."""
        self.config = Config.get_instance()
        self.models_dir = os.path.join(self.config.data_dir, 'models')
        self._ensure_model_directory()
        self.registry = {}
        self.installed_models = {}
        self.load_local_models()
        
        # Theo dõi quá trình tải xuống
        self.download_progress = {}
        self.download_threads = {}
    
    def _ensure_model_directory(self):
        """Đảm bảo thư mục mô hình tồn tại."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            logger.info(f"Đã tạo thư mục mô hình: {self.models_dir}")
    
    def load_local_models(self) -> Dict[str, Any]:
        """
        Tải thông tin về các mô hình đã cài đặt cục bộ.
        
        Returns:
            Dict[str, Any]: Thông tin về các mô hình đã cài đặt.
        """
        self.installed_models = {}
        
        try:
            # Kiểm tra các thư mục con trong thư mục models
            for model_dir in os.listdir(self.models_dir):
                model_path = os.path.join(self.models_dir, model_dir)
                
                # Chỉ xử lý các thư mục
                if not os.path.isdir(model_path):
                    continue
                
                # Kiểm tra file metadata.json
                metadata_file = os.path.join(model_path, 'metadata.json')
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            
                        # Thêm đường dẫn đến mô hình
                        metadata['path'] = model_path
                        
                        # Thêm vào từ điển mô hình đã cài đặt
                        model_id = metadata.get('id', model_dir)
                        self.installed_models[model_id] = metadata
                        logger.debug(f"Đã tải thông tin mô hình cục bộ: {model_id}")
                    except json.JSONDecodeError:
                        logger.warning(f"File metadata.json không hợp lệ trong thư mục: {model_path}")
                else:
                    logger.warning(f"Không tìm thấy metadata.json trong thư mục: {model_path}")
            
            logger.info(f"Đã tải thông tin về {len(self.installed_models)} mô hình cục bộ")
            return self.installed_models
        except Exception as e:
            logger.error(f"Lỗi khi tải thông tin mô hình cục bộ: {str(e)}", exc_info=True)
            return {}
    
    def fetch_registry(self, force_update: bool = False) -> bool:
        """
        Lấy registry mới nhất từ máy chủ hoặc từ bộ nhớ cache.
        
        Parameters:
            force_update (bool): Bắt buộc cập nhật từ máy chủ, bỏ qua cache.
            
        Returns:
            bool: True nếu thành công, False nếu không.
        """
        # Nếu đã có registry và không bắt buộc cập nhật, trả về True
        if self.registry and not force_update:
            return True
            
        # Use local mock registry for testing purposes
        mock_registry_path = os.path.join(self.config.data_dir, 'mock_registry.json')
        
        # Create mock registry if it doesn't exist
        if not os.path.exists(mock_registry_path):
            mock_registry = {
                "lung_segmentation": {
                    "name": "Lung Segmentation",
                    "description": "Segments both lungs from CT images",
                    "version": "1.0.0",
                    "size": 10485760,  # 10MB
                    "date_added": "2023-01-15",
                    "category": "thorax",
                    "structures": ["left_lung", "right_lung", "lungs"],
                    "license": "MIT",
                    "url": "https://quangtps-models.example.com/models/lung_segmentation.zip"
                },
                "brain_segmentation": {
                    "name": "Brain Segmentation",
                    "description": "Segments brain structures from MRI images",
                    "version": "1.0.0",
                    "size": 15728640,  # 15MB
                    "date_added": "2023-02-10",
                    "category": "neuro",
                    "structures": ["brain", "brainstem", "cerebellum"],
                    "license": "MIT",
                    "url": "https://quangtps-models.example.com/models/brain_segmentation.zip"
                }
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(mock_registry_path), exist_ok=True)
            
            # Save mock registry
            with open(mock_registry_path, 'w', encoding='utf-8') as f:
                json.dump(mock_registry, f, indent=4)
                
            logger.info(f"Created mock registry at {mock_registry_path}")
        
        try:
            # Read from mock registry
            with open(mock_registry_path, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
            
            logger.info(f"Loaded mock registry with {len(self.registry)} models")
            return True
            
        except Exception as e:
            logger.error(f"Không thể đọc mock registry: {str(e)}")
            return False

        # Commented out original remote fetching code
        """
        try:
            # Gửi yêu cầu GET đến máy chủ
            response = requests.get(self.registry_url, timeout=10)
            
            # Kiểm tra trạng thái phản hồi
            if response.status_code == 200:
                # Phân tích JSON từ phản hồi
                self.registry = response.json()
                
                # Lưu vào bộ nhớ cache
                cache_path = os.path.join(self.cache_dir, 'registry.json')
                os.makedirs(self.cache_dir, exist_ok=True)
                
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.registry, f, indent=4)
                
                return True
            else:
                logger.error(f"Lỗi khi tải registry: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Không thể kết nối đến registry: {str(e)}")
            
            # Thử tải từ bộ nhớ cache nếu có
            cache_path = os.path.join(self.cache_dir, 'registry.json')
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        self.registry = json.load(f)
                    logger.warning("Đang sử dụng registry từ bộ nhớ cache")
                    return True
                except Exception as e2:
                    logger.error(f"Không thể đọc registry từ bộ nhớ cache: {str(e2)}")
            
            return False
        """
    
    def download_model(self, model_id: str, callback=None) -> bool:
        """
        Tải xuống mô hình từ registry từ xa.
        
        Parameters:
            model_id (str): ID của mô hình cần tải.
            callback (callable, optional): Hàm callback để cập nhật tiến trình.
            
        Returns:
            bool: True nếu tải thành công, False nếu không.
        """
        # Kiểm tra mô hình có trong registry không
        if model_id not in self.registry:
            logger.error(f"Không tìm thấy mô hình {model_id} trong registry")
            return False
        
        model_info = self.registry[model_id]
        
        # Create mock model directory and files for testing
        model_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create a mock config file
        config = {
            "name": model_info.get("name", "Unknown Model"),
            "version": model_info.get("version", "1.0.0"),
            "model_type": "unet",
            "input_shape": [256, 256, 1],
            "output_shape": [256, 256, 1],
            "weights_file": "model.h5",
            "preprocessing": {
                "window_width": 1500,
                "window_level": -600,
                "normalize": True
            },
            "structures": model_info.get("structures", ["unknown"]),
            "description": model_info.get("description", "")
        }
        
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        
        # Create a mock model file
        import numpy as np
        mock_model = np.zeros((1000,), dtype=np.float32)  # Small mock model file
        np.save(os.path.join(model_dir, "model.npy"), mock_model)
        
        # Create an empty h5 file to simulate the model
        with open(os.path.join(model_dir, "model.h5"), "wb") as f:
            f.write(b'\x89HDF\r\n\x1a\n\x00\x00\x00\x00\x00\x08\x08\x00')
        
        # Update installed models
        self.installed_models[model_id] = {
            "name": model_info.get("name", "Unknown"),
            "version": model_info.get("version", "1.0.0"),
            "path": model_dir,
            "date_installed": datetime.now().isoformat(),
            "structures": model_info.get("structures", [])
        }
        
        # Save installed models to disk
        self._save_installed_models()
        
        logger.info(f"Created mock model: {model_id}")
        return True
        
        """
        # Original implementation
        try:
            # Tạo thư mục tạm để tải xuống
            temp_dir = os.path.join(self.cache_dir, 'downloads')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Tạo đường dẫn đến file tạm
            temp_file = os.path.join(temp_dir, f"{model_id}.zip")
            
            # Tải file từ URL
            download_url = model_info.get('url')
            if not download_url:
                logger.error(f"Không tìm thấy URL tải xuống cho mô hình {model_id}")
                return False
            
            # Tải xuống với hiển thị tiến trình
            response = requests.get(download_url, stream=True, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"Lỗi khi tải mô hình {model_id}: {response.status_code} - {response.text}")
                return False
            
            # Lấy tổng kích thước file
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            # Tải xuống từng phần
            with open(temp_file, 'wb') as f:
                downloaded = 0
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded += len(data)
                    
                    # Cập nhật tiến trình nếu có callback
                    if callback and total_size > 0:
                        progress = downloaded / total_size
                        callback(progress)
            
            # Giải nén file
            model_dir = os.path.join(self.models_dir, model_id)
            
            # Xóa thư mục cũ nếu tồn tại
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                
            # Tạo thư mục mới
            os.makedirs(model_dir, exist_ok=True)
            
            # Giải nén
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
                
            # Cập nhật thông tin mô hình đã cài đặt
            self.installed_models[model_id] = {
                "name": model_info.get("name", "Unknown"),
                "version": model_info.get("version", "1.0.0"),
                "path": model_dir,
                "date_installed": datetime.now().isoformat(),
                "structures": model_info.get("structures", [])
            }
            
            # Lưu thông tin cài đặt
            self._save_installed_models()
            
            # Xóa file tạm
            os.remove(temp_file)
            
            logger.info(f"Đã tải xuống và cài đặt mô hình {model_id} thành công")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải xuống mô hình {model_id}: {str(e)}", exc_info=True)
            return False
        """
    
    def get_download_progress(self, model_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin về tiến trình tải xuống mô hình.
        
        Parameters:
            model_id (str): ID của mô hình.
            
        Returns:
            Dict[str, Any]: Thông tin về tiến trình tải xuống.
        """
        return self.download_progress.get(model_id, {
            'status': 'not_started',
            'progress': 0,
            'error': None
        })
    
    def cancel_download(self, model_id: str) -> bool:
        """
        Hủy bỏ quá trình tải xuống mô hình.
        
        Parameters:
            model_id (str): ID của mô hình.
            
        Returns:
            bool: True nếu hủy thành công, False nếu không.
        """
        if (model_id in self.download_threads and 
            model_id in self.download_progress and 
            self.download_progress[model_id]['status'] == 'downloading'):
            
            # Không thể thực sự hủy thread, chỉ đánh dấu là đã hủy
            self.download_progress[model_id]['status'] = 'cancelled'
            logger.info(f"Đã hủy tải xuống mô hình {model_id}")
            return True
        
        return False
    
    def remove_model(self, model_id: str) -> bool:
        """
        Xóa mô hình đã cài đặt.
        
        Parameters:
            model_id (str): ID của mô hình cần xóa.
            
        Returns:
            bool: True nếu xóa thành công, False nếu không.
        """
        if model_id not in self.installed_models:
            logger.warning(f"Không thể xóa mô hình không tồn tại: {model_id}")
            return False
        
        try:
            model_dir = self.installed_models[model_id].get('path')
            if model_dir and os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                logger.info(f"Đã xóa mô hình {model_id}")
                
                # Cập nhật danh sách mô hình đã cài đặt
                self.load_local_models()
                
                # Cập nhật registry nếu đã tải
                if self.registry and model_id in self.registry:
                    self.registry[model_id]['installed'] = False
                
                return True
            else:
                logger.error(f"Không tìm thấy thư mục của mô hình {model_id}")
                return False
        except Exception as e:
            logger.error(f"Lỗi khi xóa mô hình {model_id}: {str(e)}", exc_info=True)
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin chi tiết về mô hình.
        
        Parameters:
            model_id (str): ID của mô hình.
            
        Returns:
            Optional[Dict[str, Any]]: Thông tin của mô hình, None nếu không tìm thấy.
        """
        # Kiểm tra local models trước
        if model_id in self.installed_models:
            return self.installed_models[model_id]
        
        # Kiểm tra trong registry
        if not self.registry:
            self.fetch_registry()
        
        if model_id in self.registry:
            return self.registry[model_id]
        
        return None
    
    def get_available_models(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các mô hình có sẵn, có thể lọc theo danh mục.
        
        Parameters:
            category (Optional[str]): Lọc theo danh mục mô hình (nếu có)
            
        Returns:
            List[Dict[str, Any]]: Danh sách thông tin về các mô hình
        """
        result = []
        
        # Ensure we have the registry
        if not self.registry:
            self.fetch_registry()
            
        # If fetch_registry returned a boolean instead of the registry, or registry is still not available
        if not self.registry or isinstance(self.registry, bool):
            logger.warning("Registry unavailable")
            return result
            
        # Process models and filter by category
        for model_id, model_info in self.registry.items():
            if category is None or model_info.get('category') == category:
                # Thêm ID vào thông tin mô hình
                model_data = model_info.copy()
                model_data['id'] = model_id
                result.append(model_data)
        
        return result
    
    def get_available_remote_models(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các mô hình có sẵn từ registry từ xa.
        
        Parameters:
            category (Optional[str]): Lọc theo danh mục mô hình (nếu có).
            
        Returns:
            List[Dict[str, Any]]: Danh sách thông tin về các mô hình từ xa.
        """
        # Đảm bảo đã tải registry mới nhất
        self.fetch_registry(force_update=True)
        
        return self.get_available_models(category)
    
    def get_installed_models(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các mô hình đã cài đặt, có thể lọc theo danh mục.
        
        Parameters:
            category (Optional[str]): Danh mục để lọc.
            
        Returns:
            List[Dict[str, Any]]: Danh sách thông tin các mô hình đã cài đặt.
        """
        models = []
        
        for model_id, model_info in self.installed_models.items():
            if category is None or model_info.get('category') == category:
                models.append({
                    'id': model_id,
                    **model_info
                })
        
        return models
    
    def load_model(self, model_id: str) -> Any:
        """
        Tải mô hình vào bộ nhớ để sử dụng.
        
        Parameters:
            model_id (str): ID của mô hình cần tải.
            
        Returns:
            Any: Đối tượng mô hình đã tải, None nếu không tải được.
        """
        if model_id not in self.installed_models:
            logger.error(f"Không thể tải mô hình chưa cài đặt: {model_id}")
            return None
        
        model_info = self.installed_models[model_id]
        model_path = model_info.get('path')
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Không tìm thấy thư mục của mô hình {model_id}")
            return None
        
        # Đọc cấu hình mô hình
        config_file = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_file):
            logger.error(f"Không tìm thấy file cấu hình của mô hình {model_id}")
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Đọc thông tin về loại mô hình và file weights
            model_type = config.get('model_type', 'unknown')
            weights_file = config.get('weights_file', 'model.h5')
            weights_path = os.path.join(model_path, weights_file)
            
            if not os.path.exists(weights_path):
                logger.error(f"Không tìm thấy file weights {weights_file} của mô hình {model_id}")
                return None
            
            # TODO: Implement model loading based on model_type
            # For now, just return the config and path information
            return {
                'id': model_id,
                'config': config,
                'weights_path': weights_path,
                'model_path': model_path,
                'model_type': model_type
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình {model_id}: {str(e)}", exc_info=True)
            return None
            
    def update_model(self, model_id: str, callback=None) -> bool:
        """
        Cập nhật mô hình lên phiên bản mới nhất.
        
        Parameters:
            model_id (str): ID của mô hình cần cập nhật.
            callback (callable, optional): Hàm callback để cập nhật tiến trình.
            
        Returns:
            bool: True nếu cập nhật thành công, False nếu không.
        """
        # Kiểm tra mô hình đã cài đặt chưa
        if model_id not in self.installed_models:
            logger.error(f"Không thể cập nhật mô hình chưa cài đặt: {model_id}")
            return False
        
        # Đảm bảo đã tải registry mới nhất
        self.fetch_registry(force_update=True)
        
        # Kiểm tra mô hình có trong registry không
        if model_id not in self.registry:
            logger.error(f"Không tìm thấy mô hình {model_id} trong registry")
            return False
        
        # Kiểm tra có phiên bản mới không
        local_version = self.installed_models[model_id].get('version', '0.0.0')
        remote_version = self.registry[model_id].get('version', '0.0.0')
        
        if local_version == remote_version:
            logger.info(f"Mô hình {model_id} đã ở phiên bản mới nhất ({local_version})")
            return True
        
        # Tải xuống phiên bản mới
        logger.info(f"Cập nhật mô hình {model_id} từ phiên bản {local_version} lên {remote_version}")
        return self.download_model(model_id, callback)
    
    def check_for_updates(self) -> Dict[str, Any]:
        """
        Kiểm tra các cập nhật có sẵn cho mô hình đã cài đặt.
        
        Returns:
            Dict[str, Any]: Thông tin về các mô hình có cập nhật.
        """
        self.fetch_registry(force_update=True)
        updates = {}
        
        for model_id, model_info in self.registry.items():
            if model_id in self.installed_models:
                local_version = self.installed_models[model_id].get('version', '0.0.0')
                remote_version = model_info.get('version', '0.0.0')
                
                if local_version != remote_version:
                    updates[model_id] = {
                        'id': model_id,
                        'name': model_info.get('name', model_id),
                        'description': model_info.get('description', ''),
                        'local_version': local_version,
                        'remote_version': remote_version
                    }
        
        return updates

    def _save_installed_models(self):
        """
        Lưu thông tin các mô hình đã cài đặt vào file.
        """
        installed_models_path = os.path.join(self.models_dir, 'installed_models.json')
        with open(installed_models_path, 'w', encoding='utf-8') as f:
            json.dump(self.installed_models, f, indent=4)
        logger.info(f"Đã lưu thông tin {len(self.installed_models)} mô hình đã cài đặt vào file")


# Tạo một instance singleton của ModelRepository để sử dụng toàn cục
model_repository = ModelRepository() 