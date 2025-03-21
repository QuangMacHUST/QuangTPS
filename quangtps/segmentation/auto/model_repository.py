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
    
    def fetch_registry(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Tải danh sách các mô hình có sẵn từ registry từ xa.
        
        Parameters:
            force_update (bool): Buộc cập nhật từ server, bỏ qua cache.
            
        Returns:
            Dict[str, Any]: Danh sách các mô hình có sẵn.
        """
        try:
            # Kiểm tra cache nếu không buộc cập nhật
            if not force_update and self.registry:
                return self.registry
            
            # Tải registry từ server
            response = requests.get(MODEL_REGISTRY_URL, timeout=10)
            response.raise_for_status()
            
            # Lưu vào biến registry
            self.registry = response.json()
            logger.info(f"Đã tải danh sách {len(self.registry)} mô hình từ registry")
            
            # Thêm thông tin về tình trạng cài đặt
            for model_id, model_info in self.registry.items():
                model_info['installed'] = model_id in self.installed_models
                if model_id in self.installed_models:
                    local_version = self.installed_models[model_id].get('version', '0.0.0')
                    remote_version = model_info.get('version', '0.0.0')
                    model_info['update_available'] = local_version != remote_version
                    model_info['local_version'] = local_version
            
            return self.registry
        except requests.RequestException as e:
            logger.error(f"Không thể kết nối đến registry: {str(e)}")
            return {}
        except json.JSONDecodeError:
            logger.error("Dữ liệu registry không hợp lệ")
            return {}
        except Exception as e:
            logger.error(f"Lỗi khi tải registry: {str(e)}", exc_info=True)
            return {}
    
    def download_model(self, model_id: str, callback=None) -> bool:
        """
        Tải xuống mô hình từ registry.
        
        Parameters:
            model_id (str): ID của mô hình cần tải xuống.
            callback (callable, optional): Hàm callback để cập nhật tiến trình tải xuống.
            
        Returns:
            bool: True nếu tải xuống thành công, False nếu thất bại.
        """
        # Đảm bảo đã tải registry
        if not self.registry:
            self.fetch_registry()
        
        if model_id not in self.registry:
            logger.error(f"Không tìm thấy mô hình {model_id} trong registry")
            return False
        
        model_info = self.registry[model_id]
        download_url = model_info.get('download_url')
        
        if not download_url:
            logger.error(f"Không có URL tải xuống cho mô hình {model_id}")
            return False
        
        # Khởi tạo tiến trình tải xuống
        self.download_progress[model_id] = {
            'status': 'starting',
            'progress': 0,
            'error': None
        }
        
        # Tạo và bắt đầu luồng tải xuống
        download_thread = threading.Thread(
            target=self._download_model_thread,
            args=(model_id, download_url, callback)
        )
        self.download_threads[model_id] = download_thread
        download_thread.start()
        
        return True
    
    def _download_model_thread(self, model_id: str, download_url: str, callback=None):
        """
        Luồng tải xuống mô hình.
        
        Parameters:
            model_id (str): ID của mô hình.
            download_url (str): URL tải xuống mô hình.
            callback (callable, optional): Hàm callback để cập nhật tiến trình.
        """
        try:
            # Cập nhật trạng thái
            self.download_progress[model_id]['status'] = 'downloading'
            
            # Tạo thư mục tạm thời để tải xuống
            with tempfile.TemporaryDirectory() as temp_dir:
                # Tải xuống file
                zip_path = os.path.join(temp_dir, f"{model_id}.zip")
                
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                
                # Lấy kích thước file
                total_size = int(response.headers.get('content-length', 0))
                
                # Tải xuống từng phần nhỏ và cập nhật tiến trình
                downloaded_size = 0
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Cập nhật tiến trình
                            if total_size > 0:
                                progress = int(downloaded_size * 100 / total_size)
                                self.download_progress[model_id]['progress'] = progress
                                
                                if callback:
                                    callback(model_id, progress)
                
                # Giải nén file
                self.download_progress[model_id]['status'] = 'extracting'
                model_dir = os.path.join(self.models_dir, model_id)
                
                # Xóa thư mục cũ nếu có
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                
                # Tạo thư mục mới
                os.makedirs(model_dir)
                
                # Giải nén file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)
                
                # Cập nhật thông tin metadata
                metadata_file = os.path.join(model_dir, 'metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Đảm bảo ID trong metadata trùng với ID trong registry
                    metadata['id'] = model_id
                    
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                else:
                    # Tạo metadata.json nếu không có
                    metadata = self.registry[model_id].copy()
                    metadata['id'] = model_id
                    
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                
                # Cập nhật danh sách mô hình đã cài đặt
                self.load_local_models()
                
                # Hoàn tất
                self.download_progress[model_id]['status'] = 'completed'
                self.download_progress[model_id]['progress'] = 100
                
                if callback:
                    callback(model_id, 100, status='completed')
                
                logger.info(f"Đã tải xuống và cài đặt mô hình {model_id}")
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Lỗi khi tải xuống mô hình {model_id}: {error_msg}", exc_info=True)
            
            self.download_progress[model_id]['status'] = 'failed'
            self.download_progress[model_id]['error'] = error_msg
            
            if callback:
                callback(model_id, -1, status='failed', error=error_msg)
    
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
        Lấy danh sách các mô hình có sẵn từ registry, có thể lọc theo danh mục.
        
        Parameters:
            category (Optional[str]): Danh mục để lọc (ví dụ: 'brain', 'lung'...).
            
        Returns:
            List[Dict[str, Any]]: Danh sách thông tin các mô hình.
        """
        if not self.registry:
            self.fetch_registry()
        
        models = []
        
        for model_id, model_info in self.registry.items():
            if category is None or model_info.get('category') == category:
                models.append({
                    'id': model_id,
                    **model_info
                })
        
        return models
    
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
        Kiểm tra các bản cập nhật cho các mô hình đã cài đặt.
        
        Returns:
            Dict[str, Any]: Thông tin về các mô hình có bản cập nhật.
        """
        # Đảm bảo đã tải registry mới nhất
        self.fetch_registry(force_update=True)
        
        updates = {}
        
        for model_id, model_info in self.installed_models.items():
            if model_id in self.registry:
                local_version = model_info.get('version', '0.0.0')
                remote_version = self.registry[model_id].get('version', '0.0.0')
                
                if local_version != remote_version:
                    updates[model_id] = {
                        'name': model_info.get('name', model_id),
                        'current_version': local_version,
                        'available_version': remote_version,
                        'update_url': self.registry[model_id].get('download_url'),
                        'release_notes': self.registry[model_id].get('release_notes', '')
                    }
        
        return updates


# Khởi tạo đối tượng ModelRepository global
model_repository = ModelRepository() 