#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp chức năng quản lý kho mô hình phân đoạn tự động.

Module này quản lý việc tải xuống, cập nhật và theo dõi các mô hình
phân đoạn tự động có sẵn cho QuangTPS.
"""

import os
import json
import logging
import requests
import zipfile
import tempfile
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRepository:
    """
    Quản lý kho mô hình phân đoạn tự động.
    
    Class này quản lý các mô hình có sẵn, tải xuống mô hình mới,
    và cung cấp thông tin về mô hình cho engine phân đoạn tự động.
    """
    
    # URL cơ sở đến kho mô hình mặc định
    DEFAULT_REPOSITORY_URL = "https://raw.githubusercontent.com/QuangMacHUST/QuangTPS-models/main"
    
    # Tên file danh sách mô hình
    MODEL_LIST_FILENAME = "model_list.json"
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Khởi tạo kho mô hình.
        
        Parameters
        ----------
        models_dir : str, optional
            Thư mục lưu trữ mô hình. Nếu None, sẽ sử dụng thư mục mặc định
            trong cài đặt ứng dụng.
        """
        # Xác định thư mục lưu trữ mô hình
        if models_dir is None:
            # Sử dụng thư mục mặc định trong thư mục dữ liệu người dùng
            from quangtps.common.paths import get_app_data_dir
            models_dir = os.path.join(get_app_data_dir(), "models", "auto_segmentation")
        
        self.models_dir = Path(models_dir)
        self.local_model_list_path = self.models_dir / self.MODEL_LIST_FILENAME
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Danh sách mô hình có sẵn
        self.available_models: List[Dict] = []
        
        # Các URL kho mô hình
        self.repository_urls = [self.DEFAULT_REPOSITORY_URL]
        
        # Tải danh sách mô hình từ local
        self._load_local_model_list()
    
    def _load_local_model_list(self):
        """Tải danh sách mô hình từ file local."""
        if not self.local_model_list_path.exists():
            # Tạo file danh sách mô hình rỗng nếu chưa tồn tại
            self.available_models = []
            self._save_local_model_list()
            return
        
        try:
            with open(self.local_model_list_path, 'r') as f:
                self.available_models = json.load(f)
            
            # Kiểm tra xem mô hình có tồn tại trên hệ thống hay không
            for model in self.available_models:
                model_path = self.models_dir / model['directory']
                model['available'] = model_path.exists()
                
        except Exception as e:
            logger.error(f"Error loading model list: {str(e)}")
            self.available_models = []
    
    def _save_local_model_list(self):
        """Lưu danh sách mô hình vào file local."""
        try:
            with open(self.local_model_list_path, 'w') as f:
                json.dump(self.available_models, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model list: {str(e)}")
    
    def get_available_models(self) -> List[Dict]:
        """
        Lấy danh sách các mô hình có sẵn.
        
        Returns
        -------
        List[Dict]
            Danh sách mô hình có sẵn, mỗi mô hình là một dictionary
            với các thông tin như name, description, version, v.v.
        """
        return self.available_models
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Lấy đường dẫn đến thư mục mô hình.
        
        Parameters
        ----------
        model_name : str
            Tên mô hình
            
        Returns
        -------
        Optional[str]
            Đường dẫn đến thư mục mô hình, hoặc None nếu không tìm thấy
        """
        for model in self.available_models:
            if model['name'] == model_name and model.get('available', False):
                return str(self.models_dir / model['directory'])
        return None
    
    def update_model_list(self, force_reload: bool = False) -> bool:
        """
        Cập nhật danh sách mô hình từ kho từ xa.
        
        Parameters
        ----------
        force_reload : bool, optional
            Nếu True, sẽ tải lại danh sách mô hình từ kho từ xa
            ngay cả khi đã có bản local cache.
            
        Returns
        -------
        bool
            True nếu cập nhật thành công, False nếu có lỗi
        """
        success = False
        
        for repo_url in self.repository_urls:
            try:
                # Tạo URL đến file danh sách mô hình
                model_list_url = f"{repo_url}/{self.MODEL_LIST_FILENAME}"
                
                # Tải danh sách mô hình từ kho
                response = requests.get(model_list_url, timeout=10)
                
                if response.status_code == 200:
                    # Cập nhật danh sách mô hình
                    remote_models = response.json()
                    
                    # Cập nhật trạng thái available của các mô hình trong danh sách
                    for model in remote_models:
                        model_path = self.models_dir / model['directory']
                        model['available'] = model_path.exists()
                    
                    # Hợp nhất với danh sách mô hình hiện tại
                    self._merge_model_lists(remote_models)
                    
                    # Lưu danh sách mô hình vào file local
                    self._save_local_model_list()
                    
                    success = True
                    break
                    
            except Exception as e:
                logger.error(f"Error updating model list from {repo_url}: {str(e)}")
        
        return success
    
    def _merge_model_lists(self, remote_models: List[Dict]):
        """
        Hợp nhất danh sách mô hình từ xa với danh sách local.
        
        Parameters
        ----------
        remote_models : List[Dict]
            Danh sách mô hình từ kho từ xa
        """
        # Tạo dictionary từ danh sách mô hình hiện tại để dễ tìm kiếm
        local_models_dict = {model['name']: model for model in self.available_models}
        
        # Cập nhật hoặc thêm mô hình từ danh sách từ xa
        for remote_model in remote_models:
            name = remote_model['name']
            
            if name in local_models_dict:
                # Kiểm tra phiên bản
                local_version = local_models_dict[name].get('version', '0.0.0')
                remote_version = remote_model.get('version', '0.0.0')
                
                # Nếu phiên bản từ xa mới hơn, cập nhật thông tin mô hình
                if self._compare_versions(remote_version, local_version) > 0:
                    # Giữ lại trạng thái available
                    is_available = local_models_dict[name].get('available', False)
                    local_models_dict[name] = remote_model
                    local_models_dict[name]['available'] = is_available
            else:
                # Thêm mô hình mới vào danh sách
                local_models_dict[name] = remote_model
                local_models_dict[name]['available'] = False
        
        # Cập nhật danh sách mô hình
        self.available_models = list(local_models_dict.values())
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """
        So sánh hai phiên bản, trả về 1 nếu version1 > version2,
        -1 nếu version1 < version2, 0 nếu bằng nhau.
        
        Parameters
        ----------
        version1 : str
            Phiên bản thứ nhất
        version2 : str
            Phiên bản thứ hai
            
        Returns
        -------
        int
            1 nếu version1 > version2, -1 nếu version1 < version2, 0 nếu bằng nhau
        """
        v1_parts = [int(part) for part in version1.split('.')]
        v2_parts = [int(part) for part in version2.split('.')]
        
        # Đảm bảo cả hai danh sách có cùng độ dài
        while len(v1_parts) < len(v2_parts):
            v1_parts.append(0)
        while len(v2_parts) < len(v1_parts):
            v2_parts.append(0)
        
        # So sánh từng phần
        for i in range(len(v1_parts)):
            if v1_parts[i] > v2_parts[i]:
                return 1
            elif v1_parts[i] < v2_parts[i]:
                return -1
        
        return 0
    
    def download_model(self, model_name: str, progress_callback=None) -> bool:
        """
        Tải xuống một mô hình từ kho từ xa.
        
        Parameters
        ----------
        model_name : str
            Tên mô hình cần tải xuống
        progress_callback : callable, optional
            Hàm callback để cập nhật tiến trình tải xuống,
            nhận một tham số là phần trăm hoàn thành (0-100)
            
        Returns
        -------
        bool
            True nếu tải xuống thành công, False nếu có lỗi
        """
        # Tìm thông tin mô hình
        model_info = None
        for model in self.available_models:
            if model['name'] == model_name:
                model_info = model
                break
        
        if model_info is None:
            logger.error(f"Model {model_name} not found in the repository")
            return False
        
        # Kiểm tra xem mô hình đã có sẵn chưa
        if model_info.get('available', False):
            logger.info(f"Model {model_name} is already available")
            return True
        
        # Tải xuống mô hình
        try:
            # Lấy URL tải xuống
            download_url = None
            for repo_url in self.repository_urls:
                if 'url' in model_info:
                    # Nếu có URL tải xuống cụ thể
                    download_url = model_info['url']
                else:
                    # Mặc định URL tải xuống dựa trên repo
                    download_url = f"{repo_url}/models/{model_info['directory']}.zip"
                
                # Kiểm tra URL có thể truy cập không
                try:
                    response = requests.head(download_url, timeout=5)
                    if response.status_code == 200:
                        break
                    else:
                        download_url = None
                except:
                    download_url = None
            
            if download_url is None:
                logger.error(f"Could not find a valid download URL for model {model_name}")
                return False
            
            # Tạo thư mục tạm để tải xuống
            with tempfile.TemporaryDirectory() as temp_dir:
                # Tải xuống file zip
                zip_path = os.path.join(temp_dir, f"{model_name}.zip")
                
                # Tiến hành tải xuống với cập nhật tiến trình
                with requests.get(download_url, stream=True) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(zip_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # Cập nhật tiến trình
                                if progress_callback and total_size > 0:
                                    progress = int(100 * downloaded / total_size)
                                    progress_callback(progress)
                
                # Giải nén file zip
                model_dir = self.models_dir / model_info['directory']
                os.makedirs(model_dir, exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)
            
            # Cập nhật trạng thái mô hình
            for model in self.available_models:
                if model['name'] == model_name:
                    model['available'] = True
                    break
            
            # Lưu danh sách mô hình
            self._save_local_model_list()
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            return False
    
    def add_repository(self, repo_url: str):
        """
        Thêm một URL kho mô hình mới.
        
        Parameters
        ----------
        repo_url : str
            URL của kho mô hình
        """
        if repo_url not in self.repository_urls:
            self.repository_urls.append(repo_url)
    
    def remove_model(self, model_name: str) -> bool:
        """
        Xóa một mô hình khỏi hệ thống.
        
        Parameters
        ----------
        model_name : str
            Tên mô hình cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu có lỗi
        """
        # Tìm thông tin mô hình
        model_info = None
        for model in self.available_models:
            if model['name'] == model_name:
                model_info = model
                break
        
        if model_info is None or not model_info.get('available', False):
            logger.error(f"Model {model_name} not found or not available")
            return False
        
        try:
            # Xóa thư mục mô hình
            model_dir = self.models_dir / model_info['directory']
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Cập nhật trạng thái mô hình
            for model in self.available_models:
                if model['name'] == model_name:
                    model['available'] = False
                    break
            
            # Lưu danh sách mô hình
            self._save_local_model_list()
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing model {model_name}: {str(e)}")
            return False
