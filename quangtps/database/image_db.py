#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý cơ sở dữ liệu hình ảnh cho QuangTPS.

Module này định nghĩa các lớp và hàm để lưu trữ và truy xuất 
hình ảnh y tế từ cơ sở dữ liệu.
"""

import os
import json
import uuid
import numpy as np
import SimpleITK as sitk
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import h5py

from quangtps.core.logging import get_logger
from quangtps.database.db_connector import DBConnector
from quangtps.imaging.image import Image
from quangtps.core.config import Config
from quangtps.core.exceptions import DatabaseError

logger = get_logger(__name__)

class ImageSeries:
    """
    Lớp biểu diễn thông tin về một chuỗi hình ảnh.
    """
    
    def __init__(self, series_id=None, study_id=None, patient_id=None, 
                 modality=None, description=None, metadata=None):
        """
        Khởi tạo đối tượng ImageSeries.
        
        Args:
            series_id (str): ID của chuỗi hình ảnh
            study_id (str): ID của nghiên cứu liên kết
            patient_id (str): ID của bệnh nhân liên kết
            modality (str): Loại hình ảnh (CT, MR, PT, etc.)
            description (str): Mô tả về chuỗi hình ảnh
            metadata (dict): Metadata bổ sung về chuỗi hình ảnh
        """
        self.id = series_id or str(uuid.uuid4())
        self.study_id = study_id
        self.patient_id = patient_id
        self.modality = modality
        self.description = description or ""
        self.metadata = metadata or {}
        self.image_path = None
        self.created_at = datetime.now().isoformat()
        self.created_date = datetime.now()
        self.updated_at = self.created_at
        
    def to_dict(self):
        """
        Chuyển đổi đối tượng thành dictionary.
        
        Returns:
            dict: Thông tin chuỗi hình ảnh dưới dạng dictionary
        """
        return {
            "id": self.id,
            "study_id": self.study_id,
            "patient_id": self.patient_id,
            "modality": self.modality,
            "description": self.description,
            "metadata": self.metadata,
            "image_path": self.image_path,
            "created_at": self.created_at,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "updated_at": self.updated_at
        }


class ImageDatabase:
    """
    Lớp quản lý lưu trữ và truy xuất hình ảnh y tế.
    """
    
    def __init__(self):
        """Khởi tạo đối tượng ImageDatabase."""
        self.db = DBConnector()
        self.config = Config.get_instance()
        self.images_dir = os.path.join(self.config.data_dir, 'images')
        
        # Đảm bảo thư mục lưu hình ảnh tồn tại
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            logger.info("Đã tạo thư mục lưu trữ hình ảnh: %s", self.images_dir)

    def save_image(self, image: Image, series_id: str, patient_id: str, study_id: str, 
                  description: str = None, metadata: Dict = None) -> str:
        """
        Lưu một hình ảnh vào cơ sở dữ liệu.
        
        Args:
            image: Đối tượng Image cần lưu
            series_id: ID của chuỗi hình ảnh
            patient_id: ID của bệnh nhân
            study_id: ID của nghiên cứu
            description: Mô tả về hình ảnh
            metadata: Thông tin bổ sung về hình ảnh
            
        Returns:
            str: ID của chuỗi hình ảnh đã lưu
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi lưu hình ảnh
        """
        try:
            # Tạo thư mục cho bệnh nhân và chuỗi hình ảnh
            patient_dir = os.path.join(self.images_dir, patient_id)
            if not os.path.exists(patient_dir):
                os.makedirs(patient_dir)
                
            # Tạo ID cho chuỗi hình ảnh nếu chưa có
            if not series_id:
                series_id = str(uuid.uuid4())
                
            # Tạo đường dẫn lưu file HDF5
            series_path = os.path.join(patient_dir, f"{series_id}.h5")
            
            # Lưu dữ liệu hình ảnh vào file HDF5
            with h5py.File(series_path, 'w') as f:
                # Lưu dữ liệu hình ảnh
                f.create_dataset('data', data=image.data, compression='gzip', compression_opts=9)
                
                # Lưu metadata
                metadata_all = image.metadata.copy()
                if metadata:
                    metadata_all.update(metadata)
                f.create_dataset('metadata', data=json.dumps(metadata_all))
                
                # Lưu thông tin về kích thước, vị trí và hướng
                f.create_dataset('shape', data=np.array(image.shape))
                f.create_dataset('spacing', data=np.array(image.pixel_spacing))
                f.create_dataset('origin', data=np.array(image.origin))
                f.create_dataset('direction', data=np.array(image.direction))
            
            # Chuẩn bị metadata để lưu vào cơ sở dữ liệu
            series_metadata = metadata_all.copy() if metadata_all else {}
            series_metadata.update({
                'shape': image.shape,
                'spacing': image.pixel_spacing,
                'origin': image.origin,
                'direction': list(image.direction)
            })
            
            # Chuyển đổi metadata thành JSON
            series_metadata_json = json.dumps(series_metadata)
            
            # Chuẩn bị thời gian
            current_time = datetime.now().isoformat()
            
            # Kiểm tra xem chuỗi hình ảnh đã tồn tại chưa
            query = "SELECT id FROM series WHERE id = ?"
            exists = self.db.execute_query(query, (series_id,))
            
            if exists:
                # Cập nhật chuỗi hình ảnh hiện có
                query = """
                    UPDATE series 
                    SET description = ?, metadata = ?, file_path = ?, updated_at = ? 
                    WHERE id = ?
                """
                self.db.execute_query(query, (
                    description or "", 
                    series_metadata_json, 
                    series_path, 
                    current_time, 
                    series_id
                ))
                logger.info("Đã cập nhật chuỗi hình ảnh có ID: %s", series_id)
            else:
                # Thêm mới chuỗi hình ảnh
                query = """
                    INSERT INTO series 
                    (id, study_id, patient_id, modality, description, series_date, 
                     series_time, metadata, file_path, created_at, updated_at) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                # Trích xuất ngày và giờ từ metadata nếu có
                series_date = metadata_all.get('SeriesDate', '')
                series_time = metadata_all.get('SeriesTime', '')
                modality = metadata_all.get('Modality', 'Unknown')
                
                self.db.execute_query(query, (
                    series_id, 
                    study_id, 
                    patient_id, 
                    modality, 
                    description or "", 
                    series_date, 
                    series_time, 
                    series_metadata_json, 
                    series_path, 
                    current_time, 
                    current_time
                ))
                logger.info("Đã thêm chuỗi hình ảnh mới với ID: %s", series_id)
            
            return series_id
            
        except Exception as e:
            logger.error("Lỗi khi lưu hình ảnh: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi lưu hình ảnh: %s" % str(e)) from e
    
    def load_image(self, series_id: str) -> Optional[Image]:
        """
        Nạp một hình ảnh từ cơ sở dữ liệu theo ID chuỗi.
        
        Args:
            series_id: ID của chuỗi hình ảnh cần nạp
            
        Returns:
            Image: Đối tượng Image hoặc None nếu không tìm thấy
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi nạp hình ảnh
        """
        try:
            # Truy vấn thông tin chuỗi hình ảnh
            query = "SELECT file_path, metadata FROM series WHERE id = ?"
            result = self.db.execute_query(query, (series_id,))
            
            if not result:
                logger.warning("Không tìm thấy chuỗi hình ảnh có ID: %s", series_id)
                return None
            
            file_path = result['file_path']
            
            # Kiểm tra file tồn tại
            if not os.path.exists(file_path):
                logger.error("File hình ảnh không tồn tại: %s", file_path)
                return None
            
            # Nạp dữ liệu từ file HDF5
            with h5py.File(file_path, 'r') as f:
                # Nạp dữ liệu hình ảnh
                data = f['data'][()]
                
                # Nạp metadata
                metadata_json = f['metadata'][()]
                if isinstance(metadata_json, bytes):
                    metadata_json = metadata_json.decode('utf-8')
                metadata = json.loads(metadata_json)
            
            # Tạo đối tượng Image
            image = Image(data, metadata)
            
            logger.info("Đã nạp hình ảnh có ID: %s", series_id)
            return image
            
        except Exception as e:
            logger.error("Lỗi khi nạp hình ảnh: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi nạp hình ảnh: %s" % str(e)) from e
    
    def delete_image(self, series_id: str) -> bool:
        """
        Xóa một hình ảnh khỏi cơ sở dữ liệu.
        
        Args:
            series_id: ID của chuỗi hình ảnh cần xóa
            
        Returns:
            bool: True nếu xóa thành công, False nếu không tìm thấy
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi xóa hình ảnh
        """
        try:
            # Lấy đường dẫn file hình ảnh
            query = "SELECT file_path FROM series WHERE id = ?"
            result = self.db.execute_query(query, (series_id,))
            
            if not result:
                logger.warning("Không tìm thấy chuỗi hình ảnh có ID: %s", series_id)
                return False
            
            file_path = result['file_path']
            
            # Xóa file hình ảnh nếu tồn tại
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info("Đã xóa file hình ảnh: %s", file_path)
            
            # Xóa bản ghi trong cơ sở dữ liệu
            query = "DELETE FROM series WHERE id = ?"
            self.db.execute_query(query, (series_id,))
            
            logger.info("Đã xóa chuỗi hình ảnh có ID: %s", series_id)
            return True
            
        except Exception as e:
            logger.error("Lỗi khi xóa hình ảnh: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi xóa hình ảnh: %s" % str(e)) from e
    
    def get_patient_images(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả hình ảnh của một bệnh nhân.
        
        Args:
            patient_id: ID của bệnh nhân
            
        Returns:
            List[Dict]: Danh sách các chuỗi hình ảnh dưới dạng dictionary
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi truy vấn
        """
        try:
            query = """
                SELECT s.id, s.study_id, s.patient_id, s.modality, s.description, 
                       s.series_date, s.series_time, s.metadata, s.file_path, 
                       s.created_at, s.updated_at
                FROM series s
                WHERE s.patient_id = ?
                ORDER BY s.series_date DESC, s.series_time DESC
            """
            results = self.db.execute_query(query, (patient_id,), fetchall=True)
            
            if not results:
                logger.info("Không tìm thấy hình ảnh nào cho bệnh nhân có ID: %s", patient_id)
                return []
            
            series_list = []
            for row in results:
                series_dict = dict(row)
                
                # Chuyển đổi metadata từ JSON
                if 'metadata' in series_dict and series_dict['metadata']:
                    try:
                        series_dict['metadata'] = json.loads(series_dict['metadata'])
                    except json.JSONDecodeError:
                        series_dict['metadata'] = {}
                
                series_list.append(series_dict)
            
            logger.info("Đã lấy %d chuỗi hình ảnh cho bệnh nhân có ID: %s", 
                      len(series_list), patient_id)
            return series_list
            
        except Exception as e:
            logger.error("Lỗi khi lấy danh sách hình ảnh: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi lấy danh sách hình ảnh: %s" % str(e)) from e
