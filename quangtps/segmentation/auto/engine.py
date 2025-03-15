#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module engine phân đoạn tự động dựa trên deep learning.

Module này cung cấp chức năng phân đoạn tự động các cấu trúc giải phẫu
từ hình ảnh CT/MRI sử dụng các mô hình deep learning.
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any, Union
import SimpleITK as sitk
import cv2
import tempfile

from quangtps.segmentation.auto.model import UNetModel, AttentionUNet
from quangtps.segmentation.auto.model_repository import ModelRepository

logger = logging.getLogger(__name__)


class AutoSegmentationEngine:
    """
    Engine phân đoạn tự động dựa trên deep learning.
    
    Engine này quản lý việc tải mô hình, tiền xử lý dữ liệu, phân đoạn,
    và hậu xử lý kết quả phân đoạn.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Khởi tạo engine phân đoạn tự động.
        
        Parameters
        ----------
        models_dir : str, optional
            Thư mục chứa các mô hình phân đoạn. Nếu None, sẽ sử dụng
            thư mục mặc định trong cài đặt ứng dụng.
        """
        # Khởi tạo kho mô hình
        self.repository = ModelRepository(models_dir)
        
        # Cache các mô hình đã tải
        self.loaded_models = {}
        
        # Cấu hình mặc định
        self.config = {
            'threshold': 0.5,         # Ngưỡng phân đoạn
            'min_area': 100,          # Diện tích tối thiểu của vùng (pixel)
            'smooth': True,           # Làm mịn contour
            'post_process': True,     # Hậu xử lý (loại bỏ vùng nhỏ, làm mịn)
            'target_size': (256, 256) # Kích thước hình ảnh đầu vào cho mô hình
        }
    
    def get_available_models(self) -> List[Dict]:
        """
        Lấy danh sách các mô hình có sẵn.
        
        Returns
        -------
        List[Dict]
            Danh sách mô hình có sẵn, mỗi mô hình là một dictionary
            với các thông tin như name, description, version, v.v.
        """
        return self.repository.get_available_models()
    
    def update_model_list(self) -> bool:
        """
        Cập nhật danh sách mô hình từ kho lưu trữ.
        
        Returns
        -------
        bool
            True nếu cập nhật thành công, False nếu có lỗi
        """
        return self.repository.update_model_list()
    
    def _load_model(self, structure: str) -> Optional[tf.keras.Model]:
        """
        Tải mô hình cho một cấu trúc cụ thể.
        
        Parameters
        ----------
        structure : str
            Tên cấu trúc (cũng là tên mô hình)
            
        Returns
        -------
        Optional[tf.keras.Model]
            Mô hình đã tải, hoặc None nếu không tải được
        """
        # Kiểm tra cache
        if structure in self.loaded_models:
            return self.loaded_models[structure]
        
        try:
            # Lấy đường dẫn đến thư mục mô hình
            model_path = self.repository.get_model_path(structure)
            if model_path is None:
                logger.error(f"Model for structure {structure} not found")
                return None
            
            # Đọc file cấu hình mô hình
            config_path = os.path.join(model_path, 'config.json')
            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found for model {structure}")
                return None
            
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            # Xác định loại mô hình
            model_type = model_config.get('model_type', 'unet')
            
            # Tạo mô hình phù hợp
            if model_type.lower() == 'attention_unet':
                model = AttentionUNet(
                    input_shape=tuple(model_config.get('input_shape', [256, 256, 1])),
                    n_filters=model_config.get('n_filters', 64),
                    n_classes=model_config.get('n_classes', 1)
                )
            else:  # Mặc định là UNet
                model = UNetModel(
                    input_shape=tuple(model_config.get('input_shape', [256, 256, 1])),
                    n_filters=model_config.get('n_filters', 64),
                    n_classes=model_config.get('n_classes', 1)
                )
            
            # Tải trọng số từ file
            weights_path = os.path.join(model_path, 'weights.h5')
            if not os.path.exists(weights_path):
                logger.error(f"Weights file not found for model {structure}")
                return None
            
            model.load_weights(weights_path)
            
            # Lưu vào cache
            self.loaded_models[structure] = model
            
            logger.info(f"Loaded model for structure {structure}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model for structure {structure}: {str(e)}")
            return None
    
    def _preprocess_image(self, 
                         image: np.ndarray, 
                         spacing: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Tiền xử lý hình ảnh đầu vào.
        
        Parameters
        ----------
        image : np.ndarray
            Hình ảnh đầu vào (2D)
        spacing : Tuple[float, float], optional
            Khoảng cách pixel (mm), nếu có
            
        Returns
        -------
        np.ndarray
            Hình ảnh đã tiền xử lý
        """
        # Chuyển về 8-bit nếu cần
        if image.dtype != np.uint8:
            if image.max() > 0:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Lưu kích thước gốc
        original_size = image.shape
        
        # Thay đổi kích thước để phù hợp với đầu vào của mô hình
        target_size = self.config['target_size']
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Chuẩn hóa pixel values về khoảng [0, 1]
        processed_image = resized_image.astype(np.float32) / 255.0
        
        # Thêm chiều batch và channel
        processed_image = np.expand_dims(np.expand_dims(processed_image, axis=0), axis=-1)
        
        return processed_image
    
    def _postprocess_mask(self, 
                         mask: np.ndarray, 
                         original_size: Tuple[int, int],
                         threshold: float = None,
                         post_process: bool = None,
                         smooth: bool = None) -> np.ndarray:
        """
        Hậu xử lý mask sau khi phân đoạn.
        
        Parameters
        ----------
        mask : np.ndarray
            Mask từ mô hình (khoảng [0, 1])
        original_size : Tuple[int, int]
            Kích thước gốc của hình ảnh đầu vào
        threshold : float, optional
            Ngưỡng phân đoạn, sử dụng giá trị mặc định nếu None
        post_process : bool, optional
            Có thực hiện hậu xử lý không, sử dụng giá trị mặc định nếu None
        smooth : bool, optional
            Có làm mịn contour không, sử dụng giá trị mặc định nếu None
            
        Returns
        -------
        np.ndarray
            Mask nhị phân sau khi hậu xử lý
        """
        # Sử dụng các giá trị mặc định nếu không có tham số
        if threshold is None:
            threshold = self.config['threshold']
        
        if post_process is None:
            post_process = self.config['post_process']
        
        if smooth is None:
            smooth = self.config['smooth']
        
        # Loại bỏ chiều batch và channel
        mask = mask[0, :, :, 0]
        
        # Thay đổi kích thước về kích thước gốc
        mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Áp dụng ngưỡng
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        # Nếu không cần hậu xử lý, trả về mask nhị phân
        if not post_process:
            return binary_mask
        
        # Hậu xử lý: loại bỏ các vùng nhỏ
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Tạo mask trống
        filtered_mask = np.zeros_like(binary_mask)
        
        # Vẽ lại chỉ các contour đủ lớn
        min_area = self.config['min_area']
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                if smooth:
                    # Làm mịn contour
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    cv2.drawContours(filtered_mask, [approx], 0, 255, -1)
                else:
                    cv2.drawContours(filtered_mask, [contour], 0, 255, -1)
        
        return filtered_mask
    
    def _extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Trích xuất contours từ mask nhị phân.
        
        Parameters
        ----------
        mask : np.ndarray
            Mask nhị phân (0 hoặc 255)
            
        Returns
        -------
        List[np.ndarray]
            Danh sách các contour
        """
        # Tìm các contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Chuyển đổi định dạng contour
        result_contours = []
        for contour in contours:
            # Làm phẳng mảng contour
            points = contour.reshape(-1, 2)
            result_contours.append(points)
        
        return result_contours
    
    def segment_slice(self, 
                     image: np.ndarray, 
                     structure: str,
                     spacing: Optional[Tuple[float, float]] = None,
                     params: Dict = None) -> Dict:
        """
        Phân đoạn một lát cắt hình ảnh.
        
        Parameters
        ----------
        image : np.ndarray
            Hình ảnh đầu vào (2D)
        structure : str
            Tên cấu trúc cần phân đoạn
        spacing : Tuple[float, float], optional
            Khoảng cách pixel (mm), nếu có
        params : Dict, optional
            Các tham số bổ sung cho việc phân đoạn
            
        Returns
        -------
        Dict
            Kết quả phân đoạn, bao gồm:
            - success: True/False
            - error: Thông báo lỗi (nếu có)
            - contours: Danh sách các contour
            - mask: Mask nhị phân
            - structure: Tên cấu trúc
        """
        # Khởi tạo kết quả mặc định
        result = {
            'success': False,
            'structure': structure
        }
        
        try:
            # Tải mô hình
            model = self._load_model(structure)
            if model is None:
                result['error'] = f"Could not load model for structure {structure}"
                return result
            
            # Tiền xử lý hình ảnh
            processed_image = self._preprocess_image(image, spacing)
            
            # Hợp nhất tham số
            p = self.config.copy()
            if params:
                p.update(params)
            
            # Dự đoán
            mask_pred = model.predict(processed_image)
            
            # Hậu xử lý mask
            binary_mask = self._postprocess_mask(
                mask_pred, 
                image.shape, 
                threshold=p.get('threshold'),
                post_process=p.get('post_process'),
                smooth=p.get('smooth')
            )
            
            # Trích xuất contours
            contours = self._extract_contours(binary_mask)
            
            # Tạo kết quả
            result['success'] = True
            result['contours'] = contours
            result['mask'] = binary_mask
            
            return result
            
        except Exception as e:
            logger.error(f"Error segmenting slice for structure {structure}: {str(e)}")
            result['error'] = str(e)
            return result
    
    def segment_volume(self, 
                      volume: np.ndarray, 
                      structure: str,
                      spacing: Optional[Tuple[float, float, float]] = None,
                      params: Dict = None) -> Dict:
        """
        Phân đoạn toàn bộ khối 3D.
        
        Parameters
        ----------
        volume : np.ndarray
            Khối 3D đầu vào (có thể là khối CT hoặc MRI)
        structure : str
            Tên cấu trúc cần phân đoạn
        spacing : Tuple[float, float, float], optional
            Khoảng cách voxel (mm), nếu có
        params : Dict, optional
            Các tham số bổ sung cho việc phân đoạn
            
        Returns
        -------
        Dict
            Kết quả phân đoạn, bao gồm:
            - success: True/False
            - error: Thông báo lỗi (nếu có)
            - contours_3d: Danh sách các contour theo từng lát cắt
            - mask_3d: Mask 3D nhị phân
            - structure: Tên cấu trúc
        """
        # Khởi tạo kết quả mặc định
        result = {
            'success': False,
            'structure': structure
        }
        
        try:
            # Tải mô hình
            model = self._load_model(structure)
            if model is None:
                result['error'] = f"Could not load model for structure {structure}"
                return result
            
            # Hợp nhất tham số
            p = self.config.copy()
            if params:
                p.update(params)
            
            # Tạo mask 3D trống
            mask_3d = np.zeros_like(volume, dtype=np.uint8)
            
            # Tạo danh sách contours 3D (danh sách của danh sách contours)
            contours_3d = []
            
            # Phân đoạn từng lát cắt
            for z in range(volume.shape[0]):
                # Lấy lát cắt
                slice_image = volume[z, :, :]
                
                # Tiền xử lý lát cắt
                processed_image = self._preprocess_image(slice_image)
                
                # Dự đoán
                mask_pred = model.predict(processed_image)
                
                # Hậu xử lý mask
                binary_mask = self._postprocess_mask(
                    mask_pred, 
                    slice_image.shape, 
                    threshold=p.get('threshold'),
                    post_process=p.get('post_process'),
                    smooth=p.get('smooth')
                )
                
                # Lưu mask vào mask 3D
                mask_3d[z, :, :] = binary_mask
                
                # Trích xuất contours
                contours = self._extract_contours(binary_mask)
                
                # Thêm vào danh sách contours 3D
                contours_3d.append(contours)
            
            # Tạo kết quả
            result['success'] = True
            result['contours_3d'] = contours_3d
            result['mask_3d'] = mask_3d
            
            return result
            
        except Exception as e:
            logger.error(f"Error segmenting volume for structure {structure}: {str(e)}")
            result['error'] = str(e)
            return result
    
    def segment_from_dicom(self, 
                          dicom_folder: str, 
                          structure: str,
                          output_folder: Optional[str] = None,
                          params: Dict = None) -> Dict:
        """
        Phân đoạn từ dữ liệu DICOM.
        
        Parameters
        ----------
        dicom_folder : str
            Thư mục chứa file DICOM
        structure : str
            Tên cấu trúc cần phân đoạn
        output_folder : str, optional
            Thư mục đầu ra cho kết quả phân đoạn
        params : Dict, optional
            Các tham số bổ sung cho việc phân đoạn
            
        Returns
        -------
        Dict
            Kết quả phân đoạn, bao gồm:
            - success: True/False
            - error: Thông báo lỗi (nếu có)
            - output_path: Đường dẫn đến thư mục kết quả (nếu có)
            - structure: Tên cấu trúc
        """
        # Khởi tạo kết quả mặc định
        result = {
            'success': False,
            'structure': structure
        }
        
        try:
            # Đọc dữ liệu DICOM
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            
            # Lấy thông tin spacing
            spacing = image.GetSpacing()
            
            # Chuyển đổi sang numpy array
            volume = sitk.GetArrayFromImage(image)
            
            # Gọi hàm phân đoạn khối
            seg_result = self.segment_volume(
                volume, 
                structure, 
                spacing=spacing, 
                params=params
            )
            
            if not seg_result['success']:
                return seg_result
            
            # Nếu không cung cấp thư mục đầu ra, tạo thư mục tạm
            if output_folder is None:
                output_folder = tempfile.mkdtemp(prefix=f"quangtps_{structure}_")
            else:
                os.makedirs(output_folder, exist_ok=True)
            
            # Tạo mask 3D từ SimpleITK dựa trên kết quả phân đoạn
            mask_3d = seg_result['mask_3d']
            mask_sitk = sitk.GetImageFromArray(mask_3d)
            mask_sitk.CopyInformation(image)
            
            # Lưu mask dưới dạng file NRRD
            output_nrrd = os.path.join(output_folder, f"{structure}_mask.nrrd")
            sitk.WriteImage(mask_sitk, output_nrrd)
            
            # Lưu contours dưới dạng file JSON
            output_json = os.path.join(output_folder, f"{structure}_contours.json")
            with open(output_json, 'w') as f:
                # Chuyển đổi contours thành dạng có thể JSON serialize được
                contours_json = []
                for z, contours_slice in enumerate(seg_result['contours_3d']):
                    contours_slice_json = []
                    for contour in contours_slice:
                        contours_slice_json.append(contour.tolist())
                    contours_json.append({'slice': z, 'contours': contours_slice_json})
                
                json.dump(contours_json, f)
            
            # Tạo kết quả
            result['success'] = True
            result['output_path'] = output_folder
            result['output_nrrd'] = output_nrrd
            result['output_json'] = output_json
            
            return result
            
        except Exception as e:
            logger.error(f"Error segmenting from DICOM for structure {structure}: {str(e)}")
            result['error'] = str(e)
            return result
