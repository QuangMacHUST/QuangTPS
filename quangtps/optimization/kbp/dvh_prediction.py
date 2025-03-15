"""
Module dự đoán DVH (Dose-Volume Histogram) dựa trên tri thức trong QuangTPS.

Module này cung cấp các lớp và hàm cho việc dự đoán đường cong DVH dựa trên 
thông tin giải phẫu của bệnh nhân và cơ sở tri thức từ các kế hoạch trước đó.
Điều này cho phép tạo ra các mục tiêu và ràng buộc hợp lý dựa trên kinh nghiệm lâm sàng.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import os
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from quangtps.evaluation.dvh import calculate_dvh, calculate_dvh_metrics
from quangtps.dose.dose_grid import DoseGrid
from quangtps.structures.structure import Structure
from quangtps.core.constants import EPSILON

logger = logging.getLogger(__name__)

class DVHPredictionModel:
    """
    Mô hình dự đoán DVH dựa trên đặc điểm giải phẫu của bệnh nhân.
    
    Lớp này sử dụng các mô hình học máy để dự đoán đường cong DVH
    cho các cơ quan dựa trên vị trí, kích thước, hình dạng và các đặc điểm
    giải phẫu khác.
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Khởi tạo mô hình dự đoán DVH.
        
        Args:
            model_type: Loại mô hình ("random_forest", "gradient_boosting", "knn")
        """
        self.model_type = model_type
        self.models = {}  # Dictionary chứa các mô hình cho từng cơ quan
        self.scalers = {}  # Dictionary chứa các bộ chuẩn hóa dữ liệu
        self.dose_bins = None  # Bins dùng cho DVH
        self.feature_names = []  # Tên các đặc trưng
        self.trained = False
        
    def _create_model(self):
        """Tạo mô hình học máy dựa trên loại đã chọn."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, max_depth=15, 
                                         random_state=42, n_jobs=-1)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                             learning_rate=0.1, random_state=42)
        elif self.model_type == "knn":
            return KNeighborsRegressor(n_neighbors=5, weights='distance')
        else:
            raise ValueError(f"Loại mô hình không được hỗ trợ: {self.model_type}")
            
    def _extract_features(self, structure: Structure, target_structure: Structure, 
                         structures_dict: Dict[str, Structure]) -> np.ndarray:
        """
        Trích xuất đặc trưng từ cấu trúc để dùng làm đầu vào cho mô hình.
        
        Args:
            structure: Cấu trúc cần dự đoán DVH
            target_structure: Cấu trúc đích (PTV)
            structures_dict: Từ điển chứa tất cả các cấu trúc
            
        Returns:
            np.ndarray: Vector đặc trưng
        """
        features = []
        
        # Đặc trưng về thể tích
        features.append(structure.get_volume())
        
        # Đặc trưng về vị trí tương đối so với PTV
        com_structure = structure.get_center_of_mass()
        com_target = target_structure.get_center_of_mass()
        distance = np.linalg.norm(np.array(com_structure) - np.array(com_target))
        features.append(distance)
        
        # Đặc trưng về độ chồng lấp với PTV
        overlap = structure.get_overlap_volume(target_structure)
        features.append(overlap)
        features.append(overlap / structure.get_volume())  # Overlap ratio
        
        # Đặc trưng hình dạng
        features.append(structure.get_surface_area())
        features.append(structure.get_surface_area() / (structure.get_volume() ** (2/3)))  # Sphericity
        
        # Khoảng cách tối thiểu đến PTV
        min_distance = structure.get_minimum_distance(target_structure)
        features.append(min_distance)
        
        # Thể tích PTV
        features.append(target_structure.get_volume())
        
        # Thể tích chồng lấp với các cơ quan khác
        total_overlap_with_others = 0
        for name, other_structure in structures_dict.items():
            if name != structure.name and name != target_structure.name:
                total_overlap_with_others += structure.get_overlap_volume(other_structure)
        features.append(total_overlap_with_others)
        
        return np.array(features)
    
    def train(self, training_data: List[Dict[str, Any]]):
        """
        Huấn luyện mô hình dự đoán DVH.
        
        Args:
            training_data: Danh sách các dictionary chứa thông tin huấn luyện
                           Mỗi dictionary có các khóa: 
                           'structure_features', 'target_structure', 'dvh', 'structure_name'
        """
        # Nhóm dữ liệu theo cơ quan
        organ_data = {}
        for item in training_data:
            structure_name = item['structure_name']
            if structure_name not in organ_data:
                organ_data[structure_name] = []
            organ_data[structure_name].append(item)
        
        # Tạo và huấn luyện mô hình cho từng cơ quan
        for organ_name, data in organ_data.items():
            X = np.array([item['structure_features'] for item in data])
            y = np.array([item['dvh'] for item in data])
            
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Lưu scaler
            self.scalers[organ_name] = scaler
            
            # Tạo và huấn luyện mô hình
            model = self._create_model()
            model.fit(X_scaled, y)
            
            # Lưu mô hình
            self.models[organ_name] = model
            
            logger.info(f"Đã huấn luyện mô hình cho cơ quan: {organ_name}")
        
        self.dose_bins = training_data[0]['dose_bins']
        self.feature_names = training_data[0].get('feature_names', [])
        self.trained = True
        
        logger.info(f"Đã hoàn thành huấn luyện mô hình dự đoán DVH cho {len(self.models)} cơ quan")
    
    def predict(self, structure: Structure, target_structure: Structure, 
               structures_dict: Dict[str, Structure]) -> Dict[str, Any]:
        """
        Dự đoán DVH cho một cấu trúc cụ thể.
        
        Args:
            structure: Cấu trúc cần dự đoán DVH
            target_structure: Cấu trúc đích (PTV)
            structures_dict: Từ điển chứa tất cả các cấu trúc
            
        Returns:
            Dict[str, Any]: Kết quả dự đoán DVH và các thông tin liên quan
        """
        if not self.trained:
            raise RuntimeError("Mô hình chưa được huấn luyện")
            
        # Trích xuất đặc trưng
        features = self._extract_features(structure, target_structure, structures_dict)
        
        # Kiểm tra xem có mô hình cho cơ quan này không
        organ_name = structure.name
        if organ_name not in self.models:
            # Sử dụng generic model hoặc mô hình tương tự nhất
            logger.warning(f"Không có mô hình cho cơ quan {organ_name}, đang sử dụng mô hình gần nhất")
            similar_organs = self._find_similar_organs(organ_name)
            if not similar_organs:
                raise ValueError(f"Không thể dự đoán DVH cho cơ quan {organ_name}")
            organ_name = similar_organs[0]
        
        # Chuẩn hóa đặc trưng
        features_scaled = self.scalers[organ_name].transform([features])[0]
        
        # Dự đoán DVH
        predicted_dvh = self.models[organ_name].predict([features_scaled])[0]
        
        return {
            'dvh_data': predicted_dvh,
            'dose_bins': self.dose_bins,
            'structure_name': organ_name,
            'confidence': self._calculate_confidence(organ_name, features)
        }
    
    def _find_similar_organs(self, organ_name: str) -> List[str]:
        """
        Tìm các cơ quan tương tự dựa trên tên.
        
        Args:
            organ_name: Tên cơ quan cần tìm tương tự
            
        Returns:
            List[str]: Danh sách tên các cơ quan tương tự
        """
        # Đây là một phương pháp đơn giản, có thể cải thiện bằng NLP
        similar_organs = []
        
        # Tách tên thành các phần
        parts = organ_name.lower().replace('_', ' ').split()
        
        # Tìm các cơ quan có chứa ít nhất một phần
        for model_name in self.models.keys():
            model_parts = model_name.lower().replace('_', ' ').split()
            if any(part in model_parts for part in parts):
                similar_organs.append(model_name)
                
        return similar_organs
    
    def _calculate_confidence(self, organ_name: str, features: np.ndarray) -> float:
        """
        Tính độ tin cậy của dự đoán.
        
        Args:
            organ_name: Tên cơ quan
            features: Đặc trưng của cơ quan
            
        Returns:
            float: Độ tin cậy từ 0 đến 1
        """
        # Đối với mô hình Random Forest, có thể sử dụng độ phân tán của dự đoán
        if self.model_type == "random_forest":
            # Chuẩn hóa đặc trưng
            features_scaled = self.scalers[organ_name].transform([features])[0]
            
            # Lấy dự đoán từ tất cả các cây
            trees = self.models[organ_name].estimators_
            predictions = [tree.predict([features_scaled])[0] for tree in trees]
            
            # Tính độ phân tán
            std = np.mean(np.std(predictions, axis=0))
            
            # Chuyển đổi thành độ tin cậy
            confidence = max(0, 1 - min(std / 5.0, 1.0))  # Giả sử std tối đa là 5
            return confidence
        
        # Đối với KNN, sử dụng khoảng cách đến các điểm gần nhất
        elif self.model_type == "knn":
            features_scaled = self.scalers[organ_name].transform([features])[0]
            distances, _ = self.models[organ_name].kneighbors([features_scaled])
            avg_distance = np.mean(distances[0])
            
            # Chuyển đổi thành độ tin cậy
            confidence = max(0, 1 - min(avg_distance / 3.0, 1.0))  # Giả sử khoảng cách tối đa là 3
            return confidence
        
        # Mặc định trả về 0.7
        return 0.7
    
    def save_model(self, file_path: str):
        """
        Lưu mô hình vào file.
        
        Args:
            file_path: Đường dẫn đến file lưu
        """
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'dose_bins': self.dose_bins,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'trained': self.trained
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Đã lưu mô hình vào file: {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str) -> 'DVHPredictionModel':
        """
        Tải mô hình từ file.
        
        Args:
            file_path: Đường dẫn đến file chứa mô hình
            
        Returns:
            DVHPredictionModel: Mô hình đã tải
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            
        model = cls(model_type=model_data['model_type'])
        model.models = model_data['models']
        model.scalers = model_data['scalers']
        model.dose_bins = model_data['dose_bins']
        model.feature_names = model_data['feature_names']
        model.trained = model_data['trained']
        
        logger.info(f"Đã tải mô hình từ file: {file_path}")
        return model


class DVHFeatureExtractor:
    """
    Lớp trích xuất đặc trưng cho dự đoán DVH.
    
    Lớp này trích xuất các đặc trưng giải phẫu và hình học
    từ cấu trúc bệnh nhân để sử dụng trong mô hình dự đoán DVH.
    """
    
    @staticmethod
    def extract_features_for_structure(structure: Structure, target_structure: Structure, 
                                     structures_dict: Dict[str, Structure]) -> Dict[str, Any]:
        """
        Trích xuất đặc trưng cho một cấu trúc cụ thể.
        
        Args:
            structure: Cấu trúc cần trích xuất đặc trưng
            target_structure: Cấu trúc đích (PTV)
            structures_dict: Từ điển chứa tất cả các cấu trúc
            
        Returns:
            Dict[str, Any]: Dictionary chứa các đặc trưng đã trích xuất
        """
        features = {}
        
        # Đặc trưng cơ bản
        features['volume'] = structure.get_volume()
        features['surface_area'] = structure.get_surface_area()
        features['sphericity'] = features['surface_area'] / (features['volume'] ** (2/3))
        
        # Vị trí và khoảng cách
        com_structure = structure.get_center_of_mass()
        com_target = target_structure.get_center_of_mass()
        features['distance_to_target_com'] = np.linalg.norm(np.array(com_structure) - np.array(com_target))
        features['min_distance_to_target'] = structure.get_minimum_distance(target_structure)
        
        # Thông tin chồng lấp với PTV
        features['overlap_volume'] = structure.get_overlap_volume(target_structure)
        features['overlap_ratio'] = features['overlap_volume'] / features['volume'] if features['volume'] > 0 else 0
        
        # Thông tin về PTV
        features['target_volume'] = target_structure.get_volume()
        features['target_surface_area'] = target_structure.get_surface_area()
        
        # Đặc trưng hình dạng chi tiết
        features.update(DVHFeatureExtractor._extract_shape_features(structure))
        
        # Các đặc trưng topological
        features.update(DVHFeatureExtractor._extract_topological_features(structure, structures_dict))
        
        return features
    
    @staticmethod
    def _extract_shape_features(structure: Structure) -> Dict[str, float]:
        """
        Trích xuất các đặc trưng về hình dạng chi tiết.
        
        Args:
            structure: Cấu trúc cần trích xuất đặc trưng
            
        Returns:
            Dict[str, float]: Dictionary chứa các đặc trưng hình dạng
        """
        features = {}
        
        # Kích thước theo các trục
        bbox = structure.get_bounding_box()
        features['size_x'] = bbox[1] - bbox[0]
        features['size_y'] = bbox[3] - bbox[2]
        features['size_z'] = bbox[5] - bbox[4]
        features['volume_bbox_ratio'] = structure.get_volume() / (features['size_x'] * features['size_y'] * features['size_z'])
        
        # Tính moment of inertia để hiểu hình dạng
        principal_moments = structure.get_principal_moments_of_inertia()
        if principal_moments is not None and len(principal_moments) == 3:
            features['elongation'] = principal_moments[0] / principal_moments[2]
            features['flatness'] = principal_moments[1] / principal_moments[2]
        else:
            features['elongation'] = 1.0
            features['flatness'] = 1.0
        
        return features
    
    @staticmethod
    def _extract_topological_features(structure: Structure, 
                                   structures_dict: Dict[str, Structure]) -> Dict[str, float]:
        """
        Trích xuất các đặc trưng về mối quan hệ topological với các cấu trúc khác.
        
        Args:
            structure: Cấu trúc cần trích xuất đặc trưng
            structures_dict: Từ điển chứa tất cả các cấu trúc
            
        Returns:
            Dict[str, float]: Dictionary chứa các đặc trưng topological
        """
        features = {}
        
        # Tính số lượng cấu trúc chồng lấp
        overlapping_structures = 0
        total_overlap_volume = 0
        
        for name, other in structures_dict.items():
            if name != structure.name:
                overlap = structure.get_overlap_volume(other)
                if overlap > 0:
                    overlapping_structures += 1
                    total_overlap_volume += overlap
        
        features['num_overlapping_structures'] = overlapping_structures
        features['total_overlap_volume'] = total_overlap_volume
        features['average_overlap_volume'] = total_overlap_volume / max(1, overlapping_structures)
        
        return features


def generate_dvh_constraints_from_prediction(predicted_dvh: Dict[str, Any], structure_name: str, 
                                         prescription_dose: float) -> List[Dict[str, Any]]:
    """
    Tạo các ràng buộc DVH dựa trên kết quả dự đoán.
    
    Args:
        predicted_dvh: Kết quả dự đoán DVH
        structure_name: Tên cấu trúc
        prescription_dose: Liều kê đơn
        
    Returns:
        List[Dict[str, Any]]: Danh sách các ràng buộc DVH được đề xuất
    """
    constraints = []
    dvh_data = predicted_dvh['dvh_data']
    dose_bins = predicted_dvh['dose_bins']
    
    # Xác định loại cấu trúc (PTV hay OAR)
    is_target = 'ptv' in structure_name.lower()
    
    if is_target:
        # Tạo ràng buộc cho PTV (phủ liều)
        d95_idx = np.argmin(np.abs(dvh_data - 95))
        d95_dose = dose_bins[d95_idx]
        
        constraints.append({
            'type': 'DoseVolumeConstraint',
            'structure_name': structure_name,
            'dose': 0.95 * prescription_dose,  # D95% > 95% của liều kê đơn
            'volume_percent': 95,
            'direction': 'lower',
            'priority': 'high'
        })
        
        constraints.append({
            'type': 'MaxDoseConstraint',
            'structure_name': structure_name,
            'dose': 1.1 * prescription_dose,  # Dmax < 110% của liều kê đơn
            'priority': 'high'
        })
        
    else:
        # Tạo ràng buộc cho OAR (giới hạn liều)
        # Tìm các điểm đặc trưng trên đường cong DVH
        d1cc_idx = None
        for i in range(len(dvh_data) - 1, -1, -1):
            if dvh_data[i] <= 1:  # 1cc
                d1cc_idx = i
                break
                
        if d1cc_idx is not None:
            d1cc_dose = dose_bins[d1cc_idx]
            
            constraints.append({
                'type': 'DoseVolumeConstraint',
                'structure_name': structure_name,
                'dose': d1cc_dose,
                'volume_cc': 1,
                'direction': 'upper',
                'priority': 'medium'
            })
        
        # Thêm ràng buộc liều trung bình
        for cutoff in [30, 50, 70]:
            idx = np.argmin(np.abs(dvh_data - cutoff))
            if idx < len(dose_bins):
                constraints.append({
                    'type': 'DoseVolumeConstraint',
                    'structure_name': structure_name,
                    'dose': dose_bins[idx],
                    'volume_percent': cutoff,
                    'direction': 'upper',
                    'priority': 'medium' if cutoff == 50 else 'low'
                })
        
        # Thêm ràng buộc liều tối đa
        constraints.append({
            'type': 'MaxDoseConstraint',
            'structure_name': structure_name,
            'dose': prescription_dose * 0.8,  # Giả định
            'priority': 'medium'
        })
    
    return constraints