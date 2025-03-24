#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module mô hình học máy cho tối ưu hóa dựa trên kiến thức (Knowledge-Based Planning - KBP).

Module này cung cấp các lớp và hàm để huấn luyện, đánh giá và áp dụng các mô hình học máy
cho việc dự đoán các tham số tối ưu và các ràng buộc liều dựa trên kế hoạch đã có trước đó.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import pickle
import joblib
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from quangtps.core.exceptions import ModelError
from quangtps.core.utils import get_patient_features, get_geometry_features

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Loại mô hình học máy."""
    RANDOM_FOREST = auto()
    GRADIENT_BOOSTING = auto()
    NEURAL_NETWORK = auto()
    SVR = auto()

@dataclass
class KBPFeatures:
    """Các đặc trưng được sử dụng trong mô hình KBP."""
    patient_id: str
    structure_set_id: str
    
    # Đặc trưng hình học
    ptv_volume: float
    ptv_surface_area: float
    ptv_compactness: float
    ptv_location: Tuple[float, float, float]  # Tâm của PTV (x,y,z)
    
    # Khoảng cách từ PTV đến các OAR
    distances_to_oars: Dict[str, float]
    
    # Thể tích trùng lặp
    overlap_volumes: Dict[str, float]
    
    # Tỷ lệ trùng lặp
    overlap_ratios: Dict[str, float]
    
    # Đặc trưng lâm sàng
    prescription_dose: float
    fractions: int
    site: str
    
    # Mục tiêu lâm sàng
    clinical_objectives: Dict[str, Any] = field(default_factory=dict)
    
    # Các đặc trưng khác
    additional_features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KBPPredictions:
    """Các dự đoán được tạo bởi mô hình KBP."""
    patient_id: str
    structure_set_id: str
    
    # Dự đoán độ mở rộng
    ptv_margins: Dict[str, float]
    
    # Dự đoán liều tối đa cho OAR
    oar_dose_constraints: Dict[str, float]
    
    # Dự đoán trọng số mục tiêu
    objective_weights: Dict[str, float]
    
    # Mục tiêu cụ thể cho mỗi cấu trúc
    structure_objectives: Dict[str, List[Dict[str, Any]]]
    
    # Mức độ tin cậy của dự đoán
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    prediction_time: float = field(default_factory=time.time)


class KBPFeatureExtractor:
    """
    Trích xuất đặc trưng từ dữ liệu bệnh nhân cho mô hình KBP.
    """
    
    def __init__(self):
        """Khởi tạo bộ trích xuất đặc trưng KBP."""
        pass
    
    def extract_features(
        self,
        patient_id: str,
        structure_set_id: str,
        targets: List[str],
        oars: List[str]
    ) -> KBPFeatures:
        """
        Trích xuất các đặc trưng KBP từ dữ liệu bệnh nhân và cấu trúc.
        
        Args:
            patient_id: ID bệnh nhân
            structure_set_id: ID tập cấu trúc
            targets: Danh sách cấu trúc mục tiêu (PTV)
            oars: Danh sách các cơ quan nguy cấp
            
        Returns:
            KBPFeatures: Các đặc trưng trích xuất
        """
        # Lấy thông tin cơ bản về bệnh nhân
        patient_features = get_patient_features(patient_id)
        
        # Lấy đặc trưng hình học
        geometry_features = get_geometry_features(patient_id, structure_set_id, targets, oars)
        
        # Tính các đặc trưng tổng hợp
        ptv = targets[0]  # Giả sử PTV đầu tiên là chính
        
        features = KBPFeatures(
            patient_id=patient_id,
            structure_set_id=structure_set_id,
            ptv_volume=geometry_features[f"{ptv}_volume"],
            ptv_surface_area=geometry_features[f"{ptv}_surface_area"],
            ptv_compactness=geometry_features[f"{ptv}_compactness"],
            ptv_location=geometry_features[f"{ptv}_center"],
            
            # Khoảng cách từ PTV đến các OAR
            distances_to_oars={
                oar: geometry_features[f"{ptv}_to_{oar}_distance"] 
                for oar in oars
            },
            
            # Thể tích trùng lặp
            overlap_volumes={
                oar: geometry_features[f"{ptv}_to_{oar}_overlap"] 
                for oar in oars
            },
            
            # Tỷ lệ trùng lặp
            overlap_ratios={
                oar: geometry_features[f"{ptv}_to_{oar}_overlap_ratio"] 
                for oar in oars
            },
            
            # Đặc trưng lâm sàng
            prescription_dose=patient_features.get("prescription_dose", 0),
            fractions=patient_features.get("fractions", 0),
            site=patient_features.get("site", ""),
            
            # Các đặc trưng khác
            additional_features=patient_features
        )
        
        return features
    
    def features_to_dataframe(self, features_list: List[KBPFeatures]) -> pd.DataFrame:
        """
        Chuyển đổi danh sách các đặc trưng thành DataFrame.
        
        Args:
            features_list: Danh sách các đối tượng KBPFeatures
            
        Returns:
            DataFrame: DataFrame chứa tất cả các đặc trưng
        """
        data = []
        
        for features in features_list:
            row = {
                "patient_id": features.patient_id,
                "structure_set_id": features.structure_set_id,
                "ptv_volume": features.ptv_volume,
                "ptv_surface_area": features.ptv_surface_area,
                "ptv_compactness": features.ptv_compactness,
                "ptv_location_x": features.ptv_location[0],
                "ptv_location_y": features.ptv_location[1],
                "ptv_location_z": features.ptv_location[2],
                "prescription_dose": features.prescription_dose,
                "fractions": features.fractions,
                "site": features.site,
            }
            
            # Thêm khoảng cách đến OAR
            for oar, distance in features.distances_to_oars.items():
                row[f"distance_to_{oar}"] = distance
            
            # Thêm thể tích trùng lặp
            for oar, overlap in features.overlap_volumes.items():
                row[f"overlap_volume_{oar}"] = overlap
            
            # Thêm tỷ lệ trùng lặp
            for oar, ratio in features.overlap_ratios.items():
                row[f"overlap_ratio_{oar}"] = ratio
            
            data.append(row)
        
        return pd.DataFrame(data)


class KBPModel:
    """
    Mô hình dự đoán dựa trên kiến thức cho lập kế hoạch xạ trị.
    """
    
    def __init__(
        self, 
        model_type: ModelType = ModelType.GRADIENT_BOOSTING,
        model_params: Optional[Dict[str, Any]] = None,
        model_name: str = "default_kbp_model"
    ):
        """
        Khởi tạo mô hình KBP.
        
        Args:
            model_type: Loại mô hình học máy
            model_params: Các tham số cho mô hình
            model_name: Tên mô hình
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model_name = model_name
        
        self.models = {}  # Dict chứa các mô hình đã huấn luyện cho các OAR và mục tiêu khác nhau
        self.scalers = {}  # Dict chứa các bộ chuẩn hóa dữ liệu
        self.feature_importances = {}  # Dict chứa tầm quan trọng của đặc trưng
        
        # Khởi tạo bộ trích xuất đặc trưng
        self.feature_extractor = KBPFeatureExtractor()
    
    def _create_model(self):
        """Tạo mô hình ML dựa trên loại đã chọn."""
        if self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(**self.model_params)
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(**self.model_params)
        elif self.model_type == ModelType.NEURAL_NETWORK:
            return MLPRegressor(**self.model_params)
        else:
            raise ValueError(f"Loại mô hình không được hỗ trợ: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: Dict[str, np.ndarray],
        do_grid_search: bool = False,
        grid_params: Optional[Dict[str, List[Any]]] = None,
        test_size: float = 0.2
    ) -> Dict[str, Dict[str, float]]:
        """
        Huấn luyện mô hình KBP.
        
        Args:
            X_train: DataFrame các đặc trưng đầu vào
            y_train: Dict chứa mảng đầu ra cho mỗi mục tiêu dự đoán
            do_grid_search: Có thực hiện tìm kiếm lưới tham số không
            grid_params: Dict các tham số cho tìm kiếm lưới
            test_size: Tỷ lệ phần dữ liệu kiểm tra
            
        Returns:
            Dict: Các chỉ số đánh giá cho mỗi mô hình
        """
        metrics = {}
        
        for target_name, y_values in y_train.items():
            logger.info(f"Huấn luyện mô hình cho: {target_name}")
            
            # Chia tập dữ liệu thành huấn luyện và kiểm tra
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train, y_values, test_size=test_size, random_state=42
            )
            
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_split)
            X_test_scaled = scaler.transform(X_test_split)
            
            # Lưu bộ chuẩn hóa
            self.scalers[target_name] = scaler
            
            # Tạo và huấn luyện mô hình
            if do_grid_search and grid_params:
                model = GridSearchCV(
                    self._create_model(), 
                    grid_params, 
                    cv=5, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train_split)
                logger.info(f"Tham số tốt nhất cho {target_name}: {model.best_params_}")
                
                # Sử dụng mô hình tốt nhất
                best_model = model.best_estimator_
            else:
                best_model = self._create_model()
                best_model.fit(X_train_scaled, y_train_split)
            
            # Lưu mô hình
            self.models[target_name] = best_model
            
            # Đánh giá mô hình
            y_pred = best_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test_split, y_pred)
            r2 = r2_score(y_test_split, y_pred)
            
            metrics[target_name] = {
                "mse": mse,
                "rmse": np.sqrt(mse),
                "r2": r2
            }
            
            logger.info(f"Đánh giá mô hình {target_name}: MSE={mse:.4f}, R2={r2:.4f}")
            
            # Lưu tầm quan trọng của đặc trưng nếu có
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importances[target_name] = {
                    feature: importance
                    for feature, importance in zip(X_train.columns, best_model.feature_importances_)
                }
        
        return metrics
    
    def predict(
        self,
        features: Union[KBPFeatures, pd.DataFrame]
    ) -> KBPPredictions:
        """
        Dự đoán với mô hình KBP.
        
        Args:
            features: Đặc trưng KBP hoặc DataFrame
            
        Returns:
            KBPPredictions: Kết quả dự đoán
        """
        # Chuyển đổi thành DataFrame nếu cần
        if isinstance(features, KBPFeatures):
            df = self.feature_extractor.features_to_dataframe([features])
        else:
            df = features
        
        # Kiểm tra mô hình
        if not self.models:
            raise ModelError("Mô hình chưa được huấn luyện")
        
        # Thực hiện dự đoán cho mỗi mục tiêu
        predictions = {}
        confidence = {}
        
        for target_name, model in self.models.items():
            # Chuẩn hóa dữ liệu đầu vào
            if target_name in self.scalers:
                X_scaled = self.scalers[target_name].transform(df)
            else:
                X_scaled = df.values
            
            # Dự đoán
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict(X_scaled)
                # Tính độ tin cậy dựa trên độ chắc chắn của mô hình
                proba = model.predict_proba(X_scaled)
                conf = np.max(proba, axis=1)[0]
                confidence[target_name] = conf
            else:
                y_pred = model.predict(X_scaled)
                confidence[target_name] = 0.8  # Giá trị mặc định
            
            predictions[target_name] = y_pred[0]
        
        # Dự đoán độ mở rộng
        ptv_margins = {k: v for k, v in predictions.items() if k.startswith("margin_")}
        
        # Dự đoán ràng buộc liều cho OAR
        oar_dose_constraints = {k: v for k, v in predictions.items() if k.startswith("dose_constraint_")}
        
        # Dự đoán trọng số mục tiêu
        objective_weights = {k: v for k, v in predictions.items() if k.startswith("weight_")}
        
        # Dự đoán mục tiêu cấu trúc
        structure_objectives = {}
        for k, v in predictions.items():
            if k.startswith("objective_"):
                parts = k.split("_")
                struct_name = parts[1]
                obj_type = "_".join(parts[2:])
                
                if struct_name not in structure_objectives:
                    structure_objectives[struct_name] = []
                
                structure_objectives[struct_name].append({
                    "type": obj_type,
                    "value": v
                })
        
        # Tạo đối tượng dự đoán
        kbp_predictions = KBPPredictions(
            patient_id=df["patient_id"].values[0],
            structure_set_id=df["structure_set_id"].values[0],
            ptv_margins=ptv_margins,
            oar_dose_constraints=oar_dose_constraints,
            objective_weights=objective_weights,
            structure_objectives=structure_objectives,
            confidence_scores=confidence
        )
        
        return kbp_predictions
    
    def save(self, directory: str) -> None:
        """
        Lưu mô hình vào thư mục.
        
        Args:
            directory: Đường dẫn thư mục
        """
        os.makedirs(directory, exist_ok=True)
        
        # Lưu mô hình
        for target_name, model in self.models.items():
            model_path = os.path.join(directory, f"{self.model_name}_{target_name}.joblib")
            joblib.dump(model, model_path)
        
        # Lưu bộ chuẩn hóa
        for target_name, scaler in self.scalers.items():
            scaler_path = os.path.join(directory, f"{self.model_name}_{target_name}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
        
        # Lưu metadata
        metadata = {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "model_name": self.model_name,
            "feature_importances": self.feature_importances,
            "targets": list(self.models.keys())
        }
        
        metadata_path = os.path.join(directory, f"{self.model_name}_metadata.pickle")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Đã lưu mô hình KBP vào {directory}")
    
    @classmethod
    def load(cls, directory: str, model_name: str) -> 'KBPModel':
        """
        Tải mô hình từ thư mục.
        
        Args:
            directory: Đường dẫn thư mục
            model_name: Tên mô hình
            
        Returns:
            KBPModel: Mô hình đã tải
        """
        # Tải metadata
        metadata_path = os.path.join(directory, f"{model_name}_metadata.pickle")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Khởi tạo mô hình
        model = cls(
            model_type=metadata["model_type"],
            model_params=metadata["model_params"],
            model_name=metadata["model_name"]
        )
        
        # Tải các mô hình con
        for target_name in metadata["targets"]:
            model_path = os.path.join(directory, f"{model_name}_{target_name}.joblib")
            model.models[target_name] = joblib.load(model_path)
            
            scaler_path = os.path.join(directory, f"{model_name}_{target_name}_scaler.joblib")
            if os.path.exists(scaler_path):
                model.scalers[target_name] = joblib.load(scaler_path)
        
        # Tải tầm quan trọng của đặc trưng
        model.feature_importances = metadata["feature_importances"]
        
        logger.info(f"Đã tải mô hình KBP từ {directory}")
        return model
    
    def get_feature_importances(self, target_name: str) -> Dict[str, float]:
        """
        Lấy tầm quan trọng của đặc trưng cho một mục tiêu.
        
        Args:
            target_name: Tên mục tiêu
            
        Returns:
            Dict: Tầm quan trọng của đặc trưng
        """
        if target_name not in self.feature_importances:
            raise ValueError(f"Không có thông tin tầm quan trọng cho mục tiêu {target_name}")
        
        return self.feature_importances[target_name] 