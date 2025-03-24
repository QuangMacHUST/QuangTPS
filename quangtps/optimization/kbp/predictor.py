#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module dự đoán và áp dụng các tham số tối ưu từ mô hình KBP vào quá trình lập kế hoạch.

Module này cung cấp các công cụ để áp dụng các mô hình KBP đã được huấn luyện
vào quá trình lập kế hoạch mới, tự động đề xuất các ràng buộc liều và tham số tối ưu.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import time

from quangtps.optimization.kbp.model import KBPModel, KBPFeatures, KBPFeatureExtractor, KBPPredictions
from quangtps.core.exceptions import ModelError, PredictionError
from quangtps.optimization.objectives import ObjectiveCollection, create_objective
from quangtps.optimization.constraints import ConstraintCollection, create_constraint
from quangtps.database.patient_db import PatientDatabase
from quangtps.database.structure_db import StructureDatabase

logger = logging.getLogger(__name__)

@dataclass
class KBPRecommendation:
    """Đề xuất từ mô hình KBP cho kế hoạch mới."""
    patient_id: str
    structure_set_id: str
    
    # Các ràng buộc liều được đề xuất
    dose_constraints: Dict[str, Dict[str, Any]]
    
    # Các mục tiêu tối ưu được đề xuất
    objectives: Dict[str, Dict[str, Any]]
    
    # Trọng số cho các mục tiêu
    weights: Dict[str, float]
    
    # Mức độ tin cậy
    confidence: Dict[str, float]
    
    # Các cấu trúc đã được sử dụng
    structures_used: Dict[str, Any]
    
    # Thời gian tạo
    creation_time: float = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi đề xuất thành dictionary."""
        return {
            "patient_id": self.patient_id,
            "structure_set_id": self.structure_set_id,
            "dose_constraints": self.dose_constraints,
            "objectives": self.objectives,
            "weights": self.weights,
            "confidence": self.confidence,
            "structures_used": self.structures_used,
            "creation_time": self.creation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KBPRecommendation':
        """Tạo đề xuất từ dictionary."""
        return cls(
            patient_id=data["patient_id"],
            structure_set_id=data["structure_set_id"],
            dose_constraints=data["dose_constraints"],
            objectives=data["objectives"],
            weights=data["weights"],
            confidence=data["confidence"],
            structures_used=data["structures_used"],
            creation_time=data.get("creation_time", time.time())
        )
    
    def save(self, file_path: str) -> None:
        """Lưu đề xuất vào file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'KBPRecommendation':
        """Tải đề xuất từ file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class KBPPredictor:
    """
    Dự đoán và áp dụng các tham số tối ưu từ mô hình KBP.
    """
    
    def __init__(self, models_dir: str = "models/kbp"):
        """
        Khởi tạo bộ dự đoán KBP.
        
        Args:
            models_dir: Thư mục chứa các mô hình KBP
        """
        self.models_dir = models_dir
        self.models = {}  # Lưu trữ mô hình theo vị trí điều trị
        self.feature_extractor = KBPFeatureExtractor()
        self.patient_db = PatientDatabase()
        self.structure_db = StructureDatabase()
    
    def load_model(self, site: str) -> KBPModel:
        """
        Tải mô hình KBP cho một vị trí điều trị.
        
        Args:
            site: Vị trí điều trị
            
        Returns:
            KBPModel: Mô hình KBP đã tải
        """
        if site in self.models:
            return self.models[site]
        
        # Chuẩn hóa tên vị trí
        site_lower = site.lower()
        
        # Tạo đường dẫn đến thư mục mô hình
        model_dir = os.path.join(self.models_dir, site_lower)
        model_name = f"kbp_{site_lower}"
        
        # Kiểm tra xem thư mục có tồn tại không
        if not os.path.exists(model_dir):
            available_sites = [d for d in os.listdir(self.models_dir) 
                              if os.path.isdir(os.path.join(self.models_dir, d))]
            
            if not available_sites:
                raise ModelError(f"Không tìm thấy mô hình KBP nào trong {self.models_dir}")
            
            raise ModelError(f"Không tìm thấy mô hình cho vị trí {site}. Các vị trí có sẵn: {', '.join(available_sites)}")
        
        try:
            # Tải mô hình
            model = KBPModel.load(model_dir, model_name)
            self.models[site] = model
            logger.info(f"Đã tải mô hình KBP cho vị trí {site}")
            return model
            
        except Exception as e:
            raise ModelError(f"Lỗi khi tải mô hình KBP cho vị trí {site}: {str(e)}")
    
    def predict(
        self,
        patient_id: str,
        structure_set_id: str,
        site: str,
        targets: List[str],
        oars: List[str]
    ) -> KBPPredictions:
        """
        Dự đoán các tham số tối ưu cho kế hoạch mới.
        
        Args:
            patient_id: ID bệnh nhân
            structure_set_id: ID tập cấu trúc
            site: Vị trí điều trị
            targets: Danh sách cấu trúc mục tiêu (PTV)
            oars: Danh sách các cơ quan nguy cấp
            
        Returns:
            KBPPredictions: Các dự đoán từ mô hình
        """
        # Tải mô hình cho vị trí điều trị
        model = self.load_model(site)
        
        # Trích xuất đặc trưng từ dữ liệu bệnh nhân
        features = self.feature_extractor.extract_features(
            patient_id, structure_set_id, targets, oars
        )
        
        # Chuyển đổi đặc trưng thành DataFrame
        features_df = self.feature_extractor.features_to_dataframe([features])
        
        # Dự đoán với mô hình
        predictions = model.predict(features_df)
        
        return predictions
    
    def generate_recommendation(
        self,
        patient_id: str,
        structure_set_id: str,
        site: str
    ) -> KBPRecommendation:
        """
        Tạo đề xuất KBP cho kế hoạch mới.
        
        Args:
            patient_id: ID bệnh nhân
            structure_set_id: ID tập cấu trúc
            site: Vị trí điều trị
            
        Returns:
            KBPRecommendation: Đề xuất từ mô hình KBP
        """
        # Lấy thông tin về cấu trúc
        structures = self.structure_db.get_structures_for_patient(patient_id, structure_set_id)
        
        if not structures:
            raise PredictionError(f"Không tìm thấy cấu trúc nào cho bệnh nhân {patient_id}, tập cấu trúc {structure_set_id}")
        
        # Phân loại cấu trúc
        targets = [s["name"] for s in structures if s["name"].lower().startswith("ptv")]
        oars = [s["name"] for s in structures if not s["name"].lower().startswith("ptv")]
        
        if not targets:
            raise PredictionError("Không tìm thấy cấu trúc PTV nào")
        
        # Dự đoán các tham số
        predictions = self.predict(patient_id, structure_set_id, site, targets, oars)
        
        # Tạo các ràng buộc liều
        dose_constraints = {}
        for oar_name in oars:
            oar_constraints = {}
            
            # Lấy các ràng buộc liều
            for k, v in predictions.oar_dose_constraints.items():
                if k.startswith(f"dose_constraint_{oar_name}_"):
                    constraint_type = k.replace(f"dose_constraint_{oar_name}_", "")
                    oar_constraints[constraint_type] = v
            
            if oar_constraints:
                dose_constraints[oar_name] = oar_constraints
        
        # Tạo các mục tiêu tối ưu
        objectives = {}
        for struct_name in set(targets + oars):
            struct_objectives = {}
            
            if struct_name in predictions.structure_objectives:
                for obj in predictions.structure_objectives[struct_name]:
                    struct_objectives[obj["type"]] = obj["value"]
            
            if struct_objectives:
                objectives[struct_name] = struct_objectives
        
        # Tạo đề xuất
        recommendation = KBPRecommendation(
            patient_id=patient_id,
            structure_set_id=structure_set_id,
            dose_constraints=dose_constraints,
            objectives=objectives,
            weights=predictions.objective_weights,
            confidence=predictions.confidence_scores,
            structures_used={"targets": targets, "oars": oars}
        )
        
        return recommendation
    
    def create_objective_collection(
        self, 
        recommendation: KBPRecommendation,
        prescription_dose: float
    ) -> ObjectiveCollection:
        """
        Tạo tập hợp mục tiêu tối ưu từ đề xuất KBP.
        
        Args:
            recommendation: Đề xuất KBP
            prescription_dose: Liều kê đơn
            
        Returns:
            ObjectiveCollection: Tập hợp mục tiêu tối ưu
        """
        objectives = ObjectiveCollection()
        
        # Tạo các mục tiêu cho từng cấu trúc
        for struct_name, struct_objectives in recommendation.objectives.items():
            for obj_type, obj_value in struct_objectives.items():
                # Điều chỉnh giá trị mục tiêu dựa trên liều kê đơn nếu cần
                adjusted_value = obj_value
                if obj_type in ["DOSE_PRESCRIPTION", "MAX_DOSE", "UNIFORM_DOSE"] and struct_name.lower().startswith("ptv"):
                    adjusted_value = prescription_dose * obj_value
                
                # Lấy trọng số
                weight_key = f"weight_{struct_name}_{obj_type}"
                weight = recommendation.weights.get(weight_key, 1.0)
                
                # Tạo mục tiêu
                objective = create_objective(
                    type_name=obj_type,
                    structure=struct_name,
                    value=adjusted_value
                )
                
                if objective:
                    objectives.add(objective, weight)
        
        return objectives
    
    def create_constraint_collection(
        self, 
        recommendation: KBPRecommendation,
        prescription_dose: float
    ) -> ConstraintCollection:
        """
        Tạo tập hợp ràng buộc từ đề xuất KBP.
        
        Args:
            recommendation: Đề xuất KBP
            prescription_dose: Liều kê đơn
            
        Returns:
            ConstraintCollection: Tập hợp ràng buộc
        """
        constraints = ConstraintCollection()
        
        # Tạo các ràng buộc cho từng cấu trúc
        for struct_name, struct_constraints in recommendation.dose_constraints.items():
            for const_type, const_value in struct_constraints.items():
                # Điều chỉnh giá trị ràng buộc dựa trên liều kê đơn nếu cần
                adjusted_value = const_value
                if const_type in ["MAX_DOSE", "MEAN_DOSE"]:
                    # Nếu giá trị ràng buộc là tỷ lệ (0-1), nhân với liều kê đơn
                    if 0 <= const_value <= 1:
                        adjusted_value = prescription_dose * const_value
                
                # Tạo ràng buộc
                constraint = create_constraint(
                    type_name=const_type,
                    structure=struct_name,
                    value=adjusted_value
                )
                
                if constraint:
                    constraints.add(constraint)
        
        return constraints 