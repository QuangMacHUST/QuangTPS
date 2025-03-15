"""
Module tìm kiếm và so khớp các bệnh án mẫu trong lập kế hoạch xạ trị dựa trên tri thức.

Module này cung cấp các lớp và hàm để tìm kiếm và so khớp các trường hợp tương tự
từ cơ sở dữ liệu các kế hoạch trước đó, giúp tạo ra kế hoạch mới dựa trên các kế hoạch
tương tự đã được thực hiện trước đó.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle

from quangtps.structures.structure import Structure
from quangtps.planning.plan import Plan, PlanType
from quangtps.core.patient import Patient
from quangtps.core.constants import EPSILON

logger = logging.getLogger(__name__)

class TemplateMatcher:
    """
    Lớp tìm kiếm và so khớp các bệnh án mẫu trong cơ sở dữ liệu.
    
    Lớp này cung cấp các phương thức để tìm kiếm các kế hoạch tương tự
    dựa trên đặc điểm của bệnh nhân, vùng khối u, và các cơ quan nguy cấp.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Khởi tạo trình tìm kiếm mẫu.
        
        Args:
            database_path: Đường dẫn đến cơ sở dữ liệu kế hoạch
        """
        self.database_path = database_path
        self.plan_features = []
        self.plans = []
        self.scaler = StandardScaler()
        self.nearest_neighbors = NearestNeighbors(n_neighbors=5, algorithm='auto')
        self.feature_names = []
        self.is_loaded = False
    
    def load_database(self, database_path: Optional[str] = None):
        """
        Tải cơ sở dữ liệu kế hoạch.
        
        Args:
            database_path: Đường dẫn đến cơ sở dữ liệu kế hoạch (ghi đè lên đường dẫn ban đầu)
        """
        if database_path is not None:
            self.database_path = database_path
            
        if self.database_path is None:
            raise ValueError("Chưa chỉ định đường dẫn cơ sở dữ liệu")
            
        if not os.path.exists(self.database_path):
            raise FileNotFoundError(f"Không tìm thấy cơ sở dữ liệu tại {self.database_path}")
            
        try:
            with open(self.database_path, 'rb') as f:
                data = pickle.load(f)
                
            self.plan_features = data.get('features', [])
            self.plans = data.get('plans', [])
            self.feature_names = data.get('feature_names', [])
            
            if len(self.plan_features) == 0:
                logger.warning("Cơ sở dữ liệu không chứa thông tin đặc trưng")
                return False
                
            # Chuẩn hóa dữ liệu
            self.scaler.fit(self.plan_features)
            scaled_features = self.scaler.transform(self.plan_features)
            
            # Xây dựng mô hình tìm kiếm lân cận
            self.nearest_neighbors.fit(scaled_features)
            
            self.is_loaded = True
            logger.info(f"Đã tải {len(self.plans)} kế hoạch từ cơ sở dữ liệu")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải cơ sở dữ liệu: {str(e)}")
            return False
    
    def extract_features(self, patient: Patient, target_structure: Structure) -> np.ndarray:
        """
        Trích xuất đặc trưng từ bệnh nhân và cấu trúc mục tiêu.
        
        Args:
            patient: Thông tin bệnh nhân
            target_structure: Cấu trúc mục tiêu (vùng khối u)
            
        Returns:
            np.ndarray: Vector đặc trưng
        """
        features = []
        
        # Thông tin bệnh nhân
        features.append(patient.age if hasattr(patient, 'age') else 50)  # Mặc định nếu không có
        features.append(1 if patient.gender == 'Male' else 0 if patient.gender == 'Female' else 0.5)
        
        # Thông tin khối u mục tiêu
        features.append(target_structure.get_volume())
        features.append(target_structure.get_surface_area())
        
        # Vị trí khối u - lấy trung tâm
        center = target_structure.get_center_of_mass()
        features.extend(center)
        
        # Tính chất hình dạng của khối u
        principal_moments = target_structure.get_principal_moments_of_inertia()
        if principal_moments is not None and len(principal_moments) == 3:
            features.extend(principal_moments)
            # Tỉ lệ hình dạng
            features.append(principal_moments[0] / principal_moments[2])  # Elongation
            features.append(principal_moments[1] / principal_moments[2])  # Flatness
        else:
            features.extend([1.0, 1.0, 1.0, 1.0, 1.0])  # Mặc định
        
        # Thông tin về các cơ quan nguy cấp
        oar_count = 0
        total_oar_volume = 0
        total_overlap = 0
        min_distance = float('inf')
        
        for structure in patient.structures.values():
            if structure.name != target_structure.name and structure.is_oar:
                oar_count += 1
                vol = structure.get_volume()
                total_oar_volume += vol
                
                overlap = structure.get_overlap_volume(target_structure)
                total_overlap += overlap
                
                dist = structure.get_minimum_distance(target_structure)
                min_distance = min(min_distance, dist)
        
        features.append(oar_count)
        features.append(total_oar_volume)
        features.append(total_overlap)
        features.append(min_distance if min_distance != float('inf') else 100.0)  # Mặc định 100mm nếu không có OAR
        
        return np.array(features)
    
    def find_similar_plans(self, patient: Patient, target_structure: Structure, 
                         n_neighbors: int = 3) -> List[Dict[str, Any]]:
        """
        Tìm các kế hoạch tương tự.
        
        Args:
            patient: Thông tin bệnh nhân
            target_structure: Cấu trúc mục tiêu (vùng khối u)
            n_neighbors: Số lượng kế hoạch tương tự cần tìm
            
        Returns:
            List[Dict[str, Any]]: Danh sách các kế hoạch tương tự và điểm tương đồng
        """
        if not self.is_loaded:
            raise RuntimeError("Cơ sở dữ liệu chưa được tải")
            
        # Trích xuất đặc trưng từ bệnh nhân và mục tiêu
        features = self.extract_features(patient, target_structure)
        
        # Chuẩn hóa đặc trưng
        features_scaled = self.scaler.transform([features])[0].reshape(1, -1)
        
        # Tìm kiếm lân cận gần nhất
        distances, indices = self.nearest_neighbors.kneighbors(features_scaled)
        
        # Lấy ra các kế hoạch tương tự
        similar_plans = []
        for i, idx in enumerate(indices[0]):
            if i >= n_neighbors:
                break
                
            if idx < len(self.plans):
                similarity_score = 1.0 / (1.0 + distances[0][i])  # Chuyển đổi khoảng cách thành điểm tương đồng
                
                similar_plans.append({
                    'plan': self.plans[idx],
                    'similarity_score': similarity_score,
                    'distance': distances[0][i]
                })
        
        # Sắp xếp theo điểm tương đồng giảm dần
        similar_plans.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_plans
    
    def apply_template(self, plan: Plan, template_plan: Plan) -> Plan:
        """
        Áp dụng các thông số từ kế hoạch mẫu vào kế hoạch mới.
        
        Args:
            plan: Kế hoạch cần cập nhật
            template_plan: Kế hoạch mẫu
            
        Returns:
            Plan: Kế hoạch đã được cập nhật
        """
        # Sao chép các thông số cơ bản
        if plan.plan_type == template_plan.plan_type:
            # Sao chép cấu hình chùm tia
            if hasattr(template_plan, 'beams') and hasattr(plan, 'beams'):
                # Nếu là kế hoạch IMRT/VMAT/3DCRT
                template_beams = template_plan.beams
                
                # Tạo các chùm tia mới dựa trên mẫu
                new_beams = []
                for beam in template_beams:
                    # Điều chỉnh vị trí chùm tia dựa trên tâm mục tiêu
                    new_beam = beam.copy()
                    new_beam.adjust_to_target(plan.target_structure)
                    new_beams.append(new_beam)
                
                plan.beams = new_beams
                
            # Sao chép các ràng buộc và mục tiêu
            if hasattr(template_plan, 'objectives') and hasattr(plan, 'objectives'):
                plan.objectives = self._adapt_objectives(plan, template_plan.objectives)
                
            if hasattr(template_plan, 'constraints') and hasattr(plan, 'constraints'):
                plan.constraints = self._adapt_constraints(plan, template_plan.constraints)
                
            # Sao chép các thông số tối ưu hóa
            if hasattr(template_plan, 'optimization_parameters') and hasattr(plan, 'optimization_parameters'):
                plan.optimization_parameters = template_plan.optimization_parameters.copy()
        
        return plan
    
    def _adapt_objectives(self, plan: Plan, template_objectives: List[Any]) -> List[Any]:
        """
        Điều chỉnh các mục tiêu từ kế hoạch mẫu cho phù hợp với kế hoạch mới.
        
        Args:
            plan: Kế hoạch mới
            template_objectives: Danh sách mục tiêu từ kế hoạch mẫu
            
        Returns:
            List[Any]: Danh sách mục tiêu đã được điều chỉnh
        """
        # Tạo mapping giữa các cấu trúc dựa trên tên và loại
        structure_mapping = self._map_structures(plan)
        
        adapted_objectives = []
        for obj in template_objectives:
            # Tìm cấu trúc tương ứng trong kế hoạch mới
            if hasattr(obj, 'structure_name') and obj.structure_name in structure_mapping:
                new_obj = obj.copy()  # Giả sử các objective có phương thức copy
                new_obj.structure_name = structure_mapping[obj.structure_name]
                adapted_objectives.append(new_obj)
        
        return adapted_objectives
    
    def _adapt_constraints(self, plan: Plan, template_constraints: List[Any]) -> List[Any]:
        """
        Điều chỉnh các ràng buộc từ kế hoạch mẫu cho phù hợp với kế hoạch mới.
        
        Args:
            plan: Kế hoạch mới
            template_constraints: Danh sách ràng buộc từ kế hoạch mẫu
            
        Returns:
            List[Any]: Danh sách ràng buộc đã được điều chỉnh
        """
        # Tương tự như _adapt_objectives
        structure_mapping = self._map_structures(plan)
        
        adapted_constraints = []
        for constraint in template_constraints:
            if hasattr(constraint, 'structure_name') and constraint.structure_name in structure_mapping:
                new_constraint = constraint.copy()
                new_constraint.structure_name = structure_mapping[constraint.structure_name]
                adapted_constraints.append(new_constraint)
        
        return adapted_constraints
    
    def _map_structures(self, plan: Plan) -> Dict[str, str]:
        """
        Tạo mapping giữa tên cấu trúc trong kế hoạch mẫu và kế hoạch mới.
        
        Args:
            plan: Kế hoạch mới
            
        Returns:
            Dict[str, str]: Mapping từ tên cấu trúc cũ sang tên cấu trúc mới
        """
        # Đơn giản chỉ dựa vào tên và loại cấu trúc
        # Trong thực tế, cần phương pháp tiên tiến hơn để map các cấu trúc chính xác
        mapping = {}
        
        # Giả định rằng plan.patient.structures chứa các cấu trúc hiện có
        existing_structures = {}
        for name, structure in plan.patient.structures.items():
            existing_structures[name.lower()] = name
            if hasattr(structure, 'type'):
                existing_structures[structure.type.lower()] = name
        
        # Khi gặp tên cấu trúc trong kế hoạch mẫu, tìm tên cấu trúc tương ứng trong kế hoạch mới
        for structure_name in existing_structures.keys():
            # Tìm cấu trúc giống nhất
            best_match = None
            best_score = 0
            
            for existing_name, actual_name in existing_structures.items():
                score = self._string_similarity(structure_name.lower(), existing_name.lower())
                if score > best_score:
                    best_score = score
                    best_match = actual_name
            
            if best_match and best_score > 0.6:  # Ngưỡng tương đồng
                mapping[structure_name] = best_match
        
        return mapping
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Tính độ tương đồng giữa hai chuỗi.
        
        Args:
            s1: Chuỗi thứ nhất
            s2: Chuỗi thứ hai
            
        Returns:
            float: Điểm tương đồng từ 0 đến 1
        """
        # Sử dụng Levenshtein distance hoặc cách đơn giản hơn
        # Đây là cách đơn giản để demo
        if not s1 or not s2:
            return 0.0
            
        # Tìm chuỗi con chung dài nhất
        s1, s2 = s1.lower(), s2.lower()
        m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
        longest = 0
        
        for x in range(1, 1 + len(s1)):
            for y in range(1, 1 + len(s2)):
                if s1[x-1] == s2[y-1]:
                    m[x][y] = m[x-1][y-1] + 1
                    longest = max(longest, m[x][y])
        
        return longest / max(len(s1), len(s2))
    
    def create_database_from_plans(self, plans: List[Plan], output_path: str):
        """
        Tạo cơ sở dữ liệu từ danh sách các kế hoạch.
        
        Args:
            plans: Danh sách các kế hoạch
            output_path: Đường dẫn để lưu cơ sở dữ liệu
        """
        features = []
        feature_names = []
        
        for plan in plans:
            if not hasattr(plan, 'patient') or not hasattr(plan, 'target_structure'):
                logger.warning(f"Kế hoạch thiếu thông tin bệnh nhân hoặc cấu trúc mục tiêu, bỏ qua")
                continue
                
            feature_vector = self.extract_features(plan.patient, plan.target_structure)
            features.append(feature_vector)
        
        # Lưu dữ liệu
        if len(features) > 0:
            # Chuẩn hóa dữ liệu
            self.scaler.fit(features)
            
            # Lưu vào file
            data = {
                'features': features,
                'feature_names': feature_names,
                'plans': plans
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Đã tạo cơ sở dữ liệu với {len(plans)} kế hoạch tại {output_path}")
            return True
        else:
            logger.error("Không có kế hoạch nào để tạo cơ sở dữ liệu")
            return False