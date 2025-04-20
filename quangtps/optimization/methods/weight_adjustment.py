#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module điều chỉnh trọng số tự động cho các mục tiêu tối ưu hóa.

Module này cung cấp các thuật toán để tự động điều chỉnh trọng số giữa các mục tiêu 
khác nhau trong tối ưu hóa kế hoạch xạ trị, dựa trên phân tích kết quả và các 
ràng buộc lâm sàng.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum, auto

from quangtps.optimization.objectives import ObjectiveBase, ObjectiveCollection
from quangtps.optimization.constraints import ConstraintBase, ConstraintCollection
from quangtps.dose.dose_grid import DoseGrid
from quangtps.evaluation.dvh import DVHCalculator
from quangtps.evaluation.metrics import EvaluationMetric

logger = logging.getLogger(__name__)

class WeightAdjustmentMethod(Enum):
    """Phương pháp điều chỉnh trọng số."""
    CLINICAL_PROTOCOL = auto()   # Dựa trên ngưỡng lâm sàng và ưu tiên
    ADAPTIVE = auto()            # Thích ứng dựa trên độ nhạy của mục tiêu
    BALANCED = auto()            # Cân bằng giữa các mục tiêu
    CONSTRAINT_DRIVEN = auto()   # Điều chỉnh dựa trên mức độ vi phạm ràng buộc
    PARETO_EXPLORATION = auto()  # Điều chỉnh để khám phá mặt Pareto

@dataclass
class WeightProfile:
    """Hồ sơ trọng số cho các mục tiêu tối ưu."""
    name: str
    weights: Dict[str, float]
    description: str = ""
    is_default: bool = False
    priority_order: List[str] = field(default_factory=list)
    
    def normalize(self) -> None:
        """Chuẩn hóa các trọng số để tổng bằng 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}

@dataclass
class WeightAdjustmentParameters:
    """Tham số cho điều chỉnh trọng số."""
    method: WeightAdjustmentMethod = WeightAdjustmentMethod.ADAPTIVE
    max_adjustments: int = 5  # Số lần điều chỉnh tối đa
    adjustment_scale: float = 0.1  # Tỷ lệ điều chỉnh mỗi lần
    min_weight: float = 0.01  # Trọng số tối thiểu
    max_weight: float = 0.95  # Trọng số tối đa
    convergence_threshold: float = 0.01  # Ngưỡng hội tụ
    
    # Tham số cho phương pháp CLINICAL_PROTOCOL
    clinical_thresholds: Dict[str, float] = field(default_factory=dict)
    clinical_priorities: Dict[str, int] = field(default_factory=dict)
    
    # Tham số cho phương pháp ADAPTIVE
    objective_sensitivities: Dict[str, float] = field(default_factory=dict)
    
    # Tham số cho phương pháp CONSTRAINT_DRIVEN
    constraint_penalties: Dict[str, float] = field(default_factory=dict)

class WeightAdjuster:
    """
    Bộ điều chỉnh trọng số tự động cho tối ưu hóa đa tiêu chí.
    
    Lớp này cung cấp các cơ chế để tự động điều chỉnh trọng số giữa các mục tiêu
    khác nhau, dựa trên kết quả tối ưu hiện tại và các ràng buộc lâm sàng.
    """
    
    def __init__(
        self,
        parameters: Optional[WeightAdjustmentParameters] = None
    ):
        """
        Khởi tạo bộ điều chỉnh trọng số.
        
        Args:
            parameters: Tham số điều chỉnh trọng số
        """
        self.parameters = parameters or WeightAdjustmentParameters()
        self.profiles: List[WeightProfile] = []
        self.current_profile: Optional[WeightProfile] = None
        self.adjustment_history: List[Dict[str, float]] = []
        self.evaluation_metrics: Dict[str, EvaluationMetric] = {}
        
    def add_profile(self, profile: WeightProfile) -> None:
        """
        Thêm hồ sơ trọng số.
        
        Args:
            profile: Hồ sơ trọng số
        """
        # Chuẩn hóa trọng số
        profile.normalize()
        self.profiles.append(profile)
        
        # Đặt làm mặc định nếu cần
        if profile.is_default and self.current_profile is None:
            self.current_profile = profile
    
    def set_current_profile(self, name: str) -> bool:
        """
        Đặt hồ sơ trọng số hiện tại theo tên.
        
        Args:
            name: Tên hồ sơ trọng số
            
        Returns:
            True nếu tìm thấy và đặt thành công, False nếu không
        """
        for profile in self.profiles:
            if profile.name == name:
                self.current_profile = profile
                return True
        return False
    
    def create_balanced_profile(self, objective_names: List[str]) -> WeightProfile:
        """
        Tạo hồ sơ trọng số cân bằng cho các mục tiêu.
        
        Args:
            objective_names: Danh sách tên các mục tiêu
            
        Returns:
            Hồ sơ trọng số cân bằng
        """
        n = len(objective_names)
        if n == 0:
            raise ValueError("Danh sách mục tiêu không được rỗng")
        
        weight = 1.0 / n
        weights = {name: weight for name in objective_names}
        
        return WeightProfile(
            name="Balanced",
            weights=weights,
            description="Trọng số cân bằng giữa các mục tiêu",
            is_default=True,
            priority_order=objective_names.copy()
        )
    
    def create_target_focused_profile(
        self, 
        objective_names: List[str],
        target_objectives: List[str]
    ) -> WeightProfile:
        """
        Tạo hồ sơ trọng số tập trung vào các mục tiêu đích.
        
        Args:
            objective_names: Danh sách tên tất cả các mục tiêu
            target_objectives: Danh sách tên các mục tiêu đích ưu tiên
            
        Returns:
            Hồ sơ trọng số tập trung vào mục tiêu đích
        """
        n = len(objective_names)
        t = len(target_objectives)
        
        if n == 0:
            raise ValueError("Danh sách mục tiêu không được rỗng")
        if t == 0:
            raise ValueError("Danh sách mục tiêu đích không được rỗng")
        
        # Đảm bảo tất cả mục tiêu đích nằm trong danh sách mục tiêu
        for target in target_objectives:
            if target not in objective_names:
                raise ValueError(f"Mục tiêu đích '{target}' không có trong danh sách mục tiêu")
        
        # Phân bổ trọng số: 80% cho mục tiêu đích, 20% cho các mục tiêu còn lại
        target_weight = 0.8 / t
        other_weight = 0.2 / (n - t)
        
        weights = {}
        for name in objective_names:
            if name in target_objectives:
                weights[name] = target_weight
            else:
                weights[name] = other_weight
        
        # Ưu tiên thứ tự: mục tiêu đích trước, còn lại sau
        priority_order = target_objectives.copy()
        for name in objective_names:
            if name not in target_objectives:
                priority_order.append(name)
        
        return WeightProfile(
            name="Target Focused",
            weights=weights,
            description="Ưu tiên các mục tiêu đích",
            is_default=False,
            priority_order=priority_order
        )
    
    def adjust_weights(
        self,
        current_weights: Dict[str, float],
        objective_values: Dict[str, float],
        constraint_violations: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Điều chỉnh trọng số dựa trên kết quả hiện tại.
        
        Args:
            current_weights: Trọng số hiện tại
            objective_values: Giá trị mục tiêu hiện tại
            constraint_violations: Mức độ vi phạm các ràng buộc (nếu có)
            
        Returns:
            Trọng số đã điều chỉnh
        """
        # Lưu trọng số hiện tại vào lịch sử
        self.adjustment_history.append(current_weights.copy())
        
        # Chọn phương pháp điều chỉnh
        if self.parameters.method == WeightAdjustmentMethod.CLINICAL_PROTOCOL:
            return self._adjust_by_clinical_protocol(current_weights, objective_values)
        elif self.parameters.method == WeightAdjustmentMethod.ADAPTIVE:
            return self._adjust_adaptively(current_weights, objective_values)
        elif self.parameters.method == WeightAdjustmentMethod.CONSTRAINT_DRIVEN:
            return self._adjust_by_constraints(current_weights, objective_values, constraint_violations)
        elif self.parameters.method == WeightAdjustmentMethod.BALANCED:
            return self._adjust_for_balance(current_weights, objective_values)
        else:
            return current_weights  # Không thay đổi
    
    def _adjust_by_clinical_protocol(
        self,
        current_weights: Dict[str, float],
        objective_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Điều chỉnh trọng số dựa trên ngưỡng lâm sàng và ưu tiên.
        
        Args:
            current_weights: Trọng số hiện tại
            objective_values: Giá trị mục tiêu hiện tại
            
        Returns:
            Trọng số đã điều chỉnh
        """
        new_weights = current_weights.copy()
        thresholds = self.parameters.clinical_thresholds
        priorities = self.parameters.clinical_priorities
        
        # Tìm các mục tiêu vi phạm ngưỡng
        violations = {}
        for name, value in objective_values.items():
            if name in thresholds and value > thresholds[name]:
                # Tính mức độ vi phạm (chuẩn hóa)
                violations[name] = (value - thresholds[name]) / thresholds[name]
        
        if not violations:
            return current_weights  # Không có vi phạm
        
        # Điều chỉnh trọng số dựa trên mức độ vi phạm và ưu tiên
        for name, violation in violations.items():
            priority = priorities.get(name, 1)
            adjustment = violation * priority * self.parameters.adjustment_scale
            
            # Tăng trọng số cho mục tiêu bị vi phạm
            new_weights[name] = min(
                new_weights[name] + adjustment,
                self.parameters.max_weight
            )
        
        # Chuẩn hóa trọng số
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        return new_weights
    
    def _adjust_adaptively(
        self,
        current_weights: Dict[str, float],
        objective_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Điều chỉnh trọng số thích ứng dựa trên độ nhạy của mục tiêu.
        
        Args:
            current_weights: Trọng số hiện tại
            objective_values: Giá trị mục tiêu hiện tại
            
        Returns:
            Trọng số đã điều chỉnh
        """
        # Nếu chưa có lịch sử điều chỉnh, không thể tính độ nhạy
        if len(self.adjustment_history) < 2:
            return current_weights
        
        new_weights = current_weights.copy()
        sensitivities = self.parameters.objective_sensitivities
        
        # Tính độ nhạy dựa trên thay đổi của mục tiêu và trọng số
        for name in objective_values:
            if name not in sensitivities:
                continue
                
            # Điều chỉnh dựa trên độ nhạy đã biết
            sensitivity = sensitivities[name]
            adjustment = self.parameters.adjustment_scale * sensitivity
            
            # Nếu độ nhạy cao, giảm trọng số (vì mục tiêu phản ứng mạnh)
            if sensitivity > 1.0:
                new_weights[name] = max(
                    new_weights[name] - adjustment,
                    self.parameters.min_weight
                )
            # Nếu độ nhạy thấp, tăng trọng số (vì mục tiêu phản ứng yếu)
            elif sensitivity < 1.0:
                new_weights[name] = min(
                    new_weights[name] + adjustment,
                    self.parameters.max_weight
                )
        
        # Chuẩn hóa trọng số
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        return new_weights
    
    def _adjust_by_constraints(
        self,
        current_weights: Dict[str, float],
        objective_values: Dict[str, float],
        constraint_violations: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Điều chỉnh trọng số dựa trên mức độ vi phạm ràng buộc.
        
        Args:
            current_weights: Trọng số hiện tại
            objective_values: Giá trị mục tiêu hiện tại
            constraint_violations: Mức độ vi phạm các ràng buộc
            
        Returns:
            Trọng số đã điều chỉnh
        """
        if constraint_violations is None or not constraint_violations:
            return current_weights  # Không có vi phạm
        
        new_weights = current_weights.copy()
        penalties = self.parameters.constraint_penalties
        
        # Điều chỉnh trọng số dựa trên mức độ vi phạm ràng buộc
        for constraint_name, violation in constraint_violations.items():
            if constraint_name not in penalties:
                continue
                
            penalty = penalties[constraint_name]
            adjustment = violation * penalty * self.parameters.adjustment_scale
            
            # Tìm mục tiêu liên quan đến ràng buộc này
            related_objective = constraint_name.split('_')[0]  # Giả sử tên ràng buộc có cấu trúc: objective_constraint
            
            if related_objective in new_weights:
                # Tăng trọng số cho mục tiêu liên quan
                new_weights[related_objective] = min(
                    new_weights[related_objective] + adjustment,
                    self.parameters.max_weight
                )
        
        # Chuẩn hóa trọng số
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        return new_weights
    
    def _adjust_for_balance(
        self,
        current_weights: Dict[str, float],
        objective_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Điều chỉnh trọng số để cân bằng giữa các mục tiêu.
        
        Args:
            current_weights: Trọng số hiện tại
            objective_values: Giá trị mục tiêu hiện tại
            
        Returns:
            Trọng số đã điều chỉnh
        """
        # Tìm mục tiêu có giá trị tệ nhất
        worst_objective = max(objective_values.items(), key=lambda x: x[1])[0]
        
        # Điều chỉnh trọng số cho mục tiêu tệ nhất
        new_weights = current_weights.copy()
        new_weights[worst_objective] = min(
            new_weights[worst_objective] + self.parameters.adjustment_scale,
            self.parameters.max_weight
        )
        
        # Chuẩn hóa trọng số
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        return new_weights
    
    def calculate_sensitivities(
        self,
        weight_history: List[Dict[str, float]],
        objective_history: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Tính toán độ nhạy của các mục tiêu dựa trên lịch sử điều chỉnh.
        
        Args:
            weight_history: Lịch sử trọng số
            objective_history: Lịch sử giá trị mục tiêu
            
        Returns:
            Độ nhạy của các mục tiêu
        """
        if len(weight_history) < 2 or len(objective_history) < 2:
            return {}
        
        sensitivities = {}
        
        # Lấy tất cả tên mục tiêu
        objectives = set()
        for obj_values in objective_history:
            objectives.update(obj_values.keys())
        
        for obj_name in objectives:
            # Tính độ nhạy dựa trên thay đổi tương đối
            weight_changes = []
            obj_changes = []
            
            for i in range(1, len(weight_history)):
                if obj_name in weight_history[i-1] and obj_name in weight_history[i] and \
                   obj_name in objective_history[i-1] and obj_name in objective_history[i]:
                    
                    prev_weight = weight_history[i-1][obj_name]
                    curr_weight = weight_history[i][obj_name]
                    
                    if prev_weight > 0:
                        weight_change = (curr_weight - prev_weight) / prev_weight
                        weight_changes.append(weight_change)
                        
                        prev_obj = objective_history[i-1][obj_name]
                        curr_obj = objective_history[i][obj_name]
                        
                        if prev_obj != 0:
                            obj_change = (curr_obj - prev_obj) / abs(prev_obj)
                            obj_changes.append(obj_change)
            
            # Tính độ nhạy là tỷ lệ trung bình giữa thay đổi mục tiêu và thay đổi trọng số
            if weight_changes and obj_changes and len(weight_changes) == len(obj_changes):
                # Tính trung bình của tỷ lệ thay đổi
                ratios = [abs(o/w) if w != 0 else 0 for o, w in zip(obj_changes, weight_changes)]
                sensitivities[obj_name] = sum(ratios) / len(ratios)
        
        return sensitivities
    
    def export_profiles(self, filepath: str) -> None:
        """
        Xuất các hồ sơ trọng số ra file.
        
        Args:
            filepath: Đường dẫn file
        """
        data = []
        for profile in self.profiles:
            profile_data = {
                'name': profile.name,
                'description': profile.description,
                'is_default': profile.is_default,
                'priority_order': ','.join(profile.priority_order)
            }
            
            # Thêm trọng số
            for obj_name, weight in profile.weights.items():
                profile_data[f'weight_{obj_name}'] = weight
            
            data.append(profile_data)
        
        # Tạo DataFrame và lưu ra file
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def import_profiles(self, filepath: str) -> None:
        """
        Nhập các hồ sơ trọng số từ file.
        
        Args:
            filepath: Đường dẫn file
        """
        try:
            df = pd.read_csv(filepath)
            self.profiles = []
            
            for _, row in df.iterrows():
                weights = {}
                priority_order = []
                
                # Lấy trọng số
                for col in df.columns:
                    if col.startswith('weight_'):
                        obj_name = col[7:]  # Bỏ tiền tố 'weight_'
                        weights[obj_name] = row[col]
                
                # Lấy thứ tự ưu tiên
                if 'priority_order' in row and row['priority_order']:
                    priority_order = row['priority_order'].split(',')
                
                profile = WeightProfile(
                    name=row['name'],
                    weights=weights,
                    description=row['description'] if 'description' in row else "",
                    is_default=row['is_default'] if 'is_default' in row else False,
                    priority_order=priority_order
                )
                
                self.profiles.append(profile)
                
                # Đặt làm mặc định nếu cần
                if profile.is_default and self.current_profile is None:
                    self.current_profile = profile
        except Exception as e:
            logger.error(f"Lỗi khi nhập hồ sơ trọng số: {str(e)}")
