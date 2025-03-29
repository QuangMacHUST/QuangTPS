#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module phân tích và điều chỉnh lỗi thiết lập (setup error) trong xạ trị.
Cung cấp các công cụ để ước tính và điều chỉnh sai số thiết lập.
"""

import os
import numpy as np
import logging
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum, auto
from dataclasses import dataclass

from quangtps.core.types import Patient, Image, Structure, Dose
from quangtps.core.exceptions import SetupErrorAnalysisError
from quangtps.adaptive.deformation.registration import RegistrationType

logger = logging.getLogger(__name__)

class SetupErrorType(Enum):
    """Phân loại các loại lỗi thiết lập"""
    SYSTEMATIC = auto()    # Lỗi hệ thống - xuất hiện nhất quán trong tất cả các phân đoạn
    RANDOM = auto()        # Lỗi ngẫu nhiên - thay đổi theo từng phân đoạn
    ROTATIONAL = auto()    # Lỗi xoay
    TRANSLATIONAL = auto() # Lỗi tịnh tiến
    DEFORMATIONAL = auto() # Lỗi biến dạng

class CorrectionStrategy(Enum):
    """Chiến lược điều chỉnh lỗi thiết lập"""
    NO_ACTION = auto()             # Không can thiệp
    COUCH_SHIFT = auto()           # Dịch chuyển bàn điều trị
    ONLINE_REPLANNING = auto()     # Lập kế hoạch lại trực tuyến
    RESET_PATIENT = auto()         # Đặt lại vị trí bệnh nhân
    ROTATION_CORRECTION = auto()   # Điều chỉnh góc xoay
    MARGIN_ADJUSTMENT = auto()     # Điều chỉnh lề

@dataclass
class SetupDeviation:
    """Lưu trữ thông tin về độ lệch thiết lập một phân đoạn"""
    fraction_number: int
    delivery_date: datetime.datetime
    # Độ lệch tịnh tiến (mm)
    x_shift: float = 0.0
    y_shift: float = 0.0
    z_shift: float = 0.0
    # Độ lệch xoay (độ)
    x_rotation: float = 0.0  # pitch
    y_rotation: float = 0.0  # roll
    z_rotation: float = 0.0  # yaw
    # Thông tin bổ sung
    cbct_image_id: Optional[str] = None
    planning_ct_id: Optional[str] = None
    registration_method: Optional[str] = None
    applied_correction: Optional[CorrectionStrategy] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dạng từ điển để lưu trữ"""
        return {
            "fraction_number": self.fraction_number,
            "delivery_date": self.delivery_date.isoformat(),
            "x_shift": self.x_shift,
            "y_shift": self.y_shift,
            "z_shift": self.z_shift,
            "x_rotation": self.x_rotation,
            "y_rotation": self.y_rotation,
            "z_rotation": self.z_rotation,
            "cbct_image_id": self.cbct_image_id,
            "planning_ct_id": self.planning_ct_id,
            "registration_method": self.registration_method,
            "applied_correction": self.applied_correction.name if self.applied_correction else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SetupDeviation':
        """Tạo đối tượng từ từ điển"""
        return cls(
            fraction_number=data["fraction_number"],
            delivery_date=datetime.datetime.fromisoformat(data["delivery_date"]),
            x_shift=data.get("x_shift", 0.0),
            y_shift=data.get("y_shift", 0.0),
            z_shift=data.get("z_shift", 0.0),
            x_rotation=data.get("x_rotation", 0.0),
            y_rotation=data.get("y_rotation", 0.0),
            z_rotation=data.get("z_rotation", 0.0),
            cbct_image_id=data.get("cbct_image_id"),
            planning_ct_id=data.get("planning_ct_id"),
            registration_method=data.get("registration_method"),
            applied_correction=CorrectionStrategy[data["applied_correction"]] if data.get("applied_correction") else None
        )
    
    def get_magnitude(self) -> float:
        """Tính độ lớn của độ lệch tịnh tiến (mm)"""
        return np.sqrt(self.x_shift**2 + self.y_shift**2 + self.z_shift**2)
    
    def get_rotation_magnitude(self) -> float:
        """Tính độ lớn của độ lệch xoay (độ)"""
        return np.sqrt(self.x_rotation**2 + self.y_rotation**2 + self.z_rotation**2)

class SetupErrorEstimator:
    """
    Lớp ước tính và phân tích lỗi thiết lập
    """
    
    def __init__(self, patient_id: str):
        """
        Khởi tạo đối tượng ước tính lỗi thiết lập
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        """
        self.patient_id = patient_id
        self.deviations: List[SetupDeviation] = []
        self.creation_date = datetime.datetime.now()
        self.id = f"setup_err_{patient_id}_{self.creation_date.strftime('%Y%m%d%H%M%S')}"
        self.systematic_error = None
        self.random_error = None
        self.analysis_date = None
    
    def add_deviation(self, deviation: SetupDeviation):
        """
        Thêm một độ lệch thiết lập
        
        Parameters
        ----------
        deviation : SetupDeviation
            Đối tượng độ lệch thiết lập
        """
        # Kiểm tra xem đã có độ lệch cho phân đoạn này chưa
        existing = [d for d in self.deviations if d.fraction_number == deviation.fraction_number]
        if existing:
            # Nếu đã có, thay thế nó
            for i, existing_deviation in enumerate(self.deviations):
                if existing_deviation.fraction_number == deviation.fraction_number:
                    self.deviations[i] = deviation
                    break
        else:
            # Nếu chưa có, thêm mới
            self.deviations.append(deviation)
            
        # Sắp xếp lại theo số thứ tự phân đoạn
        self.deviations.sort(key=lambda x: x.fraction_number)
        
        # Đánh dấu rằng cần phân tích lại
        self.systematic_error = None
        self.random_error = None
    
    def analyze_setup_errors(self) -> Dict[str, Any]:
        """
        Phân tích lỗi thiết lập dựa trên các độ lệch đã ghi nhận
        
        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích lỗi thiết lập
        """
        if not self.deviations:
            raise SetupErrorAnalysisError("Không có dữ liệu độ lệch thiết lập để phân tích")
        
        # Tạo mảng numpy từ các độ lệch để dễ phân tích
        shifts = np.array([[d.x_shift, d.y_shift, d.z_shift] for d in self.deviations])
        rotations = np.array([[d.x_rotation, d.y_rotation, d.z_rotation] for d in self.deviations])
        
        # Tính lỗi hệ thống (trung bình của tất cả các độ lệch)
        systematic_shift = np.mean(shifts, axis=0)
        systematic_rotation = np.mean(rotations, axis=0)
        
        # Tính lỗi ngẫu nhiên (độ lệch chuẩn của sai số)
        random_shift = np.std(shifts, axis=0)
        random_rotation = np.std(rotations, axis=0)
        
        # Lưu kết quả phân tích
        self.systematic_error = {
            "x_shift": systematic_shift[0],
            "y_shift": systematic_shift[1],
            "z_shift": systematic_shift[2],
            "x_rotation": systematic_rotation[0],
            "y_rotation": systematic_rotation[1],
            "z_rotation": systematic_rotation[2]
        }
        
        self.random_error = {
            "x_shift": random_shift[0],
            "y_shift": random_shift[1],
            "z_shift": random_shift[2],
            "x_rotation": random_rotation[0],
            "y_rotation": random_rotation[1],
            "z_rotation": random_rotation[2]
        }
        
        self.analysis_date = datetime.datetime.now()
        
        return {
            "systematic_error": self.systematic_error,
            "random_error": self.random_error,
            "analysis_date": self.analysis_date.isoformat(),
            "number_of_fractions": len(self.deviations)
        }
    
    def calculate_recommended_margins(self, recipe: str = "van_herk") -> Dict[str, float]:
        """
        Tính toán lề khuyến nghị dựa trên phân tích lỗi thiết lập
        
        Parameters
        ----------
        recipe : str, optional
            Công thức tính lề, by default "van_herk"
            
        Returns
        -------
        Dict[str, float]
            Lề khuyến nghị theo mỗi hướng (mm)
        """
        if not self.systematic_error or not self.random_error:
            self.analyze_setup_errors()
        
        # Trích xuất giá trị
        systematic = [
            self.systematic_error["x_shift"],
            self.systematic_error["y_shift"],
            self.systematic_error["z_shift"]
        ]
        random = [
            self.random_error["x_shift"],
            self.random_error["y_shift"],
            self.random_error["z_shift"]
        ]
        
        margins = {}
        
        if recipe == "van_herk":
            # Công thức Van Herk: M = 2.5Σ + 0.7σ
            # Σ: lỗi hệ thống, σ: lỗi ngẫu nhiên
            margins = {
                "x_margin": 2.5 * abs(systematic[0]) + 0.7 * random[0],
                "y_margin": 2.5 * abs(systematic[1]) + 0.7 * random[1],
                "z_margin": 2.5 * abs(systematic[2]) + 0.7 * random[2]
            }
        elif recipe == "stroom":
            # Công thức Stroom: M = 2Σ + 0.7σ
            margins = {
                "x_margin": 2.0 * abs(systematic[0]) + 0.7 * random[0],
                "y_margin": 2.0 * abs(systematic[1]) + 0.7 * random[1],
                "z_margin": 2.0 * abs(systematic[2]) + 0.7 * random[2]
            }
        elif recipe == "simple":
            # Công thức đơn giản: M = 3Σ
            margins = {
                "x_margin": 3.0 * abs(systematic[0]),
                "y_margin": 3.0 * abs(systematic[1]),
                "z_margin": 3.0 * abs(systematic[2])
            }
        else:
            raise ValueError(f"Không hỗ trợ công thức '{recipe}'")
        
        return margins
    
    def get_deviation_trends(self) -> pd.DataFrame:
        """
        Tạo DataFrame chứa xu hướng độ lệch thiết lập theo thời gian
        
        Returns
        -------
        pd.DataFrame
            DataFrame chứa thông tin xu hướng độ lệch
        """
        if not self.deviations:
            raise SetupErrorAnalysisError("Không có dữ liệu độ lệch thiết lập")
        
        # Tạo DataFrame
        data = []
        for deviation in self.deviations:
            data.append({
                "fraction": deviation.fraction_number,
                "date": deviation.delivery_date,
                "x_shift": deviation.x_shift,
                "y_shift": deviation.y_shift,
                "z_shift": deviation.z_shift,
                "x_rotation": deviation.x_rotation,
                "y_rotation": deviation.y_rotation,
                "z_rotation": deviation.z_rotation,
                "magnitude": deviation.get_magnitude(),
                "rotation_magnitude": deviation.get_rotation_magnitude(),
                "correction": deviation.applied_correction.name if deviation.applied_correction else "NONE"
            })
        
        return pd.DataFrame(data)
    
    def visualize_deviations(self, output_file: str = None) -> Optional[plt.Figure]:
        """
        Tạo biểu đồ trực quan hóa độ lệch thiết lập theo thời gian
        
        Parameters
        ----------
        output_file : str, optional
            Đường dẫn tệp tin để lưu biểu đồ, by default None
            
        Returns
        -------
        Optional[plt.Figure]
            Đối tượng Figure nếu không lưu tệp tin
        """
        df = self.get_deviation_trends()
        
        # Tạo biểu đồ
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Vẽ biểu đồ dịch chuyển
        axes[0].plot(df["fraction"], df["x_shift"], 'r-o', label='X')
        axes[0].plot(df["fraction"], df["y_shift"], 'g-o', label='Y')
        axes[0].plot(df["fraction"], df["z_shift"], 'b-o', label='Z')
        axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0].set_xlabel('Phân đoạn')
        axes[0].set_ylabel('Dịch chuyển (mm)')
        axes[0].set_title('Độ lệch dịch chuyển theo phân đoạn')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Vẽ biểu đồ xoay
        axes[1].plot(df["fraction"], df["x_rotation"], 'r-o', label='X (pitch)')
        axes[1].plot(df["fraction"], df["y_rotation"], 'g-o', label='Y (roll)')
        axes[1].plot(df["fraction"], df["z_rotation"], 'b-o', label='Z (yaw)')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_xlabel('Phân đoạn')
        axes[1].set_ylabel('Xoay (độ)')
        axes[1].set_title('Độ lệch xoay theo phân đoạn')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        fig.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            plt.close(fig)
            return None
        
        return fig
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi sang dạng từ điển để lưu trữ
        
        Returns
        -------
        Dict[str, Any]
            Từ điển biểu diễn đối tượng
        """
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "creation_date": self.creation_date.isoformat(),
            "analysis_date": self.analysis_date.isoformat() if self.analysis_date else None,
            "deviations": [d.to_dict() for d in self.deviations],
            "systematic_error": self.systematic_error,
            "random_error": self.random_error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SetupErrorEstimator':
        """
        Tạo đối tượng từ từ điển
        
        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển biểu diễn đối tượng
            
        Returns
        -------
        SetupErrorEstimator
            Đối tượng được tạo
        """
        estimator = cls(data["patient_id"])
        estimator.id = data["id"]
        estimator.creation_date = datetime.datetime.fromisoformat(data["creation_date"])
        if data.get("analysis_date"):
            estimator.analysis_date = datetime.datetime.fromisoformat(data["analysis_date"])
        
        for deviation_data in data.get("deviations", []):
            estimator.deviations.append(SetupDeviation.from_dict(deviation_data))
        
        estimator.systematic_error = data.get("systematic_error")
        estimator.random_error = data.get("random_error")
        
        return estimator

class SetupCorrectionStrategy:
    """
    Lớp đề xuất chiến lược điều chỉnh lỗi thiết lập
    """
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Khởi tạo đối tượng chiến lược điều chỉnh
        
        Parameters
        ----------
        thresholds : Dict[str, float], optional
            Ngưỡng cho các chiến lược điều chỉnh khác nhau, by default None
        """
        # Ngưỡng mặc định (mm hoặc độ)
        self.thresholds = thresholds or {
            "no_action": 2.0,        # Dưới 2mm/độ không cần điều chỉnh
            "couch_shift": 5.0,      # Dưới 5mm/độ điều chỉnh bàn
            "rotation_threshold": 3.0,  # Ngưỡng xoay cần điều chỉnh
            "replanning": 10.0,      # Trên 10mm/độ cần lập kế hoạch lại
        }
    
    def suggest_correction(self, deviation: SetupDeviation) -> CorrectionStrategy:
        """
        Đề xuất chiến lược điều chỉnh cho một độ lệch thiết lập
        
        Parameters
        ----------
        deviation : SetupDeviation
            Đối tượng độ lệch thiết lập
            
        Returns
        -------
        CorrectionStrategy
            Chiến lược điều chỉnh được đề xuất
        """
        # Tính độ lớn của độ lệch
        translation_magnitude = deviation.get_magnitude()
        rotation_magnitude = deviation.get_rotation_magnitude()
        
        # Đề xuất chiến lược dựa trên ngưỡng
        if translation_magnitude <= self.thresholds["no_action"] and rotation_magnitude <= self.thresholds["no_action"]:
            return CorrectionStrategy.NO_ACTION
        
        elif translation_magnitude > self.thresholds["replanning"] or rotation_magnitude > self.thresholds["replanning"]:
            return CorrectionStrategy.ONLINE_REPLANNING
        
        elif rotation_magnitude > self.thresholds["rotation_threshold"]:
            # Nếu có độ lệch xoay đáng kể
            if translation_magnitude > self.thresholds["couch_shift"]:
                # Nếu có cả dịch chuyển lớn, đề xuất đặt lại vị trí bệnh nhân
                return CorrectionStrategy.RESET_PATIENT
            else:
                # Nếu chỉ có xoay, điều chỉnh góc xoay
                return CorrectionStrategy.ROTATION_CORRECTION
        
        elif translation_magnitude > self.thresholds["couch_shift"]:
            return CorrectionStrategy.COUCH_SHIFT
        
        # Mặc định điều chỉnh lề
        return CorrectionStrategy.MARGIN_ADJUSTMENT
    
    def apply_correction(self, deviation: SetupDeviation) -> Dict[str, Any]:
        """
        Áp dụng chiến lược điều chỉnh và trả về thông tin điều chỉnh
        
        Parameters
        ----------
        deviation : SetupDeviation
            Đối tượng độ lệch thiết lập
            
        Returns
        -------
        Dict[str, Any]
            Thông tin về điều chỉnh đã áp dụng
        """
        strategy = self.suggest_correction(deviation)
        
        # Thông tin điều chỉnh
        correction_info = {
            "strategy": strategy.name,
            "original_deviation": deviation.to_dict(),
            "corrected_deviation": None,
            "correction_values": {}
        }
        
        # Bản sao của độ lệch để áp dụng điều chỉnh
        corrected_deviation = SetupDeviation(
            fraction_number=deviation.fraction_number,
            delivery_date=deviation.delivery_date,
            x_shift=deviation.x_shift,
            y_shift=deviation.y_shift,
            z_shift=deviation.z_shift,
            x_rotation=deviation.x_rotation,
            y_rotation=deviation.y_rotation,
            z_rotation=deviation.z_rotation,
            cbct_image_id=deviation.cbct_image_id,
            planning_ct_id=deviation.planning_ct_id,
            registration_method=deviation.registration_method,
        )
        
        # Áp dụng chiến lược điều chỉnh
        if strategy == CorrectionStrategy.NO_ACTION:
            # Không cần điều chỉnh
            pass
        
        elif strategy == CorrectionStrategy.COUCH_SHIFT:
            # Điều chỉnh bàn để bù lại độ lệch tịnh tiến
            correction_info["correction_values"] = {
                "table_x": -deviation.x_shift,
                "table_y": -deviation.y_shift,
                "table_z": -deviation.z_shift
            }
            # Sau khi điều chỉnh, độ lệch sẽ về 0
            corrected_deviation.x_shift = 0
            corrected_deviation.y_shift = 0
            corrected_deviation.z_shift = 0
        
        elif strategy == CorrectionStrategy.ROTATION_CORRECTION:
            # Điều chỉnh góc xoay
            correction_info["correction_values"] = {
                "pitch": -deviation.x_rotation,
                "roll": -deviation.y_rotation,
                "yaw": -deviation.z_rotation
            }
            # Sau khi điều chỉnh, độ lệch xoay sẽ về 0
            corrected_deviation.x_rotation = 0
            corrected_deviation.y_rotation = 0
            corrected_deviation.z_rotation = 0
        
        elif strategy == CorrectionStrategy.RESET_PATIENT:
            # Đặt lại vị trí bệnh nhân (điều chỉnh tất cả)
            correction_info["correction_values"] = {
                "table_x": -deviation.x_shift,
                "table_y": -deviation.y_shift,
                "table_z": -deviation.z_shift,
                "pitch": -deviation.x_rotation,
                "roll": -deviation.y_rotation,
                "yaw": -deviation.z_rotation
            }
            # Sau khi điều chỉnh, tất cả độ lệch sẽ về 0
            corrected_deviation.x_shift = 0
            corrected_deviation.y_shift = 0
            corrected_deviation.z_shift = 0
            corrected_deviation.x_rotation = 0
            corrected_deviation.y_rotation = 0
            corrected_deviation.z_rotation = 0
        
        elif strategy == CorrectionStrategy.ONLINE_REPLANNING:
            # Đề xuất lập kế hoạch lại trực tuyến
            correction_info["correction_values"] = {
                "requires_replanning": True,
                "reason": "Significant setup deviation"
            }
        
        elif strategy == CorrectionStrategy.MARGIN_ADJUSTMENT:
            # Đề xuất điều chỉnh lề cho các phân đoạn tiếp theo
            correction_info["correction_values"] = {
                "suggested_margin_x": 2.5 * abs(deviation.x_shift),
                "suggested_margin_y": 2.5 * abs(deviation.y_shift),
                "suggested_margin_z": 2.5 * abs(deviation.z_shift)
            }
        
        # Cập nhật chiến lược đã áp dụng
        corrected_deviation.applied_correction = strategy
        
        # Cập nhật thông tin điều chỉnh
        correction_info["corrected_deviation"] = corrected_deviation.to_dict()
        
        return correction_info
    
    def set_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Cập nhật ngưỡng cho các chiến lược điều chỉnh
        
        Parameters
        ----------
        new_thresholds : Dict[str, float]
            Ngưỡng mới
        """
        self.thresholds.update(new_thresholds)
    
    def get_thresholds(self) -> Dict[str, float]:
        """
        Lấy ngưỡng hiện tại
        
        Returns
        -------
        Dict[str, float]
            Ngưỡng hiện tại
        """
        return self.thresholds
