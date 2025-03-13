#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý phân đoạn xạ trị (Fractionation).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý các phương pháp
phân đoạn xạ trị, bao gồm các loại phân đoạn chuẩn và các tính toán liên quan.
"""

import math
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class FractionationType(str, Enum):
    """Enum đại diện cho các loại phân đoạn xạ trị."""
    STANDARD = "STANDARD"
    HYPOFRACTIONATED = "HYPOFRACTIONATED"
    HYPERFRACTIONATED = "HYPERFRACTIONATED"
    SRS = "SRS"
    SBRT = "SBRT"
    CUSTOM = "CUSTOM"


class Fractionation:
    """
    Lớp đại diện cho một phương pháp phân đoạn xạ trị.
    
    Lớp này chứa thông tin về tổng liều, số phân đoạn, và liều mỗi phân đoạn.
    """
    
    def __init__(self, total_dose: float, num_fractions: int, 
                 fractionation_type: FractionationType = FractionationType.STANDARD):
        """
        Khởi tạo một phương pháp phân đoạn xạ trị.
        
        Parameters
        ----------
        total_dose : float
            Tổng liều xạ trị (Gy)
        num_fractions : int
            Số phân đoạn
        fractionation_type : FractionationType, optional
            Loại phân đoạn xạ trị
        """
        self.total_dose = total_dose
        self.num_fractions = num_fractions
        self.fractionation_type = fractionation_type
        
        # Tính toán liều mỗi phân đoạn
        self.dose_per_fraction = total_dose / num_fractions if num_fractions > 0 else 0.0
        
        # Thông tin bổ sung
        self.alpha_beta_ratio = 10.0  # Tỉ lệ alpha/beta mặc định cho các khối u (Gy)
        self.metadata = {}
    
    def get_biologically_effective_dose(self, alpha_beta_ratio: Optional[float] = None) -> float:
        """
        Tính toán Liều Hiệu quả Sinh học (BED - Biologically Effective Dose).
        
        Parameters
        ----------
        alpha_beta_ratio : float, optional
            Tỉ lệ alpha/beta (Gy). Nếu không cung cấp, sẽ sử dụng tỉ lệ mặc định.
            
        Returns
        -------
        float
            Liều Hiệu quả Sinh học (Gy)
        """
        if alpha_beta_ratio is None:
            alpha_beta_ratio = self.alpha_beta_ratio
            
        bed = self.total_dose * (1 + self.dose_per_fraction / alpha_beta_ratio)
        return bed
    
    def get_equivalent_dose_in_2Gy(self, alpha_beta_ratio: Optional[float] = None) -> float:
        """
        Tính toán Liều tương đương trong phân đoạn 2Gy (EQD2).
        
        Parameters
        ----------
        alpha_beta_ratio : float, optional
            Tỉ lệ alpha/beta (Gy). Nếu không cung cấp, sẽ sử dụng tỉ lệ mặc định.
            
        Returns
        -------
        float
            Liều tương đương trong phân đoạn 2Gy (Gy)
        """
        if alpha_beta_ratio is None:
            alpha_beta_ratio = self.alpha_beta_ratio
            
        eqd2 = self.total_dose * ((self.dose_per_fraction + alpha_beta_ratio) / (2.0 + alpha_beta_ratio))
        return eqd2
    
    def get_effective_treatment_time(self, treatments_per_week: int = 5) -> float:
        """
        Tính toán thời gian điều trị hiệu quả.
        
        Parameters
        ----------
        treatments_per_week : int, optional
            Số lần điều trị mỗi tuần
            
        Returns
        -------
        float
            Thời gian điều trị hiệu quả (ngày)
        """
        weeks = math.ceil(self.num_fractions / treatments_per_week)
        days = weeks * 7  # Số ngày điều trị (bao gồm cả ngày nghỉ)
        return days
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin phân đoạn xạ trị thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin phân đoạn xạ trị
        """
        return {
            "total_dose": self.total_dose,
            "num_fractions": self.num_fractions,
            "dose_per_fraction": self.dose_per_fraction,
            "fractionation_type": self.fractionation_type.value,
            "alpha_beta_ratio": self.alpha_beta_ratio,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fractionation':
        """
        Tạo đối tượng Fractionation từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin phân đoạn xạ trị
            
        Returns
        -------
        Fractionation
            Đối tượng Fractionation
        """
        fractionation = cls(
            total_dose=data["total_dose"],
            num_fractions=data["num_fractions"],
            fractionation_type=FractionationType(data["fractionation_type"])
        )
        
        fractionation.alpha_beta_ratio = data["alpha_beta_ratio"]
        fractionation.metadata = data["metadata"]
        
        return fractionation


class FractionationScheme:
    """
    Lớp đại diện cho một phương án phân đoạn xạ trị.
    
    Lớp này cung cấp các phương án phân đoạn xạ trị chuẩn cho các vị trí khối u
    khác nhau và các phương pháp điều trị khác nhau.
    """
    
    # Danh sách các phương án phân đoạn xạ trị chuẩn
    STANDARD_SCHEMES = {
        "BREAST_STANDARD": {"total_dose": 50.0, "num_fractions": 25, "type": FractionationType.STANDARD},
        "BREAST_HYPO": {"total_dose": 40.0, "num_fractions": 15, "type": FractionationType.HYPOFRACTIONATED},
        "PROSTATE_STANDARD": {"total_dose": 78.0, "num_fractions": 39, "type": FractionationType.STANDARD},
        "PROSTATE_HYPO": {"total_dose": 60.0, "num_fractions": 20, "type": FractionationType.HYPOFRACTIONATED},
        "PROSTATE_EXTREME_HYPO": {"total_dose": 36.25, "num_fractions": 5, "type": FractionationType.SBRT},
        "LUNG_SBRT": {"total_dose": 54.0, "num_fractions": 3, "type": FractionationType.SBRT},
        "BRAIN_SRS": {"total_dose": 18.0, "num_fractions": 1, "type": FractionationType.SRS},
        "HEAD_NECK_STANDARD": {"total_dose": 70.0, "num_fractions": 35, "type": FractionationType.STANDARD},
        "HEAD_NECK_HYPER": {"total_dose": 74.4, "num_fractions": 62, "type": FractionationType.HYPERFRACTIONATED},
        "PELVIS_STANDARD": {"total_dose": 45.0, "num_fractions": 25, "type": FractionationType.STANDARD},
        "WHOLE_BRAIN": {"total_dose": 30.0, "num_fractions": 10, "type": FractionationType.STANDARD},
        "PALLIATIVE_BONE": {"total_dose": 8.0, "num_fractions": 1, "type": FractionationType.STANDARD},
        "PALLIATIVE_STANDARD": {"total_dose": 20.0, "num_fractions": 5, "type": FractionationType.STANDARD},
    }
    
    @classmethod
    def get_scheme(cls, scheme_name: str) -> Optional[Fractionation]:
        """
        Lấy phương án phân đoạn xạ trị theo tên.
        
        Parameters
        ----------
        scheme_name : str
            Tên của phương án phân đoạn xạ trị
            
        Returns
        -------
        Optional[Fractionation]
            Đối tượng Fractionation nếu tìm thấy, None nếu không tìm thấy
        """
        if scheme_name in cls.STANDARD_SCHEMES:
            scheme = cls.STANDARD_SCHEMES[scheme_name]
            return Fractionation(
                total_dose=scheme["total_dose"],
                num_fractions=scheme["num_fractions"],
                fractionation_type=scheme["type"]
            )
        else:
            logger.warning(f"Scheme '{scheme_name}' not found in standard fractionation schemes")
            return None
    
    @classmethod
    def get_scheme_names(cls) -> List[str]:
        """
        Lấy danh sách tên các phương án phân đoạn xạ trị chuẩn.
        
        Returns
        -------
        List[str]
            Danh sách tên các phương án phân đoạn xạ trị chuẩn
        """
        return list(cls.STANDARD_SCHEMES.keys())
    
    @classmethod
    def get_all_schemes(cls) -> Dict[str, Fractionation]:
        """
        Lấy tất cả các phương án phân đoạn xạ trị chuẩn.
        
        Returns
        -------
        Dict[str, Fractionation]
            Dictionary chứa tên và đối tượng Fractionation của các phương án
        """
        return {
            name: Fractionation(
                scheme["total_dose"],
                scheme["num_fractions"],
                scheme["type"]
            ) for name, scheme in cls.STANDARD_SCHEMES.items()
        }