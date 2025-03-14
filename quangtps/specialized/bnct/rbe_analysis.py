#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho phân tích hiệu quả sinh học tương đối (RBE) trong xạ trị bắt neutron boron (BNCT).

Module này cung cấp các lớp và phương thức để tính toán và phân tích
hiệu quả sinh học tương đối (RBE) của các thành phần liều khác nhau trong BNCT,
bao gồm liều boron, liều gamma, liều neutron nhanh và liều neutron nhiệt.
"""

import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class DoseComponent(str, Enum):
    """Enum đại diện cho các thành phần liều trong BNCT."""
    BORON = "BORON"                # Liều từ phản ứng bắt neutron boron
    GAMMA = "GAMMA"                # Liều gamma
    FAST_NEUTRON = "FAST_NEUTRON"  # Liều từ neutron nhanh
    THERMAL_NEUTRON = "THERMAL_NEUTRON"  # Liều từ neutron nhiệt

@dataclass
class RBEFactors:
    """Lớp chứa các hệ số RBE cho các thành phần liều khác nhau."""
    boron: float = 3.8       # Mặc định cho BPA
    gamma: float = 1.0
    fast_neutron: float = 3.2
    thermal_neutron: float = 2.5
    description: str = ""

# Dữ liệu RBE mặc định cho các hợp chất boron phổ biến
DEFAULT_RBE_FACTORS = {
    "BPA": RBEFactors(
        boron=3.8,
        gamma=1.0,
        fast_neutron=3.2,
        thermal_neutron=2.5,
        description="Hệ số RBE cho hợp chất BPA (Boronophenylalanine)"
    ),
    "BSH": RBEFactors(
        boron=2.5,
        gamma=1.0,
        fast_neutron=3.2,
        thermal_neutron=2.5,
        description="Hệ số RBE cho hợp chất BSH (Sodium borocaptate)"
    ),
    "CUSTOM": RBEFactors(
        boron=3.0,
        gamma=1.0,
        fast_neutron=3.2,
        thermal_neutron=2.5,
        description="Hệ số RBE cho hợp chất tùy chỉnh"
    )
}

class RBEModel:
    """Lớp cơ sở cho các mô hình RBE trong BNCT."""
    
    def __init__(self, compound_name: str = "BPA", 
                 rbe_factors: Optional[RBEFactors] = None):
        """
        Khởi tạo mô hình RBE.
        
        Parameters
        ----------
        compound_name : str
            Tên hợp chất boron
        rbe_factors : RBEFactors, optional
            Các hệ số RBE, nếu None sẽ sử dụng giá trị mặc định
        """
        self.compound_name = compound_name
        
        if rbe_factors is None:
            if compound_name in DEFAULT_RBE_FACTORS:
                self.rbe_factors = DEFAULT_RBE_FACTORS[compound_name]
            else:
                # Sử dụng giá trị mặc định cho hợp chất tùy chỉnh
                self.rbe_factors = DEFAULT_RBE_FACTORS["CUSTOM"]
        else:
            self.rbe_factors = rbe_factors
    
    def calculate_weighted_dose(self, dose_components: Dict[str, float]) -> Dict[str, float]:
        """
        Tính toán liều sinh học có trọng số từ các thành phần liều vật lý.
        
        Parameters
        ----------
        dose_components : Dict[str, float]
            Từ điển chứa các thành phần liều vật lý (Gy)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa các thành phần liều sinh học có trọng số (Gy-Eq)
        """
        weighted_doses = {}
        
        # Tính liều có trọng số cho từng thành phần
        if "boron_dose" in dose_components:
            weighted_doses["weighted_boron_dose"] = dose_components["boron_dose"] * self.rbe_factors.boron
        
        if "gamma_dose" in dose_components:
            weighted_doses["weighted_gamma_dose"] = dose_components["gamma_dose"] * self.rbe_factors.gamma
        
        if "fast_neutron_dose" in dose_components:
            weighted_doses["weighted_fast_neutron_dose"] = dose_components["fast_neutron_dose"] * self.rbe_factors.fast_neutron
        
        if "thermal_neutron_dose" in dose_components:
            weighted_doses["weighted_thermal_neutron_dose"] = dose_components["thermal_neutron_dose"] * self.rbe_factors.thermal_neutron
        
        # Tính tổng liều sinh học có trọng số
        weighted_doses["total_biologically_weighted_dose"] = sum(weighted_doses.values())
        
        return weighted_doses
    
    def calculate_therapeutic_ratio(self, tumor_dose_components: Dict[str, float],
                                  normal_dose_components: Dict[str, float]) -> float:
        """
        Tính toán tỷ lệ điều trị (liều u / liều mô lành).
        
        Parameters
        ----------
        tumor_dose_components : Dict[str, float]
            Từ điển chứa các thành phần liều vật lý cho mô u (Gy)
        normal_dose_components : Dict[str, float]
            Từ điển chứa các thành phần liều vật lý cho mô lành (Gy)
            
        Returns
        -------
        float
            Tỷ lệ điều trị (liều sinh học có trọng số cho u / liều sinh học có trọng số cho mô lành)
        """
        # Tính liều sinh học có trọng số
        tumor_weighted_doses = self.calculate_weighted_dose(tumor_dose_components)
        normal_weighted_doses = self.calculate_weighted_dose(normal_dose_components)
        
        # Tính tỷ lệ điều trị
        therapeutic_ratio = tumor_weighted_doses["total_biologically_weighted_dose"] / \
                          normal_weighted_doses["total_biologically_weighted_dose"]
        
        return therapeutic_ratio

class MicrodosimetricModel(RBEModel):
    """Mô hình RBE dựa trên vi liều lượng học."""
    
    def __init__(self, compound_name: str = "BPA", 
                 rbe_factors: Optional[RBEFactors] = None,
                 alpha: float = 0.3, beta: float = 0.03):
        """
        Khởi tạo mô hình RBE dựa trên vi liều lượng học.
        
        Parameters
        ----------
        compound_name : str
            Tên hợp chất boron
        rbe_factors : RBEFactors, optional
            Các hệ số RBE
        alpha : float
            Hệ số alpha trong mô hình tuyến tính-bậc hai (Gy^-1)
        beta : float
            Hệ số beta trong mô hình tuyến tính-bậc hai (Gy^-2)
        """
        super().__init__(compound_name, rbe_factors)
        self.alpha = alpha
        self.beta = beta
    
    def calculate_cell_survival(self, dose_components: Dict[str, float]) -> float:
        """
        Tính toán tỷ lệ sống sót của tế bào dựa trên mô hình tuyến tính-bậc hai.
        
        Parameters
        ----------
        dose_components : Dict[str, float]
            Từ điển chứa các thành phần liều vật lý (Gy)
            
        Returns
        -------
        float
            Tỷ lệ sống sót của tế bào
        """
        # Tính liều sinh học có trọng số
        weighted_doses = self.calculate_weighted_dose(dose_components)
        total_weighted_dose = weighted_doses["total_biologically_weighted_dose"]
        
        # Áp dụng mô hình tuyến tính-bậc hai
        survival_fraction = np.exp(-(self.alpha * total_weighted_dose + self.beta * total_weighted_dose**2))
        
        return survival_fraction

class CompoundBasedRBEModel(RBEModel):
    """Mô hình RBE dựa trên đặc tính của hợp chất boron."""
    
    def __init__(self, compound_name: str = "BPA", 
                 rbe_factors: Optional[RBEFactors] = None,
                 cbr_factor: float = None):
        """
        Khởi tạo mô hình RBE dựa trên đặc tính của hợp chất boron.
        
        Parameters
        ----------
        compound_name : str
            Tên hợp chất boron
        rbe_factors : RBEFactors, optional
            Các hệ số RBE
        cbr_factor : float, optional
            Hệ số CBR (Compound Biological Effectiveness), nếu None sẽ sử dụng giá trị mặc định
        """
        super().__init__(compound_name, rbe_factors)
        
        # Thiết lập CBR nếu được cung cấp
        if cbr_factor is not None:
            self.rbe_factors.boron = cbr_factor
    
    def adjust_for_oxygenation(self, dose_components: Dict[str, float], 
                             oxygen_enhancement_ratio: float = 1.0) -> Dict[str, float]:
        """
        Điều chỉnh liều dựa trên tỷ lệ tăng cường oxy.
        
        Parameters
        ----------
        dose_components : Dict[str, float]
            Từ điển chứa các thành phần liều vật lý (Gy)
        oxygen_enhancement_ratio : float
            Tỷ lệ tăng cường oxy (OER)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa các thành phần liều đã điều chỉnh (Gy)
        """
        adjusted_components = {}
        
        # Điều chỉnh liều dựa trên OER
        for component, dose in dose_components.items():
            if component == "boron_dose":
                # Liều boron ít bị ảnh hưởng bởi oxy
                adjusted_components[component] = dose
            else:
                # Các thành phần khác bị ảnh hưởng bởi oxy
                adjusted_components[component] = dose / oxygen_enhancement_ratio
        
        return adjusted_components

def plot_rbe_comparison(dose_values: List[float], rbe_models: Dict[str, RBEModel]) -> plt.Figure:
    """
    Vẽ đồ thị so sánh các mô hình RBE khác nhau.
    
    Parameters
    ----------
    dose_values : List[float]
        Danh sách các giá trị liều vật lý (Gy)
    rbe_models : Dict[str, RBEModel]
        Từ điển chứa các mô hình RBE cần so sánh
        
    Returns
    -------
    plt.Figure
        Đối tượng Figure chứa đồ thị so sánh
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, model in rbe_models.items():
        weighted_doses = []
        
        for dose in dose_values:
            # Giả định phân bố liều đơn giản
            dose_components = {
                "boron_dose": 0.7 * dose,
                "gamma_dose": 0.1 * dose,
                "fast_neutron_dose": 0.1 * dose,
                "thermal_neutron_dose": 0.1 * dose
            }
            
            weighted_dose = model.calculate_weighted_dose(dose_components)
            weighted_doses.append(weighted_dose["total_biologically_weighted_dose"])
        
        ax.plot(dose_values, weighted_doses, label=model_name, linewidth=2)
    
    ax.set_xlabel('Liều vật lý (Gy)')
    ax.set_ylabel('Liều sinh học có trọng số (Gy-Eq)')
    ax.set_title('So sánh các mô hình RBE')
    ax.legend()
    ax.grid(True)
    
    return fig