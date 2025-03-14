#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho phân tích ảnh hưởng của oxy trong xạ trị bắt neutron boron (BNCT).

Module này cung cấp các lớp và phương thức để mô hình hóa và phân tích
ảnh hưởng của nồng độ oxy đến hiệu quả sinh học của các thành phần liều
trong BNCT, đặc biệt quan trọng cho các khối u thiếu oxy (hypoxic tumors).
"""

import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class OxygenationStatus(str, Enum):
    """Enum đại diện cho các trạng thái oxy hóa của mô."""
    NORMOXIC = "NORMOXIC"          # Mô có nồng độ oxy bình thường
    HYPOXIC = "HYPOXIC"            # Mô thiếu oxy
    ANOXIC = "ANOXIC"              # Mô không có oxy
    REOXYGENATED = "REOXYGENATED"  # Mô được tái oxy hóa

@dataclass
class OxygenEffectParameters:
    """Lớp chứa các tham số ảnh hưởng của oxy."""
    oer_max: float = 3.0           # Tỷ lệ tăng cường oxy tối đa
    k_value: float = 3.0           # Hằng số mm Hg cho mô hình OER
    hypoxic_threshold: float = 10.0  # Ngưỡng thiếu oxy (mm Hg)
    normoxic_level: float = 40.0   # Nồng độ oxy bình thường (mm Hg)
    description: str = ""

# Dữ liệu mặc định cho các thành phần liều khác nhau
DEFAULT_OER_PARAMETERS = {
    "boron": OxygenEffectParameters(
        oer_max=1.6,
        k_value=4.0,
        hypoxic_threshold=10.0,
        normoxic_level=40.0,
        description="Tham số OER cho liều boron"
    ),
    "gamma": OxygenEffectParameters(
        oer_max=3.0,
        k_value=3.0,
        hypoxic_threshold=10.0,
        normoxic_level=40.0,
        description="Tham số OER cho liều gamma"
    ),
    "fast_neutron": OxygenEffectParameters(
        oer_max=2.5,
        k_value=3.5,
        hypoxic_threshold=10.0,
        normoxic_level=40.0,
        description="Tham số OER cho liều neutron nhanh"
    ),
    "thermal_neutron": OxygenEffectParameters(
        oer_max=2.0,
        k_value=3.5,
        hypoxic_threshold=10.0,
        normoxic_level=40.0,
        description="Tham số OER cho liều neutron nhiệt"
    )
}

class OxygenEffectModel:
    """Lớp cơ sở cho các mô hình ảnh hưởng của oxy."""
    
    def __init__(self, parameters: Optional[Dict[str, OxygenEffectParameters]] = None):
        """
        Khởi tạo mô hình ảnh hưởng của oxy.
        
        Parameters
        ----------
        parameters : Dict[str, OxygenEffectParameters], optional
            Từ điển chứa các tham số OER cho các thành phần liều khác nhau
        """
        self.parameters = parameters if parameters is not None else DEFAULT_OER_PARAMETERS
    
    def calculate_oer(self, oxygen_concentration: float, dose_component: str = "gamma") -> float:
        """
        Tính toán tỷ lệ tăng cường oxy (OER) cho một nồng độ oxy cụ thể.
        
        Parameters
        ----------
        oxygen_concentration : float
            Nồng độ oxy (mm Hg)
        dose_component : str, optional
            Thành phần liều cần tính OER
            
        Returns
        -------
        float
            Tỷ lệ tăng cường oxy (OER)
        """
        if dose_component not in self.parameters:
            dose_component = "gamma"  # Sử dụng tham số mặc định nếu không tìm thấy
        
        params = self.parameters[dose_component]
        
        # Mô hình Alper-Howard-Flanders
        oer = (params.oer_max * oxygen_concentration + params.k_value) / (oxygen_concentration + params.k_value)
        
        return oer
    
    def adjust_dose(self, physical_dose: float, oxygen_concentration: float, 
                   dose_component: str = "gamma") -> float:
        """
        Điều chỉnh liều vật lý dựa trên nồng độ oxy.
        
        Parameters
        ----------
        physical_dose : float
            Liều vật lý (Gy)
        oxygen_concentration : float
            Nồng độ oxy (mm Hg)
        dose_component : str, optional
            Thành phần liều cần điều chỉnh
            
        Returns
        -------
        float
            Liều hiệu dụng sau khi điều chỉnh theo oxy (Gy)
        """
        oer = self.calculate_oer(oxygen_concentration, dose_component)
        effective_dose = physical_dose / oer
        
        return effective_dose
    
    def adjust_dose_components(self, dose_components: Dict[str, float], 
                             oxygen_concentration: float) -> Dict[str, float]:
        """
        Điều chỉnh các thành phần liều dựa trên nồng độ oxy.
        
        Parameters
        ----------
        dose_components : Dict[str, float]
            Từ điển chứa các thành phần liều vật lý (Gy)
        oxygen_concentration : float
            Nồng độ oxy (mm Hg)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa các thành phần liều đã điều chỉnh (Gy)
        """
        adjusted_doses = {}
        
        # Ánh xạ tên thành phần liều trong dose_components sang tên trong parameters
        component_mapping = {
            "boron_dose": "boron",
            "gamma_dose": "gamma",
            "fast_neutron_dose": "fast_neutron",
            "thermal_neutron_dose": "thermal_neutron"
        }
        
        for component, dose in dose_components.items():
            param_key = component_mapping.get(component, component)
            if param_key in self.parameters:
                adjusted_doses[component] = self.adjust_dose(dose, oxygen_concentration, param_key)
            else:
                adjusted_doses[component] = dose  # Giữ nguyên nếu không có tham số tương ứng
        
        return adjusted_doses

class OxygenDistributionModel:
    """Lớp mô hình hóa phân bố oxy trong mô."""
    
    def __init__(self, normoxic_level: float = 40.0, hypoxic_core_level: float = 2.5):
        """
        Khởi tạo mô hình phân bố oxy.
        
        Parameters
        ----------
        normoxic_level : float, optional
            Nồng độ oxy trong mô bình thường (mm Hg)
        hypoxic_core_level : float, optional
            Nồng độ oxy trong vùng lõi thiếu oxy (mm Hg)
        """
        self.normoxic_level = normoxic_level
        self.hypoxic_core_level = hypoxic_core_level
    
    def calculate_oxygen_profile(self, distances: np.ndarray, 
                               vessel_positions: List[float]) -> np.ndarray:
        """
        Tính toán phân bố oxy dựa trên khoảng cách từ mạch máu.
        
        Parameters
        ----------
        distances : np.ndarray
            Mảng các vị trí cần tính nồng độ oxy
        vessel_positions : List[float]
            Danh sách vị trí của các mạch máu
            
        Returns
        -------
        np.ndarray
            Mảng nồng độ oxy tại các vị trí tương ứng (mm Hg)
        """
        # Khởi tạo mảng nồng độ oxy với giá trị thấp
        oxygen_profile = np.ones_like(distances) * self.hypoxic_core_level
        
        # Tính toán nồng độ oxy dựa trên khoảng cách từ mạch máu gần nhất
        for pos in vessel_positions:
            # Mô hình suy giảm theo hàm mũ từ mạch máu
            diffusion_distance = 150.0  # μm
            vessel_contribution = self.normoxic_level * np.exp(-np.abs(distances - pos) * 1000 / diffusion_distance)
            
            # Cập nhật nồng độ oxy (lấy giá trị lớn nhất)
            oxygen_profile = np.maximum(oxygen_profile, vessel_contribution)
        
        return oxygen_profile
    
    def plot_oxygen_distribution(self, distances: np.ndarray, 
                               oxygen_profile: np.ndarray,
                               vessel_positions: List[float] = None) -> plt.Figure:
        """
        Vẽ đồ thị phân bố oxy.
        
        Parameters
        ----------
        distances : np.ndarray
            Mảng các vị trí (mm)
        oxygen_profile : np.ndarray
            Mảng nồng độ oxy tại các vị trí tương ứng (mm Hg)
        vessel_positions : List[float], optional
            Danh sách vị trí của các mạch máu
            
        Returns
        -------
        plt.Figure
            Đối tượng Figure chứa đồ thị phân bố oxy
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ đồ thị phân bố oxy
        ax.plot(distances, oxygen_profile, 'b-', linewidth=2)
        
        # Đánh dấu vị trí mạch máu
        if vessel_positions is not None:
            for pos in vessel_positions:
                ax.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
                ax.annotate('Blood vessel', xy=(pos, ax.get_ylim()[1] * 0.9),
                           xytext=(pos, ax.get_ylim()[1] * 0.95),
                           ha='center', va='center',
                           bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Đánh dấu vùng thiếu oxy
        ax.axhline(y=10.0, color='g', linestyle=':', alpha=0.7)
        ax.annotate('Hypoxic threshold', xy=(ax.get_xlim()[1] * 0.95, 10.0),
                   xytext=(ax.get_xlim()[1] * 0.95, 12.0),
                   ha='right', va='bottom',
                   bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        ax.set_xlabel('Vị trí (mm)')
        ax.set_ylabel('Nồng độ oxy (mm Hg)')
        ax.set_title('Phân bố oxy trong mô')
        ax.grid(True)
        
        return fig

class FractionatedOxygenModel(OxygenEffectModel):
    """Mô hình ảnh hưởng của oxy trong xạ trị phân liều."""
    
    def __init__(self, parameters: Optional[Dict[str, OxygenEffectParameters]] = None,
                 reoxygenation_factor: float = 1.5):
        """
        Khởi tạo mô hình ảnh hưởng của oxy trong xạ trị phân liều.
        
        Parameters
        ----------
        parameters : Dict[str, OxygenEffectParameters], optional
            Từ điển chứa các tham số OER cho các thành phần liều khác nhau
        reoxygenation_factor : float, optional
            Hệ số tái oxy hóa giữa các phân liều
        """
        super().__init__(parameters)
        self.reoxygenation_factor = reoxygenation_factor
    
    def calculate_fractionated_effect(self, initial_oxygen_level: float, 
                                     dose_per_fraction: Dict[str, float],
                                     num_fractions: int) -> Dict[str, np.ndarray]:
        """
        Tính toán hiệu ứng sinh học trong xạ trị phân liều có tính đến tái oxy hóa.
        
        Parameters
        ----------
        initial_oxygen_level : float
            Nồng độ oxy ban đầu (mm Hg)
        dose_per_fraction : Dict[str, float]
            Từ điển chứa các thành phần liều cho mỗi phân liều (Gy)
        num_fractions : int
            Số phân liều
            
        Returns
        -------
        Dict[str, np.ndarray]
            Từ điển chứa hiệu ứng sinh học tích lũy theo số phân liều
        """
        # Khởi tạo kết quả
        result = {
            "effective_doses": np.zeros(num_fractions),
            "oxygen_levels": np.zeros(num_fractions),
            "oer_values": np.zeros(num_fractions)
        }
        
        # Nồng độ oxy hiện tại
        current_oxygen = initial_oxygen_level
        
        # Tính toán hiệu ứng cho từng phân liều
        for i in range(num_fractions):
            # Lưu nồng độ oxy hiện tại
            result["oxygen_levels"][i] = current_oxygen
            
            # Điều chỉnh liều theo nồng độ oxy
            adjusted_doses = self.adjust_dose_components(dose_per_fraction, current_oxygen)
            
            # Tính tổng liều hiệu dụng
            effective_dose = sum(adjusted_doses.values())
            result["effective_doses"][i] = effective_dose
            
            # Tính OER trung bình (đơn giản hóa)
            total_physical_dose = sum(dose_per_fraction.values())
            result["oer_values"][i] = total_physical_dose / effective_dose if effective_dose > 0 else 1.0
            
            # Mô phỏng tái oxy hóa giữa các phân liều
            # Giả định: nồng độ oxy tăng theo hệ số reoxygenation_factor nhưng không vượt quá mức bình thường
            current_oxygen = min(current_oxygen * self.reoxygenation_factor, self.parameters["gamma"].normoxic_level)
        
        return result
    
    def plot_fractionation_effect(self, fractionation_results: Dict[str, np.ndarray]) -> plt.Figure:
        """
        Vẽ đồ thị hiệu ứng của phân liều và tái oxy hóa.
        
        Parameters
        ----------
        fractionation_results : Dict[str, np.ndarray]
            Kết quả từ phương thức calculate_fractionated_effect
            
        Returns
        -------
        plt.Figure
            Đối tượng Figure chứa đồ thị hiệu ứng phân liều
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Số phân liều
        fractions = np.arange(1, len(fractionation_results["oxygen_levels"]) + 1)
        
        # Vẽ đồ thị nồng độ oxy
        axes[0].plot(fractions, fractionation_results["oxygen_levels"], 'b-o', linewidth=2)
        axes[0].set_ylabel('Nồng độ oxy (mm Hg)')
        axes[0].set_title('Nồng độ oxy theo phân liều')
        axes[0].grid(True)
        
        # Vẽ đường ngưỡng thiếu oxy
        axes[0].axhline(y=10.0, color='r', linestyle='--', alpha=0.7)
        axes[0].annotate('Hypoxic threshold', xy=(fractions[-1] * 0.95, 10.0),
                       xytext=(fractions[-1] * 0.95, 12.0),
                       ha='right', va='bottom',
                       bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Vẽ đồ thị OER
        axes[1].plot(fractions, fractionation_results["oer_values"], 'g-o', linewidth=2)
        axes[1].set_ylabel('OER')
        axes[1].set_title('Tỷ lệ tăng cường oxy (OER) theo phân liều')
        axes[1].grid(True)
        
        # Vẽ đồ thị liều hiệu dụng
        axes[2].plot(fractions, fractionation_results["effective_doses"], 'r-o', linewidth=2)
        axes[2].set_xlabel('Phân liều')
        axes[2].set_ylabel('Liều hiệu dụng (Gy)')
        axes[2].set_title('Liều hiệu dụng theo phân liều')
        axes[2].grid(True)
        
        # Vẽ đồ thị liều tích lũy
        cumulative_dose = np.cumsum(fractionation_results["effective_doses"])
        axes[2].plot(fractions, cumulative_dose, 'r--', linewidth=1, alpha=0.7, label='Liều tích lũy')
        axes[2].legend()
        
        plt.tight_layout()
        
        return fig


def plot_oer_curves(oxygen_concentrations: np.ndarray = None) -> plt.Figure:
    """
    Vẽ đồ thị đường cong OER cho các thành phần liều khác nhau.
    
    Parameters
    ----------
    oxygen_concentrations : np.ndarray, optional
        Mảng các giá trị nồng độ oxy (mm Hg)
        
    Returns
    -------
    plt.Figure
        Đối tượng Figure chứa đồ thị đường cong OER
    """
    if oxygen_concentrations is None:
        oxygen_concentrations = np.linspace(0, 100, 100)  # mm Hg
    
    # Tạo mô hình OER
    model = OxygenEffectModel()
    
    # Tính toán OER cho từng thành phần liều
    oer_values = {
        "Boron": [model.calculate_oer(o2, "boron") for o2 in oxygen_concentrations],
        "Gamma": [model.calculate_oer(o2, "gamma") for o2 in oxygen_concentrations],
        "Fast Neutron": [model.calculate_oer(o2, "fast_neutron") for o2 in oxygen_concentrations],
        "Thermal Neutron": [model.calculate_oer(o2, "thermal_neutron") for o2 in oxygen_concentrations]
    }
    
    # Vẽ đồ thị
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'green', 'red', 'cyan']
    for i, (component, values) in enumerate(oer_values.items()):
        ax.plot(oxygen_concentrations, values, color=colors[i], label=component, linewidth=2)
    
    # Đánh dấu ngưỡng thiếu oxy
    ax.axvline(x=10.0, color='gray', linestyle='--', alpha=0.7)
    ax.annotate('Hypoxic threshold', xy=(10.0, ax.get_ylim()[1] * 0.9),
               xytext=(15.0, ax.get_ylim()[1] * 0.9),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    ax.set_xlabel('Nồng độ oxy (mm Hg)')
    ax.set_ylabel('OER')
    ax.set_title('Đường cong OER cho các thành phần liều khác nhau')
    ax.legend()
    ax.grid(True)
    
    return fig


def compare_oxygen_effect_on_dose(physical_doses: Dict[str, float],
                                oxygen_levels: List[float],
                                labels: List[str] = None) -> plt.Figure:
    """
    So sánh ảnh hưởng của các mức oxy khác nhau đến liều hiệu dụng.
    
    Parameters
    ----------
    physical_doses : Dict[str, float]
        Từ điển chứa các thành phần liều vật lý (Gy)
    oxygen_levels : List[float]
        Danh sách các mức nồng độ oxy cần so sánh (mm Hg)
    labels : List[str], optional
        Nhãn cho các mức nồng độ oxy
        
    Returns
    -------
    plt.Figure
        Đối tượng Figure chứa đồ thị so sánh
    """
    # Tạo mô hình OER
    model = OxygenEffectModel()
    
    # Tạo nhãn mặc định nếu không được cung cấp
    if labels is None:
        labels = [f"{o2} mm Hg" for o2 in oxygen_levels]
    
    # Tính toán liều hiệu dụng cho từng mức oxy
    effective_doses = []
    for o2 in oxygen_levels:
        adjusted = model.adjust_dose_components(physical_doses, o2)
        effective_doses.append(adjusted)
    
    # Chuẩn bị dữ liệu cho biểu đồ
    components = ["boron_dose", "gamma_dose", "fast_neutron_dose", "thermal_neutron_dose"]
    component_labels = ["Boron", "Gamma", "Fast Neutron", "Thermal Neutron"]
    colors = ['blue', 'green', 'red', 'cyan']
    
    # Tạo biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Biểu đồ cột cho liều vật lý
    x = np.arange(len(components))
    width = 0.8 / len(oxygen_levels)
    
    # Vẽ biểu đồ cột cho liều hiệu dụng
    for i, (doses, label) in enumerate(zip(effective_doses, labels)):
        values = [doses.get(comp, 0.0) for comp in components]
        offset = width * i - width * (len(oxygen_levels) - 1) / 2
        axes[0].bar(x + offset, values, width, label=label, color=colors, alpha=0.7 + 0.3 * (i / len(oxygen_levels)))
    
    axes[0].set_ylabel('Liều hiệu dụng (Gy)')
    axes[0].set_title('Liều hiệu dụng theo thành phần và mức oxy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(component_labels)
    axes[0].legend()
    
    # Biểu đồ tròn cho tỷ lệ giảm liều
    total_physical = sum(physical_doses.values())
    total_effectives = [sum(doses.values()) for doses in effective_doses]
    dose_reduction = [(total_physical - total_eff) / total_physical * 100 for total_eff in total_effectives]
    
    axes[1].pie(dose_reduction, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=plt.cm.viridis(np.linspace(0, 1, len(oxygen_levels))))
    axes[1].set_title('Tỷ lệ giảm liều do thiếu oxy (%)')
    
    plt.tight_layout()
    
    return fig