#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho phân tích và tính toán phân bố liều theo độ sâu trong xạ trị bắt neutron boron (BNCT).

Module này cung cấp các lớp và phương thức để mô phỏng và tính toán
phân bố liều theo độ sâu cho các thành phần liều khác nhau trong BNCT,
bao gồm liều boron, liều gamma, liều neutron nhanh và liều neutron nhiệt.
"""

import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

class TissueType(str, Enum):
    """Enum đại diện cho các loại mô trong phân tích độ sâu."""
    BRAIN = "BRAIN"              # Mô não
    SKIN = "SKIN"                # Da
    BONE = "BONE"                # Xương
    MUSCLE = "MUSCLE"            # Cơ
    LIVER = "LIVER"              # Gan
    LUNG = "LUNG"                # Phổi
    CUSTOM = "CUSTOM"            # Mô tùy chỉnh

@dataclass
class TissueProperties:
    """Lớp chứa các thuộc tính vật lý của mô."""
    name: str
    density: float                # g/cm³
    water_content: float          # % khối lượng
    effective_z: float            # Số nguyên tử hiệu dụng
    neutron_attenuation: Dict[str, float]  # Hệ số suy giảm cho các nhóm neutron (cm⁻¹)
    gamma_attenuation: float      # Hệ số suy giảm gamma (cm⁻¹)
    description: str = ""

# Dữ liệu mặc định cho các loại mô phổ biến
DEFAULT_TISSUE_PROPERTIES = {
    TissueType.BRAIN: TissueProperties(
        name="Brain",
        density=1.04,
        water_content=0.75,
        effective_z=7.5,
        neutron_attenuation={
            "thermal": 0.12,    # cm⁻¹
            "epithermal": 0.03, # cm⁻¹
            "fast": 0.05       # cm⁻¹
        },
        gamma_attenuation=0.06,  # cm⁻¹
        description="Mô não người trưởng thành"
    ),
    TissueType.SKIN: TissueProperties(
        name="Skin",
        density=1.1,
        water_content=0.7,
        effective_z=7.8,
        neutron_attenuation={
            "thermal": 0.15,    # cm⁻¹
            "epithermal": 0.04, # cm⁻¹
            "fast": 0.06       # cm⁻¹
        },
        gamma_attenuation=0.07,  # cm⁻¹
        description="Da người"
    ),
    TissueType.BONE: TissueProperties(
        name="Bone",
        density=1.85,
        water_content=0.15,
        effective_z=13.8,
        neutron_attenuation={
            "thermal": 0.25,    # cm⁻¹
            "epithermal": 0.08, # cm⁻¹
            "fast": 0.1        # cm⁻¹
        },
        gamma_attenuation=0.12,  # cm⁻¹
        description="Xương người"
    ),
    TissueType.MUSCLE: TissueProperties(
        name="Muscle",
        density=1.05,
        water_content=0.75,
        effective_z=7.6,
        neutron_attenuation={
            "thermal": 0.13,    # cm⁻¹
            "epithermal": 0.035, # cm⁻¹
            "fast": 0.055      # cm⁻¹
        },
        gamma_attenuation=0.065,  # cm⁻¹
        description="Cơ người"
    ),
}

class DepthDoseCalculator:
    """Lớp tính toán phân bố liều theo độ sâu trong BNCT."""
    
    def __init__(self, tissue_type: TissueType = TissueType.BRAIN,
                 properties: Optional[TissueProperties] = None):
        """
        Khởi tạo bộ tính toán phân bố liều theo độ sâu.
        
        Parameters
        ----------
        tissue_type : TissueType
            Loại mô
        properties : TissueProperties, optional
            Thuộc tính của mô, nếu None sẽ sử dụng giá trị mặc định
        """
        self.tissue_type = tissue_type
        
        if properties is None:
            if tissue_type in DEFAULT_TISSUE_PROPERTIES:
                self.properties = DEFAULT_TISSUE_PROPERTIES[tissue_type]
            else:
                # Giá trị mặc định cho mô tùy chỉnh
                self.properties = TissueProperties(
                    name="Custom tissue",
                    density=1.0,
                    water_content=0.7,
                    effective_z=7.5,
                    neutron_attenuation={
                        "thermal": 0.1,    # cm⁻¹
                        "epithermal": 0.03, # cm⁻¹
                        "fast": 0.05       # cm⁻¹
                    },
                    gamma_attenuation=0.06,  # cm⁻¹
                    description="Mô tùy chỉnh"
                )
        else:
            self.properties = properties
    
    def calculate_thermal_flux_profile(self, depths: np.ndarray, 
                                     surface_thermal_flux: float,
                                     surface_epithermal_flux: float) -> np.ndarray:
        """
        Tính toán phân bố thông lượng neutron nhiệt theo độ sâu.
        
        Parameters
        ----------
        depths : np.ndarray
            Mảng các giá trị độ sâu (cm)
        surface_thermal_flux : float
            Thông lượng neutron nhiệt tại bề mặt (n/cm²/s)
        surface_epithermal_flux : float
            Thông lượng neutron trên nhiệt tại bề mặt (n/cm²/s)
            
        Returns
        -------
        np.ndarray
            Mảng thông lượng neutron nhiệt theo độ sâu (n/cm²/s)
        """
        # Hệ số suy giảm
        thermal_attenuation = self.properties.neutron_attenuation["thermal"]
        epithermal_attenuation = self.properties.neutron_attenuation["epithermal"]
        
        # Thông lượng neutron nhiệt ban đầu suy giảm theo độ sâu
        direct_thermal = surface_thermal_flux * np.exp(-thermal_attenuation * depths)
        
        # Neutron trên nhiệt chuyển thành neutron nhiệt (thermalization)
        # Mô hình đơn giản: đỉnh ở độ sâu khoảng 2-3 cm cho neutron trên nhiệt
        thermalization_peak = 2.5  # cm
        thermalization_width = 2.0  # cm
        thermalization_factor = 0.7  # Hiệu suất chuyển đổi
        
        # Hàm Gaussian mô tả quá trình thermalization
        thermalized = surface_epithermal_flux * thermalization_factor * \
                     np.exp(-(depths - thermalization_peak)**2 / (2 * thermalization_width**2)) * \
                     np.exp(-epithermal_attenuation * depths)
        
        # Tổng thông lượng neutron nhiệt
        total_thermal_flux = direct_thermal + thermalized
        
        return total_thermal_flux
    
    def calculate_epithermal_flux_profile(self, depths: np.ndarray, 
                                        surface_epithermal_flux: float) -> np.ndarray:
        """
        Tính toán phân bố thông lượng neutron trên nhiệt theo độ sâu.
        
        Parameters
        ----------
        depths : np.ndarray
            Mảng các giá trị độ sâu (cm)
        surface_epithermal_flux : float
            Thông lượng neutron trên nhiệt tại bề mặt (n/cm²/s)
            
        Returns
        -------
        np.ndarray
            Mảng thông lượng neutron trên nhiệt theo độ sâu (n/cm²/s)
        """
        # Hệ số suy giảm
        epithermal_attenuation = self.properties.neutron_attenuation["epithermal"]
        
        # Thông lượng neutron trên nhiệt suy giảm theo độ sâu
        epithermal_flux = surface_epithermal_flux * np.exp(-epithermal_attenuation * depths)
        
        return epithermal_flux
    
    def calculate_fast_flux_profile(self, depths: np.ndarray, 
                                  surface_fast_flux: float) -> np.ndarray:
        """
        Tính toán phân bố thông lượng neutron nhanh theo độ sâu.
        
        Parameters
        ----------
        depths : np.ndarray
            Mảng các giá trị độ sâu (cm)
        surface_fast_flux : float
            Thông lượng neutron nhanh tại bề mặt (n/cm²/s)
            
        Returns
        -------
        np.ndarray
            Mảng thông lượng neutron nhanh theo độ sâu (n/cm²/s)
        """
        # Hệ số suy giảm
        fast_attenuation = self.properties.neutron_attenuation["fast"]
        
        # Thông lượng neutron nhanh suy giảm theo độ sâu
        fast_flux = surface_fast_flux * np.exp(-fast_attenuation * depths)
        
        return fast_flux
    
    def calculate_gamma_dose_profile(self, depths: np.ndarray, 
                                    surface_gamma_dose_rate: float) -> np.ndarray:
        """
        Tính toán phân bố liều gamma theo độ sâu.
        
        Parameters
        ----------
        depths : np.ndarray
            Mảng các giá trị độ sâu (cm)
        surface_gamma_dose_rate : float
            Tốc độ liều gamma tại bề mặt (Gy/s)
            
        Returns
        -------
        np.ndarray
            Mảng tốc độ liều gamma theo độ sâu (Gy/s)
        """
        # Hệ số suy giảm
        gamma_attenuation = self.properties.gamma_attenuation
        
        # Liều gamma suy giảm theo độ sâu
        gamma_dose_rate = surface_gamma_dose_rate * np.exp(-gamma_attenuation * depths)
        
        return gamma_dose_rate
    
    def calculate_boron_dose_profile(self, depths: np.ndarray, 
                                    thermal_flux_profile: np.ndarray,
                                    boron_concentration_profile: np.ndarray) -> np.ndarray:
        """
        Tính toán phân bố liều boron theo độ sâu.
        
        Parameters
        ----------
        depths : np.ndarray
            Mảng các giá trị độ sâu (cm)
        thermal_flux_profile : np.ndarray
            Mảng thông lượng neutron nhiệt theo độ sâu (n/cm²/s)
        boron_concentration_profile : np.ndarray
            Mảng nồng độ boron theo độ sâu (ppm)
            
        Returns
        -------
        np.ndarray
            Mảng tốc độ liều boron theo độ sâu (Gy/s)
        """
        # Hệ số chuyển đổi từ thông lượng neutron nhiệt và nồng độ boron sang liều
        boron_dose_factor = 3.8e-11  # Gy/(ppm * n/cm²)
        
        # Tính toán liều boron
        boron_dose_rate = boron_dose_factor * boron_concentration_profile * thermal_flux_profile
        
        return boron_dose_rate
    
    def calculate_depth_dose_components(self, depths: np.ndarray,
                                       surface_thermal_flux: float,
                                       surface_epithermal_flux: float,
                                       surface_fast_flux: float,
                                       boron_concentration_surface: float,
                                       boron_concentration_depth_factor: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Tính toán các thành phần liều theo độ sâu.
        
        Parameters
        ----------
        depths : np.ndarray
            Mảng các giá trị độ sâu (cm)
        surface_thermal_flux : float
            Thông lượng neutron nhiệt tại bề mặt (n/cm²/s)
        surface_epithermal_flux : float
            Thông lượng neutron trên nhiệt tại bề mặt (n/cm²/s)
        surface_fast_flux : float
            Thông lượng neutron nhanh tại bề mặt (n/cm²/s)
        boron_concentration_surface : float
            Nồng độ boron tại bề mặt (ppm)
        boron_concentration_depth_factor : float, optional
            Hệ số suy giảm nồng độ boron theo độ sâu (cm⁻¹)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Từ điển chứa các thành phần liều theo độ sâu (Gy/s)
        """
        # Tính toán phân bố thông lượng neutron
        thermal_flux_profile = self.calculate_thermal_flux_profile(
            depths, surface_thermal_flux, surface_epithermal_flux)
        epithermal_flux_profile = self.calculate_epithermal_flux_profile(
            depths, surface_epithermal_flux)
        fast_flux_profile = self.calculate_fast_flux_profile(
            depths, surface_fast_flux)
        
        # Mô hình nồng độ boron theo độ sâu (suy giảm theo hàm mũ)
        boron_concentration_profile = boron_concentration_surface * np.exp(-boron_concentration_depth_factor * depths)
        
        # Hệ số chuyển đổi từ thông lượng neutron sang liều
        gamma_dose_factor = 2.0e-13  # Gy/(n/cm²)
        fast_neutron_dose_factor = 1.0e-12  # Gy/(n/cm²)
        thermal_neutron_dose_factor = 5.0e-13  # Gy/(n/cm²)
        
        # Tính toán các thành phần liều
        boron_dose_rate = self.calculate_boron_dose_profile(
            depths, thermal_flux_profile, boron_concentration_profile)
        
        gamma_dose_rate = gamma_dose_factor * (thermal_flux_profile + 0.5 * epithermal_flux_profile)
        fast_neutron_dose_rate = fast_neutron_dose_factor * fast_flux_profile
        thermal_neutron_dose_rate = thermal_neutron_dose_factor * thermal_flux_profile
        
        # Tổng liều
        total_dose_rate = boron_dose_rate + gamma_dose_rate + fast_neutron_dose_rate + thermal_neutron_dose_rate
        
        # Trả về từ điển chứa các thành phần liều
        return {
            "boron_dose_rate": boron_dose_rate,
            "gamma_dose_rate": gamma_dose_rate,
            "fast_neutron_dose_rate": fast_neutron_dose_rate,
            "thermal_neutron_dose_rate": thermal_neutron_dose_rate,
            "total_dose_rate": total_dose_rate
        }
    
    def calculate_total_dose(self, dose_rates: Dict[str, np.ndarray], 
                           irradiation_time: float) -> Dict[str, np.ndarray]:
        """
        Tính toán tổng liều từ tốc độ liều và thời gian chiếu xạ.
        
        Parameters
        ----------
        dose_rates : Dict[str, np.ndarray]
            Từ điển chứa các thành phần tốc độ liều (Gy/s)
        irradiation_time : float
            Thời gian chiếu xạ (giây)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Từ điển chứa các thành phần liều tích lũy (Gy)
        """
        total_doses = {}
        
        for component, dose_rate in dose_rates.items():
            total_doses[component.replace("_rate", "")] = dose_rate * irradiation_time
        
        return total_doses
    
    def plot_depth_dose(self, depths: np.ndarray, dose_components: Dict[str, np.ndarray],
                       title: str = "BNCT Depth-Dose Distribution") -> plt.Figure:
        """
        Vẽ đồ thị phân bố liều theo độ sâu.
        
        Parameters
        ----------
        depths : np.ndarray
            Mảng các giá trị độ sâu (cm)
        dose_components : Dict[str, np.ndarray]
            Từ điển chứa các thành phần liều theo độ sâu
        title : str, optional
            Tiêu đề của đồ thị
            
        Returns
        -------
        plt.Figure
            Đối tượng Figure chứa đồ thị phân bố liều
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Danh sách các thành phần liều và màu sắc tương ứng
        components = [
            ("boron_dose", "Boron Dose", "blue"),
            ("gamma_dose", "Gamma Dose", "green"),
            ("fast_neutron_dose", "Fast Neutron Dose", "red"),
            ("thermal_neutron_dose", "Thermal Neutron Dose", "cyan"),
            ("total_dose", "Total Dose", "black")
        ]
        
        # Vẽ đồ thị cho từng thành phần liều
        for key, label, color in components:
            if key in dose_components:
                ax.plot(depths, dose_components[key], color=color, label=label, linewidth=2 if key == "total_dose" else 1)
        
        ax.set_xlabel('Độ sâu (cm)')
        ax.set_ylabel('Liều (Gy)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def analyze_therapeutic_ratio(self, depths: np.ndarray, tumor_doses: Dict[str, np.ndarray],
                                normal_doses: Dict[str, np.ndarray]) -> Tuple[np.ndarray, plt.Figure]:
        """
        Phân tích tỷ lệ điều trị (liều u / liều mô lành) theo độ sâu.
        
        Parameters
        ----------
        depths : np.ndarray
            Mảng các giá trị độ sâu (cm)
        tumor_doses : Dict[str, np.ndarray]
            Từ điển chứa các thành phần liều cho mô u
        normal_doses : Dict[str, np.ndarray]
            Từ điển chứa các thành phần liều cho mô lành
            
        Returns
        -------
        Tuple[np.ndarray, plt.Figure]
            Mảng tỷ lệ điều trị theo độ sâu và đồ thị
        """
        # Tính toán tỷ lệ điều trị theo độ sâu
        therapeutic_ratio = tumor_doses["total_dose"] / normal_doses["total_dose"]
        
        # Vẽ đồ thị tỷ lệ điều trị
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(depths, therapeutic_ratio, 'b-', linewidth=2)
        ax.set_xlabel('Độ sâu (cm)')
        ax.set_ylabel('Tỷ lệ điều trị (Tumor/Normal)')
        ax.set_title('Tỷ lệ điều trị theo độ sâu')
        ax.grid(True)
        
        # Đánh dấu vị trí có tỷ lệ điều trị tối ưu
        optimal_index = np.argmax(therapeutic_ratio)
        optimal_depth = depths[optimal_index]
        optimal_ratio = therapeutic_ratio[optimal_index]
        
        ax.plot(optimal_depth, optimal_ratio, 'ro', markersize=8)
        ax.annotate(f'Optimal: {optimal_ratio:.2f} at {optimal_depth:.2f} cm',
                   xy=(optimal_depth, optimal_ratio),
                   xytext=(optimal_depth + 0.5, optimal_ratio),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        return therapeutic_ratio, fig


class MultiLayerTissueCalculator:
    """Lớp tính toán phân bố liều cho mô nhiều lớp."""
    
    def __init__(self, tissue_layers: List[Tuple[TissueType, float]]):
        """
        Khởi tạo bộ tính toán phân bố liều cho mô nhiều lớp.
        
        Parameters
        ----------
        tissue_layers : List[Tuple[TissueType, float]]
            Danh sách các lớp mô, mỗi lớp gồm loại mô và độ dày (cm)
        """
        self.tissue_layers = tissue_layers
        self.calculators = [DepthDoseCalculator(tissue_type) for tissue_type, _ in tissue_layers]
        self.layer_thicknesses = [thickness for _, thickness in tissue_layers]
        
        # Tính toán độ sâu tích lũy cho mỗi lớp
        self.cumulative_depths = np.cumsum([0] + self.layer_thicknesses)
    
    def calculate_depth_dose(self, depths: np.ndarray,
                           surface_thermal_flux: float,
                           surface_epithermal_flux: float,
                           surface_fast_flux: float,
                           boron_concentration_surface: float,
                           irradiation_time: float) -> Dict[str, np.ndarray]:
        """
        Tính toán phân bố liều theo độ sâu cho mô nhiều lớp.
        
        Parameters
        ----------
        depths : np.ndarray
            Mảng các giá trị độ sâu (cm)
        surface_thermal_flux : float
            Thông lượng neutron nhiệt tại bề mặt (n/cm²/s)
        surface_epithermal_flux : float
            Thông lượng neutron trên nhiệt tại bề mặt (n/cm²/s)
        surface_fast_flux : float
            Thông lượng neutron nhanh tại bề mặt (n/cm²/s)
        boron_concentration_surface : float
            Nồng độ boron tại bề mặt (ppm)
        irradiation_time : float
            Thời gian chiếu xạ (giây)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Từ điển chứa các thành phần liều theo độ sâu (Gy)
        """
        # Khởi tạo mảng kết quả
        result = {
            "boron_dose": np.zeros_like(depths),
            "gamma_dose": np.zeros_like(depths),
            "fast_neutron_dose": np.zeros_like(depths),
            "thermal_neutron_dose": np.zeros_like(depths),
            "total_dose": np.zeros_like(depths)
        }
        
        # Thông lượng neutron hiện tại (sẽ được cập nhật khi đi qua mỗi lớp)
        current_thermal_flux = surface_thermal_flux
        current_epithermal_flux = surface_epithermal_flux
        current_fast_flux = surface_fast_flux
        current_boron_concentration = boron_concentration_surface
        
        # Xử lý từng lớp mô
        for i, (calculator, thickness) in enumerate(zip(self.calculators, self.layer_thicknesses)):
            # Xác định các điểm nằm trong lớp hiện tại
            start_depth = self.cumulative_depths[i]
            end_depth = self.cumulative_depths[i+1]
            mask = (depths >= start_depth) & (depths < end_depth)
            
            if not np.any(mask):
                continue
            
            # Điều chỉnh độ sâu tương đối trong lớp
            relative_depths = depths[mask] - start_depth
            
            # Tính toán các thành phần liều cho lớp hiện tại
            dose_rates = calculator.calculate_depth_dose_components(
                relative_depths,
                current_thermal_flux,
                current_epithermal_flux,
                current_fast_flux,
                current_boron_concentration
            )
            
            # Tính tổng liều
            doses = calculator.calculate_total_dose(dose_rates, irradiation_time)
            
            # Cập nhật kết quả
            for key in result.keys():
                if key in doses:
                    result[key][mask] = doses[key]
            
            # Cập nhật thông lượng neutron cho lớp tiếp theo
            if i < len(self.calculators) - 1:
                # Lấy thông lượng neutron tại cuối lớp hiện tại
                end_relative_depth = thickness
                end_thermal_flux = calculator.calculate_thermal_flux_profile(
                    np.array([end_relative_depth]), current_thermal_flux, current_epithermal_flux)[0]
                end_epithermal_flux = calculator.calculate_epithermal_flux_profile(
                    np.array([end_relative_depth]), current_epithermal_flux)[0]
                end_fast_flux = calculator.calculate_fast_flux_profile(
                    np.array([end_relative_depth]), current_fast_flux)[0]
                
                # Cập nhật thông lượng cho lớp tiếp theo
                current_thermal_flux = end_thermal_flux
                current_epithermal_flux = end_epithermal_flux
                current_fast_flux = end_fast_flux
                
                # Cập nhật nồng độ boron (giả định suy giảm 10% qua mỗi lớp)
                current_boron_concentration *= 0.9
        
        return result
    
    def plot_multilayer_depth_dose(self, depths: np.ndarray, doses: Dict[str, np.ndarray]) -> plt.Figure:
        """
        Vẽ đồ thị phân bố liều theo độ sâu cho mô nhiều lớp.
        
        Parameters
        ----------
        depths : np.ndarray
            Mảng các giá trị độ sâu (cm)
        doses : Dict[str, np.ndarray]
            Từ điển chứa các thành phần liều theo độ sâu
            
        Returns
        -------
        plt.Figure
            Đối tượng Figure chứa đồ thị phân bố liều
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Danh sách các thành phần liều và màu sắc tương ứng
        components = [
            ("boron_dose", "Boron Dose", "blue"),
            ("gamma_dose", "Gamma Dose", "green"),
            ("fast_neutron_dose", "Fast Neutron Dose", "red"),
            ("thermal_neutron_dose", "Thermal Neutron Dose", "cyan"),
            ("total_dose", "Total Dose", "black")
        ]
        
        # Vẽ đồ thị cho từng thành phần liều
        for key, label, color in components:
            if key in doses:
                ax.plot(depths, doses[key], color=color, label=label, linewidth=2 if key == "total_dose" else 1)
        
        # Vẽ đường phân cách giữa các lớp mô
        for depth in self.cumulative_depths[1:-1]:
            ax.axvline(x=depth, color='gray', linestyle='--', alpha=0.7)
        
        # Thêm nhãn cho các lớp mô
        for i, (tissue_type, _) in enumerate(self.tissue_layers):
            mid_depth = (self.cumulative_depths[i] + self.cumulative_depths[i+1]) / 2
            ax.text(mid_depth, ax.get_ylim()[1] * 0.9, tissue_type.name,
                   horizontalalignment='center', verticalalignment='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        ax.set_xlabel('Độ sâu (cm)')
        ax.set_ylabel('Liều (Gy)')
        ax.set_title('Phân bố liều theo độ sâu cho mô nhiều lớp')
        ax.legend()
        ax.grid(True)
        
        return fig


def create_depth_dose_profile(depths: np.ndarray,
                             thermal_flux: float,
                             epithermal_flux: float,
                             fast_flux: float,
                             boron_concentration: float,
                             irradiation_time: float,
                             tissue_type: TissueType = TissueType.BRAIN) -> Tuple[Dict[str, np.ndarray], plt.Figure]:
    """
    Hàm tiện ích để tạo và vẽ đồ thị phân bố liều theo độ sâu.
    
    Parameters
    ----------
    depths : np.ndarray
        Mảng các giá trị độ sâu (cm)
    thermal_flux : float
        Thông lượng neutron nhiệt tại bề mặt (n/cm²/s)
    epithermal_flux : float
        Thông lượng neutron trên nhiệt tại bề mặt (n/cm²/s)
    fast_flux : float
        Thông lượng neutron nhanh tại bề mặt (n/cm²/s)
    boron_concentration : float
        Nồng độ boron tại bề mặt (ppm)
    irradiation_time : float
        Thời gian chiếu xạ (giây)
    tissue_type : TissueType, optional
        Loại mô
        
    Returns
    -------
    Tuple[Dict[str, np.ndarray], plt.Figure]
        Từ điển chứa các thành phần liều theo độ sâu và đồ thị
    """
    # Tạo bộ tính toán phân bố liều
    calculator = DepthDoseCalculator(tissue_type)
    
    # Tính toán các thành phần liều
    dose_rates = calculator.calculate_depth_dose_components(
        depths, thermal_flux, epithermal_flux, fast_flux, boron_concentration)
    
    # Tính tổng liều
    doses = calculator.calculate_total_dose(dose_rates, irradiation_time)
    
    # Vẽ đồ thị
    fig = calculator.plot_depth_dose(depths, doses)
    
    return doses, fig


def compare_neutron_beams(depths: np.ndarray,
                         beam_configs: List[Dict[str, float]],
                         beam_labels: List[str],
                         tissue_type: TissueType = TissueType.BRAIN,
                         boron_concentration: float = 20.0,
                         irradiation_time: float = 3600.0) -> plt.Figure:
    """
    So sánh phân bố liều cho các cấu hình chùm tia neutron khác nhau.
    
    Parameters
    ----------
    depths : np.ndarray
        Mảng các giá trị độ sâu (cm)
    beam_configs : List[Dict[str, float]]
        Danh sách các cấu hình chùm tia, mỗi cấu hình là một từ điển chứa
        thông lượng neutron nhiệt, trên nhiệt và nhanh
    beam_labels : List[str]
        Danh sách nhãn cho các cấu hình chùm tia
    tissue_type : TissueType, optional
        Loại mô
    boron_concentration : float, optional
        Nồng độ boron (ppm)
    irradiation_time : float, optional
        Thời gian chiếu xạ (giây)
        
    Returns
    -------
    plt.Figure
        Đối tượng Figure chứa đồ thị so sánh
    """
    # Tạo bộ tính toán phân bố liều
    calculator = DepthDoseCalculator(tissue_type)
    
    # Tạo đồ thị
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Màu sắc cho các đường
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
    
    # Tính toán và vẽ đồ thị cho từng cấu hình chùm tia
    for i, (config, label) in enumerate(zip(beam_configs, beam_labels)):
        # Lấy thông lượng neutron từ cấu hình
        thermal_flux = config.get("thermal_flux", 0.0)
        epithermal_flux = config.get("epithermal_flux", 0.0)
        fast_flux = config.get("fast_flux", 0.0)
        
        # Tính toán các thành phần liều
        dose_rates = calculator.calculate_depth_dose_components(
            depths, thermal_flux, epithermal_flux, fast_flux, boron_concentration)
        
        # Tính tổng liều
        doses = calculator.calculate_total_dose(dose_rates, irradiation_time)
        
        # Vẽ đồ thị tổng liều
        color = colors[i % len(colors)]
        ax.plot(depths, doses["total_dose"], color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Độ sâu (cm)')
    ax.set_ylabel('Liều (Gy)')
    ax.set_title('So sánh phân bố liều cho các cấu hình chùm tia neutron khác nhau')
    ax.legend()
    ax.grid(True)
    
    return fig