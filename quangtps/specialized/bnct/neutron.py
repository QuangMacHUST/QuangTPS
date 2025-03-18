#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho mô hình nguồn neutron và tương tác neutron trong xạ trị bắt neutron boron (BNCT).

Module này cung cấp các lớp và phương thức để mô hình hóa các nguồn neutron,
phổ năng lượng neutron, và tương tác của neutron với mô, phục vụ cho
việc lập kế hoạch điều trị BNCT.
"""

import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

class NeutronSourceType(str, Enum):
    """Enum đại diện cho các loại nguồn neutron sử dụng trong BNCT."""
    REACTOR = "REACTOR"              # Lò phản ứng hạt nhân
    ACCELERATOR = "ACCELERATOR"      # Máy gia tốc
    CYCLOTRON = "CYCLOTRON"          # Cyclotron
    CUSTOM = "CUSTOM"                # Nguồn tùy chỉnh

class NeutronEnergyGroup(str, Enum):
    """Enum đại diện cho các nhóm năng lượng neutron."""
    THERMAL = "THERMAL"      # Neutron nhiệt (E < 0.5 eV)
    EPITHERMAL = "EPITHERMAL"  # Neutron trên nhiệt (0.5 eV < E < 10 keV)
    FAST = "FAST"          # Neutron nhanh (E > 10 keV)

class NeutronSource:
    """Lớp cơ sở cho các mô hình nguồn neutron."""
    
    def __init__(self, source_type: NeutronSourceType = NeutronSourceType.ACCELERATOR,
                 name: str = "Default Neutron Source",
                 thermal_flux: float = 1.0e9,    # n/cm²/s
                 epithermal_flux: float = 5.0e9, # n/cm²/s
                 fast_flux: float = 1.0e8):      # n/cm²/s
        """
        Khởi tạo mô hình nguồn neutron.
        
        Parameters
        ----------
        source_type : NeutronSourceType
            Loại nguồn neutron
        name : str
            Tên mô tả cho nguồn neutron
        thermal_flux : float
            Thông lượng neutron nhiệt (n/cm²/s)
        epithermal_flux : float
            Thông lượng neutron trên nhiệt (n/cm²/s)
        fast_flux : float
            Thông lượng neutron nhanh (n/cm²/s)
        """
        self.source_type = source_type
        self.name = name
        self.flux = {
            NeutronEnergyGroup.THERMAL: thermal_flux,
            NeutronEnergyGroup.EPITHERMAL: epithermal_flux,
            NeutronEnergyGroup.FAST: fast_flux
        }
        self.energy_spectrum = None  # Sẽ được thiết lập bởi các lớp con
        
    def get_total_flux(self) -> float:
        """
        Tính tổng thông lượng neutron.
        
        Returns
        -------
        float
            Tổng thông lượng neutron (n/cm²/s)
        """
        return sum(self.flux.values())
    
    def get_flux_by_group(self, group: NeutronEnergyGroup) -> float:
        """
        Lấy thông lượng neutron theo nhóm năng lượng.
        
        Parameters
        ----------
        group : NeutronEnergyGroup
            Nhóm năng lượng neutron
            
        Returns
        -------
        float
            Thông lượng neutron của nhóm năng lượng (n/cm²/s)
        """
        return self.flux.get(group, 0.0)
    
    def set_flux(self, group: NeutronEnergyGroup, value: float) -> None:
        """
        Thiết lập thông lượng neutron cho một nhóm năng lượng.
        
        Parameters
        ----------
        group : NeutronEnergyGroup
            Nhóm năng lượng neutron
        value : float
            Giá trị thông lượng (n/cm²/s)
        """
        self.flux[group] = value

class ReactorSource(NeutronSource):
    """Lớp mô hình nguồn neutron từ lò phản ứng hạt nhân."""
    
    def __init__(self, name: str = "Reactor Source",
                 thermal_flux: float = 5.0e9,    # n/cm²/s
                 epithermal_flux: float = 1.0e10, # n/cm²/s
                 fast_flux: float = 5.0e8,       # n/cm²/s
                 moderator_type: str = "BeO"):
        """
        Khởi tạo mô hình nguồn neutron từ lò phản ứng.
        
        Parameters
        ----------
        name : str
            Tên mô tả cho nguồn neutron
        thermal_flux : float
            Thông lượng neutron nhiệt (n/cm²/s)
        epithermal_flux : float
            Thông lượng neutron trên nhiệt (n/cm²/s)
        fast_flux : float
            Thông lượng neutron nhanh (n/cm²/s)
        moderator_type : str
            Loại chất làm chậm (moderator)
        """
        super().__init__(NeutronSourceType.REACTOR, name, thermal_flux, epithermal_flux, fast_flux)
        self.moderator_type = moderator_type
        self._initialize_energy_spectrum()
    
    def _initialize_energy_spectrum(self) -> None:
        """
        Khởi tạo phổ năng lượng cho nguồn lò phản ứng.
        """
        # Tạo phổ năng lượng mẫu cho lò phản ứng
        # Thực tế, phổ này sẽ phụ thuộc vào thiết kế cụ thể của lò phản ứng và chất làm chậm
        energy_points = np.logspace(-3, 7, 1000)  # eV, từ 1 meV đến 10 MeV
        
        # Phổ Maxwell cho neutron nhiệt
        thermal_spectrum = 0.5 * (energy_points / 0.0253)**0.5 * np.exp(-energy_points / 0.0253)
        
        # Phổ 1/E cho neutron trên nhiệt
        epithermal_mask = (energy_points >= 0.5) & (energy_points <= 10000)
        epithermal_spectrum = np.zeros_like(energy_points)
        epithermal_spectrum[epithermal_mask] = 1.0 / energy_points[epithermal_mask]
        
        # Phổ Watt cho neutron nhanh
        fast_mask = energy_points > 10000
        fast_spectrum = np.zeros_like(energy_points)
        fast_spectrum[fast_mask] = np.exp(-energy_points[fast_mask] / 1.4e6) * np.sinh(np.sqrt(2.0 * energy_points[fast_mask] / 1.4e6))
        
        # Chuẩn hóa các phổ
        thermal_spectrum = thermal_spectrum / np.sum(thermal_spectrum)
        epithermal_spectrum = epithermal_spectrum / np.sum(epithermal_spectrum)
        fast_spectrum = fast_spectrum / np.sum(fast_spectrum)
        
        # Kết hợp các phổ theo tỷ lệ thông lượng
        total_flux = self.get_total_flux()
        combined_spectrum = (self.flux[NeutronEnergyGroup.THERMAL] * thermal_spectrum + 
                            self.flux[NeutronEnergyGroup.EPITHERMAL] * epithermal_spectrum + 
                            self.flux[NeutronEnergyGroup.FAST] * fast_spectrum) / total_flux
        
        # Lưu phổ năng lượng
        self.energy_spectrum = (energy_points, combined_spectrum)

class AcceleratorSource(NeutronSource):
    """Lớp mô hình nguồn neutron từ máy gia tốc."""
    
    def __init__(self, name: str = "Accelerator Source",
                 thermal_flux: float = 1.0e8,     # n/cm²/s
                 epithermal_flux: float = 1.0e10,  # n/cm²/s
                 fast_flux: float = 1.0e9,        # n/cm²/s
                 target_material: str = "Li",
                 beam_energy: float = 2.5):       # MeV
        """
        Khởi tạo mô hình nguồn neutron từ máy gia tốc.
        
        Parameters
        ----------
        name : str
            Tên mô tả cho nguồn neutron
        thermal_flux : float
            Thông lượng neutron nhiệt (n/cm²/s)
        epithermal_flux : float
            Thông lượng neutron trên nhiệt (n/cm²/s)
        fast_flux : float
            Thông lượng neutron nhanh (n/cm²/s)
        target_material : str
            Vật liệu bia (target)
        beam_energy : float
            Năng lượng chùm tia (MeV)
        """
        super().__init__(NeutronSourceType.ACCELERATOR, name, thermal_flux, epithermal_flux, fast_flux)
        self.target_material = target_material
        self.beam_energy = beam_energy
        self._initialize_energy_spectrum()
    
    def _initialize_energy_spectrum(self) -> None:
        """
        Khởi tạo phổ năng lượng cho nguồn máy gia tốc.
        """
        # Tạo phổ năng lượng mẫu cho máy gia tốc
        # Thực tế, phổ này sẽ phụ thuộc vào vật liệu bia và năng lượng chùm tia
        energy_points = np.logspace(-3, 7, 1000)  # eV, từ 1 meV đến 10 MeV
        
        # Phổ năng lượng phụ thuộc vào vật liệu bia
        if self.target_material == "Li":
            # Phổ cho phản ứng Li(p,n)
            peak_energy = self.beam_energy * 0.8 * 1e6  # eV
            spectrum = np.exp(-(energy_points - peak_energy)**2 / (2 * (peak_energy/5)**2))
        elif self.target_material == "Be":
            # Phổ cho phản ứng Be(p,n)
            peak_energy = self.beam_energy * 0.7 * 1e6  # eV
            spectrum = np.exp(-(energy_points - peak_energy)**2 / (2 * (peak_energy/4)**2))
        else:
            # Phổ mặc định
            peak_energy = self.beam_energy * 0.5 * 1e6  # eV
            spectrum = np.exp(-(energy_points - peak_energy)**2 / (2 * (peak_energy/3)**2))
        
        # Chuẩn hóa phổ
        spectrum = spectrum / np.sum(spectrum)
        
        # Lưu phổ năng lượng
        self.energy_spectrum = (energy_points, spectrum)

class NeutronInteraction:
    """Lớp mô hình tương tác neutron với mô."""
    
    def __init__(self):
        """
        Khởi tạo mô hình tương tác neutron.
        """
        # Tiết diện tương tác mặc định (cm²) cho các nguyên tố phổ biến trong mô
        self.cross_sections = {
            "H": {"thermal": 0.332, "epithermal": 0.01, "fast": 0.005},
            "C": {"thermal": 0.0035, "epithermal": 0.002, "fast": 0.001},
            "N": {"thermal": 1.9, "epithermal": 0.05, "fast": 0.01},
            "O": {"thermal": 0.00019, "epithermal": 0.0001, "fast": 0.00005},
            "B-10": {"thermal": 3837.0, "epithermal": 100.0, "fast": 1.0}
        }
        
        # Hệ số RBE (Relative Biological Effectiveness) cho các thành phần liều
        self.rbe_factors = {
            "boron_dose": 3.8,  # Mặc định cho BPA
            "gamma_dose": 1.0,
            "fast_neutron_dose": 3.2,
            "thermal_neutron_dose": 2.3
        }
    
    def calculate_kerma_factors(self, material_composition: Dict[str, float]) -> Dict[NeutronEnergyGroup, float]:
        """
        Tính toán hệ số kerma cho các nhóm năng lượng neutron dựa trên thành phần vật liệu.
        
        Parameters
        ----------
        material_composition : Dict[str, float]
            Từ điển chứa thành phần vật liệu (nguyên tố: phần trăm khối lượng)
            
        Returns
        -------
        Dict[NeutronEnergyGroup, float]
            Hệ số kerma cho mỗi nhóm năng lượng neutron (Gy·cm²)
        """
        kerma_factors = {}
        
        # Hệ số chuyển đổi từ tiết diện sang kerma (Gy·cm²)
        conversion_factors = {
            "H": 4.2e-14,
            "C": 1.1e-14,
            "N": 5.8e-14,
            "O": 2.7e-14,
            "B-10": 8.6e-12
        }
        
        # Tính toán hệ số kerma cho mỗi nhóm năng lượng
        for group in NeutronEnergyGroup:
            kerma_sum = 0.0
            for element, weight_percent in material_composition.items():
                if element in self.cross_sections and element in conversion_factors:
                    # Lấy tiết diện tương ứng với nhóm năng lượng
                    if group == NeutronEnergyGroup.THERMAL:
                        cross_section = self.cross_sections[element]["thermal"]
                    elif group == NeutronEnergyGroup.EPITHERMAL:
                        cross_section = self.cross_sections[element]["epithermal"]
                    else:  # FAST
                        cross_section = self.cross_sections[element]["fast"]
                    
                    # Tính đóng góp vào hệ số kerma
                    kerma_contribution = weight_percent * cross_section * conversion_factors[element]
                    kerma_sum += kerma_contribution
            
            kerma_factors[group] = kerma_sum
        
        return kerma_factors
    
    def calculate_dose_components(self, neutron_source: NeutronSource, 
                                material_composition: Dict[str, float],
                                boron_concentration: float = 0.0,
                                exposure_time: float = 3600.0) -> Dict[str, float]:
        """
        Tính toán các thành phần liều trong BNCT.
        
        Parameters
        ----------
        neutron_source : NeutronSource
            Nguồn neutron sử dụng
        material_composition : Dict[str, float]
            Thành phần vật liệu (nguyên tố: phần trăm khối lượng)
        boron_concentration : float, optional
            Nồng độ boron-10 trong mô (ppm)
        exposure_time : float, optional
            Thời gian chiếu xạ (giây)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa các thành phần liều (Gy)
        """
        # Tính hệ số kerma
        kerma_factors = self.calculate_kerma_factors(material_composition)
        
        # Tính các thành phần liều
        dose_components = {}
        
        # Liều từ neutron nhiệt
        thermal_dose = kerma_factors[NeutronEnergyGroup.THERMAL] * \
                      neutron_source.get_flux_by_group(NeutronEnergyGroup.THERMAL) * \
                      exposure_time
        dose_components["thermal_neutron_dose"] = thermal_dose
        
        # Liều từ neutron nhanh
        fast_dose = kerma_factors[NeutronEnergyGroup.FAST] * \
                  neutron_source.get_flux_by_group(NeutronEnergyGroup.FAST) * \
                  exposure_time
        dose_components["fast_neutron_dose"] = fast_dose
        
        # Liều gamma
        # Giả định: liều gamma tỷ lệ với thông lượng neutron nhiệt
        gamma_factor = 2.0e-13  # Gy·cm²
        gamma_dose = gamma_factor * neutron_source.get_flux_by_group(NeutronEnergyGroup.THERMAL) * \
                    exposure_time
        dose_components["gamma_dose"] = gamma_dose
        
        # Liều boron (nếu có boron)
        if boron_concentration > 0:
            # Hệ số chuyển đổi từ ppm boron sang liều
            # 7.43e-14 Gy·cm²/ppm cho phản ứng B-10(n,α)Li-7
            boron_factor = 7.43e-14
            boron_dose = boron_factor * boron_concentration * \
                       neutron_source.get_flux_by_group(NeutronEnergyGroup.THERMAL) * \
                       exposure_time
            dose_components["boron_dose"] = boron_dose
        else:
            dose_components["boron_dose"] = 0.0
        
        # Tổng liều vật lý
        dose_components["total_physical_dose"] = sum(dose_components.values())
        
        return dose_components
    
    def calculate_biologically_weighted_dose(self, dose_components: Dict[str, float], 
                                           cbr: float = 3.8) -> Dict[str, float]:
        """
        Tính toán liều sinh học tương đương.
        
        Parameters
        ----------
        dose_components : Dict[str, float]
            Từ điển chứa các thành phần liều vật lý (Gy)
        cbr : float, optional
            Hệ số hiệu quả sinh học của hợp chất boron (Compound Biological Effectiveness)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa các thành phần liều sinh học tương đương (Gy-Eq)
        """
        # Cập nhật hệ số RBE cho liều boron dựa trên CBR
        self.rbe_factors["boron_dose"] = cbr
        
        # Tính liều sinh học tương đương cho mỗi thành phần
        weighted_doses = {}
        for component, dose in dose_components.items():
            if component in self.rbe_factors:
                weighted_doses[component] = dose * self.rbe_factors[component]
            else:
                weighted_doses[component] = dose
        
        # Tổng liều sinh học tương đương
        weighted_doses["total_biologically_weighted_dose"] = sum(
            [weighted_doses[k] for k in weighted_doses if k != "total_physical_dose"]
        )
        
        return weighted_doses
    
    def plot_dose_components(self, dose_components: Dict[str, float], 
                           weighted_doses: Optional[Dict[str, float]] = None,
                           title: str = "BNCT Dose Components"):
        """
        Vẽ biểu đồ các thành phần liều.
        
        Parameters
        ----------
        dose_components : Dict[str, float]
            Từ điển chứa các thành phần liều vật lý
        weighted_doses : Dict[str, float], optional
            Từ điển chứa các thành phần liều sinh học tương đương
        title : str, optional
            Tiêu đề biểu đồ
        """
        components = ["boron_dose", "gamma_dose", "fast_neutron_dose", "thermal_neutron_dose"]
        physical_doses = [dose_components[c] for c in components]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ liều vật lý
        x = np.arange(len(components))
        bars1 = ax.bar(x - 0.2, physical_doses, width=0.4, label="Physical Dose")
        
        # Vẽ liều sinh học tương đương (nếu có)
        if weighted_doses is not None:
            weighted_values = [weighted_doses[c] for c in components]
            bars2 = ax.bar(x + 0.2, weighted_values, width=0.4, label="Biologically Weighted Dose")
        
        # Thiết lập trục và nhãn
        ax.set_xticks(x)
        ax.set_xticklabels(["Boron", "Gamma", "Fast Neutron", "Thermal Neutron"])
        ax.set_ylabel("Dose (Gy or Gy-Eq)")
        ax.set_title(title)
        ax.legend()
        
        # Hiển thị giá trị trên các cột
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f"{height:.3f}", ha="center", va="bottom")
        
        add_labels(bars1)
        if weighted_doses is not None:
            add_labels(bars2)
        
        plt.tight_layout()
        return fig

class CyclotronSource(NeutronSource):
    """Lớp mô hình nguồn neutron từ cyclotron."""
    
    def __init__(self, name: str = "Cyclotron Source",
                 thermal_flux: float = 2.0e8,     # n/cm²/s
                 epithermal_flux: float = 8.0e9,  # n/cm²/s
                 fast_flux: float = 2.0e9,        # n/cm²/s
                 target_material: str = "Be",
                 beam_energy: float = 30.0):      # MeV
        """
        Khởi tạo mô hình nguồn neutron từ cyclotron.
        
        Parameters
        ----------
        name : str
            Tên mô tả cho nguồn neutron
        thermal_flux : float
            Thông lượng neutron nhiệt (n/cm²/s)
        epithermal_flux : float
            Thông lượng neutron trên nhiệt (n/cm²/s)
        fast_flux : float
            Thông lượng neutron nhanh (n/cm²/s)
        target_material : str
            Vật liệu bia (target)
        beam_energy : float
            Năng lượng chùm tia (MeV)
        """
        super().__init__(NeutronSourceType.CYCLOTRON, name, thermal_flux, epithermal_flux, fast_flux)
        self.target_material = target_material
        self.beam_energy = beam_energy
        self._initialize_energy_spectrum()
    
    def _initialize_energy_spectrum(self) -> None:
        """
        Khởi tạo phổ năng lượng cho nguồn cyclotron.
        """
        # Tạo phổ năng lượng mẫu cho cyclotron
        # Thực tế, phổ này sẽ phụ thuộc vào vật liệu bia và năng lượng chùm tia
        energy_points = np.logspace(-3, 7, 1000)  # eV, từ 1 meV đến 10 MeV
        
        # Phổ năng lượng phụ thuộc vào vật liệu bia và năng lượng cao hơn
        if self.target_material == "Be":
            # Phổ cho phản ứng Be(d,n) hoặc Be(p,n) với năng lượng cao
            peak_energy = self.beam_energy * 0.6 * 1e6  # eV
            width = peak_energy / 3
            # Phổ cyclotron thường rộng hơn và có đuôi năng lượng cao
            spectrum = np.exp(-(energy_points - peak_energy)**2 / (2 * width**2))
            # Thêm đuôi năng lượng cao
            high_energy_tail = np.zeros_like(energy_points)
            high_energy_mask = energy_points > peak_energy
            high_energy_tail[high_energy_mask] = 0.3 * np.exp(-(energy_points[high_energy_mask] - peak_energy) / (peak_energy * 0.5))
            spectrum = spectrum + high_energy_tail
        elif self.target_material == "W":
            # Phổ cho phản ứng với tungsten (W)
            peak_energy = self.beam_energy * 0.4 * 1e6  # eV
            width = peak_energy / 2.5
            spectrum = np.exp(-(energy_points - peak_energy)**2 / (2 * width**2))
            # Thêm đỉnh thứ hai ở năng lượng thấp hơn
            second_peak = 0.7 * np.exp(-(energy_points - peak_energy*0.3)**2 / (2 * (width*0.6)**2))
            spectrum = spectrum + second_peak
        else:
            # Phổ mặc định
            peak_energy = self.beam_energy * 0.5 * 1e6  # eV
            spectrum = np.exp(-(energy_points - peak_energy)**2 / (2 * (peak_energy/2.5)**2))
        
        # Chuẩn hóa phổ
        spectrum = spectrum / np.sum(spectrum)
        
        # Lưu phổ năng lượng
        self.energy_spectrum = (energy_points, spectrum)
    
    def get_beam_parameters(self) -> Dict[str, Any]:
        """
        Lấy các tham số chùm tia của nguồn cyclotron.
        
        Returns
        -------
        Dict[str, Any]
            Từ điển chứa các tham số chùm tia
        """
        return {
            "source_type": self.source_type,
            "target_material": self.target_material,
            "beam_energy": self.beam_energy,
            "thermal_flux": self.flux[NeutronEnergyGroup.THERMAL],
            "epithermal_flux": self.flux[NeutronEnergyGroup.EPITHERMAL],
            "fast_flux": self.flux[NeutronEnergyGroup.FAST],
            "total_flux": self.get_total_flux()
        }
    
    def plot_energy_spectrum(self, log_scale: bool = True, figsize: Tuple[int, int] = (10, 6)):
        """
        Vẽ phổ năng lượng của nguồn cyclotron.
        
        Parameters
        ----------
        log_scale : bool, optional
            Sử dụng thang logarit cho trục x
        figsize : Tuple[int, int], optional
            Kích thước hình vẽ
            
        Returns
        -------
        matplotlib.figure.Figure
            Đối tượng Figure của matplotlib
        """
        if self.energy_spectrum is None:
            self._initialize_energy_spectrum()
        
        energy_points, spectrum = self.energy_spectrum
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(energy_points, spectrum, 'b-', linewidth=2)
        
        # Đánh dấu các vùng năng lượng
        ax.axvspan(0, 0.5, alpha=0.2, color='blue', label='Thermal')
        ax.axvspan(0.5, 10000, alpha=0.2, color='green', label='Epithermal')
        ax.axvspan(10000, energy_points[-1], alpha=0.2, color='red', label='Fast')
        
        if log_scale:
            ax.set_xscale('log')
        
        ax.set_xlabel('Năng lượng (eV)')
        ax.set_ylabel('Cường độ (đơn vị tùy ý)')
        ax.set_title(f'Phổ năng lượng neutron - {self.name}')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        return fig

class BaseNeutronModel:
    """Lớp cơ sở cho các mô hình nguồn neutron."""
    
    def __init__(self):
        """Khởi tạo mô hình neutron cơ bản."""
        self.source = None
        self.gamma_dose_rate = 0.0  # Gy/s
    
    def set_source(self, source: NeutronSource) -> None:
        """
        Thiết lập nguồn neutron cho mô hình.
        
        Parameters
        ----------
        source : NeutronSource
            Nguồn neutron
        """
        self.source = source
    
    def calculate_thermal_flux(self, depth: float) -> float:
        """
        Tính thông lượng neutron nhiệt tại một độ sâu.
        
        Parameters
        ----------
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        float
            Thông lượng neutron nhiệt (n/cm²/s)
        """
        if not self.source:
            return 0.0
        
        base_flux = self.source.get_flux_by_group(NeutronEnergyGroup.THERMAL)
        # Mô hình suy giảm mặc định theo hàm mũ
        return base_flux * np.exp(-0.1 * depth)
    
    def calculate_epithermal_flux(self, depth: float) -> float:
        """
        Tính thông lượng neutron trên nhiệt tại một độ sâu.
        
        Parameters
        ----------
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        float
            Thông lượng neutron trên nhiệt (n/cm²/s)
        """
        if not self.source:
            return 0.0
        
        base_flux = self.source.get_flux_by_group(NeutronEnergyGroup.EPITHERMAL)
        # Mô hình suy giảm mặc định theo hàm mũ
        return base_flux * np.exp(-0.07 * depth)
    
    def calculate_fast_flux(self, depth: float) -> float:
        """
        Tính thông lượng neutron nhanh tại một độ sâu.
        
        Parameters
        ----------
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        float
            Thông lượng neutron nhanh (n/cm²/s)
        """
        if not self.source:
            return 0.0
        
        base_flux = self.source.get_flux_by_group(NeutronEnergyGroup.FAST)
        # Mô hình suy giảm mặc định theo hàm mũ
        return base_flux * np.exp(-0.15 * depth)
    
    def calculate_gamma_dose(self, depth: float) -> float:
        """
        Tính liều gamma tại một độ sâu.
        
        Parameters
        ----------
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        float
            Liều gamma (Gy)
        """
        # Mô hình suy giảm mặc định theo hàm mũ
        return self.gamma_dose_rate * np.exp(-0.05 * depth)


class AcceleratorNeutronModel(BaseNeutronModel):
    """Lớp mô hình neutron cho nguồn từ máy gia tốc."""
    
    def __init__(self):
        """Khởi tạo mô hình neutron máy gia tốc."""
        super().__init__()
        self.source = AcceleratorSource()
        self.gamma_dose_rate = 0.05  # Gy/s
    
    def calculate_thermal_flux(self, depth: float) -> float:
        """
        Tính thông lượng neutron nhiệt tại một độ sâu cho nguồn máy gia tốc.
        
        Parameters
        ----------
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        float
            Thông lượng neutron nhiệt (n/cm²/s)
        """
        if not self.source:
            return 0.0
        
        base_flux = self.source.get_flux_by_group(NeutronEnergyGroup.THERMAL)
        # Mô hình suy giảm đặc thù cho máy gia tốc
        return base_flux * np.exp(-0.12 * depth)


class ReactorNeutronModel(BaseNeutronModel):
    """Lớp mô hình neutron cho nguồn từ lò phản ứng."""
    
    def __init__(self):
        """Khởi tạo mô hình neutron lò phản ứng."""
        super().__init__()
        self.source = ReactorSource()
        self.gamma_dose_rate = 0.1  # Gy/s
    
    def calculate_epithermal_flux(self, depth: float) -> float:
        """
        Tính thông lượng neutron trên nhiệt tại một độ sâu cho nguồn lò phản ứng.
        
        Parameters
        ----------
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        float
            Thông lượng neutron trên nhiệt (n/cm²/s)
        """
        if not self.source:
            return 0.0
        
        base_flux = self.source.get_flux_by_group(NeutronEnergyGroup.EPITHERMAL)
        # Mô hình suy giảm đặc thù cho lò phản ứng
        return base_flux * np.exp(-0.05 * depth)


class DDGeneratorModel(BaseNeutronModel):
    """Lớp mô hình neutron cho nguồn phát D-D."""
    
    def __init__(self):
        """Khởi tạo mô hình neutron máy phát D-D."""
        super().__init__()
        # Tạo một nguồn neutron tùy chỉnh phù hợp với đặc tính của máy phát D-D
        self.source = NeutronSource(
            source_type=NeutronSourceType.CUSTOM,
            name="D-D Generator",
            thermal_flux=1.0e7,     # Thông lượng thấp hơn
            epithermal_flux=5.0e8,
            fast_flux=1.0e9         # Thông lượng neutron nhanh cao hơn
        )
        self.gamma_dose_rate = 0.02  # Gy/s


class DTGeneratorModel(BaseNeutronModel):
    """Lớp mô hình neutron cho nguồn phát D-T."""
    
    def __init__(self):
        """Khởi tạo mô hình neutron máy phát D-T."""
        super().__init__()
        # Tạo một nguồn neutron tùy chỉnh phù hợp với đặc tính của máy phát D-T
        self.source = NeutronSource(
            source_type=NeutronSourceType.CUSTOM,
            name="D-T Generator",
            thermal_flux=5.0e6,     # Thông lượng thấp hơn
            epithermal_flux=2.0e8,
            fast_flux=2.0e9         # Thông lượng neutron nhanh cao hơn nhiều
        )
        self.gamma_dose_rate = 0.01  # Gy/s
        
    def calculate_fast_flux(self, depth: float) -> float:
        """
        Tính thông lượng neutron nhanh tại một độ sâu cho nguồn D-T.
        
        Parameters
        ----------
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        float
            Thông lượng neutron nhanh (n/cm²/s)
        """
        if not self.source:
            return 0.0
        
        base_flux = self.source.get_flux_by_group(NeutronEnergyGroup.FAST)
        # Mô hình suy giảm đặc thù cho máy phát D-T (neutron 14 MeV)
        return base_flux * np.exp(-0.1 * depth) * (1 + 0.05 * depth)  # Buildup factor


class GenericNeutronModel(BaseNeutronModel):
    """Lớp mô hình neutron chung."""
    
    def __init__(self):
        """Khởi tạo mô hình neutron chung."""
        super().__init__()
        self.source = NeutronSource()
        self.gamma_dose_rate = 0.03  # Gy/s


# Đảm bảo xuất các lớp mới
__all__ = [
    'NeutronSourceType', 'NeutronEnergyGroup', 'NeutronSource',
    'ReactorSource', 'AcceleratorSource', 'CyclotronSource',
    'NeutronInteraction', 'BaseNeutronModel', 'AcceleratorNeutronModel',
    'ReactorNeutronModel', 'DDGeneratorModel', 'DTGeneratorModel', 'GenericNeutronModel'
]