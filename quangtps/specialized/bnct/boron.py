#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho mô hình phân bố boron trong xạ trị bắt neutron boron (BNCT).

Module này cung cấp các lớp và phương thức để mô hình hóa và tính toán
sự phân bố của các hợp chất boron trong các mô khác nhau, phục vụ cho
việc lập kế hoạch điều trị BNCT.
"""

import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

class BoronCompoundType(str, Enum):
    """Enum đại diện cho các loại hợp chất boron sử dụng trong BNCT."""
    BPA = "BPA"          # Boronophenylalanine
    BSH = "BSH"          # Sodium borocaptate
    BOPP = "BOPP"        # Boronated porphyrin
    GB10 = "GB10"        # Gadolinium-boron compound
    CUSTOM = "CUSTOM"    # Hợp chất tùy chỉnh

@dataclass
class BoronCompoundProperties:
    """Lớp chứa các thuộc tính của hợp chất boron."""
    name: str
    molecular_weight: float  # g/mol
    boron_content: float     # % khối lượng
    cbr: float               # Compound Biological Effectiveness
    half_life: float         # Thời gian bán hủy trong máu (giờ)
    tumor_normal_ratio: float  # Tỷ lệ nồng độ u/mô lành
    description: str = ""

# Dữ liệu mặc định cho các hợp chất boron phổ biến
DEFAULT_COMPOUND_PROPERTIES = {
    BoronCompoundType.BPA: BoronCompoundProperties(
        name="Boronophenylalanine",
        molecular_weight=243.0,
        boron_content=4.9,  # % khối lượng
        cbr=3.8,
        half_life=6.2,  # giờ
        tumor_normal_ratio=3.5,
        description="Hợp chất boron phenylalanine, thường được sử dụng cho các khối u não và melanoma."
    ),
    BoronCompoundType.BSH: BoronCompoundProperties(
        name="Sodium borocaptate",
        molecular_weight=302.0,
        boron_content=59.3,  # % khối lượng
        cbr=2.5,
        half_life=4.8,  # giờ
        tumor_normal_ratio=1.2,
        description="Hợp chất boron sulfhydryl, thường được sử dụng cho các khối u não."
    ),
    BoronCompoundType.BOPP: BoronCompoundProperties(
        name="Boronated porphyrin",
        molecular_weight=1650.0,
        boron_content=5.0,  # % khối lượng
        cbr=4.2,
        half_life=12.0,  # giờ
        tumor_normal_ratio=5.0,
        description="Porphyrin boron hóa, có khả năng tích tụ cao trong các khối u."
    ),
    BoronCompoundType.GB10: BoronCompoundProperties(
        name="Gadolinium-boron compound",
        molecular_weight=1200.0,
        boron_content=8.5,  # % khối lượng
        cbr=3.2,
        half_life=8.0,  # giờ
        tumor_normal_ratio=4.0,
        description="Hợp chất boron-gadolinium, kết hợp khả năng tạo ảnh MRI và điều trị BNCT."
    ),
}

class BoronDistributionModel:
    """Lớp cơ sở cho các mô hình phân bố boron."""
    
    def __init__(self, compound_type: BoronCompoundType = BoronCompoundType.BPA,
                 properties: Optional[BoronCompoundProperties] = None):
        """
        Khởi tạo mô hình phân bố boron.
        
        Parameters
        ----------
        compound_type : BoronCompoundType
            Loại hợp chất boron
        properties : BoronCompoundProperties, optional
            Thuộc tính của hợp chất, nếu None sẽ sử dụng giá trị mặc định
        """
        self.compound_type = compound_type
        
        if properties is None:
            if compound_type in DEFAULT_COMPOUND_PROPERTIES:
                self.properties = DEFAULT_COMPOUND_PROPERTIES[compound_type]
            else:
                # Giá trị mặc định cho hợp chất tùy chỉnh
                self.properties = BoronCompoundProperties(
                    name="Custom compound",
                    molecular_weight=250.0,
                    boron_content=5.0,
                    cbr=3.0,
                    half_life=6.0,
                    tumor_normal_ratio=3.0,
                    description="Hợp chất boron tùy chỉnh"
                )
        else:
            self.properties = properties
    
    def calculate_concentration(self, time_after_injection: float, 
                              initial_concentration: float) -> float:
        """
        Tính toán nồng độ boron theo thời gian sau khi tiêm.
        
        Parameters
        ----------
        time_after_injection : float
            Thời gian sau khi tiêm (giờ)
        initial_concentration : float
            Nồng độ ban đầu (ppm)
            
        Returns
        -------
        float
            Nồng độ boron tại thời điểm chỉ định (ppm)
        """
        # Mô hình suy giảm theo hàm mũ đơn giản
        decay_constant = np.log(2) / self.properties.half_life
        return initial_concentration * np.exp(-decay_constant * time_after_injection)
    
    def tumor_to_normal_ratio(self, time_after_injection: float) -> float:
        """
        Tính toán tỷ lệ nồng độ boron trong u so với mô lành.
        
        Parameters
        ----------
        time_after_injection : float
            Thời gian sau khi tiêm (giờ)
            
        Returns
        -------
        float
            Tỷ lệ nồng độ u/mô lành
        """
        # Mô hình đơn giản, có thể mở rộng với dữ liệu thực nghiệm
        base_ratio = self.properties.tumor_normal_ratio
        
        # Giả định tỷ lệ đạt tối đa sau 2-3 giờ, sau đó giảm dần
        if time_after_injection < 2.0:
            return base_ratio * (time_after_injection / 2.0)
        elif time_after_injection < 4.0:
            return base_ratio
        else:
            decay_factor = np.exp(-0.1 * (time_after_injection - 4.0))
            return max(1.0, base_ratio * decay_factor)

class TwoCompartmentModel(BoronDistributionModel):
    """Mô hình phân bố boron hai ngăn."""
    
    def __init__(self, compound_type: BoronCompoundType = BoronCompoundType.BPA,
                 properties: Optional[BoronCompoundProperties] = None,
                 k12: float = 0.4, k21: float = 0.2, k10: float = 0.1):
        """
        Khởi tạo mô hình phân bố boron hai ngăn.
        
        Parameters
        ----------
        compound_type : BoronCompoundType
            Loại hợp chất boron
        properties : BoronCompoundProperties, optional
            Thuộc tính của hợp chất
        k12 : float
            Hằng số tốc độ từ ngăn 1 (máu) sang ngăn 2 (mô)
        k21 : float
            Hằng số tốc độ từ ngăn 2 (mô) sang ngăn 1 (máu)
        k10 : float
            Hằng số tốc độ thải trừ từ ngăn 1 (máu)
        """
        super().__init__(compound_type, properties)
        self.k12 = k12
        self.k21 = k21
        self.k10 = k10
    
    def calculate_concentration(self, time_after_injection: float, 
                              initial_concentration: float) -> Dict[str, float]:
        """
        Tính toán nồng độ boron theo mô hình hai ngăn.
        
        Parameters
        ----------
        time_after_injection : float
            Thời gian sau khi tiêm (giờ)
        initial_concentration : float
            Nồng độ ban đầu trong máu (ppm)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa nồng độ boron trong máu và mô
        """
        # Tính toán các hằng số tốc độ tổng hợp
        alpha = 0.5 * ((self.k12 + self.k21 + self.k10) + 
                      np.sqrt((self.k12 + self.k21 + self.k10)**2 - 4*self.k21*self.k10))
        beta = 0.5 * ((self.k12 + self.k21 + self.k10) - 
                     np.sqrt((self.k12 + self.k21 + self.k10)**2 - 4*self.k21*self.k10))
        
        # Tính toán nồng độ trong máu (ngăn 1)
        A = (alpha - self.k21) / (alpha - beta)
        B = (self.k21 - beta) / (alpha - beta)
        blood_conc = initial_concentration * (A * np.exp(-alpha * time_after_injection) + 
                                           B * np.exp(-beta * time_after_injection))
        
        # Tính toán nồng độ trong mô (ngăn 2)
        tissue_conc = initial_concentration * self.k12 * \
                    ((np.exp(-beta * time_after_injection) - np.exp(-alpha * time_after_injection)) / 
                     (alpha - beta))
        
        return {"blood": blood_conc, "tissue": tissue_conc}

class BoronDistributionAnalyzer:
    """Lớp phân tích và trực quan hóa phân bố boron."""
    
    def __init__(self, model: BoronDistributionModel):
        """
        Khởi tạo bộ phân tích phân bố boron.
        
        Parameters
        ----------
        model : BoronDistributionModel
            Mô hình phân bố boron
        """
        self.model = model
    
    def plot_concentration_time_curve(self, initial_concentration: float, 
                                     max_time: float = 24.0, 
                                     time_points: int = 100) -> plt.Figure:
        """
        Vẽ đồ thị nồng độ boron theo thời gian.
        
        Parameters
        ----------
        initial_concentration : float
            Nồng độ ban đầu (ppm)
        max_time : float
            Thời gian tối đa để vẽ (giờ)
        time_points : int
            Số điểm thời gian để vẽ
            
        Returns
        -------
        plt.Figure
            Đối tượng hình vẽ matplotlib
        """
        times = np.linspace(0, max_time, time_points)
        
        if isinstance(self.model, TwoCompartmentModel):
            concentrations = [self.model.calculate_concentration(t, initial_concentration) 
                             for t in times]
            blood_conc = [c["blood"] for c in concentrations]
            tissue_conc = [c["tissue"] for c in concentrations]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times, blood_conc, 'r-', label='Nồng độ trong máu')
            ax.plot(times, tissue_conc, 'b-', label='Nồng độ trong mô')
        else:
            concentrations = [self.model.calculate_concentration(t, initial_concentration) 
                             for t in times]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times, concentrations, 'g-', label='Nồng độ boron')
        
        ax.set_xlabel('Thời gian sau khi tiêm (giờ)')
        ax.set_ylabel('Nồng độ boron (ppm)')
        ax.set_title(f'Đường cong nồng độ boron theo thời gian - {self.model.properties.name}')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def plot_tumor_normal_ratio(self, max_time: float = 24.0, 
                               time_points: int = 100) -> plt.Figure:
        """
        Vẽ đồ thị tỷ lệ nồng độ boron u/mô lành theo thời gian.
        
        Parameters
        ----------
        max_time : float
            Thời gian tối đa để vẽ (giờ)
        time_points : int
            Số điểm thời gian để vẽ
            
        Returns
        -------
        plt.Figure
            Đối tượng hình vẽ matplotlib
        """
        times = np.linspace(0, max_time, time_points)
        ratios = [self.model.tumor_to_normal_ratio(t) for t in times]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, ratios, 'g-', label='Tỷ lệ nồng độ u/mô lành')
        
        ax.set_xlabel('Thời gian sau khi tiêm (giờ)')
        ax.set_ylabel('Tỷ lệ nồng độ u/mô lành')
        ax.set_title(f'Đồ thị tỷ lệ nồng độ u/mô lành theo thời gian - {self.model.properties.name}')
        ax.grid(True)
        ax.legend()
        
        return fig