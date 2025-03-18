#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các mô hình phân bố boron trong BNCT.

Module này cung cấp các lớp và phương thức để mô hình hóa sự phân bố
và tương tác của các hợp chất boron sử dụng trong xạ trị bắt neutron boron (BNCT).
"""

import logging
import numpy as np
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)

class BoronCompoundType(str, Enum):
    """Enum đại diện cho các loại hợp chất boron sử dụng trong BNCT."""
    BPA = "BPA"  # Boronophenylalanine
    BSH = "BSH"  # Sodium borocaptate
    BORONOPHENYLALANINE = "BORONOPHENYLALANINE"  # Tên đầy đủ
    CUSTOM = "CUSTOM"  # Hợp chất tùy chỉnh

class BoronDistributionModel:
    """Lớp cơ sở cho các mô hình phân bố boron."""
    
    def __init__(self, tumor_to_blood_ratio: float = 3.5):
        """
        Khởi tạo mô hình phân bố boron.
        
        Parameters
        ----------
        tumor_to_blood_ratio : float
            Tỷ lệ nồng độ boron trong u so với máu
        """
        self.tumor_to_blood_ratio = tumor_to_blood_ratio
        self.name = "Generic Boron Distribution Model"
        
    def calculate_boron_dose(self, thermal_flux: float, concentration: float, depth: float) -> Tuple[float, float]:
        """
        Tính toán liều boron dựa trên thông lượng neutron và nồng độ boron.
        
        Parameters
        ----------
        thermal_flux : float
            Thông lượng neutron nhiệt (n/cm²/s)
        concentration : float
            Nồng độ boron trong mô u (ppm)
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        Tuple[float, float]
            Liều alpha và lithium (Gy)
        """
        # Phương trình cơ bản cho liều boron:
        # Dose = Thermal flux × B concentration × Kerma factor × Irradiation time
        # Giả sử thởi gian chiếu xạ là 1 giây
        
        # Hệ số kerma cho phản ứng 10B(n,α)7Li (Gy·cm²/ppm)
        kerma_factor = 8.66e-14  # Gy·cm²/n/ppm
        
        # Tính liều từ phản ứng bắt neutron
        total_dose = thermal_flux * concentration * kerma_factor
        
        # Phân chia thành các thành phần liều
        alpha_dose = total_dose * 0.6  # 60% là từ hạt alpha
        lithium_dose = total_dose * 0.4  # 40% là từ hạt lithium-7
        
        return alpha_dose, lithium_dose
    
    def get_concentration_ratio(self, tissue_type: str) -> float:
        """
        Lấy tỷ lệ nồng độ boron giữa các loại mô.
        
        Parameters
        ----------
        tissue_type : str
            Loại mô ('tumor', 'normal', 'blood', etc.)
            
        Returns
        -------
        float
            Tỷ lệ nồng độ boron so với máu
        """
        if tissue_type == 'tumor':
            return self.tumor_to_blood_ratio
        elif tissue_type == 'normal':
            return 1.0  # Giả sử mô lành có nồng độ bằng máu
        elif tissue_type == 'blood':
            return 1.0
        else:
            return 0.5  # Giá trị mặc định cho các mô khác
    
    def calculate_concentration(self, base_concentration: float, tissue_type: str) -> float:
        """
        Tính nồng độ boron trong một loại mô cụ thể.
        
        Parameters
        ----------
        base_concentration : float
            Nồng độ boron cơ sở (ppm) - thường là nồng độ trong máu
        tissue_type : str
            Loại mô ('tumor', 'normal', 'blood', etc.)
            
        Returns
        -------
        float
            Nồng độ boron trong mô (ppm)
        """
        ratio = self.get_concentration_ratio(tissue_type)
        return base_concentration * ratio


class BPAModel(BoronDistributionModel):
    """Mô hình phân bố boron cho hợp chất BPA (Boronophenylalanine)."""
    
    def __init__(self, tumor_to_normal_ratio: float = 3.5):
        """
        Khởi tạo mô hình BPA.
        
        Parameters
        ----------
        tumor_to_normal_ratio : float
            Tỷ lệ nồng độ boron trong u so với mô lành
        """
        super().__init__(tumor_to_normal_ratio)
        self.name = "BPA Distribution Model"
        
    def calculate_boron_dose(self, thermal_flux: float, concentration: float, depth: float) -> Tuple[float, float]:
        """
        Tính toán liều boron đặc thù cho BPA.
        
        Parameters
        ----------
        thermal_flux : float
            Thông lượng neutron nhiệt (n/cm²/s)
        concentration : float
            Nồng độ boron trong mô u (ppm)
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        Tuple[float, float]
            Liều alpha và lithium (Gy)
        """
        # Hệ số kerma đặc thù cho BPA
        kerma_factor = 7.43e-14  # Gy·cm²/n/ppm
        
        # Mô hình suy giảm nồng độ theo độ sâu cho BPA
        depth_factor = np.exp(-0.02 * depth)
        effective_concentration = concentration * depth_factor
        
        # Tính liều từ phản ứng bắt neutron
        total_dose = thermal_flux * effective_concentration * kerma_factor
        
        # Phân chia thành các thành phần liều
        alpha_dose = total_dose * 0.63  # 63% là từ hạt alpha
        lithium_dose = total_dose * 0.37  # 37% là từ hạt lithium-7
        
        return alpha_dose, lithium_dose


class BSHModel(BoronDistributionModel):
    """Mô hình phân bố boron cho hợp chất BSH (Sodium borocaptate)."""
    
    def __init__(self, tumor_to_normal_ratio: float = 2.5):
        """
        Khởi tạo mô hình BSH.
        
        Parameters
        ----------
        tumor_to_normal_ratio : float
            Tỷ lệ nồng độ boron trong u so với mô lành
        """
        super().__init__(tumor_to_normal_ratio)
        self.name = "BSH Distribution Model"
        
    def calculate_boron_dose(self, thermal_flux: float, concentration: float, depth: float) -> Tuple[float, float]:
        """
        Tính toán liều boron đặc thù cho BSH.
        
        Parameters
        ----------
        thermal_flux : float
            Thông lượng neutron nhiệt (n/cm²/s)
        concentration : float
            Nồng độ boron trong mô u (ppm)
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        Tuple[float, float]
            Liều alpha và lithium (Gy)
        """
        # Hệ số kerma đặc thù cho BSH
        kerma_factor = 8.97e-14  # Gy·cm²/n/ppm
        
        # Mô hình suy giảm nồng độ theo độ sâu cho BSH
        depth_factor = np.exp(-0.03 * depth)
        effective_concentration = concentration * depth_factor
        
        # Tính liều từ phản ứng bắt neutron
        total_dose = thermal_flux * effective_concentration * kerma_factor
        
        # Phân chia thành các thành phần liều
        alpha_dose = total_dose * 0.59  # 59% là từ hạt alpha
        lithium_dose = total_dose * 0.41  # 41% là từ hạt lithium-7
        
        return alpha_dose, lithium_dose


class BoronophenylalanineModel(BPAModel):
    """Lớp mở rộng cho mô hình BPA với tên đầy đủ."""
    
    def __init__(self, tumor_to_normal_ratio: float = 3.5):
        """
        Khởi tạo mô hình Boronophenylalanine.
        
        Parameters
        ----------
        tumor_to_normal_ratio : float
            Tỷ lệ nồng độ boron trong u so với mô lành
        """
        super().__init__(tumor_to_normal_ratio)
        self.name = "Boronophenylalanine Distribution Model"


class GenericBoronModel(BoronDistributionModel):
    """Mô hình phân bố boron chung."""
    
    def __init__(self, tumor_to_normal_ratio: float = 3.0):
        """
        Khởi tạo mô hình boron chung.
        
        Parameters
        ----------
        tumor_to_normal_ratio : float
            Tỷ lệ nồng độ boron trong u so với mô lành
        """
        super().__init__(tumor_to_normal_ratio)
        self.name = "Generic Boron Model"


class BoronCompound:
    """Lớp cơ sở đại diện cho các hợp chất boron sử dụng trong BNCT."""
    
    def __init__(self, name: str, concentration: float = 10.0, tumor_to_blood_ratio: float = 3.5):
        """
        Khởi tạo hợp chất boron.
        
        Parameters
        ----------
        name : str
            Tên hợp chất
        concentration : float
            Nồng độ boron trong máu (ppm)
        tumor_to_blood_ratio : float
            Tỷ lệ nồng độ boron trong u so với máu
        """
        self.name = name
        self.concentration = concentration
        self.tumor_to_blood_ratio = tumor_to_blood_ratio
        self.model = None  # Mô hình phân bố boron
    
    def get_tumor_concentration(self) -> float:
        """
        Tính nồng độ boron trong mô u.
        
        Returns
        -------
        float
            Nồng độ boron trong mô u (ppm)
        """
        return self.concentration * self.tumor_to_blood_ratio
    
    def get_distribution_model(self):
        """
        Lấy mô hình phân bố boron.
        
        Returns
        -------
        BoronDistributionModel
            Mô hình phân bố boron
        """
        return self.model
    
    def set_concentration(self, concentration: float):
        """
        Thiết lập nồng độ boron trong máu.
        
        Parameters
        ----------
        concentration : float
            Nồng độ boron trong máu (ppm)
        """
        self.concentration = concentration

class BPA(BoronCompound):
    """Lớp đại diện cho hợp chất boron BPA (Boronophenylalanine)."""
    
    def __init__(self, concentration: float = 12.0, tumor_to_blood_ratio: float = 3.5):
        """
        Khởi tạo hợp chất BPA.
        
        Parameters
        ----------
        concentration : float
            Nồng độ boron trong máu (ppm)
        tumor_to_blood_ratio : float
            Tỷ lệ nồng độ boron trong u so với máu
        """
        super().__init__("Boronophenylalanine (BPA)", concentration, tumor_to_blood_ratio)
        self.compound_type = BoronCompoundType.BPA
        # Khởi tạo mô hình phân bố boron
        self.model = BPAModel(tumor_to_blood_ratio)

class BSH(BoronCompound):
    """Lớp đại diện cho hợp chất boron BSH (Sodium borocaptate)."""
    
    def __init__(self, concentration: float = 12.0, tumor_to_blood_ratio: float = 2.5):
        """
        Khởi tạo hợp chất BSH.
        
        Parameters
        ----------
        concentration : float
            Nồng độ boron trong máu (ppm)
        tumor_to_blood_ratio : float
            Tỷ lệ nồng độ boron trong u so với máu
        """
        super().__init__("Sodium borocaptate (BSH)", concentration, tumor_to_blood_ratio)
        self.compound_type = BoronCompoundType.BSH
        # Khởi tạo mô hình phân bố boron
        self.model = BSHModel(tumor_to_blood_ratio)

# Đảm bảo các lớp được xuất ra ngoài module
__all__ = [
    'BoronCompoundType', 'BoronDistributionModel',
    'BPAModel', 'BSHModel', 'BoronophenylalanineModel', 'GenericBoronModel',
    'BoronCompound', 'BPA', 'BSH'
]
