#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho kỹ thuật xạ trị bắt neutron boron (Boron Neutron Capture Therapy - BNCT).

Module này cung cấp các lớp và phương thức để mô phỏng và thực hiện
kỹ thuật xạ trị BNCT, một phương pháp điều trị sử dụng phản ứng hạt nhân
giữa boron-10 và neutron nhiệt để tạo ra các hạt alpha có khả năng phá hủy
tế bào ung thư một cách chọn lọc.
"""

import logging
from enum import Enum
from typing import Dict, Optional, Any, List, Union, Tuple
import uuid
import matplotlib.pyplot as plt
import numpy as np

# Import từ các module chuyên biệt
from quangtps.specialized.bnct.neutron import (
    NeutronSourceType, NeutronSource as SpecializedNeutronSource,
    ReactorSource, AcceleratorSource, CyclotronSource, NeutronInteraction
)
from quangtps.specialized.bnct.boron import (
    BoronCompoundType, BoronCompoundProperties, BoronDistributionModel, TwoCompartmentModel
)

logger = logging.getLogger(__name__)

class BoronCompound(str, Enum):
    """Enum đại diện cho các hợp chất boron sử dụng trong BNCT."""
    BPA = "BPA"          # Boronophenylalanine
    BSH = "BSH"          # Sodium borocaptate
    CUSTOM = "CUSTOM"    # Hợp chất tùy chỉnh

class NeutronSource(str, Enum):
    """Enum đại diện cho các nguồn neutron sử dụng trong BNCT."""
    REACTOR = "REACTOR"              # Lò phản ứng hạt nhân
    ACCELERATOR = "ACCELERATOR"      # Máy gia tốc
    CYCLOTRON = "CYCLOTRON"          # Cyclotron

class BNCT:
    """
    Lớp đại diện cho kỹ thuật xạ trị bắt neutron boron (BNCT).
    
    Lớp này cung cấp các phương thức để thiết lập và mô phỏng
    kỹ thuật xạ trị BNCT, một phương pháp điều trị đặc biệt sử dụng
    phản ứng hạt nhân để điều trị ung thư.
    """
    
    def __init__(self, 
                 bnct_id: Optional[str] = None,
                 name: str = "Default BNCT",
                 boron_compound: BoronCompound = BoronCompound.BPA,
                 neutron_source: NeutronSource = NeutronSource.ACCELERATOR,
                 boron_concentration: float = 20.0):
        """
        Khởi tạo một đối tượng BNCT.
        
        Parameters
        ----------
        bnct_id : str, optional
            ID duy nhất cho kỹ thuật BNCT
        name : str, optional
            Tên mô tả cho kỹ thuật BNCT
        boron_compound : BoronCompound, optional
            Hợp chất boron sử dụng
        neutron_source : NeutronSource, optional
            Nguồn neutron sử dụng
        boron_concentration : float, optional
            Nồng độ boron trong mô (ppm)
        """
        self.bnct_id = bnct_id if bnct_id else str(uuid.uuid4())
        self.name = name
        self.boron_compound = boron_compound
        self.neutron_source = neutron_source
        self.boron_concentration = boron_concentration
        self.beam_parameters = {}
        self.dose_components = {
            "boron_dose": 0.0,
            "gamma_dose": 0.0,
            "fast_neutron_dose": 0.0,
            "thermal_neutron_dose": 0.0
        }
        self.cbr = 3.8  # Compound Biological Effectiveness (mặc định cho BPA)
        
        # Khởi tạo các đối tượng từ module chuyên biệt
        self._initialize_specialized_objects()
        
    def _initialize_specialized_objects(self):
        """Khởi tạo các đối tượng từ module chuyên biệt BNCT."""
        # Khởi tạo mô hình phân bố boron
        boron_type = BoronCompoundType.BPA if self.boron_compound == BoronCompound.BPA else \
                    BoronCompoundType.BSH if self.boron_compound == BoronCompound.BSH else \
                    BoronCompoundType.CUSTOM
        self.boron_model = TwoCompartmentModel(compound_type=boron_type)
        
        # Khởi tạo nguồn neutron dựa trên loại nguồn
        source_type = NeutronSourceType.REACTOR if self.neutron_source == NeutronSource.REACTOR else \
                     NeutronSourceType.ACCELERATOR if self.neutron_source == NeutronSource.ACCELERATOR else \
                     NeutronSourceType.CYCLOTRON
        
        if source_type == NeutronSourceType.REACTOR:
            self.specialized_neutron_source = ReactorSource(name=f"{self.name} - Reactor Source")
        elif source_type == NeutronSourceType.ACCELERATOR:
            self.specialized_neutron_source = AcceleratorSource(name=f"{self.name} - Accelerator Source")
        else:  # CYCLOTRON
            self.specialized_neutron_source = CyclotronSource(name=f"{self.name} - Cyclotron Source")
        
        # Khởi tạo mô hình tương tác neutron
        self.neutron_interaction = NeutronInteraction()
        
    def set_boron_compound(self, compound: BoronCompound, concentration: float = None):
        """
        Thiết lập hợp chất boron sử dụng trong điều trị.
        
        Parameters
        ----------
        compound : BoronCompound
            Hợp chất boron sử dụng
        concentration : float, optional
            Nồng độ boron trong mô (ppm)
        
        Returns
        -------
        self : BNCT
            Đối tượng BNCT hiện tại
        """
        self.boron_compound = compound
        
        # Cập nhật CBR dựa trên hợp chất
        if compound == BoronCompound.BPA:
            self.cbr = 3.8
        elif compound == BoronCompound.BSH:
            self.cbr = 2.5
        
        if concentration is not None:
            self.boron_concentration = concentration
        
        # Cập nhật mô hình boron
        boron_type = BoronCompoundType.BPA if compound == BoronCompound.BPA else \
                    BoronCompoundType.BSH if compound == BoronCompound.BSH else \
                    BoronCompoundType.CUSTOM
        self.boron_model = TwoCompartmentModel(compound_type=boron_type)
            
        return self
    
    def set_neutron_source(self, source: NeutronSource):
        """
        Thiết lập nguồn neutron sử dụng trong điều trị.
        
        Parameters
        ----------
        source : NeutronSource
            Nguồn neutron sử dụng
        
        Returns
        -------
        self : BNCT
            Đối tượng BNCT hiện tại
        """
        self.neutron_source = source
        
        # Cập nhật nguồn neutron chuyên biệt
        source_type = NeutronSourceType.REACTOR if source == NeutronSource.REACTOR else \
                     NeutronSourceType.ACCELERATOR if source == NeutronSource.ACCELERATOR else \
                     NeutronSourceType.CYCLOTRON
        
        if source_type == NeutronSourceType.REACTOR:
            self.specialized_neutron_source = ReactorSource(name=f"{self.name} - Reactor Source")
        elif source_type == NeutronSourceType.ACCELERATOR:
            self.specialized_neutron_source = AcceleratorSource(name=f"{self.name} - Accelerator Source")
        else:  # CYCLOTRON
            self.specialized_neutron_source = CyclotronSource(name=f"{self.name} - Cyclotron Source")
        
        return self
    
    def set_beam_parameters(self, parameters: Dict[str, Any]):
        """
        Thiết lập các tham số chùm tia neutron.
        
        Parameters
        ----------
        parameters : Dict[str, Any]
            Từ điển chứa các tham số chùm tia
        
        Returns
        -------
        self : BNCT
            Đối tượng BNCT hiện tại
        """
        self.beam_parameters.update(parameters)
        
        # Cập nhật thông lượng neutron trong nguồn chuyên biệt nếu có
        if "thermal_flux" in parameters and hasattr(self, "specialized_neutron_source"):
            from quangtps.specialized.bnct.neutron import NeutronEnergyGroup
            self.specialized_neutron_source.set_flux(NeutronEnergyGroup.THERMAL, parameters["thermal_flux"])
        
        if "epithermal_flux" in parameters and hasattr(self, "specialized_neutron_source"):
            from quangtps.specialized.bnct.neutron import NeutronEnergyGroup
            self.specialized_neutron_source.set_flux(NeutronEnergyGroup.EPITHERMAL, parameters["epithermal_flux"])
            
        if "fast_flux" in parameters and hasattr(self, "specialized_neutron_source"):
            from quangtps.specialized.bnct.neutron import NeutronEnergyGroup
            self.specialized_neutron_source.set_flux(NeutronEnergyGroup.FAST, parameters["fast_flux"])
        
        return self
    
    def calculate_dose_components(self, tumor_boron_concentration: float = None,
                                normal_boron_concentration: float = None,
                                time_after_injection: float = 2.0) -> Dict[str, Dict[str, float]]:
        """
        Tính toán các thành phần liều cho mô u và mô lành.
        
        Parameters
        ----------
        tumor_boron_concentration : float, optional
            Nồng độ boron trong mô u (ppm)
        normal_boron_concentration : float, optional
            Nồng độ boron trong mô lành (ppm)
        time_after_injection : float, optional
            Thời gian sau khi tiêm (giờ)
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Từ điển chứa các thành phần liều cho mô u và mô lành
        """
        # Sử dụng nồng độ mặc định nếu không được cung cấp
        if tumor_boron_concentration is None:
            if self.boron_compound == BoronCompound.BPA:
                tumor_boron_concentration = 65.0  # ppm, giá trị điển hình cho BPA
            elif self.boron_compound == BoronCompound.BSH:
                tumor_boron_concentration = 50.0  # ppm, giá trị điển hình cho BSH
            else:
                tumor_boron_concentration = 40.0  # ppm, giá trị mặc định
        
        if normal_boron_concentration is None:
            if self.boron_compound == BoronCompound.BPA:
                normal_boron_concentration = 18.0  # ppm, giá trị điển hình cho BPA
            elif self.boron_compound == BoronCompound.BSH:
                normal_boron_concentration = 12.0  # ppm, giá trị điển hình cho BSH
            else:
                normal_boron_concentration = 10.0  # ppm, giá trị mặc định
        
        # Lấy thông lượng neutron từ tham số chùm tia
        thermal_flux = self.beam_parameters.get("thermal_flux", 1.0e9)  # n/cm²/s
        epithermal_flux = self.beam_parameters.get("epithermal_flux", 5.0e9)  # n/cm²/s
        fast_flux = self.beam_parameters.get("fast_flux", 1.0e8)  # n/cm²/s
        irradiation_time = self.beam_parameters.get("irradiation_time", 3600)  # seconds
        
        # Hệ số chuyển đổi từ thông lượng neutron sang liều
        # Các giá trị này phụ thuộc vào nhiều yếu tố và có thể được điều chỉnh
        boron_dose_factor = 3.8e-11  # Gy/(ppm * n/cm²)
        gamma_dose_factor = 2.0e-13  # Gy/(n/cm²)
        fast_neutron_dose_factor = 1.0e-12  # Gy/(n/cm²)
        thermal_neutron_dose_factor = 5.0e-13  # Gy/(n/cm²)
        
        # Tính toán các thành phần liều cho mô u
        tumor_doses = {}
        tumor_doses["boron_dose"] = boron_dose_factor * tumor_boron_concentration * thermal_flux * irradiation_time
        tumor_doses["gamma_dose"] = gamma_dose_factor * (thermal_flux + 0.5 * epithermal_flux) * irradiation_time
        tumor_doses["fast_neutron_dose"] = fast_neutron_dose_factor * fast_flux * irradiation_time
        tumor_doses["thermal_neutron_dose"] = thermal_neutron_dose_factor * thermal_flux * irradiation_time
        
        # Tính toán các thành phần liều cho mô lành
        normal_doses = {}
        normal_doses["boron_dose"] = boron_dose_factor * normal_boron_concentration * thermal_flux * irradiation_time
        normal_doses["gamma_dose"] = gamma_dose_factor * (thermal_flux + 0.5 * epithermal_flux) * irradiation_time
        normal_doses["fast_neutron_dose"] = fast_neutron_dose_factor * fast_flux * irradiation_time
        normal_doses["thermal_neutron_dose"] = thermal_neutron_dose_factor * thermal_flux * irradiation_time
        
        return {"tumor": tumor_doses, "normal": normal_doses}
    
    def calculate_biologically_weighted_dose(self, dose_components: Dict[str, float]) -> Dict[str, float]:
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
        # Import RBEModel từ module rbe_analysis
        from quangtps.specialized.bnct.rbe_analysis import RBEModel, RBEFactors
        
        # Tạo mô hình RBE dựa trên hợp chất boron
        compound_name = "BPA" if self.boron_compound == BoronCompound.BPA else \
                       "BSH" if self.boron_compound == BoronCompound.BSH else \
                       "CUSTOM"
        
        # Tạo đối tượng RBEModel
        rbe_model = RBEModel(compound_name=compound_name)
        
        # Tính toán liều sinh học có trọng số
        weighted_doses = rbe_model.calculate_weighted_dose(dose_components)
        
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
        # Import RBEModel từ module rbe_analysis
        from quangtps.specialized.bnct.rbe_analysis import RBEModel
        
        # Tạo mô hình RBE dựa trên hợp chất boron
        compound_name = "BPA" if self.boron_compound == BoronCompound.BPA else \
                       "BSH" if self.boron_compound == BoronCompound.BSH else \
                       "CUSTOM"
        
        # Tạo đối tượng RBEModel
        rbe_model = RBEModel(compound_name=compound_name)
        
        # Tính toán tỷ lệ điều trị
        therapeutic_ratio = rbe_model.calculate_therapeutic_ratio(
            tumor_dose_components, normal_dose_components)
        
        return therapeutic_ratio
    
    def plot_dose_components(self, tumor_dose_components: Dict[str, float],
                           normal_dose_components: Dict[str, float],
                           tumor_weighted_doses: Dict[str, float],
                           normal_weighted_doses: Dict[str, float]) -> plt.Figure:
        """
        Vẽ đồ thị các thành phần liều cho mô u và mô lành.
        
        Parameters
        ----------
        tumor_dose_components : Dict[str, float]
            Từ điển chứa các thành phần liều vật lý cho mô u (Gy)
        normal_dose_components : Dict[str, float]
            Từ điển chứa các thành phần liều vật lý cho mô lành (Gy)
        tumor_weighted_doses : Dict[str, float]
            Từ điển chứa các thành phần liều sinh học có trọng số cho mô u (Gy-Eq)
        normal_weighted_doses : Dict[str, float]
            Từ điển chứa các thành phần liều sinh học có trọng số cho mô lành (Gy-Eq)
            
        Returns
        -------
        plt.Figure
            Đối tượng Figure chứa đồ thị các thành phần liều
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Danh sách các thành phần liều
        components = ["boron_dose", "gamma_dose", "fast_neutron_dose", "thermal_neutron_dose"]
        component_labels = ["Boron", "Gamma", "Fast Neutron", "Thermal Neutron"]
        colors = ["blue", "green", "red", "cyan"]
        
        # Vẽ biểu đồ cột cho liều vật lý
        tumor_physical = [tumor_dose_components.get(comp, 0.0) for comp in components]
        normal_physical = [normal_dose_components.get(comp, 0.0) for comp in components]
        
        x = np.arange(len(components))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, tumor_physical, width, label="Tumor", color=colors)
        axes[0, 0].bar(x + width/2, normal_physical, width, label="Normal Tissue", color=colors, alpha=0.5)
        axes[0, 0].set_title("Physical Dose Components")
        axes[0, 0].set_ylabel("Dose (Gy)")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(component_labels)
        axes[0, 0].legend()
        
        # Vẽ biểu đồ cột cho liều sinh học có trọng số
        weighted_components = ["weighted_boron_dose", "weighted_gamma_dose", 
                             "weighted_fast_neutron_dose", "weighted_thermal_neutron_dose"]
        
        tumor_weighted = [tumor_weighted_doses.get(comp, 0.0) for comp in weighted_components]
        normal_weighted = [normal_weighted_doses.get(comp, 0.0) for comp in weighted_components]
        
        axes[0, 1].bar(x - width/2, tumor_weighted, width, label="Tumor", color=colors)
        axes[0, 1].bar(x + width/2, normal_weighted, width, label="Normal Tissue", color=colors, alpha=0.5)
        axes[0, 1].set_title("Biologically Weighted Dose Components")
        axes[0, 1].set_ylabel("Dose (Gy-Eq)")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(component_labels)
        axes[0, 1].legend()
        
        # Vẽ biểu đồ tròn cho tỷ lệ các thành phần liều vật lý
        axes[1, 0].pie(tumor_physical, labels=component_labels, colors=colors, autopct="%1.1f%%",
                      startangle=90)
        axes[1, 0].set_title("Tumor Physical Dose Distribution")
        
        # Vẽ biểu đồ tròn cho tỷ lệ các thành phần liều sinh học có trọng số
        axes[1, 1].pie(tumor_weighted, labels=component_labels, colors=colors, autopct="%1.1f%%",
                      startangle=90)
        axes[1, 1].set_title("Tumor Biologically Weighted Dose Distribution")
        
        plt.tight_layout()
        
        return fig

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng BNCT thành từ điển.
        
        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin của đối tượng BNCT
        """
        return {
            "bnct_id": self.bnct_id,
            "name": self.name,
            "boron_compound": self.boron_compound,
            "neutron_source": self.neutron_source,
            "boron_concentration": self.boron_concentration,
            "beam_parameters": self.beam_parameters,
            "dose_components": self.dose_components,
            "cbr": self.cbr
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BNCT':
        """
        Tạo đối tượng BNCT từ từ điển.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin của đối tượng BNCT
        
        Returns
        -------
        BNCT
            Đối tượng BNCT mới
        """
        bnct = cls(
            bnct_id=data.get("bnct_id"),
            name=data.get("name", "Default BNCT"),
            boron_compound=data.get("boron_compound", BoronCompound.BPA),
            neutron_source=data.get("neutron_source", NeutronSource.ACCELERATOR),
            boron_concentration=data.get("boron_concentration", 20.0)
        )
        
        if "beam_parameters" in data:
            bnct.beam_parameters = data["beam_parameters"]
            
        if "dose_components" in data:
            bnct.dose_components = data["dose_components"]
            
        if "cbr" in data:
            bnct.cbr = data["cbr"]
            
        return bnct