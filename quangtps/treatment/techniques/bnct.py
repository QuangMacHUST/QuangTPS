#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý kỹ thuật xạ trị bắt neutron boron (BNCT - Boron Neutron Capture Therapy).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý các kế hoạch
điều trị BNCT, bao gồm việc thiết lập nguồn neutron, mô hình boron, và tính toán liều sinh học.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any

from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory
from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation
from quangtps.specialized.bnct.neutron import NeutronSourceType, BaseNeutronModel, AcceleratorNeutronModel
from quangtps.specialized.bnct.neutron import ReactorNeutronModel, DDGeneratorModel, DTGeneratorModel, GenericNeutronModel
from quangtps.specialized.bnct.boron import BoronDistributionModel, BoronCompoundType, TwoCompartmentModel

logger = logging.getLogger(__name__)


class NeutronSource(str, Enum):
    """Enum đại diện cho các loại nguồn neutron."""
    ACCELERATOR = "ACCELERATOR"  # Máy gia tốc
    REACTOR = "REACTOR"  # Lò phản ứng
    DD_GENERATOR = "DD_GENERATOR"  # Máy phát neutron D-D
    DT_GENERATOR = "DT_GENERATOR"  # Máy phát neutron D-T


class BoronCompound(str, Enum):
    """Enum đại diện cho các loại hợp chất boron."""
    BPA = "BPA"  # Boronophenylalanine
    BSH = "BSH"  # Sodium borocaptate
    BORONOPHENYLALANINE = "BORONOPHENYLALANINE"  # Boronophenylalanine (tên đầy đủ)


class BNCT(BaseTreatmentTechnique):
    """
    Lớp đại diện cho kỹ thuật xạ trị bắt neutron boron (BNCT).
    
    Lớp này cung cấp các phương thức để thiết lập và mô phỏng
    kỹ thuật xạ trị BNCT, một phương pháp điều trị đặc biệt sử dụng
    phản ứng hạt nhân để điều trị ung thư.
    """
    
    def __init__(self, 
                 name: str = "Default BNCT",
                 technique_id: Optional[str] = None,
                 neutron_source_type: str = NeutronSource.ACCELERATOR,
                 boron_compound_type: str = BoronCompound.BPA,
                 boron_concentration: float = 20.0):
        """
        Khởi tạo một đối tượng BNCT.
        
        Parameters
        ----------
        name : str
            Tên của lượt điều trị BNCT
        technique_id : str, optional
            Định danh cho lượt điều trị BNCT
        neutron_source_type : str, optional
            Loại nguồn neutron sử dụng
        boron_compound_type : str, optional
            Hợp chất boron sử dụng
        boron_concentration : float, optional
            Nồng độ boron trong mô (ppm)
        """
        super().__init__(name=name, technique_id=technique_id, category=TechniqueCategory.SPECIAL)
        
        # Thiết lập các tham số BNCT
        self.neutron_source = NeutronSource(neutron_source_type)
        self.boron_compound = BoronCompound(boron_compound_type)
        self.boron_concentration = boron_concentration
        self.irradiation_time = 60.0  # Thời gian chiếu xạ mặc định (phút)
        self.tumor_to_normal_ratio = 3.5  # Tỷ lệ nồng độ boron trong u/mô lành
        
        # Các thành phần liều vật lý (Gy)
        self.physical_dose_components = {
            "neutron_thermal": 0.0,
            "neutron_epithermal": 0.0,
            "neutron_fast": 0.0,
            "gamma": 0.0,
            "alpha": 0.0,
            "lithium": 0.0
        }
        
        # Biến lưu trữ mô hình tính toán
        self._neutron_model = None
        self._boron_model = None
        
        # Khởi tạo mô hình
        self.setup_neutron_source()
        self.setup_boron_model()
        
        # Ghi log khởi tạo với định dạng lazy %
        logger.info(
            "Khởi tạo kế hoạch BNCT '%s' (ID: %s) với nguồn neutron %s, hợp chất boron %s, nồng độ %.2f ppm",
            self.name, self.technique_id, neutron_source_type, boron_compound_type, boron_concentration
        )
    
    def calculate_dose_components(self, depth: float) -> Dict[str, float]:
        """
        Tính toán các thành phần liều tại một độ sâu cụ thể.
        
        Parameters
        ----------
        depth : float
            Độ sâu tính từ bề mặt (cm)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa các thành phần liều
        """
        if not self._neutron_model or not self._boron_model:
            logger.warning(
                "Không thể tính toán liều: Chưa khởi tạo mô hình neutron hoặc boron cho kế hoạch %s",
                self.name
            )
            return {k: 0.0 for k in self.physical_dose_components}
        
        # Tính toán thành phần liều neutron
        thermal_flux = self._neutron_model.calculate_thermal_flux(depth)
        epithermal_flux = self._neutron_model.calculate_epithermal_flux(depth)
        fast_flux = self._neutron_model.calculate_fast_flux(depth)
        
        # Tính toán liều neutron (Gy)
        neutron_thermal_dose = thermal_flux * 3.8e-13  # Hệ số chuyển đổi
        neutron_epithermal_dose = epithermal_flux * 1.6e-13
        neutron_fast_dose = fast_flux * 4.5e-13
        
        # Tính toán liều gamma (Gy)
        gamma_dose = self._neutron_model.calculate_gamma_dose(depth)
        
        # Tính toán liều boron (Gy)
        alpha_dose, lithium_dose = self._boron_model.calculate_boron_dose(
            thermal_flux, self.boron_concentration, depth
        )
        
        # Cập nhật thành phần liều
        self.physical_dose_components = {
            "neutron_thermal": neutron_thermal_dose,
            "neutron_epithermal": neutron_epithermal_dose,
            "neutron_fast": neutron_fast_dose,
            "gamma": gamma_dose,
            "alpha": alpha_dose,
            "lithium": lithium_dose
        }
        
        return self.physical_dose_components
    
    def setup_neutron_source(self) -> None:
        """
        Thiết lập mô hình tính toán cho nguồn neutron.
        """
        source_type = self.neutron_source
        
        if source_type == NeutronSource.ACCELERATOR:
            self._neutron_model = AcceleratorNeutronModel()
        elif source_type == NeutronSource.REACTOR:
            self._neutron_model = ReactorNeutronModel()
        elif source_type == NeutronSource.DD_GENERATOR:
            self._neutron_model = DDGeneratorModel()
        elif source_type == NeutronSource.DT_GENERATOR:
            self._neutron_model = DTGeneratorModel()
        else:
            logger.error(
                "Loại nguồn neutron không hợp lệ cho kế hoạch %s: %s",
                self.name, source_type
            )
            self._neutron_model = GenericNeutronModel()  # Sử dụng mô hình mặc định
            
        logger.info(
            "Đã thiết lập mô hình nguồn neutron %s cho kế hoạch BNCT '%s'",
            source_type, self.name
        )
    
    def setup_boron_model(self) -> None:
        """
        Thiết lập mô hình tính toán cho hợp chất boron.
        """
        compound_type = self.boron_compound
        
        # Chuyển đổi từ enum BoronCompound (cũ) sang BoronCompoundType (mới)
        if compound_type == BoronCompound.BPA:
            boron_type = BoronCompoundType.BPA
        elif compound_type == BoronCompound.BSH:
            boron_type = BoronCompoundType.BSH
        elif compound_type == BoronCompound.BORONOPHENYLALANINE:
            boron_type = BoronCompoundType.BPA
        else:
            boron_type = BoronCompoundType.CUSTOM
            
        # Sử dụng TwoCompartmentModel để mô hình hóa phân bố boron
        self._boron_model = TwoCompartmentModel(
            compound_type=boron_type,
            k12=0.25,  # Hằng số tốc độ từ máu sang mô
            k21=0.15,  # Hằng số tốc độ từ mô sang máu
            k10=0.10   # Hằng số tốc độ thải trừ từ máu
        )
            
        logger.info(
            "Đã thiết lập mô hình hợp chất boron %s cho kế hoạch BNCT '%s' với tỷ lệ u/lành %.2f",
            compound_type, self.name, self.tumor_to_normal_ratio
        )
    
    def set_boron_concentration(self, concentration: float, tumor_to_normal_ratio: Optional[float] = None) -> None:
        """
        Thiết lập nồng độ boron và tỷ lệ nồng độ u/lành.
        
        Parameters
        ----------
        concentration : float
            Nồng độ boron trong mô u (ppm)
        tumor_to_normal_ratio : float, optional
            Tỷ lệ nồng độ boron trong u / mô lành
        """
        if concentration <= 0:
            logger.warning(
                "Nồng độ boron không hợp lệ (%.2f ppm), sử dụng giá trị mặc định 20.0 ppm cho kế hoạch %s",
                concentration, self.name
            )
            concentration = 20.0
            
        self.boron_concentration = concentration
        
        if tumor_to_normal_ratio is not None:
            if tumor_to_normal_ratio <= 1:
                logger.warning(
                    "Tỷ lệ nồng độ u/lành không hợp lệ (%.2f), sử dụng giá trị mặc định 3.5 cho kế hoạch %s",
                    tumor_to_normal_ratio, self.name
                )
                tumor_to_normal_ratio = 3.5
            
            self.tumor_to_normal_ratio = tumor_to_normal_ratio
            
            # Cập nhật mô hình boron nếu đã khởi tạo
            if self._boron_model:
                self._boron_model.tumor_to_blood_ratio = tumor_to_normal_ratio
                
        logger.info(
            "Đã thiết lập nồng độ boron %.2f ppm và tỷ lệ u/lành %.2f cho kế hoạch BNCT '%s'",
            self.boron_concentration, self.tumor_to_normal_ratio, self.name
        )
    
    def set_irradiation_time(self, irradiation_time: float) -> None:
        """
        Thiết lập thời gian chiếu xạ.
        
        Parameters
        ----------
        irradiation_time : float
            Thời gian chiếu xạ (phút)
        """
        if irradiation_time <= 0:
            logger.warning(
                "Thời gian chiếu xạ không hợp lệ (%.2f phút), sử dụng giá trị mặc định 60.0 phút cho kế hoạch %s",
                irradiation_time, self.name
            )
            irradiation_time = 60.0
            
        self.irradiation_time = irradiation_time
        
        logger.info(
            "Đã thiết lập thời gian chiếu xạ %.2f phút cho kế hoạch BNCT '%s'",
            self.irradiation_time, self.name
        )
    
    def generate_standard_beams(self) -> List[Beam]:
        """
        Tạo các chùm tia tiêu chuẩn cho BNCT.
        
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia
        """
        beams = []
        
        if not self.machine:
            logger.warning(
                "Chưa thiết lập máy điều trị, không thể tạo chùm tia cho kế hoạch BNCT '%s'",
                self.name
            )
            return beams
        
        # Tạo chùm tia neutron chính
        main_beam = Beam(beam_name=f"{self.name}_Main")
        main_beam.set_energy(0)  # Neutron không có "năng lượng" theo cách thông thường
        main_beam.geometry.gantry_angle = 0  # Thẳng góc với bệnh nhân
        main_beam.geometry.field_size = (10, 10)  # Trường chiếu 10x10 cm
        main_beam.metadata = {
            "neutron_source": self.neutron_source,
            "boron_compound": self.boron_compound,
            "boron_concentration": self.boron_concentration,
            "irradiation_time": self.irradiation_time
        }
        beams.append(main_beam)
        
        # Tùy thuộc vào vị trí khối u, có thể cần các chùm tia bổ sung
        if self.machine.has_capability("multi_field"):
            # Thêm chùm tia bổ sung nếu cần
            additional_beam = Beam(beam_name=f"{self.name}_Additional")
            additional_beam.set_energy(0)
            additional_beam.geometry.gantry_angle = 90  # Chùm tia bên
            additional_beam.geometry.field_size = (8, 8)
            additional_beam.metadata = main_beam.metadata.copy()
            beams.append(additional_beam)
        
        # Thêm chùm tia vào kế hoạch
        for beam in beams:
            self.add_beam(beam)
            
        logger.info(
            "Đã tạo %d chùm tia tiêu chuẩn cho kế hoạch BNCT '%s'",
            len(beams), self.name
        )
            
        return beams
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Thiết lập máy điều trị cho BNCT.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Máy điều trị
        """
        self.machine = machine
        logger.info("Đã thiết lập máy điều trị %s cho kế hoạch BNCT %s", machine.name, self.name)

    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Thiết lập phân liều cho kế hoạch BNCT.
        
        Parameters
        ----------
        fractionation : Fractionation
            Phương thức phân liều
        """
        self.fractionation = fractionation
        logger.info("Đã thiết lập phân liều cho kế hoạch BNCT %s: %d phân liều, %.2f Gy mỗi phân liều", 
                   self.name, fractionation.num_fractions, fractionation.dose_per_fraction)

    def add_beam(self, beam: Beam) -> None:
        """
        Thêm một chùm tia vào kế hoạch BNCT.
        
        Parameters
        ----------
        beam : Beam
            Chùm tia cần thêm
        """
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info("Đã thêm chùm tia %s vào kế hoạch BNCT %s", beam.beam_id, self.name)

    def get_beams(self) -> List[Beam]:
        """
        Lấy danh sách tất cả các chùm tia trong kế hoạch BNCT.
        
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia
        """
        return self.beams
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng BNCT thành từ điển.
        
        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin BNCT
        """
        data = super().to_dict()
        
        # Thêm các thuộc tính đặc thù của BNCT
        bnct_specific = {
            "boron_compound": self.boron_compound,
            "neutron_source": self.neutron_source,
            "boron_concentration": self.boron_concentration,
            "irradiation_time": self.irradiation_time,
            "tumor_to_normal_ratio": self.tumor_to_normal_ratio,
            "physical_dose_components": self.physical_dose_components
        }
        
        data.update(bnct_specific)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BNCT':
        """
        Tạo đối tượng BNCT từ từ điển.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin BNCT
            
        Returns
        -------
        BNCT
            Đối tượng BNCT
        """
        # Tạo đối tượng cơ sở
        bnct = super(BNCT, cls).from_dict(data)
        
        # Thiết lập các thuộc tính đặc thù của BNCT
        bnct.boron_compound = BoronCompound(data.get("boron_compound", BoronCompound.BPA))
        bnct.neutron_source = NeutronSource(data.get("neutron_source", NeutronSource.ACCELERATOR))
        bnct.boron_concentration = data.get("boron_concentration", 20.0)
        bnct.irradiation_time = data.get("irradiation_time", 60.0)
        bnct.tumor_to_normal_ratio = data.get("tumor_to_normal_ratio", 3.5)
        bnct.physical_dose_components = data.get("physical_dose_components", {
            "neutron_thermal": 0.0,
            "neutron_epithermal": 0.0,
            "neutron_fast": 0.0,
            "gamma": 0.0,
            "alpha": 0.0,
            "lithium": 0.0
        })
        
        # Khởi tạo lại các mô hình
        bnct.setup_neutron_source()
        bnct.setup_boron_model()
        
        return bnct


# Đảm bảo BNCT được xuất ra đúng cách
__all__ = ['BNCT', 'BoronCompound', 'NeutronSource']