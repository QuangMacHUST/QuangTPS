#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý máy xạ trị ion carbon (Carbon Ion Therapy Machine).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý các máy xạ trị
ion carbon được sử dụng trong xạ trị ion nặng.
"""

import uuid
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

from quangtps.treatment.machine.accelerator import Accelerator
from quangtps.treatment.machine.machine_specs import MachineSpecification

logger = logging.getLogger(__name__)


class CarbonIonDeliveryTechnique(str, Enum):
    """Enum đại diện cho các kỹ thuật phát tia ion carbon."""
    PENCIL_BEAM_SCANNING = "PBS"
    PASSIVE_SCATTERING = "PASSIVE_SCATTERING"
    RASTER_SCANNING = "RASTER_SCANNING"


class CarbonIonMachine(Accelerator):
    """
    Lớp đại diện cho một máy xạ trị ion carbon.
    
    Lớp này cung cấp các phương thức và thuộc tính cần thiết để
    mô tả một máy xạ trị ion carbon, bao gồm các thông số kỹ thuật và
    các khả năng của máy.
    """
    
    def __init__(self, 
                machine_name: str, 
                manufacturer: str = "Generic", 
                machine_id: Optional[str] = None):
        """
        Khởi tạo một máy xạ trị ion carbon.
        
        Parameters
        ----------
        machine_name : str
            Tên của máy
        manufacturer : str, optional
            Nhà sản xuất máy
        machine_id : str, optional
            ID duy nhất của máy
        """
        super().__init__(machine_name, manufacturer, machine_id)
        
        # Loại máy
        self.machine_type = "CARBON_ION"
        
        # Thông số kỹ thuật của máy
        self.specifications = MachineSpecification()
        
        # Kỹ thuật phát tia
        self.delivery_techniques = []
        
        # Thiết bị giới hạn chùm tia
        self.beam_limiting_devices = {}
        
        # Năng lượng
        self.min_energy = 100.0  # MeV/u
        self.max_energy = 430.0  # MeV/u
        self.energy_layers = []  # Các lớp năng lượng có sẵn
        
        # PBS specific parameters
        self.min_spot_size = 2.0  # mm
        self.max_spot_size = 10.0  # mm
        self.spot_spacing_options = [2.0, 3.0, 5.0]  # mm
        
        # Dải giới hạn xuyên sâu (Range limits)
        self.min_range = 2.0  # cm (water equivalent)
        self.max_range = 30.0  # cm (water equivalent)
        
        # Thông số RBE (Relative Biological Effectiveness)
        self.rbe_model = "LEM"  # Local Effect Model
        self.rbe_value = 3.0  # Giá trị RBE trung bình
        
        # Thông tin LET (Linear Energy Transfer)
        self.track_let = True
        self.let_calculation_method = "DOSE_AVERAGED"  # hoặc "TRACK_AVERAGED"
        
        # Thông tin bổ sung
        self.metadata = {}
    
    def set_energy_range(self, min_energy: float, max_energy: float):
        """
        Thiết lập dải năng lượng cho máy ion carbon.
        
        Parameters
        ----------
        min_energy : float
            Năng lượng tối thiểu (MeV/u)
        max_energy : float
            Năng lượng tối đa (MeV/u)
        """
        self.min_energy = min_energy
        self.max_energy = max_energy
        logger.info(f"Set energy range for {self.machine_name}: {min_energy}-{max_energy} MeV/u")
    
    def add_energy_layer(self, energy: float):
        """
        Thêm lớp năng lượng vào danh sách năng lượng có sẵn.
        
        Parameters
        ----------
        energy : float
            Năng lượng (MeV/u)
        """
        if energy not in self.energy_layers:
            self.energy_layers.append(energy)
            self.energy_layers.sort()
            logger.info(f"Added energy layer {energy} MeV/u to {self.machine_name}")
    
    def add_delivery_technique(self, technique: CarbonIonDeliveryTechnique):
        """
        Thêm kỹ thuật phát tia vào danh sách kỹ thuật có sẵn.
        
        Parameters
        ----------
        technique : CarbonIonDeliveryTechnique
            Kỹ thuật phát tia
        """
        if technique not in self.delivery_techniques:
            self.delivery_techniques.append(technique)
            logger.info(f"Added delivery technique {technique} to {self.machine_name}")
    
    def set_rbe_parameters(self, model: str, value: float):
        """
        Thiết lập các thông số RBE.
        
        Parameters
        ----------
        model : str
            Mô hình RBE (ví dụ: "LEM", "MKM")
        value : float
            Giá trị RBE trung bình
        """
        self.rbe_model = model
        self.rbe_value = value
        logger.info(f"Set RBE parameters for {self.machine_name}: {model}, value={value}")
    
    def set_let_parameters(self, track_let: bool, calculation_method: str):
        """
        Thiết lập các thông số LET.
        
        Parameters
        ----------
        track_let : bool
            Bật/tắt tính toán LET
        calculation_method : str
            Phương pháp tính toán LET ("DOSE_AVERAGED" hoặc "TRACK_AVERAGED")
        """
        self.track_let = track_let
        self.let_calculation_method = calculation_method
        logger.info(f"Set LET parameters for {self.machine_name}: tracking={track_let}, method={calculation_method}")
    
    def set_pbs_parameters(self, min_spot_size: float, max_spot_size: float, spot_spacing_options: List[float]):
        """
        Thiết lập các tham số PBS.
        
        Parameters
        ----------
        min_spot_size : float
            Kích thước spot tối thiểu (mm)
        max_spot_size : float
            Kích thước spot tối đa (mm)
        spot_spacing_options : List[float]
            Danh sách khoảng cách spot có sẵn (mm)
        """
        self.min_spot_size = min_spot_size
        self.max_spot_size = max_spot_size
        self.spot_spacing_options = spot_spacing_options
        logger.info(f"Set PBS parameters for {self.machine_name}")
    
    def set_range_limits(self, min_range: float, max_range: float):
        """
        Thiết lập giới hạn xuyên sâu.
        
        Parameters
        ----------
        min_range : float
            Giới hạn xuyên sâu tối thiểu (cm water equivalent)
        max_range : float
            Giới hạn xuyên sâu tối đa (cm water equivalent)
        """
        self.min_range = min_range
        self.max_range = max_range
        logger.info(f"Set range limits for {self.machine_name}: {min_range}-{max_range} cm WE")
    
    def get_closest_energy_layer(self, desired_energy: float) -> float:
        """
        Lấy lớp năng lượng gần nhất với năng lượng mong muốn.
        
        Parameters
        ----------
        desired_energy : float
            Năng lượng mong muốn (MeV/u)
            
        Returns
        -------
        float
            Lớp năng lượng gần nhất có sẵn
        """
        if not self.energy_layers:
            return min(max(desired_energy, self.min_energy), self.max_energy)
        
        closest_energy = min(self.energy_layers, key=lambda x: abs(x - desired_energy))
        logger.debug(f"Closest energy layer to {desired_energy} MeV/u is {closest_energy} MeV/u")
        return closest_energy
    
    def calculate_range(self, energy: float) -> float:
        """
        Tính toán range (xuyên sâu) dựa trên năng lượng.
        
        Parameters
        ----------
        energy : float
            Năng lượng chùm tia (MeV/u)
            
        Returns
        -------
        float
            Range (cm water equivalent)
        """
        # Approximate formula for carbon ions: R = 0.0022 * (E/12)^1.75
        range_cm = 0.0022 * ((energy / 12) ** 1.75)
        logger.debug(f"Calculated range for {energy} MeV/u: {range_cm} cm WE")
        return range_cm
    
    def calculate_energy_from_range(self, desired_range: float) -> float:
        """
        Tính toán năng lượng cần thiết để đạt được range mong muốn.
        
        Parameters
        ----------
        desired_range : float
            Range mong muốn (cm water equivalent)
            
        Returns
        -------
        float
            Năng lượng cần thiết (MeV/u)
        """
        # Inverse of the formula: E = 12 * (R/0.0022)^(1/1.75)
        energy = 12 * ((desired_range / 0.0022) ** (1 / 1.75))
        logger.debug(f"Calculated energy for range {desired_range} cm WE: {energy} MeV/u")
        return energy
    
    def calculate_rbe_at_position(self, energy: float, depth: float, dose: float) -> float:
        """
        Tính toán RBE tại một vị trí xác định.
        
        Parameters
        ----------
        energy : float
            Năng lượng ban đầu (MeV/u)
        depth : float
            Độ sâu (cm water equivalent)
        dose : float
            Liều tại vị trí (Gy)
            
        Returns
        -------
        float
            Giá trị RBE
        """
        # Simplified RBE calculation (in practice this would use tables or more complex models)
        if self.rbe_model == "LEM":
            # LEM model - RBE increases with depth and decreases with dose
            relative_depth = depth / self.calculate_range(energy)
            rbe = self.rbe_value * (1 + 0.5 * relative_depth) * np.exp(-0.06 * dose)
            return min(max(rbe, 1.0), 5.0)  # Limit to reasonable range
        else:
            # Fallback to constant RBE
            return self.rbe_value
    
    def calculate_let(self, energy: float, depth: float) -> float:
        """
        Tính toán LET (Linear Energy Transfer) tại một vị trí xác định.
        
        Parameters
        ----------
        energy : float
            Năng lượng ban đầu (MeV/u)
        depth : float
            Độ sâu (cm water equivalent)
            
        Returns
        -------
        float
            Giá trị LET (keV/µm)
        """
        if not self.track_let:
            return 0.0
        
        # Simplified LET calculation
        range_cm = self.calculate_range(energy)
        relative_depth = depth / range_cm
        
        # LET increases toward the end of range (Bragg peak)
        if relative_depth < 0.7:
            let = 10 + 20 * relative_depth
        else:
            # Sharp increase near Bragg peak
            let = 24 + 200 * (relative_depth - 0.7)**2
        
        # If we're beyond the range, LET drops sharply
        if relative_depth > 1.0:
            let = let * np.exp(-(relative_depth - 1.0) * 10)
        
        logger.debug(f"Calculated LET at depth {depth} cm: {let} keV/µm")
        return let
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin máy ion carbon thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin máy ion carbon
        """
        base_dict = super().to_dict()
        
        carbon_dict = {
            "machine_type": self.machine_type,
            "specifications": self.specifications.to_dict(),
            "delivery_techniques": [dt for dt in self.delivery_techniques],
            "beam_limiting_devices": self.beam_limiting_devices,
            "min_energy": self.min_energy,
            "max_energy": self.max_energy,
            "energy_layers": self.energy_layers,
            "min_spot_size": self.min_spot_size,
            "max_spot_size": self.max_spot_size,
            "spot_spacing_options": self.spot_spacing_options,
            "min_range": self.min_range,
            "max_range": self.max_range,
            "rbe_model": self.rbe_model,
            "rbe_value": self.rbe_value,
            "track_let": self.track_let,
            "let_calculation_method": self.let_calculation_method
        }
        
        # Merge dictionaries
        base_dict.update(carbon_dict)
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CarbonIonMachine':
        """
        Tạo đối tượng CarbonIonMachine từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin máy ion carbon
            
        Returns
        -------
        CarbonIonMachine
            Đối tượng CarbonIonMachine
        """
        machine = cls(
            machine_name=data["machine_name"],
            manufacturer=data["manufacturer"],
            machine_id=data["machine_id"]
        )
        
        # Set specifications
        if "specifications" in data:
            machine.specifications = MachineSpecification.from_dict(data["specifications"])
        
        # Set delivery techniques
        if "delivery_techniques" in data:
            for technique in data["delivery_techniques"]:
                machine.add_delivery_technique(technique)
        
        # Set beam limiting devices
        if "beam_limiting_devices" in data:
            machine.beam_limiting_devices = data["beam_limiting_devices"]
        
        # Set energy parameters
        if "min_energy" in data and "max_energy" in data:
            machine.set_energy_range(data["min_energy"], data["max_energy"])
        
        if "energy_layers" in data:
            for energy in data["energy_layers"]:
                machine.add_energy_layer(energy)
        
        # Set PBS parameters
        if all(k in data for k in ["min_spot_size", "max_spot_size", "spot_spacing_options"]):
            machine.set_pbs_parameters(
                data["min_spot_size"],
                data["max_spot_size"],
                data["spot_spacing_options"]
            )
        
        # Set range limits
        if "min_range" in data and "max_range" in data:
            machine.set_range_limits(data["min_range"], data["max_range"])
        
        # Set RBE parameters
        if "rbe_model" in data and "rbe_value" in data:
            machine.set_rbe_parameters(data["rbe_model"], data["rbe_value"])
        
        # Set LET parameters
        if "track_let" in data and "let_calculation_method" in data:
            machine.set_let_parameters(data["track_let"], data["let_calculation_method"])
        
        # Set metadata
        if "metadata" in data:
            machine.metadata = data["metadata"]
        
        return machine
