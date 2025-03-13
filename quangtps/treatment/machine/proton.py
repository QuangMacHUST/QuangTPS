#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý máy xạ trị proton (Proton Therapy Machine).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý các máy xạ trị
proton được sử dụng trong xạ trị, bao gồm cả PBS (Pencil Beam Scanning) và
Passive Scattering.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

from quangtps.treatment.machine.accelerator import Accelerator
from quangtps.treatment.machine.machine_specs import MachineSpecification

logger = logging.getLogger(__name__)


class ProtonBeamDeliveryTechnique(str, Enum):
    """Enum đại diện cho các kỹ thuật phát tia proton."""
    PENCIL_BEAM_SCANNING = "PBS"
    PASSIVE_SCATTERING = "PASSIVE_SCATTERING"
    UNIFORM_SCANNING = "UNIFORM_SCANNING"


class BeamLimitingDeviceType(str, Enum):
    """Enum đại diện cho các loại thiết bị giới hạn chùm tia cho máy proton."""
    APERTURE = "APERTURE"
    RANGE_SHIFTER = "RANGE_SHIFTER"
    RANGE_MODULATOR = "RANGE_MODULATOR"
    COMPENSATOR = "COMPENSATOR"


class ProtonMachine(Accelerator):
    """
    Lớp đại diện cho một máy xạ trị proton.
    
    Lớp này cung cấp các phương thức và thuộc tính cần thiết để
    mô tả một máy xạ trị proton, bao gồm các thông số kỹ thuật và
    các khả năng của máy.
    """
    
    def __init__(self, 
                machine_name: str, 
                manufacturer: str = "Generic", 
                machine_id: Optional[str] = None):
        """
        Khởi tạo một máy xạ trị proton.
        
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
        self.machine_type = "PROTON"
        
        # Thông số kỹ thuật của máy
        self.specifications = MachineSpecification()
        
        # Kỹ thuật phát tia
        self.delivery_techniques = []
        
        # Thiết bị giới hạn chùm tia
        self.beam_limiting_devices = {}
        
        # Năng lượng
        self.min_energy = 70.0  # MeV
        self.max_energy = 230.0  # MeV
        self.energy_layers = []  # Các lớp năng lượng có sẵn
        
        # PBS specific parameters
        self.min_spot_size = 3.0  # mm
        self.max_spot_size = 15.0  # mm
        self.spot_spacing_options = [2.0, 3.0, 5.0]  # mm
        
        # Passive scattering specific parameters
        self.available_range_shifters = []
        self.available_range_modulators = []
        
        # Dải giới hạn xuyên sâu (Range limits)
        self.min_range = 3.0  # cm (water equivalent)
        self.max_range = 38.0  # cm (water equivalent)
        
        # Thông tin bổ sung
        self.metadata = {}
    
    def set_energy_range(self, min_energy: float, max_energy: float):
        """
        Thiết lập dải năng lượng cho máy proton.
        
        Parameters
        ----------
        min_energy : float
            Năng lượng tối thiểu (MeV)
        max_energy : float
            Năng lượng tối đa (MeV)
        """
        self.min_energy = min_energy
        self.max_energy = max_energy
        logger.info(f"Set energy range for {self.machine_name}: {min_energy}-{max_energy} MeV")
    
    def add_energy_layer(self, energy: float):
        """
        Thêm lớp năng lượng vào danh sách năng lượng có sẵn.
        
        Parameters
        ----------
        energy : float
            Năng lượng (MeV)
        """
        if energy not in self.energy_layers:
            self.energy_layers.append(energy)
            self.energy_layers.sort()
            logger.info(f"Added energy layer {energy} MeV to {self.machine_name}")
    
    def add_delivery_technique(self, technique: ProtonBeamDeliveryTechnique):
        """
        Thêm kỹ thuật phát tia vào danh sách kỹ thuật có sẵn.
        
        Parameters
        ----------
        technique : ProtonBeamDeliveryTechnique
            Kỹ thuật phát tia
        """
        if technique not in self.delivery_techniques:
            self.delivery_techniques.append(technique)
            logger.info(f"Added delivery technique {technique} to {self.machine_name}")
    
    def add_beam_limiting_device(self, device_type: BeamLimitingDeviceType, device_info: Dict[str, Any]):
        """
        Thêm thiết bị giới hạn chùm tia.
        
        Parameters
        ----------
        device_type : BeamLimitingDeviceType
            Loại thiết bị
        device_info : Dict[str, Any]
            Thông tin về thiết bị
        """
        if device_type not in self.beam_limiting_devices:
            self.beam_limiting_devices[device_type] = []
        
        self.beam_limiting_devices[device_type].append(device_info)
        logger.info(f"Added {device_type} to {self.machine_name}")
    
    def add_range_shifter(self, name: str, wer: float, thickness: float):
        """
        Thêm range shifter vào danh sách thiết bị có sẵn.
        
        Parameters
        ----------
        name : str
            Tên của range shifter
        wer : float
            Water Equivalent Ratio
        thickness : float
            Độ dày (mm)
        """
        range_shifter = {
            "name": name,
            "wer": wer,
            "thickness": thickness,
            "id": str(uuid.uuid4())
        }
        self.available_range_shifters.append(range_shifter)
        logger.info(f"Added range shifter {name} to {self.machine_name}")
        
        # Also add to beam limiting devices
        self.add_beam_limiting_device(
            BeamLimitingDeviceType.RANGE_SHIFTER, 
            range_shifter
        )
    
    def add_range_modulator(self, name: str, modulation_width: float, steps: int):
        """
        Thêm range modulator vào danh sách thiết bị có sẵn.
        
        Parameters
        ----------
        name : str
            Tên của range modulator
        modulation_width : float
            Độ rộng điều biến (mm)
        steps : int
            Số bước điều biến
        """
        range_modulator = {
            "name": name,
            "modulation_width": modulation_width,
            "steps": steps,
            "id": str(uuid.uuid4())
        }
        self.available_range_modulators.append(range_modulator)
        logger.info(f"Added range modulator {name} to {self.machine_name}")
        
        # Also add to beam limiting devices
        self.add_beam_limiting_device(
            BeamLimitingDeviceType.RANGE_MODULATOR, 
            range_modulator
        )
    
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
            Năng lượng mong muốn (MeV)
            
        Returns
        -------
        float
            Lớp năng lượng gần nhất có sẵn
        """
        if not self.energy_layers:
            return min(max(desired_energy, self.min_energy), self.max_energy)
        
        closest_energy = min(self.energy_layers, key=lambda x: abs(x - desired_energy))
        logger.debug(f"Closest energy layer to {desired_energy} MeV is {closest_energy} MeV")
        return closest_energy
    
    def calculate_range(self, energy: float) -> float:
        """
        Tính toán range (xuyên sâu) dựa trên năng lượng.
        
        Parameters
        ----------
        energy : float
            Năng lượng chùm tia (MeV)
            
        Returns
        -------
        float
            Range (cm water equivalent)
        """
        # Approximate formula: R = 0.0022 * E^1.77
        range_cm = 0.0022 * (energy ** 1.77)
        logger.debug(f"Calculated range for {energy} MeV: {range_cm} cm WE")
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
            Năng lượng cần thiết (MeV)
        """
        # Inverse of the approximate formula: E = (R/0.0022)^(1/1.77)
        energy = (desired_range / 0.0022) ** (1 / 1.77)
        logger.debug(f"Calculated energy for range {desired_range} cm WE: {energy} MeV")
        return energy
    
    def get_available_range_shifters(self, energy: float = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách range shifter phù hợp với năng lượng.
        
        Parameters
        ----------
        energy : float, optional
            Năng lượng chùm tia (MeV)
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các range shifter phù hợp
        """
        if energy is None:
            return self.available_range_shifters
        
        # Filter range shifters based on energy
        range_cm = self.calculate_range(energy)
        return [rs for rs in self.available_range_shifters 
                if range_cm - (rs["thickness"] * rs["wer"] / 10) >= self.min_range]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin máy proton thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin máy proton
        """
        base_dict = super().to_dict()
        
        proton_dict = {
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
            "available_range_shifters": self.available_range_shifters,
            "available_range_modulators": self.available_range_modulators,
            "min_range": self.min_range,
            "max_range": self.max_range
        }
        
        # Merge dictionaries
        base_dict.update(proton_dict)
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtonMachine':
        """
        Tạo đối tượng ProtonMachine từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin máy proton
            
        Returns
        -------
        ProtonMachine
            Đối tượng ProtonMachine
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
        
        # Add range shifters
        if "available_range_shifters" in data:
            for rs in data["available_range_shifters"]:
                machine.available_range_shifters.append(rs)
        
        # Add range modulators
        if "available_range_modulators" in data:
            for rm in data["available_range_modulators"]:
                machine.available_range_modulators.append(rm)
        
        # Set metadata
        if "metadata" in data:
            machine.metadata = data["metadata"]
        
        return machine
