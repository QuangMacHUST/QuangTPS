#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý máy gia tốc tuyến tính (Linear Accelerator - Linac).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý các máy gia tốc
tuyến tính được sử dụng trong xạ trị.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

from quangtps.treatment.machine.accelerator import Accelerator
from quangtps.treatment.machine.machine_specs import MachineSpecification
from quangtps.treatment.mlc.mlc_model import MLCModel

logger = logging.getLogger(__name__)


class BeamLimitingDeviceType(str, Enum):
    """Enum đại diện cho các loại thiết bị giới hạn chùm tia."""
    JAW = "JAW"
    MLC = "MLC"
    APPLICATOR = "APPLICATOR"
    CONE = "CONE"


class Linac(Accelerator):
    """
    Lớp đại diện cho một máy gia tốc tuyến tính (Linac).
    
    Lớp này chứa thông tin về một máy Linac, bao gồm thông số kỹ thuật,
    năng lượng chùm, và các thiết bị phụ trợ như MLC.
    """
    
    def __init__(self, 
                machine_name: str, 
                manufacturer: str = "Generic", 
                machine_id: Optional[str] = None):
        """
        Khởi tạo một máy Linac.
        
        Parameters
        ----------
        machine_name : str
            Tên của máy Linac
        manufacturer : str, optional
            Nhà sản xuất của máy Linac
        machine_id : str, optional
            ID duy nhất của máy Linac. Nếu không cung cấp, một ID mới sẽ được tạo.
        """
        super().__init__(machine_name, manufacturer, machine_id)
        self.accelerator_type = "LINAC"
        
        # Các chùm photon và electron có sẵn
        self.photon_energies = []  # MV
        self.electron_energies = []  # MeV
        
        # Thiết bị giới hạn chùm
        self.beam_limiting_devices = {}
        
        # Độ chuẩn cho từng năng lượng (Output factor)
        self.output_factors = {}
        
        # MLC
        self.mlc_model = None
        
        # Thông số kỹ thuật
        self.specs = MachineSpecification()
        
    def add_photon_energy(self, energy: float, output_factor: float = 1.0):
        """
        Thêm năng lượng photon cho máy Linac.
        
        Parameters
        ----------
        energy : float
            Năng lượng photon (MV)
        output_factor : float, optional
            Hệ số đầu ra cho năng lượng này, mặc định là 1.0
        """
        if energy not in self.photon_energies:
            self.photon_energies.append(energy)
            self.output_factors[f"PHOTON_{energy}MV"] = output_factor
    
    def add_electron_energy(self, energy: float, output_factor: float = 1.0):
        """
        Thêm năng lượng electron cho máy Linac.
        
        Parameters
        ----------
        energy : float
            Năng lượng electron (MeV)
        output_factor : float, optional
            Hệ số đầu ra cho năng lượng này, mặc định là 1.0
        """
        if energy not in self.electron_energies:
            self.electron_energies.append(energy)
            self.output_factors[f"ELECTRON_{energy}MeV"] = output_factor
    
    def set_mlc_model(self, mlc_model: MLCModel):
        """
        Thiết lập mô hình MLC cho máy Linac.
        
        Parameters
        ----------
        mlc_model : MLCModel
            Mô hình MLC
        """
        self.mlc_model = mlc_model
        self.beam_limiting_devices["MLC"] = {
            "type": BeamLimitingDeviceType.MLC,
            "model": mlc_model.model_name,
            "num_leaves": mlc_model.num_leaves,
            "leaf_width": mlc_model.leaf_width
        }
    
    def add_beam_limiting_device(self, device_type: BeamLimitingDeviceType, 
                                device_name: str, device_specs: Dict[str, Any]):
        """
        Thêm thiết bị giới hạn chùm tia cho máy Linac.
        
        Parameters
        ----------
        device_type : BeamLimitingDeviceType
            Loại thiết bị
        device_name : str
            Tên của thiết bị
        device_specs : Dict[str, Any]
            Thông số kỹ thuật của thiết bị
        """
        self.beam_limiting_devices[device_name] = {
            "type": device_type,
            **device_specs
        }
    
    def get_beam_limiting_device(self, device_name: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin về thiết bị giới hạn chùm tia.
        
        Parameters
        ----------
        device_name : str
            Tên của thiết bị
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Thông tin về thiết bị, None nếu không tìm thấy
        """
        return self.beam_limiting_devices.get(device_name)
    
    def get_output_factor(self, beam_type: str, energy: float) -> float:
        """
        Lấy hệ số đầu ra cho một loại chùm và năng lượng cụ thể.
        
        Parameters
        ----------
        beam_type : str
            Loại chùm ("PHOTON" hoặc "ELECTRON")
        energy : float
            Năng lượng (MV cho photon, MeV cho electron)
            
        Returns
        -------
        float
            Hệ số đầu ra
        """
        key = f"{beam_type}_{energy}{'MV' if beam_type == 'PHOTON' else 'MeV'}"
        return self.output_factors.get(key, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin máy Linac thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin máy Linac
        """
        data = super().to_dict()
        data.update({
            "photon_energies": self.photon_energies,
            "electron_energies": self.electron_energies,
            "beam_limiting_devices": self.beam_limiting_devices,
            "output_factors": self.output_factors,
            "mlc_model": self.mlc_model.to_dict() if self.mlc_model else None,
            "specs": self.specs.to_dict()
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Linac':
        """
        Tạo đối tượng Linac từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin máy Linac
            
        Returns
        -------
        Linac
            Đối tượng Linac
        """
        linac = cls(
            machine_name=data["machine_name"],
            manufacturer=data["manufacturer"],
            machine_id=data["machine_id"]
        )
        
        # Cập nhật các thuộc tính
        linac.accelerator_type = data["accelerator_type"]
        linac.photon_energies = data["photon_energies"]
        linac.electron_energies = data["electron_energies"]
        linac.beam_limiting_devices = data["beam_limiting_devices"]
        linac.output_factors = data["output_factors"]
        
        # Cập nhật MLC model nếu có
        if data.get("mlc_model"):
            from quangtps.treatment.mlc.mlc_model import MLCModel
            linac.mlc_model = MLCModel.from_dict(data["mlc_model"])
        
        # Cập nhật thông số kỹ thuật
        if data.get("specs"):
            from quangtps.treatment.machine.machine_specs import MachineSpecification
            linac.specs = MachineSpecification.from_dict(data["specs"])
        
        return linac