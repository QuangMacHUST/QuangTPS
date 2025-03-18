#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa giao diện chung cho các kỹ thuật điều trị xạ trị.

Module này cung cấp lớp cơ sở và giao diện chung để tất cả các kỹ thuật điều trị
như 3D-CRT, IMRT, VMAT, SBRT, TBI, BNCT, v.v. có thể triển khai theo cách thống nhất.
Điều này giúp đảm bảo tích hợp mượt mà giữa các kỹ thuật khác nhau và dễ dàng mở rộng hệ thống.
"""

import abc
from enum import Enum
from typing import Dict, Any, List, Optional, Protocol
import uuid
import json
import logging

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation

logger = logging.getLogger(__name__)

class TechniqueCategory(str, Enum):
    """Enum đại diện cho các danh mục kỹ thuật điều trị."""
    CONVENTIONAL = "CONVENTIONAL"  # Kỹ thuật thông thường (3D-CRT)
    ADVANCED = "ADVANCED"  # Kỹ thuật tiên tiến (IMRT, VMAT, etc)
    SPECIAL = "SPECIAL"  # Kỹ thuật đặc biệt (TBI, TSET, BNCT, etc)
    PARTICLE = "PARTICLE"  # Kỹ thuật hạt nặng (Proton, Carbon, etc)
    RESEARCH = "RESEARCH"  # Kỹ thuật nghiên cứu thử nghiệm


class TreatmentTechniqueInterface(abc.ABC):
    """
    Giao diện trừu tượng cho tất cả các kỹ thuật điều trị.
    
    Lớp cơ sở này định nghĩa các phương thức mà tất cả các kỹ thuật điều trị
    cần phải triển khai.
    """
    
    @abc.abstractmethod
    def get_id(self) -> str:
        """
        Lấy định danh duy nhất của kỹ thuật.
        
        Returns
        -------
        str
            ID duy nhất của kỹ thuật
        """
        pass
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Lấy tên của kỹ thuật.
        
        Returns
        -------
        str
            Tên của kỹ thuật
        """
        pass
    
    @abc.abstractmethod
    def get_category(self) -> TechniqueCategory:
        """
        Lấy danh mục của kỹ thuật.
        
        Returns
        -------
        TechniqueCategory
            Danh mục kỹ thuật
        """
        pass
    
    @abc.abstractmethod
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Thiết lập máy điều trị.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Máy điều trị
        """
        pass
    
    @abc.abstractmethod
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Thiết lập phân liều.
        
        Parameters
        ----------
        fractionation : Fractionation
            Phân liều
        """
        pass
    
    @abc.abstractmethod
    def add_beam(self, beam: Beam) -> None:
        """
        Thêm chùm tia vào kỹ thuật.
        
        Parameters
        ----------
        beam : Beam
            Chùm tia xạ trị
        """
        pass
    
    @abc.abstractmethod
    def get_beams(self) -> List[Beam]:
        """
        Lấy danh sách chùm tia.
        
        Returns
        -------
        List[Beam]
            Danh sách chùm tia
        """
        pass
    
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi kỹ thuật thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kỹ thuật
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreatmentTechniqueInterface':
        """
        Tạo đối tượng kỹ thuật từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin kỹ thuật
            
        Returns
        -------
        TreatmentTechniqueInterface
            Đối tượng kỹ thuật
        """
        pass


class BaseTreatmentTechnique(TreatmentTechniqueInterface):
    """
    Lớp cơ sở cho tất cả các kỹ thuật điều trị.
    
    Lớp này cung cấp triển khai cơ bản các phương thức từ giao diện
    TreatmentTechniqueInterface, có thể được kế thừa bởi các kỹ thuật cụ thể.
    """
    
    def __init__(self, 
                 name: str, 
                 technique_id: Optional[str] = None,
                 category: TechniqueCategory = TechniqueCategory.CONVENTIONAL):
        """
        Khởi tạo kỹ thuật điều trị cơ bản.
        
        Parameters
        ----------
        name : str
            Tên kỹ thuật
        technique_id : str, optional
            ID duy nhất của kỹ thuật
        category : TechniqueCategory
            Danh mục kỹ thuật
        """
        self.name = name
        self.technique_id = technique_id or str(uuid.uuid4())
        self.category = category
        self.beams: List[Beam] = []
        self.machine: Optional[TreatmentMachine] = None
        self.fractionation: Optional[Fractionation] = None
        self.metadata: Dict[str, Any] = {}
    
    def get_id(self) -> str:
        """
        Lấy định danh duy nhất của kỹ thuật.
        
        Returns
        -------
        str
            ID duy nhất của kỹ thuật
        """
        return self.technique_id
    
    def get_name(self) -> str:
        """
        Lấy tên của kỹ thuật.
        
        Returns
        -------
        str
            Tên của kỹ thuật
        """
        return self.name
    
    def get_category(self) -> TechniqueCategory:
        """
        Lấy danh mục của kỹ thuật.
        
        Returns
        -------
        TechniqueCategory
            Danh mục kỹ thuật
        """
        return self.category
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Thiết lập máy điều trị.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Máy điều trị
        """
        self.machine = machine
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Thiết lập phân liều.
        
        Parameters
        ----------
        fractionation : Fractionation
            Phân liều
        """
        self.fractionation = fractionation
    
    def add_beam(self, beam: Beam) -> None:
        """
        Thêm chùm tia vào kỹ thuật.
        
        Parameters
        ----------
        beam : Beam
            Chùm tia xạ trị
        """
        self.beams.append(beam)
    
    def get_beams(self) -> List[Beam]:
        """
        Lấy danh sách chùm tia.
        
        Returns
        -------
        List[Beam]
            Danh sách chùm tia
        """
        return self.beams
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Thêm metadata vào kỹ thuật.
        
        Parameters
        ----------
        key : str
            Khóa metadata
        value : Any
            Giá trị metadata
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị metadata.
        
        Parameters
        ----------
        key : str
            Khóa metadata
        default : Any, optional
            Giá trị mặc định nếu không tìm thấy
            
        Returns
        -------
        Any
            Giá trị metadata
        """
        return self.metadata.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi kỹ thuật thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kỹ thuật
        """
        beam_dicts = [beam.to_dict() for beam in self.beams] if self.beams else []
        
        result = {
            "name": self.name,
            "technique_id": self.technique_id,
            "category": self.category,
            "beams": beam_dicts,
            "metadata": self.metadata,
        }
        
        if self.machine:
            result["machine"] = self.machine.to_dict()
        
        if self.fractionation:
            result["fractionation"] = self.fractionation.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseTreatmentTechnique':
        """
        Tạo đối tượng kỹ thuật từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin kỹ thuật
            
        Returns
        -------
        BaseTreatmentTechnique
            Đối tượng kỹ thuật
        """
        from quangtps.treatment.machine.machine_factory import MachineFactory
        from quangtps.treatment.beams.beam_factory import BeamFactory
        from quangtps.treatment.fractionation import Fractionation
        
        technique = cls(
            name=data["name"],
            technique_id=data["technique_id"],
            category=TechniqueCategory(data["category"]) if "category" in data else TechniqueCategory.CONVENTIONAL
        )
        
        # Khôi phục metadata
        technique.metadata = data.get("metadata", {})
        
        # Khôi phục máy điều trị
        if "machine" in data:
            machine_factory = MachineFactory()
            machine = machine_factory.create_from_dict(data["machine"])
            technique.set_machine(machine)
        
        # Khôi phục phân liều
        if "fractionation" in data:
            fractionation = Fractionation.from_dict(data["fractionation"])
            technique.set_fractionation(fractionation)
        
        # Khôi phục chùm tia
        if "beams" in data:
            beam_factory = BeamFactory()
            for beam_data in data["beams"]:
                beam = beam_factory.create_from_dict(beam_data)
                technique.add_beam(beam)
        
        return technique


# Đảm bảo các lớp và enum được xuất ra đúng cách
__all__ = [
    'TreatmentTechniqueInterface', 
    'BaseTreatmentTechnique', 
    'TechniqueCategory'
]
