#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các mẫu (templates) trong QuangTPS.

Module này cung cấp các lớp và phương thức để quản lý và sử dụng các mẫu kế hoạch xạ trị,
mẫu cấu hình chùm tia và mẫu giao thức điều trị để tăng tốc quá trình lập kế hoạch.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import numpy as np

from quangtps.planning.beam import BeamPlanning, BeamArrangement
from quangtps.planning.optimization import OptimizationSettings, OptimizationObjective, OptimizationConstraint
from quangtps.planning.optimization import OptimizationObjectiveType, OptimizationConstraintType

logger = logging.getLogger(__name__)


class PlanTemplate:
    """
    Lớp mẫu kế hoạch xạ trị.
    
    Lớp này cung cấp một mẫu cho kế hoạch xạ trị, bao gồm thông tin về
    cấu hình chùm tia, cài đặt tối ưu hóa và các tham số khác.
    """
    
    def __init__(self, template_id: str, name: str, description: str = "", site: str = "", technique: str = ""):
        """
        Khởi tạo một mẫu kế hoạch.
        
        Parameters
        ----------
        template_id : str
            ID duy nhất của mẫu
        name : str
            Tên mẫu
        description : str, optional
            Mô tả về mẫu
        site : str, optional
            Vị trí điều trị
        technique : str, optional
            Kỹ thuật xạ trị (IMRT, VMAT, ...)
        """
        self.template_id = template_id
        self.name = name
        self.description = description
        self.site = site
        self.technique = technique
        
        self.beam_templates = []  # List[BeamTemplate]
        self.optimization_settings = None  # OptimizationSettings
        self.protocol_templates = []  # List[ProtocolTemplate]
        self.parameters = {}  # Dict[str, Any]
        
    def add_beam_template(self, beam_template: 'BeamTemplate'):
        """
        Thêm một mẫu chùm tia vào mẫu kế hoạch.
        
        Parameters
        ----------
        beam_template : BeamTemplate
            Mẫu chùm tia cần thêm
        """
        self.beam_templates.append(beam_template)
        
    def set_optimization_settings(self, settings: OptimizationSettings):
        """
        Đặt cài đặt tối ưu hóa cho mẫu kế hoạch.
        
        Parameters
        ----------
        settings : OptimizationSettings
            Cài đặt tối ưu hóa
        """
        self.optimization_settings = settings
        
    def add_protocol_template(self, protocol: 'ProtocolTemplate'):
        """
        Thêm một mẫu giao thức vào mẫu kế hoạch.
        
        Parameters
        ----------
        protocol : ProtocolTemplate
            Mẫu giao thức cần thêm
        """
        self.protocol_templates.append(protocol)
        
    def set_parameter(self, key: str, value: Any):
        """
        Đặt một tham số cho mẫu kế hoạch.
        
        Parameters
        ----------
        key : str
            Tên tham số
        value : Any
            Giá trị tham số
        """
        self.parameters[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi mẫu kế hoạch thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin mẫu kế hoạch
        """
        return {
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'site': self.site,
            'technique': self.technique,
            'beam_templates': [bt.to_dict() for bt in self.beam_templates],
            'optimization_settings': self.optimization_settings.to_dict() if self.optimization_settings else None,
            'protocol_templates': [pt.to_dict() for pt in self.protocol_templates],
            'parameters': self.parameters
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanTemplate':
        """
        Tạo đối tượng PlanTemplate từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin mẫu kế hoạch
            
        Returns
        -------
        PlanTemplate
            Đối tượng mẫu kế hoạch
        """
        template = cls(
            template_id=data.get('template_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            site=data.get('site', ''),
            technique=data.get('technique', '')
        )
        
        # Phục hồi các mẫu chùm tia
        if 'beam_templates' in data:
            for bt_data in data['beam_templates']:
                template.add_beam_template(BeamTemplate.from_dict(bt_data))
                
        # Phục hồi cài đặt tối ưu hóa
        if 'optimization_settings' in data and data['optimization_settings']:
            template.set_optimization_settings(OptimizationSettings.from_dict(data['optimization_settings']))
            
        # Phục hồi các mẫu giao thức
        if 'protocol_templates' in data:
            for pt_data in data['protocol_templates']:
                template.add_protocol_template(ProtocolTemplate.from_dict(pt_data))
                
        # Phục hồi các tham số
        if 'parameters' in data:
            template.parameters = data['parameters']
            
        return template
        
    def save_to_file(self, file_path: str):
        """
        Lưu mẫu kế hoạch vào file.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    @classmethod
    def load_from_file(cls, file_path: str) -> 'PlanTemplate':
        """
        Tải mẫu kế hoạch từ file.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file
            
        Returns
        -------
        PlanTemplate
            Đối tượng mẫu kế hoạch
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return cls.from_dict(data)


class BeamTemplate:
    """
    Lớp mẫu chùm tia.
    
    Lớp này cung cấp một mẫu cho cấu hình chùm tia, bao gồm thông tin về
    hình dạng, góc, năng lượng và các tham số khác của chùm tia.
    """
    
    def __init__(self, template_id: str, name: str, description: str = "",
                angle_gantry: float = 0.0, angle_couch: float = 0.0, angle_collimator: float = 0.0,
                energy: str = "6MV"):
        """
        Khởi tạo một mẫu chùm tia.
        
        Parameters
        ----------
        template_id : str
            ID duy nhất của mẫu
        name : str
            Tên mẫu
        description : str, optional
            Mô tả về mẫu
        angle_gantry : float, optional
            Góc gantry (độ)
        angle_couch : float, optional
            Góc bàn (độ)
        angle_collimator : float, optional
            Góc collimator (độ)
        energy : str, optional
            Năng lượng chùm tia
        """
        self.template_id = template_id
        self.name = name
        self.description = description
        self.angle_gantry = angle_gantry
        self.angle_couch = angle_couch
        self.angle_collimator = angle_collimator
        self.energy = energy
        
        self.field_size = (10.0, 10.0)  # (width, height) in cm
        self.beam_modifiers = {}  # Dict[str, Any]
        self.mlc_type = ""
        self.isocenter_offset = (0.0, 0.0, 0.0)  # (x, y, z) in cm
        self.parameters = {}  # Dict[str, Any]
        
    def set_field_size(self, width: float, height: float):
        """
        Đặt kích thước trường.
        
        Parameters
        ----------
        width : float
            Chiều rộng trường (cm)
        height : float
            Chiều cao trường (cm)
        """
        self.field_size = (width, height)
        
    def add_beam_modifier(self, modifier_type: str, parameters: Dict[str, Any]):
        """
        Thêm một bộ điều chỉnh chùm tia.
        
        Parameters
        ----------
        modifier_type : str
            Loại bộ điều chỉnh (wedge, block, ...)
        parameters : Dict[str, Any]
            Các tham số của bộ điều chỉnh
        """
        self.beam_modifiers[modifier_type] = parameters
        
    def set_mlc_type(self, mlc_type: str):
        """
        Đặt loại MLC.
        
        Parameters
        ----------
        mlc_type : str
            Loại MLC
        """
        self.mlc_type = mlc_type
        
    def set_isocenter_offset(self, x: float, y: float, z: float):
        """
        Đặt độ lệch tâm đồng trục.
        
        Parameters
        ----------
        x : float
            Độ lệch theo trục x (cm)
        y : float
            Độ lệch theo trục y (cm)
        z : float
            Độ lệch theo trục z (cm)
        """
        self.isocenter_offset = (x, y, z)
        
    def set_parameter(self, key: str, value: Any):
        """
        Đặt một tham số cho mẫu chùm tia.
        
        Parameters
        ----------
        key : str
            Tên tham số
        value : Any
            Giá trị tham số
        """
        self.parameters[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi mẫu chùm tia thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin mẫu chùm tia
        """
        return {
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'angle_gantry': self.angle_gantry,
            'angle_couch': self.angle_couch,
            'angle_collimator': self.angle_collimator,
            'energy': self.energy,
            'field_size': self.field_size,
            'beam_modifiers': self.beam_modifiers,
            'mlc_type': self.mlc_type,
            'isocenter_offset': self.isocenter_offset,
            'parameters': self.parameters
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeamTemplate':
        """
        Tạo đối tượng BeamTemplate từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin mẫu chùm tia
            
        Returns
        -------
        BeamTemplate
            Đối tượng mẫu chùm tia
        """
        template = cls(
            template_id=data.get('template_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            angle_gantry=data.get('angle_gantry', 0.0),
            angle_couch=data.get('angle_couch', 0.0),
            angle_collimator=data.get('angle_collimator', 0.0),
            energy=data.get('energy', '6MV')
        )
        
        # Phục hồi các thuộc tính khác
        if 'field_size' in data:
            template.field_size = tuple(data['field_size'])
            
        if 'beam_modifiers' in data:
            template.beam_modifiers = data['beam_modifiers']
            
        if 'mlc_type' in data:
            template.mlc_type = data['mlc_type']
            
        if 'isocenter_offset' in data:
            template.isocenter_offset = tuple(data['isocenter_offset'])
            
        if 'parameters' in data:
            template.parameters = data['parameters']
            
        return template


class ProtocolTemplate:
    """
    Lớp mẫu giao thức điều trị.
    
    Lớp này cung cấp một mẫu cho giao thức điều trị, bao gồm thông tin về
    liều kê đơn, phân đoạn, ràng buộc liều và các tham số khác.
    """
    
    def __init__(self, template_id: str, name: str, description: str = "", site: str = ""):
        """
        Khởi tạo một mẫu giao thức.
        
        Parameters
        ----------
        template_id : str
            ID duy nhất của mẫu
        name : str
            Tên mẫu
        description : str, optional
            Mô tả về mẫu
        site : str, optional
            Vị trí điều trị
        """
        self.template_id = template_id
        self.name = name
        self.description = description
        self.site = site
        
        self.prescription_doses = {}  # Dict[str, float]
        self.fractionation = {'num_fractions': 0, 'dose_per_fraction': 0.0}
        self.objectives = []  # List[Dict]
        self.constraints = []  # List[Dict]
        self.priority_structures = []  # List[str]
        self.parameters = {}  # Dict[str, Any]
        
    def set_prescription_dose(self, structure_id: str, dose: float):
        """
        Đặt liều kê đơn cho một cấu trúc.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        dose : float
            Liều kê đơn (Gy)
        """
        self.prescription_doses[structure_id] = dose
        
    def set_fractionation(self, num_fractions: int, dose_per_fraction: float):
        """
        Đặt phân đoạn.
        
        Parameters
        ----------
        num_fractions : int
            Số phân đoạn
        dose_per_fraction : float
            Liều mỗi phân đoạn (Gy)
        """
        self.fractionation = {
            'num_fractions': num_fractions,
            'dose_per_fraction': dose_per_fraction
        }
        
    def add_objective(self, structure_id: str, objective_type: str, dose_value: Optional[float] = None,
                     volume_value: Optional[float] = None, weight: float = 1.0, priority: int = 1):
        """
        Thêm một mục tiêu tối ưu hóa.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        objective_type : str
            Loại mục tiêu
        dose_value : float, optional
            Giá trị liều (Gy)
        volume_value : float, optional
            Giá trị thể tích (%)
        weight : float
            Trọng số
        priority : int
            Độ ưu tiên
        """
        objective = {
            'structure_id': structure_id,
            'objective_type': objective_type,
            'dose_value': dose_value,
            'volume_value': volume_value,
            'weight': weight,
            'priority': priority
        }
        
        self.objectives.append(objective)
        
    def add_constraint(self, structure_id: str, constraint_type: str, dose_value: Optional[float] = None,
                      volume_value: Optional[float] = None, is_hard_constraint: bool = False, priority: int = 1):
        """
        Thêm một ràng buộc tối ưu hóa.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        constraint_type : str
            Loại ràng buộc
        dose_value : float, optional
            Giá trị liều (Gy)
        volume_value : float, optional
            Giá trị thể tích (%)
        is_hard_constraint : bool
            True nếu là ràng buộc cứng
        priority : int
            Độ ưu tiên
        """
        constraint = {
            'structure_id': structure_id,
            'constraint_type': constraint_type,
            'dose_value': dose_value,
            'volume_value': volume_value,
            'is_hard_constraint': is_hard_constraint,
            'priority': priority
        }
        
        self.constraints.append(constraint)
        
    def add_priority_structure(self, structure_id: str):
        """
        Thêm một cấu trúc ưu tiên.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        """
        if structure_id not in self.priority_structures:
            self.priority_structures.append(structure_id)
        
    def set_parameter(self, key: str, value: Any):
        """
        Đặt một tham số cho mẫu giao thức.
        
        Parameters
        ----------
        key : str
            Tên tham số
        value : Any
            Giá trị tham số
        """
        self.parameters[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi mẫu giao thức thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin mẫu giao thức
        """
        return {
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'site': self.site,
            'prescription_doses': self.prescription_doses,
            'fractionation': self.fractionation,
            'objectives': self.objectives,
            'constraints': self.constraints,
            'priority_structures': self.priority_structures,
            'parameters': self.parameters
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolTemplate':
        """
        Tạo đối tượng ProtocolTemplate từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin mẫu giao thức
            
        Returns
        -------
        ProtocolTemplate
            Đối tượng mẫu giao thức
        """
        template = cls(
            template_id=data.get('template_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            site=data.get('site', '')
        )
        
        # Phục hồi các thuộc tính khác
        if 'prescription_doses' in data:
            template.prescription_doses = data['prescription_doses']
            
        if 'fractionation' in data:
            template.fractionation = data['fractionation']
            
        if 'objectives' in data:
            template.objectives = data['objectives']
            
        if 'constraints' in data:
            template.constraints = data['constraints']
            
        if 'priority_structures' in data:
            template.priority_structures = data['priority_structures']
            
        if 'parameters' in data:
            template.parameters = data['parameters']
            
        return template
        
    def create_optimization_settings(self) -> OptimizationSettings:
        """
        Tạo cài đặt tối ưu hóa từ mẫu giao thức.
        
        Returns
        -------
        OptimizationSettings
            Cài đặt tối ưu hóa
        """
        settings = OptimizationSettings()
        
        # Thêm các mục tiêu
        for obj_data in self.objectives:
            objective = OptimizationObjective(
                structure_id=obj_data['structure_id'],
                objective_type=OptimizationObjectiveType(obj_data['objective_type']),
                dose_value=obj_data.get('dose_value'),
                volume_value=obj_data.get('volume_value'),
                weight=obj_data.get('weight', 1.0),
                priority=obj_data.get('priority', 1)
            )
            settings.add_objective(objective)
            
        # Thêm các ràng buộc
        for con_data in self.constraints:
            constraint = OptimizationConstraint(
                structure_id=con_data['structure_id'],
                constraint_type=OptimizationConstraintType(con_data['constraint_type']),
                dose_value=con_data.get('dose_value'),
                volume_value=con_data.get('volume_value'),
                is_hard_constraint=con_data.get('is_hard_constraint', False),
                priority=con_data.get('priority', 1)
            )
            settings.add_constraint(constraint)
            
        return settings
