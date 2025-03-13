#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các mẫu trong hệ thống QuangTPS.

Module này cung cấp các lớp và phương thức để quản lý, tổ chức và sử dụng
các mẫu kế hoạch xạ trị và giao thức điều trị trong hệ thống.
"""

import logging
import json
import os
import glob
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import datetime
import uuid

from quangtps.planning.templates import PlanTemplate, BeamTemplate, ProtocolTemplate
from quangtps.planning.plan import Plan, PlanType
from quangtps.planning.beam import BeamSetup, BeamArrangement

logger = logging.getLogger(__name__)


class TemplateCategory(str, Enum):
    """Enum cho các danh mục mẫu."""
    STANDARD = "Standard"  # Mẫu tiêu chuẩn của hệ thống
    INSTITUTIONAL = "Institutional"  # Mẫu của cơ sở y tế
    USER = "User"  # Mẫu cá nhân
    CUSTOM = "Custom"  # Mẫu tùy chỉnh khác


class TemplateType(str, Enum):
    """Enum cho các loại mẫu."""
    PLAN = "Plan"  # Mẫu kế hoạch
    BEAM = "Beam"  # Mẫu chùm tia
    PROTOCOL = "Protocol"  # Mẫu giao thức


class TemplateSorting(str, Enum):
    """Enum cho các kiểu sắp xếp mẫu."""
    NAME = "Name"  # Sắp xếp theo tên
    DATE = "Date"  # Sắp xếp theo ngày
    SITE = "Site"  # Sắp xếp theo vị trí điều trị
    TECHNIQUE = "Technique"  # Sắp xếp theo kỹ thuật
    CATEGORY = "Category"  # Sắp xếp theo danh mục


class TemplateManager:
    """
    Lớp quản lý mẫu (template) trong hệ thống.
    
    Lớp này quản lý việc tạo, xem, cập nhật, xóa và sử dụng
    các mẫu kế hoạch xạ trị, mẫu chùm tia và mẫu giao thức.
    """
    
    def __init__(self, base_directory: Optional[str] = None):
        """
        Khởi tạo quản lý mẫu.
        
        Parameters
        ----------
        base_directory : str, optional
            Thư mục cơ sở để lưu các mẫu
        """
        self.base_directory = base_directory
        if self.base_directory is None:
            # Sử dụng thư mục mặc định
            self.base_directory = os.path.join(os.path.expanduser('~'), 'QuangTPS', 'templates')
            
        # Tạo cấu trúc thư mục nếu chưa tồn tại
        for category in TemplateCategory:
            for template_type in TemplateType:
                directory = os.path.join(self.base_directory, category.value, template_type.value)
                os.makedirs(directory, exist_ok=True)
                
        # Cache các mẫu
        self.templates_cache = {
            TemplateType.PLAN: {},
            TemplateType.BEAM: {},
            TemplateType.PROTOCOL: {}
        }
        
        # Tải các mẫu có sẵn
        self.load_templates()
    
    def load_templates(self):
        """
        Tải tất cả các mẫu từ thư mục.
        """
        for category in TemplateCategory:
            for template_type in TemplateType:
                directory = os.path.join(self.base_directory, category.value, template_type.value)
                pattern = os.path.join(directory, '*.json')
                
                for file_path in glob.glob(pattern):
                    try:
                        if template_type == TemplateType.PLAN:
                            template = PlanTemplate.load_from_file(file_path)
                            self.templates_cache[template_type][template.template_id] = {
                                'template': template,
                                'category': category,
                                'path': file_path,
                                'metadata': self._extract_metadata(file_path)
                            }
                        elif template_type == TemplateType.BEAM:
                            template = BeamTemplate.load_from_file(file_path)
                            self.templates_cache[template_type][template.template_id] = {
                                'template': template,
                                'category': category,
                                'path': file_path,
                                'metadata': self._extract_metadata(file_path)
                            }
                        elif template_type == TemplateType.PROTOCOL:
                            template = ProtocolTemplate.load_from_file(file_path)
                            self.templates_cache[template_type][template.template_id] = {
                                'template': template,
                                'category': category,
                                'path': file_path,
                                'metadata': self._extract_metadata(file_path)
                            }
                    except Exception as e:
                        logger.error(f"Lỗi khi tải mẫu từ file {file_path}: {str(e)}")
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Trích xuất metadata từ một file mẫu.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file mẫu
            
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa metadata
        """
        stat_info = os.stat(file_path)
        created_time = datetime.datetime.fromtimestamp(stat_info.st_ctime)
        modified_time = datetime.datetime.fromtimestamp(stat_info.st_mtime)
        
        return {
            'created_date': created_time,
            'modified_date': modified_time,
            'file_size': stat_info.st_size
        }
    
    def get_templates(self, template_type: TemplateType, 
                     category: Optional[TemplateCategory] = None,
                     site: Optional[str] = None,
                     keyword: Optional[str] = None,
                     sort_by: Optional[TemplateSorting] = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các mẫu.
        
        Parameters
        ----------
        template_type : TemplateType
            Loại mẫu cần lấy
        category : TemplateCategory, optional
            Danh mục mẫu cần lọc
        site : str, optional
            Vị trí điều trị cần lọc
        keyword : str, optional
            Từ khóa tìm kiếm trong tên hoặc mô tả
        sort_by : TemplateSorting, optional
            Tiêu chí sắp xếp
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các mẫu với metadata
        """
        results = []
        
        for template_id, data in self.templates_cache[template_type].items():
            template = data['template']
            template_category = data['category']
            metadata = data['metadata']
            
            # Lọc theo danh mục
            if category and template_category != category:
                continue
                
            # Lọc theo vị trí điều trị (nếu có)
            has_site = hasattr(template, 'site')
            if site and has_site and template.site.lower() != site.lower():
                continue
                
            # Lọc theo từ khóa
            if keyword:
                keyword_lower = keyword.lower()
                name_match = keyword_lower in template.name.lower()
                desc_match = hasattr(template, 'description') and keyword_lower in template.description.lower()
                if not (name_match or desc_match):
                    continue
            
            # Thêm vào kết quả
            template_info = {
                'id': template.template_id,
                'name': template.name,
                'description': template.description if hasattr(template, 'description') else "",
                'category': template_category.value,
                'created_date': metadata['created_date'],
                'modified_date': metadata['modified_date']
            }
            
            # Thêm thông tin đặc thù của loại mẫu
            if template_type == TemplateType.PLAN:
                template_info.update({
                    'site': template.site,
                    'technique': template.technique
                })
            elif template_type == TemplateType.PROTOCOL:
                template_info.update({
                    'site': template.site
                })
                
            results.append(template_info)
        
        # Sắp xếp kết quả
        if sort_by:
            if sort_by == TemplateSorting.NAME:
                results.sort(key=lambda x: x['name'])
            elif sort_by == TemplateSorting.DATE:
                results.sort(key=lambda x: x['modified_date'], reverse=True)
            elif sort_by == TemplateSorting.SITE and 'site' in results[0]:
                results.sort(key=lambda x: x['site'])
            elif sort_by == TemplateSorting.TECHNIQUE and 'technique' in results[0]:
                results.sort(key=lambda x: x['technique'])
            elif sort_by == TemplateSorting.CATEGORY:
                results.sort(key=lambda x: x['category'])
        
        return results
    
    def get_template(self, template_type: TemplateType, template_id: str) -> Optional[Any]:
        """
        Lấy một mẫu theo ID.
        
        Parameters
        ----------
        template_type : TemplateType
            Loại mẫu
        template_id : str
            ID của mẫu
            
        Returns
        -------
        Optional[Any]
            Đối tượng mẫu nếu tìm thấy, None nếu không
        """
        if template_id in self.templates_cache[template_type]:
            return self.templates_cache[template_type][template_id]['template']
        return None
    
    def save_template(self, template: Union[PlanTemplate, BeamTemplate, ProtocolTemplate], 
                     category: TemplateCategory = TemplateCategory.USER) -> bool:
        """
        Lưu một mẫu.
        
        Parameters
        ----------
        template : Union[PlanTemplate, BeamTemplate, ProtocolTemplate]
            Mẫu cần lưu
        category : TemplateCategory
            Danh mục mẫu
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        try:
            # Xác định loại mẫu
            if isinstance(template, PlanTemplate):
                template_type = TemplateType.PLAN
            elif isinstance(template, BeamTemplate):
                template_type = TemplateType.BEAM
            elif isinstance(template, ProtocolTemplate):
                template_type = TemplateType.PROTOCOL
            else:
                logger.error(f"Loại mẫu không được hỗ trợ: {type(template)}")
                return False
                
            # Tạo tên file từ ID
            file_name = f"{template.template_id}.json"
            directory = os.path.join(self.base_directory, category.value, template_type.value)
            file_path = os.path.join(directory, file_name)
            
            # Lưu mẫu vào file
            template.save_to_file(file_path)
            
            # Cập nhật cache
            self.templates_cache[template_type][template.template_id] = {
                'template': template,
                'category': category,
                'path': file_path,
                'metadata': self._extract_metadata(file_path)
            }
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu mẫu: {str(e)}")
            return False
    
    def delete_template(self, template_type: TemplateType, template_id: str) -> bool:
        """
        Xóa một mẫu.
        
        Parameters
        ----------
        template_type : TemplateType
            Loại mẫu
        template_id : str
            ID của mẫu
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        try:
            if template_id in self.templates_cache[template_type]:
                file_path = self.templates_cache[template_type][template_id]['path']
                
                # Xóa file
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Xóa khỏi cache
                del self.templates_cache[template_type][template_id]
                
                return True
            return False
        except Exception as e:
            logger.error(f"Lỗi khi xóa mẫu: {str(e)}")
            return False
    
    def apply_plan_template(self, template_id: str, patient_id: str, plan_name: str) -> Optional[Plan]:
        """
        Áp dụng một mẫu kế hoạch cho một bệnh nhân.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu kế hoạch
        patient_id : str
            ID của bệnh nhân
        plan_name : str
            Tên kế hoạch mới
            
        Returns
        -------
        Optional[Plan]
            Đối tượng kế hoạch mới nếu thành công, None nếu thất bại
        """
        try:
            # Lấy mẫu kế hoạch
            plan_template = self.get_template(TemplateType.PLAN, template_id)
            if not plan_template:
                logger.error(f"Không tìm thấy mẫu kế hoạch có ID {template_id}")
                return None
                
            # Tạo kế hoạch mới
            plan_id = str(uuid.uuid4())
            plan = Plan(plan_name=plan_name, patient_id=patient_id, plan_id=plan_id)
            
            # Áp dụng các thông tin từ mẫu
            if plan_template.technique == "IMRT":
                plan.plan_type = PlanType.IMRT
            elif plan_template.technique == "VMAT":
                plan.plan_type = PlanType.VMAT
            elif plan_template.technique == "3DCRT":
                plan.plan_type = PlanType.THREE_D_CRT
            
            # Tạo cấu hình chùm tia từ mẫu
            beam_arrangement = BeamArrangement()
            for beam_template in plan_template.beam_templates:
                beam = BeamSetup(name=beam_template.name)
                beam.gantry_angle = beam_template.angle_gantry
                beam.couch_angle = beam_template.angle_couch
                beam.collimator_angle = beam_template.angle_collimator
                beam.energy = beam_template.energy
                beam.field_size = beam_template.field_size
                
                beam_arrangement.add_beam(beam)
            
            plan.beam_arrangement = beam_arrangement
            
            # Áp dụng cài đặt tối ưu hóa từ mẫu
            if plan_template.optimization_settings:
                plan.optimization_settings = plan_template.optimization_settings
            
            # Áp dụng các thông tin khác
            for key, value in plan_template.parameters.items():
                plan.set_parameter(key, value)
            
            return plan
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng mẫu kế hoạch: {str(e)}")
            return None
    
    def apply_beam_template(self, template_id: str, beam_setup: BeamSetup) -> bool:
        """
        Áp dụng một mẫu chùm tia cho một beam setup.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu chùm tia
        beam_setup : BeamSetup
            Đối tượng beam setup cần áp dụng mẫu
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        try:
            # Lấy mẫu chùm tia
            beam_template = self.get_template(TemplateType.BEAM, template_id)
            if not beam_template:
                logger.error(f"Không tìm thấy mẫu chùm tia có ID {template_id}")
                return False
                
            # Áp dụng thông tin từ mẫu
            beam_setup.name = beam_template.name
            beam_setup.gantry_angle = beam_template.angle_gantry
            beam_setup.couch_angle = beam_template.angle_couch
            beam_setup.collimator_angle = beam_template.angle_collimator
            beam_setup.energy = beam_template.energy
            beam_setup.field_size = beam_template.field_size
            
            # Áp dụng các bộ điều chỉnh
            for modifier_type, params in beam_template.beam_modifiers.items():
                beam_setup.beam_modifiers[modifier_type] = params.copy()
            
            # Áp dụng MLC nếu có
            if beam_template.mlc_type:
                beam_setup.mlc_type = beam_template.mlc_type
            
            # Áp dụng offset tâm đồng trục
            beam_setup.isocenter_offset = beam_template.isocenter_offset
            
            # Áp dụng các tham số khác
            for key, value in beam_template.parameters.items():
                beam_setup.set_parameter(key, value)
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng mẫu chùm tia: {str(e)}")
            return False
    
    def apply_protocol_template(self, template_id: str, plan: Plan) -> bool:
        """
        Áp dụng một mẫu giao thức cho một kế hoạch.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu giao thức
        plan : Plan
            Đối tượng kế hoạch cần áp dụng mẫu
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        try:
            # Lấy mẫu giao thức
            protocol_template = self.get_template(TemplateType.PROTOCOL, template_id)
            if not protocol_template:
                logger.error(f"Không tìm thấy mẫu giao thức có ID {template_id}")
                return False
            
            # Tạo cài đặt tối ưu hóa từ mẫu giao thức
            optimization_settings = protocol_template.create_optimization_settings()
            plan.optimization_settings = optimization_settings
            
            # Cập nhật liều kê đơn
            for struct_id, dose in protocol_template.prescription_doses.items():
                if plan.prescription:
                    plan.prescription.set_structure_prescription(struct_id, dose)
            
            # Cập nhật phân đoạn
            if protocol_template.fractionation['num_fractions'] > 0:
                if plan.prescription:
                    plan.prescription.num_fractions = protocol_template.fractionation['num_fractions']
                    plan.prescription.dose_per_fraction = protocol_template.fractionation['dose_per_fraction']
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng mẫu giao thức: {str(e)}")
            return False
    
    def create_plan_template_from_plan(self, plan: Plan, name: str, description: str = "",
                                     template_id: Optional[str] = None) -> Optional[PlanTemplate]:
        """
        Tạo một mẫu kế hoạch từ một kế hoạch hiện có.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch nguồn
        name : str
            Tên mẫu
        description : str
            Mô tả mẫu
        template_id : str, optional
            ID mẫu, sẽ tạo mới nếu không cung cấp
            
        Returns
        -------
        Optional[PlanTemplate]
            Mẫu kế hoạch mới nếu thành công, None nếu thất bại
        """
        try:
            # Tạo ID mẫu nếu không cung cấp
            if not template_id:
                template_id = str(uuid.uuid4())
                
            # Xác định kỹ thuật từ loại kế hoạch
            technique = ""
            if plan.plan_type:
                if plan.plan_type == PlanType.IMRT:
                    technique = "IMRT"
                elif plan.plan_type == PlanType.VMAT:
                    technique = "VMAT"
                elif plan.plan_type == PlanType.THREE_D_CRT:
                    technique = "3DCRT"
            
            # Tạo mẫu kế hoạch mới
            template = PlanTemplate(
                template_id=template_id,
                name=name,
                description=description,
                site=plan.get_parameter("site", ""),
                technique=technique
            )
            
            # Thêm cài đặt tối ưu hóa
            if plan.optimization_settings:
                template.set_optimization_settings(plan.optimization_settings)
            
            # Thêm cấu hình chùm tia
            if plan.beam_arrangement:
                for beam in plan.beam_arrangement.beams:
                    beam_template = BeamTemplate(
                        template_id=str(uuid.uuid4()),
                        name=beam.name,
                        description=f"Beam template from {plan.plan_name}",
                        angle_gantry=beam.gantry_angle,
                        angle_couch=beam.couch_angle,
                        angle_collimator=beam.collimator_angle,
                        energy=beam.energy
                    )
                    
                    # Thêm thông tin khác của chùm tia
                    beam_template.field_size = beam.field_size
                    beam_template.beam_modifiers = beam.beam_modifiers.copy() if hasattr(beam, 'beam_modifiers') else {}
                    beam_template.mlc_type = beam.mlc_type if hasattr(beam, 'mlc_type') else ""
                    beam_template.isocenter_offset = beam.isocenter_offset if hasattr(beam, 'isocenter_offset') else (0.0, 0.0, 0.0)
                    
                    # Thêm beam template vào plan template
                    template.add_beam_template(beam_template)
            
            # Thêm các tham số khác
            for key, value in plan.parameters.items():
                template.set_parameter(key, value)
            
            return template
        except Exception as e:
            logger.error(f"Lỗi khi tạo mẫu kế hoạch từ kế hoạch: {str(e)}")
            return None
    
    def import_template(self, file_path: str, category: TemplateCategory = TemplateCategory.USER) -> bool:
        """
        Nhập một mẫu từ file.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file mẫu
        category : TemplateCategory
            Danh mục mẫu
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        try:
            # Đọc file để xác định loại mẫu
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Xác định loại mẫu dựa trên nội dung
            template = None
            
            if 'beam_templates' in data:
                template = PlanTemplate.from_dict(data)
            elif 'angle_gantry' in data:
                template = BeamTemplate.from_dict(data)
            elif 'prescription_doses' in data:
                template = ProtocolTemplate.from_dict(data)
            else:
                logger.error(f"Không thể xác định loại mẫu từ file {file_path}")
                return False
                
            # Lưu mẫu vào thư mục phù hợp
            return self.save_template(template, category)
        except Exception as e:
            logger.error(f"Lỗi khi nhập mẫu từ file {file_path}: {str(e)}")
            return False
    
    def export_template(self, template_type: TemplateType, template_id: str, file_path: str) -> bool:
        """
        Xuất một mẫu ra file.
        
        Parameters
        ----------
        template_type : TemplateType
            Loại mẫu
        template_id : str
            ID của mẫu
        file_path : str
            Đường dẫn đến file xuất
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        try:
            # Lấy mẫu
            template = self.get_template(template_type, template_id)
            if not template:
                logger.error(f"Không tìm thấy mẫu có ID {template_id}")
                return False
                
            # Lưu mẫu ra file
            template.save_to_file(file_path)
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xuất mẫu ra file {file_path}: {str(e)}")
            return False
