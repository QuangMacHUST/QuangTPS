#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý thư viện chùm tia trong QuangTPS.

Module này cung cấp các lớp và phương thức để quản lý và sử dụng các chùm tia tiêu chuẩn,
lưu trữ và truy xuất các mẫu chùm tia phổ biến cho các kỹ thuật xạ trị khác nhau.
"""

import logging
import json
import os
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import copy

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.beams.beam_geometry import BeamGeometry
from quangtps.treatment.beams.beam_modifiers import Wedge, Block, Bolus, Compensator

logger = logging.getLogger(__name__)


class BeamTemplate:
    """
    Lớp mẫu chùm tia.
    
    Lớp này cung cấp một mẫu cho cấu hình chùm tia, để có thể sử dụng lại cho các kế hoạch xạ trị.
    """
    
    def __init__(
        self,
        template_id: str,
        name: str,
        description: str = "",
        beam_geometry: Optional[BeamGeometry] = None
    ):
        """
        Khởi tạo một mẫu chùm tia.
        
        Parameters
        ----------
        template_id : str
            ID duy nhất của mẫu
        name : str
            Tên hiển thị của mẫu
        description : str, optional
            Mô tả về mẫu và mục đích sử dụng
        beam_geometry : BeamGeometry, optional
            Hình học chùm tia, nếu không cung cấp sẽ tạo một hình học mặc định
        """
        self.template_id = template_id
        self.name = name
        self.description = description
        self.beam_geometry = beam_geometry if beam_geometry else BeamGeometry()
        
        self.energy = "6MV"
        self.modifiers = []  # List of modifiers (Wedge, Block, etc.)
        self.technique = ""  # "STATIC", "IMRT", "VMAT", etc.
        self.tags = []  # List of tags for categorization
        self.metadata = {}  # Additional metadata
        
    def set_energy(self, energy: str):
        """
        Đặt năng lượng cho chùm tia.
        
        Parameters
        ----------
        energy : str
            Năng lượng chùm tia (ví dụ: "6MV", "10MV", "6FFF", v.v.)
        """
        self.energy = energy
        
    def add_modifier(self, modifier: Union[Wedge, Block, Bolus, Compensator]):
        """
        Thêm một bộ điều chỉnh vào chùm tia.
        
        Parameters
        ----------
        modifier : Union[Wedge, Block, Bolus, Compensator]
            Bộ điều chỉnh cần thêm
        """
        self.modifiers.append(modifier)
        
    def set_technique(self, technique: str):
        """
        Đặt kỹ thuật cho chùm tia.
        
        Parameters
        ----------
        technique : str
            Kỹ thuật xạ trị ("STATIC", "IMRT", "VMAT", v.v.)
        """
        self.technique = technique
        
    def add_tag(self, tag: str):
        """
        Thêm một thẻ phân loại.
        
        Parameters
        ----------
        tag : str
            Thẻ cần thêm
        """
        if tag not in self.tags:
            self.tags.append(tag)
            
    def add_metadata(self, key: str, value: Any):
        """
        Thêm một trường metadata.
        
        Parameters
        ----------
        key : str
            Tên trường
        value : Any
            Giá trị trường
        """
        self.metadata[key] = value
        
    def create_beam(self, beam_id: str = "", beam_name: str = "") -> Beam:
        """
        Tạo một đối tượng Beam từ mẫu.
        
        Parameters
        ----------
        beam_id : str, optional
            ID cho chùm tia mới, nếu không cung cấp sẽ sử dụng template_id
        beam_name : str, optional
            Tên cho chùm tia mới, nếu không cung cấp sẽ sử dụng name của template
            
        Returns
        -------
        Beam
            Đối tượng chùm tia mới
        """
        beam = Beam(
            beam_name=beam_name if beam_name else self.name,
            beam_id=beam_id if beam_id else self.template_id
        )
        
        # Apply beam geometry
        if hasattr(beam, 'beam_geometry'):
            beam.beam_geometry = copy.deepcopy(self.beam_geometry)
            
        # Apply energy
        if hasattr(beam, 'energy'):
            beam.energy = self.energy
        
        # Thêm các bộ điều chỉnh
        for modifier in self.modifiers:
            beam.add_modifier(copy.deepcopy(modifier))
            
        # Thiết lập kỹ thuật
        beam.technique = self.technique
        
        # Sao chép metadata
        for key, value in self.metadata.items():
            beam.add_metadata(key, value)
            
        return beam
        
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
            'beam_geometry': self.beam_geometry.to_dict(),
            'energy': self.energy,
            'modifiers': [modifier.to_dict() for modifier in self.modifiers],
            'technique': self.technique,
            'tags': self.tags,
            'metadata': self.metadata
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
        beam_geometry = BeamGeometry.from_dict(data['beam_geometry']) if 'beam_geometry' in data else BeamGeometry()
        
        template = cls(
            template_id=data.get('template_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            beam_geometry=beam_geometry
        )
        
        # Thiết lập năng lượng
        template.energy = data.get('energy', '6MV')
        
        # Thiết lập kỹ thuật
        template.technique = data.get('technique', '')
        
        # Thêm thẻ
        if 'tags' in data:
            template.tags = data['tags']
            
        # Sao chép metadata
        if 'metadata' in data:
            template.metadata = data['metadata']
            
        # Thêm các bộ điều chỉnh
        if 'modifiers' in data:
            for mod_data in data['modifiers']:
                # Xác định loại bộ điều chỉnh và tạo đối tượng tương ứng
                mod_type = mod_data.get('type', '')
                
                if mod_type == 'Wedge':
                    modifier = Wedge.from_dict(mod_data)
                elif mod_type == 'Block':
                    modifier = Block.from_dict(mod_data)
                elif mod_type == 'Bolus':
                    modifier = Bolus.from_dict(mod_data)
                elif mod_type == 'Compensator':
                    modifier = Compensator.from_dict(mod_data)
                else:
                    continue
                    
                template.add_modifier(modifier)
                
        return template


class BeamArrangementTemplate:
    """
    Lớp mẫu bố trí chùm tia.
    
    Lớp này cung cấp một mẫu cho cấu hình bố trí nhiều chùm tia,
    thường được sử dụng cho các kỹ thuật xạ trị cụ thể.
    """
    
    def __init__(
        self,
        template_id: str,
        name: str,
        description: str = "",
        site: str = "",
        technique: str = ""
    ):
        """
        Khởi tạo một mẫu bố trí chùm tia.
        
        Parameters
        ----------
        template_id : str
            ID duy nhất của mẫu
        name : str
            Tên hiển thị của mẫu
        description : str, optional
            Mô tả về mẫu và mục đích sử dụng
        site : str, optional
            Vị trí điều trị (ví dụ: "Brain", "Prostate", v.v.)
        technique : str, optional
            Kỹ thuật xạ trị ("3DCRT", "IMRT", "VMAT", v.v.)
        """
        self.template_id = template_id
        self.name = name
        self.description = description
        self.site = site
        self.technique = technique
        
        self.beam_templates = []  # List of BeamTemplate
        self.tags = []  # List of tags for categorization
        self.metadata = {}  # Additional metadata
        
    def add_beam_template(self, beam_template: BeamTemplate):
        """
        Thêm một mẫu chùm tia vào bố trí.
        
        Parameters
        ----------
        beam_template : BeamTemplate
            Mẫu chùm tia cần thêm
        """
        self.beam_templates.append(beam_template)
        
    def add_tag(self, tag: str):
        """
        Thêm một thẻ phân loại.
        
        Parameters
        ----------
        tag : str
            Thẻ cần thêm
        """
        if tag not in self.tags:
            self.tags.append(tag)
            
    def add_metadata(self, key: str, value: Any):
        """
        Thêm một trường metadata.
        
        Parameters
        ----------
        key : str
            Tên trường
        value : Any
            Giá trị trường
        """
        self.metadata[key] = value
        
    def create_beams(self) -> List[Beam]:
        """
        Tạo danh sách các đối tượng Beam từ mẫu bố trí.
        
        Returns
        -------
        List[Beam]
            Danh sách các đối tượng chùm tia mới
        """
        beams = []
        
        for i, template in enumerate(self.beam_templates):
            beam_id = f"{self.template_id}_beam{i+1}"
            beam_name = f"{template.name} {i+1}"
            beam = template.create_beam(beam_id, beam_name)
            beams.append(beam)
            
        return beams
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi mẫu bố trí chùm tia thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin mẫu bố trí chùm tia
        """
        return {
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'site': self.site,
            'technique': self.technique,
            'beam_templates': [bt.to_dict() for bt in self.beam_templates],
            'tags': self.tags,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeamArrangementTemplate':
        """
        Tạo đối tượng BeamArrangementTemplate từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin mẫu bố trí chùm tia
            
        Returns
        -------
        BeamArrangementTemplate
            Đối tượng mẫu bố trí chùm tia
        """
        template = cls(
            template_id=data.get('template_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            site=data.get('site', ''),
            technique=data.get('technique', '')
        )
        
        # Thêm các mẫu chùm tia
        if 'beam_templates' in data:
            for bt_data in data['beam_templates']:
                template.add_beam_template(BeamTemplate.from_dict(bt_data))
                
        # Thêm thẻ
        if 'tags' in data:
            template.tags = data['tags']
            
        # Sao chép metadata
        if 'metadata' in data:
            template.metadata = data['metadata']
            
        return template


class BeamLibrary:
    """
    Lớp quản lý thư viện chùm tia.
    
    Lớp này cung cấp các phương thức để quản lý, lưu trữ và truy xuất
    các mẫu chùm tia và bố trí chùm tia.
    """
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Khởi tạo thư viện chùm tia.
        
        Parameters
        ----------
        library_path : str, optional
            Đường dẫn đến thư mục chứa thư viện, nếu không cung cấp sẽ sử dụng đường dẫn mặc định
        """
        self.library_path = library_path if library_path else self._get_default_library_path()
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.join(self.library_path, 'beam_templates'), exist_ok=True)
        os.makedirs(os.path.join(self.library_path, 'beam_arrangements'), exist_ok=True)
        
        self.beam_templates = {}  # Dict[str, BeamTemplate]
        self.beam_arrangements = {}  # Dict[str, BeamArrangementTemplate]
        
        # Tải thư viện
        self.load_library()
        
    def _get_default_library_path(self) -> str:
        """
        Lấy đường dẫn mặc định cho thư viện.
        
        Returns
        -------
        str
            Đường dẫn mặc định
        """
        return os.path.join(os.path.expanduser('~'), '.quangtps', 'libraries', 'beams')
        
    def load_library(self):
        """Tải thư viện từ đĩa."""
        # Tải các mẫu chùm tia
        beam_template_dir = os.path.join(self.library_path, 'beam_templates')
        if os.path.exists(beam_template_dir):
            for filename in os.listdir(beam_template_dir):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(beam_template_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            template = BeamTemplate.from_dict(data)
                            self.beam_templates[template.template_id] = template
                    except Exception as e:
                        logger.error(f"Lỗi khi tải mẫu chùm tia {filename}: {str(e)}")
                        
        # Tải các mẫu bố trí chùm tia
        beam_arrangement_dir = os.path.join(self.library_path, 'beam_arrangements')
        if os.path.exists(beam_arrangement_dir):
            for filename in os.listdir(beam_arrangement_dir):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(beam_arrangement_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            template = BeamArrangementTemplate.from_dict(data)
                            self.beam_arrangements[template.template_id] = template
                    except Exception as e:
                        logger.error(f"Lỗi khi tải mẫu bố trí chùm tia {filename}: {str(e)}")
                        
    def save_beam_template(self, template: BeamTemplate):
        """
        Lưu một mẫu chùm tia vào thư viện.
        
        Parameters
        ----------
        template : BeamTemplate
            Mẫu chùm tia cần lưu
        """
        # Thêm vào từ điển
        self.beam_templates[template.template_id] = template
        
        # Lưu vào đĩa
        filepath = os.path.join(self.library_path, 'beam_templates', f"{template.template_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template.to_dict(), f, indent=4)
            
    def save_beam_arrangement(self, template: BeamArrangementTemplate):
        """
        Lưu một mẫu bố trí chùm tia vào thư viện.
        
        Parameters
        ----------
        template : BeamArrangementTemplate
            Mẫu bố trí chùm tia cần lưu
        """
        # Thêm vào từ điển
        self.beam_arrangements[template.template_id] = template
        
        # Lưu vào đĩa
        filepath = os.path.join(self.library_path, 'beam_arrangements', f"{template.template_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template.to_dict(), f, indent=4)
            
    def get_beam_template(self, template_id: str) -> Optional[BeamTemplate]:
        """
        Lấy một mẫu chùm tia từ thư viện.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu cần lấy
            
        Returns
        -------
        BeamTemplate, optional
            Mẫu chùm tia, hoặc None nếu không tìm thấy
        """
        return self.beam_templates.get(template_id)
        
    def get_beam_arrangement(self, template_id: str) -> Optional[BeamArrangementTemplate]:
        """
        Lấy một mẫu bố trí chùm tia từ thư viện.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu cần lấy
            
        Returns
        -------
        BeamArrangementTemplate, optional
            Mẫu bố trí chùm tia, hoặc None nếu không tìm thấy
        """
        return self.beam_arrangements.get(template_id)
        
    def delete_beam_template(self, template_id: str) -> bool:
        """
        Xóa một mẫu chùm tia khỏi thư viện.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if template_id in self.beam_templates:
            # Xóa khỏi từ điển
            del self.beam_templates[template_id]
            
            # Xóa khỏi đĩa
            filepath = os.path.join(self.library_path, 'beam_templates', f"{template_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return True
        return False
        
    def delete_beam_arrangement(self, template_id: str) -> bool:
        """
        Xóa một mẫu bố trí chùm tia khỏi thư viện.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if template_id in self.beam_arrangements:
            # Xóa khỏi từ điển
            del self.beam_arrangements[template_id]
            
            # Xóa khỏi đĩa
            filepath = os.path.join(self.library_path, 'beam_arrangements', f"{template_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return True
        return False
        
    def search_beam_templates(self, query: str = "", site: str = "", technique: str = "", 
                            tags: Optional[List[str]] = None) -> List[BeamTemplate]:
        """
        Tìm kiếm các mẫu chùm tia phù hợp với tiêu chí.
        
        Parameters
        ----------
        query : str, optional
            Chuỗi tìm kiếm trong tên hoặc mô tả
        site : str, optional
            Vị trí điều trị
        technique : str, optional
            Kỹ thuật xạ trị
        tags : List[str], optional
            Danh sách các thẻ cần phù hợp
            
        Returns
        -------
        List[BeamTemplate]
            Danh sách các mẫu chùm tia phù hợp
        """
        results = []
        tags = tags if tags else []
        
        for template in self.beam_templates.values():
            # Kiểm tra tên và mô tả
            if query and query.lower() not in template.name.lower() and query.lower() not in template.description.lower():
                continue
                
            # Kiểm tra kỹ thuật
            if technique and template.technique != technique:
                continue
                
            # Kiểm tra thẻ
            if tags and not all(tag in template.tags for tag in tags):
                continue
                
            results.append(template)
            
        return results
        
    def search_beam_arrangements(self, query: str = "", site: str = "", technique: str = "", 
                               tags: Optional[List[str]] = None) -> List[BeamArrangementTemplate]:
        """
        Tìm kiếm các mẫu bố trí chùm tia phù hợp với tiêu chí.
        
        Parameters
        ----------
        query : str, optional
            Chuỗi tìm kiếm trong tên hoặc mô tả
        site : str, optional
            Vị trí điều trị
        technique : str, optional
            Kỹ thuật xạ trị
        tags : List[str], optional
            Danh sách các thẻ cần phù hợp
            
        Returns
        -------
        List[BeamArrangementTemplate]
            Danh sách các mẫu bố trí chùm tia phù hợp
        """
        results = []
        tags = tags or []
        
        for template_id, template in self.beam_arrangements.items():
            # Kiểm tra chuỗi tìm kiếm
            if query and not (query.lower() in template.name.lower() or 
                             query.lower() in template.description.lower()):
                continue
                
            # Kiểm tra vị trí điều trị
            if site and site.lower() != template.site.lower():
                continue
                
            # Kiểm tra kỹ thuật xạ trị
            if technique and technique.lower() != template.technique.lower():
                continue
                
            # Kiểm tra thẻ
            if tags and not all(tag in template.tags for tag in tags):
                continue
                
            results.append(template)
            
        return results
        
    def get_recommended_beam_arrangements(self, site: str, technique: str = "") -> List[BeamArrangementTemplate]:
        """
        Lấy các bố trí chùm tia được đề xuất cho một vị trí điều trị và kỹ thuật.
        
        Parameters
        ----------
        site : str
            Vị trí điều trị
        technique : str, optional
            Kỹ thuật xạ trị
            
        Returns
        -------
        List[BeamArrangementTemplate]
            Danh sách các mẫu bố trí chùm tia được đề xuất
        """
        results = self.search_beam_arrangements(site=site, technique=technique)
        
        # Sắp xếp theo mức độ phù hợp (có thể dựa trên metadata hoặc các tiêu chí khác)
        # Ở đây chỉ đơn giản là trả về tất cả kết quả
        return results
        
    def create_standard_beam_templates(self):
        """
        Tạo các mẫu chùm tia tiêu chuẩn cho thư viện.
        
        Phương thức này tạo ra một số mẫu chùm tia phổ biến và thêm vào thư viện.
        """
        # Tạo mẫu chùm tia anterior
        anterior = BeamTemplate(
            template_id="std_anterior",
            name="Standard Anterior",
            description="Standard anterior beam (0 degrees)"
        )
        anterior.beam_geometry = BeamGeometry()
        anterior.beam_geometry.set_gantry_angle(0)
        anterior.beam_geometry.set_collimator_angle(0)
        anterior.beam_geometry.set_couch_angle(0)
        anterior.add_tag("Standard")
        anterior.add_metadata("type", "Standard")
        self.save_beam_template(anterior)
        
        # Tạo mẫu chùm tia posterior
        posterior = BeamTemplate(
            template_id="std_posterior",
            name="Standard Posterior",
            description="Standard posterior beam (180 degrees)"
        )
        posterior.beam_geometry = BeamGeometry()
        posterior.beam_geometry.set_gantry_angle(180)
        posterior.beam_geometry.set_collimator_angle(0)
        posterior.beam_geometry.set_couch_angle(0)
        posterior.add_tag("Standard")
        posterior.add_metadata("type", "Standard")
        self.save_beam_template(posterior)
        
        # Tạo mẫu chùm tia right lateral
        right_lateral = BeamTemplate(
            template_id="std_right_lateral",
            name="Standard Right Lateral",
            description="Standard right lateral beam (90 degrees)"
        )
        right_lateral.beam_geometry = BeamGeometry()
        right_lateral.beam_geometry.set_gantry_angle(90)
        right_lateral.beam_geometry.set_collimator_angle(0)
        right_lateral.beam_geometry.set_couch_angle(0)
        right_lateral.add_tag("Standard")
        right_lateral.add_metadata("type", "Standard")
        self.save_beam_template(right_lateral)
        
        # Tạo mẫu chùm tia left lateral
        left_lateral = BeamTemplate(
            template_id="std_left_lateral",
            name="Standard Left Lateral",
            description="Standard left lateral beam (270 degrees)"
        )
        left_lateral.beam_geometry = BeamGeometry()
        left_lateral.beam_geometry.set_gantry_angle(270)
        left_lateral.beam_geometry.set_collimator_angle(0)
        left_lateral.beam_geometry.set_couch_angle(0)
        left_lateral.add_tag("Standard")
        left_lateral.add_metadata("type", "Standard")
        self.save_beam_template(left_lateral)
        
    def create_standard_beam_arrangements(self):
        """
        Tạo các mẫu bố trí chùm tia tiêu chuẩn cho thư viện.
        
        Phương thức này tạo ra một số mẫu bố trí chùm tia phổ biến và thêm vào thư viện.
        """
        # Tạo bố trí 4-field box (kỹ thuật dùng 4 trường vuông góc)
        box_arrangement = BeamArrangementTemplate(
            template_id="std_4field_box",
            name="Standard 4-Field Box",
            description="Standard 4-field box technique (AP/PA/RL/LL)",
            site="Pelvis",
            technique="3DCRT"
        )
        
        # Thêm các mẫu chùm tia vào bố trí
        anterior = self.get_beam_template("std_anterior")
        posterior = self.get_beam_template("std_posterior")
        right_lateral = self.get_beam_template("std_right_lateral")
        left_lateral = self.get_beam_template("std_left_lateral")
        
        if anterior and posterior and right_lateral and left_lateral:
            box_arrangement.add_beam_template(anterior)
            box_arrangement.add_beam_template(posterior)
            box_arrangement.add_beam_template(right_lateral)
            box_arrangement.add_beam_template(left_lateral)
            
            box_arrangement.add_tag("Standard")
            box_arrangement.add_tag("3DCRT")
            box_arrangement.add_tag("Pelvis")
            
            self.save_beam_arrangement(box_arrangement)
        else:
            logger.warning("Không thể tạo mẫu bố trí 4-field box - thiếu các mẫu chùm tia cơ bản")
            
        # Tạo bố trí 3-field (kỹ thuật dùng 3 trường cho ung thư vú)
        breast_arrangement = BeamArrangementTemplate(
            template_id="std_breast_tangents",
            name="Standard Breast Tangents + Supraclavicular",
            description="Standard 3-field breast technique with tangential fields and supraclavicular field",
            site="Breast",
            technique="3DCRT"
        )
        
        # Tạo các chùm tia đặc biệt cho kỹ thuật này
        if anterior:
            breast_arrangement.add_beam_template(anterior)  # Supraclavicular field
            
            # Tạo mẫu chùm tia medial tangent
            medial_tangent = BeamTemplate(
                template_id="breast_medial_tangent",
                name="Breast Medial Tangent",
                description="Medial tangential beam for breast treatment"
            )
            medial_tangent.beam_geometry = BeamGeometry()
            medial_tangent.beam_geometry.set_gantry_angle(300)
            medial_tangent.beam_geometry.set_collimator_angle(15)
            medial_tangent.beam_geometry.set_couch_angle(0)
            medial_tangent.add_tag("Breast")
            
            # Tạo mẫu chùm tia lateral tangent
            lateral_tangent = BeamTemplate(
                template_id="breast_lateral_tangent",
                name="Breast Lateral Tangent",
                description="Lateral tangential beam for breast treatment"
            )
            lateral_tangent.beam_geometry = BeamGeometry()
            lateral_tangent.beam_geometry.set_gantry_angle(120)
            lateral_tangent.beam_geometry.set_collimator_angle(345)
            lateral_tangent.beam_geometry.set_couch_angle(0)
            lateral_tangent.add_tag("Breast")
            
            breast_arrangement.add_beam_template(medial_tangent)
            breast_arrangement.add_beam_template(lateral_tangent)
            
            breast_arrangement.add_tag("Standard")
            breast_arrangement.add_tag("3DCRT")
            breast_arrangement.add_tag("Breast")
            
            self.save_beam_arrangement(breast_arrangement)
        else:
            logger.warning("Không thể tạo mẫu bố trí điều trị vú - thiếu mẫu chùm tia anterior")
            
    def initialize_standard_library(self):
        """
        Khởi tạo thư viện với các mẫu chùm tia và bố trí tiêu chuẩn.
        
        Phương thức này sẽ tạo các mẫu tiêu chuẩn nếu thư viện trống.
        """
        # Kiểm tra xem thư viện có trống không
        if not self.beam_templates and not self.beam_arrangements:
            logger.info("Khởi tạo thư viện chùm tia với các mẫu tiêu chuẩn")
            self.create_standard_beam_templates()
            self.create_standard_beam_arrangements()