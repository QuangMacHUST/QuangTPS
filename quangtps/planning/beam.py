#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý chùm tia trong quá trình lập kế hoạch xạ trị.

Module này cung cấp các lớp và phương thức để quản lý bố trí chùm tia và
thiết lập các thông số chùm tia trong quá trình lập kế hoạch điều trị.
"""

import logging
import uuid
from enum import Enum
from typing import Dict, Optional, Any, Tuple
import copy

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.beams.beam_geometry import BeamGeometry
from quangtps.treatment.beams.beam_modifiers import Wedge, Block, Bolus, Compensator
from quangtps.treatment.beams.beam_library import BeamArrangementTemplate

logger = logging.getLogger(__name__)


class BeamArrangementType(str, Enum):
    """Enum cho các loại bố trí chùm tia."""
    STATIC = "Static"                    # Chùm tia tĩnh
    DYNAMIC = "Dynamic"                  # Chùm tia động
    CONFORMAL = "Conformal"              # Chùm tia tuân thủ
    IMRT = "IMRT"                        # Điều biến cường độ
    VMAT = "VMAT"                        # Điều trị cung tròn điều biến cường độ
    STEREOTACTIC = "Stereotactic"        # Định vị lập thể
    ELECTRON = "Electron"                # Chùm điện tử
    MIXED = "Mixed"                      # Hỗn hợp các loại chùm tia


class BeamSetup:
    """
    Lớp thiết lập chùm tia.
    
    Lớp này chứa toàn bộ thông tin thiết lập cho một chùm tia cụ thể trong kế hoạch điều trị,
    bao gồm các thông số hình học, thông số vật lý và các modifier (nêm, block, v.v.).
    """
    
    def __init__(
        self,
        beam_id: Optional[str] = None,
        name: str = "",
        beam: Optional[Beam] = None,
        beam_geometry: Optional[BeamGeometry] = None
    ):
        """
        Khởi tạo thiết lập chùm tia.
        
        Parameters
        ----------
        beam_id : str, optional
            ID duy nhất của chùm tia
        name : str, optional
            Tên hiển thị của chùm tia
        beam : Beam, optional
            Đối tượng Beam chứa thông tin chùm tia
        beam_geometry : BeamGeometry, optional
            Thông số hình học của chùm tia
        """
        self.beam_id = beam_id if beam_id else str(uuid.uuid4())
        self.name = name
        self.beam = beam if beam else Beam(beam_name=name)
        self.beam_geometry = beam_geometry if beam_geometry else BeamGeometry()
        
        self.wedge = None                # Optional[Wedge]
        self.blocks = []                 # List[Block]
        self.bolus = None                # Optional[Bolus]
        self.compensator = None          # Optional[Compensator]
        
        self.mlc_positions = {}          # Dict[int, Tuple[float, float]]
        self.jaw_positions = (20.0, 20.0, 20.0, 20.0)  # (X1, X2, Y1, Y2)
        
        self.field_size = (10.0, 10.0)   # Field size at isocenter (width, height)
        self.monitor_units = 0.0         # Số lượng Monitor Units (MU)
        self.weight = 1.0                # Trọng số dùng cho tính toán liều
        self.isocenter_position = (0.0, 0.0, 0.0)  # (x, y, z) coordinates
        
        self.metadata = {}               # Dict[str, Any]
        
    def set_name(self, name: str):
        """
        Đặt tên cho chùm tia.
        
        Parameters
        ----------
        name : str
            Tên hiển thị mới
        """
        self.name = name
        
    def set_beam(self, beam: Beam):
        """
        Đặt đối tượng Beam.
        
        Parameters
        ----------
        beam : Beam
            Đối tượng Beam mới
        """
        self.beam = beam
        
    def set_beam_geometry(self, beam_geometry: BeamGeometry):
        """
        Đặt thông số hình học cho chùm tia.
        
        Parameters
        ----------
        beam_geometry : BeamGeometry
            Thông số hình học mới
        """
        self.beam_geometry = beam_geometry
        
    def set_wedge(self, wedge: Optional[Wedge]):
        """
        Đặt nêm (wedge) cho chùm tia.
        
        Parameters
        ----------
        wedge : Wedge, optional
            Đối tượng Wedge, hoặc None để xóa nêm
        """
        self.wedge = wedge
        
    def add_block(self, block: Block):
        """
        Thêm block cho chùm tia.
        
        Parameters
        ----------
        block : Block
            Đối tượng Block cần thêm
        """
        self.blocks.append(block)
        
    def clear_blocks(self):
        """Xóa tất cả block."""
        self.blocks = []
        
    def set_bolus(self, bolus: Optional[Bolus]):
        """
        Đặt bolus cho chùm tia.
        
        Parameters
        ----------
        bolus : Bolus, optional
            Đối tượng Bolus, hoặc None để xóa bolus
        """
        self.bolus = bolus
        
    def set_compensator(self, compensator: Optional[Compensator]):
        """
        Đặt compensator cho chùm tia.
        
        Parameters
        ----------
        compensator : Compensator, optional
            Đối tượng Compensator, hoặc None để xóa compensator
        """
        self.compensator = compensator
        
    def set_mlc_positions(self, positions: Dict[int, Tuple[float, float]]):
        """
        Đặt vị trí các lá MLC.
        
        Parameters
        ----------
        positions : Dict[int, Tuple[float, float]]
            Dictionary chứa vị trí các lá MLC, với khóa là số thứ tự của lá
            và giá trị là tuple (vị_trí_lá_trái, vị_trí_lá_phải)
        """
        self.mlc_positions = positions
        
    def set_jaw_positions(self, x1: float, x2: float, y1: float, y2: float):
        """
        Đặt vị trí các jaw.
        
        Parameters
        ----------
        x1 : float
            Vị trí jaw X1
        x2 : float
            Vị trí jaw X2
        y1 : float
            Vị trí jaw Y1
        y2 : float
            Vị trí jaw Y2
        """
        self.jaw_positions = (x1, x2, y1, y2)
        
    def set_field_size(self, width: float, height: float):
        """
        Đặt kích thước trường chiếu tại isocenter.
        
        Parameters
        ----------
        width : float
            Chiều rộng trường (cm)
        height : float
            Chiều cao trường (cm)
        """
        self.field_size = (width, height)
        # Cập nhật vị trí jaw tương ứng
        self.jaw_positions = (width/2, width/2, height/2, height/2)
        
    def set_monitor_units(self, monitor_units: float):
        """
        Đặt số lượng Monitor Units.
        
        Parameters
        ----------
        monitor_units : float
            Số lượng Monitor Units (MU)
        """
        self.monitor_units = monitor_units
        
    def set_weight(self, weight: float):
        """
        Đặt trọng số cho chùm tia.
        
        Parameters
        ----------
        weight : float
            Trọng số mới
        """
        self.weight = weight
        
    def set_isocenter_position(self, x: float, y: float, z: float):
        """
        Đặt vị trí isocenter.
        
        Parameters
        ----------
        x : float
            Tọa độ x (mm)
        y : float
            Tọa độ y (mm)
        z : float
            Tọa độ z (mm)
        """
        self.isocenter_position = (x, y, z)
        
    def set_metadata(self, key: str, value: Any):
        """
        Đặt một trường metadata.
        
        Parameters
        ----------
        key : str
            Tên trường
        value : Any
            Giá trị trường
        """
        self.metadata[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thiết lập chùm tia thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin thiết lập chùm tia
        """
        data = {
            "beam_id": self.beam_id,
            "name": self.name,
            "beam": self.beam.to_dict() if self.beam else None,
            "beam_geometry": self.beam_geometry.to_dict() if self.beam_geometry else None,
            "wedge": self.wedge.to_dict() if self.wedge else None,
            "blocks": [block.to_dict() for block in self.blocks],
            "bolus": self.bolus.to_dict() if self.bolus else None,
            "compensator": self.compensator.to_dict() if self.compensator else None,
            "mlc_positions": self.mlc_positions,
            "jaw_positions": self.jaw_positions,
            "field_size": self.field_size,
            "monitor_units": self.monitor_units,
            "weight": self.weight,
            "isocenter_position": self.isocenter_position,
            "metadata": self.metadata
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeamSetup':
        """
        Tạo đối tượng BeamSetup từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin thiết lập chùm tia
            
        Returns
        -------
        BeamSetup
            Đối tượng BeamSetup được tạo từ dữ liệu
        """
        setup = cls(
            beam_id=data.get("beam_id"),
            name=data.get("name", "")
        )
        
        # Phục hồi các đối tượng
        if "beam" in data and data["beam"]:
            setup.beam = Beam.from_dict(data["beam"])
            
        if "beam_geometry" in data and data["beam_geometry"]:
            setup.beam_geometry = BeamGeometry.from_dict(data["beam_geometry"])
            
        if "wedge" in data and data["wedge"]:
            setup.wedge = Wedge.from_dict(data["wedge"])
            
        if "blocks" in data:
            setup.blocks = [Block.from_dict(block_data) for block_data in data["blocks"]]
            
        if "bolus" in data and data["bolus"]:
            setup.bolus = Bolus.from_dict(data["bolus"])
            
        if "compensator" in data and data["compensator"]:
            setup.compensator = Compensator.from_dict(data["compensator"])
            
        # Phục hồi các thông số khác
        if "mlc_positions" in data:
            setup.mlc_positions = data["mlc_positions"]
            
        if "jaw_positions" in data:
            setup.jaw_positions = data["jaw_positions"]
            
        if "field_size" in data:
            setup.field_size = data["field_size"]
            
        if "monitor_units" in data:
            setup.monitor_units = data["monitor_units"]
            
        if "weight" in data:
            setup.weight = data["weight"]
            
        if "isocenter_position" in data:
            setup.isocenter_position = data["isocenter_position"]
            
        if "metadata" in data:
            setup.metadata = data["metadata"]
            
        return setup
    
    def copy(self) -> 'BeamSetup':
        """
        Tạo một bản sao của thiết lập chùm tia.
        
        Returns
        -------
        BeamSetup
            Bản sao của đối tượng hiện tại
        """
        # Tạo dictionary và tạo lại từ dictionary để có bản sao sâu
        data = self.to_dict()
        return BeamSetup.from_dict(data)
    
    def __str__(self) -> str:
        """Biểu diễn chuỗi của thiết lập chùm tia."""
        return f"{self.name} [{self.beam_id}] - {self.field_size[0]}x{self.field_size[1]} cm, MU: {self.monitor_units}"


class BeamArrangement:
    """
    Lớp bố trí chùm tia.
    
    Lớp này quản lý tất cả các chùm tia trong một kế hoạch điều trị,
    bao gồm các chùm tia tĩnh, động, IMRT, VMAT, v.v.
    """
    
    def __init__(
        self,
        arrangement_id: Optional[str] = None,
        name: str = "",
        arrangement_type: BeamArrangementType = BeamArrangementType.STATIC
    ):
        """
        Khởi tạo bố trí chùm tia.
        
        Parameters
        ----------
        arrangement_id : str, optional
            ID duy nhất của bố trí
        name : str, optional
            Tên hiển thị của bố trí
        arrangement_type : BeamArrangementType, optional
            Loại bố trí chùm tia
        """
        self.arrangement_id = arrangement_id if arrangement_id else str(uuid.uuid4())
        self.name = name
        self.arrangement_type = arrangement_type
        
        self.beam_setups = {}          # Dict[str, BeamSetup]
        self.isocenter = (0.0, 0.0, 0.0)  # Common isocenter (x, y, z) coordinates
        self.normalization_value = 100.0  # Giá trị chuẩn hóa (%)
        self.normalization_point = None   # Điểm chuẩn hóa (x, y, z)
        
        self.metadata = {}               # Dict[str, Any]
        
    def add_beam_setup(self, beam_setup: BeamSetup):
        """
        Thêm thiết lập chùm tia vào bố trí.
        
        Parameters
        ----------
        beam_setup : BeamSetup
            Thiết lập chùm tia cần thêm
        """
        self.beam_setups[beam_setup.beam_id] = beam_setup
        
    def get_beam_setup(self, beam_id: str) -> Optional[BeamSetup]:
        """
        Lấy thiết lập chùm tia theo ID.
        
        Parameters
        ----------
        beam_id : str
            ID của chùm tia cần lấy
            
        Returns
        -------
        BeamSetup, optional
            Thiết lập chùm tia, hoặc None nếu không tìm thấy
        """
        return self.beam_setups.get(beam_id)
        
    def remove_beam_setup(self, beam_id: str) -> bool:
        """
        Xóa thiết lập chùm tia theo ID.
        
        Parameters
        ----------
        beam_id : str
            ID của chùm tia cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if beam_id in self.beam_setups:
            del self.beam_setups[beam_id]
            return True
        return False
        
    def set_name(self, name: str):
        """
        Đặt tên cho bố trí chùm tia.
        
        Parameters
        ----------
        name : str
            Tên hiển thị mới
        """
        self.name = name
        
    def set_arrangement_type(self, arrangement_type: BeamArrangementType):
        """
        Đặt loại bố trí chùm tia.
        
        Parameters
        ----------
        arrangement_type : BeamArrangementType
            Loại bố trí mới
        """
        self.arrangement_type = arrangement_type
        
    def set_isocenter(self, x: float, y: float, z: float):
        """
        Đặt isocenter chung cho tất cả chùm tia.
        
        Parameters
        ----------
        x : float
            Tọa độ x (mm)
        y : float
            Tọa độ y (mm)
        z : float
            Tọa độ z (mm)
        """
        self.isocenter = (x, y, z)
        
        # Cập nhật isocenter cho tất cả chùm tia
        for _, beam_setup in self.beam_setups.items():
            beam_setup.set_isocenter_position(x, y, z)
        
    def set_normalization_value(self, value: float):
        """
        Đặt giá trị chuẩn hóa liều.
        
        Parameters
        ----------
        value : float
            Giá trị chuẩn hóa (%)
        """
        self.normalization_value = value
        
    def set_normalization_point(self, x: float, y: float, z: float):
        """
        Đặt điểm chuẩn hóa liều.
        
        Parameters
        ----------
        x : float
            Tọa độ x (mm)
        y : float
            Tọa độ y (mm)
        z : float
            Tọa độ z (mm)
        """
        self.normalization_point = (x, y, z)
        
    def set_metadata(self, key: str, value: Any):
        """
        Đặt một trường metadata.
        
        Parameters
        ----------
        key : str
            Tên trường
        value : Any
            Giá trị trường
        """
        self.metadata[key] = value
        
    def normalize_weights(self):
        """
        Chuẩn hóa các trọng số của các chùm tia sao cho tổng bằng 1.0.
        """
        total_weight = sum(beam.weight for beam in self.beam_setups.values())
        
        if total_weight > 0:
            for _, beam_setup in self.beam_setups.items():
                beam_setup.set_weight(beam_setup.weight / total_weight)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi bố trí chùm tia thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin bố trí chùm tia
        """
        data = {
            "arrangement_id": self.arrangement_id,
            "name": self.name,
            "arrangement_type": self.arrangement_type.value,
            "beam_setups": {beam_id: setup.to_dict() for beam_id, setup in self.beam_setups.items()},
            "isocenter": self.isocenter,
            "normalization_value": self.normalization_value,
            "normalization_point": self.normalization_point,
            "metadata": self.metadata
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeamArrangement':
        """
        Tạo đối tượng BeamArrangement từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin bố trí chùm tia
            
        Returns
        -------
        BeamArrangement
            Đối tượng BeamArrangement được tạo từ dữ liệu
        """
        arrangement = cls(
            arrangement_id=data.get("arrangement_id"),
            name=data.get("name", ""),
            arrangement_type=BeamArrangementType(data.get("arrangement_type", BeamArrangementType.STATIC.value))
        )
        
        # Phục hồi các thiết lập chùm tia
        if "beam_setups" in data:
            for beam_id, setup_data in data["beam_setups"].items():
                arrangement.beam_setups[beam_id] = BeamSetup.from_dict(setup_data)
                
        if "isocenter" in data:
            arrangement.isocenter = data["isocenter"]
            
        if "normalization_value" in data:
            arrangement.normalization_value = data["normalization_value"]
            
        if "normalization_point" in data:
            arrangement.normalization_point = data["normalization_point"]
            
        if "metadata" in data:
            arrangement.metadata = data["metadata"]
            
        return arrangement
    
    def copy(self) -> 'BeamArrangement':
        """
        Tạo một bản sao của bố trí chùm tia.
        
        Returns
        -------
        BeamArrangement
            Bản sao của đối tượng hiện tại
        """
        # Tạo dictionary và tạo lại từ dictionary để có bản sao sâu
        data = self.to_dict()
        return BeamArrangement.from_dict(data)
    
    def create_from_template(self, template: BeamArrangementTemplate) -> None:
        """
        Tạo bố trí chùm tia từ một mẫu có sẵn.
        
        Parameters
        ----------
        template : BeamArrangementTemplate
            Mẫu bố trí chùm tia
        """
        # Lấy thông tin từ template
        self.name = template.name
        
        # Xác định loại bố trí dựa trên kỹ thuật trong template
        technique = template.technique.upper()
        if "IMRT" in technique:
            self.arrangement_type = BeamArrangementType.IMRT
        elif "VMAT" in technique:
            self.arrangement_type = BeamArrangementType.VMAT
        elif "SBRT" in technique or "SRS" in technique:
            self.arrangement_type = BeamArrangementType.STEREOTACTIC
        elif "3DCRT" in technique:
            self.arrangement_type = BeamArrangementType.CONFORMAL
        else:
            self.arrangement_type = BeamArrangementType.STATIC
            
        # Tạo các chùm tia từ template
        beams = template.create_beams()
        for i, beam in enumerate(beams):
            beam_setup = BeamSetup(
                name=f"Field {i+1}",
                beam=beam
            )
            
            # Lấy thông tin hình học từ template nếu có
            if i < len(template.beam_templates) and template.beam_templates[i].beam_geometry:
                beam_setup.set_beam_geometry(copy.deepcopy(template.beam_templates[i].beam_geometry))
                
            self.add_beam_setup(beam_setup)
            
        # Sao chép metadata từ template
        for key, value in template.metadata.items():
            self.set_metadata(key, value)
    
    def __str__(self) -> str:
        """Biểu diễn chuỗi của bố trí chùm tia."""
        return f"{self.name} [{self.arrangement_id}] - {self.arrangement_type.value} with {len(self.beam_setups)} beams"