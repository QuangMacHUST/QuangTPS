#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý bộ cấu trúc (Structure Sets) trong QuangTPS.

Cung cấp các lớp và hàm để tạo, chỉnh sửa và quản lý các bộ cấu trúc
dùng trong kế hoạch xạ trị, bao gồm ROI, contours và thuộc tính liên quan.
"""

import logging
import numpy as np
import uuid
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import pydicom
from copy import deepcopy
import json
import os
from datetime import datetime

from quangtps.core.exceptions import ValidationError
from quangtps.core.config import Config
from quangtps.dicom.rt_structure import RTStructure
from quangtps.dicom.dicom_utils import get_dicom_patient_info

logger = logging.getLogger(__name__)


class StructureType(str, Enum):
    """Phân loại cấu trúc theo mục đích sử dụng trong xạ trị."""
    PTV = "PTV"  # Planning Target Volume
    CTV = "CTV"  # Clinical Target Volume
    GTV = "GTV"  # Gross Tumor Volume
    ITV = "ITV"  # Internal Target Volume
    OAR = "OAR"  # Organ At Risk
    EXTERNAL = "EXTERNAL"  # Body contour
    SUPPORT = "SUPPORT"  # Support structures
    MARKER = "MARKER"  # Markers
    OTHER = "OTHER"  # Other structures


class StructurePriority(int, Enum):
    """Mức độ ưu tiên của cấu trúc, ảnh hưởng đến quá trình tối ưu hóa."""
    HIGH = 3      # Ưu tiên cao (target chính)
    MEDIUM = 2    # Ưu tiên trung bình
    LOW = 1       # Ưu tiên thấp
    NONE = 0      # Không ưu tiên


@dataclass
class Structure:
    """
    Lớp đại diện cho một cấu trúc trong kế hoạch xạ trị.
    
    Mỗi Structure chứa thông tin về một cấu trúc giải phẫu hoặc
    target trong kế hoạch điều trị, bao gồm dữ liệu contour, loại,
    thuộc tính và các metadata liên quan.
    """
    id: str  # ID duy nhất của cấu trúc
    name: str  # Tên cấu trúc
    type: StructureType = StructureType.OTHER  # Loại cấu trúc
    contours: Dict[int, List[np.ndarray]] = field(default_factory=dict)  # Contours theo slice
    color: Tuple[int, int, int] = (255, 0, 0)  # Màu RGB
    priority: StructurePriority = StructurePriority.MEDIUM  # Độ ưu tiên
    description: str = ""  # Mô tả
    meta: Dict[str, Any] = field(default_factory=dict)  # Metadata bổ sung
    visible: bool = True  # Trạng thái hiển thị
    locked: bool = False  # Trạng thái khóa chỉnh sửa
    
    def __post_init__(self):
        """Khởi tạo sau khi các thuộc tính đã được gán."""
        # Tạo ID nếu chưa có
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Chuyển đổi contours sang numpy arrays nếu cần
        for slice_idx, contour_list in self.contours.items():
            for i, contour in enumerate(contour_list):
                if not isinstance(contour, np.ndarray):
                    self.contours[slice_idx][i] = np.array(contour)
    
    def add_contour(self, slice_idx: int, contour: np.ndarray) -> None:
        """
        Thêm contour vào slice chỉ định.
        
        Parameters:
            slice_idx (int): Chỉ số slice
            contour (np.ndarray): Mảng các điểm contour, shape (n, 2) hoặc (n, 3)
        """
        # Đảm bảo contour là numpy array
        if not isinstance(contour, np.ndarray):
            contour = np.array(contour)
            
        # Kiểm tra shape
        if contour.ndim != 2 or contour.shape[1] not in (2, 3):
            raise ValidationError(f"Contour must have shape (n, 2) or (n, 3), got {contour.shape}")
        
        # Thêm vào dictionary contours
        if slice_idx not in self.contours:
            self.contours[slice_idx] = []
        
        self.contours[slice_idx].append(contour)
    
    def remove_contour(self, slice_idx: int, contour_idx: int = None) -> None:
        """
        Xóa contour khỏi slice chỉ định.
        
        Parameters:
            slice_idx (int): Chỉ số slice
            contour_idx (int, optional): Chỉ số contour cụ thể, None để xóa tất cả
        """
        if slice_idx not in self.contours:
            return
            
        if contour_idx is None:
            # Xóa tất cả contours trong slice
            del self.contours[slice_idx]
        elif 0 <= contour_idx < len(self.contours[slice_idx]):
            # Xóa contour cụ thể
            del self.contours[slice_idx][contour_idx]
            
            # Nếu không còn contour nào trong slice, xóa entry
            if not self.contours[slice_idx]:
                del self.contours[slice_idx]
    
    def get_contours(self, slice_idx: int = None) -> Union[Dict[int, List[np.ndarray]], List[np.ndarray]]:
        """
        Lấy contours theo slice chỉ định hoặc tất cả contours.
        
        Parameters:
            slice_idx (int, optional): Chỉ số slice, None để lấy tất cả
            
        Returns:
            Dict hoặc List: Contours theo yêu cầu
        """
        if slice_idx is not None:
            return self.contours.get(slice_idx, [])
        return self.contours
    
    def get_slices(self) -> List[int]:
        """
        Lấy danh sách các slice chứa contours.
        
        Returns:
            List[int]: Danh sách chỉ số slice
        """
        return sorted(self.contours.keys())
    
    def get_mask_for_slice(self, slice_idx: int, shape: Tuple[int, int]) -> np.ndarray:
        """
        Tạo mask nhị phân từ contours cho slice chỉ định.
        
        Parameters:
            slice_idx (int): Chỉ số slice
            shape (tuple): Kích thước mask (height, width)
            
        Returns:
            np.ndarray: Mask nhị phân
        """
        if slice_idx not in self.contours:
            return np.zeros(shape, dtype=np.uint8)
            
        mask = np.zeros(shape, dtype=np.uint8)
        for contour in self.contours[slice_idx]:
            points = contour[:, :2].astype(np.int32)  # Lấy x, y
            cv2.fillPoly(mask, [points], 1)
            
        return mask
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi structure thành dictionary để lưu trữ.
        
        Returns:
            Dict[str, Any]: Dictionary chứa thông tin structure
        """
        # Chuyển đổi contours sang list để json serialization
        serialized_contours = {}
        for slice_idx, contour_list in self.contours.items():
            serialized_contours[str(slice_idx)] = [contour.tolist() for contour in contour_list]
            
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "contours": serialized_contours,
            "color": self.color,
            "priority": self.priority.value,
            "description": self.description,
            "meta": self.meta,
            "visible": self.visible,
            "locked": self.locked
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Structure':
        """
        Tạo structure từ dictionary.
        
        Parameters:
            data (Dict[str, Any]): Dictionary chứa thông tin structure
            
        Returns:
            Structure: Đối tượng structure mới
        """
        # Chuyển đổi contours từ list sang numpy arrays
        contours = {}
        for slice_idx, contour_list in data.get("contours", {}).items():
            contours[int(slice_idx)] = [np.array(contour) for contour in contour_list]
            
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Unnamed"),
            type=StructureType(data.get("type", StructureType.OTHER.value)),
            contours=contours,
            color=tuple(data.get("color", (255, 0, 0))),
            priority=StructurePriority(data.get("priority", StructurePriority.MEDIUM.value)),
            description=data.get("description", ""),
            meta=data.get("meta", {}),
            visible=data.get("visible", True),
            locked=data.get("locked", False)
        )


class StructureSet:
    """
    Quản lý một bộ các cấu trúc liên quan đến một kế hoạch xạ trị.
    
    StructureSet chứa một tập hợp các Structure và cung cấp các phương thức
    để thêm, xóa, truy xuất và quản lý chúng. Nó cũng hỗ trợ xuất/nhập từ các định dạng
    DICOM RT Structure và các định dạng lưu trữ khác.
    """
    
    def __init__(self, id: str = None, name: str = ""):
        """
        Khởi tạo bộ cấu trúc mới.
        
        Parameters:
            id (str, optional): ID duy nhất cho bộ cấu trúc
            name (str, optional): Tên bộ cấu trúc
        """
        self.id = id or str(uuid.uuid4())
        self.name = name or f"StructureSet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.structures: Dict[str, Structure] = {}
        self.meta: Dict[str, Any] = {}
        self.associated_image_id = None  # ID của hình ảnh liên kết (CT/MRI/...)
        self.creation_date = datetime.now()
        self.modified_date = self.creation_date
    
    def add_structure(self, structure: Union[Structure, Dict[str, Any]]) -> str:
        """
        Thêm structure vào bộ cấu trúc.
        
        Parameters:
            structure (Structure hoặc Dict): Structure hoặc dictionary chứa thông tin structure
            
        Returns:
            str: ID của structure đã thêm
        """
        if isinstance(structure, dict):
            structure = Structure.from_dict(structure)
            
        self.structures[structure.id] = structure
        self.modified_date = datetime.now()
        return structure.id
    
    def remove_structure(self, structure_id: str) -> bool:
        """
        Xóa structure khỏi bộ cấu trúc.
        
        Parameters:
            structure_id (str): ID của structure cần xóa
            
        Returns:
            bool: True nếu xóa thành công
        """
        if structure_id in self.structures:
            del self.structures[structure_id]
            self.modified_date = datetime.now()
            return True
        return False
    
    def get_structure(self, structure_id: str) -> Optional[Structure]:
        """
        Lấy structure theo ID.
        
        Parameters:
            structure_id (str): ID của structure
            
        Returns:
            Structure hoặc None: Structure nếu tìm thấy, None nếu không
        """
        return self.structures.get(structure_id)
    
    def get_structure_by_name(self, name: str) -> Optional[Structure]:
        """
        Lấy structure theo tên.
        
        Parameters:
            name (str): Tên structure
            
        Returns:
            Structure hoặc None: Structure đầu tiên có tên phù hợp, None nếu không tìm thấy
        """
        for structure in self.structures.values():
            if structure.name.lower() == name.lower():
                return structure
        return None
    
    def get_structures_by_type(self, type: StructureType) -> List[Structure]:
        """
        Lấy danh sách structures theo loại.
        
        Parameters:
            type (StructureType): Loại structure
            
        Returns:
            List[Structure]: Danh sách structures có loại phù hợp
        """
        return [s for s in self.structures.values() if s.type == type]
    
    def get_all_structures(self) -> List[Structure]:
        """
        Lấy tất cả structures.
        
        Returns:
            List[Structure]: Danh sách tất cả structures
        """
        return list(self.structures.values())
    
    def copy_structure(self, structure_id: str, new_name: str = None) -> Optional[str]:
        """
        Tạo bản sao của structure.
        
        Parameters:
            structure_id (str): ID của structure cần sao chép
            new_name (str, optional): Tên mới cho structure sao chép
            
        Returns:
            str hoặc None: ID của structure mới, None nếu không tìm thấy structure gốc
        """
        structure = self.get_structure(structure_id)
        if not structure:
            return None
            
        # Tạo bản sao
        structure_copy = deepcopy(structure)
        structure_copy.id = str(uuid.uuid4())
        if new_name:
            structure_copy.name = new_name
        else:
            structure_copy.name = f"{structure.name}_copy"
            
        # Thêm vào danh sách
        return self.add_structure(structure_copy)
    
    def from_rt_structure(self, rt_structure: RTStructure) -> None:
        """
        Tạo StructureSet từ đối tượng RTStructure.
        
        Parameters:
            rt_structure (RTStructure): Đối tượng RTStructure từ module dicom
        """
        # Cập nhật metadata
        self.name = rt_structure.label
        self.meta = {
            "sop_instance_uid": rt_structure.sop_instance_uid,
            "series_instance_uid": rt_structure.series_instance_uid,
            "study_instance_uid": rt_structure.study_instance_uid,
            "patient_id": rt_structure.patient_id,
            "patient_name": rt_structure.patient_name
        }
        
        # Tạo structures từ RT structures
        for roi_id, roi_data in rt_structure.roi_contour_data.items():
            structure = Structure(
                id=str(roi_id),
                name=roi_data.get("name", f"ROI_{roi_id}"),
                type=self._map_rt_structure_type(roi_data.get("type", "")),
                color=roi_data.get("color", (255, 0, 0)),
                description=roi_data.get("description", "")
            )
            
            # Thêm contours
            for slice_z, contours in roi_data.get("contours", {}).items():
                for contour in contours:
                    structure.add_contour(int(slice_z), contour)
                    
            self.add_structure(structure)
    
    def to_rt_structure(self, reference_ct=None) -> RTStructure:
        """
        Chuyển đổi StructureSet thành đối tượng RTStructure.
        
        Parameters:
            reference_ct (optional): CT tham chiếu để lấy thông tin metadata
            
        Returns:
            RTStructure: Đối tượng RTStructure
        """
        from quangtps.dicom.rt_structure import RTStructure
        
        # Tạo đối tượng RTStructure mới
        rt_structure = RTStructure()
        rt_structure.label = self.name
        
        # Thiết lập metadata nếu có
        if reference_ct:
            rt_structure.set_reference_ct(reference_ct)
        
        # Thêm các ROI
        for structure in self.structures.values():
            roi_data = {
                "name": structure.name,
                "type": self._map_structure_type_to_rt(structure.type),
                "color": structure.color,
                "description": structure.description,
                "contours": {}
            }
            
            # Chuyển đổi contours
            for slice_idx, contours in structure.contours.items():
                roi_data["contours"][slice_idx] = contours
                
            rt_structure.add_roi(roi_data)
            
        return rt_structure
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Lưu StructureSet vào file JSON.
        
        Parameters:
            filepath (str): Đường dẫn file
            
        Returns:
            bool: True nếu lưu thành công
        """
        try:
            data = {
                "id": self.id,
                "name": self.name,
                "meta": self.meta,
                "associated_image_id": self.associated_image_id,
                "creation_date": self.creation_date.isoformat(),
                "modified_date": self.modified_date.isoformat(),
                "structures": {s.id: s.to_dict() for s in self.structures.values()}
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
            return True
        
        except Exception as e:
            logger.error(f"Error saving structure set to file: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['StructureSet']:
        """
        Tải StructureSet từ file JSON.
        
        Parameters:
            filepath (str): Đường dẫn file
            
        Returns:
            StructureSet hoặc None: StructureSet nếu tải thành công, None nếu thất bại
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            structure_set = cls(id=data.get("id"), name=data.get("name", ""))
            structure_set.meta = data.get("meta", {})
            structure_set.associated_image_id = data.get("associated_image_id")
            
            # Chuyển đổi datetime
            creation_date = data.get("creation_date")
            if creation_date:
                structure_set.creation_date = datetime.fromisoformat(creation_date)
                
            modified_date = data.get("modified_date")
            if modified_date:
                structure_set.modified_date = datetime.fromisoformat(modified_date)
            
            # Tải structures
            for structure_id, structure_data in data.get("structures", {}).items():
                structure_set.add_structure(structure_data)
                
            return structure_set
            
        except Exception as e:
            logger.error(f"Error loading structure set from file: {e}")
            return None
    
    def _map_rt_structure_type(self, rt_type: str) -> StructureType:
        """
        Ánh xạ loại cấu trúc từ RT Structure sang StructureType.
        
        Parameters:
            rt_type (str): Loại từ RT Structure
            
        Returns:
            StructureType: Loại tương ứng
        """
        type_map = {
            "PTV": StructureType.PTV,
            "CTV": StructureType.CTV,
            "GTV": StructureType.GTV,
            "ITV": StructureType.ITV,
            "ORGAN": StructureType.OAR,
            "EXTERNAL": StructureType.EXTERNAL,
            "SUPPORT": StructureType.SUPPORT,
            "MARKER": StructureType.MARKER
        }
        
        # Kiểm tra từng khóa
        for key, value in type_map.items():
            if key in rt_type.upper():
                return value
                
        return StructureType.OTHER
    
    def _map_structure_type_to_rt(self, structure_type: StructureType) -> str:
        """
        Ánh xạ StructureType sang loại cấu trúc RT Structure.
        
        Parameters:
            structure_type (StructureType): Loại cấu trúc
            
        Returns:
            str: Loại tương ứng cho RT Structure
        """
        type_map = {
            StructureType.PTV: "PTV",
            StructureType.CTV: "CTV",
            StructureType.GTV: "GTV",
            StructureType.ITV: "ITV",
            StructureType.OAR: "ORGAN",
            StructureType.EXTERNAL: "EXTERNAL",
            StructureType.SUPPORT: "SUPPORT",
            StructureType.MARKER: "MARKER",
            StructureType.OTHER: "OTHER"
        }
        
        return type_map.get(structure_type, "OTHER")