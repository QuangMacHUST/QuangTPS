#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa tập hợp các cấu trúc giải phẫu trong QuangTPS.

Module này cung cấp lớp StructureSet để quản lý một tập hợp các đối tượng
Structure được sử dụng trong kế hoạch điều trị xạ trị.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Set, Union, Iterator
import uuid
import logging
import os
import json
from datetime import datetime

from quangtps.segmentation.structures.structure import Structure, StructureType, StructurePriority

logger = logging.getLogger(__name__)

class StructureSet:
    """
    Đối tượng đại diện cho một tập hợp các cấu trúc giải phẫu sử dụng trong kế hoạch điều trị.
    
    Attributes:
        id (str): ID duy nhất của tập cấu trúc
        name (str): Tên của tập cấu trúc
        description (str): Mô tả tập cấu trúc
        structures (Dict[str, Structure]): Từ điển các cấu trúc, với khóa là ID
        creation_date (str): Ngày tạo tập cấu trúc
        creator (str): Người tạo tập cấu trúc
        patient_id (str): ID của bệnh nhân
        study_id (str): ID của nghiên cứu
        metadata (Dict[str, Any]): Thông tin bổ sung
    """
    
    def __init__(self, name: str, 
                 description: str = "",
                 structures: Optional[List[Structure]] = None,
                 structure_set_id: Optional[str] = None,
                 creation_date: Optional[str] = None,
                 creator: str = "QuangTPS",
                 patient_id: Optional[str] = None,
                 study_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo tập cấu trúc.
        
        Parameters:
            name (str): Tên tập cấu trúc
            description (str): Mô tả tập cấu trúc
            structures (Optional[List[Structure]]): Danh sách các cấu trúc
            structure_set_id (Optional[str]): ID duy nhất (nếu None, tự động tạo)
            creation_date (Optional[str]): Ngày tạo (nếu None, lấy ngày hiện tại)
            creator (str): Người tạo tập cấu trúc
            patient_id (Optional[str]): ID của bệnh nhân
            study_id (Optional[str]): ID của nghiên cứu
            metadata (Optional[Dict[str, Any]]): Thông tin bổ sung
        """
        self.id = structure_set_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.structures: Dict[str, Structure] = {}
        self.creation_date = creation_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.creator = creator
        self.patient_id = patient_id
        self.study_id = study_id
        self.metadata = metadata or {}
        
        # Thêm các cấu trúc nếu được cung cấp
        if structures:
            for structure in structures:
                self.add_structure(structure)
    
    def __len__(self) -> int:
        """Số lượng cấu trúc trong tập."""
        return len(self.structures)
    
    def __iter__(self) -> Iterator[Structure]:
        """Iterator qua các cấu trúc."""
        return iter(self.structures.values())
    
    def __getitem__(self, key: Union[str, int]) -> Structure:
        """
        Truy cập cấu trúc bằng ID hoặc chỉ số.
        
        Parameters:
            key (Union[str, int]): ID của cấu trúc hoặc chỉ số trong danh sách
            
        Returns:
            Structure: Cấu trúc được tìm thấy
            
        Raises:
            KeyError: Nếu không tìm thấy cấu trúc với ID đã cho
            IndexError: Nếu chỉ số nằm ngoài phạm vi
        """
        if isinstance(key, str):
            if key in self.structures:
                return self.structures[key]
            raise KeyError(f"Structure with ID '{key}' not found")
        elif isinstance(key, int):
            if 0 <= key < len(self.structures):
                return list(self.structures.values())[key]
            raise IndexError(f"Index {key} out of range, structure set has {len(self.structures)} items")
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def get_structure_ids(self) -> List[str]:
        """
        Lấy danh sách các ID cấu trúc.
        
        Returns:
            List[str]: Danh sách ID cấu trúc
        """
        return list(self.structures.keys())
    
    def get_structure_names(self) -> List[str]:
        """
        Lấy danh sách tên các cấu trúc.
        
        Returns:
            List[str]: Danh sách tên cấu trúc
        """
        return [s.name for s in self.structures.values()]
    
    def add_structure(self, structure: Structure) -> None:
        """
        Thêm một cấu trúc vào tập.
        
        Parameters:
            structure (Structure): Cấu trúc cần thêm
        """
        self.structures[structure.id] = structure
    
    def remove_structure(self, structure_id: str) -> bool:
        """
        Xóa một cấu trúc khỏi tập.
        
        Parameters:
            structure_id (str): ID của cấu trúc cần xóa
            
        Returns:
            bool: True nếu xóa thành công, False nếu không tìm thấy
        """
        if structure_id in self.structures:
            del self.structures[structure_id]
            return True
        return False
    
    def get_structure_by_name(self, name: str) -> Optional[Structure]:
        """
        Tìm cấu trúc theo tên.
        
        Parameters:
            name (str): Tên cấu trúc cần tìm
            
        Returns:
            Optional[Structure]: Cấu trúc nếu tìm thấy, None nếu không
        """
        for structure in self.structures.values():
            if structure.name == name:
                return structure
        return None
    
    def get_structures_by_type(self, structure_type: StructureType) -> List[Structure]:
        """
        Lấy danh sách các cấu trúc theo loại.
        
        Parameters:
            structure_type (StructureType): Loại cấu trúc cần lấy
            
        Returns:
            List[Structure]: Danh sách các cấu trúc thuộc loại đã cho
        """
        return [s for s in self.structures.values() if s.type == structure_type]
    
    def get_targets(self) -> List[Structure]:
        """
        Lấy danh sách các cấu trúc mục tiêu (PTV, GTV, CTV).
        
        Returns:
            List[Structure]: Danh sách các cấu trúc mục tiêu
        """
        return self.get_structures_by_type(StructureType.TARGET)
    
    def get_oars(self) -> List[Structure]:
        """
        Lấy danh sách các cơ quan nguy cấp (OARs).
        
        Returns:
            List[Structure]: Danh sách các OARs
        """
        return self.get_structures_by_type(StructureType.OAR)
    
    def get_body(self) -> Optional[Structure]:
        """
        Lấy cấu trúc thân thể.
        
        Returns:
            Optional[Structure]: Cấu trúc thân thể nếu tồn tại, None nếu không
        """
        bodies = self.get_structures_by_type(StructureType.BODY)
        if bodies:
            return bodies[0]  # Lấy cấu trúc BODY đầu tiên
        return None
    
    def clear(self) -> None:
        """Xóa tất cả các cấu trúc trong tập."""
        self.structures.clear()
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Lưu tập cấu trúc vào file.
        
        Parameters:
            filepath (str): Đường dẫn tới file lưu trữ
            
        Returns:
            bool: True nếu lưu thành công, False nếu có lỗi
        """
        try:
            data = self.to_dict()
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save structure set to file: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'StructureSet':
        """
        Tạo tập cấu trúc từ file.
        
        Parameters:
            filepath (str): Đường dẫn tới file chứa tập cấu trúc
            
        Returns:
            StructureSet: Tập cấu trúc mới
            
        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu file không đúng định dạng
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Structure set file not found: {filepath}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid structure set file format: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load structure set from file: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi tập cấu trúc thành từ điển.
        
        Returns:
            Dict[str, Any]: Từ điển chứa dữ liệu tập cấu trúc
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "structures": [s.to_dict() for s in self.structures.values()],
            "creation_date": self.creation_date,
            "creator": self.creator,
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructureSet':
        """
        Tạo tập cấu trúc từ từ điển.
        
        Parameters:
            data (Dict[str, Any]): Từ điển chứa dữ liệu tập cấu trúc
            
        Returns:
            StructureSet: Tập cấu trúc mới
        """
        # Tạo các đối tượng Structure từ dữ liệu
        structures = []
        for structure_data in data.get("structures", []):
            try:
                structure = Structure.from_dict(structure_data)
                structures.append(structure)
            except Exception as e:
                logger.error(f"Error loading structure from data: {e}")
        
        # Tạo tập cấu trúc
        structure_set = cls(
            name=data.get("name", "Unnamed Structure Set"),
            description=data.get("description", ""),
            structures=structures,
            structure_set_id=data.get("id"),
            creation_date=data.get("creation_date"),
            creator=data.get("creator", "QuangTPS"),
            patient_id=data.get("patient_id"),
            study_id=data.get("study_id"),
            metadata=data.get("metadata", {})
        )
        
        return structure_set


class StructureSetData(StructureSet):
    """
    Phiên bản mở rộng của StructureSet với các thông tin và chức năng bổ sung.
    
    Lớp này kế thừa từ StructureSet và thêm vào các thông tin như lịch sử chỉnh sửa,
    dữ liệu theo dõi, và các tham chiếu đến ảnh liên quan.
    
    Attributes:
        id (str): ID duy nhất của tập cấu trúc
        name (str): Tên của tập cấu trúc
        description (str): Mô tả tập cấu trúc
        structures (Dict[str, Structure]): Từ điển các cấu trúc, với khóa là ID
        creation_date (str): Ngày tạo tập cấu trúc
        creator (str): Người tạo tập cấu trúc
        patient_id (str): ID của bệnh nhân
        study_id (str): ID của nghiên cứu
        metadata (Dict[str, Any]): Thông tin bổ sung
        modified_date (str): Ngày chỉnh sửa gần nhất
        modified_by (str): Người chỉnh sửa gần nhất
        image_series_id (str): ID của chuỗi ảnh liên quan
        version (int): Phiên bản của tập cấu trúc
        history (List[Dict]): Lịch sử chỉnh sửa
        clinical_status (str): Trạng thái lâm sàng
        approval_status (str): Trạng thái phê duyệt
        is_locked (bool): Trạng thái khóa
    """
    
    def __init__(self, name: str, 
                 description: str = "",
                 structures: Optional[List[Structure]] = None,
                 structure_set_id: Optional[str] = None,
                 creation_date: Optional[str] = None,
                 creator: str = "QuangTPS",
                 patient_id: Optional[str] = None,
                 study_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 modified_date: Optional[str] = None,
                 modified_by: Optional[str] = None,
                 image_series_id: Optional[str] = None,
                 version: int = 1,
                 history: Optional[List[Dict[str, Any]]] = None,
                 clinical_status: str = "Planning",
                 approval_status: str = "Unapproved",
                 is_locked: bool = False):
        """
        Khởi tạo đối tượng StructureSetData.
        
        Parameters:
            name (str): Tên tập cấu trúc
            description (str): Mô tả tập cấu trúc
            structures (Optional[List[Structure]]): Danh sách các cấu trúc
            structure_set_id (Optional[str]): ID duy nhất (nếu None, tự động tạo)
            creation_date (Optional[str]): Ngày tạo (nếu None, lấy ngày hiện tại)
            creator (str): Người tạo tập cấu trúc
            patient_id (Optional[str]): ID của bệnh nhân
            study_id (Optional[str]): ID của nghiên cứu
            metadata (Optional[Dict[str, Any]]): Thông tin bổ sung
            modified_date (Optional[str]): Ngày chỉnh sửa gần nhất
            modified_by (Optional[str]): Người chỉnh sửa gần nhất
            image_series_id (Optional[str]): ID của chuỗi ảnh liên quan
            version (int): Phiên bản của tập cấu trúc
            history (Optional[List[Dict]]): Lịch sử chỉnh sửa
            clinical_status (str): Trạng thái lâm sàng
            approval_status (str): Trạng thái phê duyệt
            is_locked (bool): Trạng thái khóa
        """
        super().__init__(
            name=name,
            description=description,
            structures=structures,
            structure_set_id=structure_set_id,
            creation_date=creation_date,
            creator=creator,
            patient_id=patient_id,
            study_id=study_id,
            metadata=metadata
        )
        
        # Thuộc tính bổ sung
        self.modified_date = modified_date or self.creation_date
        self.modified_by = modified_by or creator
        self.image_series_id = image_series_id
        self.version = version
        self.history = history or []
        self.clinical_status = clinical_status
        self.approval_status = approval_status
        self.is_locked = is_locked
    
    def update_modified_info(self, user: str = "QuangTPS") -> None:
        """
        Cập nhật thông tin chỉnh sửa.
        
        Parameters:
            user (str): Người thực hiện chỉnh sửa
        """
        self.modified_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.modified_by = user
        self.version += 1
        
        # Thêm vào lịch sử
        self.history.append({
            "timestamp": self.modified_date,
            "user": user,
            "version": self.version,
            "action": "Modified"
        })
    
    def lock(self, user: str = "QuangTPS") -> None:
        """
        Khóa tập cấu trúc để ngăn chỉnh sửa.
        
        Parameters:
            user (str): Người thực hiện khóa
        """
        self.is_locked = True
        self.update_modified_info(user)
        
        # Thêm hành động khóa vào lịch sử
        self.history.append({
            "timestamp": self.modified_date,
            "user": user,
            "version": self.version,
            "action": "Locked"
        })
    
    def unlock(self, user: str = "QuangTPS") -> None:
        """
        Mở khóa tập cấu trúc.
        
        Parameters:
            user (str): Người thực hiện mở khóa
        """
        self.is_locked = False
        self.update_modified_info(user)
        
        # Thêm hành động mở khóa vào lịch sử
        self.history.append({
            "timestamp": self.modified_date,
            "user": user,
            "version": self.version,
            "action": "Unlocked"
        })
    
    def approve(self, user: str = "QuangTPS", comments: str = "") -> None:
        """
        Phê duyệt tập cấu trúc.
        
        Parameters:
            user (str): Người thực hiện phê duyệt
            comments (str): Ghi chú khi phê duyệt
        """
        self.approval_status = "Approved"
        self.update_modified_info(user)
        
        # Thêm hành động phê duyệt vào lịch sử
        self.history.append({
            "timestamp": self.modified_date,
            "user": user,
            "version": self.version,
            "action": "Approved",
            "comments": comments
        })
    
    def unapprove(self, user: str = "QuangTPS", reason: str = "") -> None:
        """
        Hủy phê duyệt tập cấu trúc.
        
        Parameters:
            user (str): Người thực hiện hủy phê duyệt
            reason (str): Lý do hủy phê duyệt
        """
        self.approval_status = "Unapproved"
        self.update_modified_info(user)
        
        # Thêm hành động hủy phê duyệt vào lịch sử
        self.history.append({
            "timestamp": self.modified_date,
            "user": user,
            "version": self.version,
            "action": "Unapproved",
            "reason": reason
        })
    
    def add_structure(self, structure: Structure) -> None:
        """
        Thêm một cấu trúc và cập nhật thông tin chỉnh sửa.
        
        Parameters:
            structure (Structure): Cấu trúc cần thêm
        """
        if self.is_locked:
            logger.warning("Cannot add structure to locked structure set")
            return
        
        super().add_structure(structure)
        self.update_modified_info()
    
    def remove_structure(self, structure_id: str) -> bool:
        """
        Xóa một cấu trúc và cập nhật thông tin chỉnh sửa.
        
        Parameters:
            structure_id (str): ID của cấu trúc cần xóa
            
        Returns:
            bool: True nếu xóa thành công, False nếu không
        """
        if self.is_locked:
            logger.warning("Cannot remove structure from locked structure set")
            return False
        
        result = super().remove_structure(structure_id)
        if result:
            self.update_modified_info()
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thành từ điển với các thông tin bổ sung.
        
        Returns:
            Dict[str, Any]: Từ điển dữ liệu tập cấu trúc
        """
        data = super().to_dict()
        data.update({
            "modified_date": self.modified_date,
            "modified_by": self.modified_by,
            "image_series_id": self.image_series_id,
            "version": self.version,
            "history": self.history,
            "clinical_status": self.clinical_status,
            "approval_status": self.approval_status,
            "is_locked": self.is_locked
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructureSetData':
        """
        Tạo đối tượng từ từ điển.
        
        Parameters:
            data (Dict[str, Any]): Từ điển dữ liệu
            
        Returns:
            StructureSetData: Đối tượng mới
        """
        # Tạo các đối tượng Structure từ dữ liệu
        structures = []
        for structure_data in data.get("structures", []):
            try:
                structure = Structure.from_dict(structure_data)
                structures.append(structure)
            except Exception as e:
                logger.error(f"Error loading structure from data: {e}")
        
        # Tạo đối tượng mở rộng
        structure_set_data = cls(
            name=data.get("name", "Unnamed Structure Set"),
            description=data.get("description", ""),
            structures=structures,
            structure_set_id=data.get("id"),
            creation_date=data.get("creation_date"),
            creator=data.get("creator", "QuangTPS"),
            patient_id=data.get("patient_id"),
            study_id=data.get("study_id"),
            metadata=data.get("metadata", {}),
            modified_date=data.get("modified_date"),
            modified_by=data.get("modified_by"),
            image_series_id=data.get("image_series_id"),
            version=data.get("version", 1),
            history=data.get("history", []),
            clinical_status=data.get("clinical_status", "Planning"),
            approval_status=data.get("approval_status", "Unapproved"),
            is_locked=data.get("is_locked", False)
        )
        
        return structure_set_data
    
    @classmethod
    def from_structure_set(cls, structure_set: StructureSet, 
                          image_series_id: Optional[str] = None,
                          clinical_status: str = "Planning",
                          approval_status: str = "Unapproved") -> 'StructureSetData':
        """
        Tạo đối tượng StructureSetData từ StructureSet.
        
        Parameters:
            structure_set (StructureSet): Đối tượng StructureSet
            image_series_id (Optional[str]): ID chuỗi ảnh
            clinical_status (str): Trạng thái lâm sàng
            approval_status (str): Trạng thái phê duyệt
            
        Returns:
            StructureSetData: Đối tượng mới
        """
        return cls(
            name=structure_set.name,
            description=structure_set.description,
            structures=list(structure_set.structures.values()),
            structure_set_id=structure_set.id,
            creation_date=structure_set.creation_date,
            creator=structure_set.creator,
            patient_id=structure_set.patient_id,
            study_id=structure_set.study_id,
            metadata=structure_set.metadata,
            image_series_id=image_series_id,
            clinical_status=clinical_status,
            approval_status=approval_status
        )