"""
Module chứa các cấu trúc và thực thể liên quan đến cấu trúc giải phẫu trong lập kế hoạch xạ trị.

Module này định nghĩa các lớp để biểu diễn cấu trúc từ dữ liệu y khoa như ảnh CT, MRI và các
phương thức để vẽ, xử lý và tương tác với các cấu trúc này trong hệ thống lập kế hoạch.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Union, Set, Any
import logging
import numpy as np
import uuid

logger = logging.getLogger(__name__)


class StructureType(str, Enum):
    """Các loại cấu trúc giải phẫu trong lập kế hoạch xạ trị."""
    
    # Các loại cấu trúc chính
    EXTERNAL = "External"  # Đường viền ngoài của bệnh nhân
    PTV = "PTV"  # Planning Target Volume - Thể tích mục tiêu kế hoạch
    CTV = "CTV"  # Clinical Target Volume - Thể tích mục tiêu lâm sàng
    GTV = "GTV"  # Gross Tumor Volume - Thể tích u thô
    ITV = "ITV"  # Internal Target Volume - Thể tích mục tiêu nội tại
    
    # Các loại cơ quan nguy cấp
    OAR = "OAR"  # Organ At Risk - Cơ quan nguy cấp
    ORGAN = "Organ"  # Cơ quan/mô bình thường
    
    # Các loại cấu trúc bổ sung
    BOLUS = "Bolus"  # Bolus - Vật liệu đặt trên bề mặt da
    AVOIDANCE = "Avoidance"  # Vùng cần tránh
    SUPPORT = "Support"  # Cấu trúc hỗ trợ
    MARKER = "Marker"  # Điểm đánh dấu
    REGISTRATION = "Registration"  # Cấu trúc đăng ký
    ISOCENTER = "Isocenter"  # Tâm đẳng tâm
    
    # Các cấu trúc do người dùng tự định nghĩa
    UNDEFINED = "Undefined"  # Loại không xác định
    CUSTOM = "Custom"  # Tùy chỉnh


class StructureColor:
    """Quản lý màu sắc cho cấu trúc giải phẫu."""
    
    # Bảng màu mặc định cho mỗi loại cấu trúc
    DEFAULT_COLORS = {
        StructureType.EXTERNAL: (0, 255, 0),      # Xanh lá
        StructureType.PTV: (255, 0, 0),           # Đỏ
        StructureType.CTV: (255, 100, 0),         # Cam
        StructureType.GTV: (255, 0, 100),         # Hồng
        StructureType.ITV: (255, 200, 0),         # Vàng cam
        StructureType.OAR: (0, 0, 255),           # Xanh dương
        StructureType.ORGAN: (0, 200, 200),       # Xanh ngọc
        StructureType.BOLUS: (200, 200, 0),       # Vàng
        StructureType.AVOIDANCE: (200, 0, 200),   # Tím
        StructureType.SUPPORT: (150, 150, 150),   # Xám
        StructureType.MARKER: (255, 255, 255),    # Trắng
        StructureType.REGISTRATION: (100, 100, 255), # Xanh nhạt
        StructureType.ISOCENTER: (255, 255, 0),   # Vàng
        StructureType.UNDEFINED: (128, 128, 128), # Xám
        StructureType.CUSTOM: (128, 0, 128),      # Tím
    }
    
    @classmethod
    def get_default_color(cls, structure_type: StructureType) -> Tuple[int, int, int]:
        """
        Lấy màu mặc định cho loại cấu trúc.
        
        Parameters
        ----------
        structure_type : StructureType
            Loại cấu trúc
            
        Returns
        -------
        Tuple[int, int, int]
            Mã màu RGB (0-255)
        """
        return cls.DEFAULT_COLORS.get(structure_type, (128, 128, 128))
    
    @classmethod
    def get_random_color(cls) -> Tuple[int, int, int]:
        """
        Tạo màu ngẫu nhiên cho cấu trúc.
        
        Returns
        -------
        Tuple[int, int, int]
            Mã màu RGB (0-255)
        """
        import random
        return (
            random.randint(50, 230),
            random.randint(50, 230),
            random.randint(50, 230)
        )


class Structure:
    """
    Biểu diễn một cấu trúc giải phẫu trong lập kế hoạch xạ trị.
    
    Cấu trúc bao gồm thông tin như loại, tên, màu sắc, và dữ liệu đường viền
    trên mỗi lát cắt ảnh.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        structure_type: StructureType = StructureType.UNDEFINED,
        color: Optional[Tuple[int, int, int]] = None,
        parent_set_id: Optional[str] = None
    ):
        """
        Khởi tạo cấu trúc.
        
        Parameters
        ----------
        id : str
            ID duy nhất của cấu trúc
        name : str
            Tên mô tả của cấu trúc
        structure_type : StructureType, optional
            Loại cấu trúc, mặc định là UNDEFINED
        color : Optional[Tuple[int, int, int]], optional
            Màu RGB của cấu trúc, mặc định dựa vào loại cấu trúc
        parent_set_id : Optional[str], optional
            ID của tập cấu trúc chứa cấu trúc này
        """
        self.id = id
        self.name = name
        self.structure_type = structure_type
        self.color = color if color is not None else StructureColor.get_default_color(structure_type)
        self.parent_set_id = parent_set_id
        
        # Dữ liệu cơ bản của cấu trúc
        self.description = ""
        self.creation_date = datetime.now()
        self.modification_date = datetime.now()
        self.created_by = "QuangTPS"
        
        # Dữ liệu hình học
        self.contours: Dict[int, List[np.ndarray]] = {}  # Ánh xạ từ số lát cắt đến danh sách đường viền
        self.volume_cc: float = 0.0  # Thể tích tính bằng cm³
        self.surface_area_cm2: float = 0.0  # Diện tích bề mặt tính bằng cm²
        
        # Thuộc tính bổ sung
        self.display_opacity = 1.0  # Độ mờ đục khi hiển thị (0-1)
        self.is_visible = True  # Trạng thái hiển thị
        self.is_locked = False  # Khóa chỉnh sửa
        self.tags: Set[str] = set()  # Các thẻ gắn với cấu trúc
        self.metadata: Dict[str, Any] = {}  # Siêu dữ liệu bổ sung
    
    def add_contour(self, slice_index: int, points: np.ndarray) -> None:
        """
        Thêm đường viền cho một lát cắt.
        
        Parameters
        ----------
        slice_index : int
            Chỉ số của lát cắt
        points : np.ndarray
            Mảng điểm của đường viền, hình dạng (N, 3) cho N điểm trong không gian 3D
        """
        if slice_index not in self.contours:
            self.contours[slice_index] = []
        
        self.contours[slice_index].append(points)
        self.modification_date = datetime.now()
    
    def clear_contours(self) -> None:
        """Xóa tất cả đường viền của cấu trúc."""
        self.contours = {}
        self.modification_date = datetime.now()
    
    def calculate_volume(self, slice_thickness: float, pixel_spacing: Tuple[float, float]) -> float:
        """
        Tính thể tích của cấu trúc.
        
        Parameters
        ----------
        slice_thickness : float
            Độ dày của mỗi lát cắt (mm)
        pixel_spacing : Tuple[float, float]
            Khoảng cách giữa các pixel theo hai chiều (mm)
            
        Returns
        -------
        float
            Thể tích của cấu trúc (cm³)
        """
        # Chưa triển khai đầy đủ, sẽ cần thuật toán tính thể tích từ đường viền
        # Đây là phiên bản đơn giản cho ví dụ
        volume_mm3 = 0.0
        
        for slice_idx, contour_list in self.contours.items():
            slice_area = 0.0
            
            for contour in contour_list:
                # Tính diện tích của đường viền bằng công thức Green
                x = contour[:, 0]
                y = contour[:, 1]
                area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])
                
                # Chuyển đổi từ pixel sang mm²
                area *= pixel_spacing[0] * pixel_spacing[1]
                slice_area += area
            
            # Thể tích của lát cắt
            volume_mm3 += slice_area * slice_thickness
        
        # Chuyển từ mm³ sang cm³
        self.volume_cc = volume_mm3 / 1000.0
        return self.volume_cc
    
    def is_empty(self) -> bool:
        """
        Kiểm tra xem cấu trúc có rỗng không.
        
        Returns
        -------
        bool
            True nếu cấu trúc không có đường viền nào
        """
        return len(self.contours) == 0
    
    def get_center_of_mass(self) -> np.ndarray:
        """
        Tính tâm khối của cấu trúc.
        
        Returns
        -------
        np.ndarray
            Tọa độ 3D của tâm khối
        """
        if self.is_empty():
            return np.array([0.0, 0.0, 0.0])
        
        total_points = 0
        sum_points = np.zeros(3)
        
        for contour_list in self.contours.values():
            for contour in contour_list:
                num_points = contour.shape[0]
                sum_points += np.sum(contour, axis=0)
                total_points += num_points
        
        if total_points == 0:
            return np.array([0.0, 0.0, 0.0])
        
        return sum_points / total_points
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính hộp bao quanh cấu trúc.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (min_corner, max_corner) là tọa độ các đỉnh đối nhau của hộp
        """
        if self.is_empty():
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
        
        min_corner = np.array([float('inf'), float('inf'), float('inf')])
        max_corner = np.array([float('-inf'), float('-inf'), float('-inf')])
        
        for slice_idx, contour_list in self.contours.items():
            for contour in contour_list:
                min_point = np.min(contour, axis=0)
                max_point = np.max(contour, axis=0)
                
                min_corner = np.minimum(min_corner, min_point)
                max_corner = np.maximum(max_corner, max_point)
        
        return min_corner, max_corner
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi cấu trúc thành từ điển.
        
        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin cấu trúc
        """
        return {
            "id": self.id,
            "name": self.name,
            "structure_type": self.structure_type,
            "color": self.color,
            "description": self.description,
            "creation_date": self.creation_date.isoformat(),
            "modification_date": self.modification_date.isoformat(),
            "created_by": self.created_by,
            "volume_cc": self.volume_cc,
            "surface_area_cm2": self.surface_area_cm2,
            "parent_set_id": self.parent_set_id,
            "is_visible": self.is_visible,
            "is_locked": self.is_locked,
            "tags": list(self.tags),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Structure:
        """
        Tạo cấu trúc từ từ điển.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin cấu trúc
            
        Returns
        -------
        Structure
            Đối tượng cấu trúc mới
        """
        structure = cls(
            id=data["id"],
            name=data["name"],
            structure_type=data["structure_type"],
            color=tuple(data["color"]) if "color" in data else None,
            parent_set_id=data.get("parent_set_id")
        )
        
        structure.description = data.get("description", "")
        structure.created_by = data.get("created_by", "QuangTPS")
        structure.volume_cc = data.get("volume_cc", 0.0)
        structure.surface_area_cm2 = data.get("surface_area_cm2", 0.0)
        structure.is_visible = data.get("is_visible", True)
        structure.is_locked = data.get("is_locked", False)
        
        if "creation_date" in data:
            structure.creation_date = datetime.fromisoformat(data["creation_date"])
        if "modification_date" in data:
            structure.modification_date = datetime.fromisoformat(data["modification_date"])
        
        structure.tags = set(data.get("tags", []))
        structure.metadata = data.get("metadata", {})
        
        return structure
    
    def __str__(self) -> str:
        """Biểu diễn chuỗi của cấu trúc."""
        return f"Structure(id='{self.id}', name='{self.name}', type={self.structure_type})"


class StructureSet:
    """
    Biểu diễn một tập hợp các cấu trúc liên quan đến một nghiên cứu y khoa.
    
    Tập cấu trúc thường chứa các cấu trúc giải phẫu được vẽ trên cùng một bộ
    ảnh, như CT hoặc MRI.
    """
    
    def __init__(self, id: str, name: str, series_id: Optional[str] = None):
        """
        Khởi tạo tập cấu trúc.
        
        Parameters
        ----------
        id : str
            ID duy nhất của tập cấu trúc
        name : str
            Tên mô tả của tập cấu trúc
        series_id : Optional[str], optional
            ID của chuỗi ảnh mà tập cấu trúc liên kết với
        """
        self.id = id
        self.name = name
        self.description = ""
        self.series_id = series_id
        
        # Siêu dữ liệu
        self.creation_date = datetime.now()
        self.modification_date = datetime.now()
        self.created_by = "QuangTPS"
        
        # Danh sách các cấu trúc
        self.structures: Dict[str, Structure] = {}
        
        # Thuộc tính bổ sung
        self.modality = "CT"  # Phương thức hình ảnh: CT, MR, PT, etc.
        self.metadata: Dict[str, Any] = {}
    
    def add_structure(self, structure: Structure) -> None:
        """
        Thêm cấu trúc vào tập.
        
        Parameters
        ----------
        structure : Structure
            Cấu trúc cần thêm
        """
        structure.parent_set_id = self.id
        self.structures[structure.id] = structure
        self.modification_date = datetime.now()
    
    def remove_structure(self, structure_id: str) -> bool:
        """
        Xóa cấu trúc khỏi tập.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if structure_id in self.structures:
            del self.structures[structure_id]
            self.modification_date = datetime.now()
            return True
        return False
    
    def get_structure(self, structure_id: str) -> Optional[Structure]:
        """
        Lấy cấu trúc theo ID.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc cần lấy
            
        Returns
        -------
        Optional[Structure]
            Cấu trúc nếu tìm thấy, None nếu không
        """
        return self.structures.get(structure_id)
    
    def get_structure_by_name(self, name: str) -> Optional[Structure]:
        """
        Lấy cấu trúc theo tên.
        
        Parameters
        ----------
        name : str
            Tên của cấu trúc cần lấy
            
        Returns
        -------
        Optional[Structure]
            Cấu trúc đầu tiên có tên khớp, None nếu không tìm thấy
        """
        for structure in self.structures.values():
            if structure.name.lower() == name.lower():
                return structure
        return None
    
    def get_structures_by_type(self, structure_type: StructureType) -> List[Structure]:
        """
        Lấy danh sách cấu trúc theo loại.
        
        Parameters
        ----------
        structure_type : StructureType
            Loại cấu trúc cần lọc
            
        Returns
        -------
        List[Structure]
            Danh sách các cấu trúc có loại khớp
        """
        return [s for s in self.structures.values() if s.structure_type == structure_type]
    
    def get_target_structures(self) -> List[Structure]:
        """
        Lấy tất cả cấu trúc mục tiêu (PTV, CTV, GTV, ITV).
        
        Returns
        -------
        List[Structure]
            Danh sách các cấu trúc mục tiêu
        """
        target_types = {
            StructureType.PTV,
            StructureType.CTV,
            StructureType.GTV,
            StructureType.ITV
        }
        return [s for s in self.structures.values() if s.structure_type in target_types]
    
    def get_oar_structures(self) -> List[Structure]:
        """
        Lấy tất cả cấu trúc cơ quan nguy cấp.
        
        Returns
        -------
        List[Structure]
            Danh sách các cấu trúc OAR
        """
        return self.get_structures_by_type(StructureType.OAR)
    
    def get_external_structure(self) -> Optional[Structure]:
        """
        Lấy cấu trúc đường viền ngoài (thân bệnh nhân).
        
        Returns
        -------
        Optional[Structure]
            Cấu trúc EXTERNAL đầu tiên, None nếu không tìm thấy
        """
        externals = self.get_structures_by_type(StructureType.EXTERNAL)
        return externals[0] if externals else None
    
    def create_structure(
        self,
        name: str,
        structure_type: StructureType = StructureType.UNDEFINED,
        color: Optional[Tuple[int, int, int]] = None
    ) -> Structure:
        """
        Tạo cấu trúc mới và thêm vào tập.
        
        Parameters
        ----------
        name : str
            Tên mô tả của cấu trúc
        structure_type : StructureType, optional
            Loại cấu trúc, mặc định là UNDEFINED
        color : Optional[Tuple[int, int, int]], optional
            Màu RGB của cấu trúc, mặc định dựa vào loại cấu trúc
            
        Returns
        -------
        Structure
            Cấu trúc mới được tạo
        """
        structure_id = str(uuid.uuid4())
        
        if color is None:
            color = StructureColor.get_default_color(structure_type)
        
        structure = Structure(
            id=structure_id,
            name=name,
            structure_type=structure_type,
            color=color,
            parent_set_id=self.id
        )
        
        self.add_structure(structure)
        return structure
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi tập cấu trúc thành từ điển.
        
        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin tập cấu trúc
        """
        structure_dicts = {s_id: s.to_dict() for s_id, s in self.structures.items()}
        
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "series_id": self.series_id,
            "creation_date": self.creation_date.isoformat(),
            "modification_date": self.modification_date.isoformat(),
            "created_by": self.created_by,
            "modality": self.modality,
            "structures": structure_dicts,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StructureSet:
        """
        Tạo tập cấu trúc từ từ điển.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin tập cấu trúc
            
        Returns
        -------
        StructureSet
            Đối tượng tập cấu trúc mới
        """
        structure_set = cls(
            id=data["id"],
            name=data["name"],
            series_id=data.get("series_id")
        )
        
        structure_set.description = data.get("description", "")
        structure_set.created_by = data.get("created_by", "QuangTPS")
        structure_set.modality = data.get("modality", "CT")
        
        if "creation_date" in data:
            structure_set.creation_date = datetime.fromisoformat(data["creation_date"])
        if "modification_date" in data:
            structure_set.modification_date = datetime.fromisoformat(data["modification_date"])
        
        structure_set.metadata = data.get("metadata", {})
        
        # Tạo các cấu trúc
        if "structures" in data:
            for structure_id, structure_data in data["structures"].items():
                structure = Structure.from_dict(structure_data)
                structure_set.structures[structure_id] = structure
        
        return structure_set
    
    def __str__(self) -> str:
        """Biểu diễn chuỗi của tập cấu trúc."""
        return f"StructureSet(id='{self.id}', name='{self.name}', structures={len(self.structures)})"
