#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure Management Module

Module này cung cấp các class để quản lý cấu trúc giải phẫu
trong hệ thống lập kế hoạch xạ trị QuangTPS.
"""

import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class StructureType(Enum):
    """Loại cấu trúc giải phẫu."""

    PTV = "PTV"  # Planning Target Volume
    CTV = "CTV"  # Clinical Target Volume
    GTV = "GTV"  # Gross Target Volume
    OAR = "OAR"  # Organ at Risk
    NORMAL = "NORMAL"  # Normal tissue
    SUPPORT = "SUPPORT"  # Support structure
    AVOIDANCE = "AVOIDANCE"  # Avoidance structure
    OTHER = "OTHER"  # Other structures


class StructurePriority(Enum):
    """Mức độ ưu tiên của cấu trúc."""

    CRITICAL = 1  # Quan trọng nhất
    HIGH = 2  # Cao
    MEDIUM = 3  # Trung bình
    LOW = 4  # Thấp


@dataclass
class StructureProperties:
    """Thuộc tính của cấu trúc."""

    # Visual properties
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # RGB
    opacity: float = 0.7
    visible: bool = True

    # Physical properties
    density: float = 1.0  # g/cm³
    hu_value: Optional[int] = None  # Hounsfield Unit

    # Clinical properties
    type: StructureType = StructureType.OTHER
    priority: StructurePriority = StructurePriority.MEDIUM

    # Dose constraints
    max_dose: Optional[float] = None  # Gy
    mean_dose: Optional[float] = None  # Gy
    volume_constraints: Dict[str, float] = field(default_factory=dict)


class Structure:
    """
    Lớp đại diện cho một cấu trúc giải phẫu.

    Cấu trúc có thể là PTV, OAR, hoặc các cấu trúc khác
    được sử dụng trong lập kế hoạch xạ trị.
    """

    def __init__(
        self,
        name: str,
        structure_type: StructureType = StructureType.OTHER,
        priority: StructurePriority = StructurePriority.MEDIUM,
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        **kwargs,
    ):
        """
        Khởi tạo cấu trúc.

        Args:
            name: Tên cấu trúc
            structure_type: Loại cấu trúc
            priority: Mức độ ưu tiên
            color: Màu sắc hiển thị (RGB)
        """
        self.id = kwargs.get("id", f"struct_{hash(name)}")
        self.name = name
        self.properties = StructureProperties(
            type=structure_type, priority=priority, color=color
        )

        # Geometric data
        self.contours = []  # List of contour slices
        self.mask = None  # 3D binary mask
        self.volume = 0.0  # Volume in cm³

        # ROI statistics
        self.centroid = (0.0, 0.0, 0.0)
        self.bounding_box = None

        # Metadata
        self.created_date = datetime.now()
        self.modified_date = datetime.now()
        self.created_by = kwargs.get("created_by", "QuangTPS")

        # Dependencies
        self.parent_structure = kwargs.get("parent_structure")
        self.derived_from = kwargs.get("derived_from")

        logger.debug(f"Tạo structure: {self.name} ({self.properties.type.value})")

    def add_contour(self, contour_points: np.ndarray, slice_index: int):
        """
        Thêm contour cho một slice.

        Args:
            contour_points: Mảng điểm contour (N x 2)
            slice_index: Chỉ số slice
        """
        contour_data = {
            "slice_index": slice_index,
            "points": contour_points.copy(),
            "closed": True,
        }

        # Insert at correct position to maintain order
        inserted = False
        for i, existing_contour in enumerate(self.contours):
            if existing_contour["slice_index"] > slice_index:
                self.contours.insert(i, contour_data)
                inserted = True
                break

        if not inserted:
            self.contours.append(contour_data)

        self.modified_date = datetime.now()
        logger.debug(f"Thêm contour cho {self.name} tại slice {slice_index}")

    def remove_contour(self, slice_index: int):
        """Xóa contour tại slice index."""
        self.contours = [c for c in self.contours if c["slice_index"] != slice_index]
        self.modified_date = datetime.now()

    def get_contour_at_slice(self, slice_index: int) -> Optional[np.ndarray]:
        """Lấy contour tại slice index."""
        for contour in self.contours:
            if contour["slice_index"] == slice_index:
                return contour["points"]
        return None

    def get_slice_indices(self) -> List[int]:
        """Lấy danh sách các slice index có contour."""
        return [c["slice_index"] for c in self.contours]

    def calculate_volume(
        self,
        slice_thickness: float = 1.0,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ):
        """
        Tính toán thể tích từ các contours.

        Args:
            slice_thickness: Độ dày slice (mm)
            pixel_spacing: Khoảng cách pixel (mm)
        """
        total_volume = 0.0

        for contour in self.contours:
            # Calculate area of contour using shoelace formula
            points = contour["points"]
            if len(points) < 3:
                continue

            # Convert pixel coordinates to physical coordinates
            physical_points = points * np.array(pixel_spacing)

            # Shoelace formula for polygon area
            x = physical_points[:, 0]
            y = physical_points[:, 1]
            area = 0.5 * abs(
                sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1))
            )

            # Add volume for this slice
            total_volume += area * slice_thickness

        # Convert from mm³ to cm³
        self.volume = total_volume / 1000.0

        logger.debug(f"Tính toán volume cho {self.name}: {self.volume:.2f} cm³")
        return self.volume

    def create_mask(
        self,
        image_shape: Tuple[int, int, int],
        pixel_spacing: Tuple[float, float, float],
    ):
        """
        Tạo binary mask 3D từ các contours.

        Args:
            image_shape: Kích thước hình ảnh (depth, height, width)
            pixel_spacing: Khoảng cách pixel (z, y, x)
        """
        self.mask = np.zeros(image_shape, dtype=bool)

        for contour in self.contours:
            slice_idx = contour["slice_index"]
            if 0 <= slice_idx < image_shape[0]:
                points = contour["points"]
                if len(points) >= 3:
                    # Fill polygon on slice
                    try:
                        from skimage.draw import polygon

                        rr, cc = polygon(points[:, 1], points[:, 0], image_shape[1:])
                        # Ensure indices are within bounds
                        valid_mask = (
                            (rr >= 0)
                            & (rr < image_shape[1])
                            & (cc >= 0)
                            & (cc < image_shape[2])
                        )
                        self.mask[slice_idx, rr[valid_mask], cc[valid_mask]] = True
                    except ImportError:
                        logger.warning(
                            "skimage không khả dụng, sử dụng phương pháp đơn giản"
                        )
                        # Simple method: mark pixels inside bounding box
                        min_r, max_r = (
                            int(np.min(points[:, 1])),
                            int(np.max(points[:, 1])),
                        )
                        min_c, max_c = (
                            int(np.min(points[:, 0])),
                            int(np.max(points[:, 0])),
                        )

                        min_r = max(0, min_r)
                        max_r = min(image_shape[1] - 1, max_r)
                        min_c = max(0, min_c)
                        max_c = min(image_shape[2] - 1, max_c)

                        self.mask[slice_idx, min_r : max_r + 1, min_c : max_c + 1] = (
                            True
                        )
                    except Exception as e:
                        logger.warning(
                            f"Lỗi khi tạo polygon mask: {e}, sử dụng bounding box"
                        )
                        # Fallback to bounding box method
                        min_r, max_r = (
                            int(np.min(points[:, 1])),
                            int(np.max(points[:, 1])),
                        )
                        min_c, max_c = (
                            int(np.min(points[:, 0])),
                            int(np.max(points[:, 0])),
                        )

                        min_r = max(0, min_r)
                        max_r = min(image_shape[1] - 1, max_r)
                        min_c = max(0, min_c)
                        max_c = min(image_shape[2] - 1, max_c)

                        self.mask[slice_idx, min_r : max_r + 1, min_c : max_c + 1] = (
                            True
                        )

        logger.debug(f"Tạo mask cho {self.name}")
        return self.mask

    def get_mask(self) -> Optional[np.ndarray]:
        """Lấy binary mask."""
        return self.mask

    def calculate_centroid(self):
        """Tính toán centroid của cấu trúc."""
        if self.mask is None:
            logger.warning(f"Không có mask để tính centroid cho {self.name}")
            return self.centroid

        # Find centroid from mask
        indices = np.where(self.mask)
        if len(indices[0]) == 0:
            return self.centroid

        self.centroid = (
            float(np.mean(indices[2])),  # x
            float(np.mean(indices[1])),  # y
            float(np.mean(indices[0])),  # z
        )

        return self.centroid

    def calculate_bounding_box(self):
        """Tính toán bounding box."""
        if self.mask is None:
            return None

        indices = np.where(self.mask)
        if len(indices[0]) == 0:
            return None

        self.bounding_box = {
            "min_x": int(np.min(indices[2])),
            "max_x": int(np.max(indices[2])),
            "min_y": int(np.min(indices[1])),
            "max_y": int(np.max(indices[1])),
            "min_z": int(np.min(indices[0])),
            "max_z": int(np.max(indices[0])),
        }

        return self.bounding_box

    def get_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê của cấu trúc."""
        stats = {
            "name": self.name,
            "type": self.properties.type.value,
            "priority": self.properties.priority.value,
            "volume": self.volume,
            "centroid": self.centroid,
            "bounding_box": self.bounding_box,
            "num_contours": len(self.contours),
            "slice_range": (
                min(self.get_slice_indices()) if self.contours else 0,
                max(self.get_slice_indices()) if self.contours else 0,
            ),
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
        }

        return stats

    def set_color(self, color: Tuple[float, float, float]):
        """Đặt màu sắc."""
        self.properties.color = color
        self.modified_date = datetime.now()

    def set_opacity(self, opacity: float):
        """Đặt độ trong suốt."""
        self.properties.opacity = max(0.0, min(1.0, opacity))
        self.modified_date = datetime.now()

    def set_visibility(self, visible: bool):
        """Đặt tính hiển thị."""
        self.properties.visible = visible

    def copy(self, new_name: Optional[str] = None) -> "Structure":
        """Tạo bản sao của cấu trúc."""
        copy_name = new_name or f"{self.name}_copy"

        new_structure = Structure(
            name=copy_name,
            structure_type=self.properties.type,
            priority=self.properties.priority,
            color=self.properties.color,
        )

        # Copy contours
        new_structure.contours = [
            {
                "slice_index": c["slice_index"],
                "points": c["points"].copy(),
                "closed": c["closed"],
            }
            for c in self.contours
        ]

        # Copy other properties
        new_structure.volume = self.volume
        new_structure.centroid = self.centroid
        new_structure.bounding_box = (
            self.bounding_box.copy() if self.bounding_box else None
        )

        if self.mask is not None:
            new_structure.mask = self.mask.copy()

        return new_structure

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.properties.type.value,
            "priority": self.properties.priority.value,
            "color": list(self.properties.color),
            "opacity": self.properties.opacity,
            "visible": self.properties.visible,
            "volume": self.volume,
            "centroid": list(self.centroid),
            "bounding_box": self.bounding_box,
            "contours": self.contours,
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Structure":
        """Tạo Structure từ dictionary."""
        structure = cls(
            name=data["name"],
            structure_type=StructureType(data.get("type", "OTHER")),
            priority=StructurePriority(data.get("priority", 3)),
            color=tuple(data.get("color", [1.0, 0.0, 0.0])),
            id=data.get("id"),
            created_by=data.get("created_by", "QuangTPS"),
        )

        # Restore properties
        structure.properties.opacity = data.get("opacity", 0.7)
        structure.properties.visible = data.get("visible", True)
        structure.volume = data.get("volume", 0.0)
        structure.centroid = tuple(data.get("centroid", [0.0, 0.0, 0.0]))
        structure.bounding_box = data.get("bounding_box")
        structure.contours = data.get("contours", [])

        # Restore dates
        if "created_date" in data:
            structure.created_date = datetime.fromisoformat(data["created_date"])
        if "modified_date" in data:
            structure.modified_date = datetime.fromisoformat(data["modified_date"])

        return structure

    def __str__(self) -> str:
        return f"Structure(name='{self.name}', type={self.properties.type.value}, volume={self.volume:.2f}cm³)"

    def __repr__(self) -> str:
        return self.__str__()


class StructureSet:
    """
    Tập hợp các cấu trúc giải phẫu.

    Quản lý nhiều structures và cung cấp các phương thức
    để tìm kiếm, lọc và thao tác với chúng.
    """

    def __init__(self, name: str = "Default Structure Set"):
        """Khởi tạo StructureSet."""
        self.name = name
        self.structures: List[Structure] = []
        self.created_date = datetime.now()
        self.modified_date = datetime.now()

        logger.debug(f"Tạo StructureSet: {self.name}")

    def add_structure(self, structure: Structure):
        """Thêm structure vào set."""
        if structure not in self.structures:
            self.structures.append(structure)
            self.modified_date = datetime.now()
            logger.debug(f"Thêm structure {structure.name} vào {self.name}")

    def remove_structure(self, structure: Structure):
        """Xóa structure khỏi set."""
        if structure in self.structures:
            self.structures.remove(structure)
            self.modified_date = datetime.now()
            logger.debug(f"Xóa structure {structure.name} khỏi {self.name}")

    def get_structure_by_name(self, name: str) -> Optional[Structure]:
        """Tìm structure theo tên."""
        for structure in self.structures:
            if structure.name == name:
                return structure
        return None

    def get_structure_by_id(self, structure_id: str) -> Optional[Structure]:
        """Tìm structure theo ID."""
        for structure in self.structures:
            if structure.id == structure_id:
                return structure
        return None

    def get_structures_by_type(self, structure_type: StructureType) -> List[Structure]:
        """Lấy tất cả structures theo loại."""
        return [s for s in self.structures if s.properties.type == structure_type]

    def get_target_structures(self) -> List[Structure]:
        """Lấy tất cả target structures (PTV, CTV, GTV)."""
        target_types = [StructureType.PTV, StructureType.CTV, StructureType.GTV]
        return [s for s in self.structures if s.properties.type in target_types]

    def get_oar_structures(self) -> List[Structure]:
        """Lấy tất cả OAR structures."""
        return self.get_structures_by_type(StructureType.OAR)

    def get_all_structure_names(self) -> List[str]:
        """Lấy tên tất cả structures."""
        return [s.name for s in self.structures]

    def calculate_total_volume(self) -> float:
        """Tính tổng thể tích của tất cả structures."""
        return sum(s.volume for s in self.structures)

    def get_structure_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê tổng quan của structure set."""
        stats = {
            "name": self.name,
            "total_structures": len(self.structures),
            "structure_types": {},
            "total_volume": self.calculate_total_volume(),
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
        }

        # Count by type
        for structure in self.structures:
            struct_type = structure.properties.type.value
            stats["structure_types"][struct_type] = (
                stats["structure_types"].get(struct_type, 0) + 1
            )

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        return {
            "name": self.name,
            "structures": [s.to_dict() for s in self.structures],
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructureSet":
        """Tạo StructureSet từ dictionary."""
        structure_set = cls(name=data.get("name", "Default Structure Set"))

        # Restore structures
        for struct_data in data.get("structures", []):
            structure = Structure.from_dict(struct_data)
            structure_set.add_structure(structure)

        # Restore dates
        if "created_date" in data:
            structure_set.created_date = datetime.fromisoformat(data["created_date"])
        if "modified_date" in data:
            structure_set.modified_date = datetime.fromisoformat(data["modified_date"])

        return structure_set

    def __len__(self) -> int:
        return len(self.structures)

    def __iter__(self):
        return iter(self.structures)

    def __getitem__(self, index):
        return self.structures[index]

    def __str__(self) -> str:
        return f"StructureSet(name='{self.name}', structures={len(self.structures)})"

    def __repr__(self) -> str:
        return self.__str__()


# Factory functions
def create_structure(
    name: str,
    structure_type: str = "OTHER",
    priority: int = 3,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Structure:
    """
    Factory function để tạo Structure.

    Args:
        name: Tên cấu trúc
        structure_type: Loại cấu trúc (string)
        priority: Mức độ ưu tiên (1-4)
        color: Màu sắc RGB

    Returns:
        Structure instance
    """
    try:
        s_type = StructureType(structure_type.upper())
    except ValueError:
        s_type = StructureType.OTHER

    try:
        s_priority = StructurePriority(priority)
    except ValueError:
        s_priority = StructurePriority.MEDIUM

    return Structure(name=name, structure_type=s_type, priority=s_priority, color=color)


def create_structure_set(name: str = "Default Structure Set") -> StructureSet:
    """Factory function để tạo StructureSet."""
    return StructureSet(name=name)


# Standard color palette for structures
STRUCTURE_COLORS = {
    StructureType.PTV: (1.0, 0.0, 0.0),  # Red
    StructureType.CTV: (1.0, 0.5, 0.0),  # Orange
    StructureType.GTV: (1.0, 1.0, 0.0),  # Yellow
    StructureType.OAR: (0.0, 1.0, 0.0),  # Green
    StructureType.NORMAL: (0.0, 0.0, 1.0),  # Blue
    StructureType.SUPPORT: (0.5, 0.5, 0.5),  # Gray
    StructureType.AVOIDANCE: (1.0, 0.0, 1.0),  # Magenta
    StructureType.OTHER: (0.0, 1.0, 1.0),  # Cyan
}


def get_default_color(structure_type: StructureType) -> Tuple[float, float, float]:
    """Lấy màu mặc định cho loại cấu trúc."""
    return STRUCTURE_COLORS.get(structure_type, (1.0, 1.0, 1.0))
