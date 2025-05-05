#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Structure biểu diễn cấu trúc giải phẫu.

Module này cung cấp các lớp và hàm để biểu diễn, xử lý và hiển thị
các cấu trúc giải phẫu (ROI - Regions of Interest) trong kế hoạch xạ trị.
"""

import logging
import uuid
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)


class StructureType(Enum):
    """Các loại cấu trúc giải phẫu."""

    TARGET = auto()  # PTV, CTV, GTV, etc.
    OAR = auto()  # Organ At Risk
    EXTERNAL = auto()  # Bề mặt ngoài cơ thể
    BOLUS = auto()  # Bolus
    SUPPORT = auto()  # Các cấu trúc hỗ trợ
    OTHER = auto()  # Loại khác
    UNKNOWN = auto()  # Không xác định


class Structure:
    """
    Biểu diễn cấu trúc giải phẫu.

    Lưu trữ thông tin về một cấu trúc giải phẫu cụ thể, bao gồm
    contour, mask, loại cấu trúc, màu sắc và các thuộc tính khác.
    """

    def __init__(
        self,
        structure_id: Optional[str] = None,
        name: str = "Unknown Structure",
        structure_type: Union[StructureType, str] = StructureType.UNKNOWN,
        color: Tuple[float, float, float] = None,
        mask: np.ndarray = None,
        contours: List[List[np.ndarray]] = None,
    ):
        """
        Khởi tạo một cấu trúc giải phẫu.

        Args:
            structure_id: ID duy nhất của cấu trúc
            name: Tên cấu trúc
            structure_type: Loại cấu trúc
            color: Màu sắc của cấu trúc (r, g, b) với giá trị 0-1
            mask: Mảng numpy 3D biểu diễn mask của cấu trúc
            contours: Danh sách các contour 2D theo từng slice
        """
        self.id = structure_id or str(uuid.uuid4())[:8]
        self.name = name

        # Xử lý structure_type
        if isinstance(structure_type, str):
            try:
                self.type = StructureType[structure_type.upper()]
            except (KeyError, AttributeError):
                self.type = StructureType.UNKNOWN
        else:
            self.type = structure_type

        # Tạo màu ngẫu nhiên nếu không cung cấp
        if color is None:
            self.color = (np.random.random(), np.random.random(), np.random.random())
        else:
            self.color = color

        # Dữ liệu cấu trúc
        self.mask = mask
        self.contours = contours or []

        # Thông tin bổ sung
        self.volume = 0.0  # Thể tích (cc)
        self.metadata = {}  # Thông tin bổ sung

        # Nếu có mask, tính thể tích
        if self.mask is not None:
            self.calculate_volume()

    def set_mask(self, mask: np.ndarray) -> None:
        """
        Thiết lập mask cho cấu trúc.

        Args:
            mask: Mảng 3D binary mask (1 = bên trong, 0 = bên ngoài)
        """
        if mask is None:
            return

        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)

        self.mask = mask.astype(bool)

        # Cập nhật thể tích
        self.calculate_volume()

        # Cập nhật contours nếu cần
        if not self.contours:
            self._generate_contours_from_mask()

    def set_contours(self, contours: List[List[np.ndarray]]) -> None:
        """
        Thiết lập contours cho cấu trúc.

        Args:
            contours: Danh sách các contour 2D theo từng slice
        """
        self.contours = contours

        # Cập nhật mask nếu cần
        if self.mask is None:
            self._generate_mask_from_contours()

        # Cập nhật thể tích
        self.calculate_volume()

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Thiết lập metadata cho cấu trúc.

        Args:
            metadata: Thông tin bổ sung
        """
        self.metadata = metadata

    def calculate_volume(self) -> float:
        """
        Tính toán thể tích của cấu trúc.

        Returns:
            Thể tích (cc)
        """
        if self.mask is None:
            return 0.0

        try:
            # Lấy kích thước voxel từ metadata nếu có
            voxel_size = self.metadata.get("voxel_size", (1.0, 1.0, 1.0))

            # Tính thể tích (cc)
            voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]
            self.volume = np.sum(self.mask) * voxel_volume / 1000.0  # Chuyển về cc

            return self.volume

        except Exception as e:
            logger.error(f"Lỗi khi tính thể tích cấu trúc {self.name}: {e}")
            return 0.0

    def _generate_contours_from_mask(self) -> None:
        """Tạo contours từ mask."""
        if self.mask is None:
            return

        try:
            import cv2

            # Tạo contours cho từng slice
            contours = []
            for z in range(self.mask.shape[0]):
                slice_contours = []

                # Chuyển sang định dạng phù hợp với OpenCV
                slice_mask = self.mask[z].astype(np.uint8)

                # Tìm các contour trên slice này
                result = cv2.findContours(
                    slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                slice_contours = result[0] if len(result) == 2 else result[1]

                # Thêm vào danh sách kết quả
                contours.append(slice_contours)

            self.contours = contours

        except ImportError:
            logger.warning("Không thể import OpenCV để tạo contours từ mask")
        except Exception as e:
            logger.error(f"Lỗi khi tạo contours từ mask: {e}")

    def _generate_mask_from_contours(self) -> None:
        """Tạo mask từ contours."""
        if not self.contours:
            return

        try:
            import cv2

            # Lấy kích thước ảnh từ metadata hoặc ước lượng
            image_shape = self.metadata.get("image_shape", None)
            if image_shape is None:
                # Ước lượng kích thước từ contours
                max_x, max_y, max_z = 0, 0, 0
                for z, slice_contours in enumerate(self.contours):
                    for contour in slice_contours:
                        if len(contour) > 0:
                            max_x = max(max_x, np.max(contour[:, 0, 0]))
                            max_y = max(max_y, np.max(contour[:, 0, 1]))
                            max_z = max(max_z, z)

                image_shape = (max_z + 1, max_y + 1, max_x + 1)

            # Tạo mask mới
            mask = np.zeros(image_shape, dtype=bool)

            # Điền contours vào mask
            for z, slice_contours in enumerate(self.contours):
                if z < mask.shape[0]:
                    slice_mask = np.zeros(
                        (mask.shape[1], mask.shape[2]), dtype=np.uint8
                    )

                    # Vẽ các contours
                    for contour in slice_contours:
                        cv2.drawContours(slice_mask, [contour], 0, 1, -1)

                    # Gán vào mask 3D
                    mask[z] = slice_mask > 0

            self.mask = mask

        except ImportError:
            logger.warning("Không thể import OpenCV để tạo mask từ contours")
        except Exception as e:
            logger.error(f"Lỗi khi tạo mask từ contours: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi cấu trúc thành từ điển.

        Returns:
            Từ điển biểu diễn cấu trúc
        """
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.name,
            "color": self.color,
            "volume": self.volume,
            "metadata": self.metadata,
            # Không lưu mask và contours vì kích thước lớn
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Structure":
        """
        Tạo cấu trúc từ từ điển.

        Args:
            data: Từ điển biểu diễn cấu trúc

        Returns:
            Đối tượng Structure
        """
        return cls(
            structure_id=data.get("id"),
            name=data.get("name", "Unknown Structure"),
            structure_type=data.get("type", StructureType.UNKNOWN),
            color=data.get("color"),
            mask=None,  # Mask và contours cần được nạp riêng
            contours=None,
        )

    def __str__(self) -> str:
        """Biểu diễn chuỗi của cấu trúc."""
        return (
            f"Structure({self.name}, type={self.type.name}, volume={self.volume:.2f}cc)"
        )


class StructureSet:
    """
    Tập hợp các cấu trúc giải phẫu.

    Lớp này quản lý tập hợp các cấu trúc giải phẫu liên quan đến một bệnh nhân.
    """

    def __init__(
        self,
        structures: List[Structure] = None,
        structure_set_id: Optional[str] = None,
        name: str = "Structure Set",
    ):
        """
        Khởi tạo tập cấu trúc.

        Args:
            structures: Danh sách các cấu trúc
            structure_set_id: ID của tập cấu trúc
            name: Tên tập cấu trúc
        """
        self.structures = structures or []
        self.id = structure_set_id or str(uuid.uuid4())[:8]
        self.name = name
        self.metadata = {}

    def add_structure(self, structure: Structure) -> None:
        """
        Thêm một cấu trúc vào tập.

        Args:
            structure: Cấu trúc cần thêm
        """
        if not isinstance(structure, Structure):
            logger.error(
                f"Không thể thêm đối tượng không phải Structure: {type(structure)}"
            )
            return

        # Kiểm tra ID trùng lặp
        for existing in self.structures:
            if existing.id == structure.id:
                logger.warning(
                    f"Cấu trúc với ID {structure.id} đã tồn tại trong tập, sẽ bị ghi đè"
                )
                self.structures.remove(existing)
                break

        self.structures.append(structure)

    def get_structure_by_id(self, structure_id: str) -> Optional[Structure]:
        """
        Lấy cấu trúc theo ID.

        Args:
            structure_id: ID của cấu trúc cần lấy

        Returns:
            Cấu trúc, hoặc None nếu không tìm thấy
        """
        for structure in self.structures:
            if structure.id == structure_id:
                return structure

        return None

    def get_structure_by_name(
        self, name: str, case_sensitive: bool = False
    ) -> Optional[Structure]:
        """
        Lấy cấu trúc theo tên.

        Args:
            name: Tên cấu trúc cần lấy
            case_sensitive: Có phân biệt chữ hoa/thường không

        Returns:
            Cấu trúc, hoặc None nếu không tìm thấy
        """
        for structure in self.structures:
            if case_sensitive:
                if structure.name == name:
                    return structure
            else:
                if structure.name.lower() == name.lower():
                    return structure

        return None

    def get_structures_by_type(self, structure_type: StructureType) -> List[Structure]:
        """
        Lấy danh sách các cấu trúc theo loại.

        Args:
            structure_type: Loại cấu trúc cần lấy

        Returns:
            Danh sách các cấu trúc thuộc loại đã chọn
        """
        return [s for s in self.structures if s.type == structure_type]

    def remove_structure(self, structure_id: str) -> bool:
        """
        Xóa một cấu trúc khỏi tập.

        Args:
            structure_id: ID của cấu trúc cần xóa

        Returns:
            True nếu xóa thành công, False nếu không
        """
        for i, structure in enumerate(self.structures):
            if structure.id == structure_id:
                self.structures.pop(i)
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi tập cấu trúc thành từ điển.

        Returns:
            Từ điển biểu diễn tập cấu trúc
        """
        return {
            "id": self.id,
            "name": self.name,
            "structures": [s.to_dict() for s in self.structures],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructureSet":
        """
        Tạo tập cấu trúc từ từ điển.

        Args:
            data: Từ điển biểu diễn tập cấu trúc

        Returns:
            Đối tượng StructureSet
        """
        structures = []
        for s_data in data.get("structures", []):
            structures.append(Structure.from_dict(s_data))

        structure_set = cls(
            structures=structures,
            structure_set_id=data.get("id"),
            name=data.get("name", "Structure Set"),
        )

        structure_set.metadata = data.get("metadata", {})

        return structure_set

    def __str__(self) -> str:
        """Biểu diễn chuỗi của tập cấu trúc."""
        return f"StructureSet({self.name}, {len(self.structures)} structures)"
