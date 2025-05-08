#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chuyển đổi mask thành cấu trúc.

Module này cung cấp các hàm cần thiết để chuyển đổi mask phân đoạn (dạng mảng numpy)
thành đối tượng Structure (cấu trúc).
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import uuid

logger = logging.getLogger(__name__)


def mask_to_structure(
    mask: np.ndarray,
    structure_name: str,
    structure_id: Optional[str] = None,
    image_metadata: Optional[Dict] = None,
    color: Optional[Tuple[float, float, float]] = None,
):
    """
    Chuyển đổi mask phân đoạn thành đối tượng Structure.

    Args:
        mask: Mảng numpy 3D binary biểu diễn mask phân đoạn
        structure_name: Tên của cấu trúc
        structure_id: ID của cấu trúc (tự động tạo nếu không cung cấp)
        image_metadata: Thông tin về hệ tọa độ và kích thước các voxel của ảnh
        color: Màu sắc của cấu trúc dưới dạng tuple (r, g, b) với giá trị 0-1

    Returns:
        Đối tượng Structure
    """
    try:
        # Import động để tránh import lặp
        from quangtps.core.structure import Structure

        # Tạo ID nếu không cung cấp
        if structure_id is None:
            structure_id = str(uuid.uuid4())[:8]

        # Tạo màu ngẫu nhiên nếu không cung cấp
        if color is None:
            color = (np.random.random(), np.random.random(), np.random.random())

        # Tạo đối tượng Structure
        structure = Structure(
            structure_id=structure_id, name=structure_name, color=color
        )

        # Thiết lập mask và chuyển đổi thành contour
        structure.set_mask(mask)

        # Thiết lập metadata nếu có
        if image_metadata:
            structure.set_metadata(image_metadata)

        # Tính thể tích
        if hasattr(structure, "calculate_volume"):
            structure.calculate_volume()

        return structure

    except ImportError as e:
        logger.error(f"Lỗi import khi tạo Structure: {e}")

        # Tạo đối tượng giả
        class DummyStructure:
            def __init__(self, structure_id, name, color):
                self.id = structure_id
                self.name = name
                self.color = color
                self.mask = mask
                self.metadata = image_metadata
                self.volume = np.sum(mask) if mask is not None else 0

        return DummyStructure(
            structure_id or str(uuid.uuid4())[:8], structure_name, color
        )

    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi mask thành Structure: {e}")
        return None


def structure_to_mask(
    structure, image_shape: Tuple[int, int, int] = None
) -> Optional[np.ndarray]:
    """
    Chuyển đổi đối tượng Structure thành mask phân đoạn.

    Args:
        structure: Đối tượng Structure
        image_shape: Kích thước mảng 3D đầu ra (tùy chọn)

    Returns:
        Mảng numpy 3D binary biểu diễn mask phân đoạn
    """
    try:
        # Kiểm tra nếu cấu trúc đã có mask
        if hasattr(structure, "mask") and structure.mask is not None:
            if image_shape is None or structure.mask.shape == image_shape:
                return structure.mask
            else:
                # Resize mask nếu cần
                return _resize_mask(structure.mask, image_shape)

        # Trường hợp cấu trúc chỉ có dữ liệu contour
        if hasattr(structure, "to_mask"):
            return structure.to_mask(image_shape)

        logger.error("Cấu trúc không có mask hoặc phương thức để tạo mask")
        return None

    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi Structure thành mask: {e}")
        return None


def _resize_mask(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Thay đổi kích thước mask phân đoạn.

    Args:
        mask: Mảng numpy 3D binary
        target_shape: Kích thước đích

    Returns:
        Mảng numpy 3D binary đã thay đổi kích thước
    """
    try:
        from scipy.ndimage import zoom

        # Tính các tỷ lệ zoom
        factors = (
            target_shape[0] / mask.shape[0],
            target_shape[1] / mask.shape[1],
            target_shape[2] / mask.shape[2],
        )

        # Thực hiện zoom
        resized = zoom(mask, factors, order=0)  # order=0 để giữ là mask binary

        # Đảm bảo vẫn là binary
        return (resized > 0.5).astype(np.uint8)

    except ImportError:
        logger.warning("Không thể import scipy.ndimage để thực hiện resize")

        # Fallback nếu không có scipy

        result = np.zeros(target_shape, dtype=np.uint8)

        # Copy dữ liệu trong phạm vi có thể
        min_dims = [min(mask.shape[i], target_shape[i]) for i in range(3)]
        for i in range(min_dims[0]):
            for j in range(min_dims[1]):
                for k in range(min_dims[2]):
                    result[i, j, k] = mask[i, j, k]

        return result
