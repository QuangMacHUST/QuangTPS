#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module công cụ vẽ (drawing tools) cho QuangTPS.

Module này cung cấp các lớp và hàm để vẽ và chỉnh sửa contour trong TPS.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class DrawingToolType(str, Enum):
    """Các loại công cụ vẽ có sẵn."""

    BRUSH = "BRUSH"  # Bút vẽ (brush)
    PENCIL = "PENCIL"  # Bút chì (pencil)
    ERASER = "ERASER"  # Tẩy (eraser)
    POLYGON = "POLYGON"  # Vẽ đa giác (polygon)
    RECTANGLE = "RECTANGLE"  # Vẽ hình chữ nhật (rectangle)
    ELLIPSE = "ELLIPSE"  # Vẽ hình elip (ellipse)
    SMART_BRUSH = "SMART_BRUSH"  # Bút vẽ thông minh (smart brush) - sử dụng thresholds
    INTERPOLATE = "INTERPOLATE"  # Nội suy giữa các contour (interpolate)
    FREEHAND = "FREEHAND"  # Vẽ tự do (freehand)
    THRESHOLD = "THRESHOLD"  # Vẽ theo ngưỡng (threshold)
    FILL = "FILL"  # Tô (fill)


class DrawingTool:
    """
    Lớp cơ sở cho các công cụ vẽ.

    Đây là lớp trừu tượng cung cấp giao diện chung cho tất cả các công cụ vẽ.
    """

    def __init__(self, tool_type: DrawingToolType):
        """
        Khởi tạo công cụ vẽ.

        Parameters:
            tool_type (DrawingToolType): Loại công cụ vẽ
        """
        self.tool_type = tool_type
        self.color = (255, 0, 0)  # Màu mặc định là đỏ
        self.size = 5  # Kích thước mặc định
        self.opacity = 0.5  # Độ trong suốt mặc định
        self.is_active = False

    def activate(self):
        """Kích hoạt công cụ."""
        self.is_active = True
        logger.debug(f"Activated {self.tool_type} tool")

    def deactivate(self):
        """Hủy kích hoạt công cụ."""
        self.is_active = False
        logger.debug(f"Deactivated {self.tool_type} tool")

    def set_color(self, color: Tuple[int, int, int]):
        """Đặt màu cho công cụ."""
        self.color = color

    def set_size(self, size: int):
        """Đặt kích thước cho công cụ."""
        self.size = size

    def set_opacity(self, opacity: float):
        """Đặt độ trong suốt cho công cụ."""
        self.opacity = max(0.0, min(1.0, opacity))  # Giới hạn trong khoảng 0-1

    def apply(
        self, image: np.ndarray, position: Tuple[int, int], *args, **kwargs
    ) -> np.ndarray:
        """
        Áp dụng công cụ vào hình ảnh tại vị trí chỉ định.

        Parameters:
            image (np.ndarray): Hình ảnh đầu vào
            position (Tuple[int, int]): Vị trí (x, y) áp dụng công cụ

        Returns:
            np.ndarray: Hình ảnh sau khi đã áp dụng công cụ
        """
        # Lớp cơ sở không thực hiện gì, các lớp con sẽ ghi đè
        logger.warning(f"apply() not implemented for {self.tool_type}")
        return image.copy()


class BrushTool(DrawingTool):
    """Công cụ bút vẽ (brush) cho phép vẽ với bút tròn."""

    def __init__(self, size: int = 5):
        """
        Khởi tạo công cụ bút vẽ.

        Parameters:
            size (int, optional): Kích thước bút. Mặc định là 5.
        """
        super().__init__(DrawingToolType.BRUSH)
        self.size = size

    def apply(
        self, image: np.ndarray, position: Tuple[int, int], value: int = 1
    ) -> np.ndarray:
        """
        Vẽ một vòng tròn tại vị trí chỉ định.

        Parameters:
            image (np.ndarray): Hình ảnh đầu vào
            position (Tuple[int, int]): Vị trí (x, y) vẽ
            value (int, optional): Giá trị để đặt pixels. Mặc định là 1.

        Returns:
            np.ndarray: Hình ảnh sau khi vẽ
        """
        import cv2

        # Tạo bản sao để không thay đổi đầu vào
        result = image.copy()

        # Đảm bảo vị trí trong giới hạn hình ảnh
        x, y = position
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            return result

        # Vẽ một vòng tròn đặc
        cv2.circle(result, (x, y), self.size, value, -1)

        return result


class EraserTool(BrushTool):
    """Công cụ tẩy, hoạt động như bút vẽ nhưng đặt giá trị 0."""

    def __init__(self, size: int = 5):
        """
        Khởi tạo công cụ tẩy.

        Parameters:
            size (int, optional): Kích thước tẩy. Mặc định là 5.
        """
        super().__init__(size)
        self.tool_type = DrawingToolType.ERASER

    def apply(
        self, image: np.ndarray, position: Tuple[int, int], *args, **kwargs
    ) -> np.ndarray:
        """
        Xóa (đặt giá trị 0) tại vị trí chỉ định.

        Parameters:
            image (np.ndarray): Hình ảnh đầu vào
            position (Tuple[int, int]): Vị trí (x, y) xóa

        Returns:
            np.ndarray: Hình ảnh sau khi xóa
        """
        # Gọi apply của BrushTool với giá trị 0
        return super().apply(image, position, 0)


class PolygonTool(DrawingTool):
    """Công cụ vẽ đa giác."""

    def __init__(self):
        """Khởi tạo công cụ vẽ đa giác."""
        super().__init__(DrawingToolType.POLYGON)
        self.points = []  # Danh sách các điểm của đa giác

    def add_point(self, point: Tuple[int, int]):
        """
        Thêm một điểm vào đa giác.

        Parameters:
            point (Tuple[int, int]): Điểm (x, y) cần thêm
        """
        self.points.append(point)

    def reset(self):
        """Reset đa giác, xóa tất cả các điểm."""
        self.points = []

    def apply(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Vẽ và tô đa giác lên hình ảnh.

        Parameters:
            image (np.ndarray): Hình ảnh đầu vào

        Returns:
            np.ndarray: Hình ảnh sau khi vẽ đa giác
        """
        import cv2

        result = image.copy()

        if len(self.points) < 3:
            # Cần ít nhất 3 điểm để tạo đa giác
            return result

        # Chuyển đổi danh sách điểm thành mảng numpy
        pts = np.array(self.points, dtype=np.int32)

        # Vẽ và tô đa giác
        cv2.fillPoly(result, [pts], 1)

        return result


class RectangleTool(DrawingTool):
    """Công cụ vẽ hình chữ nhật."""

    def __init__(self):
        """Khởi tạo công cụ vẽ hình chữ nhật."""
        super().__init__(DrawingToolType.RECTANGLE)
        self.start_point = None
        self.end_point = None

    def set_start_point(self, point: Tuple[int, int]):
        """
        Đặt điểm bắt đầu của hình chữ nhật.

        Parameters:
            point (Tuple[int, int]): Điểm (x, y) bắt đầu
        """
        self.start_point = point

    def set_end_point(self, point: Tuple[int, int]):
        """
        Đặt điểm kết thúc của hình chữ nhật.

        Parameters:
            point (Tuple[int, int]): Điểm (x, y) kết thúc
        """
        self.end_point = point

    def reset(self):
        """Reset hình chữ nhật, xóa điểm bắt đầu và kết thúc."""
        self.start_point = None
        self.end_point = None

    def apply(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Vẽ và tô hình chữ nhật lên hình ảnh.

        Parameters:
            image (np.ndarray): Hình ảnh đầu vào

        Returns:
            np.ndarray: Hình ảnh sau khi vẽ hình chữ nhật
        """
        import cv2

        result = image.copy()

        if self.start_point is None or self.end_point is None:
            return result

        # Vẽ và tô hình chữ nhật
        cv2.rectangle(result, self.start_point, self.end_point, 1, -1)

        return result


class EllipseTool(DrawingTool):
    """Công cụ vẽ hình elip."""

    def __init__(self):
        """Khởi tạo công cụ vẽ hình elip."""
        super().__init__(DrawingToolType.ELLIPSE)
        self.center = None
        self.axes = None
        self.angle = 0  # Góc xoay

    def set_center(self, center: Tuple[int, int]):
        """
        Đặt tâm của hình elip.

        Parameters:
            center (Tuple[int, int]): Tọa độ (x, y) tâm
        """
        self.center = center

    def set_axes(self, axes: Tuple[int, int]):
        """
        Đặt bán kính của hình elip.

        Parameters:
            axes (Tuple[int, int]): Bán kính (rx, ry)
        """
        self.axes = axes

    def set_angle(self, angle: float):
        """
        Đặt góc xoay của hình elip.

        Parameters:
            angle (float): Góc xoay (độ)
        """
        self.angle = angle

    def reset(self):
        """Reset hình elip, xóa tâm và bán kính."""
        self.center = None
        self.axes = None
        self.angle = 0

    def apply(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Vẽ và tô hình elip lên hình ảnh.

        Parameters:
            image (np.ndarray): Hình ảnh đầu vào

        Returns:
            np.ndarray: Hình ảnh sau khi vẽ hình elip
        """
        import cv2

        result = image.copy()

        if self.center is None or self.axes is None:
            return result

        # Vẽ và tô hình elip
        cv2.ellipse(result, self.center, self.axes, self.angle, 0, 360, 1, -1)

        return result


class ThresholdTool(DrawingTool):
    """Công cụ vẽ theo ngưỡng."""

    def __init__(self):
        """Khởi tạo công cụ vẽ theo ngưỡng."""
        super().__init__(DrawingToolType.THRESHOLD)
        self.min_threshold = 0  # Ngưỡng dưới
        self.max_threshold = 255  # Ngưỡng trên

    def set_thresholds(self, min_val: int, max_val: int):
        """
        Đặt ngưỡng dưới và ngưỡng trên.

        Parameters:
            min_val (int): Ngưỡng dưới
            max_val (int): Ngưỡng trên
        """
        self.min_threshold = min_val
        self.max_threshold = max_val

    def apply(
        self, image: np.ndarray, ct_image: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """
        Tạo mask dựa trên ngưỡng từ dữ liệu CT.

        Parameters:
            image (np.ndarray): Hình ảnh đầu vào (mask hiện tại)
            ct_image (np.ndarray): Hình ảnh CT nguồn

        Returns:
            np.ndarray: Mask sau khi áp dụng ngưỡng
        """
        import cv2

        result = image.copy()

        # Tạo mask từ CT image dựa trên ngưỡng
        _, threshold_mask = cv2.threshold(
            ct_image,
            self.min_threshold,
            1,  # Giá trị max đặt = 1 để tạo mask nhị phân
            cv2.THRESH_BINARY
            if self.min_threshold <= self.max_threshold
            else cv2.THRESH_BINARY_INV,
        )

        # Áp dụng mask vào kết quả
        result = cv2.bitwise_or(result, threshold_mask)

        return result


class SmartBrushTool(BrushTool):
    """Công cụ bút vẽ thông minh sử dụng ngưỡng cục bộ."""

    def __init__(self, size: int = 5, tolerance: int = 10):
        """
        Khởi tạo công cụ bút vẽ thông minh.

        Parameters:
            size (int, optional): Kích thước bút. Mặc định là 5.
            tolerance (int, optional): Độ dung sai ngưỡng. Mặc định là 10.
        """
        super().__init__(size)
        self.tool_type = DrawingToolType.SMART_BRUSH
        self.tolerance = tolerance

    def set_tolerance(self, tolerance: int):
        """
        Đặt độ dung sai cho ngưỡng.

        Parameters:
            tolerance (int): Độ dung sai
        """
        self.tolerance = tolerance

    def apply(
        self,
        image: np.ndarray,
        position: Tuple[int, int],
        ct_image: np.ndarray,
        value: int = 1,
    ) -> np.ndarray:
        """
        Vẽ thông minh dựa trên ngưỡng cục bộ.

        Parameters:
            image (np.ndarray): Hình ảnh đầu vào (mask)
            position (Tuple[int, int]): Vị trí (x, y) vẽ
            ct_image (np.ndarray): Hình ảnh CT nguồn
            value (int, optional): Giá trị để đặt pixels. Mặc định là 1.

        Returns:
            np.ndarray: Hình ảnh sau khi vẽ
        """
        import cv2

        # Tạo bản sao để không thay đổi đầu vào
        result = image.copy()

        # Đảm bảo vị trí trong giới hạn hình ảnh
        x, y = position
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            return result

        # Lấy giá trị cường độ tại vị trí click
        seed_value = ct_image[y, x]

        # Tạo mask khu vực vẽ
        mask = np.zeros_like(image)
        cv2.circle(mask, (x, y), self.size, 1, -1)

        # Tạo mask ngưỡng trong khu vực vẽ
        lower_bound = max(0, seed_value - self.tolerance)
        upper_bound = min(255, seed_value + self.tolerance)

        # Áp dụng ngưỡng trong khu vực vẽ
        threshold_mask = np.logical_and(
            ct_image >= lower_bound, ct_image <= upper_bound
        ).astype(np.uint8)

        # Kết hợp mask khu vực vẽ và mask ngưỡng
        combined_mask = np.logical_and(mask, threshold_mask).astype(np.uint8)

        # Áp dụng mask kết hợp vào kết quả
        result[combined_mask > 0] = value

        return result


class InterpolateTool(DrawingTool):
    """Công cụ nội suy giữa các contour."""

    def __init__(self):
        """Khởi tạo công cụ nội suy."""
        super().__init__(DrawingToolType.INTERPOLATE)

    def interpolate(
        self, contours: Dict[int, np.ndarray], start_slice: int, end_slice: int
    ) -> Dict[int, np.ndarray]:
        """
        Nội suy contours giữa hai slice.

        Parameters:
            contours (Dict[int, np.ndarray]): Dict với khóa là chỉ số slice, giá trị là mask
            start_slice (int): Chỉ số slice bắt đầu
            end_slice (int): Chỉ số slice kết thúc

        Returns:
            Dict[int, np.ndarray]: Dict chứa các contours đã nội suy
        """
        if start_slice not in contours or end_slice not in contours:
            logger.error("Start slice or end slice not in contours")
            return {}

        if abs(end_slice - start_slice) <= 1:
            # Không cần nội suy nếu các slice liền kề
            return {}

        start_mask = contours[start_slice]
        end_mask = contours[end_slice]

        # Đảm bảo hai mask có cùng kích thước
        if start_mask.shape != end_mask.shape:
            logger.error("Start and end masks have different shapes")
            return {}

        result = {}
        num_slices = abs(end_slice - start_slice) - 1

        for i in range(1, num_slices + 1):
            # Tính toán hệ số nội suy
            alpha = i / (num_slices + 1)

            # Tạo mask nội suy
            interpolated_mask = self._linear_interpolation(start_mask, end_mask, alpha)

            # Thêm vào kết quả
            slice_idx = start_slice + i if start_slice < end_slice else start_slice - i
            result[slice_idx] = interpolated_mask

        return result

    def _linear_interpolation(
        self, mask1: np.ndarray, mask2: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        Nội suy tuyến tính giữa hai mask.

        Parameters:
            mask1 (np.ndarray): Mask đầu tiên
            mask2 (np.ndarray): Mask thứ hai
            alpha (float): Hệ số nội suy (0-1)

        Returns:
            np.ndarray: Mask đã nội suy
        """
        import cv2

        # Nội suy
        interpolated = (1 - alpha) * mask1.astype(float) + alpha * mask2.astype(float)

        # Chuyển về mask nhị phân
        _, result = cv2.threshold(interpolated, 0.5, 1, cv2.THRESH_BINARY)

        return result.astype(np.uint8)


class FreehandTool(DrawingTool):
    """Công cụ vẽ tự do."""

    def __init__(self):
        """Khởi tạo công cụ vẽ tự do."""
        super().__init__(DrawingToolType.FREEHAND)
        self.points = []  # Danh sách các điểm

    def add_point(self, point: Tuple[int, int]):
        """
        Thêm một điểm vào đường vẽ.

        Parameters:
            point (Tuple[int, int]): Điểm (x, y) cần thêm
        """
        self.points.append(point)

    def reset(self):
        """Reset đường vẽ, xóa tất cả các điểm."""
        self.points = []

    def apply(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Vẽ và tô contour tự do lên hình ảnh.

        Parameters:
            image (np.ndarray): Hình ảnh đầu vào

        Returns:
            np.ndarray: Hình ảnh sau khi vẽ contour
        """
        import cv2

        result = image.copy()

        if len(self.points) < 2:
            # Cần ít nhất 2 điểm để vẽ
            return result

        # Vẽ đường nối các điểm
        for i in range(len(self.points) - 1):
            cv2.line(result, self.points[i], self.points[i + 1], 1, thickness=1)

        # Đóng contour nếu đủ điểm
        if len(self.points) >= 3:
            # Tạo đa giác và tô
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(result, [pts], 1)

        return result
