#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module margin cho QuangTPS.

Module này chứa các lớp và hàm để tạo margin cho contour cấu trúc.
Hỗ trợ các loại margin: đồng đều, không đồng đều, vòng và bề mặt.
"""

import enum
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union

# Kiểm tra OpenCV
try:
    import cv2

    CV2_AVAILABLE = True

    # Định nghĩa các wrapper functions để tránh lỗi linter
    def cv_fill_poly(img, contours, color):
        """Wrapper cho cv2.fillPoly để tránh lỗi linter."""
        return cv2.fillPoly(img, contours, color)

    def cv_get_structuring_element(shape, ksize):
        """Wrapper cho cv2.getStructuringElement."""
        return cv2.getStructuringElement(shape, ksize)

    def cv_dilate(img, kernel, iterations=1):
        """Wrapper cho cv2.dilate."""
        return cv2.dilate(img, kernel, iterations)

    def cv_erode(img, kernel, iterations=1):
        """Wrapper cho cv2.erode."""
        return cv2.erode(img, kernel, iterations)

    def cv_find_contours(img, mode, method):
        """Wrapper cho cv2.findContours."""
        return cv2.findContours(img, mode, method)

    def cv_bitwise_or(img1, img2):
        """Wrapper cho cv2.bitwise_or."""
        return cv2.bitwise_or(img1, img2)

    def cv_subtract(img1, img2):
        """Wrapper cho cv2.subtract."""
        return cv2.subtract(img1, img2)

    # Định nghĩa các constants
    MORPH_ELLIPSE = cv2.MORPH_ELLIPSE if hasattr(cv2, "MORPH_ELLIPSE") else 2
    MORPH_RECT = cv2.MORPH_RECT if hasattr(cv2, "MORPH_RECT") else 0
    RETR_EXTERNAL = cv2.RETR_EXTERNAL if hasattr(cv2, "RETR_EXTERNAL") else 0
    CHAIN_APPROX_SIMPLE = (
        cv2.CHAIN_APPROX_SIMPLE if hasattr(cv2, "CHAIN_APPROX_SIMPLE") else 1
    )

except ImportError:
    CV2_AVAILABLE = False
    logging.warning(
        "OpenCV (cv2) không khả dụng. Margin sẽ được tính toán bằng phương pháp thay thế chậm hơn."
    )

    # Tạo các dummy functions khi không có OpenCV
    def cv_fill_poly(img, contours, color):
        """Dummy function khi không có OpenCV."""
        return img

    def cv_get_structuring_element(shape, ksize):
        """Dummy function khi không có OpenCV."""
        return np.ones(ksize)

    def cv_dilate(img, kernel, iterations=1):
        """Dummy function khi không có OpenCV."""
        return img

    def cv_erode(img, kernel, iterations=1):
        """Dummy function khi không có OpenCV."""
        return img

    def cv_find_contours(img, mode, method):
        """Dummy function khi không có OpenCV."""
        return [], None

    def cv_bitwise_or(img1, img2):
        """Dummy function khi không có OpenCV."""
        return np.maximum(img1, img2)

    def cv_subtract(img1, img2):
        """Dummy function khi không có OpenCV."""
        return np.maximum(img1 - img2, 0)

    # Define constants
    MORPH_ELLIPSE = 2
    MORPH_RECT = 0
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 1

logger = logging.getLogger(__name__)


class MarginType(enum.Enum):
    """Enum cho các loại margin."""

    UNIFORM = "UNIFORM"
    ANISOTROPIC = "ANISOTROPIC"
    RING = "RING"
    SURFACE = "SURFACE"


class MarginTool:
    """
    Công cụ để tạo margin cho contour.
    Hỗ trợ nhiều loại margin khác nhau và tối ưu hóa bằng OpenCV khi có thể.
    """

    def __init__(self):
        """Khởi tạo công cụ margin."""
        self.use_opencv = CV2_AVAILABLE

    def margin_by_type(
        self,
        contours: List[np.ndarray],
        margin_type: MarginType,
        margin_params: Dict[str, Any],
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> List[np.ndarray]:
        """
        Tạo margin cho contour dựa trên loại margin và tham số.

        Parameters:
            contours: Danh sách contour đầu vào (mỗi contour là np.ndarray có shape (N, 2))
            margin_type: Loại margin (UNIFORM, ANISOTROPIC, RING, SURFACE)
            margin_params: Tham số margin
            pixel_spacing: Khoảng cách pixel (dx, dy) trong mm

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        if not contours:
            return []

        # Gọi phương thức tương ứng dựa trên loại margin
        if margin_type == MarginType.UNIFORM:
            return self.uniform_margin(
                contours, margin_params.get("margin_mm", 0.0), pixel_spacing
            )

        elif margin_type == MarginType.ANISOTROPIC:
            margins = margin_params.get("margins_mm", {})
            return self.anisotropic_margin(
                contours,
                margins.get("ANTERIOR", 0.0),
                margins.get("POSTERIOR", 0.0),
                margins.get("LEFT", 0.0),
                margins.get("RIGHT", 0.0),
                pixel_spacing,
            )

        elif margin_type == MarginType.RING:
            return self.ring_margin(
                contours,
                margin_params.get("inner_margin_mm", 0.0),
                margin_params.get("outer_margin_mm", 0.0),
                pixel_spacing,
            )

        elif margin_type == MarginType.SURFACE:
            return self.surface_margin(
                contours, margin_params.get("thickness_mm", 3.0), pixel_spacing
            )

        else:
            raise ValueError(f"Loại margin không được hỗ trợ: {margin_type}")

    def uniform_margin(
        self,
        contours: List[np.ndarray],
        margin_mm: float,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> List[np.ndarray]:
        """
        Tạo margin đồng đều cho contour.

        Parameters:
            contours: Danh sách contour đầu vào
            margin_mm: Margin tính bằng mm (có thể âm)
            pixel_spacing: Khoảng cách pixel (dx, dy) trong mm

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        if not contours:
            return []

        # Kiểm tra nếu margin gần bằng 0, không cần xử lý
        if abs(margin_mm) < 0.01:
            return contours.copy()

        # Tính toán margin theo pixel
        margin_x = margin_mm / pixel_spacing[0]
        margin_y = margin_mm / pixel_spacing[1]

        # Sử dụng OpenCV nếu có thể
        if self.use_opencv:
            return self._opencv_uniform_margin(contours, margin_x, margin_y)
        else:
            return self._numpy_uniform_margin(contours, margin_x, margin_y)

    def _opencv_uniform_margin(
        self, contours: List[np.ndarray], margin_x: float, margin_y: float
    ) -> List[np.ndarray]:
        """
        Tạo margin đồng đều sử dụng OpenCV.

        Parameters:
            contours: Danh sách contour đầu vào
            margin_x: Margin theo trục x tính bằng pixel
            margin_y: Margin theo trục y tính bằng pixel

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        try:
            # Tìm kích thước hình ảnh cần thiết để chứa contour
            all_points = np.vstack([c for c in contours if c.size > 0])
            min_x, min_y = np.min(all_points, axis=0)
            max_x, max_y = np.max(all_points, axis=0)

            # Thêm padding để đảm bảo contour không bị cắt khi mở rộng
            padding = max(abs(margin_x), abs(margin_y)) * 2 + 100
            width = int(max_x - min_x + 2 * padding)
            height = int(max_y - min_y + 2 * padding)

            # Tạo mask từ contour
            mask = np.zeros((height, width), dtype=np.uint8)
            shifted_contours = [
                c - [min_x - padding, min_y - padding] for c in contours
            ]

            # Vẽ contour lên mask
            cv_fill_poly(mask, [c.astype(np.int32) for c in shifted_contours], 255)

            # Áp dụng phép co/giãn
            kernel_size = int(max(abs(margin_x), abs(margin_y)) * 2) + 1
            kernel = cv_get_structuring_element(
                MORPH_ELLIPSE, (kernel_size, kernel_size)
            )

            if margin_x > 0:  # Margin dương -> giãn
                result_mask = cv_dilate(mask, kernel, iterations=1)
            else:  # Margin âm -> co
                result_mask = cv_erode(mask, kernel, iterations=1)

            # Trích xuất contour từ mask
            new_contours, _ = cv_find_contours(
                result_mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
            )

            # Chuyển về định dạng numpy array và áp dụng lại offset
            result = []
            for contour in new_contours:
                # Chỉ lấy contour có kích thước > threshold
                if contour.shape[0] > 3:  # Ít nhất 4 điểm để tạo thành contour hợp lệ
                    # Chuyển từ [[[x,y]], [[x,y]], ...] sang [[x,y], [x,y], ...]
                    points = contour.reshape(-1, 2).astype(np.float64)
                    # Áp dụng lại offset
                    points += [min_x - padding, min_y - padding]
                    result.append(points)

            return result

        except Exception as e:
            logger.error(f"Lỗi khi sử dụng OpenCV cho margin đồng đều: {e}")
            # Fallback sang phương pháp numpy
            return self._numpy_uniform_margin(contours, margin_x, margin_y)

    def _numpy_uniform_margin(
        self, contours: List[np.ndarray], margin_x: float, margin_y: float
    ) -> List[np.ndarray]:
        """
        Tạo margin đồng đều sử dụng NumPy (phương pháp thay thế).

        Parameters:
            contours: Danh sách contour đầu vào
            margin_x: Margin theo trục x tính bằng pixel
            margin_y: Margin theo trục y tính bằng pixel

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        result_contours = []

        for contour in contours:
            if contour.size == 0:
                continue

            # Tính tâm contour
            center = np.mean(contour, axis=0)

            # Tính hướng từ tâm đến mỗi điểm
            directions = contour - center

            # Chuẩn hóa hướng
            norms = np.sqrt(np.sum(directions**2, axis=1)).reshape(-1, 1)
            norms[norms == 0] = 1.0  # Tránh chia cho 0
            unit_directions = directions / norms

            # Tính toán offset margin
            margin_vector = unit_directions * np.array([margin_x, margin_y])

            # Áp dụng margin
            new_contour = contour + margin_vector

            # Thêm vào kết quả
            result_contours.append(new_contour)

        return result_contours

    def anisotropic_margin(
        self,
        contours: List[np.ndarray],
        anterior_mm: float,
        posterior_mm: float,
        left_mm: float,
        right_mm: float,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> List[np.ndarray]:
        """
        Tạo margin không đồng đều cho contour.

        Parameters:
            contours: Danh sách contour đầu vào
            anterior_mm: Margin phía trước (mm)
            posterior_mm: Margin phía sau (mm)
            left_mm: Margin bên trái (mm)
            right_mm: Margin bên phải (mm)
            pixel_spacing: Khoảng cách pixel (dx, dy) trong mm

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        if not contours:
            return []

        # Nếu tất cả các margin bằng nhau -> sử dụng uniform margin
        if (
            abs(anterior_mm - posterior_mm) < 0.01
            and abs(anterior_mm - left_mm) < 0.01
            and abs(anterior_mm - right_mm) < 0.01
        ):
            return self.uniform_margin(contours, anterior_mm, pixel_spacing)

        # Chuyển đổi mm sang pixel
        anterior_px = anterior_mm / pixel_spacing[1]
        posterior_px = posterior_mm / pixel_spacing[1]
        left_px = left_mm / pixel_spacing[0]
        right_px = right_mm / pixel_spacing[0]

        # Sử dụng OpenCV nếu có thể
        if self.use_opencv:
            return self._opencv_anisotropic_margin(
                contours, anterior_px, posterior_px, left_px, right_px
            )
        else:
            return self._numpy_anisotropic_margin(
                contours, anterior_px, posterior_px, left_px, right_px
            )

    def _opencv_anisotropic_margin(
        self,
        contours: List[np.ndarray],
        anterior_px: float,
        posterior_px: float,
        left_px: float,
        right_px: float,
    ) -> List[np.ndarray]:
        """
        Tạo margin không đồng đều sử dụng OpenCV.

        Parameters:
            contours: Danh sách contour đầu vào
            anterior_px: Margin phía trước (pixel)
            posterior_px: Margin phía sau (pixel)
            left_px: Margin bên trái (pixel)
            right_px: Margin bên phải (pixel)

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        try:
            # Phương pháp làm việc với margin không đồng đều:
            # 1. Áp dụng margin đồng đều nhỏ nhất
            # 2. Áp dụng các margin bổ sung theo hướng

            min_margin = min(anterior_px, posterior_px, left_px, right_px)
            max_margin = max(anterior_px, posterior_px, left_px, right_px)

            # Áp dụng margin đồng đều nhỏ nhất
            result_contours = self._opencv_uniform_margin(
                contours, min_margin, min_margin
            )

            # Nếu min_margin và max_margin gần bằng nhau, không cần xử lý thêm
            if abs(max_margin - min_margin) < 1.0:
                return result_contours

            # Tìm kích thước hình ảnh cần thiết để chứa contour
            all_points = np.vstack([c for c in result_contours if c.size > 0])
            min_x, min_y = np.min(all_points, axis=0)
            max_x, max_y = np.max(all_points, axis=0)

            # Thêm padding
            padding = max(abs(max_margin), abs(min_margin)) * 2 + 100
            width = int(max_x - min_x + 2 * padding)
            height = int(max_y - min_y + 2 * padding)

            # Tạo mask từ contour
            mask = np.zeros((height, width), dtype=np.uint8)
            shifted_contours = [
                c - [min_x - padding, min_y - padding] for c in result_contours
            ]

            # Vẽ contour lên mask
            cv_fill_poly(mask, [c.astype(np.int32) for c in shifted_contours], 255)

            # Áp dụng margin bổ sung theo hướng
            # Tính phần bổ sung cho mỗi hướng
            anterior_diff = anterior_px - min_margin
            posterior_diff = posterior_px - min_margin
            left_diff = left_px - min_margin
            right_diff = right_px - min_margin

            # Mặt nạ cho mỗi phần
            result_mask = mask.copy()

            # 1. Phần phía trước (tương ứng phần dưới của mask)
            if anterior_diff > 0:
                kernel_size = int(anterior_diff * 2) + 1
                kernel = cv_get_structuring_element(
                    MORPH_RECT, (kernel_size, kernel_size)
                )
                bottom_mask = np.zeros_like(mask)
                bottom_mask[height // 2 :] = mask[height // 2 :]  # Lấy phần dưới
                dilated_bottom = cv_dilate(bottom_mask, kernel, iterations=1)
                result_mask = cv_bitwise_or(result_mask, dilated_bottom)

            # 2. Phần phía sau (tương ứng phần trên của mask)
            if posterior_diff > 0:
                kernel_size = int(posterior_diff * 2) + 1
                kernel = cv_get_structuring_element(
                    MORPH_RECT, (kernel_size, kernel_size)
                )
                top_mask = np.zeros_like(mask)
                top_mask[: height // 2] = mask[: height // 2]  # Lấy phần trên
                dilated_top = cv_dilate(top_mask, kernel, iterations=1)
                result_mask = cv_bitwise_or(result_mask, dilated_top)

            # 3. Phần bên trái
            if left_diff > 0:
                kernel_size = int(left_diff * 2) + 1
                kernel = cv_get_structuring_element(
                    MORPH_RECT, (kernel_size, kernel_size)
                )
                left_mask = np.zeros_like(mask)
                left_mask[:, : width // 2] = mask[:, : width // 2]  # Lấy phần trái
                dilated_left = cv_dilate(left_mask, kernel, iterations=1)
                result_mask = cv_bitwise_or(result_mask, dilated_left)

            # 4. Phần bên phải
            if right_diff > 0:
                kernel_size = int(right_diff * 2) + 1
                kernel = cv_get_structuring_element(
                    MORPH_RECT, (kernel_size, kernel_size)
                )
                right_mask = np.zeros_like(mask)
                right_mask[:, width // 2 :] = mask[:, width // 2 :]  # Lấy phần phải
                dilated_right = cv_dilate(right_mask, kernel, iterations=1)
                result_mask = cv_bitwise_or(result_mask, dilated_right)

            # Trích xuất contour từ mask
            new_contours, _ = cv_find_contours(
                result_mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
            )

            # Chuyển về định dạng numpy array và áp dụng lại offset
            result = []
            for contour in new_contours:
                # Chỉ lấy contour có kích thước > threshold
                if contour.shape[0] > 3:  # Ít nhất 4 điểm để tạo thành contour hợp lệ
                    # Chuyển từ [[[x,y]], [[x,y]], ...] sang [[x,y], [x,y], ...]
                    points = contour.reshape(-1, 2).astype(np.float64)
                    # Áp dụng lại offset
                    points += [min_x - padding, min_y - padding]
                    result.append(points)

            return result

        except Exception as e:
            logger.error(f"Lỗi khi sử dụng OpenCV cho margin không đồng đều: {e}")
            # Fallback sang phương pháp numpy
            return self._numpy_anisotropic_margin(
                contours, anterior_px, posterior_px, left_px, right_px
            )

    def _numpy_anisotropic_margin(
        self,
        contours: List[np.ndarray],
        anterior_px: float,
        posterior_px: float,
        left_px: float,
        right_px: float,
    ) -> List[np.ndarray]:
        """
        Tạo margin không đồng đều sử dụng NumPy (phương pháp thay thế).

        Parameters:
            contours: Danh sách contour đầu vào
            anterior_px: Margin phía trước (pixel)
            posterior_px: Margin phía sau (pixel)
            left_px: Margin bên trái (pixel)
            right_px: Margin bên phải (pixel)

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        result_contours = []

        for contour in contours:
            if contour.size == 0:
                continue

            # Tính tâm contour
            center = np.mean(contour, axis=0)

            # Tính hướng từ tâm đến mỗi điểm
            directions = contour - center

            # Chuẩn hóa hướng
            norms = np.sqrt(np.sum(directions**2, axis=1)).reshape(-1, 1)
            norms[norms == 0] = 1.0  # Tránh chia cho 0
            unit_directions = directions / norms

            # Xác định hướng của mỗi điểm
            # Hướng x: âm là trái, dương là phải
            # Hướng y: âm là trên (posterior), dương là dưới (anterior)
            x_dir = unit_directions[:, 0]
            y_dir = unit_directions[:, 1]

            # Tính margin cho mỗi điểm dựa trên hướng
            x_margin = np.where(x_dir < 0, left_px * np.abs(x_dir), right_px * x_dir)
            y_margin = np.where(
                y_dir < 0, posterior_px * np.abs(y_dir), anterior_px * y_dir
            )

            # Tính margin vector
            margin_vector = np.column_stack((x_margin, y_margin))

            # Áp dụng margin
            new_contour = contour + margin_vector

            # Thêm vào kết quả
            result_contours.append(new_contour)

        return result_contours

    def ring_margin(
        self,
        contours: List[np.ndarray],
        inner_margin_mm: float,
        outer_margin_mm: float,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> List[np.ndarray]:
        """
        Tạo margin vòng (ring) cho contour.

        Parameters:
            contours: Danh sách contour đầu vào
            inner_margin_mm: Margin bên trong (mm)
            outer_margin_mm: Margin bên ngoài (mm)
            pixel_spacing: Khoảng cách pixel (dx, dy) trong mm

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        if not contours:
            return []

        # Kiểm tra nếu inner_margin và outer_margin đều bằng 0
        if abs(inner_margin_mm) < 0.01 and abs(outer_margin_mm) < 0.01:
            return contours.copy()

        # Tính margin theo pixel
        inner_x = inner_margin_mm / pixel_spacing[0]
        inner_y = inner_margin_mm / pixel_spacing[1]
        outer_x = outer_margin_mm / pixel_spacing[0]
        outer_y = outer_margin_mm / pixel_spacing[1]

        # Sử dụng OpenCV nếu có thể
        if self.use_opencv:
            return self._opencv_ring_margin(
                contours, inner_x, inner_y, outer_x, outer_y
            )
        else:
            return self._numpy_ring_margin(contours, inner_x, inner_y, outer_x, outer_y)

    def _opencv_ring_margin(
        self,
        contours: List[np.ndarray],
        inner_x: float,
        inner_y: float,
        outer_x: float,
        outer_y: float,
    ) -> List[np.ndarray]:
        """
        Tạo margin vòng (ring) sử dụng OpenCV.

        Parameters:
            contours: Danh sách contour đầu vào
            inner_x: Margin bên trong theo trục x (pixel)
            inner_y: Margin bên trong theo trục y (pixel)
            outer_x: Margin bên ngoài theo trục x (pixel)
            outer_y: Margin bên ngoài theo trục y (pixel)

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        try:
            # Tìm kích thước hình ảnh cần thiết để chứa contour
            all_points = np.vstack([c for c in contours if c.size > 0])
            min_x, min_y = np.min(all_points, axis=0)
            max_x, max_y = np.max(all_points, axis=0)

            # Thêm padding
            max_margin = max(abs(inner_x), abs(inner_y), abs(outer_x), abs(outer_y))
            padding = max_margin * 2 + 100
            width = int(max_x - min_x + 2 * padding)
            height = int(max_y - min_y + 2 * padding)

            # Tạo mask từ contour
            mask = np.zeros((height, width), dtype=np.uint8)
            shifted_contours = [
                c - [min_x - padding, min_y - padding] for c in contours
            ]

            # Vẽ contour lên mask
            cv_fill_poly(mask, [c.astype(np.int32) for c in shifted_contours], 255)

            # Tạo outer mask bằng cách giãn mask
            outer_mask = mask.copy()
            if outer_x > 0 and outer_y > 0:
                kernel_size = int(max(outer_x, outer_y) * 2) + 1
                kernel = cv_get_structuring_element(
                    MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
                outer_mask = cv_dilate(mask, kernel, iterations=1)

            # Tạo inner mask bằng cách co mask
            inner_mask = mask.copy()
            if inner_x > 0 and inner_y > 0:
                kernel_size = int(max(inner_x, inner_y) * 2) + 1
                kernel = cv_get_structuring_element(
                    MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
                inner_mask = cv_erode(mask, kernel, iterations=1)

            # Tạo ring mask bằng cách trừ inner mask từ outer mask
            ring_mask = cv_subtract(outer_mask, inner_mask)

            # Trích xuất contour từ ring mask
            new_contours, _ = cv_find_contours(
                ring_mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
            )

            # Chuyển về định dạng numpy array và áp dụng lại offset
            result = []
            for contour in new_contours:
                # Chỉ lấy contour có kích thước > threshold
                if contour.shape[0] > 3:  # Ít nhất 4 điểm để tạo thành contour hợp lệ
                    # Chuyển từ [[[x,y]], [[x,y]], ...] sang [[x,y], [x,y], ...]
                    points = contour.reshape(-1, 2).astype(np.float64)
                    # Áp dụng lại offset
                    points += [min_x - padding, min_y - padding]
                    result.append(points)

            # Thêm cả các contour bên trong nếu có
            inner_contours, _ = cv_find_contours(
                inner_mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
            )
            for contour in inner_contours:
                if contour.shape[0] > 3:
                    points = contour.reshape(-1, 2).astype(np.float64)
                    points += [min_x - padding, min_y - padding]
                    result.append(points)

            return result

        except Exception as e:
            logger.error(f"Lỗi khi sử dụng OpenCV cho margin vòng: {e}")
            # Fallback sang phương pháp numpy
            return self._numpy_ring_margin(contours, inner_x, inner_y, outer_x, outer_y)

    def _numpy_ring_margin(
        self,
        contours: List[np.ndarray],
        inner_x: float,
        inner_y: float,
        outer_x: float,
        outer_y: float,
    ) -> List[np.ndarray]:
        """
        Tạo margin vòng (ring) sử dụng NumPy (phương pháp thay thế).

        Parameters:
            contours: Danh sách contour đầu vào
            inner_x: Margin bên trong theo trục x (pixel)
            inner_y: Margin bên trong theo trục y (pixel)
            outer_x: Margin bên ngoài theo trục x (pixel)
            outer_y: Margin bên ngoài theo trục y (pixel)

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        # Tạo margin bên ngoài
        outer_contours = self._numpy_uniform_margin(contours, outer_x, outer_y)

        # Nếu inner_margin bằng 0, trả về outer_contours
        if abs(inner_x) < 0.01 and abs(inner_y) < 0.01:
            return outer_contours

        # Tạo margin bên trong (contour co lại)
        inner_contours = self._numpy_uniform_margin(contours, -inner_x, -inner_y)

        # Kết hợp hai danh sách
        result_contours = outer_contours + inner_contours

        return result_contours

    def surface_margin(
        self,
        contours: List[np.ndarray],
        thickness_mm: float,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> List[np.ndarray]:
        """
        Tạo margin bề mặt (surface) cho contour.

        Parameters:
            contours: Danh sách contour đầu vào
            thickness_mm: Độ dày bề mặt (mm)
            pixel_spacing: Khoảng cách pixel (dx, dy) trong mm

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        # Tạo surface margin chính là tạo ring margin với inner_margin = thickness và outer_margin = 0
        return self.ring_margin(contours, thickness_mm, 0.0, pixel_spacing)


if __name__ == "__main__":
    # Kiểm tra mã
    import matplotlib.pyplot as plt

    # Tạo contour hình chữ nhật đơn giản
    rect = np.array([[100, 100], [300, 100], [300, 200], [100, 200], [100, 100]])

    # Tạo contour hình tròn
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([150 * np.cos(t) + 400, 100 * np.sin(t) + 150]).T

    # Tạo contour ban đầu
    contours = [rect, circle]

    # Tạo một margin tool
    margin_tool = MarginTool()

    # Áp dụng các loại margin
    margin_mm = 10
    pixel_spacing = (1.0, 1.0)

    # 1. Margin đồng đều
    uniform_result = margin_tool.margin_by_type(
        contours, MarginType.UNIFORM, {"margin_mm": margin_mm}, pixel_spacing
    )

    # 2. Margin không đồng đều
    anisotropic_result = margin_tool.margin_by_type(
        contours,
        MarginType.ANISOTROPIC,
        {"margins_mm": {"ANTERIOR": 15, "POSTERIOR": 5, "LEFT": 10, "RIGHT": 5}},
        pixel_spacing,
    )

    # 3. Margin vòng
    ring_result = margin_tool.margin_by_type(
        contours,
        MarginType.RING,
        {"inner_margin_mm": 5, "outer_margin_mm": 10},
        pixel_spacing,
    )

    # 4. Margin bề mặt
    surface_result = margin_tool.margin_by_type(
        contours, MarginType.SURFACE, {"thickness_mm": 5}, pixel_spacing
    )

    # Vẽ kết quả
    plt.figure(figsize=(12, 10))

    # Ban đầu
    plt.subplot(221)
    for c in contours:
        plt.plot(c[:, 0], c[:, 1], "b-")
    plt.title("Ban đầu")
    plt.axis("equal")

    # Margin đồng đều
    plt.subplot(222)
    for c in contours:
        plt.plot(c[:, 0], c[:, 1], "b-")
    for c in uniform_result:
        plt.plot(c[:, 0], c[:, 1], "r-")
    plt.title("Margin đồng đều")
    plt.axis("equal")

    # Margin vòng
    plt.subplot(223)
    for c in contours:
        plt.plot(c[:, 0], c[:, 1], "b-")
    for c in ring_result:
        plt.plot(c[:, 0], c[:, 1], "g-")
    plt.title("Margin vòng")
    plt.axis("equal")

    # Margin bề mặt
    plt.subplot(224)
    for c in contours:
        plt.plot(c[:, 0], c[:, 1], "b-")
    for c in surface_result:
        plt.plot(c[:, 0], c[:, 1], "m-")
    plt.title("Margin bề mặt")
    plt.axis("equal")

    plt.tight_layout()
    plt.show()
