#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module này chứa các công cụ để tạo margin cho contour trong quá trình phân đoạn.
"""

import enum
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union

# Kiểm tra xem OpenCV có khả dụng không
try:
    import cv2

    CV2_AVAILABLE = True

    # Định nghĩa các hàm wrapper cho OpenCV
    def cv_fill_poly(img, contours, color):
        """Wrapper cho cv2.fillPoly."""
        contours_int = [np.round(c).astype(np.int32) for c in contours]
        return cv2.fillPoly(img, contours_int, color)

    def cv_get_structuring_element(shape, ksize):
        """Wrapper cho cv2.getStructuringElement."""
        return cv2.getStructuringElement(shape, ksize)

    def cv_dilate(img, kernel, iterations=1):
        """Wrapper cho cv2.dilate."""
        return cv2.dilate(img, kernel, iterations=iterations)

    def cv_erode(img, kernel, iterations=1):
        """Wrapper cho cv2.erode."""
        return cv2.erode(img, kernel, iterations=iterations)

    def cv_find_contours(img, mode, method):
        """Wrapper cho cv2.findContours."""
        contours, hierarchy = cv2.findContours(img, mode, method)
        return contours, hierarchy

    def cv_bitwise_or(img1, img2):
        """Wrapper cho cv2.bitwise_or."""
        return cv2.bitwise_or(img1, img2)

    def cv_subtract(img1, img2):
        """Wrapper cho cv2.subtract."""
        return cv2.subtract(img1, img2)

    # Define constants
    MORPH_ELLIPSE = cv2.MORPH_ELLIPSE
    MORPH_RECT = cv2.MORPH_RECT
    RETR_EXTERNAL = cv2.RETR_EXTERNAL
    CHAIN_APPROX_SIMPLE = cv2.CHAIN_APPROX_SIMPLE

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
        Tạo margin đồng đều (đều theo mọi hướng) cho contour.

        Parameters:
            contours: Danh sách contour đầu vào
            margin_mm: Kích thước margin (mm), dương là phóng đại, âm là thu nhỏ
            pixel_spacing: Khoảng cách pixel (dx, dy) trong mm

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        # Chuyển đổi margin từ mm sang pixel
        margin_x = margin_mm / pixel_spacing[0]
        margin_y = margin_mm / pixel_spacing[1]

        # Sử dụng OpenCV nếu có thể
        if self.use_opencv:
            return self._opencv_uniform_margin(contours, margin_x, margin_y)
        else:
            return self._numpy_uniform_margin(contours, margin_x, margin_y)

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
        Tạo margin không đồng đều (khác nhau theo các hướng) cho contour.

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
        # Chuyển đổi margin từ mm sang pixel
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
        # Chuyển đổi margin từ mm sang pixel
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

    def _opencv_uniform_margin(
        self, contours: List[np.ndarray], margin_x: float, margin_y: float
    ) -> List[np.ndarray]:
        """
        Tạo margin đồng đều sử dụng OpenCV.

        Parameters:
            contours: Danh sách contour đầu vào
            margin_x: Margin theo trục x (pixel)
            margin_y: Margin theo trục y (pixel)

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        if not contours:
            return []

        # Tìm kích thước bounding box của tất cả contours
        all_points = np.vstack([c for c in contours if c.size > 0])
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)

        # Tạo một ảnh binary cho contour
        width = int(x_max - x_min + 2 * abs(margin_x) + 10)
        height = int(y_max - y_min + 2 * abs(margin_y) + 10)
        offset_x = int(abs(margin_x) + 5 - x_min)
        offset_y = int(abs(margin_y) + 5 - y_min)

        # Tạo mặt nạ và vẽ contour
        mask = np.zeros((height, width), dtype=np.uint8)
        shifted_contours = [c + np.array([offset_x, offset_y]) for c in contours]
        cv_fill_poly(mask, shifted_contours, 255)

        # Thực hiện morphological operation
        if margin_x > 0 and margin_y > 0:
            # Dãn ra (dilate)
            kernel_size = (int(2 * margin_x) + 1, int(2 * margin_y) + 1)
            kernel = cv_get_structuring_element(MORPH_ELLIPSE, kernel_size)
            dilated = cv_dilate(mask, kernel)
            mask = dilated
        elif margin_x < 0 and margin_y < 0:
            # Thu vào (erode)
            kernel_size = (int(-2 * margin_x) + 1, int(-2 * margin_y) + 1)
            kernel = cv_get_structuring_element(MORPH_ELLIPSE, kernel_size)
            eroded = cv_erode(mask, kernel)
            mask = eroded

        # Tìm contour từ mặt nạ
        contours, _ = cv_find_contours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        # Chuyển từ định dạng OpenCV sang numpy array và dịch lại
        result = []
        for c in contours:
            # Làm mịn contour
            if len(c) > 2:
                c = c.reshape(-1, 2).astype(np.float32)
                c -= np.array([offset_x, offset_y])
                result.append(c)

        return result

    def _numpy_uniform_margin(
        self, contours: List[np.ndarray], margin_x: float, margin_y: float
    ) -> List[np.ndarray]:
        """
        Tạo margin đồng đều sử dụng NumPy (phương pháp thay thế).

        Parameters:
            contours: Danh sách contour đầu vào
            margin_x: Margin theo trục x (pixel)
            margin_y: Margin theo trục y (pixel)

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        if not contours:
            return []

        result = []
        for contour in contours:
            if len(contour) < 3:
                continue

            # Tính vector pháp tuyến cho mỗi điểm
            n = len(contour)
            normals = np.zeros_like(contour)

            for i in range(n):
                prev = (i - 1) % n
                next = (i + 1) % n

                # Vector hướng từ điểm trước đến điểm sau
                v1 = contour[i] - contour[prev]
                v2 = contour[next] - contour[i]

                # Chuẩn hóa
                v1 = v1 / np.sqrt(np.sum(v1**2)) if np.sum(v1**2) > 0 else v1
                v2 = v2 / np.sqrt(np.sum(v2**2)) if np.sum(v2**2) > 0 else v2

                # Tính vector pháp tuyến (quay 90 độ theo chiều kim đồng hồ)
                n1 = np.array([v1[1], -v1[0]])
                n2 = np.array([v2[1], -v2[0]])

                # Trung bình hai vector pháp tuyến
                normal = (n1 + n2) / 2
                if np.sum(normal**2) > 0:
                    normal = normal / np.sqrt(np.sum(normal**2))

                normals[i] = normal

            # Áp dụng margin
            margin_vector = np.column_stack(
                [normals[:, 0] * margin_x, normals[:, 1] * margin_y]
            )
            new_contour = contour + margin_vector
            result.append(new_contour)

        return result

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
            anterior_px: Margin phía trước theo pixel
            posterior_px: Margin phía sau theo pixel
            left_px: Margin bên trái theo pixel
            right_px: Margin bên phải theo pixel

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        if not contours:
            return []

        # Tìm kích thước bounding box của tất cả contours
        all_points = np.vstack([c for c in contours if c.size > 0])
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)

        # Tính offset và kích thước ảnh
        offsets = [
            max(right_px, 0),
            max(anterior_px, 0),
            max(-left_px, 0),
            max(-posterior_px, 0),
        ]
        max_offset = max(offsets) + 10

        # Tạo một ảnh binary cho contour
        width = int(x_max - x_min + max_offset * 2)
        height = int(y_max - y_min + max_offset * 2)
        offset_x = int(max_offset - x_min)
        offset_y = int(max_offset - y_min)

        # Tạo mặt nạ và vẽ contour
        mask = np.zeros((height, width), dtype=np.uint8)
        shifted_contours = [c + np.array([offset_x, offset_y]) for c in contours]
        cv_fill_poly(mask, shifted_contours, 255)

        # Tạo 4 mặt nạ riêng biệt cho 4 hướng
        result_mask = np.zeros_like(mask)

        # Xử lý phía trên (anterior)
        if anterior_px != 0:
            anterior_kernel = cv_get_structuring_element(
                MORPH_RECT, (1, int(abs(anterior_px) * 2) + 1)
            )
            if anterior_px > 0:
                anterior_mask = cv_dilate(mask, anterior_kernel)
                anterior_mask = cv_subtract(anterior_mask, mask)
                anterior_mask[: height // 2, :] = 0  # Chỉ giữ phần phía trên
            else:
                anterior_mask = cv_erode(mask, anterior_kernel)
                anterior_mask = cv_subtract(mask, anterior_mask)
                anterior_mask[: height // 2, :] = 0  # Chỉ giữ phần phía trên
            result_mask = cv_bitwise_or(result_mask, anterior_mask)

        # Xử lý phía dưới (posterior)
        if posterior_px != 0:
            posterior_kernel = cv_get_structuring_element(
                MORPH_RECT, (1, int(abs(posterior_px) * 2) + 1)
            )
            if posterior_px > 0:
                posterior_mask = cv_dilate(mask, posterior_kernel)
                posterior_mask = cv_subtract(posterior_mask, mask)
                posterior_mask[height // 2 :, :] = 0  # Chỉ giữ phần phía dưới
            else:
                posterior_mask = cv_erode(mask, posterior_kernel)
                posterior_mask = cv_subtract(mask, posterior_mask)
                posterior_mask[height // 2 :, :] = 0  # Chỉ giữ phần phía dưới
            result_mask = cv_bitwise_or(result_mask, posterior_mask)

        # Xử lý bên trái
        if left_px != 0:
            left_kernel = cv_get_structuring_element(
                MORPH_RECT, (int(abs(left_px) * 2) + 1, 1)
            )
            if left_px > 0:
                left_mask = cv_dilate(mask, left_kernel)
                left_mask = cv_subtract(left_mask, mask)
                left_mask[:, width // 2 :] = 0  # Chỉ giữ phần bên trái
            else:
                left_mask = cv_erode(mask, left_kernel)
                left_mask = cv_subtract(mask, left_mask)
                left_mask[:, width // 2 :] = 0  # Chỉ giữ phần bên trái
            result_mask = cv_bitwise_or(result_mask, left_mask)

        # Xử lý bên phải
        if right_px != 0:
            right_kernel = cv_get_structuring_element(
                MORPH_RECT, (int(abs(right_px) * 2) + 1, 1)
            )
            if right_px > 0:
                right_mask = cv_dilate(mask, right_kernel)
                right_mask = cv_subtract(right_mask, mask)
                right_mask[:, : width // 2] = 0  # Chỉ giữ phần bên phải
            else:
                right_mask = cv_erode(mask, right_kernel)
                right_mask = cv_subtract(mask, right_mask)
                right_mask[:, : width // 2] = 0  # Chỉ giữ phần bên phải
            result_mask = cv_bitwise_or(result_mask, right_mask)

        # Kết hợp với mask gốc
        result_mask = cv_bitwise_or(result_mask, mask)

        # Tìm contour từ mặt nạ kết quả
        contours, _ = cv_find_contours(result_mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        # Chuyển từ định dạng OpenCV sang numpy array và dịch lại
        result = []
        for c in contours:
            if len(c) > 2:
                c = c.reshape(-1, 2).astype(np.float32)
                c -= np.array([offset_x, offset_y])
                result.append(c)

        return result

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
            anterior_px: Margin phía trước theo pixel
            posterior_px: Margin phía sau theo pixel
            left_px: Margin bên trái theo pixel
            right_px: Margin bên phải theo pixel

        Returns:
            Danh sách contour sau khi áp dụng margin
        """
        if not contours:
            return []

        result = []
        for contour in contours:
            if len(contour) < 3:
                continue

            # Tính vector pháp tuyến cho mỗi điểm
            n = len(contour)
            new_contour = np.zeros_like(contour)

            for i in range(n):
                prev = (i - 1) % n
                next = (i + 1) % n

                # Vector hướng từ điểm trước đến điểm sau
                v1 = contour[i] - contour[prev]
                v2 = contour[next] - contour[i]

                # Chuẩn hóa
                v1 = v1 / np.sqrt(np.sum(v1**2)) if np.sum(v1**2) > 0 else v1
                v2 = v2 / np.sqrt(np.sum(v2**2)) if np.sum(v2**2) > 0 else v2

                # Tính vector pháp tuyến (quay 90 độ theo chiều kim đồng hồ)
                n1 = np.array([v1[1], -v1[0]])
                n2 = np.array([v2[1], -v2[0]])

                # Trung bình hai vector pháp tuyến
                normal = (n1 + n2) / 2
                if np.sum(normal**2) > 0:
                    normal = normal / np.sqrt(np.sum(normal**2))

                # Xác định hướng và áp dụng margin tương ứng
                shift_x = 0
                shift_y = 0

                # Phía trên (y giảm) - anterior
                if normal[1] < 0:
                    shift_y = -normal[1] * anterior_px
                # Phía dưới (y tăng) - posterior
                else:
                    shift_y = normal[1] * posterior_px

                # Bên trái (x giảm) - left
                if normal[0] < 0:
                    shift_x = -normal[0] * left_px
                # Bên phải (x tăng) - right
                else:
                    shift_x = normal[0] * right_px

                # Cập nhật tọa độ mới
                new_contour[i] = contour[i] + np.array([shift_x, shift_y])

            result.append(new_contour)

        return result

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
        if not contours:
            return []

        # Tìm kích thước bounding box của tất cả contours
        all_points = np.vstack([c for c in contours if c.size > 0])
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)

        # Tạo một ảnh binary cho contour
        width = int(x_max - x_min + 2 * max(abs(outer_x), abs(inner_x)) + 10)
        height = int(y_max - y_min + 2 * max(abs(outer_y), abs(inner_y)) + 10)
        offset_x = int(max(abs(outer_x), abs(inner_x)) + 5 - x_min)
        offset_y = int(max(abs(outer_y), abs(inner_y)) + 5 - y_min)

        # Tạo mặt nạ và vẽ contour
        mask = np.zeros((height, width), dtype=np.uint8)
        shifted_contours = [c + np.array([offset_x, offset_y]) for c in contours]
        cv_fill_poly(mask, shifted_contours, 255)

        # Tạo mask cho outer margin
        outer_mask = np.zeros_like(mask)
        if outer_x > 0 and outer_y > 0:
            # Dãn ra (dilate)
            kernel_size = (int(2 * outer_x) + 1, int(2 * outer_y) + 1)
            kernel = cv_get_structuring_element(MORPH_ELLIPSE, kernel_size)
            outer_mask = cv_dilate(mask, kernel)

        # Tạo mask cho inner margin
        inner_mask = np.zeros_like(mask)
        if inner_x > 0 and inner_y > 0:
            # Thu vào (erode)
            kernel_size = (int(2 * inner_x) + 1, int(2 * inner_y) + 1)
            kernel = cv_get_structuring_element(MORPH_ELLIPSE, kernel_size)
            inner_mask = cv_erode(mask, kernel)
            inner_mask = cv_subtract(mask, inner_mask)

        # Kết hợp outer và inner
        result_mask = cv_bitwise_or(outer_mask, inner_mask)

        # Tìm contour từ mặt nạ kết quả
        contours, _ = cv_find_contours(result_mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        # Chuyển từ định dạng OpenCV sang numpy array và dịch lại
        result = []
        for c in contours:
            if len(c) > 2:
                c = c.reshape(-1, 2).astype(np.float32)
                c -= np.array([offset_x, offset_y])
                result.append(c)

        return result

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
