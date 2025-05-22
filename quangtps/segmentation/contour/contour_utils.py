#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contour Utilities for QuangTPS

Module cung cấp các tiện ích và hàm phụ trợ cho việc xử lý contour trong hệ thống QuangTPS.
Bao gồm các công cụ chuyển đổi, tối ưu hóa và phân tích contour.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# Kiểm tra xem OpenCV có sẵn không để cung cấp phương thức thay thế nếu cần
try:
    import cv2

    # Kiểm tra xem cv2 có các phương thức cần thiết không
    _test_contour = np.array([[[0, 0]], [[1, 1]], [[2, 2]]], dtype=np.float32)
    cv2.approxPolyDP(_test_contour, 0.1, True)  # Kiểm tra approxPolyDP
    HAS_OPENCV = True
    # Kiểm tra các hằng số cần thiết
    if not hasattr(cv2, "RETR_EXTERNAL"):
        cv2.RETR_EXTERNAL = 0  # Giá trị mặc định
    if not hasattr(cv2, "CHAIN_APPROX_SIMPLE"):
        cv2.CHAIN_APPROX_SIMPLE = 1  # Giá trị mặc định
except (ImportError, AttributeError):
    HAS_OPENCV = False
    logger.warning(
        "OpenCV không được cài đặt hoặc thiếu các phương thức cần thiết. Sẽ sử dụng phương thức thay thế."
    )

# Kiểm tra xem Shapely có sẵn không
try:
    from shapely.geometry import Polygon

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    logger.warning("Shapely không được cài đặt. Sẽ sử dụng phương thức thay thế.")

# Kiểm tra xem scipy có sẵn không
try:
    from scipy import ndimage

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy không được cài đặt. Sẽ sử dụng phương thức thay thế.")


class ContourUtils:
    """
    Lớp tiện ích để xử lý contour trong QuangTPS.
    Cung cấp các phương thức tĩnh để thực hiện các hoạt động chung trên contour.
    """

    @staticmethod
    def simplify_contour(contour: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """
        Đơn giản hóa contour bằng thuật toán Douglas-Peucker.

        Parameters
        ----------
        contour : np.ndarray
            Contour đầu vào (Nx2 hoặc Nx3)
        epsilon : float, optional
            Thông số kiểm soát mức độ đơn giản hóa, by default 0.1

        Returns
        -------
        np.ndarray
            Contour đã được đơn giản hóa
        """
        if HAS_OPENCV:
            try:
                # Đảm bảo contour có định dạng đúng cho cv2
                contour_for_cv = contour.copy()
                if contour.shape[1] > 2:  # Nếu là 3D, chỉ sử dụng 2 tọa độ đầu tiên
                    contour_for_cv = contour[:, :2]

                contour_for_cv = contour_for_cv.reshape(-1, 1, 2).astype(np.float32)
                simplified = cv2.approxPolyDP(contour_for_cv, epsilon, True)

                # Chuyển kết quả trở lại định dạng ban đầu
                simplified = simplified.reshape(-1, 2)

                # Nếu contour gốc là 3D, thêm lại tọa độ z
                if contour.shape[1] > 2:
                    # Nội suy tọa độ z cho các điểm mới
                    z_values = np.interp(
                        np.arange(len(simplified)),
                        np.linspace(0, len(simplified) - 1, len(contour)),
                        contour[:, 2],
                    )
                    simplified = np.column_stack((simplified, z_values))

                return simplified
            except Exception as e:
                logger.error(f"Lỗi khi đơn giản hóa contour với OpenCV: {str(e)}")
                return ContourUtils._simplify_contour_alternative(contour, epsilon)
        else:
            logger.info("Sử dụng thuật toán đơn giản hóa thay thế")
            return ContourUtils._simplify_contour_alternative(contour, epsilon)

    @staticmethod
    def _simplify_contour_alternative(
        contour: np.ndarray, epsilon: float = 0.1
    ) -> np.ndarray:
        """
        Triển khai thay thế cho thuật toán đơn giản hóa khi không có OpenCV.
        Thuật toán đơn giản hóa sử dụng khoảng cách.

        Parameters
        ----------
        contour : np.ndarray
            Contour đầu vào
        epsilon : float, optional
            Ngưỡng khoảng cách, by default 0.1

        Returns
        -------
        np.ndarray
            Contour đã được đơn giản hóa
        """
        if len(contour) < 3:
            return contour

        # Khởi tạo với điểm đầu tiên
        result = [contour[0]]

        # Lặp qua tất cả các điểm để chọn những điểm cần giữ lại
        for i in range(1, len(contour) - 1):
            # Tính khoảng cách từ điểm i đến đường thẳng nối điểm kết quả cuối cùng và điểm tiếp theo
            prev_point = result[-1]
            next_point = contour[i + 1]

            # Tính khoảng cách
            if contour.shape[1] >= 3:  # 3D
                dist = np.linalg.norm(
                    np.cross(next_point - prev_point, prev_point - contour[i])
                ) / np.linalg.norm(next_point - prev_point)
            else:  # 2D
                # Công thức khoảng cách từ điểm đến đường thẳng trong không gian 2D
                x1, y1 = prev_point[:2]
                x2, y2 = next_point[:2]
                x0, y0 = contour[i][:2]

                dist = abs(
                    (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1
                ) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            # Nếu khoảng cách lớn hơn epsilon, giữ lại điểm
            if dist > epsilon:
                result.append(contour[i])

        # Thêm điểm cuối cùng
        if len(contour) > 1:
            result.append(contour[-1])

        return np.array(result)

    @staticmethod
    def smooth_contour(contour: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Làm mịn contour sử dụng bộ lọc Gaussian.

        Parameters
        ----------
        contour : np.ndarray
            Contour đầu vào
        sigma : float, optional
            Độ lệch chuẩn cho bộ lọc Gaussian, by default 1.0

        Returns
        -------
        np.ndarray
            Contour đã được làm mịn
        """
        if len(contour) < 3:
            return contour

        # Tạo kernel Gaussian 1D
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Đảm bảo kernel_size là số lẻ

        kernel_size = max(3, kernel_size)  # Tối thiểu là 3

        # Tạo kernel Gaussian
        x = np.linspace(-3 * sigma, 3 * sigma, kernel_size)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / np.sum(kernel)  # Chuẩn hóa

        # Sao chép contour để xử lý
        smoothed = np.zeros_like(contour, dtype=float)

        # Xử lý riêng cho từng tọa độ
        for dim in range(contour.shape[1]):
            # Lấy dữ liệu chiều này
            data = contour[:, dim].astype(float)

            # Thêm các điểm wrap-around để xử lý contour đóng
            padded_data = np.concatenate(
                [data[-kernel_size // 2 :], data, data[: kernel_size // 2]]
            )

            # Áp dụng bộ lọc
            convolved = np.convolve(padded_data, kernel, mode="valid")

            # Lưu kết quả
            smoothed[:, dim] = convolved

        return smoothed

    @staticmethod
    def convert_mask_to_contours(
        mask: np.ndarray,
        slice_thickness: float = 1.0,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> List[np.ndarray]:
        """
        Chuyển đổi mask 3D thành danh sách các contour 2D.

        Parameters
        ----------
        mask : np.ndarray
            Mask 3D với các giá trị 0/1
        slice_thickness : float, optional
            Độ dày lát cắt (mm), mặc định là 1.0
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel (mm) theo (row, col), mặc định là (1.0, 1.0)

        Returns
        -------
        List[np.ndarray]
            Danh sách các contour theo lát cắt
        """
        try:
            # Kiểm tra tính hợp lệ của mask
            if mask is None:
                logger.error("Mask không được là None")
                return []

            if not isinstance(mask, np.ndarray):
                logger.error(f"Mask phải là numpy array, nhưng có kiểu {type(mask)}")
                return []

            if mask.size == 0:
                logger.error("Mask trống")
                return []

            if mask.ndim < 2 or mask.ndim > 3:
                logger.error(f"Mask phải là 2D hoặc 3D, nhưng có chiều {mask.ndim}")
                return []

            # Thêm thông tin chi tiết về mask để debug
            logger.debug(
                f"Mask shape: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}"
            )

            # Kiểm tra xem mask có chứa giá trị khác 0 không
            if not np.any(mask):
                logger.warning(
                    "Mask không chứa vùng cần phân đoạn (toàn bộ giá trị là 0)"
                )
                return []

            # Đảm bảo mask chỉ chứa các giá trị 0 và 1
            # Chuyển đổi các giá trị khác 0 thành 1
            binary_mask = (mask > 0).astype(np.uint8)

            # Xử lý mask 2D và 3D
            if mask.ndim == 2:
                mask_3d = binary_mask.reshape(1, *binary_mask.shape)
            else:
                mask_3d = binary_mask

            contours_by_slice = []

            # Xử lý từng lát cắt
            for z in range(mask_3d.shape[0]):
                slice_mask = mask_3d[z]

                # Kiểm tra nếu lát cắt trống
                if not np.any(slice_mask):
                    contours_by_slice.append([])
                    continue

                try:
                    # Tìm contours với OpenCV
                    import cv2

                    # Đảm bảo slice_mask có định dạng uint8 cho findContours
                    if slice_mask.dtype != np.uint8:
                        slice_mask = slice_mask.astype(np.uint8)

                    # Sử dụng cv2.findContours với xử lý tương thích cho nhiều phiên bản OpenCV
                    try:
                        contours, _ = cv2.findContours(
                            slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                    except ValueError:
                        # Phiên bản OpenCV cũ trả về 3 giá trị
                        _, contours, _ = cv2.findContours(
                            slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                    # Chuyển đổi contours về định dạng cần thiết
                    slice_contours = []
                    for contour in contours:
                        # Bỏ qua contour quá nhỏ (có thể là nhiễu)
                        if len(contour) < 3:
                            continue

                        # Chuyển contour từ định dạng OpenCV sang numpy array
                        # và áp dụng pixel spacing
                        if contour.size > 0:
                            points = contour.squeeze()

                            # Xử lý trường hợp contour chỉ có 1 điểm
                            if len(points.shape) == 1:
                                continue

                            # Kiểm tra xem points có đủ số điểm không
                            if points.shape[0] < 3:
                                continue

                            # Áp dụng pixel spacing
                            points = points.astype(np.float32)
                            points[:, 0] = points[:, 0] * pixel_spacing[1]  # cols -> x
                            points[:, 1] = points[:, 1] * pixel_spacing[0]  # rows -> y

                            # Thêm tọa độ z
                            z_coord = z * slice_thickness
                            points_3d = np.column_stack(
                                [points, np.full(len(points), z_coord)]
                            )

                            slice_contours.append(points_3d)

                    contours_by_slice.append(slice_contours)

                except ImportError:
                    logger.warning(
                        "Không tìm thấy thư viện OpenCV, sử dụng phương pháp thay thế"
                    )
                    # Sử dụng phương pháp thay thế nếu không có OpenCV
                    alt_contours = ContourUtils._convert_mask_to_contours_alternative(
                        slice_mask,
                        slice_thickness,
                        pixel_spacing,
                        z_position=z * slice_thickness,
                    )
                    contours_by_slice.append(alt_contours)
                except Exception as e:
                    logger.error(f"Lỗi khi tìm contours cho lát cắt {z}: {str(e)}")
                    # Log traceback để debug
                    import traceback

                    logger.debug(f"Traceback: {traceback.format_exc()}")

                    # Sử dụng phương pháp thay thế trong trường hợp lỗi
                    alt_contours = ContourUtils._convert_mask_to_contours_alternative(
                        slice_mask,
                        slice_thickness,
                        pixel_spacing,
                        z_position=z * slice_thickness,
                    )
                    contours_by_slice.append(alt_contours)

            # Kiểm tra xem có contour nào được tìm thấy không
            if all(len(contours) == 0 for contours in contours_by_slice):
                logger.warning("Không tìm thấy contour nào trong mask")
                return []

            # Làm phẳng danh sách contours từ tất cả các lát cắt
            all_contours = []
            for slice_contours in contours_by_slice:
                if slice_contours:  # Nếu lát cắt có contours
                    all_contours.extend(slice_contours)

            return all_contours

        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi mask thành contours: {str(e)}")
            # Log traceback để debug
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")

            # Thử phương pháp thay thế trong trường hợp lỗi
            try:
                return ContourUtils._convert_mask_to_contours_alternative(
                    mask, slice_thickness, pixel_spacing
                )
            except Exception as e2:
                logger.error(f"Lỗi khi sử dụng phương pháp thay thế: {str(e2)}")
                return []

    @staticmethod
    def _convert_mask_to_contours_alternative(
        mask: np.ndarray,
        slice_thickness: float = 1.0,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        z_position: float = 0.0,
    ) -> List[np.ndarray]:
        """
        Phương pháp thay thế để chuyển đổi mask sang contours khi không có OpenCV.

        Parameters
        ----------
        mask : np.ndarray
            Mặt nạ nhị phân 3D (z, y, x) hoặc 2D (y, x)
        slice_thickness : float, optional
            Độ dày lát cắt theo mm, mặc định 1.0
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo mm trong mặt phẳng xy, mặc định (1.0, 1.0)
        z_position : float, optional
            Vị trí z của lát cắt, mặc định 0.0

        Returns
        -------
        List[np.ndarray]
            Danh sách các contour 3D
        """
        contours = []

        # Kiểm tra xem mask có hợp lệ không
        if mask is None or mask.size == 0:
            logger.warning(
                "Mask không hợp lệ trong _convert_mask_to_contours_alternative"
            )
            return contours

        # Làm việc với mask 2D hoặc lát cắt đơn từ mask 3D
        mask_2d = mask
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask_2d = mask[0]
            else:
                logger.warning(
                    "Phương pháp thay thế chỉ xử lý một lát cắt tại một thời điểm"
                )
                mask_2d = mask[0]

        # Đảm bảo mask là nhị phân
        binary_mask = (mask_2d > 0).astype(np.uint8)

        try:
            from skimage import measure

            # Sử dụng find_contours của scikit-image thay thế cho cv2.findContours
            skimage_contours = measure.find_contours(binary_mask, 0.5)

            for contour in skimage_contours:
                # Bỏ qua contour quá nhỏ
                if len(contour) < 3:
                    continue

                # Chuyển đổi tọa độ từ (row, col) sang (x, y) và áp dụng pixel spacing
                # Đảo ngược vì scikit-image trả về (row, col) trong khi ta cần (x, y)
                contour_xy = np.flip(contour, axis=1).astype(np.float32)
                contour_xy[:, 0] = (
                    contour_xy[:, 0] * pixel_spacing[1]
                )  # x = col * spacing_x
                contour_xy[:, 1] = (
                    contour_xy[:, 1] * pixel_spacing[0]
                )  # y = row * spacing_y

                # Thêm tọa độ z
                contour_xyz = np.column_stack(
                    [contour_xy, np.full(len(contour_xy), z_position)]
                )

                contours.append(contour_xyz)

            return contours

        except ImportError:
            logger.warning("scikit-image không khả dụng")
            # Thử cách thay thế khác nếu scikit-image không có sẵn
            try:
                # Phương pháp đơn giản: tìm các pixel biên
                from scipy import ndimage

                # Tìm biên bằng cách phát hiện sự thay đổi giữa trong và ngoài mask
                edges = ndimage.binary_erosion(binary_mask) ^ binary_mask
                edge_points = np.column_stack(np.where(edges))

                if len(edge_points) < 3:
                    logger.warning("Không đủ điểm biên để tạo contour")
                    return contours

                # Sắp xếp các điểm biên để tạo contour liên tục
                # Đây là cách tiếp cận đơn giản, không hoàn hảo như OpenCV hay scikit-image
                from scipy.spatial import distance

                ordered_points = [edge_points[0]]
                remaining_points = edge_points[1:].tolist()

                while remaining_points:
                    # Tìm điểm gần nhất với điểm cuối cùng
                    last_point = ordered_points[-1]
                    distances = [
                        distance.euclidean(last_point, p) for p in remaining_points
                    ]
                    idx = np.argmin(distances)

                    # Thêm điểm gần nhất vào danh sách
                    ordered_points.append(remaining_points.pop(idx))

                # Chuyển đổi từ (row, col) sang (x, y) và áp dụng pixel spacing
                ordered_points = np.array(ordered_points, dtype=np.float32)
                contour_xy = np.column_stack(
                    [ordered_points[:, 1], ordered_points[:, 0]]
                )  # (x=col, y=row)
                contour_xy[:, 0] = (
                    contour_xy[:, 0] * pixel_spacing[1]
                )  # x = col * spacing_x
                contour_xy[:, 1] = (
                    contour_xy[:, 1] * pixel_spacing[0]
                )  # y = row * spacing_y

                # Thêm tọa độ z
                contour_xyz = np.column_stack(
                    [contour_xy, np.full(len(contour_xy), z_position)]
                )

                contours.append(contour_xyz)

            except ImportError:
                logger.warning("scipy không khả dụng, không thể tạo contour")
            except Exception as e:
                logger.error(f"Lỗi khi tạo contour thay thế: {str(e)}")

        return contours

    @staticmethod
    def convert_contours_to_mask(
        contours: List[np.ndarray],
        shape: Tuple[int, int, int],
        slice_thickness: float = 1.0,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> np.ndarray:
        """
        Chuyển đổi danh sách contour thành mặt nạ nhị phân 3D.

        Parameters
        ----------
        contours : List[np.ndarray]
            Danh sách các contour 3D
        shape : Tuple[int, int, int]
            Kích thước mặt nạ đầu ra (z, y, x)
        slice_thickness : float, optional
            Độ dày lát cắt theo mm, mặc định 1.0
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo mm trong mặt phẳng xy, mặc định (1.0, 1.0)

        Returns
        -------
        np.ndarray
            Mặt nạ nhị phân 3D
        """
        mask = np.zeros(shape, dtype=np.uint8)

        if HAS_OPENCV:
            try:
                # Nhóm contours theo lát cắt (tọa độ z)
                contours_by_slice = {}
                for contour in contours:
                    if len(contour) < 3:
                        continue

                    # Lấy tọa độ z (giả sử tất cả các điểm trong một contour có cùng z)
                    z_coord = int(round(contour[0, 2] / slice_thickness))

                    # Kiểm tra xem z_coord có nằm trong phạm vi hợp lệ không
                    if 0 <= z_coord < shape[0]:
                        # Chuyển đổi tọa độ thực về tọa độ pixel
                        contour_2d = contour[:, :2] / np.array(
                            [pixel_spacing[1], pixel_spacing[0]]
                        )
                        contour_2d = np.round(contour_2d).astype(np.int32)

                        # Thêm contour vào slice tương ứng
                        if z_coord not in contours_by_slice:
                            contours_by_slice[z_coord] = []
                        contours_by_slice[z_coord].append(contour_2d)

                # Lấp đầy mỗi contour vào mặt nạ
                for z, slice_contours in contours_by_slice.items():
                    slice_mask = np.zeros((shape[1], shape[2]), dtype=np.uint8)

                    for contour_2d in slice_contours:
                        # Kiểm tra và cắt contour để nằm trong phạm vi ảnh
                        contour_2d = np.clip(
                            contour_2d, 0, [shape[2] - 1, shape[1] - 1]
                        )

                        # Vẽ và lấp đầy contour
                        cv2.fillPoly(slice_mask, [contour_2d], 1)

                    mask[z] = slice_mask
            except Exception as e:
                logger.error(
                    f"Lỗi khi chuyển đổi contours thành mask với OpenCV: {str(e)}"
                )
                # Thử sử dụng phương pháp thay thế
                return ContourUtils._convert_contours_to_mask_alternative(
                    contours, shape, slice_thickness, pixel_spacing
                )
        else:
            # Sử dụng phương pháp thay thế
            return ContourUtils._convert_contours_to_mask_alternative(
                contours, shape, slice_thickness, pixel_spacing
            )

        return mask

    @staticmethod
    def _convert_contours_to_mask_alternative(
        contours: List[np.ndarray],
        shape: Tuple[int, int, int],
        slice_thickness: float = 1.0,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> np.ndarray:
        """
        Phương pháp thay thế để chuyển đổi contours thành mask khi không có OpenCV.

        Parameters
        ----------
        contours : List[np.ndarray]
            Danh sách các contour 3D
        shape : Tuple[int, int, int]
            Kích thước mặt nạ đầu ra (z, y, x)
        slice_thickness : float, optional
            Độ dày lát cắt theo mm, mặc định 1.0
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo mm trong mặt phẳng xy, mặc định (1.0, 1.0)

        Returns
        -------
        np.ndarray
            Mặt nạ nhị phân 3D
        """
        mask = np.zeros(shape, dtype=np.uint8)

        try:
            import matplotlib.path as mpath

            # Nhóm contours theo lát cắt
            contours_by_slice = {}
            for contour in contours:
                if len(contour) < 3:
                    continue

                # Lấy tọa độ z
                z_coord = int(round(contour[0, 2] / slice_thickness))

                if 0 <= z_coord < shape[0]:
                    # Chuyển đổi tọa độ thực về tọa độ pixel
                    contour_2d = contour[:, :2] / np.array(
                        [pixel_spacing[1], pixel_spacing[0]]
                    )

                    if z_coord not in contours_by_slice:
                        contours_by_slice[z_coord] = []
                    contours_by_slice[z_coord].append(contour_2d)

            # Xử lý từng lát cắt
            for z, slice_contours in contours_by_slice.items():
                # Tạo lưới tọa độ
                y, x = np.mgrid[: shape[1], : shape[2]]
                points = np.c_[x.ravel(), y.ravel()]

                # Tạo mặt nạ cho lát cắt hiện tại
                slice_mask = np.zeros((shape[1], shape[2]), dtype=np.uint8)

                for contour_2d in slice_contours:
                    # Tạo đường dẫn
                    path = mpath.Path(contour_2d)

                    # Kiểm tra điểm nào nằm trong contour
                    mask_points = path.contains_points(points)
                    mask_2d = mask_points.reshape(shape[1], shape[2])

                    # Gán vào mặt nạ chính
                    slice_mask = np.logical_or(slice_mask, mask_2d)

                mask[z] = slice_mask.astype(np.uint8)

        except ImportError:
            logger.error(
                "Không thể import matplotlib.path, sử dụng phương pháp đơn giản hơn"
            )
            # Phương pháp đơn giản hơn nếu không có matplotlib
            for contour in contours:
                if len(contour) < 3:
                    continue

                # Lấy tọa độ z
                z_coord = int(round(contour[0, 2] / slice_thickness))

                if 0 <= z_coord < shape[0]:
                    # Chuyển đổi tọa độ thực về tọa độ pixel
                    points = contour[:, :2] / np.array(
                        [pixel_spacing[1], pixel_spacing[0]]
                    )
                    points = np.round(points).astype(np.int32)

                    # Phương pháp đơn giản: lấp đầy hộp giới hạn
                    min_x = max(0, np.min(points[:, 0]))
                    max_x = min(shape[2] - 1, np.max(points[:, 0]))
                    min_y = max(0, np.min(points[:, 1]))
                    max_y = min(shape[1] - 1, np.max(points[:, 1]))

                    # Lấp đầy hộp giới hạn
                    mask[
                        z_coord,
                        int(min_y) : int(max_y) + 1,
                        int(min_x) : int(max_x) + 1,
                    ] = 1

        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi contours thành mask: {str(e)}")

        return mask

    @staticmethod
    def calculate_contour_area(contour: np.ndarray) -> float:
        """
        Tính diện tích của contour 2D sử dụng công thức Shoelace.

        Parameters
        ----------
        contour : np.ndarray
            Contour 2D hoặc 3D (nếu 3D, chỉ sử dụng 2 tọa độ đầu tiên)

        Returns
        -------
        float
            Diện tích của contour
        """
        if len(contour) < 3:
            return 0.0

        # Lấy tọa độ x, y
        if contour.shape[1] > 2:
            x = contour[:, 0]
            y = contour[:, 1]
        else:
            x = contour[:, 0]
            y = contour[:, 1]

        # Áp dụng công thức Shoelace (Gauss's area formula)
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def calculate_contour_centroid(contour: np.ndarray) -> np.ndarray:
        """
        Tính tâm của contour.

        Parameters
        ----------
        contour : np.ndarray
            Contour đầu vào

        Returns
        -------
        np.ndarray
            Tọa độ tâm của contour
        """
        return np.mean(contour, axis=0)

    @staticmethod
    def check_contour_intersection(contour1: np.ndarray, contour2: np.ndarray) -> bool:
        """
        Kiểm tra xem hai contour có giao nhau không.

        Parameters
        ----------
        contour1 : np.ndarray
            Contour thứ nhất
        contour2 : np.ndarray
            Contour thứ hai

        Returns
        -------
        bool
            True nếu hai contour giao nhau, False nếu không
        """
        if HAS_SHAPELY:
            try:
                # Tạo đa giác từ contour (chỉ xét tọa độ x, y)
                poly1 = Polygon(contour1[:, :2])
                poly2 = Polygon(contour2[:, :2])

                # Kiểm tra giao nhau
                return poly1.intersects(poly2)
            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra giao nhau với Shapely: {str(e)}")
                return ContourUtils._check_contour_intersection_alternative(
                    contour1, contour2
                )
        else:
            return ContourUtils._check_contour_intersection_alternative(
                contour1, contour2
            )

    @staticmethod
    def _check_contour_intersection_alternative(
        contour1: np.ndarray, contour2: np.ndarray
    ) -> bool:
        """
        Phương pháp thay thế để kiểm tra giao nhau khi không có Shapely.

        Parameters
        ----------
        contour1 : np.ndarray
            Contour thứ nhất
        contour2 : np.ndarray
            Contour thứ hai

        Returns
        -------
        bool
            True nếu hai contour giao nhau, False nếu không
        """
        try:
            import matplotlib.path as mpath

            # Tạo đường dẫn từ contour1
            path1 = mpath.Path(contour1[:, :2])

            # Kiểm tra từng điểm của contour2 có nằm trong contour1 không
            inside_points = path1.contains_points(contour2[:, :2])

            # Nếu bất kỳ điểm nào của contour2 nằm trong contour1, chúng giao nhau
            if np.any(inside_points):
                return True

            # Tạo đường dẫn từ contour2
            path2 = mpath.Path(contour2[:, :2])

            # Kiểm tra từng điểm của contour1 có nằm trong contour2 không
            inside_points = path2.contains_points(contour1[:, :2])

            # Nếu bất kỳ điểm nào của contour1 nằm trong contour2, chúng giao nhau
            return np.any(inside_points)

        except ImportError:
            logger.error(
                "Không thể import matplotlib.path, sử dụng phương pháp đơn giản"
            )

            # Phương pháp đơn giản: kiểm tra hộp giới hạn
            min1_x, min1_y = np.min(contour1[:, :2], axis=0)
            max1_x, max1_y = np.max(contour1[:, :2], axis=0)

            min2_x, min2_y = np.min(contour2[:, :2], axis=0)
            max2_x, max2_y = np.max(contour2[:, :2], axis=0)

            # Kiểm tra xem các hộp giới hạn có giao nhau không
            return not (
                max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y
            )
