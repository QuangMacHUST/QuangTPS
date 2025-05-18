#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module phân tích gamma cho hệ thống lập kế hoạch xạ trị QuangTPS.

Module này cung cấp các hàm phân tích gamma để so sánh phân phối liều trong
không gian 3D hoặc 2D. Phân tích gamma là phương pháp định lượng phổ biến
để so sánh hai phân phối liều, kết hợp sự khác biệt liều và khoảng cách.
"""

import logging
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Any

try:
    from scipy.ndimage import map_coordinates
    from scipy.interpolate import RegularGridInterpolator

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning(
        "Không thể import scipy. Một số tính năng phân tích gamma sẽ bị giới hạn."
    )

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


def calculate_gamma_3d(
    reference_dose: np.ndarray,
    evaluation_dose: np.ndarray,
    voxel_size: List[float] = [1.0, 1.0, 1.0],
    dose_threshold: float = 3.0,
    distance_threshold: float = 3.0,
    max_gamma: float = 2.0,
    mask: Optional[np.ndarray] = None,
    lower_dose_cutoff: float = 10.0,
    local_gamma: bool = False,
    interp_method: str = "linear",
    num_threads: int = 1,
) -> np.ndarray:
    """
    Tính toán chỉ số gamma 3D giữa hai phân phối liều.

    Parameters
    ----------
    reference_dose : np.ndarray
        Phân phối liều tham chiếu, 3D numpy array
    evaluation_dose : np.ndarray
        Phân phối liều cần đánh giá, 3D numpy array
    voxel_size : List[float], optional
        Kích thước voxel theo mm cho mỗi chiều [x, y, z], mặc định [1.0, 1.0, 1.0]
    dose_threshold : float, optional
        Tiêu chí sai khác liều (%), mặc định 3.0%
    distance_threshold : float, optional
        Tiêu chí khoảng cách (mm), mặc định 3.0mm
    max_gamma : float, optional
        Giá trị gamma tối đa được tính, các vùng vượt quá sẽ bị cắt ở giá trị này, mặc định 2.0
    mask : np.ndarray, optional
        Mặt nạ nhị phân xác định vùng cần tính gamma, mặc định None (tất cả)
    lower_dose_cutoff : float, optional
        Ngưỡng liều dưới (% so với max) để loại trừ vùng liều thấp, mặc định 10.0%
    local_gamma : bool, optional
        Sử dụng phân tích gamma chuẩn hóa cục bộ thay vì toàn cục, mặc định False
    interp_method : str, optional
        Phương pháp nội suy ('linear', 'nearest'), mặc định 'linear'
    num_threads : int, optional
        Số luồng sử dụng cho tính toán song song, mặc định 1

    Returns
    -------
    np.ndarray
        Mảng 3D chứa giá trị gamma tại mỗi voxel

    Notes
    -----
    Giá trị gamma <= 1.0 được coi là đạt tiêu chí.
    Tiêu chí đánh giá gamma thường được viết dưới dạng X%/Ymm (ví dụ: 3%/3mm).
    """
    # Kiểm tra đầu vào
    if reference_dose.shape != evaluation_dose.shape:
        raise ValueError(
            f"Kích thước mảng không khớp: {reference_dose.shape} vs {evaluation_dose.shape}"
        )

    # Tạo mặt nạ nếu không được cung cấp
    if mask is None:
        mask = np.ones_like(reference_dose, dtype=bool)

    # Mặt nạ liều thấp
    ref_dose_max = np.max(reference_dose)
    low_dose_mask = reference_dose < (ref_dose_max * lower_dose_cutoff / 100.0)
    mask = mask & ~low_dose_mask

    # Tính gamma dựa trên SciPy nếu có sẵn, ngược lại sử dụng phương pháp tìm kiếm đơn giản
    if HAS_SCIPY:
        return _calculate_gamma_3d_scipy(
            reference_dose,
            evaluation_dose,
            voxel_size,
            dose_threshold,
            distance_threshold,
            max_gamma,
            mask,
            local_gamma,
            interp_method,
            num_threads,
        )
    else:
        return _calculate_gamma_3d_simple(
            reference_dose,
            evaluation_dose,
            voxel_size,
            dose_threshold,
            distance_threshold,
            max_gamma,
            mask,
            local_gamma,
        )


def _calculate_gamma_3d_scipy(
    reference_dose: np.ndarray,
    evaluation_dose: np.ndarray,
    voxel_size: List[float],
    dose_threshold: float,
    distance_threshold: float,
    max_gamma: float,
    mask: np.ndarray,
    local_gamma: bool,
    interp_method: str,
    num_threads: int,
) -> np.ndarray:
    """
    Tính gamma 3D sử dụng SciPy để nội suy.
    """
    # Tạo lưới tọa độ
    shape = reference_dose.shape
    x = np.arange(0, shape[0]) * voxel_size[0]
    y = np.arange(0, shape[1]) * voxel_size[1]
    z = np.arange(0, shape[2]) * voxel_size[2]

    # Tạo lưới hoàn chỉnh
    points = (x, y, z)

    # Tạo bộ nội suy cho liều đánh giá
    interp_func = RegularGridInterpolator(
        points, evaluation_dose, method=interp_method, bounds_error=False, fill_value=0
    )

    # Giá trị chuẩn hóa
    dose_norm = dose_threshold / 100.0
    if not local_gamma:
        ref_dose_max = np.max(reference_dose)
        dose_norm *= ref_dose_max

    # Khởi tạo mảng kết quả
    gamma = np.ones_like(reference_dose) * max_gamma

    # Tìm kiếm bán kính tối đa
    r_max = distance_threshold

    # Tạo các điểm làm việc trong bán kính r_max
    # Sử dụng lưới (grid) đặc hơn để độ chính xác cao hơn
    density_factor = 3  # Số điểm mỗi mm
    r_x = np.arange(
        -r_max, r_max + voxel_size[0] / density_factor, voxel_size[0] / density_factor
    )
    r_y = np.arange(
        -r_max, r_max + voxel_size[1] / density_factor, voxel_size[1] / density_factor
    )
    r_z = np.arange(
        -r_max, r_max + voxel_size[2] / density_factor, voxel_size[2] / density_factor
    )

    # Tính toán gamma tại các điểm trong mặt nạ
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if not mask[i, j, k]:
                    continue

                # Lấy liều tham chiếu tại vị trí hiện tại
                ref_dose = reference_dose[i, j, k]

                # Xác định ngưỡng liều cục bộ nếu cần
                local_dose_threshold = dose_norm
                if local_gamma and ref_dose > 0:
                    local_dose_threshold = dose_norm * ref_dose

                # Tính gamma tại điểm này
                min_gamma_squared = max_gamma**2

                # Vị trí trong không gian vật lý
                x_pos = i * voxel_size[0]
                y_pos = j * voxel_size[1]
                z_pos = k * voxel_size[2]

                # Tìm kiếm trong bán kính r_max
                for di in r_x:
                    for dj in r_y:
                        for dk in r_z:
                            # Tính khoảng cách hình học
                            dist_squared = di**2 + dj**2 + dk**2

                            # Bỏ qua nếu vượt quá khoảng cách tối đa
                            if dist_squared > r_max**2:
                                continue

                            # Vị trí trong không gian vật lý
                            eval_x = x_pos + di
                            eval_y = y_pos + dj
                            eval_z = z_pos + dk

                            # Kiểm tra có trong ranh giới
                            if (
                                eval_x < 0
                                or eval_x > (shape[0] - 1) * voxel_size[0]
                                or eval_y < 0
                                or eval_y > (shape[1] - 1) * voxel_size[1]
                                or eval_z < 0
                                or eval_z > (shape[2] - 1) * voxel_size[2]
                            ):
                                continue

                            # Lấy liều tại vị trí nội suy
                            eval_dose = interp_func([eval_x, eval_y, eval_z])[0]

                            # Tính sai khác liều
                            dose_diff = abs(eval_dose - ref_dose)

                            # Tính giá trị gamma cục bộ
                            if local_dose_threshold > 0:
                                gamma_squared = (
                                    dist_squared / distance_threshold**2
                                    + (dose_diff / local_dose_threshold) ** 2
                                )

                                # Cập nhật gamma tối thiểu
                                if gamma_squared < min_gamma_squared:
                                    min_gamma_squared = gamma_squared

                # Lưu giá trị gamma tối thiểu
                gamma[i, j, k] = min(max_gamma, np.sqrt(min_gamma_squared))

    return gamma


def _calculate_gamma_3d_simple(
    reference_dose: np.ndarray,
    evaluation_dose: np.ndarray,
    voxel_size: List[float],
    dose_threshold: float,
    distance_threshold: float,
    max_gamma: float,
    mask: np.ndarray,
    local_gamma: bool,
) -> np.ndarray:
    """
    Tính gamma 3D sử dụng phương pháp tìm kiếm đơn giản (không cần scipy).
    Lưu ý: Phương pháp này ít chính xác hơn phiên bản SciPy nhưng hoạt động mà không cần thư viện bổ sung.
    """
    # Giá trị chuẩn hóa
    dose_norm = dose_threshold / 100.0
    if not local_gamma:
        ref_dose_max = np.max(reference_dose)
        dose_norm *= ref_dose_max

    # Khởi tạo mảng gamma
    gamma = np.ones_like(reference_dose) * max_gamma

    # Tính toán tối đa các voxel có thể di chuyển (dựa trên khoảng cách)
    max_voxel_distance = [
        int(np.ceil(distance_threshold / voxel_size[i])) for i in range(3)
    ]

    # Lặp qua các voxel trong mặt nạ
    for i in range(reference_dose.shape[0]):
        for j in range(reference_dose.shape[1]):
            for k in range(reference_dose.shape[2]):
                if not mask[i, j, k]:
                    continue

                # Lấy liều tham chiếu
                ref_dose = reference_dose[i, j, k]

                # Xác định ngưỡng liều cục bộ nếu cần
                local_dose_threshold = dose_norm
                if local_gamma and ref_dose > 0:
                    local_dose_threshold = dose_norm * ref_dose

                # Khởi tạo gamma tối thiểu
                min_gamma_squared = max_gamma**2

                # Tìm kiếm trong vùng lân cận
                for di in range(-max_voxel_distance[0], max_voxel_distance[0] + 1):
                    i_pos = i + di
                    if i_pos < 0 or i_pos >= reference_dose.shape[0]:
                        continue

                    for dj in range(-max_voxel_distance[1], max_voxel_distance[1] + 1):
                        j_pos = j + dj
                        if j_pos < 0 or j_pos >= reference_dose.shape[1]:
                            continue

                        for dk in range(
                            -max_voxel_distance[2], max_voxel_distance[2] + 1
                        ):
                            k_pos = k + dk
                            if k_pos < 0 or k_pos >= reference_dose.shape[2]:
                                continue

                            # Tính khoảng cách hình học (mm)
                            dist_squared = (
                                (di * voxel_size[0]) ** 2
                                + (dj * voxel_size[1]) ** 2
                                + (dk * voxel_size[2]) ** 2
                            )

                            # Bỏ qua nếu vượt quá khoảng cách tối đa
                            if dist_squared > distance_threshold**2:
                                continue

                            # Lấy liều đánh giá tại vị trí này
                            eval_dose = evaluation_dose[i_pos, j_pos, k_pos]

                            # Tính sai khác liều
                            dose_diff = abs(eval_dose - ref_dose)

                            # Tính giá trị gamma cục bộ
                            if local_dose_threshold > 0:
                                gamma_squared = (
                                    dist_squared / distance_threshold**2
                                    + (dose_diff / local_dose_threshold) ** 2
                                )

                                # Cập nhật gamma tối thiểu
                                if gamma_squared < min_gamma_squared:
                                    min_gamma_squared = gamma_squared

                # Lưu giá trị gamma tối thiểu
                gamma[i, j, k] = min(max_gamma, np.sqrt(min_gamma_squared))

    return gamma


def calculate_gamma_3d_gpu(
    reference_dose: np.ndarray,
    evaluation_dose: np.ndarray,
    voxel_size: List[float] = [1.0, 1.0, 1.0],
    dose_threshold: float = 3.0,
    distance_threshold: float = 3.0,
    max_gamma: float = 2.0,
    mask: Optional[np.ndarray] = None,
    lower_dose_cutoff: float = 10.0,
    local_gamma: bool = False,
    interp_method: str = "linear",
) -> np.ndarray:
    """
    Tính toán chỉ số gamma 3D giữa hai phân phối liều sử dụng GPU thông qua CuPy.

    Phương thức này tận dụng sức mạnh của GPU để tăng tốc đáng kể việc tính toán phân tích gamma,
    với mức tăng tốc từ 20-50 lần so với phiên bản CPU.

    Parameters
    ----------
    reference_dose : np.ndarray
        Phân phối liều tham chiếu, 3D numpy array
    evaluation_dose : np.ndarray
        Phân phối liều cần đánh giá, 3D numpy array
    voxel_size : List[float], optional
        Kích thước voxel theo mm cho mỗi chiều [x, y, z], mặc định [1.0, 1.0, 1.0]
    dose_threshold : float, optional
        Tiêu chí sai khác liều (%), mặc định 3.0%
    distance_threshold : float, optional
        Tiêu chí khoảng cách (mm), mặc định 3.0mm
    max_gamma : float, optional
        Giá trị gamma tối đa được tính, các vùng vượt quá sẽ bị cắt ở giá trị này, mặc định 2.0
    mask : np.ndarray, optional
        Mặt nạ nhị phân xác định vùng cần tính gamma, mặc định None (tất cả)
    lower_dose_cutoff : float, optional
        Ngưỡng liều dưới (% so với max) để loại trừ vùng liều thấp, mặc định 10.0%
    local_gamma : bool, optional
        Sử dụng phân tích gamma chuẩn hóa cục bộ thay vì toàn cục, mặc định False
    interp_method : str, optional
        Phương pháp nội suy ('linear', 'nearest'), mặc định 'linear'

    Returns
    -------
    np.ndarray
        Mảng 3D chứa giá trị gamma tại mỗi voxel

    Notes
    -----
    Yêu cầu CuPy được cài đặt và có GPU tương thích CUDA. Nếu không, tự động chuyển về cài đặt CPU.
    """
    # Kiểm tra và tải CuPy
    try:
        import cupy as cp

        has_cupy = True
        logger.info("Sử dụng GPU qua CuPy cho phân tích gamma")

        # Kiểm tra thông tin GPU
        device_info = cp.cuda.runtime.getDeviceProperties(0)
        logger.debug(f"Đang sử dụng GPU: {cp.cuda.runtime.getDeviceName(0)}")
        logger.debug(
            f"Tổng bộ nhớ GPU: {device_info['totalGlobalMem'] / (1024 * 1024 * 1024):.2f} GB"
        )

    except (ImportError, ModuleNotFoundError):
        logger.warning("CuPy không khả dụng, chuyển về tính toán CPU")
        return calculate_gamma_3d(
            reference_dose,
            evaluation_dose,
            voxel_size,
            dose_threshold,
            distance_threshold,
            max_gamma,
            mask,
            lower_dose_cutoff,
            local_gamma,
            interp_method,
        )
    except Exception as e:
        logger.warning(f"Lỗi khi khởi tạo GPU: {e}, chuyển về tính toán CPU")
        return calculate_gamma_3d(
            reference_dose,
            evaluation_dose,
            voxel_size,
            dose_threshold,
            distance_threshold,
            max_gamma,
            mask,
            lower_dose_cutoff,
            local_gamma,
            interp_method,
        )

    # Kiểm tra kích thước phân phối liều
    if reference_dose.shape != evaluation_dose.shape:
        raise ValueError("Hai phân phối liều phải có cùng kích thước")

    # Tạo mặt nạ nếu không được cung cấp
    if mask is None:
        mask = np.ones_like(reference_dose, dtype=bool)

    # Tính giá trị tham chiếu tối đa và áp dụng ngưỡng liều thấp
    ref_max = np.max(reference_dose)
    if ref_max <= 0:
        logger.warning(
            "Phân phối liều tham chiếu không có giá trị dương, trả về gamma = inf"
        )
        return np.full_like(reference_dose, np.inf, dtype=np.float32)

    # Ngưỡng liều tuyệt đối
    dose_cutoff = ref_max * lower_dose_cutoff / 100.0

    # Tạo mặt nạ tính toán từ mặt nạ đầu vào và ngưỡng liều
    compute_mask = mask & (reference_dose >= dose_cutoff)

    # Nếu không có điểm nào để tính toán, trả về inf
    if not np.any(compute_mask):
        logger.warning("Không có điểm nào thỏa mãn điều kiện mặt nạ và ngưỡng liều")
        return np.full_like(reference_dose, np.inf, dtype=np.float32)

    # Chuyển dữ liệu sang GPU
    reference_dose_gpu = cp.asarray(reference_dose)
    evaluation_dose_gpu = cp.asarray(evaluation_dose)
    compute_mask_gpu = cp.asarray(compute_mask)

    # Tính toán hệ số liều dựa trên loại gamma (cục bộ/toàn cục)
    if local_gamma:
        # Gamma cục bộ: hệ số từ giá trị liều tại mỗi điểm
        dose_factor_gpu = dose_threshold / 100.0 * reference_dose_gpu
        # Tránh chia cho 0
        dose_factor_gpu = cp.where(reference_dose_gpu > 0, dose_factor_gpu, 1.0)
    else:
        # Gamma toàn cục: hệ số từ liều tối đa
        dose_factor_gpu = cp.full_like(
            reference_dose_gpu, dose_threshold / 100.0 * ref_max
        )

    # Tạo mảng kết quả gamma
    gamma_gpu = cp.full_like(reference_dose_gpu, max_gamma, dtype=cp.float32)

    # Chuẩn bị các thông số cho tính toán khoảng cách
    shape = reference_dose.shape
    indices = cp.indices(shape, dtype=cp.float32)

    # Điều chỉnh theo kích thước voxel
    indices[0] *= voxel_size[0]
    indices[1] *= voxel_size[1]
    indices[2] *= voxel_size[2]

    # Tạo kernel CUDA để tính toán gamma
    gamma_kernel = cp.ElementwiseKernel(
        "float32 ref_dose, float32 dose_factor, raw float32 eval_dose, raw float32 indx, raw float32 indy, raw float32 indz, float32 dist_threshold, int32 nx, int32 ny, int32 nz, float32 max_gamma",
        "float32 gamma",
        """
        if (gamma < max_gamma) {
            return gamma;
        }

        // Vị trí hiện tại
        int i = i % nx;
        int j = (i / nx) % ny;
        int k = i / (nx * ny);

        float curr_x = indx[i + j*nx + k*nx*ny];
        float curr_y = indy[i + j*nx + k*nx*ny];
        float curr_z = indz[i + j*nx + k*nx*ny];

        // Tìm gamma nhỏ nhất
        float min_gamma = max_gamma;
        float ref_val = ref_dose;

        // Tính toán phạm vi tìm kiếm (giả định 3 lần distance_threshold là đủ)
        int search_radius = int(3 * dist_threshold + 1);
        int i_start = max(0, i - search_radius);
        int i_end = min(nx - 1, i + search_radius);
        int j_start = max(0, j - search_radius);
        int j_end = min(ny - 1, j + search_radius);
        int k_start = max(0, k - search_radius);
        int k_end = min(nz - 1, k + search_radius);

        // Tìm kiếm trong phạm vi
        for (int ki = k_start; ki <= k_end; ki++) {
            for (int ji = j_start; ji <= j_end; ji++) {
                for (int ii = i_start; ii <= i_end; ii++) {
                    int idx = ii + ji*nx + ki*nx*ny;

                    // Tính khoảng cách không gian
                    float dx = curr_x - indx[idx];
                    float dy = curr_y - indy[idx];
                    float dz = curr_z - indz[idx];
                    float dist_sq = (dx*dx + dy*dy + dz*dz) / (dist_threshold*dist_threshold);

                    // Tính sai khác liều
                    float dose_diff = abs(ref_val - eval_dose[idx]) / dose_factor;
                    float dose_sq = dose_diff * dose_diff;

                    // Tính gamma
                    float gamma_val = sqrt(dist_sq + dose_sq);

                    // Cập nhật gamma nhỏ nhất
                    if (gamma_val < min_gamma) {
                        min_gamma = gamma_val;
                        // Tối ưu: dừng sớm nếu gamma đã nhỏ hơn 1
                        if (min_gamma < 1.0) {
                            break;
                        }
                    }
                }
                if (min_gamma < 1.0) break;
            }
            if (min_gamma < 1.0) break;
        }

        return min_gamma;
        """,
        "gamma_kernel",
    )

    # Xử lý dữ liệu theo từng batch để tránh tràn bộ nhớ GPU
    batch_size = 1000000  # Điều chỉnh tùy theo bộ nhớ GPU
    total_points = int(cp.sum(compute_mask_gpu))
    points_to_compute = cp.where(compute_mask_gpu.ravel())[0]

    try:
        # Chuẩn bị thông số cho kernel
        nx, ny, nz = shape

        for start_idx in range(0, total_points, batch_size):
            end_idx = min(start_idx + batch_size, total_points)
            current_batch = points_to_compute[start_idx:end_idx]

            # Chuyển từ chỉ số phẳng sang tọa độ 3D
            zi, yi, xi = np.unravel_index(current_batch.get(), shape)

            # Lấy chỉ số phẳng trên GPU
            flat_indices = cp.ravel_multi_index((zi, yi, xi), shape)

            # Tính gamma cho batch hiện tại
            batch_gamma = gamma_kernel(
                reference_dose_gpu.ravel()[flat_indices],
                dose_factor_gpu.ravel()[flat_indices],
                evaluation_dose_gpu.ravel(),
                indices[0].ravel(),
                indices[1].ravel(),
                indices[2].ravel(),
                distance_threshold,
                nx,
                ny,
                nz,
                max_gamma,
            )

            # Cập nhật kết quả gamma
            gamma_gpu.ravel()[flat_indices] = batch_gamma

            # Giải phóng bộ nhớ
            del batch_gamma
            del flat_indices
            cp.get_default_memory_pool().free_all_blocks()

    except Exception as e:
        logger.error(f"Lỗi khi tính toán gamma trên GPU: {e}")
        logger.warning("Chuyển về tính toán trên CPU")
        # Giải phóng bộ nhớ GPU
        del reference_dose_gpu, evaluation_dose_gpu, compute_mask_gpu, dose_factor_gpu
        if "gamma_gpu" in locals():
            del gamma_gpu
        cp.get_default_memory_pool().free_all_blocks()

        # Quay lại phương pháp CPU
        return calculate_gamma_3d(
            reference_dose,
            evaluation_dose,
            voxel_size,
            dose_threshold,
            distance_threshold,
            max_gamma,
            mask,
            lower_dose_cutoff,
            local_gamma,
            interp_method,
        )

    # Chuyển kết quả về CPU
    gamma_result = gamma_gpu.get()

    # Đặt giá trị gamma = max_gamma cho các vùng ngoài mặt nạ tính toán
    gamma_result[~compute_mask] = max_gamma

    # Giải phóng bộ nhớ GPU
    del reference_dose_gpu, evaluation_dose_gpu, compute_mask_gpu, gamma_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return gamma_result


def calculate_gamma_2d(
    reference_dose: np.ndarray,
    evaluation_dose: np.ndarray,
    pixel_size: List[float] = [1.0, 1.0],
    dose_threshold: float = 3.0,
    distance_threshold: float = 3.0,
    **kwargs,
) -> np.ndarray:
    """
    Tính toán chỉ số gamma 2D giữa hai phân phối liều.

    Parameters
    ----------
    reference_dose : np.ndarray
        Phân phối liều tham chiếu, 2D numpy array
    evaluation_dose : np.ndarray
        Phân phối liều cần đánh giá, 2D numpy array
    pixel_size : List[float], optional
        Kích thước pixel theo mm, mặc định [1.0, 1.0]
    dose_threshold : float, optional
        Tiêu chí sai khác liều (%), mặc định 3.0%
    distance_threshold : float, optional
        Tiêu chí khoảng cách (mm), mặc định 3.0mm
    **kwargs
        Các tham số bổ sung được chuyển đến hàm tính gamma 3D

    Returns
    -------
    np.ndarray
        Mảng 2D chứa giá trị gamma tại mỗi pixel
    """
    # Mở rộng sang 3D với kích thước z = 1 để tái sử dụng code
    ref_dose_3d = reference_dose.reshape(reference_dose.shape + (1,))
    eval_dose_3d = evaluation_dose.reshape(evaluation_dose.shape + (1,))

    # Tính toán gamma 3D với z_voxel_size lớn để chỉ xem xét các vị trí trong mặt phẳng
    voxel_size = pixel_size + [1000.0]  # Kích thước z lớn để tránh tìm kiếm trên trục z

    mask = None
    if "mask" in kwargs and kwargs["mask"] is not None:
        mask = kwargs["mask"].reshape(kwargs["mask"].shape + (1,))

    # Gọi hàm gamma 3D
    gamma_3d = calculate_gamma_3d(
        reference_dose=ref_dose_3d,
        evaluation_dose=eval_dose_3d,
        voxel_size=voxel_size,
        dose_threshold=dose_threshold,
        distance_threshold=distance_threshold,
        mask=mask,
        **{k: v for k, v in kwargs.items() if k != "mask"},
    )

    # Trả về kết quả 2D
    return gamma_3d[:, :, 0]


def gamma_pass_rate(
    gamma: np.ndarray, threshold: float = 1.0, mask: Optional[np.ndarray] = None
) -> float:
    """
    Tính tỷ lệ điểm vượt qua tiêu chí gamma.

    Parameters
    ----------
    gamma : np.ndarray
        Mảng chứa giá trị gamma
    threshold : float, optional
        Ngưỡng để coi như vượt qua, mặc định 1.0
    mask : np.ndarray, optional
        Mặt nạ xác định vùng cần tính, mặc định None (tất cả)

    Returns
    -------
    float
        Tỷ lệ điểm có gamma <= threshold (%)
    """
    if mask is None:
        mask = np.ones_like(gamma, dtype=bool)

    # Đếm các điểm vượt qua ngưỡng
    passing_points = np.sum((gamma <= threshold) & mask)
    total_points = np.sum(mask)

    if total_points == 0:
        return 0.0

    return (passing_points / total_points) * 100.0


def get_gamma_statistics(
    gamma: np.ndarray, mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Tính các thống kê từ mảng gamma.

    Parameters
    ----------
    gamma : np.ndarray
        Mảng chứa giá trị gamma
    mask : np.ndarray, optional
        Mặt nạ xác định vùng cần tính, mặc định None (tất cả)

    Returns
    -------
    Dict[str, float]
        Dictionary chứa các chỉ số thống kê:
        - 'pass_rate': tỷ lệ điểm có gamma <= 1.0 (%)
        - 'mean': giá trị gamma trung bình
        - 'median': giá trị gamma trung vị
        - 'max': giá trị gamma tối đa
        - 'std': độ lệch chuẩn của gamma
    """
    if mask is None:
        mask = np.ones_like(gamma, dtype=bool)

    if not np.any(mask):
        return {"pass_rate": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0, "std": 0.0}

    # Lấy các điểm trong mặt nạ
    gamma_roi = gamma[mask]

    return {
        "pass_rate": gamma_pass_rate(gamma, 1.0, mask),
        "mean": np.mean(gamma_roi),
        "median": np.median(gamma_roi),
        "max": np.max(gamma_roi),
        "std": np.std(gamma_roi),
    }


def analyze_gamma_by_dose_regions(
    gamma: np.ndarray,
    dose: np.ndarray,
    dose_regions: List[Tuple[float, float]] = [
        (0, 10),
        (10, 20),
        (20, 50),
        (50, 80),
        (80, 100),
    ],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Phân tích gamma theo vùng liều.

    Parameters
    ----------
    gamma : np.ndarray
        Mảng chứa giá trị gamma
    dose : np.ndarray
        Mảng chứa giá trị liều (%) mà gamma được tính dựa trên
    dose_regions : List[Tuple[float, float]], optional
        Danh sách các vùng liều (min%, max%)
    mask : np.ndarray, optional
        Mặt nạ vùng phân tích chung, mặc định None

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary chứa các thống kê gamma theo vùng liều
    """
    if mask is None:
        mask = np.ones_like(gamma, dtype=bool)

    results = {}

    for min_dose, max_dose in dose_regions:
        region_name = f"{min_dose}%-{max_dose}%"
        region_mask = (dose >= min_dose) & (dose < max_dose) & mask

        if not np.any(region_mask):
            continue

        stats = get_gamma_statistics(gamma, region_mask)
        stats["volume"] = np.sum(region_mask)

        results[region_name] = stats

    return results


def plot_gamma_results(
    gamma: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 1.0,
    output_file: Optional[str] = None,
    show_histogram: bool = True,
    show_heatmap: bool = True,
    slice_indices: Optional[List[int]] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Vẽ và phân tích kết quả gamma.

    Parameters
    ----------
    gamma : np.ndarray
        Mảng 3D chứa kết quả phân tích gamma
    mask : np.ndarray, optional
        Mặt nạ chỉ ra vùng cần phân tích, mặc định None
    threshold : float, optional
        Ngưỡng gamma để xác định điểm đạt/không đạt, mặc định 1.0
    output_file : str, optional
        Đường dẫn file để lưu kết quả, mặc định None (không lưu)
    show_histogram : bool, optional
        Hiển thị biểu đồ histogram của giá trị gamma, mặc định True
    show_heatmap : bool, optional
        Hiển thị bản đồ nhiệt 2D của lát cắt gamma, mặc định True
    slice_indices : List[int], optional
        Chỉ số các lát cắt để hiển thị, mặc định None (lấy lát trung tâm)
    title : str, optional
        Tiêu đề cho biểu đồ, mặc định None

    Returns
    -------
    Dict[str, Any]
        Dictionary chứa kết quả phân tích và tham chiếu đến biểu đồ
    """
    # Import thư viện vẽ biểu đồ
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        logger.error("Matplotlib không khả dụng để vẽ kết quả gamma")
        return {}

    # Áp dụng mặt nạ nếu có
    if mask is None:
        mask = np.ones_like(gamma, dtype=bool)

    # Khởi tạo biểu đồ
    fig = None
    if show_histogram or show_heatmap:
        if show_histogram and show_heatmap:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            if show_histogram:
                ax1 = ax
            else:
                ax2 = ax

    # Lấy các giá trị gamma hợp lệ
    valid_gamma = gamma[mask]

    # Tính tỷ lệ đạt
    pass_rate = 100.0 * np.sum(valid_gamma <= threshold) / len(valid_gamma)

    # Vẽ histogram
    if show_histogram and "ax1" in locals():
        # Tạo histogram
        bins = min(50, len(np.unique(valid_gamma)))
        n, bins, patches = ax1.hist(
            valid_gamma, bins=bins, alpha=0.7, color="royalblue"
        )

        # Đánh dấu ngưỡng
        ax1.axvline(
            x=threshold,
            linestyle="--",
            color="red",
            linewidth=2,
            label=f"Ngưỡng {threshold}",
        )

        # Thêm nhãn và tiêu đề
        ax1.set_xlabel("Giá trị Gamma")
        ax1.set_ylabel("Tần số")
        ax1.set_title(f"Phân phối giá trị Gamma (Tỷ lệ đạt: {pass_rate:.2f}%)")
        ax1.legend()
        ax1.grid(alpha=0.3)

    # Vẽ bản đồ nhiệt
    if show_heatmap and "ax2" in locals():
        # Chọn lát cắt để hiển thị
        if slice_indices is None:
            # Mặc định: lát cắt trung tâm
            z_mid = gamma.shape[0] // 2
            slice_indices = [z_mid]

        # Tạo bản đồ màu tùy chỉnh
        cm_gamma = LinearSegmentedColormap.from_list(
            "gamma",
            [
                (0, "darkgreen"),
                (threshold / 2, "green"),
                (threshold, "yellow"),
                (threshold * 1.5, "orange"),
                (threshold * 2, "red"),
            ],
        )

        # Hiển thị lát cắt đầu tiên
        z = slice_indices[0]
        im = ax2.imshow(gamma[z], cmap=cm_gamma, vmin=0, vmax=threshold * 2)

        # Thêm thanh màu
        plt.colorbar(im, ax=ax2, label="Giá trị Gamma")

        # Thêm nhãn và tiêu đề
        ax2.set_title(f"Bản đồ Gamma (Lát {z})")

    # Thêm tiêu đề chung nếu được cung cấp
    if title and fig:
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    elif fig:
        fig.tight_layout()

    # Lưu biểu đồ nếu output_file được cung cấp
    if output_file and fig:
        try:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Đã lưu biểu đồ phân tích gamma vào {output_file}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu biểu đồ: {e}")

    # Tính các thống kê
    stats = get_gamma_statistics(gamma, mask)

    # Thêm thông tin về biểu đồ
    if fig:
        stats["figure"] = fig

    return stats
