#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module phân tích gamma (Gamma Analysis) cho so sánh phân phối liều.

Module này cung cấp các công cụ để thực hiện phân tích gamma 2D và 3D,
một phương pháp định lượng để so sánh hai phân phối liều.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

# Thử nhập các module tăng tốc GPU
try:
    import cupy as cp

    HAS_CUPY = True
    logger.info("CuPy đã được nhập thành công cho phân tích gamma GPU")
except ImportError:
    HAS_CUPY = False
    logger.warning("CuPy không khả dụng. Phân tích gamma GPU sẽ không thể sử dụng.")


def calculate_gamma_3d(
    reference: np.ndarray,
    evaluation: np.ndarray,
    dta_mm: float = 3.0,
    dd_percent: float = 3.0,
    threshold: float = 0.1,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_gamma: float = 5.0,
    local_normalization: bool = False,
    dose_percent: float = None,  # Tham số bổ sung cho tương thích
    distance_mm: float = None,  # Tham số bổ sung cho tương thích
) -> np.ndarray:
    """
    Tính chỉ số gamma 3D giữa hai phân phối liều.

    Chỉ số gamma là công cụ đánh giá định lượng sự khác biệt giữa hai phân phối liều,
    kết hợp cả tiêu chí sai khác khoảng cách không gian (DTA) và sai khác liều (DD).

        Parameters
        ----------
    reference : np.ndarray
        Mảng 3D phân phối liều tham chiếu
    evaluation : np.ndarray
        Mảng 3D phân phối liều cần đánh giá
    dta_mm : float, optional
        Tiêu chí khoảng cách đến sự tương đồng, tính bằng mm, mặc định là 3.0
    dd_percent : float, optional
        Tiêu chí sai khác liều, tính bằng phần trăm liều tối đa, mặc định là 3.0
    threshold : float, optional
        Ngưỡng liều tương đối (so với max) để tính gamma, mặc định là 0.1 (10%)
    voxel_size : Tuple[float, float, float], optional
        Kích thước voxel theo mm, mặc định là (1.0, 1.0, 1.0)
    max_gamma : float, optional
        Giá trị gamma tối đa, các giá trị cao hơn sẽ được gán bằng giá trị này
    local_normalization : bool, optional
        Nếu True, sử dụng chuẩn hóa cục bộ (liều tham chiếu tại mỗi điểm)
    dose_percent : float, optional
        Tham số thay thế cho dd_percent để đảm bảo tính tương thích
    distance_mm : float, optional
        Tham số thay thế cho dta_mm để đảm bảo tính tương thích

    Returns
    -------
    np.ndarray
        Mảng 3D chứa giá trị gamma tại mỗi voxel
    """
    # Xử lý tham số thay thế để đảm bảo tương thích ngược
    if dose_percent is not None:
        logger.info(
            f"Sử dụng tham số dose_percent ({dose_percent}%) thay thế cho dd_percent"
        )
        dd_percent = dose_percent

    if distance_mm is not None:
        logger.info(
            f"Sử dụng tham số distance_mm ({distance_mm}mm) thay thế cho dta_mm"
        )
        dta_mm = distance_mm

    # Kiểm tra dữ liệu đầu vào
    if reference.shape != evaluation.shape:
        raise ValueError(
            f"Kích thước mảng không khớp: reference {reference.shape}, evaluation {evaluation.shape}"
        )

    # Tạo mask dựa trên ngưỡng
    ref_max = np.max(reference)
    mask = reference >= (threshold * ref_max)

    # Không tính gamma cho các vùng dưới ngưỡng
    if np.sum(mask) == 0:
        logger.warning("Không có voxel nào vượt ngưỡng để tính gamma")
        return np.ones_like(reference) * np.inf

    start_time = time.time()

    # Chọn phương thức tính toán (GPU hoặc CPU)
    if HAS_CUPY and reference.size > 1000000:  # Chỉ sử dụng GPU cho dữ liệu lớn
        try:
            gamma = _calculate_gamma_3d_gpu(
                reference,
                evaluation,
                mask,
                dd_percent,
                dta_mm,
                voxel_size,
                max_gamma,
                local_normalization,
            )
        except Exception as e:
            logger.warning(f"Lỗi khi tính gamma trên GPU: {e}, chuyển sang CPU")
            gamma = _calculate_gamma_3d_cpu(
                reference,
                evaluation,
                mask,
                dd_percent,
                dta_mm,
                voxel_size,
                max_gamma,
                local_normalization,
            )
        else:
            gamma = _calculate_gamma_3d_cpu(
                reference,
                evaluation,
                mask,
                dd_percent,
                dta_mm,
                voxel_size,
                max_gamma,
                local_normalization,
            )

    elapsed_time = time.time() - start_time
    logger.info(f"Hoàn thành tính gamma 3D trong {elapsed_time:.2f} giây")

    return gamma


def _calculate_gamma_3d_cpu(
    reference: np.ndarray,
    evaluation: np.ndarray,
    mask: np.ndarray,
    dd: float,
    dta_mm: float,
    voxel_size: Tuple[float, float, float],
    max_gamma: float,
    local_normalization: bool,
) -> np.ndarray:
    """Phiên bản CPU của phân tích gamma 3D."""
    logger.info("Thực hiện phân tích gamma trên CPU")

    gamma = np.ones_like(reference) * np.inf
    shape = reference.shape

    # Tìm phạm vi tìm kiếm tối đa theo voxel
    search_range = [int(np.ceil(dta_mm / vs)) for vs in voxel_size]

    # Tính toán gamma cho mỗi voxel trong mask
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if not mask[i, j, k]:
                    continue

                ref_dose = reference[i, j, k]

                # Xác định giới hạn tìm kiếm cục bộ
                i_min = max(0, i - search_range[0])
                i_max = min(shape[0] - 1, i + search_range[0])
                j_min = max(0, j - search_range[1])
                j_max = min(shape[1] - 1, j + search_range[1])
                k_min = max(0, k - search_range[2])
                k_max = min(shape[2] - 1, k + search_range[2])

                # Tìm giá trị gamma nhỏ nhất trong vùng tìm kiếm
                min_gamma = np.inf

                for ni in range(i_min, i_max + 1):
                    for nj in range(j_min, j_max + 1):
                        for nk in range(k_min, k_max + 1):
                            # Tính khoảng cách không gian
                            dist_sq = (
                                ((i - ni) * voxel_size[0]) ** 2
                                + ((j - nj) * voxel_size[1]) ** 2
                                + ((k - nk) * voxel_size[2]) ** 2
                            )

                            # Tính sai khác liều
                            eval_dose = evaluation[ni, nj, nk]

                            if local_normalization:
                                dose_diff = (
                                    abs(ref_dose - eval_dose) / ref_dose
                                    if ref_dose > 0
                                    else 0
                                )
                                dose_diff = dose_diff * 100  # Chuyển sang phần trăm
                            else:
                                dose_diff = abs(ref_dose - eval_dose)

                            # Tính chỉ số gamma
                            gamma_sq = dist_sq / (dta_mm**2) + (dose_diff / dd) ** 2

                            if gamma_sq < min_gamma:
                                min_gamma = gamma_sq

                # Lưu giá trị gamma
                gamma[i, j, k] = min(np.sqrt(min_gamma), max_gamma)

    return gamma


def _calculate_gamma_3d_gpu(
    reference: np.ndarray,
    evaluation: np.ndarray,
    mask: np.ndarray,
    dd: float,
    dta_mm: float,
    voxel_size: Tuple[float, float, float],
    max_gamma: float,
    local_normalization: bool,
) -> np.ndarray:
    """Phiên bản GPU của phân tích gamma 3D sử dụng CuPy."""
    logger.info("Thực hiện phân tích gamma trên GPU với CuPy")

    # Chuyển dữ liệu lên GPU
    reference_gpu = cp.asarray(reference)
    evaluation_gpu = cp.asarray(evaluation)
    mask_gpu = cp.asarray(mask)

    # Kích thước dữ liệu
    shape = reference.shape

    # Tạo mảng kết quả
    gamma_gpu = cp.ones_like(reference_gpu) * cp.inf

    # Tìm phạm vi tìm kiếm tối đa theo voxel
    search_range = [int(np.ceil(dta_mm / vs)) for vs in voxel_size]

    # Kernel code cho CuPy
    kernel_code = """
    extern "C" __global__ void calculate_gamma3d(
        const float* reference, const float* evaluation, const bool* mask,
        float* gamma, const int nx, const int ny, const int nz,
        const float dd, const float dta_mm, const float dx, const float dy, const float dz,
        const float max_gamma, const int search_x, const int search_y, const int search_z,
        const bool local_normalization)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= nx || y >= ny || z >= nz) return;

        int idx = x + y*nx + z*nx*ny;

        if (!mask[idx]) return;

        float ref_dose = reference[idx];
        float min_gamma_sq = 1.0e10f;

        for (int i = max(0, x - search_x); i <= min(nx-1, x + search_x); i++) {
            for (int j = max(0, y - search_y); j <= min(ny-1, y + search_y); j++) {
                for (int k = max(0, z - search_z); k <= min(nz-1, z + search_z); k++) {
                    int idx2 = i + j*nx + k*nx*ny;

                    // Khoảng cách không gian bình phương
                    float dist_sq = powf((x-i)*dx, 2) + powf((y-j)*dy, 2) + powf((z-k)*dz, 2);

                    // Sai khác liều
                    float eval_dose = evaluation[idx2];
                    float dose_diff;

                    if (local_normalization) {
                        dose_diff = ref_dose > 0 ? fabsf(ref_dose - eval_dose) / ref_dose * 100 : 0;
                    } else {
                        dose_diff = fabsf(ref_dose - eval_dose);
                    }

                    // Chỉ số gamma
                    float gamma_sq = dist_sq / (dta_mm * dta_mm) + (dose_diff * dose_diff) / (dd * dd);

                    if (gamma_sq < min_gamma_sq) {
                        min_gamma_sq = gamma_sq;
                    }
                }
            }
        }

        gamma[idx] = fminf(sqrtf(min_gamma_sq), max_gamma);
    }
    """

    # Biên dịch kernel
    try:
        calculate_gamma3d = cp.RawKernel(kernel_code, "calculate_gamma3d")

        # Cấu hình block size
        block_size = (8, 8, 8)
        grid_size = (
            (shape[0] + block_size[0] - 1) // block_size[0],
            (shape[1] + block_size[1] - 1) // block_size[1],
            (shape[2] + block_size[2] - 1) // block_size[2],
        )

        # Thực thi kernel
        calculate_gamma3d(
            grid_size,
            block_size,
            (
                reference_gpu,
                evaluation_gpu,
                mask_gpu,
                gamma_gpu,
                shape[0],
                shape[1],
                shape[2],
                float(dd),
                float(dta_mm),
                float(voxel_size[0]),
                float(voxel_size[1]),
                float(voxel_size[2]),
                float(max_gamma),
                int(search_range[0]),
                int(search_range[1]),
                int(search_range[2]),
                local_normalization,
            ),
        )

        # Chuyển kết quả về CPU
        gamma = cp.asnumpy(gamma_gpu)

        # Giải phóng bộ nhớ GPU
        del reference_gpu, evaluation_gpu, mask_gpu, gamma_gpu
        cp.get_default_memory_pool().free_all_blocks()

    except Exception as e:
        logger.error(f"Lỗi khi thực thi kernel GPU: {str(e)}")
        # Fallback về CPU
        gamma = _calculate_gamma_3d_cpu(
            reference,
            evaluation,
            mask,
            dd,
            dta_mm,
            voxel_size,
            max_gamma,
            local_normalization,
        )

    return gamma


def calculate_gamma_2d(
    reference_2d: np.ndarray,
    evaluation_2d: np.ndarray,
    dta_mm: float = 3.0,
    dd_percent: float = 3.0,
    threshold: float = 0.1,
    pixel_size: Tuple[float, float] = (1.0, 1.0),
    max_gamma: float = 5.0,
    local_normalization: bool = False,
) -> np.ndarray:
    """
    Tính chỉ số gamma 2D giữa hai phân phối liều 2D.

    Parameters tương tự với phiên bản 3D, nhưng cho dữ liệu 2D.
    """
    # Kiểm tra kích thước mảng đầu vào
    if reference_2d.shape != evaluation_2d.shape:
        raise ValueError(
            f"Kích thước mảng không khớp: reference {reference_2d.shape}, evaluation {evaluation_2d.shape}"
        )

    # Chuyển đổi thành mảng 3D với chiều thứ 3 là 1
    reference_3d = np.expand_dims(reference_2d, axis=2)
    evaluation_3d = np.expand_dims(evaluation_2d, axis=2)
    voxel_size = (pixel_size[0], pixel_size[1], 1.0)

    # Gọi hàm gamma 3D
    gamma_3d = calculate_gamma_3d(
        reference_3d,
        evaluation_3d,
        dta_mm,
        dd_percent,
        threshold,
        voxel_size,
        max_gamma,
        local_normalization,
    )

    # Trả về kết quả 2D
    return gamma_3d[:, :, 0]


def gamma_pass_rate(
    gamma: np.ndarray, mask: Optional[np.ndarray] = None, pass_criteria: float = 1.0
) -> float:
    """
    Tính tỉ lệ đạt tiêu chí gamma.

    Parameters
    ----------
    gamma : np.ndarray
        Mảng giá trị gamma
    mask : np.ndarray, optional
        Mặt nạ chỉ ra các vùng cần tính pass rate, mặc định là None (tất cả các điểm)
    pass_criteria : float, optional
        Ngưỡng để xem là đạt, mặc định là 1.0

    Returns
    -------
    float
        Tỉ lệ phần trăm điểm đạt tiêu chí
    """
    if mask is None:
        mask = np.ones_like(gamma, dtype=bool)

    num_points = np.sum(mask)
    if num_points == 0:
        return 0.0

    num_pass = np.sum((gamma <= pass_criteria) & mask)
    return (num_pass / num_points) * 100.0


def plot_gamma_results(
    gamma: np.ndarray,
    mask: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    figure_size: Tuple[int, int] = (10, 8),
    colormap: str = "RdYlGn_r",
) -> Any:
    """
    Tạo hình ảnh của kết quả phân tích gamma.

        Parameters
        ----------
    gamma : np.ndarray
        Mảng giá trị gamma
    mask : np.ndarray, optional
        Mặt nạ các vùng quan tâm, mặc định là None
    slice_idx : int, optional
        Chỉ số lát cắt muốn hiển thị, mặc định là None (trung tâm)
    axis : int, optional
        Trục cho lát cắt (0, 1 hoặc 2), mặc định là 2 (trục z)
    figure_size : Tuple[int, int], optional
        Kích thước hình, mặc định là (10, 8)
    colormap : str, optional
        Bảng màu, mặc định là 'RdYlGn_r' (đỏ = không đạt, xanh = đạt)

        Returns
        -------
    Any
        Đồ thị matplotlib được tạo
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        # Chọn lát cắt hiển thị
        if slice_idx is None:
            slice_idx = gamma.shape[axis] // 2

        # Chọn lát cắt
        if axis == 0:
            gamma_slice = gamma[slice_idx, :, :]
            mask_slice = mask[slice_idx, :, :] if mask is not None else None
        elif axis == 1:
            gamma_slice = gamma[:, slice_idx, :]
            mask_slice = mask[:, slice_idx, :] if mask is not None else None
        else:  # axis == 2
            gamma_slice = gamma[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx] if mask is not None else None

        # Tạo hình
        fig, ax = plt.subplots(figsize=figure_size)

        # Tạo bảng màu tùy chỉnh nếu cần
        try:
            cmap = plt.get_cmap(colormap)
        except ValueError:
            # Bảng màu mặc định nếu không tìm thấy
            green_to_red = LinearSegmentedColormap.from_list(
                "GreenToRed", [(0, 0.7, 0), (1.0, 1.0, 0), (1.0, 0, 0)]
            )
            cmap = green_to_red

        # Áp dụng mask nếu có
        if mask_slice is not None:
            masked_gamma = np.copy(gamma_slice)
            masked_gamma[~mask_slice] = np.nan
            img = ax.imshow(masked_gamma, cmap=cmap, vmin=0, vmax=2.0)
        else:
            img = ax.imshow(gamma_slice, cmap=cmap, vmin=0, vmax=2.0)

        # Thêm thanh màu
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label("Gamma Index")

        # Tính pass rate
        if mask_slice is not None:
            pass_rate = gamma_pass_rate(gamma_slice, mask_slice)
        else:
            pass_rate = gamma_pass_rate(gamma_slice)

        ax.set_title(f"Gamma Analysis - Pass Rate: {pass_rate:.2f}%")

        return fig
    except ImportError:
        logger.warning("Matplotlib không khả dụng. Không thể tạo biểu đồ.")
        return None


def compare_dose_distributions(
    reference: np.ndarray,
    evaluation: np.ndarray,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold: float = 0.1,
    gamma_criteria: List[Tuple[float, float]] = [(3.0, 3.0), (2.0, 2.0)],
    global_normalization: bool = True,
) -> Dict[str, Any]:
    """
    So sánh hai phân phối liều với nhiều tiêu chí khác nhau.

        Parameters
        ----------
    reference : np.ndarray
        Phân phối liều tham chiếu
    evaluation : np.ndarray
        Phân phối liều cần đánh giá
    voxel_size : Tuple[float, float, float], optional
        Kích thước voxel theo mm, mặc định là (1.0, 1.0, 1.0)
    threshold : float, optional
        Ngưỡng liều tương đối để xem xét, mặc định là 0.1 (10%)
    gamma_criteria : List[Tuple[float, float]], optional
        Danh sách các tiêu chí gamma [DTA (mm), DD (%)], mặc định là [(3,3), (2,2)]
    global_normalization : bool, optional
        Nếu True, sử dụng chuẩn hóa toàn cục, ngược lại sử dụng chuẩn hóa cục bộ

        Returns
        -------
    Dict[str, Any]
        Kết quả so sánh với các chỉ số khác nhau
    """
    result = {
        "shape": reference.shape,
        "voxel_size": voxel_size,
        "max_reference": np.max(reference),
        "max_evaluation": np.max(evaluation),
        "stats": {},
        "gamma": {},
    }

    # Tạo mask từ ngưỡng
    mask = reference >= (threshold * np.max(reference))
    result["num_evaluated_voxels"] = np.sum(mask)

    # Tính các thống kê cơ bản
    diff = evaluation - reference
    abs_diff = np.abs(diff)

    # Chỉ xem xét vùng trong mask
    masked_diff = diff[mask]
    masked_abs_diff = abs_diff[mask]

    result["stats"]["mean_error"] = np.mean(masked_diff)
    result["stats"]["mean_abs_error"] = np.mean(masked_abs_diff)
    result["stats"]["max_error"] = np.max(masked_abs_diff)
    result["stats"]["min_error"] = np.min(masked_diff)
    result["stats"]["rms_error"] = np.sqrt(np.mean(np.square(masked_diff)))

    # Tính gamma cho mỗi tiêu chí
    for dta, dd in gamma_criteria:
        gamma_key = f"{dta}mm_{dd}pct"
        try:
            gamma = calculate_gamma_3d(
                reference,
                evaluation,
                dta_mm=dta,
                dd_percent=dd,
                threshold=threshold,
                voxel_size=voxel_size,
                local_normalization=not global_normalization,
            )

            pass_rate = gamma_pass_rate(gamma, mask)
            result["gamma"][gamma_key] = {
                "pass_rate": pass_rate,
                "mean": np.mean(gamma[mask]),
                "max": np.max(gamma[mask]),
                "median": np.median(gamma[mask]),
                "criteria": f"{dta}mm/{dd}%",
            }
        except Exception as e:
            logger.error(f"Lỗi khi tính gamma {dta}mm/{dd}%: {str(e)}")
            result["gamma"][gamma_key] = {"error": str(e)}

    return result


__all__ = [
    "calculate_gamma_3d",
    "calculate_gamma_2d",
    "gamma_pass_rate",
    "plot_gamma_results",
    "compare_dose_distributions",
]

__version__ = "0.7.8"
