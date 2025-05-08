#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phân tích gamma để đánh giá sự khác biệt giữa các phân bố liều.

Phân tích gamma là một phương pháp so sánh số lượng các phân bố liều 3D,
đo đồng thời sự khác biệt về liều lượng và không gian.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from scipy.ndimage import map_coordinates, zoom
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


class GammaAnalysis:
    """
    Lớp thực hiện phân tích gamma giữa hai phân bố liều.
    """

    def __init__(
        self,
        reference_dose: np.ndarray,
        evaluation_dose: np.ndarray,
        reference_spacing: Optional[Tuple[float, ...]] = None,
        evaluation_spacing: Optional[Tuple[float, ...]] = None,
    ):
        """
        Khởi tạo đối tượng phân tích gamma.

        Parameters
        ----------
        reference_dose : np.ndarray
            Phân bố liều tham chiếu (từ kế hoạch)
        evaluation_dose : np.ndarray
            Phân bố liều cần đánh giá (đo được)
        reference_spacing : Optional[Tuple[float, ...]], optional
            Khoảng cách voxel cho phân bố tham chiếu (mm), mặc định là None (1mm each)
        evaluation_spacing : Optional[Tuple[float, ...]], optional
            Khoảng cách voxel cho phân bố cần đánh giá (mm), mặc định là None (1mm each)
        """
        self.reference_dose = reference_dose
        self.evaluation_dose = evaluation_dose

        # Xác định số chiều
        self.ndim = reference_dose.ndim

        # Thiết lập khoảng cách voxel mặc định nếu không được cung cấp
        if reference_spacing is None:
            self.reference_spacing = tuple([1.0] * self.ndim)
        else:
            self.reference_spacing = reference_spacing

        if evaluation_spacing is None:
            self.evaluation_spacing = tuple([1.0] * self.ndim)
        else:
            self.evaluation_spacing = evaluation_spacing

        # Chuẩn hóa liều thành phần trăm của liều tối đa
        self.max_ref_dose = np.max(reference_dose)
        self.normalized_reference = reference_dose / self.max_ref_dose * 100.0

        self.max_eval_dose = np.max(evaluation_dose)
        self.normalized_evaluation = evaluation_dose / self.max_eval_dose * 100.0

        # Kiểm tra kích thước phân bố
        if self.reference_dose.shape != self.evaluation_dose.shape:
            logger.warning(
                f"Phân bố liều có kích thước khác nhau: {self.reference_dose.shape} vs {self.evaluation_dose.shape}. "
                "Sẽ thực hiện nội suy."
            )
            self._interpolate_doses()

        # Kết quả phân tích
        self.gamma_map = None
        self.gamma_values = []
        self.pass_rate = 0.0
        self.max_gamma = 0.0

    def _interpolate_doses(self):
        """
        Nội suy phân bố liều cần đánh giá để khớp với kích thước của phân bố tham chiếu.
        """
        try:
            # Xác định tỉ lệ kích thước giữa hai phân bố
            scale_factors = [
                self.reference_dose.shape[i] / self.evaluation_dose.shape[i]
                for i in range(self.ndim)
            ]

            # Nội suy phân bố cần đánh giá
            self.evaluation_dose = zoom(self.evaluation_dose, scale_factors, order=3)
            self.normalized_evaluation = (
                self.evaluation_dose / self.max_eval_dose * 100.0
            )

            # Cập nhật khoảng cách voxel sau khi nội suy
            self.evaluation_spacing = tuple(
                [
                    self.evaluation_spacing[i] / scale_factors[i]
                    for i in range(self.ndim)
                ]
            )

            logger.info(
                f"Đã nội suy phân bố liều cần đánh giá để khớp với kích thước: {self.reference_dose.shape}"
            )
        except Exception as e:
            logger.error(f"Lỗi khi nội suy phân bố liều: {str(e)}")

    def calculate(
        self,
        dose_percent: float = 3.0,
        distance_mm: float = 3.0,
        threshold: float = 10.0,
        max_gamma: float = 3.0,
        registration: str = "rigid",
        use_gpu: bool = False,
        num_threads: int = 4,
    ) -> Dict[str, Any]:
        """
        Tính toán phân tích gamma.

        Parameters
        ----------
        dose_percent : float, optional
            Tiêu chí phần trăm liều (%), mặc định là 3.0
        distance_mm : float, optional
            Tiêu chí khoảng cách (mm), mặc định là 3.0
        threshold : float, optional
            Ngưỡng liều tối thiểu (% liều tối đa) để xem xét trong phân tích, mặc định là 10.0
        max_gamma : float, optional
            Giá trị gamma tối đa để tính toán, mặc định là 3.0
        registration : str, optional
            Phương pháp đăng ký hình ảnh ("none", "rigid", "deform"), mặc định là "rigid"
        use_gpu : bool, optional
            Sử dụng GPU nếu có, mặc định là False
        num_threads : int, optional
            Số luồng sử dụng khi tính toán trên CPU, mặc định là 4

        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích gamma
        """
        start_time = time.time()

        # Đăng ký hình ảnh nếu cần
        if registration != "none":
            self._register_doses(method=registration)

        # Tính toán gamma
        if use_gpu:
            try:
                self._calculate_gamma_gpu(
                    dose_percent, distance_mm, threshold, max_gamma
                )
            except Exception as e:
                logger.error(
                    f"Lỗi khi tính toán gamma trên GPU: {str(e)}. Chuyển sang CPU."
                )
                self._calculate_gamma_cpu(
                    dose_percent, distance_mm, threshold, max_gamma, num_threads
                )
        else:
            self._calculate_gamma_cpu(
                dose_percent, distance_mm, threshold, max_gamma, num_threads
            )

        # Tính tỉ lệ điểm pass
        mask = self.normalized_reference >= threshold
        if np.sum(mask) > 0:
            valid_gamma = self.gamma_map[mask]
            pass_points = np.sum(valid_gamma <= 1.0)
            total_points = len(valid_gamma)
            self.pass_rate = (pass_points / total_points) * 100.0
            self.gamma_values = valid_gamma.ravel().tolist()
            self.max_gamma = np.max(valid_gamma)
        else:
            logger.warning("Không có điểm nào vượt qua ngưỡng liều tối thiểu.")
            self.pass_rate = 0.0

        # Tạo kết quả
        results = {
            "pass_rate": self.pass_rate,
            "max_gamma": self.max_gamma,
            "gamma_map": self.gamma_map,
            "gamma_values": self.gamma_values,
            "dose_percent_criteria": dose_percent,
            "distance_mm_criteria": distance_mm,
            "threshold": threshold,
            "computation_time": time.time() - start_time,
        }

        return results

    def _register_doses(self, method: str = "rigid"):
        """
        Đăng ký hình ảnh giữa phân bố tham chiếu và cần đánh giá.

        Parameters
        ----------
        method : str, optional
            Phương pháp đăng ký ("rigid" hoặc "deform"), mặc định là "rigid"
        """
        if method == "rigid":
            # Đăng ký cứng (rigid) sử dụng tương quan chéo
            if self.ndim == 2:
                from scipy.signal import correlate2d

                # Tính tương quan chéo
                corr = correlate2d(
                    self.normalized_reference, self.normalized_evaluation, mode="same"
                )

                # Tìm vị trí tương quan tối đa
                max_idx = np.unravel_index(np.argmax(corr), corr.shape)

                # Tính độ dịch chuyển
                shift = [
                    max_idx[i] - self.normalized_reference.shape[i] // 2
                    for i in range(2)
                ]

                # Dịch chuyển phân bố cần đánh giá
                from scipy.ndimage import shift as shift_image

                self.normalized_evaluation = shift_image(
                    self.normalized_evaluation, shift
                )

            elif self.ndim == 3:
                from scipy.ndimage import correlate

                # Tính tương quan chéo
                corr = correlate(
                    self.normalized_reference,
                    self.normalized_evaluation,
                    mode="constant",
                )

                # Tìm vị trí tương quan tối đa
                max_idx = np.unravel_index(np.argmax(corr), corr.shape)

                # Tính độ dịch chuyển
                shift = [
                    max_idx[i] - self.normalized_reference.shape[i] // 2
                    for i in range(3)
                ]

                # Dịch chuyển phân bố cần đánh giá
                from scipy.ndimage import shift as shift_image

                self.normalized_evaluation = shift_image(
                    self.normalized_evaluation, shift
                )

        elif method == "deform":
            # Đăng ký biến dạng (deformable)
            try:
                from skimage.registration import optical_flow_tvl1

                if self.ndim == 2:
                    # Tính optical flow
                    v, u = optical_flow_tvl1(
                        self.normalized_reference, self.normalized_evaluation
                    )

                    # Tạo lưới tọa độ
                    ny, nx = self.normalized_reference.shape
                    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

                    # Áp dụng biến dạng
                    warped = map_coordinates(
                        self.normalized_evaluation, [y + v, x + u], order=3
                    )
                    self.normalized_evaluation = warped

                elif self.ndim == 3:
                    # Optical flow 3D chưa được hỗ trợ trực tiếp, xử lý từng lát cắt
                    warped = np.zeros_like(self.normalized_evaluation)

                    for z in range(self.normalized_reference.shape[0]):
                        v, u = optical_flow_tvl1(
                            self.normalized_reference[z], self.normalized_evaluation[z]
                        )

                        # Tạo lưới tọa độ
                        ny, nx = self.normalized_reference[z].shape
                        y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

                        # Áp dụng biến dạng
                        warped[z] = map_coordinates(
                            self.normalized_evaluation[z], [y + v, x + u], order=3
                        )

                    self.normalized_evaluation = warped

            except ImportError:
                logger.warning(
                    "scikit-image không được cài đặt. Không thể thực hiện đăng ký biến dạng."
                )
                # Sử dụng đăng ký cứng thay thế
                self._register_doses(method="rigid")

        else:
            logger.warning(
                f"Phương pháp đăng ký '{method}' không được hỗ trợ. Sử dụng phân bố không đăng ký."
            )

    def _calculate_gamma_cpu(
        self,
        dose_percent: float,
        distance_mm: float,
        threshold: float,
        max_gamma: float,
        num_threads: int,
    ):
        """
        Tính toán phân tích gamma trên CPU.

        Parameters
        ----------
        dose_percent : float
            Tiêu chí phần trăm liều (%)
        distance_mm : float
            Tiêu chí khoảng cách (mm)
        threshold : float
            Ngưỡng liều tối thiểu (% liều tối đa)
        max_gamma : float
            Giá trị gamma tối đa để tính toán
        num_threads : int
            Số luồng sử dụng khi tính toán
        """
        # Khởi tạo mảng gamma
        self.gamma_map = np.ones_like(self.reference_dose) * np.inf

        # Tạo mask cho vùng liều đáng quan tâm
        mask = self.normalized_reference >= threshold

        # Tìm các vị trí cần tính gamma
        indices = np.array(np.where(mask)).T

        # Tính khoảng cách voxel thực tế theo mm
        voxel_sizes = np.array(self.reference_spacing)

        # Xác định hằng số cho công thức gamma
        dose_criteria = dose_percent / 100.0  # Chuyển đổi từ % sang phân số

        # Xác định bán kính tìm kiếm trong đơn vị voxel
        search_radius_voxel = np.ceil(distance_mm / np.min(voxel_sizes)).astype(int)

        # Thực hiện tính toán theo từng vùng để giảm bộ nhớ
        batch_size = max(1, len(indices) // num_threads)

        if self.ndim == 2:
            # Tính toán gamma cho phân bố 2D
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]

                for idx in batch_indices:
                    r_idx = tuple(idx)
                    ref_dose = self.normalized_reference[r_idx]

                    # Xác định vùng tìm kiếm
                    min_row = max(0, idx[0] - search_radius_voxel)
                    max_row = min(
                        self.normalized_reference.shape[0],
                        idx[0] + search_radius_voxel + 1,
                    )
                    min_col = max(0, idx[1] - search_radius_voxel)
                    max_col = min(
                        self.normalized_reference.shape[1],
                        idx[1] + search_radius_voxel + 1,
                    )

                    # Tính gamma cho từng điểm trong vùng tìm kiếm
                    min_gamma = np.inf

                    for r in range(min_row, max_row):
                        for c in range(min_col, max_col):
                            # Tính khoảng cách không gian
                            spatial_dist = np.sqrt(
                                sum(
                                    ((idx[i] - np.array([r, c])[i]) * voxel_sizes[i])
                                    ** 2
                                    for i in range(2)
                                )
                            )

                            # Tính khoảng cách liều
                            dose_dist = abs(ref_dose - self.normalized_evaluation[r, c])

                            # Tính giá trị gamma
                            gamma = (
                                np.sqrt(
                                    (spatial_dist / distance_mm) ** 2
                                    + (dose_dist / (ref_dose * dose_criteria)) ** 2
                                )
                                if ref_dose > 0
                                else np.inf
                            )

                            # Cập nhật gamma tối thiểu
                            if gamma < min_gamma:
                                min_gamma = gamma

                            # Dừng sớm nếu đã tìm thấy gamma < 1
                            if min_gamma <= 1.0:
                                break

                        if min_gamma <= 1.0:
                            break

                    # Giới hạn giá trị gamma tối đa
                    min_gamma = min(min_gamma, max_gamma)

                    # Lưu kết quả
                    self.gamma_map[r_idx] = min_gamma

        elif self.ndim == 3:
            # Tính toán gamma cho phân bố 3D
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]

                for idx in batch_indices:
                    r_idx = tuple(idx)
                    ref_dose = self.normalized_reference[r_idx]

                    # Xác định vùng tìm kiếm
                    min_values = [
                        max(0, idx[i] - search_radius_voxel) for i in range(3)
                    ]
                    max_values = [
                        min(
                            self.normalized_reference.shape[i],
                            idx[i] + search_radius_voxel + 1,
                        )
                        for i in range(3)
                    ]

                    # Tính gamma cho từng điểm trong vùng tìm kiếm
                    min_gamma = np.inf

                    for z in range(min_values[0], max_values[0]):
                        for y in range(min_values[1], max_values[1]):
                            for x in range(min_values[2], max_values[2]):
                                # Tính khoảng cách không gian
                                spatial_dist = np.sqrt(
                                    sum(
                                        (
                                            (idx[i] - np.array([z, y, x])[i])
                                            * voxel_sizes[i]
                                        )
                                        ** 2
                                        for i in range(3)
                                    )
                                )

                                # Bỏ qua các điểm quá xa
                                if spatial_dist > distance_mm * max_gamma:
                                    continue

                                # Tính khoảng cách liều
                                dose_dist = abs(
                                    ref_dose - self.normalized_evaluation[z, y, x]
                                )

                                # Tính giá trị gamma
                                gamma = (
                                    np.sqrt(
                                        (spatial_dist / distance_mm) ** 2
                                        + (dose_dist / (ref_dose * dose_criteria)) ** 2
                                    )
                                    if ref_dose > 0
                                    else np.inf
                                )

                                # Cập nhật gamma tối thiểu
                                if gamma < min_gamma:
                                    min_gamma = gamma

                                # Dừng sớm nếu đã tìm thấy gamma <= 1
                                if min_gamma <= 1.0:
                                    break

                            if min_gamma <= 1.0:
                                break

                        if min_gamma <= 1.0:
                            break

                    # Giới hạn giá trị gamma tối đa
                    min_gamma = min(min_gamma, max_gamma)

                    # Lưu kết quả
                    self.gamma_map[r_idx] = min_gamma

    def _calculate_gamma_gpu(
        self,
        dose_percent: float,
        distance_mm: float,
        threshold: float,
        max_gamma: float,
    ):
        """
        Tính toán phân tích gamma trên GPU.

        Parameters
        ----------
        dose_percent : float
            Tiêu chí phần trăm liều (%)
        distance_mm : float
            Tiêu chí khoảng cách (mm)
        threshold : float
            Ngưỡng liều tối thiểu (% liều tối đa)
        max_gamma : float
            Giá trị gamma tối đa để tính toán
        """
        try:
            import cupy as cp

            # Chuyển dữ liệu sang GPU
            ref_gpu = cp.array(self.normalized_reference)
            eval_gpu = cp.array(self.normalized_evaluation)

            # Khởi tạo mảng gamma
            gamma_gpu = cp.ones_like(ref_gpu) * cp.inf

            # Tạo mask cho vùng liều đáng quan tâm
            mask_gpu = ref_gpu >= threshold

            # Xác định hằng số cho công thức gamma
            dose_criteria = dose_percent / 100.0  # Chuyển đổi từ % sang phân số

            # Tạo lưới tọa độ
            if self.ndim == 2:
                y_indices, x_indices = cp.indices(ref_gpu.shape)
                voxel_sizes = cp.array(self.reference_spacing)

                # Xác định bán kính tìm kiếm trong đơn vị voxel
                search_radius_voxel = int(cp.ceil(distance_mm / cp.min(voxel_sizes)))

                # Tạo kernel
                def gamma_kernel(idx_y, idx_x, ref_dose, mask):
                    if not mask[idx_y, idx_x]:
                        return

                    # Vùng tìm kiếm
                    min_y = max(0, idx_y - search_radius_voxel)
                    max_y = min(ref_gpu.shape[0], idx_y + search_radius_voxel + 1)
                    min_x = max(0, idx_x - search_radius_voxel)
                    max_x = min(ref_gpu.shape[1], idx_x + search_radius_voxel + 1)

                    min_gamma = cp.inf

                    # Duyệt vùng tìm kiếm
                    for y in range(min_y, max_y):
                        for x in range(min_x, max_x):
                            # Khoảng cách không gian
                            spatial_dist = cp.sqrt(
                                ((idx_y - y) * voxel_sizes[0]) ** 2
                                + ((idx_x - x) * voxel_sizes[1]) ** 2
                            )

                            # Khoảng cách liều
                            dose_dist = abs(ref_dose - eval_gpu[y, x])

                            # Tính giá trị gamma
                            if ref_dose > 0:
                                gamma = cp.sqrt(
                                    (spatial_dist / distance_mm) ** 2
                                    + (dose_dist / (ref_dose * dose_criteria)) ** 2
                                )

                                if gamma < min_gamma:
                                    min_gamma = gamma

                    # Giới hạn giá trị gamma tối đa
                    gamma_gpu[idx_y, idx_x] = min(min_gamma, max_gamma)

                # Thực thi kernel
                mask_indices = cp.where(mask_gpu)
                n_points = len(mask_indices[0])
                threads_per_block = 256
                blocks_per_grid = (
                    n_points + threads_per_block - 1
                ) // threads_per_block

                gamma_kernel_gpu = cp.RawKernel(
                    r"""
                extern "C" __global__ void gamma_calculation(
                    const float* ref_dose, const float* eval_dose, float* gamma,
                    const bool* mask, const float* voxel_sizes,
                    int width, int height, float distance_mm, float dose_criteria, float max_gamma,
                    int search_radius, int n_points, const int* y_indices, const int* x_indices)
                {
                    int tid = blockIdx.x * blockDim.x + threadIdx.x;
                    if (tid >= n_points) return;

                    int idx_y = y_indices[tid];
                    int idx_x = x_indices[tid];

                    float ref_value = ref_dose[idx_y * width + idx_x];

                    if (!mask[idx_y * width + idx_x]) return;

                    float min_gamma = INFINITY;

                    int min_y = max(0, idx_y - search_radius);
                    int max_y = min(height, idx_y + search_radius + 1);
                    int min_x = max(0, idx_x - search_radius);
                    int max_x = min(width, idx_x + search_radius + 1);

                    for (int y = min_y; y < max_y; y++) {
                        for (int x = min_x; x < max_x; x++) {
                            float spatial_dist = sqrt(
                                pow((idx_y - y) * voxel_sizes[0], 2) +
                                pow((idx_x - x) * voxel_sizes[1], 2)
                            );

                            float dose_dist = abs(ref_value - eval_dose[y * width + x]);

                            if (ref_value > 0) {
                                float g = sqrt(
                                    pow(spatial_dist/distance_mm, 2) +
                                    pow(dose_dist/(ref_value*dose_criteria), 2)
                                );

                                if (g < min_gamma) {
                                    min_gamma = g;
                                    if (min_gamma <= 1.0f) break;
                                }
                            }
                        }
                        if (min_gamma <= 1.0f) break;
                    }

                    gamma[idx_y * width + idx_x] = min(min_gamma, max_gamma);
                }
                """,
                    "gamma_calculation",
                )

                # Chuyển đổi các mảng để truyền vào kernel
                ref_flat = ref_gpu.ravel()
                eval_flat = eval_gpu.ravel()
                gamma_flat = gamma_gpu.ravel()
                mask_flat = mask_gpu.ravel()
                y_indices_flat = cp.array([i for i in mask_indices[0]])
                x_indices_flat = cp.array([i for i in mask_indices[1]])

                # Gọi kernel
                gamma_kernel_gpu(
                    (blocks_per_grid,),
                    (threads_per_block,),
                    (
                        ref_flat,
                        eval_flat,
                        gamma_flat,
                        mask_flat,
                        cp.array(self.reference_spacing),
                        ref_gpu.shape[1],
                        ref_gpu.shape[0],
                        distance_mm,
                        dose_criteria,
                        max_gamma,
                        search_radius_voxel,
                        n_points,
                        y_indices_flat,
                        x_indices_flat,
                    ),
                )

                # Chuyển kết quả về CPU
                self.gamma_map = cp.asnumpy(gamma_gpu)

            elif self.ndim == 3:
                # Tương tự cho 3D nhưng phức tạp hơn
                # Quá trình tính toán 3D trên GPU cũng tương tự, thêm một chiều z
                # Code tương tự như trên, nhưng phức tạp hơn

                # Sử dụng CPU thay thế cho 3D
                logger.warning(
                    "Tính toán gamma 3D trên GPU chưa được triển khai đầy đủ. Chuyển sang CPU."
                )
                self._calculate_gamma_cpu(
                    dose_percent, distance_mm, threshold, max_gamma, 4
                )

        except ImportError:
            logger.warning(
                "CuPy không được cài đặt. Không thể sử dụng GPU. Chuyển sang CPU."
            )
            self._calculate_gamma_cpu(
                dose_percent, distance_mm, threshold, max_gamma, 4
            )

    def plot_gamma_map(self, slice_idx: Optional[int] = None, ax=None):
        """
        Vẽ biểu đồ phân bố gamma.

        Parameters
        ----------
        slice_idx : Optional[int], optional
            Chỉ số lát cắt để hiển thị (chỉ dùng cho phân bố 3D), mặc định là None (lát cắt giữa)
        ax : optional
            Trục matplotlib để vẽ, mặc định là None (tạo mới)

        Returns
        -------
        matplotlib.figure.Figure
            Đối tượng Figure chứa biểu đồ
        """
        if self.gamma_map is None:
            logger.error(
                "Chưa thực hiện phân tích gamma. Hãy gọi phương thức calculate() trước."
            )
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        # Hiển thị phân bố gamma
        if self.ndim == 3:
            if slice_idx is None:
                slice_idx = self.gamma_map.shape[0] // 2

            im = ax.imshow(
                self.gamma_map[slice_idx], cmap="jet", interpolation="nearest"
            )
            ax.set_title(
                f"Phân bố Gamma - Lát cắt {slice_idx} (Pass rate: {self.pass_rate:.1f}%)"
            )
        else:
            im = ax.imshow(self.gamma_map, cmap="jet", interpolation="nearest")
            ax.set_title(f"Phân bố Gamma (Pass rate: {self.pass_rate:.1f}%)")

        # Thêm colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Gamma value")

        # Thêm thông tin bổ sung
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        return fig

    def plot_gamma_histogram(self, ax=None):
        """
        Vẽ biểu đồ histogram của các giá trị gamma.

        Parameters
        ----------
        ax : optional
            Trục matplotlib để vẽ, mặc định là None (tạo mới)

        Returns
        -------
        matplotlib.figure.Figure
            Đối tượng Figure chứa biểu đồ
        """
        if not self.gamma_values:
            logger.error(
                "Chưa thực hiện phân tích gamma. Hãy gọi phương thức calculate() trước."
            )
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        # Vẽ histogram
        ax.hist(self.gamma_values, bins=50, alpha=0.7, color="blue")
        ax.axvline(
            x=1.0, color="red", linestyle="--", label="Pass/Fail Threshold (Gamma=1.0)"
        )

        # Thiết lập biểu đồ
        ax.set_title(f"Histogram Gamma (Pass rate: {self.pass_rate:.1f}%)")
        ax.set_xlabel("Gamma Value")
        ax.set_ylabel("Frequency")
        ax.legend()

        return fig
