#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module AAA (Anisotropic Analytical Algorithm) cho QuangTPS.

Module này cung cấp thuật toán tính toán liều AAA, một cải tiến của Eclipse TPS,
với các cải tiến về hiệu năng và độ chính xác.
"""

import os
import numpy as np
import SimpleITK as sitk
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from quangtps.dose.dose_engine import (
    DoseCalculationImplementer,
    DoseCalculationAlgorithm,
)
from quangtps.dose.dose_grid import DoseGrid
from quangtps.core.exceptions import ValidationError, AlgorithmError

logger = logging.getLogger(__name__)


class AAADoseCalculation:
    """
    Lớp AAA (Anisotropic Analytical Algorithm)

    Thuật toán tính toán liều tiên tiến với khả năng hiệu chỉnh không đồng nhất,
    triển khai tương tự như thuật toán AAA trong Eclipse của Varian.
    """

    def __init__(self):
        """Khởi tạo thuật toán AAA."""
        self.parameters = {
            "resolution": 2.5,  # Độ phân giải tính toán (mm)
            "scatter_kernel_size": 15,  # Kích thước nhân tích chập (voxel)
            "heterogeneity_correction": True,  # Bật/tắt hiệu chỉnh không đồng nhất
            "use_parallel": True,  # Bật/tắt tính toán song song
            "num_threads": multiprocessing.cpu_count(),  # Số luồng tính toán
            "algorithm_version": "2.0",  # Phiên bản thuật toán
            "energy_conservation": True,  # Bật/tắt bảo toàn năng lượng
            "accuracy_level": 2,  # Mức độ chính xác (1-3)
        }

        # Các thông số vật lý
        self.beam_config = {}
        self.density_table = {}

        # Khởi tạo các nhân tích chập
        self.kernels = {
            "photon": None,
            "electron": None,
            "scatter": None,
        }

        self.status_callback = None

    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Đặt tham số tính toán.

        Parameters:
            parameters (dict): Các tham số tính toán
        """
        for key, value in parameters.items():
            if key in self.parameters:
                self.parameters[key] = value

        # Cập nhật lại thông số vật lý khi thay đổi tham số
        self._initialize_physics()

        logger.info(f"Đã cài đặt tham số cho AAA: {parameters}")

    def _initialize_physics(self):
        """Khởi tạo các thông số vật lý và các nhân tích chập."""
        # Tạo bảng tỷ trọng mô
        # Giá trị HU -> Tỷ trọng khối lượng
        self.density_table = {
            -1000: 0.001,  # Không khí
            -750: 0.25,  # Phổi
            -500: 0.48,
            -250: 0.69,
            0: 1.0,  # Nước
            500: 1.33,  # Xương bọt
            1000: 1.66,  # Xương đặc
            1500: 1.85,
            2000: 2.0,
        }

        # Khởi tạo kernel photon (phân bố Gaussian 3D)
        sigma = self.parameters["scatter_kernel_size"] / 6.0  # 6-sigma rule
        size = self.parameters["scatter_kernel_size"]

        # Tạo lưới 3D
        x = np.linspace(-size / 2, size / 2, size)
        y = np.linspace(-size / 2, size / 2, size)
        z = np.linspace(-size / 2, size / 2, size)

        X, Y, Z = np.meshgrid(x, y, z)

        # Tạo kernel photon (Gaussian 3D)
        photon_kernel = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
        photon_kernel /= np.sum(photon_kernel)  # Chuẩn hóa

        # Tạo kernel electron (Gaussian 3D nhưng với hình dạng kéo dài theo hướng z)
        electron_sigma_xy = sigma * 0.5
        electron_sigma_z = sigma * 2.0
        electron_kernel = np.exp(
            -(X**2 + Y**2) / (2 * electron_sigma_xy**2)
            - Z**2 / (2 * electron_sigma_z**2)
        )
        electron_kernel /= np.sum(electron_kernel)  # Chuẩn hóa

        # Tạo kernel scatter (Gaussian 3D nhưng có bán kính lớn hơn)
        scatter_sigma = sigma * 3.0
        scatter_kernel = np.exp(-(X**2 + Y**2 + Z**2) / (2 * scatter_sigma**2))
        scatter_kernel /= np.sum(scatter_kernel)  # Chuẩn hóa

        # Lưu các kernel
        self.kernels["photon"] = photon_kernel
        self.kernels["electron"] = electron_kernel
        self.kernels["scatter"] = scatter_kernel

        logger.debug("Đã khởi tạo các thông số vật lý và nhân tích chập cho AAA")

    def set_beam_config(self, beam_config: Dict[str, Any]):
        """
        Đặt cấu hình chùm tia.

        Parameters:
            beam_config (dict): Cấu hình chùm tia
        """
        self.beam_config = beam_config
        logger.debug(f"Đã cài đặt cấu hình chùm tia: {beam_config}")

    def set_status_callback(self, callback):
        """
        Đặt callback để cập nhật trạng thái tính toán.

        Parameters:
            callback (callable): Hàm callback nhận thông tin trạng thái
        """
        self.status_callback = callback

    def _update_status(self, progress: float, status: str = None):
        """
        Cập nhật trạng thái tính toán.

        Parameters:
            progress (float): Tiến độ tính toán (0-1)
            status (str, optional): Thông tin trạng thái
        """
        if self.status_callback:
            self.status_callback(progress, status)

    def _hu_to_density(self, hu_value: float) -> float:
        """
        Chuyển đổi từ số HU sang tỷ trọng khối lượng.

        Parameters:
            hu_value (float): Giá trị HU

        Returns:
            float: Tỷ trọng khối lượng (g/cm³)
        """
        # Tìm hai điểm gần nhất trong bảng
        hu_values = sorted(self.density_table.keys())

        # Nếu nhỏ hơn giá trị nhỏ nhất hoặc lớn hơn giá trị lớn nhất
        if hu_value <= hu_values[0]:
            return self.density_table[hu_values[0]]
        if hu_value >= hu_values[-1]:
            return self.density_table[hu_values[-1]]

        # Tìm vị trí nội suy
        idx = 0
        while idx < len(hu_values) - 1 and hu_values[idx + 1] < hu_value:
            idx += 1

        # Nội suy tuyến tính
        x0, x1 = hu_values[idx], hu_values[idx + 1]
        y0, y1 = self.density_table[x0], self.density_table[x1]

        return y0 + (y1 - y0) * (hu_value - x0) / (x1 - x0)

    def _compute_terma(self, phantom: np.ndarray, beam: Dict[str, Any]) -> np.ndarray:
        """
        Tính toán TERMA (Total Energy Released per unit MAss) tối ưu hóa với NumPy.

        Parameters:
            phantom (numpy.ndarray): Ma trận hình ảnh CT (HU)
            beam (dict): Thông tin chùm tia

        Returns:
            numpy.ndarray: Ma trận TERMA
        """
        # Lấy kích thước ma trận
        shape = phantom.shape

        # Khởi tạo ma trận TERMA với giá trị 0
        terma = np.zeros(shape, dtype=np.float32)

        # Lấy thông tin chùm tia
        energy = beam.get("energy", 6.0)  # MV
        sad = beam.get("sad", 1000.0)  # mm (SAD = Source-Axis Distance)
        gantry_angle = beam.get("gantry_angle", 0.0)  # độ
        field_size = beam.get("field_size", (100, 100))  # mm
        fluence = beam.get("fluence", None)  # Ma trận fluence
        isocenter = beam.get("isocenter", None)  # Tọa độ isocenter (mm)
        mu = beam.get("mu", 100.0)  # Monitor Units

        # Nếu không có isocenter, sử dụng tâm của ma trận
        if isocenter is None:
            isocenter = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

        # Nếu fluence là None, tạo fluence đồng nhất
        if fluence is None:
            fluence = np.ones((100, 100), dtype=np.float32)  # Ma trận 100x100

        # Đường kính chùm tia tại isocenter (cm)
        field_width = field_size[0] / 10.0  # mm -> cm
        field_height = field_size[1] / 10.0  # mm -> cm

        # Tính toán tọa độ của mỗi voxel
        # Tạo lưới tọa độ 3D (vector hóa)
        x = np.arange(shape[0])
        y = np.arange(shape[1])
        z = np.arange(shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Tính khoảng cách từ mỗi voxel đến isocenter
        X = X - isocenter[0]
        Y = Y - isocenter[1]
        Z = Z - isocenter[2]

        # Chuyển đổi góc gantry từ độ sang radian
        angle_rad = np.radians(gantry_angle)

        # Xoay tọa độ theo góc gantry
        X_rot = X * np.cos(angle_rad) - Z * np.sin(angle_rad)
        Z_rot = X * np.sin(angle_rad) + Z * np.cos(angle_rad)

        # Tính khoảng cách từ nguồn tới mỗi voxel
        # Source position tương đối với isocenter
        source_x = -sad * np.sin(angle_rad)
        source_z = -sad * np.cos(angle_rad)

        # Khoảng cách từ source đến mỗi voxel (mm)
        distance_to_source = np.sqrt((X - source_x) ** 2 + Y**2 + (Z - source_z) ** 2)

        # Tính toán tọa độ của mỗi voxel trong hệ tọa độ Beam's Eye View (BEV)
        # (tâm chùm tia là gốc)
        field_position_x = X_rot * sad / (sad - Z_rot)
        field_position_y = Y * sad / (sad - Z_rot)

        # Tạo mask xác định voxel nằm trong trường chiếu
        half_width = field_width / 2.0 * 10  # cm -> mm
        half_height = field_height / 2.0 * 10  # cm -> mm
        field_mask = (
            (np.abs(field_position_x) <= half_width)
            & (np.abs(field_position_y) <= half_height)
            & (Z_rot < sad)  # Loại bỏ voxel phía sau nguồn
        )

        # Tính tỷ trọng tại mỗi voxel
        density = np.vectorize(self._hu_to_density)(phantom)

        # Tính hệ số giảm chùm tia (inverse square law)
        inverse_square = np.zeros_like(distance_to_source)
        mask = distance_to_source > 0
        inverse_square[mask] = (sad / distance_to_source[mask]) ** 2

        # Tính PDD theo năng lượng (Percent Depth Dose)
        # Cải tiến: sử dụng bảng PDD theo năng lượng
        pdd = self._calculate_pdd(Z_rot, energy)

        # Tính toán fluence tại mỗi vị trí trường
        fluence_map = self._calculate_fluence_map(
            field_position_x, field_position_y, fluence, field_size, field_mask
        )

        # Tính toán TERMA (vector hóa hoàn toàn)
        # TERMA = fluence * density * μ/ρ * energy
        mu_rho = self._get_mu_rho(energy, phantom)  # Hệ số hấp thụ khối lượng
        terma = fluence_map * inverse_square * pdd * density * mu_rho * mu * energy

        # Chỉ giữ lại giá trị trong trường chiếu
        terma = terma * field_mask

        # Chuẩn hóa TERMA
        if self.parameters["energy_conservation"] and np.sum(terma) > 0:
            norm_factor = mu * energy / np.sum(terma)
            terma *= norm_factor

        return terma

    def _calculate_pdd(self, depth_map: np.ndarray, energy: float) -> np.ndarray:
        """
        Tính toán Percent Depth Dose (PDD) theo năng lượng.

        Parameters:
            depth_map: Bản đồ độ sâu tương đối với chùm tia (mm)
            energy: Năng lượng chùm tia (MV)

        Returns:
            numpy.ndarray: Ma trận PDD
        """
        # Chuyển đổi độ sâu từ mm sang cm
        depth_cm = np.abs(depth_map) / 10.0

        # Tham số PDD theo năng lượng:
        # Mô hình: PDD(d) = A * exp(-a*d) + B * exp(-b*d)
        energy_params = {
            # [A, a, B, b, dmax]
            6.0: [1.0, 0.0438, 0.0, 0.0, 1.5],  # 6 MV
            10.0: [1.0, 0.0356, 0.0, 0.0, 2.5],  # 10 MV
            15.0: [1.0, 0.0315, 0.0, 0.0, 3.0],  # 15 MV
            18.0: [1.0, 0.0287, 0.0, 0.0, 3.2],  # 18 MV
        }

        # Nếu năng lượng không có trong bảng, nội suy tuyến tính
        if energy not in energy_params:
            # Tìm hai năng lượng gần nhất
            energies = sorted(energy_params.keys())
            if energy <= energies[0]:
                params = energy_params[energies[0]]
            elif energy >= energies[-1]:
                params = energy_params[energies[-1]]
            else:
                # Nội suy tuyến tính
                for i in range(len(energies) - 1):
                    if energies[i] <= energy <= energies[i + 1]:
                        e1, e2 = energies[i], energies[i + 1]
                        p1, p2 = energy_params[e1], energy_params[e2]
                        t = (energy - e1) / (e2 - e1)
                        params = [(1 - t) * p1[j] + t * p2[j] for j in range(len(p1))]
                        break
        else:
            params = energy_params[energy]

        # Giải nén tham số
        A, a, B, b, dmax = params

        # Tính PDD
        pdd = np.zeros_like(depth_cm)

        # Build-up region (trước dmax)
        buildup_mask = depth_cm < dmax
        if np.any(buildup_mask):
            # Mô hình build-up region: hàm bậc 2
            pdd[buildup_mask] = (depth_cm[buildup_mask] / dmax) ** 2

        # Vùng sau dmax: suy giảm theo hàm mũ
        falloff_mask = depth_cm >= dmax
        if np.any(falloff_mask):
            pdd[falloff_mask] = A * np.exp(-a * (depth_cm[falloff_mask] - dmax))
            if B > 0:
                pdd[falloff_mask] += B * np.exp(-b * (depth_cm[falloff_mask] - dmax))

        return pdd

    def _calculate_fluence_map(
        self,
        field_x: np.ndarray,
        field_y: np.ndarray,
        fluence: np.ndarray,
        field_size: Tuple[float, float],
        field_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Nội suy fluence từ ma trận fluence đầu vào cho mỗi vị trí trong trường.

        Parameters:
            field_x: Tọa độ x trong hệ BEV (mm)
            field_y: Tọa độ y trong hệ BEV (mm)
            fluence: Ma trận fluence tại mặt phẳng isocenter
            field_size: Kích thước trường (mm, mm)
            field_mask: Mask xác định voxel nằm trong trường chiếu

        Returns:
            numpy.ndarray: Ma trận fluence nội suy tại mỗi vị trí
        """
        # Lấy kích thước ma trận fluence
        if fluence is None:
            # Nếu không có fluence, tạo fluence đồng nhất
            fluence_result = np.ones_like(field_x)
            return fluence_result * field_mask

        fluence_height, fluence_width = fluence.shape

        # Kích thước trường (mm)
        field_width, field_height = field_size

        # Tạo ma trận fluence kết quả
        fluence_result = np.zeros_like(field_x)

        # Nội suy fluence cho mỗi vị trí trong trường chỉ khi nằm trong field_mask
        if np.any(field_mask):
            # Chuẩn hóa tọa độ trường về khoảng [-1, 1]
            norm_x = field_x[field_mask] / (field_width / 2.0)
            norm_y = field_y[field_mask] / (field_height / 2.0)

            # Ánh xạ từ khoảng [-1, 1] sang [0, fluence_size-1]
            fluence_x = (norm_x + 1) / 2.0 * (fluence_width - 1)
            fluence_y = (norm_y + 1) / 2.0 * (fluence_height - 1)

            # Nội suy song tuyến tính
            x0 = np.floor(fluence_x).astype(int)
            y0 = np.floor(fluence_y).astype(int)

            # Đảm bảo chỉ số không vượt quá biên
            x0 = np.clip(x0, 0, fluence_width - 2)
            y0 = np.clip(y0, 0, fluence_height - 2)

            x1 = x0 + 1
            y1 = y0 + 1

            # Tính trọng số nội suy
            wx = fluence_x - x0
            wy = fluence_y - y0

            # Nội suy giá trị
            # f(x,y) = (1-wx)(1-wy)f(x0,y0) + wx(1-wy)f(x1,y0) + (1-wx)wy*f(x0,y1) + wx*wy*f(x1,y1)
            val = (
                (1 - wx) * (1 - wy) * fluence[y0, x0]
                + wx * (1 - wy) * fluence[y0, x1]
                + (1 - wx) * wy * fluence[y1, x0]
                + wx * wy * fluence[y1, x1]
            )

            # Gán giá trị vào ma trận kết quả
            fluence_result[field_mask] = val

        return fluence_result

    def _get_mu_rho(self, energy: float, phantom: np.ndarray) -> np.ndarray:
        """
        Tính hệ số hấp thụ khối lượng (μ/ρ) theo năng lượng và vật liệu.

        Parameters:
            energy: Năng lượng chùm tia (MV)
            phantom: Ma trận hình ảnh CT (HU)

        Returns:
            numpy.ndarray: Ma trận hệ số hấp thụ khối lượng
        """
        # Chuyển đổi HU thành mật độ electron
        density = np.vectorize(self._hu_to_density)(phantom)

        # Hệ số hấp thụ khối lượng của nước theo năng lượng (cm²/g)
        mu_rho_water = {
            6.0: 0.0277,  # 6 MV
            10.0: 0.0231,  # 10 MV
            15.0: 0.0205,  # 15 MV
            18.0: 0.0190,  # 18 MV
        }

        # Nếu năng lượng không có trong bảng, nội suy tuyến tính
        if energy not in mu_rho_water:
            # Tìm hai năng lượng gần nhất
            energies = sorted(mu_rho_water.keys())
            if energy <= energies[0]:
                mu_rho_value = mu_rho_water[energies[0]]
            elif energy >= energies[-1]:
                mu_rho_value = mu_rho_water[energies[-1]]
            else:
                # Nội suy tuyến tính
                for i in range(len(energies) - 1):
                    if energies[i] <= energy <= energies[i + 1]:
                        e1, e2 = energies[i], energies[i + 1]
                        v1, v2 = mu_rho_water[e1], mu_rho_water[e2]
                        t = (energy - e1) / (e2 - e1)
                        mu_rho_value = (1 - t) * v1 + t * v2
                        break
        else:
            mu_rho_value = mu_rho_water[energy]

        # Hiệu chỉnh theo vật liệu: (μ/ρ) ≈ (μ/ρ)water * (ρ/ρwater)
        mu_rho = mu_rho_value * (density / 1.0)

        return mu_rho

    def _convolution(
        self, terma: np.ndarray, kernel: np.ndarray, density: np.ndarray = None
    ) -> np.ndarray:
        """
        Tích chập TERMA với kernel để tính liều.

        Parameters:
            terma: Ma trận TERMA
            kernel: Nhân tích chập
            density: Ma trận tỷ trọng (nếu có)

        Returns:
            numpy.ndarray: Ma trận liều kết quả
        """
        start_time = time.time()
        self._update_status(0.6, "Đang tính tích chập...")

        # Kiểm tra đầu vào
        if terma.size == 0 or kernel.size == 0:
            logger.error("Không thể thực hiện tích chập với ma trận rỗng")
            return np.zeros_like(terma)

        # Triển khai tích chập nhanh với FFT
        # Nếu ma trận lớn, sử dụng FFT sẽ nhanh hơn tích chập trực tiếp

        # Chuẩn bị kernel để tích chập FFT
        # Kernel phải có kích thước bằng hoặc lớn hơn TERMA
        padded_kernel = np.zeros_like(terma)

        # Đặt kernel vào giữa ma trận
        kernel_shape = kernel.shape
        kx, ky, kz = kernel_shape
        sx, sy, sz = terma.shape

        # Tính toán vị trí để đặt kernel vào giữa ma trận
        start_x = sx // 2 - kx // 2
        start_y = sy // 2 - ky // 2
        start_z = sz // 2 - kz // 2

        # Đảm bảo các chỉ số không âm
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        start_z = max(0, start_z)

        # Đặt kernel vào giữa ma trận
        padded_kernel[
            start_x : start_x + kx, start_y : start_y + ky, start_z : start_z + kz
        ] = kernel

        # Nếu có ma trận tỷ trọng, áp dụng hiệu chỉnh không đồng nhất
        if density is not None and self.parameters["heterogeneity_correction"]:
            # Hiệu chỉnh kernel theo tỷ trọng
            # Phương pháp 1: Scaling kernel theo tỷ trọng
            density_norm = density / 1.0  # Chuẩn hóa theo tỷ trọng nước

            # Áp dụng scaling theo chiều sâu từ TERMA đến mỗi voxel
            # Trong thực tế, phức tạp hơn nhiều do cần tính đường đi của tia
            # Ở đây sử dụng phương pháp đơn giản hóa

            # Phương pháp 1: Tích chập với nhân không đổi, sau đó hiệu chỉnh kết quả
            result = self._fft_convolution(terma, padded_kernel)
            result = result * density_norm

            # Phương pháp 2: Mở rộng nhân tích chập thành các kernels theo voxel
            # Đây là phương pháp chính xác hơn nhưng tốn thời gian tính toán hơn
            if self.parameters["accuracy_level"] >= 3:
                # TODO: Triển khai phương pháp mở rộng nhân
                pass

            else:
                # Tích chập đồng nhất
                result = self._fft_convolution(terma, padded_kernel)

        elapsed = time.time() - start_time
        logger.debug(f"Thời gian tích chập: {elapsed:.2f}s")
        self._update_status(0.8, "Hoàn thành tích chập")

        return result

    def _fft_convolution(self, terma: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Tích chập sử dụng Fast Fourier Transform (FFT).

        Parameters:
            terma: Ma trận TERMA
            kernel: Nhân tích chập (đã được chuẩn bị với kích thước bằng terma)

        Returns:
            numpy.ndarray: Kết quả tích chập
        """
        # Dùng FFT để tính tích chập
        terma_fft = np.fft.fftn(terma)
        kernel_fft = np.fft.fftn(kernel)

        # Nhân trong miền tần số
        result_fft = terma_fft * kernel_fft

        # Chuyển về miền không gian
        result = np.real(np.fft.ifftn(result_fft))

        return result

    def _parallel_convolution(
        self, terma: np.ndarray, density: np.ndarray = None
    ) -> np.ndarray:
        """
        Thực hiện tích chập song song của TERMA với các nhân photon, electron và tán xạ.
        Cải thiện hiệu suất với xử lý song song.

        Parameters:
            terma (numpy.ndarray): Ma trận TERMA
            density (numpy.ndarray, optional): Ma trận mật độ

        Returns:
            numpy.ndarray: Ma trận liều sau tích chập
        """
        # Cập nhật tiến độ
        self._update_status(0.4, "Đang thực hiện tích chập song song")

        # Lấy số lượng luồng tối ưu
        num_threads = min(self.parameters["num_threads"], multiprocessing.cpu_count())

        # Tỷ lệ đóng góp của các thành phần
        weights = {"photon": 0.8, "electron": 0.15, "scatter": 0.05}

        # Khởi tạo kết quả
        dose = np.zeros_like(terma)

        # Cài đặt tiến độ
        self._update_status(0.5, "Đang tích chập với kernel photon, electron và tán xạ")

        # Cải thiện hàm compute_conv để xử lý song song tốt hơn
        def compute_conv(kernel_name, terma_chunk, density_chunk=None):
            """Tính tích chập cho một phần của terma."""
            kernel = self.kernels[kernel_name]
            return self._convolution(terma_chunk, kernel, density_chunk)

        if self.parameters["use_parallel"] and num_threads > 1:
            try:
                # Chia nhỏ TERMA thành các phần để xử lý song song
                # Chia theo lát cắt dọc (z-axis) để giảm thiểu chi phí chia nhỏ
                z_slices = terma.shape[2]
                chunk_size = max(1, z_slices // num_threads)
                chunks = []

                for z_start in range(0, z_slices, chunk_size):
                    z_end = min(z_start + chunk_size, z_slices)
                    terma_chunk = terma[:, :, z_start:z_end]
                    density_chunk = (
                        density[:, :, z_start:z_end] if density is not None else None
                    )
                    chunks.append((terma_chunk, density_chunk, z_start, z_end))

                # Xử lý song song cho mỗi kernel
                for kernel_name, weight in weights.items():
                    self._update_status(
                        0.5
                        + 0.1 * list(weights.keys()).index(kernel_name) / len(weights),
                        f"Đang tích chập với kernel {kernel_name}",
                    )

                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        futures = []
                        for terma_chunk, density_chunk, _, _ in chunks:
                            futures.append(
                                executor.submit(
                                    compute_conv,
                                    kernel_name,
                                    terma_chunk,
                                    density_chunk,
                                )
                            )

                        # Thu thập kết quả và đặt vào vị trí đúng trong mảng kết quả
                        for i, future in enumerate(futures):
                            _, _, z_start, z_end = chunks[i]
                            try:
                                chunk_result = future.result()
                                dose[:, :, z_start:z_end] += chunk_result * weight
                            except Exception as e:
                                logger.error(
                                    f"Lỗi khi tích chập với kernel {kernel_name} lát cắt {i}: {e}"
                                )

            except Exception as e:
                logger.error(f"Lỗi khi tính toán song song: {e}")
                # Fallback sang phương pháp tuần tự
                for idx, (name, kernel) in enumerate(self.kernels.items()):
                    if kernel is not None:
                        result = self._convolution(terma, kernel, density)
                        dose += result * weights[name]
                        self._update_status(
                            0.5 + 0.1 * idx / len(self.kernels),
                            f"Đã hoàn thành tích chập với kernel {name}",
                        )
        else:
            # Thực hiện tích chập tuần tự
            for idx, (name, kernel) in enumerate(self.kernels.items()):
                if kernel is not None:
                    result = self._convolution(terma, kernel, density)
                    dose += result * weights[name]
                    self._update_status(
                        0.5 + 0.1 * idx / len(self.kernels),
                        f"Đã hoàn thành tích chập với kernel {name}",
                    )

        # Dọn dẹp các giá trị không hợp lệ
        dose = np.nan_to_num(dose, nan=0.0, posinf=0.0, neginf=0.0)

        # Loại bỏ các giá trị âm (không có ý nghĩa vật lý)
        dose = np.maximum(dose, 0.0)

        # Cập nhật tiến độ
        self._update_status(0.8, "Đã hoàn thành tích chập")

        return dose

    def _create_dose_grid(self, shape, spacing, origin) -> DoseGrid:
        """
        Tạo lưới liều mới.

        Parameters:
            shape (tuple): Kích thước lưới
            spacing (tuple): Khoảng cách giữa các voxel
            origin (tuple): Vị trí gốc của lưới

        Returns:
            DoseGrid: Lưới liều mới
        """
        # Tạo lưới liều
        dose_data = np.zeros(shape, dtype=np.float32)
        # Sửa cách gọi constructor phù hợp với định nghĩa lớp DoseGrid
        return DoseGrid(dose_data, spacing, origin)

    def calculate(
        self,
        patient_ct: np.ndarray,
        structures: Dict[str, np.ndarray],
        beams: List[Dict[str, Any]],
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ) -> DoseGrid:
        """
        Tính toán phân bố liều.

        Parameters:
            patient_ct (numpy.ndarray): Ma trận hình ảnh CT (HU)
            structures (dict): Dict các cấu trúc
            beams (list): Danh sách các chùm tia
            spacing (tuple): Khoảng cách giữa các voxel (mm)
            origin (tuple): Vị trí gốc của lưới liều (mm)

        Returns:
            DoseGrid: Lưới liều kết quả
        """
        start_time = time.time()
        logger.info(
            f"Bắt đầu tính toán liều với thuật toán AAA, số chùm tia: {len(beams)}"
        )

        # Khởi tạo các thông số vật lý nếu chưa
        if self.kernels["photon"] is None:
            self._initialize_physics()

        # Cập nhật trạng thái
        self._update_status(0.05, "Bắt đầu tính toán AAA")

        # Tạo ma trận mật độ từ CT
        density = np.zeros_like(patient_ct, dtype=np.float32)
        for i in range(patient_ct.shape[0]):
            for j in range(patient_ct.shape[1]):
                for k in range(patient_ct.shape[2]):
                    density[i, j, k] = self._hu_to_density(patient_ct[i, j, k])

        # Tạo lưới liều tham chiếu
        reference_grid = self._create_dose_grid(patient_ct.shape, spacing, origin)

        # Tính tổng liều từ tất cả các chùm tia
        total_dose = np.zeros_like(patient_ct, dtype=np.float32)

        # Xác định có bao nhiêu chùm tia để tính tiến độ
        num_beams = len(beams)

        # Xử lý từng chùm tia
        for i, beam in enumerate(beams):
            # Cập nhật trạng thái
            beam_progress_start = 0.05 + (i / num_beams) * 0.9
            beam_progress_end = 0.05 + ((i + 1) / num_beams) * 0.9
            self._update_status(
                beam_progress_start, f"Đang tính toán chùm tia {i + 1}/{num_beams}"
            )

            # Đặt cấu hình chùm tia
            self.set_beam_config(beam)

            # Tính TERMA cho chùm tia này
            terma = self._compute_terma(patient_ct, beam)

            # Tính tích chập để chuyển từ TERMA sang liều
            beam_dose = self._parallel_convolution(terma, density)

            # Thêm liều của chùm tia này vào tổng liều
            # Nhân với trọng số của chùm tia
            beam_weight = beam.get("weight", 1.0)
            total_dose += beam_dose * beam_weight

            # Cập nhật trạng thái
            self._update_status(
                beam_progress_end, f"Đã tính xong chùm tia {i + 1}/{num_beams}"
            )

        # Chuẩn hóa liều
        dose_data = total_dose

        # Gán dữ liệu liều vào lưới liều
        reference_grid.data = dose_data

        # Tính thời gian
        elapsed = time.time() - start_time
        logger.info(f"Hoàn thành tính toán liều AAA trong {elapsed:.2f} giây")

        # Cập nhật trạng thái
        self._update_status(1.0, "Đã hoàn thành tính toán AAA")

        return reference_grid


class AAAImplementer(DoseCalculationImplementer):
    """
    Implementer cho thuật toán AAA (Anisotropic Analytical Algorithm).

    Class này cung cấp interface tương thích với hệ thống
    dose calculation implementer system.
    """

    def __init__(self):
        """Khởi tạo AAAImplementer."""
        self.algorithm = AAADoseCalculation()
        logger.info("Khởi tạo AAAImplementer")

    def supported_algorithms(self) -> List[DoseCalculationAlgorithm]:
        """Trả về danh sách các thuật toán được hỗ trợ."""
        return [DoseCalculationAlgorithm.AAA]

    def calculate(
        self,
        beam_data: Dict[str, Any],
        patient_data: Dict[str, Any],
        dose_grid: DoseGrid,
        calculation_options: Dict[str, Any] = None,
    ) -> np.ndarray:
        """
        Tính toán liều sử dụng thuật toán AAA.

        Parameters
        ----------
        beam_data : Dict[str, Any]
            Dữ liệu chùm tia
        patient_data : Dict[str, Any]
            Dữ liệu bệnh nhân
        dose_grid : DoseGrid
            Lưới liều
        calculation_options : Dict[str, Any], optional
            Các tùy chọn tính toán

        Returns
        -------
        np.ndarray
            Mảng phân bố liều
        """
        if calculation_options is None:
            calculation_options = {}

        try:
            # Cài đặt các tham số từ calculation_options
            if calculation_options:
                self.algorithm.set_parameters(calculation_options)

            # Lấy dữ liệu CT từ patient_data
            ct_data = patient_data.get("ct_data", np.zeros((64, 64, 32)))
            spacing = patient_data.get("spacing", (1.0, 1.0, 1.0))
            origin = patient_data.get("origin", (0.0, 0.0, 0.0))
            structures = patient_data.get("structures", {})

            # Chuẩn bị dữ liệu beam - đảm bảo là list
            if isinstance(beam_data, dict):
                beams = [beam_data]
            else:
                beams = beam_data if isinstance(beam_data, list) else [beam_data]

            # Tính toán liều using AAA algorithm
            result_grid = self.algorithm.calculate(
                patient_ct=ct_data,
                structures=structures,
                beams=beams,
                spacing=spacing,
                origin=origin,
            )

            return result_grid.data

        except Exception as e:
            logger.error(f"Lỗi trong AAAImplementer.calculate: {e}")
            # Trả về dose grid trống nếu có lỗi
            return np.zeros(dose_grid.shape)

    def get_description(self) -> str:
        """Trả về mô tả thuật toán."""
        return (
            "AAA (Anisotropic Analytical Algorithm) - thuật toán tính toán liều "
            "tiên tiến với khả năng hiệu chỉnh không đồng nhất, tương tự như "
            "trong Eclipse TPS của Varian."
        )

    def get_parameters_info(self) -> Dict[str, Any]:
        """Trả về thông tin về các tham số."""
        return {
            "resolution": {
                "type": "float",
                "default": 2.5,
                "description": "Độ phân giải tính toán (mm)",
                "range": (1.0, 5.0),
            },
            "scatter_kernel_size": {
                "type": "int",
                "default": 15,
                "description": "Kích thước nhân tích chập (voxel)",
                "range": (5, 25),
            },
            "heterogeneity_correction": {
                "type": "bool",
                "default": True,
                "description": "Bật/tắt hiệu chỉnh không đồng nhất",
            },
            "use_parallel": {
                "type": "bool",
                "default": True,
                "description": "Bật/tắt tính toán song song",
            },
            "num_threads": {
                "type": "int",
                "default": 4,
                "description": "Số luồng tính toán",
                "range": (1, 16),
            },
            "accuracy_level": {
                "type": "int",
                "default": 2,
                "description": "Mức độ chính xác (1-3)",
                "range": (1, 3),
            },
        }
