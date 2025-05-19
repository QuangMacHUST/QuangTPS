#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MonteCarloGPUAlgorithm - Lớp kế thừa MonteCarloGPU và triển khai DoseCalculationAlgorithm.

Lớp này kết hợp khả năng tăng tốc GPU của MonteCarloGPU với giao diện DoseCalculationAlgorithm
để tích hợp vào hệ thống thuật toán tính liều của QuangTPS.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from quangtps.dose.algorithms.improvements.monte_carlo_gpu import MonteCarloGPU
from quangtps.dose.algorithms import DoseCalculationAlgorithm, DoseCalculationResult
from quangtps.core.types import Patient, Beam, DoseGrid
from quangtps.evaluation.metrics.gamma_analysis import calculate_gamma_3d

logger = logging.getLogger(__name__)


class MonteCarloGPUAlgorithm(DoseCalculationAlgorithm):
    """
    Thuật toán tính liều Monte Carlo sử dụng GPU để tăng tốc.

    Lớp này cung cấp giao diện tiêu chuẩn của DoseCalculationAlgorithm để tích hợp
    với hệ thống thuật toán tính liều của QuangTPS, đồng thời tận dụng khả năng
    tính toán của GPU thông qua lớp MonteCarloGPU.
    """

    def __init__(self) -> None:
        """
        Khởi tạo thuật toán Monte Carlo GPU.
        """
        super().__init__()
        self.mc_gpu = MonteCarloGPU()
        self.initialized = False
        self.patient_data = None
        self.dose_grid = None

    def get_algorithm_type(self) -> str:
        """
        Trả về loại thuật toán.

        Returns
        -------
        str
            Định danh của thuật toán
        """
        return "monte_carlo_gpu"

    def get_display_name(self) -> str:
        """
        Trả về tên hiển thị của thuật toán.

        Returns
        -------
        str
            Tên thuật toán để hiển thị trong giao diện người dùng
        """
        return "Monte Carlo GPU"

    def initialize(self, patient_data: Any) -> bool:
        """
        Khởi tạo thuật toán với dữ liệu bệnh nhân.

        Parameters
        ----------
        patient_data : Any
            Dữ liệu bệnh nhân, bao gồm hình ảnh CT và các cấu trúc

        Returns
        -------
        bool
            True nếu khởi tạo thành công, False nếu thất bại
        """
        try:
            logger.info("Khởi tạo thuật toán Monte Carlo GPU")

            # Lưu dữ liệu bệnh nhân
            self.patient_data = patient_data

            # Lấy dữ liệu CT từ patient_data
            if hasattr(patient_data, "ct_image") and patient_data.ct_image is not None:
                ct_data = patient_data.ct_image.data
                voxel_size = patient_data.ct_image.voxel_size

                # Khởi tạo MonteCarloGPU với dữ liệu CT
                self.mc_gpu.set_ct_data(ct_data, voxel_size)
                self.initialized = True
                logger.info("Đã khởi tạo thuật toán Monte Carlo GPU thành công")
                return True
            else:
                logger.error("Không có dữ liệu CT trong bệnh nhân")
                return False

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo thuật toán Monte Carlo GPU: {str(e)}")
            return False

    def calculate_dose(self, beam_arrangement: Any) -> Any:
        """
        Tính phân bố liều dựa trên bố trí chùm tia cho bệnh nhân.

        Parameters
        ----------
        beam_arrangement : Any
            Bố trí chùm tia và thông số kỹ thuật

        Returns
        -------
        Any
            Kết quả tính toán liều
        """
        if not self.initialized:
            logger.error("Thuật toán chưa được khởi tạo")
            return None

        try:
            logger.info("Bắt đầu tính toán liều bằng Monte Carlo GPU")
            start_time = time.time()

            # Thiết lập cấu hình chùm tia cho MonteCarloGPU
            beam_config = self._convert_beam_to_config(beam_arrangement)
            self.mc_gpu.set_beam_configuration(beam_config)

            # Tính toán phân bố liều
            dose_grid = self.mc_gpu.calculate_dose()

            # Tạo kết quả tính toán liều
            result = DoseCalculationResult()
            result.dose_grid = dose_grid
            result.calculation_time = time.time() - start_time
            result.algorithm = self.get_display_name()
            result.metadata = {
                "particles_per_second": self.mc_gpu.particles_per_second,
                "num_particles": self.mc_gpu.num_particles,
                "has_gpu": self.mc_gpu.has_gpu,
                "gpu_library": self.mc_gpu.gpu_library
                if hasattr(self.mc_gpu, "gpu_library")
                else "none",
            }

            logger.info(
                f"Hoàn tất tính toán liều Monte Carlo GPU trong {result.calculation_time:.2f} giây"
            )
            logger.info(
                f"Hiệu suất: {self.mc_gpu.particles_per_second:.2f} particles/second"
            )

            return result

        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều bằng Monte Carlo GPU: {str(e)}")
            return None

    def _convert_beam_to_config(self, beam_arrangement) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng beam_arrangement thành cấu hình cho MonteCarloGPU.

        Parameters
        ----------
        beam_arrangement : Any
            Đối tượng chứa thông tin về bố trí chùm tia

        Returns
        -------
        Dict[str, Any]
            Cấu hình chùm tia cho MonteCarloGPU
        """
        config = {}

        # Xử lý các loại beam_arrangement khác nhau
        if hasattr(beam_arrangement, "beams") and beam_arrangement.beams:
            # Trường hợp beam_arrangement là một đối tượng chứa danh sách beams
            beams = beam_arrangement.beams

            # Lấy thông tin từ beam đầu tiên (hoặc kết hợp nhiều beam)
            beam = beams[0]  # Đơn giản hóa: chỉ sử dụng beam đầu tiên

            config["energy"] = (
                float(beam.energy.replace("MV", "")) if hasattr(beam, "energy") else 6.0
            )
            config["gantry_angle"] = (
                beam.gantry_angle if hasattr(beam, "gantry_angle") else 0.0
            )
            config["collimator_angle"] = (
                beam.collimator_angle if hasattr(beam, "collimator_angle") else 0.0
            )
            config["couch_angle"] = (
                beam.couch_angle if hasattr(beam, "couch_angle") else 0.0
            )
            config["isocenter"] = (
                beam.isocenter if hasattr(beam, "isocenter") else [0, 0, 0]
            )

            # Xử lý thông tin MLC nếu có
            if hasattr(beam, "mlc") and beam.mlc is not None:
                config["has_mlc"] = True
                config["mlc_positions"] = beam.mlc.get_leaf_positions()
            else:
                config["has_mlc"] = False

            # Thêm thông tin về field size
            if hasattr(beam, "field_size"):
                config["field_size"] = beam.field_size
            else:
                config["field_size"] = [10.0, 10.0]  # Default field size 10x10 cm

        elif isinstance(beam_arrangement, dict):
            # Trường hợp beam_arrangement đã là một dict cấu hình
            config = beam_arrangement

        else:
            # Trường hợp mặc định
            logger.warning(
                "Không nhận dạng được định dạng beam_arrangement, sử dụng giá trị mặc định"
            )
            config = {
                "energy": 6.0,
                "gantry_angle": 0.0,
                "collimator_angle": 0.0,
                "isocenter": [0, 0, 0],
                "field_size": [10.0, 10.0],
            }

        return config

    def compare_with_dose_grid(self, reference_dose_grid, evaluation_criteria=None):
        """
        So sánh kết quả tính toán với lưới liều tham chiếu.

        Parameters
        ----------
        reference_dose_grid : np.ndarray
            Lưới liều tham chiếu
        evaluation_criteria : Dict, optional
            Tiêu chí đánh giá, mặc định là None

        Returns
        -------
        Dict
            Kết quả so sánh
        """
        if self.dose_grid is None or reference_dose_grid is None:
            logger.error("Không có dữ liệu liều để so sánh")
            return {}

        try:
            # Tiêu chí gamma mặc định nếu không được cung cấp
            if evaluation_criteria is None:
                evaluation_criteria = {
                    "dose_threshold": 3.0,  # % dose difference
                    "distance_threshold": 3.0,  # mm
                    "local_normalization": True,
                    "dose_cutoff": 10.0,  # % dose cutoff
                }

            # Thực hiện phân tích gamma
            gamma_results = calculate_gamma_3d(
                self.dose_grid,
                reference_dose_grid,
                evaluation_criteria["dose_threshold"],
                evaluation_criteria["distance_threshold"],
                local_normalization=evaluation_criteria["local_normalization"],
                dose_cutoff_percent=evaluation_criteria["dose_cutoff"],
            )

            # Tính toán chỉ số thống kê
            gamma_pass_rate = (
                (gamma_results <= 1.0).sum()
                / np.count_nonzero(gamma_results >= 0)
                * 100
            )

            return {
                "gamma_pass_rate": gamma_pass_rate,
                "gamma_mean": np.mean(gamma_results[gamma_results >= 0]),
                "gamma_median": np.median(gamma_results[gamma_results >= 0]),
                "gamma_max": np.max(gamma_results[gamma_results >= 0]),
                "evaluation_criteria": evaluation_criteria,
            }

        except Exception as e:
            logger.error(f"Lỗi khi so sánh lưới liều: {str(e)}")
            return {}


# Hàm đăng ký thuật toán với hệ thống
def register_algorithm():
    """
    Đăng ký thuật toán Monte Carlo GPU vào hệ thống thuật toán tính liều.
    """
    from quangtps.dose.algorithms import register_algorithm

    # Đăng ký MonteCarloGPUAlgorithm với hệ thống
    register_algorithm("monte_carlo_gpu", MonteCarloGPUAlgorithm)
    logger.info("Đã đăng ký thuật toán Monte Carlo GPU thành công")
