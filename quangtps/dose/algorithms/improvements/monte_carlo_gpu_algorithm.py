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

    def __init__(self, **kwargs) -> None:
        """
        Khởi tạo thuật toán Monte Carlo GPU.

        Parameters
        ----------
        **kwargs
            Tham số cấu hình bổ sung cho thuật toán Monte Carlo
            - num_particles : số hạt mô phỏng (mặc định: 1,000,000)
            - gpu_id : ID của GPU sử dụng (mặc định: 0)
            - use_cpu_fallback : Sử dụng CPU khi không có GPU (mặc định: True)
            - min_gpu_memory : Lượng bộ nhớ GPU tối thiểu để tính toán (GB) (mặc định: 2)
            - auto_adjust_particles : Tự động điều chỉnh số hạt mô phỏng theo bộ nhớ (mặc định: True)
        """
        super().__init__()

        # Tùy chỉnh cấu hình Monte Carlo
        self.config = {
            "num_particles": kwargs.get("num_particles", 1000000),
            "gpu_id": kwargs.get("gpu_id", 0),
            "use_cpu_fallback": kwargs.get("use_cpu_fallback", True),
            "min_gpu_memory": kwargs.get("min_gpu_memory", 2),  # GB
            "auto_adjust_particles": kwargs.get("auto_adjust_particles", True),
            "enable_progress_callback": kwargs.get("enable_progress_callback", True),
            "max_threads": kwargs.get("max_threads", 0),  # 0 = tự động
        }

        self.mc_gpu = MonteCarloGPU(
            num_particles=self.config["num_particles"],
            gpu_id=self.config["gpu_id"],
            auto_particles=self.config["auto_adjust_particles"],
        )

        self.initialized = False
        self.patient_data = None
        self.dose_grid = None
        self.progress_callback = None

    def set_progress_callback(self, callback_fn):
        """
        Thiết lập hàm callback để cập nhật tiến trình tính toán.

        Parameters
        ----------
        callback_fn : callable
            Hàm callback với tham số từ 0-100 cho phần trăm hoàn thành
        """
        self.progress_callback = callback_fn

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
        if self.mc_gpu and self.mc_gpu.has_gpu:
            gpu_name = getattr(self.mc_gpu, "device_name", "")
            return f"Monte Carlo GPU ({gpu_name})"
        else:
            return "Monte Carlo (CPU fallback)"

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

                # Chuẩn bị vật liệu dựa trên dữ liệu HU
                if hasattr(self.mc_gpu, "_prepare_materials"):
                    self.mc_gpu._prepare_materials()

                self.initialized = True

                # Hiển thị thông tin về GPU/CPU đang sử dụng
                if self.mc_gpu.has_gpu:
                    mem_info = getattr(
                        self.mc_gpu, "device_memory", {"free": 0, "total": 0}
                    )
                    logger.info(
                        f"Sử dụng GPU cho tính toán Monte Carlo: {self.mc_gpu.gpu_library}, "
                        f"Bộ nhớ: {mem_info.get('free', 0):.2f}GB/{mem_info.get('total', 0):.2f}GB"
                    )
                else:
                    logger.info(
                        f"Sử dụng CPU fallback cho tính toán Monte Carlo với {self.mc_gpu.num_particles} hạt"
                    )

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

            # Thông báo tiến trình
            if self.progress_callback:
                self.progress_callback(0)

            # Thiết lập cấu hình chùm tia cho MonteCarloGPU
            beam_config = self._convert_beam_to_config(beam_arrangement)
            self.mc_gpu.set_beam_configuration(beam_config)

            # Thông báo tiến trình
            if self.progress_callback:
                self.progress_callback(10)

            # Hook để cập nhật tiến trình
            original_dose_calculation = self.mc_gpu.calculate_dose

            def dose_calculation_with_progress(*args, **kwargs):
                """Wrapper cho hàm tính toán để cập nhật tiến trình"""
                # Tính toán liều
                result = original_dose_calculation(*args, **kwargs)

                # Cập nhật tiến trình từ 10% đến 90%
                if self.progress_callback:
                    self.progress_callback(90)

                return result

            # Thay thế tạm thời hàm tính toán
            if self.config["enable_progress_callback"] and self.progress_callback:
                self.mc_gpu.calculate_dose = dose_calculation_with_progress

            # Tính toán phân bố liều
            dose_grid = self.mc_gpu.calculate_dose()

            # Khôi phục hàm tính toán ban đầu
            if self.config["enable_progress_callback"] and self.progress_callback:
                self.mc_gpu.calculate_dose = original_dose_calculation

            # Thông báo tiến trình
            if self.progress_callback:
                self.progress_callback(95)

            # Tạo kết quả tính toán liều
            result = DoseCalculationResult()
            result.dose_grid = dose_grid
            result.calculation_time = time.time() - start_time
            result.algorithm = self.get_display_name()
            result.metadata = {
                "particles_per_second": getattr(self.mc_gpu, "particles_per_second", 0),
                "num_particles": self.mc_gpu.num_particles,
                "has_gpu": self.mc_gpu.has_gpu,
                "gpu_library": getattr(self.mc_gpu, "gpu_library", "none"),
                "calculation_time": result.calculation_time,
            }

            # Thông báo tiến trình đã hoàn thành
            if self.progress_callback:
                self.progress_callback(100)

            logger.info(
                f"Hoàn tất tính toán liều Monte Carlo trong {result.calculation_time:.2f} giây"
            )

            particles_per_second = getattr(
                self.mc_gpu,
                "particles_per_second",
                self.mc_gpu.num_particles / max(result.calculation_time, 0.001),
            )

            logger.info(f"Hiệu suất: {particles_per_second:.2f} hạt/giây")

            return result

        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều bằng Monte Carlo GPU: {str(e)}")
            if self.progress_callback:
                self.progress_callback(
                    100
                )  # Đảm bảo tiến trình hoàn thành ngay cả khi có lỗi
            return None

    def calculate_3d_dose_distribution(
        self, beam_arrangement, structures=None, prescription=None
    ):
        """
        Tính toán phân bố liều 3D đầy đủ cho toàn bộ bố trí chùm tia.

        Parameters
        ----------
        beam_arrangement : Any
            Bố trí chùm tia và thông số kỹ thuật
        structures : List, optional
            Danh sách cấu trúc để tối ưu hóa tính toán (tùy chọn)
        prescription : Dict, optional
            Thông tin kê đơn để chuẩn hóa liều (tùy chọn)

        Returns
        -------
        DoseGrid
            Phân bố liều 3D cho toàn bộ hình ảnh CT
        """
        # Kiểm tra khởi tạo
        if not self.initialized:
            logger.error("Thuật toán chưa được khởi tạo")
            return None

        try:
            # Tính toán liều 3D
            result = self.calculate_dose(beam_arrangement)

            if result is None or result.dose_grid is None:
                return None

            # Chuẩn hóa liều nếu có thông tin kê đơn
            if prescription and hasattr(result.dose_grid, "normalize"):
                prescription_dose = prescription.get("dose", 0)
                if prescription_dose > 0:
                    result.dose_grid.normalize(prescription_dose)

            return result.dose_grid

        except Exception as e:
            logger.error(f"Lỗi khi tính toán phân bố liều 3D: {str(e)}")
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

            # Xử lý nhiều beam
            beam_configs = []
            for beam in beams:
                beam_config = {
                    "energy": float(beam.energy.replace("MV", ""))
                    if hasattr(beam, "energy")
                    else 6.0,
                    "gantry_angle": beam.gantry_angle
                    if hasattr(beam, "gantry_angle")
                    else 0.0,
                    "collimator_angle": beam.collimator_angle
                    if hasattr(beam, "collimator_angle")
                    else 0.0,
                    "couch_angle": beam.couch_angle
                    if hasattr(beam, "couch_angle")
                    else 0.0,
                    "isocenter": beam.isocenter
                    if hasattr(beam, "isocenter")
                    else [0, 0, 0],
                    "weight": beam.weight if hasattr(beam, "weight") else 1.0,
                }

                # Xử lý thông tin MLC nếu có
                if hasattr(beam, "mlc") and beam.mlc is not None:
                    beam_config["mlc"] = {
                        "leaf_positions": beam.mlc.get_leaf_positions()
                        if hasattr(beam.mlc, "get_leaf_positions")
                        else [],
                        "type": beam.mlc.type
                        if hasattr(beam.mlc, "type")
                        else "static",
                    }

                # Xử lý thông tin jaw nếu có
                if hasattr(beam, "jaw") and beam.jaw is not None:
                    beam_config["jaw"] = {
                        "x1": beam.jaw.x1 if hasattr(beam.jaw, "x1") else -10.0,
                        "x2": beam.jaw.x2 if hasattr(beam.jaw, "x2") else 10.0,
                        "y1": beam.jaw.y1 if hasattr(beam.jaw, "y1") else -10.0,
                        "y2": beam.jaw.y2 if hasattr(beam.jaw, "y2") else 10.0,
                    }

                beam_configs.append(beam_config)

            config["beams"] = beam_configs
            config["normalization"] = getattr(beam_arrangement, "normalization", None)

        else:
            # Xử lý trường hợp beam_arrangement là một beam đơn lẻ
            config["energy"] = (
                float(beam_arrangement.energy.replace("MV", ""))
                if hasattr(beam_arrangement, "energy")
                else 6.0
            )
            config["gantry_angle"] = (
                beam_arrangement.gantry_angle
                if hasattr(beam_arrangement, "gantry_angle")
                else 0.0
            )
            config["collimator_angle"] = (
                beam_arrangement.collimator_angle
                if hasattr(beam_arrangement, "collimator_angle")
                else 0.0
            )
            config["couch_angle"] = (
                beam_arrangement.couch_angle
                if hasattr(beam_arrangement, "couch_angle")
                else 0.0
            )
            config["isocenter"] = (
                beam_arrangement.isocenter
                if hasattr(beam_arrangement, "isocenter")
                else [0, 0, 0]
            )

            # Beam đơn lẻ luôn có trọng số 1.0
            config["beams"] = [{**config, "weight": 1.0}]
            config["normalization"] = None

        return config

    def compare_with_dose_grid(self, reference_dose_grid, evaluation_criteria=None):
        """
        So sánh kết quả tính toán liều với phân phối liều tham chiếu.

        Parameters
        ----------
        reference_dose_grid : DoseGrid
            Phân phối liều tham chiếu để so sánh.
        evaluation_criteria : dict, optional
            Tiêu chí đánh giá, bao gồm:
            - 'gamma_criteria': List các tuple (dta_mm, dd_percent) hoặc (distance_mm, dose_percent)
            - 'threshold': Ngưỡng liều tương đối để tính gamma (0.0 - 1.0)

        Returns
        -------
        dict
            Kết quả so sánh với các chỉ số đánh giá.
        """
        if not evaluation_criteria:
            evaluation_criteria = {
                "gamma_criteria": [(3.0, 3.0)],  # (dta_mm, dd_percent)
                "threshold": 0.1,
            }

        try:
            # Import hàm phân tích gamma
            from quangtps.evaluation.metrics.gamma_analysis import (
                calculate_gamma_3d,
                gamma_pass_rate,
            )

            # Chuẩn bị dữ liệu liều
            ref_dose_grid = reference_dose_grid.get_grid()
            eval_dose_grid = self.dose_grid.get_grid()

            # Lấy kích thước voxel
            voxel_size = reference_dose_grid.get_voxel_size()

            # Tính toán với mỗi tiêu chí gamma
            gamma_results = {}
            for criterion in evaluation_criteria["gamma_criteria"]:
                # Hỗ trợ cả hai cách đặt tên tham số
                dta_mm, dd_percent = criterion

                # Tính toán gamma 3D - hỗ trợ cả tên tham số cũ và mới
                gamma = calculate_gamma_3d(
                    reference=ref_dose_grid,
                    evaluation=eval_dose_grid,
                    dta_mm=dta_mm,
                    dd_percent=dd_percent,
                    threshold=evaluation_criteria.get("threshold", 0.1),
                    voxel_size=voxel_size,
                    local_normalization=evaluation_criteria.get(
                        "local_normalization", False
                    ),
                )

                # Tính tỷ lệ đạt tiêu chí gamma
                pass_rate = gamma_pass_rate(gamma, pass_criteria=1.0)

                # Lưu kết quả
                criterion_str = f"{dta_mm}mm/{dd_percent}%"
                gamma_results[criterion_str] = {
                    "gamma": gamma,
                    "pass_rate": pass_rate,
                }

            return {
                "gamma_results": gamma_results,
                "status": "success",
                "message": "Phân tích gamma hoàn tất",
            }

        except Exception as e:
            logger.error(f"Lỗi khi so sánh phân phối liều: {e}")
            return {
                "status": "error",
                "message": f"Không thể tính toán gamma: {str(e)}",
            }


def register_algorithm():
    """
    Đăng ký thuật toán Monte Carlo GPU với hệ thống.
    """
    try:
        from quangtps.dose.algorithms import register_dose_algorithm

        register_dose_algorithm("monte_carlo_gpu", MonteCarloGPUAlgorithm)
        logger.info("Đã đăng ký thuật toán Monte Carlo GPU thành công")
    except Exception as e:
        logger.error(f"Không thể đăng ký thuật toán Monte Carlo GPU: {str(e)}")


# Tự động đăng ký thuật toán khi import module
register_algorithm()
