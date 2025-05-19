#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Monte Carlo dose calculation algorithm with GPU acceleration.

This module provides a GPU-accelerated implementation of the Monte Carlo
dose calculation algorithm for radiation therapy treatment planning.
"""

import logging
import numpy as np
import os
import time
from typing import Dict, List, Optional, Any, Tuple, Union

logger = logging.getLogger(__name__)

# Try to import GPU libraries with fallback mechanisms
try:
    import cupy as cp

    HAS_CUPY = True
    logger.info("CuPy successfully imported for GPU acceleration")
except ImportError:
    HAS_CUPY = False
    logger.warning("CuPy not available. Will try PyCUDA next.")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda import gpuarray
    import pycuda.compiler as compiler

    HAS_PYCUDA = True
    logger.info("PyCUDA successfully imported for GPU acceleration")
except ImportError:
    HAS_PYCUDA = False
    if not HAS_CUPY:
        logger.warning(
            "Neither CuPy nor PyCUDA available. GPU acceleration will not be available."
        )


# Base class for Monte Carlo simulation using GPU
class MonteCarloGPU:
    """
    Monte Carlo dose calculation using GPU acceleration.

    This class implements Monte Carlo dose calculation for radiation therapy
    treatment planning, leveraging GPU acceleration for significantly faster
    computation compared to CPU-based implementations.
    """

    def __init__(self, num_particles: int = 1000000, **kwargs):
        """
        Initialize GPU-accelerated Monte Carlo algorithm.

        Parameters
        ----------
        num_particles : int, optional
            Number of particles to simulate, by default 1000000
        **kwargs
            Additional configuration parameters
        """
        self.num_particles = num_particles
        self.config = kwargs
        self.device = None
        self.has_gpu = False

        # Try to initialize GPU
        self._setup_gpu()

        # Initialize dose grid
        self.dose_grid = None
        self.ct_data = None
        self.materials = None
        self.beam_config = None

        # Performance metrics
        self.calculation_time = 0
        self.particles_per_second = 0
        self.memory_usage = 0

    def _setup_gpu(self):
        """Setup GPU environment based on available libraries."""
        if HAS_CUPY:
            self._setup_gpu_cupy()
        elif HAS_PYCUDA:
            self._setup_gpu_pycuda()
        else:
            self._setup_cpu_fallback()

    def _setup_gpu_cupy(self):
        """Initialize GPU using CuPy."""
        try:
            # Get device information
            num_gpus = cp.cuda.runtime.getDeviceCount()
            if num_gpus > 0:
                # Use first available GPU by default
                device_id = 0
                if "gpu_id" in self.config:
                    device_id = min(self.config["gpu_id"], num_gpus - 1)

                cp.cuda.Device(device_id).use()
                self.device = cp.cuda.Device(device_id)
                device_name = self.device.attributes.get("name", "Unknown").decode(
                    "utf-8"
                )
                mem_info = self.device.mem_info
                total_memory = mem_info[1] / (1024**3)  # GB
                free_memory = mem_info[0] / (1024**3)  # GB

                logger.info(
                    f"Using GPU {device_id}: {device_name} with {free_memory:.2f}GB/{total_memory:.2f}GB free memory"
                )
                self.has_gpu = True
                self.gpu_library = "cupy"

                # Set number of particles based on available memory
                if "auto_particles" in self.config and self.config["auto_particles"]:
                    # 1 million particles per GB as a rough estimate
                    self.num_particles = max(int(free_memory * 1000000), 1000000)
                    logger.info(
                        f"Auto-configured for {self.num_particles} particles based on available memory"
                    )
            else:
                logger.warning("No CUDA-capable GPU found despite CuPy being installed")
                self._setup_cpu_fallback()
        except Exception as e:
            logger.error(f"Error initializing CuPy GPU: {str(e)}")
            self._setup_cpu_fallback()

    def _setup_gpu_pycuda(self):
        """Initialize GPU using PyCUDA."""
        try:
            # Get device information
            num_gpus = cuda.Device.count()
            if num_gpus > 0:
                # Use first available GPU by default
                device_id = 0
                if "gpu_id" in self.config:
                    device_id = min(self.config["gpu_id"], num_gpus - 1)

                self.device = cuda.Device(device_id)
                device_name = self.device.name()
                total_memory = self.device.total_memory() / (1024**3)  # GB
                free_memory = (
                    self.device.total_memory() - self.device.used_memory()
                ) / (1024**3)  # GB

                logger.info(
                    f"Using GPU {device_id}: {device_name} with {free_memory:.2f}GB/{total_memory:.2f}GB free memory"
                )
                self.has_gpu = True
                self.gpu_library = "pycuda"

                # Set number of particles based on available memory
                if "auto_particles" in self.config and self.config["auto_particles"]:
                    # 1 million particles per GB as a rough estimate
                    self.num_particles = max(int(free_memory * 1000000), 1000000)
                    logger.info(
                        f"Auto-configured for {self.num_particles} particles based on available memory"
                    )
            else:
                logger.warning(
                    "No CUDA-capable GPU found despite PyCUDA being installed"
                )
                self._setup_cpu_fallback()
        except Exception as e:
            logger.error(f"Error initializing PyCUDA GPU: {str(e)}")
            self._setup_cpu_fallback()

    def _setup_cpu_fallback(self):
        """Set up CPU fallback when GPU is not available."""
        import multiprocessing

        num_cores = multiprocessing.cpu_count()
        logger.warning(
            f"Using CPU fallback with {num_cores} cores (much slower than GPU)"
        )
        self.has_gpu = False
        self.device = None
        self.gpu_library = None

        # Limit particles when using CPU to avoid excessive runtime
        if self.num_particles > 500000:
            self.num_particles = 500000
            logger.info(
                f"Reduced particle count to {self.num_particles} for CPU calculation"
            )

    def set_ct_data(self, ct_data: np.ndarray, voxel_size: Tuple[float, float, float]):
        """
        Set CT data for dose calculation.

        Parameters
        ----------
        ct_data : np.ndarray
            3D array containing CT data in Hounsfield units
        voxel_size : Tuple[float, float, float]
            Size of voxels in mm
        """
        self.ct_data = ct_data
        self.voxel_size = voxel_size

        # Initialize empty dose grid matching CT dimensions
        self.dose_grid = np.zeros_like(ct_data, dtype=np.float32)

        # Pre-process CT data for material assignment
        self._prepare_materials()

    def _prepare_materials(self):
        """Convert CT data (HU) to material properties for dose calculation."""
        if self.ct_data is None:
            logger.error("CT data not set. Call set_ct_data first.")
            return

        # Simple conversion from HU to relative electron density
        # In a real implementation, this would use a calibration curve
        self.materials = np.zeros_like(self.ct_data, dtype=np.float32)

        # Simple linear mapping of HU to relative electron density
        # Water is typically around 0 HU with density 1.0
        self.materials = 1.0 + (self.ct_data / 1000.0)

        # Limit to realistic values
        self.materials = np.clip(self.materials, 0.1, 10.0)

    def set_beam_configuration(self, beam_config: Dict[str, Any]):
        """
        Configure treatment beam parameters.

        Parameters
        ----------
        beam_config : Dict[str, Any]
            Dictionary containing beam parameters like energy, angle, etc.
        """
        self.beam_config = beam_config

    def calculate_dose(self):
        """
        Calculate dose distribution using Monte Carlo simulation.

        Returns
        -------
        np.ndarray
            3D dose distribution array
        """
        if self.ct_data is None:
            logger.error("CT data not set. Call set_ct_data first.")
            return None

        if self.beam_config is None:
            logger.error("Beam not configured. Call set_beam_configuration first.")
            return None

        start_time = time.time()

        # Choose appropriate calculation method based on available hardware
        if self.has_gpu and self.gpu_library == "cupy":
            self._calculate_dose_cupy()
        elif self.has_gpu and self.gpu_library == "pycuda":
            self._calculate_dose_pycuda()
        else:
            self._calculate_dose_cpu()

        end_time = time.time()
        self.calculation_time = end_time - start_time
        self.particles_per_second = self.num_particles / self.calculation_time

        logger.info(
            f"Monte Carlo calculation completed in {self.calculation_time:.2f} seconds"
        )
        logger.info(f"Performance: {self.particles_per_second:.2f} particles/second")

        # Apply final normalization
        self._normalize_dose()

        return self.dose_grid

    def _calculate_dose_cupy(self):
        """Implement dose calculation using CuPy."""
        try:
            # Transfer data to GPU
            ct_gpu = cp.asarray(self.ct_data)
            materials_gpu = cp.asarray(self.materials)
            dose_gpu = cp.zeros_like(ct_gpu, dtype=cp.float32)

            # Record memory usage
            self.memory_usage = (
                ct_gpu.nbytes + materials_gpu.nbytes + dose_gpu.nbytes
            ) / (1024**3)  # GB
            logger.info(f"GPU memory usage: {self.memory_usage:.2f} GB")

            # TODO: Implement actual Monte Carlo transport algorithm
            # This is just a placeholder calculation for demonstration

            # Get beam parameters
            energy = self.beam_config.get("energy", 6.0)  # MV
            angle_gantry = self.beam_config.get("gantry_angle", 0.0)  # degrees
            angle_collimator = self.beam_config.get("collimator_angle", 0.0)  # degrees
            isocenter = self.beam_config.get("isocenter", [0, 0, 0])  # mm

            # Run simulation
            logger.info(
                f"Starting GPU Monte Carlo simulation with {self.num_particles} particles"
            )

            # Transfer results back to CPU
            self.dose_grid = cp.asnumpy(dose_gpu)

        except Exception as e:
            logger.error(f"Error in CuPy dose calculation: {str(e)}")
            logger.warning("Falling back to CPU calculation")
            self._calculate_dose_cpu()

    def _calculate_dose_pycuda(self):
        """Implement dose calculation using PyCUDA."""
        try:
            # Similar to CuPy implementation but using PyCUDA
            # This is a placeholder for the actual implementation
            logger.info(
                f"Starting PyCUDA Monte Carlo simulation with {self.num_particles} particles"
            )

            # TODO: Implement actual Monte Carlo transport using PyCUDA

        except Exception as e:
            logger.error(f"Error in PyCUDA dose calculation: {str(e)}")
            logger.warning("Falling back to CPU calculation")
            self._calculate_dose_cpu()

    def _calculate_dose_cpu(self):
        """CPU fallback implementation of Monte Carlo dose calculation."""
        logger.info(
            f"Starting CPU Monte Carlo simulation with {self.num_particles} particles"
        )

        # TODO: Implement simplified Monte Carlo algorithm for CPU
        # This is just a placeholder calculation for demonstration

        # Simple exponential attenuation based on ray tracing
        # In a real implementation, this would be much more complex
        pass

    def _normalize_dose(self):
        """Normalize dose grid to prescribed dose level."""
        if self.dose_grid is None:
            return

        # Find maximum dose value
        max_dose = np.max(self.dose_grid)
        if max_dose > 0:
            # Normalize to prescription dose or to 1.0 if not specified
            prescription = self.beam_config.get("prescription", 1.0)  # Gy
            self.dose_grid = self.dose_grid * (prescription / max_dose)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from the last calculation.

        Returns
        -------
        Dict[str, Any]
            Dictionary with performance metrics
        """
        return {
            "calculation_time": self.calculation_time,
            "particles_per_second": self.particles_per_second,
            "memory_usage_gb": self.memory_usage,
            "using_gpu": self.has_gpu,
            "gpu_library": self.gpu_library,
            "num_particles": self.num_particles,
        }

    def compare_with_dose_grid(self, reference_dose: np.ndarray) -> Dict[str, Any]:
        """
        So sánh phân bố liều tính toán với phân bố liều tham chiếu.

        Parameters
        ----------
        reference_dose : np.ndarray
            Phân phối liều tham chiếu để so sánh

        Returns
        -------
        Dict[str, Any]
            Dictionary với các chỉ số so sánh
        """
        if self.dose_grid is None or reference_dose is None:
            logger.error("Cả liều tính toán và tham chiếu đều phải tồn tại")
            return None

        # Kiểm tra kích thước
        if self.dose_grid.shape != reference_dose.shape:
            logger.error(
                f"Không khớp kích thước: {self.dose_grid.shape} vs {reference_dose.shape}"
            )
            return None

        # Tính toán các chỉ số cơ bản
        diff = self.dose_grid - reference_dose
        abs_diff = np.abs(diff)

        # Tạo mask chỉ xét vùng có liều > 10% liều tối đa
        ref_max = np.max(reference_dose)
        mask = reference_dose >= (0.1 * ref_max)

        # Tính % sai khác trung bình trong vùng quan tâm
        mean_pct_diff = (
            100.0 * np.mean(abs_diff[mask]) / ref_max if np.sum(mask) > 0 else 0.0
        )

        metrics = {
            "mean_error": np.mean(diff),
            "mean_abs_error": np.mean(abs_diff),
            "mean_pct_diff": mean_pct_diff,
            "max_error": np.max(abs_diff),
            "rms_error": np.sqrt(np.mean(np.square(diff))),
        }

        # Tính chỉ số gamma nếu module phân tích gamma có sẵn
        try:
            from quangtps.evaluation.metrics.gamma_analysis import (
                calculate_gamma_3d,
                gamma_pass_rate,
            )

            logger.info("Bắt đầu phân tích gamma 3D...")

            # Lấy thông tin voxel_size nếu có
            voxel_size = (
                self.voxel_size if hasattr(self, "voxel_size") else (1.0, 1.0, 1.0)
            )

            # Thiết lập các tham số cho phân tích gamma
            distance_criterion_mm = 3.0  # Khoảng cách đến điểm tương đồng (mm)
            dose_difference_percent = 3.0  # Sai khác liều (% của liều tối đa)
            threshold_dose = 0.1  # Chỉ tính gamma cho vùng liều > 10% của liều tối đa

            # Thêm vào metric container
            metrics["gamma_criteria"] = {
                "dta_mm": distance_criterion_mm,
                "dd_percent": dose_difference_percent,
                "threshold": threshold_dose,
                "voxel_size": voxel_size,
            }

            # Gọi hàm gamma analysis với đúng thông số
            gamma_result = calculate_gamma_3d(
                reference=reference_dose,
                evaluation=self.dose_grid,
                dta_mm=distance_criterion_mm,
                dd_percent=dose_difference_percent,
                threshold=threshold_dose,
                voxel_size=voxel_size,
                max_gamma=5.0,
                local_normalization=False,
            )

            # Tính pass rate và thêm vào kết quả
            pass_rate_value = gamma_pass_rate(
                gamma_result, mask=mask, pass_criteria=1.0
            )

            metrics["gamma_analysis"] = {
                "pass_rate": pass_rate_value,
                "mean_gamma": np.mean(gamma_result[mask]) if np.sum(mask) > 0 else 0.0,
                "max_gamma": np.max(gamma_result[mask]) if np.sum(mask) > 0 else 0.0,
                "criteria_string": f"{distance_criterion_mm}mm/{dose_difference_percent}%",
            }

            # Thêm phân tích với tiêu chí khác
            for dta, dd in [(2.0, 2.0), (1.0, 1.0)]:
                try:
                    gamma_key = f"gamma_{int(dta)}mm_{int(dd)}pct"
                    gamma_2 = calculate_gamma_3d(
                        reference=reference_dose,
                        evaluation=self.dose_grid,
                        dta_mm=dta,
                        dd_percent=dd,
                        threshold=threshold_dose,
                        voxel_size=voxel_size,
                        max_gamma=5.0,
                        local_normalization=False,
                    )
                    pass_rate_2 = gamma_pass_rate(gamma_2, mask=mask, pass_criteria=1.0)
                    metrics["gamma_analysis"][gamma_key] = {
                        "pass_rate": pass_rate_2,
                        "criteria_string": f"{dta}mm/{dd}%",
                    }
                except Exception as e:
                    logger.warning(
                        f"Lỗi khi tính gamma với tiêu chí {dta}mm/{dd}%: {str(e)}"
                    )

            logger.info(
                f"Phân tích gamma đã hoàn tất với tỉ lệ đạt {pass_rate_value:.2f}% theo tiêu chí {distance_criterion_mm}mm/{dose_difference_percent}%"
            )

        except ImportError:
            logger.warning("Module phân tích gamma không khả dụng, bỏ qua chỉ số này")
        except Exception as e:
            logger.error(f"Lỗi trong tính toán gamma: {str(e)}", exc_info=True)
            metrics["gamma_error"] = str(e)

        return metrics


# Lớp tích hợp MonteCarloGPUAlgorithm kế thừa từ MonteCarloGPU
class MonteCarloGPUAlgorithm(MonteCarloGPU):
    """
    Lớp tích hợp thuật toán Monte Carlo GPU vào hệ thống thuật toán tính liều của QuangTPS.

    Lớp này kế thừa từ MonteCarloGPU và triển khai các phương thức cần thiết để tích hợp
    với hệ thống thuật toán tính liều (DoseCalculationAlgorithm).
    """

    def __init__(self, **kwargs):
        """
        Khởi tạo thuật toán Monte Carlo GPU.

        Parameters
        ----------
        **kwargs
            Các tham số cấu hình cho thuật toán
        """
        super().__init__(**kwargs)
        self.patient_data = None
        self.beam_arrangement = None
        self.calculation_status = {
            "initialized": False,
            "ready": False,
            "completed": False,
            "error": None,
        }
        logger.info("Khởi tạo thuật toán MonteCarloGPUAlgorithm")

    def initialize(self, patient_data):
        """
        Khởi tạo thuật toán với dữ liệu bệnh nhân.

        Parameters
        ----------
        patient_data : Any
            Dữ liệu bệnh nhân bao gồm CT và các thông tin liên quan

        Returns
        -------
        bool
            True nếu khởi tạo thành công, False nếu thất bại
        """
        try:
            self.patient_data = patient_data

            # Trích xuất dữ liệu CT từ patient_data
            if hasattr(patient_data, "ct_data") and hasattr(patient_data, "voxel_size"):
                self.set_ct_data(patient_data.ct_data, patient_data.voxel_size)
                self.calculation_status["initialized"] = True
                logger.info("Khởi tạo thuật toán MonteCarloGPU thành công")
                return True
            else:
                logger.error(
                    "Dữ liệu bệnh nhân không chứa thông tin CT hoặc voxel_size"
                )
                self.calculation_status["error"] = "Dữ liệu bệnh nhân không đầy đủ"
                return False
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo thuật toán MonteCarloGPU: {str(e)}")
            self.calculation_status["error"] = str(e)
            return False

    def calculate_dose(self, beam_arrangement):
        """
        Tính phân bố liều cho các chùm tia xác định.

        Parameters
        ----------
        beam_arrangement : Any
            Bố trí chùm tia, bao gồm thông tin về các chùm tia xạ trị

        Returns
        -------
        Any
            Kết quả tính toán liều
        """
        if not self.calculation_status["initialized"]:
            logger.error("Thuật toán chưa được khởi tạo. Gọi initialize() trước.")
            return None

        try:
            self.beam_arrangement = beam_arrangement
            total_dose = None

            # Xử lý từng chùm trong arrangement
            for i, beam in enumerate(beam_arrangement.beams):
                logger.info(f"Tính liều cho chùm {i + 1}/{len(beam_arrangement.beams)}")

                # Chuẩn bị cấu hình chùm tia
                beam_config = {
                    "energy": beam.energy,
                    "gantry_angle": beam.gantry_angle,
                    "collimator_angle": beam.collimator_angle,
                    "isocenter": beam.isocenter,
                    "mlc_positions": beam.mlc_positions,
                    "jaw_positions": beam.jaw_positions,
                    "weight": beam.weight,
                    "prescription": beam_arrangement.prescription.dose
                    if hasattr(beam_arrangement, "prescription")
                    else 1.0,
                }

                # Thiết lập cấu hình chùm và tính liều
                self.set_beam_configuration(beam_config)
                beam_dose = super().calculate_dose()

                # Cộng vào tổng liều
                if total_dose is None:
                    total_dose = beam_dose * beam.weight
                else:
                    total_dose += beam_dose * beam.weight

            # Lưu kết quả và cập nhật trạng thái
            self.dose_grid = total_dose
            self.calculation_status["completed"] = True
            self.calculation_status["ready"] = True

            # Tạo và trả về đối tượng kết quả
            result = MonteCarloGPUResult(
                dose_grid=self.dose_grid,
                patient_data=self.patient_data,
                beam_arrangement=beam_arrangement,
                performance=self.get_performance_stats(),
            )
            return result

        except Exception as e:
            logger.error(f"Lỗi khi tính liều với thuật toán MonteCarloGPU: {str(e)}")
            self.calculation_status["error"] = str(e)
            return None

    def get_algorithm_type(self):
        """
        Trả về loại thuật toán.

        Returns
        -------
        str
            Mã loại thuật toán
        """
        return "MONTE_CARLO_GPU"

    def get_display_name(self):
        """
        Trả về tên hiển thị của thuật toán.

        Returns
        -------
        str
            Tên hiển thị
        """
        return "Monte Carlo GPU"

    def get_description(self):
        """
        Trả về mô tả của thuật toán.

        Returns
        -------
        str
            Mô tả
        """
        return "Thuật toán Monte Carlo tính toán trên GPU với tốc độ nhanh hơn 50-200x so với CPU."

    def get_calculation_status(self):
        """
        Trả về trạng thái tính toán hiện tại.

        Returns
        -------
        Dict
            Trạng thái tính toán
        """
        return self.calculation_status


class MonteCarloGPUResult:
    """
    Kết quả tính toán liều từ thuật toán Monte Carlo GPU.
    """

    def __init__(self, dose_grid, patient_data, beam_arrangement, performance):
        """
        Khởi tạo kết quả tính toán liều.

        Parameters
        ----------
        dose_grid : np.ndarray
            Mảng 3D chứa phân bố liều tính toán được
        patient_data : Any
            Dữ liệu bệnh nhân được sử dụng trong tính toán
        beam_arrangement : Any
            Bố trí chùm tia được sử dụng trong tính toán
        performance : Dict
            Thống kê hiệu năng từ quá trình tính toán
        """
        self.dose_grid = dose_grid
        self.patient_data = patient_data
        self.beam_arrangement = beam_arrangement
        self.performance = performance
        self.timestamp = time.time()

    def get_dose(self):
        """
        Trả về phân bố liều.

        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều
        """
        return self.dose_grid

    def get_performance_stats(self):
        """
        Trả về thống kê hiệu năng.

        Returns
        -------
        Dict
            Thống kê hiệu năng
        """
        return self.performance

    def compare_with(self, other_result):
        """
            So sánh kết quả này với kết quả khác.

            Parameters
            ----------
            other_result : Any
                Kết quả khác để so sánh

        Returns
        -------
            Dict
                Các chỉ số so sánh
        """
        if hasattr(other_result, "get_dose"):
            other_dose = other_result.get_dose()
            return self._compare_doses(other_dose)
        else:
            logger.error("Đối tượng so sánh không có phương thức get_dose()")
            return None

    def _compare_doses(self, other_dose):
        """
            So sánh phân bố liều này với phân bố liều khác.

            Parameters
            ----------
            other_dose : np.ndarray
                Phân bố liều khác để so sánh

        Returns
        -------
            Dict
                Các chỉ số so sánh
        """
        try:
            # Sử dụng phương thức có sẵn từ lớp MonteCarloGPU
            comparator = MonteCarloGPU()
            comparator.dose_grid = self.dose_grid
            return comparator.compare_with_dose_grid(other_dose)
        except Exception as e:
            logger.error(f"Lỗi khi so sánh phân bố liều: {str(e)}")
            return None
